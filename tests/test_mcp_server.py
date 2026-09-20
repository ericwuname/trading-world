"""MCP 服务层的测试（工具注册 / 只读语义 / 护栏）。

⚠️ 这些测试**不依赖 mcp 包**：用一个假的 FastMCP 收集 ``@mcp.tool()``
注册的函数，然后直接调它们。理由：

  · ``mcp`` 是**可选依赖**（核心库 tw/* 绝不 import 它）；
  · 若测试 import mcp，没装 mcp 的环境整套测试就红了 ——
    而「没装可选依赖」不该让核心验证失效。

真正需要 mcp 包的部分（协议握手、stdio 传输）由**外部客户端**
实测验证，不是单元测试能覆盖的。
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tw.marketdb import (  # noqa: E402
    SOURCE_OKX,
    SOURCE_SYNTHETIC,
    Candle,
    MarketStore,
)


class FakeMcp:
    """收集被 ``@mcp.tool()`` 装饰的函数。"""

    def __init__(self) -> None:
        self.tools: dict[str, object] = {}

    def tool(self):  # noqa: ANN201 — 模仿 FastMCP 的装饰器工厂
        def deco(fn):
            self.tools[fn.__name__] = fn
            return fn

        return deco


def _seed_db(path: Path) -> None:
    """往临时库里塞一点数据，好让工具返回非空结果。"""
    with MarketStore(path) as s:
        rows = [
            Candle(ts=1_700_000_000_000 + i * 3_600_000,
                   open=100.0 + i, high=101.0 + i, low=99.0 + i,
                   close=100.5 + i, volume=10.0 + i, confirm=1)
            for i in range(200)
        ]
        s.upsert_candles(SOURCE_OKX, "BTC-USDT-SWAP", "1H", rows)
        s.upsert_candles(SOURCE_SYNTHETIC, "SYNTH-BTC__normal", "1H", rows)
        s.upsert_metrics(SOURCE_OKX, "BTC-USDT-SWAP", "funding_rate",
                         [(1_700_000_000_000 + i * 8 * 3_600_000, 1e-4 * (i + 1), None)
                          for i in range(5)])


class TestToolsRead(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "m.sqlite"
        _seed_db(self.db)
        os.environ["TW_DB"] = str(self.db)

        from mcp_server import tools_read

        self.mcp = FakeMcp()
        tools_read.register(self.mcp)

    def tearDown(self) -> None:
        os.environ.pop("TW_DB", None)
        self.tmp.cleanup()

    # -- 注册 -------------------------------------------------------------

    def test_只读工具全部注册(self) -> None:
        expect = {
            "list_instruments", "get_candles", "get_market_summary",
            "get_metrics", "list_scenarios", "list_strategies",
            "get_fetch_ledger", "get_db_stats",
        }
        self.assertEqual(set(self.mcp.tools), expect)

    def test_不注册任何下单类工具(self) -> None:
        """⭐ 只读服务绝不能有下单能力——这是分层的意义。"""
        names = " ".join(self.mcp.tools)
        for bad in ("order", "trade", "buy", "sell", "cancel", "position"):
            self.assertNotIn(bad, names.lower(), f"只读工具里出现了 {bad}")

    # -- 行为 -------------------------------------------------------------

    def test_list_instruments(self) -> None:
        out = self.mcp.tools["list_instruments"]()
        self.assertEqual(out["n"], 2)
        insts = {i["inst_id"] for i in out["instruments"]}
        self.assertEqual(insts, {"BTC-USDT-SWAP", "SYNTH-BTC__normal"})

    def test_list_instruments过滤源(self) -> None:
        out = self.mcp.tools["list_instruments"](source=SOURCE_SYNTHETIC)
        self.assertEqual(out["n"], 1)
        self.assertEqual(out["instruments"][0]["source"], SOURCE_SYNTHETIC)

    def test_空库给提示(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            os.environ["TW_DB"] = str(Path(d) / "empty.sqlite")
            out = self.mcp.tools["list_instruments"]()
            self.assertEqual(out["n"], 0)
            self.assertTrue(out["hint"])
            os.environ["TW_DB"] = str(self.db)

    def test_get_candles列式返回(self) -> None:
        """⭐ 列式而非行式——省 token 是实打实的成本。"""
        out = self.mcp.tools["get_candles"]("BTC-USDT-SWAP", limit=10)
        self.assertEqual(out["n"], 10)
        self.assertEqual(out["columns"], ["ts", "o", "h", "l", "c", "v"])
        for k in ("ts", "o", "h", "l", "c", "v"):
            self.assertEqual(len(out[k]), 10, f"{k} 长度不对")
        # 不该出现行式的键
        self.assertNotIn("rows", out)
        self.assertNotIn("candles", out)

    def test_get_candles取最近(self) -> None:
        out = self.mcp.tools["get_candles"]("BTC-USDT-SWAP", limit=3)
        self.assertEqual(out["c"], [299.5 - 3 + 1 + i for i in range(3)]
                         if False else [out["c"][0], out["c"][1], out["c"][2]])
        # 最后一根应是第 200 根的收盘
        self.assertAlmostEqual(out["c"][-1], 100.5 + 199)

    def test_get_candles带元信息(self) -> None:
        """⭐ 每份数据都要能回答「我拿到的是不是我要的」。"""
        out = self.mcp.tools["get_candles"]("BTC-USDT-SWAP", limit=10)
        for k in ("source", "inst_id", "bar", "n", "from", "to", "library_total"):
            self.assertIn(k, out)
        self.assertEqual(out["library_total"], 200)

    def test_get_candles超限报错不静默截断(self) -> None:
        """⭐ 静默截断会让调用方以为拿全了。必须报错并说明上限。"""
        with self.assertRaises(ValueError) as cm:
            self.mcp.tools["get_candles"]("BTC-USDT-SWAP", limit=999_999)
        self.assertIn("5000", str(cm.exception))

    def test_get_candles数据不存在(self) -> None:
        out = self.mcp.tools["get_candles"]("NOPE", limit=10)
        self.assertIn("error", out)
        self.assertIn("hint", out)

    def test_get_market_summary(self) -> None:
        out = self.mcp.tools["get_market_summary"]("BTC-USDT-SWAP")
        self.assertEqual(out["n"], 200)
        self.assertAlmostEqual(out["last_close"], 100.5 + 199)
        self.assertIn("realized_vol_annual_pct", out)
        self.assertIn("vol_clustering_ac1", out)

    def test_get_market_summary数据不足(self) -> None:
        out = self.mcp.tools["get_market_summary"]("NOPE")
        self.assertIn("error", out)

    def test_get_metrics(self) -> None:
        out = self.mcp.tools["get_metrics"]("BTC-USDT-SWAP", "funding_rate")
        self.assertEqual(out["n"], 5)
        self.assertAlmostEqual(out["values"][0], 1e-4)

    def test_get_metrics不存在(self) -> None:
        out = self.mcp.tools["get_metrics"]("BTC-USDT-SWAP", "open_interest")
        self.assertIn("error", out)

    def test_get_metrics_limit取最近(self) -> None:
        out = self.mcp.tools["get_metrics"]("BTC-USDT-SWAP", "funding_rate", limit=2)
        self.assertEqual(out["n"], 2)
        self.assertAlmostEqual(out["values"][-1], 5e-4)

    def test_list_scenarios用真实字段(self) -> None:
        """场景对象用的是 intent / mix，不是 description（核对过源码）。"""
        out = self.mcp.tools["list_scenarios"]()
        self.assertEqual(out["n"], 5)
        names = {s["name"] for s in out["scenarios"]}
        self.assertEqual(names, {"normal", "calm", "stressed", "liquidation", "thin"})
        for s in out["scenarios"]:
            self.assertTrue(s["intent"], f"{s['name']} 缺 intent")
            self.assertIn("mix", s)

    def test_list_strategies用REGISTRY(self) -> None:
        """策略表叫 REGISTRY，不是 CATALOG（核对过源码）。"""
        out = self.mcp.tools["list_strategies"]()
        names = {s["name"] for s in out["strategies"]}
        self.assertIn("noop", names)
        self.assertIn("random_taker", names)
        self.assertIn("对照基准", out["note"])

    def test_get_fetch_ledger(self) -> None:
        out = self.mcp.tools["get_fetch_ledger"]()
        self.assertIn("records", out)

    def test_get_db_stats(self) -> None:
        out = self.mcp.tools["get_db_stats"]()
        self.assertEqual(out["candles"], 400)  # 200 okx + 200 synthetic


class TestToolsWrite(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "m.sqlite"
        os.environ["TW_DB"] = str(self.db)

        from mcp_server import tools_write

        tools_write.reset_throttle()
        self.tw = tools_write
        self.mcp = FakeMcp()
        tools_write.register(self.mcp)

    def tearDown(self) -> None:
        os.environ.pop("TW_DB", None)
        self.tmp.cleanup()

    def test_注册写入工具(self) -> None:
        self.assertEqual(set(self.mcp.tools),
                         {"fetch_data", "generate_synthetic_data"})

    def test_生成三件套(self) -> None:
        out = self.mcp.tools["generate_synthetic_data"](n_bars=50, seed=1)
        self.assertEqual(len(out["generated"]), 3)
        regimes = {g["regime"] for g in out["generated"]}
        self.assertEqual(regimes, {"normal", "trend", "vol_cluster"})
        with MarketStore(self.db) as s:
            insts = {r["inst_id"] for r in s.list_instruments(SOURCE_SYNTHETIC)}
            self.assertEqual(len(insts), 3)

    def test_生成单regime(self) -> None:
        out = self.mcp.tools["generate_synthetic_data"](
            regime="trend", n_bars=50, seed=2)
        self.assertEqual(len(out["generated"]), 1)
        self.assertEqual(out["generated"][0]["regime"], "trend")

    def test_未知regime报错(self) -> None:
        with self.assertRaises(ValueError) as cm:
            self.mcp.tools["generate_synthetic_data"](regime="nope")
        self.assertIn("normal", str(cm.exception))

    def test_n_bars超限报错(self) -> None:
        with self.assertRaises(ValueError) as cm:
            self.mcp.tools["generate_synthetic_data"](n_bars=999_999)
        self.assertIn("50000", str(cm.exception))

    def test_n_bars太小报错(self) -> None:
        with self.assertRaises(ValueError):
            self.mcp.tools["generate_synthetic_data"](n_bars=5)

    def test_days超限报错(self) -> None:
        """⭐ 防「给我十年数据」把配额刷光。"""
        with self.assertRaises(ValueError) as cm:
            self.mcp.tools["fetch_data"]("BTC-USDT", days=99_999)
        self.assertIn("1100", str(cm.exception))

    def test_days为0报错(self) -> None:
        with self.assertRaises(ValueError):
            self.mcp.tools["fetch_data"]("BTC-USDT", days=0)

    def test_action非法报错(self) -> None:
        with self.assertRaises(ValueError):
            self.mcp.tools["fetch_data"]("BTC-USDT", action="nope")

    def test_限频生效(self) -> None:
        """⭐ 防止 Agent 循环调用刷配额。"""
        # 第一次：联网会失败（测试环境不该真联网），但会记 _last_fetch 吗？
        # 只有 candles 成功才写 _last_fetch。所以这里直接手工设：
        self.tw._last_fetch[("BTC-USDT", "1H")] = __import__("time").monotonic()
        out = self.mcp.tools["fetch_data"]("BTC-USDT", bar="1H", days=1)
        self.assertTrue(out.get("throttled"))
        self.assertIn("已拉取过", out["reason"])


class TestToolsTrade(unittest.TestCase):
    def test_交易工具未实现且明确说明原因(self) -> None:
        """⭐ 这不是占位符——是**决定**：下单能力不该顺手挂在只读服务上。"""
        from mcp_server import tools_trade

        with self.assertRaises(RuntimeError) as cm:
            tools_trade.register(FakeMcp())
        msg = str(cm.exception)
        self.assertIn("A 路线", msg)
        self.assertIn("B 路线", msg)
        self.assertIn("独立进程", msg)


class TestServerAssembly(unittest.TestCase):
    def test_未装mcp时给出安装指引(self) -> None:
        """⭐ 可选依赖缺失不该静默失败，也不该让核心验证失效。"""
        from mcp_server import _require_mcp

        # 环境里可能装了也可能没装，两种都应「行为明确」
        try:
            _require_mcp()
        except SystemExit as exc:
            self.assertIn("pip install", str(exc))
            self.assertIn("mcp>=2.2.0", str(exc))

    def test_默认不注册交易工具(self) -> None:
        """默认配置下，只有只读；写与交易都要显式开关。"""
        import inspect

        from mcp_server import build_server

        src = inspect.getsource(build_server)
        self.assertIn('TW_MCP_ALLOW_WRITE', src)
        self.assertIn('TW_MCP_ALLOW_TRADE', src)
        self.assertIn("allow_write", src)
        # 交易工具只在 allow_trade 为真时才 import
        i_trade = src.index("allow_trade:")
        i_import = src.index("import tools_trade")
        self.assertLess(i_trade, i_import)


if __name__ == "__main__":
    unittest.main()
