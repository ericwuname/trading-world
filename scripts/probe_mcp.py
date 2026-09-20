"""MCP 协议层实测：起 stdio 子进程 → 官方客户端握手 → 真调工具。

这是**单元测试覆盖不到的一层**：单元测试用假 FastMCP 检查注册与行为，
但「协议握手能不能通、工具 schema 能不能被客户端解析」必须用真客户端验。
自测自己写的客户端测不出兼容性问题——所以这里用的是**官方 SDK 的客户端**。
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parent.parent


async def main() -> int:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_server"],
        cwd=str(ROOT),
    )
    ok = 0
    fail = 0

    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as sess:
            info = await sess.initialize()
            # ⚠️ v2 把 camelCase 字段改成了 snake_case
            # （``serverInfo`` → ``server_info``，``protocolVersion`` → ``protocol_version``）。
            # 用 getattr 双写法兼容，免得再被版本差异绊一次。
            si = getattr(info, "server_info", None) or getattr(info, "serverInfo", None)
            pv = (getattr(info, "protocol_version", None)
                  or getattr(info, "protocolVersion", "?"))
            print(f"已连接：{si.name} v{si.version}")
            print(f"协议版本：{pv}\n")

            tools = await sess.list_tools()
            names = [t.name for t in tools.tools]
            print(f"工具数 {len(names)}：{', '.join(names)}\n")

            # ① 只读工具清单里不能有交易类
            bad = [n for n in names
                   if any(k in n.lower() for k in
                          ("order", "trade", "buy", "sell", "cancel", "position"))]
            if bad:
                print(f"✗ 只读服务里出现了交易类工具：{bad}")
                fail += 1
            else:
                print("✓ 只读服务无交易类工具")
                ok += 1

            # ② 每个工具都要有能被客户端解析的 schema
            no_schema = [t.name for t in tools.tools
                         if not (getattr(t, "input_schema", None) or getattr(t, "inputSchema", None))]
            if no_schema:
                print(f"✗ 缺 inputSchema：{no_schema}")
                fail += 1
            else:
                print(f"✓ {len(names)} 个工具都有 inputSchema")
                ok += 1

            # ③ 真调：库里有什么
            res = await sess.call_tool("list_instruments", {})
            data = _payload(res)
            print(f"\n✓ list_instruments → {data.get('n')} 个标的")
            for i in data.get("instruments", [])[:6]:
                print(f"    {i['source']:<10} {i['inst_id']:<26} {i['bar']:<4} "
                      f"{i['n_bars']:>6} 根")
            ok += 1

            # ④ 真调：读 K 线（列式）
            res = await sess.call_tool(
                "get_candles", {"inst_id": "BTC-USDT-SWAP", "bar": "1H", "limit": 5}
            )
            data = _payload(res)
            if "error" in data:
                print(f"✗ get_candles → {data['error']}")
                fail += 1
            else:
                assert len(data["c"]) == len(data["ts"]) == 5
                print(f"\n✓ get_candles → {data['n']} 根，"
                      f"列式键 {data['columns']}")
                print(f"    最近收盘 {data['c'][-3:]}")
                ok += 1

            # ⑤ 真调：概览
            res = await sess.call_tool(
                "get_market_summary", {"inst_id": "BTC-USDT-SWAP", "bar": "1H"}
            )
            data = _payload(res)
            if "error" in data:
                print(f"✗ get_market_summary → {data['error']}")
                fail += 1
            else:
                print(f"\n✓ get_market_summary → 最新 {data['last_close']:.1f}，"
                      f"年化波动 {data['realized_vol_annual_pct']:.1f}%，"
                      f"聚集 ac1 {data['vol_clustering_ac1']:.4f}")
                ok += 1

            # ⑥ 真调：策略清单（验证用了真实 REGISTRY）
            res = await sess.call_tool("list_strategies", {})
            data = _payload(res)
            snames = {s["name"] for s in data["strategies"]}
            if {"noop", "random_taker"} <= snames:
                print(f"\n✓ list_strategies → {data['n']} 个，含两个对照基准")
                ok += 1
            else:
                print(f"✗ list_strategies 缺对照基准：{snames}")
                fail += 1

            # ⑦ 护栏：超限必须报错（不是静默截断）
            res = await sess.call_tool(
                "get_candles", {"inst_id": "BTC-USDT-SWAP", "limit": 999999}
            )
            if (getattr(res, "is_error", False) or getattr(res, "isError", False) or _payload(res).get("error")):
                print("✓ 超限被拒绝（未静默截断）")
                ok += 1
            else:
                print("✗ 超限没有被拒绝")
                fail += 1

    print(f"\n{'=' * 60}")
    print(f"通过 {ok} / 失败 {fail}")
    return 1 if fail else 0


def _payload(res) -> dict:
    """从 CallToolResult 里取出 dict（v2 有 structuredContent，回退到解析文本）。"""
    sc = (getattr(res, "structured_content", None)
          or getattr(res, "structuredContent", None))
    if isinstance(sc, dict):
        # 有的实现会包一层 {"result": {...}}
        if set(sc.keys()) == {"result"} and isinstance(sc["result"], dict):
            return sc["result"]
        return sc
    try:
        return json.loads(res.content[0].text)
    except Exception:  # noqa: BLE001
        return {}


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
