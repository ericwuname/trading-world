"""MCP 只读工具——无副作用，**默认全部启用**。

设计原则
--------
1. **返回列式而非行式**：``[[ts,o,h,l,c,v], ...]`` 比
   ``[{"ts":..,"open":..}, ...]`` 省一大半 token（键名重复几千遍）。
   对 LLM 上下文来说这是实打实的成本。
2. **一定带元信息**：每份数据都附上 ``source`` / ``range`` / ``n``，
   让调用方能判断「我拿到的是不是我要的」——
   与项目那条「报告里的数字要能追溯」是同一条纪律。
3. **不做隐式截断**：要求超限时**报错并说明上限**，
   不静默返回一部分（静默截断会让调用方以为拿全了）。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from tw.marketdb import (
    DEFAULT_DB,
    SOURCE_OKX,
    SOURCE_SYNTHETIC,
    MarketStore,
)

#: 单次返回的行数上限。超过就报错——不静默截断。
MAX_ROWS = 5000


def _db_path() -> Path:
    import os

    return Path(os.environ.get("TW_DB", str(DEFAULT_DB)))


def _fmt(ms: int) -> str:
    if not ms:
        return "-"
    return time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(ms / 1000))


def register(mcp: Any) -> None:  # noqa: ANN001 — FastMCP 实例
    """把所有只读工具注册到 ``mcp``。"""

    @mcp.tool()
    def list_instruments(source: str = "") -> dict[str, Any]:
        """列出本地行情库里有哪些标的、各自覆盖到什么时间范围。

        用于回答「我能不能分析 BTC」。库里没有的东西要用 fetch_data 拉
        （需管理员开启写入权限），或用 generate_synthetic_data 自己造。

        Args:
            source: 可选过滤，'okx'（真实拉取的）或 'synthetic'（自生成的）。
        """
        with MarketStore(_db_path()) as s:
            rows = s.list_instruments(source or None)
            out = [
                {
                    "source": r["source"],
                    "inst_id": r["inst_id"],
                    "bar": r["bar"],
                    "n_bars": r["n"],
                    "from": _fmt(r["lo"]),
                    "to": _fmt(r["hi"]),
                }
                for r in rows
            ]
        return {
            "n": len(out),
            "instruments": out,
            "hint": ("库是空的。用 fetch_data 拉真实数据，"
                     "或 generate_synthetic_data 造测试数据。") if not out else "",
        }

    @mcp.tool()
    def get_candles(
        inst_id: str,
        bar: str = "1H",
        source: str = SOURCE_OKX,
        from_ts: int = 0,
        to_ts: int = 0,
        limit: int = 500,
    ) -> dict[str, Any]:
        """读 K 线（OHLCV）。

        返回**列式**数据：``ts`` / ``o`` / ``h`` / ``l`` / ``c`` / ``v``
        六个等长数组，比逐行字典省很多 token。

        ⚠️ 只返回**已走完**的 K 线。未走完的根会让分析读到未来数据。

        Args:
            inst_id: 标的，如 'BTC-USDT-SWAP'。
            bar: 周期，如 '1m' '15m' '1H' '4H' '1D'。
            source: 'okx' 或 'synthetic'。
            from_ts: 起始毫秒时间戳（0 = 不限）。
            to_ts: 结束毫秒时间戳（0 = 不限）。
            limit: 最多返回多少根（取**最近**的 N 根；上限 5000）。
        """
        if limit > MAX_ROWS:
            # 不静默截断：告诉调用方上限是多少，让它自己决定怎么切
            raise ValueError(
                f"limit={limit} 超过上限 {MAX_ROWS}。请分段读取"
                f"（用 from_ts/to_ts 切窗口），不要指望一次拿全。"
            )
        with MarketStore(_db_path()) as s:
            cov = s.coverage(source, inst_id, bar, confirmed_only=True)
            if cov.is_empty:
                return {
                    "error": f"库里没有 {source}:{inst_id}:{bar} 的数据",
                    "hint": "先调 list_instruments 看库里有什么",
                }
            ser = s.load_candles(
                source, inst_id, bar,
                from_ts=from_ts, to_ts=to_ts, limit=limit, confirmed_only=True,
            )
            return {
                "source": source,
                "inst_id": inst_id,
                "bar": bar,
                "n": len(ser),
                "from": _fmt(int(ser.timestamp[0])) if len(ser) else "-",
                "to": _fmt(int(ser.timestamp[-1])) if len(ser) else "-",
                "library_total": cov.n_rows,
                "columns": ["ts", "o", "h", "l", "c", "v"],
                "ts": ser.timestamp.tolist(),
                "o": ser.open.tolist(),
                "h": ser.high.tolist(),
                "l": ser.low.tolist(),
                "c": ser.close.tolist(),
                "v": ser.volume.tolist(),
            }

    @mcp.tool()
    def get_market_summary(
        inst_id: str, bar: str = "1H", source: str = SOURCE_OKX
    ) -> dict[str, Any]:
        """某标的的概览：最新价、区间涨跌、波动率、覆盖范围。

        比拉全部 K 线便宜得多，适合「先看看这个标的值不值得细看」。
        """
        import numpy as np

        with MarketStore(_db_path()) as s:
            ser = s.load_candles(source, inst_id, bar, confirmed_only=True)
            if len(ser) < 2:
                return {"error": f"数据不足：{source}:{inst_id}:{bar}"}
            r = np.diff(np.log(ser.close))
            r = r[np.isfinite(r)]
            per_year = {"1m": 525600, "5m": 105120, "15m": 35040,
                        "1H": 8760, "4H": 2190, "1D": 365}.get(bar, 8760)
            return {
                "source": source, "inst_id": inst_id, "bar": bar,
                "n": len(ser),
                "from": _fmt(int(ser.timestamp[0])),
                "to": _fmt(int(ser.timestamp[-1])),
                "last_close": float(ser.close[-1]),
                "first_close": float(ser.close[0]),
                "return_pct": float((ser.close[-1] / ser.close[0] - 1) * 100),
                "high": float(ser.high.max()),
                "low": float(ser.low.min()),
                "realized_vol_annual_pct": float(r.std() * np.sqrt(per_year) * 100),
                "vol_clustering_ac1": _ac1(r),
            }

    @mcp.tool()
    def get_metrics(
        inst_id: str, kind: str = "funding_rate",
        source: str = SOURCE_OKX, limit: int = 500,
    ) -> dict[str, Any]:
        """读快照类指标：资金费率 / 未平仓量 / 标记价 / 指数价。

        这些量**没有 OHLC**，所以不在 get_candles 里。

        Args:
            kind: 'funding_rate' | 'open_interest' | 'mark_price' | 'index_price'
        """
        with MarketStore(_db_path()) as s:
            ts, vals = s.load_metrics(source, inst_id, kind)
            if ts.size == 0:
                return {"error": f"没有 {source}:{inst_id}:{kind} 的记录",
                        "hint": "用 fetch_data(action='snapshot') 抓一次"}
            if ts.size > limit:
                ts, vals = ts[-limit:], vals[-limit:]
            return {
                "source": source, "inst_id": inst_id, "kind": kind,
                "n": int(ts.size),
                "ts": ts.tolist(),
                "values": [None if v != v else float(v) for v in vals],
            }

    @mcp.tool()
    def list_scenarios() -> dict[str, Any]:
        """列出可用的市场场景（跑实验时用）。

        五个场景覆盖：基准 / 冷清 / 高波动压力 / 清算冲击 / 稀薄流动性。
        """
        from tw.scenarios import SCENARIOS

        out = [
            {
                "name": name,
                "n_agents": getattr(sc, "n_agents", None),
                "n_ticks": getattr(sc, "n_ticks", None),
                "mix": dict(getattr(sc, "mix", {}) or {}),
                "intent": getattr(sc, "intent", ""),
            }
            for name, sc in SCENARIOS.items()
        ]
        return {"n": len(out), "scenarios": out}

    @mcp.tool()
    def list_strategies() -> dict[str, Any]:
        """列出可用的示例策略（含两个对照基准）。"""
        from strategies import REGISTRY

        out = [
            {"name": name, "class": cls.__name__, "default_params": dict(params)}
            for name, (cls, params) in REGISTRY.items()
        ]
        return {
            "n": len(out),
            "strategies": out,
            "note": ("noop（PnL 恒为 0）与 random_taker（稳定亏损）是**对照基准**——"
                     "任何策略的收益都必须先赢过它们才算有本事。"),
        }

    @mcp.tool()
    def get_fetch_ledger(limit: int = 20) -> dict[str, Any]:
        """看拉取账本：每份数据是什么时候、用什么参数拉下来的。

        用来回答「这份数据可靠吗」——比如能看到某次拉取是否失败过。
        """
        with MarketStore(_db_path()) as s:
            rows = s.fetch_history(limit)
        return {
            "n": len(rows),
            "records": [
                {
                    "id": r["id"],
                    "when": _fmt(r["finished_at"]),
                    "source": r["source"],
                    "inst_id": r["inst_id"],
                    "bar": r["bar"],
                    "n_rows": r["n_rows"],
                    "pages": r["pages"],
                    "ok": bool(r["ok"]),
                    "error": r["error"],
                    "endpoint": r["endpoint"],
                }
                for r in rows
            ],
        }

    @mcp.tool()
    def get_db_stats() -> dict[str, Any]:
        """行情库的整体规模（有多少 K 线 / 指标 / 账本记录）。"""
        with MarketStore(_db_path()) as s:
            st = s.stats()
        return {"db": str(_db_path()), **st}


def _ac1(x: Any) -> float:
    """一阶自相关。用来判断有没有波动率聚集性。"""
    import numpy as np

    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if a.size < 3:
        return float("nan")
    a = a - a.mean()
    denom = float(np.sum(a * a))
    if denom <= 0:
        return float("nan")
    return float(np.sum(a[1:] * a[:-1]) / denom)


__all__ = ["MAX_ROWS", "register"]
