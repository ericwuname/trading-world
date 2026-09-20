"""MCP 写入工具——需 ``TW_MCP_ALLOW_WRITE=1`` 才注册。

为什么必须显式开启
------------------
这两个工具**会改变本机状态**（往库里写数据、联网），
而且 ``fetch_data`` 会**出去连网**。默认关闭是为了：

1. 只读的 Agent 会话**不可能**意外写入；
2. 「这次会话允许做什么」这件事有个明确的开关，而不是靠自觉。

护栏（都不依赖调用方自觉）
--------------------------
- **限频**：同一 ``(inst_id, bar)`` 在 ``MIN_INTERVAL_SEC`` 内不重复拉
- **上限**：单次 ``days`` 有上限，``bars`` 有上限 —— 防止一个 Agent
  一句「给我十年数据」把配额刷光
- **不静默**：任何拒绝都返回明确原因与当前限制值
"""

from __future__ import annotations

import time
from typing import Any

#: 同一标的+周期的最小拉取间隔（秒）。防止 Agent 循环调用刷配额。
MIN_INTERVAL_SEC = 60.0

#: 单次拉取的天数上限（约 3 年，OKX 历史端点的实际回溯上限）
MAX_DAYS = 1100

#: 单次生成的 K 线上限
MAX_BARS = 50_000

#: 进程内限频记录：``(inst_id, bar) -> 上次成功拉取时刻``
_last_fetch: dict[tuple[str, str], float] = {}


def _db_path():
    import os
    from pathlib import Path

    from tw.marketdb import DEFAULT_DB

    return Path(os.environ.get("TW_DB", str(DEFAULT_DB)))


def reset_throttle() -> None:
    """清限频记录（测试用）。"""
    _last_fetch.clear()


def register(mcp: Any) -> None:  # noqa: ANN001
    from tw.marketdb import MarketStore
    from tw.okx_data import OkxError, OkxHttp, ensure_history, snapshot_metrics
    from tw.synthetic import (
        GenerateSpec,
        available_regimes,
        default_test_set,
        generate_into_store,
    )

    @mcp.tool()
    def fetch_data(
        inst_id: str,
        bar: str = "1H",
        days: int = 90,
        action: str = "candles",
    ) -> dict[str, Any]:
        """从 OKX 拉取公开行情并存入本地库（**联网**）。

        「按需拉取」：库里已覆盖请求范围就**不联网**，直接返回现有覆盖情况。
        所以这个工具可以安全地重复调用。

        ⚠️ 只拉**公开数据**（K 线、资金费率、未平仓量、标记价）。
        不涉及账户、不涉及下单。

        Args:
            inst_id: 标的，如 'BTC-USDT-SWAP'（永续）或 'BTC-USDT'（现货）。
            bar: 周期，'1m' '15m' '1H' '4H' '1D' 等。
            days: 往回拉多少天（上限 1100）。
            action: 'candles' 只拉 K 线；'snapshot' 额外抓一次资金费率/
                    未平仓量/标记价；'both' 两者都做。
        """
        if action not in ("candles", "snapshot", "both"):
            raise ValueError(f"action 必须是 candles/snapshot/both，收到 {action!r}")
        if days > MAX_DAYS:
            raise ValueError(
                f"days={days} 超过上限 {MAX_DAYS}。OKX 历史端点实际回溯有限，"
                f"请分段拉取。"
            )
        if days < 1:
            raise ValueError("days 至少为 1")

        # 限频：同一 (inst, bar) 短时间内不重复拉
        key = (inst_id, bar)
        now = time.monotonic()
        last = _last_fetch.get(key)
        if last is not None and (now - last) < MIN_INTERVAL_SEC:
            wait = MIN_INTERVAL_SEC - (now - last)
            return {
                "throttled": True,
                "reason": (f"{inst_id} {bar} 在 {MIN_INTERVAL_SEC:.0f} 秒内已拉取过，"
                           f"请等 {wait:.0f} 秒"),
                "hint": "库里的数据现在就能用（get_candles / get_market_summary）",
            }

        out: dict[str, Any] = {"inst_id": inst_id, "bar": bar, "days": days}
        http = OkxHttp()
        try:
            if action in ("candles", "both"):
                with MarketStore(_db_path()) as s:
                    cov = ensure_history(s, inst_id, bar, days=days, http=http)
                out["candles"] = {
                    "n_bars": cov.n_rows,
                    "from_ts": cov.first_ts,
                    "to_ts": cov.last_ts,
                }
                _last_fetch[key] = time.monotonic()
            if action in ("snapshot", "both"):
                with MarketStore(_db_path()) as s:
                    out["snapshot"] = snapshot_metrics(s, inst_id, http=http)
        except OkxError as exc:
            # 失败也如实返回——**不假装成功**（账本里已经记了）
            out["error"] = str(exc)
            out["hint"] = "失败已写入 fetches 账本，可用 get_fetch_ledger 查看"
            return out
        return out

    @mcp.tool()
    def generate_synthetic_data(
        inst_id: str = "SYNTH-BTC",
        bar: str = "1H",
        n_bars: int = 2000,
        seed: int = 0,
        regime: str = "",
        start_price: float = 60000.0,
    ) -> dict[str, Any]:
        """生成**已知性质**的合成行情，用于自我测试（**离线**）。

        三种 regime 各自回答一个问题：

        - ``normal``（纯随机游走）：**零可预测性**。策略在这里赚钱 = 运气或泄漏，
          **不可能是本事**。真实数据给不了这个保证。
        - ``trend``（带确定性趋势）：**有方向可抓**。若策略在这里也没反应，
          说明它根本没在看价格。
        - ``vol_cluster``（波动率聚集）：测风控在波动率突变时会不会失效。

        不填 regime 则一次生成**三件套**（同 seed、同起点、同价格，
        所以是「同一初始条件下的三种世界」）。

        Args:
            inst_id: 标的命名前缀（三件套会自动加 __normal / __trend 等后缀）。
            bar: 周期。
            n_bars: 生成多少根（上限 50000）。
            seed: 随机种子。**同 seed 完全可复现**。
            regime: 留空 = 生成三件套。
            start_price: 起始价。
        """
        if n_bars > MAX_BARS:
            raise ValueError(f"n_bars={n_bars} 超过上限 {MAX_BARS}")
        if n_bars < 10:
            raise ValueError("n_bars 至少为 10")
        if regime and regime not in available_regimes():
            raise ValueError(
                f"未知 regime {regime!r}，可选：{', '.join(available_regimes())}"
            )

        out: dict[str, Any] = {"bar": bar, "seed": seed, "n_bars": n_bars}
        with MarketStore(_db_path()) as s:
            if regime:
                specs = [GenerateSpec(
                    inst_id=inst_id, bar=bar, n_bars=n_bars, seed=seed,
                    regime=regime, start_price=start_price,
                )]
            else:
                specs = default_test_set(
                    inst_id=inst_id, bar=bar, n_bars=n_bars,
                    seed=seed, start_price=start_price,
                )
            out["generated"] = [
                generate_into_store(s, spec) for spec in specs
            ]
        out["note"] = ("同 seed + 同参数可完全重造（参数已写入 fetches 账本）。"
                       "三件套共享初始条件，差别只在生成机制。")
        return out


__all__ = [
    "MAX_BARS",
    "MAX_DAYS",
    "MIN_INTERVAL_SEC",
    "register",
    "reset_throttle",
]
