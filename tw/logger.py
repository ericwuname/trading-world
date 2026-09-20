"""Logger：逐 tick 记录 + 成交明细（施工蓝图 §2 组件清单）。

为什么用预分配 numpy 数组而不是 list
------------------------------------
10000 tick × 每 tick 记 10 个标量，用 list 要走 Python 对象装箱；参数扫描时
要跑几十次，垃圾回收会变成主要开销。预分配数组是常数内存、无 GC 压力。

为什么不把推演过程写成日志文本
------------------------------
推演过程**不可复现地依赖执行顺序**（主体决策顺序每 tick 随机打乱），
文本日志没有任何诊断价值。真正有诊断价值的是三样东西：
    ① 逐 tick 的市场状态序列（用于统计特征）
    ② 成交明细（用于订单流/冲击分析）
    ③ 订单簿快照（用于深度演化可视化）
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import numpy as np

from .types import BookSnapshot, Trade


@dataclass
class SimLog:
    """一次模拟的全部产出。"""

    n_ticks: int
    config_label: str = ""
    config: dict = field(default_factory=dict)

    tick: np.ndarray | None = None
    mid: np.ndarray | None = None
    best_bid: np.ndarray | None = None
    best_ask: np.ndarray | None = None
    spread: np.ndarray | None = None
    fundamental: np.ndarray | None = None
    volume: np.ndarray | None = None
    n_trades: np.ndarray | None = None
    flow: np.ndarray | None = None            # 本 tick 主动买量 - 主动卖量
    bid_depth: np.ndarray | None = None       # 买侧总挂单量
    ask_depth: np.ndarray | None = None
    open_orders: np.ndarray | None = None
    n_levels_bid: np.ndarray | None = None
    n_levels_ask: np.ndarray | None = None

    trades: list[Trade] = field(default_factory=list)
    snapshots: list[BookSnapshot] = field(default_factory=list)
    agents: list[dict] = field(default_factory=list)

    #: 逐 tick 序列字段名（与上方数组字段一一对应）
    SERIES: ClassVar[tuple[str, ...]] = (
        "tick",
        "mid",
        "best_bid",
        "best_ask",
        "spread",
        "fundamental",
        "volume",
        "n_trades",
        "flow",
        "bid_depth",
        "ask_depth",
        "open_orders",
        "n_levels_bid",
        "n_levels_ask",
    )

    # --- 容量管理 -------------------------------------------------------
    def ensure_capacity(self, n: int) -> None:
        """保证序列数组至少能容纳 n 个 tick。

        为什么需要动态扩容：压力测试要在模拟跑完后**接着**再跑一段
        （"清算之后市场能否恢复"），若数组在构造时就死按配置长度分配，
        第二次 run 必然越界。让数组跟着实际 tick 走，接口才不容易被误用。
        """
        cur = self.mid.size if self.mid is not None else 0
        if n <= cur:
            return
        new = max(n, int(cur * 1.5) + 1, 64)
        for name in self.SERIES:
            arr = getattr(self, name)
            if arr is None:
                continue
            grown = np.full(new, np.nan, dtype=np.float64)
            grown[: arr.size] = arr
            setattr(self, name, grown)

    def trim(self, n: int) -> None:
        """把序列裁到实际跑过的长度，避免尾部留下一串 NaN 被统计函数读到。"""
        for name in self.SERIES:
            arr = getattr(self, name)
            if arr is not None and arr.size > n:
                setattr(self, name, arr[:n].copy())
        self.n_ticks = n

    # --- 便捷视图 -------------------------------------------------------
    @property
    def prices(self) -> np.ndarray:
        """中间价序列（字符串别名，读起来顺一点）。"""
        return self.mid

    def log_returns(self, start: int = 1) -> np.ndarray:
        p = self.mid
        if p is None:
            return np.array([])
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.diff(np.log(p))
        r = r[np.isfinite(r)]
        return r[start - 1 :] if start > 1 else r

    def total_volume(self) -> float:
        return float(sum(t.quantity for t in self.trades))

    def n_trades_total(self) -> int:
        return len(self.trades)

    def summary(self) -> dict:
        p = self.mid
        if p is None or p.size == 0:
            return {}
        finite = p[np.isfinite(p)]
        return {
            "n_ticks": int(self.n_ticks),
            "config": self.config_label,
            "price_first": float(finite[0]) if finite.size else None,
            "price_last": float(finite[-1]) if finite.size else None,
            "price_min": float(finite.min()) if finite.size else None,
            "price_max": float(finite.max()) if finite.size else None,
            "price_mean": float(finite.mean()) if finite.size else None,
            "log_drift_total": float(math.log(finite[-1] / finite[0])) if finite.size > 1 else None,
            "n_trades": len(self.trades),
            "total_volume": self.total_volume(),
            "mean_spread_bp": self._mean_spread_bp(),
            "mean_bid_depth": float(np.nanmean(self.bid_depth)) if self.bid_depth is not None else None,
            "mean_ask_depth": float(np.nanmean(self.ask_depth)) if self.ask_depth is not None else None,
        }

    def _mean_spread_bp(self) -> float | None:
        if self.spread is None or self.mid is None:
            return None
        s = self.spread
        m = self.mid
        ok = np.isfinite(s) & np.isfinite(m) & (m > 0)
        if not ok.any():
            return None
        return float(np.mean(s[ok] / m[ok]) * 1e4)

    # --- 落盘 -----------------------------------------------------------
    def write_series_csv(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cols = {
            "tick": self.tick,
            "mid": self.mid,
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "spread": self.spread,
            "fundamental": self.fundamental,
            "volume": self.volume,
            "n_trades": self.n_trades,
            "flow": self.flow,
            "bid_depth": self.bid_depth,
            "ask_depth": self.ask_depth,
            "open_orders": self.open_orders,
        }
        names = list(cols)
        arr = np.column_stack([np.asarray(cols[c], dtype=float) for c in names])
        np.savetxt(
            path,
            arr,
            delimiter=",",
            header=",".join(names),
            comments="",
            fmt="%.10g",
        )
        return path

    def write_trades_csv(self, path: str | Path, limit: int | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = self.trades if limit is None else self.trades[:limit]
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(
                [
                    "tick",
                    "price",
                    "quantity",
                    "buy_agent_id",
                    "sell_agent_id",
                    "aggressor_side",
                ]
            )
            for t in rows:
                w.writerow(
                    [t.tick, t.price, t.quantity, t.buy_agent_id, t.sell_agent_id, t.aggressor_side]
                )
        return path

    def write_summary_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": self.summary(),
            "agents": self.agents,
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return path
