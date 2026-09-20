"""核心数据结构（施工蓝图 §3.1）。

蓝图给出的 Order 只有 5 个字段。此处做三处**受控扩展**，每处都对应一个具体需求：

1. ``order_type`` —— 压力测试需要"强制平仓"（市价单）。若没有它，只能用
   "把价格打到很深的限价单"近似，后果是订单簿里会留下一张永远不会成交的假挂单，
   污染深度统计。
2. ``remaining`` —— 撮合必需的运行时状态。蓝图没写，但没有它就无法表达部分成交。
3. ``seq`` —— 时间优先的**唯一可靠依据**。同一 tick 内一个 agent 可能连下多单，
   仅靠 ``timestamp`` 无法区分先后。

市价单的 ``price`` 约定
----------------------
买单取 ``+inf``、卖单取 ``-inf``。这样撮合循环里的"是否穿越"判断可以统一写成
``taker.price >= maker_price``（买）/ ``<=``（卖），不需要为市价单写分支。
市价单永不挂入订单簿。
"""

from __future__ import annotations

import itertools
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

Side = Literal["buy", "sell"]
OrderType = Literal["limit", "market"]

# 数量比较容差。浮点数相减会留下 1e-17 量级的残渣，
# 不设容差会导致"永远差一点点没成交完"，进而死循环。
EPS = 1e-9

_seq_counter = itertools.count()


@dataclass(slots=True)
class Order:
    """一笔订单。蓝图 §3.1 的五个字段 + 三个运行时字段。"""

    agent_id: str
    side: Side
    price: float
    quantity: float
    timestamp: int
    order_id: str
    order_type: OrderType = "limit"
    remaining: float | None = None
    seq: int = field(default_factory=lambda: next(_seq_counter))

    def __post_init__(self) -> None:
        if self.side not in ("buy", "sell"):
            raise ValueError(f"side 必须是 'buy' 或 'sell'，收到 {self.side!r}")
        if self.order_type not in ("limit", "market"):
            raise ValueError(f"order_type 非法: {self.order_type!r}")
        if not (self.quantity > 0):
            raise ValueError(f"quantity 必须为正，收到 {self.quantity}")
        if self.order_type == "limit" and not math.isfinite(self.price):
            raise ValueError("限价单的 price 必须是有限值")
        if self.remaining is None:
            self.remaining = self.quantity
        elif self.remaining > self.quantity + EPS:
            raise ValueError("remaining 不能大于 quantity")

    # --- 便捷属性 -------------------------------------------------------
    @property
    def is_filled(self) -> bool:
        return self.remaining is not None and self.remaining <= EPS

    @property
    def filled_qty(self) -> float:
        return self.quantity - (self.remaining or 0.0)

    def __repr__(self) -> str:  # pragma: no cover - 仅调试用
        return (
            f"Order({self.order_id} {self.agent_id} {self.side} "
            f"{self.quantity:g}@{self.price:g} t={self.timestamp} "
            f"left={self.remaining:g})"
        )


@dataclass(slots=True)
class Trade:
    """一笔成交。

    蓝图要求记录 ``(tick, price, quantity, buy_agent_id, sell_agent_id)``。
    额外记录 ``aggressor_side`` 与两侧订单号，因为"扫单冲击"分析必须区分
    主动方/被动方——没有这个字段，事后无法重建订单流方向。
    """

    tick: int
    price: float
    quantity: float
    buy_agent_id: str
    sell_agent_id: str
    aggressor_side: Side
    buy_order_id: str
    sell_order_id: str
    seq: int = field(default_factory=lambda: next(_seq_counter))
    #: 成交**当时**的中间价，由 ``Market._settle`` 回填。
    #: 撮合引擎里拿不到市场状态，所以只能事后补；默认 nan 表示未回填。
    #: 用途见 ``tw/eval.py`` 的 markout —— 用 tick 收盘价代替它会让
    #: "成交后价格走势"失真一整条 tick（一 tick 内上百笔成交）。
    mid_at_fill: float = float("nan")

    @property
    def notional(self) -> float:
        return self.price * self.quantity

    @property
    def signed_qty(self) -> float:
        """带方向的主动成交量：主动买为 +，主动卖为 -。"""
        return self.quantity if self.aggressor_side == "buy" else -self.quantity


class PriceHistory:
    """定长环形缓冲，存放中间价序列。

    用 numpy 预分配数组而不是 list：主循环每个 tick 都要 append 一次，
    list 的无限增长会让 10000 tick × 多轮参数扫描的内存占用失控。
    """

    __slots__ = ("_buf", "_i", "_n", "_maxlen")

    def __init__(self, maxlen: int) -> None:
        if maxlen <= 0:
            raise ValueError("maxlen 必须为正")
        self._maxlen = maxlen
        self._buf = np.full(maxlen, np.nan, dtype=np.float64)
        self._i = 0  # 下一个写入位置
        self._n = 0  # 已写入总数（上限 maxlen）

    @property
    def maxlen(self) -> int:
        return self._maxlen

    def __len__(self) -> int:
        return self._n

    def append(self, value: float) -> None:
        self._buf[self._i] = value
        self._i = (self._i + 1) % self._maxlen
        if self._n < self._maxlen:
            self._n += 1

    def last(self) -> float | None:
        if self._n == 0:
            return None
        return float(self._buf[(self._i - 1) % self._maxlen])

    def lag(self, k: int) -> float | None:
        """k 期之前的值。k=0 即最后一期。历史不足时返回 None。"""
        if k < 0:
            raise ValueError("k 不能为负")
        if k >= self._n:
            return None
        return float(self._buf[(self._i - 1 - k) % self._maxlen])

    def to_array(self) -> np.ndarray:
        """按时间正序返回已写入的全部数据。"""
        if self._n < self._maxlen:
            return self._buf[: self._n].copy()
        return np.roll(self._buf, -self._i).copy()

    def std(self, window: int | None = None) -> float | None:
        """最近 window 期的标准差（收益率口径，用于做市商动态价差）。"""
        arr = self.to_array()
        if window is not None:
            arr = arr[-window:]
        if arr.size < 3:
            return None
        r = np.diff(np.log(arr))
        if r.size < 2:
            return None
        return float(np.std(r, ddof=1))


@dataclass(slots=True)
class MarketState:
    """每个 tick 交给 agent 的市场快照（蓝图 §3.4 的 ``market_state``）。

    ``history`` 是引用而非拷贝——200 个 agent × 每 tick 深拷贝一次历史
    会让主循环的开销从秒级涨到分钟级。
    """

    tick: int
    mid: float
    last_price: float
    best_bid: float | None
    best_ask: float | None
    spread: float | None
    fundamental: float
    history: PriceHistory
    # --- 二期扩展（默认值让一期代码与测试完全不受影响）-----------------
    #: 最近一次资金费率（每 tick 由 PerpetualMarket 刷新）。一期市场恒为 0。
    funding_rate: float = 0.0
    #: 近期订单流净失衡 ∈ [-1, 1]（主动买量 − 主动卖量，按窗口归一）。
    #: 阶段6 的 AdaptiveLiquidity 靠它决定"把挂单往外推多远"。
    flow_imbalance: float = 0.0
    #: 订单流窗口视图（引用，不拷贝）。阶段6 的自适应流动性需要**任意窗口**的
    #: 失衡量，而上面那个字段是固定窗口的预计算值。
    #: ⚠️ 它只是 ``log.flow`` 的一个只读视图：agent 若改它，等于篡改市场日志。
    flow: "FlowView | None" = None

    def momentum(self, n: int) -> float | None:
        """过去 n 期动量 ``(P(t) - P(t-n)) / P(t-n)``（蓝图 §3.5-C）。

        用中间价而不是最新成交价：成交价会被 bid-ask bounce 污染，
        用它算动量会把"买卖价差跳动"误读成趋势。
        """
        cur = self.history.last()
        past = self.history.lag(n)
        if cur is None or past is None or past <= 0:
            return None
        return (cur - past) / past


class FlowView:
    """订单流的**只读窗口视图**。

    为什么不让 agent 直接拿 ``market.log``：那等于把整个市场日志（预分配的
    大数组 + 全部成交流水）交到主体手里，主体一不小心就会改到它。
    这里只暴露"算一个窗口的失衡"这一件事。
    """

    __slots__ = ("_flow", "_tick")

    def __init__(self, flow: np.ndarray, tick: int) -> None:
        self._flow = flow
        self._tick = int(tick)

    def imbalance(self, window: int) -> float:
        """最近 ``window`` 个 tick 的主动买卖净失衡 ∈ [−1, 1]。

        与 ``Market._flow_imbalance`` **同一口径**（用绝对值和做分母，
        而不是笔数——用笔数会把"一笔巨量单"和"一笔碎单"等同看待）。
        两处口径必须一致，否则"标定时的拥挤度"和"主体看到的拥挤度"
        是两个不同的东西。
        """
        w = max(1, int(window))
        lo = max(0, self._tick - w)
        if self._tick <= lo:
            return 0.0
        seg = self._flow[lo:self._tick]
        den = float(np.nansum(np.abs(seg)))
        if den <= 1e-12:
            return 0.0
        return float(np.nansum(seg) / den)

    def signed(self, window: int) -> float:
        """最近 window 个 tick 的主动买卖**净量**（未归一）。"""
        w = max(1, int(window))
        lo = max(0, self._tick - w)
        return float(np.nansum(self._flow[lo:self._tick]))

    def count(self, window: int) -> int:
        """最近 window 个 tick 里有成交的 tick 数。"""
        w = max(1, int(window))
        lo = max(0, self._tick - w)
        seg = self._flow[lo:self._tick]
        return int(np.count_nonzero(np.nan_to_num(seg, nan=0.0) != 0.0))



@dataclass(slots=True)
class DepthLevel:
    price: float
    quantity: float
    n_orders: int


@dataclass(slots=True)
class BookSnapshot:
    """订单簿快照（供深度热力图使用）。"""

    tick: int
    mid: float
    bids: list[DepthLevel] = field(default_factory=list)
    asks: list[DepthLevel] = field(default_factory=list)
