"""Agent 基类（施工蓝图 §3.4）。

蓝图接口是 ``decide(market_state) -> Order | None``。此处做**一处受控扩展**：

    decide 可以返回 ``Order`` / ``list[Order]`` / ``None``

原因：做市商必须**同时**挂买单和卖单，一个返回值装不下。若不允许返回列表，
做市商就只能绕过 Market 直接操作订单簿，记账逻辑会分裂成两处——
那才是真正危险的设计。宁可扩接口，不要分裂记账。

资金/持仓约束的落实方式（蓝图 §3.5-A 的"简单预算约束"）
------------------------------------------------------
不是"下单时查一下余额"，而是**预留（reservation）**：

    可用现金   = 现金   - 已挂买单占用的资金
    可用持仓   = 持仓   - 已挂卖单占用的持仓

差别在哪：若只在下单时检查，一个 100 现金的 agent 可以连挂 10 张 100 的买单，
全部成交后现金变 -900。预留制下第 2 张单就挂不出去。
这是真实交易所的做法，也是"不能让 agent 凭空造钱"的唯一可靠实现。
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np

from ..types import EPS, MarketState, Order, Side, Trade

if TYPE_CHECKING:  # pragma: no cover
    from ..market import Market


@dataclass(slots=True)
class AgentStats:
    """单个主体的行为统计（事后归因用）。"""

    n_submitted: int = 0
    n_cancelled: int = 0
    n_trades: int = 0
    volume: float = 0.0
    notional: float = 0.0
    n_rejected: int = 0

    def reset(self) -> None:
        self.n_submitted = self.n_cancelled = self.n_trades = 0
        self.volume = self.notional = 0.0
        self.n_rejected = 0


class Agent(ABC):
    """所有主体的基类。"""

    KIND = "base"

    def __init__(
        self,
        agent_id: str,
        cash: float,
        inventory: float,
        rng: np.random.Generator,
        max_open_orders: int = 20,
        order_ttl: int = 500,
    ) -> None:
        self.agent_id = agent_id
        self.cash = float(cash)
        self.inventory = float(inventory)
        #: 初始现金。二期加：`Market.cash_conservation()` 用它做全系统守恒的基准。
        #: 不能在检查时反推——反推会把"外生现金变动"也算进基准，检查就失效了。
        self.initial_cash = float(cash)
        #: `spawn_rng` 的种子来源。默认 0，Market 建主体时会用真实 seed 覆盖。
        self._seed_key = 0
        #: 允许的卖空额度（最大空头持仓）。0 = 不能卖空（一期现货市场的行为）。
        #: 永续市场必须 > 0，否则空头一侧不存在、资金费率套利无法执行。
        self.short_limit = 0.0
        self.rng = rng
        self.max_open_orders = int(max_open_orders)
        self.order_ttl = int(order_ttl)

        self.open_orders: list[Order] = []
        self.reserved_cash = 0.0
        self.reserved_inventory = 0.0

        self.stats = AgentStats()
        self.market: "Market | None" = None
        self._oid_seq = 0
        # 初始权益基准，用于事后算收益
        self.initial_equity = float("nan")

    # --- 生命周期 -------------------------------------------------------
    def bind(self, market: "Market") -> None:
        self.market = market

    def set_initial_equity(self, price: float) -> None:
        self.initial_equity = self.cash + self.inventory * price

    # --- 账户视图 -------------------------------------------------------
    @property
    def available_cash(self) -> float:
        return self.cash - self.reserved_cash

    @property
    def available_inventory(self) -> float:
        return self.inventory - self.reserved_inventory

    def equity(self, mid: float | None) -> float:
        px = mid if mid is not None else 0.0
        return self.cash + self.inventory * px

    def equity_change(self, mid: float | None) -> float:
        if math.isnan(self.initial_equity) or mid is None:
            return 0.0
        return self.equity(mid) - self.initial_equity

    # --- 下单工具 -------------------------------------------------------
    def new_order(
        self,
        state: MarketState,
        side: Side,
        price: float,
        quantity: float,
        order_type: str = "limit",
    ) -> Order | None:
        """构造订单；数量退化到 0 或价格非法时返回 None（不产生垃圾单）。"""
        if quantity is None or not np.isfinite(quantity) or quantity <= EPS:
            return None
        if order_type == "market":
            px = math.inf if side == "buy" else -math.inf
        else:
            if not np.isfinite(price) or price <= 0:
                return None
            px = price
        self._oid_seq += 1
        return Order(
            agent_id=self.agent_id,
            side=side,
            price=px,
            quantity=float(quantity),
            timestamp=state.tick,
            order_id=f"{self.agent_id}#{self._oid_seq}",
            order_type=order_type,
        )

    def spawn_rng(self, tag: int) -> np.random.Generator:
        """为某个用途单独开一条**确定性的**随机流。

        为什么不能复用 ``self.rng``：主体的决策流消耗次数取决于它活跃多少次，
        而"活跃多少次"又受市场状态影响。如果学习 / 元订单这类新逻辑与决策流共用一条，
        两件事会互相纠缠——改了学习参数会顺带改变下单序列，实验结果无法归因。
        （一期"主体数量与背景行情解耦"是同一类问题的另一面。）

        用 ``(seed, agent_id, tag)`` 播种：与主体序号、与市场流都解耦，
        所以注入新主体、增删其他主体都不会扰动这条流。``tag`` 各用途独占，
        不能重复——重复就等于两条逻辑共用一条流。

        ⚠️ ``agent_id`` 是字符串，而 numpy 的 SeedSequence **只接受整数熵**
        （``ValueError: unrecognized seed string``）。所以要先做一次
        **跨进程稳定**的哈希：用 ``zlib.crc32`` 而不是内置 ``hash()``——
        后者带进程级随机盐，同一 ID 在不同进程里会得到不同种子，实验就不可复现。
        """
        return np.random.default_rng(
            [self._seed_key, _stable_id_key(self.agent_id), int(tag)]
        )

    def bind_seed(self, seed: int) -> None:
        """由 Market 在建主体时调用，给 ``spawn_rng`` 一个稳定的种子来源。"""
        self._seed_key = int(seed)

    # --- 下单便捷方法（二期加：让新主体不必自己拼价格）--------------------
    def buy_order(
        self, state: MarketState, quantity: float, price: float | None = None
    ) -> Order | None:
        """买入。``price=None`` 表示贴对手价（best_ask）挂限价单。"""
        px = price if price is not None else (state.best_ask or state.mid)
        if px is None or px <= 0:
            return None
        return self.new_order(state, "buy", float(px), quantity)

    def sell_order(
        self, state: MarketState, quantity: float, price: float | None = None
    ) -> Order | None:
        """卖出。``price=None`` 表示贴对手价（best_bid）挂限价单。"""
        px = price if price is not None else (state.best_bid or state.mid)
        if px is None or px <= 0:
            return None
        return self.new_order(state, "sell", float(px), quantity)

    # --- 决策（由子类实现）-----------------------------------------------
    @abstractmethod
    def decide(self, state: MarketState) -> Order | Sequence[Order] | None:
        """每个 tick 的决策。返回 None 表示本 tick 不动作。"""

    # --- 记账钩子（由 Market 调用，子类不应覆盖）------------------------
    def on_submitted(self, order: Order) -> None:
        self.stats.n_submitted += 1

    def on_rejected(self) -> None:
        self.stats.n_rejected += 1

    def on_trade(self, trade: Trade, is_buyer: bool) -> None:
        self.stats.n_trades += 1
        self.stats.volume += trade.quantity
        self.stats.notional += trade.notional

    def prune_open_orders(self) -> None:
        """把已不在订单簿上的订单（成交完 / 被撤）从本地列表剔除。"""
        if self.market is None:
            return
        book = self.market.book
        if len(self.open_orders) > 12:  # 大列表走集合查找
            alive = book.open_order_ids()
            self.open_orders = [o for o in self.open_orders if o.order_id in alive]
        else:
            self.open_orders = [
                o for o in self.open_orders if book.has_order(o.order_id)
            ]

    def recompute_reservations(self) -> None:
        """按当前挂单重算预留额度。

        市价单永不挂簿，因此不参与预留——否则 ``inf`` 价格会把预留额算成无穷大，
        该 agent 此后一张单都发不出去。
        """
        rc = 0.0
        ri = 0.0
        for o in self.open_orders:
            if o.order_type != "limit":
                continue
            rem = o.remaining or 0.0
            if rem <= EPS:
                continue
            if o.side == "buy":
                rc += rem * o.price
            else:
                ri += rem
        self.reserved_cash = rc
        self.reserved_inventory = ri

    # --- 挂单管理（撤单/超时）--------------------------------------------
    def cancel(self, order: Order) -> bool:
        if self.market is None:
            return False
        return self.market.cancel(self, order)

    def cancel_all(self) -> int:
        n = 0
        for o in list(self.open_orders):
            if self.cancel(o):
                n += 1
        return n

    def housekeeping(self, state: MarketState) -> None:
        """周期性维护：撤掉过期挂单、压掉超量的挂单。

        没有这一步，订单簿会被陈年挂单灌满——200 个 agent × 0.1 活跃率 × 10000 tick
        = 20 万张单，绝大多数永远不该成交却占着深度，深度统计会彻底失真。
        """
        if not self.open_orders:
            return
        ttl = self.order_ttl
        if ttl > 0:
            stale = [o for o in self.open_orders if state.tick - o.timestamp > ttl]
            for o in stale:
                self.cancel(o)
        excess = len(self.open_orders) - self.max_open_orders
        if excess > 0:
            # 撤最早挂的那批（它们离中间价通常最远，最不可能成交）
            for o in sorted(self.open_orders, key=lambda x: x.timestamp)[:excess]:
                self.cancel(o)

    # --- 归因 -----------------------------------------------------------
    def describe(self) -> dict[str, float]:
        return {
            "agent_id": self.agent_id,
            "kind": self.KIND,
            "cash": self.cash,
            "inventory": self.inventory,
            "n_submitted": self.stats.n_submitted,
            "n_trades": self.stats.n_trades,
            "volume": self.stats.volume,
        }

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"{self.__class__.__name__}({self.agent_id}, "
            f"cash={self.cash:,.0f}, inv={self.inventory:.2f})"
        )


_ID_KEY_CACHE: dict[str, int] = {}


def _stable_id_key(agent_id: str) -> int:
    """字符串 → 跨进程稳定的 32 位整数。

    **不能用内置 ``hash()``**：Python 字符串哈希带进程级随机盐
    （PYTHONHASHSEED），同一 ID 在不同进程里结果不同，
    于是"同一 seed 复现同一场实验"这条底线会被悄悄破坏——
    而且只在跨进程时暴露，同一进程内测不出来。
    """
    v = _ID_KEY_CACHE.get(agent_id)
    if v is None:
        import zlib

        v = int(zlib.crc32(agent_id.encode("utf-8")) & 0xFFFFFFFF)
        _ID_KEY_CACHE[agent_id] = v
    return v


def clipped_strength(value: float, ref: float, deadband: float) -> float:
    """把信号强度归一化到 [0, 1]：低于死区算 0，达到参考值算 1。

    死区（deadband）不是可有可无的调参：没有它，agent 会对
    1e-6 级别的噪声也持续下单，做市商永远拿不回价差，
    挂单簿被无穷无尽的微幅报价填满。
    """
    if ref <= 0:
        return 0.0
    a = abs(value)
    if a <= deadband:
        return 0.0
    return min(1.0, (a - deadband) / ref)
