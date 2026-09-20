"""A. 零智能交易者（施工蓝图 §3.5-A）—— Gode & Sunder (1993) 的基线主体。

规则（严格照蓝图）：
    1. 每个 tick 以概率 p_active 决定是否下单
    2. 随机选买/卖方向（各 50%）
    3. 报价在中间价 ± 一个随机幅度内均匀采样（0.1% ~ 2%）
    4. 数量从指数分布采样
    5. 不能卖空超过持仓；不能买超过现金承受能力

这类主体的意义不在"聪明"，而在**证明不需要聪明**：
Gode-Sunder 的实验结论是，连续双向拍卖机制**本身**就能让价格收敛到接近均衡。
所以阶段1的全部意义是建立一个对照基线——如果连零智能都能让价格不发散，
那么后来涌现出的肥尾、波动率聚集，功劳就记在主体的异质性上，而不是机制的bug。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..types import MarketState, Order
from .base import Agent


class ZeroIntelligence(Agent):
    KIND = "zero_intel"

    def __init__(
        self,
        agent_id: str,
        cash: float,
        inventory: float,
        rng: np.random.Generator,
        *,
        p_active: float = 0.10,
        offset_range: tuple[float, float] = (0.001, 0.02),
        qty_mean: float = 1.0,
        p_cancel: float = 0.05,
        p_buy: float = 0.5,
        max_open_orders: int = 20,
        order_ttl: int = 500,
    ) -> None:
        super().__init__(agent_id, cash, inventory, rng, max_open_orders, order_ttl)
        self.p_active = float(p_active)
        self.offset_lo, self.offset_hi = offset_range
        self.qty_mean = float(qty_mean)
        self.p_cancel = float(p_cancel)
        #: 下买单的概率。0.5 = 无方向偏好（一期默认）。
        #: 二期阶段5 用它制造"系统性做多偏好"，作为资金费率的驱动源——
        #: 真实市场的正费率占比 85.8%，背后就是这种长期净多头需求。
        self.p_buy = float(p_buy)

    def quote_offset_frac(self, state: MarketState, side: str) -> float:
        """报价幅度（**相对**中间价的比例，恒为正）。

        抽成方法是为了给二期阶段6 的 ``AdaptiveLiquidityMixin`` 一个挂载点：
        自适应流动性要按"主动订单流的方向"把**被打的那一侧**往外推。
        没有这个挂载点，那个 Mixin 就只能整体重写 ``decide``——
        而重写意味着把预算裁剪、随机撤单这些一期的细节再抄一遍，
        抄漏一处就会多出一个静默的行为差异。
        """
        return float(self.rng.uniform(self.offset_lo, self.offset_hi))

    def decide(self, state: MarketState) -> Order | Sequence[Order] | None:
        rng = self.rng

        # 随机撤单：真实零智能交易者不会永远挂着不动
        if self.p_cancel > 0 and self.open_orders and rng.random() < self.p_cancel:
            victim = self.open_orders[rng.integers(len(self.open_orders))]
            self.cancel(victim)

        if rng.random() >= self.p_active:
            return None

        mid = state.mid
        if mid is None or mid <= 0:
            return None

        side = "buy" if rng.random() < self.p_buy else "sell"
        # 报价：中间价 ± 随机幅度。买偏上、卖偏下 ——
        # 这样一部分订单会穿越价差立刻成交，另一部分挂在簿上提供深度，
        # 正是连续双向拍卖里"流动性既被消费也被提供"的最小模型。
        delta = mid * self.quote_offset_frac(state, side)
        price = mid + delta if side == "buy" else mid - delta
        if price <= 0:
            return None

        # 数量 ~ 指数分布（蓝图要求的偏态分布，不是均匀分布）
        qty = float(rng.exponential(self.qty_mean))
        if qty <= 0:
            return None

        # 预算/持仓约束：超出可用额度就裁剪，而不是直接丢单。
        # 裁剪而不是丢弃，是因为丢弃会让"约束紧的主体"完全不参与，
        # 相当于偷偷把主体池缩小了。
        if side == "buy":
            afford = self.available_cash / price
            if afford <= 0:
                return None
            qty = min(qty, afford)
        else:
            have = self.available_inventory
            if have <= 0:
                return None
            qty = min(qty, have)

        return self.new_order(state, side, price, qty)
