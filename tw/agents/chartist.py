"""C. 图表派 / 趋势跟随者（施工蓝图 §3.5-C）。

规则：
    mom = (P(t) - P(t-N)) / P(t-N)
    mom > 0 → 买入（追涨）；mom < 0 → 卖出（杀跌）
    力度正比于动量强度（或设阈值触发）

这类主体带来**趋势强化压力**，是 Lux-Marchesi 模型涌现肥尾的关键来源：
图表派赚钱 → 吸引更多图表派 → 正反馈 → 大涨大跌。

图表派与基本面派的关系是整个模型的核心张力
------------------------------------------
    图表派  : 正反馈（动量 → 更多同向交易 → 更大动量）
    基本面派: 负反馈（偏离 V → 反向交易 → 拉回）

    只有正反馈 → 价格指数爆炸
    只有负反馈 → 价格死水一潭，收益接近正态，没有肥尾
    **两者比例合适 → 真实的肥尾 + 波动率聚集**

阶段4 的参数扫描就是在找这个比例区间。这不是拟合，是在找"哪些机制组合
能在数量级上重现真实统计特征"。

一个容易踩的坑：动量用**中间价**算，不用最新成交价。
成交价含有 bid-ask bounce（成交在买价还是卖价随机），
用它算动量会把这个微观结构噪声当成趋势信号，
使图表派在完全无趋势的市场里也持续单向交易。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..types import MarketState, Order
from .base import Agent, clipped_strength


class Chartist(Agent):
    KIND = "chartist"

    def __init__(
        self,
        agent_id: str,
        cash: float,
        inventory: float,
        rng: np.random.Generator,
        *,
        lookback: int = 50,
        deadband: float = 0.002,
        ref_momentum: float = 0.02,
        aggressiveness: float = 0.004,
        qty_mean: float = 1.5,
        p_active: float = 0.25,
        max_open_orders: int = 20,
        order_ttl: int = 500,
    ) -> None:
        super().__init__(agent_id, cash, inventory, rng, max_open_orders, order_ttl)
        self.lookback = int(lookback)
        self.deadband = float(deadband)
        self.ref_momentum = float(ref_momentum)
        self.aggressiveness = float(aggressiveness)
        self.qty_mean = float(qty_mean)
        self.p_active = float(p_active)

    def decide(self, state: MarketState) -> Order | Sequence[Order] | None:
        if self.rng.random() >= self.p_active:
            return None
        mid = state.mid
        if mid is None or mid <= 0:
            return None

        mom = state.momentum(self.lookback)
        if mom is None:
            return None

        strength = clipped_strength(mom, self.ref_momentum, self.deadband)
        if strength <= 0.0:
            return None

        side = "buy" if mom > 0 else "sell"
        edge = mid * self.aggressiveness * strength
        price = mid + edge if side == "buy" else mid - edge
        if price <= 0:
            return None

        qty = float(self.rng.exponential(self.qty_mean)) * strength
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
