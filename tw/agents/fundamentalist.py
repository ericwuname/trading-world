"""B. 基本面派（施工蓝图 §3.5-B）。

规则：
    设外生基本面价值 V(t)，当前中间价 P(t)。
    若 P(t) < V(t)：买入，力度正比于 (V(t) - P(t)) / V(t)
    若 P(t) > V(t)：卖出，力度同理

这类主体带来**均值回归压力**——它是 Lux-Marchesi 模型里唯一把价格拉回基本面的力量。
没有它，正反馈的图表派会把价格推成指数爆炸。

V(t) 的取法
----------
蓝图给了两个选项："缓慢随机游走"或"用真实历史价格趋势线作为代理"。
这里选**带弱回归的随机游走**：纯随机游走会无界漂移，跑久了 V 会跑到价格的一百倍，
基本面派就全都变成单向的死多头，模拟失去意义。加一个极弱的锚点拉力
（``fu_v_pull``，量级 1e-5）既能保留"基本面本身也在动"的随机性，
又不至于让模拟在 10000 tick 后跑飞。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..types import MarketState, Order
from .base import Agent, clipped_strength


class Fundamentalist(Agent):
    KIND = "fundamentalist"

    def __init__(
        self,
        agent_id: str,
        cash: float,
        inventory: float,
        rng: np.random.Generator,
        *,
        deadband: float = 0.002,
        ref_mispricing: float = 0.02,
        aggressiveness: float = 0.004,
        qty_mean: float = 1.5,
        p_active: float = 0.25,
        max_open_orders: int = 20,
        order_ttl: int = 500,
    ) -> None:
        super().__init__(agent_id, cash, inventory, rng, max_open_orders, order_ttl)
        self.deadband = float(deadband)
        self.ref_mispricing = float(ref_mispricing)
        self.aggressiveness = float(aggressiveness)
        self.qty_mean = float(qty_mean)
        self.p_active = float(p_active)

    def decide(self, state: MarketState) -> Order | Sequence[Order] | None:
        if self.rng.random() >= self.p_active:
            return None
        mid = state.mid
        v = state.fundamental
        if mid is None or mid <= 0 or v is None or v <= 0:
            return None

        mispricing = (v - mid) / v          # >0 低估（该买）, <0 高估（该卖）
        strength = clipped_strength(mispricing, self.ref_mispricing, self.deadband)
        if strength <= 0.0:
            return None

        side = "buy" if mispricing > 0 else "sell"
        # 力度越大越激进：报价穿越中间价的比例随 strength 线性上升
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


def step_fundamental(
    v_prev: float,
    rng: np.random.Generator,
    *,
    vol: float,
    pull: float,
    anchor: float,
) -> float:
    """基本面价值走一步：几何随机游走 + 极弱 OU 回归。"""
    noise = rng.normal(0.0, vol)
    drift = pull * np.log(anchor / v_prev) if v_prev > 0 else 0.0
    return float(v_prev * np.exp(noise + drift))
