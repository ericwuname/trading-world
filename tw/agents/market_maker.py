"""D. 做市商（施工蓝图 §3.5-D，阶段3 加入）。

规则：
    同时挂买单和卖单；价差(spread)根据近期波动率动态调整；
    库存过多时主动调整报价引导库存回归中性。

这类主体提供流动性、抑制极端波动——它是压力测试的对照组：
把做市商撤掉，同样的清算冲击会造成多大的价格坑、恢复要多久？

为什么每 tick 必须先撤掉自己的旧报价
------------------------------------
做市商的报价会被市场推着走（价格动了，旧报价就变成了"错价"，
成交就等于免费送钱给对手方）。若不撤销，会出现两个后果：
  1. 簿上堆积大量历史报价，"最优价"永远是被做市商遗弃的价格
  2. 自家新旧报价互相穿越，而自成交防护会跳过程门挂单 →
     簿面出现"自己的买价 >= 自己的卖价"，深度统计彻底失真
所以标准做法是 quote-and-replace：撤旧、挂新，同一个 tick 内完成。
"""

from __future__ import annotations

from typing import Sequence

from ..types import EPS, MarketState, Order
from .base import Agent


class MarketMaker(Agent):
    KIND = "market_maker"

    def __init__(
        self,
        agent_id: str,
        cash: float,
        inventory: float,
        rng,
        *,
        base_spread: float = 0.0006,
        vol_sensitivity: float = 4.0,
        vol_window: int = 100,
        inventory_target: float = 10.0,
        skew_strength: float = 0.0004,
        quote_qty: float = 1.0,
        max_open_orders: int = 4,
        order_ttl: int = 20,
    ) -> None:
        super().__init__(agent_id, cash, inventory, rng, max_open_orders, order_ttl)
        self.base_spread = float(base_spread)
        self.vol_sensitivity = float(vol_sensitivity)
        self.vol_window = int(vol_window)
        self.inventory_target = float(inventory_target)
        self.skew_strength = float(skew_strength)
        self.quote_qty = float(quote_qty)

    def half_spread_frac(self, state: MarketState) -> float:
        """半价差的**相对**值（比例），两侧共用。"""
        vol = state.history.std(self.vol_window)
        vol = 0.0 if vol is None else vol
        return float(self.base_spread + self.vol_sensitivity * vol)

    def half_spread(self, state: MarketState, mid: float) -> float:
        """半价差 = 基础价差 + 波动率加成。

        波动率放大价差是做市商活下来的唯一原因：波动大时被"逆选"
        （adverse selection）的概率高，价差必须相应变厚，
        否则做市商会被知情交易者持续抽血。
        """
        return mid * self.half_spread_frac(state)

    def quote_half_spread_frac(self, state: MarketState, side: str) -> float:
        """**某一侧**的半价差（相对值）。给自适应流动性一个挂载点。

        为什么必须分侧：二期阶段6 要的是"**单侧**流动性变薄"——
        订单流偏买时，被打的是卖侧，于是只有卖侧该往外推。
        用一个全市场共用的 half_spread 做不出这个效果：
        两侧一起推等于"价差整体变宽"，那是波动率上升的另一种表现，
        测出来的东西不一样（详见 ``tw/agents/adaptive_liquidity.py``）。
        """
        return self.half_spread_frac(state)

    def decide(self, state: MarketState) -> Order | Sequence[Order] | None:
        # quote-and-replace：先撤旧
        self.cancel_all()

        mid = state.mid
        if mid is None or mid <= 0:
            return None

        half_bid = mid * self.quote_half_spread_frac(state, "buy")
        half_ask = mid * self.quote_half_spread_frac(state, "sell")
        if min(half_bid, half_ask) <= 0 or max(half_bid, half_ask) >= mid:
            return None

        # 库存偏移：库存高于目标 → 整体报价下移（更容易卖出、更难买入）
        skew = -self.skew_strength * (self.inventory - self.inventory_target) * mid
        skew = max(-0.5 * mid, min(0.5 * mid, skew))

        bid_px = mid - half_bid + skew
        ask_px = mid + half_ask + skew
        if bid_px <= 0:
            bid_px = self.market.book.tick_size if self.market else 0.01
        if ask_px <= bid_px:
            ask_px = bid_px + (self.market.book.tick_size * 2 if self.market else 0.02)

        orders: list[Order] = []
        # 买单只挂可用现金能覆盖的量
        bid_qty = min(self.quote_qty, self.available_cash / bid_px)
        if bid_qty > EPS:
            o = self.new_order(state, "buy", bid_px, bid_qty)
            if o is not None:
                orders.append(o)

        ask_qty = min(self.quote_qty, self.available_inventory)
        if ask_qty > EPS:
            o = self.new_order(state, "sell", ask_px, ask_qty)
            if o is not None:
                orders.append(o)

        return orders or None
