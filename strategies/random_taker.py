"""随机主动单：无信息、但确实提供了流动性需求的基准。

为什么要有它：评估"策略是否真的有能力"需要一个**下界**。
随机乱做在短期也可能赚钱（运气），但它的 markout 应当 ≈ 0
（成交后价格朝哪走与它无关），而扣掉价差后 PnL 应当稳定为负——
那半个价差就是它付给市场的"入场费"。

如果某个"聪明"策略的 markout 和随机单没有统计差异，
那么它就是随机单，不管它的内部逻辑看起来多复杂。
"""

from __future__ import annotations

from tw.strategy import Strategy


class RandomTaker(Strategy):
    """以 p_active 的概率随机方向、随机数量地吃单。"""

    KIND = "random_taker"

    def __init__(self, *args, p_active: float = 0.05, qty: float = 1.0, **kw) -> None:
        super().__init__(*args, **kw)
        self.p_active = float(p_active)
        self.qty = float(qty)

    def on_tick(self, ctx):  # noqa: ANN001
        if self.rng.random() >= self.p_active:
            return None
        q = float(self.qty * self.rng.exponential(1.0))
        if self.rng.random() < 0.5:
            return ctx.market_buy(q)
        return ctx.market_sell(q)
