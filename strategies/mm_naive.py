"""做市：固定价差双边挂单（朴素）与带库存偏移的版本。

两个版本的差别只有一处：**是否让报价随库存偏移**。
放在一起是为了让"管住库存值多少钱"这个问题的答案可直接从表里读出来
（对比 `mm_naive` 与 `mm_skewed` 的 max_dd 与 inv_abs_mean）。

做市必须每 tick 撤旧挂新（quote-and-replace）：
挂单如果留在簿上不动，价格走开之后它们就成了"单向敞口"，
看上去还在做市，实际只在被动接刀。
"""

from __future__ import annotations

from tw.strategy import Strategy


class NaiveMaker(Strategy):
    """在中间价 ± half_spread_bp 挂固定数量双边单。"""

    KIND = "mm_naive"

    def __init__(
        self,
        *args,
        half_spread_bp: float = 12.0,
        qty: float = 2.0,
        max_inventory: float = 200.0,
        **kw,
    ) -> None:
        super().__init__(*args, **kw)
        self.half = float(half_spread_bp)
        self.qty = float(qty)
        self.max_inventory = float(max_inventory)

    def on_tick(self, ctx):  # noqa: ANN001
        mid = ctx.mid
        if mid is None:
            return None
        ctx.cancel_all()  # quote-and-replace
        # 库存到顶就只挂减仓方向——否则会一路加仓直到被预算裁剪
        bid = mid * (1 - self.half / 1e4)
        ask = mid * (1 + self.half / 1e4)
        if ctx.inventory >= self.max_inventory:
            return ctx.sell(ask, self.qty)
        if ctx.inventory <= -self.max_inventory:
            return ctx.buy(bid, self.qty)
        return ctx.quote(bid, ask, self.qty)


class SkewedMaker(NaiveMaker):
    """报价随库存偏移：库存高就往下压报价，逼自己减仓。

    ``skew = (inventory - target) / target_scale``，报价整体下移
    ``skew_bp_per_unit * skew`` 个 bp。这样买单更难成交、卖单更容易成交，
    库存会自然向目标回归——真实做市商的核心风控动作。
    """

    KIND = "mm_skewed"

    def __init__(
        self,
        *args,
        target_inv: float = 0.0,
        skew_bp_per_unit: float = 3.0,
        **kw,
    ) -> None:
        super().__init__(*args, **kw)
        self.target_inv = float(target_inv)
        self.skew_bp_per_unit = float(skew_bp_per_unit)

    def on_tick(self, ctx):  # noqa: ANN001
        mid = ctx.mid
        if mid is None:
            return None
        ctx.cancel_all()
        shift_bp = (ctx.inventory - self.target_inv) * self.skew_bp_per_unit
        bid = mid * (1 - (self.half + shift_bp) / 1e4)
        ask = mid * (1 + (self.half - shift_bp) / 1e4)
        return ctx.quote(bid, ask, self.qty)
