"""动量追单：价格往一个方向走了一段就跟进去。

放进来的目的是**提供一个"看起来合理但可能亏钱"的样本**。
动量策略在很多市场里是有效的，但本项目阶段2 实测到：
图表派（正反馈）单独存在时会把波动率推到真实市场的好几倍，
而混合池里基本面派会把它拉回来。所以"追动量"在这个市场里
到底赚不赚，是个需要实测的问题，不是靠直觉能回答的。

同时它还有一个诊断价值：**它是扫单事件的主要来源之一**。
本项目做扫单研究时发现扫单不是外生事件（扫单前 20 tick 动量显著非零），
用这个策略可以主动制造"可控的扫单"，配合匹配对照
就能把那个研究真正做干净。
"""

from __future__ import annotations

from tw.strategy import Strategy


class MomentumTaker(Strategy):
    """过去 lookback tick 的动量超过阈值就顺势吃单。"""

    KIND = "momentum"

    def __init__(
        self,
        *args,
        lookback: int = 50,
        threshold_bp: float = 20.0,
        qty: float = 1.0,
        max_inventory: float = 150.0,
        **kw,
    ) -> None:
        super().__init__(*args, **kw)
        self.lookback = int(lookback)
        self.threshold_bp = float(threshold_bp)
        self.qty = float(qty)
        self.max_inventory = float(max_inventory)

    def on_tick(self, ctx):  # noqa: ANN001
        mom = ctx.momentum(self.lookback)
        if mom is None:
            return None
        bps = mom * 1e4
        if abs(bps) < self.threshold_bp:
            return None
        if bps > 0:
            if ctx.inventory >= self.max_inventory:
                return None
            return ctx.market_buy(self.qty)
        if ctx.inventory <= -self.max_inventory:
            return None
        return ctx.market_sell(self.qty)
