"""TWAP 执行：把目标仓位拆成 N 份，每 tick 执行一份。

用途是量化"分批执行省下多少冲击"。本项目阶段3 实测过一个强结论：
同一笔占市场总持仓 2% 的仓位，**一次砸完的因果滑点 −170bp，
拆成 100 份只有 −12bp（缩小 14 倍）**。
TWAP 就是把这个结论变成一个可对比的策略对象：
把它和"一次性市价执行"放在同一场景里跑，差值就是分批的价值。

实现上刻意用**限价单挂在对手价上**而不是市价单：
市价单每 tick 都吃一次价差，600 个 tick 下来光是价差就是灾难；
用限价单能站在被动方，但会面临"没成交"的风险。
两个版本都留着（``use_market=True`` 可切换），
因为这正是执行策略的真实取舍：**冲击 vs 成交确定性**。
"""

from __future__ import annotations

from tw.strategy import Strategy


class TwapExecutor(Strategy):
    """在 horizon 个 tick 内把 target_qty 均匀执行完。"""

    KIND = "twap"

    def __init__(
        self,
        *args,
        side: str = "buy",
        target_qty: float = 120.0,
        horizon: int = 600,
        use_market: bool = False,
        max_slice_frac: float = 3.0,
        **kw,
    ) -> None:
        super().__init__(*args, **kw)
        assert side in ("buy", "sell")
        self.side = side
        self.target_qty = float(target_qty)
        self.horizon = int(horizon)
        self.use_market = bool(use_market)
        #: 单片允许超出均值的倍数。太小会一直追不上进度，
        #: 太大会退化成"一次性执行"。
        self.max_slice_frac = float(max_slice_frac)
        self.done = 0.0
        self._t0: int | None = None

    def on_start(self, ctx):  # noqa: ANN001
        self._t0 = ctx.tick

    def on_fill(self, fill):  # noqa: ANN001
        if fill.side == self.side:
            self.done += fill.quantity
            return
        # 被动成交在反方向上（例如买单被市价卖单打中）会抵消进度
        self.done -= fill.quantity

    def on_tick(self, ctx):  # noqa: ANN001
        if self._t0 is None or ctx.tick - self._t0 >= self.horizon:
            return None
        remaining = self.target_qty - self.done
        if remaining <= 1e-6:
            return None
        left = max(1, self.horizon - (ctx.tick - self._t0))
        want = min(remaining, remaining / left * self.max_slice_frac)
        if want <= 1e-6:
            return None
        if self.use_market:
            return ctx.market_buy(want) if self.side == "buy" else ctx.market_sell(want)
        # 挂在对手价上（买单贴 best_ask，卖单贴 best_bid）：优先成交，
        # 但仍是限价单，不会像市价单那样无上限地穿档。
        if self.side == "buy":
            px = ctx.best_ask or ctx.mid
            return ctx.buy(px, want) if px else None
        px = ctx.best_bid or ctx.mid
        return ctx.sell(px, want) if px else None

    def on_end(self, ctx):  # noqa: ANN001
        self.completion = self.done / self.target_qty if self.target_qty else float("nan")
