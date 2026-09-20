"""资金费率套利者（二期阶段5）。

对应真实世界的经典策略：**现货多头 + 永续空头**，赚取资金费率。
永续空头在费率 > 0 时收钱，现货多头对冲掉方向风险，于是策略的
盈亏几乎只取决于累计收到的资金费率。

单资产模型里的简化
------------------
本模型没有独立的现货市场，所以那条现货腿**无法真的下单**。
处理方式：给它记一个"虚拟现货腿"——按基本面路径标记的浮盈浮亏，
存在 ``synthetic_spot_pnl`` 里。

⚠️ **这笔 PnL 不计入市场账本，也不参与 ``Market.cash_conservation()``。**
它只是这个主体自己的记账，代表一个在模型里不存在的腿。
把它算进全局守恒会立刻报出"记账 bug"——而它其实是设计的一部分。
这个区分必须显式写在文档和测试里（指导书 §2.7 专门警告过这一点）。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..types import MarketState, Order
from .base import Agent

#: spawn_rng 的用途标签。各用途必须独占，不能与他人重复。
ENTROPY_HEDGER = 0x4ED9E


class FundingArbitrageur(Agent):
    """按资金费率方向建立对冲仓位。

    规则
    ----
    · 费率 > +阈值：做空永续（收费率）
    · 费率 < −阈值：做多永续（收费率）
    · 费率回到死区内：**逐步平掉**永续腿（真实策略不会永远持有）

    持仓上限 ``max_position``：没有它会一路加到被预算裁剪，
    而预算裁剪的表现是"订单突然变小"，看起来像行情变化而不是仓位失控。

    阈值必须跟着费率标定走（重要）
    ------------------------------
    ``entry_threshold_bp`` 的默认值 **不是** 拍的 5bp，而是针对标定后的费率水平定的。
    标定后单次费率的量级是「均值 1.06bp、标准差 1.5bp」（真实 BTC 永续就是这么小）。
    在 5bp 的阈值下，费率够得着门槛的时间不到 1%，
    这个主体整场模拟基本不动——测试里表现为"套利者没能建立空头"，
    而根因不是卖空额度失效，是**门槛开在价外**。

    真实世界的资金费率套利正是靠 0.5~1bp 级别的微小利差吃饭的
    （量大、加杠杆、且要扣掉手续费），所以 0.5bp 才是有现实对应的门槛。
    """

    KIND = "hedger"

    def __init__(
        self,
        agent_id: str,
        cash: float,
        inventory: float,
        rng: np.random.Generator,
        *,
        entry_threshold_bp: float = 0.5,
        position_size: float = 1.0,
        max_position: float = 40.0,
        exit_threshold_bp: float = 0.1,
        synthetic_spot: bool = True,
        max_open_orders: int = 20,
        order_ttl: int = 500,
    ) -> None:
        super().__init__(agent_id, cash, inventory, rng, max_open_orders, order_ttl)
        self.entry_threshold_bp = float(entry_threshold_bp)
        self.exit_threshold_bp = float(exit_threshold_bp)
        self.position_size = float(position_size)
        self.max_position = float(max_position)
        # ⚠️ 永续市场的空头必须存在。不设这个额度，这个主体只能买不能卖，
        # "做空永续收正费率"就无法执行——实测它会一路做多然后持续付钱。
        self.short_limit = float(max_position)
        self.synthetic_spot = bool(synthetic_spot)
        #: 虚拟现货腿的累计盈亏。**不进入市场账本**（见模块文档）。
        self.synthetic_spot_pnl = 0.0
        #: 累计收到的资金费率（由 on_funding 累加），用于归因。
        self.funding_collected = 0.0
        self.n_funding_events = 0
        self._last_mid: float | None = None
        self._rng_hedge = None   # 懒建，保证 spawn_rng 拿到已绑定的 seed

    def _rng(self) -> np.random.Generator:
        if self._rng_hedge is None:
            self._rng_hedge = self.spawn_rng(ENTROPY_HEDGER)
        return self._rng_hedge

    # ------------------------------------------------------------------
    def on_funding(self, rate: float, mark_price: float) -> float:
        """资金费率结算时由 Market 回调（见 ``PerpetualMarket._settle_funding``）。

        只用于**归因记录**，不参与记账——真正的现金变动由 Market 走
        ``apply_cash_delta`` 完成。主体自己再改一次 cash 就是双记账。
        """
        pay = -self.inventory * rate * mark_price
        self.funding_collected += pay
        self.n_funding_events += 1
        return pay

    def decide(self, state: MarketState) -> Order | Sequence[Order] | None:
        mid = state.mid
        if mid is None or mid <= 0:
            return None
        self._mark_synthetic_spot(mid)

        rate_bp = float(state.funding_rate) * 1e4
        inv = self.inventory

        # ⭐ 目标仓位制，不是"每 tick 累加"。
        #
        # 指导书写的是 `if rate>阈值 and inventory>=0: sell()`——那是**累加式**建仓：
        # 费率翻正就一路卖到上限，翻负再一路买回来。问题是**仓位滞后于信号**：
        # 费率是每 8 tick 结算一次的，而调仓要 15 个 tick（30 手 / 每 tick 2 手），
        # 于是结算时点上它常常还站在上一次信号的方向上——
        # 实测：10 个套利者累计"收到"的资金费率是 **−40 万**（本该是正的），
        # 因为它们系统性地在结算时站错方向。
        # 改成先算目标仓位、再以每 tick 上限向目标收敛：
        # 目标一旦达成就不再交易，仓位与信号严格同向。
        target = 0.0
        if rate_bp > self.entry_threshold_bp:
            target = -self.max_position
        elif rate_bp < -self.entry_threshold_bp:
            target = +self.max_position
        elif abs(rate_bp) > self.exit_threshold_bp:
            return None          # 死区内不动作，避免来回摩擦

        delta = target - inv
        if abs(delta) <= 1e-9:
            return None
        # ⚠️ 必须**撤旧挂新**，否则上一 tick 没成交完的报价会一直留在簿上：
        #   · 它们占着 `reserved_inventory`，把卖空额度吃掉，
        #     于是 `_clamp` 判定"可卖量为 0"，之后所有单直接被拒；
        #   · 实测后果：套利者连续卖 10 次、每次 5 手（共 50 手），
        #     实际只成交了 **0.05 手**——看起来在下单，其实什么也没做。
        # 做市商类主体一期就踩过同一件事（quote-and-replace）。
        if self.open_orders:
            self.cancel_all()
        qty = min(abs(delta), self.position_size)
        # ⚠️ 必须用**市价单**，不能用"贴买一/卖一"的限价单。
        # 贴在最优档的限价单只能吃掉那一档的深度（实测 5 手只成交 0.05 手），
        # 剩下的留在簿上——调仓目标永远达不到，而这一切不会报错。
        # 这个主体存在的意义就是"必须调到位"，所以付出冲击成本是它的本分。
        return self.new_order(
            state, "sell" if delta < 0 else "buy", 0.0, qty, order_type="market"
        )

    def _mark_synthetic_spot(self, mid: float) -> None:
        """按中间价盯市虚拟现货腿。

        现货腿 = −永续持仓（delta 中性），所以它的盈亏是 ``−inventory·Δmid``。
        这正是"对冲掉方向风险"的含义：永续腿上的价格盈亏被这条腿抵消，
        净下来只剩资金费率。
        """
        if not self.synthetic_spot:
            self._last_mid = mid
            return
        if self._last_mid is not None and self._last_mid > 0:
            self.synthetic_spot_pnl += -self.inventory * (mid - self._last_mid)
        self._last_mid = mid

    def describe(self) -> dict:
        d = super().describe()
        d.update({
            "funding_collected": self.funding_collected,
            "synthetic_spot_pnl": self.synthetic_spot_pnl,
            "n_funding_events": self.n_funding_events,
            # 明示：这一项不在市场账本里
            "synthetic_spot_in_market_ledger": False,
        })
        return d
