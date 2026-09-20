"""配对交易者（二期阶段9）。

策略
----
交易的是**两个资产价格比率的 z-score**（不是单资产价格的 z-score）：

    ratio = P_A / P_B
    z     = (ratio − mean(ratio, 窗口)) / std(ratio, 窗口)
    z >  +阈值 → 比率偏高 → 卖 A、买 B
    z < −阈值 → 比率偏低 → 买 A、卖 B

⭐ 为什么这不是一个 ``Agent`` 子类
================================
``Agent.decide(state)`` 的契约是"返回**本市场**的订单"——它装不下两腿。
硬塞的办法有两种，都不好：
· 让每条腿各自独立判断 z-score → 两条腿会**用到不同的 mid 快照**
  （两个市场的 tick 内状态在不同时刻刷新），于是"同时在两边反向开仓"
  这个前提被破坏；
· 让 A 腿直接改 B 腿的持仓 → 绕过记账路径，一条腿的盈亏会凭空出现。

所以这里用**协调者**结构：每条腿在自己的市场里是一个真实的 ``Agent``
（有独立账户、走 ``Market.submit`` 这个唯一记账路径），
由 ``PairsTrader`` 在**同一个 tick 内**同时驱动两条腿。
代价是 B 腿的订单不参与 B 市场的优先级排序（见模块末的已知边界）。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from ..types import MarketState, Order
from .base import Agent


@dataclass(slots=True)
class PairsConfig:
    """配对交易的参数。"""

    #: z-score 的回看窗口（tick）
    zwindow: int = 200
    #: 开仓阈值（|z| 超过它才动手）
    zthresh: float = 2.0
    #: 平仓阈值（|z| 回到它以内就平）
    zexit: float = 0.5
    #: 每腿目标名义仓位（按自己市场的价格折算成手数）
    leg_notional: float = 2.0e5
    #: 每 tick 调仓的上限比例（0.25 = 每次最多补目标差的 1/4）
    #: 存在的意义：一步到位会在 z 抖动时来回满仓，换手率爆掉，
    #: 测出来的 PnL 里"执行成本"那部分会淹掉"信号"那部分。
    adjust_frac: float = 0.25
    #: 最小下单量（低于它就当作到位，避免无穷无尽的碎单）
    min_qty: float = 0.02

    def __post_init__(self) -> None:
        if self.zwindow < 10:
            raise ValueError("zwindow 必须 >= 10")
        if self.zthresh <= self.zexit:
            raise ValueError("zthresh 必须 > zexit，否则开平仓条件互相覆盖")
        if not (0.0 < self.adjust_frac <= 1.0):
            raise ValueError("adjust_frac 必须在 (0, 1]")


class PairsLeg(Agent):
    """配对交易的一条腿：一个**只持账户**的主体。

    ``decide`` 永远返回 None —— 下单由 ``PairsTrader`` 协调，
    因为两腿必须在同一 tick 内同时动作（见模块文档）。
    做成 ``Agent`` 而不是裸字典，是为了拿到
    「独立账户 + 预留制 + 走唯一记账路径」这三件事。
    """

    KIND = "pairs_leg"

    def decide(self, state: MarketState) -> Order | None:  # pragma: no cover
        return None


class PairsTrader:
    """跨资产配对交易的协调者。"""

    def __init__(
        self,
        assets: tuple[int, int],
        markets: list,
        legs: tuple[PairsLeg, PairsLeg],
        cfg: PairsConfig | None = None,
    ) -> None:
        if assets[0] == assets[1]:
            raise ValueError("配对的两个资产不能是同一个")
        self.i, self.j = int(assets[0]), int(assets[1])
        self.markets = markets
        self.legs = legs
        self.cfg = cfg or PairsConfig()
        self.ratio_hist: deque[float] = deque(maxlen=self.cfg.zwindow)
        #: 当前目标方向：+1 = 多 A / 空 B，−1 = 空 A / 多 B，0 = 空仓
        self.position_sign = 0
        self.n_signals = 0
        self.n_rebalances = 0
        #: 归因：z-score 序列（用于事后核对策略确实按信号在动）
        self.z_log: list[tuple[int, float]] = []

    # ------------------------------------------------------------------
    def current_ratio(self) -> float | None:
        pa = self.markets[self.i].current_mid()
        pb = self.markets[self.j].current_mid()
        if not pa or not pb or pa <= 0 or pb <= 0:
            return None
        return float(pa / pb)

    def zscore(self) -> float | None:
        if len(self.ratio_hist) < self.cfg.zwindow:
            return None
        arr = np.asarray(self.ratio_hist, dtype=float)
        sd = float(arr.std(ddof=1))
        if sd <= 0:
            return None
        return float((self.ratio_hist[-1] - arr.mean()) / sd)

    # ------------------------------------------------------------------
    def update_and_target(self) -> int:
        """更新比率历史并给出目标方向。返回 ``-1/0/+1``。"""
        r = self.current_ratio()
        if r is None:
            return self.position_sign
        self.ratio_hist.append(r)
        z = self.zscore()
        if z is None:
            return self.position_sign
        self.z_log.append((len(self.z_log), z))
        if self.position_sign == 0:
            if z > self.cfg.zthresh:
                self.position_sign = -1      # 比率偏高 → 空 A / 多 B
                self.n_signals += 1
            elif z < -self.cfg.zthresh:
                self.position_sign = +1      # 比率偏低 → 多 A / 空 B
                self.n_signals += 1
        elif abs(z) < self.cfg.zexit:
            self.position_sign = 0
        return self.position_sign

    def target_qty(self, side_asset: int, sign: int, qty_sign: int) -> float:
        """某条腿的目标手数。

        ``sign`` 是组合方向（+1 = 多 A 空 B），``qty_sign`` 是该腿在组合里的
        符号（A 腿 = +1，B 腿 = −1）。所以：
        · sign=+1（多 A 空 B）→ A 腿 +q，B 腿 −q
        · sign=−1（空 A 多 B）→ A 腿 −q，B 腿 +q
        两腿的**名义金额相同** → 组合对公共因子的净暴露 ≈ 0（这才是"市场中性"）。
        """
        if sign == 0:
            return 0.0
        px = self.markets[side_asset].current_mid()
        if not px or px <= 0:
            return 0.0
        q = self.cfg.leg_notional / px
        return float(sign * qty_sign * q)

    # ------------------------------------------------------------------
    def rebalance(self, sign: int) -> list[dict]:
        """把两条腿推向目标仓位。返回每腿的成交统计。

        ⚠️ **两腿必须在同一 tick 内同时提交**，否则"反向开仓"这个前提
        在时间上不成立——先动的腿会裸奔几个 tick，
        而裸奔期间的盈亏会被记成"配对策略的盈亏"。
        """
        out = []
        for idx, leg, qsign in ((self.i, self.legs[0], +1.0),
                                (self.j, self.legs[1], -1.0)):
            m = self.markets[idx]
            target = self.target_qty(idx, sign, qsign)
            cur = float(leg.inventory)
            delta = target - cur
            if abs(delta) < self.cfg.min_qty:
                out.append({"asset": idx, "submitted": 0, "filled": 0.0})
                continue
            step = delta * self.cfg.adjust_frac
            if abs(step) < self.cfg.min_qty:
                step = delta
            side = "buy" if step > 0 else "sell"
            o = leg.new_order(
                m._state, side, m.current_mid(), abs(step), order_type="market"
            )
            filled = 0.0
            if o is not None:
                trades = m.submit(leg, o)
                filled = float(sum(t.quantity for t in trades))
                self.n_rebalances += 1
            out.append({"asset": idx, "submitted": 1 if o else 0, "filled": filled})
        return out

    # ------------------------------------------------------------------
    def step(self) -> list[dict]:
        sign = self.update_and_target()
        return self.rebalance(sign)

    def describe(self) -> dict:
        return {
            "position_sign": self.position_sign,
            "n_signals": self.n_signals,
            "n_rebalances": self.n_rebalances,
            "leg_a_qty": float(self.legs[0].inventory),
            "leg_b_qty": float(self.legs[1].inventory),
        }


def attach_pairs_trader(
    multi, pair: tuple[int, int], *, cfg: PairsConfig | None = None,
    cash_per_leg: float = 5.0e6,
    short_headroom: float = 3.0,
) -> PairsTrader:
    """给多资产市场装上配对交易者。

    每人一条腿一个账户，初始现金相同（两条腿可比）。
    起点仓位为 0 —— 不从"已经有一腿"的状态开始，
    否则第一天的 PnL 里混进了初始建仓成本。

    ⚠️⚠️ **必须给两条腿设 ``short_limit``（这里踩过一个真 bug）。**

    ``Agent.short_limit`` 默认是 **0**，而 ``Market._clamp`` 的卖出分支是::

        cap = agent.available_inventory + agent.short_limit
        if cap <= MIN_ORDER_QTY:
            return None          # ← 订单被静默丢弃

    于是"空 B"那条腿**根本建不了仓**：从零库存卖出一律返回 ``None``，
    成交数 0、持仓 0、现金 0，**不报错、不留痕**。
    后果是配对交易在开仓时退化成"只做多 A 一条腿"——
    而它的文档写着"两腿名义金额相同 ⇒ 净公共因子暴露 ≈ 0"，
    这个前提直接不成立；报告里"配对交易跑通了"也就被高估了。

    为什么选 ``3 × leg_notional / 价格``：既要让目标仓位（``leg_notional``）
    一定够得着，又要给价格漂移留余量；同时它仍然是一个**有界的风险控制**
    （不是无穷额度），符合"空头一侧必须存在但必须有上限"的设计。
    """
    cfg = cfg or PairsConfig()
    legs = []
    for idx in pair:
        m = multi.markets[idx]
        leg = PairsLeg(f"pairs{idx}", cash_per_leg, 0.0,
                       np.random.default_rng([m.cfg.seed, 0xFA12, idx]))
        leg.bind_seed(m.cfg.seed)
        m.add_agent(leg)
        leg.set_initial_equity(m.current_mid())
        px = m.current_mid()
        leg.short_limit = float(
            short_headroom * cfg.leg_notional / px) if px > 0 else 0.0
        legs.append(leg)
    return PairsTrader(pair, multi.markets, (legs[0], legs[1]), cfg)
