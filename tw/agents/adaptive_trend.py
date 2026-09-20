"""自适应/学习型主体（二期阶段7）。

这一阶段第一次让仿真触及**反身性**：主体不再只有固定规则，而是会按自己的
历史结果调整策略参数。核心问题是——**当一个已知有效的策略被越来越多人采用时，
它自己的有效性会不会内生地衰减？**

两个主体
--------
``AdaptiveTrendFollower``  ε-greedy bandit，在候选 lookback 集合上学"哪个动量
                          窗口最好"。用来观察"学到的参数会不会收敛"。
``FixedMomentumTrader``   已经"学会了历史最优参数"、此后不再变化的追随者。
                          它是反身性实验的**被测群体**。

为什么用 bandit 而不是完整的 Q-learning
--------------------------------------
指导书的判断是对的，这里沿用：本问题是**单步决策、没有状态转移**
（选一个 lookback → 观察一个收益 → 更新一个值），
所以它就是一个多臂老虎机。上 Q-learning 会引入状态空间与折扣因子两个
没有对应物的概念，除了让结果更难解释之外没有收益。

⭐ 三个关键设计选择（都会直接改变测出来的东西）
=============================================

**① 评估的是「信号质量」（mid-to-mid），不是成交后的实际盈亏。**
   ``realized = direction · (mid[t+h] − mid[t]) / mid[t]``。
   为什么不看实际 PnL：实际 PnL 被**成交概率、库存约束、报价位置**
   一起污染——一个 lookback 可能因为"恰好报价被动成交得多"而看起来好，
   而那不是信号好。bandit 要学的是**信号**，所以评估必须只衡量信号。
   这一条会漏掉"执行成本随拥挤上升"这个渠道，属于本阶段的已知边界
   （见阶段小结的诚实边界）。

**② 方向必须来自**被评估的那个 lookback**，不能与"当前实际下单用的 lookback"脱钩。**
   如果评估用 lookback A 而下单用 lookback B，学到的东西没有对象。

**③ Q 值初值相同时必须**随机打破平局**。**
   所有 Q 初始化为 0，而 ``max(dict, key=...)`` 在平局时返回**第一个**键——
   于是第一个候选会被无条件选到，直到它的 Q 偏离 0 为止。
   表现是"学习过程有一个偏向第一个候选的起始偏差"，
   而它完全是实现细节造成的，与策略优劣无关。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..types import MarketState, Order
from .base import Agent

#: spawn_rng 的用途标签。各用途必须独占，不能与他人重复。
ENTROPY_ADAPTIVE_LEARN = 0xA1DA97


class FixedMomentumTrader(Agent):
    """固定参数的动量追随者 —— 「已经学会历史最优参数、此后不再变化」的群体。

    反身性实验的被测对象。规则刻意做得**完全刚性**（没有学习、没有随机探索），
    因为要测的是"同样的策略，人变多之后还灵不灵"，任何自适应都会
    把信号模糊掉。

    规则
    ----
    · ``momentum(lookback) > 阈值`` → 买
    · ``momentum(lookback) < −阈值`` → 卖
    · 否则不动作（死区）

    死区不是可选项：没有它，主体会对 1e-6 级的噪声也持续下单，
    库存会被噪声推着走，测出来的 PnL 里"信号"那部分被埋掉。
    """

    KIND = "fixed_momentum"

    def __init__(
        self,
        agent_id: str,
        cash: float,
        inventory: float,
        rng: np.random.Generator,
        *,
        lookback: int = 20,
        deadband: float = 0.002,
        ref_momentum: float = 0.02,
        aggressiveness: float = 0.004,
        qty_mean: float = 1.5,
        p_active: float = 0.25,
        max_open_orders: int = 20,
        order_ttl: int = 500,
    ) -> None:
        super().__init__(agent_id, cash, inventory, rng, max_open_orders, order_ttl)
        if lookback < 2:
            raise ValueError("lookback 必须 >= 2")
        self.lookback = int(lookback)
        self.deadband = float(deadband)
        self.ref_momentum = float(ref_momentum)
        self.aggressiveness = float(aggressiveness)
        self.qty_mean = float(qty_mean)
        self.p_active = float(p_active)

    def signal(self, state: MarketState) -> float:
        """当前动量。``None`` 按 0 处理（历史不足时视为没有信号）。"""
        m = state.momentum(self.lookback)
        return 0.0 if m is None else float(m)

    def decide(self, state: MarketState) -> Order | Sequence[Order] | None:
        if self.rng.random() >= self.p_active:
            return None
        mid = state.mid
        if mid is None or mid <= 0:
            return None
        mom = self.signal(state)
        if abs(mom) <= self.deadband:
            return None
        side = "buy" if mom > 0 else "sell"
        # 力度归一化 → 报价穿越中间价的比例（越强的信号越激进）
        strength = min(1.0, (abs(mom) - self.deadband) / self.ref_momentum)
        px = mid * (1.0 + self.aggressiveness * strength) if side == "buy" \
            else mid * (1.0 - self.aggressiveness * strength)
        qty = float(self.rng.exponential(self.qty_mean))
        if qty <= 0:
            return None
        if side == "buy":
            afford = self.available_cash / px
            if afford <= 0:
                return None
            qty = min(qty, afford)
        else:
            have = self.available_inventory
            if have <= 0:
                return None
            qty = min(qty, have)
        return self.new_order(state, side, px, qty)

    def describe(self) -> dict:
        d = super().describe()
        d.update({"lookback": self.lookback,
                  "deadband": self.deadband,
                  "aggressiveness": self.aggressiveness})
        return d


class AdaptiveTrendFollower(Agent):
    """ε-greedy bandit：在候选 lookback 集合上学"哪个动量窗口最好"。

    观测的是 ``(选中的 lookback, 当时的中间价, 方向)``，``eval_horizon``
    个 tick 之后按**中间价变化**结算这一次选择的价值（见模块文档 ①）。
    """

    KIND = "adaptive_trend"

    #: 待评估队列的长度上限。理论上每 tick 最多入队一条，
    #: 稳态长度 ≈ eval_horizon；设上限是为了在 eval_horizon 被调得很大时
    #: 不会让内存与归因开销失控（它同时也是一个"逻辑坏了"的哨兵：
    #: 长度顶到上限说明有评估没被结算，见 M25 的注记）。
    MAX_PENDING = 5_000

    def __init__(
        self,
        agent_id: str,
        cash: float,
        inventory: float,
        rng: np.random.Generator,
        *,
        candidate_lookbacks: Sequence[int] = (10, 20, 50, 100),
        epsilon: float = 0.1,
        learning_rate: float = 0.05,
        eval_horizon: int = 20,
        deadband: float = 0.002,
        ref_momentum: float = 0.02,
        aggressiveness: float = 0.004,
        qty_mean: float = 1.5,
        p_active: float = 0.25,
        max_open_orders: int = 20,
        order_ttl: int = 500,
    ) -> None:
        super().__init__(agent_id, cash, inventory, rng, max_open_orders, order_ttl)
        self.candidates = [int(x) for x in candidate_lookbacks]
        if not self.candidates:
            raise ValueError("candidate_lookbacks 不能为空")
        if any(x < 2 for x in self.candidates):
            raise ValueError("lookback 必须 >= 2")
        if not (0.0 <= epsilon <= 1.0):
            raise ValueError("epsilon 必须在 [0,1]")
        if not (0.0 < learning_rate <= 1.0):
            raise ValueError("learning_rate 必须在 (0,1]")
        if eval_horizon < 1:
            raise ValueError("eval_horizon 必须 >= 1")
        self.epsilon = float(epsilon)
        self.lr = float(learning_rate)
        self.eval_horizon = int(eval_horizon)
        self.deadband = float(deadband)
        self.ref_momentum = float(ref_momentum)
        self.aggressiveness = float(aggressiveness)
        self.qty_mean = float(qty_mean)
        self.p_active = float(p_active)
        #: Q 值。key = lookback。
        self.q_values: dict[int, float] = {lb: 0.0 for lb in self.candidates}
        #: 待评估：[tick, lookback, entry_mid, direction]
        self.pending_evals: list[list[float]] = []
        self._rng_learn: np.random.Generator | None = None
        #: 归因
        self.n_choices = 0
        self.n_updates = 0
        self.choice_counts: dict[int, int] = {lb: 0 for lb in self.candidates}

    # ------------------------------------------------------------------
    def learn_rng(self) -> np.random.Generator:
        """学习专用的独立随机流（懒建，理由同阶段6 的 ``meta_rng``）。"""
        if self._rng_learn is None:
            self._rng_learn = self.spawn_rng(ENTROPY_ADAPTIVE_LEARN)
        return self._rng_learn

    # ------------------------------------------------------------------
    def choose_lookback(self) -> int:
        """ε-greedy 选择。

        ⚠️ 探索时用 ``choice``，利用时**在并列最大里随机挑**——
        不能直接 ``max(q_values, key=q_values.get)``：所有 Q 初始为 0 时
        它永远返回**第一个**候选，于是第一个候选被无条件选中直到其 Q 偏离 0。
        这个起始偏差完全来自实现细节，与策略优劣无关，会把
        "学习收敛到哪个窗口"的结论污染掉。
        """
        rng = self.learn_rng()
        if float(rng.random()) < self.epsilon:
            return int(rng.choice(self.candidates))
        best = max(self.q_values.values())
        tied = [lb for lb in self.candidates if self.q_values[lb] >= best - 1e-15]
        return int(tied[0] if len(tied) == 1 else rng.choice(tied))

    def resolve_pending(self, state: MarketState) -> int:
        """结算到期的评估，返回结算条数。

        ⚠️ 每条评估**只能结算一次**。用"重建列表"的写法（把没到期的挑出来、
        其余丢弃）天然满足这一点；若改成"遍历 + 标记 + 从尾部删除"，
        很容易在删除时跳过一个元素，于是同一条评估被结算两次——
        某些 lookback 的 Q 更新次数会异常偏多（见 M25）。
        所以这里用不可变式的重建，并在归因里记 ``n_updates``。
        """
        n = 0
        still: list[list[float]] = []
        for ev in self.pending_evals:
            tick0, lb, entry, direction = ev
            if state.tick - tick0 >= self.eval_horizon:
                if entry > 0:
                    realized = direction * (state.mid - entry) / entry
                    self.q_values[int(lb)] += self.lr * (realized - self.q_values[int(lb)])
                    n += 1
            else:
                still.append(ev)
        self.pending_evals = still
        self.n_updates += n
        return n

    # ------------------------------------------------------------------
    def decide(self, state: MarketState) -> Order | Sequence[Order] | None:
        self.resolve_pending(state)
        mid = state.mid
        if mid is None or mid <= 0:
            return None
        if self.rng.random() >= self.p_active:
            return None

        lb = self.choose_lookback()
        self.n_choices += 1
        self.choice_counts[lb] = self.choice_counts.get(lb, 0) + 1
        mom = state.momentum(lb)
        if mom is None or abs(mom) <= self.deadband:
            return None
        side = "buy" if mom > 0 else "sell"
        if len(self.pending_evals) < self.MAX_PENDING:
            self.pending_evals.append(
                [float(state.tick), float(lb), float(mid), 1.0 if mom > 0 else -1.0]
            )
        strength = min(1.0, (abs(mom) - self.deadband) / self.ref_momentum)
        px = mid * (1.0 + self.aggressiveness * strength) if side == "buy" \
            else mid * (1.0 - self.aggressiveness * strength)
        qty = float(self.rng.exponential(self.qty_mean))
        if qty <= 0:
            return None
        if side == "buy":
            afford = self.available_cash / px
            if afford <= 0:
                return None
            qty = min(qty, afford)
        else:
            have = self.available_inventory
            if have <= 0:
                return None
            qty = min(qty, have)
        return self.new_order(state, side, px, qty)

    # ------------------------------------------------------------------
    def best_lookback(self) -> int:
        """当前 Q 值最高的 lookback（并列时返回候选表里靠前的）。"""
        return max(self.candidates, key=lambda lb: self.q_values[lb])

    def describe(self) -> dict:
        d = super().describe()
        d.update({
            "q_values": {str(k): v for k, v in self.q_values.items()},
            "best_lookback": self.best_lookback(),
            "n_choices": self.n_choices,
            "n_updates": self.n_updates,
            "n_pending": len(self.pending_evals),
            "choice_counts": {str(k): v for k, v in self.choice_counts.items()},
        })
        return d


def realized_pnl_bp(agent: Agent, mid: float) -> float:
    """主体相对**初始权益**的收益率（bp）。

    用 ``equity_change / initial_equity``：它是账本口径（现金 + 持仓盯市），
    不含任何自记数字，所以不会与市场账本脱节。
    初始权益为 0 或未设置时返回 nan。
    """
    if not np.isfinite(getattr(agent, "initial_equity", float("nan"))):
        return float("nan")
    eq0 = float(agent.initial_equity)
    if abs(eq0) < 1e-9:
        return float("nan")
    return float(agent.equity_change(mid) / eq0 * 1e4)
