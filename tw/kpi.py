"""KPI：目标、约束，以及**两种退化的防御**（A6）。

用户的原话
----------
> 是否要给LLM设计一个KPI，看看是否会有什么不同，
> 还有**不能因为达成KPI就不继续，亏损扩大就不交易，要平稳交易下去才行**。

这句话点出了 KPI 设计的**全部难点**。单指标一定会被刷（Goodhart）：

| 退化 | 表现 | 为什么会出现 |
|---|---|---|
| **达标即停**（打卡） | 一旦达标就大幅降低交易、或直接停手 | KPI 被当成"及格线"而不是"持续要求" |
| **亏了就装死**（躺平） | 亏损扩大后弃权率飙升，靠零暴露避免继续亏 | **不交易就不会亏** ⇒ 零暴露能刷分 |

⇒ **「平稳交易下去」的可操作化** = 一组**互相咬住**的约束：

```
收益    ≥ 目标                （要赚）
在场率  ≥ 下限                ← 防躺平（不交易 ⇒ 不达标）
回撤    ≤ 上限                ← 防赌博式达标
换手    ∈ [下限, 上限]        ← 下限防躺平、上限防乱枪打鸟
```

⚠️ 三条必须自己先说清楚的事
--------------------------
1. **写在 prompt 里的 KPI 可能"说一套做一套"**。
   本项目已有实证：`basis` 自报"照指标"，而实际动作与最简单的指标信号
   **只有 36% 一致**。⇒ **KPI 的效果一律从行为侧验证，不看它说什么。**
2. **KPI 数字本身可能引导模型去赌**（"要赚 5% 才达标" ⇒ 加杠杆）。
   这正是要加"回撤上限 + 在场率"的原因，但**它是否真的奏效必须实测**。
3. **阈值不能拍脑袋**：本模块提供 :func:`calibrate_from_baseline`，
   用**基线自己的行为分布**定阈值（基线是怎么"平稳"的，就照那个量级设）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


# ======================================================================
# 配置
# ======================================================================
@dataclass(slots=True)
class KPIConfig:
    """给 LLM 的目标与约束。**这是 A6 的实验变量。**

    ``None`` 表示不给 KPI（对照组）。
    """

    #: 窗口净收益目标（如 0.01 = 1%）
    target_return: float = 0.01
    #: 在场率下限：非空仓根数 / 总根数。**防躺平的那一条。**
    min_presence: float = 0.30
    #: 最大回撤上限。**防赌博的那一条。**
    max_drawdown: float = 0.05
    #: 换手（名义价值 / 权益）的允许区间
    turnover_lo: float = 1.0
    turnover_hi: float = 40.0

    def __post_init__(self) -> None:
        if self.min_presence < 0 or self.min_presence > 1:
            raise ValueError(f"min_presence 必须在 [0,1]，收到 {self.min_presence}")
        if self.max_drawdown <= 0:
            raise ValueError(f"max_drawdown 必须 > 0，收到 {self.max_drawdown}")
        if self.turnover_lo < 0 or self.turnover_hi < self.turnover_lo:
            raise ValueError(
                f"换手区间非法：[{self.turnover_lo}, {self.turnover_hi}]")

    def describe(self) -> dict[str, Any]:
        """可写进留痕的描述（**改 KPI 会改 `decision_id`**，所以要可复现）。"""
        return {
            "target_return": self.target_return,
            "min_presence": self.min_presence,
            "max_drawdown": self.max_drawdown,
            "turnover_lo": self.turnover_lo,
            "turnover_hi": self.turnover_hi,
        }


# ======================================================================
# 进度（截至当时的可观测量）
# ======================================================================
@dataclass(slots=True)
class KPIState:
    """**截至当前**的 KPI 进度。

    ⚠️⚠️ **它里面每一个量都必须是"当时真的能算出来"的**：
    起点权益、当前权益、已过多少根、已成交多少根、截至当时的最大回撤、
    截至当时的换手。**不能有任何未来的量**——否则就是答案泄漏。
    这与 ``visible_state`` 是同一条纪律。

    实现上它由 :func:`advance` 逐步累积（只看已经发生的事），
    所以"含未来信息"在结构上就不可能。
    """

    initial_equity: float = 0.0
    equity: float = 0.0
    bars_done: int = 0
    bars_total: int = 0
    #: 已过去那些根里非空仓的根数（含当前这根之前的）
    bars_in_position: int = 0
    #: 截至当时的权益峰值（用于回撤）
    peak_equity: float = 0.0
    #: 累计名义成交额 / 起始权益
    turnover_x: float = 0.0

    @property
    def return_so_far(self) -> float:
        if not self.initial_equity:
            return 0.0
        return self.equity / self.initial_equity - 1.0

    @property
    def presence_so_far(self) -> float:
        return (self.bars_in_position / self.bars_done) if self.bars_done else 0.0

    @property
    def drawdown_so_far(self) -> float:
        if not self.peak_equity:
            return 0.0
        return max(0.0, (self.peak_equity - self.equity) / self.peak_equity)

    @property
    def bars_left(self) -> int:
        return max(0, int(self.bars_total) - int(self.bars_done))

    def gaps(self, kpi: KPIConfig) -> dict[str, Any]:
        """距离各条约束还差多少。**用于让模型知道"现在落后不落后"。**"""
        return {
            "return_gap": kpi.target_return - self.return_so_far,
            "presence_gap": kpi.min_presence - self.presence_so_far,
            "drawdown_room": kpi.max_drawdown - self.drawdown_so_far,
            "turnover_room": kpi.turnover_hi - self.turnover_x,
            "behind_target": self.return_so_far < kpi.target_return,
            "too_flat": self.presence_so_far < kpi.min_presence,
            "near_dd_limit": self.drawdown_so_far > kpi.max_drawdown * 0.8,
        }

    def to_dict(self, kpi: KPIConfig | None = None) -> dict[str, Any]:
        d: dict[str, Any] = {
            "bars_done": int(self.bars_done),
            "bars_total": int(self.bars_total),
            "bars_left": self.bars_left,
            "initial_equity": round(self.initial_equity, 6),
            "equity": round(self.equity, 6),
            "return_so_far": round(self.return_so_far, 8),
            "presence_so_far": round(self.presence_so_far, 6),
            "drawdown_so_far": round(self.drawdown_so_far, 6),
            "turnover_x": round(self.turnover_x, 6),
        }
        if kpi is not None:
            d["gaps"] = self.gaps(kpi)
        return d


def advance(st: KPIState, *, equity: float, qty: float,
            turnover_delta: float = 0.0) -> KPIState:
    """把状态推进一根。**只吃"已经发生"的量。**

    ⚠️ 调用顺序很关键：必须是"**先记账、再决策**"。
    若先决策再记账，模型看到的进度里会含**当前这根的结果**——
    而当前这根的结果在决策时还不知道。这里用"先推进、后决策"的调用约定，
    并在 `agent_run` 里写死顺序。
    """
    st.bars_done += 1
    st.equity = float(equity)
    st.peak_equity = max(st.peak_equity, st.equity)
    if abs(float(qty)) > 1e-12:
        st.bars_in_position += 1
    st.turnover_x += float(turnover_delta) / (st.initial_equity or 1.0)
    return st


# ======================================================================
# 阈值标定：用基线自己的行为分布
# ======================================================================
def calibrate_from_baseline(
    *,
    presence: list[float],
    turnover: list[float],
    drawdown: list[float],
    presence_q: float = 0.20,
    turnover_q: tuple[float, float] = (0.10, 0.90),
    drawdown_q: float = 0.90,
) -> dict[str, Any]:
    """用**基线**的行为分布定阈值。

    ⚠️ **不要拍脑袋**：KPI 阈值若设得比基线还宽松，等于没设；
    若比基线严苛得多，测到的就是「模型被逼到墙角」而不是「KPI 有用」。

    做法：取基线各指标的**分位数**——
    - 在场率下限 ← 基线的 **20% 分位**（比多数基线更松 ⇒ 是个"别躺平"的底线）
    - 换手区间  ← 基线的 10%~90% 分位
    - 回撤上限  ← 基线的 **90% 分位**（比多数基线更松）

    ⚠️⚠️ **调用方必须先剔除"按定义就不动"的基线**（本项目的 `noop`）。
    理由：`noop` 的在场率恒为 0、换手恒为 0，它在池子里会把
    **20% 分位直接拉到 0** ⇒ 「在场率下限 0%」= **这个约束是空的**，
    而日志看起来像"标定成功"。实测踩过一次（在场率下限标成 0.0%、
    换手下限标成 0.00）。本函数**不做剔除**（它看不到基线名），
    但会把该情形显式标出来：
    ``min_presence_vacuous=True`` ⇒ 调用方**必须**报警或直接失败。
    """
    def _q(xs: list[float], q: float) -> float:
        if not xs:
            return float("nan")
        s = sorted(xs)
        k = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
        return float(s[k])

    lo, hi = turnover_q
    floor = _q(presence, presence_q)
    t_lo = _q(turnover, lo)
    return {
        "min_presence": floor,
        "turnover_lo": t_lo,
        "turnover_hi": _q(turnover, hi),
        "max_drawdown": _q(drawdown, drawdown_q),
        #: ⭐ 在场率下限退化成 0 ⇒ 「别躺平」这条约束**完全没有约束力**。
        "min_presence_vacuous": (not (floor > 0.0)),
        "turnover_lo_vacuous": (not (t_lo > 0.0)),
        "note": ("由基线行为分布标定；**不是拍的**。"
                 "但分位数本身也依赖样本量，样本小时要谨慎。"),
        "n": len(presence),
    }


# ======================================================================
# 事后评估：KPI 有没有被刷（**只看行为**）
# ======================================================================
def kpi_verdict(kpi: KPIConfig, st: KPIState) -> dict[str, Any]:
    """窗口结束后，逐条对照 KPI。**这是"它有没有达标"的官方口径。**

    ⚠️ 与"模型说它达标了"是两件事——本项目已有实证：自报不可信。
    """
    checks = {
        "return": (st.return_so_far >= kpi.target_return,
                   st.return_so_far, kpi.target_return, "净收益 ≥ 目标"),
        "presence": (st.presence_so_far >= kpi.min_presence,
                     st.presence_so_far, kpi.min_presence,
                     "在场率 ≥ 下限（防躺平）"),
        "drawdown": (st.drawdown_so_far <= kpi.max_drawdown,
                     st.drawdown_so_far, kpi.max_drawdown,
                     "回撤 ≤ 上限（防赌博）"),
        "turnover_lo": (st.turnover_x >= kpi.turnover_lo,
                        st.turnover_x, kpi.turnover_lo, "换手 ≥ 下限"),
        "turnover_hi": (st.turnover_x <= kpi.turnover_hi,
                        st.turnover_x, kpi.turnover_hi, "换手 ≤ 上限"),
    }
    failed = [k for k, (ok, *_rest) in checks.items() if not ok]
    return {
        "passed": not failed,
        "failed": failed,
        "detail": {k: {"ok": bool(v[0]), "actual": v[1],
                       "threshold": v[2], "rule": v[3]}
                   for k, v in checks.items()},
    }


__all__ = [
    "KPIConfig",
    "KPIState",
    "advance",
    "calibrate_from_baseline",
    "kpi_verdict",
]
