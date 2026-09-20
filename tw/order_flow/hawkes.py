"""Hawkes 自激点过程：订单到达的时间聚集性（二期阶段8）。

要测的是什么（以及**不是**什么）
--------------------------------
阶段6 处理的是**订单方向**的长记忆（ε 序列的自相关），
阶段8 处理的是**到达时刻**的聚集性（"大单来了之后很快又来一批"）。
两者正交，不能互相代替——这也是指导书要求分开做的理由。

机制
----
    λ(t) = μ + Σ_{t_i < t} α·exp(−β·(t − t_i))

· μ  基础强度（外生到达）
· α  每个事件带来的激发量
· β  激发衰减率
· 分支比 ``n = α/β``：一个事件平均直接引发多少个后续事件。
  ``n < 1`` 是**稳定条件**（否则强度发散）。
  在 n < 1 时，平稳平均强度是 ``λ̄ = μ/(1 − n)``。

⭐ 本模块相对指导书的两个关键改动
================================

**① 参数化改用（λ̄, n, β）而不是（μ, α, β）。**
   指导书直接给 (μ, α, β)。问题是：**（μ, α, β）与（到达率的均值）是绑在一起的**，
   扫 α 的同时会改变平均到达率。于是"聚集性变强"与"交易变多"两个效应
   完全分不开——而它们对市场统计特征的影响方向很可能相反。
   改用 (λ̄, n, β) 之后 ``λ̄`` 被**按构造固定**，扫 n 只动聚集性、不动均值。
   这不是为了方便，而是为了**可归因**。

**② 强度用增量递推实现，不用"遍历全部历史事件"。**
   指导书的 ``intensity_at`` 遍历 ``event_times``：那是 O(历史长度)，
   在 40000 tick × 每 tick 多个事件的规模下是 O(n²)——
   跑一场要几分钟到几十分钟（实测 4000 tick 就要 4.7 秒，
   外推到整场没法接受）。
   指数核有闭式递推：``A(t) = e^{−β}·(A(t−1) + events_{t−1})``，
   于是 λ(t) = μ + α·A(t) 是 **O(1)**。
   两条路径**必须给出同一个数**，``tests/test_hawkes.py`` 里有一条断言
   把递推结果与逐步求和钉在一起——否则"优化"就变成了"换了个东西"。

**③ 归一化强度 ρ(t) = λ(t)/λ̄ 才是市场要用的量。**
   市场要的是"这一 tick 有多少比例的主体能出手"，而不是绝对到达率。
   ρ 的均值**按构造等于 1**，所以关掉/打开 Hawkes 不改变平均活跃度——
   这是 E8.2「有没有额外贡献」这个判断能成立的前提。
   若 ρ 的均值漂了，"叠加 Hawkes 后 k 变了"就无法归因。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def geometric_factor(beta: float, dt: float) -> float:
    """离散递推下 ``E[A]/E[事件数]`` 的比值系数 ``f = 1/(e^{β·dt} − 1)``。

    ⭐ 为什么必须区分「连续」与「离散」——这是本模块最容易错的一处
    ---------------------------------------------------------
    连续时间的 Hawkes 用 ``α/β`` 表示分支比、平稳强度是 ``μ/(1−α/β)``。
    但本模型的时间是**离散的 tick**，递推是
    ``A ← e^{−β·dt}·(A + events)``，于是平稳关系变成

        E[A](1 − e^{−βdt}) = e^{−βdt}·E[events]  ⇒  E[A] = E[events]/(e^{βdt} − 1)

    也就是说**真正的分支比是 ``α·f`` 而不是 ``α/β``**。

    两者在 ``β·dt → 0`` 时一致，但本阶段的实际参数（β=0.15、dt=1）差了 7.9%
    （``e^0.15 − 1 = 0.1618`` vs ``0.15``）。用连续公式反解 μ，
    实际平均到达率会**系统性偏低**：

        设 λ̄ = 20、n = 0.7、β = 0.2 ⇒ 连续式给 μ = 6，实际 E[λ] = 16.3（低 18%）

    后果是 ρ 的时间均值 ≈ 0.81 而不是 1.0——
    而"打开 Hawkes 不改变平均活跃度"是 E8.2「有没有额外贡献」能成立的**前提**。
    这个偏差不会报错，只会让"聚集性"这个效应里混进"交易变少了"。

    实测抓到它的是 ``test_长期平均rho接近一``。
    """
    if beta <= 0 or dt <= 0:
        raise ValueError("beta 与 dt 必须为正")
    return 1.0 / (math.exp(beta * dt) - 1.0)



@dataclass(slots=True)
class HawkesConfig:
    """参数化：``(λ̄, n, β, dt)`` —— 见模块文档 ① 与 ``geometric_factor``。"""

    #: 基准事件到达率 λ̄（每 tick 的事件数）。它就是"没有自激时"的强度，
    #: 也是 E[λ] 被反解到的目标值。
    base_lambda: float = 40.0
    #: **离散**分支比 n = α·f ∈ [0, 1)：一个事件平均直接引发多少个后续事件。
    #: 0 = 泊松（无聚集），越大越聚集，→1 时接近临界（长记忆）。
    #: 必须 < 1，否则强度发散（见 HawkesProcess 的校验）。
    branching: float = 0.6
    #: 激发衰减率 β（每 tick）。越大衰减越快、聚集越短促。
    beta: float = 0.15
    #: 时间步（tick）。市场主循环用 1.0。
    dt: float = 1.0
    #: ρ 的截断上限（护栏）。极端分支比下 ρ 的尾部很长，
    #: 不截断会让某一 tick 的全部主体同时出手，把"聚集"变成"一次性脉冲"。
    clamp_rho: float = 4.0

    def __post_init__(self) -> None:
        if self.base_lambda <= 0:
            raise ValueError("base_lambda 必须为正")
        if not (0.0 <= self.branching < 1.0):
            raise ValueError("branching 必须在 [0,1)——>= 1 时强度发散")
        if self.beta <= 0:
            raise ValueError("beta 必须为正")
        if self.dt <= 0:
            raise ValueError("dt 必须为正")
        if self.clamp_rho <= 0:
            raise ValueError("clamp_rho 必须为正")
        # ⚠️ 提前检查「α < β」这条充分稳定条件（见 max_branching 的说明）。
        # 不在这里拦，就会在**跑起来之后**由 HawkesProcess 抛错——
        # 而那时可能已经跑了一半的参数扫描，前面的时间全白费。
        if self.branching >= self.max_branching:
            raise ValueError(
                f"branching={self.branching:g} 超过本参数下的稳定上限 "
                f"{self.max_branching:.4f}（β={self.beta:g}, dt={self.dt:g}）——"
                "此时 α ≥ β，强度过程会发散。请减小 branching 或增大 β。"
            )

    @property
    def geometric(self) -> float:
        """``f = 1/(e^{β·dt} − 1)``。"""
        return geometric_factor(self.beta, self.dt)

    @property
    def max_branching(self) -> float:
        """在「α < β」这条**充分稳定条件**下允许的最大离散分支比。

        ``α = n·(e^{β·dt} − 1) < β  ⇒  n < β/(e^{β·dt} − 1)``

        ⭐ 这不是一个多余的检查，而是**指导书的网格会踩到的一条真实边界**。
        直觉上"分支比只要 < 1 就稳定"，实际上还要同时满足 ``α < β``
        （``HawkesProcess`` 的校验），而后者更紧：

        ==========  ==============  ==========
        β           上限 n_max      说明
        ==========  ==============  ==========
        0.05        0.975          很松
        0.15        0.927          松
        0.50        **0.771**      紧——指导书网格里的 (n=0.9, β=0.5) 会被拒
        ==========  ==============  ==========

        实测踩到过：E8.1 的 (branching=0.9, beta=0.5) 直接抛
        ``ValueError: α=0.5838 ≥ β=0.5``，整个阶段8 中断。
        所以这个上限必须**在配置构造时**就被检查，让调用方能提前跳过，
        而不是跑到一半炸掉。
        """
        return self.beta / (math.exp(self.beta * self.dt) - 1.0)

    @property
    def alpha(self) -> float:
        """单个事件的激发量 α = n/f = n·(e^{β·dt} − 1)。

        ⚠️ 注意**不是** ``n·β``。用 ``n·β`` 会让离散平稳均值偏离设定值
        （见 ``geometric_factor``）。两者在 β·dt 小时数值接近，
        所以这个错误在上手时**看不出来**，要跑到几千 tick 才显现为 ρ̄ ≈ 0.81。
        """
        return float(self.branching / self.geometric)

    @property
    def mu(self) -> float:
        """外生强度 μ = λ̄(1 − n)。离散平稳式下 E[λ] = μ/(1−n) = λ̄。"""
        return float(self.base_lambda * (1.0 - self.branching))

    @property
    def continuous_branching(self) -> float:
        """连续口径的分支比 α/β（仅供与文献对照，不参与计算）。"""
        return float(self.alpha / self.beta)

    def to_process(self) -> "HawkesProcess":
        return HawkesProcess(mu=self.mu, alpha=self.alpha, beta=self.beta)



class HawkesProcess:
    """单变量 Hawkes 过程（指数核）。

    同时提供两条强度计算路径：
    · ``intensity_at(t)``：逐步求和，O(历史)，**与指导书公式逐字对应**，
      用来做单元测试的参照实现；
    · ``advance(events)``：O(1) 增量递推，市场主循环用。
    两者必须给出相同的数（有一条断言钉住）。
    """

    def __init__(self, mu: float, alpha: float, beta: float) -> None:
        if beta <= 0:
            raise ValueError("beta 必须为正")
        if mu <= 0:
            raise ValueError("mu 必须为正")
        if alpha < 0:
            raise ValueError("alpha 必须非负")
        # ⭐ 稳定条件。不拦它会让强度在几十个 tick 内指数爆炸：
        # 症状是"模拟中途性能急剧下降或数值溢出"，而根因写在参数里。
        if alpha >= beta:
            raise ValueError(
                f"必须满足 alpha < beta（当前 α={alpha:g} ≥ β={beta:g}），"
                "否则强度过程不稳定、积分发散"
            )
        self.mu = float(mu)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.event_times: list[float] = []
        #: 递推状态：A(t) = Σ e^{−β(t−t_i)}（对所有 t_i < t）
        self._decay_state = 0.0
        self._last_time = 0.0
        #: 累计激发量之和（归因用：与 λ̄ 比较可验证能量收支）
        self.total_excitation = 0.0

    # ------------------------------------------------------------------
    @property
    def branching_ratio(self) -> float:
        """n = α/β。小于 1 才稳定。"""
        return self.alpha / self.beta

    @property
    def stationary_mean(self) -> float:
        """平稳平均强度 λ̄ = μ/(1 − n)。"""
        return self.mu / (1.0 - self.branching_ratio)

    # --- 参照实现（逐步求和）------------------------------------------
    def intensity_at(self, t: float) -> float:
        """λ(t) = μ + Σ_{t_i < t} α·exp(−β(t − t_i))。

        这是指导书给的**定义式**，保留它有两个用途：
        ① 单元测试拿它当参照，验证递推实现没写错；
        ② 短序列上直接可读，便于人工核对。
        市场主循环**不要**用它（O(n²)，见模块文档 ②）。
        """
        exc = 0.0
        for ti in self.event_times:
            if ti < t:
                exc += self.alpha * np.exp(-self.beta * (t - ti))
        return self.mu + exc

    def should_fire(self, t: float, dt: float, rng: np.random.Generator) -> bool:
        """用强度做泊松稀疏采样：在 dt 内事件发生概率 ≈ λ(t)·dt。

        ⚠️ **dt 必须足够小**，使 λ(t)·dt ≪ 1，否则这个近似失效
        （概率会被截到 1，聚集性完全表达不出来）。
        函数内部把它变成一个**显式报错**而不是静默截断——
        静默截断的症状是"到了高聚集区反而没有聚集"，极难定位。
        """
        if dt <= 0:
            raise ValueError("dt 必须为正")
        p = self.intensity_at(t) * dt
        if p > 1.0:
            raise ValueError(
                f"λ(t)·dt = {p:.4g} > 1，泊松稀疏采样近似失效；"
                f"请减小 dt（当前 dt={dt:g}，λ(t)={self.mu + 0:g}+…）"
            )
        if float(rng.random()) < p:
            self.event_times.append(t)
            return True
        return False

    # --- 递推实现（市场用）--------------------------------------------
    def advance(self, dt: float, events: float) -> float:
        """推进一个步长，并把 ``events`` 个事件记在**区间起点**。

        ⚠️ 事件记在起点（即"上一步的末态"），不是终点。
        换算一下：若第 k 步的事件量是 ``e_{k}``，它发生在时刻 ``k``，
        调用 ``advance(dt, e_0)`` 之后返回的是 **λ 在时刻 1 的值**。
        把事件时间记成终点会让递推与定义式差一个 ``e^{−β·dt}``
        （在 β=0.5 时是 60% 的误差），而它**不会报错**——
        ``test_有事件时两者一致`` 就是靠这一点抓到的。

        递推：``A ← e^{−β·dt}·(A + events)``，然后 ``λ = μ + α·A``。
        推导：设 A(t) = Σ_{t_i<t} e^{−β(t−t_i)}，则
        A(t+dt) = e^{−β·dt}·A(t) + e^{−β·dt}·(事件量在 t 时刻)。
        第一项来自旧事件的衰减，第二项来自新事件在 dt 后的残余。
        """
        if dt <= 0:
            raise ValueError("dt 必须为正")
        self._decay_state = float(
            np.exp(-self.beta * dt) * (self._decay_state + float(events))
        )
        self._last_time += float(dt)
        lam = self.mu + self.alpha * self._decay_state
        self.total_excitation += max(0.0, lam - self.mu)
        return float(lam)

    def normalized_intensity(self, base_lambda: float,
                             clamp: float | None = None) -> float:
        """ρ = λ/λ̄，并施加上限护栏。均值**按构造**为 1（见模块文档 ③）。"""
        if base_lambda <= 0:
            raise ValueError("base_lambda 必须为正")
        rho = (self.mu + self.alpha * self._decay_state) / base_lambda
        if clamp is not None:
            rho = min(float(clamp), rho)
        return float(rho)

    def reset(self) -> None:
        self.event_times.clear()
        self._decay_state = 0.0
        self._last_time = 0.0
        self.total_excitation = 0.0


def simulate_event_rate(
    process: "HawkesProcess", n_steps: int, rng: np.random.Generator, *,
    base_lambda: float, clamp: float | None = None, warmup: int = 0,
) -> dict:
    """独立跑一条自激路径，返回实测事件率——用来验证"平均活跃度不变"。

    这是把 ``HawkesConfig`` 的关键恒等式（``E[λ] = μ/(1−n) = λ̄``）
    从**代数**变成**实测**的那一步。为什么必须实测：
    恒等式是"按构造成立"的（``mu`` 与 ``alpha`` 就是这么反解的），
    所以拿它自己验证自己是循环论证（本项目对这类"恒真判据"有过教训）。
    真正可能出错的是**递推实现**与**取事件的顺序**：

    · 事件必须记在区间**起点**（见 ``advance`` 的说明）；
    · 抽事件用的强度必须是**本步开始前**的 λ，不能是 ``advance`` 的返回值
      （后者已经把本步事件算进去了）。
    这两条任一做反，实测率都会系统偏离 λ̄——而它**不会报错**。

    返回 ``realized_rate``（实测平均事件率）、``rho_mean``（实测平均 ρ）、
    ``rel_err``（相对 λ̄ 的偏差）、以及用来看分布的 ``rates``/``events``。
    """
    if n_steps <= 0:
        raise ValueError("n_steps 必须为正")
    base_lambda = float(base_lambda)
    rates: list[float] = []
    events: list[float] = []
    rhos: list[float] = []
    for _ in range(int(n_steps)):
        # 顺序：先读本步开始时的 ρ → 抽事件 → 再推进。
        # ⚠️ 反过来（先 advance 再抽）会把本步事件的影响提前用上，
        #    实测率偏高约 一个 α 的量级，而且不报错。
        rho = process.normalized_intensity(base_lambda, clamp=clamp)
        lam = rho * base_lambda
        e = float(rng.poisson(lam))
        process.advance(1.0, e)
        rates.append(lam)
        rhos.append(rho)
        events.append(e)
    w = max(0, int(warmup))
    tail_ev = np.asarray(events[w:], dtype=float)
    tail_rho = np.asarray(rhos[w:], dtype=float)
    realized = float(tail_ev.mean()) if tail_ev.size else float("nan")
    return {
        "n_steps": int(n_steps),
        "warmup": w,
        "realized_rate": realized,
        "rho_mean": float(tail_rho.mean()) if tail_rho.size else float("nan"),
        "rel_err": ((realized - base_lambda) / base_lambda
                    if base_lambda else float("nan")),
        "rates": rates,
        "events": events,
    }


def rho_to_count(rho: float, total: int, *, clamp: float = 4.0) -> int:
    """把归一化强度映射成"本 tick 能出手的主体数"。

    ``E[ρ] = 1`` ⇒ 期望活跃主体数 = ``total`` × 1 = 原值，
    所以打开 Hawkes **不改变平均活跃度**，只改变它的分布。

    ⚠️ 结果是 ``int(round(...))`` 后的整数，且被夹在 ``[0, total]``。
    不夹上界会让 ``rho > 1/clamp 的倍数`` 时切出比主体总数还长的切片——
    Python 切片不会报错，只会静默返回全部主体，
    于是"超级聚集"这个本该出现的情形**恰好不会出现**。
    """
    if total <= 0:
        return 0
    k = int(round(min(float(clamp), max(0.0, float(rho))) * total))
    return max(0, min(total, k))
