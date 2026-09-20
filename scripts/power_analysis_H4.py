"""H4 象限检验的功效分析：算清「值不值得投入」，而不是盲目加种子。

任务书：`交易世界 · 回填标准配置与H4做实任务书.md` §4。

要回答的问题
------------
H4 的**宏观形式**已经成立（非对称配置下吃单突发期深度 −22.4%，
对称配置只 −3.5%，差 6 倍）。但**象限内的配对检验**
（``taker_only`` vs ``both_burst``）在 3 种子下 **p = 0.26，不显著**。

⚠️ 这里缺的是**每组的事件观测个数**（当时 taker_only 31 个点、
both_burst 42 个点），**不是档位数**——
与工作线L 解决的问题（幂律拟合缺自由度）机制不同，
所以**不能套用**"扩到 8 个种子就够了"那个经验值，必须重新算。

⚠️ 前提核对：任务书建议用 ``statsmodels.stats.power.TTestIndPower``，
但**本环境没有装 statsmodels**（实测 ImportError）。
本模块改用 **scipy 的非中心 t 分布精确计算**——
它与 statsmodels 的算法同源（都是非中心 t），且**不引入新依赖**：

    ncp  = d / sqrt(1/n1 + 1/n2)          （非中心参数）
    df   = n1 + n2 − 2
    t*   = t.ppf(1 − α/2, df)
    power = nct.sf(t*, df, ncp) + nct.cdf(−t*, df, ncp)      （双侧）

反解 n 用二分搜索（功效关于 n 单调，二分是稳的）。

用法::

    python scripts/power_analysis_H4.py            # 打印功效分析结论
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scipy import stats  # noqa: E402


# ======================================================================
def power_ttest_1samp(effect_size: float, n: float,
                      *, alpha: float = 0.05) -> float:
    """**单样本** t 检验的功效（检验均值是否异于 0）。

    为什么还需要它：EM.1 算的是"**每组事件点**"的需求，
    而 EM.3 实际用的检验是**跨运行**的——把每个运行的块级差当成一个观测，
    自由度 = 运行数，不是事件点数。两者**口径不同**，
    所以要用各自的方差分别算，不能拿一个去套另一个。
    """
    if n < 2:
        return float("nan")
    df = n - 1
    ncp = effect_size * math.sqrt(n)
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    p = float(stats.nct.sf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp))
    if p != p:
        z = stats.norm.ppf(1 - alpha / 2)
        p = float(stats.norm.sf(z - ncp) + stats.norm.cdf(-z - ncp))
    return p


def solve_n_onesample(effect_size: float, *, power: float = 0.8,
                      alpha: float = 0.05, n_max: int = 200_000) -> float:
    """反解单样本 t 检验所需观测数（用于 EM.3 反算"还需要多少次运行"）。"""
    if effect_size <= 0:
        return float("inf")
    lo, hi = 2.0, 8.0
    while power_ttest_1samp(effect_size, hi, alpha=alpha) < power:
        lo = hi
        hi *= 2
        if hi > n_max:
            return float("nan")
    for _ in range(200):
        mid = (lo + hi) / 2
        if power_ttest_1samp(effect_size, mid, alpha=alpha) < power:
            lo = mid
        else:
            hi = mid
    if power_ttest_1samp(effect_size, hi, alpha=alpha) < power:
        return float("nan")
    return math.ceil(hi)


def power_ttest_ind(effect_size: float, n1: float, *,
                    ratio: float = 1.0, alpha: float = 0.05) -> float:
    """两样本 t 检验的功效（**双侧**，不等样本量）。

    ``ratio`` 的语义：**n2 = n1 × ratio**（即 ratio 是"第二组 / 第一组"）。

    ⚠️ 方向不能反。若实现里把它当成 ``n1 = n2 × ratio``，
    在 ratio=1 时看不出任何差别（这正是它危险的地方：**默认参数下永远不会暴露**），
    但在 42:31 这种不等样本量下就会给出错误的样本量需求。
    变异体 M54 钉的就是这一处。
    """
    if n1 < 2 or ratio <= 0:
        return float("nan")
    n2 = n1 * ratio
    if n2 < 2:
        return float("nan")
    df = n1 + n2 - 2
    # 非中心参数：d / sqrt(1/n1 + 1/n2)
    ncp = effect_size / math.sqrt(1.0 / n1 + 1.0 / n2)
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    p = float(stats.nct.sf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp))
    if p != p:
        # ⚠️ 实测踩到：scipy 的 ``nct`` 在 **大 df + 大 ncp** 下会返回 nan
        #    （例：d=0.5, n1=n2=1000 ⇒ df=1998, ncp=11.2）。
        #    而 ``nan < power`` 是 False ⇒ 二分搜索会**误判"已达标"并提前收敛**，
        #    静默给出一个偏小的样本量。所以必须回退，不能让它带着 nan 往下走。
        #    回退用正态近似（大样本下非中心 t → 正态），实测两种算法在
        #    能算出来的区间（df ≲ 500）差异 < 0.002。
        z_crit = stats.norm.ppf(1 - alpha / 2)
        p = float(stats.norm.sf(z_crit - ncp) + stats.norm.cdf(-z_crit - ncp))
    return p


def solve_n_for_power(effect_size: float, *, power: float = 0.8,
                      ratio: float = 1.0, alpha: float = 0.05,
                      n_max: int = 200_000) -> float:
    """反解达到给定功效所需的第一组样本量 **n1**（第二组 = n1 × ratio）。

    用二分（功效关于 n 单调递增）。``n_max`` 是上界护栏——
    若在这个上界内都达不到目标功效，返回 ``nan`` 而不是硬凑一个数
    （"给不出答案"本身是结论）。
    """
    if effect_size <= 0:
        return float("inf")
    lo, hi = 2.0, 8.0
    while power_ttest_ind(effect_size, hi, ratio=ratio, alpha=alpha) < power:
        lo = hi
        hi *= 2
        if hi > n_max:
            return float("nan")
    for _ in range(200):
        mid = (lo + hi) / 2
        if power_ttest_ind(effect_size, mid, ratio=ratio, alpha=alpha) < power:
            lo = mid
        else:
            hi = mid
    # ⚠️ 收敛后**必须复核**：若回退路径或数值问题让功效评估失准，
    #    这里会返回一个"看起来收敛了但没达标"的数。宁可返回 nan（=给不出答案），
    #    也不许悄悄给一个偏小的样本量。
    if power_ttest_ind(effect_size, hi, ratio=ratio, alpha=alpha) < power:
        return float("nan")
    return math.ceil(hi)


@dataclass
class RequiredN:
    effect_size_cohens_d: float
    required_n_group1: float
    required_n_group2: float
    ratio: float
    alpha: float
    power: float
    total_n: float

    def as_dict(self) -> dict:
        return {
            "effect_size_cohens_d": self.effect_size_cohens_d,
            "required_n_group1": self.required_n_group1,
            "required_n_group2": self.required_n_group2,
            "total_n": self.total_n,
            "ratio": self.ratio, "alpha": self.alpha, "power": self.power,
        }


def estimate_required_n_for_quadrant_test(observed_diff: float,
                                          observed_pooled_sd: float, *,
                                          alpha: float = 0.05,
                                          power: float = 0.8,
                                          ratio: float = 42 / 31) -> RequiredN:
    """用观测到的效应量反推"要达到 80% 功效，每组需要多少个事件观测点"。

    ``ratio`` 默认 42/31 —— **保留两组不等长的实际情况**（both_burst 42 个、
    taker_only 31 个）。任务书 §4.7 明确要求不能套用等样本量公式。
    """
    if observed_pooled_sd <= 0:
        raise ValueError(f"合并标准差必须 > 0，收到 {observed_pooled_sd}")
    d = abs(observed_diff) / observed_pooled_sd
    n1 = solve_n_for_power(d, power=power, ratio=ratio, alpha=alpha)
    n2 = n1 * ratio if n1 == n1 else float("nan")
    return RequiredN(effect_size_cohens_d=d, required_n_group1=n1,
                     required_n_group2=n2, ratio=ratio, alpha=alpha,
                     power=power, total_n=(n1 + n2) if n1 == n1 else float("nan"))


def estimate_seeds_needed(required_n_events: float,
                          observed_events_per_seed: float, *,
                          observed_n_seeds: int = 3) -> dict:
    """把「需要多少个事件观测点」换算成「需要多少种子」或「单次要跑多长」。

    ``observed_events_per_seed`` = 当前每组事件数 / 当前种子数。

    ⚠️ 两条路的**成本结构不同**，不能只看"哪个数字小"：
      · **加种子**：单次运行时长不变，但要有更多随机种子、更多次独立运行；
      · **加长单次模拟**：种子数不变，但单次 tick 数变长——
        **风险是引入非平稳性**（跑太久之后市场可能已经不在标定时假设的稳态上），
        执行前必须核对背景统计特征（峰度、|r|ACF）是否仍与标定阶段一致。
    """
    if observed_events_per_seed <= 0:
        raise ValueError("每种子事件数必须 > 0")
    n_seeds = required_n_events / observed_events_per_seed
    return {
        "required_n_events": required_n_events,
        "events_per_seed": observed_events_per_seed,
        "seeds_needed": n_seeds,
        "extra_seeds_needed": max(0.0, n_seeds - observed_n_seeds),
        "length_multiplier_option": n_seeds / observed_n_seeds,
        "note": ("两条路：①种子数 × %.1f；②单次模拟时长 × %.1f（但要先核对"
                 "加长后的背景统计特征是否仍与标定阶段一致——"
                 "长窗口可能已经不在稳态上）。" % (n_seeds / observed_n_seeds,
                                            n_seeds / observed_n_seeds)),
    }


# ======================================================================
def main() -> dict:
    from _common import banner
    banner("H4 象限检验的功效分析")
    # 观测值来自工作线E 的 symmetric_both 配置（3 种子）
    OBS_DIFF = 2.98
    OBS_SD = 8.0        # 占位：真实值由 run_workstream_M.py 从产物里现读
    r = estimate_required_n_for_quadrant_test(OBS_DIFF, OBS_SD)
    print(f"  Cohen's d = {r.effect_size_cohens_d:.4f}")
    print(f"  需要每组 n1 = {r.required_n_group1:.0f}、n2 = {r.required_n_group2:.0f}"
          f"（合计 {r.total_n:.0f} 个事件观测点）")
    return r.as_dict()


if __name__ == "__main__":
    main()
