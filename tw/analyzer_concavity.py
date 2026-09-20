"""凹度判据：不依赖幂律全局拟合的「相邻档位局部弹性」。

为什么需要这个模块
------------------
诊断报告证明：把**整条**冲击曲线拟合成一个幂律指数 k（``fit_power_law``），
在「4 档 × 3 种子」下一次要用到 4 个点、且低档位信噪比很低，
结果 k 的 95% 区间宽度达到 2.0~2.7——19/19 个历史臂的区间都包含 0.5，
**这个刻度无法判定任何一对比较**。

替代思路（本模块）：**只看相邻两档**。

    线性    ：规模翻倍 → 滑点也翻倍        ⇒ 局部弹性 e = 1
    平方根律：规模翻倍 → 滑点只涨 √2≈1.414 ⇒ 局部弹性 e ≈ 0.5

    e = log(|slippage_big| / |slippage_small|) / log(size_big / size_small)

它只用**两个档位**，不做跨全曲线的全局回归，所以方差远小于 k。
判决也改成**三分类**（``concavity_verdict``），而不是"离 0.5 差多少"：

    · 显著凹          （弹性的 95% 区间上界 < 1 ⇒ 显著小于线性）
    · 显著线性或更差  （区间下界 ≥ 1）
    · 不可判定        （区间跨过 1）

⚠️ 两个诚实性设计（任务书没写，但不做会让判据本身变得不可信）
-----------------------------------------------------------
1. **不确定度怎么来**：任务书的伪代码要求"对每个种子分别算弹性"，
   但**现有产物只存了每档的均值 ± sem，没有逐种子的数组**
   （见 ``out/workstream_A_metrics.json`` 的 ``rows``：只有
   ``slippage_bp`` / ``slippage_sem_bp`` / ``n_seeds``）。
   所以本模块支持两条路径：给了逐种子数组就用逐种子；
   否则用 **delta method** 从均值 ± sem 一阶传播：
       Var(e) ≈ [ (sem_b/|b|)² + (sem_a/|a|)² ] / [log(s_b/s_a)]²
   两条路径都如实标注 ``method`` 字段——**不许把近似冒充成原始数据**。

2. **低信噪比的档对要标出来**：如果某一档的滑点均值连自身 sem 都不超过
   （``|mean| < sem``，即与 0 不可分），那么这一档的"弹性"是在拿噪声做比值。
   本模块给这样的档对打 ``low_snr=True``，**判决照给但必须连同这个标记一起读**。

3. **置信区间用 t 分布，不用 1.96**：任务书 §3.6 自己也指出，
   种子数少时正态近似不准。本项目一律用``scipy.stats.t`` 的临界值
   （与 ``diagnose_k_uncertainty`` 的 bootstrap 用的是同一套学生 t 假设）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy import stats

#: 三分类判决的取值（集中定义，避免各处手写字符串写错）
VERDICT_CONCAVE = "显著凹"
VERDICT_LINEAR_OR_WORSE = "显著线性或更差"
VERDICT_UNDECIDED = "不可判定"


@dataclass
class ElasticityResult:
    """一次相邻档位比较的结果。``ok=False`` 时 ``reason`` 说明为什么没算。"""

    ok: bool = False
    size_small: float = float("nan")
    size_big: float = float("nan")
    mean: float = float("nan")
    sem: float = float("nan")
    ci95: tuple[float, float] = (float("nan"), float("nan"))
    t_vs_linear: float = float("nan")
    p_vs_linear: float = float("nan")
    t_vs_sqrt: float = float("nan")
    p_vs_sqrt: float = float("nan")
    n_seeds: int = 0
    method: str = ""
    low_snr: bool = False
    reason: str = ""
    extras: dict = field(default_factory=dict)


def _t_crit(n: int, alpha: float = 0.05) -> float:
    """双侧 t 临界值（df = n−1）。种子少时它比正态的 1.96 宽，这是应该的。"""
    df = max(1, n - 1)
    return float(stats.t.ppf(1 - alpha / 2, df))


def pairwise_local_elasticity(
    size_small: float,
    size_big: float,
    *,
    mean_small: float,
    sem_small: float,
    mean_big: float,
    sem_big: float,
    n_seeds: int,
    per_seed_small: Sequence[float] | None = None,
    per_seed_big: Sequence[float] | None = None,
    alpha: float = 0.05,
) -> ElasticityResult:
    """相邻档位的局部弹性 + 与线性(1.0)/平方根律(0.5)的检验。

    参数取**滑点均值与其标准误**。若 ``per_seed_*`` 都给了，就走逐种子路径
    （更接近任务书原设计）；否则走 delta method（因为现有产物只存了均值 ± sem）。
    """
    r = ElasticityResult(size_small=float(size_small), size_big=float(size_big),
                         n_seeds=int(n_seeds))

    # ---------- 前置合法性 ----------
    if not (size_big > size_small > 0):
        r.reason = f"档位顺序/取值非法：small={size_small} big={size_big}"
        return r
    if n_seeds < 2:
        r.reason = f"种子数不足（{n_seeds} < 2），无法给不确定度"
        return r
    if mean_small == 0 or mean_big == 0:
        r.reason = "滑点均值为 0，弹性无定义"
        return r

    log_ratio = math.log(size_big / size_small)

    # ---------- 计算弹性 ----------
    if per_seed_small is not None and per_seed_big is not None and len(
            per_seed_small) == len(per_seed_big) == n_seeds:
        a = np.asarray(per_seed_small, dtype=float)
        b = np.asarray(per_seed_big, dtype=float)
        if np.any(a == 0) or np.any(b == 0):
            r.reason = "逐种子滑点里有 0，弹性无定义"
            return r
        e = np.log(np.abs(b) / np.abs(a)) / log_ratio
        mean_e = float(e.mean())
        sem_e = float(e.std(ddof=1) / math.sqrt(len(e)))
        r.method = "per_seed"
        r.extras["elasticities"] = [float(v) for v in e]
    else:
        # delta method：把两端"相对标准误"换算到 log 尺度再合成
        rel_a = float(sem_small) / abs(float(mean_small))
        rel_b = float(sem_big) / abs(float(mean_big))
        mean_e = math.log(abs(float(mean_big)) / abs(float(mean_small))) / log_ratio
        sem_e = math.sqrt(rel_a ** 2 + rel_b ** 2) / abs(log_ratio)
        r.method = "delta_method"

    r.ok = True
    r.mean = mean_e
    r.sem = sem_e

    # ---------- 区间与检验 ----------
    tc = _t_crit(n_seeds, alpha)
    r.ci95 = (mean_e - tc * sem_e, mean_e + tc * sem_e)
    if sem_e > 0:
        r.t_vs_linear = float((mean_e - 1.0) / sem_e)
        r.p_vs_linear = float(stats.t.sf(abs(r.t_vs_linear), n_seeds - 1) * 2)
        r.t_vs_sqrt = float((mean_e - 0.5) / sem_e)
        r.p_vs_sqrt = float(stats.t.sf(abs(r.t_vs_sqrt), n_seeds - 1) * 2)

    # ---------- 低信噪比标记 ----------
    # 任一端与 0 不可分，则这一对是在拿噪声做比值
    r.low_snr = bool(abs(float(mean_small)) < abs(float(sem_small))
                     or abs(float(mean_big)) < abs(float(sem_big)))
    return r


def concavity_verdict(res: ElasticityResult, alpha: float = 0.05) -> tuple[str, tuple]:
    """三分类判决——**这就是新的验收标准**，替代「k 要接近 0.5」。

    判据用的是**区间的端点与 1 的关系**，不是点估计：
      上界 < 1          → 显著凹（冲击比线性更"吃不动"）
      下界 ≥ 1          → 显著线性或更差
      跨过 1            → 不可判定

    ⚠️ 边界写法：用的是 ``< 1.0`` 与 ``>= 1.0``，两者在 ``ci == 1.0`` 处**互补**，
    不会出现"谁也不认领"的缝隙（变异体 M47 钉的就是这个）。
    """
    if not res.ok:
        return VERDICT_UNDECIDED, (float("nan"), float("nan"))
    lo, hi = res.ci95
    if hi < 1.0:
        return VERDICT_CONCAVE, (lo, hi)
    if lo >= 1.0:
        return VERDICT_LINEAR_OR_WORSE, (lo, hi)
    return VERDICT_UNDECIDED, (lo, hi)


def summarize_pair(res: ElasticityResult) -> str:
    """一行人类可读摘要（报告与日志共用，避免两处各写一份格式）。"""
    if not res.ok:
        return f"[{res.size_small}→{res.size_big}] 未计算：{res.reason}"
    verdict, (lo, hi) = concavity_verdict(res)
    flag = " ⚠️低信噪比" if res.low_snr else ""
    return (f"[{res.size_small}→{res.size_big}] e={res.mean:.3f} ± {res.sem:.3f}"
            f"  95%CI=[{lo:.3f}, {hi:.3f}]  → {verdict}"
            f"（t_vs_linear={res.t_vs_linear:.2f}, n={res.n_seeds},"
            f" {res.method}）{flag}")


def elasticity_from_rows(rows: list[dict], i_small: int, i_big: int,
                         *, mean_key: str = "slippage_bp",
                         sem_key: str = "slippage_sem_bp",
                         size_key: str = "shock_frac") -> ElasticityResult:
    """从「一臂的逐档 rows」里取两档做比较——复用它，不要各处手抄字段名。"""
    a, b = rows[i_small], rows[i_big]
    return pairwise_local_elasticity(
        a[size_key], b[size_key],
        mean_small=a[mean_key], sem_small=a.get(sem_key, float("nan")),
        mean_big=b[mean_key], sem_big=b.get(sem_key, float("nan")),
        n_seeds=int(a.get("n_seeds", 0)),
    )
