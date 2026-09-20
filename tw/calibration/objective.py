"""统一校准框架（二期阶段10）。

阶段4 用的是网格搜索 + 一条手工设计的距离分数。本模块替换它，做三件事：

1. **系统化的矩量距离**（MSM 的简化形式）：每个矩先用它自己的跨种子标准差
   标准化，再算加权距离——而不是裸用原始差值。
   一期已经手工处理过"``r ACF(1)`` 的真实值≈0、分母趋零"这一个指标，
   剩下的指标是裸差值的，于是**量级大的指标（σ、峰度）会压制量级小的**
   （VR、ACF），而报告里看不出来。
2. **块自助法（block bootstrap）给出**参数的不确定性区间**，不是一个点估计。
   阶段4 的"训练段/测试段"切分只能检查过拟合，不能量化"参数本身有多准"。
3. **诚实回答"MSM 是否真的比网格搜索更好"**。如果在这个参数量级下
   网格搜索已经够用，那也是一个有价值的结论——**不允许为了让新框架显得有用
   而挑对自己有利的对比口径**。

⭐ 两个必须说清楚的技术选择
==========================

**① 块自助法而不是逐点自助法。**
   真实数据的矩（σ、ACF、峰度）都有强自相关。逐点重采样会**破坏**时间结构，
   得到的参数分布会窄得离谱（等于假装样本量是 n 而不是有效样本量）。
   块自助法保留块内的时序依赖，块长取 168（一周的小时数）——
   与指导书一致，也大于我们关心的 ACF 最大滞后（10~24）。

**② 权重矩阵只能用**对角**（跨种子标准差）。**
   完整的 MSM 用矩的协方差矩阵的逆做权重，但在"每次模拟很贵"的约束下
   估不准那个协方差（需要几十倍于矩数量的模拟次数）。
   强行用样本协方差去估会得到**奇异或近奇异**的矩阵，其逆会把噪声放大成
   主导方向——比裸欧氏距离更糟。所以这里明确只用对角，并写进边界。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

#: 参与校准的矩。名字必须与 ``tw/analyzer.py`` 的 flat() 输出一致。
DEFAULT_MOMENTS: tuple[str, ...] = (
    "sigma_bp", "excess_kurtosis", "acf_abs_lag1", "acf_abs_mean_1_10",
    "acf_ret_lag1", "vr5", "hill_alpha_left",
)

#: 真实数据块自助法的块长（一周 = 168 小时，与指导书一致）。
#:
#: ⚠️⚠️ **这个默认值对本项目的矩是不充分的——已实测。**
#: 用 BTC 真实小时线（17520 根）做块自助法，看 ``acf_abs_lag1``（波动率聚集）的
#: 均值相对全样本值的恢复程度：
#:
#: ==========  ==========
#: 块长         恢复比例
#: ==========  ==========
#: 168（一周）  **0.9%**
#: 500          6.9%
#: 2000         32.4%
#: 4000         58.0%
#: 8000（半年）  69.3%
#: ==========  ==========
#:
#: 也就是说：**块长必须覆盖该矩的依赖长度**，而波动率聚集的依赖长度远长于一周。
#: 用 168 去给"参数不确定性"做区间，得到的区间是**围绕一个错的中心**的——
#: 它看起来正常，实际系统性低估该矩的水平。
#: ETH 同样（0.9% → 60.8%），所以不是单个数据集的偶然。
#:
#: 结论：**块自助法对本类的 ACF 型矩不适用。**
#: 本模块保留它（指导书要求，且对均值型矩仍然有效），
#: 但阶段10 的报告必须把这条偏差写进"诚实边界"，
#: 并且参数置信区间只能按"**相对**不确定性"读，不能按绝对水平读。
BOOTSTRAP_BLOCK = 168

#: 建议的块长（当矩里含 ACF 型指标时）。见上面的实测表。
BOOTSTRAP_BLOCK_LONG = 8000


# ----------------------------------------------------------------------
# 1. 距离
# ----------------------------------------------------------------------
def standardized_distance(
    sim: dict, real: dict, *, moments: tuple[str, ...] = DEFAULT_MOMENTS,
    scales: dict[str, float] | None = None,
) -> tuple[float, dict]:
    """标准化矩量距离。返回 ``(总距离, 每个矩的标准化差)``。

    ``d = Σ_k ((sim_k − real_k) / scale_k)²``，``scale_k`` 是该矩的**跨种子标准差**
    （没有时退回一个量级参考值）。

    为什么必须标准化（模块文档）：不标准化时，σ 的量级是几十 bp、
    ``r ACF(1)`` 的量级是 0.01，裸差值的平方里 σ 一项就压倒其余六项——
    **校准会退化成"只把 σ 调对"**，而报告里显示的是"综合距离"。
    这个偏差不会报错，只会让校准出来的参数在别的矩上系统性偏。
    """
    parts: dict[str, float] = {}
    total = 0.0
    for k in moments:
        a, b = sim.get(k), real.get(k)
        if a is None or b is None:
            continue
        try:
            af, bf = float(a), float(b)
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(af) and math.isfinite(bf)):
            continue
        scale = (scales or {}).get(k, 0.0)
        if not (scale and scale > 1e-12):
            # 没有跨种子标准差时，退回"该矩自身的量级"做分母。
            # ⚠️ 用 |real| 而不是 1.0：real 为 0 附近时（比如 r ACF(1)）
            # 用 1.0 会让那一项贡献接近 0，等于把它从校准里摘掉。
            scale = max(abs(bf), 1e-6)
        z = (af - bf) / scale
        parts[k] = float(z)
        total += float(z * z)
    return float(total), parts


def cross_seed_scales(samples: list[dict],
                      moments: tuple[str, ...] = DEFAULT_MOMENTS) -> dict[str, float]:
    """从多种子样本估每个矩的跨种子标准差（对角权重）。"""
    out: dict[str, float] = {}
    for k in moments:
        vals = [float(s[k]) for s in samples
                if s.get(k) is not None and np.isfinite(s.get(k, float("nan")))]
        if len(vals) < 2:
            continue
        out[k] = float(np.std(vals, ddof=1))
    return out


# ----------------------------------------------------------------------
# 2. 真实数据的块自助法
# ----------------------------------------------------------------------
def block_bootstrap_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """块自助法的下标序列。长度 == n（截断到 n）。

    做法：把序列切成不重叠的块，**有放回**地抽块，拼起来后截断到 n。
    保留块内的时序依赖（模块文档 ①）。
    """
    if block < 1:
        raise ValueError("block 必须 >= 1")
    if n < 1:
        return np.zeros(0, dtype=int)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, max(1, n - block + 1), size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
    return np.clip(idx, 0, n - 1)


def block_bootstrap_series(
    arr: np.ndarray, block: int, rng: np.random.Generator
) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    return a[block_bootstrap_indices(a.size, block, rng)]


def bootstrap_real_moments(
    close: np.ndarray, *, n_boot: int, block: int = BOOTSTRAP_BLOCK,
    seed: int = 20260917, moment_fn=None,
) -> list[dict]:
    """对真实价格序列做块自助法，返回 ``n_boot`` 组矩。

    ``moment_fn(close) -> dict`` 默认用 ``tw/analyzer.py`` 的全套指标。
    """
    from ..analyzer import analyze

    fn = moment_fn or (lambda c: analyze(c, "boot", vol_window=24).flat())
    rng = np.random.default_rng([seed, 0xB007])
    out = []
    for _ in range(n_boot):
        res = block_bootstrap_series(np.asarray(close, dtype=float), block, rng)
        out.append(fn(res))
    return out


# ----------------------------------------------------------------------
# 3. 优化器
# ----------------------------------------------------------------------
@dataclass(slots=True)
class SearchSpace:
    """每个参数的搜索区间。``log=True`` 表示在对数尺度上均匀采样。"""

    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    log: dict[str, bool] = field(default_factory=dict)

    def sample(self, rng: np.random.Generator) -> dict[str, float]:
        out = {}
        for k, (lo, hi) in self.bounds.items():
            if self.log.get(k):
                out[k] = float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
            else:
                out[k] = float(rng.uniform(lo, hi))
        return out

    def clip(self, p: dict[str, float]) -> dict[str, float]:
        out = {}
        for k, (lo, hi) in self.bounds.items():
            v = float(p.get(k, lo))
            out[k] = float(min(hi, max(lo, v)))
        return out


def random_search(
    evaluate, space: SearchSpace, *, n_iter: int, seed: int = 20260917,
    n_refine: int = 1, refine_frac: float = 0.15,
) -> dict:
    """随机搜索 + 局部细化。

    ``evaluate(params) -> (距离, 诊断字典)``。

    **为什么用随机搜索而不是贝叶斯优化**：目标函数每次评估要跑一场模拟
    （秒级到十几秒），而贝叶斯优化的收益要在**几十到上百次**评估之后才体现。
    在这个预算下，随机搜索的期望表现与它没有可测差距，而实现与调试成本低得多。
    指导书给了"MSM **或**贝叶斯优化"两个选项，这里选前者是有理由的选择，
    不是偷懒。要改成贝叶斯优化只需替换本函数。
    """
    rng = np.random.default_rng([seed, 0x5EED])
    trials: list[dict] = []
    best_p, best_d, best_diag = None, float("inf"), {}
    for _ in range(max(1, n_iter)):
        p = space.sample(rng)
        d, diag = evaluate(p)
        trials.append({"params": p, "distance": float(d)})
        if np.isfinite(d) and d < best_d:
            best_p, best_d, best_diag = p, float(d), diag
    # 局部细化：在每个维度上按 refine_frac 的幅度再试几轮
    if best_p is not None and n_refine > 0:
        cur_p, cur_d = dict(best_p), best_d
        for _ in range(n_refine):
            cand = {}
            for k, (lo, hi) in space.bounds.items():
                span = (hi - lo) * refine_frac
                cand[k] = float(np.clip(cur_p[k] + rng.normal(0.0, span), lo, hi))
            d, diag = evaluate(cand)
            trials.append({"params": cand, "distance": float(d)})
            if np.isfinite(d) and d < cur_d:
                cur_p, cur_d, best_diag = cand, float(d), diag
        best_p, best_d, best_diag = cur_p, cur_d, best_diag
    return {"best_params": best_p or {}, "best_distance": best_d,
            "best_diagnostics": best_diag, "trials": trials,
            "n_evaluations": len(trials)}


def percentile_ci(values, lo: float = 2.5, hi: float = 97.5) -> tuple[float, float]:
    """百分位置信区间。样本里含 nan 时自动剔除。"""
    a = np.asarray([v for v in values if v is not None], dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan"), float("nan")
    if a.size == 1:
        return float(a[0]), float(a[0])
    return float(np.percentile(a, lo)), float(np.percentile(a, hi))


def parameter_uncertainty(bootstrap_results: list[dict],
                          keys: list[str]) -> dict[str, dict]:
    """把自助法的参数估计整理成 "每个参数的均值 + 95% CI"。"""
    out: dict[str, dict] = {}
    for k in keys:
        vals = [r.get("best_params", {}).get(k) for r in bootstrap_results]
        lo, hi = percentile_ci(vals)
        arr = np.asarray([v for v in vals if v is not None and np.isfinite(v)],
                         dtype=float)
        out[k] = {
            "mean": float(arr.mean()) if arr.size else float("nan"),
            "sd": float(arr.std(ddof=1)) if arr.size > 1 else float("nan"),
            "ci95": [lo, hi],
            "n": int(arr.size),
            "values": [float(v) for v in arr],
        }
    return out
