"""优化器契约（二期阶段10）。

核心契约只有一个：

    optimizer(real_stats: dict) -> {"best_params": {...}, "best_distance": float, ...}

**为什么必须是这个形状**：块自助法要对**每一组重采样后的真实矩**重跑一次
完整校准，才能得到参数分布。如果优化器被写死成"从磁盘读一份真实矩、跑一次"，
自助法就无从下手——而"给参数不确定性区间"正是本阶段相对阶段4 的唯一实质增量。
"""

from __future__ import annotations

from typing import Callable, Protocol

import numpy as np

from .objective import (
    DEFAULT_MOMENTS,
    SearchSpace,
    cross_seed_scales,
    random_search,
    standardized_distance,
)


class OptimizerFn(Protocol):
    def __call__(self, real_stats: dict) -> dict: ...


def make_optimizer(
    simulate: Callable[[dict], dict],
    space: SearchSpace,
    *,
    moments: tuple[str, ...] = DEFAULT_MOMENTS,
    n_seeds: int = 2,
    n_iter: int = 12,
    n_refine: int = 2,
    seed: int = 20260917,
    fixed: dict | None = None,
) -> OptimizerFn:
    """造一个"给定真实矩 → 最优参数"的优化器。

    ``simulate(params) -> list[矩字典]``（每个种子一份）。
    ⚠️ 它返回的是**多种子**的矩，不是单次——因为标准化要用跨种子标准差，
    而那是**每评估一次都得重估**的：参数变了，市场行为变了，
    矩的噪声水平也跟着变。用一份固定的标准差去标准化所有候选参数，
    会让"噪声大的区域"在距离上被系统性低估。

    ``fixed`` 是不参与搜索的参数（例如已被阶段5/6 标定过的传导系数）。
    **它们必须被固定**：本阶段参数量已经不小（指导书 §3.7 明确警告过
    "参数量爆炸风险"），把全部参数一起搜会显著增加过拟合风险。
    """
    fixed = dict(fixed or {})

    def optimizer(real_stats: dict) -> dict:
        cache: dict[tuple, list[dict]] = {}

        def evaluate(p: dict) -> tuple[float, dict]:
            full = {**fixed, **p}
            key = tuple(sorted((k, round(float(v), 10)) for k, v in full.items()))
            if key in cache:
                samples = cache[key]
            else:
                samples = simulate(full)
                cache[key] = samples
            scales = cross_seed_scales(samples, moments)
            per_seed = []
            for s in samples:
                d, parts = standardized_distance(
                    s, real_stats, moments=moments, scales=scales)
                per_seed.append((d, parts))
            mean_d = float(np.mean([d for d, _ in per_seed]))
            mean_parts = {
                k: float(np.mean([p.get(k, float("nan")) for _, p in per_seed]))
                for k in (per_seed[0][1] if per_seed else {})
            }
            return mean_d, {"scales": scales, "parts": mean_parts,
                            "n_seeds": len(samples)}

        res = random_search(evaluate, space, n_iter=n_iter, seed=seed,
                            n_refine=n_refine)
        res["params_full"] = {**fixed, **res["best_params"]}
        res["moments"] = list(moments)
        return res

    return optimizer


def bootstrap_calibrate(
    optimizer: OptimizerFn, bootstrap_moments: list[dict], *,
    seed: int = 20260917,
) -> list[dict]:
    """对每一组自助重采样的真实矩各跑一次完整校准。

    ⚠️ **成本是 (n_boot × n_iter) 次模拟**。指导书的默认 ``n_bootstrap=50``
    在本项目的模拟成本下是几千场模拟（小时级）。
    所以实际调用时应当把 ``n_boot`` 压到个位数，并如实报告"这是被成本削过的"，
    而不是假装 8 次和 50 次一样能估出 2.5% 分位点——
    **8 个样本估 2.5% 分位点本身就没有意义**，报告里必须写清这一点。
    """
    out = []
    for i, rm in enumerate(bootstrap_moments):
        r = optimizer(rm)
        r["bootstrap_index"] = i
        out.append(r)
    return out


def times_ci_covers(ci: tuple[float, float], value: float) -> bool:
    lo, hi = ci
    if not (np.isfinite(lo) and np.isfinite(hi) and np.isfinite(value)):
        return False
    return bool(lo <= value <= hi)
