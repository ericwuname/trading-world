"""冲击函数与深度剖面：一期阶段3 用过的测量工具（二期阶段6 复用同一实现）。

为什么要有这个模块
------------------
阶段6 的验收标准是「冲击幂律指数 k 相比阶段3 的 0.61~1.25 显著更接近 0.5」。
**要比较，就必须用同一把尺子。** 如果阶段6 自己重写一份拟合函数，
哪怕只是把护栏放宽松一点，得到的"改善"也可能全部来自尺子变了——
而且这种事**不会报错**，产出一个"看起来达标"的数字。

所以这两个函数是从 ``scripts/run_stage3.py`` **原样搬过来**的
（连同那两道护栏的注释），阶段3 也改为从这里 import，全项目只剩一份实现。

两道护栏（都是一期踩出来的，删掉任何一道都会让拟合变得不可信）
--------------------------------------------------------------
① ``min_span``：自变量至少要跨 ``min_span`` 倍。瞬时清算在 1% 持仓以上
   完全饱和，各档实际成交量全是 13.3~13.6 手（跨度 0.7%），此时回归能给出
   ``k=1.72、R²=0.987`` 这种漂亮但**毫无意义**的结果——在数值噪声上拟合。
② ``min_yrange``：因变量也要真的在变。即便自变量勉强跨了 1.5 倍，
   若因变量全都撞在同一个下限上（例如"最差成交价"已经打到盘口最深档），
   斜率同样是被一个点决定的。这种情况要如实报"饱和"，不能报一个指数。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats


@dataclass(slots=True)
class PowerLawFit:
    """幂律拟合结果。``exponent`` 在 ``ok=False`` 时无意义，看 ``reason``。"""

    ok: bool
    exponent: float = float("nan")
    r_squared: float = float("nan")
    n_points: int = 0
    reason: str = ""
    #: 拟合用到的自变量 / 因变量（做图时要用，否则只能画出拟合线而画不出点）
    xs: np.ndarray | None = None
    ys: np.ndarray | None = None

    def verdict(self, *, linear_lo: float = 1.15, super_lo: float = 1.3) -> str:
        """把指数翻译成人话。阈值沿用一期阶段3 的口径，保持可比。"""
        if not self.ok or not np.isfinite(self.exponent):
            return f"拟合失败（{self.reason}）"
        k = self.exponent
        if k > super_lo:
            return "超线性 —— 盘口被击穿后流动性消失，同样的量造成更大冲击"
        if k < linear_lo:
            return "≈线性或更凹 —— 盘口深度足以按比例吸收"
        return "略超线性"

    def closeness_to_sqrt(self) -> float:
        """离平方根律（k=0.5）的绝对距离。阶段6 的主验收量。"""
        if not self.ok or not np.isfinite(self.exponent):
            return float("nan")
        return abs(self.exponent - 0.5)


def fit_power_law(
    rows: list[dict], key: str, *, min_span: float = 1.5, min_yrange: float = 1.2,
    x_key: str = "delivered_qty",
) -> PowerLawFit:
    """对 ``|rows[key]| ≈ rows[x_key]^k`` 做对数-对数回归。

    从 ``scripts/run_stage3.py`` 原样搬来（含两道护栏的语义）。
    ``rows`` 每项至少要有 ``x_key`` 与 ``key`` 两个字段。
    """
    x = np.array([r[x_key] for r in rows], dtype=float)
    y = np.abs(np.array([r[key] for r in rows], dtype=float))
    ok = np.isfinite(x) & (x > 0) & np.isfinite(y) & (y > 0)
    n = int(ok.sum())
    if n < 3:
        return PowerLawFit(False, n_points=n, reason="样本不足（有效档位 < 3）")
    xo, yo = x[ok], y[ok]
    if xo.max() / xo.min() < min_span:
        return PowerLawFit(
            False, n_points=n,
            reason=f"自变量跨度不足（{xo.max() / xo.min():.2f}× < {min_span}×）",
        )
    if yo.max() / yo.min() < min_yrange:
        return PowerLawFit(
            False, n_points=n,
            reason=f"因变量已饱和（{yo.max() / yo.min():.2f}× < {min_yrange}×）",
        )
    lx, ly = np.log(xo), np.log(yo)
    k, b = np.polyfit(lx, ly, 1)
    ss_res = float(((ly - (k * lx + b)) ** 2).sum())
    ss_tot = float(((ly - ly.mean()) ** 2).sum())
    return PowerLawFit(
        True, exponent=float(k),
        r_squared=(1 - ss_res / ss_tot if ss_tot > 0 else float("nan")),
        n_points=n, reason="", xs=xo, ys=yo,
    )


def unsaturated(rows: list[dict], key: str = "fill_ratio",
                threshold: float = 0.95) -> list[dict]:
    """只保留"目标被吃满"的档位。

    一旦目标吃不饱，"清算规模"这个自变量就已经不起作用了（成交额饱和），
    继续算进回归会把斜率系统性压低。饱和区本身是结论（阶段3 的饱和效应），
    但不该混进幂律斜率里。
    """
    return [r for r in rows if r.get(key, 0.0) >= threshold]


def depth_profile_gamma(dist: np.ndarray, cum: np.ndarray,
                        lo: float = 20.0) -> float:
    """累计深度剖面的幂指数 γ：``D(x) ∝ x^γ``。

    这是决定"冲击是凹还是线性"的关键量::

        Q = D(I) = c·I^γ  ⟹  I ∝ Q^(1/γ)
        · γ = 2（真实市场，远端厚）    → I ∝ Q^0.5  = 平方根律（凹）
        · γ = 1（深度沿价格均匀分布）  → I ∝ Q^1    = 线性
        · γ < 1（深度挤在近端）        → I ∝ Q^(1/γ>1) = 超线性

    阶段6 要的就是把 γ 从 0.77~0.8 推向 2。
    """
    ok = (dist >= lo) & (cum > 0) & np.isfinite(dist) & np.isfinite(cum)
    if ok.sum() < 3:
        return float("nan")
    return float(np.polyfit(np.log(dist[ok]), np.log(cum[ok]), 1)[0])


def cumulative_depth(
    dist: np.ndarray, qty: np.ndarray, *, max_bp: float = 400.0, n_bins: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    """把 (每档距离, 每档量) 折成 (分箱中点, 累计深度)。

    从 ``scripts/run_stage3.py::depth_profile`` 里搬出来的纯计算部分，
    与它逐点等价（阶段3 那边改为调用本函数）。

    ⚠️ ``dist`` 必须是**绝对值**（距中间价多远）。买盘的
    ``(p/mid - 1)`` 全是负数，忘了取绝对值会让任何阈值都包含全部档位，
    曲线直接变成一条平线——而这个错误不会报错，只会让 γ ≈ 0。
    """
    d = np.abs(np.asarray(dist, dtype=float))
    q = np.asarray(qty, dtype=float)
    edges = np.linspace(0.0, float(max_bp), int(n_bins) + 1)
    cum = np.array([q[d <= e].sum() for e in edges[1:]], dtype=float)
    return (edges[1:] + edges[:-1]) / 2.0, cum


@dataclass(slots=True)
class TierSummary:
    """一组配对实验的汇总（均值 ± 标准误 + t 统计量）。"""

    n: int = 0
    mean: float = float("nan")
    sem: float = float("nan")
    t: float = float("nan")
    values: list[float] = field(default_factory=list)

    @property
    def ci95(self) -> tuple[float, float]:
        """t 分布近似的 95% 置信区间（大样本下 t≈1.96；小样本用保守值）。"""
        if self.n < 2 or not np.isfinite(self.sem):
            return (float("nan"), float("nan"))
        crit = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
                6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}.get(
                    self.n - 1, 1.96)
        return (self.mean - crit * self.sem, self.mean + crit * self.sem)

    def covers(self, value: float) -> bool:
        lo, hi = self.ci95
        return bool(np.isfinite(lo) and lo <= value <= hi)


def classify_burst_vs_background(intensity_trace, *,
                                 percentile_threshold: float = 80.0
                                 ) -> np.ndarray:
    """按强度（ρ）把每个 tick 标成「突发期 / 背景期」。

    阈值取该次运行内 ρ 的 ``percentile_threshold`` 百分位——即**相对自己**，
    而不是一个绝对强度。理由：不同分支比的配置有不同的 ρ 分布，
    用绝对阈值会让"突发"的定义随配置漂移，跨配置就不可比了。

    ⚠️ ``percentile_threshold`` 写反（比如取 20）会得到**互补**的掩码，
    于是"突发期深度更薄"会变成"突发期深度更厚"，结论方向直接颠倒。
    这正是变异体 M35 注入的错误。
    """
    x = np.asarray(intensity_trace, dtype=float)
    if x.size == 0:
        return np.zeros(0, dtype=bool)
    thr = float(np.nanpercentile(x, percentile_threshold))
    return x > thr


def burst_depth_contrast(depth_snapshots, intensity_trace, *,
                         percentile_threshold: float = 80.0,
                         block: int = 250, min_each: int = 20) -> dict:
    """**单次运行**内的突发/背景深度对比 → 一个 log 比值（外加块级比值）。

    ⭐ 为什么这个函数的契约是"一次运行"
    ==================================
    本项目最贵的一条纪律是「**一场模拟 = 一个观测**」（见 ``summarize``
    的注释与 README 的测量纪律）。相邻 tick 的深度与强度都是强自相关的，
    把每个 tick 当独立样本会把**不确定性**算小一个数量级。

    实测（合成 AR(1) 深度 φ=0.97、强度 φ=0.95、8 次运行 × 6000 tick，
    真实效应 −15%）：::

        per_run（每次运行一个观测，n=8）  CI95 半宽 = 0.00390
        把 tick 全池化直接丢进 t 检验      隐含相对半宽 = 0.00034   ← 窄了 11.6 倍

    ⚠️ 说清楚**夸大的是什么**：tick 级的**点估计没错**
    （那里 log 比值 ≈ −0.163，与真值 −0.163 一致），
    错的是它的区间与 p 值。真值恰为 0 时它不会凭空造假阳性
    （t ≈ 0，实测中位数 p ≈ 0.26），
    **但它把"小效应"和"噪声"的分辨能力整个抹掉了**——
    任何 −1% 的微小差异都会被报成 p=0。判据一旦失去分辨力，
    "突发期显著更薄"就变成一句几乎总是成立的话。

    所以本函数**只负责一次运行内的描述**，把"跨运行合并"交给
    :func:`aggregate_burst_contrast`——那里才有诚实的自由度。
    返回值里保留 ``tick_level`` 是**故意**的：让读者亲眼看到
    那个窄了十几倍的区间长什么样，而不是假装它不存在。
    """
    d = np.asarray(depth_snapshots, dtype=float)
    x = np.asarray(intensity_trace, dtype=float)
    n = min(d.size, x.size)
    d, x = d[:n], x[:n]
    ok = np.isfinite(d) & np.isfinite(x)
    d, x = d[ok], x[ok]
    out: dict = {"n_ticks": int(d.size)}
    if d.size < 2 * min_each:
        out.update({"usable": False, "reason": "tick 数不足以分成两类",
                    "log_ratio": float("nan"), "block_ratios": []})
        return out
    mask = classify_burst_vs_background(x, percentile_threshold=percentile_threshold)
    db, dg = d[mask], d[~mask]
    out["n_burst"] = int(mask.sum())
    out["n_background"] = int((~mask).sum())
    if db.size < min_each or dg.size < min_each:
        out.update({"usable": False, "reason": "有一类样本太少",
                    "log_ratio": float("nan"), "block_ratios": []})
        return out
    out["burst_mean"] = float(db.mean())
    out["background_mean"] = float(dg.mean())
    out["pct_diff"] = float((db.mean() - dg.mean()) / dg.mean())
    # 该运行的**单一** log 比值（跨运行合并时用它，n = 运行数）
    out["log_ratio"] = (float(np.log(db.mean() / dg.mean()))
                        if db.mean() > 0 and dg.mean() > 0 else float("nan"))
    # 参考用：tick 级检验（自由度不诚实，只用来展示它有多夸张）
    try:
        t_tick, p_tick = stats.ttest_ind(db, dg, equal_var=False)
        out["tick_level"] = {"t": float(t_tick), "p": float(p_tick)}
    except Exception:                                     # pragma: no cover
        out["tick_level"] = {"t": float("nan"), "p": float("nan")}
    # 块级比值（同一次运行内多给几个观测，用于稳健性对照）
    ratios = []
    step = max(int(block), 1)
    for lo in range(0, d.size, step):
        hi = min(lo + step, d.size)
        sb, sg = mask[lo:hi], ~mask[lo:hi]
        if sb.sum() < 5 or sg.sum() < 5:
            continue
        mb, mg = float(d[lo:hi][sb].mean()), float(d[lo:hi][sg].mean())
        if mb > 0 and mg > 0:
            ratios.append(float(np.log(mb / mg)))
    out["block_ratios"] = ratios
    out["usable"] = True
    return out


def aggregate_burst_contrast(runs: list[dict], *, label: str = "") -> dict:
    """把多次运行的对比结果合并成**一个诚实检验**。

    主检验用 ``log_ratio``（每次运行一个观测，n = 运行数）做单样本 t。
    副检验把所有运行内的**块级**比值汇总（n = 运行数 × 每运行块数）——
    后者自由度更大，但块之间的自相关没有完全消除，所以
    **以主检验为准，块级只作对照**；两者结论若不一致，报告里必须写出来。
    """
    per_run = np.asarray([r["log_ratio"] for r in runs
                          if r.get("usable") and np.isfinite(r.get("log_ratio", np.nan))],
                         dtype=float)
    blocks = np.asarray([v for r in runs for v in r.get("block_ratios", [])],
                        dtype=float)
    out: dict = {"label": label, "n_runs": int(per_run.size),
                 "n_blocks_total": int(blocks.size)}

    def one_sample(a: np.ndarray) -> dict:
        if a.size < 2:
            return {"n": int(a.size), "mean": float("nan"), "t": float("nan"),
                    "p": float("nan"), "ci95": [float("nan"), float("nan")]}
        m = float(a.mean())
        sem = float(a.std(ddof=1) / np.sqrt(a.size))
        if sem <= 0:
            return {"n": int(a.size), "mean": m, "t": float("nan"),
                    "p": float("nan"), "ci95": [m, m]}
        t = m / sem
        half = float(stats.t.ppf(0.975, a.size - 1) * sem)
        return {"n": int(a.size), "mean": m, "t": float(t),
                "p": float(2 * stats.t.sf(abs(t), a.size - 1)),
                "ci95": [m - half, m + half]}

    out["per_run"] = one_sample(per_run)
    out["per_block"] = one_sample(blocks)
    # 相对差异（%）：exp(log比值) − 1
    pr = out["per_run"]
    out["pct_effect"] = (float(np.exp(pr["mean"]) - 1.0)
                         if np.isfinite(pr["mean"]) else float("nan"))
    if not (np.isfinite(pr["p"]) and pr["p"] < 0.05):
        out["verdict"] = "不显著：不能宣称突发期深度与背景期不同"
    else:
        out["verdict"] = ("突发期显著更薄" if pr["mean"] < 0
                          else "突发期显著更**厚**（与 H1 预期相反）")
    return out


def summarize(values) -> TierSummary:
    """配对差值的均值 ± 标准误 + t 统计量。

    一期踩过的坑：单条路径的冲击峰值里，随机漂移远大于冲击本身
    （实测跨种子标准差 89.5bp → 配对后降到 1.4bp）。所以**必须**多种子
    配对，只报一条曲线就是在报噪声。
    """
    a = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if a.size == 0:
        return TierSummary()
    if a.size == 1:
        return TierSummary(n=1, mean=float(a[0]), sem=float("nan"),
                           t=float("nan"), values=[float(v) for v in a])
    m = float(a.mean())
    sem = float(a.std(ddof=1) / np.sqrt(a.size))
    return TierSummary(
        n=int(a.size), mean=m, sem=sem,
        t=(m / sem if sem > 0 else float("nan")),
        values=[float(v) for v in a],
    )
