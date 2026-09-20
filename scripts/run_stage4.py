"""阶段4：参数校准与敏感性分析（施工蓝图 §4 阶段4）。

蓝图原话
--------
"用已有真实数据的统计特征作为目标，网格搜索 agent 比例、报价幅度等参数……
 **注意：这一步的目的是'理解哪些参数组合能产生真实市场的模式'，
   不是'校准出一个用来预测的模型'——警惕把这一步异化成又一轮参数过拟合。**"

所以本脚本内置了两道**防自欺**设计，不然这一步必然退化成"挑一个最好看的数字"：

**① 训练/测试切分（样本外检验）**
   真实 BTC 1h 数据切成前 60%（训练段）和后 40%（测试段），各自独立算统计特征。
   参数在训练段上搜索，然后**用同一组参数在测试段上再评一次分**。
   · 若训练段好、测试段也接近 → 参数抓到的是**稳定的机制关系**
   · 若训练段好、测试段差很多 → 抓到的是训练段的**样本噪声**，不可信
   这条检验不能省：真实市场的统计特征本身有抽样波动，
   拿全样本当目标去挑参数，等于用未来数据选参数。

**② 多随机种子**
   每个参数组合跑 2 个种子取平均。模拟的单次结果方差很大，
   不做多种子平均，"最优参数"里有多少是种子运气分不清。
   （同一个教训在用户既有的研究里已经出现过：单次回测的 Sharpe 不可信。）

评分口径（明确写出来，避免"挑了对自己有利的指标"）
------------------------------------------------
对每个指标用**对数比值的绝对值**，因为关注的是"数量级像不像"而不是"数值相等"：
    · 峰度 / |r|ACF / σ / Hill α  → |ln(sim/real)|（正比型指标）
    · 方差比 VR(5)               → |sim-real|/0.15（真实值 ≈ 1，用绝对差）
    · 收益率 ACF(1)              → |sim-real|/0.03（真实值 ≈ 0，用绝对差）
最后取平均。分数 0 = 完全一致；< 0.3 表示各项都在 ~±35% 以内。
"""

from __future__ import annotations

import sys
from pathlib import Path

import itertools
import time

import numpy as np

# 让 `import scripts.run_stageN` 也能工作（测试会这么做）：
# 平时 `python scripts/run_stageN.py` 时脚本目录自动在 sys.path 上，
# 但作为包被导入时不在 → `ModuleNotFoundError: _common`。
# 二期的 run_stage5/6/7/8/9/10/11 都有这一行；一期这四个当时没加，
# 直到 `tests/test_impact.py` 要 import run_stage3 验证重构等价性时才暴露。
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    FIG,
    banner,
    real_metrics,
    save_json,
)
from tw import Market, Population, SimConfig  # noqa: E402
from tw.analyzer import analyze  # noqa: E402

N_AGENTS = 300
N_TICKS = 5_000
SEEDS = [20260917, 424242]

#: 网格（保持规模可控：4 × 3 × 2 = 24 组 × 2 种子 = 48 次运行）
CHARTIST_SHARES = [0.15, 0.22, 0.30, 0.38]
FUND_SHARES = [0.25, 0.33, 0.42]
OFFSET_HI = [0.0012, 0.0018, 0.0028]

#: 4b 对症网格：直接针对阶段2 已诊断出的失配调参，而不是盲扫更多维度。
#: 诊断结果（阶段2 四池消融）：混合池 r ACF(1) = −0.177、VR(5) = 0.573，
#: 而真实 BTC 是 −0.017 / 0.968 —— **过度均值回归**，价格被基本面锚拽得太紧。
#: 直接控制这个机制的两个旋钮就是基本面派的「无套利死区」与「报价激进度」：
#:   · 死区越大 → 小偏离时基本面派不出手 → 均值回归越弱
#:   · 激进度越小 → 出手时报价越贴近中间价 → 价格冲击越小
#: 只在这两个轴上做小网格（3×3），比把主网格扩成四维更省算力、也更好解释。
FOCUS_DEADBAND = [0.002, 0.01, 0.03]
FOCUS_AGGRESS = [0.004, 0.002, 0.0008]

TRAIN_FRAC = 0.6


# ----------------------------------------------------------------------
def real_targets() -> dict[str, dict]:
    """真实基准：训练段 / 测试段 / 全样本 三套。"""
    from _common import real_series

    s = real_series("BTCUSDT_1h")
    n = len(s)
    cut = int(n * TRAIN_FRAC)

    def metrics_on(lo: int, hi: int, tag: str):
        return analyze(s.close[lo:hi], tag, vol_window=24).flat()

    return {
        "train": metrics_on(0, cut, "BTC 1h 训练段"),
        "test": metrics_on(cut, n, "BTC 1h 测试段"),
        "full": metrics_on(0, n, "BTC 1h 全样本"),
        "cut_index": cut,
        "n": n,
    }


def score(sim: dict, target: dict) -> tuple[float, dict]:
    """距离分数（越小越像）。返回 (总分, 分项)。"""
    parts: dict[str, float] = {}

    def logratio(key: str) -> None:
        a, b = sim.get(key), target.get(key)
        if a is None or b is None or not np.isfinite(a) or not np.isfinite(b) or b == 0:
            return
        if a <= 0:
            parts[key] = 3.0
            return
        parts[key] = abs(float(np.log(a / b)))

    for k in ("excess_kurtosis", "acf_abs_lag1", "acf_abs_mean_1_10", "sigma_bp", "hill_alpha_left"):
        logratio(k)

    def absdiff(key: str, scale: float) -> None:
        a, b = sim.get(key), target.get(key)
        if a is None or b is None or not np.isfinite(a) or not np.isfinite(b):
            return
        parts[key] = abs(float(a) - float(b)) / scale

    absdiff("vr5", 0.15)
    absdiff("acf_ret_lag1", 0.03)

    if not parts:
        return float("nan"), {}
    return float(np.mean(list(parts.values()))), parts


def run_once(
    chartist: float,
    fund: float,
    offset_hi: float,
    seed: int,
    *,
    fu_deadband: float | None = None,
    fu_aggressiveness: float | None = None,
) -> dict:
    zero = 1.0 - chartist - fund
    mix = {"zero_intel": zero, "fundamentalist": fund, "chartist": chartist}
    pop = Population.from_shares(N_AGENTS, mix)
    kw: dict = {"zi_offset_range": (0.0002, offset_hi)}
    if fu_deadband is not None:
        kw["fu_deadband"] = fu_deadband
    if fu_aggressiveness is not None:
        kw["fu_aggressiveness"] = fu_aggressiveness
    cfg = SimConfig(
        seed=seed,
        n_ticks=N_TICKS,
        population=pop,
        **kw,
    )
    m = Market(cfg)
    log = m.run()
    metrics = analyze(log.mid, "sim", vol_window=24).flat()
    ok, _ = m.health_check()
    return {
        "metrics": metrics,
        "health_ok": ok,
        "n_trades": len(log.trades),
        "sigma_bp": metrics["sigma_bp"],
    }


def main() -> dict:
    banner("阶段4  参数校准与敏感性分析（含样本外检验）")
    t0 = time.time()

    tg = real_targets()
    print(f"  真实 BTC 1h：{tg['n']} 根，训练段前 {TRAIN_FRAC:.0%}（{tg['cut_index']} 根），"
          f"测试段后 {1 - TRAIN_FRAC:.0%}（{tg['n'] - tg['cut_index']} 根）")
    print(
        f"    {'指标':<18}{'训练段':>12}{'测试段':>12}{'差异':>10}"
    )
    for k, name in (
        ("sigma_bp", "σ (bp)"),
        ("excess_kurtosis", "超额峰度"),
        ("acf_abs_lag1", "|r| ACF(1)"),
        ("acf_abs_mean_1_10", "|r| ACF(1-10)"),
        ("acf_ret_lag1", "r ACF(1)"),
        ("vr5", "VR(5)"),
        ("hill_alpha_left", "Hill α"),
        ("n_sig_lags_abs", "显著滞后数"),
    ):
        a, b = tg["train"].get(k), tg["test"].get(k)
        if a is None or b is None:
            continue
        rel = abs(b - a) / abs(a) if a else float("nan")
        print(f"    {name:<18}{a:>12.4g}{b:>12.4g}{rel:>9.1%}")

    # ---------- 网格 ----------
    combos = [
        (ch, fu, oh)
        for ch, fu, oh in itertools.product(CHARTIST_SHARES, FUND_SHARES, OFFSET_HI)
        if ch + fu <= 0.85
    ]
    print(f"\n  网格：{len(combos)} 组参数 × {len(SEEDS)} 种子 = {len(combos) * len(SEEDS)} 次运行")
    print(f"    {'零智能':>7}{'基本面':>8}{'图表':>7}{'报价上界':>10}"
          f"{'训练分':>9}{'测试分':>9}{'峰度':>8}{'|r|ACF1':>9}{'σ/tick':>8}{'VR5':>7}")

    results = []
    for i, (ch, fu, oh) in enumerate(combos, 1):
        runs = [run_once(ch, fu, oh, sd) for sd in SEEDS]
        agg = {}
        for k in runs[0]["metrics"]:
            vals = [r["metrics"].get(k) for r in runs]
            vals = [v for v in vals if isinstance(v, (int, float)) and np.isfinite(v)]
            agg[k] = float(np.mean(vals)) if vals else float("nan")
        s_tr, parts_tr = score(agg, tg["train"])
        s_te, parts_te = score(agg, tg["test"])
        results.append(
            {
                "zero_intel": round(1 - ch - fu, 3),
                "fundamentalist": fu,
                "chartist": ch,
                "offset_hi": oh,
                "train_score": s_tr,
                "test_score": s_te,
                "parts_train": parts_tr,
                "metrics": agg,
                "health_ok": all(r["health_ok"] for r in runs),
            }
        )
        print(
            f"    {1 - ch - fu:>7.2f}{fu:>8.2f}{ch:>7.2f}{oh * 1e4:>9.1f}bp"
            f"{s_tr:>9.3f}{s_te:>9.3f}{agg['excess_kurtosis']:>8.2f}"
            f"{agg['acf_abs_lag1']:>9.3f}{agg['sigma_bp']:>8.1f}{agg.get('vr5', np.nan):>7.2f}"
        )

    results.sort(key=lambda r: r["train_score"])
    best = results[0]

    # ---------- 样本外检验 ----------
    print("\n  样本外检验（训练段最优的 5 组，看测试段表现）")
    print(f"    {'零/基/图':>14}{'报价上界':>10}{'训练分':>9}{'测试分':>9}{'测试/训练':>11}")
    ratios = []
    for r in results[:5]:
        ratio = r["test_score"] / r["train_score"] if r["train_score"] > 0 else np.nan
        ratios.append(ratio)
        print(
            f"    {r['zero_intel']:.2f}/{r['fundamentalist']:.2f}/{r['chartist']:.2f}"
            f"{r['offset_hi'] * 1e4:>9.1f}bp{r['train_score']:>9.3f}"
            f"{r['test_score']:>9.3f}{ratio:>11.2f}"
        )
    med_ratio = float(np.nanmedian(ratios))
    print(
        f"\n    → 测试分/训练分 中位数 = {med_ratio:.2f}"
        f"（{'≈1，参数抓到的是稳定机制' if 0.7 <= med_ratio <= 1.6 else '偏离 1 较多，存在过拟合风险' if med_ratio > 1.6 else '测试段反而更好，说明差异主要来自真实数据的段间差异'}）"
    )

    # ---------- 机制敏感性：哪几个旋钮真正有用 ----------
    print("\n  敏感性：改变单一参数时，距离分数的变化幅度（越大 = 该参数越关键）")
    sens = {}
    for key, label in (
        ("chartist", "图表派占比"),
        ("fundamentalist", "基本面派占比"),
        ("offset_hi", "报价幅度上界"),
    ):
        if key == "offset_hi":
            import math

            groups: dict[float, list[float]] = {}
            for r in results:
                k = round(math.log10(r[key]), 6)
                groups.setdefault(k, []).append(r["train_score"])
        else:
            groups = {}
            for r in results:
                groups.setdefault(r[key], []).append(r["train_score"])
        means = {k: float(np.mean(v)) for k, v in groups.items()}
        spread = max(means.values()) - min(means.values())
        sens[key] = {"means": {str(k): v for k, v in means.items()}, "spread": spread}

        def _fmt(k: float) -> str:
            # offset_hi 的分组键是 log10，打印时要还原成 bp，否则读者看到的是 -2.74 这种数
            return f"{10 ** k * 1e4:.1f}bp" if key == "offset_hi" else f"{k:g}"

        print(f"    {label:<14} 分数跨度 {spread:.3f}  " + "  ".join(
            f"{_fmt(k)}→{v:.3f}" for k, v in sorted(means.items(), key=lambda kv: str(kv[0]))
        ))

    # ---------- 4b 对症网格：修正"过度均值回归" ----------
    print("\n  4b 对症网格（针对阶段2 诊断出的「过度均值回归」）")
    print(
        f"    基准配置：零/基/图 = {best['zero_intel']:.2f}/{best['fundamentalist']:.2f}/"
        f"{best['chartist']:.2f}，报价上界 {best['offset_hi'] * 1e4:.1f}bp"
    )
    print(
        f"    真实 BTC 目标：r ACF(1) = {tg['full'].get('acf_ret_lag1', float('nan')):+.4f}，"
        f"VR(5) = {tg['full'].get('vr5', float('nan')):.3f}"
        f"（VR(5) 越接近 1 越「无线性可预测性」）"
    )
    print(
        f"    {'死区':>8}{'激进度':>9}{'训练分':>9}{'测试分':>9}"
        f"{'峰度':>8}{'|r|ACF1':>9}{'r ACF1':>10}{'VR5':>8}{'σ(bp)':>8}"
    )
    focus = []
    for db in FOCUS_DEADBAND:
        for ag in FOCUS_AGGRESS:
            runs = [
                run_once(best["chartist"], best["fundamentalist"], best["offset_hi"], sd,
                         fu_deadband=db, fu_aggressiveness=ag)
                for sd in SEEDS
            ]
            agg = {}
            for k in runs[0]["metrics"]:
                vals = [
                    r["metrics"].get(k) for r in runs
                    if isinstance(r["metrics"].get(k), (int, float))
                    and np.isfinite(r["metrics"].get(k))
                ]
                agg[k] = float(np.mean(vals)) if vals else float("nan")
            s_tr, parts_tr = score(agg, tg["train"])
            s_te, _ = score(agg, tg["test"])
            row = {
                "fu_deadband": db,
                "fu_aggressiveness": ag,
                "train_score": s_tr,
                "test_score": s_te,
                "parts_train": parts_tr,
                "metrics": agg,
                "health_ok": all(r["health_ok"] for r in runs),
            }
            focus.append(row)
            print(
                f"    {db:>8.3f}{ag:>9.4f}{s_tr:>9.3f}{s_te:>9.3f}"
                f"{agg.get('excess_kurtosis', np.nan):>8.2f}"
                f"{agg.get('acf_abs_lag1', np.nan):>9.3f}"
                f"{agg.get('acf_ret_lag1', np.nan):>10.4f}"
                f"{agg.get('vr5', np.nan):>8.3f}"
                f"{agg.get('sigma_bp', np.nan):>8.1f}"
            )

    focus.sort(key=lambda r: r["train_score"])
    fbest = focus[0]
    base_m = best["metrics"]
    print(f"\n    → 主网格最优的 r ACF(1) = {base_m.get('acf_ret_lag1', float('nan')):+.4f}、"
          f"VR(5) = {base_m.get('vr5', float('nan')):.3f}")
    print(f"    → 对症网格最优的 r ACF(1) = {fbest['metrics'].get('acf_ret_lag1', float('nan')):+.4f}、"
          f"VR(5) = {fbest['metrics'].get('vr5', float('nan')):.3f}"
          f"（死区 {fbest['fu_deadband']:.3f}、激进度 {fbest['fu_aggressiveness']:.4f}）")
    d_r = abs(fbest["metrics"].get("acf_ret_lag1", float("nan"))
              - tg["full"].get("acf_ret_lag1", float("nan")))
    d_b = abs(base_m.get("acf_ret_lag1", float("nan"))
              - tg["full"].get("acf_ret_lag1", float("nan")))
    print(
        f"    → 与真实 r ACF(1) 的距离：主网格 {d_b:.4f} → 对症网格 {d_r:.4f}"
        f"（{'改善' if d_r < d_b else '⚠️ 未改善'}）"
    )
    print(
        "       注意：即使改善，也**不能**说'找到了正确的参数'。"
        "这里只是确认了「基本面派死区/激进度」确实是控制均值回归强弱的旋钮——"
        "把这两轴扫到最优再报成绩，本身就带一点点样本内挑选。结论以机制归因为主。"
    )

    # ---------- 出图 ----------
    _plot_landscape(results, FIG / "stage4_landscape.png")
    _plot_train_test(results, FIG / "stage4_train_test.png")
    _plot_sensitivity(results, FIG / "stage4_sensitivity.png")
    _plot_focus(focus, tg, FIG / "stage4_focus_meanreversion.png")
    print(f"\n  图已输出到 {FIG}")

    print("\n  阶段4 结论：")
    print(f"    · 训练段最优：零智能 {best['zero_intel']:.2f} / 基本面 {best['fundamentalist']:.2f} / "
          f"图表 {best['chartist']:.2f}，报价上界 {best['offset_hi'] * 1e4:.1f}bp，训练分 {best['train_score']:.3f}")
    print(f"    · 该组在测试段得分 {best['test_score']:.3f}"
          f"（比值 {best['test_score'] / best['train_score']:.2f}）")
    print("    · ⚠️ 这一步的产出是**机制理解**（哪些旋钮控制肥尾与波动率聚集），")
    print("       不是一套可外推的行情生成器。真实市场的统计特征本身随样本波动，")
    print("       把全样本当目标去挑参数，本质上是另一种过拟合。")

    out = {
        "stage": 4,
        "config": {
            "n_agents": N_AGENTS, "n_ticks": N_TICKS, "seeds": SEEDS,
            "grid": {
                "chartist_shares": CHARTIST_SHARES,
                "fund_shares": FUND_SHARES,
                "offset_hi": OFFSET_HI,
                "train_frac": TRAIN_FRAC,
            },
            "focus_grid": {
                "fu_deadband": FOCUS_DEADBAND,
                "fu_aggressiveness": FOCUS_AGGRESS,
                "reason": "针对阶段2诊断出的过度均值回归（r ACF(1) 偏离、VR(5) 偏低）",
            },
        },
        "real_targets": tg,
        "results": results,
        "best": best,
        "focus_results": focus,
        "focus_best": fbest,
        "out_of_sample_ratio_median": med_ratio,
        "sensitivity": sens,
        "elapsed_sec": time.time() - t0,
    }
    save_json(out, "stage4_metrics.json")
    print(f"\n  总用时 {time.time() - t0:.0f}s")
    return out


# ----------------------------------------------------------------------
def _plot_focus(focus: list[dict], tg: dict, path) -> None:
    """4b 对症网格：基本面派死区/激进度 对"过度均值回归"的影响。"""
    import matplotlib.pyplot as plt

    from tw.viz import C_ACCENT, C_GOOD, C_SIM, MUTED

    dbs = sorted({r["fu_deadband"] for r in focus})
    ags = sorted({r["fu_aggressiveness"] for r in focus}, reverse=True)
    true_r = tg["full"].get("acf_ret_lag1", float("nan"))
    true_vr = tg["full"].get("vr5", float("nan"))

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.4))
    for ax, key, tval, ylabel, title in (
        (axes[0], "acf_ret_lag1", true_r, "r ACF(1)", "① 收益率自相关（真实 ≈ 0，不应有线性可预测性）"),
        (axes[1], "vr5", true_vr, "VR(5)", "② 方差比（真实 ≈ 1，不应有系统性均值回归）"),
    ):
        for ag in ags:
            xs, ys = [], []
            for db in dbs:
                for r in focus:
                    if r["fu_deadband"] == db and r["fu_aggressiveness"] == ag:
                        xs.append(db)
                        ys.append(r["metrics"].get(key, float("nan")))
            ax.plot(xs, ys, "o-", lw=1.5, ms=6, label=f"激进度 {ag:g}")
        ax.axhline(tval, color=C_ACCENT, lw=1.3, ls="--", label=f"真实 BTC = {tval:+.3f}")
        ax.axhline(0 if key == "acf_ret_lag1" else 1.0, color=MUTED, lw=0.9, ls=":")
        ax.set_xscale("log")
        ax.set_xticks(dbs)
        ax.set_xticklabels([f"{v:g}" for v in dbs])
        ax.set_xlabel("基本面派无套利死区（对数刻度）")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10.5)
        ax.legend(fontsize=8.5)
        ax.grid(True, alpha=0.15)

    fig.suptitle("对症网格：均值回归强度由哪个旋钮控制？", fontsize=12)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ----------------------------------------------------------------------
def _plot_landscape(results: list[dict], path) -> None:
    import matplotlib.pyplot as plt

    from tw.viz import C_ACCENT, C_GOOD, C_REAL, C_SIM, FG, MUTED

    ch = np.array([r["chartist"] for r in results])
    fu = np.array([r["fundamentalist"] for r in results])
    for key, name, scl in (
        ("train_score", "距离分数（越小越像真实）", 1.0),
        ("excess_kurtosis", "超额峰度", 1.0),
        ("acf_abs_lag1", "|r| ACF(1)", 1.0),
    ):
        if key not in results[0] and key != "train_score":
            continue

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    specs = [
        ("excess_kurtosis", "超额峰度", None),
        ("acf_abs_lag1", "|r| ACF(lag1)", None),
        ("train_score", "距离分数（越低越像）", None),
    ]
    tg = None
    for ax, (key, label, _) in zip(axes, specs):
        vals = np.array([r["metrics"].get(key, r.get(key, np.nan)) for r in results], dtype=float)
        c = np.array([r["chartist"] for r in results])
        sc = ax.scatter(c, vals, c=fu, cmap="viridis", s=42, alpha=0.9)
        ax.set_xlabel("图表派占比")
        ax.set_ylabel(label)
        ax.set_title(label)
        fig.colorbar(sc, ax=ax, label="基本面派占比", fraction=0.045)
    fig.suptitle("参数空间的地形：肥尾与波动率聚集由什么控制", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_train_test(results: list[dict], path) -> None:
    import matplotlib.pyplot as plt

    from tw.viz import C_ACCENT, C_GOOD, C_SIM, FG, MUTED

    tr = np.array([r["train_score"] for r in results], dtype=float)
    te = np.array([r["test_score"] for r in results], dtype=float)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.scatter(tr, te, s=48, color=C_SIM, alpha=0.85, edgecolor="none")
    lim = [0, max(tr.max(), te.max()) * 1.08]
    ax.plot(lim, lim, ls="--", color=FG, lw=1.0, label="测试分 = 训练分")
    ax.plot(lim, [v * 1.5 for v in lim], ls=":", color=C_ACCENT, lw=1.0, label="1.5× 警戒线")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("训练段距离分数")
    ax.set_ylabel("测试段距离分数")
    ax.set_title("样本外检验：点落在对角线上 = 参数抓到的是稳定机制")
    ax.legend(fontsize=9)
    best = min(results, key=lambda r: r["train_score"])
    ax.annotate(
        "训练段最优",
        xy=(best["train_score"], best["test_score"]),
        xytext=(best["train_score"] + lim[1] * 0.06, best["test_score"] - lim[1] * 0.12),
        color=C_GOOD, fontsize=9,
        arrowprops=dict(arrowstyle="->", color=C_GOOD, lw=1.0),
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_sensitivity(results: list[dict], path) -> None:
    import matplotlib.pyplot as plt

    from tw.viz import C_GOOD, C_REAL, C_SIM, MUTED

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
    for ax, (key, label) in zip(
        axes,
        [("chartist", "图表派占比"), ("fundamentalist", "基本面派占比"), ("offset_hi", "报价幅度上界")],
    ):
        groups: dict[float, list[float]] = {}
        for r in results:
            groups.setdefault(r[key], []).append(r["train_score"])
        xs = sorted(groups)
        ys = [float(np.mean(groups[x])) for x in xs]
        es = [float(np.std(groups[x])) for x in xs]
        ax.errorbar(xs, ys, yerr=es, marker="o", color=C_REAL, lw=1.5, capsize=3)
        ax.set_xlabel(label)
        ax.set_ylabel("训练段距离分数")
        ax.set_title(f"对 {label} 的敏感性")
    fig.suptitle("哪个旋钮真正控制「像不像真实市场」", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
