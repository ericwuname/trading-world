"""阶段10：统一校准框架（二期指导书 §7）。

目标
----
用矩量模拟法（MSM）替换阶段4 的网格搜索，并对全部参数给出**不确定性区间**
而不是点估计。

实验编号
--------
E10.1 矩量标准化与距离   对比「裸欧氏距离」与「标准化距离」挑出的参数是否不同
E10.2 MSM 校准           3 个关键参数联合搜索（其余固定为已标定值）
E10.3 块自助法参数区间   对真实数据重采样 → 每份重跑校准 → 参数分布
E10.4 与阶段4 网格点对照 阶段4 的最优点是否落在自助法置信区间内
E10.5 MSM vs 网格搜索    同一距离口径下比一比（**允许「网格够用」这个结论**）

⭐ 三条必须先说清楚的方法论
==========================

**① 参数量必须压住。** 指导书 §3.7 明确警告过「参数量爆炸风险」。
   本阶段只搜 **3 个**参数（报价幅度上界、图表派占比、基本面派死区），
   其余（含阶段5/6 的传导系数、Hawkes 的 λ̄）一律**固定**。
   一起搜会显著增加过拟合风险，而「校准出来的参数」就失去意义。

**② 每次评估都必须重估标准化尺度。**
   参数的噪声水平随参数变化，用一份固定的跨种子标准差去标准化所有候选参数，
   会让「噪声大的区域」在距离上被系统性低估，于是最优解被推向高噪声区。

**③ 少数几个自助样本估不出 2.5% 分位点。** 指导书默认 ``n_bootstrap=50``，
   在本项目的模拟成本下是几千场模拟（小时级）。本脚本把 n_boot 压到 6，
   并**如实报告**「这个区间的分位点不可靠，只能读相对宽度」。
   假装 6 次和 50 次一样，是最容易犯的自欺。

用法::

    python scripts/run_stage10.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import FIG, banner, load_json, real_series, save_json  # noqa: E402
from tw import Market, Population, SimConfig  # noqa: E402
from tw.analyzer import analyze  # noqa: E402
from tw.calibration.objective import (  # noqa: E402
    BOOTSTRAP_BLOCK,
    DEFAULT_MOMENTS,
    SearchSpace,
    bootstrap_real_moments,
    cross_seed_scales,
    parameter_uncertainty,
    standardized_distance,
)
from tw.calibration.optimizer import (  # noqa: E402
    bootstrap_calibrate,
    make_optimizer,
    times_ci_covers,
)

SEED0 = 20260917
BOOT_SEEDS = [SEED0, SEED0 + 7]
N_AGENTS = 200
WARMUP = 1_500
OBS = 3_500
N_TICKS = WARMUP + OBS
N_BOOT = 6            # 见模块文档 ③
N_ITER = 12
N_REFINE = 2

#: 参与搜索的 3 个参数（其余固定）。
SPACE = SearchSpace(
    bounds={
        "offset_hi": (6e-4, 3.6e-3),      # zi_offset_range 的上界
        "chartist": (0.12, 0.40),
        "fu_deadband": (5e-4, 3e-2),
    },
    log={"offset_hi": True, "fu_deadband": True},
)
#: 固定参数：阶段4/5/6 已标定过的那些，**不再一起搜**（模块文档 ①）
FIXED = {"zi_p_buy": 0.50}


def simulate(params: dict) -> list[dict]:
    """``params -> [每个种子一份矩]``。

    主体构成：零智能 30% 固定（与阶段4 的网格口径一致），
    图表派由参数给定，剩下的给基本面派。
    """
    ch = float(params["chartist"])
    fu = max(0.05, 1.0 - 0.30 - ch)
    mix = {"zero_intel": 0.30, "fundamentalist": fu, "chartist": ch}
    out = []
    for sd in BOOT_SEEDS:
        cfg = SimConfig(
            seed=sd, n_ticks=N_TICKS,
            population=Population.from_shares(N_AGENTS, mix),
            zi_offset_range=(2e-4, float(params["offset_hi"])),
            fu_deadband=float(params["fu_deadband"]),
            zi_p_buy=float(params.get("zi_p_buy", 0.50)),
        )
        m = Market(cfg)
        m.run(N_TICKS)
        obs = np.asarray(m.log.mid[WARMUP: m.tick], dtype=float)
        out.append(analyze(obs, "sim", vol_window=24).flat())
    return out


def _raw_distance(sim: dict, real: dict) -> float:
    """裸欧氏距离（不标准化）—— E10.1 的对照口径。"""
    tot = 0.0
    for k in DEFAULT_MOMENTS:
        a, b = sim.get(k), real.get(k)
        if a is None or b is None:
            continue
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        tot += (float(a) - float(b)) ** 2
    return float(tot)


def evaluate_params(params: dict, real_stats: dict) -> tuple[float, dict]:
    """单点评估（用于 E10.1 的裸 vs 标准化对比）。"""
    samples = simulate(params)
    scales = cross_seed_scales(samples, DEFAULT_MOMENTS)
    raw, std, parts = [], [], {}
    for item in samples:
        raw.append(_raw_distance(item, real_stats))
        d_std, parts = standardized_distance(item, real_stats, scales=scales)
        std.append(d_std)
    return float(np.mean(std)), {"raw": float(np.mean(raw)), "scales": scales,
                                 "parts": parts}


# ----------------------------------------------------------------------
def main() -> dict:
    banner("阶段10  统一校准框架")
    t0 = time.time()
    out: dict = {"stage": 10, "config": {
        "n_agents": N_AGENTS, "warmup": WARMUP, "obs": OBS,
        "boot_seeds": BOOT_SEEDS, "n_boot": N_BOOT,
        "n_iter": N_ITER, "n_refine": N_REFINE,
        "space": SPACE.bounds, "log_space": SPACE.log, "fixed": FIXED,
        "moments": list(DEFAULT_MOMENTS), "bootstrap_block": BOOTSTRAP_BLOCK}}

    # ---------- 真实矩 ----------
    s = real_series("BTCUSDT_1h")
    real_full = analyze(s.close, "BTC 1h", vol_window=24).flat()
    print("\n真实矩（BTC 1h 全样本，口径与阶段4 一致）：")
    for k in DEFAULT_MOMENTS:
        print(f"    {k:<20}{real_full.get(k, float('nan')):>14.6g}")
    out["real_moments"] = {k: real_full.get(k) for k in DEFAULT_MOMENTS}

    # ==================================================================
    # E10.1  裸距离 vs 标准化距离
    # ==================================================================
    print("\n【E10.1】裸欧氏距离 vs 标准化距离")
    print("  同一组候选参数，两种口径下挑出的最优点是否相同？")
    print(f"    {'候选':<26}{'裸距离':>14}{'标准化距离':>14}")
    cands = [
        {"offset_hi": 1.2e-3, "chartist": 0.22, "fu_deadband": 2e-3},
        {"offset_hi": 3.0e-3, "chartist": 0.22, "fu_deadband": 2e-3},
        {"offset_hi": 1.2e-3, "chartist": 0.38, "fu_deadband": 2e-3},
        {"offset_hi": 1.2e-3, "chartist": 0.22, "fu_deadband": 2.5e-2},
    ]
    e101 = []
    for c in cands:
        d_std, diag = evaluate_params(c, real_full)
        e101.append({"offset_hi": c["offset_hi"], "chartist": c["chartist"],
                     "fu_deadband": c["fu_deadband"],
                     "std_distance": d_std, "raw_distance": diag["raw"],
                     "scales": diag["scales"], "parts": diag["parts"]})
        tag = (f"off={c['offset_hi'] * 1e4:.1f}bp ch={c['chartist']:.2f} "
               f"db={c['fu_deadband']:.0e}")
        print(f"    {tag:<26}{diag['raw']:>14.4g}{d_std:>14.4g}")
    best_raw = min(e101, key=lambda r: r["raw_distance"])
    best_std = min(e101, key=lambda r: r["std_distance"])
    same = (best_raw["offset_hi"] == best_std["offset_hi"]
            and best_raw["chartist"] == best_std["chartist"]
            and best_raw["fu_deadband"] == best_std["fu_deadband"])
    print(f"  裸距离最优：off={best_raw['offset_hi'] * 1e4:.1f}bp "
          f"ch={best_raw['chartist']:.2f} db={best_raw['fu_deadband']:.0e}")
    print(f"  标准化最优：off={best_std['offset_hi'] * 1e4:.1f}bp "
          f"ch={best_std['chartist']:.2f} db={best_std['fu_deadband']:.0e}")
    tail = "一致" if same else "**不一致**——标准化确实改变了挑选结果"
    print("  → 两者" + tail)
    out["e10_1"] = {"rows": e101, "same_choice": bool(same)}

    # ==================================================================
    # E10.2  MSM 校准
    # ==================================================================
    print(f"\n【E10.2】MSM 校准（{N_ITER} 次随机搜索 + {N_REFINE} 次局部细化）")
    print(f"  搜索参数：{list(SPACE.bounds)}；其余固定：{FIXED}")
    print(f"  每次评估 = {len(BOOT_SEEDS)} 种子 × {N_TICKS} tick × {N_AGENTS} 主体")
    optimizer = make_optimizer(simulate, SPACE, moments=DEFAULT_MOMENTS,
                               n_seeds=len(BOOT_SEEDS), n_iter=N_ITER,
                               n_refine=N_REFINE, seed=SEED0, fixed=FIXED)
    t1 = time.time()
    fit = optimizer(real_full)
    print(f"  用时 {time.time() - t1:.0f}s，共 {fit['n_evaluations']} 次评估")
    print("    最优参数：")
    for k, v in fit["best_params"].items():
        print(f"      {k:<14}{v:>14.6g}")
    print(f"    标准化距离 = {fit['best_distance']:.4f}")
    sc = fit["best_diagnostics"]["scales"]
    print("    各矩的标准化尺度（跨种子 sd）：")
    for k in DEFAULT_MOMENTS:
        print(f"      {k:<20}{sc.get(k, float('nan')):>14.6g}")
    parts = fit["best_diagnostics"]["parts"]
    print("    各矩的标准化偏差（|z| 越大越不像）：")
    for k, v in sorted(parts.items(), key=lambda kv: -abs(kv[1])):
        print(f"      {k:<20}{v:>+10.3f}  |z|={abs(v):.3f}")
    worst = max(parts.items(), key=lambda kv: abs(kv[1])) if parts else (None, 0.0)
    print(f"    → 最不像的矩是 **{worst[0]}**（|z|={abs(worst[1]):.3f}）——")
    print("       这一项就是本模型当前最大的失配，比看一个总分有用得多。")
    out["e10_2"] = fit

    # ==================================================================
    # E10.3  块自助法参数不确定性
    # ==================================================================
    print(f"\n【E10.3】块自助法参数不确定性（{N_BOOT} 份重采样 × 每份完整校准）")
    print(f"  ⚠️ 块长 {BOOTSTRAP_BLOCK}（一周）。**实测这个块长对 ACF 型矩严重低估**：")
    print("     acf_abs_lag1 只能恢复到真值的 0.9%（见 tw/calibration/objective.py）。")
    print("     所以下面的区间只能按相对宽度读，不能按绝对水平读。")
    print(f"  ⚠️ {N_BOOT} 个样本估不出 2.5% 分位点——区间的端点不可靠。")
    boot_moments = bootstrap_real_moments(
        s.close, n_boot=N_BOOT, block=BOOTSTRAP_BLOCK, seed=SEED0)
    t1 = time.time()
    boots = bootstrap_calibrate(optimizer, boot_moments, seed=SEED0)
    print(f"  用时 {time.time() - t1:.0f}s")
    keys = list(SPACE.bounds)
    unc = parameter_uncertainty(boots, keys)
    print(f"    {'参数':<14}{'均值':>12}{'标准差':>12}{'区间（端点不可靠）':>26}")
    for k in keys:
        u = unc[k]
        lo, hi = u["ci95"]
        print(f"    {k:<14}{u['mean']:>12.5g}{u['sd']:>12.4g}"
              f"{f'[{lo:.5g}, {hi:.5g}]':>26}")
    out["e10_3_bootstrap"] = {"runs": boots, "uncertainty": unc,
                              "block": BOOTSTRAP_BLOCK, "n_boot": N_BOOT}
    print("  ⚠️ 相对宽度（sd/|mean|）：")
    degenerate = []
    for k in keys:
        u = unc[k]
        rel = u["sd"] / abs(u["mean"]) if u["mean"] else float("nan")
        print(f"      {k:<14}{rel:>9.1%}")
        # ⭐ 退化检测：跨自助样本的参数**完全不动**（sd ≈ 0）。
        # 这不是"估计很精确"，而是**搜索分辨率不够**——
        # 每个自助样本都用同一套随机搜索点（同一个 seed），
        # 而 14 个评估点的网格太粗，6 个自助样本全都落在同一个格点上。
        # 于是"参数不确定性 = 0"完全是优化器的性质，与数据的信息量无关。
        # 实测踩到过：三个参数的标准差是 2.4e-19 / 3.0e-17 / 0。
        if u["sd"] == 0.0 or (abs(u["mean"]) > 0 and u["sd"] / abs(u["mean"]) < 1e-6):
            degenerate.append(k)
    if degenerate:
        print(f"\n    ⚠️⚠️ **参数不确定性估计已退化**：{degenerate} 的跨样本标准差 ≈ 0。")
        print("        这不是「估计精确」，而是**搜索分辨率不够**：")
        print(f"        每个自助样本都用同一套 {fit['n_evaluations']} 个随机搜索点，")
        print("        网格太粗 ⇒ 6 个样本全落在同一个格点上 ⇒ 区间宽度恒为 0。")
        print("        **E10.3 的输出因此不是一个有效的置信区间**，")
        print("        E10.4 的「阶段4 落在区间外」也随之失去意义（区间是一个点）。")
        print("        要修：把 n_iter 提高 1~2 个数量级，或改用连续优化器。")
    out["e10_3_degenerate"] = degenerate

    # ==================================================================
    # E10.4  与阶段4 网格最优点的对照
    # ==================================================================
    print("\n【E10.4】阶段4 的网格最优点是否落在置信区间内")
    s4 = load_json("stage4_metrics.json") or {}
    b4 = (s4.get("best") or {})
    if b4:
        p4 = {"offset_hi": float(b4.get("offset_hi", float("nan"))),
              "chartist": float(b4.get("chartist", float("nan"))),
              "fu_deadband": 2e-3}
        print(f"    阶段4 最优点：off={p4['offset_hi'] * 1e4:.2f}bp、"
              f"ch={p4['chartist']:.3f}")
        print("    （fu_deadband 阶段4 没搜过，用 2e-3 作代表值）")
        clipped = SPACE.clip(p4)
        d4, _ = evaluate_params(clipped, real_full)
        print(f"    用本阶段的标准化距离重评阶段4 最优点：{d4:.4f}"
              f"（MSM 最优 {fit['best_distance']:.4f}）")
        inside = {k: times_ci_covers(tuple(unc[k]["ci95"]), p4[k]) for k in keys}
        for k in keys:
            lo, hi = unc[k]["ci95"]
            mark = "✅ 在区间内" if inside[k] else "❌ 在区间外"
            print(f"      {k:<14}阶段4={p4[k]:.5g}  区间=[{lo:.5g}, {hi:.5g}]  {mark}")
        n_in = int(sum(inside.values()))
        print(f"    → {n_in}/{len(keys)} 个参数落在区间内")
        out["e10_4"] = {"stage4_params": p4, "stage4_distance": d4,
                        "msm_distance": fit["best_distance"],
                        "inside_ci": inside, "n_inside": n_in}
    else:
        print("    ⚠️ 找不到 stage4_metrics.json，跳过")
        out["e10_4"] = None

    # ==================================================================
    # E10.5  MSM vs 网格搜索
    # ==================================================================
    print("\n【E10.5】MSM 是否真的比网格搜索更好？")
    print("  ⚠️ 这一步允许「网格够用」这个结论——不允许为了显得新框架有用而挑口径。")
    d4_val = (out["e10_4"]["stage4_distance"] if out.get("e10_4")
              else float("nan"))
    print(f"    {'方法':<22}{'评估次数':>10}{'标准化距离':>14}")
    print(f"    {'阶段4 最优点（重评）':<22}{'—':>10}{d4_val:>14.4f}")
    print(f"    {'MSM（本阶段）':<22}{fit['n_evaluations']:>10}"
          f"{fit['best_distance']:>14.4f}")
    if np.isfinite(d4_val):
        gap = (d4_val - fit["best_distance"]) / max(abs(d4_val), 1e-12)
        print(f"    → MSM 的标准化距离比阶段4 最优点低 {gap:.1%}")
        print("      ⚠️ 这个对比是不公平的：阶段4 是 36 组粗网格里选一个，")
        print(f"         而 MSM 有 {fit['n_evaluations']} 次评估 + 局部细化。")
        print("         公平的问法是「同样的评估预算下谁更好」，那需要把网格也细到")
        print("         同样的点数——本脚本没有做，所以这里的差值")
        print("         只能读成「MSM 在其搜索空间里找到了更优点」，")
        print("         不能读成「MSM 优于网格搜索」。")
        verdict = ("MSM 找到了更优点（但对比不公平，见上）"
                   if gap > 0.02 else "两者相当，网格搜索在这个参数量级下已经够用")
    else:
        gap = float("nan")
        verdict = "无法对比（缺阶段4 结果）"
    print(f"    → 结论：{verdict}")
    out["e10_5"] = {
        "msm_distance": fit["best_distance"],
        "stage4_distance": float(d4_val) if np.isfinite(d4_val) else None,
        "gap": float(gap) if np.isfinite(gap) else None,
        "verdict": verdict}

    # ==================================================================
    _plot_distance_comparison(e101, FIG / "stage10_distance.png")
    _plot_uncertainty(unc, out.get("e10_4"), FIG / "stage10_uncertainty.png")
    print(f"\n  图已输出到 {FIG}")

    # ==================================================================
    print("\n" + "=" * 74)
    print("阶段10 「诚实边界」小结")
    print("=" * 74)
    n_inside = out["e10_4"]["n_inside"] if out.get("e10_4") else "—"
    lines = [
        "【做到了什么量级】",
        f"  · 标准化距离替换裸欧氏距离：{len(DEFAULT_MOMENTS)} 个矩各有自己的尺度",
        f"  · MSM 校准 {len(keys)} 个参数，{fit['n_evaluations']} 次评估，"
        f"标准化距离 {fit['best_distance']:.4f}",
        f"  · 块自助法（{N_BOOT} 份）× 每份完整校准 → 参数分布已产出",
        f"  · 与阶段4 最优点对照：{n_inside}/{len(keys)} 个参数落在区间内",
        f"  · MSM vs 网格：{verdict}",
        "",
        "【做不到什么】",
        "  · 块自助法的块长不充分（本阶段最重要的负面结论）。",
        f"    实测 block={BOOTSTRAP_BLOCK} 时 acf_abs_lag1 只能恢复到真值的 0.9%；",
        "    块长要到 8000（半年）才恢复到 69%。而波动率聚集的依赖长度就是那么长。",
        "    所以本阶段给出的参数区间是围绕一个被系统性低估的矩构造的——",
        "    它看起来完全正常，实际中心是偏的。这个缺陷无法靠增加自助次数修好。",
        f"  · {N_BOOT} 个自助样本估不出 2.5% 分位点。指导书默认 50 次，"
        "在本项目的模拟成本下是小时级。",
        "    所以区间端点不可靠，只能读相对宽度。这是被成本削过的，",
        "    不是「方法上可以接受」。",
        "  · 只搜了 3 个参数。指导书要求全部参数联合校准，本阶段固定了",
        "    主体数、零智能占比、做市商参数、阶段5/6 的全部传导系数。",
        "    理由是 §3.7 自己警告过的参数量爆炸风险。代价是参数之间的相关性",
        "    完全没有被刻画——固定值错了，搜出来的另 3 个参数会跟着补偿它。",
        "  · 没有做参数间的联合置信域，只报了一维边际区间。",
        "    真实校准里参数是强相关的（报价幅度与图表派占比高度互补），",
        "    一维区间会给出「两个参数都很确定」的错觉。",
        "  · MSM vs 网格的对比不公平（见 E10.5）。",
        "    正确做法是固定评估预算再比，本脚本没有做。",
        "",
        "【下个阶段依赖它的哪个假设】",
        "  · 阶段11 的综合场景需要一组「已校准」参数。本阶段的结论是：",
        "    参数不确定性没有被可靠量化，所以综合场景只能按点估计跑，",
        "    并在报告里注明这一点。",
        "",
        "【一条必须写进结论的方法论发现】",
        "  · 块自助法只对「均值型」矩有效，对「依赖型」矩（ACF、聚集）系统性低估。",
        "    这不是本项目的实现问题，是方法本身的边界：块长必须覆盖该矩的依赖长度，",
        "    而当依赖长度是样本量的一半时，块自助法已经没有可用空间了。",
        "    对这类矩，正确做法是参数化自助法（拟合一个能生成同类依赖的模型再重采样），",
        "    或直接报「不确定性未量化」。本阶段选择了后者并写明——",
        "    比给一个看起来很专业但中心是偏的区间要好。",
    ]
    for ln in lines:
        print(ln)
    out["honest_boundary"] = lines
    out["elapsed_sec"] = time.time() - t0
    save_json(out, "stage10_metrics.json")
    print(f"\n  总用时 {out['elapsed_sec']:.0f}s")
    return out


# ----------------------------------------------------------------------
def _plot_distance_comparison(e101: list[dict], path) -> None:
    import matplotlib.pyplot as plt

    from tw.viz import C_ACCENT, C_GOOD, C_SIM

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.4))
    ax = axes[0]
    labels = [f"c{i + 1}" for i in range(len(e101))]
    ax.bar(np.arange(len(e101)) - 0.2, [r["raw_distance"] for r in e101],
           width=0.38, color=C_SIM, label="裸欧氏距离")
    ax.bar(np.arange(len(e101)) + 0.2, [r["std_distance"] for r in e101],
           width=0.38, color=C_ACCENT, label="标准化距离")
    ax.set_xticks(np.arange(len(e101)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("距离（越小越像）")
    ax.set_title("① 同一组候选在两种口径下的距离\n（量级不可比，看排序）")
    ax.legend(fontsize=9)

    ax = axes[1]
    keys = list(e101[0]["parts"].keys())
    z = [abs(e101[0]["parts"][k]) for k in keys]
    order = np.argsort(z)[::-1]
    ax.barh([keys[i] for i in order], [z[i] for i in order], color=C_GOOD)
    ax.set_xlabel("|z|（标准化偏差）")
    ax.set_title("② 逐矩看失配：哪个矩最不像真实\n（比一个总分有用）")
    fig.suptitle("E10.1 标准化距离：让每个矩用自己的尺度说话", fontsize=12)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_uncertainty(unc: dict, e104, path) -> None:
    import matplotlib.pyplot as plt

    from tw.viz import C_ACCENT, C_REAL, C_SIM

    keys = list(unc.keys())
    fig, axes = plt.subplots(1, len(keys), figsize=(4.4 * len(keys), 4.2))
    if len(keys) == 1:
        axes = [axes]
    for ax, k in zip(axes, keys):
        u = unc[k]
        vals = u["values"]
        ax.hist(vals, bins=max(3, len(vals) // 2), color=C_SIM, alpha=0.85)
        ax.axvline(u["mean"], color=C_ACCENT, lw=2.0, label="自助均值")
        lo, hi = u["ci95"]
        ax.axvline(lo, color=C_REAL, ls="--", lw=1.3, label="区间端点（不可靠）")
        ax.axvline(hi, color=C_REAL, ls="--", lw=1.3)
        if e104:
            p4 = e104["stage4_params"].get(k)
            if p4 is not None:
                ax.axvline(p4, color="#e06c75", ls=":", lw=1.8, label="阶段4 网格点")
        ax.set_title(k)
        ax.legend(fontsize=8)
    n = len(next(iter(unc.values()))["values"])
    fig.suptitle(f"E10.3 参数分布（{n} 份自助重采样）—— 端点不可靠，请读宽度",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
