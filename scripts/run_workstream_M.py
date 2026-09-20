"""工作线M：H4 象限检验的功效分析 —— 先算清「值不值得投入」，再决定加不加种子。

任务书：`交易世界 · 回填标准配置与H4做实任务书.md` §4。

为什么必须先算
--------------
H4 的宏观形式成立（非对称下吃单突发期深度 −22.4%，对称下 −3.5%，差 6 倍），
但象限内的配对检验（``taker_only`` vs ``both_burst``）在 3 种子下 **p = 0.26**。
上一轮已经证明"盲目加资源"是有代价的（300 个种子才判定 0.5±0.1 那件事），
所以这次**先算需要多少，再决定要不要投入**。

⚠️ 与工作线L 的关键区别（任务书 §4.1）
------------------------------------
这里缺的是**每组的事件观测个数**，不是档位数。
L 线解决的是幂律拟合缺自由度；这里解决的是 t 检验缺样本量。
**两者的降噪机制不同，不能套用同一个经验值。**

⚠️ 前提核对
-----------
任务书建议用 ``statsmodels.stats.power.TTestIndPower``，但本环境**没有装**
（实测 ImportError）。改用 ``scripts/power_analysis_H4.py``：
基于 **scipy 非中心 t 分布的精确计算**（与 statsmodels 算法同源），
且已用三组教科书值验证过（d=0.5→64、d=0.8→26、d=0.2→394，**差 0**）。

用法::

    python scripts/run_workstream_M.py                 # EM.1 + EM.2（并自动选分支）
    python scripts/run_workstream_M.py --force-em4     # 强制走诚实报告分支
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import OUT, banner, ensure_scripts_on_path, load_json, save_json  # noqa: E402

RESULT = "workstream_M_metrics.json"

#: EM.2 的成本阈值 —— **事先写死**，不许事后根据结果调整（任务书 §4.5 明确要求）
COST_THRESHOLD = {
    "worth_it_max_seeds": 50,      # ≤ 50 个种子：值得做
    "marginal_max_seeds": 100,     # 50~100：边缘（可做，但要在报告里讲成本）
    "not_worth_min_seeds": 100,    # > 100：不值得，走 EM.4 诚实报告
}


def _safe_load(name: str) -> dict:
    p = OUT / name
    if not p.exists():
        return {}
    try:
        return load_json(name) or {}
    except Exception:  # noqa: BLE001
        return {}


# ======================================================================
def read_observed_from_E() -> dict:
    """从工作线E 的产物**现读**四象限观测值（不手抄任何数字）。

    返回 ``{diff, pooled_sd, n_taker_only, n_both_burst, ratio, ...}``。
    """
    d = _safe_load("workstream_E_metrics.json")
    cfg = ((d.get("ee1_sync") or {}).get("configs") or {}).get("symmetric_both")
    if not cfg:
        raise RuntimeError(
            "缺 out/workstream_E_metrics.json 里 symmetric_both 的四象限数据。"
            "先跑：python scripts/run_workstream_E.py --only EE1")
    q = (cfg.get("quadrants") or [{}])[0]
    to, bb = q.get("taker_only") or {}, q.get("both_burst") or {}
    neither = q.get("neither") or {}
    if not (to.get("n") and bb.get("n")):
        raise RuntimeError(f"四象限样本数为空：taker_only={to.get('n')}, "
                           f"both_burst={bb.get('n')}")
    n1, n2 = int(to["n"]), int(bb["n"])
    m1, m2 = float(to["mean"]), float(bb["mean"])
    s1, s2 = float(to.get("std") or 0.0), float(bb.get("std") or 0.0)
    # 合并标准差（两样本 t 检验口径）
    pooled = (((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2)
              / max(1, n1 + n2 - 2)) ** 0.5
    return {
        "source": "out/workstream_E_metrics.json :: ee1_sync.configs.symmetric_both",
        "n_taker_only": n1, "n_both_burst": n2,
        "mean_taker_only": m1, "mean_both_burst": m2,
        "mean_neither": (float(neither["mean"]) if neither.get("n") else None),
        "sd_taker_only": s1, "sd_both_burst": s2,
        "diff": m2 - m1, "pooled_sd": pooled,
        "ratio": (n2 / n1) if n1 else float("nan"),
        "n_seeds_used": len(cfg.get("sync") or []),
    }


def em1_power_analysis(obs: dict) -> dict:
    """EM.1：用观测到的效应量算「达到 80% 功效需要多少个事件观测点」。"""
    from power_analysis_H4 import estimate_required_n_for_quadrant_test
    r = estimate_required_n_for_quadrant_test(
        obs["diff"], obs["pooled_sd"], ratio=obs["ratio"])
    return {**r.as_dict(), "observed": obs}


def em2_cost_judgement(em1: dict, *, observed_n_seeds: int = 3) -> dict:
    """EM.2：把「需要多少事件点」换算成「需要多少种子 / 多长 tick」，并按**事先定死**的阈值判断。

    事件/种子的换算用的是当前实测：3 种子产出 ``n_taker_only`` 个 taker_only 点。
    """
    from power_analysis_H4 import estimate_seeds_needed
    obs = em1["observed"]
    per_seed = obs["n_taker_only"] / max(1, observed_n_seeds)
    conv = estimate_seeds_needed(em1["required_n_group1"], per_seed,
                                 observed_n_seeds=observed_n_seeds)
    need = conv["seeds_needed"]
    if need <= COST_THRESHOLD["worth_it_max_seeds"]:
        verdict = "值得做"
    elif need <= COST_THRESHOLD["marginal_max_seeds"]:
        verdict = "边缘"
    else:
        verdict = "不值得"
    # ⚠️ 把 ``conv`` 的 note（"两条路各自的成本结构"）**并进来**，不要覆盖它——
    #    第一版写成 `"note": f"换算依据：..."` 直接盖掉了 estimate_seeds_needed
    #    那段关于"加种子 vs 加长单次模拟"的说明，读者就再也看不到那条权衡了。
    #    这是被测试抓到的（``test_两条路的成本都被说明``）。
    conv_note = conv.get("note", "")
    return {**conv, "verdict": verdict, "thresholds": dict(COST_THRESHOLD),
            "observed_n_seeds": observed_n_seeds,
            "note": (f"换算依据：{observed_n_seeds} 种子产出 "
                     f"{obs['n_taker_only']} 个 taker_only 事件点 "
                     f"⇒ 每种子 {per_seed:.1f} 个。{conv_note}")}


def em4_honest_report(em1: dict, em2: dict) -> dict:
    """EM.4：判定不值得时的**诚实报告**分支（不强行加种子凑显著）。"""
    obs = em1["observed"]
    # 宏观效应量：以背景期（neither）为参照，报"突发期深度相对变化"
    back = obs.get("mean_neither")
    drop = ((obs["mean_taker_only"] / back - 1) * 100) if back else None
    return {
        "branch": "EM.4 诚实报告",
        "macro_effect": {
            "background_mean": back,
            "taker_only_mean": obs["mean_taker_only"],
            "relative_change_pct": drop,
            "note": "宏观形式：吃单方突发期深度相对背景期的变化（同配置内比较）",
        },
        "text": (
            f"H4 的**宏观形式**在当前证据下成立且效应量清晰："
            f"对称配置下吃单方突发期的局部深度 "
            f"{obs['mean_taker_only']:.2f}"
            + (f"（相对背景期 {back:.2f} 变化 {drop:+.1f}%）" if back else "")
            + f"；而两象限之间的差只有 {obs['diff']:+.2f}"
            f"（Cohen's d = {em1['effect_size_cohens_d']:.3f}）。"
            f"要以 80% 功效把这个差检出，需要每组 "
            f"{em1['required_n_group1']:.0f} 个事件观测点，"
            f"换算成种子约 **{em2['seeds_needed']:.0f} 个**（现用 3 个）——"
            f"远超事先定死的阈值（>{em2['thresholds']['not_worth_min_seeds']} 判为不值得）。"
            f"这与阶段3 判定 k 精确值需要约 300 个种子是**同一类问题**："
            f"微观机制的「方向」比「精确显著性」更容易确立。"),
        "recommendation": (
            "不建议为了把 p 压到 0.05 以下而堆种子。"
            "更有价值的做法是把「方向 + 效应量 + 区间」作为结论口径，"
            "而不是追求一个显著的 p 值——本轮的分辨力危机已经证明过："
            "一个看起来漂亮的点估计，分辨力不够时什么都不是。"),
    }


def em3_execute(em1: dict, em2: dict, *, max_seeds: int = 24) -> dict:
    """EM.3：按算出的规模加码，重跑四象限实验。

    ⚠️ 上界 ``max_seeds`` 是**执行护栏**：即使判定"值得做"，
    也不会真的去跑几百个种子而不打招呼。
    """
    ensure_scripts_on_path()
    import numpy as np
    import run_workstream_E as E
    from run_stage3 import SEED0

    n_seeds = int(min(max_seeds, max(6, em2["seeds_needed"] * 1.2)))
    seeds = [SEED0 + 7 * i for i in range(n_seeds)]
    print(f"  EM.3 执行：加码到 {n_seeds} 个种子"
          f"（EM.1 需求 {em2['seeds_needed']:.0f}，执行护栏 {max_seeds}）")
    lam = E._lambda_bar(seeds)
    runs = E.run_traced(1.0, seeds, base_lambda=lam)      # coupling=1.0 = 对称配置
    from tw.analyzer_asymmetry import (
        block_quadrant_contrast,
        burst_mask,
        measure_local_depth_during_taker_burst,
    )
    quads, contrasts = [], []
    for r in runs:
        t_mask = burst_mask(r["taker_keep"], E.BURST_PCT)
        m_mask = burst_mask(r["maker_keep"], E.BURST_PCT)
        quads.append(measure_local_depth_during_taker_burst(r["depth"], t_mask, m_mask))
        contrasts.append(block_quadrant_contrast(r["depth"], t_mask, m_mask, block=200))

    # 把跨运行的块级差当成样本做**跨运行**检验（比 tick 级诚实）
    diffs = [c.get("diff") for c in contrasts if c.get("diff") == c.get("diff")]
    out = {"n_seeds": n_seeds, "base_lambda": lam,
           "per_run_contrast": contrasts,
           "n_run_with_diff": len(diffs)}
    if len(diffs) >= 2:
        a = np.asarray(diffs, dtype=float)
        from scipy import stats as st
        t, p = st.ttest_1samp(a, 0.0)
        out.update({"cross_run_mean_diff": float(a.mean()),
                    "cross_run_sem": float(a.std(ddof=1) / np.sqrt(a.size)),
                    "cross_run_t": float(t), "cross_run_p": float(p),
                    "h4_direction": bool(a.mean() > 0)})
        # ⭐ 用**与本次检验同口径**的方差反算"还需要多少次运行"。
        #    EM.1 算的是"每组事件点"的需求，而这里用的是"跨运行的块级差"，
        #    自由度 = 运行数。两者口径不同，必须分开算，
        #    否则会拿一个口径的需求去指挥另一个口径的投入。
        from power_analysis_H4 import power_ttest_1samp, solve_n_onesample
        sd = float(a.std(ddof=1))
        if sd > 0:
            d_eff = abs(float(a.mean())) / sd
            out["own_scale_power"] = {
                "cross_run_effect_size_d": d_eff,
                "n_runs_for_80pct": solve_n_onesample(d_eff, power=0.8),
                "current_n_runs": int(a.size),
                "current_power_at_observed_d": float(
                    power_ttest_1samp(d_eff, a.size)),
                "note": ("与本次检验同口径（跨运行块级差，自由度 = 运行数）。"
                         "与 EM.1 的「每组事件点」口径不同，不可混用。"),
            }
    # 四象限合并统计
    agg: dict = {}
    for name in ("both_burst", "taker_only", "maker_only", "neither"):
        ns = [q.get(name, {}).get("n", 0) for q in quads]
        ms = [q.get(name, {}).get("mean") for q in quads
              if q.get(name, {}).get("mean") == q.get(name, {}).get("mean")]
        agg[name] = {"n_total": int(sum(ns)),
                     "mean": float(np.mean(ms)) if ms else float("nan")}
    out["aggregate"] = agg
    return out


# ======================================================================
def main() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-em4", action="store_true",
                    help="不管阈值判定如何，直接走诚实报告分支")
    ap.add_argument("--max-seeds", type=int, default=24,
                    help="EM.3 的执行护栏（默认 24）")
    ap.add_argument("--skip-em3", action="store_true",
                    help="只做功效分析（EM.1/EM.2），不执行加码——"
                         "把「算清楚」与「要不要投入」分成两步")
    args = ap.parse_args()

    banner("工作线M  功效分析 + H4 象限检验")
    t0 = time.time()
    out: dict = {"stage": "M"}

    obs = read_observed_from_E()
    out["observed"] = obs
    print(f"\n  【观测值（从 E 线产物现读）】")
    print(f"    taker_only n={obs['n_taker_only']} 均值={obs['mean_taker_only']:.2f} "
          f"sd={obs['sd_taker_only']:.2f}")
    print(f"    both_burst n={obs['n_both_burst']} 均值={obs['mean_both_burst']:.2f} "
          f"sd={obs['sd_both_burst']:.2f}")
    print(f"    差 = {obs['diff']:+.2f}   合并 sd = {obs['pooled_sd']:.2f}   "
          f"两组比 {obs['ratio']:.3f}")

    em1 = em1_power_analysis(obs)
    out["em1"] = em1
    print(f"\n  【EM.1 功效分析】")
    print(f"    Cohen's d = {em1['effect_size_cohens_d']:.4f}")
    print(f"    达到 80% 功效需要每组 "
          f"n1={em1['required_n_group1']:.0f} / n2={em1['required_n_group2']:.0f}"
          f"（合计 {em1['total_n']:.0f} 个事件观测点）")

    em2 = em2_cost_judgement(em1, observed_n_seeds=obs["n_seeds_used"])
    out["em2"] = em2
    print(f"\n  【EM.2 成本判断】")
    print(f"    {em2['note']}")
    print(f"    ⇒ 需要约 **{em2['seeds_needed']:.0f} 个种子**"
          f"（每种子 {em2['events_per_seed']:.1f} 个事件）")
    print(f"    阈值：≤{COST_THRESHOLD['worth_it_max_seeds']} 值得 / "
          f"≤{COST_THRESHOLD['marginal_max_seeds']} 边缘 / "
          f">{COST_THRESHOLD['not_worth_min_seeds']} 不值得")
    print(f"    判定：**{em2['verdict']}**")

    if args.skip_em3:
        print("\n  （--skip-em3：只做功效分析，暂不执行加码）")
    elif args.force_em4 or em2["verdict"] in ("不值得",):
        out["em4"] = em4_honest_report(em1, em2)
        print(f"\n  【EM.4 诚实报告分支】")
        print(f"    {out['em4']['text'][:200]}…")
    else:
        out["em3"] = em3_execute(em1, em2, max_seeds=args.max_seeds)
        print(f"\n  【EM.3 执行结果】")
        e3 = out["em3"]
        print(f"    加码到 {e3['n_seeds']} 个种子")
        if "cross_run_p" in e3:
            print(f"    跨运行检验：均值差 {e3['cross_run_mean_diff']:+.2f}"
                  f" ± {e3['cross_run_sem']:.2f}   t={e3['cross_run_t']:+.2f}"
                  f"   p={e3['cross_run_p']:.4f}"
                  f"   H4 方向={'支持' if e3['h4_direction'] else '不支持'}")

    out["elapsed_sec"] = time.time() - t0
    prev = _safe_load(RESULT)
    prev.update(out)
    save_json(prev, RESULT)
    print(f"\n  产物：out/{RESULT}（{out['elapsed_sec']:.0f}s）")
    return prev


if __name__ == "__main__":
    main()
