"""工作线O：一致性审计 —— 把「点估计方向变了」与「结论真的变了」分开。

任务书：`交易世界 · 一致性审计与收尾任务书.md` §2。

这一轮要审的是**上一轮自己的账**
------------------------------
「回填标准配置」那一轮报告里写了两条「历史结论被推翻」：

    · EO.2「+基本面派改善最大」→ 变成「比基线还差」
    · EO.4「J2 优于 J3」       → 反转成「J3 优于 J2」

这两条都只凭**点估计的方向变化**就下了「推翻」的判断——
而项目一路在纠正的，恰恰就是这个毛病。
本工作线用 EL.2 已经建立、并已被判定过「无法判定」的那套区间重叠度量，
把四条比较逐一重新检验。

⚠️ **本脚本不新增任何模拟**：全部输入来自 ``out/workstream_L_metrics.json``
（8 个家族各自的 k 与 95% CI，都是上一轮刚跑出来的）。
所以它跑得很快——**但它的结论会追溯性地改写上一轮报告里的措辞。**

用法::

    python scripts/run_workstream_O.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import OUT, banner, ensure_scripts_on_path, load_json, save_json  # noqa: E402

RESULT = "workstream_O_metrics.json"

#: (编号, A 家族, B 家族, 比较名, 上一轮报告里的措辞)
COMPARISONS = [
    ("EO.1", "EB1_chartist_only", "EB1_plus_zero_intel",
     "基线 vs +零智能", "上一轮报告说「改善」"),
    ("EO.2", "EB1_chartist_only", "EB1_plus_fundamentalist",
     "基线 vs +基本面派", "上一轮写成「推翻」← 必须严格检验"),
    ("EO.3", "EB1_chartist_only", "EB1_all_three",
     "基线 vs 三者全部", "上一轮报告说「改善」"),
    ("EO.4", "EB1_all_three", "J3_plus_A",
     "J2 vs J3", "上一轮写成「推翻」← 本工作线最重要的一项"),
]

#: 上一轮报告里那些**依赖点估计方向**的表述（若判定为「无法判定」，必须撤回/改写）
STATEMENTS_AT_RISK = {
    "EO.2": "「+基本面派改善最大」变成「比基线还差」（被当作「结论被推翻」来写）",
    "EO.4": "「J2 优于 J3」反转为「J3 优于 J2」（被当作「结论被推翻」来写）",
}


def _load_results() -> dict:
    p = OUT / "workstream_L_metrics.json"
    if not p.exists():
        raise RuntimeError(
            "缺 out/workstream_L_metrics.json——先跑 scripts/run_workstream_L.py")
    return load_json("workstream_L_metrics.json") or {}


def run_EO1_to_EO4() -> dict:
    """EO.1~EO.4：四条比较逐一给出三选一判定。"""
    ensure_scripts_on_path()
    from tw.analyzer_consistency import pairwise_verdict

    L = _load_results()
    fams = L.get("families") or {}
    print("\n【EO.1~EO.4】用区间重叠度量重新检验四条比较")
    print(f"  {'编号':<7}{'比较':<22}{'k_A':>7}{'k_B':>7}{'重叠':>9}  判定")

    out: dict = {"comparisons": [], "n_seeds": None, "threshold": None}
    for cid, a, b, label, note in COMPARISONS:
        ra, rb = fams.get(a), fams.get(b)
        if not (ra and rb and ra.get("k") is not None and rb.get("k") is not None):
            print(f"  {cid:<7}{label:<22}  ⚠️ 缺数据")
            continue
        n_seeds = ra.get("n_seeds") or 8
        v = pairwise_verdict(a, tuple(ra["ci"]), ra["k"],
                             b, tuple(rb["ci"]), rb["k"], n_seeds)
        d = v.as_dict()
        d.update({"id": cid, "label": label, "prior_statement": note,
                  "is_retraction_candidate": cid in STATEMENTS_AT_RISK})
        out["comparisons"].append(d)
        out["n_seeds"] = n_seeds
        out["threshold"] = v.threshold
        print(f"  {cid:<7}{label:<22}{ra['k']:>7.3f}{rb['k']:>7.3f}"
              f"{v.overlap_fraction * 100:>8.1f}%  {v.verdict}")
    return out


def run_EO5(audit: dict) -> dict:
    """EO.5：对全部「无法判定」给出 required_n 外推，并按工作线M 的阈值给建议。"""
    print("\n【EO.5】需要多少样本才能分辨（按工作线M 的阈值给建议）")
    # 复用工作线M 事先定死的阈值（同一套标准，不另立一套）
    from run_workstream_M import COST_THRESHOLD
    rows = []
    for c in audit.get("comparisons") or []:
        if c["verdict"] != "依然无法判定":
            continue
        need = c.get("required_n")
        if need is None or need != need:
            continue
        if need <= COST_THRESHOLD["worth_it_max_seeds"]:
            advice = "可以考虑定向加种子"
        elif need <= COST_THRESHOLD["marginal_max_seeds"]:
            advice = "边缘"
        else:
            advice = "不建议（超出阈值）"
        rows.append({"id": c["id"], "label": c["label"],
                     "required_n": need, "advice": advice,
                     "current_n": audit.get("n_seeds")})
        print(f"  {c['id']}  {c['label']:<22} 需要约 {need:>7.0f} 个种子"
              f"（现用 {audit.get('n_seeds')}）  ⇒ {advice}")
    return {"rows": rows, "thresholds": dict(COST_THRESHOLD),
            "note": ("阈值沿用工作线M 事先定死的那一套（≤50 值得 / ≤100 边缘 / "
                     ">100 不建议）——不另立标准。")}


def run_EO6_retractions(audit: dict) -> dict:
    """EO.6（本脚本新增）：把「必须撤回的表述」显式列出来。

    任务书 §2.4 的硬要求：若 EO.2/EO.4 判定为「无法判定」，
    报告里那两处「反转」表述**必须在交付小结与历史清单里正式撤回或改写**，
    不能让它们继续以「结论」的措辞留在文档里。
    """
    print("\n【撤回清单】上一轮报告里站不住的表述")
    items = []
    for c in audit.get("comparisons") or []:
        if not c.get("is_retraction_candidate"):
            continue
        if c["verdict"] == "依然无法判定":
            items.append({
                "id": c["id"], "label": c["label"],
                "original": STATEMENTS_AT_RISK.get(c["id"], ""),
                "overlap_fraction": c["overlap_fraction"],
                "action": "撤回「推翻」的措辞",
                "replacement": (
                    f"点估计方向为 {c['point_a']:.3f} → {c['point_b']:.3f}，"
                    f"但两区间重叠达较窄区间的 "
                    f"**{c['overlap_fraction'] * 100:.1f}%** ⇒ "
                    f"**统计上无法与此前的读数区分**，不能称为「结论被推翻」。"),
            })
            print(f"  ❌ {c['id']} {c['label']}：重叠 "
                  f"{c['overlap_fraction'] * 100:.1f}% ⇒ 撤回「推翻」措辞")
        else:
            print(f"  ✅ {c['id']} {c['label']}：判定「{c['verdict']}」——"
                  "可以保留方向性表述")
    if not items:
        print("  （没有需要撤回的表述）")
    return {"items": items, "n_retractions": len(items)}


def run_EO4_targeted(target_seeds: int = 16,
                     checkpoints: tuple[int, ...] = (8, 12, 16)) -> dict:
    """O.4：对 CI 最宽的两个臂**定向**加种子（并做渐进稳定性检查）。

    为什么只加这两个臂：EO.2 的 required_n 落在工作线M 的「可以考虑」区间内，
    而 EO.4 要求 193 个（超阈值）。所以按成本效益只加 EO.2 涉及的
    ``EB1_chartist_only`` 与 ``EB1_plus_fundamentalist``。

    ⚠️ **渐进检查**（任务书 §7 的明确要求）：
    「最好能看到『加到一半种子时已经出现同一个方向』这种渐进稳定的迹象，
    而不是『加满目标种子数后突然由不可判定变成可判定』这种断崖式变化，
    后者更值得怀疑。」

    实现方式：**逐种子跑**（每个种子单独调 ``paired_impact``），
    于是「前 8 个种子」「前 12 个」「全部 16 个」都是**嵌套子集**，
    可以直接重新聚合——不需要为每个 checkpoint 重跑一遍。
    （``rows`` 只存聚合的 mean±sem，没有逐种子数组，所以不能事后从
     一次 16 种子的运行里切分。这是逐种子跑的唯一原因。）
    """
    ensure_scripts_on_path()
    import numpy as np
    import run_workstream_A as A
    import run_workstream_L as L
    from diagnose_k_uncertainty import bootstrap_k
    from run_stage3 import HORIZON, SEED0, SHOCK_SIZES, make_controls, paired_impact
    from tw.impact import unsaturated

    seeds_all = [SEED0 + 7 * i for i in range(target_seeds)]
    mix = A.ea4_market_spec()["mix"]
    fams = L.families()
    out: dict = {"target_seeds": target_seeds, "checkpoints": list(checkpoints),
                 "seed_list": seeds_all, "families": {}}

    for name in ("EB1_chartist_only", "EB1_plus_fundamentalist"):
        spec = fams[name]
        fn = L._make_factory(spec, None)
        print(f"\n  【O.4 · {name}】逐种子跑 {target_seeds} 个种子（7 档）")
        per_seed = []
        for s in seeds_all:
            ctrl = make_controls(mix, [s], HORIZON, factory=fn)
            rows = [paired_impact(f"{name}|{size:.4f}|s{s}", mix, size, 20, [s],
                                  ctrl, HORIZON, factory=fn)
                    for size in SHOCK_SIZES]
            per_seed.append({"seed": s, "rows": rows})
        print(f"    逐种子跑完，开始按 checkpoint 聚合")

        cps = []
        for n in checkpoints:
            if n > len(per_seed):
                continue
            agg = []
            for i, size in enumerate(SHOCK_SIZES):
                vals = np.array([p["rows"][i]["slippage_bp"] for p in per_seed[:n]],
                                dtype=float)
                fills = np.array([p["rows"][i]["fill_ratio"] for p in per_seed[:n]],
                                 dtype=float)
                base = dict(per_seed[0]["rows"][i])
                base.update({
                    "shock_frac": float(size),
                    "slippage_bp": float(vals.mean()),
                    "slippage_sem_bp": float(vals.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan"),
                    "fill_ratio": float(fills.mean()),
                    "n_seeds": n,
                })
                agg.append(base)
            bk = bootstrap_k(agg)
            un = unsaturated(agg)
            row = {"n_seeds": n, "n_unsaturated": len(un),
                   "saturated_from": next((r["shock_frac"] for r in agg
                                           if r.get("fill_ratio", 1.0) < 0.95), None)}
            if bk.get("ok"):
                lo, hi = bk["ci95"]
                row.update({"k": bk["k_point"], "ci": [lo, hi],
                            "ci_width": bk["width"],
                            "contains_0_5": bool(lo <= 0.5 <= hi)})
            cps.append(row)
            print(f"    n={n:>3}  未饱和 {row['n_unsaturated']}/"
                  f"{len(agg)}  k={row.get('k', float('nan')):.3f}  "
                  f"CI=[{row.get('ci', [float('nan'), float('nan')])[0]:.3f}, "
                  f"{row.get('ci', [float('nan'), float('nan')])[1]:.3f}]  "
                  f"宽={row.get('ci_width', float('nan')):.3f}")
        out["families"][name] = {"per_seed_rows": [
            {"seed": p["seed"],
             "slippage_bp": [r["slippage_bp"] for r in p["rows"]],
             "fill_ratio": [r["fill_ratio"] for r in p["rows"]]}
            for p in per_seed], "checkpoints": cps}

    # 渐进稳定性判读：每个 checkpoint 都用同一批（嵌套）种子的前缀
    base_cps = (out["families"].get("EB1_chartist_only") or {}).get("checkpoints") or []
    fund_cps = (out["families"].get("EB1_plus_fundamentalist") or {}).get("checkpoints") or []
    if base_cps and fund_cps:
        from tw.analyzer_consistency import pairwise_verdict
        trend = []
        for cb, cf in zip(base_cps, fund_cps):
            if cb.get("k") is None or cf.get("k") is None:
                continue
            v = pairwise_verdict("baseline", tuple(cb["ci"]), cb["k"],
                                 "+fund", tuple(cf["ci"]), cf["k"], cb["n_seeds"])
            trend.append({"n_seeds": cb["n_seeds"], "k_baseline": cb["k"],
                          "k_fund": cf["k"],
                          "point_diff": cf["k"] - cb["k"],
                          "overlap_fraction": v.overlap_fraction,
                          "verdict": v.verdict,
                          "required_n": v.required_n})
        out["eo2_progressive"] = {
            "trend": trend,
            "note": ("渐进检查：若方向（point_diff 的符号）在 n=8/12/16 上**始终一致**，"
                     "说明它不是「加满种子后突然翻转」的运气读数；"
                     "若出现符号翻转，则更要保守。"),
        }
    return out


def main() -> dict:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--eo4", action="store_true",
                    help="额外执行 O.4：对两个宽 CI 臂定向加种子（含渐进检查，约 40 分钟）")
    ap.add_argument("--target-seeds", type=int, default=16)
    args = ap.parse_args()

    banner("工作线O  一致性审计（审上一轮自己的账）")
    t0 = time.time()
    audit = run_EO1_to_EO4()
    audit["eo5_required_n"] = run_EO5(audit)
    audit["eo6_retractions"] = run_EO6_retractions(audit)
    if args.eo4:
        audit["eo4_targeted"] = run_EO4_targeted(args.target_seeds)
    audit["conclusion"] = (
        f"四条比较里，"
        f"{sum(1 for c in audit['comparisons'] if c['verdict'] == '依然无法判定')} 条"
        f"判定为「依然无法判定」；"
        f"{audit['eo6_retractions']['n_retractions']} 条上一轮的「推翻」表述需要撤回。"
        "区间重叠检验与 required_n 外推的方法论本身是本轮的有效产出。")
    audit["elapsed_sec"] = time.time() - t0
    save_json(audit, RESULT)
    print(f"\n  ⭐ {audit['conclusion']}")
    print(f"  产物：out/{RESULT}（{audit['elapsed_sec']:.0f}s）")
    return audit


if __name__ == "__main__":
    main()

