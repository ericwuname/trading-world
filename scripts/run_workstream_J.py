"""工作线J：回溯审计 —— 阶段3 最初那个 k 是否也有分辨力问题。

任务书：`交易世界 · 分辨力危机应对任务书.md` §5。

为什么这条最重要
----------------
一期「阶段3」测出的冲击幂律指数（报告原文写的是 **γ≈0.8**，对应
**k = 1/γ ≈ 1.25**）是这整个两期项目里**被引用次数最多的一个数字**：
二期阶段6「用长记忆订单流修复冲击律」、阶段8「Hawkes 聚集」、
三线深挖 A 线——全都以「模型冲击偏线性、不满足平方根律」这条定性判断为出发点。
如果这个最初读数本身也没有分辨力，那要重新评估的不是某个数字，而是**立项依据**。

⚠️ 前提核对（任务书有四处与源码不符，逐条列在这里）
-------------------------------------------------
1. 任务书写「重跑时种子数不能超过原始种子数（**通常是 3 个**）」——
   **实际阶段3 用的是 8 个种子**（``run_stage3.K_SEEDS = 8``）。
   8 个种子的分辨力明显好于三线深挖的 3 个，这对审计结论影响很大。
2. 任务书说 ``out/stage3_*.json`` 可能"只有最终的 k 点估计"——
   **实际有完整的逐档细分数据**（``C2_staged.rows``，7 档，每档带
   ``slippage_sem_bp`` / ``slippage_t`` / ``n_seeds``），**不需要重跑**。
3. 任务书说原始 k 是「0.77~0.8」——那是 **γ**，不是 k。
   一期报告原文：。γ≈0.8），而真实市场是凸的（γ≈2）。
   本脚本两个量都报，并明确换算关系 ``k = 1/γ``（k=0.5 平方根律、k=1 线性）。
4. 任务书要求"种子数值必须与原始完全一致，否则只能作为一般性估计"——
   既然数据本来就在，本脚本**不重跑**，所以这条限制自动满足：
   审计用的就是**原始那次运行**的读数（8 个原始种子）。

用法::
    python scripts/run_workstream_J.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import OUT, banner, ensure_scripts_on_path, save_json  # noqa: E402

RESULT = "workstream_J_metrics.json"
#: 阶段3 原始运行的种子数（``run_stage3.K_SEEDS``）。审计**不许**超过它。
ORIGINAL_N_SEEDS = 8
STAGE3_BLOCK = "C2_staged"


def _safe_load(name: str) -> dict:
    """读 out/<name>；缺失或解析失败返回空 dict（见 run_workstream_I.py 的说明）。"""
    import json
    p = OUT / name
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def locate_stage3_original_data(raw: dict | None = None) -> dict:
    """EJ.1：定位阶段3 原始数据，并**显式断言**种子数不超过原始值。

    ``raw``
        允许**注入**数据（默认从 ``out/stage3_metrics.json`` 读）。
        ⚠️ 这个参数是为测试加的，而且很有必要：
        **mutation 沙箱只复制 `gui/scripts/strategies/tests/tw` + `data`，
        不复制 `out/`**（那里是几百 MB 产物）。所以任何"读 out/ 的测试"
        在沙箱里都会失败，把基线弄红——而基线一红，所有变异体都会"变红"，
        **整个变异验证就变成了假证据**（本轮实测踩到：基线 4 failures）。
        有了注入参数，测试可以用构造的假数据测断言逻辑，不依赖产物。

    任务书 §5.5 的变异体 M49 钉的就是这件事：重跑时用了更多种子，
    那测的就不再是"原始结论的分辨力"，而是"一次新实验的分辨力"。
    本脚本不重跑（数据本来就在），但断言仍然保留——因为它同时守住了
    "将来若有人改成重跑，不许偷偷扩种子"这条线。
    """
    d = raw if raw is not None else _safe_load("stage3_metrics.json")
    block = d.get(STAGE3_BLOCK) or {}
    rows = block.get("rows") or []
    info: dict = {
        "file": "out/stage3_metrics.json",
        "block": STAGE3_BLOCK,
        "found": bool(rows),
        "n_levels": len(rows),
        "has_per_level_detail": all(
            ("slippage_sem_bp" in r and "n_seeds" in r) for r in rows) if rows else False,
        "fits": {
            "fit_all": (block.get("fit_all") or {}),
            "fit_unsaturated": (block.get("fit_unsaturated") or {}),
            "rows_unsaturated": block.get("rows_unsaturated"),
        },
    }
    if not rows:
        info["error"] = "找不到阶段3 的逐档细分数据"
        return info

    n_seeds = {int(r["n_seeds"]) for r in rows}
    info["n_seeds_in_data"] = sorted(n_seeds)
    # ⚠️ 断言：数据的种子数不得超过原始值（M48 钉的正是这里）
    assert max(n_seeds) <= ORIGINAL_N_SEEDS, (
        f"种子数 {max(n_seeds)} 超过原始值 {ORIGINAL_N_SEEDS}——"
        "那审计的就不是原始结论的分辨力了")
    info["rerun_used_more_seeds"] = bool(max(n_seeds) > ORIGINAL_N_SEEDS)
    info["note"] = ("数据本来就在产物里，本脚本**不重跑** ⇒ 审计用的就是"
                    "原始那次运行的读数（8 个原始种子），"
                    "因此不存在「重采样种子」这个限定条件。")
    return info


def audit_k_via_diagnosis_toolkit(rows: list[dict]) -> dict:
    """EJ.2：复用 ``diagnose_k_uncertainty.bootstrap_k`` 算 k 的 95% 区间。

    **不重写实现**——诊断工具已经被 9 条测试钉过，重写一份只会带来口径分叉。
    """
    ensure_scripts_on_path()
    from diagnose_k_uncertainty import bootstrap_k

    r = bootstrap_k(rows)
    if not r.get("ok"):
        return {"ok": False, "reason": r.get("reason")}
    lo, hi = r["ci95"]
    return {
        "ok": True, **r,
        "covers_0.5_sqrt_law": bool(lo <= 0.5 <= hi),
        "covers_1.0_linear": bool(lo <= 1.0 <= hi),
        # γ = 1/k 是报告原文用的量；区间端点要**翻转**（单调递减映射）
        "gamma_point": (1.0 / r["k_point"]) if r["k_point"] else None,
        "gamma_ci95": ([1.0 / hi, 1.0 / lo] if lo > 0 else None),
    }


def concavity_for_stage3(rows: list[dict]) -> dict:
    """EJ.3：对阶段3 数据跑凹度判据（全部档位对，不只挑显著的）。"""
    ensure_scripts_on_path()
    from run_workstream_I import all_pairs, verdict_counts

    # 只保留未饱和档位（饱不饱和由 fill_ratio 决定，与 tw.impact.unsaturated 同口径）
    unsat = [r for r in rows if r.get("fill_ratio", 0.0) >= 0.95]
    pairs = all_pairs(unsat) if len(unsat) >= 2 else []
    return {
        "n_unsaturated": len(unsat),
        "unsaturated_sizes": [r["shock_frac"] for r in unsat],
        "pairs": pairs,
        "verdicts": verdict_counts(pairs) if pairs else {},
    }


def compare_claim_to_diagnosis(claim: dict, ej2: dict, ej3: dict) -> dict:
    """EJ.4：把「一期原表述」与「本次审计」逐条对照。"""
    k_all = ((claim.get("fits") or {}).get("fit_all") or {}).get(
        "分期|因果滑点（平均成交让步）", {}).get("k")
    k_unsat = ((claim.get("fits") or {}).get("fit_unsaturated") or {}).get(
        "分期|因果滑点", {}).get("k")
    out = {
        "original_k_fit_all_7levels": k_all,
        "original_k_fit_unsaturated_3levels": k_unsat,
        "original_gamma_implied": (1.0 / k_unsat) if k_unsat else None,
        "diagnosed_k_point": ej2.get("k_point"),
        "diagnosed_ci95": ej2.get("ci95"),
        "ci_covers_1_linear": ej2.get("covers_1.0_linear"),
        "ci_covers_0.5_sqrt": ej2.get("covers_0.5_sqrt_law"),
        "verdicts": ej3.get("verdicts") or {},
    }
    return out


def main() -> dict:
    banner("工作线J  回溯审计：阶段3 最初的 k 有没有分辨力？")
    t0 = time.time()
    out: dict = {"stage": "J"}

    ej1 = locate_stage3_original_data()
    out["ej1"] = ej1
    print(f"\n【EJ.1】数据定位")
    print(f"  {ej1['file']} :: {ej1['block']}  → 找到 {ej1.get('n_levels')} 档，"
          f"逐档细节={'有' if ej1.get('has_per_level_detail') else '**没有**'}")
    if not ej1.get("found"):
        print(f"  ❌ {ej1.get('error')}")
        save_json(out, RESULT)
        return out
    print(f"  种子数={ej1.get('n_seeds_in_data')}（原始值 {ORIGINAL_N_SEEDS}）"
          f"  未饱和档位={ej1['fits'].get('rows_unsaturated')}")

    d = _safe_load("stage3_metrics.json")
    rows = d[STAGE3_BLOCK]["rows"]

    ej2 = audit_k_via_diagnosis_toolkit(rows)
    out["ej2"] = ej2
    print(f"\n【EJ.2】置信区间审计（复用诊断工具的 bootstrap）")
    if ej2.get("ok"):
        lo, hi = ej2["ci95"]
        print(f"  k    = {ej2['k_point']:.3f}   95% CI = [{lo:.3f}, {hi:.3f}]"
              f"  （宽度 {ej2['width']:.3f}）")
        print(f"  γ=1/k= {ej2['gamma_point']:.3f}"
              + (f"   95% CI = [{ej2['gamma_ci95'][0]:.3f}, {ej2['gamma_ci95'][1]:.3f}]"
                 if ej2.get("gamma_ci95") else ""))
        print(f"  区间覆盖 0.5（平方根律）？{'✅ 是' if ej2['covers_0.5_sqrt_law'] else '❌ 否'}"
              f"    覆盖 1.0（线性）？{'✅ 是' if ej2['covers_1.0_linear'] else '❌ 否'}")
        print(f"  显著档位 {ej2['significant_levels']}/{ej2['n_levels']}"
              f"（种子 {ej2['n_seeds']} 个）")
    else:
        print(f"  ❌ {ej2.get('reason')}")

    ej3 = concavity_for_stage3(rows)
    out["ej3"] = ej3
    print(f"\n【EJ.3】凹度判据（未饱和 {ej3['n_unsaturated']} 档，"
          f"全部 {len(ej3['pairs'])} 对都报）")
    for p in ej3["pairs"]:
        if not p["ok"]:
            continue
        print(f"  {p['size_small']}→{p['size_big']:<7} e={p['e']:.3f} ± {p['sem']:.3f}"
              f"  CI=[{p['ci_lower']:.3f}, {p['ci_upper']:.3f}]  → {p['verdict']}"
              f"  (t_vs_sqrt={p['t_vs_sqrt']:.2f})"
              f"{'  ⚠️低信噪比' if p['low_snr'] else ''}")

    out["ej4"] = compare_claim_to_diagnosis(ej1, ej2, ej3)
    print(f"\n【EJ.4】原表述 vs 本次审计")
    e4 = out["ej4"]
    print(f"  一期原始 k：全 7 档 = {e4['original_k_fit_all_7levels']:.3f}"
          f"，未饱和 3 档 = {e4['original_k_fit_unsaturated_3levels']:.3f}"
          f"（对应 γ ≈ {e4['original_gamma_implied']:.3f}）")
    print(f"  本次审计 k = {e4['diagnosed_k_point']:.3f}"
          f"  CI = {[round(v, 3) for v in e4['diagnosed_ci95']]}")

    out["elapsed_sec"] = time.time() - t0
    save_json(out, RESULT)
    print(f"\n  产物：out/{RESULT}（{out['elapsed_sec']:.0f}s）")
    return out


if __name__ == "__main__":
    main()
