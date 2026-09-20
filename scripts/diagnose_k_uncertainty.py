"""k 的不确定度诊断：这个指标到底有没有分辨力？

为什么要做这件事（本轮的直接触发）
----------------------------------
前置校验发现：把 EA.4 的标定窗口从 ``[500,3000)`` 改成与实验窗口一致的
``[6000,6400)``（λ̄ 从 94.601 → 91.692，只差 3%）之后，
taker 臂的 k 从 **0.455 跳到 1.042**。

**3% 的标尺变化引起 129% 的指标变化** —— 这个敏感度不成比例，
所以必须先回答一个更基础的问题：**k 这个估计量本身有多大的不确定度？**

做法
----
`fit_power_law` 只有 4 个档位（0.1% / 0.25% / 0.5% / 1%），每档的滑点是
**3 个种子的配对差分均值 ± sem**。于是可以对每档按学生的 t 分布（df = n_seeds−1）
重采样，重算 k —— 这就是**参数 bootstrap**，完全不依赖重跑，而且
**复用的是产品代码里的同一个 ``fit_power_law``**（不写第二份实现）。

输出：每个臂的 k 中位数、95% 分位区间、区间宽度、以及逐档显著性。
判读标准：若两套口径的 95% 区间**大幅重叠**，则「0.455 vs 1.042」
这个差异在统计上不可分辨——问题不在标定，在**指标的分辨力**。

用法::
    python scripts/diagnose_k_uncertainty.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import OUT, banner, save_json  # noqa: E402
from tw.impact import fit_power_law, unsaturated  # noqa: E402

RESULT = "k_uncertainty.json"
N_BOOT = 20_000
RNG_SEED = 20260919


def bootstrap_k(rows: list[dict], *, n_boot: int = N_BOOT,
                seed: int = RNG_SEED, x_key: str = "delivered_qty",
                key: str = "slippage_bp",
                apply_unsat_filter: bool = True) -> dict:
    """对一臂的滑点做参数 bootstrap，返回 k 的分布统计。

    ⚠️ 复用 ``fit_power_law``（产品代码），不自己写一份幂律拟合——
    否则诊断与被诊断的东西口径可能不同，而那种差异**看不出来**。

    ``apply_unsat_filter``
        默认 ``True``：只用未饱和档位（与 ``fit_unsaturated`` 同口径）。
        置 ``False`` 可以看"把饱和档也算进去"的情形——**那是偏差换精度的权衡**：
        饱和档的滑点因"目标没吃满"被系统性压低，会把 k 拉偏；
        但它同时增加了自由度（``df = n − 2``），而 ``t`` 临界值随之大幅下降
        （3 档 df=1 → t=12.71；7 档 df=5 → t=2.57）。
        这个开关存在，就是为了把"多档位到底能不能救分辨力"这件事**量出来**，
        而不是靠直觉断言。
    """
    if apply_unsat_filter:
        rows = unsaturated(rows)
    if len(rows) < 3:
        return {"ok": False, "reason": "有效档位 < 3"}
    df = max(1, int(rows[0].get("n_seeds", 1)) - 1)
    mu = np.array([r[key] for r in rows], dtype=float)
    se = np.array([r["slippage_sem_bp"] for r in rows], dtype=float)
    x = np.array([r[x_key] for r in rows], dtype=float)

    rng = np.random.default_rng(seed)
    ks: list[float] = []
    n_fail = 0
    for _ in range(n_boot):
        sampled = mu + se * rng.standard_t(df, size=len(mu))
        fake = [{x_key: xi, key: vi} for xi, vi in zip(x, sampled)]
        fit = fit_power_law(fake, key, x_key=x_key)
        if fit.ok:
            ks.append(fit.exponent)
        else:
            n_fail += 1
    if not ks:
        return {"ok": False, "reason": "bootstrap 全部拟合失败"}
    ks_a = np.array(ks)
    point = fit_power_law(rows, key, x_key=x_key)
    return {
        "ok": True,
        "k_point": float(point.exponent),
        "k_median": float(np.median(ks_a)),
        "ci95": [float(np.percentile(ks_a, 2.5)),
                 float(np.percentile(ks_a, 97.5))],
        "width": float(np.percentile(ks_a, 97.5) - np.percentile(ks_a, 2.5)),
        "p_below_0.7": float((ks_a < 0.7).mean()),
        "n_boot_ok": len(ks_a),
        "n_boot_fail": n_fail,
        "n_seeds": int(rows[0].get("n_seeds", 0)),
        "significant_levels": sum(
            1 for r in rows
            if abs(r.get("slippage_t") or 0.0) > 2.0),
        "n_levels": len(rows),
    }


def _collect() -> dict:
    """收集**所有**用同一装置（``paired_impact`` + ``fit_power_law``）算出来的 k。

    ⚠️ 这一点很要紧：分辨力问题**不是 EA.4 独有的**。
    凡是同一个装置跑出来的 k 都受同一套噪声约束，所以影响范围必须
    **逐个臂量出来**，而不是靠推断说"大概也受影响"。
    """
    src: dict[str, dict] = {}
    wa = OUT / "workstream_A_metrics.json"
    if wa.exists():
        d = json.loads(wa.read_text(encoding="utf-8"))
        for blk, tag in (("ea1_uncorrected", "EA.1 uncorrected"),
                         ("ea1_corrected", "EA.1b corrected (λ̄=94.601)"),
                         ("ea4_one_sided", "EA.4 (λ̄=94.601)")):
            for arm, a in (d.get(blk) or {}).get("arms", {}).items():
                src[f"{tag} | {arm}"] = a
    vf = OUT / "verify_EA4_calibration.json"
    if vf.exists():
        d = json.loads(vf.read_text(encoding="utf-8"))
        rerun = d.get("rerun") or {}
        for arm, a in (rerun.get("arms") or {}).items():
            src[f"EA.4′ aligned (λ̄={rerun.get('base_lambda', float('nan')):.3f}) | {arm}"] = a
    # 工作线B：元订单覆盖面消融（EB.1）
    wb = OUT / "workstream_B_metrics.json"
    if wb.exists():
        d = json.loads(wb.read_text(encoding="utf-8"))
        for arm, a in ((d.get("eb1") or {}).get("arms") or {}).items():
            src[f"EB.1 | {arm}"] = a
    # 三线合并验证：J1–J3（同装置、同口径）
    wj = OUT / "workstream_joint_metrics.json"
    if wj.exists():
        d = json.loads(wj.read_text(encoding="utf-8"))
        for arm, a in ((d.get("ej1") or {}).get("arms") or {}).items():
            src[f"J1–J3 | {arm}"] = a
    return src


def seeds_needed(width_at_n: float, target_delta: float, n: int = 3) -> float:
    """把 95% 区间宽度压到 ``target_delta`` 大约需要多少种子。

    近似依据：区间宽度 ∝ 1/√n_seeds（标准误的定义）。
    ⚠️ 这是**下界式**估计——它只考虑提高种子数，不考虑
    "增加冲击档位"或"提高信噪比"等其他降噪途径。

    为什么必须给出这个数：设计者要判断"继续投入这套装置值不值"，
    靠的就是它。
    """
    if width_at_n <= 0 or target_delta <= 0:
        return float("nan")
    return n * (width_at_n / target_delta) ** 2


def main() -> dict:
    banner("k 的不确定度诊断：这个指标有没有分辨力？")
    src = _collect()
    if not src:
        print("  ❌ 找不到 out/workstream_A_metrics.json")
        return {}

    out: dict = {"n_boot": N_BOOT, "rng_seed": RNG_SEED, "arms": {}}
    print(f"  {'臂':<44}{'k':>7}{'95% 区间':>20}{'宽度':>8}{'显著档位':>10}")
    print("  " + "-" * 92)
    for tag, a in src.items():
        rows = a.get("rows") or []
        r = bootstrap_k(rows)
        out["arms"][tag] = r
        if not r["ok"]:
            print(f"  {tag:<44}{'—':>7}{'（' + r['reason'] + '）':>20}")
            continue
        lo, hi = r["ci95"]
        print(f"  {tag:<44}{r['k_point']:>7.3f}"
              f"{f'[{lo:.3f}, {hi:.3f}]':>20}{r['width']:>8.3f}"
              f"{str(r['significant_levels']) + '/' + str(r['n_levels']):>10}")

    # ---- 关键比较：同一臂、两套标定口径 ----
    print("\n  " + "=" * 92)
    print("  关键比较：同一臂、两套标定口径（legacy vs aligned）")
    print("  " + "=" * 92)
    key_legacy = "EA.4 (λ̄=94.601) | taker"
    key_aligned = "EA.4′ aligned (λ̄=91.692) | taker"
    named = {}
    for label, k in (("legacy ", key_legacy), ("aligned", key_aligned)):
        r = out["arms"].get(k)
        named[label.strip()] = r
        if r and r.get("ok"):
            lo, hi = r["ci95"]
            print(f"  {label} k={r['k_point']:.3f}  95% CI=[{lo:.3f}, {hi:.3f}]"
                  f"  宽度={r['width']:.3f}"
                  f"  显著档位={r['significant_levels']}/{r['n_levels']}")
        else:
            print(f"  {label} 缺数据（键：{k!r}）")
    a, b = named.get("legacy"), named.get("aligned")
    if a and b and a.get("ok") and b.get("ok"):
        lo = max(a["ci95"][0], b["ci95"][0])
        hi = min(a["ci95"][1], b["ci95"][1])
        overlap = max(0.0, hi - lo)
        span = min(a["width"], b["width"])
        frac = (overlap / span) if span > 0 else None
        out["taker_overlap"] = {
            "overlap": overlap,
            "narrower_width": span,
            "overlap_frac_of_narrower": frac,
            "indistinguishable": bool(frac is not None and frac > 0.5),
        }
        print(f"\n  两区间重叠 {overlap:.3f}"
              f"（较窄区间宽 {span:.3f} 的 {frac:.1%}）")
        print("  ⇒ " + ("**不可分辨**：这个装置无法区分 0.455 与 1.042"
                        if out["taker_overlap"]["indistinguishable"]
                        else "区间基本分离，差异可能是真实的"))

    save_json(out, RESULT)

    # ---- 总览：有多少个臂"能被判定"？----
    ok = {t: r for t, r in out["arms"].items() if r.get("ok")}
    if ok:
        widths = [r["width"] for r in ok.values()]
        contain_half = [t for t, r in ok.items()
                        if r["ci95"][0] <= 0.5 <= r["ci95"][1]]
        contain_zero = [t for t, r in ok.items()
                        if r["ci95"][0] <= 0.0 <= r["ci95"][1]]
        out["summary"] = {
            "n_arms": len(ok),
            "n_ci_contains_0.5": len(contain_half),
            "n_ci_contains_0": len(contain_zero),
            "width_min": float(min(widths)),
            "width_median": float(np.median(widths)),
            "width_max": float(max(widths)),
            "arms_containing_0.5": contain_half,
        }
        print("\n  " + "=" * 92)
        print("  总览：这套装置能判定什么？")
        print("  " + "=" * 92)
        print(f"  臂总数 {len(ok)}；95% 区间**包含 0.5** 的臂："
              f"{len(contain_half)}/{len(ok)}")
        print(f"  95% 区间**包含 0**（即「滑点方向都不确定」）的臂："
              f"{len(contain_zero)}/{len(ok)}")
        print(f"  区间宽度：最小 {min(widths):.3f} / 中位 {np.median(widths):.3f}"
              f" / 最大 {max(widths):.3f}")
        print("  ⇒ 若几乎每个臂的区间都包含 0.5，那么"
              "「k 离 0.5 有多远」这句话在本装置上**无法被判定**。")

    # ---- 需要多少种子？（设计者判断"值不值得继续投入"的输入）----
    if a and a.get("ok"):
        w, n = a["width"], max(1, a["n_seeds"])
        print("\n  " + "=" * 92)
        print("  要把 k 的分辨力做到可用，需要多少种子？（宽度 ∝ 1/√n，纯种子路线）")
        print("  " + "=" * 92)
        targets = [
            ("分辨 0.455 vs 1.042（差 0.587）", 0.587),
            ("分辨 0.947 vs 0.455（差 0.492）", 0.492),
            ("判定是否落在 0.5±0.1", 0.2),
            ("判定是否落在 0.5±0.05", 0.1),
            ("判定是否落在 0.5±0.02", 0.04),
        ]
        need = {}
        for label, d in targets:
            n_need = seeds_needed(w, d, n)
            need[label] = n_need
            print(f"  {label:<34} 约需 {n_need:>8.0f} 个种子"
                  f"{'（便宜）' if n_need <= 60 else '（昂贵，需换指标或增加档位）'}")
        out["seeds_needed"] = need
        out["seeds_needed_note"] = (
            "宽度 ∝ 1/√n 的近似，只考虑增加种子数。"
            "另两条降噪途径：增加冲击档位（现只有 4 档，阶段3 的 SHOCK_SIZES 有 7 档）、"
            "提高信噪比（现有 4 档里低档位 t<1.6，几乎全是噪声）。")
        save_json(out, RESULT)

    print(f"\n  产物：out/{RESULT}")
    return out


if __name__ == "__main__":
    main()
