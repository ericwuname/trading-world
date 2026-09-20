"""工作线L：把历史家族回填到「7 档 × 8 种子」标准配置。

任务书：`交易世界 · 回填标准配置与H4做实任务书.md` §3。

要解决的问题
------------
分辨力危机那一轮证明了：**扩大参与拟合的档位数**能显著收窄 CI
（3 档 → 7 档，宽度 2.662 → 1.193，代价近零）。
但历史上一大批结论还停在「3 种子 × 3~4 档」的原始配置上——
它们是《历史数值陈述清单》里"疑似裸奔"条目的主要来源。
本工作线把这些家族用标准配置重跑，给出**带 CI 的新读数**。

⚠️ 前提核对：任务书列了 10 个家族，但其中两个是重复的
----------------------------------------------------
``J1_stage6_baseline`` 与 ``EB1_chartist_only`` 是**同一个配置**
（都是"仅图表派元订单"，旧读数都是 k=1.2513572154501866——**逐位相同**）；
``J2_plus_B`` 与 ``EB1_all_three`` 同理（都是 k=0.7646304848983828）。
⇒ 实际只有 **8 个不同配置**，本脚本按 8 个跑，并在产物里显式记录这层等价关系
（省下的时间不是偷工，是**去重**；重复跑同一个配置只会浪费算力）。

⚠️ 第二个前提核对：任务书写 ``seeds=range(target_seeds)``——
``range`` 产生的种子是 0,1,2,…,7，而本项目的种子约定是
``SEED0 + 7*i``（``run_stage3.SEEDS``，8 个）。用 ``range(8)`` 会得到
**与阶段3 完全不同的另一批随机数**，于是"新旧对比"就失去了意义。
本脚本直接复用 ``run_stage3.SEEDS``（**与阶段3 逐位对齐**，任务书也要求"和阶段3 对齐"）。

用法::

    python scripts/run_workstream_L.py --only EA4_taker   # 先跑通一臂
    python scripts/run_workstream_L.py --group ea4        # EA.4 三臂
    python scripts/run_workstream_L.py                    # 全部 8 个
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import OUT, banner, ensure_scripts_on_path, load_json, save_json  # noqa: E402

RESULT = "workstream_L_metrics.json"
BRANCHING = 0.6
BETA = 0.15
SLICES = 20                 # 与 EA.4 / 工作线H 一致


def _safe_load(name: str) -> dict:
    p = OUT / name
    if not p.exists():
        return {}
    try:
        return load_json(name) or {}
    except Exception:  # noqa: BLE001
        return {}


# ======================================================================
def families() -> dict:
    """8 个待回填的家族（10 个里去掉了两个重复的）。

    ``original_k`` 是**旧配置下的原始读数**，只用于对照——
    报告里必须标注"新旧口径不可直接比较"（旧：3 种子/3~4 档；新：8 种子/未饱和档）。
    """
    ensure_scripts_on_path()
    import run_workstream_B as B
    import run_workstream_joint as J
    return {
        # ---- EA.4 三臂（单侧稀疏化）----
        "EA4_both": {"kind": "hawkes", "thin_scope": "both",
                     "original_k": 0.947, "original_n_seeds": 3, "group": "ea4"},
        "EA4_taker": {"kind": "hawkes", "thin_scope": "taker",
                      "original_k": 0.455, "original_n_seeds": 3, "group": "ea4"},
        "EA4_maker": {"kind": "hawkes", "thin_scope": "maker",
                      "original_k": 0.689, "original_n_seeds": 3, "group": "ea4"},
        # ---- EB.1 四档覆盖面消融 ----
        "EB1_chartist_only": {"kind": "stage6", "shares": J.SHARES_CHART,
                              "original_k": 1.251, "original_n_seeds": 3,
                              "group": "eb1", "alias_of": "J1_stage6_baseline"},
        "EB1_plus_zero_intel": {"kind": "stage6",
                                "shares": {"chartist": B.META_SHARE,
                                           "zero_intel": 0.5},
                                "original_k": 1.692, "original_n_seeds": 3,
                                "group": "eb1"},
        "EB1_plus_fundamentalist": {"kind": "stage6",
                                    "shares": {"chartist": B.META_SHARE,
                                               "fundamentalist": 0.5},
                                    "original_k": 0.951, "original_n_seeds": 3,
                                    "group": "eb1"},
        "EB1_all_three": {"kind": "stage6", "shares": J.SHARES_ALL,
                          "original_k": 0.765, "original_n_seeds": 3,
                          "group": "eb1", "alias_of": "J2_plus_B"},
        # ---- J3（B 三类覆盖 + A 修正 Hawkes）----
        "J3_plus_A": {"kind": "joint", "shares": J.SHARES_ALL,
                      "original_k": 0.957, "original_n_seeds": 3,
                      "group": "joint"},
    }


def backfill_seeds() -> list[int]:
    """回填用的种子 = **阶段3 的原始种子集合**（8 个，逐位对齐）。

    ⚠️ 不用 ``range(8)``：那会得到一批完全不同的随机数，
    于是"新旧对比"失去意义（任务书自己也要求"和阶段3 对齐"）。
    """
    ensure_scripts_on_path()
    from run_stage3 import SEEDS
    return list(SEEDS)


def _make_factory(spec: dict, base_lambda: float | None):
    """按家族类型给出 ``make_market`` 的 factory——**全部复用现有实现**。"""
    ensure_scripts_on_path()
    import run_workstream_B as B
    import run_workstream_joint as J
    from tw.order_flow.hawkes import HawkesConfig
    from tw.order_flow.hawkes_market import HawkesMarket

    kind = spec["kind"]
    if kind == "hawkes":
        cfg = HawkesConfig(base_lambda=base_lambda, branching=BRANCHING, beta=BETA)
        scope = spec["thin_scope"]
        return lambda sim: HawkesMarket(sim, hawkes_config=cfg, thin_scope=scope)
    if kind == "stage6":
        return B.coverage_factory(spec["shares"])        # 复用 B 线，不重写
    if kind == "joint":
        cfg = HawkesConfig(base_lambda=base_lambda, branching=BRANCHING, beta=BETA)
        return lambda sim: J.JointMarket(sim, **J.arm_kwargs(spec["shares"], cfg))
    raise ValueError(f"未知家族类型 {kind!r}")


def calibration_window() -> tuple[int, int]:
    """标定窗口 —— **必须等于实验窗口**（第六条纪律）。

    抽成独立函数是为了让它**可被测试**：如果它被写成了历史口径
    （``(500, 3000)``，含前 500 tick 瞬态），那就重蹈了 E8.2 的覆辙——
    而那种错误**不会报错**，只是让 λ̄ 偏高约 7%、机制悄悄变样。
    变异体 M53 钉的就是这里。
    """
    ensure_scripts_on_path()
    from run_stage3 import HORIZON, WARMUP
    return (WARMUP, WARMUP + HORIZON)


def calibrate_lambda(seeds) -> dict | None:
    """同配置同窗口标定 λ̄（第六条纪律），返回 ``{base_lambda, window}``。

    只有含 Hawkes 的家族需要（``stage6`` 家族不含 → 返回 ``None``）。
    """
    ensure_scripts_on_path()
    import run_workstream_A as A
    from tw.order_flow.hawkes_market import calibrate_base_lambda
    win = calibration_window()
    kw = A.ea4_calibration_kwargs(seeds, win, A.ea4_market_spec())
    return {"base_lambda": float(calibrate_base_lambda(**kw)),
            "window": list(win)}


def check_seed_consistency(rows: list[dict], expected: int, name: str) -> None:
    """**种子数一致性断言**（变异体 M52 钉这里）。

    标定 λ̄、饱和检查、幂律拟合**三处必须用同一批种子**。
    若某一处偷偷用了别的种子集合，"新旧对比"和"95% CI"都会失去意义，
    而且**不会报错**——数字只是悄悄变得不可比。
    所以这里用断言把它钉死（本脚本三处用的就是同一份 ``rows``，
    断言是防止将来有人把某一处改成"单独再跑一遍"）。
    """
    bad = [r.get("n_seeds") for r in rows if r.get("n_seeds") != expected]
    assert not bad, (
        f"{name}: 有档位的 n_seeds={bad} 与回填种子数 {expected} 不一致——"
        "标定/饱和检查/拟合三处必须用同一批种子")


def backfill_one_family(name: str, spec: dict, *, seeds=None,
                        shock_sizes=None) -> dict:
    """回填单个家族：标定 → 跑全档位 → 用未饱和档算 k 与 95% CI。"""
    ensure_scripts_on_path()
    import run_workstream_A as A
    from diagnose_k_uncertainty import bootstrap_k
    from run_stage3 import HORIZON, SHOCK_SIZES, make_controls, paired_impact
    from tw.impact import unsaturated

    seeds = list(seeds or backfill_seeds())
    shock_sizes = list(shock_sizes or SHOCK_SIZES)
    mix = A.ea4_market_spec()["mix"]

    lam_info = (calibrate_lambda(seeds)
                if spec["kind"] in ("hawkes", "joint") else None)
    lam = lam_info["base_lambda"] if lam_info else None
    fn = _make_factory(spec, lam)
    ctrl = make_controls(mix, seeds, HORIZON, factory=fn)
    rows = []
    for size in shock_sizes:
        r = paired_impact(f"{name}|{size:.4f}", mix, size, SLICES, seeds,
                          ctrl, HORIZON, factory=fn)
        rows.append(r)

    # ⚠️ **种子数一致性断言**（变异体 M52 钉这里）：
    #    标定/饱和检查/幂律拟合必须用**同一批种子**。
    #    若某一处偷偷用了别的种子集合，"新旧对比"与"CI"都会失去意义，
    #    而且**不会报错**——只是数字悄悄变得不可比。
    check_seed_consistency(rows, len(seeds), name)

    un = unsaturated(rows)
    bk = bootstrap_k(rows)          # 内部同样按 fill_ratio ≥ 0.95 过滤
    out = {
        "name": name, "kind": spec["kind"], "lambda_bar": lam,
        "n_seeds": len(seeds), "seed_list": seeds,
        "n_levels_total": len(rows),
        "unsaturated_sizes": [r["shock_frac"] for r in un],
        "n_unsaturated": len(un),
        "saturated_from": next((r["shock_frac"] for r in rows
                                if r.get("fill_ratio", 1.0) < 0.95), None),
        "rows": rows,
        "original_k": spec.get("original_k"),
        "original_n_seeds": spec.get("original_n_seeds"),
        "alias_of": spec.get("alias_of"),
        "note": ("新旧口径**不可直接比较**：旧的是 "
                 f"{spec.get('original_n_seeds')} 种子 / 3~4 档；"
                 f"新的是 {len(seeds)} 种子 / {len(un)} 档（未饱和）。"
                 "k 的差异主要来自**分辨力提升**，不能归因为机制效应变化。"),
    }
    if bk.get("ok"):
        lo, hi = bk["ci95"]
        out.update({
            "k": bk["k_point"], "ci": [lo, hi], "ci_width": bk["width"],
            "contains_0_5": bool(lo <= 0.5 <= hi),
            "contains_1_0": bool(lo <= 1.0 <= hi),
            "significant_levels": bk["significant_levels"],
        })
    else:
        out.update({"k": None, "ci": None, "ci_width": None,
                    "fit_reason": bk.get("reason")})
    return out


# ======================================================================
def el2_asymmetry_verdict(res: dict) -> dict:
    """EL.2：**非对称是否比对称更凹** —— 本工作线最重要的产出。

    做法：比较 ``EA4_taker`` 与 ``EA4_both`` 的 k 的 95% 区间。

    ⚠️ 这不是"谁的 k 更小"就完事——被同一轮证伪过的正是"看点估计比大小"。
    这里用的是**区间判定**：
      · taker 的整个区间都小于 both 的下界 ⇒ taker 显著更凹
      · 两区间重叠 ⇒ **依然无法判定**（如实报告，不许因为"投入了更多资源"
        就暗示应该有答案）
      · taker 的区间整体高于 both ⇒ 对称显著更凹
    """
    t = res.get("EA4_taker") or {}
    b = res.get("EA4_both") or {}
    out: dict = {"taker": {"k": t.get("k"), "ci": t.get("ci")},
                 "both": {"k": b.get("k"), "ci": b.get("ci")}}
    if not (t.get("ci") and b.get("ci")):
        out.update({"verdict": "数据不足", "overlap": None})
        return out
    tl, th = t["ci"]
    bl, bh = b["ci"]
    overlap = max(0.0, min(th, bh) - max(tl, bl))
    narrower = min(th - tl, bh - bl)
    out.update({
        "overlap": overlap,
        "overlap_frac_of_narrower": (overlap / narrower) if narrower > 0 else None,
        "delta_k_point": (t["k"] - b["k"]) if (t.get("k") and b.get("k")) else None,
    })
    if th < bl:
        out["verdict"] = "非对称显著更凹"
    elif tl > bh:
        out["verdict"] = "对称显著更凹"
    else:
        out["verdict"] = "依然无法判定"
    return out


def el3_eb1_direction(res: dict) -> dict:
    """EL.3：元订单覆盖面扩大后，k 是否仍显示"改善"这个方向。"""
    base = res.get("EB1_chartist_only") or {}
    out: dict = {"baseline_k": base.get("k"), "arms": {}}
    for tag in ("EB1_plus_zero_intel", "EB1_plus_fundamentalist",
                "EB1_all_three"):
        a = res.get(tag) or {}
        d = {"k": a.get("k"), "ci": a.get("ci")}
        if base.get("k") is not None and a.get("k") is not None:
            d["delta_vs_baseline"] = a["k"] - base["k"]
            d["improved"] = bool(a["k"] < base["k"])
            # 方向是否有区间支撑：基线区间不与该臂区间重叠
            if base.get("ci") and a.get("ci"):
                d["ci_separated"] = bool(
                    a["ci"][1] < base["ci"][0] or a["ci"][0] > base["ci"][1])
        out["arms"][tag] = d
    return out


def el4_joint_direction(res: dict) -> dict:
    """EL.4：J2 是否仍优于 J3（"A 在 B 之上的边际贡献为负"）。"""
    j2 = res.get("EB1_all_three") or {}     # J2 与它同配置
    j3 = res.get("J3_plus_A") or {}
    out = {"j2_k": j2.get("k"), "j3_k": j3.get("k"),
           "j2_ci": j2.get("ci"), "j3_ci": j3.get("ci"),
           "note": "J2 与 EB1_all_three 是同一配置（去重后共用一次运行）"}
    if j2.get("k") is not None and j3.get("k") is not None:
        out["delta_j3_minus_j2"] = j3["k"] - j2["k"]
        out["j2_still_better"] = bool(j2["k"] < j3["k"])
        if j2.get("ci") and j3.get("ci"):
            out["ci_separated"] = bool(
                j3["ci"][1] < j2["ci"][0] or j3["ci"][0] > j2["ci"][1])
    return out


# ======================================================================
def main() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="只跑某一个家族（按名字）")
    ap.add_argument("--group", default=None, help="只跑某一组：ea4 / eb1 / joint")
    args = ap.parse_args()

    banner("工作线L  回填 7 档 × 8 种子标准配置")
    t0 = time.time()
    fams = families()
    ensure_scripts_on_path()
    from run_stage3 import SHOCK_SIZES
    seeds = backfill_seeds()

    out = _safe_load(RESULT)
    out.setdefault("stage", "L")
    out["seeds"] = seeds
    out["shock_sizes"] = list(SHOCK_SIZES)
    out.setdefault("families", {})
    print(f"  种子 {len(seeds)} 个：{seeds[:3]}…  档位 {len(SHOCK_SIZES)} 档  "
          f"slices={SLICES}")

    todo = [(n, s) for n, s in fams.items()
            if (not args.only or n == args.only)
            and (not args.group or s.get("group") == args.group)]
    if not todo:
        print("  ⚠️ 没有匹配的家族")
        return out

    for name, spec in todo:
        print(f"\n  【{name}】kind={spec['kind']}"
              f"{'  thin_scope=' + spec['thin_scope'] if 'thin_scope' in spec else ''}"
              f"  旧读数 k={spec.get('original_k')}（{spec.get('original_n_seeds')} 种子）")
        r = backfill_one_family(name, spec, seeds=seeds)
        out["families"][name] = r
        if r.get("k") is not None:
            lo, hi = r["ci"]
            print(f"    λ̄={r['lambda_bar'] if r['lambda_bar'] else '—'}  "
                  f"未饱和 {r['n_unsaturated']}/{r['n_levels_total']} 档"
                  f"（边界 {r.get('saturated_from')}）")
            print(f"    k = {r['k']:.3f}   95% CI = [{lo:.3f}, {hi:.3f}]"
                  f"  宽 {r['ci_width']:.3f}   含0.5={'Y' if r['contains_0_5'] else 'n'}"
                  f"   显著档 {r['significant_levels']}/{r['n_unsaturated']}")
        else:
            print(f"    ⚠️ 拟合未成功：{r.get('fit_reason')}")
        # 每跑完一个就落盘（长任务，中途崩了不丢已完成的部分）
        save_json(out, RESULT)

    res = out["families"]
    if res:
        el2 = el2_asymmetry_verdict(res)
        out["el2_asymmetry"] = el2
        out["el3_eb1"] = el3_eb1_direction(res)
        out["el4_joint"] = el4_joint_direction(res)
        if el2.get("verdict"):
            print(f"\n  ⭐ EL.2 判定：**{el2['verdict']}**"
                  f"（区间重叠 {el2.get('overlap')}）")
    out["elapsed_sec"] = time.time() - t0
    save_json(out, RESULT)
    print(f"\n  产物：out/{RESULT}（{out['elapsed_sec']:.0f}s）")
    return out


if __name__ == "__main__":
    main()
