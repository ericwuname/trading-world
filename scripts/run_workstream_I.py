"""工作线I：路径B —— 凹度判据体系（新的验收标准候选）。

任务书：`交易世界 · 分辨力危机应对任务书.md` §3。

判据本体在 ``tw/analyzer_concavity.py``（相邻档位局部弹性 + 三分类判决）。
本脚本负责四件事：

    EI.1  用**现有** 4 档数据重新判读（零成本，最先做）
    EI.2  结合工作线H 的 7 档数据（更大档位、更多档位对）
    EI.3  跨配置对比：**非对称（taker）是否比对称（both）更显著凹**
          ——这是把 EA.4「非对称更好」改写成不依赖 k 的可判定形式
    EI.4  历史结论重新判读（EB.1 四臂、J1–J3 三臂），不重跑模拟

⚠️ 两条来自任务书 §3.6 的硬要求，本脚本按它们执行：
  1. **全部测试过的档位对都要报出来**，不许只挑显著的那几对展示。
     （这正是诊断报告揭示的问题的翻版：用局部信息冒充整体结论。）
  2. 种子数少时用 **t 分布**临界值，不用正态的 1.96。

用法::

    python scripts/run_workstream_I.py --only EI1
    python scripts/run_workstream_I.py            # 全部（EI.2 需要先跑工作线H）
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import OUT, banner, ensure_scripts_on_path, load_json, save_json  # noqa: E402
from tw.analyzer_concavity import (  # noqa: E402
    VERDICT_CONCAVE,
    VERDICT_LINEAR_OR_WORSE,
    VERDICT_UNDECIDED,
    concavity_verdict,
    pairwise_local_elasticity,
)

RESULT = "workstream_I_metrics.json"


def _safe_load(name: str) -> dict:
    """读 ``out/<name>``；**不存在时返回空 dict 而不是抛异常**。

    ⚠️ ``_common.load_json`` 在文件缺失时会直接 ``FileNotFoundError``——
    那是它的设计选择（调用方负责）。但本脚本有"分段跑、合并写回"的用法
    （``--only EI1`` 时产物还不存在），所以这里必须自己兜住：
    本轮实测就是在这上面第一次跑直接崩了。
    """
    p = OUT / name
    if not p.exists():
        return {}
    try:
        return load_json(name) or {}
    except Exception:  # noqa: BLE001  解析失败也不该让整条工作线停摆
        return {}


# ======================================================================
def all_pairs(rows: list[dict], *, adjacent_only: bool = False) -> list[dict]:
    """对一臂的逐档 rows 做**全部**档位对比较（默认含跨档对）。

    任务书 §3.6 明确要求不许挑档位对，所以默认就是全组合 C(n,2)。
    每一对都记录：两端档位、弹性、区间、判决、是否低信噪比。
    """
    out = []
    idx = list(range(len(rows)))
    pairs = ([(i, i + 1) for i in range(len(rows) - 1)] if adjacent_only
             else list(itertools.combinations(idx, 2)))
    for i, j in pairs:
        a, b = rows[i], rows[j]
        res = pairwise_local_elasticity(
            a["shock_frac"], b["shock_frac"],
            mean_small=a["slippage_bp"], sem_small=a.get("slippage_sem_bp"),
            mean_big=b["slippage_bp"], sem_big=b.get("slippage_sem_bp"),
            n_seeds=int(a.get("n_seeds", 0)))
        verdict, ci = concavity_verdict(res)
        out.append({
            "i_small": i, "i_big": j,
            "size_small": a["shock_frac"], "size_big": b["shock_frac"],
            "span": (j - i),
            "ok": res.ok, "reason": res.reason,
            "e": res.mean, "sem": res.sem,
            "ci_lower": ci[0], "ci_upper": ci[1],
            "verdict": verdict,
            "t_vs_linear": res.t_vs_linear, "p_vs_linear": res.p_vs_linear,
            "t_vs_sqrt": res.t_vs_sqrt, "p_vs_sqrt": res.p_vs_sqrt,
            "low_snr": res.low_snr, "method": res.method,
            "n_seeds": res.n_seeds,
        })
    return out


def verdict_counts(pairs: list[dict]) -> dict:
    """判决分布——EI.3 靠它做跨配置比较。"""
    c = {VERDICT_CONCAVE: 0, VERDICT_LINEAR_OR_WORSE: 0, VERDICT_UNDECIDED: 0}
    for p in pairs:
        c[p["verdict"]] = c.get(p["verdict"], 0) + 1
    return c


def _collect_arms(source_keys: list[tuple[str, str, str]]) -> dict:
    """从若干产物里取出「臂名 → 逐档 rows」。

    ``source_keys`` 每一项是 ``(json文件名, 顶层键, 臂名)``。
    取不到就跳过并在结果里留痕，不静默丢（否则报告会少一臂而看不出来）。
    """
    cache: dict[str, dict] = {}
    arms: dict[str, dict] = {}
    for fname, block, arm in source_keys:
        if fname not in cache:
            cache[fname] = _safe_load(fname)
        d = cache[fname]
        rows = ((d.get(block) or {}).get("arms") or {}).get(arm, {}).get("rows")
        if not rows:
            arms[f"{block}|{arm}"] = {"rows": [], "missing": True}
            continue
        arms[f"{block}|{arm}"] = {"rows": rows, "missing": False}
    return arms


# ======================================================================
def ei1_existing_data() -> dict:
    """EI.1：用现有 4 档数据判读（零成本，不重跑模拟）。"""
    print("\n【EI.1】用现有 4 档数据重新判读（不重跑）")
    arms = _collect_arms([
        ("workstream_A_metrics.json", "ea1_uncorrected", "Hawkes 关"),
        ("workstream_A_metrics.json", "ea4_one_sided", "taker"),
        ("workstream_A_metrics.json", "ea4_one_sided", "both"),
        ("workstream_A_metrics.json", "ea4_one_sided", "maker"),
    ])
    out: dict = {"source": "out/workstream_A_metrics.json（4 档 × 3 种子）", "arms": {}}
    for name, d in arms.items():
        if d["missing"]:
            out["arms"][name] = {"missing": True}
            continue
        pairs = all_pairs(d["rows"])
        out["arms"][name] = {
            "n_levels": len(d["rows"]), "pairs": pairs,
            "verdicts": verdict_counts(pairs),
            "n_low_snr": sum(1 for p in pairs if p["low_snr"]),
        }
        v = out["arms"][name]["verdicts"]
        print(f"  {name:<34} 档位={len(d['rows'])}  对数={len(pairs)}  "
              f"判决：凹={v[VERDICT_CONCAVE]} 线性/更差={v[VERDICT_LINEAR_OR_WORSE]} "
              f"不可判定={v[VERDICT_UNDECIDED]}")
    return out


def ei2_seven_levels() -> dict:
    """EI.2：用工作线H 的 7 档数据（更大档位、更多档位对）。"""
    print("\n【EI.2】工作线H 的 7 档数据")
    h = _safe_load("workstream_H_metrics.json")
    if not h or not h.get("configs"):
        print("  ⚠️ 缺 out/workstream_H_metrics.json——先跑 scripts/run_workstream_H.py")
        return {"missing": True}
    out: dict = {"source": "out/workstream_H_metrics.json", "arms": {}}
    for cfg, d in h["configs"].items():
        rows = [r for r in d["rows"] if not r["saturated"]]
        if len(rows) < 2:
            out["arms"][cfg] = {"missing": True,
                                "reason": f"未饱和档位仅 {len(rows)} 档，无法做弹性"}
            print(f"  {cfg:<24} 未饱和只有 {len(rows)} 档 → 无法比较")
            continue
        # ⚠️ 字段名对齐：H 的 rows 用的是 slippage_mean/slippage_sem
        norm = [{"shock_frac": r["size"], "slippage_bp": r["slippage_mean"],
                 "slippage_sem_bp": r["slippage_sem"], "n_seeds": r["n_seeds"]}
                for r in rows]
        pairs = all_pairs(norm)
        out["arms"][cfg] = {"n_levels": len(rows), "pairs": pairs,
                            "verdicts": verdict_counts(pairs),
                            "n_low_snr": sum(1 for p in pairs if p["low_snr"]),
                            "saturated_from": d.get("saturated_from")}
        v = out["arms"][cfg]["verdicts"]
        print(f"  {cfg:<24} 未饱和={len(rows)} 对={len(pairs)}  "
              f"凹={v[VERDICT_CONCAVE]} 线性/更差={v[VERDICT_LINEAR_OR_WORSE]} "
              f"不可判定={v[VERDICT_UNDECIDED]}")
    return out


def ei3_cross_config(ei2: dict) -> dict:
    """EI.3：**非对称是否比对称更显著凹**——本工作线最重要的产出。

    做法不是"看谁的点估计小"（那正是被推翻的做法），而是比较：
      · 各自的判决分布（显著凹占多少）
      · 在**同一对档位**上，两者的弹性差是否显著
    第二项尤其重要：同一对档位、同一个种子集合，做两样本 t 检验才是配得上的比较。
    """
    print("\n【EI.3】跨配置对比：非对称 vs 对称是否更凹")
    arms = (ei2 or {}).get("arms") or {}
    if not arms or any(a.get("missing") for a in arms.values()):
        print("  ⚠️ EI.2 数据不完整，跳过")
        return {"missing": True}
    out: dict = {"arms": {k: {"verdicts": v.get("verdicts"),
                              "n_pairs": len(v.get("pairs") or []),
                              "n_low_snr": v.get("n_low_snr")}
                          for k, v in arms.items()}}

    def _pairs_by_span(a):
        return {(p["i_small"], p["i_big"]): p for p in (a.get("pairs") or [])
                if p["ok"]}

    taker, both = arms.get("ea4_taker_only"), arms.get("ea2_both_symmetric")
    if taker and both:
        pt, pb = _pairs_by_span(taker), _pairs_by_span(both)
        shared = sorted(set(pt) & set(pb))
        rows = []
        for key in shared:
            x, y = pt[key], pb[key]
            # 两个独立臂的弹性差：均值差 / 合成标准误
            se = (x["sem"] ** 2 + y["sem"] ** 2) ** 0.5
            t_diff = ((x["e"] - y["e"]) / se) if se > 0 else float("nan")
            rows.append({
                "pair": f"{x['size_small']}→{x['size_big']}",
                "e_taker": x["e"], "e_both": y["e"],
                "delta": x["e"] - y["e"], "se_delta": se, "t_delta": t_diff,
                # 更凹 = 弹性更小 = delta < 0
                "taker_more_concave": bool(x["e"] < y["e"]),
                "significant": bool(abs(t_diff) > 2.0) if se > 0 else False,
            })
        out["taker_vs_both"] = {
            "pairs": rows,
            "n_pairs": len(rows),
            "n_taker_more_concave": sum(1 for r in rows if r["taker_more_concave"]),
            "n_significant": sum(1 for r in rows if r["significant"]),
            "n_significant_and_taker_more_concave": sum(
                1 for r in rows if r["significant"] and r["taker_more_concave"]),
        }
        d = out["taker_vs_both"]
        print(f"  配对档位 {d['n_pairs']} 对；taker 更凹的 {d['n_taker_more_concave']} 对；"
              f"其中差异显著 {d['n_significant_and_taker_more_concave']} 对")
        for r in rows:
            print(f"    {r['pair']:<18} e_taker={r['e_taker']:.3f} "
                  f"e_both={r['e_both']:.3f}  Δ={r['delta']:+.3f} "
                  f"t={r['t_delta']:+.2f}"
                  f"{'  ← taker 更凹' if r['taker_more_concave'] else ''}")
    else:
        out["taker_vs_both"] = {"missing": True,
                                "reason": "缺 taker 或 both 臂"}
        print("  ⚠️ 缺 taker/both 臂之一，无法比较")
    return out


def ei4_history() -> dict:
    """EI.4：用凹度判据重新判读历史结论（EB.1 四臂、J1–J3 三臂）。"""
    print("\n【EI.4】历史结论重新判读（换尺子读旧数据，不重跑）")
    arms = _collect_arms([
        ("workstream_B_metrics.json", "eb1", "仅图表派(阶段6基线)"),
        ("workstream_B_metrics.json", "eb1", "+零智能"),
        ("workstream_B_metrics.json", "eb1", "+基本面派"),
        ("workstream_B_metrics.json", "eb1", "三者全部"),
        ("workstream_joint_metrics.json", "ej1", "J1 阶段6 基线"),
        ("workstream_joint_metrics.json", "ej1", "J2 +B 三类覆盖"),
        ("workstream_joint_metrics.json", "ej1", "J3 +A 修正 Hawkes"),
    ])
    out: dict = {"arms": {}}
    for name, d in arms.items():
        if d["missing"]:
            out["arms"][name] = {"missing": True}
            continue
        pairs = all_pairs(d["rows"])
        out["arms"][name] = {"n_levels": len(d["rows"]), "pairs": pairs,
                             "verdicts": verdict_counts(pairs),
                             "n_low_snr": sum(1 for p in pairs if p["low_snr"])}
        v = out["arms"][name]["verdicts"]
        print(f"  {name:<30} 凹={v[VERDICT_CONCAVE]} "
              f"线性/更差={v[VERDICT_LINEAR_OR_WORSE]} 不可判定={v[VERDICT_UNDECIDED]}")
    return out


# ======================================================================
def main() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="EI1 / EI2 / EI3 / EI4")
    args = ap.parse_args()
    banner("工作线I  路径B —— 凹度判据体系")
    t0 = time.time()
    only = (args.only or "").upper()
    out: dict = {"stage": "I",
                 "note": "全部档位对都记录在产物里；报告不许只挑显著的那几对"}
    if not only or only == "EI1":
        out["ei1"] = ei1_existing_data()
    if not only or only in ("EI2", "EI3"):
        out["ei2"] = ei2_seven_levels()
        out["ei3"] = ei3_cross_config(out.get("ei2") or {})
    if not only or only == "EI4":
        out["ei4"] = ei4_history()
    out["elapsed_sec"] = time.time() - t0

    # 与前一次产物合并写回（EI1/EI4 与 EI2/EI3 是分开跑的）
    prev = _safe_load(RESULT)
    prev.update(out)
    save_json(prev, RESULT)
    print(f"\n  产物：out/{RESULT}（{out['elapsed_sec']:.0f}s）")
    return prev


if __name__ == "__main__":
    ensure_scripts_on_path()
    main()
