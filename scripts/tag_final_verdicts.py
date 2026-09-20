"""工作线P：正式把「方向性偏离检验」写成项目验收标准。

任务书：`交易世界 · 一致性审计与收尾任务书.md` §3。

为什么要改标准
--------------
原来的目标是「k 要接近 0.5」（精确逼近平方根律）。
但两轮工作证明：

  · 「是否**精确**逼近 0.5」这个要求，在当前装置下**做不到**——
    分辨它需要约 300~1000 个种子（见 `docs/前置校验-EA4-诊断报告.md`
    与 `docs/分辨力危机-交付小结.md`）；
  · 「是否**显著偏离**平方根律」（CI 是否排除 0.5）这个**方向性**要求，
    在回填后的 8 个家族里有 6 个已经给出了确定答案。

⇒ 新标准：**CI 排除 0.5 即可确认显著偏离平方根律**（方向性判据），
   不再要求点估计接近某个精确值。

⚠️ 必须分清两件不同的事（任务书 §3.2 明确要求）
--------------------------------------------
  · **单个家族自己的方向判定** —— 只依赖这个家族自己的 CI，
    **不受一致性审计（工作线O）影响**；
  · **家族之间的比较判定** —— 依赖两个 CI 的重叠，
    工作线O 判定「无法判定」的比较**必须撤回**。

`retag_all_families` 的输出里把这两类分开标注。

用法::

    python scripts/tag_final_verdicts.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import OUT, banner, ensure_scripts_on_path, load_json, save_json  # noqa: E402

RESULT = "final_verdicts.json"


def _load(name: str) -> dict:
    p = OUT / name
    if not p.exists():
        return {}
    try:
        return load_json(name) or {}
    except Exception:  # noqa: BLE001
        return {}


def tag_all_families() -> dict:
    """EP.1：对全部已回填家族 + 阶段3 原始数据打**方向性**标签。"""
    ensure_scripts_on_path()
    from tw.analyzer_consistency import direction_verdict

    rows = []
    L = _load("workstream_L_metrics.json")
    for name, r in (L.get("families") or {}).items():
        if r.get("k") is None:
            continue
        lo, hi = r["ci"]
        tag = direction_verdict(r["k"], (lo, hi))
        rows.append({
            "family": name, "source": "工作线L 回填（8 种子 × 未饱和档）",
            "n_seeds": r.get("n_seeds"), "n_unsaturated": r.get("n_unsaturated"),
            "k": r["k"], "ci": [lo, hi], "ci_width": r["ci_width"],
            "tag": tag,
            "excludes_0_5": not (lo <= 0.5 <= hi),
            "excludes_1_0": not (lo <= 1.0 <= hi),
            "original_k": r.get("original_k"),
        })

    # 阶段3 原始审计（工作线J）—— 未饱和口径只有 3 档，用于对照"旧配置下怎样"
    J = _load("workstream_J_metrics.json")
    ej2 = J.get("ej2") or {}
    if ej2.get("ok"):
        lo, hi = ej2["ci95"]
        rows.append({
            "family": "阶段3 C2_staged（原始审计）",
            "source": f"工作线J（{ej2.get('n_levels')} 档未饱和 × "
                      f"{ej2.get('n_seeds')} 种子）",
            "n_seeds": ej2.get("n_seeds"), "n_levels": ej2.get("n_levels"),
            "k": ej2["k_point"], "ci": [lo, hi],
            "ci_width": ej2["width"], "tag": direction_verdict(ej2["k_point"], (lo, hi)),
            "excludes_0_5": bool(ej2.get("covers_0.5_sqrt_law") is False),
            "excludes_1_0": bool(ej2.get("covers_1.0_linear") is False),
        })

    n_excluded = sum(1 for r in rows if r["excludes_0_5"])
    summary = {
        "n_families": len(rows),
        "n_sqrt_law_excluded": n_excluded,
        "n_sqrt_law_consistent": sum(1 for r in rows if r["tag"] == "sqrt_law_consistent"),
        "n_undetermined": sum(1 for r in rows if r["tag"] == "undetermined"),
        "note": ("方向性判据：CI 排除 0.5 即确认显著偏离平方根律。"
                 "这与「k 精确接近 0.5」是**两个不同强度的要求**——"
                 "前者在回填后的配置下多数可判定，后者做不到。"),
    }
    return {"rows": rows, "summary": summary}


def attach_comparison_status(tags: dict, audit: dict) -> dict:
    """EP.3：把工作线O 对「比较类结论」的判定挂上来（两类不混）。"""
    comps = {c["id"]: c for c in (audit.get("comparisons") or [])}
    retracted = {i["id"] for i in
                 ((audit.get("eo6_retractions") or {}).get("items") or [])}
    out = []
    for cid, c in comps.items():
        out.append({
            "id": cid, "label": c["label"],
            "verdict": c["verdict"],
            "overlap_fraction": c["overlap_fraction"],
            "required_n": c.get("required_n"),
            "retracted": cid in retracted,
        })
    return {
        "comparisons": out,
        "n_retracted": len(retracted),
        "note": ("⚠️ 这两类判定**互相独立**："
                 "某个家族自己的方向标签（上表）**不因比较被撤回而失效**；"
                 "被撤回的只是「A 比 B 更好/更差」这类**家族之间**的说法。"),
    }


def render_table(tags: dict) -> list[str]:
    rows = tags.get("rows") or []
    out = ["| 家族 | 配置 | k | 95% CI | 宽度 | 方向性判定 | 旧读数 |",
           "|---|---|---|---|---|---|---|"]
    for r in rows:
        src = r.get("source", "")
        short = ("回填 8 种子" if "工作线L" in src else src)
        lo, hi = r["ci"]
        out.append(f"| `{r['family']}` | {short} | {r['k']:.3f} | "
                   f"[{lo:.3f}, {hi:.3f}] | {r['ci_width']:.3f} | "
                   f"**{r['tag']}** | {r.get('original_k', '—')} |")
    return out


def main() -> dict:
    banner("工作线P  方向性偏离检验 → 正式验收标准")
    t0 = time.time()
    tags = tag_all_families()
    audit = _load("workstream_O_metrics.json")
    comp = attach_comparison_status(tags, audit)

    print("\n【EP.1】全部家族的方向性判定")
    for ln in render_table(tags):
        print("  " + ln if ln.startswith("|") else ln)
    s = tags["summary"]
    print(f"\n  汇总：{s['n_families']} 个家族中，"
          f"**{s['n_sqrt_law_excluded']} 个排除 0.5**（= 确认显著偏离平方根律）、"
          f"{s['n_sqrt_law_consistent']} 个与平方根律相容、"
          f"{s['n_undetermined']} 个无法判断")
    print(f"\n【EP.3】比较类结论的状态（来自工作线O）")
    for c in comp["comparisons"]:
        flag = "❌ 已撤回" if c["retracted"] else "✅ 保留"
        print(f"  {c['id']} {c['label']:<22} {c['verdict']:<8} {flag}")
    print(f"\n  {comp['note']}")

    result = {"tags": tags, "comparisons_status": comp,
              "elapsed_sec": time.time() - t0}
    save_json(result, RESULT)
    print(f"\n  产物：out/{RESULT}（{result['elapsed_sec']:.0f}s）")
    return result


if __name__ == "__main__":
    main()
