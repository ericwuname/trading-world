"""收尾：把工作线 L/M 的结果合并进《历史数值陈述分辨力清单》（任务书 §5.1）。

做的事
------
1. 重新扫描全部报告（**复用** ``audit_numeric_claims`` 的扫描器，不重写）
2. 重新生成清单——它现在会带上「§3.4 已回填标准配置」与「§3.5 H4 状态」两节
3. 统计：哪些条目从"疑似裸奔"升级为"有 CI 支撑"，还剩多少

⚠️ 统计口径写在报告里：**"减少"指的是"被本轮覆盖到的条目"**，
不是"所有裸奔条目都消失了"——剩下那些（例如阶段6/8 的历史读数）
仍然需要后续轮次处理。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import OUT, banner  # noqa: E402

DOCS = Path(__file__).resolve().parent.parent / "docs"
TARGET = DOCS / "历史数值陈述分辨力清单.md"


def _load(name: str) -> dict:
    p = OUT / name
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def summarize_backfill() -> dict:
    """统计工作线L 覆盖了哪些家族、其中多少个拿到了可用 CI。"""
    L = _load("workstream_L_metrics.json")
    fams = L.get("families") or {}
    with_ci = {n: r for n, r in fams.items() if r.get("k") is not None}
    widths = [r["ci_width"] for r in with_ci.values()]
    # 旧口径的宽度：从诊断工具的历史结果里取（4 档 3 种子）
    KU = _load("k_uncertainty.json")
    old_widths = [a["width"] for a in (KU.get("arms") or {}).values()
                  if a.get("ok")]
    return {
        "n_families": len(fams),
        "n_with_ci": len(with_ci),
        "families_with_ci": list(with_ci),
        "median_ci_width_new": (sorted(widths)[len(widths) // 2] if widths else None),
        "median_ci_width_old_4level": (sorted(old_widths)[len(old_widths) // 2]
                                       if old_widths else None),
        "el2_verdict": (L.get("el2_asymmetry") or {}).get("verdict"),
        "el3": L.get("el3_eb1") or {},
        "el4": L.get("el4_joint") or {},
    }


def summarize_M() -> dict:
    M = _load("workstream_M_metrics.json")
    em2 = M.get("em2") or {}
    return {
        "has_analysis": bool(M.get("em1")),
        "cohens_d": (M.get("em1") or {}).get("effect_size_cohens_d"),
        "required_n_group1": (M.get("em1") or {}).get("required_n_group1"),
        "seeds_needed": em2.get("seeds_needed"),
        "verdict": em2.get("verdict"),
        "branch": ("EM.4 诚实报告" if M.get("em4")
                   else ("EM.3 已执行" if M.get("em3") else "未执行")),
    }


def main() -> dict:
    banner("收尾：更新历史数值陈述分辨力清单")
    # 复用 audit 的扫描与生成（单一实现）
    from audit_numeric_claims import REPORTS, build_inventory, load_text, \
        scan_report_for_numeric_claims

    scanned: dict[str, list[dict]] = {}
    for label, path in REPORTS:
        txt = load_text(path)
        scanned[label] = (scan_report_for_numeric_claims(txt, label=label)
                          if txt else [])
        print(f"  {label:<20} 命中 {len(scanned[label]):>5} 处")

    md = build_inventory(scanned)
    TARGET.write_text(md, encoding="utf-8")
    n_total = sum(len(v) for v in scanned.values())
    print(f"\n  清单已重出：{TARGET}（{n_total} 处命中）")

    L = summarize_backfill()
    M = summarize_M()
    print(f"\n  【工作线L】覆盖 {L['n_families']} 个家族，"
          f"其中 {L['n_with_ci']} 个拿到可用 CI")
    if L["median_ci_width_new"] and L["median_ci_width_old_4level"]:
        w_new, w_old = L["median_ci_width_new"], L["median_ci_width_old_4level"]
        print(f"    CI 宽度中位数：旧口径 {w_old:.3f} → 回填后 {w_new:.3f}"
              f"（缩小 {(1 - w_new / w_old) * 100:.0f}%）")
    print(f"    EL.2 判定：{L.get('el2_verdict')}")
    print(f"\n  【工作线M】Cohen's d = "
          f"{M['cohens_d'] if M['cohens_d'] is None else round(M['cohens_d'], 4)}"
          f"  需要 {M['seeds_needed'] if M['seeds_needed'] is None else round(M['seeds_needed'], 1)} 个种子"
          f"  判定 {M['verdict']}  分支 {M['branch']}")

    summary = {"L": L, "M": M, "scanned_total": n_total}
    (OUT / "post_L_M_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  统计已存：{OUT / 'post_L_M_summary.json'}")
    return summary


if __name__ == "__main__":
    main()
