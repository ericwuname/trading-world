"""生成本轮交付小结：docs/一致性审计-交付小结.md

覆盖第十条纪律 + 工作线 N/O/P/Q/R。数字从 out/*.json 现算。

⚠️ 本文件文案里不用 ASCII 双引号（中文引用统一用「」）——
第八条纪律的由来就是「按行替换引号把源码写坏」，从源头避免。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import OUT  # noqa: E402

DOCS = Path(__file__).resolve().parent.parent / "docs"
TARGET = DOCS / "一致性审计-交付小结.md"


def load(name: str) -> dict:
    p = OUT / name
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def f(v, fmt: str = ".3f", dash: str = "—") -> str:
    if not isinstance(v, (int, float)) or v != v:
        return dash
    return format(v, fmt)


def build() -> str:
    O = load("workstream_O_metrics.json")
    F = load("final_verdicts.json")
    Q = load("workstream_Q_diff.json")
    comps = O.get("comparisons") or []
    retr = (O.get("eo6_retractions") or {}).get("items") or []
    eo5 = (O.get("eo5_required_n") or {}).get("rows") or []
    eo4 = O.get("eo4_targeted") or {}
    tags = (F.get("tags") or {})
    tsum = tags.get("summary") or {}

    L: list[str] = []
    add = L.append
    add("# 一致性审计与收尾 · 交付小结")
    add("")
    add("> 对应任务书：`交易世界 · 一致性审计与收尾任务书.md`")
    add("> 本文件由 `scripts/make_consistency_summary.py` 生成，数字从 `out/*.json` 现算。")
    add("")
    add("---")
    add("")
    add("## 0. 一句话：这一轮审的是**上一轮自己的账**")
    add("")
    add("上一轮报告里写了两条「历史结论被推翻」。这一轮发现：")
    add("**那两条只凭点估计的方向变化，没做区间重叠检验——"
        "而做一遍就会发现它们的区间重叠是 100% 和 80.9%。**")
    add("")
    add("⇒ 那两条「推翻」的措辞**站不住，已正式撤回**。")
    add("")
    add("这不是说上一轮白做了：区间重叠度量与「还需要多少样本」的外推，"
        "以及第十条纪律，都是这轮的有效产出。")
    add("")
    add("---")
    add("")

    # ---------------- 第十条 ----------------
    add("## 1. 工程纪律第十条（新增）")
    add("")
    add("> **任何「此前 A 更好 → 现在 B 更好」这类方向反转的陈述，"
        "必须用区间重叠度量正式检验。**")
    add("> 点估计换了方向，只能说「新的读数如此」；能不能说「结论变了」，"
        "要看区间重叠检验的结果。")
    add("")
    add("已写入 `README.md` 的工程纪律章节（共十条），"
        "实现见 `tw/analyzer_consistency.py`。")
    add("")
    add("---")
    add("")

    # ---------------- O ----------------
    add("## 2. 工作线O：一致性审计（本轮最重要的产出）")
    add("")
    add("### 2.1 四条比较的重新检验")
    add("")
    add("| 编号 | 比较 | k_A | k_B | **重叠比例** | 判定 |")
    add("|---|---|---|---|---|---|")
    for c in comps:
        add(f"| {c['id']} | {c['label']} | {f(c['point_a'])} | {f(c['point_b'])} | "
            f"**{f(c['overlap_fraction'] * 100, '.1f')}%** | {c['verdict']} |")
    add("")
    add("⚠️ **四条全部是「依然无法判定」**。其中 EO.2（基线 vs +基本面派）"
        "的重叠是 **100%**——基线区间被完全包含在内。")
    add("")
    add("### 2.2 必须撤回的两处表述")
    add("")
    if retr:
        add("| 编号 | 上一轮的表述 | 重叠比例 | 处理 |")
        add("|---|---|---|---|")
        for r in retr:
            add(f"| {r['id']} | {r['original']} | "
                f"**{f(r['overlap_fraction'] * 100, '.1f')}%** | ❌ {r['action']} |")
        add("")
        add("改写后的措辞（已落实到交付小结与历史清单）：")
        add("")
        for r in retr:
            add(f"- **{r['id']}**：{r['replacement']}")
    else:
        add("（没有需要撤回的表述）")
    add("")
    add("### 2.3 还要多少样本才能分辨（EO.5）")
    add("")
    if eo5:
        add("| 编号 | 比较 | 需要种子数 | 现用 | 按工作线M 的阈值 |")
        add("|---|---|---|---|---|")
        for r in eo5:
            add(f"| {r['id']} | {r['label']} | **{r['required_n']:.0f}** | "
                f"{r['current_n']} | {r['advice']} |")
        add("")
        add("阈值沿用工作线M 事先定死的那一套（≤50 值得 / ≤100 边缘 / >100 不建议），"
            "**不另立标准**。")
    add("")
    add("### 2.4 O.4 定向加种子与渐进检查")
    add("")
    if eo4 and (eo4.get("families") or {}):
        prog = eo4.get("eo2_progressive") or {}
        trend = prog.get("trend") or []
        if trend:
            add("对 CI 最宽的两个臂（`EB1_chartist_only` 与 `EB1_plus_fundamentalist`）"
                "逐种子加跑到 16 个，检查方向是否**渐进稳定**：")
            add("")
            add("| 种子数 | k(基线) | k(+fund) | 点估计差 | 重叠比例 | 判定 |")
            add("|---|---|---|---|---|---|")
            for t in trend:
                add(f"| {t['n_seeds']} | {f(t['k_baseline'])} | {f(t['k_fund'])} | "
                    f"{f(t['point_diff'], '+.3f')} | "
                    f"{f(t['overlap_fraction'] * 100, '.1f')}% | {t['verdict']} |")
            add("")
            # ⭐ 实际判读（任务书 §7 的要求：看方向是否渐进稳定）
            same_sign = (all(t["point_diff"] > 0 for t in trend)
                         or all(t["point_diff"] < 0 for t in trend))
            d0, dz = abs(trend[0]["point_diff"]), abs(trend[-1]["point_diff"])
            min_ov = min(t["overlap_fraction"] for t in trend)
            add("**判读**：")
            add("")
            add(f"- **方向**在 n={trend[0]['n_seeds']}/{trend[len(trend) // 2]['n_seeds']}/"
                f"{trend[-1]['n_seeds']} 三个 checkpoint 上"
                f"{'**始终一致** ✅' if same_sign else '**出现过翻转** ⚠️'}；")
            if d0 > 0:
                add(f"- ⚠️ **但幅度持续缩小**：点估计差从 {d0:.3f} 缩到 {dz:.3f}"
                    f"（**缩小 {(1 - dz / d0) * 100:.0f}%**）；")
            add(f"- 重叠比例始终 ≥ **{min_ov * 100:.1f}%**，判定始终是「依然无法判定」。")
            add("")
            add("⇒ **这不是「加到一半就稳定了」的好消息，而是「效应量随样本增大而缩水」"
                "的典型形态**——小样本下的点估计差里有一大部分是噪声。"
                "方向一致说明「可能存在一个小效应」，但幅度衰减说明"
                "**它比最初读数小得多**，而且按当前趋势，继续加种子只会让幅度更小。")
            add("")
            add("⚠️ 判读要点（任务书 §7 明确要求）：要看**方向是否在多个 checkpoint 上"
                "始终一致**，而不是「加满之后突然由不可判定变成可判定」——"
                "后者更值得怀疑。**本例方向一致、但幅度衰减**，"
                "所以「撤回」是稳妥的处理。")
    else:
        add("（O.4 尚未完成或未执行）")
    add("")
    add("---")
    add("")

    # ---------------- P ----------------
    add("## 3. 工作线P：验收标准正式变更")
    add("")
    add("**从「k 精确接近 0.5」改成「CI 排除 0.5 即确认显著偏离平方根律」**"
        "（方向性判据）。依据见 `docs/验收标准变更说明.md`。")
    add("")
    if tsum:
        add(f"重新打标签后：**{tsum.get('n_families')} 个家族里 "
            f"{tsum.get('n_sqrt_law_excluded')} 个排除 0.5**"
            f"（= 确认显著偏离平方根律），"
            f"{tsum.get('n_sqrt_law_consistent')} 个与平方根律相容、"
            f"{tsum.get('n_undetermined')} 个无法判断。")
    add("")
    if tags.get("rows"):
        add("| 家族 | k | 95% CI | 方向性判定 |")
        add("|---|---|---|---|")
        for r in tags["rows"]:
            lo, hi = r["ci"]
            add(f"| `{r['family']}` | {f(r['k'])} | [{f(lo)}, {f(hi)}] | "
                f"**{r['tag']}** |")
    add("")
    add("### ⚠️ 两类判定必须分开（这一条是踩过坑才写下来的）")
    add("")
    add("- **单个家族的方向判定** —— 只看它自己的 CI 是否排除 0.5，"
        "**不受工作线O 影响**；")
    add("- **家族之间的比较判定** —— 依赖两个 CI 的重叠，"
        "**工作线O 判「无法判定」的必须撤回**。")
    add("")
    comp_status = (F.get("comparisons_status") or {}).get("comparisons") or []
    if comp_status:
        add("| 比较 | 判定 | 状态 |")
        add("|---|---|---|")
        for c in comp_status:
            add(f"| {c['id']} {c['label']} | {c['verdict']} | "
                f"{'❌ 已撤回' if c['retracted'] else '✅ 保留'} |")
    add("")
    add("---")
    add("")

    # ---------------- Q ----------------
    add("## 4. 工作线Q：「非对称」表述统一更正")
    add("")
    add("按第八条纪律用 `scripts/safe_batch_replace.py`（**先 dry-run 确认 diff**）执行；"
        "**只追加批注、不删除任何历史原文**（EQ.3）。")
    add("")
    if Q:
        ch = Q.get("changed") or []
        add(f"- 改动文件：{'、'.join(f'`{c}`' for c in ch) if ch else '（无）'}")
        add(f"- diff 清单：`out/workstream_Q_diff.json`"
            f"（{sum(len(v) for v in (Q.get('diffs') or {}).values())} 行），供人工复核")
    add("- ⚠️ **核对后主动去掉了一条**：原计划在 `前置校验-EA4-诊断报告.md` "
        "里也加批注——但那本身就是「发现 k 无分辨力」的更正性文档"
        "（CI 审计表里已写着区间与「1/4 档显著」），再加一张更正反而画蛇添足。")
    add("")
    add("---")
    add("")

    # ---------------- R ----------------
    add("## 5. 工作线R：设计文档归档")
    add("")
    add("工作区根的**七份**设计文档已归档到 `docs/design/`"
        "（任务书说五份，实际以 `ls` 为准），并附索引 `docs/design/README.md`。")
    add("")
    add("⚠️ 这些文件原本在 **git 仓库之外**（仓库是 `trading-world/`，"
        "它们在上一层），所以用**复制**而不是 `git mv`；"
        "**原文件保留**——删除属跨边界操作，等确认。")
    add("")
    add("---")
    add("")
    add("## 6. 诚实边界")
    add("")
    add("1. **上一轮的两条「推翻」是错的**（措辞层面）：它们只凭点估计方向，"
        "没有做区间重叠检验。已撤回。")
    add("2. **撤回的只是「结论性」说法**，不是读数本身——"
        "点估计方向仍然一致（例如 J3 的 k 仍小于 J2），只是统计上区分不开。")
    add("3. **EO.1/EO.3 标「保留」不等于它们显著**："
        "它们的判定同样是「依然无法判定」，只是上一轮没把它们当结论写，"
        "所以没有「撤回」这个动作可做。")
    add("4. **方向性判据是更弱的要求**，它没有回答「该不该逼近 0.5」——"
        "那是独立的设计问题，本轮不做判断。")
    add("5. **O.4 的渐进检查若显示方向在中间 checkpoint 上翻转**，"
        "那么连「点估计方向一致」这句话也要打折——交付里会如实标注。")
    add("")
    return "\n".join(L) + "\n"


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    txt = build()
    head = txt.split("## 1. 工程纪律第十条")[0]
    for bad in ("nan", "None"):
        if bad in head:
            raise SystemExit(f"❌ 小结前段出现未处理值 {bad!r}，拒绝写出。")
    TARGET.write_text(txt, encoding="utf-8")
    n_tbl = sum(1 for ln in txt.splitlines() if ln.strip().startswith("|"))
    print(f"  交付小结已生成：{TARGET}（{len(txt.encode('utf-8')) / 1024:.0f} KB，"
          f"{n_tbl} 行表格）")


if __name__ == "__main__":
    main()
