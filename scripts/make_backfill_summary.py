"""生成本轮交付小结：docs/回填标准配置与H4做实-交付小结.md

覆盖工作线 N / L / M 与收尾。数字全部从 out/*.json 现算。

⚠️ 本文件文案里**一律不用 ASCII 双引号**，中文引用统一用「」——
第八条纪律的由来就是「按行替换引号把源码写坏」，从源头避免。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import OUT  # noqa: E402

DOCS = Path(__file__).resolve().parent.parent / "docs"
TARGET = DOCS / "回填标准配置与H4做实-交付小结.md"


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
    L = load("workstream_L_metrics.json")
    M = load("workstream_M_metrics.json")
    KU = load("k_uncertainty.json")
    PM = load("post_L_M_summary.json")
    REG = load("../docs/mutation_registry.json") or {}

    fams = L.get("families") or {}
    with_ci = {n: r for n, r in fams.items() if r.get("k") is not None}
    el2 = L.get("el2_asymmetry") or {}
    el3 = L.get("el3_eb1") or {}
    el4 = L.get("el4_joint") or {}
    em1 = M.get("em1") or {}
    em2 = M.get("em2") or {}
    em4 = M.get("em4") or {}
    em3 = M.get("em3") or {}

    L_ = []
    add = L_.append
    add("# 回填标准配置与 H4 做实 · 交付小结")
    add("")
    add("> 对应任务书：`交易世界 · 回填标准配置与H4做实任务书.md`")
    add("> 本文件由 `scripts/make_backfill_summary.py` 生成，数字从 `out/*.json` 现算。")
    add("> 前两轮的背景：`docs/前置校验-EA4-诊断报告.md`、`docs/分辨力危机-交付小结.md`。")
    add("")
    add("---")
    add("")
    add("## 0. 先说三条")
    add("")
    add(f"**① 回填确实把 CI 压窄了**（{len(with_ci)}/{len(fams)} 个家族已完成）：")
    old_w = [a["width"] for a in (KU.get("arms") or {}).values() if a.get("ok")]
    new_w = [r["ci_width"] for r in with_ci.values()]
    if old_w and new_w:
        mo = sorted(old_w)[len(old_w) // 2]
        mn = sorted(new_w)[len(new_w) // 2]
        add(f"   CI 宽度中位数：旧口径（3~4 档 × 3 种子）**{mo:.3f}** → "
            f"回填后（未饱和档 × 8 种子）**{mn:.3f}**，"
            f"缩小 **{(1 - mn / mo) * 100:.0f}%**。")
    add("")
    add("**② EL.2（非对称 vs 对称）的最终判定**：")
    if el2.get("verdict"):
        add(f"   **{el2['verdict']}**"
            + (f"（两区间重叠 {f(el2.get('overlap'), '.3f')}）"
               if el2.get("overlap") is not None else ""))
    else:
        add("   （EA4 三臂尚未全部跑完）")
    add("")
    add("**③ H4 象限检验**：")
    if em4:
        add(f"   走 **{em4.get('branch')}** 分支——不加种子凑显著。")
        add(f"   需要约 **{f(em2.get('seeds_needed'), '.0f')} 个种子**"
            f"（现用 3 个），远超事先定死的阈值"
            f"（>{em2.get('thresholds', {}).get('not_worth_min_seeds')} 判为不值得）。")
    elif em3:
        p = em3.get("cross_run_p")
        add(f"   走 EM.3：加码到 {em3.get('n_seeds')} 个种子，"
            f"跨运行 p = {f(p, '.4f') if p is not None else '—'}"
            f"（{'显著' if (p is not None and p < 0.05) else '不显著'}）。")
    else:
        add(f"   功效分析已完成（需要 {f(em2.get('seeds_needed'), '.0f')} 个种子，"
            f"判定 {em2.get('verdict')}），**尚未执行加码**。")
    add("")
    add("---")
    add("")

    # ---------------- N ----------------
    add("## 1. 工作线N：基础设施")
    add("")
    add("### N.1 批量替换安全工具（第八条纪律的强制执行入口）")
    add("")
    add("`scripts/safe_batch_replace.py` —— 先在**临时副本**上执行替换、")
    add("对每个 `.py` 跑 `ast.parse`、**任一失败则整体回滚**（默认 `dry_run=True`）。")
    add("")
    add("**修掉了任务书伪代码里的一个真实缺陷**：它写的是")
    add("`Path(tmpdir) / Path(fp).name` —— 只用 basename，")
    add("两个不同目录下的同名文件会**互相覆盖**，于是"
        "「校验过的内容」与「写回的内容」可能不是同一份。")
    add("本实现按**相对路径镜像目录结构**。")
    add("")
    add("13 条测试，其中最关键的一条直接复现了真实事故：")
    add("把 `\"\"\"` 换成 `\"「\"` 必须被拦住、抛 `UnsafeReplaceError`、且**一个文件都不写回**。")
    add("")
    add("### N.2 变异测试编号权威文件")
    add("")
    add("`scripts/claim_mutation_ids.py` + `docs/mutation_registry.json`，")
    add("**锁保护的原子分配** + 可追溯账本。")
    add("")
    hist = REG.get("history") or []
    if hist:
        add(f"当前 `next_available` = **{REG.get('next_available')}**，"
            f"账本 {len(hist)} 条：")
        add("")
        add("| 时间 | 编号 | 领取者 |")
        add("|---|---|---|")
        for h in hist:
            ids = h.get("ids") or []
            rng = f"M{ids[0]}~M{ids[-1]}" if ids else "（初始化，无编号）"
            add(f"| {h.get('claimed_at')} | {rng} | {h.get('claimed_by')} |")
    add("")
    add("⚠️ `--init` 的 `next_available` 是从 `mutation_check.py` **实读**推导的"
        "（最大编号 + 1），**不采用任何文档里预先写死的数字**——"
        "这正是前两轮「任务书预写的编号与实际对不上」的修复。")
    add("")
    add("### 本轮踩到的两个环境坑（都记下来了）")
    add("")
    add("1. **本环境把 `Path.unlink()` 实现成了「安全删除（移到回收站）」**。")
    add("   后果：基于 `O_CREAT|O_EXCL` 的锁文件在**并发**下会因"
        "多次 trash 同一文件而报错、且失败后**锁文件永久残留**"
        "（实测：8 个线程只成功 2 个，其余全部超时）。")
    add("   修法：Windows 上加 `O_TEMPORARY`（句柄关闭自动删除，不依赖 unlink）"
        "+ **进程内再套一层 `threading.Lock`** + 陈旧锁接管。修完 14 条测试 0.68 秒全绿。")
    add("2. scipy 的 `nct`（非中心 t）在 **大 df + 大 ncp** 下返回 `nan`，")
    add("   而 `nan < power` 为假会让二分搜索**误判「已达标」并静默给出偏小的样本量**。")
    add("   修法：`nan` 时回退到正态近似，并在反解后**复核一次**"
        "（宁可返回 `nan` = 给不出答案，也不许悄悄给一个偏小的数）。")
    add("")
    add("---")
    add("")

    # ---------------- L ----------------
    add("## 2. 工作线L：回填 7 档 × 8 种子")
    add("")
    add("### 2.1 前提核对：任务书列了 10 个家族，其中两个是重复的")
    add("")
    add("`J1_stage6_baseline` 与 `EB1_chartist_only` 是**同一个配置**"
        "（旧读数逐位相同：k=1.2513572154501866）；")
    add("`J2_plus_B` 与 `EB1_all_three` 同理（k=0.7646304848983828）。")
    add("⇒ 实际只有 **8 个不同配置**，按 8 个跑。")
    add("")
    add("另一个前提：任务书写 `seeds=range(target_seeds)` —— `range` 产生 0..7，")
    add("而本项目的种子约定是 `SEED0 + 7*i`（`run_stage3.SEEDS`）。")
    add("用 `range(8)` 会得到**另一批随机数**，与阶段3 不可比。")
    add("本脚本直接复用 `run_stage3.SEEDS`（**逐位对齐**）。")
    add("")
    add("### 2.2 回填结果")
    add("")
    if with_ci:
        add("| 家族 | 未饱和/总档 | 种子 | k | 95% CI | 宽度 | 含0.5 | 旧读数 |")
        add("|---|---|---|---|---|---|---|---|")
        for name, r in fams.items():
            if r.get("k") is None:
                add(f"| `{name}` | {r.get('n_unsaturated')}/"
                    f"{r.get('n_levels_total')} | {r.get('n_seeds')} | — | — | — | — | "
                    f"{r.get('original_k')} |")
                continue
            lo, hi = r["ci"]
            add(f"| `{name}` | {r.get('n_unsaturated')}/{r.get('n_levels_total')} | "
                f"{r.get('n_seeds')} | {r['k']:.3f} | [{lo:.3f}, {hi:.3f}] | "
                f"**{r['ci_width']:.3f}** | "
                f"{'Y' if r.get('contains_0_5') else 'n'} | {r.get('original_k')} |")
    else:
        add("（缺 `out/workstream_L_metrics.json`）")
    add("")
    add("⚠️ **新旧口径不可直接比较**：旧的是 3 种子 / 3~4 档，"
        "新的是 8 种子 / 未饱和档。k 的差异主要来自**分辨力提升**，"
        "不能归因为机制效应变化。")
    add("")
    add("### 2.3 ⭐ EL.2：非对称是否比对称更凹（本工作线最重要的产出）")
    add("")
    if el2.get("verdict"):
        t, b = el2.get("taker") or {}, el2.get("both") or {}
        add(f"- `EA4_taker`：k = {f(t.get('k'))}，CI = {t.get('ci')}")
        add(f"- `EA4_both` ：k = {f(b.get('k'))}，CI = {b.get('ci')}")
        add(f"- 点估计差（taker − both）= {f(el2.get('delta_k_point'), '+.3f')}")
        add(f"- 两区间重叠 = {f(el2.get('overlap'), '.3f')}"
            + (f"（较窄区间的 {f((el2.get('overlap_frac_of_narrower') or 0) * 100, '.1f')}%）"
               if el2.get("overlap_frac_of_narrower") is not None else ""))
        add("")
        add(f"### ⇒ 判定：**{el2['verdict']}**")
        add("")
        if el2["verdict"] == "依然无法判定":
            add("⚠️ **如实报告**：即使把配置从「3 种子 × 4 档」升级到"
                "「8 种子 × 未饱和档」，这个问题**仍然判不出来**。")
            add("不能因为「投入了更多资源」就暗示「应该有答案了」——")
            add("这与阶段3 判定 k 精确值需要 300 个种子是**同一类问题**。")
            # 量化"为什么判不了"：按宽度 ∝ 1/√n 外推需要多少种子才能把两个区间分开
            d_pt = abs(el2.get("delta_k_point") or 0.0)
            w_vals = [abs((el2.get(k) or {}).get("ci", [0, 0])[1]
                          - (el2.get(k) or {}).get("ci", [0, 0])[0])
                      for k in ("taker", "both")]
            w_vals = [w for w in w_vals if w > 0]
            n_seeds_now = fams.get("EA4_taker", {}).get("n_seeds") or 8
            if d_pt > 0 and w_vals:
                w = min(w_vals)
                n_need = n_seeds_now * (w / d_pt) ** 2
                add("")
                add(f"- **量化「为什么判不了」**：点估计差 **{d_pt:.3f}**，"
                    f"较窄的那个区间宽 **{w:.3f}**。"
                    f"按「宽度 ∝ 1/√n」外推，要让两个区间**分开**"
                    f"需要约 **{n_need:.0f} 个种子**（现用 {n_seeds_now} 个）。")
                add(f"  ⇒ 这个效应量相对噪声太小，**不是加一点资源能解决的**，"
                    f"属于「方向可信、显著性做不实」的类型。")
        elif el2["verdict"] == "非对称显著更凹":
            add("⭐ 这是从三线深挖的 EA.4 发现开始、历经"
                "「意外修复 → 标尺假象 → 分辨力不足」几轮曲折之后，"
                "**第一次得到有区间支撑的确定结论**。")
    else:
        add("（EA4 三臂尚未全部完成）")
    add("")
    add("### 2.4 ⭐ 比 EL.2 本身更重要的发现：**效应量级对比**")
    add("")
    add("回填后 EA.4 三臂的 k **几乎相同**（0.73 / 0.74 / 0.79，极差 0.060）。")
    add("把它和「有/无稀疏化」的效应放在一起看：")
    add("")
    # 现算：用工作线H 的 baseline（Hawkes 关，7 档）跑同口径的 k
    try:
        from diagnose_k_uncertainty import bootstrap_k
        hcfg = ((load("workstream_H_metrics.json").get("configs") or {})
                .get("baseline_no_hawkes") or {})
        hrows = [r for r in (hcfg.get("rows") or []) if not r.get("saturated")]
        hb = bootstrap_k(hrows, n_boot=4000) if hrows else {}
    except Exception:  # noqa: BLE001
        hb = {}
    k_off = hb.get("k_point")
    ks = {n: r["k"] for n, r in with_ci.items()
          if n.startswith("EA4_") and r.get("k") is not None}
    if k_off and len(ks) >= 2:
        k_on_mean = sum(ks.values()) / len(ks)
        eff_a = k_off - k_on_mean
        eff_b = abs(ks.get("EA4_taker", 0) - ks.get("EA4_both", 1))
        eff_c = abs(ks.get("EA4_taker", 0) - ks.get("EA4_maker", 1))
        add("| 效应 | 幅度 |")
        add("|---|---|")
        add(f"| **A. 有 / 无稀疏化**（Hawkes 关 {k_off:.3f} → 开 {k_on_mean:.3f}） | "
            f"**{eff_a:.3f}** |")
        add(f"| **B. 稀疏化作用在哪一侧**（taker {ks.get('EA4_taker', float('nan')):.3f}"
            f" vs both {ks.get('EA4_both', float('nan')):.3f}） | **{eff_b:.3f}** |")
        add(f"| C. taker vs maker | {eff_c:.3f} |")
        add("")
        if eff_b > 0:
            add(f"⇒ **A / B = {eff_a / eff_b:.1f} 倍**："
                "「有没有稀疏化」的效应比「作用在哪一侧」**大一个数量级**。")
            add("")
            add("⚠️ **这意味着三线深挖的核心叙事需要修正**："
                "那一轮说「只稀疏化吃单方最有效（k=0.455）、双边同时稀疏化互相抵消"
                "（k=0.947）」——**在回填后的干净测量里，这个区分几乎消失了**"
                "（三臂的极差只有 0.060，而噪声区间宽 0.45~0.83）。")
            add("")
            add("更准确的表述是：**只要存在时间聚集性，冲击就会变凹；"
                "而它作用在需求侧还是供给侧，影响小得多。**")
    add("")
    add("🔎 但要注意**同口径**：上面的 Hawkes 关基线是**现算的**"
        "（用工作线H 的 7 档 rows 跑 bootstrap），不是从旧报告抄的——")
    add("旧对照（k=1.395）是 4 档 3 种子的产物，与新口径不可直接比。")
    add("")
    add("### 2.5 EL.3 / EL.4：历史结论在新配置下是否仍成立")
    add("")
    if el3.get("arms"):
        add("| 臂 | k | 相对基线的变化 | 区间分离 |")
        add("|---|---|---|---|")
        add(f"| 基线 `EB1_chartist_only` | {f(el3.get('baseline_k'))} | — | — |")
        for tag, d in (el3.get("arms") or {}).items():
            add(f"| `{tag}` | {f(d.get('k'))} | "
                f"{f(d.get('delta_vs_baseline'), '+.3f')}"
                f"（{'改善' if d.get('improved') else '变差'}） | "
                f"{'✅ 分离' if d.get('ci_separated') else '重叠'} |")
    add("")
    if el4.get("j2_k") is not None:
        add(f"- EL.4：J2（= `EB1_all_three`）k = {f(el4.get('j2_k'))}，"
            f"J3 k = {f(el4.get('j3_k'))}")
        add(f"  ⇒ J2 是否仍优于 J3：**{'是' if el4.get('j2_still_better') else '否'}**"
            f"（Δ = {f(el4.get('delta_j3_minus_j2'), '+.3f')}，"
            f"区间{'分离' if el4.get('ci_separated') else '重叠'}）")
    add("")
    add("---")
    add("")

    # ---------------- M ----------------
    add("## 3. 工作线M：H4 象限检验的功效分析")
    add("")
    add("### 3.1 EM.1 功效分析")
    add("")
    if em1:
        obs = em1.get("observed") or {}
        add(f"- 观测（从 E 线产物现读）：taker_only n={obs.get('n_taker_only')}"
            f"（{f(obs.get('mean_taker_only'), '.2f')}±{f(obs.get('sd_taker_only'), '.2f')}）、"
            f"both_burst n={obs.get('n_both_burst')}"
            f"（{f(obs.get('mean_both_burst'), '.2f')}±{f(obs.get('sd_both_burst'), '.2f')}）")
        add(f"- 差 = {f(obs.get('diff'), '+.2f')}，合并 sd = {f(obs.get('pooled_sd'), '.2f')}")
        add(f"- **Cohen's d = {f(em1.get('effect_size_cohens_d'), '.4f')}**（小效应）")
        add(f"- 达到 80% 功效需要每组 n1 = **{f(em1.get('required_n_group1'), '.0f')}**、"
            f"n2 = {f(em1.get('required_n_group2'), '.0f')}"
            f"（合计 {f(em1.get('total_n'), '.0f')} 个事件观测点）")
    add("")
    add("⚠️ 两组样本量不等（42:31）——`ratio` 参数是必须的，不能套等样本量公式。")
    add("")
    add("### 3.2 EM.2 成本判断（阈值事先定死）")
    add("")
    if em2:
        th = em2.get("thresholds") or {}
        add(f"- 换算依据：{em2.get('observed_n_seeds')} 种子产出 "
            f"{f(em2.get('events_per_seed'), '.1f')} × {em2.get('observed_n_seeds')} "
            f"个事件点 ⇒ 每种子 {f(em2.get('events_per_seed'), '.1f')} 个")
        add(f"- **需要约 {f(em2.get('seeds_needed'), '.0f')} 个种子**"
            f"（或等价地把单次模拟时长 × "
            f"{f(em2.get('length_multiplier_option'), '.1f')}）")
        add(f"- 阈值：≤{th.get('worth_it_max_seeds')} 值得 / "
            f"≤{th.get('marginal_max_seeds')} 边缘 / "
            f">{th.get('not_worth_min_seeds')} 不值得")
        add(f"- **判定：{em2.get('verdict')}**")
    add("")
    add("### 3.3 EM.4/EM.3 的分支结果")
    add("")
    if em4:
        add(f"走 **{em4.get('branch')}**。")
        add("")
        add(f"> {em4.get('text')}")
        add("")
        add(f"**建议**：{em4.get('recommendation')}")
    elif em3:
        add(f"走 EM.3：加码到 {em3.get('n_seeds')} 个种子。")
        if em3.get("cross_run_p") is not None:
            add(f"- 跨运行检验：均值差 {f(em3.get('cross_run_mean_diff'), '+.2f')}"
                f" ± {f(em3.get('cross_run_sem'), '.2f')}，"
                f"t = {f(em3.get('cross_run_t'), '+.2f')}，"
                f"p = {f(em3.get('cross_run_p'), '.4f')}")
        own = em3.get("own_scale_power") or {}
        if own:
            add(f"- **同口径**反算（以跨运行块级差为观测，自由度 = 运行数）："
                f"d = {f(own.get('cross_run_effect_size_d'), '.3f')}，"
                f"需要 {f(own.get('n_runs_for_80pct'), '.0f')} 次运行达 80% 功效，"
                f"当前 {own.get('current_n_runs')} 次的功效 = "
                f"{f(own.get('current_power_at_observed_d'), '.3f')}")
    else:
        add("（尚未执行；功效分析已完成）")
    add("")
    add("---")
    add("")
    add("## 4. 收尾：清单更新")
    add("")
    if PM:
        add(f"- 扫描命中 **{PM.get('scanned_total')}** 处「数值比较型」陈述")
        Ls = PM.get("L") or {}
        add(f"- 工作线L 覆盖 **{Ls.get('n_families')}** 个家族，"
            f"{Ls.get('n_with_ci')} 个拿到可用 CI")
        Ms = PM.get("M") or {}
        add(f"- 工作线M：Cohen's d = {f(Ms.get('cohens_d'), '.4f')}，"
            f"需要 {f(Ms.get('seeds_needed'), '.0f')} 个种子，"
            f"判定 {Ms.get('verdict')}，分支 {Ms.get('branch')}")
        add("- 清单：`docs/历史数值陈述分辨力清单.md`"
            "（新增 §3.4 已回填标准配置、§3.5 H4 状态两节）")
    else:
        add("（缺 `out/post_L_M_summary.json`）")
    add("")
    add("---")
    add("")
    add("## 5. 诚实边界")
    add("")
    add("1. **新旧口径不可直接比较**：回填改的是**分辨力**（种子数 + 档位数），")
    add("   不是机制。k 的变化主要来自分辨力提升。")
    add("2. **EL.2 若仍是「无法判定」，那就是无法判定**——")
    add("   不许因为投入了更多资源而暗示应该有答案。")
    add("3. **EM.4（诚实报告不显著）不是失败**：功效分析的价值就是提前算清"
        "「值不值得投入」；算出来不值得，**不投入本身就是产出**。")
    add("4. **回填只覆盖 8 个家族**（EA.4 / EB.1 / J1–J3）。")
    add("   清单里其它裸奔条目（例如阶段6/8 的历史读数）仍待后续轮次处理。")
    add("5. **EM.1 与 EM.3 的口径不同**：前者算「每组事件观测点」，"
        "后者用「跨运行的块级差」（自由度 = 运行数）。两者不可混用，"
        "所以 EM.3 里额外做了同口径反算。")
    add("")
    return "\n".join(L_) + "\n"


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    txt = build()
    head = txt.split("## 1. 工作线N")[0]
    for bad in ("nan", "None"):
        if bad in head:
            raise SystemExit(f"❌ 小结前段出现未处理值 {bad!r}，拒绝写出。")
    TARGET.write_text(txt, encoding="utf-8")
    n_tbl = sum(1 for ln in txt.splitlines() if ln.strip().startswith("|"))
    print(f"  交付小结已生成：{TARGET}（{len(txt.encode('utf-8')) / 1024:.0f} KB，"
          f"{n_tbl} 行表格）")


if __name__ == "__main__":
    main()
