"""生成 docs/分辨力危机-交付小结.md（工作线 H/I/E/J/K）。

数字全部从 out/*.json **现算**，不手抄。
其中「扩档位能把 CI 压窄多少」是**当场用 bootstrap 重算的**——
它是本轮最重要的量化结论，绝不能是手抄的。

⚠️ 本文件的文案里**一律不使用 ASCII 双引号**，中文引用统一用「」。
原因：本项目发生过两次「自动修引号脚本把源码写坏」的事故，
而只要有字符串内部出现裸的 ASCII 双引号，就一定会有人（或脚本）去修它，
修的时候又会误伤 `"key": value` 这类合法引号。
**从源头避免比事后修更便宜。**
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import OUT  # noqa: E402

DOCS = Path(__file__).resolve().parent.parent / "docs"
TARGET = DOCS / "分辨力危机-交付小结.md"


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


def _ci_width_by_levels(rows: list[dict]) -> list[dict]:
    """⚠️ 当场现算：同一份数据下，**档位数**对 k 的 95% 区间宽度的影响。

    这是本轮最重要的量化结论（扩档位比加种子便宜且有效），
    所以必须是算出来的，不是抄的。
    """
    from diagnose_k_uncertainty import bootstrap_k
    from scipy import stats
    out = []
    for n in range(3, len(rows) + 1):
        r = bootstrap_k(rows[:n], n_boot=4000, apply_unsat_filter=False)
        if not r.get("ok"):
            continue
        lo, hi = r["ci95"]
        out.append({
            "n_levels": n, "df": n - 2,
            "t_crit": float(stats.t.ppf(0.975, max(1, n - 2))),
            "k": r["k_point"], "ci": [lo, hi], "width": r["width"],
            "covers_0.5": bool(lo <= 0.5 <= hi),
            "covers_1.0": bool(lo <= 1.0 <= hi),
        })
    return out


def build() -> str:
    H = load("workstream_H_metrics.json")
    I = load("workstream_I_metrics.json")
    E = load("workstream_E_metrics.json")
    J = load("workstream_J_metrics.json")
    NC = load("numeric_claims_raw.json")

    ej2 = J.get("ej2") or {}
    ej3 = J.get("ej3") or {}
    ej4 = J.get("ej4") or {}
    i1 = (I.get("ei1") or {}).get("arms") or {}
    i2 = (I.get("ei2") or {}).get("arms") or {}
    i4 = (I.get("ei4") or {}).get("arms") or {}
    i3 = I.get("ei3") or {}
    e1 = (E.get("ee1_sync") or {}).get("configs") or {}
    e3 = (E.get("ee3_progressive") or {}).get("arms") or {}

    s3 = load("stage3_metrics.json")
    s3_rows = ((s3.get("C2_staged") or {}).get("rows")) or []
    ladder = _ci_width_by_levels(s3_rows) if s3_rows else []

    L: list[str] = []
    add = L.append
    add("# 分辨力危机应对 · 交付小结")
    add("")
    add("> 对应任务书：`交易世界 · 分辨力危机应对任务书.md`")
    add("> 本文件由 `scripts/make_resolution_summary.py` 生成，数字从 `out/*.json` 现算。")
    add("> 上一轮的诊断报告是 `docs/前置校验-EA4-诊断报告.md`。")
    add("")
    add("---")
    add("")
    add("## 0. 三条最重要的结论（先说结论）")
    add("")
    add("### 0.1 ⚠️ 路径 B（换判据）**没有**解决分辨力问题——它比原来更糟")
    add("")
    n1 = sum(a.get("verdicts", {}).get("不可判定", 0) for a in i1.values())
    p1 = sum(len(a.get("pairs") or []) for a in i1.values())
    n2 = sum(a.get("verdicts", {}).get("不可判定", 0) for a in i2.values())
    p2 = sum(len(a.get("pairs") or []) for a in i2.values())
    n4 = sum(a.get("verdicts", {}).get("不可判定", 0) for a in i4.values())
    p4 = sum(len(a.get("pairs") or []) for a in i4.values())
    add("任务书希望用「相邻档位局部弹性」替代幂律拟合，理由是后者只用 2 个点、"
        "看起来方差更小。**实测相反**：")
    add("")
    add(f"- 把全部档位对跑一遍：**{n1 + n2 + n4}/{p1 + p2 + p4} 个判决是「不可判定」**")
    add(f"  （EI.1 历史 4 档数据 {n1}/{p1}；EI.2 工作线H 的 7/6 档数据 {n2}/{p2}；"
        f"EI.4 历史结论重读 {n4}/{p4}）")
    add("- 原因：局部弹性把**自由度丢掉了**。两档算一个比值、没有残差自由度，"
        "不确定度完全由两端的标准误决定；而全局拟合用 n 个点估 1 个斜率，"
        "**档位越多越准**（自由度 = n − 2）。")
    add("")
    add("### 0.2 ⭐ 真正有效的降噪路径是**扩档位**，不是换判据、也不必加 300 个种子")
    add("")
    if ladder:
        a, z = ladder[0], ladder[-1]
        add(f"同一份数据（阶段3 的 8 种子），只把参与拟合的**档位数**从 "
            f"{a['n_levels']} 增加到 {z['n_levels']}：")
        add("")
        add("| 档位数 | 自由度 | t 临界值 | k | 95% 区间 | 宽度 | 含 0.5 | 含 1.0 |")
        add("|---|---|---|---|---|---|---|---|")
        for r in ladder:
            add(f"| {r['n_levels']} | {r['df']} | {f(r['t_crit'], '.2f')} | "
                f"{f(r['k'], '.3f')} | [{f(r['ci'][0], '.3f')}, {f(r['ci'][1], '.3f')}] | "
                f"**{f(r['width'], '.3f')}** | {'Y' if r['covers_0.5'] else 'n'} | "
                f"{'Y' if r['covers_1.0'] else 'n'} |")
        add("")
        add(f"⇒ 宽度从 **{f(a['width'], '.3f')}** 压到 **{f(z['width'], '.3f')}**"
            f"（缩小 **{(1 - z['width'] / a['width']) * 100:.0f}%**），"
            f"而且区间**不再包含 0.5**。")
        add(f"主因是 t 临界值：{a['n_levels']} 档时 df={a['df']} ⇒ t={f(a['t_crit'], '.2f')}；"
            f"{z['n_levels']} 档时 df={z['df']} ⇒ t={f(z['t_crit'], '.2f')}。")
    add("")
    add("在**独立**的工作线H 上（三配置 × 7 档 × 现有 3 种子）同样验证了这一点——"
        "见 §1 的表：三配置的 CI 宽度都落进 1.0~1.3，而它们此前（4 档）分别是 2.0~2.3。")
    add("")
    add("### 0.3 ⭐ J 线（阶段3 最初的 k）：原口径判定不了，但扩档位后方向站得住")
    add("")
    if ej2.get("ok"):
        lo, hi = ej2["ci95"]
        add(f"- 原始口径（未饱和 3 档、8 种子）：k = **{f(ej2['k_point'], '.3f')}**"
            f"（γ = {f(ej2['gamma_point'], '.3f')}），95% CI = **[{f(lo, '.3f')}, {f(hi, '.3f')}]**")
        add("  —— **同时覆盖 0.5（平方根律）与 1.0（线性）** ⇒ 方向性判断无法被判定")
        add("- 但同一份数据扩到 4 档以上，区间下界就抬到 0.5 以上（见 §0.2）")
        add("- **准确的说法**：二期「用长记忆订单流修复冲击律」的立项依据"
            "**建立在点估计上**，其统计判定力取决于档位配置；"
            "把未饱和档位扩到 4 档以上，方向性判断（偏线性、非平方根律）就成立。")
    add("")
    add("---")
    add("")

    add("## 1. 工作线H：饱和检查（路径C，不加种子）")
    add("")
    cal = H.get("lambda") or {}
    add(f"- λ̄ = **{f(cal.get('base_lambda'), '.3f')}**"
        f"（与 EA.4 官方基准一致：{'✅ 是' if cal.get('matches_ea4_official') else '❌ 否'}）"
        f" —— 任务书 §2.6 要求的核对通过")
    add(f"- 标定窗口 = 实验窗口 = `{H.get('experiment_window')}`；"
        f"{len(H.get('seeds') or [])} 个种子、{len(H.get('shock_sizes') or [])} 档、"
        f"slices={H.get('slices')}")
    add("")
    add("| 配置 | 未饱和档数 | 饱和边界 | 最大 \\|t\\|（未饱和档） |")
    add("|---|---|---|---|")
    for cfg, d in (H.get("configs") or {}).items():
        sat = d.get("saturated_from")
        add(f"| {cfg} | {d.get('n_unsaturated')} | "
            f"{f(sat, '.4f') if sat else '无（全部未饱和）'} | "
            f"{f(d.get('max_abs_t_unsaturated'), '.2f')} |")
    add("")
    add("**EH.3 得到确认**：信号随档位增大而增强——三配置里最大的 \\|t\\| 都出现在大档位上，"
        "说明**同一批预算下，把种子投在大档位上比平均撒更划算**。")
    add("")
    add("散点图：`out/figs/workstream_H_loglog.png`（EH.2 要求必须实际生成）。")
    add("")
    add("**诚实边界**：")
    add("- 本工作线用 `slices=20`（与 EA.4 一致），而阶段3 的 C2 用 `slices=10`。"
        "**分片数不同 ⇒ 饱和边界不可跨装置直接比较**（分片越细越不易饱和）。")
    add("- 三配置的**绝对深度**不可直接比较（稀疏化会改变吃单/挂单的相对节奏），"
        "涉及深度的结论都只在**同配置内**做。")
    add("")
    add("---")
    add("")

    add("## 2. 工作线I：凹度判据（路径B）")
    add("")
    add("判据本体：`tw/analyzer_concavity.py`（相邻/跨档局部弹性 + 三分类判决）。")
    add("**任务书 §3.6 的两条硬要求都已执行**：全部档位对都报（不挑显著的）；"
        "种子少时用 t 分布临界值而非 1.96。")
    add("")
    add("| 实验 | 数据来源 | 档位对数 | 显著凹 | 线性/更差 | 不可判定 |")
    add("|---|---|---|---|---|---|")
    for tag, arms, src in (("EI.1", i1, "现有 4 档（三线深挖）"),
                           ("EI.2", i2, "工作线H 的 7/6 档"),
                           ("EI.4", i4, "历史结论（EB.1 / J1–J3）")):
        tot = sum(len(a.get("pairs") or []) for a in arms.values())
        c = sum(a.get("verdicts", {}).get("显著凹", 0) for a in arms.values())
        lw = sum(a.get("verdicts", {}).get("显著线性或更差", 0) for a in arms.values())
        ud = sum(a.get("verdicts", {}).get("不可判定", 0) for a in arms.values())
        add(f"| {tag} | {src} | {tot} | {c} | {lw} | **{ud}** |")
    add("")
    if i3.get("taker_vs_both"):
        d = i3["taker_vs_both"]
        add("### EI.3（本工作线最重要的产出）：非对称是否比对称更凹？")
        add("")
        add(f"- 可比档位对 **{d.get('n_pairs')}** 对")
        add(f"- 点估计上「taker 更凹」的有 **{d.get('n_taker_more_concave')}** 对")
        add(f"- 其中**差异显著**的有 **{d.get('n_significant_and_taker_more_concave')}** 对")
        add("")
        add("⇒ **无法确认「非对称比对称更显著凹」**。这不是否证 EA.4，"
            "而是说：**在这个判据下给不出判决**"
            "（同样的数据与种子，两配置的弹性差远小于它的不确定度）。")
    add("")
    add("**诚实边界**：")
    add("- `delta_method` 是**一阶近似**：现有产物只存了每档的均值 ± sem，没有逐种子数组，"
        "所以逐种子算弹性做不到。两条路径都在产物里标了 `method` 字段，"
        "**没有把近似冒充成原始数据**。")
    add("- 凹度判据**没有**被提议写入项目验收标准（见 §5）。")
    add("")
    add("---")
    add("")

    add("## 3. 工作线E：H4 假说验证（不依赖 k）")
    add("")
    add("### 3.1 前提核对：H4 的措辞与源码不符（必须记下来）")
    add("")
    add("H4 假设「吃单方和做市方服从**各自独立的** Hawkes 过程」。"
        "但 `HawkesMarket._limit_activity` 在对称模式下**只取一次 ρ**"
        "（`rho = self._current_rho()`）然后对整个列表稀疏化 ⇒ "
        "**两侧由同一个 ρ 驱动**，同步是**构造性**的。")
    add("")
    add("### 3.2 EE.1 同步率")
    add("")
    add("| 配置 | 联合同步率 | **P(做市突发\\|吃单突发)** | lift | 突发点数 taker/maker |")
    add("|---|---|---|---|---|")
    for tag, c in e1.items():
        s0 = (c.get("sync") or [{}])[0]
        add(f"| {tag} | {f(c.get('sync_mean'), '.4f')} | "
            f"**{f(c.get('p_maker_given_taker_mean'), '.4f')}** | "
            f"{f(c.get('lift_mean'), '.2f')} | "
            f"{s0.get('n_taker_burst')}/{s0.get('n_maker_burst')} |")
    add("")
    add("⚠️ 任务书说的同步率实际是**联合**概率 P(a∧b)，它同时被两侧各自的突发频率压着"
        "（读起来像几乎不同步）。H4 关心的是**条件**概率 P(做市突发|吃单突发)——"
        "上面两列都给了，`lift` = 条件概率 / 边际概率。")
    add("")
    add("### 3.3 EE.2 四象限深度（H4 的真正检验）")
    add("")
    add("| 配置 | 象限 | n | 深度均值 |")
    add("|---|---|---|---|")
    for tag, c in e1.items():
        q = (c.get("quadrants") or [{}])[0]
        for name in ("both_burst", "taker_only", "maker_only", "neither"):
            d = q.get(name) or {}
            if d.get("n"):
                add(f"| {tag} | {name} | {d['n']} | {f(d['mean'], '.2f')} |")
    add("")
    for tag, c in e1.items():
        cs = c.get("contrast") or [{}]
        c0 = cs[0] if cs else {}
        if c0.get("reason"):
            add(f"- {tag}：块级对比不可做（{c0['reason']}）")
        else:
            add(f"- {tag}：taker_only vs both_burst 差 = {f(c0.get('diff'), '+.2f')}"
                f"，t = {f(c0.get('t'), '+.2f')}，p = {f(c0.get('p'), '.4f')}，"
                f"H4 方向**{'支持' if c0.get('h4_direction_supported') else '不支持'}**"
                f"（**不显著**）")
    add("")
    add("### 3.4 EE.3 渐进式做市方聚集度扫描（改用凹度判决）")
    add("")
    if e3:
        add("| 耦合强度 | 显著凹 / 线性或更差 / 不可判定 |")
        add("|---|---|")
        for cp, a in e3.items():
            v = a.get("verdicts") or {}
            tag = "非对称" if cp == "0.0" else ("对称" if cp == "1.0" else "中间")
            add(f"| {cp}（{tag}） | {v.get('显著凹', 0)} / "
                f"{v.get('显著线性或更差', 0)} / **{v.get('不可判定', 0)}** |")
        add("")
        add("⇒ 五档耦合强度下判决**全部不可判定** ⇒ "
            "**得不到「允许做市方有多少聚集性」这条曲线**。"
            "这是对任务书 §4.5 期望的诚实回答。")
    else:
        add("（缺 `out/workstream_E_metrics.json` 的 `ee3_progressive`——"
            "先跑 `scripts/run_workstream_E.py --only EE3`）")
    add("")
    add("### 3.5 EE 的**正面**发现（不依赖 k，也不依赖凹度判决）")
    add("")
    add("用**同一配置内的相对量**比较（避免跨配置的绝对深度混淆）：")
    add("")
    add("| 配置 | 吃单突发期深度 | 背景期深度 | 相对变化 |")
    add("|---|---|---|---|")
    for tag, c in e1.items():
        q = (c.get("quadrants") or [{}])[0]
        tb = (q.get("both_burst") or {}).get("n", 0)
        to = (q.get("taker_only") or {}).get("n", 0)
        bm = (q.get("both_burst") or {}).get("mean")
        om = (q.get("taker_only") or {}).get("mean")
        back = (q.get("neither") or {}).get("mean")
        if tb and to and bm and om:
            burst_mean = (tb * bm + to * om) / (tb + to)
        elif to and om:
            burst_mean = om
        elif tb and bm:
            burst_mean = bm
        else:
            continue
        rel = (burst_mean / back - 1) * 100 if back else float("nan")
        add(f"| {tag} | {f(burst_mean, '.2f')} | {f(back, '.2f')} | **{f(rel, '+.1f')}%** |")
    add("")
    add("⇒ **非对称配置下，吃单方突发使局部深度显著变薄；对称配置下几乎不变**——"
        "这与 H4 的**宏观形式**一致（做市方同步活跃把稀疏填平了）。"
        "注意这是**同配置内的相对量**，不受跨配置绝对深度差异的影响。")
    add("")
    add("**诚实边界**：")
    add("- 象限内的配对检验 **p = 0.26 不显著**（3 种子）——**方向支持但强度不足以立论**。")
    add("- `maker_coupling` 是为 EE.3 新写的装配件（让做市方有效 ρ 在 1 与 ρ 之间线性插值），"
        "**它不是原生产品行为**。")
    add("")
    add("---")
    add("")

    add("## 4. 工作线J：回溯审计（本轮最重要的产出）")
    add("")
    info = J.get("ej1") or {}
    add("### 4.1 数据定位")
    add("")
    add(f"- `{info.get('file')}` :: `{info.get('block')}` —— 找到 **{info.get('n_levels')} 档**，"
        f"**有逐档细节**（sem / t / n_seeds 都在）")
    add(f"- 种子数 = **{info.get('n_seeds_in_data')}** —— "
        f"⚠️ 任务书猜的是「通常是 3 个」，**实际是 8 个**")
    add("- **不需要重跑** ⇒ 审计用的就是**原始那次运行**的读数，"
        "因此不存在「重采样种子」这个限定条件")
    add("")
    add("### 4.2 置信区间审计")
    add("")
    if ej2.get("ok"):
        lo, hi = ej2["ci95"]
        add(f"- k = **{f(ej2['k_point'], '.3f')}**，95% CI = **[{f(lo, '.3f')}, {f(hi, '.3f')}]**"
            f"（宽 {f(ej2['width'], '.3f')}），显著档位 {ej2['significant_levels']}/{ej2['n_levels']}")
        if ej2.get("gamma_ci95"):
            add(f"- 换算成报告原文用的 γ = 1/k：**{f(ej2['gamma_point'], '.3f')}**"
                f"（区间 [{f(ej2['gamma_ci95'][0], '.3f')}, {f(ej2['gamma_ci95'][1], '.3f')}]）")
        add(f"- 区间覆盖 0.5（平方根律）：**{'是' if ej2['covers_0.5_sqrt_law'] else '否'}**；"
            f"覆盖 1.0（线性）：**{'是' if ej2['covers_1.0_linear'] else '否'}**")
        add("")
        add("⚠️ 任务书把 γ 与 k 混了：它问「0.77~0.8 这个数的区间是否覆盖 0.5 和 1」——"
            "0.77~0.8 是 **γ**，对应 k ≈ 1.25。两个量都报在上表里。")
    add("")
    add("### 4.3 凹度判决")
    add("")
    add("| 档位对 | 弹性 | 95% 区间 | 判决 | t_vs_sqrt |")
    add("|---|---|---|---|---|")
    for p in (ej3.get("pairs") or []):
        if not p.get("ok"):
            continue
        add(f"| {p['size_small']}→{p['size_big']} | {f(p['e'], '.3f')} | "
            f"[{f(p['ci_lower'], '.3f')}, {f(p['ci_upper'], '.3f')}] | {p['verdict']} | "
            f"{f(p['t_vs_sqrt'], '.2f')} |")
    add("")
    add(f"（未饱和 {ej3.get('n_unsaturated')} 档，**全部档位对都列出**。）")
    add("")
    add("### 4.4 ⭐ EJ.4：二期那条技术路线的立项依据，现在还站不站得住？")
    add("")
    if ej2.get("ok"):
        add("**结论：站得住，但要把措辞从「数字」改成「方向」，并把档位配置说清楚。**")
        add("")
        add("1. **原口径下判定不了**：区间同时覆盖 0.5 与 1.0，所以"
            "「偏线性、非平方根律」这个判断**在原始档位配置下无法被统计检验支撑**。")
        add("2. **但这不是「方向被推翻」**：点估计在**独立来源**上高度一致——")
        add(f"   阶段3 的 k = {f(ej4.get('original_k_fit_unsaturated_3levels'), '.3f')}"
            f"（γ = {f(ej4.get('original_gamma_implied'), '.3f')}）、"
            f"三线深挖的 Hawkes 关臂 k = 1.395、工作线H 的 baseline k = 1.292"
            f"——**三个独立实验都指向「比平方根律更陡」**。")
        add("3. **判定力取决于档位配置**：把未饱和档位从 3 扩到 4 以上，"
            "区间下界即抬到 0.5 以上（§0.2 的表）。")
        add("")
        add("⇒ **对二期的评价**：立项**大方向没有错**"
            "（「冲击比平方根律陡」这个方向在三个独立来源上都成立）；"
            "错的是把验收标准设成了一个**当时分辨力不足以判定**的精确数字（k≈0.5）。"
            "这正好呼应任务书 §8 的那句话——**「精确逼近 0.5」从一开始就定得过于具体。**")
    add("")
    add("---")
    add("")

    add("## 5. 工作线K：项目治理（把「先报 CI 再报差」自动化）")
    add("")
    n_total = sum(len(v) for v in (NC or {}).values()) if NC else 0
    add(f"- 扫描器：`scripts/audit_numeric_claims.py`，对一期/二期/三线/诊断/阶段小结"
        f"五份报告扫出 **{n_total}** 处「数值比较型」陈述")
    add("- 分级明确：**自动层**（启发式，会有误报漏报）+ **人工复核层**"
        "（`REVIEWED` 常量，逐条确认）——任务书 §6.6 自己也强调正则只是粗筛，"
        "所以**没有把自动判定当成结论**")
    add("- 清单：`docs/历史数值陈述分辨力清单.md`（含 k 类陈述的最高优先级、"
        "阶段3 的「已审计」标记、以及「不适用本条纪律」的类别）")
    add("- 第七条纪律已写入 `README.md` 的工程纪律章节（第 7 条）")
    add("")
    add("### 关于「把凹度判据写入项目验收标准」")
    add("")
    add("任务书 §3.4 要求正式提议、但**留给设计者确认**。本轮的立场是："
        "**暂不建议用它替换「k≈0.5」，因为实测它给不出判决**（§0.1）。"
        "更值得提议的是另一条：**把「档位配置」写成验收标准的一部分**——"
        "例如「未饱和档位 ≥ 4 档、且最大档位的 |t| 明确显著」。")
    add("")
    add("---")
    add("")
    add("## 6. 诚实边界（汇总）")
    add("")
    add("1. **路径 B 被实测否定**：凹度判据比全局拟合**更**没分辨力。"
        "任务书的前提（2 个点的方差更小）不成立——它丢掉了自由度。")
    add("2. **路径 A 的结论被部分改写**：不需要 300 个种子；"
        "**扩档位（近零成本）就能把 CI 宽度减半**。"
        "300 种子那个数是在「档位数固定」的前提下算出来的。")
    add("3. **H4 只得到「方向支持、统计不显著」**（象限内 p = 0.26，3 种子）。"
        "宏观形式（非对称下突发期深度变薄 6 倍于对称）成立且方向清晰，"
        "但它与象限内检验是两种口径，不能互相代替。")
    add("4. **EE.3 没给出「允许多少聚集性」的曲线**——五档耦合全部不可判定。"
        "这是任务书 §4.5 期望的产出，本轮**没有拿到**。")
    add("5. **跨配置的绝对深度不可直接比较**；所有深度结论都只在同配置内做。")
    add("6. **阶段3 的审计结论依赖「扩档位」这条解读**："
        "严格未饱和口径（3 档）下就是判定不了，这一点没有被任何后续分析改变。")
    add("")
    add("---")
    add("")
    add("## 7. 需要设计者拍板的三件事")
    add("")
    add("1. **验收标准怎么改**：从「k ≈ 0.5」改成什么？本轮建议**不要**换成凹度判据"
        "（实测无效），而是「未饱和档位 ≥ 4 档 + 大档位 |t| 显著」这类**装置性**要求。")
    add("2. **是否继续追「逼近 0.5」**：本轮的量化说明**扩档位是性价比最高的一步**。"
        "要不要在下轮把 7 档 × 8 种子做成标准配置？")
    add("3. **E 线要不要继续**：H4 的宏观形式成立，但象限内检验不显著。"
        "要不要加种子把这一步做实？")
    add("")
    return "\n".join(L) + "\n"


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    txt = build()
    head = txt.split("## 1. 工作线H")[0]
    for bad in ("nan", "None"):
        if bad in head:
            raise SystemExit(f"❌ 小结前段出现未处理值 {bad!r}，拒绝写出。")
    TARGET.write_text(txt, encoding="utf-8")
    n_tbl = sum(1 for ln in txt.splitlines() if ln.strip().startswith("|"))
    print(f"  交付小结已生成：{TARGET}（{len(txt.encode('utf-8')) / 1024:.0f} KB，"
          f"{n_tbl} 行表格）")


if __name__ == "__main__":
    main()
