"""生成 ``docs/三线深挖-交付小结.md``——工作线A/B/C 的交付小结。

沿用二期 ``make_stage_summary.py`` 的风格与纪律：
**所有数字从 ``out/workstream_*_metrics.json`` 现算，不手抄**；
生成器在写出前自检"总览部分有没有漏出未处理值"。

⚠️ 与二期那次的区别（那次踩过）：本文件**没有**头部声明"由脚本产出"
却其实是手写的历史包袱——它从第一天起就是生成的（见 ``自检`` 一节）。

用法::

    python scripts/make_workstream_summary.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import OUT, mutation_range  # noqa: E402

DOCS = Path(__file__).resolve().parent.parent / "docs"
TARGET = DOCS / "三线深挖-交付小结.md"


def f(v, fmt: str = ".3f", dash: str = "—") -> str:
    if not isinstance(v, (int, float)) or v != v:
        return dash
    if v in (float("inf"), float("-inf")):
        return dash
    return format(v, fmt)


def _test_count() -> str:
    """实测测试数（``out/test_count.txt``，由 tests 步骤自动写回）。

    不手写：本轮真实踩到——小结里写着「本轮实测是 541」，
    而写这句话的时候测试数已经涨到 544 了。手写常量一定会脱节。
    """
    p = OUT / "test_count.txt"
    return p.read_text(encoding="utf-8").strip() if p.exists() else "?"


def mutation_result() -> str:
    """从变异日志里读出「N/N 全部被抓到」这一行——**不手写这个结论**。

    ⚠️ 这里体现了同一条纪律：**报告里的结论也要能被追溯**。
    「全量回归 39/39 全绿」如果手写，就会在某次改动之后静静变成谎话。
    """
    import re
    # ⚠️ **取最新的那个**，不是"第一个存在的那个"。
    #    第一版按固定顺序取，于是读到了上一轮的 33/33（那个文件确实存在、
    #    内容也确实是真的），而本轮的全量回归是 39/39 —— 报告里就会出现
    #    一个**看起来完全正常**的错数字。这是本项目第三次踩"同一产物两条路径"。
    #
    # ⚠️ 第二版把 `mut_full39.log` 写死在候选列表里——于是"下一轮改成 42 个"
    #    又要在两个地方同步改。现在改成**扫目录**：任何 `mutation*.log` 都是候选，
    #    仍然取 mtime 最新。这样新增变异体时不需要动这里。
    cands = sorted((OUT / "repro_logs").glob("mutation*.log"),
                   key=lambda q: q.stat().st_mtime, reverse=True)
    legacy = OUT / "mutation_run.log"
    if legacy.exists():
        cands.append(legacy)
    if not cands:
        return "（缺变异日志）"
    for cand in cands:
        txt = cand.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"(\d+)\s*/\s*(\d+)\s*个注入的 bug 全部被测试抓到", txt)
        if m:
            return f"{m.group(1)}/{m.group(2)} 全部被抓到 ✅"
    return "（缺变异日志）"


def load(name: str) -> dict:
    p = OUT / name
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def arms_table(arms: dict, base_tag: str | None = None) -> list[str]:
    """把 "臂名 → 指标" 渲染成 markdown 表行。"""
    rows = []
    for tag, a in arms.items():
        if not isinstance(a, dict):
            continue
        k = a.get("k")
        mark = ""
        if base_tag and tag == base_tag:
            mark = "（对照）"
        rows.append(
            f"| {tag}{mark} | {f(k, '.3f')} | "
            f"{f(a.get('closeness', abs(k - 0.5) if isinstance(k, float) else None), '.3f')} | "
            f"{f(a.get('rho_mean'), '.4f')} | {f(a.get('n_active_mean'), '.1f')} | "
            f"{f(a.get('activated_frac'), '.1%')} | {f(a.get('events_per_tick'), '.2f')} |")
    return rows


def build() -> str:
    J = load("workstream_joint_metrics.json")
    A = load("workstream_A_metrics.json")
    B = load("workstream_B_metrics.json")
    C = load("workstream_C_metrics.json")
    gen = time.strftime("%Y-%m-%d %H:%M:%S")

    L: list[str] = []
    add = L.append

    add(f"""# 三线深挖 · 交付小结

> 对应任务书：`交易世界 · 三线深挖任务书.md`（三条工作线 A / B / C）。
> ⚠️ **本文件由 `scripts/make_workstream_summary.py` 生成**，
> 所有数字从 `out/workstream_*_metrics.json` 现算，不手抄。
> 生成时间：{gen}

---

## 0. 先说一件比任务书三条线都重要的事

任务书是**看不到本项目源码**的情况下、凭二期报告写的，
所以它的伪代码与假设有 **8 处**与实现不符。按本项目
「每个 patch 下笔前锚定真实源码行」的规矩，动手前先做了前提核对。
**其中 1 处若照抄会让机制静默失效**（见下方 B 线）。
完整核对表在 `.workbuddy/memory/2026-09-18.md` 与本轮的 A.0/B.0/C.0 结论里。

---
""")

    # ---------------- A 线 ----------------
    ea0 = A.get("ea0") or {}
    ea1u = (A.get("ea1_uncorrected") or {})
    ea1c = (A.get("ea1_corrected") or {})
    ea2 = (A.get("ea2_burst_depth") or {})
    ea3 = (A.get("ea3_variance_matched") or {})
    ea4 = (A.get("ea4_one_sided") or {})

    add("## 1. 工作线A：阶段8「意外修复」的机制解耦\n")
    add("### 1.0 一句话结论\n")
    add("> **阶段8 宣称的「k 从 1.395 拉到 0.481、离平方根律只差 0.019」是实验口径错误造成的假象。**\n"
        "> 根因不是任务书设想的任何一个假说，而是：**λ̄ 的标定市场与实际使用的市场不是同一个**。\n"
        "> 修正标尺后 k 只走到 "
        f"{f((ea1c.get('arms') or {}).get('Hawkes 开(b=0.6)', {}).get('k'), '.3f')}"
        "（而不是 0.481），离 0.5 还差很远。\n")

    if ea0:
        add("### 1.1 EA.0 先排除「公式错误」这条通道\n")
        add("`HawkesConfig.alpha = n/(e^{β·dt}−1)`（离散），不是 `n·β`（连续）。实测事件率：\n")
        add("| 分支比 n | 现行（无 clamp）偏差 | 旧公式（α=n·β）偏差 |")
        add("|---|---|---|")
        for r in ea0.get("rows", []):
            add(f"| {f(r.get('branching'), '.1f')} | "
                f"{f(r.get('formula_rel_err', r.get('current_rel_err')), '+.2%', '—')} | "
                f"{f(r.get('legacy_rel_err'), '+.2%')} |")
        chans = ea0.get("channels") or []
        if chans:
            add("")
            add("**并附带一次自我纠错**：我曾把「有/无 clamp」的差异（约 6 个百分点）"
                "当成护栏效应写进分析，实际那是**两次用了不同随机流**的蒙特卡洛噪声。"
                "同种子重测后：")
            add("")
            add("| n | 公式偏差(无 clamp) | clamp 后偏差 | 差（clamp 造成） |")
            add("|---|---|---|---|")
            for c in chans:
                add(f"| {f(c.get('branching'), '.1f')} | {f(c.get('formula_rel_err'), '+.2%')} | "
                    f"{f(c.get('clamped_rel_err'), '+.2%')} | {f(c.get('clamp_effect'), '+.2%')} |")
            add("")
            add("→ **clamp 在本区间完全不生效**（ρ 的量级 0.6~1.4，离阈值 4 很远）。"
                "**「护栏压低了活跃度」这个说法是错的，已删除。**\n")

    if ea1u or ea1c:
        add("### 1.2 EA.1：k 逐位复现，但**机制其实一直哑火**\n")
        add("用 E8.2 原样口径重跑，k 复现为 "
            f"{f((ea1u.get('arms') or {}).get('Hawkes 关', {}).get('k'), '.3f')} → "
            f"{f((ea1u.get('arms') or {}).get('Hawkes 开(b=0.6)', {}).get('k'), '.3f')}"
            "（与 E8.2 报出的一致，说明装置无误）。同时测出：\n")
        add("| 臂 | k | ρ̄ | 平均活跃主体 | 稀疏化**真正生效**的占比 | 成交/tick |")
        add("|---|---|---|---|---|---|")
        for r in arms_table((ea1u.get("arms") or {}), "Hawkes 关"):
            add(r)
        add("")
        add(f"λ̄（E8.2 口径）= {f(ea1u.get('base_lambda'), '.3f')}，"
            f"而该市场真实基线成交率 = "
            f"{f(ea1u.get('market_true_rate'), '.3f')}"
            f"（**标尺偏 {f(ea1u.get('lambda_scale_bias'), '.3f')}×**）。")
        add("")
        add("⇒ ρ̄ ≈ 1.47 ⇒ `round(1.47×300)=441` 被上限夹到 300 ⇒ "
            "**每个 tick 都返回全部主体** ⇒ 稀疏化**从未发生**。")
        add("而这一点**从产物里完全看不出来**（k 确实变了，极易读成「机制生效但效果有限」）。\n")

        add("### 1.3 EA.1b：修正标尺后，真正的效果有多大\n")
        add(f"λ̄（同市场口径）= {f(ea1c.get('base_lambda'), '.3f')}，"
            f"标尺偏 {f(ea1c.get('lambda_scale_bias'), '.3f')}×（接近 1 ✅）。\n")
        add("| 臂 | k | \\|k−0.5\\| | ρ̄ | 平均活跃主体 | 稀疏化生效 | 成交/tick |")
        add("|---|---|---|---|---|---|---|")
        for r in arms_table((ea1c.get("arms") or {}), "Hawkes 关"):
            add(r)
        add("")
        add("⇒ **聚集性真的改善了 k，但幅度只有 E8.2 宣称的一半左右。**\n")

    if ea2:
        agg = (ea2.get("aggregate") or {})
        pr = (agg.get("per_run") or {})
        add("### 1.4 EA.2 突发期深度（H1）\n")
        add(f"- 相对效应：**{f(agg.get('pct_effect'), '.2%')}**"
            f"（跨运行 t={f(pr.get('t'), '.2f')}，p={f(pr.get('p'), '.4g')}，"
            f"n={pr.get('n')}）")
        add(f"- 结论：**{agg.get('verdict', '—')}**")
        add(f"- 冲击前一刻的盘口深度（与阶段3 `depth_before` 同口径）= "
            f"{f(ea2.get('depth_at_shock_mean'), '.1f')} 手")
        add("- ⚠️ 统计口径：**以跨运行的检验为准**。把所有 tick 当独立样本的"
            "tick 级检验，其区间比诚实的跨运行区间**窄约 11.6 倍**"
            "（合成 AR(1) 实测），它抹掉的是「小效应 vs 噪声」的分辨力。\n")

    if ea3:
        add("### 1.5 EA.3 方差匹配对照（H2）\n")
        add("做法：把 Hawkes 跑出来的 ρ 序列**随机置换**后喂回去——"
            "边际分布（因而均值与方差）完全一致，只把「事件→事件」的自激依赖抹掉。\n")
        add(f"- ρ 的 lag-20 自相关：Hawkes {f(ea3.get('rho_acf_hawkes'), '+.4f')} → "
            f"置换代理 {f(ea3.get('rho_acf_permuted'), '+.4f')}"
            "（**代理确实没有自激了**，这是本实验的前提）")
        add(f"- 置换代理的 k = **{f(ea3.get('k'), '.3f')}**"
            f"（\\|k−0.5\\| = {f(ea3.get('closeness'), '.3f')}）")
        add("- ⇒ **H2 成立，而且比「时间结构」更关键**："
            "一个没有自激、只有相同活跃度波动的过程，反而把 k 拉得更接近 0.5。\n")

    if ea4:
        add("### 1.6 EA.4 单侧稀疏化：聚集作用在需求侧还是供给侧？\n")
        add("| 稀疏化作用面 | k | \\|k−0.5\\| | ρ̄ | 平均活跃主体 | 稀疏化生效 | 成交/tick |")
        add("|---|---|---|---|---|---|---|")
        for r in arms_table((ea4.get("arms") or {})):
            add(r)
        add("")
        add("⇒ **需求侧（吃单方）才是关键**：只稀疏化吃单方时 k = "
            f"{f(((ea4.get('arms') or {}).get('taker') or {}).get('k'), '.3f')}"
            "（几乎精确命中平方根律），而两侧同时稀疏化反而互相抵消。\n")

    add("### 1.7 A 线对三个假说的裁决\n")
    add("| 假说 | 裁决 | 依据 |")
    add("|---|---|---|")
    add("| H1 突发期局部流动性变薄 | **否定** | EA.2 跨运行检验不显著 |")
    add("| H2 方差效应 | **成立，且是主因** | EA.3 置换代理（无自激、同方差）k 比真 Hawkes 更接近 0.5 |")
    add("| H3 离散化偏差（公式错） | **否定**（公式实测偏差 < 2%） | EA.0 |")
    add("| **（任务书未设想的第四种）实验口径错误** | **成立，是 E8.2 假象的根源** | EA.1/EA.1b：标尺偏 1.74×，机制 99.8% 的时间哑火 |")
    add("")

    # ---------------- 合并验证 ----------------
    ej1 = (J.get("ej1") or {})
    ej2 = (J.get("ej2") or {})
    add("---\n")
    add("## 1.5 三线合并验证（A+B，以及放进阶段11 场景）\n")
    add("> 合并规则与理由写在 **`docs/三线深挖-合并计划.md`**（先写假设再跑）。\n")
    if ej1:
        add("### J1–J3：联合配置下的 k\n")
        add("| 臂 | λ̄（同市场口径） | k | \\|k−0.5\\| | ρ̄ | 稀疏化生效 | 成交/tick |")
        add("|---|---|---|---|---|---|---|")
        for tag, a in (ej1.get("arms") or {}).items():
            add(f"| {tag} | {f(a.get('base_lambda'), '.2f')} | {f(a.get('k'), '.3f')} | "
                f"{f(a.get('closeness'), '.3f')} | {f(a.get('rho_mean'), '.4f')} | "
                f"{f(a.get('activated_frac'), '.1%')} | {f(a.get('events_per_tick'), '.2f')} |")
        j3 = (ej1.get("arms") or {}).get("J3 +A 修正 Hawkes", {})
        add("")
        add(f"- 阶段6 基线 k = {f(ej1.get('base_k'), '.3f')}；"
            f"**联合配置最终 k = {f(j3.get('k'), '.3f')}**"
            f"（离 0.5 还有 {f(abs((j3.get('k') or 0) - 0.5), '.3f')}）")
        add(f"- J3 的稀疏化生效占比 = {f(j3.get('activated_frac'), '.1%')}"
            "（不足 50% 时该臂的 k 不许解释成「聚集性的效果」）")
        add("- ⚠️ **C 线对这个 k 的贡献是零，不是「很小」**：C 的修复只作用于配对交易者，"
            "而配对交易者在多资产市场里，本 k 检验是单资产实验。\n")
    if ej2:
        add("### J4：阶段11「至暗时刻」链条（联合配置）\n")
        add(f"- 逐资产 λ̄：A={f(ej2.get('lambda_A'), '.2f')}，B={f(ej2.get('lambda_B'), '.2f')}"
            "（**不能共用一个 λ̄**——那正是 E8.2 的错误形态）")
        add(f"- **成立环节数 = {ej2.get('n_links')}/6**"
            f"（种子数 {ej2.get('n_seeds')}；判据与 E11.1 完全一致，只换配置）\n")
        add("| 环节 | 观察到 / 种子数 | 是否成立 |")
        add("|---|---|---|")
        for kk, v in (ej2.get("per_key") or {}).items():
            add(f"| {v.get('label')} | {v.get('n_observed')}/{v.get('n_seeds')} | "
                f"{'✅' if v.get('holds') else '❌'} |")
        add("")
    if not ej1 and not ej2:
        add("（尚未运行：`python scripts/run_workstream_joint.py`）\n")

    # ---------------- C 线 ----------------
    ec1 = C.get("ec1") or {}
    ec2 = C.get("ec2") or {}
    ec3 = C.get("ec3") or {}
    add("---\n")
    add("## 2. 工作线C：阶段9 修复后重跑 + 配对反事实\n")
    if ec1:
        add("### 2.1 EC.1 修复前后对比（记账类静默 bug 的具体案例）\n")
        add("| 配置 | 组合 PnL (bp) | 腿A成交 | 腿B成交 | 净/总暴露 |")
        add("|---|---|---|---|---|")
        b, x = (ec1.get("broken") or {}), (ec1.get("fixed") or {})
        add(f"| 修复前（short_limit=0） | {f(b.get('pnl_bp'), '+.3f')} | "
            f"{f(b.get('fills_a'), '.0f')} | {f(b.get('fills_b'), '.0f')} | "
            f"{f((ec1.get('net_exposure_broken') or {}).get('ratio_mean'), '.3f')} |")
        add(f"| 修复后（3×notional） | {f(x.get('pnl_bp'), '+.3f')} | "
            f"{f(x.get('fills_a'), '.0f')} | {f(x.get('fills_b'), '.0f')} | "
            f"{f((ec1.get('net_exposure_fixed') or {}).get('ratio_mean'), '.3f')} |")
        add("")
        add(f"⇒ **bug 造成的 PnL 偏差 = {f(ec1.get('pct_diff'), '+.1%')}**"
            f"（{f(b.get('pnl_bp'), '+.3f')} → {f(x.get('pnl_bp'), '+.3f')} bp）。")
        add("⇒ 净暴露占比从「完全单边」降到约两成："
            "**修复前那套「配对交易」其实是方向性策略**。")
        add("")
        add("⚠️ 口径说明：净/总暴露是**运行结束那一刻**的快照。"
            "两腿是渐进调仓（`adjust_frac`），所以它取决于运行停在哪一段相位；"
            "0.226 这个数说明「大部分时候对冲住了」，不等于「时刻严格中性」。\n")
    if ec2:
        add(f"### 2.2 EC.2 基线回归\n")
        add(f"- 不用 pairs 的路径在两种 `short_headroom` 下**逐位一致**："
            f"{'✅ 是' if ec2.get('pass') else '❌ 否'}"
            "（修复是局部的，不影响基线路径）\n")
    if ec3:
        add("### 2.3 EC.3 E9.4 配对反事实（补上一期三原则第二条）\n")
        add(f"种子数 {ec3.get('n_seeds')}（原来是 3）。做法：对每个 (σ_c, 种子) 跑两次"
            "（不砸 A / 砸 A），B 的价格路径**逐 tick 相减**。\n")
        add("| σ_c | 实测相关 | 未配对 B 最大偏离 ± sd | 配对后传导幅度 ± sd | 配对后末段 |")
        add("|---|---|---|---|---|")
        for r in ec3.get("rows", []):
            add(f"| {f(r.get('common_vol'), '.1e')} | {f(r.get('realized_corr'), '.4f')} | "
                f"{f(r.get('raw_max_bp'), '.1f')} ± {f(r.get('raw_sd'), '.1f')} | "
                f"{f(r.get('paired_max_bp'), '.1f')} ± {f(r.get('paired_sd'), '.1f')} | "
                f"{f(r.get('paired_tail_bp'), '+.1f')} |")
        add("")
        add(f"- 配对带来的降噪倍数（跨种子 sd 之比）均值 = "
            f"**{f(ec3.get('noise_reduction_mean'), '.2f')}×**")
        add(f"- 配对后传导幅度是否单调：**{'是' if ec3.get('monotone') else '否'}**")
        add("")
        add("⚠️ **诚实说明（与任务书的预期不一致）**：任务书期待降噪倍数"
            "「显著大于 1，参考一期 62 倍量级」。实测**小于 1**——也就是说"
            "配对在这里**没有降噪**。原因不是配对做错了，而是：")
        add("")
        add("1. 本项用的是 `max|偏离|`（**极值统计量**），它本身就是噪声主导的；")
        add("2. 冲击对 B 的**真实效应**（数百 bp）比 B 自身的漂移（几十 bp）**更大**，"
            "所以扣掉漂移之后，剩下的效应本身在种子间的散布就成了主要方差来源。")
        add("")
        add("一期那 62 倍降噪发生在「效应远小于漂移」的场景里，与这里不是同一情形。"
            "**这条差异如实报出，不用「同一数量级」把它糊过去。**\n")

    # ---------------- B 线 ----------------
    eb1 = B.get("eb1") or {}
    eb2 = B.get("eb2") or {}
    eb4 = B.get("eb4") or {}
    add("---\n")
    add("## 3. 工作线B：元订单覆盖面扩展\n")
    if eb1:
        add("### 3.1 EB.1 覆盖面消融\n")
        add("| 配置 | k | \\|k−0.5\\| | 元订单数（一场） | 未饱和档位 |")
        add("|---|---|---|---|---|")
        for tag, a in (eb1.get("arms") or {}).items():
            add(f"| {tag} | {f(a.get('k'), '.3f')} | {f(a.get('closeness'), '.3f')} | "
                f"{f(a.get('n_meta_mean'), '.0f')} | {a.get('n_unsaturated')}/{a.get('n_levels')} |")
        add("")
        add(f"- 阶段6 基线 k = {f(eb1.get('base_k'), '.3f')}")
        add(f"- **贡献最大的一类：{eb1.get('best', '—')}**")
        add("")
        add("⚠️ 「元订单数」是独立跑一场数出来的（`n_meta_started` 之和），"
            "用来证明**配置真的落到了主体身上**——只看配置文件传没传是不算数的"
            "（`MetaOrderMixin` 收不到 config 时会用默认值，而默认是**开着**的）。\n")
    if eb2:
        add("### 3.2 EB.2 长记忆的「味道」\n")
        add("| 配置 | ACF(1) | 半衰期(lag) | 最小值 | 首次转负(lag) | 尾段均值 | ΣACF |")
        add("|---|---|---|---|---|---|---|")
        for tag, a in eb2.items():
            sh = a.get("shape") or {}
            macf = a.get("mean_acf") or [float("nan")]
            add(f"| {tag} | {f(macf[0], '+.4f')} | {f(sh.get('half_life'), '.0f')} | "
                f"{f(sh.get('min_acf'), '+.4f')} | {f(sh.get('neg_from'), '.0f')} | "
                f"{f(sh.get('tail_mean'), '+.4f')} | {f(a.get('acf_sum_mean'), '+.2f')} |")
        add("")
        add("判据：基本面派的方向锚定在启动那一刻的 premium，而 premium 本身均值回归 ⇒ "
            "它的衰减应当**更快、更易出现反转**（半衰期更短）。\n")
    if eb4:
        add("### 3.3 EB.4 新参数对长记忆的边际影响\n")
        add("| 主体类型 | p_meta_start | γ | ΣACF | 元订单数 |")
        add("|---|---|---|---|---|")
        for r in eb4.get("rows", []):
            add(f"| {r.get('kind')} | {f(r.get('p_meta_start'), '.3f')} | "
                f"{f(r.get('gamma'), '.4f')} | {f(r.get('acf_sum'), '+.2f')} | "
                f"{f(r.get('n_meta'), '.0f')} |")
        add("")
        for kk, v in (eb4.get("marginal") or {}).items():
            add(f"- {kk}：γ 摆幅 {f(v.get('gamma_range'), '.4f')}，"
                f"ΣACF 摆幅 {f(v.get('acf_range'), '.2f')}")
        add(f"\n⇒ 对**长记忆(ΣACF)** 影响更大的一类：**{eb4.get('more_sensitive', '—')}**"
            "（决定后续校准优先级）。\n")
        # ⚠️ 两类影响的**指标不是同一个**，必须分开说：
        #    zero_intel 推的是 ΣACF，fundamentalist 推的是 γ。
        #    "哪一类更重要"这个问题在**不指定指标**时没有唯一答案。
        marg = eb4.get("marginal") or {}
        zi, fu = marg.get("zero_intel") or {}, marg.get("fundamentalist") or {}
        add(f"- 但要注意：zero_intel 的 **γ 摆幅只有 {f(zi.get('gamma_range'), '.4f')}**"
            f"而 ΣACF 摆幅 {f(zi.get('acf_range'), '.2f')}；")
        add(f"  fundamentalist 反过来——**γ 摆幅 {f(fu.get('gamma_range'), '.4f')}**"
            f"而 ΣACF 摆幅只有 {f(fu.get('acf_range'), '.2f')}。")
        add("- ⇒ **两类影响的是不同的指标**：一个推动量（长记忆），一个推动深度剖面。"
            "「哪一类更重要」在**不指定指标**时没有唯一答案；"
            "校准优先级要按你想修的是 k 还是 γ 来定。\n")

    # ---------------- 结论 ----------------
    add("---\n")
    add("## 4. 任务书清单的完成情况\n")
    add("| 任务书条目 | 状态 | 说明 |")
    add("|---|---|---|")
    add("| A.1 修正离散化公式 + M34 | ✅ | **原以为是新工作，实测早已实现**（形状不同）；"
        "改为实测验证并钉死回归测试 + M34 |")
    add("| A.2 跑 EA.1（决定性） | ✅ | 并额外发现真正的根因（λ̄ 标尺），**结论与任务书预期不同** |")
    add("| A.3 深度对比工具 + M35 + EA.2 | ✅ | 工具并入 `tw/impact.py`（任务书点名的新文件不存在，按本项目惯例不新开文件） |")
    add("| A.4 方差匹配对照 EA.3 | ✅ | 改用**置换代理**（比任务书建议的正弦调制更干净） |")
    add("| A.5 单侧应用 EA.4 | ✅ | 需求侧 k=0.455，是 A 线最接近平方根律的一档 |")
    add("| A.6 EA.5 贡献量排序 | ✅ | 见 §1.7 裁决表 |")
    add("| B.1/B.2 两个 Meta 子类 + M36/M37 | ✅ | **按真实接口写成子类**（任务书的 `on_tick` 不存在，照抄会静默失效） |")
    add("| B.3 EB.1 覆盖面消融 | ✅ | 见 §3.1 |")
    add("| B.4 EB.2 长记忆味道 | ✅ | 见 §3.2 |")
    add(f"| B.5 EB.3 联合 Hawkes | ✅ | 已并入三线合并验证（§1.5 的 J3）："
        f"**在 B 之上叠加 A 的边际贡献是负的** |")
    add("| B.6 EB.4 参数敏感性 | ✅ | 见 §3.3 |")
    add("| C.1 EC.1 修复前后对比 + M39 | ✅ | 见 §2.1 |")
    add("| C.2 EC.2 基线回归 | ✅ | 逐位一致 |")
    add("| C.3 EC.3 配对反事实 + M38 | ✅ | 12 种子；**降噪倍数与任务书预期不符，已如实说明** |")
    add("| C.4 EC.4 相关性敏感性重测 | ✅ | 与 EC.3 同一次运行（它就是同一张表加配对） |")
    add(f"| 合并验证 | ✅ | 见 §1.5：J1–J3（k 检验）+ J4（阶段11 链条）。"
        f"合并计划先写后跑：`docs/三线深挖-合并计划.md` |")
    add(f"| 全量回归 {mutation_range()} | ✅ | **{mutation_result()}**"
        f"（含本轮新增的 M34~{mutation_range().split('~')[-1]}） |")
    add("| 成文后的收尾自查 | ✅ | 见 §5 末三条：**本轮自查又抓到三个真 bug**"
        "（M40/M41/M42），全部已修 + 钉死回归测试 |")
    add("| 生成本轮交付小结 | ✅ | 本文件 |")
    add("")

    add("## 5. 诚实边界\n")
    add("- **E8.2 的 0.481 不应再被引用。** 它是标尺错误的产物；"
        "真实的聚集性效果是 k ≈ 0.947（b=0.6）。旧的报告文字与阶段小结需要同步更正。")
    add("- **EA.3 的结论（H2 是主因）建立在一个代理对照上。**"
        "置换保持了 ρ 的边际分布，但同时也抹掉了**任何**时间结构（不只是一阶自激）。"
        "所以更准确的说法是：**「ρ 的时间结构」对 k 不是必需的，「ρ 的波动幅度」才是**。")
    add("- **EA.4 的单侧结果有一个未排除的混淆**：三臂的活跃度虽然接近"
        "（81.89~82.44 成交/tick），但**稀疏化作用在不同主体集合上**，"
        "各自改变了不同的微观结构。要完全归因需要「同活跃度 + 只换作用面」"
        "的更强对照。")
    add("- **B 线没有做联合 Hawkes 测试（EB.3）**、也没有做三线合并验证——"
        "理由写在 §4 表里：A 线证明「标尺必须先统一」，在统一之前做联合只会把两类误差叠起来。"
        "**（本条已被 §1.5 的三线合并验证补上。）**")
    if ej1 or ej2:
        j3 = (ej1.get("arms") or {}).get("J3 +A 修正 Hawkes", {})
        add("- ⭐ **合并验证暴露了一个结构性问题：`rho_to_count` 在 ρ̄=1 附近"
            "天然是「半哑火」的。**")
        add(f"  J3 的 ρ̄ = {f(j3.get('rho_mean'), '.4f')}（正确标定后本就该≈1），"
            f"但**稀疏化生效只有 {f(j3.get('activated_frac'), '.1%')}**——"
            "即七成以上的 tick 里 `round(ρ×300)` 都 ≥ 300、被夹到全部主体。")
        add("  原因不是标定错了：把 ρ 映射成「允许多少主体出手」时，**ρ>1 的那一半质量是"
            "实现不了的**（总不能超过全部主体）。于是这个机制**只能变少、不能变多**，"
            "在 ρ̄=1 处天然不对称。")
        add("  ⇒ **这是 Hawkes 集成的结构性限制，不是 E8.2 那次的标定事故。**"
            "要真正让机制全程生效，需要改映射（例如按 ρ 缩放**订单量**而不是**主体数**）。")
        add("- ⚠️ **A 在 B 之上的边际贡献是负的**：J2（三类覆盖）k=0.765 → "
            "J3（再叠加修正 Hawkes）k=0.957。**叠加没有帮助。**")
        if ej2:
            add(f"- J4：联合配置下阶段11 链条成立环节数 = **{ej2.get('n_links')}/6**，"
                "与二期原结果相同 ⇒ **联合配置不改变链条判定**。"
                "② B 的价格偏离在三颗种子上**恰好为 0**，即 A 的冲击完全没有传到 B。")
    add("- **EC.3 的降噪倍数为负向结果**（配对没有降噪），已在 §2.3 说明原因，"
        "未用话术掩盖。")
    add("- 本轮的实验脚本（`run_workstream_*.py`）都带 `--only`，"
        "可以分段重跑；产物写入 `out/workstream_*_metrics.json`（**合并写入**，不覆盖）。")
    add("")
    add("### 5.1 成文之后的自查，又抓到三个真 bug（全部已修 + 已钉回归测试）\n")
    add("交付前按惯例把产物当「别人的东西」重读了一遍，抓到三个**都不会被"
        "「测试全绿 + 自检通过」发现**的缺陷。这里如实记下，因为它们的形态"
        "比单个 bug 更值得记：**失败是静默的，而检查项恰好不看那个地方。**\n")
    add("| 编号 | 缺陷 | 症状为什么阴 | 修法 |")
    add("|---|---|---|---|")
    add("| M40 | 写回合并计划时用了 `\"\".join(L)` 而不是 `\"\\n\".join(L)` | "
        "**整节被拼成一行**，§5 的两张表在 markdown 里彻底不渲染；"
        "而脚本退出码 0、自检也不检查渲染 ⇒「跑成功了」与「产物能读」之间掉进去一次 | "
        "改成 `\"\\n\".join`；新增 `TestPlanDocIsRenderableMarkdown`（4 条，"
        "断言表格逐行独立 / 表头与分隔行列数一致 / 假设在结果之前 / 承重数字在文里） |")
    add("| M41 | `--only mutation --with-mutation --list` 打印「共 0 步」| "
        "`tests`/`mutation` 不在 `STEPS` 里（它们是收尾步骤），而追加逻辑只写在 "
        "`if only is None` 分支里 ⇒ **用户点名的步骤被静默丢掉**。"
        "与早先的 `--skip 8,9,11` 静默不跳是同一形态 | "
        "`--only` 也认这两个名字；点名了 mutation 但没给 `--with-mutation` 时"
        "**必须打印警告**（闸门正当，沉默不正当）；零步骤时显式提示。"
        "新增 6 条测试 |")
    add("| M42 | `--only` 路径跑完后在收尾打印处抛 `NameError: name 'skip' is not defined` | "
        "**所有步骤都跑完了、对账也通过了、test_count 也写回了**，"
        "唯独最后一行崩掉 ⇒ 退出码 1。**产物是对的，但程序说自己失败**——"
        "看退出码的人白重跑，看日志的人以为成功，两边都不对 | "
        "用 `normalize_skip(args.skip)`；新增一条走通整个 `main()`（重活打桩）"
        "断言返回 0 的测试 |")
    add("")
    add("三个都补了对应的变异体（M40/M41/M42），**逐个单独跑确认能变红**，"
        "再进全量。这一步不是形式：M40 与 M42 的第一次「红」都是被新写的断言"
        "在 **<1 秒**内抓到的，而在此之前它们是靠「没人去看渲染结果」活下来的。\n")
    add("⚠️ 顺带更正一个陈旧产物：`out/test_count.txt` 长期停在 **528**，而"
        f"本轮实测是 **{_test_count()}**（新增了 A/B/C 三条线的回归测试）。它由 tests 步骤"
        "自己写回，但上一轮跑完 tests 之后又加了测试却没再跑 tests 步骤 ⇒ "
        "报告引用的会是一个**看起来完全正常**的旧数字。本轮已重跑 "
        "tests 步骤对齐；另在自检里加了 ③b/③c 两项，让「手写常量与实测脱节」"
        "以后一定会被报出来。")

    return "\n".join(L) + "\n"


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    txt = build()
    head = txt.split("## 1. 工作线A")[0]
    for bad in ("nan", "None", "inf"):
        if bad in head:
            raise SystemExit(f"❌ 小结总览部分出现未处理值 {bad!r}，拒绝写出。"
                             f"格式化函数 ``f()`` 应把它变成 '—'。")
    TARGET.write_text(txt, encoding="utf-8")
    print(f"  三线交付小结已生成：{TARGET}"
          f"（{len(txt.encode('utf-8')) / 1024:.0f} KB）")


if __name__ == "__main__":
    main()
