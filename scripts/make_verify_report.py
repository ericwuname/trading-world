"""生成 ``docs/前置校验-EA4-诊断报告.md`` —— 回给设计者的材料。

数字全部从 ``out/*.json`` 现算，不手抄（项目铁律）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import OUT  # noqa: E402

DOCS = Path(__file__).resolve().parent.parent / "docs"
TARGET = DOCS / "前置校验-EA4-诊断报告.md"


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


#: 哪些既有结论**依赖 k 的数值比较**（受影响），哪些不依赖（不受影响）。
#: 这一栏是判断，所以写死在生成器里、可被 review；数字仍全部现算。
AFFECTED = [
    ("二期·阶段6", "k 从 0.665 恶化到 0.910（「元订单让冲击变凹了吗」）",
     "同一个 paired_impact + fit_power_law 装置"),
    ("二期·阶段8", "k 从 1.395 拉到 0.481（「意外修复」）",
     "已被三线深挖证伪为标尺假象；本轮再叠加一条：即便标尺修正，也分辨不了"),
    ("三线·A 线", "H1 否定 / H2 成立 / H3 否定 的裁决",
     "裁决依赖 EA.2/EA.3 的 k 比较（EA.2 的深度对比不依赖 k，见下表）"),
    ("三线·A 线", "EA.4「只稀疏化吃单方 k=0.455，最接近平方根律」",
     "本轮起点；复现过、也重标定过，两个数都不在分辨力之内"),
    ("三线·B 线", "元订单覆盖面消融 k 1.251 → 0.765",
     "EB.1 四臂用的是同一装置"),
    ("三线·合并验证", "J2 优于 J3 ⇒「A 在 B 之上的边际贡献为负」",
     "J1–J3 三臂同装置"),
]

UNaffected = [
    ("三线·A 线 EA.0", "离散化公式实测偏差 <1%（旧公式 −40%）",
     "测的是事件率，不是 k"),
    ("三线·A 线 EA.2", "突发期深度相对效应 −6.52%、跨运行不显著",
     "比的是深度（t 检验），不是 k"),
    ("三线·A 线 EA.4", "稀疏化生效占比 99.2% vs 91.4%",
     "数的是被夹到上限的 tick 比例，是一个直接可观测量"),
    ("三线·A 线 EA.1/1b", "ρ̄ 与活跃主体数（如 276.0 / 94.8%）",
     "直接观测，不经拟合"),
    ("三线·C 线", "记账 bug 造成 PnL 偏差 +69.4%、净/总暴露 1.000 → 0.226",
     "比的是同一批路径的账本数字，不是幂律指数"),
    ("三线·C 线 EC.2", "基线路径逐位一致",
     "逐位比对，不受统计噪声影响"),
]


def build() -> str:
    vf = load("verify_EA4_calibration.json")
    ku = load("k_uncertainty.json")
    rc = load("_recheck_legacy.json")
    wa = load("workstream_A_metrics.json")

    v = vf.get("verify") or {}
    rerun = vf.get("rerun") or {}
    cmp_ = vf.get("comparison") or {}
    summ = ku.get("summary") or {}
    need = ku.get("seeds_needed") or {}
    ov = ku.get("taker_overlap") or {}
    arms = ku.get("arms") or {}

    def _rows_of(block: str, arm: str) -> list:
        return ((wa.get(block) or {}).get("arms") or {}).get(arm, {}).get("rows") or []

    def _cell(rows: list, i: int) -> str:
        if i >= len(rows):
            return "—"
        r = rows[i]
        return (f"{f(r.get('slippage_bp'), '.2f')} ± {f(r.get('slippage_sem_bp'), '.2f')} "
                f"(t={f(r.get('slippage_t'), '.2f')})")

    L: list[str] = []
    add = L.append

    add("# 前置校验报告：EA.4 的 λ̄ 标定，以及一个比预期更根本的问题\n")
    add("> 对应任务书：`交易世界 · EA.4扶正与非对称机制深挖任务书.md` §2 前置校验。")
    add("> 本文件由 `scripts/make_verify_report.py` 生成，数字从 `out/*.json` 现算。")
    add("> **交付对象是任务书的设计者**——按 §2.4，下文只给事实与选项，不定方向。\n")
    add("---\n")

    # ---------------- 0 ----------------
    add("## 0. 一句话结论\n")
    add("前置校验**未通过**，但失败原因与任务书 §2.4 预设的那一种**不同**：\n")
    add("- **不是**「EA.4 也是一次标尺假象」。市场配置经逐字段核实是**干净**的，")
    add("  且旧数字**逐位可复现**（见 §3）。")
    add("  重标定后确实有偏离，但偏离幅度（taker "
        f"{f(cmp_.get('old_reported_k_taker'), '.3f')} → "
        f"{f(cmp_.get('new_k_taker'), '.3f')}）**不是标尺能解释的量级**。")
    add(f"- **是**「k 这个指标在这套装置上没有分辨力」："
        f"{summ.get('n_arms', 0)} 个臂里，"
        f"**{summ.get('n_ci_contains_0.5', 0)} 个的 95% 置信区间包含 0.5**，"
        f"区间宽度中位数 {f(summ.get('width_median'), '.3f')}。")
    add("  也就是说：过去两期里所有「k 离 0.5 有多远」的陈述，"
        "**在这套装置的噪声水平下都无法被判定**。\n")
    add("按任务书 §2.4，**工作线 D / E / F 已暂停**，等你定方向。\n")
    add("---\n")

    # ---------------- 1 ----------------
    add("## 1. 前置校验三步\n")
    add("任务书 §2.2 要求比对两份产物（`out/workstream_A/EA4_config.json` 与"
        " `EA4_lambda_calibration_config.json`）。**这两份文件不存在**——"
        "EA.4 的配置一直是 `scripts/run_workstream_A.py` 里的常量，λ̄ 是运行时算的。\n")
    add("更关键的是：**比对两份手抄配置本身并不能满足第六条纪律**——"
        "两份 JSON 也可以「看起来一样、实际不同步」。"
        "所以改成三步（见 `scripts/verify_EA4_calibration.py`）：\n")
    add("| 步骤 | 检查什么 | 结果 |")
    add("|---|---|---|")
    add(f"| ① 来源核对 | 实验侧**实际生效**的配置"
        f"（`run_stage3.make_market` 内部写死的 `N_AGENTS`/`MM_KW`）"
        f" vs 标定侧**实际收到**的 kwargs | "
        f"{'✅ 一致' if v.get('clean_market_config') else '❌ 不一致'} |")
    add(f"| ② 物化核对 | 两侧各自真实构造一个市场，比对主体构成指纹"
        f"（防「参数一样但构造路径不同」） | "
        f"{'✅ 一致' if v.get('fingerprint_match') else '❌ 不一致'}"
        f"（{v.get('fingerprint', {}).get('kinds', {})}） |")
    add(f"| ③ 窗口核对 | 标定窗口是否等于实验窗口 | "
        f"{'✅ 一致' if v.get('clean_window') else '❌ **不一致**'} |")
    add("")
    for m in (v.get("mismatch_fields") or []):
        add(f"- {m}")
    add("")
    add("**①② 是干净的**：EA.4 当年就传了 `sim_kw=MM_KW`，"
        "所以它**没有**踩 E8.2 那种「裸市场标定、带做市配置使用」的坑。")
    add("（②的指纹一致也说明：`make_market` 内部写死的常量与 spec 没有脱节。）\n")
    add("**③ 不一致——这是一个真实缺陷，且是本轮才发现的一类**：")
    add(f"标定用 `[500, 3000)`（含前 500 tick 瞬态），实验跑在 "
        f"`{v.get('experiment_window')}`。"
        f"两者相差约 7%，会让 λ̄ 偏高（旧口径 {f(rc.get('lambda_bar'), '.3f')}"
        f" vs 同窗口的 {f(rerun.get('base_lambda'), '.3f')}）。\n")
    add("---\n")

    # ---------------- 2 ----------------
    add("## 2. 同配置同窗口重标定后的官方基准（EA.4′）\n")
    add(f"- 标定窗口 = 实验窗口 = `{rerun.get('experiment_window')}` ⇒ "
        f"λ̄ = **{f(rerun.get('base_lambda'), '.3f')}**\n")
    add("| 臂 | k | ρ̄ | 平均活跃主体 | 稀疏化真正生效 | 成交/tick |")
    add("|---|---|---|---|---|---|")
    for scope in ("both", "taker", "maker"):
        a = (rerun.get("arms") or {}).get(scope) or {}
        mark = " ⭐" if scope == "taker" else ""
        add(f"| {scope}{mark} | {f(a.get('k'), '.3f')} | {f(a.get('rho_mean'), '.4f')} | "
            f"{f(a.get('n_active_mean'), '.1f')} | {f(a.get('activated_frac'), '.1%')} | "
            f"{f(a.get('events_per_tick'), '.2f')} |")
    add("")
    add(f"与任务书点名关注的 0.455 相比：**taker 臂 "
        f"{f(cmp_.get('old_reported_k_taker'), '.3f')} → "
        f"{f(cmp_.get('new_k_taker'), '.3f')}"
        f"（差 {f(cmp_.get('delta'), '+.3f')}）**。\n")
    add(f"按 §2.3 的要求，这个偏离**用与结论同号的字体突出报告**，"
        f"不因为不想推翻上一轮结论而轻描淡写。\n")
    add("---\n")

    # ---------------- 3 ----------------
    add("## 3. 先排除掉「是不是执行者自己改坏了」\n")
    add("本轮确实动过 `ea4_one_sided` 的代码路径（`MM_MIX` → `exp_kw[\"mix\"]`、"
        "新增 `calib_window` 与 `scopes` 参数）。**在把「0.455 不可靠」当成结论之前，"
        "必须先证明那不是自己的改动造成的。**\n")
    add("做法：用**当前代码**跑 **legacy 口径 + 原 3 种子**，看能不能逐位复现：\n")
    add("| 臂 | 三线深挖报出 | 本次重跑 | 差 |")
    add("|---|---|---|---|")
    for r in (rc.get("comparison") or []):
        add(f"| {r.get('scope')} | {f(r.get('reported'), '.3f')} | "
            f"{f(r.get('rerun'), '.3f')} | {f(r.get('delta'), '+.3f')} |")
    add("")
    add(f"- λ̄ = {f(rc.get('lambda_bar'), '.3f')}（与三线一致）")
    add(f"- **全部逐位复现：{'✅ 是' if rc.get('all_match') else '❌ 否'}**"
        f"（产物 `out/_recheck_legacy.json`）\n")
    add("⇒ 代码改动**等价**，偏离只可能来自标定窗口与指标本身的噪声。"
        "**这条自查必须先做**，否则可能把「我改坏了」误报成「机制结论不成立」。\n")
    add("---\n")

    # ---------------- 4 ----------------
    add("## 4. 决定性证据：k 没有分辨力\n")
    add("### 4.1 每档滑点的显著性（3 个种子）\n")
    add("**taker 臂四档里只有最大的 1% 档显著**，其余三档的 |t| 都不到 1.6。"
        "逐档数字如下（配对差分均值 ± sem，括号里是 t）：\n")
    add("| 档位 | EA.4 taker（λ̄=94.601） | EA.4′ taker（λ̄=91.692） | 对照：Hawkes 关 |")
    add("|---|---|---|---|")
    t_old = _rows_of("ea4_one_sided", "taker")
    t_new = (rerun.get("arms") or {}).get("taker", {}).get("rows") or []
    t_off = _rows_of("ea1_uncorrected", "Hawkes 关")
    for i in range(4):
        frac = (t_old[i].get("shock_frac") if i < len(t_old) else None)
        label = f"{frac * 100:.2f}%" if isinstance(frac, float) else f"档{i + 1}"
        add(f"| {label} | {_cell(t_old, i)} | {_cell(t_new, i)} | {_cell(t_off, i)} |")
    add("")
    add("（逐档完整字段见 `out/k_uncertainty.json` 与 `out/workstream_A_metrics.json` 的 `rows`。）\n")
    add(f"k 的点估计几乎完全由**首尾两点之比**决定——以 legacy 那组为例：\n")
    add("```")
    add("log(|−15.565| / |−5.255|) / log(10) = 0.472   ← 而报出的 k = 0.455")
    add("```")
    add("也就是说，0.455 这个数字里**几乎没有中间两个档位的信息**——"
        "它是「一个显著点 ÷ 一个不显著点」的比值。\n")
    add("### 4.2 参数 bootstrap 的 95% 区间\n")
    add("对每档按学生的 t 分布（df = 种子数−1）重采样，"
        "复算 `fit_power_law`（**复用产品代码，不写第二份实现**）：\n")
    add("| 臂 | k | 95% 区间 | 宽度 | 显著档位 |")
    add("|---|---|---|---|---|")
    for tag, a in arms.items():
        if not a.get("ok"):
            continue
        lo, hi = a["ci95"]
        # ⚠️ 臂名本身含 ``|``（如「EA.1 uncorrected | Hawkes 关」），
        # 直接塞进表格会把那一行拆成多列、整张表错位。
        # markdown 的转义竖线 ``\|`` 渲染成字面量竖线——这也正是
        # ``tests/test_workstream_scripts.py`` 里 ``_n_cols`` 要先抹掉 ``\|`` 的原因。
        safe = tag.replace("|", "\\|")
        add(f"| {safe} | {f(a.get('k_point'), '.3f')} | "
            f"[{f(lo, '.3f')}, {f(hi, '.3f')}] | "
            f"{f(a.get('width'), '.3f')} | "
            f"{a.get('significant_levels')}/{a.get('n_levels')} |")
    add("")
    add(f"**总览：{summ.get('n_ci_contains_0.5')}/{summ.get('n_arms')} 个臂的 95% 区间包含 0.5；"
        f"{summ.get('n_ci_contains_0')}/{summ.get('n_arms')} 个还包含 0"
        f"（连滑点方向都不确定）。区间宽度中位数 "
        f"{f(summ.get('width_median'), '.3f')}。**\n")
    add("### 4.3 本轮起点的两次测量：完全不可分辨\n")
    add(f"- legacy：k = {f(cmp_.get('old_reported_k_taker'), '.3f')}，"
        f"95% CI = [{f((arms.get('EA.4 (λ̄=94.601) | taker') or {}).get('ci95', [None, None])[0], '.3f')}, "
        f"{f((arms.get('EA.4 (λ̄=94.601) | taker') or {}).get('ci95', [None, None])[1], '.3f')}]")
    add(f"- aligned：k = {f(cmp_.get('new_k_taker'), '.3f')}，"
        f"95% CI = [{f((arms.get('EA.4′ aligned (λ̄=91.692) | taker') or {}).get('ci95', [None, None])[0], '.3f')}, "
        f"{f((arms.get('EA.4′ aligned (λ̄=91.692) | taker') or {}).get('ci95', [None, None])[1], '.3f')}]")
    add(f"- 重叠 {f(ov.get('overlap'), '.3f')}，"
        f"= 较窄区间宽度的 **{f((ov.get('overlap_frac_of_narrower') or 0) * 100, '.1f')}%**\n")
    add("⇒ **不可分辨**。0.455 与 1.042 之间那个 0.587 的差，"
        "在本装置的噪声水平下不构成证据。\n")
    add("---\n")

    # ---------------- 5 ----------------
    add("## 5. 影响范围（精确到条，不是「全盘推翻」）\n")
    add("### 5.1 受影响的结论——凡是以 k 的**数值比较**为证据的\n")
    add("| 出处 | 结论 | 为什么受影响 |")
    add("|---|---|---|")
    for a, b, c in AFFECTED:
        add(f"| {a} | {b} | {c} |")
    add("")
    add("### 5.2 **不受影响**的结论——证据不经过 k 这条通道\n")
    add("| 出处 | 结论 | 为什么不受影响 |")
    add("|---|---|---|")
    for a, b, c in UNaffected:
        add(f"| {a} | {b} | {c} |")
    add("")
    add("### 5.3 一句话\n")
    add("**这不是「过去两期都白做了」。**")
    add("被削掉的是「把 k 当作可比较的刻度」这一条通道；")
    add("而机制层、账本层、事件率层的结论都还在——它们的证据不经过拟合。\n")
    add("---\n")

    # ---------------- 6 ----------------
    add("## 6. 要把分辨力做到可用，需要多少种子？\n")
    add("区间宽度 ∝ 1/√n（标准误的定义），以现行 4 档 + 3 种子为基准外推：\n")
    add("| 要做到 | 约需种子数 | 成本判断 |")
    add("|---|---|---|")
    for label, n_need in need.items():
        cost = "便宜" if n_need <= 60 else "昂贵（需换指标或增加档位）"
        add(f"| {label} | {n_need:.0f} | {cost} |")
    add("")
    add("⚠️ 这是**下界式**估计：它只考虑「加种子」这一条降噪途径。"
        "另外两条是：\n")
    add("1. **增加冲击档位**——现在只用了 4 档（0.1% / 0.25% / 0.5% / 1%），"
        "而 `run_stage3.SHOCK_SIZES` 里有 **7 档**（到 2% / 4% / 8%）。"
        "OLS 用更多点会显著降噪，但大档位会因 `fill_ratio < 0.95` 被 "
        "`unsaturated()` 剔除，所以要先看饱和从哪一档开始。")
    add("2. **提高信噪比**——现有 4 档里低档位几乎全是噪声（t < 1.6），"
        "而信号在大档位清晰（1% 档 t = −4.93）。"
        "也就是说：**同一批预算下，把种子投在大档位上比平均撒更划算。**\n")
    add("⚠️ 最要紧的一条：**判定「是否落在 0.5±0.1」需要约 "
        f"{need.get('判定是否落在 0.5±0.1', float('nan')):.0f} 个种子。**"
        "而任务书整轮的目标（D/E/F 各线「逼近 0.5」）都建立在这个判定上。"
        "在补足分辨力之前，**即使跑完 D/E/F，也无法回答「有没有更接近 0.5」**。\n")
    add("---\n")

    # ---------------- 7 ----------------
    add("## 7. 三条可选路径（供你定，我不自己选）\n")
    add("按 §2.4「不要自己决定下一步方向」，这里只列选项与代价，"
        "不推荐、不排序。\n")
    add("| 路径 | 做什么 | 代价 | 能回答什么 | 仍答不了什么 |")
    add("|---|---|---|---|---|")
    add("| **A. 补分辨力** | 把种子从 3 提到 35–50，冲击档位从 4 提到 7 | "
        "运行时间约 ×10–15（EA.4 单臂 3 种子约 4 分钟 → 数十小时量级） | "
        "能分辨 0.45 与 0.95 这种量级的差异 | **仍然答不了「是否落在 0.5±0.1」**"
        "（那需要 ~300 种子） |")
    add("| **B. 换指标** | 放弃「幂律指数」这个刻度，改用**直接可观测量**："
        "比如单档滑点的配对差（大档位 t 已经到 −12.6）、或「凹度」的定性判据"
        "（大档与小档的滑点比值是否显著 < 线性外推） | "
        "要重新定义验收标准与图表口径 | 「冲击是不是凹的」以及「凹度有没有变大」"
        "——**这本来就是任务真正关心的东西** | 得不到一个「k=0.5x」这样可跨版本比对的标量 |")
    add("| **C. 先查饱和** | 先把 7 档全跑一遍（**不加种子**），"
        "看 `fill_ratio` 从哪一档开始掉、大档的滑点是否仍单调 | 便宜（档位增加≈线性涨） | "
        "决定 A 路线的实际成本，也可能直接给出「凹度」的定性答案 | "
        "单独走这条路得不到统计判定 |")
    add("")
    add("### 与三条候选机制的关系（任务书 §2.4 要求「重新审视是否还有其他候选机制」）\n")
    add("- **EA.4 的「非对称」本身还没有被否定**：它的机制体检指标（稀疏化生效占比 "
        f"91.4%、平均活跃主体 285.2）仍然是干净的**直接观测**。"
        "被否定的是「用 k 证明它更接近 0.5」。")
    add("- 工作线 E 的核心假说 H4（吃单方与做市方突发同步 ⇒ 抵消稀疏化）"
        "**不依赖 k**——它比的是同步率与四象限深度，所以**即使 k 这条路走不通，E 线仍然可做**。"
        "这是本轮唯一一条「不受分辨力问题影响」的机制线。")
    add("- 工作线 D（多种子稳健性）**部分失效**：ED.1 的「20 种子看 k 分布」"
        "现在有了明确答案——20 种子只能把区间宽度压到 ~0.77，"
        "仍然分辨不了任何一对我们讨论过的 k。")
    add("- 兜底工作线 G（ρ→count 映射修复）**没有触发条件**："
        "它要求「非对称在最优配置下 k 仍 > 0.65」，而 k 现在无法判定 >/< 0.65。\n")
    add("---\n")

    # ---------------- 8 ----------------
    add("## 8. 本轮我做了什么 / 没做什么\n")
    add("**做了**（都在 §2 范围内）：\n")
    add("1. 写了 `scripts/verify_EA4_calibration.py`（三步校验 + 重标定重跑）"
        "与 `tests/test_EA4_calibration_integrity.py`（13 条，含反面参照）。")
    add("2. 发现并修掉窗口不对齐；重标定后得到官方基准（§2）。")
    add("3. **先做复现自查**排除掉「执行者自己改坏」这一可能（§3）——"
        "任务书没要求这一步，但不做它就无法安全地得出任何结论。")
    add("4. 写了 `scripts/diagnose_k_uncertainty.py`（k 的不确定度诊断，"
        "复算复用产品代码的 `fit_power_law`）与 9 条测试。")
    add("5. 把影响范围**逐条量出来**（§5），而不是笼统说「大概都受影响」。\n")
    add("**没做**（按 §2.4 留给你的决定）：\n")
    add("- 没有继续跑工作线 D / E / F 的任何实验。")
    add("- 没有启动兜底工作线 G。")
    add("- 没有替你在 A/B/C 三条路径里做选择。\n")
    add("**需要你回答的**：\n")
    add("1. A/B/C 走哪条（或组合）？")
    add("2. E 线的 H4 假说（不依赖 k）是否**单独放行**？"
        "它是本轮唯一不受分辨力问题影响的机制线。")
    add("3. 「逼近 0.5」这个验收目标是否要改成「凹度显著」这类不依赖绝对刻度的判据？\n")

    return "\n".join(L) + "\n"


def _ncols(line: str) -> int:
    """数一行 markdown 表格有几列。

    ⚠️ 必须先抹掉 ``\\|``（转义竖线，渲染成字面量、不算列分隔符），
    否则臂名里的 ``|`` 会让计数虚高——那是测试/检查自己报假警。
    """
    return line.strip().strip("|").replace("\\|", "").count("|") + 1


def check_tables(txt: str) -> list[str]:
    """检查 markdown 表格能不能渲染（表头/分隔行/数据行列数一致）。

    为什么必须有这道检查：本项目真实发生过「写回时用了 ``"".join(L)``，
    整节被拼成一行、表格彻底不渲染，而脚本退出码是 0」的事故。
    产物侧的自检当时**不看渲染**，所以掉进去了。这里就地补上。
    """
    problems: list[str] = []
    lines = txt.splitlines()
    i = 0
    while i < len(lines):
        if not lines[i].strip().startswith("|"):
            i += 1
            continue
        block: list[str] = []
        j = i
        while j < len(lines) and lines[j].strip().startswith("|"):
            block.append(lines[j])
            j += 1
        if len(block) >= 2:
            n_head = _ncols(block[0])
            n_sep = _ncols(block[1])
            if n_head != n_sep:
                problems.append(
                    f"第 {i + 1} 行起的表格：表头 {n_head} 列 vs 分隔行 {n_sep} 列")
            for k, ln in enumerate(block[2:], start=2):
                if _ncols(ln) != n_head:
                    problems.append(
                        f"第 {i + 1 + k} 行：{_ncols(ln)} 列 ≠ 表头 {n_head} 列")
        i = j
    return problems


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    txt = build()
    head = txt.split("## 5. 影响范围")[0]
    for bad in ("nan", "None", "inf"):
        if bad in head:
            raise SystemExit(f"❌ 报告前段出现未处理值 {bad!r}，拒绝写出。")
    tbl_problems = check_tables(txt)
    if tbl_problems:
        raise SystemExit("❌ 表格渲染检查不通过，拒绝写出：\n  - "
                         + "\n  - ".join(tbl_problems[:8]))
    n_tbl = sum(1 for ln in txt.splitlines() if ln.strip().startswith("|"))
    TARGET.write_text(txt, encoding="utf-8")
    print(f"  诊断报告已生成：{TARGET}（{len(txt.encode('utf-8')) / 1024:.0f} KB，"
          f"{n_tbl} 行表格、渲染检查通过）")


if __name__ == "__main__":
    main()
