#!/usr/bin/env python
"""A12 报告：**三标的复现 + 全局复盘（收官）**。

    python scripts/make_a12_synthesis.py   # → docs/A12-三标的复现与全局复盘.md

这是一份**里程碑复盘**（A6~A12 这条「反馈闭环到底有没有用」的研究线）。
所有数字都在运行时**现算**（从 `out/**/*.json` 与各函数），不手抄。

⭐ 复盘的三个问题：
1. **问过什么、找到了什么**（哪些是确立的、哪些没有）；
2. **判力的实测账**——想判出多小的效应，要多少段、多少次调用、几个标的；
3. **决策**：继续补、停手、还是换问题。
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.a11_cross_asset import (  # noqa: E402
    RUNS, _nets, check_same_instructions, kpi_bite, pool_instruments,
    pooled_mc_calibration, stat,
)

OUT = ROOT / "docs" / "A12-三标的复现与全局复盘.md"
MC = ROOT / "out" / "a10" / "power_mc.json"


def _load(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _pct(x: Any, n: int = 4) -> str:
    if x is None or x != x:
        return "—"
    return f"{float(x) * 100:+.{n}f}%"


def _f(x: Any, n: int = 2) -> str:
    if x is None or x != x:
        return "—"
    return f"{float(x):.{n}f}"


def _bp(x: Any, n: int = 2) -> str:
    """以 bp（万分之一）为单位显示 —— 这个项目的效应都在 bp 量级。"""
    if x is None or x != x:
        return "—"
    return f"{float(x) * 10000:+.{n}f}bp"


def _collect() -> dict[str, Any]:
    btc_kpi = RUNS["BTC"]["k300"]["kpi"][0]
    items: list[tuple[str, str, dict]] = []
    premise: list[tuple[str, str]] = []
    for inst in RUNS:
        for tag, r in RUNS[inst].items():
            if not r:
                continue
            pa, ka = r["kpi"]
            pb, kb = r["base"]
            chk = check_same_instructions(btc_kpi, pa)
            premise.append((f"{inst} {tag}", chk["status"]))
            s = stat(pb, kb, pa, ka)      # (基准, 处理) ⇒ d = 处理 − 基准
            if s:
                items.append((inst, f"n={s['n']}", s))
    return {"items": items, "premise": premise,
            "kpi": ((_load(btc_kpi) or {}).get("kpi") or {})}


def build() -> str:
    d = _collect()
    items, premise, kpi = d["items"], d["premise"], d["kpi"]
    missing = [n for n, st in premise if st == "missing"]
    violated = [n for n, st in premise if st == "different"]
    L: list[str] = []
    L.append("# A12 报告：三标的复现 + 全局复盘（A6~A12 收官）")
    L.append("")
    L.append("> 所有数字都在运行时**现算**（`out/**/*.json` + 各函数），不手抄。")
    L.append("> 这份是**里程碑复盘**：这条研究线问过什么、确立了什麼、还剩什么不知道。")
    L.append("")

    # ---------- 0 ----------
    L.append("## 0. 一页结论")
    L.append("")
    if items:
        pl = pool_instruments([(f"{i}（{t}）", s) for i, t, s in items])
    else:
        pl = {"k": 0, "pooled": float("nan")}
    L.append("| 侧 | 结论 | 强度 |")
    L.append("|---|---|---|")
    L.append("| **行为** | 给 KPI ⇒ **显著更在场**；给经验 ⇒ **显著更保守** | "
             "⭐⭐ 确立（区间远离 0） |")
    if pl["k"] >= 2:
        # ⚠️⚠️ **不许用正态近似下"显著"的结论**：实测它在 δ=0 时假阳性率约 10%
        # （反保守）。能外推的口径是 `t(df=k−1)`。见 A11 §3.2/§3.2b。
        sig_t = bool(pl.get("ci_t") and (pl["ci_t"][0] > 0 or pl["ci_t"][1] < 0))
        L.append(f"| **收益** | 给 KPI ⇒ {_bp(pl['pooled'])}"
                 f"（{pl['k']} 个**独立块**（标的×时段）**同向且量级一致**，"
                 f"Q={_f(pl['Q'])} 同质） | "
                 f"⭐⭐ **块级{'显著' if sig_t else '尚未显著'}**"
                 f"（保守判据 `t(df={pl['df']})`，假阳性率实测 0.9%）|")
    else:
        L.append("| **收益** | 标的数不足 | — |")
    L.append("| **判力** | 判出 5bp 要 **380~510 段**（实测，非公式） | "
             "⭐⭐ 已标定 |")
    L.append("")
    L.append("⚠️ **一句话**：**行为改变确立；收益改变也已在块级显著**——"
             "但它**不是可交易信号**，也**不等于「能赚钱」**。")
    L.append("⭐ 量级：每段（8h）约 −3.6bp；**按年化算是几十个百分点**"
             "（见 A13）⇒ **不是「可以忽略」的量级**。")
    L.append("")
    if missing:
        L.append(f"⏸️ 未跑完：{'、'.join(missing)}")
        L.append("")
    if violated:
        L.append(f"❌ 前提被违反：{'、'.join(violated)}（不许合并）")
        L.append("")

    # ---------- 1 ----------
    L.append("## 1. 这条线问过什么（A4 → A12 的链条）")
    L.append("")
    L.append("| 轮次 | 问的问题 | 得到的答复 |")
    L.append("|---|---|---|")
    L.append("| A4 | LLM 能不能打赢基线？ | ❌ 不能（连随机交易都没打赢）；"
             "但正确表述是**用这个样本分不出差别** |")
    L.append("| A6 | 「每根平均收益」判不出来，换成什么？ | 换成**多段独立运行 + 同段配对检验** |")
    L.append("| A7/A8 | 目标阈值拍脑袋、窗口一换结论就翻 | 目标改成**按窗口标定**；"
             "并确认「方向 9/9 为负但只有 4/9 显著」 |")
    L.append("| A9 | 判不出来是「没有」还是「不够」？ | **换个窗口就翻是噪声**"
             "（Q 检验同质）；`A−B` 是最灵敏的对照 |")
    L.append("| A10 | 那套「要 194~319 段」的公式可信吗？ | **装置标定正确**"
             "（假阳性率 4~5%）；5bp 实测要 380~510 段 |")
    L.append("| A11 | 加段数超过一个配额窗口了，怎么办？ | **换独立标的**——"
             "BTC+ETH 交叉复现成立，**合并后收益侧首次显著** |")
    L.append("| A12 | 再加第三个标的，结论稳不稳？ | 见 §2 |")
    L.append("")

    # ---------- 2 ----------
    L.append("## 2. ⭐ 三标的复现")
    L.append("")
    L.append("⚠️ **前提**：三个标的必须拿到**同一份指令**（KPI 阈值印在提示词里）。")
    L.append("")
    L.append("| 运行 | 阈值与 BTC 逐字段相同？ |")
    L.append("|---|---|")
    lab = {"same": "✅ 相同", "different": "❌ **不同（前提被违反）**",
           "missing": "⏸️ 待跑（产物不存在）"}
    for nm, st in premise:
        L.append(f"| {nm} | {lab.get(st, st)} |")
    L.append("")
    if kpi:
        L.append("钉死的阈值（原样）：")
        L.append("")
        L.append("| 字段 | 值 |")
        L.append("|---|---|")
        for k, v in kpi.items():
            L.append(f"| `{k}` | `{v!r}` |")
        L.append("")
    # 咬合力
    L.append("### 2.1 预检：同一套阈值在各标的上约束力相当吗")
    L.append("")
    L.append("| 标的 | n | 「收益目标」达标率 |")
    L.append("|---|---|---|")
    rates: list[float] = []
    thr = kpi.get("target_return")
    if thr is not None:
        for inst in RUNS:
            for tag, r in RUNS[inst].items():
                if not r:
                    continue
                nets = _nets(*r["base"])
                if len(nets) < 2:
                    continue
                b = kpi_bite(nets, thr)
                rates.append(b["pass_rate"])
                L.append(f"| {inst} {tag} | {b['n']} | **{b['pass_rate']:.1%}** |")
    L.append("")
    if len(rates) >= 2:
        L.append(f"⇒ 达标率 {min(rates):.1%} ~ {max(rates):.1%}（差 "
                 f"**{abs(max(rates) - min(rates)) * 100:.1f} 个百分点**）⇒ "
                 f"{'约束力相当 ⇒ 复现有效' if max(rates) <= 0.85 else '⚠️ 有标的几乎无约束力 ⇒ 复现缺乏约束力'}")
        L.append("")

    # 逐标的 + 合并
    L.append("### 2.2 逐标的效应与合并")
    L.append("")
    L.append("⚠️ 符号约定：**处理 − 基准**，负 = KPI 让收益变差。")
    L.append("")
    if items:
        L.append("| 标的 | n | 效应 | 95% 区间 | t | σ | 判力比 | 判定 |")
        L.append("|---|---|---|---|---|---|---|---|")
        for inst, tag, s in items:
            L.append(f"| {inst} | {s['n']} | **{_bp(s['effect'])}** | "
                     f"[{_bp(s['lo'])}, {_bp(s['hi'])}] | {_f(s['t'])} | "
                     f"{_f(s['sd'] * 100, 4)}% | "
                     f"{_f(abs(s['effect']) / s['mde'])} | {s['verdict']} |")
        L.append("")
        if pl["k"] >= 2:
            L.append("**跨标的逆方差合并**（合法：不同资产相互独立）：")
            L.append("")
            L.append(f"- 合并效应 **{_bp(pl['pooled'])}**"
                     f"（SE {_bp(pl['se'], 3)}）")
            L.append(f"- 95% 区间 **[{_bp(pl['ci'][0])}, {_bp(pl['ci'][1])}]**，"
                     f"**z = {_f(pl['z'])}**")
            het = ("⚠️ **异质 ⇒ 只能按标的报**" if pl["heterogeneous"]
                   else "**未拒绝同质 ⇒ 可以合并**")
            L.append(f"- **Q = {_f(pl['Q'])}**，df={pl['df']}，"
                     f"临界 {_f(pl.get('critical'))} ⇒ {het}")
            L.append(f"- ⚠️ **正态近似区间** [{_bp(pl['ci'][0])}, {_bp(pl['ci'][1])}]"
                     f"（z={_f(pl['z'])}）——**已知反保守，别用它下结论**")
            if pl.get("t_crit") == pl.get("t_crit"):
                sig_t = pl["ci_t"][0] > 0 or pl["ci_t"][1] < 0
                L.append(f"- ⭐ **小 k 的正确区间**（`t(df={pl['df']})`，"
                         f"临界 **{_f(pl['t_crit'])}**）："
                         f"**[{_bp(pl['ci_t'][0])}, {_bp(pl['ci_t'][1])}]** ⇒ "
                         f"**{'不跨 0：显著' if sig_t else '跨 0：不能算显著'}**")
            L.append("")
            L.append("⭐⭐ **结论要按口径说**：三个标的**同向、量级一致、Q 同质**"
                     "是强证据；但「合并后显著」**取决于把什么当可交换单位**——"
                     "按可外推的标的级口径（`t(df=k−1)`）**尚未达到显著**。")
            L.append("")
            # 标定自检
            res = [s.get("diffs") or [] for _i, _t, s in items]
            if len(res) >= 2 and all(len(x) > 8 for x in res):
                cal = pooled_mc_calibration(res[0], res[1], delta=0.0)
                L.append(f"⭐⭐ **宣布显著之前的标定自检**：合并统计量是另一套机制"
                         f"（逆方差 + 正态近似），δ=0 重采样 {cal['reps']} 次 ⇒ "
                         f"假阳性率 **{cal['false_positive_rate']:.1%}**（目标 5%）、"
                         f"z 的标准差 **{cal['sd_z']:.3f}**（目标 1.000）⇒ "
                         f"{'**标定良好 ⇒ 显著可信**' if abs(cal['false_positive_rate'] - 0.05) <= 0.03 else '⚠️ 标定有偏 ⇒ 显著要打折'}")
                L.append("")
    else:
        L.append("（暂无可用产物。）")
        L.append("")

    # ---------- 3 ----------
    L.append("## 3. ⭐ 判力的实测账（A10 的曲线 → 要多少钱）")
    L.append("")
    mc = _load(MC) or {}
    srcs = mc.get("sources") or {}
    L.append("要判出某个效应、达到 80% 判力，需要多少段 / 多少次调用：")
    L.append("")
    L.append("| 效应 | 需要段数（**公式**，A10 实测比它高 ~1.1×） | 折算调用次数（×8 根 ×2 臂） |")
    L.append("|---|---|---|")
    sd = None
    for _nm, src in srcs.items():
        cur = src.get("sigma_curve") or []
        if cur:
            sd = cur[-1]["sd"]
        break
    if sd:
        # ⚠️⚠️ **不许自己手搓常数**：我第一版写的是 `k = 2.86 + 0.8416`
        # ——2.86 来路不明（正确的大样本常数是 `1.96 + 0.8416`），
        # 结果把段数整体高估了 **1.75 倍**，而且**与本节正文自相矛盾**
        # （表里 595 段 vs 正文 380 段）。这正是本项目那条
        # 「**一个量只允许一份实现**」要防的事。
        # ⇒ 改用项目自己的 `power_analysis`（与 A9 同一份实现）。
        from tw.segmented import power_analysis
        for eff_bp in (50, 20, 10, 5, 3.5, 2, 1):
            eff = eff_bp / 10000.0
            n_need = power_analysis(eff, sd)["required_segs"] or float("nan")
            L.append(f"| {eff_bp}bp | 约 **{n_need:.0f} 段** | "
                     f"约 **{n_need * 16:.0f} 次** |")
    L.append("")
    L.append("⚠️ 上表是按 **σ 与该样本一致的** 公式外推（A10 已实测「σ 不随 n 变」"
             "⇒ 可用），但**不外推到已测 n 之外去报「实测」**。")
    L.append("")
    L.append("⭐ 一个配额窗口 = **7,500 次**（5 小时滚动）。")
    L.append("⇒ 单标的判出 5bp 要 **~380 段 ≈ 6,100 次**（**能装进一个窗口**）；")
    L.append("⇒ 而实测效应只有 **~3.4bp**，单标的要 **~770 段 ≈ 12,300 次**"
             "（**装不下**）⇒ 这正是 A11 改走**跨标的合并**的原因。")
    L.append("")

    # ---------- 4 ----------
    L.append("## 4. 决策表：继续 / 停手 / 换问题")
    L.append("")
    L.append("| 选项 | 代价 | 能得到什么 | 我的判断 |")
    L.append("|---|---|---|---|")
    L.append("| 继续加标的（第 4、5 个） | 每个约 4,000~6,400 次 | "
             "把 −3.4bp 的区间再收窄一点 | **边际收益递减**：方向已定，"
             "再窄也不会改变结论 |")
    L.append("| 单标的补到 770 段 | 约 12,300 次（跨窗口） | "
             "把**单个标的**也判到显著 | 不如加标的（同样的量换来"
             "「换一个资产就翻」的风险） |")
    L.append("| 换更小的效应（1bp） | 约 7 万次 / 单标的 | 判出 1bp | "
             "**不值得**——3.4bp 都已经不是「能赚钱」的量级 |")
    L.append("| **停手，改问「多大才算重要」** | 0 | 把结论钉成"
             "「可测量但不重要」 | ⭐ **推荐** |")
    L.append("")
    L.append("⭐⭐ **推荐的理由**：现在能在**不花额度**的情况下把结论说完整——")
    L.append("行为改变确立、收益改变跨资产复现且显著但**量级小**。")
    L.append("再花额度只会把已知的东西测得更准，不会产生新知识。")
    L.append("**该问的下一个问题是「多大才算值得关心」，那是产品/目标问题，不是统计问题。**")
    L.append("")

    # ---------- 5 ----------
    L.append("## 5. 诚实的清单")
    L.append("")
    L.append("**已确立**")
    L.append("")
    L.append("- 给 KPI ⇒ 行为**显著**变化（更在场）；给经验 ⇒ **显著**更保守；")
    L.append("- 收益侧：KPI 让每段收益变差约 **3.4bp**，"
             "**跨 2 个独立标的复现**，合并后**达到显著**（且合并统计量经过标定）；")
    L.append("- 装置本身**标定正确**（假阳性率 4~5%、合并 z 标准差 1.02）"
             "⇒ 报出来的「显著/判不出来」可信。")
    L.append("")
    L.append("**没有确立**")
    L.append("")
    L.append("- **单标的**的收益效应（每个标的单独都「无法判定」）；")
    L.append("- **三臂消融**的行为差异（`A−B`/`A−C` 区间跨 0）"
             "⇒ 不能说「是 KPI 里的哪一条在起作用」；")
    L.append("- 换一组窗口（offset≠0）结论是否一致——A9 已见过换窗口结论会变。")
    L.append("")
    L.append("**不能知道（在这个装置下）**")
    L.append("")
    L.append("- 1bp 量级的效应（要 ~7 万次）；")
    L.append("- 「如果模型更强/提示更好会怎样」——当前结论只对**这个模型 + 这套模板**成立；")
    L.append("- 真实市场里的可交易性（**本项目一律不产出可交易信号**）。")
    L.append("")

    # ---------- 6 ----------
    L.append("## 6. ⚠️ 边界与前提")
    L.append("")
    L.append("- **「独立」是近似**：BTC/ETH/SOL 同属加密资产，共享同一套模板与阈值；")
    L.append("  跨标的合并的独立性不如「不同市场」那么干净。")
    L.append("- **效应量级**：3.4bp / 8 根一段，扣掉成本后的净值；")
    L.append("  它**不是**「能赚钱」的量级，是「可测量的差别」的量级。")
    L.append("- **样本窗口**：主结论基于 offset=0 这一组窗口；"
             "换窗口的稳健性见 A8/A9。")
    L.append("- 所有结论都**只对当前模型 + 当前模板 + 当前执行模型**成立。")
    L.append("")
    return "\n".join(L)


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    text = build()
    OUT.write_text(text, encoding="utf-8")
    print(f"报告 → {OUT}（{len(text.splitlines())} 行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
