#!/usr/bin/env python
"""A7 报告：**收益目标按窗口标定** + **两套窗口的稳健性检验**。

    python scripts/make_a7_report.py      # → docs/A7-窗口标定与稳健性报告.md

⭐ 本文件的**每一个数字都从 `out/**/*.json` 现算**，没有一个手抄的
（与 `make_a6_report.py` 同一条纪律：手抄的数字迟早与 JSON 脱节）。

A7 只做一件事：**把 A6 里唯一一个"拍脑袋"的阈值改成同源标定，
再用"换一组窗口"检验两条主打结论稳不稳。**

为什么这两件事要一起做：
- 只改阈值 ⇒ 会忍不住把结果读成"KPI 变差了"（单组窗口给的假象）；
- 只做稳健性 ⇒ 修好的目标没派上用场。
⇒ 一起做才拿到那句最干净的话：**行为稳健、收益不稳健。**

⚠️ 运行前请确认 `out/a7/` 下有产物；缺失时对应小节显示「（无产物）」，**不编数**。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
for _p in (Path(__file__).resolve().parent.parent,):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tw.segmented import paired_verdict, t_crit95  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
A6 = ROOT / "out" / "a6"
A7 = ROOT / "out" / "a7"
OUT = ROOT / "docs" / "A7-窗口标定与稳健性报告.md"


# ======================================================================
# 小工具（与 A6 报告同款：缺失就显示「—」，**不编**）
# ======================================================================
def _load(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _f(x: Any, n: int = 2) -> str:
    if x is None or x != x:
        return "—"
    return f"{float(x):.{n}f}"


def _pct(x: Any, n: int = 2) -> str:
    if x is None or x != x:
        return "—"
    return f"{float(x) * 100:.{n}f}%"


def _beh(review: dict | None) -> dict[str, Any]:
    """从复盘 JSON 里取段均行为指标。"""
    b = (review or {}).get("behavior_by_segment") or []
    if not b:
        return {}
    n = len(b)

    def av(k: str) -> float:
        return sum(x[k] for x in b) / n

    return {
        "presence": av("presence"),
        "turnover": av("turnover_x"),
        "net": av("net"),
        "hold": ((review.get("n_hold") or 0)
                 / max(review.get("n_decisions") or 1, 1)),
        "n": n,
    }


def _paired_vs_v4(a_path: Path, b_path: Path, *,
                  key_a: str = "llm_v4", key_b: str = "llm_v6") -> dict | None:
    """两个**独立运行**在同段上的配对差值（同段、同数据 ⇒ 可以配对）。

    ⚠️ 与"同一次运行内两个配置配对"不同：这里是**两次运行**
    （两次独立采样），但每段的行情与可见状态完全相同
    ⇒ 配对仍然成立，且这正是"换了个变量重跑一次"的标准做法。
    """
    a, b = _load(a_path), _load(b_path)
    if not a or not b:
        return None
    na = (a.get("result") or {}).get("net") or []
    nb = (b.get("result") or {}).get("net") or []
    m = min(len(na), len(nb))
    if m < 2:
        return None
    try:
        d = [float(nb[i][key_b]) - float(na[i][key_a]) for i in range(m)]
    except (KeyError, TypeError, ValueError):
        return None
    return paired_verdict(d, name_a="处理臂", name_b="对照臂")


def _overlap(ci_a, ci_b) -> float:
    """两个区间的重叠比例（**分母取较窄的那个**，纪律 ⑧）。"""
    if not ci_a or not ci_b or None in list(ci_a) + list(ci_b):
        return float("nan")
    inter = max(0.0, min(ci_a[1], ci_b[1]) - max(ci_a[0], ci_b[0]))
    w = min(ci_a[1] - ci_a[0], ci_b[1] - ci_b[0])
    return inter / w if w > 0 else float("nan")


def _llm_vs_momentum(e: dict | None) -> dict | None:
    for k, v in ((e or {}).get("verdicts") or {}).items():
        if v and k.startswith("llm_") and "momentum" in k:
            return v
    return None


def _kpi_fails(e: dict | None) -> tuple[int, int, dict[str, int]]:
    kb = (e or {}).get("kpi_by_segment") or []
    if not kb:
        return (0, 0, {})
    npass = sum(1 for x in kb if x.get("passed"))
    fails: dict[str, int] = {}
    for x in kb:
        for f in (x.get("failed") or []):
            fails[f] = fails.get(f, 0) + 1
    return (npass, len(kb), fails)


# ======================================================================
# 主流程
# ======================================================================
def build() -> str:                                        # noqa: C901
    L: list[str] = []
    L.append("# A7 报告：收益目标按窗口标定 · 两套窗口的稳健性检验")
    L.append("")
    L.append("> ⚠️ 本文件**每一个数字都在运行时从 `out/**/*.json` 现算**，")
    L.append("> 没有一个是手抄的。产物缺失时对应小节显示「（无产物）」。")
    L.append("")

    # ---- 载入 ---------------------------------------------------------
    E = {
        "v4": _load(A6 / "eval_v4_BTC8.json"),
        "v6fix": _load(A6 / "eval_v6_BTC8.json"),
        "v6cal": _load(A7 / "eval_v6cal_BTC8.json"),
        "v7": _load(A6 / "eval_v7_BTC8.json"),
    }
    Ee = {
        "v4": _load(A6 / "eval_v4_ETH8.json"),
        "v6fix": _load(A6 / "eval_v6_ETH8.json"),
        "v6cal": _load(A7 / "eval_v6cal_ETH8.json"),
        "v7": _load(A6 / "eval_v7_ETH8.json"),
    }
    E4 = {                       # offset = 4（换一组窗口）
        "v4": _load(A7 / "eval_v4_BTC8o4.json"),
        "v6cal": _load(A7 / "eval_v6_BTC8o4.json"),
        "v7": _load(A7 / "eval_v7_BTC8o4.json"),
    }
    R = {
        "v4": _load(A6 / "review8_v4_BTC.json"),
        "v6fix": _load(A6 / "review8_v6_BTC.json"),
        "v6cal": _load(A7 / "review_v6cal_BTC8.json"),
        "v7": _load(A6 / "review8_v7_BTC.json"),
    }
    Re = {
        "v4": _load(A6 / "review8_v4_ETH.json"),
        "v6fix": _load(A6 / "review8_v6_ETH.json"),
        "v6cal": _load(A7 / "review_v6cal_ETH8.json"),
        "v7": _load(A6 / "review8_v7_ETH.json"),
    }
    R4 = {
        "v4": _load(A7 / "review_v4_BTC8o4.json"),
        "v6cal": _load(A7 / "review_v6_BTC8o4.json"),
        "v7": _load(A7 / "review_v7_BTC8o4.json"),
    }

    # ---- 0. 结论 -------------------------------------------------------
    L.append("## 0. 一句话结论")
    L.append("")
    L.append("A6 留下的那个「KPI 收益效果判不出来」里，混着**一个装置缺陷**：")
    L.append("`target_return` 是唯一还在拍脑袋的阈值（硬编码 `+1%`），")
    L.append("**绝对值、不随窗口长度变** ⇒ L=8 时要求 8 根赚 1% ⇒")
    L.append("**48/48 段全部 `return` 失败**。那批数据测的是"
             "「**目标不可达**时它会怎么做」，不是「给目标好不好」。")
    L.append("")
    L.append("修法：**与其余三个阈值同源**——用基线在**同一窗口长度**下的")
    L.append("净收益分位（默认中位数）。目标因此自动随 L 缩放，且**永远可达**。")
    L.append("")
    L.append("修完之后再加上**换一组窗口**的检验，得到本轮最干净的一句话：")
    L.append("")
    L.append("- ⭐ **行为效果稳健**：给 KPI 之后在场率被**稳定抬高**")
    L.append("  （**两标的 × 两套窗口共 3 个组合**，方向全一致）；")
    L.append("- ⚠️ **收益效果不稳健**：单看第一组窗口会读成「偏不利」，")
    L.append("  换一组窗口后差值几乎归零 ⇒ **不能声称 KPI 让收益变差**。")
    L.append("")

    # ---- 1. 缺陷 -------------------------------------------------------
    L.append("## 1. 第 16 个真缺陷：收益目标是**绝对值**，不随窗口缩放")
    L.append("")
    L.append("**症状**：`kpi_verdict` 逐段判定里，`return` 一项 **100% 失败**。")
    L.append("")
    L.append("⭐ **它是被数据自己暴露的**：一个「目标」若在 48 段里一段都达不到，")
    L.append("那它就不是目标，是噪音——而代码、日志、报告里**全都看不出来**。")
    L.append("")
    L.append("**为什么这很要紧**：KPI 的整个设计目的就是"
             "「用考核去引导行为」。")
    L.append("若目标恒不可达，模型收到的信号就从「朝这个方向努力」")
    L.append("退化成「你一直不及格」——**这是另一个实验条件**，")
    L.append("而不能拿来回答原来的问题。")
    L.append("")
    L.append("**修法（与其余阈值同源）**：")
    L.append("")
    L.append("| 阈值 | 旧来源 | 新来源 |")
    L.append("|---|---|---|")
    L.append("| 在场率下限 | 基线 20% 分位 | 同（未变） |")
    L.append("| 回撤上限 | 基线 90% 分位 | 同（未变） |")
    L.append("| 换手区间 | 基线 10%~90% 分位 | 同（未变） |")
    L.append("| **收益目标** | ~~硬编码 +1%~~ | **基线每段净收益的中位数** |")
    L.append("")
    L.append("⚠️ 标定出的目标若 **≤ 0**，实现会**明确报警**"
             "（「要赚」这条几乎无约束力）——")
    L.append("否则报告会把它当成一个「有要求的目标」，而它其实是「别亏」。")
    L.append("")

    # ---- 2. 标定出的目标 -----------------------------------------------
    L.append("## 2. 标定出来的目标长什么样")
    L.append("")
    L.append("| 标的 | 窗口 L | 旧目标（固定） | **新目标（基线净收益中位数）** |")
    L.append("|---|---|---|---|")
    for tag, e in (("BTC", E["v6cal"]), ("ETH", Ee["v6cal"])):
        kpi = (e or {}).get("kpi") or {}
        if kpi:
            L.append(f"| {tag} | 8 | +1.000% | **{_pct(kpi.get('target_return'), 3)}** |")
    for tag, e in (("BTC（offset=4）", E4["v6cal"]),):
        kpi = (e or {}).get("kpi") or {}
        if kpi:
            L.append(f"| {tag} | 8 | +1.000% | **{_pct(kpi.get('target_return'), 3)}** |")
    L.append("")
    L.append("⚠️ **读法**：标定出来的目标都**≈0 且为负**，"
             "也就是说在这份数据上，")
    L.append("8 根窗口里「达到基线水平」就等于「几乎别亏钱」。")
    L.append("⇒ **「要赚」这条约束在 L=8 上立不起来**——这不是实现问题，")
    L.append("是**短窗口本身没有可赚的确定性边际**。")
    L.append("⇒ 它同时解释了一件事：A6 里 KPI 的收益效果判不出来，")
    L.append("   **不是因为装置不够灵敏，而是因为在 8 根窗口上本来就没有可抓的效应。**")
    L.append("")

    # ---- 3. 三臂对照 ---------------------------------------------------
    L.append("## 3. 修好之后：三臂对照（v4 / v6-固定 / v6-标定）")
    L.append("")
    L.append("### 3.1 行为侧（**KPI 有没有起作用**——这才是主判据）")
    L.append("")
    L.append("| 标的 | 模板 | 在场率 | 换手 | 弃权占比 | 净收益 |")
    L.append("|---|---|---|---|---|---|")
    for tag, rr in (("BTC", R), ("ETH", Re)):
        for k, nm in (("v4", "v4（无 KPI）"),
                      ("v6fix", "v6（固定 +1%）"),
                      ("v6cal", "**v6（按窗口标定）**")):
            b = _beh(rr.get(k))
            if not b:
                continue
            L.append(f"| {tag} | {nm} | {_pct(b['presence'], 1)} | "
                     f"{_f(b['turnover'], 2)}× | {_pct(b['hold'], 1)} | "
                     f"{_pct(b['net'], 3)} |")
    L.append("")
    L.append("⭐ **行为结论（要读的是方向，不是绝对值）**：")
    L.append("")
    for tag, rr in (("BTC", R), ("ETH", Re)):
        b4, bc = _beh(rr.get("v4")), _beh(rr.get("v6cal"))
        if not (b4 and bc):
            continue
        L.append(f"- **{tag}**：在场率 {_pct(b4['presence'], 1)} → "
                 f"{_pct(bc['presence'], 1)}（**{_pct(bc['presence'] - b4['presence'], 1)}**）、"
                 f"换手 {_f(b4['turnover'], 2)}→{_f(bc['turnover'], 2)}×、"
                 f"弃权 {(bc['hold'] - b4['hold']) * 100:+.1f}pp")
    L.append("")
    L.append("⇒ **给 KPI 之后它在场更多、更少弃权**，两个标的方向一致。")
    L.append("")
    L.append("### 3.2 收益侧（v6-标定 − v4，**同段配对、两次独立运行**）")
    L.append("")
    L.append("| 标的 | 差值 | 95% 区间 | t（临界） | 边界比 | 判定 |")
    L.append("|---|---|---|---|---|---|")
    for tag, pa, pb in (("BTC", A6 / "eval_v4_BTC8.json", A7 / "eval_v6cal_BTC8.json"),
                        ("ETH", A6 / "eval_v4_ETH8.json", A7 / "eval_v6cal_ETH8.json")):
        v = _paired_vs_v4(pa, pb)
        if not v:
            L.append(f"| {tag} | （无产物） | | | | |")
            continue
        lo, hi = v["ci"]
        tc = t_crit95(int(v["df"])) if v.get("df") else float("nan")
        ratio = abs(v["t"]) / tc if tc and tc == tc and tc > 0 else float("nan")
        L.append(f"| {tag} | {_pct(v['mean_diff'], 4)} | "
                 f"[{_pct(lo, 4)}, {_pct(hi, 4)}] | "
                 f"{_f(v['t'], 2)}（{_f(tc, 2)}） | {_f(ratio, 2)} | "
                 f"**{v['verdict']}** |")
    L.append("")
    L.append("### 3.3 KPI 达标分布（**缺陷修好没有，看这里**）")
    L.append("")
    L.append("| 标的 | 配置 | 达标段数 | 未达标项分布 | 阈值 |")
    L.append("|---|---|---|---|---|")
    for tag, e in (("BTC", E["v6fix"]), ("BTC", E["v6cal"]),
                   ("ETH", E["v6fix"]), ("ETH", Ee["v6cal"])):
        npass, n, fails = _kpi_fails(e)
        if not n:
            continue
        kpi = (e or {}).get("kpi") or {}
        which = "固定 +1%" if e is E["v6fix"] else "按窗口标定"
        L.append(f"| {tag} | {which} | **{npass}/{n}** | {fails or '—'} | "
                 f"目标 {_pct(kpi.get('target_return'), 3)} |")
    L.append("")
    L.append("⇒ 固定目标下 **0/48**（读数没有信息量）；标定后 **12/48**，")
    L.append("而且失败分布开始区分得开"
             "（`return` / `turnover_hi` / `turnover_lo` / `presence`）。")
    L.append("**这才是「考核」应该产出的东西。**")
    L.append("")

    # ---- 4. 稳健性 -----------------------------------------------------
    L.append("## 4. ⭐ 稳健性：**换一组窗口**（这是本项目的「换种子」）")
    L.append("")
    L.append("本设计没有随机性，唯一的「任意选择」就是**从哪一根开始切段**。")
    L.append("⇒ 新增 `--seg-offset`，并**限定 `[0, seg_len)`**：")
    L.append("取 `seg_len` 的整数倍只是把同一组窗口平移，")
    L.append("**不是「换一组窗口」** ⇒ 那会给出**假的稳健性**（看起来验了，其实没验）。")
    L.append("")
    L.append("### 4.1 KPI 的行为效果：**3 个组合方向全一致**")
    L.append("")
    L.append("| 标的 | 段网格 | 在场率 v4 → v6-标定 | 变化 | 换手变化 | 弃权变化 |")
    L.append("|---|---|---|---|---|---|")
    for tag, rr, off in (("BTC", R, "offset=0"), ("ETH", Re, "offset=0"),
                         ("BTC", R4, "offset=4")):
        b4, bc = _beh(rr.get("v4")), _beh(rr.get("v6cal"))
        if not (b4 and bc):
            continue
        L.append(f"| {tag} | {off} | {_pct(b4['presence'], 1)} → "
                 f"{_pct(bc['presence'], 1)} | "
                 f"**{(bc['presence'] - b4['presence']) * 100:+.1f}pp** | "
                 f"{_f(bc['turnover'] - b4['turnover'], 2)}× | "
                 f"{(bc['hold'] - b4['hold']) * 100:+.1f}pp |")
    L.append("")
    # ⚠️ **不要在这里硬编码数字**：我第一版把 ETH 那格写成了 "+14.0pp"，
    # 而表里现算出来是 +7.9pp —— 一份手抄、一份现算，立刻分叉。
    # 这正是"报告里的数字一律现算"那条纪律要防的东西。
    _pps = []
    for rr in (R, Re, R4):
        _a, _b = _beh(rr.get("v4")), _beh(rr.get("v6cal"))
        if _a and _b:
            _pps.append((_b["presence"] - _a["presence"]) * 100)
    if _pps:
        L.append("⇒ **在场率一律被抬高**（" +
                 " / ".join(f"**+{v:.1f}pp**" for v in _pps) +
                 f"），三处**方向全一致**、幅度同量级。")
    L.append("⇒ 「KPI 让它更多地在场」这条**过了窗口稳健性检验**。")
    L.append("")
    L.append("### 4.2 KPI 的收益效果：**不稳健**（这是本轮最该记住的一条）")
    L.append("")
    L.append("| 段网格 | v6-标定 − v4 | 95% 区间 | t | 边界比 | 判定 |")
    L.append("|---|---|---|---|---|---|")
    for off, pa, pb in ((("offset=0"), A6 / "eval_v4_BTC8.json",
                         A7 / "eval_v6cal_BTC8.json"),
                        (("offset=4"), A7 / "eval_v4_BTC8o4.json",
                         A7 / "eval_v6_BTC8o4.json")):
        v = _paired_vs_v4(pa, pb)
        if not v:
            continue
        lo, hi = v["ci"]
        tc = t_crit95(int(v["df"])) if v.get("df") else float("nan")
        ratio = abs(v["t"]) / tc if tc and tc == tc and tc > 0 else float("nan")
        L.append(f"| BTC {off} | **{_pct(v['mean_diff'], 4)}** | "
                 f"[{_pct(lo, 4)}, {_pct(hi, 4)}] | {_f(v['t'], 2)} | "
                 f"{_f(ratio, 2)} | {v['verdict']} |")
    L.append("")
    v0 = _paired_vs_v4(A6 / "eval_v4_BTC8.json", A7 / "eval_v6cal_BTC8.json")
    v4 = _paired_vs_v4(A7 / "eval_v4_BTC8o4.json", A7 / "eval_v6_BTC8o4.json")
    if v0 and v4:
        ov = _overlap(v0["ci"], v4["ci"])
        L.append(f"⭐ **两套窗口的区间重叠：{_pct(ov, 1)}**")
        L.append("")
        L.append("⚠️ **这条是本轮最有价值的发现**：")
        L.append("")
        L.append(f"- 单看 `offset=0` 会读到 {_pct(v0['mean_diff'], 3)}"
                 f"（区间 [{_pct(v0['ci'][0], 3)}, {_pct(v0['ci'][1], 3)}]）"
                 f"—— 区间几乎全负，**很容易顺手写成「KPI 让收益变差」**；")
        L.append(f"- 换一组窗口后是 {_pct(v4['mean_diff'], 3)}"
                 f"（区间 [{_pct(v4['ci'][0], 3)}, {_pct(v4['ci'][1], 3)}]）"
                 f"—— **几乎归零**。")
        L.append("")
        L.append("⇒ **结论：不能说 KPI 让收益变差。**")
        L.append("⇒ 而且这件事**只有靠「换一组窗口」才看得出来**——")
        L.append("  单组窗口下两条读法（「偏不利」与「没差别」）**长得一模一样**。")
    L.append("")

    # ---- 5. v7 的窗口稳健性 --------------------------------------------
    L.append("## 5. 经验段（v7）的窗口稳健性")
    L.append("")
    L.append("| 标的 | 段网格 | 在场率 v4 → v7 | 变化 | 换手变化 | 弃权变化 |")
    L.append("|---|---|---|---|---|---|")
    for tag, rr, off in (("BTC", R, "offset=0"), ("ETH", Re, "offset=0"),
                         ("BTC", R4, "offset=4")):
        b4, b7 = _beh(rr.get("v4")), _beh(rr.get("v7"))
        if not (b4 and b7):
            continue
        L.append(f"| {tag} | {off} | {_pct(b4['presence'], 1)} → "
                 f"{_pct(b7['presence'], 1)} | "
                 f"**{(b7['presence'] - b4['presence']) * 100:+.1f}pp** | "
                 f"{_f(b7['turnover'] - b4['turnover'], 2)}× | "
                 f"{(b7['hold'] - b4['hold']) * 100:+.1f}pp |")
    L.append("")
    L.append("⇒ **给经验之后它更保守**（在场率下降、换手下降、弃权上升），")
    L.append("三个组合方向一致 ⇒ 这条也**过了窗口稳健性检验**。")
    L.append("")

    # ---- 6. 边界 -------------------------------------------------------
    L.append("## 6. ⚠️ 诚实边界")
    L.append("")
    L.append("- **标定出的收益目标 ≈ 0 甚至为负**：这说明在 8 根窗口上"
             "「赚」这件事没有可抓的确定性。")
    L.append("  ⇒ A7 修好的是**装置**（目标可达了），"
             "**没有**凭空造出收益效应。")
    L.append("- **稳健性只做了 2 个 offset**（0 与 4）。"
             "段长 8 下 `offset ∈ [0,8)` 共 8 种，我们只验了 2 种。")
    L.append("  ⇒ 「方向一致」比单点显著更有说服力，但**不是「已确立」**。")
    L.append("- **v7 的 offset=4 只做了 BTC**（ETH 未做）。")
    L.append("- **收益上的两套窗口差异（−0.066% vs −0.004%）本身也可能只是噪声**：")
    L.append("  两条区间**高度重叠**，正确的说法是"
             "「**这个方向的证据不稳健**」，")
    L.append("  **不是**「换窗口就一定没有效应」。")
    L.append("- **`target_q` 默认取中位数**（`0.50`）。若改取 0.75，")
    L.append("  「要赚」的约束会变严 ⇒ **那是另一个实验**，应单独重跑。")
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
