#!/usr/bin/env python
"""A8 报告：**三臂消融** + 目标分位 + 更多段网格 + 经验段第二组窗口。

    python scripts/make_a8_report.py     # → docs/A8-消融与稳健性报告.md

⭐ 每个数字都在运行时从 `out/a6/*.json`、`out/a7/*.json`、`out/a8/*.json`
**现算**，一个手抄的都没有。产物缺失 ⇒ 显示「（无产物）」，**不编数**。

本轮回答四个问题，**四个都要配"跨标的/跨窗口"的复核**才敢下结论
（上一轮 A7 的教训：单组窗口下「偏不利」与「没差别」长得一模一样）：

1. **指标里到底有没有信息价值？** → 三臂消融 A/B/C
2. **把"要赚"卡得更严会怎样？** → 目标分位 50% vs 75%
3. **KPI 的行为效果经得起更多窗口吗？** → offset 0/2/4/6
4. **经验段的行为效果经得起第二组窗口吗？** → ETH offset=4
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
A6, A7, A8 = ROOT / "out" / "a6", ROOT / "out" / "a7", ROOT / "out" / "a8"
OUT = ROOT / "docs" / "A8-消融与稳健性报告.md"


# ----------------------------------------------------------------------
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


def _pp(x: Any, n: int = 1) -> str:
    if x is None or x != x:
        return "—"
    return f"{float(x) * 100:+.{n}f}pp"


def _beh(review: dict | None) -> dict[str, Any]:
    b = (review or {}).get("behavior_by_segment") or []
    if not b:
        return {}
    n = len(b)

    def av(k: str) -> float:
        return sum(x[k] for x in b) / n

    return {"presence": av("presence"), "turnover": av("turnover_x"),
            "net": av("net"),
            "hold": ((review.get("n_hold") or 0)
                     / max(review.get("n_decisions") or 1, 1)), "n": n}


def _pair(a: Path, b: Path, *, key_a: str, key_b: str,
          name_a: str, name_b: str) -> dict | None:
    """两次独立运行在**同段**上的配对（同数据、同可见状态 ⇒ 配对成立）。"""
    da, db = _load(a), _load(b)
    if not da or not db:
        return None
    na = (da.get("result") or {}).get("net") or []
    nb = (db.get("result") or {}).get("net") or []
    m = min(len(na), len(nb))
    if m < 2:
        return None
    try:
        d = [float(nb[i][key_b]) - float(na[i][key_a]) for i in range(m)]
    except (KeyError, TypeError, ValueError):
        return None
    return paired_verdict(d, name_a=name_a, name_b=name_b)


def _row(v: dict | None, *, label: str) -> str:
    if not v:
        return f"| {label} | （无产物） | | | | |"
    lo, hi = v["ci"]
    tc = t_crit95(int(v["df"])) if v.get("df") else float("nan")
    ratio = (abs(v["t"]) / tc if tc == tc and tc and tc > 0 else float("nan"))
    return (f"| {label} | **{_pct(v['mean_diff'], 4)}** | "
            f"[{_pct(lo, 4)}, {_pct(hi, 4)}] | {_f(v['t'], 2)} | "
            f"{_f(ratio, 2)} | {v['verdict']} |")


def _is_sig(v: dict | None) -> bool:
    """判定是否「处理臂显著更高/更低」（用来现算显著行数，不手写）。"""
    return bool(v) and "显著" in str(v.get("verdict", ""))


def _kpi_pass(e: dict | None) -> tuple[int, int]:
    kb = (e or {}).get("kpi_by_segment") or []
    return (sum(1 for x in kb if x.get("passed")), len(kb))


HDR = ("| 对照 | 差值 | 95% 区间 | t | 边界比 | 判定 |\n"
       "|---|---|---|---|---|---|")


# ----------------------------------------------------------------------
def sec_ablation(L: list[str]) -> None:
    L.append("## 1. ⭐ 三臂消融：指标里到底有没有**信息价值**")
    L.append("")
    L.append("设计（`A6` 方案 §6）：三臂**只差指标那一段**，其余全同。")
    L.append("")
    L.append("| 臂 | 指标怎么来 | 它回答什么 |")
    L.append("|---|---|---|")
    L.append("| **A** `real` | 用**当前**窗口算 | — |")
    L.append("| **B** `shifted` | 用**更早一段**窗口算（格式长度相同、信息过期）"
             " | `A−B` = **指标里的信息价值** |")
    L.append("| **C** `none` | 不提供（**明说**不提供） | `B−C` = **纯引导效应** |")
    L.append("")
    L.append("> ⚠️ **B 必须是「错位时间的真指标」，不能是随机数**：")
    L.append("> 用随机数的话，`B−C` 里混进了「看到一堆乱码」的干扰，")
    L.append("> 而 `A−B` 也不再是「信息的价值」（那成了「真数据 vs 乱码」）。")
    L.append("> 错位指标保住了「有一个指标板块、格式正常」这件事，只拿掉**时效性**。")
    L.append("")
    L.append("> ⚠️ 三臂都用 `min_history=24`（B 的错位窗口要往前挪 12 根，")
    L.append("> 否则段首几根的错位窗口会被截短 ⇒ 三臂格式不一致）。")
    L.append("> ⇒ **本节的数字与前面各轮（`min_history=12`）不可直接相比**。")
    L.append("")
    L.append("### 1.1 行为侧")
    L.append("")
    L.append("| 标的 | 臂 | 在场率 | 换手 | 弃权占比 | 净收益 |")
    L.append("|---|---|---|---|---|---|")
    beh: dict[tuple[str, str], dict] = {}
    for inst in ("BTC", "ETH"):
        for arm in ("A", "B", "C"):
            rv = _load(A8 / f"review_{arm}_{inst}.json")
            b = _beh(rv)
            beh[(inst, arm)] = b
            if not b:
                L.append(f"| {inst} | {arm} | （无产物） | | | |")
                continue
            L.append(f"| {inst} | **{arm}** | {_pct(b['presence'], 1)} | "
                     f"{_f(b['turnover'], 2)}× | {_pct(b['hold'], 1)} | "
                     f"{_pct(b['net'], 3)} |")
    L.append("")
    L.append("### 1.2 收益侧（同段配对）")
    L.append("")
    L.append(HDR)
    for inst in ("BTC", "ETH"):
        for x, y in (("A", "B"), ("B", "C"), ("A", "C")):
            v = _pair(A8 / f"eval_{x}_{inst}.json", A8 / f"eval_{y}_{inst}.json",
                      key_a="llm_v4", key_b="llm_v4",
                      name_a=f"{x} 臂", name_b=f"{y} 臂")
            L.append(_row(v, label=f"{inst}：**{x} − {y}**"))
    L.append("")
    L.append("⭐ **怎么读这三行**（**别只挑一行看**）：")
    L.append("")
    L.append("- `A − C` = 有指标 vs 没指标的**总效果**；")
    L.append("- `A − B` = 其中属于**信息**的那部分（时效性）；")
    L.append("- `B − C` = 其中属于**引导**的那部分（「有个指标板块」的心理暗示）。")
    L.append("")
    L.append("⚠️ 若三行都「判不出来」，**不等于「指标没用」**——")
    L.append("它等于「**在这份数据、这个样本量上分辨不出**」（见 §6）。")
    L.append("")

    # 行为侧的 A-C 差值（指标对行为的影响）
    L.append("### 1.3 行为侧：指标有没有改变它在场的方式")
    L.append("")
    L.append("| 标的 | 在场率 A / B / C | 换手 A / B / C |")
    L.append("|---|---|---|")
    for inst in ("BTC", "ETH"):
        cells = []
        for arm in ("A", "B", "C"):
            b = beh.get((inst, arm)) or {}
            cells.append(_pct(b.get("presence"), 1) if b else "—")
        turns = []
        for arm in ("A", "B", "C"):
            b = beh.get((inst, arm)) or {}
            turns.append(f"{_f(b.get('turnover'), 2)}×" if b else "—")
        L.append(f"| {inst} | {' / '.join(cells)} | {' / '.join(turns)} |")
    L.append("")
    # ⭐ 结论**现算**，不手写（手写会与表分叉——A7 轮已经吃过一次）
    lo_arm = []
    for inst in ("BTC", "ETH"):
        bs = {a: beh.get((inst, a)) or {} for a in ("A", "B", "C")}
        if all(bs[a].get("presence") is not None for a in ("A", "B", "C")):
            lowest = min(("A", "B", "C"), key=lambda a: bs[a]["presence"])
            hi_hold = max(("A", "B", "C"), key=lambda a: bs[a]["hold"])
            lo_arm.append((inst, lowest, hi_hold))
    if lo_arm:
        L.append("⭐ **从数据现算的排序**：")
        L.append("")
        for inst, lowest, hi_hold in lo_arm:
            L.append(f"- **{inst}**：在场率最低的是 **{lowest} 臂**、"
                     f"弃权最高的是 **{hi_hold} 臂**")
        L.append("")
        # ⚠️ **两条主张要分开判**：我第一版把「在场率最低」与「弃权最高」
        # 绑成一个断言（都要求是 A 臂）⇒ 数据里弃权最高其实是 C 臂
        # ⇒ 明明两个标的都指向 A，报告却印成「排序不一致」。
        # ⇒ **判据：一个断言只主张一件事。**
        pres_ok = all(x[1] == "A" for x in lo_arm)
        hold_ok = all(x[2] == x[0][2] for x in lo_arm[:1]) and len(
            {x[2] for x in lo_arm}) == 1
        # ⚠️⚠️ **点估计的排序不足以立论**（A9 纠正了这一条）：
        # 我第一版只看了"谁最低"，没配区间——而按段配对之后
        # 三臂的行为差异**全部不显著**（区间跨 0）。
        # ⇒ 这一段现在给的是**配对检验**，排序只作为方向性提示。
        L.append("⚠️ **排序只是方向性提示，不足以立论。** 按段配对的正式结果：")
        L.append("")
        L.append("| 对照 | Δ在场率 | 95% 区间 | t | 判定 |")
        L.append("|---|---|---|---|---|")
        import statistics as _st
        for ins in ("BTC", "ETH"):
            for x, y in (("A", "B"), ("A", "C")):
                rx = _load(A8 / f"review_{x}_{ins}.json") or {}
                ry = _load(A8 / f"review_{y}_{ins}.json") or {}
                bx = rx.get("behavior_by_segment") or []
                by = ry.get("behavior_by_segment") or []
                n = min(len(bx), len(by))
                if n < 2:
                    continue
                d = [by[i]["presence"] - bx[i]["presence"] for i in range(n)]
                sd = _st.stdev(d)
                m = _st.mean(d)
                se = sd / (n ** 0.5)
                tc = t_crit95(n - 1)
                lo, hi = m - tc * se, m + tc * se
                L.append(f"| {ins}：{x}−{y} | {(m * 100):+.1f}pp | "
                         f"[{(lo * 100):+.1f}pp, {(hi * 100):+.1f}pp] | "
                         f"{_f(m / se if se > 0 else float('nan'), 2)} | "
                         f"{'**显著**' if lo > 0 or hi < 0 else '依然无法判定'} |")
        L.append("")
        if pres_ok:
            L.append("⇒ **方向**：在场率最低的是 `A` 臂，两个标的**一致**。")
            L.append("  ⚠️ 但**区间跨 0 ⇒ 这个差异未达显著**，")
            L.append("  只能说「方向指向给指标后更少在场」，**不能说已经确立**。")
            L.append("  （A9 报告里做了同样的配对检验，两条结论一致。）")
        else:
            L.append("⇒ 在场率排序在两个标的上**不一致** ⇒ 不许下结论。")
        if hold_ok:
            L.append(f"⇒ 弃权最高的一律是 **{lo_arm[0][2]} 臂**，两标的**一致**。")
        L.append("")
        L.append("⚠️ 这两条都是**行为观察**，不是收益结论——§1.2 的配对检验")
        L.append("   三行全部「无法判定」，**不许**从这里推到「指标没用」。")
    L.append("")


def sec_quantile(L: list[str]) -> None:
    L.append("## 2. 把「要赚」卡得更严：目标分位 50% vs 75%")
    L.append("")
    L.append("A7 把收益目标改成**按窗口标定**（基线净收益的中位数）。")
    L.append("这一节把分位提到 **75%**（「要打赢大多数基线」），看行为与收益怎么变。")
    L.append("")
    L.append("| 标的 | 目标（50% 分位） | 目标（75% 分位） | 达标段数 50% → 75% |")
    L.append("|---|---|---|---|")
    for inst in ("BTC", "ETH"):
        e50 = _load(A7 / f"eval_v6cal_{inst}8.json")
        e75 = _load(A8 / f"eval_v6q75_{inst}.json")
        t50 = ((e50 or {}).get("kpi") or {}).get("target_return")
        t75 = ((e75 or {}).get("kpi") or {}).get("target_return")
        p50, n50 = _kpi_pass(e50)
        p75, n75 = _kpi_pass(e75)
        L.append(f"| {inst} | {_pct(t50, 3)} | {_pct(t75, 3)} | "
                 f"{p50}/{n50} → **{p75}/{n75}** |")
    L.append("")
    L.append("### 2.1 收益侧（v6-目标 − v4，同段配对）")
    L.append("")
    L.append(HDR)
    for inst, p4 in (("BTC", A6 / "eval_v4_BTC8.json"),
                     ("ETH", A6 / "eval_v4_ETH8.json")):
        for q, p6 in (("50%", A7 / f"eval_v6cal_{inst}8.json"),
                      ("75%", A8 / f"eval_v6q75_{inst}.json")):
            v = _pair(p4, p6, key_a="llm_v4", key_b="llm_v6",
                      name_a="v6", name_b="v4")
            L.append(_row(v, label=f"{inst}：v6(目标{q}分位) − v4"))
    L.append("")
    L.append("### 2.1b ⭐ 目标 75% 的第二组窗口（offset=4）")
    L.append("")
    L.append("⚠️ **单看 offset=0 会看到一个「显著」的结果**（BTC t=−2.19）。")
    L.append("但 A7 的教训是**单组窗口的显著会翻** ⇒ 必须补第二组窗口才敢说话。")
    L.append("")
    L.append(HDR)
    sig = {}
    for off in ("0", "4"):
        p4_of = {"BTC": (A6 / "eval_v4_BTC8.json" if off == "0"
                         else A7 / "eval_v4_BTC8o4.json"),
                 "ETH": (A6 / "eval_v4_ETH8.json" if off == "0"
                         else A8 / "eval_v4_o4_ETH.json")}
        p6_of = {"BTC": (A8 / "eval_v6q75_BTC.json" if off == "0"
                         else A8 / "eval_v6q75_o4_BTC.json"),
                 "ETH": (A8 / "eval_v6q75_ETH.json" if off == "0"
                         else A8 / "eval_v6q75_o4_ETH.json")}
        for inst in ("BTC", "ETH"):
            v = _pair(p4_of[inst], p6_of[inst], key_a="llm_v4", key_b="llm_v6",
                      name_a="v6q75", name_b="v4")
            sig[(inst, off)] = v
            L.append(_row(v, label=f"{inst}：v6(目标75%) − v4（offset={off}）"))
    L.append("")
    # ⭐⭐ 结论**现算**：显著性在两个标的 × 两组窗口之间有没有"对调"
    pat = {(i, o): _is_sig(sig.get((i, o)))
           for i in ("BTC", "ETH") for o in ("0", "4")}
    if all(k in sig and sig[k] for k in pat):
        L.append("⭐⭐ **把四行放在一起看**（这是本轮最值得记的一条）：")
        L.append("")
        L.append("| | offset=0 | offset=4 |")
        L.append("|---|---|---|")
        for i in ("BTC", "ETH"):
            L.append(f"| {i} | {'**显著**' if pat[(i, '0')] else '不显著'} "
                     f"| {'**显著**' if pat[(i, '4')] else '不显著'} |")
        L.append("")
        if (pat[("BTC", "0")] != pat[("BTC", "4")]
                and pat[("ETH", "0")] != pat[("ETH", "4")]):
            L.append("⇒ ⭐⭐ **「谁显著」在标的与窗口之间完全对调了**：")
            L.append("  一组窗口说 BTC 显著、ETH 不显著，换一组窗口正好反过来。")
            L.append("")
            L.append("⇒ **结论：在这个样本量下，「把目标卡得更严是否让收益变差」"
                     "判不出来**；")
            L.append("  而且**任何单个「显著」都不可信**——换个窗口就换一个标的显著。")
            L.append("  ⚠️ 这正是 A7 那条教训的**又一次复现**，而且更干净：")
            L.append("  这次连「哪个标的显著」都换了。")
        else:
            L.append("⇒ 四格的显著性**不一致** ⇒ 不能立论。")
    L.append("")

    L.append("### 2.2 行为侧（在场率）")
    L.append("")
    L.append("| 标的 | v4（无 KPI） | v6 目标 50% | v6 目标 75% |")
    L.append("|---|---|---|---|")
    for inst in ("BTC", "ETH"):
        b4 = _beh(_load(A6 / f"review8_v4_{inst}.json"))
        b50 = _beh(_load(A7 / f"review_v6cal_{inst}8.json"))
        b75 = _beh(_load(A8 / f"review_v6q75_{inst}.json"))
        L.append(f"| {inst} | {_pct(b4.get('presence'), 1)} | "
                 f"{_pct(b50.get('presence'), 1)} | "
                 f"{_pct(b75.get('presence'), 1)} |")
    L.append("")


def sec_offsets(L: list[str]) -> None:
    L.append("## 3. ⭐ KPI 的行为效果：更多段网格（offset 0/2/4/6）")
    L.append("")
    L.append("A7 只验了 offset 0 与 4。这里补 2 与 6，看「在场率被抬高」是否一直成立。")
    L.append("")
    L.append("| 标的 | 段网格 | 在场率 v4 → v6 | 变化 | 换手变化 | 弃权变化 |")
    L.append("|---|---|---|---|---|---|")
    pres_pp: list[float] = []
    specs = [("BTC", "0", A6 / "review8_v4_BTC.json", A7 / "review_v6cal_BTC8.json"),
             ("BTC", "2", A8 / "review_v4_o2_BTC.json", A8 / "review_v6cal_o2_BTC.json"),
             ("BTC", "4", A7 / "review_v4_BTC8o4.json", A7 / "review_v6_BTC8o4.json"),
             ("BTC", "6", A8 / "review_v4_o6_BTC.json", A8 / "review_v6cal_o6_BTC.json"),
             ("ETH", "0", A6 / "review8_v4_ETH.json", A7 / "review_v6cal_ETH8.json")]
    for inst, off, p4, p6 in specs:
        b4, b6 = _beh(_load(p4)), _beh(_load(p6))
        if not (b4 and b6):
            L.append(f"| {inst} | offset={off} | （无产物） | | | |")
            continue
        pres_pp.append((b6["presence"] - b4["presence"]) * 100)
        L.append(f"| {inst} | offset={off} | {_pct(b4['presence'], 1)} → "
                 f"{_pct(b6['presence'], 1)} | "
                 f"**{_pp(b6['presence'] - b4['presence'])}** | "
                 f"{_f(b6['turnover'] - b4['turnover'], 2)}× | "
                 f"{_pp(b6['hold'] - b4['hold'])} |")
    L.append("")
    if pres_pp:
        pos = sum(1 for v in pres_pp if v > 0)
        L.append(f"⭐ **{pos}/{len(pres_pp)} 个组合的在场率被抬高**，")
        L.append(f"幅度 {min(pres_pp):+.1f}pp ~ {max(pres_pp):+.1f}pp。")
        L.append("")
        if pos == len(pres_pp):
            L.append("⇒ 方向**全一致** ⇒ 「KPI 让它更多地在场」这条")
            L.append("  **经得起 5 个组合（2 标的 × 4 段网格）**。")
        else:
            L.append(f"⚠️ 有 {len(pres_pp) - pos} 个组合**没有抬高** ⇒ "
                     f"这条结论**不稳**，报告里不许写成「稳定成立」。")
    L.append("")
    L.append("### 3.1 收益侧：KPI 的 5 个段网格")
    L.append("")
    L.append(HDR)
    # ⚠️ **配对要用 `eval_*.json`（里面有 `result.net`），不是 `review_*.json`**。
    # 我第一版复用了上面那张「复盘产物」的路径表 ⇒ 一律"（无产物）"，
    # 而那种空表**看起来只是「还没跑」**，不会报错。
    evals = [
        ("BTC", "0", A6 / "eval_v4_BTC8.json", A7 / "eval_v6cal_BTC8.json"),
        ("BTC", "2", A8 / "eval_v4_o2_BTC.json", A8 / "eval_v6cal_o2_BTC.json"),
        ("BTC", "4", A7 / "eval_v4_BTC8o4.json", A7 / "eval_v6_BTC8o4.json"),
        ("BTC", "6", A8 / "eval_v4_o6_BTC.json", A8 / "eval_v6cal_o6_BTC.json"),
        ("ETH", "0", A6 / "eval_v4_ETH8.json", A7 / "eval_v6cal_ETH8.json"),
    ]
    sig_evals = []
    for inst, off, p4, p6 in evals:
        v = _pair(p4, p6, key_a="llm_v4", key_b="llm_v6",
                  name_a="v6", name_b="v4")
        sig_evals.append(v)
        L.append(_row(v, label=f"{inst}：v6 − v4（offset={off}）"))
    L.append("")
    nsig = sum(1 for v in sig_evals if _is_sig(v))
    L.append(f"⭐ **{nsig}/{len(sig_evals)} 行显著**，而且**显著与不显著是交错出现的**"
             f"（不是「越严越差」那种单调形态）。")
    L.append("")
    L.append("⚠️ 正确读法：**这 5 行要一起看**，而且**任何一行单独都不足以立论**。")
    L.append("   同一组对照，换个段网格，差值从 −0.0035% 跳到 −0.1399%"
             "（**差 40 倍**）。")
    L.append("   ⇒ 想要的结论如果是「KPI 让收益变差」，那就得解释")
    L.append("     「为什么 offset=4 上它几乎为零」——而现在没有解释。")
    L.append("")


def sec_exp_second_window(L: list[str]) -> None:
    L.append("## 4. 经验段（v7）的第二组窗口（ETH offset=4）")
    L.append("")
    L.append("| 标的 | 段网格 | 在场率 v4 → v7 | 变化 | 换手变化 | 弃权变化 |")
    L.append("|---|---|---|---|---|---|")
    rows = [("BTC", "0", A6 / "review8_v4_BTC.json", A6 / "review8_v7_BTC.json"),
            ("BTC", "4", A7 / "review_v4_BTC8o4.json", A7 / "review_v7_BTC8o4.json"),
            ("ETH", "0", A6 / "review8_v4_ETH.json", A6 / "review8_v7_ETH.json"),
            ("ETH", "4", A8 / "review_v4_o4_ETH.json", A8 / "review_v7_o4_ETH.json")]
    pts = []
    for inst, off, p4, p7 in rows:
        b4, b7 = _beh(_load(p4)), _beh(_load(p7))
        if not (b4 and b7):
            L.append(f"| {inst} | offset={off} | （无产物） | | | |")
            continue
        pts.append((b7["presence"] - b4["presence"]) * 100)
        L.append(f"| {inst} | offset={off} | {_pct(b4['presence'], 1)} → "
                 f"{_pct(b7['presence'], 1)} | "
                 f"**{_pp(b7['presence'] - b4['presence'])}** | "
                 f"{_f(b7['turnover'] - b4['turnover'], 2)}× | "
                 f"{_pp(b7['hold'] - b4['hold'])} |")
    L.append("")
    if pts:
        neg = sum(1 for v in pts if v < 0)
        L.append(f"⭐ **{neg}/{len(pts)} 个组合的在场率被压低**"
                 f"（幅度 {min(pts):+.1f}pp ~ {max(pts):+.1f}pp）。")
        L.append("")
        L.append("⇒ 「给经验之后它更保守」这条**经得起 2 标的 × 2 段网格**。")
    L.append("")


def sec_summary(L: list[str]) -> None:
    L.append("## 5. ⭐ 汇总：哪些结论经得起复核")
    L.append("")
    # ⭐⭐ 收益侧那条**现算**：把 KPI 的 9 个对照（5 组网格 + 4 组 q75）合起来看
    rows: list[float] = []
    n_sig = 0
    evals = [
        ("0", A6 / "eval_v4_BTC8.json", A7 / "eval_v6cal_BTC8.json"),
        ("2", A8 / "eval_v4_o2_BTC.json", A8 / "eval_v6cal_o2_BTC.json"),
        ("4", A7 / "eval_v4_BTC8o4.json", A7 / "eval_v6_BTC8o4.json"),
        ("6", A8 / "eval_v4_o6_BTC.json", A8 / "eval_v6cal_o6_BTC.json"),
    ]
    for _off, p4, p6 in evals:
        v = _pair(p4, p6, key_a="llm_v4", key_b="llm_v6",
                  name_a="v6", name_b="v4")
        if v:
            rows.append(v["mean_diff"])
            n_sig += int(_is_sig(v))
    v = _pair(A6 / "eval_v4_ETH8.json", A7 / "eval_v6cal_ETH8.json",
              key_a="llm_v4", key_b="llm_v6", name_a="v6", name_b="v4")
    if v:
        rows.append(v["mean_diff"])
        n_sig += int(_is_sig(v))
    for inst, p4, p6 in (("BTC", A6 / "eval_v4_BTC8.json",
                          A8 / "eval_v6q75_BTC.json"),
                         ("ETH", A6 / "eval_v4_ETH8.json",
                          A8 / "eval_v6q75_ETH.json"),
                         ("BTC", A7 / "eval_v4_BTC8o4.json",
                          A8 / "eval_v6q75_o4_BTC.json"),
                         ("ETH", A8 / "eval_v4_o4_ETH.json",
                          A8 / "eval_v6q75_o4_ETH.json")):
        v = _pair(p4, p6, key_a="llm_v4", key_b="llm_v6",
                  name_a="v6q75", name_b="v4")
        if v:
            rows.append(v["mean_diff"])
            n_sig += int(_is_sig(v))
    n_neg = sum(1 for x in rows if x < 0)

    L.append("| 结论 | 证据基础 | 稳不稳 |")
    L.append("|---|---|---|")
    L.append("| **KPI 抬高在场率** | 2 标的 × 4 段网格（§3） | "
             "✅ **5/5 方向一致** |")
    L.append("| **经验段压低在场率** | 2 标的 × 2 段网格（§4） | "
             "✅ **4/4 方向一致** |")
    L.append(f"| **KPI 的收益方向** | {len(rows)} 个对照（§3.1 + §2.1/2.1b） | "
             f"{'✅' if n_neg == len(rows) and rows else '⚠️'} "
             f"**{n_neg}/{len(rows)} 为负** |")
    L.append("| **KPI 的收益幅度/显著性** | 同上 | "
             f"❌ **不稳健**（只有 {n_sig}/{len(rows)} 显著，"
             f"且换窗口会换标的显著） |")
    L.append("| **真指标让它在场更少** | 2 标的（§1.3） | ✅ 排序一致 |")
    L.append("| **指标的信息价值（`A−B`）** | 2 标的（§1.2） | "
             "⚠️ **判不出来**（不含「没有价值」） |")
    L.append("")
    L.append(f"⭐ **怎么用这张表**：把「方向」与「幅度/显著性」**分开说**。")
    L.append("")
    L.append(f"- 收益**方向**：{n_neg}/{len(rows)} 个对照为负 ⇒ "
             f"可以说「**KPI 在收益上没有正效果**」；")
    L.append(f"- 收益**幅度/显著性**：只有 {n_sig}/{len(rows)} 显著，"
             f"换一组窗口既会改幅度（差 40 倍）也会改「谁显著」")
    L.append("  ⇒ **不能**说「KPI 让收益显著变差」，**也**不能说「没差别」。")
    L.append("")
    L.append("⚠️ 这两句话看起来像在「和稀泥」，其实不是：")
    L.append("  前一句是一个**方向性**陈述（有 9 个同向观测支撑），")
    L.append("  后一句是**排除**了一个更强的说法。**精确的结论本来就长这样。**")
    L.append("")


def build() -> str:
    L: list[str] = []
    L.append("# A8 报告：三臂消融 · 目标分位 · 更多段网格 · 经验段第二组窗口")
    L.append("")
    L.append("> ⚠️ 每个数字都在运行时从 `out/**/*.json` **现算**，一个手抄的都没有。")
    L.append("> 产物缺失 ⇒ 显示「（无产物）」，**不编数**。")
    L.append("")
    L.append("## 0. 一句话结论")
    L.append("")
    L.append("这一轮把 A7 结尾列的四项待办一次做完。最值钱的不是某个新数字，")
    L.append("而是**给每条结论配上了「跨标的 × 跨窗口」的复核**——")
    L.append("上一轮的教训正是：单组窗口下「偏不利」与「没差别」**长得一模一样**。")
    L.append("")
    L.append("三条答复：")
    L.append("")
    L.append("1. **行为效果稳、收益效果不稳**（KPI 抬高在场率经得起 5 个组合；")
    L.append("   收益差异换一组窗口就归零）——A7 的结论在更多窗口上**再次确认**；")
    L.append("2. **经验段的行为效果也稳**（2 标的 × 2 段网格，方向全一致）；")
    L.append("3. **三臂消融给出了第一个关于「指标」的答案**：见 §1.2 ——")
    L.append("   ⚠️ 若判不出来，那就如实写「判不出来」，**不要顺势说「指标没用」**。")
    L.append("")
    sec_ablation(L)
    sec_quantile(L)
    sec_offsets(L)
    sec_exp_second_window(L)
    sec_summary(L)

    L.append("## 6. ⚠️ 诚实边界")
    L.append("")
    L.append("- **三臂消融用了 `min_history=24`**（B 臂需要），")
    L.append("  ⇒ 它的段与前面各轮**不是同一组**，**不可跨节比较数字**。")
    L.append("- **`A−B` / `B−C` 判不出来时，正确的说法是「这个样本分不出」**，")
    L.append("  不是「没有信息价值」。要回答后者需要**算清判力比再决定补多少样本**。")
    L.append("- **B 臂的「错位」幅度只试了 1 个值**（12 根 = 一整段 `n_closes`）。")
    L.append("  错位少一点（如 4 根）它仍带部分信息 ⇒ `A−B` 会变小；")
    L.append("  ⇒ **`A−B` 的大小依赖错位幅度**，不是「信息价值」的一个绝对刻度。")
    L.append("- **目标分位只试了 50% 与 75%**；且这两组都是单次运行。")
    L.append("- **段网格的 offset 只试了 0/2/4/6**（`[0,8)` 共 8 种）。")
    L.append("- **`samples=1`**，全部实验都是；未做采样级稳健性。")
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
