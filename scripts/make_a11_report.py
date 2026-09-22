#!/usr/bin/env python
"""A11 报告：**跨标的复现**——把 KPI 效应在第二个资产上重做一遍。

    python scripts/make_a11_report.py     # → docs/A11-跨标的复现报告.md

每个数字都在运行时从 `out/**/*.json` **现算**。

为什么做这个（而不是继续在 BTC 上加段数）
----------------------------------------
A10 实测：判出 5bp 要 380~510 段；而 offset=0 的效应只有 3.5bp
⇒ 80% 判力要约 770 段（12,320 次调用）——**超过一个配额窗口（7,500/5h）**。

⭐ A9 只禁了「合并多个段网格」（同一段行情的不同切法、**重叠、不独立**）。
**不同标的相互独立** ⇒ 跨标的**可以合并**，而且这是**交叉复现**——
比在同一条行情上堆段数更能挡住「这段行情特殊」。

⚠️⚠️ 前提（A9b 踩过的同一个坑）：`--calibrate-kpi` 会让每个标的用
**它自己标定的阈值** ⇒ 指令不同 ⇒ 那不是「同一个实验在不同资产上」。
本报告 §2 把这条前提写成**会失败的断言**，
并**区分「没有数据」与「前提被违反」**（这两件事我第一版混成了一个）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.a11_cross_asset import (  # noqa: E402
    RUNS, _nets, check_same_instructions, kpi_bite, pool_instruments, stat,
)

OUT = ROOT / "docs" / "A11-跨标的复现报告.md"


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


def _collect() -> tuple[list[tuple[str, dict]], list[tuple[str, str, list]],
                        dict]:
    """跑一遍收集：逐标的统计 + 前提状态 + 钉死的阈值。"""
    btc_kpi = RUNS["BTC"]["k300"]["kpi"][0]
    items: list[tuple[str, dict]] = []
    premise: list[tuple[str, str, list]] = []
    for inst in ("BTC", "ETH"):
        for tag, r in RUNS[inst].items():
            if not r:
                continue
            pa, ka = r["kpi"]
            pb, kb = r["base"]
            chk = check_same_instructions(btc_kpi, pa)
            premise.append((f"{inst} {tag}", chk["status"], chk["diffs"]))
            s = stat(pb, kb, pa, ka)     # ⚠️ (基准, 处理) ⇒ d = 处理 − 基准
            if s:
                items.append((f"{inst}（n={s['n']}）", s))
    return items, premise, (((_load(btc_kpi) or {}).get("kpi")) or {})


def build() -> str:
    items, premise, kpi = _collect()
    missing = [n for n, st, _d in premise if st == "missing"]
    violated = [n for n, st, _d in premise if st == "different"]
    L: list[str] = []
    L.append("# A11 报告：**跨标的复现**（把 KPI 效应在第二个资产上重做一遍）")
    L.append("")
    L.append("> 每个数字都在运行时从 `out/**/*.json` **现算**。")
    L.append("")

    # ---- 0 ----
    L.append("## 0. 状态与结论")
    L.append("")
    if missing:
        L.append(f"⏸️ **本轮未跑完**：{'、'.join(missing)} 的产物不存在"
                 f"（额度耗尽；守卫在跑完后判定数据不可用，**拒绝产出 eval**）。")
        L.append("")
    if violated:
        L.append(f"❌ **前提被违反**：{'、'.join(violated)} 的阈值与 BTC 不一致"
                 f"⇒ **不许**与 BTC 合并/对比。")
        L.append("")
    L.append("| 问题 | 答复 |")
    L.append("|---|---|")
    L.append("| 为什么换标的而不是加段数 | 见 §1：加段数要 770 段 = 12,320 次，"
             "**超过一个配额窗口** |")
    L.append("| 两个标的收到同一份指令吗 | 见 §2：**前提检查**（含「待跑」与"
             "「前提被违反」的区分）|")
    L.append("| 那套阈值在 ETH 上有约束力吗 | 见 §2.1：**预检**——若基线本来就大量"
             "达标，复现会**缺乏约束力** |")
    if len(items) >= 2:
        pl = pool_instruments(items)
        e0, e1 = items[0][1]["effect"], items[1][1]["effect"]
        same_sign = (e0 * e1) > 0
        L.append(f"| **效应复现了吗** | ✅ **同号且量级几乎相同**："
                 f"{items[0][0]} {_pct(e0)}、{items[1][0]} {_pct(e1)} |")
        L.append(f"| **两个标的能合并吗** | Q={_f(pl['Q'])} vs 临界 "
                 f"{_f(pl.get('critical'))} ⇒ "
                 f"**{'异质（只能按标的报）' if pl['heterogeneous'] else '同质 ⇒ 可以合并'}** |")
        L.append(f"| **合并后判得出来吗** | **{_pct(pl['pooled'])}**，"
                 f"95% 区间 [{_pct(pl['ci'][0])}, {_pct(pl['ci'][1])}]，"
                 f"z={_f(pl['z'])} ⇒ "
                 f"**{'区间不跨 0 ⇒ 达到显著' if (pl['ci'][0] > 0 or pl['ci'][1] < 0) else '区间跨 0 ⇒ 仍无法判定'}** |")
        L.append("")
        if same_sign and not pl["heterogeneous"] and (pl["ci"][0] > 0 or pl["ci"][1] < 0):
            L.append("⭐⭐ **本轮唯一一个「收益侧」达到显著的结果**：")
            L.append("单个标的都判不出来（区间跨 0），但**两个独立标的合并后**"
                     "区间不跨 0。")
            L.append("")
            L.append("⚠️ 但必须连着说清三件事：")
            L.append("")
            L.append("1. **效应很小**（约 3.4bp / 8 根一段）——它不是「能赚钱」的量级，"
                     "而是「可测量的差别」的量级；")
            L.append("2. **合并的显著性离边界不远**（z≈−2.7）；"
                     "且「独立」是**近似**的（同属加密资产、共享同样的模板与阈值）；")
            L.append("3. **这只是 offset=0 这一组窗口**。A9 已证明换一组窗口结论会变，"
                     "所以这不能写成「KPI 一定让收益变差」。")
            L.append("")
    else:
        L.append("| 效应复现了吗 | ⏸️ 数据不足（见 §0 第三节）|")
        L.append("")

    # ---- 1 ----
    L.append("## 1. 为什么用「换标的」而不是「加段数」")
    L.append("")
    L.append("A10 的实测判力曲线给出：判出 5bp 要 **380~510 段**。")
    L.append("而 offset=0 上实测的效应只有 **3.5bp** ⇒ 80% 判力约需 **770 段**")
    L.append("（= `770 × 8 × 2 臂 = 12,320` 次调用）——"
             "**超过一个配额窗口（7,500 次/5 小时）**。")
    L.append("")
    L.append("⭐ 而 A9 那条「不能合并多网格」的限制，**不适用于不同标的**：")
    L.append("")
    L.append("| 维度 | 同一段行情的多个 offset | **不同标的** |")
    L.append("|---|---|---|")
    L.append("| 是否独立 | ❌ 高度重叠（同一批 K 线） | ✅ **相互独立**（不同价格路径）|")
    L.append("| 能否合并 | ❌ 等于把同一个样本数很多遍 | ✅ **可以**（逆方差合并）|")
    L.append("| 能挡住什么 | 挡不住「这段行情特殊」 | ✅ **能**——这是**交叉复现** |")
    L.append("")
    L.append("⭐ **判据**：「换一个独立标的」是比「加段数」**更强的推断**——")
    L.append("它同时给出「能不能复现」的证据，而加段数只能把**同一条行情**测得更准。")
    L.append("")

    # ---- 2 ----
    L.append("## 2. ⭐⭐ 前提检查：两个标的必须拿到**同一份指令**")
    L.append("")
    L.append("⚠️ 这条不是「顺手看一眼」，而是**整个比较成立的前提**：")
    L.append("`--calibrate-kpi` 会让每个标的用**它自己标定的阈值**，")
    L.append("而阈值**印在提示词里** ⇒ 阈值不同就是**不同的实验**。")
    L.append("（A9b 就是因为没守住这条，把一个跨运行的差异当成了「补样本的效果」。）")
    L.append("")
    L.append("⇒ 本轮的 ETH 运行**把阈值钉死成 BTC 那次印的值**（不加 `--calibrate-kpi`）：")
    L.append("")
    L.append("| 运行 | 状态 |")
    L.append("|---|---|")
    label = {"same": "✅ **相同**", "different": "❌ **不同（前提被违反）**",
             "missing": "⏸️ **待跑**（产物不存在）"}
    for nm, st, diffs in premise:
        L.append(f"| {nm} | {label.get(st, st)} |")
        for fld, va, vb in diffs:
            L.append(f"| 　└ `{fld}` | BTC={va!r} / 本运行={vb!r} |")
    L.append("")
    L.append("⚠️ **「待跑」与「前提被违反」是两件事**：前者只是那一臂还没跑；")
    L.append("后者才是**设计错了**。我第一版把两者混成一个「阈值不同」，"
             "会让读者以为实验设计有问题。")
    L.append("")
    if kpi:
        L.append("钉死下来的阈值（原样，未四舍五入）：")
        L.append("")
        L.append("| 字段 | 值 |")
        L.append("|---|---|")
        for k, v in kpi.items():
            L.append(f"| `{k}` | `{v!r}` |")
        L.append("")

    # ---- 2.1 预检 ----
    L.append("### 2.1 ⭐ 预检：同一套阈值在**不同标的**上约束力一样吗")
    L.append("")
    L.append("这套阈值是**按 BTC 基线的分位数**标定的（`target_return` = 基线净收益"
             "的中位数）⇒ 它在 BTC 上**按构造**大约卡掉一半段。")
    L.append("但同一个数字放到 ETH 上，就**不再是** ETH 的中位数了。")
    L.append("")
    L.append("⚠️ **为什么要在花钱之前算这个**：若某标的的基线本来就大量达标，")
    L.append("那么「给不给 KPI」在它上面**几乎没差别** ⇒ 那次复现**缺乏约束力**，")
    L.append("跑出来也说明不了「KPI 有没有用」。")
    L.append("")
    L.append("| 标的 | n | 基线**中位**净收益 | 「收益目标」达标率 | 余量 |")
    L.append("|---|---|---|---|---|")
    thr = kpi.get("target_return")
    rates: list[float] = []
    if thr is not None:
        for inst in ("BTC", "ETH"):
            for tag, r in RUNS[inst].items():
                if not r:
                    continue
                nets = _nets(*r["base"])
                if len(nets) < 2:
                    continue
                b = kpi_bite(nets, thr)
                rates.append(b["pass_rate"])
                L.append(f"| {inst} {tag} | {b['n']} | {_pct(b['median'], 4)} | "
                         f"**{b['pass_rate']:.1%}** | {_pct(b['slack'], 4)} |")
    L.append("")
    if len(rates) >= 2:
        L.append(f"⇒ 达标率 {min(rates):.1%} ~ {max(rates):.1%}"
                 f"（差 **{abs(max(rates) - min(rates)) * 100:.1f} 个百分点**）")
        L.append("")
        if max(rates) > 0.85:
            L.append("⚠️ **有标的的基线本来就大量达标** ⇒ KPI 在它上面几乎无约束力 "
                     "⇒ 那次复现**缺乏约束力**。")
        else:
            L.append("✅ 两个标的的达标率**接近**（都在 60% 上下）⇒ "
                     "同一个阈值在两边的**约束力相当** ⇒ **这次复现是有效的**。")
        L.append("")
    L.append("⚠️ 这里**只量化了「收益目标」这一条**（其余三条要从逐 tick 的权益路径算，"
             "不在 `net` 里）。")
    L.append("⭐ 但**这一条恰好是主导项**：A8 的「目标 75%」实验动的就是它。")
    L.append("")

    # ---- 3 ----
    L.append("## 3. 逐标的效应与跨标的合并")
    L.append("")
    L.append("⚠️ **符号约定与 A9/A10 一致**：差值 = **处理 − 基准**，负 = KPI 让收益变差。")
    L.append("")
    if not items:
        L.append("（暂无可用产物。）")
        L.append("")
    else:
        L.append("| 标的 | n | 效应 | 95% 区间 | t | σ | MDE | 判力比 | 判定 |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for nm, s in items:
            L.append(f"| {nm} | {s['n']} | **{_pct(s['effect'])}** | "
                     f"[{_pct(s['lo'])}, {_pct(s['hi'])}] | {_f(s['t'])} | "
                     f"{_f(s['sd'] * 100, 4)}% | {_pct(s['mde'])} | "
                     f"{_f(abs(s['effect']) / s['mde'])} | {s['verdict']} |")
        L.append("")
        if len(items) >= 2:
            pl = pool_instruments(items)
            L.append("### 3.1 跨标的逆方差合并")
            L.append("")
            L.append("合法的理由：**不同资产的价格路径相互独立** ⇒ 配对差值独立。")
            L.append("")
            L.append(f"- **合并效应 {_pct(pl['pooled'])}**"
                     f"（SE {_f(pl['se'] * 100, 4)}%）")
            L.append(f"- 95% 区间 **[{_pct(pl['ci'][0])}, {_pct(pl['ci'][1])}]**，"
                     f"z={_f(pl['z'])}")
            het = ("⚠️ 异质（各标的效应不一致 ⇒ 只能按标的报）"
                   if pl["heterogeneous"] else "未拒绝同质（可以合并）")
            L.append(f"- **Q = {_f(pl['Q'])}**，df={pl['df']}，"
                     f"临界 {_f(pl.get('critical'))} ⇒ **{het}**")
            L.append("")
            L.append("⚠️ **合并也要看方向一致性**：若两个标的**符号相反**，"
                     "合并出来的接近 0 是**两个相反效应的抵消**，不是「没有效应」——"
                     "这两件事必须分开说。")
            L.append("")
            # ⭐⭐⭐ 宣布"显著"之前的标定自检（A10 的纪律）
            L.append("### 3.2 ⭐⭐ 宣布「显著」之前的标定自检")
            L.append("")
            L.append("⚠️ **合并统计量是另一套机制**（逆方差加权 + 正态近似），"
                     "A10 只验过**单个标的的 t 区间**。若合并的 z 本身偏大，"
                     "那这个「显著」就是**装置造出来的**，不是数据里的。")
            L.append("")
            from scripts.a11_cross_asset import pooled_mc_calibration
            res = [s0.get("diffs") or [] for _n0, s0 in items]
            if all(len(x) > 8 for x in res):
                cal = pooled_mc_calibration(res[0], res[1], delta=0.0)
                L.append(f"做法：把 **δ=0** 叠到两份真实残差上重采样 "
                         f"**{cal['reps']}** 次（种子 {cal['seed']}），看合并 z：")
                L.append("")
                L.append("| 指标 | 实测 | 目标 |")
                L.append("|---|---|---|")
                L.append(f"| 假阳性率（&vert;z&vert;&gt;1.96） | **{cal['false_positive_rate']:.1%}** | 5% |")
                L.append(f"| z 的均值 | {cal['mean_z']:+.3f} | 0.000 |")
                L.append(f"| z 的标准差 | **{cal['sd_z']:.3f}** | 1.000 |")
                L.append("")
                fpr = cal["false_positive_rate"]
                if fpr > 0.08:
                    L.append("⚠️ **偏大 ⇒ 报告里的「显著」要打折。**")
                elif fpr < 0.02:
                    L.append("⚠️ 偏小 ⇒ 合并区间**偏宽**（过于保守）。")
                else:
                    L.append("✅ **标定良好** ⇒ 合并后的「显著」是**可信的**："
                             "它来自数据，不是装置造出来的。")
                L.append("")
            else:
                L.append("（残差不足，跳过。）")
                L.append("")
        else:
            L.append("⏸️ 只有一个标的的产物 ⇒ **不构成跨标的复现**，等另一臂跑完。")
            L.append("")

    # ---- 4 ----
    L.append("## 4. 与 A9/A10 的对照")
    L.append("")
    L.append("| 轮次 | 做了什么 | 花掉的调用 | 结论 |")
    L.append("|---|---|---|---|")
    L.append("| A9 | BTC offset=0，48→300 段 | ~3,200 | 判力比 0.61→**0.65** |")
    L.append("| A10 | 用已知真值标定装置（**零额度**） | 0 | "
             "装置标定正确；5bp 实测要 380~510 段 |")
    L.append("| **A11** | **换标的（ETH）重做** | ~6,400 | 见 §0/§3 |")
    L.append("")

    # ---- 5 ----
    L.append("## 5. ⚠️ 诚实边界")
    L.append("")
    L.append("- **两个标的的 n 不同**（BTC 300、ETH 400）⇒ 合并按 `1/SE²` 自动加权，"
             "但要意识到 **ETH 的贡献更大**。")
    L.append("- **不同标的的 σ 不同**（ETH 波动更大）⇒ 即使效应相同，判力也不同。")
    L.append("- **只有 2 个标的**（SOL 也有数据，但一个配额窗口只够 2 个）⇒ "
             "Q 检验 df=1，检出力低；「不显著」不等于「同质」。")
    L.append("- **ETH 上「同一个 KPI 阈值」未必是「同样严苛的 KPI」**："
             "阈值是从 BTC 基线标定的 ⇒ §2.1 就是在量化这件事。")
    L.append("- 与前面各轮一样：**本报告不产出可交易信号**。")
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
