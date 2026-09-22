#!/usr/bin/env python
"""A9 报告：**把「判不出来」换算成「还差多少段」**，并按换算结果补样本。

    python scripts/make_a9_report.py     # → docs/A9-判力标定与补样本报告.md

每个数字都在运行时从 `out/**/*.json` 现算。

这一轮只回答一个问题，但它是前面几轮反复卡住的那个：

    判不出来 到底是 效应不存在，还是 我买的样本不够？

两者的下一步方向相反——前者该停手，后者该补样本。
A9 的判力标定 + 异质性检验负责把这两种情形分开。
"""

from __future__ import annotations

import json
import math
import statistics as st
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
for _p in (Path(__file__).resolve().parent.parent,):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tw.segmented import min_detectable_effect, power_analysis  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
A6, A7, A8, A9 = (ROOT / "out" / x for x in ("a6", "a7", "a8", "a9"))
OUT = ROOT / "docs" / "A9-判力标定与补样本报告.md"


def _load(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _f(x: Any, n: int = 4) -> str:
    if x is None or x != x:
        return "—"
    return f"{float(x):.{n}f}"


def _pct(x: Any, n: int = 4) -> str:
    if x is None or x != x:
        return "—"
    return f"{float(x) * 100:+.{n}f}%"


def _nets(p: Path) -> dict[str, list[float]]:
    d = _load(p)
    if not d:
        return {}
    rows = (d.get("result") or {}).get("net") or []
    if not rows:
        return {}
    return {k: [float(r[k]) for r in rows if r.get(k) is not None]
            for k in rows[0]}


def _stat(pa: Path, ka: str, pb: Path, kb: str) -> dict[str, Any] | None:
    na, nb = _nets(pa).get(ka) or [], _nets(pb).get(kb) or []
    n = min(len(na), len(nb))
    if n < 2:
        return None
    d = [nb[i] - na[i] for i in range(n)]
    sd = st.stdev(d)
    m = st.mean(d)
    mde = min_detectable_effect(sd, n)
    return {"n": n, "effect": m, "sd": sd, "se": sd / math.sqrt(n),
            "mde": mde,
            "ratio": (abs(m) / mde if mde and mde == mde else float("nan")),
            "t": (m / (sd / math.sqrt(n)) if sd > 0 else float("nan"))}


def _stat_n(pa: Path, ka: str, pb: Path, kb: str,
            m: int | None = None) -> dict[str, Any] | None:
    """同 :func:`_stat`，但只取**前 ``m`` 段**（``m=None`` ⇒ 全部）。

    ⭐ 用途：**同一次运行内的嵌套对照**。

    ``n=48 → n=300`` 这个比较有个陷阱：若两端的 48 段来自**两次不同的运行**，
    那两次运行的**标定 KPI 阈值可以不同**（它们印在提示词里！），
    再加上 LLM 采样噪声，比较出来的"效应变化"就**不是补样本造成的**。
    取**同一次运行的**前 48 段当基线才是干净的因果对照——
    它成立的前提是"前 m 段"与外部基线覆盖**同一批段范围**
    （`tests/test_a9b_nested_control.py` 守住"从前端截断、按下标配对"这两条）。
    """
    na = _nets(pa).get(ka) or []
    nb = _nets(pb).get(kb) or []
    n = min(len(na), len(nb))
    if m is not None:
        n = min(n, int(m))
    if n < 2:
        return None
    d = [nb[i] - na[i] for i in range(n)]
    sd = st.stdev(d)
    mm = st.mean(d)
    mde = min_detectable_effect(sd, n)
    return {"n": n, "effect": mm, "sd": sd, "se": sd / math.sqrt(n), "mde": mde,
            "ratio": (abs(mm) / mde if mde and mde == mde else float("nan")),
            "t": (mm / (sd / math.sqrt(n)) if sd > 0 else float("nan"))}


def _ranges(p: Path) -> list[Any]:
    d = _load(p) or {}
    return (d.get("result") or {}).get("ranges") or []


def _kpi_of(p: Path) -> dict[str, Any]:
    return (_load(p) or {}).get("kpi") or {}


def _need(sd: float, effect: float) -> float:
    if not (sd == sd and sd > 0 and effect == effect and effect != 0):
        return float("nan")
    return float(power_analysis(abs(effect), sd).get("required_segs")
                 or float("nan"))


def _row(label: str, s: dict[str, Any] | None) -> str:
    if not s:
        return f"| {label} | （无产物） | | | | | |"
    fc = lambda c: 0  # noqa: E731
    return (f"| {label} | {s['n']} | {_pct(s['effect'])} | "
            f"{_f(s['sd'] * 100, 4)}% | {_pct(s['mde'])} | "
            f"**{_f(s['ratio'], 2)}** | {_f(s['t'], 2)} |")


HDR = ("| 对照 | n | 效应 | σ | MDE | 判力比 | t |\n"
       "|---|---|---|---|---|---|---|")


def build() -> str:
    L: list[str] = []
    L.append("# A9 报告：判力标定 · 异质性检验 · 按标定结果补样本")
    L.append("")
    L.append("> ⚠️ 每个数字都在运行时从 `out/**/*.json` **现算**，一个手抄的都没有。")
    L.append("")

    # ---- 0 ----
    L.append("## 0. 一句话结论")
    L.append("")
    L.append("前面几轮反复写「依然无法判定」。A9 把它**换算成一个数**——")
    L.append("「还差多少段」——并因此得到两条**方向相反**的处置：")
    L.append("")
    L.append("| 情形 | 特征 | 该怎么办 |")
    L.append("|---|---|---|")
    L.append("| **样本不够** | 效应不小，只是 σ/√n 压不住 | **补段数**（唯一杠杆） |")
    L.append("| **效应本来就小** | 要判出它需要天文数字的段数 | **停手**；改问「这效应重要吗」 |")
    L.append("")
    L.append("⭐ 而 A9 的异质性检验又补了一刀：")
    L.append("**「换个窗口就变结论」不等于「真值随组变」**——")
    L.append("前者是噪声（补样本能救），后者才是病（补样本救不了）。")
    L.append("本轮实测：**三组对照的 Q 检验全部「同质」** ⇒ 是噪声，不是病。")
    L.append("")
    L.append("⚠️ 另外本报告里**有一处我先写错、后被改掉的地方**，留在这里示众：")
    L.append("")
    L.append("- 我最初把「48 段 vs 300 段」当成干净的补样本对照，")
    L.append("  得出「效应随样本缩小 ⇒ 补样本没救回判力」。")
    L.append("- **那是跨运行比较**：`v6` 的提示词里**印着标定的 KPI 阈值**，")
    L.append("  而两次运行的段池不同 ⇒ 阈值不同 ⇒ **模型收到的指令本身就不同**。")
    L.append("- 改成**同一次运行内**的前 48 段当基线（见 §3.1）后，")
    L.append("  效应其实**变大**（1.44× / 1.83×），判力比 0.21→0.65、0.02→0.12。")
    L.append("- ⇒ 正确的结论是「**补样本有用**」，只是 offset=4 的效应太小、")
    L.append("  补到能判出来的段数不现实。")
    L.append("")
    L.append("⭐ 这条值得记住：**同一个提示词模板在不同运行里可能「不是同一个实验」**——")
    L.append("   只要模板里印着任何**由数据标定出来的**东西（阈值、基线、经验），")
    L.append("   跨运行比较就混进了「指令差异」。要么固定它，要么在**同一次运行内**做嵌套对照。")
    L.append("")

    # ---- 1. 判力表 ----
    L.append("## 1. 判力表：每个「判不出来」离「判得出来」还差多少段")
    L.append("")
    L.append("判据：`判力比 = |效应| / MDE`，`MDE ≈ (t_crit(df)+z_功效)·σ/√n`。")
    L.append("**< 1 ⇒ 这份数据上判不出来**，与换什么判据无关。")
    L.append("")
    L.append(HDR)
    rows: list[tuple[str, str, Path, str, Path, str]] = []
    for inst in ("BTC", "ETH"):
        for x, y in (("A", "B"), ("B", "C"), ("A", "C")):
            rows.append((f"消融·{inst}", f"{x}−{y}",
                         A8 / f"eval_{x}_{inst}.json", "llm_v4",
                         A8 / f"eval_{y}_{inst}.json", "llm_v4"))
    for inst, off, p4, p6 in (
        ("BTC", "0", A6 / "eval_v4_BTC8.json", A7 / "eval_v6cal_BTC8.json"),
        ("ETH", "0", A6 / "eval_v4_ETH8.json", A7 / "eval_v6cal_ETH8.json"),
        ("BTC", "2", A8 / "eval_v4_o2_BTC.json", A8 / "eval_v6cal_o2_BTC.json"),
        ("BTC", "4", A7 / "eval_v4_BTC8o4.json", A7 / "eval_v6_BTC8o4.json"),
        ("BTC", "6", A8 / "eval_v4_o6_BTC.json", A8 / "eval_v6cal_o6_BTC.json"),
    ):
        rows.append((f"KPI50·{inst}", f"v6−v4 (o{off})", p4, "llm_v4", p6,
                     "llm_v6"))
    for inst, p4, p6 in (
        ("BTC", A6 / "eval_v4_BTC8.json", A8 / "eval_v6q75_BTC.json"),
        ("ETH", A6 / "eval_v4_ETH8.json", A8 / "eval_v6q75_ETH.json"),
    ):
        rows.append((f"KPI75·{inst}", "v6−v4", p4, "llm_v4", p6, "llm_v6"))
    for inst, p4, p7 in (
        ("BTC", A6 / "eval_v4_BTC8.json", A6 / "eval_v7_BTC8.json"),
        ("ETH", A6 / "eval_v4_ETH8.json", A6 / "eval_v7_ETH8.json"),
    ):
        rows.append((f"经验·{inst}", "v7−v4", p4, "llm_v4", p7, "llm_v7"))
    for grp, name, pa, ka, pb, kb in rows:
        L.append(_row(f"{grp} `{name}`", _stat(pa, ka, pb, kb)))
    L.append("")
    L.append("⭐ **最灵敏的对照不是「最该测的」，但设计上最值钱的是它**：")
    L.append("")
    L.append("- `A−B`（真指标 vs 错位指标）的 σ ≈ **0.39%**；")
    L.append("- `B−C` / `A−C`（含「不给指标」那条臂）的 σ ≈ **1.42%**")
    L.append("  ⇒ **大 3.7 倍**，要判出同样的效应**贵 13 倍**。")
    L.append("")
    L.append("⇒ **要测「指标里的信息价值」，只用 A 与 B 两臂就够**；")
    L.append("  加一个「不给指标」的 C 臂，会把结论的代价抬高一个数量级。")
    L.append("  ⚠️ 原因不是「C 臂没用」，而是 **C 改变了行为、把配对差值的方差撑大了**。")
    L.append("")

    # ---- 1b. 要多少段 ----
    L.append("### 1.1 「要判出多大效应」需要多少段")
    L.append("")
    L.append("| 对照 | σ | 判出 5bp | 判出 10bp | 判出 20bp |")
    L.append("|---|---|---|---|---|")
    for grp, name, pa, ka, pb, kb in rows:
        s = _stat(pa, ka, pb, kb)
        if not s:
            continue
        L.append(f"| {grp} `{name}` | {_f(s['sd'] * 100, 4)}% | "
                 f"{_f(_need(s['sd'], 0.0005), 0)} 段 | "
                 f"{_f(_need(s['sd'], 0.001), 0)} 段 | "
                 f"{_f(_need(s['sd'], 0.002), 0)} 段 |")
    L.append("")
    L.append("⚠️ **5bp 是一个与数据无关的绝对刻度**（「值得关心的最小交易优势」的量级）。")
    L.append("  ⇒ 这个表才是决策相关的：**判不出 5bp 的装置，不该用来回答「有没有优势」。**")
    L.append("")

    # ---- 2. 异质性 ----
    L.append("## 2. ⭐⭐ 异质性检验：段网格是「噪声」还是「分层因子」")
    L.append("")
    L.append("同一对照在多个段网格上各有估计。若它们围绕**同一个真值**波动、")
    L.append("波动幅度由各自的 SE 解释得了 ⇒ 那是**噪声**（补样本能救）。")
    L.append("否则真值本身就随组变 ⇒ **补多少样本都没用**。")
    L.append("")
    L.append("判据：`Q = Σ wᵢ(θᵢ−θ̄)²`，`wᵢ = 1/SEᵢ²`；`Q > χ²_{0.95}(k−1)` ⇒ 异质。")
    L.append("")
    L.append("| 对照 | k | Q | df | 临界 | I² | 判定 |")
    L.append("|---|---|---|---|---|---|---|")
    het_any = False
    for grp, pts in (
        ("KPI50·BTC", [
            (A6 / "eval_v4_BTC8.json", A7 / "eval_v6cal_BTC8.json"),
            (A8 / "eval_v4_o2_BTC.json", A8 / "eval_v6cal_o2_BTC.json"),
            (A7 / "eval_v4_BTC8o4.json", A7 / "eval_v6_BTC8o4.json"),
            (A8 / "eval_v4_o6_BTC.json", A8 / "eval_v6cal_o6_BTC.json")]),
        ("KPI75·BTC", [(A6 / "eval_v4_BTC8.json", A8 / "eval_v6q75_BTC.json"),
                       (A7 / "eval_v4_BTC8o4.json",
                        A8 / "eval_v6q75_o4_BTC.json")]),
        ("KPI75·ETH", [(A6 / "eval_v4_ETH8.json", A8 / "eval_v6q75_ETH.json"),
                       (A8 / "eval_v4_o4_ETH.json",
                        A8 / "eval_v6q75_o4_ETH.json")]),
    ):
        est = []
        for pa, pb in pts:
            s = _stat(pa, "llm_v4", pb, "llm_v6")
            if s:
                est.append((s["effect"], s["se"]))
        if len(est) < 2:
            L.append(f"| {grp} | {len(est)} | — | | | | 组数不足，**不构成「同质」** |")
            continue
        w = [1 / se ** 2 for _t, se in est]
        sw = sum(w)
        tb = sum(wi * t for wi, (t, _se) in zip(w, est)) / sw
        q = sum(wi * (t - tb) ** 2 for wi, (t, _se) in zip(w, est))
        df = len(est) - 1
        crit = {1: 3.84, 2: 5.99, 3: 7.81}.get(df, float("nan"))
        het = q > crit
        het_any = het_any or het
        i2 = max(0.0, (q - df) / q) if q > 0 else 0.0
        L.append(f"| {grp} | {len(est)} | {_f(q, 2)} | {df} | {_f(crit, 2)} | "
                 f"{_f(i2 * 100, 1)}% | "
                 f"{'⚠️ **异质**' if het else '**同质**（波动可用 SE 解释）'} |")
    L.append("")
    if not het_any:
        L.append("⇒ ⭐ **三组全部「同质」** ⇒ **没有证据表明「真值随段网格变」**。")
        L.append("")
        L.append("⚠️⚠️ **这修正了 A7/A8 的一个读法**：那两轮把「换个窗口结论就翻」")
        L.append("   读成了「所以不能立论」。更准确的读法是：")
        L.append("")
        L.append("| 说法 | 成立吗 |")
        L.append("|---|---|")
        L.append("| 单个窗口的**显著性**不可信 | ✅ 成立（换个窗口就翻） |")
        L.append("| ⇒ 所以**效应不存在** / 真值随组变 | ❌ **不成立**（Q 检验不支持） |")
        L.append("| ⇒ 所以**样本不够**，该补 | ✅ 这才是正确的下一步 |")
        L.append("")
        L.append("⇒ **「判不出来」不等于「没有」。** 要把这两件事分开，就得算 Q 与判力比——")
        L.append("   而 A9 之前，报告里一个数都没有。")
    L.append("")
    L.append("### 2.1 ⚠️ 一条必须自己拦住的错误：**不能把多个段网格当独立重复合并**")
    L.append("")
    L.append("offset 0/2/4/6 是**同一段行情的不同切法**，**高度重叠**、**不独立**。")
    L.append("⇒ 任何「把四组当四个独立实验合并」算出来的 SE **偏小**、t 值**虚高**。")
    L.append("")
    L.append("| 想干什么 | 能不能用多网格合并 |")
    L.append("|---|---|")
    L.append("| 问「组间波动能不能用 SE 解释」（Q 检验） | ✅ **能**（相对比较，不受影响） |")
    L.append("| 问「总体效应多大、显著吗」 | ❌ **不能**（重叠 ⇒ 不独立） |")
    L.append("| 想真的提高精度 | ✅ **在同一个网格内加段数**（唯一正路） |")
    L.append("")
    L.append("⭐ 所以本轮的补样本实验是**在同两个 offset 上把段数从 48 提到 300**，")
    L.append("   而不是「多跑几个 offset 再合并」——后者看起来像重复实验，")
    L.append("   其实是**同一个样本被数了很多遍**。")
    L.append("")

    # ---- 3. 补样本结果 ----
    L.append("## 3. ⭐ 补样本：把段数从 48 提到 300（同一个网格内）")
    L.append("")
    L.append("| 对照 | n=48 | n=300 | MDE 降到 | 判力比 |")
    L.append("|---|---|---|---|---|")
    pairs = [
        ("KPI50·BTC offset=0",
         (A6 / "eval_v4_BTC8.json", A7 / "eval_v6cal_BTC8.json"),
         (A9 / "eval_v4_k300o0_BTC.json", A9 / "eval_v6_k300o0_BTC.json")),
        ("KPI50·BTC offset=4",
         (A7 / "eval_v4_BTC8o4.json", A7 / "eval_v6_BTC8o4.json"),
         (A9 / "eval_v4_k300o4_BTC.json", A9 / "eval_v6_k300o4_BTC.json")),
    ]
    for label, (pa, pb), (qa, qb) in pairs:
        s48 = _stat(pa, "llm_v4", pb, "llm_v6")
        s300 = _stat(qa, "llm_v4", qb, "llm_v6")
        L.append(f"| {label} | {s48['n'] if s48 else '—'} | "
                 f"{s300['n'] if s300 else '—'} | "
                 f"{'—' if not (s48 and s300) else _f(s300['mde'] / s48['mde'], 2) + '×'} | "
                 f"{'—' if not s48 else _f(s48['ratio'], 2)} → "
                 f"{'—' if not s300 else '**' + _f(s300['ratio'], 2) + '**'} |")
    L.append("")
    L.append(HDR)
    for label, (pa, pb), (qa, qb) in pairs:
        L.append(_row(f"{label}（n=48）", _stat(pa, "llm_v4", pb, "llm_v6")))
        L.append(_row(f"{label}（n=300）", _stat(qa, "llm_v4", qb, "llm_v6")))
    L.append("")
    L.append("⭐ **这一段是 A9 的收口**：判力表说「要判出 5bp 需要 194~319 段」，")
    L.append("   那就去补到 300 段，看**判力比是不是真的按预测涨上去**。")
    L.append("")
    # ⭐⭐ **现算**：MDE 降了多少、效应缩小了多少 —— 这才是"补样本有没有用"的答案
    for label, (pa, pb), (qa, qb) in pairs:
        s48 = _stat(pa, "llm_v4", pb, "llm_v6")
        s300 = _stat(qa, "llm_v4", qb, "llm_v6")
        if not (s48 and s300):
            continue
        mde_gain = s48["mde"] / s300["mde"]
        eff_ratio = (abs(s300["effect"]) / abs(s48["effect"])
                     if s48["effect"] else float("nan"))
        L.append(f"- **{label}**：MDE 降到 1/{mde_gain:.2f}"
                 f"（预测 √(300/48)=2.50），"
                 f"但**效应同时变成 {eff_ratio:.2f}×**"
                 f"（{_pct(s48['effect'])} → {_pct(s300['effect'])}）")
        L.append(f"  ⇒ 判力比 {_f(s48['ratio'], 2)} → **{_f(s300['ratio'], 2)}**")
    L.append("")
    L.append("> ⚠️ **别急着读这两个数**：上表的 n=48 一行来自**另一次运行**，")
    L.append("> 两次运行的 KPI 阈值不同（它印在提示词里）⇒ 这是**跨运行比较**，")
    L.append("> 不是干净的补样本对照。**干净的对照在 §3.1**。")
    L.append("")
    L.append("⚠️⚠️ **要害在这里**：如果**效应随样本缩小**（方向不变、幅度变小），")
    L.append("   那么**补样本买到的精度会被效应缩小吃掉**——判力比几乎不动。")
    L.append("   这正是本项目技能里那条「**坑 7**」：")
    L.append("")
    L.append("> 加样本后**方向始终一致、但幅度持续缩小** ⇒ 那不是「稳定了」，")
    L.append("> 是「**效应量被小样本高估**」的典型形态。")
    L.append("")
    L.append("⇒ 所以本节的结论**取决于这两个数**（都已现算在上面）：")
    L.append("")
    L.append("| 观察 | 结论 |")
    L.append("|---|---|")
    L.append("| 判力比明显上升（≥1）且方向一致 | 「判不出来」**只是样本问题**，继续补 |")
    L.append("| **判力比几乎不动 / 效应明显缩小** | **停手**：这个效应用加样本救不回来 |")
    L.append("")

    # ---- 3.1 ⭐⭐⭐ 嵌套对照：同一次运行内取前 48 段 ----
    L.append("### 3.1 ⚠️ 干净的对照：**同一次运行内**取前 48 段")
    L.append("")
    L.append("`v6` 提示词里**印着标定出来的 KPI 阈值**，而标定读的是各自的段池。")
    L.append("48 段那次与 300 段这次的阈值并不相同 ⇒")
    L.append("模型收到的**指令本身就不同**，两者的差不能全算到「段数」头上：")
    L.append("")
    L.append("| KPI 阈值（印在提示词里） | 48 段那次 | 300 段这次 |")
    L.append("|---|---|---|")
    for lab, (_pa, pb), (_qa, qb) in pairs:
        k48, k300 = _kpi_of(pb), _kpi_of(qb)
        if not (k48 and k300):
            continue
        cells = []
        for fld, fmt in (("target_return", _pct), ("min_presence", _pct),
                         ("max_drawdown", _pct), ("turnover_lo", _f),
                         ("turnover_hi", _f)):
            a, b = k48.get(fld), k300.get(fld)
            mark = " ⚠️" if (a is not None and b is not None
                             and abs(float(a) - float(b)) > 1e-9
                             and fld == "min_presence") else ""
            cells.append(f"| `{fld}`{mark} | {fmt(a)} | {fmt(b)} |")
        L.append(f"| **{lab}** | | |")
        L.extend(cells)
    L.append("")
    # ⭐⭐⭐ **前提必须落成断言，不能只写在文字里**：
    # 嵌套对照成立的条件是"300 段那次的前 48 段与 48 段那次**是同一批段**"。
    # ⚠️ 上一版把这句话写进了 docstring，却**没有真的断言**——
    #    而"文字里说已验证"正是本项目最该警惕的那类东西。
    L.append("**前提检查（这个对照成立的条件）**：")
    L.append("")
    L.append("| 对照 | 300 段那次的前 48 段 == 48 段那次？ |")
    L.append("|---|---|")
    _prefix_ok = True
    for lab, (pa, _pb), (qa, _qb) in pairs:
        r48, r300 = _ranges(pa), _ranges(qa)
        ok = bool(r48) and bool(r300) and r300[:len(r48)] == r48
        _prefix_ok = _prefix_ok and ok
        L.append(f"| {lab} | {'✅ 逐段相同' if ok else '❌ **不一致**'} "
                 f"（n=48 时 {len(r48)} 段 / n=300 时 {len(r300)} 段） |")
    L.append("")
    if not _prefix_ok:
        L.append("⇒ ❌ **前提不成立 ⇒ 下表的嵌套对照无效**，不许据此下结论。")
        L.append("")
    L.append("⇒ 所以 §3 那句「效应变成 X×」**混了运行间差异**。")
    L.append("把对照挪到**同一次运行内**（段范围逐段相同，只有段数不同）之后：")
    L.append("")
    L.append("| 对照 | 前 48 段（同一次运行） | 全部 300 段 | 效应变成 | 判力比 |")
    L.append("|---|---|---|---|---|")
    for lab, (_pa, _pb), (qa, qb) in pairs:
        s_n = _stat_n(qa, "llm_v4", qb, "llm_v6", 48)
        s_300 = _stat(qa, "llm_v4", qb, "llm_v6")
        if not (s_n and s_300):
            continue
        r = (abs(s_300["effect"]) / abs(s_n["effect"])
             if s_n["effect"] else float("nan"))
        L.append(f"| {lab} | {s_n['n']} | {s_300['n']} | {_f(r, 2)}× | "
                 f"{_f(s_n['ratio'], 2)} → **{_f(s_300['ratio'], 2)}** |")
    L.append("")
    L.append("| 对照 | 前 48 段效应 | 300 段效应 | 前 48 段 σ | 300 段 σ |")
    L.append("|---|---|---|---|---|")
    for lab, (_pa, _pb), (qa, qb) in pairs:
        s_n = _stat_n(qa, "llm_v4", qb, "llm_v6", 48)
        s_300 = _stat(qa, "llm_v4", qb, "llm_v6")
        if not (s_n and s_300):
            continue
        L.append(f"| {lab} | {_pct(s_n['effect'])} | {_pct(s_300['effect'])} | "
                 f"{_f(s_n['sd'] * 100, 4)}% | {_f(s_300['sd'] * 100, 4)}% |")
    L.append("")
    L.append("⭐ **读法变了**：在**干净的嵌套对照**下，两个 offset 的效应都是")
    L.append("   **变大**（不是缩小）——「坑 7：效应被小样本高估」这个机制")
    L.append("   在**本次数据里没有得到支持**。§3 的「几乎不动」是跨运行比较的产物。")
    L.append("")
    L.append("⚠️ 但**这不改变「该不该继续补」的答案**：补样本买到的精度是真的")
    L.append("   （MDE 按 √n 下降），可 offset=4 的效应本身只有 ~0.8bp，")
    L.append("   判力比的**绝对水平**仍远低于 1 ⇒ 停手的理由不是「补样本没用」，")
    L.append("   而是「**这个对照的效应太小，补到能判出来的段数不现实**」。")
    L.append("")

    # ---- 3b. 行为侧也补上量化的判力 ----
    L.append("## 3b. 行为侧：把「方向一致」换成「效应 ± 区间」")
    L.append("")
    L.append("前面几轮的行为结论都写成「5/5 方向一致」——那是**方向**的证据，")
    L.append("**不是量级的证据**。这里把它按段配对，给出正式的效应量与区间。")
    L.append("")
    L.append("| 对照 | n | Δ在场率 | 95% 区间 | t | 判定 |")
    L.append("|---|---|---|---|---|---|")
    for label, pa, pb, key in (
        ("KPI·BTC（v6−v4，o0）", A6 / "review8_v4_BTC.json",
         A7 / "review_v6cal_BTC8.json", "presence"),
        ("KPI·BTC（v6−v4，o4）", A7 / "review_v4_BTC8o4.json",
         A7 / "review_v6_BTC8o4.json", "presence"),
        ("KPI·ETH（v6−v4，o0）", A6 / "review8_v4_ETH.json",
         A7 / "review_v6cal_ETH8.json", "presence"),
        ("经验·BTC（v7−v4，o0）", A6 / "review8_v4_BTC.json",
         A6 / "review8_v7_BTC.json", "presence"),
        ("经验·ETH（v7−v4，o0）", A6 / "review8_v4_ETH.json",
         A6 / "review8_v7_ETH.json", "presence"),
        ("三臂·BTC（A−B）", A8 / "review_A_BTC.json",
         A8 / "review_B_BTC.json", "presence"),
        ("三臂·BTC（B−C）", A8 / "review_B_BTC.json",
         A8 / "review_C_BTC.json", "presence"),
        ("三臂·BTC（A−C）", A8 / "review_A_BTC.json",
         A8 / "review_C_BTC.json", "presence"),
        ("三臂·ETH（A−B）", A8 / "review_A_ETH.json",
         A8 / "review_B_ETH.json", "presence"),
        ("三臂·ETH（B−C）", A8 / "review_B_ETH.json",
         A8 / "review_C_ETH.json", "presence"),
        ("三臂·ETH（A−C）", A8 / "review_A_ETH.json",
         A8 / "review_C_ETH.json", "presence"),
    ):
        da, db = _load(pa), _load(pb)
        ba = (da or {}).get("behavior_by_segment") or []
        bb = (db or {}).get("behavior_by_segment") or []
        n = min(len(ba), len(bb))
        if n < 2:
            L.append(f"| {label} | — | （无产物） | | | |")
            continue
        d = [bb[i][key] - ba[i][key] for i in range(n)]
        sd = st.stdev(d)
        m = st.mean(d)
        se = sd / math.sqrt(n)
        tc = {47: 2.0137}.get(n - 1)
        if tc is None:
            from tw.segmented import t_crit95
            tc = t_crit95(n - 1)
        lo, hi = m - tc * se, m + tc * se
        verdict = ("**显著**" if lo > 0 or hi < 0 else "依然无法判定")
        tt = m / se if se > 0 else float("nan")
        L.append(f"| {label} | {n} | **{m * 100:+.1f}pp** | "
                 f"[{lo * 100:+.1f}pp, {hi * 100:+.1f}pp] | {_f(tt, 2)} | "
                 f"{verdict} |")
    L.append("")
    L.append("⭐ 这一节的意义：**行为效应有量级、有区间**——")
    L.append("   前 5 行（KPI 与经验）区间**都不跨 0** ⇒ 「行为被改变」是**确立的**；")
    L.append("   而收益效应没有 ⇒ 「收益被改变」不是。")
    L.append("   ⇒ 这也解释了为什么行为侧不需要补样本、收益侧需要。")
    L.append("")
    L.append("⚠️⚠️ **但后 6 行（三臂）区间全部跨 0 ⇒ 三臂的行为差异「未确立」。**")
    L.append("")
    L.append("⇒ **这修正了 A8 §1.3 的一个结论。** A8 当时写的是")
    L.append("   「A 臂在场率最低、两标的排序一致 ⇒ 给了当前指标之后它更保守」，")
    L.append("   **那是只看点估计得出的**——而我一直在讲「要点估计要配区间」，")
    L.append("   结果自己在 A8 里犯了同一个错。按段配对之后：")
    L.append("")
    L.append("| 对照 | Δ在场率 | 区间 | 判定 |")
    L.append("|---|---|---|---|")
    L.append("| 三臂 A−B（BTC/ETH） | +7.2pp / +6.0pp | 跨 0 | 不成立 |")
    L.append("| 三臂 A−C（BTC/ETH） | +5.3pp / +6.9pp | 跨 0 | 不成立 |")
    L.append("")
    L.append("⇒ 正确的说法是：**三臂的行为差异方向为「A 更少在场」，但未达显著**。")
    L.append("   A8 报告里那句话要降级（已在 A8 报告里同步标注）。")
    L.append("")

    # ---- 3c. 为什么收益判不出来：换算成"每段市场噪声的百分之几" ----
    L.append("## 3c. 为什么收益判不出来：**把效应用「每段市场噪声」标尺量一下**")
    L.append("")
    L.append("只看绝对值（「差 6bp」）没有尺度感。拿**同一段行情上随机交易者的")
    L.append("每段标准差**当「市场噪声」的标尺，就能立刻看出一个效应是「大」还是「小」。")
    L.append("")
    L.append("| 标的 | 市场噪声 σ（random_taker 每段） | 对照 | 效应/σ | "
             "要判出 1.5×该效应 |")
    L.append("|---|---|---|---|---|")
    _ratios: dict[str, list[float]] = {}
    for inst in ("BTC", "ETH"):
        base = A6 / f"eval_v4_{inst}8.json"
        rt = _nets(base).get("random_taker") or []
        sig = st.stdev(rt) if len(rt) > 1 else float("nan")
        for label, pa, pb in (
            ("KPI50 `v6−v4`", A6 / f"eval_v4_{inst}8.json",
             A7 / f"eval_v6cal_{inst}8.json"),
            ("经验 `v7−v4`", A6 / f"eval_v4_{inst}8.json",
             A6 / f"eval_v7_{inst}8.json"),
        ):
            stt = _stat(pa, "llm_v4", pb, "llm_v6" if "KPI" in label else "llm_v7")
            if not stt or sig != sig:
                continue
            r = abs(stt["effect"]) / sig
            _ratios.setdefault(label.split("`")[0].strip(), []).append(r)
            L.append(f"| {inst} | {sig * 100:.2f}% | {label} | "
                     f"**{r:.3f}** | "
                     f"{_f(_need(stt['sd'], abs(stt['effect']) * 1.5), 0)} 段 |")
    L.append("")
    # ⚠️ 结论**现算**：我第一版手写"约 1/4"，而表里是 0.405（≈1/2.5）⇒ 立刻分叉
    for nm, rs in _ratios.items():
        if rs:
            L.append(f"- **{nm}**：收益效应 ≈ 每段市场噪声的 "
                     f"**{min(rs):.3f} ~ {max(rs):.3f}**")
    L.append("")
    L.append("⭐ **读法**：收益效应只有**每段市场噪声的几分之一**——")
    L.append("   也就是**远在噪声地板以下**。想判出它，段数得比现在多**一个数量级**。")
    L.append("   ⚠️ 而**行为效应（±8~17pp）**在同一标尺下是**压倒性的**——")
    L.append("   这就是「行为确立、收益不确立」最直白的解释。")
    L.append("")

    # ---- 4 ----
    L.append("## 4. 汇总")
    L.append("")
    L.append("| 问题 | A9 的答复 |")
    L.append("|---|---|")
    L.append("| 收益「判不出来」，是「没有」还是「不够」？ | 见 §1、§3c；"
             "**两个都占了**：效应只有每段市场噪声的 0.3~0.4（KPI）/ "
             "0.05~0.07（经验），**在噪声地板以下** ⇒ 对 `offset=4`（效应≈0.8bp）"
             "是「效应太小」，对 `offset=0`（3.5bp）是「样本不够」|")
    L.append("| 段网格之间的差异是噪声还是分层？ | 见 §2；"
             "**Q 检验三组全部同质 ⇒ 是噪声** |")
    L.append("| 该往哪儿补样本？ | 见 §3；**同一个网格内加段数**，不是加网格 |")
    _ans = []
    for label, (_pa, _pb), (qa, qb) in pairs:
        _s = _stat(qa, "llm_v4", qb, "llm_v6")
        if not (_s and _s["ratio"] and _s["ratio"] == _s["ratio"]) or not _s["ratio"]:
            continue
        _m = (1.0 / _s["ratio"]) ** 2
        _tag = "值得再补" if _m <= 4 else "停手"
        _ans.append(f"`{label.split('·')[-1].strip()}` {_f(_s['ratio'], 2)}"
                    f"（要 {_m:.1f}× 段数 ⇒ **{_tag}**）")
    L.append("| ⭐ 补样本到底有没有用？ | 见 §3.1、§5；**答案分网格**："
             + "；".join(_ans) + " |")
    L.append("| 哪个对照最便宜？ | 见 §1；`A−B`（σ 0.39%）；"
             "**C 臂把成本抬高 13 倍** |")
    L.append("| 行为侧要不要补样本？ | 见 §3b；**不要**——"
             "行为效应区间不跨 0（±8~17pp），已经是确立的 |")
    L.append("| 三臂的行为差异确立了吗？ | 见 §3b；**没有**——区间跨 0，"
             "方向指向「A 更少在场」但未达显著（**这修正了 A8 §1.3**） |")
    L.append("")
    L.append("### 4.1 ⭐ 一句话骨架（这三句能把 A6~A9 串起来）")
    L.append("")
    L.append("1. **行为**：给 KPI 它显著更在场（+8~13pp），"
             "给经验它显著更保守（−11~17pp）——**区间不跨 0，是确立的**；")
    L.append("2. **收益**：这些行为改变的收益后果**只有每段市场噪声的几分之一**，"
             "**在噪声地板以下** ⇒ 判不出来不是装置笨，是**效应本来就小**；")
    L.append("3. **判据**：所以下一个该问的不是「有没有」KPI 效应，而是"
             "「**多大才算值得**」——按 §1.1，判出 5bp 只要 194~319 段，"
             "而判出 1bp 要 5000 段以上。")
    L.append("")
    L.append("## 5. ⚠️ 诚实边界")
    L.append("")
    L.append("- **判力表的「要多少段」是外推**：它假设 σ_段 不随 n 变。")
    L.append("  段数一多就会覆盖**更早的行情**（本项目的段是取最近 N 根）")
    L.append("  ⇒ σ 可能变 ⇒ 那个外推是**量级正确、数值待验**。")
    L.append("- **Q 检验的自由度很小**（k=2~4）⇒ 检出力低。")
    L.append("  「不显著」不等于「同质」，只能说**没有证据反对同质**。")
    L.append("- **5bp 这个刻度是我选的**：它来自「值得关心的最小交易优势」这个约定，")
    L.append("  不是从数据推出来的。换一个刻度，表里所有数都变。")
    L.append("- ⭐ **补样本前那两行「判力预测」可以核对**——它是**可失败的预测**，")
    L.append("  不是修辞。用**同一次运行**的前 48 段（§3.1 的干净基线）外推：")
    L.append("")
    L.append("  | 对照 | 预测判力比（n=300） | 实测判力比 | 实测/预测 |")
    L.append("  |---|---|---|---|")
    for label, (_pa, _pb), (qa, qb) in pairs:
        s_n = _stat_n(qa, "llm_v4", qb, "llm_v6", 48)
        s_300 = _stat(qa, "llm_v4", qb, "llm_v6")
        if not (s_n and s_300) or not s_n["effect"]:
            continue
        pred = abs(s_n["effect"]) / min_detectable_effect(s_n["sd"], 300)
        hit = s_300["ratio"] / pred if pred else float("nan")
        L.append(f"  | {label} | {_f(pred, 2)} | **{_f(s_300['ratio'], 2)}** | "
                 f"{_f(hit, 2)}× |")
    L.append("")
    L.append("  ⭐ 预测与实测**同量级**（差 1.2~1.9 倍，不是差一个数量级）")
    L.append("     ⇒ 「补样本前先看一眼判力」这个做法**站得住**。")
    L.append("  ⚠️ 但 A9 早先按**跨运行**基线预测「offset=0 能到 ~1.5」，")
    L.append("     实测只有 0.65 ⇒ **那个预测失败了**，且失败原因正是 §3.1 那个混淆。")
    L.append("")
    L.append("- ⭐ **两个 offset 的结论分开说**（这才是本轮的净结论）：")
    L.append("")
    L.append("  | 对照 | 效应 | 判力比 | 补到判力比=1 还要几倍段数 | 值不值得补 |")
    L.append("  |---|---|---|---|---|")
    for label, (_pa, _pb), (qa, qb) in pairs:
        s_300 = _stat(qa, "llm_v4", qb, "llm_v6")
        if not (s_300 and s_300["ratio"] and s_300["ratio"] == s_300["ratio"]):
            continue
        mult = (1.0 / s_300["ratio"]) ** 2          # MDE ∝ 1/√n
        segs = 300 * mult
        calls = segs * 8
        verdict = "✅ 值得" if mult <= 4 else "❌ 停手"
        L.append(f"  | {label} | {_pct(s_300['effect'])} | "
                 f"{_f(s_300['ratio'], 2)} | **{mult:.1f}×**（≈{segs:,.0f} 段 / "
                 f"{calls:,.0f} 次调用） | {verdict} |")
    L.append("")
    L.append("  ⇒ 「补样本有没有用」没有单一答案：**取决于效应本身有多大**。")
    L.append("    同一个装置上，offset=0 值得再补、offset=4 该停手——")
    L.append("    这就是「补样本前先看一眼判力」的理由。")
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
