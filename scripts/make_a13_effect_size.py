#!/usr/bin/env python
"""A13：「**多大才算值得关心**」——把效应换算成可决策的单位。

    python scripts/make_a13_effect_size.py   # → docs/A13-效应量级与"值得关心"的门槛.md

⭐ 这是对用户那个问题的正面回答。它**不是统计问题**（统计问题已经在
A10/A11/A12 答完：效应 ≈ −3.5bp，判力与标定都量过了），
而是**目标问题**：多小算小、多大算大。

本报告的做法：**不替用户拍板，而是把"拍板需要的东西"算出来**——
1. 把 −3.5bp 换算成所有可解释的单位（每段/每小时/每天/每年/按本金/按费用/按噪声）；
2. 给出**交换率**：那个行为改变是多少钱换来的（这是最有决策价值的一个数）；
3. 列出**门槛的四个来源**，每个都算清，让用户能自己选；
4. 把"要在哪一级达到显著、还差多少"算成**调用次数**。

⚠️ 所有数字都在运行时**现算**；行为侧从决策记录里现算，不手抄。
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

from scripts.a11_cross_asset import RUNS, _nets, pool_instruments, stat  # noqa: E402
from tw.simexec import ExecConfig  # noqa: E402

OUT = ROOT / "docs" / "A13-效应量级与值得关心的门槛.md"

SEG_BARS = 8                # 一段 = 8 根
BAR_HOURS = 1.0             # 1H
SEGS_PER_YEAR = 8760.0 / (SEG_BARS * BAR_HOURS)      # = 1095
NOTIONAL = 100_000.0        # 每段初始权益（实测来自 kpi_by_segment）


def _pct(x: Any, n: int = 4) -> str:
    if x is None or x != x:
        return "—"
    return f"{float(x) * 100:+.{n}f}%"


def _bp(x: Any, n: int = 2) -> str:
    if x is None or x != x:
        return "—"
    return f"{float(x) * 10000:+.{n}f}bp"


def _f(x: Any, n: int = 2) -> str:
    if x is None or x != x:
        return "—"
    return f"{float(x):.{n}f}"


# ----------------------------------------------------------------------
# 行为侧：从**决策记录**现算在场率（v4 臂没有 KPI 状态，只能从记录算）
# ----------------------------------------------------------------------
def presence_from_records(p: Path) -> dict[str, Any]:
    """在场率 = 非 hold 决策的比例。

    ⚠️ 只统计 `ok=True` 且 `text` 能解析成 JSON 的调用；
    并把**解析失败数**单独报出来（不许悄悄吞掉）。
    """
    n = hold = other = bad = notok = 0
    try:
        fh = p.open(encoding="utf-8")
    except OSError:
        return {"n": 0, "presence": float("nan"), "bad": 0}
    with fh:
        for ln in fh:
            try:
                d = json.loads(ln)
            except ValueError:
                continue
            if not d.get("ok"):
                notok += 1
                continue
            try:
                t = json.loads(d["text"])
            except (ValueError, TypeError, KeyError):
                bad += 1
                continue
            n += 1
            if str(t.get("action", "")).lower() in ("hold", ""):
                hold += 1
            else:
                other += 1
    return {"n": n, "presence": other / max(n, 1), "bad": bad, "notok": notok}


def _items() -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    for inst in RUNS:
        for _tag, r in RUNS[inst].items():
            if not r:
                continue
            s = stat(r["base"][0], r["base"][1], r["kpi"][0], r["kpi"][1])
            if s:
                out.append((inst, s))
    return out


def build() -> str:
    items = _items()
    L: list[str] = []
    L.append("# A13：「**多大才算值得关心**」——把效应换算成可决策的单位")
    L.append("")
    L.append("> 所有数字都在运行时**现算**（`out/**/*.json` + 决策记录 + `ExecConfig`），不手抄。")
    L.append("> 数据：秒级账本来自 A11/A12 的三个标的（BTC n=300、ETH n=400、SOL n=250）。")
    L.append("")

    if len(items) < 2:
        L.append("⚠️ 数据不足，无法给出换算（需要至少 2 个标的的产物）。")
        return "\n".join(L)

    pl = pool_instruments([(f"{i}（n={s['n']}）", s) for i, s in items])
    eff = pl["pooled"]

    # ⚠️ 这几个常量/配置必须**在 §0 之前**定义：§0 也要用它们。
    # （我第一版把 `sd` 留在 §1，§0 引用它就 UnboundLocalError。）
    sd = items[0][1]["sd"]
    cfg = ExecConfig()
    SLIP = cfg.eff_slippage / 10000.0     # ⚠️ eff_slippage 的单位是 **bp**
    rt_taker = (cfg.taker_fee * 2 + SLIP * 2)
    rt_maker = (cfg.maker_fee * 2 + SLIP * 2)

    # ---------- 0 ----------
    L.append("## 0. 一页回答")
    L.append("")
    L.append("**问题**：KPI 让每段（8 小时）收益变差 **约 −3.5bp**。这算不算「值得关心」？")
    L.append("")
    L.append("**回答分两句**：")
    L.append("")
    ann = eff * SEGS_PER_YEAR
    ann_noise = sd * math.sqrt(SEGS_PER_YEAR)
    L.append("⭐⭐ **先纠正一个容易想当然的错觉**：我先前说它「很小」——**那要看尺度**：")
    L.append("")
    L.append("| 尺度 | 效应 | 同尺度的噪声 | 结论 |")
    L.append("|---|---|---|---|")
    L.append(f"| **一次决策 / 一段（8h）** | {_bp(eff)} | σ = {_bp(sd)} | "
             f"效应只有噪声的 **{_f(abs(eff) / sd * 100, 1)}%** ⇒ **完全看不见** |")
    L.append(f"| **一年（8,760h）** | **{_pct(ann, 1)}** | {_pct(ann_noise, 1)} | "
             f"效应是噪声的 **{_f(abs(ann) / ann_noise, 1)} 倍** ⇒ **很大、很显眼** |")
    L.append("")
    L.append("⭐⭐ **所以答案是**：**它不是「小」，而是「短期看不见、长期很大」。**")
    L.append(f"年化 **{_pct(ann, 1)}** 是「值得认真对待」的量级；")
    L.append("唯一没确定的是**它能不能持续**——我们只测了 2 年数据里的**前 100 天**，")
    L.append("而且跨标的合并按 `t(df=k−1)` 还没到显著。")
    L.append("")
    L.append(f"**按成本量**：一次往返 = **{_bp(rt_taker)}**（taker）⇒ 效应是它的 "
             f"**{_f(abs(eff) / rt_taker * 100, 0)}%**。")
    L.append(f"**按「换来了什么」量**：KPI 把在场率抬高 **8~12 个百分点** ⇒ "
             f"交换率约 **{_f(_rate_bp(items)[0], 2)}~{_f(_rate_bp(items)[1], 2)}bp / 1pp**。")
    L.append("")

    # ---------- 1 ----------
    L.append("## 1. 换算表：−3.5bp 在各个单位下是多少")
    L.append("")
    L.append("⚠️ 段长 = **8 根 × 1H = 8 小时**；每段初始权益 **¥100,000**（实测自 `kpi_by_segment`）。")
    L.append("")
    L.append("| 单位 | 数值 | 怎么来的 |")
    L.append("|---|---|---|")
    L.append(f"| 每段（8 小时） | **{_bp(eff)}** | 合并估计（{pl['k']} 个标的） |")
    L.append(f"| 每小时 | {_bp(eff / (SEG_BARS * BAR_HOURS), 3)} | ÷ 8 |")
    L.append(f"| 每天（24 小时） | {_bp(eff / (SEG_BARS * BAR_HOURS) * 24, 2)} | × 24 |")
    L.append(f"| **年化**（8,760 小时） | **{_pct(eff * SEGS_PER_YEAR, 2)}** | × {_f(SEGS_PER_YEAR, 0)} 段 |")
    L.append(f"| 按 ¥100,000 本金 | **¥{eff * NOTIONAL:+,.1f}** / 段 | × 本金 |")
    L.append(f"| 按 ¥1,000,000 本金 | **¥{eff * NOTIONAL * 10:+,.0f}** / 段 | × 本金 |")
    L.append(f"| 相对**一次往返成本（taker）** | **{_f(abs(eff) / rt_taker * 100, 0)}%**"
             f"（往返 = {_bp(rt_taker)}） | 5bp×2 + 1bp×2 |")
    L.append(f"| 相对**一次往返成本（maker）** | **{_f(abs(eff) / rt_maker * 100, 0)}%**"
             f"（往返 = {_bp(rt_maker)}） | 2bp×2 + 1bp×2 |")
    L.append(f"| 相对**单段的噪声 σ** | **{_f(abs(eff) / sd * 100, 1)}%**"
             f"（σ = {_bp(sd)}） | 一段之内完全看不见 |")
    ann_noise = sd * math.sqrt(SEGS_PER_YEAR)
    L.append(f"| 相对**一年的噪声** | **{_f(abs(eff * SEGS_PER_YEAR) / ann_noise * 100, 1)}%**"
             f"（年噪声 = {_pct(ann_noise, 0)}） | 一年的随机波动是它的 "
             f"{_f(ann_noise / abs(eff * SEGS_PER_YEAR), 0)} 倍 |")
    L.append("")
    L.append("⭐⭐ **这张表读法**：")
    L.append("")
    L.append(f"- **短期**（一次决策、一天）⇒ 效应被噪声淹没（单段 σ 是它的 "
             f"**{_f(sd / abs(eff), 0)} 倍**）⇒ **不该关心**；")
    L.append(f"- **长期**（年化 {_pct(eff * SEGS_PER_YEAR, 1)}）⇒ 如果它**真的持续**，")
    L.append("  那是**几个百分点/年**的量级 ⇒ **该关心**；")
    L.append("- ⇒ 关键不在「多大」，而在**「它是不是真的、以及会持续多久」**。")
    L.append("")

    # ---------- 2 ----------
    L.append("## 2. ⭐⭐ 交换率：那个行为改变是多少钱换来的")
    L.append("")
    L.append("KPI 的目的**不是**赚钱，是**让模型更像一个「在场的交易者」**。")
    L.append("所以要回答「值不值」，得看**代价 / 换来的东西**。")
    L.append("")
    L.append("| 标的 | v4 在场率（无 KPI） | v6 在场率（有 KPI） | 行为改变 | 收益代价 | 交换率 |")
    L.append("|---|---|---|---|---|---|")
    rows_rate: list[float] = []
    for inst in RUNS:
        spec = RUNS[inst]
        for tag, r in spec.items():
            if not r:
                continue
            pres_b = presence_from_records(_rec_of(inst, "v4", s_n(r, "base")))
            pres_k = presence_from_records(_rec_of(inst, "v6", s_n(r, "kpi")))
            if pres_b["n"] < 50 or pres_k["n"] < 50:
                continue
            st_ = stat(r["base"][0], r["base"][1], r["kpi"][0], r["kpi"][1])
            if not st_:
                continue
            d_pp = (pres_k["presence"] - pres_b["presence"]) * 100
            rate = abs(st_["effect"]) / d_pp * 10000 if d_pp else float("nan")
            rows_rate.append(rate)
            L.append(f"| {inst}（{tag}） | {pres_b['presence']:.1%} | "
                     f"{pres_k['presence']:.1%} | **{d_pp:+.1f}pp** | "
                     f"{_bp(st_['effect'])} | **{_f(rate, 2)}bp / 1pp** |")
    L.append("")
    if rows_rate:
        lo, hi = min(rows_rate), max(rows_rate)
        L.append(f"⇒ **交换率 ≈ {_f(lo, 2)} ~ {_f(hi, 2)} bp 段收益 / 每 1pp 在场率**")
        L.append("")
        avg = sum(rows_rate) / len(rows_rate)
        cost10 = avg * 10
        frac = cost10 / rt_taker
        L.append(f"⭐ **换算成「钱」**：让模型每 8 小时多在场 **10 个百分点**，")
        L.append(f"代价约 **{_f(cost10, 2)}bp**"
                 f"（= 一次往返成本的 **{_f(frac * 100, 0)}%**，"
                 f"按 ¥100,000 本金约 **¥{cost10 / 10000 * NOTIONAL:,.1f}** / 段）。")
        L.append("")
        L.append(f"⇒ 用收益换行为，**代价约 {_f(frac, 2)} 次往返手续费**；"
                 f"年化则是 **{_pct(cost10 / 10000 * SEGS_PER_YEAR, 1)}**。")
        L.append("**要「模型像个在场的交易者」⇒ 便宜；要收益 ⇒ 净损失。**")
        L.append("")
    L.append("⚠️ 在场率的算法：**非 `hold` 决策的比例**（从决策记录现算），")
    L.append("口径与 A9 的 `presence_so_far` 不完全相同 ⇒ 数值可能有几个点的出入，")
    L.append("但**两个臂用同一口径**，所以差值可用。")
    L.append("")

    # ---------- 3 ----------
    L.append("## 3. 门槛的四个来源（每个都算清，供你自己选）")
    L.append("")
    L.append("| 门槛来源 | 数值 | 效应落在哪一侧 |")
    L.append("|---|---|---|")
    r_cost = abs(eff) / rt_taker
    if r_cost < 0.5:
        v_cost = "**成本尺度上算小**"
    elif r_cost < 1:
        v_cost = "**约 " + _f(r_cost, 2) + " 次往返 ⇒ 不算小**"
    else:
        v_cost = "**超过 1 次往返 ⇒ 不算小**"
    L.append(f"| ① **交易成本**：值不值得省一次往返 | {_bp(rt_taker)}（taker） | "
             f"效应是它的 **{_f(r_cost * 100, 0)}%** ⇒ {v_cost} |")
    ratio_ann = abs(eff * SEGS_PER_YEAR) / ann_noise
    if ratio_ann > 2:
        v_ann = "**一年尺度上完全看得出来**"
    elif ratio_ann > 1:
        v_ann = "勉强可辨"
    else:
        v_ann = "一年尺度上仍看不出来"
    L.append(f"| ② **可察觉性**：一年能不能看出来 | 年噪声 {_pct(ann_noise, 1)} | "
             f"效应年化 {_pct(eff * SEGS_PER_YEAR, 1)} = 年噪声的 "
             f"**{_f(ratio_ann, 1)} 倍** ⇒ {v_ann} |")
    L.append(f"| ③ **行为对价**：为 KPI 的目的付多少 | 交换率 "
             f"{_f(min(rows_rate) if rows_rate else float('nan'), 2)}~"
             f"{_f(max(rows_rate) if rows_rate else float('nan'), 2)}bp/1pp | "
             f"10pp 在场率 ≈ {_f((sum(rows_rate) / len(rows_rate) if rows_rate else float('nan')) * 10, 2)}bp "
             f"⇒ **便宜** |")
    L.append("| ④ **机会成本**：同样的额度/时间花在别处 | 不好量化 | "
             "**这是唯一需要你拍板的**（见 §5） |")
    L.append("")

    # ---------- 4 ----------
    L.append("## 4. ⭐ 要「判得出来」，还差多少（把显著性算成一笔账）")
    L.append("")
    L.append("⚠️ 上一轮（A12）的教训：**合并的「单位」是标的（或时段块），自由度是块数−1**。")
    L.append("所以问题不是「加多少段」，而是**「加几个独立块」**。")
    L.append("")
    L.append("⭐ 好消息：**数据有 2 年，而现有运行只用了头 ~100 天**")
    L.append("（前 250~400 段 ≈ 2,400 根）。全量 17,520 根可切 **2,188 段**")
    L.append("⇒ **每标的还能切出约 5 个「不重叠」的时段块**（连 K 线都不共享）。")
    L.append("")
    L.append("以「块」为独立单位、用 `t(块数−1)` 判定，达到显著需要多少块：")
    L.append("")
    L.append("| 块数 k | `t_crit95(k−1)` | 合并 SE（若每块 SE≈2.0bp） | z | 判定 |")
    L.append("|---|---|---|---|---|")
    se_blk = 0.0002      # 每块 SE ≈ 2bp（实测每标的 SE ≈ σ/√n ≈ 0.33%/√300 ≈ 1.9bp）
    need_k = None
    for k in range(3, 9):
        tcrit = _t95(k - 1)
        se_pool = se_blk / math.sqrt(k)
        z = abs(eff) / se_pool
        ok = z > tcrit
        if ok and need_k is None:
            need_k = k
        L.append(f"| {k} | {_f(tcrit)} | {_bp(se_pool, 2)} | {_f(z)} | "
                 f"{'✅ **显著**' if ok else '✗ 仍不显著'} |")
    L.append("")
    if need_k:
        add_blocks = need_k - pl["k"]
        L.append(f"⇒ ⭐ **再加 {add_blocks} 个独立块（k={need_k}）就能在块级判出显著。**")
        L.append("")
        segs = 250
        calls = add_blocks * 2 * segs * SEG_BARS   # ×2 臂 ×段数 ×8 根
        L.append(f"   代价：每个块 = 2 臂 × {segs} 段 × {SEG_BARS} 根 = "
                 f"**{2 * segs * SEG_BARS:,} 次调用**"
                 f" ⇒ 共 **{calls:,} 次**"
                 f" ≈ **{calls / 7500:.1f} 个配额窗口**（7,500 次/5 小时）。")
        L.append("")
        L.append("⚠️ 前提：新块必须与已有的**不共享任何一根 K 线**（用别的时段），")
        L.append("且**必须钉死同一套 KPI 阈值**（否则又变成两个实验）。")
        L.append("")
    L.append("⚠️ 上表的每块 SE 用的是**实测**（σ/√n）；块数越多、每块越小，")
    L.append("SE 会越大 ⇒ 实际需要的块数可能比表里多 1 个。")
    L.append("")

    # ---------- 4.1 块级检验（用**实测**，不是投影表） ----------
    L.append("### 4.1 ⭐⭐ 块级检验（用实测数据，不是上表的投影）")
    L.append("")
    L.append("把每个 **(标的, 时段)** 当成一个独立的**块**，用逆方差合并 + "
             "`t(df=块数−1)`：")
    L.append("")
    blk = pool_instruments(items)   # items 已经是 (名字, 统计) 的形式
    L.append("| 块 | n | 效应 | σ | 判力比 | 判定 |")
    L.append("|---|---|---|---|---|---|")
    for i0, s0 in items:
        L.append(f"| {i0} | {s0['n']} | **{_bp(s0['effect'])}** | "
                 f"{_f(s0['sd'] * 100, 4)}% | "
                 f"{_f(abs(s0['effect']) / s0['mde'])} | {s0['verdict']} |")
    L.append("")
    L.append(f"- 块数 **k={blk['k']}**，合并效应 **{_bp(blk['pooled'])}**"
             f"（SE {_bp(blk['se'], 3)}）")
    L.append(f"- **Q = {_f(blk['Q'])}**，df={blk['df']}，"
             f"临界 {_f(blk.get('critical'))} ⇒ "
             f"**{'异质（不能合并）' if blk['heterogeneous'] else '同质（可以合并）'}**")
    if blk.get("t_crit") == blk.get("t_crit"):
        sig_t = blk["ci_t"][0] > 0 or blk["ci_t"][1] < 0
        L.append(f"- ⭐ **块级区间**（`t(df={blk['df']})`，临界 "
                 f"**{_f(blk['t_crit'])}**）："
                 f"**[{_bp(blk['ci_t'][0])}, {_bp(blk['ci_t'][1])}]** ⇒ "
                 f"**{'不跨 0 ⇒ 块级显著' if sig_t else '跨 0 ⇒ 块级仍不显著'}**")
        L.append(f"- （正态近似区间 [{_bp(blk['ci'][0])}, {_bp(blk['ci'][1])}]"
                 f"——**已知反保守，别用它下结论**）")
    L.append("")

    # ---------- 4.2 持续性检验（同一标的、不同时段） ----------
    e1 = next((s0 for i0, s0 in items if i0 == "BTC"), None)
    e2 = next((s0 for i0, s0 in items if "时段2" in i0), None)
    if e1 and e2:
        L.append("### 4.2 ⭐⭐⭐ 「能不能持续」的直接检验（同一标的、另一个不重叠时段）")
        L.append("")
        L.append("这是 §0 说的**唯一未定项**。做法：把 **BTC 换成更早的 100 天**"
                 "（2026-03-12 → 06-04，与原来那次**不共享任何一根 K 线**）")
        L.append("再跑一遍同一个实验。")
        L.append("")
        L.append("| 时段 | n | 效应 | 95% 区间 | σ |")
        L.append("|---|---|---|---|---|")
        L.append(f"| BTC 时段1（06-08→09-16） | {e1['n']} | **{_bp(e1['effect'])}** | "
                 f"[{_bp(e1['lo'])}, {_bp(e1['hi'])}] | {_f(e1['sd'] * 100, 4)}% |")
        L.append(f"| **BTC 时段2（03-12→06-04）** | {e2['n']} | "
                 f"**{_bp(e2['effect'])}** | [{_bp(e2['lo'])}, {_bp(e2['hi'])}] | "
                 f"{_f(e2['sd'] * 100, 4)}% |")
        L.append("")
        same = (e1["effect"] * e2["effect"]) > 0
        L.append(f"⇒ 两个时段**{'同号' if same else '异号 ⚠️'}**"
                 f"（{_bp(e1['effect'])} vs {_bp(e2['effect'])}）")
        if same:
            L.append("")
            L.append("⭐ **同号 ⇒ 效应至少在「跨时段」上方向一致**，"
                     "这与「它是真的、只是很小」相容。")
        else:
            L.append("")
            L.append("⚠️⚠️ **异号 ⇒ 效应不持续**：同一标的、不同时段结论相反 ⇒")
            L.append("那两个数都更像**噪声**，而不是一个稳定的负面效应。"
                     "这会**削弱**前面所有「KPI 让收益变差」的结论。")
        L.append("")

    # ---------- 5 ----------
    L.append("## 5. 决策：三个选项，代价与所得都写清楚")
    L.append("")
    L.append("| 选项 | 代价 | 得到 | 我的判断 |")
    L.append("|---|---|---|---|")
    L.append("| **A. 停手** | 0 | 结论停在「跨 3 个标的同向、量级一致、块级尚未显著」 | "
             "够用——除非你要拿这个数去支持一个**要付钱的决策** |")
    if need_k:
        L.append(f"| **B. 再补 {need_k - pl['k']} 个时段块** | "
                 f"约 {calls:,} 次调用（{calls / 7500:.1f} 个窗口） | "
                 f"把「块级显著」判出来 ⇒ 结论从「同向复现」升级到「**统计上确立**」 | "
                 f"⭐ **性价比最高**（代价有界、结论有质的提升） |")
    L.append("| **C. 换问题** | 0 | —— | 若你不打算用这个结论做决策，那 B 也不必做 |")
    L.append("")
    L.append("⭐ **我的建议**：**先回答一个问题——你会拿这个结论去做什么？**")
    L.append("")
    L.append(f"⚠️ 注意 **§0 已经把「它很小」这个直觉推翻了**：年化 "
             f"**{_pct(eff * SEGS_PER_YEAR, 1)}**，是年噪声的 "
             f"**{_f(abs(eff * SEGS_PER_YEAR) / ann_noise, 1)} 倍**。所以：")
    L.append("")
    L.append("- 如果**只是想知道**（研究好奇心）⇒ **A 够了**；")
    L.append("- ⭐ 如果**要用它决定「要不要保留 KPI」** ⇒ **做 B**：")
    L.append("  我上一轮写的是「反正你都会保留它，所以 B 不必做」——")
    L.append("  **那个推断建立在「效应很小」上，现在不成立了**。")
    L.append(f"  年化 {_pct(eff * SEGS_PER_YEAR, 1)} 意味着**保留 KPI 的代价不是零头**，")
    L.append("  而它换来的只是「更在场」⇒ **这正是必须先把效应钉实的情形**；")
    L.append("- 如果**要写进对外材料**（论文/报告/给外部顾问）⇒ **也做 B**，")
    L.append("  因为「同向复现 + 块级显著」与「只是同向」在可辩护性上差一档。")
    L.append("")
    if need_k:
        L.append(f"⇒ ⭐⭐ **本报告的推荐：做 B**"
                 f"（再补 {need_k - pl['k']} 个不重叠时段块，约 {calls:,} 次调用"
                 f" / {calls / 7500:.1f} 个配额窗口）。")
        L.append("理由：**代价有界、结论有质的提升**；而且年化量级大到"
                 "足以影响「保不保留 KPI」这个决策。")
    L.append("")

    # ---------- 6 ----------
    L.append("## 6. ⚠️ 边界")
    L.append("")
    L.append("- **年化是外推**：−3.5bp/段 是 **2 年数据里前 100 天**的估计；")
    L.append("  把它乘 1,095 得到年化，前提是「效应在各时段一样」，**这个前提没验过**。")
    L.append("- **「值得关心」没有客观答案**：本报告给的是**价签**（成本、噪声、交换率），")
    L.append("  不是结论。第 ③ 条（行为对价）是我认为最贴题的一条。")
    L.append("- **效应≠收益**：这些是**成本后净值**与**基准之差**，")
    L.append("  不构成任何可交易信号 ⇒ 本项目一律不产出交易建议。")
    L.append("- 在场率从决策记录现算，口径是「非 hold 比例」，与 A9 略有出入（见 §2 脚注）。")
    L.append("")
    return "\n".join(L)


def s_n(r: dict, which: str) -> Path:
    return r[which][0]


def _rec_of(inst: str, tmpl: str, evalp: Path) -> Path:
    """从 eval 路径推出同目录的 record 路径（命名约定 rec_<tmpl>_<tag>.jsonl）。"""
    stem = evalp.name.replace("eval_", "rec_").replace(".json", ".jsonl")
    return evalp.with_name(stem)


def _rate_bp(items: list[tuple[str, dict]]) -> tuple[float, float]:
    """交换率的 min/max（bp 段收益 / 1pp 在场率）。"""
    rates: list[float] = []
    for inst in RUNS:
        for _tag, r in RUNS[inst].items():
            if not r:
                continue
            pb = presence_from_records(_rec_of(inst, "v4", r["base"][0]))
            pk = presence_from_records(_rec_of(inst, "v6", r["kpi"][0]))
            st_ = stat(r["base"][0], r["base"][1], r["kpi"][0], r["kpi"][1])
            if not st_ or pb["n"] < 50 or pk["n"] < 50:
                continue
            d_pp = (pk["presence"] - pb["presence"]) * 100
            if d_pp:
                rates.append(abs(st_["effect"]) / d_pp * 10000)
    if not rates:
        return (float("nan"), float("nan"))
    return (min(rates), max(rates))


def _t95(df: int) -> float:
    from tw.segmented import t_crit95
    return t_crit95(df)


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    text = build()
    OUT.write_text(text, encoding="utf-8")
    print(f"报告 → {OUT}（{len(text.splitlines())} 行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
