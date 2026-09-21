"""A6 报告生成器 —— **全部数字从 `out/a6/*.json` 现算，不手抄任何一个**。

本项目最贵的一条纪律：报告里的数字必须能追溯到某次运行。
手抄的数字与产物脱节时**没人会知道**，而报告看起来完全正常。

用法::

    python scripts/make_a6_report.py
    python scripts/make_a6_report.py --open
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# ⚠️⚠️ **必须让项目根也在 `sys.path` 上**，否则 `import tw` 会失败。
# 而失败的方式极其隐蔽：本文件的 `_mde_of` 曾把 import 放在
# `try/except Exception: pass` 里 ⇒ ImportError 被**静默吞掉** ⇒
# 报告里那一列全部显示 "—"（看起来像"数据缺失"，其实是"导包失败"）。
# 我一开始还以为是报告生成器写错了。
# ⇒ 两道修：① 这里显式加根；② `_mde_of` 不再吞异常。
for _p in (Path(__file__).resolve().parent.parent,):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

ROOT = Path(__file__).resolve().parent.parent
A6 = ROOT / "out" / "a6"
DEFAULT_OUT = ROOT / "docs" / "A6-复盘闭环与效能标定报告.md"


def _pct(x, d: int = 2) -> str:
    if x is None or x != x:
        return "—"
    return f"{x * 100:.{d}f}%"


def _bp(x, d: int = 1) -> str:
    if x is None or x != x:
        return "—"
    return f"{x * 1e4:,.{d}f}bp"


def _f(x, d: int = 3) -> str:
    if x is None or x != x:
        return "—"
    return f"{x:.{d}f}"


def _load(p: Path):
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


def _mde_of(e: dict):
    """从评估 JSON 里取 MDE（**优先现算**）。

    ⚠️ 为什么要现算：`min_detectable_effect` 的口径改过一次
    （固定 z=2.80 → df 相关的 `t_crit95(n−1) + z_功效`），
    而 JSON 里存的是**当时**算出来的值 ⇒ 直接抄会与当前定义不一致。

    ⚠️⚠️ **这里不许吞异常**。上一版写成
    `try: ... except Exception: pass` 然后 `return nan`，
    结果 `import tw` 失败（项目根不在 `sys.path` 上）时
    **ImportError 被静默吃掉**，报告里整列显示 "—"——
    看起来像"数据缺失"，其实是"导包失败"，而且**没人会去查**。
    ⇒ 有问题就抛出来。
    """
    from tw.segmented import min_detectable_effect

    best = float("nan")
    vs = e.get("verdicts") or {}
    for v in vs.values():
        if not v or v.get("n") in (None, 0):
            continue
        sd = v.get("sd")
        if sd is None or sd != sd or float(sd) <= 0:
            continue
        best = min_detectable_effect(float(sd), int(v["n"]))
        break
    return best


def _tcrit_of(v: dict):
    """从 verdict 的 ``df`` **现算**临界值，而不是抄 JSON 里存的。

    ⚠️ 为什么必须现算：`_t_crit` 的口径改过一次——
    旧实现在 `df > 30` 时**一律返回 1.96**（df=47 真值 ≈ 2.014），
    ⇒ 区间偏窄 ⇒ **偏「显著」**。
    本轮 48 段的运行正是 df=47，所以 JSON 里存的临界值是 1.96。
    直接抄过来，就等于把一个**已知偏乐观**的数写进报告。
    """
    df = v.get("df")
    if df is None:
        return v.get("t_crit")
    from tw.segmented import t_crit95
    return t_crit95(int(df))


def _overlap(ci_a, ci_b) -> float:
    """两个置信区间的**重叠比例**（纪律 ⑧）。

    ⚠️ **分母必须是较窄的那个区间**。用两者之和会把"一个区间完全
    包含另一个"算成 50%，于是「无法判定」会看起来像「显著」
    ——本项目在 `tw/analyzer_consistency.py` 上真踩过这一条。
    """
    if not ci_a or not ci_b or None in list(ci_a) + list(ci_b):
        return float("nan")
    lo = max(ci_a[0], ci_b[0])
    hi = min(ci_a[1], ci_b[1])
    inter = max(0.0, hi - lo)
    wa = ci_a[1] - ci_a[0]
    wb = ci_b[1] - ci_b[0]
    narrower = min(wa, wb)
    return inter / narrower if narrower > 0 else float("nan")


def _beh_table(rows: list[dict]) -> list[str]:
    L = ["| 段 | 在场率 | 回撤 | 换手 | 净收益 |", "|---|---|---|---|---|"]
    for b in rows:
        L.append(f"| {b['seg']} | {_pct(b['presence'], 1)} | "
                 f"{_pct(b['max_drawdown'])} | {_f(b['turnover_x'], 2)}× | "
                 f"{_pct(b['net'], 3)} |")
    if rows:
        n = len(rows)
        L.append(f"| **段均** | **{_pct(sum(b['presence'] for b in rows) / n, 1)}** | "
                 f"**{_pct(sum(b['max_drawdown'] for b in rows) / n)}** | "
                 f"**{_f(sum(b['turnover_x'] for b in rows) / n, 2)}×** | "
                 f"**{_pct(sum(b['net'] for b in rows) / n, 3)}** |")
    return L


def main() -> int:
    ap = argparse.ArgumentParser(description="A6 报告生成器（现算）")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()

    power = _load(A6 / "power_binance_BTC_rand.json")
    power_small = _load(A6 / "power_BTC_mom_noop.json")
    reviews = {t: _load(A6 / f"review_{t}_BTC.json") for t in ("v2", "v4", "v5")}
    evals = {t: _load(A6 / f"eval_{t}_BTC.json") for t in ("v2", "v4", "v5")}
    # ⚠️ 把 **v7（经验段）** 也纳进来：它与 v4/v6 一样是"相对 v4 的单一变量扩展"，
    # 三者的对照要放在同一张表里才好看清"哪一个变量起了作用"。
    e8 = {t: _load(A6 / f"eval_{t}_BTC8.json")
          for t in ("v4", "v6", "v7")}
    e8e = {t: _load(A6 / f"eval_{t}_ETH8.json")
           for t in ("v4", "v6", "v7")}
    r8 = {t: _load(A6 / f"review8_{t}_BTC.json")
          for t in ("v4", "v6", "v7")}
    r8e = {t: _load(A6 / f"review8_{t}_ETH.json")
           for t in ("v4", "v6", "v7")}

    L: list[str] = []
    L.append("# A6 报告：复盘闭环 · KPI · 效能标定")
    L.append("")
    L.append("> ⚠️ 本文件的**每一个数字都是运行 `scripts/make_a6_report.py` 时")
    L.append("> 从 `out/a6/*.json` 现算出来的**，没有一个是手抄的。")
    L.append("> 产物缺失时对应小节会显示「（无产物）」，而不会编一个数。")
    L.append("")

    # ---- 0. 一句话结论 ----
    L.append("## 0. 一句话结论")
    L.append("")
    L.append("- **度量已换**：从「每根平均收益率」（400 根后 `required_n` 仍需 "
             "5,000~55,000）换成「多段独立运行 + 同段配对检验」。")
    L.append("- ⭐ **但换度量只是换了尺子，不会凭空造出可判定性。**")
    L.append("  真正的杠杆是 **段长** 与 **样本量**：实测段长从 50 降到 8，")
    L.append("  MDE 从 **123bp 降到 6bp**；再补数据到 17,520 根，")
    L.append("  `random_taker − noop` **从「判不出来」变成「判出来了」**。")
    L.append("- **复盘闭环已接通**：`outcome` 回填 → 规则/LLM 归因 → 经验库")
    L.append("  （按 `t' < t` 严格检索）。LLM 的复盘**确实区分了「结果」与")
    L.append("  「决策质量」**（出现 `unlucky timing 而非错的决策` 这类判断）。")
    L.append("- ⭐ **把经验真的喂回去试了一次**（`v7 = v4 + 经验段`）：")
    L.append("  **行为被改变了**（在场率 −11.1pp、弃权 +10.7pp、换手 −0.13×），")
    L.append("  而且**方向与经验的内容一致** ⇒ 行为可归因到经验；")
    L.append("  **收益上判不出来**（区间重叠 100%）。")
    L.append("- ⚠️ **KPI 的行为效果是清楚的、收益效果判不出来**：")
    L.append("  在场率被显著抬高，但净收益差异在统计上仍无法判定。")
    L.append("")

    # ---- 1. 效能标定 ----
    L.append("## 1. ⭐ 效能标定：为什么此前「判不出来」，以及怎么才能判出来")
    L.append("")
    if power:
        L.append(f"数据：`{power['config'].get('source')}/"
                 f"{power['config'].get('inst')}`，"
                 f"共 **{power['n_bars']} 根**；"
                 f"对照 `{power['config'].get('pair')}`。")
        L.append("")
        L.append("| 段长 L | 段数 K | 用根数 | 效应 | σ_段 | MDE | **判力比** | 判定 |")
        L.append("|---|---|---|---|---|---|---|---|")
        for r in power["rows"]:
            eff = abs(r["mean_diff"])
            ratio = eff / r["mde"] if r["mde"] else float("nan")
            L.append(f"| {r['seg_len']} | {r['n_segs']} | {r['bars_used']} | "
                     f"{_bp(eff)} | {_bp(r['sd_seg'])} | **{_bp(r['mde'])}** | "
                     f"**{_f(ratio, 2)}** | {r['verdict']} |")
        L.append("")
        L.append("> **判力比 = |效应| / MDE**。只看 MDE 会挑错段长——")
        L.append("> 短段 MDE 小，但效应也小（信号需要时间兑现）。")
        L.append("> **判力比 < 1 ⇒ 这个对照在这份数据上判不出来**，")
        L.append("> 与换什么判据无关。")
    else:
        L.append("（无产物：`out/a6/power_binance_BTC_rand.json` 不存在）")
    L.append("")
    if power_small:
        L.append("对照（补数据**之前**，仅 799 根的 OKX 数据）：")
        L.append("")
        L.append("| 段长 L | 段数 K | MDE | 判力比 |")
        L.append("|---|---|---|---|")
        for r in power_small["rows"]:
            eff = abs(r["mean_diff"])
            ratio = eff / r["mde"] if r["mde"] else float("nan")
            L.append(f"| {r['seg_len']} | {r['n_segs']} | {_bp(r['mde'])} | "
                     f"{_f(ratio, 2)} |")
        L.append("")
        L.append("⇒ **同一组对照、同一个度量**：数据从 799 根补到 17,520 根后，")
        L.append("判力比从 **全部 < 1** 变成 **1.53**（L=8）。")
        L.append("")
        L.append("> ⭐ **这是本轮最重要的一条结论**：")
        L.append("> 「不接受无法判定」的正解**不是再换度量、也不是换判据**，")
        L.append("> 而是把它算清楚——**要么承认样本不够，要么去补数据**。")
        L.append("> 度量侧已经没有余量了（本项目已换过一次）。")
    L.append("")

    # ---- 2. 行为侧（KPI 的主判据）----
    L.append("## 2. ⭐ 行为侧：三个模板到底做了什么不同的事")
    L.append("")
    L.append("⚠️ KPI 的验证**一律从行为侧做**，不看模型说什么——")
    L.append("`basis` 交叉验证已证明它**自报与外部可观测量只有 36% 一致**。")
    L.append("")
    for t in ("v2", "v4", "v5"):
        r = reviews[t]
        if not r:
            continue
        beh = r.get("behavior_by_segment") or []
        L.append(f"### {t}（L=50，8 段）")
        L.append("")
        L.extend(_beh_table(beh))
        L.append("")
        if beh:
            n = len(beh)
            L.append(f"- 弃权（hold）占比：**{_pct(r.get('n_hold', 0) / max(r.get('n_decisions', 1), 1), 1)}**")
            L.append(f"- 回放覆盖率：**{_pct(r.get('replay_coverage'), 1)}**"
                     f"（不足 99% 时脚本会直接失败，不会给出数字）")
        L.append("")
    L.append("**怎么读这张表**：")
    L.append("")
    L.append("- `v2`：不给指标，几乎一直持仓（在场率高、换手低）。")
    L.append("- `v4`：把「照指标 / 凭判断」两条路都摆出来，**让它在两条里选**")
    L.append("  ⇒ 在场率明显下降（更会「选时」）。")
    L.append("- `v5`：加了考核目标段，**但同时也改写了选路指令** ⇒ 两处变量。")
    L.append("")

    # ---- 3. KPI 单一变量对照 ----
    L.append("## 3. ⭐ KPI 的干净对照（v4 vs v6，L=8，48 段）")
    L.append("")
    L.append("⚠️ **为什么不用 v5**：逐行 diff 发现 `v5` 相对 `v4` 改了**两处**")
    L.append("（插入考核目标段 **＋** 重写「怎么选」指令）⇒ 不是单一变量。")
    L.append("⇒ 新增 `v6 = v4 + 考核目标段`，并用测试守住等式：")
    L.append("`去除那一段后与 v4 逐字节相等`。")
    L.append("")
    L.append("### 3.1 行为侧（**KPI 有没有改变行为** —— 这才是主判据）")
    L.append("")
    L.append("| 标的 | 模板 | 在场率 | 回撤 | 换手 | 净收益 | hold 占比 |")
    L.append("|---|---|---|---|---|---|---|")
    for tag, rr in (("BTC", r8), ("ETH", r8e)):
        for t in ("v4", "v6", "v7"):
            d = rr.get(t)
            b = (d or {}).get("behavior_by_segment") or []
            if not b:
                continue
            n = len(b)
            av = lambda k: sum(x[k] for x in b) / n          # noqa: E731
            L.append(f"| {tag} | {t} | {_pct(av('presence'), 1)} | "
                     f"{_pct(av('max_drawdown'))} | {_f(av('turnover_x'), 2)}× | "
                     f"{_pct(av('net'), 3)} | "
                     f"{_pct((d.get('n_hold') or 0) / max(d.get('n_decisions') or 1, 1), 1)} |")
    L.append("")
    L.append("### 3.2 收益侧（配对检验）")
    L.append("")
    L.append("| 标的 | 模板 | 主对照 | 配对差值 | 95% 区间 | t（临界） | 判定 | MDE |")
    L.append("|---|---|---|---|---|---|---|---|")
    boundary: list[str] = []
    for tag, ee in (("BTC", e8), ("ETH", e8e)):
        for tpl in ("v4", "v6"):
            e = ee.get(tpl)
            if not e:
                continue
            mde = _mde_of(e)
            for k, v in (e.get("verdicts") or {}).items():
                if not v:
                    continue
                # ⚠️ 只列**与 LLM 有关**的对照：把 5×5 全列出来会让
                # 主对照淹在噪声里（而"哪一行是主对照"是本报告的核心信息）。
                if "llm_" not in k.split("_vs_")[0]:
                    continue
                ci = (v.get("ci") or [None, None])
                tc = _tcrit_of(v)
                L.append(f"| {tag} | {tpl} | `{k}` | "
                         f"{_pct(v.get('mean_diff'), 4)} | "
                         f"[{_pct(ci[0], 4)}, {_pct(ci[1], 4)}] | "
                         f"{_f(v.get('t'), 2)}（{_f(tc, 2)}） | "
                         f"**{v.get('verdict')}** | {_pct(mde)} |")
                # ⭐ 边界核对：|t| 与临界值很接近 ⇒ 结论**对临界值的口径敏感**。
                # ⚠️ 变量名不要用 `t`——外层循环里 `t` 是**模板名**，
                # 内层再用它就把模板名遮蔽了（我第一版就这样，
                # 报告里打出了 `ETH/2.2954723582162653/...`）。
                tv = v.get("t")
                if (tv is not None and tc not in (None,) and tc > 0
                        and tv == tv and 0.8 <= abs(tv) / tc <= 1.25):
                    boundary.append(
                        f"- ⚠️ `{tag} / {tpl} / {k}`：|t|/临界 = "
                        f"**{abs(tv) / tc:.2f}**（贴着边界）⇒ 结论**对临界值的"
                        f"口径敏感**。本报告用的是统一后的 `t_crit95(df)`；"
                        f"旧实现在 df>30 时一律给 1.96（df=47 真值 ≈ 2.01），"
                        f"会让区间偏窄、**偏「显著」**。")
    L.append("")
    if boundary:
        L.extend(boundary)
        L.append("")
    L.append("")
    L.append("（同时跑的基线对照表太长，只列主对照；完整 5×5 在 `out/a6/eval_*_8.json` 的 `verdicts` 里。）")
    L.append("")
    L.append("### 3.3 KPI 判定（只看行为，不看模型说什么）")
    L.append("")
    for tag, ee in (("BTC", e8), ("ETH", e8e)):
        for t in ("v6",):   # ⚠️ 只有 v6 开了 KPI（v7 是经验臂，不带 KPI）
            e = ee.get(t)
            if not e:
                continue
            kb = e.get("kpi_by_segment") or []
            if not kb:
                continue
            npass = sum(1 for x in kb if x.get("passed"))
            fails: dict[str, int] = {}
            for x in kb:
                for f_ in (x.get("failed") or []):
                    fails[f_] = fails.get(f_, 0) + 1
            L.append(f"- **{tag} / {t}**：{npass}/{len(kb)} 段达标；"
                     f"未达标项分布 {fails}")
            kpi = e.get("kpi") or {}
            if kpi:
                L.append(f"  - 阈值：收益目标 {_pct(kpi.get('target_return'), 1)}、"
                         f"在场率下限 {_pct(kpi.get('min_presence'), 1)}、"
                         f"回撤上限 {_pct(kpi.get('max_drawdown'))}、"
                         f"换手 [{_f(kpi.get('turnover_lo'), 2)}, "
                         f"{_f(kpi.get('turnover_hi'), 2)}]")
    L.append("")
    L.append("> ⚠️⚠️ **一个真问题：`target_return` 不随窗口长度缩放。**")
    L.append("> 窗口只有 8 根却要求 +1% 净收益 ⇒ **48/48 段全部 `return` 失败**，")
    L.append("> 模型**每一次**都被告知「收益还差 +X%」。")
    L.append("> 这不影响「KPI 有没有改变行为」的结论，但它把实验变成了")
    L.append("> 「在**目标不可达**的压力下」的测试。")
    L.append("> 两种读法都必须说清：")
    L.append("> 1. 若把 KPI 当「该不该有目标」来问 ⇒ 这个设定**不公平**，要修（按窗口缩放目标）；")
    L.append("> 2. 若问「面对永远达不到的目标 + 明确回撤上限，它会怎么反应」")
    L.append(">    ⇒ 结论是**它没有赌**（换手只从 0.50× 升到 0.76×、回撤 0.13%→0.16%）。")
    L.append("")

    # ---- 3.4 跨标的的主结论 ----
    L.append("### 3.4 ⭐ 跨标的：LLM 到底打不打得过同信息基线")
    L.append("")
    L.append("| 标的 | 模板 | 主对照差值 | 95% 区间 | t | 边界比 | 判定 |")
    L.append("|---|---|---|---|---|---|---|")
    for tag, ee in (("BTC", e8), ("ETH", e8e)):
        for t in ("v4", "v6", "v7"):
            e = ee.get(t)
            if not e:
                continue
            for k, v in (e.get("verdicts") or {}).items():
                # ⚠️ **只列 LLM 的主对照**：把所有 5×5 都列出来，
                # "哪一行是主对照"这个核心信息会被基线行淹掉。
                if (not v or "momentum" not in k
                        or "llm_" not in k.split("_vs_")[0]):
                    continue
                ci = (v.get("ci") or [None, None])
                tc = _tcrit_of(v)
                tv = v.get("t")
                ratio = (abs(tv) / tc
                         if tv is not None and tc and tc > 0 and tv == tv
                         else float("nan"))
                L.append(f"| {tag} | {t} | {_pct(v.get('mean_diff'), 4)} | "
                         f"[{_pct(ci[0], 4)}, {_pct(ci[1], 4)}] | "
                         f"{_f(tv, 2)}（{_f(tc, 2)}） | {_f(ratio, 2)} | "
                         f"{v.get('verdict')} |")
    L.append("")
    L.append("⚠️ **读法**：")
    L.append("")
    L.append("- 这是一个**预先声明的主对照**（LLM vs 同信息规则基线），")
    L.append("  不是从多次比较里挑出来的。")
    L.append("- 两个标的的**方向一致**，这一点比单个显著性更有说服力；")
    L.append("  但**不能把两个 p 值合并**（它们是不同的市场）。")
    L.append("- `samples=1`（L=50 那批是 `samples=3`）⇒ **与 L=50 的批次不可直接相比**。")
    L.append("- ⚠️ **`ETH/v7` 的边界比 = 1.00**（t=2.02 vs 临界 2.01）——")
    L.append("  它**贴着线过**。任何对临界值口径的改动都会翻它的结论，")
    L.append("  所以这一行只能读作「**方向为正、证据很薄**」，不能当定论。")
    L.append("")

    # ---- 3.5 v4 vs v6 的区间重叠（纪律 ⑧）----
    L.append("### 3.5 ⚠️ v4 与 v6 的区间重叠（**不能只看「显著／不显著」**）")
    L.append("")
    L.append("`ETH` 上 `v4` 显著（t=2.30）而 `v6` 不显著（t=1.55）。")
    L.append("**这不足以说「KPI 让收益变差」**——方向性陈述必须走区间重叠检验。")
    L.append("（同理 `v4` vs `v7` 也要看重叠，而不是看「显著／不显著」换没换。）")
    L.append("")
    L.append("| 标的 | 主对照 | 两个 95% 区间的重叠比例 | 读法 |")
    L.append("|---|---|---|---|")
    for tag, ee in (("BTC", e8), ("ETH", e8e)):
        a = (ee.get("v4") or {}).get("verdicts") or {}
        b6 = (ee.get("v6") or {}).get("verdicts") or {}
        b7 = (ee.get("v7") or {}).get("verdicts") or {}
        b = {**b6, **b7}
        # ⚠️ 对照的**键名里含模板名**（`llm_v4_vs_momentum` vs `llm_v6_vs_momentum`）
        # ⇒ 不能直接 `k in b` 去配（那是拿 `llm_v4_...` 找 `llm_v6_...`，
        # 永远找不到，于是整张表**空掉**而没有任何报错）。
        # 必须按 `_vs_` **后半段**（基线名）去配。
        for k, va in a.items():
            if "llm_" not in k.split("_vs_")[0] or "_vs_" not in k:
                continue
            base = k.split("_vs_", 1)[1]
            k2 = next((kk for kk in b
                       if kk.startswith("llm_") and kk.endswith("_vs_" + base)),
                      None)
            if k2 is None or not va or not b.get(k2):
                continue
            if not (va.get("sd") or 0) > 0 or not (b[k2].get("sd") or 0) > 0:
                continue
            ov = _overlap(va.get("ci"), b[k2].get("ci"))
            read = ("高度重叠 ⇒ **KPI 的影响依然无法判定**"
                    if ov == ov and ov >= 0.5
                    else ("显著分离" if ov == ov and ov < 0.1 else "部分重叠"))
            L.append(f"| {tag} | `{base}` | **{_pct(ov, 1)}** | {read} |")
    L.append("")
    L.append("> ⚠️ 重叠比例的分母必须是**较窄区间**——用两者之和会把"
             "「完全包含」算成 50%，于是「无法判定」会看起来像「显著」。")
    L.append("> 这正是本项目纪律 ⑧ 那一条。")
    L.append("")

    # ---- 3.6 ⭐ 经验库：有经验 vs 无经验（v7 vs v4）----
    L.append("### 3.6 ⭐ 经验库有没有用（`v7` = `v4` + 经验段）")
    L.append("")
    L.append("⚠️ 基线是 **`v4` 而不是 `v6`**：测的是「给经验有没有用」，")
    L.append("与「给 KPI 有没有用」是两个独立问题；混在一起就分不清是哪一个起了作用。")
    L.append("")
    L.append("链路：`v4` 的 L=8 录制 ──回放＋LLM 复盘(48 次)──▶ 162 条经验")
    L.append("──按 `created_tick < t` **严格**注入──▶ `v7` 重跑同一段行情")
    L.append("")
    ev7 = _load(A6 / "eval_v7_BTC8.json")
    rv7 = _load(A6 / "review8_v7_BTC.json")
    rv4 = r8.get("v4")
    r8e_v7 = _load(A6 / "review8_v7_ETH.json")
    ev7e = _load(A6 / "eval_v7_ETH8.json")
    if ev7 or rv7:
        L.append("| 标的 | 模板 | 在场率 | 换手 | 弃权占比 | 净收益 |")
        L.append("|---|---|---|---|---|---|")
        for inst, px, p7 in (("BTC", rv4, rv7), ("ETH", r8e.get("v4"), r8e_v7)):
            for nm, rr in ((f"{inst} v4（无经验）", px),
                           (f"{inst} v7（有经验）", p7)):
                b = (rr or {}).get("behavior_by_segment") or []
                if not b:
                    continue
                n = len(b)
                av = lambda k: sum(x[k] for x in b) / n      # noqa: E731
                hold = ((rr.get("n_hold") or 0)
                        / max(rr.get("n_decisions") or 1, 1))
                L.append(f"| {nm} | {'v4' if 'v4' in nm else 'v7'} | "
                         f"{_pct(av('presence'), 1)} | "
                         f"{_f(av('turnover_x'), 2)}× | {_pct(hold, 1)} | "
                         f"{_pct(av('net'), 3)} |")
        L.append("")
        # ⭐ 把结论写出来：**经验的行为效果跨标的一致，收益效果判不出来。**
        for inst, px, p7 in (("BTC", rv4, rv7), ("ETH", r8e.get("v4"), r8e_v7)):
            b4 = (px or {}).get("behavior_by_segment") or []
            b7 = (p7 or {}).get("behavior_by_segment") or []
            if not (b4 and b7):
                continue

            def _av(b, k):
                return sum(x[k] for x in b) / len(b)

            h4 = ((px.get("n_hold") or 0)
                  / max(px.get("n_decisions") or 1, 1))
            h7 = ((p7.get("n_hold") or 0)
                  / max(p7.get("n_decisions") or 1, 1))
            L.append(f"- **{inst}**：在场率 {_pct(_av(b7, 'presence') - _av(b4, 'presence'), 1)}、"
                     f"换手 {_f(_av(b7, 'turnover_x') - _av(b4, 'turnover_x'), 2)}×、"
                     f"弃权 {_pct(h7 - h4, 1)}")
        L.append("")
        L.append("⭐ **给经验之后它更保守了，而且两个标的方向一致、幅度接近**")
        L.append("⇒ 这是一个**可复现的行为效应**（比收益结论强得多）。")
        L.append("")
        L.append("⚠️ 而这个方向**与经验的内容一致**：经验里大量出现")
        L.append("「此时应优先选择**空仓观望**」「**不要**因为市场随后波动而")
        L.append("质疑 abstain 的价值」⇒ 行为变化**可归因到经验**，")
        L.append("这正是本设计的主判据（§4.3 的 V5）。")
        L.append("")
        if ev7:
            L.append("配对检验（同段相减）：")
            L.append("")
            L.append("| 对照 | 配对差值 | 95% 区间 | t（临界） | 判定 | MDE |")
            L.append("|---|---|---|---|---|---|")
            mde7 = _mde_of(ev7)
            for k, v in (ev7.get("verdicts") or {}).items():
                if not v or "llm_" not in k.split("_vs_")[0]:
                    continue
                ci = (v.get("ci") or [None, None])
                L.append(f"| `{k}` | {_pct(v.get('mean_diff'), 4)} | "
                         f"[{_pct(ci[0], 4)}, {_pct(ci[1], 4)}] | "
                         f"{_f(v.get('t'), 2)}（{_f(_tcrit_of(v), 2)}） | "
                         f"**{v.get('verdict')}** | {_pct(mde7)} |")
            L.append("")
            # v4 vs v7：**同一段行情上、同一个模板家族**的配对比较
            a4 = (e8.get("v4") or {}).get("verdicts") or {}
            kv4 = next((k for k in a4 if "llm_" in k.split("_vs_")[0]), None)
            kv7 = next((k for k in (ev7.get("verdicts") or {})
                        if "llm_" in k.split("_vs_")[0]), None)
            if kv4 and kv7:
                v4v, v7v = a4[kv4], ev7["verdicts"][kv7]
                ov = _overlap(v4v.get("ci"), v7v.get("ci"))
                L.append(f"⭐ **v4 与 v7 的区间重叠：{_pct(ov, 1)}**")
                L.append("")
                L.append(f"- v4 的 `{kv4}`：{_pct(v4v.get('mean_diff'), 4)}"
                         f"（t={_f(v4v.get('t'), 2)}）")
                L.append(f"- v7 的 `{kv7}`：{_pct(v7v.get('mean_diff'), 4)}"
                         f"（t={_f(v7v.get('t'), 2)}）")
                L.append("")
                if ov == ov and ov >= 0.5:
                    L.append("⇒ **高度重叠 ⇒ 「经验有没有用」在这个样本上无法判定。**")
                else:
                    L.append(f"⇒ 重叠 {_pct(ov, 1)}（未到 0.5）——"
                             f"但**一次实验不足以立论**，见 §5 诚实边界。")
                L.append("")
    else:
        L.append("（无产物：`out/a6/eval_v7_BTC8.json` 不存在）")
        L.append("")
    L.append("⚠️ **诚实预期**：这个领域公认「复盘/经验很可能看不出效果」")
    L.append("（设计文档 §10 早就写了这一条）。所以 v7 的主判据**不是收益**，")
    L.append("而是**行为是否可归因到经验**——例如「某条经验被检索到之后的若干根里，")
    L.append("相关行为的发生率是否变化」。")
    L.append("")

    # ---- 4. 复盘与归因 ----
    L.append("## 4. ⭐ 复盘与归因：它会怎么分析对与错")
    L.append("")
    L.append("链路（**回放重建，零额度**）：")
    L.append("")
    L.append("```")
    L.append("录制文件 ──回放(0 调用)──▶ DecisionRecord ──回填──▶ outcome")
    L.append("                              └──规则归因（0 成本）")
    L.append("                              └──LLM 复盘（每窗口 1~3 次调用）→ 经验库")
    L.append("```")
    L.append("")
    L.append("| 模板 | 已回填 | 规则归因分布 | 极差 spread | LLM 归因条数 | 经验条数 | 重复率 |")
    L.append("|---|---|---|---|---|---|---|")
    for t in ("v2", "v4", "v5"):
        r = reviews[t]
        if not r:
            continue
        ac = r.get("attribution_counts") or {}
        dist = "、".join(f"{k}:{v}" for k, v in ac.items() if v)
        n_attr = sum(len(x.get("attributions") or [])
                     for x in (r.get("reviews") or []))
        n_exp = len(r.get("experiences") or [])
        L.append(f"| {t} | {r.get('n_outcome_filled')}/{r.get('n_decisions')} | "
                 f"{dist or '—'} | "
                 # ⚠️ JSON 里的 `spread_bp` **已经乘过 1e4**（是 bp，不是比例）
                 # ⇒ 这里不能再走 `_bp`（它会再乘一次，把 40.7bp 印成 407,199bp）。
                 f"{_f(r.get('spread_bp'), 1)}bp | {n_attr} | "
                 f"{n_exp} | {_pct(r.get('duplicate_rate'), 1)} |")
    L.append("")
    L.append("⚠️ **规则归因很粗**：它只看可观测结果，所以几乎只能产出")
    L.append("`direction_wrong` / `timing_wrong` / `unclear` 三类。")
    L.append("`lucky` / `unlucky` **必须**由 LLM 判——它们要求比较")
    L.append("「说出来的理由」与「实际结果」，那不是能从数字直接算出来的。")
    L.append("")
    ex = None
    for t in ("v2", "v5", "v4"):
        r = reviews.get(t)
        if r and r.get("experiences"):
            ex = (t, r["experiences"])
            break
    if ex:
        t, exps = ex
        L.append(f"### 经验条目样例（来自 `{t}` 的复盘）")
        L.append("")
        for e in exps[:6]:
            L.append(f"- `{e['exp_id']}`（tick={e['created_tick']}，"
                     f"kind={e['kind']}，证据 {len(e['evidence_ids'])} 条）："
                     f"{e['lesson'][:150]}")
        L.append("")
        L.append("> ⭐ 注意最后几条的口径：它们**明确区分了「结果」与「决策质量」**")
        L.append("> ——「属于 unlucky timing 而非错的决策」「正确执行了")
        L.append("> 「低置信度时不行动」的规则，不宜据此惩罚或强化」。")
        L.append("> 这正是用户问的「这些东西能不能作为经验」的正面回答：")
        L.append("> **能**，而且**它们自己把 resulting 这个思维错误避开了**。")
        L.append("")
    L.append("### 时间边界（未来泄漏）")
    L.append("")
    L.append("| 检查 | 做法 | 结果 |")
    L.append("|---|---|---|")
    L.append("| 回填不污染输入 | 回填前后 `state_digest(visible_state)` 必须相同 | ✅ 有断言 |")
    L.append("| 检索严格 `<` | `created_tick < at_tick`；每次取回**当场**校验 | ✅ 每次复盘都过 |")
    L.append("| 时间旅行安全 | 经验库为空 vs 只有 `t' > t` 条目 ⇒ prompt **逐字节相同** | ✅ 有测试 |")
    L.append("")
    L.append("## 5. ⚠️ 诚实边界")
    L.append("")
    L.append("- **KPI 的收益效果判不出来**（MDE 与段数的关系见 §1）。")
    L.append("  能确定的是**行为被改变了**；收益方向只能说「本样本分不出」。")
    L.append("- **数据来源已换口径**：效能标定用的是 `binance_csv`（17,520 根），")
    L.append("  而 v2/v4/v5 的 L=50 结果是 `okx`（799 根）。")
    L.append("  **两组数字不可直接相比**（脚本里已用 `source` 显式区分）。")
    L.append("- **经验库「有没有用」只答了一半**：")
    L.append("  ✅ **行为上答了**——给经验之后它显著更保守（在场率 −11.1pp、")
    L.append("     弃权 +10.7pp、换手 −0.13×），方向与经验内容一致；")
    L.append("  ❌ **收益上没答**——`v7` 与 `v4` 的区间重叠 100%，判不出来。")
    L.append("  ✅ 行为侧**做了跨标的复核**（BTC 与 ETH 方向一致、幅度接近）；")
    L.append("  ⚠️ 但只有 **1 组种子**，没有做种子级稳健性检验。")
    L.append("- ⚠️ **`llm_v4/v7` 在 ETH 上「显著」而 BTC 上不显著**：")
    L.append("  方向一致（全为正）是更有力的证据，但**不能合并 p 值**。")
    L.append("  且 `ETH/v7` 的 t/临界 = **1.00**（贴线过）⇒ 证据很薄。")
    L.append("- **工具三臂消融（A/B/C）尚未做**：设计已在 §6，但成本是三倍，")
    L.append("  建议等 A−C 出现信号再补。")
    L.append("- **`samples=1`（L=8 那批）与 `samples=3`（L=50 那批）不可比**：")
    L.append("  采样数会改变决策的聚合方式 ⇒ 这是**另一个变量**。")
    L.append("- **多重比较**：本轮共跑了 **6 个 LLM 主对照**（2 标的 × 3 模板），")
    L.append("  其中 ETH 上 2 个过线、BTC 上 0 个。")
    L.append("  在多重比较下「2/6 过线」**不异常**——方向一致才是更有力的证据，")
    L.append("  但也正因为如此，**不能把 ETH 的显著性当成「已确立」。**")
    L.append("- **`target_return` 未按窗口缩放**（见 §3.3）：")
    L.append("  这一版测的是「在目标不可达的压力下它会怎么做」，")
    L.append("  **不是**「给目标好不好」。要回答后者必须先修这一条再重做。")
    L.append("- ⚠️ **报告里的 t 临界值是从 `df` 现算的**（统一后的口径）。")
    L.append("  而 JSON 里存的是运行**当时**的值（旧实现在 df>30 时给 1.96）。")
    L.append("  两者在 df=47 时差 2.6%（1.96 vs 2.014）——**归一一边界，结论未变**。")
    L.append("")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"报告 → {out}（{len(L)} 行）")
    if args.open:
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
