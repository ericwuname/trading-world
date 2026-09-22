#!/usr/bin/env python
"""A9：**把"判不出来"换算成"还差多少样本"**（零额度，纯读产物）。

    python scripts/a9_power_ablation.py        # 打表 + → out/a9/power.json
    python scripts/a9_power_ablation.py --json out/a9/power.json

为什么要有这个脚本
------------------
A6~A8 攒了一批"依然无法判定"。但**"判不出来"不是一个终点，它是一个待换算的量**：

    判力比 = |效应| / MDE        MDE ≈ (t_crit(df) + z_功效) · σ / √n

⭐ **判力比低有两种完全不同的原因**，而这个脚本负责把它们分开：

| 原因 | 特征 | 该怎么办 |
|---|---|---|
| **样本不够** | 效应本身不小，只是 σ/√n 压不住 | **补段数**（唯一杠杆） |
| **效应本来就小** | 要判出它需要天文数字的段数 | **别补**；改问"这个效应重要吗" |

⚠️ 所以本脚本对每个对照都同时给出：
- 现状：效应 / σ / MDE / 判力比
- **"要判出 k×现状效应，各需要多少段"**（k = 1.0 / 1.5 / 2.0）
- 以及**"这个效应值不值得判"**的换算：效应 = 1bp / 5bp 时各需要多少段

⭐ 判据：**报告里写"判不出来"时，必须附带"要多少段才判得出"**。
没有这个数的"判不出来"，读者无法决定下一步——而"下一步"才是报告存在的意义。
"""

from __future__ import annotations

import argparse
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

from tw.segmented import (  # noqa: E402
    min_detectable_effect, power_analysis, t_crit95,
)

ROOT = Path(__file__).resolve().parent.parent
A6, A7, A8 = ROOT / "out" / "a6", ROOT / "out" / "a7", ROOT / "out" / "a8"
OUTD = ROOT / "out" / "a9"

#: 80% 功效对应的正态分位（与 tw.segmented 同源，不要再抄一份）
Z_POWER = 0.8416


def _load(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _nets(p: Path) -> dict[str, list[float]]:
    d = _load(p)
    if not d:
        return {}
    return {k: [float(x) for x in v]
            for k, v in (d.get("result") or {}).get("net") and
            {kk: [row[kk] for row in d["result"]["net"]]
             for kk in d["result"]["net"][0]}.items()}


def _diff(p_a: Path, key_a: str, p_b: Path, key_b: str
          ) -> tuple[list[float], str]:
    na = _nets(p_a).get(key_a) or []
    nb = _nets(p_b).get(key_b) or []
    n = min(len(na), len(nb))
    if n < 2:
        return [], ""
    return [nb[i] - na[i] for i in range(n)], ""


def _segs_for(sd: float, effect: float, *, n_now: int) -> float:
    """判出 ``effect`` 需要多少段（含功效项；与 `min_detectable_effect` 互逆）。"""
    if not (sd == sd) or sd <= 0 or effect == 0 or effect != effect:
        return float("nan")
    res = power_analysis(abs(effect), sd)
    return float(res.get("required_segs") or float("nan"))


# ----------------------------------------------------------------------
# 对照清单：每一行是 (分组, 名字, A 产物, A 键, B 产物, B 键)
# ----------------------------------------------------------------------
def _contrasts() -> list[tuple[str, str, Path, str, Path, str]]:
    rows: list[tuple[str, str, Path, str, Path, str]] = []
    # ---- 三臂消融 ----
    for inst in ("BTC", "ETH"):
        for x, y in (("A", "B"), ("B", "C"), ("A", "C")):
            rows.append((f"消融·{inst}", f"{x}−{y}",
                         A8 / f"eval_{x}_{inst}.json", "llm_v4",
                         A8 / f"eval_{y}_{inst}.json", "llm_v4"))
    # ---- KPI：目标 50% 分位（v6) vs v4 ----
    for inst, p4, p6 in (
        ("BTC", A6 / "eval_v4_BTC8.json", A7 / "eval_v6cal_BTC8.json"),
        ("ETH", A6 / "eval_v4_ETH8.json", A7 / "eval_v6cal_ETH8.json"),
    ):
        rows.append((f"KPI50·{inst}", "v6−v4", p4, "llm_v4", p6, "llm_v6"))
    rows.append(("KPI50·BTC", "v6−v4(o2)",
                 A8 / "eval_v4_o2_BTC.json", "llm_v4",
                 A8 / "eval_v6cal_o2_BTC.json", "llm_v6"))
    rows.append(("KPI50·BTC", "v6−v4(o4)",
                 A7 / "eval_v4_BTC8o4.json", "llm_v4",
                 A7 / "eval_v6_BTC8o4.json", "llm_v6"))
    rows.append(("KPI50·BTC", "v6−v4(o6)",
                 A8 / "eval_v4_o6_BTC.json", "llm_v4",
                 A8 / "eval_v6cal_o6_BTC.json", "llm_v6"))
    # ---- KPI：目标 75% 分位 ----
    for inst, p4, p6 in (
        ("BTC", A6 / "eval_v4_BTC8.json", A8 / "eval_v6q75_BTC.json"),
        ("ETH", A6 / "eval_v4_ETH8.json", A8 / "eval_v6q75_ETH.json"),
        ("BTC", A7 / "eval_v4_BTC8o4.json", A8 / "eval_v6q75_o4_BTC.json"),
        ("ETH", A8 / "eval_v4_o4_ETH.json", A8 / "eval_v6q75_o4_ETH.json"),
    ):
        rows.append((f"KPI75·{inst}", "v6−v4", p4, "llm_v4", p6, "llm_v6"))
    # ---- 经验段 v7 vs v4 ----
    for inst, p4, p7 in (
        ("BTC", A6 / "eval_v4_BTC8.json", A6 / "eval_v7_BTC8.json"),
        ("ETH", A6 / "eval_v4_ETH8.json", A6 / "eval_v7_ETH8.json"),
        ("BTC", A7 / "eval_v4_BTC8o4.json", A7 / "eval_v7_BTC8o4.json"),
        ("ETH", A8 / "eval_v4_o4_ETH.json", A8 / "eval_v8_UNUSED.json"),
    ):
        rows.append((f"经验·{inst}", "v7−v4", p4, "llm_v4", p7, "llm_v7"))
    return rows


#: χ² 的 95% 临界值（df 1~10）。**只用于异质性检验**，所以只需要这一列。
_CHI2_95 = {1: 3.84, 2: 5.99, 3: 7.81, 4: 9.49, 5: 11.07,
            6: 12.59, 7: 14.07, 8: 15.51, 9: 16.92, 10: 18.31}


def heterogeneity(estimates: list[tuple[float, float]]) -> dict[str, Any]:
    """⭐⭐ **段网格到底是"噪声"还是"分层因子"？**

    输入的每一组是 ``(效应, 标准误)``——同一对照在**不同段网格**上的估计。

    直觉：如果段网格只是"换了一批随机样本"，那么各组的估计应当围绕
    **同一个真值**波动，波动幅度由各自的 SE 解释得了。
    若组间波动**显著大于** SE 能解释的范围，那就说明
    **真值本身在组之间就不同** ⇒ 报一个"总体效应"是不合适的。

    判据（DerSimonian–Laird 的 Q）：

        Q = Σ wᵢ (θᵢ − θ̄_w)²,   wᵢ = 1/SEᵢ²

    ``Q > χ²_{0.95}(k−1)`` ⇒ **拒绝"组间同质"** ⇒ 必须按组报告。

    ⭐ 这一条解决了本项目一个反复出现的困扰：
    A7/A8 里"同一对照换个窗口结论就翻"，一直被当成"噪声太大"；
    但**噪声大**与**真值随组变**是两件事——
    前者靠补样本能救，**后者补多少样本都救不了**。
    """
    xs = [(t, se) for t, se in estimates
          if t == t and se == se and se > 0]
    k = len(xs)
    if k < 2:
        return {"k": k, "Q": float("nan"), "df": 0,
                "critical": float("nan"), "heterogeneous": False,
                "note": "少于 2 组，无法检验（**不是「同质」**）"}
    w = [1.0 / (se ** 2) for _t, se in xs]
    sw = sum(w)
    theta_bar = sum(wi * t for wi, (t, _se) in zip(w, xs)) / sw
    q = sum(wi * (t - theta_bar) ** 2 for wi, (t, _se) in zip(w, xs))
    df = k - 1
    crit = _CHI2_95.get(df, float("nan"))
    return {
        "k": k, "Q": q, "df": df, "critical": crit,
        "pooled": theta_bar,
        "theta_bar_se": math.sqrt(1.0 / sw),
        "heterogeneous": bool(crit == crit and q > crit),
        "I2": max(0.0, (q - df) / q) if q > 0 else 0.0,
        "note": ("Q > 临界值 ⇒ **组间异质**：真值随段网格变，"
                 "补样本救不了，必须按组报"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(OUTD / "power.json"))
    args = ap.parse_args()

    results: list[dict[str, Any]] = []
    print("=" * 100)
    print("A9 · 判力标定：每个「判不出来」离「判得出来」还差多少段")
    print("=" * 100)
    print()
    hdr = (f"{'分组':<12}{'对照':<12}{'n':>4}{'效应':>10}{'σ':>10}"
           f"{'MDE':>10}{'判力比':>8}  {'要判出 1.5×效应':>16}  {'要判出 5bp':>12}")
    print(hdr)
    print("-" * 100)
    for grp, name, pa, ka, pb, kb in _contrasts():
        d, _ = _diff(pa, ka, pb, kb)
        if len(d) < 2:
            print(f"{grp:<12}{name:<12}{'—':>4}  （无产物）")
            continue
        n = len(d)
        sd = st.stdev(d)
        m = st.mean(d)
        mde = min_detectable_effect(sd, n)
        ratio = abs(m) / mde if mde == mde and mde > 0 else float("nan")
        need15 = _segs_for(sd, abs(m) * 1.5 if m else float("nan"), n_now=n)
        need5bp = _segs_for(sd, 0.0005, n_now=n)
        print(f"{grp:<12}{name:<12}{n:>4}{m * 100:>9.4f}%{sd * 100:>9.4f}%"
              f"{mde * 100:>9.4f}%{ratio:>8.2f}  "
              f"{('%.0f 段' % need15) if need15 == need15 else '—':>16}  "
              f"{('%.0f 段' % need5bp) if need5bp == need5bp else '—':>12}")
        results.append({
            "group": grp, "contrast": name, "n": n,
            "effect": m, "sd": sd, "mde": mde, "power_ratio": ratio,
            "segs_for_1_5x": need15, "segs_for_5bp": need5bp,
        })

    print()
    print("=" * 100)
    print("⭐ 怎么读")
    print("=" * 100)
    print("""\
· **判力比 ≥ 1.5** ⇒ 有余量；**< 1** ⇒ 这份数据上判不出来（与换判据无关）。
· 「要判出 1.5×效应」= 若效应是真的，要多少段才能以 80% 功效判出来。
· 「要判出 5bp」= 一个**与数据无关的绝对刻度**（5bp 是「值得关心的最小交易优势」的量级）。
  ⚠️ 后者才是决策相关的那个数：**判不出 5bp 的装置，不该用来回答「有没有优势」**。
· ⚠️ 段数与**数据长度**要一起看：n 段 × L 根 ≤ 库里可用根数。
""")
    print()
    print("=" * 100)
    print("⭐⭐ 异质性检验：段网格是「噪声」还是「分层因子」？")
    print("=" * 100)
    print()
    # 同一对照在多个段网格上的估计（效应, SE）
    groups: dict[str, list[tuple[float, float]]] = {}
    off_pairs = [
        ("KPI50·BTC", "0", A6 / "eval_v4_BTC8.json", "llm_v4",
         A7 / "eval_v6cal_BTC8.json", "llm_v6"),
        ("KPI50·BTC", "2", A8 / "eval_v4_o2_BTC.json", "llm_v4",
         A8 / "eval_v6cal_o2_BTC.json", "llm_v6"),
        ("KPI50·BTC", "4", A7 / "eval_v4_BTC8o4.json", "llm_v4",
         A7 / "eval_v6_BTC8o4.json", "llm_v6"),
        ("KPI50·BTC", "6", A8 / "eval_v4_o6_BTC.json", "llm_v4",
         A8 / "eval_v6cal_o6_BTC.json", "llm_v6"),
        ("KPI75·BTC", "0", A6 / "eval_v4_BTC8.json", "llm_v4",
         A8 / "eval_v6q75_BTC.json", "llm_v6"),
        ("KPI75·BTC", "4", A7 / "eval_v4_BTC8o4.json", "llm_v4",
         A8 / "eval_v6q75_o4_BTC.json", "llm_v6"),
        ("KPI75·ETH", "0", A6 / "eval_v4_ETH8.json", "llm_v4",
         A8 / "eval_v6q75_ETH.json", "llm_v6"),
        ("KPI75·ETH", "4", A8 / "eval_v4_o4_ETH.json", "llm_v4",
         A8 / "eval_v6q75_o4_ETH.json", "llm_v6"),
    ]
    for grp, off, pa, ka, pb, kb in off_pairs:
        d, _ = _diff(pa, ka, pb, kb)
        if len(d) < 2:
            continue
        se = st.stdev(d) / math.sqrt(len(d))
        groups.setdefault(grp, []).append((st.mean(d), se))
    het_rows = []
    for grp, xs in groups.items():
        h = heterogeneity(xs)
        het_rows.append({"group": grp, **h,
                         "members": [{"offset": o, "effect": t, "se": se}
                                     for (o, (t, se)) in zip(
                                         [p[1] for p in off_pairs
                                          if p[0] == grp], xs)]})
        mark = "⚠️ **异质**" if h["heterogeneous"] else "同质（波动可用 SE 解释）"
        print(f"  {grp}（k={h['k']} 组）: Q={h['Q']:.2f}, df={h['df']}, "
              f"临界={h['critical']:.2f}, I²={h['I2']:.1%}  → {mark}")
        print(f"      合并估计 {h['pooled'] * 100:+.4f}%"
              f"（SE {h['theta_bar_se'] * 100:.4f}%）"
              f"  ⚠️ **这个 SE 不能用**，见下面那条限制")
    print()
    print("""⚠️⚠️ **一条我自己必须拦住的错误：不能把这 k 组当"独立重复"合并。**

本项目的"多个段网格"（offset 0/2/4/6）是**同一段行情的不同切法**——
它们**高度重叠**（差 2~6 根），**不是独立样本**。
⇒ 上面那个「合并估计」的 SE 是**按独立算的** ⇒ **偏小** ⇒ t 值**虚高**。
⇒ **它只能当参考，不能当结论。**

| 想干什么 | 能不能用多网格合并 |
|---|---|
| 问"组间波动能不能用 SE 解释" | ✅ **能用**（Q 检验是相对比较，不受此影响） |
| 问"总体效应有多大、显著吗" | ❌ **不能用**（重叠 ⇒ 不独立 ⇒ SE 无效） |
| 想真的提高精度 | ✅ **在同一个网格内加段数**（唯一正路） |

⭐ 所以本轮的补样本实验是**在同两个 offset 上把段数从 48 提到 300**，
   而不是"多跑几个 offset 再合并"——后者看起来像"重复实验"，
   其实是**同一个样本被数了很多遍**。
""")
    print("""
⭐ **这一节解决本项目一个反复出现的困扰**：A7/A8 里"同一对照换个窗口结论就翻"，
   一直被当成「噪声太大」。但**噪声大**与**真值随组变**是两件事：

| | 特征 | 补样本能救吗 |
|---|---|---|
| **噪声大** | Q 不显著（波动可用 SE 解释） | ✅ 能 |
| **真值随组变** | **Q 显著** | ❌ **不能**；必须按组报告 |

⇒ 判据：**任何「换一组窗口就变结论」的对照，先做异质性检验再决定补多少样本。**
   没有这一步，补样本只是在**给一个不存在的总体效应买精度**。
""")

    outp = Path(args.json)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps({"rows": results, "heterogeneity": het_rows},
                               ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print(f"JSON → {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
