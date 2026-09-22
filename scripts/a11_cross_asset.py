#!/usr/bin/env python
"""A11：**跨标的复现**——把 KPI 效应在第二个独立的资产上重做一遍。

    python scripts/a11_cross_asset.py          # 打表 + → out/a11/cross_asset.json
    python scripts/make_a11_report.py           # → docs/A11-跨标的复现报告.md

为什么做这个（而不是继续在 BTC 上加段数）
----------------------------------------
A10 实测出：判出 5bp 要 **380~510 段**。而 offset=0 的实测效应是 **3.5bp**
⇒ 80% 判力要约 **770 段**（12,320 次调用）——**超过一个配额窗口（7,500/5h）**。

⭐ A9 只禁了"合并多个**段网格**"（它们是同一段行情的不同切法、**重叠、不独立**）。
**不同标的（BTC / ETH / SOL）是独立的** ⇒ 跨标的**可以合并**，而且这是
**交叉复现**——比在同一条行情上堆段数**更有说服力**（能挡住"这段行情特殊"）。

⚠️⚠️ 但跨标的比较有一个**致命前提**（A9b 那次踩过的同一个坑）：
`--calibrate-kpi` 会让每个标的用**它自己标定的阈值** ⇒ 模型收到的
**指令本身就不同** ⇒ 那测的就不是"同一个实验在不同资产上"，而是
"两个不同的实验"。⇒ 本轮把 ETH 的阈值**钉死成 BTC 那次印的值**，
并且**把这个前提写成会失败的断言**（本文件的 `check_same_instructions`）。
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

from tw.segmented import min_detectable_effect, paired_verdict  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
A9, A11 = ROOT / "out" / "a9", ROOT / "out" / "a11"
OUTD = ROOT / "out" / "a11"

#: 跨标的要用的 (标的, 处理臂产物, 处理键, 基准臂产物, 基准键)
RUNS: dict[str, dict[str, Any]] = {
    "BTC": {
        "k400": None,                     # BTC 没有 400 段那次（只有 300）
        "k300": {
            "kpi": (A9 / "eval_v6_k300o0_BTC.json", "llm_v6"),
            "base": (A9 / "eval_v4_k300o0_BTC.json", "llm_v4"),
        },
    },
    "ETH": {
        "k400": {
            "kpi": (A11 / "eval_v6_k400o0_ETH.json", "llm_v6"),
            "base": (A11 / "eval_v4_k400o0_ETH.json", "llm_v4"),
        },
    },
}


def _load(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _nets(p: Path, key: str) -> list[float]:
    d = _load(p) or {}
    rows = (d.get("result") or {}).get("net") or []
    return [float(r[key]) for r in rows if r.get(key) is not None]


def _ranges(p: Path) -> list[Any]:
    d = _load(p) or {}
    return (d.get("result") or {}).get("ranges") or []


# ⭐⭐⭐ 前提检查：两个标的必须**拿到同一份指令** -----
def check_same_instructions(pa: Path, pb: Path) -> dict[str, Any]:
    """两个运行的 KPI 阈值必须**逐字段完全相同**。

    ⚠️ 这不是"顺手看一眼"，而是整个跨标的比较**成立的前提**：
    KPI 阈值印在提示词里 ⇒ 阈值不同就是**不同的实验**。
    A9b 就是因为没守住这条，把一个跨运行的差异当成了"补样本的效果"。
    """
    da, db = _load(pa), _load(pb)
    ka = (da or {}).get("kpi") or {}
    kb = (db or {}).get("kpi") or {}
    # ⭐⭐ **必须区分"没有数据"与"前提被违反"**：
    #   - 产物不存在 ⇒ 状态 `missing` ⇒ 那是"**待跑**"，不是"不许比"；
    #   - 产物在但阈值不同 ⇒ 状态 `different` ⇒ **这才是前提被违反**。
    # ⚠️ 我第一版把两者都报成"阈值不同"——读者会以为实验设计错了，
    #   而真实情况只是"那一臂还没跑"。
    if da is None or db is None:
        return {"status": "missing", "same": False,
                "missing": [str(x) for x, d in ((pa, da), (pb, db))
                            if d is None],
                "fields": [], "diffs": [], "kpi_a": ka, "kpi_b": kb}
    if not ka or not kb:
        return {"status": "missing", "same": False,
                "missing": [str(x) for x, k in ((pa, ka), (pb, kb)) if not k],
                "fields": [], "diffs": [], "kpi_a": ka, "kpi_b": kb}
    keys = sorted(set(ka) | set(kb))
    diffs = []
    for k in keys:
        va, vb = ka.get(k), kb.get(k)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            if abs(float(va) - float(vb)) > 1e-12:
                diffs.append((k, va, vb))
        elif va != vb:
            diffs.append((k, va, vb))
    return {"status": "same" if not diffs else "different",
            "same": not diffs, "missing": [],
            "fields": keys, "diffs": diffs, "kpi_a": ka, "kpi_b": kb}


def check_base_arm_matches(pa: Path, pb: Path) -> dict[str, Any]:
    """两臂的**基准**必须同名同配置（否则对比的不是同一件事）。"""
    ca = (_load(pa) or {}).get("config") or {}
    cb = (_load(pb) or {}).get("config") or {}
    keys = ("seg_len", "seg_offset", "n_segs", "samples", "template",
            "min_history", "features_mode")
    diffs = []
    for k in keys:
        if k in ca or k in cb:
            if ca.get(k) != cb.get(k):
                diffs.append((k, ca.get(k), cb.get(k)))
    return {"same": not diffs, "diffs": diffs}


# ----------------------------------------------------------------------
def stat(pa: Path, ka: str, pb: Path, kb: str) -> dict[str, Any] | None:
    """配对差值 = **``pb`` 臂 − ``pa`` 臂**。

    ⚠️ 调用方必须按「`pa` = 基准、`pb` = 处理」传，才能得到
    `d = 处理 − 基准`（负 = 处理让收益变差）——**与 A9/A10 同一约定**。
    """
    na, nb = _nets(pa, ka), _nets(pb, kb)
    n = min(len(na), len(nb))
    if n < 2:
        return None
    d = [nb[i] - na[i] for i in range(n)]
    v = paired_verdict(d, name_a="处理", name_b="基准")
    sd = st.stdev(d)
    mde = min_detectable_effect(sd, n)
    return {"n": n, "effect": st.mean(d), "sd": sd,
            "se": sd / math.sqrt(n), "mde": mde,
            "lo": v["ci"][0], "hi": v["ci"][1], "t": v["t"],
            "verdict": v["verdict"], "diffs": d}


def pool_instruments(items: list[tuple[str, dict[str, Any]]]
                     ) -> dict[str, Any]:
    """**逆方差合并各标的**（合法：不同资产相互独立）。

    ⭐ 判据：`wᵢ = 1/SEᵢ²`；合并估计 `θ̄ = Σwᵢθᵢ/Σwᵢ`。
    ⚠️ 同时给 **Q 检验**：若各标的效应**不一致**（Q 显著），
    那么"合并"这件事本身就不能报——只能按标的分别报。
    （与 A9 的做法同源；区别是这里各组**确实独立**，所以合并合法。）
    """
    xs = [(nm, s) for nm, s in items if s and s["se"] > 0]
    if len(xs) < 2:
        return {"k": len(xs), "pooled": float("nan"),
                "se": float("nan"), "Q": float("nan"), "df": 0,
                "heterogeneous": False,
                "note": "少于 2 个标的，不构成跨标的复现"}
    w = [1.0 / (s["se"] ** 2) for _nm, s in xs]
    sw = sum(w)
    th = sum(wi * s["effect"] for wi, (_nm, s) in zip(w, xs)) / sw
    q = sum(wi * (s["effect"] - th) ** 2 for wi, (_nm, s) in zip(w, xs))
    df = len(xs) - 1
    crit = {1: 3.84, 2: 5.99, 3: 7.81, 4: 9.49}.get(df, float("nan"))
    se = math.sqrt(1.0 / sw)
    return {"k": len(xs), "pooled": th, "se": se, "Q": q, "df": df,
            "critical": crit, "heterogeneous": bool(crit == crit and q > crit),
            "z": th / se if se > 0 else float("nan"),
            "ci": (th - 1.96 * se, th + 1.96 * se),
            "note": ("Q > 临界 ⇒ **各标的效应不一致** ⇒ 不能合并，要按标的报"
                     if (crit == crit and q > crit) else
                     "未拒绝同质 ⇒ 可以合并")}


# ⭐⭐⭐ 预检：**同一套阈值在不同标的上"咬合力"不同**
def kpi_bite(nets: list[float], target_return: float) -> dict[str, Any]:
    """量化「收益目标」这一条的**通过率**——即这个 KPI 有多大约束力。

    ⭐ 为什么要算这个**在花钱之前**：
    本轮的阈值是**按 BTC 基线的分位数**标定的（`target_return` = BTC 基线
    净收益的**中位数**）⇒ 在 BTC 上它**按构造**大约卡掉一半段；
    但同一个数字放到 **ETH** 上，就不再是 ETH 的中位数了。
    ⇒ 若 ETH 的基线本来就大量达标，那么"给不给 KPI"在 ETH 上**几乎没差别**
      ⇒ 这次复现**缺乏约束力**，跑了也说明不了问题。

    ⚠️ 这里**只量化"收益目标"这一条**（其余三条要从逐 tick 的权益路径算，
    不在 `net` 里）。至少把最容易失效的那条先量出来，并**写明只量了它**。
    """
    if len(nets) < 2:
        return {"n": 0, "pass_rate": float("nan"), "median": float("nan")}
    ok = sum(1 for x in nets if x >= target_return)
    return {"n": len(nets), "pass_rate": ok / len(nets),
            "median": st.median(nets),
            "target": target_return,
            "slack": st.median(nets) - target_return}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(OUTD / "cross_asset.json"))
    args = ap.parse_args()

    print("=" * 100)
    print("A11 · 跨标的复现：KPI 效应在第二个资产上重做一遍")
    print("=" * 100)

    # ---- 前提检查（先做，失败就不许继续下结论）----
    btc_kpi = RUNS["BTC"]["k300"]["kpi"][0]
    print("\n【前提检查 1】两个运行的 KPI 阈值必须逐字段相同")
    items: list[tuple[str, dict[str, Any]]] = []
    premise_ok = True
    for inst in ("BTC", "ETH"):
        for tag, r in RUNS[inst].items():
            if not r:
                print(f"  {inst} {tag}: （本轮未跑）")
                continue
            pa, ka = r["kpi"]
            pb, kb = r["base"]
            chk = check_same_instructions(btc_kpi, pa)
            if chk["status"] == "missing":
                print(f"  {inst} {tag} vs BTC 基准: ⏸️ **待跑**"
                      f"（产物不存在：{chk['missing']}）")
            elif chk["status"] == "different":
                print(f"  {inst} {tag} vs BTC 基准: ❌ **阈值不同**"
                      f"（前提被违反）")
                premise_ok = False
            else:
                print(f"  {inst} {tag} vs BTC 基准: ✅ 阈值相同")
            for fld, va, vb in chk["diffs"]:
                print(f"      ⚠️ {fld}: BTC={va!r}  本运行={vb!r}")
            # ⚠️ **符号约定必须与 A9/A10 一致**：`d = 处理 − 基准`
            # （负 = KPI 让收益变差）。我第一版传反了 ⇒ 报告里正负会误导。
            s = stat(pb, kb, pa, ka)
            if s:
                items.append((f"{inst}(n={s['n']})", s))
                print(f"      效应 {s['effect'] * 100:+.4f}%  "
                      f"区间 [{s['lo'] * 100:+.4f}%, {s['hi'] * 100:+.4f}%]  "
                      f"t={s['t']:+.2f}  σ={s['sd'] * 100:.4f}%  "
                      f"判力比 {abs(s['effect']) / s['mde']:.2f}  "
                      f"→ {s['verdict']}")
    if not premise_ok:
        print("\n❌ **前提不成立 ⇒ 不许把两个标的合并/对比。**")

    print("\n【前提检查 2】段网格是否一致（同 offset ⇒ 同一组切法）")
    for inst in ("BTC", "ETH"):
        for tag, r in RUNS[inst].items():
            if not r:
                continue
            rg = _ranges(r["kpi"][0])
            print(f"  {inst} {tag}: {len(rg)} 段，前 3 {rg[:3]}，末 1 {rg[-1:]}")

    # ---- ⭐ 预检：同一套阈值在各标的上的"咬合力" ----
    print("\n【预检】同样的 `target_return` 在不同标的上有多少约束力")
    kpi_thr = ((_load(btc_kpi) or {}).get("kpi") or {}).get("target_return")
    results_pilot: dict[str, Any] = {}
    if kpi_thr is None:
        print("  （读不到 BTC 的 target_return，跳过）")
    else:
        print(f"  钉死的 target_return = {kpi_thr * 100:+.4f}%")
        print(f"  {'标的':<10}{'n':>5}{'基线中位净收益':>16}{'达标率':>10}{'余量':>12}")
        bites = {}
        for inst, spec in RUNS.items():
            for tag, r in (spec or {}).items():
                if not r:
                    continue
                nets = _nets(*r["base"])
                if len(nets) < 2:
                    continue
                b = kpi_bite(nets, kpi_thr)
                bites[f"{inst} {tag}"] = b
                print(f"  {inst + ' ' + tag:<10}{b['n']:>5}"
                      f"{b['median'] * 100:>15.4f}%{b['pass_rate']:>10.1%}"
                      f"{b['slack'] * 100:>11.4f}%")
        if len(bites) >= 2:
            rates = [b["pass_rate"] for b in bites.values()]
            print(f"  ⇒ 达标率在 {min(rates):.1%} ~ {max(rates):.1%} 之间"
                  f"（差 {abs(max(rates) - min(rates)) * 100:.1f} 个百分点）")
            if max(rates) > 0.85:
                print("  ⚠️ **有标的的基线本来就大量达标 ⇒ KPI 在那上面几乎无约束力**；"
                      "\n     ⇒ 若目标标的就是它，这次复现**缺乏约束力**，"
                      "跑出来也说明不了「KPI 有没有用」。")
        results_pilot = bites

    # ---- 跨标的合并 ----
    print("\n" + "=" * 100)
    print("跨标的合并（**合法**：不同资产相互独立）")
    print("=" * 100)
    pl = pool_instruments(items)
    if pl["k"] >= 2:
        print(f"  标的数 k={pl['k']}  合并效应 {pl['pooled'] * 100:+.4f}%"
              f"（SE {pl['se'] * 100:.4f}%）")
        print(f"  95% 区间 [{pl['ci'][0] * 100:+.4f}%, {pl['ci'][1] * 100:+.4f}%]"
              f"  z={pl['z']:+.2f}")
        print(f"  Q={pl['Q']:.2f}, df={pl['df']}, 临界={pl['critical']:.2f}"
              f"  → {'⚠️ **异质**（不能合并）' if pl['heterogeneous'] else '同质（可合并）'}")
        # ⚠️ 合并后的 MDE：用**合并 SE** 反推等效 σ，再按总段数算。
        n_tot = sum(s["n"] for _nm, s in items)
        sd_eff = pl["se"] * math.sqrt(len(items))   # = σ/√n 的反推
        print(f"  ⇒ 等效 σ={sd_eff * 100:.4f}%，总段数（各标的之和）"
              f"={n_tot}，合并 MDE="
              f"{min_detectable_effect(sd_eff, n_tot) * 100:.4f}%")
    else:
        print(f"  {pl['note']}")

    outp = Path(args.json)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(
        {"premise_ok": premise_ok,
         "per_instrument": [{"name": nm, **{k: v for k, v in s.items()
                                           if k != "diffs"}} for nm, s in items],
         "pooled": pl, "kpi_bite": results_pilot},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nJSON → {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
