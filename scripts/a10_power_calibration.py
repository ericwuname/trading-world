#!/usr/bin/env python
"""A10：**判力的实测标定**——不信公式，用真实残差跑 Monte Carlo。

    python scripts/a10_power_calibration.py            # 打表 + → out/a10/power_mc.json
    python scripts/a10_power_calibration.py --reps 4000

为什么要有这个脚本
------------------
A9 用**解析公式** `MDE ≈ (t_crit + z)·σ/√n` 给出「判出 5bp 需要 194~319 段」。
但那句话里有**两个没验过的假设**：

| 假设 | 为什么可疑 |
|---|---|
| **σ 不随 n 变** | A9 实测：σ 从 26.27bp（n=48）涨到 32.95bp（n=300）——**它变了 25%** |
| **残差近似正态** | 段收益是**有重尾**的（加密行情），t 区间可能**标定不准** |

⇒ 本脚本做三件**零额度**的事，把这两条假设**实测掉**：

1. **σ(n) 曲线**：在**同一个网格内**按前缀算 σ，看它是否稳定。
2. **Monte Carlo 判力曲线**：把**真实残差**（中心化后的配对差值）重采样，
   叠加一个**已知真值 δ**，再跑一遍项目的 `paired_verdict`
   ⇒ 得到**实测的**检出率，而不是公式算的。
3. **区间覆盖率**：δ=0 时的"假阳性率"应当 ≈5%；
   区间应当以 ≈95% 的概率盖住真值 δ。
   ⚠️ 偏离 95% 就说明 **t 区间在这份数据上标定不准** —— 那比"判力低"更严重：
   它意味着报告里的"显著"可能只是**区间太窄**。

⚠️ **本脚本测的是「统计装置」，不是「策略」**。
它回答："如果真实效应是 δ，我这套配对检验能不能判出来、判得准不准"。
（δ 是**注入的已知量**；真实策略的效应仍由 A6~A9 的实测负责。）

⚠️ **重采样的前提**：段差值**可交换**（近似 i.i.d.）。
相邻段共享行情路径 ⇒ 可能有正自相关 ⇒ 会让重采样**低估** σ。
所以本脚本同时**报出 lag-k 自相关**，不藏起来。
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
A6, A7, A8, A9 = (ROOT / "out" / x for x in ("a6", "a7", "a8", "a9"))
OUTD = ROOT / "out" / "a10"

#: ⚠️ **固定随机种子**：Monte Carlo 必须可复现，否则报告里的数字无法复核。
DEFAULT_SEED = 20260922


# ----------------------------------------------------------------------
def _load(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _nets(p: Path, key: str) -> list[float]:
    d = _load(p) or {}
    rows = (d.get("result") or {}).get("net") or []
    return [float(r[key]) for r in rows if r.get(key) is not None]


def _diffs(pa: Path, ka: str, pb: Path, kb: str) -> list[float]:
    na, nb = _nets(pa, ka), _nets(pb, kb)
    n = min(len(na), len(nb))
    return [nb[i] - na[i] for i in range(n)]


# ----------------------------------------------------------------------
# ① σ(n) 曲线
# ----------------------------------------------------------------------
def sigma_curve(d: list[float], ks: list[int]) -> list[dict[str, Any]]:
    """在**同一个网格内**取前 k 段算 σ —— 看它随 n 怎么动。

    ⭐ 这一条直接检验 A9 的外推前提。
    若 σ 单调上升（本项目的实测就是），那么
    「MDE ∝ 1/√n」这套外推会**系统性乐观**：
    段数翻倍时 MDE 降得**比 √2 少**。
    """
    out = []
    for k in ks:
        if k < 2 or k > len(d):
            continue
        sub = d[:k]
        sd = st.stdev(sub)
        mde = min_detectable_effect(sd, k)
        out.append({"n": k, "sd": sd, "mean": st.mean(sub), "mde": mde})
    return out


def autocorr(d: list[float], lags: int = 5) -> list[float]:
    """lag-k 自相关。⚠️ 重采样法要求近似 i.i.d.；正自相关会**低估** σ。"""
    n = len(d)
    m = st.mean(d)
    den = sum((x - m) ** 2 for x in d)
    if den == 0:
        return [float("nan")] * lags
    out = []
    for k in range(1, lags + 1):
        num = sum((d[i] - m) * (d[i + k] - m) for i in range(n - k))
        out.append(num / den)
    return out


# ----------------------------------------------------------------------
# ② Monte Carlo：重采样真实残差 + 已知真值 δ
# ----------------------------------------------------------------------
def _paired_verdict_fast(diffs: list[float]) -> tuple[float, float, float]:
    """只用 `paired_verdict` 的**核心三个数**（点估计 / 区间 / t）。

    ⚠️ 仍需与 `paired_verdict` **同源**（纪律：一个量只允许一份实现）。
    所以这里**直接调它**，只取需要的字段——不做第二份口径。
    """
    v = paired_verdict(diffs, name_a="处理", name_b="基准")
    lo, hi = v["ci"]
    return v["mean_diff"], lo, hi


def monte_carlo(resid: list[float], *, deltas: list[float], ns: list[int],
                reps: int, seed: int = DEFAULT_SEED
                ) -> dict[tuple[float, int], dict[str, Any]]:
    """对每个 (δ, n) 做 ``reps`` 次实验，报检出率 / 覆盖率 / 偏差。

    - **检出率（power）**：区间**不含 0** 的比例。δ=0 时它应 ≈5%（假阳性率）。
    - **覆盖率（coverage）**：区间**含真值 δ** 的比例。应 ≈95%。
      ⚠️ 若明显 < 95% ⇒ 区间**太窄** ⇒ 报告里的"显著"被高估。
      ⚠️ 若明显 > 95% ⇒ 区间**太宽** ⇒ "判不出来"被高估（更保守，但同样不对）。
    - **偏差**：点估计的均值 − δ。配对检验应当**无偏**。

    ⚠️ 用 `random.Random(seed)` 而不是全局 `random`：
    全局种子会被别的调用改掉，导致**同一个脚本两次跑出不同数**。
    """
    import random
    rng = random.Random(seed)
    m = st.mean(resid)
    centered = [x - m for x in resid]      # 中心化：只留"形状"，δ 才是唯一的真值
    L = len(centered)
    out: dict[tuple[float, int], dict[str, Any]] = {}
    for delta in deltas:
        for n in ns:
            hit = 0
            cov = 0
            est: list[float] = []
            for _ in range(reps):
                d = [centered[rng.randrange(L)] + delta for _ in range(n)]
                mm, lo, hi = _paired_verdict_fast(d)
                est.append(mm)
                if lo > 0 or hi < 0:
                    hit += 1
                if lo <= delta <= hi:
                    cov += 1
            out[(delta, n)] = {
                "delta": delta, "n": n, "reps": reps,
                "power": hit / reps, "coverage": cov / reps,
                "bias": st.mean(est) - delta,
                "se_of_mean": st.stdev(est) / math.sqrt(reps) if reps > 1 else float("nan"),
            }
    return out


# ----------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--json", default=str(OUTD / "power_mc.json"))
    args = ap.parse_args()

    # 残差来源：取 A9 里 n=300 的那两次运行（段最多、最接近"真实分布"）
    sources = [
        ("KPI50·BTC offset=0（n=300）",
         A9 / "eval_v4_k300o0_BTC.json", "llm_v4",
         A9 / "eval_v6_k300o0_BTC.json", "llm_v6"),
        ("KPI50·BTC offset=4（n=300）",
         A9 / "eval_v4_k300o4_BTC.json", "llm_v4",
         A9 / "eval_v6_k300o4_BTC.json", "llm_v6"),
        ("三臂 A−B·BTC（n=48）",
         A8 / "eval_A_BTC.json", "llm_v4",
         A8 / "eval_B_BTC.json", "llm_v4"),
    ]
    deltas = [0.0, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005]
    ns = [24, 48, 100, 200, 300, 600, 1000]
    results: dict[str, Any] = {}

    print("=" * 100)
    print("A10 · 判力实测标定（真实残差 Monte Carlo，**零额度**）")
    print("=" * 100)

    for name, pa, ka, pb, kb in sources:
        d = _diffs(pa, ka, pb, kb)
        if len(d) < 8:
            print(f"\n【{name}】（无产物）")
            continue
        print(f"\n【{name}】n={len(d)}")

        # ---- σ(n) 曲线 ----
        # ⚠️ 用 `sorted(set(...))` 去重：我第一版是 `ks + [len(d)]`，
        # 而 len(d) 往往已经在 ks 里 ⇒ 表里出现**两行一模一样的 n**
        # （看起来像"两个不同的点给了同一个数"，其实只是重复打印）。
        ks = sorted({k for k in (24, 48, 100, 200, 300, 600, 1000)
                     if k <= len(d)} | {len(d)})
        curve = sigma_curve(d, ks)
        print("  σ(n) 曲线（同一网格内的前缀）：")
        for r in curve:
            print(f"     n={r['n']:>4}  σ={r['sd'] * 100:>7.4f}%  "
                  f"MDE={r['mde'] * 100:>7.4f}%  效应={r['mean'] * 100:+.4f}%")
        if len(curve) >= 2 and curve[0]["sd"] > 0:
            grow = curve[-1]["sd"] / curve[0]["sd"]
            print(f"  ⇒ σ 从 n={curve[0]['n']} 到 n={curve[-1]['n']} "
                  f"变成 **{grow:.2f}×**")
            print(f"     ⚠️ 据此外推：段数翻倍时 MDE 降得比 √2 少"
                  f"（实测降 {curve[0]['mde'] / curve[-1]['mde']:.2f}×）")

        # ---- 自相关（重采样前提）----
        ac = autocorr(d, 5)
        print("  lag-k 自相关：" + "  ".join(f"{x:+.3f}" for x in ac))
        worst = max(abs(x) for x in ac if x == x) if any(x == x for x in ac) else float("nan")
        if worst == worst and worst > 2 / math.sqrt(len(d)):
            print(f"     ⚠️ 最大 |自相关| = {worst:.3f} 超过 2/√n = "
                  f"{2 / math.sqrt(len(d)):.3f} ⇒ 段间**不独立**，"
                  f"重采样可能**低估** σ")

        # ---- Monte Carlo ----
        mc = monte_carlo(d, deltas=deltas, ns=[n for n in ns if n <= max(len(d) * 4, 300)],
                         reps=args.reps, seed=args.seed)
        print(f"\n  实测判力（重采样真实残差，{args.reps} 次/格）：")
        hdr = "     n \\  δ  " + "".join(f"{dl * 100:>8.2f}bp" for dl in deltas)
        print(hdr)
        for n in sorted({k[1] for k in mc}):
            row = f"     {n:>5}  "
            for dl in deltas:
                cell = mc.get((dl, n))
                row += f"{cell['power']:>9.1%}" if cell else f"{'—':>10}"
            print(row)
        print("     （δ=0 那一列 = **假阳性率**，应当 ≈5%）")
        cov0 = [mc[(0.0, n)]["coverage"] for n in sorted({k[1] for k in mc})]
        print(f"  δ=0 的覆盖率：{'  '.join(f'{c:.1%}' for c in cov0)}")
        print(f"  ⇒ 假阳性率：{'  '.join(f'{1 - c:.1%}' for c in cov0)}"
              f"   （应 ≈5%；明显偏高 ⇒ t 区间**太窄**）")

        results[name] = {
            "n": len(d), "sigma_curve": curve, "autocorr": ac,
            "mc": {f"d{k[0]}_n{k[1]}": v for k, v in mc.items()},
        }

    outp = Path(args.json)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps({"reps": args.reps, "seed": args.seed,
                                "sources": results},
                               ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nJSON → {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
