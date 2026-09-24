#!/usr/bin/env python
"""独立重跑《右尾策略完全手册》的全部可计算声明 → out/a15/right_tail.json

    python scripts/verify_right_tail_doc.py

文档声明 → 验法：
  §1/§2  E(N,p) = (2p)^N − 1 与 §2.2 数值表      → 纯算术（逐格核对）
  §3     赌徒破产 P×终值 = W₀（1% 天花板）        → 蒙特卡洛（玩法 A）+ B 解析
  §4     三玩法成功率 A/B/C @ p=0.50/0.48/0.45    → 蒙特卡洛 / 解析
  §5     收割比例 f=0/0.5/1 @ p=0.48, N=10        → 蒙特卡洛
  §6     连赢分布 P(连赢k)=0.5^(k+1)；最长连赢≈log₂n → 蒙特卡洛（10 万次下注）
  §7     分阶段止盈的关卡衰减表                    → 解析（串独立 ⇒ 精确）
固定种子，可复现。对照报告见 make_a15_right_tail_verify.py。

⭐ 记账模型（文档 §2.1 的串定义）：
  每串下注 1 元，赢则筹码翻倍（用赢利翻，不追加本金），失败则输掉这 1 元，
  连赢 N 次收手（拿走 2^N，净赚 2^N−1）。
  收割比例 f：每次赢后收割 f×(累计利润)，剩余继续翻（f=1 ⇒ 永远只押 1 元）。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUTD = ROOT / "out" / "a15"
SEED = 20260924


# ----------------------------------------------------------------------
# §1/§2 闭式解
# ----------------------------------------------------------------------
def E_closed(N: int, p: float) -> float:
    """E = p^N(2^N − 1) − (1 − p^N) = (2p)^N − 1。"""
    return (2.0 * p) ** N - 1.0


def check_closed_form() -> dict[str, Any]:
    """§2.2 表逐格核对（文档给 3 位小数）。"""
    Ns = [1, 5, 10, 20, 30]
    ps = [0.45, 0.48, 0.50, 0.52, 0.55]
    doc = {
        1:  [-0.100, -0.040, 0.000, +0.040, +0.100],
        5:  [-0.410, -0.185, 0.000, +0.217, +0.611],
        10: [-0.651, -0.335, 0.000, +0.480, +1.594],
        20: [-0.878, -0.558, 0.000, +1.191, +5.727],
        30: [-0.958, -0.706, 0.000, +2.243, +16.449],
    }
    rows, bad = [], []
    for N in Ns:
        row = {"N": N}
        for j, p in enumerate(ps):
            e = E_closed(N, p)
            row[f"p={p}"] = round(e, 4)
            if abs(e - doc[N][j]) >= 5e-4:
                bad.append({"N": N, "p": p, "ours": round(e, 4),
                            "doc": doc[N][j]})
        rows.append(row)
    return {"rows": rows, "mismatches": bad, "all_match": not bad}


# ----------------------------------------------------------------------
# 向量化最长连赢
# ----------------------------------------------------------------------
def max_run_rows(bits: np.ndarray) -> np.ndarray:
    """bits: (m, L) bool ⇒ 每行的最长连续 True 段（列间向量化）。"""
    m = bits.shape[0]
    cur = np.zeros(m, dtype=np.int64)
    best = np.zeros(m, dtype=np.int64)
    for col in bits.T:
        cur = np.where(col, cur + 1, 0)
        np.maximum(best, cur, out=best)
    return best


# ----------------------------------------------------------------------
# §6 连赢分布与最长连赢
# ----------------------------------------------------------------------
def streak_distribution(n_flips: int = 100_000, p: float = 0.5,
                        seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    flips = rng.random(n_flips) < p
    flips = np.append(flips, False)          # 收口最后一段
    counts: dict[int, int] = {}
    run = 0
    for w in flips:
        if w:
            run += 1
        else:
            if run:
                counts[run] = counts.get(run, 0) + 1
            run = 0
    sim = {k: counts.get(k, 0) for k in range(1, 15)}
    theory = {k: round(0.5 ** (k + 1) * n_flips) for k in range(1, 15)}
    return {"n_flips": n_flips, "sim": sim, "theory": theory}


def max_streak_vs_log2(seed: int = 0) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed + 1)
    out = []
    for n, doc_avg, doc_p90 in ((100, 6.0, 8), (1000, 9.4, 11),
                                (10000, 13.4, 15), (100000, 16.7, 18)):
        trials = max(300, min(3000, 3_000_000 // n))
        flips = rng.random((trials, n)) < 0.5
        best = max_run_rows(flips)
        out.append({"n": n, "trials": trials,
                    "sim_avg": float(best.mean()),
                    "sim_p90": float(np.percentile(best, 90)),
                    "doc_avg": doc_avg, "doc_p90": doc_p90,
                    "log2n": round(math.log2(n), 2)})
    return out


# ----------------------------------------------------------------------
# §4 玩法 A（串铺满翻币流 ⇒ 成功 = 最长连赢 ≥ N）
# ----------------------------------------------------------------------
def play_A(n_accounts: int, p: float, N: int = 20, budget: int = 10_000,
           seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    # 串 = 翻币直到失败；串铺满翻币流 ⇒ 翻币流长度 ≈ budget/(1−p)
    flips_per = int(budget / (1 - p)) + N + 64
    chunk = 1000
    succ = 0
    for s0 in range(0, n_accounts, chunk):
        e = min(s0 + chunk, n_accounts)
        bits = rng.random((e - s0, flips_per)) < p
        succ += int((max_run_rows(bits) >= N).sum())
    return {"n_accounts": n_accounts, "successes": succ,
            "rate": succ / n_accounts}


def play_B(n_accounts: int, p: float, W: int = 7, seed: int = 0
           ) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    flips = rng.random((n_accounts, W)) < p
    succ = int(np.all(flips, axis=1).sum())
    return {"n_accounts": n_accounts, "successes": succ,
            "rate": succ / n_accounts, "analytic": p ** W}


# ----------------------------------------------------------------------
# §4/§7 玩法 C（分阶段止盈）：串独立 ⇒ 每关通过率解析精确
# ----------------------------------------------------------------------
def play_C(p: float, stages: int = 7, w0: int = 10_000
           ) -> dict[str, Any]:
    """第 i 关：资本 W_i，需一串连赢 k_i = ⌈log₂(W_i+1)⌉；预算 = W_i 串。

    串独立 ⇒ 每关通过率 = 1 − (1 − p^k)^W（解析精确）。
    过关后资本翻倍（文档口径：1万→2万→4万…）。
    """
    W = w0
    ks, curve = [], [w0]
    rate = 1.0
    for _s in range(stages):
        k_need = math.ceil(math.log2(W + 1))
        ks.append(k_need)
        p_stage = 1.0 - (1.0 - p ** k_need) ** W
        rate *= p_stage
        curve.append(rate * w0)
        W *= 2
    return {"ks": ks, "survivors": [round(x) for x in curve],
            "success_rate": rate}


# ----------------------------------------------------------------------
# §3 赌徒破产：P × 终值 ≈ W₀
# ----------------------------------------------------------------------
def gambler_ruin(n_accounts: int = 4000, N: int = 20, budget: int = 10_000,
                 seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    flips_per = int(budget / 0.5) + N + 64
    bits = rng.random((n_accounts, flips_per)) < 0.5
    hit = max_run_rows(bits) >= N
    n_hit = int(hit.sum())
    tv = 10_000 + 2 ** N - 1
    pB = 0.5 ** 7
    return {
        "A": {"P": n_hit / n_accounts, "终值": tv,
              "P乘终值": n_hit / n_accounts * tv, "n_hit": n_hit},
        "B": {"P": pB, "终值": 10_000 * 2 ** 7, "P乘终值": pB * 10_000 * 2 ** 7},
        "W0": 10_000,
    }


# ----------------------------------------------------------------------
# §5 收割比例 f（p=0.48, N=10）
# ----------------------------------------------------------------------
def play_harvest(p: float = 0.48, N: int = 10, 串_budget: int = 10_000,
                 n_accounts: int = 2000, seed: int = 0
                  ) -> list[dict[str, Any]]:
    """f=0/0.5/1：wealth 从 1 起，赢 ⇒ wealth×2 并收割 f×(wealth−1)；
    失败 ⇒ wealth 归零、串结束（这串净 = 已收割 − 1）；赢满 N 次 ⇒ 收手。

    账户 = 串_budget 串，每串风险 1 元；破产 = cash ≤ 0。
    """
    rng = np.random.default_rng(seed)
    out = []
    for f in (0.0, 0.5, 1.0):
        cash = np.full(n_accounts, 10000.0)
        alive = np.ones(n_accounts, dtype=bool)
        total_net = np.zeros(n_accounts)
        n串 = np.zeros(n_accounts)
        for _s in range(串_budget):
            act = np.where(alive)[0]
            if len(act) == 0:
                break
            wealth = np.ones(len(act))
            harv = np.zeros(len(act))
            alive_now = np.ones(len(act), dtype=bool)
            for _step in range(N):
                idx = np.where(alive_now)[0]
                if len(idx) == 0:
                    break
                w = rng.random(len(idx)) < p
                # 赢：收割 f×(wealth−1)，wealth = wealth − 收割 + 2×wealth？ 
                # 语义：翻倍后 wealth×2，收割 f×(wealth_before)，保留其余
                wb = wealth[idx].copy()
                w2 = 2.0 * wb                       # 赢 ⇒ 翻倍
                harv_delta = f * (w2 - 1.0)          # 收割 f×累计利润
                wealth[idx] = np.where(w, w2 - harv_delta, 0.0)
                harv[idx] += np.where(w, harv_delta, 0.0)
                # ⚠️ 我第一版漏了「wealth −= 收割」⇒ 边收割边复利，凭空生钱
                #   （f=1 的 E 跑成 +2.8；修正后应 ≈ −0.077 与文档吻合）
                alive_now[idx] = w
                if not alive_now.any():
                    break
            # 串结束：赢满 N ⇒ 净 = harv + wealth − 1；失败 ⇒ 净 = harv − 1
            net = np.where(alive_now, harv + wealth - 1.0, harv - 1.0)
            total_net[act] += net
            n串[act] += 1
            cash[act] += net
            alive &= cash > 0
        fin = 10000.0 + total_net
        out.append({"f": f, "E_per_串": float(total_net.sum() / max(n串.sum(), 1)),
                    "median_final": float(np.median(fin)),
                    "ruin_rate": float((fin <= 1.0).mean())})
    return out


def main() -> int:
    OUTD.mkdir(parents=True, exist_ok=True)
    res: dict[str, Any] = {}
    res["closed_form"] = check_closed_form()
    res["streak_dist"] = streak_distribution(100_000, 0.5, SEED + 2)
    res["max_streak"] = max_streak_vs_log2(SEED)
    res["A"] = {str(p): play_A(10000, p, seed=SEED + 3) for p in (0.50, 0.48, 0.45)}
    res["B"] = {str(p): play_B(10_000, p, seed=SEED + 5) for p in (0.50, 0.48, 0.45)}
    res["C"] = {str(p): play_C(p) for p in (0.50, 0.48, 0.45)}
    res["ruin"] = gambler_ruin(n_accounts=10000, seed=SEED + 9)
    res["harvest"] = play_harvest(seed=SEED + 11)
    (OUTD / "right_tail.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1, default=float),
        encoding="utf-8")
    print("closed_form all_match:", res["closed_form"]["all_match"],
          res["closed_form"]["mismatches"][:3])
    print("A:", {p: round(v["rate"] * 100, 3) for p, v in res["A"].items()})
    print("B:", {p: (round(v["rate"] * 100, 3), round(v["analytic"] * 100, 3))
                 for p, v in res["B"].items()})
    print("C:", {p: (round(v["sim_rate"] if "sim_rate" in v else v["success_rate"] * 100, 3))
                 for p, v in res["C"].items()})
    print("C curve p=.50:", res["C"]["0.5"]["survivors"])
    print("C curve p=.48:", res["C"]["0.48"]["survivors"])
    print("harvest:", [(h["f"], round(h["E_per_串"], 3),
                         round(h["median_final"]), round(h["ruin_rate"], 3))
                        for h in res["harvest"]])
    print("ruin:", {k: (round(v["P"] * 100, 3), round(v["P乘终值"]))
                    for k, v in res["ruin"].items() if k != "W0"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
