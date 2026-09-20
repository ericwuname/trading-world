"""工作线C：阶段9 修复后重跑 + 补上缺失的配对反事实。

C.0 要回答的两件事（任务书）
----------------------------
1. ``short_limit=0`` 那个静默丢单 bug 修好后，E9.2/E9.3 的数字有没有被
   **完整重跑并更新**？（→ 已在 2026-09-18 17:15–18:03 完整重跑，本脚本补
   **修复前后对比表**：这是"记账类静默 bug 如何扭曲策略结论"的具体案例）
2. E9.4 缺的**同种子配对反事实**要补上——这是二期报告 §11.1 反复强调的
   「因果测量三原则」第二条，在 E9.4 上一直没做。

⚠️ 前提核对（任务书 vs 源码）
------------------------------
· 任务书以为阶段9 还没重跑 → **已经重跑过**（用修复后的代码、全量）。
· 任务书建议新建 ``rerun_stage9_post_fix.py`` 并用 ``short_limit_override=0.0``
  复现 bug → 做法对，但旋钮已存在（``attach_pairs_trader(short_headroom=)``），
  无需额外加参数；本脚本直接用它。
· 任务书说 E9.2/E9.3 要"确认修复后数字有更新" → E9.2 是**不用 pairs** 的
  基线路径，修复它不可能影响；EC.2 就是去**证明**这一点（而不是假定）。

用法::

    python scripts/run_workstream_C.py                 # 全部
    python scripts/run_workstream_C.py --only EC1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import OUT, banner, load_json, save_json  # noqa: E402
from run_stage9 import (  # noqa: E402
    N_TICKS,
    SEEDS,
    WARMUP,
    build,
    run_market,
)
from tw.agents.adaptive_trend import realized_pnl_bp  # noqa: E402
from tw.agents.pairs_trader import PairsConfig  # noqa: E402

#: EC.1/EC.2 用 6 个种子（任务书：EC.1/EC.2 可先用较少种子）
SEEDS6 = [20260917 + 7 * i for i in range(6)]
#: EC.3/EC.4 用 12 个种子（任务书明确要求 EC.3 至少 12 个）
SEEDS12 = [20260917 + 7 * i for i in range(12)]
CV_GRID = (3e-4, 1e-3, 2e-3, 8e-3)
PAIR_CFG = PairsConfig(zwindow=100, zthresh=1.0, zexit=0.3)


def f(v, fmt: str = ".3f", dash: str = "—") -> str:
    if not isinstance(v, (int, float)) or v != v:
        return dash
    return format(v, fmt)


# ======================================================================
# EC.1  修复前后对比：半条腿 vs 两条腿
# ======================================================================
def pairs_pnl(short_headroom: float, seeds, cv: float = 2e-3,
              cfg: PairsConfig | None = None) -> dict:
    """跑一轮配对交易，返回组合 PnL（bp）与两腿成交笔数。"""
    cfg = cfg or PairsConfig(zwindow=100, zthresh=1.5, zexit=0.5)
    pnl, nfa, nfb, qa, qb = [], [], [], [], []
    for sd in seeds:
        m, tr, _ = build(2, cv, 4e-4, sd, pairs=(0, 1), pairs_cfg=cfg,
                         short_headroom=short_headroom)
        run_market(m, tr)
        mid = [m.markets[i].current_mid() for i in range(2)]
        pa = realized_pnl_bp(tr.legs[0], mid[0])
        pb = realized_pnl_bp(tr.legs[1], mid[1])
        pnl.append(float(np.nanmean([pa, pb])))
        nfa.append(_n_fills(m, 0, tr.legs[0].agent_id))
        nfb.append(_n_fills(m, 1, tr.legs[1].agent_id))
        qa.append(float(tr.legs[0].inventory))
        qb.append(float(tr.legs[1].inventory))
    return {"pnl_bp": float(np.nanmean(pnl)), "pnl_sd": float(np.nanstd(pnl, ddof=1)),
            "pnl_per_seed": pnl, "fills_a": float(np.mean(nfa)),
            "fills_b": float(np.mean(nfb)),
            "qty_a": float(np.mean(qa)), "qty_b": float(np.mean(qb))}


def _n_fills(m, idx: int, agent_id: str) -> float:
    """某一腿作为**买或卖任一侧**参与的成交笔数。

    ``Trade`` 的字段是 ``buy_agent_id`` / ``sell_agent_id``（不是 maker/taker）——
    第一版写成 maker_id 直接 AttributeError。注意这**不是**"主动方/被动方"：
    那由 ``aggressor_side`` 表达，而腿是市价调仓的 taker。
    """
    return float(len([t for t in m.markets[idx].log.trades
                      if t.buy_agent_id == agent_id
                      or t.sell_agent_id == agent_id]))


def net_exposure_ratio(short_headroom: float, seeds) -> dict:
    """净暴露 / 总暴露 —— 配对交易的定义性质（市场中性）。"""
    ratios = []
    for sd in seeds:
        m, tr, _ = build(2, 2e-3, 4e-4, sd, pairs=(0, 1), pairs_cfg=PAIR_CFG,
                         short_headroom=short_headroom)
        run_market(m, tr)
        na = float(tr.legs[0].inventory) * m.markets[0].current_mid()
        nb = float(tr.legs[1].inventory) * m.markets[1].current_mid()
        g = abs(na) + abs(nb)
        ratios.append(abs(na + nb) / g if g > 0 else float("nan"))
    return {"ratio_mean": float(np.nanmean(ratios)), "per_seed": ratios}


def ec1_pre_post(seeds=SEEDS6) -> dict:
    print("\n【EC.1】修复前后对比：short_limit=0（半条腿）vs 3×（两条腿）")
    print(f"  同种子、同参数网格；种子数 {len(seeds)}；观测 {N_TICKS} tick（预热 {WARMUP}）")
    broken = pairs_pnl(0.0, seeds)
    fixed = pairs_pnl(3.0, seeds)
    ex_b = net_exposure_ratio(0.0, seeds)
    ex_f = net_exposure_ratio(3.0, seeds)
    pct = ((fixed["pnl_bp"] - broken["pnl_bp"]) / abs(broken["pnl_bp"])
           if broken["pnl_bp"] else float("nan"))
    print(f"    {'配置':<24}{'组合PnL(bp)':>13}{'腿A成交':>10}{'腿B成交':>10}"
          f"{'净/总暴露':>11}")
    print(f"    {'修复前(short_limit=0)':<24}{f(broken['pnl_bp'], '+.3f'):>13}"
          f"{f(broken['fills_a'], '.0f'):>10}{f(broken['fills_b'], '.0f'):>10}"
          f"{f(ex_b['ratio_mean'], '.3f'):>11}")
    print(f"    {'修复后(3×notional)':<24}{f(fixed['pnl_bp'], '+.3f'):>13}"
          f"{f(fixed['fills_a'], '.0f'):>10}{f(fixed['fills_b'], '.0f'):>10}"
          f"{f(ex_f['ratio_mean'], '.3f'):>11}")
    print(f"    {'变化':<24}{f(pct, '+.1%'):>13}")
    print(f"\n  → **bug 造成的 PnL 偏差 = {pct:+.1%}**"
          f"（{broken['pnl_bp']:+.3f} → {fixed['pnl_bp']:+.3f} bp）")
    print(f"  → 净暴露占比 {f(ex_b['ratio_mean'], '.3f')} → "
          f"{f(ex_f['ratio_mean'], '.3f')}：修复前**几乎完全单边**，"
          f"即「配对交易」其实是方向性策略")
    return {"broken": broken, "fixed": fixed, "pct_diff": pct,
            "net_exposure_broken": ex_b, "net_exposure_fixed": ex_f,
            "n_seeds": len(seeds)}


# ======================================================================
# EC.2  基线回归：修复是否意外影响了不用 pairs 的路径
# ======================================================================
def ec2_baseline_regression(seeds=SEEDS6) -> dict:
    """两个资产用**完全相同**的参数（只有初始价格不同），统计特征应当接近。

    这一项存在的意义是**证明**"修复 pairs_trader 不会影响不带 pairs 的路径"，
    而不是假定它。``short_headroom`` 只在挂 pairs 时才起作用，
    所以两次结果应当**逐位相同**——这正是要验证的。
    """
    print("\n【EC.2】基线回归：修复是否影响不用 pairs 的路径")
    rows = []
    for sd in seeds[:3]:
        for hr in (0.0, 3.0):
            m, _, _ = build(2, 2e-3, 4e-4, sd, pairs=None,
                            short_headroom=hr)
            run_market(m, None)
            rows.append({"seed": sd, "short_headroom": hr,
                         "sigma_bp": float(np.nanstd(np.diff(np.log(
                             np.asarray(m.markets[0].log.mid[:m.tick],
                                        dtype=float))) * 1e4, ddof=1)),
                         "n_trades": int(len(m.markets[0].log.trades)),
                         "mid_sum": float(np.nansum(m.markets[0].log.mid[:m.tick]))})
    # 按种子比对两次是否逐位一致
    same = []
    for sd in seeds[:3]:
        a = [r for r in rows if r["seed"] == sd and r["short_headroom"] == 0.0][0]
        b = [r for r in rows if r["seed"] == sd and r["short_headroom"] == 3.0][0]
        same.append(a["n_trades"] == b["n_trades"]
                    and abs(a["mid_sum"] - b["mid_sum"]) < 1e-6)
    print(f"    不带 pairs 时，参数改动是否改变了行情：{same}")
    print(f"    → {'✅ 逐位一致（修复是局部的，不影响基线路径）' if all(same) else '❌ 竟然改了基线路径，必须查'}")
    return {"rows": rows, "identical": same,
            "pass": bool(all(same))}


# ======================================================================
# EC.3  E9.4 配对反事实
# ======================================================================
def _liquidate_A(m) -> None:
    """对资产 A 里持仓最重的一批主体按市价砸出去（与 E9.4 同做法）。"""
    victims = sorted(m.markets[0].agents,
                     key=lambda a: a.inventory, reverse=True)[:20]
    for a in victims:
        if a.inventory <= 1e-6:
            continue
        o = a.new_order(m.markets[0]._state, "sell",
                        m.markets[0].current_mid(), a.inventory,
                        order_type="market")
        if o is not None:
            m.markets[0].submit(a, o)


def paired_propagation(cv: float, seed: int, horizon: int = 300,
                       short_headroom: float = 3.0) -> dict:
    """同种子跑两次（不砸 / 砸 A），逐 tick 相减得到 B 的**传导路径**。

    ⭐ 这是二期报告 §11.1「因果测量三原则」第二条（同种子配对）在 E9.4 上
    的正式补课。原来只报了"B 的最大偏离"，而那是**处理路径自己**的偏离——
    里面混着"这个种子本来就有的漂移"。配对相减才把它扣掉。
    """
    def run(trigger: bool):
        m, tr, _ = build(2, cv, 4e-4, seed, pairs=(0, 1), pairs_cfg=PAIR_CFG,
                         short_headroom=short_headroom)
        run_market(m, tr, WARMUP)
        p_b0 = m.markets[1].current_mid()
        if trigger:
            _liquidate_A(m)
        path = []
        for _ in range(horizon):
            m.step()
            if tr is not None:
                tr.step()
            path.append(m.markets[1].current_mid() / p_b0 - 1.0)
        return np.asarray(path, dtype=float) * 1e4

    treated = run(True)
    control = run(False)
    n = min(treated.size, control.size)
    if n == 0:
        # ⚠️ 空路径必须显式返回 nan，不能让 ``np.nanmax`` 去 reduce 空数组
        #    （那会抛 ValueError，而不是给出一个"没有数据"的答案）。
        #    这不是生产路径（horizon=300），但测试会用 horizon=0 去只验证
        #    "种子怎么传"——一个只在该边界炸掉的实现会让测试无法表达意图。
        return {"treated": [], "control": [], "paired": [],
                "treated_max": float("nan"), "paired_max": float("nan"),
                "paired_tail": float("nan"), "n": 0}
    paired = treated[:n] - control[:n]
    return {"treated": treated.tolist(), "control": control.tolist(),
            "paired": paired.tolist(), "n": int(n),
            "treated_max": float(np.nanmax(np.abs(treated))),
            "paired_max": float(np.nanmax(np.abs(paired))),
            "paired_tail": float(np.nanmean(paired[-50:]))}


def ec3_paired_counterfactual(seeds=SEEDS12) -> dict:
    print("\n【EC.3】E9.4 补配对反事实（同种子：不砸 A vs 砸 A）")
    print(f"  种子数 {len(seeds)}（原来是 3）；每个 (cv, seed) 跑两次")
    print(f"    {'σ_c':>10}{'实测相关':>10}{'未配对 B 最大偏离':>18}"
          f"{'配对后传导幅度':>16}{'配对后末段':>12}")
    rows = []
    for cv in CV_GRID:
        pairs_, raw_, tails_, cors_ = [], [], [], []
        for sd in seeds:
            r = paired_propagation(cv, sd)
            pairs_.append(r["paired_max"])
            raw_.append(r["treated_max"])
            tails_.append(r["paired_tail"])
            cors_.append(_realized_corr(cv, sd))
        raw_arr = np.asarray(raw_, dtype=float)
        pair_arr = np.asarray(pairs_, dtype=float)
        # 降噪倍数 = 跨种子标准差之比（配对应显著更小）
        noise = (float(raw_arr.std(ddof=1) / pair_arr.std(ddof=1))
                 if pair_arr.std(ddof=1) > 0 else float("nan"))
        rows.append({"common_vol": cv, "realized_corr": float(np.nanmean(cors_)),
                     "raw_max_bp": float(np.nanmean(raw_arr)),
                     "raw_sd": float(raw_arr.std(ddof=1)),
                     "paired_max_bp": float(np.nanmean(pair_arr)),
                     "paired_sd": float(pair_arr.std(ddof=1)),
                     "paired_tail_bp": float(np.nanmean(tails_)),
                     "paired_ci95": _ci95(pair_arr), "noise_reduction": noise})
        r = rows[-1]
        print(f"    {cv:>10.1e}{f(r['realized_corr'], '.4f'):>10}"
              f"{f(r['raw_max_bp'], '.1f') + ' ±' + f(r['raw_sd'], '.1f'):>18}"
              f"{f(r['paired_max_bp'], '.1f') + ' ±' + f(r['paired_sd'], '.1f'):>16}"
              f"{f(r['paired_tail_bp'], '+.1f'):>12}")
    nr = float(np.nanmean([r["noise_reduction"] for r in rows]))
    mono = all(rows[i]["paired_max_bp"] <= rows[i + 1]["paired_max_bp"]
               for i in range(len(rows) - 1))
    print(f"\n  → 配对带来的降噪倍数（跨种子 sd 之比）均值 = **{nr:.1f}×**"
          f"（一期阶段3 曾达 62×，同一数量级即可）")
    print(f"  → 配对后传导幅度是否随 σ_c 单调上升：{'是' if mono else '否'}")
    for r in rows:
        print(f"     σ_c={r['common_vol']:.1e}: 配对后 {r['paired_max_bp']:.1f}bp "
              f"CI95 [{r['paired_ci95'][0]:.1f}, {r['paired_ci95'][1]:.1f}]")
    return {"rows": rows, "noise_reduction_mean": nr, "monotone": bool(mono),
            "n_seeds": len(seeds)}


def _ci95(a: np.ndarray) -> list[float]:
    a = np.asarray([v for v in a if np.isfinite(v)], dtype=float)
    if a.size < 2:
        return [float("nan"), float("nan")]
    from scipy import stats
    sem = a.std(ddof=1) / np.sqrt(a.size)
    h = float(stats.t.ppf(0.975, a.size - 1) * sem)
    return [float(a.mean() - h), float(a.mean() + h)]


def _realized_corr(cv: float, seed: int) -> float:
    m, tr, _ = build(2, cv, 4e-4, seed, pairs=(0, 1), pairs_cfg=PAIR_CFG)
    run_market(m, tr)
    return float(m.realized_correlation(0, 1, WARMUP // 2))


def ec4_sweep_with_pairing(seeds=SEEDS12) -> dict:
    """EC.4：用 EC.3 的配对方法重跑相关性敏感性；看"不单调"是否还在。"""
    print("\n【EC.4】相关性敏感性重测（配对方法）")
    r = ec3_paired_counterfactual(seeds)
    if r["monotone"]:
        print("  → 配对后**变单调** ⇒ 原来的「不单调」是噪音；"
              "阶段9/11 的诚实边界文字需要更新")
    else:
        print("  → 配对后**依然不单调** ⇒ 不单调本身是真实现象，不是噪音；"
              "原结论文字保留并补充说明")
    return r


# ======================================================================
def main() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="EC1 / EC2 / EC3 / EC4")
    args = ap.parse_args()
    banner("工作线C  阶段9 修复后重跑 + 配对反事实")
    t0 = time.time()
    out: dict = {"stage": "C", "config": {"seeds6": SEEDS6, "seeds12": SEEDS12,
                                          "cv_grid": list(CV_GRID),
                                          "n_ticks": N_TICKS}}
    only = (args.only or "").upper()
    if not only or only == "EC1":
        out["ec1"] = ec1_pre_post()
    if not only or only == "EC2":
        out["ec2"] = ec2_baseline_regression()
    if only == "EC4":
        out["ec4"] = ec4_sweep_with_pairing()
    elif not only or only == "EC3":
        out["ec3"] = ec3_paired_counterfactual()
    out["elapsed_sec"] = time.time() - t0
    # ⚠️ **合并写入**，不是覆盖：`--only` 是分批跑的，
    #    如果用 ``save_json(out, ...)`` 直接覆盖，跑 EA3 就会把 EA0/EA1
    #    的结果抹掉（"两个写者各写一半"是本项目反复踩的同一类问题）。
    merged = load_json("workstream_C_metrics.json") if (OUT / "workstream_C_metrics.json").exists() else {}
    merged.update(out)
    merged["elapsed_sec"] = out["elapsed_sec"]
    save_json(merged, "workstream_C_metrics.json")
    print(f"\n  总用时 {out['elapsed_sec']:.0f}s → out/workstream_C_metrics.json")
    return out


if __name__ == "__main__":
    main()
