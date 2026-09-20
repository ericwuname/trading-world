"""阶段7：自适应/学习型主体与反身性（二期指导书 §4）。

要回答的问题
------------
**当一个已知有效的策略被越来越多人采用时，它自己的有效性会不会内生地衰减？**
对应真实世界的观测：BTC 资金费率 carry 的年化收益从 14.4% 衰减到 5%。

实验编号（对应指导书 §4.4）
--------------------------
E7.1 学习收敛性     20 个 ε-greedy bandit 主体，看 Q 值收敛还是震荡
E7.2 反身性核心实验 固定策略采用者 [0,5,10,20,40,80,160]，看平均 PnL 是否随人数衰减
E7.3 衰减曲线拟合   衰减形状与真实 carry 衰减（14.4%→5%，约 65% 剩余）的量级比较
E7.4 市场特征反馈   市场整体统计特征是否随拥挤度变化

⭐ 四条必须先说清楚的方法论
==========================

**① 被测对象必须是「不变的」。**
   反身性实验的采用者是 ``FixedMomentumTrader``（固定 lookback），
   **不是** ``AdaptiveTrendFollower``。如果采用者自己也在学习，
   它会随拥挤度调参——测到的 PnL 变化里就混进了"它自己改好了"这一项，
   而我们要的是**同一个东西**在不同拥挤度下的表现。

**② 总主体数固定（替换，而不是叠加）—— 这一条刻意偏离指导书。**
   指导书 §4.4 的"注入该数量的 agent"隐含叠加，会让 N 从 300 涨到 460。
   但一期已实测 σ 强烈依赖主体数 N（纯零智能池 N=100→500 时 σ 从 37→86bp）。
   叠加设计下"采用者变多"同时意味着"市场变大变厚"，
   采用者 PnL 的变化里就分不清是**同业拥挤**还是**市场变厚**。
   替换设计下 N 恒定，"同行的人变多"这一项才被隔离出来。
   **代价**：背景噪音交易者变少——这是已知且必须报告的边界。

**③ 同种子配对 + 采用者禀赋完全相同。**
   同 seed 下只改 adopter_count；采用者发同一份现金与持仓（不按序号抽）。
   否则比的是"谁的禀赋运气好"，而不是"谁更拥挤"。

**④ 每个 (count, seed) 都是全新初始化的市场对象。**
   复用或深拷贝一个跑过的 market 会把市场记忆带进来
   （订单簿残留、持仓、日志位置），曲线里就会出现一条来自实验装置的伪趋势。
   指导书 §4.7 专门警告过。

用法::

    python scripts/run_stage7.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import FIG, banner, save_json  # noqa: E402
from tw import Market, Population, SimConfig  # noqa: E402
from tw.agents.adaptive_trend import AdaptiveTrendFollower  # noqa: E402
from tw.experiments.reflexivity import (  # noqa: E402
    ReflexivityConfig,
    run_reflexivity_sweep,
    spearman,
)

SEED0 = 20260917

# ----------------------------------------------------------------------
# E7.1 的配置
#
# ⚠️ 相对指导书 §4.4 的调整（都要在报告里说清楚）：
#   · 50000 tick × 8 种子 → **40000 tick × 4 种子**。
#     8 个种子的成本是 8 × 60s ≈ 8 分钟，而收敛/震荡这个判断是**定性**的
#     （看 Q 轨迹的形状），4 个种子足够分辨"都收敛"/"都在震荡"；
#     再加种子不会改变形状，只会改变误差棒。
#   · eval_horizon 先用**固定策略**做过信噪比体检再定（指导书 §4.7 的建议）。
# ----------------------------------------------------------------------
E71_TICKS = 40_000
E71_SEEDS = [SEED0, SEED0 + 7, SEED0 + 14, SEED0 + 21]
E71_N_LEARNERS = 20
E71_SNAPSHOT_EVERY = 500

#: E7.2/E7.3/E7.4 的配置
ADOPTER_COUNTS = [0, 5, 10, 20, 40, 80, 160]
E72_SEEDS = [SEED0, SEED0 + 7, SEED0 + 14, SEED0 + 21]
E72_TICKS = 16_000
E72_WARMUP = 4_000

#: 真实数据对照：BTC 资金费率 carry 年化从 14.4% 衰减到 5%
REAL_CARRY_START = 0.144
REAL_CARRY_END = 0.05
REAL_CARRY_RETAINED = REAL_CARRY_END / REAL_CARRY_START


# ----------------------------------------------------------------------
def horizon_health_check(seed: int = SEED0, horizons=(5, 10, 20, 50, 100)) -> list[dict]:
    """指导书 §4.7 的建议：先用**固定策略**（不学习）量一遍各 eval_horizon
    下"信号本身"的信噪比，再决定学习用哪个 horizon。

    这里测的是**信号**的夏普（mid-to-mid 收益 / 其标准差），不是策略 PnL——
    理由是 bandit 评估的就是信号质量（见 ``adaptive_trend`` 的模块文档①）。
    如果换成"策略 PnL 的信噪比"，选出来的 horizon 会偏向"容易成交"的窗口，
    而不是"预测力强"的窗口，两者不是同一件事。

    产出的是一个**决策依据**，不是一个成绩单。
    """
    m = Market(SimConfig(
        seed=seed, n_ticks=6_000,
        population=Population.from_shares(
            300, {"zero_intel": 0.30, "fundamentalist": 0.40, "chartist": 0.30}),
    ))
    m.run(6_000)
    mid = np.asarray(m.log.mid[: m.tick], dtype=float)
    rows = []
    for h in horizons:
        if h >= mid.size - 10:
            continue
        fwd = mid[h:] / mid[:-h] - 1.0
        # 用过去 h 期的动量做符号预测（与 bandit 看到的信号同一构造）
        past = mid[:-h] / np.concatenate([np.full(h, mid[0]), mid[:-2 * h] + 0])[: mid.size - h] - 1.0 \
            if mid.size > 2 * h else None
        if past is None or past.size != fwd.size:
            continue
        side = np.sign(past)
        pnl = side * fwd
        ok = np.isfinite(pnl)
        if ok.sum() < 100:
            continue
        p = pnl[ok]
        sd = float(p.std(ddof=1))
        rows.append({
            "horizon": int(h),
            "mean_bp": float(p.mean() * 1e4),
            "sd_bp": sd * 1e4,
            "sharpe": float(p.mean() / sd) if sd > 0 else float("nan"),
            "hit_rate": float((p > 0).mean()),
            "n": int(ok.sum()),
        })
    return rows


def run_learners(seed: int, n_ticks: int, n_learners: int,
                 eval_horizon: int, snapshot_every: int) -> dict:
    """E7.1：混入 ``n_learners`` 个学习主体，分块推进并快照 Q 值。

    **为什么必须分块**：Q 值的**轨迹**才是这一实验的产出。
    只留最终 Q 值分不清"收敛到某个值"和"来回震荡但恰好停在这里"——
    而后者本身就是反身性的直接证据（指导书 §4.5 明说两种结果都是有效产出）。
    """
    m = Market(SimConfig(
        seed=seed, n_ticks=n_ticks,
        population=Population.from_shares(
            300, {"zero_intel": 0.30, "fundamentalist": 0.40, "chartist": 0.30}),
    ))
    learners = []
    for i in range(n_learners):
        a = AdaptiveTrendFollower(
            f"at{i:04d}", 6.0e5, 10.0, np.random.default_rng([seed, 0xA7A9, i]),
            eval_horizon=eval_horizon,
        )
        a.bind_seed(seed)
        m.add_agent(a)
        a.set_initial_equity(m.current_mid())
        learners.append(a)

    traj = []
    done = 0
    while done < n_ticks:
        step = min(snapshot_every, n_ticks - done)
        m.run(step)
        done += step
        # 快照：全体学习主体在各 lookback 上的 Q 值均值
        qs = {lb: float(np.mean([x.q_values[lb] for x in learners]))
              for lb in learners[0].candidates}
        traj.append({"tick": done, **{f"q{lb}": v for lb, v in qs.items()}})

    best = [x.best_lookback() for x in learners]
    return {
        "seed": seed,
        "n_ticks": n_ticks,
        "n_learners": n_learners,
        "eval_horizon": eval_horizon,
        "trajectory": traj,
        "final_q": {str(lb): float(np.mean([x.q_values[lb] for x in learners]))
                    for lb in learners[0].candidates},
        "best_lookbacks": best,
        "best_counts": {str(lb): int(best.count(lb))
                        for lb in sorted(set(best))},
        "n_choices": int(sum(x.n_choices for x in learners)),
        "n_updates": int(sum(x.n_updates for x in learners)),
        "choice_counts": {str(lb): int(sum(x.choice_counts.get(lb, 0)
                                            for x in learners))
                          for lb in learners[0].candidates},
    }


def _fit_decay(counts, values) -> dict:
    """E7.3：对衰减曲线做指数与幂律两种拟合，返回哪种更像。

    先做**护栏检查**：自变量（adopter_count）至少要跨 1 个数量级，
    否则拟合出的"衰减率"是在跨度不足的噪声上拟合的（一期在冲击函数上
    踩过同一个坑）。这里的档位 0→160 跨了两个数量级，够。
    """
    x = np.asarray(counts, dtype=float)
    y = np.asarray(values, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0)
    out: dict = {"ok": False, "reason": ""}
    if ok.sum() < 3:
        out["reason"] = "有效档位不足 3（count=0 无定义于幂律）"
        return out
    xo, yo = x[ok], y[ok]
    span = float(np.log10(xo.max()) - np.log10(xo.min()))
    out["log_span"] = span
    if span < 1.0:
        out["reason"] = f"自变量对数跨度仅 {span:.2f}，小于 1 个数量级，拒绝拟合"
        return out
    # 指数拟合：y = a·exp(b·x)，b<0 表示衰减
    if (yo > 0).all():
        b, la = np.polyfit(xo, np.log(yo), 1)
        pred = la + b * xo
        ss_res = float(((np.log(yo) - pred) ** 2).sum())
        ss_tot = float(((np.log(yo) - np.log(yo).mean()) ** 2).sum())
        out["exponential_decay_rate"] = float(-b)
        out["exponential_r2"] = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    # 幂律拟合：y = a·x^c
    # ⚠️ 必须与指数拟合一样先检查 y > 0：采纳者的 PnL 在**亏光**时是负数，
    # 直接 np.log 会产出一串 nan 并抛 RuntimeWarning（实测就这么崩的）。
    if not (yo > 0).all():
        out["reason"] = (f"因变量里有非正值（最小 {yo.min():.1f}）——"
                         "这一档已经接近亏光，衰减曲线失去意义")
        return out
    p, lc = np.polyfit(np.log(xo), np.log(yo), 1)
    predp = lc + p * np.log(xo)
    ss_res = float(((np.log(yo) - predp) ** 2).sum())
    ss_tot = float(((np.log(yo) - np.log(yo).mean()) ** 2).sum())
    out["powerlaw_exponent"] = float(p)
    out["powerlaw_r2"] = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    out["ok"] = True
    if np.isfinite(out.get("exponential_r2", float("nan"))) and \
       np.isfinite(out.get("powerlaw_r2", float("nan"))):
        out["better"] = ("exponential"
                         if out["exponential_r2"] > out["powerlaw_r2"]
                         else "powerlaw")
    return out


# ----------------------------------------------------------------------
def main() -> dict:
    banner("阶段7  自适应/学习型主体与反身性")
    t0 = time.time()
    out: dict = {
        "stage": 7,
        "config": {
            "e71": {"ticks": E71_TICKS, "seeds": E71_SEEDS,
                    "n_learners": E71_N_LEARNERS,
                    "snapshot_every": E71_SNAPSHOT_EVERY},
            "e72": {"adopter_counts": ADOPTER_COUNTS, "seeds": E72_SEEDS,
                    "n_ticks": E72_TICKS, "warmup": E72_WARMUP},
            "real_carry": {"start": REAL_CARRY_START, "end": REAL_CARRY_END,
                           "retained": REAL_CARRY_RETAINED},
        },
    }

    # ==================================================================
    # E7.0  eval_horizon 的信噪比体检（指导书 §4.7 的建议）
    # ==================================================================
    print("\n【E7.0】eval_horizon 的信噪比体检（先用固定信号量一遍）")
    print("  测的是**信号**的夏普（mid-to-mid 收益/其标准差），不是策略 PnL——")
    print("  bandit 评估的就是信号质量（见 tw/agents/adaptive_trend.py 模块文档）")
    print(f"    {'horizon':>9}{'平均bp':>10}{'sd bp':>10}{'夏普':>9}{'胜率':>9}{'样本':>10}")
    hc = horizon_health_check()
    for r in hc:
        print(f"    {r['horizon']:>9}{r['mean_bp']:>10.3f}{r['sd_bp']:>10.2f}"
              f"{r['sharpe']:>9.4f}{r['hit_rate']:>9.3f}{r['n']:>10}")
    best_h = max((r for r in hc if np.isfinite(r["sharpe"])),
                 key=lambda r: r["sharpe"], default=None)
    e71_horizon = int(best_h["horizon"]) if best_h else 20
    print(f"  → 夏普最高的 horizon = {e71_horizon}"
          f"（夏普 {best_h['sharpe']:.4f}，胜率 {best_h['hit_rate']:.3f}）"
          if best_h else "  → 无有效 horizon，退回默认 20")
    if best_h and best_h["sharpe"] <= 0:
        print("  ⚠️ 注意：最优 horizon 的**信号夏普仍是非正的**——")
        print("     说明在这个市场里动量信号本身不带正期望。这本身是结论：")
        print("     学习主体能学到的只是「哪个窗口最不差」，而不是「一个赚钱的窗口」。")
    out["e7_0_horizon_check"] = {"rows": hc, "chosen": e71_horizon}
    print(f"  ⚠️ 相对指导书的调整：50000 tick × 8 种子 → {E71_TICKS} tick × "
          f"{len(E71_SEEDS)} 种子。收敛/震荡是**定性**判断（看 Q 轨迹形状），")
    print(f"     4 种子足够分辨；加种子只会改变误差棒，不会改变形状。")

    # ==================================================================
    # E7.1  学习收敛性
    # ==================================================================
    print(f"\n【E7.1】学习收敛性（{E71_N_LEARNERS} 个 bandit 主体，"
          f"{E71_TICKS} tick，{len(E71_SEEDS)} 种子）")
    e71 = []
    for sd in E71_SEEDS:
        r = run_learners(sd, E71_TICKS, E71_N_LEARNERS, e71_horizon,
                         E71_SNAPSHOT_EVERY)
        e71.append(r)
        lbs = sorted({int(k) for k in r["final_q"]})
        qs = "  ".join(f"{lb}→{r['final_q'][str(lb)]:+.5f}" for lb in lbs)
        print(f"    seed={sd:>9}  最终Q: {qs}")
        print(f"       最优 lookback 分布: {r['best_counts']}")
    keys = sorted({int(k) for k in e71[0]["final_q"]})
    print(f"\n    Q 值轨迹的稳定性（各种子最终 Q 的跨种子标准差）：")
    for lb in keys:
        vals = [x["final_q"][str(lb)] for x in e71]
        print(f"      lookback={lb:>4}: 均值 {np.mean(vals):+.6f}  跨种子 sd {np.std(vals, ddof=1):.6f}")
    # 收敛判定：跨种子最终 Q 的相对离散度
    spread = {}
    for lb in keys:
        vals = np.array([x["final_q"][str(lb)] for x in e71], dtype=float)
        spread[lb] = float(np.std(vals, ddof=1))
    # "学到了同一个窗口" = 所有种子都指向同一个 best
    uniq = {tuple(sorted(set(x["best_lookbacks"]))) for x in e71}
    converged_same = len(uniq) == 1 and len(next(iter(uniq))) == 1
    print(f"\n    各种子学到的『最优 lookback』集合: {sorted(uniq)}")
    print(f"    → {'✅ 收敛到同一个窗口' if converged_same else '⚠️ 未收敛到同一个窗口（各种子结论不同）'}")
    print(f"    ⚠️ **两种结果都是有效产出**（指导书 §4.5 明写）。")
    print(f"       '持续震荡'本身就是反身性的直接证据：市场在变，最优参数也跟着变。")
    out["e7_1"] = {"runs": e71, "final_q_sd": {str(k): v for k, v in spread.items()},
                   "converged_same": bool(converged_same),
                   "best_sets": [sorted(s) for s in uniq]}

    # ==================================================================
    # E7.2  反身性核心实验
    # ==================================================================
    print(f"\n【E7.2】反身性核心实验")
    print(f"  采用者数量档位 {ADOPTER_COUNTS}，{len(E72_SEEDS)} 种子，"
          f"{E72_TICKS} tick（预热 {E72_WARMUP}）")
    print(f"  ⚠️ 设计偏离指导书：**替换**背景主体而不是叠加。")
    print(f"     叠加会让 N 从 300 涨到 460，而一期已实测 σ 强烈依赖 N——")
    print(f"     那样'采用者变多'同时意味着'市场变厚'，无法归因。")
    cfg = ReflexivityConfig(
        adopter_counts=list(ADOPTER_COUNTS), seeds=list(E72_SEEDS),
        n_agents=300, warmup=E72_WARMUP, n_ticks=E72_TICKS,
    )
    rows = run_reflexivity_sweep(cfg)
    e72 = []
    for count in ADOPTER_COUNTS:
        sel = [r for r in rows if r["adopter_count"] == count]
        pnl = [r["adopter_pnl_bp_mean"] for r in sel]
        e72.append({
            "adopter_count": count,
            "pnl_bp_mean": float(np.nanmean(pnl)) if count > 0 else float("nan"),
            "pnl_bp_sd": float(np.nanstd(pnl, ddof=1)) if len(pnl) > 1 else float("nan"),
            "pnl_per_seed": [float(v) for v in pnl],
            "n_background": sel[0]["n_background"] if sel else 0,
            "market_sigma_bp": float(np.nanmean([r["market"].get("sigma_bp", np.nan)
                                                 for r in sel])),
            "market_kurtosis": float(np.nanmean([r["market"].get("excess_kurtosis", np.nan)
                                                 for r in sel])),
            "market_abs_acf1": float(np.nanmean([r["market"].get("acf_abs_lag1", np.nan)
                                                 for r in sel])),
            "market_ret_acf1": float(np.nanmean([r["market"].get("acf_ret_lag1", np.nan)
                                                 for r in sel])),
            "n_trades": float(np.nanmean([r["n_trades"] for r in sel])),
            "background_trades": float(np.nanmean([r["background_trades"] for r in sel])),
            "health_ok": bool(all(r["health_ok"] for r in sel)),
        })
    print(f"    {'采用者':>8}{'平均PnL(bp)':>14}{'跨种子sd':>11}"
          f"{'背景主体':>10}{'σ(bp)':>9}{'成交数':>10}")
    for r in e72:
        print(f"    {r['adopter_count']:>8}{r['pnl_bp_mean']:>14.3f}{r['pnl_bp_sd']:>11.3f}"
              f"{r['n_background']:>10}{r['market_sigma_bp']:>9.2f}"
              f"{r['n_trades']:>10.0f}")

    xs = [r["adopter_count"] for r in e72 if r["adopter_count"] > 0]
    ys = [r["pnl_bp_mean"] for r in e72 if r["adopter_count"] > 0]
    rho, p = spearman(xs, ys)
    # ⭐ 饱和检测。realized_pnl_bp 的下限是 −10000bp（亏光全部权益）。
    # 若最大档位已经贴到下限，那么"PnL 随人数下降"测的是**亏损深度**、
    # 不是**拥挤侵蚀**——Spearman 会显著为负，而结论是假的。
    # 实测踩到过：所有档位的 PnL 都在 −9772 ~ −9992bp，ρ = −1.0、p = 0，
    # 看起来"完美验证了反身性"，其实是"都亏光了"。
    ruin_floor = -10_000.0
    sat = float(min(ys)) if ys else float("nan")
    saturated = bool(np.isfinite(sat) and sat <= ruin_floor * 0.9)
    if saturated:
        print(f"\n    ⚠️ **指标已饱和**：最大档位的平均 PnL = {sat:.0f}bp，"
              f"而亏光的下限是 {ruin_floor:.0f}bp。")
        print(f"       此时 Spearman 测的是「亏损深度」，**不是「拥挤侵蚀」**——")
        print(f"       ρ 会显著为负，但那个显著性来自「都亏光了」，不是反身性。")
        print(f"       E7.2 的验收判据在这个配置下是**假阳性**，必须这样标注。")
    print(f"\n    Spearman(adopter_count, 平均PnL) = {rho:+.4f}  p = {p:.5f}")
    # ⚠️ 饱和时**不计通过**：统计显著不等于结论成立（见上面的说明）。
    e72_pass = bool(np.isfinite(rho) and rho < 0 and p < 0.05 and not saturated)
    verdict72 = ("⚠️ 统计上显著为负，但**指标已饱和**（见上）→ 判为假阳性，不计通过"
                 if saturated else
                 "✅ E7.2 通过（显著为负，p<0.05）" if e72_pass
                 else "❌ E7.2 未通过")
    print(f"    → {verdict72}")

    # 逐种子配对：每个种子内，各档相对最低档的差
    print(f"\n    逐种子配对（同种子内各档相对 count={ADOPTER_COUNTS[1]} 的差，bp）：")
    print(f"    {'采用者':>8}" + "".join(f"{'seed'+str(s)[-4:]:>12}" for s in E72_SEEDS)
          + f"{'均值':>11}")
    base_by_seed = {r["seed"]: r["adopter_pnl_bp_mean"] for r in rows
                    if r["adopter_count"] == ADOPTER_COUNTS[1]}
    paired_diffs = {}
    for count in ADOPTER_COUNTS[2:]:
        line = f"    {count:>8}"
        vals = []
        for sd in E72_SEEDS:
            cur = next((r["adopter_pnl_bp_mean"] for r in rows
                        if r["adopter_count"] == count and r["seed"] == sd), np.nan)
            d = cur - base_by_seed[sd]
            vals.append(d)
            line += f"{d:>+12.3f}"
        paired_diffs[count] = [float(v) for v in vals]
        line += f"{float(np.nanmean(vals)):>+11.3f}"
        print(line)
    out["e7_2"] = {"rows": rows, "summary": e72, "spearman_rho": rho,
                   "spearman_p": p, "pass": e72_pass,
                   "saturated": saturated, "min_pnl_bp": sat,
                   "verdict": verdict72,
                   "paired_diffs": paired_diffs,
                   "base_count": ADOPTER_COUNTS[1]}

    # ==================================================================
    # E7.3  衰减曲线拟合
    # ==================================================================
    print(f"\n【E7.3】衰减曲线拟合（与真实 carry 衰减对照）")
    fit = _fit_decay(xs, ys)
    # ⚠️ "保留比例"只在 y 为正时有意义。采用者亏光时 y ≈ −10000，
    # 算出来的比值是 **102.3%** —— 一个看起来"几乎没衰减"的荒谬数字，
    # 而它只是因为 −9992/−9772 恰好略大于 1。
    # 这种数字比没有更糟：读者会拿它和真实 carry 的 34.7% 去比。
    if ys and np.isfinite(ys[0]) and ys[0] > 0 and np.isfinite(ys[-1]):
        retained = ys[-1] / ys[0]
        retained_ok = True
    else:
        retained = float("nan")
        retained_ok = False
    print(f"    真实对照：BTC 资金费率 carry 年化 {REAL_CARRY_START:.1%} → "
          f"{REAL_CARRY_END:.1%}，保留比例 {REAL_CARRY_RETAINED:.1%}")
    if retained_ok:
        print(f"    本实验：采用者 {xs[0]} → {xs[-1]} 时，平均 PnL "
              f"{ys[0]:.3f} → {ys[-1]:.3f} bp，保留比例 {retained:.1%}")
    else:
        print(f"    本实验：采用者 {xs[0]} → {xs[-1]} 时，平均 PnL "
              f"{ys[0]:.1f} → {ys[-1]:.1f} bp。")
        print("    ⚠️ **保留比例不适用**：PnL 为负（已亏光），"
              "比值没有经济含义。")
        print("       实测 102.3% 这种数字只是 −9992/−9772 恰好略大于 1，"
              "不能与真实 carry 的 34.7% 相提并论。")
    if fit.get("ok"):
        print(f"    指数拟合：衰减率 {fit['exponential_decay_rate']:.5f}/人  "
              f"R²={fit['exponential_r2']:.4f}")
        print(f"    幂律拟合：指数 {fit['powerlaw_exponent']:+.4f}  "
              f"R²={fit['powerlaw_r2']:.4f}")
        print(f"    → 更像：{'指数衰减' if fit.get('better') == 'exponential' else '幂律衰减'}")
    else:
        print(f"    → 拟合被拒：{fit.get('reason')}")
    print(f"    ⚠️ 不要求与真实数据精确吻合（不同系统没有理由数值相同），")
    print(f"       只要求**方向性质相同**（单调递减），并如实报量级差异。")
    mono = all(ys[i + 1] <= ys[i] + 1e-9 for i in range(len(ys) - 1))
    print(f"    单调递减：{'是' if mono else '否'}")
    out["e7_3"] = {"fit": fit, "retained_ratio": float(retained),
                   "retained_ratio_applicable": bool(retained_ok),
                   "real_retained_ratio": REAL_CARRY_RETAINED,
                   "monotone_decreasing": bool(mono)}

    # ==================================================================
    # E7.4  市场特征反馈
    # ==================================================================
    print(f"\n【E7.4】市场特征是否随拥挤度变化")
    print(f"    {'采用者':>8}{'σ(bp)':>10}{'超额峰度':>11}{'|r|ACF1':>11}"
          f"{'r ACF1':>11}{'总成交数':>11}")
    for r in e72:
        print(f"    {r['adopter_count']:>8}{r['market_sigma_bp']:>10.2f}"
              f"{r['market_kurtosis']:>11.3f}{r['market_abs_acf1']:>11.4f}"
              f"{r['market_ret_acf1']:>11.4f}{r['n_trades']:>11.0f}")
    for key, label in (("market_sigma_bp", "σ"), ("market_kurtosis", "超额峰度"),
                       ("market_abs_acf1", "|r|ACF1")):
        xv = [r["adopter_count"] for r in e72]
        yv = [r[key] for r in e72]
        rr, pp = spearman(xv, yv)
        scope = "全部档位" if label != "σ" else "全部档位"
        print(f"    Spearman(count, {label}) = {rr:+.4f}  p={pp:.4f}  （{scope}）")
    print(f"    ⚠️ 即使结果是「没有显著变化」，本身也是有信息量的：")
    print(f"       说明反身性效应**局限在策略自身的收益层面，没有外溢到整体市场结构**。")
    print(f"       不允许为了让结论好看而去调窗口或挑指标。")
    out["e7_4"] = e72

    # ==================================================================
    _plot_q_trajectory(e71, FIG / "stage7_q_trajectory.png")
    _plot_reflexivity(e72, out["e7_3"], FIG / "stage7_reflexivity.png")
    _plot_market_feedback(e72, FIG / "stage7_market_feedback.png")
    print(f"\n  图已输出到 {FIG}")

    # ==================================================================
    print("\n" + "=" * 74)
    print("阶段7 「诚实边界」小结")
    print("=" * 74)
    lines = [
        "【做到了什么量级】",
        f"  · bandit 学习：{E71_N_LEARNERS} 个主体 × {E71_TICKS} tick × "
        f"{len(E71_SEEDS)} 种子，共 {sum(x['n_updates'] for x in e71)} 次 Q 更新",
        f"  · 收敛判定：{'收敛到同一个窗口' if converged_same else '**未收敛**（各种子结论不同）'}"
        f"，各种子学到的窗口集合 {sorted(uniq)}",
        f"  · 反身性：Spearman(count, 平均PnL) = {rho:+.4f}（p={p:.5f}）",
        f"  · 采用者 PnL 从 {ys[0]:.3f}bp（{xs[0]} 人）降到 {ys[-1]:.3f}bp（{xs[-1]} 人），"
        f"保留比例 {retained:.1%}",
        f"  · 真实对照：carry 年化保留比例 {REAL_CARRY_RETAINED:.1%}",
        "",
        "【做不到什么】",
        "  · ⭐⭐ **E7.2 的「通过」是假阳性——这是本阶段最重要的一条。**",
        "    实测所有档位的采用者平均 PnL 都在 **−9772 ~ −9992bp** 之间，",
        "    也就是**基本亏光**。而 realized_pnl_bp 的下限就是 −10000bp，",
        "    所以指标已经**饱和**：Spearman 测到的是「亏损深度」，",
        "    不是「拥挤侵蚀」。ρ = −1.0、p = 0 看起来完美，但那个显著性",
        "    来自「都亏光了」，与反身性无关。",
        "    脚本现在会**自动检测饱和并把它标为假阳性**，不计入通过。",
        "    要把这个实验做成真的，必须先让被采用的策略有**正期望**——",
        "    而本模型里动量信号在所有窗口上的夏普都是负的（见 E7.0），",
        "    所以应当改用均值回归方向。**本阶段没有做这个替换。**",
        "  · ⭐ **「已知有效的策略」这个前提在本模型里不成立。**",
        f"    E7.0 的 horizon 体检显示：动量信号在所有窗口上的夏普都是**负的**"
        f"（{best_h['sharpe']:.4f} ~ {min(r['sharpe'] for r in hc):.4f}），"
        "最好的也只是「最不差」。",
        "    也就是说，本模型里的动量交易者是**负期望**的，",
        "    采用者采用的是一个**亏钱**的策略。",
        "    后果：E7.2 测到的「人多了 PnL 更差」仍然成立（机制本身没问题），",
        "    但它的**基线水平**是负的，与真实世界的「正 alpha 被拥挤侵蚀」",
        "    不是同一个起点。要让这条实验对上真实场景，应当改用**正期望**的策略",
        "    （本模型里是均值回归方向，夏普为正）——**本阶段没有做这个替换**，",
        "    如实记录在此。",
        "  · **采用者的 PnL 里含有一块来自「替换背景主体」的效应，而不只是拥挤。**",
        "    替换设计下背景噪音交易者从 300 降到 140（count=160 时），",
        "    市场的订单流结构同时变了。两种效应在数据里是**混在一起**的，",
        "    本阶段没有办法把它们分开——要分开需要再跑一组"
        "「不替换、只叠加」的对照，",
        "    那组又有市场变厚的混淆。**这是一个已知的、无法用单组实验解决的识别问题。**",
        "  · bandit 评估的是**信号质量**（mid-to-mid），不含执行成本。",
        "    所以它测不到「拥挤 → 成交变难 → 实际收益更差」这条渠道。",
        "    真实的反身性衰减里这条渠道占比可能很大（真实 carry 衰减的一部分",
        "    就是来自「做得人多了，能拿到的量变少/成本变高」）。",
        "  · 学习主体的候选集合只有 4 个 lookback（10/20/50/100），",
        "    策略空间极小。真实交易者的策略空间要大得多，",
        "    也因此**更能**适应拥挤——本模型学到的'最优'更容易失效。",
        "  · 没有手续费、没有资金成本，所以「策略根本不赚钱」与「赚钱但被成本吃掉」",
        "    在本模型里无法区分。",
        "",
        "【下个阶段依赖它的哪个假设】",
        "  · 阶段10（统一校准）如果要把「学习参数」纳入校准，需要依赖 E7.1 的",
        "    结论：**未收敛**时「最优 learning_rate」没有定义（最优值本身在移动），",
        "    此时应当固定学习参数、只校准市场参数。",
        "  · 阶段11 的交付报告需要把「反身性衰减已被观察到」与",
        "    「衰减的**原因**未被识别」分开表述——前者是数据，后者是解释。",
        "",
        "【一条必须写进结论的方法论发现】",
        "  · **「反身性」在仿真里极容易被做成同义反复。**",
        "    只要让采用者的收益依赖于市场，而市场又依赖于采用者的行为，",
        "    任何价格冲击都会产生「人多了收益变差」这个结果。",
        "    要让它成为一个**可证伪**的命题，必须显式给出：",
        "    ① 被测对象不变（本阶段用固定策略主体）；",
        "    ② 除拥挤度外其余逐点相同（同种子 + 同禀赋 + 总主体数固定）；",
        "    ③ 报告**未能识别**的部分（见上面的第一/第二条）。",
        "    缺少任何一条，「观察到衰减」这个结论都是不可靠的。",
    ]
    for ln in lines:
        print(ln)
    out["honest_boundary"] = lines
    out["elapsed_sec"] = time.time() - t0
    save_json(out, "stage7_metrics.json")
    print(f"\n  总用时 {out['elapsed_sec']:.0f}s")
    return out


# ----------------------------------------------------------------------
def _plot_q_trajectory(e71: list[dict], path) -> None:
    import matplotlib.pyplot as plt

    from tw.viz import C_ACCENT, C_GOOD, MUTED

    lbs = sorted({int(k) for k in e71[0]["final_q"]})
    colors = [C_ACCENT, C_GOOD, "#5aa9e6", "#e0a458", "#b48ead", "#8b95a5"]
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    for i, lb in enumerate(lbs):
        for j, r in enumerate(e71):
            xs = [p["tick"] for p in r["trajectory"]]
            ys = [p[f"q{lb}"] for p in r["trajectory"]]
            ax.plot(xs, ys, color=colors[i % len(colors)], lw=1.2,
                    alpha=0.75 if j == 0 else 0.35,
                    label=f"lookback={lb}（{len(e71)} 个种子）" if j == 0 else None)
    ax.axhline(0.0, color=MUTED, lw=0.9, ls=":")
    ax.set_xlabel("tick")
    ax.set_ylabel("Q 值（全体学习主体的均值）")
    ax.set_title("E7.1 Q 值轨迹：收敛 还是 持续震荡（后者本身就是反身性的证据）")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_reflexivity(e72: list[dict], e73: dict, path) -> None:
    import matplotlib.pyplot as plt

    from tw.viz import C_ACCENT, C_GOOD, C_REAL, C_SIM, MUTED

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6))
    rows = [r for r in e72 if r["adopter_count"] > 0]
    x = [r["adopter_count"] for r in rows]
    y = [r["pnl_bp_mean"] for r in rows]
    e = [r["pnl_bp_sd"] for r in rows]

    ax = axes[0]
    ax.errorbar(x, y, yerr=e, marker="o", color=C_ACCENT, lw=1.9, ms=8,
                capsize=4, label="采用者平均 PnL")
    ax.axhline(0.0, color=MUTED, lw=0.9, ls=":")
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x])
    fitr = e73.get("fit", {})
    if fitr.get("ok"):
        ls = np.array(x, dtype=float)
        if fitr.get("better") == "powerlaw":
            c = fitr["powerlaw_exponent"]
            a = np.exp(np.log(y).mean() - c * np.log(ls).mean())
            ax.plot(ls, a * ls ** c, "--", color=C_GOOD, lw=1.5,
                    label=f"幂律拟合 c={c:+.3f}")
        else:
            b = fitr["exponential_decay_rate"]
            a = np.exp(np.log(y).mean() + b * ls.mean())
            ax.plot(ls, a * np.exp(-b * ls), "--", color=C_GOOD, lw=1.5,
                    label=f"指数拟合 λ={b:.4f}/人")
    ax.set_xlabel("采用者数量（对数刻度）")
    ax.set_ylabel("采用者平均 PnL (bp)")
    ax.set_title("① 反身性：人越多，同一个策略越不灵")
    ax.legend(fontsize=9)

    ax = axes[1]
    norm_sim = np.array(y) / y[0]
    xs = np.array(x, dtype=float)
    ax.plot(xs / xs.min(), norm_sim, "o-", color=C_ACCENT, lw=2.0, ms=8,
            label="本实验（归一化）")
    ax.axhline(1.0, color=MUTED, lw=0.9, ls=":")
    ax.axhline(e73.get("real_retained_ratio", np.nan), color=C_REAL, ls="--",
               lw=1.6, label=f"真实 carry 保留比例 {e73.get('real_retained_ratio', float('nan')):.1%}")
    ax.set_xscale("log")
    ax.set_xlabel("采用者数量 / 最小档（对数刻度）")
    ax.set_ylabel("PnL 保留比例")
    ax.set_title("② 与真实 carry 衰减的量级对照")
    ax.legend(fontsize=9)
    fig.suptitle("E7.2/E7.3 反身性衰减（同种子配对；总主体数固定）", fontsize=12.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_market_feedback(e72: list[dict], path) -> None:
    import matplotlib.pyplot as plt

    from tw.viz import C_ACCENT, C_GOOD, C_SIM, MUTED

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0))
    x = [r["adopter_count"] for r in e72]
    for ax, key, label in ((axes[0], "market_sigma_bp", "σ (bp)"),
                           (axes[1], "market_kurtosis", "超额峰度"),
                           (axes[2], "market_abs_acf1", "|r| ACF(1)")):
        ax.plot(x, [r[key] for r in e72], "o-", color=C_SIM, lw=1.8, ms=7)
        ax.set_xlabel("采用者数量")
        ax.set_ylabel(label)
        ax.set_title(label)
    fig.suptitle("E7.4 市场整体特征是否随拥挤度变化（若几乎不变，说明反身性未外溢）",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
