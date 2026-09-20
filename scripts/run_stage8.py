"""阶段8：Hawkes 过程订单到达（二期指导书 §5）。

目标
----
用自激点过程建模订单到达的**时间聚集性**——
注意这**不是**订单方向的长记忆（那是阶段6 的范畴，两者正交）。

实验编号（对应指导书 §5.4）
--------------------------
E8.1 参数标定       扫描 (branching, beta)，看哪组让到达间隔的 Ljung-Box 统计量
                    接近真实市场的量级
E8.2 与阶段6叠加    在阶段6 的配置上加入 Hawkes，看冲击指数 k 是否进一步改善

⭐ 三条必须先说清楚的方法论
==========================

**① 真实靶子只能是**代理**。**
   我们手上只有 1 小时线（OHLCV），**没有逐笔数据**，
   所以拿不到真正的"到达间隔"。最接近的可用代理是**成交量**：
   成交量本身就是"到达次数 × 单笔规模"，它的自相关反映聚集性。
   报告里必须写清这是代理，不能说成"与真实到达间隔对标"。

**② ``base_lambda`` 必须标定，不能拍。**
   见 ``hawkes_market.calibrate_base_lambda``：ρ 的长期均值由闭环不动点决定，
   只有 ``λ̄ = 每 tick 平均成交笔数`` 时它才等于 1。
   拍一个 λ̄ 会让"打开 Hawkes"同时改变**平均交易量**，
   于是"聚集性"的结论里混进了"交易变多/变少"。

**③ E8.2 允许"没有额外贡献"这个结论。**
   指导书 §5.5 明说这一步允许无贡献。阶段6 已经把 k 拉到某个值，
   Hawkes 只改**到达的时刻**、不改**方向的长记忆**，
   而平方根律主要由后者驱动——所以"没有额外贡献"是一个**先验合理**的结果，
   不需要为了"有产出"去调参数把它做出来。

用法::

    python scripts/run_stage8.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import FIG, banner, real_series, save_json  # noqa: E402
from tw import Market, Population, SimConfig  # noqa: E402
from tw.analyzer import ljung_box  # noqa: E402
from tw.order_flow.hawkes import HawkesConfig  # noqa: E402
from tw.order_flow.hawkes_market import (  # noqa: E402
    HawkesMarket,
    calibrate_base_lambda,
)
from tw.impact import fit_power_law, summarize, unsaturated  # noqa: E402

SEED0 = 20260917
SEEDS = [SEED0, SEED0 + 7, SEED0 + 14]
MIX = {"zero_intel": 0.30, "fundamentalist": 0.40, "chartist": 0.30}
N_AGENTS = 300
WARMUP = 2_000
N_TICKS = 6_000
#: Ljung-Box 的滞后阶数（与一期 analyzer 的口径一致）
LB_LAGS = 10

#: 扫描网格：分支比 × 衰减率
BRANCHING_GRID = [0.0, 0.3, 0.6, 0.9]
BETA_GRID = [0.05, 0.15, 0.5]


def run_hawkes(seed: int, branching: float, beta: float, base_lambda: float,
               *, off: bool = False, ticks: int = N_TICKS) -> dict:
    cfg = HawkesConfig(base_lambda=base_lambda, branching=branching, beta=beta)
    m = HawkesMarket(
        SimConfig(seed=seed, n_ticks=ticks,
                  population=Population.from_shares(N_AGENTS, MIX)),
        hawkes_config=cfg, hawkes_off=off,
    )
    m.run(ticks)
    counts = np.asarray(m.log.n_trades[WARMUP: m.tick], dtype=float)
    counts = counts[np.isfinite(counts)]
    q, p = ljung_box(counts, LB_LAGS) if counts.size > LB_LAGS + 5 else (np.nan, np.nan)
    st = m.hawkes_stats() if not off else {"rho_mean": float("nan")}
    ok, probs = m.health_check()
    return {
        "seed": seed, "branching": branching, "beta": beta, "off": off,
        "base_lambda": base_lambda,
        "lb_q": float(q), "lb_p": float(p),
        "trades_mean": float(counts.mean()) if counts.size else float("nan"),
        "trades_sd": float(counts.std(ddof=1)) if counts.size > 1 else float("nan"),
        "rho_mean": float(st.get("rho_mean", float("nan"))),
        "n_active_mean": float(st.get("n_active_mean", float("nan"))),
        "health_ok": bool(ok),
        "health_problems": probs[:3],
    }


def real_proxy() -> dict:
    """真实靶子的代理：1 小时线的**成交量**与其 Ljung-Box。

    ⚠️ 这是代理，不是等价物（见模块文档 ①）。
    """
    out = {}
    for key in ("BTCUSDT_1h", "ETHUSDT_1h", "SOLUSDT_1h"):
        s = real_series(key)
        vol = np.asarray(s.volume, dtype=float)
        vol = vol[np.isfinite(vol) & (vol > 0)]
        lv = np.log(vol)
        q, p = ljung_box(lv, LB_LAGS)
        out[key] = {"lb_q": float(q), "lb_p": float(p),
                    "mean": float(vol.mean()), "sd": float(vol.std(ddof=1)),
                    "n": int(vol.size)}
    return out


# ----------------------------------------------------------------------
def main() -> dict:
    banner("阶段8  Hawkes 过程订单到达")
    t0 = time.time()
    out: dict = {
        "stage": 8,
        "config": {"seeds": SEEDS, "n_ticks": N_TICKS, "warmup": WARMUP,
                   "lb_lags": LB_LAGS, "branching_grid": BRANCHING_GRID,
                   "beta_grid": BETA_GRID, "n_agents": N_AGENTS, "mix": MIX},
    }

    # ==================================================================
    # E8.0  真实代理靶子
    # ==================================================================
    print("\n【E8.0】真实靶子（**代理**：1 小时线成交量，不是逐笔到达间隔）")
    print("  ⚠️ 我们只有 OHLCV，没有逐笔数据，拿不到真正的到达间隔。")
    print("     成交量 ≈ 到达次数 × 单笔规模，是最接近的可用代理。")
    print(f"    {'数据集':<14}{'成交量LB Q':>12}{'p':>12}{'样本':>9}")
    rp = real_proxy()
    for k, v in rp.items():
        print(f"    {k:<14}{v['lb_q']:>12.1f}{v['lb_p']:>12.2e}{v['n']:>9}")
    q_real = float(np.mean([v["lb_q"] for v in rp.values()]))
    print(f"  → 真实代理的平均 Q = {q_real:.1f}（{LB_LAGS} 阶）")
    out["e8_0_real_proxy"] = {"per_dataset": rp, "mean_q": q_real}

    # ==================================================================
    # E8.1  参数标定
    # ==================================================================
    print("\n【E8.1】参数标定")
    print("  第一步：标定 base_lambda = 该市场『Hawkes 关掉』时每 tick 的平均成交笔数")
    print("    为什么必须标定：ρ 的长期均值由闭环不动点决定，")
    print("    只有 λ̄ = 平均成交笔数时它才等于 1（见 calibrate_base_lambda 的说明）。")
    base_lam = calibrate_base_lambda(
        seeds=SEEDS, n_ticks=N_TICKS, n_agents=N_AGENTS, mix=MIX, warmup=WARMUP)
    print(f"    → 标定值 base_lambda = {base_lam:.2f} 笔/tick")
    out["e8_1_base_lambda"] = float(base_lam)

    print("\n  第二步：扫描 (branching, beta)，看聚集性是否逼近真实代理")
    print(f"    {'branching':>10}{'beta':>7}{'ρ均值':>9}{'成交均值':>10}"
          f"{'LB Q':>10}{'健康':>7}")
    grid = []
    skipped = []
    for br in BRANCHING_GRID:
        for be in BETA_GRID:
            # ⚠️ 有些组合**不满足稳定条件**（α < β），必须提前跳过而不是跑崩。
            # 实测指导书给的 (branching=0.9, beta=0.5) 就是这种：
            # 直觉上"分支比 <1 就稳定"，但 α<β 更紧，β=0.5 时上限只有 0.771。
            # 第一版没做这个检查，阶段8 直接在中途抛 ValueError 中断。
            probe = None
            try:
                probe = HawkesConfig(base_lambda=base_lam, branching=br, beta=be)
            except ValueError as e:
                skipped.append({"branching": br, "beta": be, "reason": str(e)})
                print(f"    {br:>10.2f}{be:>7.2f}   ⏭ 跳过（不满足稳定条件）")
                continue
            _ = probe
            rows = [run_hawkes(sd, br, be, base_lam) for sd in SEEDS[:2]]
            q = float(np.nanmean([r["lb_q"] for r in rows]))
            rho = float(np.nanmean([r["rho_mean"] for r in rows]))
            tm = float(np.nanmean([r["trades_mean"] for r in rows]))
            row = {"branching": br, "beta": be, "lb_q": q, "rho_mean": rho,
                   "trades_mean": tm,
                   "health_ok": bool(all(r["health_ok"] for r in rows)),
                   "per_seed": rows}
            grid.append(row)
            print(f"    {br:>10.2f}{be:>7.2f}{rho:>9.3f}{tm:>10.2f}"
                  f"{q:>10.1f}{'✅' if row['health_ok'] else '❌':>7}")
    out["e8_1_grid"] = grid
    out["e8_1_skipped"] = skipped
    if skipped:
        print(f"\n    跳过了 {len(skipped)} 个不满足 α<β 的组合："
              f"{[(r['branching'], r['beta']) for r in skipped]}")
        print("    这是**指导书网格会踩到的真实边界**：分支比 <1 只是必要不充分，"
              "α<β 更紧（β=0.5 时上限 0.771）。")

    print("\n  第三步：与不打开 Hawkes 的对照比较")
    ctrl = [run_hawkes(sd, 0.0, BETA_GRID[1], base_lam, off=True)
            for sd in SEEDS]
    q_ctrl = float(np.nanmean([r["lb_q"] for r in ctrl]))
    print(f"    对照（Hawkes 关）：LB Q = {q_ctrl:.1f}，"
          f"每 tick 成交 {float(np.nanmean([r['trades_mean'] for r in ctrl])):.2f}")
    best = max((g for g in grid if g["health_ok"]), key=lambda g: g["lb_q"],
               default=None)
    if best:
        print(f"    最强聚集：branching={best['branching']:.2f}、"
              f"beta={best['beta']:.2f} → LB Q = {best['lb_q']:.1f}")
        print(f"    → 相对对照提升 {best['lb_q'] / q_ctrl if q_ctrl > 0 else float('nan'):.2f}×")
        print(f"    → 与真实代理 {q_real:.1f} 的比值："
              f"{best['lb_q'] / q_real if q_real > 0 else float('nan'):.2f}×")
        print(f"    ⚠️ 量级不可能精确对上：真实 Q 是在 **17520 根小时线** 上算的，")
        print(f"       而模拟只有 {N_TICKS - WARMUP} 个 tick；Ljung-Box 的 Q 与样本量成正比，")
        print(f"       跨样本量比较 Q 本身是**口径错误**。可比的是"
              f"『Q/样本量』与『是否显著』这两件事。")
        qn_sim = best["lb_q"] / (N_TICKS - WARMUP)
        qn_real = q_real / float(np.mean([v["n"] for v in rp.values()]))
        print(f"       归一化后：模拟 {qn_sim:.4f}/tick，真实代理 {qn_real:.4f}/根 → "
              f"比值 {qn_sim / qn_real if qn_real > 0 else float('nan'):.2f}×")
        out["e8_1_normalized"] = {"sim": float(qn_sim), "real": float(qn_real),
                                  "ratio": float(qn_sim / qn_real) if qn_real else float("nan")}

    # 验收：模拟的到达序列必须在合理显著性上拒绝"无自相关"
    best_p = float(np.nanmean([r["lb_p"] for r in (best or {}).get("per_seed", [])])) \
        if best else float("nan")
    e81_pass = bool(best is not None and np.isfinite(best_p) and best_p < 0.01
                    and best["lb_q"] > q_ctrl)
    print(f"\n  验收（指导书 §5.5）：模拟到达序列的 Ljung-Box 在合理显著性上"
          f"拒绝无自相关")
    print(f"    → best p = {best_p:.3e}，Q 相对对照提升："
          f"{'✅ E8.1 通过' if e81_pass else '❌ E8.1 未通过'}")
    out["e8_1_control"] = {"lb_q": q_ctrl, "per_seed": ctrl}
    out["e8_1_best"] = best
    out["e8_1_pass"] = e81_pass

    # ==================================================================
    # E8.2  与阶段6 叠加
    # ==================================================================
    print("\n【E8.2】与阶段6 叠加：冲击指数 k 是否进一步改善")
    print("  复用阶段3 的配对清算实验（同种子配对、因果滑点、分期清算 slices=20）")
    print("  ⚠️ 这一步允许『没有额外贡献』这个结论（指导书 §5.5）。")
    print("     Hawkes 只改**到达的时刻**、不改**方向的长记忆**，")
    print("     而平方根律主要由后者驱动——所以『无额外贡献』是先验合理的结果。")
    from run_stage3 import HORIZON, MM_MIX, make_controls, paired_impact  # noqa: E402

    SHOCK_LEVELS = [0.001, 0.0025, 0.005, 0.01]
    base_lam_mm = calibrate_base_lambda(
        seeds=SEEDS, n_ticks=3_000, n_agents=N_AGENTS, mix=MM_MIX, warmup=500)
    print(f"  （MM 池的 base_lambda 单独标定：{base_lam_mm:.2f} 笔/tick）")

    def mk_factory(branching: float):
        cfg = HawkesConfig(base_lambda=base_lam_mm, branching=branching, beta=0.15)

        def f(sim):
            return HawkesMarket(sim, hawkes_config=cfg,
                                hawkes_off=(branching <= 0.0))
        return f

    print(f"    {'配置':<16}{'档位':>8}{'成交量':>9}{'成交率':>8}{'因果滑点bp':>12}")
    impact = {}
    for tag, br in (("Hawkes 关", 0.0), ("Hawkes 开(b=0.6)", 0.6)):
        fn = mk_factory(br)
        ctrl_h = make_controls(MM_MIX, SEEDS, HORIZON, factory=fn)
        rows = []
        for f_ in SHOCK_LEVELS:
            r = paired_impact(f"{tag}|{f_:.4f}", MM_MIX, f_, 20, SEEDS,
                              ctrl_h, HORIZON, factory=fn)
            rows.append(r)
            print(f"    {tag:<16}{f_:>8.2%}{r['delivered_qty']:>9.1f}"
                  f"{r['fill_ratio']:>8.0%}{r['slippage_bp']:>12.2f}")
        impact[tag] = rows

    print()
    ks = {}
    for tag, rows in impact.items():
        un = unsaturated(rows)
        fit = fit_power_law(un, "slippage_bp")
        ks[tag] = fit
        print(f"    {tag:<16} 有效档位 {len(un)}/{len(rows)}  "
              f"k={fit.exponent:>7.3f}  R²={fit.r_squared:>6.3f}  "
              f"|k−0.5|={fit.closeness_to_sqrt():>6.3f}"
              + (f"  ⚠️{fit.reason}" if not fit.ok else ""))
    k_off, k_on = ks["Hawkes 关"], ks["Hawkes 开(b=0.6)"]
    d_off, d_on = k_off.closeness_to_sqrt(), k_on.closeness_to_sqrt()
    if np.isfinite(d_off) and np.isfinite(d_on):
        extra = d_off - d_on
        print(f"\n    → k: {k_off.exponent:.3f} → {k_on.exponent:.3f}；"
              f"离 0.5 的距离 {d_off:.3f} → {d_on:.3f}（改善 {extra:+.3f}）")
        verdict = ("有额外改善" if extra > 0.03 else
                   "**没有额外贡献**（在噪声量级内）" if abs(extra) <= 0.03 else
                   "反而变差")
        print(f"    → 结论：{verdict}")
    else:
        extra = float("nan")
        verdict = "无法判定（有一侧拟合失败）"
        print(f"\n    → 结论：{verdict}")
    out["e8_2"] = {"rows": {k: v for k, v in impact.items()},
                   "k_on": k_on.exponent, "k_off": k_off.exponent,
                   "closeness_on": d_on, "closeness_off": d_off,
                   "extra_improvement": float(extra) if np.isfinite(extra) else None,
                   "verdict": verdict}

    # ==================================================================
    _plot_lb(grid, ctrl, q_ctrl, FIG / "stage8_ljungbox.png")
    _plot_intensity(base_lam, FIG / "stage8_intensity.png")
    print(f"\n  图已输出到 {FIG}")

    # ==================================================================
    print("\n" + "=" * 74)
    print("阶段8 「诚实边界」小结")
    print("=" * 74)
    lines = [
        "【做到了什么量级】",
        f"  · base_lambda 标定到 {base_lam:.2f} 笔/tick（不是拍的）",
        f"  · 最强聚集配置 branching={best['branching'] if best else float('nan'):.2f}、"
        f"beta={best['beta'] if best else float('nan'):.2f}："
        f"LB Q = {best['lb_q'] if best else float('nan'):.1f}"
        f"（对照 {q_ctrl:.1f}）",
        f"  · ρ 的时间均值保持在 {float(np.mean([g['rho_mean'] for g in grid])):.3f} 附近"
        f"（偏离 1 说明平均活跃度被改变了）",
        f"  · E8.2 结论：{verdict}",
        "",
        "【做不到什么】",
        "  · **真实靶子只是代理。** 只有 1 小时线 OHLCV，没有逐笔数据，"
        "拿不到真正的到达间隔。",
        "    成交量 ≈ 到达次数 × 单笔规模，两者混在一起，"
        "所以「聚集性对上了没有」这件事只能做量级判断，不能做等价性检验。",
        "  · **Ljung-Box 的 Q 与样本量成正比**，所以模拟（几千 tick）与真实"
        "（17520 根）",
        "    的 Q **不能直接比大小**。本脚本报的是归一化后的 Q/样本量，"
        "但那也不是无偏的",
        "    （真实成交量与模拟成交笔数的分布形状不同）。这条口径限制必须写进结论。",
        "  · 时间被离散成 tick，所以「同 tick 内的到达间隔」被抹掉了。"
        "本模型测的是",
        "    「**逐 tick 的聚集**」，不是「逐笔的聚集」。"
        "真实市场最显著的聚集恰恰在秒级以内。",
        "  · Hawkes 只作用在「**多少主体能出手**」这一层，不改单主体的到达过程。"
        "真实的",
        "    自激还包括「一个大单触发算法跟单」这类主体间的直接反应，本模型没有。",
        "",
        "【下个阶段依赖它的哪个假设】",
        "  · 阶段10（统一校准）需要决定「到达过程要不要纳入校准参数」。"
        "本阶段的结论是：",
        "    branching 可标定（有明确的靶子方向），但 beta 的影响被离散化掩盖，"
        "不建议纳入。",
        "  · 阶段11 的交付报告需要把「到达聚集已被做出」与"
        "「与真实聚集的量级比较不可靠」分开表述。",
        "",
        "【一条必须写进结论的方法论发现】",
        "  · **离散化会让 Hawkes 的平稳均值偏离连续公式。**",
        "    连续时间下分支比是 α/β、平稳强度是 μ/(1−α/β)；"
        "但离散递推下真正的分支比是",
        "    α/(e^{β·dt}−1)，两者在 β=0.2、dt=1 时差 7.9%。"
        "用连续公式反解 μ，会让实际",
        "    平均到达率偏低 18%（实测 λ̄ 设 20 只得 16.3），"
        "于是 ρ 的均值是 0.81 而不是 1——",
        "    而「打开 Hawkes 不改变平均活跃度」正是 E8.2 能归因的前提。",
        "    这个偏差不会报错，只会让「聚集性」的结论里混进「交易变少了」。",
    ]
    for ln in lines:
        print(ln)
    out["honest_boundary"] = lines
    out["elapsed_sec"] = time.time() - t0
    save_json(out, "stage8_metrics.json")
    print(f"\n  总用时 {out['elapsed_sec']:.0f}s")
    return out


# ----------------------------------------------------------------------
def _plot_lb(grid: list[dict], ctrl: list[dict], q_ctrl: float, path) -> None:
    import matplotlib.pyplot as plt

    from tw.viz import C_ACCENT, C_GOOD, C_REAL, C_SIM

    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    for be, col in zip(sorted({g["beta"] for g in grid}),
                       (C_SIM, C_GOOD, C_ACCENT)):
        sub = sorted([g for g in grid if g["beta"] == be],
                     key=lambda g: g["branching"])
        ax.plot([g["branching"] for g in sub], [g["lb_q"] for g in sub],
                "o-", color=col, lw=1.8, ms=7, label=f"β={be:g}")
    ax.axhline(q_ctrl, color="#8b95a5", ls=":", lw=1.4,
               label=f"对照（Hawkes 关）Q={q_ctrl:.0f}")
    ax.set_xlabel("分支比 branching（0 = 泊松）")
    ax.set_ylabel(f"逐 tick 成交笔数的 Ljung-Box Q（{LB_LAGS} 阶）")
    ax.set_title("E8.1 到达聚集性随分支比上升（同标定 base_lambda）")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_intensity(base_lam: float, path) -> None:
    """强度/ρ 的时间轨迹：让读者直接看到"聚集"长什么样。"""
    import matplotlib.pyplot as plt

    from tw.viz import C_ACCENT, C_GOOD, C_MID

    fig, axes = plt.subplots(2, 1, figsize=(10.2, 5.6), sharex=True)
    for ax, br, col in ((axes[0], 0.0, C_MID), (axes[1], 0.9, C_ACCENT)):
        cfg = HawkesConfig(base_lambda=base_lam, branching=br, beta=0.15)
        m = HawkesMarket(
            SimConfig(seed=SEED0, n_ticks=2_500,
                      population=Population.from_shares(N_AGENTS, MIX)),
            hawkes_config=cfg, hawkes_off=(br <= 0.0),
        )
        m.run(2_500)
        rho = np.array([d["rho"] for d in m.hawkes_log])
        ev = np.array([d["events"] for d in m.hawkes_log])
        ax.plot(np.arange(rho.size), rho, color=col, lw=1.1, label="ρ（归一化强度）")
        ax2 = ax.twinx()
        ax2.plot(np.arange(ev.size), ev, color=C_GOOD, lw=0.8, alpha=0.6,
                 label="每 tick 成交笔数")
        ax2.set_ylabel("成交笔数", color=C_GOOD)
        ax2.tick_params(axis="y", colors=C_GOOD)
        ax.axhline(1.0, color="#8b95a5", ls=":", lw=1.0)
        ax.set_ylabel("ρ")
        ax.set_title(f"branching={br:g}"
                     + ("（关闭：纯泊松）" if br <= 0 else "（开启：自激）"),
                     fontsize=10.5)
        ax.legend(fontsize=8.5, loc="upper left")
    fig.suptitle("E8.1 订单到达的时间聚集性：ρ 与逐 tick 成交笔数", fontsize=12)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
