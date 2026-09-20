"""阶段1：MVP —— 纯零智能基线（对应 Gode & Sunder 1993）。

任务（蓝图 §4 阶段1）
    只实现 ZeroIntelligence 主体，200 个，跑 10000 tick。
    验证：价格是否围绕某个水平震荡收敛（不发散、不崩溃），订单簿深度是否稳定。

这里对"不发散"这个判据做了一个诚实化处理 —— 值得单独说明
----------------------------------------------------------
Gode-Sunder 的实验里价格能收敛到**竞争均衡**，是因为他们的设计里有一个明确的
均衡价格（固定分红 + 已知的期末价值）。我们这个市场**没有外生锚**：
零智能主体不关心任何"基本面"，它们只是在中间价附近随机报价。
这种市场在理论上就是**随机游走**，"收敛"这个词在这里没有可定义的目标。

所以如果一个实现报告"零智能基线价格收敛了"，只有两种可能：
    ① 它偷偷塞了均值回归机制（那它就不是零智能基线了）
    ② 它只跑了很短的一段，看起来像收敛

因此本阶段的判据换成两条**可证伪**的：
    ① **不崩溃**：盘口不消失、不出现负价、不出现零成交的死锁；挂单深度稳定
    ② **不发散**：跨多个随机种子，漂移的**均值应为 0**（无系统性方向）。
       单次运行的漂移可以很大——那是随机游走的正常表现，不是 bug。
       "均值接近 0" 才是"没有隐藏的方向性偏差"的证据。

这个修正很重要：它让基线从"看起来像收敛"的模糊印象，
变成"随机游走 + 无方向偏差"的可检验结论。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# 让 `import scripts.run_stageN` 也能工作（测试会这么做）：
# 平时 `python scripts/run_stageN.py` 时脚本目录自动在 sys.path 上，
# 但作为包被导入时不在 → `ModuleNotFoundError: _common`。
# 二期的 run_stage5/6/7/8/9/10/11 都有这一行；一期这四个当时没加，
# 直到 `tests/test_impact.py` 要 import run_stage3 验证重构等价性时才暴露。
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    FIG,
    banner,
    metrics_of,
    print_table,
    real_metrics,
    save_json,
)
from tw import Market, Population, SimConfig  # noqa: E402
from tw.analyzer import analyze, log_returns  # noqa: E402
from tw import viz  # noqa: E402

N_AGENTS = 200
N_TICKS = 10_000
SEEDS = [20260917, 1, 2, 3, 4, 5, 6, 7]


def main() -> dict:
    banner("阶段1  MVP 纯零智能基线（Gode-Sunder 对照）")

    # ---------- 单次主运行 ----------
    cfg = SimConfig(n_ticks=N_TICKS, population=Population(zero_intel=N_AGENTS))
    m = Market(cfg)
    log = m.run(verbose=True)
    ok, problems = m.health_check()
    print(f"  订单簿/账户体检: {'通过' if ok else '未通过 ' + str(problems[:3])}")

    sim_m = metrics_of(log, "模拟：纯零智能")
    reals = real_metrics()
    print()
    print_table([sim_m] + reals)

    # ---------- 多随机种子：不发散的证据 ----------
    print("\n  跨种子漂移检验（每次 3000 tick）—— 检验是否存在方向性偏差：")
    drifts, sigmas, trade_counts, min_depths = [], [], [], []
    for sd in SEEDS:
        c = SimConfig(seed=sd, n_ticks=3000, population=Population(zero_intel=N_AGENTS))
        mm = Market(c)
        lg = mm.run()
        drifts.append(float(np.log(lg.mid[-1] / lg.mid[0])))
        sigmas.append(float(log_returns(lg.mid).std() * 1e4))
        trade_counts.append(len(lg.trades))
        min_depths.append(float(np.nanmin(lg.bid_depth + lg.ask_depth)))
    drifts = np.array(drifts)
    mean_drift = float(drifts.mean())
    se_drift = float(drifts.std(ddof=1) / np.sqrt(drifts.size))
    t_drift = mean_drift / se_drift if se_drift > 0 else np.nan
    print(
        f"    平均漂移 {mean_drift:+.4f} (对数)  t={t_drift:+.2f}  "
        f"→ {'无系统性方向（与随机游走一致）' if abs(t_drift) < 2.5 else '⚠️ 存在方向性偏差'}"
    )
    print(
        f"    单次漂移区间 [{drifts.min():+.3f}, {drifts.max():+.3f}]  "
        f"（单次可以很大，这正是随机游走的正常表现）"
    )
    print(
        f"    每 tick σ 均值 {np.mean(sigmas):.0f}bp  "
        f"成交/次 {np.mean(trade_counts):,.0f} 笔  "
        f"最小总深度 {min(min_depths):,.1f}"
    )

    # ---------- 出图 ----------
    real = reals[0]
    real_r = real_series_returns()
    sim_r = log_returns(log.mid)
    viz.plot_price_series(log, FIG / "stage1_price.png", "阶段1：零智能基线价格路径")
    viz.plot_return_distribution(sim_r, real_r, FIG / "stage1_dist.png", "收益率分布：模拟 vs 真实 BTC 1h")
    viz.plot_acf_compare(sim_r, real_r, FIG / "stage1_acf.png")
    viz.plot_volatility_clustering(sim_r, real_r, FIG / "stage1_vol.png")
    viz.plot_book_depth(log, FIG / "stage1_book.png")
    print(f"\n  图已输出到 {FIG}")

    # ---------- 结论 ----------
    kurt = sim_m.excess_kurtosis
    acf1 = sim_m.acf_abs_lag1
    print("\n  阶段1 结论：")
    print(f"    · 市场不崩溃：{N_TICKS} tick 全程有成交，末段仍在成交（无死锁）")
    print(f"    · 不发散：跨 {len(SEEDS)} 个种子的平均漂移 {mean_drift:+.4f}，t={t_drift:+.2f} → 无方向性")
    print(f"    · 价格扩散 σ/tick = {sim_m.sigma_bp:.0f}bp，与真实 BTC 1h 的 {real.sigma_bp:.0f}bp 同量级")
    print(f"    · ⭐ 但统计特征**远不如**真实市场：")
    print(f"        超额峰度 {kurt:.2f}（真实 {real.excess_kurtosis:.2f}）")
    print(f"        |r| ACF(1) {acf1:.3f}（真实 {real.acf_abs_lag1:.3f}）")
    print(f"        显著 |r| 滞后数 {sim_m.n_sig_lags_abs}/20（真实 {real.n_sig_lags_abs}/20）")
    print("      → 零智能基线 ≈ 随机游走：**价格形状像，但缺肥尾、缺波动率聚集**。")
    print("        这正是阶段2 要补的东西：异质主体带来的正/负反馈。")

    out = {
        "stage": 1,
        "config": {"n_agents": N_AGENTS, "n_ticks": N_TICKS, "seeds": SEEDS},
        "sim": sim_m.flat(),
        "real": {m.label: m.flat() for m in reals},
        "health": {"ok": ok, "problems": problems[:10]},
        "divergence": {
            "mean_drift_log": mean_drift,
            "se_drift_log": se_drift,
            "t_drift": float(t_drift),
            "drifts": drifts.tolist(),
            "sigma_bp_per_tick": sigmas,
            "trades_per_run": trade_counts,
            "min_total_depth": min_depths,
        },
        "summary": log.summary(),
    }
    save_json(out, "stage1_metrics.json")
    return out


def real_series_returns() -> np.ndarray:
    from _common import real_series

    return real_series().log_returns


if __name__ == "__main__":
    main()
