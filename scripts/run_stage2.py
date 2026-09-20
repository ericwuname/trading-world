"""阶段2：加入基本面派 + 图表派（对应 Lux & Marchesi）。

任务（蓝图 §4 阶段2）
    保留 30% 零智能作为"噪音底噪"，加入 40% Fundamentalist、30% Chartist。
    验证：峰度、波动率聚集、收益率自相关是否逼近真实 BTC。

本脚本在蓝图基础上加了一件事：**消融实验（ablation）**
--------------------------------------------------------
蓝图只要求跑最终配比。但如果最终配比"像了"，我们并不知道是**哪个成分**的功劳；
如果"不像"，也不知道该调哪个旋钮。所以这里额外跑四种主体池：

    ① 仅零智能        （阶段1 基线，作为参照）
    ② 零智能 + 基本面派（只有负反馈）
    ③ 零智能 + 图表派  （只有正反馈）
    ④ 三者混合        （完整模型）

理论预期（Lux-Marchesi 的逻辑）：
    · 只有负反馈 → 价格被摁死，收益接近正态，**没有肥尾**
    · 只有正反馈 → 价格指数爆炸，**方差无界**
    · 两者共存   → 正反馈制造大波动、负反馈时不时把它拉回来，
                   两者的**间歇性对抗**才是肥尾与波动率聚集的来源

四个池子跑完，"哪些机制是必需的"就是**可观测的实验结论**，而不是论文转述。
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
    relative_position,
    save_json,
)
from tw import Market, Population, SimConfig, viz  # noqa: E402
from tw.analyzer import analyze  # noqa: E402

N_AGENTS = 300
N_TICKS = 12_000
SEED = 20260917

#: 蓝图 §4 阶段的配比
MIX = {"zero_intel": 0.30, "fundamentalist": 0.40, "chartist": 0.30}

POOLS: list[tuple[str, dict[str, float]]] = [
    ("① 仅零智能", {"zero_intel": 1.0}),
    ("② 零智能+基本面派", {"zero_intel": 0.43, "fundamentalist": 0.57}),
    ("③ 零智能+图表派", {"zero_intel": 0.50, "chartist": 0.50}),
    ("④ 三者混合（蓝图配比）", MIX),
]


def build(mix: dict[str, float], n_ticks: int = N_TICKS, seed: int = SEED) -> tuple[Market, object]:
    pop = Population.from_shares(N_AGENTS, mix)
    cfg = SimConfig(seed=seed, n_ticks=n_ticks, population=pop)
    m = Market(cfg)
    log = m.run()
    return m, log


def main() -> dict:
    banner("阶段2  异质主体：基本面派 + 图表派（Lux-Marchesi）")

    reals = real_metrics()
    real = reals[0]
    print(f"  真实基准（{real.label}）：σ={real.sigma_bp:.1f}bp  "
          f"峰度={real.excess_kurtosis:.2f}  |r|ACF(1)={real.acf_abs_lag1:.3f}  "
          f"显著滞后={real.n_sig_lags_abs}/20")

    results: dict[str, dict] = {}
    metrics = []
    logs = {}
    for name, mix in POOLS:
        pop = Population.from_shares(N_AGENTS, mix)
        m, log = build(mix)
        ok, problems = m.health_check()
        mm = metrics_of(log, name)
        metrics.append(mm)
        logs[name] = log
        results[name] = {
            "mix": pop.shares(),
            "counts": {
                "zero_intel": pop.zero_intel,
                "fundamentalist": pop.fundamentalist,
                "chartist": pop.chartist,
                "market_maker": pop.market_maker,
            },
            "metrics": mm.flat(),
            "ratio_to_real": relative_position(mm.flat(), real.flat()),
            "health_ok": ok,
            "n_trades": len(log.trades),
            "price_first": float(log.mid[0]),
            "price_last": float(log.mid[-1]),
            "price_min": float(log.mid.min()),
            "price_max": float(log.mid.max()),
        }
        print(
            f"\n  {name}  N={pop.total}"
            f"(零{pop.zero_intel}/基{pop.fundamentalist}/图{pop.chartist})"
            f"  体检={'通过' if ok else '未过'}"
            f"  成交={len(log.trades):,}"
            f"  价格 {log.mid.min():,.0f}~{log.mid.max():,.0f}"
        )

    print("\n" + "=" * 74)
    print("四池对照（含真实基准）")
    print("=" * 74)
    print_table(metrics + reals)

    # ---------- 机制归因表 ----------
    print("\n  机制归因（相对真实 BTC 1h 的倍数，1.00 = 完全一致）：")
    hdr = f"    {'指标':<20}" + "".join(f"{n.split(' ')[0]:>12}" for n, _ in POOLS)
    print(hdr)
    for key, label in (
        ("excess_kurtosis", "超额峰度"),
        ("acf_abs_lag1", "|r| ACF(1)"),
        ("acf_abs_mean_1_10", "|r| ACF(1-10)"),
        ("n_sig_lags_abs", "显著滞后数"),
        ("sigma_bp", "σ/tick"),
    ):
        line = f"    {label:<20}"
        for name, _ in POOLS:
            r = results[name]["ratio_to_real"].get(key)
            line += f"{r:>11.2f}×" if r is not None else f"{'n/a':>12}"
        print(line)

    # ---------- 出图 ----------
    from _common import real_series

    real_r = real_series().log_returns
    best_name = max(
        POOLS, key=lambda p: min(
            results[p[0]]["ratio_to_real"].get("excess_kurtosis", 0) or 0,
            results[p[0]]["ratio_to_real"].get("acf_abs_lag1", 0) or 0,
        )
    )[0]
    best_log = logs[best_name]

    viz.plot_price_series(
        best_log, FIG / "stage2_price.png",
        f"阶段2：{best_name} 的价格路径（基本面价值为虚线）",
    )
    viz.plot_return_distribution(
        best_log.log_returns(), real_r, FIG / "stage2_dist.png",
        f"收益率分布：{best_name} vs 真实 BTC 1h",
    )
    viz.plot_acf_compare(best_log.log_returns(), real_r, FIG / "stage2_acf.png")
    viz.plot_volatility_clustering(best_log.log_returns(), real_r, FIG / "stage2_vol.png")
    viz.plot_metric_bars(
        results[best_name]["metrics"], real.flat(), FIG / "stage2_metric_bars.png",
        f"关键指标对照：{best_name} / 真实 BTC 1h",
    )
    # 四池的 |r| ACF 叠加对比
    fig_acf_pools(logs, real_r)
    print(f"\n  图已输出到 {FIG}")

    # ---------- 结论 ----------
    print("\n  阶段2 结论：")
    for name, _ in POOLS:
        r = results[name]["ratio_to_real"]
        print(
            f"    {name:<22} 峰度 {results[name]['metrics']['excess_kurtosis']:>6.2f}"
            f"（真实 {real.excess_kurtosis:.2f}，比 {r.get('excess_kurtosis', float('nan')):.2f}×）"
            f"  |r|ACF1 {results[name]['metrics']['acf_abs_lag1']:>5.3f}"
            f"（比 {r.get('acf_abs_lag1', float('nan')):.2f}×）"
        )
    _verdict(results, real)

    out = {
        "stage": 2,
        "config": {"n_agents": N_AGENTS, "n_ticks": N_TICKS, "seed": SEED, "mix": MIX},
        "pools": results,
        "real": {m.label: m.flat() for m in reals},
        "best_pool": best_name,
    }
    save_json(out, "stage2_metrics.json")
    return out


def fig_acf_pools(logs: dict, real_r: np.ndarray) -> None:
    """四个池子的 |r| 自相关叠加，一眼看出哪个成分带来了波动率聚集。"""
    import matplotlib.pyplot as plt

    from tw.analyzer import acf, acf_ci

    fig, ax = plt.subplots(figsize=(10, 4.2))
    lags = np.arange(1, 21)
    palette = ["#7e8ba3", "#8fbf6a", "#e06c75", "#5aa9e6"]
    for (name, log), c in zip(logs.items(), palette):
        r = log.log_returns()
        a = acf(np.abs(r), 20)[1:21]
        ax.plot(lags, a, marker="o", ms=3.5, lw=1.5, color=c, label=name)
    a = acf(np.abs(real_r), 20)[1:21]
    ax.plot(lags, a, marker="s", ms=4, lw=2.2, color="#e0a458", label="真实 BTC 1h")
    band = acf_ci(real_r.size)
    ax.axhspan(-band, band, color="#8b95a5", alpha=0.15)
    ax.axhline(0, color="#8b95a5", lw=0.7)
    ax.set_xlabel("滞后 (tick)")
    ax.set_ylabel("|收益率| 的自相关")
    ax.set_title("波动率聚集的来源：哪个主体池最接近真实市场？")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "stage2_acf_pools.png")
    plt.close(fig)


def _verdict(results: dict, real) -> None:
    kz = real.excess_kurtosis
    az = real.acf_abs_lag1
    got_k = [results[n]["metrics"]["excess_kurtosis"] for n, _ in POOLS]
    got_a = [results[n]["metrics"]["acf_abs_lag1"] for n, _ in POOLS]
    print()
    if max(got_k) > 1.5 * kz and max(got_a) > 0.8 * az:
        print("    ✅ 异质主体成功把峰度与波动率聚集同时推到真实市场量级。")
    elif max(got_k) > 1.2 * kz or max(got_a) > 0.8 * az:
        print("    🟡 部分指标改善：至少有一个机制贡献明显，但未同时达标。")
    else:
        print("    ❌ 未能复现真实的肥尾/波动率聚集 —— 需要继续找参数区间（见阶段4）。")
    print("    注：本阶段只跑蓝图给定配比；参数搜索在阶段4，避免在这里偷做调参。")


if __name__ == "__main__":
    main()
