"""可视化（施工蓝图 §5「可视化」）。

设计约定
--------
1. **深色主题**：与工作环境一致，避免白底图在深色界面里刺眼。
2. **中文标签**：显式指定字体。matplotlib 默认字体没有中文，不设置会画出一堆方框，
   而且不报错——图看着"有内容"，实际没人能读。
3. **每张图只回答一个问题**：不做"六宫格大杂烩"。一张图要能一眼看出结论，
   否则它就只是装饰。
4. **模拟与真实用同一套坐标轴**：对照图必须能直接叠着看，
   分成两张图再让读者自己换算，等于没对照。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无显示环境

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .analyzer import acf, acf_ci, log_returns  # noqa: E402
from .logger import SimLog  # noqa: E402

# --- 主题 ------------------------------------------------------------------
BG = "#16181d"
PANEL = "#1d2026"
FG = "#d8dee9"
MUTED = "#8b95a5"
GRID = "#2c313a"
C_MID = "#5aa9e6"
C_DIM = "#7e8ba3"
C_REAL = "#e0a458"
C_SIM = "#5aa9e6"
C_ACCENT = "#e06c75"
C_GOOD = "#8fbf6a"

plt.rcParams.update(
    {
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.facecolor": BG,
        "axes.facecolor": PANEL,
        "savefig.facecolor": BG,
        "axes.edgecolor": GRID,
        "axes.labelcolor": FG,
        "text.color": FG,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "grid.color": GRID,
        "grid.alpha": 0.6,
        "axes.grid": True,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "figure.dpi": 110,
        "legend.framealpha": 0.15,
        "legend.facecolor": PANEL,
        "legend.edgecolor": GRID,
    }
)


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _note(ax, text: str) -> None:
    ax.text(
        0.99, 0.02, text, transform=ax.transAxes, ha="right", va="bottom",
        fontsize=8.5, color=MUTED,
    )


# ----------------------------------------------------------------------
def plot_price_series(
    log: SimLog, path: str | Path, title: str, fundamentals: bool = True
) -> Path:
    """价格路径 + 对数收益率，双面板。"""
    p = log.mid
    r = log_returns(p)
    fig, (a1, a2) = plt.subplots(
        2, 1, figsize=(11, 6.2), sharex=False, gridspec_kw={"height_ratios": [2.2, 1]}
    )
    t = np.arange(p.size)
    a1.plot(t, p, color=C_MID, lw=1.1, label="中间价")
    if fundamentals and log.fundamental is not None:
        a1.plot(t, log.fundamental, color=C_DIM, lw=1.0, ls="--", label="基本面价值 V(t)")
    a1.set_ylabel("价格")
    a1.set_title(title)
    a1.legend(loc="upper left", fontsize=9)
    a1.set_xlabel("tick")
    _note(a1, f"区间 {p.min():,.0f} ~ {p.max():,.0f}")

    a2.plot(np.arange(1, r.size + 1), r * 1e4, color=C_ACCENT, lw=0.6)
    a2.axhline(0, color=MUTED, lw=0.6)
    a2.set_ylabel("对数收益率 (bp)")
    a2.set_xlabel("tick")
    _note(a2, f"σ={r.std() * 1e4:.0f}bp/ tick")
    return _save(fig, path)


def plot_return_distribution(
    sim_r: np.ndarray, real_r: np.ndarray | None, path: str | Path, title: str
) -> Path:
    """收益率分布 vs 正态。肥尾的直接证据。"""
    from scipy import stats

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    for ax, r, name, color in (
        (axes[0], sim_r, "模拟", C_SIM),
        (axes[1], real_r if real_r is not None else sim_r, 
         "真实" if real_r is not None else "模拟", C_REAL if real_r is not None else C_SIM),
    ):
        r = r[np.isfinite(r)]
        x = r * 1e4
        ax.hist(x, bins=140, density=True, color=color, alpha=0.55, label="实际分布")
        xs = np.linspace(x.min(), x.max(), 400)
        ax.plot(
            xs,
            stats.norm.pdf(xs, x.mean(), x.std(ddof=1)),
            color=FG, lw=1.4, ls="--", label="同均值方差的正态",
        )
        k = stats.kurtosis(r, fisher=True, bias=False)
        ax.set_yscale("log")
        ax.set_xlabel("对数收益率 (bp)")
        ax.set_ylabel("密度（对数轴）")
        ax.set_title(f"{name}：超额峰度 = {k:.2f}")
        ax.legend(fontsize=9)
        ax.set_xlim(np.quantile(x, 0.0005), np.quantile(x, 0.9995))

    fig.suptitle(title, fontsize=12, fontweight="bold")
    _note(axes[1], "正态分布时超额峰度应为 0")
    return _save(fig, path)


def plot_acf_compare(
    sim_r: np.ndarray, real_r: np.ndarray | None, path: str | Path, nlags: int = 24
) -> Path:
    """关键不对称：收益率自身无自相关，但 |收益率| 有。"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))

    for ax, f, name in ((axes[0], np.abs, "|收益率|"), (axes[1], lambda x: x, "收益率本身")):
        y_sim = acf(f(sim_r), nlags)
        band = acf_ci(sim_r.size)
        lags = np.arange(nlags + 1)
        ax.bar(lags - 0.2, y_sim, width=0.4, color=C_SIM, label="模拟")
        if real_r is not None:
            y_real = acf(f(real_r), nlags)
            ax.bar(lags + 0.2, y_real, width=0.4, color=C_REAL, label="真实 BTC 1h")
        ax.axhspan(-band, band, color=MUTED, alpha=0.18, label="白噪声 95% 带")
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_title(f"{name} 的自相关")
        ax.set_xlabel("滞后 (tick)")
        ax.set_ylabel("自相关")
        ax.legend(fontsize=8.5)

    fig.suptitle("波动率聚集的判别：左边应显著为正，右边应几乎为零", fontweight="bold")
    return _save(fig, path)


def plot_volatility_clustering(
    sim_r: np.ndarray, real_r: np.ndarray | None, path: str | Path, window: int = 24
) -> Path:
    """滚动波动率：聚集 = 高低波动成片出现，而不是随机交替。"""
    fig, ax = plt.subplots(figsize=(11, 3.8))

    def rv(r: np.ndarray) -> np.ndarray:
        cs = np.concatenate([[0.0], np.cumsum(r**2)])
        return np.sqrt((cs[window:] - cs[:-window]) / window) * 1e4

    ax.plot(rv(sim_r), color=C_SIM, lw=0.8, label="模拟")
    if real_r is not None:
        ax.plot(rv(real_r)[: sim_r.size], color=C_REAL, lw=0.8, alpha=0.85, label="真实 BTC 1h")
    ax.set_xlabel("tick")
    ax.set_ylabel(f"{window} 期滚动波动率 (bp)")
    ax.set_title("滚动波动率：聚集表现为成片的高低波动区")
    ax.legend(fontsize=9)
    _note(ax, "若模拟是纯白噪声，这条线应像随机噪声没有成片结构")
    return _save(fig, path)


def plot_book_depth(log: SimLog, path: str | Path, n_levels: int = 15) -> Path:
    """订单簿深度演化热力图（蓝图 §5 要求的"订单簿深度热力图"）。"""
    snaps = [s for s in log.snapshots if s.bids and s.asks]
    if len(snaps) < 3:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(0.5, 0.5, "订单簿快照不足，无法绘制", ha="center", va="center")
        ax.axis("off")
        return _save(fig, path)

    n = min(n_levels, min(len(s.bids) for s in snaps), min(len(s.asks) for s in snaps))
    bid = np.array([[lv.quantity for lv in s.bids[:n]] for s in snaps]).T
    ask = np.array([[lv.quantity for lv in s.asks[:n]] for s in snaps]).T
    grid = np.vstack([bid[::-1], ask])
    ticks = np.array([s.tick for s in snaps])

    fig, ax = plt.subplots(figsize=(11, 4.4))
    im = ax.imshow(
        grid, aspect="auto", origin="lower", cmap="magma",
        extent=[ticks[0], ticks[-1], -n, n], interpolation="nearest",
    )
    ax.axhline(0, color=FG, lw=0.9, ls="--")
    ax.text(ticks[0], n - 0.7, "卖侧（越靠上越远离中间价）", fontsize=8.5, color=FG)
    ax.text(ticks[0], -n + 0.2, "买侧", fontsize=8.5, color=FG)
    ax.set_xlabel("tick")
    ax.set_ylabel("距中间价的档位")
    ax.set_title("订单簿深度演化（每格是一档挂单量）")
    fig.colorbar(im, ax=ax, label="挂单量", fraction=0.03)
    return _save(fig, path)


def plot_impact_curve(
    curves: dict[str, dict], path: str | Path, title: str
) -> Path:
    """冲击事件研究的响应曲线：向上=延续，向下=反转。"""
    fig, ax = plt.subplots(figsize=(9, 4.4))
    palette = [C_SIM, C_REAL, C_GOOD, C_ACCENT, "#b58cd6"]
    for i, (name, imp) in enumerate(curves.items()):
        if not imp or "curve" not in imp:
            continue
        hs = sorted(imp["curve"])
        ys = np.array([imp["curve"][h]["mean_bp"] for h in hs])
        es = np.array([imp["curve"][h]["se_bp"] for h in hs])
        c = palette[i % len(palette)]
        ax.errorbar(hs, ys, yerr=1.96 * es, marker="o", ms=4, color=c, lw=1.5,
                    capsize=3, label=f"{name}  (n={imp['n_events']})")
    ax.axhline(0, color=MUTED, lw=0.9, ls="--")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 50])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("冲击后的期数 k（对数轴）")
    ax.set_ylabel("方向对齐后的平均价格响应 (bp)")
    ax.set_title(title)
    ax.legend(fontsize=9)
    _note(ax, "曲线向上 = 延续；向下 = 反转；误差棒为 95% 置信区间")
    return _save(fig, path)


def plot_metric_bars(
    sim: dict[str, float], real: dict[str, float], path: str | Path, title: str
) -> Path:
    """关键指标对照柱状图（真实值归一化为 1，看模拟的相对位置）。"""
    keys = [
        ("excess_kurtosis", "超额峰度"),
        ("acf_abs_lag1", "|r| ACF(1)"),
        ("acf_abs_mean_1_10", "|r| ACF(1-10)"),
        ("n_sig_lags_abs", "显著 |r| 滞后数"),
        ("p999_bp", "|r| 99.9 分位"),
    ]
    keys = [(k, n) for k, n in keys if np.isfinite(real.get(k, np.nan)) and real.get(k, 0) != 0]
    labels = [n for _, n in keys]
    real_v = np.array([real[k] for k, _ in keys], dtype=float)
    sim_v = np.array([sim.get(k, np.nan) for k, _ in keys], dtype=float)
    ratio = sim_v / real_v

    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(labels))
    ax.bar(x, ratio, color=[C_GOOD if 0.5 <= r <= 2.0 else C_ACCENT for r in ratio], width=0.55)
    ax.axhline(1.0, color=FG, lw=1.2, ls="--")
    ax.axhspan(0.5, 2.0, color=C_GOOD, alpha=0.10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("模拟 / 真实 BTC 1h")
    ax.set_title(title)
    for xi, r in zip(x, ratio):
        ax.text(xi, r, f"{r:.2f}×", ha="center", va="bottom" if r >= 0 else "top", fontsize=9)
    _note(ax, "1.0 = 与真实完全一致；绿带 = 同一数量级（0.5×~2×）")
    return _save(fig, path)


def plot_sweep(
    df: dict[str, np.ndarray], x_key: str, x_label: str, path: str | Path, title: str
) -> Path:
    """阶段4 参数扫描：指标随某参数的变化。"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    specs = [
        ("excess_kurtosis", "超额峰度", C_SIM),
        ("acf_abs_lag1", "|r| ACF(lag1)", C_REAL),
        ("sigma_bp", "σ / tick (bp)", C_GOOD),
    ]
    x = df[x_key]
    order = np.argsort(x)
    for ax, (key, label, color) in zip(axes, specs):
        if key not in df:
            ax.axis("off")
            continue
        ax.plot(x[order], df[key][order], marker="o", ms=4, color=color, lw=1.4)
        ax.set_xlabel(x_label)
        ax.set_title(label)
    fig.suptitle(title, fontweight="bold")
    return _save(fig, path)
