"""长记忆订单流分析（二期阶段6）。

要测的三件事
------------
1. **订单符号自相关** ``ACF(k)``：真实市场的订单流有长记忆——
   ``ACF(k) ≈ k^{−γ}, γ ≈ 0.4~0.6``，一直显著到几百笔。一期的基线接近 0。
2. **衰减指数 γ 与 Hurst 指数**：``H = 1 − γ/2``。真实值 H ≈ 0.7~0.8
   （订单流的"持续性"），一期的基线 H ≈ 0.5（无关随机）。
3. **冲击函数指数 k**：真实 ≈ 0.5（平方根律）。放在 ``tw/impact.py``。

⚠️ 三个必须写进代码的统计护栏（都是一期踩过的同类错误的变体）
============================================================

**① 显著性 ≠ 效应量。**
   N 笔订单下，iid 零假设的标准误是 ``1/√N``。10 万笔成交时 se ≈ 0.003，
   于是 ``ACF(1) = 0.01`` 就能"显著"。**它能显著，但它是零**——
   对冲击函数毫无影响。所以本模块**同时报效应量**（ACF 的绝对值、
   ``ΣACF``、Hurst），并且验收判据写成"ACF 显著为正 **且** 量级达到某个下界"。

**② 拟合前必须检查自变量的对数跨度。**
   ``log-log`` 回归在跨度不足时会在噪声上给出漂亮的 ``R²``。
   一期的冲击函数拟合已经踩过（瞬时清算各档成交量跨度 0.7%，
   回归给出 ``k=1.72、R²=0.987``）。这里把同一道护栏搬过来：
   自变量的对数跨度 < 1 个数量级 → 拒绝拟合，返回原因。
   二阶护栏：因变量也要真的在变（``ACF`` 不能全是同一个数）。

**③ 只对**正的** ACF 做幂律拟合。**
   ACF 在长 lag 上会穿过 0 变成小的负值（纯噪声）。取 log 会得到 NaN
   或把负值当正值处理。规范做法是取**连续为正的那一段**，
   且在遇到第一个非正值就停止——而不是"筛掉负值再拟合"。
   后者会在尾部噪声里挑出零星的正值，把斜率往 0 靠（假长记忆）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: 一期把"对数跨度不足"的护栏定在 1 个数量级，这里沿用同一数值。
MIN_LOG_SPAN = 1.0


# ----------------------------------------------------------------------
# 1. 订单符号序列
# ----------------------------------------------------------------------
def trade_signs(trades) -> np.ndarray:
    """逐笔成交的**主动方符号**：主动买 +1，主动卖 −1。

    ⚠️ **这不是 LMF 的 ε 序列，别拿它算长记忆。** 见 ``order_signs``。
    保留它只用于低阶诊断（例如买占比）。
    """
    if not trades:
        return np.zeros(0, dtype=np.float64)
    return np.asarray(
        [1.0 if t.aggressor_side == "buy" else -1.0 for t in trades],
        dtype=np.float64,
    )


def order_signs(trades) -> np.ndarray:
    """按**主动订单**聚合的符号序列 —— 这才是 LMF 意义上的 ε。

    ⭐ 为什么必须聚合，为什么这是一条**定义**问题而不是优化
    ----------------------------------------------------
    一个主动单会连续吃掉多个价位，于是产生**多笔成交**，
    而这些成交的 ``aggressor_side`` 完全相同。
    直接对逐笔序列算自相关，会得到一个**纯机械的**正自相关：
    它不是"订单流有记忆"，而是"同一张单留下了多条相邻记录"。

    实测：一期基线（完全没有元订单机制）的**逐笔** ACF(1) = **+0.22**，
    看起来像"已经很长的记忆"；但那是记账粒度的产物。
    LMF 的原始定义是 **ε_i = 第 i 张订单的符号**（一笔一张），
    对成交笔数算 ACF 得到的数是**不可比的**——真实市场一张单也常扫多档，
    所以这个偏差会同时污染"模拟"和"真实"两侧，而两侧的扫单深度分布
    不一样，偏差方向也不同，于是**看起来像机制差异**。

    识别方法：同一张主动单的所有成交应当有相同的
    ``buy_order_id``（主动买）或 ``sell_order_id``（主动卖）。
    按它分组，每组只取一个符号。
    """
    if not trades:
        return np.zeros(0, dtype=np.float64)
    out: list[float] = []
    prev_oid: str | None = None
    for t in trades:
        oid = t.buy_order_id if t.aggressor_side == "buy" else t.sell_order_id
        if oid == prev_oid:
            continue
        out.append(1.0 if t.aggressor_side == "buy" else -1.0)
        prev_oid = oid
    return np.asarray(out, dtype=np.float64)


def tick_signed_flow(flow: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """按 tick 聚合的带符号主动成交量（``log.flow`` 的切片）。"""
    return np.nan_to_num(np.asarray(flow[lo:hi], dtype=np.float64), nan=0.0)


# ----------------------------------------------------------------------
# 2. 自相关与显著性
# ----------------------------------------------------------------------
def order_sign_acf(signs: np.ndarray, max_lag: int = 200) -> np.ndarray:
    """订单符号序列的样本自相关，返回 ``lag = 1..max_lag``。

    去均值后按 ``Σx_t·x_{t+k} / Σx_t²`` 算（**有偏**估计量）。
    **对有偏/无偏的选择说明**：LMF 的原始测量用的是有偏（分母固定为
    ``Σx²``）。用无偏（分母按重叠数 ``n−k`` 调整）会在**大 lag 上系统性
    放大噪声**，而在 lag 接近 n 时相关系数会乱跳——那正是我们最关心的尾部。
    所以照 LMF 用有偏口径，别"顺手改得更严谨"。

    实现用 **FFT 自相关**而不是双重循环。为什么要换：一段 8000 tick 的观测
    有 ~40 万张订单，``max_lag=200`` 的双重循环是 8000 万次 Python 级运算，
    单个种子要跑几十秒（实测探针里三个种子跑了两分钟还没出结果）——
    而 E6 要做多种子配对对照，这个代价直接决定了实验做不做得成。

    FFT 口径是**循环**自相关；在 ``n ≫ max_lag`` 时（这里 n=40 万、lag≤200）
    缠绕进来的部分只涉及尾部 ``max_lag`` 个样本，相对误差 ~5e-4 量级。
    ``tests/test_longmemory.py`` 里有一条断言把 FFT 结果与直接算法钉在一起，
    所以这不是"为了快而放松口径"。
    """
    x = np.asarray(signs, dtype=np.float64)
    n = x.size
    if n < 4:
        return np.zeros(0, dtype=np.float64)
    k_max = int(min(max_lag, n - 2))
    if k_max < 1:
        return np.zeros(0, dtype=np.float64)
    xc = x - x.mean()
    den = float(np.dot(xc, xc))
    if den <= 1e-18:
        return np.zeros(0, dtype=np.float64)
    m = 1 << int(np.ceil(np.log2(max(2 * n, 4))))
    f = np.fft.rfft(xc, m)
    ac = np.fft.irfft(f * np.conj(f), m)[: k_max + 1]
    return np.asarray(ac[1: k_max + 1] / den, dtype=np.float64)


def acf_white_noise_band(n: int, z: float = 3.0) -> float:
    """iid 零假设下 ACF 的 ``z`` 倍标准误带（Bartlett 一阶近似）。

    ``se ≈ 1/√n``。返回的是**双尾带**，用于判断"某个 lag 上是否显著"。
    只报带不报效应量是不够的——见模块文档护栏 ①。
    """
    if n <= 1:
        return float("inf")
    return float(z / np.sqrt(n))


def acf_significant_lags(acf: np.ndarray, band: float) -> int:
    """有多少个 lag 的 ACF 超过显著带（**只看正的**）。"""
    a = np.asarray(acf, dtype=np.float64)
    return int(np.count_nonzero(a > band))


def acf_positive_run(acf: np.ndarray) -> int:
    """从 lag=1 起、连续为正的 lag 数。

    这个量是"长记忆"最直观的刻画：随机流的 ACF 通常 lag=1 就是小正值、
    lag=2 或 3 就穿零；有长记忆的流能连着几百个 lag 都为正。
    """
    a = np.asarray(acf, dtype=np.float64)
    n = int(np.argmax(a <= 0)) if np.any(a <= 0) else a.size
    return n


def acf_sum(acf: np.ndarray) -> float:
    """``Σ ACF(k)``（到第一个非正值为止）。

    与冲击函数的关系：在 LMF 的框架下，订单流的长记忆强度决定了
    "流动性提供者能预判多远"，进而决定冲击的凹度。
    ``ΣACF`` 是这条链最简洁的强度指标。
    """
    a = np.asarray(acf, dtype=np.float64)
    if a.size == 0:
        return 0.0
    n = acf_positive_run(a)
    return float(a[:n].sum()) if n > 0 else 0.0


# ----------------------------------------------------------------------
# 3. 幂律衰减拟合
# ----------------------------------------------------------------------
@dataclass(slots=True)
class DecayFit:
    """幂律衰减的拟合结果。``ok=False`` 时 ``reason`` 说明为什么拒绝。"""

    ok: bool
    exponent: float = float("nan")
    r_squared: float = float("nan")
    n_points: int = 0
    log_span: float = float("nan")
    reason: str = ""
    lags: np.ndarray | None = None
    values: np.ndarray | None = None

    @property
    def hurst(self) -> float:
        """由衰减指数反推的 Hurst 指数：``H = 1 − γ/2``。

        推导：若 ``ACF(k) ∝ k^{−γ}``，则部分和的方差
        ``Var(Σ) ∝ n^{2−γ}``，而 ``Var(Σ) ∝ n^{2H}`` ⇒ ``H = 1 − γ/2``。
        · γ = 1 → H = 0.5（无关随机，无记忆）
        · γ = 0.5 → H = 0.75（真实订单流量级）
        """
        if not self.ok or not np.isfinite(self.exponent):
            return float("nan")
        return 1.0 - self.exponent / 2.0


def fit_power_law_decay(
    acf: np.ndarray, *, min_lag: int = 1, max_lag: int | None = None,
    min_points: int = 5, min_yrange: float = 1.5,
) -> DecayFit:
    """对 ``ACF(k) ∝ k^{−γ}`` 做 log-log 回归。

    三道护栏（缺一不可，都是一期同类错误的重演）：

    ① **只取从 lag=1 起连续为正的那一段。** 遇到第一个非正值就停。
       若改成"筛掉负值再拟合"，会在尾部噪声里挑出零星正值，
       把斜率往 0 靠——测出**假的长记忆**。
    ② **自变量对数跨度 ≥ 1 个数量级。** 跨度不足时的漂亮 ``R²``
       是在数值噪声上拟合出来的（一期冲击函数实测：跨度 0.7% 也能拟合出
       ``R²=0.987``）。
    ③ **因变量也要真的在变**（``max/min ≥ min_yrange``）。
       若 ACF 在该段基本是常数，斜率只由两个端点决定。
    """
    a = np.asarray(acf, dtype=np.float64)
    if a.size == 0:
        return DecayFit(False, reason="ACF 为空")
    hi = a.size if max_lag is None else min(a.size, int(max_lag))
    seg = a[:hi]
    n_pos = acf_positive_run(seg)
    if n_pos == 0:
        return DecayFit(False, reason="lag=1 处 ACF 非正——根本没有记忆")
    # min_lag 是 1-based 的 lag，切片要减 1
    start = max(0, int(min_lag) - 1)
    if n_pos <= start + min_points - 1:
        return DecayFit(
            False, reason=f"连续正的 lag 只有 {n_pos} 个，不足 {start + min_points} 个"
        )
    y = seg[start:n_pos]
    x = np.arange(start + 1, n_pos + 1, dtype=np.float64)

    span = float(np.log10(x.max()) - np.log10(x.min()))
    if span < MIN_LOG_SPAN:
        return DecayFit(
            False, log_span=span, n_points=int(x.size),
            reason=f"自变量对数跨度仅 {span:.2f} 个数量级，小于 {MIN_LOG_SPAN} 的护栏，拒绝拟合",
        )
    if y.min() <= 0:
        return DecayFit(False, log_span=span, n_points=int(x.size),
                        reason="有效段内出现非正值")
    yrange = float(y.max() / y.min())
    if yrange < min_yrange:
        return DecayFit(
            False, log_span=span, n_points=int(x.size),
            reason=f"因变量已退化（max/min = {yrange:.2f} < {min_yrange}），斜率由端点决定",
        )

    lx, ly = np.log(x), np.log(y)
    slope, intercept = np.polyfit(lx, ly, 1)
    ss_res = float(((ly - (slope * lx + intercept)) ** 2).sum())
    ss_tot = float(((ly - ly.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return DecayFit(
        True, exponent=float(-slope), r_squared=float(r2), n_points=int(x.size),
        log_span=span, reason="", lags=x, values=y,
    )


# ----------------------------------------------------------------------
# 4. 汇总
# ----------------------------------------------------------------------
def longmemory_summary(
    trades, flow: np.ndarray, lo: int, hi: int, max_lag: int = 200,
) -> dict:
    """一次模拟的订单流长记忆全套指标。

    ``lo/hi`` 是观测窗口（**必须已经去掉预热**）。``max_lag`` 默认 200，
    与 LMF 文献一致。

    ⚠️ 符号序列用 ``order_signs``（**按订单**聚合），不是 ``trade_signs``
    （逐笔成交）。混用会让不同深度的扫单产生不同的**机械**正自相关，
    把记账粒度差异读成机制差异（详见 ``order_signs`` 的说明）。
    """
    win_trades = [t for t in trades if lo <= t.tick < hi]
    signs = order_signs(win_trades)
    n = signs.size
    out: dict = {
        "n_trades": int(len(win_trades)),
        "n_orders": int(n),
        "window": [int(lo), int(hi)],
        "max_lag": int(max_lag),
    }
    if n < 10:
        out.update({"acf": [], "band": float("nan"), "n_sig_lags": 0,
                    "positive_run": 0, "acf_sum": 0.0, "buy_frac": float("nan"),
                    "fit_ok": False, "fit_reason": "订单样本不足"})
        return out

    acf = order_sign_acf(signs, max_lag=max_lag)
    band = acf_white_noise_band(n, z=3.0)
    fit = fit_power_law_decay(acf)
    out.update({
        "buy_frac": float((signs > 0).mean()),
        "acf": [float(v) for v in acf],
        # 逐阶取几个**有代表性的** lag：1（最短期）、10、50、100、200。
        # 为什么不只报 lag1：本阶段的结论是"记忆变长了多少"，
        # 而"长"只体现在**大 lag 上**。只报 lag1 会让"衰减更慢"这件事
        # 完全看不见——两个机制的 lag1 可能差不多，但一个在 lag 200 上
        # 已经归零、另一个还在正区间。
        **{f"acf_lag{k}": (float(acf[k - 1]) if acf.size >= k else float("nan"))
           for k in (1, 2, 5, 10, 20, 50, 100, 200)},
        "band": float(band),
        "n_sig_lags": acf_significant_lags(acf, band),
        "positive_run": acf_positive_run(acf),
        "acf_sum": acf_sum(acf),
        "fit_ok": bool(fit.ok),
        "fit_reason": fit.reason,
        "decay_exponent": float(fit.exponent),
        "hurst": float(fit.hurst),
        "fit_r2": float(fit.r_squared),
        "fit_n_points": int(fit.n_points),
        "fit_log_span": float(fit.log_span),
    })

    # tick 级带符号流的长记忆（与"逐笔"是两个不同粒度，都要看：
    # 逐笔看"符号持续性"，tick 级看"净流量持续性"——冲击函数由后者驱动。）
    sf = tick_signed_flow(flow, lo, hi)
    nz = sf[sf != 0.0]
    if nz.size > 10:
        sgn = np.sign(nz)
        acf2 = order_sign_acf(sgn, max_lag=min(max_lag, 100))
        fit2 = fit_power_law_decay(acf2)
        out.update({
            "tick_acf_lag1": float(acf2[0]) if acf2.size else float("nan"),
            "tick_positive_run": acf_positive_run(acf2),
            "tick_acf_sum": acf_sum(acf2),
            "tick_nonzero_ticks": int(nz.size),
            "tick_fit_ok": bool(fit2.ok),
            "tick_decay_exponent": float(fit2.exponent),
            "tick_hurst": float(fit2.hurst),
            "tick_fit_reason": fit2.reason,
        })
    return out
