"""Analyzer：统计特征计算与真实/模拟对照（施工蓝图 §2 组件清单、§4）。

这个模块是项目的**裁判**，所以它的实现必须比被测对象更保守：
每一处近似、每一个口径选择都写在注释里，因为如果裁判自己偷懒，
实验结论就没有意义了。

刻意保留的三个"口径注意事项"
----------------------------
1. **峰度用超额峰度（excess kurtosis）**，正态分布为 0。
   若用原始峰度，正态是 3，容易和"远大于 3"的表述混起来算错。
2. **ACF 的 95% 置信带用 Bartlett 公式 ±1.96/√T**（白噪声近似）。
   只要滞后期数远小于 T，这个近似就够用；报告里同时给出"超带滞后期数"，
   避免"看一眼图形觉得有聚集"这种主观判断。
3. **方差比用 Lo-MacKinlay 的异方差稳健版**，不用简版。
   加密收益率有强异方差，简版方差比的 z 统计量会系统性偏大，
   把随机游走误判成"有均值回归"。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

# ----------------------------------------------------------------------
# 基础量
# ----------------------------------------------------------------------


def log_returns(price: np.ndarray, drop_zero: bool = False) -> np.ndarray:
    """对数收益率。非有限值（价格出现 0 或 NaN）一律剔除。"""
    p = np.asarray(price, dtype=np.float64)
    if p.size < 2:
        return np.array([])
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.diff(np.log(p))
    m = np.isfinite(r)
    if drop_zero:
        m &= r != 0.0
    return r[m]


def acf(x: np.ndarray, nlags: int) -> np.ndarray:
    """自相关函数（含 lag=0，返回长度 nlags+1）。

    用 FFT 不是必须的，但去均值 + 直接相关是这里最不容易写错的做法。
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 3:
        return np.full(nlags + 1, np.nan)
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom <= 0:
        return np.full(nlags + 1, np.nan)
    out = np.empty(nlags + 1, dtype=np.float64)
    out[0] = 1.0
    for k in range(1, nlags + 1):
        out[k] = float(np.dot(x[:-k], x[k:])) / denom
    return out


def acf_ci(n: int, level: float = 0.95) -> float:
    """白噪声下 ACF 的置信半宽（Bartlett）。"""
    if n < 4:
        return np.nan
    z = stats.norm.ppf(0.5 + level / 2.0)
    return float(z / np.sqrt(n))


def ljung_box(x: np.ndarray, lags: int = 10) -> tuple[float, float]:
    """Ljung-Box 检验。返回 (Q, p)。p 很小 = 存在自相关。"""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = x.size
    if n < lags + 5:
        return (np.nan, np.nan)
    r = acf(x, lags)
    if not np.all(np.isfinite(r[1:])):
        return (np.nan, np.nan)
    q = n * (n + 2) * float(np.sum(r[1:] ** 2 / (n - np.arange(1, lags + 1))))
    p = float(stats.chi2.sf(q, lags))
    return (float(q), p)


def variance_ratio(r: np.ndarray, q: int) -> tuple[float, float]:
    """Lo-MacKinlay (1988) 异方差稳健方差比。返回 (VR(q), z)。

    VR = 1  → 随机游走
    VR > 1  → 正自相关（趋势）
    VR < 1  → 负自相关（均值回归）
    """
    r = np.asarray(r, dtype=np.float64)
    r = r[np.isfinite(r)]
    n = r.size
    if n < q * 4 or q < 2:
        return (np.nan, np.nan)
    mu = r.mean()
    dev = r - mu
    var1 = float(np.sum(dev**2)) / (n - 1)
    if var1 <= 0:
        return (np.nan, np.nan)

    # 重叠的 q 期收益率
    csum = np.concatenate([[0.0], np.cumsum(r)])
    rq = csum[q:] - csum[:-q]
    nq = rq.size
    muq = rq.mean()
    varq = float(np.sum((rq - muq) ** 2)) / (nq - 1)
    vr = varq / (q * var1)

    # 异方差稳健方差
    s2 = float(np.sum(dev**2))
    theta = 0.0
    for j in range(1, q):
        num = n * float(np.sum(dev[j:] ** 2 * dev[:-j] ** 2))
        dj = num / (s2**2) if s2 > 0 else 0.0
        theta += ((2.0 * (q - j) / q) ** 2) * dj
    if theta <= 0:
        return (float(vr), np.nan)
    # ⚠️ 必须乘 √n。踩过的坑：最初写成 (vr-1)/sqrt(theta)，
    # 结果 z 被系统性低估约 √n 倍（n=5e4 时低估 224 倍），
    # 真实存在的自相关会被判成"不显著"——一个只会漏报、不会误报的错。
    # 校验方式：iid 情形下 θ→Σ[2(q-j)/q]²，此时 z 的分布应渐近 N(0,1)；
    # 用 AR(1)/MA(1) 做过已知答案的测试（见 tests/test_analyzer.py）。
    z = np.sqrt(n) * (vr - 1.0) / np.sqrt(theta)
    return (float(vr), float(z))


def hill_alpha(x: np.ndarray, k: int | None = None) -> float:
    """Hill 尾部指数 α。``x`` 必须是正的尾部样本（如 |负收益|）。

    α 越小尾巴越厚。正态分布对应 α → ∞（Hill 估计不适用）；
    加密资产日频收益的 α 常见区间约 2~4。
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x) & (x > 0)]
    n = x.size
    if n < 20:
        return np.nan
    if k is None:
        k = max(10, int(n * 0.05))  # 默认取最大的 5%
    k = min(k, n - 1)
    s = np.sort(x)[::-1]
    tail = s[:k]
    denom = np.mean(np.log(tail)) - np.log(s[k])
    if denom <= 0:
        return np.nan
    return float(1.0 / denom)


# ----------------------------------------------------------------------
# 特征打包
# ----------------------------------------------------------------------


@dataclass
class SeriesMetrics:
    """一段价格序列的全部统计特征。模拟与真实共用同一套口径。"""

    label: str
    n: int = 0
    n_trades: int = 0
    # 分布
    sigma_bp: float = np.nan
    stdev_ratio: float = np.nan  # std / mean|r|，肥尾的粗测度
    skew: float = np.nan
    excess_kurtosis: float = np.nan
    kurtosis_raw: float = np.nan
    jarque_bera_p: float = np.nan
    min_bp: float = np.nan
    max_bp: float = np.nan
    p999_bp: float = np.nan
    hill_alpha_left: float = np.nan
    hill_alpha_right: float = np.nan
    # 自相关
    acf_ret: list[float] = field(default_factory=list)
    acf_abs: list[float] = field(default_factory=list)
    acf_ci95: float = np.nan
    n_sig_lags_ret: int = 0
    n_sig_lags_abs: int = 0
    acf_abs_mean: float = np.nan
    lb_ret_p: float = np.nan
    lb_abs_p: float = np.nan
    vr2: float = np.nan
    vr5: float = np.nan
    vr10: float = np.nan
    vr2_z: float = np.nan
    vr5_z: float = np.nan
    # 波动率
    vol_acf_lag1: float = np.nan
    vol_of_vol: float = np.nan  # 滚动波动率的标准差 / 均值
    # 冲击事件研究
    impact: dict = field(default_factory=dict)

    # --- 派生指标（对照表按名字取值，必须有同名属性）---------------------
    @property
    def acf_abs_lag1(self) -> float:
        return self.acf_abs[1] if len(self.acf_abs) > 1 else np.nan

    @property
    def acf_abs_lag5(self) -> float:
        return self.acf_abs[5] if len(self.acf_abs) > 5 else np.nan

    @property
    def acf_ret_lag1(self) -> float:
        return self.acf_ret[1] if len(self.acf_ret) > 1 else np.nan

    @property
    def acf_abs_mean_1_10(self) -> float:
        return self.acf_abs_mean

    def flat(self) -> dict[str, float | str | int]:
        d = {
            "label": self.label,
            "n": self.n,
            "sigma_bp": self.sigma_bp,
            "excess_kurtosis": self.excess_kurtosis,
            "skew": self.skew,
            "acf_ret_lag1": self.acf_ret_lag1,
            "acf_abs_lag1": self.acf_abs_lag1,
            "acf_abs_lag5": self.acf_abs_lag5,
            "acf_abs_mean_1_10": self.acf_abs_mean,
            "n_sig_lags_abs": self.n_sig_lags_abs,
            "vr5": self.vr5,
            "hill_alpha_left": self.hill_alpha_left,
            "p999_bp": self.p999_bp,
        }
        if self.n_trades:
            d["n_trades"] = self.n_trades
        return d


def analyze(
    price: np.ndarray,
    label: str = "series",
    *,
    nlags: int = 20,
    hill_frac: float = 0.05,
    n_trades: int = 0,
    flow: np.ndarray | None = None,
    volume: np.ndarray | None = None,
    vol_window: int = 24,
) -> SeriesMetrics:
    """把一段价格序列压成可对照的统计特征。"""
    p = np.asarray(price, dtype=np.float64)
    p = p[np.isfinite(p) & (p > 0)]
    r = log_returns(p)
    m = SeriesMetrics(label=label, n=int(r.size), n_trades=n_trades)
    if r.size < 30:
        return m

    m.sigma_bp = float(r.std(ddof=1) * 1e4)
    m.skew = float(stats.skew(r, bias=False))
    m.excess_kurtosis = float(stats.kurtosis(r, fisher=True, bias=False))
    m.kurtosis_raw = m.excess_kurtosis + 3.0
    try:
        m.jarque_bera_p = float(stats.jarque_bera(r).pvalue)
    except Exception:  # pragma: no cover - 极短序列
        m.jarque_bera_p = np.nan
    mean_abs = float(np.mean(np.abs(r)))
    m.stdev_ratio = float(r.std(ddof=1) / mean_abs) if mean_abs > 0 else np.nan
    m.min_bp = float(r.min() * 1e4)
    m.max_bp = float(r.max() * 1e4)
    m.p999_bp = float(np.quantile(np.abs(r), 0.999) * 1e4)

    neg = -r[r < 0]
    pos = r[r > 0]
    k = max(10, int(neg.size * hill_frac))
    m.hill_alpha_left = hill_alpha(neg, k)
    m.hill_alpha_right = hill_alpha(pos, k)

    m.acf_ci95 = acf_ci(r.size)
    m.acf_ret = [float(v) for v in acf(r, nlags)]
    absr = np.abs(r)
    m.acf_abs = [float(v) for v in acf(absr, nlags)]
    band = m.acf_ci95
    if np.isfinite(band):
        m.n_sig_lags_ret = int(np.sum(np.abs(m.acf_ret[1:]) > band))
        m.n_sig_lags_abs = int(np.sum(np.abs(m.acf_abs[1:]) > band))
    hi = min(10, len(m.acf_abs) - 1)
    if hi >= 1:
        m.acf_abs_mean = float(np.mean(m.acf_abs[1 : hi + 1]))
    _, m.lb_ret_p = ljung_box(r, 10)
    _, m.lb_abs_p = ljung_box(absr, 10)
    m.vr2, m.vr2_z = variance_ratio(r, 2)
    m.vr5, m.vr5_z = variance_ratio(r, 5)
    m.vr10, _ = variance_ratio(r, 10)

    # 滚动波动率：波动率本身的波动（"波动率聚集"的直观量）
    if r.size > vol_window * 3:
        cs = np.concatenate([[0.0], np.cumsum(r**2)])
        rv = np.sqrt((cs[vol_window:] - cs[:-vol_window]) / vol_window)
        rv = rv[np.isfinite(rv) & (rv > 0)]
        if rv.size > 10 and rv.mean() > 0:
            m.vol_of_vol = float(rv.std(ddof=1) / rv.mean())
            a = acf(np.log(rv), 1)
            m.vol_acf_lag1 = float(a[1]) if np.isfinite(a[1]) else np.nan

    if flow is not None:
        m.impact = impact_event_study(p, flow)
    return m


# ----------------------------------------------------------------------
# 冲击事件研究（"扫单后继续走，还是反转？"）
# ----------------------------------------------------------------------


def impact_event_study(
    price: np.ndarray,
    flow: np.ndarray,
    *,
    horizons: tuple[int, ...] = (1, 2, 3, 5, 10, 20, 50),
    quantile: float = 0.95,
    min_gap: int = 5,
) -> dict:
    """激进订单流冲击后，价格是**延续**还是**反转**？

    做法（事件研究）
    ----------------
    1. 找出"净主动成交额"绝对值超过 ``quantile`` 分位的 tick 作为事件点
    2. 按事件方向取符号 s = sign(flow)，把价格路径对齐成"冲击方向为正"
    3. 计算 s * (log P(t+k) - log P(t))，对所有事件求均值

    读法
    ----
    均值曲线在 k 增大时**继续上升** → 延续（扫单后继续走）
    均值曲线在 k 增大时**回落** → 反转（冲击被吸收、价格弹回）
    ``continuation_ratio`` = 长期响应 / 即时响应：
        > 1 表示延续、< 1 表示反转、≈ 0 表示完全吸收

    为什么必须做价格对齐
    --------------------
    不能把"买冲击"和"卖冲击"的路径直接平均——两者方向相反会互相抵消，
    得到的接近 0 的曲线既可能是"没有冲击效应"，也可能是"效应很强但被抵消"。
    对齐后符号才有意义。

    为什么设 min_gap
    ----------------
    相邻 tick 常常都超过阈值，会形成一串高度重叠的"事件"，
    等于把同一个事件数了很多遍，人为夸大样本数。设最小间隔做去重。
    """
    p = np.asarray(price, dtype=np.float64)
    f = np.asarray(flow, dtype=np.float64)
    n = min(p.size, f.size)
    p, f = p[:n], f[:n]
    out: dict = {"n_events": 0, "horizons": list(horizons)}
    if n < 100:
        return out

    active = np.isfinite(f) & (f != 0.0)
    if active.sum() < 50:
        return out
    thr = float(np.quantile(np.abs(f[active]), quantile))
    if thr <= 0:
        return out

    idx = np.flatnonzero(active & (np.abs(f) >= thr))
    if idx.size == 0:
        return out
    # 事件去重：保留间隔至少 min_gap 的事件
    kept = [int(idx[0])]
    for i in idx[1:]:
        if i - kept[-1] >= min_gap:
            kept.append(int(i))
    ev = np.asarray(kept, dtype=int)

    logp = np.log(p)
    maxh = max(horizons)
    usable = ev[ev < n - maxh - 1]
    if usable.size < 10:
        out["n_events"] = int(usable.size)
        return out

    sign = np.sign(f[usable])
    curve = {}
    for h in horizons:
        resp = sign * (logp[usable + h] - logp[usable])
        resp = resp[np.isfinite(resp)]
        if resp.size < 10:
            continue
        mean = float(np.mean(resp) * 1e4)
        se = float(np.std(resp, ddof=1) / np.sqrt(resp.size) * 1e4)
        curve[h] = {
            "mean_bp": mean,
            "se_bp": se,
            "t": mean / se if se > 0 else np.nan,
            "positive_share": float(np.mean(resp > 0)),
        }

    out["n_events"] = int(usable.size)
    out["threshold"] = thr
    out["curve"] = curve

    if not curve:
        return out
    near = min(curve)
    far = max(curve)
    imm = curve[near]
    lng = curve[far]
    out["immediate_bp"] = imm["mean_bp"]
    out["immediate_t"] = imm["t"]
    out["long_bp"] = lng["mean_bp"]
    out["long_t"] = lng["t"]
    out["peak_horizon"] = max(curve, key=lambda h: abs(curve[h]["mean_bp"]))

    # ⚠️ 延续比只在**即时响应显著**时才有意义。
    # 踩过的坑：真实数据上即时响应是 -5.7bp、t=-1.42（完全不显著），
    # 长期响应 -25.6bp，两者相除得到 +4.51 —— 一个看起来像"强延续"的数字，
    # 实际上只是"两个都不显著的小数相除"。分子分母都不可信时，比值只会骗人。
    if abs(imm["t"]) >= 2.0 and abs(imm["mean_bp"]) > 1e-9:
        out["continuation_ratio"] = lng["mean_bp"] / imm["mean_bp"]
        if out["continuation_ratio"] > 1.5:
            out["verdict"] = "延续（冲击后继续同向走）"
        elif out["continuation_ratio"] < 0.5:
            out["verdict"] = "衰减/反转（冲击被吸收）"
        else:
            out["verdict"] = "接近完全吸收"
    else:
        out["continuation_ratio"] = float("nan")
        out["verdict"] = "即时响应不显著 —— 延续比无意义，不作结论"
    return out


def bar_flow_proxy(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray, v: np.ndarray) -> np.ndarray:
    """真实 K 线数据缺少订单流，只能构造代理变量（必须显式标注为代理）。

    采用**收盘位置**（CLV, close location value）加权成交量::

        CLV = ((C - L) - (H - C)) / (H - L)  ∈ [-1, +1]

    CLV = +1 表示收在最高价（买方主导当日），-1 表示收在最低价。

    为什么不用 sign(C - O)：当 K 线实体很小时（C≈O），
    符号会被极小的价格差决定，噪声极大；CLV 用**振幅**做了归一化，
    对"小实体、长影线"的 K 线更稳健。
    """
    rng = np.asarray(h, dtype=np.float64) - np.asarray(l, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        clv = ((np.asarray(c, dtype=np.float64) - np.asarray(l, dtype=np.float64))
               - (np.asarray(h, dtype=np.float64) - np.asarray(c, dtype=np.float64))) / rng
    clv = np.where(rng > 0, clv, 0.0)
    clv = np.nan_to_num(clv, nan=0.0, posinf=0.0, neginf=0.0)
    return clv * np.asarray(v, dtype=np.float64)


# ----------------------------------------------------------------------
# 对照
# ----------------------------------------------------------------------

#: 对照表的行定义: (显示名, 取值器, 单位, "越高越像真实" 的方向说明)
COMPARE_ROWS: list[tuple[str, str, str, str]] = [
    ("收益率标准差", "sigma_bp", "bp", "量级可比即可（时间尺度对齐后）"),
    ("超额峰度", "excess_kurtosis", "", "★ 真实市场远大于 0（肥尾）"),
    ("偏度", "skew", "", "接近 0，略偏负"),
    ("|r| ACF(lag1)", "acf_abs_lag1", "", "★ 真实市场显著为正（波动率聚集）"),
    ("|r| ACF(lag5)", "acf_abs_lag5", "", "★ 多个滞后期显著为正"),
    ("|r| ACF 1-10 均值", "acf_abs_mean_1_10", "", "★ 越高越像"),
    ("显著 |r| 滞后数(共20)", "n_sig_lags_abs", "个", "★ 真实市场多于纯噪声下的~1个"),
    ("收益率 ACF(lag1)", "acf_ret_lag1", "", "→ 接近 0（弱有效市场）"),
    ("方差比 VR(5)", "vr5", "", "→ 接近 1（随机游走）"),
    ("左尾 Hill α", "hill_alpha_left", "", "越小尾巴越厚"),
    ("|r| 99.9 分位", "p999_bp", "bp", "尾部厚度"),
]


def compare_rows(metrics: list[SeriesMetrics]) -> list[dict]:
    rows = []
    for name, key, unit, note in COMPARE_ROWS:
        row = {"metric": name, "unit": unit, "note": note, "values": {}}
        for m in metrics:
            v = getattr(m, key, np.nan)
            row["values"][m.label] = v
        rows.append(row)
    return rows


def format_compare_table(metrics: list[SeriesMetrics], float_fmt: str = "{:>14,.4g}") -> str:
    labels = [m.label for m in metrics]
    w = max(24, max((len(x) for x in labels), default=8) + 2)
    head = "指标".ljust(22) + "".join(l.rjust(w) for l in labels)
    lines = [head, "-" * len(head)]
    for row in compare_rows(metrics):
        line = row["metric"].ljust(22)
        for l in labels:
            v = row["values"][l]
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                s = "n/a"
            elif isinstance(v, (int, np.integer)):
                s = str(int(v))
            else:
                s = f"{float(v):.4g}"
            line += s.rjust(w)
        lines.append(line)
    return "\n".join(lines)
