"""非对称性分析：突发同步率与四象限深度。

任务书：`交易世界 · 分辨力危机应对任务书.md` §4（沿用上一份任务书的 §4 设计）。

要检验的假说 H4
---------------
「只稀疏化吃单方有效、双侧同时稀疏化抵消」的机制解释是：

    如果做市方也服从聚集过程，那么**吃单方最活跃的时刻，做市方也最活跃**，
    局部流动性反而被补充起来，抵消了吃单方聚集本该造成的稀疏。
    只有做市方保持泊松时，吃单方的突发才能真正造成"来不及补充"的稀疏。

⚠️⚠️ **动手前的前提核对：H4 的措辞与实现不符（必须记下来）**
----------------------------------------------------------------
H4 原文假设"吃单方和做市方服从**各自独立的** Hawkes 聚集过程"。
但源码里（``tw/order_flow/hawkes_market.py::_limit_activity``）：

    rho = self._current_rho()          # ← 只取一次

    if self.thin_scope == "both":
        k = rho_to_count(rho, len(order), ...)     # 对整个列表按同一个 ρ 稀疏化

也就是说，**对称配置下两侧由同一个 ρ 驱动**，不是两个独立过程。
于是"同步率"在对称配置下几乎是**构造性**的：
同一个 ρ 的两个投影，当然同步。

⇒ 因此 ``measure_burst_synchrony`` 的结果**不能单独作为 H4 的证据**。
   它能说明的是"构造上确实同步"，而 H4 真正要问的是**同步是否真的抵消了稀疏**
   —— 那要靠四象限深度（``measure_local_depth_during_taker_burst``）来回答：
   如果 both_burst 象限的深度**不比** taker_only 象限更薄，
   那就说明做市方的同步活跃确实填平了吃单方造成的缺口。

⚠️ 另一个必须写明的口径：**突发阈值是两侧各自独立标定的**（各自的 80 分位）。
两侧的基线活跃度不同（吃单方与做市方的出手频率本来就不一样），
所以"突发"不是同一个绝对阈值，报告里必须写清楚。

统计口径（沿用项目纪律）
------------------------
tick 级检验的区间会因为**自相关**而虚假变窄（本项目实测过：比诚实的跨运行区间
窄约 11.6 倍）。所以四象限对比**不做裸的 tick 级 t 检验**，
而是先把窗口切成块、对每块取象限均值，再跨块做检验——这样自相关被块平均吸收。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


# ======================================================================
def rolling_cooccurrence_rate(a: np.ndarray, b: np.ndarray,
                              window: int) -> np.ndarray:
    """两个二值序列在滚动窗口内的**同时为真比例**。

    语义（⚠️ 边界必须写死，否则差一格的实现会让结果系统性偏低）：
    位置 ``i`` 的窗口是 ``[max(0, i−window+1), i]`` —— **包含 i 本身**。
    窗口长度在头部不足时按实际长度算（不补零，补零会把"还没开始"算成"不同步"）。
    """
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(f"两条 trace 长度不一致：{a.shape} vs {b.shape}")
    if window < 1:
        raise ValueError(f"window 必须 ≥ 1，收到 {window}")
    both = (a & b).astype(float)
    n = both.size
    out = np.full(n, np.nan)
    csum = np.concatenate([[0.0], np.cumsum(both)])
    for i in range(n):
        lo = max(0, i - window + 1)
        # ⚠️ 区间是 [lo, i] **两端都含**。写成 csum[i] − csum[lo] 就漏掉了 i 这一格，
        #    结果是系统性偏低（变异体 M42 钉的正是这一处）。
        total = csum[i + 1] - csum[lo]
        out[i] = total / (i - lo + 1)
    return out


def burst_mask(x: np.ndarray, pct: float = 80.0) -> np.ndarray:
    """突发掩码：``x > percentile(x, pct)``。

    ⚠️ 用严格大于而不是 `>=`：活跃度通常是**离散的整数**（每 tick 出手的主体数），
    大量并列值会让 `>=` 把"没有突发"的背景期也算成突发
    （阶段8 就踩过同类问题：`classify_burst_vs_background` 用的是 `x > thr`）。
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.zeros(0, dtype=bool)
    thr = float(np.percentile(x, pct))
    return x > thr


@dataclass
class SynchronyResult:
    mean_sync: float
    median_sync: float
    trace: np.ndarray
    n_taker_burst: int
    n_maker_burst: int
    taker_threshold: float
    maker_threshold: float
    window: int
    pct: float
    #: ⭐ **条件概率**：P(做市方突发 | 吃单方突发)
    #:
    #: ⚠️ 这是本轮补上的量。任务书说的"同步率"实际算出来是**联合**概率
    #: ``P(a ∧ b)``——它同时被两边各自的突发频率压着（实测只有 0.087），
    #: 读起来像"几乎不同步"，其实含义完全不同。
    #: H4 问的是"**当**吃单方突发时，做市方是不是也更活跃"，
    #: 那是条件概率。两者不能混。
    p_maker_given_taker: float = float("nan")
    p_taker_given_maker: float = float("nan")
    #: 提升度 = P(m|t) / P(m)。=1 表示条件独立；>1 表示确实同向活跃。
    lift: float = float("nan")


def measure_burst_synchrony(taker_activity: np.ndarray, maker_activity: np.ndarray,
                            *, pct: float = 80.0,
                            window: int = 50) -> SynchronyResult:
    """两侧各自标定突发阈值后的滚动同步率 + 条件同步率。

    ⚠️ 阈值**分别**从各自的 80 分位算（见模块文档）：两侧基线活跃度不同，
    用同一个绝对阈值没有意义。
    """
    t = np.asarray(taker_activity, dtype=float)
    m = np.asarray(maker_activity, dtype=float)
    t_mask = burst_mask(t, pct)
    m_mask = burst_mask(m, pct)
    rate = rolling_cooccurrence_rate(t_mask, m_mask, window)

    n_t, n_m = int(t_mask.sum()), int(m_mask.sum())
    n_both = int((t_mask & m_mask).sum())
    p_m_given_t = (n_both / n_t) if n_t else float("nan")
    p_t_given_m = (n_both / n_m) if n_m else float("nan")
    p_m = float(m_mask.mean()) if m_mask.size else float("nan")
    lift = (p_m_given_t / p_m) if (p_m and p_m > 0) else float("nan")

    return SynchronyResult(
        mean_sync=float(np.nanmean(rate)) if rate.size else float("nan"),
        median_sync=float(np.nanmedian(rate)) if rate.size else float("nan"),
        trace=rate,
        n_taker_burst=n_t, n_maker_burst=n_m,
        taker_threshold=float(np.percentile(t, pct)) if t.size else float("nan"),
        maker_threshold=float(np.percentile(m, pct)) if m.size else float("nan"),
        window=window, pct=pct,
        p_maker_given_taker=float(p_m_given_t),
        p_taker_given_maker=float(p_t_given_m),
        lift=float(lift),
    )


# ======================================================================
QUADRANTS = ("both_burst", "taker_only", "maker_only", "neither")


def quadrant_masks(taker_burst: np.ndarray, maker_burst: np.ndarray) -> dict:
    """四象限掩码。

    ⚠️ ``taker_only`` 是「吃单方突发 **且** 做市方**不**突发」——
    两个掩码写反是很容易犯又很难发现的错（结果会与 H4 预期方向相反，
    看起来像"假说被否证"）。变异体 M43 钉的就是这里。
    """
    t = np.asarray(taker_burst, dtype=bool)
    m = np.asarray(maker_burst, dtype=bool)
    if t.shape != m.shape:
        raise ValueError(f"两个掩码长度不一致：{t.shape} vs {m.shape}")
    return {
        "both_burst": t & m,
        "taker_only": t & ~m,
        "maker_only": ~t & m,
        "neither": ~t & ~m,
    }


def measure_local_depth_during_taker_burst(
        depth: np.ndarray, taker_burst: np.ndarray,
        maker_burst: np.ndarray) -> dict:
    """四个象限各自的**平均局部深度**。"""
    d = np.asarray(depth, dtype=float)
    masks = quadrant_masks(taker_burst, maker_burst)
    out = {}
    for name, mask in masks.items():
        vals = d[mask]
        vals = vals[np.isfinite(vals)]
        out[name] = {
            "n": int(vals.size),
            "mean": float(vals.mean()) if vals.size else float("nan"),
            "median": float(np.median(vals)) if vals.size else float("nan"),
            "std": float(vals.std(ddof=1)) if vals.size > 1 else float("nan"),
        }
    return out


def block_quadrant_contrast(depth: np.ndarray, taker_burst: np.ndarray,
                            maker_burst: np.ndarray, *, block: int = 200,
                            min_block_n: int = 5) -> dict:
    """``taker_only`` vs ``both_burst`` 的**块级**对比（抗自相关）。

    做法：把整段窗口切成长 ``block`` 的块，每块内分别算两个象限的深度均值
    （块内样本数少于 ``min_block_n`` 就跳过），然后在**块级**做 Welch t 检验。

    为什么不用裸的 tick 级检验：同一段路径的 tick 高度自相关，
    tick 级 t 会把区间算得**虚假地窄**（本项目实测窄约 11.6 倍）。
    块平均把自相关吸收掉，代价是样本数变少——这个代价是**应该付**的。
    """
    d = np.asarray(depth, dtype=float)
    masks = quadrant_masks(taker_burst, maker_burst)
    n = d.size
    a_blocks, b_blocks = [], []
    for lo in range(0, n, block):
        hi = min(n, lo + block)
        seg = slice(lo, hi)
        dseg = d[seg]
        for name, store in (("taker_only", a_blocks), ("both_burst", b_blocks)):
            mk = masks[name][seg]
            vals = dseg[mk]
            vals = vals[np.isfinite(vals)]
            if vals.size >= min_block_n:
                store.append(float(vals.mean()))
    a = np.asarray(a_blocks, dtype=float)
    b = np.asarray(b_blocks, dtype=float)
    res = {
        "n_blocks_taker_only": int(a.size),
        "n_blocks_both_burst": int(b.size),
        "taker_only_block_mean": float(a.mean()) if a.size else float("nan"),
        "both_burst_block_mean": float(b.mean()) if b.size else float("nan"),
        "block": block,
    }
    if a.size >= 2 and b.size >= 2:
        t, p = stats.ttest_ind(a, b, equal_var=False)
        res.update({
            "t": float(t), "p": float(p),
            # 差 = both_burst − taker_only。H4 预测 **正**（做市方同步补充 ⇒ the both 象限更厚）
            "diff": float(b.mean() - a.mean()),
            "h4_direction_supported": bool(b.mean() > a.mean()),
        })
    else:
        res.update({"t": float("nan"), "p": float("nan"), "diff": float("nan"),
                    "h4_direction_supported": None,
                    "reason": f"块数不足（{a.size} / {b.size}）"})
    return res
