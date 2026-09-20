"""把逐 tick 的成交与价格，整理成「人能看的图」所需要的形式。

为什么要有这一层
----------------
桌面端原本的主图是**折线**（中间价 + 基本面锚），但交易界面看价格走势
几乎都是**蜡烛图**——不是审美问题，而是折线会丢掉"这个窗口内
最高/最低到过哪里"这个信息，而那正是判断冲击与波动时最要紧的。

三条纪律
--------
1. **不改引擎**。所需数据（逐 tick 中间价、逐笔成交、策略成交）
   引擎本来就在记，这一层只做**聚合与整形**。
2. **纯函数**。不碰 IO、不依赖市场对象——这样才可测
   （项目里所有"算出数字"的东西都必须是可测的纯函数）。
3. **口径写在名字里**。「成交笔数」和「成交金额」是两回事；
   「单笔成交」和「完整回合」也是两回事。不写清楚就会像
   `worst_fill_bp` 那次一样，被当成另一个量来用。

⚠️ 一个容易踩的坑：做市策略的「胜率」**不能按单笔成交算**。
赚价差与接刀是同时发生的两件事（项目已有 `markout` 专门分开它们），
所以本模块计算绩效时**只提供口径明确的量**，
不提供那种"看起来对做市也适用"的通用胜率。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ======================================================================
@dataclass(slots=True)
class Bar:
    """一根 K 线。"""
    tick_open: int
    tick_close: int
    open: float
    high: float
    low: float
    close: float
    n_ticks: int
    volume: float          # 该窗口内的成交量（绝对值之和）
    n_trades: int          # 该窗口内的成交笔数

    def as_dict(self) -> dict:
        return {
            "t": self.tick_open, "t_end": self.tick_close,
            "o": self.open, "h": self.high, "l": self.low, "c": self.close,
            "n": self.n_ticks, "v": self.volume, "nt": self.n_trades,
        }


def ohlc_from_series(mid: np.ndarray, *, window: int,
                     volume: np.ndarray | None = None,
                     n_trades: np.ndarray | None = None,
                     from_tick: int = 0) -> list[Bar]:
    """把逐点中间价聚合成 K 线。

    ``window`` 是一个窗口包含多少个 tick。
    ``volume`` / ``n_trades`` 若给了，就按同一窗口**求和**并挂到对应的 K 线上
    （必须与 ``mid`` 等长——长度不符直接报错，不做静默截断：
    静默截断会让图和数错位，而且**看不出错**）。

    ⚠️ 约定：
      · 不满一个窗口的**尾部**也会产出最后一根（用实际长度），
        否则最后一段行情会凭空消失；
      · ``high``/``low`` 取的是窗口内的**极值**，这正是折线丢掉的信息。
    """
    if window < 1:
        raise ValueError(f"window 必须 ≥ 1，收到 {window}")
    arr = np.asarray(mid, dtype=np.float64)
    if arr.size == 0:
        return []
    n = arr.size
    for name, extra in (("volume", volume), ("n_trades", n_trades)):
        if extra is not None and len(np.asarray(extra)) != n:
            raise ValueError(
                f"{name} 长度 {len(np.asarray(extra))} 与 mid 长度 {n} 不一致")

    bars: list[Bar] = []
    for start in range(0, n, window):
        stop = min(start + window, n)
        seg = arr[start:stop]
        seg = seg[np.isfinite(seg)]
        if seg.size == 0:
            continue
        vol = float(np.sum(np.abs(np.asarray(volume, dtype=np.float64)[start:stop]))) \
            if volume is not None else 0.0
        nt = int(np.sum(np.asarray(n_trades)[start:stop])) if n_trades is not None else 0
        bars.append(Bar(
            tick_open=from_tick + start,
            tick_close=from_tick + stop - 1,
            open=float(seg[0]), high=float(seg.max()),
            low=float(seg.min()), close=float(seg[-1]),
            n_ticks=int(stop - start), volume=vol, n_trades=nt,
        ))
    return bars


def bars_payload(bars: list[Bar]) -> dict:
    """把 K 线列表压成**列式** payload（比逐根对象小得多，前端也更好用）。"""
    return {
        "t": [b.tick_open for b in bars],
        "te": [b.tick_close for b in bars],
        "o": [round(b.open, 6) for b in bars],
        "h": [round(b.high, 6) for b in bars],
        "l": [round(b.low, 6) for b in bars],
        "c": [round(b.close, 6) for b in bars],
        "v": [round(b.volume, 4) for b in bars],
        "nt": [b.n_trades for b in bars],
    }


# ======================================================================
def equity_drawdown(equity: np.ndarray) -> dict:
    """权益曲线的回撤统计（峰谷与持续期）。

    ⚠️ ``equity`` 必须**包含成交之前的起点**——
    本项目踩过这个坑：``equity[0]`` 是"成交之后"的值，
    而恒等式假设的是"成交之前"，于是路径整体偏移一格。
    这里不修正它，只是**不假装看不见**：返回值里带上 ``n`` 让调用者能核对。
    """
    eq = np.asarray(equity, dtype=np.float64)
    eq = eq[np.isfinite(eq)]
    if eq.size < 2:
        return {"n": int(eq.size), "max_dd": None, "max_dd_pct": None,
                "peak_tick": None, "trough_tick": None, "duration": None}
    peak = np.maximum.accumulate(eq)
    # 相对回撤（以峰值为分母）；峰值为 0 时无法定义比例，只报绝对值
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(peak != 0, (eq - peak) / np.abs(peak), 0.0)
    trough = int(np.argmin(dd))
    peak_idx = int(np.argmax(eq[: trough + 1])) if trough > 0 else 0
    return {
        "n": int(eq.size),
        "max_dd": float(eq[trough] - peak[trough]),
        "max_dd_pct": float(dd[trough]),
        "peak_tick": peak_idx,
        "trough_tick": trough,
        "duration": int(trough - peak_idx),
    }


def fill_stats(rows: list[dict]) -> dict:
    """逐笔成交的**口径明确**的统计。

    ``rows`` 每项至少要有 ``pnl``（该笔的已实现盈亏，元）与 ``markout_bp``。
    刻意**不**返回「胜率」——对做市策略而言"这一笔赚了"与"这笔是不是接刀"
    是两件事，单看盈亏会把两者混起来（项目已有 ``markout`` 专门分开它们）。
    需要胜率请在自己那层明确口径后再算。
    """
    if not rows:
        return {"n": 0, "gross_win": 0.0, "gross_loss": 0.0,
                "profit_factor": None, "expectancy": None,
                "avg_win": None, "avg_loss": None,
                "max_consecutive_loss": 0, "top3_share": None}
    pnl = np.asarray([float(r.get("pnl") or 0.0) for r in rows], dtype=np.float64)
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    gross_win = float(wins.sum())
    gross_loss = float(-losses.sum())
    # 最大连续亏损笔数
    run = best = 0
    for x in pnl:
        run = run + 1 if x < 0 else 0
        best = max(best, run)
    total_abs = float(np.abs(pnl).sum())
    tops = np.sort(np.abs(pnl))[::-1][:3].sum() if pnl.size else 0.0
    return {
        "n": int(pnl.size),
        "gross_win": gross_win,
        "gross_loss": gross_loss,
        "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else None,
        "expectancy": float(pnl.mean()),
        "avg_win": float(wins.mean()) if wins.size else None,
        "avg_loss": float(losses.mean()) if losses.size else None,
        "max_consecutive_loss": int(best),
        # 少数几笔贡献了多少——专业报告用来识别"靠运气"的集中度诊断
        "top3_share": (float(tops / total_abs) if total_abs > 0 else None),
    }


def report_flags(stats: dict) -> list[str]:
    """按专业回测报告的判读基准，给出**红旗提示**。

    阈值来源见 ``docs/GUI改进设计.md`` 的参考列表
    （profit factor >1.5 良好 / 期望值须为正 / <100 笔统计不可靠 /
    少数几笔贡献大部分利润是"运气"的信号）。
    """
    out: list[str] = []
    n = int(stats.get("n") or 0)
    if n < 20:
        out.append(f"成交只有 {n} 笔，样本太小，什么都说明不了（参考：≥100 笔才有统计信心）")
    elif n < 100:
        out.append(f"成交 {n} 笔，够看出方向但不够下结论（参考：≥100 笔才较可信）")
    pf = stats.get("profit_factor")
    if pf is not None:
        if pf < 1.0:
            out.append(f"盈亏比 {pf:.2f} < 1.0 —— 这个口径下是亏钱的")
        elif pf < 1.3:
            out.append(f"盈亏比 {pf:.2f} 偏弱（<1.3 通常被认为脆弱）")
    exp = stats.get("expectancy")
    if exp is not None and exp < 0:
        out.append(f"每笔期望值 {exp:.1f} 为负 —— 没有优势")
    share = stats.get("top3_share")
    if share is not None and share > 0.5:
        out.append(f"前 3 笔贡献了 {share * 100:.0f}% 的总盈亏 —— 先确认这几笔是不是运气")
    return out
