"""策略评估工具：把"这个策略做得好不好"变成可比较、可证伪的数字。

为什么单独一个模块
------------------
跑完一场模拟，手里有的只是"这个主体最终赚了多少钱"。这个数字**几乎不可用**：

  · 它没有区分"赚了价差"和"运气好拿着多头"。
    做市商在一段上涨行情里做多，看起来赚钱，实际每笔成交都在接刀。
  · 它没有区分"主动追单"和"被动挂单"。挂单被吃是赚价差，
    主动吃单是付价差，两者混在一起平均掉之后什么都看不出来。
  · 它没有时间维度。"最后赚了 1000"和"过程里最大浮亏 50000、
    最后赚了 1000"是完全不同的两件事。
  · 单次结果方差极大。没有多种子，一次漂亮的结果说明不了任何事。

本模块给出四组互补的指标，覆盖策略评估的两个主要用途
（**做市/提供流动性** 与 **执行/建仓**）：

1. **Markout（成交后价格走势）** —— 评估"成交质量"的核心工具。
   成交后价格朝不利方向走 = 你接了有毒的单（adverse selection）。
   拆成三项后"赚价差"和"接刀"不再混在一起：
     ``capture``   成交瞬间相对中间价的优势（被动挂单为正）
     ``drift(h)``  成交后 h tick 的中间价变动（中性策略应 ≈ 0；负值 = 逆向选择）
     ``realized(h)`` = capture + drift（每单位成交的实际盈亏）

2. **PnL 精确分解** —— 三个分量之和**恒等于**权益变化（有测试保证）：
     ``pnl_capture``    逐笔相对成交时中间价赚到的
     ``pnl_inventory``  持仓期间中间价变动带来的
     ``pnl_initial``    初始底仓被重新估值的
   恒等式让"钱从哪来"没有含糊空间。

3. **库存风险** —— 时间加权平均库存、最大库存、贴平时长占比。
   做市商赚不赚钱往往不取决于价差，而取决于库存管不管得住。

4. **权益路径风险** —— 最大回撤、单位风险收益（Sharpe 的 tick 版本）。

一条必须说清楚的纪律
--------------------
所有指标都必须**跨种子报告均值 ± 标准误**，不能只看一次运行。
本市场是肥尾的，单次权益曲线的方差极大；
``scripts/lab.py`` 因此默认跑多个种子并给出标准误。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .types import EPS, Side

#: markout 的观察期（tick）。覆盖"立刻"到"一小时后"三档：
#: 1~5 看执行质量，20~60 看是否留下永久性价格影响。
DEFAULT_HORIZONS: tuple[int, ...] = (1, 2, 5, 20, 60)


# ----------------------------------------------------------------------
# 基础数据
# ----------------------------------------------------------------------
@dataclass(slots=True, frozen=True)
class Fill:
    """策略视角的一笔成交。"""

    tick: int
    side: Side          #: 策略在这一笔里是买还是卖
    price: float
    quantity: float
    mid: float          #: 成交**当时**的中间价（不是 tick 收盘价）
    is_maker: bool      #: 策略是被动方（挂单被吃）为 True
    counterparty: str


def mid_series(market, from_tick: int = 0) -> np.ndarray:  # noqa: ANN001
    """本场模拟的中间价序列（长度 = 实际跑过的 tick 数）。

    ``from_tick`` 用来**只取观测窗口**。GUI 与实验台都会先跑一段预热，
    再把策略注入进去；如果指标把预热段也算进去：

      · 回撤会把"策略还没上台"的那一段平坦区算进峰值；
      · ``pnl_initial_mark`` 会拿 tick 0 的价格当基准，
        而策略实际是从 ``t0`` 开始持有的 —— 底仓重估项直接算错。

    所以凡是有预热的场合，都应当显式传 ``from_tick``。
    """
    lo = max(0, int(from_tick))
    n = int(market.tick)
    mid = np.asarray(market.log.mid[lo:n], dtype=np.float64).copy()
    # 单边空缺时会写成 NaN，做前向填充——否则后续所有统计都被 NaN 吞掉
    bad = ~np.isfinite(mid) | (mid <= 0)
    if bad.any():
        idx = np.where(~bad, np.arange(mid.size), 0)
        np.maximum.accumulate(idx, out=idx)
        mid = mid[idx]
    return mid


def collect_fills(market, agent_id: str, from_tick: int = 0) -> list[Fill]:  # noqa: ANN001
    """把主日志里的成交过滤成"这个主体视角"的成交列表。

    ``from_tick`` 会**同时做两件事**：丢掉观测窗口之前的成交，
    并把保留下来的成交 tick **换算成窗口内的相对下标**
    （即 ``tick - from_tick``）。少了这一步，Markout 会去查窗口外的价格，
    或者直接因为下标越界而静默丢弃——两种都是错的。
    """
    lo = max(0, int(from_tick))
    out: list[Fill] = []
    for t in market.log.trades:
        if t.tick < lo:
            continue
        if t.buy_agent_id == agent_id:
            side: Side = "buy"
            maker = t.aggressor_side == "sell"
            cp = t.sell_agent_id
        elif t.sell_agent_id == agent_id:
            side = "sell"
            maker = t.aggressor_side == "buy"
            cp = t.buy_agent_id
        else:
            continue
        out.append(
            Fill(
                tick=int(t.tick) - lo,
                side=side,
                price=float(t.price),
                quantity=float(t.quantity),
                mid=float(t.mid_at_fill),
                is_maker=maker,
                counterparty=cp,
            )
        )
    return out


def _tstat(a: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    if a.size < 3:
        return float("nan")
    sd = a.std(ddof=1)
    return float(a.mean() / (sd / math.sqrt(a.size))) if sd > 0 else float("nan")


def _agg(a: np.ndarray) -> dict:
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0, "mean": float("nan"), "t": float("nan"), "sem": float("nan")}
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "t": _tstat(a),
        "sem": float(a.std(ddof=1) / math.sqrt(a.size)) if a.size > 1 else float("nan"),
    }


# ----------------------------------------------------------------------
# 1. Markout
# ----------------------------------------------------------------------
def markout(
    fills: Sequence[Fill],
    mid: np.ndarray,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> dict:
    """成交后的价格走势，按策略方向对齐（bp）。

    对每一笔成交，设 s = +1（策略买入）或 −1（策略卖出），m = 成交当时中间价：

        capture  = s·(m − price) / m · 1e4     成交瞬间相对中间价的优势（bp）
        drift(h) = s·(mid[t+h] − m) / m · 1e4  成交后价格朝哪走（bp）
        realized(h) = capture + drift(h)        每单位成交的实际盈亏（bp）

    怎么读这些数：

      · **被动挂单（maker）**：capture 为正（赚了半个价差），
        drift 若显著为负，说明成交后价格朝对自己不利的方向走——**逆向选择**。
        realized = capture + drift：做市商能不能赚钱，就看这一项是不是稳定为正。
      · **主动吃单（taker）**：capture 为负（付了半个价差）。
        drift 若显著为正，说明吃对了方向（有信息），能补回价差；
        若 drift ≈ 0，就是纯粹的亏价差。

    注意 ``drift`` 用的是**成交当时**的中间价（``Trade.mid_at_fill``），
    不是该 tick 的收盘中间价。一个 tick 内有上百笔成交，
    用收盘价当基准会让 drift 整体错位一整条 tick——足以把结论颠倒。
    """
    n = mid.size
    rows: dict[str, list[float]] = {"capture": [], "age_ticks": []}
    for h in horizons:
        rows[f"drift_{h}"] = []
        rows[f"realized_{h}"] = []
    maker_rows: dict[str, list[float]] = {f"drift_{h}": [] for h in horizons}
    taker_rows: dict[str, list[float]] = {f"drift_{h}": [] for h in horizons}

    for f in fills:
        m = f.mid
        if not (m and m > 0 and np.isfinite(f.price)):
            continue
        s = 1.0 if f.side == "buy" else -1.0
        cap = s * (m - f.price) / m * 1e4
        rows["capture"].append(cap)
        rows["age_ticks"].append(n - 1 - f.tick if f.tick < n else 0)
        tgt = maker_rows if f.is_maker else taker_rows
        for h in horizons:
            j = f.tick + h
            if j >= n:
                continue
            mh = mid[j]
            if not (mh > 0):
                continue
            drift = s * (mh - m) / m * 1e4
            rows[f"drift_{h}"].append(drift)
            rows[f"realized_{h}"].append(cap + drift)
            tgt[f"drift_{h}"].append(drift)

    out = {"capture_bp": _agg(np.asarray(rows["capture"]))}
    for h in horizons:
        out[f"drift_{h}_bp"] = _agg(np.asarray(rows[f"drift_{h}"]))
        out[f"realized_{h}_bp"] = _agg(np.asarray(rows[f"realized_{h}"]))
        out[f"maker_drift_{h}_bp"] = _agg(np.asarray(maker_rows[f"drift_{h}"]))
        out[f"taker_drift_{h}_bp"] = _agg(np.asarray(taker_rows[f"drift_{h}"]))
    return out


# ----------------------------------------------------------------------
# 2. PnL 分解
# ----------------------------------------------------------------------
def strategy_paths(
    market,  # noqa: ANN001
    agent_id: str,
    from_tick: int = 0,
) -> tuple[list[Fill], dict[str, np.ndarray]]:
    """从成交记录反推策略的完整路径：现金、库存、权益（逐 tick）。

    **不需要额外日志**：起始状态 = 末状态 − 全部流量。
    比"在 Agent 里每 tick 记一次"更可靠——不会漏记，也不会因为
    主体没被调度到而断点。
    """
    mid = mid_series(market, from_tick)
    n = mid.size
    agent = market.by_id[agent_id]
    fills = collect_fills(market, agent_id, from_tick)

    dq = np.zeros(n, dtype=np.float64)
    dcash = np.zeros(n, dtype=np.float64)
    for f in fills:
        t = min(max(f.tick, 0), n - 1)
        s = 1.0 if f.side == "buy" else -1.0
        dq[t] += s * f.quantity
        dcash[t] += -s * f.quantity * f.price

    inv0 = float(agent.inventory) - float(dq.sum())
    cash0 = float(agent.cash) + float(-dcash.sum())

    # ⚠️ 权益路径必须**补一个"成交之前"的起点**，否则序列的第 0 个点
    # 是"第 0 个 tick 的成交做完之后"的状态，而恒等式假设它是成交之前的。
    # 从 from_tick=0 调用时这个错位侥幸看不出来（第一笔成交通常不在下标 0），
    # 一旦有预热、成交被换算到下标 0，残差立刻出现（实测 121.51）。
    mid_pre = float(mid[0])
    if from_tick > 0:
        prev = int(market.tick)
        if from_tick - 1 < prev:
            v = float(market.log.mid[from_tick - 1])
            if np.isfinite(v) and v > 0:
                mid_pre = v
    inv = np.concatenate(([inv0], inv0 + np.cumsum(dq)))
    cash = np.concatenate(([cash0], cash0 + np.cumsum(dcash)))
    equity = cash + inv * np.concatenate(([mid_pre], mid))
    return fills, {
        "mid": mid,
        "inventory": inv,
        "cash": cash,
        "equity": equity,
        "inv0": float(inv0),
        "cash0": float(cash0),
        "mid_pre": mid_pre,
    }


def pnl_breakdown(
    fills: Sequence[Fill],
    mid: np.ndarray,
    inv0: float = 0.0,
    mid_pre: float | None = None,
) -> dict:
    """PnL 三分解。三项之和**恒等于**权益变化（有测试保证这个恒等式）。

    pnl_total = pnl_capture + pnl_inventory + pnl_initial_mark

      pnl_capture       Σ s·q·(mid_i − price_i)   逐笔相对**成交时**中间价赚到的
      pnl_inventory     Σ s·q·(mid_end − mid_i)   持仓期间中间价变动带来的
      pnl_initial_mark  inv0·(mid_end − mid_pre) 初始底仓被重新估值的

    其中 ``mid_pre`` 是**窗口开始那一刻、还没有任何成交之前**的中间价。
    传 `None` 时退化成 `mid[0]`（窗口从第 0 个 tick 起、且底仓为 0 的场合等价）。

    为什么这个分解有用：它把"策略做得好"与"行情帮了忙"彻底分开。
    一个在市场上涨期持有多头的策略，``pnl_initial_mark`` 会很大——
    那是行情给的，不是策略赚的；而它可能根本**没有交易过**
    （``pnl_capture`` 和 ``pnl_inventory`` 都是 0）。
    做市商应该靠 ``pnl_capture`` 吃饭，``pnl_inventory`` 是不可避免的副产品，
    应当尽量小。
    """
    n = mid.size
    if n == 0:
        return {}
    cap = 0.0
    inv_flow = 0.0
    dq = 0.0
    for f in fills:
        s = 1.0 if f.side == "buy" else -1.0
        m = f.mid if (f.mid and f.mid > 0) else mid[min(max(f.tick, 0), n - 1)]
        cap += s * f.quantity * (m - f.price)
        inv_flow += s * f.quantity * (mid[-1] - m)
        dq += s * f.quantity
    base = float(mid[0]) if mid_pre is None or not np.isfinite(mid_pre) else float(mid_pre)
    initial_mark = float(inv0) * float(mid[-1] - base)
    return {
        "pnl_capture": float(cap),
        "pnl_inventory": float(inv_flow),
        "pnl_initial_mark": float(initial_mark),
        "pnl_flow": float(cap + inv_flow + initial_mark),
        "net_qty": float(dq),
        "mid_end": float(mid[-1]),
        "mid_start": float(mid[0]),
        "mid_pre": base,
    }


def drawdown_stats(equity: np.ndarray) -> dict:
    """权益路径的最大回撤（金额与百分比）与"单位风险收益"。"""
    e = np.asarray(equity, dtype=np.float64)
    e = e[np.isfinite(e)]
    if e.size < 3:
        return {"max_dd": float("nan"), "max_dd_pct": float("nan"), "sharpe": float("nan")}
    peak = np.maximum.accumulate(e)
    dd = e - peak
    # 相对回撤要用"峰值权益"做分母；峰值可能为 0 或负（纯空头账户），此时跳过
    denom = np.where(np.abs(peak) > EPS, np.abs(peak), np.nan)
    dd_pct = dd / denom
    r = np.diff(e) / np.where(np.abs(e[:-1]) > EPS, np.abs(e[:-1]), np.nan)
    r = r[np.isfinite(r)]
    sd = r.std(ddof=1) if r.size > 1 else float("nan")
    return {
        "max_dd": float(dd.min()),
        "max_dd_pct": float(np.nanmin(dd_pct)) if np.isfinite(dd_pct).any() else float("nan"),
        # 逐 tick 收益的单位风险收益。**不做年化**：年化要先约定 1 tick = 多久，
        # 本项目的约定是 1 tick ≈ 1 小时，但那只是刻度，不是真实时序。
        # 需要与其他策略横向比较时，这个原始比值就够用。
        "sharpe_per_tick": float(r.mean() / sd) if sd and sd > 0 else float("nan"),
    }


def inventory_stats(inv: np.ndarray) -> dict:
    """库存风险：做市/执行策略赚不赚钱常常取决于库存管不管得住。"""
    x = np.asarray(inv, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {}
    flat = np.abs(x) < 1e-6
    return {
        "inv_mean": float(x.mean()),
        "inv_abs_mean": float(np.abs(x).mean()),
        "inv_abs_max": float(np.abs(x).max()),
        "inv_std": float(x.std(ddof=1)) if x.size > 1 else float("nan"),
        "flat_frac": float(flat.mean()),
        "inv_end": float(x[-1]),
    }


# ----------------------------------------------------------------------
# 3. 汇总
# ----------------------------------------------------------------------
def evaluate(
    market,  # noqa: ANN001
    agent_id: str,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    from_tick: int = 0,
) -> dict:
    """把一个策略在本场模拟里的表现压成一个字典。

    ``from_tick`` 应当设为**策略被注入的那个 tick**。
    默认 0 是给"策略从第 0 个 tick 就参与"的场合；
    凡是有预热的实验台/GUI 都必须显式传，否则底仓重估项会算错。
    """
    fills, path = strategy_paths(market, agent_id, from_tick)
    agent = market.by_id[agent_id]
    mid = path["mid"]
    equity = path["equity"]

    maker_qty = sum(f.quantity for f in fills if f.is_maker)
    total_qty = sum(f.quantity for f in fills) or float("nan")
    eq0 = float(equity[0]) if equity.size else float("nan")
    eq_end = float(equity[-1]) if equity.size else float("nan")

    out: dict = {
        "agent_id": agent_id,
        "from_tick": int(from_tick),
        "kind": getattr(agent, "KIND", "?"),
        "n_ticks": int(mid.size),
        "n_fills": len(fills),
        "volume": float(sum(f.quantity for f in fills)),
        "n_submitted": int(agent.stats.n_submitted),
        "n_cancelled": int(agent.stats.n_cancelled),
        "n_rejected": int(agent.stats.n_rejected),
        # 被动成交占比：做市类策略的核心结构指标
        "maker_volume_frac": float(maker_qty / total_qty) if total_qty == total_qty else float("nan"),
        "equity_start": eq0,
        "equity_end": eq_end,
    }
    out.update(
        pnl_breakdown(
            fills, mid, inv0=float(path["inv0"]), mid_pre=float(path["mid_pre"])
        )
    )
    # 权益变化，用来核对分解恒等式（测试会断言两者相等）
    out["pnl_total_from_equity"] = float(eq_end - eq0) if np.isfinite(eq0) else float("nan")
    out["pnl_identity_residual"] = float(out["pnl_flow"] - out["pnl_total_from_equity"])
    out.update(markout(fills, mid, horizons))
    out.update(drawdown_stats(equity))
    out.update(inventory_stats(path["inventory"]))
    return out


def _fmt(v, nd: int = 1) -> str:
    if v is None:
        return "—"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not np.isfinite(x):
        return "—"
    return f"{x:,.{nd}f}"


#: 汇总表列定义：(列名, 取值路径, 小数位, 是否显著性列)
TABLE_COLS: list[tuple[str, str, int]] = [
    ("策略", "label", 0),
    ("成交笔数", "n_fills", 0),
    ("成交量", "volume", 1),
    ("被动占比", "maker_volume_frac", 3),
    ("价差捕获(bp)", "capture_bp|mean", 2),
    ("漂移h1(bp)", "drift_1_bp|mean", 2),
    ("漂移h20(bp)", "drift_20_bp|mean", 2),
    ("实现h20(bp)", "realized_20_bp|mean", 2),
    ("h20的t", "drift_20_bp|t", 2),
    ("PnL合计", "pnl_total_from_equity", 0),
    ("其中价差", "pnl_capture", 0),
    ("其中库存", "pnl_inventory", 0),
    ("最大回撤", "max_dd", 0),
    ("库存均值", "inv_abs_mean", 2),
    ("贴平占比", "flat_frac", 3),
]


def _pick(d: dict, key: str):
    cur = d
    for part in key.split("|"):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def format_table(rows: list[dict]) -> str:
    """把若干策略的评估结果排成一张对照表。"""
    heads = [c[0] for c in TABLE_COLS]
    widths = [max(len(h), 10) for h in heads]
    body = []
    for r in rows:
        line = []
        for name, key, nd in TABLE_COLS:
            v = _pick(r, key)
            s = str(v) if key == "label" else _fmt(v, nd)
            line.append(s)
            widths[0] = max(widths[0], len(heads[0]))
        body.append(line)
    for i, h in enumerate(heads):
        widths[i] = max(widths[i], *(len(b[i]) for b in body)) if body else widths[i]

    def row(cells: list[str]) -> str:
        return "  ".join(c.rjust(widths[i]) for i, c in enumerate(cells))

    out = [row(heads), "-" * (sum(widths) + 2 * (len(widths) - 1))]
    out += [row(b) for b in body]
    return "\n".join(out)
