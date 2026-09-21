"""分层评估（A4）—— 把一次回路运行拆成**三层**分别报，并与基线对照。

为什么必须分层（设计方案 §2）
------------------------------
「**Agent 可能在一层有用、在另一层有害**」——混在一起评就看不出来。
三层是：**决策**（方向/仓位/弃权）·**执行**（订单类型/时机/成本）·
**结果**（毛收益 / 净收益）。本模块按这三层分别出数字。

⭐ 最关键的一条：**LLM 必须显著优于 ``noop``**。
项目里 ``noop`` 的 PnL 恰好是 0——这条基准线现成且无争议。
如果 Agent 打不过 noop，那它做的所有"分析"都只是在给自己找理由交易。

⚠️ 本模块刻意**不做**的三件事（都会制造假结论）
------------------------------------------------
1. **不把"点估计的差距"叫作结论。** 两个 Agent 的收益差 3 倍，
   在样本很小时可能完全是噪声。所以比较**一律走区间重叠检验**
   （复用 ``tw/analyzer_consistency.py``，它承载了本项目最贵的教训）。
2. **不给"成本 ×2 后"的分析式外推当结论。** 成本变了，策略**可能
   就会改变行为**（多交易一次就多亏一次，甚至触发强平）。
   本模块的做法是：**用同一批已经录下来的决策重跑一遍**（回放，
   零额外 API 成本），而不是在旧收益上减一个数。
   ``cost_sensitivity_analytic`` 只作为**一阶参考**保留，且在返回里
   明确标注它是近似。
3. **不把"未成交"当"没发生"。** 挂单没成交、下单被风控拒掉，
   都是真实结果，要计入 ``n_decisions_not_traded``。
   只统计成交的那些会系统性高估成交率。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from .agent_run import RunResult
from .analyzer_consistency import pairwise_verdict

#: 夏普的年化因子（1H K 线 → 一年约 24×365 根）。
BARS_PER_YEAR_1H = 24 * 365


def json_safe(o: Any) -> Any:
    """把 ``NaN`` / ``±Inf`` 递归换成 ``None``。

    ⚠️⚠️ 这不是"顺手清理"，是一个**真踩过的互操作 bug**：

    ``json.dumps(float("nan"))`` 在 Python 里默认输出裸的 ``NaN``——
    而 **``NaN`` 不是合法 JSON**（RFC 8259 没有它）。后果：
      · Python 自己读得回来（它的解析器**超集**接受 NaN），所以**自测全绿**；
      · 但 JS 的 ``JSON.parse`` 直接报 ``Unexpected token 'N'``；
      · Go / Rust / Java 的标准 JSON 库同样拒绝。
    ⇒ **产物对任何非 Python 消费者都是坏的，而我们自己发现不了。**

    本项目真的撞上了：把评估 JSON 嵌进页面时，浏览器的
    ``JSON.parse`` 抛错、整页退化成"读取失败"，而 Python 侧一切正常。
    来源是 ``sharpe``：不交易（如 ``noop``）时它按设计返回 ``nan``
    （"算不出来"），于是**最基准的那条配置**产出了非法 JSON。

    修法：**写文件/嵌入页面前一律过这个函数**，并配
    ``json.dumps(..., allow_nan=False)`` —— 让它**再也不能**悄悄写出 NaN。
    """
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_safe(v) for v in o]
    return o

#: 置信度校准的默认前瞻窗口（根）。
DEFAULT_CALIB_HORIZON = 4

#: 理由里出现这些词 ⇒ 大概率引用了**快照里没有的外部信息**。
#: ⚠️ 这是**关键词启发式**，不是判定器。它的用途是**把可疑样本捞出来
#: 让人看**，不是自动给结论。词表要小而准：宁可漏，不要误报——
#: 误报多了这条指标会被整体忽略（与 `_is_leaky_key` 的教训同源）。
_EXTERNAL_TERMS = (
    "标普", "纳斯达克", "道琼斯", "恒指", "恒生", "日经",
    "美联储", "加息", "降息", "非农", "CPI", "议息",
    "美股", "A股", "港股", "黄金", "原油", "美元指数",
    "ETF", "灰度", "减半", "宏观", "新闻", "消息面",
)


# ======================================================================
# 小工具
# ======================================================================
def _pct(n: int, d: int) -> float:
    return float(n) / d if d else 0.0


def _quantile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return float(s[k])


def per_bar_returns(equity: list[float]) -> list[float]:
    """逐根收益率。⚠️ 两个权益点之间才是**一根**的收益——
    曲线长 T+1 就给 T 个收益，不能少（少一个会让夏普偏）。
    本项目的权益曲线**从成交前那格开始**（见 ``agent_run``），
    所以这里天然对得上。"""
    out: list[float] = []
    for a, b in zip(equity, equity[1:]):
        if a and a > 0:
            out.append(b / a - 1.0)
    return out


def max_drawdown(equity: list[float]) -> float:
    """最大回撤（正数，表示从峰值掉下来的比例）。"""
    peak, mdd = -math.inf, 0.0
    for e in equity:
        peak = max(peak, e)
        if peak > 0:
            mdd = max(mdd, (peak - e) / peak)
    return float(mdd)


def sharpe(returns: list[float], *, bars_per_year: int = BARS_PER_YEAR_1H) -> float:
    """年化夏普（不做无风险利率调整）。

    ⚠️ 样本 < 2 或方差为 0 时返回 ``nan`` 而不是 0——
    0 会被读成"夏普为零"，而真相是"算不出来"。两者完全不同。
    """
    n = len(returns)
    if n < 2:
        return float("nan")
    mu = sum(returns) / n
    var = sum((r - mu) ** 2 for r in returns) / (n - 1)
    if var <= 0:
        return float("nan")
    return float(mu / math.sqrt(var) * math.sqrt(bars_per_year))


def block_bootstrap_ci(
    returns: list[float],
    *,
    n_boot: int = 2000,
    block: int = 24,
    alpha: float = 0.05,
    seed: int = 12345,
) -> tuple[float, float]:
    """**块**自助法给出的"均值收益率"置信区间。

    ⚠️ 为什么必须是**块**而不是逐点重抽：逐点重抽会**打散自相关**，
    于是把区间算得过窄（项目里实测过：块自助法对本项目 ACF 型矩
    本来就**系统性低估**，块长 168 只恢复 0.9%——报告里要写明这条限制）。
    本函数把它用在这里是**相对**比较用途（两个 Agent 同法处理），
    所以系统性低估对"谁更高"这个方向判断影响较小，
    但会**低估所需样本量**。这个限制必须写在报告里。

    固定 ``seed`` ⇒ 结果可复现。
    """
    import random

    n = len(returns)
    if n < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    b = max(1, min(int(block), n))
    n_blocks = max(1, (n + b - 1) // b)
    means: list[float] = []
    for _ in range(int(n_boot)):
        acc = 0.0
        cnt = 0
        for _ in range(n_blocks):
            st = rng.randrange(0, n)
            for k in range(b):
                acc += returns[(st + k) % n]
                cnt += 1
        means.append(acc / max(cnt, 1))
    means.sort()
    lo = means[min(len(means) - 1, int(alpha / 2 * len(means)))]
    hi = means[min(len(means) - 1, int((1 - alpha / 2) * len(means)))]
    return (float(lo), float(hi))


# ======================================================================
# 三层
# ======================================================================
def decision_layer(res: RunResult) -> dict[str, Any]:
    """**决策层**：弃权率、置信度分布与校准、理由可用性、外部引用扫描。"""
    recs = res.records
    n = len(recs)
    if n == 0:
        return {"n": 0}
    n_hold = sum(1 for r in recs if r.parsed.get("action") == "hold")
    n_parse_fail = sum(1 for r in recs if not r.parse_ok)

    confs = [float(r.parsed["confidence"]) for r in recs
             if isinstance(r.parsed.get("confidence"), (int, float))]
    reasons = [str(r.parsed.get("reason") or "") for r in recs]
    n_reason = sum(1 for x in reasons if len(x.strip()) >= 8)

    hits: dict[str, int] = {}
    for rs in reasons:
        for w in _EXTERNAL_TERMS:
            if w in rs:
                hits[w] = hits.get(w, 0) + 1

    out: dict[str, Any] = {
        "n": n,
        "abstain_frac": _pct(n_hold, n),
        "parse_fail_frac": _pct(n_parse_fail, n),
        "rejected_frac": _pct(sum(1 for r in recs if r.reject_code), n),
        "resized_frac": _pct(
            sum(1 for r in recs
                if r.risk.get("resized_to") is not None
                and r.risk.get("resize_is_material", True)), n),
        "confidence_mean": (sum(confs) / len(confs)) if confs else None,
        "confidence_p50": _quantile(confs, 0.5) if confs else None,
        "reason_usable_frac": _pct(n_reason, n),
        # ⚠️ 关键词启发式（见 _EXTERNAL_TERMS 的注释）
        "external_ref_terms": hits,
        "external_ref_frac": _pct(
            sum(1 for rs in reasons if any(w in rs for w in _EXTERNAL_TERMS)), n),
    }
    # ⭐ 决策依据自报分布（v4 起）。老模板自然全是"未自报"。
    out["basis"] = basis_distribution(recs)
    return out


#: 决策依据自报的显示名。**唯一一份**——
#: ``gui.agent_api`` 从这里转发，不另写。
#: （本项目的老教训：同一个量有多份实现，就一定会分叉。）
BASIS_LABELS: dict[str, str] = {

    "rule": "照指标执行",
    "judgment": "凭自己判断",
    "both": "两者结合",
    "": "未自报",
}


def basis_distribution(recs: list) -> dict[str, Any]:
    """统计模型**自报**的决策依据分布。

    ⚠️ 这是**模型自己说的**，不是我们测出来的。它可能为了"显得一致"
    而把两种理由都写上（于是 ``both`` 虚高）。⇒ 它是**声明**，
    不是**证据**。要交叉验证，得看两类在**实际行为**上的差异
    （换手率、与 momentum 信号的一致率、置信度分布）。
    报告里必须这么写，否则这个数字会被当成"模型真的这么做了"。
    """
    counts: dict[str, int] = {}
    for r in recs:
        k = str(r.parsed.get("basis") or "")
        if k not in BASIS_LABELS:
            k = ""
        counts[k] = counts.get(k, 0) + 1
    n = sum(counts.values())
    if not n:
        return {"n": 0, "counts": {}, "frac": {}, "labels": BASIS_LABELS}
    # 只报"有自报的"那一部分的比例（v1~v3 全未自报时不要给一堆 100%）
    n_declared = n - counts.get("", 0)
    return {
        "n": n,
        "n_declared": n_declared,
        "counts": counts,
        "frac": {k: v / n for k, v in counts.items()},
        "frac_of_declared": (
            {k: v / n_declared for k, v in counts.items() if k}
            if n_declared else {}),
        "labels": BASIS_LABELS,
        "caveat": "这是**模型自报**的依据，不是测出来的；只作声明看。",
    }


def confidence_calibration(
    res: RunResult,
    closes: list[float],
    *,
    horizon: int = DEFAULT_CALIB_HORIZON,
    bins: tuple[float, ...] = (0.0, 0.4, 0.6, 0.8, 1.01),
) -> dict[str, Any]:
    """置信度校准：**说 0.7 的是不是真 70% 对**。

    "对"的定义：决策方向与之后 ``horizon`` 根的**收益方向一致**。
    ⚠️ 这个定义有三个必须写进报告的局限：
      1. **弃权不参与**（弃权没有方向，无法判对错——它由"弃权率"单独衡量）；
      2. 只用了 ``horizon`` 根的价格变化，**不含成本**——所以它是
         **方向准确率**，不是"赚不赚钱"；
      3. 样本通常很小 ⇒ 每个分箱里的比例极不稳定。
         因此本函数**同时返回每箱的 n**，让读者自己判断可信度。
    """
    rows: list[tuple[float, int]] = []
    for r in res.records:
        c = r.parsed.get("confidence")
        a = r.parsed.get("action")
        if not isinstance(c, (int, float)) or a not in ("buy", "sell"):
            continue
        # ``tick`` 就是 K 线索引（``run_agent_session`` 传的是 i），
        # 所以可以直接拿它去索引 ``closes``。
        t = int(r.tick)
        if t + horizon >= len(closes):
            continue
        p0 = float(closes[t])
        p1 = float(closes[t + horizon])
        if p0 <= 0:
            continue
        up = p1 > p0
        ok = 1 if (up and a == "buy") or ((not up) and a == "sell") else 0
        rows.append((float(c), ok))

    if not rows:
        return {"n": 0, "bins": []}

    out_bins = []
    for lo, hi in zip(bins, bins[1:]):
        sel = [ok for c, ok in rows if lo <= c < hi]
        if not sel:
            out_bins.append({"lo": lo, "hi": hi, "n": 0,
                             "predicted_mean": None, "realized": None, "gap": None})
            continue
        pred = sum(c for c, _ in rows if lo <= c < hi) / len(sel)
        real = sum(sel) / len(sel)
        out_bins.append({"lo": lo, "hi": hi, "n": len(sel),
                         "predicted_mean": pred, "realized": real,
                         "gap": real - pred})
    n = len(rows)
    return {
        "n": n,
        "horizon_bars": horizon,
        "bins": out_bins,
        "overall_realized": sum(ok for _, ok in rows) / n,
        "overall_predicted": sum(c for c, _ in rows) / n,
        "note": ("方向准确率，不含成本；样本小则每箱不稳定，"
                 "请连同 n 一起读。弃权不参与校准。"),
    }


#: 由**提交的订单**产生的成交原因。
#: ⚠️ TP/SL（``tp``/``sl``）与强平（``liquidate``）**不在**这里：
#: 它们是挂在仓位上的触发单，**没有对应的"已提交订单"**。
#: 把它们算进"订单成交率"会让比率超过 100%——这个 bug 真的出现过
#: （llm_v2 显示成交率 **142%**，19 张订单 27 笔成交），
#: 而 >100% 的比率看起来只是"数字有点怪"，很容易被放过。
_ORDER_FILL_REASONS = frozenset({"market", "limit", "post_only", "ioc", "fok"})


def execution_layer(res: RunResult) -> dict[str, Any]:
    """**执行层**：成交率、滑点（实际 vs 预期）、订单类型分布、延迟。"""
    recs = res.records
    n = len(recs)
    n_exec = sum(1 for r in recs if r.executed)
    fills = res.fills
    slip_bps = [f.slippage_bps for f in fills if f.ref_price]
    slip_cost = [f.slippage_cost for f in fills]
    lat = [int(r.latency_ms) for r in recs if int(r.latency_ms) > 0]
    lat.sort()

    by_reason: dict[str, int] = {}
    for f in fills:
        by_reason[f.reason] = by_reason.get(f.reason, 0) + 1
    # ⚠️ 分两类：订单成交 vs 触发单成交（TP/SL/强平）。
    # 分母必须与分子同源，否则比率可以超过 100%。
    n_order_fills = sum(1 for f in fills if f.reason in _ORDER_FILL_REASONS)
    n_exit_fills = sum(1 for f in fills if f.reason in ("tp", "sl"))
    n_liq_fills = sum(1 for f in fills if f.reason == "liquidate")

    # ⚠️ 成交率的分母是**落成的订单**，不是全部决策——
    # 弃权根本没有订单，"弃权率高"不该被读成"成交率低"。
    n_orders = sum(1 for r in recs if r.order)
    return {
        "n_decisions": n,
        "n_executed_intents": n_exec,
        "n_orders_built": n_orders,
        "n_decisions_not_traded": n - n_exec,
        #: 订单成交率 = 由订单产生的成交 / 落成的订单（≤ 100%）
        "fill_rate_of_orders": _pct(n_order_fills, n_orders),
        "n_order_fills": n_order_fills,
        "n_exit_fills_tp_sl": n_exit_fills,
        "n_liquidate_fills": n_liq_fills,
        "n_fills": len(fills),
        "fills_by_reason": by_reason,
        "slippage_bps_mean": (sum(slip_bps) / len(slip_bps)) if slip_bps else 0.0,
        "slippage_bps_p50": _quantile(slip_bps, 0.5),
        "slippage_bps_max": max(slip_bps) if slip_bps else 0.0,
        "slippage_cost_total": float(sum(slip_cost)),
        "latency_ms_p50": lat[len(lat) // 2] if lat else 0,
        "latency_ms_max": lat[-1] if lat else 0,
        "n_expired_orders": int(res.meta.get("exec", {}).get("n_expired", 0)),
        "n_liquidations": int(res.meta.get("exec", {}).get("n_liquidations", 0)),
    }


def result_layer(res: RunResult) -> dict[str, Any]:
    """**结果层**：毛/净收益、夏普、最大回撤、换手、持仓期、交易/不交易数。

    ⚠️ **毛 vs 净的口径**（必须写清楚，否则数字会被误读）：
    - 账本 ``cash`` 里**已经扣过手续费**（``apply_fill`` 就扣），
      所以权益变化 = **净收益**。
    - **毛收益** = 净收益 + 手续费 + 滑点成本（把成本加回去）。
      ⚠️ 加回滑点是**近似**：限价单成交在自己的价上，本来就没有滑点，
      而那部分 ``slippage_cost`` 是 0，加回去没有副作用；
      市价单的滑点则是真实付出的成本，加回去得到"如果按参考价成交"的收益。
      这个口径不一致是**故意**的：它让"毛收益"读作
      「**假设不付任何交易成本**」。报告里必须这么写。
    """
    eq = res.equity_curve
    rets = per_bar_returns(eq)
    net = res.final_equity - res.initial_equity
    fees = float(sum(f.fee for f in res.fills))
    slip = float(sum(f.slippage_cost for f in res.fills))
    gross = net + fees + slip
    base = res.initial_equity
    rt = res.round_trips()
    bip = res.bars_in_position()
    return {
        "initial_equity": base,
        "final_equity": res.final_equity,
        "net_pnl": float(net),
        "net_return": (net / base) if base else float("nan"),
        "gross_pnl": float(gross),
        "gross_return": (gross / base) if base else float("nan"),
        "fees_paid": fees,
        "slippage_cost": slip,
        "turnover": float(sum(f.notional for f in res.fills)),
        "turnover_x_equity": (sum(f.notional for f in res.fills) / base)
        if base else float("nan"),
        "sharpe": sharpe(rets),
        "max_drawdown": max_drawdown(eq),
        "n_bars": len(eq),
        "n_trades": len(res.fills),
        "n_round_trips": rt,
        "bars_in_position": bip,
        "avg_holding_bars": (bip / rt) if rt else 0.0,
        "exposure_frac": _pct(bip, len(eq)),
    }


# ======================================================================
# 成本敏感性
# ======================================================================
def cost_sensitivity_analytic(res: RunResult,
                              multipliers: tuple[float, ...] = (1.0, 2.0, 5.0),
                              ) -> list[dict[str, Any]]:
    """**一阶**成本敏感性（不重跑）。

    净收益 −（额外手续费 + 额外滑点），其中"额外" = 原成本 ×(m−1)。

    ⚠️ **这是近似，且只在这里作为参考**。它假设"成本变了、
    交易行为不变"——而真实情况下成本变高可能让某些单不再划算
    （或被强平）。**正式结论请用重跑（回放）的那一版**
    （:func:`cost_sensitivity_rerun`），两者的差值本身就是信息：
    差得越多，说明行为对成本越敏感。
    """
    base = res.initial_equity
    fees = float(sum(f.fee for f in res.fills))
    slip = float(sum(f.slippage_cost for f in res.fills))
    net = res.final_equity - res.initial_equity
    out = []
    for m in multipliers:
        extra = (fees + slip) * (max(m, 1.0) - 1.0)
        out.append({
            "cost_multiplier": float(m),
            "net_pnl": float(net - extra),
            "net_return": ((net - extra) / base) if base else float("nan"),
            "extra_cost": float(extra),
            "method": "analytic_approx",
        })
    return out


#: 回放覆盖率低于这个值时，**拒绝**把重跑结果当成本敏感性结论。
#: 理由见 :func:`cost_sensitivity_rerun` 的文档——那时数字是"回放失败"的
#: 产物，不是"策略行为"的产物。
MIN_REPLAY_COVERAGE = 0.95


def cost_sensitivity_rerun(
    runs: dict[float, RunResult],
    *,
    min_coverage: float = MIN_REPLAY_COVERAGE,
) -> list[dict[str, Any]]:
    """**重跑版**成本敏感性（每个倍数各跑一次真实回路）。

    ⚠️⚠️ **回放版的成本敏感性对 LLM 是不成立的**——这一条是本项目
    实测踩出来的，必须写清楚：

    LLM 的 prompt 里**含账户状态**（持仓、权益），而账户状态取决于成交、
    成交取决于成本。所以"换个成本重放同一段行情"会让路径**迅速发散**：
    实测 BTC 上成本 ×1 的回放覆盖率 100%（逐笔复现真机），
    成本 ×2 时掉到 **0.8%**（363 次调用只命中 3 次）——
    于是 Agent 几乎全程弃权，交易从 27 笔变成 2 笔，
    而账面净收益从 −33 变成 −444。

    **那个 −444 看起来像一个成本敏感性结果，其实是"回放失败"。**

    ⇒ 本函数因此**检查覆盖率**：低于 ``min_coverage`` 时，
    这一档**不出数字**，只标 ``valid=False`` 并说明原因。
    这与本项目「宁可报'测不出来'，也不要一个漂亮的假结论」一致。

    对**规则基线**而言回放是成立的：它们的决策**只依赖可见状态**，
    不含账户路径依赖，所以同一段行情换成本重跑是逐笔可复现的。
    """
    out = []
    for m in sorted(runs):
        r = runs[m]
        base = r.initial_equity
        net = r.final_equity - r.initial_equity
        cov = r.replay_coverage
        row: dict[str, Any] = {
            "cost_multiplier": float(m),
            "net_pnl": float(net),
            "net_return": (net / base) if base else float("nan"),
            "n_trades": len(r.fills),
            "n_liquidations": int(r.meta.get("exec", {}).get("n_liquidations", 0)),
            "max_drawdown": max_drawdown(r.equity_curve),
            "method": "rerun",
            "replay_coverage": cov,
        }
        if cov is not None and cov < min_coverage:
            # ⚠️ 数字**留着但标成无效**：直接删掉会让人以为"这一档没跑"，
            # 而真相是"跑了但回放失败"——两者要采取的行动完全不同。
            row["valid"] = False
            row["invalid_reason"] = (
                f"回放覆盖率只有 {cov:.1%}（阈值 {min_coverage:.0%}）："
                f"prompt 含账户状态 ⇒ 换成本后路径发散、大面积未命中，"
                f"Agent 的弃权是**回放失败**造成的，不是策略行为。"
                f"要测这一档必须**真的重跑**（花额度），不能用回放。"
            )
        else:
            row["valid"] = True
        out.append(row)
    return out


# ======================================================================
# 与基线的对照
# ======================================================================
def compare_to_baselines(
    target: RunResult,
    baselines: dict[str, RunResult],
    *,
    n_boot: int = 2000,
    block: int = 24,
    seed: int = 12345,
) -> dict[str, Any]:
    """把目标 Agent 的**逐根收益率均值**与每条基线对照。

    ⚠️ 比较的量是「**每根的平均收益率**」而不是「总收益」——
    总收益是单点、没有分布，无法做检验；逐根收益有 T 个观测，
    块自助法才能给出区间。这与本项目「点估计不是结论」那条纪律同源。

    ⭐ **两种对照要用两种检验**（这是本项目纪律 8 的原话：
    「单家族自己的方向判定」与「家族之间的比较判定」是**两件事**）：

    ==========================  ==========================================
    基线的情况                   用什么检验
    ==========================  ==========================================
    **零方差**（如 ``noop``：      **CI 是否排除基准值**——
    不交易 ⇒ 收益恒为 0）          问"目标的区间排不排除 0"
    有方差（其它交易型基线）        **区间重叠**——问"两个区间是不是分开"
    ==========================  ==========================================

    ⚠️ 第一行是**踩出来的**：最初一律走重叠检验，于是 vs ``noop``
    的输出是「无法判定（输入退化）」——因为 noop 的区间宽度是 0，
    重叠比例的分母没有意义。而"能不能打赢 noop"**恰恰是 A4
    最核心的那个问题**（设计方案 §6：「LLM 必须显著优于 noop」）。
    用错检验会让最关键的问题**答不出来**，而且看起来像"数据不够"。

    ⚠️ 同时提醒一条**已知限制**：块自助法会**低估**所需样本量
    （块长取小了会漏掉长记忆）。所以判「显著」时要比判「不显著」
    更谨慎——**本模块只在区间干净分离时才说"显著"**。
    """
    t_rets = per_bar_returns(target.equity_curve)
    t_ci = block_bootstrap_ci(t_rets, n_boot=n_boot, block=block, seed=seed)
    t_pt = sum(t_rets) / len(t_rets) if t_rets else float("nan")
    out: dict[str, Any] = {
        "target": target.name,
        "metric": "per_bar_return_mean",
        "target_point": t_pt,
        "target_ci": list(t_ci),
        "n_bars": len(t_rets),
        "bootstrap": {"n_boot": n_boot, "block": block, "seed": seed,
                      "note": "块自助法会低估所需样本量；判'显著'要比判'不显著'更谨慎"},
        "verdicts": [],
    }
    for nm, b in baselines.items():
        b_rets = per_bar_returns(b.equity_curve)
        if not b_rets:
            continue
        b_pt = sum(b_rets) / len(b_rets)
        b_ci = block_bootstrap_ci(b_rets, n_boot=n_boot, block=block, seed=seed)
        width = abs(b_ci[1] - b_ci[0])
        scale = max(abs(b_pt), 1e-12)
        if width <= scale * 1e-9:
            # ---- 零方差基线（noop）：改问"区间排不排除基准值" ----
            lo, hi = t_ci
            excludes = (lo > b_pt) or (hi < b_pt)
            # ⚠️ 零方差的基线（不交易 ⇒ 收益恒 0）用重叠检验会退化，
            # 所以这里改用"CI 是否排除基准值"。判据比重合严格：
            # 必须**整个区间都在基准之上（或之下）**才算显著。
            if excludes and lo > b_pt:
                verdict, direction = "显著优于", f"{target.name} 更高"
            elif excludes and hi < b_pt:
                verdict, direction = "显著劣于", f"{nm} 更高"
            else:
                verdict, direction = "依然无法判定", None
            out["verdicts"].append({
                "name_a": target.name, "name_b": nm,
                "point_a": t_pt, "point_b": b_pt,
                "ci_a": list(t_ci), "ci_b": list(b_ci),
                "overlap_length": float("nan"), "overlap_fraction": float("nan"),
                "point_diff": t_pt - b_pt,
                "verdict": verdict, "direction": direction,
                "required_n": None,
                "reason": (f"基线区间宽度为 0（{nm} 的收益恒为 {b_pt:g}），"
                           f"重叠比例无意义 ⇒ 改用「CI 是否排除基准值」："
                           f"目标 95% 区间 [{lo * 1e4:+.4f}, {hi * 1e4:+.4f}] bp/根 "
                           f"{'排除' if excludes else '包含'} {b_pt * 1e4:+.4f}"),
                "test": "ci_excludes_baseline",
                "threshold": None,
            })
            continue
        v = pairwise_verdict(target.name, t_ci, t_pt, nm, b_ci, b_pt,
                             current_n=float(len(t_rets)))
        d = v.as_dict()
        d["test"] = "ci_overlap"
        out["verdicts"].append(d)
    return out


# ======================================================================
# 总报告
# ======================================================================
def evaluate_run(res: RunResult, closes: list[float] | None = None) -> dict[str, Any]:
    """一个 Agent 的完整分层评估（没给 closes 就跳过置信度校准）。"""
    d: dict[str, Any] = {
        "name": res.name,
        "inst_id": res.inst_id,
        "bar": res.bar,
        "run_id": res.run_id,
        "meta": dict(res.meta),
        "exec_config": dict(res.exec_config),
        "decision": decision_layer(res),
        "execution": execution_layer(res),
        "result": result_layer(res),
        "cost_sensitivity_analytic": cost_sensitivity_analytic(res),
        "n_skipped": len(res.skipped),
    }
    if closes is not None:
        d["confidence_calibration"] = confidence_calibration(res, closes)
    return d


def format_report_lines(evals: dict[str, dict[str, Any]],
                        comparison: dict[str, Any] | None = None) -> list[str]:
    """把评估结果排成可直接贴进报告的纯文本。"""
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("分层评估（决策层 / 执行层 / 结果层）")
    lines.append("=" * 78)
    hdr = (f"{'配置':<16}{'决策':>5}{'弃权%':>8}{'成交':>6}"
           f"{'毛收益%':>10}{'净收益%':>10}{'回撤%':>8}{'夏普':>8}{'回合':>5}")
    lines.append(hdr)
    lines.append("-" * 78)
    for nm, e in evals.items():
        r, dec = e["result"], e["decision"]
        sh = r["sharpe"]
        lines.append(
            f"{nm:<16}{dec.get('n', 0):>5}"
            f"{dec.get('abstain_frac', 0) * 100:>8.1f}"
            f"{r['n_trades']:>6}"
            f"{r['gross_return'] * 100:>10.2f}"
            f"{r['net_return'] * 100:>10.2f}"
            f"{r['max_drawdown'] * 100:>8.2f}"
            f"{(sh if sh == sh else float('nan')):>8.2f}"
            f"{r['n_round_trips']:>5}"
        )
    lines.append("")
    lines.append("成本敏感性（净收益%）—— ⚠️ 标 `不可测` 的档位是**回放覆盖率过低**，"
                 "数字无效（详见 JSON 的 invalid_reason）：")
    for nm, e in evals.items():
        rr = {c["cost_multiplier"]: c for c in (e.get("cost_sensitivity_rerun") or [])}
        cs = e.get("cost_sensitivity_analytic", [])
        if rr:
            seg = []
            for c in cs:
                m = c["cost_multiplier"]
                r = rr.get(m)
                if r is None or not r.get("valid", True):
                    cov = (r or {}).get("replay_coverage")
                    tag = "不可测" if cov is None else f"不可测(回放{cov:.0%})"
                    seg.append(f"×{m:g}: {tag}")
                else:
                    seg.append(f"×{m:g}: {r['net_return'] * 100:+.3f}%")
            lines.append(f"  {nm:<16}" + "  ".join(seg))
        else:
            seg = "  ".join(f"×{c['cost_multiplier']:g}: {c['net_return'] * 100:+.3f}%"
                            for c in cs)
            lines.append(f"  {nm:<16}{seg}")

    if comparison:
        lines.append("")
        lines.append("与基线的区间重叠检验"
                     f"（度量：{comparison['metric']}，"
                     f"块自助 {comparison['bootstrap']['n_boot']}×"
                     f"block={comparison['bootstrap']['block']}）：")
        tp = comparison["target_point"]
        tlo, thi = comparison["target_ci"]
        lines.append(f"  {comparison['target']:<16}"
                     f"点估计 {tp * 1e4:+.4f} bp/根   "
                     f"95% 区间 [{tlo * 1e4:+.4f}, {thi * 1e4:+.4f}]")
        for v in comparison["verdicts"]:
            lines.append(f"  vs {v['name_b']:<14}{v['verdict']:<12}"
                         f"重叠 {v['overlap_fraction'] * 100:.1f}%   "
                         f"{v['reason'][:64]}")
    return lines


__all__ = [
    "BARS_PER_YEAR_1H",
    "json_safe",
    "BASIS_LABELS",
    "DEFAULT_CALIB_HORIZON",
    "decision_layer",
    "basis_distribution",
    "execution_layer",
    "result_layer",
    "confidence_calibration",
    "cost_sensitivity_analytic",
    "cost_sensitivity_rerun",
    "compare_to_baselines",
    "evaluate_run",
    "format_report_lines",
    "per_bar_returns",
    "max_drawdown",
    "sharpe",
    "block_bootstrap_ci",
]
