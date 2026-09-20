"""扫单冲击事件研究：扫单之后价格是**继续走**还是**反转**？

蓝图动机
--------
施工蓝图要检验的现象之一是"扫单后继续走，而不是立刻反转"。
这在真实数据上很难干净地测——公开 K 线没有订单流，只能用 CLV×成交量
之类的代理变量去猜哪根 K 线是被扫的（真实侧实测：事件 465 个，
即时响应 -5.7bp、t=-1.42，**不显著**，代理变量本身就把信号抹掉了）。

模拟市场的好处正在这里：**订单流是内生的、可见的**。可以直接定义
"扫单"= 单笔激进订单吃穿订单簿 ≥ ``MIN_LEVELS`` 个价位，不需要代理。

测量设计（四个指标，锚全部干净）
--------------------------------
记 p_pre = 扫单前中间价、p_now = 扫单刚砸完的中间价、p_end = 扫单所在
tick 的收盘中间价、p_h = 之后 h tick 的收盘中间价。按扫单方向对齐符号
（买单 +1、卖单 −1）后：

  ``imm_book``  = s·(p_now/p_pre − 1)   扫单对**盘口**的即时冲击
  ``move_tick`` = s·(p_end/p_pre − 1)   扫单当 tick 收盘相对扫单前
  ``cont_h``    = s·(p_h/p_end − 1)     从下一 tick 起的延续（>0）/反转（<0）
  ``cum_h``     = s·(p_h/p_pre − 1)     相对扫单前的累计移动

为什么必须四个都要
------------------
一开始我只用 ``p_now`` 当锚算后续移动（``s·(p_h/p_now − 1)``），
测出 h=1/2/5 的"延续"高达 +52/+47/+41bp、t 都在 3~5。
换成 ``p_end`` 当锚，同样的"延续"只剩 +11/+7/+0bp、**全部不显著**。
差额稳定在 ~40bp —— 那部分是**扫单后同一 tick 内**后续主体跟风推动的价格，
不是跨 tick 的趋势。

而 ``p_now`` 之所以会把同 tick 内的移动算成"之后"的延续，是因为它是
**瞬时倾斜的锚**：扫单刚吃完一侧盘口时，(best_bid+best_ask)/2 里的
一侧被推得很远，这个中间价并不代表"市场认可的价"；更糟的是当整侧被吃空时，
``current_mid()`` 会回退到最新成交价。用它做后续移动的基准，
"盘口恢复"会被误读成"价格继续走"。

结论纪律：只用 ``p_pre`` 与 ``p_end`` / ``p_h`` 这几个**真实盘口算出的**
中间价下判断；``p_now`` 只用于描述扫单对自己盘口的一次性冲击，
并且必须同时报告"砸完后盘口单边空缺"的比例，让读者知道这个数有多脏。
"""

from __future__ import annotations

import numpy as np

from _common import FIG, banner, real_metrics, save_json  # noqa: E402
from tw import Market, Population, SimConfig  # noqa: E402

N_AGENTS = 300
WARMUP = 6_000
N_TICKS = 24_000
SEED = 20260917
K_SEEDS = 4

MIN_LEVELS = 3      # 吃穿 ≥ 3 个价位才算"扫单"
MIN_QTY = 2.0       # 且总量 ≥ 2 手（滤掉碎单）
HORIZONS = [1, 2, 5, 10, 20, 60]
MAX_H = 60

BASE_MIX = {"zero_intel": 0.30, "fundamentalist": 0.40, "chartist": 0.30}
MM_MIX = {
    "zero_intel": 0.27,
    "fundamentalist": 0.36,
    "chartist": 0.27,
    "market_maker": 0.10,
}
MM_KW = dict(
    mm_base_spread=0.00005,
    mm_vol_sensitivity=0.05,
    mm_quote_qty=2.0,
    mm_inventory_target=10.0,
    mm_skew_strength=0.0004,
)


# --------------------------------------------------------------------------
# 捕捉
# --------------------------------------------------------------------------
def run_and_detect(mix: dict[str, float], seed: int, n_ticks: int = N_TICKS) -> dict:
    """跑一场模拟，插桩捕捉扫单事件，返回事件表 + 完整中间价序列。

    单次 ``run`` 跑到底，插桩里用 ``m.tick >= WARMUP`` 滤掉预热期的扫单
    （预热期市场还在建立盘口，事件不代表稳态）。
    不用"先 run(WARMUP) 再 run(剩余)"的写法：``SimLog`` 每次 run 结束都会
    ``trim`` 数组并可能在续跑时重新分配，事件表的 tick 与数组下标一旦错位
    就是静默的错误数据。
    """
    pop = Population.from_shares(N_AGENTS, mix)
    cfg = SimConfig(seed=seed, n_ticks=n_ticks, population=pop, **MM_KW)
    m = Market(cfg)

    events: list[dict] = []
    orig_submit = m.submit

    def wrapped(agent, order):
        mid_before = m.current_mid()
        one_sided_before = m.book.mid_price() is None
        t_now = m.tick
        pre_mom = np.nan
        if t_now >= WARMUP + 30:
            lag = m.history.lag(20)
            if lag is not None and lag > 0:
                pre_mom = (mid_before / lag - 1.0) * 1e4
        trades = orig_submit(agent, order)
        if len(trades) >= MIN_LEVELS and m.tick >= WARMUP:
            levels = {round(t.price, 10) for t in trades}
            qty = float(sum(t.quantity for t in trades))
            if len(levels) >= MIN_LEVELS and qty >= MIN_QTY:
                events.append(
                    {
                        "tick": m.tick,
                        "side": order.side,
                        "order_type": order.order_type,
                        "n_levels": len(levels),
                        "n_trades": len(trades),
                        "qty": qty,
                        "notional": float(sum(t.notional for t in trades)),
                        "p_pre": float(mid_before),
                        "p_now": float(m.current_mid()),
                        # 扫单前 20 tick 的价格动量（bp）。
                        # **这是内生性诊断**：如果扫单系统性地发生在"价格刚涨完"之后，
                        # 那么之后测到的"反转"里就混着"扫单前那段趋势的回归"，
                        # 而随机时点基准控制不了这个选择性偏差。
                        "pre_mom_bp": float(pre_mom),
                        "one_sided_before": bool(one_sided_before),
                        "one_sided_after": m.book.mid_price() is None,
                    }
                )
        return trades

    m.submit = wrapped  # 实例属性遮蔽类方法，仅本场有效
    m.run(n_ticks)

    mid = np.asarray(m.log.mid[: m.tick], dtype=float)
    ok, problems = m.health_check()
    return {
        "events": events,
        "mid": mid,
        "health_ok": ok,
        "health_problems": problems[:5],
        "n_trades_total": int(np.nansum(m.log.n_trades[: m.tick])),
        "depth_total": float(
            np.nanmean(m.log.bid_depth[: m.tick] + m.log.ask_depth[: m.tick])
        ),
    }


def dedupe(events: list[dict], min_gap: int = MAX_H) -> list[dict]:
    """事件去重：同一段窗口内只保留第一笔，避免重叠窗口把 t 值虚高。"""
    kept: list[dict] = []
    last = -(10**9)
    for e in sorted(events, key=lambda x: x["tick"]):
        if e["tick"] - last >= min_gap:
            kept.append(e)
            last = e["tick"]
    return kept


def _t(a) -> float:
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if a.size < 3:
        return float("nan")
    sd = a.std(ddof=1)
    return float(a.mean() / (sd / np.sqrt(a.size))) if sd > 0 else float("nan")


def event_series(events: list[dict], mid: np.ndarray) -> dict:
    """单场：把每个事件折成四组逐事件数值。**必须逐场调用**。

    为什么不能把多种子的事件合起来再统一取价格：事件属于各自的 run，
    拿某一场的价格序列去取另一场事件的"之后价格"就是张冠李戴。
    """
    n = mid.size
    sign = {"buy": 1.0, "sell": -1.0}
    out = {"imm_book": [], "move_tick": []}
    for h in HORIZONS:
        out[f"cont_{h}"] = []
        out[f"cum_{h}"] = []
    for e in events:
        t = e["tick"]
        if t + MAX_H >= n:
            continue
        s = sign[e["side"]]
        p_pre, p_now, p_end = e["p_pre"], e["p_now"], mid[t]
        if not (p_pre > 0 and p_now > 0 and p_end > 0):
            continue
        out["imm_book"].append(s * (p_now / p_pre - 1.0) * 1e4)
        out["move_tick"].append(s * (p_end / p_pre - 1.0) * 1e4)
        for h in HORIZONS:
            p_h = mid[t + h]
            if p_h > 0:
                out[f"cont_{h}"].append(s * (p_h / p_end - 1.0) * 1e4)
                out[f"cum_{h}"].append(s * (p_h / p_pre - 1.0) * 1e4)
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def split_by_premom(events: list[dict]) -> tuple[list[dict], list[dict]]:
    """按"扫单前 20 tick 的动量"与扫单方向是否同向，把事件切成两组。

    为什么必须切：实测扫单前动量按方向对齐后是 **−26.6bp（t=−7.5）**，
    即扫单系统性地发生在「价格刚往**反**方向走了一段」之后（追跌买、追涨卖）。
    于是扫单后的价格路径里混着两样东西：
      (a) 扫单自身的冲击；
      (b) **那段既有走势的回归**——而回归方向和扫单方向相同，会被误读成"延续"。
    随机时点基准的扫单前动量均值约为 0，**控制不了这个选择性偏差**。

    切分能给出可判断的证据：若两组（反向组 vs 同向组）的后续路径形状明显不同，
    说明 (b) 的贡献不可忽略，结论必须降级。
    """
    sign = {"buy": 1.0, "sell": -1.0}
    contra, mom = [], []
    for e in events:
        pm = e.get("pre_mom_bp", float("nan"))
        if not np.isfinite(pm):
            continue
        (mom if sign[e["side"]] * pm >= 0 else contra).append(e)
    return contra, mom


def random_baseline(mid: np.ndarray, n_events: int, rng, min_gap: int = MAX_H) -> dict:
    """随机基准：同一条价格序列上随机取 (时点, 方向)，做同样的统计。

    必需。市场有波动率聚集与微弱自相关，随便挑个时点看"之后 20 tick 往哪走"
    也可能测出非零均值。只有相对这个基准显著，结论才有内容。

    基准的"即时"项用同一 tick 内的一步变动（mid[t-1] → mid[t]）代理，
    长度与扫单窗口一致。
    """
    n = mid.size
    lo, hi = WARMUP + 100, n - MAX_H - 1
    out = {"imm_book": [], "move_tick": []}
    for h in HORIZONS:
        out[f"cont_{h}"] = []
        out[f"cum_{h}"] = []
    if hi <= lo or n_events <= 0:
        return {k: np.asarray(v, dtype=float) for k, v in out.items()}
    picks = rng.choice(np.arange(lo, hi), size=min(n_events * 4, hi - lo), replace=False)
    kept, last = [], -(10**9)
    for t in np.sort(picks):
        if t - last >= min_gap:
            kept.append(int(t))
            last = int(t)
    kept = kept[:n_events]
    signs = rng.choice([-1.0, 1.0], size=len(kept))
    for t, s in zip(kept, signs):
        p_pre, p_now, p_end = mid[t - 1], mid[t], mid[t]
        if not (p_pre > 0 and p_now > 0 and p_end > 0):
            continue
        out["imm_book"].append(s * (p_now / p_pre - 1.0) * 1e4)
        out["move_tick"].append(s * (p_end / p_pre - 1.0) * 1e4)
        for h in HORIZONS:
            p_h = mid[t + h]
            if p_h > 0:
                out[f"cont_{h}"].append(s * (p_h / p_end - 1.0) * 1e4)
                out[f"cum_{h}"].append(s * (p_h / p_pre - 1.0) * 1e4)
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def agg(a: np.ndarray) -> dict:
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0, "mean_bp": float("nan"), "t": float("nan"), "sem_bp": float("nan")}
    return {
        "n": int(a.size),
        "mean_bp": float(a.mean()),
        "t": _t(a),
        "sem_bp": float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else float("nan"),
    }


def combine(parts: list[dict]) -> dict:
    merged: dict[str, list] = {}
    for p in parts:
        for k, v in p.items():
            merged.setdefault(k, []).append(v)
    return {
        k: (np.concatenate([a for a in v if a.size]) if any(a.size for a in v) else np.asarray([]))
        for k, v in merged.items()
    }


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main() -> dict:
    banner("扫单冲击事件研究：扫单之后是延续还是反转？")
    reals = real_metrics()
    rng = np.random.default_rng(SEED + 1)

    result = {"pool": {}, "real_reference": {m.label: m.flat() for m in reals}}
    for tag, mix in (("无做市商", BASE_MIX), ("有做市商", MM_MIX)):
        print(f"\n  ▸ 池：{tag}（{N_AGENTS} 主体 × {N_TICKS} tick × {K_SEEDS} 种子）")
        ev_parts, bl_parts = [], []
        pm_parts: list[np.ndarray] = []
        con_parts, mom_parts = [], []
        raw_n = kept_n = tot_trades = 0
        osb = osa = 0
        lv_all: list[float] = []
        qty_all: list[float] = []
        for i in range(K_SEEDS):
            sd = SEED + 13 * i
            r = run_and_detect(mix, sd)
            if not r["health_ok"]:
                print(f"    ⚠️ seed {sd} 体检未过：{r['health_problems']}")
            tot_trades += r["n_trades_total"]
            raw_n += len(r["events"])
            osb += sum(1 for e in r["events"] if e["one_sided_before"])
            osa += sum(1 for e in r["events"] if e["one_sided_after"])
            lv_all.extend(e["n_levels"] for e in r["events"])
            qty_all.extend(e["qty"] for e in r["events"])
            ev = dedupe(r["events"], MAX_H)
            kept_n += len(ev)
            ev_parts.append(event_series(ev, r["mid"]))
            bl_parts.append(random_baseline(r["mid"], max(len(ev), 50), rng))
            sgn = {"buy": 1.0, "sell": -1.0}
            pm_parts.append(
                np.asarray([sgn[e["side"]] * e["pre_mom_bp"] for e in ev], dtype=float)
            )
            contra, mom = split_by_premom(ev)
            con_parts.append(event_series(contra, r["mid"]))
            mom_parts.append(event_series(mom, r["mid"]))
            print(f"      seed {sd}: 扫单 {len(r['events']):>5d} 笔 → 去重后 {len(ev):>4d} 笔"
                  f"（反向 {len(contra)} / 同向 {len(mom)}）")

        E = combine(ev_parts)
        B = combine(bl_parts)
        E_con = combine(con_parts)
        E_mom = combine(mom_parts)
        pm_all = (
            np.concatenate([a for a in pm_parts if a.size])
            if any(a.size for a in pm_parts)
            else np.asarray([])
        )
        pm_all = pm_all[np.isfinite(pm_all)]
        lv = np.asarray(lv_all, dtype=float)
        qt = np.asarray(qty_all, dtype=float)
        if E["imm_book"].size == 0:
            print("    ❌ 没有可用事件，跳过")
            continue

        ea = {k: agg(v) for k, v in E.items()}
        ba = {k: agg(v) for k, v in B.items()}
        pm_stat = agg(pm_all)
        cona = {k: agg(v) for k, v in E_con.items()}
        moma = {k: agg(v) for k, v in E_mom.items()}
        result["pool"][tag] = {
            "events": ea, "baseline": ba, "pre_momentum": pm_stat,
            "by_premom": {"反向（追跌买/追涨卖）": cona, "同向（追涨买/追跌卖）": moma},
            "n_raw": raw_n, "n_dedup": kept_n,
            "one_sided_before_frac": osb / max(raw_n, 1),
            "one_sided_after_frac": osa / max(raw_n, 1),
            "level_mean": float(lv.mean()) if lv.size else None,
            "qty_mean": float(qt.mean()) if qt.size else None,
            "trade_share": raw_n / max(tot_trades, 1),
        }

        print(
            f"    捕捉到扫单 {raw_n} 笔 / {tot_trades:,} 笔成交"
            f"（占 {raw_n / max(tot_trades, 1):.3%}）；"
            f"平均吃穿 {lv.mean():.1f} 档、{qt.mean():.2f} 手"
        )
        print(
            f"    盘口单边空缺：扫单前 {osb / max(raw_n, 1):.1%}，"
            f"砸完后 {osa / max(raw_n, 1):.1%}"
            f"（后者越高，`p_now` 这个锚越脏）"
        )
        print(f"    去重后（相邻 ≥ {MAX_H} tick）保留 {kept_n} 笔独立事件")

        print(f"\n    {'即时冲击':<16}{ea['imm_book']['mean_bp']:>10.2f}bp"
              f"   (t={ea['imm_book']['t']:>5.2f}, n={ea['imm_book']['n']})   ← 锚 p_now，瞬时倾斜")
        print(f"    {'当 tick 总冲击':<16}{ea['move_tick']['mean_bp']:>10.2f}bp"
              f"   (t={ea['move_tick']['t']:>5.2f}, n={ea['move_tick']['n']})   ← 锚 p_pre→p_end，干净")
        print(f"    {'随机基准(同口径)':<16}{ba['move_tick']['mean_bp']:>10.2f}bp"
              f"   (t={ba['move_tick']['t']:>5.2f}, n={ba['move_tick']['n']})")

        extra = ea["move_tick"]["mean_bp"] - ea["imm_book"]["mean_bp"]
        print(
            f"\n    → 同一 tick 内后续主体的跟风推动 ≈ {extra:+.2f}bp"
            f"（当 tick 总冲击 − 扫单自身冲击）"
        )
        pmt = pm_stat["t"]
        print(
            f"    → ⚠️ 内生性诊断：扫单**前** 20 tick 的价格动量（按扫单方向对齐）= "
            f"{pm_stat['mean_bp']:+.2f}bp (t={pmt:+.2f}, n={pm_stat['n']})"
        )
        if np.isfinite(pmt) and abs(pmt) >= 2.0:
            if pm_stat["mean_bp"] < 0:
                print(
                    "       **显著为负 ⇒ 扫单是反向的**：买单扫单通常发生在价格先跌一段之后"
                    "（追跌买），卖单扫单发生在先涨之后。"
                    "\n       后果：扫单后的路径里混着**那段既有走势的回归**——"
                    "回归方向与扫单同向，会被误读成「延续」。"
                    "\n       随机时点基准的扫单前动量均值约为 0，**控制不住这个选择性偏差**。"
                    "下面用「按扫单前动量分组」给出可判断的证据。"
                )
            else:
                print(
                    "       **显著为正 ⇒ 扫单是顺势的**（追涨买/追跌卖）："
                    "路径里可能含有既有趋势的延续，结论需要同样的分组核对。"
                )
        else:
            print("       不显著：扫单不是系统性的追涨杀跌，后续路径可更直接归因于扫单本身。")

        # 分组对照：反向组 vs 同向组
        con = result["pool"][tag]["by_premom"]["反向（追跌买/追涨卖）"]
        mom = result["pool"][tag]["by_premom"]["同向（追涨买/追跌卖）"]
        print(
            f"\n    分组对照（按扫单前动量与扫单方向是否同向）："
        )
        print(
            f"      {'组':<22}{'n':>6}{'当tick总冲击':>18}{'h=1':>18}{'h=5':>18}{'h=20':>18}"
        )
        for label, g in (("反向（追跌买/追涨卖）", con), ("同向（追涨买/追跌卖）", mom)):
            if g.get("move_tick", {}).get("n", 0) == 0:
                continue
            cells = []
            for key in ("move_tick", "cont_1", "cont_5", "cont_20"):
                s = g.get(key, {"mean_bp": float("nan"), "t": float("nan")})
                cells.append(f"{s['mean_bp']:+.2f}(t={s['t']:+.1f})")
            print(
                f"      {label:<22}{g['move_tick']['n']:>6}"
                + "".join(f"{c:>18}" for c in cells)
            )
        if con.get("cont_5", {}).get("n", 0) and mom.get("cont_5", {}).get("n", 0):
            d5 = con["cont_5"]["mean_bp"] - mom["cont_5"]["mean_bp"]
            print(
                f"      → 两组在 h=5 的差异 = {d5:+.2f}bp。"
                f"若两组形状明显不同，说明「既有走势的回归」贡献不可忽略，"
                f"结论只能限定为「扫单之后的价格路径形状」。"
            )

        print(
            f"\n    {'h':>4}{'延续(cont_h)':>22}{'累计(cum_h)':>22}"
            f"{'基准延续':>20}{'判定':>10}"
        )
        for h in HORIZONS:
            c = ea[f"cont_{h}"]
            cu = ea[f"cum_{h}"]
            b = ba.get(f"cont_{h}", {"mean_bp": float("nan"), "t": float("nan")})
            if not np.isfinite(c["t"]):
                verdict = "样本不足"
            elif abs(c["t"]) < 2.0:
                verdict = "不显著"
            elif c["mean_bp"] > 0:
                verdict = "延续"
            else:
                verdict = "反转"
            print(
                f"    {h:>4d}{c['mean_bp']:>12.2f}bp(t={c['t']:>5.2f})"
                f"{cu['mean_bp']:>12.2f}bp(t={cu['t']:>5.2f})"
                f"{b['mean_bp']:>12.2f}bp(t={b['t']:>5.2f})"
                f"{verdict:>10}"
            )

        h1 = ea["cont_1"]
        h20 = ea["cont_20"]
        c20 = ea["cum_20"]
        print(
            f"\n    → 判据①扫单确实推动价格：当 tick 总冲击 {ea['move_tick']['mean_bp']:+.1f}bp "
            f"(t={ea['move_tick']['t']:.1f})，基准同口径 "
            f"{ba['move_tick']['mean_bp']:+.1f}bp (t={ba['move_tick']['t']:.1f})"
        )
        print(
            f"    → 判据②下一 tick 起是否延续：h=1 {h1['mean_bp']:+.1f}bp (t={h1['t']:.2f})，"
            f"h=20 {h20['mean_bp']:+.1f}bp (t={h20['t']:.2f})"
        )
        print(
            f"    → 判据③20 tick 后相对扫单前：累计 {c20['mean_bp']:+.1f}bp (t={c20['t']:.2f})"
            f"（|累计| < 当 tick 总冲击则说明已被大部分抹平）"
        )

    out = {
        "stage": "sweep_event_study",
        "protocol": {
            "n_agents": N_AGENTS, "warmup": WARMUP, "n_ticks": N_TICKS,
            "min_levels": MIN_LEVELS, "min_qty": MIN_QTY,
            "horizons": HORIZONS, "seeds": [SEED + 13 * i for i in range(K_SEEDS)],
            "design": "逐场计算后汇总；事件去重（相邻≥60 tick）；随机时点基准；双锚对照",
            "primary": "move_tick = s·(mid[t]/p_pre − 1)，两个锚都由真实盘口算出",
        },
        **result,
    }
    save_json(out, "sweep_study.json")
    _plot(result, FIG / "sweep_event_study.png")
    print(f"\n  图已输出到 {FIG}")
    return out


def _plot(result: dict, path) -> None:
    import matplotlib.pyplot as plt

    from tw.viz import C_ACCENT, C_DIM, C_GOOD, C_SIM, MUTED

    pools = list(result["pool"].keys())
    fig, axes = plt.subplots(1, len(pools), figsize=(6.6 * len(pools), 4.6), squeeze=False)
    for ax, tag in zip(axes[0], pools):
        ea = result["pool"][tag]["events"]
        ba = result["pool"][tag]["baseline"]
        hs = HORIZONS
        x = np.arange(len(hs))
        m_c = [ea[f"cont_{h}"]["mean_bp"] for h in hs]
        s_c = [1.96 * ea[f"cont_{h}"]["sem_bp"] for h in hs]
        m_u = [ea[f"cum_{h}"]["mean_bp"] for h in hs]
        s_u = [1.96 * ea[f"cum_{h}"]["sem_bp"] for h in hs]
        m_b = [ba[f"cont_{h}"]["mean_bp"] for h in hs]
        s_b = [1.96 * ba[f"cont_{h}"]["sem_bp"] for h in hs]
        ax.errorbar(x, m_c, yerr=s_c, fmt="o-", color=C_SIM, lw=1.6, ms=6, capsize=3,
                    label="从下一 tick 起的净移动 cont_h（>0=延续）")
        ax.errorbar(x, m_u, yerr=s_u, fmt="^-", color=C_GOOD, lw=1.5, ms=5, capsize=3,
                    label="相对扫单前的累计 cum_h")
        ax.errorbar(x, m_b, yerr=s_b, fmt="s--", color=C_DIM, lw=1.3, ms=5, capsize=3,
                    label="随机基准（同口径）")
        ax.axhline(0, color=MUTED, lw=1.0, ls="--")
        tk = ea["move_tick"]["mean_bp"]
        ax.axhline(tk, color=C_ACCENT, lw=1.0, ls=":",
                   label=f"当 tick 总冲击 {tk:+.1f}bp（扫单自身 + 同 tick 跟风）")
        ax.set_xticks(x)
        ax.set_xticklabels([str(h) for h in hs])
        ax.set_xlabel("扫单之后经过的 tick 数")
        ax.set_ylabel("按扫单方向对齐的价格移动 (bp)")
        ax.set_title(
            f"{tag}　扫单 {result['pool'][tag]['n_dedup']} 个独立事件\n"
            f"扫单自身冲击 {ea['imm_book']['mean_bp']:+.1f}bp，"
            f"当 tick 总冲击 {tk:+.1f}bp",
            fontsize=10,
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.15)
    fig.suptitle("扫单之后：延续还是反转？（锚全部取自真实盘口）", fontsize=12)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
