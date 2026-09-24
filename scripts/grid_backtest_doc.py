#!/usr/bin/env python
"""验证《个人交易决策系统方案.docx》——按其 §4 规格实现现货网格，在真实数据上回测。

    python scripts/grid_backtest_doc.py        # → out/a14/grid_results.json
    python scripts/make_a14_doc_verify.py      # → docs/A14-个人交易决策系统方案-验证报告.md

验证的声明：
  §4.1 等比格距公式（另见 ① 纯算术核对）
  §4.2 主规格：k=±25~30%、N=10/20、半仓、每168根更新中心、21天无成交平仓、1x 现货
  §3 禁区：21天无成交离场「回撤 0.563→0.167」  → 开/关对照
  §3 禁区：破网后移仓重启「年化 0.025 vs 0.09」 → 重启/不重启对照
  §7 预期：年化分位、最大回撤中位 20.3%、只有 38% 跑赢买入持有 → 多配置分布
⚠️ 诚实边界：文档的分布来自「13 个品种」，我们只有 3 个标的的 2 年 1H
   （BTC/ETH/SOL）⇒ 用 3 标的 × 3 个 k × 2 个 N = 18 个配置凑分布，
   **总体不同**，只能对量级、不能对分位数。

成本（项目 ExecConfig）：网格挂单是**限价单** ⇒ maker 2bp、无滑点；
市价单（初始半仓、21天平仓）⇒ taker 5bp + 滑点 1bp = 6bp/边。
"""

from __future__ import annotations

import dataclasses
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tw.marketdb import MarketStore  # noqa: E402
from tw.simexec import ExecConfig  # noqa: E402

OUTD = ROOT / "out" / "a14"
INSTRUMENTS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
KS = (0.10, 0.25, 0.30)
NS = (10, 20)

CFG = ExecConfig()
MAKER = CFG.eff_maker                                  # 0.0002
TAKER = CFG.eff_taker + CFG.eff_slippage / 10000.0     # 0.0006
EXIT_NO_FILL_BARS = 21 * 24                            # 21 天（1H bar）
RECENTER_EVERY = 168                                   # 文档 §4.2：每 168 根


def grid_levels(center: float, k: float, n: int) -> list[float]:
    """等比格位（文档 §4.1）：i=0..n，共 n+1 个。"""
    g = ((1 + k) / (1 - k)) ** (1.0 / n) - 1
    lo = center * (1 - k)
    return [lo * ((1 + k) / (1 - k)) ** (i / n) for i in range(n + 1)]


@dataclasses.dataclass
class GridResult:
    instrument: str
    k: float
    n: int
    exit_rule: bool
    restart_on_break: bool
    final_equity: float
    total_return: float
    annualized: float
    max_drawdown: float
    bh_total: float
    bh_annualized: float
    beats_bh: bool
    n_fills: int
    n_recenters: int
    n_break_restarts: int
    n_exit_closes: int
    bars_no_fill_max: int
    years: float
    n_bars: int


def run_grid(series: Any, *, k: float, n: int, exit_rule: bool,
             restart_on_break: bool, initial_cash: float = 100_000.0,
             recenter_every: int = RECENTER_EVERY) -> GridResult:
    """按文档 §4 规格跑现货网格（**买区/卖区模型**——这就是「初始半仓」的含义）。

    梯位 i=0..n（等比）。**下半格位（低于中心）是买区，上半格位是卖区**：
      - 初始半仓 = 持有上半各格位的存货（在 P0 估值）⇒ 价格上穿上半格位时卖出；
      - 价格下穿**下半**格位 ⇒ 买入一格（存货栈：弹出买入价最低的来配对卖出）；
      - 每对（低买 L，高卖 L'）赚一个格差，扣除双边 maker 2bp。
    ⚠️ 我先后写错过三版（成本价之上才卖 / 上穿上就卖 / 顶格也当买点）——
    全部被「正弦震荡必须赚钱」这条健全性测试抓出来。这条测试就是这道防线的价值。
    """
    close = [float(x) for x in series.close]
    high = [float(x) for x in series.high]
    low = [float(x) for x in series.low]
    nb = len(close)
    years = nb / (24.0 * 365.0)

    m = initial_cash / n                        # 每格名义额
    center0 = close[0]
    levels = grid_levels(center0, k, n)         # levels[0]=最低 .. levels[n]=最高
    half = n // 2                               # 中心格位下标

    cash = initial_cash
    inv: list[tuple[float, float]] = []         # 存货栈 (买入价, 币数量)
    # 初始半仓：持有上半各格位的存货，按 P0 估值买入
    for j in range(half, n):
        q = (m * (1 - MAKER)) / close[0]
        inv.append((close[0], q))
    cash -= sum(q * close[0] for _lv, q in inv)
    inv.sort(key=lambda t: t[0])                # 买入价升序

    peak = cash + sum(q * close[0] for _lv, q in inv)
    max_dd = 0.0
    fills = recenters = break_restarts = exit_closes = 0
    since_fill = 0
    no_fill_max = 0

    def equity(price: float) -> float:
        return cash + sum(q * price for _lv, q in inv)

    for i in range(1, nb):
        pc = close[i - 1]
        lo_b, hi_b, c = low[i], high[i], close[i]
        down = c < pc
        filled = False

        # ① 卖出：价格**上穿上半格位** ⇒ 卖出一格（弹出买入价最低的存货）
        for j in range(half, n):
            L = levels[j]
            if min(pc, lo_b) < L <= max(pc, hi_b) and inv:
                # 只配对买入价低于卖价的存货（否则是亏本卖出）
                cand = [t for t in range(len(inv)) if inv[t][0] < L]
                if cand:
                    t = min(cand, key=lambda x: inv[x][0])
                    lv, q = inv.pop(t)
                    cash += q * L * (1 - MAKER)
                    fills += 1
                    filled = True

        # ② 买入：价格**下穿下半格位** ⇒ 买一格（限价挂在格位上，maker）
        def _walk_down(a: float, b: float) -> list[float]:
            if b >= a:
                return []
            ls = [L for L in levels[:half] if b <= L < a]
            return list(reversed(ls))

        if down:
            buys = _walk_down(pc, lo_b) + _walk_down(hi_b, c)
        else:
            buys = _walk_down(hi_b, lo_b)
        seen: set[float] = set()
        for L in buys:
            if L in seen:
                continue
            seen.add(L)
            if cash >= m and sum(q for _lv, q in inv) / L < m * n / L:
                cash -= m
                inv.append((L, m * (1 - MAKER) / L))
                fills += 1
                filled = True

        eq = equity(c)
        peak = max(peak, eq)
        if peak > 0:
            max_dd = max(max_dd, 1.0 - eq / peak)

        if filled:
            since_fill = 0
        else:
            since_fill += 1
            no_fill_max = max(no_fill_max, since_fill)
            if exit_rule and since_fill >= EXIT_NO_FILL_BARS and inv:
                cash += sum(q for _lv, q in inv) * c * (1 - TAKER)
                inv.clear()
                exit_closes += 1
                since_fill = 0

        if recenter_every and i % recenter_every == 0:
            center0 = c
            levels = grid_levels(center0, k, n)
            half = n // 2
            recenters += 1

        if restart_on_break and (c < levels[0] or c > levels[-1]):
            center0 = c
            levels = grid_levels(center0, k, n)
            half = n // 2
            break_restarts += 1

    final = equity(close[-1])
    total = final / initial_cash - 1.0
    ann = (final / initial_cash) ** (1.0 / years) - 1.0 if years > 0 else float("nan")
    bh = (close[-1] / close[0]) * (1 - TAKER)
    bh_ann = bh ** (1.0 / years) - 1.0 if years > 0 else float("nan")
    return GridResult(
        instrument="", k=k, n=n, exit_rule=exit_rule,
        restart_on_break=restart_on_break, final_equity=final,
        total_return=total, annualized=ann, max_drawdown=max_dd,
        bh_total=bh - 1.0, bh_annualized=bh_ann, beats_bh=final > bh * initial_cash,
        n_fills=fills, n_recenters=recenters, n_break_restarts=break_restarts,
        n_exit_closes=exit_closes, bars_no_fill_max=no_fill_max,
        years=years, n_bars=nb,
    )


def load(inst: str, limit: int = 17520) -> Any:
    st = MarketStore()
    try:
        return st.load_candles("binance_csv", inst, "1H", limit=limit)
    finally:
        st.close()


def main() -> int:
    OUTD.mkdir(parents=True, exist_ok=True)
    #: 四种变体：文档规格（照字面）/ 修正版（不重设）/ 无离场 / 破网重启
    VARIANTS = {
        "doc_spec":   dict(recenter_every=168, exit_rule=True,  restart_on_break=False),
        "no_recenter": dict(recenter_every=0,   exit_rule=True,  restart_on_break=False),
        "no_exit":     dict(recenter_every=0,   exit_rule=False, restart_on_break=False),
        "restart":     dict(recenter_every=0,   exit_rule=True,  restart_on_break=True),
    }
    results: list[dict[str, Any]] = []
    for inst in INSTRUMENTS:
        series = load(inst)
        print(f"[{inst}] {len(series)} 根", flush=True)
        for k in KS:
            for n in NS:
                for vname, kw in VARIANTS.items():
                    r = run_grid(series, k=k, n=n, **kw)
                    results.append(dataclasses.asdict(r) | {
                        "instrument": inst, "variant": vname})
        best = max((x for x in results if x["instrument"] == inst
                    and x["variant"] == "no_recenter"),
                   key=lambda x: x["annualized"])
        print(f"   no_recenter 最优: k={best['k']} N={n} 年化 "
              f"{best['annualized']:+.1%} 回撤 {best['max_drawdown']:.1%}",
              flush=True)
    (OUTD / "grid_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{len(results)} 条结果 → {OUTD / 'grid_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
