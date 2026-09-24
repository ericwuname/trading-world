#!/usr/bin/env python
"""把两份文档的策略接进 trading-world 系统**实测** → out/a16/doc_strategies.json

    python scripts/a16_doc_strategies_system.py

⭐ 与 A14/A15 的独立回测不同：这里走系统**同一条管线**——
   TradingAgent(policy=…) ⇒ 同一套风控/订单/留痕 ⇒ BarExecutor ⇒
   MarginAccount（1x）⇒ ExecConfig 成本（maker 2bp / taker 5bp+滑点 1bp）。
   同一数据（BTC/ETH/SOL 1H × 2 年连续会话）、同一口径，对比两个"形状"。

验证文档 §9 的核心声明：**左尾右尾期望都是 0，换的只是形状**；
§8：**成本是唯一确定性杀手**。
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.append(str(ROOT / "scripts"))

from tw.agent import AgentConfig, TradingAgent  # noqa: E402
from tw.agent_run import run_agent_session  # noqa: E402
from tw.marketdb import MarketStore  # noqa: E402
from tw.policies_doc import GridPolicyDoc, RightTailPolicyDoc  # noqa: E402
from tw.simexec import ExecConfig  # noqa: E402

OUTD = ROOT / "out" / "a16"
INSTRUMENTS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
INITIAL_CASH = 100_000.0
YEARS = 2.0


def load(inst: str, limit: int = 17520) -> Any:
    st = MarketStore()
    try:
        return st.load_candles("binance_csv", inst, "1H", limit=limit)
    finally:
        st.close()


def max_drawdown(curve: list[float]) -> float:
    peak, dd = float("-inf"), 0.0
    for x in curve:
        peak = max(peak, x)
        if peak > 0:
            dd = max(dd, 1.0 - x / peak)
    return dd


def trade_stats(fills: list[Any]) -> dict[str, Any]:
    """按净头寸配对，算每笔已实现盈亏（含手续费；做空为负头寸）。"""
    pos = 0.0
    entry = 0.0
    trades: list[float] = []
    fees = 0.0
    slip = 0.0
    for f in fills:
        q = float(f.qty) * (1.0 if f.side == "buy" else -1.0)
        fees += float(f.fee)
        slip += float(getattr(f, "slippage_cost", 0.0) or 0.0)
        if pos == 0.0 or (q > 0) == (pos > 0):          # 加仓（同向）
            tot = abs(pos) + abs(q)
            entry = (entry * abs(pos) + f.price * abs(q)) / tot if tot > 0 else 0.0
            pos += q
        else:                                            # 反向 ⇒ 平仓（可能翻仓）
            closed = min(abs(q), abs(pos))
            if closed > 1e-12:
                pnl = closed * (f.price - entry) * (1.0 if pos > 0 else -1.0)
                trades.append(pnl - f.fee)
            pos += q
            entry = f.price if abs(pos) > 1e-12 and pos * q < 0 else entry
    total = sum(trades)
    n = len(trades)
    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    skew = float("nan")
    if n >= 3:
        mu = total / n
        m2 = sum((t - mu) ** 2 for t in trades) / n
        m3 = sum((t - mu) ** 3 for t in trades) / n
        skew = m3 / (m2 ** 1.5) if m2 > 0 else float("nan")
    return {
        "n_trades": n, "total_realized": total,
        "E_per_trade": total / n if n else float("nan"),
        "win_rate": len(wins) / n if n else float("nan"),
        "avg_win": (sum(wins) / len(wins)) if wins else float("nan"),
        "avg_loss": (sum(losses) / len(losses)) if losses else float("nan"),
        "payoff": ((sum(wins) / len(wins)) / abs(sum(losses) / len(losses)))
                  if wins and losses else float("nan"),
        "skew": skew, "fees": fees, "slippage": slip,
        "net_after_costs": total - fees - slip,
    }


def run_one(inst: str, series: Any, policy: Any, name: str,
            exec_config: ExecConfig) -> dict[str, Any]:
    agent = TradingAgent(
        policy=policy,
        config=AgentConfig(inst_id=inst, n_closes=64, agent_id=name),
    )
    from tw.account import MarginAccount, MarginConfig  # 局部导入防环
    account = MarginAccount(cash=INITIAL_CASH, cfg=MarginConfig())
    res = run_agent_session(
        agent, series, account=account, start=0, end=len(series.close) - 1,
        name=name, run_id=f"{inst}-{name}",
        exec_config=exec_config, lever=1.0,
    )
    final = float(res.equity_curve[-1])
    ann = (final / INITIAL_CASH) ** (1.0 / YEARS) - 1.0
    bh = (float(series.close[-1]) / float(series.close[0])) * (1 - 0.0006)
    bh_ann = bh ** (1.0 / YEARS) - 1.0
    ts = trade_stats(res.fills)
    return {
        "instrument": inst, "strategy": name,
        "final_equity": final, "total_return": final / INITIAL_CASH - 1.0,
        "annualized": ann, "max_drawdown": max_drawdown(res.equity_curve),
        "bh_annualized": bh_ann, "beats_bh": ann > bh_ann,
        "n_decisions": len(res.records), "n_fills": len(res.fills),
        "n_errors": sum(1 for rec in res.records
                        if not getattr(rec, "ok", True)),
        **ts,
    }


def main() -> int:
    OUTD = ROOT / "out" / "a16"
    OUTD.mkdir(parents=True, exist_ok=True)
    exec_config = ExecConfig()
    results: list[dict[str, Any]] = []
    for inst in INSTRUMENTS:
        series = load(inst)
        print(f"[{inst}] {len(series.close)} 根", flush=True)
        for name, mk in (
            ("grid_doc", lambda: GridPolicyDoc()),
            ("righttail_doc", lambda: RightTailPolicyDoc()),
        ):
            r = run_one(inst, series, mk(), name, exec_config)
            results.append(r)
            print(f"  {name:<14} 年化 {r['annualized']:+7.1%}  "
                  f"回撤 {r['max_drawdown']:5.1%}  交易 {r['n_trades']:>4}  "
                  f"胜率 {r['win_rate']:.0%}  偏度 {r['skew']:+.2f}  "
                  f"E/笔 {r['E_per_trade']:+.1f} 元  B&H {r['bh_annualized']:+.1%}",
                  flush=True)
    (OUTD / "doc_strategies.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1, default=float),
        encoding="utf-8")
    print(f"\n{len(results)} 条 → {OUTD / 'doc_strategies.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
