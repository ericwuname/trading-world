"""A4 分层评估入口 —— **LLM Agent vs 规则基线 vs noop**。

用法
----
    # 1) 只看规则基线（离线、秒级、不需要 key）
    python scripts/agent_eval.py --inst BTC-USDT-SWAP --n 200 --rules-only

    # 2) 真机跑 LLM 并录制（需要 key），同时跑全部基线
    python scripts/agent_eval.py --inst BTC-USDT-SWAP --n 80 --samples 3 \\
        --template v2 --record out/a4/rec_BTC.jsonl --out-json out/a4/eval_BTC.json

    # 3) 用录制的决策做成本敏感性（回放，零 API 成本）
    #    —— 由脚本自动完成（见 --cost-multipliers）

为什么成本敏感性要"回放"而不是"重跑"
--------------------------------------
成本变了，策略**可能就会改变行为**（某单变得不划算、甚至被强平）。
所以在旧收益上减一个数只是**一阶近似**。正确做法是**用同一批
已录制的决策重跑一遍**——而 A3 的录制/回放机制让这件事
**零额外 API 成本**。两个版本的差值本身就是信息：
差得越多，说明该策略对成本越敏感。

⚠️ 本脚本**不下载行情**（决策路径只读库）。要新数据先跑
``data_cli.py fetch``。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import banner  # noqa: E402

from tw.account import MarginAccount, MarginConfig  # noqa: E402
from tw.agent import AgentConfig, TradingAgent  # noqa: E402
from tw.agent_run import run_agent_session  # noqa: E402
from tw.eval_agent import (  # noqa: E402
    compare_to_baselines,
    cost_sensitivity_analytic,
    cost_sensitivity_rerun,
    evaluate_run,
    format_report_lines,
    json_safe,
)
from tw.llm import HTTPClient, LLMConfig, Recorder, ReplayClient  # noqa: E402
from tw.marketdb import SOURCE_OKX, MarketStore  # noqa: E402
from tw.policy import make_policy  # noqa: E402
from tw.risk import RiskLimits  # noqa: E402
from tw.simexec import ExecConfig  # noqa: E402

#: 默认跑的规则基线（含 noop —— 它必须是第一对照）
RULE_BASELINES = ("noop", "momentum", "meanrevert", "random_taker")


def _load(store, inst: str, bar: str, n: int):
    s = store.load_candles(SOURCE_OKX, inst, bar, limit=n)
    if s.is_empty:
        raise SystemExit(
            f"库里没有 {inst} {bar}。先跑："
            f"python scripts/data_cli.py fetch --inst {inst} --bar {bar} --days 30"
        )
    return s


def _run_one(series, *, name, run_id, inst, account_cash, exec_cfg, lever,
             agent: TradingAgent, start: int, end: int, log=None):
    acc = MarginAccount(cash=float(account_cash), cfg=MarginConfig())
    return run_agent_session(
        agent, series, account=acc, start=start, end=end, name=name,
        run_id=run_id, exec_config=exec_cfg, lever=lever,
        data_snapshot=f"okx:{inst}", record_log=log,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="A4 分层评估")
    ap.add_argument("--inst", default="BTC-USDT-SWAP")
    ap.add_argument("--bar", default="1H")
    ap.add_argument("--n", type=int, default=80, help="跑多少根 K 线（每根决策一次）")
    ap.add_argument("--equity", type=float, default=100_000.0)
    ap.add_argument("--lever", type=float, default=3.0)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--template", default="v2")
    ap.add_argument("--provider", default="agnes")
    ap.add_argument("--model", default=None)
    ap.add_argument("--slippage-bps", type=float, default=1.0)
    ap.add_argument("--cost-multipliers", default="1,2,5")
    ap.add_argument("--rules-only", action="store_true",
                    help="只跑规则基线（离线、秒级）")
    ap.add_argument("--replay", default="",
                    help="用这个录制文件代替真机（离线）")
    ap.add_argument("--record", default="", help="真机调用录到这里")
    ap.add_argument("--out-json", default="", help="评估结果 JSON 落这里")
    ap.add_argument("--out-log", default="", help="决策留痕 JSONL 落这里")
    ap.add_argument("--seed", type=int, default=12345, help="自助法种子")
    args = ap.parse_args()

    banner("A4 分层评估：LLM 交易 Agent")
    store = MarketStore()
    try:
        series = _load(store, args.inst, args.bar, args.n + 5)
    finally:
        store.close()
    n_bars = len(series)
    start = max(0, n_bars - args.n)
    end = n_bars - 1
    print(f"  行情：{args.inst} {args.bar}  用第 {start}~{end} 根（共 {n_bars} 根已确认）")
    print(f"  初始权益 {args.equity:,.0f}   杠杆 {args.lever}x   "
          f"滑点 {args.slippage_bps}bp   成本倍数 {args.cost_multipliers}")
    print()

    mults = tuple(float(x) for x in args.cost_multipliers.split(",") if x.strip())
    base_exec = ExecConfig(slippage_bps=args.slippage_bps)
    closes = [float(x) for x in series.close]

    runs: dict[str, object] = {}
    record_log = None
    if args.out_log:
        from tw.decision_log import DecisionLog
        record_log = DecisionLog(args.out_log).open("a")

    try:
        # ---- 1. 规则基线（秒级、离线）---------------------------------
        for bn in RULE_BASELINES:
            pol = make_policy(bn, seed=7)
            ag = TradingAgent(policy=pol,
                              config=AgentConfig(inst_id=args.inst, bar=args.bar,
                                                 lever=args.lever),
                              limits=RiskLimits())
            runs[bn] = _run_one(series, name=bn, run_id=f"a4-{bn}",
                                inst=args.inst, account_cash=args.equity,
                                exec_cfg=base_exec, lever=args.lever, agent=ag,
                                start=start, end=end)
            print(f"  ✅ 基线 {bn:<14} 成交 {len(runs[bn].fills):>3}  "
                  f"净收益 {runs[bn].final_equity - runs[bn].initial_equity:>+12,.2f}")

        # ---- 2. LLM（真机 / 回放）-------------------------------------
        llm_name = f"llm_{args.template}"
        if not args.rules_only:
            cfg = LLMConfig(provider=args.provider,
                            temperature=args.temperature)
            if args.model:
                cfg.model = args.model
            if args.replay:
                client = ReplayClient(records_path=Path(args.replay), config=cfg,
                                      strict=False)
                print(f"\n  LLM：回放 {args.replay}")
            else:
                client = HTTPClient(config=cfg)
                if args.record:
                    client = Recorder(client, args.record)
                print(f"\n  LLM：真机 {cfg.provider}/{cfg.model}"
                      f"  {args.samples} 采样/决策")
            agent = TradingAgent(
                client=client, limits=RiskLimits(),
                config=AgentConfig(inst_id=args.inst, bar=args.bar,
                                   template=args.template, lever=args.lever,
                                   n_samples=args.samples,
                                   temperature=args.temperature),
            )
            t0 = time.time()
            runs[llm_name] = _run_one(
                series, name=llm_name, run_id=f"a4-{llm_name}", inst=args.inst,
                account_cash=args.equity, exec_cfg=base_exec,
                lever=args.lever, agent=agent, start=start, end=end,
                log=record_log,
            )
            rr = runs[llm_name]
            print(f"  ✅ {llm_name:<14} 成交 {len(rr.fills):>3}  "
                  f"净收益 {rr.final_equity - rr.initial_equity:>+12,.2f}"
                  f"   用时 {time.time() - t0:.0f}s")

        # ---- 3. 成本敏感性（**重跑**，LLM 走回放）---------------------
        cost_runs: dict[str, dict[float, object]] = {}
        src = args.replay or args.record
        for nm, r0 in list(runs.items()):
            per: dict[float, object] = {}
            for m in mults:
                ec = ExecConfig(slippage_bps=args.slippage_bps,
                                cost_multiplier=m)
                if nm.startswith("llm_"):
                    if not src:
                        continue     # 没录制就没法重跑，只能给一阶近似
                    cfg = LLMConfig(provider=args.provider,
                                    temperature=args.temperature)
                    if args.model:
                        cfg.model = args.model
                    ag = TradingAgent(
                        client=ReplayClient(records_path=Path(src), config=cfg,
                                            strict=False),
                        limits=RiskLimits(),
                        config=AgentConfig(inst_id=args.inst, bar=args.bar,
                                           template=args.template,
                                           lever=args.lever,
                                           n_samples=args.samples,
                                           temperature=args.temperature),
                    )
                    used_log = None
                else:
                    ag = TradingAgent(
                        policy=make_policy(nm, seed=7),
                        config=AgentConfig(inst_id=args.inst, bar=args.bar,
                                           lever=args.lever),
                        limits=RiskLimits(),
                    )
                    used_log = None
                per[m] = _run_one(series, name=nm, run_id=f"a4-{nm}-c{m:g}",
                                  inst=args.inst, account_cash=args.equity,
                                  exec_cfg=ec, lever=args.lever, agent=ag,
                                  start=start, end=end, log=used_log)
            cost_runs[nm] = per
        print(f"\n  ✅ 成本敏感性重跑完成（倍数 {mults}）")
    finally:
        if record_log is not None:
            record_log.close()

    # ---- 4. 评估 -------------------------------------------------------
    evals = {nm: evaluate_run(r, closes=closes) for nm, r in runs.items()}
    for nm in evals:
        if nm in cost_runs and cost_runs[nm]:
            evals[nm]["cost_sensitivity_rerun"] = cost_sensitivity_rerun(
                cost_runs[nm])
    if evals.get(llm_name) and not cost_runs.get(llm_name):
        evals[llm_name]["cost_sensitivity_note"] = (
            "没有录制文件 ⇒ 无法重跑，只有一阶近似（见 "
            "cost_sensitivity_analytic）；重跑不可用的原因：未 --record/--replay"
        )

    target = llm_name if llm_name in runs else "momentum"
    comparison = compare_to_baselines(
        runs[target], {k: v for k, v in runs.items() if k != target},
        seed=args.seed,
    )
    comparison["target"] = target

    print()
    for ln in format_report_lines(evals, comparison):
        print(ln)

    # 重跑版成本敏感性（这才是正式结论）
    any_rerun = False
    lines = ["", "成本敏感性（**重跑版**：用同一批决策换成本重跑）"]
    for nm, e in evals.items():
        cs = e.get("cost_sensitivity_rerun")
        if not cs:
            continue
        any_rerun = True
        lines.append(f"  {nm:<16}" + "  ".join(
            f"×{c['cost_multiplier']:g}: {c['net_return'] * 100:+.3f}%"
            f"（{c['n_trades']} 笔）" for c in cs))
    if any_rerun:
        print("\n".join(lines))

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        # ⚠️ **必须 sanitize + allow_nan=False**：Python 的 json 默认会写出
        # 裸的 `NaN`，而它不是合法 JSON（JS/Go/Rust 都读不了），
        # 而 Python 自己读得回来 ⇒ **自测全绿、产物却是坏的**。
        # 详见 ``tw.eval_agent.json_safe`` 的文档。
        Path(args.out_json).write_text(json.dumps(
            json_safe({"config": vars(args), "evals": evals,
                       "comparison": comparison, "target": target}),
            ensure_ascii=False, indent=2, default=str,
            allow_nan=False), encoding="utf-8")
        print(f"\n  JSON → {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
