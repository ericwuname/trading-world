"""LLM 交易 Agent 的会话入口（A3）—— 在**真实行情**上跑一次决策会话。

用法
----
    # 1) 只看提示词（不联网、不需要 key）—— 先确认"模型会看到什么"
    python scripts/agent_cli.py --show-prompt

    # 2) 离线回放（不需要 key；用已录制的响应重跑）
    python scripts/agent_cli.py --n 5 --replay out/llm_records.jsonl

    # 3) 真机跑并录制（需要 key）
    export AGNES_KEY=...
    python scripts/agent_cli.py --n 20 --record out/llm_records.jsonl \\
        --out out/decisions.jsonl --samples 3

为什么默认**不联网**
--------------------
本脚本的默认模式是"只打印提示词"。理由与本项目其余部分一致：
**先能看见，再能回放，最后才真跑。** 一个上来就联网的脚本会在
"key 没配好 / 网络不通 / 模型名写错"这三件事上各花掉一次调试，
而它们全都可以在离线阶段排除掉。

⚠️ 三条纪律（与 ``tw/agent.py`` 的文档一致，这里再强调一次）
-------------------------------------------------------------
1. **决策路径只读库**。行情从 ``market.sqlite`` 读；联网只发生在
   数据准备阶段（``data_cli.py fetch``）。本脚本**不下载任何行情**——
   要新数据请先跑 ``data_cli.py fetch``。
2. **可见状态只取 tick ≤ t**。见 ``visible_from_series``。
3. **录制的文件里没有密钥**（只有 prompt/响应/配置描述），
   所以它可以安全地提交进仓库、发给别人复核。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import banner  # noqa: E402

from tw.agent import (  # noqa: E402
    AgentConfig,
    TradingAgent,
    run_session,
    visible_from_series,
)
from tw.decision_log import log_stats  # noqa: E402
from tw.llm import (  # noqa: E402
    HTTPClient,
    LLMConfig,
    Recorder,
    ReplayClient,
)
from tw.marketdb import SOURCE_OKX, MarketStore  # noqa: E402
from tw.prompts import build_messages  # noqa: E402
from tw.risk import RiskLimits  # noqa: E402


def _load(store: MarketStore, inst: str, bar: str, n: int):
    s = store.load_candles(SOURCE_OKX, inst, bar, limit=n)
    if s.is_empty:      # ⚠️ is_empty 是 @property，不是方法
        raise SystemExit(
            f"库里没有 {inst} {bar} 的数据。\n"
            f"  先跑：python scripts/data_cli.py fetch --inst {inst} --bar {bar} --days 30"
        )
    return s


def main() -> int:
    ap = argparse.ArgumentParser(description="LLM 交易 Agent 会话（A3）")
    ap.add_argument("--inst", default="BTC-USDT-SWAP")
    ap.add_argument("--bar", default="1H")
    ap.add_argument("--n", type=int, default=1, help="跑多少次决策（取最近的 N 根）")
    ap.add_argument("--samples", type=int, default=1, help="每个决策采样几次")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--template", default=None)
    ap.add_argument("--equity", type=float, default=100_000.0)
    ap.add_argument("--lever", type=float, default=3.0)
    ap.add_argument("--provider", default="agnes")
    ap.add_argument("--model", default=None)
    ap.add_argument("--show-prompt", action="store_true",
                    help="只打印模型会看到的提示词（不联网、不需要 key）")
    ap.add_argument("--record", default="", help="把真实调用录到这个 JSONL")
    ap.add_argument("--replay", default="", help="从这个 JSONL 回放（不联网）")
    ap.add_argument("--out", default="", help="决策留痕写到这个 JSONL")
    ap.add_argument("--run-id", default="", help=(
        "会话 ID。⚠️ 默认按**参数派生**（不含墙钟时间）——"
        "见下面 _derive_run_id 的注释：带时间戳的 run_id 会让"
        "'回放能否核对上 decision_id' 永远失败。"))
    ap.add_argument("--db", default="", help="行情库路径（默认 data/market.sqlite）")
    args = ap.parse_args()

    banner("LLM 交易 Agent 会话（A3）")

    # ---- 读行情（只读库，不联网）--------------------------------------
    store = MarketStore(args.db) if args.db else MarketStore()
    try:
        series = _load(store, args.inst, args.bar, max(args.n + 40, 120))
    finally:
        store.close()
    n_bars = len(series)
    print(f"  行情：{args.inst} {args.bar}  读入 {n_bars} 根（已确认）")
    print(f"  区间：{_ts(series.timestamp[0])} → {_ts(series.timestamp[-1])}")

    cfg_kw = {}
    if args.template:
        cfg_kw["template"] = args.template
    agent_cfg = AgentConfig(
        inst_id=args.inst, bar=args.bar,
        n_samples=args.samples, temperature=args.temperature,
        lever=args.lever, **cfg_kw,
    )
    limits = RiskLimits()

    # ---- 模式一：只看提示词 -------------------------------------------
    if args.show_prompt:
        vis = visible_from_series(series, n_bars - 1, n_closes=agent_cfg.n_closes,
                                  equity=args.equity, cash=args.equity)
        msgs = build_messages(vis, inst_id=args.inst, bar=args.bar,
                              tick=n_bars - 1, template=agent_cfg.template,
                              limits=limits, max_size=args.equity * 0.3 / vis["mid"])
        print("\n" + "=" * 72)
        print("模型会看到的内容（这就是全部，没有别的）")
        print("=" * 72 + "\n")
        for m in msgs:
            print(f"—— {m['role']} ——")
            print(m["content"])
            print()
        print("=" * 72)
        print("提示词哈希（同输入同哈希，回放靠它定位）：")
        from tw.llm import prompt_hash
        print("  " + prompt_hash(msgs, model=args.model or LLMConfig(
            provider=args.provider).model,
            temperature=args.temperature, max_tokens=agent_cfg.max_tokens))
        return 0

    # ---- 选客户端 -----------------------------------------------------
    llm_cfg = LLMConfig(provider=args.provider, temperature=args.temperature,
                        max_tokens=agent_cfg.max_tokens)
    if args.model:
        llm_cfg.model = args.model
    if args.replay:
        client = ReplayClient(records_path=Path(args.replay), config=llm_cfg,
                              strict=False)
        mode = f"回放（{args.replay}）"
    else:
        client = HTTPClient(config=llm_cfg)          # 缺 key 会在这里抛
        mode = f"真机（{llm_cfg.provider} / {llm_cfg.model}）"
        if args.record:
            client = Recorder(client, args.record)
    print(f"  模式：{mode}")
    print(f"  采样：{args.samples} 次/决策   温度 {args.temperature}")
    print(f"  模板：{agent_cfg.template}   杠杆 {args.lever}x\n")

    agent = TradingAgent(client=client, config=agent_cfg, limits=limits)

    # ---- 跑（取最近 N 根，逐根收盘决策）-------------------------------
    start = max(0, n_bars - args.n)
    run_id = args.run_id or _derive_run_id(args, start=start, end=n_bars - 1)
    print(f"  会话 run_id：{run_id}")
    t0 = time.time()
    res = run_session(
        agent, series, start=start, end=n_bars - 1, run_id=run_id,
        equity=args.equity, cash=args.equity,
        data_snapshot=f"okx:{args.inst}:{args.bar}",
        log_path=args.out or None,
    )
    dt = time.time() - t0

    # ---- 报告 ---------------------------------------------------------
    _report(res, dt=dt, n_requested=args.n)
    if args.record and hasattr(client, "n"):
        print(f"  已录制 {client.n} 次调用 → {args.record}")
    if args.replay:
        print(f"  回放覆盖率：{client.coverage:.1%}"
              f"（命中 {client.hits} / 未命中 {client.misses}）")
    if args.out:
        print(f"  决策留痕 → {args.out}")
    return 0


def _ts(ms: int) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(int(ms) / 1000))


def _derive_run_id(args, *, start: int, end: int) -> str:
    """按**参数**派生 run_id（不含墙钟时间）。

    ⚠️ 第一版写成 ``f"cli-{int(time.time())}"``，后果很隐蔽：
    ``decision_id`` 含 ``run_id``，所以同一份数据、同一份 prompt
    在两次运行里会算出**不同的 decision_id**——于是
    「回放能不能核对上」这件事**从构造上就不可能成功**，
    而它恰恰是回放存在的唯一理由（设计文档 §5）。

    ⇒ run_id 必须是"这次实验的配置"的函数，不是"这次运行的时刻"的函数。
    要看时间的话留痕里有 ``wall_ms``，不必挤进 ID。
    """
    parts = [args.inst, args.bar, f"n{args.n}", f"s{args.samples}",
             f"t{args.temperature}", args.template or "v1",
             f"e{args.equity:g}", f"b{start}-{end}"]
    if args.model:
        parts.append(args.model)
    return ":".join(parts)


def _report(res, *, dt: float, n_requested: int) -> None:
    s = log_stats(res.records)
    print("=" * 72)
    print("会话结果")
    print("=" * 72)
    print(f"  决策数：{len(res)} / 请求 {n_requested}"
          f"（跳过 {len(res.skipped)}，用时 {dt:.1f}s）")
    if not res.records:
        print("  ⚠️ 没有产生任何决策——检查行情区间与索引。")
        return
    print(f"  解析成功率 parse_ok_frac：{s.get('parse_ok_frac', 0):.1%}"
          f"   ← ⚠️ 低说明模型没按 JSON 输出，不是策略差")
    print(f"  成交比例 executed_frac  ：{s.get('executed_frac', 0):.1%}")
    print(f"  主动弃权 abstain_frac   ：{s.get('abstain_frac', 0):.1%}")
    print(f"  被风控拒 rejected_frac  ：{s.get('rejected_frac', 0):.1%}")
    print(f"  被风控改量 resized_frac ：{s.get('resized_frac', 0):.1%}"
          f"   ← 这个数字**就是风控的贡献**")
    print(f"  延迟中位/最大：{s.get('latency_ms_p50', 0)} / "
          f"{s.get('latency_ms_max', 0)} ms")

    # 风控拦截明细：按 rule 分组
    by_rule: dict[str, int] = {}
    for r in res.records:
        rule = str(r.risk.get("rule") or "?")
        by_rule[rule] = by_rule.get(rule, 0) + 1
    print("\n  风控规则命中：")
    for k, v in sorted(by_rule.items(), key=lambda kv: -kv[1]):
        print(f"    {k:<24} {v}")

    # 决策稳定性
    votes = [r.parsed.get("_vote") for r in res.records if r.parse_ok]
    if votes and votes[0]:
        fracs = [v.get("majority_frac", 1.0) for v in votes if v]
        if fracs:
            avg = sum(fracs) / len(fracs)
            print(f"\n  决策稳定性（多采样多数派占比均值）：{avg:.3f}")
            if avg >= 0.999 and res.records[0].n_samples > 1:
                print("    说明：全部采样一致。⚠️ 这不一定是好事——"
                      "先确认温度不是 0；")
                print("          温度正常时，也可能是这个上下文对模型太容易"
                      "（比如数据里没有可分辨的分歧点）。")

    print("\n  逐条：")
    for r in res.records:
        print(f"    {r.summary_line()}")
        if r.reject_msg:
            print(f"            ↳ {r.reject_msg[:72]}")


if __name__ == "__main__":
    raise SystemExit(main())
