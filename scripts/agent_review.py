"""复盘与归因分析（A6 第 ② 块）—— **零额度**回答用户的三个问题。

用户原话
--------
> 如果 LLM 判断错误，他会怎么分析这个错误？判断成功又怎么分析？
> 这些东西能不能作为经验呢？
> 而不是对了就对了，错了就错了，有种无所谓的感觉。

本脚本把这条链路真的跑一遍，并且**不花额度**：

```
已有的录制文件（llm 的原始回答）
   └─ 回放重建（ReplayClient，0 次调用）→ 决策留痕 DecisionRecord
        └─ 回填结果（outcome.py，horizon 根之后）
             └─ 规则归因（reflect.classify_by_rules，0 成本）
                  └─（可选）LLM 复盘（每段 1 次调用）→ 结构化归因 + 经验条目
```

为什么要走**回放**而不是分析 out/*.json
--------------------------------------
`out/a6/eval_*.json` 里只有收益数字，**没有逐条决策**；
要归因就必须拿到 `DecisionRecord`（含 `outcome`）。
录制文件里存的是"当时实际发出去的 messages 与回答"，
回放能把它们**逐条重建**成同样的决策 —— 而回放**零调用**。

⚠️ 两个必须说清楚的边界
----------------------
1. **复盘不是决策**。复盘时看到结果是应该的；危险的是把复盘结论喂回决策。
   本脚本只做分析，**不把任何东西写回决策路径**。
2. **规则归因是 LLM 归因的对照**。如果 LLM 的归因与规则归因高度一致，
   那"归因有信息"可能只是"复述了结果"——这一点必须报出来，不能藏。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import banner  # noqa: E402

from tw.agent import AgentConfig, TradingAgent  # noqa: E402
from tw.kpi import KPIConfig  # noqa: E402
from tw.llm import LLMConfig, Recorder, ReplayClient  # noqa: E402
from tw.marketdb import SOURCE_OKX, MarketStore  # noqa: E402
from tw.outcome import BackfillConfig, backfill  # noqa: E402
from tw.reflect import (  # noqa: E402
    ATTR_REASONS,
    ExperienceStore,
    Exp,
    ExperienceStore,
    ReviewConfig,
    attribution_association,
    build_review_messages,
    classify_by_rules,
    duplicate_rate,
    merits_to_store,
    parse_review,
    review_at_tick,
)
from tw.risk import RiskLimits  # noqa: E402
from tw.segmented import run_paired_segments  # noqa: E402
from tw.simexec import ExecConfig  # noqa: E402


def behavior_metrics(rr) -> dict:
    """从一次段运行的曲线算**行为指标**（不联网、不看模型说什么）。

    为什么要单独算而不是直接读 ``meta``：``meta["exec"]["turnover"]`` 是
    成交额累计，**在场率与回撤完全不在 meta 里**。而"KPI 有没有改变行为"
    这三个数才是主判据（自报不可信 —— `basis` 的教训：自报"照指标"与
    外部可观测量只有 36% 一致）。
    """
    eq = list(rr.equity_curve or [])
    qty = list(rr.qty_curve or [])
    peak = eq[0] if eq else 0.0
    mdd = 0.0
    for e in eq:
        peak = max(peak, e)
        if peak > 0:
            mdd = max(mdd, (peak - e) / peak)
    pres = (sum(1 for q in qty if abs(q) > 1e-12) / len(qty)) if qty else 0.0
    turn = float((rr.meta or {}).get("exec", {}).get("turnover", 0.0))
    base = float(rr.initial_equity or 1.0)
    return {"presence": pres, "max_drawdown": mdd,
            "turnover_x": turn / base if base else 0.0,
            "net": (eq[-1] - base) / base if eq else 0.0}


def main() -> int:
    ap = argparse.ArgumentParser(description="复盘与归因分析（A6，默认零额度）")
    ap.add_argument("--replay", required=True,
                    help="已有的录制文件（rec_*.jsonl），用它回放重建决策")
    ap.add_argument("--template", default="v2")
    ap.add_argument("--kpi", action="store_true",
                    help="重建 prompt 时带上 KPI（v5 录制必须加）")
    ap.add_argument("--exp-from", default="",
                    help="重建 prompt 时注入经验库（v7 录制必须给）。"
                         "⚠️ 与 `--kpi-json` 同理：**prompt 必须与被录制时"
                         "逐字节一致**，否则回放全部未命中（覆盖率 0.0%）")
    ap.add_argument("--kpi-json", default="",
                    help="从这份 segmented 评估 JSON 里读回 ``kpi`` 配置。"
                         "⚠️ **回放的 prompt 必须与被录制时逐字节一致**——"
                         "少了 KPI 段就会全部未命中（实测覆盖率 0.0%），"
                         "而「照跑照出数字」是**最危险的**失败方式")
    ap.add_argument("--kpi-target", type=float, default=0.01)
    ap.add_argument("--kpi-presence", type=float, default=0.5)
    ap.add_argument("--kpi-dd", type=float, default=0.02)
    ap.add_argument("--kpi-turnover", default="1.75,3.69")
    ap.add_argument("--inst", default="BTC-USDT-SWAP")
    ap.add_argument("--source", default=SOURCE_OKX,
                    help="⚠️ 必须与录制时**同一个来源**——否则回放的是"
                         "另一份行情（覆盖率会掉到 0，脚本会直接失败）")
    ap.add_argument("--bar", default="1H")
    ap.add_argument("--seg-len", type=int, default=50)
    ap.add_argument("--max-fail-rate", type=float, default=0.01,
                    help="⭐ 调用失败率上限；超了直接失败（额度耗尽会让经验库悄悄变空）")
    ap.add_argument("--min-history", type=int, default=12,
                    help="⚠️ 必须与录制时一致（三臂消融用了 24）")
    ap.add_argument("--features-mode", default="real",
                    choices=("real", "shifted", "none"),
                    help="⚠️ 三臂消融的录制必须原样重建，否则覆盖率 0%")
    ap.add_argument("--feature-shift", type=int, default=0)
    ap.add_argument("--seg-offset", type=int, default=0,
                    help="⚠️ 必须与录制时**同一个偏移**——否则重建的是**另一组窗口**")
    ap.add_argument("--segs", type=int, default=8)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=4,
                    help="回填窗口（决策后看多少根）")
    ap.add_argument("--review-period", type=int, default=24,
                    help="每多少根复盘一次（默认 24 ≈ 一天）")
    ap.add_argument("--review-records", type=int, default=8,
                    help="一次复盘喂多少条决策。⚠️ 它决定输出长度："
                         "太大 ⇒ 模型输出被 max_tokens 截断 ⇒ 归因全丢。"
                         "实测 24 条时 700 token 会在 1365 字符处截断")
    ap.add_argument("--review-max-tokens", type=int, default=2000)
    ap.add_argument("--llm-review", action="store_true",
                    help="额外做 LLM 复盘（每段 1 次调用；不加则纯离线）")
    ap.add_argument("--provider", default="agnes")
    ap.add_argument("--model", default=None)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="复盘用低温（要的是分类，不是创意）")
    ap.add_argument("--record-review", default="",
                    help="把复盘对话也录下来（证据）")
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    banner("复盘与归因分析（A6）")

    # ⚠️ 同 agent_segmented：必须用 `--min-history`，写死 12 会少取根数
    need = int(args.min_history) + args.segs * args.seg_len + 5
    store = MarketStore()
    try:
        series = store.load_candles(args.source, args.inst, args.bar,
                                    limit=need)
    finally:
        store.close()

    # ---- 回放重建决策（0 次真实调用）----------------------------------
    cfg = LLMConfig(provider=args.provider, temperature=0.2)
    if args.model:
        cfg.model = args.model

    # ⚠️⚠️ **回放的 prompt 必须与被录制时完全一致**，否则全部未命中。
    # 凡是影响 prompt 正文的配置（模板版本、KPI）都必须原样重建。
    kpi_cfg = None
    src_kpi = ""
    if args.kpi_json:
        _p = Path(args.kpi_json)
        _d = json.loads(_p.read_text(encoding="utf-8"))
        _k = _d.get("kpi")
        if not _k:
            raise SystemExit(f"❌ {_p} 里没有 kpi 配置 ⇒ 无法重建 prompt")
        kpi_cfg = KPIConfig(target_return=float(_k["target_return"]),
                            min_presence=float(_k["min_presence"]),
                            max_drawdown=float(_k["max_drawdown"]),
                            turnover_lo=float(_k["turnover_lo"]),
                            turnover_hi=float(_k["turnover_hi"]))
        src_kpi = f"{_p}（沿用当时实际用的 KPI）"
    elif args.kpi:
        lo, hi = (float(x) for x in args.kpi_turnover.split(","))
        kpi_cfg = KPIConfig(target_return=args.kpi_target,
                            min_presence=args.kpi_presence,
                            max_drawdown=args.kpi_dd,
                            turnover_lo=lo, turnover_hi=hi)
        src_kpi = "命令行"
    if kpi_cfg is not None:
        print(f"  KPI（用于重建 prompt）：{kpi_cfg.describe()}")
        print(f"     来源：{src_kpi}")

    # ⚠️ 同理：注入经验库也是"影响 prompt 正文"的配置，必须原样重建。
    exp_store = None
    if args.exp_from:
        _ep = Path(args.exp_from)
        _ed = json.loads(_ep.read_text(encoding="utf-8"))
        _items = _ed.get("experiences") or []
        if not _items:
            raise SystemExit(f"❌ {_ep} 里没有经验 ⇒ 重建出的 prompt 不是 v7")
        exp_store = ExperienceStore.from_dict(_items)
        print(f"  经验库（用于重建 prompt）：{len(exp_store)} 条，来自 {_ep}")

    client = ReplayClient(records_path=Path(args.replay), config=cfg,
                          strict=False)
    agent_cfg = AgentConfig(inst_id=args.inst, bar=args.bar,
                            template=args.template, n_samples=args.samples,
                            temperature=0.2, kpi=kpi_cfg,
                            exp_store=exp_store,
                            features_mode=args.features_mode,
                            features_shift=int(args.feature_shift))
    llm = f"llm_{args.template}"
    factories = {llm: (lambda _c: lambda: TradingAgent(
        client=_c, config=agent_cfg, limits=RiskLimits()))(client)}
    print(f"  回放：{args.replay}")
    print(f"  重建：{args.segs} 段 × {args.seg_len} 根")

    res = run_paired_segments(
        series, factories, seg_len=args.seg_len,
        min_history=int(args.min_history),
        max_segs=args.segs, exec_config=ExecConfig(),
        parallel=1, keep_runs=True, seg_offset=args.seg_offset,
    )
    cov = getattr(client, "coverage", float("nan"))
    print(f"  回放覆盖率：{cov:.1%}（命中 {getattr(client, 'hits', 0)} / "
          f"未命中 {getattr(client, 'misses', 0)}）")
    # ⚠️⚠️ **覆盖率必须报出来，而且低覆盖率必须直接失败。**
    # 未命中时回放不会报错：它会退化成"没有回答" ⇒ agent 一律 hold
    # ⇒ 整段行为变成"从不交易"，于是行为指标全是 0、归因全是 unclear，
    # **而日志看起来很干净**。实测踩到过（v5 覆盖率 0.0%）。
    # 拿这样的"重建"去做归因，是在分析**另一段历史**。
    if not (cov >= 0.99):
        raise SystemExit(
            f"❌ 回放覆盖率 {cov:.1%} < 99% ⇒ 重建出来的不是当时的历史。"
            "最可能的原因：**重建 prompt 的配置与录制时不一致**"
            "（例如 v5 录制必须带 --kpi / --kpi-json）。"
            "**不要用这份结果下任何结论。**")

    # ---- 回填结果 + 归因 ---------------------------------------------
    bcfg = BackfillConfig(horizon=args.horizon)
    all_pairs: list[tuple[str, float]] = []
    per_seg: list[dict] = []
    n_rec = n_filled = n_pending = n_bad = 0
    n_hold = 0
    behaviors: list[dict] = []
    examples: dict[str, list[dict]] = {r: [] for r in ATTR_REASONS}
    boundary_violations = 0

    for k, per in enumerate(res.runs or []):
        rr = per.get(llm)
        if rr is None:
            continue
        stat = backfill(rr.records, series, cfg=bcfg,
                        equity_curve=rr.equity_curve)
        n_rec += stat["n"]
        n_filled += stat["n_filled"]
        n_pending += stat["n_pending"]
        n_bad += stat["n_bad_input"]
        beh = behavior_metrics(rr)
        behaviors.append({"seg": k, **beh})
        n_hold += sum(1 for r in rr.records
                      if str(r.parsed.get("action", "hold")).lower() == "hold")
        counts: dict[str, int] = {r: 0 for r in ATTR_REASONS}
        for rec in rr.records:
            reason = classify_by_rules(rec)
            counts[reason] = counts.get(reason, 0) + 1
            if rec.outcome:
                mo = float(rec.outcome.get("markout_h", 0.0))
                all_pairs.append((reason, mo))
                if len(examples[reason]) < 3:
                    examples[reason].append({
                        "seg": k, "tick": rec.tick,
                        "action": rec.parsed.get("action"),
                        "markout_bp": round(mo * 1e4, 2),
                        "mfe_bp": (round(float(rec.outcome["mfe"]) * 1e4, 2)
                                   if "mfe" in rec.outcome else None),
                        "mae_bp": (round(float(rec.outcome["mae"]) * 1e4, 2)
                                   if "mae" in rec.outcome else None),
                        "reason": str(rec.parsed.get("reason", ""))[:80],
                    })
        per_seg.append({"seg": k, "counts": counts,
                        "n_outcome": sum(counts.values())})

    assoc = attribution_association(all_pairs)

    # ---- LLM 复盘（可选；经验库按 t' < t 检索，**这是安全边界**）------
    rcfg = ReviewConfig(period=args.review_period,
                        max_records=args.review_records,
                        max_tokens=args.review_max_tokens)
    exp_store = ExperienceStore()
    reviews: list[dict] = []
    rclient = None
    if args.llm_review:
        rcfg2 = LLMConfig(provider=args.provider,
                          temperature=args.temperature,
                          max_tokens=args.review_max_tokens)
        if args.model:
            rcfg2.model = args.model
        from tw.llm import HTTPClient
        rclient = HTTPClient(config=rcfg2)
        if args.record_review:
            rclient = Recorder(rclient, args.record_review)

    for k, per in enumerate(res.runs or []):
        rr = per.get(llm)
        if rr is None:
            continue
        recs = [r for r in rr.records]
        # ⚠️ 复盘窗口：每 period 根一次。**窗口再按 max_records 切块**——
        # 输出被截断时归因会整批丢，宁可多打几次。
        step = max(1, rcfg.period)
        for i in range(step, len(recs) + 1, step):
            window = recs[max(0, i - step):i]
            # ⚠️⚠️ **`at_tick` 必须是「窗口最后一根 + horizon」，不是最后一根。**
            # 复盘用到的 outcome 延伸到 `last + horizon`，
            # 若把经验记在 `last`，它们就会在 `last+1` 可见却"知道"到 `last+4`
            # ⇒ **凭空多出 horizon−1 根前视**，而且不报错。
            # （我第一版就写成了 `window[-1].tick`。）
            at_tick = review_at_tick(window, horizon=args.horizon)
            # ⭐ 时间层边界：只用**严格更早**的经验
            exps = exp_store.retrieve(at_tick=at_tick,
                                      k=rcfg.max_experiences)
            # ⭐⭐ **每次取回都当场验证边界**（不是事后统计）：
            # 只要有一条 `created_tick >= at_tick`，就是未来泄漏。
            # 这比"在别处算一个违规数"可靠——**检查要贴着被检查的动作**。
            for e in exps:
                if e.created_tick >= at_tick:
                    boundary_violations += 1
            entry = {"seg": k, "at_tick": at_tick,
                     "n_window": len(window),
                     "n_exp_seen": len(exps),
                     "exp_ids_seen": [e.exp_id for e in exps]}
            if rclient is not None:
                entry["n_chunks"] = 0
                entry["attributions"] = []
                entry["lessons"] = []
                entry["ok"] = True
                entry["repaired_chunks"] = 0
                entry["n_bad_reason"] = 0
                for c0 in range(0, len(window), rcfg.max_records):
                    chunk = window[c0:c0 + rcfg.max_records]
                    msgs = build_review_messages(chunk, at_tick=at_tick,
                                                 experiences=exps)
                    resp = rclient.chat(msgs)
                    parsed = parse_review(resp.text if resp.ok else "")
                    entry["n_chunks"] += 1
                    entry["ok"] = entry["ok"] and parsed["ok"]
                    entry["repaired_chunks"] += int(
                        bool(parsed.get("repaired")))
                    entry["n_bad_reason"] += parsed["n_bad_reason"]
                    entry["attributions"].extend(parsed["attributions"])
                    # ⚠️ ``created_tick`` 由**外部时钟**给，不是模型说的
                    new_exps = merits_to_store(parsed, created_tick=at_tick)
                    for e in new_exps:
                        exp_store.add(e)
                    entry["lessons"].extend(e.lesson for e in new_exps)
                entry["n_exp_added"] = len(entry["lessons"])
            reviews.append(entry)

    # ---- 报告 ---------------------------------------------------------
    # ---- ⭐ 复盘调用失败率守卫（同 agent_segmented 的理由）------------
    if args.record_review:
        from _common import check_call_fail_rate
        try:
            _st = check_call_fail_rate(args.record_review,
                                       max_rate=args.max_fail_rate,
                                       where="复盘")
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from None
        print(f"\n  ⭐ 复盘调用失败率：{_st['n_bad']}/{_st['n_all']}"
              f"（{_st['rate']:.2%}）")

    print("\n" + "=" * 74)
    print(f"复盘与归因（回放重建，回放覆盖率 {cov:.1%}）")
    print("=" * 74)
    print(f"  决策 {n_rec} 条，其中已回填结果 {n_filled} 条"
          f"（horizon={args.horizon}）；待回填 {n_pending}（未来还没发生，正常）"
          f"、输入不可用 {n_bad}（**这个不为 0 就要查**）")

    # ⭐ 行为侧（**KPI 的主判据**，不看模型说什么）
    if behaviors:
        def _avg(key: str) -> float:
            xs = [b[key] for b in behaviors]
            return sum(xs) / len(xs) if xs else float("nan")
        print("\n  ⭐ 行为侧（逐段 + 段均；**这才是 KPI 有没有效果的判据**）：")
        for b in behaviors:
            print(f"    段{b['seg']}  在场率 {b['presence']:>6.1%}  "
                  f"回撤 {b['max_drawdown']:>6.2%}  "
                  f"换手 {b['turnover_x']:>5.2f}×  "
                  f"净收益 {b['net']:>+7.3%}")
        print(f"    ── 段均：在场率 {_avg('presence'):.1%}  "
              f"回撤 {_avg('max_drawdown'):.2%}  "
              f"换手 {_avg('turnover_x'):.2f}×  "
              f"净收益 {_avg('net'):+.3%}")
        print(f"    弃权（hold）占比 {n_hold / max(n_rec, 1):.1%}"
              f"（{n_hold}/{n_rec} 条决策）")
    print("\n  ⭐ 归因分布（规则判定，**不看模型给的理由**）：")
    for r in ATTR_REASONS:
        n = assoc["n_by_reason"].get(r, 0)
        if n:
            m = assoc["mean_result_by_reason"].get(r, float("nan"))
            print(f"    {r:<16} n={n:<4} 平均结果 {m * 1e4:>+7.2f} bp")
    print(f"\n  分类间极差 spread = {assoc['spread'] * 1e4:.2f} bp "
          f"（越大 ⇒ 分类越「有信息」）")
    print(f"  解耦类（lucky/unlucky）内部结果与标签相反的比例 = "
          f"{assoc['decoupled_share']:.1%}")
    print("  ⚠️ 这只做描述，**不下显著结论**——"
          "显著性要走配对检验（见 tw/segmented.paired_verdict）")

    if args.llm_review:
        n_ok = sum(1 for x in reviews if x.get("ok"))
        n_bad = sum(int(x.get("n_bad_reason", 0)) for x in reviews)
        n_rep = sum(int(x.get("repaired_chunks", 0)) for x in reviews)
        n_chunk = sum(int(x.get("n_chunks", 0)) for x in reviews)
        n_attr = sum(len(x.get("attributions") or []) for x in reviews)
        print(f"\n  ⭐ LLM 复盘：{len(reviews)} 次窗口 / {n_chunk} 次调用，"
              f"解析成功 {n_ok}，归因 {n_attr} 条，非法分类 {n_bad} 条")
        if n_rep:
            print(f"     ⚠️ 其中 {n_rep} 次调用的输出**被截断**（已按条目抢救）"
                  f" ⇒ 调大 --review-max-tokens 或调小 --review-records")
        print(f"     经验库累计 {len(exp_store)} 条，"
              f"教训文本重复率 {duplicate_rate(exp_store.all()):.1%}")
        if reviews:
            last = reviews[-1]
            print(f"     最后一次复盘看到 {last['n_exp_seen']} 条经验"
                  f"（t' < {last['at_tick']}）")
            for ls in (last.get("lessons") or [])[:4]:
                print(f"       · {ls[:96]}")
        # ⭐ 经验条目的示例（含它引用了哪些决策 —— 可回溯）
        for e in exp_store.all()[:5]:
            print(f"     [{e.exp_id}] tick={e.created_tick} kind={e.kind} "
                  f"evidence={len(e.evidence_ids)} 条 → {e.lesson[:80]}")

    # ⭐ 安全自检：**同一 tick 生成的经验不能在当前 tick 被看到**
    print(f"\n  ✅ 时间边界自检：{len(reviews)} 次复盘，"
          f"每次都只看到 t' < at_tick 的经验"
          f"（违规 {boundary_violations} 次）")
    if boundary_violations:
        raise AssertionError(
            f"未来泄漏：{boundary_violations} 条经验在被生成的那一刻就被看到了。"
            "检查 ExperienceStore.retrieve 是否退回了 `<=`。")

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        from tw.eval_agent import json_safe
        Path(args.out_json).write_text(json.dumps(json_safe({
            "config": vars(args),
            "replay_coverage": cov,
            "n_decisions": n_rec, "n_outcome_filled": n_filled,
            "attribution_counts": assoc["n_by_reason"],
            "attribution_mean_result_bp": {
                k: (v * 1e4 if v == v else None)
                for k, v in assoc["mean_result_by_reason"].items()},
            "spread_bp": (assoc["spread"] * 1e4
                          if assoc["spread"] == assoc["spread"] else None),
            "decoupled_share": assoc["decoupled_share"],
            "behavior_by_segment": behaviors,
            "n_hold": n_hold,
            "examples": examples,
            "reviews": reviews,
            "experiences": exp_store.to_dict(),
            "duplicate_rate": duplicate_rate(exp_store.all()),
        }), ensure_ascii=False, indent=2, default=str, allow_nan=False),
            encoding="utf-8")
        print(f"\n  JSON → {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
