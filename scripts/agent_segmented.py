"""多段配对实验入口（A6）—— 用**判得出来的度量**跑 LLM vs 基线。

用法::

    # 先看段怎么切、要花多少次调用（**不联网**）
    python scripts/agent_segmented.py --inst BTC-USDT-SWAP --seg-len 50 --segs 8 --dry

    # 真跑（需要 key/额度）
    python scripts/agent_segmented.py --inst BTC-USDT-SWAP --seg-len 50 --segs 8 \
        --template v2 --record out/a6/rec_v2_BTC.jsonl --out-json out/a6/eval_v2_BTC.json

    # 回放（零成本，用于重建结果）
    python scripts/agent_segmented.py ... --replay out/a6/rec_v2_BTC.jsonl

为什么要这个脚本
----------------
`agent_eval.py` 用的是「每根平均收益率」——那个度量在 400 根后
`required_n` 仍有 **55,314**（判不出来）。本脚本换成
「**多段独立运行 + 同段配对检验**」：同一段行情上 LLM 与每条基线各跑一次，
对 K 个**配对差值**做检验。

⚠️ 三条纪律
----------
1. **每段独立起跑**（新 agent、新账户）——这是度量效能高的唯一来源。
   基线免费，但**必须每段也重跑**：配对的前提就是"同一段上比"。
2. **额度先算再跑**：`调用数 = 段数 × 段长 × 采样数`，`--dry` 会先告诉你。
3. **并行按段**（段之间独立）——并行**不改变结果**，
   但录制文件要线程安全（`Recorder` 已加锁）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import banner  # noqa: E402

from tw.agent import AgentConfig, TradingAgent  # noqa: E402
from tw.llm import HTTPClient, LLMConfig, Recorder, ReplayClient  # noqa: E402
from tw.kpi import KPIConfig, calibrate_from_baseline  # noqa: E402
from tw.marketdb import SOURCE_OKX, MarketStore  # noqa: E402
from tw.policy import make_policy  # noqa: E402
from tw.prompts import build_messages  # noqa: E402
from tw.risk import RiskLimits  # noqa: E402
from tw.segmented import (  # noqa: E402
    min_detectable_effect,
    paired_verdict,
    run_paired_segments,
    segment_correlation,
    segment_ranges,
)
from tw.simexec import ExecConfig  # noqa: E402

#: 默认跑的规则基线（`noop` 是零方差基准，`momentum` 是同信息基线）
BASELINES = ("noop", "momentum", "meanrevert", "random_taker")

#: ⚠️ **不参与 KPI 阈值标定的基线**：`noop` 按定义就不动
#: （在场率 0、换手 0）⇒ 留在池里会把分位数下限拉到 0，
#: 于是「别躺平」那三条约束全是空的，而日志看起来像标定成功。
KPI_CALIB_EXCLUDE = ("noop",)

#: ⭐ KPI **在场率下限**的语义钳位：`(下界, 上界)`。
#:
#: 为什么需要钳位——分位数标定本身没错，但**池子决定了语义**：
#: 本项目参与标定的基线（`momentum`/`meanrevert`/`random_taker`）
#: 都是"每根都在场"的策略，在场率 ≈ 100% ⇒ 取 20% 分位得到 **≈75%**。
#: 那是「几乎必须一直持仓」，不是「别躺平」；它会把**选时**这条路
#: 完全堵死，于是测到的是「模型被逼着过度交易」，不是「KPI 有没有用」。
#:
#: - 下界 **0.05**：> 0 才**有牙**——挡住「永不交易」这条退化策略
#:   （KPI 的第一用途就是不让它用"躺平"回避考核）。
#: - 上界 **0.50**：给"选时"留一半空间（做一半的时间在场仍算合格）。
#: 钳到边界时会**显式打印**，不静默。
PRESENCE_CLIP = (0.05, 0.50)


def main() -> int:
    ap = argparse.ArgumentParser(description="多段配对实验（A6）")
    ap.add_argument("--inst", default="BTC-USDT-SWAP")
    ap.add_argument("--source", default=SOURCE_OKX,
                    help="数据来源。库里有 okx（799 根）与 binance_csv"
                         "（17520 根）——**样本量直接决定判力**，"
                         "见 `scripts/a6_power.py`")
    ap.add_argument("--bar", default="1H")
    ap.add_argument("--seg-len", type=int, default=50, help="每段根数 L")
    ap.add_argument("--segs", type=int, default=8, help="段数 K")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--template", default="v2")
    ap.add_argument("--provider", default="agnes")
    ap.add_argument("--model", default=None)
    ap.add_argument("--equity", type=float, default=100_000.0)
    ap.add_argument("--lever", type=float, default=3.0)
    ap.add_argument("--slippage-bps", type=float, default=1.0)
    ap.add_argument("--parallel", type=int, default=4, help="按段并行的线程数")
    ap.add_argument("--compare-to", default="momentum",
                    help="主对照（配对检验的目标）")
    # ---- KPI（A6 的实验变量）------------------------------------------
    ap.add_argument("--kpi", action="store_true",
                    help="给这一组开 KPI（目标 + 在场率下限 + 回撤上限 + 换手区间）")
    ap.add_argument("--calibrate-kpi", action="store_true",
                    help="先用**基线行为分布**标定 KPI 阈值，再跑（推荐）")
    ap.add_argument("--kpi-target", type=float, default=0.01)
    ap.add_argument("--kpi-presence", type=float, default=0.30)
    ap.add_argument("--kpi-dd", type=float, default=0.05)
    ap.add_argument("--kpi-turnover", default="1,40",
                    help="换手区间 lo,hi（×权益）")
    ap.add_argument("--rules-only", action="store_true",
                    help="只跑规则基线（离线、秒级、零额度）")
    ap.add_argument("--dry", action="store_true",
                    help="只算切段与调用次数，**不联网**")
    ap.add_argument("--replay", default="")
    ap.add_argument("--record", default="")
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    # ⚠️ **KPI 只影响带「考核目标」段的模板**。
    # `v5` 与 `v6` 都有那一段，但：
    #   · `v5` 同时还改写了「怎么选」的指令 ⇒ `v4 → v5` **不是单一变量**；
    #   · `v6 = v4 + 考核目标段`（机械验证过）⇒ **KPI 对照请用 v6**。
    # 用别的模板配 `--kpi`，实验变量根本没进 prompt，
    # 而 ``decision_id`` 仍会带 ``#kpi`` 哈希 ⇒ 看起来"做了 KPI 对照"，
    # 实际测的是同一个 prompt 的两遍。**这一类错误必须挡在跑之前。**
    if args.kpi and args.template not in ("v5", "v6"):
        raise SystemExit(
            f"❌ --kpi 只对 --template v5/v6 生效（当前 {args.template}）。"
            f"做单一变量对照请用 **v6**（v5 相对 v4 是两处改动）。")

    banner("多段配对实验（A6）")

    # 需要的总根数 = min_history + K*L，多取一点
    # ⚠️ 取的是**最近**的 `need` 根（`load_candles` 默认从最新往回取）。
    # 段长越短、段数越多，同一个库里可用的段就越多——
    # **这是"能不能判"的主要杠杆**（见 `scripts/a6_power.py` 的判力比）。
    need = 12 + int(args.segs) * int(args.seg_len) + 5
    store = MarketStore()
    try:
        series = store.load_candles(args.source, args.inst, args.bar,
                                    limit=need)
    finally:
        store.close()
    n = len(getattr(series, "close", []))
    rngs = segment_ranges(n, args.seg_len, min_history=12)[: int(args.segs)]
    n_calls = len(rngs) * int(args.seg_len) * int(args.samples)
    print(f"  行情 {args.inst} {args.bar}：读入 {n} 根")
    print(f"  切段：L={args.seg_len}，可用 {len(rngs)} 段（请求 {args.segs}）")
    print(f"  区间：{rngs[0] if rngs else '—'} … {rngs[-1] if rngs else '—'}")
    print(f"  ⚠️ 额度预估：**{n_calls} 次调用**"
          f"（= {len(rngs)} 段 × {args.seg_len} 根 × {args.samples} 采样）")

    if args.dry:
        print("\n  （--dry：不联网，退出）")
        return 0
    if not rngs:
        raise SystemExit("切不出段；把 --segs 调小或 --seg-len 调小")

    base_exec = ExecConfig(slippage_bps=args.slippage_bps)
    factories: dict[str, object] = {}

    # ---- 规则基线：每段都要重跑（配对的前提）--------------------------
    for bn in BASELINES:
        factories[bn] = (lambda _bn: lambda: TradingAgent(
            policy=make_policy(_bn, seed=7), limits=RiskLimits(),
            config=AgentConfig(inst_id=args.inst, bar=args.bar,
                               lever=args.lever)))(bn)

    # ---- KPI：阈值标定（**用基线行为分布，不拍脑袋**）------------------
    # ⚠️⚠️ **必须建在 LLM 工厂之前**：``kpi_cfg`` 要作为构造参数传进
    # ``AgentConfig``。写在后面会 ``UnboundLocalError``（我第一版就写反了，
    # 而这种错**只在真的带 ``--kpi`` 跑时才炸**，不跑就看不出来）。
    kpi_cfg = None
    if args.kpi:
        pres, turn, dd = [], [], []
        if args.calibrate_kpi:
            # 先只跑**基线**（规则策略，免费、秒级）拿行为分布，再据此定阈值。
            # **不要拍脑袋**：阈值比基线松 = 等于没设；比基线严苛得多 =
            # 测到的是「模型被逼到墙角」，不是「KPI 有用」。
            print("\n  标定 KPI：先跑一遍基线取行为分布（免费）…")
            # ⚠️ **`noop` 必须排除出标定池**：它按定义就不动
            # （在场率 0、换手 0），留在池里会把 20% 分位**拉到 0**
            # ⇒ 「在场率下限 0%」= 「别躺平」这条约束是空的，
            # 而日志看起来像标定成功。**实测踩过一次。**
            calib_factories = {k: v for k, v in factories.items()
                               if k not in KPI_CALIB_EXCLUDE}
            print(f"    标定池：{sorted(calib_factories)}"
                  f"（已剔除 {list(KPI_CALIB_EXCLUDE)}，理由见代码注释）")
            pre = run_paired_segments(
                series, calib_factories, seg_len=args.seg_len, min_history=12,
                initial_cash=args.equity, lever=args.lever,
                max_segs=args.segs, exec_config=base_exec,
                parallel=max(1, args.parallel),
                # ⚠️ ``keep_runs=True`` **不能省**：默认 False 时 ``pre.runs`` 是
                # 空的 ⇒ 标定拿不到数据 ⇒ 静默退回命令行阈值，
                # 而日志看起来像「标定成功」。这是"沉默失败"，最难查的一类。
                keep_runs=True,
            )
            for per in (pre.runs or []):
                for _nm, rr in per.items():
                    if _nm in KPI_CALIB_EXCLUDE:
                        continue
                    eq, qty = rr.equity_curve, rr.qty_curve
                    if not eq:
                        continue
                    peak, mdd = eq[0], 0.0
                    for e in eq:
                        peak = max(peak, e)
                        if peak > 0.0:
                            mdd = max(mdd, (peak - e) / peak)
                    pres.append(sum(1 for q in qty if abs(q) > 1e-12)
                                / max(len(qty), 1))
                    dd.append(mdd)
                    turn.append(
                        float(rr.meta.get("exec", {}).get("turnover", 0.0))
                        / max(rr.initial_equity, 1.0))
            if pres:
                cal = calibrate_from_baseline(presence=pres, turnover=turn,
                                              drawdown=dd)
                print(f"    标定结果：在场率下限 {cal['min_presence']:.1%}、"
                      f"回撤上限 {cal['max_drawdown']:.2%}、"
                      f"换手区间 [{cal['turnover_lo']:.2f}, "
                      f"{cal['turnover_hi']:.2f}]（n={cal['n']} 个段观测）")
                # ⚠️ **标定结果必须被检查，不能直接采信**：退化成 0 时
                # 那条约束是空的，而"跑出来的数字"看起来完全正常。
                if cal["min_presence_vacuous"]:
                    raise SystemExit(
                        "❌ KPI 标定失败：在场率下限退化成 0 ⇒ "
                        "「别躺平」这条约束**没有任何约束力**。"
                        "检查标定池里是否混进了按定义不动的基线。")
                _raw = float(cal["min_presence"])
                _clo, _chi = PRESENCE_CLIP
                _clipped = min(max(_raw, _clo), _chi)
                if abs(_clipped - _raw) > 1e-12:
                    print(f"    ⭐ 在场率下限钳位：{_raw:.1%} → {_clipped:.1%}"
                          f"（上界 {_chi:.0%}：留出「选时」空间；"
                          f"下界 {_clo:.0%}：挡住「永不交易」）")
                args.kpi_presence = _clipped
                args.kpi_dd = float(cal["max_drawdown"])
                args.kpi_turnover = f"{cal['turnover_lo']},{cal['turnover_hi']}"
            else:
                print("    ⚠️ 标定失败（拿不到基线行为分布）⇒ 沿用命令行阈值")
        lo, hi = (float(x) for x in args.kpi_turnover.split(","))
        kpi_cfg = KPIConfig(target_return=args.kpi_target,
                            min_presence=args.kpi_presence,
                            max_drawdown=args.kpi_dd,
                            turnover_lo=lo, turnover_hi=hi)
        print(f"  ⭐ KPI 已开启：{kpi_cfg.describe()}")

    llm_name = f"llm_{args.template}"
    client = None
    if not args.rules_only:
        cfg = LLMConfig(provider=args.provider, temperature=args.temperature)
        if args.model:
            cfg.model = args.model
        if args.replay:
            client = ReplayClient(records_path=Path(args.replay), config=cfg,
                                  strict=False)
            print(f"\n  LLM：回放 {args.replay}")
        else:
            client = HTTPClient(config=cfg)
            print(f"\n  LLM：真机 {cfg.provider}/{cfg.model}"
                  f"  {args.samples} 采样/决策")
        # ⚠️ ``--record`` 必须在**两个分支都生效**：我第一版只在真机分支里包
        # ``Recorder`` ⇒ 回放模式下 ``--record`` 被**静默忽略**，
        # 于是"读实际发出去的 prompt"这条自检退化成"只查模板标签"（强度不足），
        # 而日志只是淡淡地说了一句"未给 --record"。**静默忽略参数**是最坏的一种。
        if args.record:
            client = Recorder(client, args.record)
        agent_cfg = AgentConfig(inst_id=args.inst, bar=args.bar,
                                template=args.template, lever=args.lever,
                                n_samples=args.samples,
                                temperature=args.temperature,
                                # ⚠️⚠️ **这一行曾经漏掉过**：CLI 没传 kpi，
                                # 于是 v5 的 prompt 渲染成「本窗口未设置考核目标」，
                                # 而实验**照跑、照出数字** —— 差一点得出
                                # 「KPI 无效」的假结论。
                                # ⇒ 是"读实际发出去的 prompt"才发现的。
                                kpi=kpi_cfg)
        factories[llm_name] = (lambda _c: lambda: TradingAgent(
            client=_c, config=agent_cfg, limits=RiskLimits()))(client)

    # ---- 跑 -----------------------------------------------------------
    t0 = time.time()
    last = [0]

    def _prog(i: int, total: int) -> None:
        if i - last[0] >= 4 or i == total:
            last[0] = i
            print(f"    进度 {i}/{total}", flush=True)

    res = run_paired_segments(
        series, factories, seg_len=args.seg_len, min_history=12,
        initial_cash=args.equity, lever=args.lever, max_segs=args.segs,
        exec_config=base_exec, parallel=max(1, args.parallel),
        progress=None if args.dry else _prog,
        # ⚠️ ``keep_runs=True`` **不能省**：末段的「KPI 逐段判定」与接线自检
        # 都读 ``res.runs``。默认 False 时它们是空的 ⇒ 那两段会被**静默跳过**，
        # 日志看起来完全正常（又一次"沉默失败"）。
        keep_runs=True,
    )
    dt = time.time() - t0

    # ---- ⭐ 接线自检：**读实际发出去的 prompt**，不看参数 ---------------
    # 上一次的教训：CLI 漏传 ``kpi`` 时**参数看着是对的**，
    # 是去读 prompt 正文才发现渲染成了「本窗口未设置考核目标」，
    # 而实验**照跑、照出数字** —— 差一点得出「KPI 无效」的假结论。
    #
    # ⚠️ 我第一版自检写成"用留痕的 visible_state + ``kpi_state=None``
    # 重新渲染" ⇒ **误报**（``_kpi_text`` 见 ``st is None`` 就渲染成
    # 「未设置」）。重建出来的 prompt 与真正发出去的不是同一个东西。
    # ⇒ 现在直接读 **``--record`` 记下的 messages 原文**：那是唯一
    #    无法辩驳的证据（"读实际发出去的东西"）。
    if kpi_cfg is not None and not args.rules_only:
        seen_labels, prompt_ok, src = [], None, ""
        for per in (res.runs or []):
            for nm, rr in per.items():
                if nm.startswith("llm_") and rr.records:
                    seen_labels.append(rr.records[0].prompt_template)
            if seen_labels:
                break
        if args.record and Path(args.record).exists():
            with open(args.record, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    txt = " ".join(str(m.get("content", ""))
                                   for m in (row.get("messages") or []))
                    if not txt:
                        continue
                    prompt_ok = ("在场率下限" in txt) and ("未设置" not in txt)
                    src = f"{args.record}（录制原文）"
                    break
        else:
            src = "（未给 --record，退化为只查 prompt_template 标签）"
        _bad = [x for x in seen_labels if "#kpi" not in str(x)]
        print("\n  ⭐ KPI 接线自检（读**实际发出去的 prompt**，不看参数）")
        print(f"     证据来源：{src}")
        print(f"     decision_id.prompt_template 例：{seen_labels[:1]}")
        print(f"     prompt 正文含 KPI 段：{prompt_ok}")
        if _bad or prompt_ok is False:
            raise AssertionError(
                "❌ KPI 没真的接上："
                f"prompt_template 缺 '#kpi' 的有 {len(_bad)} 段；"
                f"prompt 正文检查={prompt_ok!r}。"
                "这一组数据**不可用于 KPI 结论**，必须修好重跑。")
        if prompt_ok is None:
            print("     ⚠️ 没拿到 prompt 原文（缺 --record）⇒ "
                  "只验证了模板标签，**强度不足**，建议补 --record 重跑。")

    # ---- 报告 ---------------------------------------------------------
    print("\n" + "=" * 74)
    print(f"多段配对结果（{res.n_segs} 段 × {args.seg_len} 根，用时 {dt:.0f}s）")
    print("=" * 74)
    bl = args.compare_to
    for nm in res.configs():
        xs = res.series(nm)
        m = sum(xs) / len(xs)
        cr = segment_correlation(res.net, nm)
        print(f"  {nm:<16} 段均 {m:>+8.4%}   段间相关 {cr:>+6.3f}")
    if llm_name in res.configs() and bl in res.configs():
        d = res.diffs(llm_name, bl)
        v = paired_verdict(d, name_a=llm_name, name_b=bl)
        mde = min_detectable_effect(_sd_via(d), len(d))
        print(f"\n  主对照（{llm_name} − {bl}）：")
        print(f"    {v['reason']}")
        print(f"    t = {v['t']:.2f}（df={v['df']}，临界 {v['t_crit']:.2f}）")
        print(f"    → **{v['verdict']}**")
        print(f"    本实验在 n={len(d)} 时的最小可分辨效应 ≈ {mde:.4%}")
        if llm_name in res.configs() and "noop" in res.configs():
            vn = paired_verdict(res.diffs(llm_name, "noop"),
                                name_a=llm_name, name_b="noop")
            print(f"    （vs noop：{vn['verdict']}）")

    # ⭐ KPI 的官方判定（**只看行为**，不看模型说什么）
    if kpi_cfg is not None and res.runs:
        print("\n  ⭐ KPI 判定（逐段，只看行为）：")
        for k, per in enumerate(res.runs):
            for nm, rr in per.items():
                if nm.startswith("llm_"):
                    kf = rr.meta.get("kpi_final") or {}
                    ks = rr.meta.get("kpi_state") or {}
                    print(f"    段{k} {nm}: passed={kf.get('passed')}"
                          f"  failed={kf.get('failed')}"
                          f"  在场率 {ks.get('presence_so_far', 0):.1%}"
                          f"  回撤 {ks.get('drawdown_so_far', 0):.2%}"
                          f"  换手 {ks.get('turnover_x', 0):.2f}×")

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        from tw.eval_agent import json_safe

        Path(args.out_json).write_text(json.dumps(
            json_safe({"config": vars(args), "result": res.to_dict(),
                       "kpi": (kpi_cfg.describe() if kpi_cfg is not None
                               else None),
                       "kpi_by_segment": [
                           {"seg": k, "config": nm,
                            "passed": (rr.meta.get("kpi_final") or {}).get(
                                "passed"),
                            "failed": (rr.meta.get("kpi_final") or {}).get(
                                "failed"),
                            "state": rr.meta.get("kpi_state") or {}}
                           for k, per in enumerate(res.runs or [])
                           for nm, rr in per.items()
                           if nm.startswith("llm_")
                       ],
                       "verdicts": {
                           f"{n2}_vs_{bl}": paired_verdict(
                               res.diffs(n2, bl), name_a=n2, name_b=bl)
                           for n2 in res.configs() if bl in res.configs()
                       }}),
            ensure_ascii=False, indent=2, default=str, allow_nan=False),
            encoding="utf-8")
        print(f"\n  JSON → {args.out_json}")
    if client is not None and hasattr(client, "coverage"):
        print(f"  回放覆盖率：{client.coverage:.1%}"
              f"（命中 {client.hits} / 未命中 {client.misses}）")
    return 0


def _sd_via(xs: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    m = sum(xs) / n
    return (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5


if __name__ == "__main__":
    raise SystemExit(main())
