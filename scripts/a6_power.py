"""效能标定（A6）——**零额度**回答"这个度量到底判不判得出来"。

用户拍板：「**不接受无法判定**」。
但这句要求必须先回答一个更基础的问题：**想判的效应有多大？**
本脚本用**免费基线**（规则策略，不调 LLM）把这件事量化出来。

为什么用基线而不是 LLM
----------------------
- 基线零成本 ⇒ 可以把网格扫得很密（段长 × 段数）。
- 基线之间的真实差异是**已知量级**的（例如 `momentum − noop`），
  拿它对照 LLM 的差异，就知道"LLM 那边需要多大样本"。

⭐ 一个可以直接算出来的结论：**MDE ∝ 段长**
------------------------------------------
总根数 `N = K·L` 固定时：

- 每段的配对差值 ≈ `L` 根上差异的**和** ⇒ `σ_段 ∝ √L`
- `MDE = t_crit(df=K−1) · σ_段 / √K`，而 `K = N/L`
⇒ `MDE ∝ √L / √(N/L) = L / √N`

**也就是说：给定总根数，段越短，可分辨的效应越小（越好）。**
⚠️ 但这条推导假设各根差异近似独立。如果差异有自相关（趋势/波动聚集），
   `σ_段` 会比 `√L` 长得慢，结论会变弱。**所以必须实测，不能只信推导。**

⇒ 本脚本就是那个实测。

用法::

    python scripts/a6_power.py --inst BTC-USDT-SWAP
    python scripts/a6_power.py --inst BTC-USDT-SWAP --seg-lens 8,12,16,25,50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import banner  # noqa: E402

from tw.agent import AgentConfig, TradingAgent  # noqa: E402
from tw.marketdb import SOURCE_OKX, MarketStore  # noqa: E402
from tw.policy import make_policy  # noqa: E402
from tw.risk import RiskLimits  # noqa: E402
from tw.segmented import (  # noqa: E402
    min_detectable_effect,
    paired_verdict,
    power_analysis,
    run_paired_segments,
    segment_correlation,
    segment_ranges,
)
from tw.simexec import ExecConfig  # noqa: E402

#: 默认扫的段长。上界 50 = 历史配置；下界 8 ≈ 半天的 1H 线。
DEFAULT_SEG_LENS = (8, 12, 16, 25, 50)


def _sd(xs: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    m = sum(xs) / n
    return (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5


def main() -> int:
    ap = argparse.ArgumentParser(description="MDE 标定（A6，零额度）")
    ap.add_argument("--inst", default="BTC-USDT-SWAP")
    ap.add_argument("--source", default=SOURCE_OKX,
                    help="数据来源。⚠️ 换 source 就换了口径——"
                         "本库现在有 okx（799 根）与 binance_csv（17520 根）两种，"
                         "**不要混着比**")
    ap.add_argument("--bar", default="1H")
    ap.add_argument("--seg-lens", default=",".join(
        str(x) for x in DEFAULT_SEG_LENS))
    ap.add_argument("--pair", default="momentum,noop",
                    help="要标定的对照（a,b）；配对差值 = a − b")
    ap.add_argument("--target-effect", type=float, default=0.003,
                    help="想知道「要判出多大的效应需要多少段」时填的效应量"
                         "（默认 0.3% ≈ 30bp/段）")
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    banner("MDE 标定（A6，零额度）")

    store = MarketStore()
    try:
        # ⚠️ **把库里所有根都取出来**：此前只取了 `12 + K·L + 5` 根，
        # 于是 799 根的库里只用了 412 根 ⇒ **白白丢掉一半数据**，
        # 而"段数不够 ⇒ 判不出来"的结论就建立在这个浪费上。
        series = store.load_candles(args.source, args.inst, args.bar,
                                    limit=10 ** 6)
    finally:
        store.close()
    n_bars = len(getattr(series, "close", []))
    a_name, b_name = (x.strip() for x in args.pair.split(","))
    print(f"  行情 {args.source}/{args.inst} {args.bar}："
          f"库里共 **{n_bars} 根**")
    print(f"  对照：{a_name} − {b_name}（两者都免费，本脚本**不调 LLM**）")

    seg_lens = [int(x) for x in args.seg_lens.split(",") if x.strip()]
    rows: list[dict] = []
    for L in seg_lens:
        rngs = segment_ranges(n_bars, L, min_history=12)
        if len(rngs) < 2:
            print(f"\n  L={L:>3}：可用段 {len(rngs)} ⇒ 跳过（至少要 2 段）")
            continue
        facts = {}
        for nm in (a_name, b_name):
            facts[nm] = (lambda _n: lambda: TradingAgent(
                policy=make_policy(_n, seed=7), limits=RiskLimits(),
                config=AgentConfig(inst_id=args.inst, bar=args.bar,
                                   lever=3.0)))(nm)
        res = run_paired_segments(
            series, facts, seg_len=L, min_history=12, max_segs=0,
            exec_config=ExecConfig(slippage_bps=1.0),
            parallel=max(1, args.parallel),
        )
        d = res.diffs(a_name, b_name)
        v = paired_verdict(d, name_a=a_name, name_b=b_name)
        mde = min_detectable_effect(_sd(d), len(d))
        pa = power_analysis(effect=abs(sum(d) / len(d)) or 1e-9, sd=_sd(d))
        rows.append({
            "seg_len": L, "n_segs": len(rngs), "bars_used": len(rngs) * L,
            "mean_diff": sum(d) / len(d), "sd_seg": _sd(d),
            "t": v["t"], "t_crit": v["t_crit"], "verdict": v["verdict"],
            "mde": mde,
            "required_n_for_observed": pa.get("required_segs"),
            "corr": segment_correlation(res.net, a_name),
        })
        print(f"\n  ── L={L:>3}：{len(rngs)} 段（用 {len(rngs) * L} 根）")
        print(f"     配对差值均值 {rows[-1]['mean_diff'] * 1e4:>+8.2f} bp，"
              f"段级 σ {rows[-1]['sd_seg'] * 1e4:>7.2f} bp")
        print(f"     t = {v['t']:>6.2f}（df={v['df']}，临界 {v['t_crit']:.2f}）"
              f" → **{v['verdict']}**")
        print(f"     MDE（n={len(d)} 时能分辨的最小效应）"
              f" ≈ **{mde * 1e4:.1f} bp = {mde:.3%}**")

    if rows:
        print("\n" + "=" * 74)
        print("⭐ 汇总：MDE 与段长的关系（**这是「能不能判」的量化答案**）")
        print("=" * 74)
        # ⚠️⚠️ **只看 MDE 会误导**：段越短 MDE 越小，但**真实效应也越小**
        # （信号需要时间才能兑现）。所以必须同时给出「判力比」
        # `|效应| / MDE`——它才是"这段长度能不能判"的答案。
        # 本项目纪律 ⑥/⑧ 的同型错误：拿一个孤立的数当结论。
        print(f"  {'L':>4} {'K':>4} {'用根数':>7} {'效应(bp)':>9} "
              f"{'σ_段(bp)':>9} {'MDE(bp)':>8} {'判力比':>7}  判定")
        for r in rows:
            eff = abs(r["mean_diff"]) * 1e4
            ratio = (eff / (r["mde"] * 1e4)) if r["mde"] > 0 else float("nan")
            print(f"  {r['seg_len']:>4} {r['n_segs']:>4} {r['bars_used']:>7} "
                  f"{eff:>9.2f} {r['sd_seg'] * 1e4:>9.2f} "
                  f"{r['mde'] * 1e4:>8.1f} {ratio:>7.2f}  {r['verdict']}")
        print("\n  ⚠️ 判力比 < 1 表示**这个对照在本库上判不出来**；"
              "≈1 是临界；> 1.5 才算有余量。")
        best = max(rows, key=lambda r: (abs(r["mean_diff"]) / r["mde"]
                                        if r["mde"] > 0 else -1))
        print(f"  ⇒ 本库（{n_bars} 根）上判力比最高的是 **L={best['seg_len']}**"
              f"（{abs(best['mean_diff']) / best['mde']:.2f}）")
        mde_best = min(rows, key=lambda r: r["mde"])
        print(f"  ⇒ 纯 MDE 最小的是 L={mde_best['seg_len']}"
              f"（{mde_best['mde'] * 1e4:.1f} bp）"
              f"——但**要与效应一起看**，别只挑这个数。")
        print(f"  ⚠️ 注意 σ_段 若随 L 增长慢于 √L，说明差异有自相关，"
              f"`MDE ∝ L` 的推导会偏乐观。看 σ_段/√L 这一列：")
        base = rows[0]["sd_seg"] / (rows[0]["seg_len"] ** 0.5)
        for r in rows:
            k = (r["sd_seg"] / (r["seg_len"] ** 0.5)) / base if base else 0
            print(f"     L={r['seg_len']:>3}  σ_段/√L 相对首行 "
                  f"{k:>5.2f}×（≈1 = 符合 √L 假设）")
        tgt = abs(args.target_effect)
        print(f"\n  若想把 **{tgt:.2%}** 的效应判出来，在当前最优 L 下需要多少段：")
        for r in rows:
            if not (r["sd_seg"] > 0):
                continue
            # ⚠️ 外推用的 t 临界值随 df 变，这里用迭代逼近而不是硬套 1.96
            k = _segs_needed(r["sd_seg"], tgt)
            print(f"     L={r['seg_len']:>3}：≈ **{k} 段**"
                  f"（{k * r['seg_len']} 根；本库有 {n_bars} 根）")

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(
            {"config": vars(args), "n_bars": n_bars, "rows": rows},
            ensure_ascii=False, indent=2, default=str),
            encoding="utf-8")
        print(f"\n  JSON → {args.out_json}")
    return 0


def _segs_needed(sd_seg: float, effect: float) -> int:
    """要判出 ``effect`` 需要多少段。

    ⚠️⚠️ **不要再自己抄一份 t 表**——本项目元教训第 ② 条：
    **同一个量有多份实现，就一定会分叉。**
    本函数曾经自己抄了一份（只含 t 临界值、不含功效项），
    于是它与 `tw.segmented.min_detectable_effect`（固定 2.80）
    **不互为逆运算**，报出来的"需要多少段"偏小约 2 倍。
    ⇒ 现在只有一份实现，在 `tw.segmented` 里。
    """
    res = power_analysis(effect=effect, sd=sd_seg)
    return int(res.get("required_segs") or 0)


if __name__ == "__main__":
    raise SystemExit(main())
