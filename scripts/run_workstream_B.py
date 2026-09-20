"""工作线B：把元订单覆盖面扩展到零智能 + 基本面派。

B.0 目标（任务书）
------------------
阶段6 的元订单机制**只给了 40% 的图表派**，而二期的诚实边界里写着
"真实市场里所有大额执行者都拆单，覆盖面不足会低估效应"。
本线把机制扩展到零智能与基本面派，检验 γ 与 k 是否进一步改善。

⚠️ 前提核对（任务书 vs 源码）——**两处必须改**
----------------------------------------------
1. 任务书给的是 ``on_tick(market_state)`` 的覆写。
   **``Agent`` 没有 ``on_tick``**，接口是 ``decide(state)``。
   照抄的后果是**机制静默失效**（方法永不被调用、不报错、效果为零）。
2. 任务书以为"锁定方向 / 执行期跳过基础逻辑"需要自己写。
   **``MetaOrderMixin.decide`` 已经做了**（三种情形），
   所以正确实现是**子类化**：``class MetaZeroIntelligence(MetaOrderMixin, ZeroIntelligence)``。
   见 ``run_stage6.META_CLASS_OF``。

⇒ 本线**没有**改动 ``tw/agents/zero_intel.py`` / ``fundamentalist.py``：
   那是库层，而这两个子类是实验装配件（与 ``MetaChartist`` 同一层次，
   它就一直住在 ``run_stage6.py`` 里）。放进库里反而会让
   "实验用的机制组合"变成库的对外契约。

依赖工作线A 的结论（任务书 B.0）
--------------------------------
A 线已确认：**Hawkes 是真实机制，但 E8.2 宣称的幅度被口径错误放大约一倍**
（真正修正标尺后 k 从 1.395 到 0.947，而不是 0.481）。
⇒ 本线**不叠加 Hawkes**（EB.3 因此不做），理由是：
   叠加必须同时处理两个口径修正，会把两条线的问题混在一起，
   而任务书也说了"如果 A 发现主要是离散化假象就不叠加"。
   这里的判断是"幅度被高估、但机制为真"，所以更稳妥的做法是
   **先把 B 线自己的效应量测准**，在合并阶段再统一处理。

用法::

    python scripts/run_workstream_B.py             # EB.1 + EB.2 + EB.4
    python scripts/run_workstream_B.py --only EB1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import OUT, banner, load_json, save_json  # noqa: E402
from run_stage6 import (  # noqa: E402
    FLOW_SENS,
    FLOW_WINDOW,
    MAX_LAG,
    META_CFG,
    META_SHARE,
    MM_MIX,
    N_AGENTS,
    N_TICKS,
    WARMUP,
    Stage6Market,
)
from tw import Population, SimConfig  # noqa: E402
from tw.analyzer_longmemory import order_sign_acf, order_signs  # noqa: E402
from tw.impact import (  # noqa: E402
    cumulative_depth,
    depth_profile_gamma,
    fit_power_law,
    unsaturated,
)
from tw.order_flow.meta_order import MetaOrderConfig  # noqa: E402

SEED0 = 20260917
SEEDS = [SEED0 + 7 * i for i in range(3)]
#: 各类主体的激活比例。图表派 0.4 沿用阶段6 的设定，便于与 E6.x 对照。
SHARE_CHART = META_SHARE
SHARE_ZI = 0.5
SHARE_FUND = 0.5
ARMS = (
    ("仅图表派(阶段6基线)", {"chartist": SHARE_CHART}),
    ("+零智能", {"chartist": SHARE_CHART, "zero_intel": SHARE_ZI}),
    ("+基本面派", {"chartist": SHARE_CHART, "fundamentalist": SHARE_FUND}),
    ("三者全部", {"chartist": SHARE_CHART, "zero_intel": SHARE_ZI,
                  "fundamentalist": SHARE_FUND}),
)


def f(v, fmt: str = ".3f", dash: str = "—") -> str:
    if not isinstance(v, (int, float)) or v != v:
        return dash
    return format(v, fmt)


# ----------------------------------------------------------------------
def coverage_factory(shares: dict[str, float], *, adapt_mm: bool = True,
                     p_meta_start: float | None = None):
    """造 ``SimConfig -> Market`` 的工厂，按类型激活元订单。

    ``p_meta_start`` 默认取 ``META_CFG``（与阶段6 同一强度），
    只有 EB.4 的参数扫描会覆盖它——这样 EB.1/EB.2 的可比性不被污染。
    """
    cfg = MetaOrderConfig(**{**META_CFG,
                             **({"p_meta_start": p_meta_start}
                                if p_meta_start is not None else {})})
    return lambda sim: Stage6Market(
        sim, meta_cfg=cfg, meta_shares=dict(shares), adapt_mm=adapt_mm,
        flow_window=FLOW_WINDOW, flow_sensitivity=FLOW_SENS)


def meta_census(m) -> dict:
    """数一数各类主体里有多少真的在拆单——"激活了"必须能被验证。

    只看配置传进去了不算数：``MetaOrderMixin`` 收不到 config 时会用默认值，
    而默认值是**开着**的（本项目踩过这个坑）。
    所以这里读的是主体**实际启动过的元订单数**。
    """
    out: dict[str, dict] = {}
    for a in m.agents:
        k = a.KIND
        d = out.setdefault(k, {"n": 0, "n_active": 0, "n_meta": 0, "n_children": 0})
        d["n"] += 1
        p = float(getattr(getattr(a, "meta_config", None), "p_meta_start", 0.0) or 0.0)
        if p > 0:
            d["n_active"] += 1
        d["n_meta"] += int(getattr(a, "n_meta_started", 0))
        d["n_children"] += int(getattr(a, "n_children", 0))
    return out


def gamma_and_acf(shares: dict[str, float], seeds=SEEDS,
                  n_ticks: int = N_TICKS, *,
                  p_meta_start: float | None = None) -> dict:
    """跑几场，返回深度剖面指数 γ 与订单符号 ΣACF。

    ⚠️ ``p_meta_start`` **必须显式声明并真的用上**。
    第一版这里是 ``**fkw`` 并把它**丢掉**了——于是 EB.4 扫的三个档位
    产出了**逐位相同**的 γ / ΣACF / 元订单数，"参数敏感性"变成一场空实验。
    症状极具代表性：**三档数字一模一样**。任何"扫参数"的结果如果对参数
    完全不敏感，第一件该怀疑的事就是"参数有没有真的传进去"。
    """
    cfg_kw = dict(META_CFG)
    if p_meta_start is not None:
        cfg_kw["p_meta_start"] = float(p_meta_start)
    gs, acfs, runs = [], [], []
    for sd in seeds:
        m = Stage6Market(
            SimConfig(seed=sd, n_ticks=n_ticks,
                      population=Population.from_shares(N_AGENTS, MM_MIX)),
            meta_cfg=MetaOrderConfig(**cfg_kw), meta_shares=dict(shares),
            adapt_mm=True, flow_window=FLOW_WINDOW, flow_sensitivity=FLOW_SENS)
        m.run(n_ticks)
        mid = m.current_mid()
        levels = m.book.depth("buy", n=500)
        dist = np.array([p for p, _, _ in levels], dtype=float)
        qty = np.array([q for _, q, _ in levels], dtype=float)
        d_grid, cum = cumulative_depth(np.abs(dist / mid - 1.0) * 1e4, qty,
                                       max_bp=400.0, n_bins=40)
        gs.append(depth_profile_gamma(d_grid, cum, lo=20.0))
        win = [t for t in m.log.trades if WARMUP <= t.tick < n_ticks]
        acf = order_sign_acf(order_signs(win), max_lag=MAX_LAG)
        acfs.append(acf)
        runs.append({"seed": sd, "gamma": float(gs[-1]),
                     "n_orders": int(len(order_signs(win))),
                     "acf_sum": float(np.nansum(acf)) if acf.size else float("nan"),
                     "acf": acf.tolist(),
                     "census": meta_census(m),
                     "n_meta": int(sum(getattr(a, "n_meta_started", 0)
                                       for a in m.agents))})
    return {"gamma_mean": float(np.mean(gs)), "runs": runs,
            "acf_sum_mean": float(np.mean([r["acf_sum"] for r in runs]))}


# ----------------------------------------------------------------------
# EB.1  覆盖面消融（k 与 γ）
# ----------------------------------------------------------------------
def eb1_coverage_ablation(seeds=SEEDS, shock_levels=None) -> dict:
    from run_stage3 import HORIZON, MM_MIX as W3_MIX, make_controls, paired_impact

    shock_levels = shock_levels or [0.001, 0.0025, 0.005, 0.01]
    print("\n【EB.1】覆盖面消融：逐步加入每一类主体的拆单行为")
    print("  复用语阶段3 的同种子配对清算实验；四档配置的**其他部分完全相同**")
    print(f"    {'配置':<20}{'k':>8}{'|k−0.5|':>9}{'η(元订单数)':>12}"
          f"{'未饱和档':>10}")
    arms = {}
    for tag, shares in ARMS:
        fn = coverage_factory(shares)
        ctrl = make_controls(W3_MIX, seeds, HORIZON, factory=fn)
        rows = []
        for f_ in shock_levels:
            rows.append(paired_impact(f"{tag}|{f_:.4f}", W3_MIX, f_, 20, seeds,
                                      ctrl, HORIZON, factory=fn))
        un = unsaturated(rows)
        fit = fit_power_law(un, "slippage_bp")
        # 元订单规模：单独跑一场拿 census（清算实验的 factory 内部自建市场）
        g = gamma_and_acf(shares, seeds=seeds[:1], n_ticks=N_TICKS)
        n_meta = float(np.mean([r["n_meta"] for r in g["runs"]]))
        arms[tag] = {"shares": shares, "k": fit.exponent,
                     "closeness": fit.closeness_to_sqrt(), "r2": fit.r_squared,
                     "n_unsaturated": len(un), "n_levels": len(rows),
                     "rows": rows, "n_meta_mean": n_meta,
                     "census": g["runs"][0]["census"]}
        print(f"    {tag:<20}{f(fit.exponent, '.3f'):>8}"
              f"{f(fit.closeness_to_sqrt(), '.3f'):>9}{n_meta:>12.0f}"
              f"{len(un):>10}")
    base = arms[ARMS[0][0]]
    print(f"\n  【相对阶段6 基线的改善量】")
    ranked = []
    for tag, a in arms.items():
        dk = a["k"] - base["k"]
        ranked.append((tag, dk))
        if tag == ARMS[0][0]:
            continue
        print(f"    {tag:<16} k {base['k']:.3f} → {a['k']:.3f}（{dk:+.3f}）"
              f"  元订单数 {a['n_meta_mean']:.0f} vs {base['n_meta_mean']:.0f}")
    best = min(((t, d) for t, d in ranked if t != ARMS[0][0]),
               key=lambda kv: kv[1], default=None)
    if best:
        print(f"  → 贡献最大的一类：**{best[0]}**（Δk = {best[1]:+.3f}）")
    return {"base_k": base["k"], "arms": arms, "best": best[0] if best else None}


# ----------------------------------------------------------------------
# EB.2  长记忆的"味道"差异
# ----------------------------------------------------------------------
def _acf_shape(acf: np.ndarray) -> dict:
    """把 ACF 曲线压成几个**可比较**的数，而不是"看起来不一样"。

    · ``half_life``   衰减到首个值的一半所需的 lag（趋势型越长）
    · ``min_acf``     1..MAX_LAG 内的最小值（负值 ⇒ 出现**反转**迹象）
    · ``neg_from``    第一次变负的 lag（nan = 全程为正）
    · ``tail_mean``   后 1/4 段的均值（趋势型的尾巴仍显著为正）
    """
    a = np.asarray(acf, dtype=float)
    if a.size == 0:
        return {"half_life": float("nan"), "min_acf": float("nan"),
                "neg_from": float("nan"), "tail_mean": float("nan")}
    a0 = a[0]
    half = float("nan")
    if a0 > 0:
        idx = np.where(a < a0 / 2)[0]
        half = float(idx[0] + 1) if idx.size else float("inf")
    neg = np.where(a < 0)[0]
    return {"half_life": half, "min_acf": float(a.min()),
            "neg_from": float(neg[0] + 1) if neg.size else float("inf"),
            "tail_mean": float(a[-(a.size // 4):].mean())}


def eb2_flavour(seeds=SEEDS) -> dict:
    print("\n【EB.2】长记忆的「味道」：图表派 vs 基本面派")
    print("  假说：基本面派的方向锚定在**启动那一刻的 premium**，而 premium 本身")
    print("        是均值回归的 ⇒ 它的订单符号长记忆应当**更快衰减、更易出现反转**；")
    print("        图表派锚定动量 ⇒ 更接近**趋势延续**（衰减慢、尾巴仍为正）。")
    arms = {
        "仅图表派元订单": {"chartist": SHARE_CHART},
        "仅基本面派元订单": {"fundamentalist": SHARE_FUND},
        "仅零智能元订单": {"zero_intel": SHARE_ZI},
    }
    out: dict = {}
    print(f"    {'配置':<20}{'ACF(1)':>9}{'半衰期':>9}{'最小值':>10}"
          f"{'首次转负':>10}{'尾段均值':>10}{'ΣACF':>9}")
    for tag, shares in arms.items():
        r = gamma_and_acf(shares, seeds=seeds)
        acfs = np.array([run["acf"] for run in r["runs"]], dtype=float)
        mean_acf = np.nanmean(acfs, axis=0)
        sh = _acf_shape(mean_acf)
        out[tag] = {"mean_acf": mean_acf.tolist(), "shape": sh,
                    "acf_sum_mean": r["acf_sum_mean"],
                    "gamma_mean": r["gamma_mean"],
                    "n_orders_mean": float(np.mean([q["n_orders"]
                                                    for q in r["runs"]]))}
        print(f"    {tag:<20}{f(mean_acf[0], '+.4f'):>9}"
              f"{f(sh['half_life'], '.0f'):>9}{f(sh['min_acf'], '+.4f'):>10}"
              f"{f(sh['neg_from'], '.0f'):>10}{f(sh['tail_mean'], '+.4f'):>10}"
              f"{f(r['acf_sum_mean'], '+.2f'):>9}")
    print("\n  【判读】")
    ch = out["仅图表派元订单"]["shape"]
    fu = out["仅基本面派元订单"]["shape"]
    print(f"    图表派：半衰期 {f(ch['half_life'], '.0f')}，尾段 {f(ch['tail_mean'], '+.4f')}")
    print(f"    基本面：半衰期 {f(fu['half_life'], '.0f')}，尾段 {f(fu['tail_mean'], '+.4f')}")
    faster = (isinstance(fu["half_life"], (int, float))
              and isinstance(ch["half_life"], (int, float))
              and fu["half_life"] < ch["half_life"])
    print(f"    → 基本面派衰减更快：{'是' if faster else '否'}"
          f"（假说{'成立' if faster else '不成立'}）")
    return out


# ----------------------------------------------------------------------
# EB.4  新参数对 k 的边际影响（只看 γ 与 ΣACF，不跑清算）
# ----------------------------------------------------------------------
def eb4_param_sensitivity(seeds=SEEDS) -> dict:
    print("\n【EB.4】参数敏感性：新增两类主体的 p_meta_start 各扫 3 档")
    print("  口径与 E6.5 一致：**只看深度剖面 γ 与订单符号 ΣACF**")
    print("  （每档都跑清算要 24 场 × 2 组，成本爆炸——这是刻意的取舍）")
    print(f"    {'kind':<16}{'p_meta_start':>14}{'γ':>10}{'ΣACF':>10}{'元订单数':>10}")
    rows = []
    for kind, share_key in (("zero_intel", "zero_intel"),
                            ("fundamentalist", "fundamentalist")):
        for p in (0.005, 0.02, 0.08):
            shares = {"chartist": SHARE_CHART, share_key: SHARE_FUND}
            r = gamma_and_acf(shares, seeds=seeds, p_meta_start=p)
            n_meta = float(np.mean([q["n_meta"] for q in r["runs"]]))
            rows.append({"kind": kind, "p_meta_start": p,
                         "gamma": r["gamma_mean"], "acf_sum": r["acf_sum_mean"],
                         "n_meta": n_meta})
            print(f"    {kind:<16}{p:>14.3f}{f(r['gamma_mean'], '.4f'):>10}"
                  f"{f(r['acf_sum_mean'], '+.2f'):>10}{n_meta:>10.0f}")
    # 边际影响：同 kind 内 p 的变化范围
    marg = {}
    for kind in ("zero_intel", "fundamentalist"):
        sub = [r for r in rows if r["kind"] == kind]
        marg[kind] = {"gamma_range": float(max(r["gamma"] for r in sub)
                                           - min(r["gamma"] for r in sub)),
                      "acf_range": float(max(r["acf_sum"] for r in sub)
                                         - min(r["acf_sum"] for r in sub))}
    print("\n  【边际影响（同 kind 内最大−最小）】")
    for k, v in marg.items():
        print(f"    {k:<16} γ 摆幅 {v['gamma_range']:.4f}   ΣACF 摆幅 {v['acf_range']:.2f}")
    more = max(marg, key=lambda k: marg[k]["acf_range"])
    print(f"  → 对长记忆影响更大的一类：**{more}**"
          f"（决定后续校准优先级）")
    return {"rows": rows, "marginal": marg, "more_sensitive": more}


# ----------------------------------------------------------------------
def main() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="EB1 / EB2 / EB4")
    args = ap.parse_args()
    banner("工作线B  元订单覆盖面扩展")
    t0 = time.time()
    out: dict = {"stage": "B", "config": {
        "seeds": SEEDS, "arms": [t for t, _ in ARMS],
        "share_chart": SHARE_CHART, "share_zi": SHARE_ZI, "share_fund": SHARE_FUND,
        "meta_cfg": {k: v for k, v in META_CFG.items()}}}
    only = (args.only or "").upper()
    if not only or only == "EB1":
        out["eb1"] = eb1_coverage_ablation()
    if not only or only == "EB2":
        out["eb2"] = eb2_flavour()
    if not only or only == "EB4":
        out["eb4"] = eb4_param_sensitivity()
    out["elapsed_sec"] = time.time() - t0
    merged = load_json("workstream_B_metrics.json") if (OUT / "workstream_B_metrics.json").exists() else {}
    merged.update(out)
    merged["elapsed_sec"] = out["elapsed_sec"]
    save_json(merged, "workstream_B_metrics.json")
    print(f"\n  总用时 {out['elapsed_sec']:.0f}s → out/workstream_B_metrics.json")
    return out


if __name__ == "__main__":
    main()
