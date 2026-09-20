"""工作线E：H4 假说验证（非对称有效、对称失效的机制解释）。

任务书：`交易世界 · 分辨力危机应对任务书.md` §4（沿用上一份任务书 §4 的设计）。

    EE.1  同步率对比（对称 vs 非对称配置）
    EE.2  四象限深度对比（检验"做市方同步补充抵消了稀疏"）
    EE.3  渐进式做市方聚集度扫描 —— **改用凹度判决，不用 k**
          （任务书 §4.0 明确的唯一改动）

⚠️⚠️ 前提核对：H4 的措辞与源码不符（必须写在交付物里）
------------------------------------------------------
H4 假设"吃单方和做市方服从**各自独立的** Hawkes 聚集过程"。
但 ``HawkesMarket._limit_activity`` 在 ``thin_scope="both"`` 下**只取一次 ρ**
（``rho = self._current_rho()``）然后对整个列表稀疏化 ⇒ **两侧由同一个 ρ 驱动**，
同步是**构造性**的，不是统计现象。

⇒ 因此 EE.1 的结果**不能单独作为 H4 的证据**，它只能说明"构造上确实同步"。
   H4 真正要问的是**同步是否真的把稀疏填平了**——那是 EE.2 的事。

⚠️ 第二个前提核对：``thin_scope="taker"`` 时做市方**完全不受 ρ 调制**，
   其活跃度序列是**常量**。常量序列在 ``x > percentile(x,80)`` 下
   **没有任何突发点**（``x > x`` 恒假）⇒ 该配置下四象限退化成
   「taker_only」与「neither」两类，"both_burst"恒为空。
   所以四象限对比**只能在对称配置下做**（与任务书 §4.4 一致）。

用法::

    python scripts/run_workstream_E.py --only EE1
    python scripts/run_workstream_E.py --only EE3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import OUT, banner, ensure_scripts_on_path, save_json  # noqa: E402
from tw.analyzer_asymmetry import (  # noqa: E402
    block_quadrant_contrast,
    burst_mask,
    measure_burst_synchrony,
    measure_local_depth_during_taker_burst,
    quadrant_masks,
)

RESULT = "workstream_E_metrics.json"
COUPLINGS = (0.0, 0.25, 0.5, 0.75, 1.0)   #: 做市方聚集度耦合（0=非对称, 1=对称）
N_SEEDS_SYNC = 3
WINDOW = 50
BURST_PCT = 80.0


def _safe_load(name: str) -> dict:
    import json
    p = OUT / name
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _make_recording_class():
    """构造一个会在 ``_limit_activity`` 处记录**分侧保留数**的市场类。

    为什么在 ``_limit_activity`` 处观察：它是"本 tick 谁能出手"的唯一出口，
    观察它**不依赖 ρ 的内部实现**——比读 ``_pending_rho`` 更硬
    （后者只是在复述实现细节）。
    """
    ensure_scripts_on_path()
    from tw.order_flow.hawkes_market import rho_to_count
    from tw.order_flow.hawkes_market import HawkesMarket

    class RecordingHawkes(HawkesMarket):
        """记录分侧活跃度；可选地把做市方的稀疏化耦合到吃单方的 ρ 上。

        ``maker_coupling``
            0.0 ⇒ 做市方完全不稀疏化（= ``thin_scope="taker"``，EA.4 的非对称配置）
            1.0 ⇒ 做市方与吃单方同 ρ（= 对称配置）
            中间值 ⇒ 连续过渡。实现方式是让做市方的有效 ρ 在 1 与 ρ 之间线性插值：
                ``rho_maker = 1 − coupling · (1 − ρ)``
            这样 **ρ 的边际分布形状被保留**（只是被压缩到 [ρ, 1]），
            所以"聚集强度"是可控地递增的，不是换了个东西。
        """

        def __init__(self, *a, maker_coupling: float = 0.0, trace=None, **kw):
            super().__init__(*a, **kw)
            self.maker_coupling = float(maker_coupling)
            self.trace = trace if trace is not None else {
                "taker_keep": [], "maker_keep": [], "taker_all": [],
                "maker_all": [], "depth": []}

        def _limit_activity(self, order):
            n_t_all = sum(1 for a in order
                          if getattr(a, "KIND", "") != "market_maker")
            n_m_all = len(order) - n_t_all
            if self.hawkes_off:
                out = order
            elif self.maker_coupling <= 0.0:
                # 完全非对称：走基类的 taker 分支（做市方全保留）
                out = super()._limit_activity(order)
            else:
                rho = self._current_rho()
                takers = [a for a in order
                          if getattr(a, "KIND", "") != "market_maker"]
                makers = [a for a in order
                          if getattr(a, "KIND", "") == "market_maker"]
                k_t = rho_to_count(rho, len(takers),
                                   clamp=self.hawkes_cfg.clamp_rho)
                rho_m = 1.0 - self.maker_coupling * (1.0 - rho)
                k_m = rho_to_count(rho_m, len(makers),
                                   clamp=self.hawkes_cfg.clamp_rho)
                out = takers[:k_t] + makers[:k_m]
                self._pending_rho = rho
                self._pending_k = len(out)
            n_t_keep = sum(1 for a in out if getattr(a, "KIND", "") != "market_maker")
            self.trace["taker_all"].append(n_t_all)
            self.trace["taker_keep"].append(n_t_keep)
            self.trace["maker_all"].append(n_m_all)
            self.trace["maker_keep"].append(len(out) - n_t_keep)
            return out

        def step(self):
            d = self._book_depth_total() if self.record_depth else float("nan")
            self.trace["depth"].append(d)
            super().step()

    return RecordingHawkes


def run_traced(maker_coupling: float, seeds: list[int], *,
               base_lambda: float, window=(6_000, 6_400),
               record_depth: bool = True) -> list[dict]:
    """按给定耦合强度跑若干场（每场一个种子），返回每场的 trace 统计。"""
    ensure_scripts_on_path()
    from tw import Population, SimConfig
    import run_workstream_A as A
    from tw.order_flow.hawkes import HawkesConfig

    RCls = _make_recording_class()
    spec = A.ea4_market_spec()
    hcfg = HawkesConfig(base_lambda=base_lambda, branching=0.6, beta=A.BETA)

    runs = []
    for sd in seeds:
        trace: dict = {"taker_keep": [], "maker_keep": [], "taker_all": [],
                       "maker_all": [], "depth": []}
        pop = Population.from_shares(spec["n_agents"], spec["mix"])
        cfg = SimConfig(seed=sd, n_ticks=window[1], population=pop, **spec["sim_kw"])
        m = RCls(cfg, hawkes_config=hcfg, maker_coupling=maker_coupling,
                 record_depth=record_depth, trace=trace)
        m.run(window[1])
        lo, hi = window
        sl = slice(lo, min(hi, len(trace["taker_keep"])))
        runs.append({
            "seed": sd,
            "taker_keep": np.asarray(trace["taker_keep"][sl], dtype=float),
            "maker_keep": np.asarray(trace["maker_keep"][sl], dtype=float),
            "depth": np.asarray(trace["depth"][sl], dtype=float),
        })
    return runs


def _lambda_bar(seeds) -> float:
    ensure_scripts_on_path()
    import run_workstream_A as A
    from run_stage3 import WARMUP, HORIZON
    from tw.order_flow.hawkes_market import calibrate_base_lambda
    kw = A.ea4_calibration_kwargs(seeds, (WARMUP, WARMUP + HORIZON),
                                  A.ea4_market_spec())
    return float(calibrate_base_lambda(**kw))


# ======================================================================
def ee1_and_ee2(seeds=None) -> dict:
    """EE.1 同步率对比 + EE.2 四象限深度（共用同一批运行，省一半时间）。"""
    print("\n【EE.1/EE.2】同步率与四象限深度")
    ensure_scripts_on_path()
    import run_workstream_A as A
    seeds = list(seeds or A.SEEDS[:N_SEEDS_SYNC])
    lam = _lambda_bar(seeds)
    print(f"  λ̄={lam:.3f}  种子={len(seeds)}")

    out: dict = {"base_lambda": lam, "window": WINDOW, "burst_pct": BURST_PCT,
                 "configs": {}}
    for tag, coupling in (("symmetric_both", 1.0), ("asymmetric_taker", 0.0)):
        runs = run_traced(coupling, seeds, base_lambda=lam)
        syncs, quads, contrasts = [], [], []
        for r in runs:
            s = measure_burst_synchrony(r["taker_keep"], r["maker_keep"],
                                        pct=BURST_PCT, window=WINDOW)
            syncs.append({"mean": s.mean_sync, "median": s.median_sync,
                          "n_taker_burst": s.n_taker_burst,
                          "n_maker_burst": s.n_maker_burst,
                          "taker_thr": s.taker_threshold,
                          "maker_thr": s.maker_threshold,
                          # 条件概率才是 H4 要的量（联合率会被两侧频率压住）
                          "p_maker_given_taker": s.p_maker_given_taker,
                          "p_taker_given_maker": s.p_taker_given_maker,
                          "lift": s.lift})
            t_mask = burst_mask(r["taker_keep"], BURST_PCT)
            m_mask = burst_mask(r["maker_keep"], BURST_PCT)
            quads.append(measure_local_depth_during_taker_burst(
                r["depth"], t_mask, m_mask))
            contrasts.append(block_quadrant_contrast(
                r["depth"], t_mask, m_mask, block=200))
        sync_means = [s["mean"] for s in syncs]
        out["configs"][tag] = {
            "coupling": coupling,
            "sync": syncs,
            "sync_mean": float(np.mean(sync_means)) if sync_means else float("nan"),
            "sync_sem": (float(np.std(sync_means, ddof=1) / np.sqrt(len(sync_means)))
                         if len(sync_means) > 1 else float("nan")),
            "quadrants": quads,
            "contrast": contrasts,
        }
        # 打印（四象限只有对称配置有 both_burst，见模块文档）
        q0 = quads[0] if quads else {}
        pr = [s["p_maker_given_taker"] for s in syncs if s["p_maker_given_taker"] == s["p_maker_given_taker"]]
        lf = [s["lift"] for s in syncs if s["lift"] == s["lift"]]
        out["configs"][tag]["p_maker_given_taker_mean"] = float(np.mean(pr)) if pr else float("nan")
        out["configs"][tag]["lift_mean"] = float(np.mean(lf)) if lf else float("nan")
        print(f"\n  [{tag}] coupling={coupling}"
              f"  联合同步率={out['configs'][tag]['sync_mean']:.4f}"
              f"  P(做市突发|吃单突发)={out['configs'][tag]['p_maker_given_taker_mean']:.4f}"
              f"  lift={out['configs'][tag]['lift_mean']:.2f}"
              f"\n         突发点数 taker/maker={syncs[0]['n_taker_burst']}/"
              f"{syncs[0]['n_maker_burst']}")
        for name in ("both_burst", "taker_only", "maker_only", "neither"):
            d = q0.get(name) or {}
            print(f"    {name:<12} n={d.get('n', 0):>5}  深度均值={d.get('mean', float('nan')):.2f}")
        c = contrasts[0] if contrasts else {}
        if c.get("reason"):
            print(f"    ⚠️ 块级对比：{c['reason']}")
        else:
            print(f"    块级对比 taker_only vs both_burst："
                  f"差={c.get('diff', float('nan')):+.2f}  t={c.get('t', float('nan')):+.2f}"
                  f"  p={c.get('p', float('nan')):.4f}"
                  f"  H4方向={'支持' if c.get('h4_direction_supported') else '不支持'}")

    # ---- EE.1 的构造性说明（必须写进产物，避免被当成发现）----
    out["ee1_caveat"] = (
        "⚠️ 对称配置下两侧由**同一个 ρ** 驱动（HawkesMarket._limit_activity 只取一次 "
        "_current_rho()），所以高同步率是**构造性**的、不是统计发现。"
        "非对称配置下做市方活跃度是**常量**，常量序列在 x > percentile(x,80) 下"
        "没有任何突发点 ⇒ 四象限退化为 taker_only / neither 两类。"
        "因此 EE.1 只能证明「构造上确实同步」，H4 必须靠 EE.2 的四象限深度来检验。")
    return out


def ee3_progressive(couplings=None, seeds=None, shock_levels=None) -> dict:
    """EE.3：渐进式做市方聚集度扫描 —— 用**凹度判决**替代 k。

    对每个耦合强度，在它自己的配置上跑清算冲击实验，算**全部档位对**的
    局部弹性与三分类判决（而不是报一个 k）。
    """
    print("\n【EE.3】渐进式做市方聚集度扫描（凹度判决）")
    ensure_scripts_on_path()
    import run_workstream_A as A
    from run_stage3 import HORIZON, WARMUP, make_controls, paired_impact
    from run_workstream_I import all_pairs, verdict_counts
    from tw.order_flow.hawkes import HawkesConfig

    couplings = list(couplings or COUPLINGS)
    seeds = list(seeds or A.SEEDS[:N_SEEDS_SYNC])
    shock_levels = list(shock_levels or [0.001, 0.0025, 0.005, 0.01])
    RCls = _make_recording_class()
    spec = A.ea4_market_spec()
    mix = spec["mix"]
    lam = _lambda_bar(seeds)
    print(f"  λ̄={lam:.3f}  耦合档={couplings}")

    out: dict = {"base_lambda": lam, "couplings": couplings,
                 "seeds": seeds, "shock_levels": shock_levels, "arms": {}}
    for cp in couplings:
        hcfg = HawkesConfig(base_lambda=lam, branching=0.6, beta=A.BETA)

        def fn(sim, cp=cp, hcfg=hcfg):
            return RCls(sim, hawkes_config=hcfg, maker_coupling=cp)

        ctrl = make_controls(mix, seeds, HORIZON, factory=fn)
        rows = []
        for size in shock_levels:
            r = paired_impact(f"cp={cp}|{size:.4f}", mix, size, 20, seeds,
                              ctrl, HORIZON, factory=fn)
            rows.append({"shock_frac": float(size),
                         "slippage_bp": float(r["slippage_bp"]),
                         "slippage_sem_bp": float(r["slippage_sem_bp"]),
                         "slippage_t": r.get("slippage_t"),
                         "fill_ratio": float(r["fill_ratio"]),
                         "n_seeds": len(seeds)})
        pairs = all_pairs(rows)
        out["arms"][str(cp)] = {"rows": rows, "pairs": pairs,
                                "verdicts": verdict_counts(pairs)}
        v = out["arms"][str(cp)]["verdicts"]
        print(f"  coupling={cp:<5} " + "  ".join(
            f"{r['shock_frac']}:{r['slippage_bp']:>8.3f}" for r in rows))
        print(f"            判决：凹={v.get('显著凹', 0)} "
              f"线性/更差={v.get('显著线性或更差', 0)} "
              f"不可判定={v.get('不可判定', 0)}")
    return out


# ======================================================================
def main() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="EE1 / EE3")
    args = ap.parse_args()
    banner("工作线E  H4 假说验证（非对称有效、对称失效）")
    t0 = time.time()
    only = (args.only or "").upper()
    out: dict = {"stage": "E"}
    if not only or only == "EE1":
        e = ee1_and_ee2()
        out["ee1_sync"] = e
        out["ee2_quadrant"] = {"note": "与 EE.1 共用同一批运行，数据在 ee1_sync.configs.*.quadrants",
                               "contrast": {k: v.get("contrast")
                                            for k, v in e["configs"].items()}}
    if not only or only == "EE3":
        out["ee3_progressive"] = ee3_progressive()
    out["elapsed_sec"] = time.time() - t0
    prev = _safe_load(RESULT)
    prev.update(out)
    save_json(prev, RESULT)
    print(f"\n  产物：out/{RESULT}（{out['elapsed_sec']:.0f}s）")
    return prev


if __name__ == "__main__":
    main()
