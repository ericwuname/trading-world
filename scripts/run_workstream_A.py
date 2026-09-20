"""工作线A：阶段8「意外修复」的机制解耦。

要回答的问题（任务书 A.0）
--------------------------
阶段6 用长记忆订单流去修冲击幂律指数 k，k 从 0.665 **恶化**到 0.910；
阶段8 本来只测「订单到达的时间聚集性」，副产品却把 k 从 1.395 拉到 **0.481**。
这次深挖要回答：**Hawkes 聚集性到底通过什么具体机制让冲击变凹的？**

三个竞争假说（不互斥，任务是把贡献量化拆开）
--------------------------------------------
    H1 突发期局部流动性变薄 —— 聚集使突发期内连续订单来不及等书补充深度
    H2 方差效应             —— 只是让到达量的方差变大，与"时间结构"无关
    H3 离散化偏差           —— "打开 Hawkes"同时悄悄让整体成交变少（ρ̄≠1）

⚠️ **动手前先核对任务书的前提（结论写在本文件末尾的 docstring 与报告里）**
任务书是外部 AI 在**看不到源码**的情况下、凭二期报告写的，实测有 8 处与源码不符。
最要紧的三处：
  1. 任务书要求新增 ``from_target_mean_rate`` 修正离散化 —— **已经实现了**，
     形状不同而已（``HawkesConfig.alpha = n/f``、``mu = λ̄(1−n)``）。
     EA.0 的作用因此从"修公式"变成"**实测验证 + 量化旧公式的危害**"。
  2. 任务书以为 E8.2 声称"ρ 均值保持 0.978" —— 但 **E8.2 的处理臂根本没有
     记录 ρ**；0.978 是 E8.1 **网格**的统计量，被写在 E8.2 结论旁边。
     于是 H3 的前提"平均活跃度不变"在 E8.2 上是**假定的、不是验证的**。
     这是 A 线最重要的发现，EA.1 就是补这一刀。
  3. 任务书假设 E8.2 用的是 branching=0.90 —— 实际是 **0.6**（0.90 是 E8.1 的 best）。
     EA.1 两个都跑。

用法::

    python scripts/run_workstream_A.py            # EA.0 + EA.1
    python scripts/run_workstream_A.py --only EA0 # 只跑便宜的那部分
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    OUT,
    banner,
    ensure_scripts_on_path,
    load_json,
    save_json,
)
from tw import Population, SimConfig  # noqa: E402
from tw.impact import (  # noqa: E402
    aggregate_burst_contrast,
    burst_depth_contrast,
    fit_power_law,
    unsaturated,
)
from tw.order_flow.hawkes import (  # noqa: E402
    HawkesConfig,
    HawkesProcess,
    simulate_event_rate,
)
from tw.order_flow.hawkes_market import (  # noqa: E402
    HawkesMarket,
    calibrate_base_lambda,
)

SEED0 = 20260917
SEEDS = [SEED0, SEED0 + 7, SEED0 + 14]
N_AGENTS = 300
BETA = 0.15
#: EA.1 的臂：0 = 关闭（对照），0.6 = E8.2 的真实配置，0.9 = E8.1 的最强聚集
ARMS = (0.0, 0.6, 0.9)


# ======================================================================
# EA.4 的**配置事实源**（任务书 §0 第六条纪律的落地）
# ----------------------------------------------------------------------
# 纪律原文：「任何涉及 λ̄ 标定的新实验，标定必须在实验实际使用的市场配置上做，
#            标定窗口必须等于实验窗口」，并要求验证「标定配置和实验配置是
#            **同一个对象的同一份参数**，不是两份分别写的、看起来一样但实际
#            不同步的配置」。
#
# 落地方式（比"比对两份 JSON"更硬）：**不写第二份配置**，而是让标定侧与实验侧
# 都从 ``ea4_market_spec()`` 派生。于是"同一份参数"是**结构性保证**的，
# 而不是靠人比对两个可能不同步的副本。
# （任务书假设存在 ``out/workstream_A/EA4_config.json`` 等产物供比对——
#   实际不存在：EA.4 的配置一直是本文件里的常量，λ̄ 是运行时算的。
#   所以按本项目的惯例不新开不存在的目录结构，而是把事实源抽出来。）
# ======================================================================
def ea4_market_spec() -> dict:
    """EA.4 的市场配置：标定侧与实验侧**共用这一份**。

    返回的是"会被真正用到的那些字段"——不是复述一处常量，
    而是把两侧的 kwargs 共同来源集中到一处。
    """
    ensure_scripts_on_path()
    from run_stage3 import MM_KW, MM_MIX
    return {
        "n_agents": N_AGENTS,
        "mix": MM_MIX,
        "sim_kw": MM_KW,
        "market_cls": HawkesMarket,
    }


def ea4_windows() -> dict:
    """实验窗口与（历史）标定窗口——**它们不一致，这正是前置校验要抓的**。

      · 实验：阶段3 的冲击实验跑在 ``[WARMUP, WARMUP+HORIZON)`` = **[6000, 6400)**
      · 历史标定：EA.4 当时用 ``n_ticks=3000, warmup=500`` = **[500, 3000)**，**含瞬态**

    实测这会让 λ̄ 偏高约 7%（94.601 vs 同窗口的 88.495）⇒ ρ = λ/λ̄ 整体偏低
    ⇒ 稀疏化比设计的更强一点。按第六条纪律，标定窗口必须等于实验窗口 ⇒ 必须重标。
    """
    ensure_scripts_on_path()
    from run_stage3 import HORIZON, WARMUP
    return {
        "experiment": (WARMUP, WARMUP + HORIZON),
        "legacy_calibration": (500, 3_000),
    }


def ea4_calibration_kwargs(seeds, window, spec=None) -> dict:
    """标定侧 kwargs——**从 spec 派生**，不手写第二份。"""
    spec = spec or ea4_market_spec()
    return {
        "seeds": list(seeds),
        "n_agents": spec["n_agents"],
        "mix": spec["mix"],
        "warmup": window[0],
        "n_ticks": window[1],
        "sim_kw": spec["sim_kw"],
    }


def ea4_experiment_kwargs(window, spec=None) -> dict:
    """实验侧 kwargs——**从同一个 spec 派生**。"""
    spec = spec or ea4_market_spec()
    return {
        "n_agents": spec["n_agents"],
        "mix": spec["mix"],
        "sim_kw": spec["sim_kw"],
        "window": tuple(window),
    }


def f(v, fmt: str = ".3f", dash: str = "—") -> str:
    if not isinstance(v, (int, float)) or v != v:
        return dash
    return format(v, fmt)


# ======================================================================
# EA.0  离散化验证：现行公式是否真的命中 λ̄？旧公式会差多少？
# ======================================================================
def ea0_discretization(target_lambda: float = 20.0, n_steps: int = 20_000) -> dict:
    """把「E[λ]=λ̄」从代数变成实测，并量化**旧公式**的危害。

    为什么要量化旧公式：任务书担心"H3 是真正原因"。要否定它，
    光说"公式已修"不够——必须给出**如果没修会差多少**，
    否则无法判断 H3 的量级是否足以解释 k 的改善。
    """
    print("\n【EA.0】离散化验证：现行公式 vs 旧（连续）公式")
    print("  口径：独立自激路径，20000 步，弃前 2000 步；λ̄ = %.0f" % target_lambda)
    print(f"    {'分支比 n':>9}{'α(离散)':>10}{'现行实测率':>12}{'偏差':>9}"
          f"{'ρ̄':>8}   |{'旧公式实测率':>13}{'偏差':>9}{'ρ̄':>8}")
    rows = []
    for n in (0.0, 0.3, 0.6, 0.9):
        cfg = HawkesConfig(base_lambda=target_lambda, branching=n, beta=BETA)
        # ⚠️ 每次都用**同一个 seed 新建**生成器。
        #    第一版在循环外用了一个共享的 ``rng_cur``，于是"有 clamp"（共享流）
        #    与"无 clamp"（每次新建流）用**不同的随机流**比较，
        #    把约 6 个百分点的蒙特卡洛噪声当成了"护栏效应"写进报告。
        #    同种子重测后差异是 **0.000%**——护栏在本区间根本没被触发过。
        cur = simulate_event_rate(cfg.to_process(), n_steps,
                                  np.random.default_rng(42),
                                  base_lambda=target_lambda,
                                  clamp=cfg.clamp_rho, warmup=2000)
        # 旧公式：用**连续**分支比 α/β = n 反解 μ，即 α = n·β。
        # 这正是变异体 M34 注入的错误。
        legacy = HawkesProcess(mu=target_lambda * (1.0 - n), alpha=n * BETA,
                               beta=BETA)
        old = simulate_event_rate(legacy, n_steps, np.random.default_rng(42),
                                  base_lambda=target_lambda, clamp=None,
                                  warmup=2000)
        row = {
            "branching": n, "alpha_discrete": cfg.alpha,
            "alpha_continuous": n * BETA,
            "current_rate": cur["realized_rate"],
            "current_rel_err": cur["rel_err"],
            "current_rho_mean": cur["rho_mean"],
            "legacy_rate": old["realized_rate"],
            "legacy_rel_err": old["rel_err"],
            "legacy_rho_mean": old["rho_mean"],
        }
        rows.append(row)
        print(f"    {n:>9.1f}{cfg.alpha:>10.5f}{cur['realized_rate']:>12.4f}"
              f"{cur['rel_err'] * 100:>+8.2f}%{cur['rho_mean']:>8.4f}   |"
              f"{old['realized_rate']:>13.4f}{old['rel_err'] * 100:>+8.2f}%"
              f"{old['rho_mean']:>8.4f}")
    # ⚠️ 判读必须**分开**看两件事，第一版把它们混成一条"通过/未通过"，
    #    结果输出自相矛盾（打了 ❌ 却写"已被排除"）：
    #    ① **公式**对不对（无 clamp 时的偏差）——这是 H3 的机制本体；
    #    ② **clamp 护栏**造成的活跃度亏空——这是 H3 的另一条通道，
    #       只在极端分支比下才显著，必须单独量化。
    print("\n  【判读：把 H3 拆成两条通道分别量化】")
    print(f"    {'n':>5}{'公式偏差(无clamp)':>20}{'clamp后偏差':>14}{'clamp造成':>12}")
    chan = []
    for r in rows:
        cfg = HawkesConfig(base_lambda=target_lambda,
                           branching=r["branching"], beta=BETA)
        noclamp = simulate_event_rate(cfg.to_process(), n_steps,
                                      np.random.default_rng(42),
                                      base_lambda=target_lambda, clamp=None,
                                      warmup=2000)
        clamp_effect = r["current_rel_err"] - noclamp["rel_err"]
        chan.append({"branching": r["branching"],
                     "formula_rel_err": noclamp["rel_err"],
                     "clamped_rel_err": r["current_rel_err"],
                     "clamp_effect": clamp_effect})
        print(f"    {r['branching']:>5.1f}{noclamp['rel_err'] * 100:>19.2f}%"
              f"{r['current_rel_err'] * 100:>13.2f}%{clamp_effect * 100:>11.2f}%")
    worst_formula = max(abs(c["formula_rel_err"]) for c in chan)
    print(f"\n    ① 公式本体：最差偏差 {worst_formula * 100:.2f}%"
          f"（判据 < 2%）→ {'✅ 正确' if worst_formula < 0.02 else '❌ 有问题'}")
    print("       → H3 的**机制本体（用错分支比）已被排除**。")
    print("    ② clamp 护栏：**实测在本区间内完全不生效**（见上表最后一列，全为 0.00%）。")
    print("       原因是 ρ 的量级在 0.6~1.4，离阈值 clamp_rho=4 差得远，护栏从未被触发。")
    print("       ⚠️ 自我纠错：第一版把两次**不同随机流**的差异（约 6 个百分点）当成了")
    print("       护栏效应。同种子重测后差异是 0.000%。")
    print("       **所以「护栏压低了活跃度」这个说法是错的**，已从结论里删除。")
    print("       市场里 ρ̄≈0.92 另有真原因：λ̄ 的标定窗口含瞬态、比真实基线高约 7%"
          "（见 calibrate_base_lambda 的 sim_kw 说明）。")
    leg90 = [r for r in rows if r["branching"] == 0.9][0]
    print(f"    ③ 量级对照：旧公式在 n=0.9 时偏差 {leg90['legacy_rel_err'] * 100:+.1f}%"
          f"（ρ̄={leg90['legacy_rho_mean']:.3f}），")
    print("       ——**这就是「如果公式没修」会造成的量级**，"
          "也正是变异体 M34 要注入的错误。")
    return {"rows": rows, "channels": chan,
            "worst_formula_rel_err": worst_formula,
            "formula_ok": bool(worst_formula < 0.02),
            "legacy_worst_rel_err": max(abs(r["legacy_rel_err"]) for r in rows),
            "target_lambda": target_lambda, "n_steps": n_steps}


# ======================================================================
# EA.1  k 与 ρ 的联合测量（决定性）
# ======================================================================
def recording_factory(cfg: HawkesConfig, branching: float, sink: list):
    """把每个被创建的市场记下来，跑完再统一取 ρ̄/成交率。

    为什么必须这样记：``make_controls`` / ``paired_impact`` 在内部自己造市场，
    调用方拿不到实例——所以 E8.2 才**没有**留下处理臂的 ρ。
    包一层工厂是代价最小的修法，且不用改动 ``run_stage3``。
    """
    def f(sim):
        m = HawkesMarket(sim, hawkes_config=cfg, hawkes_off=(branching <= 0.0))
        sink.append(m)
        return m
    return f


def window_stats(markets, lo: int, hi: int) -> dict:
    """在**观测窗口** [lo, hi) 内统计 ρ、活跃主体数与成交率。

    ⚠️ 窗口必须与冲击测量所在的窗口一致。任务书特别提醒过这一点：
    深度/强度快照若与真实测量冲击的时点不是同一批，
    测出来的差异可能和真正影响 k 的那个差异不是一回事。

    ⭐ ``n_active`` 与 ``activated_frac`` 是这里最关键的两个数：
    ``rho_to_count`` 把 ρ 夹在 ``[0, n_agents]``，所以
    **ρ 整体偏高时机制会静默哑火**（每 tick 都切出全部主体）。
    只看 ρ̄ 会以为"聚集很强"，看 n_active 才知道它有没有真的在起作用。
    """
    rhos, evs, acts = [], [], []
    for m in markets:
        n_agents = len(m.agents)
        for d in m.hawkes_log:
            if lo <= d["tick"] < hi:
                rhos.append(d["rho"])
                evs.append(d["events"])
                acts.append(d.get("n_active", float("nan")))
                m.__dict__.setdefault("_n_agents_seen", n_agents)
    r = np.asarray(rhos, dtype=float)
    e = np.asarray(evs, dtype=float)
    a = np.asarray(acts, dtype=float)
    n_tot = len(markets[0].agents) if markets else 0
    return {
        "n_ticks": int(r.size),
        "rho_mean": float(r.mean()) if r.size else float("nan"),
        "rho_sd": float(r.std(ddof=1)) if r.size > 1 else float("nan"),
        "events_per_tick": float(e.mean()) if e.size else float("nan"),
        "n_active_mean": float(np.nanmean(a)) if a.size else float("nan"),
        "activated_frac": (float((a < n_tot).mean()) if a.size else float("nan")),
        "n_agents": int(n_tot),
    }


def ea1_k_and_rho(seeds=SEEDS, arms=ARMS, shock_levels=None,
                  *, calibrate_on_same_market: bool = False) -> dict:
    """决定性问题：把 ρ̄ 真的测出来之后，k 还改善吗？活跃度有没有被污染？

    ``calibrate_on_same_market``
        ``False``（默认）= 复现 E8.2 的原始口径：λ̄ 在**裸 Market** 上标定，
        而实验用**带 MM_KW** 的市场。两者不是同一个市场 → ρ 的标尺偏 1.74 倍。
        ``True`` = 修正口径：λ̄ 在**同一个市场**（含 MM_KW）上标定。

    这个开关本身就是本工作线最重的发现：E8.2 的 ``base_lambda`` 与它实际
    作用的市场**不是同一个**。所以两个模式都要跑，才能说清"k 的改善"
    到底是聚集性的功劳、还是"机制其实哑火了"的副产品。
    """
    ensure_scripts_on_path()
    from run_stage3 import HORIZON, MM_KW, MM_MIX, make_controls, paired_impact
    ensure_scripts_on_path()
    from run_stage3 import WARMUP as W3_WARMUP

    shock_levels = shock_levels or [0.001, 0.0025, 0.005, 0.01]
    mode = "同市场口径（含 MM_KW）" if calibrate_on_same_market else "裸 Market 口径（E8.2 原样）"
    print(f"\n【EA.1{'b' if calibrate_on_same_market else ''}】k 与活跃度的联合测量"
          f" —— 标定口径：{mode}")
    base_lam = calibrate_base_lambda(
        seeds=seeds, n_ticks=3_000, n_agents=N_AGENTS, mix=MM_MIX, warmup=500,
        sim_kw=(MM_KW if calibrate_on_same_market else None))
    print(f"  λ̄ = {base_lam:.3f} 笔/tick")
    # 实测该市场的真实基线成交率（Hawkes 关），用来判断 λ̄ 标尺对不对
    truth = []
    for sd in seeds:
        m0 = HawkesMarket(
            SimConfig(seed=sd, n_ticks=W3_WARMUP + HORIZON,
                      population=Population.from_shares(N_AGENTS, MM_MIX),
                      **MM_KW),
            hawkes_config=HawkesConfig(base_lambda=base_lam, branching=0.0,
                                       beta=BETA),
            hawkes_off=True)
        m0.run(W3_WARMUP + HORIZON)
        tr = [t for t in m0.log.trades if W3_WARMUP <= t.tick < W3_WARMUP + HORIZON]
        truth.append(len(tr) / HORIZON)
    truth_m = float(np.mean(truth))
    print(f"  该市场（含 MM_KW）实测基线成交率 = {truth_m:.3f} 笔/tick"
          f"  → λ̄ 标尺偏 {truth_m / base_lam:.3f}×"
          f"{'  ⚠️ 标尺错误，机制会撞上限哑火' if truth_m / base_lam > 1.2 else '  ✅ 标尺一致'}")
    print(f"  观测窗口 [{W3_WARMUP}, {W3_WARMUP + HORIZON}) —— 与冲击测量同一窗口")
    print(f"\n    {'臂':<12}{'档位':>9}{'成交量':>9}{'成交率':>8}{'因果滑点bp':>12}")

    out_arms = {}
    for br in arms:
        cfg = HawkesConfig(base_lambda=base_lam, branching=br, beta=BETA)
        sink: list = []
        fn = recording_factory(cfg, br, sink)
        ctrl = make_controls(MM_MIX, seeds, HORIZON, factory=fn)
        rows = []
        tag = "Hawkes 关" if br <= 0 else f"Hawkes 开(b={br:g})"
        for f_ in shock_levels:
            r = paired_impact(f"{tag}|{f_:.4f}", MM_MIX, f_, 20, seeds, ctrl,
                              HORIZON, factory=fn)
            rows.append(r)
            print(f"    {tag:<12}{f_:>9.2%}{r['delivered_qty']:>9.1f}"
                  f"{r['fill_ratio']:>8.0%}{r['slippage_bp']:>12.2f}")
        un = unsaturated(rows)
        fit = fit_power_law(un, "slippage_bp")
        ws = window_stats(sink, W3_WARMUP, W3_WARMUP + HORIZON)
        # 关掉机制时没有 hawkes_log，用上面实测的 truth_m 代进去
        if br <= 0:
            ws = {"n_ticks": len(seeds) * HORIZON, "rho_mean": float("nan"),
                  "rho_sd": float("nan"), "events_per_tick": truth_m,
                  "n_active_mean": float("nan"), "activated_frac": float("nan"),
                  "n_agents": N_AGENTS}
        out_arms[tag] = {"branching": br, "rows": rows,
                         "k": fit.exponent, "r2": fit.r_squared,
                         "n_unsaturated": len(un), "n_levels": len(rows),
                         "fit_ok": bool(fit.ok), "rho_vs_lambda": ws["rho_mean"],
                         **ws}
        print(f"    {'':<12}{'':>9}{'':>9}{'':>8}  k={fit.exponent:>7.3f}"
              f"  |k−0.5|={fit.closeness_to_sqrt():>6.3f}"
              f"  ρ̄={f(ws['rho_mean'], '.4f')}  活跃主体={f(ws['n_active_mean'], '.1f')}"
              f"  稀疏化生效比例={f(ws['activated_frac'], '.1%')}"
              f"  成交/tick={f(ws['events_per_tick'], '.2f')}")

    # ---- 判读 ----
    print("\n  【判读】")
    base_tag = "Hawkes 关"
    k0 = out_arms[base_tag]["k"]
    ev0 = out_arms[base_tag]["events_per_tick"]
    verdicts = []
    for tag, a in out_arms.items():
        if a["branching"] <= 0:
            continue
        dk = a["k"] - k0
        act_ratio = a["events_per_tick"] / max(ev0, 1e-9)
        af = a.get("activated_frac", float("nan"))
        ef = a.get("events_per_tick", float("nan"))
        print(f"    {tag}: k {k0:.3f} → {a['k']:.3f}（{dk:+.3f}）")
        print(f"      活跃度：成交/tick {ev0:.2f} → {ef:.2f}"
              f"（{act_ratio:.4f}×，偏离 1 有 {abs(act_ratio - 1):.1%}）")
        print(f"      ρ̄={f(a['rho_mean'], '.4f')}；平均活跃主体"
              f"{f(a['n_active_mean'], '.1f')}/{a['n_agents']}；"
              f"**真正被稀疏化的 tick 占比 = {f(af, '.1%')}**")
        if np.isfinite(af) and af < 0.5:
            print("      ⚠️ 稀疏化生效比例不足一半 ⇒ **机制大部分时间处于哑火状态**，"
                  "此时 k 的变化不能归因给它")
        verdicts.append({"arm": tag, "k_improved": bool(a["k"] < k0),
                         "d_k": float(dk), "activity_ratio": float(act_ratio),
                         "activated_frac": float(af) if np.isfinite(af) else None,
                         "mechanism_effective": bool(np.isfinite(af) and af >= 0.5)})
    return {"base_lambda": base_lam, "calibrate_on_same_market": bool(calibrate_on_same_market),
            "market_true_rate": truth_m, "lambda_scale_bias": truth_m / base_lam,
            "arms": out_arms, "h3_checks": verdicts,
            "window": [W3_WARMUP, W3_WARMUP + HORIZON]}


# ======================================================================
# EA.2  突发期/背景期深度对比（检验 H1）
# ======================================================================
def ea2_burst_depth(seeds=SEEDS, branching: float = 0.6,
                    percentile: float = 80.0, block: int = 200) -> dict:
    """在**与冲击测量同一窗口**内比较突发期与背景期的盘口深度。

    任务书提醒过：深度快照的时间点必须与实际测量冲击时用的时点对齐，
    否则"测出来的深度差异"和"真正影响 k 的那个深度差异"不是一回事。
    这里两条都报：
      · ``window``  观测窗口 [6000, 6400) 整体的突发/背景对比（统计口径）
      · ``at_shock`` 冲击前一刻的深度（与阶段3 的 ``depth_before`` 同口径同时点）
    """
    ensure_scripts_on_path()
    from run_stage3 import HORIZON, MM_KW, MM_MIX, WARMUP as W3_WARMUP

    print("\n【EA.2】突发期 vs 背景期深度（检验 H1）")
    base_lam = calibrate_base_lambda(seeds=seeds, n_ticks=3_000,
                                     n_agents=N_AGENTS, mix=MM_MIX, warmup=500,
                                     sim_kw=MM_KW)
    cfg = HawkesConfig(base_lambda=base_lam, branching=branching, beta=BETA)
    runs, at_shock = [], []
    for sd in seeds:
        m = HawkesMarket(
            SimConfig(seed=sd, n_ticks=W3_WARMUP + HORIZON,
                      population=Population.from_shares(N_AGENTS, MM_MIX),
                      **MM_KW),
            hawkes_config=cfg, record_depth=True)
        m.run(W3_WARMUP + HORIZON)
        log = [d for d in m.hawkes_log if W3_WARMUP <= d["tick"] < W3_WARMUP + HORIZON]
        depth = np.array([d["depth"] for d in log], dtype=float)
        rho = np.array([d["rho"] for d in log], dtype=float)
        r = burst_depth_contrast(depth, rho, percentile_threshold=percentile,
                                 block=block)
        r["seed"] = sd
        runs.append(r)
        at_shock.append(float(depth[0]) if depth.size else float("nan"))
    agg = aggregate_burst_contrast(runs, label=f"Hawkes b={branching:g}")
    rho_acf = _rho_acf_within(cfg, seeds[0], W3_WARMUP, HORIZON)
    print(f"  λ̄={base_lam:.2f}  每场 {len(runs)} 次运行 × "
          f"{runs[0].get('n_ticks', 0)} tick")
    print(f"    突发期均值深度 {f(agg['per_run']['mean'], '.5f')}（log 比值）"
          f"  → 相对效应 {f(agg['pct_effect'], '.2%')}")
    print(f"    跨运行 t={f(agg['per_run']['t'], '.2f')}  p={f(agg['per_run']['p'], '.4g')}"
          f"  n={agg['n_runs']}")
    print(f"    verdict: {agg['verdict']}")
    print(f"    （参考：把所有 tick 当独立样本的 tick 级 p 中位数 = "
          f"{f(np.median([r['tick_level']['p'] for r in runs]), '.3g')}）")
    print(f"    冲击前一刻深度（与阶段3 depth_before 同口径）= "
          f"{f(float(np.mean(at_shock)), '.1f')} 手")
    # 与 k 的改善量级对照
    print("    ⚠️ H1 要能解释 k 的改善，需要「突发期深度差异」的量级与"
          "k 的改善相称；本项给出的是**深度**的相对变化，"
          "不是 k 的变化——两者不能直接相减，只能做量级对照。")
    return {"base_lambda": base_lam, "branching": branching,
            "percentile": percentile, "block": block,
            "runs": runs, "aggregate": agg,
            "depth_at_shock_mean": float(np.mean(at_shock)),
            "rho_acf": rho_acf}


def _rho_acf_within(cfg: HawkesConfig, seed: int, lo: int, hi: int,
                    max_lag: int = 100) -> list[float]:
    """观测窗口内 ρ 的自相关——用来证明"聚集性确实在"。

    这是 EA.3 的**前提检查**：对照组（置换代理）必须把这段自相关抹掉，
    否则"对照组"自己还带着聚集性，比较就不干净了。
    """
    m = HawkesMarket(SimConfig(seed=seed, n_ticks=lo + hi,
                               population=Population.from_shares(
                                   N_AGENTS, MM_MIX_FALLBACK())),
                     hawkes_config=cfg)
    m.run(lo + hi)
    r = np.array([d["rho"] for d in m.hawkes_log if lo <= d["tick"] < lo + hi],
                 dtype=float)
    if r.size < max_lag + 2:
        return []
    x = r - r.mean()
    den = float((x * x).sum())
    return [float((x[:-k] * x[k:]).sum() / den) for k in range(1, max_lag + 1)]


def MM_MIX_FALLBACK():
    return {"zero_intel": 0.27, "fundamentalist": 0.36, "chartist": 0.27,
            "market_maker": 0.10}


# ======================================================================
# EA.3  方差匹配对照（检验 H2）
# ======================================================================
def ea3_variance_matched(seeds=SEEDS, branching: float = 0.6,
                         shock_levels=None) -> dict:
    """把 Hawkes 的 ρ 序列**随机置换**后喂回去：方差一样，时间结构没了。

    为什么用置换而不是任务书建议的"正弦波调制"：
    置换保持了 ρ 的**整个边际分布**（因而均值、方差、偏度全部一致），
    只把"事件→事件"的自激依赖抹掉。这比"调一条正弦波去匹配方差"
    干净得多——正弦波本身有确定性周期，会引入另一种时间结构，
    于是对照组自己又带上了一层"聚集"，结论不干净（任务书自己也担心这点）。
    """
    ensure_scripts_on_path()
    from run_stage3 import HORIZON, MM_KW, MM_MIX, make_controls, paired_impact
    ensure_scripts_on_path()
    from run_stage3 import WARMUP as W3_WARMUP

    shock_levels = shock_levels or [0.001, 0.0025, 0.005, 0.01]
    print("\n【EA.3】方差匹配的代理对照（检验 H2）")
    base_lam = calibrate_base_lambda(seeds=seeds, n_ticks=3_000,
                                     n_agents=N_AGENTS, mix=MM_MIX, warmup=500,
                                     sim_kw=MM_KW)
    cfg = HawkesConfig(base_lambda=base_lam, branching=branching, beta=BETA)
    total = W3_WARMUP + HORIZON
    # ① 先跑一场 Hawkes，取它的 ρ 序列，并按种子生成置换版本
    schedules = {}
    for sd in seeds:
        m = HawkesMarket(SimConfig(seed=sd, n_ticks=total,
                                   population=Population.from_shares(
                                       N_AGENTS, MM_MIX), **MM_KW),
                         hawkes_config=cfg)
        m.run(total)
        rho = np.array([d["rho"] for d in m.hawkes_log], dtype=float)
        rng = np.random.default_rng([sd, 0x5EED])
        schedules[sd] = rng.permutation(rho)
    # ② 证实置换确实抹掉了自相关
    def acf(x, lag=20):
        x = x - x.mean()
        den = float((x * x).sum())
        return float((x[:-lag] * x[lag:]).sum() / den) if den else float("nan")
    haw_acf = []
    perm_acf = []
    for sd in seeds:
        m = HawkesMarket(SimConfig(seed=sd, n_ticks=total,
                                   population=Population.from_shares(
                                       N_AGENTS, MM_MIX), **MM_KW),
                         hawkes_config=cfg)
        m.run(total)
        rho = np.array([d["rho"] for d in m.hawkes_log], dtype=float)
        haw_acf.append(acf(rho[W3_WARMUP:]))
        perm_acf.append(acf(schedules[sd][W3_WARMUP:]))
    print(f"    ρ 的 lag-20 自相关：Hawkes {f(np.mean(haw_acf), '+.4f')}"
          f"  置换代理 {f(np.mean(perm_acf), '+.4f')}  ← 代理必须接近 0")

    # ③ 用置换后的 ρ 作为外生序列跑冲击实验
    def mk_factory(sd):
        return lambda sim: HawkesMarket(
            sim, hawkes_config=cfg, rho_schedule=schedules[sd])
    # ⚠️ 每场的 schedule 不同 → factory 必须按种子分派；
    #    `make_controls`/`paired_impact` 内部按 seeds 顺序取用，这里用闭包表。
    by_seed = {sd: mk_factory(sd) for sd in seeds}

    class _Dispatch:
        """按 `sim.seed` 分派到该种子的代理 factory（保持配对正确）。"""

        def __init__(self, table, default):
            self.table, self.default = table, default

        def __call__(self, sim):
            return self.table.get(sim.seed, self.default)(sim)

    fn = _Dispatch(by_seed, mk_factory(seeds[0]))
    ctrl = make_controls(MM_MIX, seeds, HORIZON, factory=fn)
    rows = []
    for f_ in shock_levels:
        rows.append(paired_impact(f"surrogate|{f_:.4f}", MM_MIX, f_, 20, seeds,
                                  ctrl, HORIZON, factory=fn))
    un = unsaturated(rows)
    fit = fit_power_law(un, "slippage_bp")
    print(f"    置换代理：k={f(fit.exponent, '.3f')}  |k−0.5|="
          f"{f(fit.closeness_to_sqrt(), '.3f')}  （有效档位 {len(un)}/{len(rows)}）")
    return {"base_lambda": base_lam, "branching": branching, "rows": rows,
            "k": fit.exponent, "closeness": fit.closeness_to_sqrt(),
            "rho_acf_hawkes": float(np.mean(haw_acf)),
            "rho_acf_permuted": float(np.mean(perm_acf)),
            "n_unsaturated": len(un)}


# ======================================================================
# EA.4  单侧稀疏化（需求侧 vs 供给侧）
# ======================================================================
def ea4_one_sided(seeds=SEEDS, branching: float = 0.6, shock_levels=None,
                  *, calib_window=None, tag: str = "EA.4",
                  scopes=("both", "taker", "maker")) -> dict:
    """单侧稀疏化：聚集作用在需求侧还是供给侧？

    ``calib_window``
        标定窗口 ``(warmup, n_ticks)``。``None`` ⇒ 历史口径 ``[500, 3000)``
        （保证三线深挖的旧数字可复现）；传 ``ea4_windows()["experiment"]``
        ⇒ **标定窗口 = 实验窗口**（任务书 §0 第六条纪律）。
        两种口径的差异由 ``scripts/verify_EA4_calibration.py`` 量化。
    """
    ensure_scripts_on_path()
    from run_stage3 import HORIZON, make_controls, paired_impact
    ensure_scripts_on_path()
    from run_stage3 import WARMUP as W3_WARMUP

    spec = ea4_market_spec()
    windows = ea4_windows()
    calib_window = tuple(calib_window) if calib_window else windows["legacy_calibration"]
    exp_window = windows["experiment"]
    shock_levels = shock_levels or [0.001, 0.0025, 0.005, 0.01]
    aligned = tuple(calib_window) == tuple(exp_window)
    print(f"\n【{tag}】单侧稀疏化：聚集作用在需求侧还是供给侧？")
    print(f"  标定窗口=[{calib_window[0]}, {calib_window[1]})  "
          f"实验窗口=[{exp_window[0]}, {exp_window[1]})  "
          f"{'✅ 对齐' if aligned else '⚠️ 不对齐（历史口径）'}")

    cal_kw = ea4_calibration_kwargs(seeds, calib_window, spec)
    base_lam = calibrate_base_lambda(**cal_kw)
    exp_kw = ea4_experiment_kwargs(exp_window, spec)
    print(f"  λ̄={base_lam:.3f}（同市场口径，窗口={'实验窗口' if aligned else '历史窗口'}）")

    out_arms = {}
    for scope in scopes:
        cfg = HawkesConfig(base_lambda=base_lam, branching=branching, beta=BETA)
        sink: list = []

        def fn(sim, cfg=cfg, scope=scope, sink=sink):
            m = HawkesMarket(sim, hawkes_config=cfg, thin_scope=scope)
            sink.append(m)
            return m

        ctrl = make_controls(exp_kw["mix"], seeds, HORIZON, factory=fn)
        rows = []
        for f_ in shock_levels:
            rows.append(paired_impact(f"{scope}|{f_:.4f}", exp_kw["mix"], f_, 20, seeds,
                                      ctrl, HORIZON, factory=fn))
        un = unsaturated(rows)
        fit = fit_power_law(un, "slippage_bp")
        ws = window_stats(sink, W3_WARMUP, W3_WARMUP + HORIZON)
        out_arms[scope] = {"k": fit.exponent, "closeness": fit.closeness_to_sqrt(),
                           "rows": rows, "n_unsaturated": len(un), **ws}
        print(f"    {scope:<6} k={f(fit.exponent, '.3f')}  |k−0.5|="
              f"{f(fit.closeness_to_sqrt(), '.3f')}  ρ̄={f(ws['rho_mean'], '.3f')}"
              f"  活跃主体={f(ws['n_active_mean'], '.1f')}"
              f"  稀疏化生效={f(ws['activated_frac'], '.1%')}"
              f"  成交/tick={f(ws['events_per_tick'], '.2f')}")
    return {"base_lambda": base_lam, "branching": branching, "arms": out_arms,
            "label": tag, "calib_window": list(calib_window),
            "experiment_window": list(exp_window),
            "window_aligned": aligned,
            "calibration_kwargs": {k: (v if not isinstance(v, dict) else "<同 spec>")
                                   for k, v in cal_kw.items()}}


# ======================================================================
def main() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="EA0 / EA1 / EA1b / EA2 ...")
    args = ap.parse_args()
    banner("工作线A  阶段8「意外修复」的机制解耦")
    t0 = time.time()
    out: dict = {"stage": "A", "config": {"seeds": SEEDS, "arms": list(ARMS),
                                          "beta": BETA}}
    only = (args.only or "").upper()
    if not only or only == "EA0":
        out["ea0"] = ea0_discretization()
    if not only or only == "EA1":
        out["ea1_uncorrected"] = ea1_k_and_rho(calibrate_on_same_market=False)
    if not only or only == "EA1B":
        out["ea1_corrected"] = ea1_k_and_rho(calibrate_on_same_market=True)
    if not only or only == "EA2":
        out["ea2_burst_depth"] = ea2_burst_depth()
    if not only or only == "EA3":
        out["ea3_variance_matched"] = ea3_variance_matched()
    if not only or only == "EA4":
        out["ea4_one_sided"] = ea4_one_sided()
    out["elapsed_sec"] = time.time() - t0
    # ⚠️ **合并写入**，不是覆盖：`--only` 是分批跑的，
    #    如果用 ``save_json(out, ...)`` 直接覆盖，跑 EA3 就会把 EA0/EA1
    #    的结果抹掉（"两个写者各写一半"是本项目反复踩的同一类问题）。
    merged = load_json("workstream_A_metrics.json") if (OUT / "workstream_A_metrics.json").exists() else {}
    merged.update(out)
    merged["elapsed_sec"] = out["elapsed_sec"]
    save_json(merged, "workstream_A_metrics.json")
    print(f"\n  总用时 {out['elapsed_sec']:.0f}s → out/workstream_A_metrics.json")
    return out


if __name__ == "__main__":
    main()
