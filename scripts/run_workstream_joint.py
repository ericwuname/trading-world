"""三线合并验证（EJ.1 / EJ.2）——任务书的最后一项。

⚠️ 前置纪律：**先写合并计划，再动手**（见 ``docs/三线深挖-合并计划.md``）。
本脚本的 ``write_result_section()`` 在跑完后把结果**追加/替换**回那份计划的 §5，
所以"假设"与"结果"在同一份文件里，读者能看到假设是在跑之前定的。

组合方式与理由（详见合并计划）：

    J1  阶段6 基线（仅图表派元订单 + 自适应流动性）
    J2  J1 + B 的三类覆盖
    J3  J2 + A 的修正 Hawkes（λ̄ 用**同市场口径**标定）
    J4  J3 的配置放进阶段11「至暗时刻」场景，看链条判定环节数有没有变化

**两条硬规矩（都是 A 线用血换来的）**：
1. ``λ̄`` 必须在**它将被使用的那个市场配置**上标定，且**标定窗口 = 实验窗口**。
   E8.2 错在"裸 Market 标定、带 MM_KW 的市场使用"（偏 1.767×），
   以及"标定窗口含瞬态"（再偏约 7%）。
2. 每个臂都要报**机制体检**：ρ̄ / 平均活跃主体 / **稀疏化真正生效的 tick 占比** /
   成交每 tick。只看 k 不看活跃度，正是 E8.2 翻车的形态。

用法::

    python scripts/run_workstream_joint.py            # EJ.1 + EJ.2
    python scripts/run_workstream_joint.py --only EJ1
"""

from __future__ import annotations

import argparse
import functools
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import OUT, banner, load_json, save_json  # noqa: E402
from run_stage11 import (  # noqa: E402
    DarkMomentAsset,
    SEEDS as S11_SEEDS,
    paired_chain,
    window_stats,
)
from run_stage6 import (  # noqa: E402
    FLOW_SENS,
    FLOW_WINDOW,
    META_CFG,
    META_SHARE,
    MM_MIX,
    N_AGENTS,
    Stage6Market,
)
from tw.multi_asset import AssetMarket  # noqa: E402
from tw.order_flow.hawkes import HawkesConfig  # noqa: E402
from tw.order_flow.hawkes_market import (  # noqa: E402
    HawkesMarket,
    calibrate_base_lambda,
)
from tw.order_flow.meta_order import MetaOrderConfig  # noqa: E402

BETA = 0.15
BRANCHING = 0.6
SHARES_CHART = {"chartist": META_SHARE}
SHARES_ALL = {"chartist": META_SHARE, "zero_intel": 0.5, "fundamentalist": 0.5}
PLAN = Path(__file__).resolve().parent.parent / "docs" / "三线深挖-合并计划.md"


# ----------------------------------------------------------------------
# 联合市场：元订单（三类）+ 修正后的 Hawkes
#
# MRO 已验证：JointMarket → Stage6Market → HawkesMarket → Market。
# 之所以能这么干净地组合，是因为两个类都用 ``**kw`` 透传没消费掉的参数，
# 而且各管一段互不重叠的方法（``decorate_agent`` vs ``_limit_activity``/``step``）。
# ⚠️ 若哪天有人给某一方加了 ``step`` 或 ``_limit_activity``，这条链会静默改变行为——
#    所以 ``test_workstream_scripts.py`` 里钉了一条"联合市场确实同时具备两种行为"的测试。
# ----------------------------------------------------------------------
class JointMarket(Stage6Market, HawkesMarket):
    """单资产联合市场（用于 k 值检验）。"""


class JointDarkMoment(DarkMomentAsset, Stage6Market, HawkesMarket):
    """阶段11 的资产 A：资金费率 + 元订单 + Hawkes。"""


class JointAsset(AssetMarket, Stage6Market, HawkesMarket):
    """阶段11 的资产 B：外部锚点 + 元订单 + Hawkes（不带杠杆）。"""


def f(v, fmt: str = ".3f", dash: str = "—") -> str:
    if not isinstance(v, (int, float)) or v != v:
        return dash
    return format(v, fmt)


def arm_kwargs(shares: dict, hawkes_cfg: HawkesConfig | None):
    """某个臂的市场构造参数（Hawkes 关时用 ``hawkes_off`` 而不是换类）。"""
    return dict(
        meta_cfg=MetaOrderConfig(**META_CFG), meta_shares=dict(shares),
        adapt_mm=True, flow_window=FLOW_WINDOW, flow_sensitivity=FLOW_SENS,
        hawkes_config=hawkes_cfg, hawkes_off=(hawkes_cfg is None),
    )


def census(markets) -> dict:
    """机制体检——**只看 k 不看活跃度，正是 E8.2 翻车的形态**。"""
    rho, act, ev, n_agents = [], [], [], 0
    for m in markets:
        if not getattr(m, "hawkes_log", None):
            continue
        n_agents = max(n_agents, len(m.agents))
        for d in m.hawkes_log:
            rho.append(d["rho"])
            act.append(d.get("n_active", float("nan")))
            ev.append(d["events"])
    if not rho:
        return {"n_agents": n_agents, "rho_mean": float("nan"),
                "n_active_mean": float("nan"), "activated_frac": float("nan"),
                "events_per_tick": float("nan")}
    a = np.asarray(act, dtype=float)
    return {"n_agents": int(n_agents), "rho_mean": float(np.mean(rho)),
            "n_active_mean": float(np.nanmean(a)),
            "activated_frac": float((a < n_agents).mean()),
            "events_per_tick": float(np.mean(ev))}


# ======================================================================
# EJ.1  k 值检验（J1 / J2 / J3）
# ======================================================================
ARM_SPECS = (
    ("J1 阶段6 基线", SHARES_CHART, False),
    ("J2 +B 三类覆盖", SHARES_ALL, False),
    ("J3 +A 修正 Hawkes", SHARES_ALL, True),
)


def ej1_joint_k(seeds=None, shock_levels=None) -> dict:
    from run_stage3 import HORIZON, MM_KW, make_controls, paired_impact
    from run_stage3 import WARMUP as W3

    seeds = seeds or [20260917, 20260924, 20260931]
    shock_levels = shock_levels or [0.001, 0.0025, 0.005, 0.01]
    n_ticks = W3 + HORIZON
    print("\n【EJ.1】联合配置下的冲击幂律指数 k")
    print(f"  标定口径：λ̄ 在**每条臂自己的市场配置**上标定，窗口 = 实验窗口 "
          f"[{W3}, {n_ticks})")

    arms = {}
    for tag, shares, use_hawkes in ARM_SPECS:
        cls = functools.partial(JointMarket, **arm_kwargs(shares, None))
        lam = calibrate_base_lambda(seeds=seeds, n_ticks=n_ticks, warmup=W3,
                                    n_agents=N_AGENTS, mix=MM_MIX, sim_kw=MM_KW,
                                    market_cls=cls)
        cfg = (HawkesConfig(base_lambda=lam, branching=BRANCHING, beta=BETA)
               if use_hawkes else None)
        kw = arm_kwargs(shares, cfg)
        sink: list = []

        def fn(sim, kw=kw, sink=sink):
            m = JointMarket(sim, **kw)
            sink.append(m)
            return m

        ctrl = make_controls(MM_MIX, seeds, HORIZON, factory=fn)
        rows = []
        for f_ in shock_levels:
            rows.append(paired_impact(f"{tag}|{f_:.4f}", MM_MIX, f_, 20, seeds,
                                      ctrl, HORIZON, factory=fn))
        un = rows  # 饱和判定在 fit 里做
        from tw.impact import fit_power_law, unsaturated
        un = unsaturated(rows)
        fit = fit_power_law(un, "slippage_bp")
        cs = census(sink)
        arms[tag] = {"shares": shares, "use_hawkes": use_hawkes,
                     "base_lambda": lam, "k": fit.exponent,
                     "closeness": fit.closeness_to_sqrt(), "r2": fit.r_squared,
                     "n_unsaturated": len(un), "n_levels": len(rows),
                     "rows": rows, **cs}
        print(f"    {tag:<20} λ̄={lam:>7.2f}  k={f(fit.exponent, '.3f')}"
              f"  |k−0.5|={f(fit.closeness_to_sqrt(), '.3f')}"
              f"  ρ̄={f(cs['rho_mean'], '.4f')}"
              f"  稀疏化生效={f(cs['activated_frac'], '.1%')}"
              f"  成交/tick={f(cs['events_per_tick'], '.2f')}")

    print("\n  【判读】")
    base = arms["J1 阶段6 基线"]["k"]
    for tag in ("J2 +B 三类覆盖", "J3 +A 修正 Hawkes"):
        a = arms[tag]
        print(f"    {tag:<20} k {base:.3f} → {a['k']:.3f}"
              f"（Δ={a['k'] - base:+.3f}）")
    j3 = arms["J3 +A 修正 Hawkes"]
    print(f"    → 联合配置最终 k = {f(j3['k'], '.3f')}"
          f"（离 0.5 还有 {f(abs(j3['k'] - 0.5), '.3f')}）")
    if not (np.isfinite(j3["activated_frac"]) and j3["activated_frac"] > 0.5):
        print("    ⚠️ J3 的稀疏化生效占比不足一半 ⇒ 该臂的 k 不许被解释成"
              "「聚集性的效果」（E8.2 的教训）")
    return {"arms": arms, "base_k": base,
            "window": [W3, n_ticks], "n_seeds": len(seeds)}


# ======================================================================
# EJ.2  阶段11「至暗时刻」链条（J4）
# ======================================================================
def ej2_joint_chain() -> dict:
    """把 J3 的配置放进阶段11 场景，看链条判定环节数有没有变化。

    做法：**猴补丁替换 ``run_stage11`` 的两个市场类名**，复用它的
    ``build`` / ``paired_chain`` / ``window_stats`` 全部逻辑——
    不复制那 40 行构造代码（复制就等于造出第二份会分叉的实现，
    本项目为此吃过亏）。
    """
    import run_stage11 as S11

    print("\n【EJ.2】阶段11「至暗时刻」链条（联合配置）")
    # 每个资产自己的 λ̄：两个市场的价格/角色不同，成交率也可能不同。
    # **不能共用一个 λ̄** —— 那正是 E8.2 的错误形态。
    lams = {}
    for name, cls, shares in (("A", functools.partial(JointDarkMoment, **arm_kwargs(SHARES_ALL, None)), SHARES_ALL),
                              ("B", functools.partial(JointAsset, **arm_kwargs(SHARES_ALL, None)), SHARES_ALL)):
        lams[name] = calibrate_base_lambda(
            seeds=S11_SEEDS, n_ticks=S11.N_TICKS, warmup=S11.WARMUP,
            n_agents=S11.N_AGENTS, mix=S11.MIX,
            sim_kw={**S11.MM_KW, "zi_p_buy": 0.55 if name == "A" else 0.50},
            market_cls=cls)
    print(f"    逐资产 λ̄：A={lams['A']:.2f}  B={lams['B']:.2f}")
    cfgA = HawkesConfig(base_lambda=lams["A"], branching=BRANCHING, beta=BETA)
    cfgB = HawkesConfig(base_lambda=lams["B"], branching=BRANCHING, beta=BETA)
    S11.DarkMomentAsset = functools.partial(
        JointDarkMoment, **arm_kwargs(SHARES_ALL, cfgA))
    S11.AssetMarket = functools.partial(
        JointAsset, **arm_kwargs(SHARES_ALL, cfgB))

    chain_keys = [
        ("devA", "① A 的价格偏离(bp)"),
        ("devB", "② B 的价格偏离(bp)"),
        ("depthA", "③ A 的买盘深度(手)"),
        ("trend_inv", "④ 趋势追随者净持仓"),
        ("fundingA", "⑤ A 的资金费率"),
        ("arb_inv", "⑥ 套利者净持仓"),
    ]
    per_key: dict[str, list] = {k: [] for k, _ in chain_keys}
    print(f"    {'seed':>10}{'环节':<22}{'差':>13}{'z':>9}{'判定':>10}")
    for sd in S11_SEEDS:
        pr = paired_chain(sd)
        for k, label in chain_keys:
            st = window_stats(pr["diff"], k, pr["shock_tick"])
            per_key[k].append(st)
            print(f"    {sd:>10}{label:<22}{st['delta']:>13.4g}"
                  f"{st['z']:>9.2f}{'✅观察到' if st['observed'] else '❌未观察到':>10}")
    summary, n_obs = {}, 0
    for k, label in chain_keys:
        sts = per_key[k]
        obs = [bool(s["observed"]) for s in sts]
        n_obs += sum(obs)
        # 判据与 E11.1 **完全一致**：多数种子观察到才算"该环节成立"
        summary[k] = {"label": label, "n_observed": int(sum(obs)),
                      "n_seeds": len(obs), "holds": bool(sum(obs) * 2 > len(obs))}
    n_links = int(sum(1 for v in summary.values() if v["holds"]))
    print(f"\n    → 联合配置下成立环节数 = **{n_links}/6**"
          f"（逐种子观察次数合计 {n_obs}）")
    return {"per_key": summary, "n_links": n_links,
            "lambda_A": lams["A"], "lambda_B": lams["B"],
            "n_seeds": len(S11_SEEDS)}


# ======================================================================
def write_result_section(J: dict) -> None:
    """把结果写回合并计划的 §5（假设在跑之前就在同一份文件里，读者可对照）。"""
    if not PLAN.exists():
        return
    txt = PLAN.read_text(encoding="utf-8")
    head = txt.split("## 5. 结果与判读")[0]
    j1 = J.get("ej1") or {}
    j2 = J.get("ej2") or {}
    L = ["## 5. 结果与判读\n"]
    if j1:
        L.append("\n### J1–J3：联合配置下的 k\n")
        L.append("| 臂 | λ̄（同市场口径） | k | \\|k−0.5\\| | ρ̄ | 稀疏化生效 | 成交/tick |")
        L.append("|---|---|---|---|---|---|---|")
        for tag, a in (j1.get("arms") or {}).items():
            L.append(f"| {tag} | {f(a.get('base_lambda'), '.2f')} | {f(a.get('k'), '.3f')} | "
                     f"{f(a.get('closeness'), '.3f')} | {f(a.get('rho_mean'), '.4f')} | "
                     f"{f(a.get('activated_frac'), '.1%')} | {f(a.get('events_per_tick'), '.2f')} |")
        j3 = (j1.get("arms") or {}).get("J3 +A 修正 Hawkes", {})
        L.append("")
        L.append(f"- 阶段6 基线 k = {f(j1.get('base_k'), '.3f')}")
        L.append(f"- **联合配置最终 k = {f(j3.get('k'), '.3f')}**"
                 f"（离 0.5 还有 {f(abs((j3.get('k') or 0) - 0.5), '.3f')}）")
        L.append(f"- J3 的稀疏化生效占比 = {f(j3.get('activated_frac'), '.1%')}"
                 "（这一项不足 50% 时，该臂的 k 不许解释成「聚集性的效果」）")
        L.append("")
        L.append("**C 线对这个 k 的贡献是零，不是「很小」**——C 的修复只作用于配对交易者，"
                 "而配对交易者在多资产市场里；本 k 检验是单资产实验。")
    if j2:
        L.append("\n### J4：阶段11「至暗时刻」链条（联合配置）\n")
        L.append(f"- 逐资产 λ̄：A={f(j2.get('lambda_A'), '.2f')}，B={f(j2.get('lambda_B'), '.2f')}")
        L.append(f"- **成立环节数 = {j2.get('n_links')}/6**"
                 f"（种子数 {j2.get('n_seeds')}；判据与 E11.1 完全一致，只换配置）")
        L.append("")
        L.append("| 环节 | 观察到 / 种子数 | 是否成立 |")
        L.append("|---|---|---|")
        for k, v in (j2.get("per_key") or {}).items():
            L.append(f"| {v.get('label')} | {v.get('n_observed')}/{v.get('n_seeds')} | "
                     f"{'✅' if v.get('holds') else '❌'} |")
        L.append("")
    L.append("\n### 与任务书三个预期的对照\n")
    L.append("- 任务书希望「综合起来 k 能否落在 0.5 附近」→ 见上表。**差多少就报多少，"
             "不因为没到 0.5 就只写「机制已实现」。**")
    L.append("- 任务书希望「至暗时刻传导链条判定环节数是否有变化」→ 与阶段11 原结果对照即可。")
    L.append("- 任务书要求的三条线**叠加**里，C 对 k 的贡献为零（口径不同，见合并计划 §4）；"
             "J4 里 C 的修复（`short_headroom`）才真正参与。")
    # ⚠️ 必须 "\n".join —— 本轮真实踩过：写成 "".join 时整节被拼成**一行**，
    #    表格在渲染器里彻底失效，而脚本退出码 0、自检也不看渲染，
    #    「跑成功了」与「产物能读」之间就掉进去了一次。
    #    回归测试：tests/test_workstream_scripts.py::TestPlanDocIsRenderableMarkdown
    PLAN.write_text(head + "\n".join(L), encoding="utf-8")


def main() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="EJ1 / EJ2")
    args = ap.parse_args()
    banner("三线合并验证")
    t0 = time.time()
    out: dict = {"stage": "J", "config": {
        "shares_chart": SHARES_CHART, "shares_all": SHARES_ALL,
        "branching": BRANCHING, "beta": BETA,
        "meta_cfg": {k: v for k, v in META_CFG.items()}}}
    only = (args.only or "").upper()
    if not only or only == "EJ1":
        out["ej1"] = ej1_joint_k()
    if not only or only == "EJ2":
        out["ej2"] = ej2_joint_chain()
    out["elapsed_sec"] = time.time() - t0
    merged = load_json("workstream_joint_metrics.json") if (
        OUT / "workstream_joint_metrics.json").exists() else {}
    merged.update(out)
    merged["elapsed_sec"] = out["elapsed_sec"]
    save_json(merged, "workstream_joint_metrics.json")
    # 只要本次或**以往**跑出过结果，就把 §5 重写一遍。
    # ⚠️ 原来只在"本次跑了 EJ1"时才写，于是分批跑（先 EJ1 后 EJ2）之后，
    #    §5 里永远缺 J4 那一段——"分两次跑"这个正常用法会产出不完整的交付物。
    if merged.get("ej1") or merged.get("ej2"):
        write_result_section(merged)
        print(f"\n  结果已写回 {PLAN.name} 的 §5")
    print(f"\n  总用时 {out['elapsed_sec']:.0f}s → out/workstream_joint_metrics.json")
    return out


if __name__ == "__main__":
    main()
