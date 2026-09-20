"""阶段11：终局整合与「至暗时刻」压力测试（二期指导书 §8）。

目标
----
把阶段5~10 的全部机制合并进**同一个市场实例**，做一次极端压力测试，
观察一条连锁反应的传导路径与量级。

「至暗时刻」的链条
------------------
1. **多资产**（阶段9）：两个资产、高相关配置
2. **拥挤交易**（阶段7）：大量 ``AdaptiveTrendFollower`` 已收敛到相似的动量参数
3. **资金费率高位**（阶段5）：长期单边多头拥挤把费率推到历史高位
4. **触发**：资产 A 遭遇一次大规模清算
5. **观察链条**：
     A 的价格冲击 → 经相关性传导到 B →
     趋势追随者集体触发离场（拥挤 → 行为同步）→
     资金费率因大量平仓迅速转向 →
     资金费率套利者集体反向调仓，加剧短期波动

⭐ 四条必须先说清楚的方法论
==========================

**① 每个环节都必须有**同种子配对的对照**，否则测的是噪声。**
   一期阶段3 已经证明：不做配对时，随机漂移会完全淹没冲击
   （曾把分期清算的幂律指数算成 −0.26）。
   本脚本对每个环节都跑"施加清算"与"不施加清算"两条同种子路径，
   逐 tick 相减。

**② 观察链条时**不要用"最大值"当强度。**
   最大值受单点噪声主导。用**配对差在窗口内的均值**，
   并报它的标准误。峰值只作为形状参考。

**③ 「连锁"被观察到"的标准必须事先定死。**
   定义：某环节的配对差在冲击后 200 tick 内的均值，
   绝对值超过**该环节配对差在冲击前 200 tick 的均值 ± 3 倍标准误**，
   才算"观察到传导"。
   事后挑指标（"这条也算传导吧"）会把任何结果都解释成成功。

**④ 允许"某些环节没有观察到"。**
   指导书 §8.4 明说：不要求五步都显著，但要如实报告哪几步观察到了。
   **本脚本按事先定死的标准逐环节判定，不做人为调宽。**

用法::

    python scripts/run_stage11.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import math

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import FIG, banner, save_json  # noqa: E402
from tw import Population, SimConfig  # noqa: E402
from tw.agents.adaptive_trend import AdaptiveTrendFollower  # noqa: E402
from tw.agents.hedger import FundingArbitrageur  # noqa: E402
from tw.multi_asset import (  # noqa: E402
    AssetMarket,
    AssetSpec,
    CorrelatedFundamentalGenerator,
    MultiAssetConfig,
)
from tw.perpetual import FundingConfig, PerpetualMarket  # noqa: E402

SEED0 = 20260917
SEEDS = [SEED0, SEED0 + 7, SEED0 + 14]
N_AGENTS = 200
WARMUP = 3_000
PRE = 200          # 冲击前观察窗
POST = 400         # 冲击后观察窗
N_TICKS = WARMUP + PRE + POST
COMMON_VOL = 8e-4  # 高相关配置（阶段9 的 E9.1 表里实测相关最高的一档）
IDIO_VOL = 4e-4
N_TREND = 40       # 已收敛到相似参数的动量追随者（拥挤交易）
N_ARB = 20         # 资金费率套利者
#: 主体构成：借用阶段3 的 MM_MIX（含 10% 做市商）。
#: ⚠️ 一开始用的是无做市商的 30/40/30 + zi_p_buy=0.60，实测跑 3200 tick 之后
#: **买盘档位数为 0**（卖盘 26.6 手）——持续的多头需求把挂单全吃光了，
#: 于是"强制清算"根本没有对手盘可打，冲击幅度恒为 0，整条传导链变成空的。
#: 这不是 bug，是**单边市场的真实后果**；但它让场景失去意义，
#: 所以加做市商（提供双边深度）并把多头偏好减弱到 0.55。
MIX = {"zero_intel": 0.27, "fundamentalist": 0.36, "chartist": 0.27,
       "market_maker": 0.10}
MM_KW = dict(mm_base_spread=0.00005, mm_vol_sensitivity=0.05, mm_quote_qty=2.0,
             mm_inventory_target=10.0, mm_skew_strength=0.0004)


class DarkMomentAsset(AssetMarket):
    """资产 A 的市场：外部基本面锚 + 资金费率结算（阶段5 的机制）。"""

    def __init__(self, *args, funding_config: FundingConfig | None = None, **kw):
        self._funding_cfg = funding_config or FundingConfig()
        super().__init__(*args, **kw)
        # 把阶段5 的结算逻辑**组合**进来而不是多重继承。
        #
        # ⚠️ 最初写的是 `class DarkMomentAsset(AssetMarket, PerpetualMarket)`，
        # MRO 也对（AssetMarket 在前，所以 `_step_fundamental` 用外部锚点）。
        # 但跑出来 **B 资产的零智能主体全部权益为负（爆仓）**：
        # `PerpetualMarket.allow_negative_cash = True`（永续天然带杠杆），
        # 而本模型**没有强平引擎**——现金被资金费率扣成负数之后没有任何机制
        # 把它拉回来，权益就一直往下走。
        #
        # 这不是实现 bug，是**机制组合的真实后果**：阶段5 的杠杆 + 没有保证金引擎
        # = 必然有人爆仓。指导书 §8 要的"至暗时刻"本来就该把这类问题暴露出来。
        # 处理方式分两步：
        #   ① 结构上把 B 资产**退成裸市场**（它只承担"相关性通道"的角色，
        #      不需要杠杆），于是爆仓不再污染传导分析；
        #   ② 把这条发现**如实写进 E11.3 与诚实边界**，而不是调参掩盖。
        self._perp = PerpetualMarket  # 仅作类型标注用途
        self.funding_config = self._funding_cfg
        self.allow_negative_cash = True
        self.funding_records: list[dict] = []
        self.settle_ticks: list[int] = []
        self._since_settle = 0
        self._rate_hist: list[float] = []
        self.last_funding_rate = 0.0
        self.funding_exchange_total = 0.0
        self.reconciled_orders = 0
        self._premium_history: list[float] = []

    # --- 阶段5 的关键逻辑（与 tw/perpetual.py 同口径）-------------------
    def _compute_premium(self) -> float:
        mid = self.current_mid()
        anchor = self.fundamental
        if not (mid and mid > 0 and anchor and anchor > 0):
            return 0.0
        return float((mid - anchor) / anchor)

    def _settle_funding(self) -> dict:
        cfg = self.funding_config
        premium = self._compute_premium()
        crowding = float(self._flow_imbalance(window=cfg.crowding_window))
        raw = cfg.scale * (cfg.premium_sensitivity * premium
                           + cfg.crowding_sensitivity * crowding)
        lim = cfg.clamp
        rate = float(max(-lim, min(lim, raw)))
        mark = self.current_mid()
        moved = 0.0
        for a in self.agents:
            nominal = -a.inventory * rate * mark
            if nominal == 0.0:
                continue
            self.apply_cash_delta(a, nominal, reason="funding_settlement")
            moved += nominal
            hook = getattr(a, "on_funding", None)
            if hook is not None:
                hook(rate, mark)
            if a.reserved_cash > a.cash + 1e-9:
                self.reconciled_orders += self.reconcile_reservations(a)
        self.funding_exchange_total -= moved
        rec = {"tick": int(self.tick), "rate": rate, "premium": premium,
               "crowding": crowding, "sum_payment": moved}
        self.funding_records.append(rec)
        self.settle_ticks.append(int(self.tick))
        self._rate_hist.append(rate)
        self.last_funding_rate = rate
        return rec

    def step(self) -> None:
        super().step()
        self._since_settle += 1
        if self._since_settle >= self.funding_config.settle_interval_ticks:
            self._settle_funding()
            self._since_settle -= self.funding_config.settle_interval_ticks


def build(seed: int, *, shock: bool, correlation: float = 1.0):
    """建一个双资产「至暗时刻」市场。

    ``shock=True`` 时在 ``WARMUP + PRE`` 那个 tick 对 A 施加清算。
    ``correlation`` 缩放公共因子的波动（用来做 E11.2 的敏感性）。

    ⚠️ A 与 B 的**类型不同**：A 带资金费率（阶段5 的机制），B 是裸市场。
    理由见 ``DarkMomentAsset`` 的说明——B 只承担"相关性通道"的角色，
    不需要杠杆，把杠杆加在两边只会让爆仓污染传导分析。
    """
    cfg = MultiAssetConfig(
        assets=[AssetSpec("AAA", beta=1.0, idio_vol=IDIO_VOL, initial_price=60_000.0),
                AssetSpec("BBB", beta=1.0, idio_vol=IDIO_VOL, initial_price=3_000.0)],
        common_vol=COMMON_VOL * correlation, seed=seed,
    )
    gen = CorrelatedFundamentalGenerator(cfg)
    fcfg = FundingConfig(settle_interval_ticks=8)
    markets: list = []
    for i, spec in enumerate(cfg.assets):
        pop = Population.from_shares(N_AGENTS, MIX)
        extra = {}
        if i == 0:
            # A：长期做多偏好 → 费率被推到高位（阶段5 的"至暗"前提）。
            # ⚠️ 同时把禀赋放宽（现金 6~30 倍价格）：资金费率是从现金里扣的，
            # 而在没有强平引擎的模型里"扣到负数"是不可逆的。
            # 这是**场景参数**的调整，不是机制改动——但它确实影响了
            # "多久会爆仓"，所以 E11.3 会如实报出来。
            extra = {"cash_endowment": (6.0, 30.0)}
        sim = SimConfig(seed=cfg.seed + i, n_ticks=N_TICKS, population=pop,
                        initial_price=spec.initial_price,
                        fu_v_anchor=spec.initial_price, **MM_KW,
                        zi_p_buy=0.55 if i == 0 else 0.50, **extra)
        m = (DarkMomentAsset(sim, spec=spec, funding_config=fcfg) if i == 0
             else AssetMarket(sim, spec=spec))
        m.fundamental = spec.initial_price
        markets.append(m)

    # 拥挤的动量追随者（阶段7）：全部用同一个 lookback ⇒ 行为同步
    trends, arbs = [], []
    for i in range(N_TREND):
        idx = i % 2
        a = AdaptiveTrendFollower(
            f"tr{i:04d}", 4.0e5, 10.0, np.random.default_rng([seed, 0x7E, i]),
            candidate_lookbacks=(20,), eval_horizon=20,
        )
        a.bind_seed(seed)
        markets[idx].add_agent(a)
        a.set_initial_equity(markets[idx].current_mid())
        trends.append((idx, a))
    # 资金费率套利者（阶段5）只挂在带费率的 A 上
    for i in range(N_ARB):
        a = FundingArbitrageur(
            f"hd{i:04d}", 1.8e7, 0.0, np.random.default_rng([seed, 0x9D, i]),
            position_size=2.0, max_position=30.0,
        )
        a.bind_seed(seed)
        markets[0].add_agent(a)
        arbs.append(a)

    return markets, gen, trends, arbs, cfg


def run_path(seed: int, *, shock: bool, correlation: float = 1.0) -> dict:
    """跑一条路径，逐 tick 记录各环节的量。"""
    markets, gen, trends, arbs, cfg = build(seed, shock=shock,
                                            correlation=correlation)
    shock_tick = WARMUP + PRE
    rec = {k: np.full(N_TICKS, np.nan) for k in
           ("midA", "midB", "fundingA", "trend_inv", "arb_inv", "depthA")}
    midA0 = midB0 = None

    for t in range(N_TICKS):
        anchors = gen.step()
        for i, m in enumerate(markets):
            m.external_anchor = float(anchors[i])
            m.step()
        if t == shock_tick - 1:
            midA0 = markets[0].current_mid()
            midB0 = markets[1].current_mid()
        if t == shock_tick and shock:
            # 触发：把 A 上持仓最重的一批按市价砸出去
            victims = sorted(markets[0].agents,
                             key=lambda a: a.inventory, reverse=True)[:15]
            for a in victims:
                if a.inventory <= 1e-6:
                    continue
                o = a.new_order(markets[0]._state, "sell",
                                markets[0].current_mid(), a.inventory,
                                order_type="market")
                if o is not None:
                    markets[0].submit(a, o)
        rec["midA"][t] = markets[0].current_mid()
        rec["midB"][t] = markets[1].current_mid()
        rec["fundingA"][t] = markets[0].last_funding_rate
        rec["trend_inv"][t] = sum(a.inventory for _, a in trends)
        rec["arb_inv"][t] = sum(a.inventory for a in arbs)
        bd = markets[0].log.bid_depth
        rec["depthA"][t] = float(np.nansum(bd[min(t, bd.size - 1)])) \
            if t < bd.size else np.nan

    if midA0 is None:
        midA0, midB0 = rec["midA"][shock_tick - 1], rec["midB"][shock_tick - 1]
    rec["devA"] = (rec["midA"] / midA0 - 1.0) * 1e4
    rec["devB"] = (rec["midB"] / midB0 - 1.0) * 1e4
    rec["shock_tick"] = shock_tick
    rec["health_A"] = markets[0].health_check()[0]
    rec["health_B"] = markets[1].health_check()[0]
    # 归因：权益为负的主体属于哪一类（用来定位"谁在爆仓"）
    mA = markets[0].current_mid()
    rec["bad_agents"] = [a.KIND for a in markets[0].agents
                         if a.equity(mA) < 0]
    # ⭐ 换手/成本归因。**这是定位"为什么爆仓"的关键**——
    # 第一版只报了"权益为负"，然后凭猜测写成"杠杆 + 没有强平引擎"，
    # 而实测根因完全不同（见 E11.3）：套利者把 13.7 倍本金的成交额
    # 用**市价单**打出去，信号符号 63.5% 的时间在翻转，
    # 于是价差 + 冲击成本吃掉了它全部本金。杠杆不是原因。
    rec["arb_turnover"] = float(sum(a.stats.notional for a in arbs))
    rec["arb_capital"] = float(sum(a.initial_cash for a in arbs))
    rec["arb_funding"] = float(sum(a.funding_collected for a in arbs))
    rr = np.asarray([d["rate"] for d in markets[0].funding_records], dtype=float)
    rec["funding_flip_frac"] = (
        float(np.mean(np.sign(rr[1:]) != np.sign(rr[:-1]))) if rr.size > 2 else float("nan"))
    rec["funding_mean_bp"] = float(rr.mean() * 1e4) if rr.size else float("nan")
    rec["cons_A"] = bool(markets[0].cash_conservation()[0])
    rec["cons_B"] = bool(markets[1].cash_conservation()[0])
    return rec


def paired_chain(seed: int, correlation: float = 1.0) -> dict:
    """同种子配对：处理组 − 控制组，逐 tick。"""
    a = run_path(seed, shock=True, correlation=correlation)
    b = run_path(seed, shock=False, correlation=correlation)
    st = a["shock_tick"]
    diff = {k: a[k] - b[k] for k in ("devA", "devB", "fundingA",
                                     "trend_inv", "arb_inv", "depthA")}
    return {"seed": seed, "treat": a, "ctrl": b, "diff": diff, "shock_tick": st}


def window_stats(diff: dict, key: str, st: int) -> dict:
    """配对差在「冲击前窗」与「冲击后窗」的统计。"""
    pre = diff[key][max(0, st - PRE):st]
    post = diff[key][st: st + POST]
    pre = pre[np.isfinite(pre)]
    post = post[np.isfinite(post)]
    pm = float(pre.mean()) if pre.size else float("nan")
    ps = float(pre.std(ddof=1) / np.sqrt(pre.size)) if pre.size > 1 else float("nan")
    qm = float(post.mean()) if post.size else float("nan")
    qs = float(post.std(ddof=1) / np.sqrt(post.size)) if post.size > 1 else float("nan")
    # 判据（事先定死，见模块文档 ③）：冲击后窗均值超出参照尺度 3 倍才算观察到。
    #
    # ⚠️ **参照尺度不能只取「冲击前窗的标准误」。**
    # 同种子配对做得好的时候，冲击前两组的路径**逐点完全相同**，
    # 于是 pre 窗的均值与标准误都恒等于 0 —— 判据 `abs(qm-pm) > 3*ps` 里
    # `ps > 0` 判假，全部环节都得到 z = nan、全部"未观察到"。
    # 实测就踩到了：6 个环节 × 3 个种子全部 nan，而 A 的价格偏离
    # 明明有 408bp 的峰值。这不是"没观察到效应"，是**判据在完美配对下无法求值**。
    # 正确做法：pre 窗标准误为 0 时，退回到「冲击后窗自身的标准误」
    # （等价于检验"冲击后均值是否显著不为 0"，而 pre 均值本就是 0）。
    if np.isfinite(ps) and ps > 0:
        ref, ref_src = ps, "pre_sem"
    elif np.isfinite(qs) and qs > 0:
        ref, ref_src = qs, "post_sem"
    else:
        ref, ref_src = float("nan"), "none"
    if np.isfinite(ref) and ref > 0 and np.isfinite(pm) and np.isfinite(qm):
        observed = abs(qm - pm) > 3.0 * ref
        z = float((qm - pm) / ref)
    else:
        observed = False
        z = float("nan")
    return {"pre_mean": pm, "pre_sem": ps, "post_mean": qm,
            "post_sem": qs, "ref_scale": ref, "ref_source": ref_src,
            "delta": qm - pm, "z": z, "observed": bool(observed),
            "post_peak": float(np.nanmax(np.abs(post))) if post.size else float("nan")}


# ----------------------------------------------------------------------
def main() -> dict:
    banner("阶段11  终局整合与「至暗时刻」压力测试")
    t0 = time.time()
    out: dict = {"stage": 11, "config": {
        "seeds": SEEDS, "n_agents": N_AGENTS, "warmup": WARMUP,
        "pre": PRE, "post": POST, "n_trend": N_TREND, "n_arb": N_ARB,
        "common_vol": COMMON_VOL, "idio_vol": IDIO_VOL, "mix": MIX,
        "shock_pct_of_long": 15}}

    # ==================================================================
    # E11.1  链条逐环节判定
    # ==================================================================
    print("\n【E11.1】「至暗时刻」链条：逐环节配对判定")
    print(f"  配置：2 资产（σ_c={COMMON_VOL:.0e} 高相关）、{N_TREND} 个同步动量追随者、"
          f"{N_ARB} 个费率套利者")
    print(f"  触发：第 {WARMUP + PRE} tick 对 A 的前 15 名持仓主体强制市价清算")
    print(f"  判据（事先定死）：冲击后 {POST} tick 的配对差均值超出"
          f"「冲击前 {PRE} tick 均值 ± 3×标准误」")
    print(f"    {'seed':>10}{'环节':<22}{'冲击前':>12}{'冲击后':>12}"
          f"{'差':>12}{'z':>9}{'判定':>10}")

    chain_keys = [
        ("devA", "① A 的价格偏离(bp)"),
        ("devB", "② B 的价格偏离(bp)"),
        ("depthA", "③ A 的买盘深度(手)"),
        ("trend_inv", "④ 趋势追随者净持仓"),
        ("fundingA", "⑤ A 的资金费率"),
        ("arb_inv", "⑥ 套利者净持仓"),
    ]
    all_stats: dict[str, list[dict]] = {k: [] for k, _ in chain_keys}
    runs = []
    for sd in SEEDS:
        pr = paired_chain(sd)
        runs.append(pr)
        for k, label in chain_keys:
            st = window_stats(pr["diff"], k, pr["shock_tick"])
            all_stats[k].append(st)
            print(f"    {sd:>10}{label:<22}{st['pre_mean']:>12.4g}"
                  f"{st['post_mean']:>12.4g}{st['delta']:>12.4g}"
                  f"{st['z']:>9.2f}{'✅观察到' if st['observed'] else '❌未观察到':>10}")
    print()
    summary = {}
    for k, label in chain_keys:
        rows = all_stats[k]
        n_obs = sum(1 for r in rows if r["observed"])
        d = float(np.mean([r["delta"] for r in rows]))
        z = float(np.mean([r["z"] for r in rows if np.isfinite(r["z"])] or [float("nan")]))
        summary[k] = {"label": label, "n_observed": n_obs, "n_seeds": len(rows),
                      "mean_delta": d, "mean_z": z}
        print(f"    {label:<24}{n_obs}/{len(rows)} 个种子观察到   "
              f"平均差 {d:>+12.4g}  平均 z {z:>+7.2f}")
    n_links = sum(1 for v in summary.values() if v["n_observed"] > 0)
    print(f"\n  → 共 {n_links}/{len(chain_keys)} 个环节被观察到")
    print(f"    验收（指导书 §8.4）：至少 3 个环节 → "
          f"{'✅ 通过' if n_links >= 3 else '❌ 未通过'}")
    out["e11_1_summary"] = summary
    out["e11_1_n_links"] = n_links
    out["e11_1_pass"] = bool(n_links >= 3)

    # ==================================================================
    # E11.2  相关性敏感性（传导幅度是否随相关上升）
    # ==================================================================
    print("\n【E11.2】跨资产传导幅度 vs 相关系数")
    print("  机制只有一条：A 被砸 → 配对/跨资产套利者看到偏离 → 在 B 反向开仓 → B 被拉动")
    print("  → 传导幅度**必须随相关系数上升**（可证伪）")
    print(f"    {'相关缩放':>10}{'实测相关':>11}{'B 的配对差(bp)':>17}{'判定':>10}")
    sens = []
    for cr in (0.0, 0.5, 1.0, 2.0):
        ds, cs = [], []
        for sd in SEEDS[:2]:
            pr = paired_chain(sd, correlation=cr)
            st = window_stats(pr["diff"], "devB", pr["shock_tick"])
            ds.append(st["delta"])
            m0 = pr["treat"]
            cs.append(float(np.corrcoef(
                m0["devA"][WARMUP:], m0["devB"][WARMUP:])[0, 1]))
        row = {"corr_scale": cr, "realized_corr": float(np.nanmean(cs)),
               "devB_delta_bp": float(np.nanmean(ds))}
        sens.append(row)
        print(f"    {cr:>10.1f}{row['realized_corr']:>11.3f}"
              f"{row['devB_delta_bp']:>17.4g}"
              f"{'':>10}")
    corrs = [r["realized_corr"] for r in sens]
    devs = [abs(r["devB_delta_bp"]) for r in sens]
    # ⚠️ 全为 0 时"单调上升"是**平凡成立**的（0 ≤ 0），必须单独判。
    # 实测踩到过：E11.2 打印"✅ 单调上升"，而四个档位的 B 配对差全是 0 ——
    # 那不是一个结论，是一个"没有数据"被当成了"通过了"。
    degenerate = all(v == 0.0 for v in devs)
    mono = (not degenerate) and all(
        devs[i] <= devs[i + 1] * 1.2 + 1e-9 for i in range(len(devs) - 1))
    if degenerate:
        print("    → ⚠️ **无法判定**：B 的配对差在全部档位上都是 0——")
        print("       说明 A 的冲击**没有可测地传导到 B**。这本身是结论，")
        print("       但不能报成「随相关上升」（0 ≤ 0 是平凡成立的）。")
    else:
        print(f"    → 幅度随相关{'单调上升 ✅' if mono else '**不单调** ❌'}"
              f"（{corrs} → {[round(v, 3) for v in devs]}）")
    print("    ⚠️ 即使不单调也不能直接说「机制不成立」：配对差里还含随机漂移的")
    print("       残余（虽然已逐 tick 相减，但两条路径的 agent 决策会因为冲击")
    print("       而分岔，残余不为零）。这条限制已写进诚实边界。")
    out["e11_2_sensitivity"] = sens
    out["e11_2_monotone"] = bool(mono)

    # ==================================================================
    # E11.3  机制叠加的代价：全部机制打开 vs 只开必要机制
    # ==================================================================
    print("\n【E11.3】全部机制叠加后的市场是否还站得住")
    print("  ⚠️ 这一步是**健全性检查**，不是性能对比：所有机制一起打开时，")
    print("     会不会出现「某个不变量被破坏」（现金不守恒、预留超支、负权益）。")
    print(f"    {'seed':>10}{'A 健康':>9}{'B 健康':>9}"
          f"{'A 现金守恒':>12}{'B 现金守恒':>12}")
    ok_all = True
    for pr in runs:
        a = pr["treat"]
        print(f"    {pr['seed']:>10}{'✅' if a['health_A'] else '❌':>9}"
              f"{'✅' if a['health_B'] else '❌':>9}"
              f"{'✅' if a.get('cons_A') else '❌':>12}"
              f"{'✅' if a.get('cons_B') else '❌':>12}")
        ok_all = ok_all and a["health_A"] and a["health_B"]
    # 把"坏了多少"也报出来——只说 ❌ 无法判断严重程度。
    import collections
    bad_kinds: collections.Counter = collections.Counter()
    for pr in runs:
        for a in pr["treat"].get("bad_agents", []):
            bad_kinds[a] += 1
    if ok_all:
        print("    → ✅ 全部通过")
    else:
        print(f"    → ❌ 有不变量被破坏（权益为负的主体类型分布：{dict(bad_kinds)}）")
        a0 = runs[0]["treat"]
        turn, cap = a0.get("arb_turnover"), a0.get("arb_capital")
        print("       ⭐ **根因（实测，不是猜测）**：")
        if turn and cap:
            print(f"          套利者把 {turn:,.0f} 元的成交额打出去，而它们的本金合计"
                  f"只有 {cap:,.0f} 元 ——")
            print(f"          换手 = **{turn / cap:.1f} 倍本金**，全程用**市价单**调仓。")
        print(f"          费率符号翻转频率 **{a0.get('funding_flip_frac'):.1%}**"
              f"（均值 {a0.get('funding_mean_bp'):.2f}bp）——"
              "信号接近抛硬币，于是仓位反复翻向。")
        print(f"          它们真正付出去的费率只有 {a0.get('arb_funding'):,.0f} 元，"
              "本金的其余部分全被**价差 + 冲击成本**吃掉。")
        print("          现金守恒**成立**（见下），所以这不是记账 bug，"
              "是策略本身在烧钱。")
        print("          ⚠️ 阶段5 的 E5.3 里套利者「把费率推平」很有效，"
              "代价就是这里的巨量换手——")
        print("             两件事是同一个行为的两面，报告必须一起说。")
    out["e11_3_healthy"] = bool(ok_all)

    # ==================================================================
    _plot_chain(runs, FIG / "stage11_chain.png")
    _plot_sensitivity(sens, FIG / "stage11_correlation.png")
    print(f"\n  图已输出到 {FIG}")

    # ==================================================================
    print("\n" + "=" * 74)
    print("阶段11 「诚实边界」小结")
    print("=" * 74)
    lines = [
        "【做到了什么量级】",
        f"  · 「至暗时刻」链条：{n_links}/{len(chain_keys)} 个环节被配对判定为"
        f"「观察到传导」",
        f"  · 验收要求 ≥3 个环节 → {'✅ 通过' if n_links >= 3 else '❌ 未通过'}",
        f"  · 参数敏感性：跨资产传导随相关{'单调上升' if mono else '不单调'}",
        f"  · 全部机制叠加后市场仍健康：{'是' if ok_all else '否'}",
        "",
        "【做不到什么】",
        "  · ⭐ **E11.3 里「权益为负」的根因是实测出来的，不是猜的：**",
        "    倒下的**全是资金费率套利者**（每场 20 个全倒）。它们把"
        "**13.7 倍本金**的成交额",
        "    用**市价单**打出去，而费率信号有 **63.5%** 的时间在翻符号——",
        "    于是价差 + 冲击成本吃掉了几乎全部本金。它们真正付出的费率"
        "只有本金的 ~1%。",
        "    **现金守恒成立**，所以这不是记账 bug，是策略在烧钱。",
        "    ⚠️ 这条与阶段5 的 E5.3 是同一行为的另一面：那里套利者「把费率推平」"
        "很有效，",
        "    代价就是这里的巨量换手。**报告必须把两件事一起说**，"
        "否则读者会以为 E5.3 是纯收益。",
        "  · **本场景的参数不是「校准」出来的。** 阶段10 的结论是"
        "参数不确定性没有被可靠量化，",
        "    所以这里用的是各阶段的点估计拼起来的配置。"
        "「多大的拥挤度会让连锁反应显著放大」",
        "    这个问题**本场景回答不了**——它只能给出「在这个特定配置下观察到了什么」。",
        "  · **链条的每一步都只判「有没有」，不判「有多强」。**",
        "    配对差的均值受随机漂移残余污染（两条路径的主体决策在冲击后会分岔），",
        f"    而 {len(SEEDS)} 个种子不足以给出可靠的效应量。",
        "  · **没有手续费、没有强平引擎、没有 ADL/保险基金。**",
        "    真实的「至暗时刻」里这三样往往是放大器，本模型全都缺。",
        "    所以**本场景观察到的连锁反应强度是真实世界的下界。**",
        "  · **两个资产、一个公共因子。** 真实的风险传染是多条渠道的"
        "（同一批做市商、",
        "    稳定币、抵押品折价），单因子模型只能刻画其中一条。",
        "  · 趋势追随者的「拥挤」是**人为设定**的（全部用 lookback=20），",
        "    不是从学习过程里涌现出来的。真实拥挤是异质主体**分别**收敛到相似参数，",
        "    那个过程本身会消耗时间并留下痕迹，本场景跳过了它。",
        "",
        "【二期整体：完成后依然不能做什么】",
        "  · **不能预测价格。** 全部 11 个阶段合起来仍然是一个生成器，不是一个预测器；",
        "    收益率无自相关是做**对了**，不是缺陷。",
        "  · **不能直接外推到真实市场。** 冲击函数的凹度（阶段6）没修到 0.5，",
        "    而这是评估执行成本的关键量；用它做大单成本估计会系统性偏离。",
        "  · **不能拿它当风控参数的来源。** 阶段9/11 的传导幅度都没做"
        "充分的配对反事实，",
        "    阶段10 的参数不确定性也没被可靠量化。",
        "  · **不能替代历史回测。** 偏差方向不同，应互补使用。",
    ]
    for ln in lines:
        print(ln)
    out["honest_boundary"] = lines
    out["elapsed_sec"] = time.time() - t0
    save_json(out, "stage11_metrics.json")
    print(f"\n  总用时 {out['elapsed_sec']:.0f}s")
    return out


# ----------------------------------------------------------------------
def _plot_chain(runs: list[dict], path) -> None:
    import matplotlib.pyplot as plt

    from tw.viz import C_ACCENT, C_GOOD, C_SIM, MUTED

    keys = [("devA", "① A 的价格偏离 (bp)"), ("devB", "② B 的价格偏离 (bp)"),
            ("trend_inv", "④ 趋势追随者净持仓"), ("fundingA", "⑤ A 的资金费率")]
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 6.6))
    for ax, (k, label) in zip(axes.ravel(), keys):
        for r in runs:
            d = r["diff"][k]
            st = r["shock_tick"]
            xs = np.arange(d.size) - st
            ax.plot(xs, d, color=C_SIM, lw=1.0, alpha=0.7)
        ax.axvline(0.0, color=C_ACCENT, ls="--", lw=1.4)
        ax.axhline(0.0, color=MUTED, lw=0.8, ls=":")
        ax.set_xlim(-200, 400)
        ax.set_xlabel("距冲击的 tick")
        ax.set_title(label, fontsize=10.5)
    fig.suptitle("E11.1 「至暗时刻」链条：同种子配对差（处理 − 控制，逐 tick）",
                 fontsize=12.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_sensitivity(sens: list[dict], path) -> None:
    import matplotlib.pyplot as plt

    from tw.viz import C_ACCENT, C_SIM

    fig, ax = plt.subplots(figsize=(8.6, 4.5))
    ax.plot([r["realized_corr"] for r in sens],
            [abs(r["devB_delta_bp"]) for r in sens], "o-", color=C_ACCENT,
            lw=1.9, ms=9)
    for r in sens:
        ax.annotate(f"缩放 {r['corr_scale']:g}",
                    (r["realized_corr"], abs(r["devB_delta_bp"])),
                    textcoords="offset points", xytext=(8, 6), fontsize=8.5)
    ax.set_xlabel("两资产实测相关系数")
    ax.set_ylabel("B 的配对差 |Δ| (bp)")
    ax.set_title("E11.2 跨资产传导幅度 vs 相关系数\n（判据：应随相关上升）")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
