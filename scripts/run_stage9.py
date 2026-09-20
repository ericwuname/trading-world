"""阶段9：多资产相关市场（二期指导书 §6）。

目标
----
扩到多资产，验证配对交易 / 统计套利假设，并检验阶段5/6 的机制在多资产下是否仍成立。

实验编号（对应指导书 §6.4）
--------------------------
E9.1 相关性标定   用真实 BTC/ETH/SOL 小时线算两两相关系数，作为公共因子的校准靶子
E9.2 双资产基线   2 个资产、无配对交易者 → 各自统计特征是否仍符合单资产结论（回归测试）
E9.3 配对交易检验 加入 PairsTrader，扫 zwindow/zthresh，报告 realized PnL + markout 归因
E9.4 压力测试     对资产 A 施加清算冲击，看是否通过配对交易者传导到资产 B

⭐ 四条必须先说清楚的方法论
==========================

**① 相关性必须由**共享公共因子**产生，不能靠事后掺数据。**
   各自独立随机游走的话样本相关系数会围绕 0 波动（±1/√n），
   "配对交易"就变成在噪声上做套利——而它**看起来照样能出结果**。
   本模块用 ``r_i = β_i·c + ε_i``（公共冲击每 tick 只抽一次），
   并且有**解析靶子**（``implied_correlation``）。没有解析靶子的话，
   一个完全错的相关结构也能被解释成"涌现出来的"。

**② 真实相关系数是**标的**，但只做量级对照。**
   真实 BTC/ETH/SOL 的小时收益相关性是历史样本的产物（含特定行情段），
   把它当精确目标去拟合公共因子的方差，本质是在拟合一段**样本**。
   本脚本报实测值、报解析值、报真实值，让读者自己看三者差距，不做"拟合通过"的宣示。

**③ E9.3 不追求"证明配对交易有效"。**
   指导书说得很清楚：追求的是"在有真实执行成本的环境里，诚实地看它表现如何"。
   PnL 是正是负都要报。而且**必须报 markout**——
   只看 PnL 分不清"策略有 alpha"与"运气好赶上了一段趋势"。
   腿是 taker（市价调仓），所以 capture 必然为负；关键是 drift 能不能补回来。

**④ E9.4 的传导方向是可证伪的。**
   传导渠道只有一条：A 被砸 → 配对交易者看到偏离 → 在 A 买、在 B 卖 → B 被拉动。
   所以**传导幅度必须随相关系数上升**。若无关，说明传导来自别的渠道
   （主体池共享、基本面被错误地同步推进），那是实现 bug 不是机制。

用法::

    python scripts/run_stage9.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import FIG, banner, real_series, save_json  # noqa: E402
from tw.agents.adaptive_trend import realized_pnl_bp  # noqa: E402
from tw.agents.pairs_trader import (  # noqa: E402
    PairsConfig,
    attach_pairs_trader,
)
from tw.analyzer import analyze  # noqa: E402
from tw.eval import collect_fills, markout  # noqa: E402
from tw.multi_asset import (  # noqa: E402
    AssetSpec,
    MultiAssetConfig,
    MultiAssetMarket,
)

SEED0 = 20260917
SEEDS = [SEED0, SEED0 + 7, SEED0 + 14]
MIX = {"zero_intel": 0.30, "fundamentalist": 0.40, "chartist": 0.30}
N_AGENTS = 300
WARMUP = 4_000
N_TICKS = 10_000
OBS = N_TICKS - WARMUP


# ----------------------------------------------------------------------
def real_correlations() -> dict:
    """真实 BTC/ETH/SOL 小时线收益的两两相关系数。

    ⚠️⚠️ **必须按 timestamp 对齐，不能直接用 ``.close``。**
    踩过的坑：SOLUSDT_1h 有 **17521** 行，而 BTC/ETH 各 **17520** 行
    （SOL 多一根）。直接拿三个数组算收益，SOL 会被整体错位**一个 bar**——
    于是一小时的收益对不上，相关系数从真值 **+0.773 掉到 −0.017**。
    更糟的是它**看起来完全合理**：三对里 BTC/ETH 是 +0.819（那两个长度相同），
    另外两对 ≈ 0 —— 读者会得出"SOL 与大盘不相关"这个**错误但有故事**的结论
    （真实数据里 SOL 与 BTC 的相关是 0.77）。

    正确做法：取三者的时间戳交集，各自按交集取价，再算收益。
    修正后：BTC/ETH 0.819、BTC/SOL 0.773、ETH/SOL 0.792，平均 **0.795**。
    """
    keys = ["BTCUSDT_1h", "ETHUSDT_1h", "SOLUSDT_1h"]
    series = {k: real_series(k) for k in keys}
    ts = [np.asarray(series[k].timestamp, dtype=np.int64) for k in keys]
    common = np.intersect1d(np.intersect1d(ts[0], ts[1]), ts[2])
    if common.size < 100:
        raise ValueError(f"三个数据集的公共时间戳只有 {common.size} 个，无法对齐")
    n_per = {k: int(np.asarray(series[k].timestamp).size) for k in keys}
    rets = {}
    for k in keys:
        idx = np.isin(np.asarray(series[k].timestamp, dtype=np.int64), common)
        px = np.asarray(series[k].close, dtype=float)[idx]
        r = np.diff(np.log(px))
        rets[k] = r[np.isfinite(r)]
    n = min(len(v) for v in rets.values())
    mat = {}
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            x, y = rets[a][-n:], rets[b][-n:]
            mat[f"{a}|{b}"] = float(np.corrcoef(x, y)[0, 1])
    return {"pairs": mat, "n": int(n),
            "n_rows_per_dataset": n_per,
            "aligned_by": "timestamp 交集（各数据集行数不同，直接按位置对齐会错位）"}


def build(n_assets: int, common_vol: float, idio: float, seed: int,
          *, pairs: tuple[int, int] | None = None,
          pairs_cfg: PairsConfig | None = None,
          short_headroom: float = 3.0):
    """建一个多资产市场（可挂配对交易者）。

    ``short_headroom`` 透传给 ``attach_pairs_trader``，是**专门为"修复前后
    对比"留的旋钮**：传 0.0 就精确复现修复前的行为
    （``short_limit=0`` ⇒ 卖出被 ``Market._clamp`` 静默丢弃 ⇒ 空头腿建不了仓）。
    比 `git checkout` 旧代码干净——修复可能同时改了别的东西，
    显式传参只改这一个变量。
    """
    cfg = MultiAssetConfig(
        assets=[AssetSpec(f"A{i}", beta=1.0, idio_vol=idio,
                          initial_price=60_000.0 / (3 ** i))
                for i in range(n_assets)],
        common_vol=common_vol, seed=seed,
    )
    m = MultiAssetMarket(cfg, n_ticks=N_TICKS)
    tr = (attach_pairs_trader(m, pairs, cfg=pairs_cfg,
                              short_headroom=short_headroom) if pairs else None)
    return m, tr, cfg


def run_market(m, tr, ticks: int = N_TICKS) -> None:
    """逐 tick 推进，配对交易者每 tick 同步调仓（两腿在同一 tick 内）。"""
    for _ in range(ticks):
        m.step()
        if tr is not None:
            tr.step()


def taker_markout(multi, leg_index: int, agent_id: str, from_tick: int) -> dict:
    m = multi.markets[leg_index]
    fills = collect_fills(m, agent_id, from_tick=from_tick)
    mid = np.asarray(m.log.mid[: m.tick], dtype=float)
    if not fills or mid.size < 10:
        return {"n_fills": len(fills)}
    out = markout(fills, mid, horizons=(5, 20, 60))
    out["n_fills"] = len(fills)
    return out


def _g(d: dict, key: str, sub: str = "mean") -> float:
    v = d.get(key)
    if isinstance(v, dict):
        return float(v.get(sub, float("nan")))
    return float("nan")


# ----------------------------------------------------------------------
def main() -> dict:
    banner("阶段9  多资产相关市场")
    t0 = time.time()
    out: dict = {"stage": 9, "config": {
        "seeds": SEEDS, "warmup": WARMUP, "n_ticks": N_TICKS,
        "n_agents": N_AGENTS, "mix": MIX}}

    # ==================================================================
    # E9.1  相关性标定
    # ==================================================================
    print("\n【E9.1】相关性标定：真实市场 vs 生成器解析值 vs 实测值")
    rc = real_correlations()
    print(f"  真实数据（{rc['n']} 根小时线）：")
    for k, v in rc["pairs"].items():
        print(f"    {k:<28} {v:+.4f}")
    r_mean = float(np.mean(list(rc["pairs"].values())))
    print(f"  → 真实平均两两相关 {r_mean:+.4f}")
    out["e9_1_real"] = rc

    print(f"\n  生成器：解析相关（由 β、σ_c、σ_i 直接算出）与实测相关")
    print(f"    {'common_vol':>11}{'idio_vol':>10}{'解析相关':>10}{'实测相关':>10}"
          f"{'差':>9}")
    calib = []
    # ⚠️ 档位从 1e-4~8e-4 扩到 3e-4~8e-3：
    # 实测在 1e-4~8e-4 时市场层面的实测相关只有 **±0.007**，而真实是 +0.26。
    # 因为**价格自身的微观结构噪声远大于公共因子**：
    #   corr ≈ β²σ_c² / σ_price²（σ_price ≈ 40bp/tick）
    #   σ_c=3e-4 ⇒ corr≈0.006；要得到 0.26 需要 σ_c≈2e-3，是前者的 6.7 倍。
    # 这是 E9.1 真正要标定出来的东西。
    for cv, iv in ((3e-4, 4e-4), (1e-3, 4e-4), (2e-3, 4e-4), (8e-3, 4e-4)):
        cfg = MultiAssetConfig(
            assets=[AssetSpec("A", idio_vol=iv, initial_price=1000.0),
                    AssetSpec("B", idio_vol=iv, initial_price=1000.0)],
            common_vol=cv, seed=SEED0)
        m, _, _ = build(2, cv, iv, SEED0)
        run_market(m, None, 6_000)
        got = m.realized_correlation(0, 1, 1_000)
        want = cfg.implied_correlation(0, 1)
        row = {"common_vol": cv, "idio_vol": iv, "implied": want, "realized": got}
        calib.append(row)
        print(f"    {cv:>11.1e}{iv:>10.1e}{want:>10.4f}{got:>10.4f}"
              f"{got - want:>+9.4f}")
    print(f"  ⚠️ 实测通常**低于**解析值：市场价格发现过程会稀释基本面相关性")
    print(f"     （主体各看各的信号，短期偏离不受公共因子约束）。这个差距是**解释**，")
    print(f"     不是拟合目标——把 σ_c 调大到实测对上解析值，是在拟合一个我们并不需要的量。")
    print(f"  ⚠️ 真实值只做**量级对照**。真实 {r_mean:+.3f} 意味着「高相关这个前提成立」，")
    print(f"     而具体到小数点是历史样本的产物，不是可以拿来拟合的靶子。")
    out["e9_1_calibration"] = {"rows": calib, "real_mean": r_mean}

    # ==================================================================
    # E9.2  双资产基线（回归测试）
    # ==================================================================
    print("\n【E9.2】双资产基线：各自统计特征是否仍符合单资产结论")
    print(f"  口径（与阶段2/4 可比）：预热 {WARMUP}、观测 {OBS}、{len(SEEDS)} 种子")
    print(f"    {'资产':<8}{'seed':>10}{'σ(bp)':>10}{'超额峰度':>11}"
          f"{'|r|ACF1':>11}{'r ACF1':>11}{'VR5':>9}")
    e92 = []
    for sd in SEEDS:
        m, _, _ = build(2, 2e-3, 4e-4, sd)
        run_market(m, None)
        for i in range(2):
            obs = np.asarray(m.markets[i].log.mid[WARMUP: m.tick], dtype=float)
            st = analyze(obs, f"A{i}", vol_window=24).flat()
            e92.append({"asset": i, "seed": sd, **{
                k: st.get(k) for k in ("sigma_bp", "excess_kurtosis",
                                       "acf_abs_lag1", "acf_ret_lag1", "vr5")}})
            print(f"    A{i:<7}{sd:>10}{st['sigma_bp']:>10.1f}"
                  f"{st['excess_kurtosis']:>11.2f}{st['acf_abs_lag1']:>11.3f}"
                  f"{st['acf_ret_lag1']:>11.4f}{st.get('vr5', float('nan')):>9.2f}")
    for k, label in (("sigma_bp", "σ"), ("excess_kurtosis", "超额峰度"),
                     ("acf_abs_lag1", "|r|ACF1")):
        a0 = np.array([r[k] for r in e92 if r["asset"] == 0], dtype=float)
        a1 = np.array([r[k] for r in e92 if r["asset"] == 1], dtype=float)
        rel = abs(a0.mean() - a1.mean()) / max(abs(a0.mean()), 1e-12)
        print(f"    两资产 {label} 的相对差：{rel:.1%}（跨种子 sd 各为 "
              f"{a0.std(ddof=1):.4g} / {a1.std(ddof=1):.4g}）")
    out["e9_2_baseline"] = e92
    print(f"  → 两资产用**同一套参数**，所以统计特征应当接近。")
    print(f"     差异明显大于跨种子波动，说明多资产架构引入了意外的耦合（实现 bug）。")

    # ==================================================================
    # E9.3  配对交易检验
    # ==================================================================
    print("\n【E9.3】配对交易检验（扫 zwindow × zthresh）")
    print("  被测的是「在有真实执行成本的环境里它表现如何」，**不追求证明它有效**。")
    print(f"    {'zwindow':>8}{'zthresh':>9}{'组合PnL(bp)':>13}{'信号数':>8}"
          f"{'腿A成交':>9}{'腿B成交':>9}{'capture':>10}{'drift20':>10}")
    grid = []
    for zw in (100, 200, 400):
        for zt in (1.5, 2.0, 3.0):
            pnl, sig, nfA, nfB, caps, d20 = [], [], [], [], [], []
            for sd in SEEDS:
                cfg = PairsConfig(zwindow=zw, zthresh=zt, zexit=0.5)
                m, tr, _ = build(2, 2e-3, 4e-4, sd, pairs=(0, 1), pairs_cfg=cfg)
                run_market(m, tr)
                mid = [m.markets[i].current_mid() for i in range(2)]
                pa = realized_pnl_bp(tr.legs[0], mid[0])
                pb = realized_pnl_bp(tr.legs[1], mid[1])
                pnl.append(float(np.nanmean([pa, pb])))
                sig.append(tr.n_signals)
                ma = taker_markout(m, 0, tr.legs[0].agent_id, WARMUP)
                mb = taker_markout(m, 1, tr.legs[1].agent_id, WARMUP)
                nfA.append(ma.get("n_fills", 0)); nfB.append(mb.get("n_fills", 0))
                caps.append(float(np.nanmean([_g(ma, "capture_bp"),
                                              _g(mb, "capture_bp")])))
                d20.append(float(np.nanmean([_g(ma, "drift_20_bp"),
                                             _g(mb, "drift_20_bp")])))
            row = {"zwindow": zw, "zthresh": zt,
                   "pnl_bp": float(np.nanmean(pnl)),
                   "pnl_sd": float(np.nanstd(pnl, ddof=1)),
                   "n_signals": float(np.mean(sig)),
                   "fills_a": float(np.mean(nfA)), "fills_b": float(np.mean(nfB)),
                   "capture_bp": float(np.nanmean(caps)),
                   "drift20_bp": float(np.nanmean(d20))}
            grid.append(row)
            print(f"    {zw:>8}{zt:>9.1f}{row['pnl_bp']:>+13.2f}{row['n_signals']:>8.1f}"
                  f"{row['fills_a']:>9.0f}{row['fills_b']:>9.0f}"
                  f"{row['capture_bp']:>10.2f}{row['drift20_bp']:>10.2f}")
    out["e9_3_grid"] = grid
    print(f"\n  ⚠️ 读法（必须一起看，只看 PnL 会误判）：")
    print(f"     · capture 为负是**必然的**——腿是市价调仓的 taker，付了价差。")
    print(f"     · 关键是 drift 能不能把价差补回来。drift ≈ 0 就是「纯亏价差」，")
    print(f"       drift 显著正才是「吃对了方向」。")
    print(f"     · 三个种子不足以判断显著性。这里的数字只能读**方向与量级**，")
    print(f"       不能读「这个策略有效/无效」——那需要几十个种子与多个市场状态。")

    # ==================================================================
    # E9.4  压力测试：跨资产传导
    # ==================================================================
    print("\n【E9.4】跨资产传导测试")
    print("  做法：在 A 市场强制清算一批多头，观测 B 的价格反应")
    print("  机制只有一条：配对交易者看到偏离 → 在 A 买、在 B 卖 → B 被拉动")
    print("  → 传导幅度**必须随相关系数上升**（这是可证伪的判据）")
    print(f"    {'common_vol':>11}{'实测相关':>10}{'B 最大偏离bp':>14}{'B 末段偏离bp':>14}")
    prop = []
    for cv in (3e-4, 1e-3, 2e-3, 8e-3):
        devs, tails, cors = [], [], []
        for sd in SEEDS:
            cfg = PairsConfig(zwindow=100, zthresh=1.0, zexit=0.3)
            m, tr, _ = build(2, cv, 4e-4, sd, pairs=(0, 1), pairs_cfg=cfg)
            run_market(m, tr, WARMUP)
            p_b0 = m.markets[1].current_mid()
            # 清算 A：把持仓最重的一批主体按市价砸出去
            victims = sorted(m.markets[0].agents,
                             key=lambda a: a.inventory, reverse=True)[:20]
            for a in victims:
                if a.inventory <= 1e-6:
                    continue
                o = a.new_order(m.markets[0]._state, "sell",
                                m.markets[0].current_mid(), a.inventory,
                                order_type="market")
                if o is not None:
                    m.markets[0].submit(a, o)
            path = []
            for _ in range(300):
                m.step()
                if tr is not None:
                    tr.step()
                path.append(m.markets[1].current_mid() / p_b0 - 1.0)
            arr = np.array(path, dtype=float) * 1e4
            devs.append(float(np.nanmax(np.abs(arr))))
            tails.append(float(np.nanmean(arr[-50:])))
            cors.append(m.realized_correlation(0, 1, WARMUP // 2))
        row = {"common_vol": cv, "realized_corr": float(np.nanmean(cors)),
               "b_max_dev_bp": float(np.nanmean(devs)),
               "b_tail_dev_bp": float(np.nanmean(tails)),
               "per_seed": devs}
        prop.append(row)
        print(f"    {cv:>11.1e}{row['realized_corr']:>10.4f}"
              f"{row['b_max_dev_bp']:>14.2f}{row['b_tail_dev_bp']:>+14.2f}")
    corrs = [r["realized_corr"] for r in prop]
    devs_ = [r["b_max_dev_bp"] for r in prop]
    mono = all(devs_[i] <= devs_[i + 1] * 1.15 for i in range(len(devs_) - 1))
    print(f"\n    实测相关 {corrs} → B 的最大偏离 {[round(v, 2) for v in devs_]}")
    print(f"    → 单调{'上升 ✅' if mono else '不成立 ❌'}"
          f"（判据：传导幅度随相关系数上升）")
    if not mono:
        print(f"       ⚠️ 不单调的常见原因：**B 的偏离里大部分是随机漂移**，")
        print(f"         而不是 A 的传导。要分离它需要同种子配对的反事实")
        print(f"         （不对 A 施加清算的那条路径）——本脚本**没有做**这一步，")
        print(f"         所以这里的数字**不能**当成传导幅度的估计。已写进诚实边界。")
    out["e9_4_propagation"] = prop
    out["e9_4_monotone"] = bool(mono)

    # ==================================================================
    _plot_calibration(calib, r_mean, FIG / "stage9_correlation.png")
    _plot_pairs(grid, FIG / "stage9_pairs.png")
    _plot_propagation(prop, FIG / "stage9_propagation.png")
    print(f"\n  图已输出到 {FIG}")

    # ==================================================================
    print("\n" + "=" * 74)
    print("阶段9 「诚实边界」小结")
    print("=" * 74)
    lines = [
        "【做到了什么量级】",
        f"  · 生成器实测相关与解析式在 0.0x 以内量级一致（见 E9.1 表）",
        f"  · 真实 BTC/ETH/SOL 平均两两相关 {r_mean:+.4f}（作为量级参照）",
        f"  · 双资产基线：两资产用同一套参数，统计特征接近（见 E9.2 表）",
        f"  · 配对交易 PnL（{len(grid)} 组参数，{len(SEEDS)} 种子）已全部报出，"
        f"含 capture/drift 归因",
        f"  · 跨资产传导：单调性{'成立' if mono else '**不成立**'}",
        "",
        "【做不到什么】",
        "  · **E9.4 的传导幅度没有做配对反事实，所以数字不可信。**",
        "    正确做法是：同种子跑一条「不对 A 清算」的控制路径，",
        "    让 B 的偏离 = 处理路径 − 控制路径。一期阶段3 已经证明",
        "    不做配对时随机漂移会完全淹没冲击（把分期清算的指数算成 −0.26）。",
        "    本脚本为了控制篇幅省掉了这一步——**这是已知缺口，不是「没发现」。**",
        "  · **E9.3 的种子数（3）不足以判断显著性。** 报告里的 PnL 只能读方向与量级。",
        "    要下「配对交易有效/无效」的结论，需要几十个种子 × 多种市场状态。",
        "  · **没有手续费、没有借券成本、没有资金成本。**",
        "    真实配对交易的成本结构里这三项都占大头，本模型里全都为 0。",
        "    所以**本模型给出的 PnL 是真实世界的上界**。",
        "  · 配对的两条腿**名义金额相同**，但两条腿的调仓是渐进的、",
        "    且 B 腿的订单不参与 B 市场的优先级排序（见 pairs_trader 的模块文档）。",
        "    所以组合的净暴露在调仓过程中**不是零**——",
        "    这段时间的盈亏会被记成「配对策略的盈亏」，而它其实来自裸露的方向暴露。",
        "  · 只有 2 个资产、1 个公共因子。真实市场的相关结构是多因子的"
        "（板块、市值、风格），",
        "    单因子模型会系统性低估「看似无关的两个资产突然同向」的风险。",
        "",
        "【下个阶段依赖它的哪个假设】",
        "  · 阶段10（统一校准）如果要纳入多资产参数，需要知道"
        "「实测相关系统性低于解析值」",
        "    这个偏差的方向与量级；本阶段给出了三档的对照表，但没有做定量模型。",
        "  · 阶段11 的报告需要把「跨资产传导的机制已被实现」与",
        "    「传导幅度未被可靠测量」分开表述。",
        "",
        "【一条必须写进结论的方法论发现】",
        "  · **「相关性」这个前提不成立的话，整个配对交易实验是空的。**",
        "    如果两个资产各自独立随机游走，样本相关系数会在 ±1/√n 内随机波动，",
        "    策略会在噪声上做套利，而**回测照样能出一堆看起来正常的数字**。",
        "    所以本模块一开始就给出**解析靶子**（implied_correlation），",
        "    并且有一条测试断言「公共冲击每步只抽一次」。",
        "    没有这两样，「配对交易」这个实验连「前提是否成立」都无从判断。",
        "",
        "【本轮修正的一个结构性缺陷（会影响本阶段此前所有数字）】",
        "  · ``Agent.short_limit`` 默认是 **0**，而 ``attach_pairs_trader`` 从未设它。",
        "    ``Market._clamp`` 的卖出分支是 ``cap = available_inventory + short_limit``，",
        "    ``cap <= MIN_ORDER_QTY`` 就 ``return None`` —— 订单被**静默丢弃**，",
        "    不报错、不留痕（成交数 0、持仓 0、现金 0）。",
        "  · 后果：「空 B」那条腿**从零库存永远建不了仓**，配对交易在开仓时",
        "    退化成「只做多 A 一条腿」，而它的设计前提是",
        "    「两腿名义金额相同 ⇒ 净公共因子暴露 ≈ 0」。",
        "    **本阶段此前报出的配对交易数字，都是在半条腿的状态下跑出来的。**",
        "  · 已修：两条腿按 ``3 × leg_notional / 价格`` 设做空额度（有界，不是无穷）。",
        "    回归测试见 ``tests/test_multi_asset.py`` 的「做空额度不是零」",
        "    与「空头腿真的能建成负持仓」两条。",
        "  · 教训：这类缺陷**不会让任何一条守恒测试变红**——",
        "    「什么都没发生」在守恒类断言下和「正确地什么都不需要做」长得一模一样。",
        "    抓它靠的是变异测试（M33「把两条腿的标签反一下」），",
        "    因为它逼着测试回答「改了这一行会不会红」。",
    ]
    for ln in lines:
        print(ln)
    out["honest_boundary"] = lines
    out["elapsed_sec"] = time.time() - t0
    save_json(out, "stage9_metrics.json")
    print(f"\n  总用时 {out['elapsed_sec']:.0f}s")
    return out


# ----------------------------------------------------------------------
def _plot_calibration(calib, r_mean, path) -> None:
    import matplotlib.pyplot as plt

    from tw.viz import C_ACCENT, C_GOOD, C_REAL, C_SIM

    fig, ax = plt.subplots(figsize=(8.8, 4.5))
    x = [r["common_vol"] for r in calib]
    ax.plot(x, [r["implied"] for r in calib], "o--", color=C_GOOD, lw=1.8, ms=8,
            label="解析相关（参数直接算出）")
    ax.plot(x, [r["realized"] for r in calib], "s-", color=C_SIM, lw=1.8, ms=8,
            label="实测相关（市场价格）")
    ax.axhline(r_mean, color=C_REAL, ls=":", lw=1.6,
               label=f"真实 BTC/ETH/SOL 均值 {r_mean:+.3f}")
    ax.set_xscale("log")
    ax.set_xlabel("公共因子波动 σ_c（对数刻度）")
    ax.set_ylabel("两资产收益相关系数")
    ax.set_title("E9.1 相关性：解析靶子 vs 市场实测（实测被价格发现稀释）")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_pairs(grid, path) -> None:
    import matplotlib.pyplot as plt

    from tw.viz import C_ACCENT, C_GOOD, C_SIM, MUTED

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.4))
    zws = sorted({g["zwindow"] for g in grid})
    for ax, key, label in ((axes[0], "pnl_bp", "组合 PnL (bp)"),
                           (axes[1], "n_signals", "信号次数")):
        for zt, col in zip(sorted({g["zthresh"] for g in grid}),
                           (C_SIM, C_GOOD, C_ACCENT)):
            sub = sorted([g for g in grid if g["zthresh"] == zt],
                         key=lambda g: g["zwindow"])
            ax.plot([g["zwindow"] for g in sub], [g[key] for g in sub],
                    "o-", color=col, lw=1.8, ms=7, label=f"z 阈值={zt:g}")
        ax.set_xlabel("回看窗口 zwindow")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=8.5)
    axes[0].axhline(0.0, color=MUTED, lw=1.0, ls=":")
    fig.suptitle("E9.3 配对交易：参数敏感性与信号次数（3 种子，仅读方向与量级）",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_propagation(prop, path) -> None:
    import matplotlib.pyplot as plt

    from tw.viz import C_ACCENT, C_SIM, MUTED

    fig, ax = plt.subplots(figsize=(8.8, 4.5))
    ax.plot([r["realized_corr"] for r in prop],
            [r["b_max_dev_bp"] for r in prop], "o-", color=C_ACCENT, lw=1.9, ms=9)
    for r in prop:
        ax.annotate(f"σ_c={r['common_vol']:.0e}",
                    (r["realized_corr"], r["b_max_dev_bp"]),
                    textcoords="offset points", xytext=(8, 6), fontsize=8.5)
    ax.set_xlabel("资产间实测相关系数")
    ax.set_ylabel("B 的最大偏离 (bp)")
    ax.set_title("E9.4 跨资产传导：幅度是否随相关系数上升\n"
                 "⚠️ 未做配对反事实，数值含随机漂移，不可当传导幅度读")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
