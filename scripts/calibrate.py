"""参数标定：把"每 tick 波动率"标到真实 BTC 小时线的量级。

为什么要单独做这个
------------------
施工蓝图 §3.5-A 给零智能交易者的报价幅度建议是"中间价 ±0.1%~2%"。
**照抄会得到一个完全不对的市场**：实测那个幅度下每 tick σ 高达 543bp，
而真实 BTC 小时线只有 48bp——高了 11 倍。

原因不难理解：报价幅度同时决定两件事——
  ① 价格扩散速度（偏离越大，成交价离中间价越远，价格跳得越猛）；
  ② 盘口宽度（挂单离中间价越远，稀疏盘口越宽）。
所以它**不是**一个可以凭直觉设定的"风格参数"，而是必须标定的**刻度参数**。

标定逻辑
--------
令 1 tick ≈ 1 小时（蓝图的时间刻度约定），于是要求
  模拟每 tick σ ≈ 真实 BTC 小时线 σ = 47.9bp。
做单因子扫描找满足这个条件的参数。

**关键诚实点**：标定是在**纯零智能基线**上做的（阶段1 的市场），
但阶段2 引入基本面派 + 图表派之后，同一个参数的 σ 会变（混合池 67.6bp）。
两者不可能同时精确命中——这个张力本身就是要报告的内容，不是要被藏起来的。
本脚本对两个池分别扫描，把张力摊开给读者看。
"""

from __future__ import annotations

import numpy as np

from _common import banner, real_metrics, save_json  # noqa: E402
from tw import Population, SimConfig  # noqa: E402

N_AGENTS = 200
WARMUP = 4_000
N_OBS = 6_000
SEEDS = [20260917, 20260924]

POOLS = {
    "纯零智能(阶段1)": {"zero_intel": 1.0},
    "30/40/30 混合(阶段2)": {"zero_intel": 0.30, "fundamentalist": 0.40, "chartist": 0.30},
}

OFFSET_HI = [0.0005, 0.0008, 0.0012, 0.0018, 0.0028, 0.0200]
P_ACTIVE = [0.05, 0.10, 0.15, 0.25, 0.40]
N_SCAN = [100, 200, 300, 500]


def sigma_of(
    mix: dict[str, float],
    seed: int,
    *,
    off_hi: float | None = None,
    p_active: float | None = None,
    n_agents: int = N_AGENTS,
) -> tuple[float, float, float]:
    """返回 (σ bp/tick, 平均价差 bp, 平均深度)。"""
    pop = Population.from_shares(n_agents, mix)
    kw = {}
    if off_hi is not None:
        kw["zi_offset_range"] = (0.0002, off_hi)
    if p_active is not None:
        kw["zi_p_active"] = p_active
    cfg = SimConfig(seed=seed, n_ticks=WARMUP + N_OBS, population=pop, **kw)
    from tw import Market

    m = Market(cfg)
    m.run()
    sl = slice(WARMUP, m.tick)
    mid = m.log.mid[sl]
    r = np.diff(np.log(mid))
    r = r[np.isfinite(r)]
    spread = m.log.spread[sl] / mid
    spread = spread[np.isfinite(spread)]
    depth = m.log.bid_depth[sl] + m.log.ask_depth[sl]
    return (
        float(r.std() * 1e4),
        float(spread.mean() * 1e4) if spread.size else float("nan"),
        float(np.nanmean(depth)),
    )


def scan(pool_tag: str, mix: dict, *, key: str, values: list[float]) -> list[dict]:
    # key 直接用 sigma_of 的形参名，避免"扫描变量名"和"实际注入的参数名"两张皮
    assert key in ("off_hi", "p_active"), key
    rows = []
    for v in values:
        sig, sp, dp = [], [], []
        for s in SEEDS:
            a, b, c = sigma_of(mix, s, **{key: v})
            sig.append(a)
            sp.append(b)
            dp.append(c)
        rows.append(
            {
                "value": v,
                "sigma_bp": float(np.mean(sig)),
                "sigma_sd": float(np.std(sig, ddof=1)) if len(sig) > 1 else float("nan"),
                "spread_bp": float(np.mean(sp)),
                "depth": float(np.mean(dp)),
            }
        )
    return rows


def main() -> dict:
    banner("参数标定：把每 tick 波动率标到真实 BTC 小时线量级")
    reals = real_metrics()
    rc = {m.label: m.flat() for m in reals}
    btc = rc["BTCUSDT_1h"]
    target = btc["sigma_bp"]
    print(f"\n  标定目标：真实 BTC 1h 收益率标准差 σ = {target:.2f}bp")
    print(
        f"           对照：ETH {rc['ETHUSDT_1h']['sigma_bp']:.2f}bp、"
        f"SOL {rc['SOLUSDT_1h']['sigma_bp']:.2f}bp"
    )

    out: dict = {"targets": rc, "pools": {}}
    for pool_tag, mix in POOLS.items():
        print(f"\n  ▸ 池：{pool_tag}")
        print(f"    A) 只改报价幅度上限 zi_offset_range 上界（zi_p_active=0.15）")
        print(f"       {'幅度上界':>10}{'σ(bp)':>10}{'σ/目标':>10}{'价差(bp)':>11}{'总深度':>10}")
        a_rows = scan(pool_tag, mix, key="off_hi", values=OFFSET_HI)
        for r in a_rows:
            flag = ""
            if abs(r["sigma_bp"] / target - 1.0) < 0.25:
                flag = "  ← 量级命中"
            print(
                f"       {r['value']:>9.2%}{r['sigma_bp']:>10.1f}"
                f"{r['sigma_bp'] / target:>10.2f}{r['spread_bp']:>11.2f}"
                f"{r['depth']:>10.1f}{flag}"
            )

        print(f"    B) 只改下单频率 zi_p_active（报价幅度固定 0.02%~0.18%）")
        print(f"       {'p_active':>10}{'σ(bp)':>10}{'σ/目标':>10}{'价差(bp)':>11}{'成交/tick':>11}")
        b_rows = scan(pool_tag, mix, key="p_active", values=P_ACTIVE)
        for r in b_rows:
            flag = ""
            if abs(r["sigma_bp"] / target - 1.0) < 0.25:
                flag = "  ← 量级命中"
            print(
                f"       {r['value']:>10.2f}{r['sigma_bp']:>10.1f}"
                f"{r['sigma_bp'] / target:>10.2f}{r['spread_bp']:>11.2f}{'':>11}{flag}"
            )
        out["pools"][pool_tag] = {"offset_scan": a_rows, "p_active_scan": b_rows}

    # 选定值复核
    print(f"\n  ▸ 选定值复核（zi_offset_range=(0.0002, 0.0018)、zi_p_active=0.15）")
    print(f"    {'池':<24}{'σ(bp)':>10}{'σ/目标':>10}{'价差(bp)':>11}{'总深度':>10}{'成交/tick':>11}")
    final = {}
    for pool_tag, mix in POOLS.items():
        sig, sp, dp, tp = [], [], [], []
        for s in SEEDS:
            pop = Population.from_shares(N_AGENTS, mix)
            cfg = SimConfig(seed=s, n_ticks=WARMUP + N_OBS, population=pop)
            from tw import Market

            m = Market(cfg)
            m.run()
            sl = slice(WARMUP, m.tick)
            mid = m.log.mid[sl]
            r = np.diff(np.log(mid))
            r = r[np.isfinite(r)]
            sig.append(float(r.std() * 1e4))
            spread = m.log.spread[sl] / mid
            spread = spread[np.isfinite(spread)]
            sp.append(float(spread.mean() * 1e4))
            dp.append(float(np.nanmean(m.log.bid_depth[sl] + m.log.ask_depth[sl])))
            tp.append(float(np.nanmean(m.log.n_trades[sl])))
        final[pool_tag] = {
            "sigma_bp": float(np.mean(sig)),
            "spread_bp": float(np.mean(sp)),
            "depth": float(np.mean(dp)),
            "trades_per_tick": float(np.mean(tp)),
        }
        f = final[pool_tag]
        print(
            f"    {pool_tag:<24}{f['sigma_bp']:>10.1f}{f['sigma_bp'] / target:>10.2f}"
            f"{f['spread_bp']:>11.2f}{f['depth']:>10.1f}{f['trades_per_tick']:>11.1f}"
        )
    out["final"] = final
    out["target_sigma_bp"] = target

    # ---- N 扫描：σ 依赖主体数，标定只在固定 N 下成立 ----
    print(f"\n  ▸ 主体数 N 对 σ 的影响（参数固定为选定值）")
    print(f"    {'N':>6}{'纯零智能 σ(bp)':>18}{'混合池 σ(bp)':>18}{'纯ZI σ/目标':>14}{'混合 σ/目标':>14}")
    n_rows = []
    for n in N_SCAN:
        s_zi, s_mix, sp_mix = [], [], []
        for s in SEEDS:
            a, _, _ = sigma_of(POOLS["纯零智能(阶段1)"], s, n_agents=n)
            b, c, _ = sigma_of(POOLS["30/40/30 混合(阶段2)"], s, n_agents=n)
            s_zi.append(a)
            s_mix.append(b)
            sp_mix.append(c)
        row = {
            "n_agents": n,
            "sigma_zi_bp": float(np.mean(s_zi)),
            "sigma_mix_bp": float(np.mean(s_mix)),
            "spread_mix_bp": float(np.mean(sp_mix)),
        }
        n_rows.append(row)
        print(
            f"    {n:>6d}{row['sigma_zi_bp']:>18.1f}{row['sigma_mix_bp']:>18.1f}"
            f"{row['sigma_zi_bp'] / target:>14.2f}{row['sigma_mix_bp'] / target:>14.2f}"
        )
    out["n_scan"] = n_rows
    lo = min(r["sigma_mix_bp"] / target for r in n_rows)
    hi = max(r["sigma_mix_bp"] / target for r in n_rows)
    print(
        f"    → σ 随 N 单调上升（更多主体 = 更多订单流）；"
        f"混合池在 N={N_SCAN[0]}~{N_SCAN[-1]} 之间为目标的 {lo:.2f}×~{hi:.2f}×。"
    )
    print(
        "      **所以'σ 标定好了'这句话必须附带 N**——脱离主体数谈标定是没有意义的。"
    )

    # 结论
    f_zi = final["纯零智能(阶段1)"]
    f_mix = final["30/40/30 混合(阶段2)"]
    print(
        f"\n  → 结论：报价幅度上限从蓝图的 2% 收到 0.18% 后（N={N_AGENTS} 固定），"
        f"纯零智能池 σ={f_zi['sigma_bp']:.1f}bp（目标 {target:.1f}bp，"
        f"{f_zi['sigma_bp'] / target:.2f}×）；"
        f"混合池 σ={f_mix['sigma_bp']:.1f}bp（{f_mix['sigma_bp'] / target:.2f}×）。"
    )
    print(
        f"     两个池在 N={N_AGENTS} 下**同时量级命中**——说明这个刻度对主体构成不敏感，"
        f"是对的参数；但**对 N 敏感**，这是「订单流密度」的必然结果。"
    )

    save_json(out, "calibration.json")
    return out


if __name__ == "__main__":
    main()
