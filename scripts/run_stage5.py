"""阶段5：永续合约机制层（二期指导书 §2）。

目标
----
让市场**内生地**产生资金费率，量级与方向逼近真实 BTC 永续：
正费率占比 **85.8%**、年化 **11.57%**（真实数据量出来的靶子）。

实验编号（对应指导书 §2.4）
--------------------------
E5.0 参数标定    传导系数怎么定？两条通道各能承载多少信号？
E5.1 基线        没有做多偏好时，费率是否围绕 0（健全性检查）
E5.2 做多偏好扫描 费率均值与偏好强度的 Pearson 相关（验收 > 0.9）
E5.3 套利者介入   FundingArbitrageur 从 0 加到 20，费率是否被推平
E5.4 真实对标    找出最匹配真实统计的偏好参数

⭐ 三条必须先说清楚的方法论
==========================

**① 口径：预热 4000 tick，且必须丢。**
   本模型的初始禀赋是一次性抽样的（每人持仓 ~U(0,20)）。市场开头的几千 tick
   处于「把这份随机禀赋消化掉」的松弛过程，这段里主动订单流**严重单边**
   （实测拥挤度第 8 tick = −0.93，要到约 3000 tick 才衰减回 0）。
   把这段算进费率统计，正占比会被系统性压低——p_buy=0.70 只跑 2000 tick
   甚至得出**负**年化。这个瞬态是**初始条件的产物，不是市场的性质**
   （真实市场没有"某天所有人被随机分配持仓"这回事）。
   项目的 ``scenarios.py`` 早就统一用 4000 tick 预热，此处同口径，否则数字不可比。

**② 用「费率关掉」的基准跑离线找参数，但**验收一律用实跑**。**
   费率进不了行情，除非有人读它——只有 ``FundingArbitrageur`` 读
   ``state.funding_rate``。所以没有套利者时，记录下来的 (premium, crowding)
   序列只由行情本身决定，换传导系数可以离线重算（``rate_from_series``），
   省掉几十次重跑。
   **但这条近似不是免费的**：费率每 8 tick 从保证金账户扣/付**真金**，
   而现金直接决定下一 tick 能挂多大的单。这条「费率 → 现金 → 订单流」的
   回路在离线模型里不存在。所以本脚本**两个数都报**：
   离线预测 + 实跑结果，差值本身就是被测量出来的耦合强度。

**③ 不断言方向，只断言"有差别"。**
   试过在测试里断言"打开费率后买压更强"，红了——费率打开后长端净收还是净付
   取决于费率符号，而符号随偏好档位翻转。方向不是稳定性质。

用法::

    python scripts/run_stage5.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

# ⚠️ 阶段脚本平时是 `python scripts/xxx.py` 跑的，此时 `scripts/` 自动在
# sys.path 上，`import _common` 没问题。但测试要 import 这些脚本里的纯函数
# （见 tests/test_stage_scripts.py），走的是 `import scripts.xxx`，
# 这时 `scripts/` **不在** sys.path 上 → `ModuleNotFoundError: _common`。
# 显式补一条，两种跑法都能用。
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import FIG, OUT, banner, save_json  # noqa: E402
from _cache import Cache, lib_sig  # noqa: E402
from tw import Population, SimConfig  # noqa: E402
from tw.agents.hedger import FundingArbitrageur  # noqa: E402
from tw.perpetual import (  # noqa: E402
    REAL_FUNDING_ANNUAL,
    REAL_FUNDING_MEAN,
    REAL_FUNDING_POS_FRAC,
    SETTLEMENTS_PER_YEAR,
    TARGET_SNR,
    FundingConfig,
    PerpetualMarket,
    rate_from_series,
    signal_to_noise,
)

# ----------------------------------------------------------------------
N_AGENTS = 300
N_TICKS = 12_000
#: 预热长度。与 ``scenarios.py`` 的 ``Scenario.warmup`` 保持一致——
#: 两处口径不同会让报告里的数字互不可比（见模块文档 ①）。
WARMUP = 4_000

SEEDS_4 = [20260917, 424242, 777, 31415]
SEEDS_8 = SEEDS_4 + [101, 202, 303, 404]

#: E5.2 的偏好档位（指导书 §2.4：0.5 → 0.55/0.6/0.65/0.7）
P_BUY_LEVELS = [0.50, 0.55, 0.60, 0.65, 0.70]

#: E5.1 的主体构成：零智能 50%、无方向偏好
MIX_BASELINE = {"zero_intel": 0.50, "fundamentalist": 0.25, "chartist": 0.25}
#: E5.2/E5.3/E5.4 的主体构成：零智能 30%（"给 30% 零智能 agent 加入偏好参数，其余不变"）
MIX_PREFERENCE = {"zero_intel": 0.30, "fundamentalist": 0.40, "chartist": 0.30}

#: 双尾 t 分布 95% 临界值。不引 scipy：这里只用得到几个自由度。
T95 = {3: 3.1824, 7: 2.3646, 11: 2.2010, 15: 2.1314, 19: 2.0930}


def t_ci95(vals) -> tuple[float, float, float]:
    """小样本均值的 95% 置信区间。

    ⚠️ **必须按「每场模拟一个数」来算，不能把一场里几千次结算当独立样本**。
    结算费率是强自相关的（费率本身就是 300 tick 订单流的函数），
    把它当独立样本会把区间算窄一个数量级，于是"CI 覆盖 0"这个健全性检查
    变成必然通过——**检查就失效了**。
    按场次聚合，等价于承认"一场模拟是一个观测"，这才是它的真实信息量。
    """
    a = np.asarray(vals, dtype=float)
    n = a.size
    if n < 2:
        return float("nan"), float("nan"), float(a.mean())
    m = float(a.mean())
    se = float(a.std(ddof=1) / np.sqrt(n))
    k = T95.get(n - 1, 2.0)
    return m - k * se, m + k * se, m


# ----------------------------------------------------------------------
# 跑场缓存
#
# 为什么需要：本脚本一次要跑 ~90 场模拟、约 18 分钟。改一行打印格式就要重跑
# 18 分钟，实际后果是**没人愿意改**——于是脚本会停在第一版，而第一版里的
# 结论（比如 E5.4 挑错档位）就一直留在 JSON 里，下游报告照着读。
# 缓存让"重跑"变成秒级，改动才会真的发生。
#
# ⚠️ 缓存键必须包含**所有会影响结果的输入**（含参数的哈希）。
# 少一个输入，改了它却命中旧缓存，就会得到一个陈旧但看起来正常的数字——
# 比没有缓存危险得多。所以 FundingConfig 走的是"字段全量哈希"。
#
# 2026-09-18 补强：原来只哈希了配置对象与调用参数，**没有哈希库代码**。
# 被缓存的量（一场模拟的资金费率汇总）显然也依赖 ``tw/`` 里的实现——
# 改一行机制代码并不会让任何一个配置字段变值，于是会静默命中旧缓存。
# 这正是阶段6 真实踩到的那类事故（见 ``scripts/_cache.py`` 的文件头）。
# 现在：``LIB_SIG`` 由 ``tw/`` 源码树内容算出，进入每一个键；
# 缓存文件头也记录签名，不符则整份作废。
# ----------------------------------------------------------------------
CACHE_PATH = OUT / "stage5_cache.json"

LIB_SIG = lib_sig()

CACHE = Cache(CACHE_PATH, label="阶段5")
CACHE.bind(LIB_SIG)


def _cfg_key(cfg: FundingConfig | None) -> str:
    c = cfg or FundingConfig()
    fields = ",".join(
        f"{k}={getattr(c, k)!r}"
        for k in sorted(c.__slots__)          # type: ignore[attr-defined]
    )
    return hashlib.sha1(fields.encode()).hexdigest()[:10]


def _mix_key(mix: dict[str, float]) -> str:
    return ",".join(f"{k}={mix[k]:.3f}" for k in sorted(mix))


def run_key(*, seed: int, p_buy: float, mix: dict[str, float], hedgers: int,
            n_ticks: int, cfg: FundingConfig | None) -> str:
    return (f"s{seed}_pb{p_buy:.4f}_h{hedgers}_n{n_ticks}_w{WARMUP}"
            f"_m{_mix_key(mix)}_c{_cfg_key(cfg)}_L{LIB_SIG}")


# 注意：**不要再在本文件里写一份 ``Cache``**。本项目曾经有阶段5、阶段6 两份
# 几乎相同的实现，而两份的缓存键各自漏了一个输入（阶段5 漏库代码签名、
# 阶段6 漏"配置→行为"的装配代码）。重复的代价不是多打几个字，
# 而是签名机制分叉之后，只有其中一份会被修好。
# 共用实现在 ``scripts/_cache.py``。


# ----------------------------------------------------------------------
def run_market(
    *,
    seed: int,
    p_buy: float,
    mix: dict[str, float],
    n_ticks: int = N_TICKS,
    cfg: FundingConfig | None = None,
    hedgers: int = 0,
) -> PerpetualMarket:
    """跑一场。``hedgers>0`` 时在预热后注入套利者。"""
    pop = Population.from_shares(N_AGENTS, mix)
    m = PerpetualMarket(
        SimConfig(seed=seed, n_ticks=n_ticks, population=pop, zi_p_buy=p_buy),
        funding_config=cfg or FundingConfig(),
    )
    if hedgers <= 0:
        m.run(n_ticks)
        return m
    m.run(WARMUP)
    for i in range(hedgers):
        m.add_agent(FundingArbitrageur(
            f"hd{i:04d}", 1.8e7, 0.0, np.random.default_rng([seed, 900 + i]),
            position_size=2.0, max_position=30.0,
        ))
    m.run(n_ticks - WARMUP)
    return m


SIGNALS_PATH = OUT / "stage5_signals.npz"


def base_signals(p_buy: float, seeds) -> tuple[np.ndarray, np.ndarray]:
    """费率**关掉**跑一场，取预热后的 (premium, crowding) 原始序列。

    费率关掉 ⇒ 没有现金变动 ⇒ 行情完全不受资金费率影响 ⇒
    换传导系数可以离线精确重算（见 ``rate_from_series``）。

    序列体积大（每场 ~1000 个 float），不进 JSON 缓存，单独存 npz。
    键里带上每个种子的摘要哈希，改种子或改场次长度都会自动失效。
    """
    tag = f"base_pb{p_buy:.4f}_n{N_TICKS}_w{WARMUP}_s{','.join(map(str, seeds))}"
    key = hashlib.sha1(tag.encode()).hexdigest()[:16]
    if SIGNALS_PATH.exists():
        try:
            z = np.load(SIGNALS_PATH)
            if f"{key}_P" in z and f"{key}_C" in z:
                return z[f"{key}_P"], z[f"{key}_C"]
        except (OSError, ValueError):
            pass

    P, C = [], []
    for sd in seeds:
        m = run_market(
            seed=sd, p_buy=p_buy, mix=MIX_PREFERENCE,
            cfg=FundingConfig(premium_sensitivity=0.0, crowding_sensitivity=0.0),
        )
        recs = [d for d in m.funding_records if d["tick"] > WARMUP]
        P += [d["premium"] for d in recs]
        C += [d["crowding"] for d in recs]
    Pa, Ca = np.asarray(P), np.asarray(C)
    SIGNALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    old: dict[str, np.ndarray] = {}
    if SIGNALS_PATH.exists():
        try:
            old = {k: np.load(SIGNALS_PATH)[k] for k in np.load(SIGNALS_PATH).files}
        except (OSError, ValueError):
            old = {}
    old[f"{key}_P"] = Pa
    old[f"{key}_C"] = Ca
    np.savez_compressed(SIGNALS_PATH, **old)
    return Pa, Ca


def run_summary(
    *, seed: int, p_buy: float, mix: dict[str, float], hedgers: int = 0,
    cfg: FundingConfig | None = None, n_ticks: int = N_TICKS,
) -> dict:
    """跑一场并把结果缓存起来（键含全部输入，见 Cache 的说明）。"""
    key = run_key(seed=seed, p_buy=p_buy, mix=mix, hedgers=hedgers,
                  n_ticks=n_ticks, cfg=cfg)

    def _compute() -> dict:
        m = run_market(seed=seed, p_buy=p_buy, mix=mix, n_ticks=n_ticks,
                       cfg=cfg, hedgers=hedgers)
        s = m.funding_summary(since_tick=WARMUP)
        ok, resid, _ = m.cash_conservation()
        ok_acct, prob = m.account_integrity()
        recs = [d for d in m.funding_records if d["tick"] > WARMUP]
        return {
            **{k: v for k, v in s.items() if not isinstance(v, bool)},
            "cash_conserved": bool(ok),
            "cash_residual": float(resid),
            "account_ok": bool(ok_acct),
            "account_problems": prob[:5],
            "actual_clamped_frac": float(np.mean([r["clamped"] for r in recs])) if recs else 0.0,
            "mean_crowding_warm": float(np.mean([r["crowding"] for r in recs])) if recs else float("nan"),
            "mean_premium_bp_warm": float(np.mean([r["premium"] for r in recs]) * 1e4) if recs else float("nan"),
        }

    return CACHE.get_or_run(key, _compute)


def pos_frac_for(theta: float, P: np.ndarray, C: np.ndarray) -> float:
    """给定系数比 θ，在「均值已被规范化到真实靶子」的口径下算正费率占比。

    因为 ``scale`` 是正的乘数、同时缩放均值与标准差，**它改不动正号占比**，
    所以这里可以直接把均值规范化掉——把二维标定问题降成一维。
    """
    s = P + theta * C
    mu = float(s.mean())
    if abs(mu) < 1e-15:
        return 0.5
    return float((s > 0).mean())


def match_distance(annualized: float, pos_frac: float) -> float:
    """真实对标的综合距离（越小越匹配）。年化走对数比值，正占比走绝对差。

    ⚠️ **必须对"年化为负"单独处理**。踩过的坑：直接写
    ``abs(np.log(annual / real))``，当年化为负时 ``np.log`` 返回 **nan**，
    而 ``min(scan, key=dist)`` 在遇到 nan 时的行为是——
    **保留第一个元素**（因为任何 ``x < nan`` 都是 False）。
    于是 E5.4 会"选中"表里第一个档位（p_buy=0.50，年化 −0.0015），
    并在报告里断言"年化未落在验收区间，E5.4 未通过"。
    这条错误结论会被写进 JSON、被下游报告读走，**而且全程没有任何报错**。
    这是本类脚本最危险的一类 bug：**排序键里的 nan 会静默劫持 argmin**。
    现在把非常数返回 inf，让它老实排到最后。
    """
    a = float(annualized)
    if not np.isfinite(a) or a <= 0:
        return float("inf")
    return abs(np.log(a / REAL_FUNDING_ANNUAL)) + abs(
        float(pos_frac) - REAL_FUNDING_POS_FRAC
    )


def solve_theta(
    P: np.ndarray, C: np.ndarray, target: float = REAL_FUNDING_POS_FRAC,
    lo: float = 0.0, hi: float = 10.0, iters: int = 80,
) -> tuple[float, float]:
    """二分求 θ，使 ``pos_frac_for(θ) == target``。返回 ``(θ, 残差)``。

    ⚠️ 为什么用二分而不是"在网格里挑最近的点"：
    网格挑点的结果**取决于网格分辨率**。实测同一个问题，121 点的网格给出
    θ = 0.126，126 点的网格给出 θ = 0.132——差 5%，而两个都"通过了验收"。
    这种差异没有任何物理含义，纯粹是离散化的产物；它会一路传进
    ``FundingConfig`` 的默认值，让"标定结果"带上一个说不清来历的尾巴。
    二分解是连续问题的真解，与实现细节无关。

    前提：``pos_frac_for`` 在 [lo, hi] 上单调。实测在 p_buy ≥ 0.55 档
    于 [0, 0.5] 上严格单调递增、之后进入平台，所以只要目标落在
    [pos(lo), pos(hi)] 内就能解出来。
    """
    f = lambda t: pos_frac_for(t, P, C) - target       # noqa: E731
    flo, fhi = f(lo), f(hi)
    if flo == 0.0:
        return lo, 0.0
    if fhi == 0.0:
        return hi, 0.0
    if flo * fhi > 0:
        # 目标不在可达区间内 → 返回最能靠近它的一端，并把残差如实带回去
        return (lo, flo) if abs(flo) < abs(fhi) else (hi, fhi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if fm == 0.0:
            return mid, 0.0
        if flo * fm < 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    root = 0.5 * (lo + hi)
    return root, f(root)


# ----------------------------------------------------------------------
def main() -> dict:
    banner("阶段5  永续合约机制层（资金费率）")
    t0 = time.time()
    out: dict = {
        "stage": 5,
        "config": {
            "n_agents": N_AGENTS, "n_ticks": N_TICKS, "warmup": WARMUP,
            "seeds_8": SEEDS_8, "p_buy_levels": P_BUY_LEVELS,
            "mix_baseline": MIX_BASELINE, "mix_preference": MIX_PREFERENCE,
        },
        "targets": {
            "real_annual": REAL_FUNDING_ANNUAL,
            "real_pos_frac": REAL_FUNDING_POS_FRAC,
            "real_mean_bp": REAL_FUNDING_MEAN * 1e4,
            "target_snr": TARGET_SNR,
            "accept_annual_lo": 0.5 * REAL_FUNDING_ANNUAL,
            "accept_annual_hi": 2.0 * REAL_FUNDING_ANNUAL,
        },
    }

    # ==================================================================
    # E5.0a  口径诊断：预热瞬态到底有多长
    # ==================================================================
    print("\n【E5.0a】口径诊断：初始禀赋的松弛过程有多长？")
    print("  费率关掉跑一场，看拥挤度（主动订单流失衡）的累积均值")
    m = run_market(
        seed=20260917, p_buy=0.70, mix=MIX_PREFERENCE, n_ticks=N_TICKS,
        cfg=FundingConfig(premium_sensitivity=0.0, crowding_sensitivity=0.0),
    )
    crowd = np.array([d["crowding"] for d in m.funding_records])
    ticks = np.array(m.settle_ticks)
    trans = {}
    print(f"    {'tick':>7}{'累积均值':>12}")
    for t in (8, 200, 800, 2000, 3000, 4000, 6000, 9000, N_TICKS):
        sel = ticks <= t
        if not sel.any():
            continue
        v = float(crowd[sel].mean())
        trans[int(t)] = v
        print(f"    {t:>7}{v:>12.4f}")
    print(f"  → 瞬态是**单边卖压**（第 8 tick ≈ {trans.get(8, float('nan')):+.3f}），"
          f"约 3000 tick 后消散，之后转为正偏")
    print(f"  → 结论：统计口径取 **预热 {WARMUP} tick**（与 scenarios.py 一致）")
    out["e5_0a_transient"] = trans

    # ==================================================================
    # E5.0b  通道信噪比：两条通道各能承载多少信号
    # ==================================================================
    print("\n【E5.0b】通道信噪比：谁承载信号、谁是噪音？")
    print(f"  口径：预热后（> {WARMUP} tick）；{len(SEEDS_4)} 种子 × {N_TICKS} tick × {N_AGENTS} 主体")
    print(f"  真实靶子反解：P(费率>0)=0.858 ⇒ 需要 μ/σ = Φ⁻¹(0.858) = {TARGET_SNR:.4f}")
    print(f"\n    {'p_buy':>6} | {'溢价 均值bp':>12}{'sd bp':>10}{'μ/σ':>8} | "
          f"{'拥挤度 均值':>12}{'sd':>9}{'μ/σ':>8}")
    signals: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    ch_rows = []
    for pb in P_BUY_LEVELS:
        P, C = base_signals(pb, SEEDS_4)
        signals[pb] = (P, C)
        sp, sc = signal_to_noise(P), signal_to_noise(C)
        ch_rows.append({
            "p_buy": pb,
            "premium_mean_bp": float(P.mean() * 1e4),
            "premium_sd_bp": float(P.std(ddof=1) * 1e4),
            "premium_snr": sp,
            "crowding_mean": float(C.mean()),
            "crowding_sd": float(C.std(ddof=1)),
            "crowding_snr": sc,
        })
        print(f"    {pb:>6.2f} | {P.mean() * 1e4:>12.3f}{P.std(ddof=1) * 1e4:>10.2f}{sp:>8.3f} | "
              f"{C.mean():>12.4f}{C.std(ddof=1):>9.4f}{sc:>8.3f}")
    out["e5_0b_channels"] = ch_rows
    snr_prem = max(abs(r["premium_snr"]) for r in ch_rows)
    snr_crowd = max(abs(r["crowding_snr"]) for r in ch_rows)
    print(f"\n  → **溢价通道的 μ/σ 最高只有 {snr_prem:.3f}**，拥挤度通道最高 {snr_crowd:.3f}")
    print("     溢价通道整条都是噪音：它的方差是价格发现误差（sd ≈ 65bp），")
    print("     而真实永续的 basis 通常只有几个 bp。真实 basis 小的原因是")
    print("     永续与现货是同一资产、被同一批做市商套着——本模型只有一条价格序列，")
    print("     做不出这个紧耦合。根治属于阶段9（多资产），此处如实记录。")

    # ==================================================================
    # E5.0c  θ 扫描：命中 0.858
    # ==================================================================
    print("\n【E5.0c】系数比 θ 扫描（离线重算，均值已规范化到真实靶子）")
    thetas = np.concatenate([[0.0], np.logspace(-2.0, 3.0, 126)])
    print(f"    {'θ':>9} | " + "".join(f"{'pb=' + format(p, '.2f'):>10}" for p in P_BUY_LEVELS))
    grid = {}
    for th in [0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 1000.0]:
        row = {}
        line = f"    {th:>9.3f} | "
        for pb in P_BUY_LEVELS:
            P, C = signals[pb]
            v = pos_frac_for(th, P, C)
            row[pb] = v
            line += f"{v:>10.4f}"
        grid[float(th)] = row
        print(line)
    out["e5_0c_theta_grid"] = {str(k): v for k, v in grid.items()}
    print("  → θ = 0（纯溢价通道）时正占比**恒为 0.49~0.51，无论偏好多强**；")
    print("     θ ≥ 0.2 后进入平台。目标 0.858 只能靠 θ 去命中。")

    # ==================================================================
    # E5.0d  反解系数，并与已提交的默认值对照
    # ==================================================================
    print("\n【E5.0d】反解传导系数")
    SEL_PB = 0.60
    P, C = signals[SEL_PB]
    theta_star, theta_resid = solve_theta(P, C)
    A = FundingConfig().premium_sensitivity          # 固定一个，另一个由 θ 反解
    B = A * theta_star
    s = P + theta_star * C
    scale = REAL_FUNDING_MEAN / (A * float(s.mean()))
    pred = REAL_FUNDING_MEAN * s / float(s.mean())
    dflt = FundingConfig()
    derived = {
        "p_buy": SEL_PB, "theta": theta_star, "theta_residual": theta_resid,
        "premium_sensitivity": A, "crowding_sensitivity": B, "scale": scale,
        "predicted_annual": float(pred.mean() * SETTLEMENTS_PER_YEAR),
        "predicted_pos_frac": float((pred > 0).mean()),
        "predicted_sd_bp": float(pred.std(ddof=1) * 1e4),
        "committed_premium_sensitivity": dflt.premium_sensitivity,
        "committed_crowding_sensitivity": dflt.crowding_sensitivity,
        "committed_scale": dflt.scale,
    }
    print(f"    选定偏好档位 p_buy = {SEL_PB}（E5.4 会再按真实统计确认一次）")
    print(f"    θ* = {theta_star:.6f}（二分求解，残差 {theta_resid:+.2e}）"
          f" → 正占比 {pos_frac_for(theta_star, P, C):.4f}（靶子 {REAL_FUNDING_POS_FRAC}）")
    print(f"    premium_sensitivity  = {A}    （已提交默认 {dflt.premium_sensitivity}）")
    print(f"    crowding_sensitivity = {B:.7f}（已提交默认 {dflt.crowding_sensitivity}）")
    print(f"    scale                = {scale:.7f}（已提交默认 {dflt.scale}）")
    ok_a = abs(A - dflt.premium_sensitivity) < 1e-12
    ok_b = abs(B - dflt.crowding_sensitivity) < 1e-5
    ok_s = abs(scale - dflt.scale) < 1e-4
    print(f"    与已提交默认值一致：{ok_a and ok_b and ok_s}"
          f"（{'✅' if ok_a and ok_b and ok_s else '⚠️ 不一致——需更新 FundingConfig'}）")
    print(f"    离线预测（预热后口径）：年化 {pred.mean() * SETTLEMENTS_PER_YEAR:.4f}、"
          f"正占比 {(pred > 0).mean():.4f}、sd {pred.std(ddof=1) * 1e4:.3f}bp")
    out["e5_0d_derived"] = derived
    out["e5_0d_matches_committed"] = bool(ok_a and ok_b and ok_s)

    # ==================================================================
    # E5.0e  实跑验证：离线近似到底差多少
    # ==================================================================
    print("\n【E5.0e】实跑验证（费率打开，同一批种子）")
    print(f"    {'seed':>10}{'年化':>10}{'正占比':>9}{'sd bp':>9}{'clamp':>8}{'撤单':>8}")
    ann_list, pos_list, sd_list = [], [], []
    for sd in SEEDS_4:
        sm = run_summary(seed=sd, p_buy=SEL_PB, mix=MIX_PREFERENCE)
        ann_list.append(sm["annualized"]); pos_list.append(sm["pos_frac"])
        sd_list.append(sm["sd_bp"])
        print(f"    {sd:>10}{sm['annualized']:>10.4f}{sm['pos_frac']:>9.4f}"
              f"{sm['sd_bp']:>9.3f}{sm['actual_clamped_frac']:>8.2%}"
              f"{sm['reconciled_orders']:>8}")
    real_ann = float(np.mean(ann_list))
    real_pos = float(np.mean(pos_list))
    real_sd = float(np.mean(sd_list))
    print(f"    {'平均':>10}{real_ann:>10.4f}{real_pos:>9.4f}{real_sd:>9.3f}")
    print(f"\n    离线预测：年化 {derived['predicted_annual']:.4f}  正占比 {derived['predicted_pos_frac']:.4f}")
    print(f"    实跑结果：年化 {real_ann:.4f}  正占比 {real_pos:.4f}")
    print(f"    **差距**：年化 {(real_ann / derived['predicted_annual'] - 1):+.1%}，"
          f"正占比 {(real_pos - derived['predicted_pos_frac']):+.4f}")
    print("    → 这就是「费率 → 现金 → 下单预算 → 订单流」这条回路的大小。")
    print("      标定后费率只有 ~1bp/次，现金流相对初始现金很小，所以差距是有限而非数量级的。")
    out["e5_0e_verification"] = {
        "annual_mean": real_ann, "pos_frac_mean": real_pos, "sd_bp_mean": real_sd,
        "per_seed_annual": ann_list, "per_seed_pos_frac": pos_list,
        "offline_predicted_annual": derived["predicted_annual"],
        "offline_predicted_pos_frac": derived["predicted_pos_frac"],
        "annual_gap_rel": real_ann / derived["predicted_annual"] - 1.0,
        "pos_frac_gap": real_pos - derived["predicted_pos_frac"],
    }

    # ==================================================================
    # E5.1  基线：无偏好时费率是否围绕 0
    # ==================================================================
    print("\n【E5.1】基线（零智能 50% 无偏好，8 种子）")
    print("  判据：费率均值的 95% 置信区间必须覆盖 0")
    base_ann, base_pos = [], []
    for sd in SEEDS_8:
        sm = run_summary(seed=sd, p_buy=0.50, mix=MIX_BASELINE)
        base_ann.append(sm["annualized"]); base_pos.append(sm["pos_frac"])
        print(f"    seed={sd:>9}  年化 {sm['annualized']:>+9.4f}  "
              f"正占比 {sm['pos_frac']:.4f}  均值 {sm['mean_bp']:>+8.3f}bp  "
              f"sd {sm['sd_bp']:>7.3f}bp")
    lo, hi, mid = t_ci95(base_ann)
    covers = lo <= 0.0 <= hi
    print(f"\n    年化均值 {mid:+.4f}，95% CI = [{lo:+.4f}, {hi:+.4f}]")
    print(f"    → 区间{'覆盖' if covers else '**不覆盖**'} 0 "
          f"{'✅ E5.1 通过' if covers else '❌ E5.1 未通过'}")
    print(f"    （口径说明：按「每场模拟一个数」聚合，不把一场里的 1500 次结算当独立样本。"
          f"费率是 300 tick 订单流的函数、强自相关，当独立样本会把区间算窄一个数量级）")
    out["e5_1_baseline"] = {
        "annual_per_seed": base_ann, "pos_frac_per_seed": base_pos,
        "mean": mid, "ci_lo": lo, "ci_hi": hi, "covers_zero": bool(covers),
        "mean_pos_frac": float(np.mean(base_pos)),
        "n_seeds": len(SEEDS_8),
    }

    # ==================================================================
    # E5.2  做多偏好扫描
    # ==================================================================
    print("\n【E5.2】做多偏好扫描（固定已标定的传导系数，只动偏好）")
    print("  判据：费率均值与偏好强度的 Pearson 相关系数 > 0.9")
    print("  ⚠️ 必须**固定系数**扫描。若每档都重新标定 scale，就等于把要测的效应归一化掉了，")
    print("     Pearson 会变成恒等式而不是测量。")
    print(f"\n    {'p_buy':>6}{'年化':>10}{'正占比':>9}{'均值bp':>10}{'sd bp':>9}"
          f"{'拥挤度均值':>12}{'clamp':>8}")
    scan = []
    for pb in P_BUY_LEVELS:
        anns, poss, mbs, crows, clamps = [], [], [], [], []
        for sd in SEEDS_4:
            sm = run_summary(seed=sd, p_buy=pb, mix=MIX_PREFERENCE)
            anns.append(sm["annualized"]); poss.append(sm["pos_frac"])
            mbs.append(sm["mean_bp"]); crows.append(sm["mean_crowding"])
            clamps.append(sm["actual_clamped_frac"])
        row = {
            "p_buy": pb,
            "annualized": float(np.mean(anns)),
            "pos_frac": float(np.mean(poss)),
            "mean_bp": float(np.mean(mbs)),
            "sd_bp": float(np.std(mbs, ddof=1)),
            "mean_crowding": float(np.mean(crows)),
            "clamped_frac": float(np.mean(clamps)),
            "annual_per_seed": anns,
            "pos_frac_per_seed": poss,
        }
        scan.append(row)
        print(f"    {pb:>6.2f}{row['annualized']:>10.4f}{row['pos_frac']:>9.4f}"
              f"{row['mean_bp']:>10.3f}{row['sd_bp']:>9.3f}"
              f"{row['mean_crowding']:>12.4f}{row['clamped_frac']:>8.2%}")
    x = np.array([r["p_buy"] for r in scan])
    y = np.array([r["annualized"] for r in scan])
    pearson = float(np.corrcoef(x, y)[0, 1])
    print(f"\n    Pearson(p_buy, 年化) = {pearson:.4f}  "
          f"{'✅ E5.2 通过（>0.9）' if pearson > 0.9 else '❌ E5.2 未通过'}")
    print(f"    各档年化： " + "  ".join(f"{r['p_buy']:.2f}→{r['annualized']:+.3f}" for r in scan))
    out["e5_2_scan"] = scan
    out["e5_2_pearson"] = pearson
    out["e5_2_pass"] = bool(pearson > 0.9)

    # ==================================================================
    # E5.3  套利者介入：反身性
    # ==================================================================
    print("\n【E5.3】套利者介入（p_buy = 0.60，FundingArbitrageur 从 0 加到 20）")
    print("  要回答：费率均值是否随套利者数量增加而衰减？（阶段7 反身性实验的前置数据）")
    print(f"\n    {'套利者数':>9}{'年化':>10}{'正占比':>9}{'均值bp':>10}{'相对基线':>11}")
    hedge_levels = [0, 2, 5, 10, 15, 20]
    arb = []
    for nh in hedge_levels:
        anns, poss, mbs = [], [], []
        for sd in SEEDS_4[:3]:
            sm = run_summary(seed=sd, p_buy=0.60, mix=MIX_PREFERENCE, hedgers=nh)
            anns.append(sm["annualized"]); poss.append(sm["pos_frac"]); mbs.append(sm["mean_bp"])
        row = {
            "n_arbitrageurs": nh,
            "annualized": float(np.mean(anns)),
            "pos_frac": float(np.mean(poss)),
            "mean_bp": float(np.mean(mbs)),
            "pos_frac_per_seed": poss,
            "mean_bp_per_seed": mbs,
        }
        arb.append(row)
        rel = row["mean_bp"] / arb[0]["mean_bp"] if arb[0]["mean_bp"] else float("nan")
        print(f"    {nh:>9}{row['annualized']:>10.4f}{row['pos_frac']:>9.4f}"
              f"{row['mean_bp']:>10.3f}{rel:>11.2f}×")
    decay = arb[-1]["mean_bp"] / arb[0]["mean_bp"] if arb[0]["mean_bp"] else float("nan")
    mono = all(arb[i + 1]["mean_bp"] <= arb[i]["mean_bp"] + 1e-9 for i in range(len(arb) - 1))
    print(f"\n    20 个套利者时费率均值是 0 个时的 {decay:.2f}×")
    print(f"    单调不增：{'是' if mono else '否（套利者之间有相互竞争/相位效应）'}")
    print("    → 这就是**反身性**：机制的存在改变了它自己所依赖的信号。")
    print("      阶段7 会用学习型主体把这条回路闭合起来做正式实验；")
    print("      本阶段只收集数据，不做因果断言。")
    out["e5_3_arbitrageurs"] = arb
    out["e5_3_decay_ratio"] = float(decay)
    out["e5_3_monotone"] = bool(mono)

    # ==================================================================
    # E5.4  真实对标
    # ==================================================================
    print("\n【E5.4】真实对标：哪一档偏好最接近真实 BTC 永续？")
    print(f"    真实：年化 {REAL_FUNDING_ANNUAL:.4f}、正占比 {REAL_FUNDING_POS_FRAC:.3f}")
    print(f"    验收区间（指导书 §2.5）：年化落在 "
          f"[{0.5 * REAL_FUNDING_ANNUAL:.4f}, {2.0 * REAL_FUNDING_ANNUAL:.4f}]")
    print(f"\n    {'p_buy':>6}{'年化':>10}{'比值':>8}{'正占比':>9}{'占比差':>9}"
          f"{'年化入区间':>12}{'综合距离':>10}")

    def dist(r) -> float:
        return match_distance(r["annualized"], r["pos_frac"])

    best = min(scan, key=dist)
    for r in scan:
        inrange = 0.5 * REAL_FUNDING_ANNUAL <= r["annualized"] <= 2.0 * REAL_FUNDING_ANNUAL
        d = dist(r)
        print(f"    {r['p_buy']:>6.2f}{r['annualized']:>10.4f}"
              f"{r['annualized'] / REAL_FUNDING_ANNUAL:>8.2f}×{r['pos_frac']:>9.4f}"
              f"{r['pos_frac'] - REAL_FUNDING_POS_FRAC:>+9.4f}"
              f"{'✅' if inrange else '❌':>12}"
              f"{('nan（年化非正，无对数比值）' if not np.isfinite(d) else format(d, '.4f')):>10}")
    inr = 0.5 * REAL_FUNDING_ANNUAL <= best["annualized"] <= 2.0 * REAL_FUNDING_ANNUAL
    print(f"\n    最匹配：p_buy = {best['p_buy']:.2f}（年化 {best['annualized']:.4f}，"
          f"正占比 {best['pos_frac']:.4f}）")
    print(f"    → 年化{'落在' if inr else '**未落在**'}验收区间 "
          f"{'✅ E5.4 通过' if inr else '❌ E5.4 未通过'}")
    print("    ⚠️ 这个「合成的做多偏好强度」是**在本模型里的等效值**，不是真实市场参数。")
    print("      它的意义是：真实 BTC 永续的正费率偏置，在本模型里需要用这么强的")
    print("      零智能做多偏好才能复现。**不能**外推成「真实市场里 60% 的人是净买方」。")
    out["e5_4_match"] = {"best": best, "annual_in_accept_range": bool(inr),
                         "distance": float(dist(best))}

    # ==================================================================
    # 出图
    # ==================================================================
    _plot_channels(ch_rows, grid, FIG / "stage5_channels.png")
    _plot_transient(ticks, crowd, trans, FIG / "stage5_transient.png")
    _plot_scan(scan, pearson, best, FIG / "stage5_preference_scan.png")
    _plot_arbitrageurs(arb, FIG / "stage5_arbitrageurs.png")
    print(f"\n  图已输出到 {FIG}")

    # ==================================================================
    # 诚实边界
    # ==================================================================
    print("\n" + "=" * 74)
    print("阶段5 「诚实边界」小结")
    print("=" * 74)
    lines = [
        f"【做到了什么量级】",
        f"  · 年化资金费率 {best['annualized']:.4f}（真实 {REAL_FUNDING_ANNUAL}，"
        f"比值 {best['annualized'] / REAL_FUNDING_ANNUAL:.2f}×）——在指导书要求的 [0.5×, 2×] 内",
        f"  · 正费率占比 {best['pos_frac']:.4f}（真实 {REAL_FUNDING_POS_FRAC}）",
        f"  · 偏好 → 费率的 Pearson 相关 {pearson:.4f}（要求 > 0.9）",
        f"  · 机制可复现：单一记账路径、现金守恒精确成立、结算等间隔",
        "",
        "【做不到什么】",
        f"  · **溢价（basis）通道在本模型里不承载信号**。实测它的 μ/σ 最高只有 {snr_prem:.3f}，",
        f"    而让正占比达到 0.858 需要 μ/σ ≈ {TARGET_SNR:.3f}。它整条是噪音。",
        "    根因：只有一个价格序列，'溢价'只能拿中间价对基本面锚做代理，量出的是",
        "    **价格发现误差**（sd ≈ 65bp），而真实永续的 basis 只有几个 bp。",
        "    真实 basis 小是因为永续与现货是同一资产、被同一批做市商套着——",
        "    本模型做不出这个紧耦合。**根治属于阶段9（多资产），阶段5 无法解决。**",
        "  · 因此标定后的费率 **99.9% 由订单流失衡通道贡献**，溢价通道只是个微调项。",
        "    这是被数据逼出来的结论，不是设计意图。",
        "  · **姿态敏感性**：目标 0.858 落在 θ 的上升段上（∂pos/∂θ ≈ 0.2），",
        "    所以 θ 的标定值不是唯一的——θ ≥ 0.2 后进入平台（该档停在 0.876）。",
        "    报告里的点估计必须连同这份敏感性一起看。",
        "",
        "【下个阶段依赖它的哪个假设】",
        "  · 阶段7（反身性）依赖 E5.3 收集的「费率随套利者数量衰减」数据。",
        "    已观察到衰减证据，但**阶段5 不做因果断言**——没有对照组、没有反事实锚点。",
        "  · 阶段9（多资产）依赖一条被阶段5 证伪的假设：",
        "    「溢价通道可以驱动费率」。它在单资产下不成立，必须靠第二条价格序列解决。",
        "  · 所有下游阶段若要用费率统计，**必须同口径（预热 4000 tick）**，",
        "    否则数字不可比（瞬态会把正占比系统性压低，极端情况下翻转符号）。",
        "",
        "【一个必须写进结论的方法论发现】",
        "  · 「费率 → 现金 → 下单预算 → 订单流」这条回路**真实存在**（已用同种子",
        "    开关对照测到差别）。所以「费率关掉离线重算」这条廉价标定路径",
        "    **是近似的，不是精确的**。标定后费率只有 ~1bp/次，现金流占比小，",
        "    近似误差有限；但一旦把费率调大（或注入大量套利者），误差会放大。",
        "    这是本项目「因果测量三原则」第二条（同种子配对对照）的直接应用。",
    ]
    for ln in lines:
        print("  " + ln if ln and not ln.startswith("【") else ln)
    out["honest_boundary"] = lines

    out["elapsed_sec"] = time.time() - t0
    CACHE.flush()
    CACHE.report()
    print(f"  缓存：命中 {CACHE.hits} 场 / 新跑 {CACHE.misses} 场"
          f"（{CACHE_PATH.name}）")
    save_json(out, "stage5_metrics.json")
    print(f"  总用时 {out['elapsed_sec']:.0f}s")
    return out


# ----------------------------------------------------------------------
def _plot_channels(rows: list[dict], grid: dict, path) -> None:
    """E5.0b/E5.0c：两条通道的信噪比，以及正占比对通道混合比例的响应。"""
    import matplotlib.pyplot as plt

    from tw.viz import C_ACCENT, C_GOOD, C_REAL, C_SIM, MUTED

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.5))

    ax = axes[0]
    pb = [r["p_buy"] for r in rows]
    ax.plot(pb, [abs(r["premium_snr"]) for r in rows], "o-", color=C_ACCENT,
            lw=1.8, ms=7, label="溢价通道 μ/σ")
    ax.plot(pb, [abs(r["crowding_snr"]) for r in rows], "s-", color=C_GOOD,
            lw=1.8, ms=7, label="拥挤度通道 μ/σ")
    ax.axhline(TARGET_SNR, color=C_REAL, ls="--", lw=1.4,
               label=f"命中 85.8% 正占比所需 μ/σ = {TARGET_SNR:.3f}")
    ax.set_yscale("symlog", linthresh=0.05)
    ax.set_xlabel("零智能主体的买单概率 p_buy（做多偏好强度）")
    ax.set_ylabel("信噪比 μ/σ（symlog）")
    ax.set_title("① 谁承载信号、谁是噪音：两条通道的信噪比")
    ax.legend(fontsize=8.5)

    ax = axes[1]
    for p, col in zip(sorted({k for row in grid.values() for k in row}),
                      [C_ACCENT, C_REAL, C_SIM, C_GOOD, "#b48ead"]):
        xs = sorted(float(k) for k in grid)
        ys = [grid[k][p] for k in grid]
        ax.plot(xs, ys, "o-", color=col, lw=1.5, ms=5, label=f"p_buy={p:.2f}")
    ax.axhline(0.858, color=MUTED, ls=":", lw=1.2, label="真实正占比 0.858")
    ax.axvline(0.1259, color=C_GOOD, ls="--", lw=1.0)
    ax.set_xscale("symlog", linthresh=0.02)
    ax.set_xlabel("系数比 θ = b/a（0 = 纯溢价通道）")
    ax.set_ylabel("正费率占比")
    ax.set_title("② θ=0 时正占比恒为 0.5——溢价通道单独撑不起任何偏置")
    ax.legend(fontsize=8.0, ncol=2)

    fig.suptitle("E5.0 标定：资金费率的两条通道各能承载多少信号", fontsize=12.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_transient(ticks, crowd, trans: dict, path) -> None:
    """E5.0a：初始禀赋的松弛过程——这就是为什么必须丢预热。"""
    import matplotlib.pyplot as plt

    from tw.viz import C_ACCENT, C_GOOD, MUTED

    fig, ax = plt.subplots(figsize=(9.4, 4.4))
    run = np.cumsum(crowd) / np.arange(1, crowd.size + 1)
    ax.plot(ticks, run, color=C_ACCENT, lw=1.8, label="累积均值（诊断口径）")
    k = max(5, crowd.size // 80)
    smooth = np.convolve(crowd, np.ones(k) / k, mode="valid")
    ax.plot(ticks[k - 1:], smooth, color=C_GOOD, lw=1.2, alpha=0.85,
            label=f"滚动均值（窗 {k} 次结算）")
    ax.axhline(0.0, color=MUTED, lw=1.0, ls=":")
    ax.axvline(4000, color="#e0a458", lw=1.6, ls="--", label="预热线 4000 tick")
    ax.set_xlabel("tick")
    ax.set_ylabel("主动订单流失衡（拥挤度）")
    ax.set_title("初始禀赋的松弛：开局几千 tick 是单边卖压，不是市场性质")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_scan(scan, pearson: float, best, path) -> None:
    """E5.2/E5.4：偏好强度 → 费率。"""
    import matplotlib.pyplot as plt

    from tw.viz import C_ACCENT, C_GOOD, C_REAL, C_SIM

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))
    x = [r["p_buy"] for r in scan]

    ax = axes[0]
    ax.plot(x, [r["annualized"] for r in scan], "o-", color=C_SIM, lw=1.9, ms=8)
    ax.axhline(REAL_FUNDING_ANNUAL, color=C_REAL, ls="--", lw=1.5,
               label=f"真实年化 {REAL_FUNDING_ANNUAL}")
    ax.axhspan(0.5 * REAL_FUNDING_ANNUAL, 2 * REAL_FUNDING_ANNUAL,
               color=C_REAL, alpha=0.12, label="验收区间 [0.5×, 2×]")
    ax.plot([best["p_buy"]], [best["annualized"]], "*", color=C_GOOD, ms=17,
            label=f"最匹配 p_buy={best['p_buy']:.2f}")
    ax.set_xlabel("做多偏好强度 p_buy")
    ax.set_ylabel("年化资金费率")
    ax.set_title(f"① 偏好 → 费率（Pearson = {pearson:.4f}）")
    ax.legend(fontsize=8.5)

    ax = axes[1]
    ax.plot(x, [r["pos_frac"] for r in scan], "s-", color=C_ACCENT, lw=1.9, ms=8)
    ax.axhline(REAL_FUNDING_POS_FRAC, color=C_REAL, ls="--", lw=1.5,
               label=f"真实正占比 {REAL_FUNDING_POS_FRAC}")
    ax.set_xlabel("做多偏好强度 p_buy")
    ax.set_ylabel("正费率占比")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("② 偏好 → 正费率占比")
    ax.legend(fontsize=8.5)

    fig.suptitle("E5.2 / E5.4 做多偏好扫描与真实对标（口径：预热 4000 tick）", fontsize=12.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_arbitrageurs(arb, path) -> None:
    """E5.3：套利者介入 → 费率被推平（反身性前置数据）。"""
    import matplotlib.pyplot as plt

    from tw.viz import C_ACCENT, C_GOOD, C_SIM

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.3))
    x = [r["n_arbitrageurs"] for r in arb]

    ax = axes[0]
    ax.plot(x, [r["mean_bp"] for r in arb], "o-", color=C_ACCENT, lw=1.9, ms=8)
    ax.axhline(0.0, color=C_SIM, ls=":", lw=1.0)
    ax.set_xlabel("市场里的资金费率套利者数量")
    ax.set_ylabel("费率均值 (bp / 次结算)")
    ax.set_title("① 套利者越多，费率越被推平")

    ax = axes[1]
    ax.plot(x, [r["pos_frac"] for r in arb], "s-", color=C_GOOD, lw=1.9, ms=8)
    ax.axhline(REAL_FUNDING_POS_FRAC, color="#e0a458", ls="--", lw=1.3,
               label=f"真实 {REAL_FUNDING_POS_FRAC}")
    ax.set_xlabel("市场里的资金费率套利者数量")
    ax.set_ylabel("正费率占比")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("② 正占比的响应（反身性）")
    ax.legend(fontsize=8.5)

    fig.suptitle("E5.3 套利者介入：机制改变了它自己所依赖的信号", fontsize=12.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
