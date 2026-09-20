"""反身性实验（二期阶段7）。

要回答的问题
------------
**当越来越多主体采用同一个"已知有效"的策略时，这个策略自己的有效性
会不会内生地衰减？**

对应真实世界的观测：BTC 资金费率 carry 的年化收益从 14.4% 衰减到 5%。

⭐ 两个必须写清楚的设计选择
==========================

**① 采用者是「已经学会、此后不再变化」的固定策略主体，不是学习主体。**
   ``FixedMomentumTrader``（固定 lookback），不是 ``AdaptiveTrendFollower``。
   理由：反身性实验要测的是"同一个策略，人多了还灵不灵"。
   如果采用者自己也在学习，它会随拥挤度调整参数——
   那么测到的 PnL 变化里混进了"它自己改好了"这一项，
   而我们要的是**同一个东西**在不同拥挤度下的表现。**被测对象必须是不变的。**

**② 总主体数保持不变（用采用者**替换**背景主体，而不是叠加）。**
   指导书 §4.4 的写法是"注入该数量的 agent"，隐含"叠加"。
   这里**刻意偏离**，理由是：
   · 叠加会让总主体数从 300 涨到 460（adopter_count=160），
     而一期已实测 σ 强烈依赖主体数 N（纯零智能池 N=100→500 时 σ 从 37→86bp）。
     于是"采用者变多"同时意味着"市场变大、流动性变多"，
     采用者 PnL 的变化里就分不清是**同业拥挤**还是**市场变厚**。
   · 替换设计下 N 恒定、背景池等比缩小，"同行的人变多"这一项才被隔离出来。
   · 代价：噪音交易者变少（后半段尤其明显），这是**已知且必须报告**的边界。
   两个设计都能做，但混在一起做就没有结论——所以选一个、说清楚、
   并把代价写进诚实边界。

**③ 采用者的禀赋必须完全相同。**
   每人发同一份现金与持仓（不按序号抽）。否则不同 adopter_count 之间
   比的是"谁的禀赋运气好"，而不是"谁更拥挤"。
   这是配对设计的直接要求：**除了被测变量，其余必须逐点相同。**
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..agents.adaptive_trend import FixedMomentumTrader, realized_pnl_bp
from ..analyzer import analyze
from ..config import Population, SimConfig
from ..market import Market

#: 采用者的固定禀赋（不按序号抽 → 跨 adopter_count 逐点可比）
ADOPTER_CASH = 6.0e5
ADOPTER_INVENTORY = 10.0

#: 采用者随机流的用途标签。
#: ⚠️ 必须是**整数**。numpy 的 ``SeedSequence`` 只接受整数熵，
#: 传字符串会直接 ``ValueError: unrecognized seed string``——
#: 本项目在 ``tw/agents/base.py::_stable_id_key`` 里已经踩过同一个坑。
#: 这里用 ``[seed, ENTROPY_ADOPTER, i]`` 而不是 ``[seed, "adopter", i]``。
ENTROPY_ADOPTER = 0xADA970


@dataclass(slots=True)
class ReflexivityConfig:
    """反身性扫描的配置。"""

    #: 采用者数量档位（替换背景主体，总主体数不变）
    adopter_counts: list[int] = field(
        default_factory=lambda: [0, 5, 10, 20, 40, 80]
    )
    #: 被采用的策略的参数。默认 lookback=20：
    #: 它是一期图表派在标定时的窗口（`SimConfig.ch_lookback = 50`，但
    #: 阶段4 的敏感性扫描显示短窗口的动量信号更强），
    #: 这里取 20 作为"历史上看起来最优"的代表。
    fixed_lookback: int = 20
    seeds: list[int] = field(default_factory=list)
    n_agents: int = 300
    warmup: int = 4_000
    n_ticks: int = 20_000
    #: 背景池构成（采用者按比例从这三类里替换）
    mix: dict[str, float] = field(
        default_factory=lambda: {"zero_intel": 0.30, "fundamentalist": 0.40,
                                 "chartist": 0.30}
    )

    def total_ticks(self) -> int:
        return self.warmup + self.n_ticks


def build_market(cfg: ReflexivityConfig, seed: int, adopter_count: int) -> Market:
    """建一个市场，把 ``adopter_count`` 个背景主体替换成固定动量采用者。

    ⚠️ **每个 (count, seed) 组合必须是完全独立初始化的市场对象。**
    复用或深拷贝一个已经跑过的 market 会把"市场记忆"带进来
    （订单簿残留、agent 持仓、日志位置），后面的档位就混进了前面档位的残留。
    指导书 §4.7 专门警告过这一条。
    """
    if adopter_count < 0:
        raise ValueError("adopter_count 不能为负")
    pool = max(1, cfg.n_agents - adopter_count)
    pop = Population.from_shares(pool, cfg.mix)
    m = Market(SimConfig(
        seed=seed, n_ticks=cfg.total_ticks(), population=pop,
        zi_p_buy=0.5,
    ))
    # 精简到恰好 pool 个（from_shares 的四舍五入可能多给一个）
    while len(m.agents) > pool:
        a = m.agents.pop()
        m.by_id.pop(a.agent_id, None)

    # 采用者：**同一份禀赋**、**同一个 rng 构造方式**，只有 agent_id 不同。
    # rng 用 [seed, ENTROPY_ADOPTER, i]：与主体序号解耦，且跨 adopter_count 可比
    # （第 i 个采用者在任何档位里都是同一条随机流）。
    for i in range(adopter_count):
        a = FixedMomentumTrader(
            f"fm{i:04d}", ADOPTER_CASH, ADOPTER_INVENTORY,
            np.random.default_rng([seed, ENTROPY_ADOPTER, i]),
            lookback=cfg.fixed_lookback,
        )
        a.bind_seed(seed)
        m.add_agent(a)
        a.set_initial_equity(m.current_mid())
    return m


def run_one(cfg: ReflexivityConfig, seed: int, adopter_count: int) -> dict:
    """跑一个 (count, seed) 组合并收集结果。"""
    m = build_market(cfg, seed, adopter_count)
    m.run(cfg.total_ticks())
    mid = m.current_mid()

    adopters = [a for a in m.agents if a.KIND == "fixed_momentum"]
    pnls = [realized_pnl_bp(a, mid) for a in adopters]
    pnls = [p for p in pnls if np.isfinite(p)]

    # 背景主体的活跃度（用于确认"替换"确实改变了背景池规模，
    # 而不是悄悄把市场变成了另一回事）
    bg = [a for a in m.agents if a.KIND != "fixed_momentum"]
    n_bg_trades = sum(a.stats.n_trades for a in bg)

    obs = np.asarray(m.log.mid[cfg.warmup: m.tick], dtype=float)
    stats = analyze(obs, "sim", vol_window=24).flat() if obs.size > 100 else {}

    ok, prob = m.health_check()
    return {
        "adopter_count": int(adopter_count),
        "seed": int(seed),
        # ⭐ ``seed_effective`` 是**市场实际用的**种子（从 cfg 里读回来）。
        # 为什么要单独记一个：配对设计的前提是"同种子下只改 adopter_count"。
        # 如果哪天有人在构造 SimConfig 时对 seed 做了加工
        # （比如 seed+count、seed*7），配对就**悄悄破了**——
        # 而返回值里仍然写着原始 seed，从外面完全看不出来。
        # 记下实际值，测试就能把配对钉住（见 tests/test_adaptive_trend.py）。
        "seed_effective": int(m.cfg.seed),
        "n_agents_total": len(m.agents),
        "n_background": len(bg),
        "n_adopters": len(adopters),
        "adopter_pnl_bp_mean": float(np.mean(pnls)) if pnls else float("nan"),
        "adopter_pnl_bp_median": float(np.median(pnls)) if pnls else float("nan"),
        "adopter_pnl_bp_sd": float(np.std(pnls, ddof=1)) if len(pnls) > 1 else float("nan"),
        "adopter_pnl_bp_per_agent": pnls,
        "adopter_trades": int(sum(a.stats.n_trades for a in adopters)),
        "adopter_volume": float(sum(a.stats.volume for a in adopters)),
        "background_trades": int(n_bg_trades),
        "n_trades": int(len(m.log.trades)),
        "health_ok": bool(ok),
        "health_problems": prob[:3],
        "market": {
            k: stats.get(k) for k in
            ("sigma_bp", "excess_kurtosis", "acf_abs_lag1", "acf_abs_mean_1_10",
             "acf_ret_lag1", "vr5", "hill_alpha_left")
        },
    }


def run_reflexivity_sweep(cfg: ReflexivityConfig) -> list[dict]:
    """对每个 (adopter_count, seed) 组合跑一次。

    **同种子配对**：同一个 seed 下，只改 adopter_count，其余构造完全相同。
    这样"PnL 随人数下降"才能干净地归因于拥挤，而不是随机波动。
    """
    seeds = cfg.seeds or [20260917]
    out: list[dict] = []
    for count in cfg.adopter_counts:
        for seed in seeds:
            out.append(run_one(cfg, seed, count))
    return out


def paired_by_seed(rows: list[dict], key: str = "adopter_pnl_bp_mean") -> dict:
    """把结果整理成 ``{seed: {count: value}}``，便于做逐种子配对。

    配对的做法：对每个种子，计算"各档相对 count=0 档的差"，
    再跨种子平均。这样市场层面的共同漂移（每种子各自的行情）被逐点消掉。
    """
    by_seed: dict[int, dict[int, float]] = {}
    for r in rows:
        by_seed.setdefault(r["seed"], {})[r["adopter_count"]] = r.get(key)
    return by_seed


def spearman(x, y) -> tuple[float, float]:
    """Spearman 秩相关与正态近似 p 值（双尾）。不引 scipy。

    ``rho = 1 − 6Σd²/(n(n²−1))``（无并列时的等价形式，这里用秩的 Pearson
    实现，对并列更稳健）。p 值用 ``t = rho·sqrt((n−2)/(1−rho²))`` 的正态近似。
    """
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    n = a.size
    if n < 3:
        return float("nan"), float("nan")

    def rank(v: np.ndarray) -> np.ndarray:
        order = np.argsort(v, kind="stable")
        r = np.empty(v.size, dtype=float)
        r[order] = np.arange(1, v.size + 1, dtype=float)
        # 平均秩处理并列
        sv = v[order]
        i = 0
        while i < sv.size:
            j = i
            while j + 1 < sv.size and sv[j + 1] == sv[i]:
                j += 1
            if j > i:
                r[order[i: j + 1]] = (i + j + 2) / 2.0
            i = j + 1
        return r

    ra, rb = rank(a), rank(b)
    sa, sb = ra.std(ddof=0), rb.std(ddof=0)
    if sa <= 0 or sb <= 0:
        return float("nan"), float("nan")
    rho = float(((ra - ra.mean()) * (rb - rb.mean())).mean() / (sa * sb))
    # ⚠️ **不要为了避开除零而把 rho 截断成 ±0.999999**。
    # 那样"完全单调"的输入会返回 0.999999 而不是 1.0——
    # 看似无害，但它**破坏了这个函数最基本的可验证性质**：
    # 完全单调的数据应该给出 ±1。阶段7 的验收判据要求"rho 显著为负"，
    # 一个被截断的数会让"到底是不是完全单调"这个判断失去依据。
    # 正确做法：rho 原样返回，只在算 t 时单独处理 |rho|→1 的极端。
    if rho >= 1.0:
        rho = 1.0
        t = float("inf")
    elif rho <= -1.0:
        rho = -1.0
        t = float("-inf")
    else:
        t = rho * np.sqrt((n - 2) / (1 - rho * rho))
    # 双尾正态近似
    p = 0.0 if not np.isfinite(t) else float(2.0 * (1.0 - _norm_cdf(abs(t))))
    return rho, p


def _norm_cdf(z: float) -> float:
    """标准正态 CDF（Abramowitz–Stegun 7.1.26 的 erf 近似，误差 < 1.5e-7）。"""
    import math

    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
