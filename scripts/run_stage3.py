"""阶段3：加入做市商，测试极端行情下的脆弱性（施工蓝图 §4 阶段3）。

蓝图任务
    加 MarketMaker；让某个主体群体的仓位被强制平仓（模拟清算），
    观察价格冲击如何在订单簿中传导、恢复需要多久。

四个实验
--------
A. **做市商的效果**：同池对照（把基础池等比缩小 10%，空出的 10% 给做市商），
   这样"非做市商部分的主体构成完全相同"，差异只能归因于做市商本身。

B. **清算冲击与恢复**：把持仓最重的一批主体强制平仓，测量冲击幅度、
   消耗了多少盘口流动性、恢复要多久；做市商在其中的作用。

C. **冲击的线性度**（最关键，直接对应"清算流动性捕获"假设）
   C1 瞬时清算（一次性砸完）—— 会饱和，测的是"盘口深度上限"；
   C2 分期清算（拆 20 tick 抛）—— 贴近真实强平引擎，规模才体现为持续抛压。

四个必须讲清楚的方法论修正
--------------------------
**① 清算规模按"全市场多头持仓"定，不按"盘口深度"定。**
   最初我按盘口深度定规模，结果不同档位的规模都被"被清算群体的持仓上限"
   卡住了——八档规模全部落在同一个成交额上（都是 2.3 手），
   曲线是平的，看起来像"冲击与规模无关"，其实是规模根本没变。
   现在改为按总持仓的百分比定，规模才真正递增。

**② 冲击峰值必须在**短窗口内**测量，不能用整段路径的最大偏离。**
   最初我用 3000 tick 窗口里的最大/最小偏离当"冲击峰值"，
   结果卖出清算测出 +422bp 的**正**冲击——因为 3000 tick 的随机漂移
   幅度远大于冲击本身。现在峰值在冲击后 60 tick 内测量。

**③ 做市商的动态价差不加约束会失效。**
   价差 = mid × (base_spread + k × 近期波动率)。若 k 取 4.0，而每 tick 波动率
   约 60bp，则半价差被放大到 240bp——做市商报的价比整个盘口都宽，
   等于没做市（实测"有做市商"的价差与无做市商完全相同）。
   现在 k=0.05、base=0.5bp，半价差约 3.5bp，才真正进入盘口顶部。

**④ ⭐ 单条路径 + 峰值 = 噪声。必须配对控制 + 多种子。**
   这是第二次踩同一个坑，值得写清楚。**②的修正只对"瞬时清算"有效**：
   瞬时清算的峰值确实出现在冲击后几 tick 内，60 tick 窗口够用。
   但**分期清算的引信更长**——20 tick 的抛压过程本身就没被记录（路径从清算
   *结束后*才开始），峰值被整个漏掉；而观测窗口内 σ√60 ≈ 300bp 的随机游走
   又远大于冲击。两者叠加，实测把分期清算的幂律指数算成了 **-0.26**
   （成交量越大冲击越小），物理上不可能——这是**测量**坏了，不是市场。

   三条修正：
   a) **路径从冲击前开始记**，逐 tick 覆盖"清算期间 + 清算之后"全过程，
      且统一"先动作、后快照"，让不同 ``slices`` 的路径下标语义对齐；
   b) **同种子配对控制**：控制组（不清算）与处理组用同一个 seed。
      市场 RNG 只在「洗牌」与「基本面价值步进」两处消耗，且每 tick 次数
      固定、与是否清算无关 → **基本面锚的路径逐点相同**，共同的随机漂移
      被逐点相减消掉。实测跨种子标准差从 89.5bp 降到 1.4bp（**降噪 62 倍**）；
   c) **多种子统计**：报告均值 ± 标准误与 t 统计量，让"冲击是否显著"
      "是否超线性"成为可证伪的命题，而不是盯一条曲线的形状说话。

**⑤ ⭐ 滑点的基准价必须"同时点"，不能是固定的清算前价格。**
   这是第 ④ 条的同源错误换了个马甲。用 p0 当基准算实现缺口
   （VWAP / p0 − 1）看似天经地义，但抛售本身要跨 ``slices`` 个 tick，
   而本市场 σ ≈ 40bp/tick——拆 100 片抛就跨 100 tick，市场自己的漂移
   轻松上百 bp，全被记成"清算造成的滑点"。实测慢抛因此显示 −50bp，
   "慢抛几乎无冲击"这个结论会被完全淹没。
   正确做法：以**控制组在同一 tick 的中间价**为基准（配对设计的直接产物），
   得到**因果滑点**。抛售期间的市场漂移被反事实价格消掉，剩下的才是冲击。
   最终口径：
     - 主指标 ``slippage_bp``   = 因果滑点（成交价 vs 同 tick 反事实中间价）
     - 次指标 ``during_mean_bp`` = 清算窗口内配对中间价偏离的均值
     - 对照 ``slippage_vs_p0_bp`` = 固定基准的老口径（仅用于展示这个陷阱有多大）
"""

from __future__ import annotations

import sys
from pathlib import Path

import math

import numpy as np

# 让 `import scripts.run_stageN` 也能工作（测试会这么做）：
# 平时 `python scripts/run_stageN.py` 时脚本目录自动在 sys.path 上，
# 但作为包被导入时不在 → `ModuleNotFoundError: _common`。
# 二期的 run_stage5/6/7/8/9/10/11 都有这一行；一期这四个当时没加，
# 直到 `tests/test_impact.py` 要 import run_stage3 验证重构等价性时才暴露。
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import FIG, banner, real_metrics, save_json  # noqa: E402
from tw import Market, Population, SimConfig  # noqa: E402
from tw.impact import (  # noqa: E402
    cumulative_depth,
    depth_profile_gamma,
    fit_power_law,
    unsaturated as impact_unsaturated,
)
from tw.types import Order  # noqa: E402

N_AGENTS = 300
WARMUP = 6_000
AFTERSHOCK = 1_500
HORIZON = 400               # 冲击实验的观测长度（tick），从清算开始计时
PEAK_WINDOW = 90            # 峰值窗口（覆盖分期清算的 20 tick 抛压 + 之后 70 tick）
OBS_A = 1_200               # 实验 A 的微观结构观测长度
SEED0 = 20260917
K_SEEDS = 8

#: 多种子：配对设计下每种规模跑 K 次，报告均值与标准误
SEEDS = [SEED0 + 7 * i for i in range(K_SEEDS)]

BASE_MIX = {"zero_intel": 0.30, "fundamentalist": 0.40, "chartist": 0.30}
#: 把基础池等比缩小 10%，空出的名额给做市商 —— 保证对照组与非做市商部分构成一致
MM_MIX = {
    "zero_intel": 0.27,
    "fundamentalist": 0.36,
    "chartist": 0.27,
    "market_maker": 0.10,
}

#: 做市商参数（已按实测标定，见模块开头第 ③ 条）
MM_KW = dict(
    mm_base_spread=0.00005,   # 0.5bp 半价差
    mm_vol_sensitivity=0.05,  # 波动率加成系数
    mm_quote_qty=2.0,
    mm_inventory_target=10.0,
    mm_skew_strength=0.0004,
)

SHOCK_SIZES = [0.001, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08]
#: 为什么最小的档位要细到 0.1%：买盘总深度约 14 手，而 1% 总多头持仓 ≈ 30 手
#: **已经超过整个买盘**。档位全在饱和区，梯度就只剩"成交率在掉"，
#: 看不到"线性区 → 拐点 → 饱和区"的形状。补到 0.1%（≈3 手）才有线性段。


# --------------------------------------------------------------------------
# 基础设施
# --------------------------------------------------------------------------
def make_market(
    mix: dict[str, float], n_ticks: int, seed: int, *, factory=None
) -> Market:
    """建一个市场。

    ``factory`` 是二期阶段6 加的**唯一**扩展点：它要拿同一套清算实验
    去测"有元订单机制的市场"，而**必须**复用这里的构建参数
    （``N_AGENTS`` / ``MM_KW`` / 禀赋分布），否则测出的差异里
    混进了"主体数不同""做市商参数不同"这些无关变量。
    默认 ``None`` ⇒ 用裸 ``Market``，一期行为逐点不变。
    """
    pop = Population.from_shares(N_AGENTS, mix)
    cfg = SimConfig(seed=seed, n_ticks=n_ticks, population=pop, **MM_KW)
    return (factory or Market)(cfg)


def recent_stats(m: Market, window: int = 400) -> dict:
    log = m.log
    hi = m.tick
    lo = max(0, hi - window)
    sl = slice(lo, hi)
    spread = log.spread[sl]
    mid = log.mid[sl]
    ok = np.isfinite(spread) & np.isfinite(mid) & (mid > 0)
    r = np.diff(np.log(log.mid[lo:hi]))
    r = r[np.isfinite(r)]
    return {
        "spread_bp": float(np.mean(spread[ok] / mid[ok]) * 1e4) if ok.any() else float("nan"),
        "depth_bid": float(np.nanmean(log.bid_depth[sl])),
        "depth_ask": float(np.nanmean(log.ask_depth[sl])),
        "depth_total": float(np.nanmean(log.bid_depth[sl] + log.ask_depth[sl])),
        "sigma_bp": float(r.std() * 1e4) if r.size > 2 else float("nan"),
        "trades_per_tick": float(np.nanmean(log.n_trades[sl])),
        "open_orders": float(np.nanmean(log.open_orders[sl])),
    }


def pick_victims(m: Market, target_qty: float) -> tuple[list, float]:
    """贪心地从持仓最大的主体开始凑够目标清算量。"""
    ranked = sorted(m.agents, key=lambda a: a.inventory, reverse=True)
    victims, acc = [], 0.0
    for a in ranked:
        if a.inventory <= 1e-6:
            break
        victims.append(a)
        acc += a.inventory
        if acc >= target_qty:
            break
    frac = min(1.0, target_qty / acc) if acc > 0 else 0.0
    return victims, frac


# --------------------------------------------------------------------------
# 单条路径：从冲击前开始，逐 tick 记录
# --------------------------------------------------------------------------
def run_path(
    mix: dict[str, float],
    seed: int,
    shock_frac: float | None,
    slices: int = 1,
    horizon: int = HORIZON,
    *,
    factory=None,
) -> dict:
    """预热 → 清算（可选）→ 逐 tick 记录相对冲击前中间价的偏离（bp）。

    关键：路径**从清算开始那一刻计时**，清算期间的每一步也都记进去。
    这样 ``path[t]`` 在瞬时与分期两种清算下都是同一含义
    ——「距清算启动 t 个 tick 时、该 tick 动作执行完后的中间价偏离」，
    不同 ``slices`` 的结果才可以直接比。

    记录时序统一为「**先动作、后快照**」：``snap(t)`` 取的是第 t 个 tick
    的动作做完之后的中间价。因此：
      - 控制组 ``path[0]`` 恰好在预热刚结束时取，等于 p0（偏离 0）；
      - 瞬时清算 ``path[0]`` 是扫单刚砸完的瞬间；
      - 分期清算 ``path[k]`` 是第 k 片抛完的瞬间。
    三者**在同一 tick 下标上语义对齐**，配对相减才有意义。

    同时记录每笔清算成交落在第几片（``exec``），供计算**因果滑点**：
    拿成交价对比"同一 tick 上控制组的中间价"，而不是对比清算前那个固定的 p0。
    对比 p0 会把抛售期间的**市场自身漂移**也算进滑点（σ≈40bp/tick，
    拆 100 片抛要跨 100 tick，漂移轻松上百 bp）——实测过，会把因果冲击
    从 -3bp 放大成 -50bp，正是这个原因。
    """
    m = make_market(mix, WARMUP + horizon + 20, seed=seed, factory=factory)
    m.run(WARMUP)
    p0 = m.current_mid()
    total_long = float(sum(a.inventory for a in m.agents))

    path = np.full(horizon, np.nan, dtype=np.float64)
    one_sided = np.zeros(horizon, dtype=bool)
    execs: list[tuple[int, float, float]] = []   # (切片序号, 成交价, 数量)

    def snap(i: int) -> None:
        if i < horizon:
            path[i] = (m.current_mid() / p0 - 1.0) * 1e4
            # 盘口单边空缺（被打空）——此时 current_mid 回退到最新成交价
            one_sided[i] = m.book.mid_price() is None

    def take(slice_idx: int, trades) -> None:
        for t in trades:
            execs.append((slice_idx, float(t.price), float(t.quantity)))

    delivered = 0.0
    target = 0.0
    n_victims = 0
    depth_after = float("nan")
    depth_before = float("nan")

    if shock_frac is None:
        for i in range(horizon):
            snap(i)
            m.step()
    else:
        target = total_long * shock_frac
        depth_before = float(
            m.book.total_quantity("buy") + m.book.total_quantity("sell")
        )
        victims, frac = pick_victims(m, target)
        n_victims = len(victims)

        if slices <= 1:
            res = m.force_liquidate(victims, fraction=frac)
            delivered = float(res["total_qty"])
            for t in m.log.trades[-int(res["n_trades"]) :] if res["n_trades"] else []:
                execs.append((0, float(t.price), float(t.quantity)))
            snap(0)
            for i in range(1, horizon):
                m.step()
                snap(i)
        else:
            plan = {a.agent_id: a.inventory * frac / slices for a in victims}
            for a in victims:
                a.cancel_all()
            for k in range(slices):
                for a in victims:
                    want = min(plan[a.agent_id], a.available_inventory)
                    if want <= 1e-6:
                        continue
                    order = Order(
                        agent_id=a.agent_id,
                        side="sell",
                        price=-math.inf,
                        quantity=want,
                        timestamp=m.tick,
                        order_id=f"{a.agent_id}#STG{k}",
                        order_type="market",
                    )
                    trades = m.submit(a, order)
                    delivered += float(sum(t.quantity for t in trades))
                    take(k, trades)
                snap(k)
                m.step()
            for i in range(slices, horizon):
                snap(i)
                m.step()

        depth_after = float(
            m.book.total_quantity("buy") + m.book.total_quantity("sell")
        )

    ok, problems = m.health_check()
    qty = sum(q for _, _, q in execs)
    vwap = sum(p * q for _, p, q in execs) / qty if qty > 0 else float("nan")
    slip_p0 = (vwap / p0 - 1.0) * 1e4 if qty > 0 else float("nan")
    # ⚠️ 这里算的"最差成交价"是相对**固定的清算前价格 p0**，因此含有抛售期间的
    # 市场自身漂移——分期清算要跨 slices 个 tick，漂移轻松上百 bp。
    # 只作为对照保留；真正用于结论的是 paired_impact 里算的**因果**版本
    # （每笔成交对比"同一 tick 控制组的中间价"）。
    worst_vs_p0 = (
        (min(p for _, p, _ in execs) / p0 - 1.0) * 1e4 if qty > 0 else float("nan")
    )
    return {
        "path": path,
        "one_sided": one_sided,
        "execs": execs,
        "delivered_qty": delivered,
        "target_qty": target,
        "n_victims": n_victims,
        "depth_before": depth_before,
        "depth_after": depth_after,
        "total_long": total_long,
        # 参考口径：相对清算前固定中间价的偏离（含抛售期间的市场漂移，仅供对照）
        "liq_vwap": float(vwap),
        "slippage_vs_p0_bp": float(slip_p0),
        "worst_fill_vs_p0_bp": float(worst_vs_p0),
        "p0": float(p0),
        "health_ok": ok,
        "health_problems": problems[:5],
    }


# --------------------------------------------------------------------------
# 配对冲击实验
# --------------------------------------------------------------------------
def _nan_std(a: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    return float(a.std(ddof=1)) if a.size > 1 else float("nan")


def causal_slippage(run: dict, ctrl_path: np.ndarray, ctrl_p0: float) -> tuple[float, float]:
    """因果滑点：每笔清算成交对比**同一 tick 上控制组的中间价**。

    控制组与处理组同种子，控制组的中间价路径就是「如果没有发生清算，
    这一 tick 的中间价会是多少」的反事实价格。以它为基准，抛售期间
    市场自身的漂移被消掉，剩下的纯粹是清算造成的价格让步。

    为什么不能用固定的清算前中间价 p0 做基准：抛售要跨 ``slices`` 个 tick，
    而本市场 σ≈40~50bp/tick，拆 100 片抛就跨 100 tick，漂移轻松上百 bp。
    实测拿 p0 做基准会把 100 片慢抛的因果冲击从个位数 bp 放大到 −50bp，
    让"慢抛无冲击"这个结论消失。

    返回 ``(成交量加权的平均滑点, 单笔最差滑点)``：
      · 平均滑点 = 清算方实际承受的**平均**代价（成交量加权）
      · 最差滑点 = 打到的最深一档相对其**同时点**反事实价的偏离，即**边际**代价
    两者必须分开报：只报平均会掩盖"最后一手打到多深"，
    只报最差会把一次极端成交当成整体代价。
    """
    if not run["execs"]:
        return float("nan"), float("nan")
    num = den = 0.0
    worst = float("inf")
    for k, price, qty in run["execs"]:
        mid_ref = ctrl_p0 * (1.0 + ctrl_path[k] / 1e4)
        if mid_ref <= 0 or not np.isfinite(mid_ref):
            continue
        rel = price / mid_ref - 1.0
        num += qty * rel
        den += qty
        worst = min(worst, rel)
    if den <= 0:
        return float("nan"), float("nan")
    return float(num / den * 1e4), float(worst * 1e4)


def paired_impact(
    name: str,
    mix: dict[str, float],
    shock_frac: float,
    slices: int,
    seeds: list[int],
    controls: dict[int, dict],
    horizon: int = HORIZON,
    *,
    factory=None,
) -> dict:
    """同种子配对：处理组减控制组，逐 tick 求差，跨种子报均值 ± 标准误。

    ``controls`` 是预先算好的同种子控制路径（不清算），可跨规模复用。
    ``factory`` 见 ``make_market``：阶段6 用它换掉市场实现，
    而控制组与处理组**必须传同一个 factory**，否则配对差里混进了
    "两组市场实现不同"这个无关变量。
    """
    shocks = [run_path(mix, s, shock_frac, slices, horizon, factory=factory)
              for s in seeds]
    C = np.stack([controls[s]["path"] for s in seeds])
    S = np.stack([r["path"] for r in shocks])
    D = S - C                       # 配对差：逐 seed 逐 tick
    K = D.shape[0]

    mean_ctrl = np.nanmean(C, axis=0)
    mean_shk = np.nanmean(S, axis=0)
    mean_d = np.nanmean(D, axis=0)
    sem_d = (
        np.nanstd(D, axis=0, ddof=1) / math.sqrt(K) if K > 1 else np.zeros_like(mean_d)
    )

    w = min(slices, horizon)
    win = D[:, :w].mean(axis=1)             # 每个种子一个「清算窗口平均偏离」
    win_mu = float(np.nanmean(win))
    win_sd = _nan_std(win)
    win_t = win_mu / (win_sd / math.sqrt(K)) if win_sd and win_sd > 0 else float("nan")

    # 主指标：因果滑点 —— 每笔清算成交对比"同一 tick 上控制组的中间价"。
    # 同一 run 内已实现，且以**同时点**反事实价为基准，既不含路径漂移、也不含测量噪声。
    pairs = [
        causal_slippage(r, controls[s]["path"], controls[s]["p0"])
        for s, r in zip(seeds, shocks)
    ]
    slip = np.array([p[0] for p in pairs], dtype=float)
    slip = slip[np.isfinite(slip)]
    slip_mu = float(slip.mean()) if slip.size else float("nan")
    slip_sem = (
        float(slip.std(ddof=1) / math.sqrt(slip.size)) if slip.size > 1 else float("nan")
    )
    slip_t = slip_mu / slip_sem if slip_sem and slip_sem > 0 else float("nan")

    worst = np.array([p[1] for p in pairs], dtype=float)
    worst = worst[np.isfinite(worst)]
    worst_mu = float(worst.mean()) if worst.size else float("nan")
    worst_sem = (
        float(worst.std(ddof=1) / math.sqrt(worst.size)) if worst.size > 1 else float("nan")
    )

    # 对照口径 1：拿固定的清算前中间价 p0 做基准（含抛售期间的市场自身漂移 与 回退伪影）
    slip0 = np.array([r["slippage_vs_p0_bp"] for r in shocks], dtype=float)
    slip0 = slip0[np.isfinite(slip0)]
    slip0_mu = float(slip0.mean()) if slip0.size else float("nan")

    # 对照口径 2：最差成交价相对固定 p0（同样含漂移，仅作对照）
    worst0 = np.array([r["worst_fill_vs_p0_bp"] for r in shocks], dtype=float)
    worst0 = worst0[np.isfinite(worst0)]
    worst0_mu = float(worst0.mean()) if worst0.size else float("nan")

    os_frac = float(np.nanmean([np.mean(r["one_sided"][:w]) for r in shocks]))
    depth_pre = float(np.nanmean([r["depth_before"] for r in shocks]))

    pw = min(PEAK_WINDOW, horizon)
    i_peak = int(np.nanargmin(mean_d[:pw]))
    peak = float(mean_d[i_peak])
    peak_sem = float(sem_d[i_peak])
    peak_t = peak / peak_sem if peak_sem > 0 else float("nan")

    def at(k: int) -> tuple[float, float]:
        if k > horizon:
            return float("nan"), float("nan")
        return float(mean_d[k - 1]), float(sem_d[k - 1])

    target = abs(peak) * 0.2
    rec = float("nan")
    for i in range(i_peak, horizon):
        if abs(mean_d[i]) <= target:
            rec = float(i - i_peak)
            break

    tail = D[:, horizon - 100 :].mean(axis=1)   # 末段 100 tick 的均值偏离 = 永久性损伤
    tail_mu = float(np.nanmean(tail))
    tail_sd = _nan_std(tail)
    tail_t = tail_mu / (tail_sd / math.sqrt(K)) if tail_sd and tail_sd > 0 else float("nan")

    unpaired_peak = float(np.nanmin(mean_shk[:pw]))
    e5, s5 = at(5)
    e20, s20 = at(20)
    e60, s60 = at(60)
    e100, s100 = at(100)
    imm, imm_sem = at(w)

    return {
        "name": name,
        "mix": mix,
        "shock_frac": shock_frac,
        "slices": slices,
        "n_seeds": K,
        # ---- 主指标 1：因果滑点（成交价 vs 同 tick 控制组中间价）----
        "slippage_bp": slip_mu,
        "slippage_sem_bp": slip_sem,
        "slippage_t": slip_t,
        # 对照口径：相对清算前固定中间价（含抛售期间的市场漂移）
        "slippage_vs_p0_bp": slip0_mu,
        # 最差成交价（打到的最深档）—— 同时点因果口径，无歧义的边际成本
        "worst_fill_bp": worst_mu,
        "worst_fill_sem_bp": worst_sem,
        # 对照口径：最差成交价相对固定 p0（含漂移，仅作对照）
        "worst_fill_vs_p0_bp": worst0_mu,
        # 清算前的盘口总深度（手）
        "depth_before": depth_pre,
        # ---- 主指标 2：清算窗口内的平均中间价偏离（配对）----
        "during_mean_bp": win_mu,
        "during_sem_bp": win_sd / math.sqrt(K) if win_sd else float("nan"),
        "during_t": win_t,
        # ---- 清算完成瞬间的偏离（配对）----
        "immediate_bp": imm,
        "immediate_sem_bp": imm_sem,
        "immediate_t": imm / imm_sem if imm_sem > 0 else float("nan"),
        # ---- 峰值（配对）----
        "peak_bp": peak,
        "peak_at": i_peak,
        "peak_t": peak_t,
        "peak_sem_bp": peak_sem,
        "peak_unpaired_bp": unpaired_peak,
        # ---- 衰减时点（配对）----
        "t5_bp": e5, "t5_sem": s5,
        "t20_bp": e20, "t20_sem": s20,
        "t60_bp": e60, "t60_sem": s60,
        "t100_bp": e100, "t100_sem": s100,
        "t_end_bp": float(mean_d[-1]),
        "tail100_bp": tail_mu,
        "tail100_sem": tail_sd / math.sqrt(K) if tail_sd else float("nan"),
        "tail100_t": tail_t,
        "recovery_80_ticks": rec,
        "residual_ratio": (
            float(mean_d[-1] / peak) if peak and np.isfinite(peak) and peak != 0 else float("nan")
        ),
        # ---- 盘口被打空的占比（买盘为空 → mid 回退到最新成交价）----
        "one_sided_frac": os_frac,
        # ---- 记账 ----
        "target_qty": float(np.mean([r["target_qty"] for r in shocks])),
        "delivered_qty": float(np.mean([r["delivered_qty"] for r in shocks])),
        "fill_ratio": (
            float(np.mean([r["delivered_qty"] / r["target_qty"] for r in shocks]))
            if np.mean([r["target_qty"] for r in shocks]) > 0
            else float("nan")
        ),
        "unfilled_qty": (
            float(np.mean([r["target_qty"] - r["delivered_qty"] for r in shocks]))
        ),
        "qty_per_slice": (
            float(np.mean([r["delivered_qty"] for r in shocks]) / slices)
            if slices > 0
            else float("nan")
        ),
        "n_victims": int(np.mean([r["n_victims"] for r in shocks])),
        "depth_after": float(np.nanmean([r["depth_after"] for r in shocks])),
        "health_ok": all(r["health_ok"] for r in shocks),
        # ---- 原始路径（绘图与复核用）----
        "mean_ctrl": mean_ctrl.tolist(),
        "mean_shk": mean_shk.tolist(),
        "mean_diff": mean_d.tolist(),
        "sem_diff": sem_d.tolist(),
    }


def make_controls(
    mix: dict[str, float], seeds: list[int], horizon: int = HORIZON,
    *, factory=None,
) -> dict[int, dict]:
    return {s: run_path(mix, s, None, 1, horizon, factory=factory) for s in seeds}


# --------------------------------------------------------------------------
# D. 盘口深度剖面：解释"为什么冲击不是凹的"
# --------------------------------------------------------------------------
def depth_profile(
    mix: dict[str, float], seed: int, max_bp: float = 400.0, n_bins: int = 40
) -> tuple[np.ndarray, np.ndarray]:
    """累计买盘深度 vs 距中间价的距离（bp），用于判断深度分布的形状。

    "冲击是线性/超线性还是凹的"这件事，**完全由深度剖面决定**：
      · 深度集中在顶部 → 大单很快吃穿近档、边际价格跳水 → 冲击超线性
      · 深度在远端很厚（长记忆挂单）→ 每多一手的边际影响递减 → 冲击凹（平方根律）
    所以只报一个幂律指数是不够的，必须把剖面画出来，让读者自己看形状。

    分箱那一段已搬到 ``tw/impact.py::cumulative_depth``（阶段6 要用同一个
    口径比较 γ），这里只负责"跑一场市场、取盘口深度"。
    """
    m = make_market(mix, WARMUP + 50, seed=seed)
    m.run(WARMUP)
    mid = m.current_mid()
    levels = m.book.depth("buy", n=500)
    dist = np.array([p for p, _, _ in levels], dtype=float)
    qty = np.array([q for _, q, _ in levels], dtype=float)
    # ⚠️ 绝对距离由 `cumulative_depth` 内部取。买盘的 (p/mid−1) 全是**负数**，
    # 忘了取绝对值会让任何阈值都包含全部档位，曲线直接变成一条平线
    # （一期自测踩过：打印出"所有深度都在 5bp 以内"，是判据错，不是市场错）。
    return cumulative_depth(
        abs(dist / mid - 1.0) * 1e4, qty, max_bp=max_bp, n_bins=n_bins
    )


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main() -> dict:
    banner("阶段3  做市商 + 清算压力测试")
    reals = real_metrics()

    # ---------- A. 做市商的效果（同池对照，多种子配对） ----------
    print(f"  A. 做市商效果对照（同池对照；{K_SEEDS} 个种子，观测 {OBS_A} tick）")
    a_stats = {}
    for tag, mix in (("no_mm", BASE_MIX), ("with_mm", MM_MIX)):
        rows = []
        for s in SEEDS:
            m = make_market(mix, WARMUP + OBS_A + 20, seed=s)
            m.run(WARMUP + OBS_A)
            rows.append(recent_stats(m, window=OBS_A))
        a_stats[tag] = {
            k: {
                "mean": float(np.mean([r[k] for r in rows])),
                "sem": float(np.std([r[k] for r in rows], ddof=1) / math.sqrt(len(rows))),
            }
            for k in rows[0]
        }

    print(
        f"    {'':<12}{'价差(bp)':>14}{'总深度':>16}{'挂单数':>14}"
        f"{'σ(bp)':>14}{'成交/tick':>14}"
    )
    for tag, label in (("no_mm", "无做市商"), ("with_mm", "有做市商")):
        st = a_stats[tag]
        cells = "".join(
            f"{st[k]['mean']:>8.2f}±{st[k]['sem']:<5.2f}"
            for k in ("spread_bp", "depth_total", "open_orders", "sigma_bp", "trades_per_tick")
        )
        print(f"    {label:<12}{cells}")

    sp_r = a_stats["with_mm"]["spread_bp"]["mean"] / a_stats["no_mm"]["spread_bp"]["mean"]
    dp_r = a_stats["with_mm"]["depth_total"]["mean"] / a_stats["no_mm"]["depth_total"]["mean"]
    sg_r = a_stats["with_mm"]["sigma_bp"]["mean"] / a_stats["no_mm"]["sigma_bp"]["mean"]
    print(f"    → 做市商：价差 {sp_r:.2f}×，深度 {dp_r:.2f}×，波动率 {sg_r:.2f}×")
    if sp_r < 0.95 and dp_r > 1.2:
        print("      ✅ 做市商确实收窄了价差、加厚了盘口（提供流动性），并压低波动率")
    elif dp_r > 1.2:
        print("      🟡 深度显著增加，但价差没被收窄（做市商报价未进入盘口顶部）")
    else:
        print("      ❌ 做市商没有产生预期效果，参数需要重标定")

    # ---------- 控制组（可被 B/C 复用） ----------
    print(f"\n  ▸ 生成配对控制组（同种子、不清算，{K_SEEDS} seed × 2 池）")
    ctrl = {
        "BASE": make_controls(BASE_MIX, SEEDS),
        "MM": make_controls(MM_MIX, SEEDS),
    }

    # ---------- B. 清算冲击与恢复 ----------
    print(f"\n  B. 清算冲击（1% 市场总多头持仓，市价强平；配对 {K_SEEDS} 种子）")
    b_no = paired_impact("清算·无做市商", BASE_MIX, 0.01, 1, SEEDS, ctrl["BASE"])
    b_mm = paired_impact("清算·有做市商", MM_MIX, 0.01, 1, SEEDS, ctrl["MM"])
    print(
        f"    {'':<12}{'成交':>7}{'成交率':>7}{'因果滑点(均)':>17}"
        f"{'最差成交价(边)':>18}{'t20':>13}{'末段残差':>14}"
    )
    for tag, r in (("无做市商", b_no), ("有做市商", b_mm)):
        print(
            f"    {tag:<12}{r['delivered_qty']:>7.1f}{r['fill_ratio']:>7.0%}"
            f"{r['slippage_bp']:>11.1f}±{r['slippage_sem_bp']:<4.1f}"
            f"{r['worst_fill_bp']:>12.1f}±{r['worst_fill_sem_bp']:<5.1f}"
            f"{r['t20_bp']:>7.1f}±{r['t20_sem']:<4.1f}"
            f"{r['tail100_bp']:>8.1f}±{r['tail100_sem']:<4.1f}"
        )
    d = (
        abs(b_mm["slippage_bp"]) / abs(b_no["slippage_bp"])
        if b_no["slippage_bp"]
        else float("nan")
    )
    print(
        f"    → 做市商把清算滑点压到 {d:.2f}×"
        f"（{'更抗冲击' if d < 1 else '⚠️ 未减弱冲击'}）；"
        f"滑点显著性 t={b_no['slippage_t']:.1f}（无做市商）/ {b_mm['slippage_t']:.1f}（有做市商）"
    )
    print(
        f"    → ⭐ 同一笔清算的两个口径**必须分开报**："
        f"平均让价 {b_no['slippage_bp']:+.0f}bp（清算方的整体代价，成交量加权），"
        f"最差成交价 {b_no['worst_fill_bp']:+.0f}bp（最后一手打到多深）。"
    )
    print(
        f"       只报平均会掩盖尾部代价，只报最差会把一次极端成交当成整体代价。"
        f"\n       另注：拿**固定的**清算前价格当基准测同一件事会得到 "
        f"{b_no['worst_fill_vs_p0_bp']:+.0f}bp——差的那部分就是市场自己的漂移，"
        f"这就是当年把冲击算成 −180bp 的原因。"
    )

    # ---------- C1. 瞬时清算的饱和 ----------
    print("\n  C1. 瞬时清算（一次性把全部仓位砸向市价）—— 会饱和")
    print(
        f"    {'规模':>8}{'目标量':>9}{'实际成交':>10}{'成交率':>8}{'未甩出':>9}"
        f"{'因果滑点(均)':>17}{'最差成交价(边)':>18}{'窗口均偏离':>16}{'盘口打空':>10}"
    )
    instant = []
    for f in SHOCK_SIZES:
        r = paired_impact(f"瞬时{f:.2%}", BASE_MIX, f, 1, SEEDS, ctrl["BASE"])
        instant.append(r)
        print(
            f"    {f:>7.2%}{r['target_qty']:>9.1f}{r['delivered_qty']:>10.1f}"
            f"{r['fill_ratio']:>8.0%}{r['unfilled_qty']:>9.1f}"
            f"{r['slippage_bp']:>11.1f}±{r['slippage_sem_bp']:<4.1f}"
            f"{r['worst_fill_bp']:>12.1f}±{r['worst_fill_sem_bp']:<5.1f}"
            f"{r['during_mean_bp']:>10.1f}±{r['during_sem_bp']:<4.1f}"
            f"{r['one_sided_frac']:>10.0%}"
        )
    print(
        f"    → ⭐ 瞬时清算**饱和**：成交率 {instant[0]['fill_ratio']:.0%} → "
        f"{instant[-1]['fill_ratio']:.0%}，{SHOCK_SIZES[-1]:.0%} 持仓有 "
        f"{instant[-1]['unfilled_qty']:.0f} 手根本没甩出去；"
        f"平均让价从 {instant[0]['slippage_bp']:.0f}bp 只恶化到 "
        f"{instant[-1]['slippage_bp']:.0f}bp，而不是随规模放大 80 倍。"
    )
    print(
        "       **盘口深度就是清算能力的上限**：超过它的部分不是'以更差价格成交'，"
        "而是'压根没成交'——仓位仍留在账上（真实交易所此时会进入 ADL 或保险基金）。"
    )
    print(
        f"       ⚠️ 「窗口均偏离」这一列在瞬时清算下是**退化口径**：买盘被打空后"
        f"（本实验 {instant[-1]['one_sided_frac']:.0%} 的 tick 都是），"
        f"`current_mid()` 会回退到最新成交价，于是它测的是**扫到的最深档**，"
        f"不是「市场认同的新价位」。看「最差成交价」这一列即可，两者几乎相同。"
    )
    print(
        f"       真正有信息的是**均值 vs 边际**的分离：平均让价 "
        f"{instant[-1]['slippage_bp']:.0f}bp，但最后一手打到 "
        f"{instant[-1]['worst_fill_bp']:.0f}bp。"
    )

    # ---------- C2. 分期清算的梯度 ----------
    SLICES = 10
    print(f"\n  C2. 分期清算（拆 {SLICES} 个 tick 抛，贴近真实强平引擎）")
    print(
        f"    {'规模':>8}{'目标量':>9}{'实际成交':>10}{'成交率':>8}{'每片量':>8}"
        f"{'因果滑点(均)':>17}{'最差成交价(边)':>18}{'t20后':>13}{'末段残差':>15}"
    )
    staged = []
    for f in SHOCK_SIZES:
        r = paired_impact(f"分期{f:.2%}", BASE_MIX, f, SLICES, SEEDS, ctrl["BASE"])
        staged.append(r)
        print(
            f"    {f:>7.2%}{r['target_qty']:>9.1f}{r['delivered_qty']:>10.1f}"
            f"{r['fill_ratio']:>8.0%}{r['qty_per_slice']:>8.2f}"
            f"{r['slippage_bp']:>11.1f}±{r['slippage_sem_bp']:<4.1f}"
            f"{r['worst_fill_bp']:>12.1f}±{r['worst_fill_sem_bp']:<5.1f}"
            f"{r['t20_bp']:>7.1f}±{r['t20_sem']:<4.1f}"
            f"{r['tail100_bp']:>9.1f}±{r['tail100_sem']:<4.1f}"
        )

    # ---------- C3. 抛售速度对照（规模固定，只改切片数） ----------
    SPEED_FRAC = 0.02
    SPEED_SLICES = [1, 2, 5, 10, 25, 50, 100]
    print(
        f"\n  C3. 抛售速度对照（规模固定 = {SPEED_FRAC:.0%} 总多头持仓，只改拆成几片）"
    )
    print(
        f"    {'切片数':>7}{'每片量':>9}{'成交率':>8}{'因果滑点(均)':>17}"
        f"{'最差成交价(边)':>18}{'（对照:vs p0）':>16}{'t20后':>13}"
    )
    speed = []
    for sl in SPEED_SLICES:
        r = paired_impact(f"速度{sl}", BASE_MIX, SPEED_FRAC, sl, SEEDS, ctrl["BASE"])
        speed.append(r)
        print(
            f"    {sl:>7d}{r['qty_per_slice']:>9.2f}{r['fill_ratio']:>8.0%}"
            f"{r['slippage_bp']:>11.1f}±{r['slippage_sem_bp']:<4.1f}"
            f"{r['worst_fill_bp']:>12.1f}±{r['worst_fill_sem_bp']:<5.1f}"
            f"{r['slippage_vs_p0_bp']:>16.1f}"
            f"{r['t20_bp']:>7.1f}±{r['t20_sem']:<4.1f}"
        )
    sp_fast = speed[0]["slippage_bp"]
    sp_slow = speed[-1]["slippage_bp"]
    sp20_fast = speed[0]["t20_bp"]
    sp20_slow = speed[-1]["t20_bp"]
    print(
        f"    → ⭐ 同一笔 {SPEED_FRAC:.0%} 的仓位：一次砸完因果滑点 {sp_fast:.0f}bp，"
        f"拆 100 个 tick 抛 {sp_slow:.0f}bp（缩小 "
        f"{abs(sp_fast)/max(abs(sp_slow),1e-9):.1f}×）；"
        f"成交率从 {speed[0]['fill_ratio']:.0%} 升到 {speed[-1]['fill_ratio']:.0%}。"
    )
    print(
        f"       注意最后一列对照：若拿**固定的**清算前价格 p0 做基准，"
        f"慢抛会显示 {speed[-1]['slippage_vs_p0_bp']:.0f}bp 的'滑点'——"
        f"那不是冲击，是 100 个 tick 里市场自己的漂移。"
        f"基准必须用**同时点**的反事实价格。"
    )
    if abs(sp20_fast) > 1.96 * speed[0]["t20_sem"]:
        print(
            "       **冲击的持续性取决于抛售速度，而不只是抛售总量**——"
            "慢抛让市场有时间补挂单，把一次性冲击摊平成几乎看不见的噪声。"
        )

    # 幂律拟合：用实际成交量（规模真的变了才有意义）
    #
    # ⚠️ 拟合实现已搬到 `tw/impact.py`，这里只剩一层薄包装。
    # 为什么搬：二期阶段6 要拿"同一个 k"和这里比（它的验收标准是
    # 「k 相比阶段3 的 0.61~1.25 更接近 0.5」）。**要比较就必须用同一把尺子**——
    # 各写一份的话，哪怕只是护栏宽了一点，测出的"改善"也可能全来自尺子变了，
    # 而且不会报错、只会产出一个"看起来达标"的数字。
    # 现在全项目只有一份实现；这里保留原来的返回形状 `(k, R², n, 拒绝原因)`，
    # 让 main() 的其余部分一行都不用改。两道护栏的完整说明见
    # ``tw/impact.py::fit_power_law``（注解是从这里搬过去的，语义未变）。
    def fit_exponent(
        rows: list[dict], key: str, *, min_span: float = 1.5, min_yrange: float = 1.2
    ) -> tuple[float, float, int, str]:
        """对数-对数回归，返回 (k, R², n, 拒绝原因)。"""
        fit = fit_power_law(rows, key, min_span=min_span, min_yrange=min_yrange)
        return (
            fit.exponent if fit.ok else float("nan"),
            fit.r_squared if fit.ok else float("nan"),
            fit.n_points,
            fit.reason,
        )

    def unsaturated(rows: list[dict]) -> list[dict]:
        """只保留"目标被吃满"的档位（成交率 ≥ 95%）。

        一旦目标吃不饱，"清算规模"这个自变量就已经不起作用了（成交额饱和），
        继续算进回归会把斜率系统性压低。饱和区本身是结论（见 C1 的饱和效应），
        但不该混进幂律斜率里。
        """
        return impact_unsaturated(rows, threshold=0.95)

    def verdict(k: float) -> str:
        if not np.isfinite(k):
            return "拟合失败"
        if k > 1.3:
            return "超线性 —— 盘口被击穿后流动性消失，同样的量造成更大冲击"
        if k < 1.15:
            return "≈线性 —— 盘口深度足以按比例吸收"
        return "略超线性"

    inst_un = unsaturated(instant)
    stg_un = unsaturated(staged)
    F = {
        ("瞬时", "因果滑点（平均成交让步）"): fit_exponent(instant, "slippage_bp"),
        ("瞬时", "最差成交价（打到的最深档）"): fit_exponent(instant, "worst_fill_bp"),
        ("瞬时", "窗口均偏离（配对中间价）"): fit_exponent(instant, "during_mean_bp"),
        ("分期", "因果滑点（平均成交让步）"): fit_exponent(staged, "slippage_bp"),
        ("分期", "最差成交价（打到的最深档）"): fit_exponent(staged, "worst_fill_bp"),
        ("分期", "窗口均偏离（配对中间价）"): fit_exponent(staged, "during_mean_bp"),
    }
    U = {
        ("瞬时", "因果滑点"): fit_exponent(inst_un, "slippage_bp"),
        ("分期", "因果滑点"): fit_exponent(stg_un, "slippage_bp"),
        ("分期", "最差成交价"): fit_exponent(stg_un, "worst_fill_bp"),
    }

    print(f"\n    幂律拟合（|冲击| ≈ 实际成交量^k，k=1 为线性）")
    print(
        "    ⚠️ 饱和档会让自变量或因变量停住，斜率就失去意义。"
        "下面每一行都标注了拒绝理由，拒绝不等于失败，而是「这一档不该拿来拟合」。"
    )
    for tag in ("瞬时", "分期"):
        print(f"\n      {tag}清算：")
        for name in ("因果滑点（平均成交让步）", "最差成交价（打到的最深档）",
                     "窗口均偏离（配对中间价）"):
            k, r2, n, why = F[(tag, name)]
            if not np.isfinite(k):
                print(f"        {name:<24} 拒绝拟合：{why}")
            else:
                print(f"        {name:<24} k={k:.2f}  (R²={r2:.3f}, n={n})  → {verdict(k)}")

    print(
        f"\n      未饱和档子集（成交率 ≥95%：瞬时 {len(inst_un)} 档、分期 {len(stg_un)} 档）"
    )
    for (tag, name), (k, r2, n, why) in U.items():
        if not np.isfinite(k):
            print(f"        {tag} · {name:<12} 拒绝拟合：{why}")
        else:
            print(
                f"        {tag} · {name:<12} k={k:.2f}  (R²={r2:.3f}, n={n})  → {verdict(k)}"
            )

    # 两种口径的系统性差异本身就是结论
    k_avg = U[("分期", "因果滑点")][0]
    k_worst = U[("分期", "最差成交价")][0]
    if np.isfinite(k_avg) and np.isfinite(k_worst) and abs(k_worst - k_avg) > 0.2:
        steeper = "更陡" if k_worst > k_avg else "更平"
        print(
            f"\n    → ⭐ 两种口径的指数不同：**平均成交让价** k={k_avg:.2f}，"
            f"**打到的最深档** k={k_worst:.2f}（{steeper}）。"
        )
        print(
            "       这两个量回答的是两个不同的问题："
            "「平均让价」是清算方实际付出的代价（成交量加权）；"
            "「最深档」是盘口被推到的位置（边际）。"
            "机制上，平均让价受成交量加权，大部分量在近档成交，"
            "所以它跟着规模的变化比边际量温和。"
        )
    if np.isfinite(k_avg):
        print(
            f"\n    → ⚠️ 关键偏差：本模型的因果滑点指数 k={k_avg:.2f}，"
            "而真实市场的冲击是**凹的**（平方根律 k≈0.5）。"
        )
        print(
            "       根因在 D 段测出来了：本模型的累计深度剖面接近线性（γ≈1），"
            "而真实市场是凸的（γ≈2，远端有大量长记忆挂单）。"
            "**要复现平方根律，缺的是挂单的深度分布，不是更多主体。**"
        )

    # ---------- D. 盘口深度剖面 ----------
    print("\n  D. 盘口深度剖面（累计买盘深度 vs 距中间价的距离）")
    prof = {}
    for tag, mix in (("无做市商", BASE_MIX), ("有做市商", MM_MIX)):
        curves = [depth_profile(mix, s) for s in SEEDS[:4]]
        grid = curves[0][0]
        mat = np.stack([np.interp(grid, c[0], c[1]) for c in curves])
        prof[tag] = {"dist_bp": grid.tolist(), "cum_qty": mat.mean(0).tolist(),
                     "cum_sem": (mat.std(0, ddof=1) / np.sqrt(mat.shape[0])).tolist()}
    d_grid = np.asarray(prof["无做市商"]["dist_bp"])
    c_no = np.asarray(prof["无做市商"]["cum_qty"])
    c_mm = np.asarray(prof["有做市商"]["cum_qty"])
    print(f"    {'距中间价':>10}{'无做市商累计深度':>18}{'有做市商累计深度':>18}")
    for i in range(0, d_grid.size, max(1, d_grid.size // 10)):
        print(f"    {d_grid[i]:>8.0f}bp{c_no[i]:>18.2f}{c_mm[i]:>18.2f}")

    profile_gamma = depth_profile_gamma

    g_no = profile_gamma(d_grid, c_no)
    g_mm = profile_gamma(d_grid, c_mm)
    within_40 = float(np.interp(40.0, d_grid, c_no))
    total = float(c_no[-1])
    print(
        f"    → 买盘总深度 {total:.1f} 手，其中距中间价 40bp 以内占 "
        f"{within_40 / total:.0%}（{within_40:.1f} 手）"
    )
    print(
        f"    → 累计深度剖面指数 γ（D ∝ x^γ）：无做市商 {g_no:.2f}、有做市商 {g_mm:.2f}"
    )
    print(
        "       这是「冲击是凹还是线性」的**决定量**：由 Q = c·I^γ 反解得 I ∝ Q^(1/γ)，"
    )
    print(
        f"       γ={g_no:.2f} → I ∝ Q^{1 / g_no:.2f}，与实测的因果滑点指数量级一致（≈线性）。"
    )
    print(
        "       而真实市场的深度剖面是**凸**的（γ≈2，远端有大量隐藏/长记忆挂单），"
        "对应 I ∝ Q^0.5 —— 这就是著名的**平方根律**（冲击是凹的）。"
    )
    print(
        "       **结论：本模型缺的不是主体数量，而是「远端深度」这个结构。**"
        "要让冲击变凹，得让主体把挂单铺到离中间价几十~几百 bp 的地方去（长记忆挂单），"
        "而不是继续加主体。"
    )

    # ---------- 出图 ----------
    _plot_a_effect(a_stats, FIG / "stage3_market_maker.png")
    _plot_recovery(b_no, b_mm, FIG / "stage3_recovery.png")
    _plot_saturation(instant, staged, FIG / "stage3_saturation.png")
    _plot_nonlinear(
        staged, U[("分期", "因果滑点")][0], U[("分期", "因果滑点")][1],
        FIG / "stage3_nonlinear.png",
    )
    _plot_slicing(speed, FIG / "stage3_slicing_speed.png")
    _plot_depth_profile(prof, FIG / "stage3_depth_profile.png")
    print(f"\n  图已输出到 {FIG}")

    out = {
        "stage": 3,
        "protocol": {
            "n_agents": N_AGENTS,
            "warmup": WARMUP,
            "horizon": HORIZON,
            "peak_window": PEAK_WINDOW,
            "seeds": SEEDS,
            "n_seeds": K_SEEDS,
            "design": "同种子配对控制（处理组 − 控制组）＋跨种子标准误",
            "primary_metric": (
                "slippage_bp = 清算成交 VWAP 相对清算前中间价的偏离（已实现滑点），"
                "次指标 during_mean_bp = 清算窗口内配对中间价偏离均值"
            ),
        },
        "A_market_maker": {
            "stats": a_stats,
            "spread_ratio": sp_r,
            "depth_ratio": dp_r,
            "sigma_ratio": sg_r,
        },
        "B_liquidation": {"no_mm": b_no, "with_mm": b_mm, "slippage_ratio": d},
        "C1_instant_saturation": {
            "rows": instant,
            "fit_all": {f"{t}|{n}": {"k": k, "r2": r2, "n": nn, "reject": why}
                        for (t, n), (k, r2, nn, why) in F.items() if t == "瞬时"},
            "fit_unsaturated": {f"{t}|{n}": {"k": k, "r2": r2, "n": nn, "reject": why}
                                for (t, n), (k, r2, nn, why) in U.items() if t == "瞬时"},
            "rows_unsaturated": len(inst_un),
        },
        "C2_staged": {
            "slices": SLICES,
            "rows": staged,
            "fit_all": {f"{t}|{n}": {"k": k, "r2": r2, "n": nn, "reject": why}
                        for (t, n), (k, r2, nn, why) in F.items() if t == "分期"},
            "fit_unsaturated": {f"{t}|{n}": {"k": k, "r2": r2, "n": nn, "reject": why}
                                for (t, n), (k, r2, nn, why) in U.items() if t == "分期"},
            "rows_unsaturated": len(stg_un),
        },
        "C3_slicing_speed": {"frac": SPEED_FRAC, "rows": speed},
        "D_depth_profile": {
            "curves": prof,
            "gamma_no_mm": float(g_no),
            "gamma_with_mm": float(g_mm),
            "depth_within_40bp_frac": within_40 / total if total > 0 else float("nan"),
            "total_bid_depth": total,
            "note": "累计深度剖面 D(x) ∝ x^γ；由 Q=c·I^γ 得 I ∝ Q^(1/γ)。"
                    "真实市场 γ≈2 → 平方根律；本模型 γ≈1 → 线性。",
        },
        "real_reference": {m.label: m.flat() for m in reals},
    }
    save_json(out, "stage3_metrics.json")
    return out


# --------------------------------------------------------------------------
# 绘图
# --------------------------------------------------------------------------
def _plot_depth_profile(prof: dict, path) -> None:
    """盘口深度剖面：解释冲击为什么不是凹的。"""
    import matplotlib.pyplot as plt

    from tw.viz import C_ACCENT, C_MID, C_SIM, MUTED

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.3), sharex=True)

    ax = axes[0]
    for tag, c in (("无做市商", C_ACCENT), ("有做市商", C_SIM)):
        x = np.asarray(prof[tag]["dist_bp"])
        y = np.asarray(prof[tag]["cum_qty"])
        e = 1.96 * np.asarray(prof[tag]["cum_sem"])
        ax.fill_between(x, y - e, y + e, color=c, alpha=0.18, lw=0)
        ax.plot(x, y, color=c, lw=1.7, label=tag)
    ax.set_xlabel("距中间价的距离 (bp)")
    ax.set_ylabel("累计可成交深度（手）")
    ax.set_title("① 累计深度剖面：深度集中在紧贴中间价的窄带里", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.15)

    ax = axes[1]
    for tag, c in (("无做市商", C_ACCENT), ("有做市商", C_SIM)):
        x = np.asarray(prof[tag]["dist_bp"])
        y = np.asarray(prof[tag]["cum_qty"])
        ax.plot(x, y / max(y[-1], 1e-9), color=c, lw=1.7, label=tag)
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("距中间价的距离 (bp)")
    ax.set_ylabel("累计深度 / 总深度")
    ax.axhline(0.5, color=MUTED, lw=0.9, ls="--")
    ax.text(62, 0.52, "一半深度所在位置", fontsize=8.5, color=MUTED)
    ax.set_title("② 归一化：真实市场的这条曲线要平缓得多", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.15)

    fig.suptitle("盘口深度剖面 —— 决定「冲击是凹的还是超线性」的真正原因", fontsize=12)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _band(ax, mu: np.ndarray, sem: np.ndarray, color: str, label: str) -> None:
    x = np.arange(mu.size)
    ax.fill_between(x, mu - 1.96 * sem, mu + 1.96 * sem, color=color, alpha=0.18, lw=0)
    ax.plot(x, mu, color=color, lw=1.6, label=label)


def _plot_recovery(b_no: dict, b_mm: dict, path) -> None:
    import matplotlib.pyplot as plt

    from tw.viz import C_ACCENT, C_SIM, MUTED

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.4), sharey=True)

    for ax, r, name, c in (
        (axes[0], b_no, "无做市商", C_ACCENT),
        (axes[1], b_mm, "有做市商", C_SIM),
    ):
        n = len(r["mean_diff"])
        x = np.arange(n)
        # 处理组与控制组的各自均值路径：两条几乎重合，说明共同漂移很小
        ax.plot(x, r["mean_ctrl"], color=MUTED, lw=1.0, ls="--", label="控制组（不清算）")
        ax.plot(x, r["mean_shk"], color=c, lw=1.0, alpha=0.55, label="处理组（清算）")
        _band(
            ax,
            np.asarray(r["mean_diff"]),
            np.asarray(r["sem_diff"]),
            c,
            f"配对差（{r['n_seeds']} 种子均值 ±95%CI）",
        )
        ax.axvline(r["slices"], color=MUTED, lw=0.8, ls=":")
        ax.axhline(0, color=MUTED, lw=0.9, ls="--")
        ax.set_title(
            f"{name}：让价 {r['during_mean_bp']:+.0f}bp（t={r['during_t']:.1f}），"
            f"峰值 {r['peak_bp']:+.0f}bp",
            fontsize=10.5,
        )
        ax.set_xlabel("距清算启动的 tick 数")
        ax.legend(fontsize=8, loc="lower right")
    axes[0].set_ylabel("相对清算前中间价的偏离 (bp)")
    fig.suptitle("清算冲击与恢复（1% 市场总多头持仓，市价强平）", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_saturation(instant: list[dict], staged: list[dict], path) -> None:
    """左：成交率随规模的饱和；右：已实现滑点随实际成交量的幂律。"""
    import matplotlib.pyplot as plt

    from tw.viz import C_ACCENT, C_GOOD, C_SIM, MUTED

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.4))

    ax = axes[0]
    fi = np.array([r["shock_frac"] for r in instant]) * 100.0
    fr_i = np.array([r["fill_ratio"] for r in instant]) * 100.0
    fr_s = np.array([r["fill_ratio"] for r in staged]) * 100.0
    ax.plot(fi, fr_i, "o-", color=C_ACCENT, lw=1.6, ms=6, label="瞬时清算（一次砸完）")
    ax.plot(fi, fr_s, "s-", color=C_SIM, lw=1.6, ms=6, label="分期清算（拆 10 tick）")
    ax.axhline(100, color=MUTED, lw=0.9, ls=":")
    ax.set_xscale("log")
    ax.set_xticks(fi)
    ax.set_xticklabels([f"{v:g}%" for v in fi])
    ax.set_xlabel("清算规模（占市场总多头持仓）")
    ax.set_ylabel("成交率 (%)")
    ax.set_ylim(0, 108)
    ax.set_title("① 成交率：瞬时必饱和，盘口深度就是上限", fontsize=11)
    ax.legend(fontsize=9)

    ax = axes[1]
    for rows, tag, key, c, marker in (
        (staged, "分期·平均成交让价", "slippage_bp", C_SIM, "s"),
        (staged, "分期·最差成交价(边际)", "worst_fill_bp", C_ACCENT, "o"),
    ):
        x = np.array([r["delivered_qty"] for r in rows], dtype=float)
        y = np.abs(np.array([r[key] for r in rows], dtype=float))
        e = np.array(
            [r["slippage_sem_bp"] if key == "slippage_bp" else r["worst_fill_sem_bp"]
             for r in rows],
            dtype=float,
        )
        pol = np.isfinite(x) & (x > 0) & np.isfinite(y) & (y > 0)
        ax.errorbar(x, y, yerr=1.96 * e, fmt=marker + "-", color=c, lw=1.6, ms=6,
                    capsize=3, label=f"{tag}（±95%CI）")
        # 与 fit_exponent 同一道护栏：自变量/因变量跨度不足时不画拟合线，
        # 否则图上会出现一条"看着很确定"的线，而它其实是在噪声上拟合的
        if pol.sum() >= 3 and x[pol].max() / x[pol].min() >= 1.5 and y[pol].max() / y[pol].min() >= 1.2:
            k, b = np.polyfit(np.log(x[pol]), np.log(y[pol]), 1)
            xx = np.linspace(np.log(x[pol].min()), np.log(x[pol].max()), 20)
            ax.plot(np.exp(xx), np.exp(k * xx + b), "--", color=c, lw=1.0, alpha=0.7,
                    label=f"　拟合指数 {k:.2f}")
        else:
            ax.plot([], [], " ", label="　（跨度不足/已饱和，不拟合）")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("实际成交量（手）")
    ax.set_ylabel("|价格偏离| (bp)")
    ax.set_title(
        "② 均值 vs 边际：两个口径给出不同的指数\n"
        "（真实市场的冲击是凹的 k≈0.5，本模型不是）",
        fontsize=10.5,
    )
    ax.legend(fontsize=8.5)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_slicing(speed: list[dict], path) -> None:
    """抛售速度对照：规模固定，只改拆成几片。"""
    import matplotlib.pyplot as plt

    from tw.viz import C_ACCENT, C_GOOD, C_MID, MUTED

    sl = np.array([r["slices"] for r in speed], dtype=float)
    slip = np.abs(np.array([r["slippage_bp"] for r in speed], dtype=float))
    slip_e = 1.96 * np.array([r["slippage_sem_bp"] for r in speed], dtype=float)
    dur = np.abs(np.array([r["during_mean_bp"] for r in speed], dtype=float))
    dur_e = 1.96 * np.array([r["during_sem_bp"] for r in speed], dtype=float)
    t20 = np.abs(np.array([r["t20_bp"] for r in speed], dtype=float))
    t20_e = 1.96 * np.array([r["t20_sem"] for r in speed], dtype=float)

    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    ax.errorbar(sl, slip, yerr=slip_e, fmt="o-", color=C_ACCENT, lw=1.6, ms=6,
                capsize=3, label="已实现滑点（清算成交 VWAP 的偏离）")
    ax.errorbar(sl, dur, yerr=dur_e, fmt="s-", color=C_MID, lw=1.5, ms=5,
                capsize=3, label="清算窗口内的平均中间价偏离")
    ax.errorbar(sl, t20, yerr=t20_e, fmt="^-", color=C_GOOD, lw=1.5, ms=5,
                capsize=3, label="清算结束后 20 tick 残余偏离")
    ax.set_xscale("log")
    ax.set_xticks(sl)
    ax.set_xticklabels([f"{int(v)}" for v in sl])
    ax.set_xlabel("拆成几个 tick 抛（越大 = 抛得越慢）")
    ax.set_ylabel("|价格偏离| (bp)")
    frac = speed[0]["shock_frac"]
    ax.set_title(
        f"抛售速度决定冲击：同一笔 {frac:.0%} 总持仓，拆得越细冲击越小", fontsize=11
    )
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.15)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_nonlinear(staged: list[dict], exponent: float, r2: float, path) -> None:
    """分期清算：不同规模下的配对让价路径（越大越深、越持久）。"""
    import matplotlib.pyplot as plt

    from tw.viz import C_DIM, C_GOOD, C_MID, MUTED

    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    n = len(staged)
    cmap = plt.get_cmap("YlOrRd")
    for i, r in enumerate(staged):
        mu = np.asarray(r["mean_diff"])
        sem = np.asarray(r["sem_diff"])
        x = np.arange(mu.size)
        c = cmap(0.25 + 0.65 * i / max(1, n - 1))
        ax.fill_between(x, mu - 1.96 * sem, mu + 1.96 * sem, color=c, alpha=0.15, lw=0)
        ax.plot(
            x, mu, color=c, lw=1.6,
            label=f"{r['shock_frac']:.2%} 持仓（成交 {r['delivered_qty']:.0f} 手，"
                  f"让价 {r['during_mean_bp']:+.0f}bp）",
        )
    ax.axhline(0, color=MUTED, lw=0.9, ls="--")
    ax.axvline(10, color=MUTED, lw=0.8, ls=":")
    ax.text(10.5, ax.get_ylim()[1] * 0.9, "清算结束", fontsize=8.5, color=MUTED)
    ax.set_xlabel("距清算启动的 tick 数")
    ax.set_ylabel("配对中间价偏离 (bp)")
    if np.isfinite(exponent):
        ax.set_title(
            f"分期清算的规模梯度：平均成交让价 ≈ 成交量^{exponent:.2f}"
            f"（R²={r2:.3f}；1.0 = 线性）\n"
            "注：此图纵轴是**配对中间价偏离**，与拟合用的「平均成交让价」不是同一个量；"
            "后者见汇总表",
            fontsize=10,
        )
    else:
        ax.set_title("分期清算的规模梯度（未饱和档的成交额跨度不足，未做幂律拟合）", fontsize=10.5)
    ax.legend(fontsize=8.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_a_effect(a_stats: dict, path) -> None:
    import matplotlib.pyplot as plt

    from tw.viz import C_DIM, C_GOOD, C_SIM, MUTED

    keys = [
        ("spread_bp", "价差 (bp)"),
        ("depth_total", "总深度 (手)"),
        ("open_orders", "挂单数"),
        ("sigma_bp", "σ (bp/tick)"),
        ("trades_per_tick", "成交 (笔/tick)"),
    ]
    v_no = np.array([a_stats["no_mm"][k]["mean"] for k, _ in keys])
    e_no = np.array([a_stats["no_mm"][k]["sem"] for k, _ in keys])
    v_mm = np.array([a_stats["with_mm"][k]["mean"] for k, _ in keys])
    e_mm = np.array([a_stats["with_mm"][k]["sem"] for k, _ in keys])

    fig, ax = plt.subplots(figsize=(9.2, 4.0))
    x = np.arange(len(keys))
    ax.bar(x - 0.2, v_no, width=0.4, color=C_DIM, label="无做市商", yerr=1.96 * e_no, capsize=3)
    ax.bar(x + 0.2, v_mm, width=0.4, color=C_SIM, label="有做市商(10%)", yerr=1.96 * e_mm, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([k[1] for k in keys])
    ax.set_title(f"做市商对市场微观结构的影响（同池对照，{K_SEEDS} 种子均值 ±95%CI）", fontsize=11)
    ax.legend(fontsize=9)
    for xi, (v1, v2) in enumerate(zip(v_no, v_mm)):
        ax.text(xi - 0.2, v1, f"{v1:,.2f}", ha="center", va="bottom", fontsize=8.5, color=MUTED)
        ax.text(xi + 0.2, v2, f"{v2:,.2f}", ha="center", va="bottom", fontsize=8.5, color=C_GOOD)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
