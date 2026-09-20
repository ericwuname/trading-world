"""Hawkes 自激到达的市场集成（二期阶段8）。

集成方式
--------
``HawkesMarket`` 覆盖 ``Market._limit_activity``：本 tick 能出手的主体数
由归一化强度 ρ(t) 决定，而 ρ(t) 由**刚刚发生的真实订单流**激发。

    ρ(t) = λ(t)/λ̄,  λ(t) = μ + α·Σ_{s<t} events_s·e^{−β(t−s)}

于是形成闭环：订单多 → 强度高 → 更多主体出手 → 订单更多。
这就是"到达的时间聚集性"在被建模的那件事。

⚠️ 三个必须写清楚的约束
======================

**① 平均活跃度必须不变。** ρ 的均值按构造等于 1，所以打开 Hawkes
   **不改变平均订单量**，只改变它的时间分布。
   这是 E8.2「叠加 Hawkes 有没有额外贡献」这个判断能成立的前提——
   否则测到的差异里混进了"交易变多了"。

**② 子集必须是均匀随机的。** 传进来的列表是按各主体**自己的**调度流
   抽出的优先级排序的，所以"取前 k 个"= 均匀随机抽 k 个。
   一旦 `_next_order` 改成按别的键排序，"取前 k 个"就会偏向某一类主体。

**③ 激发事件用"实际成交笔数"，不用"提交订单数"。**
   提交的单里有相当一部分被预算/持仓裁剪拒掉，它们**没有**成为市场事件。
   用提交数当激发源会让强度被"未成交的意图"抬高，
   而真实市场里别人的单没成交，我不会因此变急。
"""

from __future__ import annotations

import numpy as np

from ..market import Market
from ..config import Population, SimConfig
from .hawkes import HawkesConfig, HawkesProcess, rho_to_count


def calibrate_base_lambda(
    seeds=(20260917, 20260924), n_ticks: int = 2_000, n_agents: int = 300,
    mix: dict[str, float] | None = None, warmup: int = 500,
    *, sim_kw: dict | None = None, market_cls=None,
) -> float:
    """测出「Hawkes 关掉」时市场**每 tick 的平均成交笔数**。

    ⭐ 为什么 ``base_lambda`` 必须标定，不能拍
    ----------------------------------------
    E8.2 要回答的是"叠加 Hawkes 有没有额外贡献"。这个判断成立的前提是
    **打开 Hawkes 不改变平均活跃度**（只改变它的时间分布）。
    而 ρ 的长期均值由闭环的**不动点**决定：

        设 c = 每个主体每 tick 的平均成交笔数，N = 主体数，n = 分支比，
        E[events] = c·N·ρ̄，而平稳式给 E[λ] = λ̄(1−n) + n·E[events]
        ⇒ ρ̄ = λ̄(1−n) / (λ̄ − n·c·N)

    代进去就能看出：**只有当 λ̄ = c·N 时 ρ̄ 才等于 1**。
    若 λ̄ 取得比实际成交率小，ρ̄ 会系统性**偏高**（或反之），
    于是"聚集性变强"这个结论里混进了"平均交易量变了"。

    实测过两端的例子：
      · λ̄ 取 45 而该市场实际约 24 笔/tick ⇒ ρ̄ 只有 **0.71**
        （"打开 Hawkes 之后交易少了三成"）；
      · λ̄ 取 50 而该市场实际 **87.2** 笔/tick ⇒ ρ̄ ≈ **1.48** ≫ 1，
        撞上 ``rho_to_count`` 的上限 ⇒ **每个 tick 都切出全部主体**，
        稀疏化**从未发生**、机制哑火（见下面 ``sim_kw`` 的警告）。

    ⚠️⚠️ ``sim_kw`` 必须与实际使用 λ̄ 的那个市场**完全一致**
    ------------------------------------------------------------------
    λ̄ 是 ρ = λ/λ̄ 的**标尺**，标尺偏了 ρ 就整体偏，
    而 ``rho_to_count`` 会把 ρ 夹在 ``[0, n_agents]``——
    于是"机制在跑但几乎不起作用"，看起来像"机制效果弱"。

    **实测踩到**：E8.2 用 ``MM_MIX`` 建市场时带了 ``MM_KW``
    （``mm_base_spread`` / ``mm_quote_qty`` / ``mm_inventory_target`` …），
    但调用本函数时**没传**这些参数。结果 λ̄ = 50.09，
    而那个市场真实的成交率是 **87.22**（1.74 倍）⇒ ρ̄ ≈ 1.48 ⇒
    ``round(1.48 × 300) = 442`` 被夹到 300 ⇒ **每个 tick 都返回全部主体**
    ⇒ 机制实际哑火（实测 ``n_active`` 恒为 300.0）。
    而这一点从产物里**完全看不出来**——k 确实变了（1.395→0.481），
    很容易把"机制没生效"误读成"机制生效但效果有限"。

    所以：**标定与被使用的市场必须是同一个**。``sim_kw`` 就是为此加的；
    ``market_cls`` 允许用 Hawkes 关掉的同类市场来标定（口径最稳）。
    """
    mix = mix or {"zero_intel": 0.30, "fundamentalist": 0.40, "chartist": 0.30}
    vals = []
    for sd in seeds:
        cfg = SimConfig(seed=sd, n_ticks=n_ticks,
                        population=Population.from_shares(n_agents, mix),
                        **(sim_kw or {}))
        m = (market_cls or Market)(cfg)
        m.run(n_ticks)
        hi = m.tick
        lo = min(warmup, hi // 2)
        vals.append(float(len([t for t in m.log.trades if lo <= t.tick < hi])
                          / max(1, hi - lo)))
    return float(sum(vals) / len(vals))


class HawkesMarket(Market):
    """把订单到达交给自激点过程决定的市场。

    三个可选开关（都服务于"把 Hawkes 的效应拆开"这件事，默认全关）：

    ``rho_schedule``  **外生 ρ 序列**（方差匹配的代理对照）。
        给定时 ρ(t) 直接读 ``rho_schedule[tick]``，**不再**由事件激发。
        这是 H2（方差效应）的对照：把 Hawkes 跑出来的 ρ 序列**随机置换**
        后喂回来，边际分布（因而均值与方差）完全一样，
        但"事件→事件"的自激结构被抹掉。于是
        「方差变了」与「时间结构变了」这两件事第一次可以分开测。
        ⚠️ 它**故意**打破闭环，所以实测平均活跃度可能与 Hawkes 臂不同——
        必须把两臂的 realized events/tick 一起报出来，不能只看 ρ 的分布。

    ``thin_scope``  把稀疏化**只作用在一侧**（EA.4 用）。
        ``"both"``（默认）不区分；``"taker"`` 只稀疏化非做市商主体；
        ``"maker"`` 只稀疏化做市商。用来拆解"聚集性作用在需求侧还是供给侧"。
        两侧各自仍按"每主体独立优先级"排序，所以"取前 k 个"在**每一侧内部**
        仍然是均匀随机子集（见模块文档 ②）。

    ``record_depth``  逐 tick 记一份盘口总深度。
        口径与阶段3 冲击实验里的 ``depth_before`` **完全一致**
        （``book.total_quantity("buy") + total_quantity("sell")``）——
        任务书特别提醒过：如果深度快照与实际测量冲击时用的时点/口径不是同一批，
        测出来的深度差异可能和真正影响 k 的那个差异不是一回事。
    """

    def __init__(self, *args, hawkes_config: HawkesConfig | None = None,
                 hawkes_off: bool = False,
                 rho_schedule: "np.ndarray | None" = None,
                 thin_scope: str = "both",
                 record_depth: bool = False, **kw) -> None:
        # ⚠️ 必须在 super().__init__ 之前：基类的 __init__ 就会调
        # `_limit_activity`（经由 step 之外没有，但 _build_agents 会调
        # decorate_agent；保险起见先建好）。
        self.hawkes_cfg = hawkes_config or HawkesConfig()
        self.hawkes_off = bool(hawkes_off)
        self.hawkes = self.hawkes_cfg.to_process()
        self.rho_schedule = None if rho_schedule is None else np.asarray(
            rho_schedule, dtype=float)
        if thin_scope not in ("both", "taker", "maker"):
            raise ValueError(f"thin_scope 必须是 both/taker/maker，收到 {thin_scope!r}")
        self.thin_scope = thin_scope
        self.record_depth = bool(record_depth)
        #: 每 tick 的 (强度, ρ, 活跃主体数, 事件数) 记录，供事后分析
        self.hawkes_log: list[dict] = []
        super().__init__(*args, **kw)

    # ------------------------------------------------------------------
    def _current_rho(self) -> float:
        if self.rho_schedule is not None:
            t = min(int(self.tick), self.rho_schedule.size - 1)
            return float(self.rho_schedule[t])
        return float(self.hawkes.normalized_intensity(
            self.hawkes_cfg.base_lambda, clamp=self.hawkes_cfg.clamp_rho))

    def _limit_activity(self, order):
        if self.hawkes_off:
            return order
        rho = self._current_rho()
        if self.thin_scope == "both":
            k = rho_to_count(rho, len(order), clamp=self.hawkes_cfg.clamp_rho)
            self._pending_rho = rho
            self._pending_k = k
            return order[:k]
        # 单侧：把列表按"是否做市商"切开，只对选中的那一侧做 ρ 稀疏化。
        makers = [a for a in order if getattr(a, "KIND", "") == "market_maker"]
        takers = [a for a in order if getattr(a, "KIND", "") != "market_maker"]
        if self.thin_scope == "maker":
            keep = rho_to_count(rho, len(makers), clamp=self.hawkes_cfg.clamp_rho)
            out = takers + makers[:keep]
        else:                                   # "taker"
            keep = rho_to_count(rho, len(takers), clamp=self.hawkes_cfg.clamp_rho)
            out = takers[:keep] + makers
        self._pending_rho = rho
        self._pending_k = len(out)
        return out

    def _book_depth_total(self) -> float:
        """与阶段3 冲击实验的 ``depth_before`` 同口径。"""
        return float(self.book.total_quantity("buy")
                     + self.book.total_quantity("sell"))

    def step(self) -> None:
        if self.hawkes_off:
            super().step()
            return
        # 深度快照取在**本 tick 主体动作之前**——这正是阶段3 里
        # `depth_before` 被测量的时点，两者必须对齐才有可比性。
        depth_pre = self._book_depth_total() if self.record_depth else float("nan")
        tick_now = int(self.tick)
        n_before = len(self.log.trades)
        super().step()
        # 本 tick 的**实际成交笔数**当作激发事件（见模块文档 ③）
        events = float(len(self.log.trades) - n_before)
        if self.rho_schedule is None:
            lam = self.hawkes.advance(1.0, events)
        else:
            # 代理对照里闭环是**故意**断开的：不推进、只用外生序列。
            lam = float(self._current_rho() * self.hawkes_cfg.base_lambda)
        self.hawkes_log.append({
            # ⚠️ 记 tick_now 而不是 self.tick：`Market.step` 末尾会 tick += 1，
            #    所以在这里读 self.tick 会得到**下一格**的编号，
            #    让整条日志与 ρ/成交数错开一格（原来就是这么错的）。
            "tick": tick_now,
            "lambda": float(lam),
            "rho": float(self._pending_rho),
            "n_active": int(self._pending_k),
            "events": events,
            "depth": float(depth_pre),
        })

    # ------------------------------------------------------------------
    def hawkes_stats(self) -> dict:
        """强度/活跃度的汇总，用于核对"均值不变"这条前提。"""
        if not self.hawkes_log:
            return {"n": 0}
        import numpy as np

        lam = np.array([d["lambda"] for d in self.hawkes_log], dtype=float)
        rho = np.array([d["rho"] for d in self.hawkes_log], dtype=float)
        ev = np.array([d["events"] for d in self.hawkes_log], dtype=float)
        act = np.array([d["n_active"] for d in self.hawkes_log], dtype=float)
        return {
            "n": int(lam.size),
            "lambda_mean": float(lam.mean()),
            "lambda_sd": float(lam.std(ddof=1)) if lam.size > 1 else float("nan"),
            "rho_mean": float(rho.mean()),
            "rho_sd": float(rho.std(ddof=1)) if rho.size > 1 else float("nan"),
            "events_mean": float(ev.mean()),
            "events_sd": float(ev.std(ddof=1)) if ev.size > 1 else float("nan"),
            "n_active_mean": float(act.mean()),
            "branching": float(self.hawkes.branching_ratio),
            "stationary_mean_expected": float(self.hawkes_cfg.base_lambda),
        }
