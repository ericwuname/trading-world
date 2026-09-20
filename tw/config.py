"""模拟参数（施工蓝图 §4）。

价格刻度取真实 BTC 量级（60000 美元、tick 0.01），
好处是模拟出来的价格、成交量可以直接和真实 BTC 小时线放在同一张表里比，
不必再做量纲换算——换算一次就多一个出错机会。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Population:
    """主体构成（蓝图 §4 阶段2 的 30/40/30 配比就落在这里）。"""

    zero_intel: int = 200
    fundamentalist: int = 0
    chartist: int = 0
    market_maker: int = 0

    @property
    def total(self) -> int:
        return self.zero_intel + self.fundamentalist + self.chartist + self.market_maker

    def shares(self) -> dict[str, float]:
        n = self.total or 1
        return {
            "zero_intel": self.zero_intel / n,
            "fundamentalist": self.fundamentalist / n,
            "chartist": self.chartist / n,
            "market_maker": self.market_maker / n,
        }

    @classmethod
    def from_shares(cls, n_agents: int, shares: dict[str, float]) -> "Population":
        """按比例分配 n_agents 个主体，保证四舍五入后总数不跑偏。"""
        keys = ["zero_intel", "fundamentalist", "chartist", "market_maker"]
        total_w = sum(max(0.0, shares.get(k, 0.0)) for k in keys) or 1.0
        raw = {k: max(0.0, shares.get(k, 0.0)) / total_w * n_agents for k in keys}
        counts = {k: int(raw[k]) for k in keys}
        # 余数给小数部分最大的那几类，避免出现 0 主体的空类
        rest = n_agents - sum(counts.values())
        for k in sorted(keys, key=lambda k: raw[k] - int(raw[k]), reverse=True)[:rest]:
            counts[k] += 1
        return cls(**counts)


@dataclass(slots=True)
class SimConfig:
    """一次模拟的全部参数。默认值 = 蓝图阶段1（纯零智能基线）。"""

    # --- 全局 ---
    seed: int = 20260917
    n_ticks: int = 10_000
    tick_size: float = 0.01
    initial_price: float = 60_000.0
    population: Population = field(default_factory=Population)

    # --- 主体禀赋 ---
    inv_endowment: tuple[float, float] = (0.0, 20.0)   # 初始持仓 ~ U(lo, hi)
    cash_endowment: tuple[float, float] = (2.0, 18.0)  # 初始现金（单位：当前价格倍数）
    max_open_orders: int = 20
    order_ttl: int = 500          # 挂单存活上限（tick），超时自动撤
    housekeeping_interval: int = 10

    # --- A. 零智能交易者（蓝图 §3.5-A）---
    zi_p_active: float = 0.15          # 每个 tick 下单概率
    zi_offset_range: tuple[float, float] = (0.0002, 0.0018)  # 报价偏离中间价的比例区间
    #   ↑ 蓝图建议 0.1%~2%，但那个幅度下每 tick σ 高达 543bp（真实 BTC 小时线 48bp）。
    #     报价幅度直接决定价格扩散速度与盘口宽度，是**必须标定**的，不能照抄。
    #     取值依据见 docs/标定说明.md：令 1 tick 约等于 1 小时，把 σ 标到真实量级。
    zi_qty_mean: float = 1.0           # 下单量 ~ 指数分布
    zi_p_cancel: float = 0.01          # 额外随机撤单概率（配合 order_ttl 共同决定挂单寿命）
    zi_p_buy: float = 0.5              # 下买单概率（0.5 = 无方向偏好；二期阶段5 用它制造净多头）

    # --- B. 基本面派（蓝图 §3.5-B）---
    fu_v_vol: float = 4e-4             # 基本面价值对数随机游走的每 tick 波动
    fu_v_pull: float = 2e-5            # 向长期锚点回归的强度（防 V 无限漂移）
    fu_v_anchor: float | None = None   # None => 用 initial_price
    fu_deadband: float = 0.002         # 无套利死区：|m| 小于此值不下单
    fu_ref_mispricing: float = 0.02    # 力度归一化参考值（|m| 达到它就满力）
    fu_aggressiveness: float = 0.004   # 报价穿越中间价的最大比例
    fu_qty_mean: float = 1.5
    fu_p_active: float = 0.25

    # --- C. 图表派（蓝图 §3.5-C）---
    ch_lookback: int = 50              # 动量观察窗口 N
    ch_deadband: float = 0.002         # 动量低于此值不下单
    ch_ref_momentum: float = 0.02      # 力度归一化参考值
    ch_aggressiveness: float = 0.004
    ch_qty_mean: float = 1.5
    ch_p_active: float = 0.25

    # --- D. 做市商（蓝图 §3.5-D）---
    mm_base_spread: float = 0.0006     # 基础半价差（相对中间价）
    mm_vol_sensitivity: float = 4.0    # 波动率对价差的放大系数
    mm_vol_window: int = 100
    mm_inventory_target: float = 10.0  # 目标库存
    mm_skew_strength: float = 0.0004   # 库存偏离 1 单位时的报价偏移比例
    mm_quote_qty: float = 1.0

    # --- 记录 ---
    record_snapshot_every: int = 100   # 每多少 tick 存一次订单簿快照
    snapshot_levels: int = 20

    def validate(self) -> None:
        assert self.n_ticks > 0
        assert self.tick_size > 0
        assert self.initial_price > 0
        assert self.population.total > 0, "至少要有一个主体"
        assert 0.0 <= self.zi_p_active <= 1.0
        assert self.zi_offset_range[0] > 0 and self.zi_offset_range[1] >= self.zi_offset_range[0]
        assert self.ch_lookback >= 2
        assert self.mm_quote_qty > 0

    @property
    def v_anchor(self) -> float:
        return self.fu_v_anchor if self.fu_v_anchor is not None else self.initial_price

    def label(self) -> str:
        s = self.population.shares()
        parts = [
            f"{k}={v:.0%}" for k, v in s.items() if v > 0
        ]
        return f"N={self.population.total} " + " ".join(parts)
