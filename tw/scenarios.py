"""场景库：一个策略"在常态下能赚"远远不够，要看它在各种市场状态下还站不站得住。

为什么必须有这个模块
--------------------
单场景评测是策略研究里最经典的自我欺骗：调参直到在默认参数下好看，
然后对外宣布"策略有效"。真实市场不会一直待在默认参数里。
本模块把"市场状态"显式化，让稳健性检验变成**默认动作**而不是加分项。

五个场景的设计依据
------------------
每个场景都对应本项目实测过的某种机制，不是拍脑袋调的：

``normal``     蓝图标定的基准配置（30/40/30 混合池）。
               作为"主场景"，所有策略都应在它上面先站住。
``calm``       零智能活跃度降到 1/3、报价幅度收窄。
               订单流稀、价差窄——做市商容易赚价差但难拿量；
               执行类策略滑点小但对手盘少。**用来测"红利消失后的下限"**。
``stressed``   零智能报价幅度放宽、图表派占比提高到 45%。
               实测图表派正反馈会把波动率推高数倍（阶段2 消融：
               图表派单独存在时 σ 达 398bp）。**用来测回撤与库存失控**。
``liquidation``在承压基础上，让持仓最重的一批主体被强制平仓。
               实测瞬时清算会**饱和**（超过盘口深度的部分根本成交不了），
               价格被推向单边。**用来测"别人爆仓时你会怎样"**。
``thin``       主体数减到 1/4。
               实测 σ 强烈依赖主体数 N（纯零智能池 N=100→500 时 σ 从 37→86bp）。
               主体少 = 订单流稀 = 深度薄。**用来测流动性依赖度**。

用法
----
    from tw.scenarios import get_scenario
    sc = get_scenario("stressed")
    market = sc.build(seed=1234)          # 建好市场（含背景主体）
    market.add_agent(my_strategy)         # 注入待测策略
    market.run()
    market.apply_shock()                  # 场景带冲击时，在预定 tick 施加
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import Population, SimConfig
from .market import Market

#: 项目实测标定出的默认配置（见 docs/标定说明.md）。
BASE_MIX: dict[str, float] = {
    "zero_intel": 0.30,
    "fundamentalist": 0.40,
    "chartist": 0.30,
}

#: 做市商参数（阶段3 标定值：价差 0.53×、深度 1.60×、σ 0.55×）
MM_KW: dict = dict(
    mm_base_spread=0.00005,
    mm_vol_sensitivity=0.05,
    mm_quote_qty=2.0,
    mm_inventory_target=10.0,
    mm_skew_strength=0.0004,
)


@dataclass(slots=True)
class Scenario:
    """一个可复现的市场状态。"""

    name: str
    #: 人类可读的说明：**这个场景想测什么**。
    #: 不写清楚的话，跑出来的数字没人知道该怎么解读。
    intent: str
    n_agents: int = 300
    warmup: int = 4_000
    n_ticks: int = 12_000
    mix: dict[str, float] = field(default_factory=lambda: dict(BASE_MIX))
    overrides: dict = field(default_factory=dict)
    #: 是否在 midpoint 施加一次强制平仓（配合 ``shock_frac``）
    shock_frac: float | None = None
    #: 相对 warmup 结束的偏移，在哪个 tick 触发冲击
    shock_at: int = 1_000

    # --- 构造 -----------------------------------------------------------
    def config(self, seed: int) -> SimConfig:
        pop = Population.from_shares(self.n_agents, self.mix)
        kw = dict(MM_KW)
        kw.update(self.overrides)
        return SimConfig(
            seed=seed,
            n_ticks=self.warmup + self.n_ticks + 50,
            population=pop,
            **kw,
        )

    def build(self, seed: int) -> Market:
        """建市场并跑完预热——策略应当在**稳态**市场里开始工作。

        预热阶段一起跑掉是刻意的：如果让策略从 tick 0 就开始决策，
        它会在一个"盘口还没建立起来"的市场里乱下单，
        测出来的表现反映的是建场过程，不是策略本身。
        """
        m = Market(self.config(seed))
        m.run(self.warmup)
        return m

    # --- 冲击 -----------------------------------------------------------
    def apply_shock(self, market: Market) -> dict | None:
        """施加场景自带的冲击（如果有）。在 ``shock_at`` tick 之后调用。"""
        if self.shock_frac is None:
            return None
        total_long = float(sum(a.inventory for a in market.agents))
        target = total_long * self.shock_frac
        ranked = sorted(market.agents, key=lambda a: a.inventory, reverse=True)
        victims, acc = [], 0.0
        for a in ranked:
            if a.inventory <= 1e-6:
                break
            victims.append(a)
            acc += a.inventory
            if acc >= target:
                break
        frac = min(1.0, target / acc) if acc > 0 else 0.0
        return market.force_liquidate(victims, fraction=frac, reason=self.name)

    def describe(self) -> dict:
        return {
            "name": self.name,
            "intent": self.intent,
            "n_agents": self.n_agents,
            "n_ticks": self.n_ticks,
            "warmup": self.warmup,
            "mix": self.mix,
            "overrides": self.overrides,
            "shock_frac": self.shock_frac,
        }


SCENARIOS: dict[str, Scenario] = {
    "normal": Scenario(
        name="normal",
        intent="基准场景：蓝图标定配置。所有策略必须先在这里站住，再看其他场景。",
    ),
    "calm": Scenario(
        name="calm",
        intent="流动性稀薄但波动温和：测策略在「红利消失」后的下限，"
               "看它是不是只在热闹市场里有效。",
        overrides={
            "zi_p_active": 0.05,          # 订单流密度降到 1/3
            "zi_offset_range": (0.0002, 0.0008),
            "fu_p_active": 0.10,
            "ch_p_active": 0.10,
        },
    ),
    "stressed": Scenario(
        name="stressed",
        intent="高波动压力：图表派正反馈主导（阶段2 实测单独存在时 σ 达 398bp）。"
               "测回撤、库存失控、以及策略会不会在趋势里越亏越加仓。",
        mix={"zero_intel": 0.25, "fundamentalist": 0.30, "chartist": 0.45},
        overrides={"zi_offset_range": (0.0002, 0.0028)},
    ),
    "liquidation": Scenario(
        name="liquidation",
        intent="清算冲击：承压之上再加一次强制平仓。实测瞬时清算会饱和——"
               "超过盘口深度的部分根本成交不了，价格被推向单边。"
               "测策略在别人爆仓时是被扫到还是能接住。",
        mix={"zero_intel": 0.25, "fundamentalist": 0.35, "chartist": 0.40},
        overrides={"zi_offset_range": (0.0002, 0.0022)},
        shock_frac=0.01,
        shock_at=800,
    ),
    "thin": Scenario(
        name="thin",
        intent="主体数减到 1/4：实测 σ 强烈依赖主体数（订单流密度），"
               "主体少 = 深度薄。测策略的流动性依赖度——"
               "它在厚市场里的表现有多少来自「盘口托住了它」。",
        n_agents=75,
    ),
}


def warmup_of(name: str) -> int:
    """按名字取该场景的预热长度（不构造实例）。给"估算批量规模"之类的校验用。"""
    return get_scenario(name).warmup


def get_scenario(name: str) -> Scenario:
    try:
        return SCENARIOS[name]
    except KeyError as exc:
        raise KeyError(
            f"未知场景 {name!r}；可用：{sorted(SCENARIOS)}"
        ) from exc


def list_scenarios() -> list[tuple[str, str]]:
    """返回 ``[(名字, 这个场景测什么)]``，供 CLI 打印。"""
    return [(s.name, s.intent) for s in SCENARIOS.values()]
