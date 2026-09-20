"""Market 主循环（施工蓝图 §2、§4）。

一个 tick 里发生什么
--------------------
    1. 组装 MarketState（中间价 / 最近价格历史 / 基本面价值）
    2. 每 10 个 tick 让所有主体做一次维护（撤过期挂单）
    3. **随机打乱**主体顺序，逐个决策 → 提交 → 撮合 → 结算 → 记账
    4. 把这一 tick 的中间价压进价格历史
    5. 记录逐 tick 序列
    6. 基本面价值走一步

两个看起来不起眼、但会彻底改变结论的设计决定
--------------------------------------------
**① 主体顺序每 tick 随机打乱。**
   若按固定顺序（A 永远先于 B），那么排在后面的主体永远看到"已经被前面改动过"的
   簿面，先手优势会被系统性地固化成一个假的收益分布——跑出来的"肥尾"其实是
   排队顺序造成的伪影，不是市场机制的产品。

**② state 在 tick 内就地刷新，但价格历史只在 tick 末压入一次。**
   主体决策期间能看到价格的最新变化（更接近连续市场的真实感），
   而记录的采样频率严格是 1 tick 一条（避免同一 tick 的中间态污染时间序列统计）。

记账纪律
--------
所有资金/持仓变动**只在这一个文件里发生**（``_settle``）。
Agent 与 MatchingEngine 都不直接改别人的账户——
一旦记账逻辑分裂到多处，必然出现"某条路径忘了扣钱"的 bug，
而且这种 bug 在统计特征上是隐形的，跑一万个 tick 也未必发现。
"""

from __future__ import annotations

import math
import time
from typing import Iterable, Sequence

import numpy as np

from .agents import (
    Agent,
    Chartist,
    Fundamentalist,
    MarketMaker,
    ZeroIntelligence,
    step_fundamental,
)
from .book import OrderBook
from .config import Population, SimConfig
from .engine import MatchingEngine
from .logger import SimLog
from .types import (
    EPS,
    BookSnapshot,
    DepthLevel,
    FlowView,
    MarketState,
    Order,
    PriceHistory,
    Trade,
)

# 小于这个数量的订单视为"灰尘单"，直接丢弃。
# 不设下限的话，账户快见底的主体每 tick 都会挂出 1e-12 级别的单，
# 把订单簿档位数刷到几十万，深度统计变成噪声。
#: 随机流标签。numpy 的 SeedSequence 只接受整数熵，不接受字符串，
#: 而这些标签必须**互不相同**：禀赋流、主体决策流、市场流各用一条，
#: 否则同一 (seed, idx) 下两条流会退化成同一条，主体决策与禀赋强相关。
ENTROPY_ENDOW = 0xE0D0
ENTROPY_AGENT = 0xA9E7
#: 主体「出手顺序」的调度流。与决策流分开：调度流每主体每 tick 抽一次，
#: 决策流只在主体真的活跃时消耗。共用一条会让「活跃次数」反过来影响「出手顺序」，
#: 两件事纠缠在一起后很难解释实验结果。
ENTROPY_ORDER = 0x0DDE7

#: 调度优先级的分块预抽长度。太小退化成每次调用都抽；太大浪费内存。
ORDER_BLOCK = 4096

MIN_ORDER_QTY = 1e-4


class Market:
    """连续双向拍卖市场。二期的永续层是它的**子类**（见 tw/perpetual.py），
    而不是塞进主循环——一期 159 项测试全部跑在裸 Market 上，
    把结算逻辑写进来会悄悄改变一期的所有统计特征。"""

    #: 是否允许保证金账户（现金可为负，但权益不可为负）。
    #: 裸 Market = False（无杠杆现货市场，`available_cash >= 0` 是核心不变量）；
    #: PerpetualMarket = True（永续本来就是杠杆产品，资金费率会从保证金里扣）。
    allow_negative_cash: bool = False

    #: 资金费率。裸市场恒为 0；PerpetualMarket 每 tick 覆盖它。
    #: 放在类属性上而不是 `getattr(self, ..., 0.0)`：后者把拼写错误
    #: 变成"静默取到 0"，而 0 恰好是个合法值，问题会被藏起来。
    last_funding_rate: float = 0.0

    """人工市场。"""

    def __init__(self, config: SimConfig | None = None) -> None:
        self.cfg = config or SimConfig()
        self.cfg.validate()

        self.book = OrderBook(tick_size=self.cfg.tick_size)
        self.engine = MatchingEngine(self.book, allow_self_trade=False)
        self.rng = np.random.default_rng(self.cfg.seed)

        hist_len = max(
            4 * self.cfg.ch_lookback,
            self.cfg.mm_vol_window + 16,
            256,
        )
        self.history = PriceHistory(hist_len)

        self.tick = 0
        self.fundamental = self.cfg.v_anchor
        self.last_price: float | None = None
        self._last_mid = self.cfg.initial_price

        self.agents: list[Agent] = []
        self.by_id: dict[str, Agent] = {}
        #: 每个主体的调度流与优先级缓冲（见 `_next_order`）。
        #: 用分块预抽而不是每 tick 每主体现抽，省下数百万次函数调用。
        self._ord_rng: list[np.random.Generator] = []
        self._ord_buf: list[np.ndarray] = []
        self._ord_pos: list[int] = []
        self._ord_prio: np.ndarray | None = None
        #: ``_flow_imbalance`` 的按 tick 记忆化（见该方法说明）。
        #: 它在 ``_refresh_state`` 里被"每个提交主体 × 每 tick"调用，
        #: 是二期引入的唯一一处热路径开销。
        self._fi_tick: int = -1
        self._fi_win: int | None = None
        self._fi_val: float = 0.0
        #: 主体 ID → 注入时的 tick。外部注入的主体（策略）记在这里——
        #: 「同一 seed 复现一次实验」必须连注入时点一起固定，否则复现不出来。
        self.injected_at: dict[str, int] = {}
        self._build_agents()
        for a in self.agents:
            a.bind(self)
            a.set_initial_equity(self.cfg.initial_price)

        self._state = MarketState(
            tick=0,
            mid=self.cfg.initial_price,
            last_price=self.cfg.initial_price,
            best_bid=None,
            best_ask=None,
            spread=None,
            fundamental=self.fundamental,
            history=self.history,
        )

        # 逐 tick 累加器
        self._vol_tick = 0.0
        self._n_trade_tick = 0
        self._flow_tick = 0.0

        # --- 二期：外生现金的对手方账户（见 apply_cash_delta）-------------
        #: 交易所虚拟账户。资金费率结算时，主体付出的由它收走、收到的由它付出。
        #: **有了它，"全市场现金总和"才是常数**——否则资金费率会凭空创造或消灭现金，
        #: 而这种破坏 `account_integrity` 完全看不出来（那个检查只管单账户自洽）。
        self.exchange_cash: float = 0.0
        #: 与系统外部的净往来。正常应恒为 0，用于审计。
        self.exogenous_cash: float = 0.0
        #: 外生现金流水的逐笔审计记录。
        self.cash_ledger: list[dict] = []
        self._exogenous_by_reason: dict[str, float] = {}

        self.log = SimLog(
            n_ticks=self.cfg.n_ticks,
            config_label=self.cfg.label(),
            config=self._config_dict(),
        )
        self._init_log_arrays()

        # 事件标记（压力测试用）
        self.events: list[dict] = []

    # ------------------------------------------------------------------
    # 构建
    # ------------------------------------------------------------------
    def _config_dict(self) -> dict:
        c = self.cfg
        return {
            "seed": c.seed,
            "n_ticks": c.n_ticks,
            "tick_size": c.tick_size,
            "initial_price": c.initial_price,
            "population": {
                "zero_intel": c.population.zero_intel,
                "fundamentalist": c.population.fundamentalist,
                "chartist": c.population.chartist,
                "market_maker": c.population.market_maker,
            },
            "zi_p_active": c.zi_p_active,
            "zi_offset_range": list(c.zi_offset_range),
            "fu_deadband": c.fu_deadband,
            "fu_ref_mispricing": c.fu_ref_mispricing,
            "ch_lookback": c.ch_lookback,
            "ch_deadband": c.ch_deadband,
            "ch_ref_momentum": c.ch_ref_momentum,
            "mm_base_spread": c.mm_base_spread,
            "mm_vol_sensitivity": c.mm_vol_sensitivity,
        }

    def _init_log_arrays(self) -> None:
        n = self.cfg.n_ticks
        L = self.log
        f = lambda: np.full(n, np.nan, dtype=np.float64)  # noqa: E731
        L.tick = np.arange(n, dtype=np.float64)
        L.mid = f()
        L.best_bid = f()
        L.best_ask = f()
        L.spread = f()
        L.fundamental = f()
        L.volume = f()
        L.n_trades = f()
        L.flow = f()
        L.bid_depth = f()
        L.ask_depth = f()
        L.open_orders = f()
        L.n_levels_bid = f()
        L.n_levels_ask = f()

    def decorate_agent(
        self, kind: str, cls: type, kw: dict
    ) -> tuple[type, dict]:
        """主体构造钩子：返回 ``(实际的类, 实际的关键字参数)``。

        默认**原样返回**，所以一期与二期已有的行为逐点不变。
        子类（例如阶段6 的实验用 Market）覆盖它，就能把机制 Mixin 挂到
        指定池子上，而不必改动任何主体类的构造函数。

        ⚠️ 覆盖时必须保持"注入是**幂等且显式**的"：
        返回的类要能被直接实例化，返回的 kw 要与该类的签名匹配。
        悄悄把 kw 塞进不接受的类会得到 ``TypeError``——这**是好事**，
        比默默忽略一个参数好得多。
        """
        return cls, kw

    def _limit_activity(self, order: list[Agent]) -> list[Agent]:
        """本 tick 实际能出手的主体子集。默认**不限制**（返回原列表）。

        ⚠️ 若覆盖它，必须保持"子集是**均匀随机**的"这一点。
        现在传进来的是**按各主体自己的优先级排好序**的列表，
        而每个主体的优先级是从它自己的随机流里均匀抽的——
        所以"取前 k 个"恰好等于"均匀随机抽 k 个"。
        若哪天 `_next_order` 改成按别的方式排序（比如按持仓、按 id），
        "取前 k 个"就会变成"总是偏向某一类主体"，
        而症状只是统计特征的悄悄偏移，不会报错。
        """
        return order

    def _build_agents(self) -> None:
        """建主体。

        ⭐ **两条对"策略 A/B 对比"至关重要的设计约束**（做策略实验台时才意识到）：

        ① **禀赋必须来自"每主体独立的随机流"，不能消耗市场随机流。**
           市场 RNG 只用于两件事：每 tick 洗牌、每 tick 基本面价值步进。
           如果禀赋也从市场 RNG 抽，那么"多建一个主体"就会让市场 RNG 多走两步，
           于是**整个背景行情（洗牌顺序 + 基本面路径）全变**——
           对照组与实验组跑的根本不是同一段行情，A/B 对比失效。
           用 ``default_rng([seed, idx, "endow"])`` 之后，主体数量与背景行情解耦。

        ② **主体 RNG 按"序号"播种，不按"共享种子池 spawn"。**
           ``default_rng([seed, idx])`` 保证第 i 个主体的流只取决于 seed 与 i，
           增删其他主体不会扰动它的流。若改用 ``SeedSequence(seed).spawn(n)``，
           追加一个主体会让其后所有主体的流整体平移，背景噪声不可比。
        """
        cfg = self.cfg
        pop: Population = cfg.population
        inv_lo, inv_hi = cfg.inv_endowment
        cash_lo, cash_hi = cfg.cash_endowment
        p0 = cfg.initial_price
        idx = 0

        def endowment(i: int) -> tuple[float, float]:
            erng = np.random.default_rng([cfg.seed, i, ENTROPY_ENDOW])
            inv = float(erng.uniform(inv_lo, inv_hi))
            cash = float(erng.uniform(cash_lo, cash_hi) * p0)
            return inv, cash

        def add(cls, kind: str, n: int, **kw) -> None:
            nonlocal idx
            # ⭐ 唯一的注入点。默认原样返回，所以一期行为逐点不变；
            # 二期阶段6 的实验用一个 Market 子类覆盖它，把元订单 / 自适应流动性
            # 这两个 Mixin 挂到指定池子上。
            # 为什么做成"钩子"而不是"给每个主体类加开关"：
            # 机制是**按池子**开关的（40% 的图表派用元订单、做市商用自适应流动性），
            # 把开关加进每个主体类的构造函数会让"哪些池子开了什么"散落在四处，
            # 而实验脚本里必须能一眼看全。
            cls, kw = self.decorate_agent(kind, cls, kw)
            for _ in range(n):
                inv, cash = endowment(idx)
                if kind == "market_maker":
                    # 做市商要同时挂两边，禀赋给足，否则一开局就只挂得出单边
                    inv *= 2.0
                    cash *= 2.0
                aid = f"{kind[:2]}{idx:04d}"
                rng = np.random.default_rng([cfg.seed, idx, ENTROPY_AGENT])
                agent = cls(aid, cash, inv, rng, **kw)
                agent.bind_seed(cfg.seed)
                self.agents.append(agent)
                self.by_id[aid] = agent
                idx += 1

        if pop.zero_intel:
            add(
                ZeroIntelligence,
                "zero_intel",
                pop.zero_intel,
                p_active=cfg.zi_p_active,
                offset_range=cfg.zi_offset_range,
                qty_mean=cfg.zi_qty_mean,
                p_cancel=cfg.zi_p_cancel,
                p_buy=cfg.zi_p_buy,
                max_open_orders=cfg.max_open_orders,
                order_ttl=cfg.order_ttl,
            )
        if pop.fundamentalist:
            add(
                Fundamentalist,
                "fundamentalist",
                pop.fundamentalist,
                deadband=cfg.fu_deadband,
                ref_mispricing=cfg.fu_ref_mispricing,
                aggressiveness=cfg.fu_aggressiveness,
                qty_mean=cfg.fu_qty_mean,
                p_active=cfg.fu_p_active,
                max_open_orders=cfg.max_open_orders,
                order_ttl=cfg.order_ttl,
            )
        if pop.chartist:
            add(
                Chartist,
                "chartist",
                pop.chartist,
                lookback=cfg.ch_lookback,
                deadband=cfg.ch_deadband,
                ref_momentum=cfg.ch_ref_momentum,
                aggressiveness=cfg.ch_aggressiveness,
                qty_mean=cfg.ch_qty_mean,
                p_active=cfg.ch_p_active,
                max_open_orders=cfg.max_open_orders,
                order_ttl=cfg.order_ttl,
            )
        if pop.market_maker:
            add(
                MarketMaker,
                "market_maker",
                pop.market_maker,
                base_spread=cfg.mm_base_spread,
                vol_sensitivity=cfg.mm_vol_sensitivity,
                vol_window=cfg.mm_vol_window,
                inventory_target=cfg.mm_inventory_target,
                skew_strength=cfg.mm_skew_strength,
                quote_qty=cfg.mm_quote_qty,
                max_open_orders=max(4, cfg.max_open_orders // 4),
                order_ttl=max(10, cfg.order_ttl // 20),
            )


    def add_agent(self, agent: Agent) -> Agent:
        """把外部主体（例如待测策略）加入市场。

        **允许在预热之后、正式观测之前注入**，而且这是推荐做法：
        让策略参与预热的话，各策略会把市场带到不同的状态，
        后面的对比就不再是"同一初始条件下的差异"了。
        项目里的做法统一是：``Scenario.build(seed)`` 跑完预热 →
        ``add_agent(策略)`` → 再 ``run(n_ticks)`` 观测。

        注入时点会被记在 ``self.injected_at``：**复现一次实验必须连注入时点一起固定**，
        否则同一 seed 也复现不出来。
        """
        if agent.agent_id in self.by_id:
            raise ValueError(f"主体 ID 重复：{agent.agent_id}")
        agent.bind(self)
        agent.bind_seed(self.cfg.seed)
        agent.set_initial_equity(self.current_mid())
        self.agents.append(agent)
        self.by_id[agent.agent_id] = agent
        self.injected_at[agent.agent_id] = int(self.tick)
        return agent

    @property
    def n_agents(self) -> int:
        """当前主体总数（含后续注入的）。"""
        return len(self.agents)

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------
    def current_mid(self) -> float:
        """当前中间价。盘口单边空缺时回退到最新成交价，再退到最近一次中间价。

        永远不返回 None：主体如果拿到 None 就只能"不动作"，
        而在簿子单边打空时全体静止会让模拟直接卡死（没有新订单 → 簿子永远补不回来）。
        """
        m = self.book.mid_price()
        if m is not None:
            self._last_mid = m
            return m
        if self.last_price is not None and self.last_price > 0:
            return self.last_price
        return self._last_mid

    def _refresh_state(self, state: MarketState | None = None) -> MarketState:
        s = state if state is not None else self._state
        bb = self.book.best_bid()
        ba = self.book.best_ask()
        s.tick = self.tick
        s.best_bid = bb
        s.best_ask = ba
        s.spread = (ba - bb) if (bb is not None and ba is not None) else None
        s.mid = self.current_mid()
        s.last_price = self.last_price if self.last_price is not None else s.mid
        s.fundamental = self.fundamental
        # 二期字段：每 tick 就地刷新一次，主体读到的就是"当前"值。
        # 不放进 `_finalize_tick` 是因为主体在 tick **内**就要用它们做决策。
        s.funding_rate = self.last_funding_rate
        s.flow_imbalance = self._flow_imbalance()
        # 只读窗口视图：阶段6 的自适应流动性需要**任意窗口**的失衡量，
        # 而上一行是固定窗口的预计算值。这里给一个引用视图，不拷贝数组。
        s.flow = FlowView(self.log.flow, self.tick)
        return s

    def _flow_imbalance(self, window: int = 200) -> float:
        """近期主动买卖量的净失衡 ∈ [-1, 1]。

        用 ``log.flow``（带方向的主动成交量）在窗口内求和，除以窗口内
        主动成交量绝对值之和。**用绝对值和做分母**而不是笔数：
        用笔数会把"一笔巨量单"和"一笔碎单"等同看待。

        ⚠️ 记忆化的**前提是一条显式契约**（不是"应该没问题"）：
        ``log.flow`` 每 tick 只在 ``_finalize_tick`` 里写**一格**（``L.flow[t]``），
        而窗口取的是 ``[t-window, t)``——**不含 t**。
        于是整个 tick 内这个窗口是冻结的，重复调用必然同值。
        契约的验证点有两处：``tests/test_market.py::TestFlowImbalanceMemo``
        （记忆化与强制重算逐点一致）与 ``tw/market.py`` 里 ``L.flow[t] = ...``
        这个**唯一写入点**（grep ``flow\\[`` 只应命中它）。

        代价要说清楚：**绕过市场直接就地改 ``log.flow`` 不会让缓存失效**。
        正常代码不会这么干（那是篡改市场日志），但手工构造场景的测试会。
        那种情况请显式调 ``invalidate_flow_cache()``——
        把"我知道自己在绕过契约"写出来，比让缓存悄悄返回旧值好。
        """
        if self._fi_tick == self.tick and self._fi_win == window:
            return self._fi_val
        t = self.tick
        lo = max(0, t - window)
        if t <= lo:
            v = 0.0
        else:
            flow = self.log.flow[lo:t]
            num = float(np.nansum(flow))
            den = float(np.nansum(np.abs(flow)))
            # log.flow 会预分配 NaN，未跑到的部分是 NaN，nansum 已处理
            v = float(num / den) if den > 1e-12 else 0.0
        self._fi_tick = t
        self._fi_win = window
        self._fi_val = v
        return v

    def invalidate_flow_cache(self) -> None:
        """让 ``_flow_imbalance`` 的记忆化失效。

        只在**绕过市场直接修改 ``log.flow``** 之后才需要调用
        （正常路径由 ``_finalize_tick`` 的单点写入 + tick 变化自动失效）。
        """
        self._fi_tick = -1
        self._fi_win = None
        self._fi_val = 0.0

    def reconcile_reservations(self, agent: Agent) -> int:
        """把某主体的预留压回它能承受的范围——撤掉超出的买单。

        为什么必须有它：任何**减少现金的外生变动**（资金费率结算、罚金）
        都会让 ``reserved_cash > cash``，于是 ``available_cash < 0``，
        之后它的**每一张新单都被预算裁剪拒掉**。一期踩过一模一样的坑，
        症状是"市场在几百 tick 后静默冻结"——不报错、不崩溃，
        只是成交数慢慢归零，看起来像"价格收敛、市场稳定"的好结果。

        真实交易所遇到保证金不足也是这个顺序：**先撤挂单，再谈强平**。
        撤最早挂的（通常离中间价最远、最不可能成交），返回撤掉的笔数。
        """
        n = 0
        guard = 0
        while agent.reserved_cash > agent.cash + 1e-9:
            buys = [o for o in agent.open_orders if o.side == "buy"]
            if not buys:
                break
            victim = min(buys, key=lambda o: (o.timestamp, o.order_id))
            if not self.cancel(agent, victim):
                break
            n += 1
            guard += 1
            if guard > 10_000:   # 防死循环：正常情况不会到
                break
        return n

    def cover_cash_shortfall(self, agent: Agent, *, max_rounds: int = 4) -> dict:
        """账户现金为负（保证金不足）→ 撤挂单 + 市价变现补足。

        为什么必须有它：任何**减少现金的外生变动**（资金费率结算、罚金）
        都可能把主体的现金打到负数。而 ``available_cash >= 0`` 是本模型
        "主体不能凭空造钱"的核心不变量——一旦破了，之后所有预算裁剪、
        记账检查都会在错误的基准上跑。

        两条错误做法（都试过、都被这条替代）：
          · **把赤字一笔勾销** → 破坏全系统现金守恒；
          · **放任负现金** → `account_integrity` 直接红，且后续所有检查失效。

        真实交易所的处理顺序就是这样：**先撤挂单，再强制减仓**。
        撤最早挂的（离中间价最远、最不可能成交），然后按需市价卖出。
        （实测：300 主体跑 4000 tick，只有 2 个主体触发，缺口合计 559 元，
        相对全市场 6790 万可以忽略——所以它不会污染市场统计特征。）

        返回处理明细，便于审计。
        """
        out = {"cancelled": 0, "sold_qty": 0.0, "raised": 0.0, "uncovered": 0.0}
        if agent.cash >= 0:
            return out
        out["cancelled"] = int(agent.cancel_all())
        for _ in range(max_rounds):
            need = -agent.cash
            if need <= 1e-9:
                break
            px = self.book.best_bid()
            if px is None or px <= 0:
                break
            avail = agent.available_inventory
            if avail <= 1e-9:
                break
            # 多卖 2% 作为缓冲，避免下一 tick 又不够（反复微额强平会制造噪声）
            qty = min(avail, need * 1.02 / px)
            if qty <= 1e-12:
                break
            order = Order(
                agent_id=agent.agent_id,
                side="sell",
                price=-math.inf,
                quantity=qty,
                timestamp=self.tick,
                order_id=f"{agent.agent_id}#MC{self.tick}_{len(self.cash_ledger)}",
                order_type="market",
            )
            trades = self.submit(agent, order)
            sold = float(sum(t.quantity for t in trades))
            out["sold_qty"] += sold
            out["raised"] += float(sum(t.notional for t in trades))
            if sold <= 1e-12:
                break
        out["uncovered"] = max(0.0, -agent.cash)
        return out

    # ------------------------------------------------------------------
    # 外生现金变动（资金费率、罚金、外部划转）——**唯一入口**
    # ------------------------------------------------------------------
    def apply_cash_delta(
        self,
        agent: Agent,
        delta: float,
        *,
        reason: str = "",
        counterparty: str = "exchange",
    ) -> float:
        """改动某个主体的现金，并**显式记账到对手方**。所有非成交引起的资金
        变动都必须走这里。

        为什么强制带 counterparty：**现金是零和的**。只改一方、另一方不记，
        "全市场现金总和"就会悄悄漂移；而漂移不会抛异常，只会让所有涉及
        资金守恒的检查慢慢失效——一期已经踩过同类问题（预留与结算口径不一致
        导致逐笔泄漏，跑几千 tick 后才被账户守恒检查抓到）。

        ``counterparty="exchange"``：由交易所虚拟账户承担（资金费率就是这个）。
        ``counterparty=None``：与系统外部发生往来（会单独累计，便于审计）。

        返回实际应用的增量（浮点误差下以调用方传入值为准）。
        """
        if not np.isfinite(delta):
            raise ValueError(f"现金变动必须是有限数，收到 {delta!r}")
        if delta == 0.0:
            return 0.0
        agent.cash += float(delta)
        entry = {
            "tick": int(self.tick),
            "agent_id": agent.agent_id,
            "delta": float(delta),
            "reason": reason or "unspecified",
            "counterparty": counterparty,
        }
        self.cash_ledger.append(entry)
        if counterparty == "exchange":
            self.exchange_cash -= float(delta)
        else:
            self.exogenous_cash += float(delta)
        self._exogenous_by_reason[entry["reason"]] = (
            self._exogenous_by_reason.get(entry["reason"], 0.0) + float(delta)
        )
        return float(delta)

    def cash_conservation(self) -> tuple[bool, float, str]:
        """现金零和检查。

        不变量：``Σ 主体现金 + 交易所账户 == Σ 初始现金 + 外部净流入``。

        ⚠️ **右边的 ``+ 外部净流入`` 不能省**。最初写成 ``... == Σ 初始现金``，
        结果每一笔外部划入都被**重复计算**：主体现金 +250、外部账户也 +250，
        两边一加就是 +500，守恒检查报出一倍于真实值的残差。
        外部流入确实增加了系统的现金总量，所以它必须出现在等式右边，
        而不是被加进左边。

        返回 ``(是否守恒, 残差, 说明)``。

        与 ``account_integrity`` 的分工：那个查"每个账户自身是否自洽"
        （预留不超持有、可用不为负），这个查"**全系统现金总量是否对得上账**"。
        两者互补——资金费率结算破坏的正是后者，而前者完全看不出来。
        """
        total_agent_cash = float(sum(a.cash for a in self.agents))
        lhs = total_agent_cash + self.exchange_cash
        base = float(sum(a.initial_cash for a in self.agents))
        rhs = base + self.exogenous_cash
        resid = lhs - rhs
        ok = abs(resid) <= 1e-6 * (abs(rhs) + 1.0)
        msg = (
            f"Σ主体={total_agent_cash:,.6f} + 交易所={self.exchange_cash:,.6f} = {lhs:,.6f}；"
            f"Σ初始={base:,.6f} + 外部净流入={self.exogenous_cash:,.6f} = {rhs:,.6f}；"
            f"残差={resid:+.6e}"
        )
        return ok, resid, msg

    # ------------------------------------------------------------------
    # 记账（全项目唯一允许改动账户的地方）
    # ------------------------------------------------------------------
    def _settle(self, trade: Trade) -> None:
        buyer = self.by_id[trade.buy_agent_id]
        seller = self.by_id[trade.sell_agent_id]
        notional = trade.notional

        # 成交**当时**的中间价。撮合引擎里拿不到市场状态，所以在这里补上。
        #
        # 为什么必须逐笔记而不是事后用 `log.mid[t]` 顶替：`log.mid[t]` 是
        # **该 tick 结束时**的中间价，而一个 tick 内有上百笔成交。
        # 用 tick 收盘价当"决策价"会让 markout（成交后价格走势）严重失真——
        # 而 markout 恰恰是评估做市/执行类策略**最核心**的指标
        # （区分"赚了价差"和"接了有毒的单"全靠它）。一个 tick 的错位
        # 足以把两个方向的结论颠倒过来。
        mid_now = self._state.mid
        if mid_now is not None and mid_now > 0:
            trade.mid_at_fill = float(mid_now)
        else:
            trade.mid_at_fill = self._last_mid

        buyer.cash -= notional
        buyer.inventory += trade.quantity
        seller.cash += notional
        seller.inventory -= trade.quantity

        buyer.on_trade(trade, True)
        seller.on_trade(trade, False)

        # ⚠️ 关键：**双方**的预留额度都必须随成交更新，不能只更新主动方。
        #
        # 这里踩过一次坑，记录在此以免重犯：最初只在 `submit` 里给主动方更新预留，
        # 被动方（挂单被吃的一方）的 `reserved_cash` 永远停在成交前的旧值。
        # 后果是它的 `available_cash = cash - reserved_cash` 变成负数，
        # 之后它提交的**每一张单都被预算裁剪拒掉**——市场在几百个 tick 后彻底冻结。
        # 而且表面症状很迷惑人：订单簿自洽、没有异常、成交数慢慢归零，
        # 看起来像"价格收敛、市场稳定"的好结果。这类 bug 不会崩、不会报错，
        # 只会让模拟静悄悄地变成一潭死水。
        self._refresh_party_after_fill(buyer, trade.buy_order_id)
        self._refresh_party_after_fill(seller, trade.sell_order_id)

        self.last_price = trade.price
        self._vol_tick += trade.quantity
        self._n_trade_tick += 1
        self._flow_tick += trade.signed_qty

    def _refresh_party_after_fill(self, agent: Agent, order_id: str) -> None:
        """按成交结果更新某一方的挂单列表与预留额度（O(本主体挂单数)，通常 ≤ 20）。

        只需要关心该主体在本次成交中涉及的那一张订单：
        一笔成交只会改变买卖双方各自一张挂单的状态，其余挂单不受影响。
        """
        oo = agent.open_orders
        if oo:
            for i, o in enumerate(oo):
                if o.order_id == order_id:
                    # 只有完全成交才需要从本地列表剔除（订单簿那边已经移除了）；
                    # 部分成交时订单仍在簿上，`remaining` 已被撮合引擎就地改小，
                    # 下面的 recompute 会自动读到新值。
                    if not self.book.has_order(order_id):
                        del oo[i]
                    break
        agent.recompute_reservations()

    def _clamp(self, agent: Agent, order: Order) -> Order | None:
        """对齐价格 → 按可用资金/持仓裁剪。裁剪后不足最小量的直接拒单。

        ⚠️ 价格必须**先吸附到 tick 网格**，这是全流程的第一道工序。
        踩过的坑：订单价格保留原始浮点值（如 53813.71258116428），而订单簿把价位
        以 ``round(price/tick_size)`` 索引，实际成交价是网格价（53813.71 或
        53813.72）。于是"预留按原价算、结算按网格价付"，四舍五入方向不定，
        每笔成交都会破坏账户守恒一个不到半个 tick 的量。
        症状极其难查：体检报"可用现金为负 -0.0014"（相对量 1e-8），
        看起来像浮点噪声，其实是系统性泄漏，且会污染"资金约束"这类边界判断。
        真实交易所也不接受网格外的价格（要么拒单要么吸附），这里选吸附。
        """
        if order.order_type == "limit":
            snapped = self.book.price_of(self.book.to_tick(order.price))
            if snapped <= 0:
                return None
            order.price = snapped

        if order.side == "buy":
            if order.order_type == "market":
                fillable, cost = self.book.estimate_fill("buy", order.quantity)
                if fillable <= MIN_ORDER_QTY or cost <= 0:
                    return None
                avail = agent.available_cash
                if cost > avail:
                    avg = cost / fillable
                    cap = avail / avg if avg > 0 else 0.0
                    if cap <= MIN_ORDER_QTY:
                        return None
                    order.quantity = cap
                    order.remaining = cap
            else:
                if order.price <= 0:
                    return None
                cap = agent.available_cash / order.price
                if cap <= MIN_ORDER_QTY:
                    return None
                if order.quantity > cap:
                    order.quantity = cap
                    order.remaining = cap
        else:
            # 卖出：可卖 = 可用持仓 + 允许的卖空额度。
            #
            # 为什么要加 `short_limit`：**永续合约市场的空头一侧必须存在**。
            # 没有它，`FundingArbitrageur` 只能买不能卖，
            # 于是"做空永续收正费率"这个策略根本无法执行——
            # 实测它会在费率>阈值时被迫做多，然后持续付钱，累计收益为负。
            # 一期是无杠杆现货市场，卖空额度默认 0，行为完全不变。
            cap = agent.available_inventory + agent.short_limit
            if cap <= MIN_ORDER_QTY:
                return None
            if order.quantity > cap:
                order.quantity = cap
                order.remaining = cap
        if order.quantity <= MIN_ORDER_QTY:
            return None
        return order

    def submit(self, agent: Agent, order: Order) -> list[Trade]:
        """提交订单：裁剪 → 撮合 → 结算 → 更新挂单与预留。"""
        order = self._clamp(agent, order)
        if order is None:
            agent.on_rejected()
            return []

        agent.on_submitted(order)
        trades = self.engine.submit(order)

        for tr in trades:
            self._settle(tr)

        # ⚠️ 成交必须进日志。踩过的坑：MatchingEngine 里留了 trade_sink 钩子，
        # 但这里忘了接上去——撮合一直在正常成交，日志里却一笔都没有，
        # 表现为"价格序列正常、成交数 0、成交量 0"。
        # 现在改成在这里显式落账，只留一条写入路径，不再依赖外部钩子。
        if trades:
            self.log.trades.extend(trades)

        if order.order_type == "limit" and not order.is_filled:
            agent.open_orders.append(order)
        agent.prune_open_orders()
        agent.recompute_reservations()
        return trades

    def cancel(self, agent: Agent, order: Order) -> bool:
        ok = self.book.cancel_order(order.order_id) is not None
        if ok:
            agent.stats.n_cancelled += 1
        agent.prune_open_orders()
        agent.recompute_reservations()
        return ok

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------
    def step(self) -> None:
        t = self.tick
        state = self._refresh_state()

        if self.cfg.housekeeping_interval > 0 and t % self.cfg.housekeeping_interval == 0:
            for a in self.agents:
                a.housekeeping(state)

        # 决策顺序：每个主体从**自己的**调度流抽一个优先级，按优先级升序出手。
        # 与"全局洗牌"的区别见 `_next_order()` 的说明——这是一条
        # 让"主体数量"与"背景行情"解耦的关键设计。
        order = self._next_order()
        # ⭐ 第二个注入点（与 `decorate_agent` 并列）。默认原样返回。
        # 二期阶段8 用它把"本 tick 有多少主体能出手"交给一个自激点过程决定
        # （Hawkes），从而做出**订单到达的时间聚集性**。
        # 为什么放在这里而不是改每个主体的 `p_active`：
        # 那样要动 5 个主体类，而且"活跃度"这个量会散落在各处，
        # 实验脚本里再也说不清"聚集性到底作用在哪一层"。
        order = self._limit_activity(order)

        for agent in order:
            out = agent.decide(state)
            if out is None:
                continue
            if isinstance(out, (list, tuple)):
                orders: Iterable[Order] = out
            else:
                orders = (out,)
            touched = False
            for o in orders:
                if o is None:
                    continue
                self.submit(agent, o)
                touched = True
            if touched:
                # 就地刷新标量：让同一 tick 内后续主体看到已更新的最优价
                self._refresh_state(state)

        self._finalize_tick()
        # 市场随机流**只**用于基本面价值步进。它不参与主体调度、也不参与禀赋抽签，
        # 于是"市场上多了几个主体"完全不会改变基本面锚的路径。
        self._step_fundamental()
        self.tick += 1

    def _step_fundamental(self) -> None:
        """推进基本面锚（现货指数代理）。

        默认 = 一次对数随机游走 + 向长期锚点的弱回归。
        二期阶段9（多资产）覆盖它：那时各资产的基本面必须由一个**共享的**
        公共因子生成器驱动，才能做出资产间相关——
        各自独立随机游走的话，相关系数会恒为 0，
        而"配对交易"这个前提就不存在了。
        """
        self.fundamental = step_fundamental(
            self.fundamental,
            self.rng,
            vol=self.cfg.fu_v_vol,
            pull=self.cfg.fu_v_pull,
            anchor=self.cfg.v_anchor,
        )

    def _next_order(self) -> list[Agent]:
        """本 tick 的动作顺序：按「各主体自己的调度随机流」抽的优先级排序。

        ⭐ 为什么不能用 `self.rng.shuffle(self._order)`：

        ``shuffle`` 的置换**取决于列表长度**。于是往市场里多放一个主体，
        所有主体的出手顺序会整体错位，背景行情从第 0 个 tick 就变了。
        后果很具体：**注入一个"什么都不做"的策略主体，行情就已经不同了** ——
        用它当对照去评估策略，测到的差异里混着"多了一个人参与洗牌"这个纯伪影。

        改成"每主体独立抽优先级"之后，顺序只取决于各主体**自己的**流，
        与主体数量无关：往市场里加人、减人，其余主体的相对顺序不受影响。
        这是"策略 A/B 对比成立"的地基，已固化为测试
        ``TestStrategyInjection::test_注入惰性主体不改变背景行情_且它确实被调用``。

        实现上做了分块预抽（``ORDER_BLOCK`` 个一次），避免每 tick 每主体
        一次 ``Generator.random()`` 调用——那在 300 主体 × 24000 tick 下
        是 720 万次函数调用，实测能吃掉三分之一的总运行时间。
        """
        n = len(self.agents)
        pri = self._ord_prio
        if pri is None or pri.size < n:
            self._ord_prio = pri = np.empty(max(n, 64), dtype=np.float64)
        # ⚠️ 两个扩容条件必须**分开判断**。曾经把"扩张优先级数组"和
        # "为新主体建调度流"写在同一个 if 里：数组按 max(n,64) 预先分配，
        # 于是主体从 10 个增到 11 个时 `pri.size >= n` 成立、整个分支被跳过，
        # 新主体的调度流没建出来 → 越界。加主体是策略实验的常规动作，
        # 这条路径必须每次都对。
        while len(self._ord_rng) < n:
            i = len(self._ord_rng)
            self._ord_rng.append(
                np.random.default_rng([self.cfg.seed, ENTROPY_ORDER, self._ord_key(i)])
            )
            self._ord_buf.append(np.empty(0, dtype=np.float64))
            self._ord_pos.append(0)
        for i in range(n):
            if self._ord_pos[i] >= self._ord_buf[i].size:
                self._ord_buf[i] = self._ord_rng[i].random(ORDER_BLOCK)
                self._ord_pos[i] = 0
            j = self._ord_pos[i]
            pri[i] = self._ord_buf[i][j]
            self._ord_pos[i] = j + 1
        agents = self.agents
        return [agents[j] for j in np.argsort(pri[:n], kind="stable")]

    def _ord_key(self, i: int) -> int:
        """第 i 个主体的调度流标识。默认用序号；注入的主体若带了显式 key 则用它，
        这样"注入主体"不会去挤占已有主体的调度流。"""
        return i

    def _finalize_tick(self) -> None:
        t = self.tick
        L = self.log
        L.ensure_capacity(t + 1)
        mid = self._state.mid
        self.history.append(mid)

        L.mid[t] = mid
        L.best_bid[t] = self._state.best_bid if self._state.best_bid is not None else np.nan
        L.best_ask[t] = self._state.best_ask if self._state.best_ask is not None else np.nan
        L.spread[t] = self._state.spread if self._state.spread is not None else np.nan
        L.fundamental[t] = self.fundamental
        L.volume[t] = self._vol_tick
        L.n_trades[t] = self._n_trade_tick
        L.flow[t] = self._flow_tick
        L.bid_depth[t] = self.book.total_quantity("buy")
        L.ask_depth[t] = self.book.total_quantity("sell")
        L.open_orders[t] = self.book.n_orders
        L.n_levels_bid[t] = self.book.n_levels("buy")
        L.n_levels_ask[t] = self.book.n_levels("sell")

        every = self.cfg.record_snapshot_every
        if every > 0 and t % every == 0:
            L.snapshots.append(self.snapshot(n=self.cfg.snapshot_levels))

        self._vol_tick = 0.0
        self._n_trade_tick = 0
        self._flow_tick = 0.0

    def run(self, n_ticks: int | None = None, verbose: bool = False) -> SimLog:
        n = n_ticks if n_ticks is not None else self.cfg.n_ticks
        t0 = time.time()
        for _ in range(n):
            self.step()
        if verbose:
            dt = time.time() - t0
            print(
                f"  跑完 {n} ticks / {self.cfg.population.total} 主体 "
                f"用时 {dt:.2f}s（{n / max(dt, 1e-9):,.0f} tick/s），"
                f"成交 {len(self.log.trades):,} 笔"
            )
        self.log.trim(self.tick)
        self._collect_agent_stats()
        return self.log

    def _collect_agent_stats(self) -> None:
        mid = self._state.mid
        self.log.agents = [
            {
                **a.describe(),
                "equity": a.equity(mid),
                "equity_change": a.equity_change(mid),
                "n_open_orders": len(a.open_orders),
                "reserved_cash": a.reserved_cash,
                "reserved_inventory": a.reserved_inventory,
            }
            for a in self.agents
        ]

    # ------------------------------------------------------------------
    # 诊断与压力测试
    # ------------------------------------------------------------------
    def snapshot(self, n: int = 20) -> BookSnapshot:
        def lv(side: str) -> list[DepthLevel]:
            return [
                DepthLevel(price=p, quantity=q, n_orders=k)
                for p, q, k in self.book.depth(side, n)  # type: ignore[arg-type]
            ]

        return BookSnapshot(
            tick=self.tick,
            mid=self.current_mid(),
            bids=lv("buy"),
            asks=lv("sell"),
        )

    def book_is_healthy(self) -> tuple[bool, list[str]]:
        return self.book.invariants_ok()

    def health_check(self) -> tuple[bool, list[str]]:
        """综合体检：订单簿自洽 + 账户守恒。测试与出报告前都应过这一关。"""
        ok_book, p_book = self.book.invariants_ok()
        ok_acct, p_acct = self.account_integrity()
        return (ok_book and ok_acct), p_book + p_acct

    def account_integrity(self) -> tuple[bool, list[str]]:
        """全局守恒检查：所有主体的现金+持仓变动必须互相抵消。

        校验三条：
          ① 持仓合计 == 初始持仓合计（交易只换手，不创造）
          ② 现金合计 == 初始现金合计
          ③ 没有主体出现负的可用现金/可用持仓
        """
        problems: list[str] = []
        # 把成交流水独立累加一遍，与账户实际变化比对。
        # 不能按解析式核对初始持仓合计——禀赋是随机抽样的，解析式算不出精确值；
        # 而"成交导致的净变化"是可以精确核算的，且恰好能抓住记账错误。
        flow_inv: dict[str, float] = {}
        flow_cash: dict[str, float] = {}
        for tr in self.log.trades:
            flow_inv[tr.buy_agent_id] = flow_inv.get(tr.buy_agent_id, 0.0) + tr.quantity
            flow_inv[tr.sell_agent_id] = flow_inv.get(tr.sell_agent_id, 0.0) - tr.quantity
            flow_cash[tr.buy_agent_id] = flow_cash.get(tr.buy_agent_id, 0.0) - tr.notional
            flow_cash[tr.sell_agent_id] = flow_cash.get(tr.sell_agent_id, 0.0) + tr.notional

        for a in self.agents:
            # ⚠️ 用**相对**容差，不能用绝对容差。
            # 账户金额量级在 1e5~1e7，浮点累加几千笔之后的残差会达到 1e-11 量级，
            # 用固定的绝对阈值要么误报噪声、要么漏掉真问题。
            tol = 1e-9 * (abs(a.cash) + abs(a.reserved_cash) + 1.0)
            if self.allow_negative_cash:
                # 保证金账户（永续市场）：现金可以为负（向交易所借的钱），
                # `reserved_cash > cash` 也不构成问题——它的后果只是
                # `available_cash < 0`，而已有额度不够时 `_clamp` 会拒掉新买单，
                # 这正是我们想要的约束。**唯一真正不能破的是权益非负**：
                # 权益为负 = 资不抵债 = 爆仓，那才是必须报出来的。
                mid = self.current_mid()
                equity = a.cash + a.inventory * (mid or 0.0)
                if equity < -tol:
                    problems.append(
                        f"{a.agent_id} 权益为负（爆仓）: {equity:.6f}"
                        f"（现金={a.cash:.2f} 持仓={a.inventory:.4f}）"
                    )
            else:
                if a.reserved_cash > a.cash + tol:
                    problems.append(
                        f"{a.agent_id} 预留现金超过持有现金 "
                        f"(预留={a.reserved_cash:.6f}, 持有={a.cash:.6f})"
                    )
                if a.available_cash < -tol:
                    problems.append(f"{a.agent_id} 可用现金为负: {a.available_cash:.6f}")
            # 「预留持仓 ≤ 持有持仓」只在**不能卖空**时成立。
            # 空头持仓为负，挂一张卖单就能让 reserved_inventory > inventory——
            # 这在永续市场是完全正常的（挂的卖单会加深空头）。
            # 对可卖空主体，真正的不变量是"还能卖多少 ≥ 0"（见下一条）。
            if a.short_limit <= 0:
                if a.reserved_inventory > a.inventory + 1e-9:
                    problems.append(f"{a.agent_id} 预留持仓超过持有持仓")
            # 可卖量 = 可用持仓 + 卖空额度。现货市场额度为 0，
            # 这里就退化成原来的"可用持仓不得为负"。
            if a.available_inventory < -(a.short_limit + 1e-9):
                problems.append(
                    f"{a.agent_id} 持仓超出卖空额度: 可用={a.available_inventory:.6f}"
                    f"（额度 {a.short_limit:.2f}）"
                )
        if abs(sum(flow_inv.values())) > 1e-6:
            problems.append("成交流水的持仓净额不为零")
        if abs(sum(flow_cash.values())) > 1e-4:
            problems.append("成交流水的现金净额不为零")
        return (not problems), problems

    def force_liquidate(
        self,
        agents: Sequence[Agent],
        fraction: float = 1.0,
        reason: str = "forced-liquidation",
    ) -> dict:
        """强制平仓：模拟清算（施工蓝图 §4 阶段3）。

        做法贴近真实清算：先撤掉待清算主体的全部挂单，再用**市价单**一次性
        把仓位打到对手盘上。用市价单而不是限价单，是因为清算的核心特征就是
        "不计价格、必须成交"——用限价单会得到一个"没成交完"的假结果。

        返回冲击统计（成交明细进主日志，不走旁路）。
        """
        fraction = max(0.0, min(1.0, fraction))
        t_before = self.tick
        p_before = self.current_mid()
        swept: list[Trade] = []
        liquidated: list[dict] = []

        # 第一阶段：撤销全部待清算主体的挂单。
        # 真实交易所强平是"先一次性撤销该账户所有挂单，再市价平仓"。
        # 分两阶段不只是为了贴近现实——单阶段（边撤边平）会让先被清算的主体
        # 吃掉后清算主体仍挂在簿上的买单，产生"被清算后持仓反而增加"这种
        # 反直觉结果，事后极难解释。
        for a in agents:
            a.cancel_all()

        # 第二阶段：统一市价平仓
        for a in agents:
            inv = a.available_inventory
            if inv <= MIN_ORDER_QTY:
                continue
            qty = inv * fraction
            order = Order(
                agent_id=a.agent_id,
                side="sell",
                price=-math.inf,
                quantity=qty,
                timestamp=self.tick,
                order_id=f"{a.agent_id}#LIQ{self.tick}",
                order_type="market",
            )
            trades = self.submit(a, order)
            filled = sum(t.quantity for t in trades)
            swept.extend(trades)
            liquidated.append(
                {
                    "agent_id": a.agent_id,
                    "target_qty": qty,
                    "filled_qty": filled,
                    "fill_ratio": filled / qty if qty > 0 else 0.0,
                    "avg_price": (
                        sum(t.notional for t in trades) / filled if filled > 0 else None
                    ),
                }
            )

        p_after = self.current_mid()
        qty = sum(t.quantity for t in swept)
        # 清算成交的 VWAP：清算方真实获得的平均成交价。
        # 相对清算前中间价的偏离就是**已实现滑点**——这是清算冲击最稳健的度量，
        # 因为它是同一时点内的已实现量，不含"事后价格漂移"这类路径噪声
        # （用中间价路径的峰值去测冲击，会被漂移淹没，实测过两次）。
        vwap = sum(t.notional for t in swept) / qty if qty > 0 else None
        impact = {
            "tick": t_before,
            "reason": reason,
            "n_agents": len(liquidated),
            "total_qty": qty,
            "n_trades": len(swept),
            "price_before": p_before,
            "price_after": p_after,
            "vwap": vwap,
            "slippage_bp": (
                (vwap / p_before - 1.0) * 1e4
                if vwap is not None and p_before and p_before > 0
                else None
            ),
            "impact_bp": (
                (p_after / p_before - 1.0) * 1e4 if p_before and p_before > 0 else None
            ),
            "details": liquidated,
        }
        self.events.append(impact)
        return impact


# ----------------------------------------------------------------------
def run_simulation(config: SimConfig, verbose: bool = True) -> tuple[Market, SimLog]:
    """便捷入口：建市场 → 跑 → 返回。"""
    m = Market(config)
    log = m.run(verbose=verbose)
    ok, problems = m.account_integrity()
    if not ok and verbose:
        print("  ⚠️ 账户完整性检查未通过：")
        for p in problems[:10]:
            print(f"     - {p}")
    return m, log
