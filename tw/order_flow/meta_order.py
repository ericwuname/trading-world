"""长记忆订单流：元订单与拆分执行（二期阶段6，LMF 理论）。

要修的是什么
------------
一期报告定位到但没修的核心缺陷：**冲击函数不是凹的**。
一期的冲击幂律指数 k 落在 0.61~1.25，而真实市场是 **k ≈ 0.5（平方根律）**。
不凹，意味着"大单的成本被线性放大"——用它评估执行类策略会系统性高估大单成本。

理论链条（Lillo–Mike–Farmer）
-----------------------------
    大额交易者必须**拆分**元订单（不然一次砸穿盘口）
      → 子订单**同向且持续**，于是订单符号序列出现**长记忆正自相关**
      → 流动性提供者发现"同方向的单还会来"
      → 主动在**该方向稀疏挂单**（撤单 / 挂远）
      → 局部流动性变薄，但只在**那个方向**
      → 小额订单也有不成比例的大冲击
      → 冲击函数变**凹**

本模块负责链条的**第一环**：让主体真的拆元订单。
第二环（流动性提供者感知并稀疏挂单）在 ``tw/agents/adaptive_liquidity.py``。

⭐ 关键设计：方向从**基础决策**里来，不由元订单自己编
----------------------------------------------------
``MetaOrderMixin`` 不自己产生方向信号——它调用被它包装的那个类的
``decide()``，看它返回什么方向的单，然后**把那个方向锁定并持续执行**。

为什么不自己生成方向：那样做出来的"长记忆"是**外生塞进去的**
（我们凭空写了一个相关性进去，再把它测出来，等于自证）。
而"把主体本来就要做的一笔小单放大成元订单"才是真实机制：
方向仍然来自主体的私有信号（图表派的动量、零智能的随机），
长记忆是**拆分执行这个行为**的副产品，不是假设。
这是本项目「因果测量」要求的一个直接推论：**不要把你打算测量的东西
当成输入喂进去。**

不过要说清楚它测出了什么：方向来自基础信号，而基础信号（尤其图表派的
动量）本身可能有自相关。所以 E6.1 的对照跑（同样池子、不开启元订单）
是**必需**的——只有"开启 − 不开"的差才归因给这一环。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..types import MarketState, Order

#: spawn_rng 的用途标签。各用途必须独占，不能与他人重复。
ENTROPY_META_ORDER = 0x3E7A11


@dataclass(slots=True)
class MetaOrderConfig:
    """元订单参数。默认值是**起点**，最终取值由 E6.5 的敏感性扫描定。"""

    #: 每个"元订单型主体"每 tick 启动一个新元订单的概率。
    p_meta_start: float = 0.02
    #: 元订单总量的 Pareto 分布形状参数。越小尾部越重。
    #: ⚠️ 必须 > 1，否则分布均值发散、拆分会永不结束。
    pareto_alpha: float = 1.5
    #: Pareto 的尺度下界（最小元订单总量，单位：手）。
    pareto_xmin: float = 1.0
    #: 总量标度。天然单位是"手"，但一个主体一次想做 3 手和一次想做 30 手
    #: 对整个市场的影响差一个数量级，所以这个乘子是把"主体级的意图"
    #: 映射到"市场级的冲击"的旋钮。E6.5 会扫它。
    pareto_scale: float = 1.0
    #: 子订单到达强度：每 tick 真的吐出一张子单的概率（否则空转一 tick）。
    participation_rate: float = 0.15
    #: 子订单规模 ~ Exp(child_qty_mean)。
    child_qty_mean: float = 1.0
    #: 单个元订单最多拆多少张子单。防止极端 Pareto 采样导致无限拆分
    #: （Pareto 的尾部没有上界，没有这道闸门就会遇到"一个主体执行一万 tick"）。
    max_child_orders: int = 200

    def __post_init__(self) -> None:
        if not (0.0 <= self.p_meta_start <= 1.0):
            raise ValueError("p_meta_start 必须在 [0,1]")
        if self.pareto_alpha <= 1.0:
            raise ValueError("pareto_alpha 必须 > 1，否则均值发散、拆分不终止")
        if self.pareto_xmin <= 0:
            raise ValueError("pareto_xmin 必须为正")
        if self.pareto_scale <= 0:
            raise ValueError("pareto_scale 必须为正")
        if not (0.0 < self.participation_rate <= 1.0):
            raise ValueError("participation_rate 必须在 (0,1]")
        if self.child_qty_mean <= 0:
            raise ValueError("child_qty_mean 必须为正")
        if self.max_child_orders < 1:
            raise ValueError("max_child_orders 必须 >= 1")

    def expected_total(self) -> float:
        """Pareto 分布的均值。``alpha <= 1`` 时发散（已在构造时拦住）。

        存在的意义是给标定一个解析参照：``pareto_scale`` 该调多大，
        可以直接从"期望总量应该是多少手"倒推，不必盲扫。
        """
        return self.pareto_xmin * self.pareto_alpha / (self.pareto_alpha - 1.0)

    def expected_children(self) -> float:
        """一个元订单平均拆成多少张子单（受 ``max_child_orders`` 截断）。

        这决定了**元订单的最大执行周期**，而后者决定了 E6.4 的观测窗口
        至少要开多长——观测窗口短于执行周期就会低估长记忆（见模块文档末）。
        截断用解析近似（Pareto 尾部的截断期望没有闭式），只做量级参照用。
        """
        n_by_qty = self.expected_total() * self.pareto_scale / self.child_qty_mean
        return float(min(self.max_child_orders, max(1.0, n_by_qty)))

    def max_execution_ticks(self) -> int:
        """元订单最长可能占用的 tick 数（上界）。用到最坏情况，不用均值。"""
        return int(np.ceil(self.max_child_orders / self.participation_rate))


class MetaOrderState:
    """挂在主体实例上：它当前是否在一个未完成的元订单执行过程中。"""

    __slots__ = ("active", "direction", "remaining_qty", "child_count",
                 "issued_qty", "started_tick")

    def __init__(self) -> None:
        self.active = False
        self.direction: str | None = None      # 'buy' / 'sell'
        self.remaining_qty = 0.0
        self.child_count = 0
        #: 已发出的子单量累计（归因用：能核对"发出去的量 == 总量"）。
        self.issued_qty = 0.0
        self.started_tick = -1

    def start(self, direction: str, total_qty: float, tick: int) -> None:
        if direction not in ("buy", "sell"):
            raise ValueError(f"方向必须是 buy/sell，收到 {direction!r}")
        self.active = True
        self.direction = direction
        self.remaining_qty = float(total_qty)
        self.child_count = 0
        self.issued_qty = 0.0
        self.started_tick = int(tick)

    def finish(self) -> None:
        self.active = False
        self.direction = None
        self.remaining_qty = 0.0

    @property
    def locked_direction(self) -> str:
        """当前锁定的方向。**没有在执行的元订单时调用它应当直接报错。**

        存在的意义是防一类静默 bug：拆分过程中从 ``direction`` 读方向时，
        如果它已经是 ``None``（例如代码顺序写成"先 ``finish()`` 再读方向"），
        ``str(None)`` 会得到字符串 ``"None"``，然后
        ``if side == "buy"`` 判假 → **每张尾单都反向**。
        不报错、不崩溃，只是订单符号 ACF 被系统性压低。
        （自测时真的踩到了，由 ``test_方向全程不翻转`` 抓到。）
        """
        if not self.active or self.direction is None:
            raise RuntimeError(
                "元订单已结束，方向不再有效——"
                "调用方大概率把 finish() 写在了读方向之前"
            )
        return str(self.direction)

    def describe(self) -> dict:
        return {
            "active": self.active,
            "direction": self.direction,
            "remaining_qty": self.remaining_qty,
            "child_count": self.child_count,
            "issued_qty": self.issued_qty,
        }


class MetaOrderMixin:
    """让任意 Agent 子类获得"把一笔意图拆成多张子单持续执行"的能力。

    用法::

        class MetaOrderChartist(MetaOrderMixin, Chartist): ...

    ⚠️ MRO 顺序不能反。``MetaOrderMixin`` 必须在前，它的 ``decide`` 才会
    覆盖基类的；反过来写（``class X(Chartist, MetaOrderMixin)``）会让基类的
    ``decide`` 生效，元订单机制**静默失效**——不报错，只是长记忆永远不出现。

    ⚠️ ``__init__`` 里**不能**立刻建随机流。``spawn_rng`` 依赖
    ``self._seed_key``，而那个值由 ``Market.add_agent`` 在建主体时写入，
    比 ``__init__`` 晚。一期踩过同一个坑（``Agent.bind_seed`` 的注释）。
    所以随机流是懒建的。
    """

    #: 供归因区分"哪一种主体"，不覆盖基类的 KIND 语义
    META_KIND = "meta"

    def __init__(self, *args, meta_config: MetaOrderConfig | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.meta_config = meta_config or MetaOrderConfig()
        self.meta_state = MetaOrderState()
        self._rng_meta: np.random.Generator | None = None
        #: 归因：本主体累计启动了多少个元订单、发出多少子单。
        self.n_meta_started = 0
        self.n_children = 0

    # ------------------------------------------------------------------
    def meta_rng(self) -> np.random.Generator:
        """元订单专用的独立随机流（懒建）。

        为什么不用 ``self.rng``：主体的决策流消耗次数取决于它活跃多少次，
        而"活跃多少次"受行情影响。元订单的抽取若与决策流共用一条，
        改了行情（或改了别的参数）就会顺带改变元订单序列，
        于是"元订单参数的影响"和"随机流相位的影响"纠缠在一起，无法归因。
        （一期"主体数量与背景行情解耦"是同一类问题的另一面。）
        """
        if self._rng_meta is None:
            self._rng_meta = self.spawn_rng(ENTROPY_META_ORDER)
        return self._rng_meta

    # ------------------------------------------------------------------
    def sample_meta_total(self) -> float:
        """Pareto 采样元订单总量。

        ``X = xmin · (1−U)^(−1/α)``，``U ~ U(0,1)``。
        这是 Pareto **逆变换采样**的标准形式：
        P(X > x) = (xmin/x)^α，尾部按幂律衰减，所以存在"偶尔一笔巨单"。

        ⚠️ 用错公式（例如漏掉 ``−1/α`` 的幂运算）会让分布退化成近均匀，
        尾部消失——而"长记忆来自重尾"这条链就断了。
        症状不是报错，是**订单符号 ACF 掉得很快**（没有长尾就没有长记忆）。
        """
        cfg = self.meta_config
        u = float(self.meta_rng().random())
        # u 可能恰好为 0（numpy 的 random() 是 [0,1)），此时 (1-u)=1 → X=xmin，合法
        base = cfg.pareto_xmin * (1.0 - u) ** (-1.0 / cfg.pareto_alpha)
        return base * cfg.pareto_scale

    def sample_child_qty(self) -> float:
        """子单规模 ~ Exp(child_qty_mean)。"""
        return max(1e-6, float(self.meta_rng().exponential(self.meta_config.child_qty_mean)))

    def maybe_start_meta_order(self, direction: str, tick: int) -> bool:
        """没有在执行中时，按概率启动一个新元订单。返回是否真的启动了。

        ⚠️ ``p_meta_start <= 0`` 时**提前返回，连随机数都不抽**。
        为什么在意这一次 ``random()``：本阶段要做的是"同一批主体，
        只切换元订单开关"的 A/B 对照。如果关掉开关仍然每 tick 消耗一次
        随机流，那个主体的**其他**随机消费（在它自己的流上）就会错位——
        于是两组的差异里混进了"随机流相位差"，不再只是机制之差的度量。
        "关闭 = 零副作用"是对照实验能成立的前提。
        """
        if self.meta_config.p_meta_start <= 0.0:
            return False
        if self.meta_state.active:
            return False
        if direction not in ("buy", "sell"):
            return False
        if float(self.meta_rng().random()) >= self.meta_config.p_meta_start:
            return False
        total = self.sample_meta_total()
        if total <= 1e-9:
            return False
        self.meta_state.start(direction, total, tick)
        self.n_meta_started += 1
        return True

    def next_child(self) -> tuple[str, float] | None:
        """决定这一 tick 是否吐出一张子单；是则返回 ``(方向, 数量)``。

        返回 None 表示"这一 tick 不下子单"——但**方向保持锁定**，
        下一次吐单仍是同一方向。这正是长记忆的来源：
        方向在多个 tick 上被重复，而不是每 tick 重新抽。
        """
        st = self.meta_state
        if not st.active:
            return None
        cfg = self.meta_config
        if float(self.meta_rng().random()) > cfg.participation_rate:
            return None
        # ⚠️ **必须在 finish() 之前把方向取出来。**
        # `finish()` 会把 `direction` 置成 None，而"最后一张子单"恰恰
        # 是紧接着 `finish()` 之后返回的那张——先 finish 再读方向，
        # 拿到的就是 None，然后 `_child_order` 里 `side == "buy"` 判假，
        # 于是**每个元订单的最后一张子单都会反向**。
        # 这个 bug 不会报错，症状是订单符号 ACF 被系统性压低
        # （每一段同向序列的尾巴都被翻掉），看起来像"长记忆不够强"。
        # 自测时正是 `test_方向全程不翻转` 抓到的。
        # 用 locked_direction 而不是裸读 direction：它会在"已经 finish 了
        # 却还要读方向"时直接抛错，而不是悄悄返回 "None"（见该属性的说明）。
        side = st.locked_direction
        qty = min(st.remaining_qty, self.sample_child_qty())
        if qty <= 1e-9:
            st.finish()
            return None
        st.remaining_qty -= qty           # ⚠️ 必须是**减**。写成 += 会让
                                          # 剩余量越滚越大、永不终止，
                                          # 直到 max_child_orders 兜住——
                                          # 于是每个元订单都恰好拆满 200 张，
                                          # 重尾分布被抹平成常数。
        st.issued_qty += qty
        st.child_count += 1
        self.n_children += 1
        if (st.remaining_qty <= 1e-9
                or st.child_count >= cfg.max_child_orders):
            st.finish()
        return (side, qty)


    # ------------------------------------------------------------------
    def decide(self, state: MarketState) -> Order | Sequence[Order] | None:
        """包住基类的 ``decide``，插入拆分执行逻辑。

        三种情形：
        ① 正在执行元订单 → 吐出子单（方向沿用启动时锁定的那个）；这一 tick
           **不**走基础逻辑，否则主体的活跃度会被翻倍（基础单 + 子单）。
        ② 不在执行中 → 先问基础类要一张单，从它的方向上取"意图"
           （方向是**外生给定**的，不是本模块编的）；
        ③ 基础类这一 tick 不下单 → 什么都不做。
        """
        st = self.meta_state
        if st.active:
            child = self.next_child()
            if child is None:
                return None
            side, qty = child
            return self._child_order(state, side, qty)

        base = super().decide(state)           # type: ignore[misc]
        side = self._side_of(base)
        if side is None:
            return base
        # 启动成功的那个 tick **不额外下单**：否则"启动瞬间"会同时出现
        # 基础单和第一张子单，主体瞬时活跃度翻倍，污染"活跃度"这类统计。
        self.maybe_start_meta_order(side, state.tick)
        return base

    @staticmethod
    def _side_of(decision) -> str | None:
        """从基础决策里取出方向。返回 None 表示"这一 tick 没有意图"。

        只取**第一张**单的方向。做市商那种"同时挂买卖"的返回里，
        净值是零，谈不上"一笔有方向的意图"，所以返回 None——
        元订单机制对这类主体不适用（它本来就不是拿方向赌的）。
        """
        if decision is None:
            return None
        if isinstance(decision, (list, tuple)):
            sides = {o.side for o in decision if o is not None}
            if len(sides) != 1:
                return None
            return next(iter(sides)) if sides else None
        return getattr(decision, "side", None)

    def _child_order(self, state: MarketState, side: str, qty: float):
        """把子单转成订单。

        子单**贴对手价挂限价单**（不是市价单）：真实执行算法用被动报价
        赚价差，而且市价单会立刻吃掉盘口、把"冲击函数"这件事从
        "流动性被消耗"变成"盘口被打穿"，两者不是同一个机制。
        贴对手价成交概率高、又不会无脑吃穿多档。
        """
        if side == "buy":
            return self.buy_order(state, qty)
        return self.sell_order(state, qty)

    # ------------------------------------------------------------------
    def describe(self) -> dict:
        d = super().describe()                  # type: ignore[misc]
        d.update({
            "n_meta_started": self.n_meta_started,
            "n_children": self.n_children,
            "meta_active": self.meta_state.active,
            "meta_remaining": self.meta_state.remaining_qty,
        })
        return d
