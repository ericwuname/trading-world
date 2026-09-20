"""OKX 风格订单模型与账户账本（A1）。

为什么单独一个模块，而不是给 ``types.Order`` 加字段
----------------------------------------------------
``types.Order`` 是一期就冻结的**撮合层**结构，159 项一期测试 + 二期全部
统计特征都跑在它上面。往里加 ``tdMode``/``posSide``/``attachAlgoOrds`` 会：

1. 让每个 ``Order(...)`` 的构造点都要补参（或依赖默认值，于是新字段
   在旧路径上**永远是默认值**，看起来"支持了"，其实从没被用过）；
2. 让「一期现货语义」与「永续合约语义」在同一个类型里混着，
   而这两者的**不变量是不同的**（现货：``available_cash >= 0``；
   永续：``equity >= 0`` 而 cash 可以为负）。

所以 A1 的做法是**并列一层**：``tw/order_model.py`` 定义 OKX 语义的类型与
换算，``tw/account.py`` 定义保证金账本。原有的 ``Market``/``Order``/``Engine``
一个字不改 —— **A1 的全部行为都可以在完全不动一期代码的前提下验证**。

OKX 语义里最容易搞错的四件事（都查证过真实 API）
=================================================

**① ``posSide`` 与持仓方向不是一回事。**
   真实 OKX 在**单向持仓模式**（``posSide="net"``）下，一个合约只有一个净仓位；
   在**双向持仓模式**下，同一个合约可以同时有多仓与空仓两个独立仓位
   （``posSide="long"`` / ``"short"``），它们**不互相抵消**。
   这不是细节：双向模式下"买 1 手"与"卖 1 手"的结果是**两个仓位同时变大**，
   而不是净仓位归零。本模块两种模式都实现，默认 ``net``
   （因为净仓位模式的不变量更容易验证，而 Agent 一开始用不到双向）。

**② ``reduceOnly`` 不是"只允许减少"。** 它是一条**硬约束**：
   如果这张单会让仓位**变大**（哪怕只是超减一点点），交易所**直接拒单**
   （OKX 错误码 ``51169`` / ``51022`` 一类），不是"部分成交后停下"。
   所以本模块在**提交前**就判，且判定用的是**提交时的仓位**，
   不是"预期待成交后的仓位"。

**③ ``mark`` 触发价是防插针的，不是可选优化。**
   用 ``last``（最新成交价）触发止盈止损，一笔异常成交（插针）就能在
   没人真的想成交的价位触发全市场止损。真实交易所用**标记价**（由
   指数价 + 基差算出，抗单笔操纵）正是为此。
   这与本项目「锚点必须用同时点的真实盘口价」是同一种思路。
   ⇒ 本模块**默认 ``mark``**，要用 ``last`` 必须显式传。

**④ ``post_only`` / ``fok`` / ``ioc`` 是「时间在价格之前」的三个不同取舍。**

   ============  ==========================  ============================
   类型           如果会立即成交              如果没立即成交
   ============  ==========================  ============================
   ``post_only`` **拒单**（绝不主动成交）    留在簿上（保证当 maker）
   ``ioc``       立即成交，**剩余撤销**       整单撤销
   ``fok``       **全部成交否则整单拒**       整单撤销
   ============  ==========================  ============================

   ``post_only`` 的价值在于**保证付 maker 手续费而不是 taker**——
   在真实交易所这是 2~5 倍的费率差。所以它**必须拒单而不是改价**
   （改价就成了另一个策略，且可能成交在比预期差的位置）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Literal

#: 保证金模式。``cash`` 是现货（无杠杆），``cross`` 全仓，``isolated`` 逐仓。
TdMode = Literal["cash", "cross", "isolated"]
#: 持仓方向。``net`` 单向持仓；``long``/``short`` 双向持仓的两条腿。
PosSide = Literal["net", "long", "short"]
#: 委托类型。前两个一期已有，后三个是 OKX 的「时间优先」变体。
OrdType = Literal["limit", "market", "post_only", "ioc", "fok"]
#: 触发价类型。``mark`` 抗插针（默认），``last`` 用最新成交价，``index`` 用指数价。
TriggerPxType = Literal["mark", "last", "index"]

#: 委托价 ``-1`` 在 OKX 里表示「触发后按市价」。这是**协议约定**，
#: 不是本项目的发明；用常量而不是裸 ``-1`` 是为了让读代码的人知道它有含义。
MARKET_ON_TRIGGER = -1.0

#: 数量比较容差，与 ``types.EPS`` 同源。
EPS = 1e-9


# ======================================================================
# 止盈止损
# ======================================================================
@dataclass(slots=True, frozen=True)
class AlgoOrder:
    """随主单附带的止盈止损（OKX ``attachAlgoOrds`` 的语义）。

    用 **frozen** 而不是可变：止盈止损一旦挂在交易所就不该被悄悄改，
    要改必须**撤单重挂**。可变类型会让"我改了 algo 但没通知交易所"
    这类不一致在代码里看不出来。
    """

    #: 触发价。None = 这一侧不挂（只挂止盈或只挂止损是允许的）。
    tp_trigger_px: float | None = None
    sl_trigger_px: float | None = None
    #: 委托价。``MARKET_ON_TRIGGER``(-1) = 触发后走市价（默认，最不容易漏成交）。
    tp_ord_px: float = MARKET_ON_TRIGGER
    sl_ord_px: float = MARKET_ON_TRIGGER
    #: 触发价取哪个价。默认 ``mark``（见模块文档 ③）。
    tp_trigger_px_type: TriggerPxType = "mark"
    sl_trigger_px_type: TriggerPxType = "mark"
    #: 附加单自己的 ID（OKX ``attachAlgoClOrdId``，≤32 字符）。
    attach_algo_cl_ord_id: str = ""

    def __post_init__(self) -> None:
        if self.tp_trigger_px is None and self.sl_trigger_px is None:
            raise ValueError("止盈止损至少要有一边")
        for nm, px in (("tp", self.tp_trigger_px), ("sl", self.sl_trigger_px)):
            if px is not None and not (math.isfinite(px) and px > 0):
                raise ValueError(f"{nm}_trigger_px 必须是正的有限值，收到 {px!r}")
        for nm, typ in (
            ("tp_trigger_px_type", self.tp_trigger_px_type),
            ("sl_trigger_px_type", self.sl_trigger_px_type),
        ):
            if typ not in ("mark", "last", "index"):
                raise ValueError(f"{nm} 非法: {typ!r}")

    def triggers(self, *, mark: float, last: float, index: float,
                 side: str) -> str | None:
        """按当前价格判断是否触发。返回 ``"tp"`` / ``"sl"`` / ``None``。

        ⚠️ **方向不能搞反**（这是最容易写错的一处）：
        给定**开仓方向** ``side``，止盈止损的比较方向是**反的**：

        ==========  ==================  ==================
        开仓方向     止盈触发条件        止损触发条件
        ==========  ==================  ==================
        buy（多）    price **>=** tp     price **<=** sl
        sell（空）   price **<=** tp     price **>=** sl
        ==========  ==================  ==================

        写成"止盈一律 >=、止损一律 <="是**错的**——那个写法里空头的止盈
        会在价格**上涨**时触发，等于把止盈做成了止损。
        """
        def pick(t: TriggerPxType) -> float:
            if t == "mark":
                return mark
            if t == "last":
                return last
            return index

        if side == "buy":
            if self.tp_trigger_px is not None:
                p = pick(self.tp_trigger_px_type)
                if p >= self.tp_trigger_px:
                    return "tp"
            if self.sl_trigger_px is not None:
                p = pick(self.sl_trigger_px_type)
                if p <= self.sl_trigger_px:
                    return "sl"
        else:
            if self.tp_trigger_px is not None:
                p = pick(self.tp_trigger_px_type)
                if p <= self.tp_trigger_px:
                    return "tp"
            if self.sl_trigger_px is not None:
                p = pick(self.sl_trigger_px_type)
                if p >= self.sl_trigger_px:
                    return "sl"
        return None


# ======================================================================
# 订单请求
# ======================================================================
@dataclass(slots=True)
class OrderRequest:
    """一张**待提交**的 OKX 风格订单请求（= 真实 API 的请求体）。

    刻意与撮合层的 ``types.Order`` 分开：
    本类型是**用户/Agent 的意图**，``types.Order`` 是**撮合器的状态机**。
    两者之间的翻译由 ``to_native()`` 显式完成，翻译规则集中在一处。
    """

    inst_id: str
    side: Literal["buy", "sell"]
    ord_type: OrdType
    sz: float
    #: 限价单的价格。市价单必须为 None。
    px: float | None = None
    #: 保证金模式。``cash`` = 现货无杠杆。
    td_mode: TdMode = "cash"
    #: 持仓方向。``net`` = 单向持仓模式。
    pos_side: PosSide = "net"
    #: 只减仓。见模块文档 ②。
    reduce_only: bool = False
    #: 附带止盈止损。
    attach_algo: AlgoOrder | None = None
    #: 客户端订单 ID（OKX 限 32 字符，字母数字）。
    cl_ord_id: str = ""
    #: 杠杆倍数。``td_mode="cash"`` 时必须为 1（现货没有杠杆）。
    lever: float = 1.0

    def __post_init__(self) -> None:
        if self.side not in ("buy", "sell"):
            raise ValueError(f"side 必须是 buy/sell，收到 {self.side!r}")
        if self.ord_type not in ("limit", "market", "post_only", "ioc", "fok"):
            raise ValueError(f"ord_type 非法: {self.ord_type!r}")
        if self.td_mode not in ("cash", "cross", "isolated"):
            raise ValueError(f"td_mode 非法: {self.td_mode!r}")
        if self.pos_side not in ("net", "long", "short"):
            raise ValueError(f"pos_side 非法: {self.pos_side!r}")
        if not (math.isfinite(self.sz) and self.sz > 0):
            raise ValueError(f"sz 必须为正的有限值，收到 {self.sz!r}")
        if not (math.isfinite(self.lever) and self.lever >= 1.0):
            raise ValueError(f"lever 必须 >= 1，收到 {self.lever!r}")
        if self.ord_type == "market":
            if self.px is not None and self.px != MARKET_ON_TRIGGER:
                # 市价单带价格 = 调用方搞错了类型（想要限价单）。
                # 真实 OKX 会直接报参数错误，这里也拒绝，不做"猜意图"的兜底。
                raise ValueError("市价单不能带价格（px 必须为 None）")
        else:
            if self.px is None or not (math.isfinite(self.px) and self.px > 0):
                raise ValueError(
                    f"{self.ord_type} 单必须有正的有限价格，收到 {self.px!r}"
                )
        if self.td_mode == "cash" and self.lever != 1.0:
            raise ValueError("td_mode='cash'（现货）不接受杠杆，lever 必须为 1")
        if self.td_mode != "cash" and self.pos_side != "net" and self.td_mode == "cross":
            # 全仓模式下双向持仓在 OKX 上是允许的，但本项目尚未实现其保证金
            # 口径（双仓共享保证金），先拒绝而不是"看起来支持了"。
            raise ValueError("cross + long/short 的保证金口径未实现，请用 isolated")
        if len(self.cl_ord_id) > 32:
            raise ValueError(f"cl_ord_id 最多 32 字符，收到 {len(self.cl_ord_id)}")

    # ------------------------------------------------------------------
    @property
    def is_market_like(self) -> bool:
        """会不会**立刻**吃对手盘（= 付 taker 费）。

        ``post_only`` **不在**这里——它的定义就是"绝不主动成交"。
        """
        return self.ord_type in ("market", "ioc", "fok")

    @property
    def must_be_maker(self) -> bool:
        return self.ord_type == "post_only"

    @property
    def cancels_remainder(self) -> bool:
        """未成交的部分是否立刻撤销（``ioc`` / ``fok`` 是，``limit`` 不是）。"""
        return self.ord_type in ("ioc", "fok")

    @property
    def requires_full_fill(self) -> bool:
        """是否要求**全部**成交否则整单不成立（只有 ``fok``）。"""
        return self.ord_type == "fok"

    def to_native(self, *, tick: int, order_id: str, price_of, to_tick):
        """翻译成撮合层能吃的 ``Order``。

        翻译规则（集中在这里，别处不许再写一遍）：

        ==============  =====================================================
        OKX 概念         撮合层表示
        ==============  =====================================================
        市价             ``price = ±inf`` + ``order_type="market"``
        限价类           ``price = 吸附到网格`` + ``"limit"``
        ``ioc``/``fok``  先按 limit 进去撮合，**事后**由调用方按
                         ``cancels_remainder`` 撤剩余（撮合器一次调用就撮完，
                         没有"等一会儿"的概念，所以只能这样表达）
        ==============  =====================================================

        ``price_of`` / ``to_tick`` 从订单簿传进来——**价格吸附必须在唯一的
        那处做**（见 ``Market._clamp`` 的注释：预留按原价、结算按网格价
        会逐笔泄漏，且症状极难查）。
        """
        from .types import Order  # 局部 import：避免与 types 循环引用

        if self.is_market_like and self.ord_type == "market":
            px = math.inf if self.side == "buy" else -math.inf
            return Order(
                agent_id="",  # 由调用方填
                side=self.side,
                price=px,
                quantity=self.sz,
                timestamp=tick,
                order_id=order_id,
                order_type="market",
            )
        # 其余（limit / post_only / ioc / fok）都按限价单进撮合器
        assert self.px is not None
        snapped = float(price_of(to_tick(self.px)))
        if snapped <= 0:
            raise ValueError(f"价格 {self.px!r} 吸附到网格后非法（{snapped}）")
        return Order(
            agent_id="",
            side=self.side,
            price=snapped,
            quantity=self.sz,
            timestamp=tick,
            order_id=order_id,
            order_type="limit",
        )

    def with_size(self, sz: float) -> "OrderRequest":
        """改量后返回新对象（风控裁剪用）。

        返回**新对象**而不是就地改：风控层改过的东西必须与原请求可对照，
        否则留痕里"Agent 要了多少"和"实际下了多少"就只剩一个数了——而
        这两个数的差**正是风控层的贡献**，是 A4 评估里要单独报的一项。
        """
        return replace(self, sz=float(sz))


# ======================================================================
# 拒单原因
# ======================================================================
class OrderRejected(Exception):
    """订单被拒。``code`` 对齐 OKX 的错误码语义，便于日后接真实网关。"""

    def __init__(self, code: str, msg: str) -> None:
        super().__init__(f"{code}: {msg}")
        self.code = code
        self.msg = msg


#: 本项目用到的拒单码。**不复用真实 OKX 的数字码**——
#: 真实码的含义会随交易所版本变，而我们的语义是稳定的；
#: 用 ``TW-`` 前缀一眼能看出"这是模拟器自己的判断"而不是"交易所回的"。
REJECT_CODES: dict[str, str] = {
    "TW-1001": "post_only 会立即成交，已拒（保证当 maker）",
    "TW-1002": "fok 无法全部成交，整单已拒",
    "TW-1003": "reduce_only 但会增大仓位",
    "TW-1004": "reduce_only 但当前无仓位可减",
    "TW-1005": "保证金不足（保证金率低于开仓要求）",
    "TW-1006": "超出最大可用余额（现货）",
    "TW-1007": "数量低于最小下单量",
    "TW-1008": "已达最大挂单数",
    "TW-1009": "杠杆超出该合约上限",
    "TW-1010": "该持仓方向在当前模式下不可用",
}


def _reject(code: str) -> OrderRejected:
    return OrderRejected(code, REJECT_CODES.get(code, "未定义的拒单原因"))


# ======================================================================
# 预检：不需要账户也能判的那几件事
# ======================================================================
def precheck(
    req: OrderRequest,
    *,
    best_bid: float | None,
    best_ask: float | None,
    quantity_step: float = 0.0,
    min_qty: float = 0.0,
    n_open_orders: int = 0,
    max_open_orders: int = 10_000,
    max_lever: float = 125.0,
    fillable_qty: float | None = None,
) -> None:
    """提交前的**结构性**检查（不看账户）。不通过就抛 ``OrderRejected``。

    为什么要把这部分单独拆出来：这几条**只依赖订单本身与盘口**，
    与"是谁下的、他有多少钱"无关。拆出来后：
      · 可以在 Agent 拿到账户之前就拒掉明显非法的单（省一次昂贵计算）；
      · 测试可以只构造盘口就覆盖全部分支，不用搭一个完整市场。

    ``fillable_qty``：这张单**当下**最多能成交多少（由调用方用订单簿算）。
    只有 ``post_only`` 与 ``fok`` 需要它——因为它们的结果取决于"现在会不会成交"。
    """
    if req.sz <= min_qty + EPS:
        raise _reject("TW-1007")
    if quantity_step > 0:
        # 数量必须落在步长网格上。真实交易所是**拒单**而不是向下取整——
        # 静默改量会让"我要买 1 个"变成"买了 0.99 个"，而 Agent 无从知道。
        n = req.sz / quantity_step
        if abs(n - round(n)) > 1e-6:
            raise OrderRejected(
                "TW-1007",
                f"数量 {req.sz} 不在步长 {quantity_step} 的网格上",
            )
    if n_open_orders >= max_open_orders:
        raise _reject("TW-1008")
    if req.lever > max_lever:
        raise _reject("TW-1009")

    if req.must_be_maker:
        # post_only：只要**会**和盘口交叉就必须拒。
        if req.side == "buy":
            if best_ask is not None and req.px is not None and req.px >= best_ask:
                raise _reject("TW-1001")
        else:
            if best_bid is not None and req.px is not None and req.px <= best_bid:
                raise _reject("TW-1001")

    if req.requires_full_fill and fillable_qty is not None:
        if fillable_qty + EPS < req.sz:
            raise _reject("TW-1002")


def apply_reduce_only(
    req: OrderRequest, *, position_qty: float, position_side: str
) -> OrderRequest:
    """落实 ``reduce_only`` 的**硬约束**（见模块文档 ②）。

    ``position_qty`` 是**带符号**的当前仓位（多正空负）。

    两条规则：
      1. 方向必须**与仓位相反**（平仓才叫 reduce）；
         同向就是加仓 ⇒ 拒。
      2. **数量不得超过仓位绝对值** ⇒ 超出部分**裁剪**而不是拒单。

    ⚠️ 第 2 条为什么是裁剪而不是拒：真实 OKX 对 ``reduce_only`` 超量是
    **按仓位上限裁掉**（``sz`` 被静默改成可减的量），因为它明确知道
    你的意图是"平掉"；而第 1 条方向错了是**语义错误**，交易所无法猜意图，
    只能拒。本项目照此办理——**"拒绝"和"改量"的边界由「能不能推出意图」决定**。
    """
    if not req.reduce_only:
        return req
    if abs(position_qty) <= EPS:
        raise _reject("TW-1004")
    closing_side = "sell" if position_qty > 0 else "buy"
    if req.side != closing_side:
        raise _reject("TW-1003")
    if req.sz > abs(position_qty) + EPS:
        return req.with_size(abs(position_qty))
    return req
