"""逐根 K 线的执行模拟器（A4）—— 没有它，任何"收益"都是空话。

为什么必须单独有一个执行层
--------------------------
A3 能回答"Agent 想干什么"，但回答不了"**干了之后赚不赚**"。
要回答后者，必须把"意图"变成"成交"，并把成交记进账本
（``MarginAccount``，A1 已经建好且测穷）。

⚠️⚠️ **这一层是整个 A4 最需要在报告里写清楚"它不精确"的地方。**
我们手上只有 OHLCV——**没有逐笔成交、没有盘口、没有排队位置**。
用一根 K 线的四个数去还原"这一小时里价格怎么走的"，必然丢掉信息。
所以本模块的立场是：

> **宁可系统性偏悲观，也不要偏乐观。**
> 一个偏乐观的回测会让人以为策略能上线；一个偏悲观的回测最多让人错过。

据此定下四条**明确的假设**（不是"实现细节"，是结论的一部分）：

| # | 假设 | 偏乐观还是偏悲观 |
|---|---|---|
| 1 | 市价单在**次根开盘价** ± 滑点成交 | 中性（次根开盘是决策后最早的可得价） |
| 2 | 限价单只要 `low ≤ px ≤ high` 就算成交，成交价 = px | ⚠️ **偏乐观**（没有排队位置，可能排在队尾没轮到） |
| 3 | 当根内 TP 与 SL **同时**落在区间里 ⇒ **按 SL 处理** | ✅ **偏悲观**（这是我们主动选的保守解） |
| 4 | 标记价用当根**收盘价**近似 | 中性偏乐观（真实 mark 来自标记价指数，波动更小） |

第 3 条值得单独说：**这是 bar 级回测无法回避的歧义**。
真实世界里 TP 和 SL 哪个先到，取决于这一小时内价格的**路径**；
而我们只看到一个区间。任何"选一个"的做法都是在补路径——
选 TP 会让回测变好看，选 SL 会让它变难看。
**A1 的 ``AlgoOrder.triggers`` 回答不了这个问题**：它问的是
"给定一个已知价格，触发了吗"；而这里问的是"只给一个区间，
哪个先触发"——**严格更难的问题，且必须带一个假设**。
本模块显式选 SL，并把这个选择记在 ``Fill.reason`` 里，可被审计。

另外两条工程约定
----------------
- **决策与成交必须错开一根**：bar i 收盘做决策（可见数据到 i），
  订单只能从 bar i+1 开始成交。不做这个错位，就等于**让订单回到过去成交**。
- **成本必须事先定义**（设计方案 §2「成本模型事先定义」）：
  费率、滑点、成本倍数都在 :class:`ExecConfig` 里，不散落在调用处。
  「在漂亮回测之后再加成本，是边缘系统活太久的原因」。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .account import MarginAccount
from .order_model import AlgoOrder, OrderRequest

#: 一根 K 线里能拿到的四个数。
#: ⚠️ 刻意用一个**窄接口**（而不是直接吃 ``marketdb.Series``）：
#: 这样测试不需要造一整个 Series，也让"执行层到底看了哪几个数"
#: 这件事在类型上就是显式的。
Bar = tuple[float, float, float, float]   # (open, high, low, close)


# ======================================================================
# 成本模型
# ======================================================================
@dataclass(slots=True)
class ExecConfig:
    """成本与成交假设。**全部事先写死在这里。**

    ``cost_multiplier`` 是成本敏感性实验用的（设计方案 §2 要求报
    成本 ×2 / ×5 后的表现）：它同时放大手续费与滑点。
    ⚠️ 只放大成本、**不改成交概率**——那才是"成本敏感性"的定义。
    若连成交率也一起改，量就变成了"换一个更差的市场"，不是"成本更高"。
    """

    taker_fee: float = 0.0005
    maker_fee: float = 0.0002
    #: 市价单额外滑点（基点）。1bp = 万分之一。
    slippage_bps: float = 1.0
    #: 成本倍数（×1 / ×2 / ×5）。
    cost_multiplier: float = 1.0
    #: 限价单最多等几根；过了就过期（避免"永远挂着一单，事后说它总会成交"）。
    max_wait_bars: int = 3
    #: 未成交的限价单是否自动过期。⚠️ 关掉它（False）会让回测里
    #: 堆积一批"迟早会成交"的挂单——那是典型的乐观偏差。
    expire_unfilled: bool = True

    def __post_init__(self) -> None:
        for nm, v in (("taker_fee", self.taker_fee), ("maker_fee", self.maker_fee),
                      ("slippage_bps", self.slippage_bps),
                      ("cost_multiplier", self.cost_multiplier)):
            if v != v or v < 0:
                raise ValueError(f"{nm} 必须是非负有限值，收到 {v!r}")
        if self.max_wait_bars < 1:
            raise ValueError(f"max_wait_bars 必须 >= 1，收到 {self.max_wait_bars}")

    @property
    def eff_taker(self) -> float:
        return self.taker_fee * self.cost_multiplier

    @property
    def eff_maker(self) -> float:
        return self.maker_fee * self.cost_multiplier

    @property
    def eff_slippage(self) -> float:
        return self.slippage_bps * self.cost_multiplier

    def fee(self, notional: float, *, is_maker: bool) -> float:
        r = self.eff_maker if is_maker else self.eff_taker
        return abs(float(notional)) * r

    def describe(self) -> dict[str, Any]:
        """可写进报告的假设描述（数字必须能被复核）。"""
        return {
            "taker_fee": self.eff_taker,
            "maker_fee": self.eff_maker,
            "slippage_bps": self.eff_slippage,
            "cost_multiplier": self.cost_multiplier,
            "max_wait_bars": self.max_wait_bars,
            "expire_unfilled": self.expire_unfilled,
        }

    @classmethod
    def with_multiplier(cls, m: float, **kw: Any) -> "ExecConfig":
        return cls(cost_multiplier=m, **kw)


# ======================================================================
# 成交
# ======================================================================
@dataclass(slots=True)
class Fill:
    """一笔成交。``reason`` 是**审计字段**——它说明这笔是怎么来的。"""

    bar_index: int
    inst_id: str
    side: str
    qty: float
    price: float
    is_maker: bool
    fee: float
    #: ``market`` / ``limit`` / ``post_only`` / ``tp`` / ``sl`` / ``liquidate``
    reason: str
    #: 关联的订单号（风控后落成的 ClOrdId）
    cl_ord_id: str = ""
    #: ⭐ **不含滑点的参考价**。有了它才能把"滑点成本"从总成本里
    #: 单独拆出来（执行层要报「实际滑点 vs 预期」）。
    #: 没有这个字段时，滑点会**融进成交价**、再也分不出来——
    #: 于是「成本 ×2 后还剩多少」这个问题只能靠重跑回答，不能靠现算。
    ref_price: float = 0.0
    #: 滑点成本（正数 = 吃亏）。买贵了或卖便宜了都记正。
    slippage_cost: float = 0.0

    @property
    def notional(self) -> float:
        return abs(self.qty * self.price)

    @property
    def slippage_bps(self) -> float:
        """相对参考价的滑点（基点）。``ref_price`` 为 0 时返回 0。"""
        if not self.ref_price:
            return 0.0
        d = (self.price - self.ref_price) / self.ref_price
        if self.side == "sell":
            d = -d
        return d * 10_000.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "bar": self.bar_index,
            "side": self.side,
            "qty": self.qty,
            "price": self.price,
            "ref_price": self.ref_price,
            "slippage_bps": round(self.slippage_bps, 4),
            "slippage_cost": self.slippage_cost,
            "is_maker": self.is_maker,
            "fee": self.fee,
            "reason": self.reason,
            "clOrdId": self.cl_ord_id,
        }


@dataclass(slots=True)
class OpenOrder:
    """一张已提交、等着成交的订单。"""

    req: OrderRequest
    submit_bar: int
    cl_ord_id: str
    #: 已经试着成交过几根
    waited: int = 0

    @property
    def is_market_like(self) -> bool:
        return self.req.is_market_like

    @property
    def must_be_maker(self) -> bool:
        return self.req.must_be_maker


# ======================================================================
# 执行器
# ======================================================================
@dataclass(slots=True)
class BarExecutor:
    """把订单意图变成成交，并记进 ``MarginAccount``。

    状态：未成交订单、挂在仓上的 TP/SL。
    ⚠️ **刻意是有状态的**（与 A3 的 Agent 相反）：账本与挂单本来就是
    有状态的东西，硬做成无状态只会把状态推到调用方，让"谁负责挂单"
    变得不清楚。**该有状态的地方有状态，该无状态的地方无状态**——
    两者的判据是"它是不是一次决策的固有部分"。
    """

    account: MarginAccount
    config: ExecConfig = field(default_factory=ExecConfig)
    inst_id: str = "BTC-USDT-SWAP"
    lever: float = 1.0
    #: 未成交的挂单
    open_orders: list[OpenOrder] = field(default_factory=list)
    #: 挂在当前仓上的止盈止损（开仓时设置，平仓时清除）
    algo: AlgoOrder | None = None
    #: 全部成交（按时间顺序）
    fills: list[Fill] = field(default_factory=list)
    #: 被强平的次数与由此产生的成本
    liquidations: int = 0
    #: 因为"没过期但也没成交"而被丢弃的订单数（乐观偏差的体检指标）
    expired_orders: int = 0
    #: 提交过的订单总数
    submitted: int = 0
    #: 每根的权益快照（用于回撤/夏普）
    equity_curve: list[float] = field(default_factory=list)

    # -- 提交 ------------------------------------------------------------
    def submit(self, req: OrderRequest, *, submit_bar: int,
               cl_ord_id: str = "") -> OpenOrder:
        """提交一张订单。**只能从 ``submit_bar + 1`` 开始成交**（见模块文档）。

        ⚠️ **必须校验标的一致**：一张 ``inst_id="BTC"`` 的订单被
        ``inst_id="BTC-USDT-SWAP"`` 的执行器接下来，会**静默地
        对着另一个标的的仓位记账**——留痕里看不出任何异常，
        只有逐笔对账才发现。这类"接错了对象"是本项目最警惕的一类，
        所以这里**直接拒**，不做任何"自动对齐"。
        """
        if req.inst_id != self.inst_id:
            raise ValueError(
                f"订单标的 {req.inst_id!r} 与本执行器的 {self.inst_id!r} 不一致。"
                f"静默接下会让记账对错仓位——请为每个标的各建一个执行器。"
            )
        self.submitted += 1
        oo = OpenOrder(req=req, submit_bar=int(submit_bar),
                       cl_ord_id=cl_ord_id or req.cl_ord_id)
        self.open_orders.append(oo)
        return oo

    def cancel_all(self, *, reason: str = "") -> int:
        n = len(self.open_orders)
        self.open_orders.clear()
        return n

    # -- 每根推进 ---------------------------------------------------------
    def on_bar(self, i: int, bar: Bar) -> list[Fill]:
        """用第 ``i`` 根的 OHLC 尝试成交。

        顺序（**有意如此**）：
        1. 先处理 TP/SL（**保护性出场优先**：真实交易里止盈止损是
           交易所侧的触发单，不等新订单排队）；
        2. 再处理挂着的限价/市价单。

        反过来的话，同一根里"先开新仓再触发出场"会发生，
        而那在真实交易所里几乎不可能——等于**白送一次当根的进出**。
        """
        o, h, l, c = (float(x) for x in bar)  # noqa: E741
        out: list[Fill] = []

        # ---- 1. TP/SL（保护性优先）------------------------------------
        out.extend(self._check_algo(i, h, l, c))

        # ---- 2. 挂单 ---------------------------------------------------
        still: list[OpenOrder] = []
        for oo in self.open_orders:
            # ⚠️ 决策与成交错开一根：submit_bar 那根**不能**成交
            if i <= oo.submit_bar:
                still.append(oo)
                continue
            oo.waited += 1
            f = self._try_fill(oo, i, o, h, l, c)
            if f is not None:
                out.append(f)
                continue
            if oo.is_market_like:
                # 市价单不该成交不了；成交不了说明遇到了非法 bar（如 high<low）
                raise ValueError(
                    f"市价单在第 {i} 根没能成交：bar={(o, h, l, c)}。"
                    f"这通常意味着行情数据本身有问题，不该静默跳过。"
                )
            if self.config.expire_unfilled and oo.waited >= self.config.max_wait_bars:
                self.expired_orders += 1
                continue
            still.append(oo)
        self.open_orders = still
        return out

    # -- 成交判定 ---------------------------------------------------------
    def _try_fill(self, oo: OpenOrder, i: int, o: float, h: float,
                  l: float, c: float) -> Fill | None:  # noqa: E741
        req = oo.req
        if req.ord_type == "market":
            # 次根开盘 + 滑点（买单向上滑、卖单向下滑）
            slip = self.config.eff_slippage / 10_000.0
            px = o * (1.0 + slip) if req.side == "buy" else o * (1.0 - slip)
            return self._record(i, req, px, is_maker=False,
                                reason="market", coid=oo.cl_ord_id, ref=o)

        px = float(req.px)  # 限价类一定有价（OrderRequest 已校验）
        touched = (l <= px <= h)
        if not touched:
            return None
        # ⚠️ 假设 2：碰到就算成交（偏乐观）。理由写进模块文档。
        is_maker = req.ord_type in ("limit", "post_only")
        return self._record(i, req, px, is_maker=is_maker,
                            reason=req.ord_type, coid=oo.cl_ord_id, ref=px)

    def _record(self, i: int, req: OrderRequest, px: float, *, is_maker: bool,
                reason: str, coid: str, ref: float | None = None) -> Fill:
        # ⚠️ **手续费显式由 ``ExecConfig`` 算，不走账户的 MarginConfig。**
        # 理由：本层是"**模拟**的成本模型"，而 ``cost_multiplier`` 属于
        # 模拟假设。不显式传的话 ``cost_multiplier`` 会只影响滑点、
        # 完全不影响手续费——而配置看上去是同时控制两者的。
        # 那种失效是静默的（成本 ×5 后收益几乎没变，看起来像
        # "策略对成本不敏感"）。
        fee = self.config.fee(abs(float(req.sz) * float(px)), is_maker=is_maker)
        res = self.account.apply_fill(
            inst_id=req.inst_id, side=req.side, qty=req.sz, price=px,
            is_maker=is_maker, pos_side=req.pos_side, lever=self.lever,
            fee_override=fee,
        )
        # 滑点成本：买贵了 / 卖便宜了都记正（吃亏为正）。
        # ref 缺省 = px ⇒ 成本 0（限价单成交在自己的价上，没有滑点）。
        ref_px = float(px if ref is None else ref)
        slip_cost = (px - ref_px) * req.sz if req.side == "buy" else \
                    (ref_px - px) * req.sz
        f = Fill(bar_index=int(i), inst_id=req.inst_id, side=req.side,
                 qty=float(req.sz), price=float(px), is_maker=is_maker,
                 fee=float(res["fee"]), reason=reason, cl_ord_id=coid,
                 ref_price=ref_px, slippage_cost=float(slip_cost))
        self.fills.append(f)
        # 开仓成功后把附带的 TP/SL 挂上；反手/平仓则清掉
        if abs(self.account.position(req.inst_id, req.pos_side).qty) > 1e-12:
            if req.attach_algo is not None:
                self.algo = req.attach_algo
        else:
            self.algo = None
        return f

    # -- TP/SL -----------------------------------------------------------
    def _check_algo(self, i: int, h: float, l: float,  # noqa: E741
                    c: float) -> list[Fill]:
        """检查止盈止损。见模块文档「假设 3」。"""
        if self.algo is None:
            return []
        p = self.account.position(self.inst_id)
        if p.is_flat:
            self.algo = None
            return []
        long = p.qty > 0
        a = self.algo
        hit_tp = hit_sl = False
        if long:
            hit_tp = a.tp_trigger_px is not None and h >= a.tp_trigger_px
            hit_sl = a.sl_trigger_px is not None and l <= a.sl_trigger_px
        else:
            hit_tp = a.tp_trigger_px is not None and l <= a.tp_trigger_px
            hit_sl = a.sl_trigger_px is not None and h >= a.sl_trigger_px

        if not (hit_tp or hit_sl):
            return []
        # ⚠️⚠️ **假设 3：两个都碰到时按 SL 处理**（保守）。
        if hit_sl:
            reason, px = "sl", float(a.sl_trigger_px)
        else:
            reason, px = "tp", float(a.tp_trigger_px)
        side = "sell" if long else "buy"
        req = OrderRequest(
            inst_id=self.inst_id, side=side,
            ord_type="market",      # 触发后走市价（与 tp_ord_px=MARKET_ON_TRIGGER 一致）
            sz=abs(p.qty), px=None,
            td_mode="isolated", pos_side=p.pos_side, reduce_only=True,
            lever=self.lever,
        )
        # 触发后按市价成交 ⇒ 不能假设成交在触发价，要加滑点
        slip = self.config.eff_slippage / 10_000.0
        fill_px = px * (1.0 - slip) if side == "sell" else px * (1.0 + slip)
        f = self._record(i, req, fill_px, is_maker=False, reason=reason,
                         coid="", ref=px)
        self.algo = None
        if hit_tp and hit_sl:
            f.reason = "sl"   # 保持显式：这一笔是"同时碰到，按 SL 处理"
        return [f]

    # -- 强平 -------------------------------------------------------------
    def check_liquidation(self, i: int, mark: float) -> Fill | None:
        """账户权益不足以维持时强平。**必须是显式的**——
        没有强平的杠杆回测会给出"买入并持有加杠杆"的假高收益。"""
        marks = {self.inst_id: float(mark)}
        if not self.account.is_liquidated(marks):
            return None
        p = self.account.position(self.inst_id)
        if p.is_flat:
            return None
        mark = float(mark)
        # 强平手续费同样要走 ExecConfig（否则成本倍数在强平样本上失效）
        liq_fee = self.config.fee(p.notional(mark), is_maker=False)
        res = self.account.liquidate(self.inst_id, marks,
                                     pos_side=p.pos_side,
                                     fee_override=liq_fee)
        self.liquidations += 1
        self.algo = None
        self.cancel_all(reason="liquidated")
        f = Fill(bar_index=int(i), inst_id=self.inst_id,
                 side="sell" if p.qty > 0 else "buy",
                 qty=abs(p.qty), price=float(res.get("price", mark)),
                 is_maker=False, fee=float(res.get("fee", 0.0)),
                 reason="liquidate")
        self.fills.append(f)
        return f

    # -- 快照 -------------------------------------------------------------
    def mark(self, bar: Bar) -> float:
        """当根的标记价近似。⚠️ 假设 4：用收盘价。"""
        return float(bar[3])

    def snapshot(self, bar: Bar) -> dict[str, Any]:
        marks = {self.inst_id: self.mark(bar)}
        return {
            "equity": self.account.equity(marks),
            "cash": self.account.cash,
            "margin_ratio": self.account.margin_ratio(marks),
            "position": self.account.position(self.inst_id).to_dict()
            if hasattr(self.account.position(self.inst_id), "to_dict") else {
                "qty": self.account.position(self.inst_id).qty,
                "avg_px": self.account.position(self.inst_id).avg_px,
                "realized_pnl": self.account.position(self.inst_id).realized_pnl,
            },
        }

    # -- 报告用的汇总 ------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        mk = sum(1 for f in self.fills if f.is_maker)
        return {
            "n_submitted": self.submitted,
            "n_fills": len(self.fills),
            "n_maker": mk,
            "n_taker": len(self.fills) - mk,
            "n_open_at_end": len(self.open_orders),
            "n_expired": self.expired_orders,
            "n_liquidations": self.liquidations,
            "total_fees": float(self.account.total_fees),
            "turnover": float(sum(f.notional for f in self.fills)),
        }


def bars_from_series(series: Any, i: int) -> Bar:
    """从 ``marketdb.Series`` 取第 ``i`` 根的 OHLC。"""
    return (float(series.open[i]), float(series.high[i]),
            float(series.low[i]), float(series.close[i]))


__all__ = [
    "Bar",
    "ExecConfig",
    "Fill",
    "OpenOrder",
    "BarExecutor",
    "bars_from_series",
]
