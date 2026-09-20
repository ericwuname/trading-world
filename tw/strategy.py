"""策略 API：写一个可测策略只需要实现一个方法。

    from tw.strategy import Strategy

    class MyMaker(Strategy):
        def on_tick(self, ctx):
            if ctx.spread is None:
                return None
            half = ctx.spread_bp * 0.4          # 想赚半个价差的 40%
            return ctx.quote(
                bid=ctx.mid * (1 - half / 1e4),
                ask=ctx.mid * (1 + half / 1e4),
                qty=2.0,
            )

然后交给实验台跑多种子、多场景、和一组对照策略比：

    python scripts/lab.py --strategy strategies.mm_naive:NaiveMaker

设计上刻意做了三件事
--------------------
**1. 只暴露"能用的东西"。** ``ctx`` 里没有 ``market`` 对象——拿不到订单簿内部、
拿不到别人的账户。策略只能通过"看公开行情 + 下单"参与，和真实交易者一致。
（想深入研究的可以自己拿 ``self.market``，但那是"研究模式"，不是默认路径。）

**2. 下单工具全部走 Agent.new_order**，因此自动享有预留制记账与 tick 网格吸附。
绕过它直接构造 ``Order`` 会破坏"不能让主体凭空造钱"这条不变量。

**3. ``on_tick`` 可以返回 ``None`` / 一个订单 / 一个订单列表**，
三种写法都支持——做市商必须同时挂双边，只允许返回一个订单会逼它绕过记账层。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .agents.base import Agent
from .eval import Fill
from .types import MarketState, Order, Side


class StrategyContext:
    """每个 tick 交给策略的只读快照 + 下单工具。

    所有属性都是**当 tick 的实时值**：市场快照是就地刷新的，
    因此同一 tick 内后出手的策略看到的是前面策略已经改变过的盘口
    （与真实市场一致：你的单挂出去，别人马上就能吃到）。
    """

    __slots__ = ("_a", "_s", "started")

    def __init__(self, agent: "Strategy", state: MarketState) -> None:
        self._a = agent
        self._s = state
        self.started = agent._started

    # ---- 市场 ----------------------------------------------------------
    @property
    def tick(self) -> int:
        return self._s.tick

    @property
    def mid(self) -> float | None:
        m = self._s.mid
        return float(m) if m and m > 0 else None

    @property
    def best_bid(self) -> float | None:
        return self._s.best_bid

    @property
    def best_ask(self) -> float | None:
        return self._s.best_ask

    @property
    def spread(self) -> float | None:
        """绝对价差（价格单位）。"""
        return self._s.spread

    @property
    def spread_bp(self) -> float:
        """价差（bp）。盘口单边空缺时返回 nan。"""
        m, sp = self._s.mid, self._s.spread
        if not m or not sp:
            return float("nan")
        return float(sp / m * 1e4)

    @property
    def fundamental(self) -> float:
        return float(self._s.fundamental)

    @property
    def last_price(self) -> float | None:
        p = self._s.last_price
        return float(p) if p else None

    def mid_lag(self, n: int) -> float | None:
        """n 个 tick 之前的中间价。"""
        return self._s.history.lag(n)

    def momentum(self, n: int = 20) -> float | None:
        """过去 n tick 的动量（中间价口径）。

        用中间价而不是成交价：成交价会被买卖价差跳动污染，
        用它算动量会把"价差来回跳"读成趋势。
        """
        return self._s.momentum(n)

    def mid_std(self, n: int = 100) -> float | None:
        """近期中间价变动幅度（bp），做市商用来调价差宽度。"""
        v = self._s.history.std(n)
        if v is None or self._s.mid is None:
            return None
        return float(v / self._s.mid * 1e4)

    # ---- 账户 ----------------------------------------------------------
    @property
    def cash(self) -> float:
        return float(self._a.cash)

    @property
    def inventory(self) -> float:
        return float(self._a.inventory)

    @property
    def available_cash(self) -> float:
        return float(self._a.available_cash)

    @property
    def available_inventory(self) -> float:
        return float(self._a.available_inventory)

    @property
    def open_orders(self) -> tuple[Order, ...]:
        return tuple(self._a.open_orders)

    @property
    def n_open_orders(self) -> int:
        return len(self._a.open_orders)

    def equity(self) -> float:
        return float(self._a.equity(self._s.mid))

    def pnl(self) -> float:
        return float(self._a.equity_change(self._s.mid))

    # ---- 下单 ----------------------------------------------------------
    def buy(self, price: float, qty: float) -> Order | None:
        return self._a.new_order(self._s, "buy", price, self._clip_buy(qty, price))

    def sell(self, price: float, qty: float) -> Order | None:
        return self._a.new_order(self._s, "sell", price, self._clip_sell(qty))

    def market_buy(self, qty: float) -> Order | None:
        return self._a.new_order(self._s, "buy", 0.0, qty, order_type="market")

    def market_sell(self, qty: float) -> Order | None:
        return self._a.new_order(self._s, "sell", 0.0, qty, order_type="market")

    def limit_at(self, side: Side, offset_bp: float, qty: float) -> Order | None:
        """在"中间价 ± offset_bp"处挂一张限价单。"""
        m = self.mid
        if m is None:
            return None
        px = m * (1.0 - offset_bp / 1e4) if side == "buy" else m * (1.0 + offset_bp / 1e4)
        return self.buy(px, qty) if side == "buy" else self.sell(px, qty)

    def quote(self, bid: float, ask: float, qty: float) -> list[Order]:
        """同时挂双边（做市商的常用动作）。返回实际生效的订单列表。"""
        out: list[Order] = []
        for o in (self.buy(bid, qty), self.sell(ask, qty)):
            if o is not None:
                out.append(o)
        return out

    def cancel_all(self) -> int:
        return self._a.cancel_all()

    # ---- 内部：把订单裁到"可用额度"以内 --------------------------------
    # 刻意不在下单失败时静默返回 None，而是**先裁量**：
    # 直接拒单会让策略在资金紧张时完全停摆，而真相是"它能做小一点"。
    def _clip_buy(self, qty: float, price: float) -> float:
        if not np.isfinite(qty) or qty <= 0 or price <= 0:
            return 0.0
        cap = self._a.available_cash / price
        return float(min(qty, max(cap, 0.0)))

    def _clip_sell(self, qty: float) -> float:
        if not np.isfinite(qty) or qty <= 0:
            return 0.0
        return float(min(qty, max(self._a.available_inventory, 0.0)))

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"StrategyContext(t={self.tick}, mid={self.mid}, "
            f"inv={self.inventory:.2f}, cash={self.cash:,.0f})"
        )


class Strategy(Agent):
    """用户策略基类。子类只需实现 :meth:`on_tick`。

    与内置主体（零智能/基本面派/图表派/做市商）完全平等：
    同样受预留制记账、tick 网格、挂单上限、超时撤单约束，
    同样只通过 ``Market.submit`` 参与撮合。
    """

    KIND = "strategy"

    def __init__(
        self,
        agent_id: str,
        cash: float,
        inventory: float,
        rng: np.random.Generator,
        max_open_orders: int = 20,
        order_ttl: int = 500,
        **kwargs,
    ) -> None:
        super().__init__(
            agent_id, cash, inventory, rng,
            max_open_orders=max_open_orders, order_ttl=order_ttl,
        )
        self._started = False
        #: 策略抛异常的累计次数与首个异常。不静默吞掉——实验台会把它们
        #: 汇总到结果里，否则「策略什么都没做」会被误读成「策略判断不交易」。
        self.n_errors = 0
        self.first_error: str | None = None

    # --- 用户要实现的部分 -----------------------------------------------
    def on_tick(self, ctx: StrategyContext) -> Order | Sequence[Order] | None:
        """每个 tick 的决策。返回 None / 一个订单 / 一组订单。"""
        raise NotImplementedError(
            f"{self.__class__.__name__} 必须实现 on_tick(ctx)"
        )

    def on_start(self, ctx: StrategyContext) -> None:
        """第一次决策前调用一次。可用来做参数校验、打印、建缓冲区。"""

    def on_fill(self, fill: Fill) -> None:
        """自己的订单成交时调用。做执行类策略通常需要在这里推进"已成交进度"。"""

    def on_end(self, ctx: StrategyContext) -> None:
        """模拟结束时调用一次。"""

    # --- 框架部分（子类不需要覆盖）--------------------------------------
    def decide(self, state: MarketState) -> Order | Sequence[Order] | None:
        ctx = StrategyContext(self, state)
        if not self._started:
            self._started = True
            self.on_start(ctx)
        try:
            return self.on_tick(ctx)
        except Exception as exc:  # noqa: BLE001
            # 策略抛异常时不能把整个模拟带崩——那会浪费几十秒的算力。
            # 记下来、本 tick 不动作，让实验跑完再由实验台汇总报错。
            self.n_errors += 1
            if self.first_error is None:
                self.first_error = f"{type(exc).__name__}: {exc}"
            return None

    def on_trade(self, trade, is_buyer: bool) -> None:
        super().on_trade(trade, is_buyer)
        m = getattr(trade, "mid_at_fill", float("nan"))
        self.on_fill(
            Fill(
                tick=int(trade.tick),
                side="buy" if is_buyer else "sell",
                price=float(trade.price),
                quantity=float(trade.quantity),
                mid=float(m) if m and np.isfinite(m) else float("nan"),
                is_maker=(trade.aggressor_side == ("sell" if is_buyer else "buy")),
                counterparty=(
                    trade.sell_agent_id if is_buyer else trade.buy_agent_id
                ),
            )
        )

    def describe(self) -> dict:
        d = super().describe()
        d["n_errors"] = self.n_errors
        d["first_error"] = self.first_error or ""
        return d


def make_strategy(
    cls: type[Strategy],
    agent_id: str,
    *,
    seed: int,
    cash: float,
    inventory: float,
    **kwargs,
) -> Strategy:
    """按实验台的约定造一个策略主体。

    随机数流用 ``[seed, ENTROPY_STRATEGY, hash(id)]`` 独立播种，
    与背景主体的流完全隔开——这样换策略、改策略都不会扰动背景行情。
    """
    rng = np.random.default_rng([seed, ENTROPY_STRATEGY, _stable_key(agent_id)])
    return cls(agent_id, cash, inventory, rng, **kwargs)


#: 策略的随机流标签。与 ENTROPY_AGENT / ENTROPY_ENDOW / ENTROPY_ORDER 区分开。
ENTROPY_STRATEGY = 0x57A7E

_CACHE: dict[str, int] = {}


def _stable_key(s: str) -> int:
    """跨进程稳定的字符串 → 整数映射。

    不能用内置 ``hash()``：Python 的字符串哈希带进程级随机盐，
    同一个策略名在不同进程里会得到不同的种子，实验就无法复现。
    """
    v = _CACHE.get(s)
    if v is None:
        import zlib

        v = zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF
        _CACHE[s] = v
    return v
