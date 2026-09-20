"""订单簿：价格优先 + 时间优先（施工蓝图 §3.2）。

核心工程决定：**价位用整数 tick 索引做 key，不用 float。**
--------------------------------------------------------------------
原因是浮点数的字典键不可靠：``0.1 + 0.2 != 0.3``。若直接用 float 当价位 key，
数学上同一个价位会被拆成两个不同的层级，价格优先排序随之失效——订单簿会出现
"幽灵价位"：best_bid 显示 100.0，但还有一张 100.00000000000001 的买单排在它前面。
真实市场里这叫价位碎片化，是我们绝不能自己引入的伪影。

做法：``tick_index = round(price / tick_size)``，层级 key 用 int；
对外暴露的成交价/最优价再还原成 ``tick_index * tick_size``。

性能
----
- 价位列表用 ``bisect.insort`` 维护有序性（O(log n) 插入），
  避免蓝图 §5 提醒的"每次插入都全量排序"。
- 同价位内部用 ``deque``，FIFO 即时间优先，入队出队都是 O(1)。
"""

from __future__ import annotations

import bisect
from collections import deque
from typing import Iterator

from .types import EPS, Order, Side


class OrderBook:
    """限价订单簿。

    内部状态
    --------
    ``_bids`` / ``_asks``      : tick_index -> deque[Order]（FIFO = 时间优先）
    ``_bid_ticks`` / ``_ask_ticks`` : 有序 tick 索引列表（均升序）
    ``_index``                 : order_id -> (side, tick_index, Order)
    """

    __slots__ = (
        "tick_size",
        "_bids",
        "_asks",
        "_bid_ticks",
        "_ask_ticks",
        "_index",
        "_n_orders",
    )

    def __init__(self, tick_size: float = 0.01) -> None:
        if tick_size <= 0:
            raise ValueError("tick_size 必须为正")
        self.tick_size = float(tick_size)
        self._bids: dict[int, deque[Order]] = {}
        self._asks: dict[int, deque[Order]] = {}
        self._bid_ticks: list[int] = []  # 升序，最优买价 = 末位
        self._ask_ticks: list[int] = []  # 升序，最优卖价 = 首位
        self._index: dict[str, tuple[Side, int, Order]] = {}
        self._n_orders = 0

    # --- 价格 <-> tick 索引 --------------------------------------------
    def to_tick(self, price: float) -> int:
        return int(round(price / self.tick_size))

    def price_of(self, tick: int) -> float:
        return tick * self.tick_size

    # --- 内部工具 -------------------------------------------------------
    def _side_refs(
        self, side: Side
    ) -> tuple[dict[int, deque[Order]], list[int]]:
        if side == "buy":
            return self._bids, self._bid_ticks
        return self._asks, self._ask_ticks

    # --- 查询 -----------------------------------------------------------
    @property
    def n_orders(self) -> int:
        return self._n_orders

    def __len__(self) -> int:
        return self._n_orders

    def best_tick(self, side: Side) -> int | None:
        _, ticks = self._side_refs(side)
        if not ticks:
            return None
        return ticks[-1] if side == "buy" else ticks[0]

    def best_bid(self) -> float | None:
        t = self.best_tick("buy")
        return None if t is None else self.price_of(t)

    def best_ask(self) -> float | None:
        t = self.best_tick("sell")
        return None if t is None else self.price_of(t)

    def spread(self) -> float | None:
        bb, ba = self.best_bid(), self.best_ask()
        if bb is None or ba is None:
            return None
        return ba - bb

    def mid_price(self) -> float | None:
        bb, ba = self.best_bid(), self.best_ask()
        if bb is None or ba is None:
            return None
        return 0.5 * (bb + ba)

    def level(self, side: Side, tick: int) -> deque[Order] | None:
        levels, _ = self._side_refs(side)
        return levels.get(tick)

    def get_order(self, order_id: str) -> Order | None:
        rec = self._index.get(order_id)
        return None if rec is None else rec[2]

    def has_order(self, order_id: str) -> bool:
        return order_id in self._index

    def open_order_ids(self) -> set[str]:
        """当前全部挂单的订单号集合（供测试与诊断使用）。"""
        return set(self._index)

    def iter_side(self, side: Side, best_first: bool = True) -> Iterator[tuple[float, Order]]:
        """按价格优先顺序遍历某一侧的全部挂单。"""
        levels, ticks = self._side_refs(side)
        ordered = list(reversed(ticks)) if (side == "buy") == best_first else list(ticks)
        for t in ordered:
            p = self.price_of(t)
            for o in levels[t]:
                yield p, o

    def depth(self, side: Side, n: int = 10) -> list[tuple[float, float, int]]:
        """前 n 档深度：[(price, total_qty, n_orders), ...] 按价格优先排列。"""
        levels, ticks = self._side_refs(side)
        if side == "buy":
            chosen = list(reversed(ticks))[:n]
        else:
            chosen = ticks[:n]
        out: list[tuple[float, float, int]] = []
        for t in chosen:
            q = levels[t]
            out.append((self.price_of(t), sum(o.remaining or 0.0 for o in q), len(q)))
        return out

    def total_quantity(self, side: Side) -> float:
        levels, _ = self._side_refs(side)
        return float(
            sum(o.remaining or 0.0 for q in levels.values() for o in q)
        )

    def n_levels(self, side: Side) -> int:
        _, ticks = self._side_refs(side)
        return len(ticks)

    # --- 增删 -----------------------------------------------------------
    def add(self, order: Order) -> None:
        """挂单。同价位追加到队尾（时间优先）。"""
        if order.order_type == "market":
            raise ValueError("市价单不能挂入订单簿")
        if order.remaining is None or order.remaining <= EPS:
            raise ValueError("剩余量为 0 的订单不能挂入订单簿")
        tick = self.to_tick(order.price)
        levels, ticks = self._side_refs(order.side)
        level = levels.get(tick)
        if level is None:
            level = deque()
            levels[tick] = level
            bisect.insort(ticks, tick)
        level.append(order)
        self._index[order.order_id] = (order.side, tick, order)
        self._n_orders += 1

    def remove(self, order: Order) -> bool:
        """摘除指定订单（完全成交或撤单）。"""
        rec = self._index.pop(order.order_id, None)
        if rec is None:
            return False
        side, tick, _ = rec
        levels, ticks = self._side_refs(side)
        level = levels.get(tick)
        if level is not None:
            # 用身份比较（is）而不是 == ：dataclass 的 == 是值比较，
            # 理论上可能误删一张字段完全相同的另一张单。
            for i, o in enumerate(level):
                if o is order:
                    del level[i]
                    break
            if not level:
                del levels[tick]
                j = bisect.bisect_left(ticks, tick)
                if j < len(ticks) and ticks[j] == tick:
                    ticks.pop(j)
        self._n_orders -= 1
        return True

    def cancel_order(self, order_id: str) -> Order | None:
        """按订单号撤单，返回被撤的订单（不存在则返回 None）。"""
        rec = self._index.get(order_id)
        if rec is None:
            return None
        self.remove(rec[2])
        return rec[2]

    def cancel_all(self, agent_id: str | None = None) -> list[Order]:
        """撤销某 agent 的全部挂单（agent_id=None 表示清空整个簿）。"""
        victims = [
            rec[2]
            for rec in self._index.values()
            if agent_id is None or rec[2].agent_id == agent_id
        ]
        for o in victims:
            self.remove(o)
        return victims

    # --- 供撮合引擎使用 -------------------------------------------------
    def estimate_fill(
        self, side: Side, quantity: float, limit_price: float | None = None
    ) -> tuple[float, float]:
        """按当前簿面估算吃单能力：返回 ``(可成交量, 总名义额)``。

        用途有两个：
          ① 市价买单的**资金约束**。市价单要扫多档，平均成交价高于最优价，
             若按最优价算预算，扫单时会透支现金。
          ② 冲击成本分析（一笔单吃穿几档、滑点多少）。

        ``side`` 是**吃单方**的方向；内部扫描的是**对手方**的簿面。

        ⚠️ 这个方向曾经写反，是个静默 bug。原实现直接对 ``side`` 取
        ``_side_refs``，于是"市价买单能不能成交"变成了"**买盘上有没有挂单**"——
        两者毫不相干。后果：**买盘被打空时，市价买单会被直接拒掉，
        而卖盘明明满着**（实测：买深 0.00、卖深 15.79 时，
        ``estimate_fill("buy", 5)`` 返回 ``(0.0, 0.0)``）。
        一期没暴露，因为一期唯一的市价单场合（强制平仓）都是**卖出**，
        而卖单走的是另一条不调用本方法的裁剪路径。
        它一直在悄悄压制策略实验台里动量/随机策略的成交笔数。
        """
        # 吃单方买 → 对手方是卖盘；吃单方卖 → 对手方是买盘。
        book_side: Side = "sell" if side == "buy" else "buy"
        levels, ticks = self._side_refs(book_side)
        # 买单从最低卖价往上吃；卖单从最高买价往下吃。
        ordered = list(ticks) if side == "buy" else list(reversed(ticks))
        left = quantity
        qty = 0.0
        cost = 0.0
        for t in ordered:
            p = self.price_of(t)
            if limit_price is not None:
                if side == "buy" and p > limit_price:
                    break
                if side == "sell" and p < limit_price:
                    break
            for o in levels[t]:
                rem = o.remaining or 0.0
                if rem <= EPS:
                    continue
                take = rem if rem < left else left
                qty += take
                cost += take * p
                left -= take
                if left <= EPS:
                    break
            if left <= EPS:
                break
        return qty, cost

    def find_eligible(
        self, side: Side, exclude_agent: str | None = None
    ) -> tuple[Order, int] | None:
        """返回该侧价格最优的**可成交**挂单及其 tick 索引。

        价格优先：买侧从最高价往下、卖侧从最低价往上逐档扫描。
        若某档全是 ``exclude_agent`` 自家挂单，则跳过该档继续往下——
        这是自成交防护。代价是可能出现"放着自家更优价的挂单不成交、
        却去和别人在更差价位成交"，真实交易所的 cancel-oldest 策略更激进，
        这里选择保守做法：**不撤单，只跳过**。
        """
        levels, ticks = self._side_refs(side)
        if side == "buy":
            candidates = reversed(ticks)
        else:
            candidates = iter(ticks)
        for t in candidates:
            level = levels[t]
            for o in level:
                if exclude_agent is None or o.agent_id != exclude_agent:
                    return o, t
        return None

    # --- 诊断 -----------------------------------------------------------
    def invariants_ok(self) -> tuple[bool, list[str]]:
        """自检：订单簿内部状态是否自洽。

        这是给测试用的"探针"——撮合若有 bug，最先崩的往往是这里的不变量：
        索引与队列不一致、空档位残留、买卖价穿越（应该被撮合掉却还挂着）。
        """
        problems: list[str] = []
        seen: set[str] = set()
        for side in ("buy", "sell"):
            levels, ticks = self._side_refs(side)
            if sorted(ticks) != ticks:
                problems.append(f"{side} 价位列表未保持有序")
            if len(set(ticks)) != len(ticks):
                problems.append(f"{side} 价位列表存在重复档位")
            for t in ticks:
                if t not in levels:
                    problems.append(f"{side} 价位列表含空洞档位 {t}")
                    continue
                lvl = levels[t]
                if not lvl:
                    problems.append(f"{side} 存在空队列档位 {t}")
                for o in lvl:
                    if o.side != side:
                        problems.append(f"订单 {o.order_id} 的 side 与所在档位不符")
                    if self.to_tick(o.price) != t:
                        problems.append(f"订单 {o.order_id} 价格与档位不符")
                    if (o.remaining or 0.0) <= EPS:
                        problems.append(f"订单 {o.order_id} 剩余量为 0 却仍在簿上")
                    if o.order_id in seen:
                        problems.append(f"订单 {o.order_id} 重复出现")
                    seen.add(o.order_id)
                    rec = self._index.get(o.order_id)
                    if rec is None:
                        problems.append(f"订单 {o.order_id} 未登记在索引中")
                    elif rec[0] != side or rec[1] != t:
                        problems.append(f"订单 {o.order_id} 索引与所在档位不一致")
        for oid in self._index:
            if oid not in seen:
                problems.append(f"索引中的订单 {oid} 不在任何档位里")
        if len(seen) != self._n_orders:
            problems.append(
                f"计数器 _n_orders={self._n_orders} 与实际挂单数 {len(seen)} 不符"
            )
        # 买卖价穿越检查。
        # 注意：不能简单断言 best_bid < best_ask。自成交防护（见 find_eligible）会
        # 故意跳过自家挂单，做市商同时挂买卖两边时，簿面上可能出现"自己的买价 >=
        # 自己的卖价"——这不是引擎 bug，是防护策略的已知代价。
        # 真正该报错的只有一种情况：最优档上存在**不同 agent** 的买卖双方却仍未成交
        # （意味着一笔本可成交的交易被漏掉了）。
        bb, ba = self.best_bid(), self.best_ask()
        if bb is not None and ba is not None and bb >= ba:
            bid_agents = {o.agent_id for o in self._bids[self.best_tick("buy")]}
            ask_agents = {o.agent_id for o in self._asks[self.best_tick("sell")]}
            if bid_agents != ask_agents:
                problems.append(
                    f"买卖价穿越未撮合: best_bid={bb}(来自{sorted(bid_agents)}) "
                    f">= best_ask={ba}(来自{sorted(ask_agents)})"
                )
        return (not problems), problems
