"""撮合引擎：连续双向拍卖（施工蓝图 §3.3）。

规则（严格照蓝图）
------------------
1. 新订单进来，若与对手方最优价**穿越**（买价 >= 最优卖价）则成交。
2. **成交价取被动方（先挂单方）的价格**，不是主动方的报价。
   这是蓝图明确要求的，也是真实 CDA 市场的做法：
   主动方接受既有报价，因此是"价格接受者"。
3. 成交量取双方剩余量的较小值。
4. 未完全成交的**限价单**剩余部分挂入订单簿；市价单剩余部分直接丢弃。
5. 每笔成交记录 ``(tick, price, quantity, buy_agent_id, sell_agent_id)``。

自成交防护
----------
同一 agent 既挂买单又挂卖单（做市商必然如此）时，若不处理会出现自己和自己成交。
本引擎默认**跳过**自家挂单而非撤单，详见 ``OrderBook.find_eligible`` 的说明。
"""

from __future__ import annotations

from .book import OrderBook
from .types import EPS, Order, Trade


class MatchingEngine:
    """把订单送进订单簿并产出成交明细。"""

    __slots__ = ("book", "allow_self_trade", "_trade_sink")

    def __init__(
        self,
        book: OrderBook,
        allow_self_trade: bool = False,
        trade_sink: list[Trade] | None = None,
    ) -> None:
        """``trade_sink`` 若给出，成交会直接追加进去，省掉每 tick 的列表拼接。"""
        self.book = book
        self.allow_self_trade = allow_self_trade
        self._trade_sink = trade_sink

    # ------------------------------------------------------------------
    def submit(self, order: Order) -> list[Trade]:
        """提交一笔订单，返回它产生的全部成交（可能跨多个价位）。

        注意：``order.remaining`` 会被就地修改；限价单的剩余部分会挂入订单簿。
        """
        trades: list[Trade] = []
        if order.side == "buy":
            self._match(order, "sell", trades)
        else:
            self._match(order, "buy", trades)
        if order.remaining > EPS and order.order_type == "limit":
            self.book.add(order)
        return trades

    # ------------------------------------------------------------------
    def _match(self, taker: Order, book_side: str, trades: list[Trade]) -> None:
        """让 taker 去吃掉 ``book_side`` 一侧的挂单。"""
        book = self.book
        taker_is_buy = book_side == "sell"
        exclude = None if self.allow_self_trade else taker.agent_id

        while taker.remaining > EPS:
            found = book.find_eligible(book_side, exclude)
            if found is None:
                break
            maker, maker_tick = found
            maker_price = book.price_of(maker_tick)

            # --- 限价单的穿越判断 ---------------------------------------
            if taker.order_type == "limit":
                if taker_is_buy and taker.price < maker_price:
                    break
                if (not taker_is_buy) and taker.price > maker_price:
                    break

            # --- 成交量 -------------------------------------------------
            if maker.remaining is None or maker.remaining <= EPS:
                # 防御：簿上不该有零剩余量的订单。真出现说明别处有 bug，
                # 这里清掉并继续，否则 find_eligible 会一直返回它 → 死循环。
                book.remove(maker)
                continue

            qty = taker.remaining if taker.remaining < maker.remaining else maker.remaining
            price = maker_price  # ← 被动方价格优先（蓝图 §3.3 第 1 条）

            if taker_is_buy:
                buy_agent, sell_agent = taker.agent_id, maker.agent_id
                buy_oid, sell_oid = taker.order_id, maker.order_id
            else:
                buy_agent, sell_agent = maker.agent_id, taker.agent_id
                buy_oid, sell_oid = maker.order_id, taker.order_id

            trade = Trade(
                tick=taker.timestamp,
                price=price,
                quantity=qty,
                buy_agent_id=buy_agent,
                sell_agent_id=sell_agent,
                aggressor_side=taker.side,
                buy_order_id=buy_oid,
                sell_order_id=sell_oid,
            )
            trades.append(trade)
            if self._trade_sink is not None:
                self._trade_sink.append(trade)

            taker.remaining -= qty
            maker.remaining -= qty
            if maker.remaining <= EPS:
                book.remove(maker)
            # 循环继续 → 可能吃到下一档，这就是"扫单"（sweep）

    # ------------------------------------------------------------------
    def cancel(self, order: Order) -> bool:
        return self.book.remove(order)
