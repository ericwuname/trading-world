"""撮合内核单元测试（施工蓝图 §6 第 1 项）。

设计原则：**每个断言都必须能失败。**
这里不用 pytest，只用标准库 unittest —— 少一个依赖，少一个"环境不对所以没跑"的借口。
变异验证见 ``scripts/mutation_check.py``：它会把撮合引擎故意改坏，验证本测试套件
确实会变红。

价格约定：除特别说明外，测试统一用 ``tick_size=1.0``、基准价 100，
让 "100.0" 是干净的一档，避免浮点噪声干扰对"价格优先"的观察。
"""

from __future__ import annotations

import math
import random
import unittest

from tw.book import OrderBook
from tw.engine import MatchingEngine
from tw.types import EPS, Order, Trade


def mk(
    agent: str,
    side: str,
    price: float,
    qty: float,
    tick: int = 0,
    oid: str | None = None,
    otype: str = "limit",
) -> Order:
    """构造订单的工厂函数。oid 省略时按 (agent, side, 序号) 自动生成。"""
    if oid is None:
        oid = f"{agent}-{side[0]}-{price:g}-{qty:g}-{tick}"
    return Order(
        agent_id=agent,
        side=side,
        price=price,
        quantity=qty,
        timestamp=tick,
        order_id=oid,
        order_type=otype,
    )


def mk_market(agent: str, side: str, qty: float, tick: int = 0, oid: str | None = None) -> Order:
    price = math.inf if side == "buy" else -math.inf
    return mk(agent, side, price, qty, tick, oid or f"{agent}-mkt-{side}", "market")


class TestOrderValidation(unittest.TestCase):
    def test_数量必须为正(self) -> None:
        with self.assertRaises(ValueError):
            mk("A", "buy", 100.0, 0.0)
        with self.assertRaises(ValueError):
            mk("A", "buy", 100.0, -1.0)

    def test_方向必须是买卖(self) -> None:
        with self.assertRaises(ValueError):
            mk("A", "long", 100.0, 1.0)

    def test_限价单价格必须有限(self) -> None:
        with self.assertRaises(ValueError):
            mk("A", "buy", math.inf, 1.0, otype="limit")

    def test_市价单允许无穷价格(self) -> None:
        o = mk_market("A", "buy", 1.0)
        self.assertTrue(math.isinf(o.price))

    def test_初始剩余量等于数量(self) -> None:
        o = mk("A", "buy", 100.0, 7.0)
        self.assertAlmostEqual(o.remaining, 7.0)
        self.assertFalse(o.is_filled)

    def test_剩余量不得超过数量(self) -> None:
        with self.assertRaises(ValueError):
            Order("A", "buy", 100.0, 1.0, 0, "x", "limit", remaining=2.0)


class TestOrderBook(unittest.TestCase):
    def setUp(self) -> None:
        self.book = OrderBook(tick_size=1.0)

    def test_最优买卖价与中点(self) -> None:
        self.book.add(mk("A", "buy", 99.0, 1.0, oid="b99"))
        self.book.add(mk("A", "buy", 100.0, 1.0, oid="b100"))
        self.book.add(mk("A", "buy", 98.0, 1.0, oid="b98"))
        self.book.add(mk("B", "sell", 102.0, 1.0, oid="s102"))
        self.book.add(mk("B", "sell", 101.0, 1.0, oid="s101"))
        self.assertEqual(self.book.best_bid(), 100.0)
        self.assertEqual(self.book.best_ask(), 101.0)
        self.assertEqual(self.book.mid_price(), 100.5)
        self.assertEqual(self.book.spread(), 1.0)

    def test_空簿返回None(self) -> None:
        self.assertIsNone(self.book.best_bid())
        self.assertIsNone(self.book.best_ask())
        self.assertIsNone(self.book.mid_price())
        self.assertIsNone(self.book.spread())
        self.assertIsNone(self.book.best_tick("buy"))

    def test_同价位按时间优先_FIFO(self) -> None:
        o1 = mk("A", "sell", 101.0, 1.0, tick=0, oid="first")
        o2 = mk("B", "sell", 101.0, 1.0, tick=1, oid="second")
        o3 = mk("C", "sell", 101.0, 1.0, tick=2, oid="third")
        for o in (o1, o2, o3):
            self.book.add(o)
        order_in_level = [o.order_id for o in self.book.level("sell", 101)]
        self.assertEqual(order_in_level, ["first", "second", "third"])
        self.assertIs(self.book.find_eligible("sell")[0], o1)

    def test_浮点价位不应碎片化(self) -> None:
        """0.1 + 0.2 != 0.3 —— 若用 float 直接当价位 key，同一档会被拆成两档。"""
        book = OrderBook(tick_size=0.01)
        book.add(mk("A", "buy", 0.1 + 0.2, 1.0, oid="f1"))
        book.add(mk("B", "buy", 0.3, 1.0, oid="f2"))
        self.assertEqual(book.n_levels("buy"), 1, "同一价位被拆成了多个档位")
        self.assertEqual(len(book.level("buy", 30)), 2)
        self.assertEqual(book.best_bid(), 0.3)

    def test_撤单(self) -> None:
        self.book.add(mk("A", "buy", 100.0, 1.0, oid="x"))
        self.assertEqual(self.book.n_orders, 1)
        got = self.book.cancel_order("x")
        self.assertIsNotNone(got)
        self.assertEqual(self.book.n_orders, 0)
        self.assertEqual(self.book.n_levels("buy"), 0)
        self.assertIsNone(self.book.best_bid())
        self.assertIsNone(self.book.cancel_order("不存在"))

    def test_撤单后档位清理不留空洞(self) -> None:
        for i, p in enumerate([98.0, 99.0, 100.0]):
            self.book.add(mk("A", "buy", p, 1.0, oid=f"o{i}"))
        self.book.cancel_order("o1")  # 撤掉中间的 99.0
        ok, probs = self.book.invariants_ok()
        self.assertTrue(ok, probs)
        self.assertEqual(self.book.best_bid(), 100.0)

    def test_撤掉全部挂单(self) -> None:
        for i in range(5):
            self.book.add(mk("A", "buy", 100.0 + i, 1.0, oid=f"a{i}"))
            self.book.add(mk("B", "sell", 200.0 + i, 1.0, oid=f"b{i}"))
        removed = self.book.cancel_all("A")
        self.assertEqual(len(removed), 5)
        self.assertEqual(self.book.n_levels("buy"), 0)
        self.assertEqual(self.book.n_orders, 5)
        ok, probs = self.book.invariants_ok()
        self.assertTrue(ok, probs)

    def test_市价单不能挂簿(self) -> None:
        with self.assertRaises(ValueError):
            self.book.add(mk_market("A", "buy", 1.0))

    def test_零剩余量不能挂簿(self) -> None:
        o = mk("A", "buy", 100.0, 1.0, oid="z")
        o.remaining = 0.0
        with self.assertRaises(ValueError):
            self.book.add(o)

    def test_深度统计(self) -> None:
        self.book.add(mk("A", "buy", 100.0, 2.0, oid="d1"))
        self.book.add(mk("B", "buy", 100.0, 3.0, oid="d2"))
        self.book.add(mk("C", "buy", 99.0, 1.0, oid="d3"))
        depth = self.book.depth("buy", n=5)
        self.assertEqual(len(depth), 2)
        self.assertEqual(depth[0][0], 100.0)
        self.assertAlmostEqual(depth[0][1], 5.0)
        self.assertEqual(depth[0][2], 2, "同档应记录两张挂单")
        self.assertAlmostEqual(self.book.total_quantity("buy"), 6.0)


class TestMatchingPricePriority(unittest.TestCase):
    """成交价必须取**被动方**价格（蓝图 §3.3 第 1 条）。"""

    def setUp(self) -> None:
        self.book = OrderBook(tick_size=1.0)
        self.engine = MatchingEngine(self.book)

    def test_买单主动_成交价取挂单的卖价(self) -> None:
        self.engine.submit(mk("maker", "sell", 100.0, 5.0, tick=0, oid="s1"))
        trades = self.engine.submit(mk("taker", "buy", 103.0, 5.0, tick=1, oid="b1"))
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0].price, 100.0, "成交价取被动方价格，不应是主动方的 103")
        self.assertEqual(trades[0].quantity, 5.0)
        self.assertEqual(trades[0].buy_agent_id, "taker")
        self.assertEqual(trades[0].sell_agent_id, "maker")
        self.assertEqual(trades[0].aggressor_side, "buy")

    def test_卖单主动_成交价取挂单的买价(self) -> None:
        self.engine.submit(mk("maker", "buy", 100.0, 5.0, tick=0, oid="b1"))
        trades = self.engine.submit(mk("taker", "sell", 97.0, 5.0, tick=1, oid="s1"))
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0].price, 100.0, "成交价取被动方价格，不应是主动方的 97")
        self.assertEqual(trades[0].aggressor_side, "sell")

    def test_恰好平价应成交(self) -> None:
        self.engine.submit(mk("A", "sell", 100.0, 1.0, oid="s"))
        trades = self.engine.submit(mk("B", "buy", 100.0, 1.0, oid="b"))
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0].price, 100.0)

    def test_不穿越则不成交(self) -> None:
        self.engine.submit(mk("A", "sell", 101.0, 1.0, oid="s"))
        trades = self.engine.submit(mk("B", "buy", 100.0, 1.0, oid="b"))
        self.assertEqual(trades, [])
        self.assertEqual(self.book.best_ask(), 101.0)
        self.assertEqual(self.book.best_bid(), 100.0)
        ok, probs = self.book.invariants_ok()
        self.assertTrue(ok, probs)

    def test_部分成交_剩余挂入簿中(self) -> None:
        self.engine.submit(mk("A", "sell", 100.0, 5.0, oid="s"))
        trades = self.engine.submit(mk("B", "buy", 100.0, 2.0, oid="b"))
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0].quantity, 2.0)
        self.assertEqual(self.book.best_ask(), 100.0)
        self.assertAlmostEqual(self.book.total_quantity("sell"), 3.0)
        self.assertIsNone(self.book.best_bid(), "主动方已完全成交，不该留下买单")
        ok, probs = self.book.invariants_ok()
        self.assertTrue(ok, probs)

    def test_主动方剩余挂入簿中(self) -> None:
        self.engine.submit(mk("A", "sell", 100.0, 2.0, oid="s"))
        trades = self.engine.submit(mk("B", "buy", 100.0, 5.0, oid="b"))
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0].quantity, 2.0)
        rest = self.book.get_order("b")
        self.assertIsNotNone(rest)
        self.assertAlmostEqual(rest.remaining, 3.0)
        self.assertEqual(self.book.best_bid(), 100.0)
        ok, probs = self.book.invariants_ok()
        self.assertTrue(ok, probs)

    def test_完全成交的挂单应从簿上移除(self) -> None:
        self.engine.submit(mk("A", "sell", 100.0, 5.0, oid="s"))
        self.engine.submit(mk("B", "buy", 100.0, 5.0, oid="b"))
        self.assertEqual(self.book.n_orders, 0)
        self.assertFalse(self.book.has_order("s"))
        self.assertEqual(self.book.n_levels("sell"), 0)


class TestMatchingSweep(unittest.TestCase):
    """扫单：一张大单吃掉多档。"""

    def setUp(self) -> None:
        self.book = OrderBook(tick_size=1.0)
        self.engine = MatchingEngine(self.book)
        # 卖侧三档：100×1、101×2、102×5
        self.engine.submit(mk("A", "sell", 100.0, 1.0, tick=0, oid="a100"))
        self.engine.submit(mk("B", "sell", 101.0, 2.0, tick=0, oid="b101"))
        self.engine.submit(mk("C", "sell", 102.0, 5.0, tick=0, oid="c102"))

    def test_市价扫单吃穿多档且每档各按其价(self) -> None:
        trades = self.engine.submit(mk_market("HUNGRY", "buy", 4.0, tick=1, oid="m1"))
        self.assertEqual(len(trades), 3)
        self.assertEqual([t.price for t in trades], [100.0, 101.0, 102.0])
        self.assertEqual([t.quantity for t in trades], [1.0, 2.0, 1.0])
        self.assertAlmostEqual(sum(t.quantity for t in trades), 4.0)
        self.assertEqual(trades[0].sell_agent_id, "A")
        self.assertEqual(trades[2].sell_agent_id, "C")
        # 102 档还剩 4
        self.assertEqual(self.book.best_ask(), 102.0)
        self.assertAlmostEqual(self.book.total_quantity("sell"), 4.0)
        ok, probs = self.book.invariants_ok()
        self.assertTrue(ok, probs)

    def test_限价扫单只吃到限价为止(self) -> None:
        trades = self.engine.submit(mk("HUNGRY", "buy", 101.0, 5.0, tick=1, oid="l1"))
        self.assertEqual([t.price for t in trades], [100.0, 101.0])
        self.assertAlmostEqual(sum(t.quantity for t in trades), 3.0)
        rest = self.book.get_order("l1")
        self.assertIsNotNone(rest, "限价单未成交部分应挂在簿上")
        self.assertAlmostEqual(rest.remaining, 2.0)
        self.assertEqual(self.book.best_bid(), 101.0)
        self.assertEqual(self.book.best_ask(), 102.0)

    def test_市价单吃空簿后不挂单(self) -> None:
        trades = self.engine.submit(mk_market("HUNGRY", "buy", 100.0, tick=1, oid="m2"))
        self.assertAlmostEqual(sum(t.quantity for t in trades), 8.0)
        self.assertEqual(self.book.n_orders, 0)
        self.assertFalse(self.book.has_order("m2"))

    def test_空簿市价单不产生成交(self) -> None:
        book = OrderBook(tick_size=1.0)
        eng = MatchingEngine(book)
        trades = eng.submit(mk_market("A", "buy", 1.0))
        self.assertEqual(trades, [])
        self.assertEqual(book.n_orders, 0)

    def test_价格优先_先吃最便宜的一档(self) -> None:
        """乱序挂单后，最优价必须仍是最便宜的卖价。"""
        book = OrderBook(tick_size=1.0)
        eng = MatchingEngine(book)
        for p in (105.0, 101.0, 103.0, 100.0, 104.0):
            eng.submit(mk("X", "sell", p, 1.0, oid=f"s{p:g}"))
        self.assertEqual(book.best_ask(), 100.0)
        trades = eng.submit(mk_market("H", "buy", 1.0, tick=1, oid="m"))
        self.assertEqual(trades[0].price, 100.0)

    def test_买侧价格优先_最高价先成交(self) -> None:
        book = OrderBook(tick_size=1.0)
        eng = MatchingEngine(book)
        for p in (95.0, 99.0, 97.0):
            eng.submit(mk("X", "buy", p, 1.0, oid=f"b{p:g}"))
        self.assertEqual(book.best_bid(), 99.0)
        trades = eng.submit(mk_market("H", "sell", 1.0, tick=1, oid="m"))
        self.assertEqual(trades[0].price, 99.0)

    def test_时间优先_同价先挂先成交(self) -> None:
        book = OrderBook(tick_size=1.0)
        eng = MatchingEngine(book)
        eng.submit(mk("X", "sell", 100.0, 1.0, tick=0, oid="早"))
        eng.submit(mk("Y", "sell", 100.0, 1.0, tick=1, oid="晚"))
        trades = eng.submit(mk_market("H", "buy", 1.0, tick=2, oid="m"))
        self.assertEqual(trades[0].sell_agent_id, "X", "应按时间优先成交先挂的那张")
        self.assertEqual(trades[0].sell_order_id, "早")


class TestSelfTradePrevention(unittest.TestCase):
    def setUp(self) -> None:
        self.book = OrderBook(tick_size=1.0)
        self.engine = MatchingEngine(self.book, allow_self_trade=False)

    def test_不与自己成交(self) -> None:
        self.engine.submit(mk("MM", "sell", 100.0, 5.0, oid="s"))
        trades = self.engine.submit(mk("MM", "buy", 100.0, 5.0, oid="b"))
        self.assertEqual(trades, [], "不该出现自成交")
        self.assertEqual(self.book.best_ask(), 100.0)

    def test_跳过自家挂单后仍可与他人成交(self) -> None:
        # A 先挂，B 后挂，同价；A 再来自买
        self.engine.submit(mk("A", "sell", 100.0, 1.0, tick=0, oid="sA"))
        self.engine.submit(mk("B", "sell", 100.0, 1.0, tick=1, oid="sB"))
        trades = self.engine.submit(mk("A", "buy", 100.0, 1.0, tick=2, oid="bA"))
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0].sell_agent_id, "B")
        self.assertEqual(trades[0].buy_agent_id, "A")
        # A 自己的卖单还留在簿上
        self.assertTrue(self.book.has_order("sA"))

    def test_允许自成交时才会自成交(self) -> None:
        book = OrderBook(tick_size=1.0)
        eng = MatchingEngine(book, allow_self_trade=True)
        eng.submit(mk("MM", "sell", 100.0, 5.0, oid="s"))
        trades = eng.submit(mk("MM", "buy", 100.0, 5.0, oid="b"))
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0].buy_agent_id, trades[0].sell_agent_id)


class TestInvariantsUnderFuzz(unittest.TestCase):
    """随机压力下的不变量检查。

    这是最强的一道防线：逐个功能的断言可能都过，但组合起来仍会漏单/串档。
    2000 次随机操作（限价 / 市价 / 撤单）之后，订单簿内部状态必须自洽。
    """

    def test_随机操作后订单簿保持自洽(self) -> None:
        rng = random.Random(20260917)
        book = OrderBook(tick_size=1.0)
        engine = MatchingEngine(book, allow_self_trade=False)
        agents = [f"A{i}" for i in range(6)]
        issued: set[str] = set()
        cancelled: list[str] = []
        all_trades: list[Trade] = []
        oid_n = 0
        ops = 0

        for step in range(2000):
            r = rng.random()
            if r < 0.12 and issued:
                # 从"当前簿上"的订单里随机挑一个撤
                candidates = sorted(book.open_order_ids())
                if candidates:
                    oid = candidates[rng.randrange(len(candidates))]
                    if book.cancel_order(oid) is not None:
                        cancelled.append(oid)
                ops += 1
                continue
            oid_n += 1
            oid = f"o{oid_n}"
            issued.add(oid)
            agent = rng.choice(agents)
            side = rng.choice(["buy", "sell"])
            if r < 0.22:
                o = mk_market(agent, side, rng.uniform(0.1, 5.0), tick=step, oid=oid)
                all_trades.extend(engine.submit(o))
            else:
                price = round(rng.uniform(94.0, 106.0))
                o = mk(agent, side, price, rng.uniform(0.1, 5.0), tick=step, oid=oid)
                all_trades.extend(engine.submit(o))
            ops += 1

            ok, probs = book.invariants_ok()
            self.assertTrue(ok, f"第 {step} 步后订单簿不自洽: {probs}")
            for t in all_trades[-4:]:
                self.assertGreater(t.quantity, 0.0)
                self.assertGreater(t.price, 0.0)
                self.assertNotEqual(
                    t.buy_agent_id, t.sell_agent_id, "自成交防护失效"
                )

        self.assertGreater(ops, 0)
        self.assertGreater(len(all_trades), 100, "随机压力下应产生足量成交")
        for oid in cancelled:
            self.assertFalse(book.has_order(oid), f"已撤销的 {oid} 仍在簿上")
        # 簿里不允许出现"来路不明"的订单：每一张挂单都必须是本次测试提交过的。
        phantom = book.open_order_ids() - issued
        self.assertFalse(phantom, f"订单簿中出现未提交过的幽灵订单: {phantom}")
        # 挂单数 > 0 时，簿子必须真的能给出最优价（不能只有索引没有档位）
        if book.n_orders > 0:
            self.assertTrue(
                (book.best_bid() is not None) or (book.best_ask() is not None),
                "簿上有挂单却取不到最优价",
            )

    def test_随机压力下持仓守恒(self) -> None:
        """每一笔成交都一买一卖 —— 全市场各 agent 的持仓变动净额必须为零。

        这是撮合最硬的一条守恒律：任何"凭空多出/少了仓位"的实现错误都会在这里现形。
        不能简化成"买单量总和 == 卖单量总和"——那种写法恒真，等于没测。
        """
        rng = random.Random(7)
        book = OrderBook(tick_size=1.0)
        engine = MatchingEngine(book, allow_self_trade=False)
        trades: list[Trade] = []
        for step in range(800):
            agent = f"A{rng.randrange(4)}"
            side = rng.choice(["buy", "sell"])
            o = mk(agent, side, round(rng.uniform(96.0, 104.0)), rng.uniform(0.1, 3.0), step, f"x{step}")
            trades.extend(engine.submit(o))

        self.assertGreater(len(trades), 50, "样本太少，守恒检验失去意义")

        net: dict[str, float] = {}
        cash_flow: dict[str, float] = {}
        for t in trades:
            net[t.buy_agent_id] = net.get(t.buy_agent_id, 0.0) + t.quantity
            net[t.sell_agent_id] = net.get(t.sell_agent_id, 0.0) - t.quantity
            cash_flow[t.buy_agent_id] = cash_flow.get(t.buy_agent_id, 0.0) - t.notional
            cash_flow[t.sell_agent_id] = cash_flow.get(t.sell_agent_id, 0.0) + t.notional

        self.assertAlmostEqual(
            sum(net.values()), 0.0, places=9, msg="全市场持仓净额不为零 —— 凭空创造了仓位"
        )
        self.assertAlmostEqual(
            sum(cash_flow.values()), 0.0, places=6, msg="全市场现金流不为零 —— 凭空创造了现金"
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
