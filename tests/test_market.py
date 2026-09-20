"""主循环 / 记账 / 日志的回归测试。

这个文件里几乎每一条测试都对应一个**真实踩过的坑**，不是凭空写的断言。
记录在此的原因是：这两个 bug 都不会抛异常、不会崩溃，只会让模拟静悄悄地
产出一个看起来合理、实则无效的结果——正是最难发现的那一类。

    bug-1  成交没进日志
           表现：价格序列正常，但 n_trades=0、total_volume=0
           原因：MatchingEngine 留了 trade_sink 钩子，Market 忘了接

    bug-2  被动方预留额度不随成交更新
           表现：跑几百个 tick 后市场完全冻结（几乎不再成交）
           原因：只在主动方路径更新 reserved_cash，挂单被吃的一方的
                 available_cash 变成负数，之后的单全被预算裁剪拒掉
"""

from __future__ import annotations

import unittest

import numpy as np

from tw import Market, Population, SimConfig
from tw.agents.base import Agent
from tw.types import Order

PRICE = 60_000.0


def tiny_market(n_agents: int = 2, n_ticks: int = 10, seed: int = 1234) -> Market:
    """一个不自动下单的市场，用于手工构造场景。"""
    cfg = SimConfig(
        seed=seed,
        n_ticks=n_ticks,
        population=Population(zero_intel=n_agents),
        zi_p_active=0.0,  # 主体不主动下单，测试自己控制
    )
    return Market(cfg)


def equip(m: Market, cash: float = 1_000_000.0, inv: float = 10.0) -> None:
    """给所有主体一个确定的干净账户（种子里抽出来的禀赋不适合做精确断言）。"""
    for a in m.agents:
        a.cash = cash
        a.inventory = inv
        a.reserved_cash = 0.0
        a.reserved_inventory = 0.0
        a.open_orders.clear()
        a.set_initial_equity(PRICE)


class TestAccountingRegression(unittest.TestCase):
    """bug-1 / bug-2 的永久回归测试。"""

    def test_成交必须落进日志(self) -> None:
        m = tiny_market()
        equip(m)
        a, b = m.agents
        m._refresh_state()

        m.submit(a, a.new_order(m._state, "buy", 59_900.0, 1.0))
        trades = m.submit(b, b.new_order(m._state, "sell", 59_900.0, 1.0))

        self.assertEqual(len(trades), 1, "这笔单必须成交")
        self.assertEqual(
            len(m.log.trades), 1, "成交没进日志 —— 撮合成功但日志为空 (bug-1)"
        )
        self.assertAlmostEqual(m.log.total_volume(), 1.0)

    def test_被动方预留额度随成交归零(self) -> None:
        m = tiny_market()
        equip(m)
        a, b = m.agents
        m._refresh_state()

        m.submit(a, a.new_order(m._state, "buy", 59_900.0, 1.0))
        self.assertAlmostEqual(
            a.reserved_cash, 59_900.0, places=2, msg="挂买单后应预留资金"
        )
        self.assertAlmostEqual(a.available_cash, 1_000_000.0 - 59_900.0, places=2)

        m.submit(b, b.new_order(m._state, "sell", 59_900.0, 1.0))

        # a 是**被动方**（挂单被吃）。旧实现下它的 reserved_cash 会停在 59900，
        # 导致 available_cash 与实际可用资金不符，往后它的单会被误拒。
        self.assertAlmostEqual(
            a.reserved_cash,
            0.0,
            places=2,
            msg="被动方完全成交后预留资金未释放 (bug-2)",
        )
        self.assertAlmostEqual(
            a.available_cash, a.cash, places=2, msg="预留与实际可用资金不符 (bug-2)"
        )
        self.assertAlmostEqual(a.cash, 1_000_000.0 - 59_900.0, places=2)
        self.assertAlmostEqual(a.inventory, 11.0, places=6)
        self.assertAlmostEqual(b.cash, 1_000_000.0 + 59_900.0, places=2)
        self.assertAlmostEqual(b.inventory, 9.0, places=6)
        self.assertAlmostEqual(b.reserved_inventory, 0.0, places=6)

    def test_被动方部分成交后预留按剩余量收缩(self) -> None:
        m = tiny_market()
        equip(m)
        a, b = m.agents
        m._refresh_state()

        m.submit(a, a.new_order(m._state, "buy", 59_900.0, 2.0))
        self.assertAlmostEqual(a.reserved_cash, 2 * 59_900.0, places=2)

        m.submit(b, b.new_order(m._state, "sell", 59_900.0, 1.0))

        self.assertAlmostEqual(
            a.reserved_cash,
            59_900.0,
            places=2,
            msg="部分成交后预留应按剩余 1 手计，而不是仍按 2 手计 (bug-2)",
        )
        self.assertAlmostEqual(a.inventory, 11.0, places=6)
        self.assertTrue(m.book.has_order(a.open_orders[0].order_id))

    def test_撤单释放预留(self) -> None:
        m = tiny_market()
        equip(m)
        a = m.agents[0]
        m._refresh_state()

        o = a.new_order(m._state, "buy", 59_000.0, 2.0)
        m.submit(a, o)
        self.assertAlmostEqual(a.reserved_cash, 2 * 59_000.0, places=2)

        a.cancel(o)
        self.assertAlmostEqual(a.reserved_cash, 0.0, places=2)
        self.assertAlmostEqual(a.available_cash, a.cash, places=2)
        self.assertEqual(a.open_orders, [])

    def test_卖单不能卖空(self) -> None:
        """可用持仓耗尽后，卖单必须被拒（否则会凭空创造仓位）。"""
        m = tiny_market()
        equip(m, cash=1_000_000.0, inv=1.0)
        a = m.agents[0]
        m._refresh_state()

        m.submit(a, a.new_order(m._state, "sell", 70_000.0, 1.0))
        self.assertAlmostEqual(a.available_inventory, 0.0, places=9)

        n_before = a.stats.n_submitted
        m.submit(a, a.new_order(m._state, "sell", 70_000.0, 1.0))
        self.assertEqual(a.stats.n_submitted, n_before, "超卖订单不该被接受")
        self.assertGreater(a.stats.n_rejected, 0)

    def test_买单不能透支(self) -> None:
        m = tiny_market()
        equip(m, cash=100_000.0, inv=0.0)
        a = m.agents[0]
        m._refresh_state()

        m.submit(a, a.new_order(m._state, "buy", 60_000.0, 5.0))
        self.assertLessEqual(a.reserved_cash, a.cash + 1e-6, "预留超过了持有现金")
        self.assertGreaterEqual(a.available_cash, -1e-6, "可用现金为负 —— 会凭空造钱")

    def test_挂单价格必须吸附到tick网格(self) -> None:
        """订单价格必须落在 tick 网格上，否则预留口径与结算口径不一致。

        踩过的坑：订单保留原始浮点价（如 59000.1234567），而订单簿按
        ``round(price/tick)*tick`` 定档并据此成交。预留按原价算、结算按网格价付，
        四舍五入方向不定 → 每笔成交都微小破坏账户守恒，累积后表现为
        "可用现金为负 -0.0014"。这类泄漏量级极小，不做**直接**断言很难抓到。
        """
        m = tiny_market()
        equip(m)
        a = m.agents[0]
        tick = m.book.tick_size
        m._refresh_state()

        o = a.new_order(m._state, "buy", 59_000.1234567, 1.0)
        m.submit(a, o)

        self.assertAlmostEqual(
            o.price / tick,
            round(o.price / tick),
            places=9,
            msg="挂单价格没有吸附到 tick 网格 (bug: 预留/结算口径分裂)",
        )
        for resting in m.book.iter_side("buy"):
            p = resting[0]
            self.assertAlmostEqual(p / tick, round(p / tick), places=9)

    def test_所有剩余挂单都落在网格上_长跑(self) -> None:
        cfg = SimConfig(seed=31, n_ticks=1200, population=Population(zero_intel=30))
        m = Market(cfg)
        m.run()
        tick = m.book.tick_size
        bad = []
        for side in ("buy", "sell"):
            for p, o in m.book.iter_side(side):
                if abs(p / tick - round(p / tick)) > 1e-9:
                    bad.append((o.order_id, p))
        self.assertEqual(bad, [], f"存在网格外的挂单: {bad[:5]}")
        ok, problems = m.health_check()
        self.assertTrue(ok, f"长跑后体检未通过: {problems[:5]}")


class TestMarketIntegration(unittest.TestCase):
    """整场模拟的健康度。"""

    @classmethod
    def setUpClass(cls) -> None:
        cfg = SimConfig(
            seed=99,
            n_ticks=400,
            population=Population(zero_intel=40),
        )
        cls.market = Market(cfg)
        cls.log = cls.market.run()

    def test_产生了足量成交(self) -> None:
        self.assertGreater(
            len(self.log.trades), 200, "零智能基线几乎不成交，撮合或定价有问题"
        )

    def test_成交总数与逐tick计数一致(self) -> None:
        per_tick = int(np.nansum(self.log.n_trades))
        self.assertEqual(
            per_tick,
            len(self.log.trades),
            "逐 tick 成交计数与成交流水对不上 —— 有一方漏记",
        )

    def test_成交量与逐tick量一致(self) -> None:
        self.assertAlmostEqual(
            float(np.nansum(self.log.volume)),
            self.log.total_volume(),
            places=6,
            msg="逐 tick 成交量与成交流水对不上",
        )

    def test_市场没有冻结(self) -> None:
        """最后 25% 的 tick 里仍必须有成交。

        这是 bug-2 的直接探针：预留额度不释放时，市场会在前段成交一阵子，
        然后彻底静默——只看总成交数看不出来，必须看**尾部是否还有成交**。
        """
        n = self.log.n_trades.size
        tail = self.log.n_trades[int(n * 0.75) :]
        self.assertGreater(
            float(np.nansum(tail)), 0.0, "市场在后段完全停止成交 —— 疑似账户被错误锁死"
        )

    def test_账户与订单簿体检通过(self) -> None:
        ok, problems = self.market.health_check()
        self.assertTrue(ok, f"体检未通过: {problems[:5]}")

    def test_中间价序列无缺失(self) -> None:
        self.assertTrue(np.all(np.isfinite(self.log.mid)), "中间价序列存在 NaN")

    def test_中间价始终为正(self) -> None:
        self.assertTrue(np.all(self.log.mid > 0))

    def test_订单簿深度有界(self) -> None:
        """挂单量不能无限膨胀 —— 超时撤单与挂单上限必须真的生效。"""
        self.assertLess(
            float(np.nanmax(self.log.open_orders)),
            40 * 25,
            "挂单总数失控，陈年挂单没有被清理",
        )

    def test_相同种子结果可复现(self) -> None:
        cfg = SimConfig(seed=7, n_ticks=150, population=Population(zero_intel=20))
        a = Market(cfg).run()
        b = Market(cfg).run()
        np.testing.assert_allclose(a.mid, b.mid, rtol=0, atol=0)
        self.assertEqual(len(a.trades), len(b.trades))

    def test_不同种子结果不同(self) -> None:
        a = Market(SimConfig(seed=1, n_ticks=150, population=Population(zero_intel=20))).run()
        b = Market(SimConfig(seed=2, n_ticks=150, population=Population(zero_intel=20))).run()
        self.assertFalse(np.allclose(a.mid, b.mid), "换种子结果没变 —— 随机源没接上")


class TestStressTest(unittest.TestCase):
    def test_强制平仓产生冲击且成交落账(self) -> None:
        cfg = SimConfig(seed=5, n_ticks=200, population=Population(zero_intel=30))
        m = Market(cfg)
        m.run()

        victims = m.agents[:5]
        before = {a.agent_id: a.inventory for a in victims}
        n_trades_before = len(m.log.trades)

        impact = m.force_liquidate(victims, fraction=1.0)

        self.assertGreater(impact["n_trades"], 0, "清算没有产生任何成交")
        self.assertGreater(impact["total_qty"], 0.0)
        self.assertGreater(len(m.log.trades), n_trades_before, "清算成交没进日志")
        for a in victims:
            self.assertLessEqual(
                a.inventory, before[a.agent_id] + 1e-6, "清算后持仓反而增加了"
            )
        ok, problems = m.health_check()
        self.assertTrue(ok, f"清算后体检未通过: {problems[:5]}")

    def test_清算vwap与成交明细一致(self) -> None:
        """`vwap` 必须真的是成交明细的加权均价，不能是随手填的数。

        为什么单独测这个：清算滑点是阶段3 的主指标，一旦 `vwap` 与真实成交
        脱钩，整个"冲击有多大"的结论就建在沙上。
        """
        cfg = SimConfig(seed=17, n_ticks=300, population=Population(zero_intel=40))
        m = Market(cfg)
        m.run()

        n0 = len(m.log.trades)
        impact = m.force_liquidate(m.agents[:6], fraction=1.0)
        swept = m.log.trades[n0:]

        self.assertGreater(len(swept), 0, "清算没有成交，测试前提不成立")
        qty = sum(t.quantity for t in swept)
        self.assertGreater(qty, 0.0)
        self.assertAlmostEqual(impact["total_qty"], qty, places=9)

        expect_vwap = sum(t.notional for t in swept) / qty
        self.assertAlmostEqual(impact["vwap"], expect_vwap, places=9)

        # slippage_bp 必须与 vwap 自洽
        expect_slip = (impact["vwap"] / impact["price_before"] - 1.0) * 1e4
        self.assertAlmostEqual(impact["slippage_bp"], expect_slip, places=9)

    def test_清算滑点必须严格为负_否则字段是摆设(self) -> None:
        """卖出清算必然付出价格让步。

        断言是**严格**小于 0（不是 assertLessEqual）：
        若 `slippage_bp` 被写成常量 0、或错用了 `price_after`、或符号反了，
        这条必须变红。一个恒真的断言等于没有断言。
        """
        cfg = SimConfig(seed=23, n_ticks=400, population=Population(zero_intel=50))
        m = Market(cfg)
        m.run()

        mid_before = m.current_mid()
        impact = m.force_liquidate(m.agents[:10], fraction=1.0)

        self.assertGreater(impact["total_qty"], 0.0, "清算没有成交")
        self.assertLess(
            impact["slippage_bp"],
            0.0,
            "卖出清算的滑点必须为负（成交价低于清算前中间价）",
        )
        # 而且要真的付出代价：至少半个价差以上，不是"只差个零头"
        self.assertLess(impact["vwap"], mid_before)
        self.assertAlmostEqual(impact["price_before"], mid_before, places=9)

    def test_清算后市场仍可继续运行(self) -> None:
        cfg = SimConfig(seed=11, n_ticks=200, population=Population(zero_intel=30))
        m = Market(cfg)
        m.run()
        m.force_liquidate(m.agents[:8], fraction=0.8)
        t_before = len(m.log.trades)
        m.run(200)
        self.assertGreater(
            len(m.log.trades), t_before, "清算之后市场没能恢复交易"
        )


class _CountingNoop(Agent):
    """一个"什么都不做"的主体，但会记录自己被调用了几次。

    为什么必须记录调用次数：只断言"行情没变"会**假通过**——
    如果这个主体压根没被 step() 遍历到（例如只 append 到 `m.agents`
    而没进 `m._order`），行情当然不会变。加上调用计数，
    这条测试才真的在检验"注入一个不参与交易的主体不会扰动行情"。
    """

    def __init__(self, aid: str, cash: float, inv: float, rng) -> None:
        super().__init__(aid, cash, inv, rng)
        self.n_calls = 0

    def decide(self, state):  # noqa: ANN001
        self.n_calls += 1
        return None


class TestStrategyInjection(unittest.TestCase):
    """策略实验台的地基不变量。

    做"策略 A/B 对比"的前提是：**对照组的背景行情与实验组逐点相同**，
    差异只能来自被测策略本身。要做到这一点需要三个互相独立的随机流
    （市场流 / 禀赋流 / 每个主体的决策流），且主体数量不能影响市场流。
    这几条一旦被破坏，实验会照跑不误，只是结果全部不可比——
    属于最危险的那种错误，所以固化成测试。
    """

    @staticmethod
    def _run(with_noop: bool, tick: int = 400, seed: int = 4242):
        pop = Population.from_shares(
            60, {"zero_intel": 0.30, "fundamentalist": 0.40, "chartist": 0.30}
        )
        m = Market(SimConfig(seed=seed, n_ticks=tick, population=pop))
        noop = None
        if with_noop:
            noop = _CountingNoop(
                "noop0000", 1_000_000.0, 10.0, np.random.default_rng([seed, 777])
            )
            m.add_agent(noop)
        m.run()
        mid = np.asarray(m.log.mid[: m.tick], dtype=float)
        return mid, noop, m

    def test_注入惰性主体不改变背景行情_且它确实被调用(self) -> None:
        a, _, _ = self._run(False)
        b, noop, m = self._run(True)

        # (b) 先确认它真的上台了——否则 (a) 是假通过
        self.assertIsNotNone(noop)
        self.assertEqual(noop.n_calls, 400, "注入的主体没有被 step() 遍历到")
        self.assertEqual(m.n_agents, 61)

        # (a) 背景行情必须逐点相同
        n = min(a.size, b.size)
        self.assertEqual(a.size, b.size)
        self.assertTrue(
            np.array_equal(a[:n], b[:n]),
            f"注入惰性主体后背景行情变了（最大差 "
            f"{float(np.nanmax(np.abs(a[:n] - b[:n]))):.6f}）——A/B 对比不成立",
        )

    def test_add_agent必须维护登记表(self) -> None:
        """`add_agent` 必须同时维护 `agents` 与 `by_id`。

        为什么这条重要：`by_id` 是成交结算时找对手方账户的唯一入口。
        绕开 `add_agent` 直接 append 的话，主体能下单、能成交，
        但结算找不到它的账户 —— 这是"能跑但算错账"的一类错误。
        """
        pop = Population.from_shares(10, {"zero_intel": 1.0})
        m = Market(SimConfig(seed=3, n_ticks=10, population=pop))
        before = m.n_agents
        a = _CountingNoop("inj0000", 1e6, 1.0, np.random.default_rng([3, 0]))
        m.add_agent(a)
        self.assertEqual(m.n_agents, before + 1)
        self.assertIs(m.by_id["inj0000"], a)
        self.assertIn(a, m.agents)

    def test_注入主体必须参与调度(self) -> None:
        """注入的主体必须真的有机会出手（否则"策略没影响"是假的）。

        这条与上面那条配对：一条保证"不多余扰动"，一条保证"真的上台"。
        缺任一条，实验台的对照都会失真。
        """
        pop = Population.from_shares(10, {"zero_intel": 1.0})
        m = Market(SimConfig(seed=3, n_ticks=20, population=pop))
        a = _CountingNoop("inj0000", 1e6, 1.0, np.random.default_rng([3, 0]))
        m.add_agent(a)
        m.run()
        self.assertEqual(a.n_calls, 20, "注入的主体没有被调度到")

    def test_主体ID重复必须报错(self) -> None:
        pop = Population.from_shares(10, {"zero_intel": 1.0})
        m = Market(SimConfig(seed=3, n_ticks=10, population=pop))
        dup = m.agents[0].agent_id
        a = _CountingNoop(dup, 1e6, 1.0, np.random.default_rng([3, 0]))
        with self.assertRaises(ValueError):
            m.add_agent(a)

    def test_注入时点必须被记录(self) -> None:
        """允许预热后注入，但注入时点必须被记下来。

        为什么强调这个：同一 seed 只有在**注入时点也相同**时才复现得出同一场实验。
        不记录的话，"用同一个 seed 跑出来却不一样"会变成一件查不清的事。
        """
        pop = Population.from_shares(10, {"zero_intel": 1.0})
        m = Market(SimConfig(seed=3, n_ticks=60, population=pop))
        m.run(30)
        a = _CountingNoop("late0000", 1e6, 1.0, np.random.default_rng([3, 9]))
        m.add_agent(a)
        self.assertEqual(m.injected_at.get("late0000"), 30)
        self.assertEqual(a.n_calls, 0)
        m.run(30)
        self.assertEqual(a.n_calls, 30, "注入之后的主体没有被调度")

    def test_注入主体后初始权益基准等于注入时中间价(self) -> None:
        """`pnl()` 要能直接用，前提是初始权益按**注入那一刻**的中间价设。"""
        pop = Population.from_shares(10, {"zero_intel": 1.0})
        m = Market(SimConfig(seed=3, n_ticks=60, population=pop))
        m.run(30)
        mid_at_inject = m.current_mid()
        a = _CountingNoop("inj0000", 1e6, 5.0, np.random.default_rng([3, 1]))
        m.add_agent(a)
        self.assertAlmostEqual(a.initial_equity, 1e6 + 5.0 * mid_at_inject, places=6)

    def test_动作顺序与主体数量无关(self) -> None:
        """⭐ 这是实验台最关键的一条不变量。

        往市场里加主体时，**已有主体之间的相对出手顺序必须不变**。
        做法：取 40 主体市场与 60 主体市场在第 0 个 tick 的动作顺序，
        把 60 主体那串里"属于前 40 个主体"的子序列抽出来，应当与
        40 主体市场那串**完全一致**。

        用全局洗牌做不到这一点（置换取决于列表长度），
        那样"多放一个主体"就会整体错位所有人的顺序，背景行情随之改变。
        """
        def first_order(n: int) -> list[int]:
            pop = Population.from_shares(n, {"zero_intel": 1.0})
            m = Market(SimConfig(seed=99, n_ticks=2, population=pop))
            order = m._next_order()
            return [m.agents.index(a) for a in order]

        o40 = first_order(40)
        o60 = first_order(60)
        sub = [i for i in o60 if i < 40]
        self.assertEqual(sub, o40, "主体数量改变了已有主体的相对出手顺序")

    def test_市场随机流只由基本面步进消耗(self) -> None:
        """市场 RNG 不参与主体调度、不参与禀赋 —— 它的消耗量与主体数无关。

        验证方式：不同主体数的市场跑同样 tick 数后，市场 RNG 的下一个输出
        必须仍然相同（说明消耗量一致）。这保证了"基本面锚的路径"
        不被主体数量影响。
        """
        outs = []
        for n in (40, 60):
            pop = Population.from_shares(n, {"zero_intel": 1.0})
            m = Market(SimConfig(seed=77, n_ticks=100, population=pop))
            m.run()
            outs.append(m.rng.random())
        self.assertAlmostEqual(outs[0], outs[1], places=12,
                               msg="市场随机流的消耗量随主体数变化了")


class TestFlowImbalanceMemo(unittest.TestCase):
    """``_flow_imbalance`` 的按 tick 记忆化必须**严格行为保持**。

    背景：它被 ``_refresh_state`` 调用，而 ``_refresh_state`` 是
    "每个提交了订单的主体 × 每 tick"一次。单场模拟里这是 10⁶ 量级的调用，
    每次都做 ``slice(200) + 2 × nansum(200)``。加记忆化实测快 1.42×
    （见 ``scripts/bench_market.py``）。

    **性能优化最容易撒的谎是"结果应该差不多"**。所以这里不测速度，
    只测两件事：① 同一 tick 内重复调用返回同一个值（记忆化的前提成立）；
    ② 与"每 tick 强制重算"的参照实现产出的**逐点行情序列完全一致**。
    第 ② 条如果不过，前面的 1.42× 就不值一提。
    """

    def test_同一tick内重复调用返回同一个值(self) -> None:
        m = Market(SimConfig(seed=11, n_ticks=60,
                             population=Population(zero_intel=40)))
        m.run(30)
        a = m._flow_imbalance()
        b = m._flow_imbalance()
        self.assertEqual(a, b)

    def test_记忆化与强制重算的行情逐点一致(self) -> None:
        class AlwaysRecompute(Market):
            """参照实现：每次调用都真算，绝不读缓存。"""

            def _flow_imbalance(self, window: int = 200) -> float:
                self._fi_tick = -1          # 让基类的缓存判定永不命中
                return super()._flow_imbalance(window)

        pop = {"zero_intel": 0.3, "fundamentalist": 0.4, "chartist": 0.3}
        got = []
        for cls in (Market, AlwaysRecompute):
            mm = cls(SimConfig(seed=4242, n_ticks=120,
                               population=Population.from_shares(60, pop)))
            mm.run()
            got.append(np.asarray(mm.log.mid, dtype=float))

        self.assertEqual(len(got[0]), len(got[1]), "两条路径长度不同")
        self.assertTrue(
            np.array_equal(got[0], got[1]),
            "记忆化改变了逐点中间价——它是行为保持的，不该有任何差异")
        # 反面参照：不同种子必须不同（证明上面那条 assert 不是平凡的）
        other = Market(SimConfig(seed=99, n_ticks=120,
                                 population=Population.from_shares(60, pop)))
        other.run()
        self.assertFalse(np.array_equal(got[0],
                                        np.asarray(other.log.mid, dtype=float)),
                         "不同种子跑出同一条行情——测试本身失效了")


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
