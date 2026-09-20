"""元订单拆分执行的测试（二期阶段6，LMF 第一环）。

这个文件里最要紧的是三条**结构性性质**——它们错了，机制会静默失效，
而症状只是"订单符号 ACF 没变正"，看起来像"市场本来就没长记忆"：

1. **总量守恒**：发出的子单量之和 == 采样的元订单总量（不多不少）
2. **方向锁定**：一个元订单执行期间，所有子单同向
3. **一定终止**：``max_child_orders`` 与剩余量两个闸门都真的会关
"""

from __future__ import annotations

import unittest

import numpy as np

from tw import Market, Population, SimConfig
from tw.agents.chartist import Chartist
from tw.agents.zero_intel import ZeroIntelligence
from tw.order_flow.meta_order import (
    MetaOrderConfig,
    MetaOrderMixin,
    MetaOrderState,
)
from tw.types import MarketState, PriceHistory


class _Fixed(ZeroIntelligence):
    """每次决策都返回指定方向的单，用来把"方向"这个变量钉死。"""

    KIND = "test_fixed"
    WANT = "buy"

    def decide(self, state):
        qty = float(self.rng.exponential(1.0))
        if self.WANT == "buy":
            return self.buy_order(state, qty, state.mid * 0.999)
        return self.sell_order(state, qty, state.mid * 1.001)


class MetaFixed(MetaOrderMixin, _Fixed):
    pass


class MetaZI(MetaOrderMixin, ZeroIntelligence):
    pass


class MetaChartist(MetaOrderMixin, Chartist):
    pass


def _state(mid: float = 100.0, tick: int = 0, best_bid=None, best_ask=None):
    hist = PriceHistory(16)
    hist.append(mid)
    return MarketState(
        tick=tick, mid=mid, last_price=mid,
        best_bid=best_bid if best_bid is not None else mid - 1.0,
        best_ask=best_ask if best_ask is not None else mid + 1.0,
        spread=2.0, fundamental=mid, history=hist,
    )


def _agent(cls, **kw):
    a = cls("m0000", 1e7, 100.0, np.random.default_rng([7]), **kw)
    a.bind_seed(1234)
    return a


# ----------------------------------------------------------------------
class TestParetoSampling(unittest.TestCase):
    """⭐ M23（Pareto 公式）的正向护栏。

    用错公式（例如漏掉 ``−1/α`` 的幂）会让分布退化成近均匀、尾部消失，
    而"长记忆来自重尾"这条链就断了。症状不是报错，是订单符号 ACF 掉得很快。
    """

    def test_尾部指数是对的(self) -> None:
        """Pareto 的定义性质：``P(X > t·xmin) = t^{−α}``。

        用经验互补累积分布去核对**解析式**，而不是核对"看起来很长"。
        """
        cfg = MetaOrderConfig(pareto_alpha=1.5, pareto_xmin=1.0, pareto_scale=1.0)
        a = _agent(MetaFixed, meta_config=cfg)
        rng_backup = a._rng_meta
        a._rng_meta = np.random.default_rng([2026])
        xs = np.array([a.sample_meta_total() for _ in range(200_000)])
        a._rng_meta = rng_backup
        for t in (2.0, 4.0, 10.0):
            emp = float((xs > t).mean())
            theory = t ** -cfg.pareto_alpha
            self.assertAlmostEqual(
                emp, theory, delta=0.25 * theory + 0.002,
                msg=f"P(X>{t}) 经验值 {emp:.5f} vs 解析值 {theory:.5f}",
            )

    def test_分布确实是重尾而不是均匀(self) -> None:
        cfg = MetaOrderConfig(pareto_alpha=1.5, pareto_xmin=1.0)
        a = _agent(MetaFixed, meta_config=cfg)
        a._rng_meta = np.random.default_rng([11])
        xs = np.array([a.sample_meta_total() for _ in range(100_000)])
        # 均匀分布不会有超过 xmin 三个数量级的最大值
        self.assertGreater(xs.max(), 100.0, "尾部消失了——Pareto 公式可能写错")
        self.assertGreater(float(np.percentile(xs, 99)), 5.0)

    def test_均值与解析式一致(self) -> None:
        cfg = MetaOrderConfig(pareto_alpha=1.5, pareto_xmin=1.0)
        a = _agent(MetaFixed, meta_config=cfg)
        a._rng_meta = np.random.default_rng([13])
        xs = np.array([a.sample_meta_total() for _ in range(200_000)])
        self.assertAlmostEqual(float(xs.mean()), cfg.expected_total(), delta=0.05)

    def test_scale乘子线性生效(self) -> None:
        c1 = MetaOrderConfig(pareto_scale=1.0)
        c2 = MetaOrderConfig(pareto_scale=10.0)
        a1, a2 = _agent(MetaFixed, meta_config=c1), _agent(MetaFixed, meta_config=c2)
        a1._rng_meta = np.random.default_rng([17])
        a2._rng_meta = np.random.default_rng([17])
        x1 = np.array([a1.sample_meta_total() for _ in range(2000)])
        x2 = np.array([a2.sample_meta_total() for _ in range(2000)])
        np.testing.assert_allclose(x2, x1 * 10.0, rtol=1e-12)

    def test_非法参数被拒(self) -> None:
        for bad in ({"pareto_alpha": 1.0}, {"pareto_alpha": 0.5},
                    {"pareto_xmin": 0.0}, {"pareto_scale": 0.0},
                    {"participation_rate": 0.0}, {"participation_rate": 1.5},
                    {"max_child_orders": 0}, {"p_meta_start": 1.5}):
            with self.assertRaises(ValueError, msg=f"{bad} 应当被拒"):
                MetaOrderConfig(**bad)


class TestSplitting(unittest.TestCase):
    """⭐ 拆分的三条结构性性质。"""

    def _run_one(self, seed=5, **cfgkw):
        cfg = MetaOrderConfig(p_meta_start=1.0, participation_rate=1.0, **cfgkw)
        a = _agent(MetaFixed, meta_config=cfg)
        a._rng_meta = np.random.default_rng([seed])
        st = _state()
        a.maybe_start_meta_order("buy", 0)
        self.assertTrue(a.meta_state.active)
        total = a.meta_state.remaining_qty
        issued, sides, kids = 0.0, [], 0
        for t in range(1, 20_000):
            st.tick = t
            out = a.decide(st)
            if out is None:
                if not a.meta_state.active:
                    break
                continue
            issued += float(out.quantity)
            sides.append(out.side)
            kids += 1
            if not a.meta_state.active:
                break
        return a, total, issued, sides, kids

    def test_总量守恒(self) -> None:
        """发出的子单量之和必须等于采样的总量（在浮点容差内）。"""
        a, total, issued, _, kids = self._run_one()
        self.assertGreater(kids, 0)
        self.assertAlmostEqual(issued, total, places=6,
                               msg="子单量之和 != 元订单总量（拆分要么漏要么多）")

    def test_方向全程不翻转(self) -> None:
        """⭐ M24（方向翻转）的正向护栏。

        方向在拆分过程中翻转，等于把长记忆直接抹掉——
        订单符号 ACF 会变成接近 0 甚至负的，而**不会报任何错**。
        """
        for seed in (1, 2, 3, 4):
            _, _, _, sides, _ = self._run_one(seed=seed)
            self.assertTrue(sides, "一个子单都没发出")
            self.assertEqual(set(sides), {"buy"},
                             f"seed={seed} 的子单方向出现了翻转：{set(sides)}")

    def test_卖出方向同样被锁定(self) -> None:
        cfg = MetaOrderConfig(p_meta_start=1.0, participation_rate=1.0)
        a = _agent(MetaFixed, meta_config=cfg)
        a._rng_meta = np.random.default_rng([8])
        a.maybe_start_meta_order("sell", 0)
        st = _state()
        sides = []
        for t in range(1, 5000):
            st.tick = t
            out = a.decide(st)
            if out is not None:
                sides.append(out.side)
            if not a.meta_state.active:
                break
        self.assertTrue(sides)
        self.assertEqual(set(sides), {"sell"})

    def test_一定终止(self) -> None:
        """无论 Pareto 采到多大的总量，拆分都必须停下（两个闸门都要能关）。"""
        for seed in range(12):
            a, total, issued, _, kids = self._run_one(seed=seed, max_child_orders=30)
            self.assertLessEqual(kids, 30, f"seed={seed} 超过 max_child_orders")
            self.assertFalse(a.meta_state.active, f"seed={seed} 的元订单没有终止")
            # 被 max_child_orders 截断时，发出的量应当 ≤ 总量
            self.assertLessEqual(issued, total + 1e-6)

    def test_子单数量随规模增长(self) -> None:
        """总量越大拆得越多（否则"重尾 → 长记忆"这条链断了：
        拆分数与规模无关，持续时间就没有重尾）。"""
        small = self._run_one(seed=21, pareto_xmin=1.0, pareto_scale=1.0,
                              max_child_orders=10_000, child_qty_mean=1.0)[4]
        big = self._run_one(seed=21, pareto_xmin=1.0, pareto_scale=20.0,
                            max_child_orders=10_000, child_qty_mean=1.0)[4]
        self.assertGreater(big, small * 3)

    def test_剩余量为负要能提前收尾(self) -> None:
        """边界：剩余量比一张子单还小，这张子单应当把它用光并立刻收尾。"""
        cfg = MetaOrderConfig(participation_rate=1.0, child_qty_mean=100.0)
        a = _agent(MetaFixed, meta_config=cfg)
        a._rng_meta = np.random.default_rng([3])
        a.meta_state.start("buy", 0.5, 0)
        child = a.next_child()
        self.assertIsNotNone(child)
        _, qty = child
        self.assertAlmostEqual(qty, 0.5, places=12)
        self.assertFalse(a.meta_state.active)


class TestDirectionSource(unittest.TestCase):
    """方向必须来自**基础决策**，不能由元订单自己编。

    这是本项目「因果测量」要求的一个直接推论：**不要把你打算测量的东西
    当成输入喂进去**。如果我们自己随机生成方向，测出的相关性就是自证。
    """

    def test_方向取自基础决策(self) -> None:
        for want in ("buy", "sell"):
            cls = type(f"MetaFixed_{want}", (MetaOrderMixin, _Fixed), {"WANT": want})
            a = _agent(cls, meta_config=MetaOrderConfig(
                p_meta_start=1.0, participation_rate=1.0))
            a._rng_meta = np.random.default_rng([4])
            st = _state()
            out = a.decide(st)
            self.assertIsNotNone(out)
            # 第一 tick 不额外下单（避免活跃度翻倍），但元订单已被启动
            self.assertTrue(a.meta_state.active, f"want={want} 没有启动元订单")
            self.assertEqual(a.meta_state.direction, want)
            st.tick = 1
            child = a.decide(st)
            self.assertIsNotNone(child)
            self.assertEqual(child.side, want)

    def test_基础类不表态时不启动(self) -> None:
        """基础决策返回 None → 没有"意图"→ 不启动元订单。"""
        class Silent(_Fixed):
            def decide(self, state):
                return None

        class MetaSilent(MetaOrderMixin, Silent):
            pass

        a = _agent(MetaSilent, meta_config=MetaOrderConfig(p_meta_start=1.0))
        a._rng_meta = np.random.default_rng([6])
        self.assertIsNone(a.decide(_state()))
        self.assertFalse(a.meta_state.active)

    def test_双边报价不算有方向的意图(self) -> None:
        """做市商那种"同时挂买卖"的返回净值是零，谈不上方向 → 不启动。"""
        from tw.agents.market_maker import MarketMaker

        class MetaMM(MetaOrderMixin, MarketMaker):
            pass

        a = _agent(MetaMM, meta_config=MetaOrderConfig(p_meta_start=1.0))
        a._rng_meta = np.random.default_rng([6])
        st = _state()
        out = a.decide(st)
        self.assertIsNotNone(out)          # 它确实挂了单
        self.assertFalse(a.meta_state.active, "双边报价竟然被当成了有方向的意图")

    def test_单边返回的顺序不影响方向判定(self) -> None:
        from tw.types import Order

        class TwoSell(_Fixed):
            def decide(self, state):
                return [
                    Order("x", "sell", state.mid * 1.001, 1.0, state.tick, "x#1"),
                    Order("x", "sell", state.mid * 1.002, 1.0, state.tick, "x#2"),
                ]

        class MetaTwoSell(MetaOrderMixin, TwoSell):
            pass

        a = _agent(MetaTwoSell, meta_config=MetaOrderConfig(p_meta_start=1.0))
        a._rng_meta = np.random.default_rng([6])
        a.decide(_state())
        self.assertTrue(a.meta_state.active)
        self.assertEqual(a.meta_state.direction, "sell")


class TestMixinWiring(unittest.TestCase):
    def test_mro顺序正确时mechanism生效(self) -> None:
        """Mixin 必须排在前面。反了会**静默失效**——不报错，只是永远不长记忆。"""
        a = _agent(MetaZI, meta_config=MetaOrderConfig(p_meta_start=1.0))
        self.assertTrue(hasattr(a, "meta_state"))
        m = Market(SimConfig(seed=1, n_ticks=300, population=Population(zero_intel=40)))
        m.add_agent(a)
        m.run(300)
        self.assertGreater(a.n_meta_started, 0, "一整场模拟一个元订单都没启动")

    def test_随机流是懒建的(self) -> None:
        """``__init__`` 里建随机流会拿到未绑定的 seed（一期踩过同一个坑）。"""
        a = _agent(MetaZI, meta_config=MetaOrderConfig())
        self.assertIsNone(a._rng_meta)
        a.meta_rng()
        self.assertIsNotNone(a._rng_meta)

    def test_随机流跨进程可复现(self) -> None:
        """``spawn_rng`` 用 crc32 而不是内置 hash()——后者带进程随机盐。"""
        a1 = _agent(MetaZI)
        a2 = _agent(MetaZI)
        np.testing.assert_allclose(
            a1.meta_rng().random(5), a2.meta_rng().random(5), rtol=0, atol=0
        )

    def test_不注入时行情完全不变(self) -> None:
        """⭐ 解耦检验：元订单机制**完全不动作**时，行情必须逐点不变。

        ⚠️ 基础类必须**真的什么都不做**（返回 None）。
        第一版用的是"能正常交易的零智能 + p_meta_start=0"，结果行情变了——
        但那不是元订单机制的锅，是那个主体**自己在交易**。
        一个"会不会污染市场"的测试，被测对象必须本身是惰性的，
        否则它测的是别的东西。
        """
        class _Silent(_Fixed):
            def decide(self, state):
                return None

        class MetaSilent(MetaOrderMixin, _Silent):
            pass

        def run(with_meta: bool):
            pop = Population.from_shares(
                60, {"zero_intel": 0.30, "fundamentalist": 0.40, "chartist": 0.30}
            )
            m = Market(SimConfig(seed=99, n_ticks=500, population=pop))
            if with_meta:
                m.add_agent(MetaSilent(
                    "zz0000", 6e5, 10.0, np.random.default_rng([99]),
                    meta_config=MetaOrderConfig(p_meta_start=1.0),
                ))
            m.run(500)
            return np.asarray(m.log.mid[: m.tick], dtype=float)

        a, b = run(False), run(True)
        self.assertEqual(a.size, b.size)
        self.assertTrue(
            np.array_equal(a, b),
            f"注入一个完全不动作的元订单主体后行情变了（最大差 "
            f"{float(np.nanmax(np.abs(a - b))):.6f}）——"
            "说明 Mixin 自己消耗了共享随机流",
        )

    def test_尾单方向不会因为finish而丢失(self) -> None:
        """⭐ 回归：``finish()`` 会把 ``direction`` 置 None。

        如果代码顺序写成"先 finish 再读方向"，`str(None)` 得到字符串 ``"None"``，
        ``if side == "buy"`` 判假 → **每个元订单的最后一张子单都反向**。
        不报错，只是订单符号 ACF 被系统性压低（每段同向序列的尾巴被翻掉）。
        """
        cfg = MetaOrderConfig(participation_rate=1.0, pareto_xmin=1.0,
                              pareto_scale=1.0, child_qty_mean=1.0)
        a = _agent(MetaFixed, meta_config=cfg)
        a._rng_meta = np.random.default_rng([31])
        st = _state()
        a.meta_state.start("buy", 3.0, 0)
        last = None
        while a.meta_state.active:
            out = a.next_child()
            if out is not None:
                last = out
        self.assertIsNotNone(last)
        self.assertEqual(last[0], "buy", "最后一张子单的方向丢了")

    def test_结束后读方向会报错(self) -> None:
        from tw.order_flow.meta_order import MetaOrderState

        st = MetaOrderState()
        st.start("buy", 1.0, 0)
        self.assertEqual(st.locked_direction, "buy")
        st.finish()
        with self.assertRaises(RuntimeError):
            _ = st.locked_direction

    def test_归因字段可核对(self) -> None:
        a = _agent(MetaZI, meta_config=MetaOrderConfig(
            p_meta_start=1.0, participation_rate=1.0))
        m = Market(SimConfig(seed=2, n_ticks=200, population=Population(zero_intel=30)))
        m.add_agent(a)
        m.run(200)
        d = a.describe()
        self.assertIn("n_meta_started", d)
        self.assertIn("n_children", d)
        self.assertGreaterEqual(d["n_children"], d["n_meta_started"])


class TestMetaOrderState(unittest.TestCase):
    def test_非法方向被拒(self) -> None:
        with self.assertRaises(ValueError):
            MetaOrderState().start("sideways", 1.0, 0)

    def test_finish清空方向(self) -> None:
        st = MetaOrderState()
        st.start("buy", 5.0, 3)
        st.finish()
        self.assertFalse(st.active)
        self.assertIsNone(st.direction)
        self.assertEqual(st.remaining_qty, 0.0)


class TestExecutionWindow(unittest.TestCase):
    """E6.4 的观测窗口必须长于元订单的最大执行周期（指导书 §3.7 的预判坑）。"""

    def test_最大执行周期上界(self) -> None:
        cfg = MetaOrderConfig(max_child_orders=200, participation_rate=0.15)
        self.assertEqual(cfg.max_execution_ticks(), int(np.ceil(200 / 0.15)))

    def test_期望拆分数随scale线性增长(self) -> None:
        lo = MetaOrderConfig(pareto_scale=1.0, child_qty_mean=1.0).expected_children()
        hi = MetaOrderConfig(pareto_scale=2.0, child_qty_mean=1.0).expected_children()
        self.assertGreater(hi, lo)

    def test_期望拆分数被上限截断(self) -> None:
        cfg = MetaOrderConfig(pareto_scale=1e6, max_child_orders=50)
        self.assertLessEqual(cfg.expected_children(), 50)

    def test_拆一单所需tick与参与率成反比(self) -> None:
        a = MetaOrderConfig(participation_rate=0.5).max_execution_ticks()
        b = MetaOrderConfig(participation_rate=0.1).max_execution_ticks()
        self.assertGreater(b, a)


if __name__ == "__main__":
    unittest.main()
