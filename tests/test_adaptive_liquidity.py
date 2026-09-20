"""自适应流动性的测试（二期阶段6，LMF 第二环）。

这个文件里最要紧的是一条**不变量**和一条**不对称性**：

1. **不变量：只许推远，不许拉近。** 允许负的 shift 会让流动性提供者
   主动往对手方向让价，自己造出跨价套利机会。这在模型里表现为
   价差系统性变负、盘口自我吞噬，而深度剖面**反而变平**——
   于是"流动性变薄"这个结论会被一个相反的事实盖住。
2. **不对称性：只有"将来会被打的那一侧"往外推。** 两侧一起推等于
   "价差整体变宽"，那是波动率上升的另一种表现，测的不是同一个机制。
"""

from __future__ import annotations

import unittest

import numpy as np

from tw import Market, Population, SimConfig
from tw.agents.adaptive_liquidity import AdaptiveLiquidityMixin
from tw.agents.market_maker import MarketMaker
from tw.agents.zero_intel import ZeroIntelligence
from tw.types import FlowView, MarketState, PriceHistory


class AdaptiveZI(AdaptiveLiquidityMixin, ZeroIntelligence):
    pass


class AdaptiveMM(AdaptiveLiquidityMixin, MarketMaker):
    pass


def _state(imbalance: float = 0.0, mid: float = 100.0, tick: int = 100,
           window: int = 50) -> MarketState:
    """构造一个带指定订单流失衡的 state。

    直接用 ``FlowView`` 灌一个真实数组，而不是手工设 ``flow_imbalance`` 字段——
    前者测的是"主体真正会走的那条路径"，后者只测字段本身。
    """
    flow = np.zeros(400, dtype=np.float64)
    if imbalance != 0.0:
        # 在窗口内均匀灌入：n 个 +1 与 m 个 −1，使净失衡 = target
        # (n−m)/(n+m) = target ⇒ 取 n+m = window
        w = int(window)
        n_plus = int(round(w * (1.0 + imbalance) / 2.0))
        n_plus = max(0, min(w, n_plus))
        flow[tick - w: tick - w + n_plus] = 1.0
        flow[tick - w + n_plus: tick] = -1.0
    hist = PriceHistory(16)
    hist.append(mid)
    return MarketState(
        tick=tick, mid=mid, last_price=mid, best_bid=mid - 0.01,
        best_ask=mid + 0.01, spread=0.02, fundamental=mid, history=hist,
        flow=FlowView(flow, tick),
    )


class TestFlowView(unittest.TestCase):
    def test_净失衡计算(self) -> None:
        flow = np.array([1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0])
        v = FlowView(flow, 8)
        self.assertAlmostEqual(v.imbalance(8), float(flow.sum() / np.abs(flow).sum()))
        self.assertAlmostEqual(v.imbalance(8), 2.0 / 8.0)

    def test_窗口只看最近的(self) -> None:
        flow = np.array([100.0, 100.0, -1.0, 1.0])
        v = FlowView(flow, 4)
        # 只看最后 2 个：−1, +1 → 净 0
        self.assertAlmostEqual(v.imbalance(2), 0.0)

    def test_full_buy_imbalance(self) -> None:
        flow = np.ones(10) * 2.0
        self.assertAlmostEqual(FlowView(flow, 10).imbalance(10), 1.0)

    def test_full_sell_imbalance(self) -> None:
        flow = -np.ones(10) * 2.0
        self.assertAlmostEqual(FlowView(flow, 10).imbalance(10), -1.0)

    def test_空窗口返回0(self) -> None:
        self.assertEqual(FlowView(np.zeros(5), 0).imbalance(10), 0.0)

    def test_带nan的未写入段被忽略(self) -> None:
        flow = np.array([np.nan, np.nan, 1.0, 1.0])
        self.assertAlmostEqual(FlowView(flow, 4).imbalance(4), 1.0)

    def test_signed与count(self) -> None:
        flow = np.array([0.0, 2.0, -1.0, 0.0])
        v = FlowView(flow, 4)
        self.assertAlmostEqual(v.signed(4), 1.0)
        self.assertEqual(v.count(4), 2)

    def test_窗口超过历史不崩(self) -> None:
        v = FlowView(np.array([1.0, 1.0]), 2)
        self.assertAlmostEqual(v.imbalance(10_000), 1.0)

    def test_与Market内部口径一致(self) -> None:
        """⭐ 两条实现必须同口径。

        ``Market._flow_imbalance(window)`` 与 ``FlowView.imbalance(window)``
        是同一个公式的两次实现：一处服务资金费率的拥挤度通道，
        一处服务主体的自适应流动性。**两处口径必须一致**——
        否则"标定时的拥挤度"和"主体看到的拥挤度"是两个不同的量，
        而标定结论会被悄悄搬到另一个量上。
        """
        rng = np.random.default_rng(5)
        flow = rng.normal(0.0, 1.0, 500)
        tick = 400
        view = FlowView(flow, tick)
        for w in (1, 7, 50, 199, 200, 400):
            lo = max(0, tick - w)
            seg = flow[lo:tick]
            want = float(np.nansum(seg) / np.nansum(np.abs(seg)))
            self.assertAlmostEqual(view.imbalance(w), want, places=12,
                                   msg=f"window={w} 两处口径不一致")
        # 还要和 Market 的私有实现一致
        m = Market(SimConfig(seed=1, n_ticks=60, population=Population(zero_intel=30)))
        m.run(60)
        for w in (10, 60):
            self.assertAlmostEqual(
                FlowView(m.log.flow, m.tick).imbalance(w),
                m._flow_imbalance(window=w), places=12,
            )


class TestOnlyPushAway(unittest.TestCase):
    """⭐ 不变量：只许推远，不许拉近。"""

    def _agent(self, **kw):
        a = AdaptiveZI("a0000", 1e6, 10.0, np.random.default_rng([1]), **kw)
        a.bind_seed(7)
        return a

    def test_买单在买压下一动不动(self) -> None:
        """别人在买 → 我的买单不吃亏 → 不推。"""
        a = self._agent()
        st = _state(imbalance=+1.0)
        self.assertAlmostEqual(a.adjusted_offset(0.001, st, "buy"), 0.001)

    def test_卖单在卖压下一动不动(self) -> None:
        a = self._agent()
        st = _state(imbalance=-1.0)
        self.assertAlmostEqual(a.adjusted_offset(0.001, st, "sell"), 0.001)

    def test_卖单在买压下被推远(self) -> None:
        a = self._agent()
        st = _state(imbalance=+1.0)
        out = a.adjusted_offset(0.001, st, "sell")
        self.assertGreater(out, 0.001)

    def test_买单在卖压下被推远(self) -> None:
        a = self._agent()
        st = _state(imbalance=-1.0)
        out = a.adjusted_offset(0.001, st, "buy")
        self.assertGreater(out, 0.001)

    def test_任何输入都不会拉近(self) -> None:
        """⭐ 这条是 M25（允许负 shift）的正面护栏。

        穷举 imbalance × side × base_offset 的组合，输出必须**恒** ≥ 输入。
        踩过的后果：允许拉近时流动性提供者会主动往对手方向让价，
        价差系统性变负、盘口自我吞噬，而深度剖面**反而变平**——
        "流动性变薄"这个结论被一个相反的事实盖住。
        """
        a = self._agent()
        for imb in np.linspace(-1.0, 1.0, 41):
            st = _state(imbalance=float(imb))
            for side in ("buy", "sell"):
                for base in (1e-6, 1e-4, 1e-3, 0.01, 0.1):
                    out = a.adjusted_offset(base, st, side)
                    self.assertGreaterEqual(
                        out, base - 1e-18,
                        f"imb={imb:.2f} side={side} base={base} 被拉近到 {out}",
                    )

    def test_敏感度为0时完全退化成基类(self) -> None:
        a = self._agent(flow_sensitivity=0.0)
        for imb in (-1.0, 0.0, 1.0):
            st = _state(imbalance=imb)
            for side in ("buy", "sell"):
                self.assertAlmostEqual(a.adjusted_offset(0.002, st, side), 0.002)

    def test_放大倍数被上限约束(self) -> None:
        """没有上限时，敏感度一大报价就被推出盘口，主体静默退出市场——
        那测出来的是"主体消失了"，不是"流动性变薄了"。"""
        a = self._agent(flow_sensitivity=100.0, max_stretch=3.0)
        st = _state(imbalance=1.0)
        self.assertAlmostEqual(a.adjusted_offset(0.001, st, "sell"), 0.003)

    def test_推的幅度随失衡单调(self) -> None:
        a = self._agent(flow_sensitivity=0.5, max_stretch=100.0)
        outs = [a.adjusted_offset(0.001, _state(imbalance=i), "sell")
                for i in (0.0, 0.25, 0.5, 0.75, 1.0)]
        self.assertTrue(all(outs[i] <= outs[i + 1] + 1e-18 for i in range(4)))

    def test_零或负基准直接返回(self) -> None:
        a = self._agent()
        st = _state(imbalance=1.0)
        self.assertEqual(a.adjusted_offset(0.0, st, "sell"), 0.0)
        self.assertEqual(a.adjusted_offset(-0.001, st, "sell"), -0.001)

    def test_非法侧要报错(self) -> None:
        a = self._agent()
        with self.assertRaises(ValueError):
            a.adjusted_offset(0.001, _state(imbalance=1.0), "sideways")

    def test_非法参数被拒(self) -> None:
        with self.assertRaises(ValueError):
            self._agent(flow_sensitivity=-0.1)
        with self.assertRaises(ValueError):
            self._agent(flow_window=0)
        with self.assertRaises(ValueError):
            self._agent(max_stretch=0.5)


class TestAsymmetry(unittest.TestCase):
    """⭐ 不对称性：推的是"将被吃的那一侧"，不是两侧一起推。"""

    def _agent(self, **kw):
        a = AdaptiveZI("a0000", 1e6, 10.0, np.random.default_rng([2]), **kw)
        a.bind_seed(7)
        return a

    def test_买压下只有卖侧被推(self) -> None:
        a = self._agent()
        st = _state(imbalance=+0.8)
        self.assertAlmostEqual(a.adjusted_offset(0.001, st, "buy"), 0.001)
        self.assertGreater(a.adjusted_offset(0.001, st, "sell"), 0.001)

    def test_卖压下只有买侧被推(self) -> None:
        a = self._agent()
        st = _state(imbalance=-0.8)
        self.assertGreater(a.adjusted_offset(0.001, st, "buy"), 0.001)
        self.assertAlmostEqual(a.adjusted_offset(0.001, st, "sell"), 0.001)

    def test_做市商两侧价差被单侧拉开(self) -> None:
        """做市商：买压下 ask 侧半价差必须比 bid 侧宽。

        如果写成两侧一起推，bid/ask 会同步变宽——那是"价差整体变厚"，
        属于波动率上升的另一种表现，与"单侧流动性变薄"不是同一件事。
        """
        a = AdaptiveMM("mm0000", 1e7, 10.0, np.random.default_rng([3]))
        a.bind_seed(11)
        st = _state(imbalance=+0.9)
        st.history.append(st.mid)
        bid_frac = a.quote_half_spread_frac(st, "buy")
        ask_frac = a.quote_half_spread_frac(st, "sell")
        self.assertGreater(ask_frac, bid_frac,
                           "买压下卖侧没有比买侧更宽——两侧被同步推了")


class TestWiring(unittest.TestCase):
    def test_零智能挂载点生效(self) -> None:
        a = AdaptiveZI("a0000", 1e6, 10.0, np.random.default_rng([4]),
                       offset_range=(0.001, 0.002))
        a.bind_seed(7)
        m = Market(SimConfig(seed=3, n_ticks=400, population=Population(zero_intel=50)))
        m.add_agent(a)
        m.run(400)
        self.assertGreater(a.n_stretched, 0, "整场模拟一次都没推开过报价")

    def test_做市商挂载点生效(self) -> None:
        a = AdaptiveMM("mm0000", 1e7, 10.0, np.random.default_rng([5]))
        a.bind_seed(7)
        m = Market(SimConfig(seed=3, n_ticks=400, population=Population(zero_intel=50)))
        m.add_agent(a)
        m.run(400)
        self.assertGreater(a.n_stretched, 0, "做市商整场都没推开过报价")

    def test_敏感度为0时行情与裸基类逐点一致(self) -> None:
        """⭐ 解耦检验：机制关掉时必须**逐点**退回基类行为。

        ⚠️ 对照组是「**同样注入一个裸 ZeroIntelligence**」，不是「不注入」。
        第一版写成跟"不注入"比，结果行情差了几千——但那不是 Mixin 的锅，
        是那个主体**本来就在交易**（注入任何主体都会改变行情）。
        要证明"这个 Mixin 自己不引入任何行为"，就得让它和有它/没它
        但**其余完全相同**的两场去比。
        """
        def run(adaptive: bool):
            pop = Population.from_shares(
                60, {"zero_intel": 0.30, "fundamentalist": 0.40, "chartist": 0.30}
            )
            m = Market(SimConfig(seed=41, n_ticks=400, population=pop))
            cls = AdaptiveZI if adaptive else ZeroIntelligence
            kw = {"flow_sensitivity": 0.0} if adaptive else {}
            m.add_agent(cls("ad0000", 6e5, 10.0, np.random.default_rng([41]), **kw))
            m.run(400)
            return np.asarray(m.log.mid[: m.tick], dtype=float)

        a, b = run(False), run(True)
        self.assertEqual(a.size, b.size)
        self.assertTrue(
            np.array_equal(a, b),
            f"敏感度为 0 时行情与裸基类不同（最大差 "
            f"{float(np.nanmax(np.abs(a - b))):.6f}）——"
            "Mixin 在敏感度为 0 时仍然改变了行为（大概率是多消耗了一条随机流）",
        )

    def test_敏感度大于0时行情确实改变了(self) -> None:
        """上一条的对照组：机制真的打开时，行情**必须**变。

        两条合起来才有信息量：只测"关掉时不变"无法排除"Mixin 根本没接上"。
        """
        def run(sens: float):
            pop = Population.from_shares(
                60, {"zero_intel": 0.30, "fundamentalist": 0.40, "chartist": 0.30}
            )
            m = Market(SimConfig(seed=41, n_ticks=400, population=pop))
            m.add_agent(AdaptiveZI("ad0000", 6e5, 10.0, np.random.default_rng([41]),
                                   flow_sensitivity=sens))
            m.run(400)
            return np.asarray(m.log.mid[: m.tick], dtype=float)

        a, b = run(0.0), run(0.5)
        self.assertEqual(a.size, b.size)
        self.assertFalse(np.array_equal(a, b), "打开敏感度后行情一点没变——机制没接上")

    def test_归因字段(self) -> None:
        a = AdaptiveZI("a0000", 1e6, 10.0, np.random.default_rng([6]))
        a.bind_seed(7)
        d = a.describe()
        for k in ("n_stretched", "mean_stretch", "flow_window", "flow_sensitivity"):
            self.assertIn(k, d)


class TestMarketStateFallback(unittest.TestCase):
    def test_没有flow视图时退回预计算字段(self) -> None:
        a = AdaptiveZI("a0000", 1e6, 10.0, np.random.default_rng([8]))
        a.bind_seed(7)
        hist = PriceHistory(8)
        hist.append(100.0)
        st = MarketState(100.0 and 0, 100.0, 100.0, 99.0, 101.0, 2.0, 100.0, hist,
                         flow_imbalance=-0.5, flow=None)
        self.assertAlmostEqual(a.flow_imbalance(st), -0.5)

    def test_有时优先用flow视图(self) -> None:
        a = AdaptiveZI("a0000", 1e6, 10.0, np.random.default_rng([8]))
        a.bind_seed(7)
        st = _state(imbalance=+1.0)
        st.flow_imbalance = -1.0        # 故意与视图矛盾
        self.assertAlmostEqual(a.flow_imbalance(st), 1.0)


if __name__ == "__main__":
    unittest.main()
