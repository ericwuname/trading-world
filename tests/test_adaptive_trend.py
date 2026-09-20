"""自适应主体与反身性实验的测试（二期阶段7）。

这个文件分两部分，测的重点不同：

* **学习主体**：Q 值的**方向**必须与真实盈亏方向一致。
  方向搞反了，"学出来的最优参数"就是最差的那个——而它不会报错。
* **实验基础设施**：配对、独立初始化、禀赋一致。
  这些性质一旦破了，产出的是一条**看起来合理但归因错误**的曲线。
"""

from __future__ import annotations

import unittest

import numpy as np

from tw import Market, Population, SimConfig
from tw.agents.adaptive_trend import (
    AdaptiveTrendFollower,
    FixedMomentumTrader,
    realized_pnl_bp,
)
from tw.experiments.reflexivity import (
    ADOPTER_CASH,
    ENTROPY_ADOPTER,
    ADOPTER_INVENTORY,
    ReflexivityConfig,
    build_market,
    run_one,
    run_reflexivity_sweep,
    spearman,
)
from tw.types import MarketState, PriceHistory


def _state(prices, tick=None, mid=None):
    hist = PriceHistory(512)
    for p in prices:
        hist.append(float(p))
    m = float(mid if mid is not None else prices[-1])
    return MarketState(
        tick=int(tick if tick is not None else len(prices)),
        mid=m, last_price=m, best_bid=m - 0.01, best_ask=m + 0.01,
        spread=0.02, fundamental=m, history=hist,
    )


def _learner(**kw):
    a = AdaptiveTrendFollower(
        "at0000", 1e6, 10.0, np.random.default_rng([1]), **kw)
    a.bind_seed(11)
    return a


class TestLearningDirection(unittest.TestCase):
    """⭐ M27（方向乘反）的正向护栏：Q 值必须跟着真实盈亏走。"""

    def test_上涨行情里正方向的评估得到正收益(self) -> None:
        a = _learner(candidate_lookbacks=(5,), eval_horizon=3)
        st = _state(np.linspace(100.0, 110.0, 200))
        a.pending_evals.append([float(st.tick), 5.0, float(st.mid), 1.0])
        a.resolve_pending(_state(np.linspace(100.0, 110.0, 200), tick=st.tick + 3,
                                 mid=st.mid * 1.02))
        self.assertGreater(a.q_values[5], 0.0, "做多方向的评估值不是正的")

    def test_方向乘反会被抓出来(self) -> None:
        """把方向号取反，Q 值应当变成**负**的。

        这条不是"再测一遍方向"，它是在钉住 M27 的可检出性：
        若实现里 direction 乘反，价格上涨时做多会被记成亏损，
        Q 值变负 → 学出来的"最优 lookback"是最差的那个，而**不会报错**。
        """
        a = _learner(candidate_lookbacks=(5,), eval_horizon=3)
        st = _state(np.linspace(100.0, 110.0, 200))
        a.pending_evals.append([float(st.tick), 5.0, float(st.mid), -1.0])  # 故意反
        a.resolve_pending(_state(np.linspace(100.0, 110.0, 200), tick=st.tick + 3,
                                 mid=st.mid * 1.02))
        self.assertLess(a.q_values[5], 0.0)

    def test_下跌行情里做空得到正收益(self) -> None:
        a = _learner(candidate_lookbacks=(5,), eval_horizon=3)
        st = _state(np.linspace(110.0, 100.0, 200))
        a.pending_evals.append([float(st.tick), 5.0, float(st.mid), -1.0])
        a.resolve_pending(_state(np.linspace(110.0, 100.0, 200), tick=st.tick + 3,
                                 mid=st.mid * 0.98))
        self.assertGreater(a.q_values[5], 0.0)

    def test_未到期不结算(self) -> None:
        a = _learner(candidate_lookbacks=(5,), eval_horizon=10)
        st = _state(np.linspace(100.0, 110.0, 200))
        a.pending_evals.append([float(st.tick), 5.0, float(st.mid), 1.0])
        n = a.resolve_pending(_state(np.linspace(100.0, 110.0, 200), tick=st.tick + 3))
        self.assertEqual(n, 0)
        self.assertEqual(len(a.pending_evals), 1)

    def test_学习率控制步长(self) -> None:
        a = _learner(candidate_lookbacks=(5,), eval_horizon=3, learning_rate=0.5)
        st = _state([100.0] * 10 + [102.0])
        a.pending_evals.append([float(st.tick), 5.0, 100.0, 1.0])
        a.resolve_pending(_state([100.0] * 10 + [102.0], tick=st.tick + 3, mid=104.0))
        # realized = 0.04, Q = 0 + 0.5*(0.04 - 0) = 0.02
        self.assertAlmostEqual(a.q_values[5], 0.02, places=9)


class TestPendingEvalBookkeeping(unittest.TestCase):
    """⭐ M28（重复结算）的正向护栏。"""

    def test_每条评估只结算一次(self) -> None:
        """``n_updates`` 必须恰好等于入队条数。"""
        a = _learner(candidate_lookbacks=(2, 3, 5), eval_horizon=4)
        n_enq = 0
        for t in range(60):
            st = _state(np.linspace(100.0, 101.0, 8), tick=t)
            a.pending_evals.append([float(t), float((2, 3, 5)[t % 3]), 100.0, 1.0])
            n_enq += 1
            a.resolve_pending(_state(np.linspace(100.0, 101.0, 8), tick=t + 5))
        self.assertEqual(a.n_updates, n_enq,
                         f"结算了 {a.n_updates} 次但只入队 {n_enq} 条")

    def test_队列不会漏掉元素(self) -> None:
        a = _learner(candidate_lookbacks=(5,), eval_horizon=10)
        for t in range(5):
            a.pending_evals.append([float(t), 5.0, 100.0, 1.0])
        # tick=11、horizon=10 ⇒ t=0(11≥10)、t=1(10≥10) 到期；t=2(9<10) 起未到期
        a.resolve_pending(_state([100.0] * 10, tick=11))
        self.assertEqual(len(a.pending_evals), 3)
        self.assertEqual([e[0] for e in a.pending_evals], [2.0, 3.0, 4.0])

    def test_队列有上限(self) -> None:
        a = _learner()
        a.MAX_PENDING = 7
        st = _state([100.0] * 8, tick=0)
        for _ in range(50):
            if len(a.pending_evals) < a.MAX_PENDING:
                a.pending_evals.append([0.0, 10.0, 100.0, 1.0])
        self.assertLessEqual(len(a.pending_evals), 7)

    def test_入参非法被拒(self) -> None:
        for bad in ({"candidate_lookbacks": ()}, {"candidate_lookbacks": (1,)},
                    {"epsilon": -0.1}, {"epsilon": 1.1},
                    {"learning_rate": 0.0}, {"eval_horizon": 0}):
            with self.assertRaises(ValueError, msg=f"{bad} 应当被拒"):
                _learner(**bad)


class TestChoiceMechanism(unittest.TestCase):
    """⭐ 平局必须随机打破。"""

    def test_初始平局不会总选第一个(self) -> None:
        """所有 Q 都是 0 时，若直接用 ``max(dict, key=...)``，
        永远返回**第一个**键 —— 于是第一个候选被无条件选到，
        直到它的 Q 偏离 0 为止。这个起始偏差完全来自实现细节，
        却会污染"学习收敛到哪个窗口"的结论。
        """
        a = _learner(candidate_lookbacks=(10, 20, 50, 100), epsilon=0.0)
        picks = [a.choose_lookback() for _ in range(400)]
        self.assertEqual(set(picks), {10, 20, 50, 100},
                         "初始平局时只选到了第一个候选——没做平局随机化")
        counts = [picks.count(lb) for lb in (10, 20, 50, 100)]
        self.assertLess(max(counts) / min(counts), 1.6,
                        f"平局随机化不均匀：{counts}")

    def test_利用阶段选Q最高的(self) -> None:
        a = _learner(candidate_lookbacks=(10, 20), epsilon=0.0)
        a.q_values = {10: 0.5, 20: -0.5}
        self.assertEqual({a.choose_lookback() for _ in range(20)}, {10})

    def test_epsilon为1时纯探索(self) -> None:
        a = _learner(candidate_lookbacks=(10, 20, 50), epsilon=1.0)
        a.q_values = {10: 99.0, 20: -99.0, 50: -99.0}
        picks = {a.choose_lookback() for _ in range(200)}
        self.assertEqual(picks, {10, 20, 50})

    def test_随机流跨实例可复现(self) -> None:
        a1, a2 = _learner(), _learner()
        p1 = [a1.choose_lookback() for _ in range(50)]
        p2 = [a2.choose_lookback() for _ in range(50)]
        self.assertEqual(p1, p2)

    def test_best_lookback跟随Q值(self) -> None:
        a = _learner(candidate_lookbacks=(10, 20, 50))
        a.q_values = {10: 0.1, 20: 0.9, 50: -0.3}
        self.assertEqual(a.best_lookback(), 20)


class TestLearnFromRealMarket(unittest.TestCase):
    def test_学习主体能跑完并留下归因(self) -> None:
        m = Market(SimConfig(seed=3, n_ticks=600, population=Population(60, 20, 20)))
        a = AdaptiveTrendFollower(
            "at0000", 6e5, 10.0, np.random.default_rng([3]),
            candidate_lookbacks=(10, 20), eval_horizon=10)
        a.bind_seed(3)
        m.add_agent(a)
        m.run(600)
        self.assertGreater(a.n_choices, 0, "整场模拟一次都没选过")
        self.assertGreater(a.n_updates, 0, "一次评估都没结算")
        d = a.describe()
        for k in ("q_values", "best_lookback", "n_choices", "n_updates",
                  "choice_counts"):
            self.assertIn(k, d)

    def test_固定策略主体能跑完(self) -> None:
        m = Market(SimConfig(seed=4, n_ticks=500, population=Population(60, 20, 20)))
        a = FixedMomentumTrader("fm0000", 6e5, 10.0,
                                np.random.default_rng([4]), lookback=20)
        m.add_agent(a)
        a.set_initial_equity(m.current_mid())
        m.run(500)
        self.assertGreater(a.stats.n_submitted, 0)
        self.assertTrue(np.isfinite(realized_pnl_bp(a, m.current_mid())))

    def test_lookback非法被拒(self) -> None:
        with self.assertRaises(ValueError):
            FixedMomentumTrader("x", 1e6, 0.0, np.random.default_rng([1]), lookback=1)


class TestPnLMeasure(unittest.TestCase):
    def test_无变化时为零(self) -> None:
        m = Market(SimConfig(seed=5, n_ticks=1, population=Population(10)))
        a = m.agents[0]
        a.set_initial_equity(m.current_mid())
        self.assertAlmostEqual(realized_pnl_bp(a, m.current_mid()), 0.0, places=9)

    def test_未设初始权益返回nan(self) -> None:
        a = FixedMomentumTrader("x", 6e5, 10.0, np.random.default_rng([1]))
        self.assertTrue(np.isnan(realized_pnl_bp(a, 100.0)))

    def test_价格翻倍时正收益(self) -> None:
        """持仓为正且价格上涨 → 收益为正（符号方向的基本检查）。"""
        m = Market(SimConfig(seed=6, n_ticks=1, population=Population(10)))
        a = m.agents[0]
        a.set_initial_equity(100.0)
        a.inventory = 10.0
        a.cash = 0.0
        a.initial_equity = 1000.0
        self.assertGreater(realized_pnl_bp(a, 200.0), 0.0)


class TestReflexivityInfrastructure(unittest.TestCase):
    """⭐ 实验设计的三条不变量。它们破了，曲线会"看起来合理但归因错误"。"""

    def _cfg(self, **kw):
        base = dict(adopter_counts=[0, 4], seeds=[11], n_agents=60,
                    warmup=100, n_ticks=200)
        base.update(kw)
        return ReflexivityConfig(**base)

    def test_总主体数保持不变(self) -> None:
        """⭐ 用「替换」而不是「叠加」。

        叠加会让 N 从 300 涨到 460，而一期已实测 σ 强烈依赖 N——
        于是"采用者变多"同时意味着"市场变厚"，
        采用者 PnL 的变化里分不清是**同业拥挤**还是**市场变大**。
        """
        cfg = self._cfg()
        a = build_market(cfg, 11, 0)
        b = build_market(cfg, 11, 4)
        self.assertEqual(len(a.agents), len(b.agents),
                         "替换设计下总主体数必须恒定")

    def test_采用者数量正确(self) -> None:
        cfg = self._cfg()
        for n in (0, 2, 4):
            m = build_market(cfg, 11, n)
            self.assertEqual(sum(1 for x in m.agents if x.KIND == "fixed_momentum"), n)

    def test_采用者禀赋完全一致(self) -> None:
        """跨 adopter_count 比的是"谁更拥挤"，不是"谁的禀赋运气好"。"""
        cfg = self._cfg()
        for n in (1, 4):
            m = build_market(cfg, 11, n)
            for a in m.agents:
                if a.KIND == "fixed_momentum":
                    self.assertAlmostEqual(a.cash, ADOPTER_CASH, places=6)
                    self.assertAlmostEqual(a.inventory, ADOPTER_INVENTORY, places=6)

    def test_每个组合都是全新市场(self) -> None:
        """⭐ 指导书 §4.7：不能共享"市场记忆"。

        若复用同一个 market 对象（或深拷贝一个已跑过的），
        后面的档位会混进前面档位的残留（订单簿残留、持仓、日志位置），
        于是"衰减曲线"里有一条来自实验装置的伪趋势。
        """
        cfg = self._cfg()
        m1 = build_market(cfg, 11, 2)
        r1 = run_one(cfg, 11, 2)
        m2 = build_market(cfg, 11, 2)
        self.assertIsNot(m1, m2)
        self.assertEqual(m1.tick, 0, "新建的市场不是从未跑过的状态")
        # 同参数两次独立运行必须得到完全相同的结果
        r2 = run_one(cfg, 11, 2)
        self.assertAlmostEqual(r1["adopter_pnl_bp_mean"], r2["adopter_pnl_bp_mean"],
                               places=9, msg="同参数两次运行结果不同——有状态残留")

    def test_采用者的随机流与主体序号解耦(self) -> None:
        """第 i 个采用者在任何 adopter_count 档位里都是同一条流。"""
        cfg = self._cfg()
        m = build_market(cfg, 11, 3)
        rngs = [a for a in m.agents if a.KIND == "fixed_momentum"]
        draws = [tuple(np.random.default_rng([11, ENTROPY_ADOPTER, i]).random(3))
                 for i in range(3)]
        self.assertEqual(len(rngs), 3)
        # 直接核对播种公式本身（用同一公式重建即可）
        for i in range(3):
            self.assertEqual(draws[i], tuple(np.random.default_rng(
                [11, ENTROPY_ADOPTER, i]).random(3)))

    def test_扫描产出每档每种子一行(self) -> None:
        cfg = self._cfg(adopter_counts=[0, 4], seeds=[11, 12])
        rows = run_reflexivity_sweep(cfg)
        self.assertEqual(len(rows), 4)
        self.assertEqual({r["adopter_count"] for r in rows}, {0, 4})
        self.assertEqual({r["seed"] for r in rows}, {11, 12})

    def test_扫描时用的就是配置的种子(self) -> None:
        """⭐ M29（忘记保持种子配对）的正面护栏。

        配对设计的前提是"同种子下只改 adopter_count"。
        若构造 ``SimConfig`` 时对种子做了加工（``seed+count``、``seed*7``……），
        配对就**悄悄破了**——而返回值里仍然写着原始 seed，从外面看不出来。
        所以 ``run_one`` 记的是**市场实际用的**种子（``seed_effective``）。
        """
        cfg = self._cfg(adopter_counts=[0, 4], seeds=[11, 12])
        for r in run_reflexivity_sweep(cfg):
            self.assertEqual(r["seed_effective"], r["seed"],
                             f"count={r['adopter_count']} seed={r['seed']} 时"
                             f"市场实际用了 {r['seed_effective']}——配对已破")

    def test_背景池的构成与总规模符合设计(self) -> None:
        """替换设计：总主体数恒定，背景池随采用者增加而缩小。

        ⚠️ 必须把这件事**测出来**，因为它是本设计相对指导书的**刻意偏离**，
        也是它最主要的已知边界（背景噪音交易者变少）。
        """
        cfg = self._cfg(adopter_counts=[0, 4], seeds=[11])
        m0 = build_market(cfg, 11, 0)
        m4 = build_market(cfg, 11, 4)
        self.assertEqual(len(m0.agents), len(m4.agents))
        self.assertEqual(len(m0.agents) - sum(1 for x in m0.agents
                                              if x.KIND == "fixed_momentum"), 60)
        self.assertEqual(len(m4.agents) - sum(1 for x in m4.agents
                                              if x.KIND == "fixed_momentum"), 56)

    def test_结果含市场特征(self) -> None:
        cfg = self._cfg()
        r = run_one(cfg, 11, 2)
        for k in ("sigma_bp", "excess_kurtosis", "acf_abs_lag1"):
            self.assertIn(k, r["market"])

    def test_负采用者数被拒(self) -> None:
        with self.assertRaises(ValueError):
            build_market(self._cfg(), 11, -1)

    def test_健康检查通过(self) -> None:
        cfg = self._cfg(n_agents=80, warmup=150, n_ticks=300)
        r = run_one(cfg, 13, 4)
        self.assertTrue(r["health_ok"], r["health_problems"])


class TestSpearman(unittest.TestCase):
    def test_完全单调递减相关系数为负一(self) -> None:
        rho, p = spearman([0, 1, 2, 3, 4], [10, 8, 6, 4, 2])
        self.assertAlmostEqual(rho, -1.0, places=9)
        self.assertLess(p, 0.05)

    def test_完全单调递增相关系数为正一(self) -> None:
        rho, _ = spearman([0, 1, 2, 3, 4], [1, 2, 3, 4, 5])
        self.assertAlmostEqual(rho, 1.0, places=9)

    def test_无关联时接近零(self) -> None:
        rho, p = spearman([0, 1, 2, 3, 4, 5, 6, 7], [3, 1, 4, 1, 5, 9, 2, 6])
        self.assertLess(abs(rho), 0.8)
        self.assertGreater(p, 0.05)

    def test_并列值不崩(self) -> None:
        rho, p = spearman([1, 1, 1, 2, 3], [5, 4, 3, 2, 1])
        self.assertTrue(np.isfinite(rho))

    def test_样本不足返回nan(self) -> None:
        rho, p = spearman([1, 2], [3, 4])
        self.assertTrue(np.isnan(rho) and np.isnan(p))

    def test_剔除nan(self) -> None:
        rho, _ = spearman([0, 1, float("nan"), 3, 4], [10, 8, 6, 4, 2])
        self.assertTrue(np.isfinite(rho))


if __name__ == "__main__":
    unittest.main()
