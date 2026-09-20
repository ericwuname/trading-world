"""Hawkes 自激过程的测试（二期阶段8）。

三条必须钉住的性质：

1. **递推实现 == 定义式。** 市场用 O(1) 递推，测试用指导书的逐步求和。
   两者不一致就等于"优化时换了个东西"，而统计特征会悄悄变。
2. **稳定条件 α < β 必须拦住。** 不拦的后果是强度指数爆炸；
   它不会立刻报错，只会让模拟越跑越慢直到溢出。
3. **ρ 的均值必须是 1。** 这是"打开 Hawkes 不改变平均活跃度"的前提，
   也是 E8.2「有没有额外贡献」能成立的地基。
"""

from __future__ import annotations

import unittest

import numpy as np

from tw.analyzer import ljung_box
from tw.order_flow.hawkes import (
    HawkesConfig,
    HawkesProcess,
    rho_to_count,
)
from tw.order_flow.hawkes_market import HawkesMarket
from tw import Market, Population, SimConfig


class TestStabilityGuard(unittest.TestCase):
    """⭐ M30（α ≥ β 未被拦下）的正面护栏。"""

    def test_alpha大于等于beta被拒(self) -> None:
        for alpha, beta in ((0.2, 0.2), (0.3, 0.2), (1.0, 0.5)):
            with self.assertRaises(ValueError, msg=f"α={alpha} β={beta} 应当被拒"):
                HawkesProcess(mu=1.0, alpha=alpha, beta=beta)

    def test_alpha小于beta可以构造(self) -> None:
        p = HawkesProcess(mu=1.0, alpha=0.1, beta=0.5)
        self.assertAlmostEqual(p.branching_ratio, 0.2)
        self.assertAlmostEqual(p.stationary_mean, 1.0 / 0.8)

    def test_非法基础参数被拒(self) -> None:
        for kw in ({"mu": 0.0, "alpha": 0.1, "beta": 0.5},
                   {"mu": -1.0, "alpha": 0.1, "beta": 0.5},
                   {"mu": 1.0, "alpha": -0.1, "beta": 0.5},
                   {"mu": 1.0, "alpha": 0.1, "beta": 0.0}):
            with self.assertRaises(ValueError, msg=f"{kw} 应当被拒"):
                HawkesProcess(**kw)

    def test_config的branching必须小于一(self) -> None:
        with self.assertRaises(ValueError):
            HawkesConfig(branching=1.0)
        with self.assertRaises(ValueError):
            HawkesConfig(branching=1.5)
        with self.assertRaises(ValueError):
            HawkesConfig(branching=-0.1)

    def test_分支比上限由alpha小于beta决定(self) -> None:
        """⭐ 指导书的网格会踩到这条边界。

        直觉上"分支比 < 1 就稳定"，但还要同时满足 ``α < β``，而后者更紧：

            α = n·(e^{β·dt} − 1) < β  ⇒  n < β/(e^{β·dt} − 1)

        β=0.5 时上限只有 **0.771**——所以 (branching=0.9, beta=0.5) 是不合法的。
        实测踩到过：E8.1 的参数扫描跑到一半抛 ValueError，阶段8 整个中断。
        现在这条检查被提前到**配置构造时**，调用方可以及早跳过。
        """
        for beta, want in ((0.05, 0.975), (0.15, 0.927), (0.5, 0.771)):
            cfg = HawkesConfig(base_lambda=10.0, branching=0.1, beta=beta)
            self.assertAlmostEqual(cfg.max_branching, want, delta=0.002,
                                   msg=f"β={beta} 的上限不符")

    def test_超过上限在构造时就被拒(self) -> None:
        with self.assertRaises(ValueError, msg="branching=0.9, beta=0.5 应当被拒"):
            HawkesConfig(base_lambda=10.0, branching=0.9, beta=0.5)

    def test_上限之内可以构造(self) -> None:
        cfg = HawkesConfig(base_lambda=10.0, branching=0.7, beta=0.5)
        self.assertLess(cfg.alpha, cfg.beta)
        p = cfg.to_process()
        self.assertLess(p.branching_ratio, 1.0)

    def test_config非法参数被拒(self) -> None:
        for kw in ({"base_lambda": 0.0}, {"base_lambda": -5.0},
                   {"beta": 0.0}, {"clamp_rho": 0.0}):
            with self.assertRaises(ValueError, msg=f"{kw} 应当被拒"):
                HawkesConfig(**kw)

    def test_稳定性意味着强度不爆炸(self) -> None:
        """把强度推到远超平稳值之后，它必须**回落**。

        这是"稳定"的可操作定义：扰动会衰减，不会自持。
        """
        p = HawkesProcess(mu=0.1, alpha=0.5, beta=1.0)   # n = 0.5 < 1
        lam = p.advance(1.0, 1000.0)                     # 巨大扰动
        first = lam
        for _ in range(60):
            lam = p.advance(1.0, 0.0)
        self.assertLess(lam, first * 0.01, "强度没有回落——过程不稳定")


class TestRecursionMatchesDefinition(unittest.TestCase):
    """⭐ 递推实现必须与指导书的定义式逐点一致。"""

    def test_无事件时两者一致(self) -> None:
        p = HawkesProcess(mu=2.0, alpha=0.1, beta=0.5)
        for t in range(1, 20):
            self.assertAlmostEqual(p.advance(1.0, 0.0), p.intensity_at(t), places=9)

    def test_有事件时两者一致(self) -> None:
        p = HawkesProcess(mu=2.0, alpha=0.1, beta=0.5)
        events = [3.0, 0.0, 7.0, 0.0, 0.0, 1.0, 0.0, 5.0]
        for i, ev in enumerate(events, start=1):
            got = p.advance(1.0, ev)
            # ⚠️ 两条约定都要对上：
            # ① 事件记在**区间起点**（第 i 步的事件发生在时刻 i−1）。
            #    记成终点会让递推与定义式差一个 e^{−β·dt}（β=0.5 时 60%），
            #    而它不会报错。
            # ② `advance` 收到的是事件**数**，而定义式里每个 entry 是 1 个事件，
            #    所以要 append `int(ev)` 份（ev=3 就 append 3 次）。
            for _ in range(int(ev)):
                p.event_times.append(float(i - 1))
            self.assertAlmostEqual(
                got, p.intensity_at(float(i)), places=9,
                msg=f"第 {i} 步递推与定义式不一致",
            )

    def test_分数步长也一致(self) -> None:
        p = HawkesProcess(mu=1.0, alpha=0.3, beta=0.8)
        t = 0.0
        for i in range(30):
            ev = 1.0 if i % 4 == 0 else 0.0
            if ev > 0:
                p.event_times.append(t)      # 事件记在区间起点
            got = p.advance(0.25, ev)
            t += 0.25
            self.assertAlmostEqual(got, p.intensity_at(t), places=8)

    def test_递推不会累积误差到可见(self) -> None:
        """长序列（2000 步）之后两者仍要一致。"""
        p = HawkesProcess(mu=5.0, alpha=0.05, beta=0.3)
        rng = np.random.default_rng(3)
        for i in range(1, 2001):
            ev = 1.0 if rng.random() < 0.4 else 0.0
            if ev > 0:
                p.event_times.append(float(i - 1))   # 事件记在区间起点
            got = p.advance(1.0, ev)
        self.assertAlmostEqual(got, p.intensity_at(2000.0), places=7)


class TestShouldFire(unittest.TestCase):
    def test_dt过大要报错而不是静默截断(self) -> None:
        """⭐ 静默截断的症状是"到了高聚集区反而没有聚集"，极难定位。"""
        p = HawkesProcess(mu=10.0, alpha=0.1, beta=0.5)
        with self.assertRaises(ValueError):
            p.should_fire(0.0, 1.0, np.random.default_rng([1]))

    def test_dt非法要报错(self) -> None:
        p = HawkesProcess(mu=1.0, alpha=0.1, beta=0.5)
        with self.assertRaises(ValueError):
            p.should_fire(0.0, 0.0, np.random.default_rng([1]))

    def test_高强度时事件率更高(self) -> None:
        """把强度拉高，相同 dt 下的点火频率必须更高。"""
        def rate(mu: float, n: int = 4000) -> float:
            p = HawkesProcess(mu=mu, alpha=0.1, beta=10.0)   # β 大 → 几乎不累积
            rng = np.random.default_rng([7])
            fires = sum(1 for i in range(n)
                        if p.should_fire(i * 0.01, 0.01, rng))
            return fires / n

        self.assertGreater(rate(8.0), rate(1.0))

    def test_事件被记录(self) -> None:
        # λ·dt 必须落在合理区间才能观察到点火：μ=800、dt=1e-3 ⇒ p≈0.8
        p = HawkesProcess(mu=800.0, alpha=100.0, beta=5000.0)
        rng = np.random.default_rng([2])
        n_before = len(p.event_times)
        for i in range(500):
            p.should_fire(i * 0.001, 0.001, rng)
        self.assertGreater(len(p.event_times), n_before)


class TestNormalizedIntensity(unittest.TestCase):
    """⭐ ρ 的均值必须为 1（模块文档 ①③）。"""

    def test_初始时刻rho等于mu除以lambdabar(self) -> None:
        """⚠️ ρ=1 只在**平稳态**成立，不在初始时刻。

        初始没有事件 ⇒ λ = μ，而 ρ = μ/λ̄ = 1 − n。
        把它写成"ρ 恒为 1"是一个常见的误读：μ 是**基础强度**，
        不是平稳均值 λ̄。两者的差就是自激贡献的那部分。
        """
        cfg = HawkesConfig(base_lambda=10.0, branching=0.5, beta=0.2)
        p = cfg.to_process()
        self.assertAlmostEqual(p.normalized_intensity(cfg.base_lambda),
                               1.0 - cfg.branching, places=9)

    def test_长期平均rho接近一(self) -> None:
        """模拟一段带自激的到达序列，ρ 的时间平均必须接近 1。

        这是"打开 Hawkes 不改变平均活跃度"的数值证据。
        """
        cfg = HawkesConfig(base_lambda=20.0, branching=0.7, beta=0.2)
        p = cfg.to_process()
        rng = np.random.default_rng([11])
        vals = []
        for _ in range(20_000):
            rho = p.normalized_intensity(cfg.base_lambda)
            vals.append(rho)
            # 事件量按泊松抽出（均值 = 当前强度）
            events = float(rng.poisson(max(0.0, rho * cfg.base_lambda)))
            p.advance(1.0, events)
        self.assertAlmostEqual(float(np.mean(vals)), 1.0, delta=0.12)

    def test_分支比越高rho波动越大(self) -> None:
        """分支比就是"聚集强度"的旋钮：它必须真的改变分布，而不只是均值。"""
        def spread(branching: float) -> float:
            cfg = HawkesConfig(base_lambda=20.0, branching=branching, beta=0.2)
            p = cfg.to_process()
            rng = np.random.default_rng([13])
            vals = []
            for _ in range(20_000):
                vals.append(p.normalized_intensity(cfg.base_lambda))
                events = float(rng.poisson(max(0.0, vals[-1] * cfg.base_lambda)))
                p.advance(1.0, events)
            return float(np.std(vals, ddof=1))

        self.assertGreater(spread(0.8), spread(0.2))

    def test_分支比为零时rho恒为一(self) -> None:
        cfg = HawkesConfig(base_lambda=10.0, branching=0.0, beta=0.3)
        p = cfg.to_process()
        rng = np.random.default_rng([17])
        for _ in range(2000):
            self.assertAlmostEqual(p.normalized_intensity(cfg.base_lambda), 1.0,
                                   places=9)
            p.advance(1.0, float(rng.poisson(10.0)))

    def test_clamp生效(self) -> None:
        cfg = HawkesConfig(base_lambda=1.0, branching=0.9, beta=0.05, clamp_rho=3.0)
        p = cfg.to_process()
        p.advance(1.0, 10_000.0)
        self.assertAlmostEqual(p.normalized_intensity(cfg.base_lambda, clamp=3.0), 3.0)


class TestRhoToCount(unittest.TestCase):
    def test_均值不变(self) -> None:
        """ρ=1 时必须返回全部主体（期望活跃度 = 原值）。"""
        self.assertEqual(rho_to_count(1.0, 300, clamp=4.0), 300)

    def test_上界被夹住(self) -> None:
        """⭐ 不夹上界会让切片静默返回全部主体，
        于是"超级聚集"这个本该出现的情形**恰好不会出现**。"""
        self.assertEqual(rho_to_count(99.0, 300, clamp=4.0), 300)

    def test_下界被夹住(self) -> None:
        self.assertEqual(rho_to_count(-5.0, 300, clamp=4.0), 0)

    def test_中间值按比例(self) -> None:
        self.assertEqual(rho_to_count(0.5, 300, clamp=4.0), 150)

    def test_零主体返回零(self) -> None:
        self.assertEqual(rho_to_count(1.0, 0), 0)

    def test_结果不超界(self) -> None:
        for rho in np.linspace(-1.0, 10.0, 50):
            k = rho_to_count(float(rho), 100, clamp=4.0)
            self.assertTrue(0 <= k <= 100)


class TestHawkesMarket(unittest.TestCase):
    """市场集成：均值不变 + 聚集性真的出现。"""

    #: ⭐ base_lambda 必须**标定**，不能拍（见 hawkes_market.calibrate_base_lambda）。
    #: 这里在首次调用时测一次并缓存——测试里也要走和生产一样的
    #: "先标定再打开"流程，否则测的是一个生产不会用的配置。
    _BASE: float | None = None

    @classmethod
    def _base(cls, ticks: int = 700) -> float:
        if cls._BASE is None:
            from tw.order_flow.hawkes_market import calibrate_base_lambda
            cls._BASE = calibrate_base_lambda(
                seeds=(3,), n_ticks=ticks, n_agents=200, warmup=100)
        return cls._BASE

    def _run(self, *, branching: float, off: bool = False, ticks: int = 2_000,
             seed: int = 3):
        cfg = HawkesConfig(base_lambda=self._base(), branching=branching,
                           beta=0.15)
        m = HawkesMarket(
            SimConfig(seed=seed, n_ticks=ticks,
                      population=Population.from_shares(
                          200, {"zero_intel": 0.30, "fundamentalist": 0.40,
                                "chartist": 0.30})),
            hawkes_config=cfg, hawkes_off=off,
        )
        m.run(ticks)
        return m

    def test_关掉时不改变一期的行情(self) -> None:
        """⭐ 解耦检验：hawkes_off=True 必须**逐点**等于裸 Market。"""
        def run(use_base: bool):
            cfg = HawkesConfig(base_lambda=self._base(), branching=0.6, beta=0.15)
            sim = SimConfig(seed=3, n_ticks=800,
                            population=Population.from_shares(
                                200, {"zero_intel": 0.30,
                                      "fundamentalist": 0.40, "chartist": 0.30}))
            m = Market(sim) if use_base else HawkesMarket(sim, hawkes_config=cfg,
                                                          hawkes_off=True)
            m.run(800)
            return np.asarray(m.log.mid[: m.tick], dtype=float)

        a, b = run(True), run(False)
        self.assertEqual(a.size, b.size)
        self.assertTrue(np.array_equal(a, b),
                        f"关闭 Hawkes 后行情仍不同（最大差 "
                        f"{float(np.nanmax(np.abs(a - b))):.6f}）")

    def test_买卖双方成交与账本仍然守恒(self) -> None:
        m = self._run(branching=0.7)
        ok, prob = m.health_check()
        self.assertTrue(ok, prob[:5])

    def test_rho的均值接近一(self) -> None:
        """⭐ 打开 Hawkes 不能改变平均活跃度（模块文档 ①）。

        ⚠️ 前提是 ``base_lambda`` 已经**标定**到该市场的实际成交率。
        没标定时 ρ̄ 会系统性偏离 1（实测 λ̄ 取 45 而实际 24 笔/tick 时 ρ̄=0.71），
        那时"聚集性"的结论里就混进了"平均交易量变了"。
        """
        m = self._run(branching=0.7)
        s = m.hawkes_stats()
        self.assertAlmostEqual(s["rho_mean"], 1.0, delta=0.20,
                               msg=f"ρ 均值 {s['rho_mean']:.3f} 偏离 1 太多——"
                                   "'平均活跃度不变'这条前提不成立")

    def test_聚集性真的出现了(self) -> None:
        """高分支比下，逐 tick 成交笔数的自相关必须比低分支比更强。

        这是本阶段的核心命题：**到达的时间聚集性**。
        Ljung-Box 的 Q 统计量越大 = 拒绝"无自相关"的证据越强。
        """
        ev_hi = np.array([d["events"] for d in self._run(branching=0.85).hawkes_log])
        ev_lo = np.array([d["events"] for d in self._run(branching=0.05).hawkes_log])
        q_hi, _ = ljung_box(ev_hi, 10)
        q_lo, _ = ljung_box(ev_lo, 10)
        self.assertGreater(q_hi, q_lo,
                           f"高分支比的聚集性({q_hi:.1f})没有强于低分支比({q_lo:.1f})")

    def test_高分支比下强度序列自相关更强(self) -> None:
        hi = self._run(branching=0.9)
        lo = self._run(branching=0.1)
        xh = np.array([d["lambda"] for d in hi.hawkes_log], dtype=float)
        xl = np.array([d["lambda"] for d in lo.hawkes_log], dtype=float)
        qh, _ = ljung_box(xh, 10)
        ql, _ = ljung_box(xl, 10)
        self.assertGreater(qh, ql)

    def test_归因字段齐全(self) -> None:
        s = self._run(branching=0.5).hawkes_stats()
        for k in ("lambda_mean", "rho_mean", "rho_sd", "events_mean",
                  "n_active_mean", "branching"):
            self.assertIn(k, s)

    def test_期望平稳强度与配置一致(self) -> None:
        """λ 的时间均值应当接近配置的 ``base_lambda``。

        对不上说明"事件量"与"基础强度"的量纲没有对齐，
        此时扫 branchig 会同时改变均值，"聚集性"与"交易量"两个效应分不开。
        """
        m = self._run(branching=0.5, ticks=4_000)
        s = m.hawkes_stats()
        self.assertAlmostEqual(s["lambda_mean"], m.hawkes_cfg.base_lambda,
                               delta=0.45 * m.hawkes_cfg.base_lambda)


class TestLjungBoxOnRealVolumeProxy(unittest.TestCase):
    """真实数据的代理靶子。

    ⚠️ 我们没有真实市场的**逐笔**数据，所以拿不到真正的"到达间隔"。
    最接近的可用代理是 1 小时线的**成交量**（成交量本身就是到达强度×规模）。
    这条测试把代理的口径钉住，并在报告里如实说明它是代理而非等价物。
    """

    def test_真实成交量的成交量聚集显著(self) -> None:
        from tw import realdata

        s = realdata.load_builtin("BTCUSDT_1h")
        lv = np.log(np.maximum(np.asarray(s.volume, dtype=float), 1e-12))
        q, p = ljung_box(lv, 10)
        self.assertGreater(q, 0.0)
        self.assertLess(p, 0.01, "真实成交量竟然没有自相关——代理口径有问题")


if __name__ == "__main__":
    unittest.main()
