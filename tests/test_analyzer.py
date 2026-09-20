"""Analyzer 与真实数据载入的测试。

为什么要测裁判
--------------
这一层是整个项目的结论来源。如果 ACF 算错了、方差比用错了公式、
Hill 估计器的尾部方向搞反了，那么"模拟像不像真实市场"的所有判断都是废的——
而且不会报错，只会给出一个看起来很像结论的错误数字。

所以这里的每一条测试都用**已知答案的合成数据**：
    白噪声的 ACF 应该在置信带内
    AR(1) 的方差比应该大于 1，MA(1) 应该小于 1
    Pareto(α=3) 的 Hill 估计应该收敛到 3
    人为构造"延续"和"反转"两种冲击响应，事件研究必须分别给出正负比值
"""

from __future__ import annotations

import unittest

import numpy as np
from scipy import stats

from tw import realdata
from tw.analyzer import (
    acf,
    acf_ci,
    analyze,
    bar_flow_proxy,
    hill_alpha,
    impact_event_study,
    log_returns,
    variance_ratio,
)


class TestBasicStatistics(unittest.TestCase):
    def test_对数收益率(self) -> None:
        p = np.array([100.0, 110.0, 121.0])
        r = log_returns(p)
        self.assertEqual(r.size, 2)
        np.testing.assert_allclose(r, [np.log(1.1)] * 2, rtol=1e-12)

    def test_对数收益率剔除非法值(self) -> None:
        # 0 价格与 NaN 价格都不允许污染收益率序列（log(0) = -inf）
        p = np.array([100.0, 100.0, 0.0, 121.0, 121.0])
        r = log_returns(p)
        self.assertTrue(np.all(np.isfinite(r)))
        self.assertEqual(r.size, 2)

    def test_白噪声ACF在置信带内(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(size=20_000)
        a = acf(x, 20)
        band = acf_ci(x.size)
        self.assertAlmostEqual(a[0], 1.0, places=12)
        # 20 个滞后里漏出置信带的期望约为 1 个
        self.assertLessEqual(int(np.sum(np.abs(a[1:]) > band)), 3)

    def test_AR1的自相关被正确识别(self) -> None:
        rng = np.random.default_rng(1)
        n, phi = 20_000, 0.8
        e = rng.normal(size=n)
        x = np.empty(n)
        x[0] = e[0]
        for i in range(1, n):
            x[i] = phi * x[i - 1] + e[i]
        a = acf(x, 5)
        self.assertAlmostEqual(a[1], phi, delta=0.03)
        self.assertAlmostEqual(a[2], phi**2, delta=0.04)

    def test_ACF对常数序列不崩(self) -> None:
        a = acf(np.ones(100), 5)
        self.assertTrue(np.all(np.isnan(a[1:])))


class TestVarianceRatio(unittest.TestCase):
    def test_随机游走的方差比接近1(self) -> None:
        rng = np.random.default_rng(2)
        r = rng.normal(0, 0.01, size=50_000)
        vr5, z5 = variance_ratio(r, 5)
        self.assertAlmostEqual(vr5, 1.0, delta=0.03)
        self.assertLess(
            abs(z5), 4.0, "随机游走不该被判为有自相关（z 应落在大样本噪声范围内）"
        )

    def test_正自相关序列方差比大于1(self) -> None:
        rng = np.random.default_rng(3)
        n, phi = 50_000, 0.4
        e = rng.normal(0, 0.01, size=n)
        x = np.empty(n)
        x[0] = e[0]
        for i in range(1, n):
            x[i] = phi * x[i - 1] + e[i]
        vr5, z5 = variance_ratio(x, 5)
        self.assertGreater(vr5, 1.3, "AR(1) 的方差比应显著大于 1")
        self.assertGreater(z5, 3.0)

    def test_均值回归序列方差比小于1(self) -> None:
        rng = np.random.default_rng(4)
        n = 50_000
        e = rng.normal(0, 0.01, size=n)
        x = e[1:] - 0.5 * e[:-1]  # MA(1), θ = -0.5
        vr5, z5 = variance_ratio(x, 5)
        self.assertLess(vr5, 0.8, "MA(1) 负相关序列的方差比应显著小于 1")
        self.assertLess(z5, -3.0)

    def test_序列过短返回NaN(self) -> None:
        vr, z = variance_ratio(np.random.default_rng(0).normal(size=5), 10)
        self.assertTrue(np.isnan(vr))


class TestTailIndex(unittest.TestCase):
    def test_Pareto尾指数估计正确(self) -> None:
        rng = np.random.default_rng(5)
        alpha = 3.0
        u = rng.uniform(size=200_000)
        x = (1.0 - u) ** (-1.0 / alpha)
        est = hill_alpha(x, k=5000)
        self.assertAlmostEqual(est, alpha, delta=0.35, msg="Hill 估计未能复原已知 α")

    def test_尾巴越厚alpha越小(self) -> None:
        rng = np.random.default_rng(6)
        u = rng.uniform(size=100_000)
        thin = (1.0 - u) ** (-1.0 / 5.0)   # α=5
        thick = (1.0 - u) ** (-1.0 / 2.0)  # α=2
        self.assertLess(hill_alpha(thick, 3000), hill_alpha(thin, 3000))


class TestImpactEventStudy(unittest.TestCase):
    """事件研究必须能同时识别"延续"和"反转"两种方向。

    这是本项目最关键的一个判别工具（对应"扫单后继续走而非反转"）。
    如果它只会输出正数、或者方向搞反，整个结论就是错的。
    """

    def _build(self, mode: str, n: int = 8000, step: int = 80):
        rng = np.random.default_rng(7)
        flow = np.zeros(n)
        events = np.arange(100, n - 300, step)
        flow[events] = 5.0
        r = rng.normal(0.0, 0.001, n)
        if mode == "continuation":
            for e in events:
                r[e + 1 : e + 21] += 0.002
        elif mode == "reversal":
            for e in events:
                r[e + 1] += 0.04
                r[e + 2 : e + 21] -= 0.0024
        price = 100.0 * np.exp(np.cumsum(r))
        return price, flow, events

    def test_延续情形识别为正(self) -> None:
        price, flow, events = self._build("continuation")
        out = impact_event_study(price, flow)
        self.assertGreaterEqual(out["n_events"], 0.7 * events.size)
        c = out["curve"]
        self.assertGreater(c[20]["mean_bp"], 0, "延续情形判成了负响应")
        self.assertGreater(c[20]["mean_bp"], c[1]["mean_bp"], "响应曲线没有继续上升")
        self.assertGreater(out["continuation_ratio"], 1.0)

    def test_反转情形识别为负(self) -> None:
        price, flow, events = self._build("reversal")
        out = impact_event_study(price, flow)
        c = out["curve"]
        self.assertGreater(c[1]["mean_bp"], 0, "即时冲击应为正")
        self.assertLess(c[20]["mean_bp"], 0, "反转情形判成了正响应")
        self.assertLess(out["continuation_ratio"], 0.0)

    def test_无冲击时不给结论(self) -> None:
        price = 100 * np.exp(np.cumsum(np.random.default_rng(8).normal(0, 0.001, 5000)))
        out = impact_event_study(price, np.zeros(5000))
        self.assertEqual(out["n_events"], 0)
        self.assertNotIn("curve", out)

    def test_事件去重生效(self) -> None:
        """连续 tick 全部超阈值时，事件数应被最小间隔压下来。"""
        rng = np.random.default_rng(9)
        n = 3000
        price = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
        flow = np.full(n, 10.0)  # 每个 tick 都是巨额冲击
        out = impact_event_study(price, flow, min_gap=5)
        self.assertLessEqual(out["n_events"], n / 5 + 2)
        self.assertGreater(out["n_events"], 0)


class TestBarFlowProxy(unittest.TestCase):
    def test_收在最高价时代理流为正(self) -> None:
        o = np.array([10.0]); h = np.array([12.0]); l = np.array([9.0])
        c = np.array([12.0]); v = np.array([100.0])
        f = bar_flow_proxy(o, h, l, c, v)
        self.assertGreater(f[0], 0)

    def test_收在最低价时代理流为负(self) -> None:
        o = np.array([11.0]); h = np.array([12.0]); l = np.array([9.0])
        c = np.array([9.0]); v = np.array([100.0])
        f = bar_flow_proxy(o, h, l, c, v)
        self.assertLess(f[0], 0)

    def test_零振幅K线不产生除零(self) -> None:
        f = bar_flow_proxy(
            np.array([10.0]), np.array([10.0]), np.array([10.0]),
            np.array([10.0]), np.array([50.0]),
        )
        self.assertTrue(np.all(np.isfinite(f)))
        self.assertEqual(f[0], 0.0)


class TestAnalyzeBundle(unittest.TestCase):
    def test_正态序列的峰度接近0(self) -> None:
        rng = np.random.default_rng(10)
        price = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 30_000)))
        m = analyze(price, "normal")
        self.assertAlmostEqual(m.excess_kurtosis, 0.0, delta=0.25)
        self.assertAlmostEqual(m.vr2, 1.0, delta=0.08)
        self.assertLessEqual(m.n_sig_lags_abs, 4, "白噪声不该表现出波动率聚集")

    def test_肥尾序列的峰度显著为正(self) -> None:
        rng = np.random.default_rng(11)
        r = rng.standard_t(df=3, size=30_000) * 0.005
        price = 100 * np.exp(np.cumsum(r))
        m = analyze(price, "fat-tail")
        self.assertGreater(m.excess_kurtosis, 3.0)
        self.assertLess(m.hill_alpha_left, 5.0)

    def test_波动率聚集被识别(self) -> None:
        """构造 GARCH 型波动率：|r| 的 ACF 必须显著为正。"""
        rng = np.random.default_rng(12)
        n = 30_000
        r = np.zeros(n)
        s = np.full(n, 0.01)
        for i in range(1, n):
            s[i] = np.sqrt(0.00001 + 0.10 * r[i - 1] ** 2 + 0.85 * s[i - 1] ** 2)
            r[i] = s[i] * rng.normal()
        price = 100 * np.exp(np.cumsum(r))
        m = analyze(price, "garch")
        self.assertGreater(m.acf_abs_lag1, 0.05, "GARCH 序列的 |r| ACF 应显著为正")
        self.assertGreater(m.n_sig_lags_abs, 5)
        self.assertLess(abs(m.acf_ret[1]), 0.05, "GARCH 的收益率本身不应有明显自相关")

    def test_序列过短时优雅退化(self) -> None:
        m = analyze(np.array([100.0, 101.0, 102.0]), "tiny")
        self.assertEqual(m.n, 2)
        self.assertTrue(np.isnan(m.excess_kurtosis))


class TestRealData(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.s = realdata.load_builtin("BTCUSDT_1h")

    def test_载入成功且行数正确(self) -> None:
        self.assertGreater(len(self.s), 17_000)
        self.assertEqual(self.s.name, "BTCUSDT_1h")

    def test_时间戳严格递增(self) -> None:
        self.assertTrue(np.all(np.diff(self.s.timestamp) > 0))

    def test_价格有效(self) -> None:
        self.assertTrue(np.all(self.s.close > 0))
        self.assertTrue(np.all(self.s.high >= self.s.low))
        self.assertTrue(np.all(self.s.high >= self.s.close - 1e-6))
        self.assertTrue(np.all(self.s.low <= self.s.close + 1e-6))

    def test_真实BTC具备肥尾与波动率聚集(self) -> None:
        """把目标的量级固定成测试——避免以后参数改了却没人察觉对照基准漂了。"""
        m = analyze(self.s.close, "BTC 1h")
        self.assertGreater(m.excess_kurtosis, 5.0, "真实 BTC 小时线应有明显肥尾")
        self.assertGreater(m.acf_abs_lag1, 0.05, "真实 BTC 应有波动率聚集")
        self.assertGreater(m.n_sig_lags_abs, 5)
        self.assertLess(m.sigma_bp, 200.0)

    def test_其它内置数据集也可载入(self) -> None:
        for key in realdata.available_builtin():
            s = realdata.load_builtin(key)
            self.assertGreater(len(s), 100, f"{key} 数据过少")


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
