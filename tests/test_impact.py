"""冲击函数与深度剖面的测量工具（``tw/impact.py``）。

为什么这些纯函数必须测
----------------------
阶段6 的**主验收项**是「冲击幂律指数 k 比阶段3 的 0.61~1.25 更接近 0.5」。
这个结论完全由 ``fit_power_law`` 的两个护栏决定：

· 护栏放宽松 → 拟合会在饱和的数值噪声上给出漂亮的 k，**看起来达标**；
· 护栏太紧 → 明明有信号也报"拟合失败"，把改善说成没改善。

两种错法都不报错，只产出一个数字。所以这里把两道护栏**各自**钉一条测试，
并且用一期真实踩过的那个场景（跨度 0.7%、k=1.72、R²=0.987）作为反例。
"""

from __future__ import annotations

import unittest

import numpy as np

from tw.impact import (
    PowerLawFit,
    cumulative_depth,
    depth_profile_gamma,
    fit_power_law,
    summarize,
    unsaturated,
)


class TestFitPowerLawGuards(unittest.TestCase):
    def test_正常数据能拟合出已知指数(self) -> None:
        """构造 y = 3·x^0.5，应当精确还原 k=0.5（平方根律）。"""
        x = np.array([1.0, 3.0, 10.0, 30.0, 100.0, 300.0])
        rows = [{"delivered_qty": float(xi), "impact": 3.0 * xi ** 0.5} for xi in x]
        fit = fit_power_law(rows, "impact")
        self.assertTrue(fit.ok, fit.reason)
        self.assertAlmostEqual(fit.exponent, 0.5, places=6)
        self.assertGreater(fit.r_squared, 0.999)

    def test_自变量跨度不足要拒绝(self) -> None:
        """⭐ 一期真实场景：瞬时清算饱和后各档成交量 13.3~13.6 手。

        跨度只有 2%，却能在数值噪声上拟合出 ``k=1.72、R²=0.987``。
        护栏必须拦住它——否则"改善了多少"这个结论就是假的。
        """
        x = np.linspace(13.3, 13.6, 7)
        rows = [{"delivered_qty": float(xi), "impact": 5.0 * xi ** 1.72}
                for xi in x]
        fit = fit_power_law(rows, "impact")
        self.assertFalse(fit.ok, "跨度 2% 的饱和数据竟然通过了拟合护栏")
        self.assertIn("跨度不足", fit.reason)
        self.assertTrue(np.isnan(fit.exponent))

    def test_因变量饱和要拒绝(self) -> None:
        """自变量跨了 4 倍，但因变量撞在同一个下限上 → 也要拒绝。"""
        rows = [{"delivered_qty": q, "impact": 100.0}
                for q in (1.0, 2.0, 4.0, 8.0, 16.0)]
        fit = fit_power_law(rows, "impact")
        self.assertFalse(fit.ok)
        self.assertIn("饱和", fit.reason)

    def test_样本不足要拒绝(self) -> None:
        fit = fit_power_law([{"delivered_qty": 1.0, "impact": 1.0}], "impact")
        self.assertFalse(fit.ok)
        self.assertIn("样本不足", fit.reason)

    def test_零与负值被剔除后可能样本不足(self) -> None:
        rows = [{"delivered_qty": q, "impact": v}
                for q, v in ((1.0, 0.0), (2.0, -1.0), (0.0, 5.0), (4.0, 3.0))]
        fit = fit_power_law(rows, "impact")
        self.assertFalse(fit.ok)

    def test_自定义横轴字段(self) -> None:
        x = np.array([1.0, 10.0, 100.0])
        rows = [{"qty": float(xi), "impact": 2.0 * xi ** 0.5} for xi in x]
        fit = fit_power_law(rows, "impact", x_key="qty")
        self.assertTrue(fit.ok, fit.reason)
        self.assertAlmostEqual(fit.exponent, 0.5, places=6)

    def test_离平方根律的距离(self) -> None:
        """``closeness_to_sqrt`` 是阶段6 的主验收量，量纲必须是绝对差。"""
        x = np.array([1.0, 10.0, 100.0, 1000.0])
        for k in (0.5, 1.0, 1.5):
            rows = [{"delivered_qty": float(xi), "impact": 2.0 * xi ** k}
                    for xi in x]
            fit = fit_power_law(rows, "impact")
            self.assertAlmostEqual(fit.closeness_to_sqrt(), abs(k - 0.5), places=5)


class TestUnsaturated(unittest.TestCase):
    def test_只保留吃满的档位(self) -> None:
        rows = [{"fill_ratio": r} for r in (0.4, 0.94, 0.95, 1.0)]
        kept = unsaturated(rows)
        self.assertEqual(len(kept), 2)
        self.assertTrue(all(r["fill_ratio"] >= 0.95 for r in kept))

    def test_阈值可调(self) -> None:
        rows = [{"fill_ratio": r} for r in (0.5, 0.8, 0.9)]
        self.assertEqual(len(unsaturated(rows, threshold=0.75)), 2)


class TestDepthProfile(unittest.TestCase):
    def test_已知幂律剖面能还原指数(self) -> None:
        """构造 D(x) = x^2（真实市场的远端厚形状）→ γ 必须是 2。"""
        d = np.linspace(1.0, 400.0, 400)
        cum = d ** 2.0
        self.assertAlmostEqual(depth_profile_gamma(d, cum, lo=20.0), 2.0, places=5)

    def test_线性剖面gamma为一(self) -> None:
        d = np.linspace(1.0, 400.0, 400)
        self.assertAlmostEqual(depth_profile_gamma(d, d, lo=20.0), 1.0, places=5)

    def test_深度挤在近端时gamma小于一(self) -> None:
        """γ<1 ⟹ 冲击指数 1/γ>1 = 超线性。这正是阶段3 的失配方向。"""
        d = np.linspace(1.0, 400.0, 400)
        cum = d ** 0.78
        self.assertLess(depth_profile_gamma(d, cum, lo=20.0), 1.0)

    def test_分箱累计深度单调不减(self) -> None:
        dist = np.array([1.0, 5.0, 20.0, 100.0, 300.0])
        qty = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        _, cum = cumulative_depth(dist, qty, max_bp=400.0, n_bins=40)
        self.assertTrue(np.all(np.diff(cum) >= -1e-12))

    def test_距离必须是绝对值(self) -> None:
        """⚠️ 踩过的坑：买盘的 (p/mid−1) 是**负数**，不取绝对值会让
        任何阈值都包含全部档位，曲线变成平线，γ ≈ 0。

        ``cumulative_depth`` 内部自己取绝对值，所以传负数也应当得到
        与正数相同的结果——这条测试把该行为钉住。
        """
        qty = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pos = np.array([1.0, 5.0, 20.0, 100.0, 300.0])
        _, c_pos = cumulative_depth(pos, qty, max_bp=400.0, n_bins=40)
        _, c_neg = cumulative_depth(-pos, qty, max_bp=400.0, n_bins=40)
        np.testing.assert_allclose(c_pos, c_neg)

    def test_全部档位都在同一个箱里时不会崩(self) -> None:
        dist = np.array([1.0, 1.0, 1.0])
        qty = np.array([1.0, 1.0, 1.0])
        d, cum = cumulative_depth(dist, qty, max_bp=400.0, n_bins=10)
        self.assertEqual(d.size, cum.size)
        self.assertAlmostEqual(float(cum[-1]), 3.0)


class TestSummarize(unittest.TestCase):
    def test_均值与标准误(self) -> None:
        s = summarize([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(s.n, 4)
        self.assertAlmostEqual(s.mean, 2.5)
        expected_sem = np.std([1, 2, 3, 4], ddof=1) / 2
        self.assertAlmostEqual(s.sem, float(expected_sem), places=12)

    def test_置信区间覆盖真值(self) -> None:
        rng = np.random.default_rng(9)
        s = summarize(rng.normal(0.5, 0.01, 40))
        lo, hi = s.ci95
        self.assertLess(lo, 0.5)
        self.assertGreater(hi, 0.5)
        self.assertTrue(s.covers(0.5))

    def test_空输入不崩(self) -> None:
        s = summarize([])
        self.assertEqual(s.n, 0)
        self.assertTrue(np.isnan(s.mean))

    def test_单样本的区间是nan(self) -> None:
        s = summarize([3.0])
        self.assertEqual(s.mean, 3.0)
        lo, hi = s.ci95
        self.assertTrue(np.isnan(lo) and np.isnan(hi))

    def test_无穷值被剔除(self) -> None:
        s = summarize([1.0, float("inf"), 3.0])
        self.assertEqual(s.n, 2)
        self.assertAlmostEqual(s.mean, 2.0)


class TestPowerLawFitVerdict(unittest.TestCase):
    def test_判语分档(self) -> None:
        self.assertIn("超线性", PowerLawFit(True, exponent=1.5).verdict())
        self.assertIn("线性", PowerLawFit(True, exponent=0.5).verdict())
        self.assertIn("拟合失败", PowerLawFit(False, reason="x").verdict())


if __name__ == "__main__":
    unittest.main()


# ----------------------------------------------------------------------
class TestRefactorEquivalence(unittest.TestCase):
    """⭐ 抽公共实现**不能改变数字**。

    ``fit_exponent`` / ``unsaturated`` / ``profile_gamma`` 原来内联在
    ``scripts/run_stage3.py`` 里，二期为了让阶段6 用同一把尺子，把它们搬到了
    ``tw/impact.py``。搬动时必须证明**行为逐点不变**——
    否则阶段6 里那句"相比阶段3 的 0.61~1.25 更接近 0.5"就失去了基准，
    而**报告里的数字不会有任何变化**（它读的是写死的 JSON），
    没人会发现基准换了。

    下面的参照实现是**原样的旧代码**（从搬运前的源文件里抄下来），
    与 `tw/impact.py` 的新实现逐点对比。
    """

    @staticmethod
    def _legacy_fit_exponent(rows, key, *, min_span=1.5, min_yrange=1.2):
        """搬运前 `scripts/run_stage3.py::fit_exponent` 的原文（逐字）。"""
        x = np.array([r["delivered_qty"] for r in rows], dtype=float)
        y = np.abs(np.array([r[key] for r in rows], dtype=float))
        ok = np.isfinite(x) & (x > 0) & np.isfinite(y) & (y > 0)
        n = int(ok.sum())
        if n < 3:
            return float("nan"), float("nan"), n, "样本不足（有效档位 < 3）"
        xo, yo = x[ok], y[ok]
        if xo.max() / xo.min() < min_span:
            return (float("nan"), float("nan"), n,
                    f"自变量跨度不足（{xo.max() / xo.min():.2f}× < {min_span}×）")
        if yo.max() / yo.min() < min_yrange:
            return (float("nan"), float("nan"), n,
                    f"因变量已饱和（{yo.max() / yo.min():.2f}× < {min_yrange}×）")
        lx, ly = np.log(xo), np.log(yo)
        k, b = np.polyfit(lx, ly, 1)
        ss_res = float(((ly - (k * lx + b)) ** 2).sum())
        ss_tot = float(((ly - ly.mean()) ** 2).sum())
        return (float(k), (1 - ss_res / ss_tot if ss_tot > 0 else float("nan")),
                n, "")

    @staticmethod
    def _legacy_profile_gamma(dist, cum, lo=20.0):
        """搬运前 `profile_gamma` 的原文（逐字）。"""
        ok = (dist >= lo) & (cum > 0)
        if ok.sum() < 3:
            return float("nan")
        return float(np.polyfit(np.log(dist[ok]), np.log(cum[ok]), 1)[0])

    def test_fit_power_law与旧实现逐点一致(self) -> None:
        rng = np.random.default_rng(2026)
        for trial in range(12):
            n = int(rng.integers(3, 9))
            qty = np.sort(rng.uniform(1.0, 100.0, n))
            k_true = float(rng.uniform(0.3, 1.6))
            impact = 3.0 * qty ** k_true * rng.uniform(0.9, 1.1, n)
            rows = [{"delivered_qty": float(q), "x": float(v)}
                    for q, v in zip(qty, impact)]
            lk, lr, ln_, lreason = self._legacy_fit_exponent(rows, "x")
            fit = fit_power_law(rows, "x")
            if not np.isfinite(lk):
                self.assertFalse(fit.ok,
                                 f"trial {trial}: 旧实现拒绝但新实现通过")
                self.assertEqual(fit.n_points, ln_)
                self.assertEqual(fit.reason, lreason)
                continue
            self.assertTrue(fit.ok, f"trial {trial}: 旧实现通过但新实现拒绝"
                                    f"（{fit.reason}）")
            self.assertAlmostEqual(fit.exponent, lk, places=12)
            self.assertAlmostEqual(fit.r_squared, lr, places=12)
            self.assertEqual(fit.n_points, ln_)

    def test_unsaturated与旧实现一致(self) -> None:
        for th in (0.9, 0.95, 0.99):
            rows = [{"fill_ratio": v} for v in
                    (0.0, 0.5, th - 1e-9, th, th + 1e-9, 1.0)]
            legacy = [r for r in rows if r["fill_ratio"] >= 0.95]
            if th == 0.95:
                self.assertEqual(len(unsaturated(rows)), len(legacy))
                self.assertEqual([r["fill_ratio"] for r in unsaturated(rows)],
                                 [r["fill_ratio"] for r in legacy])
            else:
                self.assertEqual(len(unsaturated(rows, threshold=th)),
                                 len([r for r in rows if r["fill_ratio"] >= th]))

    def test_profile_gamma与旧实现逐点一致(self) -> None:
        rng = np.random.default_rng(7)
        for _ in range(10):
            d = np.sort(rng.uniform(1.0, 400.0, 40))
            c = d ** float(rng.uniform(0.2, 2.5))
            a = self._legacy_profile_gamma(d, c, lo=20.0)
            b = depth_profile_gamma(d, c, lo=20.0)
            if np.isnan(a):
                self.assertTrue(np.isnan(b))
            else:
                self.assertAlmostEqual(a, b, places=12)

    def test_stage3仍在用同一实现(self) -> None:
        """阶段3 必须**import** 公共实现，而不是自己留一份副本。"""
        import scripts.run_stage3 as s3
        from tw import impact

        self.assertIs(s3.fit_power_law, impact.fit_power_law)
        self.assertIs(s3.depth_profile_gamma, impact.depth_profile_gamma)
        self.assertIs(s3.cumulative_depth, impact.cumulative_depth)
