"""A10 的测试：守护**标定本身**。

⚠️ 为什么这个文件重要：A10 的整个结论是「装置的区间标定得准」。
如果**标定工具自己**是坏的，那个结论就是空的——
所以这里用**有已知答案的合成数据**反过来检验 Monte Carlo：

- 用**正态残差**时，`monte_carlo` 报出的假阳性率必须 ≈5%、覆盖率必须 ≈95%
  （这是统计学上已知的答案，代码算错就会偏）；
- **固定种子必须完全可复现**（报告里的数字要能被别人复核）；
- `_n_for_power` **不许外推**（超出已测范围要返回 nan）——
  这正是 A10 要纠的那个毛病，工具自己不能再犯。
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))
# ⚠️ 用 **append**：插到前面会让 `scripts/gui.py` 遮蔽 `gui/` 包
if str(ROOT / "scripts") not in sys.path:
    sys.path.append(str(ROOT / "scripts"))

from scripts.a10_power_calibration import (  # noqa: E402
    autocorr, monte_carlo, sigma_curve,
)


class TestSigmaCurve(unittest.TestCase):
    """σ(n) 曲线必须**从同一序列的前端**截断。"""

    def test_前缀语义(self):
        d = [0.0] * 10 + [1.0] * 10
        c = sigma_curve(d, [10, 20])
        self.assertEqual(len(c), 2)
        # 前 10 个全是 0 ⇒ σ=0
        self.assertAlmostEqual(c[0]["sd"], 0.0, places=12)
        # 全 20 个 ⇒ σ>0
        self.assertGreater(c[1]["sd"], 0.0)

    def test_超出长度就跳过(self):
        c = sigma_curve([1.0, 2.0, 3.0], [2, 99])
        self.assertEqual([r["n"] for r in c], [2])

    def test_n小于2跳过(self):
        self.assertEqual(sigma_curve([1.0, 2.0, 3.0], [1]), [])

    def test_常量序列σ为零(self):
        """⚠️ σ=0 是**最强**的证据（完全一致），不能变成 nan。"""
        c = sigma_curve([0.001] * 8, [8])
        self.assertEqual(c[0]["sd"], 0.0)
        self.assertEqual(c[0]["mde"], 0.0)


class TestAutocorr(unittest.TestCase):
    def test_白噪声接近零(self):
        import random
        rng = random.Random(7)
        d = [rng.gauss(0, 1) for _ in range(4000)]
        ac = autocorr(d, 3)
        for x in ac:
            self.assertLess(abs(x), 0.06, f"白噪声不该有自相关：{ac}")

    def test_AR1有正自相关(self):
        """`x_t = 0.8 x_{t-1} + e` ⇒ lag1 应接近 0.8。"""
        import random
        rng = random.Random(11)
        d = [0.0]
        for _ in range(4000):
            d.append(0.8 * d[-1] + rng.gauss(0, 1))
        ac = autocorr(d, 2)
        self.assertGreater(ac[0], 0.7)
        self.assertLess(ac[0], 0.9)

    def test_常量序列不炸(self):
        ac = autocorr([1.0] * 10, 2)
        self.assertTrue(all(x != x for x in ac))   # 全 nan，但不抛


class TestMonteCarloCalibration(unittest.TestCase):
    """⭐⭐ **用已知答案检验标定工具**（正态残差 ⇒ 假阳性率应 ≈5%）。"""

    def _normal(self, n=4000, sd=0.003, seed=3):
        import random
        rng = random.Random(seed)
        return [rng.gauss(0, sd) for _ in range(n)]

    def test_正态残差下假阳性率约5个百分点(self):
        resid = self._normal()
        mc = monte_carlo(resid, deltas=[0.0], ns=[500], reps=2000, seed=123)
        cell = mc[(0.0, 500)]
        fp = 1.0 - cell["coverage"]
        # 二项标准误 ≈ sqrt(.05*.95/2000) ≈ 0.49pp ⇒ 3σ ≈ 1.5pp
        self.assertLess(abs(fp - 0.05), 0.02,
                        f"假阳性率应 ≈5%，实测 {fp:.2%}")

    def test_正态残差下覆盖率约95个百分点(self):
        resid = self._normal()
        mc = monte_carlo(resid, deltas=[0.001], ns=[500], reps=2000, seed=123)
        cov = mc[(0.001, 500)]["coverage"]
        self.assertLess(abs(cov - 0.95), 0.02, f"覆盖率应 ≈95%，实测 {cov:.2%}")

    def test_效应为零时判力等于假阳性率(self):
        resid = self._normal()
        mc = monte_carlo(resid, deltas=[0.0], ns=[500], reps=800, seed=5)
        self.assertLess(mc[(0.0, 500)]["power"], 0.12)

    def test_效应越大判力越高(self):
        resid = self._normal(sd=0.003)
        mc = monte_carlo(resid, deltas=[0.0005, 0.005], ns=[200],
                         reps=800, seed=5)
        self.assertLess(mc[(0.0005, 200)]["power"], mc[(0.005, 200)]["power"])

    def test_配对检验应当无偏(self):
        """点估计的均值应等于注入的 δ（配对差值检验是无偏的）。"""
        resid = self._normal()
        mc = monte_carlo(resid, deltas=[0.002], ns=[300], reps=1500, seed=8)
        self.assertLess(abs(mc[(0.002, 300)]["bias"]), 2e-4)

    def test_固定种子完全可复现(self):
        """⭐⭐ 报告里的数字要能被别人复核 ⇒ 同种子必须逐位相同。"""
        resid = self._normal(n=500)
        a = monte_carlo(resid, deltas=[0.001], ns=[200], reps=200, seed=42)
        b = monte_carlo(resid, deltas=[0.001], ns=[200], reps=200, seed=42)
        self.assertEqual(a[(0.001, 200)], b[(0.001, 200)])

    def test_换种子会变但落在合理范围(self):
        """⚠️ 分辨力补强：不能"因为固定种子所以永远一个数"。

        ⚠️ 我第一版断言"判力应离 0.5 不超过 0.2"，**猜错了**：
        δ=10bp、σ=30bp、n=200 ⇒ MDE≈6bp ⇒ 判力本来就在 **0.98 左右**。
        ⇒ 教训同前：测**性质**（种子会变 / 判力落在合理带宽），
        不要猜一个具体值。
        """
        resid = self._normal(n=500, sd=0.003)
        a = monte_carlo(resid, deltas=[0.001], ns=[200], reps=400, seed=1)
        b = monte_carlo(resid, deltas=[0.001], ns=[200], reps=400, seed=2)
        self.assertNotEqual(a[(0.001, 200)]["power"],
                            b[(0.001, 200)]["power"])
        # δ (10bp) 远大于 MDE (≈6bp) ⇒ 判力应当很高，但**不能假装是 1.0**
        for r in (a, b):
            self.assertGreater(r[(0.001, 200)]["power"], 0.9)
            self.assertLessEqual(r[(0.001, 200)]["power"], 1.0)

    def test_delta等于MDE时判力约八成(self):
        """⭐⭐ **`MDE` 的定义点是 80% 判力**（公式含 `z_功效 = 0.84`）。

        ⚠️ 我第一版写的是"δ=MDE ⇒ 判力 ≈50%"，**错**——那是把
        「含功效项的 MDE」与「50% 门槛 `t_crit·σ/√n`」搞混了。
        实测 0.812 ⇒ 正好印证 MDE 是 80% 判力点。
        """
        from scripts.a10_power_calibration import min_detectable_effect
        resid = self._normal(n=500, sd=0.003)
        mde = min_detectable_effect(0.003, 200)
        mc = monte_carlo(resid, deltas=[mde], ns=[200], reps=2000, seed=17)
        p = mc[(mde, 200)]["power"]
        self.assertGreater(p, 0.72, f"δ=MDE 时判力应≈80%，实测 {p:.1%}")
        self.assertLess(p, 0.90, f"δ=MDE 时判力应≈80%，实测 {p:.1%}")

    def test_五成判力的门槛低于MDE(self):
        """50% 门槛（少功效项）必须**小于** MDE ⇒ 两者是不同的量。"""
        from scripts.a10_power_calibration import min_detectable_effect
        from tw.segmented import t_crit95
        sd, n = 0.003, 200
        mde = min_detectable_effect(sd, n)
        thr50 = t_crit95(n - 1) * sd / (n ** 0.5)
        self.assertLess(thr50, mde)

    def test_注入的真值被真实还原(self):
        """δ 是真值 ⇒ 覆盖率必须围绕它（而不是围绕 0）。"""
        resid = self._normal(sd=0.001)
        mc = monte_carlo(resid, deltas=[0.01], ns=[100], reps=800, seed=9)
        # 效应远大于噪声 ⇒ 判力应当很高
        self.assertGreater(mc[(0.01, 100)]["power"], 0.95)


class TestNForPowerNoExtrapolation(unittest.TestCase):
    """⚠️ **不许外推**——这正是 A10 要纠的那个毛病。"""

    def _src(self, pairs):
        mc = {}
        for n, p in pairs:
            mc[f"d0.0005_n{n}"] = {"delta": 0.0005, "n": n, "power": p,
                                   "coverage": 0.95, "bias": 0.0,
                                   "se_of_mean": 0.0, "reps": 100}
        return {"mc": mc}

    def test_区间内插值(self):
        from scripts.make_a10_report import _n_for_power
        src = self._src([(100, 0.5), (300, 1.0)])
        n = _n_for_power(src, 0.0005, 0.8)
        self.assertAlmostEqual(n, 100 + (0.8 - 0.5) / (1.0 - 0.5) * 200,
                               places=6)

    def test_超出范围返回nan_不外推(self):
        from scripts.make_a10_report import _n_for_power
        src = self._src([(100, 0.1), (300, 0.3)])   # 最高只到 30%
        n = _n_for_power(src, 0.0005, 0.8)
        self.assertTrue(n != n, "超出已测范围必须返回 nan，不许外推")

    def test_全为正样本时取最小n(self):
        from scripts.make_a10_report import _n_for_power
        src = self._src([(50, 0.9), (100, 1.0)])
        self.assertEqual(_n_for_power(src, 0.0005, 0.8), 50.0)

    def test_缺该delta返回nan(self):
        from scripts.make_a10_report import _n_for_power
        src = self._src([(100, 0.5), (300, 1.0)])
        self.assertTrue(_n_for_power(src, 0.002, 0.8) !=
                        _n_for_power(src, 0.002, 0.8))  # nan != nan


if __name__ == "__main__":
    unittest.main(verbosity=2)
