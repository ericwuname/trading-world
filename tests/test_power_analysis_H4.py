"""功效分析的回归测试（工作线M）。

任务书 §4.6 点名要钉 **ratio 方向**（把 n2/n1 写成 n1/n2）。
这类错误的可怕之处在于：**ratio=1 时永远看不出来**——
只有 42:31 这种不等样本量才会暴露，而那正是本项目的实际情况。

所以本文件用**两组已知答案**来钉：
  ① **教科书锚点**（Cohen 表：d=0.5→64、d=0.8→26、d=0.2→394，
     这三个值在 80% 功效、α=0.05、等样本量下是标准结果）；
  ② **方向性质**（ratio>1 时第一组可以更少）。
只有 ①，ratio 写反也能过（ratio=1 时两种写法等价）；
只有 ②，公式整体错也能"方向正确"。
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.power_analysis_H4 import (  # noqa: E402
    estimate_required_n_for_quadrant_test,
    estimate_seeds_needed,
    power_ttest_1samp,
    power_ttest_ind,
    solve_n_for_power,
    solve_n_onesample,
)

_SCRIPTS = str(ROOT / "scripts")
while _SCRIPTS in sys.path:
    sys.path.remove(_SCRIPTS)

#: Cohen 表的标准值（80% 功效、α=0.05、两组等样本量）
TEXTBOOK = [(0.2, 394), (0.5, 64), (0.8, 26)]


class TestTextbookAnchors(unittest.TestCase):
    """① 教科书锚点——公式整体的正确性。"""

    def test_功效值对上教科书(self) -> None:
        for d, n in TEXTBOOK:
            p = power_ttest_ind(d, n)
            self.assertAlmostEqual(p, 0.80, delta=0.02,
                                   msg=f"d={d}, n={n} 的功效 {p:.4f} 偏离 0.80 太多")

    def test_反解精确命中(self) -> None:
        for d, expect in TEXTBOOK:
            got = solve_n_for_power(d, power=0.8)
            self.assertEqual(got, expect,
                             f"d={d} 反解得到 {got}，教科书是 {expect}")

    def test_单样本版也自洽(self) -> None:
        """⚠️ 这条断言**第一版写反了**，值得留个记号。

        我原以为"同样 n 下单样本功效更低（因为观测少一半）"——
        实测**相反**：n=64 时单样本 0.976 > 双样本 0.801。
        原因是**非中心参数的口径不同**：
            单样本 ``ncp = d·√n``
            双样本 ``ncp = d/√(1/n1+1/n2) = d·√(n/2)``（等样本量时）
        所以同样 n 下单样本的 ncp 更大。**不是实现错，是我的直觉错。**
        """
        p1 = power_ttest_1samp(0.5, 64)
        p2 = power_ttest_ind(0.5, 64)
        self.assertGreater(p1, p2,
                           f"单样本 {p1:.4f} 应当高于双样本 {p2:.4f}"
                           "（ncp 口径：√n vs √(n/2)）")
        self.assertGreater(p1, 0.9)


class TestRatioDirection(unittest.TestCase):
    """② 方向性质——M54 的照妖镜。"""

    def test_ratio大于一时第一组可以更少(self) -> None:
        """n2 更大 ⇒ n1 需求更小。写反了这条必红（而 ratio=1 时看不出来）。"""
        n_equal = solve_n_for_power(0.37, power=0.8, ratio=1.0)
        n_uneven = solve_n_for_power(0.37, power=0.8, ratio=42 / 31)
        self.assertLess(n_uneven, n_equal,
                        f"ratio>1 时 n1 反而更大（{n_uneven} vs {n_equal}）"
                        "——ratio 的分子分母写反了")

    def test_ratio小于一时第一组需要更多(self) -> None:
        n_equal = solve_n_for_power(0.37, power=0.8, ratio=1.0)
        n_small2 = solve_n_for_power(0.37, power=0.8, ratio=0.5)
        self.assertGreater(n_small2, n_equal)

    def test_总样本量在合理范围内随ratio变化(self) -> None:
        """两个方向的**总量**都要比等样本量多（不等样本浪费自由度）。"""
        tot_equal = 2 * solve_n_for_power(0.37, power=0.8, ratio=1.0)
        n_u = solve_n_for_power(0.37, power=0.8, ratio=42 / 31)
        self.assertGreater(n_u * (1 + 42 / 31), tot_equal * 0.95)

    def test_给定n1时n2按ratio放大(self) -> None:
        r = estimate_required_n_for_quadrant_test(1.0, 3.0, ratio=2.0)
        self.assertAlmostEqual(r.required_n_group2, r.required_n_group1 * 2.0,
                               places=6)


class TestBoundaryProperties(unittest.TestCase):
    """边界性质——防止公式在极端处给出垃圾。"""

    def test_零效应时功效等于alpha(self) -> None:
        self.assertAlmostEqual(power_ttest_ind(0.0, 50), 0.05, places=6)

    def test_功效随样本单调上升(self) -> None:
        ps = [power_ttest_ind(0.5, n) for n in (10, 30, 100, 300)]
        self.assertEqual(ps, sorted(ps), "功效不是单调的")

    def test_大样本不返回nan(self) -> None:
        """⭐ 实测踩到：scipy 的 ``nct`` 在 df≈2000 时返回 nan，
        而 ``nan < power`` 为假会让二分**误判达标**、静默给出偏小的样本量。"""
        for n in (500, 1000, 2000):
            p = power_ttest_ind(0.5, n)
            self.assertTrue(math.isfinite(p), f"n={n} 处功效是 nan")
            self.assertGreater(p, 0.99)

    def test_非法输入不崩(self) -> None:
        self.assertTrue(math.isnan(power_ttest_ind(0.5, 1)))
        self.assertTrue(math.isnan(power_ttest_ind(0.5, 10, ratio=0)))
        self.assertEqual(solve_n_for_power(0.0), float("inf"))
        self.assertTrue(math.isnan(solve_n_for_power(1e-9, power=0.8,
                                                     n_max=1_000)))

    def test_反解结果必须真的达标(self) -> None:
        """反解后复核——宁可返回 nan 也不许悄悄给偏小的 n。"""
        for d in (0.2, 0.5, 0.8):
            n = solve_n_for_power(d, power=0.8)
            self.assertGreaterEqual(power_ttest_ind(d, n), 0.8,
                                    f"d={d} 反解出 n={n} 却没达标")


class TestObservedScaleConversion(unittest.TestCase):
    """把"需要多少事件点"换算成"需要多少种子/多长 tick"。"""

    def test_换算与两条路都被说明(self) -> None:
        out = estimate_seeds_needed(487, 10.3, observed_n_seeds=3)
        self.assertAlmostEqual(out["seeds_needed"], 487 / 10.3, places=6)
        self.assertGreater(out["extra_seeds_needed"], 0)
        self.assertIn("单次模拟时长", out["note"])
        self.assertIn("稳态", out["note"], "没有提醒「加长可能引入非平稳性」")

    def test_非法输入报错(self) -> None:
        with self.assertRaises(ValueError):
            estimate_seeds_needed(100, 0.0)


class TestSingleSampleSolver(unittest.TestCase):
    """EM.3 用的是**跨运行**检验（自由度 = 运行数），需要单样本解。"""

    def test_单样本反解单调(self) -> None:
        self.assertLess(solve_n_onesample(0.5), solve_n_onesample(0.2))

    def test_单样本与双样本量级关系(self) -> None:
        """⚠️ 这条也**写反过一次**：单样本所需 n ≈ 双样本**每组** n 的 **一半**。

        因为双样本的每组只贡献一半的信息：
        单样本 ``ncp = d·√n``，双样本 ``ncp = d·√(n/2)``。
        实测 d=0.5 时单样本 34、双样本每组 64 —— 34/64 ≈ 0.53 ✓
        """
        n1 = solve_n_onesample(0.5, power=0.8)
        n2 = solve_n_for_power(0.5, power=0.8)
        self.assertAlmostEqual(n1 / n2, 0.5, delta=0.1,
                               msg=f"单样本 {n1} vs 双样本每组 {n2}"
                                   "——应当是约 1:2 的关系")


if __name__ == "__main__":
    unittest.main(verbosity=2)
