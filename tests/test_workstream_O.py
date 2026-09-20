"""一致性审计的回归测试（工作线O）。

任务书 §2.5 点名要钉两处：
  · **M55**：``ci_overlap_fraction`` 分子分母算反（用两区间宽度**之和**做分母）
  · **M56**：``pairwise_verdict`` 阈值方向反（重叠大时反而判「显著」）

两处都用 **EL.2 已经报过的真实数字**做锚点——
那是本项目唯一一组「已被判定过」的重叠数据，
用它做回归测试，等于把新函数挂在旧结论上校准：

    taker CI = [0.549, 1.218]，both CI = [0.550, 1.378]
    绝对重叠 = 0.668（EL.2 报的）    比例 = 99.8%（EL.2 报的）
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tw.analyzer_consistency import (  # noqa: E402
    DEFAULT_OVERLAP_THRESHOLD,
    SIGNIFICANT_MAX_OVERLAP,
    ci_overlap_fraction,
    ci_overlap_length,
    direction_verdict,
    pairwise_verdict,
    required_n_to_separate,
)

#: EL.2 的实测值（工作线E，对称配置）
EL2_TAKER_CI = (0.549, 1.218)
EL2_BOTH_CI = (0.550, 1.378)
EL2_OVERLAP_LENGTH = 0.668
EL2_OVERLAP_FRACTION = 0.9985      # 0.668 / 0.669


class TestOverlapAgainstEL2(unittest.TestCase):
    """⭐ M55：用 EL.2 的真实数字做回归锚点。"""

    def test_绝对重叠复现EL2的0点668(self) -> None:
        got = ci_overlap_length(EL2_TAKER_CI, EL2_BOTH_CI)
        self.assertAlmostEqual(got, EL2_OVERLAP_LENGTH, places=3)

    def test_比例复现EL2的99点8百分比(self) -> None:
        got = ci_overlap_fraction(EL2_TAKER_CI, EL2_BOTH_CI)
        self.assertAlmostEqual(got, EL2_OVERLAP_FRACTION, places=3)

    def test_分母必须是较窄的那个而不是两者之和(self) -> None:
        """⚠️ 用「之和」做分母会把 100% 包含算成 ~50%，于是把
        「无法判定」错看成「显著」——方向恰好是危险的。"""
        w_taker = EL2_TAKER_CI[1] - EL2_TAKER_CI[0]
        w_both = EL2_BOTH_CI[1] - EL2_BOTH_CI[0]
        frac = ci_overlap_fraction(EL2_TAKER_CI, EL2_BOTH_CI)
        wrong = EL2_OVERLAP_LENGTH / (w_taker + w_both)
        self.assertGreater(frac, 0.9, "比例应当接近 1（几乎完全重叠）")
        self.assertLess(wrong, 0.6, "用之和做分母会得到约 0.45——这正是不该发生的")
        self.assertAlmostEqual(frac / wrong, (w_taker + w_both) / min(w_taker, w_both),
                               places=6)

    def test_完全包含时比例为一(self) -> None:
        self.assertAlmostEqual(ci_overlap_fraction((0.0, 2.0), (0.5, 1.0)), 1.0,
                               places=9)

    def test_完全不重叠时为零(self) -> None:
        self.assertAlmostEqual(ci_overlap_fraction((0.0, 1.0), (2.0, 3.0)), 0.0,
                               places=9)

    def test_退化输入返回nan而不是零(self) -> None:
        """宽度为 0 时「比例」没有意义——不许给个 0 假装有意义。"""
        self.assertTrue(math.isnan(ci_overlap_fraction((1.0, 1.0), (0.5, 1.5))))

    def test_对称性_交换两个区间结果不变(self) -> None:
        self.assertAlmostEqual(ci_overlap_fraction(EL2_TAKER_CI, EL2_BOTH_CI),
                               ci_overlap_fraction(EL2_BOTH_CI, EL2_TAKER_CI),
                               places=12)


class TestRequiredNFormula(unittest.TestCase):
    """外推公式：用 EL.2 的「约 1000 个种子」做量级校准。"""

    def test_EL2量级一致(self) -> None:
        """taker vs both：宽度 0.669/0.828、Δ=0.060、现用 8 个种子。"""
        n = required_n_to_separate(0.669, 0.828, 0.060, 8)
        self.assertGreater(n, 800, "应当落在「约 1000 个种子」这个量级")
        self.assertLess(n, 2000)

    def test_用两区间之和而不是较窄那个(self) -> None:
        """⚠️ 只用较窄宽度会**低估**所需样本量（≈995 vs ≈1250）——
        低估的方向是危险的：它会把「不值得」看成「值得」。"""
        w_a, w_b, d, n0 = 0.669, 0.828, 0.060, 8
        full = required_n_to_separate(w_a, w_b, d, n0)
        naive = n0 * (min(w_a, w_b) / (2 * d)) ** 2      # 只用较窄宽度（错的做法）
        self.assertGreater(full, naive,
                           "用两者之和应当得到更大的所需样本量")

    def test_效应越大所需样本越少(self) -> None:
        a = required_n_to_separate(0.8, 0.8, 0.05, 8)
        b = required_n_to_separate(0.8, 0.8, 0.20, 8)
        self.assertGreater(a, b)

    def test_零效应需要无穷多(self) -> None:
        self.assertEqual(required_n_to_separate(0.8, 0.8, 0.0, 8), float("inf"))

    def test_target_overlap放宽后所需样本变少(self) -> None:
        strict = required_n_to_separate(0.8, 0.8, 0.10, 8, target_overlap=0.0)
        loose = required_n_to_separate(0.8, 0.8, 0.10, 8, target_overlap=0.05)
        self.assertLess(loose, strict)


class TestPairwiseVerdict(unittest.TestCase):
    """⭐ M56：三选一的判定方向不能反。"""

    def test_完全不重叠判显著(self) -> None:
        v = pairwise_verdict("A", (0.0, 1.0), 0.5, "B", (2.0, 3.0), 2.5, 8)
        self.assertEqual(v.verdict, "显著")
        self.assertEqual(v.overlap_fraction, 0.0)
        self.assertIsNotNone(v.direction)

    def test_显著时方向由点估计决定(self) -> None:
        up = pairwise_verdict("A", (0.0, 1.0), 0.5, "B", (2.0, 3.0), 2.5, 8)
        down = pairwise_verdict("A", (2.0, 3.0), 2.5, "B", (0.0, 1.0), 0.5, 8)
        self.assertIn("B", up.direction)
        self.assertIn("A", down.direction)

    def test_完全包含判无法判定(self) -> None:
        """⭐ M56 的照妖镜：重叠 100% 时**绝不能**判「显著」。"""
        v = pairwise_verdict("A", (0.5, 1.5), 1.0, "B", (0.8, 1.2), 1.1, 8)
        self.assertEqual(v.verdict, "依然无法判定")
        self.assertAlmostEqual(v.overlap_fraction, 1.0, places=9)
        self.assertIsNotNone(v.required_n)

    def test_EL2的真实比较被判无法判定(self) -> None:
        v = pairwise_verdict("taker", EL2_TAKER_CI, 0.734,
                             "both", EL2_BOTH_CI, 0.794, 8)
        self.assertEqual(v.verdict, "依然无法判定")

    def test_中间地带判边缘不强行归类(self) -> None:
        # 构造重叠 ≈ 25%（落在 0.20~0.30 之间）
        v = pairwise_verdict("A", (0.0, 1.0), 0.5, "B", (0.75, 1.75), 1.25, 8)
        self.assertAlmostEqual(v.overlap_fraction, 0.25, places=6)
        self.assertEqual(v.verdict, "边缘")

    def test_阈值方向不会被写反(self) -> None:
        """扫一遍重叠比例，验证「重叠越大越不可能判显著」这个单调性。"""
        verdicts = []
        for shift in (0.05, 0.3, 0.6, 0.9, 1.0):
            v = pairwise_verdict("A", (0.0, 1.0), 0.5, "B",
                                 (shift, shift + 1.0), 0.5 + shift, 8)
            verdicts.append((shift, v.overlap_fraction, v.verdict))
        # 重叠比例随 shift 单调下降 ⇒ 判定应当从「无法判定」走向「显著」
        fracs = [f for _, f, _ in verdicts]
        self.assertEqual(fracs, sorted(fracs, reverse=True))
        self.assertEqual(verdicts[0][2], "依然无法判定")
        self.assertEqual(verdicts[-1][2], "显著")

    def test_阈值常量本身合理(self) -> None:
        self.assertGreater(DEFAULT_OVERLAP_THRESHOLD, SIGNIFICANT_MAX_OVERLAP,
                           "两个阈值之间必须留出「边缘」地带")
        self.assertAlmostEqual(SIGNIFICANT_MAX_OVERLAP, 0.20, places=6)
        self.assertAlmostEqual(DEFAULT_OVERLAP_THRESHOLD, 0.30, places=6)


class TestDirectionVerdict(unittest.TestCase):
    """工作线P 的单个家族方向判定（与「比较」是两件事）。"""

    def test_EA4_taker被判偏线性(self) -> None:
        """任务书 §3.5 的指定用例：CI=[0.549,1.218] ⇒ 排除 0.5。"""
        v = direction_verdict(0.734, (0.549, 1.218))
        self.assertIn("sqrt_law_excluded", v)
        self.assertIn("偏线性", v)

    def test_含0点5不含1点0是consistent(self) -> None:
        v = direction_verdict(0.7, (0.3, 0.9))
        self.assertEqual(v, "sqrt_law_consistent")

    def test_同时含0点5与1点0是undetermined(self) -> None:
        v = direction_verdict(0.9, (0.4, 1.4))
        self.assertEqual(v, "undetermined")

    def test_整体低于0点5判偏超凹(self) -> None:
        v = direction_verdict(0.3, (0.1, 0.45))
        self.assertIn("偏超凹", v)

    def test_边界恰好等于0点5不算排除(self) -> None:
        """``ci_lower == 0.5`` 时 0.5 仍在区间内 ⇒ 不算排除（严格）。"""
        v = direction_verdict(0.8, (0.5, 1.2))
        self.assertNotIn("excluded", v)


if __name__ == "__main__":
    unittest.main(verbosity=2)
