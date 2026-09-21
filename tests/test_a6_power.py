"""`scripts/a6_power.py` 里那些**不跑模拟也会错**的纯逻辑。

为什么这些也要测（与 `tests/test_stage_scripts.py` 同一条理由）：
"要多少段才判得出"这个数字是**写进报告、决定用户要不要继续投额度**的，
它错了不会报错——只会让报告给出一个**听起来很有依据的错结论**。

⭐ 本文件最重要的一条：**"MDE 是多少"与"需要多少段"必须互为逆运算。**
它们曾经各有各的实现（一份含功效项、一份不含），
于是 `power_analysis` 报的 n **偏小约 2 倍**。
⇒ 现在只有一份实现（`tw.segmented`），下面用**等式**把它钉住。
"""

from __future__ import annotations

import math
import unittest

from tw.segmented import (
    min_detectable_effect,
    power_analysis,
    t_crit95,
)


class TestTCrit95(unittest.TestCase):
    def test_小样本临界值远大于1_96(self):
        """df=1 时双侧 95% 临界值是 12.706，不是 1.96。
        它正是"扩档位能救分辨力"那条结论的机理（df 1→5：12.71→2.57）。"""
        self.assertAlmostEqual(t_crit95(1), 12.706, places=3)
        self.assertGreater(t_crit95(1), 6.0)

    def test_大样本趋近1_96(self):
        self.assertAlmostEqual(t_crit95(10 ** 6), 1.960, places=3)
        self.assertLess(t_crit95(120), 2.0)

    def test_随自由度单调不增(self):
        """⚠️ 单调性必须成立：否则迭代逼近会**震荡不收敛**，
        而迭代 200 次后返回的那个数看起来完全正常。"""
        prev = float("inf")
        for df in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20,
                   25, 30, 40, 60, 120, 10 ** 6]:
            cur = t_crit95(df)
            self.assertLessEqual(cur, prev + 1e-12,
                                 f"df={df} 时临界值反而变大了")
            prev = cur

    def test_非法自由度返回无穷(self):
        """⚠️ `df <= 0` 返回 **inf**，不是"一个很大的数"。

        语义上：df=0 表示**没有自由度可用** ⇒ 区间无穷宽 ⇒
        "任何效应都无法判定"。返回 12.706 会让它看起来像
        "有一点点信息"，那是**假的精度**。
        （⚠️ 这条断言改过一次：我第一版写的是 12.706，
        后来把两份 t 表统一到 `_t_crit` 时恢复成了 inf——
        恢复的是**原来的、更诚实**的行为。）
        """
        self.assertEqual(t_crit95(0), float("inf"))
        self.assertEqual(t_crit95(-5), float("inf"))


class TestMdeAndPowerAreInverse(unittest.TestCase):
    """⭐⭐ **本文件的核心：两个方向的估计必须是同一套口径。**"""

    def test_互逆(self):
        """`min_detectable_effect(sd, power_analysis(effect, sd).n)`
        应当 ≈ `effect`（且**不超过**它——超过就是"算出来的 n 不够用"）。
        """
        for sd, eff in [(0.02, 0.01), (0.10, 0.02), (0.30, 0.05),
                        (0.006, 0.001), (0.30, 0.20), (0.10, 0.01)]:
            n = power_analysis(eff, sd)["required_segs"]
            m = min_detectable_effect(sd, n)
            self.assertLessEqual(m, eff * 1.001,
                                 f"sd={sd} eff={eff} n={n} 反算 MDE={m}")
            self.assertGreater(m, eff * 0.85,
                               f"sd={sd} eff={eff} n={n} 过于保守（MDE={m}）")

    def test_比正态近似更保守(self):
        """⚠️ 用固定 `1.96+0.84` 会**低估**所需段数。
        正确的（含 df 相关临界值的）解必须 **≥** 正态近似的解。
        若有人把实现改回正态公式，这条会立刻变红。
        """
        for sd, eff in [(0.02, 0.01), (0.10, 0.02), (0.30, 0.05),
                        (0.30, 0.20), (0.10, 0.01)]:
            n_t = power_analysis(eff, sd)["required_segs"]
            z = 1.96 + 0.8416
            n_z = math.ceil((z ** 2) * (sd ** 2) / (eff ** 2))
            self.assertGreaterEqual(n_t, n_z,
                                    f"sd={sd} eff={eff}: {n_t} < {n_z}")

    def test_小样本区差异最大(self):
        """在小样本区（n 小 ⇒ df 小 ⇒ t_crit 大）差异百分比最大。"""
        def rel(sd, effect):
            n_t = power_analysis(effect, sd)["required_segs"]
            z = 1.96 + 0.8416
            n_z = math.ceil((z ** 2) * (sd ** 2) / (effect ** 2))
            return (n_t - n_z) / n_z
        self.assertGreater(rel(0.30, 0.20), rel(0.10, 0.01))

    def test_效应越大需要越少段(self):
        self.assertLess(power_analysis(0.04, 0.02)["required_segs"],
                        power_analysis(0.01, 0.02)["required_segs"])

    def test_噪声越大需要越多段(self):
        self.assertGreater(power_analysis(0.01, 0.04)["required_segs"],
                           power_analysis(0.01, 0.01)["required_segs"])

    def test_退化输入返回None(self):
        self.assertIsNone(power_analysis(0.0, 0.01)["required_segs"])
        self.assertIsNone(power_analysis(0.01, 0.0)["required_segs"])
        self.assertIsNone(power_analysis(0.01, float("nan"))["required_segs"])

    def test_nan输入不崩(self):
        nan = float("nan")
        self.assertIsNone(power_analysis(nan, 0.01)["required_segs"])
        self.assertTrue(math.isnan(min_detectable_effect(nan, 5)))
        self.assertTrue(math.isnan(min_detectable_effect(0.01, 1)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
