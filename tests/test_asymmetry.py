"""工作线E 的非对称性分析测试（变异体 M44 / M45 的照妖镜）。

任务书 §4.6 点名的两处都在这里：
  · M44：``rolling_cooccurrence_rate`` 的**窗口边界**（漏掉窗口最后一个 tick）
  · M45：四象限里 ``taker_only`` 与 ``maker_only`` 的掩码**写反**

两处的共同特点是：错了不会报错，只会让结果"看起来像另一种结论"——
窗口漏一格 ⇒ 同步率系统性偏低；掩码写反 ⇒ 结果与 H4 预期方向相反，
读起来像"假说被否证"。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tw.analyzer_asymmetry import (  # noqa: E402
    block_quadrant_contrast,
    burst_mask,
    measure_burst_synchrony,
    measure_local_depth_during_taker_burst,
    quadrant_masks,
    rolling_cooccurrence_rate,
)


class TestRollingWindowBoundary(unittest.TestCase):
    """⭐ M44：窗口必须**包含当前 tick**（区间两端都含）。"""

    def test_窗口包含当前点(self) -> None:
        # 只有最后一格同时为真
        a = np.array([0, 0, 0, 1], dtype=bool)
        b = np.array([0, 0, 0, 1], dtype=bool)
        rate = rolling_cooccurrence_rate(a, b, window=2)
        # i=3 的窗口是 [2,3]，其中 a&b = [0,1] ⇒ 0.5
        self.assertAlmostEqual(rate[3], 0.5, places=9,
                               msg="窗口漏掉了当前 tick——同步率会系统性偏低")

    def test_头部窗口按实际长度算_不补零(self) -> None:
        a = np.array([1, 1, 1], dtype=bool)
        b = np.array([1, 1, 1], dtype=bool)
        rate = rolling_cooccurrence_rate(a, b, window=10)
        # 全部为真 ⇒ 任何位置的窗口内比例都是 1（若补零会得到 0.1/0.2/0.3）
        np.testing.assert_allclose(rate, [1.0, 1.0, 1.0],
                                   err_msg="头部窗口被补零了——那会把'还没开始'算成'不同步'")

    def test_全假时为零(self) -> None:
        z = np.zeros(5, dtype=bool)
        np.testing.assert_allclose(rolling_cooccurrence_rate(z, z, 3),
                                   np.zeros(5))

    def test_窗口更大时结果不超出一(self) -> None:
        rng = np.random.default_rng(7)
        a = rng.random(200) < 0.3
        b = rng.random(200) < 0.4
        rate = rolling_cooccurrence_rate(a, b, 25)
        self.assertTrue(np.all((rate >= 0) & (rate <= 1)))

    def test_长度不一致时报错(self) -> None:
        with self.assertRaises(ValueError):
            rolling_cooccurrence_rate(np.zeros(3, bool), np.zeros(4, bool), 2)

    def test_非法窗口报错(self) -> None:
        with self.assertRaises(ValueError):
            rolling_cooccurrence_rate(np.zeros(3, bool), np.zeros(3, bool), 0)


class TestQuadrantMasks(unittest.TestCase):
    """⭐ M45：四个象限的语义必须各自正确。"""

    def setUp(self) -> None:
        #  t = [1,1,0,0]   m = [1,0,1,0]
        self.t = np.array([True, True, False, False])
        self.m = np.array([True, False, True, False])

    def test_both_burst只含同时突发(self) -> None:
        mk = quadrant_masks(self.t, self.m)
        np.testing.assert_array_equal(mk["both_burst"], [True, False, False, False])

    def test_taker_only只含吃单突发(self) -> None:
        mk = quadrant_masks(self.t, self.m)
        np.testing.assert_array_equal(
            mk["taker_only"], [False, True, False, False],
            err_msg="taker_only 的位置不对——掩码很可能与 maker_only 写反了")

    def test_maker_only只含做市突发(self) -> None:
        mk = quadrant_masks(self.t, self.m)
        np.testing.assert_array_equal(mk["maker_only"], [False, False, True, False])

    def test_neither是两边都不突发(self) -> None:
        mk = quadrant_masks(self.t, self.m)
        np.testing.assert_array_equal(mk["neither"], [False, False, False, True])

    def test_四象限构成完整划分(self) -> None:
        """四个掩码互斥且并为全集——任一写反都会破坏这一点。"""
        mk = quadrant_masks(self.t, self.m)
        stacked = np.stack([mk[k] for k in
                            ("both_burst", "taker_only", "maker_only", "neither")])
        self.assertTrue(np.all(stacked.sum(axis=0) == 1),
                        "四象限没有构成完整划分（有重叠或遗漏）")

    def test_长度不一致时报错(self) -> None:
        with self.assertRaises(ValueError):
            quadrant_masks(np.zeros(3, bool), np.zeros(4, bool))


class TestBurstMaskSemantics(unittest.TestCase):
    """突发阈值：严格大于，且两侧各自独立标定。"""

    def test_严格大于(self) -> None:
        # 全是同一个值 ⇒ 没有任何点能"大于"它 ⇒ 无突发
        self.assertEqual(burst_mask(np.full(10, 3.0), 80.0).sum(), 0)

    def test_比例大致符合百分位(self) -> None:
        x = np.arange(100, dtype=float)
        self.assertAlmostEqual(burst_mask(x, 80.0).sum() / 100, 0.20, delta=0.03)

    def test_空输入不崩(self) -> None:
        self.assertEqual(burst_mask(np.array([]), 80.0).size, 0)


class TestSynchronyReportsBothMetrics(unittest.TestCase):
    """联合率与条件率必须都给——两者含义不同，不能混。"""

    def test_条件率与联合率都算出且不相等(self) -> None:
        rng = np.random.default_rng(11)
        t = rng.integers(0, 10, 400).astype(float)
        m = rng.integers(0, 10, 400).astype(float)
        s = measure_burst_synchrony(t, m, pct=80.0, window=20)
        self.assertTrue(np.isfinite(s.mean_sync))
        self.assertTrue(np.isfinite(s.p_maker_given_taker))
        self.assertGreaterEqual(s.p_maker_given_taker, 0.0)
        self.assertLessEqual(s.p_maker_given_taker, 1.0)
        # 联合率 ≈ P(a)·P(b|a)，一定不超过条件率
        self.assertLessEqual(s.mean_sync, s.p_maker_given_taker + 1e-9,
                             "联合率大于条件率——定义算错了")

    def test_完全同步时条件率为一(self) -> None:
        # ⚠️ 值域必须**多于两个**取值：若序列只有 {0,1}，80 分位恰好等于最大值 1，
        #    而突发判据是**严格大于** ⇒ 一个突发点都没有，条件率会是 nan。
        #    （第一版测试就是这么写的，结果报了个假失败——是数据构造的问题，
        #      不是实现的问题。留下来当个记号。）
        t = np.tile(np.array([0.0, 1.0, 2.0, 3.0, 4.0]), 80)   # 400 点，20% 是 4.0
        s = measure_burst_synchrony(t, t.copy(), pct=80.0, window=5)
        self.assertGreater(s.n_taker_burst, 0, "构造的数据没有突发点")
        self.assertAlmostEqual(s.p_maker_given_taker, 1.0, places=9)
        self.assertGreater(s.lift, 1.0)


class TestQuadrantDepthAndContrast(unittest.TestCase):
    """四象限深度与块级对比。"""

    def test_四象限深度各取各的均值(self) -> None:
        depth = np.array([10.0, 20.0, 30.0, 40.0])
        t = np.array([True, True, False, False])
        m = np.array([True, False, True, False])
        out = measure_local_depth_during_taker_burst(depth, t, m)
        self.assertAlmostEqual(out["both_burst"]["mean"], 10.0)
        self.assertAlmostEqual(out["taker_only"]["mean"], 20.0)
        self.assertAlmostEqual(out["maker_only"]["mean"], 30.0)
        self.assertAlmostEqual(out["neither"]["mean"], 40.0)

    def test_空象限返回nan而不崩(self) -> None:
        depth = np.arange(10.0)
        t = np.zeros(10, bool)
        m = np.zeros(10, bool)
        out = measure_local_depth_during_taker_burst(depth, t, m)
        self.assertTrue(out["both_burst"]["mean"] != out["both_burst"]["mean"])
        self.assertEqual(out["neither"]["n"], 10)

    def test_块级对比方向正确(self) -> None:
        """构造：taker_only 期的深度**明显更薄** ⇒ diff(=both−taker_only) 应为正。"""
        n = 1000
        t = np.zeros(n, bool)
        m = np.zeros(n, bool)
        t[:500] = True           # 前 500：taker 突发
        m[:250] = True           # 其中前 250：做市方也突发
        depth = np.full(n, 50.0)
        depth[250:500] = 20.0    # taker_only 段很薄
        c = block_quadrant_contrast(depth, t, m, block=100)
        self.assertGreater(c["diff"], 0,
                           "taker_only 更薄却算出 diff ≤ 0——方向反了")
        self.assertTrue(c["h4_direction_supported"])

    def test_块数不足时明确说明(self) -> None:
        depth = np.full(50, 10.0)
        t = np.zeros(50, bool)
        m = np.zeros(50, bool)
        c = block_quadrant_contrast(depth, t, m, block=1000)
        self.assertIn("reason", c)
        self.assertTrue(c["t"] != c["t"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
