"""GUI 数据整形层的回归测试（K 线聚合 / 回撤 / 逐笔统计 / 红旗规则）。

这一层的价值在于**口径**，所以测试的重点不是"跑得通"，
而是**口径不能被悄悄改掉**：

  · ``high``/``low`` 必须是窗口内**极值**（折线丢掉的正是这个）；
  · 尾部不满一个窗口**也要产出**（否则最后一段行情凭空消失）；
  · 长度不符必须**报错**而不是静默截断（静默截断会让图和数错位且看不出）；
  · 逐笔统计**刻意不给"胜率"**（做市策略的赚/亏与是否接刀是两件事）。
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tw.viz_gui import (  # noqa: E402
    bars_payload,
    equity_drawdown,
    fill_stats,
    ohlc_from_series,
    report_flags,
)


class TestOHLC(unittest.TestCase):
    def test_基本极值(self) -> None:
        mid = np.array([10.0, 12.0, 11.0, 13.0, 9.0, 14.0])
        bars = ohlc_from_series(mid, window=3)
        self.assertEqual(len(bars), 2)
        b0, b1 = bars
        self.assertEqual((b0.open, b0.close, b0.high, b0.low), (10.0, 11.0, 12.0, 10.0))
        self.assertEqual((b1.open, b1.close, b1.high, b1.low), (13.0, 14.0, 14.0, 9.0))

    def test_高点低点是窗口内极值而不是端点(self) -> None:
        """⭐ 折线丢掉的就是这个：中间的极值。"""
        mid = np.array([5.0, 100.0, 5.0])
        bars = ohlc_from_series(mid, window=3)
        self.assertEqual(bars[0].high, 100.0)
        self.assertEqual(bars[0].low, 5.0)

    def test_尾巴不满窗口也产出(self) -> None:
        """⭐ 否则最后一段行情会凭空消失。"""
        mid = np.arange(7, dtype=float)
        bars = ohlc_from_series(mid, window=3)
        self.assertEqual(len(bars), 3)
        self.assertEqual(bars[-1].n_ticks, 1)
        self.assertEqual(bars[-1].close, 6.0)

    def test_窗口正好整除时不产生空尾巴(self) -> None:
        bars = ohlc_from_series(np.arange(6, dtype=float), window=3)
        self.assertEqual(len(bars), 2)

    def test_tick编号带上了起点偏移(self) -> None:
        bars = ohlc_from_series(np.arange(6, dtype=float), window=3, from_tick=100)
        self.assertEqual([b.tick_open for b in bars], [100, 103])
        self.assertEqual(bars[0].tick_close, 102)

    def test_成交量按同窗口求和(self) -> None:
        mid = np.arange(4, dtype=float)
        vol = np.array([1.0, 2.0, 3.0, 4.0])
        bars = ohlc_from_series(mid, window=2, volume=vol)
        self.assertAlmostEqual(bars[0].volume, 3.0)
        self.assertAlmostEqual(bars[1].volume, 7.0)

    def test_长度不符直接报错不带病通过(self) -> None:
        """⭐ 静默截断会让图和数错位，而且看不出错。"""
        with self.assertRaises(ValueError):
            ohlc_from_series(np.arange(5, dtype=float), window=2,
                             volume=np.arange(3, dtype=float))
        with self.assertRaises(ValueError):
            ohlc_from_series(np.arange(5, dtype=float), window=2,
                             n_trades=np.arange(2))

    def test_非法窗口报错(self) -> None:
        with self.assertRaises(ValueError):
            ohlc_from_series(np.arange(5, dtype=float), window=0)

    def test_空输入返回空(self) -> None:
        self.assertEqual(ohlc_from_series(np.array([]), window=3), [])

    def test_跳过非有限值但不炸(self) -> None:
        mid = np.array([1.0, np.nan, 3.0, 4.0])
        bars = ohlc_from_series(mid, window=2)
        # 第 0 根 = [1.0, nan]：有效值只有 1.0
        self.assertEqual(bars[0].high, 1.0)
        self.assertEqual(bars[0].low, 1.0)
        # 第 1 根 = [3.0, 4.0]
        self.assertEqual(bars[1].high, 4.0)
        self.assertEqual(bars[1].low, 3.0)
        # 关键：nan 不能把极值污染成 nan（否则整根蜡烛画不出来）
        self.assertTrue(math.isfinite(bars[0].high))
        self.assertTrue(math.isfinite(bars[0].close))

    def test_整根都是nan时跳过该根而不是产出坏蜡烛(self) -> None:
        mid = np.array([np.nan, np.nan, 5.0, 6.0])
        bars = ohlc_from_series(mid, window=2)
        self.assertEqual(len(bars), 1)
        self.assertEqual(bars[0].open, 5.0)

    def test_payload是列式的(self) -> None:
        bars = ohlc_from_series(np.arange(4, dtype=float), window=2)
        p = bars_payload(bars)
        self.assertEqual(set(p), {"t", "te", "o", "h", "l", "c", "v", "nt"})
        self.assertEqual(len(p["o"]), 2)
        self.assertEqual(len(p["t"]), len(p["c"]))


class TestDrawdown(unittest.TestCase):
    def test_已知路径的回撤(self) -> None:
        # 100 → 120 → 90 → 110：峰值 120、谷 90 ⇒ 回撤 −25%
        eq = np.array([100.0, 120.0, 90.0, 110.0])
        d = equity_drawdown(eq)
        self.assertAlmostEqual(d["max_dd"], -30.0, places=9)
        self.assertAlmostEqual(d["max_dd_pct"], -0.25, places=9)
        self.assertEqual(d["peak_tick"], 1)
        self.assertEqual(d["trough_tick"], 2)
        self.assertEqual(d["duration"], 1)

    def test_单调上升没有回撤(self) -> None:
        d = equity_drawdown(np.array([1.0, 2.0, 3.0]))
        self.assertAlmostEqual(d["max_dd"], 0.0, places=9)

    def test_样本不足时给None而不是假数字(self) -> None:
        d = equity_drawdown(np.array([1.0]))
        self.assertIsNone(d["max_dd"])
        self.assertEqual(d["n"], 1)

    def test_返回值带样本数便于核对(self) -> None:
        """本项目踩过 equity[0] 起点偏移的坑 —— 让调用者能核对长度。"""
        d = equity_drawdown(np.arange(10, dtype=float))
        self.assertEqual(d["n"], 10)


class TestFillStats(unittest.TestCase):
    @staticmethod
    def _rows(pnls: list[float]) -> list[dict]:
        return [{"pnl": p} for p in pnls]

    def test_盈亏比与期望值(self) -> None:
        s = fill_stats(self._rows([100.0, 50.0, -40.0, -30.0]))
        self.assertAlmostEqual(s["gross_win"], 150.0)
        self.assertAlmostEqual(s["gross_loss"], 70.0)
        self.assertAlmostEqual(s["profit_factor"], 150.0 / 70.0)
        self.assertAlmostEqual(s["expectancy"], 20.0)

    def test_最大连续亏损(self) -> None:
        s = fill_stats(self._rows([10.0, -1.0, -2.0, -3.0, 5.0, -1.0]))
        self.assertEqual(s["max_consecutive_loss"], 3)

    def test_全部盈利时盈亏比为None(self) -> None:
        """没有亏损 ⇒ 比值无定义，给 None 而不是 inf。"""
        s = fill_stats(self._rows([10.0, 20.0]))
        self.assertIsNone(s["profit_factor"])
        self.assertEqual(s["gross_loss"], 0.0)

    def test_集中度_前三笔占比(self) -> None:
        s = fill_stats(self._rows([100.0, 50.0, 25.0, -5.0]))
        self.assertAlmostEqual(s["top3_share"], 175.0 / 180.0)

    def test_刻意不提供胜率(self) -> None:
        """⭐ 做市策略的"这笔赚了"与"这笔是不是接刀"是两件事，
        单看盈亏混起来会误导（项目用 markout 专门分开它们）。"""
        s = fill_stats(self._rows([1.0, -1.0]))
        self.assertNotIn("win_rate", s)

    def test_空列表不崩(self) -> None:
        s = fill_stats([])
        self.assertEqual(s["n"], 0)
        self.assertIsNone(s["profit_factor"])


class TestReportFlags(unittest.TestCase):
    def test_样本太小被指出(self) -> None:
        flags = report_flags({"n": 5})
        self.assertTrue(any("样本太小" in f for f in flags))

    def test_样本中等给出提示(self) -> None:
        flags = report_flags({"n": 50})
        self.assertTrue(any("不够下结论" in f for f in flags))

    def test_盈亏比过低被指出(self) -> None:
        flags = report_flags({"n": 200, "profit_factor": 0.8})
        self.assertTrue(any("亏钱" in f for f in flags))

    def test_期望值为负被指出(self) -> None:
        flags = report_flags({"n": 200, "expectancy": -3.0})
        self.assertTrue(any("没有优势" in f for f in flags))

    def test_利润集中被指出(self) -> None:
        flags = report_flags({"n": 200, "top3_share": 0.8})
        self.assertTrue(any("运气" in f for f in flags))

    def test_健康样本不误报(self) -> None:
        flags = report_flags({"n": 300, "profit_factor": 1.8,
                              "expectancy": 25.0, "top3_share": 0.2})
        self.assertEqual(flags, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
