"""A13 的测试：**"独立块"这个前提必须落成断言**。

⚠️ A13 的全部推断都建立在「时段1 与时段2 **不共享任何一根 K 线**」之上。
这种前提如果只在文字里"声称"，早晚会漂移（本项目在 A9b 就吃过一次：
声称"前 48 段已核对"，实际没有逐段比对）。
⇒ 所以把它写成**会失败的断言**，用**真实数据**去验。
"""

from __future__ import annotations

import dataclasses
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

from tw.marketdb import MarketStore  # noqa: E402
from tw.segmented import segment_ranges  # noqa: E402

INST = "BTCUSDT"
BAR = "1H"
SKIP = 2500          # 时段2 用的 skip（必须与实跑一致）


def _load_window(limit: int, skip: int = 0):
    st = MarketStore()
    try:
        s = st.load_candles("binance_csv", INST, BAR, limit=limit + skip)
    finally:
        st.close()
    if skip > 0:
        keep = len(s) - skip
        s = dataclasses.replace(
            s, timestamp=s.timestamp[:keep], open=s.open[:keep],
            high=s.high[:keep], low=s.low[:keep], close=s.close[:keep],
            volume=s.volume[:keep], confirm=s.confirm[:keep])
    return s


class TestEpochBlocksDoNotOverlap(unittest.TestCase):
    """⭐⭐ A13 的核心前提：两个时段块**不共享任何一根 K 线**。"""

    def test_时段1与时段2完全不重叠(self):
        need1 = 12 + 300 * 8 + 5          # 时段1（已有 BTC k300）的口径
        s1 = _load_window(need1)
        r1 = segment_ranges(len(s1), 8, min_history=12)[:300]
        lo1, hi1 = int(s1.timestamp[r1[0][0]]), int(s1.timestamp[r1[-1][1]])

        need2 = 12 + 250 * 8 + 5          # 时段2 的口径
        s2 = _load_window(need2, skip=SKIP)
        r2 = segment_ranges(len(s2), 8, min_history=12)[:250]
        self.assertTrue(r2, "时段2 切不出段 ⇒ 前提无法成立")
        lo2, hi2 = int(s2.timestamp[r2[0][0]]), int(s2.timestamp[r2[-1][1]])

        self.assertLess(hi2, lo1,
                        f"时段2 必须在时段1 **之前**结束："
                        f"{hi2} 应 < {lo1}（不重叠）")

    def test_时段2确实更早(self):
        """⚠️ 分辨力补强：不能"两边一样"也算通过。"""
        s0 = _load_window(12 + 250 * 8 + 5)
        s2 = _load_window(12 + 250 * 8 + 5, skip=SKIP)
        self.assertLess(int(s2.timestamp[-1]), int(s0.timestamp[-1]))

    def test_skip为零时不改变窗口(self):
        a = _load_window(200)
        b = _load_window(200, skip=0)
        self.assertEqual(len(a), len(b))
        self.assertEqual(int(a.timestamp[-1]), int(b.timestamp[-1]))

    def test_时段2段数足够(self):
        s2 = _load_window(12 + 250 * 8 + 5, skip=SKIP)
        r2 = segment_ranges(len(s2), 8, min_history=12)
        self.assertGreaterEqual(len(r2), 250, "时段2 必须够 250 段")


class TestPersistenceVerdict(unittest.TestCase):
    """`§4.2` 的「同号 / 异号」判定——这是"能不能持续"的结论开关。"""

    def _verdict(self, a: float, b: float) -> str:
        return "同号" if a * b > 0 else "异号"

    def test_两个都负判同号(self):
        self.assertEqual(self._verdict(-0.0035, -0.0021), "同号")

    def test_一正一负判异号(self):
        """⚠️ 异号必须被判成异号——它是"效应不持续"的信号。"""
        self.assertEqual(self._verdict(-0.0035, +0.0012), "异号")

    def test_零会判成异号而不是同号(self):
        """⚠️ 边界：0 乘任何数都不是正 ⇒ 落到"异号"分支。
        这是**保守**的方向（宁可说"不一致"），但要在报告里看清楚。"""
        self.assertEqual(self._verdict(0.0, -0.0035), "异号")


class TestBlockPoolingAtK4(unittest.TestCase):
    """k=4 时的临界值 —— A13 说"再加 1 块就到显著"，靠的是这个数。"""

    def test_k等于4时临界约3点18(self):
        from scripts.a11_cross_asset import pool_instruments
        def s(eff, se, n=250):
            return {"n": n, "effect": eff, "se": se, "sd": se * (n ** 0.5),
                    "mde": 2.86 * se, "lo": eff - 2 * se, "hi": eff + 2 * se,
                    "t": eff / se if se else float("nan"),
                    "verdict": "依然无法判定", "diffs": []}
        pl = pool_instruments([("a", s(-0.0035, 0.0019)),
                              ("b", s(-0.0033, 0.0017)),
                              ("c", s(-0.0094, 0.0084)),
                              ("d", s(-0.0030, 0.0019))])
        self.assertEqual(pl["df"], 3)
        self.assertAlmostEqual(pl["t_crit"], 3.182, places=3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
