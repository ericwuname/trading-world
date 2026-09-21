"""A6 测试：结果回填（证据链第 ⑥ 项）。

本文件的核心只有一件事：**回填只准动 ``outcome``，绝不碰 ``visible_state``**。

为什么它值得单独一组测试：一旦 outcome 渗进可见状态，
"复盘学到的经验"就会带着未来信息回流到决策里 —— 整条反馈链路失去意义，
而**症状是收益看起来变好**（因为它在偷看答案）。这类错误必须机械拦住。
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from tw.decision_log import (
    DecisionRecord,
    build_visible_state,
    state_digest,
)
from tw.outcome import (
    BackfillConfig,
    backfill,
    backfill_one,
    outcome_summary,
)


class _Series:
    def __init__(self, closes, spread: float = 0.004):
        self.close = np.asarray(closes, dtype=float)
        self.open = self.close
        self.high = self.close * (1.0 + spread)
        self.low = self.close * (1.0 - spread)
        self.timestamp = np.arange(len(self.close), dtype=np.int64) * 3_600_000


def _rec(tick: int, action: str = "buy", *, executed: bool | None = None,
         mid: float = 100.0) -> DecisionRecord:
    vis = build_visible_state(mid=float(mid), equity=100_000.0)
    return DecisionRecord(
        tick=int(tick), visible_state=vis, parsed={"action": action},
        executed=(action != "hold" if executed is None else executed),
    )


# ======================================================================
# ⭐ 核心守卫：只动 outcome
# ======================================================================
class TestOnlyTouchesOutcome(unittest.TestCase):
    def test_回填不改visible_state(self):
        """⭐⭐ **本文件最重要的一条。**

        回填前后 ``state_digest(visible_state)`` 必须**完全相同**。
        它把"回填没污染输入"从"靠人记得别手滑"变成**机械可验证的等式**。
        """
        s = _Series([100, 101, 102, 103, 104, 105])
        recs = [_rec(i) for i in range(4)]
        before = [state_digest(r.visible_state) for r in recs]
        snap = [dict(r.visible_state) for r in recs]

        backfill(recs, s, cfg=BackfillConfig(horizon=2))

        for r, d, v in zip(recs, before, snap):
            self.assertEqual(state_digest(r.visible_state), d,
                             "回填改动了 visible_state —— 未来信息渗进输入了")
            self.assertEqual(r.visible_state, v, "可见状态的字段被动了")

    def test_回填确实写了outcome(self):
        s = _Series([100, 101, 102, 103, 104, 105])
        r = _rec(0)
        self.assertFalse(r.outcome_filled)
        self.assertTrue(backfill_one(r, s, cfg=BackfillConfig(horizon=2)))
        self.assertTrue(r.outcome_filled)
        self.assertIn("markout_h", r.outcome)
        self.assertEqual(r.outcome["horizon"], 2)

    def test_单条回填也守住(self):
        """逐条那条路径同样要守（批量那条不能是唯一的守卫）。"""
        s = _Series([100, 101, 102, 103, 104, 105])
        r = _rec(1)
        d = state_digest(r.visible_state)
        backfill_one(r, s, cfg=BackfillConfig(horizon=3))
        self.assertEqual(state_digest(r.visible_state), d)


# ======================================================================
# ⭐ 回填延迟 = 复盘的可见边界
# ======================================================================
class TestPending(unittest.TestCase):
    def test_未来未发生的跳过(self):
        """⭐ 这条实现了"复盘不可能事后诸葛亮"。

        当前时刻 t，只有 ``t + horizon <= t_now`` 的决策才有 outcome。
        ⇒ 复盘时**拿不到**未实现的结果。
        """
        s = _Series([100] * 10)
        near_end = _rec(9)          # 展望 2 根会越界
        self.assertFalse(backfill_one(near_end, s, cfg=BackfillConfig(horizon=2)))
        self.assertFalse(near_end.outcome_filled)
        self.assertEqual(near_end.outcome, {})

    def test_恰好够窗口就回填(self):
        """边界：``t + horizon`` **正好**是最后一根 —— 应当回填。"""
        s = _Series([100] * 6)
        r = _rec(4)
        self.assertTrue(backfill_one(r, s, cfg=BackfillConfig(horizon=1)))

    def test_统计分开pending与bad_input(self):
        """⚠️ "未来未发生"与"输入不可用"必须分开：

        前者是设计（正常），后者是数据问题。混在一起会让人
        以为"跳过很多 = 正常"，从而漏掉数据缺陷。
        """
        s = _Series([100] * 10)
        recs = [_rec(0), _rec(8), _rec(9)]
        st = backfill(recs, s, cfg=BackfillConfig(horizon=2))
        self.assertEqual(st["n_filled"], 1)      # tick=0
        self.assertEqual(st["n_pending"], 2)     # tick=8/9 未来未到
        self.assertEqual(st["n_bad_input"], 0)

    def test_重复回填不重复计数(self):
        s = _Series([100] * 10)
        recs = [_rec(0), _rec(1)]
        c = BackfillConfig(horizon=2)
        backfill(recs, s, cfg=c)
        st = backfill(recs, s, cfg=c)      # 再来一次
        self.assertEqual(st["n_filled"], 0, "已回填的不该重复计数")


# ======================================================================
# 方向与数值
# ======================================================================
class TestMarkout(unittest.TestCase):
    def test_多头上漲算对(self):
        s = _Series([100, 105, 110])
        r = _rec(0, "buy")
        backfill_one(r, s, cfg=BackfillConfig(horizon=1))
        self.assertGreater(r.outcome["markout_h"], 0)
        self.assertAlmostEqual(r.outcome["markout_h"], 0.05, places=9)

    def test_空头上漲算错(self):
        s = _Series([100, 105, 110])
        r = _rec(0, "sell")
        backfill_one(r, s, cfg=BackfillConfig(horizon=1))
        self.assertLess(r.outcome["markout_h"], 0)

    def test_空头下跌算对(self):
        s = _Series([100, 95, 90])
        r = _rec(0, "sell")
        backfill_one(r, s, cfg=BackfillConfig(horizon=1))
        self.assertGreater(r.outcome["markout_h"], 0)

    def test_弃权没有方向所以markout为零(self):
        """⚠️ 弃权没有"对错方向"——它的质量要靠**弃权质量**另外衡量。

        这里给 0 并把 ``markout_applies`` 置 False，
        免得统计时把弃权混进"决策准确率"。
        """
        s = _Series([100, 105, 110])
        r = _rec(0, "hold")
        backfill_one(r, s, cfg=BackfillConfig(horizon=1))
        self.assertEqual(r.outcome["markout_h"], 0.0)
        self.assertFalse(r.outcome["markout_applies"])

    def test_未成交的markout标记为不适用(self):
        """⭐ 没成交时 markout 是"如果当时做了会怎样"，不是成绩。"""
        s = _Series([100, 110, 120])
        r = _rec(0, "buy", executed=False)
        backfill_one(r, s, cfg=BackfillConfig(horizon=1))
        self.assertFalse(r.outcome["filled"])
        self.assertFalse(r.outcome["markout_applies"])
        self.assertGreater(r.outcome["markout_h"], 0)   # 数值仍算，但标为不适用

    def test_窗口外的偏移用正确的根(self):
        s = _Series([100, 101, 102, 103, 104])
        for h in (1, 2, 3):
            r = _rec(0, "buy")
            backfill_one(r, s, cfg=BackfillConfig(horizon=h))
            self.assertAlmostEqual(r.outcome["mid_after_h"],
                                   float(s.close[h]), places=9)

    def test_mfe_mae_存在(self):
        s = _Series([100, 101, 102, 103])
        r = _rec(0, "buy")
        backfill_one(r, s, cfg=BackfillConfig(horizon=2, with_mfe_mae=True))
        self.assertIn("mfe", r.outcome)
        self.assertIn("mae", r.outcome)
        self.assertLessEqual(r.outcome["mae"], r.outcome["mfe"])

    def test_关闭mfe_mae时不写(self):
        s = _Series([100, 101, 102])
        r = _rec(0)
        backfill_one(r, s, cfg=BackfillConfig(horizon=1, with_mfe_mae=False))
        self.assertNotIn("mfe", r.outcome)

    def test_mid不可用时算输入问题(self):
        class Bad:
            close = np.array([100.0, float("nan"), 102.0])
            high = low = close
        r = _rec(1)
        self.assertFalse(backfill_one(r, Bad(),
                                      cfg=BackfillConfig(horizon=1)))


# ======================================================================
# 摘要（给复盘用）
# ======================================================================
class TestSummary(unittest.TestCase):
    def test_未到结果时说明白(self):
        r = _rec(0)
        self.assertIn("未到", outcome_summary(r))

    def test_对的被判对(self):
        s = _Series([100, 110, 120])
        r = _rec(0, "buy")
        backfill_one(r, s, cfg=BackfillConfig(horizon=1))
        self.assertIn("对", outcome_summary(r))

    def test_错的被判错(self):
        s = _Series([100, 110, 120])
        r = _rec(0, "sell")
        backfill_one(r, s, cfg=BackfillConfig(horizon=1))
        self.assertIn("错", outcome_summary(r))

    def test_未成交标为未成交(self):
        s = _Series([100, 110, 120])
        r = _rec(0, "buy", executed=False)
        backfill_one(r, s, cfg=BackfillConfig(horizon=1))
        self.assertIn("未成交", outcome_summary(r))


# ======================================================================
# 配置
# ======================================================================
class TestConfig(unittest.TestCase):
    def test_horizon必须为正(self):
        with self.assertRaises(ValueError):
            BackfillConfig(horizon=0)
        with self.assertRaises(ValueError):
            BackfillConfig(horizon=-1)

    def test_默认值(self):
        c = BackfillConfig()
        self.assertGreaterEqual(c.horizon, 1)
        self.assertTrue(c.with_mfe_mae)


if __name__ == "__main__":
    unittest.main(verbosity=2)
