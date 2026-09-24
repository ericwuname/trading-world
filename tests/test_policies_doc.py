"""两个文档策略（网格/右尾）的单元测试——接入系统前的行为断言。

⚠️ A14 的教训：策略实现的 bug 会被「难看的结果」掩盖。
这两个策略接进系统前，先用**已知答案的可见状态**验行为。
"""

from __future__ import annotations

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

from tw.policies_doc import (  # noqa: E402
    GridPolicyDoc, RightTailPolicyDoc, grid_levels,
)


def _vis(mid: float, inv: float = 0.0, equity: float = 100_000.0,
         closes: list[float] | None = None) -> dict:
    return {"mid": mid, "inventory": inv, "equity": equity,
            "cash": equity - inv * mid,
            "recent_closes": closes if closes is not None else [mid] * 64}


class TestGridPolicy(unittest.TestCase):
    def test_首根建初始半仓(self):
        """文档 §4.2：初始仓位 0.5。"""
        g = GridPolicyDoc()
        out = g.decide(_vis(100.0), max_size=10.0)
        self.assertEqual(out["action"], "buy")
        self.assertAlmostEqual(out["sz"], 5.0, places=9)   # 一半

    def test_下穿买区格位触发买入(self):
        g = GridPolicyDoc()
        g.decide(_vis(100.0), max_size=10.0)               # 建半仓，units=5
        g._prev_mid, g._prev_inv = 95.0, 5.0               # 模拟已成交
        lv = grid_levels(100.0, 0.25, 10)
        lower = [L for L in lv[:5]]                        # 买区
        out = g.decide(_vis(lower[-1]), max_size=10.0)     # 跌到最低买区格
        self.assertEqual(out["action"], "buy", f"下穿买区应买入：{out['reason']}")

    def test_横盘不动(self):
        g = GridPolicyDoc()
        g.decide(_vis(100.0), max_size=10.0)
        g._prev_mid, g._prev_inv = 100.0, 5.0
        out = g.decide(_vis(100.0, inv=5.0), max_size=10.0)
        self.assertEqual(out["action"], "hold", "横盘不该触发任何格位")

    def test_卖出不超过持仓_不做空(self):
        """⚠️ 系统是净头寸账户：卖超了会开空 ⇒ 必须用 inventory 守卫。"""
        g = GridPolicyDoc()
        g.decide(_vis(100.0), max_size=10.0)
        g._prev_mid, g._prev_inv = 120.0, 0.5              # 只剩 0.5 格
        lv = grid_levels(100.0, 0.25, 10)
        upper = lv[6]
        out = g.decide(_vis(upper, inv=0.5), max_size=10.0)
        if out["action"] == "sell":
            self.assertLessEqual(out["sz"], 0.5 + 1e-9, "卖出量不得超过持仓")

    def test_21天无成交触发清仓(self):
        """⚠️ 清仓发生在第 504 根，其后是 hold ⇒ 必须扫整个循环找那笔卖单
        （我第一版只看最后一根，被后续 hold 覆盖，误判失败）。"""
        g = GridPolicyDoc(exit_no_fill_bars=504)
        g.decide(_vis(100.0), max_size=10.0)
        g._prev_mid, g._prev_inv = 100.0, 5.0
        sells = []
        for _ in range(600):
            out = g.decide(_vis(100.0, inv=5.0), max_size=10.0)
            if out["action"] == "sell":
                sells.append(out["sz"])
        self.assertTrue(any(abs(sz - 5.0) < 1e-6 for sz in sells),
                        f"21 天无成交必须清仓（§3）；卖单={sells}")


class TestRightTailPolicy(unittest.TestCase):
    def _closes(self, breakout: str | None) -> list[float]:
        base = [100.0] * 64
        if breakout == "up":
            base[-1] = 105.0                                # 突破前高
        if breakout == "down":
            base[-1] = 95.0
        return base

    def test_向上突破挂多单且带止损止盈(self):
        g = RightTailPolicyDoc(stop_pct=0.02, target_pct=0.20)
        out = g.decide(_vis(105.0, inv=0.0, closes=self._closes("up")),
                       max_size=10.0)
        self.assertEqual(out["action"], "buy")
        self.assertAlmostEqual(out["sl"], 105.0 * 0.98, places=6)
        self.assertAlmostEqual(out["tp"], 105.0 * 1.20, places=6)

    def test_向下突破挂空单(self):
        g = RightTailPolicyDoc()
        out = g.decide(_vis(95.0, inv=0.0, closes=self._closes("down")),
                       max_size=10.0)
        self.assertEqual(out["action"], "sell")

    def test_未突破不动(self):
        g = RightTailPolicyDoc()
        out = g.decide(_vis(100.0, inv=0.0, closes=self._closes(None)),
                       max_size=10.0)
        self.assertEqual(out["action"], "hold")

    def test_持仓中不动_等tp_sl(self):
        g = RightTailPolicyDoc()
        out = g.decide(_vis(105.0, inv=2.0, closes=self._closes("up")),
                       max_size=10.0)
        self.assertEqual(out["action"], "hold", "有仓时应等 tp/sl 触发")

    def test_仓位按风险定_止损打掉只亏risk_frac(self):
        g = RightTailPolicyDoc(stop_pct=0.02, risk_frac=0.02)
        # ⚠️ max_size 要给够大：风控上限会截断风险定价的仓位
        #（截断是**正确行为**——我第一版给 100，截断后对不上账，差点当成 bug）
        # ⚠️ mid 必须与 closes 一致（突破价 105）：
        #    我第一版 mid=100 但 closes[-1]=105 ⇒ 无突破 ⇒ hold ⇒ sz=0。
        out = g.decide(_vis(105.0, inv=0.0, closes=self._closes("up")),
                       max_size=10_000.0)
        sz = out["sz"]
        # ⚠️ 止损距离 = 突破价 × stop_pct（105×2% = 2.1），
        #    我第一版硬编码 100 ⇒ 算出 0（sz 已按 105 定过风险，差价对不上）。
        loss_at_stop = sz * 105.0 * 0.02
        self.assertAlmostEqual(loss_at_stop, 100_000.0 * 0.02, delta=1.0,
                               msg="止损打掉的损失应≈equity×risk_frac")


if __name__ == "__main__":
    unittest.main(verbosity=2)
