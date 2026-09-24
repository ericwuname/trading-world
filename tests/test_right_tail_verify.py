"""右尾手册验证的回归测试：闭式解 E(N,p) = (2p)^N − 1 必须与文档 §2.2 表逐格吻合。

⭐ 这张表是《右尾策略完全手册》全部结论的地基——
「让利润奔跑不产生期望、只放大成本」整个论证都从它推出。
若哪天有人改动公式或文档，这里会先红。
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

from scripts.verify_right_tail_doc import E_closed, check_closed_form  # noqa: E402


class TestClosedForm(unittest.TestCase):
    def test_p等于零5时任意N期望恒为零(self):
        """文档铁律 1：p=0.50 ⇒ E=1^N−1=0，任意 N 严格为 0。"""
        for N in (1, 5, 10, 20, 30, 100):
            self.assertAlmostEqual(E_closed(N, 0.5), 0.0, places=12,
                                   msg=f"N={N} 时 p=0.5 的期望应严格为 0")

    def test_文档表25格全部吻合(self):
        res = check_closed_form()
        self.assertTrue(res["all_match"],
                        f"与文档 §2.2 不符的格：{res['mismatches'][:3]}")

    def test_p小于零5时N越大越接近负一(self):
        """文档：p<0.50 ⇒ (2p)^N → 0 ⇒ E → −1（N 越大越接近每串必亏）。"""
        prev = None
        for N in (5, 10, 20, 30):
            e = E_closed(N, 0.45)
            if prev is not None:
                self.assertLess(e, prev, "p=0.45 时 E 应随 N 单调下降")
            prev = e
        self.assertGreater(E_closed(30, 0.45), -1.0)
        self.assertLess(E_closed(200, 0.45), -0.999)

    def test_p大于零5时N越大越好(self):
        """文档：p>0.50 ⇒ (2p)^N → ∞。"""
        self.assertLess(E_closed(10, 0.52), E_closed(20, 0.52))
        self.assertGreater(E_closed(20, 0.52), 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
