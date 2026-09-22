"""k 不确定度诊断工具的回归测试。

为什么它也要被测
----------------
这份诊断现在被用来**推翻一批既有结论**（"k 从 1.395 改善到 0.455"）。
一个用来否定别人的工具，自己必须先是可信的——否则它造成的伤害比不做还大
（本项目已经吃过这个亏：自检工具自己误报，见 ``TestSelfcheckPlaceholderExemptions``）。

所以这里测三件事：
  ① ``seeds_needed`` 的算术（能对上解析式）；
  ② ``bootstrap_k`` 在**零噪声**输入下退化成一个点（说明它确实在算拟合，不是恒返回宽区间）；
  ③ ``bootstrap_k`` 在**有噪声**输入下给出宽区间（说明它确实在传播不确定性）；
  ②与③互为反面参照——只有②会「永远说不可分辨」，只有③会「永远说显著」。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / "scripts"))  # 追加：插到前面会遮蔽同名包（scripts/gui.py vs gui/）

import numpy as np  # noqa: E402

from scripts.diagnose_k_uncertainty import (  # noqa: E402
    bootstrap_k,
    seeds_needed,
)


def _rows(bps, sems, qty=None, t=None, n_seeds=3):
    """构造一臂的 rows（只放诊断用得上的字段）。"""
    qty = qty or [3.0, 7.5, 15.0, 30.0]
    return [
        {"name": f"x|{i}", "shock_frac": 0.001 * (i + 1),
         "delivered_qty": q, "slippage_bp": bp, "slippage_sem_bp": se,
         "slippage_t": (bp / se if se else 0.0) if t is None else t,
         "fill_ratio": 1.0, "n_seeds": n_seeds}
        for i, (q, bp, se) in enumerate(zip(qty, bps, sems))
    ]


class TestSeedsNeededArithmetic(unittest.TestCase):
    """宽度 ∝ 1/√n ⇒ n_need = n·(W/Δ)²。"""

    def test_对上解析式(self) -> None:
        # 3 × (2.0/0.5)² = 3 × 16 = 48
        self.assertAlmostEqual(seeds_needed(2.0, 0.5, n=3), 48.0)

    def test_目标越细需要的种子越多(self) -> None:
        self.assertLess(seeds_needed(2.0, 0.6), seeds_needed(2.0, 0.1))

    def test_种子越多区间越窄_反比于根号n(self) -> None:
        # 从 n=3 换到 n=12（4 倍），所需种子数随 (W/Δ)² 变化，
        # 而 W 自身 ∝ 1/√n ⇒ 结论：把 W 从 2.0 降到 1.0 需要 n×4
        self.assertAlmostEqual(seeds_needed(1.0, 1.0, n=12), 12.0)

    def test_非法输入不崩且返回nan(self) -> None:
        self.assertTrue(np.isnan(seeds_needed(0.0, 1.0)))
        self.assertTrue(np.isnan(seeds_needed(1.0, 0.0)))


class TestBootstrapIsNotDegenerate(unittest.TestCase):
    """② vs ③：零噪声 → 点；有噪声 → 宽区间。缺任一面都说明工具不可信。"""

    def test_零噪声时退化成一个点(self) -> None:
        """sem=0 ⇒ 分布应当塌缩到点估计附近（宽度≈0）。"""
        bps = [-0.5, -2.0, -5.0, -10.0]      # 完美的幂律：k≈1.something
        sems = [0.0, 0.0, 0.0, 0.0]
        r = bootstrap_k(_rows(bps, sems), n_boot=200)
        self.assertTrue(r["ok"], r.get("reason"))
        self.assertAlmostEqual(r["k_point"], r["k_median"], places=6)
        self.assertLess(r["width"], 1e-6,
                        f"零噪声下区间宽度居然有 {r['width']}——诊断在制造假的不确定性")

    def test_有噪声时给出非零宽度(self) -> None:
        bps = [-0.5, -2.0, -5.0, -10.0]
        sems = [1.0, 1.0, 1.0, 1.0]
        r = bootstrap_k(_rows(bps, sems), n_boot=2_000)
        self.assertTrue(r["ok"], r.get("reason"))
        self.assertGreater(r["width"], 0.05,
                           "有噪声却给出近乎零的宽度——不确定性没有传播出去")

    def test_噪声越大区间越宽(self) -> None:
        bps = [-0.5, -2.0, -5.0, -10.0]
        w_small = bootstrap_k(_rows(bps, [0.2] * 4), n_boot=1_500)["width"]
        w_big = bootstrap_k(_rows(bps, [2.0] * 4), n_boot=1_500)["width"]
        self.assertLess(w_small, w_big,
                        f"噪声小的区间({w_small:.3f}) 反而比噪声大的({w_big:.3f}) 宽")

    def test_点估计与fit_power_law一致(self) -> None:
        """bootstrap 的中位数应当落在点估计附近——说明复用的是同一个拟合。"""
        from tw.impact import fit_power_law
        bps = [-0.5, -2.0, -5.0, -10.0]
        rows = _rows(bps, [0.3] * 4)
        want = fit_power_law(rows, "slippage_bp", x_key="delivered_qty").exponent
        r = bootstrap_k(rows, n_boot=2_000)
        self.assertAlmostEqual(r["k_point"], want, places=9)
        self.assertLess(abs(r["k_median"] - want), 0.15,
                        "bootstrap 中位数偏离点估计太远，两者的口径可能不是同一个")

    def test_档位不足时明确失败而不是给个数(self) -> None:
        """只有 2 档时不许硬拟合出 k —— 必须报"样本不足"。"""
        r = bootstrap_k(_rows([-0.5, -2.0], [0.1, 0.1], qty=[3.0, 7.5]), n_boot=50)
        self.assertFalse(r["ok"])
        self.assertIn("档位", r["reason"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
