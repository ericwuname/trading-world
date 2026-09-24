"""网格回测的健全性测试：**逻辑对不对，用已知答案的场景验**。

⚠️ 这些测试守护的是"实现没有系统性错误"：
- 横盘 ⇒ 不该有成交、不该亏钱；
- 规则正弦震荡 ⇒ 网格**必须赚钱**（低买高卖收割波动）——
  这是网格存在的唯一理由，若它不赚，实现一定有 bug；
- 单边下跌 ⇒ 网格必须亏（接飞刀），且 21 天离场必须把回撤压下来。
"""

from __future__ import annotations

import dataclasses
import math
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

from scripts.grid_backtest_doc import grid_levels, run_grid  # noqa: E402


class _Series:
    """最小化 Series 鸭子类型（grid 只用 open/high/low/close）。"""

    def __init__(self, close, high=None, low=None):
        self.close = close
        self.high = high if high is not None else close
        self.low = low if low is not None else close


class TestGridGeometry(unittest.TestCase):
    def test_文档例题_格距524(self):
        lv = grid_levels(100.0, 0.25, 10)
        self.assertAlmostEqual(lv[0], 75.0, places=9)
        self.assertAlmostEqual(lv[-1], 125.0, places=9)
        g = lv[1] / lv[0] - 1
        self.assertAlmostEqual(g * 100, 5.241, places=2)


class TestGridSanity(unittest.TestCase):
    def test_横盘无成交不亏钱(self):
        n = 3000
        s = _Series([100.0] * n, [100.2] * n, [99.8] * n)
        r = run_grid(s, k=0.25, n=10, exit_rule=False,
                     restart_on_break=False, recenter_every=0)
        self.assertEqual(r.n_fills, 0, "±0.2% 的波动不该碰到 5.24% 的格距")
        self.assertLess(r.max_drawdown, 0.01)

    def test_覆盖全梯子的正弦震荡必须赚钱(self):
        """⭐ 核心健全性：震荡**覆盖整个梯子**时网格必须赚（低买高卖收割波动）。

        ⚠️ 振幅要 ≥ k（±25%）且略超出，这样每个格位与其 TP 都会被触及；
        若振幅只有 ±20%，顶部格位（119→TP 125）永远无法止盈 ⇒
        那是**策略特性**（区间外沿的格子被套住），不是实现 bug——
        我第一版用 ±20% 振幅测出 -8%，差点把策略特性误判成实现错误。
        """
        n = 6000
        amp = 0.26                                   # ±26%，略超 ±25% 的梯子
        close, high, low = [], [], []
        for i in range(n):
            p = 100.0 * (1 + amp * math.sin(2 * math.pi * i / 240))
            close.append(p)
            high.append(p * 1.01)
            low.append(p * 0.99)
        r = run_grid(_Series(close, high, low), k=0.25, n=10,
                     exit_rule=False, restart_on_break=False,
                     recenter_every=0)
        self.assertGreater(r.annualized, 0.02,
                           f"全梯子震荡里网格必须赚钱，实测年化 {r.annualized:+.2%}")
        self.assertGreater(r.n_fills, 100, "震荡里该有大量成交")

    def test_单边下跌必须亏_且离场压回撤(self):
        n = 6000
        close = [100.0 * (0.9995 ** i) for i in range(n)]   # 缓慢单边跌
        high = [c * 1.005 for c in close]
        low = [c * 0.995 for c in close]
        off = run_grid(_Series(close, high, low), k=0.25, n=10,
                       exit_rule=False, restart_on_break=False,
                       recenter_every=0)
        on = run_grid(_Series(close, high, low), k=0.25, n=10,
                      exit_rule=True, restart_on_break=False,
                      recenter_every=0)
        self.assertLess(off.total_return, 0, "单边下跌里网格必须亏")
        self.assertLess(on.max_drawdown, off.max_drawdown,
                        "21 天离场必须把回撤压下来")

    def test_初始半仓(self):
        """文档 §4.2：初始仓位 0.5 ⇒ 起始权益 = 本金（现金+半仓市值）。"""
        n = 500
        s = _Series([100.0] * n, [100.0] * n, [100.0] * n)
        r = run_grid(s, k=0.25, n=10, exit_rule=False,
                     restart_on_break=False, recenter_every=0,
                     initial_cash=100_000.0)
        self.assertAlmostEqual(r.final_equity, 100_000.0, delta=50,
                               msg="横盘 + 半仓 ⇒ 权益应≈本金")


if __name__ == "__main__":
    unittest.main(verbosity=2)
