"""A11 的测试：**前提检查的三种状态**与**跨标的合并**。

⚠️ 这一组测的是"分析脚本自己会不会骗人"：
- 把「**没有数据**」误报成「**前提被违反**」，会让读者以为设计错了；
- 把「**两个标的符号相反**」合并成一个接近 0 的数，
  再写成「没有效应」——那是**把抵消说成不存在**。

两个坑我都当真踩过/差点踩到，所以各写成一条断言。
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))
# ⚠️ 用 **append**：插到前面会让 `scripts/gui.py` 遮蔽 `gui/` 包
if str(ROOT / "scripts") not in sys.path:
    sys.path.append(str(ROOT / "scripts"))

from scripts.a11_cross_asset import (  # noqa: E402
    check_same_instructions, pool_instruments, pooled_mc_calibration, stat,
)

KPI_A = {"target_return": -0.00022384643115001381,
         "min_presence": 0.4444444444444444,
         "max_drawdown": 0.005649651603050114,
         "turnover_lo": 0.1983280232142,
         "turnover_hi": 0.9841583672147}


def _write(d: Path, name: str, kpi: dict | None) -> Path:
    p = d / name
    payload = {"result": {"net": []}}
    if kpi is not None:
        payload["kpi"] = kpi
    p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return p


class TestPremiseHasThreeStates(unittest.TestCase):
    """⭐⭐ **「没有数据」与「前提被违反」必须分开**（我第一版把两者混成一个）。"""

    def setUp(self):
        self.d = Path(tempfile.mkdtemp())

    def test_阈值相同(self):
        a = _write(self.d, "a.json", dict(KPI_A))
        b = _write(self.d, "b.json", dict(KPI_A))
        r = check_same_instructions(a, b)
        self.assertEqual(r["status"], "same")
        self.assertTrue(r["same"])
        self.assertEqual(r["diffs"], [])

    def test_阈值不同(self):
        a = _write(self.d, "a.json", dict(KPI_A))
        b = _write(self.d, "b.json", {**KPI_A, "min_presence": 0.3333})
        r = check_same_instructions(a, b)
        self.assertEqual(r["status"], "different")
        self.assertFalse(r["same"])
        self.assertEqual(len(r["diffs"]), 1)
        self.assertEqual(r["diffs"][0][0], "min_presence")

    def test_产物不存在时报待跑而不是报前提违反(self):
        a = _write(self.d, "a.json", dict(KPI_A))
        b = self.d / "不存在.json"
        r = check_same_instructions(a, b)
        self.assertEqual(r["status"], "missing")
        self.assertIn(str(b), r["missing"][0])

    def test_产物在但没有kpi块也算待跑(self):
        """⚠️ 那个臂可能还没跑（v4 臂本来就没有 kpi 块）。"""
        a = _write(self.d, "a.json", dict(KPI_A))
        b = _write(self.d, "b.json", None)
        r = check_same_instructions(a, b)
        self.assertEqual(r["status"], "missing")

    def test_浮点容差不该误报(self):
        """1e-13 级别的差不该被当成"阈值不同"。"""
        a = _write(self.d, "a.json", dict(KPI_A))
        b = _write(self.d, "b.json",
                   {**KPI_A, "turnover_lo": KPI_A["turnover_lo"] + 1e-13})
        self.assertEqual(check_same_instructions(a, b)["status"], "same")


class TestPooling(unittest.TestCase):
    """跨标的合并：**合法**（不同资产独立），但**不许把抵消说成不存在**。"""

    def _s(self, eff, se, n=300):
        # ⚠️ se=0 要单独挡（我第一版直接 `eff / se` ⇒ ZeroDivisionError）。
        t = (eff / se) if se > 0 else float("nan")
        return {"n": n, "effect": eff, "se": se, "sd": se * (n ** 0.5),
                "mde": 2.86 * se, "lo": eff - 2 * se, "hi": eff + 2 * se,
                "t": t, "verdict": "依然无法判定", "diffs": []}

    def test_少于两个标的时不构成复现(self):
        r = pool_instruments([("BTC", self._s(-0.001, 0.0002))])
        self.assertEqual(r["k"], 1)
        self.assertIn("少于 2 个标的", r["note"])

    def test_同号两标的会加强(self):
        r = pool_instruments([("BTC", self._s(-0.001, 0.0002)),
                              ("ETH", self._s(-0.001, 0.0002))])
        self.assertAlmostEqual(r["pooled"], -0.001, places=9)
        # 合并的 SE 比单个小（√2 倍）
        self.assertAlmostEqual(r["se"], 0.0002 / (2 ** 0.5), places=9)
        self.assertFalse(r["heterogeneous"])
        # ⚠️ 两个同号的合并 ⇒ |z| 明显大于单个
        self.assertLess(r["z"], -3)

    def test_反号合并接近零_但两项都还在(self):
        """⚠️⚠️ **不许把"抵消"说成"不存在"**：合并≈0，但两个标的各自的值必须都能看到。"""
        items = [("BTC", self._s(-0.002, 0.0002)),
                 ("ETH", self._s(+0.002, 0.0002))]
        r = pool_instruments(items)
        self.assertAlmostEqual(r["pooled"], 0.0, places=9)
        # ⚠️ 这正是"合并会掩盖抵消"的形态 ⇒ Q 检验必须显著
        self.assertTrue(r["heterogeneous"],
                        "两个反号的大效应合并成 0 时，Q 必须显著地指出异质")

    def test_效应不一致时Q显著(self):
        r = pool_instruments([("BTC", self._s(-0.01, 0.0001)),
                              ("ETH", self._s(+0.001, 0.0001))])
        self.assertTrue(r["heterogeneous"])
        self.assertGreater(r["Q"], r["critical"])

    def test_SE为零的标的被排除(self):
        r = pool_instruments([("BTC", self._s(-0.001, 0.0002)),
                              ("ETH", self._s(-0.001, 0.0))])
        self.assertEqual(r["k"], 1)

    def test_合并的SE小于任一单个(self):
        r = pool_instruments([("BTC", self._s(-0.001, 0.0002)),
                              ("ETH", self._s(-0.001, 0.0003))])
        self.assertLess(r["se"], 0.0002)


class TestSignConvention(unittest.TestCase):
    """⚠️ 符号约定：`d = pb − pa`，调用方须按「pa=基准、pb=处理」传。"""

    def test_处理减基准(self):
        d = Path(tempfile.mkdtemp())
        base = d / "base.json"
        kpi = d / "kpi.json"
        base.write_text(json.dumps(
            {"result": {"net": [{"llm": 0.01}, {"llm": 0.01}]}}),
            encoding="utf-8")
        kpi.write_text(json.dumps(
            {"result": {"net": [{"llm": 0.00}, {"llm": 0.02}]}}),
            encoding="utf-8")
        s = stat(base, "llm", kpi, "llm")     # (基准, 处理) ⇒ 处理 − 基准
        self.assertAlmostEqual(s["effect"], 0.0, places=12)

    def test_处理更差时效应为负(self):
        d = Path(tempfile.mkdtemp())
        base = d / "base.json"
        kpi = d / "kpi.json"
        base.write_text(json.dumps(
            {"result": {"net": [{"llm": 0.01}, {"llm": 0.01}]}}),
            encoding="utf-8")
        kpi.write_text(json.dumps(
            {"result": {"net": [{"llm": 0.00}, {"llm": 0.00}]}}),
            encoding="utf-8")
        s = stat(base, "llm", kpi, "llm")
        self.assertLess(s["effect"], 0.0, "处理更差 ⇒ 效应必须是负的")


class TestPooledStatisticIsCalibrated(unittest.TestCase):
    """⭐⭐ **宣布「显著」之前必须先验合并统计量本身**（A10 的纪律）。

    本轮第一次出现「合并后显著」（z≈−2.7）。若合并的 z 本身偏大，
    那这个显著就是**装置造出来的**，不是数据里的。
    ⇒ 用**有已知答案的合成数据**（正态残差、δ=0）反过来检验它。
    """

    def _normal(self, n, sd, seed):
        import random
        rng = random.Random(seed)
        return [rng.gauss(0, sd) for _ in range(n)]

    def test_delta为零时假阳性率约5个百分点(self):
        a = self._normal(300, 0.0033, 1)
        b = self._normal(400, 0.0033, 2)
        cal = pooled_mc_calibration(a, b, delta=0.0, reps=2000, seed=7)
        # 二项标准误 ≈ 0.49pp ⇒ 3σ ≈ 1.5pp
        self.assertLess(abs(cal["false_positive_rate"] - 0.05), 0.02,
                        f"假阳性率应 ≈5%，实测 {cal['false_positive_rate']:.1%}")

    def test_z的标准差约1(self):
        """⚠️ z 的**尺度**必须对（均值 0、标准差 1）——只查假阳性率会漏掉尺度偏差。"""
        a = self._normal(300, 0.0033, 3)
        b = self._normal(400, 0.0033, 4)
        cal = pooled_mc_calibration(a, b, delta=0.0, reps=2000, seed=8)
        self.assertLess(abs(cal["sd_z"] - 1.0), 0.12,
                        f"z 的标准差应 ≈1，实测 {cal['sd_z']:.3f}")
        self.assertLess(abs(cal["mean_z"]), 0.1)

    def test_注入大效应时几乎必然检出(self):
        """⚠️ 分辨力补强：不能"永远报不显著"。"""
        a = self._normal(300, 0.001, 5)
        b = self._normal(400, 0.001, 6)
        cal = pooled_mc_calibration(a, b, delta=0.01, reps=400, seed=9)
        self.assertGreater(cal["false_positive_rate"], 0.99)

    def test_固定种子可复现(self):
        a = self._normal(200, 0.003, 11)
        b = self._normal(200, 0.003, 12)
        c1 = pooled_mc_calibration(a, b, reps=300, seed=99)
        c2 = pooled_mc_calibration(a, b, reps=300, seed=99)
        self.assertEqual(c1, c2)

    def test_常量残差时不炸(self):
        """σ=0 的极端输入不许抛（本项目对 σ=0 有专门的教训）。"""
        cal = pooled_mc_calibration([0.0] * 50, [0.0] * 50, reps=20, seed=3)
        self.assertEqual(cal["reps"], 0)     # 全部被"se>0"挡掉 ⇒ 分母为空


if __name__ == "__main__":
    unittest.main(verbosity=2)
