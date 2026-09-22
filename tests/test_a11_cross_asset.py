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
    check_same_instructions, pool_instruments, pooled_mc_calibration,
    pooled_signflip_pvalue, stat,
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


class TestSmallKInterval(unittest.TestCase):
    """⭐⭐ **小 k 的合并区间必须用 `t(df=k−1)`，不是正态 1.96**。

    本项目的"显著"结论第一次出现在 k=3 的合并上。用 1.96 会把它报成显著
    （z=−2.83 > 1.96），但正确临界是 `t(df=2)=4.30` ⇒ **不显著**。
    实测也印证：正态近似的 z 在 δ=0 时假阳性率约 **9.9%**（反保守）。
    """

    def _s(self, eff, se, n=300):
        return {"n": n, "effect": eff, "se": se, "sd": se * (n ** 0.5),
                "mde": 2.86 * se, "lo": eff - 2 * se, "hi": eff + 2 * se,
                "t": (eff / se) if se > 0 else float("nan"),
                "verdict": "依然无法判定", "diffs": []}

    def test_k等于3时临界值约4点3(self):
        pl = pool_instruments([("BTC", self._s(-0.0035, 0.0012)),
                              ("ETH", self._s(-0.0035, 0.0012)),
                              ("SOL", self._s(-0.0035, 0.0012))])
        self.assertEqual(pl["df"], 2)
        self.assertAlmostEqual(pl["t_crit"], 4.303, places=2)

    def test_用t口径时窄区间会变成不显著(self):
        """|z|≈2.9 的效应：正态说显著，`t(df=2)` 说不显著——**后者才对**。"""
        # ⚠️ 参数要**调到能体现对比**：每组 se=0.001*√3 ⇒ 合并 se=0.001
        # ⇒ z = −0.0028/0.001 = −2.8（>1.96 但 < 4.30）。
        # 我第一版随手写了 −0.0035/0.0012 ⇒ 合并 |z|≈5 ⇒ 两个口径都显著，
        # **对比根本没出现**（那测试就白写了）。
        se_g = 0.001 * (3 ** 0.5)
        pl = pool_instruments([("BTC", self._s(-0.0028, se_g)),
                              ("ETH", self._s(-0.0028, se_g)),
                              ("SOL", self._s(-0.0028, se_g))])
        self.assertGreater(abs(pl["z"]), 1.96)      # 正态口径 ⇒ 显著
        self.assertLess(pl["ci"][1], 0)             # 正态区间不含 0
        # t 口径（临界 4.30）⇒ 区间更宽且**跨 0**
        self.assertLess(pl["ci_t"][0], pl["ci"][0])
        self.assertGreater(pl["ci_t"][1], pl["ci"][1])
        self.assertTrue(pl["ci_t"][0] < 0 < pl["ci_t"][1])   # 跨 0 ⇒ 不显著

    def test_k等于2时临界值约12点7(self):
        pl = pool_instruments([("BTC", self._s(-0.0035, 0.0012)),
                              ("ETH", self._s(-0.0035, 0.0012))])
        self.assertEqual(pl["df"], 1)
        self.assertAlmostEqual(pl["t_crit"], 12.706, places=2)


class TestSignFlipPvalue(unittest.TestCase):
    """随机化检验（不依赖正态近似）——⚠️ 它自己也有前提，要一并测出来。"""

    def _normal(self, n, sd, seed):
        import random
        rng = random.Random(seed)
        return [rng.gauss(0, sd) for _ in range(n)]

    def test_delta为零时p值不小(self):
        g = [self._normal(300, 0.003, 1), self._normal(300, 0.003, 2)]
        sf = pooled_signflip_pvalue(*g, reps=1000, seed=11)
        self.assertGreater(sf["p_value"], 0.2, "无效应时不该频繁报小 p")

    def test_注入大效应时p值很小(self):
        import random
        rng = random.Random(3)
        g = [[rng.gauss(0.01, 0.003) for _ in range(300)] for _ in range(2)]
        sf = pooled_signflip_pvalue(*g, reps=1000, seed=12)
        self.assertLess(sf["p_value"], 0.01)

    def test_观测z必须非零(self):
        """⚠️⚠️ 我第一版**先按组中心化** ⇒ 观测 z 恒为 0、p 恒为 1。
        零假设是"没有效应"，所以**观测值必须保留**。"""
        import random
        rng = random.Random(5)
        g = [[rng.gauss(0.004, 0.002) for _ in range(200)] for _ in range(3)]
        sf = pooled_signflip_pvalue(*g, reps=200, seed=13)
        # ⚠️ 这条的**要点只是"不为 0"**（我第一版断言 |z|<10，那是随手猜的：
        # 注入的效应远大于噪声时 z 本来就可以是 50）。
        self.assertNotAlmostEqual(sf["z_obs"], 0.0, places=6)
        self.assertEqual(sf["z_obs"] == sf["z_obs"], True)   # 不是 nan

    def test_报出偏度_不许藏(self):
        """⚠️ 对称性是它的前提 ⇒ 偏度必须报出来（SOL 的实测偏度 −12.9）。"""
        import random
        rng = random.Random(7)
        g = [[rng.gauss(0, 0.002) for _ in range(200)]]
        g.append([rng.gauss(0, 0.002) for _ in range(200)])
        sf = pooled_signflip_pvalue(*g, reps=100, seed=14)
        self.assertEqual(len(sf["skew"]), 2)
        for x in sf["skew"]:
            self.assertTrue(abs(x) < 1.0, f"正态样本的偏度不该这么大：{x}")

    def test_固定种子可复现(self):
        g = [self._normal(100, 0.003, 21), self._normal(100, 0.003, 22)]
        a = pooled_signflip_pvalue(*g, reps=200, seed=77)
        b = pooled_signflip_pvalue(*g, reps=200, seed=77)
        self.assertEqual(a, b)


class TestTCriterionIsAlsoCalibrated(unittest.TestCase):
    """⭐⭐ **我们最终用的判据（`t(df=k−1)`）自己也必须被验。**

    A12 教训：正态 z 是**反保守**的（δ=0 假阳性率实测 9.9%）。
    所以本轮改用 `t(df=k−1)`——但**换了判据就得重新验它**，
    否则只是把"没验过的判据"从正态换成了 t。
    """

    def _normal(self, n, sd, seed):
        import random
        rng = random.Random(seed)
        return [rng.gauss(0, sd) for _ in range(n)]

    def test_报出t判据的假阳性率(self):
        g = [self._normal(300, 0.003, s) for s in (1, 2, 3, 4)]
        cal = pooled_mc_calibration(*g, delta=0.0, reps=800, seed=5)
        self.assertIn("false_positive_rate_t", cal)
        self.assertIn("t_crit", cal)
        self.assertEqual(cal["k"], 4)
        self.assertAlmostEqual(cal["t_crit"], 3.182, places=3)

    def test_t判据在这份数据上不比正态更松(self):
        """⚠️ 核心性质：t 判据**不能**比正态更宽松（否则换它没意义）。"""
        g = [self._normal(300, 0.003, s) for s in (11, 12, 13, 14)]
        cal = pooled_mc_calibration(*g, delta=0.0, reps=2000, seed=6)
        self.assertLessEqual(cal["false_positive_rate_t"] + 1e-9,
                             cal["false_positive_rate"] + 0.02,
                             "t 判据的假阳性率不该高于正态判据")

    def test_正态样本下t判据假阳性率落在保守侧(self):
        g = [self._normal(300, 0.003, s) for s in (21, 22, 23, 24)]
        cal = pooled_mc_calibration(*g, delta=0.0, reps=3000, seed=7)
        # 名义 5%，k=4 时 t 更严 ⇒ 实测应明显低于 5%
        self.assertLess(cal["false_positive_rate_t"], 0.05)

    def test_注入效应时t判据能检出(self):
        """⚠️ 分辨力补强：不能"永远不显著"。

        ⚠️ 我第一版把效应做进**残差**里（`gauss(0.004, ...)`）再传 `delta=0.0`
        ——**错**：这个函数会**先把每组中心化**，效应会被消掉，
        于是实测 0.0（看起来像"判不出来"）。
        ⇒ 正确用法：传**零均值**残差 + 把效应放进 `delta`。
        """
        import random
        rng = random.Random(9)
        g = [[rng.gauss(0.0, 0.003) for _ in range(300)] for _ in range(4)]
        cal = pooled_mc_calibration(*g, delta=0.004, reps=800, seed=10)
        self.assertGreater(cal["false_positive_rate_t"], 0.9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
