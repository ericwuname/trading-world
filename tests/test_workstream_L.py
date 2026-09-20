"""工作线L（回填标准配置）的回归测试。

任务书 §3.6 点名要钉两处，两条都在这里：
  · **M52**：``fit`` 用的种子数与饱和检查用的不一致 → 断言必须抓住
  · **M53**：λ̄ 标定复用了旧窗口而不是实验窗口 → 必须被检测出来

另外测 EL.2 的三选一判定逻辑——它是本轮最重要的产出，
判错方向会直接把结论写反（而这正是被推翻过的那种错误）。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.run_workstream_L import (  # noqa: E402
    backfill_seeds,
    calibration_window,
    check_seed_consistency,
    el2_asymmetry_verdict,
    el3_eb1_direction,
    el4_joint_direction,
    families,
)

_SCRIPTS = str(ROOT / "scripts")
while _SCRIPTS in sys.path:
    sys.path.remove(_SCRIPTS)


class TestCalibrationWindow(unittest.TestCase):
    """⭐ M53：标定窗口**必须**等于实验窗口（第六条纪律）。"""

    def test_标定窗口等于实验窗口(self) -> None:
        from run_stage3 import HORIZON, WARMUP
        win = calibration_window()
        self.assertEqual(win, (WARMUP, WARMUP + HORIZON))
        self.assertEqual(win, (6000, 6400),
                         f"标定窗口变成了 {win}——若退化成 (500, 3000) 就是"
                         "重蹈 E8.2 的覆辙（λ̄ 偏高约 7%，且不报错）")

    def test_不是历史口径(self) -> None:
        self.assertNotEqual(calibration_window(), (500, 3000),
                            "标定窗口退回历史口径了")


class TestSeedSet(unittest.TestCase):
    """种子必须与阶段3 **逐位对齐**（不是 ``range(8)``）。"""

    def test_八个种子且与阶段3一致(self) -> None:
        from run_stage3 import SEEDS
        s = backfill_seeds()
        self.assertEqual(len(s), 8, "标准配置要求 8 个种子")
        self.assertEqual(s, list(SEEDS))

    def test_不是零基序列(self) -> None:
        """``range(8)`` 会得到 0..7——那是**另一批随机数**，新旧对比会失去意义。"""
        s = backfill_seeds()
        self.assertNotEqual(s, list(range(8)),
                            "种子退化成 range(8) 了——与阶段3 不可比")
        self.assertGreater(s[0], 1000, "种子应当是日期型的大整数")


class TestSeedConsistencyGuard(unittest.TestCase):
    """⭐ M52：种子数不一致时必须断言。"""

    @staticmethod
    def _rows(n_seeds_each: list[int]) -> list[dict]:
        return [{"shock_frac": 0.001 * (i + 1), "n_seeds": n}
                for i, n in enumerate(n_seeds_each)]

    def test_一致时通过(self) -> None:
        check_seed_consistency(self._rows([8, 8, 8]), 8, "t")   # 不应抛

    def test_不一致时抛断言(self) -> None:
        with self.assertRaises(AssertionError) as ctx:
            check_seed_consistency(self._rows([8, 8, 3]), 8, "某家族")
        self.assertIn("必须用同一批种子", str(ctx.exception))

    def test_全部不一致也要报出(self) -> None:
        with self.assertRaises(AssertionError):
            check_seed_consistency(self._rows([2, 2]), 8, "t")

    def test_空rows不误报(self) -> None:
        check_seed_consistency([], 8, "t")      # 不应抛


class TestFamilyRegistry(unittest.TestCase):
    """8 个家族（10 个里去重两个），且别名关系被记录。"""

    def test_八个家族(self) -> None:
        f = families()
        self.assertEqual(len(f), 8, f"应当是 8 个去重后的家族，实际 {len(f)}")

    def test_别名被显式记录(self) -> None:
        """J1/J2 与 EB1 的两个配置是**同一个配置**，必须标出来。"""
        f = families()
        self.assertEqual(f["EB1_chartist_only"].get("alias_of"),
                         "J1_stage6_baseline")
        self.assertEqual(f["EB1_all_three"].get("alias_of"), "J2_plus_B")

    def test_每个家族都有旧读数与分组(self) -> None:
        for name, spec in families().items():
            self.assertIn("original_k", spec, f"{name} 缺 original_k")
            self.assertIn("group", spec, f"{name} 缺 group")
            self.assertIn(spec["kind"], ("hawkes", "stage6", "joint"))


class TestEL2Verdict(unittest.TestCase):
    """EL.2 三选一：判错方向会直接把结论写反。"""

    @staticmethod
    def _res(t_ci, b_ci):
        return {"EA4_taker": {"k": sum(t_ci) / 2, "ci": list(t_ci)},
                "EA4_both": {"k": sum(b_ci) / 2, "ci": list(b_ci)}}

    def test_taker整体低于both时判非对称更凹(self) -> None:
        v = el2_asymmetry_verdict(self._res((0.3, 0.6), (0.9, 1.2)))
        self.assertEqual(v["verdict"], "非对称显著更凹")
        self.assertEqual(v["overlap"], 0.0)

    def test_taker整体高于both时判对称更凹(self) -> None:
        v = el2_asymmetry_verdict(self._res((1.0, 1.4), (0.4, 0.8)))
        self.assertEqual(v["verdict"], "对称显著更凹")

    def test_区间重叠时如实报不可判定(self) -> None:
        """⚠️ 不许因为"投入了更多资源"就暗示应该有答案。"""
        v = el2_asymmetry_verdict(self._res((0.4, 1.1), (0.8, 1.5)))
        self.assertEqual(v["verdict"], "依然无法判定")
        self.assertGreater(v["overlap"], 0)

    def test_数据不足时明确说明(self) -> None:
        v = el2_asymmetry_verdict({"EA4_taker": {}, "EA4_both": {}})
        self.assertIn(v["verdict"], ("数据不足", "依然无法判定"))

    def test_重叠占比被算出(self) -> None:
        v = el2_asymmetry_verdict(self._res((0.0, 1.0), (0.5, 1.5)))
        self.assertAlmostEqual(v["overlap"], 0.5, places=9)
        self.assertAlmostEqual(v["overlap_frac_of_narrower"], 0.5, places=9)


class TestEL3EL4Directions(unittest.TestCase):
    """EL.3（覆盖面）与 EL.4（J2 vs J3）的方向判读。"""

    def test_EL3识别改善方向(self) -> None:
        res = {
            "EB1_chartist_only": {"k": 1.25, "ci": [1.0, 1.5]},
            "EB1_all_three": {"k": 0.80, "ci": [0.6, 1.0]},
        }
        out = el3_eb1_direction(res)
        d = out["arms"]["EB1_all_three"]
        self.assertTrue(d["improved"])
        self.assertLess(d["delta_vs_baseline"], 0)

    def test_EL3区间重叠时不宣称分离(self) -> None:
        res = {
            "EB1_chartist_only": {"k": 1.0, "ci": [0.5, 1.5]},
            "EB1_all_three": {"k": 0.9, "ci": [0.6, 1.2]},
        }
        out = el3_eb1_direction(res)
        self.assertFalse(out["arms"]["EB1_all_three"]["ci_separated"])

    def test_EL4识别J2优于J3(self) -> None:
        res = {"EB1_all_three": {"k": 0.70, "ci": [0.5, 0.9]},
               "J3_plus_A": {"k": 0.95, "ci": [0.8, 1.1]}}
        out = el4_joint_direction(res)
        self.assertTrue(out["j2_still_better"])
        self.assertGreater(out["delta_j3_minus_j2"], 0)

    def test_EL4缺数据时不崩(self) -> None:
        out = el4_joint_direction({})
        self.assertIsNone(out["j2_k"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
