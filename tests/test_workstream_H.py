"""工作线H（饱和检查）与工作线J（回溯审计）的回归测试。

两个工作线各钉一处任务书点名的地方：
  · H：饱和判定的**方向**（变异体 M46）——写反了不报错，只是未饱和集变空
  · J：重跑时**种子数不许超过原始值**（变异体 M49）——
       超过之后审计的就不是"原始结论的分辨力"，而是"一次新实验的分辨力"
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / "scripts"))  # 追加：插到前面会遮蔽同名包（scripts/gui.py vs gui/）

from scripts.run_workstream_H import (  # noqa: E402
    CONFIGS,
    SAT_THRESHOLD,
    is_saturated,
)

_SCRIPTS = str(ROOT / "scripts")
while _SCRIPTS in sys.path:
    sys.path.remove(_SCRIPTS)


class TestSaturationDirection(unittest.TestCase):
    """⚠️ 方向：``fill_ratio < 0.95`` 才是饱和（M46）。"""

    def test_吃不满即饱和(self) -> None:
        self.assertTrue(is_saturated(0.90))

    def test_吃满不算饱和(self) -> None:
        self.assertFalse(is_saturated(1.00))

    def test_边界值恰好等于阈值不算饱和(self) -> None:
        """``== 0.95`` 不算饱和（判据是严格小于）——与 `unsaturated()` 口径一致。"""
        self.assertFalse(is_saturated(SAT_THRESHOLD))

    def test_阈值方向写反会立刻暴露(self) -> None:
        """反面参照：如果判据被写成 ``>``，这一条必须红。

        构造一组**真实的** fill_ratio（阶段3 的 7 档实测值）：
        如果方向反了，全部 7 档都会被判成"饱和"，未饱和集为空——
        这正是 M46 描述的现象。
        """
        real_fills = [1.0000, 1.0000, 0.9975, 0.9014, 0.6586, 0.3551, 0.1626]
        sat = [is_saturated(f) for f in real_fills]
        self.assertEqual(sat.count(True), 4,
                         f"饱和档数应当是 4（0.01 及以后），实际 {sat.count(True)}"
                         f"——判据方向可能写反了")
        self.assertEqual(sat.count(False), 3, "未饱和档数应当是 3")

    def test_全档吃满时未饱和集非空(self) -> None:
        """反面参照的另一半：全 1.0 时不该有任何饱和。"""
        self.assertEqual([is_saturated(1.0) for _ in range(7)].count(True), 0)


class TestHConfigsDeclared(unittest.TestCase):
    """三个配置必须都在，且名字在报告与产物里对得上。"""

    def test_三配置齐全(self) -> None:
        self.assertEqual(set(CONFIGS),
                         {"baseline_no_hawkes", "ea4_taker_only",
                          "ea2_both_symmetric"})


class TestStage3AuditSeedGuard(unittest.TestCase):
    """工作线J：审计用的种子数**不得超过**原始值（M49）。

    ⚠️ 这里**不依赖 out/ 产物**：mutation 沙箱不复制 `out/`，
    任何读产物的测试在沙箱里都会把基线弄红——而基线一红，
    所有变异体都会"变红"，整个变异验证就变成了假证据。
    所以数据依赖检查会**显式跳过**，断言逻辑则用**注入的假数据**来测。
    """

    def test_定位到逐档数据且种子数不超原始值(self) -> None:
        from scripts.run_workstream_J import (
            ORIGINAL_N_SEEDS,
            locate_stage3_original_data,
        )

        info = locate_stage3_original_data()
        if not info["found"]:
            self.skipTest("沙箱/环境里没有 out/stage3_metrics.json"
                          "（mutation 沙箱不复制 out/）——跳过数据依赖检查")
        self.assertTrue(info["has_per_level_detail"],
                        "阶段3 的产物没有逐档细节（sem/t）——审计需要它")
        self.assertGreaterEqual(max(info["n_seeds_in_data"]), 2,
                                "种子数太少，无法给不确定度")
        self.assertLessEqual(
            max(info["n_seeds_in_data"]), ORIGINAL_N_SEEDS,
            f"数据里的种子数超过原始值 {ORIGINAL_N_SEEDS}，"
            "那审计的就不是原始结论的分辨力了")
        self.assertFalse(info["rerun_used_more_seeds"])

    @staticmethod
    def _fake_stage3(n_seeds: int, levels: int = 3) -> dict:
        return {"C2_staged": {"rows": [
            {"fill_ratio": 1.0, "slippage_bp": -5.0 * (i + 1),
             "slippage_sem_bp": 1.0, "shock_frac": 0.001 * (i + 1),
             "n_seeds": n_seeds} for i in range(levels)]}}

    def test_种子上限的断言真的会拦住(self) -> None:
        """反面参照：上限压到 2，喂 8 种子的数据必须触发断言。"""
        import scripts.run_workstream_J as J

        orig = J.ORIGINAL_N_SEEDS
        try:
            J.ORIGINAL_N_SEEDS = 2
            with self.assertRaises(AssertionError):
                J.locate_stage3_original_data(self._fake_stage3(8))
        finally:
            J.ORIGINAL_N_SEEDS = orig

    def test_合规种子数不触发断言(self) -> None:
        """正面参照：8 种子对上 8 的上限必须通过（否则守卫太紧会误伤）。"""
        import scripts.run_workstream_J as J

        info = J.locate_stage3_original_data(self._fake_stage3(8))
        self.assertTrue(info["found"])
        self.assertFalse(info["rerun_used_more_seeds"])
        self.assertEqual(info["n_seeds_in_data"], [8])

    def test_缺数据时不崩且明确报found为假(self) -> None:
        import scripts.run_workstream_J as J

        info = J.locate_stage3_original_data({})
        self.assertFalse(info["found"])
        self.assertIn("error", info)


if __name__ == "__main__":
    unittest.main(verbosity=2)
