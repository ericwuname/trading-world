"""工作线M 的决策逻辑测试（EM.2 阈值 / EM.4 分支）。

任务书 §4.5 要求：「EM.2 的判断阈值需要在报告里写清楚具体标准，
不能是一个模糊的主观判断」。所以这里把阈值**钉成可测的**——
它一旦被改成"看着办"，测试就会红。

（EM.1 的功效公式本身在 ``tests/test_power_analysis_H4.py`` 里测，
那里用的是教科书锚点；本文件测的是**决策**这一层。）
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / "scripts"))  # 追加：插到前面会遮蔽同名包（scripts/gui.py vs gui/）

from scripts.run_workstream_M import (  # noqa: E402
    COST_THRESHOLD,
    em2_cost_judgement,
    em4_honest_report,
    read_observed_from_E,
)

_SCRIPTS = str(ROOT / "scripts")
while _SCRIPTS in sys.path:
    sys.path.remove(_SCRIPTS)


def _fake_em1(diff: float, pooled_sd: float, n_taker: int = 31,
              n_both: int = 42, n_seeds: int = 3) -> dict:
    """构造一份 em1 —— ``required_n_group1`` **调真实的功效计算**，不写死。

    ⚠️ 第一版把 ``required_n_group1`` 硬编码成 100，结果阈值测试的"不值得"
    分支永远走不到（100 个事件点 ÷ 6 个/种子 = 50 个种子，正好卡在阈值上）。
    测试数据**必须由被测公式生成**，否则测的是我自己编的数字。
    """
    from scripts.power_analysis_H4 import estimate_required_n_for_quadrant_test
    r = estimate_required_n_for_quadrant_test(diff, pooled_sd,
                                              ratio=n_both / n_taker)
    return {
        **r.as_dict(),
        "observed": {"diff": diff, "pooled_sd": pooled_sd,
                     "n_taker_only": n_taker, "n_both_burst": n_both,
                     "n_seeds_used": n_seeds,
                     "mean_taker_only": 58.0, "mean_both_burst": 60.9,
                     "mean_neither": 61.8},
    }


class TestCostThresholds(unittest.TestCase):
    """阈值必须**事先写死**且可测（不许事后按结果调）。"""

    def test_阈值结构固定(self) -> None:
        self.assertEqual(COST_THRESHOLD["worth_it_max_seeds"], 50)
        self.assertEqual(COST_THRESHOLD["marginal_max_seeds"], 100)
        self.assertEqual(COST_THRESHOLD["not_worth_min_seeds"], 100)

    def test_小需求判值得做(self) -> None:
        """每种子产出很多事件 ⇒ 需要的种子少。"""
        em1 = _fake_em1(3.0, 8.0, n_taker=200)      # 每种子 66 个事件
        em2 = em2_cost_judgement(em1, observed_n_seeds=3)
        self.assertEqual(em2["verdict"], "值得做")
        self.assertLessEqual(em2["seeds_needed"],
                             COST_THRESHOLD["worth_it_max_seeds"])

    def test_巨大需求判不值得(self) -> None:
        """事件产出极少 + 需求高 ⇒ 需要成百上千个种子。"""
        em1 = _fake_em1(0.5, 20.0, n_taker=6)       # 每种子 2 个事件
        em2 = em2_cost_judgement(em1, observed_n_seeds=3)
        self.assertEqual(em2["verdict"], "不值得")
        self.assertGreater(em2["seeds_needed"],
                           COST_THRESHOLD["not_worth_min_seeds"])

    def test_判定与阈值保持一致(self) -> None:
        """扫一遍需求区间，验证判定的边界与阈值定义一致（不留缝隙）。"""
        for n_taker in (3, 10, 31, 100, 400, 2000):
            em1 = _fake_em1(2.7, 16.0, n_taker=n_taker)
            em2 = em2_cost_judgement(em1, observed_n_seeds=3)
            need = em2["seeds_needed"]
            if need <= COST_THRESHOLD["worth_it_max_seeds"]:
                self.assertEqual(em2["verdict"], "值得做", f"n_taker={n_taker}")
            elif need <= COST_THRESHOLD["marginal_max_seeds"]:
                self.assertEqual(em2["verdict"], "边缘", f"n_taker={n_taker}")
            else:
                self.assertEqual(em2["verdict"], "不值得", f"n_taker={n_taker}")

    def test_两条路的成本都被说明(self) -> None:
        em2 = em2_cost_judgement(_fake_em1(2.7, 16.0), observed_n_seeds=3)
        self.assertIn("seeds_needed", em2)
        self.assertIn("length_multiplier_option", em2)
        self.assertIn("单次模拟时长", em2["note"])
        self.assertIn("稳态", em2["note"])


class TestEM4HonestBranch(unittest.TestCase):
    """EM.4 必须给出可读的结论文本与建议，而不是一个空分支。"""

    def test_诚实报告含关键要素(self) -> None:
        em1 = _fake_em1(2.7, 16.0, n_taker=6)
        em2 = em2_cost_judgement(em1, observed_n_seeds=3)
        rep = em4_honest_report(em1, em2)
        self.assertEqual(rep["branch"], "EM.4 诚实报告")
        txt = rep["text"]
        for needle in ("Cohen's d", "种子", "同一类问题"):
            self.assertIn(needle, txt, f"结论文本少了 {needle}")
        # 宏观效应量必须被单独结构化，而不是混在散文里
        self.assertIn("relative_change_pct", rep["macro_effect"])
        self.assertIsNotNone(rep["macro_effect"]["background_mean"])

    def test_建议里不鼓励堆种子(self) -> None:
        em1 = _fake_em1(2.7, 16.0, n_taker=6)
        em2 = em2_cost_judgement(em1, observed_n_seeds=3)
        rep = em4_honest_report(em1, em2)
        self.assertIn("不建议", rep["recommendation"])

    def test_缺背景均值时不崩(self) -> None:
        em1 = _fake_em1(2.7, 16.0)
        em1["observed"]["mean_neither"] = None
        em2 = em2_cost_judgement(em1, observed_n_seeds=3)
        rep = em4_honest_report(em1, em2)      # 不应抛
        self.assertIsNone(rep["macro_effect"]["relative_change_pct"])


class TestObservedReading(unittest.TestCase):
    """观测值必须从 E 线产物读；数据不在时**明确报错**而不是给 0。

    ⚠️ 依赖 ``out/workstream_E_metrics.json``。mutation 沙箱现在会复制
    ``out/`` 顶层的 ``*.json``（见 ``prepare_sandbox``），
    但换个环境仍可能缺——所以这里做**显式跳过**兜底：
    缺文件直接断言存在会让沙箱基线变红，而**基线一红所有变异体都会"变红"**。
    """

    @staticmethod
    def _need_data() -> None:
        from scripts.run_workstream_M import read_observed_from_E
        try:
            read_observed_from_E()
        except RuntimeError as e:
            raise unittest.SkipTest(f"缺 E 线观测数据：{e}")

    def test_现读观测值(self) -> None:
        self._need_data()
        obs = read_observed_from_E()
        self.assertGreater(obs["n_taker_only"], 0)
        self.assertGreater(obs["n_both_burst"], 0)
        self.assertAlmostEqual(obs["ratio"], obs["n_both_burst"] / obs["n_taker_only"],
                               places=9)
        self.assertIn("workstream_E_metrics.json", obs["source"])

    def test_合并标准差的计算口径(self) -> None:
        """pooled_sd 必须是两样本 t 检验口径（不是简单平均）。"""
        self._need_data()
        obs = read_observed_from_E()
        n1, n2 = obs["n_taker_only"], obs["n_both_burst"]
        s1, s2 = obs["sd_taker_only"], obs["sd_both_burst"]
        want = (((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2)
                / (n1 + n2 - 2)) ** 0.5
        self.assertAlmostEqual(obs["pooled_sd"], want, places=9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
