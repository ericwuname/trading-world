"""凹度判据的回归测试（工作线I）。

为什么这份测试必须先红后绿
--------------------------
凹度判据是**新的验收标准候选**——它一旦算错，会以"看起来合理"的方式
给出错误的判决（这正是本项目最怕的形态：不报错、只给错答案）。
任务书 §3.5 点名要钉两处：log 底数/顺序、判决边界条件。
本文件按"构造已知答案的数据"来测，而不是测"跑起来没崩"。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tw.analyzer_concavity import (  # noqa: E402
    VERDICT_CONCAVE,
    VERDICT_LINEAR_OR_WORSE,
    VERDICT_UNDECIDED,
    ElasticityResult,
    concavity_verdict,
    pairwise_local_elasticity,
)


def _res(e, sem, n=3):
    """直接构造一个结果对象（用于测判决函数本身）。"""
    tc = 4.303 if n == 3 else 1.96
    return ElasticityResult(ok=True, mean=e, sem=sem, n_seeds=n,
                            ci95=(e - tc * sem, e + tc * sem))


class TestElasticityArithmetic(unittest.TestCase):
    """构造**已知答案**的数据：线性必须给出 1.0，平方根律必须给出 0.5。"""

    def test_完美线性给出1(self) -> None:
        # 规模 0.001→0.002（×2），滑点也 ×2 ⇒ e = log2/log2 = 1
        r = pairwise_local_elasticity(0.001, 0.002, mean_small=-5.0,
                                      sem_small=0.1, mean_big=-10.0,
                                      sem_big=0.1, n_seeds=3)
        self.assertTrue(r.ok, r.reason)
        self.assertAlmostEqual(r.mean, 1.0, places=9)

    def test_完美平方根律给出0点5(self) -> None:
        # 规模 ×4，滑点 ×2 ⇒ e = log2/log4 = 0.5
        r = pairwise_local_elasticity(0.001, 0.004, mean_small=-5.0,
                                      sem_small=0.1, mean_big=-10.0,
                                      sem_big=0.1, n_seeds=3)
        self.assertAlmostEqual(r.mean, 0.5, places=9)

    def test_顺序反了会被抓到(self) -> None:
        """把大小档位对调，弹性应当**不变**（比值取倒数，log 变号后相除仍为正）。

        ⚠️ 这条同时是 M47 的照妖镜：如果实现把 ``size_big/size_small`` 或
        ``|b|/|a|`` 的某一边写反、或者漏了取绝对值，这里就会红。
        """
        a = pairwise_local_elasticity(0.001, 0.002, mean_small=-5.0,
                                      sem_small=0.1, mean_big=-10.0,
                                      sem_big=0.1, n_seeds=3)
        b = pairwise_local_elasticity(0.001, 0.002, mean_small=5.0,
                                      sem_small=0.1, mean_big=10.0,
                                      sem_big=0.1, n_seeds=3)
        self.assertAlmostEqual(a.mean, b.mean, places=9,
                               msg="符号翻转后弹性变了——说明没取绝对值")

    def test_滑点不增长时弹性为零(self) -> None:
        r = pairwise_local_elasticity(0.001, 0.002, mean_small=-5.0,
                                      sem_small=0.1, mean_big=-5.0,
                                      sem_big=0.1, n_seeds=3)
        self.assertAlmostEqual(r.mean, 0.0, places=9)

    def test_逐种子路径与解析式一致(self) -> None:
        # 每个种子的比值都恰好是 2 ⇒ 弹性恒为 1
        r = pairwise_local_elasticity(
            0.001, 0.002, mean_small=-5.0, sem_small=0.0, mean_big=-10.0,
            sem_big=0.0, n_seeds=3,
            per_seed_small=[-4.0, -5.0, -6.0], per_seed_big=[-8.0, -10.0, -12.0])
        self.assertEqual(r.method, "per_seed")
        self.assertAlmostEqual(r.mean, 1.0, places=9)
        self.assertAlmostEqual(r.sem, 0.0, places=9)

    def test_无逐种子数据时标注delta方法(self) -> None:
        r = pairwise_local_elasticity(0.001, 0.002, mean_small=-5.0,
                                      sem_small=0.1, mean_big=-10.0,
                                      sem_big=0.1, n_seeds=3)
        self.assertEqual(r.method, "delta_method",
                         "没有逐种子数据却标成了别的来源——不许把近似冒充原始数据")


class TestGuardsAndFlags(unittest.TestCase):
    """护栏与标记：不许在无意义的地方硬给一个数。"""

    def test_档位顺序非法时拒绝(self) -> None:
        r = pairwise_local_elasticity(0.002, 0.001, mean_small=-5.0,
                                      sem_small=0.1, mean_big=-10.0,
                                      sem_big=0.1, n_seeds=3)
        self.assertFalse(r.ok)
        self.assertIn("非法", r.reason)

    def test_种子数不足时拒绝(self) -> None:
        r = pairwise_local_elasticity(0.001, 0.002, mean_small=-5.0,
                                      sem_small=0.1, mean_big=-10.0,
                                      sem_big=0.1, n_seeds=1)
        self.assertFalse(r.ok)
        self.assertIn("种子数不足", r.reason)

    def test_滑点为0时拒绝(self) -> None:
        r = pairwise_local_elasticity(0.001, 0.002, mean_small=0.0,
                                      sem_small=0.1, mean_big=-10.0,
                                      sem_big=0.1, n_seeds=3)
        self.assertFalse(r.ok)

    def test_低信噪比被标出(self) -> None:
        # 小档位的滑点均值(0.5)连自身 sem(3.0) 都不超过
        r = pairwise_local_elasticity(0.001, 0.002, mean_small=-0.5,
                                      sem_small=3.0, mean_big=-10.0,
                                      sem_big=0.1, n_seeds=3)
        self.assertTrue(r.low_snr, "这一档明显是噪声却没被标出来")

    def test_高信噪比不误标(self) -> None:
        r = pairwise_local_elasticity(0.001, 0.002, mean_small=-50.0,
                                      sem_small=1.0, mean_big=-100.0,
                                      sem_big=1.0, n_seeds=3)
        self.assertFalse(r.low_snr)


class TestVerdictBoundaries(unittest.TestCase):
    """判决的三分类与**边界**（M48 钉的就是这里）。"""

    def test_上界小于1是显著凹(self) -> None:
        v, _ = concavity_verdict(_res(0.5, 0.05))     # CI=[0.285, 0.715]
        self.assertEqual(v, VERDICT_CONCAVE)

    def test_下界等于1是线性或更差(self) -> None:
        """⚠️ 边界：``ci_lower == 1.0`` 必须归入「线性或更差」而不是无人认领。"""
        r = ElasticityResult(ok=True, mean=1.5, sem=0.0, n_seeds=3,
                             ci95=(1.0, 2.0))
        v, _ = concavity_verdict(r)
        self.assertEqual(v, VERDICT_LINEAR_OR_WORSE,
                         "下界恰好 1.0 时判决落进了缝隙")

    def test_上界恰好1不是凹(self) -> None:
        """⚠️ 另一侧的边界：``ci_upper == 1.0`` **不算**显著凹（凹要严格 < 1）。"""
        r = ElasticityResult(ok=True, mean=0.5, sem=0.0, n_seeds=3,
                             ci95=(0.0, 1.0))
        v, _ = concavity_verdict(r)
        self.assertEqual(v, VERDICT_UNDECIDED,
                         "上界恰好 1.0 被判成了显著凹——边界写反了")

    def test_跨过1是不可判定(self) -> None:
        v, _ = concavity_verdict(_res(0.9, 0.2))      # CI=[0.039, 1.761]
        self.assertEqual(v, VERDICT_UNDECIDED)

    def test_三分类互补无缝隙(self) -> None:
        """扫一遍区间端点，任何取值都必须落进三类之一（不留缝隙）。"""
        for lo in (0.0, 0.5, 0.999, 1.0, 1.001, 2.0):
            for hi in (lo, lo + 0.5, 2.0):
                r = ElasticityResult(ok=True, mean=(lo + hi) / 2, sem=0.0,
                                     n_seeds=3, ci95=(lo, hi))
                v, _ = concavity_verdict(r)
                self.assertIn(v, (VERDICT_CONCAVE, VERDICT_LINEAR_OR_WORSE,
                                  VERDICT_UNDECIDED))

    def test_未成功计算时判决为不可判定(self) -> None:
        v, ci = concavity_verdict(ElasticityResult(ok=False, reason="x"))
        self.assertEqual(v, VERDICT_UNDECIDED)
        self.assertTrue(ci[0] != ci[0])   # nan


if __name__ == "__main__":
    unittest.main(verbosity=2)
