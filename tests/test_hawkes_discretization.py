"""离散化公式的正确性——这是 A 线 H3 的**机制本体**。

背景（为什么这个文件必须存在）
------------------------------
任务书担心"H3 离散化偏差"才是阶段8 k 改善的真因。
`HawkesConfig` 早就用了正确的**离散**分支比
（``alpha = n·(e^{β·dt}−1)``，不是 ``n·β``），但**没有任何测试**
直接钉住这件事——它只是"文档里写过"。

本项目对"文档里写过"的立场很明确：**没有测试的约定等于注释。**
所以这里把它钉死，并配变异体 M34（把公式换回连续版）。

实测的量级（EA.0，独立自激路径 20000 步）：
    用错公式时，n=0.3 → −3.0%、n=0.6 → −10.0%、n=0.9 → **−40.1%**（ρ̄=0.600）。
这个量级足以伪造出"打开机制后交易变少/变多"的假结论。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tw.order_flow.hawkes import (  # noqa: E402
    HawkesConfig,
    geometric_factor,
    simulate_event_rate,
)

BETA = 0.15


class TestDiscreteFormula(unittest.TestCase):
    def test_alpha_用的是离散分支比而不是n乘beta(self) -> None:
        """⭐ 这条直接对应变异体 M34。

        ``alpha`` 必须是 ``n / f``，其中 ``f = 1/(e^{β·dt} − 1)``，
        等价地 ``alpha = n·(e^{β·dt} − 1)``。
        写成 ``n·β`` 时两者在 β·dt 很小时**看起来差不多**（0.15 vs 0.1618），
        所以这个错误在上手阶段看不出来，要跑到几千 tick 才显现为 ρ̄ 偏离 1。
        """
        cfg = HawkesConfig(base_lambda=20.0, branching=0.6, beta=BETA, dt=1.0)
        f = geometric_factor(BETA, 1.0)
        self.assertAlmostEqual(cfg.alpha, 0.6 / f, places=12)
        self.assertAlmostEqual(cfg.alpha, 0.6 * (np.exp(BETA) - 1.0), places=12)
        # 反面参照：**必须**与错误的 n·β 不同，否则这条测试没有分辨力
        self.assertNotAlmostEqual(cfg.alpha, 0.6 * BETA, places=6)

    def test_mu_按离散平稳式反解(self) -> None:
        cfg = HawkesConfig(base_lambda=20.0, branching=0.6, beta=BETA)
        self.assertAlmostEqual(cfg.mu, 20.0 * (1.0 - 0.6), places=12)

    def test_几何因子在beta趋零时退化为连续式(self) -> None:
        """β·dt → 0 时 ``1/(e^{βdt}−1) ≈ 1/(βdt)``，于是 ``α ≈ n·β``。"""
        tiny = 1e-6
        self.assertAlmostEqual(geometric_factor(tiny, 1.0), 1.0 / tiny, delta=1.0)


class TestRealizedRateHitsTarget(unittest.TestCase):
    """⭐ A.3.2 要求的验证：实测事件率必须命中 λ̄（相对误差 < 2%）。

    这条把"E[λ]=λ̄"从**代数**变成**实测**。为什么必须实测：
    那个恒等式是"按构造成立"的（``mu``/``alpha`` 就是这么反解的），
    拿它验证自己是循环论证。真正可能出错的是递推实现与取事件的顺序。
    """

    def test_实测事件率命中目标_不含clamp(self) -> None:
        for n in (0.0, 0.6, 0.9):
            with self.subTest(branching=n):
                cfg = HawkesConfig(base_lambda=20.0, branching=n, beta=BETA)
                r = simulate_event_rate(cfg.to_process(), 8000,
                                        np.random.default_rng(42),
                                        base_lambda=cfg.base_lambda,
                                        clamp=None, warmup=1000)
                self.assertLess(abs(r["rel_err"]), 0.02,
                                f"n={n} 实测率 {r['realized_rate']:.4f} "
                                f"偏差 {r['rel_err']:+.2%}")

    def test_护栏在本区间内不咬_且这件事必须被测出来(self) -> None:
        """``clamp_rho=4.0`` 在这套参数下**几乎从不生效**——这是要记录的事实。

        ⚠️ 这条测试的存在本身就是一次自我纠错：
        我最初拿两次**不同随机流**的运行去比较"有 clamp / 无 clamp"，
        把蒙特卡洛差异（约 6 个百分点）当成了护栏效应，还写进了报告。
        用**同一个种子**重测，差异是 **0.000%**——因为 ρ 的量级在 0.6~1.4，
        离阈值 4 差得远，护栏根本没被触发过。

        所以：**"护栏压低了活跃度"这个说法在本项目是错的。**
        市场里 ρ̄ ≈ 0.92 < 1 另有一个真原因——λ̄ 的标定窗口含瞬态、
        比市场真实基线高约 7%（见 ``calibrate_base_lambda`` 的 ``sim_kw`` 说明
        与闭环不动点公式 ρ̄ = λ̄(1−n)/(λ̄ − n·c·N)）。
        """
        cfg = HawkesConfig(base_lambda=20.0, branching=0.9, beta=BETA)
        a = simulate_event_rate(cfg.to_process(), 20000,
                               np.random.default_rng(42),
                               base_lambda=cfg.base_lambda, clamp=None,
                               warmup=2000)
        b = simulate_event_rate(cfg.to_process(), 20000,
                               np.random.default_rng(42),
                               base_lambda=cfg.base_lambda,
                               clamp=cfg.clamp_rho, warmup=2000)
        self.assertAlmostEqual(a["realized_rate"], b["realized_rate"], places=9,
                              msg="同种子下有/无 clamp 竟然不同——"
                                  "要么护栏被触发了（那 ρ 分布变了），"
                                  "要么两次用的不是同一个随机流")
        self.assertLess(max(a["rho_mean"], b["rho_mean"]), cfg.clamp_rho,
                        "ρ̄ 已经逼近 clamp 阈值——护栏开始咬，"
                        "此时上面的等式不再成立，本测试需要重新设计")

    def test_事件率对beta不敏感_只要分支比不变(self) -> None:
        """分支比 n 才是决定长期活跃度的量；β 只决定聚集的时间尺度。"""
        rates = []
        for beta in (0.15, 0.5):
            cfg = HawkesConfig(base_lambda=20.0, branching=0.5, beta=beta)
            r = simulate_event_rate(cfg.to_process(), 8000,
                                    np.random.default_rng(7),
                                    base_lambda=cfg.base_lambda,
                                    clamp=None, warmup=1000)
            rates.append(r["realized_rate"])
        self.assertLess(abs(rates[0] - rates[1]) / 20.0, 0.05,
                        f"β 改变了长期活跃度：{rates}")


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
