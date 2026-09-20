"""突发/背景期深度对比工具（A 线 H1）的回归测试。

为什么要专门测这个工具
----------------------
EA.2 是**唯一**能回答 H1（"突发期局部流动性变薄"）的实验，而它的结论
完全依赖两个纯函数：``classify_burst_vs_background`` 的阈值方向，
以及 ``burst_depth_contrast`` 的符号约定。方向写反不会报错，
只会让结论**反过来**（"更薄"变成"更厚"）——这正是变异体 M35 的形态。

另外这里钉住一条**统计纪律**：同一个量在"tick 级"与"跨运行级"下的
不确定性差一个数量级，结论必须以跨运行为准。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tw.impact import (  # noqa: E402
    aggregate_burst_contrast,
    burst_depth_contrast,
    classify_burst_vs_background,
)


def _synthetic(seed: int, *, effect: float, n: int = 4000,
               rho_phi: float = 0.9, depth_phi: float = 0.95):
    """造一条"强度自相关 + 深度自相关"的序列，可选地让突发期深度变薄。

    用自相关（而不是独立同分布）是**故意的**：真实盘口不会每 tick 独立重抽，
    而"tick 级检验的区间窄了一个数量级"这个事实只有在自相关下才成立。
    """
    r = np.random.default_rng(seed)

    def ar1(phi, sd):
        e = r.normal(0, sd * np.sqrt(1 - phi ** 2), n)
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = phi * x[i - 1] + e[i]
        return x

    rho = 1.0 + ar1(rho_phi, 0.25)
    depth = 100.0 + ar1(depth_phi, 3.0)
    mask = classify_burst_vs_background(rho, percentile_threshold=80.0)
    if effect != 0.0:
        depth = depth + effect * mask          # effect < 0 ⇒ 突发期更薄
    return depth, rho, mask


class TestClassifyBurst(unittest.TestCase):
    def test_高强度被标成突发(self) -> None:
        x = np.array([1.0, 1.0, 1.0, 1.0, 10.0])
        m = classify_burst_vs_background(x, percentile_threshold=80.0)
        self.assertTrue(m[-1], "最高强度那个 tick 必须被标成突发")
        self.assertEqual(int(m.sum()), 1)

    def test_掩码选中的必须是高强度那一侧(self) -> None:
        """⭐ 这是变异体 M35 的直接回归。

        ⚠️ **第一版这条测试是错的**：它拿 p80 与 p20 两个阈值去比掩码是否互斥，
        而实现的语义始终是 ``x > thr``——p20 选出的是**上 80%**，本来就与
        p80 大量重叠。那是在测"阈值取值"，不是在测"比较方向"。
        正确做法是直接断言：**被选中的那一侧的均值必须更高**。
        实现若写成 ``x < thr``，选中的就是低强度那批，这条立刻变红。
        """
        rng = np.random.default_rng(0)
        x = rng.uniform(0, 1, 1000)
        m = classify_burst_vs_background(x, percentile_threshold=80.0)
        self.assertGreater(m.sum(), 0)
        self.assertLess(m.sum(), x.size)
        self.assertGreater(float(x[m].mean()), float(x[~m].mean()),
                           "被标成突发的那些 tick 强度反而更低 —— "
                           "比较方向反了（M35 的形态）")

    def test_空输入返回空而不是崩(self) -> None:
        self.assertEqual(classify_burst_vs_background([]).size, 0)


class TestBurstDepthContrast(unittest.TestCase):
    def test_突发期更薄时符号为负(self) -> None:
        d, rho, _ = _synthetic(1, effect=-15.0)
        c = burst_depth_contrast(d, rho, percentile_threshold=80.0, block=500)
        self.assertTrue(c["usable"])
        self.assertLess(c["log_ratio"], 0,
                        "突发期深度更低时 log 比值必须为负")
        self.assertLess(c["pct_diff"], 0)

    def test_突发期更厚时符号为正(self) -> None:
        d, rho, _ = _synthetic(2, effect=+15.0)
        c = burst_depth_contrast(d, rho, percentile_threshold=80.0, block=500)
        self.assertGreater(c["log_ratio"], 0)

    def test_真值零时不宣称有差异(self) -> None:
        """⭐ 这条防的是"判据恒真"。

        8 次运行、真值 0，跨运行检验不应显著。
        如果这里开始报显著，说明实现里混进了某种系统性偏差。
        """
        runs = [burst_depth_contrast(*_synthetic(sd, effect=0.0)[:2],
                                     percentile_threshold=80.0, block=500)
                for sd in range(8)]
        agg = aggregate_burst_contrast(runs)
        self.assertFalse(np.isfinite(agg["per_run"]["p"]) and agg["per_run"]["p"] < 0.05,
                         f"真值 0 却显著（p={agg['per_run']['p']:.4g}）")
        self.assertIn("不显著", agg["verdict"])


class TestStatisticLevels(unittest.TestCase):
    """⭐ 统计纪律：tick 级区间比跨运行级**窄一个数量级**，结论以后者为准。"""

    def test_tick级区间明显窄于跨运行级(self) -> None:
        runs = []
        for sd in range(8):
            d, rho, mask = _synthetic(100 + sd, effect=-15.0)
            runs.append(burst_depth_contrast(d, rho, percentile_threshold=80.0,
                                             block=500))
        agg = aggregate_burst_contrast(runs)
        ci = agg["per_run"]["ci95"]
        half_run = (ci[1] - ci[0]) / 2

        # tick 级：把所有 tick 当独立样本（任务书原本的做法）
        db, dg = [], []
        for sd in range(8):
            d, rho, mask = _synthetic(100 + sd, effect=-15.0)
            db.append(d[mask])
            dg.append(d[~mask])
        a, b = np.concatenate(db), np.concatenate(dg)
        se = float(np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size))
        half_tick = se / b.mean()

        self.assertLess(half_tick, half_run,
                        "tick 级区间竟然不更窄——要么数据不相关，要么实现变了")
        ratio = half_run / max(half_tick, 1e-12)
        self.assertGreater(ratio, 3.0,
                           f"tick 级只窄了 {ratio:.1f} 倍，"
                           f"按本项目的实测应当是 10 倍量级")
        # 并确认结论用的是**跨运行**那一套
        self.assertEqual(agg["per_run"]["n"], 8)


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
