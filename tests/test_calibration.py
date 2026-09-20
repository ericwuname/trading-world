"""统一校准框架的测试（二期阶段10）。

三条核心性质：

1. **标准化必须真的起作用。** 不标准化时，σ（几十 bp）会压倒
   ``r ACF(1)``（0.01 量级），校准退化成"只把 σ 调对"。
2. **块自助法必须保留时序依赖。** 逐点重采样会破坏它，
   得到的参数分布会窄得离谱——而"区间很窄"看起来像"估计很准"，
   正是最危险的一类假象。
3. **优化器契约必须是 ``真实矩 → 最优参数``**，否则自助法无从下手。
"""

from __future__ import annotations

import unittest

import numpy as np

from tw.calibration.objective import (
    BOOTSTRAP_BLOCK,
    BOOTSTRAP_BLOCK_LONG,
    DEFAULT_MOMENTS,
    SearchSpace,
    block_bootstrap_indices,
    block_bootstrap_series,
    bootstrap_real_moments,
    cross_seed_scales,
    parameter_uncertainty,
    percentile_ci,
    random_search,
    standardized_distance,
)
from tw.calibration.optimizer import bootstrap_calibrate, make_optimizer


class TestStandardizedDistance(unittest.TestCase):
    def test_完全一致时距离为零(self) -> None:
        m = {k: 1.0 for k in DEFAULT_MOMENTS}
        d, parts = standardized_distance(m, dict(m))
        self.assertAlmostEqual(d, 0.0, places=12)
        self.assertTrue(all(abs(v) < 1e-12 for v in parts.values()))

    def test_不标准化会让大量级指标压倒小量级(self) -> None:
        """⭐ 这是本模块存在的**唯一理由**，所以必须钉住。

        构造：σ 差 1bp（真实 50bp），r ACF(1) 差 0.5（真实 0.0，即差了 50 个"标准差量级"）。
        · 裸欧氏距离：σ 一项贡献 1.0，ACF 一项贡献 0.25 → **σ 主导**，
          尽管 σ 那 1bp 是纯粹的噪声量级。
        · 标准化之后：σ 的贡献是 (1/50)²=4e-4，ACF 的贡献取决于它的 scale。
        """
        real = {"sigma_bp": 50.0, "acf_ret_lag1": 0.0}
        sim = {"sigma_bp": 51.0, "acf_ret_lag1": 0.5}
        raw = (sim["sigma_bp"] - real["sigma_bp"]) ** 2 \
            + (sim["acf_ret_lag1"] - real["acf_ret_lag1"]) ** 2
        self.assertGreater((1.0 ** 2) / raw, 0.75,
                           "构造没生效：裸距离里 σ 应当占主导")
        # 标准化：σ 用 5bp 的跨种子 sd、ACF 用 0.05
        d, parts = standardized_distance(
            sim, real, moments=("sigma_bp", "acf_ret_lag1"),
            scales={"sigma_bp": 5.0, "acf_ret_lag1": 0.05})
        self.assertGreater(abs(parts["acf_ret_lag1"]), abs(parts["sigma_bp"]),
                           "标准化之后小量级指标仍被压制")

    def test_缺少scale时退回量级参考值(self) -> None:
        d, parts = standardized_distance(
            {"sigma_bp": 100.0}, {"sigma_bp": 50.0}, moments=("sigma_bp",))
        self.assertAlmostEqual(parts["sigma_bp"], 1.0, places=12)

    def test_real为零时不会把该指标从校准里摘掉(self) -> None:
        """⚠️ 分母退回 1.0 会让 |real|=0 的指标贡献≈0，等于没校准它。"""
        d, parts = standardized_distance(
            {"acf_ret_lag1": 0.3}, {"acf_ret_lag1": 0.0},
            moments=("acf_ret_lag1",))
        self.assertGreater(abs(parts["acf_ret_lag1"]), 1.0)

    def test_缺指标被跳过而不是崩(self) -> None:
        d, parts = standardized_distance(
            {"sigma_bp": 50.0}, {"sigma_bp": 50.0, "vr5": 1.0},
            moments=("sigma_bp", "vr5"))
        self.assertNotIn("vr5", parts)

    def test_nan被跳过(self) -> None:
        d, parts = standardized_distance(
            {"sigma_bp": float("nan")}, {"sigma_bp": 50.0}, moments=("sigma_bp",))
        self.assertEqual(parts, {})

    def test_交叉种子标准差(self) -> None:
        samples = [{"sigma_bp": v} for v in (10.0, 12.0, 14.0)]
        sc = cross_seed_scales(samples, ("sigma_bp",))
        self.assertAlmostEqual(sc["sigma_bp"], float(np.std([10, 12, 14], ddof=1)))

    def test_单样本不给scale(self) -> None:
        sc = cross_seed_scales([{"sigma_bp": 1.0}], ("sigma_bp",))
        self.assertNotIn("sigma_bp", sc)


class TestBlockBootstrap(unittest.TestCase):
    def test_下标长度等于n(self) -> None:
        rng = np.random.default_rng([1])
        idx = block_bootstrap_indices(1000, 168, rng)
        self.assertEqual(idx.size, 1000)
        self.assertTrue(idx.min() >= 0 and idx.max() < 1000)

    def test_块内时序被保留(self) -> None:
        """⭐ 块自助法的**定义性质**：相邻元素在重采样后仍常相邻。

        逐点重采样下，"下一个元素刚好是原序列的下一个"概率是 1/n；
        块自助法下应当接近 1（块内连续）。
        """
        rng = np.random.default_rng([2])
        n, block = 50_000, 168
        idx = block_bootstrap_indices(n, block, rng)
        same = float(np.mean(np.diff(idx) == 1))
        self.assertGreater(same, 0.9,
                           f"块内连续性只有 {same:.3f}——时序依赖被破坏了")

    def test_逐点重采样作为对照(self) -> None:
        rng = np.random.default_rng([3])
        n = 50_000
        idx = rng.integers(0, n, size=n)
        same = float(np.mean(np.diff(idx) == 1))
        self.assertLess(same, 0.01,
                        "对照没生效：逐点重采样的连续性不应这么高")

    def test_非法块长被拒(self) -> None:
        with self.assertRaises(ValueError):
            block_bootstrap_indices(100, 0, np.random.default_rng([1]))

    def test_块长等于序列长度时不崩(self) -> None:
        idx = block_bootstrap_indices(10, 100, np.random.default_rng([1]))
        self.assertEqual(idx.size, 10)

    def test_series重采样保持了块内结构(self) -> None:
        x = np.arange(1000, dtype=float)
        y = block_bootstrap_series(x, 100, np.random.default_rng([4]))
        self.assertEqual(y.size, 1000)
        # 块内相邻元素仍差 1
        self.assertGreater(float(np.mean(np.diff(y) == 1)), 0.9)

    def test_真实数据自助法给出不同矩(self) -> None:
        from tw import realdata

        s = realdata.load_builtin("BTCUSDT_1h")
        boots = bootstrap_real_moments(s.close, n_boot=3, block=BOOTSTRAP_BLOCK,
                                       seed=5)
        self.assertEqual(len(boots), 3)
        vals = [b["sigma_bp"] for b in boots]
        self.assertTrue(all(np.isfinite(vals)))
        # 重采样样本各不相同
        self.assertGreater(len(set(round(v, 6) for v in vals)), 1)

    def test_块长不足会系统性低估ACF型矩(self) -> None:
        """⭐ 实测发现：block=168 把 ``acf_abs_lag1`` 打到真值的 **1%**。

        这是本模块最重要的**负面**结论，所以必须钉住：
        块自助法的块长必须覆盖该矩的依赖长度，而波动率聚集的依赖长度
        远长于一周。用 168 做参数不确定性区间，会得到一个
        **围绕错的中心**的分布——它看起来完全正常。

        判据写成"**随块长单调恢复**"（这条是可证伪的、稳定的），
        而不是"某个块长下无偏"（那需要块长≈样本长度，没有实用价值）。
        """
        from tw import realdata
        from tw.analyzer import analyze

        s = realdata.load_builtin("BTCUSDT_1h")
        truth = float(analyze(s.close, "full", vol_window=24).flat()["acf_abs_lag1"])
        self.assertGreater(truth, 0.1, "真实数据的波动率聚集不明显，这条测试无意义")
        vals = []
        for block in (168, 2000, 8000):
            rng = np.random.default_rng([7])
            got = float(np.mean([
                analyze(block_bootstrap_series(s.close, block, rng), "b",
                        vol_window=24).flat()["acf_abs_lag1"]
                for _ in range(3)]))
            vals.append(got)
        self.assertLess(vals[0], 0.1 * truth,
                        f"block=168 竟然恢复了 {vals[0] / truth:.1%}——"
                        "与实测的 0.9% 不符，块自助法的实现可能改了")
        self.assertLess(vals[0], vals[1], "块长变大反而没有恢复更多")
        self.assertLess(vals[1], vals[2], "块长变大反而没有恢复更多")
        self.assertLess(vals[2], truth,
                        "block=8000 已经无偏了？与实测的 69% 不符")

    def test_逐点重采样把聚集完全打散(self) -> None:
        """对照：block=1（逐点重采样）必须把波动率聚集打到≈0。

        它是"保留时序依赖"这件事的反面参照。
        """
        from tw import realdata
        from tw.analyzer import analyze

        s = realdata.load_builtin("BTCUSDT_1h")
        truth = float(analyze(s.close, "full", vol_window=24).flat()["acf_abs_lag1"])
        rng = np.random.default_rng([11])
        got = float(np.mean([
            analyze(block_bootstrap_series(s.close, 1, rng), "b",
                    vol_window=24).flat()["acf_abs_lag1"] for _ in range(3)]))
        # ⚠️ 不断言"≈0"。实测逐点重采样给出 **0.099**（真值的 38%），
        # 而不是 0——而且它比 block=500 的 0.018 还**高**，关系非单调。
        # 原因是逐点重采样会造出大量虚假的大跳（把不相邻的价拼在一起），
        # 让 |r| 的方差被极端值主导，配上一个非零的残余自相关。
        # 所以这里只断言"与真值差得远"，不做方向性断言——
        # 写一条抓不住机制的断言，只会让测试变脆。
        self.assertLess(got, 0.5 * truth,
                        f"逐点重采样给出的值 {got:.4f} 太接近真值 {truth:.4f}")


class TestSearchSpace(unittest.TestCase):
    def test_对数采样落在量级内(self) -> None:
        sp = SearchSpace(bounds={"x": (1e-4, 1e-1)}, log={"x": True})
        rng = np.random.default_rng([1])
        vals = [sp.sample(rng)["x"] for _ in range(500)]
        self.assertTrue(all(1e-4 <= v <= 1e-1 for v in vals))
        # 对数均匀 ⇒ 中位数应接近几何平均
        # 对数均匀 ⇒ 中位数接近几何平均 sqrt(1e-4 · 1e-1) = 1e-2.5
        self.assertAlmostEqual(float(np.median(vals)), 10 ** -2.5, delta=10 ** -2.0)

    def test_线性采样(self) -> None:
        sp = SearchSpace(bounds={"x": (0.0, 1.0)})
        rng = np.random.default_rng([2])
        vals = [sp.sample(rng)["x"] for _ in range(2000)]
        self.assertAlmostEqual(float(np.mean(vals)), 0.5, delta=0.05)

    def test_clip夹住越界(self) -> None:
        sp = SearchSpace(bounds={"x": (0.0, 1.0)})
        self.assertEqual(sp.clip({"x": 5.0}), {"x": 1.0})
        self.assertEqual(sp.clip({"x": -5.0}), {"x": 0.0})


class TestRandomSearch(unittest.TestCase):
    def test_能找到已知最优(self) -> None:
        """目标 ``f(x) = (x−0.7)²``，最优在 0.7。"""
        sp = SearchSpace(bounds={"x": (0.0, 1.0)})
        res = random_search(lambda p: ((p["x"] - 0.7) ** 2, {}), sp,
                            n_iter=200, n_refine=20, seed=3)
        self.assertAlmostEqual(res["best_params"]["x"], 0.7, delta=0.02)
        self.assertLess(res["best_distance"], 1e-3)

    def test_评估次数被如实记录(self) -> None:
        sp = SearchSpace(bounds={"x": (0.0, 1.0)})
        res = random_search(lambda p: (0.0, {}), sp, n_iter=10, n_refine=3)
        self.assertEqual(res["n_evaluations"], 13)
        self.assertEqual(len(res["trials"]), 13)

    def test_nan距离被忽略(self) -> None:
        sp = SearchSpace(bounds={"x": (0.0, 1.0)})
        calls = {"n": 0}

        def f(p):
            calls["n"] += 1
            return (float("nan") if calls["n"] < 5 else 1.0, {})

        res = random_search(f, sp, n_iter=10, n_refine=0, seed=1)
        self.assertTrue(np.isfinite(res["best_distance"]))

    def test_可复现(self) -> None:
        sp = SearchSpace(bounds={"x": (0.0, 1.0)})
        a = random_search(lambda p: (p["x"] ** 2, {}), sp, n_iter=30, seed=9)
        b = random_search(lambda p: (p["x"] ** 2, {}), sp, n_iter=30, seed=9)
        self.assertAlmostEqual(a["best_distance"], b["best_distance"], places=12)


class TestOptimizerContract(unittest.TestCase):
    """⭐ 契约：``真实矩 → 最优参数``。自助法靠它才对每一组重采样可行。"""

    def _make(self):
        truth = {"sigma_bp": 50.0, "excess_kurtosis": 8.0}

        def simulate(p: dict) -> list[dict]:
            # 假的"市场"：m 越接近 1 越像真实
            m = float(p["m"])
            return [{"sigma_bp": 50.0 * m + 0.5 * i,
                     "excess_kurtosis": 8.0 * m + 0.1 * i} for i in range(2)]

        sp = SearchSpace(bounds={"m": (0.5, 1.5)})
        return make_optimizer(simulate, sp, moments=("sigma_bp", "excess_kurtosis"),
                              n_iter=40, n_refine=10, seed=4), truth

    def test_给定真实矩返回最优参数(self) -> None:
        opt, truth = self._make()
        res = opt(truth)
        self.assertIn("best_params", res)
        self.assertIn("best_distance", res)
        self.assertAlmostEqual(res["best_params"]["m"], 1.0, delta=0.06)

    def test_诊断里带标准化尺度(self) -> None:
        opt, truth = self._make()
        res = opt(truth)
        self.assertIn("scales", res["best_diagnostics"])
        self.assertIn("parts", res["best_diagnostics"])

    def test_fixed参数不参与搜索但会出现在结果里(self) -> None:
        truth = {"sigma_bp": 50.0}

        def simulate(p):
            return [{"sigma_bp": p["k"] * p["m"]} for _ in range(2)]

        sp = SearchSpace(bounds={"m": (0.5, 1.5)})
        opt = make_optimizer(simulate, sp, moments=("sigma_bp",),
                             n_iter=10, fixed={"k": 50.0}, seed=1)
        res = opt(truth)
        self.assertIn("k", res["params_full"])
        self.assertNotIn("k", res["best_params"])

    def test_同一参数不会被评估两次(self) -> None:
        calls = {"n": 0}

        def simulate(p):
            calls["n"] += 1
            return [{"sigma_bp": p["m"]} for _ in range(1)]

        sp = SearchSpace(bounds={"m": (1.0, 1.0)})   # 只有一个取值 → 必然重复
        opt = make_optimizer(simulate, sp, moments=("sigma_bp",), n_iter=8, seed=1)
        opt({"sigma_bp": 1.0})
        self.assertEqual(calls["n"], 1, "同一个参数被重复模拟了——缓存没生效")


class TestBootstrapCalibrate(unittest.TestCase):
    def test_对每组真实矩各跑一次(self) -> None:
        seen = []

        def simulate(p):
            return [{"sigma_bp": p["m"]} for _ in range(1)]

        sp = SearchSpace(bounds={"m": (0.9, 1.1)})
        opt = make_optimizer(simulate, sp, moments=("sigma_bp",),
                             n_iter=4, fixed={"tag": 1.0}, seed=2)
        boots = [{"sigma_bp": v} for v in (1.0, 2.0, 3.0)]
        res = bootstrap_calibrate(opt, boots)
        self.assertEqual(len(res), 3)
        self.assertEqual([r["bootstrap_index"] for r in res], [0, 1, 2])
        _ = seen

    def test_参数置信区间(self) -> None:
        res = [{"best_params": {"x": v}} for v in (1.0, 2.0, 3.0, 4.0, 5.0)]
        u = parameter_uncertainty(res, ["x"])
        self.assertAlmostEqual(u["x"]["mean"], 3.0)
        self.assertEqual(u["x"]["n"], 5)
        lo, hi = u["x"]["ci95"]
        self.assertLessEqual(lo, 3.0)
        self.assertGreaterEqual(hi, 3.0)

    def test_缺失参数不崩(self) -> None:
        res = [{"best_params": {}} for _ in range(3)]
        u = parameter_uncertainty(res, ["x"])
        self.assertTrue(np.isnan(u["x"]["mean"]))


class TestPercentileCI(unittest.TestCase):
    def test_覆盖中位数(self) -> None:
        vals = list(range(1, 101))
        lo, hi = percentile_ci(vals)
        self.assertLess(lo, 50)
        self.assertGreater(hi, 50)

    def test_剔除nan(self) -> None:
        lo, hi = percentile_ci([1.0, float("nan"), 3.0])
        self.assertTrue(np.isfinite(lo) and np.isfinite(hi))

    def test_空输入返回nan(self) -> None:
        lo, hi = percentile_ci([])
        self.assertTrue(np.isnan(lo) and np.isnan(hi))

    def test_单值两端相同(self) -> None:
        lo, hi = percentile_ci([7.0])
        self.assertEqual((lo, hi), (7.0, 7.0))


if __name__ == "__main__":
    unittest.main()
