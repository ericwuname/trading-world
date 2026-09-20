"""长记忆分析器的测试（二期阶段6）。

这个文件测的是**统计护栏**，不是业务逻辑。理由很直接：
阶段6 的结论（"订单流出现了长记忆"）完全由这些函数的判据决定，
而它们的失效方式是**给出一个漂亮的数字**，不是抛异常。

三条护栏各有一组测试：
1. 显著性 ≠ 效应量（N 大时 ACF(1)=0.01 也"显著"，但它是零）
2. 拟合前必须检查自变量对数跨度（一期的 k=1.72/R²=0.987 就是这么来的）
3. 只对**连续为正**的那一段拟合（筛负值再拟合会测出假长记忆）
"""

from __future__ import annotations

import unittest

import numpy as np

from tw.analyzer_longmemory import (
    MIN_LOG_SPAN,
    acf_positive_run,
    acf_significant_lags,
    acf_sum,
    acf_white_noise_band,
    fit_power_law_decay,
    longmemory_summary,
    order_sign_acf,
    order_signs,
    tick_signed_flow,
    trade_signs,
)


class _T:
    """最小成交替身：tick / aggressor_side / 两侧订单号。

    订单号是必需的——``order_signs`` 靠它把"同一张主动单扫出的多笔成交"
    归成一个符号。不给它就没法测 LMF 意义上的 ε 序列。
    """

    __slots__ = ("tick", "aggressor_side", "buy_order_id", "sell_order_id")

    def __init__(self, tick: int, side: str, oid: str | None = None) -> None:
        self.tick = tick
        self.aggressor_side = side
        self.buy_order_id = (oid or f"b{tick}") if side == "buy" else "passive"
        self.sell_order_id = (oid or f"s{tick}") if side == "sell" else "passive"


def _ar1_signs(n: int, rho: float, seed: int = 0):
    """生成一个 AR(1) 型符号序列：``x_t = rho·x_{t-1} + noise``，取符号。

    用来验证"有长记忆时 ACF 应该长什么样"。注意：AR(1) 的 ACF 是**指数**
    衰减，不是幂律——所以它够用来测方向性，但**不该**被拟合出漂亮的幂律指数。
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    eps = rng.normal(0.0, 1.0, n)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + eps[i]
    return np.sign(x)


def _powerlaw_signs(n: int, gamma: float, seed: int = 0, scale: float = 1.0):
    """用**分数布朗运动**造一个 ACF ∝ k^{−γ} 的符号序列。

    做法：生成 Hurst = 1 − γ/2 的分数高斯噪声，取符号。
    这是 LMF 用来描述订单流的同一族过程，所以用它验证拟合是无偏的。
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    H = 1.0 - gamma / 2.0
    # 谱合成法生成 fGn
    k = np.arange(1, n // 2 + 1)
    f = rng.normal(0.0, 1.0, k.size) + 1j * rng.normal(0.0, 1.0, k.size)
    S = np.abs(f) ** 2 * k ** (-(2.0 * H - 1.0))
    z = np.sqrt(S) * (rng.normal(0.0, 1.0, k.size)
                      + 1j * rng.normal(0.0, 1.0, k.size))
    x = np.fft.irfft(z, n=n)
    x = x * scale
    return np.sign(x[x != 0.0])


class TestTradeSigns(unittest.TestCase):
    def test_方向映射(self) -> None:
        ts = [_T(0, "buy"), _T(1, "sell"), _T(2, "buy")]
        signs = trade_signs(ts)
        self.assertEqual(list(signs), [1.0, -1.0, 1.0])

    def test_空输入(self) -> None:
        self.assertEqual(trade_signs([]).size, 0)

    def test_均值在有偏流下不为零(self) -> None:
        ts = [_T(i, "buy") for i in range(10)]
        self.assertAlmostEqual(float(trade_signs(ts).mean()), 1.0)


class TestOrderSigns(unittest.TestCase):
    """⭐ 记账粒度：LMF 的 ε 是**每张订单一个符号**，不是每笔成交。"""

    def test_同一张单的多笔成交只算一个符号(self) -> None:
        """一个主动单扫多档 → 多笔成交 → 只能贡献**一个** ε。"""
        ts = [
            _T(0, "buy"), _T(0, "buy"), _T(0, "buy"),      # 同一张买单扫 3 档
            _T(1, "sell"),                                  # 另一张卖单
        ]
        for t, oid in zip(ts, ("a#1", "a#1", "a#1", "b#1")):
            t.buy_order_id = oid if t.aggressor_side == "buy" else "x"
            t.sell_order_id = oid if t.aggressor_side == "sell" else "y"
        signs = order_signs(ts)
        self.assertEqual(list(signs), [1.0, -1.0])

    def test_连续两张不同单不会被合并(self) -> None:
        ts = [_T(0, "buy"), _T(0, "buy")]
        ts[0].buy_order_id, ts[1].buy_order_id = "a#1", "a#2"
        self.assertEqual(list(order_signs(ts)), [1.0, 1.0])

    def test_聚合会显著降低机械自相关(self) -> None:
        """⭐ 这条把"为什么要聚合"钉成可复现的事实。

        构造：每张买卖单扫 5 档（产生 5 笔同向成交），订单符号本身随机。
        逐笔算 ACF(1) 会得到 **0.8 左右**的机械正自相关；
        按订单聚合后应当回到 0 附近。

        这正是探针里一期基线"逐笔 ACF(1)=+0.22"的来源——
        不是"订单流有记忆"，是记账粒度的产物。
        """
        rng = np.random.default_rng(5)
        trades = []
        for i in range(4000):
            side = "buy" if rng.random() < 0.5 else "sell"
            for _ in range(5):
                t = _T(i, side)
                # 一张主动单的身份：买单用 buy_order_id，卖单用 sell_order_id
                t.buy_order_id = f"o{i}" if side == "buy" else "passive"
                t.sell_order_id = f"o{i}" if side == "sell" else "passive"
                trades.append(t)
        a_trade = float(order_sign_acf(trade_signs(trades), max_lag=5)[0])
        a_order = float(order_sign_acf(order_signs(trades), max_lag=5)[0])
        self.assertGreater(a_trade, 0.5, "构造没生效：逐笔 ACF 应当被机械地抬高")
        self.assertLess(abs(a_order), 0.1, "按订单聚合后仍有机械自相关")


class TestAcf(unittest.TestCase):
    def test_fft实现与直接算法一致(self) -> None:
        """⭐ FFT 只是**加速**，不是放松口径。

        ``order_sign_acf`` 从双重循环换成了 FFT 自相关，因为 40 万张订单
        × 200 lag 的循环要跑几十秒，多种子配对实验根本做不完。
        FFT 给的是**循环**自相关，n ≫ lag 时与线性口径的差可忽略——
        但"可忽略"必须被验证，不能假设。
        """
        rng = np.random.default_rng(77)
        for n in (500, 5000):
            x = rng.choice([-1.0, 1.0], n)
            got = order_sign_acf(x, max_lag=60)
            xc = x - x.mean()
            den = float(np.dot(xc, xc))
            want = np.array([float(np.dot(xc[: n - k], xc[k:]) / den)
                             for k in range(1, 61)])
            np.testing.assert_allclose(got, want, rtol=0, atol=5e-4,
                                       err_msg=f"n={n} 时 FFT 与直接算法不一致")

    def test_记忆为空时acf接近0(self) -> None:
        rng = np.random.default_rng(1)
        a = order_sign_acf(rng.choice([-1.0, 1.0], 50_000), max_lag=20)
        self.assertLess(float(np.abs(a).max()), 0.05)

    def test_常数列的acf为0(self) -> None:
        """全买（或全卖）序列去均值后恒为 0，分母为 0 → 返回空，不崩。"""
        a = order_sign_acf([1.0] * 100, max_lag=10)
        self.assertEqual(a.size, 0)

    def test_正记忆时acf为正(self) -> None:
        a = order_sign_acf(_ar1_signs(20_000, 0.9, seed=3), max_lag=30)
        self.assertGreater(float(a[0]), 0.2)
        self.assertGreater(float(a[9]), 0.0)

    def test_lag数受样本长度限制(self) -> None:
        a = order_sign_acf([1.0, -1.0] * 10, max_lag=1000)
        self.assertLessEqual(a.size, 18)

    def test_太短序列返回空(self) -> None:
        self.assertEqual(order_sign_acf([1.0, -1.0]).size, 0)


class TestSignificanceVsEffectSize(unittest.TestCase):
    """⭐ 护栏①：显著性 ≠ 效应量。"""

    def test_样本越大显著带越窄(self) -> None:
        self.assertLess(acf_white_noise_band(1_000_000), acf_white_noise_band(100))

    def test_巨大样本下微小的acf也会显著(self) -> None:
        """10 万笔时 se≈0.003，ACF(1)=0.01 就能"显著"——**但它是零**。

        这条测试存在的意义是把这件事写进代码：验收判据不能只看"显著"，
        必须同时给效应量下界。
        """
        n = 100_000
        band = acf_white_noise_band(n, z=3.0)
        self.assertLess(band, 0.02)
        self.assertEqual(acf_significant_lags([0.01] * 50, band), 50)

    def test_显著带比例于1除以根号n(self) -> None:
        self.assertAlmostEqual(
            acf_white_noise_band(400, z=3.0), 3.0 / 20.0, places=12
        )

    def test_连续为正的长度比显著个数更严(self) -> None:
        a = [0.5, 0.4, -0.01, 0.3, 0.3]
        self.assertEqual(acf_positive_run(a), 2)
        self.assertEqual(acf_significant_lags(a, 0.01), 4)

    def test_acf_sum只累加连续正的段(self) -> None:
        self.assertAlmostEqual(acf_sum([0.3, 0.2, -0.5, 0.9]), 0.5)

    def test_acf_sum空输入(self) -> None:
        self.assertEqual(acf_sum([]), 0.0)


class TestDecayFitGuards(unittest.TestCase):
    """⭐ 护栏②③：跨度与"只取连续正段"。"""

    def test_幂律信号能还原指数(self) -> None:
        acf = 0.5 * np.arange(1, 201, dtype=float) ** (-0.5)
        fit = fit_power_law_decay(acf)
        self.assertTrue(fit.ok, fit.reason)
        self.assertAlmostEqual(fit.exponent, 0.5, places=3)
        self.assertGreater(fit.r_squared, 0.99)

    def test_hurst由指数反解(self) -> None:
        """γ=0.5 ⇒ H=0.75，正是真实订单流的量级。"""
        acf = 0.5 * np.arange(1, 201, dtype=float) ** (-0.5)
        fit = fit_power_law_decay(acf)
        self.assertAlmostEqual(fit.hurst, 0.75, places=3)

    def test_γ为1时hurst为0点5(self) -> None:
        """γ=1 ⇒ H=0.5 = 无关随机，即"没有记忆"。"""
        acf = 0.5 * np.arange(1, 201, dtype=float) ** (-1.0)
        fit = fit_power_law_decay(acf)
        self.assertAlmostEqual(fit.hurst, 0.5, places=3)

    def test_跨度不足要拒绝(self) -> None:
        """⭐ 一期真实场景：lag 只到 5（跨度 0.7 个数量级）时能拟合出漂亮指数。

        这里故意让 ``min_points`` 这一关**过不去**，好让护栏②（跨度）真正
        成为拦下它的那一道——否则测的是护栏①，不是护栏②。
        lag=1..5 的对数跨度 = log10(5) ≈ 0.70 < 1.0。
        """
        acf = 0.5 * np.arange(1, 6, dtype=float) ** (-0.5)
        fit = fit_power_law_decay(acf, min_points=3)
        self.assertFalse(fit.ok)
        self.assertIn("跨度", fit.reason)
        self.assertLess(fit.log_span, MIN_LOG_SPAN)

    def test_点数不足也要拒绝(self) -> None:
        """护栏的另一道：可用 lag 点太少连护栏②都轮不到。"""
        fit = fit_power_law_decay([0.5, 0.45, 0.4])
        self.assertFalse(fit.ok)
        self.assertIn("不足", fit.reason)

    def test_首lag非正直接拒绝(self) -> None:
        fit = fit_power_law_decay([-0.1] + [0.2] * 100)
        self.assertFalse(fit.ok)
        self.assertIn("非正", fit.reason)

    def test_不筛负值只取连续正段(self) -> None:
        """⭐ 护栏③：筛掉负值再拟合会在尾部噪声里挑出零星正值，
        把斜率往 0 靠——测出**假的长记忆**。

        构造：前 50 个点严格幂律衰减，第 51 个点是**非正值**（截断标记），
        之后是"平时接近 0、偶尔冒一个正值"的尾部噪声。
        正确行为是在第 51 个点处**停下**（只能用 50 个点）；
        若错误地"筛掉负值再拟合"，会把这几十个尾部正值也吃进去，
        被拉成一条接近水平的线，指数趋近 0。
        """
        head = 0.5 * np.arange(1, 51, dtype=float) ** (-0.5)
        mid = np.array([-1e-3])            # 第 51 个：非正 → 截断点
        tail = np.full(120, 1e-6)
        tail[::7] = 0.02                   # 零星正值
        fit = fit_power_law_decay(np.concatenate([head, mid, tail]))
        self.assertTrue(fit.ok, fit.reason)
        self.assertEqual(fit.n_points, 50,
                         f"拟合吃进了截断点之后的点（用了 {fit.n_points} 个）")
        self.assertAlmostEqual(fit.exponent, 0.5, places=2)

    def test_因变量退化要拒绝(self) -> None:
        acf = np.full(200, 0.3)
        fit = fit_power_law_decay(acf)
        self.assertFalse(fit.ok)
        self.assertIn("退化", fit.reason)

    def test_指数衰减的信号不该拟合出漂亮幂律(self) -> None:
        """AR(1) 的 ACF 是指数衰减，不是幂律。

        这里不要求它被拒绝（跨度够、值也在变，护栏管不到），
        但要求它拟合出的指数**明显小于**真幂律的情形——
        说明这个测量对"记忆的形状"有区分力，不是个好坏的万金油。
        """
        import numpy as np

        ar = 0.9 ** np.arange(0, 200)
        pl = 1.0 * np.arange(1, 201, dtype=float) ** (-0.5)
        f_ar = fit_power_law_decay(ar)
        f_pl = fit_power_law_decay(pl)
        self.assertTrue(f_ar.ok and f_pl.ok)
        self.assertGreater(f_ar.exponent, f_pl.exponent)

    def test_空输入不崩(self) -> None:
        fit = fit_power_law_decay([])
        self.assertFalse(fit.ok)


class TestSummary(unittest.TestCase):
    def _mk(self, n: int = 4000, rho: float = 0.0, seed: int = 0):
        import numpy as np

        rng = np.random.default_rng(seed)
        signs = (_ar1_signs(n, rho, seed) if rho > 0
                 else rng.choice([-1.0, 1.0], n))
        trades = [_T(i, "buy" if s > 0 else "sell") for i, s in enumerate(signs)]
        flow = np.nan_to_num(signs.astype(float), nan=0.0)
        return trades, flow

    def test_无记忆流的关键字段(self) -> None:
        trades, flow = self._mk(5000, 0.0, seed=2)
        s = longmemory_summary(trades, flow, 0, 5000, max_lag=50)
        self.assertEqual(s["n_trades"], 5000)
        self.assertAlmostEqual(s["buy_frac"], 0.5, delta=0.05)
        self.assertLess(abs(s["acf_lag1"]), 0.1)
        self.assertEqual(len(s["acf"]), 50)

    def test_有记忆流的acf更长(self) -> None:
        t0, f0 = self._mk(5000, 0.0, seed=4)
        t1, f1 = self._mk(5000, 0.95, seed=4)
        a = longmemory_summary(t0, f0, 0, 5000, max_lag=100)
        b = longmemory_summary(t1, f1, 0, 5000, max_lag=100)
        self.assertGreater(b["acf_lag1"], a["acf_lag1"])
        self.assertGreater(b["positive_run"], a["positive_run"])
        self.assertGreater(b["acf_sum"], a["acf_sum"])

    def test_窗口参数真的生效(self) -> None:
        trades, flow = self._mk(6000, 0.8, seed=6)
        s = longmemory_summary(trades, flow, 1000, 3000, max_lag=20)
        self.assertEqual(s["window"], [1000, 3000])
        self.assertEqual(s["n_trades"], 2000)

    def test_样本不足时返回可读结论而不是崩(self) -> None:
        s = longmemory_summary([], __import__("numpy").zeros(10), 0, 10)
        self.assertFalse(s["fit_ok"])
        self.assertIn("不足", s["fit_reason"])

    def test_tick级指标存在且与逐笔分开(self) -> None:
        trades, flow = self._mk(4000, 0.9, seed=8)
        s = longmemory_summary(trades, flow, 0, 4000, max_lag=50)
        self.assertIn("tick_acf_lag1", s)
        self.assertIn("tick_nonzero_ticks", s)
        self.assertIn("tick_acf_sum", s)

    def test_全零flow不崩(self) -> None:
        import numpy as np

        trades = [_T(i, "buy") for i in range(100)]
        s = longmemory_summary(trades, np.zeros(100), 0, 100, max_lag=20)
        self.assertIn("buy_frac", s)

    def test_tick_signed_flow切片(self) -> None:
        import numpy as np

        flow = np.array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(tick_signed_flow(flow, 1, 3), [2.0, 3.0])

    def test_tick_signed_flow处理nan(self) -> None:
        import numpy as np

        flow = np.array([np.nan, 1.0, np.nan])
        np.testing.assert_allclose(tick_signed_flow(flow, 0, 3), [0.0, 1.0, 0.0])


class TestOnRealMarket(unittest.TestCase):
    """在真实模拟上跑一遍：确保没有"只在合成数据上对"的假设。"""

    def test_模拟上不崩且字段齐全(self) -> None:
        from tw import Market, Population, SimConfig

        m = Market(SimConfig(seed=17, n_ticks=800,
                             population=Population.from_shares(
                                 80, {"zero_intel": 0.3, "fundamentalist": 0.4,
                                      "chartist": 0.3})))
        m.run(800)
        s = longmemory_summary(m.log.trades, m.log.flow, 400, 800, max_lag=100)
        self.assertGreater(s["n_trades"], 50)
        for k in ("acf", "band", "n_sig_lags", "positive_run", "acf_sum",
                  "fit_ok", "hurst", "decay_exponent"):
            self.assertIn(k, s)
        self.assertGreaterEqual(s["positive_run"], 0)
        self.assertLessEqual(s["positive_run"], 100)


if __name__ == "__main__":
    unittest.main()
