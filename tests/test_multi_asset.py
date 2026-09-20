"""多资产相关市场与配对交易的测试（二期阶段9）。

三条核心性质：

1. **相关性必须来自共享的公共因子。** 各自独立随机游走时相关系数应≈0；
   把公共因子的方差调大时，实测相关系数必须跟着**接近解析值**。
   没有解析靶子的话，一个完全错的相关结构也能被解释成"涌现出来的"。
2. **每个资产是独立的市场对象。** 共享订单簿或主体池会把不同资产的价格
   混进同一条序列——那不是"相关"，那是记账串了。
3. **配对交易的买卖标签不能反。** 反了的话策略 PnL 与 z-score 逻辑完全相反，
   而"PnL 是负的"这个结果看起来照样像"策略不灵"。
"""

from __future__ import annotations

import unittest

import numpy as np

from tw import Market, Population, SimConfig
from tw.agents.pairs_trader import (
    PairsConfig,
    PairsLeg,
    PairsTrader,
    attach_pairs_trader,
)
from tw.multi_asset import (
    AssetMarket,
    AssetSpec,
    CorrelatedFundamentalGenerator,
    MultiAssetConfig,
    MultiAssetMarket,
)


def _cfg(common_vol=3e-4, idio=3e-4, n=2, seed=7):
    return MultiAssetConfig(
        assets=[AssetSpec(f"A{i}", beta=1.0, idio_vol=idio,
                          initial_price=1000.0 * (i + 1)) for i in range(n)],
        common_vol=common_vol, seed=seed,
    )


class TestConfigAndAnalytics(unittest.TestCase):
    def test_解析相关系数(self) -> None:
        """β、σ_c、σ_i 已知时相关系数有闭式解，必须与它一致。"""
        c = _cfg(common_vol=3e-4, idio=3e-4)
        # cov = 1·1·σ_c² = 9e-8；var = σ_c²+σ_i² = 1.8e-7 ⇒ corr = 0.5
        self.assertAlmostEqual(c.implied_correlation(0, 1), 0.5, places=9)

    def test_特异性波动为零时完全相关(self) -> None:
        c = _cfg(common_vol=3e-4, idio=0.0)
        self.assertAlmostEqual(c.implied_correlation(0, 1), 1.0, places=9)

    def test_公共波动为零时不相关(self) -> None:
        c = _cfg(common_vol=0.0, idio=3e-4)
        self.assertAlmostEqual(c.implied_correlation(0, 1), 0.0, places=9)

    def test_相关系数矩阵对称且对角为一(self) -> None:
        m = _cfg(n=4).correlation_matrix()
        np.testing.assert_allclose(m, m.T)
        np.testing.assert_allclose(np.diag(m), np.ones(4))

    def test_少于两资产被拒(self) -> None:
        with self.assertRaises(ValueError):
            MultiAssetConfig(assets=[AssetSpec("A")]).validate()

    def test_非法参数被拒(self) -> None:
        with self.assertRaises(ValueError):
            MultiAssetConfig(assets=[AssetSpec("A", idio_vol=-1.0),
                                     AssetSpec("B")]).validate()
        with self.assertRaises(ValueError):
            MultiAssetConfig(assets=[AssetSpec("A", initial_price=0.0),
                                     AssetSpec("B")]).validate()
        with self.assertRaises(ValueError):
            MultiAssetConfig(assets=[AssetSpec("A"), AssetSpec("B")],
                             common_vol=-1.0).validate()


class TestGenerator(unittest.TestCase):
    def test_公共冲击每步只抽一次(self) -> None:
        """⭐ 见模块文档①：抽多次会让各资产的共同成分不同步，
        实测相关系数会系统性低于解析值。"""
        c = _cfg(n=3)
        g = CorrelatedFundamentalGenerator(c)
        for _ in range(500):
            g.step()
        self.assertEqual(g.n_common_draws, g.n_steps,
                         "公共冲击的抽样次数 != 步数")

    def test_实测相关系数接近解析值(self) -> None:
        """⭐ 这是 E9.1 的核心判据。

        ⚠️ 生成器层面（不经市场）测：市场里还有个价格发现过程，
        会把相关性稀释，那是市场的事，不是生成器的事。
        两者必须分开测，否则分不清"参数错了"与"市场稀释了"。
        """
        c = _cfg(common_vol=3e-4, idio=3e-4)
        g = CorrelatedFundamentalGenerator(c)
        vals = np.array([g.step() for _ in range(20_000)], dtype=float)
        r = np.diff(np.log(vals), axis=0)
        got = float(np.corrcoef(r[:, 0], r[:, 1])[0, 1])
        want = c.implied_correlation(0, 1)
        self.assertAlmostEqual(got, want, delta=0.03,
                               msg=f"实测 {got:.3f} vs 解析 {want:.3f}")

    def test_相关系数随公共波动单调上升(self) -> None:
        def realized(cv: float) -> float:
            c = _cfg(common_vol=cv, idio=3e-4)
            g = CorrelatedFundamentalGenerator(c)
            v = np.array([g.step() for _ in range(8_000)], dtype=float)
            r = np.diff(np.log(v), axis=0)
            return float(np.corrcoef(r[:, 0], r[:, 1])[0, 1])

        vals = [realized(cv) for cv in (1e-4, 3e-4, 1e-3)]
        self.assertTrue(all(vals[i] < vals[i + 1] for i in range(2)),
                        f"相关系数没有随公共波动上升：{vals}")

    def test_无公共因子时相关接近零(self) -> None:
        c = _cfg(common_vol=0.0, idio=3e-4)
        g = CorrelatedFundamentalGenerator(c)
        v = np.array([g.step() for _ in range(20_000)], dtype=float)
        r = np.diff(np.log(v), axis=0)
        got = float(np.corrcoef(r[:, 0], r[:, 1])[0, 1])
        self.assertLess(abs(got), 0.05)

    def test_价格恒为正(self) -> None:
        g = CorrelatedFundamentalGenerator(_cfg(common_vol=2e-2, idio=2e-2))
        for _ in range(3000):
            v = g.step()
            self.assertTrue(np.all(v > 0))

    def test_可复现(self) -> None:
        g1 = CorrelatedFundamentalGenerator(_cfg())
        g2 = CorrelatedFundamentalGenerator(_cfg())
        for _ in range(50):
            np.testing.assert_allclose(g1.step(), g2.step())


class TestMultiAssetMarket(unittest.TestCase):
    def _mk(self, n=2, ticks=1200, **kw):
        c = _cfg(n=n, **kw)
        m = MultiAssetMarket(c, n_ticks=ticks)
        m.run(ticks)
        return m

    def test_每个资产是独立市场(self) -> None:
        m = self._mk(n=3, ticks=300)
        self.assertEqual(len(m.markets), 3)
        books = [id(x.book) for x in m.markets]
        self.assertEqual(len(set(books)), 3, "两个资产共享了同一个订单簿")
        agents = [id(a) for x in m.markets for a in x.agents]
        self.assertEqual(len(set(agents)), len(agents), "主体池被共享了")

    def test_外部锚点驱动基本面(self) -> None:
        """⭐ 市场的基本面必须**逐点**等于生成器给出的值。

        若有偏差，说明 `_step_fundamental` 还在抽自己的随机数——
        那会给资产收益里混进一段与公共因子无关的噪声。
        """
        c = _cfg(n=2)
        m = MultiAssetMarket(c, n_ticks=200)
        for _ in range(100):
            anchors = m.generator.step()
            for i, mk in enumerate(m.markets):
                mk.external_anchor = float(anchors[i])
                mk.step()
                self.assertAlmostEqual(mk.fundamental, float(anchors[i]), places=9)

    def test_单资产不变量_外部锚点为None时行为不变(self) -> None:
        """没有外部锚点时，`AssetMarket` 必须与裸 `Market` 逐点一致。"""
        def run(asset_cls: bool):
            sim = SimConfig(seed=11, n_ticks=400,
                            population=Population.from_shares(
                                100, {"zero_intel": 0.3, "fundamentalist": 0.4,
                                      "chartist": 0.3}))
            mk = AssetMarket(sim, spec=AssetSpec("A")) if asset_cls else Market(sim)
            mk.run(400)
            return np.asarray(mk.log.mid[: mk.tick], dtype=float)

        a, b = run(False), run(True)
        np.testing.assert_array_equal(a, b)

    def test_实测相关系数随参数上升(self) -> None:
        lo = self._mk(common_vol=4e-5, idio=6e-4, ticks=3000)
        hi = self._mk(common_vol=8e-4, idio=6e-4, ticks=3000)
        c_lo = lo.realized_correlation(0, 1, lo.markets[0].cfg.n_ticks // 4)
        c_hi = hi.realized_correlation(0, 1, hi.markets[0].cfg.n_ticks // 4)
        self.assertLess(c_lo, c_hi, f"相关系数没有随公共波动上升：{c_lo:.3f} vs {c_hi:.3f}")

    def test_健康检查(self) -> None:
        m = self._mk(ticks=600)
        ok, probs = m.health_check()
        self.assertTrue(ok, probs[:5])

    def test_mid与mids可用(self) -> None:
        m = self._mk(n=3, ticks=200)
        ms = m.mids()
        self.assertEqual(len(ms), 3)
        self.assertTrue(all(x is None or x > 0 for x in ms))

    def test_相关系数矩阵对称(self) -> None:
        m = self._mk(n=3, ticks=800)
        cm = m.realized_correlation_matrix(200)
        np.testing.assert_allclose(cm, cm.T, atol=1e-9)


class TestRealDataAlignment(unittest.TestCase):
    """⭐ 真实多资产数据的**时序对齐**。踩过一次，值得单列。

    ``SOLUSDT_1h`` 有 17521 行，而 BTC/ETH 各 17520 行。
    直接把三个 ``.close`` 数组当同一时刻的序列去算收益，
    SOL 会整体错位**一个 bar**：BTC/SOL 的相关系数从真值 **+0.773
    掉到 −0.017**。

    这个 bug 的危险之处在于**它看起来完全合理**：
    三对里 BTC/ETH 是 +0.819（这两个长度相同，天然对齐），
    另外两对 ≈ 0 —— 读者会得出「SOL 与大盘不相关」这个
    错误但有故事的结论。
    """

    def test_三个数据集的行数并不相同(self) -> None:
        from tw import realdata

        n = {k: int(np.asarray(realdata.load_builtin(k).timestamp).size)
             for k in ("BTCUSDT_1h", "ETHUSDT_1h", "SOLUSDT_1h")}
        self.assertNotEqual(n["BTCUSDT_1h"], n["SOLUSDT_1h"],
                            "行数已经一样了——这条对齐测试的前提变了，请复核")

    def test_取尾部对齐会得出错误的低相关(self) -> None:
        """⭐ 反面参照：复现原实现的确切错法。

        ⚠️ 关键是**从尾部截断**（``rets[-n:]``），不是从头截断。
        两者的区别很反直觉：
        · 从头截断（取前 n 个）→ 时间戳仍然对齐 → 相关 **0.773**（正确）；
        · 从尾截断（取后 n 个）→ SOL 多出的那**最后一根**bar 把整段序列
          相对 BTC 错开了 1 格 → 相关掉到 **−0.017**。
        第一版这条测试写的是"从头截断"，断言 `|r| < 0.2` 直接失败——
        因为从头截断根本没有 bug。**复现反面案例必须复现它的确切做法**，
        否则测的是另一个东西（这条教训和"测试靠扫现成状态取样"同源）。
        """
        from tw import realdata

        a = realdata.load_builtin("BTCUSDT_1h")
        c = realdata.load_builtin("SOLUSDT_1h")
        ra_all = np.diff(np.log(np.asarray(a.close, dtype=float)))
        rc_all = np.diff(np.log(np.asarray(c.close, dtype=float)))
        n = min(ra_all.size, rc_all.size)
        wrong = float(np.corrcoef(ra_all[-n:], rc_all[-n:])[0, 1])
        self.assertLess(abs(wrong), 0.2,
                        f"尾部对齐竟然给出了 {wrong:.3f}——前提变了")

    def test_按时间戳对齐得到高相关(self) -> None:
        from scripts.run_stage9 import real_correlations

        r = real_correlations()
        for pair, want in (("BTCUSDT_1h|ETHUSDT_1h", 0.82),
                           ("BTCUSDT_1h|SOLUSDT_1h", 0.77),
                           ("ETHUSDT_1h|SOLUSDT_1h", 0.79)):
            got = r["pairs"][pair]
            self.assertAlmostEqual(got, want, delta=0.05,
                                   msg=f"{pair} 对齐后应为 {want}，实测 {got:.3f}")

    def test_对齐信息被写进结果(self) -> None:
        from scripts.run_stage9 import real_correlations

        r = real_correlations()
        self.assertIn("n_rows_per_dataset", r)
        self.assertIn("aligned_by", r)


class TestPairsConfig(unittest.TestCase):
    def test_阈值必须大于平仓阈值(self) -> None:
        with self.assertRaises(ValueError):
            PairsConfig(zthresh=1.0, zexit=1.0)
        with self.assertRaises(ValueError):
            PairsConfig(zthresh=0.5, zexit=1.0)

    def test_窗口与调仓比例校验(self) -> None:
        with self.assertRaises(ValueError):
            PairsConfig(zwindow=5)
        with self.assertRaises(ValueError):
            PairsConfig(adjust_frac=0.0)
        with self.assertRaises(ValueError):
            PairsConfig(adjust_frac=1.5)

    def test_同一资产不能配对(self) -> None:
        c = _cfg(n=2)
        m = MultiAssetMarket(c, n_ticks=50)
        legs = tuple(PairsLeg(f"p{i}", 1e6, 0.0, np.random.default_rng([i]))
                     for i in range(2))
        with self.assertRaises(ValueError):
            PairsTrader((0, 0), m.markets, legs)  # type: ignore[arg-type]


class TestPairsTargetLogic(unittest.TestCase):
    """⭐ 买卖标签不能反（M31）。"""

    def _trader(self, n=2, ticks=100):
        c = _cfg(n=n)
        m = MultiAssetMarket(c, n_ticks=ticks)
        m.run(ticks)
        t = attach_pairs_trader(m, (0, 1))
        return m, t

    def test_多A空B时两腿符号相反(self) -> None:
        m, t = self._trader()
        qa = t.target_qty(0, +1, +1.0)
        qb = t.target_qty(1, +1, -1.0)
        self.assertGreater(qa, 0, "sign=+1（多A空B）时 A 腿应为多头")
        self.assertLess(qb, 0, "sign=+1（多A空B）时 B 腿应为空头")

    def test_空A多B时两腿符号相反(self) -> None:
        m, t = self._trader()
        qa = t.target_qty(0, -1, +1.0)
        qb = t.target_qty(1, -1, -1.0)
        self.assertLess(qa, 0)
        self.assertGreater(qb, 0)

    def test_空仓时目标为零(self) -> None:
        m, t = self._trader()
        self.assertEqual(t.target_qty(0, 0, +1.0), 0.0)
        self.assertEqual(t.target_qty(1, 0, -1.0), 0.0)

    def test_两腿名义金额相同(self) -> None:
        """⭐ 市场中性的前提：两腿名义金额相等，净公共因子暴露 ≈ 0。"""
        m, t = self._trader()
        pa, pb = m.mid(0), m.mid(1)
        qa = t.target_qty(0, +1, +1.0)
        qb = t.target_qty(1, +1, -1.0)
        self.assertAlmostEqual(abs(qa * pa), abs(qb * pb), delta=1e-6 * abs(qa * pa))

    def test_反号会破坏中性(self) -> None:
        """把 B 腿的符号写反（M31 的形态），两腿就从对冲变成双倍同向暴露。"""
        m, t = self._trader()
        correct = t.target_qty(1, +1, -1.0)
        wrong = t.target_qty(1, +1, +1.0)
        self.assertGreater(wrong, 0)
        self.assertLess(correct, 0)

    def _tuned(self, ratio: float, ticks: int = 400):
        """灌一段**有正常波动**的历史，并把"当前比率"钉成指定值。

        ⚠️ 第一版把历史灌成**全 1.0**，那会让样本标准差趋近 0，
        于是任何一个不等的当前值都会得到 |z| 极大——测试测的是
        "除零附近的伪影"，不是"信号方向"。历史必须有真实的变异。
        """
        import math

        m, t = self._trader(ticks=ticks)
        t.ratio_hist.clear()
        # ⚠️ `zscore()` 读的是 `ratio_hist[-1]`，不是 `current_ratio()`——
        # 所以"当前比率"必须**真的进到历史里**（留一格给最后一点）。
        # 同时把 `current_ratio` 也钉住，保证 `update_and_target()` 的 append
        # 不会把最后一点改掉。
        for k in range(t.cfg.zwindow - 1):
            t.ratio_hist.append(1.0 + 0.05 * math.sin(k))
        t.ratio_hist.append(ratio)
        t.current_ratio = lambda: ratio          # type: ignore[method-assign]
        return m, t

    def test_信号方向与z符号一致(self) -> None:
        """z 高 → 比率偏高 → 应当空 A 多 B；z 低 → 反过来。"""
        # 历史 ≈ 1.0 ± 0.035 ⇒ z=+3 对应 ratio≈1.11、z=−3 对应 ≈0.89
        m, t = self._tuned(1.5)
        self.assertGreater(t.zscore() or 0.0, t.cfg.zthresh)
        self.assertEqual(t.update_and_target(), -1,
                         "比率偏高时应当做空 A / 做多 B")

        m, t = self._tuned(0.5)
        self.assertLess(t.zscore() or 0.0, -t.cfg.zthresh)
        self.assertEqual(t.update_and_target(), +1,
                         "比率偏低时应当做多 A / 做空 B")

    def test_z不足时不动作(self) -> None:
        m, t = self._tuned(1.0)          # 正好在均值上 → z≈0
        # 正弦历史样本的均值不精确等于 1.0，所以 z 是个小量而不是精确 0；
        # 判据是"落在死区内"，不是"等于 0"。
        self.assertLess(abs(t.zscore() or 9.0), 0.1)
        self.assertEqual(t.update_and_target(), 0)
        self.assertEqual(t.n_signals, 0)

    def test_平仓阈值内的既有仓位会被平掉(self) -> None:
        m, t = self._tuned(1.0)
        t.position_sign = +1          # 假装已经有多 A / 空 B 的仓位
        t.update_and_target()
        self.assertEqual(t.position_sign, 0, "|z| 回到平仓阈值内却没有平仓")


class TestPairsExecution(unittest.TestCase):
    def test_两条腿在同一tick内提交(self) -> None:
        c = _cfg(n=2)
        m = MultiAssetMarket(c, n_ticks=400)
        m.run(400)
        t = attach_pairs_trader(m, (0, 1), cfg=PairsConfig(zwindow=20, zthresh=0.5,
                                                          zexit=0.1))
        inv_before = (t.legs[0].inventory, t.legs[1].inventory)
        t.position_sign = +1
        res = t.rebalance(+1)
        self.assertEqual(len(res), 2, "只提交了一条腿——两腿必须在同一 tick 内")
        self.assertNotEqual((t.legs[0].inventory, t.legs[1].inventory), inv_before)

    def test_两腿的实际持仓方向与组合方向一致(self) -> None:
        """⭐ 这条是被变异体 M33 **逼出来的**。

        原来只测到"持仓变了"和 ``target_qty`` 单独调用的符号——
        而 M33 反的是 ``rebalance`` 里的**调用点**映射
        ``((A腿, +1), (B腿, −1))`` → ``((A腿, −1), (B腿, +1))``。
        ``target_qty`` 本身没动，所以那几条测试全绿；
        "持仓变了"也照样成立。**换手方向反了却没人发现。**

        这正是变异测试存在的意义：它不问你"测了没有"，只问
        "把这一行改错，测试会不会红"。所以这里直接断言**实际持仓的符号**。
        """
        c = _cfg(n=2)
        m = MultiAssetMarket(c, n_ticks=400)
        m.run(400)
        t = attach_pairs_trader(m, (0, 1), cfg=PairsConfig(zwindow=20, zthresh=0.5,
                                                          zexit=0.1))
        # 组合方向 +1 = 多 A / 空 B
        t.position_sign = +1
        t.rebalance(+1)
        self.assertGreater(
            float(t.legs[0].inventory), 0.0,
            "组合方向 +1 时 A 腿必须建成多头（否则两腿从对冲变成双倍同向暴露）")
        self.assertLess(float(t.legs[1].inventory), 0.0,
                        "组合方向 +1 时 B 腿必须建成空头")

        # 反向：组合方向 −1 = 空 A / 多 B。
        # ⚠️ 必须用**新的、从零仓位的**交易者：``adjust_frac=0.25``
        # 让调仓是渐进的（见 ``test_调仓是渐进的``），从"已做多 41 手"
        # 调一次到 −1 只会走到 26 手，**不会**翻成负数。
        # 从零起步时 ``delta = target``，一次调仓的符号就等于目标符号。
        t2 = self._flat_trader()
        t2.position_sign = -1
        t2.rebalance(-1)
        self.assertLess(float(t2.legs[0].inventory), 0.0, "组合方向 −1 时 A 腿应为空头")
        self.assertGreater(float(t2.legs[1].inventory), 0.0, "组合方向 −1 时 B 腿应为多头")

    def _flat_trader(self):
        """造一个**从零仓位起步**的配对交易者（每个方向各用一个）。"""
        c = _cfg(n=2)
        m = MultiAssetMarket(c, n_ticks=400)
        m.run(400)
        return attach_pairs_trader(m, (0, 1),
                                   cfg=PairsConfig(zwindow=20, zthresh=0.5,
                                                   zexit=0.1))

    def test_两条腿的做空额度不是零(self) -> None:
        """⭐ 这条守的是一个**曾经真实存在**的结构性缺陷。

        ``Agent.short_limit`` 默认 0，而 ``Market._clamp`` 卖出时
        ``cap = available_inventory + short_limit``，``cap <= MIN_ORDER_QTY``
        就 ``return None``——订单被**静默丢弃**，不报错、不留痕。
        于是"空 B"那条腿从零库存永远建不了仓，
        配对交易在开仓时退化成"只做多 A 一条腿"，
        而它的文档写着"两腿名义金额相同 ⇒ 净公共因子暴露 ≈ 0"。

        症状极隐蔽：成交数 0、持仓 0、现金 0，任何"守恒"类测试都照过。
        所以这里直接断言**额度存在且够用**，而不是断言"跑起来没报错"。
        """
        c = _cfg(n=2)
        m = MultiAssetMarket(c, n_ticks=200)
        m.run(200)
        t = attach_pairs_trader(m, (0, 1), cfg=PairsConfig(zwindow=20))
        for idx, leg in enumerate(t.legs):
            need = abs(t.target_qty(idx, +1, +1.0 if idx == 0 else -1.0))
            self.assertGreater(leg.short_limit, 0.0,
                               f"第 {idx} 条腿的做空额度是 0 —— 它永远建不了空头仓")
            self.assertGreaterEqual(
                leg.short_limit, need,
                f"第 {idx} 条腿的做空额度 {leg.short_limit:.2f} 小于目标仓位 "
                f"{need:.2f}，空头腿会被裁剪到够不着的量")

    def test_空头腿真的能建成负持仓(self) -> None:
        """端到端地确认：从零库存出发，"空 B"确实能变成负持仓。"""
        c = _cfg(n=2)
        m = MultiAssetMarket(c, n_ticks=400)
        m.run(400)
        t = attach_pairs_trader(m, (0, 1), cfg=PairsConfig(zwindow=20, zthresh=0.5,
                                                          zexit=0.1))
        t.position_sign = +1
        t.rebalance(+1)
        self.assertLess(float(t.legs[1].inventory), 0.0,
                        "B 腿没有建成空头——做空额度或是被裁剪逻辑吃掉了")

    def test_调仓是渐进的(self) -> None:
        """一步到位会让换手率爆掉，测出来的 PnL 里执行成本淹掉信号。"""
        c = _cfg(n=2)
        m = MultiAssetMarket(c, n_ticks=400)
        m.run(400)
        t = attach_pairs_trader(m, (0, 1), cfg=PairsConfig(zwindow=20,
                                                          adjust_frac=0.25))
        t.position_sign = +1
        t.rebalance(+1)
        target = abs(t.target_qty(0, +1, +1.0))
        self.assertLess(abs(float(t.legs[0].inventory)), target + 1e-9)

    def test_账户守恒不被破坏(self) -> None:
        c = _cfg(n=2)
        m = MultiAssetMarket(c, n_ticks=800)
        t = attach_pairs_trader(m, (0, 1), cfg=PairsConfig(zwindow=30, zthresh=0.8,
                                                          zexit=0.2))
        for _ in range(500):
            m.step()
            t.step()
        ok, probs = m.health_check()
        self.assertTrue(ok, probs[:5])

    def test_腿走的是唯一记账路径(self) -> None:
        """两腿必须是真主体（有账户），而不是"直接改持仓"。

        直接改会绕过预留制与账本，一条腿的盈亏凭空出现，
        而 `cash_conservation` 会报出一个看起来像浮点噪声的残差。
        """
        c = _cfg(n=2)
        m = MultiAssetMarket(c, n_ticks=300)
        m.run(300)
        t = attach_pairs_trader(m, (0, 1), cfg=PairsConfig(zwindow=20, zthresh=0.5,
                                                          zexit=0.1))
        for leg in t.legs:
            self.assertIsInstance(leg, PairsLeg)
        for i, leg in enumerate(t.legs):
            self.assertIn(leg.agent_id, m.markets[i].by_id)
        # 跑一段并核对现金守恒
        for _ in range(200):
            m.step()
            t.step()
        for mk in m.markets:
            ok, resid, msg = mk.cash_conservation()
            self.assertTrue(ok, msg)

    def test_归因字段齐全(self) -> None:
        c = _cfg(n=2)
        m = MultiAssetMarket(c, n_ticks=200)
        m.run(200)
        t = attach_pairs_trader(m, (0, 1))
        d = t.describe()
        for k in ("position_sign", "n_signals", "n_rebalances",
                  "leg_a_qty", "leg_b_qty"):
            self.assertIn(k, d)


    def test_组合的净暴露远小于总暴露(self) -> None:
        """⭐ 这才是"配对交易"的定义性质：**净暴露 ≈ 0**。

        上面几条测的是"每条腿方向对不对"。但真正定义配对交易的是
        **两腿合起来的净暴露要接近 0**——否则它就是一个方向性策略，
        只是被叫做"配对"。

        这条断言在修复前**必然失败**：那时 ``short_limit=0``，
        空头腿的卖单被 ``Market._clamp`` 静默丢弃，
        组合在开仓期只建成了一条腿 → 净暴露 ≈ 总暴露。
        所以它同时是"缺陷回归测试"和"策略定义检查"。

        口径：``净 / 总``（都用名义金额），阈值取 **0.1**。
        实测：修复前 = **1.000**（完全单边），修复后 = 0.003~0.026（推进 20~150 tick）。
        阈值放在两者中间，既不是卡着实测值，也远离"单边"那一端。

        ⚠️ 必须**让市场推进**（``m.step()``）再调仓。第一版在同一个 tick 里
        连调 8 次，第二次之后盘口没恢复、市价单吃不到量，
        测出来是"卡在 0.145 就不动"——那是测试的假象，不是策略的问题。
        """
        c = _cfg(n=2)
        m = MultiAssetMarket(c, n_ticks=600)
        m.run(600)
        t = attach_pairs_trader(m, (0, 1), cfg=PairsConfig(zwindow=20, zthresh=0.5,
                                                          zexit=0.1))
        t.position_sign = +1
        for _ in range(60):              # 每个 tick 推进一次市场，让盘口恢复
            m.step()
            t.rebalance(+1)
        na = float(t.legs[0].inventory) * m.mid(0)
        nb = float(t.legs[1].inventory) * m.mid(1)
        gross = abs(na) + abs(nb)
        net = abs(na + nb)
        self.assertGreater(gross, 0.0, "两腿都没建仓，无从判断暴露")
        self.assertLess(
            net / gross, 0.1,
            f"净暴露/总暴露 = {net / gross:.3f}（净 {net:,.0f} / 总 {gross:,.0f}）"
            f"—— 两腿没有对冲起来，这不是配对交易，是方向性策略")


if __name__ == "__main__":
    unittest.main()
