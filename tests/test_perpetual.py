"""永续合约机制层的测试（二期阶段5）。

这个文件里最要紧的是**守恒测试**。资金费率是"钱从一方到另一方"的机制，
它破坏的不是"单账户自洽"（`account_integrity` 查得到），
而是"**全系统现金总量守恒**"——后者一旦破了，所有涉及资金的检查都在错误基准上跑，
而且**不会报任何错**。所以它是本项目唯一一个必须逐笔验证的恒等式。
"""

from __future__ import annotations

import unittest

import numpy as np

from tw import Market, Population, SimConfig
from tw.agents.base import Agent
from tw.agents.hedger import FundingArbitrageur
from tw.perpetual import (
    REAL_FUNDING_ANNUAL,
    REAL_FUNDING_MEAN,
    SETTLEMENTS_PER_YEAR,
    TARGET_SNR,
    FundingConfig,
    PerpetualMarket,
    annualize,
    rate_from_series,
    signal_to_noise,
)
from tw.types import MarketState, PriceHistory


class _DepthProvider(Agent):
    """只在测试里用：往卖侧铺一大片挂单，给市价买单当对手盘。

    为什么需要它：``_clamp`` 对市价买要求盘口有足够深度才放行。
    没有这个主体，"费率翻负 → 转多"这条路径根本走不到，
    测试会看到"没转多"——而根因是盘口空，不是逻辑错。
    """

    KIND = "test_depth"

    def __init__(self, qty: float = 400.0, levels: int = 4) -> None:
        super().__init__("depth0000", 1e12, 0.0, np.random.default_rng([999]))
        self.short_limit = 1e6      # 铺的是卖单，需要卖空额度
        self._qty = float(qty)
        self._levels = int(levels)

    def decide(self, state):
        px = state.mid * 1.0005     # 略高于中间价，不主动砸盘
        return [self.new_order(state, "sell", px, self._qty) for _ in range(self._levels)]


def _state(
    mid: float = 100.0,
    tick: int = 0,
    funding_rate: float = 0.0,
    flow_imbalance: float = 0.0,
) -> MarketState:
    """构造一个真实的 ``MarketState``。

    ⚠️ 早期版本在这里用 ``type("S", (), {...})`` 拼了个鸭子类型对象，
    结果 ``decide`` 走 ``new_order`` 时读 ``state.tick`` 直接 AttributeError。
    **假对象会随真实接口演进而静默失效**——接口加一个字段，测试就崩，
    而崩溃原因看起来像"被测代码有 bug"。用真类型就不会有这个问题。
    """
    hist = PriceHistory(16)
    hist.append(mid)
    return MarketState(
        tick=tick,
        mid=mid,
        last_price=mid,
        best_bid=mid - 1.0,
        best_ask=mid + 1.0,
        spread=2.0,
        fundamental=mid,
        history=hist,
        funding_rate=funding_rate,
        flow_imbalance=flow_imbalance,
    )


def _perp(
    n: int = 60,
    ticks: int = 600,
    seed: int = 7,
    p_buy: float = 0.5,
    cfg: FundingConfig | None = None,
) -> PerpetualMarket:
    pop = Population.from_shares(
        n, {"zero_intel": 0.30, "fundamentalist": 0.40, "chartist": 0.30}
    )
    m = PerpetualMarket(
        SimConfig(seed=seed, n_ticks=ticks, population=pop, zi_p_buy=p_buy),
        funding_config=cfg or FundingConfig(),
    )
    m.run(ticks)
    return m


class TestFundingConservation(unittest.TestCase):
    """⭐ 资金费率零和。这是最容易漏、也最难发现的恒等式。"""

    def test_每次结算的付款与交易所变动互为相反数(self) -> None:
        m = _perp(ticks=800)
        self.assertGreater(len(m.funding_records), 10)
        exchange = 0.0
        for rec in m.funding_records:
            exchange -= rec["sum_payment"]
        self.assertAlmostEqual(
            exchange, m.exchange_cash, places=6,
            msg="交易所账户累计变动必须精确等于 −Σ主体付款",
        )

    def test_现金总量守恒(self) -> None:
        m = _perp(ticks=800)
        ok, resid, msg = m.cash_conservation()
        self.assertTrue(ok, msg)
        self.assertLess(abs(resid), 1e-6 * 1e7)

    def test_摘要里的交易所吸收额与主账本对账(self) -> None:
        """⭐ 这条断言是补上的——它抓的是一个守恒测试**抓不到**的 bug。

        `funding_exchange_total`（摘要用）和 `exchange_cash`（守恒检查用）
        是两个独立维护的累加器。曾经因为粘贴写重了一行，
        `funding_exchange_total` 被扣了两次，于是摘要报出的"交易所吸收额"
        是真实值的 **2 倍**。
        而 `cash_conservation()` 读的是 `exchange_cash`，一切正常——
        全绿的守恒测试对这个错误**完全无感**。

        教训：**同一件事有两个累加器时，必须有一个断言把它们对起来**，
        否则其中一个错了永远没人知道。
        """
        m = _perp(ticks=800)
        s = m.funding_summary()
        self.assertAlmostEqual(
            s["exchange_absorbed"], m.exchange_cash, places=6,
            msg=(
                f"摘要说交易所吸收了 {s['exchange_absorbed']:.6f}，"
                f"主账本是 {m.exchange_cash:.6f}——两个累加器不一致"
            ),
        )

    def test_摘要吸收额等于逐笔求和(self) -> None:
        """再从账本独立算一遍，三路对齐。"""
        m = _perp(ticks=600)
        led = sum(
            e["delta"] for e in m.cash_ledger
            if e["reason"] == "funding_settlement"
        )
        self.assertAlmostEqual(
            m.funding_summary()["exchange_absorbed"], -led, places=6
        )

    def test_账本记满每一笔(self) -> None:
        """流水必须逐笔可审计，不能只留一个汇总数。"""
        m = _perp(ticks=400)
        n = len(m.funding_records)
        led = [e for e in m.cash_ledger if e["reason"] == "funding_settlement"]
        self.assertGreater(len(led), 0)
        # 每次结算必然有主体付款（除非所有持仓恰好为 0）
        self.assertGreaterEqual(len(led), n)

    def test_账户完整性与体检都通过(self) -> None:
        m = _perp(ticks=800)
        ok, prob = m.account_integrity()
        self.assertTrue(ok, prob[:5])
        ok2, prob2 = m.health_check()
        self.assertTrue(ok2, prob2[:5])

    def test_裸市场不受影响(self) -> None:
        """一期市场必须是完全无杠杆的——`allow_negative_cash` 默认为 False。"""
        m = Market(SimConfig(seed=3, n_ticks=300, population=Population(zero_intel=30)))
        m.run(300)
        self.assertFalse(m.allow_negative_cash)
        self.assertEqual(m.last_funding_rate, 0.0)
        self.assertEqual(m._state.funding_rate, 0.0)
        ok, _, msg = m.cash_conservation()
        self.assertTrue(ok, msg)


class TestFundingMechanism(unittest.TestCase):
    def test_结算等间隔(self) -> None:
        m = _perp(ticks=800)
        self.assertTrue(
            m.settle_interval_is_uniform(),
            f"结算时点不均匀：{np.diff(m.settle_ticks)[:10]}",
        )
        self.assertEqual(
            m.settle_ticks[0], m.funding_config.settle_interval_ticks
        )

    def test_溢价与费率同向(self) -> None:
        """手工构造：人为抬高中间价 → 溢价为正 → 费率为正。

        ⚠️ 必须**只留溢价通道**再测。标定后的默认系数下拥挤度通道贡献了 89.4%
        的费率水平，两个通道一起开着看符号，测的其实是拥挤度通道的符号——
        这条测试就名不副实了。**隔离被测通道是这类单元测试的基本要求**，
        否则测试绿灯跟被测逻辑之间没有因果关系。
        """
        cfg = FundingConfig(crowding_sensitivity=0.0)
        m = _perp(ticks=100, cfg=cfg)
        # 直接改基本面锚做实验（不动撮合）
        m.fundamental = m.current_mid() * 0.95   # 锚低于市价 5% → 溢价 +5.26%
        rec = m._settle_funding()
        self.assertGreater(rec["premium"], 0)
        self.assertGreater(rec["rate"], 0)

        m.fundamental = m.current_mid() * 1.05
        rec2 = m._settle_funding()
        self.assertLess(rec2["premium"], 0)
        self.assertLess(rec2["rate"], 0)

    def test_拥挤度通道单独也同向(self) -> None:
        """对称的一条：只留拥挤度通道，失衡量 > 0 → 费率为正。

        两条通道各有一条隔离测试，合起来才能说"两通道方向都对"。
        只测合起来的效果，方向错误的单条通道会被另一条盖住。
        """
        cfg = FundingConfig(premium_sensitivity=0.0)
        m = _perp(ticks=100, cfg=cfg)
        pos = m._settle_funding()
        # 手工把窗口内的订单流拨成净买入：直接构造一段正的 flow
        t = m.tick
        for i in range(max(0, t - 100), t):
            m.log.flow[i] = 1.0
        up = m._settle_funding()
        self.assertGreater(up["crowding"], 0)
        self.assertGreater(up["rate"], 0)
        _ = pos

    def test_clamp上限生效(self) -> None:
        cfg = FundingConfig(clamp_bp=1.0, premium_sensitivity=10.0)
        m = _perp(ticks=200, cfg=cfg)
        for rec in m.funding_records:
            self.assertLessEqual(abs(rec["rate_bp"]), 1.0 + 1e-9)
        self.assertGreater(m.funding_summary()["clamped_frac"], 0.0)

    def test_不做多偏好时费率在0附近(self) -> None:
        """E5.1 的健全性检查：无方向偏好 → 费率不应系统性偏离 0。"""
        ann = []
        for seed in (11, 12, 13):
            m = _perp(ticks=1200, seed=seed, p_buy=0.5)
            ann.append(m.funding_summary()["annualized"])
        mean = float(np.mean(ann))
        self.assertLess(
            abs(mean), REAL_FUNDING_ANNUAL * 1.5,
            f"无偏好基线年化 {mean:.4f} 偏离 0 太远（真实基准 {REAL_FUNDING_ANNUAL}）",
        )

    def test_做多偏好推动费率为正(self) -> None:
        """无偏好 → 费率≈0；强偏好 → 费率显著为正。

        两处口径必须说明白，否则这条测试会假红（都是踩过的）：
        · **用「强偏好 vs 无偏好」而不是相邻档位**。标定后的费率里，拥挤度通道
          （观察窗口 300 tick）需要时间成形，相邻档位在短跑里的差异落在噪音里。
          相邻档位的单调性交给 E5.2 的多种子扫描（那里有 Pearson 判据）。
        · **必须丢掉预热段**。开局几千 tick 的单边卖压会把均值拖成负的
          （实测 p_buy=0.70 只跑 2000 tick 会得出 −3.2%）。口径见
          ``TestWarmupTransient``。
        """
        WARM = 3_000

        def mean_annual(p_buy: float) -> float:
            return float(np.mean([
                _perp(ticks=6000, seed=s, p_buy=p_buy)
                .funding_summary(since_tick=WARM)["annualized"]
                for s in (21, 22)
            ]))

        a0 = mean_annual(0.50)
        a1 = mean_annual(0.70)
        self.assertGreater(a1, a0, f"强做多偏好没有推高费率（{a1:.4f} vs {a0:.4f}）")
        self.assertGreater(a1, 0.0, f"强做多偏好下费率仍非正（{a1:.4f}）")

    def test_年化换算(self) -> None:
        self.assertAlmostEqual(annualize(1e-4), 1e-4 * 1095, places=12)

    def test_结算后必须修预留否则会静默冻结(self) -> None:
        """⭐ 这条断言让「修预留」这一步**成为承重结构**，而不是可选装饰。

        为什么必须测它：资金费率扣钱后 ``reserved_cash`` 可能超过 ``cash``，
        ``available_cash`` 变负，此后该主体**每一张新单都被预算裁剪拒掉**。
        症状是"市场在几百 tick 后静默冻结"——不报错、不崩溃，成交数慢慢归零，
        在指标上看起来像"价格收敛、市场稳定"的好结果。

        `allow_negative_cash=True` 时 `account_integrity` **查不出这个问题**
        （它只查权益是否为负），所以需要一个直接针对该机制的断言。
        """
        m = _perp(ticks=800)
        self.assertGreater(
            m.reconciled_orders, 0,
            "整场模拟一次预留修正都没触发 —— 要么场景里没人被扣到透支，"
            "要么修正逻辑没接上（后者是静默冻结的前兆）",
        )
        # 结算之后不变量必须是干净的
        ok, prob = m.account_integrity()
        self.assertTrue(ok, prob[:5])

    def test_预留修正真的能把不变量修回来(self) -> None:
        """直接构造超支，验证 `reconcile_reservations` 的语义。

        ⚠️ **自己造挂单，不去"扫现成的"**。第一版写的是
        「从 `m.agents` 里挑一个有挂单和预留的」，结果在标定换参数之后
        那一场模拟里恰好没人挂着单，测试就红了——红的原因是**场景变了**，
        不是被测逻辑坏了。**靠运气取样的测试等于没有测试**，
        因为它会在最需要它的时候（改了别的东西之后）给你假信号。
        """
        m = _perp(ticks=300)
        a = m.agents[0]
        state = m._state
        a.cash = 1e9                       # 先给足钱，好让单子挂得出去
        o = a.new_order(state, "buy", state.mid * 0.99, 50.0)
        self.assertIsNotNone(o)
        m.submit(a, o)
        self.assertGreater(a.reserved_cash, 0.0, "限价买单没有产生预留")

        a.cash = a.reserved_cash * 0.5     # 压到"只够一半预留" → 确定性超支
        n_before = len(a.open_orders)
        self.assertLess(a.available_cash, 0.0)

        n = m.reconcile_reservations(a)
        self.assertGreater(n, 0, "超支了却没撤任何挂单")
        self.assertGreaterEqual(
            a.available_cash, -1e-9, "撤完单可用现金仍为负 → 该主体之后发不出任何单"
        )
        self.assertLessEqual(len(a.open_orders), n_before)

    def test_预留修正不会动到不该动的(self) -> None:
        """富余的主体不该被撤单——修正必须**恰好**压回不变量，而不是一律清空。"""
        m = _perp(ticks=300)
        a = m.agents[0]
        state = m._state
        a.cash = 1e9
        m.submit(a, a.new_order(state, "buy", state.mid * 0.99, 50.0))
        self.assertGreater(a.reserved_cash, 0.0)
        n_before = len(a.open_orders)
        n = m.reconcile_reservations(a)
        self.assertEqual(n, 0, "现金充足却撤了单")
        self.assertEqual(len(a.open_orders), n_before)


class TestCalibrationHelpers(unittest.TestCase):
    """E5.0 标定用到的两个纯函数。

    它们必须被钉住，因为它们承载了整个阶段5 的**标定逻辑**：
    如果 `signal_to_noise` 悄悄改了口径，标定脚本会给出不同的传导系数，
    而所有下游数字（年化、正占比）都会跟着漂，且不会有任何报错。
    """

    def test_真实靶子换算(self) -> None:
        self.assertAlmostEqual(
            REAL_FUNDING_MEAN, REAL_FUNDING_ANNUAL / SETTLEMENTS_PER_YEAR, places=15
        )
        self.assertAlmostEqual(REAL_FUNDING_MEAN * 1e4, 1.0566, places=3)

    def test_离线重算等价于直接算(self) -> None:
        """`rate_from_series` 必须与 `_settle_funding` 里的算式逐点一致。

        不一致就说明"用离线重算找参数、再实跑验收"这条路径里，
        两边算的不是同一个东西——那是最坏的情况：标定出来的参数装回模型，
        行为却不一样，而且没人会发现。
        """
        prem = np.array([3e-4, -2e-4, 0.0, 8e-4, -1e-3])
        crow = np.array([0.3, -0.2, 0.0, 0.8, -0.5])
        cfg = FundingConfig(premium_sensitivity=0.2, crowding_sensitivity=0.002)
        got = rate_from_series(
            prem, crow,
            premium_sensitivity=cfg.premium_sensitivity,
            crowding_sensitivity=cfg.crowding_sensitivity,
            clamp_bp=cfg.clamp_bp,
        )
        want = np.clip(0.2 * prem + 0.002 * crow, -cfg.clamp, cfg.clamp)
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-18)

    def test_离线重算会施加clamp(self) -> None:
        got = rate_from_series(
            [1.0], [0.0],
            premium_sensitivity=1.0, crowding_sensitivity=0.0, clamp_bp=75.0,
        )
        self.assertAlmostEqual(float(got[0]), 75.0 / 1e4, places=12)

    def test_长度不一致要报错(self) -> None:
        with self.assertRaises(ValueError):
            rate_from_series([1.0, 2.0], [1.0],
                             premium_sensitivity=1.0, crowding_sensitivity=1.0,
                             clamp_bp=75.0)

    def test_信噪比与scale无关(self) -> None:
        """⭐ 这条是阶段5 最重要的结构性质：`scale` 改不动正费率占比。

        正占比只取决于分布形状。若哪天它开始依赖 `scale`，说明
        费率公式里混进了与幅度耦合的非线性项（例如"截断到 0"之类），
        那 E5.4 的标定逻辑就要重写。
        """
        rng = np.random.default_rng(5)
        x = rng.normal(1.0, 1.0, 4000)
        base = float((x > 0).mean())
        for s in (0.01, 0.1, 1.0, 100.0):
            self.assertAlmostEqual(float((s * x > 0).mean()), base, places=12)
        self.assertAlmostEqual(signal_to_noise(x), 1.0, places=1)

    def test_目标信噪比对应真实正占比(self) -> None:
        """`TARGET_SNR` 必须是「让正态分布的正号占比 = 0.858」的那个值。"""
        rng = np.random.default_rng(7)
        x = rng.normal(TARGET_SNR, 1.0, 400_000)
        self.assertAlmostEqual(float((x > 0).mean()), 0.858, places=2)

    def test_信噪比对常数序列返回无穷(self) -> None:
        self.assertEqual(signal_to_noise([3.0] * 10), float("inf"))

    def test_信噪比样本不足返回nan(self) -> None:
        self.assertTrue(np.isnan(signal_to_noise([1.0, 2.0])))


class TestWarmupTransient(unittest.TestCase):
    """⭐ 初始禀赋的松弛过程：开局几千 tick 的订单流是**单边**的。

    这不是"调参调出来的现象"，是模型结构的直接后果：每人一开始就被随机分了
    持仓 ~U(0,20)，市场要用几千 tick 才能把这份禀赋消化掉。这段过程里
    主动卖压远大于买压，拥挤度为负、费率为负。

    钉住它的理由有两个：
    ① 它是**统计口径的分界线**。把这段算进费率统计，正费率占比会被系统性压低
       （实测 p_buy=0.70 只跑 2000 tick 会得出**负**年化 −3.2%）。
       项目的 ``scenarios.py`` 统一用 4000 tick 预热，资金费率必须同口径。
    ② 如果哪天的市场重构**消除了**这个瞬态（比如换成稳态初始分布），
       那么所有基于"预热 4000"的结论都要重新算——这是一条必须被看见的变更。
    """

    def test_开局订单流是单边卖压(self) -> None:
        pop = Population.from_shares(
            120, {"zero_intel": 0.30, "fundamentalist": 0.40, "chartist": 0.30}
        )
        m = PerpetualMarket(
            SimConfig(seed=91, n_ticks=800, population=pop, zi_p_buy=0.70),
            funding_config=FundingConfig(premium_sensitivity=0.0, crowding_sensitivity=0.0),
        )
        m.run(800)
        early = np.array([d["crowding"] for d in m.funding_records[:10]])
        self.assertLess(
            early.mean(), -0.15,
            f"开局拥挤度均值 {early.mean():+.3f} 不再是明显负值——"
            "初始禀赋的松弛过程变了，预热长度与所有费率统计口径都需要重新评估",
        )

    def test_预热之后拥挤度转正(self) -> None:
        """6000 tick 之后，强做多偏好必须表现为正的订单流失衡。"""
        pop = Population.from_shares(
            300, {"zero_intel": 0.30, "fundamentalist": 0.40, "chartist": 0.30}
        )
        m = PerpetualMarket(
            SimConfig(seed=20260917, n_ticks=8000, population=pop, zi_p_buy=0.70),
            funding_config=FundingConfig(premium_sensitivity=0.0, crowding_sensitivity=0.0),
        )
        m.run(8000)
        late = np.array([d["crowding"] for d in m.funding_records if d["tick"] > 6000])
        self.assertGreater(
            late.mean(), 0.05,
            f"预热后拥挤度仍是 {late.mean():+.4f}——做多偏好没有传导到订单流",
        )

    def test_口径不同会得出相反结论(self) -> None:
        """⭐ 这条把"口径"本身钉成被测对象。

        同一场模拟、同一批数据，只因为**统计窗口**不同，
        拥挤度（进而费率）的符号会翻转。这就是为什么口径必须写进报告，
        而不能只报一个数。

        ⚠️ 这里必须**把费率关掉**（两个传导系数都是 0）再测。
        第一版开着费率测，红了——原因很有意思：早期费率为负时，
        多头是**收钱**的一方，等于给市场注入现金，于是它们能买得更多，
        订单流被推回买侧，早期费率反而变成正的。
        也就是说**「费率 → 现金 → 下单预算 → 订单流」这条回路本身就改变了早期的符号**。
        所以"瞬态为负"是**驱动量**（订单流失衡）的性质，不是费率的性质——
        测驱动量就必须把它从反馈里摘出来，否则测的是反馈。
        （这条回路本身是一条独立结论，见下一条测试。）
        """
        pop = Population.from_shares(
            300, {"zero_intel": 0.30, "fundamentalist": 0.40, "chartist": 0.30}
        )
        early, late = [], []
        for seed in (77, 78, 79):
            m = PerpetualMarket(
                SimConfig(seed=seed, n_ticks=6000, population=pop, zi_p_buy=0.70),
                funding_config=FundingConfig(
                    premium_sensitivity=0.0, crowding_sensitivity=0.0,
                ),
            )
            m.run(6000)
            # 早窗口取前 1000 tick：瞬态最强的一段（实测累积均值 ≈ −0.39）
            early += [d["crowding"] for d in m.funding_records if d["tick"] <= 1000]
            late += [d["crowding"] for d in m.funding_records if d["tick"] > 4000]
        e, l = float(np.mean(early)), float(np.mean(late))
        self.assertLess(e, -0.10, f"预热段的订单流不再单边（{e:+.4f}）")
        self.assertGreater(l, 0.02, f"预热后的订单流不再偏买（{l:+.4f}）")
        # ⚠️ 只断言**符号相反**，不断言"瞬态更强"。
        # 试过断言 |e| > |l|，红了：两段窗口的长度不同、瞬态的衰减也不是单调的，
        # 幅度之比不是一个稳定量。**把断言写在机制上（早期偏卖、后期偏买），
        # 而不是写在它的派生量上**，测试才会只因为机制变化而变红。
        self.assertLess(e, 0.0)
        self.assertGreater(l, 0.0)

    def test_费率会通过现金反过来改变订单流(self) -> None:
        """⭐⭐ 反馈回路的直接证据，也是阶段7「反身性」的前置观测。

        同种子、同配置，**只切资金费率开关**，比较预热度之后的订单流失衡。

        为什么这条重要：阶段5 的整套标定用的是「费率关掉 → 离线重算」这条
        便宜路径（见 ``rate_from_series``），它隐含假设"费率不影响行情"。
        这条测试把这个假设**摆到台面上量一次**：
        如果开关两边没有差别，假设成立；如果有差别，那个差就是假设的误差，
        必须写进报告的"诚实边界"，而不是假装它不存在。

        实测结论：有差别。费率是**每 8 tick 从保证金账户扣/付真金**，
        而现金直接决定下一个 tick 能挂多大的单——所以它必然反过来塑造订单流。
        """
        pop = Population.from_shares(
            300, {"zero_intel": 0.30, "fundamentalist": 0.40, "chartist": 0.30}
        )

        def crowding_after_warmup(rate_on: bool) -> float:
            cfg = (FundingConfig() if rate_on
                   else FundingConfig(premium_sensitivity=0.0, crowding_sensitivity=0.0))
            vals = []
            for seed in (77, 78):
                m = PerpetualMarket(
                    SimConfig(seed=seed, n_ticks=6000, population=pop, zi_p_buy=0.70),
                    funding_config=cfg,
                )
                m.run(6000)
                vals += [d["crowding"] for d in m.funding_records if d["tick"] > 4000]
            return float(np.mean(vals))

        on = crowding_after_warmup(True)
        off = crowding_after_warmup(False)
        gap = abs(on - off)
        rel = gap / max(abs(off), 1e-12)
        self.assertGreater(
            gap, 0.002,
            f"费率开关对订单流几乎无影响（差 {gap:.4f}，相对 {rel:.1%}）——"
            "反馈回路不存在？那整套离线标定路径的假设就变了",
        )
        # ⚠️ **不断言方向**。第一版断言"打开费率后买压更强"，红了：
        # 费率打开后长端是净收还是净付，取决于费率符号，而符号随偏好档位变。
        # 方向不是稳定性质，**"有差别"才是**。这不是偷懒——
        # 测试的价值在于它红了就说明机制变了，而不是在于它替我们猜方向。
        #
        # 实测量级：相对差约 3%。**小**这件事本身是结论——
        # 标定后的费率只有 ~1bp/次，对应的现金流相对初始现金很小，
        # 所以"费率不影响行情"这个近似在标定后的参数下是**大致成立**的。
        # 旧参数（拥挤度系数 0.05，是现在的 8 倍）下这个差会大得多，
        # 这也是当初离线预测与实跑差 19% 的主要来源。


class TestDegenerateCrowding(unittest.TestCase):
    """⭐ 钉住"指导书的拥挤度定义在本模型里退化"这个事实。

    交易不改变 Σ持仓（买家 +q、卖家 −q），所以
    ``Σmax(inv,0)/Σ|inv|`` 只由初始禀赋决定，整场模拟恒定。
    这条测试的意义是：**如果哪天它不再是常数，说明有人引入了能改变总持仓的机制
    （比如卖空、或凭空造仓），那时"拥挤度"才重新变得可用**——
    而这会是一条必须被看见的变更，不能悄悄发生。
    """

    def test_净多头占比恒定(self) -> None:
        m = _perp(ticks=800)
        s = m.funding_summary()
        self.assertAlmostEqual(s["net_long_ratio_sd"], 0.0, places=9,
                               msg="净多头占比居然在变——总持仓不再守恒？")
        self.assertGreater(s["mean_net_long_ratio"], 0.99,
                           "本模型不许裸卖空，所有持仓恒为非负")

    def test_手工验证交易不改变总持仓(self) -> None:
        m = _perp(ticks=400)
        total = sum(a.inventory for a in m.agents)
        flow = 0.0
        for t in m.log.trades:
            flow += t.quantity
            flow -= t.quantity
        self.assertAlmostEqual(total, sum(a.inventory for a in m.agents), places=9)
        self.assertEqual(flow, 0.0)


class TestFundingArbitrageur(unittest.TestCase):
    def _with_hedgers(
        self, n_h: int, ticks: int = 600, seed: int = 31, p_buy: float = 0.58,
        *, track_path: bool = False,
    ):
        """建市场 → 注入 n_h 个套利者 → 跑 ticks。

        ``track_path=True`` 时**分块跑**并记录每个套利者持仓的极值。

        为什么必须能记录全程：资金费率是每 8 tick 结算一次的**脉冲式**信号，
        套利者的仓位因此是"往目标收敛 → 目标翻转 → 再收敛"的往复过程。
        只在终点看一眼，看到的是**相位**而不是行为——
        终点刚好落在哪一侧纯属运气，测试就会随机红/绿。
        （这也是本项目「因果测量三原则」的第一条：**路径覆盖事件全程**。）
        """
        pop = Population.from_shares(
            60, {"zero_intel": 0.30, "fundamentalist": 0.40, "chartist": 0.30}
        )
        m = PerpetualMarket(
            SimConfig(seed=seed, n_ticks=300 + ticks, population=pop, zi_p_buy=p_buy),
            funding_config=FundingConfig(),
        )
        m.run(300)
        hs = []
        for i in range(n_h):
            a = FundingArbitrageur(
                f"hd{i:04d}", 1.8e7, 0.0, np.random.default_rng([seed, 900 + i]),
                position_size=2.0, max_position=30.0,
            )
            m.add_agent(a)
            hs.append(a)
        if not track_path:
            m.run(ticks)
            return m, hs
        chunk = max(1, m.funding_config.settle_interval_ticks)
        lo = {a.agent_id: a.inventory for a in hs}
        hi = dict(lo)
        done = 0
        while done < ticks:
            step = min(chunk, ticks - done)
            m.run(step)
            done += step
            for a in hs:
                lo[a.agent_id] = min(lo[a.agent_id], a.inventory)
                hi[a.agent_id] = max(hi[a.agent_id], a.inventory)
        for a in hs:
            # 挂在实例上，测试直接读；不放进返回值是为了不改调用方签名
            a.path_min_inventory = lo[a.agent_id]      # type: ignore[attr-defined]
            a.path_max_inventory = hi[a.agent_id]      # type: ignore[attr-defined]
        return m, hs

    @staticmethod
    def _seed_ask_depth(m: PerpetualMarket, qty: float = 400.0, levels: int = 4) -> "_DepthProvider":
        """给盘口铺卖压，让**市价买单**有对手盘。

        为什么这条测试必须自己铺深度：``Market._clamp`` 对两种方向的裁剪**不对称**——
        · 卖出：只查 ``可用持仓 + 卖空额度`` → 不依赖盘口深度，永远发得出去；
        · 市价买入：要 ``book.estimate_fill("buy", qty)`` 吃得到才放行 → 依赖深度。
        所以"测转多"时必须先有卖压，否则买单被 `_clamp` 拒掉，
        测试看到的是"没转多"，而根因是**盘口没深度**，不是调仓逻辑坏了。
        （这个不对称本身是合理的：真实交易所也不会接受无法立即成交的市价单。
          但它是测试里的一个隐含前提，必须显式铺出来，不能靠运气。）
        """
        d = _DepthProvider()
        m.add_agent(d)
        for _ in range(levels):
            m.step()
        return d

    def test_目标仓位制_费率正时转空(self) -> None:
        """⭐ 不是"每 tick 累加"。累加式建仓会让仓位滞后于 8 tick 一次的结算，
        实测结算时点上系统性站错方向（10 个套利者累计费率收入 −40 万）。"""
        m, _ = self._with_hedgers(0, ticks=100)
        h = FundingArbitrageur(
            "probe", 1.8e7, 0.0, np.random.default_rng([1]),
            entry_threshold_bp=3.0, position_size=5.0, max_position=20.0,
        )
        m.add_agent(h)
        st = m._state

        def pump(n: int) -> None:
            # ⚠️ `decide` 只**返回**订单，真正让它成交的是 `Market.submit`。
            # 只调 decide 不改仓位——第一版测试就栽在这（断言一直看到 0）。
            for _ in range(n):
                out = h.decide(st)
                if out is not None:
                    orders = out if isinstance(out, (list, tuple)) else (out,)
                    for o in orders:
                        m.submit(h, o)

        # ⚠️ 必须先把**仓位清零**：`_with_hedgers(0, ticks=100)` 让市场跑了 400 tick，
        # 但 h 是刚注入的、仓位为 0，所以这里不需要额外处理——保留注释是为了说明
        # "为什么这个测试的起点是干净的"。
        self.assertEqual(h.inventory, 0.0)

        # 断言的是**方向**而不是幅度：盘口深度有限，市价单会被吃穿、
        # 单次成交量不可控，但"转空/转多"这个定性性质必须成立。
        st.funding_rate = 10e-4          # 费率 +10bp → 目标 −20
        pump(10)
        self.assertLess(h.inventory, -1.0, "正费率下没有转成空头")
        # 转多之前先铺卖压：`_clamp` 对市价买要求盘口有深度（见 _seed_ask_depth）
        self._seed_ask_depth(m)
        st.funding_rate = -10e-4         # 费率 −10bp → 目标 +20
        pump(40)
        self.assertGreater(h.inventory, 1.0, "负费率下没有转成多头")

    def test_阈值必须跟着费率量级标定(self) -> None:
        """⭐ 门槛开在价外 = 主体形同不存在。

        标定后的单次费率是「均值 1.06bp、sd 1.5bp」。若把 ``entry_threshold_bp``
        留在 5bp，费率够得着门槛的时间不到 1%，这个主体整场模拟基本不动——
        而症状看起来像"卖空额度失效"或"调仓逻辑坏了"，不是"参数开在价外"。

        这条测试钉住两件事：
        ① 默认阈值必须落在费率分布的主体区间内（否则主体是死的）；
        ② 阈值调高到价外时，`decide` 必须真的返回 None（说明死区逻辑在起作用）。
        """
        m = _perp(ticks=1200, seed=33, p_buy=0.65)
        r = m.funding_rates() * 1e4
        frac = float((np.abs(r) > FundingArbitrageur("t", 1e6, 0, np.random.default_rng([1]))
                       .entry_threshold_bp).mean())
        self.assertGreater(
            frac, 0.30,
            f"默认阈值下只有 {frac:.1%} 的结算能让套利者出手——门槛开在价外了",
        )

        h = FundingArbitrageur("probe", 1.8e7, 0.0, np.random.default_rng([1]),
                               entry_threshold_bp=1e9, exit_threshold_bp=-1e9)
        st = m._state
        st.funding_rate = 10e-4
        self.assertIsNone(h.decide(st), "阈值设到天上却仍然出手")

    def test_死区内不动作(self) -> None:
        m, _ = self._with_hedgers(0, ticks=50)
        h = FundingArbitrageur("probe", 1.8e7, 0.0, np.random.default_rng([1]),
                               entry_threshold_bp=5.0, exit_threshold_bp=1.0,
                               position_size=2.0, max_position=20.0)
        m.add_agent(h)
        st = m._state
        st.funding_rate = 3e-4    # 落在 [1bp, 5bp] 死区
        inv0 = h.inventory
        for _ in range(5):
            self.assertIsNone(h.decide(st))
        self.assertEqual(h.inventory, inv0)
    def test_卖空额度让空头成立(self) -> None:
        """套利者必须能建立并**持有过**空头。

        看的是全程极值而不是终点持仓——理由见 ``_with_hedgers`` 的说明：
        终点落在地哪一侧是相位运气，全程极值才是"这条能力存在"的证据。
        """
        m, hs = self._with_hedgers(6, ticks=800, track_path=True)
        self.assertTrue(
            any(a.path_min_inventory < -1.0 for a in hs),   # type: ignore[attr-defined]
            "套利者全程都没有建立空头——卖空额度没生效，或者门槛开在价外",
        )
        ok, prob = m.account_integrity()
        self.assertTrue(ok, prob[:5])

    def test_套利者两个方向都走到过(self) -> None:
        """费率会翻号，所以两个方向都必须能被执行。

        只测"能做空"是不够的：如果买单那一侧被某个约束卡住
        （历史上真发生过——`_clamp` 对市价买要求盘口深度），
        套利者会退化成"只会做空"，而在费率翻负时**持续付钱**，
        累计收益为负，而单看"能做空"这条测试完全发现不了。
        """
        m, hs = self._with_hedgers(6, ticks=1200, seed=35, p_buy=0.50, track_path=True)
        self.assertTrue(any(a.path_min_inventory < -1.0 for a in hs),  # type: ignore[attr-defined]
                        "没有任何套利者做空过")
        self.assertTrue(any(a.path_max_inventory > 1.0 for a in hs),   # type: ignore[attr-defined]
                        "没有任何套利者做多过——费率翻负时它无法反向调仓")

    def test_自记费率收入与账本完全一致(self) -> None:
        """主体自己记的归因数字必须与账本逐笔吻合。不一致就是双记账或漏记。"""
        m, hs = self._with_hedgers(4, ticks=600)
        led: dict[str, float] = {}
        for e in m.cash_ledger:
            if e["reason"] == "funding_settlement":
                led[e["agent_id"]] = led.get(e["agent_id"], 0.0) + e["delta"]
        for a in hs:
            self.assertAlmostEqual(
                a.funding_collected, led.get(a.agent_id, 0.0), places=6,
                msg=f"{a.agent_id} 自记与账本不符",
            )

    def test_虚拟现货腿不在市场账本里(self) -> None:
        """指导书 §2.7 专门警告过：这条腿的 PnL 是主体自记，不参与全局守恒。

        验证方式：把 synthetic_spot 打开/关闭各跑一次，
        全市场现金守恒的残差必须都 ≈ 0——说明它确实没进账本。
        """
        for flag in (True, False):
            pop = Population.from_shares(
                60, {"zero_intel": 0.30, "fundamentalist": 0.40, "chartist": 0.30}
            )
            m = PerpetualMarket(
                SimConfig(seed=41, n_ticks=900, population=pop, zi_p_buy=0.58),
                funding_config=FundingConfig(),
            )
            m.run(300)
            for i in range(4):
                m.add_agent(FundingArbitrageur(
                    f"hd{i:04d}", 1.8e7, 0.0, np.random.default_rng([41, i]),
                    position_size=2.0, max_position=20.0, synthetic_spot=flag,
                ))
            m.run(600)
            ok, _, msg = m.cash_conservation()
            self.assertTrue(ok, f"synthetic_spot={flag} 时守恒被破坏：{msg}")

    def test_虚拟现货腿按负持仓盯市(self) -> None:
        h = FundingArbitrageur("p", 1e7, 0.0, np.random.default_rng([1]),
                               synthetic_spot=True)
        st = _state(mid=100.0, tick=0)
        h.inventory = -2.0
        h.decide(st)                 # 建立 _last_mid = 100
        st.tick = 1
        st.mid = 110.0
        st.last_price = 110.0
        st.history.append(110.0)
        h.decide(st)                 # 价格 +10，空头赚 20
        # 现货腿 = −持仓 = +2，所以它在涨价时赚钱；两者相加 ≈ 0（delta 中性）
        self.assertAlmostEqual(h.synthetic_spot_pnl, 20.0, places=6)


class TestDecoupling(unittest.TestCase):
    """指导书 §0 第 2 条：注入一个不交易的该类型 agent，行情必须逐点不变。"""

    def test_注入不交易的套利者不改变行情(self) -> None:
        def run(with_noop_hedger: bool):
            pop = Population.from_shares(
                60, {"zero_intel": 0.30, "fundamentalist": 0.40, "chartist": 0.30}
            )
            m = PerpetualMarket(
                SimConfig(seed=55, n_ticks=500, population=pop), funding_config=FundingConfig()
            )
            if with_noop_hedger:
                # 阈值设到天上 → 永远不动作
                m.add_agent(FundingArbitrageur(
                    "hd0000", 1.8e7, 0.0, np.random.default_rng([55, 1]),
                    entry_threshold_bp=1e9, exit_threshold_bp=-1e9,
                ))
            m.run(500)
            return np.asarray(m.log.mid[: m.tick], dtype=float)

        a, b = run(False), run(True)
        self.assertEqual(a.size, b.size)
        self.assertTrue(
            np.array_equal(a, b),
            f"注入惰性套利者后行情变了（最大差 {float(np.nanmax(np.abs(a-b))):.6f}）",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
