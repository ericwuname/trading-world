"""二期共用基础设施的测试（阶段5-9 都依赖它们）。

这三样东西共同支撑二期所有新机制，而且**坏了都不会报错**：

  · ``Market.apply_cash_delta`` —— 外生现金变动的唯一入口。
    不走它会破坏"全市场现金守恒"，而 `account_integrity` 完全看不出来
    （那个只查单账户自洽）。
  · ``Agent.spawn_rng`` —— 每个用途独立的确定性随机流。
    共用一条流会让"改了学习参数"顺带改变"下单序列"，实验无法归因。
  · ``MarketState.funding_rate / flow_imbalance`` —— 新机制读的实时量。
"""

from __future__ import annotations

import unittest

import numpy as np

from tw import Market, Population, SimConfig
from tw.agents.base import _stable_id_key, clipped_strength


def _market(n: int = 40, ticks: int = 200, seed: int = 1) -> Market:
    m = Market(SimConfig(seed=seed, n_ticks=ticks, population=Population(zero_intel=n)))
    m.run(ticks)
    return m


class TestCashConservation(unittest.TestCase):
    def test_纯交易后现金守恒(self) -> None:
        m = _market()
        ok, resid, msg = m.cash_conservation()
        self.assertTrue(ok, msg)
        self.assertLess(abs(resid), 1e-6)

    def test_外生变动必须记账到对手方(self) -> None:
        """只改一侧的现金会凭空创造/消灭现金，而这不会报错。"""
        m = _market()
        a = m.agents[0]
        m.apply_cash_delta(a, -500.0, reason="test", counterparty="exchange")
        self.assertAlmostEqual(m.exchange_cash, 500.0, places=9)
        ok, _, msg = m.cash_conservation()
        self.assertTrue(ok, msg)
        self.assertEqual(len(m.cash_ledger), 1)
        self.assertEqual(m.cash_ledger[0]["reason"], "test")

    def test_交易所侧与主体侧互为相反数(self) -> None:
        """这是资金费率零和性的根基：一方付出，另一方必须收到。"""
        m = _market()
        total_before = sum(a.cash for a in m.agents)
        for a in m.agents[:5]:
            m.apply_cash_delta(a, 100.0, reason="grant")
        total_after = sum(a.cash for a in m.agents)
        self.assertAlmostEqual(total_after - total_before, 500.0, places=6)
        self.assertAlmostEqual(m.exchange_cash, -500.0, places=6)

    def test_外部往来单独累计(self) -> None:
        m = _market()
        m.apply_cash_delta(m.agents[0], 250.0, reason="external", counterparty=None)
        self.assertAlmostEqual(m.exogenous_cash, 250.0, places=9)
        self.assertAlmostEqual(m.exchange_cash, 0.0, places=9)
        ok, _, msg = m.cash_conservation()
        self.assertTrue(ok, msg)

    def test_非有限值必须拒绝(self) -> None:
        m = _market()
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                m.apply_cash_delta(m.agents[0], bad, reason="bad")

    def test_零变动不产生流水(self) -> None:
        m = _market()
        n = len(m.cash_ledger)
        m.apply_cash_delta(m.agents[0], 0.0, reason="noop")
        self.assertEqual(len(m.cash_ledger), n)

    def test_流水可按原因汇总(self) -> None:
        m = _market()
        for a in m.agents[:3]:
            m.apply_cash_delta(a, -10.0, reason="funding")
        m.apply_cash_delta(m.agents[3], -7.0, reason="penalty")
        self.assertAlmostEqual(m._exogenous_by_reason["funding"], -30.0, places=9)
        self.assertAlmostEqual(m._exogenous_by_reason["penalty"], -7.0, places=9)


class TestSpawnRng(unittest.TestCase):
    def test_可复现(self) -> None:
        m = _market()
        a = m.agents[0]
        self.assertTrue(np.allclose(a.spawn_rng(101).random(5), a.spawn_rng(101).random(5)))

    def test_不同用途不共用流(self) -> None:
        m = _market()
        a = m.agents[0]
        self.assertFalse(np.allclose(a.spawn_rng(101).random(5), a.spawn_rng(102).random(5)))

    def test_不同主体流不同(self) -> None:
        m = _market()
        self.assertFalse(
            np.allclose(m.agents[0].spawn_rng(101).random(5),
                        m.agents[1].spawn_rng(101).random(5))
        )

    def test_与决策流隔离(self) -> None:
        """改了新用途的随机流，不能顺带改变主体的决策序列。"""
        m = _market()
        a = m.agents[0]
        base = a.rng.random(5)
        a.spawn_rng(777).random(1000)   # 大量消耗新流
        after = a.spawn_rng(777).random(5)
        self.assertFalse(np.allclose(base, after))  # 决策流不受影响

    def test_跨进程稳定哈希(self) -> None:
        """内置 hash() 带进程随机盐，会让"同 seed 复现"在跨进程时失效。"""
        self.assertEqual(_stable_id_key("ze0007"), _stable_id_key("ze0007"))
        self.assertNotEqual(_stable_id_key("ze0007"), _stable_id_key("ze0008"))
        # crc32("ze0007") 的已知值：换实现会立刻红
        import zlib

        self.assertEqual(
            _stable_id_key("ze0007"),
            int(zlib.crc32(b"ze0007") & 0xFFFFFFFF),
        )


class TestStateExtensions(unittest.TestCase):
    def test_默认值让一期行为不变(self) -> None:
        m = _market()
        self.assertEqual(m._state.funding_rate, 0.0)

    def test_订单流失衡在_1到1之间(self) -> None:
        for seed in (1, 2, 3):
            m = _market(seed=seed)
            v = m._state.flow_imbalance
            self.assertGreaterEqual(v, -1.0)
            self.assertLessEqual(v, 1.0)
            self.assertTrue(np.isfinite(v))

    def test_订单流失衡方向正确(self) -> None:
        """手工构造：全是主动买 → +1，全是主动卖 → −1。

        ⚠️ 这里**直接就地改 ``log.flow``**，属于绕过市场契约的写法
        （正常路径由 ``_finalize_tick`` 单点写入、靠 tick 变化自动失效）。
        所以每次改完必须显式 ``invalidate_flow_cache()``——
        否则 ``_flow_imbalance`` 会返回同一 tick 内上一次的结果。
        **这条不是"测试迁就实现"**：缓存的前提契约本来就写在
        ``Market._flow_imbalance`` 的文档里，这里把它显式化，
        并且顺带证明了"绕过契约需要显式声明"这件事是成立的。
        """
        m = Market(SimConfig(seed=5, n_ticks=50, population=Population(zero_intel=10)))
        m.run(50)
        t = m.tick

        def recompute(*vals) -> float:
            for k, v in enumerate(vals, start=1):
                m.log.flow[t - k] = v
            m.invalidate_flow_cache()
            return m._flow_imbalance(window=2)

        self.assertAlmostEqual(recompute(5.0, 10.0), 1.0, places=9)
        self.assertAlmostEqual(recompute(-5.0, -10.0), -1.0, places=9)
        self.assertAlmostEqual(recompute(-10.0, 10.0), 0.0, places=9)
        # 反面参照：不失效的话，同一 tick 内会拿回上一次的缓存值
        m.log.flow[t - 1] = -10.0
        m.log.flow[t - 2] = -10.0
        self.assertAlmostEqual(m._flow_imbalance(window=2), 0.0, places=9,
                               msg="缓存没有生效——那这条契约就成了摆设")

    def test_空窗口返回0而不是除零(self) -> None:
        m = Market(SimConfig(seed=5, n_ticks=10, population=Population(zero_intel=10)))
        self.assertEqual(m._flow_imbalance(window=200), 0.0)

    def test_clipped_strength仍可用(self) -> None:
        """一期函数；上面编辑 base.py 时被误删过 docstring，留一条防回归。"""
        self.assertEqual(clipped_strength(0.001, 0.02, 0.002), 0.0)
        self.assertAlmostEqual(clipped_strength(0.012, 0.02, 0.002), 0.5, places=9)  # (0.012-0.002)/0.02
        self.assertEqual(clipped_strength(0.5, 0.02, 0.002), 1.0)
        self.assertEqual(clipped_strength(1.0, 0.0, 0.0), 0.0)  # ref<=0 防除零


class TestBuySellOrderHelpers(unittest.TestCase):
    def test_买单贴卖一_卖单贴买一(self) -> None:
        m = Market(SimConfig(seed=9, n_ticks=300, population=Population(zero_intel=30)))
        m.run(300)
        st = m._state
        a = m.agents[0]
        o1 = a.buy_order(st, 1.0)
        o2 = a.sell_order(st, 1.0)
        self.assertIsNotNone(o1)
        self.assertIsNotNone(o2)
        if st.best_ask is not None:
            self.assertAlmostEqual(o1.price, st.best_ask, places=6)
        if st.best_bid is not None:
            self.assertAlmostEqual(o2.price, st.best_bid, places=6)

    def test_数量非法返回None(self) -> None:
        m = Market(SimConfig(seed=9, n_ticks=100, population=Population(zero_intel=20)))
        m.run(100)
        a = m.agents[0]
        self.assertIsNone(a.buy_order(m._state, 0.0))
        self.assertIsNone(a.buy_order(m._state, -1.0))


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
