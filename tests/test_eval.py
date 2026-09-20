"""策略评估工具的测试。

这个文件的意义比内核测试更微妙：**评估代码出错不会崩，只会给出错的结论**。
一个能赚钱的策略被评估工具算成亏钱、或者反过来，实验照跑不误，
只是所有决策都建立在假数字上。所以这里用两条路子交叉验证：

  · **人工构造的已知答案**（合成成交 + 合成价格序列）—— 精确断言数值，
    不给"看起来差不多"留余地；
  · **真实模拟上的恒等式** —— PnL 三分解之和必须**精确等于**权益变化。
    这条恒等式是数学上必然成立的，不成立就一定是代码错了。
"""

from __future__ import annotations

import unittest

import numpy as np

from tw import Market, Population, SimConfig
from tw.eval import (
    Fill,
    collect_fills,
    drawdown_stats,
    evaluate,
    inventory_stats,
    markout,
    mid_series,
    pnl_breakdown,
    strategy_paths,
)
from tw.scenarios import get_scenario
from tw.strategy import make_strategy

from strategies import Noop


def _fill(tick, side, price, qty, mid, is_maker=True):
    return Fill(tick=tick, side=side, price=price, quantity=qty, mid=mid,
                is_maker=is_maker, counterparty="x")


class TestMarkoutKnownAnswers(unittest.TestCase):
    """合成数据上的精确断言。"""

    def test_买入后价格上涨_漂移为正且数值精确(self) -> None:
        mid = np.array([100.0, 101.0, 102.0, 103.0])
        fills = [_fill(0, "buy", 100.0, 1.0, 100.0)]
        m = markout(fills, mid, horizons=(1, 2, 3))
        # (101-100)/100 = 1% = 100bp
        self.assertAlmostEqual(m["drift_1_bp"]["mean"], 100.0, places=6)
        self.assertAlmostEqual(m["drift_2_bp"]["mean"], 200.0, places=6)
        self.assertAlmostEqual(m["capture_bp"]["mean"], 0.0, places=6)

    def test_卖出后价格下跌_漂移为正_方向必须对齐(self) -> None:
        """按策略方向对齐：卖出后价格跌，对策略是**好**事，漂移应为正。

        这是最容易写反的一处。如果符号搞反了，做市商测出来会像在接刀，
        主动单测出来会像有信息——两个方向的结论同时颠倒。
        """
        mid = np.array([100.0, 99.0, 98.0])
        fills = [_fill(0, "sell", 100.0, 1.0, 100.0)]
        m = markout(fills, mid, horizons=(1, 2))
        self.assertAlmostEqual(m["drift_1_bp"]["mean"], 100.0, places=6)
        self.assertAlmostEqual(m["drift_2_bp"]["mean"], 200.0, places=6)

    def test_被动挂单的价差捕获为正(self) -> None:
        """买单成交价低于中间价 = 赚到价差，capture 为正。"""
        mid = np.array([100.0, 100.0])
        m = markout([_fill(0, "buy", 99.0, 1.0, 100.0)], mid, horizons=(1,))
        self.assertAlmostEqual(m["capture_bp"]["mean"], 100.0, places=6)

    def test_必须用成交时中间价_不能用tick收盘价(self) -> None:
        """漂移的基准是 ``mid_at_fill``，不是 ``log.mid[tick]``。

        构造一个两者明显不同的场景：
          mid_at_fill = 110（成交那一刻的中间价）
          log.mid[0]  = 100（该 tick 收盘价）
          log.mid[1]  = 120
        正确算法：(120 − 110)/110 = 909bp
        错误算法（用收盘价）：(120 − 100)/100 = 2000bp
        两者差一倍多，足以把结论颠倒。
        """
        mid = np.array([100.0, 120.0])
        fills = [_fill(0, "buy", 110.0, 1.0, 110.0)]
        m = markout(fills, mid, horizons=(1,))
        self.assertAlmostEqual(m["drift_1_bp"]["mean"], 909.090909, places=4)
        self.assertNotAlmostEqual(m["drift_1_bp"]["mean"], 2000.0, places=1)

    def test_样本不足时t值为nan而不是0(self) -> None:
        """t 值在样本 <3 时必须是 nan，不能悄悄返回 0。

        返回 0 会让"没有样本"看起来像"效应恰好为零、且非常显著地为零"。
        """
        mid = np.arange(100.0, 110.0)
        m = markout([_fill(0, "buy", 100.0, 1.0, 100.0)], mid, horizons=(1,))
        self.assertTrue(np.isnan(m["drift_1_bp"]["t"]))


class TestPnLIdentity(unittest.TestCase):
    """PnL 三分解之和 = 权益变化。这是数学恒等式，不成立就是代码错。"""

    def test_合成成交上恒等式精确成立(self) -> None:
        mid = np.array([100.0, 102.0, 101.0, 105.0])
        fills = [
            _fill(0, "buy", 99.5, 3.0, 100.0),
            _fill(1, "buy", 103.0, 1.0, 102.0),
            _fill(2, "sell", 102.0, 2.0, 101.0),
        ]
        inv0 = 4.0
        b = pnl_breakdown(fills, mid, inv0=inv0)
        cash0 = 0.0
        # 手工推权益
        cash = cash0
        inv = inv0
        for f in fills:
            s = 1.0 if f.side == "buy" else -1.0
            cash -= s * f.quantity * f.price
            inv += s * f.quantity
        eq0 = cash0 + inv0 * mid[0]
        eq1 = cash + inv * mid[-1]
        self.assertAlmostEqual(b["pnl_flow"], eq1 - eq0, places=9)

    def test_真实模拟上恒等式成立(self) -> None:
        """整场模拟跑完，逐笔累加的结果必须与权益变化一致。

        这条同时验证了"用成交记录反推起始状态"（``strategy_paths``）的正确性——
        如果反推错了（比如漏算某笔），残差会立刻暴露。
        """
        sc = get_scenario("normal")
        sc.n_ticks = 600
        sc.warmup = 400
        m = sc.build(seed=1357)
        from strategies import NaiveMaker

        st = make_strategy(NaiveMaker, "strat", seed=1357,
                           cash=60_000.0 * 200, inventory=0.0,
                           half_spread_bp=10.0, qty=2.0)
        m.add_agent(st)
        m.run(sc.n_ticks)
        r = evaluate(m, "strat")
        self.assertGreater(r["n_fills"], 50, "样本太少，测试前提不成立")
        self.assertLess(
            abs(r["pnl_identity_residual"]), 1e-6,
            f"PnL 恒等式残差 {r['pnl_identity_residual']:.6f}",
        )

    def test_什么都不做必须零成交零盈亏(self) -> None:
        """Noop 是整套评估的口径基准。它不为 0，说明评估代码有问题，
        而不是"策略表现特殊"。"""
        sc = get_scenario("normal")
        sc.n_ticks = 400
        sc.warmup = 400
        m = sc.build(seed=99)
        m.add_agent(make_strategy(Noop, "noop", seed=99, cash=1e9, inventory=0.0))
        m.run(sc.n_ticks)
        r = evaluate(m, "noop")
        self.assertEqual(r["n_fills"], 0)
        self.assertEqual(r["volume"], 0.0)
        self.assertAlmostEqual(r["pnl_total_from_equity"], 0.0, places=9)
        self.assertAlmostEqual(r["pnl_capture"], 0.0, places=9)
        self.assertAlmostEqual(r["pnl_inventory"], 0.0, places=9)


class TestPathAndRiskStats(unittest.TestCase):
    def test_回撤计算(self) -> None:
        e = np.array([100.0, 120.0, 90.0, 130.0, 110.0])
        d = drawdown_stats(e)
        self.assertAlmostEqual(d["max_dd"], -30.0, places=9)      # 120 → 90
        self.assertAlmostEqual(d["max_dd_pct"], -0.25, places=9)  # -30/120

    def test_库存统计(self) -> None:
        inv = np.array([0.0, 0.0, 4.0, -4.0])
        s = inventory_stats(inv)
        self.assertAlmostEqual(s["inv_abs_mean"], 2.0, places=9)
        self.assertAlmostEqual(s["inv_abs_max"], 4.0, places=9)
        self.assertAlmostEqual(s["flat_frac"], 0.5, places=9)

    def test_中间价序列做前向填充(self) -> None:
        """单边盘口空缺时日志里会写 NaN；统计前必须填掉，
        否则后面所有指标都会被 NaN 吞成空。"""
        cfg = SimConfig(seed=1, n_ticks=5, population=Population(zero_intel=5))
        m = Market(cfg)
        m.run()
        m.log.mid[2] = np.nan
        s = mid_series(m)
        self.assertTrue(np.all(np.isfinite(s)))
        self.assertEqual(s.size, m.tick)


class TestStrategyApi(unittest.TestCase):
    """策略 API 的边界行为。"""

    def test_策略抛异常不会带崩模拟_且会被记录(self) -> None:
        from tw.strategy import Strategy

        class Boom(Strategy):
            def on_tick(self, ctx):  # noqa: ANN001
                raise ValueError("故意炸")

        sc = get_scenario("normal")
        sc.n_ticks = 60
        sc.warmup = 300
        m = sc.build(seed=7)
        st = make_strategy(Boom, "boom", seed=7, cash=1e6, inventory=0.0)
        m.add_agent(st)
        m.run(sc.n_ticks)  # 不应当抛出
        self.assertEqual(st.n_errors, sc.n_ticks)
        self.assertIn("故意炸", st.first_error or "")

    def test_下单会裁剪到可用额度(self) -> None:
        """想下 1000 手但只买得起 2 手时，应当**下 2 手**而不是拒单。

        直接拒单会让策略在资金紧张时完全停摆，而真相是"它能做小一点"。
        """
        sc = get_scenario("normal")
        sc.n_ticks = 30
        sc.warmup = 300
        m = sc.build(seed=8)
        from tw.strategy import Strategy

        class Greedy(Strategy):
            def on_tick(self, ctx):  # noqa: ANN001
                return ctx.buy(ctx.mid or 1.0, 1000.0)

        st = make_strategy(Greedy, "g", seed=8, cash=1000.0, inventory=0.0)
        m.add_agent(st)
        m.run(sc.n_ticks)
        self.assertGreater(st.stats.n_submitted, 0, "被整单拒掉了，应当裁剪")
        self.assertGreaterEqual(st.cash, -1e-6, "现金被下成负数")

    def test_limit_at方向正确(self) -> None:
        sc = get_scenario("normal")
        sc.n_ticks = 10
        sc.warmup = 300
        m = sc.build(seed=4)
        from tw.strategy import Strategy

        seen = {}

        class Probe(Strategy):
            def on_tick(self, ctx):  # noqa: ANN001
                if not seen:
                    o1 = ctx.limit_at("buy", 10.0, 1.0)
                    o2 = ctx.limit_at("sell", 10.0, 1.0)
                    seen["buy"] = o1.price if o1 else None
                    seen["sell"] = o2.price if o2 else None
                    seen["mid"] = ctx.mid
                return None

        st = make_strategy(Probe, "p", seed=4, cash=1e7, inventory=10.0)
        m.add_agent(st)
        m.run(sc.n_ticks)
        self.assertLess(seen["buy"], seen["mid"])
        self.assertGreater(seen["sell"], seen["mid"])

    def test_quote返回双边(self) -> None:
        sc = get_scenario("normal")
        sc.n_ticks = 10
        sc.warmup = 300
        m = sc.build(seed=6)
        from tw.strategy import Strategy

        got = {}

        class Q(Strategy):
            def on_tick(self, ctx):  # noqa: ANN001
                if not got:
                    m_ = ctx.mid or 1.0
                    got["n"] = len(ctx.quote(m_ * 0.999, m_ * 1.001, 1.0))
                return None

        st = make_strategy(Q, "q", seed=6, cash=1e7, inventory=10.0)
        m.add_agent(st)
        m.run(sc.n_ticks)
        self.assertEqual(got["n"], 2)


class TestSettlementUnchanged(unittest.TestCase):
    """成交明细的字段完备性：评估工具靠它，缺一个就静默失真。"""

    def test_每笔成交都带成交时中间价(self) -> None:
        sc = get_scenario("normal")
        sc.n_ticks = 300
        sc.warmup = 400
        m = sc.build(seed=11)
        m.run(sc.n_ticks)
        trades = m.log.trades
        self.assertGreater(len(trades), 100)
        bad = [t for t in trades if not (t.mid_at_fill and t.mid_at_fill > 0)]
        self.assertEqual(bad, [], f"{len(bad)} 笔成交没有回填 mid_at_fill")

    def test_成交时中间价与tick收盘价确实不同(self) -> None:
        """如果两者总是相同，说明 mid_at_fill 根本没起作用（可能写成了
        `log.mid[tick]` 的副本），那条"必须用成交时中间价"的断言就失去意义。"""
        sc = get_scenario("normal")
        sc.n_ticks = 300
        sc.warmup = 400
        m = sc.build(seed=12)
        m.run(sc.n_ticks)
        mid = mid_series(m)
        diffs = [
            abs(t.mid_at_fill - mid[min(t.tick, mid.size - 1)])
            for t in m.log.trades
            if t.tick < mid.size
        ]
        self.assertGreater(np.mean(diffs), 0.0, "两者完全相同，字段可能没被真正使用")


class TestFromTick(unittest.TestCase):
    """有预热的实验必须显式传 `from_tick`，否则指标会算在错误的窗口上。"""

    @staticmethod
    def _make(t0: int, ticks: int = 500, seed: int = 31):
        sc = get_scenario("normal")
        sc.warmup = t0
        sc.n_ticks = ticks
        m = sc.build(seed=seed)
        from strategies import NaiveMaker

        st = make_strategy(NaiveMaker, "strat", seed=seed,
                           cash=60_000.0 * 300, inventory=0.0,
                           half_spread_bp=10.0, qty=2.0)
        m.add_agent(st)
        m.run(ticks)
        return m, st

    def test_from_tick把窗口裁到注入之后(self) -> None:
        m, _ = self._make(t0=600)
        full, path_full = strategy_paths(m, "strat", 0)
        win, path_win = strategy_paths(m, "strat", m.injected_at["strat"])
        # 窗口内的长度 = 观测长度，且成交笔数不会变多
        self.assertEqual(path_win["mid"].size, m.tick - m.injected_at["strat"])
        self.assertLessEqual(len(win), len(full))
        # 成交 tick 必须被换算成窗口内下标
        if win:
            self.assertLess(max(f.tick for f in win), path_win["mid"].size)

    def test_from_tick下恒等式依然精确成立(self) -> None:
        m, _ = self._make(t0=600)
        r = evaluate(m, "strat", from_tick=m.injected_at["strat"])
        self.assertGreater(r["n_fills"], 30)
        self.assertLess(abs(r["pnl_identity_residual"]), 1e-6)

    def test_带底仓时from_tick影响底仓重估项(self) -> None:
        """底仓非零时，用错窗口会让 `pnl_initial_mark` 直接算错。

        构造：给策略一笔初始底仓，比较 from_tick=0 与 from_tick=t0 的结果。
        两者都必须满足各自的恒等式，但 initial_mark 的取值不同
        （因为基准价从 tick 0 的价格变成了注入时刻的价格）。
        """
        sc = get_scenario("normal")
        sc.warmup = 600
        sc.n_ticks = 400
        m = sc.build(seed=77)
        from strategies import NaiveMaker

        mid_at_inject = m.current_mid()
        st = make_strategy(NaiveMaker, "strat", seed=77,
                           cash=60_000.0 * 300, inventory=20.0,
                           half_spread_bp=10.0, qty=1.0)
        m.add_agent(st)
        m.run(400)

        t0 = m.injected_at["strat"]
        a = evaluate(m, "strat", from_tick=0)
        b = evaluate(m, "strat", from_tick=t0)
        self.assertLess(abs(a["pnl_identity_residual"]), 1e-6)
        self.assertLess(abs(b["pnl_identity_residual"]), 1e-6)
        # 窗口内的"成交前基准价"必须等于注入那一刻的中间价。
        # 注意不能拿 `mid_start` 去比：那是第 0 个 tick **结束**时的价格，
        # 而注入发生在该 tick 开始之前，两者差着这一整个 tick 的行情。
        self.assertAlmostEqual(b["mid_pre"], mid_at_inject, places=6)
        self.assertNotAlmostEqual(a["mid_pre"], b["mid_pre"], places=2)


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
