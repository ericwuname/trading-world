"""A4 测试：执行模拟 / 规则基线 / 完整回路 / 分层评估。

**全部离线**（LLM 部分用 ``ScriptedClient`` 或录制回放）。

这里的重点与前几轮不同：A4 的产物是**数字**（收益、夏普、回撤），
而数字最大的风险是"**算错了也看不出来**"。所以本文件的测试围绕
三类会静默出错的量展开：

1. **账本自检**：``noop`` 的收益必须**恰好**是 0。
   它不为 0 ⇒ 记账/执行有漏，**先修账本再谈策略**。
2. **时间错位**：决策在 bar i、成交最早在 bar i+1。
   少了这个错位会凭空变好，且不报错。
3. **成本口径**：``cost_multiplier`` 必须真的影响**手续费**，
   而不只是滑点（这个 bug 真实存在过：配置看着控制两者，
   实际只控制了一半，于是"成本 ×5 后收益几乎没变"看起来像
   "策略对成本不敏感"——一个漂亮且错误的结论）。
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from tw.account import MarginAccount, MarginConfig
from tw.agent import AgentConfig, TradingAgent, MODE_BASELINE
from tw.agent_run import order_request_from_record, run_agent_session
from tw.decision_log import DecisionRecord
from tw.eval_agent import (
    block_bootstrap_ci,
    compare_to_baselines,
    confidence_calibration,
    cost_sensitivity_analytic,
    cost_sensitivity_rerun,
    decision_layer,
    evaluate_run,
    execution_layer,
    max_drawdown,
    per_bar_returns,
    result_layer,
    sharpe,
)
from tw.llm import LLMConfig, LLMResponse, ScriptedClient
from tw.order_model import AlgoOrder, OrderRequest
from tw.policy import (
    AlwaysHold,
    MeanRevert,
    Momentum,
    RandomTaker,
    make_policy,
    validate_policy_output,
)
from tw.prompts import TEMPLATES
from tw.risk import RiskLimits
from tw.simexec import BarExecutor, ExecConfig


# ======================================================================
# 辅助
# ======================================================================
class _Series:
    """最小 Series 替身。

    ⚠️ 必须带 ``high`` / ``low`` / ``timestamp``——真实的
    ``marketdb.Series`` 永远有它们。缺字段的替身会让失败信号
    变成 ``AttributeError``（"崩了"）而不是断言失败（"错了"）。
    """

    def __init__(self, closes, spread: float = 0.002):
        self.close = np.asarray(closes, dtype=float)
        self.open = self.close
        self.high = self.close * (1.0 + spread)
        self.low = self.close * (1.0 - spread)
        self.timestamp = np.arange(len(self.close), dtype=np.int64) * 3_600_000


def _series(*, n=60, drift=0.0, seed=1, spread=0.002):
    rng = np.random.default_rng(seed)
    steps = rng.normal(drift, 0.003, n)
    return _Series(100.0 * np.exp(np.cumsum(steps)), spread=spread)


def _acc(cash=100_000.0):
    return MarginAccount(cash=float(cash), cfg=MarginConfig())


def _order(side="buy", sz=0.1, ot="market", px=None, algo=None,
           reduce_only=False, inst="X"):
    return OrderRequest(inst_id=inst, side=side, ord_type=ot, sz=sz, px=px,
                        td_mode="isolated", pos_side="net",
                        reduce_only=reduce_only, attach_algo=algo, lever=3.0)


def _run(policy, series, *, name="", exec_cfg=None, cash=100_000.0, lever=3.0,
         log=None):
    ag = TradingAgent(policy=policy,
                      config=AgentConfig(inst_id="X", lever=lever),
                      limits=RiskLimits())
    return run_agent_session(
        ag, series, account=_acc(cash), name=name or policy.name, run_id="T",
        exec_config=exec_cfg or ExecConfig(slippage_bps=1.0), lever=lever,
        record_log=log,
    )


# ======================================================================
# 执行模拟器
# ======================================================================
class TestBarExecutor(unittest.TestCase):
    def test_市价单在次根开盘成交(self):
        """⭐ 决策与成交必须错开一根（否则订单回到过去成交）。"""
        ex = BarExecutor(account=_acc(), config=ExecConfig(slippage_bps=0.0),
                         inst_id="X", lever=3.0)
        ex.submit(_order(), submit_bar=0)
        self.assertEqual(ex.on_bar(0, (100, 101, 99, 100.5)), [])
        fs = ex.on_bar(1, (100.5, 103, 100, 102))
        self.assertEqual(len(fs), 1)
        self.assertAlmostEqual(fs[0].price, 100.5)   # 次根开盘
        self.assertEqual(fs[0].reason, "market")

    def test_市价单带滑点且方向正确(self):
        """买单向上滑、卖单向下滑。**写反了会让回测凭空赚钱。**"""
        ex = BarExecutor(account=_acc(),
                         config=ExecConfig(slippage_bps=10.0, cost_multiplier=1.0),
                         inst_id="X", lever=3.0)
        ex.submit(_order("buy", 1.0), submit_bar=0)
        b = ex.on_bar(1, (100.0, 101, 99, 100))[0]
        self.assertGreater(b.price, 100.0)
        self.assertAlmostEqual(b.price, 100.1)       # 100 × (1+10bp)
        ex2 = BarExecutor(account=_acc(),
                          config=ExecConfig(slippage_bps=10.0), inst_id="X",
                          lever=3.0)
        ex2.submit(_order("sell", 1.0), submit_bar=0)
        s = ex2.on_bar(1, (100.0, 101, 99, 100))[0]
        self.assertLess(s.price, 100.0)

    def test_滑点成本被记下且为正(self):
        ex = BarExecutor(account=_acc(), config=ExecConfig(slippage_bps=10.0),
                         inst_id="X", lever=3.0)
        ex.submit(_order("buy", 1.0), submit_bar=0)
        f = ex.on_bar(1, (100.0, 101, 99, 100))[0]
        self.assertGreater(f.slippage_cost, 0)
        self.assertAlmostEqual(f.slippage_bps, 10.0, places=4)
        self.assertAlmostEqual(f.ref_price, 100.0)

    def test_限价单未触及不成交(self):
        ex = BarExecutor(account=_acc(), config=ExecConfig(), inst_id="X",
                         lever=3.0)
        ex.submit(_order(ot="limit", px=90.0), submit_bar=0)
        self.assertEqual(ex.on_bar(1, (100, 101, 99, 100)), [])
        self.assertEqual(ex.on_bar(2, (100, 101, 99, 100)), [])
        self.assertEqual(ex.on_bar(3, (100, 101, 99, 100)), [])
        self.assertEqual(ex.on_bar(4, (100, 101, 99, 100)), [])
        self.assertEqual(ex.expired_orders, 1, "过了 max_wait_bars 必须过期")

    def test_限价单触及即成交(self):
        ex = BarExecutor(account=_acc(), config=ExecConfig(slippage_bps=0.0),
                         inst_id="X", lever=3.0)
        ex.submit(_order(ot="limit", px=98.0), submit_bar=0)
        fs = ex.on_bar(1, (100, 101, 97, 99))
        self.assertEqual(len(fs), 1)
        self.assertAlmostEqual(fs[0].price, 98.0)     # 成交在自己的价上
        self.assertTrue(fs[0].is_maker)
        self.assertEqual(fs[0].slippage_cost, 0.0)    # 限价单没有滑点

    def test_同根同时触及tp与sl必须按sl(self):
        """⭐⭐ 模块文档的「假设 3」，也是全模块最重要的一条断言。

        bar 级回测看不到这一小时内的价格**路径**，所以 TP 和 SL
        同时落在区间里时无法知道谁先到。本模块**显式选悲观解**：
        按 SL 处理。选 TP 会让回测变好看——那正是"偏乐观的回测"
        让人以为策略能上线的机制。
        """
        ex = BarExecutor(account=_acc(), config=ExecConfig(slippage_bps=0.0),
                         inst_id="X", lever=3.0)
        ex.submit(_order(algo=AlgoOrder(tp_trigger_px=110.0,
                                        sl_trigger_px=95.0)), submit_bar=0)
        ex.on_bar(1, (100, 101, 99, 100))            # 开仓
        fs = ex.on_bar(2, (100, 112, 93, 100))       # 同根触及 110 与 95
        self.assertEqual(len(fs), 1)
        self.assertEqual(fs[0].reason, "sl")
        self.assertAlmostEqual(fs[0].price, 95.0)
        self.assertEqual(ex.account.position("X").qty, 0.0)

    def test_只触及tp时按tp(self):
        ex = BarExecutor(account=_acc(), config=ExecConfig(slippage_bps=0.0),
                         inst_id="X", lever=3.0)
        ex.submit(_order(algo=AlgoOrder(tp_trigger_px=110.0,
                                        sl_trigger_px=90.0)), submit_bar=0)
        ex.on_bar(1, (100, 101, 99, 100))
        fs = ex.on_bar(2, (100, 115, 99, 110))
        self.assertEqual(fs[0].reason, "tp")

    def test_空头的tp方向相反(self):
        """空头止盈在**下方**——写成和多头一样就等于把止盈做成止损。"""
        ex = BarExecutor(account=_acc(), config=ExecConfig(slippage_bps=0.0),
                         inst_id="X", lever=3.0)
        ex.submit(_order("sell", 1.0,
                         algo=AlgoOrder(tp_trigger_px=90.0,
                                        sl_trigger_px=110.0)), submit_bar=0)
        ex.on_bar(1, (100, 101, 99, 100))
        fs = ex.on_bar(2, (100, 101, 88, 90))        # 跌到 88 ⇒ 空头止盈
        self.assertEqual(fs[0].reason, "tp")

    def test_标的必须一致(self):
        """⭐ 接错标的会对着另一个仓位记账，留痕里看不出异常。"""
        ex = BarExecutor(account=_acc(), config=ExecConfig(), inst_id="X",
                         lever=3.0)
        with self.assertRaises(ValueError):
            ex.submit(_order(inst="Y"), submit_bar=0)

    def test_强平会关掉仓位(self):
        ex = BarExecutor(account=_acc(cash=10.0), config=ExecConfig(),
                         inst_id="X", lever=20.0)
        ex.submit(_order(sz=1.0), submit_bar=0)
        ex.on_bar(1, (100, 101, 99, 100))
        self.assertGreater(abs(ex.account.position("X").qty), 0)
        f = ex.check_liquidation(2, mark=90.0)       # 下跌 10% ⇒ 爆
        self.assertIsNotNone(f)
        self.assertEqual(f.reason, "liquidate")
        self.assertEqual(ex.liquidations, 1)

    def test_成本倍数真的影响手续费(self):
        """⭐⭐ 真实踩过的 bug。

        ``ExecConfig.cost_multiplier`` 最初**只影响滑点**——
        因为手续费走的是账户的 ``MarginConfig``，与执行配置无关。
        症状：成本 ×5 后收益几乎没变，看起来像"策略对成本不敏感"
        （一个漂亮且错误的结论）。
        """
        def total_fee(m):
            ex = BarExecutor(account=_acc(),
                             config=ExecConfig(slippage_bps=0.0,
                                               cost_multiplier=m),
                             inst_id="X", lever=1.0)
            for k in range(3):
                ex.submit(_order("buy", 1.0), submit_bar=k)
                ex.on_bar(k + 1, (100, 101, 99, 100))
                ex.submit(_order("sell", 1.0, reduce_only=True), submit_bar=k + 1)
                ex.on_bar(k + 2, (100, 101, 99, 100))
            return ex.account.total_fees

        f1, f2, f5 = total_fee(1.0), total_fee(2.0), total_fee(5.0)
        self.assertAlmostEqual(f2 / f1, 2.0, places=9)
        self.assertAlmostEqual(f5 / f1, 5.0, places=9)

    def test_成交返回成交价与手续费(self):
        ex = BarExecutor(account=_acc(), config=ExecConfig(slippage_bps=0.0),
                         inst_id="X", lever=3.0)
        ex.submit(_order("buy", 2.0), submit_bar=0)
        f = ex.on_bar(1, (100.0, 101, 99, 100))[0]
        self.assertAlmostEqual(f.notional, 200.0)
        self.assertGreater(f.fee, 0)


class TestExecConfig(unittest.TestCase):
    def test_默认值合理(self):
        c = ExecConfig()
        self.assertEqual(c.cost_multiplier, 1.0)
        self.assertGreater(c.max_wait_bars, 0)

    def test_负成本被拒(self):
        for kw in ({"taker_fee": -1}, {"slippage_bps": -1},
                   {"cost_multiplier": -1}):
            with self.subTest(**kw):
                with self.assertRaises(ValueError):
                    ExecConfig(**kw)

    def test_等待根数至少1(self):
        with self.assertRaises(ValueError):
            ExecConfig(max_wait_bars=0)

    def test_倍数放大费率与滑点(self):
        c = ExecConfig(taker_fee=0.001, slippage_bps=2.0, cost_multiplier=5.0)
        self.assertAlmostEqual(c.eff_taker, 0.005)
        self.assertAlmostEqual(c.eff_slippage, 10.0)

    def test_描述可写进报告(self):
        d = ExecConfig(cost_multiplier=2.0).describe()
        self.assertEqual(d["cost_multiplier"], 2.0)
        self.assertIn("taker_fee", d)


# ======================================================================
# 规则基线
# ======================================================================
class TestPolicies(unittest.TestCase):
    def _vis(self, closes):
        return {"mid": closes[-1], "recent_closes": list(closes)}

    def test_noop永远弃权(self):
        p = AlwaysHold()
        for closes in ([1, 2, 3], [3, 2, 1], [1, 1, 1]):
            out = p.decide(self._vis(closes), max_size=1.0)
            self.assertEqual(out["action"], "hold")

    def test_momentum顺势(self):
        p = Momentum(threshold=0.001)
        self.assertEqual(p.decide(self._vis([100, 101, 102, 105]),
                                  max_size=1.0)["action"], "buy")
        self.assertEqual(p.decide(self._vis([105, 102, 101, 100]),
                                  max_size=1.0)["action"], "sell")

    def test_meanrevert反向(self):
        p = MeanRevert(threshold=0.001)
        self.assertEqual(p.decide(self._vis([100, 101, 102, 105]),
                                  max_size=1.0)["action"], "sell")
        self.assertEqual(p.decide(self._vis([105, 102, 101, 100]),
                                  max_size=1.0)["action"], "buy")

    def test_阈值以下是弃权(self):
        p = Momentum(threshold=0.05)
        self.assertEqual(p.decide(self._vis([100, 100.1, 100.2]),
                                  max_size=1.0)["action"], "hold")

    def test_无可用量时弃权(self):
        """``max_size = 0``（没有权益/没有额度）时**任何**策略都必须弃权——
        否则会下出风控必然要拒的单，白记一堆 `size_cap`。"""
        vis = self._vis([100, 101, 102, 103])
        for p in (Momentum(threshold=0.0001), MeanRevert(threshold=0.0001),
                  RandomTaker(p_trade=1.0)):
            with self.subTest(policy=p.name):
                self.assertEqual(p.decide(vis, max_size=0.0)["action"], "hold")

    def test_样本不足时弃权(self):
        for p in (Momentum(), MeanRevert()):
            with self.subTest(policy=p.name):
                out = p.decide(self._vis([100, 101]), max_size=1.0)
                self.assertEqual(out["action"], "hold")
                self.assertIn("样本不足", out["reason"])

    def test_阈值必须为正(self):
        """⭐ 阈值为 0 ⇒ 每根都交易 ⇒ 那是"高频交手续费"，不是趋势跟随。"""
        with self.assertRaises(ValueError):
            Momentum(threshold=0.0)
        with self.assertRaises(ValueError):
            MeanRevert(threshold=-1.0)

    def test_random可复现(self):
        """⭐ 同一份输入 ⇒ 同一个决策（不靠全局推进的随机流）。"""
        a = RandomTaker(seed=3)
        b = RandomTaker(seed=3)
        vis = self._vis([100, 101, 102])
        self.assertEqual(a.decide(vis, max_size=1.0)["action"],
                         b.decide(vis, max_size=1.0)["action"])

    def test_make_policy过滤无关参数(self):
        """统一 kw 包不该炸掉没有那个字段的策略。"""
        p = make_policy("noop", seed=7, threshold=0.002)
        self.assertEqual(p.name, "noop")
        with self.assertRaises(KeyError):
            make_policy("不存在")

    def test_输出键必须规范(self):
        """⭐ 接口写错**不会报错**，只会让基线静默变成"从不交易"。"""
        with self.assertRaises(ValueError):
            validate_policy_output({"act": "buy", "size": 1})
        with self.assertRaises(ValueError):
            validate_policy_output({"action": "long"})
        validate_policy_output({"action": "hold", "sz": 0.0, "px": None,
                                "tp": None, "sl": None, "ordType": "market",
                                "confidence": 0.0, "reason": "x"})


# ======================================================================
# 完整回路
# ======================================================================
class TestRunAgentSession(unittest.TestCase):
    def test_noop收益恰好为零(self):
        """⭐⭐ **整套账本与执行层的自检。**

        不交易 ⇒ 收益必须**恰好** 0。它不为 0 说明记账有漏
        （漏记手续费 / 漏记滑点 / 根本没接上成交）。
        **先修账本，再谈策略。**
        """
        r = _run(AlwaysHold(), _series(n=50))
        self.assertEqual(r.final_equity, r.initial_equity)
        self.assertEqual(len(r.fills), 0)
        self.assertEqual(r.round_trips(), 0)

    def test_noop在有杠杆与成本下仍为零(self):
        r = _run(AlwaysHold(), _series(n=50, drift=0.01),
                 exec_cfg=ExecConfig(slippage_bps=50.0, cost_multiplier=5.0))
        self.assertEqual(r.final_equity, r.initial_equity)

    def test_账户状态回流到下一次决策(self):
        """⭐ A3 的跑批每次都看到"空仓 / 权益不变"——**那不算策略运行**。
        建仓后，下一次决策的 ``inventory`` 必须非零。"""
        r = _run(Momentum(threshold=0.001, lookback=6),
                 _series(n=60, drift=0.01, seed=4))
        invs = [float(rec.visible_state.get("inventory", 0.0)) for rec in r.records]
        self.assertTrue(any(abs(x) > 0 for x in invs),
                        "建仓后 visible_state 的 inventory 应当非零")
        eqs = {round(float(rec.visible_state.get("equity", 0.0)), 6)
               for rec in r.records}
        self.assertGreater(len(eqs), 1, "权益应当随行情变化，不是恒值")

    def test_权益曲线长度与根数对齐(self):
        """⚠️ 曲线**从成交前那格开始**，所以长度 = 根数 + 1。
        少一格会让夏普算错，多一格会让最后一根的收益被算两次。"""
        n = 40
        s = _series(n=n)
        r = run_agent_session(TradingAgent(policy=AlwaysHold(),
                                           config=AgentConfig(inst_id="X")),
                              s, account=_acc(), start=0, end=n - 1, run_id="T")
        self.assertEqual(len(r.equity_curve), n + 1)
        self.assertEqual(len(r.qty_curve), n + 1)

    def test_区间可从中间开始(self):
        s = _series(n=50)
        ag = TradingAgent(policy=AlwaysHold(), config=AgentConfig(inst_id="X"))
        r = run_agent_session(ag, s, account=_acc(), start=10, end=20, run_id="T")
        self.assertEqual([rec.tick for rec in r.records], list(range(10, 21)))

    def test_区间非法报错(self):
        ag = TradingAgent(policy=AlwaysHold(), config=AgentConfig(inst_id="X"))
        with self.assertRaises(ValueError):
            run_agent_session(ag, _series(n=10), account=_acc(),
                              start=8, end=2, run_id="T")

    def test_留痕足以重建订单(self):
        """⭐ 这是**留痕完整性**的硬检验：重建不出来 ⇒
        「事后能否复核这次下单」的答案是**不能**。"""
        r = _run(Momentum(threshold=0.001, lookback=6),
                 _series(n=60, drift=0.01, seed=4))
        n_exec = sum(1 for rec in r.records if rec.executed)
        self.assertGreater(n_exec, 0, "需要至少一次成交才能检验重建")
        for rec in r.records:
            req = order_request_from_record(rec)
            if rec.executed:
                self.assertIsNotNone(req)
                self.assertEqual(req.side, rec.order["side"])
                self.assertAlmostEqual(req.sz, float(rec.order["sz"]))
            else:
                self.assertIsNone(req)

    def test_订单缺字段时重建必须报错(self):
        rec = DecisionRecord(order={"instId": "X"})   # 缺一堆字段
        with self.assertRaises(ValueError):
            order_request_from_record(rec)

    def test_决策日志可落盘(self):
        import tempfile
        from pathlib import Path

        from tw.decision_log import DecisionLog

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "d.jsonl"
            log = DecisionLog(p).open("a")
            try:
                r = _run(AlwaysHold(), _series(n=20), log=log)
            finally:
                log.close()
            self.assertEqual(len(DecisionLog(p).read_all()), len(r.records))

    def test_规则模式被标进留痕(self):
        r = _run(AlwaysHold(), _series(n=10))
        self.assertEqual(r.records[0].mode, MODE_BASELINE)
        self.assertTrue(r.records[0].prompt_template.startswith("rule:"))
        self.assertEqual(r.records[0].model, "")
        self.assertIn("无 LLM", r.records[0].llm_raw)

    def test_llm模式仍可跑(self):
        """A4 不该把 A3 的路断掉。"""
        client = ScriptedClient(
            config=LLMConfig(provider="agnes"),
            responses=[LLMResponse(ok=True,
                                   text='{"action":"hold","reason":"x"}')],
            on_exhausted="hold",
        )
        ag = TradingAgent(client=client,
                          config=AgentConfig(inst_id="X", n_samples=1,
                                             temperature=0.0))
        r = run_agent_session(ag, _series(n=10), account=_acc(), run_id="T")
        self.assertEqual(len(r.records), 10)
        self.assertEqual(r.records[0].mode, "live")


# ======================================================================
# 分层评估
# ======================================================================
class TestResultLayer(unittest.TestCase):
    def test_毛净口径(self):
        """毛 = 净 + 手续费 + 滑点（把成本加回去）。"""
        r = _run(Momentum(threshold=0.001, lookback=6),
                 _series(n=60, drift=0.01, seed=4))
        L = result_layer(r)
        self.assertAlmostEqual(
            L["gross_pnl"], L["net_pnl"] + L["fees_paid"] + L["slippage_cost"],
            places=6)
        self.assertGreaterEqual(L["fees_paid"], 0.0)
        self.assertGreaterEqual(L["turnover"], 0.0)

    def test_noop的结果层全零(self):
        L = result_layer(_run(AlwaysHold(), _series(n=30)))
        for k in ("net_pnl", "gross_pnl", "fees_paid", "slippage_cost",
                  "turnover", "max_drawdown", "n_trades"):
            self.assertEqual(L[k], 0, k)

    def test_回合与持仓期(self):
        r = _run(Momentum(threshold=0.001, lookback=6),
                 _series(n=80, drift=0.01, seed=5))
        L = result_layer(r)
        self.assertGreaterEqual(L["n_round_trips"], 0)
        if L["n_round_trips"]:
            self.assertGreaterEqual(L["avg_holding_bars"], 0.0)
            self.assertLessEqual(L["exposure_frac"], 1.0)

    def test_夏普样本不足给nan不是零(self):
        """⭐ 0 会被读成"夏普为零"，而真相是"算不出来"。两者完全不同。"""
        self.assertTrue(math.isnan(sharpe([])))
        self.assertTrue(math.isnan(sharpe([0.01])))
        self.assertTrue(math.isnan(sharpe([0.01, 0.01, 0.01])))

    def test_最大回撤(self):
        self.assertAlmostEqual(max_drawdown([100, 120, 60, 90]), 0.5)
        self.assertEqual(max_drawdown([100, 110, 120]), 0.0)

    def test_逐根收益个数比曲线少一(self):
        eq = [100, 101, 102, 103]
        self.assertEqual(len(per_bar_returns(eq)), 3)


class TestDecisionAndExecutionLayer(unittest.TestCase):
    def test_弃权率与拒单率(self):
        d = decision_layer(_run(AlwaysHold(), _series(n=30)))
        self.assertAlmostEqual(d["abstain_frac"], 1.0)
        self.assertAlmostEqual(d["parse_fail_frac"], 0.0)

    def test_外部引用扫描能捞出幻觉(self):
        """⭐ 实测 v1/BTC 的理由写过「标普期货与恒指期货均上涨」，
        而可见状态里没有任何外部数据。这是关键词启发式，
        用途是**把可疑样本捞出来给人看**，不是自动给结论。"""
        recs = [
            DecisionRecord(parsed={"action": "hold", "reason": "标普期货与恒指期货均上涨"}),
            DecisionRecord(parsed={"action": "hold", "reason": "价格窄幅震荡，弃权"}),
        ]

        class _R:
            records = recs
        d = decision_layer(_R())
        self.assertEqual(d["external_ref_terms"].get("标普"), 1)
        self.assertEqual(d["external_ref_terms"].get("恒指"), 1)
        self.assertAlmostEqual(d["external_ref_frac"], 0.5)

    def test_成交率的分母是订单不是决策(self):
        """⭐ "弃权率高"不该被读成"成交率低"。"""
        r = _run(AlwaysHold(), _series(n=30))
        e = execution_layer(r)
        self.assertEqual(e["n_decisions"], 30)
        self.assertEqual(e["n_orders_built"], 0)
        self.assertEqual(e["n_decisions_not_traded"], 30)
        self.assertEqual(e["fill_rate_of_orders"], 0.0)

    def test_成交率不能超过百分之百(self):
        """⭐⭐ 真机报告里出现过的 bug：``llm_v2`` 的成交率是 **142%**。

        原因：分母是"落成的订单"（19），分子却用了**全部成交**（27）——
        而其中 8 笔是 **TP/SL 触发单**，它们**没有对应的已提交订单**。
        比率超过 100% 看起来只是"数字有点怪"，很容易被放过；
        但它说明**分子分母不同源**，而这个错误会随 TP/SL 使用率线性放大。

        ⇒ 分子必须只算"由订单产生的成交"（``market``/``limit``/…），
        TP/SL 与强平单独计。
        """
        # 造一个"订单少、触发单多"的场景：开仓带 TP，随后 TP 触发。
        s = _Series([100.0] * 8 + [120.0] * 6)

        class _TPPolicy:
            name = "tp_policy"

            def decide(self, visible, *, max_size=0.0):
                if max_size <= 0:
                    return {"action": "hold", "sz": 0.0, "px": None, "tp": None,
                            "sl": None, "ordType": "market", "confidence": 0.0,
                            "reason": "no capacity"}
                return {"action": "buy", "sz": float(max_size), "px": None,
                        "tp": float(visible["mid"]) * 1.05, "sl": None,
                        "ordType": "market", "confidence": 0.5,
                        "reason": "带止盈做多"}

        r = _run(_TPPolicy(), s, exec_cfg=ExecConfig(slippage_bps=0.0))
        e = execution_layer(r)
        self.assertGreaterEqual(e["n_orders_built"], 1)
        self.assertLessEqual(e["fill_rate_of_orders"], 1.0,
                             f"成交率 {e['fill_rate_of_orders']} 不能超过 100%")
        self.assertGreaterEqual(e["n_exit_fills_tp_sl"], 0)
        # 总量守恒：订单成交 + 触发单成交 + 强平成交 = 全部成交
        self.assertEqual(
            e["n_order_fills"] + e["n_exit_fills_tp_sl"] + e["n_liquidate_fills"],
            e["n_fills"])

    def test_执行层报滑点(self):
        r = _run(Momentum(threshold=0.001, lookback=6),
                 _series(n=60, drift=0.01, seed=4, spread=0.004),
                 exec_cfg=ExecConfig(slippage_bps=5.0))
        e = execution_layer(r)
        if e["n_fills"]:
            self.assertGreater(e["slippage_bps_mean"], 0.0)
            self.assertGreater(e["slippage_cost_total"], 0.0)


class TestCostSensitivity(unittest.TestCase):
    def test_解析式是近似且被标注(self):
        r = _run(Momentum(threshold=0.001, lookback=6),
                 _series(n=60, drift=0.01, seed=4))
        cs = cost_sensitivity_analytic(r)
        self.assertEqual([c["cost_multiplier"] for c in cs], [1.0, 2.0, 5.0])
        for c in cs:
            self.assertEqual(c["method"], "analytic_approx")
        # ×1 必须恰好等于净收益
        self.assertAlmostEqual(cs[0]["net_pnl"], r.final_equity - r.initial_equity,
                               places=6)
        # 成本越高越差（单调）
        self.assertGreater(cs[0]["net_pnl"], cs[2]["net_pnl"])

    def test_重跑版真的重跑(self):
        """⭐ 成本变了，策略**可能改变行为**（本项目实测出现过
        交易笔数从 43 掉到 41）——所以不能只在旧收益上减一个数。"""
        runs = {}
        s = _series(n=80, drift=0.01, seed=4)
        for m in (1.0, 2.0, 5.0):
            runs[m] = _run(Momentum(threshold=0.001, lookback=6), s,
                           exec_cfg=ExecConfig(slippage_bps=1.0,
                                               cost_multiplier=m))
        cs = cost_sensitivity_rerun(runs)
        for c in cs:
            self.assertEqual(c["method"], "rerun")
        # 成本 ×1 时重跑版与解析式应当接近（同一件事的两种算法）
        an = cost_sensitivity_analytic(runs[1.0])
        self.assertAlmostEqual(cs[0]["net_pnl"], an[0]["net_pnl"], places=6)

    def test_成本倍数越高净收益越低(self):
        s = _series(n=80, drift=0.01, seed=4)
        pnl = {}
        for m in (1.0, 5.0):
            r = _run(Momentum(threshold=0.001, lookback=6), s,
                     exec_cfg=ExecConfig(slippage_bps=1.0, cost_multiplier=m))
            pnl[m] = r.final_equity - r.initial_equity
        self.assertGreater(pnl[1.0], pnl[5.0])

    def test_回放覆盖率过低时拒绝出数字(self):
        """⭐⭐ 这是本轮最重要的方法论修正。

        **LLM 的 prompt 里含账户状态**（持仓、权益），而账户状态取决于
        成交、成交取决于成本 ⇒ 换个成本重放同一段行情会让路径**迅速发散**。

        实测（BTC，成本 ×1 vs ×2）：
          ×1：回放覆盖率 **100%**（逐笔复现真机），交易 27 笔，净 −33
          ×2：回放覆盖率 **0.8%**（363 次调用只命中 3 次），交易 **2** 笔，净 −444
          → 那个 −444 **看起来像一个成本敏感性结果，其实是"回放失败"**。

        所以本函数必须在覆盖率过低时**拒绝出数字**——
        宁可报「测不出来」，也不要一个漂亮的假结论。
        """
        s = _series(n=40, drift=0.01, seed=4)
        good = _run(Momentum(threshold=0.001, lookback=6), s,
                    exec_cfg=ExecConfig(cost_multiplier=1.0))
        good.replay_coverage = 1.0
        bad = _run(Momentum(threshold=0.001, lookback=6), s,
                   exec_cfg=ExecConfig(cost_multiplier=2.0))
        bad.replay_coverage = 0.008          # 实测值

        rows = {c["cost_multiplier"]: c for c in
                cost_sensitivity_rerun({1.0: good, 2.0: bad})}
        self.assertTrue(rows[1.0]["valid"])
        self.assertFalse(rows[2.0]["valid"], "覆盖率 0.8% 必须判无效")
        self.assertIn("回放覆盖率", rows[2.0]["invalid_reason"])
        # ⚠️ 数字**要留着**（删掉会让人以为"这一档没跑"），只是标无效
        self.assertIn("net_return", rows[2.0])

    def test_非回放的运行不受覆盖率检查影响(self):
        """规则基线不含路径依赖 ⇒ 没有 replay_coverage ⇒ 一律有效。"""
        s = _series(n=40, seed=4)
        r = _run(Momentum(threshold=0.001, lookback=6), s)
        self.assertIsNone(r.replay_coverage)
        rows = cost_sensitivity_rerun({1.0: r})
        self.assertTrue(rows[0]["valid"])


class TestComparison(unittest.TestCase):
    def test_相对noop用CI排除检验(self):
        """⭐⭐ **A4 最核心的那个问题：能不能打赢 noop。**

        noop 的收益恒为 0 ⇒ 它的区间宽度为 0 ⇒ **重叠比例没有意义**。
        最初一律走重叠检验，于是这条输出「无法判定（输入退化）」——
        用错检验会让最关键的问题**答不出来**，而且看起来像"数据不够"。
        正确做法是问"目标的 CI 排不排除 0"。
        """
        s = _series(n=120, drift=0.004, seed=11)
        noop = _run(AlwaysHold(), s, name="noop")
        long_pol = Momentum(threshold=0.0005, lookback=6)
        tgt = _run(long_pol, s, name="tgt")
        cmp = compare_to_baselines(tgt, {"noop": noop})
        v = cmp["verdicts"][0]
        self.assertEqual(v["test"], "ci_excludes_baseline")
        self.assertIn(v["verdict"], ("显著优于", "显著劣于", "依然无法判定"))
        self.assertEqual(v["point_b"], 0.0)

    def test_有方差基线用重叠检验(self):
        s = _series(n=120, drift=0.004, seed=11)
        a = _run(Momentum(threshold=0.001, lookback=6), s, name="m")
        b = _run(MeanRevert(threshold=0.001, lookback=6), s, name="r")
        cmp = compare_to_baselines(a, {"r": b})
        v = cmp["verdicts"][0]
        self.assertEqual(v["test"], "ci_overlap")
        self.assertIn(v["verdict"], ("显著", "依然无法判定", "边缘"))

    def test_自己比自己无法判定(self):
        """⭐ 已知限制的自检：同一条曲线 vs 自己必须判"无法判定"，
        否则说明检验的尺度是错的。"""
        s = _series(n=100, seed=3)
        a = _run(Momentum(threshold=0.001, lookback=6), s, name="a")
        cmp = compare_to_baselines(a, {"b": a})
        self.assertEqual(cmp["verdicts"][0]["verdict"], "依然无法判定")

    def test_自助法可复现(self):
        rets = [0.001 * ((i % 7) - 3) for i in range(200)]
        self.assertEqual(block_bootstrap_ci(rets, seed=1),
                         block_bootstrap_ci(rets, seed=1))

    def test_自助法样本太少给nan(self):
        lo, hi = block_bootstrap_ci([])
        self.assertTrue(math.isnan(lo))


class TestConfidenceCalibration(unittest.TestCase):
    def test_校准能发现过度自信(self):
        """⭐ "说 0.7 的是不是真 70% 对"。

        构造：confidence 恒 0.9 但方向只有一半对（随机游走）。
        期望 ``gap`` 明显为负（过度自信）。
        """
        s = _series(n=100, seed=9)
        client = ScriptedClient(
            config=LLMConfig(provider="agnes"),
            responses=[], on_exhausted="hold",
        )
        # 直接造记录更可控：不经过 LLM
        recs = []
        for i in range(0, 60):
            recs.append(DecisionRecord(
                tick=i, parsed={"action": "buy", "confidence": 0.9,
                                "reason": "看涨理由充分"},
                parse_ok=True, executed=True,
            ))

        class _R:
            records = recs

        cal = confidence_calibration(_R(), [float(x) for x in s.close])
        self.assertGreater(cal["n"], 0)
        self.assertAlmostEqual(cal["overall_predicted"], 0.9)
        # 随机游走上"一直看涨"的准确率不该接近 0.9
        self.assertLess(cal["overall_realized"], 0.9)

    def test_弃权不参与校准(self):
        recs = [DecisionRecord(tick=0, parsed={"action": "hold",
                                               "confidence": 0.9})]

        class _R:
            records = recs

        cal = confidence_calibration(_R(), [100.0] * 50)
        self.assertEqual(cal["n"], 0)

    def test_前瞻窗口不够时跳过(self):
        recs = [DecisionRecord(tick=48, parsed={"action": "buy",
                                                "confidence": 0.5})]

        class _R:
            records = recs

        cal = confidence_calibration(_R(), [100.0] * 50, horizon=4)
        self.assertEqual(cal["n"], 0)


class TestEvaluateRun(unittest.TestCase):
    def test_总报告含三层(self):
        r = _run(Momentum(threshold=0.001, lookback=6),
                 _series(n=60, drift=0.01, seed=4))
        e = evaluate_run(r, closes=[float(x) for x in _series(n=60).close])
        for k in ("decision", "execution", "result",
                  "cost_sensitivity_analytic", "exec_config", "meta"):
            self.assertIn(k, e)
        self.assertIn("confidence_calibration", e)

    def test_不带closes时跳过校准(self):
        r = _run(AlwaysHold(), _series(n=20))
        e = evaluate_run(r)
        self.assertNotIn("confidence_calibration", e)


if __name__ == "__main__":
    unittest.main(verbosity=2)
