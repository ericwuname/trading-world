"""A6 测试：KPI（目标 + 约束 + 两种退化的防御）。

核心是用户那句话的**可操作化**：

> 「不能因为达成KPI就不继续，亏损扩大就不交易，要平稳交易下去才行。」

⇒ 单指标一定会被刷，所以 KPI 必须是一组**互相咬住**的约束。
本文件守住三件事：
1. **防躺平**：靠"不交易"绝不能达标（在场率下限）；
2. **防赌博**：靠"加大仓位追目标"绝不能达标（回撤上限）；
3. **进度不含未来**：KPI 进度里每一个量都必须是"当时能算出来"的。
"""

from __future__ import annotations

import unittest

from tw.kpi import (
    KPIConfig,
    KPIState,
    advance,
    calibrate_from_baseline,
    kpi_verdict,
)


def _state(**kw) -> KPIState:
    base = dict(initial_equity=100_000.0, equity=100_000.0,
                peak_equity=100_000.0, bars_done=10, bars_total=50,
                bars_in_position=3, turnover_x=2.0)
    base.update(kw)
    return KPIState(**base)


# ======================================================================
# ⭐ 防躺平：不交易绝不能达标
# ======================================================================
class TestAntiLyingFlat(unittest.TestCase):
    def test_全弃权不达标(self):
        """⭐⭐ **这是用户点名要防的第一种退化。**

        "亏损扩大就不交易"之所以能刷分，是因为**不交易就不会亏**。
        ⇒ 必须在 KPI 里加"在场率下限"，让零暴露策略**直接不达标**。
        """
        kpi = KPIConfig(target_return=0.0, min_presence=0.30)
        st = _state(bars_done=50, bars_in_position=0, equity=100_000.0)
        v = kpi_verdict(kpi, st)
        self.assertFalse(v["passed"])
        self.assertIn("presence", v["failed"])

    def test_在场率达标就能过这一条(self):
        kpi = KPIConfig(target_return=0.0, min_presence=0.30,
                        turnover_lo=0.0)
        st = _state(bars_done=50, bars_in_position=20)   # 40% > 30%
        self.assertTrue(kpi_verdict(kpi, st)["detail"]["presence"]["ok"])

    def test_在场率恰好等于下限算达标(self):
        """边界：`>=` 而不是 `>`（与项目其余部分的口径一致）。"""
        kpi = KPIConfig(target_return=0.0, min_presence=0.30, turnover_lo=0.0)
        st = _state(bars_done=50, bars_in_position=15)   # 恰 30%
        self.assertTrue(kpi_verdict(kpi, st)["detail"]["presence"]["ok"])

    def test_换手下限挡住极端轻仓(self):
        """在场率高但每根只碰一点点，也该被换手下限拦住一部分。"""
        kpi = KPIConfig(target_return=0.0, min_presence=0.0, turnover_lo=1.0)
        st = _state(turnover_x=0.1)
        self.assertFalse(kpi_verdict(kpi, st)["detail"]["turnover_lo"]["ok"])


# ======================================================================
# ⭐ 防赌博：加大仓位追目标要能被拦住
# ======================================================================
class TestAntiGambling(unittest.TestCase):
    def test_回撤超限不达标(self):
        """⭐⭐ **用户点名要防的第二种退化**：落后目标就加杠杆去追。"""
        kpi = KPIConfig(target_return=0.05, max_drawdown=0.05)
        st = _state(equity=90_000.0, peak_equity=100_000.0)   # 回撤 10%
        v = kpi_verdict(kpi, st)
        self.assertFalse(v["passed"])
        self.assertIn("drawdown", v["failed"])

    def test_回撤恰好等于上限算达标(self):
        kpi = KPIConfig(target_return=0.0, max_drawdown=0.05,
                        min_presence=0.0, turnover_lo=0.0)
        st = _state(equity=95_000.0, peak_equity=100_000.0)   # 恰 5%
        self.assertTrue(kpi_verdict(kpi, st)["detail"]["drawdown"]["ok"])

    def test_换手上限挡住乱枪打鸟(self):
        kpi = KPIConfig(target_return=0.0, min_presence=0.0,
                        turnover_hi=10.0)
        st = _state(turnover_x=80.0)
        self.assertFalse(kpi_verdict(kpi, st)["detail"]["turnover_hi"]["ok"])


# ======================================================================
# 两种退化**同时**被咬住（这是设计的要点）
# ======================================================================
class TestBothDegenerationsBlocked(unittest.TestCase):
    def test_躺平与赌博都不达标(self):
        """⭐ 单指标一定会被刷；一组互相咬住的约束才防得住。"""
        kpi = KPIConfig(target_return=0.02, min_presence=0.30,
                        max_drawdown=0.05, turnover_lo=1.0, turnover_hi=40.0)

        lie_flat = _state(bars_done=50, bars_in_position=0,
                          equity=100_000.0, turnover_x=0.0)
        gambling = _state(bars_done=50, bars_in_position=40,
                          equity=88_000.0, peak_equity=100_000.0,
                          turnover_x=200.0)

        v1 = kpi_verdict(kpi, lie_flat)
        v2 = kpi_verdict(kpi, gambling)
        self.assertFalse(v1["passed"], "躺平不该达标")
        self.assertFalse(v2["passed"], "赌博不该达标")
        self.assertIn("presence", v1["failed"])
        self.assertIn("drawdown", v2["failed"])

    def test_平稳交易能达标(self):
        """正向：在场率够、回撤小、换手在区间内 ⇒ 达标。"""
        kpi = KPIConfig(target_return=0.005, min_presence=0.30,
                        max_drawdown=0.05, turnover_lo=1.0, turnover_hi=40.0)
        st = _state(bars_done=50, bars_in_position=25,
                    equity=103_000.0, peak_equity=104_000.0, turnover_x=8.0)
        v = kpi_verdict(kpi, st)
        self.assertTrue(v["passed"], f"应当达标，未过：{v['failed']}")


# ======================================================================
# ⭐ 进度不含未来
# ======================================================================
class TestStateHasNoFuture(unittest.TestCase):
    def test_advance只吃已经发生的量(self):
        """⭐⭐ KPI 进度里每个量都必须是"当时能算出来"的。

        `advance` 的入参只有"当前已发生的权益/持仓/成交额"——
        **没有任何未来的量**，所以"含未来信息"在结构上就不可能。
        这与 `visible_state` 是同一条纪律。
        """
        st = KPIState(initial_equity=100_000.0, equity=100_000.0,
                      peak_equity=100_000.0, bars_done=0, bars_total=10)
        advance(st, equity=101_000.0, qty=1.0, turnover_delta=50_000.0)
        self.assertEqual(st.bars_done, 1)
        self.assertAlmostEqual(st.return_so_far, 0.01, places=9)
        self.assertAlmostEqual(st.presence_so_far, 1.0)
        self.assertAlmostEqual(st.turnover_x, 0.5)

    def test_回撤只看历史峰值(self):
        st = KPIState(initial_equity=100.0, equity=100.0, peak_equity=100.0,
                      bars_total=10)
        advance(st, equity=110.0, qty=0.0)
        advance(st, equity=99.0, qty=0.0)
        self.assertAlmostEqual(st.peak_equity, 110.0)
        self.assertAlmostEqual(st.drawdown_so_far, 0.1, places=9)

    def test_空仓不计入在场(self):
        st = KPIState(initial_equity=100.0, equity=100.0, peak_equity=100.0,
                      bars_total=5)
        advance(st, equity=100.0, qty=0.0)
        advance(st, equity=100.0, qty=1.0)
        self.assertAlmostEqual(st.presence_so_far, 0.5)

    def test_bars_left不越界(self):
        st = KPIState(bars_done=10, bars_total=5)
        self.assertEqual(st.bars_left, 0)


# ======================================================================
# 差距提示（让模型知道该不该冒进）
# ======================================================================
class TestGaps(unittest.TestCase):
    def test_落后目标被标出(self):
        kpi = KPIConfig(target_return=0.05)
        g = _state(equity=100_000.0).gaps(kpi)
        self.assertTrue(g["behind_target"])
        self.assertGreater(g["return_gap"], 0)

    def test_太躺被标出(self):
        kpi = KPIConfig(min_presence=0.50)
        g = _state(bars_done=10, bars_in_position=1).gaps(kpi)
        self.assertTrue(g["too_flat"])

    def test_接近回撤上限被标出(self):
        """⭐ 这条是给模型的**预警**：它该知道"再加风险会不达标"。"""
        kpi = KPIConfig(max_drawdown=0.05)
        g = _state(equity=95_500.0, peak_equity=100_000.0).gaps(kpi)
        self.assertTrue(g["near_dd_limit"])

    def test_达标时不标落后(self):
        kpi = KPIConfig(target_return=0.01, min_presence=0.1)
        g = _state(equity=105_000.0, bars_done=10,
                   bars_in_position=5).gaps(kpi)
        self.assertFalse(g["behind_target"])
        self.assertFalse(g["too_flat"])


# ======================================================================
# 配置
# ======================================================================
class TestConfig(unittest.TestCase):
    def test_在场率必须在0到1(self):
        for bad in (-0.1, 1.5):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    KPIConfig(min_presence=bad)

    def test_回撤上限必须为正(self):
        with self.assertRaises(ValueError):
            KPIConfig(max_drawdown=0.0)

    def test_换手区间必须有序(self):
        with self.assertRaises(ValueError):
            KPIConfig(turnover_lo=10.0, turnover_hi=1.0)

    def test_描述可写进留痕(self):
        d = KPIConfig(target_return=0.02).describe()
        self.assertEqual(d["target_return"], 0.02)
        self.assertIn("min_presence", d)


# ======================================================================
# 阈值标定（**用基线分布，不拍脑袋**）
# ======================================================================
class TestCalibration(unittest.TestCase):
    def test_按分位数标定(self):
        presence = [i / 100 for i in range(1, 101)]      # 0.01 … 1.00
        turnover = [float(i) for i in range(1, 101)]
        dd = [i / 100 for i in range(1, 101)]
        c = calibrate_from_baseline(presence=presence, turnover=turnover,
                                    drawdown=dd)
        # 20% 分位 ≈ 0.20
        self.assertLess(abs(c["min_presence"] - 0.20), 0.02)
        self.assertLess(c["turnover_lo"], c["turnover_hi"])
        self.assertIn("不是拍的", c["note"])

    def test_空输入不崩(self):
        c = calibrate_from_baseline(presence=[], turnover=[], drawdown=[])
        self.assertTrue(c["min_presence"] != c["min_presence"])   # nan


# ======================================================================
# ⭐⭐ 端到端：KPI 真的进了 prompt 与留痕吗
# ======================================================================
class TestKPIIsWiredEndToEnd(unittest.TestCase):
    """⭐⭐ 由 M102 漏网暴露出的**真测试缺口**。

    我原来只测了 `KPIConfig`/`KPIState`/`kpi_verdict` 这些**纯函数**，
    却**从来没跑过一次带 KPI 的会话**。
    ⇒ 变异体把"建 KPI 状态"那一步关掉（`if False`）时，**没有任何测试发现**。

    后果有多严重：prompt 里没有考核目标、`meta` 里没有 kpi_state，
    而**实验会照跑、照出数字** ⇒ 最后得出「KPI 无效」的**假结论**。
    这正是本项目一路在防的那类错误。
    """

    @staticmethod
    def _series(n=60, seed=5):
        import numpy as np

        class S:
            def __init__(s, c):
                s.close = np.asarray(c, dtype=float)
                s.open = s.close
                s.high = s.close * 1.002
                s.low = s.close * 0.998
                s.timestamp = np.arange(len(c), dtype=np.int64) * 3_600_000

        rng = np.random.default_rng(seed)
        return S(100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.004, n))))

    def _run(self, *, kpi, template="v5", n=48):
        import numpy as np

        from tw.account import MarginAccount, MarginConfig
        from tw.agent import AgentConfig, TradingAgent
        from tw.agent_run import run_agent_session
        from tw.llm import LLMConfig, LLMResponse, ScriptedClient
        from tw.simexec import ExecConfig

        series = self._series(n + 12)
        client = ScriptedClient(
            config=LLMConfig(provider="agnes"), on_exhausted="hold",
            responses=[LLMResponse(ok=True,
                                   text='{"action":"hold","reason":"观望"}')] * 200)
        ag = TradingAgent(
            client=client,
            config=AgentConfig(inst_id="X", template=template, kpi=kpi,
                               n_samples=1, temperature=0.0))
        acc = MarginAccount(cash=100_000.0, cfg=MarginConfig())
        res = run_agent_session(ag, series, account=acc, start=12, end=12 + n - 1,
                                run_id="T", exec_config=ExecConfig(slippage_bps=1.0))
        # 把实际发出去的 prompt 拿回来（ScriptedClient 记了 calls）
        prompts = ["".join(str(m.get("content", "")) for m in c)
                   for c in client.calls]
        return res, prompts, client

    def test_prompt里真的有考核目标(self):
        """⭐ KPI 段的文字必须真的出现在 prompt 里。"""
        kpi = KPIConfig(target_return=0.02, min_presence=0.35,
                        max_drawdown=0.04)
        _res, prompts, _c = self._run(kpi=kpi)
        self.assertTrue(prompts, "没有记录到 prompt")
        body = prompts[0]
        self.assertIn("考核目标", body)
        self.assertIn("+2.00%", body, "净收益目标没渲染进去")
        self.assertIn("35%", body, "在场率下限没渲染进去")
        self.assertIn("4.00%", body, "回撤上限没渲染进去")
        self.assertIn("在场率", body)

    def test_不给KPI时prompt里就说没有(self):
        _res, prompts, _c = self._run(kpi=None)
        self.assertIn("未设置考核目标", prompts[0])

    def test_进度真的在推进(self):
        """⭐ 进度不推进 ⇒ 模型永远以为自己在第 1 根。"""
        kpi = KPIConfig()
        _res, prompts, _c = self._run(kpi=kpi, n=10)
        self.assertIn("第 1/10 根", prompts[0])
        self.assertIn("第 5/10 根", prompts[4])
        self.assertIn("第 10/10 根", prompts[9])
        self.assertIn("还剩 **0** 根", prompts[9])

    def test_meta里有KPI判定(self):
        """⭐ 没有它，实验跑完也拿不到"KPI 过没过"。"""
        kpi = KPIConfig()
        res, _p, _c = self._run(kpi=kpi, n=20)
        self.assertIn("kpi", res.meta)
        self.assertIn("kpi_final", res.meta)
        self.assertIn("kpi_state", res.meta)
        self.assertIn("passed", res.meta["kpi_final"])
        self.assertEqual(res.meta["kpi_state"]["bars_total"], 20)

    def test_全弃权必KPI不过(self):
        """⭐⭐ 端到端地把"防躺平"坐实。

        Agent 全程弃权（脚本给的就是 hold）⇒ 在场率 0 ⇒ **KPI 必须判不过**。
        若这条通过了，说明"在场率下限"没接上——而那正是防躺平的那一条。
        """
        kpi = KPIConfig(target_return=0.0, min_presence=0.30)
        res, _p, _c = self._run(kpi=kpi, n=20)
        self.assertFalse(res.meta["kpi_final"]["passed"],
                         "全弃权不该判过 —— 防躺平那条没生效")
        self.assertIn("presence", res.meta["kpi_final"]["failed"])

    def test_KPI进decision_id(self):
        """⭐ 有 KPI 与无 KPI 是**两个实验**，不该算出同一个 ID。"""
        vis = {"mid": 100.0, "equity": 100_000.0, "recent_closes": [1.0, 2.0]}
        a = self._decide_once(vis, kpi=None)
        b = self._decide_once(vis, kpi=KPIConfig())
        self.assertNotEqual(a.decision_id, b.decision_id,
                            "KPI 是实验变量，必须进 decision_id")
        self.assertIsNone(a.model_params.get("kpi"))
        self.assertIn("target_return", b.model_params.get("kpi") or {})

    @staticmethod
    def _decide_once(vis, *, kpi):
        from tw.agent import AgentConfig, TradingAgent
        from tw.llm import LLMConfig, LLMResponse, ScriptedClient

        client = ScriptedClient(
            config=LLMConfig(provider="agnes"), on_exhausted="hold",
            responses=[LLMResponse(ok=True,
                                   text='{"action":"hold","reason":"x"}')] * 5)
        ag = TradingAgent(client=client,
                          config=AgentConfig(inst_id="X", kpi=kpi,
                                             n_samples=1, temperature=0.0))
        return ag.decide(visible=vis, tick=1, run_id="R", equity=100_000.0)


# ======================================================================
# ⭐⭐ v6：把「单一变量」从一句承诺变成**可验证的等式**
# ======================================================================
class TestV6IsStrictSingleVariable(unittest.TestCase):
    """⚠️ 由一次**实测发现**的真缺陷驱动（2026-09-21）。

    `v5` 的注册注释写着「与 v4 的差别**只有一个变量**：多了一段考核目标」。
    逐行 diff 后发现**这是错的**：v5 还顺手重写了「怎么选」那段指令
    （并删掉了「这些指标由确定性代码从下面的收盘价算出…」一行）。

    ⇒ `v4 → v5` 是**两处改动**：
      「KPI 有没有用」这个对照**不干净**——入场率 45.8% → 85.0% 里，
      有一部分来自指令改写，不是 KPI。

    处置（按纪律「改模板要新增版本而不是原地改」）：
    保留 v5 原样（它已被录制过，改动会让回放与历史结论失效），
    新增 **v6 = v4 + 考核目标段**，并用本组测试机械守住"只有这一个变量"。
    """

    def test_v6去掉那一段就是v4(self):
        """⭐⭐ **本组的核心等式**（也是 v6 存在的全部理由）。

        `_USER_V6` 去掉 `_KPI_SECTION` 后必须与 `_USER_V4` **逐字节相等**。
        ⇒ "只多了一段"从一句承诺变成一个能在测试里跑通的等式。
        """
        from tw.prompts import _KPI_SECTION, _USER_V4, _USER_V6
        self.assertNotEqual(_USER_V6, _USER_V4)
        self.assertEqual(_USER_V6.replace(_KPI_SECTION, "", 1), _USER_V4)

    def test_v5不是单一变量扩展(self):
        """⚠️ **把缺陷本身固化成测试**。

        断言 v5 去掉 KPI 段**不等于** v4 ⇒ 以后没人可以再写
        「v5 与 v4 只差一个变量」而不被测试打脸。
        """
        from tw.prompts import _KPI_SECTION, _USER_V4, _USER_V5
        self.assertIn(_KPI_SECTION, _USER_V5)
        self.assertNotEqual(_USER_V5.replace(_KPI_SECTION, "", 1), _USER_V4)

    def test_v6的锚点真的存在(self):
        """⚠️ `str.replace` 锚点失效时**静默返回原串** ⇒ v6 退化成 v4。

        `tw/prompts.py` 里已有 import 期 assert；这里再独立守一道，
        因为"锚点失效"是**静默**的，只看代码看不出来。
        """
        from tw.prompts import _USER_V4, _V6_ANCHOR
        self.assertIn(_V6_ANCHOR, _USER_V4)

    def test_v6版号已注册且system与v4一致(self):
        """v6 与 v4 必须共用同一个 system 提示词——否则又多了一个变量。"""
        from tw.prompts import TEMPLATES
        self.assertIn("v6", TEMPLATES)
        self.assertEqual(TEMPLATES["v6"]["system"], TEMPLATES["v4"]["system"])

    def test_v6带KPI时正文含四个考核项(self):
        """渲染层面再验一次：四类约束（收益/在场率/回撤/换手）都在。"""
        from tw.kpi import KPIConfig, KPIState
        from tw.prompts import build_messages
        k = KPIConfig(target_return=0.01, min_presence=0.5,
                      max_drawdown=0.02, turnover_lo=1.7, turnover_hi=3.7)
        ms = build_messages({"mid": 100.0}, inst_id="X", bar="1H",
                            template="v6", kpi=k, kpi_state=KPIState())
        body = " ".join(str(m.get("content", "")) for m in ms)
        for token in ("净收益目标", "在场率下限", "最大回撤上限", "换手区间",
                      "不是「及格线」"):
            self.assertIn(token, body)

    def test_v6不给KPI时明说未设置(self):
        from tw.prompts import build_messages
        ms = build_messages({"mid": 100.0}, inst_id="X", bar="1H",
                            template="v6", kpi=None, kpi_state=None)
        body = " ".join(str(m.get("content", "")) for m in ms)
        self.assertIn("未设置", body)


# ======================================================================
# ⭐⭐ 收益目标必须**随窗口长度缩放**（第 16 个真缺陷）
# ======================================================================
class TestCalibratedTargetReturn(unittest.TestCase):
    """由「48/48 段全部 `return` 失败」暴露出来的缺陷（2026-09-22）。

    原实现把 `target_return` 硬编码成 `+1%`——**绝对值、不随窗口长度变**。
    窗口只有 8 根时要求 8 根赚 1% ⇒ **一段都达不到**，
    模型每一次都被告知"你还差得远"。

    ⇒ 那批实验实际测的是「**目标不可达**时它会怎么做」，
      而不是「给目标好不好」。
    ⭐ **这是被数据自己暴露的**：一个"目标"若 100% 不达标，
      那它就不是目标，是噪音。

    修法：与其余三个阈值**同源**——用基线在**同一窗口长度**下的净收益分位。
    """

    NETS = [-0.010, -0.004, 0.000, 0.002, 0.006, 0.011]

    def _cal(self, **kw):
        from tw.kpi import calibrate_from_baseline
        base = dict(presence=[0.4, 0.6, 0.5], turnover=[0.3, 0.6, 0.9],
                    drawdown=[0.01, 0.02, 0.03], nets=self.NETS)
        base.update(kw)
        return calibrate_from_baseline(**base)

    def test_目标取净收益分位(self):
        cal = self._cal()
        self.assertFalse(cal["target_return_missing"])
        # 中位数分位（取整法）落在 0.000 或 0.002 上
        self.assertIn(cal["target_return"], (0.0, 0.002))

    def test_目标随窗口长度线性缩放(self):
        """⭐⭐ **本组的核心**：目标与 `nets` **同单位** ⇒ nets 按 L 缩放，
        目标就按 L 缩放。这正是原实现缺的性质。
        """
        a = self._cal(nets=[x * 1 for x in self.NETS])["target_return"]
        b = self._cal(nets=[x * 5 for x in self.NETS])["target_return"]
        if a == 0.0:
            # 中位数恰好是 0 时比值无意义 ⇒ 换个分位再验
            a = self._cal(nets=[x for x in self.NETS], target_q=0.9)["target_return"]
            b = self._cal(nets=[x * 5 for x in self.NETS],
                          target_q=0.9)["target_return"]
        self.assertAlmostEqual(b / a, 5.0, places=6)

    def test_目标q越高目标越高(self):
        lo = self._cal(target_q=0.2)["target_return"]
        hi = self._cal(target_q=0.9)["target_return"]
        self.assertGreater(hi, lo)

    def test_没有nets时明确报缺失而不是编一个(self):
        """⚠️ **不许静默退回固定值**——那正是要修的东西。"""
        cal = self._cal(nets=None)
        self.assertTrue(cal["target_return_missing"])
        self.assertNotEqual(cal["target_return"], cal["target_return"])  # nan

    def test_全负时标记无约束力(self):
        cal = self._cal(nets=[-0.02, -0.01, -0.005])
        self.assertTrue(cal["target_return_not_positive"])

    def test_基线自己有一半段达标(self):
        """⭐⭐ **"永远可达"这条性质要能被测**。

        目标 = 基线净收益的中位数 ⇒ **至少一半的基线段达标**。
        若有人把目标改回固定值（或改错分位方向），这条会变红。
        """
        from tw.kpi import KPIConfig, KPIState, kpi_verdict
        cal = self._cal()
        kpi = KPIConfig(target_return=cal["target_return"],
                        min_presence=0.0, max_drawdown=1.0,
                        turnover_lo=0.0, turnover_hi=1e9)
        ok = 0
        for net in self.NETS:
            st = KPIState(initial_equity=100.0, equity=100.0 * (1 + net),
                          bars_done=8, bars_total=8)
            if kpi_verdict(kpi, st)["detail"]["return"]["ok"]:
                ok += 1
        self.assertGreaterEqual(ok, len(self.NETS) // 2)

    def test_固定目标比标定目标苛刻得多(self):
        """⚠️ **把缺陷本身写成断言**：同一批段，固定 +1% 目标的达标数
        必须**显著少于**标定目标。这样有人把它改回固定值会立刻被发现。

        ⚠️ 我第一版写的是「固定目标下一段都不该达标」——**错了**：
        这组样本里恰好有一段 1.1% > 1%。断言要写成**关系**，
        不要写成"我以为的具体数字"（本项目"断言存在 ≠ 有分辨力"的老毛病）。
        """
        from tw.kpi import KPIConfig, KPIState, kpi_verdict

        def n_ok(tgt):
            kpi = KPIConfig(target_return=tgt, min_presence=0.0,
                            max_drawdown=1.0, turnover_lo=0.0,
                            turnover_hi=1e9)
            c = 0
            for net in self.NETS:
                st = KPIState(initial_equity=100.0,
                              equity=100.0 * (1 + net),
                              bars_done=8, bars_total=8)
                if kpi_verdict(kpi, st)["detail"]["return"]["ok"]:
                    c += 1
            return c

        n_fixed = n_ok(0.01)
        n_cal = n_ok(self._cal()["target_return"])
        self.assertLess(n_fixed, n_cal)
        self.assertGreaterEqual(n_cal, len(self.NETS) // 2)

    def test_标定的其余三项不受影响(self):
        """⚠️ 加参数**不能改已有的行为**（回归护栏）。"""
        a = self._cal(nets=None)
        b = self._cal()
        for k in ("min_presence", "turnover_lo", "turnover_hi",
                  "max_drawdown", "min_presence_vacuous",
                  "turnover_lo_vacuous"):
            self.assertEqual(a[k], b[k], f"{k} 变了")


if __name__ == "__main__":
    unittest.main(verbosity=2)
