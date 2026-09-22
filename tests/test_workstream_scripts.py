"""三条工作线的**实验装配件**回归测试。

为什么实验脚本也要进测试范围
----------------------------
本项目已经吃过这个亏（阶段5 的 E5.4 挑选函数里一个 nan 排序键静默劫持 argmin，
让脚本选中了错误的档位并写下与数据相反的结论）。此后确立的规矩是：
**凡"决定报告里写什么"的逻辑，都要进测试**，哪怕它住在 scripts/ 里。

本文件覆盖三件最容易静默出错的事：
  · 元订单**执行期不能同时走基础逻辑**（否则活跃度翻倍）—— M36
  · 基本面派的子单方向必须**锚定启动那一刻**（否则"均值回归味道"消失）—— M37
  · 配对反事实的两条路径**必须同种子**（否则配对无效）—— M38
  · "修复前"这个对照组必须**真的**关了做空额度（否则前后对比无意义）—— M39
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / "scripts"))  # 追加：插到前面会遮蔽同名包（scripts/gui.py vs gui/）

from tw.agents.base import Agent  # noqa: E402
from tw.order_flow.meta_order import (  # noqa: E402
    MetaOrderConfig,
    MetaOrderMixin,
)

# ⚠️ 用完就把 scripts/ 撤出 sys.path（理由见 tests/test_cache_keys.py：
#    scripts/gui.py 会遮蔽仓库根的 gui/ 包）。
_SCRIPTS = str(ROOT / "scripts")
while _SCRIPTS in sys.path:
    sys.path.remove(_SCRIPTS)


class _StubBase(Agent):
    """一个记录"基础 decide 被调用了几次"的桩主体。"""

    KIND = "chartist"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.base_decide_calls = 0

    def decide(self, state):
        self.base_decide_calls += 1
        return self.buy_order(state, 1.0)


class _MetaStub(MetaOrderMixin, _StubBase):
    pass


class _NullState:
    """``decide`` 只用到 ``tick`` 与 ``best_ask/bid``（桩里都不用真价格）。"""

    tick = 0
    mid = 100.0
    best_ask = 100.0
    best_bid = 100.0
    last_price = 100.0
    fundamental = 100.0
    spread = 0.1
    flow_imbalance = 0.0
    flow = None
    best_bid_size = best_ask_size = 1.0


class TestMetaOrderExecutionSkipsBaseDecision(unittest.TestCase):
    """⭐ M36：执行期必须**跳过**基础决策，否则一个 tick 会出两笔单。"""

    def _agent(self, p_start: float):
        # ⚠️ participation_rate 默认是 0.15 ⇒ 85% 的 tick 不吐子单。
        #    第一版没设它，12 次调用一张子单都没拿到，测试直接"无效"。
        #    测试要的是**方向语义**，所以把参与率拉满，排除抽样噪声。
        a = _MetaStub("m0", 1e6, 0.0, np.random.default_rng(0),
                      meta_config=MetaOrderConfig(p_meta_start=p_start,
                                                  participation_rate=1.0))
        a.bind_seed(1)
        return a

    def test_执行期不再调用基础decide(self) -> None:
        a = self._agent(0.0)
        st = _NullState()
        a.meta_state.start("buy", 10.0, 0)
        before = a.base_decide_calls
        out = a.decide(st)
        self.assertEqual(a.base_decide_calls, before,
                         "执行元订单时仍然调用了基础 decide —— "
                         "这正是 M36 的形态：同一 tick 会出两笔方向不同的单")
        self.assertIsNotNone(out)
        self.assertEqual(out.side, "buy", "子单方向必须沿用锁定方向")

    def test_非执行期才走基础decide(self) -> None:
        a = self._agent(0.0)
        st = _NullState()
        before = a.base_decide_calls
        a.decide(st)
        self.assertEqual(a.base_decide_calls, before + 1)


class TestFundamentalistAnchoring(unittest.TestCase):
    """⭐ M37：基本面派的子单方向必须**锚定启动那一刻**，逐个子单都不变。"""

    def _agent(self):
        # scripts/ 已从 sys.path 撤出，用文件路径精确加载，避免模块名冲突
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_rs6_for_test", str(ROOT / "scripts" / "run_stage6.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.path.insert(0, _SCRIPTS)
        try:
            spec.loader.exec_module(mod)
            a = mod.MetaFundamentalist("f0", 1e6, 0.0,
                                       np.random.default_rng(0),
                                       meta_config=MetaOrderConfig(
                                           p_meta_start=0.0,
                                           participation_rate=1.0))
            a.bind_seed(2)
        finally:
            while _SCRIPTS in sys.path:
                sys.path.remove(_SCRIPTS)
        return a

    def test_方向全程不翻转(self) -> None:
        a = self._agent()
        a.meta_state.start("buy", 40.0, 0)
        sides = []
        for _ in range(12):
            r = a.next_child()
            if r is None:
                break
            sides.append(r[0])
        self.assertGreater(len(sides), 3, "元订单没吐出足够的子单，测试无效")
        self.assertEqual(set(sides), {"buy"},
                         f"子单方向在拆分过程中变了：{sides} —— "
                         f"这正是 M37 的形态，"
                         f"'锚定启动方向'这个性质被破坏")

    def test_主动收尾后方向不再是锁定的(self) -> None:
        a = self._agent()
        a.meta_state.start("sell", 40.0, 0)
        a.meta_state.finish()
        self.assertFalse(a.meta_state.active)
        with self.assertRaises(RuntimeError):
            _ = a.meta_state.locked_direction


class TestPairedCounterfactualSameSeed(unittest.TestCase):
    """⭐ M38：配对反事实的两条路径**必须用同一个种子**。"""

    def test_两条路径用的是同一个种子(self) -> None:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_wsC_for_test", str(ROOT / "scripts" / "run_workstream_C.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.path.insert(0, _SCRIPTS)
        try:
            spec.loader.exec_module(mod)
            seen: list[int] = []
            real_build = mod.build

            def spy(n, cv, idio, seed, **kw):
                seen.append(int(seed))
                return real_build(n, cv, idio, seed, **kw)

            mod.build = spy
            # ⚠️ 把真实的 tick 推进关掉：本测试只验证"种子怎么传"，
            #    不需要跑仿真。第一版跑了完整 6000 tick 预热，13 项测试花 77 秒——
            #    而变异验证要把它乘 39 遍，成本直接爆炸。
            mod.WARMUP = 0
            real_run = mod.run_market
            mod.run_market = lambda *a, **kw: None
            try:
                mod.paired_propagation(2e-3, 20260917, horizon=0)
            finally:
                mod.run_market = real_run
        finally:
            while _SCRIPTS in sys.path:
                sys.path.remove(_SCRIPTS)
        self.assertEqual(len(seen), 2, f"应当跑两条路径，实际 {len(seen)}")
        self.assertEqual(seen[0], seen[1],
                         f"处理路径与对照路径用了不同的种子 {seen} —— "
                         f"配对相减就不再抵消漂移，M38 就是这个形态")


class TestPrePostFixControlIsReal(unittest.TestCase):
    """⭐ M39：'修复前'对照必须真的把做空额度设回 0。

    ⚠️ **这条测试第一版打错了靶子**：它直接调 ``pairs_pnl(0.0, …)`` 去验证
    "参数被透传"，而 M39 改的是 **``ec1_pre_post`` 里的调用点**
    （把 ``pairs_pnl(0.0, …)`` 写成 ``pairs_pnl(3.0, …)``）。
    结果 M39 注入后这条测试照样通过 —— **变异体验证当场抓出它是摆设**。

    教训：**测试要驱动"会被改的那一层"**。
    验证"下层函数会透传参数"不等于验证"上层调用点传对了参数"。
    所以这里改成驱动 ``ec1_pre_post`` 本身，并把它依赖的两个重活替换成桩。
    """

    def test_修复前那一臂必须传0(self) -> None:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_wsC3_for_test", str(ROOT / "scripts" / "run_workstream_C.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.path.insert(0, _SCRIPTS)
        try:
            spec.loader.exec_module(mod)
            seen: list[float] = []

            def spy_pnl(short_headroom, seeds, *a, **kw):
                seen.append(round(float(short_headroom), 6))
                return {"pnl_bp": 1.0, "pnl_sd": 0.1, "pnl_per_seed": [1.0],
                        "fills_a": 1.0, "fills_b": 1.0, "qty_a": 0.0, "qty_b": 0.0}

            def spy_exposure(short_headroom, seeds):
                return {"ratio_mean": float(short_headroom), "per_seed": []}

            mod.pairs_pnl = spy_pnl
            mod.net_exposure_ratio = spy_exposure
            mod.ec1_pre_post(seeds=[20260917])
        finally:
            while _SCRIPTS in sys.path:
                sys.path.remove(_SCRIPTS)

        self.assertEqual(len(seen), 2, f"应当跑修复前/后两臂，实际 {seen}")
        self.assertIn(0.0, seen,
                      f"没有任何一臂用了 short_headroom=0 —— M39 的形态就是"
                      f"把'修复前'那一臂也写成 3.0，于是两组相同、"
                      f"bug 的影响被算成 0。实际传的是 {seen}")
        self.assertIn(3.0, seen, f"没有'修复后'那一臂：{seen}")


class TestJointMarketComposition(unittest.TestCase):
    """⭐ 联合市场必须**同时**具备元订单与 Hawkes 两种行为。

    为什么这条必须存在：三线合并用的是多重继承
    ``JointMarket(Stage6Market, HawkesMarket)``。
    它现在能工作，靠的是两个前提：
      ① 两边都用 ``**kw`` 透传自己没消费的参数（``__init__`` 链不会断）；
      ② 两边各管互不重叠的方法（``decorate_agent`` vs ``_limit_activity``/``step``）。
    **任一条被破坏，症状都是"机制静默失效"**（不报错、效果为零）——
    比如有人给 ``Stage6Market`` 也加一个 ``step``，MRO 会选到它，
    Hawkes 的 ``step`` 就再也不跑了，而 k 只是"看起来没改善"。

    所以这里断言的是**两种行为同时在场**，而不是"跑起来没报错"。
    反面参照：把基类顺序反过来，元订单必须消失——
    证明这条测试的分辨力不是来自"随便跑一场"。
    """

    def _build(self, cls, shares, cfg):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_wsJ_for_test", str(ROOT / "scripts" / "run_workstream_joint.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.path.insert(0, _SCRIPTS)
        try:
            spec.loader.exec_module(mod)
            from tw import Population, SimConfig
            from tw.order_flow.meta_order import MetaOrderConfig
            m = cls(SimConfig(seed=5, n_ticks=120,
                              population=Population.from_shares(60, mod.MM_MIX)),
                    meta_cfg=MetaOrderConfig(**mod.META_CFG), meta_shares=dict(shares),
                    hawkes_config=cfg)
            m.run(120)
        finally:
            while _SCRIPTS in sys.path:
                sys.path.remove(_SCRIPTS)
        return m

    def test_两种行为同时在场(self) -> None:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_wsJ2_for_test", str(ROOT / "scripts" / "run_workstream_joint.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.path.insert(0, _SCRIPTS)
        try:
            spec.loader.exec_module(mod)
            from tw.order_flow.hawkes import HawkesConfig
            shares = mod.SHARES_ALL
            cfg = HawkesConfig(base_lambda=40.0, branching=0.6, beta=0.15)
            m = self._build(mod.JointMarket, shares, cfg)
        finally:
            while _SCRIPTS in sys.path:
                sys.path.remove(_SCRIPTS)

        n_meta = sum(int(getattr(a, "n_meta_started", 0)) for a in m.agents)
        n_on = sum(1 for a in m.agents
                   if float(getattr(getattr(a, "meta_config", None),
                                   "p_meta_start", 0.0) or 0.0) > 0)
        self.assertGreater(n_on, 0, "联合市场里没有任何主体被激活元订单——"
                                    "基类顺序或 __init__ 链被破坏了")
        self.assertGreater(n_meta, 0, f"激活了 {n_on} 个主体但一个元订单都没启动")
        self.assertTrue(getattr(m, "hawkes_log", None),
                        "hawkes_log 是空的——Hawkes 的 step 没有跑"
                        "（多半是有人给 Stage6Market 也加了 step）")
        self.assertEqual(len(m.hawkes_log), 120,
                         "hawkes_log 的条数与 tick 数不符——step 链断了")

    def test_单一行为的类过不了这两条断言_反面参照(self) -> None:
        """证明上面的断言有分辨力——不是「随便跑一场都能过」。

        ⚠️ **第一版这里写错了前提**：我以为「把基类顺序反过来元订单会消失」，
        于是断言 ``n_on == 0``。实测**顺序反过来照样激活 25 个主体**——
        因为 ``__init__`` 的 ``**kw`` 链在两种顺序下都会把两边都跑到，
        顺序只决定「哪个 ``decorate_agent`` / ``step`` 生效」。
        那条断言因此是错的（是我的假设错，不是实现错），当场被测试自己抓出来。

        正确的反面参照是**只具备单一行为**的类：
          · ``Stage6Market`` 单独 → 有元订单、**没有** hawkes_log
          · ``HawkesMarket`` 单独 → 有 hawkes_log、**没有**元订单
        两者各缺一样，才说明上面那两条断言真的在检查「两种行为同时在场」。
        """
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_wsJ3_for_test", str(ROOT / "scripts" / "run_workstream_joint.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.path.insert(0, _SCRIPTS)
        try:
            spec.loader.exec_module(mod)
            from run_stage6 import Stage6Market
            from tw.order_flow.hawkes import HawkesConfig
            from tw.order_flow.hawkes_market import HawkesMarket

            cfg = HawkesConfig(base_lambda=40.0, branching=0.6, beta=0.15)

            # ⚠️ 参照类要能**吞掉**自己那一样用不到的参数，
            #    否则 ``_build`` 统一传的 kwargs 会让它们直接 TypeError——
            #    那样测的就不是"行为有没有"，而是"签名兼不兼容"了。
            class OnlyMeta(Stage6Market):
                def __init__(self, *a, hawkes_config=None, hawkes_off=False,
                             rho_schedule=None, thin_scope="both",
                             record_depth=False, **kw):
                    super().__init__(*a, **kw)

            class OnlyHawkes(HawkesMarket):
                def __init__(self, *a, meta_cfg=None, meta_shares=None,
                             meta_share=0.0, adapt_mm=False, flow_window=0,
                             flow_sensitivity=0.0, **kw):
                    super().__init__(*a, **kw)

            results = {}
            for name, cls in (("OnlyMeta", OnlyMeta),
                              ("OnlyHawkes", OnlyHawkes)):
                m = self._build(cls, mod.SHARES_ALL, cfg)
                n_on = sum(1 for a in m.agents
                           if float(getattr(getattr(a, "meta_config", None),
                                           "p_meta_start", 0.0) or 0.0) > 0)
                results[name] = (n_on, len(getattr(m, "hawkes_log", []) or []))
        finally:
            while _SCRIPTS in sys.path:
                sys.path.remove(_SCRIPTS)

        only_meta, only_hawkes = results["OnlyMeta"], results["OnlyHawkes"]
        self.assertEqual(only_meta[1], 0,
                         f"Stage6Market 单独跑竟然有 hawkes_log：{only_meta}")
        self.assertEqual(only_hawkes[0], 0,
                         f"HawkesMarket 单独跑竟然激活了元订单：{only_hawkes}")
        self.assertTrue(only_meta[0] > 0 and only_hawkes[1] > 0,
                        "参照版连自己那一样行为都没有，这条参照是无效的")


class TestPlanDocIsRenderableMarkdown(unittest.TestCase):
    """合并计划 §5 必须写成**能渲染的 markdown**。

    为什么这条值得一个测试
    ----------------------
    这个 bug 是本轮真实发生的：``write_result_section`` 最后一句用了
    ``"".join(L)`` 而不是 ``"\\n".join(L)``。后果不是报错，而是
    **整节被拼成一行**——表格在渲染器里彻底失效，
    但脚本退出码是 0、自检也不看渲染，所以「跑成功了」和「产物能读」之间
    掉进去了一次。这正是本项目反复吃的「同一产物两条路径」/
    「静默失效」形态，必须有回归测试钉住。

    断言取向：不检查文案，只检查**结构**——
      · 表格行必须各自独立成行（不能整段一行）
      · 表头与分隔行的列数必须一致（否则渲染器会当成普通段落）
      · §5 必须是 §4 之后（假设在前、结果在后，顺序不能反）
    """

    def _render(self, J: dict) -> str:
        """把一份构造好的 J 灌进 write_result_section，取回写出的全文。"""
        import importlib.util
        import tempfile

        spec = importlib.util.spec_from_file_location(
            "_wsJ_doc_for_test", str(ROOT / "scripts" / "run_workstream_joint.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.path.insert(0, _SCRIPTS)
        try:
            spec.loader.exec_module(mod)
            with tempfile.TemporaryDirectory() as td:
                tmp_plan = Path(td) / "plan.md"
                tmp_plan.write_text(
                    "# 三线合并计划\n\n## 4. 合并里不能做的事\n\n- 甲\n- 乙\n",
                    encoding="utf-8")
                real_plan = mod.PLAN
                mod.PLAN = tmp_plan
                try:
                    mod.write_result_section(J)
                    return tmp_plan.read_text(encoding="utf-8")
                finally:
                    mod.PLAN = real_plan
        finally:
            while _SCRIPTS in sys.path:
                sys.path.remove(_SCRIPTS)

    @staticmethod
    def _fake_J() -> dict:
        """一份最小但**形状完整**的 J（两个臂各一行 + 两个环节各一行）。"""
        return {
            "ej1": {
                "base_k": 1.251,
                "arms": {
                    "J1 阶段6 基线": {
                        "base_lambda": 90.58, "k": 1.251, "closeness": 0.751,
                        "rho_mean": None, "activated_frac": None,
                        "events_per_tick": None},
                    "J3 +A 修正 Hawkes": {
                        "base_lambda": 71.84, "k": 0.957, "closeness": 0.457,
                        "rho_mean": 1.0319, "activated_frac": 0.244,
                        "events_per_tick": 75.75},
                },
            },
            "ej2": {
                "lambda_A": 50.89, "lambda_B": 50.67, "n_links": 4, "n_seeds": 3,
                "per_key": {
                    "a_price": {"label": "① A 的价格偏离(bp)",
                                "n_observed": 0, "n_seeds": 3, "holds": False},
                    "a_depth": {"label": "③ A 的买盘深度(手)",
                                "n_observed": 2, "n_seeds": 3, "holds": True},
                },
            },
        }

    @staticmethod
    def _table_lines(text: str) -> list:
        return [ln for ln in text.splitlines() if ln.strip().startswith("|")]

    @staticmethod
    def _n_cols(line: str) -> int:
        """数一行 markdown 表格有几列。

        ⚠️ **这里自己错过一次**：第一版用 ``line.count("|")``，
        结果把表头 ``\\|k−0.5\\|`` 里的**转义竖线**也算成列分隔符，
        于是"表头 9 列、分隔行 7 列"报红——**是测试错，不是产物错**
        （markdown 的 ``\\|`` 渲染成字面量竖线，本来就不算列）。
        所以先把 ``\\|`` 抹掉再数。
        """
        body = line.strip().strip("|").replace("\\|", "")
        return body.count("|") + 1

    def test_表格必须逐行独立而不是被拼成一行(self) -> None:
        text = self._render(self._fake_J())
        lines = self._table_lines(text)
        # 两个表：J1–J3（1 表头 + 1 分隔 + 2 数据）与 J4（1 + 1 + 2）
        self.assertGreaterEqual(
            len(lines), 8,
            f"表格行只有 {len(lines)} 行——多半又被 \"\".join() 拼成一行了：\n{text[-400:]}")
        # 整段成行的典型症状：某一行里出现「表头紧跟分隔行」
        self.assertNotIn("|---|---|", lines[0],
                         "表头与分隔行粘在同一行了——markdown 表格不会渲染")

    def test_表头与分隔行的列数一致(self) -> None:
        text = self._render(self._fake_J())
        lines = self._table_lines(text)
        header, sep = lines[0], lines[1]
        n_col = self._n_cols(header)
        n_sep = self._n_cols(sep)
        self.assertEqual(
            n_sep, n_col,
            f"表头 {n_col} 列，分隔行 {n_sep} 列——渲染器会把它当普通段落\n"
            f"  表头：{header}\n  分隔：{sep}")
        self.assertTrue(all(set(c) <= set("-: ") for c in sep.strip("|").split("|")),
                        f"分隔行含有非分隔字符：{sep!r}")

    def test_假设在前结果在后(self) -> None:
        """§4（不能做的事）必须仍在 §5（结果）之前——先写假设再跑是本项目纪律。"""
        text = self._render(self._fake_J())
        self.assertIn("## 4. 合并里不能做的事", text)
        self.assertIn("## 5. 结果与判读", text)
        self.assertLess(text.index("## 4."), text.index("## 5."),
                        "§5 跑到 §4 前面去了——说明写结果时把假设段截掉了")
        self.assertIn("- 甲", text, "§4 的原有内容被覆盖掉了")

    def test_承重数字真的出现在文里(self) -> None:
        """k=0.957 与 4/6 必须在文中——否则"写回"这一步白做了。"""
        text = self._render(self._fake_J())
        self.assertIn("0.957", text)
        self.assertIn("4/6", text)
        self.assertIn("24.4%", text)


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
