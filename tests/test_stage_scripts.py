"""实验脚本里那些**不跑模拟也会错**的纯逻辑。

为什么这些也要测
----------------
`scripts/` 目录一直不在测试范围内——它们是"实验"，似乎错了也无所谓。
但阶段5 发生的事说明这个假设是错的：E5.4 的挑选函数里，一个
**nan 排序键静默劫持了 argmin**，导致脚本选中了错误的偏好档位，
并在报告里写下"E5.4 未通过"这个**与数据相反的结论**。

它不报错、不崩溃，产物是一个格式完全正常的 JSON。
下游报告读走它，就会把错结论当成结论印发出去。

所以凡是"决定报告里写什么"的纯函数，都得进来测。
"""

from __future__ import annotations

import unittest

import numpy as np

from scripts.run_stage5 import match_distance, pos_frac_for, t_ci95
from tw.perpetual import REAL_FUNDING_ANNUAL, REAL_FUNDING_POS_FRAC


class TestStage6Wiring(unittest.TestCase):
    """⭐ 机制"关闭"必须是**真的关**。

    这一类 bug 的形态是："对照组"看起来是关的，其实开着——
    于是结论会悄悄变成"机制没用"，而且**不会报错**。

    真实踩到的版本：``decorate_agent`` 没有显式传 ``meta_config``，
    而 ``MetaOrderMixin.__init__`` 在收不到它时会用**默认值**
    （``p_meta_start=0.02``，也就是开着）。
    症状极具误导性——"基线"与"scale=1.0"两行数字**一模一样**，
    看起来像"小规模元订单无效"，其实是同一件事跑了两遍。
    """

    def _market(self, *, meta_scale: float, n_agents: int = 120,
                n_ticks: int = 1_600, seed: int = 5):
        from scripts.run_stage6 import MM_MIX, factory
        from tw import Population, SimConfig

        pop = Population.from_shares(n_agents, MM_MIX)
        m = factory(meta_scale=meta_scale)(
            SimConfig(seed=seed, n_ticks=n_ticks, population=pop)
        )
        m.run(n_ticks)
        return m

    def test_关闭时没有任何主体启动元订单(self) -> None:
        m = self._market(meta_scale=0.0)
        starts = sum(int(getattr(a, "n_meta_started", 0)) for a in m.agents)
        kids = sum(int(getattr(a, "n_children", 0)) for a in m.agents)
        self.assertEqual(starts, 0, "对照组竟然启动了元订单——关闭根本没生效")
        self.assertEqual(kids, 0, "对照组竟然发出了子单")

    def test_打开时确实有主体在拆单(self) -> None:
        m = self._market(meta_scale=20.0)
        starts = sum(int(getattr(a, "n_meta_started", 0)) for a in m.agents)
        kids = sum(int(getattr(a, "n_children", 0)) for a in m.agents)
        self.assertGreater(starts, 0, "处理组一个元订单都没启动")
        self.assertGreater(kids, starts, "元订单没有拆成多张子单（机制没生效）")

    def test_打开时每单拆出多张子单(self) -> None:
        """pareto_scale=20 时理论拆分数 ≈ E[总量]/E[子单] ≈ 60/1 = 60。

        实测约 42（受 max_child_orders 与终止条件影响）。
        只要求"显著大于 10"，把量级钉住即可——具体数值随代码调整会变，
        钉死它会让测试变脆，而**量级**才是这次要守住的东西。
        """
        m = self._market(meta_scale=20.0)
        starts = sum(int(getattr(a, "n_meta_started", 0)) for a in m.agents)
        kids = sum(int(getattr(a, "n_children", 0)) for a in m.agents)
        self.assertGreater(kids / max(1, starts), 10.0,
                           f"每单只拆了 {kids / max(1, starts):.1f} 张——"
                           "pareto_scale 可能没传到主体身上")

    def test_关闭组的配置是惰性的(self) -> None:
        m = self._market(meta_scale=0.0)
        for a in m.agents:
            cfg = getattr(a, "meta_config", None)
            if cfg is not None:
                self.assertEqual(cfg.p_meta_start, 0.0,
                                 "对照组主体的 p_meta_start 不是 0")

    def test_两组的主体构成一致(self) -> None:
        """配对的前提：两组的类与主体数完全相同，只有参数不同。"""
        a = self._market(meta_scale=0.0)
        b = self._market(meta_scale=20.0)
        self.assertEqual(len(a.agents), len(b.agents))
        ka = sorted(type(x).__name__ for x in a.agents)
        kb = sorted(type(x).__name__ for x in b.agents)
        self.assertEqual(ka, kb, "两组的类构成不同——配对差里混进了'类不同'")


class TestMatchDistance(unittest.TestCase):
    """⭐ 真实对标的挑选函数。它错了，整章的结论就是错的。"""

    def test_命中真实值时距离为零(self) -> None:
        self.assertAlmostEqual(
            match_distance(REAL_FUNDING_ANNUAL, REAL_FUNDING_POS_FRAC), 0.0, places=12
        )

    def test_年化为负返回无穷而不是nan(self) -> None:
        """核心回归：负年化必须返回 inf。

        返回 nan 会让 ``min(scan, key=...)`` **保留第一个元素**，
        于是挑选函数会"选中"表里第一个档位——一个静默的、与数据无关的结果。
        """
        d = match_distance(-0.0015, 0.477)
        self.assertTrue(np.isinf(d), f"负年化应返回 inf，实际 {d}")
        self.assertFalse(np.isnan(d))

    def test_年化为零返回无穷(self) -> None:
        self.assertTrue(np.isinf(match_distance(0.0, 0.5)))

    def test_年化为nan返回无穷(self) -> None:
        self.assertTrue(np.isinf(match_distance(float("nan"), 0.5)))

    def test_负年化不会劫持min(self) -> None:
        """把踩过的场景原样复现一遍：坏值在**列表第一个**。"""
        scan = [
            {"p_buy": 0.50, "annualized": -0.0015, "pos_frac": 0.4770},
            {"p_buy": 0.60, "annualized": 0.1375, "pos_frac": 0.9015},
            {"p_buy": 0.70, "annualized": 0.2688, "pos_frac": 0.9930},
        ]
        best = min(scan, key=lambda r: match_distance(r["annualized"], r["pos_frac"]))
        self.assertEqual(
            best["p_buy"], 0.60,
            "挑选函数选中了错误的档位——负年化的 nan 又劫持了 argmin",
        )

    def test_对称性(self) -> None:
        """年化高于/低于真实值一倍，距离应相同（对数口径）。"""
        a = match_distance(REAL_FUNDING_ANNUAL * 0.5, REAL_FUNDING_POS_FRAC)
        b = match_distance(REAL_FUNDING_ANNUAL * 2.0, REAL_FUNDING_POS_FRAC)
        self.assertAlmostEqual(a, b, places=12)


class TestPosFracFor(unittest.TestCase):
    """θ 扫描用的正占比计算。它错了，传导系数就标错了。"""

    def test_纯噪音通道的正占比是0点5(self) -> None:
        """θ=0（只留溢价）且溢价是零均值噪音 → 正占比 ≈ 0.5。

        这是阶段5 最关键的结构性结论：**正号占比只取决于分布形状，
        与幅度无关**。若哪天它开始依赖幅度，标定逻辑就要重写。
        """
        rng = np.random.default_rng(3)
        P = rng.normal(0.0, 1e-3, 20_000)
        C = rng.normal(0.0, 0.1, 20_000)
        self.assertAlmostEqual(pos_frac_for(0.0, P, C), 0.5, delta=0.02)
        # 幅度放大 1000 倍，正占比不变
        self.assertAlmostEqual(pos_frac_for(0.0, P * 1000, C), 0.5, delta=0.02)

    def test_加进有偏通道后正占比上升(self) -> None:
        rng = np.random.default_rng(4)
        P = rng.normal(0.0, 1e-3, 20_000)
        C = rng.normal(0.15, 0.1, 20_000)
        lo = pos_frac_for(0.0, P, C)
        hi = pos_frac_for(10.0, P, C)
        self.assertGreater(hi, lo + 0.3)

    def test_全零信号返回0点5(self) -> None:
        z = np.zeros(100)
        self.assertEqual(pos_frac_for(1.0, z, z), 0.5)


class TestTCI(unittest.TestCase):
    """E5.1 的置信区间。口径错了会得出相反的健全性结论。"""

    def test_区间覆盖已知均值(self) -> None:
        rng = np.random.default_rng(11)
        x = rng.normal(5.0, 1.0, 30)
        lo, hi, m = t_ci95(x)
        self.assertLess(lo, 5.0)
        self.assertGreater(hi, 5.0)
        self.assertAlmostEqual(m, float(np.mean(x)), places=12)

    def test_样本越多区间越窄(self) -> None:
        rng = np.random.default_rng(12)
        few = rng.normal(0.0, 1.0, 5)
        many = rng.normal(0.0, 1.0, 200)
        w1 = t_ci95(few)[1] - t_ci95(few)[0]
        w2 = t_ci95(many)[1] - t_ci95(many)[0]
        self.assertLess(w2, w1)

    def test_零均值样本的区间覆盖0(self) -> None:
        rng = np.random.default_rng(13)
        lo, hi, _ = t_ci95(rng.normal(0.0, 1.0, 8))
        self.assertLessEqual(lo, 0.0)
        self.assertGreaterEqual(hi, 0.0)

    def test_单样本不崩(self) -> None:
        lo, hi, m = t_ci95([1.0])
        self.assertTrue(np.isnan(lo))
        self.assertEqual(m, 1.0)


class TestReproduceOnlyNotSilentlyIgnored(unittest.TestCase):
    """``--only`` 点名了某个步骤，就必须真的产出那一步。

    这一类的失效形态在本项目出现过三次，全部是同一个样子：
    **用户说了，程序没听见，而且不吱声。**

      · ``--skip 8,9,11`` 静默不跳（跳过了但没跳成，用户以为跳了）
      · ``--only`` 忘了归一化 → 静默选出零步
      · 本次：``--only mutation`` 算出「共 0 步」——
        因为 ``tests`` / ``mutation`` 不在 ``STEPS`` 里（它们是收尾步骤），
        而第一版的追加逻辑只写在 ``if only is None`` 分支里。

    前两次的教训是"加了守卫"，这一次的教训更细：
    **"默认不跑"与"点名了也不跑"是两件事，前者正当，后者是 bug。**
    所以这里断言两件事：
      ① 点名 + 给闸门 → 必须产出那一步（`--only tests`、`--only mutation --with-mutation`）
      ② 点名但不给闸门 → 步数为 0 **且必须打印警告**（不能静默）
    """

    @staticmethod
    def _args(only=None, skip=None, with_mutation=False):
        import argparse
        return argparse.Namespace(
            only=only, skip=skip, with_mutation=with_mutation,
            no_report=False, no_tests=False, no_selfcheck=False, frm=None)

    def _plan(self, **kw) -> tuple[list, str]:
        import contextlib
        import io

        from scripts.reproduce_all import plan_steps

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            steps = plan_steps(self._args(**kw))
        return [s[0] for s in steps], buf.getvalue()

    def test_点名tests必须产出tests(self) -> None:
        names, _ = self._plan(only="tests")
        self.assertEqual(names, ["tests"],
                         f"--only tests 算出 {names}——点名的步骤被静默丢掉了")

    def test_点名mutation且给闸门必须产出mutation(self) -> None:
        names, _ = self._plan(only="mutation", with_mutation=True)
        self.assertEqual(names, ["mutation"],
                         f"--only mutation --with-mutation 算出 {names}")

    def test_点名mutation不给闸门时不跑但必须开口(self) -> None:
        """闸门正当，但沉默不正当。"""
        names, out = self._plan(only="mutation")
        self.assertEqual(names, [], "没给 --with-mutation 却跑了 mutation（长任务被误触）")
        self.assertIn("with-mutation", out,
                      "点了名却什么都没发生，而且一个字都没说——" "这正是要根除的形态")

    def test_点名不存在的步骤要提示而不是静默零步(self) -> None:
        names, out = self._plan(only="stage99")
        self.assertEqual(names, [])
        self.assertIn("一个步骤都没有", out,
                      "零步骤必须显式提示，否则「用户打错名字」和「本来就没步骤」"
                      "看起来一模一样")

    def test_不给only时收尾步骤仍然照常追加(self) -> None:
        """反向参照：修复不能顺手把"默认追加"也删掉。"""
        names, _ = self._plan(with_mutation=True)
        self.assertIn("tests", names)
        self.assertIn("mutation", names)

    def test_不给only也不给闸门时mutation默认不跑(self) -> None:
        names, _ = self._plan()
        self.assertNotIn("mutation", names, "变异验证是重活，默认不该跑")
        self.assertIn("tests", names, "tests 是默认要跑的，别连它一起关了")

    def test_收尾不再因为未定义的skip而崩(self) -> None:
        """``main()`` 走完必须返回 0，而不是在最后一句抛 ``NameError``。

        本轮真实踩到：收尾那行原来写的是裸 ``skip``（``plan_steps`` 的局部变量），
        于是 ``--only`` 路径跑完全部步骤、对账也通过之后，
        **最后一行崩掉 → 退出码 1**。

        这个症状特别阴：产物全对，但程序说自己失败。
        看退出码的人会白重跑一遍，看日志的人会以为成功——**两边都不对**。
        所以这里把整条 main() 走通（把重活打桩），断言返回 0。
        """
        import contextlib
        import io
        import sys as _sys
        from unittest import mock

        from scripts import reproduce_all as R

        with mock.patch.object(_sys, "argv",
                               ["reproduce_all.py", "--only", "stage1"]), \
                mock.patch.object(R, "archive_before_overwrite"), \
                mock.patch.object(R, "run_step", return_value=(True, 1.0)), \
                mock.patch.object(R, "save_timings"), \
                mock.patch.object(R, "verify", return_value=[]):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = R.main()
        self.assertEqual(rc, 0, f"main() 返回 {rc}，输出尾部：\n{buf.getvalue()[-800:]}")


class TestSelfcheckPlaceholderExemptions(unittest.TestCase):
    """自检工具**自己**也要被审计：⑦ 项的豁免范围不能太大也不能太小。

    背景：⑦ 项静态扫"忘加 f 前缀"的模板串。``mutation_check.MUTATIONS`` 里的
    old/new **就是源码片段**（M42 的锚点里写着 ``{sorted(skipped)}``），
    于是自检对自己的代码报了 2 个 FAIL —— **假阳性**。
    假阳性比没有检查更坏：它会训练人忽略自检。

    但豁免必须**精确**：只豁免 MUTATIONS 里的字符串，
    定义之外的真·模板串仍然要被抓到。这个类就测这两面。
    """

    def test_变异体锚点被豁免(self) -> None:
        import ast

        from scripts.selfcheck import mutation_patch_string_positions

        src = (  # noqa: placeholder —— 下面这些字符串是"构造的源码样本"，不是模板串
            "MUTATIONS = [\n"
            "    ('M9', 'desc', [\n"
            "        ('f.py', 'print(f\"{x(1)}\")', 'print(1)'),\n"
            "    ]),\n"
            "]\n"
        )
        pos = mutation_patch_string_positions(ast.parse(src))
        # ⚠️ 不硬编码列号：列号会随缩进调整而失效，那属于"测试自己变脆"。
        #    改成"按内容找到那个字符串，再问它在不在豁免集合里"——
        #    这正是要断言的性质。
        tree = ast.parse(src)
        # ⚠️ 注意这里写的是 ``x(1)}`` 而不是完整的花括号形态——
        #    写完整形态会让**这一行自己**被 ⑦ 项扫到（自指），
        #    而要豁免它就得再挂一个 noqa 标记，越描越黑。
        #    取子串匹配既能找到目标，又不触发检查。
        hit = [n for n in ast.walk(tree)
               if isinstance(n, ast.Constant) and "x(1)}" in n.value]
        self.assertEqual(len(hit), 1,
                         "构造的样本没解析出含模板串的锚点，测试本身失效了")
        t = hit[0]
        self.assertIn((t.lineno, t.col_offset), pos,
                      "含模板串的锚点没被豁免——自检会对自己误报")

    def test_定义之外的真模板串不被豁免(self) -> None:
        import ast

        from scripts.selfcheck import mutation_patch_string_positions

        src = (  # noqa: placeholder —— 同上：构造的样本
            "MUTATIONS = [('M9', 'd', [('f.py', 'a', 'b')])]\n"
            "OTHER = '这里的 {f(x)} 是真忘加前缀'\n"
        )
        pos = mutation_patch_string_positions(ast.parse(src))
        # 第 2 行那个字符串必须**不在**豁免集合里（用它自己的真实位置去查）
        tree = ast.parse(src)
        other = [n for n in ast.walk(tree)
                 if isinstance(n, ast.Constant) and n.value == "a"]
        self.assertTrue(other, "构造的源码没解析出锚点字符串，测试本身失效了")
        self.assertNotIn((2, 8), pos, "把 MUTATIONS 之外的东西也豁免了——检查会漏")

    def test_真实仓库上不再误报(self) -> None:
        """端到端：对真实仓库跑一次 ⑦ 项，必须 0 个 FAIL。

        这才是本轮修复真正要保证的东西：**自检自己不再叫错**。
        """
        import contextlib
        import io

        from scripts import selfcheck as S

        r = S.Report()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            S.check_source_placeholders(r)
        self.assertEqual(r.fail, 0,
                         f"⑦ 项报出 {r.fail} 个 FAIL（多半又是假阳性）：\n"
                         f"{buf.getvalue()[-600:]}")


if __name__ == "__main__":
    unittest.main()
