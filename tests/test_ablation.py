"""三臂消融（A/B/C）的护栏。

对齐 A6 §6 的设计：

| 臂 | 指标怎么来 | 它单独回答什么 |
|---|---|---|
| **A** `real` | 用**当前**窗口算 | — |
| **B** `shifted` | 用**更早一段**窗口算（格式长度相同、信息过期） | `A−B` = **指标里的信息价值** |
| **C** `none` | 不提供（**明说**不提供） | `B−C` = **纯引导效应** |

⭐ **为什么 B 必须是"错位时间的真指标"而不是随机数**：
用随机数的话，B 与 C 的差别里混进了"看到一堆乱码"的干扰，
而 `A−B` 也不再是"信息的价值"（那成了"真数据 vs 乱码"）。
错位指标保住了"有一个指标板块、格式正常"这件事，
只把**时效性**拿掉 ⇒ 差值才干净。

⚠️ 本文件里最要紧的两条：
1. **A 与 B 的 prompt 只允许数字不同**（否则不是单一变量）；
2. **C 必须说"本次实验不提供"，不能说"样本不足"**——
   后者是**假话**，而假话本身会改变行为（让它更保守），
   那测到的就成了"被误导的效应"。
"""

from __future__ import annotations

import difflib
import unittest

from tw.agent import AgentConfig, TradingAgent, visible_from_series


def _txt(template: str = "v4", *, mode: str = "real", fc=None,
         rc=None) -> str:
    from tw.prompts import build_messages
    ms = build_messages(
        {"mid": 104.0, "recent_closes": rc or [100.0 + i * 0.35
                                               for i in range(12)]},
        inst_id="X", bar="1H", template=template, n_closes=12,
        feature_closes=fc, features_mode=mode)
    return "\n".join(str(m.get("content", "")) for m in ms)


class TestArmsAreSingleVariable(unittest.TestCase):
    """A 与 B 的差别**只许是数字**。"""

    def setUp(self):
        self.closes = [100.0 + i * 0.35 for i in range(12)]
        self.shifted = [x * 0.97 for x in self.closes]

    def test_长度几乎相同(self):
        a = _txt(mode="real", rc=self.closes)
        b = _txt(mode="shifted", fc=self.shifted, rc=self.closes)
        self.assertLess(abs(len(a) - len(b)), 40,
                        "A 与 B 的 prompt 长度差太多 ⇒ 可能不是同一套模板")

    def test_非数字行完全相同(self):
        """⭐⭐ **本组的核心护栏**：用"把数字抹掉再比"来证明
        两臂只差数字，而不是靠人眼看 diff。"""
        import re

        def skeleton(t: str) -> list[str]:
            return [re.sub(r"[-+]?\d[\d,]*\.?\d*", "#", ln) for ln in t.splitlines()]

        a = skeleton(_txt(mode="real", rc=self.closes))
        b = skeleton(_txt(mode="shifted", fc=self.shifted, rc=self.closes))
        d = [x for x in difflib.unified_diff(a, b, lineterm="", n=0)
             if x[:1] in "+-" and x[:3] not in ("---", "+++")]
        self.assertEqual(d, [], f"A 与 B 除了数字还有别的差异：{d}")

    def test_错位指标真的变了值(self):
        """⚠️ 分辨力补强：若两臂**值也一样**，上面那条会空过。"""
        a = _txt(mode="real", rc=self.closes)
        b = _txt(mode="shifted", fc=self.shifted, rc=self.closes)
        self.assertNotEqual(a, b)

    def test_缺错位数据时说明原因(self):
        b = _txt(mode="shifted", fc=None, rc=self.closes)
        self.assertIn("更早的一段行情", b)


class TestArmCSaysTheTruth(unittest.TestCase):
    """⭐ C 臂**不许说假话**。"""

    def test_明说不提供(self):
        c = _txt(mode="none")
        self.assertIn("本次实验不提供", c)

    def test_指标段里不说样本不足(self):
        """⚠️ **"样本不足"是假话**（数据是够的，是这次不给）。
        假话本身会改变行为 ⇒ 测到的是"被误导的效应"，不是"没有指标的效应"。

        ⚠️ 只查**指标段那一行**：模板的「怎么选」里另有一句通用 boilerplate
        （「没有可算的指标时（样本不足），只能走 B」），三臂都有、不构成差异。
        我第一版写成全文 `assertNotIn` ⇒ 被那句 boilerplate 绊红。
        ⇒ **断言要盯住"会被自变量改变的那一处"，不要顺手查全文。**
        """
        c = _txt(mode="none")
        line = [x for x in c.splitlines() if "本次实验不提供" in x]
        self.assertEqual(len(line), 1)
        self.assertNotIn("样本不足", line[0])

    def test_仍然是单一变量(self):
        """C 只是把指标**内容**换掉，规矩与格式都还在。"""
        c = _txt(mode="none")
        self.assertIn("照**已算好的指标**执行", c)


class TestFeatureClosesHaveNoFuture(unittest.TestCase):
    """⚠️ `feature_closes` 只许用**已收盘**的根，且整体前移。"""

    class _S:
        close = [100.0 + i for i in range(60)]

    def test_错位窗口的右端真的更早(self):
        vis0 = visible_from_series(self._S(), 40, n_closes=12)
        vis4 = visible_from_series(self._S(), 40, n_closes=12, feature_shift=12)
        self.assertIsNone(vis0.get("feature_closes"))
        fc = vis4["feature_closes"]
        self.assertEqual(len(fc), 12)
        # 右端 = closes[40-12] = 128.0；**必须早于当前根**
        self.assertEqual(fc[-1], 128.0)
        self.assertLess(fc[-1], vis4["recent_closes"][-1])

    def test_不开启时不多写字段(self):
        """⚠️ 回归护栏：默认行为不能变（多一个字段就改了 state_digest
        ⇒ **所有历史 decision_id 全变**）。"""
        vis = visible_from_series(self._S(), 40, n_closes=12)
        self.assertNotIn("feature_closes", vis)

    def test_历史不够时截短而不越界(self):
        vis = visible_from_series(self._S(), 5, n_closes=12, feature_shift=12)
        fc = vis["feature_closes"]
        self.assertIsInstance(fc, list)
        self.assertLessEqual(len(fc), 12)
        self.assertTrue(all(x >= 100.0 for x in fc))


class TestAblationWiring(unittest.TestCase):
    """配置层：**消融不许静默空转**。"""

    def test_shifted必须有正的shift(self):
        """⚠️ `shifted` + `shift=0` ⇒ B 臂与 A 臂**一模一样**，
        消融会静默变成空转（跑两遍同一件事，还照样出数字）。"""
        with self.assertRaises(ValueError):
            AgentConfig(inst_id="X", features_mode="shifted", features_shift=0)

    def test_未知模式要报错(self):
        with self.assertRaises(ValueError):
            AgentConfig(inst_id="X", features_mode="real2")

    def test_默认就是real且不写额外字段(self):
        cfg = AgentConfig(inst_id="X")
        self.assertEqual(cfg.features_mode, "real")
        self.assertEqual(cfg.features_shift, 0)

    def test_换臂会换decision_id(self):
        """⭐⭐ 否则「A 臂」与「B 臂」在同一 run/tick 下算出同一个 ID，
        而两者 prompt 不同 ⇒ 留痕里两条"长得一样"、归因失效。"""
        from tw.llm import LLMConfig, LLMResponse, ScriptedClient

        def run(cfg_kw) -> str:
            client = ScriptedClient(
                config=LLMConfig(provider="agnes"), on_exhausted="hold",
                responses=[LLMResponse(ok=True,
                                       text='{"action":"hold","reason":"x"}')] * 3)
            ag = TradingAgent(
                client=client,
                config=AgentConfig(inst_id="X", template="v4", n_samples=1,
                                   temperature=0.0, **cfg_kw))
            vis = {"mid": 100.0, "equity": 100_000.0}
            return ag.decide(visible=vis, tick=10, run_id="R",
                             equity=100_000.0).decision_id

        a = run({})
        b = run({"features_mode": "shifted", "features_shift": 12})
        c = run({"features_mode": "none"})
        self.assertEqual(len({a, b, c}), 3, "三个臂必须给三个不同的 decision_id")


class TestHeterogeneity(unittest.TestCase):
    """⭐ **段网格是「噪声」还是「分层因子」**——这个判据决定"补样本有没有用"。

    两种情形**下一步方向相反**：
    - 噪声大（Q 不显著）⇒ 补样本能救；
    - 真值随组变（Q 显著）⇒ **补多少样本都没用**，必须按组报。

    ⚠️ 我第一版在 A7/A8 里把"换个窗口结论就翻"直接读成了"所以不能立论"，
    而 Q 检验说那其实只是噪声 ⇒ **该补样本，不是该停手**。
    """

    def _h(self, xs):
        """⚠️⚠️ **不许把 `scripts/` 永久留在 `sys.path` 上**（我第一版就这么干的）。

        真实后果：本文件按字母序**排在 `test_agent_gui` 之前**，
        于是 `scripts/` 被插到 `sys.path[0]` 并**留在那里** ⇒
        之后 `import gui` 找到的是 **`scripts/gui.py`（模块）** 而不是
        **`gui/`（包）** ⇒ `ModuleNotFoundError: 'gui' is not a package`
        ⇒ **`test_agent_gui` 整个模块导入失败，一次丢掉 22 项测试**，
        而全量输出只写一句 `Ran 1497` —— **看不出少了什么**。

        ⭐ 判据：**测试辅助函数不许永久改全局状态**（`sys.path` 是最常见的一个）。
        要临时用就 `try/finally` 还回去。
        """
        import sys as _s
        from pathlib import Path as _P
        sp = str(_P(__file__).resolve().parent.parent / "scripts")
        added = sp not in _s.path
        if added:
            _s.path.insert(0, sp)
        try:
            from a9_power_ablation import heterogeneity
            return heterogeneity(xs)
        finally:
            if added:
                try:
                    _s.path.remove(sp)
                except ValueError:
                    pass

    def test_手算例子(self):
        """两组、效应 0.0 与 0.0、SE 都 1.0 ⇒ Q 恒为 0（完全同质）。"""
        h = self._h([(0.0, 1.0), (0.0, 1.0)])
        self.assertAlmostEqual(h["Q"], 0.0, places=12)
        self.assertFalse(h["heterogeneous"])

    def test_手算例子2(self):
        """两组差 4、SE 都 1 ⇒ Q = 4 ≠ 3.84 ⇒ 异质（df=1）。"""
        h = self._h([(-2.0, 1.0), (2.0, 1.0)])
        # θ̄ = 0, Q = 1*4 + 1*4 = 8
        self.assertAlmostEqual(h["Q"], 8.0, places=9)
        self.assertTrue(h["heterogeneous"])

    def test_大SE能把差异解释掉(self):
        """⚠️ 分辨力补强：**同样的效应差，SE 大了就不再是「异质」**。

        ⚠️ 我第一版拿 SE=1.0 当"大"，**算错了**：
        `Q = Σ(θᵢ−θ̄)²/SEᵢ²`，SE=1 时 Q = (2²+2²)/1 = **8 > 3.84** ⇒ 仍异质。
        要让它同质得 SE ≥ √(8/3.84) ≈ 1.44 ⇒ 这里取 2.0（Q = 2 < 3.84）。
        ⭐ **判据：「差异大」与「异质」不是一回事**——后者要拿 SE 去比。"""
        small = self._h([(-2.0, 0.1), (2.0, 0.1)])   # Q = 800
        big = self._h([(-2.0, 2.0), (2.0, 2.0)])     # Q = 2
        self.assertAlmostEqual(small["Q"], 800.0, places=6)
        self.assertAlmostEqual(big["Q"], 2.0, places=6)
        self.assertTrue(small["heterogeneous"])
        self.assertFalse(big["heterogeneous"])

    def test_一组时明确说不是同质(self):
        """⚠️ **"少于 2 组"不是"同质"**——不许把它读成通过检验。"""
        h = self._h([(1.0, 0.5)])
        self.assertFalse(h["heterogeneous"])
        self.assertIn("不是", h["note"])

    def test_排除无效SE(self):
        h = self._h([(1.0, 0.5), (2.0, 0.0), (3.0, float("nan")), (4.0, 0.5)])
        self.assertEqual(h["k"], 2)

    def test_I2在0到1之间(self):
        for xs in ([(0.0, 1.0), (0.0, 1.0)], [(-5.0, 1.0), (5.0, 1.0)]):
            h = self._h(xs)
            self.assertGreaterEqual(h["I2"], 0.0)
            self.assertLessEqual(h["I2"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
