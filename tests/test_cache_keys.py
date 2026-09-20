"""缓存键的完备性——本项目**真实踩过**的一类事故。

为什么值得专门写一个测试文件
----------------------------
阶段6 的缓存键原来是 ``{tag, seed, n_ticks, mix}``。它"看起来该有的都有了"，
但被缓存的量还依赖 ``Stage6Market.decorate_agent`` / ``factory`` 的行为。
那段代码里有个 bug：对照组本应"机制全关"，却混进了 ``MetaOrderMixin``
的默认 ``p_meta_start=0.02``，**对照组根本不是对照**。

修掉代码后重跑，键**一个字符没变** → 全部命中陈旧缓存 → 产出的数字
与修 bug 之前**逐位相同**。如果没有事后核对，这个陈旧基线就会被当成
"修复后"的证据写进报告。

这类事故的特征：**不报错、不崩溃、产物格式完全正常**。
所以它必须由测试来守，而不是靠人眼。

这里测的不是"缓存能不能用"，而是四件更锐利的事：
  1. 代码签名会随**行为定义代码**变化而变化（不是随打印语句变）；
  2. 代码签名会随**模块级常量**变化而变化；
  3. 签名不符时，缓存**整份作废**（而不是"新键查不到就用旧值"）；
  4. **旧格式（无签名）的缓存文件一律丢弃**——用没有签名机制的代码
     写出来的缓存，无法判断是否陈旧，宁可重算。
"""

from __future__ import annotations

import json
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from scripts._cache import (  # noqa: E402
    Cache,
    consts_sig,
    file_sig,
    lib_sig,
    source_sig,
)

# ⚠️⚠️ **必须把 ``scripts/`` 从 sys.path 里撤掉。**
#
# 真实事故（就在加这个文件的那一轮）：``scripts/gui.py`` 与仓库根下的
# ``gui/`` **包重名**。谁先把 ``scripts/`` 放进 sys.path，后面
# ``tests/test_gui.py`` 里的 ``from gui import desktop`` 就会解析到
# ``scripts/gui.py``，报 ``partially initialized module`` ——
# 而症状是**基线测试挂了一条**，看起来像"新写的代码有问题"，
# 实际是**模块解析顺序**问题。
#
# 本文件按字母序排在 ``test_gui`` 之前，正好踩中。
# 撤掉之后，后面导入的测试看到的就是原来的 sys.path。
_SCRIPTS = str(Path(__file__).resolve().parent.parent / "scripts")
while _SCRIPTS in sys.path:
    sys.path.remove(_SCRIPTS)


def _write_module(dir_: Path, name: str, body: str) -> Path:
    p = dir_ / f"{name}.py"
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return p


def _stage_module(name: str):
    """导入 ``run_stage5`` / ``run_stage6``——**自己负责 sys.path，不靠别人**。

    ⚠️ 真实踩到：这两个模块住在 ``scripts/`` 下，而本文件在 import 时
    把 ``scripts/`` 从 sys.path 撤掉了（为了不遮蔽仓库根的 ``gui/`` 包，
    见上方说明）。原先这里写的是裸 ``import run_stage6 as m``，
    在主仓库里**碰巧**能过——因为 ``tests/test_stage_scripts.py``
    （按字母序排在后面）在 import 时会由 ``run_stage6`` 自己把
    ``scripts/`` 放回 sys.path。而在变异沙箱里那个顺序不成立，
    4 条测试直接 ModuleNotFoundError，**基线整体变红、整轮变异验证作废**。

    教训：**测试对 sys.path 的需求必须自己声明**，
    依赖"别的测试碰巧先做了某件事"是把测试顺序当成隐式契约。
    """
    import importlib
    sys.path.insert(0, _SCRIPTS)
    try:
        return importlib.import_module(name)
    finally:
        while _SCRIPTS in sys.path:
            sys.path.remove(_SCRIPTS)


class TestSourceSig(unittest.TestCase):
    """① 签名要跟着**行为定义代码**走，而不是跟着整个文件走。"""

    def test_行为代码变了签名就变(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            _write_module(d, "m_v1", """
                def decide(x):
                    return x > 0.5
            """)
            _write_module(d, "m_v2", """
                def decide(x):
                    return x >= 0.5
            """)
            sys.path.insert(0, str(d))
            try:
                import m_v1
                import m_v2
                s1 = source_sig(m_v1.decide)
                s2 = source_sig(m_v2.decide)
            finally:
                sys.path.remove(str(d))
                for m in ("m_v1", "m_v2"):
                    sys.modules.pop(m, None)
            self.assertNotEqual(
                s1, s2,
                "改了判断逻辑（> 变成 >=）签名却没变——这正是不完备的签名，"
                "会导致改完代码还命中旧缓存")

    def test_改注释与打印也会改签名_保守但安全(self) -> None:
        """签名**会**对注释/打印敏感。

        这是 ``inspect.getsource`` 的必然结果，方向是**保守**的：
        改注释会让缓存多失效一次（代价：重跑），
        而不是让改了行为却复用旧值（代价：**错误的结论**）。
        两个方向的代价不对称，所以宁可保守。
        真正要保证的是上面那条——**改行为必须改签名**。
        """
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            _write_module(d, "n_v1", """
                def decide(x):
                    # 判断阈值
                    print("deciding")
                    return x > 0.5
            """)
            _write_module(d, "n_v2", """
                def decide(x):
                    # 阈值判断（说明文字改了）
                    print("decide now")
                    print("extra log line")
                    return x > 0.5
            """)
            sys.path.insert(0, str(d))
            try:
                import n_v1
                import n_v2
                s1 = source_sig(n_v1.decide)
                s2 = source_sig(n_v2.decide)
            finally:
                sys.path.remove(str(d))
                for m in ("n_v1", "n_v2"):
                    sys.modules.pop(m, None)
            self.assertNotEqual(s1, s2)  # 源码文本变了 → 签名变（保守）

    def test_同一个函数两次签名一致(self) -> None:
        def f(x):
            return x + 1

        self.assertEqual(source_sig(f), source_sig(f))


class TestConstsSig(unittest.TestCase):
    """② 模块级常量不在参数里，是最容易漏掉的一类输入。"""

    def test_常量值变了签名就变(self) -> None:
        a = consts_sig(META_CFG={"scale": 20})
        b = consts_sig(META_CFG={"scale": 40})
        self.assertNotEqual(a, b)

    def test_键顺序不影响签名(self) -> None:
        self.assertEqual(consts_sig(a=1, b=2), consts_sig(b=2, a=1))

    def test_常量变了但没进签名会撞车_反面参照(self) -> None:
        """反面参照：证明上面那条不是平凡成立。

        不把常量放进 ``consts_sig`` 时，改常量当然不会改签名——
        这正是"少一个输入"的样子。
        """
        self.assertEqual(consts_sig(x=1), consts_sig(x=1))
        self.assertNotEqual(consts_sig(x=1), consts_sig(x=1, y=2))


class TestLibSig(unittest.TestCase):
    """③ ``tw/`` 源码树的内容哈希。"""

    def test_库文件内容变了签名就变(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "tw").mkdir()
            p = d / "tw" / "core.py"
            (d / "scripts").mkdir()
            p.write_text("X = 1\n", encoding="utf-8")
            (d / "scripts" / "_common.py").write_text("", encoding="utf-8")
            s1 = lib_sig(d)
            p.write_text("X = 2\n", encoding="utf-8")
            s2 = lib_sig(d)
            self.assertNotEqual(s1, s2, "库源码改了签名没变")

    def test_真实库的签名非空且稳定(self) -> None:
        s = lib_sig()
        self.assertTrue(s)
        self.assertEqual(s, lib_sig())

    def test_库签名没有被缓存住(self) -> None:
        """⭐ 这条抓过一个真 bug。

        ``lib_sig`` 第一版加了 ``functools.lru_cache``：返回值取决于
        **磁盘上的文件内容**，而 ``lru_cache`` 只按**参数**记忆。
        参数没变（还是同一个 root）→ 永远返回第一次的结果 →
        "用来发现文件变了"的函数自己看不见文件变化。

        同一个 ``root`` 必须能被连续调用并反映中间的文件修改。
        """
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "tw").mkdir()
            (d / "scripts").mkdir()
            (d / "scripts" / "_common.py").write_text("", encoding="utf-8")
            f = d / "tw" / "core.py"
            f.write_text("X = 1\n", encoding="utf-8")
            s1 = lib_sig(d)
            f.write_text("X = 2\n", encoding="utf-8")
            s2 = lib_sig(d)
            self.assertNotEqual(
                s1, s2,
                "同一个 root 连续调两次、中间改了文件，签名却没变——"
                "说明签名被缓存住了")

    def test_文件签名对缺文件不炸(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "a.py").write_text("1\n", encoding="utf-8")
            s1 = file_sig([d / "a.py"])
            s2 = file_sig([d / "a.py", d / "missing.py"])
            self.assertNotEqual(s1, s2)


class TestCacheInvalidation(unittest.TestCase):
    """④ 签名不符 → 整份作废；旧格式 → 一律丢弃。"""

    def test_签名变了旧条目全部作废(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.json"
            c = Cache(p)
            c.bind("SIG_A")
            c.get_or_run("k1", lambda: {"v": 1})
            c.flush()
            self.assertEqual(len(c.data), 1)

            c2 = Cache(p)
            c2.bind("SIG_B")          # 代码变了
            self.assertEqual(len(c2.data), 0,
                             "签名不符却保留了旧条目——这正是陈旧缓存事故")
            c2.get_or_run("k1", lambda: {"v": 2})
            self.assertEqual(c2.data["k1"]["v"], 2)

    def test_签名不变时命中缓存(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.json"
            c = Cache(p)
            c.bind("SIG_A")
            c.get_or_run("k1", lambda: {"v": 1})
            c.flush()

            calls = []

            def fn():
                calls.append(1)
                return {"v": 99}

            c2 = Cache(p)
            c2.bind("SIG_A")
            got = c2.get_or_run("k1", fn)
            self.assertEqual(got, {"v": 1})
            self.assertEqual(calls, [], "签名没变却重跑了——缓存失效了")
            self.assertEqual(c2.hits, 1)
            self.assertEqual(c2.misses, 0)

    def test_旧格式缓存文件被丢弃(self) -> None:
        """真实场景：``out/stage6_cache.json`` 原来是裸 dict，没有 ``_sig`` 外壳。"""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "legacy.json"
            p.write_text(json.dumps({"abc123": {"k": 1}}), encoding="utf-8")
            c = Cache(p)
            self.assertEqual(c.data, {},
                             "旧格式（无签名）缓存被沿用了——无法判断它是否陈旧")
            c.bind("SIG_X")
            self.assertEqual(len(c.data), 0)

    def test_键不同则互不干扰(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            c = Cache(Path(d) / "c.json")
            c.bind("S")
            a = c.get_or_run(c.key("run", seed=1, sig="S"), lambda: "A")
            b = c.get_or_run(c.key("run", seed=2, sig="S"), lambda: "B")
            self.assertEqual((a, b), ("A", "B"))

    def test_flush是原子替换(self) -> None:
        """中途崩了不能留下半个 JSON——那会让下一次运行直接解析失败。"""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.json"
            c = Cache(p)
            c.bind("S")
            c.get_or_run("k", lambda: {"v": 1})
            c.flush()
            c.get_or_run("k2", lambda: {"v": 2})
            c.flush()
            self.assertFalse(p.with_suffix(".json.tmp").exists())
            self.assertEqual(len(json.loads(p.read_text(encoding="utf-8"))["_entries"]), 2)


class TestStage6KeyActuallyCoversBehavior(unittest.TestCase):
    """⭐ 这条才是这次事故的**直接**回归测试。

    阶段6 的键必须随 ``Stage6Market`` / ``factory`` 的行为代码变化。
    做法：把真实模块的源码复制一份、只改 ``decorate_agent`` 里的一个判断，
    导入副本，比较 ``source_sig``。
    """

    def test_改装配代码会改签名(self) -> None:
        m = _stage_module("run_stage6")

        src = Path(m.__file__).read_text(encoding="utf-8")
        # 挑一处**行为**改动：把惰性配置的那句改掉（正是当初的 bug 形态）
        #
        # ⚠️ 这个 needle 在 2026-09-18 被**这条测试自己**逼着更新过一次：
        #    工作线B 把 ``decorate_agent`` 重构成表驱动（``META_CLASS_OF[kind]``），
        #    原来那句 ``return MetaChartist, {...}`` 不再存在 →
        #    ``assertIn`` 立刻变红，提示"这个回归测试已失效，必须同步更新"。
        #    **这正是它该做的事**：一个锚死在旧代码上的回归测试，
        #    在代码重构后会静静变成一句永远通过的废话。
        #    所以这里改为锚定**语义不变的那一小段**：
        #    「惰性配置」的本质是那个常量 ``p_meta_start=0.0``。
        needle = '**kw, "meta_config": MetaOrderConfig(p_meta_start=0.0)}'
        self.assertIn(
            needle, src,
            "找不到那句惰性配置——如果它被改名或删掉，这个回归测试已失效，"
            "必须同步更新；否则它会变成一条「永远通过」的假测试")
        mutated = src.replace(
            needle,
            '**kw, "meta_config": MetaOrderConfig(p_meta_start=0.02)}')

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            # 副本要能 import（同目录放一个 _common/_cache 的转发），
            # 最省事的做法是把 scripts/ 加进 path，只覆盖这一个模块名。
            mod_path = d / "run_stage6_mut.py"
            mod_path.write_text(mutated, encoding="utf-8")
            sys.path.insert(0, str(d))
            sys.path.insert(0, str(Path(m.__file__).resolve().parent))
            try:
                mm = _stage_module("run_stage6_mut")
                s_orig = source_sig(m.Stage6Market, m.factory)
                s_mut = source_sig(mm.Stage6Market, mm.factory)
            finally:
                for p_ in (str(d), str(Path(m.__file__).resolve().parent)):
                    if p_ in sys.path:
                        sys.path.remove(p_)
                sys.modules.pop("run_stage6_mut", None)

        self.assertNotEqual(
            s_orig, s_mut,
            "改了装配代码（惰性配置 → 默认配置）但签名没变——"
            "这正是 2026-09-18 那次「修完 bug 仍命中陈旧缓存」的事故成因")

    def test_阶段6的键里真的带上了签名(self) -> None:
        m = _stage_module("run_stage6")

        k = m.CACHE.key("lmf_run", tag="t", seed=1, n_ticks=1, mix=[],
                        sig=m.BEHAVIOR_SIG)
        k2 = m.CACHE.key("lmf_run", tag="t", seed=1, n_ticks=1, mix=[],
                         sig=m.BEHAVIOR_SIG + "x")
        self.assertNotEqual(k, k2)
        # 签名本身必须是真的算出来的（不是空串/常量）
        self.assertEqual(m.CACHE.sig, m.BEHAVIOR_SIG)
        self.assertGreaterEqual(len(m.BEHAVIOR_SIG), 32)


class TestStage5KeyCoversCode(unittest.TestCase):
    def test_阶段5的键里带库签名(self) -> None:
        m = _stage_module("run_stage5")

        self.assertIn(m.LIB_SIG, m.run_key(
            seed=1, p_buy=0.5, mix={"a": 0.5}, hedgers=0, n_ticks=100, cfg=None))

    def test_阶段5的键随配置变化(self) -> None:
        m = _stage_module("run_stage5")
        from tw.perpetual import FundingConfig

        kw = dict(seed=1, p_buy=0.5, mix={"a": 0.5}, hedgers=0, n_ticks=100)
        self.assertNotEqual(
            m.run_key(cfg=None, **kw),
            m.run_key(cfg=FundingConfig(scale=0.2), **kw))


if __name__ == "__main__":
    unittest.main()
