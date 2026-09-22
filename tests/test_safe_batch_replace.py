"""批量替换安全工具的回归测试（工程纪律第八条的验收）。

这份测试的第一条就是**复现真实事故**：
上一轮用"按行替换引号"的脚本去修文案，
把模块/函数 docstring 的 `\"\"\"` 改成 `"「"`、把字典键改成 `"k「: r[」k_point「]`，
而且**没有立刻报错**——它在继续制造新的语法错误。

所以这里不测"跑起来没崩"，而是测三件事：
  ① 破坏语法的替换**必须被拦住**，且**一个文件都不许写回**；
  ② 全部通过时**必须真的写回**（否则工具等于禁用）；
  ③ 不同目录的同名文件**不许互相覆盖**（任务书伪代码里的真实缺陷）。
"""

from __future__ import annotations

import ast
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / "scripts"))  # 追加：插到前面会遮蔽同名包（scripts/gui.py vs gui/）

from scripts.safe_batch_replace import (  # noqa: E402
    UnsafeReplaceError,
    safe_batch_replace,
)

_SCRIPTS = str(ROOT / "scripts")
while _SCRIPTS in sys.path:
    sys.path.remove(_SCRIPTS)

#: 真实事故的形态：把三引号换成中文引号（会破坏 docstring）
DOCSTRING_BREAKER = [(r'"""', '"「"')]


class _Sandbox:
    """在临时目录里造文件，测试结束自动清理。"""

    def __init__(self) -> None:
        self._td = tempfile.TemporaryDirectory(prefix="sbr_test_")
        self.root = Path(self._td.name)

    def write(self, rel: str, text: str) -> Path:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        return p

    def read(self, rel: str) -> str:
        return (self.root / rel).read_text(encoding="utf-8")

    def cleanup(self) -> None:
        self._td.cleanup()


GOOD_SOURCE = '''"""模块 docstring。"""


def f():
    """函数 docstring。"""
    return {"k": 1}
'''


class TestSyntaxGuardBlocksBadReplace(unittest.TestCase):
    """① 破坏语法的替换必须被拦住，且不写回。"""

    def setUp(self) -> None:
        self.sb = _Sandbox()

    def tearDown(self) -> None:
        self.sb.cleanup()

    def test_复现真实事故_三引号被换成中文引号必须被拦截(self) -> None:
        """⭐ 这就是上一轮真实踩的坑：`\"\"\"` → `"「"`。"""
        p = self.sb.write("m.py", GOOD_SOURCE)
        before = self.sb.read("m.py")
        with self.assertRaises(UnsafeReplaceError) as ctx:
            safe_batch_replace([p], DOCSTRING_BREAKER, dry_run=False,
                               root=self.sb.root)
        self.assertIn("语法校验失败", str(ctx.exception))
        self.assertEqual(self.sb.read("m.py"), before,
                         "被拦住了却还是写回了文件——护栏形同虚设")

    def test_语法失败时即使dry_run也要报错(self) -> None:
        """dry-run 也不是"什么都能看"——它会破坏语法就必须报出来。"""
        p = self.sb.write("m.py", GOOD_SOURCE)
        with self.assertRaises(UnsafeReplaceError):
            safe_batch_replace([p], DOCSTRING_BREAKER, dry_run=True,
                               root=self.sb.root)

    def test_一个文件坏了则整体回滚_连好的也不写(self) -> None:
        """⭐ 要么全部成功、要么全部不生效——部分写入比不写更危险。"""
        good = self.sb.write("good.py", 'x = "ABC"\n')
        bad = self.sb.write("bad.py", GOOD_SOURCE)
        before_good = self.sb.read("good.py")
        with self.assertRaises(UnsafeReplaceError):
            # 这条规则对 good.py 无害（那里没有三引号），对 bad.py 有害
            safe_batch_replace([good, bad], DOCSTRING_BREAKER, dry_run=False,
                               root=self.sb.root)
        self.assertEqual(self.sb.read("good.py"), before_good,
                         "坏文件被拦住，好文件却被写回了——不是原子操作")
        self.assertEqual(self.sb.read("bad.py"), GOOD_SOURCE)

    def test_错误信息指出出错行号(self) -> None:
        p = self.sb.write("m.py", GOOD_SOURCE)
        with self.assertRaises(UnsafeReplaceError) as ctx:
            safe_batch_replace([p], DOCSTRING_BREAKER, root=self.sb.root)
        self.assertIn("行", str(ctx.exception))


class TestWritesBackWhenClean(unittest.TestCase):
    """② 全部通过时必须真的写回（否则工具等于禁用）。"""

    def setUp(self) -> None:
        self.sb = _Sandbox()

    def tearDown(self) -> None:
        self.sb.cleanup()

    def test_dry_run不写回(self) -> None:
        p = self.sb.write("m.py", 'x = "OLD"\n')
        res = safe_batch_replace([p], [(r"OLD", "NEW")], dry_run=True,
                                 root=self.sb.root)
        self.assertFalse(res["applied"])
        self.assertIn(str(p), res["changed"])
        self.assertEqual(self.sb.read("m.py"), 'x = "OLD"\n')

    def test_apply写回且内容正确(self) -> None:
        p = self.sb.write("m.py", 'x = "OLD"\n')
        res = safe_batch_replace([p], [(r"OLD", "NEW")], dry_run=False,
                                 root=self.sb.root)
        self.assertTrue(res["applied"])
        self.assertEqual(self.sb.read("m.py"), 'x = "NEW"\n')
        ast.parse(self.sb.read("m.py"))   # 写回后仍必须是合法 Python

    def test_未变化的文件不计入changed(self) -> None:
        p = self.sb.write("m.py", 'x = "KEEP"\n')
        res = safe_batch_replace([p], [(r"NOTHERE", "X")], dry_run=False,
                                 root=self.sb.root)
        self.assertEqual(res["changed"], [])
        self.assertFalse(res["applied"], "没有改动却说 applied 了")

    def test_多条规则按顺序应用(self) -> None:
        p = self.sb.write("m.py", "a = 1\n")
        safe_batch_replace([p], [(r"a = 1", "b = 2"), (r"b = 2", "c = 3")],
                           dry_run=False, root=self.sb.root)
        self.assertEqual(self.sb.read("m.py"), "c = 3\n")

    def test_diff能看出改了什么(self) -> None:
        p = self.sb.write("m.py", 'x = "OLD"\n')
        res = safe_batch_replace([p], [(r"OLD", "NEW")], root=self.sb.root)
        joined = "\n".join(res["diffs"][str(p)])
        self.assertIn("-x =", joined)
        self.assertIn("+x =", joined)


class TestNoBasenameCollision(unittest.TestCase):
    """③ 不同目录的同名文件不许互相覆盖（任务书伪代码的真实缺陷）。

    ``Path(tmpdir) / Path(fp).name`` 会让 ``a/util.py`` 与 ``b/util.py``
    落到**同一个临时文件**上 ⇒ 校验过的内容与写回的内容对不上。
    临时目录里必须**镜像相对路径**。
    """

    def setUp(self) -> None:
        self.sb = _Sandbox()

    def tearDown(self) -> None:
        self.sb.cleanup()

    def test_两个同名文件各改各的(self) -> None:
        a = self.sb.write("a/util.py", 'v = "AAA"\n')
        b = self.sb.write("b/util.py", 'v = "BBB"\n')
        res = safe_batch_replace([a, b],
                                 [(r'"AAA"', '"A2"'), (r'"BBB"', '"B2"')],
                                 dry_run=False, root=self.sb.root)
        self.assertTrue(res["applied"])
        self.assertEqual(self.sb.read("a/util.py"), 'v = "A2"\n')
        self.assertEqual(self.sb.read("b/util.py"), 'v = "B2"\n')

    def test_同名文件之一坏掉时两个都不写(self) -> None:
        a = self.sb.write("a/m.py", 'v = "AAA"\n')
        b = self.sb.write("b/m.py", GOOD_SOURCE)
        with self.assertRaises(UnsafeReplaceError):
            safe_batch_replace([a, b], DOCSTRING_BREAKER, dry_run=False,
                               root=self.sb.root)
        self.assertEqual(self.sb.read("a/m.py"), 'v = "AAA"\n')
        self.assertEqual(self.sb.read("b/m.py"), GOOD_SOURCE)


class TestInputValidation(unittest.TestCase):
    def test_文件不存在时报错(self) -> None:
        with self.assertRaises(UnsafeReplaceError):
            safe_batch_replace([Path("__definitely_missing__.py")],
                               [("a", "b")], root=ROOT)

    def test_非python文件不做语法校验(self) -> None:
        """``.md`` 里出现三引号是完全正常的，不许因此拦下来。"""
        sb = _Sandbox()
        try:
            p = sb.write("doc.md", '```\n"""\n```\n')
            res = safe_batch_replace([p], [(r'"""', '"「"')], dry_run=False,
                                     root=sb.root)
            self.assertTrue(res["applied"],
                            "非 .py 文件被语法校验误拦了")
        finally:
            sb.cleanup()


if __name__ == "__main__":
    unittest.main(verbosity=2)
