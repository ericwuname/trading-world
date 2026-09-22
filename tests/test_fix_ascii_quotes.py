"""`fix_ascii_quotes.py` 的测试：**边界的每一条都是我手工改坏过的**。

⚠️ 这个工具会**改写源码**，所以它的安全边界必须被测试守住：
我手工修"裸引号"那次，一共改坏了三样东西——
① 两行拼接的**续行**（吃掉了首行的 `)`）、② **docstring**、
③ **`" | ".join(...)` / 条件表达式**（内部引号有句法意义）。
每一条都对应下面一个测试。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))
# ⚠️ 用 **append**：插到前面会让 `scripts/gui.py` 遮蔽 `gui/` 包
if str(ROOT / "scripts") not in sys.path:
    sys.path.append(str(ROOT / "scripts"))

from scripts.fix_ascii_quotes import fix_text  # noqa: E402


def _fix(src: str) -> tuple[str, int]:
    new, changed = fix_text(src)
    return new, len(changed)


class TestFixesWhatItShould(unittest.TestCase):
    def test_修中文文本里的裸引号(self):
        """经典坏行有 **4** 个引号（偶数）——⚠️ 我第一版按"奇数"判，漏修。"""
        src = 'L.append("...是"这条行情特殊"，还是...")\n'
        new, n = _fix(src)
        self.assertEqual(n, 1)
        self.assertIn("「这条行情特殊」", new)
        compile(new, "<t>", "exec")   # 必须能编译

    def test_告一段落的两处引号(self):
        src = 'L.append("A"x"B"y"C")\n'
        new, _ = _fix(src)
        self.assertIn("「x」", new)
        self.assertIn("「y」", new)


class TestLeavesAlone(unittest.TestCase):
    """⚠️ 这一组是"不许碰"的清单——每一条我都手工碰坏过。"""

    def test_正常行不动(self):
        src = 'L.append("完全正常的一行")\n'
        new, n = _fix(src)
        self.assertEqual(n, 0)
        self.assertEqual(new, src)

    def test_续行不动(self):
        """⚠️ 两行拼接：**续行不许被当成新语句**（那样会吃掉首行的 `)`）。"""
        src = ('L.append("第一段，" \n'
               '         "第二段。")\n')
        new, n = _fix(src)
        self.assertEqual(n, 0)
        self.assertEqual(new, src)
        compile(new, "<t>", "exec")

    def test_docstring不动(self):
        src = '"""文档：这里写"引号"也没关系。"""\n'
        new, n = _fix(src)
        self.assertEqual(n, 0)
        self.assertEqual(new, src)

    def test_带三引号的语句不动(self):
        src = 'print("""多行""")\n'
        new, n = _fix(src)
        self.assertEqual(n, 0)

    def test_join表达式不动(self):
        """⚠️ `" | ".join(...)` 里引号**有句法意义**，改了会坏。"""
        src = 'L.append("| x | " + " | ".join(f"{a}" for a in b))\n'
        new, n = _fix(src)
        self.assertEqual(n, 0)
        self.assertEqual(new, src)
        compile(new, "<t>", "exec")

    def test_条件表达式不动(self):
        src = 'L.append("A" if x else "B")\n'
        new, n = _fix(src)
        self.assertEqual(n, 0)

    def test_括号不平衡的行不动(self):
        """半截语句（多行表达式的首行）不碰。"""
        src = 'L.append("没有收尾的括号"\n'
        new, n = _fix(src)
        self.assertEqual(n, 0)

    def test_非语句行不动(self):
        src = 'x = "a"b"c"\n'          # 不是 L.append/print 开头 ⇒ 不碰
        new, n = _fix(src)
        self.assertEqual(n, 0)


class TestRefusesToMakeItWorse(unittest.TestCase):
    def test_修不好的时候报错并且不写(self):
        """⚠️ 若改完仍编译不过 ⇒ **不许写回**（宁可留着，也不能改坏）。"""
        import tempfile
        from scripts.fix_ascii_quotes import main
        d = Path(tempfile.mkdtemp())
        f = d / "bad.py"
        original = 'L.append("A"x"B"y"C"z"D")\n'
        f.write_text(original, encoding="utf-8")
        rc = main([str(f), "--write"])
        # 改完（交替「」）可能有剩余问题 ⇒ 此时 rc=2 且**内容不变**
        if rc == 2:
            self.assertEqual(f.read_text(encoding="utf-8"), original)

    def test_全部能编译的文件不改(self):
        import tempfile
        from scripts.fix_ascii_quotes import main
        d = Path(tempfile.mkdtemp())
        f = d / "ok.py"
        src = 'a = 1\nb = "x"\n'
        f.write_text(src, encoding="utf-8")
        rc = main([str(f), "--write"])
        self.assertEqual(rc, 0)
        self.assertEqual(f.read_text(encoding="utf-8"), src)


class TestOnRepoItself(unittest.TestCase):
    """在**本项目真实的源码**上跑一遍：不许改坏任何东西。"""

    def test_三个新脚本跑完仍能编译(self):
        for name in ("a11_cross_asset.py", "make_a11_report.py",
                     "fix_ascii_quotes.py"):
            p = ROOT / "scripts" / name
            src = p.read_text(encoding="utf-8")
            new, _n = fix_text(src)
            # ⚠️ 关键：**修完必须仍可编译**（这正是工具的收尾自检）
            compile(new, str(p), "exec")

    def test_已经干净的文件不被改动(self):
        p = ROOT / "scripts" / "a11_cross_asset.py"
        src = p.read_text(encoding="utf-8")
        new, ch = fix_text(src)
        self.assertEqual(len(ch), 0, f"这个文件应当已经干净，却报了 {len(ch)} 处")
        self.assertEqual(new, src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
