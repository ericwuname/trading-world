"""前置校验诊断报告的**渲染回归测试**。

为什么单独一个文件
------------------
这是本项目第二次在"文本产物的渲染"上摔跤：上一次是合并计划被 ``"".join``
拼成一行、表格彻底不渲染，而**脚本退出码 0、自检也不看渲染**。
所以凡是"生成给人读的 markdown"的地方，都要就地钉一条结构断言 ——
这份报告也不例外。

本文件测两面（缺任一面这条检查都不可信）：
  ① 该抓的能抓到：列数不一致、未转义的竖线把行拆错；
  ② 真实产物必须过：对 ``docs/前置校验-EA4-诊断报告.md`` 实跑一次，断言零问题。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / "scripts"))  # 追加：插到前面会遮蔽同名包（scripts/gui.py vs gui/）

from scripts.make_verify_report import _ncols, check_tables  # noqa: E402

_SCRIPTS = str(ROOT / "scripts")
while _SCRIPTS in sys.path:
    sys.path.remove(_SCRIPTS)

REPORT = ROOT / "docs" / "前置校验-EA4-诊断报告.md"


class TestNcols(unittest.TestCase):
    """数列数必须**先抹掉转义竖线**，否则臂名里的 ``|`` 会让人报假警。"""

    def test_普通表格(self) -> None:
        self.assertEqual(_ncols("| a | b | c |"), 3)

    def test_转义竖线不算列分隔(self) -> None:
        self.assertEqual(_ncols("| a \\| b | c |"), 2,
                         "把转义竖线算成列分隔了——markdown 里它渲染成字面量")

    def test_只有转义竖线时是单列(self) -> None:
        self.assertEqual(_ncols("| EA.1 uncorrected \\| Hawkes 关 |"), 1)


class TestCheckTablesHasResolution(unittest.TestCase):
    """① 该抓的能抓到。"""

    def test_抓未转义竖线导致的列数错位(self) -> None:
        txt = "| 臂 | k |\n|---|---|\n| EA.1 | Hawkes 关 | 1.395 |\n"
        problems = check_tables(txt)
        self.assertTrue(problems, "列数不一致却没被抓到")
        self.assertIn("第 3 行", problems[0])

    def test_抓表头与分隔行列数不一致(self) -> None:
        txt = "| a | b |\n|---|\n| 1 | 2 |\n"
        problems = check_tables(txt)
        self.assertTrue(problems)
        self.assertIn("表头", problems[0])

    def test_抓整段被拼成一行(self) -> None:
        """这正是上一次真实事故的形态：两行粘成一行。"""
        txt = ("| a | b |\n|---|---|| 1 | 2 || 3 | 4 |\n")
        self.assertTrue(check_tables(txt),
                        "两行粘成一行却没报警——形同没有检查")

    def test_合法表格零问题(self) -> None:
        txt = ("前言\n\n| 臂 | k |\n|---|---|\n| A | 1.0 |\n| B | 2.0 |\n\n后记\n")
        self.assertEqual(check_tables(txt), [])

    def test_转义竖线的合法表格零问题(self) -> None:
        txt = "| 臂 | k |\n|---|---|\n| EA.1 \\| 关 | 1.395 |\n"
        self.assertEqual(check_tables(txt), [],
                         "含转义竖线的合法表格被误报")


class TestRealReportRenders(unittest.TestCase):
    """② 真实产物必须过——这条才是对本轮交付物的直接保证。

    ⚠️ **缺文件时必须显式跳过，不能直接断言存在**：
    mutation 沙箱只复制 ``gui/scripts/strategies/tests/tw`` + ``data``，
    **不复制 ``docs/``**。硬断言会让沙箱基线变红——
    而**基线一红，每个变异体都会"变红"**，整份变异验证就变成假证据
    （本项目已经吃过一次同源的亏，见 mutation_check 的模块文档）。
    """

    @staticmethod
    def _skip_if_missing() -> None:
        if not REPORT.exists():
            raise unittest.SkipTest(
                f"报告不在（mutation 沙箱不复制 docs/）：{REPORT}")

    def test_报告存在(self) -> None:
        self._skip_if_missing()
        self.assertTrue(REPORT.exists())

    def test_报告表格全部可渲染(self) -> None:
        self._skip_if_missing()
        problems = check_tables(REPORT.read_text(encoding="utf-8"))
        self.assertEqual(problems, [],
                         "报告里有表格渲染不了：\n  - " + "\n  - ".join(problems[:6]))

    def test_报告里的核心结论数字在文中(self) -> None:
        """承重数字必须真的出现——否则"写回"这一步白做了。"""
        self._skip_if_missing()
        txt = REPORT.read_text(encoding="utf-8")
        for needle in ("0.455", "1.042", "19/19", "91.692"):
            self.assertIn(needle, txt, f"报告里找不到 {needle}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
