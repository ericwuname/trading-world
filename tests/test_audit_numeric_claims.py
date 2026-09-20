"""工作线K（数值陈述盘点）的回归测试。

任务书 §6.5 的变异体 M50 是「正则漏掉某类常见表述」——
这类失败不会报错，只会让**扫描覆盖率静默下降**：
清单看起来正常生成，但漏掉的恰恰是最该被抓的那几条。
所以本文件的第一组测试就是**已知语句必须全部命中**。

第二组测"上下文判定"与"清单渲染"：
盘点清单本身必须是能渲染的 markdown（本项目已经在这上面摔过一次）。
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.audit_numeric_claims import (  # noqa: E402
    COMPARISON_PATTERNS,
    scan_report_for_numeric_claims,
)

_SCRIPTS = str(ROOT / "scripts")
while _SCRIPTS in sys.path:
    sys.path.remove(_SCRIPTS)

#: ⚠️ 这 9 条是**第一版实现漏掉过的**（或差点漏掉的）真实表述形态。
#    它们全部取自本项目报告里的真实句式——
#    任何一条被漏掉，都说明扫描的覆盖率出现了已知边界（M50）。
KNOWN_STATEMENTS = [
    "k 从 1.395 拉到 0.481",
    "k 从 0.665 恶化到 0.910",
    "k 从 1.251 改善到 0.765",
    "k=1.252 → 0.765",
    "taker k = 0.455，最接近平方根律",
    "改善了 69.4%",
    "滑点从 -5.3bp 到 -15.6bp",
    "分离度提升 12%",
    "冲击指数从 1.4 变到 0.95",
]


class TestCoverageOfKnownStatements(unittest.TestCase):
    """⭐ M50：已知句式必须全部命中。"""

    def test_已知语句全部命中(self) -> None:
        missed = []
        for s in KNOWN_STATEMENTS:
            if not any(re.search(p, s) for _, p in COMPARISON_PATTERNS):
                missed.append(s)
        self.assertEqual(missed, [],
                         f"这些已知表述没被任何模式命中：{missed}")

    def test_负号不会被漏掉(self) -> None:
        """滑点几乎总是负的——正则不含 ``[-+]?`` 会把最该抓的一类全漏掉。"""
        hits = [k for k, p in COMPARISON_PATTERNS
                if re.search(p, "从 -5.3bp 到 -15.6bp")]
        self.assertTrue(hits, "带负号的区间变化没被命中")

    def test_中间夹动词不会被漏掉(self) -> None:
        """真实报告写「从 A **拉到** B」，不是「从 A 到 B」。"""
        hits = [k for k, p in COMPARISON_PATTERNS
                if re.search(p, "k 从 1.395 拉到 0.481")]
        self.assertTrue(hits, "「从 A 拉到 B」这类夹动词的表述没被命中")

    def test_单独出现的k赋值不会被漏掉(self) -> None:
        """很多结论只写一个 ``k = 0.455``，后面并没有箭头。"""
        hits = [k for k, p in COMPARISON_PATTERNS
                if re.search(p, "taker k = 0.455")]
        self.assertTrue(hits, "单独的 k = X 没被命中")


class TestContextJudgement(unittest.TestCase):
    """上下文里有没有 CI 证据——这是自动层的唯一判据。"""

    def test_附近有CI时标记为有支撑(self) -> None:
        txt = "k 从 1.2 走到 0.9，95% 置信区间为 [0.5, 1.1]，种子数 8。"
        rows = scan_report_for_numeric_claims(txt, label="t")
        self.assertTrue(rows)
        self.assertTrue(all(r["has_ci_nearby_heuristic"] for r in rows),
                        "上下文里有 CI 却没被识别")

    def test_裸奔陈述被标为无支撑(self) -> None:
        txt = "k 从 1.2 走到 0.9。"
        rows = scan_report_for_numeric_claims(txt, label="t")
        self.assertTrue(rows)
        self.assertFalse(any(r["has_ci_nearby_heuristic"] for r in rows),
                         "没有 CI 却被判成有支撑")

    def test_每条命中都带上下文(self) -> None:
        txt = "前言。" * 50 + "k 从 1.2 走到 0.9。" + "后记。" * 50
        rows = scan_report_for_numeric_claims(txt, label="t")
        self.assertTrue(rows)
        for r in rows:
            self.assertIn(r["matched"], r["context"])
            self.assertLessEqual(len(r["context"]), 600)

    def test_报告标签被带上(self) -> None:
        rows = scan_report_for_numeric_claims("k 从 1 到 2", label="某报告")
        self.assertTrue(all(r["report"] == "某报告" for r in rows))


class TestHtmlStripping(unittest.TestCase):
    """扫描 HTML 报告前必须剥掉 style/script/base64，否则假 token 爆炸。"""

    def test_剥掉base64图(self) -> None:
        from scripts.selfcheck import strip_html_noise

        html = ('<p>k 从 1 到 2</p>'
                '<img src="data:image/png;base64,'
                + "A" * 500 + '">')
        out = strip_html_noise(html)
        self.assertNotIn("AAAA", out)
        self.assertIn("k 从 1 到 2", out)

    def test_剥掉style与script(self) -> None:
        from scripts.selfcheck import strip_html_noise

        html = "<style>k{color:red}</style><script>k=1</script><p>k 从 1 到 2</p>"
        out = strip_html_noise(html)
        self.assertNotIn("color:red", out)
        self.assertNotIn("k=1", out)
        self.assertIn("k 从 1 到 2", out)


class TestInventoryRenders(unittest.TestCase):
    """清单必须是能渲染的 markdown（本项目在这上面摔过一次）。"""

    @staticmethod
    def _ncols(line: str) -> int:
        return line.strip().strip("|").replace("\\|", "").count("|") + 1

    def test_清单文件存在且表格可渲染(self) -> None:
        p = ROOT / "docs" / "历史数值陈述分辨力清单.md"
        if not p.exists():
            # ⚠️ mutation 沙箱不复制 docs/ —— 缺文件时跳过而不是断言存在，
            #    否则沙箱基线会红，而基线一红所有变异体都会"变红"（假证据）。
            raise unittest.SkipTest(f"清单不在（mutation 沙箱不复制 docs/）：{p}")
        lines = p.read_text(encoding="utf-8").splitlines()
        i = 0
        bad = []
        while i < len(lines):
            if not lines[i].strip().startswith("|"):
                i += 1
                continue
            block = []
            j = i
            while j < len(lines) and lines[j].strip().startswith("|"):
                block.append(lines[j])
                j += 1
            if len(block) >= 2:
                n = self._ncols(block[0])
                if self._ncols(block[1]) != n:
                    bad.append(f"第 {i + 1} 行：表头/分隔行列数不一致")
                for k, ln in enumerate(block[2:], start=2):
                    if self._ncols(ln) != n:
                        bad.append(f"第 {i + 1 + k} 行：列数与表头不一致")
            i = j
        self.assertEqual(bad, [], "清单里有渲染不了的表格：\n  - " + "\n  - ".join(bad[:6]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
