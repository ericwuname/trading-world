"""A9b 的**嵌套对照**逻辑：`make_a9_report._stat_n`。

为什么这个也要测（与 `tests/test_a6_power.py` 同一条理由）：
§3.1 的结论（「补样本后效应变大还是变小」）**直接决定用户要不要继续投额度**，
而它由 `_stat_n(..., m=48)` 这一个截断动作算出来。
截断方向错了、或配对下标错位了，**不会报错**——
只会让报告给出一个**看起来很有依据的错结论**（本项目最怕的"沉默失败"）。

⭐ 本文件钉住两条语义，并各配了一个**变异**验证过它真能变红：

| 变异 | 会被哪个测试抓到 |
|---|---|
| 截断改成「从末尾取 m 段」 | `test_从前端截断而不是从末尾` |
| 配对改成「两列各自排序后再配」 | `test_按下标配对而不是先排序` |

⚠️ 第二个变异有个坑（我第一版就踩了）：**排序配对不改变均值**——
        Σ(nb−na) 与排列无关 ⇒ mean 永远相同；它只改变 **σ**。
        所以那条断言必须落在 σ 上，只断均值的话变异能安然通过。
"""

from __future__ import annotations

import importlib.util
import json
import statistics as st
import sys
import tempfile
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "scripts" / "make_a9_report.py"

# ⚠️⚠️ ``make_a9_report`` 在自己模块顶部做 ``sys.path.insert(0, <scripts/>)``
# （它要能 ``import tw.*``）。**这个副作用会泄漏给同进程里后面导入的测试**：
# ``scripts/`` 一旦排在最前，``from gui import agent_api`` 就会命中
# ``scripts/gui.py``（一个模块）而不是根目录的 ``gui/``（一个包），
# 于是 ``tests/test_agent_gui.py`` 以
# ``ModuleNotFoundError: No module named 'gui.cli'; 'gui' is not a package`` 挂掉。
# 踩过一次：全量从「1519 通过」变成「1497 + 1 error」（少掉的那 22 条正是它）。
# ⇒ 只在本模块导入期间借一下 sys.path，导完立刻还原。
_saved_path = list(sys.path)
try:
    _spec = importlib.util.spec_from_file_location("_a9_report", _SRC)
    assert _spec and _spec.loader
    a9 = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(a9)
finally:
    sys.path[:] = _saved_path

#: 用 `llm_v6` 当键，省得再拉一个臂进来。
K = "llm_v6"


class _Fixture(unittest.TestCase):
    """给每个测试一块自己的临时目录。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def pair(self, a: list[float], b: list[float],
             na: str = "a.json", nb: str = "b.json") -> tuple[Path, Path]:
        pa, pb = self.tmp / na, self.tmp / nb
        for p, vals in ((pa, a), (pb, b)):
            p.write_text(json.dumps(
                {"result": {"net": [{K: v} for v in vals]}}), encoding="utf-8")
        return pa, pb


class TestStatN(_Fixture):
    def test_从前端截断而不是从末尾(self) -> None:
        """⭐ 方向测试：信号只在前半 ⇒ 取前 30 段必须是**正**效应。

        变异「从末尾取 m 段」会拿到后 30 段（负效应）⇒ 变红。
        """
        a = [0.0] * 60
        b = [0.001 + (i % 3) * 1e-5 for i in range(30)] + \
            [-0.001 + (i % 3) * 1e-5 for i in range(30)]
        pa, pb = self.pair(a, b)

        s = a9._stat_n(pa, K, pb, K, 30)
        self.assertIsNotNone(s)
        self.assertEqual(s["n"], 30)
        # 独立算一遍当预言（不是抄实现）
        expect = st.mean([b[i] - a[i] for i in range(30)])
        self.assertAlmostEqual(s["effect"], expect, places=12)
        self.assertGreater(s["effect"], 0.0,
                           "前 30 段是正信号；截断方向错了才会变负")

        # 整体效应应当明显不同（否则这个函数没在截断）
        s_all = a9._stat_n(pa, K, pb, K, None)
        self.assertIsNotNone(s_all)
        self.assertEqual(s_all["n"], 60)
        self.assertNotAlmostEqual(s["effect"], s_all["effect"], places=6)

    def test_按下标配对而不是先排序(self) -> None:
        """⭐ 配对语义：差值必须是 ``nb[i]-na[i]``，**不能各自排序后再配**。

        数据取 ``a=0..29``、``b=a 的倒序 + 0.001``：
        - 下标配对 ⇒ 差值 29−2i+0.001，**σ 很大**；
        - 排序配对 ⇒ 差值恒为 0.001，**σ = 0**。

        ⚠️ 两者的 **mean 都是 0.001**（Σ(nb−na) 与排列无关），
        所以必须断 σ —— 只断 mean 的话这个变异抓不到。
        """
        n = 30
        a = [float(i) for i in range(n)]
        b = [float(n - 1 - i) + 0.001 for i in range(n)]
        pa, pb = self.pair(a, b)

        s = a9._stat_n(pa, K, pb, K, n)
        self.assertIsNotNone(s)
        self.assertAlmostEqual(s["sd"], st.stdev([b[i] - a[i] for i in range(n)]),
                               places=12)
        self.assertAlmostEqual(s["effect"], 0.001, places=12)
        self.assertGreater(s["sd"], 1.0,
                           "排序配对会把 σ 压成 0（最小方差匹配），这里必须非 0")

    def test_m等于None时与_stat完全一致(self) -> None:
        """m=None 必须是"全部"，不能变成 0 段或只剩一段。

        ⚠️ 这里刻意用**能让排序配对露出马脚**的数据，
        于是这个测试同时覆盖 ``_stat`` 与 ``_stat_n`` 两条路径。
        """
        n = 20
        a = [float(i) for i in range(n)]
        b = [float(n - 1 - i) + 0.002 for i in range(n)]
        pa, pb = self.pair(a, b)

        s = a9._stat(pa, K, pb, K)
        s_n = a9._stat_n(pa, K, pb, K, None)
        self.assertIsNotNone(s)
        self.assertIsNotNone(s_n)
        for k in ("n", "effect", "sd", "mde", "ratio", "t"):
            self.assertEqual(s[k], s_n[k], f"字段 {k} 在 m=None 时应与 _stat 一致")

    def test_截断长度大于总长时按总长算(self) -> None:
        """m > n 不该报错，也不该凭空补段。"""
        pa, pb = self.pair([float(i) for i in range(7)],
                           [float(i) + 1 for i in range(7)])
        s = a9._stat_n(pa, K, pb, K, 999)
        self.assertIsNotNone(s)
        self.assertEqual(s["n"], 7)

    def test_截断后样本不足时返回None(self) -> None:
        """n<2 时返回 None——报告据此打印「（无产物）」，而不是除以 0。"""
        pa, pb = self.pair([0.0, 1.0], [1.0, 2.0])
        for m in (0, 1):
            self.assertIsNone(a9._stat_n(pa, K, pb, K, m), f"m={m} 应返回 None")

    def test_两臂长度不等时取较短(self) -> None:
        """与 ``_stat`` 一致：取 ``min(len(na), len(nb))``，不越界。"""
        pa, pb = self.pair([float(i) for i in range(10)],
                           [float(i) + 1 for i in range(4)])
        s = a9._stat_n(pa, K, pb, K, None)
        self.assertIsNotNone(s)
        self.assertEqual(s["n"], 4)


class TestRanges与Kpi读取(_Fixture):
    """`_ranges` / `_kpi_of` 读不到东西时必须给空值，不能抛。"""

    def test_缺文件不抛(self) -> None:
        missing = self.tmp / "__definitely_missing__.json"
        self.assertEqual(a9._ranges(missing), [])
        self.assertEqual(a9._kpi_of(missing), {})

    def test_坏JSON不抛(self) -> None:
        p = self.tmp / "bad.json"
        p.write_text("{not json", encoding="utf-8")
        self.assertEqual(a9._ranges(p), [])
        self.assertEqual(a9._kpi_of(p), {})

    def test_正常读出ranges与kpi(self) -> None:
        p = self.tmp / "ok.json"
        p.write_text(json.dumps({
            "result": {"net": [{K: 0.0}], "ranges": [[16, 23], [24, 31]]},
            "kpi": {"min_presence": 0.4444},
        }), encoding="utf-8")
        self.assertEqual(a9._ranges(p), [[16, 23], [24, 31]])
        self.assertEqual(a9._kpi_of(p)["min_presence"], 0.4444)


if __name__ == "__main__":
    unittest.main()
