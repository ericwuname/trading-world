"""A5 测试：GUI 的 Agent 留痕接口 + 首屏预载 + JSON 互操作。

**全部离线**（起一个临时 HTTP 服务打自己的接口，不回外网）。

本轮的两个重点：
1. **只读且不接受用户路径**——那个界面还允许跑代码，不该再开一个越权口子。
2. ⭐ **NaN 不是合法 JSON**——真踩到的互操作 bug：Python 写出的
   ``NaN`` 自己读得回来（解析器是超集），但 JS 的 ``JSON.parse`` 直接抛错。
   症状是"Python 侧全绿、页面上写着读取失败"。
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gui import agent_api as A  # noqa: E402

from tw.decision_log import DecisionLog, DecisionRecord  # noqa: E402
from tw.eval_agent import json_safe  # noqa: E402


def _mk_rec(tick: int, *, equity: float = 100_000.0, action: str = "hold",
           basis: str = "", executed: bool = False) -> DecisionRecord:
    return DecisionRecord(
        run_id="T", tick=tick, agent_id="a1", context_hash="h",
        visible_state={"mid": 100.0, "equity": equity, "inventory": 0.0},
        prompt_template="v2", model="m",
        parsed={"action": action, "reason": "理由充分", "basis": basis},
        parse_ok=True, executed=executed,
        risk={"rule": "hold", "accepted": True},
        llm_raw="raw", latency_ms=10, mode="live",
    )


# ======================================================================
# ⭐ NaN 不是合法 JSON（真踩到的互操作 bug）
# ======================================================================
class TestJsonInterop(unittest.TestCase):
    def test_nan不是合法JSON(self):
        """⭐⭐ 这是本轮最重要的回归守卫。

        ``json.dumps(float("nan"))`` 默认输出裸的 ``NaN``——
        而 ``NaN`` **不是合法 JSON**。Python 读得回来（超集），
        所以**自测全绿**；但 JS 的 ``JSON.parse`` 直接报
        ``Unexpected token 'N'``。

        ⇒ 产物对任何非 Python 消费者都是坏的，而**我们自己发现不了**。
        本项目真的撞上了：整页退化成"读取失败"，而 Python 侧一切正常。
        """
        import json as _j

        self.assertIn("NaN", _j.dumps({"x": float("nan")}))
        # 修法：sanitize + allow_nan=False（写不出来才是安全的）
        with self.assertRaises(ValueError):
            _j.dumps({"x": float("nan")}, allow_nan=False)
        self.assertEqual(_j.dumps(json_safe({"x": float("nan")})), '{"x": null}')

    def test_json_safe递归处理(self):
        o = {"a": [1.0, float("inf"), {"b": float("-inf")}],
             "c": float("nan"), "d": "ok", "e": None}
        got = json_safe(o)
        self.assertEqual(got["a"][0], 1.0)
        self.assertIsNone(got["a"][1])
        self.assertIsNone(got["a"][2]["b"])
        self.assertIsNone(got["c"])
        self.assertEqual(got["d"], "ok")
        self.assertIsNone(got["e"])
        # 能产出合法 JSON（关键：能被 JS 解析）
        blob = json.dumps(got, allow_nan=False)
        self.assertNotIn("NaN", blob)
        self.assertNotIn("Infinity", blob)

    def test_noop的夏普是nan而产物仍是合法JSON(self):
        """⭐ 触发源：不交易时 ``sharpe`` 按设计返回 ``nan``
        （"算不出来"），于是**最基准的那条配置**产出了非法 JSON。"""
        from tw.eval_agent import sharpe

        self.assertTrue(math.isnan(sharpe([])))
        self.assertIsNone(json_safe({"sharpe": sharpe([])})["sharpe"])
        self.assertEqual(
            json.dumps(json_safe({"sharpe": sharpe([])}), allow_nan=False),
            '{"sharpe": null}')


# ======================================================================
# 目录扫描与安全
# ======================================================================
class TestDiscover(unittest.TestCase):
    def test_只扫固定目录(self):
        """⭐ 用户提供的路径**一律不接受**——那等于给一个允许跑代码的
        本机服务再开一个任意文件读的口子。"""
        es = A.discover(ROOT)
        for e in es:
            self.assertIn(e.source_dir, A.RUN_DIRS)
            self.assertTrue(e.dec_path.is_file())

    def test_id不含路径分隔符(self):
        for e in A.discover(ROOT):
            self.assertNotIn("/", e.id)
            self.assertNotIn("\\", e.id)
            self.assertNotIn("..", e.id)

    def test_未知id报错(self):
        with self.assertRaises(KeyError):
            A.run_summary(ROOT, "不存在的运行")

    def test_越权id被拒(self):
        """``../`` 之类不可能命中扫描出来的条目。"""
        for bad in ("../../etc/passwd", "..%2F..%2Fx", "docs/../../secret"):
            with self.subTest(bad=bad):
                with self.assertRaises(KeyError):
                    A.run_summary(ROOT, bad)

    def test_空目录不崩(self):
        with tempfile.TemporaryDirectory() as d:
            got = A.list_runs(Path(d))
            self.assertEqual(got["n"], 0)


# ======================================================================
# 取数
# ======================================================================
class _TmpRun:
    """在临时目录里造一个可被扫描到的运行。"""

    def __init__(self) -> None:
        self.dir = None

    def __enter__(self):
        self.dir = tempfile.TemporaryDirectory()
        d = Path(self.dir.name) / "out" / "a4"
        d.mkdir(parents=True)
        # ⚠️ DecisionLog 的 ``__enter__`` 自己会 open，**不要**先 open 再 with
        # （重复 open 会 RuntimeError）。这与 "with open() as f" 的直觉不同。
        with DecisionLog(d / "dec_TEST.jsonl") as log:
            for i in range(5):
                log.append(_mk_rec(i, equity=100_000.0 + i * 10))
        (d / "eval_TEST.json").write_text(
            json.dumps({"target": "T", "config": {"inst": "X", "n": 5},
                        "evals": {}, "comparison": None}), encoding="utf-8")
        return Path(self.dir.name)

    def __exit__(self, *exc):
        self.dir.cleanup()


class TestRunQueries(unittest.TestCase):
    def test_列表含新造的运行(self):
        with _TmpRun() as root:
            runs = A.list_runs(root)
            self.assertEqual(runs["n"], 1)
            self.assertTrue(runs["runs"][0]["has_eval"])

    def test_概要与曲线来自留痕(self):
        """⭐ 权益曲线**不是另存的**——它取自每条留痕①可见状态的
        ``equity``。留痕本来就是为回放而存的，画图只是顺带。"""
        with _TmpRun() as root:
            rid = A.list_runs(root)["runs"][0]["id"]
            s = A.run_summary(root, rid)
            self.assertEqual(s["n"], 5)
            c = s["curve"]
            self.assertEqual(c["n"], 5)
            self.assertEqual(c["initial_equity"], 100_000.0)
            self.assertEqual(c["final_equity"], 100_040.0)
            self.assertTrue(c["per_bar_decision"])

    def test_分页且列表不含正文(self):
        """⚠️ 列表只回摘要——正文（llm_raw / visible_state）占绝大部分
        体积，塞进列表会让几百条的运行把浏览器拖死。"""
        with _TmpRun() as root:
            rid = A.list_runs(root)["runs"][0]["id"]
            p = A.decisions_page(root, rid, offset=1, limit=2)
            self.assertEqual(p["total"], 5)
            self.assertEqual([r["tick"] for r in p["rows"]], [1, 2])
            for r in p["rows"]:
                self.assertNotIn("llm_raw", r)
                self.assertNotIn("visible_state", r)

    def test_分页上限被夹住(self):
        with _TmpRun() as root:
            rid = A.list_runs(root)["runs"][0]["id"]
            p = A.decisions_page(root, rid, limit=10_000)
            self.assertLessEqual(p["limit"], A.MAX_PAGE)

    def test_序号越界报错(self):
        with _TmpRun() as root:
            rid = A.list_runs(root)["runs"][0]["id"]
            with self.assertRaises(KeyError):
                A.decision_detail(root, rid, 99)

    def test_详情按证据链六项分组(self):
        """⭐ 留痕字段名是给机器/SQL 用的（扁平、好查），
        而这个接口是给人看的——所以分组返回。"""
        with _TmpRun() as root:
            rid = A.list_runs(root)["runs"][0]["id"]
            d = A.decision_detail(root, rid, 0)
            for k in ("1_visible", "2_belief", "3_suggestion",
                      "4_whether_traded", "5_risk", "6_order"):
                self.assertIn(k, d["chain"])
            self.assertIn("llm_raw", d["raw"])
            self.assertIn("decision_id", d["identity"])

    def test_首屏预载打包齐了(self):
        with _TmpRun() as root:
            d = A.first_paint(root)
            self.assertEqual(len(d["runs"]), 1)
            self.assertTrue(d["sel"])
            self.assertIsNotNone(d["summary"])
            self.assertIsNotNone(d["page"])

    def test_首屏可指定运行(self):
        with _TmpRun() as root:
            rid = A.list_runs(root)["runs"][0]["id"]
            self.assertEqual(A.first_paint(root, rid)["sel"], rid)
            # 给了不存在的 id 要**回退**而不是报错（深链不该白屏）
            self.assertEqual(A.first_paint(root, "不存在")["sel"], rid)

    def test_首屏数据可序列化为合法JSON(self):
        """⭐ 首屏是要**嵌进 HTML** 的，所以必须是标准 JSON。"""
        with _TmpRun() as root:
            blob = json.dumps(json_safe(A.first_paint(root)),
                              ensure_ascii=False, allow_nan=False)
            self.assertNotIn("NaN", blob)
            self.assertNotIn("Infinity", blob)
            json.loads(blob)      # 标准解析器能读

    def test_空目录时首屏不崩(self):
        with tempfile.TemporaryDirectory() as d:
            got = A.first_paint(Path(d))
            self.assertEqual(got["runs"], [])
            self.assertIsNone(got["summary"])


# ======================================================================
# 依据自报分布（v4）
# ======================================================================
class TestBasisDistribution(unittest.TestCase):
    def test_统计自报分布(self):
        recs = [_mk_rec(0, basis="rule"), _mk_rec(1, basis="judgment"),
                _mk_rec(2, basis="both"), _mk_rec(3, basis="")]
        b = A.basis_distribution(recs)
        self.assertEqual(b["n"], 4)
        self.assertEqual(b["n_declared"], 3)
        self.assertAlmostEqual(b["frac_of_declared"]["rule"], 1 / 3)

    def test_老模板全未自报时不报一堆100百分比(self):
        """v1~v3 没有 basis 字段——那时不该给"未自报 100%"这种噪声。"""
        b = A.basis_distribution([_mk_rec(0), _mk_rec(1)])
        self.assertEqual(b["n"], 2)
        self.assertEqual(b["n_declared"], 0)
        self.assertEqual(b["frac_of_declared"], {})

    def test_带免责声明(self):
        """⭐ 自报是**声明不是证据**——报告里必须写清楚。"""
        b = A.basis_distribution([_mk_rec(0, basis="rule")])
        self.assertIn("自报", b["caveat"])


# ======================================================================
# HTTP 层（起真服务打自己的接口）
# ======================================================================
class TestHttpRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from gui import server as S

        cls.srv = S.GuiServer(host="127.0.0.1", port=0)
        cls.srv.start_background()

    @classmethod
    def tearDownClass(cls):
        cls.srv.stop()

    def _get(self, path: str):
        import urllib.error
        import urllib.parse
        import urllib.request

        # ⚠️ 路径里可能有非 ASCII（比如故意用一个中文的运行 id 试 404）。
        # http.client 只接受 ASCII 请求行，不编码会抛 UnicodeEncodeError
        # ——那是**测试自己**的问题，不是被测代码的。
        url = self.srv.base_url + urllib.parse.quote(path, safe="/?&=%")
        req = urllib.request.Request(url,
                                     headers={"X-TW-Token": self.srv.token})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def test_缺token被拒(self):
        import urllib.error
        import urllib.request

        try:
            with urllib.request.urlopen(
                    self.srv.base_url + "/api/agent/runs", timeout=30) as r:
                code = r.status
        except urllib.error.HTTPError as e:
            code = e.code
        self.assertEqual(code, 403)

    def test_runs返回合法JSON(self):
        code, body = self._get("/api/agent/runs")
        self.assertEqual(code, 200)
        json.loads(body)                       # 标准解析
        self.assertNotIn(b"NaN", body)

    def test_列表接口的limit不是列表类型(self):
        """⚠️ 踩过：``parse_qs`` 的值是**列表**，直接 ``int()`` 抛
        TypeError ⇒ 用户看到 500 而不是"参数错"。"""
        rid = A.discover(ROOT)[0].id
        code, body = self._get(
            f"/api/agent/run/{rid}/decisions?offset=0&limit=3")
        self.assertEqual(code, 200)
        self.assertEqual(len(json.loads(body)["rows"]), 3)

    def test_非法limit给可读的400(self):
        rid = A.discover(ROOT)[0].id
        code, body = self._get(
            f"/api/agent/run/{rid}/decisions?offset=0&limit=abc")
        self.assertIn(code, (400, 500))
        # 不论是 400 还是兜底的 500，都要有可读信息而不是空响应
        self.assertTrue(body)

    def test_summary含曲线(self):
        rid = A.discover(ROOT)[0].id
        code, body = self._get(f"/api/agent/run/{rid}/summary")
        self.assertEqual(code, 200)
        self.assertTrue(json.loads(body)["curve"]["n"] > 0)

    def test_未知运行给404(self):
        code, _ = self._get("/api/agent/run/不存在/summary")
        self.assertEqual(code, 404)

    def test_越权路径给404(self):
        code, _ = self._get("/api/agent/run/..%2F..%2Fetc%2Fpasswd/summary")
        self.assertEqual(code, 404)

    def test_深链时嵌首屏数据(self):
        """⭐ 嵌数据让首屏**同步**渲染——否则无头截图/DOM dump 会在
        fetch 之前拍，拿到占位图，而"检查有没有内容"会**通过**。
        验证方式骗人比 bug 更危险。"""
        rid = A.discover(ROOT)[0].id
        code, body = self._get(f"/?token=x&tab=agent&run={rid}")
        self.assertEqual(code, 200)
        html = body.decode("utf-8")
        self.assertIn('id="agentPreload"', html)
        i = html.find('id="agentPreload"')
        st = html.find(">", i) + 1
        en = html.find("</script>", st)
        blob = html[st:en]
        d = json.loads(blob)                   # 标准 JSON 解析（JS 也会这么读）
        self.assertEqual(d["sel"], rid)
        self.assertTrue(d["runs"])

    def test_非agent深链不嵌数据(self):
        """只在该嵌的时候嵌——其它情况一个字节都不改。"""
        code, body = self._get("/")
        self.assertEqual(code, 200)
        self.assertNotIn(b'id="agentPreload"', body)

    def test_注入转义(self):
        """⭐⭐ **必须让数据里真的出现 ``</script>``**，否则这条测试是摆设。

        ⚠️ 它第一版就是摆设：断言 ``"</" not in blob``，但测试数据里
        根本没有 ``</``——所以**把转义那行删掉，测试照样绿**。
        这正是本项目的老毛病「变异体没注入任何东西」：
        断言存在 ≠ 有分辨力。修法是造一个**含 ``</script>`` 的理由**。
        """
        from gui import server as S

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            d = root / "out" / "a4"
            d.mkdir(parents=True)
            nasty = "价格突破后回落</script><script>alert(1)</script>"
            with DecisionLog(d / "dec_X.jsonl") as log:
                for i in range(3):
                    rec = _mk_rec(i)
                    rec.parsed["reason"] = nasty
                    log.append(rec)

            html = b"<html><body>hi</body></html>"
            got = S._inject_agent_preload(
                html, {"run": ["out_a4_dec_X.jsonl"]}, root=root)
            text = got.decode("utf-8")

            self.assertIn('id="agentPreload"', text)
            i = text.find('id="agentPreload"')
            st = text.find(">", i) + 1
            en = text.find("</script>", st)
            blob = text[st:en]
            # ① 块内不能出现裸的 `</`（否则浏览器提前结束脚本块 = 注入）
            self.assertNotIn("</", blob, "预载块里出现了未转义的 </")
            # ② 转义后仍要是**合法 JSON**，且内容原样保留
            data = json.loads(blob)
            self.assertIn(nasty, json.dumps(data, ensure_ascii=False))

    def test_数据里没有尖括号时也要能嵌(self):
        """反向：数据干净时不该被转义逻辑弄坏。"""
        from gui import server as S

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            d = root / "out" / "a4"
            d.mkdir(parents=True)
            with DecisionLog(d / "dec_Y.jsonl") as log:
                for i in range(3):
                    log.append(_mk_rec(i))
            got = S._inject_agent_preload(
                b"<html><body>x</body></html>",
                {"run": ["out_a4_dec_Y.jsonl"]}, root=root)
            text = got.decode("utf-8")
            self.assertIn('id="agentPreload"', text)
            i = text.find('id="agentPreload"')
            st = text.find(">", i) + 1
            en = text.find("</script>", st)
            self.assertTrue(json.loads(text[st:en])["runs"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
