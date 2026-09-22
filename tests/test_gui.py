"""GUI 后端（任务队列 / API / HTTP 服务）的测试。

这里测的都是**不会崩、只会静默出错**的地方：

  · 场景覆盖写回全局注册表 → 之后每次运行都带着上一次的参数，而界面显示的
    还是默认值。症状是"同样的配置跑出不同结果"，极难归因。
  · token / Host / 目录穿越 → 破了不会报错，只是别人能从网页里打到你的本机服务。
  · 作业失败把工作线程带走 → 后续所有作业永远排队，界面只显示"排队中"。
  · 参数校验缺失 → 用户点了运行、等了半天，才知道策略名打错了。

HTTP 层的测试会真的起一个服务，用无代理的 opener 直连回环地址
（本机环境变量里有 http_proxy，不绕过它会被代理挡掉）。
"""

from __future__ import annotations

import json
import sys
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# ⚠️ **强制置顶，不要"已在就跳过"**：别的测试文件可能先插了
# `scripts/`（而那会让 `scripts/gui.py` 遮蔽 `gui/` 包）
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

# ⚠️⚠️ **必须先把 `scripts/` 从 sys.path 里摘掉，再 import gui。**
#
# 仓库根下有 ``gui/`` 包，而 ``scripts/gui.py`` 是**同名模块**。
# 只要 ``scripts/`` 排在 sys.path 前面，``from gui import api`` 就会解析到
# ``scripts/gui.py``（它自己也 import gui，于是报
# ``partially initialized module``）。
#
# 这个坑真实发生过：新加的一个测试文件（按字母序排在 test_gui 之前）
# 把 ``scripts/`` 放进了 sys.path，于是**基线里 test_gui 整模块导入失败**。
# 症状极具误导性——看起来像"新代码把 GUI 弄坏了"，其实是模块解析顺序。
#
# 在这里（失败点）做一次净化：让本模块的导入**不依赖**别的测试怎么改 sys.path。
_SCRIPTS_DIR = str(ROOT / "scripts")
while _SCRIPTS_DIR in sys.path:
    sys.path.remove(_SCRIPTS_DIR)

from gui import api  # noqa: E402
from gui.jobs import CHUNK, Job, JobManager  # noqa: E402
from gui.server import GuiServer  # noqa: E402
from tw.scenarios import SCENARIOS  # noqa: E402


class TestJobManager(unittest.TestCase):
    def test_作业跑完并带回结果(self) -> None:
        mgr = JobManager(workers=1)

        def fn(job: Job):
            job.phase = "干活"
            job.progress = 0.5
            return {"ok": True}, {"x": [1, 2, 3]}

        job = mgr.submit("market", {"a": 1}, fn)
        for _ in range(100):
            time.sleep(0.02)
            if job.state in ("done", "error"):
                break
        self.assertEqual(job.state, "done")
        self.assertEqual(job.result, {"ok": True})
        self.assertEqual(job.series, {"x": [1, 2, 3]})
        self.assertEqual(job.progress, 1.0)
        mgr.shutdown()

    def test_作业抛异常不会带走工作线程(self) -> None:
        """工作线程是池里共享的资源。它死了之后**所有后续作业**都会永远排队，
        而界面只会显示"排队中"，看不出任何异常。"""
        mgr = JobManager(workers=1)
        bad = mgr.submit("market", {}, lambda job: (_ for _ in ()).throw(RuntimeError("炸")))
        for _ in range(100):
            time.sleep(0.02)
            if bad.state in ("done", "error"):
                break
        self.assertEqual(bad.state, "error")
        self.assertIn("炸", bad.error)
        self.assertTrue(bad.traceback)

        # 关键：池还活着，下一个作业必须能跑
        good = mgr.submit("market", {}, lambda job: ({"ok": 1}, None))
        for _ in range(100):
            time.sleep(0.02)
            if good.state in ("done", "error"):
                break
        self.assertEqual(good.state, "done", "工作线程被上一个作业带走了")
        mgr.shutdown()

    def test_取消排队中的作业(self) -> None:
        mgr = JobManager(workers=1)
        gate = mgr.submit("market", {}, lambda job: (time.sleep(0.5) or {"ok": 1}, None))
        second = mgr.submit("market", {}, lambda job: ({"ok": 2}, None))
        self.assertTrue(mgr.cancel(second.id))
        for _ in range(200):
            time.sleep(0.02)
            if gate.state == "done" and second.state != "queued":
                break
        self.assertEqual(second.state, "cancelled")
        self.assertIsNone(second.result)
        mgr.shutdown()

    def test_作业登记表不会无限增长(self) -> None:
        """series 可能很大，登记表必须有上限，否则跑一天就把内存吃光。"""
        mgr = JobManager(workers=1)
        for i in range(70):
            mgr.submit("market", {"i": i}, lambda job: ({"ok": 1}, None))
        time.sleep(1.2)
        self.assertLessEqual(len(mgr.list()), 60)
        mgr.shutdown()


class TestApiValidate(unittest.TestCase):
    def test_未知策略被拒(self) -> None:
        with self.assertRaises(api.ApiError) as cm:
            api.validate_spec({"kind": "strategy", "strategy": "不存在", "scenario": "normal"})
        self.assertIn("未知策略", cm.exception.message)

    def test_自定义代码语法错误被拒并指出行号(self) -> None:
        with self.assertRaises(api.ApiError) as cm:
            api.validate_spec({"kind": "strategy", "strategy": "custom",
                               "code": "def f(:\n  pass", "scenario": "normal"})
        self.assertIn("语法错误", cm.exception.message)
        self.assertIn("行", cm.exception.message)

    def test_自定义代码里没有Strategy子类被拒(self) -> None:
        with self.assertRaises(api.ApiError) as cm:
            api.validate_spec({"kind": "strategy", "strategy": "custom",
                               "code": "x = 1", "scenario": "normal"})
        self.assertIn("Strategy", cm.exception.message)

    def test_参数覆盖不是对象被拒(self) -> None:
        with self.assertRaises(api.ApiError) as cm:
            api.validate_spec({"kind": "strategy", "strategy": "mm_naive",
                               "scenario": "normal", "params": "not-an-object"})
        self.assertIn("JSON 对象", cm.exception.message)

    def test_批量规模超上限被拒(self) -> None:
        """⭐ 护栏必须**真的能触发**。

        最初只卡"组合数 ≤ 300"，而 7 策略 × 5 场景 × 8 种子 = 280 —— 上限永远
        够不着，等于没有护栏。这条测试就是把这个死角钉死：
        只卡组合数不够，还要卡真正的成本驱动量（总 tick·次）。
        """
        from strategies import REGISTRY

        full = {"kind": "lab", "strategies": list(REGISTRY),
                "scenarios": ["normal", "calm", "stressed", "liquidation", "thin"],
                "n_seeds": 8}
        # 小规模要能通过（否则护栏把正常用法也堵死了）
        api.validate_spec({**full, "ticks": 200})
        # 大规模必须被拦
        with self.assertRaises(api.ApiError) as cm:
            api.validate_spec({**full, "ticks": 20_000})
        self.assertIn("规模过大", cm.exception.message)

    def test_组合数与规模上限都依据真实上限(self) -> None:
        """确认两个上限都不是"够不着的摆设"。"""
        from strategies import REGISTRY

        max_combo = (len(REGISTRY) + 1) * len(SCENARIOS) * 8
        self.assertGreater(max_combo, api.MAX_LAB_RUNS * 0.5,
                           "组合数上限远高于可能的最大值，是死代码")
        # 最大可能规模必须能超过 tick 上限，否则那条护栏同样是摆设
        self.assertGreater(max_combo * 20_000, api.MAX_LAB_TICKS)

    def test_合法参数通过(self) -> None:
        api.validate_spec({"kind": "market", "scenario": "normal", "ticks": 500})
        api.validate_spec({"kind": "strategy", "strategy": "mm_naive", "scenario": "thin"})


class TestScenarioOverride(unittest.TestCase):
    """⭐ 场景覆盖绝不能写回全局注册表。

    这是最容易犯、最难查的一类错：`SCENARIOS` 是模块级单例，
    `_scenario_from_spec` 里少一次复制，用户的每次覆盖就会**永久生效**。
    症状是"界面显示的配置和实际跑的不一样"，而且重启进程才恢复。
    """

    def test_覆盖不污染全局注册表(self) -> None:
        before = dict(SCENARIOS["thin"].overrides)
        before_agents = SCENARIOS["thin"].n_agents
        sc = api._scenario_from_spec({"scenario": "thin", "n_agents": 123, "warmup": 7})
        self.assertEqual(sc.n_agents, 123)
        self.assertEqual(sc.warmup, 7)
        self.assertEqual(SCENARIOS["thin"].n_agents, before_agents, "全局场景被改了")
        self.assertEqual(dict(SCENARIOS["thin"].overrides), before, "全局覆盖字典被改了")
        # 再取一次必须还是原值
        again = api._scenario_from_spec({"scenario": "thin"})
        self.assertEqual(again.n_agents, before_agents)

    def test_mix字典也是复制的(self) -> None:
        sc = api._scenario_from_spec({"scenario": "normal"})
        sc.mix["zero_intel"] = 0.99
        self.assertNotEqual(SCENARIOS["normal"].mix["zero_intel"], 0.99)


class TestApiPayloads(unittest.TestCase):
    def test_meta结构完整(self) -> None:
        m = api.meta()
        self.assertIn("scenarios", m)
        self.assertIn("strategies", m)
        self.assertIn("limits", m)
        self.assertGreaterEqual(len(m["scenarios"]), 5)
        names = [s["name"] for s in m["scenarios"]]
        self.assertIn("normal", names)
        self.assertIn("liquidation", names)

    def test_下采样保留端点(self) -> None:
        import numpy as np

        a = np.arange(5000.0)
        d, stride = api._decimate(a, 1000)
        self.assertGreater(stride, 1)
        self.assertLessEqual(d.size, 1000)
        self.assertEqual(d[0], a[0])
        self.assertGreater(d[-1], a[-1] * 0.999)  # 尾部不能丢太多

    def test_直方图含正态对照且不含NaN(self) -> None:
        import numpy as np

        rng = np.random.default_rng(0)
        # 故意造一段有异常值的序列，检查 0.1% 截断与 JSON 安全性
        mid = 60000 * np.exp(np.cumsum(rng.normal(0, 0.0005, 4000)))
        mid[100] *= 1.5
        h = api._returns_hist(mid)
        self.assertIsNotNone(h)
        self.assertEqual(len(h["centers"]), len(h["count"]))
        self.assertEqual(len(h["count"]), len(h["normal"]))
        self.assertGreater(h["n_outside"], 0, "截断应当确实排除了极端值")
        # 不能出现 NaN/inf：json.dumps 会写出非法 JSON 字面量，前端 JSON.parse 直接抛
        blob = json.dumps(h, default=api._jsonable)
        self.assertNotIn("NaN", blob)
        self.assertNotIn("Infinity", blob)

    def test_真实数据载荷(self) -> None:
        r = api.real_payload("BTCUSDT_1h")
        self.assertEqual(r["symbol"], "BTCUSDT_1h")
        self.assertGreater(r["n"], 1000)
        self.assertIn("excess_kurtosis", r["metrics"])
        with self.assertRaises(api.ApiError):
            api.real_payload("不存在的数据集")

    def test_CSV导出可读(self) -> None:
        job = Job(id="x", kind="strategy", spec={})
        job.result = {"exec": {"pnl_capture": 1.5, "note": "含,逗号"}, "label": "A"}
        csv = api.export_csv(job)
        self.assertTrue(csv.startswith("key,value"))
        self.assertIn("exec.pnl_capture,1.5", csv)
        self.assertIn('"含,逗号"', csv)  # 逗号必须被引号包起来


class TestHttpServer(unittest.TestCase):
    server: GuiServer

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = GuiServer(port=0, workers=1)
        cls.server.start_background()
        assert cls.server.wait_ready(5.0), "服务没能启动"
        cls.op = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.stop()

    def _get(self, path: str, headers: dict | None = None, host: str | None = None):
        req = urllib.request.Request(self.server.base_url + path, headers=headers or {})
        if host:
            req.add_header("Host", host)
        try:
            r = self.op.open(req, timeout=20)
            return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def test_首页可访问且不加载外部资源(self) -> None:
        s, b = self._get("/")
        self.assertEqual(s, 200)
        html = b.decode("utf-8")
        self.assertIn("交易世界", html)
        # 离线可用：不能有 CDN / 外部字体 / 外部脚本
        for bad in ("http://cdn", "https://cdn", "unpkg.com", "jsdelivr", "googleapis"):
            self.assertNotIn(bad, html, f"首页引用了外部资源 {bad}，离线就打不开")

    def test_没有token的API请求被拒(self) -> None:
        s, _ = self._get("/api/meta")
        self.assertEqual(s, 403)

    def test_带token可以访问(self) -> None:
        s, b = self._get("/api/meta", {"X-TW-Token": self.server.token})
        self.assertEqual(s, 200)
        self.assertIn("scenarios", json.loads(b))

    def test_错误的token被拒(self) -> None:
        s, _ = self._get("/api/meta", {"X-TW-Token": "wrong-token"})
        self.assertEqual(s, 403)

    def test_非回环Host被拒(self) -> None:
        """拦住 DNS rebinding：让恶意域名解析到 127.0.0.1 也打不进来。"""
        s, _ = self._get("/api/meta", {"X-TW-Token": self.server.token},
                         host="evil.example.com")
        self.assertEqual(s, 403)

    def test_目录穿越被拒(self) -> None:
        for p in ("/static/../tw/eval.py", "/static/..%2Ftw/eval.py",
                  "/static/../../etc/passwd"):
            s, _ = self._get(p, {"X-TW-Token": self.server.token})
            self.assertIn(s, (403, 404), f"{p} 竟然返回了 {s}")

    def test_未知路径404(self) -> None:
        s, _ = self._get("/api/nope", {"X-TW-Token": self.server.token})
        self.assertEqual(s, 404)

    def test_非法JSON请求体被拒而不是500(self) -> None:
        req = urllib.request.Request(
            self.server.base_url + "/api/run", data=b"{not json",
            headers={"X-TW-Token": self.server.token, "Content-Type": "application/json"})
        try:
            self.op.open(req, timeout=20)
            self.fail("应当被拒")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 400)

    def test_非法参数在提交时就报错(self) -> None:
        """不能让用户等作业跑完才知道参数写错了。"""
        body = json.dumps({"kind": "strategy", "strategy": "不存在", "scenario": "normal"})
        req = urllib.request.Request(
            self.server.base_url + "/api/run", data=body.encode(),
            headers={"X-TW-Token": self.server.token, "Content-Type": "application/json"})
        try:
            self.op.open(req, timeout=20)
            self.fail("应当被拒")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 400)
            self.assertIn("未知策略", json.loads(e.read())["error"])

    def test_完整跑一次策略作业(self) -> None:
        """端到端：提交 → 轮询 → 取时序 → 结果自洽。"""
        body = json.dumps({
            "kind": "strategy", "scenario": "normal", "seed": 7,
            "ticks": 300, "warmup": 400, "strategy": "mm_naive",
        })
        req = urllib.request.Request(
            self.server.base_url + "/api/run", data=body.encode(),
            headers={"X-TW-Token": self.server.token, "Content-Type": "application/json"})
        jid = json.loads(self.op.open(req, timeout=20).read())["job_id"]
        for _ in range(300):
            time.sleep(0.1)
            _, b = self._get("/api/job/" + jid, {"X-TW-Token": self.server.token})
            j = json.loads(b)
            if j["state"] in ("done", "error"):
                break
        self.assertEqual(j["state"], "done", j.get("error"))
        e = j["result"]["exec"]
        self.assertGreater(e["n_fills"], 0)
        # PnL 分解恒等式：这条是评估代码的命门，端到端也必须成立
        self.assertLess(abs(e["pnl_identity_residual"]), 1e-6)

        _, b = self._get(f"/api/job/{jid}/series", {"X-TW-Token": self.server.token})
        ser = json.loads(b)["series"]
        self.assertTrue(ser["ready"] if "ready" in ser else True)
        self.assertGreater(len(ser["mid"]), 10)
        self.assertEqual(len(ser["mid"]), len(ser["tick"]))

        s, b = self._get(f"/api/export/{jid}.csv", {"X-TW-Token": self.server.token})
        self.assertEqual(s, 200)
        self.assertIn(b"pnl_capture", b)


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
