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
import re
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

from gui import agent_api, api, api_doc, routes  # noqa: E402
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

    def test_作业类型只有一份事实源(self) -> None:
        """⭐ 白名单与执行分发必须同源。

        这里原来有**两份** kind 白名单（validate_spec 一份、execute 一份），
        报错文本还一模一样，加类型时漏改一处就会被自己的错误信息误导。
        现在两条路径都必须从 `api.KINDS` 派生——这条测试把它钉死。
        """
        self.assertEqual(sorted(api.KINDS),
                         ["doc_strategy", "lab", "market", "strategy"])
        # 每个登记的类型都能通过校验（否则就是又出现了第二份白名单）
        for kind in api.KINDS:
            spec = {"kind": kind}
            self.assertEqual(api.check_kind(spec), kind)

    def test_执行分发覆盖全部登记类型(self) -> None:
        """execute() 必须能派发每一个 KINDS 里的类型。

        做法：把每个作业函数替换成哨兵，只验证"派发到了正确的函数"，
        不真的跑模拟（那要几秒到几分钟）。运行期若走了 if-链的 default 分支，
        会抛 ApiError 而不是返回哨兵值。
        """
        sentinel = lambda job: ({"marker": job.kind}, None)  # noqa: E731
        for kind in api.KINDS:
            with self.subTest(kind=kind):
                saved = api.KINDS[kind]
                api.KINDS[kind] = sentinel
                try:
                    job = Job(id="t", kind=kind, spec={})
                    result, series = api.execute(job)
                    self.assertEqual(result["marker"], kind)
                    self.assertIsNone(series)
                finally:
                    api.KINDS[kind] = saved

    def test_未知作业类型报错带可用列表(self) -> None:
        with self.assertRaises(api.ApiError) as cm:
            api.validate_spec({"kind": "bogus"})
        self.assertIn("未知作业类型", cm.exception.message)
        self.assertIn("可用", cm.exception.message)

    def test_子模块登记了doc_strategy(self) -> None:
        """⭐ `gui/api_doc.py` 的登记必须真的生效。

        拆模块之后，`doc_strategy` 的登记动作跑在 `gui.api_doc` 里，
        由 `api.py` 末尾那行 import 触发。**那行 import 一旦被删掉，
        整条路径会静默地不再存在**（KINDS 里少一个 key，
        validate_spec 开始拒绝 doc_strategy，而报错看起来像参数写错了）。
        这条测试就是钉住那行 import。
        """
        import gui.api_doc as api_doc

        self.assertIn("doc_strategy", api.KINDS)
        # 必须是**同一份**注册表，不是各持一份
        self.assertIs(api_doc.KINDS, api.KINDS)
        self.assertIs(api.KINDS["doc_strategy"], api_doc.run_doc_strategy)

    def test_文档报告载荷可读(self) -> None:
        """全局页的数据源：三份报告都要能被列出来（缺文件也要如实说）。"""
        d = api.doc_reports_payload()
        titles = [r["title"] for r in d["reports"]]
        self.assertEqual(len(titles), 3)
        self.assertTrue(any(t.startswith("A14") for t in titles))
        self.assertTrue(any(t.startswith("A15") for t in titles))
        self.assertTrue(any(t.startswith("A16") for t in titles))
        for r in d["reports"]:
            self.assertIn("report_exists", r)
            self.assertIn("json_exists", r)

    def test_打包清单覆盖了界面会读的磁盘文件(self) -> None:
        """⭐ 漏打一个文件的后果**不是报错**，而是那个功能在**安装包里静默失效**。

        实测踩到（2026-09-28）：第一版 spec 只带了 `data/` 与 `gui/static/`，
        `out/a14|a15|a16` 三份结果 JSON 没进包 ⇒
        「文档验证」页在安装包里永远显示"产物缺失"，
        而**开发机上一切正常**（那里 `out/` 就在手边）——
        所以只在本机跑是发现不了的，只有"打包后再实测"或"对照清单"能发现。

        这条测试对照 `api_doc.DOC_REPORTS` 检查 `bundle_data` 的覆盖面：
        两边各写一份是故意的，**脱节必须被报出来**。
        """
        from gui import bundle_data

        for title, jp, mp in api_doc.DOC_REPORTS:
            for p in (jp, mp):
                rel = p.relative_to(api_doc.ROOT).as_posix()
                self.assertTrue(
                    bundle_data.covers(rel),
                    f"「{title}」要读 {rel}，但打包清单没覆盖它 —— "
                    f"该页在安装包里会静默失效",
                )

    def test_打包清单是按相对位置摆放的(self) -> None:
        """⚠️ 数据必须按**原相对位置**放进运行时目录，否则
        `Path(__file__).parent.parent` 那套定位会全错
        （那是"路径其实不用打补丁"的前提，见 packaging/README）。"""
        from gui import bundle_data

        for d in bundle_data.BUNDLE_DIRS:
            self.assertFalse(d.startswith("/"), d)
            self.assertNotIn("\\", d, f"{d} 必须用正斜杠（跨平台）")
        for f in bundle_data.BUNDLE_FILES:
            self.assertIn("/", f, f"{f} 要带上它所在的子目录")
        self.assertTrue(bundle_data.covers("data/market.sqlite"))
        self.assertTrue(bundle_data.covers("gui/static/index.html"))
        self.assertTrue(bundle_data.covers("out/a16/doc_strategies.json"))
        # 运行产物**不在**清单里（Agent 页在全新安装里为空是正常的）
        self.assertFalse(bundle_data.covers("out/a5/dec_v2_BTC.jsonl"))


class TestRoutes(unittest.TestCase):
    """⭐ 「URL ↔ 处理函数」必须只有**一处**事实源 —— 钉住路由表的登记。

    原来 URL 映射写在 `server.py::_api_get/_api_post` 的一串
    `if path == ...` / `if path.startswith(...)` 里，而处理函数住在
    `api.py` / `api_doc.py` / `agent_api.py`。于是加一个端点要改两个文件，
    而**只改一处不会报错**——只会得到一个"函数写好了但谁也调不到"的 404。
    这与 `KINDS` 的两份白名单是**同一类**失效（本项目被坑过半小时的那个），
    所以修法也一样：一处登记、其余派生。

    这个类的重点不是"URL 拼对了"，而是**"登记确实生效了"** ——
    只测"文件能被导入"是测不到这类问题的（拆模块之后，模块级副作用
    由谁触发才是坏点）。
    """

    def test_三个业务模块都贡献了路由(self) -> None:
        """⚠️ 从各模块自己的函数名反推，**不在这里重抄一遍 URL 清单** ——
        否则测试自己就变成了第二份事实源，而两份迟早分叉。"""
        names = {r.name for r in routes.ROUTES}
        for fn in (api.api_meta, api.api_run, api.api_job, api.api_job_series,
                   api.api_jobs, api.api_job_cancel, api.api_real, api.api_export,
                   api_doc.api_doc_reports, agent_api.api_agent_runs,
                   agent_api.api_agent_summary, agent_api.api_agent_decisions,
                   agent_api.api_agent_decision, agent_api.api_agent_run):
            self.assertIn(fn.__name__, names, f"{fn.__name__} 没被登记进路由表")

    def test_每条路由都能被自己的模式匹配到(self) -> None:
        """从 pattern **推导**出一个具体路径，再要求它命中同一条路由。

        这样就不用手工维护"示例路径"清单（那又是一份会过时的副本）。
        """
        for r in routes.ROUTES:
            path = re.sub(r"\(\?P<(\w+)>[^)]*\)", lambda m: m.group(1), r.pattern)
            hit = routes.try_match(r.method, path)
            self.assertIsNotNone(hit, f"{r} 匹配不到自己推导出的 {path}")
            self.assertIs(hit[0], r, f"{path} 命中了 {hit[0]}，不是 {r}")

    def test_整条路径才算命中(self) -> None:
        """⚠️ 用 `match`（前缀）而不是 `fullmatch` 会让 `/api/meta/x` 也返回 200。
        那种错很难被注意到，却是后面加子路径时踩雷的起点。"""
        self.assertIsNone(routes.try_match("GET", "/api/meta/x"))
        self.assertIsNone(routes.try_match("GET", "/api/jobs/extra"))

    def test_方法不匹配就找不到(self) -> None:
        self.assertIsNone(routes.try_match("POST", "/api/meta"))
        self.assertIsNone(routes.try_match("GET", "/api/run"))

    def test_重复登记会被拒(self) -> None:
        """表里出现两条同模式的规则 = 后一条永远轮不到，**静默失效**。"""
        with self.assertRaises(RuntimeError):
            routes.route("GET", "/api/meta")(lambda ctx: {})
        # 同一个函数被登记两次（名字撞了）也要拒
        with self.assertRaises(RuntimeError):
            routes.route("GET", "/api/brand-new-endpoint")(api.api_meta)

    def test_agent那组是只读的(self) -> None:
        """⭐ `KINDS` 是"作业类型"表，agent 接口按设计**只读、不起作业**。
        它们共用的是"URL 在哪定义"这件事，不是同一张表。
        所以：agent 前缀下**一条 POST 都不该有**。"""
        for r in routes.ROUTES:
            if r.pattern.startswith("/api/agent"):
                self.assertEqual(r.method, "GET", f"{r} 是写入口，破坏了只读约定")

    def test_权限翻译只声明了KeyError(self) -> None:
        """按 id 取运行失败 → 404（用户的输入问题）。
        ⚠️ 但**不能**把"所有异常"都翻译成 4xx —— 那会把服务端 bug
        伪装成"你要的东西不存在"，排查方向直接被带偏。"""
        for r in routes.ROUTES:
            if "/api/agent/run/" in r.pattern:
                self.assertIn(KeyError, r.errors, f"{r} 没声明 KeyError→404")
                self.assertEqual(r.errors[KeyError], 404)
                self.assertEqual(len(r.errors), 1, f"{r} 声明了多余的异常翻译")


class TestApiDocExtraction(unittest.TestCase):
    """⭐ 搬模块时的**静默改坏**是真实风险，必须用测试钉住。

    把 `run_doc_strategy` 那一节从 `api.py` 搬进 `api_doc.py` 时，
    我第一版**顺手重写**了 `_doc_trade_stats` 与 `_doc_load`——
    结果是：键名从 `total_realized` 变成了 `total`、成本口径从
    「读 fill 自己的 fee」变成了「另算 12bp」、`_doc_load` 丢了
    `finally: st.close()`。**三处都不会报错**（JSON 照样出得来），
    但数字与 A16 对不上了。

    这个类的存在意义：把"这段代码的**可观察契约**"写死，
    于是"搬动"必须是纯搬动，任何改写都会红。
    """

    def test_交易统计的键名与成本口径(self) -> None:
        from gui import api_doc

        class _Fill:
            def __init__(self, side, qty, price, fee=0.0, slippage_cost=0.0):
                self.side, self.qty, self.price = side, qty, price
                self.fee, self.slippage_cost = fee, slippage_cost

        # 一开一平，赚 10 元毛利、花 1 元手续费
        fills = [_Fill("buy", 1.0, 100.0, fee=0.5, slippage_cost=0.25),
                 _Fill("sell", 1.0, 110.0, fee=0.5, slippage_cost=0.25)]
        ts = api_doc._doc_trade_stats(fills)
        # 键名是契约（A16 报告与前端都按这些 key 取数）
        for k in ("n_trades", "total_realized", "E_per_trade", "win_rate",
                  "skew", "fees", "slippage", "net_after_costs"):
            self.assertIn(k, ts, f"缺少键 {k}（搬模块时被改名了？）")
        self.assertEqual(ts["n_trades"], 1)
        # 成本是**从 fill 里读**的：0.5+0.5 手续费、0.25+0.25 滑点
        self.assertAlmostEqual(ts["fees"], 1.0)
        self.assertAlmostEqual(ts["slippage"], 0.5)
        # 毛利 10 − 平仓那笔的手续费 0.5（原实现的口径）
        self.assertAlmostEqual(ts["total_realized"], 9.5)
        self.assertAlmostEqual(ts["net_after_costs"], 9.5 - 1.0 - 0.5)

    def test_载入K线后连接必须关掉(self) -> None:
        """⭐ 搬模块时最容易丢的就是 `finally`。

        `_doc_load` 必须保证异常路径也 `close()`——否则连跑几次失败作业
        会攒下未关闭的 sqlite 连接。这里用一个假的 marketdb 注入异常，
        断言 close 被调用过。
        """
        import sys
        import types
        from gui import api_doc

        closed = {"n": 0}

        class _FakeStore:
            def load_candles(self, *a, **k):
                raise RuntimeError("模拟读库失败")

            def close(self):
                closed["n"] += 1

        fake = types.ModuleType("tw.marketdb")
        fake.MarketStore = _FakeStore
        saved = sys.modules.get("tw.marketdb")
        sys.modules["tw.marketdb"] = fake
        try:
            with self.assertRaises(RuntimeError):
                api_doc._doc_load("BTCUSDT")
        finally:
            if saved is not None:
                sys.modules["tw.marketdb"] = saved
            else:
                sys.modules.pop("tw.marketdb", None)
        self.assertEqual(closed["n"], 1, "异常路径没有 close() —— 连接泄漏")


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

    def test_路径多一段不会被前缀吃掉(self) -> None:
        """路由表用 fullmatch：`/api/meta/extra` 不该命中 `/api/meta`。"""
        s, _ = self._get("/api/meta/extra", {"X-TW-Token": self.server.token})
        self.assertEqual(s, 404)

    def test_POST打不到只读的GET接口(self) -> None:
        req = urllib.request.Request(
            self.server.base_url + "/api/meta", data=b"{}",
            headers={"X-TW-Token": self.server.token})
        try:
            self.op.open(req, timeout=20)
            self.fail("GET 接口竟然被 POST 命中了")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 404)

    def test_agent的未知运行是404而不是500(self) -> None:
        """取不存在的 run 是**用户的输入问题**，不该回 500 的 traceback。

        ⚠️ 判据不能只看状态码。改动前那段代码写的是
        `except KeyError: 404`，看起来也对；但它把**所有** KeyError 都
        当成 404 —— 内部一个笔误抛的 KeyError 也会被翻译成
        "你要的东西不存在"。所以现在只翻路由**自己声明**的异常类型。
        """
        s, b = self._get("/api/agent/run/definitely-not-a-run/summary",
                         {"X-TW-Token": self.server.token})
        self.assertEqual(s, 404)
        self.assertIn("没有这个运行", json.loads(b)["error"])

    def test_agent的非法序号是400(self) -> None:
        s, _ = self._get("/api/agent/run/x/decision/abc",
                         {"X-TW-Token": self.server.token})
        self.assertEqual(s, 400)

    def test_全局页与agent接口都能真的调到(self) -> None:
        """⭐ 测的是**接线**，不是处理函数。

        这三条端点的处理函数都有自己的单测（`TestApiValidate` /
        `TestApiPayloads`），但它们**仍然可能一个 URL 都挂不上**——
        例如漏写 `@route` 装饰器、或路径写错。那种情况下
        "直接调函数"的测试全绿，而前端只会看到 404 / 永远"载入中…"。
        ⇒ 所以这里从 HTTP 面再打一遍。
        """
        s, b = self._get("/api/doc/reports", {"X-TW-Token": self.server.token})
        self.assertEqual(s, 200)
        self.assertEqual(len(json.loads(b)["reports"]), 3)

        s, b = self._get("/api/agent/runs", {"X-TW-Token": self.server.token})
        self.assertEqual(s, 200)
        self.assertIn("runs", json.loads(b))

        s, b = self._get("/api/real/BTCUSDT_1h", {"X-TW-Token": self.server.token})
        self.assertEqual(s, 200)
        self.assertEqual(json.loads(b)["symbol"], "BTCUSDT_1h")

        # 未知数据集仍要在**路由层**给出 404（real_payload 自己抛的 ApiError）
        s, _ = self._get("/api/real/no-such-symbol", {"X-TW-Token": self.server.token})
        self.assertEqual(s, 404)

    def test_导出的响应头带文件名(self) -> None:
        """⚠️ 少了 `Content-Disposition`，浏览器会在标签页里**直接打开** CSV，
        用户拿不到文件 —— 而状态码与内容都是对的，**只有响应头能证明它**。
        同理 BOM 只能在字节层面看（少了它 Excel 打开中文乱码）。"""
        body = json.dumps({"kind": "strategy", "scenario": "normal", "seed": 3,
                           "ticks": 60, "warmup": 200, "strategy": "noop"})
        req = urllib.request.Request(
            self.server.base_url + "/api/run", data=body.encode(),
            headers={"X-TW-Token": self.server.token,
                     "Content-Type": "application/json"})
        jid = json.loads(self.op.open(req, timeout=20).read())["job_id"]
        for _ in range(200):
            time.sleep(0.05)
            _, b = self._get("/api/job/" + jid, {"X-TW-Token": self.server.token})
            if json.loads(b)["state"] in ("done", "error"):
                break
        req = urllib.request.Request(f"{self.server.base_url}/api/export/{jid}.csv",
                                     headers={"X-TW-Token": self.server.token})
        r = self.op.open(req, timeout=20)
        self.assertEqual(r.status, 200)
        disp = r.headers.get("Content-Disposition", "")
        self.assertIn("attachment", disp)
        self.assertIn(f"tw-{jid}.csv", disp)
        self.assertTrue(r.read().startswith(b"\xef\xbb\xbf"), "缺 BOM，Excel 打开会乱码")

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
