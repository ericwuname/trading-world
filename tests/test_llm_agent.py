"""A3 测试：LLM 通道 / prompt 模板 / 解析 / 决策回路。

**全部离线**——没有一个测试会发网络请求。这是刻意的：
"能离线把全部分支测穷"正是 ``tw/llm.py`` 把 Transport 抽出来的原因。

每个测试都问了一句"它能不能失败"（本项目纪律）。
写"永远通过"的断言是这里最容易犯的错——所以重点覆盖的是
**静默失效**的那几类：缺 key 静默弃权、解析层"帮模型改数"、
回放未命中静默返回空、可见状态取到未来的根。
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tw.agent import (
    AgentConfig,
    TradingAgent,
    run_session,
    visible_from_series,
)
from tw.decision_log import DecisionLog, build_visible_state
from tw.llm import (
    HTTPClient,
    LLMConfig,
    LLMResponse,
    Recorder,
    ReplayClient,
    ScriptedClient,
    TransportError,
    prompt_hash,
)
from tw.parse import (
    consistency,
    decision_signature,
    extract_json,
    majority_sample,
    parse_decision,
)
from tw.prompts import (
    DEFAULT_TEMPLATE,
    OUTPUT_KEYS,
    TEMPLATES,
    build_messages,
    template_fingerprint,
)
from tw.risk import RiskLimits

AGNES = LLMConfig(provider="agnes")


# ======================================================================
# 辅助
# ======================================================================
class _Series:
    """最小的 Series 替身。

    ⚠️ **必须带上 ``high`` / ``low``**——真实的 ``marketdb.Series``
    永远有这两个字段。第一版替身只有 ``close``，于是"M82（mid 误用
    高低均值）"在多数测试里表现为 ``AttributeError`` 崩溃而不是
    断言失败：也能变红，但**信号是"崩了"而不是"错了"**，
    读日志的人得自己去分辨。替身要像真的，测试失败才说明问题本身。
    """

    def __init__(self, closes, *, high=None, low=None):
        self.close = np.asarray(closes, dtype=float)
        # 缺省让 high=low=close：这是"无影线"的退化形态，
        # 但至少**字段存在**，与真实 Series 的形状一致。
        self.high = self.close if high is None else np.asarray(high, dtype=float)
        self.low = self.close if low is None else np.asarray(low, dtype=float)
        self.timestamp = np.arange(len(self.close), dtype=np.int64) * 3_600_000


def _agent(texts, *, n_samples=1, temperature=0.0, limits=None):
    client = ScriptedClient(
        config=AGNES,
        responses=[LLMResponse(ok=True, text=t) for t in texts],
        on_exhausted="hold",
    )
    cfg = AgentConfig(n_samples=n_samples, temperature=temperature)
    return TradingAgent(client=client, config=cfg, limits=limits or RiskLimits())


def _vis(price=102.44, equity=100_000.0):
    return build_visible_state(mid=price, cash=equity, equity=equity,
                               recent_closes=[price] * 3)


BUY_OK = ('{"action":"buy","size":0.5,"order_type":"limit","limit_price":102.44,'
          '"take_profit":104.0,"stop_loss":100.0,"confidence":0.7,'
          '"reason":"趋势向上"}')


# ======================================================================
# prompt_hash
# ======================================================================
class TestPromptHash(unittest.TestCase):
    def test_同输入同哈希(self):
        m = [{"role": "user", "content": "hi"}]
        self.assertEqual(prompt_hash(m, model="m", temperature=0.2),
                         prompt_hash(m, model="m", temperature=0.2))

    def test_对温度敏感(self):
        """⭐ 温度不同 = 两个不同的实验。哈希一样会让 A/B 混成一条。"""
        m = [{"role": "user", "content": "hi"}]
        self.assertNotEqual(prompt_hash(m, model="m", temperature=0.2),
                            prompt_hash(m, model="m", temperature=1.0))

    def test_对模型敏感(self):
        m = [{"role": "user", "content": "hi"}]
        self.assertNotEqual(prompt_hash(m, model="a", temperature=0.2),
                            prompt_hash(m, model="b", temperature=0.2))

    def test_对正文敏感(self):
        a = [{"role": "user", "content": "hi"}]
        b = [{"role": "user", "content": "hi "}]
        self.assertNotEqual(prompt_hash(a, model="m", temperature=0.2),
                            prompt_hash(b, model="m", temperature=0.2))

    def test_键顺序不影响(self):
        a = [{"role": "user", "content": "x", "name": "n"}]
        b = [{"name": "n", "content": "x", "role": "user"}]
        self.assertEqual(prompt_hash(a, model="m", temperature=0.2),
                         prompt_hash(b, model="m", temperature=0.2))


# ======================================================================
# LLMConfig
# ======================================================================
class TestLLMConfig(unittest.TestCase):
    def test_从注册表补全(self):
        c = LLMConfig(provider="agnes")
        self.assertEqual(c.model, "agnes-2.5-flash")
        self.assertEqual(c.api_key_env, "AGNES_KEY")
        self.assertTrue(c.url.endswith("/chat/completions"))

    def test_未知provider必须给base_url(self):
        with self.assertRaises(ValueError):
            LLMConfig(provider="不存在的通道")

    def test_未知provider给了base_url就行(self):
        c = LLMConfig(provider="custom", base_url="http://x/v1", model="m")
        self.assertEqual(c.url, "http://x/v1/chat/completions")

    def test_重试数必须至少一次(self):
        with self.assertRaises(ValueError):
            LLMConfig(provider="agnes", retries=0)

    def test_描述里不含密钥值(self):
        """⭐ 留在留痕里的配置描述**只能有变量名**。"""
        d = LLMConfig(provider="agnes").describe()
        self.assertEqual(d["api_key_env"], "AGNES_KEY")
        self.assertNotIn("key", {k for k in d if k != "api_key_env"})
        blob = json.dumps(d)
        self.assertNotIn("cpk-", blob)
        self.assertNotIn("sk-", blob)


# ======================================================================
# HTTPClient —— 假 transport，不发网络
# ======================================================================
class _FakeTransport:
    """记录调用、可编程返回。"""

    def __init__(self, results):
        # results: list of ("ok", dict) | ("err", Exception)
        self.results = list(results)
        self.calls = 0

    def __call__(self, url, body, headers, timeout):
        self.calls += 1
        kind, payload = self.results.pop(0)
        if kind == "err":
            raise payload
        return payload


def _resp_obj(text):
    return {"choices": [{"message": {"content": text}}],
            "usage": {"total_tokens": 42}}


class TestHTTPClient(unittest.TestCase):
    def test_缺key必须抛错而不是静默弃权(self):
        """⭐⭐ 本文件最重要的一条。

        一个"没 key 就返回 hold"的实现，跑起来**看起来完全正常**：
        每条决策都合规、都留痕、都弃权。它把"系统是坏的"
        伪装成"Agent 很保守"。所以缺 key 必须在**构造期**就炸。

        ⚠️ 必须把 secrets 目录也指向空地方——否则这条测试会在
        "本机碰巧有密钥文件"时**静默失效**（它一度就是这样：
        加完 secrets 文件支持后这条测试直接变绿了，什么都没验证到）。
        依赖机器状态的测试，在不同机器上结论不同 = 等于没有测试。
        """
        from unittest.mock import patch

        old = os.environ.pop("AGNES_KEY", None)
        try:
            with patch("tw.llm.SECRETS_DIR", Path(tempfile.gettempdir())
                       / "绝对不存在的secrets目录"):
                with self.assertRaises(RuntimeError) as cm:
                    HTTPClient(config=AGNES)
                self.assertIn("AGNES_KEY", str(cm.exception))
                self.assertIn("secrets", str(cm.exception))
        finally:
            if old is not None:
                os.environ["AGNES_KEY"] = old

    def test_密钥可从仓库外文件读(self):
        """secrets 文件支持是**为安全加的**：命令行上的 key 会进
        shell 历史与进程表。这里验证它真的能读到，且来源被标出来。"""
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "agnes.json"
            p.write_text(json.dumps({"api_key": "cpk-from-file"}),
                         encoding="utf-8")
            old = os.environ.pop("AGNES_KEY", None)
            try:
                with patch("tw.llm.SECRETS_DIR", Path(d)):
                    c = HTTPClient(config=AGNES, transport=_FakeTransport([]))
                self.assertEqual(c.api_key, "cpk-from-file")
                self.assertEqual(c.key_source, "file")
            finally:
                if old is not None:
                    os.environ["AGNES_KEY"] = old

    def test_环境变量优先于文件(self):
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "agnes.json").write_text(
                json.dumps({"api_key": "cpk-file"}), encoding="utf-8")
            old = os.environ.get("AGNES_KEY")
            os.environ["AGNES_KEY"] = "cpk-env"
            try:
                with patch("tw.llm.SECRETS_DIR", Path(d)):
                    c = HTTPClient(config=AGNES, transport=_FakeTransport([]))
                self.assertEqual(c.api_key, "cpk-env")
                self.assertEqual(c.key_source, "env")
            finally:
                if old is None:
                    os.environ.pop("AGNES_KEY", None)
                else:
                    os.environ["AGNES_KEY"] = old

    def test_异常信息里不含密钥(self):
        """⭐⭐ 异常会被打日志、被贴进 issue、被发给外部 AI 复核。
        里面出现密钥就等于泄漏。"""
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "agnes.json").write_text(
                json.dumps({"api_key": "cpk-SHOULD-NOT-APPEAR"}),
                encoding="utf-8")
            old = os.environ.pop("AGNES_KEY", None)
            try:
                with patch("tw.llm.SECRETS_DIR", Path(tempfile.gettempdir())
                           / "绝对不存在的secrets目录"):
                    with self.assertRaises(RuntimeError) as cm:
                        HTTPClient(config=AGNES)
                self.assertNotIn("cpk-SHOULD-NOT-APPEAR", str(cm.exception))
            finally:
                if old is not None:
                    os.environ["AGNES_KEY"] = old

    def test_无key需求的通道不检查(self):
        """ollama 本地不需要 key，不该被这条守卫误杀。"""
        c = LLMConfig(provider="ollama")
        self.assertEqual(c.api_key_env, "")
        HTTPClient(config=c)   # 不应抛

    def test_成功路径(self):
        t = _FakeTransport([("ok", _resp_obj("hello"))])
        c = HTTPClient(config=AGNES, transport=t, api_key="k")
        r = c.chat([{"role": "user", "content": "x"}])
        self.assertTrue(r.ok)
        self.assertEqual(r.text, "hello")
        self.assertEqual(r.attempts, 1)
        self.assertEqual(r.usage["total_tokens"], 42)
        self.assertTrue(r.prompt_hash)

    def test_重试后成功(self):
        t = _FakeTransport([
            ("err", TransportError("HTTP 500: boom")),
            ("ok", _resp_obj("ok2")),
        ])
        c = HTTPClient(config=LLMConfig(provider="agnes", retries=3),
                       transport=t, api_key="k")
        r = c.chat([{"role": "user", "content": "x"}])
        self.assertTrue(r.ok)
        self.assertEqual(r.text, "ok2")
        self.assertEqual(r.attempts, 2)
        self.assertEqual(t.calls, 2)

    def test_全部失败返回失败而不是抛(self):
        """失败要能被记进留痕，所以返回 ``ok=False`` 而不是抛。"""
        t = _FakeTransport([("err", TransportError("boom"))] * 2)
        c = HTTPClient(config=LLMConfig(provider="agnes", retries=2),
                       transport=t, api_key="k")
        r = c.chat([{"role": "user", "content": "x"}])
        self.assertFalse(r.ok)
        self.assertIn("boom", r.error)
        self.assertEqual(r.attempts, 2)

    def test_响应缺choices要报错不要当空串(self):
        """⭐ 把结构错误当成"模型答了个空"会让两类问题混在一起。"""
        t = _FakeTransport([("ok", {"error": {"message": "bad model"}})] * 3)
        c = HTTPClient(config=AGNES, transport=t, api_key="k")
        r = c.chat([{"role": "user", "content": "x"}])
        self.assertFalse(r.ok)
        self.assertIn("choices", r.error)

    def test_失败响应也带prompt哈希(self):
        """回放要靠它定位"哪一次调用失败了"。"""
        t = _FakeTransport([("err", TransportError("x"))])
        c = HTTPClient(config=LLMConfig(provider="agnes", retries=1),
                       transport=t, api_key="k")
        r = c.chat([{"role": "user", "content": "x"}])
        self.assertEqual(
            r.prompt_hash,
            prompt_hash([{"role": "user", "content": "x"}],
                        model=AGNES.model, temperature=AGNES.temperature,
                        max_tokens=AGNES.max_tokens),
        )


# ======================================================================
# ScriptedClient
# ======================================================================
class TestScriptedClient(unittest.TestCase):
    def test_用尽必须报错(self):
        """静默地一直返回最后一条，会让"测试只覆盖了前 N 次调用"
        这个事实消失。"""
        c = ScriptedClient.from_json_texts(['{"action":"hold"}'])
        c.chat([{"role": "user", "content": "1"}])
        with self.assertRaises(RuntimeError):
            c.chat([{"role": "user", "content": "2"}])

    def test_用尽可配置(self):
        c = ScriptedClient.from_json_texts(['{"action":"hold"}'],
                                           on_exhausted="hold")
        c.chat([{"role": "user", "content": "1"}])
        r = c.chat([{"role": "user", "content": "2"}])
        self.assertIn("hold", r.text)

    def test_带prompt哈希(self):
        c = ScriptedClient.from_json_texts(['{"action":"hold"}'])
        r = c.chat([{"role": "user", "content": "x"}])
        self.assertTrue(r.prompt_hash)


# ======================================================================
# Recorder + ReplayClient
# ======================================================================
class TestRecorderAndReplay(unittest.TestCase):
    def _record(self, tmp: Path, texts):
        t = _FakeTransport([("ok", _resp_obj(x)) for x in texts])
        inner = HTTPClient(config=AGNES, transport=t, api_key="k")
        rec = Recorder(inner, tmp)
        for i, _ in enumerate(texts):
            rec.chat([{"role": "user", "content": f"p{i}"}])
        return rec

    def test_录制后能回放且内容一致(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "rec.jsonl"
            self._record(p, ["A", "B"])
            rp = ReplayClient(records_path=p)
            r1 = rp.chat([{"role": "user", "content": "p0"}])
            r2 = rp.chat([{"role": "user", "content": "p1"}])
            self.assertEqual((r1.text, r2.text), ("A", "B"))
            self.assertEqual(rp.coverage, 1.0)

    def test_录制里不含密钥(self):
        """⭐⭐ 录制文件是要提交进仓库的——密钥绝不能进去。"""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "rec.jsonl"
            secret = "cpk-SUPERSECRETVALUE"
            t = _FakeTransport([("ok", _resp_obj("A"))])
            inner = HTTPClient(config=AGNES, transport=t, api_key=secret)
            rec = Recorder(inner, p)
            rec.chat([{"role": "user", "content": "x"}])
            blob = p.read_text(encoding="utf-8")
            self.assertNotIn(secret, blob)
            self.assertNotIn("Authorization", blob)

    def test_同prompt多次调用按次序发不同响应(self):
        """⭐⭐ 真机跑出来的坑，而且是**静默**的那种。

        多采样时同一 prompt 会调用多次、每次都录一条。第一版索引写成
        ``dict[hash] = 记录``，后写的把先写的**覆盖**掉 ⇒ 回放时 3 次
        都拿最后 1 条。后果：真实的"2 hold / 1 buy"（多数派 hold）
        变成"3 buy"（多数派 buy）——**方向整个反过来**，
        而覆盖率仍然显示 100%、没有任何报错。
        """
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "r.jsonl"
            # 录：同一 prompt，两次不同响应
            t = _FakeTransport([("ok", _resp_obj("A")), ("ok", _resp_obj("B"))])
            inner = HTTPClient(config=AGNES, transport=t, api_key="k")
            rec = Recorder(inner, p)
            msgs = [{"role": "user", "content": "同一个问题"}]
            rec.chat(msgs)
            rec.chat(msgs)

            rp = ReplayClient(records_path=p)
            self.assertEqual(rp.n_prompts, 1, "同 prompt 应只算 1 个 prompt")
            got = [rp.chat(msgs).text for _ in range(2)]
            self.assertEqual(got, ["A", "B"], "必须按录制次序发，不能只发最后一条")
            self.assertEqual(rp.coverage, 1.0)

    def test_回放条数不够要报错(self):
        """⭐ 记录里只有 2 条却要 3 条采样 ⇒ 必须报错，
        静默少发一条会让"这次决策有几个样本"这个事实消失。"""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "r.jsonl"
            t = _FakeTransport([("ok", _resp_obj("A"))])
            Recorder(HTTPClient(config=AGNES, transport=t, api_key="k"),
                     p).chat([{"role": "user", "content": "q"}])
            rp = ReplayClient(records_path=p)
            rp.chat([{"role": "user", "content": "q"}])
            with self.assertRaises(KeyError) as cm:
                rp.chat([{"role": "user", "content": "q"}])
            self.assertIn("只记了 1 条", str(cm.exception))

    def test_回放未命中必须报错(self):
        """⭐ 未命中静默返回空 ⇒ 回放"成功地"跑出一堆弃权，是假绿。"""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "rec.jsonl"
            self._record(p, ["A"])
            rp = ReplayClient(records_path=p)
            with self.assertRaises(KeyError):
                rp.chat([{"role": "user", "content": "没录过的问题"}])
            self.assertEqual(rp.misses, 1)

    def test_多采样整个回合可复现(self):
        """⭐⭐ 端到端：真机录 3 次采样 → 回放 → 多数派必须一致。

        这是那个静默 bug 的**回归守卫**：它会在覆盖式索引下变红
        （回放的多数派会与真机不同）。
        """
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "r.jsonl"
            texts = ['{"action":"hold","reason":"a"}',
                     '{"action":"hold","reason":"b"}',
                     '{"action":"buy","size":0.1,"confidence":0.5,"reason":"c"}']
            t = _FakeTransport([("ok", _resp_obj(x)) for x in texts])
            rec = Recorder(HTTPClient(config=AGNES, transport=t, api_key="k"), p)
            cfg = AgentConfig(n_samples=3, temperature=0.2)
            live = TradingAgent(client=rec, config=cfg).decide(
                visible=_vis(), tick=1, run_id="R", equity=1e5)
            self.assertEqual(live.parsed["_vote"], {"hold": 2, "buy": 1})

            ag = TradingAgent(client=ReplayClient(records_path=p), config=cfg)
            rp_rec = ag.decide(visible=_vis(), tick=1, run_id="R", equity=1e5)
            self.assertEqual(rp_rec.parsed["_vote"], live.parsed["_vote"])
            self.assertEqual(rp_rec.parsed["action"], live.parsed["action"])
            self.assertEqual(rp_rec.decision_id, live.decision_id)
            self.assertEqual(rp_rec.executed, live.executed)

    def test_decision_id跨会话可在给定run_id下复现(self):
        """⭐ ``decision_id`` 含 ``run_id``，所以 run_id 若带墙钟时间，
        "回放能否核对上"就**从构造上不可能成功**。
        给定同一个 run_id，两次独立会话必须算出同一个 ID。"""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "r.jsonl"
            t = _FakeTransport([("ok", _resp_obj(BUY_OK))])
            rec = Recorder(HTTPClient(config=AGNES, transport=t, api_key="k"), p)
            a = TradingAgent(client=rec, config=AgentConfig()).decide(
                visible=_vis(), tick=5, run_id="固定ID", equity=1e5)
            b = TradingAgent(client=ReplayClient(records_path=p),
                             config=AgentConfig()).decide(
                visible=_vis(), tick=5, run_id="固定ID", equity=1e5)
            self.assertEqual(a.decision_id, b.decision_id)

    def test_非严格模式记未命中(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "rec.jsonl"
            self._record(p, ["A"])
            rp = ReplayClient(records_path=p, strict=False)
            r = rp.chat([{"role": "user", "content": "没录过"}])
            self.assertFalse(r.ok)
            self.assertEqual(r.error, "replay_miss")

    def test_温度变了就找不到(self):
        """⭐ 同一 prompt 在不同温度下是**两次不同实验**，
        回放必须区分——否则 A/B 会读到同一条记录。"""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "rec.jsonl"
            self._record(p, ["A"])          # 温度 0.2
            rp = ReplayClient(records_path=p,
                              config=LLMConfig(provider="agnes",
                                               temperature=1.0))
            with self.assertRaises(KeyError):
                rp.chat([{"role": "user", "content": "p0"}])

    def test_文件不存在时不崩(self):
        rp = ReplayClient(records_path=Path(tempfile.gettempdir()) / "没有这个文件.jsonl",
                          strict=False)
        self.assertEqual(rp.coverage, 1.0)


# ======================================================================
# 模板
# ======================================================================
class TestPrompts(unittest.TestCase):
    def test_默认版本在(self):
        self.assertIn(DEFAULT_TEMPLATE, TEMPLATES)

    def test_每个模板都有system和user(self):
        for v, t in TEMPLATES.items():
            with self.subTest(version=v):
                self.assertIn("system", t)
                self.assertIn("user", t)
                self.assertTrue(t["system"].strip())

    def test_指纹是16位且稳定(self):
        f = template_fingerprint()
        self.assertEqual(len(f), 16)
        self.assertEqual(f, template_fingerprint())

    def test_模板里有JSON契约(self):
        """输出 schema 必须写在 system 里，否则模型无从遵守。"""
        sysmsg = TEMPLATES[DEFAULT_TEMPLATE]["system"]
        for k in OUTPUT_KEYS:
            self.assertIn(k, sysmsg, f"system 提示里没有告诉模型字段 {k}")

    def test_系统提示明确说不给未来数据(self):
        """⭐ "你能看到什么"必须显式声明——这是防泄漏的**声明**面。"""
        sysmsg = TEMPLATES[DEFAULT_TEMPLATE]["system"]
        self.assertIn("看不到未来", sysmsg)
        self.assertIn("唯一的信息来源", sysmsg)

    def test_系统提示把弃权说成合格决策(self):
        """否则模型会把"必须给动作"理解成"必须下单"。"""
        self.assertIn("弃权", TEMPLATES[DEFAULT_TEMPLATE]["system"])

    def test_未知模板报错(self):
        with self.assertRaises(KeyError):
            build_messages({"mid": 1.0}, inst_id="X", template="v99")

    def test_消息结构是system加user(self):
        m = build_messages({"mid": 1.0}, inst_id="X")
        self.assertEqual([x["role"] for x in m], ["system", "user"])

    def test_可见字段真的渲染进user正文(self):
        """⭐ 不给模型看，却指望它用——这是"假能力"的来源。"""
        m = build_messages(
            {"mid": 102.44, "best_bid": 102.40, "best_ask": 102.48,
             "spread_bp": 8.1, "mark": 102.5, "funding_rate": 0.0001,
             "cash": 1000.0, "equity": 999.0, "margin_ratio": 0.42,
             "recent_closes": [101.0, 102.0]},
            inst_id="BTC-USDT-SWAP", tick=7)
        body = m[1]["content"]
        for token in ("102.44", "102.4", "102.48", "8.1", "102.5",
                      "0.0001", "1,000", "999", "0.42", "tick 7"):
            self.assertIn(token, body, f"user 正文里缺 {token}")

    def test_金额不会因为整数格式化被剥掉尾零(self):
        """⭐ 回归：``_fmt`` 曾把 ``20,000`` 剥成 ``20,``
        （rstrip 用在没有小数点的情况下，尾零属于整数部分）。"""
        m = build_messages({"mid": 1.0}, inst_id="X")
        self.assertIn("20,000", m[1]["content"])
        self.assertNotIn("20, ", m[1]["content"])

    def test_缺失字段显示为无而不是None(self):
        m = build_messages({"mid": 1.0}, inst_id="X")
        self.assertNotIn("None", m[1]["content"])
        self.assertIn("（无）", m[1]["content"])

    def test_版本指纹各不相同(self):
        """⭐ 两个版本能有不同指纹，才说明"改模板加版本"这套机制
        真的在起作用（否则文件变了但指纹一样 = 版本号是装饰）。"""
        self.assertNotEqual(template_fingerprint("v1"), template_fingerprint("v2"))


class TestPromptV2(unittest.TestCase):
    """v2 是**真机跑出问题之后**才加的版本，不是凭感觉改的。

    触发原因：v1 下模型的弃权理由是「快照无买卖盘和资金费率信号」——
    它把注意力花在了数据源本来就没有的字段上。见 ``prompts.py`` 的注释。
    """

    def test_v2把不可得字段折叠成一行(self):
        body = build_messages({"mid": 1.0}, inst_id="X",
                              template="v2")[1]["content"]
        self.assertIn("不提供", body)
        for zh in ("资金费率", "买一价"):
            self.assertIn(zh, body)
        # 折叠成一行 ⇒ 不再逐个占版面（v1 是每个字段一行）
        self.assertNotIn("买一 / 卖一：", body)

    def test_v2告诉模型能用收盘价推断(self):
        """⭐ 只列"缺什么"会让模型把弃权当默认；必须同时说"你有什么"。"""
        body = build_messages({"mid": 1.0, "recent_closes": [1.0, 2.0]},
                              inst_id="X", template="v2")[1]["content"]
        self.assertIn("你能从上面推出什么", body)
        self.assertIn("方向", body)

    def test_v2即使字段齐全也有明确措辞(self):
        """留空白会让模型分不清"没有"与"你没写清"。"""
        body = build_messages(
            {"mid": 1.0, "spread_bp": 1.0, "best_bid": 1.0, "best_ask": 1.0,
             "funding_rate": 0.0001},
            inst_id="X", template="v2")[1]["content"]
        self.assertIn("都有", body)

    def test_v2不含未替换占位符(self):
        """⭐ 少传一个 format 变量会留下 ``{xxx}``，
        而模型会把它当成一个要照抄的字段名——静默的严重错误。"""
        for t in TEMPLATES:
            with self.subTest(template=t):
                body = build_messages({"mid": 1.0}, inst_id="X",
                                      template=t)[1]["content"]
                self.assertNotIn("{", body)
                self.assertNotIn("}", body)

    def test_v2不改动v1(self):
        """⭐ 原版本必须原样保留——v1 已经跑出过真实记录，
        原地改会让那些记录的 decision_id 对应到另一套提示词。"""
        self.assertIn("买一 / 卖一：", TEMPLATES["v1"]["user"])
        self.assertNotIn("你能从上面推出什么", TEMPLATES["v1"]["user"])


# ======================================================================
# 解析
# ======================================================================
class TestExtractJson(unittest.TestCase):
    def test_整段(self):
        obj, how, err = extract_json('{"a":1}')
        self.assertEqual(obj, {"a": 1})
        self.assertEqual(how, "whole")
        self.assertEqual(err, "")

    def test_代码块(self):
        obj, how, _ = extract_json('前言\n```json\n{"a":1}\n```\n后记')
        self.assertEqual(obj, {"a": 1})
        self.assertEqual(how, "fence")

    def test_花括号(self):
        obj, how, _ = extract_json('分析：{"a":1} 完毕')
        self.assertEqual(obj, {"a": 1})
        self.assertEqual(how, "braces")

    def test_取最后一个右括号(self):
        """⭐ 用第一个 ``}`` 会截断嵌套对象。"""
        text = '{"a":{"b":1}}'
        obj, _, _ = extract_json(text)
        self.assertEqual(obj, {"a": {"b": 1}})

    def test_尾随逗号(self):
        obj, how, _ = extract_json('{"a":1,}')
        self.assertEqual(obj, {"a": 1})
        self.assertIn("loose", how)

    def test_单引号(self):
        obj, _, _ = extract_json("{'a':1}")
        self.assertEqual(obj, {"a": 1})

    def test_空文本(self):
        obj, _, err = extract_json("")
        self.assertIsNone(obj)
        self.assertIn("空", err)

    def test_完全不是JSON(self):
        obj, _, err = extract_json("我觉得应该买入")
        self.assertIsNone(obj)
        self.assertIn("找不到", err)

    def test_顶层是数组也算失败(self):
        r = parse_decision("[1,2,3]")
        self.assertFalse(r.ok)
        self.assertIn("不是 JSON 对象", r.error)


class TestParseDecision(unittest.TestCase):
    def test_标准输出(self):
        r = parse_decision(BUY_OK)
        self.assertTrue(r.ok)
        self.assertEqual(r.parsed["action"], "buy")
        self.assertEqual(r.parsed["sz"], 0.5)
        self.assertEqual(r.parsed["px"], 102.44)
        self.assertEqual(r.parsed["tp"], 104.0)
        self.assertEqual(r.parsed["sl"], 100.0)
        self.assertEqual(r.parsed["ordType"], "limit")

    def test_别名映射(self):
        r = parse_decision('{"action":"long","qty":2,"price":10,'
                           '"stop_loss":9,"take_profit":12,"conf":0.5,'
                           '"reasoning":"x"}')
        self.assertTrue(r.ok)
        self.assertEqual(r.parsed["action"], "buy")
        self.assertEqual(r.parsed["sz"], 2.0)
        self.assertEqual(r.parsed["px"], 10.0)
        self.assertEqual(r.parsed["sl"], 9.0)
        self.assertEqual(r.parsed["tp"], 12.0)
        self.assertEqual(r.parsed["confidence"], 0.5)
        self.assertEqual(r.parsed["reason"], "x")

    def test_一百倍错误必须原样传下去(self):
        """⭐⭐ 本层最反直觉的一条。**不许"帮它改成 103.5"**。

        因为"帮它猜"会同时把"模型在数值上不可靠"这个事实抹掉——
        而那正是最需要被量化的东西（A4 要报的指标）。
        把关在风控层，不在这里。
        """
        r = parse_decision('{"action":"buy","size":0.1,"take_profit":10350,'
                           '"stop_loss":10180,"confidence":0.3,"reason":"x"}')
        self.assertTrue(r.ok)
        self.assertEqual(r.parsed["tp"], 10350.0)
        self.assertEqual(r.parsed["sl"], 10180.0)

    def test_数字字符串可解析(self):
        r = parse_decision('{"action":"buy","size":"0.5",'
                           '"take_profit":"103.5 USDT","stop_loss":"1,000",'
                           '"confidence":"0.8","reason":"x"}')
        self.assertEqual(r.parsed["sz"], 0.5)
        self.assertEqual(r.parsed["tp"], 103.5)
        self.assertLess(abs(r.parsed["sl"] - 1000.0), 1e-9)  # 千分位
        self.assertEqual(r.parsed["confidence"], 0.8)

    def test_无法识别的动作不猜(self):
        """⭐ ``close`` 的方向取决于持仓，而解析层不知道持仓——
        硬猜会让"平多"在空仓时变成开空。宁可报错。"""
        r = parse_decision('{"action":"close","size":1,"reason":"x"}')
        self.assertFalse(r.ok)
        self.assertIn("close", r.error)

    def test_缺action算失败(self):
        r = parse_decision('{"size":1,"reason":"x"}')
        self.assertFalse(r.ok)
        self.assertIn("action", r.error)

    def test_未知字段不丢弃(self):
        """⭐ 丢弃会让"模型开始输出新字段"这件事完全不可见。"""
        r = parse_decision('{"action":"hold","reason":"x","新字段":1}')
        self.assertTrue(r.ok)
        self.assertIn("新字段", r.parsed["_unknown"])

    def test_同义字段冲突要可见(self):
        r = parse_decision('{"action":"buy","size":1,"qty":2,"reason":"x"}')
        self.assertTrue(r.ok)
        self.assertTrue(any("冲突" in u for u in r.parsed["_unknown"]))

    def test_置信度超界被夹住且留痕(self):
        r = parse_decision('{"action":"buy","size":1,"confidence":1.5,"reason":"x"}')
        self.assertEqual(r.parsed["confidence"], 1.0)
        self.assertEqual(r.parsed["_confidence_clamped_from"], 1.5)

    def test_置信度缺失为None(self):
        r = parse_decision('{"action":"buy","size":1,"reason":"x"}')
        self.assertIsNone(r.parsed["confidence"])

    def test_可选数值键总是存在(self):
        """风控层的契约是"键在但可为 None"，缺键会让它走另一条分支。"""
        r = parse_decision('{"action":"hold","reason":"x"}')
        for k in ("sz", "px", "tp", "sl"):
            self.assertIn(k, r.parsed)
            self.assertIsNone(r.parsed[k])

    def test_没给订单类型时按有无限价推(self):
        a = parse_decision('{"action":"buy","size":1,"limit_price":5,"reason":"x"}')
        b = parse_decision('{"action":"buy","size":1,"reason":"x"}')
        self.assertEqual(a.parsed["ordType"], "limit")
        self.assertEqual(b.parsed["ordType"], "market")

    def test_动作归一化留痕(self):
        r = parse_decision('{"action":"LONG","size":1,"reason":"x"}')
        self.assertEqual(r.parsed["action"], "buy")
        # ⚠️ ``_raw_action`` 存**模型写字面上的原文**（大写也照存）——
        # 归一化后的值已经在 action 里了，留痕要的是"它当时到底写了什么"。
        self.assertEqual(r.parsed["_raw_action"], "LONG")


class TestConsistency(unittest.TestCase):
    def test_签名只看动作(self):
        """⭐ 含 reason 会让一致性永远接近 0——那衡量的是"会不会换词"。"""
        self.assertEqual(decision_signature({"action": "buy", "reason": "a"}),
                         decision_signature({"action": "buy", "reason": "完全不同的说法"}))
        self.assertNotEqual(decision_signature({"action": "buy"}),
                            decision_signature({"action": "sell"}))

    def test_多数票统计(self):
        s = [{"action": "sell"}, {"action": "hold"}, {"action": "hold"}]
        st = consistency(s)
        self.assertEqual(st["majority"], "hold")
        self.assertAlmostEqual(st["majority_frac"], 2 / 3)

    def test_全一致(self):
        st = consistency([{"action": "buy"}] * 3)
        self.assertAlmostEqual(st["majority_frac"], 1.0)

    def test_平票退回hold(self):
        """⭐ 平票 = 模型自己也没定。按"少数服从多数"硬选方向是把噪声当信号。"""
        got = majority_sample([{"action": "buy", "confidence": 0.9},
                               {"action": "sell", "confidence": 0.9}])
        self.assertEqual(got["action"], "hold")
        self.assertIn("_vote", got)

    def test_多数派里取置信度最高(self):
        got = majority_sample([
            {"action": "buy", "confidence": 0.3, "sz": 1},
            {"action": "buy", "confidence": 0.9, "sz": 2},
            {"action": "hold", "confidence": 0.9},
        ])
        self.assertEqual(got["action"], "buy")
        self.assertEqual(got["sz"], 2)

    def test_全空返回hold(self):
        got = majority_sample([{}, {}])
        self.assertEqual(got["action"], "hold")

    def test_空列表(self):
        self.assertEqual(consistency([])["n"], 0)


# ======================================================================
# visible_from_series —— 防泄漏
# ======================================================================
class TestVisibleFromSeries(unittest.TestCase):
    def test_只用tick及之前的数据(self):
        """⭐⭐ 最要紧的一条。

        ``series.close[-N:]`` 取的是**序列末尾**。调用方若把整条序列传进来
        而 ``i`` 停在中间（回放历史某一天），它会取到**未来**的根——
        不报错，只让那天的决策"神奇地准"。
        """
        s = _Series(list(range(100, 200)))       # 100..199
        v = visible_from_series(s, 10, n_closes=5)
        self.assertEqual(v["recent_closes"], [106.0, 107.0, 108.0, 109.0, 110.0])
        self.assertNotIn(111.0, v["recent_closes"])
        self.assertEqual(v["mid"], 110.0)        # closes[10]

    def test_mid取收盘价而不是高低均值(self):
        """⭐ high/low 是**事后才知道**的极值。用它们当 mid，
        等于把"这根 K 线内最高能到哪"提前告诉模型——
        同一根 K 线内的泄漏，看起来"不过分"，其实很致命。"""
        class S2:
            close = np.array([100.0, 104.0])
            high = np.array([100.0, 110.0])
            low = np.array([100.0, 90.0])
        v = visible_from_series(S2(), 1)
        self.assertEqual(v["mid"], 104.0)        # 不是 (110+90)/2 = 100
        self.assertNotIn("high", v)
        self.assertNotIn("low", v)

    def test_序列头部不越界(self):
        s = _Series([1.0, 2.0, 3.0])
        v = visible_from_series(s, 0, n_closes=12)
        self.assertEqual(v["recent_closes"], [1.0])

    def test_索引越界报错(self):
        s = _Series([1.0, 2.0])
        with self.assertRaises(ValueError):
            visible_from_series(s, 5)

    def test_负索引报错(self):
        s = _Series([1.0, 2.0])
        with self.assertRaises(ValueError):
            visible_from_series(s, -1)

    def test_不带fundamental(self):
        """真实市场没有"基本面价值"，不该伪造一个。"""
        v = visible_from_series(_Series([1.0, 2.0]), 1)
        self.assertNotIn("fundamental", v)

    def test_不带未来字段(self):
        v = visible_from_series(_Series([1.0, 2.0]), 1)
        self.assertFalse([k for k in v if k.startswith("future")])


# ======================================================================
# TradingAgent
# ======================================================================
class TestTradingAgent(unittest.TestCase):
    def test_正常下单(self):
        r = _agent([BUY_OK]).decide(visible=_vis(), tick=1, equity=1e5)
        self.assertTrue(r.parse_ok)
        self.assertTrue(r.executed)
        self.assertEqual(r.order["side"], "buy")
        self.assertEqual(r.order["ordType"], "limit")
        self.assertEqual(r.order["attachAlgoOrds"][0]["tpTriggerPx"], 104.0)
        self.assertEqual(r.risk["rule"], "pass")

    def test_一百倍错误被风控拦住(self):
        """⭐⭐ A3 与 A2 的接缝处最重要的一条断言。

        实测 agnes 第 2 次调用给出 ``tp=10350``（应为 103.5）。
        解析层**原样放行**（不许替它改），所以必须由风控拦住——
        如果这里没拦住，那个 TP 会挂在 100 倍远、SL 永不触发，
        而留痕显示"已设止盈止损"。
        """
        r = _agent(['{"action":"buy","size":0.1,"take_profit":10350,'
                    '"stop_loss":10180,"confidence":0.5,"reason":"x"}']
                   ).decide(visible=_vis(), tick=1, equity=1e5)
        self.assertFalse(r.risk["accepted"])
        self.assertEqual(r.risk["rule"], "tp_sl_off_market")
        self.assertFalse(r.executed)
        self.assertEqual(r.order, {})
        self.assertTrue(any("偏离" in n for n in r.risk["notes"]))

    def test_解析失败也留痕且不算弃权(self):
        """⭐ 用 ``action=="hold"`` 统计弃权率，会把管线故障
        算进"Agent 很保守"里。区分它们的是 ``parse_ok``。"""
        r = _agent(["我觉得可以买一点。"]).decide(visible=_vis(), tick=1, equity=1e5)
        self.assertFalse(r.parse_ok)
        self.assertFalse(r.executed)
        self.assertEqual(r.llm_raw, "我觉得可以买一点。")   # 原文不截断
        self.assertTrue(r.parse_error)
        # ⚠️ ``parsed`` 保持**空**是诚实的：模型什么都没给出来。
        # 不要为了"看起来整齐"把它填成 hold——那会伪造一条意图。
        self.assertEqual(r.parsed, {})
        # 但**最终行为**（风控后）是 hold，必须能从留痕里看到
        self.assertEqual(r.risk["final_intent"]["action"], "hold")
        # ⭐ 且必须显式说明"这个 hold 是解析失败导致的"
        self.assertTrue(any("解析失败" in n for n in r.risk["notes"]))

    def test_风控后最终意图被记录(self):
        """⭐ 不记 final 会出现荒谬的空缺：留痕能看到"被拒了"，
        却看不到"最后到底是什么"。"""
        r = _agent(['{"action":"buy","size":9999,"confidence":0.9,"reason":"x"}']
                   ).decide(visible=_vis(), tick=1, equity=1e5)
        self.assertEqual(r.risk["final_intent"]["action"], "buy")
        self.assertLess(r.risk["final_intent"]["sz"], 9999.0)   # 已是裁剪后的量

    def test_被拒时最终意图是带理由的hold(self):
        r = _agent(['{"action":"buy","size":0.1,"take_profit":10350,'
                    '"stop_loss":10180,"confidence":0.5,"reason":"x"}']
                   ).decide(visible=_vis(), tick=1, equity=1e5)
        self.assertEqual(r.risk["final_intent"]["action"], "hold")
        self.assertTrue(r.risk["final_intent"]["reason"])

    def test_调用失败也是留痕(self):
        client = ScriptedClient(config=AGNES, responses=[
            LLMResponse(ok=False, text="", error="HTTP 500: boom"),
        ], on_exhausted="hold")
        ag = TradingAgent(client=client, config=AgentConfig())
        r = ag.decide(visible=_vis(), tick=1, equity=1e5)
        self.assertFalse(r.parse_ok)
        self.assertFalse(r.executed)

    def test_量大被裁剪但保留原请求(self):
        r = _agent(['{"action":"buy","size":100000,"confidence":0.9,"reason":"重仓"}']
                   ).decide(visible=_vis(), tick=1, equity=1e5)
        self.assertTrue(r.risk["accepted"])
        self.assertEqual(r.risk["rule"], "size_cap")
        self.assertEqual(r.requested["sz"], 100000.0)        # 风控前
        self.assertLess(r.order["sz"], 100000.0)             # 风控后
        self.assertTrue(r.executed)

    def test_弃权不成交(self):
        r = _agent(['{"action":"hold","reason":"看不清"}']
                   ).decide(visible=_vis(), tick=1, equity=1e5)
        self.assertTrue(r.risk["accepted"])
        self.assertFalse(r.executed)
        self.assertEqual(r.order, {})
        self.assertEqual(r.risk["rule"], "hold")

    def test_市价单不带价格(self):
        r = _agent(['{"action":"buy","size":0.1,"order_type":"market",'
                    '"limit_price":102.44,"confidence":0.5,"reason":"x"}']
                   ).decide(visible=_vis(), tick=1, equity=1e5)
        self.assertTrue(r.executed)
        self.assertEqual(r.order["ordType"], "market")
        self.assertIsNone(r.order["px"])

    def test_只有止盈也能挂(self):
        r = _agent(['{"action":"buy","size":0.1,"take_profit":110.0,'
                    '"confidence":0.5,"reason":"x"}']
                   ).decide(visible=_vis(), tick=1, equity=1e5)
        self.assertTrue(r.executed)
        algo = r.order["attachAlgoOrds"][0]
        self.assertEqual(algo["tpTriggerPx"], 110.0)
        self.assertIsNone(algo["slTriggerPx"])

    def test_无有效mid时不该发起决策(self):
        """记一条无意义的记录比不记更糟。"""
        ag = _agent([BUY_OK])
        with self.assertRaises(ValueError):
            ag.decide(visible={"mid": None}, tick=1, equity=1e5)

    def test_多采样进入留痕(self):
        r = _agent(['{"action":"buy","size":0.1,"confidence":0.5,"reason":"a"}',
                    '{"action":"buy","size":0.2,"confidence":0.9,"reason":"b"}'],
                   n_samples=2, temperature=0.2).decide(visible=_vis(), tick=1, equity=1e5)
        self.assertEqual(r.n_samples, 2)
        self.assertEqual(len(r.samples), 2)
        self.assertEqual(r.parsed["_vote"], {"buy": 2})
        # 多数派里取置信度最高的
        self.assertEqual(r.parsed["sz"], 0.2)

    def test_多采样时部分失败仍算成功(self):
        """⭐⭐ 真机跑出来的坑。

        3 次采样里第 1 次失败、后 2 次成功并选出了多数派。
        当时的判据是「第一次有没有成功」（``parsed_list[0]``），
        于是记成 ``parse_ok=False``——而且 ``parse_error`` 还是**空串**，
        一条"失败但没说为什么"的记录。

        正确判据是「**我们有没有拿到可用的意图**」。
        """
        ag = _agent(["这不是 JSON", '{"action":"buy","size":0.1,"confidence":0.5,"reason":"a"}',
                     '{"action":"buy","size":0.1,"confidence":0.5,"reason":"b"}'],
                    n_samples=3, temperature=0.2)
        r = ag.decide(visible=_vis(), tick=1, equity=1e5)
        self.assertTrue(r.parse_ok, "拿到了可用意图就该算成功")
        self.assertEqual(r.parsed["action"], "buy")
        # 但部分失败要留痕（它降低可信度，不代表不可用）
        self.assertIn("部分采样解析失败", r.parse_error)
        self.assertIn("3 次中 1 次", r.parse_error)

    def test_全部采样失败才算解析失败(self):
        ag = _agent(["废话一", "废话二"], n_samples=2, temperature=0.2)
        r = ag.decide(visible=_vis(), tick=1, equity=1e5)
        self.assertFalse(r.parse_ok)
        self.assertEqual(r.parsed, {})
        self.assertIn("全部解析失败", r.parse_error)

    def test_解析失败必有说明(self):
        """⭐ 不允许出现"标记失败但没说为什么"的记录。"""
        for texts, n, t in ((["废话"], 1, 0.0), (["废话", "废话"], 2, 0.2)):
            with self.subTest(n=n):
                r = _agent(texts, n_samples=n, temperature=t).decide(
                    visible=_vis(), tick=1, equity=1e5)
                self.assertFalse(r.parse_ok)
                self.assertTrue(r.parse_error.strip(),
                                "parse_ok=False 时 parse_error 不能是空的")
        r = _agent(['{"action":"buy","size":0.1,"confidence":0.9,"reason":"a"}',
                    '{"action":"sell","size":0.1,"confidence":0.9,"reason":"b"}'],
                   n_samples=2, temperature=0.2).decide(visible=_vis(), tick=1, equity=1e5)
        self.assertEqual(r.parsed["action"], "hold")
        self.assertFalse(r.executed)

    def test_单采样时温度0允许(self):
        AgentConfig(n_samples=1, temperature=0.0)   # 不该抛

    def test_多采样时温度0被拒(self):
        """⭐ 温度 0 下多次采样是同输入同输出，一致性恒为 1，
        看起来完美但什么都没测到。"""
        with self.assertRaises(ValueError):
            AgentConfig(n_samples=3, temperature=0.0)

    def test_n_samples至少1(self):
        with self.assertRaises(ValueError):
            AgentConfig(n_samples=0)

    def test_模式被记进留痕(self):
        r = _agent([BUY_OK]).decide(visible=_vis(), tick=1, equity=1e5)
        self.assertEqual(r.mode, "live")

    def test_回放模式被标记(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "r.jsonl"
            # ⚠️ 必须**用 agent 自己发出去的 prompt** 录制。
            # 手工 ``build_messages`` 出来的 prompt 与 agent 实际用的
            # 不同（少了 limits / max_size / position），哈希对不上，
            # 回放会（正确地）报未命中——这正是回放严格性的证据。
            t = _FakeTransport([("ok", _resp_obj(BUY_OK))])
            rec = Recorder(HTTPClient(config=AGNES, transport=t, api_key="k"), p)
            TradingAgent(client=rec, config=AgentConfig()).decide(
                visible=_vis(), tick=1, equity=1e5)
            self.assertEqual(rec.n, 1)

            ag = TradingAgent(client=ReplayClient(records_path=p),
                              config=AgentConfig())
            r = ag.decide(visible=_vis(), tick=1, equity=1e5)
            self.assertEqual(r.mode, "replay_nonet")
            self.assertEqual(r.retrieved, [])

    def test_decision_id确定性(self):
        """同 run / 同 tick / 同可见状态 / 同模板 ⇒ 同 ID（回放才能核对）。"""
        a = _agent([BUY_OK]).decide(visible=_vis(), tick=7, run_id="R", equity=1e5)
        b = _agent([BUY_OK]).decide(visible=_vis(), tick=7, run_id="R", equity=1e5)
        self.assertEqual(a.decision_id, b.decision_id)

    def test_decision_id对延迟不敏感(self):
        """⭐ 延迟每次不同，进 ID 就会让"同一次决策"算出不同 ID。"""
        ag1 = _agent([BUY_OK])
        ag1.client.responses[0].latency_ms = 5
        ag2 = _agent([BUY_OK])
        ag2.client.responses[0].latency_ms = 9999
        self.assertEqual(
            ag1.decide(visible=_vis(), tick=7, run_id="R", equity=1e5).decision_id,
            ag2.decide(visible=_vis(), tick=7, run_id="R", equity=1e5).decision_id,
        )

    def test_decision_id对模板版本敏感(self):
        """⭐ 换了模板版本却算出同一个 ID，A/B 的两条记录就会混成一条，
        而"哪版成绩更好"这个问题从此算不出来。"""
        from unittest.mock import patch

        a = _agent([BUY_OK]).decide(visible=_vis(), tick=7, run_id="R", equity=1e5)
        with patch.dict(TEMPLATES, {"v2": dict(TEMPLATES[DEFAULT_TEMPLATE]),
                                    }, clear=False):
            ag = TradingAgent(client=ScriptedClient.from_json_texts([BUY_OK]),
                              config=AgentConfig(template="v2"))
            b = ag.decide(visible=_vis(), tick=7, run_id="R", equity=1e5)
        self.assertNotEqual(a.prompt_template, b.prompt_template)
        self.assertNotEqual(a.decision_id, b.decision_id)

    def test_模板版本必须真实存在(self):
        """版本号不是任意字符串——写错一个不存在的版本应当立刻报错，
        而不是悄悄退回默认模板（那会让"用的是哪版"变成谎言）。"""
        ag = TradingAgent(client=ScriptedClient.from_json_texts([BUY_OK]),
                          config=AgentConfig(template="v99"))
        with self.assertRaises(KeyError):
            ag.decide(visible=_vis(), tick=7, run_id="R", equity=1e5)

    def test_建议最大量与风控同口径(self):
        """⭐ 两处口径不同会让模型经常撞上限，而"撞上限"不说明策略好坏。"""
        lim = RiskLimits(max_notional=1000.0, max_equity_frac=1.0,
                         max_inst_exposure_frac=1.0)
        ag = _agent([BUY_OK], limits=lim)
        got = ag._suggest_max_size(mid=100.0, equity=100_000.0)
        self.assertAlmostEqual(got, 10.0)      # min(1000, 100000)/100

    def test_无权益时建议量为零(self):
        ag = _agent([BUY_OK])
        self.assertEqual(ag._suggest_max_size(mid=100.0, equity=0.0), 0.0)

    def test_建议量永不超上限(self):
        """⭐⭐ 回归守卫。实测：给模型 17 位有效数字时，它会写 6 位小数
        从而**比上限大**，触发 ``size_cap``——于是 ``resized_frac``
        （"风控贡献了多少"）被纯四舍五入抬高。

        这条测试覆盖各种量级：建议值乘回价格后**必须** ≤ 上限。
        """
        lim = RiskLimits(max_notional=20_000.0, max_equity_frac=0.30)
        ag = _agent([BUY_OK], limits=lim)
        for equity in (0.0, 1.0, 100.0, 1_000.0, 100_000.0, 10_000_000.0):
            for mid in (0.001, 0.5, 50.0, 2_650.0, 80_872.5, 1_000_000.0):
                with self.subTest(equity=equity, mid=mid):
                    got = ag._suggest_max_size(mid=mid, equity=equity)
                    cap = (min(lim.max_notional, equity * lim.max_equity_frac)
                           if equity > 0 else 0.0)
                    # 允许 1e-9 的浮点余量（下取整理论上不该超，但别让
                    # 浮点表示把这条守卫变成偶发红）
                    self.assertLessEqual(got * mid, cap + 1e-9,
                                         f"建议量 {got}×{mid} 超过上限 {cap}")

    def test_建议量是干净的数(self):
        """模型不需要再取整 ⇒ 不会因为取整而超限。"""
        ag = _agent([BUY_OK])
        got = ag._suggest_max_size(mid=80_872.5, equity=100_000.0)
        # 2 位有效数字：0.24 而不是 0.24730285325666945
        self.assertEqual(got, 0.24)

    def test_风控前请求被完整保留(self):
        """⭐ "Agent 要了多少"与"实际下了多少"的差 = 风控的贡献。"""
        r = _agent(['{"action":"buy","size":9999,"confidence":0.9,"reason":"x"}']
                   ).decide(visible=_vis(), tick=1, equity=1e5)
        self.assertEqual(r.requested["action"], "buy")
        self.assertEqual(r.requested["sz"], 9999.0)
        self.assertEqual(r.risk["resized_to"], r.order["sz"])

    def test_四舍五入噪声不算风控干预(self):
        """⭐⭐ 真机跑出来的坑。

        模型照抄"建议最大量"但四舍五入到 6 位小数（``0.247303`` vs
        上限 ``0.247302853…``），触发 ``size_cap``。
        量**确实**被夹了（正确性），但这不是风控干预（归因）——
        而 ``resized_frac`` 是给风控算功劳的。
        """
        ag = _agent([BUY_OK])
        cap_sz = ag._suggest_max_size(mid=102.44, equity=1e5)  # 干净值
        # 手工构造一个"比上限大 1e-6 相对"的请求
        raw = min(ag.limits.max_notional, 1e5 * ag.limits.max_equity_frac) / 102.44
        bumped = raw * (1 + 1e-6)
        r = _agent([json.dumps({"action": "buy", "size": bumped,
                                "confidence": 0.9, "reason": "照抄建议量"})]
                   ).decide(visible=_vis(price=102.44), tick=1, equity=1e5)
        self.assertEqual(r.risk["rule"], "size_cap")
        self.assertIsNotNone(r.risk["resized_to"])          # 量确实被夹了
        self.assertFalse(r.risk["resize_is_material"])      # 但不是干预
        self.assertTrue(any("四舍五入" in n for n in r.risk["notes"]))

    def test_实质超量算风控干预(self):
        r = _agent(['{"action":"buy","size":9999,"confidence":0.9,"reason":"重仓"}']
                   ).decide(visible=_vis(), tick=1, equity=1e5)
        self.assertEqual(r.risk["rule"], "size_cap")
        self.assertTrue(r.risk["resize_is_material"])

    def test_统计区分噪声与干预(self):
        """⭐ ``resized_frac`` 只数实质干预；``resized_any_frac`` 两者都数。"""
        from tw.decision_log import log_stats

        raw = min(20_000.0, 1e5 * 0.30) / 102.44
        noisy = _agent([json.dumps({"action": "buy", "size": raw * (1 + 1e-6),
                                    "confidence": 0.9, "reason": "x"})]
                       ).decide(visible=_vis(price=102.44), tick=1, equity=1e5)
        real = _agent(['{"action":"buy","size":9999,"confidence":0.9,"reason":"x"}']
                      ).decide(visible=_vis(price=102.44), tick=2, equity=1e5)
        s = log_stats([noisy, real])
        self.assertAlmostEqual(s["resized_any_frac"], 1.0)   # 两条都被夹过
        self.assertAlmostEqual(s["resized_frac"], 0.5)       # 只有一条是真干预


# ======================================================================
# run_session
# ======================================================================
class TestRunSession(unittest.TestCase):
    def test_每根K线一条决策(self):
        ag = _agent(['{"action":"hold","reason":"a"}'] * 5)
        res = run_session(ag, _Series([1.0, 2.0, 3.0, 4.0]), run_id="R",
                          equity=1000.0)
        self.assertEqual(len(res), 4)
        self.assertEqual(res.stats()["n"], 4)

    def test_区间可从中间开始(self):
        ag = _agent(['{"action":"hold","reason":"a"}'] * 5)
        res = run_session(ag, _Series(list(range(10))), start=3, end=5,
                          run_id="R", equity=1000.0)
        self.assertEqual([r.tick for r in res.records], [3, 4, 5])

    def test_区间非法报错(self):
        ag = _agent(['{"action":"hold","reason":"a"}'])
        with self.assertRaises(ValueError):
            run_session(ag, _Series([1.0, 2.0]), start=5, end=1, run_id="R")

    def test_落盘并读回(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "d.jsonl"
            ag = _agent(['{"action":"hold","reason":"a"}'] * 3)
            run_session(ag, _Series([1.0, 2.0, 3.0]), run_id="R",
                        equity=1000.0, log_path=str(p))
            recs = DecisionLog(p).read_all()
            self.assertEqual(len(recs), 3)
            self.assertTrue(all(r.decision_id for r in recs))

    def test_账户钩子能覆盖状态(self):
        seen = []

        def hook(rec, i, st):
            seen.append(dict(st))
            return {"equity": st["equity"] + 100, "position_qty": float(i)}

        ag = _agent(['{"action":"hold","reason":"a"}'] * 3)
        res = run_session(ag, _Series([1.0, 2.0, 3.0]), run_id="R",
                          equity=1000.0, on_record=hook)
        self.assertEqual(seen[0]["equity"], 1000.0)
        self.assertEqual(seen[1]["equity"], 1100.0)
        self.assertEqual(seen[2]["equity"], 1200.0)
        self.assertEqual(len(res), 3)

    def test_可见状态写进留痕且不含未来(self):
        ag = _agent(['{"action":"hold","reason":"a"}'] * 2)
        res = run_session(ag, _Series([10.0, 20.0, 30.0]), run_id="R",
                          equity=1000.0)
        self.assertEqual(res.records[0].visible_state["mid"], 10.0)
        self.assertEqual(res.records[0].visible_state["recent_closes"], [10.0])
        self.assertEqual(res.records[1].visible_state["recent_closes"], [10.0, 20.0])

    def test_stats可直接消费(self):
        ag = _agent(['{"action":"hold","reason":"a"}',
                     '{"action":"buy","size":0.01,"confidence":0.5,"reason":"b"}'])
        res = run_session(ag, _Series([1.0, 2.0]), run_id="R", equity=1000.0)
        s = res.stats()
        self.assertAlmostEqual(s["abstain_frac"], 0.5)
        self.assertIn("n_skipped", s)


# ======================================================================
# 回放端到端：同输入 ⇒ 同输出
# ======================================================================
class TestReplayEndToEnd(unittest.TestCase):
    def test_回放产出同一条决策(self):
        """⭐ 这是「管线没有隐藏状态」的证据。

        录一次真机调用（用假 transport 代替网络），之后用回放客户端
        重跑，``parsed`` / ``decision_id`` / ``order`` 必须逐位一致。
        """
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "r.jsonl"
            vis = _vis()
            # ---- 第一遍：录 ----
            t = _FakeTransport([("ok", _resp_obj(BUY_OK))])
            rec = Recorder(HTTPClient(config=AGNES, transport=t, api_key="k"), p)
            live_rec = TradingAgent(client=rec, config=AgentConfig()).decide(
                visible=vis, tick=3, run_id="R", equity=1e5)
            self.assertEqual(rec.n, 1)

            # ---- 第二遍：回放 ----
            ag = TradingAgent(client=ReplayClient(records_path=p),
                              config=AgentConfig())
            rp_rec = ag.decide(visible=vis, tick=3, run_id="R", equity=1e5)

            self.assertEqual(live_rec.parsed, rp_rec.parsed)
            self.assertEqual(live_rec.decision_id, rp_rec.decision_id)
            self.assertEqual(live_rec.order, rp_rec.order)
            self.assertEqual(live_rec.risk["rule"], rp_rec.risk["rule"])

    def test_回放不能用相似场景顶替(self):
        """⭐ 回放**不能**被用来"重演一个相似的场景"——
        可见状态变了就该找不到，而不是静默给一条旧响应。"""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "r.jsonl"
            t = _FakeTransport([("ok", _resp_obj(BUY_OK))])
            rec = Recorder(HTTPClient(config=AGNES, transport=t, api_key="k"), p)
            TradingAgent(client=rec, config=AgentConfig()).decide(
                visible=_vis(102.44), tick=3, run_id="R", equity=1e5)

            ag = TradingAgent(client=ReplayClient(records_path=p),
                              config=AgentConfig())
            with self.assertRaises(KeyError):
                ag.decide(visible=_vis(103.99), tick=3, run_id="R", equity=1e5)

    def test_回放对限额变化也敏感(self):
        """限额变了 ⇒ 发给模型的"建议最大量"就变了 ⇒ prompt 变了。
        回放必须能看出来——否则"换了风控参数"的实验会读到旧响应。"""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "r.jsonl"
            t = _FakeTransport([("ok", _resp_obj(BUY_OK))])
            rec = Recorder(HTTPClient(config=AGNES, transport=t, api_key="k"), p)
            TradingAgent(client=rec, config=AgentConfig(),
                         limits=RiskLimits(max_notional=20_000.0)).decide(
                visible=_vis(), tick=3, run_id="R", equity=1e5)

            ag = TradingAgent(client=ReplayClient(records_path=p),
                              config=AgentConfig(),
                              limits=RiskLimits(max_notional=1_000.0))
            with self.assertRaises(KeyError):
                ag.decide(visible=_vis(), tick=3, run_id="R", equity=1e5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
