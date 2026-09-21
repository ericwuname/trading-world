"""A2 决策留痕 + 风控闸门的测试。

这组测试的重点不是"代码能跑"，而是**证据链的六个字段真的被记下来了**，
以及**风控真的拦得住第 6 轮实测到的那个 100 倍量级错误**。
"""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from tw.decision_log import (
    REQUIRED_VISIBLE_KEYS,
    SCHEMA_VERSION,
    DecisionLog,
    DecisionRecord,
    build_visible_state,
    _is_leaky_key,
    decision_stability,
    fill_outcome,
    log_stats,
    risk_contribution,
    state_digest,
)
from tw.risk import RiskLimits, check, is_decision_bar


def _rec(**kw):
    base = dict(run_id="run1", tick=100, agent_id="llm#1",
                prompt_template="v1.2", model="agnes-2.5-flash",
                context_hash="abc123")
    base.update(kw)
    return DecisionRecord(**base)


# ======================================================================
# decision_id
# ======================================================================
class TestDecisionId(unittest.TestCase):
    def test_确定性_同输入同id(self):
        a = _rec().finalize()
        b = _rec().finalize()
        self.assertEqual(a.decision_id, b.decision_id)
        self.assertEqual(len(a.decision_id), 32)

    def test_不同tick不同id(self):
        a = _rec(tick=100).finalize()
        b = _rec(tick=101).finalize()
        self.assertNotEqual(a.decision_id, b.decision_id)

    def test_不同run不同id(self):
        self.assertNotEqual(_rec(run_id="a").finalize().decision_id,
                            _rec(run_id="b").finalize().decision_id)

    def test_不同模板版本不同id(self):
        """⭐ 模板改了就是另一条决策 —— 否则 A/B 换 prompt 的实验无法归因。"""
        self.assertNotEqual(_rec(prompt_template="v1").finalize().decision_id,
                            _rec(prompt_template="v2").finalize().decision_id)

    def test_不同模型不同id(self):
        self.assertNotEqual(_rec(model="m1").finalize().decision_id,
                            _rec(model="m2").finalize().decision_id)

    def test_不同上下文不同id(self):
        self.assertNotEqual(_rec(context_hash="x").finalize().decision_id,
                            _rec(context_hash="y").finalize().decision_id)

    def test_墙钟时间不参与id(self):
        """⭐ 时间每次不同，参与进去确定性就没了。"""
        a = _rec(wall_ms=1).finalize()
        b = _rec(wall_ms=999999).finalize()
        self.assertEqual(a.decision_id, b.decision_id)

    def test_延迟不参与id(self):
        a = _rec(latency_ms=10).finalize()
        b = _rec(latency_ms=5000).finalize()
        self.assertEqual(a.decision_id, b.decision_id)

    def test_事后结果不参与id(self):
        """⭐ 回填结果不能改 ID，否则"回填"会让记录对不上。"""
        a = _rec().finalize()
        b = _rec(outcome={"pnl": 123.0}, outcome_filled=True).finalize()
        self.assertEqual(a.decision_id, b.decision_id)

    def test_llm原文不参与id(self):
        """模型原始输出不参与：同一输入两次采样可能给出不同文本，
        但它们**是同一条决策**（回放要能核对上）。"""
        a = _rec(llm_raw="a").finalize()
        b = _rec(llm_raw="b").finalize()
        self.assertEqual(a.decision_id, b.decision_id)

    def test_id不自洽要抛错(self):
        """⭐ 防事后篡改：字段改了但 ID 没重算，必须报错而不是静默修好。"""
        r = _rec().finalize()
        r.tick = 999
        with self.assertRaises(ValueError):
            r.finalize()


# ======================================================================
# 序列化 / schema
# ======================================================================
class TestSerialization(unittest.TestCase):
    def test_往返一致(self):
        r = _rec(visible_state={"mid": 100.0}, parsed={"action": "buy"}).finalize()
        r2 = DecisionRecord.from_dict(r.to_dict())
        self.assertEqual(r2.decision_id, r.decision_id)
        self.assertEqual(r2.visible_state, r.visible_state)
        self.assertEqual(r2.schema, SCHEMA_VERSION)

    def test_未知字段要报错(self):
        """⭐ schema 变了却不升版本，静默读旧记录会出错得无声无息。"""
        d = _rec().to_dict()
        d["brand_new_field"] = 1
        with self.assertRaises(ValueError) as cm:
            DecisionRecord.from_dict(d)
        self.assertIn("brand_new_field", str(cm.exception))

    def test_summary_line_可读(self):
        r = _rec(parsed={"action": "buy", "reason": "趋势向上"}, executed=True)
        self.assertIn("buy", r.summary_line())
        self.assertIn("趋势向上", r.summary_line())

    def test_summary_line_标记被拒(self):
        r = _rec(parsed={"action": "buy"}, executed=False, reject_code="TW-1005")
        self.assertIn("被拒", r.summary_line())
        self.assertIn("TW-1005", r.summary_line())

    def test_summary_line_标记改量(self):
        r = _rec(parsed={"action": "buy"}, executed=True,
                 risk={"resized_to": 0.5})
        self.assertIn("改量", r.summary_line())

    def test_summary_line_四种未成交要分开(self):
        """⭐ 回归守卫（真机跑出来的）。

        一开始"弃权"和"被拒"都显示成 ``[未执行 risk]``——
        而它们含义完全相反：弃权是 Agent 的决定，被拒是风控的决定。
        混在一起看会让人以为风控很激进（其实它什么都没做）。
        """
        cases = {
            "解析失败": _rec(parsed={"action": "hold"}, executed=False,
                          parse_ok=False),
            "被拒": _rec(parsed={"action": "hold"}, executed=False,
                       reject_code="TW-1005"),
            "弃权": _rec(parsed={"action": "hold"}, executed=False,
                       risk={"accepted": True}),
        }
        got = {k: v.summary_line() for k, v in cases.items()}
        for label, tag in (("解析失败", "[解析失败]"), ("被拒", "[被拒"),
                           ("弃权", "[弃权]")):
            with self.subTest(label=label):
                self.assertIn(tag, got[label])
        # 三者互不相同（否则"分开显示"是假的）
        self.assertEqual(len(set(got.values())), 3, got)


# ======================================================================
# visible_state
# ======================================================================
class TestVisibleState(unittest.TestCase):
    def test_必需字段在(self):
        st = build_visible_state(mid=100.0, fundamental=99.0)
        for k in REQUIRED_VISIBLE_KEYS:
            self.assertIn(k, st)

    def test_可选字段缺失不影响(self):
        st = build_visible_state(mid=100.0, fundamental=99.0)
        self.assertNotIn("spread_bp", st)
        self.assertNotIn("recent_closes", st)

    def test_不接受未来数据参数(self):
        """⭐ 从签名上堵住答案泄漏：没有 future_* 这类参数。"""
        import inspect
        sig = inspect.signature(build_visible_state)
        for name in sig.parameters:
            self.assertFalse(name.startswith("future"),
                             f"参数 {name} 像是未来数据，不该出现在这里")
            self.assertFalse(name.startswith("outcome"),
                             f"参数 {name} 像是事后结果，不该出现在这里")

    def test_extra不能挂未来字段(self):
        """⭐ 只有签名守卫是不够的。

        ``**extra`` 是一个不受限的入口——没有这道守卫的话，
        ``build_visible_state(mid=1, fundamental=1, future_close=999)``
        就能把未来数据塞进可见状态，签名上"没有 future_* 参数"的承诺
        会被整个架空。守卫必须落在**键名**上，而不是只落在签名上。
        """
        with self.assertRaises(ValueError) as cm:
            build_visible_state(mid=100.0, fundamental=99.0,
                                future_close=999.0)
        self.assertIn("future_close", str(cm.exception))

    def test_extra不能挂事后结果(self):
        with self.assertRaises(ValueError):
            build_visible_state(mid=100.0, fundamental=99.0,
                                outcome_pnl=12345.0)

    def test_extra大小写与空白也拦(self):
        """只要留一个变体漏过去，守卫就等于没有。"""
        for k in ("Future_Close", "OUTCOME_PNL", " lookahead_x",
                  "hindsight_return"):
            with self.subTest(key=k):
                with self.assertRaises(ValueError):
                    build_visible_state(mid=100.0, fundamental=99.0, **{k: 1.0})

    def test_自定义字段仍然可用(self):
        """⚠️ 守卫不能"一刀切禁止 extra" —— 自定义字段确实有用途，
        要挡的是命名空间，不是这个入口本身。"""
        st = build_visible_state(mid=100.0, fundamental=99.0, news_score=0.7)
        self.assertEqual(st["news_score"], 0.7)

    def test_泄漏键判定是纯函数(self):
        self.assertTrue(_is_leaky_key("future_close"))
        self.assertTrue(_is_leaky_key("outcome"))
        self.assertTrue(_is_leaky_key("LOOKAHEAD"))
        self.assertFalse(_is_leaky_key("prompt_template"))
        self.assertFalse(_is_leaky_key("recent_closes"))
        # ⭐ "future" 只作为**前缀**才算——中间出现不算，
        # 否则 "futures_basis"（期货基差，是当时真的能看到的数据）会被误杀。
        self.assertFalse(_is_leaky_key("futures_basis"))

    def test_哈希确定性(self):
        a = state_digest({"mid": 100.0, "b": 2, "a": 1})
        b = state_digest({"a": 1, "b": 2, "mid": 100.0})
        self.assertEqual(a, b)   # 键顺序不该影响

    def test_哈希对值敏感(self):
        self.assertNotEqual(state_digest({"mid": 100.0}),
                            state_digest({"mid": 100.0001}))

    def test_哈希处理numpy类型(self):
        """⭐ 不静默转 str：那会让同一个值不同表示产生不同哈希。"""
        import numpy as np
        a = state_digest({"mid": np.float64(100.0)})
        b = state_digest({"mid": 100.0})
        self.assertEqual(a, b)
        c = state_digest({"n": np.int64(5)})
        d = state_digest({"n": 5})
        self.assertEqual(c, d)

    def test_哈希拒绝未知类型(self):
        class Weird:
            pass
        with self.assertRaises(TypeError):
            state_digest({"x": Weird()})


# ======================================================================
# DecisionLog 落盘
# ======================================================================
class TestDecisionLog(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "decisions.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def test_追加与读回(self):
        with DecisionLog(self.path) as log:
            log.append(_rec(tick=1, parsed={"action": "buy"}))
            log.append(_rec(tick=2, parsed={"action": "hold"}))
            log.flush()
        recs = DecisionLog(self.path).read_all()
        self.assertEqual(len(recs), 2)
        self.assertEqual(recs[0].tick, 1)

    def test_只追加不覆盖(self):
        """⭐ 与数据层同一条纪律：覆盖掉就没有"当时看到什么"的答案了。"""
        with DecisionLog(self.path) as log:
            log.append(_rec(tick=1))
        with DecisionLog(self.path) as log:
            log.append(_rec(tick=2))
        self.assertEqual(len(DecisionLog(self.path).read_all()), 2)

    def test_拒绝写模式w(self):
        with self.assertRaises(ValueError):
            DecisionLog(self.path).open("w")

    def test_未打开就写要报错(self):
        with self.assertRaises(RuntimeError):
            DecisionLog(self.path).append(_rec())

    def test_重复打开要报错(self):
        log = DecisionLog(self.path).open("a")
        try:
            with self.assertRaises(RuntimeError):
                log.open("a")
        finally:
            log.close()

    def test_空文件读出空列表(self):
        self.assertEqual(DecisionLog(self.path).read_all(), [])

    def test_坏行要报错不静默跳过(self):
        """⚠️ 静默跳过坏行会让"留痕有 3% 的行损坏"这个事实完全消失。"""
        self.path.write_text('{"run_id":"a"}\nNOT JSON\n', encoding="utf-8")
        with self.assertRaises(ValueError) as cm:
            DecisionLog(self.path).read_all()
        self.assertIn("第 2 行", str(cm.exception))

    def test_空行可跳过(self):
        good = json.dumps(_rec(tick=1).to_dict(), ensure_ascii=False)
        self.path.write_text(good + "\n\n\n", encoding="utf-8")
        self.assertEqual(len(DecisionLog(self.path).read_all()), 1)

    def test_按run与tick过滤(self):
        with DecisionLog(self.path) as log:
            log.append(_rec(run_id="A", tick=1))
            log.append(_rec(run_id="A", tick=2))
            log.append(_rec(run_id="B", tick=1))
            log.flush()
        log = DecisionLog(self.path)
        self.assertEqual(len(log.load_run("A")), 2)
        self.assertEqual(len(log.load_run("B")), 1)
        self.assertEqual(len(log.load_tick("A", 1)), 1)

    def test_流式读(self):
        with DecisionLog(self.path) as log:
            for i in range(5):
                log.append(_rec(tick=i))
            log.flush()
        self.assertEqual(len(list(DecisionLog(self.path).iter_records())), 5)

    def test_中文不转义(self):
        """留痕是给人看的，中文 reason 必须可读。"""
        with DecisionLog(self.path) as log:
            log.append(_rec(parsed={"reason": "趋势向上，加仓"}))
            log.flush()
        raw = self.path.read_text(encoding="utf-8")
        self.assertIn("趋势向上", raw)


# ======================================================================
# 结果回填
# ======================================================================
class TestFillOutcome(unittest.TestCase):
    def test_回填记成交(self):
        r = _rec().finalize()
        fill_outcome(r, fills=[
            {"side": "buy", "qty": 1.0, "price": 100.0, "mid": 99.5,
             "is_maker": True},
            {"side": "buy", "qty": 0.5, "price": 100.5, "mid": 99.5,
             "is_maker": True},
        ], fee=0.03, pnl=12.5, markout={"h1": 0.4, "h20": -2.1})
        self.assertTrue(r.outcome_filled)
        self.assertEqual(r.outcome["n_fills"], 2)
        self.assertAlmostEqual(r.outcome["filled_qty"], 1.5)
        self.assertAlmostEqual(r.outcome["fill_value"], 150.25)
        self.assertAlmostEqual(r.outcome["pnl"], 12.5)
        self.assertEqual(r.outcome["markout"]["h20"], -2.1)

    def test_回填幂等(self):
        r = _rec().finalize()
        fill_outcome(r, fills=[], pnl=1.0)
        fill_outcome(r, fills=[], pnl=2.0)
        self.assertAlmostEqual(r.outcome["pnl"], 2.0)

    def test_回填不改id(self):
        r = _rec().finalize()
        cid = r.decision_id
        fill_outcome(r, fills=[{"side": "buy", "qty": 1.0, "price": 10.0}])
        self.assertEqual(r.decision_id, cid)
        r.finalize()   # 不该抛

    def test_记录不自洽时拒绝回填(self):
        """⭐ 万一有人先改了字段，回填必须炸而不是把脏数据写进去。"""
        r = _rec().finalize()
        r.tick = 424242
        with self.assertRaises(ValueError):
            fill_outcome(r, fills=[])

    def test_未finalize的记录也能回填(self):
        r = _rec()
        fill_outcome(r, fills=[])
        self.assertTrue(r.decision_id)


# ======================================================================
# 统计
# ======================================================================
class TestLogStats(unittest.TestCase):
    def _mk(self, n=10, **kw):
        return [_rec(tick=i, **kw) for i in range(n)]

    def test_空列表(self):
        self.assertEqual(log_stats([]), {"n": 0})

    def test_各比例(self):
        recs = [
            _rec(tick=0, parse_ok=True, executed=True,
                 parsed={"action": "buy"}, outcome_filled=True),
            _rec(tick=1, parse_ok=False, executed=False, reject_code="TW-1005",
                 parsed={"action": "buy"}),
            _rec(tick=2, parse_ok=True, executed=False,
                 parsed={"action": "hold"}, outcome_filled=True),
            _rec(tick=3, parse_ok=True, executed=True,
                 parsed={"action": "sell"}, risk={"resized_to": 0.5}),
        ]
        s = log_stats(recs)
        self.assertEqual(s["n"], 4)
        self.assertAlmostEqual(s["parse_ok_frac"], 0.75)
        self.assertAlmostEqual(s["executed_frac"], 0.5)
        self.assertAlmostEqual(s["abstain_frac"], 0.25)
        self.assertAlmostEqual(s["resized_frac"], 0.25)
        self.assertAlmostEqual(s["rejected_frac"], 0.25)
        self.assertAlmostEqual(s["outcome_filled_frac"], 0.5)

    def test_incomplete_visible_是回放会退化的信号(self):
        """缺**必需**字段算不完整。

        ⚠️ 判据从 ``REQUIRED_VISIBLE_KEYS`` 现取，不写死字段名——
        曾经这里写死 ``fundamental``，而它后来被改成可选
        （真实市场没有基本面价值，写死会让真实数据 100% 判"不完整"，
        这个指标就退化成噪声）。写死字段名的测试会跟着一起退化。
        """
        req = REQUIRED_VISIBLE_KEYS[0]
        good = _rec(tick=0, visible_state={"mid": 1.0, req: 1.0})
        bad = _rec(tick=1, visible_state={})
        s = log_stats([good, bad])
        self.assertAlmostEqual(s["incomplete_visible_frac"], 0.5)

    def test_可选字段缺失不算不完整(self):
        """⭐ ``fundamental`` 是 ABM 概念，真实数据上必然缺——
        它**不能**被算成"回放会退化"。"""
        rec = _rec(tick=0, visible_state={"mid": 1.0})
        s = log_stats([rec])
        self.assertAlmostEqual(s["incomplete_visible_frac"], 0.0)

    def test_延迟分位(self):
        recs = [_rec(tick=i, latency_ms=v) for i, v in enumerate([10, 20, 30, 40])]
        s = log_stats(recs)
        self.assertEqual(s["latency_ms_max"], 40)
        self.assertGreater(s["latency_ms_p50"], 0)


class TestDecisionStability(unittest.TestCase):
    def test_无多采样(self):
        s = decision_stability([_rec()])
        self.assertEqual(s["n_multi"], 0)
        self.assertTrue(math.isnan(s["mean_consistency"]))

    def test_完全一致(self):
        r = _rec(samples=[{"action": "buy"}, {"action": "buy"},
                          {"action": "buy"}])
        s = decision_stability([r])
        self.assertAlmostEqual(s["mean_consistency"], 1.0)
        self.assertAlmostEqual(s["unanimous_frac"], 1.0)

    def test_三次里两票一致(self):
        r = _rec(samples=[{"action": "sell"}, {"action": "hold"},
                          {"action": "hold"}])
        s = decision_stability([r])
        self.assertAlmostEqual(s["mean_consistency"], 2.0 / 3.0)
        self.assertAlmostEqual(s["unanimous_frac"], 0.0)

    def test_三次全不同(self):
        r = _rec(samples=[{"action": "buy"}, {"action": "sell"},
                          {"action": "hold"}])
        self.assertAlmostEqual(decision_stability([r])["mean_consistency"],
                               1.0 / 3.0)

    def test_实测Agnes那种情形(self):
        """第 6 轮实测：三次调用 1 sell / 2 hold ⇒ 一致性 0.667。
        这个数是"这一次的收益有多少是运气"的直接输入。"""
        r = _rec(samples=[{"action": "sell"}, {"action": "hold"},
                          {"action": "hold"}])
        self.assertAlmostEqual(decision_stability([r])["mean_consistency"],
                               0.6667, places=3)


class TestRiskContribution(unittest.TestCase):
    def test_裁剪比例(self):
        recs = [
            _rec(tick=0, requested={"sz": 10.0}, order={"sz": 5.0}),
            _rec(tick=1, requested={"sz": 10.0}, order={"sz": 10.0}),
            _rec(tick=2, requested={"sz": 10.0}, order={"sz": 0.0},
                 reject_code="TW-1005"),
        ]
        s = risk_contribution(recs)
        self.assertEqual(s["n_sized"], 3)
        self.assertEqual(s["n_fully_allowed"], 1)
        self.assertIn("reject:TW-1005", s["by_rule"])

    def test_按规则分组(self):
        recs = [_rec(tick=i, risk={"rule": "size_cap"}) for i in range(3)]
        self.assertEqual(risk_contribution(recs)["by_rule"]["size_cap"], 3)


# ======================================================================
# ⭐ 风控闸门
# ======================================================================
class TestRiskGateBasic(unittest.TestCase):
    def _ok(self, parsed, **kw):
        base = dict(mid=100.0, equity=10_000.0)
        base.update(kw)
        return check(parsed, **base)

    def test_hold直接通过(self):
        out, d = self._ok({"action": "hold", "reason": "看不清"})
        self.assertTrue(d.accepted)
        self.assertEqual(d.rule, "hold")
        self.assertEqual(out["action"], "hold")

    def test_正常买单通过(self):
        out, d = self._ok({"action": "buy", "sz": 1.0, "px": 100.0,
                           "tp": 110.0, "sl": 95.0, "reason": "突破",
                           "confidence": 0.7})
        self.assertTrue(d.accepted, d.notes)
        self.assertEqual(out["sz"], 1.0)
        self.assertIsNone(d.resized_to)

    def test_正常卖单通过(self):
        out, d = self._ok({"action": "sell", "sz": 1.0, "tp": 90.0,
                           "sl": 110.0})
        self.assertTrue(d.accepted, d.notes)

    def test_action非法被拒(self):
        out, d = self._ok({"action": "yolo", "sz": 1.0})
        self.assertFalse(d.accepted)
        self.assertEqual(d.rule, "bad_action")
        self.assertEqual(out["action"], "hold")
        self.assertTrue(out["reason"])   # 弃权必须带理由

    def test_没有mid无法校验被拒(self):
        out, d = check({"action": "buy", "sz": 1.0}, mid=float("nan"),
                       equity=10_000.0)
        self.assertFalse(d.accepted)
        self.assertEqual(d.rule, "no_mid")


class TestRiskGateSize(unittest.TestCase):
    def _ok(self, parsed, **kw):
        base = dict(mid=100.0, equity=10_000.0)
        base.update(kw)
        return check(parsed, **base)

    def test_数量不可解析被拒(self):
        out, d = self._ok({"action": "buy", "sz": "很多"})
        self.assertFalse(d.accepted)
        self.assertEqual(d.rule, "bad_size")
        self.assertIn("很多", out["reason"])

    def test_数量非正被拒(self):
        for bad in (0.0, -1.0, float("inf"), float("nan")):
            out, d = self._ok({"action": "buy", "sz": bad})
            self.assertFalse(d.accepted, f"sz={bad} 应被拒")

    def test_量超上限被裁剪而不是拒单(self):
        """⭐ 意图明确（"买大了"）时裁剪；这是唯一会改量的一步。"""
        # max_notional=20000, equity=10000*0.3=3000 ⇒ cap=3000 ⇒ sz<=30
        out, d = self._ok({"action": "buy", "sz": 1000.0})
        self.assertTrue(d.accepted)
        self.assertEqual(d.rule, "size_cap")
        self.assertAlmostEqual(out["sz"], 30.0)
        self.assertAlmostEqual(d.resized_to, 30.0)

    def test_裁剪保留原请求在留痕里(self):
        """⭐ "Agent 要了多少"必须留痕 —— 差值就是风控的贡献。"""
        out, d = self._ok({"action": "buy", "sz": 1000.0})
        self.assertAlmostEqual(out["sz"], 30.0)
        self.assertIn("1000", " ".join(d.notes))

    def test_额度为0时拒单(self):
        out, d = check({"action": "buy", "sz": 1.0}, mid=100.0, equity=0.0)
        self.assertFalse(d.accepted)
        self.assertEqual(d.rule, "no_capacity")

    def test_负权益也没有额度(self):
        """equity < 0 时同样不许开仓（不变量要求 equity >= 0）。"""
        out, d = check({"action": "buy", "sz": 1.0}, mid=100.0, equity=-500.0)
        self.assertFalse(d.accepted)
        self.assertEqual(d.rule, "no_capacity")

    # ------------------------------------------------------------------
    # ⭐ 步骤顺序的不变量：**裁量必须早于敞口**（第八轮施工时发现的真缺陷）
    # ------------------------------------------------------------------
    def test_量大时被裁剪而不是被敞口抢先拒掉(self):
        """⭐ 回归守卫。

        施工时这条一开始是**红的**：``sz=1000`` 在 equity=10000 下，
        敞口算出 1000% 远超 60%，于是 ``exposure_cap`` 抢先拒单
        ⇒ ``size_cap`` 那条裁剪分支**不可达**。

        正确的语义是：Agent 说"买 1000"只是**想大了**（意图明确），
        应当按上限裁到 30；拒单会把它错误地归因成"Agent 语义错了"。
        两者的留痕在后续归因里是两个完全不同的结论。
        """
        out, d = self._ok({"action": "buy", "sz": 1000.0})
        self.assertTrue(d.accepted, d.notes)
        self.assertEqual(d.rule, "size_cap")
        # 裁剪后的量必须**自身也满足敞口**，否则等于裁完还是超限
        self.assertLessEqual((out["sz"] * 100.0) / 10_000.0, 0.60)

    def test_裁剪后仍超敞口才拒(self):
        """裁剪不是万能的：若上限本身就把敞口顶穿了，仍要拒。

        构造：把敞口上限压到 5%，而 ``max_equity_frac`` 仍会给 30% 的额度
        ⇒ 裁到 30% 之后依然 ≥ 5%，于是**裁剪发生了、但最终结论是拒**。

        注意 ``rule`` 记的是**决定性的那条**（最后让决策结束的规则），
        所以这里是 ``exposure_cap`` 而不是 ``size_cap`` ——
        "裁过"这个事实在 ``notes`` 里，两者都要能看见（见下一条测试）。
        """
        lim = RiskLimits(max_inst_exposure_frac=0.05)
        out, d = check({"action": "buy", "sz": 100.0}, mid=100.0,
                       equity=10_000.0, limits=lim)
        self.assertFalse(d.accepted)
        self.assertEqual(d.rule, "exposure_cap")
        self.assertIsNone(d.resized_to)   # 拒单时不写入裁剪结果
        self.assertEqual(out["action"], "hold")

    def test_裁剪与敞口都在notes里可见(self):
        lim = RiskLimits(max_inst_exposure_frac=0.05)
        _, d = check({"action": "buy", "sz": 100.0}, mid=100.0,
                     equity=10_000.0, limits=lim)
        joined = " ".join(d.notes)
        self.assertIn("裁剪后名义价值", joined)   # 第 5 步发生了
        self.assertIn("敞口", joined)             # 第 6 步也发生了

    def test_被拒时resized_to必须为空(self):
        """⭐ 回归守卫：``resized_to`` 的语义是"**最终采纳了多少**"。

        若拒单还留着裁剪值，下游会把它算进 ``resized_frac`` 统计，
        并在 ``summary_line`` 里打印成 ``[改量→30]`` —— 而实际一手都没成交。
        这是"字段语义被两种含义混用"的典型，静默且难以发现。
        """
        lim = RiskLimits(max_inst_exposure_frac=0.05)
        _, d = check({"action": "buy", "sz": 100.0}, mid=100.0,
                     equity=10_000.0, limits=lim)
        self.assertFalse(d.accepted)
        self.assertIsNone(d.resized_to)

    def test_被拒的裁剪不计入resized_frac(self):
        """把上面那条从风控层接到留痕层：统计口径必须是干净的。"""
        from tw.decision_log import log_stats
        lim = RiskLimits(max_inst_exposure_frac=0.05)
        _, d = check({"action": "buy", "sz": 100.0}, mid=100.0,
                     equity=10_000.0, limits=lim)
        r = _rec(parsed={"action": "hold", "sz": 0.0, "reason": "风险"},
                 risk=d.to_dict(), parse_ok=True, executed=False,
                 reject_code=d.code)
        st = log_stats([r])
        self.assertEqual(st["resized_frac"], 0.0)   # 没改量
        self.assertEqual(st["rejected_frac"], 1.0)  # 是拒单

    def test_保证金拒单时也不留resized_to(self):
        _, d = check({"action": "buy", "sz": 100.0}, mid=100.0,
                     equity=10_000.0, can_open_ok=False,
                     can_open_reason="TW-1005")
        self.assertFalse(d.accepted)
        self.assertIsNone(d.resized_to)

    def test_敞口用裁剪后的量算而不是原请求(self):
        """⭐ 用裁剪前的量算敞口，等于拿一个不会发生的订单去拒绝一个会发生的订单。"""
        # sz=1000 → 待裁；裁后 30 → 敞口 30%。若用原量算则 1000% 必拒。
        _, d = self._ok({"action": "buy", "sz": 1000.0})
        self.assertNotIn("exposure_cap", d.rule)
        self.assertIn("30.0%", " ".join(d.notes))


class TestRiskGateTpSl(unittest.TestCase):
    """⭐ 这一组直接对应第 6 轮实测到的 100 倍量级错误。"""

    def _ok(self, parsed, **kw):
        base = dict(mid=100.0, equity=10_000.0)
        base.update(kw)
        return check(parsed, **base)

    def test_拦截100倍量级错误(self):
        """实测原样复现：mid≈102.4，模型给出 tp=10350 / sl=10180。
        不过校验就下单 ⇒ TP 挂 100 倍远、SL 永不触发。"""
        out, d = self._ok({"action": "sell", "sz": 0.15, "tp": 10350.0,
                           "sl": 10180.0, "reason": "做空"})
        self.assertFalse(d.accepted)
        self.assertEqual(d.rule, "tp_sl_off_market")
        self.assertEqual(out["action"], "hold")
        self.assertIn("量级错误", out["reason"])

    def test_正常止盈止损通过(self):
        out, d = self._ok({"action": "buy", "sz": 1.0, "tp": 110.0, "sl": 95.0})
        self.assertTrue(d.accepted, d.notes)

    def test_多头止盈在下侧被拒(self):
        """⚠️ 反了会让"止盈"变成立即触发的止损。"""
        out, d = self._ok({"action": "buy", "sz": 1.0, "tp": 90.0})
        self.assertFalse(d.accepted)
        self.assertEqual(d.rule, "tp_sl_wrong_side")
        self.assertIn("上方", out["reason"])

    def test_多头止损在上侧被拒(self):
        out, d = self._ok({"action": "buy", "sz": 1.0, "sl": 110.0})
        self.assertFalse(d.accepted)
        self.assertEqual(d.rule, "tp_sl_wrong_side")
        self.assertIn("下方", out["reason"])

    def test_空头止盈在上侧被拒(self):
        out, d = self._ok({"action": "sell", "sz": 1.0, "tp": 110.0})
        self.assertFalse(d.accepted)
        self.assertEqual(d.rule, "tp_sl_wrong_side")

    def test_空头止损在下侧被拒(self):
        out, d = self._ok({"action": "sell", "sz": 1.0, "sl": 90.0})
        self.assertFalse(d.accepted)
        self.assertEqual(d.rule, "tp_sl_wrong_side")

    def test_止损恰好等于mid被拒(self):
        """恰好相等时方向无法成立（既不在上方也不在下方）。"""
        out, d = self._ok({"action": "buy", "sz": 1.0, "sl": 100.0})
        self.assertFalse(d.accepted)

    def test_止盈止损不可解析被拒(self):
        out, d = self._ok({"action": "buy", "sz": 1.0, "tp": "高一点"})
        self.assertFalse(d.accepted)
        self.assertEqual(d.rule, "bad_tp_sl")

    def test_未设止盈止损默认允许(self):
        out, d = self._ok({"action": "buy", "sz": 1.0})
        self.assertTrue(d.accepted, d.notes)
        self.assertIn("未设", " ".join(d.notes))

    def test_要求TP_SL时缺失被拒(self):
        out, d = check({"action": "buy", "sz": 1.0}, mid=100.0, equity=10_000.0,
                       limits=RiskLimits(require_tp_sl=True))
        self.assertFalse(d.accepted)
        self.assertEqual(d.rule, "missing_tp_sl")


class TestRiskGatePxLeverMargin(unittest.TestCase):
    def _ok(self, parsed, **kw):
        base = dict(mid=100.0, equity=10_000.0)
        base.update(kw)
        return check(parsed, **base)

    def test_委托价偏离过大被拒(self):
        out, d = self._ok({"action": "buy", "sz": 1.0, "px": 10000.0})
        self.assertFalse(d.accepted)
        self.assertEqual(d.rule, "px_off_market")

    def test_委托价非正被拒(self):
        out, d = self._ok({"action": "buy", "sz": 1.0, "px": -5.0})
        self.assertFalse(d.accepted)
        self.assertEqual(d.rule, "bad_px")

    def test_委托价不可解析被拒(self):
        out, d = self._ok({"action": "buy", "sz": 1.0, "px": "市价"})
        self.assertFalse(d.accepted)
        self.assertEqual(d.rule, "bad_px")

    def test_市价单不带px也通过(self):
        out, d = self._ok({"action": "buy", "sz": 1.0, "px": None})
        self.assertTrue(d.accepted, d.notes)

    def test_杠杆超限被拒(self):
        out, d = self._ok({"action": "buy", "sz": 1.0, "lever": 100.0})
        self.assertFalse(d.accepted)
        self.assertEqual(d.rule, "lever_cap")
        self.assertEqual(d.code, "TW-1009")

    def test_杠杆非法被拒(self):
        out, d = self._ok({"action": "buy", "sz": 1.0, "lever": 0.5})
        self.assertFalse(d.accepted)
        self.assertEqual(d.rule, "bad_lever")

    def test_保证金不足被拒(self):
        out, d = self._ok({"action": "buy", "sz": 1.0},
                          can_open_ok=False, can_open_reason="TW-1005")
        self.assertFalse(d.accepted)
        self.assertEqual(d.rule, "margin")
        self.assertEqual(d.code, "TW-1005")

    def test_平仓不受保证金限制(self):
        """已有空仓时买入是平仓，不该被"开仓保证金"拦。"""
        out, d = self._ok({"action": "buy", "sz": 1.0},
                          position_qty=-5.0, can_open_ok=False,
                          can_open_reason="TW-1005")
        self.assertTrue(d.accepted, d.notes)

    def test_持仓数超限被拒(self):
        out, d = self._ok({"action": "buy", "sz": 1.0}, n_open_positions=3)
        self.assertFalse(d.accepted)
        self.assertEqual(d.rule, "too_many_positions")

    def test_敞口超限被拒(self):
        out, d = self._ok({"action": "buy", "sz": 1.0},
                          inst_exposure=0.59 * 10_000.0)
        self.assertFalse(d.accepted)
        self.assertEqual(d.rule, "exposure_cap")

    def test_敞口恰好等于上限被拒(self):
        """边界用 >=：上限的语义是"不得超过"，刚好踩线应当算超。"""
        out, d = self._ok({"action": "buy", "sz": 1.0},
                          inst_exposure=0.59 * 10_000.0)
        # 0.59*10000 + 1*100 = 6000 = 60% 恰好踩线
        self.assertFalse(d.accepted)
        self.assertEqual(d.rule, "exposure_cap")


class TestRiskDecisionShape(unittest.TestCase):
    """证据链第 ⑤ 项必须完整：accept/reject/resize + 规则名 + 逐条说明。"""

    def test_to_dict字段齐全(self):
        _, d = check({"action": "buy", "sz": 1.0}, mid=100.0, equity=10_000.0)
        out = d.to_dict()
        for k in ("accepted", "resized_to", "code", "rule", "notes"):
            self.assertIn(k, out)
        self.assertIsInstance(out["notes"], list)

    def test_被拒时有规则名与说明(self):
        _, d = check({"action": "buy", "sz": 1.0, "tp": 10000.0},
                     mid=100.0, equity=10_000.0)
        self.assertFalse(d.accepted)
        self.assertTrue(d.rule)
        self.assertTrue(d.notes)

    def test_通过时也有逐条说明(self):
        """⭐ "检查了什么、结果如何"——不只是"通过了"。"""
        _, d = check({"action": "buy", "sz": 1.0, "tp": 110.0, "sl": 95.0,
                      "lever": 3.0}, mid=100.0, equity=10_000.0)
        self.assertTrue(d.accepted)
        joined = " ".join(d.notes)
        self.assertIn("tp", joined)
        self.assertIn("sl", joined)
        self.assertIn("杠杆", joined)


# ======================================================================
# 决策窗口
# ======================================================================
class TestDecisionBar(unittest.TestCase):
    def test_每N根一次(self):
        hits = [t for t in range(10) if is_decision_bar(t, bar_ticks=4)]
        self.assertEqual(hits, [0, 4, 8])

    def test_offset让时点整体平移(self):
        """⭐ 没有 offset，预热带长度一变决策时点就整体错位，
        而错位的结果会被误读成"策略表现不同"。"""
        hits = [t for t in range(12) if is_decision_bar(t, bar_ticks=4, offset=2)]
        self.assertEqual(hits, [2, 6, 10])

    def test_offset效果是平移而不是缩放(self):
        a = [t for t in range(20) if is_decision_bar(t, bar_ticks=5)]
        b = [t for t in range(20) if is_decision_bar(t, bar_ticks=5, offset=3)]
        self.assertEqual(len(b), len(a))
        self.assertEqual([x - y for x, y in zip(b, a)], [3] * len(a))

    def test_bar_ticks必须为正(self):
        with self.assertRaises(ValueError):
            is_decision_bar(0, bar_ticks=0)

    def test_每根K线收盘一次是合理默认(self):
        """1 tick ≈ 1 小时，1H 线 ⇒ 每 tick 都是决策窗口。"""
        self.assertTrue(all(is_decision_bar(t, bar_ticks=1) for t in range(5)))


if __name__ == "__main__":
    unittest.main()
