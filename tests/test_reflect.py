"""A6 测试：复盘 / 归因 / 经验库。

本文件的核心是**两条机械等式**（不是"我们会注意"）：

- **V4**：``Exp.created_tick == at_tick`` 的经验**不能**出现在 ``at_tick``
  的检索结果里（严格小于）。写 ``<=`` 就会让"本轮刚生成的经验"
  喂给"本轮的决策"。
- **V2**：同一条决策，在「经验库为空」与「经验库只有 ``t' > t`` 的条目」
  两种情况下，**复盘 prompt 逐字节相同**。

V2 是最强的一条：它把"会不会泄漏未来"变成一个能在测试里跑通的等式。
"""

from __future__ import annotations

import json
import unittest

import numpy as np

from tw.decision_log import DecisionRecord, build_visible_state
from tw.outcome import BackfillConfig, backfill_one
from tw.reflect import (
    ATTR_REASONS,
    EXP_KINDS,
    Exp,
    ExperienceStore,
    ReviewConfig,
    attribution_association,
    build_review_messages,
    classify_by_rules,
    duplicate_rate,
    make_exp_id,
    merits_to_store,
    parse_review,
)


class _Series:
    def __init__(self, closes, spread: float = 0.004):
        self.close = np.asarray(closes, dtype=float)
        self.open = self.close
        self.high = self.close * (1.0 + spread)
        self.low = self.close * (1.0 - spread)
        self.timestamp = np.arange(len(self.close), dtype=np.int64) * 3_600_000


def _rec(tick: int, action: str = "buy", *, executed: bool | None = None,
         mid: float = 100.0, did: str = "") -> DecisionRecord:
    vis = build_visible_state(mid=float(mid), equity=100_000.0)
    r = DecisionRecord(
        tick=int(tick), visible_state=vis,
        parsed={"action": action, "reason": "看均线向上"},
        executed=(action != "hold" if executed is None else executed),
    )
    r.decision_id = did or f"d{tick}"
    return r


def _exp(tick: int, lesson: str = "别追高", kind: str = "rule") -> Exp:
    return Exp(exp_id=make_exp_id(created_tick=tick, lesson=lesson,
                                  evidence_ids=[]),
               created_tick=tick, lesson=lesson, kind=kind)


# ======================================================================
# V4：检索的**严格**时间边界
# ======================================================================
class TestRetrieveStrictBoundary(unittest.TestCase):
    def test_同tick的经验不可见(self):
        """⭐⭐ 本文件最重要的一条之一。

        ``created_tick == at_tick`` 必须**看不见**。
        用 ``<=`` 时，同一根 K 线内"复盘产生的经验"会被"本轮的决策"读到
        ⇒ 微小但**真实**的未来泄漏（复盘看得到结果，决策看不到）。
        """
        st = ExperienceStore([_exp(5, "五"), _exp(10, "十"), _exp(10, "十之二")])
        got = st.retrieve(at_tick=10, k=10)
        self.assertEqual([e.lesson for e in got], ["五"])
        self.assertNotIn("十", [e.lesson for e in got])

    def test_只取严格更早的(self):
        st = ExperienceStore([_exp(i, f"e{i}") for i in range(6)])
        got = st.retrieve(at_tick=4, k=10)
        self.assertEqual([e.created_tick for e in got], [0, 1, 2, 3])

    def test_k_取最近的k条(self):
        st = ExperienceStore([_exp(i, f"e{i}") for i in range(10)])
        got = st.retrieve(at_tick=9, k=3)
        self.assertEqual([e.created_tick for e in got], [6, 7, 8])

    def test_排序是确定性的(self):
        """⚠️ 排序必须**只依赖内容**。若按插入顺序，同一条决策在不同运行里
        会看到不同的经验，V2（逐字节相等）就不可能成立。

        ⚠️ 样本用**互不相同的 tick 且插入顺序 ≠ tick 顺序**——
        否则"删掉 sort"这条变异体可能碰巧通过（**没有分辨力**）。
        """
        a = [_exp(5, "e5"), _exp(1, "e1"), _exp(3, "e3")]
        s1 = ExperienceStore(a).retrieve(at_tick=9, k=10)
        s2 = ExperienceStore(list(reversed(a))).retrieve(at_tick=9, k=10)
        self.assertEqual([e.exp_id for e in s1], [e.exp_id for e in s2])
        self.assertEqual([e.created_tick for e in s1], [1, 3, 5])

    def test_空库返回空(self):
        self.assertEqual(ExperienceStore().retrieve(at_tick=100, k=5), [])

    def test_k非正返回空(self):
        st = ExperienceStore([_exp(1)])
        self.assertEqual(st.retrieve(at_tick=100, k=0), [])


# ======================================================================
# ⭐⭐ V2：时间旅行安全（prompt 逐字节相同）
# ======================================================================
class TestTimeTravelSafety(unittest.TestCase):
    def test_未来经验不改变prompt(self):
        """⭐⭐ **最强的验证。**

        同一条决策：
        - 情形 A：经验库为空
        - 情形 B：经验库里**只有** ``t' > t`` 的条目（即"未来才产生的经验"）

        ⇒ 两种情形下复盘 prompt **必须逐字节相同**。
        这等价于"经验库不可能把未来信息带进 prompt"。
        """
        recs = [_rec(10), _rec(11)]
        s = _Series([100, 101, 102, 103, 104, 105, 106, 107, 108,
                     109, 110, 111, 112, 113, 114, 115])
        for r in recs:
            backfill_one(r, s, cfg=BackfillConfig(horizon=2))

        empty = ExperienceStore([])
        future = ExperienceStore([_exp(999, "未来才知道的教训"),
                                  _exp(1000, "更未来的教训")])

        pa = build_review_messages(
            recs, at_tick=12, experiences=empty.retrieve(at_tick=12, k=5))
        pb = build_review_messages(
            recs, at_tick=12, experiences=future.retrieve(at_tick=12, k=5))
        self.assertEqual(json.dumps(pa, ensure_ascii=False, sort_keys=True),
                         json.dumps(pb, ensure_ascii=False, sort_keys=True))

    def test_过去经验会改变prompt(self):
        """反面：``t' < t`` 的经验**必须**出现在 prompt 里。

        ⚠️ 没有这一条，上面那条测试可以靠"永远不注入经验"通过——
        也就是**没有分辨力**（本项目已踩过三次的坑）。
        """
        recs = [_rec(10)]
        past = ExperienceStore([_exp(2, "过去知道的教训")])
        p = build_review_messages(recs, at_tick=12,
                                  experiences=past.retrieve(at_tick=12, k=5))
        self.assertIn("过去知道的教训", p[1]["content"])

    def test_复盘只吃已实现结果(self):
        """内容层边界：**未回填**的决策在 prompt 里 outcome 是空的。

        回填的延迟由 ``tw/outcome.py`` 提供，这里验证它确实
        反映到了复盘 prompt 上（否则"内容层"只是文档里的一句话）。
        """
        s = _Series([100 + i for i in range(8)])
        near = _rec(6)          # horizon=4 时 6+4=10 > 7 ⇒ 拿不到结果
        backfill_one(near, s, cfg=BackfillConfig(horizon=4))
        p = build_review_messages([near], at_tick=7, experiences=[])
        body = p[1]["content"]
        self.assertIn('"outcome": {}', body.replace("  ", " "))

    def test_at_tick不同prompt不同(self):
        """⚠️ 分辨力补强：确认 ``at_tick`` 真的进了正文，
        否则"逐字节相同"可能只是因为 prompt 里没写时刻。"""
        r = [_rec(3)]
        a = build_review_messages(r, at_tick=5, experiences=[])[1]["content"]
        b = build_review_messages(r, at_tick=7, experiences=[])[1]["content"]
        self.assertNotEqual(a, b)
        self.assertIn("tick 5", a)
        self.assertIn("tick 7", b)


# ======================================================================
# exp_id 与 Exp 的结构约束
# ======================================================================
class TestExpIdentity(unittest.TestCase):
    def test_id确定性且与顺序无关(self):
        a = make_exp_id(created_tick=7, lesson="L", evidence_ids=["b", "a"])
        b = make_exp_id(created_tick=7, lesson="L", evidence_ids=["a", "b"])
        self.assertEqual(a, b)

    def test_id含tick(self):
        self.assertNotEqual(
            make_exp_id(created_tick=1, lesson="L", evidence_ids=[]),
            make_exp_id(created_tick=2, lesson="L", evidence_ids=[]))

    def test_无id的经验被拒(self):
        """没有 ID 就无法审计它从哪来 ⇒ 经验库会变成不可证伪的传言。"""
        with self.assertRaises(ValueError):
            ExperienceStore().add(Exp(exp_id="", created_tick=1, lesson="x"))

    def test_kind必须合法(self):
        with self.assertRaises(ValueError):
            Exp(exp_id="a", created_tick=1, lesson="x", kind="乱写")

    def test_序列化往返(self):
        e = _exp(4, "教训", kind="warning")
        e2 = Exp.from_dict(e.to_dict())
        self.assertEqual(e.to_dict(), e2.to_dict())
        st = ExperienceStore.from_dict([e.to_dict()])
        self.assertEqual(len(st), 1)
        self.assertEqual(st.all()[0].lesson, "教训")


# ======================================================================
# ReviewConfig 校验
# ======================================================================
class TestReviewConfig(unittest.TestCase):
    def test_period必须为正(self):
        with self.assertRaises(ValueError):
            ReviewConfig(period=0)

    def test_max_records必须为正(self):
        with self.assertRaises(ValueError):
            ReviewConfig(max_records=0)

    def test_默认是24根(self):
        self.assertEqual(ReviewConfig().period, 24)


# ======================================================================
# 机械归因
# ======================================================================
class TestClassifyByRules(unittest.TestCase):
    def _with_outcome(self, action: str, **out) -> DecisionRecord:
        r = _rec(3, action)
        r.outcome = {"action": action, "markout_applies": True, **out}
        return r

    def test_没结果时unclear(self):
        self.assertEqual(classify_by_rules(_rec(1)), "unclear")

    def test_弃权时unclear(self):
        r = _rec(1, "hold")
        r.outcome = {"action": "hold", "markout_applies": False,
                     "markout_h": 0.0}
        self.assertEqual(classify_by_rules(r), "unclear")

    def test_方向反了(self):
        """⚠️ 这条测试背后有一个**我第一版写错**的判据：
        用 ``markout_h`` 与 ``signed_move_h`` 比符号是**恒假**的
        （markout 已带方向符号）⇒ ``direction_wrong`` 永远判不出来，
        判据成了死代码（不报错、只是永不命中）。"""
        r = self._with_outcome("buy", signed_move_h=-0.01, markout_h=-0.01)
        self.assertEqual(classify_by_rules(r), "direction_wrong")

    def test_做空但价涨也算方向反了(self):
        r = _rec(3, "sell")
        r.outcome = {"action": "sell", "markout_applies": True,
                     "signed_move_h": 0.01, "markout_h": -0.01}
        self.assertEqual(classify_by_rules(r), "direction_wrong")

    def test_走早了算timing(self):
        r = self._with_outcome("buy", signed_move_h=0.02, markout_h=0.001,
                               mfe=0.03, mae=-0.001)
        self.assertEqual(classify_by_rules(r), "timing_wrong")

    def test_曾经有利却回吐算timing(self):
        r = self._with_outcome("buy", signed_move_h=-0.004, markout_h=-0.004,
                               mfe=0.01, mae=-0.004)
        self.assertEqual(classify_by_rules(r), "timing_wrong")

    def test_价格几乎没动算成本吃掉(self):
        r = self._with_outcome("buy", signed_move_h=-0.00002,
                               markout_h=-0.00002)
        self.assertEqual(classify_by_rules(r), "cost_ate_it")

    def test_赢了不硬找原因(self):
        """赢了而 ``mfe`` 不突出 ⇒ ``unclear``。
        **不给"赢了"强行分配一个错误原因**，否则分类会被噪声填满。"""
        r = self._with_outcome("buy", signed_move_h=0.01, markout_h=0.01,
                               mfe=0.012)
        self.assertEqual(classify_by_rules(r), "unclear")

    def test_所有输出都在枚举里(self):
        cases = [
            {"signed_move_h": -0.01, "markout_h": -0.01},
            {"signed_move_h": 0.0, "markout_h": -0.001},
            {"signed_move_h": 0.02, "markout_h": 0.001, "mfe": 0.03},
            {},
        ]
        for out in cases:
            self.assertIn(classify_by_rules(self._with_outcome("buy", **out)),
                          ATTR_REASONS)


# ======================================================================
# V3：归因与可观测结果的关联（只做描述，不下显著结论）
# ======================================================================
class TestAttributionAssociation(unittest.TestCase):
    def test_分类完全决定结果时spread大(self):
        res = attribution_association([
            ("direction_wrong", -0.01), ("direction_wrong", -0.02),
            ("lucky", 0.01), ("lucky", 0.02),
        ])
        self.assertGreater(res["spread"], 0.0)
        self.assertEqual(res["n_by_reason"]["direction_wrong"], 2)
        self.assertEqual(res["n"], 4)

    def test_解耦类内部混杂时share低(self):
        res = attribution_association([("lucky", 0.01), ("lucky", -0.01),
                                       ("unlucky", -0.01), ("unlucky", 0.01)])
        self.assertAlmostEqual(res["decoupled_share"], 0.5)

    def test_解耦类内部清一色时share高(self):
        res = attribution_association([("lucky", 0.01), ("lucky", 0.02)])
        self.assertAlmostEqual(res["decoupled_share"], 0.0)

    def test_空输入不崩(self):
        res = attribution_association([])
        self.assertEqual(res["n"], 0)
        self.assertTrue(res["spread"] != res["spread"])   # nan

    def test_单分类时spread为nan(self):
        res = attribution_association([("lucky", 0.01)])
        self.assertTrue(res["spread"] != res["spread"])


# ======================================================================
# 解析容错
# ======================================================================
class TestParseReview(unittest.TestCase):
    def test_正常解析(self):
        txt = json.dumps({
            "attributions": [{"decision_id": "d1", "reason": "lucky"}],
            "experiences": [{"condition": {}, "lesson": "别追高",
                             "kind": "warning", "evidence_ids": ["d1"]}],
        }, ensure_ascii=False)
        p = parse_review(txt)
        self.assertTrue(p["ok"])
        self.assertEqual(p["attributions"][0]["reason"], "lucky")
        self.assertEqual(p["experiences"][0]["lesson"], "别追高")

    def test_带前后废话也能解析(self):
        p = parse_review('好的，分析如下：{"attributions":[],'
                         '"experiences":[]} 以上。')
        self.assertTrue(p["ok"])

    def test_非法reason落到unclear且被计数(self):
        """⚠️ **不静默丢弃**：丢弃会让分类分布看起来比实际干净。"""
        p = parse_review(json.dumps({
            "attributions": [{"decision_id": "d1", "reason": "我编的"}],
            "experiences": []}, ensure_ascii=False))
        self.assertEqual(p["attributions"][0]["reason"], "unclear")
        self.assertEqual(p["n_bad_reason"], 1)

    def test_坏JSON不抛异常且留原文(self):
        p = parse_review("{这不是: json}")
        self.assertFalse(p["ok"])
        self.assertIn("解析失败", p["error"])
        self.assertEqual(p["raw"], "{这不是: json}")

    def test_没有花括号时也不抛(self):
        p = parse_review("我拒绝回答")
        self.assertFalse(p["ok"])
        self.assertIn("找不到", p["error"])

    def test_空输出(self):
        p = parse_review("   ")
        self.assertFalse(p["ok"])
        self.assertEqual(p["error"], "空输出")

    def test_没经验的条目被跳过(self):
        p = parse_review(json.dumps({
            "attributions": [],
            "experiences": [{"lesson": "  "}, {"lesson": "有效"}]},
            ensure_ascii=False))
        self.assertEqual(len(p["experiences"]), 1)

    def test_非dict条目被跳过(self):
        p = parse_review(json.dumps({
            "attributions": ["乱写", {"decision_id": "d", "reason": "lucky"}],
            "experiences": ["乱写"]}, ensure_ascii=False))
        self.assertEqual(len(p["attributions"]), 1)

    def test_非法kind落回observation(self):
        p = parse_review(json.dumps({
            "attributions": [],
            "experiences": [{"lesson": "L", "kind": "乱写"}]},
            ensure_ascii=False))
        self.assertEqual(p["experiences"][0]["kind"], "observation")
        self.assertIn(p["experiences"][0]["kind"], EXP_KINDS)


# ======================================================================
# 经验入库：created_tick 由**外部时钟**决定
# ======================================================================
class TestMeritsToStore(unittest.TestCase):
    def test_created_tick由调用方给(self):
        """⚠️ **模型不能决定自己的经验何时生效。**

        若采用模型正文里的时间，它就可以写一个更早的 tick 来绕过
        ``created_tick < at_tick`` 这条检索边界 ⇒ 安全边界被绕开。
        """
        parsed = {"experiences": [{"condition": {}, "lesson": "L",
                                   "kind": "rule",
                                   "evidence_ids": [],
                                   "created_tick": 0}]}
        exps = merits_to_store(parsed, created_tick=77)
        self.assertEqual(exps[0].created_tick, 77)
        self.assertTrue(exps[0].exp_id)

    def test_空lesson被跳过(self):
        self.assertEqual(merits_to_store({"experiences": [{"lesson": ""}]},
                                         created_tick=1), [])

    def test_入库后立刻不可见(self):
        """把两件事接起来：**同一 tick 生成的经验，本 tick 看不见。**
        这是 V2 与 V4 的联合验证。"""
        parsed = {"experiences": [{"condition": {}, "lesson": "刚学到的",
                                   "kind": "rule", "evidence_ids": []}]}
        st = ExperienceStore()
        for e in merits_to_store(parsed, created_tick=50):
            st.add(e)
        self.assertEqual(st.retrieve(at_tick=50, k=5), [])
        self.assertEqual(len(st.retrieve(at_tick=51, k=5)), 1)


# ======================================================================
# V6 的一半：重复率
# ======================================================================
class TestDuplicateRate(unittest.TestCase):
    def test_全不重复(self):
        self.assertAlmostEqual(
            duplicate_rate([_exp(1, "a"), _exp(2, "b")]), 0.0)

    def test_全重复(self):
        self.assertAlmostEqual(
            duplicate_rate([_exp(1, "a"), _exp(2, "a")]), 0.5)

    def test_空返回nan(self):
        self.assertTrue(duplicate_rate([]) != duplicate_rate([]))

    def test_同tick自我重复也算(self):
        self.assertAlmostEqual(
            duplicate_rate([_exp(3, "a"), _exp(3, "a")]), 0.5)


if __name__ == "__main__":
    unittest.main()
