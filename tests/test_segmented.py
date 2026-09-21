"""A6 测试：多段配对度量。

两件最要紧的事：
1. **段必须真的独立**（每段新建 agent + 新建账户）——
   复用账户会让前一段的持仓带进下一段，"独立"就是假的，
   而它正是这个度量效能高的**唯一来源**。
2. **先验证度量本身灵敏**（拿已知差异试），再上真比较。
   一个连人为差异都判不出的度量，在真实比较里也判不出真差异。
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from tw.agent import AgentConfig, TradingAgent
from tw.agent_run import run_agent_session
from tw.account import MarginAccount, MarginConfig
from tw.policy import AlwaysHold, Momentum
from tw.risk import RiskLimits
from tw.segmented import (
    _sd,
    min_detectable_effect,
    paired_verdict,
    power_analysis,
    run_paired_segments,
    segment_correlation,
    segment_ranges,
    sensitivity_check,
)
from tw.simexec import ExecConfig


class _Series:
    """最小 Series 替身（⚠️ 要带真实 Series 必有的字段）。"""

    def __init__(self, closes, spread: float = 0.002):
        self.close = np.asarray(closes, dtype=float)
        self.open = self.close
        self.high = self.close * (1.0 + spread)
        self.low = self.close * (1.0 - spread)
        self.timestamp = np.arange(len(self.close), dtype=np.int64) * 3_600_000


def _series(n=200, drift=0.001, seed=3, vol=0.004):
    rng = np.random.default_rng(seed)
    return _Series(100.0 * np.exp(np.cumsum(rng.normal(drift, vol, n))))


def _mk(policy, **kw):
    def f():
        return TradingAgent(policy=policy,
                            config=AgentConfig(inst_id="X", lever=3.0, **kw),
                            limits=RiskLimits())
    return f


# ======================================================================
# 切段
# ======================================================================
class TestSegmentRanges(unittest.TestCase):
    def test_基本切分(self):
        r = segment_ranges(100, 20, min_history=0)
        self.assertEqual(r, [(0, 19), (20, 39), (40, 59), (60, 79), (80, 99)])

    def test_留下等量历史(self):
        """⭐ 第 0 段也要有等量历史——否则各段**输入不等价**，配对不成立。

        若允许第 0 段从 0 开始，它的 ``recent_closes`` 会比别的段短
        （可见状态里的序列长度不同）⇒ 那不是同一个实验。
        """
        r = segment_ranges(100, 20, min_history=12)
        self.assertEqual(r[0][0], 12)
        for a, b in zip(r, r[1:]):
            self.assertEqual(b[0], a[1] + 1, "段间必须首尾相接且不重叠")

    def test_段不重叠且不相邻留空(self):
        r = segment_ranges(103, 25, min_history=0)
        for a, b in zip(r, r[1:]):
            self.assertEqual(b[0], a[1] + 1)
        self.assertEqual(r[-1][1], 99)      # 尾部余数不用

    def test_长度不够返回空(self):
        self.assertEqual(segment_ranges(20, 50, min_history=12), [])

    def test_段长非法报错(self):
        with self.assertRaises(ValueError):
            segment_ranges(100, 0)


# ======================================================================
# 配对跑：**"独立"必须是真的**
# ======================================================================
class TestRunPairedSegments(unittest.TestCase):
    def test_每段都是全新的账户(self):
        """⭐⭐ 本文件最要紧的一条。

        如果复用同一个账户，前一段的**持仓会带进下一段**
        ⇒ 段之间不再独立，而"段独立"是这个度量效能高的唯一来源。
        这里用一个**会一直持仓**的策略来暴露它：
        复用账户时，第 2 段的起始持仓不为 0。
        """
        s = _series(n=200, drift=0.004, seed=5)
        r = run_paired_segments(s, {"hold_all": _mk(Momentum(threshold=1e-9))},
                                seg_len=40, min_history=12, keep_runs=True)
        self.assertGreater(r.n_segs, 2)
        for k, per in enumerate(r.runs):
            first = per["hold_all"].records[0]
            inv = float(first.visible_state.get("inventory", 0.0))
            eq = float(first.visible_state.get("equity", 0.0))
            self.assertEqual(inv, 0.0, f"第 {k} 段起始持仓应为 0（账户必须新建）")
            self.assertAlmostEqual(eq, 100_000.0, places=6,
                                   msg=f"第 {k} 段起始权益应为初始值")

    def test_同一段的两个配置读同样的行情(self):
        s = _series(n=200, seed=7)
        r = run_paired_segments(
            s, {"a": _mk(AlwaysHold()), "b": _mk(Momentum(threshold=0.001))},
            seg_len=40, min_history=12, keep_runs=True)
        for per in r.runs:
            va = per["a"].records[0].visible_state
            vb = per["b"].records[0].visible_state
            self.assertEqual(va["mid"], vb["mid"], "同段两配置必须看同一根 K 线")
            self.assertEqual(va["recent_closes"], vb["recent_closes"])

    def test_noop每段都是零(self):
        s = _series(n=200, seed=8)
        r = run_paired_segments(s, {"noop": _mk(AlwaysHold())},
                                seg_len=40, min_history=12)
        for seg in r.net:
            self.assertEqual(seg["noop"], 0.0)

    def test_同一段内各配置的账户也彼此独立(self):
        """⭐⭐ 由 M98 漏网暴露出的**真测试缺口**。

        我原来只检查了"**跨段**账户是否新建"，却没检查
        "**同段内不同配置**是否各用各的账户"。
        后者同样致命：若两个配置**共用一个账户**，
        先跑的那个的成交会改变后跑的那个的权益 ⇒
        **配对就配错了**（B 的成绩里混进了 A 的盈亏）。

        这里用 `momentum`（会一直交易）+ `noop`（永不交易）来暴露：
        共账户时 `noop` 的收益**不再是 0**。
        """
        s = _series(n=200, drift=0.004, seed=16)
        r = run_paired_segments(
            s, {"mom": _mk(Momentum(threshold=0.001)), "noop": _mk(AlwaysHold())},
            seg_len=40, min_history=12)
        for k, seg in enumerate(r.net):
            self.assertEqual(seg["noop"], 0.0,
                             f"第 {k} 段 noop 的收益应恰为 0"
                             f"（不为 0 ⇒ 它共用了别人的账户）")
        # 且这组数据里 mom 确实在动（否则上面的检查没有分辨力）
        self.assertTrue(any(abs(seg["mom"]) > 1e-9 for seg in r.net),
                        "本测试依赖 mom 真的交易过，否则它是摆设")

    def test_段收益率的口径(self):
        """净收益率 = (末权益 − 初始权益) / 初始权益。"""
        s = _series(n=160, seed=9)
        r = run_paired_segments(s, {"m": _mk(Momentum(threshold=0.001))},
                                seg_len=40, min_history=12, keep_runs=True)
        for (a, b), seg, per in zip(r.ranges, r.net, r.runs):
            res = per["m"]
            want = (res.final_equity - res.initial_equity) / res.initial_equity
            self.assertAlmostEqual(seg["m"], want, places=12)

    def test_max_segs可截断(self):
        s = _series(n=400, seed=1)
        r = run_paired_segments(s, {"n": _mk(AlwaysHold())}, seg_len=40,
                                min_history=12, max_segs=3)
        self.assertEqual(r.n_segs, 3)

    def test_切不出段时报错(self):
        with self.assertRaises(ValueError):
            run_paired_segments(_series(n=20), {"n": _mk(AlwaysHold())},
                                seg_len=100, min_history=12)

    def test_进度回调(self):
        seen: list[tuple[int, int]] = []
        s = _series(n=120, seed=2)
        run_paired_segments(s, {"a": _mk(AlwaysHold()), "b": _mk(Momentum())},
                            seg_len=40, min_history=12,
                            progress=lambda i, n: seen.append((i, n)))
        self.assertTrue(seen)
        self.assertEqual(seen[-1][0], seen[-1][1], "最后一次进度应到 100%")


# ======================================================================
# 配对检验
# ======================================================================
class TestPairedVerdict(unittest.TestCase):
    def test_明显为正判显著(self):
        v = paired_verdict([0.01] * 10, name_a="A", name_b="B")
        self.assertEqual(v["verdict"], "A 显著更高")
        self.assertGreater(v["ci"][0], 0)

    def test_明显为负判显著(self):
        v = paired_verdict([-0.01] * 10, name_a="A", name_b="B")
        self.assertEqual(v["verdict"], "B 显著更高")
        self.assertLess(v["ci"][1], 0)

    def test_区间含零判无法判定(self):
        """⭐⭐ 含零 ⇒ **"无法判定"，不是"没有差别"**。

        这两句话在本项目里是严格区分的（见「十条测量纪律」）。
        把"判不出来"说成"没有差别"是本项目一路在纠的毛病。
        """
        v = paired_verdict([0.01, -0.01] * 5, name_a="A", name_b="B")
        self.assertEqual(v["verdict"], "依然无法判定")
        self.assertIsNone(v["direction"])
        self.assertLessEqual(v["ci"][0], 0)
        self.assertGreaterEqual(v["ci"][1], 0)

    def test_完全相等判无法判定(self):
        v = paired_verdict([0.0] * 8)
        self.assertEqual(v["verdict"], "依然无法判定")

    def test_段数不足(self):
        self.assertEqual(paired_verdict([0.1])["verdict"], "依然无法判定")
        self.assertEqual(paired_verdict([])["n"], 0)

    def test_配对检验比不配对灵敏(self):
        """⭐ 配对的全部意义所在：把**段间共同波动**从误差里消掉。

        构造：每个段有一个共同的"行情档位"（大波动），
        而两个配置的差值是恒定的小量。配对后 CI 应该**窄得多**。
        """
        base = [0.05, -0.04, 0.03, -0.02, 0.06, -0.05, 0.02, -0.01]
        eps = 0.002
        a = [b + eps for b in base]
        b = list(base)
        unpaired_ci_hi = (max(a) + min(a)) / 2 - (max(b) + min(b)) / 2

        v = paired_verdict([x - y for x, y in zip(a, b)],
                           name_a="a", name_b="b")
        paired_width = v["ci"][1] - v["ci"][0]
        self.assertEqual(v["verdict"], "a 显著更高")
        self.assertLess(paired_width, abs(unpaired_ci_hi) + 0.01,
                        "配对后区间应当很窄（共同波动被消掉）")

    def test_t值方向正确(self):
        up = paired_verdict([0.01] * 8)["t"]
        dn = paired_verdict([-0.01] * 8)["t"]
        self.assertGreater(up, 0)
        self.assertLess(dn, 0)

    def test_小样本用t临界而不是1_96(self):
        """⚠️ n=3 时 t 临界值 ≈ 4.3，**用 1.96 会把"判不出"说成"显著"**。

        这条守的是"小样本别乱下结论"——而它最容易错的方式就是
        图省事用正态近似（1.96）顶替 t 临界值。
        """
        # 挑一组 **1.96 < t < t_crit** 的样本：误用 1.96 会判显著，
        # 用正确的 t 临界值则判不出。这才是这条测试有分辨力的地方。
        d = [0.010, 0.015, 0.005]        # 均值 0.010，σ = 0.005，t ≈ 3.46
        v = paired_verdict(d)
        self.assertGreater(v["t_crit"], 4.0, "n=3 的 t 临界值应 ≈ 4.3")
        self.assertLess(abs(v["t"]), v["t_crit"], "t 应小于临界值（判不出）")
        self.assertGreater(abs(v["t"]), 1.96, "但大于 1.96（误用就会判显著）")
        self.assertEqual(v["verdict"], "依然无法判定")
        # 对照：同一个样本，用正态近似的区间会排除 0
        import math as _m
        se = v["sd"] / _m.sqrt(len(d))
        self.assertGreater(v["mean_diff"] - 1.96 * se, 0,
                           "错用 1.96 时区间会排除 0 —— 这正是要防的误判")


# ======================================================================
# 段间相关（"独立"必须实测）
# ======================================================================
class TestSegmentCorrelation(unittest.TestCase):
    def test_独立序列相关接近零(self):
        rng = np.random.default_rng(11)
        net = [{"x": float(v)} for v in rng.normal(0, 0.01, 200)]
        self.assertLess(abs(segment_correlation(net, "x")), 0.15)

    def test_正自相关能被测出(self):
        """⭐ "段独立"不能假设——段在时间上连续，行情有自相关。"""
        xs, x = [], 0.0
        for _ in range(200):
            x = 0.9 * x + float(np.random.default_rng(len(xs)).normal(0, 0.01))
            xs.append(x)
        net = [{"x": v} for v in xs]
        self.assertGreater(segment_correlation(net, "x"), 0.5)

    def test_太少段返回nan(self):
        self.assertTrue(math.isnan(segment_correlation([{"x": 1.0}], "x")))


# ======================================================================
# 功效分析
# ======================================================================
class TestPower(unittest.TestCase):
    def test_效应越小需要越多段(self):
        few = power_analysis(0.02, 0.01)["required_segs"]
        many = power_analysis(0.002, 0.01)["required_segs"]
        self.assertLess(few, many)

    def test_方差越大需要越多段(self):
        a = power_analysis(0.01, 0.01)["required_segs"]
        b = power_analysis(0.01, 0.05)["required_segs"]
        self.assertLess(a, b)

    def test_零效应返回None(self):
        self.assertIsNone(power_analysis(0.0, 0.01)["required_segs"])

    def test_最小可分辨效应随段数下降(self):
        a = min_detectable_effect(0.01, 4)
        b = min_detectable_effect(0.01, 100)
        self.assertGreater(a, b)
        # ⚠️ **这条断言改过一次，值得记下来。**
        # 旧版写的是"4 倍段数 ⇒ 效应减半（√ 关系）"，即 `a/b == 5`。
        # 它当时成立**只因为** `min_detectable_effect` 把临界值写成了
        # 常数 1.96+0.84（于是两个 n 的 z 一样，比值只剩 √n）。
        # 改成 df 相关的 `t_crit95(n−1)` 后，**小样本那一端的临界值更大**
        # （df=3 时 3.18，df=99 时 1.98）⇒ 比值变成 **7.11 > 5**。
        # ⇒ 正确的表述是：**"加段数"的收益比 √n 更大**，
        #    因为同时还在买"临界值变小"这件事。
        self.assertGreater(a / b, math.sqrt(100 / 4))
        self.assertAlmostEqual(a / b, 7.112, places=2)

    def test_段数不足返回nan(self):
        self.assertTrue(math.isnan(min_detectable_effect(0.01, 1)))


# ======================================================================
# ⭐ 灵敏度自检（用已知差异试新度量）
# ======================================================================
class TestSensitivityCheck(unittest.TestCase):
    def test_能判出已知的成本差异(self):
        """⭐⭐ **这是新度量上线前的验收**。

        同一个策略、同一段行情，只改**成本倍数**——
        这是一个方向确定、量级大致已知的差异（成本越高、净收益越低）。
        如果新度量连这个都判不出来，它在真实比较里也判不出真差异。

        ⇒ 先在**人造差异**上过这一关，再上真实验。
        """
        s = _series(n=400, drift=0.002, seed=13)
        res = sensitivity_check(s, _mk(Momentum(threshold=0.001)),
                                seg_len=50, min_history=12,
                                cost_low=1.0, cost_high=4.0,
                                max_segs=6)
        self.assertTrue(res["passed"],
                        f"应判出「成本×1 显著更高」，实际 {res['verdict']['verdict']}")
        self.assertEqual(res["n_segs"], 6)
        self.assertGreater(res["verdict"]["mean_diff"], 0)
        self.assertGreater(res["verdict"]["ci"][0], 0)

    def test_自检说明里带最小可分辨效应(self):
        s = _series(n=300, drift=0.002, seed=14)
        res = sensitivity_check(s, _mk(Momentum(threshold=0.001)),
                                seg_len=50, min_history=12, max_segs=4)
        self.assertTrue(math.isfinite(res["min_detectable_effect"]))
        self.assertIn("度量", res["note"])

    def test_noop下成本差异恒为零(self):
        """noop 不交易 ⇒ 成本倍数是恒等变换 ⇒ 自检**应该判不出来**。

        ⚠️ 这条是**反向守卫**：如果它被判成"显著"，
        说明度量在拿噪声当信号。
        """
        s = _series(n=300, seed=15)
        res = sensitivity_check(s, _mk(AlwaysHold()),
                                seg_len=50, min_history=12, max_segs=5)
        self.assertFalse(res["passed"])
        self.assertEqual(res["verdict"]["mean_diff"], 0.0)
        self.assertEqual(res["verdict"]["verdict"], "依然无法判定")


# ======================================================================
# 与旧度量的对比（说明为什么换）
# ======================================================================
class TestVersusOldMetric(unittest.TestCase):
    def test_新度量在新段上判出旧度量判不出的差异(self):
        """⭐ 把"为什么要换"做成可复现的断言。

        同一个人造的、方向确定的差异（成本 ×1 vs ×4）：
        - 旧度量（每根平均收益）：差值相对**每根**极小（bp 量级），
          而每根收益的方差由行情主导 ⇒ 区间宽到包含 0；
        - 新度量（段总收益配对）：差值被放大到段量级，且配对消掉了共同波动。

        ⇒ 这里比较的是**"配对差值 / 其标准误"**（即 t 值），
        新度量应当高得多。
        """
        s = _series(n=400, drift=0.002, seed=21)
        mk = _mk(Momentum(threshold=0.001))

        low = run_paired_segments(s, {"x": mk}, seg_len=50, min_history=12,
                                  exec_config=ExecConfig(cost_multiplier=1.0))
        high = run_paired_segments(s, {"x": mk}, seg_len=50, min_history=12,
                                   exec_config=ExecConfig(cost_multiplier=4.0))
        diffs = [a["x"] - b["x"] for a, b in zip(low.net, high.net)]
        v = paired_verdict(diffs)
        self.assertGreater(abs(v["t"]), 2.0,
                           "人造差异在新度量下应当判得出（|t|>2）")
        self.assertGreater(v["ci"][0], 0)


class TestSegmentOffsetRobustness(unittest.TestCase):
    """⭐ `offset` = 本项目里的**稳健性旋钮**（相当于别的实验的"换种子"）。

    本设计没有随机性，唯一的"任意选择"就是**从哪一根开始切段**。
    单靠一组窗口得出的结论，与"单标的"一样不可靠。
    """

    def test_偏移真的换了窗口(self):
        from tw.segmented import segment_ranges
        a = segment_ranges(200, 8, min_history=12, offset=0)
        b = segment_ranges(200, 8, min_history=12, offset=4)
        self.assertNotEqual(a, b)
        # 而且**没有一段是同一段**（起点全不同）
        self.assertEqual(set(a) & set(b), set())

    def test_偏移不改变段长与历史可见量(self):
        from tw.segmented import segment_ranges
        for off in range(8):
            rs = segment_ranges(200, 8, min_history=12, offset=off)
            for lo, hi in rs:
                self.assertEqual(hi - lo + 1, 8)
                self.assertGreaterEqual(lo, 12)   # 历史始终够

    def test_偏移为0时与旧行为逐段相同(self):
        """⚠️ 回归护栏：新参数**不能改变默认行为**。"""
        from tw.segmented import segment_ranges
        self.assertEqual(segment_ranges(200, 8, min_history=12),
                         segment_ranges(200, 8, min_history=12, offset=0))

    def test_偏移过大要报错(self):
        """⚠️ **必须挡住 `offset >= seg_len`**：那是把同一组窗口整体平移，
        不是"换一组窗口" ⇒ 会给出**假的稳健性**（看起来验了，其实没验）。"""
        from tw.segmented import segment_ranges
        for bad in (8, 9, -1):
            with self.assertRaises(ValueError):
                segment_ranges(200, 8, min_history=12, offset=bad)

    def test_段数随偏移变化(self):
        """尾部会少掉最多一段（不同 offset 段数可能差 1）——
        这不是 bug，但配对比较时要知道两个 run 的段数可能不同。"""
        from tw.segmented import segment_ranges
        ns = {len(segment_ranges(203, 8, min_history=12, offset=o))
              for o in range(8)}
        self.assertLessEqual(max(ns) - min(ns), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
