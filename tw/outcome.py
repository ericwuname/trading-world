"""结果回填（A6）—— 把证据链第 ⑥ 项接上。**复盘的地基。**

为什么这是 A6 的第一件事
------------------------
A2 的留痕设计里，第 ⑥ 项「扣掉成本后的结果」字段 ``DecisionRecord.outcome``
**一直在，但从来没有人填过**——证据链六项，实际只做到了五项。

没有 ⑥，Agent 就**永远不知道自己上次做得对不对**：
它每根看一次快照、给一个动作、然后忘掉。
用户说的"对了就对了，错了就错了，有种无所谓的感觉"
**不是态度问题，是结构问题**——没有任何机制把结果传回给它。

⚠️⚠️ 一条必须用断言守住的约束
------------------------------
**``outcome`` 只能回填进 ``rec.outcome``，绝不能进 ``rec.visible_state``。**

这条不是新写的纪律——A3 已经用 ``_is_leaky_key()`` 守住了
（拦 ``future_*`` / ``outcome_*`` 等键进可见状态）。本模块要做的是：
1. 回填函数**显式断言**它只写 ``outcome``；
2. 回填前后 ``state_digest(visible_state)`` **必须完全相同**。

⭐ 为什么这是**强**守卫：它把"回填没污染输入"从"靠人记得不要手滑"
变成**机械可验证**的等式。而它是复盘的**正确性前提**——
复盘要用到 outcome，一旦 outcome 渗进可见状态，
"复盘学到的经验"就会带着未来信息回到决策里，整条链路失去意义。

⭐ 回填的**延迟** = 复盘的**可见边界**
------------------------------------
当前时刻 ``t``，只有 ``t + horizon <= t_now`` 的决策才有 outcome。
⇒ 这天然定义了"复盘时能看到什么"：**只有已实现的那些**。
⇒ 这也让"事后诸葛亮"在结构上不可能发生——复盘时拿不到未实现的结果。

于是三层的安全边界是**递进**的：
```
回填延迟（本模块）  →  复盘只吃已实现结果  →  经验按 t' < t 检索
```
缺任何一层都有一处漏。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

from .decision_log import DecisionRecord, state_digest


# ======================================================================
# 配置
# ======================================================================
@dataclass(slots=True)
class BackfillConfig:
    """回填窗口。

    ``horizon`` 是"决策后看多少根"。本项目决策频率是"每根 K 线一次"，
    所以 4 根 ≈ 4 小时；资金费率/手续费按天结算，取 24 也行。
    ⚠️ **窗口要与"策略的持有期"量级相当**：太短看不出对错，
    太长就把"另一件事"的结果算到这条决策头上（混杂）。
    """

    horizon: int = 4
    #: 是否算"最大有利/不利偏移"（区分"方向错"与"时机错"要用它）
    with_mfe_mae: bool = True

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError(f"horizon 必须 >= 1，收到 {self.horizon}")


# ======================================================================
# 回填
# ======================================================================
def _mid_at(series: Any, i: int) -> float | None:
    closes = getattr(series, "close", None)
    if closes is None or not (0 <= i < len(closes)):
        return None
    v = float(closes[i])
    return v if math.isfinite(v) and v > 0 else None


def backfill_one(rec: DecisionRecord, series: Any, *,
                 cfg: BackfillConfig | None = None,
                 equity_curve: list[float] | None = None) -> bool:
    """给**一条**决策回填 ``outcome``。返回是否真的填了。

    没填的情况（都是正常的）：
    - 未来还没发生（``tick + horizon`` 超出序列）——**这是主要的跳过原因**
    - 当时的 mid 不可用（无法算偏移）

    ⚠️ 全程**只读** series、**只写** ``rec.outcome``。
    """
    c = cfg or BackfillConfig()
    t = int(rec.tick)
    p0 = _mid_at(series, t)
    if p0 is None:
        return False
    p1 = _mid_at(series, t + c.horizon)
    if p1 is None:
        # ⚠️ 这一条就是"回填延迟"的实现：
        # 未来还没发生的决策**拿不到 outcome** ⇒ 复盘时看不到它。
        return False

    action = str(rec.parsed.get("action") or "hold").lower()
    # 方向符号：多 +1 / 空 −1 / 弃权 0（弃权没有方向，所以 markout 记 0，
    # 它的对错要靠"弃权质量"另外衡量——见模块末尾的说明）
    sgn = 1.0 if action == "buy" else (-1.0 if action == "sell" else 0.0)
    markout = sgn * (p1 - p0) / p0

    out: dict[str, Any] = {
        "horizon": int(c.horizon),
        "mid_at_decision": p0,
        f"mid_after_h": p1,
        "signed_move_h": (p1 - p0) / p0,
        "markout_h": markout,
        "action": action,
        "filled": bool(rec.executed),
        #: ⚠️ 只有成交了 markout 才有经济含意；没成交它是一个"如果当时做了会怎样"
        "markout_applies": bool(rec.executed and sgn != 0.0),
    }

    if c.with_mfe_mae:
        hi = getattr(series, "high", None)
        lo = getattr(series, "low", None)
        seg = range(t, min(t + c.horizon, len(series.close)) + 1)
        try:
            hmax = max(float(hi[i]) for i in seg)
            lmin = min(float(lo[i]) for i in seg)
            # 相对决策价的有利/不利偏移（按方向带符号）
            up = (hmax - p0) / p0
            dn = (lmin - p0) / p0
            out["mfe"] = max(up, -dn) if sgn >= 0 else max(-dn, up)
            out["mae"] = min(up, -dn) if sgn >= 0 else min(-dn, up)
        except (TypeError, ValueError, IndexError):
            pass

    if equity_curve is not None:
        j = t + c.horizon
        if 0 <= j < len(equity_curve):
            out["equity_after_h"] = float(equity_curve[j])

    # ⚠️ **守卫**：回填只准动 outcome。这里显式断言，
    # 因为一旦它渗进可见状态，"复盘的经验"就会带着未来信息回到决策里。
    digest_before = state_digest(rec.visible_state)
    rec.outcome = out
    rec.outcome_filled = True
    if state_digest(rec.visible_state) != digest_before:
        raise AssertionError(
            "回填改动了 visible_state！outcome 只能进 rec.outcome（第 ⑥ 项），"
            "绝不能进 ① 可见状态——否则复盘学到的经验会带着未来信息回流到决策。"
        )
    return True


def backfill(
    records: Iterable[DecisionRecord],
    series: Any,
    *,
    cfg: BackfillConfig | None = None,
    equity_curve: list[float] | None = None,
) -> dict[str, Any]:
    """批量回填。返回统计（**回填了多少、跳过多少、为什么跳过**）。

    ⚠️ 返回值里刻意分开"**未来未发生**"与"**输入不可用**"两类跳过：
    前者是正常的（那是回填延迟的设计），后者是数据问题。
    混在一起会让人以为"跳过很多 = 正常"，从而漏掉数据缺陷。
    """
    c = cfg or BackfillConfig()
    n_filled = 0
    n_pending = 0
    n_bad_input = 0
    for r in records:
        t = int(r.tick)
        before = r.outcome_filled
        ok = backfill_one(r, series, cfg=c, equity_curve=equity_curve)
        if ok:
            n_filled += 0 if before else 1
        elif _mid_at(series, t) is None:
            n_bad_input += 1
        else:
            n_pending += 1
    return {
        "n": n_filled + n_pending + n_bad_input,
        "n_filled": n_filled,
        "n_pending": n_pending,      # 未来还没发生 —— 正常
        "n_bad_input": n_bad_input,  # 输入不可用 —— 要查
        "horizon": int(c.horizon),
    }


# ======================================================================
# 复盘要用的摘要（把它要判的"对错"先算成人能读的）
# ======================================================================
def outcome_summary(rec: DecisionRecord) -> str:
    """一条决策的事后小结。给复盘 prompt 用（**只含已实现的信息**）。"""
    o = rec.outcome
    if not o:
        return "（结果未到）"
    act = o.get("action", "?")
    mo = o.get("markout_h")
    if mo is None:
        return f"{act} → 结果缺失"
    verdict = "对" if (o.get("markout_applies") and mo > 0) else (
        "错" if o.get("markout_applies") else "未成交")
    return (f"{act} → 后续 {o.get('horizon')} 根 "
            f"{o.get('signed_move_h', 0) * 100:+.3f}%"
            f"（同向 markout {mo * 100:+.3f}%）→ {verdict}")


__all__ = [
    "BackfillConfig",
    "backfill",
    "backfill_one",
    "outcome_summary",
]
