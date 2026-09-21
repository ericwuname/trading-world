"""风控闸门（A2 的一部分）。

为什么风控层**不是可选项**
--------------------------
第六轮的实测给了最直接的证据：Agnes 三次调用里，**第二次把止盈止损的
量级写错了 100 倍**（``10350`` 而不是 ``103.5``）。如果不过校验就直接下单，
TP 挂在 100 倍远的地方、SL 永远不会触发——**账户会带着一个裸仓位一直跑**，
而留痕里看起来一切正常（"Agent 设置了止盈止损"）。

这印证了搜索到的判断：「LLM 可能擅长总结宏观，却是个糟糕的仓位引擎」。

所以本层的定位不是"多一道保险"，而是**把 LLM 的数值输出当成不可信输入**，
像对待用户表单输入一样对待它。

四条规则（每一条都对应上面那个失败的某种形态）
==============================================

**① 数值必须在合理区间内（``sanity``）。**
   TP/SL 相对 mid 的偏离超过 ``max_tp_sl_pct`` 就**退回 hold 并记录原因**。
   这一条直接拦截"10350 而非 103.5"。

**② 方向必须自洽（``direction``）。**
   多头仓位的止盈必须在**上方**、止损在**下方**。反了就是"止盈做成止损"，
   而成品交易系统里这种单子会**立即触发**——比没有更糟。

**③ 量必须有上限（``size``）。**
   单笔名义价值 <= ``max_notional``，且 <= 权益的 ``max_equity_frac``。
   这一条拦"模型说买 10000 手"。

**④ 必须能过账户检查（``margin``）。**
   交给 ``MarginAccount.can_open`` 判，不过就拒（``TW-1005``）。

设计原则：**拒 → 退回 hold，不是退回"小一点的单"**
--------------------------------------------------
第 ③ 条是**裁剪**（Agent 想大了，意图明确，按上限裁），
第 ①②④ 条是**拒绝整单并退回 hold**（数值非法/方向错/保证金不够时，
我们无法知道 Agent 真正想要什么，替它猜一个小单是把自己的判断
伪装成它的判断——留痕里就看不出区别了）。

这个边界与 ``order_model.apply_reduce_only`` 的边界**是同一条**：
**拒绝**与**改量**的分界由「能不能推出意图」决定。

⚠️ 步骤顺序：**裁量必须排在敞口/保证金之前**
------------------------------------------------
这不是风格问题，是**可达性**问题（第八轮施工时被 4 条测试逼出来的）。

反例：Agent 说"买 1000"，账户 10,000。若先判敞口，
``(0 + 1000×100)/10000 = 1000%`` 远超 60% ⇒ 判 ``exposure_cap`` **拒单**，
``size_cap`` 那条裁剪分支**永远走不到**——它成了不可达代码，
而 Agent 明明只是"想大了"（意图明确，本该裁）。

所以顺序是：**先按上限把量裁到合规区间，再用裁剪后的量算敞口与保证金**。
这也符合直觉——敞口和保证金本来就是"实际要下的那个量"的函数，
用裁剪前的量去算它们，等于拿一个不会发生的订单去拒绝一个会发生的订单。

配套的一个反直觉推论：**裁剪步之间不能互相"抢救"**。
若先判敞口、超限就拒，那"想大了"的 Agent 永远学不到"你的量被裁了"，
留痕里只有一片拒单——而拒单和裁剪在归因上是**两件完全不同的事**
（前者说"Agent 语义错了"，后者说"Agent 方向对但尺度大"）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RiskLimits:
    """风控限额。默认值是**保守但可用**的起点，不是"拍出来的"。"""

    #: 单笔名义价值的绝对上限（USDT）。
    max_notional: float = 20_000.0
    #: 单笔名义价值不超过权益的这个比例。
    max_equity_frac: float = 0.30
    #: 杠杆上限。
    max_lever: float = 20.0
    #: 止盈止损相对 mid 的最大偏离（比例）。0.5 = 50%。
    #: 设 0.5 而不是 0.05，是因为真实的止盈有时确实放到 20~30%；
    #: 但 100 倍（``10350`` vs ``103.5``）会被这一条干净地拦下来。
    max_tp_sl_pct: float = 0.5
    #: 同时允许的未平仓决策数（防"一口气开 50 个仓"）。
    max_open_positions: int = 3
    #: 单标的敞口占权益上限。
    max_inst_exposure_frac: float = 0.60
    #: 是否要求止盈止损**必须**存在（= 不允许裸仓）。
    #: 默认 False：强平机制已经兜底，强制 TP/SL 会让 Agent 无法做
    #: "让利润奔跑"的策略。但要留痕里能看出来有没有。
    require_tp_sl: bool = False


@dataclass(slots=True)
class RiskDecision:
    """风控的判定结果。**这就是证据链第 ⑤ 项。**

    ``accepted`` / ``resized_to`` / ``rule`` / ``notes`` 四个字段
    要能完整回答"什么规则接受/拒绝/改了量"。
    """

    accepted: bool = False
    #: 裁剪后的量。None = 没有裁剪（或整单被拒）。
    resized_to: float | None = None
    #: ⭐ 这次裁剪是否**实质**（而不是模型把"建议最大量"四舍五入了一下）。
    #:
    #: 为什么必须单独记：实测真机跑时，模型会**照抄**提示词里的
    #: "建议最大量"，而那个数有 17 位有效数字（``0.24730285325666945``），
    #: 模型写成 ``0.247303`` —— 比上限大了约 1e-5 的相对量。
    #: 不做区分的话，``resized_frac`` 会被这种纯四舍五入抬高，
    #: 而这个指标是**用来量化"风控贡献了多少"的**（A4 要报）——
    #: 把四舍五入算成风控干预，等于给风控记了一笔假功劳。
    #: ⇒ 两个事实都要留：量确实被夹了（正确性），
    #: 但这次不算干预（归因）。**正确性和归因是两件事。**
    resize_is_material: bool = True
    #: 拒单码（``order_model.REJECT_CODES`` 里的键）。
    code: str = ""
    #: 触发的主要规则名（用于分组统计）。
    rule: str = ""
    #: 逐条说明。**要写"检查了什么、结果如何"**，不只是"通过了"。
    notes: list[str] = field(default_factory=list)

    def note(self, s: str) -> None:
        self.notes.append(s)

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": bool(self.accepted),
            "resized_to": self.resized_to,
            "resize_is_material": bool(self.resize_is_material),
            "code": self.code,
            "rule": self.rule,
            "notes": list(self.notes),
        }


def _hold(reason: str) -> dict[str, Any]:
    """退回 hold 的解析结果。**必须带理由**（弃权要评分）。"""
    return {"action": "hold", "sz": 0.0, "px": None, "tp": None, "sl": None,
            "reason": reason, "confidence": 0.0, "downgraded_from": ""}


#: 小于这个**相对**幅度的裁剪，算"四舍五入噪声"而不是风控干预。
#:
#: 标定依据：实测模型会照抄提示词里的"建议最大量"，而那个数有 17 位
#: 有效数字，模型写 6 位小数 ⇒ 相对超出约 1e-5。取 1e-4 留一个数量级
#: 余量。⚠️ 不要再放大这个阈值：10% 的裁剪是**真**干预，
#: 而把 10% 判成噪声会让"风控贡献"这个指标失真——宁可漏判噪声，
#: 不可错判干预。
_MATERIAL_RESIZE_EPS = 1e-4


# ======================================================================
# 主闸门
# ======================================================================
def check(
    parsed: dict[str, Any],
    *,
    mid: float,
    equity: float,
    limits: RiskLimits | None = None,
    position_qty: float = 0.0,
    n_open_positions: int = 0,
    inst_exposure: float = 0.0,
    can_open_ok: bool = True,
    can_open_reason: str = "",
) -> tuple[dict[str, Any], RiskDecision]:
    """把模型解析出来的意图过一遍风控。

    返回 ``(最终意图, 风控判定)``。
    最终意图要么等于输入（通过/裁剪），要么是一个**带理由的 hold**（拒绝）。

    ``can_open_ok`` / ``can_open_reason`` 由调用方用 ``MarginAccount.can_open``
    算好传进来——本函数**不依赖账户对象**，因此可以在没有市场的环境下测试
    全部分支（风控逻辑的重要性决定了它必须能廉价地测穷）。
    """
    lim = limits or RiskLimits()
    d = RiskDecision()

    action = str(parsed.get("action") or "hold").lower()
    reason = str(parsed.get("reason") or "")

    # ---- 0. hold 直接通过（弃权也是一条合法的决策）--------------------
    if action == "hold":
        d.accepted = True
        d.rule = "hold"
        d.note("Agent 选择弃权（hold）")
        return dict(parsed), d

    if action not in ("buy", "sell"):
        d.rule = "bad_action"
        d.code = "TW-1007"
        d.note(f"action 非法: {action!r}")
        return _hold(f"风控拒绝：action 非法（{action}）"), d

    if not (math.isfinite(mid) and mid > 0):
        d.rule = "no_mid"
        d.code = "TW-1007"
        d.note(f"mid 非法: {mid!r}（无有效盘口，无法校验数值）")
        return _hold("风控拒绝：没有有效的中间价，无法校验价格类参数"), d

    # ---- 1. 数量 -------------------------------------------------------
    sz_raw = parsed.get("sz")
    try:
        sz = float(sz_raw)
    except (TypeError, ValueError):
        d.rule = "bad_size"
        d.code = "TW-1007"
        d.note(f"sz 不可解析: {sz_raw!r}")
        return _hold(f"风控拒绝：数量无法解析（{sz_raw!r}）"), d

    if not (math.isfinite(sz) and sz > 0):
        d.rule = "bad_size"
        d.code = "TW-1007"
        d.note(f"sz 非正或非有限: {sz!r}")
        return _hold(f"风控拒绝：数量非法（{sz}）"), d

    # ---- 2. 价格（如果给了限价）----------------------------------------
    px = parsed.get("px")
    if px is not None:
        try:
            pxf = float(px)
        except (TypeError, ValueError):
            d.rule = "bad_px"
            d.code = "TW-1007"
            d.note(f"px 不可解析: {px!r}")
            return _hold(f"风控拒绝：委托价无法解析（{px!r}）"), d
        if not (math.isfinite(pxf) and pxf > 0):
            d.rule = "bad_px"
            d.code = "TW-1007"
            d.note(f"px 非正或非有限: {pxf!r}")
            return _hold(f"风控拒绝：委托价非法（{pxf}）"), d
        # 限价偏离 mid 超过 max_tp_sl_pct 也拒——那不是"挂远一点"，
        # 那是把数字搞错了（同 10350 的形态）。
        dev = abs(pxf - mid) / mid
        if dev > lim.max_tp_sl_pct:
            d.rule = "px_off_market"
            d.code = "TW-1007"
            d.note(f"委托价偏离 mid {dev:.1%} > 上限 {lim.max_tp_sl_pct:.1%}")
            return _hold(
                f"风控拒绝：委托价 {pxf} 偏离中间价 {mid:.4f} 达 {dev:.1%}，"
                f"超出上限 {lim.max_tp_sl_pct:.0%}"
            ), d
        d.note(f"委托价 {pxf} 偏离 mid {dev:.2%} ≤ {lim.max_tp_sl_pct:.0%}")
    else:
        d.note("未给委托价（按市价）")

    # ---- 3. 止盈止损：⭐ 直接拦"100 倍量级错误" -------------------------
    #: 开仓方向决定 TP/SL 应该在价格的哪一侧。
    #: 多头：TP 在上、SL 在下；空头：TP 在下、SL 在上。
    for nm, should_be in (
        ("tp", "above" if action == "buy" else "below"),
        ("sl", "below" if action == "buy" else "above"),
    ):
        v = parsed.get(nm)
        if v is None:
            if lim.require_tp_sl:
                d.rule = "missing_tp_sl"
                d.code = "TW-1007"
                d.note(f"缺少 {nm}，但 require_tp_sl=True")
                return _hold(f"风控拒绝：要求必须带止盈止损，但缺少 {nm}"), d
            d.note(f"{nm} 未设（允许）")
            continue
        try:
            vf = float(v)
        except (TypeError, ValueError):
            d.rule = "bad_tp_sl"
            d.code = "TW-1007"
            d.note(f"{nm} 不可解析: {v!r}")
            return _hold(f"风控拒绝：{nm} 无法解析（{v!r}）"), d
        if not (math.isfinite(vf) and vf > 0):
            d.rule = "bad_tp_sl"
            d.code = "TW-1007"
            d.note(f"{nm} 非正或非有限: {vf!r}")
            return _hold(f"风控拒绝：{nm} 非法（{vf}）"), d

        dev = abs(vf - mid) / mid
        if dev > lim.max_tp_sl_pct:
            d.rule = "tp_sl_off_market"
            d.code = "TW-1007"
            d.note(
                f"{nm}={vf} 偏离 mid {dev:.1%} > 上限 {lim.max_tp_sl_pct:.1%}"
            )
            return _hold(
                f"风控拒绝：{nm} 触发价 {vf} 偏离中间价 {mid:.4f} 达 {dev:.1%}，"
                f"超出上限 {lim.max_tp_sl_pct:.0%}（疑似量级错误）"
            ), d

        #: 方向自洽检查。⚠️ 反了会让"止盈"变成"立即触发的止损"。
        if should_be == "above" and vf <= mid:
            d.rule = "tp_sl_wrong_side"
            d.code = "TW-1007"
            d.note(f"{nm}={vf} 应在 mid({mid:.4f}) **上方**")
            return _hold(
                f"风控拒绝：{nm} 触发价 {vf} 在中间价 {mid:.4f} 的错误一侧"
                f"（{action} 方向应在上方）"
            ), d
        if should_be == "below" and vf >= mid:
            d.rule = "tp_sl_wrong_side"
            d.code = "TW-1007"
            d.note(f"{nm}={vf} 应在 mid({mid:.4f}) **下方**")
            return _hold(
                f"风控拒绝：{nm} 触发价 {vf} 在中间价 {mid:.4f} 的错误一侧"
                f"（{action} 方向应在下方）"
            ), d
        d.note(f"{nm}={vf} 方向与偏离（{dev:.2%}）均正常")

    # ---- 4. 杠杆 -------------------------------------------------------
    lev_raw = parsed.get("lever", 1.0)
    try:
        lev = float(lev_raw)
    except (TypeError, ValueError):
        lev = 1.0
    if not (math.isfinite(lev) and lev >= 1.0):
        d.rule = "bad_lever"
        d.code = "TW-1009"
        d.note(f"lever 非法: {lev_raw!r}")
        return _hold(f"风控拒绝：杠杆非法（{lev_raw!r}）"), d
    if lev > lim.max_lever:
        d.rule = "lever_cap"
        d.code = "TW-1009"
        d.note(f"杠杆 {lev} > 上限 {lim.max_lever}")
        return _hold(
            f"风控拒绝：杠杆 {lev} 超出上限 {lim.max_lever}"
        ), d
    d.note(f"杠杆 {lev} ≤ {lim.max_lever}")

    # ---- 5. 量上限（**唯一的裁剪步**，必须排在敞口/保证金之前）--------
    # 顺序理由见模块 docstring「裁量必须排在敞口/保证金之前」。
    notional = sz * mid
    cap = lim.max_notional
    if equity > 0:
        cap = min(cap, equity * lim.max_equity_frac)
    elif equity <= 0:
        # ⚠️ equity == 0 时**不能**退回 max_notional：账户没有权益就没有
        # 可用额度，"上限"应当是 0 而不是 20,000。若这里不显式清零，
        # equity=0 的单子会以"名义价值 100 ≤ 上限 20,000"通过——
        # 一个 0 本金的账户能开仓，是账户层不变量（equity >= 0）被绕过的口子。
        cap = 0.0
    if cap <= 0:
        d.rule = "no_capacity"
        d.code = "TW-1006"
        d.note(f"可用额度为 0（equity={equity:,.2f}，上限 {cap:,.2f}）")
        return _hold("风控拒绝：没有可用额度"), d
    if notional > cap:
        new_sz = cap / mid
        d.resized_to = float(new_sz)
        d.rule = "size_cap"
        # ⭐ 区分"实质裁剪"与"四舍五入噪声"（见 ``RiskDecision.resize_is_material``）。
        # 判据是**相对**幅度，不是绝对量：0.001 手对 1 手是大裁剪，
        # 对 1,000,000 手是噪声；而阈值必须比典型浮点误差
        # （~1e-16 相对）大几个数量级，否则永远判不出"实质"。
        rel = (sz - new_sz) / sz if sz > 0 else 0.0
        d.resize_is_material = rel > _MATERIAL_RESIZE_EPS
        if d.resize_is_material:
            d.note(
                f"名义价值 {notional:,.2f} > 上限 {cap:,.2f}；"
                f"量 {sz:g} → {new_sz:g}（原请求 {sz:g} 留痕）"
            )
        else:
            d.note(
                f"量 {sz:g} 仅超上限 {rel:.2e}（相对）——"
                f"属四舍五入噪声，夹到 {new_sz:g}；"
                f"**不计入风控干预**（resize_is_material=False）"
            )
        sz = float(new_sz)
        notional = sz * mid
        d.note(f"裁剪后名义价值 {notional:,.2f} ≤ 上限 {cap:,.2f}")
    else:
        d.note(f"名义价值 {notional:,.2f} ≤ 上限 {cap:,.2f}")

    # ---- 6. 持仓数 / 敞口（用**裁剪后**的量算）-------------------------
    opening = (action == "buy" and position_qty >= 0) or (
        action == "sell" and position_qty <= 0
    )
    if opening and n_open_positions >= lim.max_open_positions:
        d.rule = "too_many_positions"
        d.code = "TW-1008"
        d.note(f"已有 {n_open_positions} 个仓位 ≥ 上限 {lim.max_open_positions}")
        return _hold(
            f"风控拒绝：已持有 {n_open_positions} 个仓位，"
            f"达到上限 {lim.max_open_positions}"
        ), d
    if opening and equity > 0:
        exp_frac = (inst_exposure + notional) / equity
        # 用 >= 而不是 >：上限就是"不得超过"，恰好在边界上应当算超。
        # （这里与 max_open_positions 的口径一致，都是 >=。）
        if exp_frac >= lim.max_inst_exposure_frac:
            d.rule = "exposure_cap"
            d.code = "TW-1005"
            d.note(
                f"该标的敞口将达权益的 {exp_frac:.1%} ≥ "
                f"上限 {lim.max_inst_exposure_frac:.0%}"
            )
            # ⚠️ 拒单时必须清掉 resized_to：留着它会让下游把这条决策统计进
            # 「改量」并打印成 `[改量→30]`，而实际上一手都没成交。
            # 裁剪**发生过**这件事在 notes 里，不该借 resized_to 这个
            # "最终采纳了多少"的字段来表达。
            d.resized_to = None
            return _hold(
                f"风控拒绝：该标的敞口将达权益的 {exp_frac:.1%}，"
                f"达到/超出上限 {lim.max_inst_exposure_frac:.0%}"
            ), d
        d.note(f"该标的敞口占比 {exp_frac:.1%} < {lim.max_inst_exposure_frac:.0%}")

    # ---- 7. 保证金可开 --------------------------------------------------
    if opening and not can_open_ok:
        d.rule = "margin"
        d.code = can_open_reason or "TW-1005"
        d.note(f"保证金检查未通过（{can_open_reason or 'TW-1005'}）")
        d.resized_to = None   # 同 exposure_cap：拒单不留"采纳量"
        return _hold(
            f"风控拒绝：保证金不足以开仓（{can_open_reason or 'TW-1005'}）"
        ), d
    if opening:
        d.note("保证金检查通过")

    d.accepted = True
    if not d.rule:
        d.rule = "pass"
    # 只有踩到 size_cap 时才回写 sz，其余情况原样返回（保持 parsed 不变，
    # 这样"通过"与"未通过风控的输入"在留痕里可以直接对拍）。
    if d.resized_to is None:
        return dict(parsed), d
    out = dict(parsed)
    out["sz"] = sz
    out["reason"] = reason
    return out, d


# ======================================================================
# 决策频率：每个 K 线收盘决策一次
# ======================================================================
def is_decision_bar(tick: int, *, bar_ticks: int, offset: int = 0) -> bool:
    """本 tick 是不是一个"决策窗口"。

    设计方案 §9 的建议是**每根 K 线收盘决策一次**，理由：
      · 与 OKX 数据天然的粒度对齐，不用自己聚合；
      · LLM 延迟 2~6 秒，tick 级根本来不及；
      · 「收盘」是真实交易员的标准决策时点。

    ⚠️ 用 ``(tick - offset) % bar_ticks`` 而不是 ``tick % bar_ticks``：
    带 ``offset`` 才能表达"从第一个可决策 tick 起算"，
    这在**预热之后才注入 Agent** 的实验台里是必需的——
    否则同样的策略、同样的种子，预热带长度一变，决策时点整体错位，
    而错位的对比结果会被误读成"策略表现不同"。
    """
    if bar_ticks < 1:
        raise ValueError(f"bar_ticks 必须 >= 1，收到 {bar_ticks}")
    return (int(tick) - int(offset)) % int(bar_ticks) == 0
