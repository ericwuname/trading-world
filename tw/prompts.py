"""Prompt 模板（A3）—— **版本化**，且必须显式声明"你能看到什么"。

为什么模板要版本化
------------------
``DecisionRecord.prompt_template`` 是 ``decision_id`` 的组成部分。
如果模板改了但不改版本号，会发生一件非常隐蔽的事：
**同一个 ``decision_id`` 对应了两套不同的提示词**，于是
「A 版模板的成绩比 B 版好」这个结论再也算不出来——因为两条记录
在留痕里长得一样。这不是"不够规范"，是**归因失效**。

所以本模块的约定是：``TEMPLATES`` 是一个 ``版本号 → 模板`` 的字典，
**改模板必须新增一个版本**，不许原地改。旧版本留在文件里
（它的历史记录还要能读懂）。

为什么要把"你能看到什么"显式写进 prompt（而不是只发数据）
--------------------------------------------------------
这是本项目测量纪律的直接推论：**模型"能推断出什么"和"我们告诉了它什么"
是两件事**，而评估 Agent 时最危险的一类错误是——
模型***没有***某个信息，却"猜对了"，于是这个策略被记成一笔真实的成绩。

显式声明的作用是让这个错误**在留痕里可见**：
``visible_state`` 里有哪几个字段、prompt 里说了哪几项，必须一一对应。
回放时就能核对"当时它到底知道不知道资金费率"。

⚠️ 三条写 prompt 的硬约束（都是从 A2/probe 的实测里推出来的）
------------------------------------------------------------
1. **不让模型算数**。实测 agnes-2.5-flash 把 ``103.5`` 写成 ``10350``
   （第 2 次调用，见设计文档 §0.5）——差 100 倍。
   ⇒ 模板里明确要求"照抄我看到的价格"，**不给它换算的余地**；
   真正的数值把关仍然在风控层（``tw/risk.py`` 的 ``max_tp_sl_pct``）。
2. **弃权是一等动作**，模板必须给 hold 一个正面表述
   （"不交易也是一个决定，需要理由"），否则模型会把"必须给个动作"
   理解成"必须下单"，于是产生**为了有输出而交易**的噪声。
3. **只要 JSON、不要解释**。要求"只输出一个 JSON 对象"能显著降低
   解析失败的次数；但解析层仍然要容错（``tw/parse.py``），
   因为"模型一定会听话"不是可以依赖的事实。

模板里的 ``{...}`` 由 ``build_messages`` 填，**不要在别处再拼一次 prompt**
——两处拼接迟早分叉，而分叉的表现是"留痕里的 prompt 与实际发出去的不一致"。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

#: 当前默认模板版本。**改模板必须加新版本**，见模块文档。
DEFAULT_TEMPLATE = "v1"

#: 输出字段的规范名（内部用）。模板里的 JSON 示例必须与之一致，
#: 否则 parse 层要写一堆别名——别名的每一层都是"看起来对"的静默风险。
OUTPUT_KEYS: tuple[str, ...] = (
    "action", "size", "order_type", "limit_price",
    "take_profit", "stop_loss", "confidence", "reason",
)

# ======================================================================
# v1 系统提示
# ======================================================================
_SYSTEM_V1 = """你是一个加密货币永续合约的交易决策者。你的输出会被一个自动化的风控与执行系统消费。

## 你的角色边界（必须遵守）

- 你**只做决策**，不做计算。仓位大小、保证金、强平价格全部由确定性代码重算。
- 你**看不到未来**。下面的市场快照是你唯一的信息来源。不要假设、不要检索、不要引用任何未在快照中出现的信息。
- **不要自己换算单位或量级**。需要写价格时，就照抄快照里出现过的数字，不要乘以或除以任何倍数。
- 不确定时，**弃权（hold）是完全合格的决策**，不是失败。为一个含糊的形势硬给一个动作，比弃权更糟。

## 输出格式（严格遵守）

只输出**一个 JSON 对象**，不要 Markdown 代码块，不要任何前后解释文字：

{"action": "buy|sell|hold", "size": 数字, "order_type": "limit|market",
 "limit_price": 数字或null, "take_profit": 数字或null, "stop_loss": 数字或null,
 "confidence": 0到1的小数, "reason": "一句话中文理由"}

字段说明：
- action：`buy` 开多 / `sell` 开空 / `hold` 不动作
- size：合约张数。**不得超过快照里给出的"建议最大量"**
- order_type：`limit` 需要同时给 `limit_price`；`market` 时 `limit_price` 必须是 null
- take_profit / stop_loss：**必须是你实际看到的价格水平**。做多时止盈要高于当前价、止损要低于当前价；做空相反。不设就填 null
- confidence：你对这个决策的把握，0 到 1
- reason：为什么这么做。这是给人工审计看的，要具体（提到你依据了快照里的哪几项）"""


# ======================================================================
# v1 用户提示
# ======================================================================
_USER_V1 = """## 你能看到什么（这是你的全部信息）

以下数据全部来自 tick ≤ {tick} 的市场。**没有**未来数据。

### 市场
- 标的：{inst_id}（{bar} K 线）
- 当前 tick：{tick}
- 中间价 mid：{mid}
- 买一 / 卖一：{best_bid} / {best_ask}
- 相对价差：{spread_bp} bp
- 标记价 mark：{mark}
- 资金费率：{funding_rate_txt}

### 最近 {n_closes} 根收盘价（旧 → 新）
{recent_closes_txt}

### 你的账户
- 持仓：{position_txt}
- 现金 cash：{cash}
- 权益 equity：{equity}
- 保证金率 margin_ratio：{margin_ratio_txt}
- 可用杠杆上限：{max_lever}x
- 建议最大量：{max_size} 张（这是风控允许的上限，你可以更保守）

### 风控约束（你的输出会被这些规则检查）
- 止盈止损相对 mid 的偏离不得超过 {max_tp_sl_pct_txt}
- 单标的敞口上限：{max_inst_exposure_txt} 权益
- 单笔名义价值上限：{max_notional}

请给出你在 **tick {tick}** 的决策。只输出 JSON。"""


# ======================================================================
# v2 用户提示
# ======================================================================
_USER_V2 = """## 你能看到什么（这是你的全部信息）

以下数据全部来自 tick ≤ {tick} 的市场。**没有**未来数据。

### 市场
- 标的：{inst_id}（{bar} K 线）　当前 tick：{tick}
- 中间价 mid：{mid}
- 标记价 mark：{mark}
- 最近 {n_closes} 根收盘价（旧 → 新）：
{recent_closes_txt}

### 本次数据源**不提供**的信息（不要因为它缺失而拒绝决策）
{unavailable_txt}

### 你能从上面推出什么（这是你的可用依据）
- **方向**：把最近 {n_closes} 根收盘价按新旧排序，可以看出上涨/下跌/横盘的倾向
- **幅度**：相邻收盘价之差就是每根 K 线的涨跌幅
- **位置**：当前 mid 在最近 {n_closes} 根里处于高位还是低位
- **账户**：下面的持仓与权益决定了你的风险承受度

### 你的账户
- 持仓：{position_txt}
- 现金 cash：{cash}　权益 equity：{equity}
- 保证金率 margin_ratio：{margin_ratio_txt}
- 可用杠杆上限：{max_lever}x　建议最大量：{max_size} 张

### 风控约束（你的输出会被这些规则检查）
- 止盈止损相对 mid 的偏离不得超过 {max_tp_sl_pct_txt}
- 单标的敞口上限：{max_inst_exposure_txt} 权益　单笔名义价值上限：{max_notional}

请给出你在 **tick {tick}** 的决策。只输出 JSON。"""


TEMPLATES: dict[str, dict[str, str]] = {
    "v1": {"system": _SYSTEM_V1, "user": _USER_V1},
    # ------------------------------------------------------------------
    # v2：真机跑 v1 之后加的版本（**不是凭感觉改的**）
    #
    # 触发原因（实测记录，BTC-USDT-SWAP 1H，tick=119，2026-09-21）：
    #   v1 下模型的弃权理由是
    #     「价格在80,872.5但快照无买卖盘和资金费率信号，缺乏明确趋势…」
    #   ⇒ 它把注意力花在了**数据源本来就没有的东西**上
    #     （真实 OHLCV 库里确实没有盘口与资金费率），
    #     于是"缺什么"压倒了"有什么"，Agent 几乎必然弃权。
    #
    # v2 的两处改动：
    #   1. **不可得字段折叠成一行**，不再逐条占版面；
    #   2. **明确告诉它能从什么推断**（收盘价序列 → 方向/波动），
    #      这是它**真的**有的信息，此前 prompt 从没说过"你可以用这个"。
    #
    # ⚠️ 为什么新增而不是改 v1：v1 已经跑出过真实记录，
    # 原地改会让那些记录的 ``decision_id`` 对应到另一套提示词，
    # 「v1 的成绩 vs v2 的成绩」这个问题从此算不出来。
    # ------------------------------------------------------------------
    "v2": {"system": _SYSTEM_V1, "user": _USER_V2},
}


# ======================================================================
# 构造 messages
# ======================================================================
def _fmt(v: Any, digits: int = 6) -> str:
    """把数字格式化成人读的样子。``None`` 显示为 ``（无）`` 而不是 ``None``
    ——前者模型能正确理解成"没有这个信息"，后者有时会被它当成一个值。"""
    if v is None:
        return "（无）"
    if isinstance(v, bool):
        return "是" if v else "否"
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        if v != v or v in (float("inf"), float("-inf")):
            return str(v)
        # 去掉无意义的尾零，但保留足够精度。
        # ⚠️ 只有在**确实有小数点**时才 rstrip——否则 digits=0 时
        # "20,000" 会被剥成 "20,"（尾零属于整数部分，不是小数尾零）。
        s = f"{v:,.{digits}f}"
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        return s or "0"
    return str(v)


def _position_text(pos: dict[str, Any] | None) -> str:
    if not pos or not pos.get("qty"):
        return "空仓"
    side = "多头" if float(pos.get("qty", 0)) > 0 else "空头"
    qty = abs(float(pos.get("qty", 0)))
    avg = pos.get("avg_px")
    upl = pos.get("upl")
    seg = f"{side} {_fmt(qty)} 张，入场均价 {_fmt(avg)}"
    if upl is not None:
        seg += f"，未实现盈亏 {_fmt(upl)}"
    return seg


#: 数据源**本来就可能没有**的字段（真实 OHLCV 库里没有盘口、没有资金费率）。
#: ⚠️ 与"忘了传"是两件事：后者是管线 bug，前者是数据源的限制。
#: 本函数的输出明确说"这是数据源不提供"，避免模型把"没给"理解成
#: "你该去找"。v2 模板靠它把不可得信息折叠成一行。
_OPTIONAL_SOURCE_FIELDS: tuple[tuple[str, str], ...] = (
    ("spread_bp", "买卖价差"),
    ("best_bid", "买一价"),
    ("best_ask", "卖一价"),
    ("funding_rate", "资金费率"),
)


def _unavailable_text(visible: dict[str, Any]) -> str:
    """把"本次没给的字段"列成一行。

    ⚠️ 空的时候也要给一句明确的话，不能留空白——
    留空白会让模型分不清"没有"与"你没写清"。
    """
    missing = [zh for key, zh in _OPTIONAL_SOURCE_FIELDS
               if visible.get(key) is None]
    if not missing:
        return "（本次快照里都有）"
    return "、".join(missing) + "——这些取决于数据源，缺失时请改用收盘价序列推断。"


def build_messages(
    visible: dict[str, Any],
    *,
    inst_id: str,
    bar: str = "1H",
    tick: int = 0,
    template: str = DEFAULT_TEMPLATE,
    limits: Any = None,
    position: dict[str, Any] | None = None,
    max_size: float = 0.0,
    n_closes: int = 12,
) -> list[dict[str, Any]]:
    """把 ``visible_state`` 变成 ``messages``。

    ⚠️ **本函数只读 ``visible``**，不从别处补数据。
    如果这里偷偷补了一个 ``visible`` 里没有的字段（比如调用方从``行情``
    另取一个"最新价"），那么留痕里的 ``visible_state`` 就不再是
    "模型看到的全部"——回放会变成重演一个信息更少的场景。
    这条是 A2「可见状态必须完整」在 A3 侧的对应约定。

    ``limits`` 只用于**告诉模型规则**（它该知道自己被什么约束着），
    真正执行仍然在 ``tw/risk.py``。
    """
    tpl = TEMPLATES.get(template)
    if tpl is None:
        raise KeyError(
            f"未知模板 {template!r}；可选 {sorted(TEMPLATES)}。"
            f"⚠️ 改模板要**新增版本**而不是原地改（见模块文档）。"
        )

    rc = list(visible.get("recent_closes") or [])[-int(n_closes):]

    def _lim(name: str, default: Any) -> Any:
        return getattr(limits, name, default) if limits is not None else default

    max_tp_sl_pct = float(_lim("max_tp_sl_pct", 0.5))
    max_expo = float(_lim("max_inst_exposure_frac", 0.60))
    max_lever = float(_lim("max_lever", 20.0))
    max_notional = float(_lim("max_notional", 20_000.0))

    user = tpl["user"].format(
        tick=int(tick),
        inst_id=inst_id,
        bar=bar,
        mid=_fmt(visible.get("mid"), 2),
        best_bid=_fmt(visible.get("best_bid"), 2),
        best_ask=_fmt(visible.get("best_ask"), 2),
        spread_bp=_fmt(visible.get("spread_bp"), 2),
        mark=_fmt(visible.get("mark"), 2),
        funding_rate_txt=(f"{_fmt(visible.get('funding_rate'), 8)}"
                          if visible.get("funding_rate") is not None else "（无）"),
        unavailable_txt=_unavailable_text(visible),
        n_closes=len(rc),
        recent_closes_txt=("、".join(_fmt(x, 2) for x in rc) if rc else "（无）"),
        position_txt=_position_text(position),
        cash=_fmt(visible.get("cash"), 2),
        equity=_fmt(visible.get("equity"), 2),
        margin_ratio_txt=_fmt(visible.get("margin_ratio"), 4),
        max_lever=_fmt(max_lever, 2),
        max_size=_fmt(max_size, 6),
        max_tp_sl_pct_txt=f"{max_tp_sl_pct * 100:.1f}%",
        max_inst_exposure_txt=f"{max_expo * 100:.0f}%",
        max_notional=_fmt(max_notional, 0),
    )
    return [
        {"role": "system", "content": tpl["system"]},
        {"role": "user", "content": user},
    ]


def template_fingerprint(template: str = DEFAULT_TEMPLATE) -> str:
    """模板内容的指纹——用来在测试里守"改了模板却忘了加版本"。"""
    import hashlib

    t = TEMPLATES[template]
    blob = json.dumps(t, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


__all__ = [
    "DEFAULT_TEMPLATE",
    "TEMPLATES",
    "OUTPUT_KEYS",
    "build_messages",
    "template_fingerprint",
]
