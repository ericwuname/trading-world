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
import math
from dataclasses import dataclass
from typing import Any

#: 当前默认模板版本。**改模板必须加新版本**，见模块文档。
DEFAULT_TEMPLATE = "v2"

#: v4 起输出里多一个「决策依据自报」字段。
#: ⚠️ 它的作用是**让模型把自己的选择讲出来**——用户 2026-09-21 的问题
#: 就是"它会照规则走，还是按主观判断"。没有这个字段，那个问题
#: 只能靠人去读理由猜；有了它才**可统计**。
#: ⚠️ 但它是**声明，不是证据**（模型可能为了显得一致而都写上）——
#: 报告里必须这么写。
BASIS_KEYS: tuple[str, ...] = ("rule", "judgment", "both")

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


# ======================================================================
# v3 用户提示 —— ⭐「给它算好的特征」
# ======================================================================
#: 算好的特征里各项的说明。**与 ``market_features`` 的键一一对应。**
_FEATURE_ROWS: tuple[tuple[str, str], ...] = (
    ("ret_lookback", "区间净变化"),
    ("vol_pct", "区间涨跌幅标准差（每根）"),
    ("pos_pctile", "当前价在区间里的位置（0=最低，1=最高）"),
    ("ma_fast", "短均线（最近 3 根收盘均值）"),
    ("ma_slow", "长均线（全部区间收盘均值）"),
    ("ma_gap", "短均线相对长均线的偏离"),
    ("last_ret", "最近一根的涨跌幅"),
    ("up_frac", "区间里上涨根数占比"),
)

_USER_V3 = """## 你能看到什么（这是你的全部信息）

以下数据全部来自 tick ≤ {tick} 的市场。**没有**未来数据。

### 市场
- 标的：{inst_id}（{bar} K 线）　当前 tick：{tick}
- 中间价 mid：{mid}　标记价 mark：{mark}

### 已为你算好的指标（由确定性代码从**下面那串收盘价**算出，直接可用）
{features_txt}

### 原始数据：最近 {n_closes} 根收盘价（旧 → 新）
{recent_closes_txt}

### 本次数据源**不提供**的信息
{unavailable_txt}

### 你的账户
- 持仓：{position_txt}
- 现金 cash：{cash}　权益 equity：{equity}
- 保证金率 margin_ratio：{margin_ratio_txt}
- 可用杠杆上限：{max_lever}x　建议最大量：{max_size} 张

### 风控约束
- 止盈止损相对 mid 的偏离不得超过 {max_tp_sl_pct_txt}
- 单标的敞口上限：{max_inst_exposure_txt} 权益　单笔名义价值上限：{max_notional}

请给出你在 **tick {tick}** 的决策。只输出 JSON。"""


# ======================================================================
# v4 系统提示 —— 在 v1 基础上**加一个「决策依据自报」字段**
# ======================================================================
_SYSTEM_V4 = _SYSTEM_V1.replace(
    """{"action": "buy|sell|hold", "size": 数字, "order_type": "limit|market",
 "limit_price": 数字或null, "take_profit": 数字或null, "stop_loss": 数字或null,
 "confidence": 0到1的小数, "reason": "一句话中文理由"}""",
    """{"action": "buy|sell|hold", "size": 数字, "order_type": "limit|market",
 "limit_price": 数字或null, "take_profit": 数字或null, "stop_loss": 数字或null,
 "confidence": 0到1的小数, "basis": "rule|judgment|both", "reason": "一句话中文理由"}""",
).replace(
    """- reason：为什么这么做。这是给人工审计看的，要具体（提到你依据了快照里的哪几项）""",
    """- **basis：这次决策你**主要**依据了什么**。三选一，必须诚实填：
  - `rule` = 主要照**算好的指标**执行
  - `judgment` = 主要凭**自己对原始价格的主观判断**
  - `both` = 两者结合、权重相当
  这不是考核，也不会改变风控；**它只是把你的选择记下来供事后分析**。
  两种方式**都是允许的**——请按你当时真实的做法填，不要为了显得一致而都写 both。
- reason：为什么这么做。这是给人工审计看的，要具体（提到你依据了快照里的哪几项；若 basis=rule，请点出用了哪几个指标）""",
)


# ======================================================================
# v4 用户提示 —— ⭐「两条路都给它，由它自己选」
# ======================================================================
_USER_V4 = """## 你能看到什么（这是你的全部信息）

以下数据全部来自 tick ≤ {tick} 的市场。**没有**未来数据。

### 市场
- 标的：{inst_id}（{bar} K 线）　当前 tick：{tick}
- 中间价 mid：{mid}　标记价 mark：{mark}

## 你有两种决策方式，**由你自己选**（也可以结合）

### 方式 A：照**已算好的指标**执行
这些指标由确定性代码从下面的收盘价算出，你可以直接照着它们做规则式的判断：
{features_txt}

### 方式 B：凭**自己对原始价格**的判断
- 最近 {n_closes} 根收盘价（旧 → 新）：
{recent_closes_txt}

### 怎么选
两种方式**都可以**，没有哪种"更正确"：
- 如果你认为指标已经足够表达当前形势 → 用 **A**（`basis`: `rule`）
- 如果你认为指标丢掉了你想用的信息、或你有别的看法 → 用 **B**（`basis`: `judgment`）
- 如果你想同时参考两者 → 用 **A+B**（`basis`: `both`）
没有可算的指标时（样本不足），只能走 B。

### 本次数据源**不提供**的信息
{unavailable_txt}

### 你的账户
- 持仓：{position_txt}
- 现金 cash：{cash}　权益 equity：{equity}
- 保证金率 margin_ratio：{margin_ratio_txt}
- 可用杠杆上限：{max_lever}x　建议最大量：{max_size} 张

### 风控约束
- 止盈止损相对 mid 的偏离不得超过 {max_tp_sl_pct_txt}
- 单标的敞口上限：{max_inst_exposure_txt} 权益　单笔名义价值上限：{max_notional}

请给出你在 **tick {tick}** 的决策。**别忘了填 `basis`**。只输出 JSON。"""


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
    # ------------------------------------------------------------------
    # v3：⭐「给它算好的特征」（2026-09-21 用户拍板「要」）
    #
    # 触发原因（A4 的实测结论）：**LLM 输给了一行代码的 momentum**。
    # 它要做的事是"从 12 个价格里判断方向"——而那正是规则最擅长的事。
    # v3 检验的是：**如果直接把算好的指标给它，它会不会用？**
    #
    # ⚠️⚠️ **这改变了实验的性质，必须说清楚**：
    #   · v1/v2 问的是「**模型能不能自己从原始价格里看出信号**」
    #   · v3  问的是「**给出信号后，模型会不会用、用得对不对**」
    # 这是两个不同的问题。v3 赢了**不能**推断"模型有行情判断力"，
    # 只能推断"它在有现成信号时会照着做"。报告里必须分开写。
    #
    # 特征由 ``market_features()`` 用确定性代码算——**不是模型算的**。
    # 这与本项目「LLM 只做决策、不做算数」那条边界一致。
    # ------------------------------------------------------------------
    "v3": {"system": _SYSTEM_V1, "user": _USER_V3},
    # ------------------------------------------------------------------
    # v4：⭐「两条路都给它，由它自己选」+ 自报依据
    #      （2026-09-21 用户追加的设计——比我原来的 v3 更对）
    #
    # 用户的原话：
    #   「分成 2 个部分，一个是算好特征，一个是自己能不能看出来，
    #     我觉得都要，只是 LLM 选择而已，就像是人是按照既定规则执行，
    #     还是按照当时的主观判断一样。」
    #
    # ⭐ 关键差别（v3 → v4）：
    #   · v3 **我替它选了路**（给特征、并引导它用特征）
    #   · v4 **把选择权交回去**：两条路都摆在面前，明确说"由你决定"，
    #     并要求它**自报走了哪条**（`basis` 字段）。
    #
    # ⇒ v4 顺带把用户那个问题变成**可统计的**：
    #   「LLM 会像人一样选择照规则还是凭判断吗」——看 `basis` 分布。
    # ⚠️ 但 basis 是**声明不是证据**（模型可能都写 both 来显得一致），
    #   所以报告里必须配合"两类行为的实际差异"一起看。
    # ------------------------------------------------------------------
    "v4": {"system": _SYSTEM_V4, "user": _USER_V4},
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


# ======================================================================
# ⭐ v3 用：算好的特征（`market_features`）
# ======================================================================
def market_features(recent_closes: list[float]) -> dict[str, float]:
    """从收盘价序列算出一组**确定性**指标。

    ⚠️ **这是"算数"，不是"判断"**——所以它由代码做，不由模型做。
    本项目的一条边界是「LLM 只做决策、不做算数」（设计文档 §0.5
    实测过它会写出 100 倍的价格）。把算好的数喂给它，是**尊重**这条边界。

    ⚠️ **这些指标本身不是信号**：每根 K 线都算得出来，但它们能否预测
    下一根，取决于市场是否有效。本函数**不声称**它们有预测力——
    它只做算术。v3 实验检验的是"模型会不会用"，不是"指标灵不灵"。

    返回空 dict 表示样本不足（< 2 根）——调用方要能处理这种情况，
    **不能**用 0 顶替（0 是一个会参与推理的合法值）。
    """
    cs = [float(x) for x in recent_closes if x is not None]
    n = len(cs)
    if n < 2:
        return {}
    first, last = cs[0], cs[-1]
    if first <= 0:
        return {}
    rets = [(b - a) / a for a, b in zip(cs, cs[1:]) if a > 0]
    mean_ret = sum(rets) / len(rets) if rets else 0.0
    var = (sum((r - mean_ret) ** 2 for r in rets) / (len(rets) - 1)
           if len(rets) > 1 else 0.0)
    lo, hi = min(cs), max(cs)
    k = min(3, n)
    fast = sum(cs[-k:]) / k
    slow = sum(cs) / n
    return {
        "ret_lookback": (last - first) / first,
        "vol_pct": math.sqrt(max(var, 0.0)),
        "pos_pctile": ((last - lo) / (hi - lo)) if hi > lo else 0.5,
        "ma_fast": fast,
        "ma_slow": slow,
        "ma_gap": (fast - slow) / slow if slow > 0 else 0.0,
        "last_ret": (cs[-1] - cs[-2]) / cs[-2] if cs[-2] > 0 else 0.0,
        "up_frac": (sum(1 for r in rets if r > 0) / len(rets)) if rets else 0.0,
    }


def _features_text(feats: dict[str, float]) -> str:
    """把特征排成人读的几行。

    ⚠️ **算不出来时要说"算不出来"**，不能用 0 顶替——
    0 会被模型当成一个真实的观测值参与推理。
    """
    if not feats:
        return "（样本不足，本次算不出指标——请只用下面的原始收盘价判断）"
    fmt = {
        "ret_lookback": lambda v: f"{v:+.3%}",
        "vol_pct": lambda v: f"{v:.3%}",
        "pos_pctile": lambda v: f"{v:.2f}",
        "ma_fast": lambda v: f"{v:,.2f}",
        "ma_slow": lambda v: f"{v:,.2f}",
        "ma_gap": lambda v: f"{v:+.3%}",
        "last_ret": lambda v: f"{v:+.3%}",
        "up_frac": lambda v: f"{v:.0%}",
    }
    rows = []
    for key, zh in _FEATURE_ROWS:
        if key in feats:
            rows.append(f"- {zh}：{fmt[key](feats[key])}")
    return "\n".join(rows) if rows else "（无可算指标）"


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
        # ⚠️ 特征只从 ``visible`` 里的 ``recent_closes`` 算——
        # **不额外取数据**。否则留痕里的 visible_state 就不再是
        # "模型看到的全部"，回放会变成重演一个信息更少的场景。
        features_txt=_features_text(market_features(rc)),
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
    "market_features",
]
