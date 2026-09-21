"""输出解析（A3）—— 把模型的自由文本变成**结构化意图**。

为什么解析要单独成层，而不是在 agent 里写个 ``json.loads``
------------------------------------------------------------
因为实测（设计文档 §0.5，3 次调用）表明：

| 次数 | 现象 |
|---|---|
| 1 | 直接给纯 JSON ✅ |
| 2 | prompt 说"照抄价格"，它给了 **10350**（应为 103.5）— **量级错 100 倍** |
| 3 | 纯 JSON，但结论与前两次不同（1 sell / 2 hold，temperature=0.2） |

这三条各自对应解析层的一条设计：

1. **格式必须容错**（``json`` 代码块、前后废话、单引号、尾随逗号都出现过）。
2. **数值必须原样传递，不许"聪明地"修正**。
   ⚠️ 这一条是反直觉的：看到 ``10350`` 会不会想"它显然想写 103.5，帮它改一下"？
   **不能改。** 因为"帮它猜"会同时把"模型错了"这个事实抹掉——
   而"模型在数值上不可靠"正是这个系统最需要量化的东西（A4 要报的指标之一）。
   把关的责任在**风控层**（``tw/risk.py`` 的 ``max_tp_sl_pct``），
   不在解析层。解析层只做**翻译**，不做**审校**。
3. **解析失败必须留痕**（``parse_ok=False`` + ``llm_raw`` 原文 + ``parse_error``）。
   只记成功解析的那些，会把"模型输出了 40% 垃圾"这个区间从数据里删掉——
   而那恰恰是它真实的表现。

字段名映射（为什么要有这一层）
------------------------------
模型输出的是 ``size`` / ``take_profit`` / ``stop_loss`` / ``order_type`` /
``limit_price``（这是 prompt 里的 schema，好读）；
风控层（A2，已定稿并测穷）吃的是 ``sz`` / ``tp`` / ``sl`` / ``ordType`` / ``px``。
**两套名字必须在一处对齐**，并且别名要显式列出——否则
"模型写 ``qty`` 而解析层只认 ``size``"这种问题会表现为一个静默的空仓。

⚠️ 别名是**有限枚举**，不做模糊匹配（模糊匹配 = 猜意图，
而"猜"会把解析失败率这个指标弄脏）。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

#: 内部规范字段（= 风控层的输入契约，见 ``tw/risk.py``）。
CANONICAL_KEYS: tuple[str, ...] = (
    "action", "sz", "ordType", "px", "tp", "sl", "confidence", "reason",
    #: ⭐ 决策依据自报（v4 起）。**只记录，不参与风控**——
    #: 它与"下不下单"无关，风控也不看它。加进规范键是为了
    #: 让它能落进 ``parsed``、进而可被统计。
    "basis",
)

#: 必填字段——缺了算解析失败（``reason`` 不算必填：模型偶尔会把理由
#: 写在正文里而不是字段里，那种情况留痕仍完整可用，不该被算成失败）。
REQUIRED_CANONICAL: tuple[str, ...] = ("action",)

#: 别名表：``模型可能写的名字 → 内部名字``。
#: ⚠️ 只列**真实见过或显然会见到**的变体。加别名要克制——
#: 每加一个都在扩大"看起来解析成功、实则语义错位"的面。
_ALIASES: dict[str, str] = {
    # 动作
    "action": "action",
    "side": "action",
    "decision": "action",
    # 数量
    "size": "sz",
    "sz": "sz",
    "qty": "sz",
    "quantity": "sz",
    "amount": "sz",
    "position_size": "sz",
    # 订单类型
    "order_type": "ordType",
    "ordertype": "ordType",
    "ordType": "ordType",
    "type": "ordType",
    # 限价
    "limit_price": "px",
    "price": "px",
    "px": "px",
    "entry_price": "px",
    # 止盈止损
    "take_profit": "tp",
    "takeprofit": "tp",
    "tp": "tp",
    "take_profit_price": "tp",
    "stop_loss": "sl",
    "stoploss": "sl",
    "sl": "sl",
    "stop_loss_price": "sl",
    # 置信度
    "confidence": "confidence",
    "conf": "confidence",
    "conviction": "confidence",
    # 理由
    "reason": "reason",
    "reasoning": "reason",
    "rationale": "reason",
    "explanation": "reason",
    # 决策依据自报（v4）
    "basis": "basis",
    "basis_type": "basis",
    "decision_basis": "basis",
    "依据": "basis",
}

#: ``basis`` 的取值归一化。**只认这三种**——多出来的取值宁可归到"未自报"，
#: 也不要猜（猜错会把"模型没按要求填"这个信号抹掉）。
_BASIS_ALIASES: dict[str, str] = {
    "rule": "rule", "rules": "rule", "rule_based": "rule",
    "indicator": "rule", "indicators": "rule", "指标": "rule",
    "公式": "rule", "规则": "rule",
    "judgment": "judgment", "judgement": "judgment", "discretionary": "judgment",
    "intuition": "judgment", "subjective": "judgment", "discretion": "judgment",
    "判断": "judgment", "主观": "judgment",
    "both": "both", "mixed": "both", "combination": "both", "hybrid": "both",
    "两者": "both", "结合": "both",
}

#: 中文短语的**兜底**归一化：按词族匹配，且**只在恰好命中一族时**才采用。
#: ⚠️ 为什么不用朴素 substring：``"既看了指标也凭判断"`` 同时命中两族，
#: 那种情况**必须**判"未自报"而不是猜——猜错会把"模型没按要求填"
#: 这个信号抹掉，而 v4 观察的正是它。
#: （与 ``_is_leaky_key`` 同一条教训：归一化要**限定边界**，
#: 否则要么误杀、要么瞎猜。）
_BASIS_FAMILIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("rule", ("指标", "规则", "公式", "按算好的")),
    ("judgment", ("判断", "主观", "直觉", "盘感", "经验")),
    ("both", ("结合", "两者", "同时", "兼顾", "综合")),
)


def _canon_basis(raw: Any) -> str:
    """把模型写的 `basis` 归一化。认不出就返回 ``""``（未自报）。"""
    if raw is None:
        return ""
    key = str(raw).strip().lower().replace(" ", "_")
    hit = _BASIS_ALIASES.get(key)
    if hit:
        return hit
    s = str(raw)
    matched = {fam for fam, words in _BASIS_FAMILIES if any(w in s for w in words)}
    return matched.pop() if len(matched) == 1 else ""

#: 动作名归一化（模型爱写 ``close`` / ``long`` / ``flat``）。
_ACTION_ALIASES: dict[str, str] = {
    "buy": "buy", "long": "buy", "open_long": "buy", "做多": "buy",
    "sell": "sell", "short": "sell", "open_short": "sell", "做空": "sell",
    "hold": "hold", "wait": "hold", "none": "hold", "flat": "hold",
    "noop": "hold", "no_op": "hold", "abstain": "hold", "pass": "hold",
    "观望": "hold", "弃权": "hold", "不动作": "hold",
    # ⚠️ ``close`` 不映射成 buy/sell：平仓的方向取决于当前持仓，
    # 而解析层**不知道持仓**（它只吃一段文本）。硬猜会让"平多"
    # 在空仓时变成开空。⇒ 明确不识别，让它落到"未知动作"上，
    # 由留痕暴露出来。这是"宁可报错也不猜"的一处具体应用。
}


@dataclass(slots=True)
class ParseResult:
    """解析结果。**失败也是结果**（见模块文档第 3 条）。"""

    ok: bool
    parsed: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    #: 是否从代码块里抠出来的（诊断用：长期统计能看出模型行为漂移）
    extracted: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "error": self.error, "extracted": self.extracted}


# ======================================================================
# 第一步：从自由文本里抠出 JSON
# ======================================================================
_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)


def extract_json(text: str) -> tuple[Any, str, str]:
    """``(对象, 提取方式, 错误)``。

    尝试顺序（从"最可能正确"到"最宽松"）：

    1. 整段就是一个 JSON（agnes 最常见的情况）
    2. ```` ```json ... ``` ```` 代码块（其他模型常见）
    3. 第一个 ``{`` 到最后一个 ``}``（有前后废话时）
    4. 去掉尾随逗号再试（LLM 高频笔误）

    ⚠️ 第 3 步用 ``rfind`` 取**最后一个** ``}``：用第一个会截断嵌套对象。
    而 nested JSON 在"同时给多个价位"的输出里真的出现过。
    """
    s = (text or "").strip()
    if not s:
        return None, "", "输出为空"

    # 1. 整段
    try:
        return json.loads(s), "whole", ""
    except json.JSONDecodeError:
        pass

    # 2. 代码块
    m = _JSON_BLOCK.search(s)
    if m:
        inner = m.group(1).strip()
        try:
            return json.loads(inner), "fence", ""
        except json.JSONDecodeError:
            # 3/4 步也用在代码块内容上
            obj = _loose(inner)
            if obj is not None:
                return obj, "fence+loose", ""
            return None, "fence", f"代码块内容不是合法 JSON：{inner[:120]}"

    # 3. 第一个 { 到最后一个 }
    i, j = s.find("{"), s.rfind("}")
    if 0 <= i < j:
        cand = s[i:j + 1]
        try:
            return json.loads(cand), "braces", ""
        except json.JSONDecodeError:
            obj = _loose(cand)
            if obj is not None:
                return obj, "braces+loose", ""
            return None, "braces", f"花括号内的内容不是合法 JSON：{cand[:120]}"

    return None, "", "输出里找不到 JSON 对象"


_TRAILING_COMMA = re.compile(r",\s*([}\]])")


def _loose(s: str) -> Any:
    """最后的兜底：去尾随逗号、把单引号换成双引号。

    ⚠️ 单引号替换是**有风险**的（字符串内部的撇号会被误伤），
    所以它是**最后一步**——前面几步能成功就不会走到这里。
    走到这里说明输出已经很脏了，此时"尝试一下"比"直接失败"更有价值，
    但解析方式会被记进 ``extracted``，便于事后知道有多少比例是"脏解析"。
    """
    for cand in (s, _TRAILING_COMMA.sub(r"\1", s),
                 _TRAILING_COMMA.sub(r"\1", s.replace("'", '"'))):
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            continue
    return None


# ======================================================================
# 第二步：把模型字段名翻成内部规范名
# ======================================================================
def _to_float(v: Any) -> float | None:
    """尽力转 float。**转不了就返回 None**，不抛——
    让"缺少/非法"这两种情况在风控层被统一处理（那里有明确的拒单理由）。"""
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return f if f == f else None  # NaN → None
    s = str(v).strip().replace(",", "").replace("%", "").replace("USDT", "").strip()
    if not s or s.lower() in ("null", "none", "n/a", "na", "-", ""):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _canonicalize(obj: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """返回 ``(规范化的 dict, 未知字段名列表)``。

    未知字段**不丢弃、也不报错**，而是挂到 ``_unknown`` 上——
    丢弃会让"模型开始输出一个新字段"这件事完全不可见，
    而那往往是 prompt 该改的信号（或者模型行为漂移的早期迹象）。
    """
    out: dict[str, Any] = {}
    unknown: list[str] = []
    for k, v in obj.items():
        key = str(k).strip()
        canon = _ALIASES.get(key) or _ALIASES.get(key.lower())
        if canon is None:
            unknown.append(key)
            continue
        # ⚠️ 同名冲突（既给了 size 又给了 qty）：**保留先出现的**，
        # 并在 unknown 里记下冲突的那个。静默覆盖会让
        # "模型自相矛盾"这个信号消失。
        if canon in out:
            unknown.append(f"{key}(与已有 {canon} 冲突)")
            continue
        out[canon] = v
    if unknown:
        out["_unknown"] = unknown
    return out, unknown


# ======================================================================
# 主入口
# ======================================================================
def parse_decision(text: str) -> ParseResult:
    """把模型输出解析成**风控层能直接吃**的 ``parsed``。

    成功时 ``parsed`` 的键落在 :data:`CANONICAL_KEYS` 上
    （外加可能的 ``_unknown`` / ``_raw_action``）。
    ⚠️ **不做归一化之外的任何"修正"**：``10350`` 原样传下去，
    由风控层的偏离度检查拦（见模块文档第 2 条）。
    """
    obj, how, err = extract_json(text)
    if obj is None:
        return ParseResult(ok=False, parsed={}, error=err, extracted=how)
    if not isinstance(obj, dict):
        return ParseResult(
            ok=False, parsed={}, extracted=how,
            error=f"顶层不是 JSON 对象而是 {type(obj).__name__}",
        )

    parsed, unknown = _canonicalize(obj)

    # ---- 动作归一化 ---------------------------------------------------
    raw_action = parsed.get("action")
    a_txt = str(raw_action).strip().lower() if raw_action is not None else ""
    action = _ACTION_ALIASES.get(a_txt, "")
    if not action:
        if not a_txt:
            return ParseResult(
                ok=False, parsed=parsed, extracted=how,
                error="缺少 action 字段",
            )
        # ⚠️ 未知动作**不猜**（见 _ACTION_ALIASES 里 close 的注释）
        return ParseResult(
            ok=False, parsed=parsed, extracted=how,
            error=f"无法识别的 action：{raw_action!r}",
        )
    parsed["action"] = action
    if a_txt != action:
        parsed["_raw_action"] = raw_action

    # ---- 数值字段 -----------------------------------------------------
    for key in ("sz", "px", "tp", "sl"):
        if key in parsed:
            parsed[key] = _to_float(parsed[key])
    # 显式补上缺失的可选数值键：风控层的契约是"键存在但可为 None"，
    # 缺键会让风控走 `parsed.get(...)` 的默认分支，两条路径要一致。
    for key in ("sz", "px", "tp", "sl"):
        parsed.setdefault(key, None)

    # ---- 订单类型 -----------------------------------------------------
    ot = str(parsed.get("ordType") or "").strip().lower()
    if not ot:
        # 没给就按"有限价用限价，否则市价"推——这是**格式默认**，
        # 不是语义猜测（意图仍然明确）。
        ot = "limit" if parsed.get("px") is not None else "market"
    parsed["ordType"] = ot

    # ---- 置信度（夹到 [0,1]，超界要能看出来）---------------------------
    conf = _to_float(parsed.get("confidence"))
    if conf is None:
        parsed["confidence"] = None
    else:
        clamped = min(1.0, max(0.0, conf))
        if clamped != conf:
            parsed["_confidence_clamped_from"] = conf
        parsed["confidence"] = clamped

    # ---- 理由（只做类型规范，不判好坏）--------------------------------
    r = parsed.get("reason")
    parsed["reason"] = "" if r is None else str(r)

    # ---- 决策依据自报（v4）--------------------------------------------
    # ⚠️ **认不出就记** ``""``（未自报）**而不是猜一个**。
    # 猜错会把"模型没按要求填"这个信号抹掉——而那正是 v4 要观察的东西。
    # ⚠️ 缺这个字段**不算解析失败**：v1~v3 本来就没有它。
    b_raw = parsed.get("basis")
    if b_raw is None:
        parsed["basis"] = ""
    else:
        canon = _canon_basis(b_raw)
        parsed["basis"] = canon
        if not canon:
            parsed["_raw_basis"] = b_raw

    # ---- 必填校验 -----------------------------------------------------
    missing = [k for k in REQUIRED_CANONICAL if not parsed.get(k)]
    if missing:
        return ParseResult(ok=False, parsed=parsed, extracted=how,
                           error=f"缺必填字段：{missing}")

    return ParseResult(ok=True, parsed=parsed, extracted=how)


# ======================================================================
# 多采样：决策稳定性
# ======================================================================
def decision_signature(parsed: dict[str, Any]) -> str:
    """把一个决策压成一个可比较的签名（用于多采样一致性）。

    ⚠️ **不含 ``reason``**：理由是自然语言，措辞每次不同，
    含进去会让一致性永远接近 0 —— 而那衡量的是"模型会不会换词"，
    不是"决策稳不稳定"。
    ⚠️ **不含数值细节**（price 的微小差别）：一致性关心的是
    方向是否稳定（buy/sell/hold），那是"这一笔收益有多少是运气"的前提。
    """
    return f"{parsed.get('action', '?')}"


def consistency(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """多采样的一致性统计。

    返回 ``{n, n_ok, majority, majority_frac, counts}``。
    ``majority_frac`` 就是设计文档里要单独报的 ``mean_consistency``。
    """
    sigs = [decision_signature(s) for s in samples if s]
    counts: dict[str, int] = {}
    for s in sigs:
        counts[s] = counts.get(s, 0) + 1
    n = len(sigs)
    if n == 0:
        return {"n": 0, "n_ok": 0, "majority": "", "majority_frac": 0.0,
                "counts": {}}
    majority, k = max(counts.items(), key=lambda kv: (kv[1], kv[0]))
    return {"n": n, "n_ok": n, "majority": majority,
            "majority_frac": k / n, "counts": counts}


def majority_sample(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """按多数票选一条。

    ⚠️ **平票时保守**：退回 ``hold``。理由与本项目其余部分的
    "不确定时不要装作知道"一致——平票等于"模型自己也没定"，
    此时按"少数服从多数"硬选一个方向，是在把噪声当信号。
    """
    valid = [s for s in samples if s and s.get("action")]
    if not valid:
        return {"action": "hold", "sz": None, "px": None, "tp": None,
                "sl": None, "ordType": "market", "confidence": None,
                "reason": "多采样全部失败，按弃权处理"}
    st = consistency(valid)
    if st["majority_frac"] <= 0.5:
        # 平票（或不过半）：保守弃权，但把票型留在理由里
        return {"action": "hold", "sz": None, "px": None, "tp": None,
                "sl": None, "ordType": "market", "confidence": None,
                "reason": f"多采样未过半（{st['counts']}），保守弃权",
                "_vote": st["counts"]}
    # 在多数派里挑**置信度最高**的那条（同分取先出现的，保证确定性）
    pool = [s for s in valid if decision_signature(s) == st["majority"]]
    best = max(pool, key=lambda s: (
        s.get("confidence") if isinstance(s.get("confidence"), (int, float)) else -1.0,
    ))
    out = dict(best)
    out["_vote"] = st["counts"]
    return out


__all__ = [
    "CANONICAL_KEYS",
    "REQUIRED_CANONICAL",
    "ParseResult",
    "parse_decision",
    "extract_json",
    "decision_signature",
    "consistency",
    "majority_sample",
]
