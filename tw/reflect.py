"""复盘与归因 + 经验库（A6 第 ②③ 块）。

用户的问题
----------
> 如果 LLM 判断错误，他会怎么分析这个错误？成功又怎么分析？
> 这些东西能不能作为经验？
> 而不是对了就对了，错了就错了，有种无所谓的感觉。

`tw/outcome.py` 解决了"它看不看得见结果"；本模块解决"看见之后它做什么"。

三个必须分开的东西
------------------
1. **结果**（``outcome``：赚没赚）
2. **决策质量**（当时的理由站不站得住）
3. **经验**（可被检索、可审计的「条件 + 教训」）

混同 1 与 2 就是"一切以结果论"。专业决策领域管这个思维错误叫
**resulting**：用结果好坏反推决策好坏。`lucky` / `unlucky` 两个分类
就是为拆开它而存在的：

- ``lucky``   结果好，但理由站不住（蒙对的）
- ``unlucky`` 理由站得住，但结果差

⭐ **安全边界是三层递进的**（缺一层就有一处漏）
------------------------------------------------
```
回填延迟（outcome.py） → 复盘只吃已实现结果（本模块 §复盘） → 经验按 t' < t 检索（本模块 §经验库）
```

- **内容层**：复盘的输入只到复盘时刻为止；``outcome`` 本来就只含已实现的。
- **时间层**：经验条目带 ``created_tick``；检索用 **严格小于**。

⚠️ 为什么检索必须是**严格** ``<`` 而不是 ``<=``：
同一根 K 线内可能发生多次决策，"本轮刚生成的经验"若在同一 tick 内
被取回，就会喂给"本轮的决策" ⇒ 微小但**真实**的泄漏。
代码里的这一行是本设计的**安全边界**，已写进 ``retrieve`` 的 docstring。

⭐ 最强的验证是 V2（时间旅行安全）
---------------------------------
同一条决策，在「经验库为空」与「经验库只有 ``t' > t`` 的条目」两种情况下，
**复盘/决策 prompt 必须逐字节相同**。
它把"会不会泄漏未来"从一个承诺变成一条可以在测试里跑通的等式。
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from .decision_log import DecisionRecord


# ======================================================================
# 归因分类（枚举，**自然语言理由无法统计**）
# ======================================================================
#: 每条决策的「结果原因」。前 5 个是"确实有错"，后 3 个是**认识论**分类。
ATTR_REASONS = (
    "direction_wrong",   # 方向看反了
    "timing_wrong",      # 方向对，但进/出场时点不好
    "size_wrong",        # 方向对，但仓位过大/过小
    "should_abstain",    # 当时该弃权却动手了
    "cost_ate_it",       # 毛赚了但成本吃掉了
    "lucky",             # ⭐ 结果好但理由站不住（蒙对的）
    "unlucky",           # ⭐ 理由站得住但结果差
    "unclear",           # 说不清
)

#: 把「结果」与「决策质量」分开的那两个分类。
RESULT_QUALITY_DECOUPLED = ("lucky", "unlucky")

EXP_KINDS = ("rule", "warning", "observation")


# ======================================================================
# 配置
# ======================================================================
@dataclass(slots=True)
class ReviewConfig:
    """复盘窗口。

    ``period`` 建议 **24 根（一天）**：太频会被噪声带着走——
    单根的结果里噪声占绝对多数，让它每 4 根复盘一次，
    学到的会主要是噪声。
    """

    period: int = 24
    #: 一次复盘最多喂多少条决策。
    #:
    #: ⚠️⚠️ **这个数字有硬约束：它决定输出长度，输出被截断就是 0 条归因。**
    #: 实测（`agnes-2.5-flash`，`max_tokens=700`）：一条归因 ≈ 90 字符，
    #: 喂 **24 条**时输出在 1365 字符处被**截断**，JSON 缺尾 ⇒ 解析全废、
    #: 16 次复盘「解析成功 0」。
    #: ⇒ 默认降到 **8 条**（≈720 字符），并把 `max_tokens` 提到 2000。
    #: **宁可多打几次，不要一次问一堆然后整批丢掉。**
    max_records: int = 8
    #: 检索多少条经验进 prompt
    max_experiences: int = 5
    #: 归因里最多保留几条（按"最该说的"排序，超出丢掉）
    max_attributions: int = 24
    #: 复盘调用的输出上限。**太小会把归因截断成 0 条**（见 `max_records`）。
    max_tokens: int = 2000

    def __post_init__(self) -> None:
        if self.period < 1:
            raise ValueError(f"period 必须 >= 1，收到 {self.period}")
        if self.max_records < 1:
            raise ValueError(f"max_records 必须 >= 1，收到 {self.max_records}")
        if self.max_tokens < 64:
            raise ValueError(f"max_tokens 必须 >= 64，收到 {self.max_tokens}")


# ======================================================================
# 经验条目
# ======================================================================
@dataclass(slots=True)
class Exp:
    """一条经验 = 「条件 + 教训」，**必须可检索、可审计**。

    ``created_tick`` 是安全边界的关键字段：检索时按它过滤。
    ``evidence_ids`` 指回支撑它的决策 ``decision_id``——**可回溯**，
    否则经验库会变成"不可证伪的传言"。
    """

    exp_id: str = ""
    created_tick: int = 0
    condition: dict[str, Any] = field(default_factory=dict)
    lesson: str = ""
    evidence_ids: list[str] = field(default_factory=list)
    kind: str = "observation"

    def __post_init__(self) -> None:
        if self.kind not in EXP_KINDS:
            raise ValueError(
                f"kind 必须是 {EXP_KINDS} 之一，收到 {self.kind!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "exp_id": self.exp_id,
            "created_tick": int(self.created_tick),
            "condition": dict(self.condition),
            "lesson": self.lesson,
            "evidence_ids": list(self.evidence_ids),
            "kind": self.kind,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Exp":
        return cls(
            exp_id=str(d.get("exp_id", "")),
            created_tick=int(d.get("created_tick", 0)),
            condition=dict(d.get("condition") or {}),
            lesson=str(d.get("lesson", "")),
            evidence_ids=list(d.get("evidence_ids") or []),
            kind=str(d.get("kind", "observation")),
        )


def make_exp_id(*, created_tick: int, lesson: str,
                evidence_ids: Iterable[str]) -> str:
    """确定性 ``exp_id``。

    用内容哈希而不是计数器/时间戳：**同样的输入永远得到同样的 ID**
    ⇒ 重跑不会产生"看起来是新的"经验，也让 V2/V4 的等式可验证。
    ⚠️ 注意**不含**任何"读取顺序"信息（那会让 ID 依赖调度）。
    """
    payload = "|".join([
        str(int(created_tick)),
        lesson,
        ",".join(sorted(str(x) for x in evidence_ids)),
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# ======================================================================
# 经验库
# ======================================================================
class ExperienceStore:
    """经验的容器。**检索的时间过滤是这里唯一容易写错的地方。**"""

    def __init__(self, items: Iterable[Exp] | None = None) -> None:
        self._items: list[Exp] = list(items or [])

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def all(self) -> list[Exp]:
        return list(self._items)

    def add(self, exp: Exp) -> None:
        if not exp.exp_id:
            raise ValueError(
                "经验必须带 exp_id——没有 ID 就无法审计它从哪来，"
                "经验库会退化成不可证伪的传言。")
        self._items.append(exp)

    def retrieve(self, *, at_tick: int, k: int = 5) -> list[Exp]:
        """取 ``at_tick`` 时刻**可以合法看到**的经验。

        ⚠️⚠️ **本函数就是本设计的安全边界，一行都不能放松**：

        - 必须是 **严格小于** ``created_tick < at_tick``。
          朴素写法 ``<=`` 在"同一 tick 内多次决策"时，会把**本轮刚生成**
          的经验喂给**本轮的决策** ⇒ 微小但真实的未来泄漏。
        - 排序必须是**确定性的**（按 ``created_tick`` 再按 ``exp_id``），
          不能用"插入顺序"或字典序以外的不稳定键——
          否则同一条决策在不同运行里会看到不同的经验，
          V2（时间旅行安全）就不可能逐字节相等。
        """
        pool = [e for e in self._items if e.created_tick < int(at_tick)]
        pool.sort(key=lambda e: (e.created_tick, e.exp_id))
        return pool[-k:] if k > 0 else []

    def to_dict(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self._items]

    @classmethod
    def from_dict(cls, items: Iterable[dict[str, Any]]) -> "ExperienceStore":
        return cls(Exp.from_dict(d) for d in (items or []))


# ======================================================================
# 机械归因（不联网也能跑，且是 LLM 归因的对照）
# ======================================================================
#: 判「价格几乎没动」的阈值（≈ 0.5bp）。低于它时亏的钱只可能是成本。
#: ⚠️ 用**带方向的** ``markout`` 的绝对值比，不是原始涨跌幅。
_COST_EPS = 5e-5


def classify_by_rules(rec: DecisionRecord) -> str:
    """**从可观测结果**推出一个归因分类——纯规则，零成本。

    为什么必须有它：
    1. 它是 LLM 归因的**对照**——若 LLM 的归因与它高度一致，
       那"归因有信息"可能只是"复述了结果"，不是真的在分析；
       若完全不一致，要问是不是在编。
    2. 它让"复盘 → 经验"整条链路**在零额度下可端到端测试**。

    ⚠️⚠️ 一个**我第一版写错的地方**，值得写下来：
    ``outcome.markout_h`` 已经**带方向符号**（``= sgn × signed_move_h``），
    所以「比较 markout 与 signed_move 的符号」是**恒假**的条件
    —— 于是 ``direction_wrong`` 永远判不出来，判据变成死代码（不报错、只是永不命中）。
    ⇒ 正确做法是**只比 ``signed_move_h``**（原始涨跌），
    再由「当时的多空方向」决定它是有利还是不利。

    判据（全部来自 ``outcome``，**不看模型给的理由**）：
    - 弃权 / 未成交 / 没回填 ⇒ ``unclear``
    - 赢了（``markout_h > 0``）：
      - ``mfe`` 远大于到手（>2 倍）⇒ ``timing_wrong``（**走早了**）
      - 否则 ⇒ ``unclear``（赢了不必硬找原因）
    - 亏了：
      - ``|markout_h|`` 小于 0.5bp ⇒ ``cost_ate_it``（价格几乎没动，亏的是成本）
      - 曾经有利（``mfe`` 明显 > 0）却回吐成亏 ⇒ ``timing_wrong``（**该走没走**）
      - 否则 ⇒ ``direction_wrong``（一开始就站错了边）
    """
    out = rec.outcome or {}
    if not out:
        return "unclear"
    action = str(out.get("action", "hold")).lower()
    if action == "hold" or not out.get("markout_applies"):
        return "unclear"

    mo = float(out.get("markout_h", 0.0))
    mfe = out.get("mfe")
    mfe_f = float(mfe) if mfe is not None else None

    if mo > 0.0:
        if mfe_f is not None and mfe_f > 2.0 * mo:
            return "timing_wrong"
        return "unclear"
    # ---- 亏了 ----
    if abs(mo) < _COST_EPS:
        return "cost_ate_it"
    if mfe_f is not None and mfe_f > _COST_EPS:
        return "timing_wrong"
    return "direction_wrong"


def attribution_association(pairs: Iterable[tuple[str, float]]) -> dict[str, Any]:
    """V3 的检验：**归因分类与可观测结果是否有关联**。

    ``pairs`` = ``(分类, 该决策的可观测结果)``（用 ``markout_h`` 之类）。

    做法：如果归因是"有信息"的，那么**「理由站得住」的那一类内部**，
    结果符号的分布应当更接近 50/50（正负都有）；而纯分类之外的
    整体均值不该由分类完全决定。
    这里用一个便宜且可机械检查的量：
    - ``n_by_reason``：各分类的条数
    - ``mean_result_by_reason``：各分类的结果均值
    - ``spread``：各分类结果均值的极差（**越大 ⇒ 分类越"有信息"**）
    - ``decoupled_share``：``lucky``/``unlucky`` 两类里**结果与分类
      期望相反**的比例——这两类存在的意义就是"结果不能解释理由"，
      所以它们的内部结果应当**混杂**，而不是清一色。

    ⚠️ 本函数只做**描述**，不下"显著"结论——
    显著性要走 :func:`tw.segmented.paired_verdict` 那套配对检验。
    拍脑袋把极差当效应量，就是纪律 6/8 里那条最贵的错误。
    """
    by: dict[str, list[float]] = {}
    for reason, val in pairs:
        by.setdefault(str(reason), []).append(float(val))
    means = {k: (sum(v) / len(v) if v else float("nan"))
             for k, v in by.items()}
    finite = [m for m in means.values() if m == m]
    spread = (max(finite) - min(finite)) if len(finite) >= 2 else float("nan")

    dec = 0
    dec_n = 0
    for reason in RESULT_QUALITY_DECOUPLED:
        for v in by.get(reason, []):
            dec_n += 1
            # lucky：结果却是负的 / unlucky：结果却是正的 ⇒ "结果与标签相反"
            if (reason == "lucky" and v < 0.0) or (reason == "unlucky" and v > 0.0):
                dec += 1
    return {
        "n_by_reason": {k: len(v) for k, v in by.items()},
        "mean_result_by_reason": means,
        "spread": spread,
        "decoupled_share": (dec / dec_n) if dec_n else float("nan"),
        "n": sum(len(v) for v in by.values()),
    }


# ======================================================================
# 复盘 prompt（内容层边界在这里落地）
# ======================================================================
_SYS = (
    "你是一个交易主体的复盘助手。你只做一件事："
    "对刚才这段窗口里的每一条决策，指出它的结果是由什么原因造成的。\n"
    "⚠️ 重要：你要区分「结果」与「决策质量」。\n"
    "  - 赚了但理由站不住，那是 lucky，不是对的决策。\n"
    "  - 亏了但理由站得住，那是 unlucky，不是错的决策。\n"
    "不要用结果反推理由的质量。\n"
    "只输出 JSON，不要解释。"
)


def _rec_brief(rec: DecisionRecord) -> dict[str, Any]:
    """把一条决策压成复盘用的小字典。

    ⚠️ **只放已实现的结果**。`outcome` 由 ``tw/outcome.py`` 回填，
    它对"未来还没发生"的决策天然是空的 ⇒ 这里不需要额外判断
    （这正是"回填延迟 = 可见边界"的好处：边界在**上游**就封好了）。
    """
    out = rec.outcome or {}
    return {
        "decision_id": rec.decision_id,
        "tick": int(rec.tick),
        "action": str(rec.parsed.get("action", "hold")),
        "filled": bool(rec.executed),
        "reason_excerpt": str(rec.parsed.get("reason", ""))[:200],
        "confidence": rec.parsed.get("confidence"),
        "outcome": {
            k: out.get(k) for k in
            ("signed_move_h", "markout_h", "mfe", "mae", "horizon")
            if k in out
        },
    }


def _exp_lines(exps: Iterable[Exp]) -> str:
    items = list(exps)
    if not items:
        return "（无）"
    return "\n".join(
        f"- [{e.exp_id}] {json.dumps(e.condition, ensure_ascii=False)}"
        f" → {e.lesson}" for e in items)


def build_review_messages(
    records: Iterable[DecisionRecord],
    *,
    at_tick: int,
    experiences: Iterable[Exp] | None = None,
) -> list[dict[str, str]]:
    """构造复盘 prompt。

    ⚠️ **``at_tick`` 只用于在正文里声明"复盘时刻"**，
    真正的数据过滤发生在**调用方**（``rec.outcome`` 是否已回填、
    ``ExperienceStore.retrieve`` 的严格小于）。本函数**不做**过滤——
    它拿到的就是"该看到的那些"。
    ⇒ 所以 V2（时间旅行安全）的等式能成立：
    经验库为空 vs 只有未来条目，调用方 ``retrieve`` 返回的都是 `[]`，
    本函数收到同样的 ``experiences`` ⇒ prompt 逐字节相同。
    """
    recs = [_rec_brief(r) for r in records]
    user = (
        f"当前时刻：tick {int(at_tick)}\n"
        f"复盘窗口内共 {len(recs)} 条决策。\n\n"
        f"# 已有经验（只含当时已存在的）\n{_exp_lines(experiences or [])}\n\n"
        f"# 决策与结果\n"
        f"{json.dumps(recs, ensure_ascii=False, indent=1)}\n\n"
        f"# 要求\n"
        f"对每条决策给出一个原因，取值必须是：\n"
        f"{', '.join(ATTR_REASONS)}\n"
        f"并给出 1~3 条可复用的经验（条件 + 教训）。\n"
        f"输出格式：\n"
        f'{{"attributions":[{{"decision_id":"...","reason":"..."}}],'
        f'"experiences":[{{"condition":{{}},"lesson":"...",'
        f'"kind":"rule|warning|observation","evidence_ids":["..."]}}]}}'
    )
    return [{"role": "system", "content": _SYS},
            {"role": "user", "content": user}]


# ======================================================================
# 复盘输出解析（容错）
# ======================================================================
_JSON_BLOCK = re.compile(r"\{.*\}", re.S)
#: 抢救用：从**被截断**的 JSON 里抠出已经完整的那几条。
_ATTR_OBJ = re.compile(
    r'\{\s*"decision_id"\s*:\s*"([^"]*)"\s*,\s*"reason"\s*:\s*"([^"]*)"\s*\}')
_EXP_OBJ = re.compile(r'\{\s*"lesson"\s*:\s*"((?:[^"\\]|\\.)*)"')


def _repair_truncated(text: str) -> tuple[list, list]:
    """从**被截断**的输出里抢救出完整的条目。

    ⚠️ 为什么需要它（实测踩到）：
    `max_tokens` 不够时，模型输出的 JSON 会在数组中间**硬截断**
    （实测 1365 字符处断在 `"reason": "direction_wrong"`）⇒
    `json.loads` 直接失败 ⇒ **整批归因变成 0 条**。
    而"少了尾巴"不等于"全都不能用"：前面那些条目是完整、可用的。

    返回 ``(attributions, experiences)``，都只含**结构完整**的条目。
    调用方必须把"发生过抢救"报出来（``repaired=True``）——
    否则"归因少了 80%"会和"模型只归因了 20%"看起来一模一样。
    """
    attrs: list[dict[str, str]] = []
    for did, reason in _ATTR_OBJ.findall(text):
        r = reason.strip()
        attrs.append({"decision_id": did,
                      "reason": r if r in ATTR_REASONS else "unclear"})
    exps: list[dict[str, Any]] = []
    for m in _EXP_OBJ.finditer(text):
        try:
            lesson = json.loads('"' + m.group(1) + '"')
        except (ValueError, TypeError):
            lesson = m.group(1)
        lesson = str(lesson).strip()
        if lesson:
            exps.append({"condition": {}, "lesson": lesson,
                         "kind": "observation", "evidence_ids": []})
    return attrs, exps


def parse_review(text: str) -> dict[str, Any]:
    """把模型的复盘输出解析成 ``{attributions, experiences, ok, error}``。

    **容错到"永不抛异常"**：解析失败时 ``ok=False`` + 尽量把原文抠出来，
    因为解析失败时**原文是唯一的证据**（与 ``tw.agent`` 的解析同一原则）。
    归因里出现未定义的 ``reason`` ⇒ 落到 ``unclear`` 并计数，
    **不静默丢弃**（丢弃会让"分类分布"看起来比实际干净）。
    """
    res: dict[str, Any] = {"attributions": [], "experiences": [],
                           "ok": False, "error": "", "n_bad_reason": 0,
                           "repaired": False, "raw": text}
    if not isinstance(text, str) or not text.strip():
        res["error"] = "空输出"
        return res
    m = _JSON_BLOCK.search(text)
    obj = None
    if m is not None:
        try:
            obj = json.loads(m.group(0))
        except (ValueError, TypeError) as exc:
            # ⭐ **抢救**：输出被截断（`max_tokens` 不够）时，前面那些条目
            # 是完整的。整批丢弃会让"少了尾巴"看起来像"什么都没说"。
            attrs, exps = _repair_truncated(text)
            if attrs or exps:
                res["attributions"] = attrs
                res["experiences"] = exps
                res["repaired"] = True
                res["ok"] = True
                res["error"] = f"输出被截断，已抢救：{exc}"
                return res
            res["error"] = f"JSON 解析失败：{exc}"
            return res
    if obj is None:
        attrs, exps = _repair_truncated(text)
        if attrs or exps:
            res["attributions"] = attrs
            res["experiences"] = exps
            res["repaired"] = True
            res["ok"] = True
            res["error"] = "没有完整 JSON，已按条目抢救"
            return res
        res["error"] = "找不到 JSON 对象"
        return res
    if not isinstance(obj, dict):
        res["error"] = "顶层不是对象"
        return res

    for a in (obj.get("attributions") or []):
        if not isinstance(a, dict):
            continue
        did = str(a.get("decision_id", ""))
        reason = str(a.get("reason", "")).strip()
        if reason not in ATTR_REASONS:
            res["n_bad_reason"] += 1
            reason = "unclear"
        res["attributions"].append({"decision_id": did, "reason": reason})

    for e in (obj.get("experiences") or []):
        if not isinstance(e, dict):
            continue
        kind = str(e.get("kind", "observation"))
        if kind not in EXP_KINDS:
            kind = "observation"
        lesson = str(e.get("lesson", "")).strip()
        if not lesson:
            continue
        res["experiences"].append({
            "condition": dict(e.get("condition") or {}),
            "lesson": lesson,
            "kind": kind,
            "evidence_ids": [str(x) for x in (e.get("evidence_ids") or [])],
        })
    res["ok"] = True
    return res


def merits_to_store(
    parsed: dict[str, Any], *, created_tick: int
) -> list[Exp]:
    """把解析出的经验变成 :class:`Exp`（补 ``exp_id`` 与 ``created_tick``）。

    ``created_tick`` 由**调用方**给：经验产生于"复盘发生的时刻"，
    而不是模型说的任何时间。**这一点必须由外部的时钟决定**，
    否则模型可以在正文里写一个更早的 tick 来绕过检索边界。
    """
    out: list[Exp] = []
    for e in (parsed.get("experiences") or []):
        lesson = str(e.get("lesson", "")).strip()
        if not lesson:
            continue
        ev = [str(x) for x in (e.get("evidence_ids") or [])]
        out.append(Exp(
            exp_id=make_exp_id(created_tick=int(created_tick),
                               lesson=lesson, evidence_ids=ev),
            created_tick=int(created_tick),
            condition=dict(e.get("condition") or {}),
            lesson=lesson,
            evidence_ids=ev,
            kind=str(e.get("kind", "observation")),
        ))
    return out


def duplicate_rate(exps: Iterable[Exp]) -> float:
    """V6 的一半：经验的**重复率**（同一条教训被反复产出 = 没学到新东西）。

    用 ``exp_id`` 的去重比例。⚠️ 注意 ``exp_id`` 含 ``created_tick``，
    所以"同一条教训在不同 tick 出现"**不算**重复——那正是它该有的样子
    （条件变了结论重申）。要测的是"同一 tick 内自己重复自己"以及
    "教训文本完全一致"的堆积，所以这里按 **lesson 文本** 统计。
    """
    items = list(exps)
    if not items:
        return float("nan")
    texts = [e.lesson for e in items]
    return 1.0 - len(set(texts)) / len(texts)


__all__ = [
    "ATTR_REASONS",
    "RESULT_QUALITY_DECOUPLED",
    "EXP_KINDS",
    "ReviewConfig",
    "Exp",
    "ExperienceStore",
    "make_exp_id",
    "classify_by_rules",
    "attribution_association",
    "build_review_messages",
    "parse_review",
    "merits_to_store",
    "duplicate_rate",
]
