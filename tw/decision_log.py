"""决策留痕层（A2）—— 用户需求的正中心。

用户原话：
    「让这个 ai 去交易、去做决策、去下单，然后**记录下单的理由、原因**，
      然后入场价格、出场价格等等的订单信息，然后用于数据分析验证」

为什么这一层必须是**独立的、可回放的结构**，而不是一堆 ``logger.info``
--------------------------------------------------------------------------
设计方案 §2 查到的这个领域公认标准里，**最小证据链有六项**：

    ① 决策时**能看到什么** ② Agent **相信什么**/引用了什么
    ③ 建议的**动作** ④ 建议**是否真的变成交易**（还是被风控拒了）
    ⑤ 什么规则**接受/拒绝/改了量** ⑥ 扣掉成本后的**结果**

这六项里，**④⑤ 是最容易被糊弄的两项**——"Agent 说要买"和"真的买了"
在日志里长得很像，而它们中间隔着风控层（改量、拒单、熔断）。
一个只记 `action` 和最终成交的实现，**无法回答"如果风控没拦，会怎样"**，
于是"风控层的贡献"永远算不出来，而**风控层的贡献恰恰是这个系统
能不能活下来的关键**。

所以本模块的核心设计是：**一条记录里 ①~⑤ 是并列的完整字段**，
并且 ``requested`` 与 ``final`` **都保留**，中间夹着 ``risk`` 决策。

回放（replay）为什么必须**逐位一致**
------------------------------------
决策留痕如果不回放，它就只是"日志"；能回放，它才是**证据**。
而回放要成立，必须做到三件事，缺一件就是假的：

1. **``decision_id`` 必须确定性**（同样输入 → 同样 ID）。
   用 ``sha256`` 而不是 uuid4：uuid4 每次不同，回放时无法核对"这是同一条决策"。
2. **``visible_state`` 必须完整**——它是回放的输入。
   如果只记一部分（比如忘了记 ``funding_rate``），回放时就必须用"当时的默认值"
   顶替，而那个默认值**很可能不是当时的值**，回放就变成了"重演一个相似的场景"
   而不是"重放这一次"。
3. **no-internet 回放模式**。原方案：LLM 在回放时必须**禁掉外部检索**，
   否则模型会读到"后来发生了什么"——那是答案泄漏，不是回测。

⚠️ 回放能验证什么、不能验证什么（**这条必须诚实**）
----------------------------------------------------
**能**：给定同一份 ``visible_state``，决策管线是否产出同样的 ``parsed``
（同一 seed / 同一温度下的确定性）。这验证的是**管线没有隐藏状态**。

**不能**：验证"当时的市场数据是对的"。如果当时喂进去的行情本身是错的，
回放会**忠实地**重现那个错误。留痕保证的是"决策过程可复核"，
不是"决策依据正确"。后者要靠数据层的 ``snapshot_id``（见 ``marketdb``）
去回答"那次决策看到的是哪一版数据"。

两条纪律
--------
1. **解析失败也要留痕**（``llm_raw`` 原文不截断）。只记成功解析的那些，
   会把"模型输出了 40% 的垃圾"这个事实从数据里抹掉——而那正是
   该区间的真实表现。
2. **弃权（hold / 不动作）也是一条决策**，也必须记录理由。
   好系统知道何时不交易；把弃权过滤掉等于只统计赢的那些。
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

#: 决策的动作。``hold`` 是**一等的动作**（弃权要评分），不是"没有动作"。
Action = Literal["buy", "sell", "hold"]

#: 留痕记录的 schema 版本。**每次改字段语义都必须 +1**，
#: 否则回放时会拿新代码去读旧记录，而字段名没变、错误无声。
SCHEMA_VERSION = 1


# ======================================================================
# 一条决策记录
# ======================================================================
@dataclass(slots=True)
class DecisionRecord:
    """一次决策的完整因果链。字段分组对齐证据链的六项。

    刻意用**扁平字段 + 嵌套小 dict**而不是深层嵌套：
    留痕要能被 ``pandas``/SQL 直接吃，深层嵌套会让"按 rule 分组统计
    风控拦截率"这种最常见的查询变得很难写。
    """

    # ---- 身份与定位（证据链 ⓪）-----------------------------------------
    decision_id: str = ""
    run_id: str = ""
    tick: int = 0
    agent_id: str = ""
    #: 墙钟时间（毫秒）。**不参与 decision_id**——它每次不同，
    #: 参与进去会让确定性 ID 失效（见模块文档）。
    wall_ms: int = 0

    # ---- ① 决策时能看到什么 --------------------------------------------
    #: 当时可见数据的哈希。用来回答"这条记录对应哪一版输入"。
    context_hash: str = ""
    visible_state: dict[str, Any] = field(default_factory=dict)
    #: 数据血缘：这次决策读的是哪个快照（见 ``marketdb.snapshot_id``）。
    #: 与 ``context_hash`` 的区别：``context_hash`` 是"喂给模型的那份投影"，
    #: ``data_snapshot`` 是"底层数据是哪一版"。两者都要，缺一不可。
    data_snapshot: str = ""

    # ---- ② Agent 相信什么 ----------------------------------------------
    prompt_template: str = ""
    model: str = ""
    model_params: dict[str, Any] = field(default_factory=dict)
    #: 工具调用记录（外部 Agent 才有；内部 Agent 为空）。
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    #: 检索到的内容（no-internet 回放时为空，且**必须显式标记是否被禁**）。
    retrieved: list[dict[str, Any]] = field(default_factory=list)
    #: 回放模式：``live``（联网）/ ``replay_nonet``（禁外部检索）。
    mode: str = "live"

    # ---- ③ 建议的动作（模型的原始输出 + 解析结果）----------------------
    #: 模型原始输出，**不截断**。解析失败时这里是唯一的证据。
    llm_raw: str = ""
    n_samples: int = 1
    #: 多次采样的解析结果（用于"决策稳定性"这个指标）。
    samples: list[dict[str, Any]] = field(default_factory=list)
    parsed: dict[str, Any] = field(default_factory=dict)
    parse_ok: bool = True
    parse_error: str = ""
    latency_ms: int = 0

    # ---- ④ 建议是否真的变成交易 ----------------------------------------
    #: 风控之前请求的量（**保留下**，见模块文档）。
    requested: dict[str, Any] = field(default_factory=dict)
    risk: dict[str, Any] = field(default_factory=dict)
    executed: bool = False
    #: 最终落成的订单（OKX 语义）。
    order: dict[str, Any] = field(default_factory=dict)
    reject_code: str = ""
    reject_msg: str = ""

    # ---- ⑥ 结果（事后回填）--------------------------------------------
    outcome: dict[str, Any] = field(default_factory=dict)
    #: 是否已经回填过结果。回填是**幂等**的（可重跑）。
    outcome_filled: bool = False

    schema: int = SCHEMA_VERSION

    # ------------------------------------------------------------------
    def compute_id(self) -> str:
        """确定性 ``decision_id``。

        ⚠️ **参与哈希的字段必须全部是"当次决策固有的"**：
        时间戳（``wall_ms``）、延迟（``latency_ms``）、
        事后回填的 ``outcome`` **都不能进**——它们每次不同，
        进来就会让"同一次决策算出的 ID 不同"，回放核对直接失效。

        这里参与的是：``run_id`` / ``tick`` / ``agent_id`` /
        ``prompt_template`` / ``model`` / ``context_hash``。
        也就是说：**同样的 run、同样的 tick、同样的主体、同样的输入、
        同样的模板与模型 ⇒ 同一条决策**。这正是回放要核对的等价关系。
        """
        payload = "|".join([
            str(self.run_id),
            str(int(self.tick)),
            str(self.agent_id),
            str(self.prompt_template),
            str(self.model),
            str(self.context_hash),
        ])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]

    def finalize(self) -> "DecisionRecord":
        """算出 ``decision_id``。若已有 ID 但算出来的不同，说明字段被改过——
        **抛错而不是覆盖**（覆盖会让"记录被事后篡改"变成静默发生的事）。"""
        cid = self.compute_id()
        if self.decision_id and self.decision_id != cid:
            raise ValueError(
                "decision_id 与当前字段不一致：记录可能被改动过。"
                f"已有 {self.decision_id!r}，重算 {cid!r}"
            )
        self.decision_id = cid
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DecisionRecord":
        """从字典还原。**未知字段要报错**（schema 变了却不升版本的信号）。"""
        known = {f for f in cls.__dataclass_fields__}
        extra = set(d) - known
        if extra:
            raise ValueError(
                f"记录含未知字段 {sorted(extra)}；"
                f"schema 版本可能已变（当前 SCHEMA_VERSION={SCHEMA_VERSION}）"
            )
        return cls(**{k: v for k, v in d.items() if k in known})

    # ------------------------------------------------------------------
    def summary_line(self) -> str:
        """一行摘要，给人工快速扫（不用于机器解析）。"""
        act = self.parsed.get("action", "?")
        reason = str(self.parsed.get("reason", ""))[:48]
        tail = ""
        if not self.executed:
            tail = f" [未执行 {self.reject_code or 'risk'}]"
        elif self.risk.get("resized_to") is not None:
            tail = f" [改量→{self.risk['resized_to']}]"
        return f"t={self.tick:<6} {act:<4} {reason}{tail}"


# ======================================================================
# 可见状态快照
# ======================================================================
#: 决策时**应当**可见的字段。用来做"点对点上下文"检查——
#: 少记一个字段，回放就会退化（见模块文档 ②）。
REQUIRED_VISIBLE_KEYS: tuple[str, ...] = ("mid", "fundamental")

#: 建议可见的字段（缺了不报错，但要能看出来缺）。
RECOMMENDED_VISIBLE_KEYS: tuple[str, ...] = (
    "spread_bp", "best_bid", "best_ask", "inventory", "cash",
    "equity", "margin_ratio", "funding_rate", "recent_closes",
)


def build_visible_state(
    *,
    mid: float | None,
    fundamental: float,
    spread_bp: float | None = None,
    best_bid: float | None = None,
    best_ask: float | None = None,
    inventory: float = 0.0,
    cash: float = 0.0,
    equity: float | None = None,
    margin_ratio: float | None = None,
    funding_rate: float = 0.0,
    recent_closes: Iterable[float] = (),
    **extra: Any,
) -> dict[str, Any]:
    """组装 ``visible_state``。

    ⭐ **这里只放"当时真的有"的东西。** 任何"事后再补进去"的字段
    （比如事后算出的收益、事后的价格）都会让回放变成答案泄漏。
    所以本函数**不接受** ``future_*`` 这类参数——从签名上就堵住。
    """
    st: dict[str, Any] = {
        "mid": float(mid) if mid is not None else None,
        "fundamental": float(fundamental),
        "inventory": float(inventory),
        "cash": float(cash),
        "funding_rate": float(funding_rate),
    }
    if spread_bp is not None:
        st["spread_bp"] = float(spread_bp)
    if best_bid is not None:
        st["best_bid"] = float(best_bid)
    if best_ask is not None:
        st["best_ask"] = float(best_ask)
    if equity is not None:
        st["equity"] = float(equity)
    if margin_ratio is not None:
        st["margin_ratio"] = float(margin_ratio)
    rc = [float(x) for x in recent_closes]
    if rc:
        st["recent_closes"] = rc
    st.update(extra)
    return st


def state_digest(state: dict[str, Any]) -> str:
    """``visible_state`` 的确定性哈希。

    用 ``sort_keys=True`` + 固定 ``separators``：字典顺序在不同 Python
    版本/不同插入顺序下可能不同，而哈希必须只取决于**内容**。
    浮点用 ``repr``（``json`` 默认的 ``float.__repr__`` 是往返精确的）。
    """
    blob = json.dumps(state, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=_json_default)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


def _json_default(o: Any):  # noqa: ANN401
    """非 JSON 原生类型的兜底。**不静默转 str** ——那会让"同一个值不同表示"
    产生不同哈希，回放核对就会失败在一个纯粹的表象差异上。"""
    import numpy as np

    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return [_json_default(x) for x in o.tolist()]
    raise TypeError(f"无法确定性序列化 {type(o).__name__}；请显式转换后再记入留痕")


def now_ms() -> int:
    return int(time.time() * 1000)


# ======================================================================
# 留痕库（JSONL 落盘）
# ======================================================================
class DecisionLog:
    """决策留痕的落盘与回读。

    格式选 **JSONL**（每行一条）而不是一个大 JSON：
      · 追加写（崩溃时已写的不丢）；
      · **流式**读（几十万条也不用全进内存）；
      · 行号 = 顺序，天然支持"第 N 条"定位。

    ⚠️ **只追加不覆盖**。与数据层的「入库后不覆盖」是同一条纪律：
    "那次决策看到什么"必须事后可答，覆盖掉就没有答案了。
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = None

    # -- 生命周期 -------------------------------------------------------
    def open(self, mode: str = "a") -> "DecisionLog":
        if self._fh is not None:
            raise RuntimeError("DecisionLog 已经打开了")
        #: 只允许追加或读取，**不允许 "w"** —— 见类文档。
        if mode not in ("a", "r"):
            raise ValueError(f"只允许 'a'（追加）或 'r'（读），收到 {mode!r}")
        self._fh = open(self.path, mode, encoding="utf-8")
        return self

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "DecisionLog":
        return self.open("a")

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- 写 -------------------------------------------------------------
    def append(self, rec: DecisionRecord) -> None:
        if self._fh is None:
            raise RuntimeError("DecisionLog 未打开；用 `with DecisionLog(p) as log:`")
        if not rec.decision_id:
            rec.finalize()
        self._fh.write(
            json.dumps(rec.to_dict(), ensure_ascii=False, default=_json_default) + "\n"
        )

    def append_many(self, recs: Iterable[DecisionRecord]) -> int:
        n = 0
        for r in recs:
            self.append(r)
            n += 1
        return n

    def flush(self) -> None:
        if self._fh is not None:
            self._fh.flush()

    # -- 读 -------------------------------------------------------------
    def read_all(self) -> list[DecisionRecord]:
        """读全部记录。

        ⚠️ 空行**跳过**（``flush`` 时序可能留下半行），但**坏行报错**——
        静默跳过坏行会让"留痕有 3% 的行损坏"这个事实完全消失。
        """
        out: list[DecisionRecord] = []
        if not self.path.exists():
            return out
        with open(self.path, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh, 1):
                s = line.strip()
                if not s:
                    continue
                try:
                    out.append(DecisionRecord.from_dict(json.loads(s)))
                except (json.JSONDecodeError, ValueError, TypeError) as exc:
                    raise ValueError(f"{self.path} 第 {i} 行解析失败: {exc}") from exc
        return out

    def iter_records(self) -> Iterable[DecisionRecord]:
        """流式读，不全进内存。"""
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                s = line.strip()
                if s:
                    yield DecisionRecord.from_dict(json.loads(s))

    def load_run(self, run_id: str) -> list[DecisionRecord]:
        return [r for r in self.read_all() if r.run_id == run_id]

    def load_tick(self, run_id: str, tick: int) -> list[DecisionRecord]:
        return [r for r in self.read_all()
                if r.run_id == run_id and int(r.tick) == int(tick)]


# ======================================================================
# ⭐ 结果回填
# ======================================================================
def fill_outcome(
    rec: DecisionRecord,
    *,
    fills: list[dict[str, Any]],
    markout: dict[str, float] | None = None,
    pnl: float | None = None,
    fee: float = 0.0,
) -> None:
    """把事后结果回填进记录。**就地改**（这是唯一允许改留痕的地方）。

    ⚠️ 回填**不能**改任何参与 ``decision_id`` 的字段——
    本函数只写 ``outcome`` / ``outcome_filled``，
    且在写之前**重新校验** ``decision_id`` 仍然自洽（防止有人手滑改了别的字段）。

    ``fills`` 的每一项是 ``{side, qty, price, mid, is_maker}``。
    入场价/出场价在 OKX 语义下**不是订单字段**，而是从这些成交里
    按持仓方向推出来的（见设计方案 §5），所以这里只记成交，
    由上层（A4 评估）去算 avgPx。
    """
    cid = rec.compute_id()
    if rec.decision_id and rec.decision_id != cid:
        raise ValueError("回填前决策记录已不自洽（字段被改过），拒绝回填")
    if not rec.decision_id:
        rec.decision_id = cid

    rec.outcome = {
        "fills": fills,
        "n_fills": len(fills),
        "filled_qty": float(sum(abs(float(f.get("qty", 0.0))) for f in fills)),
        "fill_value": float(
            sum(abs(float(f.get("qty", 0.0)) * float(f.get("price", 0.0)))
                for f in fills)
        ),
        "fee": float(fee),
        "pnl": None if pnl is None else float(pnl),
        "markout": dict(markout or {}),
    }
    rec.outcome_filled = True


# ======================================================================
# 统计：留痕本身的体检
# ======================================================================
def log_stats(recs: list[DecisionRecord]) -> dict[str, Any]:
    """留痕的体检指标。

    ⭐ 为什么一定要有这个：留痕质量**不会自动体现在收益上**。
    一个"只记成功解析"的实现跑出来收益更好看（因为垃圾输出被丢了），
    而它的证据链是坏的。所以必须**独立**报告留痕的完整性。

    这些指标都是**直接可观测量**（不是拟合出来的刻度），
    不受"分辨力"问题的困扰。
    """
    n = len(recs)
    if n == 0:
        return {"n": 0}
    n_parse_ok = sum(1 for r in recs if r.parse_ok)
    n_exec = sum(1 for r in recs if r.executed)
    n_hold = sum(1 for r in recs if r.parsed.get("action") == "hold")
    n_resized = sum(1 for r in recs if r.risk.get("resized_to") is not None)
    n_reject = sum(1 for r in recs if r.reject_code)
    n_filled = sum(1 for r in recs if r.outcome_filled)
    n_multi = sum(1 for r in recs if int(r.n_samples) > 1)
    #: 有缺字段的可见状态 —— 回放会退化的直接信号
    n_incomplete = sum(
        1 for r in recs
        if any(k not in r.visible_state for k in REQUIRED_VISIBLE_KEYS)
    )
    lat = [int(r.latency_ms) for r in recs if int(r.latency_ms) > 0]
    lat.sort()

    def pct(x: int) -> float:
        return float(x) / n

    return {
        "n": n,
        "parse_ok_frac": pct(n_parse_ok),
        "executed_frac": pct(n_exec),
        "abstain_frac": pct(n_hold),
        "resized_frac": pct(n_resized),
        "rejected_frac": pct(n_reject),
        "outcome_filled_frac": pct(n_filled),
        "multi_sample_frac": pct(n_multi),
        "incomplete_visible_frac": pct(n_incomplete),
        "latency_ms_p50": lat[len(lat) // 2] if lat else 0,
        "latency_ms_max": lat[-1] if lat else 0,
        "n_runs": len({r.run_id for r in recs}),
    }


def decision_stability(recs: list[DecisionRecord]) -> dict[str, Any]:
    """**同一决策窗口内多次采样的一致性**。

    ⭐ 为什么这是一等指标：实测 Agnes 在 temperature=0.2 下，
    三次调用给出 1 sell / 2 hold。**决策不稳定的话，「这一次的收益」
    有多少是运气就无法回答**——而这恰恰是"能不能上线"的核心问题。

    定义：对每个有多采样的决策，取 ``samples`` 里 ``action`` 的众数占比。
    一致 = 1.0；三次里两个不同 = 0.667。
    """
    ratios: list[float] = []
    for r in recs:
        acts = [str(s.get("action")) for s in r.samples if s.get("action")]
        if len(acts) < 2:
            continue
        top = max(acts.count(a) for a in set(acts))
        ratios.append(top / len(acts))
    if not ratios:
        return {"n_multi": 0, "mean_consistency": float("nan")}
    return {
        "n_multi": len(ratios),
        "mean_consistency": float(sum(ratios) / len(ratios)),
        "min_consistency": float(min(ratios)),
        "unanimous_frac": float(sum(1 for x in ratios if x >= 1.0 - 1e-9)
                                 / len(ratios)),
    }


def risk_contribution(recs: list[DecisionRecord]) -> dict[str, Any]:
    """⭐ **风控层的贡献** —— 这个数只有把 ``requested`` 保留下来才算得出。

    回答的问题：如果**没有**风控层，Agent 会做多大？
    ``size_ratio`` = 实际执行量 / 原始请求量，按笔取中位数。
    显著小于 1 说明风控在积极裁剪（可能是好事，也可能说明阈值太紧——
    要结合 ``rejected_frac`` 一起看）。
    """
    ratios: list[float] = []
    by_rule: dict[str, int] = {}
    for r in recs:
        req = float(r.requested.get("sz") or 0.0)
        fin = float(r.order.get("sz") or 0.0)
        if req > 0:
            ratios.append(fin / req)
        rule = str(r.risk.get("rule") or "")
        if rule:
            by_rule[rule] = by_rule.get(rule, 0) + 1
        if r.reject_code:
            by_rule[f"reject:{r.reject_code}"] = by_rule.get(
                f"reject:{r.reject_code}", 0) + 1
    ratios.sort()
    return {
        "n_sized": len(ratios),
        "median_size_ratio": ratios[len(ratios) // 2] if ratios else float("nan"),
        "n_fully_allowed": sum(1 for x in ratios if x >= 1.0 - 1e-9),
        "by_rule": dict(sorted(by_rule.items())),
    }
