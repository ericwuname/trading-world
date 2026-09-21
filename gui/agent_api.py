"""A5 GUI 的 Agent 数据接口 —— 把决策留痕搬进桌面端。

三条设计约定
------------
1. **不接收用户给的路径。** 所有产物都在启动时**扫描成目录**，
   前端只能按目录里的 **id** 取。用户提供路径 = 任意文件读，
   而这是本机服务（那个界面还允许跑代码）——**不该再开一个口子**。
   ⇒ 这是"宁可少一个功能，也不要一个越权接口"的一处具体应用。

2. **只读**。这个模块**不写任何文件**，也不起作业。
   要在 GUI 里跑一段，走既有的作业队列（``jobs.py``），
   复用那套进度/取消机制，不在这里另造一套。
   ⚠️ 与 MCP 那边"只读/写入/交易三级分离"是同一条思路：
   **读数据的接口不该同时有写能力**。

3. **大文件要分页。** 一条决策留痕含完整 ``llm_raw`` 与
   ``visible_state``，几百条就是几 MB。一次性返回会把浏览器拖死，
   而且**看不出慢在哪**。⇒ 列表只回摘要，正文按需取单条。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tw import eval_agent as _eval_agent
from tw.decision_log import DecisionLog

#: 扫描哪些目录找产物。**固定清单，不由用户提供。**
#: 相对项目根目录。
RUN_DIRS: tuple[str, ...] = (
    "out/a5",
    "out/a4",
    "docs/llm-run-20260921-a4",
    "docs/llm-run-20260921",
)

#: 决策留痕文件的候选名模式。
_DEC_PATTERNS: tuple[str, ...] = ("decisions_*.jsonl*", "dec_*.jsonl*")

#: 列表接口单次最多返回多少条（正文按需取）。
MAX_PAGE = 200


@dataclass(slots=True)
class RunEntry:
    """一个可浏览的运行。``id`` 是**扫描时生成的**，前端只能用它。"""

    id: str
    label: str
    source_dir: str
    dec_path: Path
    eval_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "label": self.label,
            "dir": self.source_dir,
            "has_eval": self.eval_path is not None,
        }
        if self.eval_path is not None:
            try:
                ev = json.loads(self.eval_path.read_text(encoding="utf-8"))
                d["target"] = ev.get("target", "")
                d["config"] = {
                    k: ev.get("config", {}).get(k)
                    for k in ("inst", "n", "template", "samples", "equity",
                              "lever", "slippage_bps")
                }
                # 只带"概览级"数字：前端的列表不该塞整份评估
                evs = ev.get("evals", {})
                d["net_return"] = {
                    nm: (e.get("result", {}).get("net_return"))
                    for nm, e in evs.items()
                }
                d["n_decisions"] = {
                    nm: e.get("decision", {}).get("n") for nm, e in evs.items()
                }
            except (OSError, json.JSONDecodeError):
                d["config_error"] = "评估 JSON 读不出来"
        return d


# ======================================================================
# 目录扫描（**唯一的"路径来源"**）
# ======================================================================
def _slug(rel: str) -> str:
    """把相对路径变成 id。⚠️ 只允许安全字符——
    id 会进 URL，而 URL 里带路径分隔符是越权的第一步。"""
    out = []
    for ch in rel.replace("\\", "/"):
        out.append(ch if (ch.isalnum() or ch in "-_.") else "_")
    return "".join(out)


def discover(root: Path | str) -> list[RunEntry]:
    """扫描固定目录，建出可浏览的运行清单。

    ⚠️ **只扫固定清单里的目录**（见 :data:`RUN_DIRS`），
    不接受调用方给路径。这样"用户能读到的文件"在代码里就是**穷举的**。
    """
    root = Path(root)
    entries: list[RunEntry] = []
    for rel in RUN_DIRS:
        d = root / rel
        if not d.is_dir():
            continue
        for pat in _DEC_PATTERNS:
            for f in sorted(d.glob(pat)):
                # 排除 `.jsonl.gz` 被 `*.jsonl*` 匹配两次的情况由 set 处理
                pass
        seen: set[Path] = set()
        for pat in _DEC_PATTERNS:
            for f in sorted(d.glob(pat)):
                if f in seen or not f.is_file():
                    continue
                seen.add(f)
                rid = _slug(f"{rel}/{f.name}")
                lbl = f"{rel} · {f.stem}"
                ev = _find_eval(d, f)
                entries.append(RunEntry(id=rid, label=lbl, source_dir=rel,
                                        dec_path=f, eval_path=ev))
    # 按目录顺序 + 名字排序，保证稳定
    order = {r: i for i, r in enumerate(RUN_DIRS)}
    entries.sort(key=lambda e: (order.get(e.source_dir, 99), e.label))
    return entries


def _find_eval(d: Path, dec: Path) -> Path | None:
    """给一个决策文件找配套的评估 JSON。

    命名约定：``dec_v2_BTC.jsonl`` ↔ ``eval_v2_BTC.json``；
    ``decisions_v2_BTC.jsonl`` ↔ ``eval_BTC.json``（A4 的旧命名）。
    """
    name = dec.name
    for suf in (".jsonl.gz", ".jsonl"):
        if name.endswith(suf):
            stem = name[: -len(suf)]
            break
    else:
        stem = dec.stem
    cands: list[str] = []
    if stem.startswith("dec_"):
        cands.append("eval_" + stem[4:] + ".json")
    if stem.startswith("decisions_"):
        # decisions_v2_BTC → eval_BTC.json
        parts = stem[len("decisions_"):].split("_")
        if len(parts) >= 2:
            cands.append(f"eval_{parts[-1]}.json")
    for c in cands:
        p = d / c
        if p.is_file():
            return p
    return None


# ======================================================================
# 取数据
# ======================================================================
def _entry(root: Path | str, run_id: str) -> RunEntry:
    """按 id 取条目。⚠️ 只可能命中扫描出来的那些文件。"""
    for e in discover(root):
        if e.id == run_id:
            return e
    raise KeyError(f"没有这个运行 {run_id!r}（先调 /api/agent/runs 看清单）")


def list_runs(root: Path | str) -> dict[str, Any]:
    es = discover(root)
    return {"root": str(Path(root)), "dirs": list(RUN_DIRS),
            "runs": [e.to_dict() for e in es], "n": len(es)}


def equity_from_records(recs: list, *, points: int = 480) -> dict[str, Any]:
    """从决策留痕里**取出**权益与持仓曲线。

    ⭐ 这是一个"留痕本来就有"的复利：每条记录 ①可见状态里都写了
    当时的 ``equity`` 与 ``inventory``——所以**权益路径不需要另存**。
    想画图时直接从留痕里取即可，而留痕本来就是为了回放而存的。

    ⚠️ 与 ``RunResult.equity_curve`` 的区别：那个是**逐根**的（且含
    成交前那格）；这里只有**每根决策点**的值。本项目的跑批是
    "每根决策一次"，所以两者等价；但**别处若改了决策频率，
    这个等价就不成立**——所以返回里带 ``per_bar_decision`` 标记。

    ``points`` 是下采样目标（前端画 480 个点足够，多了只是更慢）。
    """
    eq: list[float] = []
    qty: list[float] = []
    ticks: list[int] = []
    for r in recs:
        vs = r.visible_state or {}
        e = vs.get("equity")
        if not isinstance(e, (int, float)):
            continue
        eq.append(float(e))
        q = vs.get("inventory")
        qty.append(float(q) if isinstance(q, (int, float)) else 0.0)
        ticks.append(int(r.tick))
    n = len(eq)
    if n == 0:
        return {"n": 0, "ticks": [], "equity": [], "qty": [],
                "per_bar_decision": False}
    step = max(1, n // max(1, int(points)))
    idx = list(range(0, n, step))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    return {
        "n": n,
        "ticks": [ticks[i] for i in idx],
        "equity": [round(eq[i], 6) for i in idx],
        "qty": [round(qty[i], 8) for i in idx],
        "initial_equity": round(eq[0], 6),
        "final_equity": round(eq[-1], 6),
        "per_bar_decision": True,
    }


def run_summary(root: Path | str, run_id: str) -> dict[str, Any]:
    """一个运行的概览：条数、动作分布、风控分布、**权益曲线**。"""
    e = _entry(root, run_id)
    recs = DecisionLog(e.dec_path).read_all()
    acts: dict[str, int] = {}
    rules: dict[str, int] = {}
    for r in recs:
        a = str(r.parsed.get("action") or "?")
        acts[a] = acts.get(a, 0) + 1
        ru = str(r.risk.get("rule") or "?")
        rules[ru] = rules.get(ru, 0) + 1
    out: dict[str, Any] = {
        "id": e.id,
        "label": e.label,
        "dir": e.source_dir,
        "file": e.dec_path.name,
        "n": len(recs),
        "actions": acts,
        "risk_rules": rules,
        "executed": sum(1 for r in recs if r.executed),
        "parse_fail": sum(1 for r in recs if not r.parse_ok),
        "modes": sorted({str(r.mode) for r in recs}),
        "templates": sorted({str(r.prompt_template) for r in recs}),
        "models": sorted({str(r.model) for r in recs if r.model}),
        "latency_ms_p50": _p50([int(r.latency_ms) for r in recs
                                if int(r.latency_ms) > 0]),
        "tick_range": ([int(recs[0].tick), int(recs[-1].tick)]
                       if recs else None),
        "curve": equity_from_records(recs),
    }
    if e.eval_path is not None:
        try:
            out["eval"] = json.loads(e.eval_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            out["eval_error"] = "评估 JSON 读不出来"
    # ⭐ 若这批留痕带「依据自报」（v4），把分布一起算出来——
    # 这正是用户要看的：「它到底选了规则还是主观判断」。
    basis = basis_distribution(recs)
    if basis["n"]:
        out["basis"] = basis
    return out


#: v4 里模型自报的决策依据。**三分法**（见 prompts 的 v4 说明）。
#: ⚠️ 从 ``tw.eval_agent`` **转发**，不在这里另写一份。
#: 本项目的老教训：**同一个量有多份实现，就一定会分叉**——
#: 我第一版真的在两个模块各写了一遍 ``basis_distribution``，
#: 于是两边的返回字段不一样（一个多 ``n_declared``），
#: 而 GUI 用的是一个、评估报告用的是另一个。
BASIS_LABELS: dict[str, str] = _eval_agent.BASIS_LABELS


def basis_distribution(recs: list) -> dict[str, Any]:
    """统计模型**自报**的决策依据分布（转发 ``tw.eval_agent``）。"""
    return _eval_agent.basis_distribution(recs)



def _p50(xs: list[int]) -> int:
    if not xs:
        return 0
    xs = sorted(xs)
    return int(xs[len(xs) // 2])


def decisions_page(root: Path | str, run_id: str, *, offset: int = 0,
                   limit: int = 50) -> dict[str, Any]:
    """决策**摘要**列表（给时间线用）。

    ⚠️ **不返回 ``llm_raw`` 与 ``visible_state``**——它们占一条记录的
    绝大部分体积。列表里塞正文会让几百条的运行把浏览器拖死，
    而用户其实只想扫一眼"它什么时候做了什么"。正文按需取单条。
    """
    e = _entry(root, run_id)
    recs = DecisionLog(e.dec_path).read_all()
    n = len(recs)
    lim = max(1, min(int(limit), MAX_PAGE))
    off = max(0, min(int(offset), max(0, n - 1)))
    rows = []
    for r in recs[off: off + lim]:
        rows.append({
            "tick": int(r.tick),
            "id": r.decision_id[:12],
            "action": str(r.parsed.get("action") or "?"),
            "sz": r.parsed.get("sz"),
            "conf": r.parsed.get("confidence"),
            "reason": str(r.parsed.get("reason") or "")[:160],
            "executed": bool(r.executed),
            "parse_ok": bool(r.parse_ok),
            "rule": str(r.risk.get("rule") or ""),
            "reject": str(r.reject_code or ""),
            "resized_to": r.risk.get("resized_to"),
            "n_samples": int(r.n_samples),
            "latency_ms": int(r.latency_ms),
            "mode": str(r.mode),
        })
    return {"run": run_id, "total": n, "offset": off, "limit": lim, "rows": rows}


def decision_detail(root: Path | str, run_id: str, idx: int) -> dict[str, Any]:
    """**一条完整记录**——按证据链六项组织（见 ``decision_log`` 的模块文档）。

    ⚠️ 刻意把六项**分组返回**而不是直接吐原始 dict：
    留痕的字段名是给机器和 SQL 用的（扁平、好查），
    而这个接口是给人看的。分组的成本只有几十行，
    换来的是"打开一条决策就知道该看哪"。
    """
    e = _entry(root, run_id)
    recs = DecisionLog(e.dec_path).read_all()
    if not (0 <= int(idx) < len(recs)):
        raise KeyError(f"序号 {idx} 越界（该运行共 {len(recs)} 条）")
    r = recs[int(idx)]
    return {
        "run": run_id,
        "idx": int(idx),
        "total": len(recs),
        "identity": {
            "decision_id": r.decision_id,
            "run_id": r.run_id,
            "tick": int(r.tick),
            "agent_id": r.agent_id,
            "wall_ms": int(r.wall_ms),
            "schema": int(r.schema),
        },
        "chain": {
            "1_visible": r.visible_state,
            "2_belief": {
                "prompt_template": r.prompt_template,
                "model": r.model,
                "model_params": r.model_params,
                "mode": r.mode,
                "tool_calls": r.tool_calls,
                "retrieved": r.retrieved,
                "context_hash": r.context_hash,
                "data_snapshot": r.data_snapshot,
            },
            "3_suggestion": {
                "parsed": r.parsed,
                "parse_ok": bool(r.parse_ok),
                "parse_error": r.parse_error,
                "n_samples": int(r.n_samples),
                "samples": r.samples,
                "latency_ms": int(r.latency_ms),
            },
            "4_whether_traded": {
                "requested": r.requested,
                "executed": bool(r.executed),
                "reject_code": r.reject_code,
                "reject_msg": r.reject_msg,
            },
            "5_risk": r.risk,
            "6_order": r.order,
            "outcome": r.outcome,
            "outcome_filled": bool(r.outcome_filled),
        },
        "raw": {"llm_raw": r.llm_raw, "summary": r.summary_line()},
    }


def progress(root: Path | str, run_id: str) -> dict[str, Any]:
    """轻量进度：条数 + 权益曲线。

    ⚠️ 权益曲线从**成交前那格**开始（见 ``agent_run``），所以
    长度 = 决策数 + 1。前端画图时要按这个对齐，否则最后一根会少一格。
    """
    e = _entry(root, run_id)
    recs = DecisionLog(e.dec_path).read_all()
    return {"run": run_id, "n": len(recs), "label": e.label}


def first_paint(root: Path | str, run_id: str = "", *,
                page_limit: int = 200) -> dict[str, Any]:
    """一次性把**首屏需要的全部数据**打好包。

    ⭐ 为什么要这个：前端原本是"先渲染占位 → 异步取数 → 再渲染"。
    对人是"闪一下"（可接受），但它让**无头验证变得不可靠**——
    截图/DOM dump 会在 fetch 之前拍，拿到的是占位图，而检查
    只看"有没有内容"时**会通过**。本项目的老教训是
    「验证方式骗人比 bug 更危险」，所以这里改成：
    **服务端把首屏数据嵌进 HTML**，前端同步渲染 ⇒ 首屏是确定的。

    顺带的好处：深链（`?tab=agent&run=...`）打开即完整呈现，不闪。
    """
    runs = list_runs(root)
    if not runs["runs"]:
        return {"runs": [], "sel": "", "summary": None, "page": None}
    rid = run_id
    if not rid or not any(r["id"] == rid for r in runs["runs"]):
        first = next((r for r in runs["runs"] if r.get("has_eval")),
                     runs["runs"][0])
        rid = first["id"]
    return {
        "runs": runs["runs"],
        "sel": rid,
        "summary": run_summary(root, rid),
        "page": decisions_page(root, rid, offset=0, limit=page_limit),
    }


__all__ = [
    "RUN_DIRS",
    "MAX_PAGE",
    "BASIS_LABELS",
    "RunEntry",
    "discover",
    "list_runs",
    "run_summary",
    "decisions_page",
    "decision_detail",
    "equity_from_records",
    "basis_distribution",
    "first_paint",
    "progress",
]
