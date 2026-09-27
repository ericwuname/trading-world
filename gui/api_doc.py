"""A16：两份文档策略的实跑 + A14/A15/A16 验证报告摘要。

从 `gui/api.py` 拆出来的一节（原来跟主流程混在同一个 979 行的文件里）。
**为什么值得单独一个模块**：这里的逻辑与「造市场 / 跑策略 / 批量对照」
那三条主流程没有共享状态——它读的是 `market.sqlite` 里的真实 1H K 线，
用的是文档定义的策略类（`tw.policies_doc`），输出的是"按文档口径"的
年化 / 回撤 / 三阶矩。放在一起会让人误以为它可以复用市场模拟的中间结果。

⚠️ 依赖方向是**单向**的：`api_doc` 从 `api` 取工具函数（`_num`/
`_int`/`_vec`/`_decimate`/`ApiError`/`ROOT`/`KINDS`），`api` 只在
需要时 `import` 本模块。**不要在 `api_doc` 里 import `api` 的私有名
之外的东西**（会绕成环）。

`run_doc_strategy` 的登记（`@register_kind`）放在这里，`api.py` 底部
显式 `import gui.api_doc` 一行来触发它——注册表 (KINDS) 仍然是**唯一
事实源**，只是登记动作由各模块自己做。

⭐ 同理，`/api/doc/reports` 这条路由也用 `@route` 在**本模块**声明
（登记表在 `gui/routes.py`），而不是写在 `server.py` 的分支里。
"""

from __future__ import annotations

import json
import time

import numpy as np

from .api import (  # noqa: F401  (部分是为了 re-export 给测试/调用方)
    ApiError,
    KINDS,
    ROOT,
    _decimate,
    _fmt,
    _int,
    _num,
    _vec,
    register_kind,
)
from .jobs import Job
from .routes import Ctx, route

#: 文档策略只在这三个标的上实跑（与 A16 报告一致；换标的要先确认数据在库）
_DOC_INSTRUMENTS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")


def _doc_load(inst: str):
    """从 `market.sqlite` 载入 2 年 1H K 线（与 A16 同一条数据路径）。

    ⚠️ `finally: st.close()` 不能省——`load_candles` 抛异常时也要把连接还回去，
    否则连跑几次失败作业会攒下一堆没关的连接。
    """
    from tw.marketdb import MarketStore

    st = MarketStore()
    try:
        return st.load_candles("binance_csv", inst, "1H", limit=17520)
    finally:
        st.close()


def _doc_trade_stats(fills) -> dict:
    """按净头寸配对算每笔已实现盈亏（含手续费；支持空头）。

    ⚠️ 成本口径是**从 fill 自己读**（`f.fee` / `f.slippage_cost`，由
    `tw.simexec` 成交时写进去的），不是在这里另算一个 bp 数——
    另算等于造第二套口径。这里是原样搬移，没有改过算法。
    """
    pos = entry = 0.0
    trades: list[float] = []
    fees = slip = 0.0
    for f in fills:
        q = float(f.qty) * (1.0 if f.side == "buy" else -1.0)
        fees += float(f.fee)
        slip += float(getattr(f, "slippage_cost", 0.0) or 0.0)
        if pos == 0.0 or (q > 0) == (pos > 0):
            tot = abs(pos) + abs(q)
            entry = (entry * abs(pos) + f.price * abs(q)) / tot if tot else 0.0
            pos += q
        else:
            closed = min(abs(q), abs(pos))
            if closed > 1e-12:
                trades.append(closed * (f.price - entry)
                              * (1.0 if pos > 0 else -1.0) - f.fee)
            pos += q
            entry = f.price if abs(pos) > 1e-12 else entry
    n = len(trades)
    total = sum(trades)
    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]  # noqa: F841  (保留：原实现同款，见 git 记录)
    skew = float("nan")
    if n >= 3:
        mu = total / n
        m2 = sum((t - mu) ** 2 for t in trades) / n
        m3 = sum((t - mu) ** 3 for t in trades) / n
        skew = m3 / (m2 ** 1.5) if m2 > 0 else float("nan")
    return {"n_trades": n, "total_realized": total,
            "E_per_trade": total / n if n else float("nan"),
            "win_rate": len(wins) / n if n else float("nan"),
            "skew": skew, "fees": fees, "slippage": slip,
            "net_after_costs": total - fees - slip}


@register_kind("doc_strategy")
def run_doc_strategy(job: Job) -> tuple[dict, dict | None]:
    """A16：把《两份文档》的策略放进系统同一条管线实跑（连续 2 年）。"""
    from tw.account import MarginAccount, MarginConfig
    from tw.agent import AgentConfig, TradingAgent
    from tw.agent_run import run_agent_session
    from tw.policies_doc import GridPolicyDoc, RightTailPolicyDoc
    from tw.simexec import ExecConfig

    spec = job.spec
    inst = str(spec.get("instrument") or "BTCUSDT")
    name = str(spec.get("doc_strategy") or "grid_doc")
    if inst not in _DOC_INSTRUMENTS:
        raise ApiError(f"未知标的 {inst!r}；可用 {list(_DOC_INSTRUMENTS)}")
    cash = _num(spec, "cash", 100_000.0, 1_000.0, 1e9)
    k = _num(spec, "k", 0.25, 0.05, 0.6)
    n = _int(spec, "n", 10, 2, 100)
    stop_pct = _num(spec, "stop_pct", 0.02, 0.002, 0.2)
    target_pct = _num(spec, "target_pct", 0.20, 0.02, 3.0)

    job.say(f"载入 {inst} 1H（market.sqlite，2 年）…")
    series = _doc_load(inst)
    if name == "grid_doc":
        policy = GridPolicyDoc(k=k, n=n)
        label = f"网格 k=±{k:.0%} N={n}"
    elif name == "righttail_doc":
        policy = RightTailPolicyDoc(stop_pct=stop_pct, target_pct=target_pct)
        label = f"右尾 止损{stop_pct:.1%}/目标{target_pct:.0%}"
    else:
        raise ApiError(f"未知文档策略 {name!r}；可用 grid_doc / righttail_doc")

    agent = TradingAgent(policy=policy,
                         config=AgentConfig(inst_id=inst, n_closes=64,
                                            agent_id=name))
    account = MarginAccount(cash=cash, cfg=MarginConfig())
    job.say(f"{label}｜{inst}｜1x｜连续 2 年（{len(series.close):,} 根）")
    t0 = time.time()
    res = run_agent_session(agent, series, account=account, start=0,
                            end=len(series.close) - 1, name=name,
                            run_id=f"{inst}-{name}",
                            exec_config=ExecConfig(), lever=1.0)
    final = float(res.equity_curve[-1])
    years = max(len(res.equity_curve) / (24.0 * 365.0), 1e-9)
    ann = (final / cash) ** (1.0 / years) - 1.0
    peak, dd = float("-inf"), 0.0
    for x in res.equity_curve:
        peak = max(peak, x)
        if peak > 0:
            dd = max(dd, 1.0 - x / peak)
    ts = _doc_trade_stats(res.fills)
    bh = float(series.close[-1]) / float(series.close[0]) * (1 - 0.0006)
    eq = np.asarray(res.equity_curve, dtype=np.float64)
    eq_d, stride = _decimate(eq)
    result = {
        "kind": "doc_strategy", "instrument": inst, "doc_strategy": name,
        "strategy_label": label,
        "params": {"k": k, "n": n, "stop_pct": stop_pct,
                   "target_pct": target_pct},
        "final_equity": final, "total_return": final / cash - 1.0,
        "annualized": ann, "max_drawdown": dd, "years": years,
        "bh_annualized": (bh ** (1.0 / years) - 1.0),
        "n_fills": len(res.fills), "n_decisions": len(res.records),
        "wall_sec": round(time.time() - t0, 1),
        **ts,
    }
    series_out = {"equity": _vec(eq_d, 2), "strategy_stride": stride,
                  "pnl_curve": _vec(eq_d - cash, 2)}
    job.say(f"完成：{ts['n_trades']} 笔，年化 {_fmt(ann)}，"
            f"回撤 {_fmt(dd * 100, 1)}%，E/笔 {_fmt(ts['E_per_trade'], 1)} 元")
    return result, series_out


#: A14/A15/A16 三份验证报告：(标题, 结果 JSON, 报告 md)。
#:
#: ⭐ 提到**模块级**是为了让 `tests/test_gui.py` 能对照它检查
#: `gui/bundle_data.py` 的打包清单覆盖了没 —— 漏打一个文件的后果
#: 不是报错，而是「文档验证」页在**安装包里**静默失效
#: （2026-09-28 实测踩到：三份 JSON 都没打进去，开发机上一切正常）。
DOC_REPORTS: tuple[tuple[str, Path, Path], ...] = (
    ("A14 网格（左尾）", ROOT / "out" / "a14" / "grid_results.json",
     ROOT / "docs" / "A14-个人交易决策系统方案-验证报告.md"),
    ("A15 右尾（概率结构）", ROOT / "out" / "a15" / "right_tail.json",
     ROOT / "docs" / "A15-右尾策略完全手册-验证报告.md"),
    ("A16 系统实测（本 GUI）", ROOT / "out" / "a16" / "doc_strategies.json",
     ROOT / "docs" / "A16-两文档策略接入系统-实测报告.md"),
)


def doc_reports_payload() -> dict:
    """A14/A15/A16 三份验证报告的摘要（结果 JSON 现算 + 报告文件路径）。

    ⚠️ 文件不存在时**不抛异常**，而是如实回 `report_exists/json_exists`：
    打包后少带了文件，用户该看到的是一句"缺哪个"，而不是一个 500。
    """
    reports = []
    for title, jp, mp in DOC_REPORTS:
        item = {"title": title, "report_file": mp.name,
                "report_exists": mp.exists(), "json_exists": jp.exists()}
        if jp.exists():
            try:
                data = json.loads(jp.read_text(encoding="utf-8"))
                item["n_rows"] = len(data)
                if title.startswith("A16"):
                    g = [x for x in data if x.get("strategy") == "grid_doc"]
                    rr = [x for x in data if x.get("strategy") == "righttail_doc"]
                    item["summary"] = {
                        "grid_annualized": [round(x["annualized"], 4) for x in g],
                        "righttail_annualized": [round(x["annualized"], 4) for x in rr],
                        "grid_skew": [round(x["skew"], 2) for x in g],
                        "righttail_skew": [round(x["skew"], 2) for x in rr],
                        "righttail_win_rate": [round(x["win_rate"], 3) for x in rr],
                    }
                elif title.startswith("A15"):
                    item["summary"] = {"closed_form_all_match":
                        bool(data.get("closed_form", {}).get("all_match"))}
                elif title.startswith("A14"):
                    item["summary"] = {"n_configs": len(data)}
            except Exception as exc:  # noqa: BLE001
                item["error"] = str(exc)
        reports.append(item)
    return {"reports": reports}


# ======================================================================
# 路由声明
# ======================================================================
# ⭐ URL 与处理函数写在一起（见 `gui/routes.py`）。原来是
#    `server.py` 里一句 `if path == "/api/doc/reports"` ——
#    端点多了之后，"哪个 URL 归谁"就散在 server 里了。
#
# ⚠️ 这是**全局页**：与当前作业无关（A14/A15/A16 三份报告是磁盘上的
#    既有产物）。前端把它单独分了一组标签，别混进作业结果里。
@route("GET", "/api/doc/reports")
def api_doc_reports(ctx: Ctx) -> dict:
    """A14/A15/A16 三份验证报告的摘要。"""
    return doc_reports_payload()
