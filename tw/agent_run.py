"""完整回路（A4）—— **Agent + 风控 + 执行 + 账本**，跑一整段真实行情。

为什么要有这一层
----------------
A3 的 ``run_session`` 只跑到"落成订单意图"就停了：它能回答
"Agent 想干什么"，但**不能回答"干了之后账户变成什么样"**。
A4 要出收益报告，就必须把接下去那段补上：

    决策 → 风控 → 订单 → **成交** → **记账** → 下一个决策看到新的账户

⭐ **账户状态必须回流到下一个决策。** 在 A3 的验证跑批里，
每次决策看到的都是"空仓 / equity 不变"——那不算策略运行，
只算了"决策序列"。本模块把 ``equity / position`` 真正接上，
所以它的输出才**有可能**被叫做收益。

三个必须做对的地方（都会静默出错）
----------------------------------
1. **决策与成交错开一根。** bar i 收盘做决策（可见数据到 i）⇒
   订单最早在 bar i+1 成交。少了这个错位，等于**让订单回到过去成交**，
   收益会凭空变好而没有任何报错。
2. **订单可重建校验。** 本模块**从留痕里的 ``order`` dict 重建
   ``OrderRequest``**，而不是把 Agent 内部的对象直接拿来用。
   多这一步的意义是：**留痕必须足以重建订单**——
   若重建失败，说明记录缺字段，那是一个必须立刻暴露的问题，
   而不是"以后再说"。（这也顺带保证"记录里写的"与"实际下的"是同一样东西。）
3. **权益曲线逐根记。** 最大回撤与夏普都从它算；
   只记首尾两点的"总收益"看不出中间是否爆过仓。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .account import MarginAccount
from .agent import TradingAgent, visible_from_series
from .decision_log import DecisionRecord
from .order_model import AlgoOrder, OrderRequest
from .kpi import KPIState, advance
from .simexec import BarExecutor, ExecConfig, Fill, bars_from_series


# ======================================================================
# 结果容器
# ======================================================================
@dataclass(slots=True)
class RunResult:
    """一次完整回路的产出。**评估层只吃这个**，不再碰账户对象。"""

    name: str
    run_id: str
    inst_id: str
    bar: str
    #: 初始权益（用来算收益）
    initial_equity: float
    records: list[DecisionRecord] = field(default_factory=list)
    fills: list[Fill] = field(default_factory=list)
    #: 逐根权益（**必须与敲定的 K 线逐根对齐**，不能只记首尾）
    equity_curve: list[float] = field(default_factory=list)
    #: 逐根的持仓量（带符号）
    qty_curve: list[float] = field(default_factory=list)
    #: 逐根的时间戳（画图/对齐用）
    timestamps: list[int] = field(default_factory=list)
    #: 跳过的根及原因
    skipped: list[dict[str, Any]] = field(default_factory=list)
    exec_config: dict[str, Any] = field(default_factory=dict)
    #: 策略元信息（模板版本 / 规则名 / 模型名）
    meta: dict[str, Any] = field(default_factory=dict)
    #: ⭐ 若本次是**回放**跑出来的，这里是回放覆盖率（命中/总调用）。
    #:
    #: ⚠️ 这个字段的存在理由很具体：**LLM 的 prompt 里含账户状态**
    #: （持仓、权益），而账户状态取决于成交、成交取决于成本。
    #: 所以"换个成本重放同一段行情"会让路径迅速发散、回放大面积未命中，
    #: 于是 Agent 全部弃权——**那不是"策略变保守了"，那是回放失败了**。
    #: 实测：成本 ×1 时覆盖率 100%、成本 ×2 时掉到 **0.8%**，
    #: 而账面净收益只从 −33 变成 −444，**看起来像一个成本敏感性结果**。
    #: ⇒ 没有这个字段就没法把"回放失败"与"策略行为"分开。
    replay_coverage: float | None = None
    replay_hits: int = 0
    replay_misses: int = 0

    def __len__(self) -> int:
        return len(self.records)

    @property
    def final_equity(self) -> float:
        return self.equity_curve[-1] if self.equity_curve else self.initial_equity

    def round_trips(self) -> int:
        """完整回合数（平 → 不持平 → 再平的次数）。"""
        n, prev = 0, 0.0
        for q in self.qty_curve:
            if abs(prev) <= 1e-12 and abs(q) > 1e-12:
                n += 1
            prev = q
        return n

    def bars_in_position(self) -> int:
        return sum(1 for q in self.qty_curve if abs(q) > 1e-12)


# ======================================================================
# 从留痕重建订单（**留痕完整性校验**）
# ======================================================================
def order_request_from_record(rec: DecisionRecord) -> OrderRequest | None:
    """从 ``rec.order``（OKX 字段名）重建 ``OrderRequest``。

    没有成交就不返回（``None``）。

    ⚠️ **这个函数同时是一个校验**：它跑不通，说明留痕里缺了
    重建订单所需的信息——那意味着"事后能不能复核这次下单"
    这个问题的答案是"不能"。所以它**抛错**而不是返回 None，
    除非是"本来就没有订单"。
    """
    d = rec.order
    if not d:
        return None
    algo = None
    aas = d.get("attachAlgoOrds") or []
    if aas:
        a0 = aas[0]
        algo = AlgoOrder(
            tp_trigger_px=a0.get("tpTriggerPx"),
            sl_trigger_px=a0.get("slTriggerPx"),
            tp_ord_px=float(a0.get("tpOrdPx", -1.0)),
            sl_ord_px=float(a0.get("slOrdPx", -1.0)),
            tp_trigger_px_type=a0.get("tpTriggerPxType", "mark"),
            sl_trigger_px_type=a0.get("slTriggerPxType", "mark"),
        )
    try:
        return OrderRequest(
            inst_id=str(d["instId"]),
            side=str(d["side"]),
            ord_type=str(d["ordType"]),
            sz=float(d["sz"]),
            px=(None if d.get("px") is None else float(d["px"])),
            td_mode=str(d.get("tdMode", "cash")),
            pos_side=str(d.get("posSide", "net")),
            reduce_only=bool(d.get("reduceOnly", False)),
            attach_algo=algo,
            cl_ord_id=str(d.get("clOrdId", "")),
            lever=float(d.get("lever", 1.0)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"留痕里的 order 无法重建为订单请求：{exc}。"
            f"字段={sorted(d)} ⇒ 「事后能否复核这次下单」的答案是**不能**。"
        ) from exc


# ======================================================================
# 主回路
# ======================================================================
def run_agent_session(
    agent: TradingAgent,
    series: Any,
    *,
    account: MarginAccount,
    start: int = 0,
    end: int | None = None,
    name: str = "",
    run_id: str = "",
    bar: str = "1H",
    exec_config: ExecConfig | None = None,
    lever: float = 3.0,
    data_snapshot: str = "",
    record_log: Any = None,
) -> RunResult:
    """跑一段真实行情：每根 K 线收盘决策一次（用户 2026-09-21 拍板）。

    ``record_log`` 给一个 ``DecisionLog`` 就顺手落盘（只追加）。

    ⚠️ 决策频率是**每根**：用户选的是"每根 K 线收盘决策一次"。
    本函数不做"隔 N 根"的限流——那是调用方该决定的，
    而且限流规则一变，实验就换了一个，不该藏在实现里。
    """
    cfg_exec = exec_config or ExecConfig()
    n = len(getattr(series, "close", []))
    if end is None:
        end = n - 1
    if not (0 <= start <= end < n):
        raise ValueError(f"区间非法：start={start} end={end} 长度={n}")

    inst = agent.config.inst_id
    ex = BarExecutor(account=account, config=cfg_exec, inst_id=inst, lever=lever)
    res = RunResult(
        name=name or inst, run_id=run_id, inst_id=inst, bar=bar,
        initial_equity=float(account.equity({inst: float(series.close[start])})),
        exec_config=cfg_exec.describe(),
    )
    # 起点也记一格：权益路径**必须从成交前开始**，
    # 否则第一根就成交的收益会被整根吞掉（GUI 那轮踩过同形的坑）。
    res.equity_curve.append(res.initial_equity)
    res.qty_curve.append(0.0)
    res.timestamps.append(int(series.timestamp[start]) if hasattr(series, "timestamp") else start)

    #: KPI 进度用一格列表包着 —— 循环里要就地改它，而 list 是可变的。
    #: ⚠️ 起点权益用**第一根之前的权益**（= 初始权益），保证
    #: ``return_so_far`` 从 0 开始算。
    kpi_state_box = [KPIState(
        initial_equity=res.initial_equity,
        equity=res.initial_equity,
        peak_equity=res.initial_equity,
        bars_done=0,
        bars_total=int(end) - int(start) + 1,
    )]

    for i in range(int(start), int(end) + 1):
        bar_i = bars_from_series(series, i)

        # ---- 1. 先用当根成交挂单与 TP/SL -----------------------------
        ex.on_bar(i, bar_i)
        # ---- 2. 强平检查（杠杆回测没有强平就等于没有杠杆）-------------
        ex.check_liquidation(i, bar_i[3])

        mark = float(series.close[i])
        pos = account.position(inst)
        equity = float(account.equity({inst: mark}))

        # ---- 3. 记权益（含当根）--------------------------------------
        res.equity_curve.append(equity)
        res.qty_curve.append(float(pos.qty))
        res.timestamps.append(int(series.timestamp[i])
                              if hasattr(series, "timestamp") else i)

        # ---- 4. 决策 -------------------------------------------------
        try:
            vis = visible_from_series(
                series, i, n_closes=agent.config.n_closes,
                equity=equity, cash=float(account.cash),
                inventory=float(pos.qty),
                mark=mark,
            )
        except ValueError as exc:
            res.skipped.append({"i": i, "reason": str(exc)})
            continue

        can_ok, can_why = account.can_open(
            inst_id=inst, side="buy",
            qty=max(agent._suggest_max_size(mid=mark, equity=equity), 1e-12),
            price=mark, lever=lever, marks={inst: mark},
        )
        # ---- KPI 进度（A6）-------------------------------------------
        # ⚠️ **必须"先推进、再决策"**：进度里若含当前这根的结果，
        # 那就是决策时还不知道的信息 = 答案泄漏。
        # 这里显式在 decide 之前推进，并只喂"已经发生"的量。
        kpi_state = None
        if agent.config.kpi is not None:
            kpi_state = advance(
                kpi_state_box[0], equity=equity, qty=float(pos.qty),
                turnover_delta=float(ex.fills[-1].notional) if (
                    ex.fills and ex.fills[-1].bar_index == i) else 0.0,
            )
        rec = agent.decide(
            visible=vis, tick=i, run_id=run_id,
            equity=equity, cash=float(account.cash),
            position={"qty": float(pos.qty), "avg_px": float(pos.avg_px),
                      "upl": float(pos.upl(mark))} if not pos.is_flat else None,
            position_qty=float(pos.qty),
            n_open_positions=len(account.open_positions()),
            inst_exposure=pos.notional(mark),
            can_open_ok=bool(can_ok), can_open_reason=str(can_why),
            data_snapshot=data_snapshot,
            kpi_state=kpi_state,
        )
        res.records.append(rec)
        if record_log is not None:
            record_log.append(rec)

        # ---- 5. 提交（从下一根开始才能成交）---------------------------
        req = order_request_from_record(rec)
        if req is not None:
            ex.submit(req, submit_bar=i, cl_ord_id=rec.decision_id[:28])

    res.fills = list(ex.fills)
    # ⚠️ 把回放覆盖率带出来（见 ``RunResult.replay_coverage`` 的说明）。
    _cl = getattr(agent, "client", None)
    if _cl is not None and hasattr(_cl, "coverage"):
        res.replay_coverage = float(_cl.coverage)
        res.replay_hits = int(getattr(_cl, "hits", 0))
        res.replay_misses = int(getattr(_cl, "misses", 0))
    res.meta = {
        "agent_id": agent.config.agent_id,
        "template": agent.config.template,
        "policy": getattr(agent.policy, "name", "") if agent.policy else "",
        "mode": (res.records[0].mode if res.records else ""),
        "n_samples": agent.config.n_samples,
        "lever": lever,
        "exec": ex.stats(),
    }
    # ⚠️ KPI 的最终判定必须写在 ``res.meta = {...}`` **之后**——
    # 写在前面会被那次整体赋值覆盖掉（我第一次就写错了）。
    if agent.config.kpi is not None:
        from .kpi import kpi_verdict

        st = kpi_state_box[0]
        res.meta["kpi"] = agent.config.kpi.describe()
        res.meta["kpi_final"] = kpi_verdict(agent.config.kpi, st)
        res.meta["kpi_state"] = st.to_dict(agent.config.kpi)
    return res


__all__ = ["RunResult", "run_agent_session", "order_request_from_record"]
