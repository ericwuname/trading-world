"""LLM 交易 Agent 的决策回路（A3）—— 把 A0/A1/A2 串成一条完整的因果链。

一次决策的六个动作
------------------
    ① 组装可见状态（**只取 tick ≤ t**）        → build_visible_state
    ② 构造 prompt（版本化模板）                → prompts.build_messages
    ③ 调用 LLM（N 次采样）                     → llm.LLMClient
    ④ 解析输出（容错 + 失败留痕）              → parse.parse_decision
    ⑤ 过风控（改量 / 拒单 / 放行）             → risk.check
    ⑥ 落成订单意图 + 写一条 DecisionRecord      → order_model.OrderRequest

本模块**只做串联**，不实现任何一步的细节。这是有意的：
- 每一步都能在 **没有网络、没有 key、没有市场**的情况下单独测穷；
- 换掉其中一步（换模型 / 换模板 / 换风控规则）不影响其余部分；
- 出问题时，"是模型不行还是管线有 bug"这个问题**能回答**——
  只要看留痕里 ``llm_raw`` 与 ``parsed`` 是否对得上。

⚠️ 五条必须遵守的约定（每一条都是前面几轮踩出来的）
----------------------------------------------------
1. **可见状态只取 tick ≤ t 的数据。** 这是防泄漏的第一道，也是唯一
   一道真正有效的——"嘱咐模型别用未来数据"不是防线。
   本模块提供的 :func:`visible_from_series` 在取 ``recent_closes`` 时
   **明确用切片而非"最新 N 根"**，因为后者在 t 靠前时会静默取到更晚的根。
2. **失败不能伪装成弃权。** 调用失败 / 解析失败时，
   ``parsed`` 是空 dict、``parse_ok=False``、``llm_raw`` 原文在，
   但仍然会走一遍风控并记成 hold（因为"没动作"确实是最终行为）。
   ⇒ 区分它们的字段是 ``parse_ok`` 与 ``risk.rule``，
   **不是** ``action``。用 ``action=="hold"`` 去统计"弃权率"会把
   管线故障算进策略的保守程度里。
3. **风控改量与拒单必须能被分开统计。** ``requested``（风控前）
   与 ``order``（风控后）都保留，中间是 ``risk.resized_to``。
4. **``decision_id`` 只由固有字段决定**（A2 已定）。所以同 run / 同
   tick / 同可见状态 / 同模板 ⇒ 同 ID，回放才能核对。
   本模块**不往 ID 里塞任何东西**，只是把字段填对。
5. **弃权也要有理由**。hold 是合法决策，但它必须带 ``reason``——
   没有理由的弃权无法评估（"它是谨慎还是坏了？"）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .decision_log import DecisionRecord, build_visible_state, state_digest
from .llm import LLMClient, ReplayClient
from .order_model import AlgoOrder, OrderRequest
from .parse import consistency, majority_sample, parse_decision
from .policy import validate_policy_output
from .prompts import DEFAULT_TEMPLATE, TEMPLATES, build_messages
from .risk import RiskLimits, check

#: 决策模式。``live`` = 真的联网问；``replay_nonet`` = 只从录制里取；
#: ``baseline_rule`` = 规则基线（**不问 LLM**）。
#: ⚠️ 必须**显式记进留痕**："这次决策有没有可能读到外部信息"
#: 是评估时第一个要回答的问题（设计文档 §4「回放模式默认关外部访问」）。
MODE_LIVE = "live"
MODE_REPLAY = "replay_nonet"
MODE_BASELINE = "baseline_rule"


# ======================================================================
# 可见状态的构造（point-in-time，防泄漏的第一道）
# ======================================================================
def visible_from_series(
    series: Any,
    i: int,
    *,
    n_closes: int = 12,
    mark: float | None = None,
    funding_rate: float = 0.0,
    inventory: float = 0.0,
    cash: float = 0.0,
    equity: float | None = None,
    margin_ratio: float | None = None,
    spread_bp: float | None = None,
    best_bid: float | None = None,
    best_ask: float | None = None,
    feature_shift: int = 0,
    **extra: Any,
) -> dict[str, Any]:
    """从 K 线序列 + 账户状态构造可见状态。

    ``i`` 是**已收盘**的最后一根的索引（调用方保证）。

    ⚠️ **用切片不用"最新 N 根"**（写这行的理由）：
    ``series.close[-N:]`` 取的是**序列末尾**的 N 根。如果调用方把整条
    序列传进来、而 ``i`` 停在中间（回放历史某一天），
    ``[-N:]`` 会取到**未来**的根——而且不会报错，只会让那一天的决策
    "神奇地准"。切片 ``[max(0, i-n+1) : i+1]`` 才不会。

    ⚠️ ``mid`` 取**收盘价**而不是 ``(high+low)/2``：收盘价是那一刻
    真正可成交的参考，而 high/low 是**事后才知道**的极值——
    用它们当 mid 等于把"这根 K 线内最高能到哪"提前告诉模型。
    这是一处很隐蔽的泄漏：数据全在同一根 K 线里，看起来"不过分"。
    """
    if i < 0:
        raise ValueError(f"索引必须 >= 0，收到 {i}")
    closes = list(getattr(series, "close", []))
    if i >= len(closes):
        raise ValueError(
            f"索引 {i} 超出序列长度 {len(closes)}——"
            f"调用方必须传**已收盘**的根的位置"
        )
    lo = max(0, i - int(n_closes) + 1)
    rc = [float(x) for x in closes[lo:i + 1]]
    mid = float(closes[i])
    # ⭐ **B 臂（错位指标）**：同一根数、**同一个格式化器**，
    # 只是把窗口整体往前挪 `feature_shift` 根 ⇒ 信息过期、格式不变。
    # ⚠️ 它与 `recent_closes` 一起进 visible_state：**模型确实看到了它**，
    # 所以它就该被留痕；也正因如此，回放能逐字节重建（不需要第二套口径）。
    shifted: list[float] | None = None
    if int(feature_shift) > 0:
        hi_s = i - int(feature_shift)
        lo_s = max(0, hi_s - int(n_closes) + 1)
        shifted = [float(x) for x in closes[lo_s:hi_s + 1]]
        if hi_s < 0:
            shifted = []
    return build_visible_state(
        mid=mid,
        mark=float(mark) if mark is not None else mid,
        spread_bp=spread_bp,
        best_bid=best_bid,
        best_ask=best_ask,
        inventory=inventory,
        cash=cash,
        equity=equity,
        margin_ratio=margin_ratio,
        funding_rate=funding_rate,
        recent_closes=rc,
        **({"feature_closes": shifted} if shifted is not None else {}),
        **extra,
    )


# ======================================================================
# 配置
# ======================================================================
@dataclass(slots=True)
class AgentConfig:
    """一个 Agent 的身份与执行参数。

    ``lever`` / ``td_mode`` / ``pos_side`` 属于**执行层**选择
    （不是模型能改的），所以放在配置里而不是 prompt 里。
    ⚠️ 模型只决定"方向与量"，**不决定用多大杠杆**——
    那是账户层的风险决策。把杠杆交回给模型是本项目明确拒绝的一步
    （它连价格量级都会写错 100 倍）。
    """

    agent_id: str = "llm-agent-1"
    inst_id: str = "BTC-USDT-SWAP"
    bar: str = "1H"
    template: str = DEFAULT_TEMPLATE
    #: 每个决策窗口采样几次（A2 的决策稳定性指标需要 >1）。
    n_samples: int = 1
    #: 温度。⚠️ 采样多次时**不要设 0**：温度 0 下多次采样没有信息量
    #: （同输入同输出），"一致性"会恒等于 1，看起来完美但什么都没测。
    temperature: float = 0.2
    max_tokens: int = 700
    #: 执行层参数。
    td_mode: str = "isolated"
    pos_side: str = "net"
    lever: float = 3.0
    #: 每次决策时喂给模型的历史收盘价根数。
    n_closes: int = 12
    #: 是否在风控拒绝/裁剪后**重试**一次（暂不启用，留接口）。
    retry_on_reject: bool = False
    #: ⭐ **KPI 配置**（A6 的实验变量）。``None`` = 不给 KPI（对照组）。
    #: ⚠️ 它会被写进 ``model_params``（因而进 ``decision_id``）——
    #: 换了 KPI 就是换了实验，两条记录不该算出同一个 ID。
    kpi: Any = None
    #: ⭐ **经验库**（A6 的实验变量 v7）。``None`` = 不给经验（对照组）。
    #:
    #: ⚠️⚠️ 它必须是 ``tw.reflect.ExperienceStore`` 而不是一个 `list`：
    #: 取经验**必须按 `created_tick < 当前 tick` 严格过滤**，
    #: 而"当前 tick"只有 ``decide()`` 知道。
    #: 让调用方预先过滤是不可靠的——调用方可能忘了，也可能用了 `<=`。
    #: ⇒ 把过滤放进 `ExperienceStore.retrieve`（**唯一**的实现），
    #: 由 ``decide()`` 每根调用一次。
    exp_store: Any = None
    #: 每次决策最多取几条经验进 prompt（默认 5；与 `ReviewConfig` 的默认一致）。
    n_experiences: int = 5
    #: ⭐ **三臂消融的变量**（A6 §6）：``real`` / ``shifted`` / ``none``。
    #: - ``real``   = 用当前窗口算指标（A 臂）
    #: - ``shifted``= 用更早一段窗口算（B 臂）——格式长度相同、信息过期
    #: - ``none``   = 不提供指标（C 臂）
    #: ⇒ ``A−B`` = 指标的信息价值；``B−C`` = 纯引导效应。**两个问题要分开。**
    features_mode: str = "real"
    #: ``features_mode="shifted"`` 时窗口往前挪几根（默认 12 = 一整段 `n_closes`）。
    features_shift: int = 0

    def __post_init__(self) -> None:
        if self.n_samples < 1:
            raise ValueError(f"n_samples 必须 >= 1，收到 {self.n_samples}")
        if self.features_mode not in ("real", "shifted", "none"):
            raise ValueError(
                f"features_mode 只能是 real/shifted/none，收到 "
                f"{self.features_mode!r}")
        if self.features_mode == "shifted" and self.features_shift <= 0:
            raise ValueError(
                "features_mode='shifted' 时必须给 features_shift > 0"
                "——否则 B 臂与 A 臂**完全一样**，消融会静默变成空转。")
        if self.n_samples > 1 and self.temperature <= 0.0:
            raise ValueError(
                "n_samples > 1 时 temperature 不能为 0："
                "温度 0 下多次采样是同输入同输出，一致性恒为 1，"
                "看起来完美但什么都没测到"
            )


# ======================================================================
# Agent
# ======================================================================
@dataclass(slots=True)
class TradingAgent:
    """把 LLM 接进交易回路的那个东西。

    刻意做成**无状态**的：``decide`` 吃"这一 tick 的全部输入"、吐"一条记录"。
    状态（持仓、账户、日志）全部在调用方手里。
    理由与本项目其余部分一致——**无状态的东西才能被冻结、被回放、
    被在两个不同时间点跑出同样的结果**。
    """

    client: LLMClient | None = None
    config: AgentConfig = field(default_factory=AgentConfig)
    limits: RiskLimits = field(default_factory=RiskLimits)
    #: ⭐ **规则基线模式**（A4）：给了它就不问 LLM——
    #: 跳过 ②prompt ③采样 ④解析 三步，直接从可见状态算出一个意图，
    #: 而 ⑤风控 ⑥订单 与留痕**完全走同一条路**。
    #:
    #: 这是"同信息基线"能成立的关键：如果基线走的是另一条管线，
    #: 「LLM 赢了基线」就可能只是「两条管线的差异」，不是模型的贡献。
    policy: Any = None

    def __post_init__(self) -> None:
        if self.client is None and self.policy is None:
            raise ValueError("必须给 client（LLM）或 policy（规则基线）之一")
        if self.client is not None and self.policy is not None:
            # ⚠️ 两个都给会让"这次是谁决定的"变得不确定——
            # 而留痕里 mode 只有一个字段，无法表达"混合"。
            # 宁可报错，不要一个说不清的 mode。
            raise ValueError("client 与 policy 只能给一个（否则 mode 无法表达）")

    # -- 内部：一次采样 -------------------------------------------------
    def _sample(self, messages: list[dict[str, Any]]) -> tuple[list[str], list[dict[str, Any]], int]:
        """跑 ``n_samples`` 次。返回 ``(原文列表, 解析结果列表, 总延迟ms)``。

        ⚠️ 采样的顺序必须**确定性**（同步串行，不用线程池）：
        并发会让"第 i 次采样对应哪条响应"取决于调度，而
        ``samples`` 要进留痕、要能被回放核对。
        """
        raws: list[str] = []
        parsed_list: list[dict[str, Any]] = []
        total_ms = 0
        for _ in range(self.config.n_samples):
            resp = self.client.chat(
                messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            total_ms += int(resp.latency_ms)
            raws.append(resp.text)
            pr = parse_decision(resp.text)
            parsed_list.append(pr.parsed if pr.ok else {})
        return raws, parsed_list, total_ms

    # -- 主入口 ---------------------------------------------------------
    def decide(
        self,
        *,
        visible: dict[str, Any],
        tick: int,
        run_id: str = "",
        equity: float = 0.0,
        cash: float = 0.0,
        position: dict[str, Any] | None = None,
        position_qty: float = 0.0,
        n_open_positions: int = 0,
        inst_exposure: float = 0.0,
        can_open_ok: bool = True,
        can_open_reason: str = "",
        data_snapshot: str = "",
        wall_ms: int = 0,
        kpi_state: Any = None,
    ) -> DecisionRecord:
        """跑一次完整决策，返回**已 finalize** 的记录（不落盘，由调用方决定）。

        ``equity`` / ``position`` 等账户信息由调用方算好传进来——
        ⚠️ 本模块**不依赖 ``MarginAccount`` 对象**，所以能在没有账户、
        没有市场的环境下把全部分支测穷（与 ``risk.check`` 同一条理由）。
        """
        cfg = self.config
        mid = visible.get("mid")
        if not isinstance(mid, (int, float)) or not math.isfinite(float(mid)):
            raise ValueError(
                f"visible_state 里的 mid 非法（{mid!r}）——"
                f"没有可用的中间价时**不该发起决策**，"
                f"而应由调用方跳过这一 tick（否则会记下一条无意义的记录）"
            )
        mid = float(mid)

        # ---- 建议最大量（给模型"合规的天花板"；基线也用同一个）--------
        max_size = self._suggest_max_size(mid=mid, equity=equity)
        context_hash = state_digest(visible)

        # ---- 分支 A：规则基线（跳过 prompt / 采样 / 解析）--------------
        if self.policy is not None:
            probe = self.policy.decide(visible, max_size=max_size)
            validate_policy_output(probe)
            parsed = dict(probe)
            parsed_list = [parsed]
            raws = ["（规则基线：无 LLM 调用）"]
            latency_ms = 0
            n_ok = 1
            vote = consistency(parsed_list)
            parse_ok = True
            parse_error = ""
            messages = []
        else:
            # ---- ② 构造 prompt ---------------------------------------
            #: 经验段的占位符。用它判断**这个模板到底会不会渲染经验段**——
            #: 比"有没有传 exp_store"准确（v7 即使没传 store 也会渲染"为空"那句）。
            _EXP_SLOT = "{experiences_txt}"
            _tpl_user = str(TEMPLATES.get(cfg.template, {}).get("user", ""))
            # ⚠️ **在 decide 内部取经验**（不是让调用方传一个 list）：
            # 过滤要用**当前 tick**，而只有这里才知道它。
            # `retrieve` 用严格 `<`（同 tick 生成的经验本轮不可见）。
            _exps = (cfg.exp_store.retrieve(at_tick=int(tick),
                                            k=cfg.n_experiences)
                     if cfg.exp_store is not None else [])
            messages = build_messages(
                visible,
                inst_id=cfg.inst_id,
                bar=cfg.bar,
                tick=tick,
                template=cfg.template,
                limits=self.limits,
                position=position,
                max_size=max_size,
                n_closes=cfg.n_closes,
                kpi=cfg.kpi,
                kpi_state=kpi_state,
                experiences=_exps,
                feature_closes=visible.get("feature_closes"),
                features_mode=cfg.features_mode,
            )
            # ---- ③ 采样 ---------------------------------------------
            raws, parsed_list, latency_ms = self._sample(messages)
            # ---- ④ 解析：取"多数派"或"第一条" -----------------------
            # ⚠️ ``parse_ok`` 的判据是「**有没有任何一次采样解析成功**」，
            # 而且必须**看采样列表，不能看最终 ``parsed``**。
            # 两个坑（都真实踩到过）：
            #
            #   坑 1：只看 ``parsed_list[0]`` ⇒ 第 1 次失败、后 2 次成功时
            #         被记成失败，而且 ``parse_error`` 是空串
            #         （一条"失败但没说为什么"的记录）。
            #   坑 2：看最终 ``parsed`` 里的 action ⇒ ``majority_sample``
            #         在全部失败时会**造一条 hold 兜底**，那条兜底的
            #         action 也是 "hold"，于是"全失败"被读成"解析成功"。
            #         造一个看起来合法的兜底值，正是本项目最警惕的
            #         "把故障伪装成正常"的形态。
            n_ok = sum(1 for p in parsed_list if p.get("action"))
            vote = consistency(parsed_list)
            if n_ok == 0:
                # 全失败 ⇒ ``parsed`` 必须保持**空**。
                # 不能把 ``majority_sample`` 造的兜底 hold 写进去：
                # 那会让"这条记录有没有意图"这个判断依赖别的字段，
                # 而空 dict 才是诚实的状态（模型什么都没给出来）。
                parsed = {}
                parse_error = (f"{len(parsed_list)} 次采样全部解析失败"
                               f"（原文见 llm_raw）")
            elif cfg.n_samples > 1:
                parsed = majority_sample(parsed_list)
                parse_error = ("" if n_ok == len(parsed_list) else
                               f"部分采样解析失败：{len(parsed_list)} 次中 "
                               f"{len(parsed_list) - n_ok} 次无有效结果"
                               f"（本次用的是成功那些的多数派）")
            else:
                parsed = dict(parsed_list[0])
                parse_error = ""
            parse_ok = n_ok > 0
        # ⚠️ 规则基线的 ``parse_ok`` 恒为 True 且 ``llm_raw`` 写明
        # "无 LLM 调用"——**不能留空**。留空会让"这条记录的模型输出
        # 去哪了"变成一个问题，而答案其实是"根本没有模型"。

        # ---- ⑤ 风控 ---------------------------------------------------
        requested = dict(parsed)          # 风控**之前**的意图，必须保留
        final, risk_d = check(
            parsed if parsed else {"action": "hold", "reason": "解析失败"},
            mid=mid,
            equity=equity,
            limits=self.limits,
            position_qty=position_qty,
            n_open_positions=n_open_positions,
            inst_exposure=inst_exposure,
            can_open_ok=can_open_ok,
            can_open_reason=can_open_reason,
        )
        risk_out = risk_d.to_dict()
        # ⚠️ **必须把风控后的最终意图也记下来**：
        # ``parsed`` 是"模型建议"，而风控可能把它改成了 hold。
        # 不记 final，就会出现一个荒谬的空缺——留痕里能看到
        # "被拒了"或"被裁了"，却看不到"最后到底是什么"。
        risk_out["final_intent"] = dict(final)
        if not parse_ok:
            # ⚠️ 不解析失败时风控会返回一条 `rule="hold"` 的判定，
            # 那条 note 写的是"Agent 选择弃权"——对一个解析失败而言
            # 这是**错的**（Agent 什么都没选，是我们没读懂它）。
            # 把区别显式写进 notes：否则用 ``rule=="hold"`` 统计弃权率
            # 会把管线故障算成"Agent 很保守"。
            risk_out["notes"] = [
                *risk_out.get("notes", []),
                "⚠️ 本次 hold 是**解析失败**导致的，不是 Agent 主动弃权"
                "（判据用 parse_ok，不要用 action）",
            ]

        # ---- ⑥ 落成订单意图 -------------------------------------------
        order_d: dict[str, Any] = {}
        reject_code, reject_msg = "", ""
        executed = False
        action = str(final.get("action") or "hold").lower()
        if risk_d.accepted and action in ("buy", "sell"):
            try:
                req = self._build_order(final, tick=tick)
                order_d = _order_to_dict(req)
                executed = True
            except (ValueError, TypeError) as exc:
                # 订单构造失败 = 风控放行但执行层接不住。
                # ⚠️ 这**不该**被算成"Agent 弃权"——它是管线缺陷，
                # 用 reject_code 标记出来，好让它在统计里单独可见。
                reject_code = "TW-9001"
                reject_msg = f"订单构造失败：{exc}"
                executed = False

        # ---- 写记录 ---------------------------------------------------
        if self.policy is not None:
            mode = MODE_BASELINE
            prompt_label = f"rule:{getattr(self.policy, 'name', 'policy')}"
            model_name = ""
            n_samples_eff = 1
        else:
            mode = (MODE_REPLAY if isinstance(self.client, ReplayClient)
                    else MODE_LIVE)
            # ⚠️⚠️ **`prompt_template` 必须唯一标识"实际发出去的那份 prompt"**。
            # 它进 `decision_id`（A2 的公式），而 KPI 会改变 prompt 正文
            # ⇒ 若标签还只是 "v5"，那么
            #    「v5 + 有 KPI」与「v5 + 无 KPI」两条**不同的**决策
            #    会算出**同一个** decision_id（同样的 run/tick/可见状态）。
            # 那不是"ID 不美观"，是**回放核对与 A/B 归因同时失效**：
            # 留痕里两条记录长得一样，事后分不清哪条是哪个实验。
            # ⇒ 把 KPI 的指纹拼进标签（改 KPI 配置 = 换实验 = 换 ID）。
            prompt_label = cfg.template
            if cfg.kpi is not None:
                import hashlib as _h
                import json as _j
                _blob = _j.dumps(cfg.kpi.describe(), sort_keys=True,
                                 ensure_ascii=False)
                prompt_label = (f"{cfg.template}#kpi"
                                f"{_h.sha256(_blob.encode('utf-8')).hexdigest()[:6]}")
            if _EXP_SLOT in _tpl_user:
                # ⚠️⚠️ **这里第一版写成只拼一个 `#exp` 存在性标记，是错的。**
                # 后果：**空经验库**与**有 1 条可见经验**在同一 run/tick/可见状态
                # 下算出**同一个 decision_id**——而两者的 prompt **内容不同**。
                # ⇒ 留痕里两条"长得一样"，A/B 归因与回放核对同时失效。
                # 这正是 KPI 那条教训（"`prompt_template` 必须唯一标识
                # **实际发出去的** prompt"）的第一次复现。
                #
                # ⭐ 现在两条规矩：
                #   ① 触发的判据是**模板里有没有那个占位符**
                #      （不是"有没有传 store"）⇒ prompt 相同必然 ID 相同；
                #   ② 拼的是**实际渲染出来的经验文本的哈希**
                #      ⇒ prompt 不同必然 ID 不同。
                import hashlib as _h
                from .reflect import exp_lines as _el
                _sig = _h.sha256(_el(_exps).encode("utf-8")).hexdigest()[:6]
                prompt_label = f"{prompt_label}#exp{_sig}"
            if cfg.features_mode != "real" or cfg.features_shift:
                # ⚠️ 同 KPI/经验的道理：**换了臂 = 换了实验** ⇒ 必须换 ID，
                # 否则「A 臂」与「B 臂」在同一 run/tick 下算出同一个 decision_id，
                # 而两者的 prompt 不同 ⇒ 留痕里两条"长得一样"、归因失效。
                prompt_label = (f"{prompt_label}#feat{cfg.features_mode}"
                                f"{cfg.features_shift}")
            # 模型名从**客户端配置**现取，不在配置里存第二份——
            # 两处存同一个东西，迟早会不一致，而留痕里"到底用了哪个模型"
            # 正是 A/B 归因的依据。
            model_name = str(getattr(getattr(self.client, "config", None),
                                     "model", "") or "")
            n_samples_eff = cfg.n_samples
        rec = DecisionRecord(
            run_id=run_id,
            tick=int(tick),
            agent_id=cfg.agent_id,
            wall_ms=int(wall_ms),
            context_hash=context_hash,
            visible_state=dict(visible),
            data_snapshot=data_snapshot,
            # ⚠️ 规则基线把"用了哪条规则"写进 prompt_template 这个字段：
            # 它是 `decision_id` 的组成部分，所以换了规则就会换 ID——
            # 这正是我们要的（不同规则是不同的实验）。
            prompt_template=prompt_label,
            model=model_name,
            model_params={
                "temperature": cfg.temperature,
                "max_tokens": cfg.max_tokens,
                "n_samples": n_samples_eff,
                # ⚠️ KPI 是实验变量 ⇒ 必须进 model_params（进而进 decision_id），
                # 否则"有 KPI"与"无 KPI"两条决策会算出同一个 ID。
                "kpi": (cfg.kpi.describe() if cfg.kpi is not None else None),
                # ⚠️ 经验库同样是实验变量 ⇒ 进 model_params（因而进 decision_id）
                "exp_store_size": (len(cfg.exp_store)
                                   if cfg.exp_store is not None else None),
                "n_experiences": (cfg.n_experiences
                                  if cfg.exp_store is not None else None),
                "features_mode": cfg.features_mode,
                "features_shift": (cfg.features_shift
                                   if cfg.features_mode == "shifted" else 0),
            },
            tool_calls=[],
            # ⚠️ 非空即表示"可能有外部信息进来"；回放模式下必须为空
            retrieved=[],
            mode=mode,
            llm_raw=("\n--- sample ---\n".join(raws) if len(raws) > 1
                     else (raws[0] if raws else "")),
            n_samples=n_samples_eff,
            samples=[{"parsed": p, "_ok": bool(p)} for p in parsed_list],
            parsed=dict(parsed),
            parse_ok=bool(parse_ok),
            parse_error=parse_error,
            latency_ms=int(latency_ms),
            requested=requested,
            risk=risk_out,
            executed=executed,
            order=order_d,
            reject_code=reject_code or (risk_d.code if not risk_d.accepted else ""),
            reject_msg=reject_msg or ("" if risk_d.accepted else
                                      str(final.get("reason") or "")),
        )
        if not parse_ok and not rec.parse_error:
            rec.parse_error = "解析失败（原文见 llm_raw）"
        # 解析失败时 ``parsed`` 必须保持**空**——往里塞一个 ``_vote``
        # 会让「这条记录有没有意图」这个判断依赖"字典是不是空"，
        # 而那是错的判据（空 dict 才是诚实的状态：模型什么都没给出来）。
        if parse_ok:
            rec.parsed.setdefault("_vote", vote)
        return rec.finalize()

    # -- 辅助 -----------------------------------------------------------
    def _suggest_max_size(self, *, mid: float, equity: float) -> float:
        """给模型的"建议最大量"。与风控的裁量上限**同一口径**——
        两处口径不同会让模型经常撞上限，而"撞上限"本身不说明策略好坏。

        ⚠️ **必须向下取整到一个干净的数**。实测：给模型 17 位有效数字
        （``0.24730285325666945``）时，它会写 6 位小数 ``0.247303`` ——
        **比上限大**，于是触发 ``size_cap``。虽然现在归因已经能区分
        "实质裁剪"与"四舍五入噪声"（见 ``RiskDecision.resize_is_material``），
        但**从一开始就别邀请这个错误**更省事：给一个模型不需要再取整的数。
        取 2 位有效数字：模型要么照抄（正好在上限内），要么明显偏离
        （那是真的超量，该被夹）。
        """
        lim = self.limits
        cap = lim.max_notional
        if equity > 0:
            cap = min(cap, equity * lim.max_equity_frac)
        else:
            cap = 0.0
        if mid <= 0:
            return 0.0
        raw = cap / mid
        if raw <= 0:
            return 0.0
        # 下取整到 2 位有效数字（0.247… → 0.24）。至少保留一点量，
        # 否则"建议最大量为 0"会让模型以为不能交易。
        import math

        exp = math.floor(math.log10(raw)) - 1
        step = 10.0 ** exp
        return max(step, math.floor(raw / step) * step)

    def _build_order(self, final: dict[str, Any],
                     *, tick: int) -> OrderRequest:
        """把风控后的意图翻成 ``OrderRequest``。

        ⚠️ 这一层**不再改量**（量已由风控定稿），只做翻译与格式校验。
        若翻译失败（比如市价单带了价格），抛出去由调用方记成管线缺陷。
        """
        cfg = self.config
        action = str(final["action"]).lower()
        ot = str(final.get("ordType") or "market").lower()
        if ot not in ("limit", "market", "post_only", "ioc", "fok"):
            ot = "market"
        px = final.get("px")
        if ot == "market":
            px = None
        elif px is None:
            # 限价单必须有价；没有就退化成市价（**格式**兜底，不是语义猜测：
            # "买入"这个意图仍然明确，只是没指定价）
            ot = "market"

        tp, sl = final.get("tp"), final.get("sl")
        algo = None
        if tp is not None or sl is not None:
            algo = AlgoOrder(
                tp_trigger_px=float(tp) if tp is not None else None,
                sl_trigger_px=float(sl) if sl is not None else None,
                attach_algo_cl_ord_id="",  # 由调用方按需填
            )
        return OrderRequest(
            inst_id=cfg.inst_id,
            side="buy" if action == "buy" else "sell",
            ord_type=ot,
            sz=float(final["sz"]),
            px=float(px) if px is not None else None,
            td_mode=cfg.td_mode,
            pos_side=cfg.pos_side,
            reduce_only=bool(final.get("reduceOnly", False)),
            attach_algo=algo,
            cl_ord_id=f"tw-{int(tick)}-{cfg.agent_id}"[:32],
            lever=float(cfg.lever),
        )


# ======================================================================
# 记录辅助
# ======================================================================
def _order_to_dict(req: OrderRequest) -> dict[str, Any]:
    """``OrderRequest`` → 可 JSON 化的 dict（OKX 字段名）。

    用 OKX 的名字（``tdMode`` / ``ordType`` / ``attachAlgoOrds``）而不是
    内部名字：留痕的消费者是**人和外部工具**，他们查 OKX 文档时
    应该能对上号。内部名字只在代码里用。
    """
    d: dict[str, Any] = {
        "instId": req.inst_id,
        "side": req.side,
        "ordType": req.ord_type,
        "sz": req.sz,
        "px": req.px,
        "tdMode": req.td_mode,
        "posSide": req.pos_side,
        "reduceOnly": req.reduce_only,
        "lever": req.lever,
        "clOrdId": req.cl_ord_id,
    }
    if req.attach_algo is not None:
        a = req.attach_algo
        d["attachAlgoOrds"] = [{
            "tpTriggerPx": a.tp_trigger_px,
            "tpOrdPx": a.tp_ord_px,
            "slTriggerPx": a.sl_trigger_px,
            "slOrdPx": a.sl_ord_px,
            "tpTriggerPxType": a.tp_trigger_px_type,
            "slTriggerPxType": a.sl_trigger_px_type,
        }]
    return d


# ======================================================================
# 会话：走一串 K 线，每根收盘决策一次
# ======================================================================
@dataclass(slots=True)
class SessionResult:
    """一次会话的产出。"""

    run_id: str
    records: list[DecisionRecord] = field(default_factory=list)
    #: 跳过的根（mid 非法等）——**跳过了什么也要能看见**，
    #: 否则"决策数少于 K 线数"会被误读成"Agent 漏了决策"。
    skipped: list[dict[str, Any]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.records)

    def stats(self) -> dict[str, Any]:
        from .decision_log import log_stats

        s = log_stats(self.records)
        s["n_skipped"] = len(self.skipped)
        return s

    def decisions(self) -> list[str]:
        return [str(r.parsed.get("action") or "?") for r in self.records]


def run_session(
    agent: TradingAgent,
    series: Any,
    *,
    start: int = 0,
    end: int | None = None,
    run_id: str = "session",
    equity: float = 0.0,
    cash: float = 0.0,
    n_open_positions: int = 0,
    data_snapshot: str = "",
    log_path: str | None = None,
    on_record: Any = None,
) -> SessionResult:
    """在 K 线上跑一串决策（**每根收盘一次**）。

    决策频率按用户 2026-09-21 拍板：**每根 K 线收盘决策一次**。
    理由（设计文档 §8.2）：LLM 单次延迟几百 ms ~ 数秒，
    高频用它没有意义；而"每根 K 线一次"是它与人类交易员最接近的节奏。

    ⚠️ 本函数用 ``start``/``end`` 而**不是**"从第 0 根到最后"：
    回放某一段历史时必须能从中间开始，否则每跑一次都要重头，
    而"从头"意味着前期决策会影响后期状态——那是一段不同的历史。

    ``on_record`` 是给调用方的钩子（比如"成交后更新账户"）。
    如果它返回一个 ``dict``，其中的 ``equity`` / ``cash`` /
    ``position`` / ``position_qty`` 会**覆盖**后续 tick 的账户输入。
    这样账户状态由调用方持有（本模块保持无状态）。

    ``log_path`` 给了就顺手落 JSONL（用 ``DecisionLog`` 的只追加语义）。
    """
    n = len(getattr(series, "close", []))
    if end is None:
        end = n - 1
    if not (0 <= start <= end < n):
        raise ValueError(
            f"区间非法：start={start} end={end} 序列长度={n}"
            "（要跑一根也得有 start<=end<n）"
        )

    res = SessionResult(run_id=run_id)
    log = None
    if log_path:
        from .decision_log import DecisionLog

        log = DecisionLog(log_path).open("a")

    try:
        st = {"equity": equity, "cash": cash,
              "position": None, "position_qty": 0.0}
        for i in range(int(start), int(end) + 1):
            try:
                vis = visible_from_series(
                    series, i,
                    n_closes=agent.config.n_closes,
                    equity=st["equity"], cash=st["cash"],
                    inventory=st["position_qty"],
                )
            except ValueError as exc:
                res.skipped.append({"i": i, "reason": str(exc)})
                continue
            rec = agent.decide(
                visible=vis, tick=i, run_id=run_id,
                equity=st["equity"], cash=st["cash"],
                position=st["position"], position_qty=st["position_qty"],
                n_open_positions=n_open_positions,
                data_snapshot=data_snapshot,
            )
            res.records.append(rec)
            if log is not None:
                log.append(rec)
                log.flush()
            if on_record is not None:
                upd = on_record(rec, i, st)
                if isinstance(upd, dict):
                    st.update(upd)
    finally:
        if log is not None:
            log.close()
    return res
