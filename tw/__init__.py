"""交易世界 —— 基于主体建模(ABM)的人工市场。

施工蓝图：`../交易世界实例化：基于主体建模(ABM)施工蓝图.md`

模块地图
--------
types.py      Order / Trade / MarketState / PriceHistory —— 数据结构
book.py       OrderBook —— 价格优先 + 时间优先的限价订单簿
engine.py     MatchingEngine —— 连续双向拍卖撮合
config.py     SimConfig / Population —— 一次模拟的全部参数
agents/       ZeroIntelligence / Fundamentalist / Chartist / MarketMaker
market.py     主循环：驱动决策 -> 撮合 -> 结算 -> 记账
logger.py     SimLog：逐 tick 序列 + 成交明细 + 订单簿快照
analyzer.py   统计特征：峰度、|r| 自相关、方差比、Hill 尾部指数、冲击事件研究
realdata.py   真实行情 CSV 载入（复用既有交付包，不重新下载）
eval.py       **策略评估**：markout、PnL 分解、库存风险、权益回撤
strategy.py   **策略 API**：写 `on_tick(ctx)` 即可接入实验台
scenarios.py  **场景库**：平静 / 常态 / 承压 / 清算 / 稀薄，用于稳健性检验
marketdb.py   数据层：SQLite 三源共用 schema（真实 / 合成 / 回放）
okx_data.py   拉真实 OHLCV（分页游标减 1ms；失败也记账）
synthetic.py  离线自测试数据（可复现、OHLC 有影线）
order_model.py **A1** OKX 风格订单模型：tdMode / posSide / reduceOnly / 附带止盈止损
account.py     **A1** 保证金账户账本：分档 MMR / 多空强平价 / 穿仓反手
decision_log.py **A2** 决策留痕：证据链六项 + 确定性 decision_id + JSONL
risk.py        **A2** 风控闸门：数值 / 方向 / 量 / 保证金；**只有量上限会裁剪**
llm.py         **A3** LLM 通道：真机 / 回放 / 脚本化三种实现，缺 key 必须报错
prompts.py     **A3** 版本化 prompt 模板（改模板必须加版本，否则归因失效）
parse.py       **A3** 容错解析：只翻译不修正；解析失败也留痕
agent.py       **A3** 决策回路：可见状态 → prompt → LLM → 解析 → 风控 → 订单意图
simexec.py     **A4** 逐根 K 线执行模拟（bar 级成交的四个假设**全部显式声明**）
policy.py      **A4** 同信息规则基线（noop / momentum / meanrevert / random_taker）
agent_run.py   **A4** 完整回路：Agent + 风控 + 执行 + 账本，账户状态回流到下次决策
eval_agent.py  **A4** 分层评估：决策层 / 执行层 / 结果层 + 成本敏感性 + 区间重叠检验
segmented.py   **A6** 多段独立运行 + 同段配对检验（换掉判不出来的"每根均值"）
outcome.py     **A6** 结果回填（证据链第 ⑥ 项；只写 outcome，不碰可见状态）
kpi.py         **A6** KPI：多目标互相咬住（在场率下限 + 回撤上限 + 换手区间）
reflect.py     **A6** 复盘归因（含 lucky/unlucky）+ 经验库（按 t' < t 检索）
"""

__version__ = "0.8.0"

from .account import MarginAccount, MarginConfig, Position
from .agent import (
    AgentConfig,
    SessionResult,
    TradingAgent,
    run_session,
    visible_from_series,
)
from .agent_run import RunResult, order_request_from_record, run_agent_session
from .book import OrderBook
from .config import Population, SimConfig
from .decision_log import DecisionLog, DecisionRecord, build_visible_state, log_stats
from .engine import MatchingEngine
from .eval import evaluate, format_table, markout, mid_series, pnl_breakdown
from .eval_agent import (
    block_bootstrap_ci,
    compare_to_baselines,
    confidence_calibration,
    cost_sensitivity_analytic,
    cost_sensitivity_rerun,
    decision_layer,
    evaluate_run,
    execution_layer,
    format_report_lines,
    max_drawdown,
    result_layer,
    sharpe,
)
from .llm import (
    HTTPClient,
    LLMClient,
    LLMConfig,
    LLMResponse,
    Recorder,
    ReplayClient,
    ScriptedClient,
    make_client,
    prompt_hash,
)
from .market import Market, run_simulation
from .order_model import (
    AlgoOrder,
    OrderRejected,
    OrderRequest,
    apply_reduce_only,
    precheck,
)
from .outcome import (
    BackfillConfig,
    backfill,
    backfill_one,
    outcome_summary,
)
from .parse import (
    ParseResult,
    consistency,
    decision_signature,
    majority_sample,
    parse_decision,
)
from .policy import (
    AlwaysHold,
    MeanRevert,
    Momentum,
    Policy,
    RandomTaker,
    list_policies,
    make_policy,
    validate_policy_output,
)
from .prompts import DEFAULT_TEMPLATE, TEMPLATES, build_messages
from .risk import RiskDecision, RiskLimits, check, is_decision_bar
from .segmented import (
    SegmentedRuns,
    min_detectable_effect,
    paired_verdict,
    power_analysis,
    run_paired_segments,
    segment_correlation,
    segment_ranges,
    sensitivity_check,
)
from .scenarios import SCENARIOS, Scenario, get_scenario, list_scenarios
from .simexec import BarExecutor, ExecConfig, Fill, bars_from_series
from .strategy import Strategy, StrategyContext, make_strategy
from .types import (
    BookSnapshot,
    DepthLevel,
    MarketState,
    Order,
    PriceHistory,
    Trade,
)

__all__ = [
    "Order",
    "Trade",
    "MarketState",
    "PriceHistory",
    "BookSnapshot",
    "DepthLevel",
    "OrderBook",
    "MatchingEngine",
    "Market",
    "run_simulation",
    "SimConfig",
    "Population",
    # 策略实验台
    "Strategy",
    "StrategyContext",
    "make_strategy",
    "evaluate",
    "markout",
    "pnl_breakdown",
    "mid_series",
    "format_table",
    "Scenario",
    "SCENARIOS",
    "get_scenario",
    "list_scenarios",
    # A1：订单模型 + 账户
    "AlgoOrder",
    "OrderRequest",
    "OrderRejected",
    "precheck",
    "apply_reduce_only",
    "MarginAccount",
    "MarginConfig",
    "Position",
    # A2：留痕 + 风控
    "DecisionLog",
    "DecisionRecord",
    "build_visible_state",
    "log_stats",
    "RiskLimits",
    "RiskDecision",
    "check",
    "is_decision_bar",
    # A3：LLM 通道 + 模板 + 解析 + 决策回路
    "LLMConfig",
    "LLMResponse",
    "LLMClient",
    "HTTPClient",
    "ScriptedClient",
    "ReplayClient",
    "Recorder",
    "make_client",
    "prompt_hash",
    "TEMPLATES",
    "DEFAULT_TEMPLATE",
    "build_messages",
    "ParseResult",
    "parse_decision",
    "consistency",
    "decision_signature",
    "majority_sample",
    "AgentConfig",
    "TradingAgent",
    "SessionResult",
    "run_session",
    "visible_from_series",
    # A4：执行模拟 + 规则基线 + 完整回路 + 分层评估
    "ExecConfig",
    "Fill",
    "BarExecutor",
    "bars_from_series",
    "Policy",
    "AlwaysHold",
    "Momentum",
    "MeanRevert",
    "RandomTaker",
    "make_policy",
    "list_policies",
    "validate_policy_output",
    "RunResult",
    "run_agent_session",
    "order_request_from_record",
    "decision_layer",
    "execution_layer",
    "result_layer",
    "confidence_calibration",
    "cost_sensitivity_analytic",
    "cost_sensitivity_rerun",
    "compare_to_baselines",
    "evaluate_run",
    "format_report_lines",
    "block_bootstrap_ci",
    "max_drawdown",
    "sharpe",
    # A6：多段配对度量（换掉判不出来的那个）
    "segment_ranges",
    "SegmentedRuns",
    "run_paired_segments",
    "paired_verdict",
    "segment_correlation",
    "power_analysis",
    "min_detectable_effect",
    "sensitivity_check",
    # A6：结果回填（证据链第 ⑥ 项，复盘的地基）
    "BackfillConfig",
    "backfill",
    "backfill_one",
    "outcome_summary",
    "__version__",
]
