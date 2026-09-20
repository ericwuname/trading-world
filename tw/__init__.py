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
"""

__version__ = "0.2.0"

from .book import OrderBook
from .config import Population, SimConfig
from .engine import MatchingEngine
from .eval import evaluate, format_table, markout, mid_series, pnl_breakdown
from .market import Market, run_simulation
from .scenarios import SCENARIOS, Scenario, get_scenario, list_scenarios
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
    "__version__",
]
