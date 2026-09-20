"""主体注册表。"""

from __future__ import annotations

from .base import Agent, AgentStats, clipped_strength
from .chartist import Chartist
from .fundamentalist import Fundamentalist, step_fundamental
from .market_maker import MarketMaker
from .zero_intel import ZeroIntelligence

AGENT_CLASSES: dict[str, type[Agent]] = {
    "zero_intel": ZeroIntelligence,
    "fundamentalist": Fundamentalist,
    "chartist": Chartist,
    "market_maker": MarketMaker,
}

__all__ = [
    "Agent",
    "AgentStats",
    "clipped_strength",
    "ZeroIntelligence",
    "Fundamentalist",
    "Chartist",
    "MarketMaker",
    "step_fundamental",
    "AGENT_CLASSES",
]
