"""示例策略集。同时也是"写一个策略"的模板。

三类各给了样本，刻意包含两个**反面基准**（什么都不做 / 随机乱做）——
没有它们，任何"我的策略赚钱了"都没有参照系：

  对照类   noop          什么都不做。**必须是"零成交、零盈亏"**，
                         它同时也是实验台的健全性检查：注入它不该改变背景行情。
           random_taker  随机方向市价单。代表"完全无信息但提供了流动性需求"，
                         是评估"信息/技能是否真的加分"的下界。
  做市类   mm_naive      固定价差双边挂单。最朴素的做市，
                         用来回答"只要挂双边就能赚价差吗"。
           mm_skewed     加库存偏移的做市。测"管住库存"值多少钱。
  执行类   twap          把目标仓位拆成 N 份、每 tick 一份执行。
                         用来量化"分批执行省下多少冲击"。
           momentum      动量追单。测"追涨杀跌"在这个市场里是赚还是亏。

每个策略都控制在 30 行以内——不是为了短，而是因为**能被审阅的策略才有意义**。
"""

from .mm_naive import NaiveMaker, SkewedMaker
from .momentum import MomentumTaker
from .noop import Noop
from .random_taker import RandomTaker
from .twap import TwapExecutor

#: 名称 → (类, 默认参数)。实验台按名称查表。
REGISTRY: dict[str, tuple[type, dict]] = {
    "noop": (Noop, {}),
    "random_taker": (RandomTaker, {"p_active": 0.05, "qty": 1.0}),
    "mm_naive": (NaiveMaker, {"half_spread_bp": 12.0, "qty": 2.0}),
    "mm_skewed": (SkewedMaker, {"half_spread_bp": 12.0, "qty": 2.0, "target_inv": 0.0}),
    "twap": (TwapExecutor, {"side": "buy", "target_qty": 120.0, "horizon": 600}),
    "momentum": (MomentumTaker, {"lookback": 50, "threshold_bp": 20.0, "qty": 1.0}),
}

__all__ = [
    "Noop",
    "RandomTaker",
    "NaiveMaker",
    "SkewedMaker",
    "TwapExecutor",
    "MomentumTaker",
    "REGISTRY",
]
