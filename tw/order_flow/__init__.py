"""订单流机制（二期）。

模块划分
--------
``meta_order``   元订单生成与拆分执行 —— LMF 链条的第一环：
                 大额交易者把一笔意图拆成多张同向子单，制造订单符号的长记忆。
``adaptive_liquidity``（在 ``tw/agents/`` 下）—— 第二环：
                 流动性提供者感知到订单流的方向性，主动在该方向稀疏挂单，
                 于是局部流动性变薄、冲击函数变凹。

为什么单独开一个子包而不是塞进 ``tw/``
--------------------------------------
这两块是**机制**，不是"主体类型"也不是"分析工具"。它们的共同特征是
"改变订单到达过程"，而一期的全部结论都建立在"订单到达近似泊松"之上。
把它们放在一个显式的命名空间里，下游任何一处看到 ``tw.order_flow``
就应该警觉：**这里的开关会改变一期的基线统计特征。**
"""

from .meta_order import (
    ENTROPY_META_ORDER,
    MetaOrderConfig,
    MetaOrderMixin,
    MetaOrderState,
)

__all__ = [
    "ENTROPY_META_ORDER",
    "MetaOrderConfig",
    "MetaOrderMixin",
    "MetaOrderState",
]
