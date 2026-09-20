"""专用实验运行器（二期）。

与 ``scripts/`` 下那些"跑一次、出一堆图表"的脚本不同，这里的模块是
**可被测试直接调用的实验逻辑**：它们把"实验怎么做"（配对方式、禀赋处理、
观测窗口、统计检验）固化成代码，而不是散在脚本里。

这么分的原因：脚本里的实验设计一旦写错（比如配对没配对、每档重新初始化
忘了重置），产出的是一条**看起来合理但归因错误**的曲线，
而它不会被任何测试拦住——因为脚本不在测试范围内。
把实验逻辑搬进 ``tw/`` 之后，配对方式、独立性、禀赋一致性
都可以被断言钉住。
"""

from .reflexivity import (
    ADOPTER_CASH,
    ADOPTER_INVENTORY,
    ReflexivityConfig,
    build_market,
    paired_by_seed,
    run_one,
    run_reflexivity_sweep,
    spearman,
)

__all__ = [
    "ADOPTER_CASH",
    "ADOPTER_INVENTORY",
    "ReflexivityConfig",
    "build_market",
    "paired_by_seed",
    "run_one",
    "run_reflexivity_sweep",
    "spearman",
]
