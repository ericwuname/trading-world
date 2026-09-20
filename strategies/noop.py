"""对照基准：什么都不做。

它有两个用途，都不是"凑数"：

1. **零假设**。任何策略的表现都要和"什么都不做"比。在上涨行情里
   随便持有点什么都能赚钱——那不是策略的功劳。
   实验台会把它和每个策略放在同一张表里，`pnl_capture` 应当为 0。

2. **实验台的健全性检查**。注入 Noop 之后，背景行情必须与不注入时
   **逐点完全相同**（见 `tests/test_market.py::TestStrategyInjection`）。
   如果这条不成立，说明"主体数量"在扰动市场随机流，
   那么所有 A/B 对照测到的差异里都混着伪影。
"""

from __future__ import annotations

from tw.strategy import Strategy


class Noop(Strategy):
    """不提交任何订单。"""

    KIND = "noop"

    def on_tick(self, ctx):  # noqa: ANN001
        return None
