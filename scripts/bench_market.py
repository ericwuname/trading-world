"""市场内核的性能与等价性基准。

为什么需要它
------------
二期的注入点（``decorate_agent`` / ``_limit_activity`` / ``_step_fundamental``）
本身是**零成本**的（默认原样返回）。但 ``_refresh_state`` 里多算了一个
``flow_imbalance``，而 ``_refresh_state`` 是**每个提交了订单的主体、每 tick
调用一次**。于是凭空多出 O(主体数 × 窗口长度) 的开销：

    slice(200) + 2 × nansum(200)   ×   ~150 主体  ×  ~12000 tick

实测后果是整条流水线从"约 70 分钟"变成接近三小时。**慢本身不是 bug**，
但"文档说 70 分钟、实际 2.7 小时"会让复现的人以为卡死了，
而且没有人会去怀疑一个从来没被测量过的数字。

这个基准做两件事，缺一不可：

1. **测成本**：同一场模拟，注入点带/不带 ``flow_imbalance`` 重算，比墙钟；
2. **测等价**：两次的**逐点行情序列必须完全一致**。
   只测速度不测等价，等于用"看起来差不多"换性能——
   而这条项目线已经反复证明"看起来差不多"会掩盖真问题。

用法::

    python scripts/bench_market.py           # 默认 3000 tick、4 种子
    python scripts/bench_market.py --ticks 8000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tw import Population, SimConfig  # noqa: E402
from tw.market import Market  # noqa: E402


class CacheFlowImbalanceMarket(Market):
    """把 ``_flow_imbalance`` 按 tick 记忆化。

    ``_flow_imbalance(window)`` 的结果只依赖 ``self.tick`` 与
    ``self.log.flow[:tick]``。``log.flow`` 是**每 tick 写一格**的，
    所以同一个 tick 内重复调用必然返回同一个值——记忆化是
    **行为保持**的，不是近似。
    """

    def __init__(self, *a, **kw) -> None:
        self._fi_tick = -1
        self._fi_win = None
        self._fi_val = 0.0
        super().__init__(*a, **kw)

    def _flow_imbalance(self, window: int = 200) -> float:
        if self._fi_tick == self.tick and self._fi_win == window:
            return self._fi_val
        v = super()._flow_imbalance(window)
        self._fi_tick = self.tick
        self._fi_win = window
        self._fi_val = v
        return v


def run_once(factory, seed: int, n_ticks: int,
             n_agents: int = 200) -> tuple[float, dict]:
    pop = Population.from_shares(n_agents, {"zero_intel": 0.3,
                                           "fundamentalist": 0.4,
                                           "chartist": 0.3})
    t0 = time.perf_counter()
    m = factory(SimConfig(seed=seed, n_ticks=n_ticks, population=pop))
    m.run(n_ticks)
    dt = time.perf_counter() - t0
    snap = {
        "mid": np.asarray(m.log.mid, dtype=float).copy(),
        "n_trades": len(m.log.trades),
        "flow_sum": float(np.nansum(m.log.flow)),
    }
    return dt, snap


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--n-agents", type=int, default=200)
    args = ap.parse_args()

    seeds = [20260917 + 7 * i for i in range(args.seeds)]
    print("=" * 74)
    print(f"市场内核基准：{args.ticks} tick × {len(seeds)} 种子 × "
          f"{args.n_agents} 主体")
    print("=" * 74)

    t_old = t_new = 0.0
    identical_mid = identical_all = 0
    for sd in seeds:
        d_old, s_old = run_once(Market, sd, args.ticks, args.n_agents)
        d_new, s_new = run_once(CacheFlowImbalanceMarket, sd,
                                args.ticks, args.n_agents)
        t_old += d_old
        t_new += d_new
        same_mid = np.array_equal(s_old["mid"], s_new["mid"])
        same_rest = (s_old["n_trades"] == s_new["n_trades"]
                     and s_old["flow_sum"] == s_new["flow_sum"])
        identical_mid += int(same_mid)
        identical_all += int(same_mid and same_rest)
        print(f"  种子 {sd}：原版 {d_old:6.2f}s  记忆化 {d_new:6.2f}s  "
              f"加速 {d_old / d_new:4.2f}×  "
              f"逐点中间价一致 {'✅' if same_mid else '❌'}  "
              f"成交数/流量一致 {'✅' if same_rest else '❌'}")

    print()
    print(f"  合计：原版 {t_old:.1f}s → 记忆化 {t_new:.1f}s"
          f"（加速 {t_old / max(t_new, 1e-9):.2f}×）")
    print(f"  逐点中间价完全一致的场次：{identical_mid}/{len(seeds)}")
    print(f"  逐点中间价 + 成交数 + 流量 全一致的场次：{identical_all}/{len(seeds)}")
    if identical_all != len(seeds):
        print("  ❌ 不等价——记忆化改变了结果，不能用。")
        return 1
    print("  ✅ 等价（逐点一致）。记忆化是行为保持的，可以安全启用。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
