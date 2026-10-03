"""最小实验 02：市场的深度来自时间，不是来自钱

对应《交易世界-讲给人听》第二章。

它做什么
--------
固定"要卖掉的量"（占持仓 1%），只改**拆成几片卖**，其余条件完全相同：

    1 片（一次砸完）   vs   10 片   vs   100 片

关键设计：因果滑点
------------------
"价格被推走了多少"不能用**卖出前的价格**当基准 —— 因为拆 100 片要跨 100 个
tick，而这个市场每 tick 的自然波动就有 40~50bp，**市场自己也在走**。
所以要给每一组配一个**同种子、但完全不下单"的对照组**，
用对照组同一时刻的价格当基准（这就是"反事实价格"）。
少了这一步，拆得越细反而看起来冲击越大（实测会把 −14bp 的慢抛放大成 −50bp），
结论会完全反过来。

为什么脚本不断言精确值
--------------------
换种子数字就会变。这里断言的是**方向与量级**（拆 100 片的冲击必须明显小于
拆 1 片），并把报告里的参考值一起打出来。

用法：python experiments/explainer/02_深度来自时间.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import run_stage2 as S2  # noqa: E402
import run_stage3 as S3  # noqa: E402

MIX = S2.MIX                     # 蓝图完整配方（两类人 + 零智能底噪）
SHOCK_FRAC = 0.01                # 目标清算量 = 持仓的 1%
SEEDS = (101, 102)
SLICES = (1, 10, 100)

# 报告里的参考值（docs/交易世界-讲给人听.md 第二章）
REPORT_REF = {1: "-148.4bp / 成交率 16%", 10: "-84.7bp / 66%", 100: "-14.1bp / 98%"}


def main() -> int:
    print("=" * 72)
    print("最小实验 02：深度来自时间，不是来自钱")
    print(f"配方：{MIX}")
    print(f"固定清算量：持仓的 {SHOCK_FRAC:.0%}　只改拆成几片　种子：{list(SEEDS)}")
    print("=" * 72)

    t0 = time.time()
    # 对照组：同种子、但完全不下单 —— 它的价格路径就是"如果没有这次抛售会怎么走"
    controls = S3.make_controls(MIX, list(SEEDS))
    print(f"\n对照组（不下单）已跑完，用时 {time.time() - t0:.1f}s")

    print(f"\n{'拆成几片':>8}{'成交率':>9}{'因果滑点(bp)':>15}{'标准误':>9}")
    print("-" * 43)
    rows: dict[int, tuple[float, float]] = {}
    for k in SLICES:
        slips, rates = [], []
        for s in SEEDS:
            run = S3.run_path(MIX, s, shock_frac=SHOCK_FRAC, slices=k)
            mean, sem = S3.causal_slippage(run, controls[s]["path"], controls[s]["p0"])
            slips.append(mean)
            rates.append(run["delivered_qty"] / max(run["target_qty"], 1e-9))
        rows[k] = (float(np.mean(slips)), float(np.mean(rates)))
        print(f"{k:>8}{rows[k][1] * 100:>8.0f}%{rows[k][0]:>15.1f}"
              f"{float(np.mean([abs(x) for x in slips])) / len(slips):>9.1f}")
    print(f"\n（总用时 {time.time() - t0:.1f}s）")

    s1, r1 = rows[1]
    s100, r100 = rows[100]
    print("-" * 72)
    print(f"检查 1　拆细之后冲击必须显著变小")
    print(f"        1 片 {s1:.1f}bp　→　100 片 {s100:.1f}bp"
          f"　⇒ 缩小到 {abs(s100 / s1):.0%}")
    ok1 = abs(s100) < abs(s1) * 0.5

    print(f"\n检查 2　拆细之后成交率必须显著变高")
    print(f"        1 片 {r1 * 100:.0f}%　→　100 片 {r100 * 100:.0f}%")
    ok2 = r100 > r1 + 0.2

    print("\n参考（报告里的值，不是断言 —— 换种子会差几个点）：")
    for k, v in REPORT_REF.items():
        print(f"  {k:>3} 片：{v}")

    if not (ok1 and ok2):
        print("\n结论：现象**没有**在这次运行里复现。")
        print("      可能是规模/种子不同；也可能是真的没复现。")
        print("      请把上面的表连同种子一起贴出来讨论，不要只说'没复现'。")
        return 1
    print("\n结论：同样的量、同样的市场，**只因为'卖得急不急'，代价差了一个数量级**。")
    print("      所以大单的流动性需求，本质上是对**时间**的需求，不是对**钱**的需求。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
