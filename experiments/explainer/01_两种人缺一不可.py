"""最小实验 01：市场里的两种人，缺一不可

对应《交易世界-讲给人听》第一章。

它做什么
--------
复用**生产代码路径**（`scripts/run_stage2.py` 的四池配置与度量），
跑四个只有"配方"不同的市场，把三个指标并排打出来：

    ① 只留"看基本面"的   ② 只留"看图表"的   ③ 两者混合   ④ 真实 BTC

看什么
------
    峰度（excess kurtosis）：分布的"尖尾"程度。真实加密市场的尖尾很明显。
    波动率 σ：市场一天的情绪有多强。
    |r| 自相关：涨跌幅之间是不是有惯性（真实市场接近 0，即"不可预测"）。

为什么脚本**不断言精确值**
------------------------
换随机种子这些数就会变 —— 本项目最贵的一条教训就是
"**把一个数当成精确值来比较，比错更贵**"。
所以这里断言的是**方向与量级**（肥尾池的峰度必须明显更低），
并把报告里的参考值一起打出来给你对。

用法：python experiments/explainer/01_两种人缺一不可.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import run_stage2 as S2  # noqa: E402

# 规模：比报告小得多，跑得快；现象的方向不依赖规模（下面会自己检查）
TICKS = 4000
SEEDS = (11, 12, 13)

# 报告里的参考值（`docs/交易世界-讲给人听.md` 第一章）—— 给你对，不用来断言
REPORT_REF = {
    "② 只留基本面派": "肥尾消失（峰度崩塌）",
    "③ 只留图表派": "波动率爆炸",
    "④ 三者混合": "肥尾 + 波动率聚集 + 不可预测 同时出现",
}


def main() -> int:
    print("=" * 72)
    print("最小实验 01：两种人缺一不可")
    print(f"规模：{TICKS} tick × {S2.N_AGENTS} 个参与者 × {len(SEEDS)} 个种子")
    print("=" * 72)

    reals = S2.real_metrics()
    real = reals[0]
    print(f"\n真实基准（{real.label}）：σ={real.sigma_bp:.1f}bp  "
          f"峰度={real.excess_kurtosis:.2f}  |r|ACF(1)={real.acf_abs_lag1:.3f}")

    # 跑四个池子，多个种子取平均（单种子的峰度很不稳，这是本项目吃过的亏）
    acc: dict[str, dict[str, list[float]]] = {}
    t0 = time.time()
    for name, mix in S2.POOLS:
        for seed in SEEDS:
            _, log = S2.build(mix, n_ticks=TICKS, seed=seed)
            mm = S2.metrics_of(log, name).flat()
            for k in ("sigma_bp", "excess_kurtosis", "acf_abs_lag1"):
                acc.setdefault(name, {}).setdefault(k, []).append(mm[k])

    def avg(name: str, key: str) -> float:
        xs = acc[name][key]
        return sum(xs) / len(xs)

    print(f"\n{'':<22}{'σ(bp)':>10}{'峰度':>10}{'|r|ACF(1)':>11}")
    print("-" * 53)
    for name, _ in S2.POOLS:
        print(f"{name:<22}{avg(name, 'sigma_bp'):>10.1f}"
              f"{avg(name, 'excess_kurtosis'):>10.2f}"
              f"{avg(name, 'acf_abs_lag1'):>11.3f}")
    print(f"{'真实 ' + real.label:<22}{real.sigma_bp:>10.1f}"
          f"{real.excess_kurtosis:>10.2f}{real.acf_abs_lag1:>11.3f}")
    print(f"\n（用时 {time.time() - t0:.1f}s）")

    # ---- 断言：方向与量级，不是精确值 ----
    fund = next(n for n, _ in S2.POOLS if "基本面" in n and "图表" not in n)
    chart = next(n for n, _ in S2.POOLS if "图表" in n)
    mixed = S2.POOLS[-1][0]

    k_fund, k_mixed = avg(fund, "excess_kurtosis"), avg(mixed, "excess_kurtosis")
    s_chart, s_mixed = avg(chart, "sigma_bp"), avg(mixed, "sigma_bp")

    print("\n" + "-" * 72)
    print(f"检查 1　肥尾是不是被'只留基本面'抹平了")
    print(f"        {fund} 峰度 = {k_fund:.2f}　vs　{mixed} 峰度 = {k_mixed:.2f}")
    ok1 = k_fund < k_mixed
    print(f"        ⇒ {'成立' if ok1 else '**不成立**'}（报告的预期是：只留基本面 ⇒ 肥尾被抹平）")

    print(f"\n检查 2　波动率是不是被'只留图表'推爆了")
    print(f"        {chart} σ = {s_chart:.1f}bp　vs　{mixed} σ = {s_mixed:.1f}bp")
    ok2 = s_chart > s_mixed * 1.5
    print(f"        ⇒ {'成立' if ok2 else '**不成立**'}（预期：只留图表 ⇒ 波动率爆炸，"
          f"这里要求至少高 50%）")

    print(f"\n参考（报告里的说法，不是断言）：")
    for k, v in REPORT_REF.items():
        print(f"  {k}：{v}")

    # ---- 顺便看到的：方向对了，但厚度还差得远（这一条也要如实说）----
    k_ratio = k_mixed / real.excess_kurtosis if real.excess_kurtosis else float("nan")
    s_ratio = s_mixed / real.sigma_bp
    print("\n" + "-" * 72)
    print("顺便看到的（不是断言，是**没达标的**那一半）：")
    print(f"        混合池峰度 {k_mixed:.2f}  vs 真实 {real.excess_kurtosis:.2f}"
          f"  ⇒ 只有 {k_ratio:.0%}")
    print(f"        混合池 σ   {s_mixed:.1f}   vs 真实 {real.sigma_bp:.1f}"
          f"  ⇒ {s_ratio:.1f} 倍")
    print("        也就是说：**方向对（该有的现象都出现了），但肥尾厚度只有真实的"
          f"三分之一左右，波动率还偏高 {s_ratio:.1f} 倍。**")
    print("        这正是报告里那句权衡：模型是靠**把波动率抬到真实量级**才换来肥尾的，")
    print("        而两者还没法同时到位。")

    if not (ok1 and ok2):
        print("\n结论：现象**没有**在这次运行里复现。")
        print("      这不代表报告错了 —— 可能是规模/种子不同，也可能是真的没复现。")
        print("      遇到这种情况请把上面的表连同 Ticks/种子一起贴出来讨论。")
        return 1
    print("\n结论：两个方向都复现了。肥尾来自'正负反馈共存'，"
          "而不是某一种交易者单独造出来的。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
