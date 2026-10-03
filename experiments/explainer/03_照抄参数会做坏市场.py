"""最小实验 03：照抄别人的参数，会把这个市场做坏

对应《交易世界-讲给人听》第四章。

它做什么
--------
复用生产代码里的标定扫描（`scripts/calibrate.py` 的 `scan`），
只改一个参数：**零智能交易者的报价幅度上界**，看市场波动率怎么变。

    上界 2.00%（施工蓝图建议）　vs　上界 0.18%（本项目收窄后）

看什么
------
    σ（每 tick 收益率标准差）与真实 BTC 小时线的比值。
    比值 1.0 = 命中；大于 3 = 这个市场已经不是加密市场了。

顺便验证第二件事：σ 依赖参与者数量
--------------------------------
同一个参数下， participants 从 100 增到 500，看 σ 怎么变。
⇒ "σ 标定好了"这句话**必须同时说"在多少个参与者下标定的"**，否则没有意义。

为什么脚本不断言精确值
--------------------
换种子/池子数字就会变（报告里 511.6 对应的是纯零智能池）。
这里断言的是**量级**（照抄 ⇒ 至少 3 倍于真实），并把参考值打出来。

用法：python experiments/explainer/03_照抄参数会做坏市场.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import calibrate as C  # noqa: E402

# 报告里的两个上界（%）：蓝图建议 vs 本项目收窄后
BLUEPRINT = 0.02
TUNED = 0.0018


def pick_pool() -> tuple[str, dict]:
    """挑一个池子。报告 §5 那张表是"纯零智能"池的，优先用它。"""
    for tag, mix in C.POOLS.items():
        if "零智能" in tag:
            return tag, mix
    tag = next(iter(C.POOLS))
    return tag, C.POOLS[tag]


def main() -> int:
    print("=" * 72)
    print("最小实验 03：照抄参数会把市场做坏")
    print("=" * 72)

    reals = C.real_metrics()
    target = {m.label: m.flat()["sigma_bp"] for m in reals}
    btc = target["BTCUSDT_1h"]
    print(f"\n标定目标：真实 BTC 1h 的 σ = {btc:.2f}bp")
    print(f"          （对照 ETH {target.get('ETHUSDT_1h', float('nan')):.2f}bp、"
          f"SOL {target.get('SOLUSDT_1h', float('nan')):.2f}bp）")

    tag, mix = pick_pool()
    print(f"\n池子：{tag}　（与报告 §5 用的是同一个池）")

    print(f"\n{'报价幅度上界':>12}{'σ(bp)':>16}{'σ/真实BTC':>12}")
    print("-" * 40)
    # ⚠️ `calibrate.scan` 的每行键是 **value**（不是扫描变量名 off_hi/p_active）——
    #    这一点本脚本第一次写错过（读成 off_hi ⇒ 打印 nan ⇒ 自检正确地拦住了它）。
    rows = C.scan(tag, mix, key="off_hi", values=[BLUEPRINT, TUNED])
    got: dict[float, float] = {}
    for r in rows:
        v = float(r["value"])
        s = float(r["sigma_bp"])
        sd = r.get("sigma_sd")
        sd_s = f"±{sd:.1f}" if isinstance(sd, (int, float)) and sd == sd else "—"
        got[v] = s
        name = "（蓝图建议）" if abs(v - BLUEPRINT) < 1e-12 else "（收窄后）"
        print(f"{v * 100:>11.2f}%{s:>10.1f} {sd_s:>5}{s / btc:>11.2f}×  {name}")
    print("\n（σ 后的 ± 是**种子之间的标准差** —— 这就是「别把一个数当精确值」的实体：")
    print("  同一参数换一组种子就会挪动，所以本脚本只断言量级，不断言精确值。）")

    s_blue = got.get(BLUEPRINT)
    s_tune = got.get(TUNED)
    if s_blue is None or s_tune is None:
        print("\n结论：扫描没返回预期的两个点，参数口径可能变了 —— "
              "请检查 calibrate.scan 的 off_hi 键。")
        return 1

    print("-" * 72)
    print(f"检查 1　照抄蓝图参数 ⇒ 市场波动率至少是真实 BTC 的 3 倍")
    print(f"        {s_blue / btc:.2f} 倍")
    ok1 = s_blue / btc >= 3.0

    print(f"\n检查 2　收窄后应当回到真实量级（1 倍上下，容差 ±30%）")
    print(f"        {s_tune / btc:.2f} 倍")
    ok2 = abs(s_tune / btc - 1.0) < 0.30

    print("\n参考（报告里的值，不是断言）：")
    print("  上界 2.00% → σ=511.6bp = 真实 BTC 的 10.67 倍（完全失真）")
    print("  上界 0.18% → σ=52.8bp  = 1.10 倍（量级命中）")
    print("  ⚠️ 注意：这两个数来自 out/calibration.json（σ=511.64 ÷ 47.93 = 10.67）。")
    print("     另一份技术文档写的是 515.7 / 10.76 倍 —— 那是更早一次运行，")
    print("     两个都是真的，本脚本以产物为准。")

    if not (ok1 and ok2):
        print("\n结论：量级对不上。请连同上面的表一起讨论，不要直接引用报告数字。")
        return 1
    print("\n结论：一个看起来只是\"风格选择\"的参数，决定了这个市场的情绪量级。")
    print("      照抄别人的参数，等于把市场的情绪放大十倍 —— 那已经不是它了。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
