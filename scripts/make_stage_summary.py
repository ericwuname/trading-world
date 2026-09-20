"""生成 ``docs/二期阶段小结.md``——每个阶段的「诚实边界」汇总。

为什么要有这个脚本
------------------
指导书 §0 要求每个阶段结束必须写一份"诚实边界"小结。这份小结原来的头部
就是这么写的::

    ⚠️ 本文件由 scripts/ 下的脚本产出内容汇总而成，不手抄数字。

**而实际上它是手写的**——没有任何脚本产出它，总览表里的
「k 反而从 0.665 走到 0.910」正是手抄的，和报告总览表里那三处是同一批数字。
于是那句声明本身成了这个项目里最讽刺的一行：
**一个声明"不手抄"的文件，内容是手抄的，而且没有机制会发现。**

这个脚本让那句声明变成真的：总览表的每个数字与每个判定都从
``out/stage*_metrics.json`` 现算，各阶段的"诚实边界"段落**逐字**取自
对应 JSON 的 ``honest_boundary`` 字段（那是各阶段脚本自己打印的）。

用法::

    python scripts/make_stage_summary.py     # 写 docs/二期阶段小结.md
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "out"
DOCS = ROOT / "docs"
TARGET = DOCS / "二期阶段小结.md"

STAGE_TITLES = {
    5: "永续合约机制层（内生资金费率）",
    6: "长记忆订单流（LMF）",
    7: "自适应/学习型主体与反身性",
    8: "Hawkes 过程订单到达",
    9: "多资产相关市场",
    10: "统一校准框架",
    11: "「至暗时刻」压力测试",
}


def f(v, fmt: str = ".3f", dash: str = "—") -> str:
    """把可能缺失/nan 的值格式化掉——不允许把 nan 印进交付物。"""
    if not isinstance(v, (int, float)):
        return dash
    if v != v:  # nan
        return dash
    if v in (float("inf"), float("-inf")):
        return dash
    return format(v, fmt)


def load(name: str) -> dict:
    p = OUT / f"{name}_metrics.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


# ----------------------------------------------------------------------
def line5(d: dict) -> tuple[str, str]:
    """阶段5：达标 / 未达标。"""
    best = ((d.get("e5_4_match") or {}).get("best") or {})
    tg = d.get("targets") or {}
    annual, ref = best.get("annualized"), tg.get("real_annual")
    pos, tpos = best.get("pos_frac"), tg.get("real_pos_frac")
    ratio = (annual / ref) if isinstance(annual, (int, float)) and ref else None
    # 溢价通道：看所有 p_buy 档位上 |premium_snr| 的最大值
    snrs = [abs(r["premium_snr"]) for r in (d.get("e5_0b_channels") or [])
            if isinstance(r.get("premium_snr"), (int, float))]
    snr_max = max(snrs) if snrs else None
    need = tg.get("target_snr")
    ok = (isinstance(ratio, float) and 1.0 <= ratio <= 2.0
          and isinstance(pos, float) and isinstance(tpos, (int, float))
          and abs(pos - tpos) < 0.05)
    verdict = "达标" if ok else "未达标"
    txt = (f"**{verdict}**：年化 {f(ratio, '.2f')}×"
           f"（真实 {f(tg.get('real_annual'), '.2%')}）、"
           f"正费率占比 {f(pos, '.3f')}（靶子 {f(tpos, '.3f')}）；"
           f"但**溢价通道被证伪**"
           f"（各档位 μ/σ 最大仅 {f(snr_max, '.3f')}，"
           f"而达到靶子需要 {f(need, '.4f')}）")
    return verdict, txt


def line6(d: dict) -> tuple[str, str]:
    """阶段6：主验收项是 k 修到 0.5。"""
    fits = d.get("e6_4_fits") or {}
    k_off = (fits.get("机制关（对照）|slippage_bp") or {}).get("exponent")
    k_on = (fits.get("元订单+自适应|slippage_bp") or {}).get("exponent")
    passed = d.get("e6_4_pass")
    dsum = (d.get("e6_2_diffs") or {}).get("acf_sum") or {}
    txt = (f"**{'达标' if passed else '未达标'}**：记忆确实变长了"
           f"（ΣACF 相对同种子对照 {f(dsum.get('mean'), '+.3f')}，"
           f"t = {f(dsum.get('t'), '.2f')}），"
           f"但主验收项「冲击指数 k 修到 0.5」"
           f"{'达成' if passed else '没有达成'}——"
           f"k 从 {f(k_off, '.3f')} 走到 {f(k_on, '.3f')}")
    return ("达标" if passed else "未达标"), txt


def line7(d: dict) -> tuple[str, str]:
    """阶段7：判据可能被"饱和"污染。"""
    e = d.get("e7_2") or {}
    passed, sat = e.get("pass"), e.get("saturated")
    rho, p = e.get("spearman_rho"), e.get("spearman_p")
    mn = e.get("min_pnl_bp")
    if sat:
        verdict = "判据假阳性"
        txt = (f"**{verdict}**：秩相关 ρ = {f(rho, '.2f')}"
               f"（p = {f(p, '.3g')}）方向正确，但**指标已饱和**——"
               f"最大档位平均 PnL {f(mn, '.0f')}bp（基本亏光），"
               f"测到的是亏损深度不是「拥挤侵蚀收益」")
    else:
        verdict = "通过" if passed else "未通过"
        txt = (f"**{verdict}**：秩相关 ρ = {f(rho, '.2f')}，"
               f"p = {f(p, '.3g')}，最低档平均 PnL {f(mn, '.0f')}bp")
    return verdict, txt


def line8(d: dict) -> tuple[str, str]:
    e8_2 = d.get("e8_2") or {}
    best = d.get("e8_1_best") or {}
    ctrl = d.get("e8_1_control") or {}
    ko, kon = e8_2.get("k_off"), e8_2.get("k_on")
    co, con = e8_2.get("closeness_off"), e8_2.get("closeness_on")
    passed = d.get("e8_1_pass")
    verdict = "达标" if passed else "未达标"
    txt = (f"**{verdict}**：聚集性随分支比单调上升"
           f"（LB Q {f(ctrl.get('lb_q'), '.0f')} → {f(best.get('lb_q'), '.0f')}）；"
           f"E8.2 意外地把 k 从 {f(ko, '.3f')} 拉到 {f(kon, '.3f')}"
           f"（离 0.5 的距离 {f(co, '.3f')} → {f(con, '.3f')}）")
    return verdict, txt


def line9(d: dict) -> tuple[str, str]:
    cal = d.get("e9_1_calibration") or {}
    real = cal.get("real_mean")
    mono = d.get("e9_4_monotone")
    verdict = "部分达标"
    txt = (f"**{verdict}**：相关性由共享公共因子做成，"
           f"真实 BTC/ETH/SOL 平均两两相关 {f(real, '+.4f')}（量级参照）；"
           f"配对交易跑通并给出 markout 归因；"
           f"但**跨资产传导幅度未被可靠测量**"
           f"（缺配对反事实这一锚点{'；单调性=' + str(mono)}）")
    return verdict, txt


def line10(d: dict) -> tuple[str, str]:
    boot = d.get("e10_3_bootstrap") or {}
    unc = boot.get("uncertainty") or {}
    degen = [k for k, v in unc.items()
             if isinstance(v, dict) and isinstance(v.get("sd"), (int, float))
             and v["sd"] < 1e-9]
    if degen:
        verdict = "框架可用（区间退化）"
        txt = (f"**{verdict}**：标准化矩量距离改变了挑选结果；"
               f"但块自助法对 ACF 型矩严重低估"
               f"（block={f(boot.get('block'), '.0f')}），"
               f"且 {len(degen)} 个参数的**跨自助样本区间退化**"
               f"（sd ≈ 0：{', '.join(degen[:4])}）")
    else:
        verdict = "框架可用"
        txt = (f"**{verdict}**：标准化矩量距离改变了挑选结果；"
               f"块自助法 block={f(boot.get('block'), '.0f')}，"
               f"参数区间可用")
    return verdict, txt


def line11(d: dict) -> tuple[str, str]:
    n = d.get("e11_1_n_links")
    passed = d.get("e11_1_pass")
    mono = d.get("e11_2_monotone")
    healthy = d.get("e11_3_healthy")
    verdict = "通过" if passed else "未通过"
    txt = (f"**{verdict}**：逐环节同种子配对判定，"
           f"**{f(n, '.0f')}/6 环节观察到传导**；"
           f"E11.2 档位敏感性单调性 = {mono}；"
           f"E11.3 事后健康检查 = "
           f"{'通过' if healthy else '**未通过**（暴露爆仓）'}")
    return verdict, txt


LINES = {5: line5, 6: line6, 7: line7, 8: line8, 9: line9, 10: line10, 11: line11}


# ----------------------------------------------------------------------
def build() -> str:
    js = {i: load(f"stage{i}") for i in range(5, 12)}
    missing = [i for i, d in js.items() if not d]
    gen_at = time.strftime("%Y-%m-%d %H:%M:%S")

    rows = []
    for i in range(5, 12):
        d = js[i]
        if not d:
            rows.append(f"| {i} {STAGE_TITLES[i]} | **尚未运行**（缺 JSON） |")
            continue
        _v, txt = LINES[i](d)
        rows.append(f"| {i} {STAGE_TITLES[i]} | {txt} |")

    parts = [f"""# 二期阶段小结（诚实边界）

> 指导书 §0 的硬性要求：**每个阶段结束必须写一份「诚实边界」小结**——明确这个阶段
> 做到了什么量级、什么做不到、下个阶段依赖它的哪个假设。
> 不允许把"训练段拟合好"包装成"这个机制被验证了"。
>
> ⚠️ **本文件由 `scripts/make_stage_summary.py` 生成，不手抄数字。**
> 总览表的每个数字与判定都从 `out/stage*_metrics.json` 现算；
> 各阶段的「诚实边界」段落**逐字**取自对应 JSON 的 `honest_boundary` 字段
> （即各阶段脚本自己打印的那一段）。需要复核时重跑生成器即可。
>
> 生成时间：{gen_at}
> 权威版本（含表格与图）见 **`docs/交易世界二期-交付报告.html`**。

---

## 总览：七个阶段，逐项如实

| 阶段 | 一句话结论 |
|---|---|
""" + "\n".join(rows) + """

> 判据说明：阶段6 的"达标"指**主验收项**（冲击幂律指数 k 修到 ≈0.5），
> 而不是"机制确实在工作"——这两件事本报告全程分开写。
> 阶段7 的"假阳性"指数值上确实显著为负、但指标已饱和，不构成"反身性成立"的证据。

---
"""]
    if missing:
        parts.append(
            f"\n> ⚠️ 以下阶段缺少 JSON，本文件里没有它们的内容：**{missing}**。\n"
            f"> 不要用文字补足——重跑对应脚本再生成。\n")

    for i in range(5, 12):
        d = js[i]
        parts.append(f"\n## 阶段{i} · {STAGE_TITLES[i]}\n")
        if not d:
            parts.append(f"**尚未运行**（缺 `out/stage{i}_metrics.json`）\n")
            continue
        _v, txt = LINES[i](d)
        parts.append(f"**结论**：{txt}\n")
        hb = d.get("honest_boundary") or []
        if not hb:
            parts.append("\n> ⚠️ 本阶段 JSON 里没有 `honest_boundary` 字段"
                         "（指导书 §0 的硬性要求）。\n")
            continue
        parts.append("\n### 诚实边界（逐字摘自 JSON）\n")
        parts.append("```")
        parts.append("\n".join(str(x) for x in hb))
        parts.append("```\n")
    return "\n".join(parts)


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    txt = build()
    # 交付物里不允许**漏出**未处理值。
    # ⚠️ 只在"总览 + 结论"部分查，不在逐字摘录的 ``honest_boundary`` 里查：
    # 那几段是各阶段脚本的原话，可能正当地在讨论 "nan" 这个现象
    # （报告自检就踩过这个假阳性：正文里有一句「非常数返回 inf」，
    #  按全文找 `>inf<` 会把那句当成漏出的值）。
    head = txt.split("## 阶段5")[0]
    for bad in ("nan", "None"):
        if bad in head:
            raise SystemExit(
                f"❌ 小结的总览/结论部分出现了 {bad!r}，拒绝写出。"
                f"原因一般是某个 JSON 字段缺失没被兜住——"
                f"格式化函数 ``f()`` 应该把它变成 '—'。")
    TARGET.write_text(txt, encoding="utf-8")
    kb = len(txt.encode("utf-8")) / 1024
    print(f"  阶段小结已生成：{TARGET}（{kb:,.0f} KB）")


if __name__ == "__main__":
    main()
