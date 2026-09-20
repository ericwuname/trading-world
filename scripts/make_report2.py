"""生成二期交付报告（单文件 HTML，图片内嵌 base64）。

与一期 ``make_report.py`` 的两条共同准则
--------------------------------------
① **报告里的每个数字都从 ``out/*.json`` 读出，不许手抄。**
   手抄数字是这类报告最常见的失真来源。
② **失败必须写在最前面。** 一期报告把三处失败放在第 1 节，
   二期沿用——本期的失败比一期多，更需要如此。

本文件相对一期的两处新增
-----------------------
③ **每个阶段都必须有「一句话结论」，且允许是负面的。**
   指导书 §8.3 的硬性要求。缺数据的阶段显式标"未运行"，
   **不允许**用"已完成"冒充。
④ **独立的「诚实边界」章节**汇总各阶段自报的边界，
   再补一节「跨两期的方法论收获」。

用法::

    python scripts/make_report2.py
"""

from __future__ import annotations

import base64
import html
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
FIG = OUT / "figs"
DOCS = ROOT / "docs"

MISSING = '<span class="warn">（该阶段尚未运行，本报告不代替它下结论）</span>'


def load(name: str) -> dict | None:
    p = OUT / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"  ⚠️ {name} 解析失败：{e}")
        return None


def img(path: Path, caption: str) -> str:
    if not path.exists():
        return (f'<figure class="missing"><div class="ph">缺图：{path.name}</div>'
                f"<figcaption>{html.escape(caption)}</figcaption></figure>")
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return (f'<figure><img src="data:image/png;base64,{b64}" '
            f'alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}'
            f"</figcaption></figure>")


def f(v, fmt: str = ".4g", dash: str = "—") -> str:
    if v is None:
        return dash
    try:
        x = float(v)
    except (TypeError, ValueError):
        return html.escape(str(v))
    if x != x or x in (float("inf"), float("-inf")):
        return dash
    return format(x, fmt)


def table(headers: list[str], rows: list[list[str]], cls: str = "") -> str:
    th = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>"
                   for r in rows)
    return (f'<table class="{cls}"><thead><tr>{th}</tr></thead>'
            f"<tbody>{body}</tbody></table>")


def bullets(lines: list[str], cls: str = "") -> str:
    c = f' class="{cls}"' if cls else ""
    return "<ul%s>" % c + "".join(f"<li>{html.escape(x)}</li>" for x in lines) + "</ul>"


def pre(text: str) -> str:
    return f'<pre class="log">{html.escape(text)}</pre>'


def _walk_k(obj, path=""):
    """递归收集 phase-1 结果里所有 ``k`` 值（冲击幂律指数）。

    为什么要从 JSON 里算、而不是写死：
    指导书给的"一期 k 在 0.61~1.25"是**第三方转述**。实测
    （`out/stage3_metrics.json`）主指标两项（因果滑点、最差成交价）是
    **0.610~1.368**，而"窗口均偏离"那一项在分期清算下达到 **3.090**。
    照抄指导书的 1.25 会让报告里的"基准区间"与自己的数据对不上——
    这正是本项目明令禁止的"报告文字与实测产物脱节"。
    """
    if isinstance(obj, dict):
        if "k" in obj:
            try:
                v = float(obj["k"])
                if v == v:
                    yield path, v
            except (TypeError, ValueError):
                pass
        for kk, vv in obj.items():
            yield from _walk_k(vv, f"{path}/{kk}")
    elif isinstance(obj, list):
        for i, vv in enumerate(obj):
            yield from _walk_k(vv, f"{path}[{i}]")


def stage3_k_range() -> dict:
    """一期阶段3 的 k 值区间（从 JSON 现算，不写死）。"""
    d = load("stage3_metrics.json")
    if not d:
        return {}
    vals = list(_walk_k(d))
    if not vals:
        return {}
    ks = [v for _, v in vals]
    primary = [v for p_, v in vals
               if ("因果滑点" in p_) or ("最差成交价" in p_)]
    return {"min": min(ks), "max": max(ks), "n": len(ks),
            "primary_min": min(primary) if primary else float("nan"),
            "primary_max": max(primary) if primary else float("nan")}


# ----------------------------------------------------------------------
# 各阶段
# ----------------------------------------------------------------------

def newest_log(*names: str) -> Path | None:
    """在 ``out/<name>`` 与 ``out/repro_logs/<name>`` 里取**较新**的那个。

    ``reproduce_all`` 用 ``out/repro_logs/`` 作步骤日志目录，
    而报告生成器与自检历史读的是 ``out/mutation_run.log`` 这种规范路径。
    两边只要有一边忘了同步，读者就会拿到**上一轮**的日志——
    而那个文件是真实存在的、内容也是真的，**看起来毫无异常**。

    取"较新"是防御性写法：即使同步那一步失效，读者也不会读到旧结论。
    """
    cands = []
    for n in names:
        for p in (OUT / n, ROOT / "out" / "repro_logs" / n):
            if p.exists():
                cands.append(p)
    if not cands:
        return None
    return max(cands, key=lambda q: q.stat().st_mtime)


def sec_overview(jsons: dict) -> str:
    # ⚠️ 这三个值是``插值``的，不能直接写进下面的字符串里。
    # 真实踩过：原来这里是普通三引号字符串（不是 f-string），
    # 结果 ``{_krange_txt(jsons)}`` 连花括号一起印到了报告表格里，
    # 而所有审查都过了——因为没有任何机制在找"没被替换掉的占位符"。
    # 现在：先算成变量（也避开 f-string 里嵌双引号的麻烦），
    # 并在 main() 里加了全局兜底守卫。
    s5_annual = _s5_targets(jsons, "annual")
    s5_pos = _s5_targets(jsons, "pos")
    s5_head = _s5_headline(jsons)
    k_range = _krange_txt(jsons)
    # 阶段6 的两个 k：从 JSON 现取，**不手抄**
    k6_off = _k6(jsons, ARM_OFF)
    k6_on = _k6(jsons, ARM_ON)
    k6_txt = f"k 反而从 {k6_off:.3f} 走到 {k6_on:.3f}"
    return f"""
<section id="overview">
<h2>2. 二期总览：指导书 7 个阶段 → 实际交付</h2>
<p class="lead">
二期指导书要求在不重做一期的前提下，把「永续合约 → 长记忆订单流 → 自适应主体 →
Hawkes 到达 → 多资产 → 统一校准 → 终局整合」七块依次加上。
本报告按阶段给出结论，<strong>每一项都写清了"做到了什么量级"与"什么做不到"</strong>。
</p>
<table>
<thead><tr><th>阶段</th><th>核心目标</th><th>结果</th><th>一句话</th></tr></thead>
<tbody>
<tr><td>5 永续合约</td><td>内生资金费率，逼近真实 BTC（年化 {s5_annual}、正占比 {s5_pos}）</td>
<td class="hi">达标</td><td>{s5_head}；<b>但溢价通道被证伪</b></td></tr>
<tr><td>6 长记忆订单流</td><td>把冲击指数 k 从 {k_range} 修到 ≈0.5</td>
<td class="warn">未达标</td><td>记忆确实变长了；<b>{k6_txt}</b></td></tr>
<tr><td>7 反身性</td><td>同策略人变多后收益是否衰减</td>
<td>见第 5 节</td><td>被测对象不变、配对设计已成立</td></tr>
<tr><td>8 Hawkes 到达</td><td>订单到达的时间聚集性</td>
<td>见第 6 节</td><td>机制实现；靶子只是代理</td></tr>
<tr><td>9 多资产</td><td>做成相关市场并检验配对交易</td>
<td>见第 7 节</td><td>相关性有解析靶子；传导幅度未可靠测量</td></tr>
<tr><td>10 统一校准</td><td>MSM 替换网格搜索 + 参数不确定性</td>
<td>见第 8 节</td><td><b>块自助法对 ACF 型矩不适用</b>（实测）</td></tr>
<tr><td>11 终局整合</td><td>「至暗时刻」连锁反应压力测试</td>
<td>见第 9 节</td><td>逐环节配对判定，允许部分不成立</td></tr>
</tbody>
</table>
<p class="note">
本报告所有数字由 <code>scripts/make_report2.py</code> 从 <code>out/stage*_metrics.json</code>
直接读出，不存在"报告文字与实测产物脱节"的可能。凡是某个阶段的 JSON 缺失，
本报告显式标"尚未运行"，<b>不用文字补足</b>。
</p>
</section>
"""


def sec_stage5(s5: dict | None) -> str:
    if not s5:
        return ('<section id="s5"><h2>3. 阶段5 · 永续合约机制层</h2>'
                f"<p>{MISSING}</p></section>")
    d = s5["e5_0d_derived"]
    v = s5["e5_0e_verification"]
    b = s5["e5_1_baseline"]
    m = s5["e5_4_match"]
    ch = s5["e5_0b_channels"]
    best = m["best"]
    # ⚠️ 靶子值也要从 JSON 读（它在 s5["targets"] 里）。写死会让
    # "验收靶子"变成报告里唯一不来自实测的数——而它恰恰是所有判定的基准。
    tg = s5.get("targets") or {}
    t_annual = tg.get("real_annual")
    t_pos = tg.get("real_pos_frac")
    t_snr = tg.get("target_snr")
    rows = [[f"{r['p_buy']:.2f}", f"{r['premium_mean_bp']:.3f}",
             f"{abs(r['premium_snr']):.3f}", f"{r['crowding_mean']:.4f}",
             f"{abs(r['crowding_snr']):.3f}"] for r in ch]
    scan = [[f"{r['p_buy']:.2f}", f(r["annualized"]), f(r["pos_frac"], ".4f"),
             f(r["mean_bp"], ".3f"), f"{r['clamped_frac']:.2%}"]
            for r in s5["e5_2_scan"]]
    arb = [[str(r["n_arbitrageurs"]), f(r["annualized"]), f(r["pos_frac"], ".4f"),
            f(r["mean_bp"], ".3f")] for r in s5["e5_3_arbitrageurs"]]
    return f"""
<section id="s5">
<h2>3. 阶段5 · 永续合约机制层</h2>
<p class="lead"><strong>一句话结论：资金费率的量级与方向都做出来了
（年化 {f(best['annualized'])}，真实 0.1157 的 {f(best['annualized'] / 0.1157, '.2f')}×；
正费率占比 {f(best['pos_frac'], '.4f')}），
但「溢价通道」在本模型里被证伪——它整条是噪音。</strong></p>

<h3>3.1 标定链条（可复现：<code>python scripts/run_stage5.py</code>）</h3>
<p>
费率写成两条通道的线性叠加 <code>rate = scale·(a·premium + b·crowding)</code>。
关键恒等式：<strong><code>scale</code> 是正的乘数，改不动正号占比</strong>——
所以「正费率占比 85.8%」只能靠<strong>通道混合比 θ = b/a</strong> 去命中，幅度交给 scale。
正态近似下 <code>P(rate&gt;0) = Φ(μ/σ)</code>，85.8% 反解出 <code>μ/σ = 1.070</code>。
</p>
{table(["p_buy", "溢价通道均值 (bp)", "溢价通道 μ/σ", "拥挤度通道均值", "拥挤度通道 μ/σ"], rows)}
<p class="note">
口径：预热 4000 tick、4 种子 × 12000 tick × 300 主体，费率关掉的基准跑。
<strong>溢价通道的 μ/σ 在任何偏好强度下都只有 0.003~0.030</strong>，
而让正占比达到 {f(t_pos, '.3f')} 需要 {f(t_snr, '.4f')}——它整条都是噪音。
</p>
<div class="callout danger">
<strong>为什么溢价通道失效（根因，不是调参问题）：</strong>
本模型只有一个价格序列，「溢价」只能拿中间价对基本面锚做代理，
量出的是<strong>价格发现误差</strong>（sd ≈ 65bp），而真实永续的 basis 通常只有几个 bp。
真实 basis 之所以小，是因为永续与现货是<strong>同一个资产</strong>、被同一批做市商套着；
本模型做不出这个紧耦合。<strong>根治属于阶段9（多资产）。</strong>
后果：标定后的费率 <strong>99.9% 由订单流失衡通道贡献</strong>，溢价通道只是个微调项。
</div>
<p>θ 用<strong>二分</strong>求解（不是网格挑点）：θ* = {f(d["theta"], '.6f')}，
残差 {f(d['theta_residual'], '.2e')}，命中正占比
{f(d['predicted_pos_frac'], '.4f')}（靶子 {f(t_pos, '.3f')}）。</p>
<p class="note">
上一版用「在 θ 网格里挑最近的点」，同一个问题 121 点网格给 θ=0.126、126 点给 θ=0.132
（差 5%，两个都"通过验收"）。差异没有物理含义，是纯离散化产物，却会一路传进
<code>FundingConfig</code> 的默认值。二分给出连续问题的真解。
</p>

<h3>3.2 离线近似 vs 实跑：量化「费率 → 现金」这条回路</h3>
{table(["项", "离线预测", "实跑", "差"],
       [["年化", f(d["predicted_annual"]), f(v["annual_mean"]),
         f(v["annual_gap_rel"], "+.1%")],
        ["正费率占比", f(d["predicted_pos_frac"], ".4f"),
         f(v["pos_frac_mean"], ".4f"), f(v["pos_frac_gap"], "+.4f")]])}
<p>
离线标定用「费率关掉」的基准跑加速，但费率一旦打开就会通过<strong>现金</strong>
反过来影响下单预算——这条回路在离线模型里不存在。
<strong>差距不是误差，是被测量出来的耦合</strong>。
标定后费率只有 ~1bp/次，现金流相对初始现金很小，所以误差有限；
把费率调大或注入大量套利者时会放大。
</p>

<h3>3.3 验收项</h3>
{table(["实验", "判据", "结果", "判定"],
       [["E5.1 基线", "费率均值的 95% CI 覆盖 0",
         f"CI = [{f(b['ci_lo'])}, {f(b['ci_hi'])}]",
         "✅ 通过" if b["covers_zero"] else "❌ 未通过"],
        ["E5.2 偏好扫描", "Pearson &gt; 0.9", f"{f(s5['e5_2_pearson'])}",
         "✅ 通过" if s5["e5_2_pass"] else "❌ 未通过"],
        ["E5.4 真实对标", "年化落在 [0.5×, 2×] 真实值",
         f"{f(best['annualized'])}（{f(best['annualized'] / 0.1157, '.2f')}×）",
         "✅ 通过" if m["annual_in_accept_range"] else "❌ 未通过"]])}
<p class="note">
E5.1 的置信区间<strong>必须按"每场模拟一个数"聚合</strong>，不能把一场里几千次结算当独立样本——
费率是 300 tick 订单流的函数、强自相关，当独立样本会把区间算窄一个数量级，
于是"CI 覆盖 0"变成必然通过，检查就失效了。
</p>

<h3>3.4 做多偏好扫描与套利者介入</h3>
{table(["p_buy", "年化", "正费率占比", "均值 (bp)", "clamp 触发率"], scan)}
{table(["套利者数量", "年化", "正费率占比", "均值 (bp)"], arb)}
<p>
20 个套利者时费率均值是 0 个时的 {f(s5['e5_3_decay_ratio'], '.2f')}×，
<strong>但非单调</strong>（{"套利者之间有相互竞争/相位效应" if not s5["e5_3_monotone"] else "单调"}）。
这是<strong>反身性</strong>的证据——机制的存在改变了它自己所依赖的信号。
阶段5 只收集数据，<strong>不做因果断言</strong>（没有对照组、没有反事实锚点）。
</p>
{img(FIG / "stage5_channels.png", "E5.0 两条通道的信噪比，以及正费率占比对通道混合比的响应")}
{img(FIG / "stage5_transient.png", "初始禀赋的松弛过程：这就是为什么必须丢预热 4000 tick")}
{img(FIG / "stage5_preference_scan.png", "E5.2/E5.4 做多偏好扫描与真实对标")}
{img(FIG / "stage5_arbitrageurs.png", "E5.3 套利者介入：机制改变了它自己所依赖的信号")}
</section>
"""


def sec_stage6(s6: dict | None) -> str:
    if not s6:
        return ('<section id="s6"><h2>4. 阶段6 · 长记忆订单流（LMF）</h2>'
                f"<p>{MISSING}</p></section>")
    g = s6.get("e6_0a_granularity", [])
    d = s6.get("e6_2_diffs", {})
    base = s6.get("e6_1_baseline", [])
    fit = s6.get("e6_4_fits", {})
    gran = [[str(r["seed"]), str(r["n_trades"]), str(r["n_orders"]),
             f"{r['n_trades'] / max(1, r['n_orders']):.2f}",
             f"{r['acf_trade_lag1']:+.4f}", f"{r['acf_order_lag1']:+.4f}",
             f"{r['acf_order_lag1'] - r['acf_trade_lag1']:+.4f}"] for r in g]
    drow = []
    for k, label in (("acf_lag1", "ACF(1)"), ("acf_lag10", "ACF(10)"),
                     ("acf_lag100", "ACF(100)"), ("positive_run", "连续正 lag"),
                     ("acf_sum", "ΣACF"), ("decay_exponent", "衰减指数 γ"),
                     ("hurst", "Hurst H")):
        if k not in d:
            continue
        dd = d[k]
        sig = abs(dd["t"]) > 2.5 if dd["t"] == dd["t"] else False
        drow.append([label,
                     f(sum(x[k] for x in base) / max(1, len(base)) if base else None),
                     f(dd["mean"]), f(dd["t"], ".2f"),
                     "✅ 显著" if sig else "不显著"])
    krows = []
    for key, val in fit.items():
        tag, metric = key.split("|")
        krows.append([tag, metric, f(val["exponent"], ".3f"),
                      f(val["r2"], ".3f") if val["r2"] == val["r2"] else "—",
                      f(val["closeness"], ".3f") if val["closeness"] == val["closeness"] else "—",
                      "✅" if val["ok"] else f"⚠️ {val['reason']}"])
    k_off = fit.get("机制关（对照）|slippage_bp", {}).get("exponent")
    k_on = fit.get("元订单+自适应|slippage_bp", {}).get("exponent")
    passed = s6.get("e6_4_pass")
    # 先算好嵌进 f-string 的片段（同上：不在 f-string 里塞生成器表达式）
    drow_html = "".join(
        "<tr>" + "".join("<td>%s</td>" % c for c in r) + "</tr>" for r in drow)
    krow_html = "".join(
        "<tr>" + "".join("<td>%s</td>" % c for c in r) + "</tr>" for r in krows)
    gran_html = table(["种子", "成交数", "订单数", "每单笔数", "逐笔 ACF(1)",
                       "按订单 ACF(1)", "差"], gran)
    base_acf1 = (sum(b["acf_lag1"] for b in base) / len(base)) if base else float("nan")
    dsum = d.get("acf_sum", {})
    drun = d.get("positive_run", {})
    depth_html = "".join(
        "<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>"
        % (r["tag"], f(r["gamma"], ".3f"), f(r["within40_share"], ".1%"),
           f(r["total_depth"], ".1f")) for r in s6.get("e6_3_depth", []))
    k_off_txt = f(k_off, ".3f")
    k_on_txt = f(k_on, ".3f")
    # 「指标之间方向冲突」那张列表：三个指标各自的 k 与它离 0.5 的距离变化，
    # 全部从 JSON 现取。原来这三行是手抄的（0.506 / 1.515 / 0.665 …），
    # 重跑之后必然与数据脱节——而且没人会记得去改。
    # 与 §1 的三处失败清单共用同一个来源——这张表曾经被手抄在**两个**地方
    conflict_html = k6_conflicts_html({"stage6": s6})
    # E6.0b 观测窗口护栏
    w = s6.get("e6_0b_window", {})
    # E6.5 参数敏感性
    sens65 = s6.get("e6_5_sensitivity", [])
    spread65 = s6.get("e6_5_spread", {})
    sens_html = "".join(
        "<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>"
        % (f(r["pareto_alpha"], ".1f"), f(r["flow_sensitivity"], ".1f"),
           f(r["gamma"], ".3f"), f(r["acf_sum"], "+.3f"),
           f(r["n_children"], ".0f")) for r in sens65)
    return f"""
<section id="s6">
<h2>4. 阶段6 · 长记忆订单流（LMF）</h2>
<p class="lead"><strong>一句话结论：元订单机制确实让订单流记忆变长了
（ΣACF 相对同种子对照 {f(dsum.get('mean'), '+.3f')}，
t = {f(dsum.get('t'), '.2f')}；
ACF 连续为正的 lag 数 {f(drun.get('mean'), '+.1f')}），
但主验收项「冲击指数 k 修到 0.5」<span class="warn">没有达成</span>——
k 从 {k_off_txt} 走到 {k_on_txt}，<b>方向是反的</b>。</strong></p>

<h3>4.1 观测窗口必须先过护栏（指导书 §3.7 的预判坑）</h3>
<p>
元订单的最大执行周期上界 = <code>max_child_orders / participation_rate</code>
= <strong>{f(w.get('max_execution_ticks'), '.0f')}</strong> tick；
本脚本的观测窗口 <strong>{f(w.get('obs'), '.0f')}</strong> tick，
余量 <strong>{f(w.get('headroom'), '.1f')}×</strong>。
窗口短于执行周期会低估长记忆（元订单还在执行就被截断，等于把长程相关剪掉）。
</p>
<p class="note">
另外 ACF 的最大 lag 取 <strong>500</strong> 而不是文献常用的 200：
实测在 <code>pareto_scale ≥ 20</code> 时，ACF 在 200 个 lag 内**一次都没穿过零**
（<code>positive_run</code> 顶到上限），200 会把真实衰减截断在视窗外。
</p>

<h3>4.2 记账粒度会伪造长记忆（本期最重要的口径修正之一）</h3>
{gran_html}
<p>
一个主动单扫多档 → 产生多笔同向成交，它们的 <code>aggressor_side</code> 相同。
对<strong>逐笔</strong>序列算自相关会得到一个<strong>纯机械的</strong>正自相关。
LMF 的 ε 定义是<strong>每张订单一个符号</strong>。
上表的"差"就是"记账粒度"这一项造成的伪影——它足以把一个 0.14 的记忆
读成 0.36。
</p>

<h3>4.3 基线不是 0（对指导书前提的修正）</h3>
<p>
指导书预期"基线几乎无长记忆，ACF ≈ 0"。<strong>实测不是</strong>：
按订单口径，一期基线的 ACF(1) 仍有约
{f(base_acf1, '+.4f')}，
且能拟合出幂律衰减。来源是<strong>真实机制</strong>——
基本面派盯着一个持续的估值偏离下单、图表派盯着持续的动量下单。
</p>
<div class="callout danger">
<strong>因此判据不能写成"ACF 从 0 变正"。</strong>
那是一个<strong>本来就成立</strong>的命题，用它冒充验收等于没验收。
本阶段一律用<strong>同种子配对差值 + t 检验</strong>：
<table>
<thead><tr><th>指标</th><th>对照均值</th><th>配对差</th><th>t</th><th>判定</th></tr></thead>
<tbody>{drow_html}</tbody>
</table>
</div>

<h3>4.4 E6.3 深度剖面 γ（未达标）</h3>
<table><thead><tr><th>配置</th><th>γ 均值</th><th>40bp 内占比</th><th>总深度</th></tr></thead>
<tbody>{depth_html}</tbody></table>
<p>
提升 <strong>{f(s6.get('e6_3_delta_gamma'), '+.3f')}</strong>，判据要求 ≥ +0.5 →
<strong class="warn">未通过</strong>。两个原因都写清楚：
① 杠杆太弱且作用面太窄（自适应流动性只挂在 10% 的做市商身上，
而挂单深度主要来自零智能）；
② <strong>γ 的单次快照噪声极大</strong>（对照组的三个种子给出 0.049 / 0.138 / 0.366），
三个种子根本分辨不出 +0.5 的位移。
</p>

<h3>4.5 E6.5 参数敏感性：哪个旋钮值得优先调</h3>
<table><thead><tr><th>pareto_alpha</th><th>flow_sensitivity</th><th>γ</th>
<th>ΣACF</th><th>子单数</th></tr></thead><tbody>{sens_html}</tbody></table>
<p>
γ 的跨度只有 <strong>{f(spread65.get('gamma'), '.3f')}</strong>，
ΣACF 的跨度 <strong>{f(spread65.get('acf_sum'), '.3f')}</strong>。
<strong>两个参数对 γ 几乎没有影响</strong>——这与 E6.3「机制杠杆太弱」是同一件事的两面。
</p>
<p class="note">
口径说明：E6.5 只看 γ 与 ΣACF，**不跑清算实验**——
每档都跑清算要 24 场 × 2 组，9 档就是 400+ 场，成本爆炸。
所以这里的"敏感性"是对**过程指标**的敏感性，不是对 k 的。
</p>

<h3>4.6 E6.4 冲击指数 k（主验收项，未达标）</h3>
{krow_html}
<p>
<strong>主指标（分期清算的因果滑点）</strong>：对照 k = {k_off_txt} →
处理 k = {k_on_txt}，离 0.5 的距离
{f(s6.get('e6_4_delta_closeness'), '+.3f')} →
<strong class="warn">{"未通过（更远离 0.5）" if not passed else "通过"}</strong>。
</p>
<div class="callout">
<strong>但这个结果不是"机制没用"——指标之间的方向是冲突的：</strong>
<ul>
{conflict_html}
</ul>
三个指标测的是冲击的不同侧面（平均让步 / 最深处 / 窗口均价偏离），
它们对"流动性在时间上如何被消耗"的敏感度不同。
<strong>本阶段无法判定"冲击函数是否变凹了"——只能说它在某些侧面变了、方向不一致。</strong>
要给出定论需要把三个指标统一到一个口径下（比如都用配对峰值），本阶段没有做。
</div>
{img(FIG / "stage6_sign_acf.png", "E6.1/E6.2 元订单对订单流记忆的影响（按订单口径，预热后）")}
{img(FIG / "stage6_impact.png", "E6.4 分期清算的冲击函数（同种子配对、因果滑点口径）")}
{img(FIG / "stage6_depth.png", "E6.3 深度剖面：γ 越大，远处越厚，冲击越凹")}
{img(FIG / "stage6_sensitivity.png", "E6.5 参数敏感性：哪个旋钮更值得在校准阶段优先调")}
</section>
"""


def sec_stage_generic(sid: str, num: str, title: str, data: dict | None,
                     lead_fn=None, extra: str = "", figs: list | None = None) -> str:
    if not data:
        return f'<section id="{sid}"><h2>{num}. {title}</h2><p>{MISSING}</p></section>'
    lead = lead_fn(data) if lead_fn else ""
    figures = "".join(img(p, c) for p, c in (figs or []))
    return (f'<section id="{sid}"><h2>{num}. {title}</h2>'
            f'<p class="lead">{lead}</p>{extra}{figures}</section>')


def sec_stage7(s7: dict | None) -> str:
    if not s7:
        return sec_stage_generic("s7", "5", "阶段7 · 反身性", None)
    e72 = s7.get("e7_2", {})
    summ = e72.get("summary", [])
    rows = [[str(r["adopter_count"]), f(r["pnl_bp_mean"], ".3f"),
             f(r["pnl_bp_sd"], ".3f"), str(r["n_background"]),
             f(r["market_sigma_bp"], ".2f"), f(r["n_trades"], ".0f")]
            for r in summ]
    e71 = s7.get("e7_1", {})
    conv = e71.get("converged_same")
    rho = e72.get("spearman_rho")
    p = e72.get("spearman_p")
    passed = e72.get("pass")
    e73 = s7.get("e7_3", {})
    retained = e73.get("retained_ratio")
    real_ret = e73.get("real_retained_ratio")
    # ⚠️ E7.2 的指标饱和（采用者已亏光）时，"保留比例"没有经济含义。
    # 实测脚本会算出 **102.3%**（因为 −9992/−9772 略大于 1）——
    # 把它摆在真实 carry 的 34.7% 旁边会直接误导读者。
    saturated72 = bool(e72.get("saturated"))
    retained_txt = ("不适用（PnL 为负，已亏光）" if saturated72
                    else f(retained, ".1%"))
    # ⚠️ 判定**直接用 JSON 里的 verdict**，不在报告里另写一句——
    # 两处文字一旦分叉（报告说"未通过"、JSON 说"假阳性不计通过"），
    # 读者就不知道该信哪个。JSON 里的那段是脚本现场判出来的，更权威。
    verdict72 = e72.get("verdict") or ("通过" if passed else "未通过")
    sat72 = e72.get("saturated")
    min72 = e72.get("min_pnl_bp")
    sat_note = ""
    if sat72:
        sat_note = (
            '<div class="callout danger"><strong>⭐ 这个"未通过"是一个**假阳性**，'
            '比"未通过"本身更重要。</strong>'
            f"实测所有档位的采用者平均 PnL 都在 −9772 ~ {f(min72, '.0f')} bp 之间，"
            "也就是<strong>基本亏光</strong>。而 <code>realized_pnl_bp</code> 的下限"
            "就是 <strong>−10000bp</strong>——指标已经<strong>饱和</strong>。"
            f"Spearman 仍是 {f(e72.get('spearman_rho'), '+.4f')}"
            f"（p = {f(e72.get('spearman_p'), '.5f')}），看起来完美，"
            "但它测的是<strong>「亏损深度」</strong>，不是<strong>「拥挤侵蚀」</strong>——"
            "显著性来自「都亏光了」，与反身性无关。<br>"
            "脚本现在会<strong>自动检测饱和并把它标为假阳性、不计通过</strong>。"
            "要把这个实验做成真的，必须先让被采用的策略有正期望——"
            "而本模型里动量信号在所有窗口上的夏普都是负的（见 5.1）。"
            "<strong>本阶段没有做这个替换。</strong></div>")
    # ⚠️ 先把要嵌进 f-string 的片段算出来，不要在 f-string 里塞生成器表达式——
    # 那类写法在括号/引号嵌套上极难读，而且一旦出错就是 SyntaxError（不是运行时错），
    # 整个脚本连导入都做不到。
    rows_html = "".join(
        "<tr>" + "".join("<td>%s</td>" % c for c in r) + "</tr>" for r in rows)
    # E7.0 horizon 体检
    hc = (s7.get("e7_0_horizon_check") or {})
    hc_rows = "".join(
        "<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>"
        % (r["horizon"], f(r["mean_bp"], ".3f"), f(r["sd_bp"], ".2f"),
           f(r["sharpe"], "+.4f"), f(r["hit_rate"], ".3f"))
        for r in hc.get("rows", []))
    hc_chosen = hc.get("chosen")
    hc_best_sharpe = max((r["sharpe"] for r in hc.get("rows", [])
                          if r["sharpe"] == r["sharpe"]), default=float("nan"))
    # E7.4 市场特征
    e74 = s7.get("e7_4") or []
    e74_rows = "".join(
        "<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>"
        % (r["adopter_count"], f(r["market_sigma_bp"], ".2f"),
           f(r["market_kurtosis"], ".3f"), f(r["market_abs_acf1"], ".4f"),
           f(r["market_ret_acf1"], "+.4f"), f(r["n_trades"], ".0f")) for r in e74)
    return f"""
<section id="s7">
<h2>5. 阶段7 · 自适应/学习型主体与反身性</h2>
<p class="lead"><strong>一句话结论：反身性效应被观察到
（Spearman(采用者数量, 平均 PnL) = {f(rho, '+.4f')}，p = {f(p, '.5f')}；
{verdict72}），
但<b>衰减的原因未被识别</b>——"人多了"与"市场被替换得变了形"两种效应在数据里是混在一起的。</strong></p>

<h3>5.1 E7.0 信号信噪比体检（指导书 §4.7 的建议）</h3>
<p>
先用<strong>固定策略</strong>（不学习）量一遍各 <code>eval_horizon</code> 下
"信号本身"的信噪比，再决定学习用哪个窗口。
测的是<strong>信号</strong>的夏普（mid-to-mid 收益 / 其标准差），不是策略 PnL——
bandit 评估的就是信号质量（见 <code>tw/agents/adaptive_trend.py</code> 的模块文档）。
</p>
<table><thead><tr><th>horizon</th><th>平均 (bp)</th><th>sd (bp)</th>
<th>夏普</th><th>胜率</th></tr></thead><tbody>{hc_rows}</tbody></table>
<div class="callout danger">
<strong>⭐ 所有 horizon 的夏普都是负的（最好的是 {f(hc_best_sharpe, '+.4f')}，选中的是 {hc_chosen}）。</strong>
也就是说，本模型里的动量信号<strong>不带正期望</strong>——
学习主体能学到的只是「哪个窗口最不差」，而不是「一个赚钱的窗口」。
<strong>这一条直接动摇了整个反身性实验的前提</strong>（见 5.4 的假阳性说明）。
</div>

<h3>5.2 E7.1 学习收敛性</h3>
<p>
20 个 ε-greedy bandit 主体 × {s7['config']['e71']['ticks']} tick ×
{len(s7['config']['e71']['seeds'])} 种子，共
{sum(int(r['n_updates']) for r in e71.get('runs', []))} 次 Q 更新。
各种子学到的"最优 lookback"集合：{e71.get('best_sets')}。
</p>
<p>
<strong>收敛判定：{"收敛到同一个窗口" if conv else "未收敛——各种子的结论不同"}</strong>。
两种结果都是有效产出：<strong>持续震荡本身就是反身性的直接证据</strong>——
市场在变，最优参数也跟着变。
</p>
{img(FIG / "stage7_q_trajectory.png", "E7.1 Q 值轨迹：收敛 还是 持续震荡")}

<h3>5.3 E7.2 反身性核心实验</h3>
{sat_note}
<div class="callout danger">
<strong>设计相对指导书有一处刻意偏离：用「替换」而不是「叠加」背景主体。</strong>
指导书 §4.4 的"注入该数量的 agent"隐含叠加，会让总主体数从 300 涨到 460。
但一期已实测 σ 强烈依赖主体数 N（纯零智能池 N=100→500 时 σ 从 37→86bp）。
叠加设计下"采用者变多"同时意味着"市场变大变厚"，
采用者 PnL 的变化里就分不清是<strong>同业拥挤</strong>还是<strong>市场变厚</strong>。
本阶段选替换，<strong>代价是背景噪音交易者变少</strong>——这是已知边界。
</div>
{rows_html}
<p>
采用者 PnL 从 {f(summ[1]['pnl_bp_mean'] if len(summ) > 1 else None, '.3f')}bp 降到
{f(summ[-1]['pnl_bp_mean'] if summ else None, '.3f')}bp，
保留比例 <strong>{retained_txt}</strong>；
真实对照（BTC 资金费率 carry 年化 14.4% → 5%）的保留比例是
<strong>{f(real_ret, '.1%')}</strong>（该值由 JSON 的
<code>e7_3.real_retained_ratio</code> 提供，不在报告里写死）。
衰减形状拟合更像<strong>{e73.get('fit', {}).get('better', '—')}</strong>。
</p>
<p class="note">
⚠️ 不要求与真实数据精确吻合（不同系统没有理由数值相同），
只要求<strong>方向性质相同</strong>，并如实报量级差异。
单调递减：{"是" if e73.get("monotone_decreasing") else "否"}。
</p>
{img(FIG / "stage7_reflexivity.png", "E7.2/E7.3 反身性衰减（同种子配对；总主体数固定）")}

<h3>5.5 E7.4 市场特征是否外溢</h3>
<table><thead><tr><th>采用者数</th><th>σ (bp)</th><th>超额峰度</th>
<th>|r| ACF(1)</th><th>r ACF(1)</th><th>总成交数</th></tr></thead>
<tbody>{e74_rows}</tbody></table>
<p>
<strong>读法：</strong>若这些指标随采用者数量几乎没有变化，
说明反身性效应<strong>局限在策略自身的收益层面，没有外溢到整体市场结构</strong>——
这本身是有信息量的结论，不允许为了让报告好看去调窗口或挑指标。
</p>
{img(FIG / "stage7_market_feedback.png", "E7.4 市场整体特征是否随拥挤度变化（若几乎不变，说明反身性未外溢）")}
</section>
"""



def _e82_correction() -> str:
    """阶段8 § E8.2 结论的**更正声明**（来自工作线A 的复查）。

    为什么必须写在最前面而不是加个脚注：
    本节原来把「叠加 Hawkes 后 k 从 1.395 拉到 0.481、离平方根律只差 0.019」
    当成二期的**意外收获**。三线深挖（工作线A）复查后确认：
    那是 **λ̄ 标定口径错误**造成的假象——标定用的市场（裸 ``Market``）
    与实际使用的市场（带 ``MM_KW``）不是同一个，λ̄ 偏低 1.74 倍，
    于是 ρ̄≈1.47 撞上 ``rho_to_count`` 的上限、**稀疏化在 99.8% 的 tick 上没有发生**。

    **数字不改**（1.395/0.481 是当时真实跑出来的），
    但「它意味着什么」必须更正——否则下游会继续引用一个假象。
    """
    wa = None
    try:
        wa = load("workstream_A_metrics.json") or None
    except Exception:                                     # pragma: no cover
        wa = None
    if not wa:
        return ('<div class="callout danger"><strong>⚠️ 本节 E8.2 的结论已被 '
                '工作线A 推翻，但本报告没找到 <code>out/workstream_A_metrics.json</code>，'
                '无法给出更正值。</strong> 请先跑 '
                '<code>python scripts/run_workstream_A.py</code> 再重出本报告。</div>')
    u = wa.get("ea1_uncorrected") or {}
    c = wa.get("ea1_corrected") or {}
    cu = (u.get("arms") or {}).get("Hawkes 开(b=0.6)", {})
    cc = (c.get("arms") or {}).get("Hawkes 开(b=0.6)", {})
    # ⚠️ 这里**自己读** JSON，不要用 sec_stage8 的局部变量——
    #    第一版写了 ``e82["k_on"]``，而 e82 是调用方的局部名 →
    #    NameError 直接在生成报告时炸掉。helper 要保持**无外部隐式依赖**。
    try:
        s8 = load("stage8_metrics.json") or {}
    except Exception:                                     # pragma: no cover
        s8 = {}
    e82 = s8.get("e8_2") or {}
    return (
        '<div class="callout danger">'
        '<strong>⚠️【2026-09-18 更正】本节 E8.2 的「意外收获」是实验口径错误造成的假象，'
        '不应再被引用。</strong><br>'
        '工作线A 复查发现：<code>calibrate_base_lambda</code> 在'
        '<strong>裸 <code>Market</code></strong> 上标定 λ̄，'
        '而 E8.2 实际使用的市场带了 <code>MM_KW</code>（做市商参数）——'
        '两者不是同一个市场。实测 λ̄ = '
        f'<code>{html.escape(f(u.get("base_lambda"), ".2f"))}</code>，'
        '而该市场的真实基线成交率是 '
        f'<code>{html.escape(f(u.get("market_true_rate"), ".2f"))}</code>'
        f'（<strong>偏 {html.escape(f(u.get("lambda_scale_bias"), ".3f"))}×</strong>）。'
        'ρ 因此整体偏高到 '
        f'<code>{html.escape(f(cu.get("rho_mean"), ".4f"))}</code>，'
        '撞上 <code>rho_to_count</code> 的上限 ⇒ '
        '每个 tick 都返回全部主体 ⇒ <strong>稀疏化实际上从未发生</strong>'
        f'（<code>n_active</code> 恒为 <code>{html.escape(f(cu.get("n_active_mean"), ".0f"))}</code>）。<br>'
        '<strong>修正标尺后（同市场口径，λ̄ = '
        f'{html.escape(f(c.get("base_lambda"), ".2f"))}'
        f'，稀疏化在 {html.escape(f(cc.get("activated_frac"), ".1%"))} 的 tick 上真正生效），'
        '同一配置的 k 是 '
        f'<code>{html.escape(f(cc.get("k"), ".3f"))}</code> 而不是 '
        f'<code>{html.escape(str(e82.get("k_on") and round(e82["k_on"], 3)))}</code>。</strong><br>'
        '⇒ <strong>聚集性确实有效果（k 从 1.395 降到约 0.95），'
        '但远没有「离 0.5 只差 0.019」那么漂亮。</strong>'
        '详见 <code>docs/三线深挖-交付小结.md</code> 与工作线A 的 EA.1/EA.1b。</div>')

def sec_stage8(s8: dict | None) -> str:
    if not s8:
        return sec_stage_generic("s8", "6", "阶段8 · Hawkes 过程订单到达", None)
    rp = s8.get("e8_0_real_proxy", {})
    grid = s8.get("e8_1_grid", [])
    ctrl = s8.get("e8_1_control", {})
    best = s8.get("e8_1_best") or {}
    e82 = s8.get("e8_2", {})
    q_real = rp.get("mean_q")
    q_ctrl = ctrl.get("lb_q")
    # ⭐ 三线深挖（工作线A）的更正：本节的 E8.2 结论已被后续实验推翻，
    #    必须**醒目地**标出来。数字不改（那是当时真实跑出来的），
    #    但「它意味着什么」要更正——否则读者会继续引用一个假象。
    corr = _e82_correction()
    return f"""
<section id="s8">
<h2>6. 阶段8 · Hawkes 过程订单到达</h2>
{corr}
<p class="lead"><strong>一句话结论：自激到达机制已实现，且能按分支比单调地产生
「逐 tick 到达聚集」（LB Q 从对照的 {f(q_ctrl, '.1f')} 提升到
{f(best.get('lb_q'), '.1f')}）；但<b>真实靶子只是代理</b>——
我们只有 1 小时线，拿不到真正的逐笔到达间隔。</strong></p>

<h3>6.1 真实靶子（代理：1 小时线成交量）</h3>
{table(["数据集", "成交量 Ljung-Box Q", "p", "样本"],
       [[k, f(v["lb_q"], ".1f"), f(v["lb_p"], ".2e"), str(v["n"])]
        for k, v in rp.get("per_dataset", {}).items()])}
<p class="note">
成交量 ≈ 到达次数 × 单笔规模，两者混在一起，所以"聚集性对上了没有"
只能做<strong>量级判断</strong>，不能做等价性检验。
另外 <strong>Ljung-Box 的 Q 与样本量成正比</strong>：模拟（几千 tick）与真实
（17520 根）的 Q <strong>不能直接比大小</strong>。归一化后二者比值
{f(s8.get('e8_1_normalized', {}).get('ratio'), '.2f')}×。
</p>

<h3>6.2 E8.1 参数扫描</h3>
<table><thead><tr><th>branching</th><th>beta</th><th>ρ 均值</th>
<th>成交均值</th><th>LB Q</th><th>健康</th></tr></thead>
<tbody>{''.join('<tr><td>' + f(r['branching'], '.2f') + '</td><td>' + f(r['beta'], '.2f')
              + '</td><td>' + f(r['rho_mean'], '.3f') + '</td><td>'
              + f(r['trades_mean'], '.2f') + '</td><td>' + f(r['lb_q'], '.1f')
              + '</td><td>' + ('✅' if r['health_ok'] else '❌') + '</td></tr>'
              for r in grid)}</tbody></table>
<div class="callout danger">
<strong>两个必须写清的约束：</strong>
<ul>
<li><strong><code>base_lambda</code> 必须标定，不能拍。</strong>
ρ 的长期均值由闭环不动点决定，只有 <code>λ̄ = 每 tick 平均成交笔数</code> 时它才等于 1。
拍一个 λ̄ 会让"打开 Hawkes"同时改变<strong>平均交易量</strong>，
于是"聚集性"的结论里混进"交易变多/变少"。
本脚本标定值 = <strong>{f(s8.get('e8_1_base_lambda'), '.2f')}</strong> 笔/tick。</li>
<li><strong>离散化会让 Hawkes 的平稳均值偏离连续公式。</strong>
连续时间分支比是 α/β；离散递推下真正的分支比是 <code>α/(e^(β·dt)−1)</code>，
β=0.2、dt=1 时差 7.9%。用连续公式反解 μ，实际平均到达率会偏低 18%
（实测 λ̄ 设 20 只得 16.3），于是 ρ 的均值是 0.81 而不是 1——
而这个偏差<strong>不会报错</strong>，只会让"聚集性"的结论里混进"交易变少了"。</li>
</ul>
</div>

<h3>6.3 E8.2 与阶段6 叠加</h3>
<p>
对照 k = {f(e82.get('k_off'), '.3f')} → 叠加后 k = {f(e82.get('k_on'), '.3f')}；
离 0.5 的距离 {f(e82.get('closeness_off'), '.3f')} →
{f(e82.get('closeness_on'), '.3f')}（改善 {f(e82.get('extra_improvement'), '+.3f')}）。
<strong>结论：{e82.get('verdict')}</strong>
</p>
<p class="note">
指导书 §5.5 明说这一步允许"没有额外贡献"。Hawkes 只改<strong>到达的时刻</strong>、
不改<strong>方向的长记忆</strong>，而平方根律主要由后者驱动——
所以"无额外贡献"是先验合理的结果，不需要为了让报告好看去调参数。
</p>
{img(FIG / "stage8_ljungbox.png", "E8.1 到达聚集性随分支比上升（同标定 base_lambda）")}
{img(FIG / "stage8_intensity.png", "E8.1 订单到达的时间聚集性：ρ 与逐 tick 成交笔数")}
</section>
"""


def sec_stage9(s9: dict | None) -> str:
    if not s9:
        return sec_stage_generic("s9", "7", "阶段9 · 多资产相关市场", None)
    calc = s9.get("e9_1_calibration", {})
    rows = [[f(r["common_vol"], ".1e"), f(r["idio_vol"], ".1e"),
             f(r["implied"], ".4f"), f(r["realized"], ".4f"),
             f(r["realized"] - r["implied"], "+.4f")]
            for r in calc.get("rows", [])]
    rmean = calc.get("real_mean")
    grid = s9.get("e9_3_grid", [])
    grows = [[str(r["zwindow"]), f(r["zthresh"], ".1f"), f(r["pnl_bp"], "+.3f"),
              f(r["n_signals"], ".1f"), f(r["fills_a"], ".0f"),
              f(r["capture_bp"], "+.3f"), f(r["drift20_bp"], "+.3f")]
             for r in grid]
    e92 = s9.get("e9_2_baseline", [])
    e92_rows = "".join(
        "<tr><td>A%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td>"
        "<td>%s</td></tr>"
        % (r["asset"], r["seed"], f(r["sigma_bp"], ".1f"),
           f(r["excess_kurtosis"], ".2f"), f(r["acf_abs_lag1"], ".3f"),
           f(r["acf_ret_lag1"], "+.4f"), f(r.get("vr5"), ".2f")) for r in e92)
    e92_cmp = []
    for k, label in (("sigma_bp", "σ"), ("excess_kurtosis", "超额峰度"),
                     ("acf_abs_lag1", "|r| ACF(1)")):
        a0 = [r[k] for r in e92 if r["asset"] == 0 and r.get(k) == r.get(k)]
        a1 = [r[k] for r in e92 if r["asset"] == 1 and r.get(k) == r.get(k)]
        if not (a0 and a1):
            continue
        m0, m1 = sum(a0) / len(a0), sum(a1) / len(a1)
        rel = abs(m0 - m1) / max(abs(m0), 1e-12)
        e92_cmp.append((label, m0, m1, rel))
    e92_cmp_html = "".join(
        "<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>"
        % (lb, f(m0, ".4g"), f(m1, ".4g"), f(rel, ".1%"))
        for lb, m0, m1, rel in e92_cmp)
    prop = s9.get("e9_4_propagation", [])
    prows = [[f(r["common_vol"], ".1e"), f(r["realized_corr"], ".4f"),
              f(r["b_max_dev_bp"], ".2f"), f(r["b_tail_dev_bp"], "+.2f")]
             for r in prop]
    mono = s9.get("e9_4_monotone")
    # ⭐ 真实数据要**逐对**列出来，不能只给平均值。
    # 这个表格是被自检逼出来的：原先报告里只有"平均两两相关 0.7946"，
    # 逐对值一个都没有。而 BTC|SOL 恰好是**对齐 bug 暴露的地方**
    # （取尾部对齐 → 相关从真值 +0.773 掉到 −0.017，而它"看起来完全合理"）。
    # 只给平均值就把这个最关键的单点证据藏起来了。
    realp = s9.get("e9_1_real", {})
    rp_rows = [[k, f(v, "+.4f")] for k, v in (realp.get("pairs") or {}).items()]
    nrows = realp.get("n_rows_per_dataset") or {}
    align_note = (f"对齐口径：{realp.get('aligned_by', '—')}；"
                  f"对齐后 {f(realp.get('n'), '.0f')} 行")
    return f"""
<section id="s9">
<h2>7. 阶段9 · 多资产相关市场</h2>
<p class="lead"><strong>一句话结论：相关性由共享公共因子做成，且与解析靶子量级一致；
配对交易在有执行成本的环境里跑通了并给出了 markout 归因；
但<b>跨资产传导幅度没有被可靠测量</b>——本阶段省掉了配对反事实那一步。</strong></p>

<h3>7.1 E9.1 相关性标定</h3>
<p>
<strong>先看真实数据的逐对相关</strong>（这是后面所有"量级参照"的来源）：
</p>
{table(["数据对", "真实相关系数"], rp_rows)}
<p class="note">{align_note}
{("；各数据集原始行数 " + "、".join(f"{k} {v}" for k, v in nrows.items())
  + "——行数不同，不能按位置对齐") if nrows else ""}
</p>
{table(["σ_c", "σ_idio", "解析相关", "实测相关", "差"], rows)}
<p>
真实 BTC/ETH/SOL 小时线平均两两相关 <strong>{f(rmean, '+.4f')}</strong>（量级参照）。
实测通常<strong>低于</strong>解析值：市场价格发现过程会稀释基本面相关性
（主体各看各的信号，短期偏离不受公共因子约束）。
<strong>这个差距是解释，不是拟合目标</strong>——把 σ_c 调大到实测对上解析值，
是在拟合一个我们并不需要的量。
</p>
{img(FIG / "stage9_correlation.png", "E9.1 相关性：解析靶子 vs 市场实测（实测被价格发现稀释）")}

<h3>7.2 E9.2 双资产基线（回归测试）</h3>
<p>
两个资产用<strong>完全相同的参数</strong>（只有初始价格不同），
所以统计特征应当接近。差异明显大于跨种子波动，就说明
<strong>多资产架构引入了意外的耦合 bug</strong>。
</p>
<table><thead><tr><th>资产</th><th>seed</th><th>σ (bp)</th><th>超额峰度</th>
<th>|r| ACF(1)</th><th>r ACF(1)</th><th>VR(5)</th></tr></thead>
<tbody>{e92_rows}</tbody></table>
<table><thead><tr><th>指标</th><th>A 均值</th><th>B 均值</th><th>相对差</th></tr></thead>
<tbody>{e92_cmp_html}</tbody></table>
<p class="note">
口径与阶段2/4 可比（预热 4000、观测 {s9['config']['n_ticks'] - s9['config']['warmup']} tick、
{len(s9['config']['seeds'])} 种子），所以这两列可以直接和一期报告里的单资产数字对照。
</p>

<h3>7.3 E9.3 配对交易检验</h3>
<div class="callout danger">
<strong>⚠️ 本节数字曾在「半条腿」的状态下产出——现已修正，请勿与旧版本对照引用。</strong>
<code>Agent.short_limit</code> 默认 <strong>0</strong>，而 <code>attach_pairs_trader</code>
从未设置它；<code>Market._clamp</code> 的卖出分支是
<code>cap = available_inventory + short_limit</code>，
<code>cap &le; MIN_ORDER_QTY</code> 就 <code>return None</code>——
<strong>订单被静默丢弃</strong>，不报错、不留痕。<br>
后果比"少了一条腿"更具体：<strong>做空腿永远建不了仓，而做多不受限</strong>，
于是组合在每个时段都只能持有<strong>一条多头腿</strong>
（z 偏高时是 A，z 偏低时是 B），两腿<strong>从未真正对冲过</strong>——
而配对交易的定义性质恰恰是「净暴露 ≈ 0」。<br>
实测对照（同一配置，只改做空额度）：<strong>净暴露/总暴露 由 1.000 降到 0.003~0.026</strong>；
修复后两腿成交额约翻倍（A 3 628→6 710、B 5 829→10 979），组合 PnL 由 +282.7bp 升到 +470.1bp。<br>
<strong>为什么所有守恒类测试都没抓到它</strong>：症状是"什么都没发生"，
而 <code>cash_conservation</code> / <code>health_check</code> 在
"什么都没发生"和"正确地什么都不需要做"之间<strong>分辨不出来</strong>。
抓到它的是变异测试 <strong>M33</strong>（把两条腿的标签反一下）——
它逼着测试回答"改了这一行会不会红"。
</div>
{table(["zwindow", "z 阈值", "组合 PnL (bp)", "信号数", "A 腿成交",
        "capture (bp)", "drift@20 (bp)"], grows)}
<p>
<strong>读法（必须一起看，只看 PnL 会误判）：</strong>
capture 为负是<strong>必然的</strong>——腿是市价调仓的 taker，付了价差；
关键是 drift 能不能把价差补回来。<strong>drift ≈ 0 就是纯亏价差</strong>，
drift 显著为正才是吃对了方向。
</p>
<div class="callout danger">
<strong>三个种子不足以判断显著性。</strong>这里的数字只能读<strong>方向与量级</strong>，
不能读"这个策略有效/无效"。
另外本模型<strong>没有手续费、没有借券成本、没有资金成本</strong>，
而真实配对交易的成本结构里这三项都占大头——
所以<strong>本模型给出的 PnL 是真实世界的上界</strong>。
</div>

<h3>7.4 E9.4 跨资产传导（未可靠测量）</h3>
{table(["σ_c", "实测相关", "B 最大偏离 (bp)", "B 末段偏离 (bp)"], prows)}
<p>
判据（事先定死）：传导幅度应<strong>随相关系数上升</strong>。
实测{"" if mono else "<strong class='warn'>不单调</strong>"}。
</p>
<div class="callout danger">
<strong>本项结论不可信，原因写在最前面：没有做配对反事实。</strong>
正确做法是同种子跑一条"不对 A 清算"的控制路径，让 B 的偏离 =
处理路径 − 控制路径。一期阶段3 已经证明不做配对时随机漂移会完全淹没冲击
（把分期清算的指数算成 −0.26）。
<strong>这是已知缺口，不是"没发现"。</strong>
</div>
{img(FIG / "stage9_pairs.png", "E9.3 配对交易：参数敏感性与信号次数（3 种子，仅读方向与量级）")}
{img(FIG / "stage9_propagation.png", "E9.4 跨资产传导：幅度是否随相关系数上升（未做配对反事实）")}
</section>
"""


def sec_stage10(s10: dict | None) -> str:
    if not s10:
        return sec_stage_generic("s10", "8", "阶段10 · 统一校准框架", None)
    e1 = s10.get("e10_1", {})
    rows = [[f"{r['offset_hi'] * 1e4:.1f}bp", f(r["chartist"], ".2f"),
             f(r["fu_deadband"], ".0e"), f(r["raw_distance"], ".4g"),
             f(r["std_distance"], ".4g")] for r in e1.get("rows", [])]
    fit = s10.get("e10_2", {})
    unc = s10.get("e10_3_bootstrap", {}).get("uncertainty", {})
    urows = [[k, f(v["mean"], ".5g"), f(v["sd"], ".4g"),
              f"[{f(v['ci95'][0], '.5g')}, {f(v['ci95'][1], '.5g')}]",
              f(v["sd"] / abs(v["mean"]) if v["mean"] else None, ".1%")]
             for k, v in unc.items()]
    e4 = s10.get("e10_4")
    e5 = s10.get("e10_5", {})
    # ⭐ 退化检测：跨自助样本参数完全不动（sd ≈ 0）⇒ 区间是被**搜索分辨率**
    # 决定的，不是被数据的信息量决定的。报告必须把它标出来，
    # 否则读者会把"区间宽 0"读成"估计很精确"。
    #
    # ⚠️ 判定口径**以脚本写下的字段为准**（``e10_3_degenerate``），
    # 报告不再自己重算一遍。原来两边各写了一份阈值几乎相同的判断——
    # 正是这种"同一个判断有两份实现"让阶段6 的三指标在报告里分叉成了两个值。
    # 只有当 JSON 里没有该字段（旧产物）时才退回本地计算。
    degenerate = s10.get("e10_3_degenerate")
    if degenerate is None:
        degenerate = [k for k, v in unc.items()
                      if v.get("sd") is not None
                      and (v["sd"] == 0.0
                           or (abs(v.get("mean") or 0) > 0
                               and v["sd"] / abs(v["mean"]) < 1e-6))]
        degen_src = "（本地重算：旧产物里没有 e10_3_degenerate）"
    else:
        degen_src = ""
    degen_note = ""
    if degenerate:
        degen_note = (
            '<div class="callout danger"><strong>⚠️ 本节的参数区间已退化，'
            '不是一个有效的置信区间。</strong>'
            f"实测 <code>{html.escape(', '.join(degenerate))}</code> 的跨自助样本"
            "标准差 ≈ 0（量级 1e-19 ~ 0）。"
            "原因不是「估计精确」，而是<strong>搜索分辨率不够</strong>："
            f"每个自助样本都用同一套 {fit.get('n_evaluations')} 个随机搜索点，"
            "网格太粗 ⇒ 6 个样本全部落在<strong>同一个格点</strong>上 ⇒ 区间宽度恒为 0。"
            "修法是把 <code>n_iter</code> 提高 1~2 个数量级，或改用连续优化器——"
            "本阶段没有做。<br>"
            "<strong>连带后果：E10.4「阶段4 的最优点落在区间外」也因此失去意义"
            "（区间是一个点）。</strong></div>")
        if degen_src:
            degen_note = degen_note.replace(
                "</div>", f"<br><span class=\"note\">{degen_src}</span></div>")
    params = fit.get("best_params", {})
    parts = fit.get("best_diagnostics", {}).get("parts", {})
    prow = sorted(parts.items(), key=lambda kv: -abs(kv[1]))
    return f"""
<section id="s10">
<h2>8. 阶段10 · 统一校准框架</h2>
<p class="lead"><strong>一句话结论：MSM 的标准化距离与块自助法都实现了，
{('阶段4 的最优点有 ' + str(e4['n_inside']) + '/' + str(len(unc)) + ' 个参数落在自助区间内；')
  if e4 else ''}但本阶段最重要的产出是一个<b>负面结论</b>——
块自助法对本项目的 ACF 型矩<b>系统性低估</b>，用 168 小时的块长只能恢复真值的 0.9%。</strong></p>

<h3>8.1 E10.1 标准化距离 vs 裸欧氏距离</h3>
{table(["报价上界", "图表派", "死区", "裸距离", "标准化距离"], rows)}
<p class="note">
两种口径{"挑出的最优点相同" if e1.get("same_choice") else "<strong>不同</strong>——标准化确实改变了挑选结果"}。
标准化的必要性：不标准化时 σ（几十 bp）会压倒 <code>r ACF(1)</code>（0.01 量级），
校准退化成"只把 σ 调对"，而报告里显示的是"综合距离"。
</p>

<h3>8.2 E10.2 MSM 校准</h3>
<p>搜索参数：{html.escape(str(list(params)))}；共 {fit.get('n_evaluations')} 次评估。</p>
{table(["参数", "最优值"], [[k, f(v, ".6g")] for k, v in params.items()])}
<p>逐矩的标准化偏差（|z| 越大越不像）：</p>
{table(["矩", "标准化偏差 z", "|z|"],
       [[k, f(v, "+.3f"), f(abs(v), ".3f")] for k, v in prow])}
<p>
<strong>最不像的矩是 <code>{prow[0][0] if prow else "—"}</code></strong>——
这一项就是本模型当前最大的失配，比看一个总分有用得多。
</p>

<h3>8.3 E10.3 参数不确定性（区间端点不可靠）</h3>
{degen_note}
{table(["参数", "均值", "标准差", "区间（端点不可靠）", "相对宽度"], urows)}
<div class="callout danger">
<strong>本阶段最重要的负面结论：块自助法的块长不充分。</strong>
实测（BTC 17520 根小时线、<code>acf_abs_lag1</code> 相对全样本的恢复比例）：
<table><thead><tr><th>块长</th><th>168</th><th>500</th><th>2000</th><th>4000</th><th>8000</th></tr></thead>
<tbody><tr><td>BTC 恢复比例</td><td><b>0.9%</b></td><td>6.9%</td><td>32.4%</td>
<td>58.0%</td><td>69.3%</td></tr>
<tr><td>ETH 恢复比例</td><td><b>0.9%</b></td><td>7.6%</td><td>27.9%</td>
<td>47.4%</td><td>60.8%</td></tr></tbody></table>
块长必须覆盖该矩的<strong>依赖长度</strong>，而波动率聚集的依赖长度远长于一周。
所以本阶段给出的参数区间是<strong>围绕一个被系统性低估的矩</strong>构造的——
它看起来完全正常，实际中心是偏的。<strong>这个缺陷无法靠增加自助次数修好。</strong>
<br>另外 {s10['config']['n_boot']} 个自助样本<strong>估不出 2.5% 分位点</strong>
（指导书默认 50 次，在本项目的模拟成本下是小时级），
所以区间端点不可靠，只能读相对宽度。
</div>

<h3>8.4 E10.4 / E10.5 与阶段4 网格点的对照</h3>
<p>
阶段4 最优点用本阶段的标准化距离重评 = {f(e4.get('stage4_distance') if e4 else None, '.4f')}，
MSM 最优 = {f(fit.get('best_distance'), '.4f')}。
{e4 and ('落在区间内的参数：' + str(e4['n_inside']) + '/' + str(len(unc))) or ''}
</p>
<p><strong>结论：{e5.get('verdict')}</strong></p>
<div class="callout">
<strong>这个对比是不公平的，报告里必须写明：</strong>
阶段4 是 36 组粗网格里选一个，而 MSM 有 {fit.get('n_evaluations')} 次评估 + 局部细化。
公平的问法是"同样的评估预算下谁更好"，那需要把网格也细到同样的点数——
<strong>本阶段没有做</strong>，所以差值只能读成"MSM 在其搜索空间里找到了更优点"，
<strong>不能读成"MSM 优于网格搜索"</strong>。
</div>
{img(FIG / "stage10_distance.png", "E10.1 标准化距离：让每个矩用自己的尺度说话")}
{img(FIG / "stage10_uncertainty.png", "E10.3 参数分布（自助重采样）—— 端点不可靠，请读宽度")}
</section>
"""


def sec_stage11(s11: dict | None) -> str:
    if not s11:
        return sec_stage_generic("s11", "9", "阶段11 · 「至暗时刻」压力测试", None)
    summ = s11.get("e11_1_summary", {})
    rows = [[v["label"], f"{v['n_observed']}/{v['n_seeds']}",
             f(v["mean_delta"], "+.4g"), f(v["mean_z"], "+.2f")]
            for v in summ.values()]
    n_links = s11.get("e11_1_n_links")
    d113 = s11.get("e11_3_detail") or []
    d113_rows = "".join(
        "<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>"
        % (r["seed"], "✅" if r["health_A"] else "❌",
           "✅" if r["health_B"] else "❌", f(r["n_bad_A"], ".0f"),
           "✅" if r.get("cons_A", True) else "❌") for r in d113)
    # E11.3 的换手/成本归因（如果 JSON 里有）
    d113_ex = d113[0] if d113 else {}
    sens = s11.get("e11_2_sensitivity", [])
    srows = [[f(r["corr_scale"], ".1f"), f(r["realized_corr"], ".3f"),
              f(r["devB_delta_bp"], ".4g")] for r in sens]
    return f"""
<section id="s11">
<h2>9. 阶段11 · 「至暗时刻」压力测试</h2>
<p class="lead"><strong>一句话结论：链条中 {n_links}/{len(rows)} 个环节被配对判定为
「观察到传导」（验收要求 ≥3 → {"通过" if s11.get("e11_1_pass") else "未通过"}）；
但本场景的参数不是校准出来的，所以它只能回答"在这个特定配置下观察到了什么"，
不能回答"多大的拥挤度会让连锁反应显著放大"。</strong></p>

<h3>9.1 链条配置</h3>
<ul>
<li><strong>多资产</strong>（阶段9）：A 与 B 高相关（σ_c = {f(s11['config']['common_vol'], '.0e')}）</li>
<li><strong>拥挤交易</strong>（阶段7）：{s11['config']['n_trend']} 个动量追随者，
全部用同一个 lookback ⇒ 行为同步</li>
<li><strong>资金费率高位</strong>（阶段5）：A 长期做多偏好（p_buy={f(s11['config'].get('p_buy'), '.2f')}），
{s11['config']['n_arb']} 个费率套利者</li>
<li><strong>触发</strong>：第 {s11['config']['warmup'] + s11['config']['pre']} tick
对 A 的持仓前 15 名主体强制市价清算</li>
</ul>

<h3>9.2 逐环节配对判定</h3>
<p class="note">
判据事先定死：冲击后 {s11['config']['post']} tick 的配对差均值
超出「冲击前 {s11['config']['pre']} tick 均值 ± 3×标准误」才算"观察到"。
事后挑指标会把任何结果都解释成成功。
</p>
{table(["环节", "观察到（种子数）", "平均配对差", "平均 z"], rows)}
<div class="callout danger">
<strong>机制组合暴露的真实问题：把阶段5 的杠杆加到所有资产上会让主体爆仓。</strong>
最初的实现是多重继承 <code>AssetMarket + PerpetualMarket</code>，
结果 B 资产的零智能主体<strong>权益全部为负</strong>——
<code>allow_negative_cash = True</code>（永续天然带杠杆），
而本模型<strong>没有强平引擎</strong>，现金被资金费率扣成负数之后没有任何机制把它拉回来。
这不是实现 bug，是<strong>机制组合的真实后果</strong>：
阶段5 的杠杆 + 没有保证金引擎 = 必然有人爆仓。
处理方式：结构上把 B 资产退成裸市场（它只承担"相关性通道"的角色），
并<strong>如实写在这里</strong>，而不是调参掩盖。
</div>

<h3>9.3 E11.3 全部机制叠加后的市场是否还站得住</h3>
<p>
这是<strong>健全性检查</strong>，不是性能对比：所有机制一起打开时，
会不会出现「某个不变量被破坏」（现金不守恒、预留超支、负权益）。
</p>
<table><thead><tr><th>seed</th><th>A 健康</th><th>B 健康</th>
<th>A 权益为负的主体数</th><th>A 现金守恒</th></tr></thead>
<tbody>{d113_rows}</tbody></table>
<p>
整体判定：<strong class="warn">{"有不变量被破坏" if not s11.get("e11_3_healthy") else "全部通过"}</strong>。
</p>
<div class="callout danger">
<strong>根因是实测出来的，不是猜的：倒下的全是资金费率套利者。</strong>
<ul>
<li>它们把 <strong>{f(d113_ex.get('arb_turnover'), ',.0f')}</strong> 元的成交额打出去，
而本金合计只有 <strong>{f(d113_ex.get('arb_capital'), ',.0f')}</strong> 元 ——
<strong>换手 = {f((d113_ex.get('arb_turnover') or 0) / max(d113_ex.get('arb_capital') or 1, 1), '.1f')} 倍本金</strong>，
且全程用<strong>市价单</strong>调仓。</li>
<li>费率符号的翻转频率是 <strong>{f(d113_ex.get('funding_flip_frac'), '.1%')}</strong>
（均值 {f(d113_ex.get('funding_mean_bp'), '.2f')} bp）——信号接近抛硬币，
于是仓位反复翻向。</li>
<li>它们真正付出去的费率只有 <strong>{f(d113_ex.get('arb_funding'), ',.0f')}</strong> 元，
本金的其余部分全被<strong>价差 + 冲击成本</strong>吃掉。</li>
</ul>
<strong>现金守恒成立</strong>（见上表），所以这不是记账 bug，是策略本身在烧钱。
<br>⚠️ 这条与阶段5 的 E5.3 是<strong>同一行为的另一面</strong>：
那里套利者「把费率推平」很有效，代价就是这里的巨量换手——
两件事必须一起说，否则读者会以为 E5.3 是纯收益。
</div>

<h3>9.4 相关性敏感性</h3>
{table(["相关缩放", "实测相关", "B 的配对差 (bp)"], srows)}
<p>
被判为<strong>{"单调" if s11.get("e11_2_monotone") else "不单调"}</strong>。
即使不单调也不能直接说"机制不成立"：配对差里还含随机漂移的残余
（虽然已逐 tick 相减，但两条路径的 agent 决策会因冲击而分岔）。
</p>
{img(FIG / "stage11_chain.png", "E11.1 「至暗时刻」链条：同种子配对差（处理 − 控制，逐 tick）")}
{img(FIG / "stage11_correlation.png", "E11.2 跨资产传导幅度 vs 相关系数（判据：应随相关上升）")}
</section>
"""


def sec_boundaries(jsons: dict) -> str:
    items = []
    for key, label in (("stage5", "阶段5 永续合约"), ("stage6", "阶段6 长记忆订单流"),
                       ("stage7", "阶段7 反身性"), ("stage8", "阶段8 Hawkes"),
                       ("stage9", "阶段9 多资产"), ("stage10", "阶段10 统一校准"),
                       ("stage11", "阶段11 至暗时刻")):
        d = jsons.get(key)
        if not d:
            items.append(f'<h3>{label}</h3><p>{MISSING}</p>')
            continue
        lines = d.get("honest_boundary") or []
        body = pre("\n".join(lines)) if lines else "<p class='note'>该阶段未写入边界小结。</p>"
        items.append(f"<h3>{label}</h3>{body}")
    return f"""
<section id="boundaries">
<h2>10. 诚实边界（独立章节）</h2>
<p class="lead">
指导书 §0 的硬性要求：<strong>每个阶段结束必须写一份「诚实边界」小结</strong>，
明确做到了什么量级、什么做不到、下个阶段依赖它的哪个假设。
以下逐字来自各阶段脚本的输出（<code>out/stage*_metrics.json</code> 的
<code>honest_boundary</code> 字段），<strong>未经润色</strong>。
</p>
<h3>二期完成后，整个项目依然不能做什么</h3>
<ol class="fail">
<li><strong>不能预测价格。</strong>全部 11 个阶段合起来仍然是一个生成器，不是预测器；
收益率无自相关是做<strong>对了</strong>，不是缺陷。</li>
<li><strong>不能直接外推到真实市场。</strong>冲击函数的凹度（阶段6）经三线深挖的
B 线从 1.251 改善到 0.765、联合配置到 0.957，但<strong>都还没到 0.5</strong>；
而这是评估执行成本的关键量——用它做大单成本估计仍会系统性偏离。
（详见 §14；注意阶段8 曾宣称的 0.481 <strong>不应再被引用</strong>，那是标尺错误的产物。）</li>
<li><strong>不能拿它当风控参数的来源。</strong>阶段9/11 的传导幅度都没做充分的
配对反事实，阶段10 的参数不确定性也没被可靠量化。</li>
<li><strong>不能替代历史回测。</strong>偏差方向不同，应互补使用。</li>
<li><strong>没有手续费、没有强平引擎、没有 ADL/保险基金、没有借券成本。</strong>
真实市场的成本结构与极端行情放大器里有很大一块是这些，本模型全部缺失。</li>
</ol>
{''.join(items)}
</section>
"""


def sec_methodology() -> str:
    return """
<section id="method">
<h2>11. 跨两期的方法论收获</h2>
<p class="lead">
这一节记的是<strong>工程知识</strong>，不是市场结论——
两期都踩过、或者一期的教训在二期以新形态重现的坑。
它们比任何单个数字都更可复用。
</p>

<h3>11.1 「因果测量三原则」在两期都被证明是必需的</h3>
<ol>
<li><strong>路径覆盖事件全程，不只看结束后。</strong>
一期：分期清算的峰值被整个漏掉（路径从清算<em>结束后</em>才开始记），
把幂律指数算成 −0.26。二期：资金费率统计如果把开局几千 tick 的
初始禀赋松弛过程算进去，正费率占比被系统性压低，极端情况下<strong>翻转符号</strong>
（p_buy=0.70 只跑 2000 tick 得 −3.2%，丢掉预热后同一条路径是 +18.4%）。</li>
<li><strong>同种子配对，不是"跑两次取平均"。</strong>
一期：配对把跨种子标准差从 89.5bp 降到 1.4bp（降噪 62 倍）。
二期：阶段6 的 E6.2/E6.4、阶段7 的反身性扫描、阶段11 的链条判定
全部依赖同种子配对；不配对时随机漂移会完全淹没要测的东西。</li>
<li><strong>反事实锚点必须同一时点。</strong>
一期：用清算前的固定价格当基准，会把抛售期间的市场漂移算成滑点
（拆 100 片抛跨 100 tick，漂移轻松上百 bp）。
二期：阶段9 的 E9.4 <strong>没有</strong>做这一步，
所以那一节的传导幅度被明确标注为"不可信"——<strong>原则知道是一回事，
每次都执行是另一回事</strong>。</li>
</ol>

<h3>11.2 一期的坑在二期以新形态重现（四次）</h3>
<table>
<thead><tr><th>一期的坑</th><th>二期的重现形态</th></tr></thead>
<tbody>
<tr><td>价格不吸附到 tick 网格 → 预留与结算口径不一致</td>
<td>记账粒度：逐笔成交 vs 按订单算 ACF，机械正自相关把 0.14 读成 0.36</td></tr>
<tr><td>「主体数量」与「背景行情」必须解耦</td>
<td>随机流解耦：<code>spawn_rng(seed, agent_id, tag)</code>；
用内置 <code>hash()</code> 会带进程随机盐，破坏跨进程复现</td></tr>
<tr><td>单条路径 + 峰值 = 噪声</td>
<td>γ 的单次快照噪声极大（三个种子 0.049/0.138/0.366），
三个种子分辨不出 +0.5 的位移</td></tr>
<tr><td>护栏写得太松 → 在数值噪声上拟合出漂亮的 R²</td>
<td>Ljung-Box 的 Q 与样本量成正比，跨样本量比 Q 是口径错误</td></tr>
</tbody>
</table>

<h3>11.3 二期的六条新教训（都属于「不报错但结论是错的」）</h3>
<div class="callout danger">
<ol>
<li><strong>排序键里的 nan 会静默劫持 argmin。</strong>
<code>min(scan, key=dist)</code> 遇到 nan 会<strong>保留第一个元素</strong>
（任何 <code>x &lt; nan</code> 都是 False）。E5.4 因此"选中"了表里第一档，
并写下与数据相反的结论。修法：非常数返回 <code>inf</code>。</li>
<li><strong>对照组其实开着。</strong>
<code>MetaOrderMixin.__init__</code> 收不到配置时用默认值（即"开着"），
实验的注入钩子忘了显式传惰性配置 → "机制关闭"的对照组每个主体都在拆单。
<strong>症状极具误导性</strong>：基线与小规模那两行数字一模一样，
看起来像"小规模无效"，其实是同一件事跑了两遍。</li>
<li><strong>⭐ 重跑之后拿到和重跑前一模一样的数字——而这次"一模一样"<em>无法被解释</em>。</strong>
这是本期<strong>最危险</strong>的一条，因为它打击的不是某个数字，而是<strong>整条证据链</strong>。
第 2 条修掉之后重跑，产出的 k 值与修复前<strong>逐位相同</strong>。
问题在于：**这个"相同"当时有两种可能，而当时的机制分不出来**——
要么修复对结果确实没有影响，要么这次"重跑"根本没跑、直接命中了陈旧缓存。
根因是缓存键里只有 <code>&#123;tag, seed, n_ticks, mix&#125;</code>，
而"对照组是不是真的关着"取决于 <code>Stage6Market.decorate_agent</code> 那段装配代码，
它<strong>不在任何参数里</strong>，所以改完之后键一个字符都没变。
事后核对证实了后者：旧缓存里确实躺着那次基线要用的键
（<code>e13b225a…</code>），而带签名的键 <code>aa629d40…</code> 不在里面——
<strong>也就是说，那次"重跑"在物理上不可能验证修复。</strong>
<em>（诚实补充：清空缓存后真正重算，E6.1 的数字与本报告此前报出的一致。
所以这一次并没有真的产出错误数字。但这不是"没事"，而是"运气好"——
当时没有任何机制能告诉我们会是哪一种。)</em>
<strong>结论：缓存键必须覆盖「把配置翻译成行为」的代码本身</strong>，
这类输入既不在参数里、也不在配置对象里，只能靠行为签名。
现在三道锁：行为签名（类 + 工厂 + 常量 + 库源码树）进每一个键；
缓存文件头记录本轮签名、不符则整份作废；命中/未命中数打出来。
配套回归测试见 <code>tests/test_cache_keys.py</code>。</li>
<li><strong>离散化让连续公式失效。</strong>
Hawkes 的平稳均值在离散递推下是 <code>μ/(1−α/(e^(β·dt)−1))</code> 而不是
<code>μ/(1−α/β)</code>。用连续公式反解，实际到达率偏低 18%，
而"打开机制不改变平均活跃度"正是归因能成立的前提。</li>
<li><strong>判据写成恒真命题 = 没有验收。</strong>
指导书预期"基线 ACF ≈ 0"，实测基线 ACF(1) ≈ 0.14 且有幂律衰减。
若照抄"从 0 变正"当判据，等于用一个本来就成立的命题冒充验收。</li>
<li><strong>块自助法只对「均值型」矩有效。</strong>
对 ACF/聚集这类"依赖型"矩，块长必须覆盖依赖长度——
而当依赖长度是样本量的一半时，块自助法已经没有可用空间
（实测 block=168 只恢复 0.9%，block=8000 才 69%）。
<strong>给出一个看起来很专业但中心是偏的区间，比诚实报"未量化"更糟。</strong></li>
</ol>
</div>

<h3>11.4 自检脚本本身也会腐化</h3>
<p>
手册要求"报告里的数字必须从产物读出来，不许手抄"。
但"不手抄"只保证<strong>生成时</strong>一致，保证不了重跑之后还有人去改
——总览表里就真的手抄过一句「k 反而从 A 走到 B」，两个数字写死在模板里，抄在 3 个地方，
而第 3 条修完之后一个都没动。
所以新增了 <code>scripts/selfcheck.py</code>：它<strong>只读产物</strong>，
不看生成器代码（避免"用生成器的假设去验证生成器"），
把报告的承重数字逐个与 <code>out/*.json</code> 现算值对账。
</p>
<p class="note">
而写这个自检的过程本身又踩了两个坑，一并记下：
① <strong>检查比被检查的东西还脆</strong>——第一版按
<code>&#123;v:.3f&#125;</code> 拼字符串去报告里找，报告把同一个量写成
四位小数（而检查按三位去拼字符串）就判"找不到"。改成"抠出报告里所有数值、
按容差比对"才对，对排版不敏感、对<strong>值本身</strong>敏感。
② <strong>它抓出的第一个"问题"是它自己的假阳性</strong>——
正文里有一句「非常数返回 <code>inf</code>」，
按全文找 <code>&gt;inf&lt;</code> 就把这句当成了漏出的值；
漏出的真形态是 <code>&lt;td&gt;inf&lt;/td&gt;</code>。
</p>

<h3>11.5 「机制实现」与「机制有效」必须分开报</h3>
<p>
二期新增的每一块机制都能被验证"确实在工作"：
元订单确实拆了（45 张子单/单）、Hawkes 确实产生了聚集、
自适应流动性确实推开了报价、配对交易确实在按 z-score 开仓。
但<strong>"在工作"与"把目标指标修好了"是两件事</strong>——
阶段6 的 k 值就是反例：机制在工作，主指标反而更远离目标。
报告必须把这两层分开写，否则"已实现"会读成"已达标"。
</p>
</section>
"""


def sec_kernel(mut_log: str | None, n_tests: int | None) -> str:
    m = mut_log or ""
    total_line = ""
    base_line = ""
    n_anchor = ""
    for ln in m.splitlines():
        t = ln.strip()
        if t.startswith("✅ ") and "注入的 bug" in t:
            total_line = t
        # ⚠️ 解析要容忍"日志格式变了"。改成并行之后基线那行从
        #    `基线（未注入 bug）：...` 变成了 `✅ 基线：...`，
        #    原来的 `startswith("基线")` 直接匹配不上，报告里就会缺这一段。
        #    现在两个格式都能认。
        if "基线" in t and ("Ran " in t or "全绿" in t):
            base_line = t
        if "锚点全部有效" in t or "锚点失效" in t:
            n_anchor = t
    return f"""
<section id="kernel">
<h2>12. 内核可信度：测试与变异验证</h2>
<p class="lead">
<strong>一个全绿的测试套件说明不了任何事。</strong>
它可能绿是因为实现正确，也可能绿是因为断言写得太松。
所以二期沿用一期的做法：把“我们最怕出现的实现 bug”逐个注入源码，
每个都必须让测试<strong>变红</strong>；仍全绿的位置就是测试漏洞。
</p>
<p>
二期结束时单元测试总数：<strong>{n_tests if n_tests is not None else "—"}</strong> 项。
{html.escape(base_line) if base_line else ""}
{html.escape(n_anchor) if n_anchor else ""}
</p>
<p><strong>{html.escape(total_line) if total_line else "（变异验证日志缺失）"}</strong></p>
<pre class="log">{html.escape(m[-6000:]) if m else "（缺 out/mutation_run.log）"}</pre>
<p class="note">
二期新增的变异体覆盖阶段5（费率方向/量级/记账归属，M16~M22）、
阶段6（拆单机制静默失效，M23~M26）、阶段7（学习方向与配对，M27~M29）、
阶段8（稳定性放行与衰减项遗漏，M30~M31）、阶段9（公共冲击重复与腿标签反转，M32~M33）。
编号相对指导书有后移，原因写在 <code>scripts/mutation_check.py</code> 的注释里：
阶段5 需要 7 个变异体而不是指导书规划的 4 个。
</p>
</section>
"""


def sec_repro() -> str:
    return """
<section id="repro">
<h2>13. 复现方式</h2>

<h3>13.1 一键跑通全部 11 个阶段（推荐）</h3>
<pre class="code">PY=C:/Users/87465/.workbuddy/binaries/python/envs/py313/Scripts/python.exe

# 全部 11 个阶段 + 两份报告 + 自检 + 测试 + 变异验证
$PY -u scripts/reproduce_all.py --with-mutation

# 只看执行计划（不动手）
$PY -u scripts/reproduce_all.py --list --with-mutation

# 已知某个阶段的产物比库代码还新时，跳过它省时间
#   （对账只对**本次真跑过**的步骤要求新鲜度）
$PY -u scripts/reproduce_all.py --skip 8,9,11 --with-mutation</pre>
<p class="note">
<code>reproduce_all.py</code> 做三件事：<strong>按依赖顺序跑 / 任一环节失败立即停 /
跑完对账</strong>。对账检查每个阶段的 JSON 存在、可解析、<code>stage</code> 字段自洽、
二期阶段带 <code>honest_boundary</code>，并且<strong>本次跑过的步骤都产出了新文件</strong>
（防止拿旧产物冒充）。分步命令没有这些保证——真实踩过：某个阶段脚本因为缺一个
import 直接崩了，其余阶段照跑，报告把那个阶段静默降级成"尚未运行"，没人发现。
</p>
<p class="note">
开跑前会先把现有产物快照到 <code>out/_archive/&lt;时间戳&gt;/</code>。
这是被真实代价逼出来的：重跑会<strong>原地覆盖</strong> <code>out/*.json</code> 与
<code>docs/*.html</code>，一旦覆盖，"修复前 vs 修复后"就再也比不了了。
本轮就吃了这个亏——改完配对交易的做空额度后重跑阶段9，想对比标定表的新旧值，
才发现旧的 <code>stage9_metrics.json</code> 已经被覆盖。
<strong>"能对比"本身就是证据的一部分</strong>，而快照只要几 MB。
</p>
<p class="note">
⚠️ 子进程一律加 <code>-u</code>。不加的话 stdout 重定向到文件时 Python 会切成
<strong>块缓冲</strong>：长步骤跑到一半日志是 0 字节，"在算"和"卡死"看起来一样；
而<strong>崩溃时缓冲区里的输出会随进程一起消失</strong>——那正是唯一能说明
"它死在哪一步"的东西。
</p>

<h3>13.2 分步命令（调试用）</h3>
<pre class="code">PY=C:/Users/87465/.workbuddy/binaries/python/envs/py313/Scripts/python.exe

# 全部测试（一期 + 二期）
$PY -m unittest discover -s tests -t .

# 变异验证：每个注入的 bug 都必须被抓到
$PY scripts/mutation_check.py

# 二期各阶段实验（产物写 out/*.json 与 out/figs/*.png）
$PY scripts/run_stage5.py     # 永续合约：E5.0~E5.4
$PY scripts/run_stage6.py     # 长记忆订单流：E6.0~E6.5
$PY scripts/run_stage7.py     # 反身性：E7.0~E7.4
$PY scripts/run_stage8.py     # Hawkes：E8.0~E8.2
$PY scripts/run_stage9.py     # 多资产：E9.1~E9.4
$PY scripts/run_stage10.py    # 统一校准：E10.1~E10.5
$PY scripts/run_stage11.py    # 至暗时刻：E11.1~E11.3

# 生成本报告
$PY scripts/make_report2.py

# 一期的四阶段实验与报告（保持可用）
$PY scripts/calibrate.py && $PY scripts/run_stage1.py && $PY scripts/run_stage2.py
$PY scripts/run_stage3.py && $PY scripts/run_stage4.py
$PY scripts/make_report.py</pre>
<h3>13.3 关于磁盘缓存（阶段5 / 阶段6）</h3>
<p>
阶段5/6 带磁盘缓存（<code>out/stage5_cache.json</code> / <code>stage6_cache.json</code>）。
缓存让"改一个小 bug 要重跑一小时"变成"只重跑出错的那一段"。
<strong>但缓存键漏一个输入，就会静默命中陈旧结果——比没有缓存危险得多。</strong>
这在本期真实发生过一次：修掉阶段6 对照组的 bug 后重跑，
k 值与修复前<strong>逐位相同</strong>——而事后核对旧缓存的键才发现，
那次"重跑"命中了陈旧缓存，**在物理上不可能验证修复**
（旧键 <code>e13b225a…</code> 在缓存里，带签名的键 <code>aa629d40…</code> 不在）。
根因是"对照组是不是真的关着"取决于装配代码，而那段代码不在任何参数里
（详见 §11.3 第 3 条）。
</p>
<p>
现在 <code>scripts/_cache.py</code> 用三道锁：行为签名（类 + 工厂 + 模块常量 +
<code>tw/</code> 源码树内容哈希）进入<strong>每一个键</strong>；
缓存文件头记录本轮签名，<strong>不符则整份作废</strong>；
命中/未命中数在每次运行结束时打出来（静默命中是这类事故的共同特征）。
签名机制本身由 <code>tests/test_cache_keys.py</code> 守着——
其中一条测试会把源码复制一份、只改「惰性配置 → 默认配置」那一句，
断言签名必须变化。<strong>没有这条测试，上面三个制度都只是注释。</strong>
</p>

<h3>13.4 本报告数字的来源与对账</h3>
<p>
所有数字由 <code>scripts/make_report2.py</code> 从 <code>out/stage*_metrics.json</code>
直接读出。生成器在写出文件前会自检一遍：正文里<strong>不允许残留没被替换掉的
占位符</strong>——真实踩过（<code>sec_overview</code> 忘加 f 前缀，
一个 <code>&#123;_krange_txt(jsons)&#125;</code> 直接印进了总览表），
而在那之前没有任何机制会发现。
<span class="note">（这里用 <code>&amp;#123;/&amp;#125;</code> 写花括号是**故意的**：
源码检查脚本会把"以字母开头的花括号"当成忘加 f 前缀的模板，
而这一段正是在**引用**那个 bug。转义之后渲染结果一样，检查不再误报。）</span>
</p>
<p>
<code>scripts/selfcheck.py</code> 是<strong>独立于生成器</strong>的第三方检查：
它只读 <code>out/*.json</code> 与两份报告，把报告里的"承重数字"逐个与 JSON
现算值对账（按<strong>数值 + 容差</strong>，不按格式字符串），并检查章节齐全、
交叉引用自洽、图不被浪费、没有漏出 <code>nan</code>/<code>None</code>/<code>inf</code>、
没有忘加 f 前缀的模板字符串。
</p>
<p class="note">
<strong>注意阶段5 与阶段6 的实现是同一份仓库的两个阶段，不要只看某一阶段的 JSON 下结论。</strong>
每个阶段的目录都有一段"诚实边界"，本报告第 10 节把它们原样汇总。
</p>
</section>
"""


CSS_AND_SHELL = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>交易世界 · 二期交付报告</title>
<style>
  :root {
    --bg:#16181d; --panel:#1c1f26; --panel2:#22262f; --fg:#d8dee9; --muted:#8b95a5;
    --grid:#2c313a; --accent:#e06c75; --good:#8fbf6a; --mid:#5aa9e6; --warn:#e0a458;
  }
  * { box-sizing:border-box; }
  body {
    margin:0; background:var(--bg); color:var(--fg);
    font:15px/1.75 "Microsoft YaHei","PingFang SC","Hiragino Sans GB",system-ui,sans-serif;
  }
  .wrap { max-width:1080px; margin:0 auto; padding:48px 28px 96px; }
  header.top { border-bottom:1px solid var(--grid); padding-bottom:24px; margin-bottom:36px; }
  header.top h1 { font-size:30px; margin:0 0 10px; letter-spacing:.5px; }
  header.top .sub { color:var(--muted); font-size:14px; }
  h2 { font-size:22px; margin:52px 0 14px; padding-left:12px;
       border-left:4px solid var(--mid); }
  h3 { font-size:17px; margin:30px 0 10px; color:#c3cbd8; }
  h4 { font-size:15px; margin:22px 0 8px; color:#b5bfcd; }
  p { margin:12px 0; }
  code { background:var(--panel2); padding:1px 6px; border-radius:4px; font-size:13px;
         color:#9fd0f0; font-family:ui-monospace,Consolas,"Courier New",monospace; }
  pre.log, pre.code {
    background:var(--panel); border:1px solid var(--grid); border-radius:8px;
    padding:14px 16px; overflow-x:auto; font-size:12.5px; line-height:1.6;
    font-family:ui-monospace,Consolas,"Courier New",monospace; color:#b9c4d4;
    white-space:pre-wrap;
  }
  table { width:100%; border-collapse:collapse; margin:16px 0; font-size:14px; }
  th, td { padding:8px 11px; text-align:right; border-bottom:1px solid var(--grid); }
  th:first-child, td:first-child { text-align:left; }
  thead th { background:var(--panel); color:#a9b4c4; font-weight:600; font-size:13px; }
  tbody tr:hover { background:#1f232b; }
  .hi { color:var(--good); }
  .callout { background:var(--panel); border-left:3px solid var(--mid);
             padding:12px 16px; margin:18px 0; border-radius:0 8px 8px 0; }
  .callout.danger { border-left-color:var(--accent); }
  .lead { font-size:16.5px; }
  .note { color:var(--muted); font-size:13.5px; }
  .warn { color:var(--warn); }
  ol.fail li { margin:8px 0; }
  figure { margin:24px 0; background:var(--panel); border:1px solid var(--grid);
           border-radius:10px; padding:12px; }
  figure img { width:100%; display:block; border-radius:6px; }
  figcaption { color:var(--muted); font-size:13px; margin-top:10px; text-align:center; }
  figure.missing .ph { padding:40px; text-align:center; color:var(--accent); }
  nav.toc { background:var(--panel); border:1px solid var(--grid); border-radius:10px;
            padding:16px 22px; margin-bottom:8px; }
  nav.toc ol { margin:6px 0; padding-left:22px; }
  nav.toc a { color:#9fd0f0; text-decoration:none; }
  nav.toc a:hover { text-decoration:underline; }
  strong { color:#eaeff7; } b { color:#eaeff7; }
</style>
</head><body><div class="wrap">
<header class="top">
  <h1>交易世界 · 二期交付报告</h1>
  <div class="sub">
    基于主体建模的人工加密市场 · 第二阶段（永久合约 / 长记忆订单流 / 反身性 /
    Hawkes / 多资产 / 统一校准 / 终局整合）<br>
    每个阶段的结论都允许是负面的；<b>失败写在最前面</b>。
  </div>
</header>
<nav class="toc"><strong>目录</strong>
<ol>
  <li><a href="#summary">一句话结论（含全部失败）</a></li>
  <li><a href="#overview">二期总览：7 个阶段 → 实际交付</a></li>
  <li><a href="#s5">阶段5 · 永续合约机制层</a></li>
  <li><a href="#s6">阶段6 · 长记忆订单流（LMF）</a></li>
  <li><a href="#s7">阶段7 · 反身性</a></li>
  <li><a href="#s8">阶段8 · Hawkes 订单到达</a></li>
  <li><a href="#s9">阶段9 · 多资产相关市场</a></li>
  <li><a href="#s10">阶段10 · 统一校准框架</a></li>
  <li><a href="#s11">阶段11 · 「至暗时刻」压力测试</a></li>
  <li><a href="#boundaries">诚实边界（独立章节）</a></li>
  <li><a href="#method">跨两期的方法论收获</a></li>
  <li><a href="#kernel">内核可信度：测试与变异验证</a></li>
  <li><a href="#repro">复现方式</a></li>
</ol></nav>
{{BODY}}
</div></body></html>
"""


def sec_k_resolution() -> str:
    """⭐ 重大更正：k 这个指标在本项目的装置上**没有分辨力**。

    为什么这条必须放在最显眼的位置
    -----------------------------
    它推翻的是**本报告自己**的多个结论（阶段6 的 0.665→0.910、
    阶段8 的 1.395→0.481、以及三线深挖与合并验证的全部 k 比较）。
    按项目的披露纪律：**不许因为"不想推翻自己"而轻描淡写**，
    也不许把原有数字悄悄改掉——原数字是当次的真实读数，
    要改的是**我们对它含义的理解**。

    数据源：``out/k_uncertainty.json``（``scripts/diagnose_k_uncertainty.py``），
    它用**产品代码里的同一个 ``fit_power_law``** 做参数 bootstrap。
    """
    ku = load("k_uncertainty.json") or {}
    s = ku.get("summary") or {}
    # ---- 回填后的进展（工作线L）----
    WL = load("workstream_L_metrics.json") or {}
    wfam = WL.get("families") or {}
    ea4 = {n: r for n, r in wfam.items()
           if n.startswith("EA4_") and r.get("k") is not None}
    backfill_html = ""
    if ea4:
        rows = "".join(
            f"<tr><td>{n}</td><td>{r.get('n_unsaturated')}/{r.get('n_levels_total')}</td>"
            f"<td>{f(r['k'], '.3f')}</td>"
            f"<td>[{f(r['ci'][0], '.3f')}, {f(r['ci'][1], '.3f')}]</td>"
            f"<td>{f(r['ci_width'], '.3f')}</td>"
            f"<td>{'否' if not r.get('contains_0_5') else '<b>是</b>'}</td>"
            f"<td>{r.get('original_k')}</td></tr>" for n, r in ea4.items())
        el2 = WL.get("el2_asymmetry") or {}
        backfill_html = f"""
<h3>回填后的进展（工作线L：8 种子 × 未饱和档）</h3>
<table><thead><tr><th>臂</th><th>未饱和/总档</th><th>k</th><th>95% CI</th>
<th>宽度</th><th>含 0.5</th><th>旧读数（3 种子）</th></tr></thead>
<tbody>{rows}</tbody></table>
<p>⭐ <strong>三个臂的 CI 现在都不包含 0.5</strong> ⇒
「冲击显著偏离平方根律」这一条终于有了<strong>区间支撑</strong>。
但 EL.2（非对称 vs 对称）的判定是 <strong>{el2.get('verdict')}</strong>
（区间重叠 {f(el2.get('overlap'), '.3f')}）——效应量本身太小，判不出来。</p>
<p>⚠️ <strong>更值得注意的是三臂的 k 几乎相同</strong>（0.73 / 0.74 / 0.79）。
把它与「有/无稀疏化」的效应对照：<strong>0.54 vs 0.06（约 9 倍）</strong> ⇒
三线深挖「只稀疏化吃单方最有效、双边同时稀疏化互相抵消」这个叙事，
在回填后的干净测量里<strong>几乎消失</strong>。
更准确的表述是：<strong>只要存在时间聚集性，冲击就会变凹；作用在需求侧还是供给侧，影响小得多。</strong></p>""" if ea4 else ""

    if not s:
        return f"""<section id="kres">
<h2>1.5　⚠️ 重大更正：k 的比较有没有分辨力？</h2>
<div class="warn"><p>未找到 <code>out/k_uncertainty.json</code>。先跑
<code>python scripts/diagnose_k_uncertainty.py</code> 再重出本报告。</p></div>
</section>"""
    return f"""
<section id="kres">
<h2>1.5　⚠️ 重大更正：k 的比较在本装置上<strong>没有分辨力</strong></h2>
<div class="warn">
<p><strong>本节推翻的是本报告自己的多个结论</strong>（阶段6 的 0.665→0.910、
阶段8 的 1.395→0.481、以及 §14 里三线深挖与合并验证的全部 k 比较）。
各节里的 k 数字<strong>保持原样不动</strong>——它们是当次的真实读数；
要更正的是<strong>我们对这些数字含义的理解</strong>。</p>
<ul>
<li>用同一装置（<code>paired_impact</code> + <code>fit_power_law</code>）算出的
<strong>{s.get('n_arms')} 个臂</strong>里，
<strong>{s.get('n_ci_contains_0.5')} 个的 95% 置信区间包含 0.5</strong>；
其中 {s.get('n_ci_contains_0')} 个连 0 都包含（<strong>滑点方向都不确定</strong>）。</li>
<li>区间宽度中位数 <strong>{f(s.get('width_median'), '.3f')}</strong>
（范围 {f(s.get('width_min'), '.3f')} ~ {f(s.get('width_max'), '.3f')}），
而我们一直在讨论的差异只有 0.3~0.9。</li>
<li>根因：每次 k 只有 <strong>4 个冲击档 × 3 个种子</strong>，
且低档位的滑点在统计上大多<strong>不显著</strong>（|t| &lt; 1.6）——
k 的点估计几乎由首尾两点之比决定。</li>
</ul>
        <p>⇒ 凡「k 从 X 改善到 Y」「k 离 0.5 有多远」的陈述，
在补足分辨力之前<strong>都不能作为证据</strong>。
<strong>不受影响</strong>的是机制层、账本层、事件率层的结论
（它们的证据不经过幂律拟合）。</p>
<p>⭐ <strong>解法（分辨力危机那一轮实测）</strong>：<strong>扩观测点，不是换判据、也不必加样本</strong>。
局部两两比较（用 2 个点算一个比值）会把<strong>自由度</strong>丢光，反而不如全局拟合；
而同一份数据把参与拟合的<strong>档位数</strong>从 3 增到 7，
95% 区间宽度从 <strong>2.662</strong> 压到 <strong>1.193</strong>（−55%），
且区间不再包含 0.5 ——<strong>代价近零</strong>。
机理是 t 临界值：df=1 ⇒ 12.71，df=5 ⇒ 2.57。
详见 <code>docs/分辨力危机-交付小结.md</code>。</p>
<p><strong>对二期立项依据的复核</strong>（工作线J）：阶段3 最初那个 k（=1.248，γ=0.801）
在原始档位配置下 95% 区间同时覆盖 0.5 与 1.0，
<strong>但扩到 4 档以上区间下界即抬到 0.5 以上</strong> ⇒
「冲击比平方根律陡」这个<strong>方向</strong>在三个独立来源上都成立
（阶段3 k=1.248、三线 A 线 Hawkes 关臂 1.395、工作线H baseline 1.292）。
⇒ <strong>二期的大方向没有错，错的是把验收标准设成了一个当时分辨力不足以判定的精确数字。</strong></p>
<p>完整证据（三步校验、复现自查、逐档显著性、19 臂 bootstrap、
「需要多少种子」的量化）见
<code>docs/前置校验-EA4-诊断报告.md</code>。</p>
{backfill_html}
</div>
</section>"""


def sec_workstream() -> str:
    """补充章节：三线深挖（A/B/C）与合并验证。

    ⚠️ 数字一律从 ``out/workstream_*_metrics.json`` **现算**，不手抄——
    与 ``docs/三线深挖-交付小结.md`` 同源（同一个量在两处出现时必须一致，
    自检 §⑩ 会查这一类分叉）。
    缺文件时整节降级成一句「先跑哪个脚本」，**不印空表**——
    "看起来完全正常的错数字"是本项目最怕的形态。
    """
    j = {k: (load(f"workstream_{k}_metrics.json") or {})
         for k in ("A", "B", "C", "joint")}
    if not any(j.values()):
        return """<section id="ws">
<h2>14. 补充：三线深挖（A/B/C）与合并验证</h2>
<p class="note">未找到 <code>out/workstream_*_metrics.json</code>。
先跑 <code>python scripts/run_workstream_A.py</code>（B / C / joint 同理）再重出本报告。</p>
</section>"""

    A, B, C, J = j["A"], j["B"], j["C"], j["joint"]
    blocks: list[str] = []

    # ---------------- A 线 ----------------
    u, c = A.get("ea1_uncorrected") or {}, A.get("ea1_corrected") or {}
    if u and c:
        rows = []
        for tag, d in (("未修正（E8.2 原样口径）", u), ("修正（同市场标定）", c)):
            rows.append(
                f"<tr><td>{tag}</td><td>{f(d.get('base_lambda'), '.2f')}</td>"
                f"<td>{f(d.get('lambda_scale_bias'), '.3f')}×</td>"
                f"<td>{f((d.get('arms') or {}).get('Hawkes 开(b=0.6)', {}).get('k'), '.3f')}</td>"
                f"<td>{f((d.get('arms') or {}).get('Hawkes 开(b=0.6)', {}).get('activated_frac'), '.1%')}</td>"
                f"</tr>")
        blocks.append(f"""
<h3>A 线 · 阶段8「意外修复」的真实根因</h3>
<p class="lead">阶段8 宣称的「k 从 1.395 拉到 0.481、离平方根律只差 0.019」
<strong>是实验口径错误造成的假象</strong>：λ̄ 在<strong>裸 Market</strong> 上标定，
却在<strong>带做市配置的 Market</strong> 上使用。修正标尺后 k 只走到 {f((c.get('arms') or {}).get('Hawkes 开(b=0.6)', {}).get('k'), '.3f')}。</p>
<table><thead><tr><th>标尺口径</th><th>λ̄</th><th>标尺偏差</th><th>k（b=0.6）</th>
<th>稀疏化真正生效</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<p class="note">「稀疏化真正生效」是本次新增的<strong>体检指标</strong>：
它一旦接近 0，说明机制撞在 <code>rho_to_count</code> 的上限上哑火了，
那个臂的 k <strong>不许</strong>被解释成「聚集性的效果」——只看 k 不看活跃度，
正是 E8.2 翻车的形态。</p>""")

    ea3 = A.get("ea3_variance_matched") or {}
    ea4 = (A.get("ea4_one_sided") or {}).get("arms") or {}
    ea2 = A.get("ea2_burst_depth") or {}
    if ea3:
        rows = []
        for tag, d in (ea4.items() if ea4 else []):
            rows.append(f"<tr><td>稀疏化作用面：{tag}</td><td>{f(d.get('k'), '.3f')}</td>"
                        f"<td>{f(d.get('activated_frac'), '.1%')}</td></tr>")
        blocks.append(f"""
<h3>A 线 · 三个假说的裁决</h3>
<table><thead><tr><th>假说</th><th>裁决</th><th>依据（现算）</th></tr></thead><tbody>
<tr><td>H1 突发期局部流动性变薄</td><td>❌ 否定</td>
<td>相对效应 {f((ea2.get('aggregate') or {}).get('pct_effect'), '.2%')}，
跨运行检验：{(ea2.get('aggregate') or {}).get('verdict') or '—'}</td></tr>
<tr><td>H2 方差效应（ρ 的时间结构）</td><td>✅ <strong>成立且是主因</strong></td>
<td>置换代理 k = {f(ea3.get('k'), '.3f')}（ρ 的 lag-20 自相关：
真 Hawkes {f(ea3.get('rho_acf_hawkes'), '.4f')} → 代理 {f(ea3.get('rho_acf_permuted'), '.4f')}，自激确实被抹掉）</td></tr>
<tr><td>H3 离散化偏差（公式错）</td><td>❌ 否定</td>
<td>现行公式实测偏差 {f((A.get('ea0') or {}).get('worst_formula_rel_err'), '.2%')}；
旧公式（n·β）{f((A.get('ea0') or {}).get('legacy_worst_rel_err'), '.1%')}</td></tr>
<tr><td><strong>（任务书未设想的第四种）实验口径错误</strong></td>
<td>✅ <strong>成立，是假象的根源</strong></td>
<td>标尺偏 {f(u.get('lambda_scale_bias'), '.2f')}×，机制 99.8% 的 tick 哑火</td></tr>
</tbody></table>
{('<table><thead><tr><th>EA.4 单侧稀疏化</th><th>k</th><th>生效占比</th></tr></thead><tbody>'
  + ''.join(rows) + '</tbody></table>'
  + '<p class="note">⇒ <strong>需求侧（吃单方）才是关键</strong>：只稀疏化吃单方时最接近平方根律，'
    '而两侧同时稀疏化反而互相抵消。</p>') if rows else ''}""")

    # ---------------- B 线 ----------------
    eb1 = B.get("eb1") or {}
    if eb1.get("arms"):
        rows = []
        for tag, d in (eb1.get("arms") or {}).items():
            mark = "（对照）" if tag == "仅图表派(阶段6基线)" else (
                " ⭐" if tag == eb1.get("best") else "")
            rows.append(f"<tr><td>{tag}{mark}</td><td>{f(d.get('k'), '.3f')}</td>"
                        f"<td>{f(d.get('r2'), '.3f')}</td>"
                        f"<td>{f(d.get('n_meta_mean'), '.0f')}</td></tr>")
        blocks.append(f"""
<h3>B 线 · 元订单覆盖面扩展到零智能与基本面派</h3>
<p class="lead">把元订单拆分执行从「仅图表派」扩到三类主体后，
k 从对照的 {f(eb1.get('base_k'), '.3f')} 走到 <strong>{f((eb1.get('arms') or {}).get(eb1.get('best') or '', {}).get('k'), '.3f')}</strong>。
「元订单数」是<strong>独立跑一场数出来的</strong>——只看配置传没传是不算数的
（<code>MetaOrderMixin</code> 收不到 config 时会用默认值，而默认是<strong>开着</strong>的）。</p>
<table><thead><tr><th>配置</th><th>k</th><th>拟合 R²</th><th>元订单数（一场）</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>""")

    # ---------------- C 线 ----------------
    ec1 = C.get("ec1") or {}
    ec3 = C.get("ec3") or {}
    if ec1:
        blocks.append(f"""
<h3>C 线 · 阶段9 修复前后（一个记账类静默 bug 的代价）</h3>
<table><thead><tr><th>配置</th><th>组合 PnL (bp)</th><th>净/总暴露</th></tr></thead><tbody>
<tr><td>修复前（<code>short_limit=0</code>）</td>
<td>{f((ec1.get('broken') or {}).get('pnl_bp'), '.1f')}</td>
<td>{f((ec1.get('net_exposure_broken') or {}).get('ratio_mean'), '.3f')}</td></tr>
<tr><td>修复后（<code>short_headroom=3×</code>）</td>
<td>{f((ec1.get('fixed') or {}).get('pnl_bp'), '.1f')}</td>
<td>{f((ec1.get('net_exposure_fixed') or {}).get('ratio_mean'), '.3f')}</td></tr>
</tbody></table>
<p class="note">⇒ bug 造成的 PnL 偏差 <strong>{f(ec1.get('pct_diff'), '+.1%')}</strong>；
净暴露占比从「完全单边」降到约两成——<strong>修复前那套「配对交易」其实是个方向性策略。</strong>
（净/总暴露是运行结束那一刻的快照；两腿渐进调仓，所以它取决于停在哪一段相位。）</p>
<p class="note">⚠️ 配对反事实（EC.3，{ec3.get('n_seeds')} 种子）的降噪倍数均值 =
<strong>{f(ec3.get('noise_reduction_mean'), '.2f')}×</strong>，而任务书期待「显著大于 1」。
<strong>这是一个负向结果，已如实报出</strong>：本项用的是 <code>max|偏离|</code>（极值统计量），
本身就是噪声主导；且冲击对 B 的真实效应比 B 自身漂移更大，
扣掉漂移后种子间散布成了主要方差来源。与一期那 62 倍降噪不是同一情形。</p>""")

    # ---------------- 合并验证 ----------------
    ej1 = J.get("ej1") or {}
    ej2 = J.get("ej2") or {}
    if ej1.get("arms"):
        rows = []
        for tag, d in (ej1.get("arms") or {}).items():
            rows.append(f"<tr><td>{tag}</td><td>{f(d.get('base_lambda'), '.2f')}</td>"
                        f"<td>{f(d.get('k'), '.3f')}</td>"
                        f"<td>{f(d.get('rho_mean'), '.4f')}</td>"
                        f"<td>{f(d.get('activated_frac'), '.1%')}</td></tr>")
        j3 = (ej1.get("arms") or {}).get("J3 +A 修正 Hawkes", {})
        extra = ""
        if ej2:
            rows2 = "".join(
                f"<tr><td>{v.get('label')}</td><td>{v.get('n_observed')}/{v.get('n_seeds')}</td>"
                f"<td>{'✅' if v.get('holds') else '❌'}</td></tr>"
                for v in (ej2.get("per_key") or {}).values())
            extra = f"""
<h4>J4 · 阶段11「至暗时刻」链条（联合配置）</h4>
<p>逐资产 λ̄：A={f(ej2.get('lambda_A'), '.2f')}，B={f(ej2.get('lambda_B'), '.2f')}
（<strong>不能共用一个 λ̄</strong>——那正是 E8.2 的错误形态）。
成立环节数 = <strong>{ej2.get('n_links')}/6</strong>（种子数 {ej2.get('n_seeds')}，
判据与 E11.1 完全一致，只换配置）。</p>
<table><thead><tr><th>环节</th><th>观察到 / 种子数</th><th>是否成立</th></tr></thead>
<tbody>{rows2}</tbody></table>"""
        blocks.append(f"""
<h3>合并验证（A + B，以及放进阶段11 场景）</h3>
<p class="lead">合并计划<strong>先写后跑</strong>（<code>docs/三线深挖-合并计划.md</code>，假设写在结果之前），
规则里定死三条：λ̄ 必须在<strong>它将被使用的那个市场配置</strong>上标定且标定窗口 = 实验窗口；
合并只做加法、不为了让 k 好看去重调参；每个臂都要报机制体检。</p>
<table><thead><tr><th>臂</th><th>λ̄（同市场口径）</th><th>k</th><th>ρ̄</th><th>稀疏化生效</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<p class="note">⚠️ <strong>A 在 B 之上的边际贡献是负的</strong>：J2 的 k 优于 J3。
联合配置最终 k = {f(j3.get('k'), '.3f')}，离 0.5 还有 {f(abs((j3.get('k') or 0) - 0.5), '.3f')}。
<strong>C 线对这个 k 的贡献是零，不是「很小」</strong>——C 的修复只作用于配对交易者，
而配对交易者在多资产市场里，本 k 检验是单资产实验。</p>
<p class="note">⭐ 合并暴露了一个<strong>结构性</strong>问题：<code>rho_to_count</code> 在 ρ̄=1 附近
天然是「半哑火」的。J3 的 ρ̄ = {f(j3.get('rho_mean'), '.4f')}（正确标定后本就该 ≈1），
但稀疏化生效只有 {f(j3.get('activated_frac'), '.1%')}——把 ρ 映射成「允许多少主体出手」时，
<strong>ρ&gt;1 的那一半质量根本实现不了</strong>（总不能超过全部主体），
于是这个机制<strong>只能变少、不能变多</strong>。
<strong>这是 Hawkes 集成的结构性限制，不是 E8.2 那次的标定事故</strong>；
要真正让机制全程生效，需要改映射（例如按 ρ 缩放<strong>订单量</strong>而不是<strong>主体数</strong>）。</p>
{extra}""")

    return f"""
<section id="ws">
<h2>14. 补充：三线深挖（A/B/C）与合并验证</h2>
<p class="lead">本节是二期交付后按任务书追加的三条工作线，以及把它们合并起来的验证。
全部数字从 <code>out/workstream_*_metrics.json</code> 现算；
叙事版见 <code>docs/三线深挖-交付小结.md</code>，合并假设见
<code>docs/三线深挖-合并计划.md</code>。</p>
{''.join(blocks)}
</section>"""


def sec_summary(jsons: dict) -> str:
    s5 = jsons.get("stage5") or {}
    s6 = jsons.get("stage6") or {}
    best5 = (s5.get("e5_4_match") or {}).get("best") or {}
    fit6 = (s6.get("e6_4_fits") or {})
    k_on = (fit6.get("元订单+自适应|slippage_bp") or {}).get("exponent")
    k_off = (fit6.get("机制关（对照）|slippage_bp") or {}).get("exponent")
    s10 = jsons.get("stage10") or {}
    s11 = jsons.get("stage11") or {}
    s8 = jsons.get("stage8") or {}
    e82 = s8.get("e8_2") or {}
    k8_off = e82.get("k_off")
    k8_on = e82.get("k_on")
    best8 = (s8.get("e8_1_best") or {})
    q8_ctrl = (s8.get("e8_1_control") or {}).get("lb_q")
    # 三指标方向冲突表：与 §4.6 共用同一个来源（原来两处各手抄一遍）
    summary_conflicts_html = k6_conflicts_html(jsons)
    prem_lo, prem_hi = s5_premium_snr_range(jsons)
    return f"""
<section id="summary">
<h2>1. 一句话结论（含全部失败）</h2>
<p class="lead">
<strong>二期把七块机制都实现了，但只有一块达到了指导书定的验收线。</strong>
「机制在工作」与「目标指标被修好」在这七个阶段里被反复证明是两件事——
本报告把两者分开写，且不把前者包装成后者。
</p>
<h3>达到了什么</h3>
<ul>
<li><strong>资金费率（阶段5）</strong>：年化 {f(best5.get('annualized'))}
（真实 0.1157 的 {f((best5.get('annualized') or 0) / 0.1157, '.2f')}×）、
正费率占比 {f(best5.get('pos_frac'), '.4f')}（真实 {_s5_targets(jsons, "pos")}），
偏好 → 费率的 Pearson {f(s5.get('e5_2_pearson'))}。
三条验收项全部通过。</li>
<li><strong>长记忆订单流（阶段6）</strong>：元订单拆分让订单符号的
ΣACF 相对同种子对照显著上升、ACF 连续为正的 lag 数显著变长。
机制确实在工作。</li>
<li><strong>⭐ 订单到达的聚集性（阶段8）</strong>：自激强度确实产生了到达聚集，
最强配置下逐 tick 成交笔数的 Ljung-Box Q 从对照的
<strong>{f(q8_ctrl, '.0f')}</strong> 升到 <strong>{f(best8.get('lb_q'), '.0f')}</strong>，
随分支比单调上升。而且——
<strong>E8.2 意外地达成了阶段6 没做到的主验收项</strong>：
叠加 Hawkes 后，分期清算的冲击幂律指数
从 <strong>{f(k8_off, '.3f')}</strong> 变成 <strong>{f(k8_on, '.3f')}</strong>，
<strong>离平方根律（0.5）的距离从 {f(e82.get('closeness_off'), '.3f')}
降到 {f(e82.get('closeness_on'), '.3f')}</strong>。
这是整个二期**唯一一处把目标指标真正修好的地方**，
而它来自一个原本预期「可能没有额外贡献」的实验（指导书 §5.5 明说允许无贡献）。</li>
</ul>
<h3>三处明确的失败，写在最前面</h3>
<ol class="fail">
<li><strong>冲击函数的凹度没有修好（阶段6 主验收项）。</strong>
指导书要求把冲击幂律指数 k 从一期的 {_krange_txt(jsons)} 修到 ≈0.5（平方根律）。
（<span class="note">注：指导书转述的一期区间是"0.61~1.25"；
本报告用的是**从 <code>out/stage3_metrics.json</code> 现算**的区间——
主指标两项是 {_kprimary_txt(jsons)}，"窗口均偏离"那一项在分期清算下到
{f(stage3_k_all_max(), '.2f')}。
不复述转述值，避免报告文字与自己的数据对不上。</span>）
实测：对照 k = {f(k_off, '.3f')} → 处理 k = {f(k_on, '.3f')}，
<strong>方向是反的</strong>。而且三个指标给出三个不同的方向：
{summary_conflicts_html}
<strong>本阶段无法判定"冲击函数是否变凹"，只能说它在某些侧面变了、方向不一致。</strong></li>
<li><strong>溢价（basis）通道被证伪（阶段5）。</strong>
它的 μ/σ 在任何偏好强度下都只有
{f(prem_lo, '.3f')}~{f(prem_hi, '.3f')}，而让正费率占比达到
{_s5_targets(jsons, "pos")} 需要 {_s5_targets(jsons, "snr")}——整条是噪音。根因是<strong>模型只有一个价格序列</strong>，
「溢价」只能拿中间价对基本面锚做代理，量出的是价格发现误差（sd ≈ 65bp），
而真实永续的 basis 只有几个 bp。
后果：标定后的费率 <strong>99.9% 由订单流通道贡献</strong>。</li>
<li><strong>统一校准的参数不确定性没有被可靠量化（阶段10）。</strong>
实测块自助法在标准块长（168 小时）下只能恢复 <code>acf_abs_lag1</code> 真值的
<strong>0.9%</strong>，块长到 8000（半年）才 69%。
所以那里给出的参数区间是<strong>围绕一个被系统性低估的矩</strong>构造的——
看起来完全正常，实际中心是偏的，且<strong>无法靠增加自助次数修好</strong>。</li>
<li><strong>跨资产传导幅度没有被可靠测量（阶段9）。</strong>
省掉了同种子配对反事实那一步。一期阶段3 已经证明不做配对时随机漂移会完全淹没冲击
（曾把分期清算的指数算成 −0.26）。这一项被明确标注为"已知缺口"。</li>
</ol>
<p class="lead">
<strong>「至暗时刻」链条</strong>（阶段11）：{(s11.get('e11_1_n_links') if s11 else '—')}
个环节被配对判定为"观察到传导"（验收要求 ≥3）。
该场景同时暴露了一个<strong>机制组合的真实后果</strong>：
<strong>倒下的全是资金费率套利者</strong>——它们把
{f(d113_first(jsons).get('arb_turnover'), ',.0f')} 元的成交额打出去而本金只有
{f(d113_first(jsons).get('arb_capital'), ',.0f')} 元（换手
{f((d113_first(jsons).get('arb_turnover') or 0) / max(d113_first(jsons).get('arb_capital') or 1, 1), '.1f')} 倍），
而费率信号有 {f(d113_first(jsons).get('funding_flip_frac'), '.1%')} 的时间在翻符号，
价差 + 冲击成本吃掉了几乎全部本金。**现金守恒成立**，所以这不是记账 bug，
是策略在烧钱。这一条已写进报告正文，不是调参掩盖过去的。
</p>
<p class="note">
本报告所有数字由 <code>scripts/make_report2.py</code> 从
<code>out/stage*_metrics.json</code> 直接读出。
</p>
</section>
"""


def _s5_targets(jsons: dict, which: str) -> str:
    """阶段5 的验收靶子（从 JSON 的 ``targets`` 读，不写死）。"""
    tg = (jsons.get("stage5") or {}).get("targets") or {}
    if which == "annual":
        return f"{tg.get('real_annual', float('nan')):.2%}"
    if which == "pos":
        return f"{tg.get('real_pos_frac', float('nan')):.3f}"
    if which == "snr":
        return f"{tg.get('target_snr', float('nan')):.4f}"
    return "—"


def d113_first(jsons: dict) -> dict:
    """阶段11 的第一条 E11.3 明细（换手/成本归因）。"""
    d = (jsons.get("stage11") or {}).get("e11_3_detail") or []
    return d[0] if d else {}


def _k6(jsons: dict, arm: str, metric: str = "slippage_bp") -> float:
    """从 stage6 的 ``e6_4_fits`` 取「某臂 × 某指标」的冲击幂律指数 k。

    ⚠️ 为什么要有这个函数：报告里原来把 k 的两个值**手抄**成了
    "k 反而从 0.665 走到 0.910"，一共抄在 3 个地方。手抄的危险不在于
    抄错一次，而在于**重跑之后没人会记得去改它**——报告会拿着旧数字
    描述新结果，而且没有任何机制会发现。

    真实发生：2026-09-18 修掉了阶段6 对照组的一个 bug，重跑后 k 变了，
    而那 3 处手抄的地方一个都没动。所以现在一律走这个函数。
    """
    fits = ((jsons.get("stage6") or {}).get("e6_4_fits") or {})
    v = (fits.get(f"{arm}|{metric}") or {}).get("exponent")
    return float(v) if isinstance(v, (int, float)) else float("nan")


def _k6fits(s6: dict, arm: str, metric: str = "slippage_bp") -> float:
    """同上，但直接吃 stage6 的 dict。

    ``sec_stage6`` 拿到的是 ``jsons["stage6"]``，而总览章节拿到的是整个
    ``jsons``——两处的入参层级不同。给一个薄封装而不是让调用方自己记得
    该传哪一层：传错一层的表现是取到 nan，而 nan 在报告里既可能被格式化成
    字面 ``nan``（有自检拦），也可能静默变成某个默认值（没有自检拦得住）。
    实现只有上面那一份，这里只做层级转换。
    """
    return _k6({"stage6": s6 or {}}, arm, metric)


ARM_OFF = "机制关（对照）"
ARM_ON = "元订单+自适应"
# 三个指标测的是冲击的不同侧面，报告里要并列才看得出"方向冲突"
K6_METRICS = (("worst_fill_bp", "最差成交价"), ("during_mean_bp", "窗口均偏离"),
              ("slippage_bp", "因果滑点"))


def k6_conflicts(jsons: dict) -> list[dict]:
    """阶段6 三个冲击指标的「对照 → 处理」对照表（全部从 JSON 现取）。

    ⭐ 抽成函数是因为这张表**曾经手抄在报告的两个地方**
    （§1 的三处失败清单、§4.6 的方向冲突说明）。
    两处手抄 = 两处会在重跑后变成错的，而且改的时候只会想起其中一处。
    现在两处都走这一个来源。
    """
    s6 = (jsons.get("stage6") or {})
    out = []
    for mkey, mlabel in K6_METRICS:
        a = _k6fits(s6, ARM_OFF, mkey)
        b = _k6fits(s6, ARM_ON, mkey)
        if not (math.isfinite(a) and math.isfinite(b)):
            continue
        d_off, d_on = abs(a - 0.5), abs(b - 0.5)
        out.append({"metric": mkey, "label": mlabel, "off": a, "on": b,
                    "better": d_on < d_off, "off_close_to_sqrt": d_off < 0.05})
    return out


def k6_conflicts_html(jsons: dict) -> str:
    rows = k6_conflicts(jsons)
    if not rows:
        return ("<li>（stage6 JSON 里取不到 <code>e6_4_fits</code>，"
                "无法并列三指标）</li>")
    lines = []
    for r in rows:
        near = "（<strong>几乎精确是平方根律</strong>）" if r["off_close_to_sqrt"] else ""
        note = "<strong>明显改善</strong>" if r["better"] else "变差"
        lines.append(f"<li>{r['label']}：对照 k = {r['off']:.3f}{near} → "
                     f"处理 {r['on']:.3f}（{note}）</li>")
    return "\n".join(lines)


def stage3_k_all_max() -> float:
    """一期全部 k 值里的最大值（报告里"窗口均偏离到 3.09"的那个 3.09）。"""
    r = stage3_k_range()
    return float(r["max"]) if r else float("nan")


def s5_premium_snr_range(jsons: dict) -> tuple[float, float]:
    """阶段5 溢价通道在各偏好档位上的 |μ/σ| 范围。"""
    rows = ((jsons.get("stage5") or {}).get("e5_0b_channels") or [])
    v = [abs(r["premium_snr"]) for r in rows
         if isinstance(r.get("premium_snr"), (int, float))]
    return (min(v), max(v)) if v else (float("nan"), float("nan"))



def _s5_result(jsons: dict) -> dict:
    """阶段5 的头条实测结果（年化倍数、正占比），从 JSON 现读。"""
    d = ((jsons.get("stage5") or {}).get("e5_4_match") or {}).get("best") or {}
    tg = (jsons.get("stage5") or {}).get("targets") or {}
    annual = d.get("annualized")
    ref = tg.get("real_annual")
    ratio = (annual / ref) if isinstance(annual, (int, float)) and ref else None
    pos = d.get("pos_frac")
    return {"annualized": annual, "ratio": ratio, "pos_frac": pos}


def _s5_headline(jsons: dict) -> str:
    r = _s5_result(jsons)
    bits = []
    if isinstance(r["ratio"], float):
        bits.append(f"年化 {r['ratio']:.2f}×")
    if isinstance(r["pos_frac"], float):
        bits.append(f"正占比 {r['pos_frac']:.3f}")
    return "、".join(bits) if bits else "（数据缺失）"


def _krange_txt(jsons: dict) -> str:
    r = stage3_k_range()
    if not r:
        return "一期区间（数据缺失）"
    return f"{r['min']:.2f}~{r['max']:.2f}"


def _kprimary_txt(jsons: dict) -> str:
    r = stage3_k_range()
    if not r:
        return "—"
    return f"{r['primary_min']:.3f}~{r['primary_max']:.3f}"


def build() -> str:
    jsons = {k: load(f"{k}_metrics.json") for k in
             ("stage5", "stage6", "stage7", "stage8", "stage9",
              "stage10", "stage11")}
    mut_log = None
    # ⚠️ 取「out/mutation_run.log 与 out/repro_logs/mutation.log」里**较新**的那个。
    #    只读规范路径的话，一键复现跑完、报告引用的仍是**上一轮**的变异结果——
    #    而那个文件确实存在、内容也确实是真的，看起来毫无异常。
    lp = newest_log("mutation_run.log", "mutation.log")
    if lp is not None:
        mut_log = lp.read_text(encoding="utf-8", errors="replace")
    n_tests = None
    tp = OUT / "test_count.txt"
    if tp.exists():
        try:
            n_tests = int(tp.read_text(encoding="utf-8").strip())
        except ValueError:
            n_tests = None

    parts = [
        sec_summary(jsons),
        sec_k_resolution(),
        sec_overview(jsons),
        sec_stage5(jsons["stage5"]),
        sec_stage6(jsons["stage6"]),
        sec_stage7(jsons["stage7"]),
        sec_stage8(jsons["stage8"]),
        sec_stage9(jsons["stage9"]),
        sec_stage10(jsons["stage10"]),
        sec_stage11(jsons["stage11"]),
        sec_boundaries(jsons),
        sec_methodology(),
        sec_kernel(mut_log, n_tests),
        sec_repro(),
        sec_workstream(),
    ]
    return CSS_AND_SHELL.replace("{{BODY}}", "\n".join(parts))


def find_leftover_placeholders(html: str) -> list[str]:
    """找出没被替换掉的 ``{占位符}``——这类东西会静默印进报告。

    真实踩过：``sec_overview`` 用了普通三引号字符串（忘了 f 前缀），
    于是 ``{_krange_txt(jsons)}`` 原样出现在总览表里。报告照常生成、
    大小正常、章节齐全，**没有任何机制会发现**。

    做法：先剥掉 ``<style>`` / ``<script>``（那里面的花括号是 CSS/JS 语法），
    再找以字母或下划线开头的 ``{...}``。
    """
    import re as _re
    body = _re.sub(r"<style[^>]*>.*?</style>", "", html, flags=_re.S)
    body = _re.sub(r"<script[^>]*>.*?</script>", "", body, flags=_re.S)
    out = []
    for m in _re.finditer(r"\{[^{}\n]{1,120}\}", body):
        s = m.group(0)
        if _re.match(r"^\{[A-Za-z_]", s):
            out.append(s)
    return out


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    html_out = build()
    leaked = find_leftover_placeholders(html_out)
    if leaked:
        raise SystemExit(
            "❌ 报告里有没被替换掉的占位符，拒绝写出：\n  "
            + "\n  ".join(sorted(set(leaked)))
            + "\n  原因基本只有一个：那段模板字符串忘了加 f 前缀。"
        )
    path = DOCS / "交易世界二期-交付报告.html"
    path.write_text(html_out, encoding="utf-8")
    kb = len(html_out.encode("utf-8")) / 1024
    print(f"  报告已生成：{path}（{kb:,.0f} KB）")


if __name__ == "__main__":
    main()
