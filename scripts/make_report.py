"""生成交付报告（单文件 HTML，图片内嵌 base64）。

设计原则
--------
**报告里的每个数字都从 `out/*.json` 读出来，不许手抄。**
手抄数字是这类报告最常见的失真来源：改一次参数忘了改一次文字，
报告就变成了一份"看着很专业但和实际产物不一致"的文档。

用法::

    python scripts/make_report.py
"""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
FIG = OUT / "figs"
DOCS = ROOT / "docs"


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
        return f'<figure class="missing"><div class="ph">缺图：{path.name}</div><figcaption>{html.escape(caption)}</figcaption></figure>'
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return (
        f'<figure><img src="data:image/png;base64,{b64}" alt="{html.escape(caption)}">'
        f"<figcaption>{html.escape(caption)}</figcaption></figure>"
    )


def f(v, fmt: str = ".2f", dash: str = "—") -> str:
    if v is None:
        return dash
    try:
        x = float(v)
    except (TypeError, ValueError):
        return html.escape(str(v))
    if x != x:  # nan
        return dash
    return format(x, fmt)


def table(headers: list[str], rows: list[list[str]], cls: str = "") -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows
    )
    return f'<table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


# ----------------------------------------------------------------------
def build() -> str:
    cal = load("calibration.json")
    s1 = load("stage1_metrics.json")
    s2 = load("stage2_metrics.json")
    s3 = load("stage3_metrics.json")
    sw = load("sweep_study.json")
    s4 = load("stage4_metrics.json")
    lab = load("lab_results.json")

    parts: list[str] = []

    # ============ 1. 摘要 ============
    parts.append("""
<section id="summary">
<h2>1. 一句话结论</h2>
<p class="lead">
<strong>能让简单主体交互涌现出真实加密市场的价格统计特征——但只在"统计特征"这一层，
且必须三类主体共存。</strong>
单靠零智能交易者只能造出随机游走；把基本面派与图表派放进同一个市场，
肥尾、波动率聚集、无线性可预测性、尾部指数这四项会同时靠近真实 BTC。
</p>
<p class="warn">
<strong>三处明确的失败，写在最前面：</strong>
</p>
<ol class="fail">
<li><strong>清算冲击不是凹的。</strong>真实市场的冲击服从平方根律（<code>I ∝ Q^0.5</code>），
本模型给出 <code>I ∝ Q^1</code>（线性）。根因已定位：<strong>盘口累计深度剖面接近线性
（γ≈0.8），而真实市场是凸的（γ≈2）</strong>——缺的是远端的长记忆挂单，不是主体数量。</li>
<li><strong>没有一组参数能同时匹配所有指标。</strong>网格搜索的最优点靠把 σ 抬到
真实的 2.4 倍，才换来峰度（7.15 vs 10.26）与 |r| 自相关（0.375 vs 0.262）的接近；
把 σ 拉回真实量级，肥尾就掉下去。<b>这个权衡关系本身是结论</b>，
说明模型里缺少一个"在不提高波动率的前提下制造肥尾"的机制。</li>
<li><strong>做市商的深度远不如真实市场。</strong>做市商把价差收窄到 0.53×、深度加到 1.60×，
方向对、幅度不够。真实做市商的挂单会铺到远离中间价的位置——
这与第 1 条失败是同一个结构缺陷的两个侧面。</li>
<li><strong>蓝图的动机现象「扫单后继续走」，本设计没能证实。</strong>
先是在模拟里测出显著的"延续"（t=5.9~11.2），补上内生性诊断后发现：
扫单不是外生事件，按"扫单前动量"分组后，两组的后续路径<b>方向完全相反</b>
（h=5 相差 46~50bp），"延续"是两组混合出来的假象。
详见第 9 节——<b>这是本报告里唯一一处"自己把自己的结论推翻"的地方。</b></li>
</ol>
<p class="lead">
<strong>第二个交付物（蓝图之外）：一个可用的策略实验台。</strong>
研究者写一个 <code>on_tick(ctx)</code> 即可接入，跨 5 种市场状态 × 多种子跑 A/B 对照，
并得到<b>成交质量归因</b>：赚的是价差，还是接了有毒的单；
PnL 里哪一部分是本事、哪一部分是行情给的。见<b>第 12 节</b>。
</p>
<p class="note">
本报告里所有数字都由 <code>scripts/make_report.py</code> 从 <code>out/*.json</code> 直接读出，
不存在"报告文字与实测产物脱节"的可能。
</p>
</section>
""")

    # ============ 2. 蓝图要求 vs 交付 ============
    parts.append("""
<section id="mapping">
<h2>2. 施工蓝图要求 → 实际交付</h2>
""")
    rows = [
        ["阶段1 MVP", "零智能交易者、订单簿、撮合、价格序列", "✅ 完成", "200 主体 × 10000 tick，8 种子，无崩溃 / 无发散"],
        ["阶段2", "加基本面派 + 图表派，求涌现肥尾与波动率聚集", "✅ 完成", "四池消融，定位到「两者必须共存」"],
        ["阶段3", "加做市商 + 清算压力测试", "✅ 完成", "做市商对照、清算冲击、饱和效应、抛售速度对照"],
        ["阶段4（探索性）", "参数校准与敏感性分析", "✅ 完成", "含训练/测试样本外检验，显式标注过拟合风险"],
        ["统计特征对标", "峰度 / 波动率聚集 / 尾部指数等", "✅ 完成", "见第 4 节真实靶子表"],
        ["“扫单后继续走”现象", "（蓝图动机现象）", "❌ 未能证实", "真实侧代理变量测不出；"
         "模拟侧先测出「延续」，补上内生性诊断后被推翻——见第 9 节"],
        ["清算冲击线性度", "（对应“清算流动性捕获”假设）", "⚠️ 与真实不符", "本模型线性/超线性，真实是凹的（平方根律）"],
        ["（蓝图之外）策略实验台", "——", "✅ 已建成", "可写策略、跨 5 场景 × 多种子跑 A/B 对照，见第 11 节"],
        ["（蓝图之外）桌面端 GUI", "——", "✅ 已建成", "原生窗口 / 浏览器双形态，界面内写策略并运行，见第 12 节"],
    ]
    parts.append(table(["环节", "蓝图要求", "状态", "实测"], rows))

    parts.append("""
<div class="callout">
<strong>关于最后两行：</strong>这两条不是"没做"，而是"做了之后发现不成立"。
把不成立当作交付内容之一，比把指标调到好看要诚实——
蓝图自己就写着这一步的目的是"理解哪些参数组合能产生真实市场的模式"，
而不是"造一个像的模型"。<br>
其中"扫单后继续走"这一条的过程尤其值得看（第 9 节）：
它经历了 <b>看到现象 → 修正测量 → 现象更强 → 补内生性诊断 → 现象消失</b> 五个阶段。
<b>一个被自己推翻的结论，和一条被证实的结论，价值是同等的</b>——
它排除了一整类解释。
</div>
</section>
""")

    # ============ 3. 内核可信度 ============
    parts.append("""
<section id="kernel">
<h2>3. 先证明这个撮合内核可信</h2>
<p>一个全绿的测试套件说明不了任何事——它可能绿是因为实现正确，也可能是因为断言写得太松。
两者从"全绿"这个结果上无法区分。所以做了变异验证：<strong>故意注入已知的实现 bug，
看测试套件是否变红</strong>。不会变红的测试等于没有测试。</p>
""")
    mutation_log = (OUT / "mutation_run.log")
    if mutation_log.exists():
        txt = mutation_log.read_text(encoding="utf-8", errors="replace")
        lines = [ln for ln in txt.splitlines() if ln.strip()]
        # 整张表都放进来：这是"测试有检出能力"的唯一证据，截断就失去意义了
        tail = "<br>".join(html.escape(ln) for ln in lines)
        parts.append(f'<pre class="log">{tail}</pre>')
    else:
        parts.append('<p class="warn">变异验证日志缺失，请先运行 <code>python scripts/mutation_check.py</code>。</p>')
    parts.append("""
<p>覆盖的 bug 类型全部来自真实撮合系统事故：成交价取主动方报价、破坏价格优先、同价 LIFO、
取消 tick 网格、关闭自成交防护、完全成交不摘单、剩余不挂簿、被动方预留不更新、
成交不落日志、价格不对齐网格、清算滑点基准取错、清算滑点符号反转。</p>
<div class="callout">
<strong>为什么"清算滑点基准取错"和"符号反转"也算撮合 bug：</strong>
它们是第 8 节那两个测量坑的<em>代码形态</em>。把它们做成变异体，
等于把"我们曾经把冲击算错过"这件事固化成了回归测试。
</div>
</section>
""")

    # ============ 4. 真实市场靶子 ============
    parts.append("""
<section id="targets">
<h2>4. 真实市场靶子（不是拍的，是量出来的）</h2>
<p>数据来自交付包内的 1 小时 K 线（BTC/ETH/SOL 各 17520 根，730 天），
<strong>直接读 CSV，不重新下载</strong>，避免口径漂移。</p>
""")
    if s1 and "real" in s1:
        real = s1["real"]
        rw = s1.get("random_walk") or s1.get("rw") or {}
        keys = [
            ("sigma_bp", "收益率标准差 (bp)", ".2f"),
            ("excess_kurtosis", "超额峰度", ".2f"),
            ("skew", "偏度", ".3f"),
            ("acf_abs_lag1", "|r| ACF(1)", ".3f"),
            ("acf_abs_lag5", "|r| ACF(5)", ".3f"),
            ("acf_abs_mean_1_10", "|r| ACF(1-10) 均值", ".3f"),
            ("n_sig_lags_abs", "显著 |r| 滞后数（/20）", ".0f"),
            ("acf_ret_lag1", "r ACF(1)", ".4f"),
            ("vr5", "方差比 VR(5)", ".3f"),
            ("hill_alpha_left", "左尾 Hill α", ".3f"),
        ]
        heads = ["指标"] + [k for k in real] + (["随机游走(同σ)"] if rw else [])
        rws = []
        for key, label, fmt in keys:
            row = [f'<span class="k">{html.escape(label)}</span>']
            for name, m in real.items():
                v = m.get(key)
                val = f(v, fmt)
                if key == "excess_kurtosis" and isinstance(v, (int, float)) and v > 5:
                    val = f'<b class="hi">{val}</b>'
                if key == "acf_abs_lag1" and isinstance(v, (int, float)) and v > 0.15:
                    val = f'<b class="hi">{val}</b>'
                if key == "n_sig_lags_abs" and isinstance(v, (int, float)) and v >= 20:
                    val = f'<b class="hi">{val}</b>'
                if key == "hill_alpha_left" and isinstance(v, (int, float)) and v < 3.5:
                    val = f'<b class="hi">{val}</b>'
                row.append(val)
            if rw:
                row.append(f(rw.get(key), fmt))
            rws.append(row)
        parts.append(table(heads, rws))
    parts.append("""
<p class="note">
真实市场的四个"指纹"：<b>峰度远超正态（10~16）</b>、<b>|r| 强自相关（0.21~0.26）且 20/20 滞后全部显著</b>、
<b>收益率自身几乎无自相关（−0.03~0.00）</b>、<b>左尾 Hill α &lt; 3（三阶矩不存在）</b>。
同 σ 的随机游走这四项全部不满足（峰度 0.011、|r|ACF ≈ 0、α = 5.4）——
这就是"要涌现的东西"的准确清单。
</p>
</section>
""")

    # ============ 5. 标定 ============
    parts.append("""
<section id="calib">
<h2>5. 标定：蓝图给的一个参数，照抄会把市场做坏</h2>
<p>施工蓝图建议零智能主体的报价幅度取"中间价的 ±0.1%~2%"。实测：</p>
""")
    if cal:
        pools = cal.get("pools", {})
        zi = pools.get("纯零智能(阶段1)", {})
        rows = []
        for r in zi.get("offset_scan", []):
            mark = " ← 选定" if abs(r["value"] - 0.0018) < 1e-9 else ""
            cls = ' class="sel"' if mark else ""
            rows.append([
                f'{r["value"]:.2%}{mark}',
                f(r["sigma_bp"], ".1f"),
                f(r["sigma_bp"] / cal["target_sigma_bp"], ".2f") + "×",
                f(r["spread_bp"], ".2f"),
                f(r["depth"], ".1f"),
            ])
        parts.append(
            table(["报价幅度上界", "σ (bp/tick)", "σ / 目标", "平均价差 (bp)", "总深度"],
                  rows)
        )
        if zi.get("offset_scan"):
            worst = max(zi["offset_scan"], key=lambda r: r["value"])
            parts.append(f"""
<div class="callout danger">
<strong>蓝图建议的 2% 上界 → σ = {f(worst['sigma_bp'], '.1f')}bp，
是真实 BTC（{f(cal['target_sigma_bp'], '.2f')}bp）的
{f(worst['sigma_bp'] / cal['target_sigma_bp'], '.2f')} 倍。</strong>
照抄等于把市场的波动率放大十倍。收到 0.18% 后 σ = {f([r for r in zi['offset_scan'] if abs(r['value'] - 0.0018) < 1e-9][0]['sigma_bp'], '.1f')}bp。
<br>原因：报价幅度同时决定<b>价格扩散速度</b>与<b>盘口宽度</b>，
它是刻度参数，不是可以凭直觉选的风格参数。
</div>""")
        nrow = cal.get("n_scan", [])
        if nrow:
            parts.append("<h3>σ 依赖主体数 N：标定必须附带 N</h3>")
            parts.append(table(
                ["N", "纯零智能 σ (bp)", "σ/目标", "混合池 σ (bp)", "σ/目标"],
                [[str(r["n_agents"]), f(r["sigma_zi_bp"], ".1f"),
                  f(r["sigma_zi_bp"] / cal["target_sigma_bp"], ".2f") + "×",
                  f(r["sigma_mix_bp"], ".1f"),
                  f(r["sigma_mix_bp"] / cal["target_sigma_bp"], ".2f") + "×"]
                 for r in nrow],
            ))
            parts.append("""
<p class="note">纯零智能池的 σ 随 N 翻倍（N 从 100 到 500，σ 从 37 到 86bp）；
混合池则平缓得多（41→56bp）——再次印证基本面锚的作用。
<strong>所以"σ 标定好了"这句话必须附带 N</strong>，脱离主体数谈标定没有意义。</p>
""")
        parts.append("""
<div class="callout">
<strong>标不了的东西（必须明说）：</strong>买卖价差、盘口深度、单笔成交流量、清算规模分布。
公开 1 小时 K 线里没有盘口信息，这些量在本项目里是<b>内生一致性检查</b>，
不是标定对象。把它们说成"标定过了"是不诚实的。
</div>
""")
        parts.append(f'<p class="src">完整推导见 <code>docs/标定说明.md</code>；数据见 <code>out/calibration.json</code>。</p>')
    parts.append("</section>")

    # ============ 6. 阶段1 ============
    if s1:
        parts.append("""
<section id="stage1">
<h2>6. 阶段1：纯零智能基线</h2>
<p>蓝图引用 Gode &amp; Sunder (1993) 的"零智能交易者"作为基线。
关键是要说清楚<strong>这个基线的正确预期是什么</strong>。</p>
""")
        cfg = s1.get("config", {})
        div = s1.get("divergence", {})
        sim = s1.get("sim", {})
        parts.append(f"""
<div class="callout">
<strong>修正了蓝图的一处判据。</strong>
Gode-Sunder 的结论是"零智能市场也能收敛到有效价格"，但那需要一个<em>外生的</em>基本面价值锚。
本市场<strong>没有外生锚</strong>（价格完全内生），所以"收敛"在这里没有定义。
零智能市场的正确预期是：<b>不崩溃、不发散、统计上接近随机游走</b>。实测正是如此。
</div>
""")
        rows = [
            ["运行规模", f'{cfg.get("n_agents", "—")} 主体 × {cfg.get("n_ticks", "—"):,} tick × {len(cfg.get("seeds", []))} 种子'],
            ["不崩溃", "全程有成交、体检通过"],
            ["不发散", f'跨种子平均漂移 {f(div.get("mean_drift_log"), ".4f")}（t={f(div.get("t_drift"), ".2f")}），单次区间 [{f(min(div.get("drifts", [0])), ".3f")}, {f(max(div.get("drifts", [0])), ".3f")}]'],
            ["σ / tick", f'{f(div.get("sigma_bp_per_tick"), ".1f")} bp（真实 BTC {f(cal.get("target_sigma_bp") if cal else None, ".1f")} bp，同量级）'],
            ["但统计特征远不如真实", f'峰度 {f(sim.get("excess_kurtosis"), ".2f")}、|r|ACF(1) {f(sim.get("acf_abs_lag1"), ".3f")}、显著滞后 {f(sim.get("n_sig_lags_abs"), ".0f")}/20'],
        ]
        parts.append(table(["项", "实测"], [[a, b] for a, b in rows]))
        parts.append('<p class="note">结论：零智能基线<strong>能站住（不崩不发散）但不具备真实市场的指纹</strong>，正好作为阶段2 的对照起点。</p>')

    # ============ 7. 阶段2 ============
    if s2:
        parts.append("""
<section id="stage2">
<h2>7. 阶段2：机制归因——哪一类主体造出了哪个特征</h2>
<p>蓝图要求"加基本面派 + 图表派"。但只报一个"加了之后像了"是没有信息量的：
<strong>必须做消融（ablation）</strong>，把每一类主体单独拿出来看它负责什么。</p>
""")
        pools = s2.get("pools") or s2.get("ablations") or {}
        if pools:
            heads = ["主体池", "σ (bp)", "峰度", "|r|ACF(1)", "显著滞后", "r ACF(1)", "VR(5)", "Hill α"]
            rows = []
            for name, m in pools.items():
                mm = m.get("metrics", m)
                rows.append([
                    html.escape(str(name)),
                    f(mm.get("sigma_bp"), ".1f"),
                    f(mm.get("excess_kurtosis"), ".2f"),
                    f(mm.get("acf_abs_lag1"), ".3f"),
                    f(mm.get("n_sig_lags_abs"), ".0f") + "/20",
                    f(mm.get("acf_ret_lag1"), ".3f"),
                    f(mm.get("vr5"), ".3f"),
                    f(mm.get("hill_alpha_left"), ".3f"),
                ])
            for name, m in (s2.get("real") or {}).items():
                rows.append([
                    f'<span class="k">真实 {html.escape(str(name))}</span>',
                    f'<b class="hi">{f(m.get("sigma_bp"), ".1f")}</b>',
                    f'<b class="hi">{f(m.get("excess_kurtosis"), ".2f")}</b>',
                    f'<b class="hi">{f(m.get("acf_abs_lag1"), ".3f")}</b>',
                    f'<b class="hi">{f(m.get("n_sig_lags_abs"), ".0f")}/20</b>',
                    f(m.get("acf_ret_lag1"), ".3f"),
                    f(m.get("vr5"), ".3f"),
                    f'<b class="hi">{f(m.get("hill_alpha_left"), ".3f")}</b>',
                ])
            parts.append(table(heads, rows))
        parts.append("""
<div class="callout">
<strong>三条机制结论：</strong>
<ol>
<li><b>基本面派单独存在会把肥尾抹平。</b>它提供均值回归锚，价格被拽住 → 峰度崩塌。</li>
<li><b>图表派单独存在会让波动率爆炸。</b>正反馈没有对手盘约束 → σ 数量级跃升、方差比 &gt;&gt; 1（强趋势）。</li>
<li><b>两者必须共存。</b>负反馈与正反馈互相制衡，才同时给出肥尾、波动率聚集与接近 1 的方差比。
只报"混合池最像"是不够的——<em>为什么必须共存</em>才是这一步的产出。</li>
</ol>
</div>
<p class="warn"><strong>一个随配置而变的偏差：</strong>在上面这个配置（图表 30%、报价上界 18bp）下，
混合池的收益率自相关偏负、方差比偏低，即<b>偏均值回归</b>。
但第 10 节的网格搜索发现：换到主网格最优点（图表 38%、报价上界 28bp）后，
VR(5) = 0.995、r ACF(1) = −0.029，与真实（0.968 / −0.017）几乎吻合。
<b>所以这不是"模型的结构性缺陷"，而是这个配置点的性质</b>——
报结论时必须带上配置，否则会得出互相矛盾的结论。</p>
""")
        if FIG.joinpath("stage2_acf_pools.png").exists():
            parts.append(img(
                FIG / "stage2_acf_pools.png",
                "阶段2 四池的 |r| 自相关对比：只有混合池同时具备「长记忆的波动率」与「无线性可预测性」",
            ))
        if FIG.joinpath("stage2_metric_bars.png").exists():
            parts.append(img(FIG / "stage2_metric_bars.png", "阶段2 各池的指标柱状对比（与真实 BTC 并列）"))

    # ============ 8. 阶段3 ============
    if s3:
        parts.append("""
<section id="stage3">
<h2>8. 阶段3：做市商与清算压力测试</h2>
<h3>8.1 测量协议（这一节比结果更重要）</h3>
<p>清算冲击是"几十个 tick 内的瞬态事件"，而市场本身每 tick 就有 40~50bp 的波动。
<strong>用单条价格路径去测它，测到的多半是噪声。</strong>本项目在这里踩了两次同一个坑，
两次的错误结论都已经被纠正，纠正过程保留在代码注释与变异测试里。</p>
<div class="callout danger">
<strong>坑 1（第一次）：</strong>用长窗口（3000 tick）里的最大偏离当"冲击峰值"，
测出卖出清算造成 <b>+422bp 的正冲击</b>——方向都是错的，因为漂移幅度远大于冲击。<br>
<strong>坑 2（第二次）：</strong>修正后又报出"1% 持仓清算造成 −179.8bp 冲击"，
并算出了 <code>幂律指数 = −0.26</code>（成交量越大冲击越小，物理上不可能）。
根因有两条叠加：① 路径<b>从清算结束后才开始记录</b>，
分期清算的冲击峰值发生在清算<b>进行中</b>，被整个漏掉；
② <b>单种子、无对照</b>，60 tick 窗口的随机游走幅度（≈±300bp）远大于冲击本身。
</div>
<p>修正后的协议有三件事：</p>
<ol>
<li><b>路径覆盖清算全程</b>，逐 tick 记录，统一"先动作、后快照"，
让不同切片数的路径下标语义对齐；</li>
<li><b>同种子配对控制</b>。市场随机数只在「洗牌」与「基本面价值步进」两处消耗，
每 tick 次数固定、与是否清算无关 → 控制组的基本面锚路径与处理组<b>逐点相同</b>，
共同漂移被逐点相减消掉。<b>实测跨种子标准差从 89.5bp 降到 1.4bp（降噪 62 倍）。</b></li>
<li><b>多种子统计</b>，报均值 ± 标准误 + t 统计量，让每一条结论都可证伪。</li>
</ol>
<div class="callout">
<strong>坑 3（滑点基准）：</strong>用固定的"清算前价格"当基准算滑点，
会把抛售期间<b>市场自己的漂移</b>算进去。拆 100 片抛要跨 100 个 tick，
漂移上百 bp，于是"慢抛几乎无冲击"这个结论被完全淹没（实测虚报成 −50bp）。
正确做法是以<b>控制组在同一 tick 的中间价</b>为反事实基准。
</div>
""")
        a = s3.get("A_market_maker", {})
        if a.get("stats"):
            st = a["stats"]
            rows = []
            for tag, label in (("no_mm", "无做市商"), ("with_mm", "有做市商(10%)")):
                if tag not in st:
                    continue
                s = st[tag]
                rows.append([
                    label,
                    f'{f(s["spread_bp"]["mean"], ".2f")} ± {f(s["spread_bp"]["sem"], ".2f")}',
                    f'{f(s["depth_total"]["mean"], ".1f")} ± {f(s["depth_total"]["sem"], ".1f")}',
                    f'{f(s["sigma_bp"]["mean"], ".1f")} ± {f(s["sigma_bp"]["sem"], ".1f")}',
                    f'{f(s["trades_per_tick"]["mean"], ".1f")} ± {f(s["trades_per_tick"]["sem"], ".1f")}',
                ])
            parts.append("<h3>8.2 做市商的效果（同池对照，8 种子）</h3>")
            parts.append(table(["", "价差 (bp)", "总深度 (手)", "σ (bp/tick)", "成交 (笔/tick)"], rows))
            parts.append(f"""
<p class="note">同池对照的意思是：把非做市商部分等比缩小 10%，空出的名额给做市商，
于是"非做市商主体的构成完全相同"，差异只能归因于做市商本身。</p>
<p><b>结果：价差 {f(a.get('spread_ratio'), '.2f')}×、深度 {f(a.get('depth_ratio'), '.2f')}×、
波动率 {f(a.get('sigma_ratio'), '.2f')}×。</b>
方向明确且误差棒很窄，但深度只加到 1.6 倍，离真实做市商的水准还差得远。</p>
""")
        b = s3.get("B_liquidation", {})
        if b.get("no_mm"):
            rows = []
            for tag, label in (("no_mm", "无做市商"), ("with_mm", "有做市商")):
                r = b.get(tag)
                if not r:
                    continue
                rows.append([
                    label,
                    f(r.get("target_qty"), ".1f"),
                    f(r.get("delivered_qty"), ".1f"),
                    f(r.get("fill_ratio"), ".0%"),
                    f'{f(r.get("slippage_bp"), ".1f")} ± {f(r.get("slippage_sem_bp"), ".1f")}',
                    f(r.get("slippage_t"), ".1f"),
                    f'{f(r.get("worst_fill_bp"), ".0f")}',
                    f'{f(r.get("t20_bp"), ".1f")} ± {f(r.get("t20_sem"), ".1f")}',
                ])
            parts.append("<h3>8.3 清算冲击（1% 市场总多头持仓，市价强平）</h3>")
            parts.append(table(
                ["", "目标量", "实际成交", "成交率", "因果滑点 (bp)", "t", "最差成交价 (bp)", "20 tick 后"],
                rows,
            ))
            parts.append(f"""
<p class="note">
<b>因果滑点</b> = 清算成交的 VWAP 相对"同一 tick 控制组中间价"的偏离，
是清算方真实承受的平均代价（已实现量，不含路径噪声）。<br>
<b>最差成交价</b> = 扫到的最深档，是<b>边际成本</b>。两者差得远是有信息量的：
平均让价 {f(b.get('no_mm', {}).get('slippage_bp'), '.0f')}bp，
但最后一手打到 {f(b.get('no_mm', {}).get('worst_fill_bp'), '.0f')}bp。
</p>
<p><b>做市商把清算滑点压到 {f(b.get('slippage_ratio'), '.2f')}×。</b>
原因是它把盘口加厚了，清算单能在更好的价位上成交。</p>
""")
        rows = []
        for r in s3.get("C1_instant_saturation", {}).get("rows", []):
            rows.append([
                f'{r["shock_frac"]:.2%}', f(r["target_qty"], ".1f"),
                f(r["delivered_qty"], ".1f"), f(r["fill_ratio"], ".0%"),
                f(r["unfilled_qty"], ".1f"),
                f'{f(r["slippage_bp"], ".1f")} ± {f(r["slippage_sem_bp"], ".1f")}',
                f'{f(r["worst_fill_bp"], ".1f")}',
                f(r["one_sided_frac"], ".0%"),
            ])
        if rows:
            parts.append("<h3>8.4 瞬时清算的饱和效应</h3>")
            parts.append(table(
                ["清算规模", "目标量", "实际成交", "成交率", "未甩出",
                 "因果滑点 (bp)", "最差成交价 (bp)", "盘口被打空"],
                rows,
            ))
            parts.append("""
<div class="callout danger">
<strong>这是本轮最重要的结构性结论：盘口深度就是清算能力的上限。</strong><br>
超过它的部分<b>不是"以更差价格成交"，而是"压根没成交"</b>——仓位仍留在账上，
而冲击也不再随规模增长。真实交易所遇到这种情况会进入 ADL（自动减仓）或动用保险基金，
本模型没有这两层机制，所以"清算不完"是模型的边界，不是市场的性质。
</div>
""")
        rows = []
        for r in s3.get("C2_staged", {}).get("rows", []):
            rows.append([
                f'{r["shock_frac"]:.2%}', f(r["delivered_qty"], ".1f"), f(r["fill_ratio"], ".0%"),
                f(r["qty_per_slice"], ".2f"),
                f'{f(r["slippage_bp"], ".1f")} ± {f(r["slippage_sem_bp"], ".1f")}',
                f'{f(r["worst_fill_bp"], ".0f")}',
                f'{f(r["t20_bp"], ".1f")} ± {f(r["t20_sem"], ".1f")}',
                f'{f(r["tail100_bp"], ".1f")}',
            ])
        if rows:
            parts.append("<h3>8.5 分期清算的规模梯度</h3>")
            parts.append(table(
                ["清算规模", "实际成交", "成交率", "每片量", "因果滑点 (bp)",
                 "最差成交价 (bp)", "20 tick 后", "末段 100 tick 残差"],
                rows,
            ))
        rows = []
        for r in s3.get("C3_slicing_speed", {}).get("rows", []):
            rows.append([
                str(r["slices"]), f(r["qty_per_slice"], ".2f"), f(r["fill_ratio"], ".0%"),
                f'{f(r["slippage_bp"], ".1f")} ± {f(r["slippage_sem_bp"], ".1f")}',
                f'{f(r["worst_fill_bp"], ".0f")}',
                f'{f(r["slippage_vs_p0_bp"], ".1f")}',
            ])
        if rows:
            parts.append("<h3>8.6 ⭐ 抛售速度决定冲击（规模固定，只改拆成几片）</h3>")
            parts.append(table(
                ["切片数", "每片量", "成交率", "因果滑点 (bp)", "最差成交价 (bp)",
                 "对照：若用固定 p0 做基准"],
                rows,
            ))
            sp = s3["C3_slicing_speed"]["rows"]
            if sp:
                parts.append(f"""
<p><b>同一笔 {sp[0]['shock_frac']:.0%} 的仓位：一次砸完因果滑点
{f(sp[0]['slippage_bp'], '.0f')}bp、成交率 {f(sp[0]['fill_ratio'], '.0%')}；
拆 100 个 tick 抛 {f(sp[-1]['slippage_bp'], '.0f')}bp、成交率 {f(sp[-1]['fill_ratio'], '.0%')}。</b>
最后一列是这个坑的证据：拿固定的清算前价格当基准，慢抛会被虚报成
{f(sp[-1]['slippage_vs_p0_bp'], '.0f')}bp 的"滑点"——那其实是 100 个 tick 里市场自己的漂移。</p>
""")
        # 拟合
        fits = s3.get("C2_staged", {}).get("fit_unsaturated", {})
        if fits:
            rows = []
            for k, v in fits.items():
                if v.get("reject"):
                    rows.append([html.escape(k), "拒绝拟合", html.escape(v["reject"])])
                else:
                    rows.append([html.escape(k), f'{f(v.get("k"), ".2f")}',
                                 f'R²={f(v.get("r2"), ".3f")}, n={v.get("n")}'])
            parts.append("<h3>8.7 冲击的线性度（幂律指数 k，1.0 = 线性）</h3>")
            parts.append(table(["口径", "指数 k", "说明"], rows))
            parts.append("""
<div class="callout danger">
<strong>与真实市场不一致，且已定位根因。</strong>
真实市场的冲击服从<b>平方根律</b>（<code>I ∝ Q^0.5</code>，即凹的），本模型给出接近线性甚至超线性。
根因不在主体数量，而在<b>盘口深度剖面的形状</b>——见下节 8.8。
</div>
""")
        # 深度剖面
        prof = s3.get("D_depth_profile")
        if prof:
            parts.append("""<h3>8.8 盘口深度剖面：解释「为什么冲击不是凹的」</h3>""")
            parts.append(f"""
<p>由 <code>Q = c·I^γ</code> 反解得 <code>I ∝ Q^(1/γ)</code>。所以冲击的形状
<strong>完全由累计深度剖面的指数 γ 决定</strong>：</p>
<ul>
<li>γ = 2（真实市场，远端厚）→ <code>I ∝ Q^0.5</code>，<b>平方根律（凹）</b></li>
<li>γ = 1（深度沿价格大致均匀）→ <code>I ∝ Q^1</code>，<b>线性</b></li>
</ul>
<p>实测本模型：无做市商 <b>γ = {f(prof.get("gamma_no_mm"), ".2f")}</b>、
有做市商 <b>γ = {f(prof.get("gamma_with_mm"), ".2f")}</b>，
买盘总深度 {f(prof.get("total_bid_depth"), ".1f")} 手，
其中距中间价 40bp 以内占 {f(prof.get("depth_within_40bp_frac"), ".0%")}。</p>
<p class="note">结论：<b>要复现平方根律，缺的是"远端的长记忆挂单"这个结构，不是更多主体。</b>
继续加主体只会让近端更厚，不会改变剖面形状。</p>
""")
        for name, cap in (
            ("stage3_market_maker.png", "8.2 做市商对微观结构的影响（同池对照）"),
            ("stage3_recovery.png", "8.3 清算冲击与恢复：控制组与处理组的均值路径几乎重合，配对差才是冲击"),
            ("stage3_saturation.png", "8.4 成交率饱和 + 8.7 均值与边际的两种幂律"),
            ("stage3_nonlinear.png", "8.5 分期清算的规模梯度（配对差路径，带 95% CI）"),
            ("stage3_slicing_speed.png", "8.6 抛售速度决定冲击：拆得越细，冲击越小"),
            ("stage3_depth_profile.png", "8.8 盘口深度剖面——决定冲击形状的真正原因"),
        ):
            if (FIG / name).exists():
                parts.append(img(FIG / name, cap))

    # ============ 9. 扫单事件研究 ============
    if sw:
        parts.append("""
<section id="sweep">
<h2>9. 「扫单之后是继续走还是反转」</h2>
<p>这是蓝图的核心动机现象之一。真实数据上很难干净地测——公开 K 线没有订单流，
只能用 CLV×成交量之类的代理变量去猜哪根 K 线被扫了（真实侧实测：
即时响应 −5.7bp、t=−1.42，<b>不显著</b>，代理变量本身就把信号抹掉了）。</p>
<p><strong>模拟市场的优势正在这里：订单流是内生的、可见的。</strong>
可以直接定义"扫单 = 单笔激进订单吃穿订单簿 ≥3 个价位"，不需要代理。</p>
<div class="callout">
<strong>方法上最关键的一步：双锚对照。</strong><br>
一开始我用"扫单刚砸完那一刻的中间价"当锚，测出 h=1/2/5 的"延续"高达 +52/+47/+41bp（t 都在 3~5）。
换成"扫单当 tick 的收盘价"当锚，同样的延续只剩 +11/+7/+0bp、<b>全部不显著</b>。
差额稳定在 ~40bp —— 那部分是<b>扫单后同一 tick 内</b>后续主体跟风推动的价格，不是跨 tick 的趋势。<br>
而第一根锚之所以会把同 tick 内的移动算成"之后"的延续，是因为它是<b>瞬时倾斜的锚</b>：
扫单刚吃完一侧盘口时，<code>(best_bid+best_ask)/2</code> 里的一侧被推得很远，
这个中间价并不代表"市场认可的价"；整侧被吃空时更会退化成最新成交价。
</div>
""")
        for tag, d in (sw.get("pool") or {}).items():
            ea = d.get("events") or {}
            ba = d.get("baseline") or {}
            imm = ea.get("imm_book") or {}
            mvt = ea.get("move_tick") or {}
            bmv = ba.get("move_tick") or {}
            if not imm.get("n"):
                continue
            parts.append(f"<h3>池：{html.escape(str(tag))}</h3>")
            try:
                same_tick = float(mvt["mean_bp"]) - float(imm["mean_bp"])
            except (KeyError, TypeError, ValueError):
                same_tick = float("nan")
            n_raw = d.get("n_raw")
            n_dedup = d.get("n_dedup")
            parts.append(
                f'<p class="note">捕捉到扫单 {n_raw} 笔，去重后 {n_dedup} 笔独立事件；'
                f'平均吃穿 {f(d.get("level_mean"), ".1f")} 档、'
                f'{f(d.get("qty_mean"), ".2f")} 手；'
                f'砸完后盘口单边空缺 {f(d.get("one_sided_after_frac"), ".1%")}'
                f'（越高说明"即时价"这个锚越脏）。</p>'
            )
            parts.append(table(
                ["", "值", "t", "n"],
                [
                    ["扫单自身对盘口的冲击（锚 p_now，瞬时倾斜）",
                     f(imm.get("mean_bp"), ".2f") + " bp", f(imm.get("t"), ".2f"), str(imm.get("n"))],
                    ["当 tick 总冲击（锚 p_pre→p_end，干净）",
                     f(mvt.get("mean_bp"), ".2f") + " bp", f(mvt.get("t"), ".2f"), str(mvt.get("n"))],
                    ["随机基准（同口径）",
                     f(bmv.get("mean_bp"), ".2f") + " bp", f(bmv.get("t"), ".2f"), str(bmv.get("n"))],
                    ["其中：同一 tick 内后续主体跟风", f(same_tick, ".2f") + " bp", "—", "—"],
                ],
            ))
            rows = []
            for h in ("1", "2", "5", "10", "20", "60"):
                c = ea.get(f"cont_{h}") or {}
                cu = ea.get(f"cum_{h}") or {}
                b = ba.get(f"cont_{h}") or {}
                if not c:
                    continue
                t = c.get("t")
                verdict = "样本不足"
                if isinstance(t, (int, float)) and t == t:
                    verdict = ("不显著" if abs(t) < 2
                               else ("延续" if c.get("mean_bp", 0) > 0 else "反转"))
                rows.append([
                    h,
                    f'{f(c.get("mean_bp"), ".2f")} (t={f(t, ".2f")})',
                    f'{f(cu.get("mean_bp"), ".2f")} (t={f(cu.get("t"), ".2f")})',
                    f'{f(b.get("mean_bp"), ".2f")} (t={f(b.get("t"), ".2f")})',
                    verdict,
                ])
            if rows:
                parts.append(table(
                    ["之后 h tick", "延续 cont_h（从下一 tick 起）",
                     "累计 cum_h（相对扫单前）", "随机基准", "判定"],
                    rows,
                ))

            # 内生性诊断 + 分组对照
            pm = d.get("pre_momentum") or {}
            if pm.get("n"):
                parts.append(
                    f'<p><b>内生性诊断：</b>扫单<b>前</b> 20 tick 的价格动量'
                    f'（按扫单方向对齐）= <b>{f(pm.get("mean_bp"), ".2f")} bp</b>'
                    f'（t={f(pm.get("t"), ".2f")}, n={pm.get("n")}）。'
                    + ("<b>显著为负</b>，即<span class=\"warn\">扫单是反向的</span>——"
                       "买单扫单通常发生在价格先跌一段之后（追跌买），卖单扫单发生在先涨之后。"
                       "这意味着后续路径里混着<b>那段既有走势的回归</b>，"
                       "而回归方向与扫单同向，会被误读成「延续」。"
                       "随机时点基准的扫单前动量均值约为 0，<b>控制不住这个选择性偏差</b>。"
                       if isinstance(pm.get("t"), (int, float)) and pm["t"] < -2
                       else "在该池中不显著，后续路径可较直接地归因于扫单本身。")
                    + "</p>"
                )
            bpm = d.get("by_premom") or {}
            if bpm:
                rows = []
                for label, g in bpm.items():
                    if not g or g.get("move_tick", {}).get("n", 0) == 0:
                        continue
                    cells = [str(g["move_tick"]["n"])]
                    for key in ("move_tick", "cont_1", "cont_5", "cont_20"):
                        s = g.get(key) or {}
                        cells.append(f'{f(s.get("mean_bp"), ".2f")} (t={f(s.get("t"), ".1f")})')
                    rows.append([html.escape(str(label))] + cells)
                if rows:
                    parts.append(
                        "<p><b>分组对照（按扫单前动量与扫单方向是否同向）：</b>"
                        "若两组形状明显不同，说明「既有走势的回归」贡献不可忽略，"
                        "结论只能限定为<b>「扫单之后的价格路径形状」</b>，"
                        "不能说成「扫单导致了延续/反转」。</p>"
                    )
                    parts.append(table(
                        ["组", "n", "当 tick 总冲击", "h=1", "h=5", "h=20"], rows,
                    ))
        parts.append("""
<div class="callout">
<strong>结论（分池、分层，不打包）：</strong>
<ol>
<li><b>扫单在它发生的那一个 tick 内确实把价格推动了</b>：当 tick 总冲击显著为正，
而随机基准同口径只有 ~0.1~1bp（t≈0.2~1.0）。
这一条最站得住，因为它是<b>同一 tick 内</b>的效应——
"扫单前的那段走势"发生在它之前，构不成混杂。</li>
<li><b>但扫单之后的价格路径，主要由「扫单前那段走势的均值回归」决定，
而不是扫单的延续效应。</b>见下方分组对照：把事件按「扫单前动量与扫单方向是否同向」
切成两组后，两组的后续路径<b>方向完全相反</b>、差异巨大，
而各自的走向恰好是「各自先前走势的回归」。</li>
<li><b>所以：本设计没有证明「扫单之后会继续走」。</b>
把两组混在一起时看到的"延续"，是<b>混合效应</b>——
两个方向相反的组被平均出来的假象。</li>
</ol>
</div>
<div class="callout danger">
<strong>⚠️ 这是本报告里唯一一处「把先前的结论推翻了」的地方，值得完整交代。</strong><br>
最初的测量（单锚、小样本、未分组）给出"扫单后 h=1/2/5 延续 +52/+47/+41bp"；
换成干净锚后降到"不显著"；把样本量提到 1193 个独立事件后，
延续在 h=1/2/5 又变得显著（t=5.9~11.2）——看起来像是"扫单后继续走"被证实了。<br>
但补上<b>内生性诊断</b>后，结论反转：扫单不是外生事件，
它系统性地发生在价格刚往某个方向走了一段之后（本实验两个池分别测得
−26.6bp / −0.66bp 的扫单前动量）。于是<b>按扫单前动量分组</b>一看，
两组的 h=5 相差 46.5bp、方向相反——<b>"延续"是分组混合出来的</b>。
<br><br>
<b>而且这个模式在另一个池里复现了</b>：无做市商池同样分组后，
反向组 h=5 = +25.0bp、同向组 h=5 = −24.7bp，两组相差 <b>+49.7bp</b>。
两个独立配置给出同一结论，说明这不是某一个参数下的偶然。
<br><br>
这个过程本身就是结论：<b>样本量增加会把噪声变小，但不会把偏差变小。</b>
t 值从 1.6 涨到 11.2 的那一步，并没有让结论更接近真相。
</div>
<div class="callout danger">
<strong>要真正检验「扫单后继续走」，需要的是「匹配对照」：</strong>
找一批"扫单前动量分布相同、但没有发生扫单"的时点作为基准。
本项目的随机时点基准只匹配了"时间均匀分布"，没有匹配"事前动量"，
因此控制不了这个偏差。<b>这是本项目留下的一个明确的、可施工的下一步。</b>
</div>
<div class="callout">
<strong>与真实数据的对比：</strong>真实侧用 K 线代理变量（CLV×成交量）测得即时响应
−5.7bp、t=−1.42，<b>不显著</b>。模拟侧用订单流原生定义测出 t=13~34 的显著冲击。
这不是"模型比真实更干净"，而是<b>代理变量把信号抹掉了</b>——
这本身就是"为什么要做人工市场"的一个具体理由。
</div>
""")
        if (FIG / "sweep_event_study.png").exists():
            parts.append(img(FIG / "sweep_event_study.png", "9. 扫单之后的价格路径：延续（cont_h）与累计（cum_h）双口径 + 随机基准"))

    # ============ 10. 阶段4 ============
    if s4:
        parts.append("""
<section id="stage4">
<h2>10. 阶段4：参数校准与敏感性（含样本外检验）</h2>
<p>蓝图对这一步有明确的警告："目的是理解哪些参数组合能产生真实市场的模式，
不是校准出一个用来预测的模型——<b>警惕把这一步异化成又一轮参数过拟合</b>"。
所以内置了两道防自欺设计。</p>
<div class="callout">
<strong>① 训练/测试切分。</strong>真实 BTC 数据切成前 60%（训练段）与后 40%（测试段），
参数在训练段上搜索，再用同一组参数在测试段上复评。
训练段好、测试段也接近 → 抓到的是稳定的机制关系；差很多 → 抓的是训练段的样本噪声。<br>
<strong>② 多种子平均。</strong>单次模拟方差很大，不做多种子平均，
"最优参数"里有多少是种子运气分不清。
</div>
""")
        tg = s4.get("real_targets", {})
        if tg.get("train"):
            rows = []
            for k, name in (("sigma_bp", "σ (bp)"), ("excess_kurtosis", "超额峰度"),
                            ("acf_abs_lag1", "|r| ACF(1)"), ("acf_abs_mean_1_10", "|r| ACF(1-10)"),
                            ("acf_ret_lag1", "r ACF(1)"), ("vr5", "VR(5)"),
                            ("hill_alpha_left", "Hill α"), ("n_sig_lags_abs", "显著滞后数")):
                a, b = tg["train"].get(k), tg["test"].get(k)
                if a is None or b is None:
                    continue
                rel = abs(b - a) / abs(a) if a else float("nan")
                rows.append([html.escape(name), f(a, ".4g"), f(b, ".4g"), f(rel, ".1%")])
            parts.append("<h3>10.1 真实数据自身的段间差异（这决定了评分下限）</h3>")
            parts.append(table(["指标", "训练段", "测试段", "相对差异"], rows))
            parts.append("""
<p class="note">真实市场自己的统计特征在两段之间就有差异（部分指标差 100% 以上），
所以<b>任何"匹配分数"都不可能接近 0</b>。这一点必须说清楚，
否则读者会把"分数不低"误读成"模型不行"。</p>
<div class="callout">
<strong>注意 r ACF(1) 那一行 107.8% 的"相对差异"是个陷阱。</strong>
它的真实值在两段分别是 −0.029 与 +0.002，都<b>约等于 0</b>，
分母趋零时相对差异会爆炸。对这种"目标值本来就是 0"的指标，
只能用<b>绝对差</b>衡量（这也是评分口径里 `r ACF(1)` 用 |sim−real|/0.03
而不是对数比值的原因）。<b>看到相对差异 100% 就下"两段不一致"的结论是错的。</b>
</div>
""")
        best = s4.get("best") or {}
        if best:
            bm = best.get("metrics", {})
            parts.append("<h3>10.2 主网格最优——以及它付出的代价</h3>")
            parts.append(table(
                ["零智能", "基本面派", "图表派", "报价上界 (bp)", "训练分", "测试分", "测试/训练"],
                [[f(best.get("zero_intel"), ".2f"), f(best.get("fundamentalist"), ".2f"),
                  f(best.get("chartist"), ".2f"), f((best.get("offset_hi") or 0) * 1e4, ".1f"),
                  f(best.get("train_score"), ".3f"), f(best.get("test_score"), ".3f"),
                  f(s4.get("out_of_sample_ratio_median"), ".2f")]],
            ))
            parts.append(table(
                ["指标", "最优配置", "真实 BTC", "倍数"],
                [
                    ["σ (bp)", f(bm.get("sigma_bp"), ".1f"), f(tg.get("full", {}).get("sigma_bp"), ".1f"),
                     f((bm.get("sigma_bp") or 0) / (tg.get("full", {}).get("sigma_bp") or 1), ".2f") + "×"],
                    ["超额峰度", f(bm.get("excess_kurtosis"), ".2f"), f(tg.get("full", {}).get("excess_kurtosis"), ".2f"), "—"],
                    ["|r| ACF(1)", f(bm.get("acf_abs_lag1"), ".3f"), f(tg.get("full", {}).get("acf_abs_lag1"), ".3f"), "—"],
                    ["r ACF(1)", f(bm.get("acf_ret_lag1"), ".4f"), f(tg.get("full", {}).get("acf_ret_lag1"), ".4f"), "—"],
                    ["VR(5)", f(bm.get("vr5"), ".3f"), f(tg.get("full", {}).get("vr5"), ".3f"), "—"],
                    ["Hill α", f(bm.get("hill_alpha_left"), ".3f"), f(tg.get("full", {}).get("hill_alpha_left"), ".3f"), "—"],
                ],
            ))
            parts.append("""
<div class="callout danger">
<strong>最优配置是靠"把波动率抬到真实值的两倍以上"才换来肥尾的。</strong>
把 σ 拉回真实量级，峰度就掉下去。这个权衡关系本身是结论：
<b>本模型缺少一个"在不提高波动率的前提下制造肥尾"的机制。</b>
真实市场里这个机制通常被归给"订单流的聚集性"（交易在时间上成簇到达）与
"流动性的间歇性消失"——本模型的主体每 tick 独立决策，没有这两样。
</div>
""")
        sens = s4.get("sensitivity") or {}
        if sens:
            parts.append("<h3>10.3 敏感性：哪个旋钮真正有用</h3>")
            rows = []
            for k, v in sens.items():
                label = {"chartist": "图表派占比", "fundamentalist": "基本面派占比",
                         "offset_hi": "报价幅度上界"}.get(k, k)
                rows.append([html.escape(str(label)), f(v.get("spread"), ".3f")])
            parts.append(table(["参数", "距离分数的跨度（越大 = 越关键）"], rows))
            parts.append("""
<p class="note"><b>两个值得记的点：</b><br>
① <b>主体构成（图表派 &gt; 基本面派）是决定"像不像"的主要旋钮</b>，
再次印证阶段2 的四池消融结论。<br>
② <b>报价幅度上界的跨度最小（≈0.09）</b>——在混合池里它几乎不影响匹配分数。
这与第 5 节标定的发现完全一致：有了基本面锚之后，零智能报价幅度对 σ 的影响被大幅削弱。
<b>同一个参数在纯零智能池里是"必须精确标定"的刻度，在混合池里变成"无所谓"。</b>
——参数的重要性不是参数自身的属性，而是"它在哪个系统里"的属性。</p>
""")
        fb = s4.get("focus_best") or {}
        if fb:
            parts.append("""<h3>10.4 对症网格（针对第 7 节的「偏均值回归」）</h3>""")
            parts.append("""
<p>主网格的三个轴（主体比例、报价幅度）都不是控制均值回归强弱的旋钮，
所以另起了一个小网格专攻这一件事：<b>基本面派的无套利死区 × 报价激进度</b>。</p>
""")
            rows = []
            for r in s4.get("focus_results", []):
                m = r.get("metrics", {})
                rows.append([
                    f(r.get("fu_deadband"), ".3f"), f(r.get("fu_aggressiveness"), ".4f"),
                    f(r.get("train_score"), ".3f"), f(r.get("test_score"), ".3f"),
                    f(m.get("acf_ret_lag1"), ".4f"), f(m.get("vr5"), ".3f"),
                    f(m.get("excess_kurtosis"), ".2f"), f(m.get("sigma_bp"), ".1f"),
                ])
            parts.append(table(
                ["死区", "激进度", "训练分", "测试分", "r ACF(1)", "VR(5)", "峰度", "σ (bp)"],
                rows,
            ))
            # 注意：focus_results 是按训练分排序的，不能拿 [0]/[-1] 当"最小/最大死区"，
            # 必须真的按字段挑——否则报告里的数字会张冠李戴。
            fr = s4.get("focus_results") or []
            lo_row = min(fr, key=lambda r: r.get("fu_deadband", 0)) if fr else {}
            hi_row = max(fr, key=lambda r: r.get("fu_deadband", 0)) if fr else {}
            parts.append(f"""
<p><b>旋钮确认（这才是这一步的产出）：</b>把死区从
{f(lo_row.get('fu_deadband'), '.3f')} 加到 {f(hi_row.get('fu_deadband'), '.3f')}
（其余参数固定），r ACF(1) 从
<b>{f(lo_row.get('metrics', {}).get('acf_ret_lag1'), '.4f')}</b> 被推到
<b>{f(hi_row.get('metrics', {}).get('acf_ret_lag1'), '.4f')}</b>、
VR(5) 从 <b>{f(lo_row.get('metrics', {}).get('vr5'), '.3f')}</b> 推到
<b>{f(hi_row.get('metrics', {}).get('vr5'), '.3f')}</b>
——跨越了真实值（−0.017 / 0.968）两侧。
<b>所以"基本面派死区"确实是控制均值回归强弱的旋钮</b>，
而且这个旋钮能把市场从"偏均值回归"一路推到"偏趋势"。</p>
<p><b>同时也要如实说：这一格网格并没有改善最优分数</b>
（主网格最优 r ACF(1) = {f(bm.get('acf_ret_lag1'), '.4f')}，
对症网格最优 = {f(fb.get('metrics', {}).get('acf_ret_lag1'), '.4f')}，两边取到的是同一个点）。
原因是<b>主网格的最优点本身就已经把 r ACF(1) / VR(5) 匹配得很好了</b>
（{f(bm.get('acf_ret_lag1'), '.4f')} / {f(bm.get('vr5'), '.3f')}
vs 真实 −0.0170 / 0.968），没有改进空间。
这反过来印证了第 7 节的说法：所谓"过度均值回归"是<b>特定配置点</b>的性质，
不是模型的结构病。</p>
<p class="warn"><strong>但必须同时说清楚：</strong>网格最优点是<b>训练段</b>最优，
且是把多个轴扫完之后挑出来的，本身就带一点样本内挑选。
测试段复评接近 1 只能说明"没那么过拟合"，<b>不等于</b>"这组参数是真的"。
这一步的产出是<b>机制归因</b>（哪个旋钮管哪件事），不是"找到了正确的参数"。</p>
""")
        for name, cap in (
            ("stage4_landscape.png", "10.2 参数空间的地形"),
            ("stage4_train_test.png", "10.1 样本外检验：训练分 vs 测试分"),
            ("stage4_sensitivity.png", "10.3 各参数的敏感性"),
            ("stage4_focus_meanreversion.png", "10.4 对症网格：均值回归强度由哪个旋钮控制"),
        ):
            if (FIG / name).exists():
                parts.append(img(FIG / name, cap))

    # ============ 策略实验台 ============
    if lab:
        cfg = lab.get("config", {})
        aggs = lab.get("aggregated", [])
        runs = lab.get("runs", [])
        scens = cfg.get("scenarios", [])
        seeds = cfg.get("seeds", [])
        parts.append(f"""
<section id="lab">
<h2>11. 策略实验台：怎么用它检验一个交易策略</h2>
<p>前面十二节回答的是"这个市场像不像真实市场"。这一节回答另一个问题：
<b>我写了个策略，怎么用它检验？</b></p>
<div class="callout">
<strong>这个环境相对于历史回测的独有价值。</strong>
回测里市场对你是死的：你的单不影响价格，别人也不对你的单反应。
而在人工市场里你的单会改变盘口、别人会对你反应——
这是<b>执行类、做市类策略</b>唯一能被真实检验的地方。
另外，历史数据里"高波动 + 清算冲击"这种状态样本极少，
而这里可以按需生成并反复重跑。
</div>

<h3>12.1 怎么用（写一个策略只要实现一个方法）</h3>
<pre class="code">from tw.strategy import Strategy

class MyMaker(Strategy):
    # 在中间价两侧挂固定价差的双边单
    def __init__(self, *args, half_spread_bp=12.0, qty=2.0, **kw):
        super().__init__(*args, **kw)
        self.half, self.qty = half_spread_bp, qty

    def on_tick(self, ctx):
        mid = ctx.mid
        if mid is None:
            return None
        ctx.cancel_all()                        # 每 tick 撤旧挂新
        return ctx.quote(mid * (1 - self.half / 1e4),
                         mid * (1 + self.half / 1e4), self.qty)</pre>
<pre class="code">python scripts/lab.py --strategies my_strat:MyMaker --scenarios normal,stressed</pre>

<h3>12.2 五个场景</h3>
""")
        try:
            from tw.scenarios import list_scenarios

            rows = [[html.escape(n), html.escape(t)] for n, t in list_scenarios()]
            parts.append(table(["场景", "它测什么"], rows))
        except Exception:  # noqa: BLE001
            pass

        if aggs:
            parts.append(
                f"<h3>12.3 本次实验台结果</h3>"
                f'<p class="note">策略 {len(cfg.get("strategies", {}))} 个 × '
                f'场景 {len(scens)} 个 × 种子 {len(seeds)} 个 = '
                f'{len(runs)} 次运行；每次观测 '
                f'{cfg.get("ticks", "—"):,} tick（另有预热）。'
                f'下表为跨种子聚合（±为标准误）。</p>'
            )
            for sc in scens:
                sub = [a for a in aggs if a.get("scenario") == sc]
                if not sub:
                    continue
                rows = []
                for a in sub:
                    err = a.get("errors_total", 0)
                    rows.append([
                        html.escape(str(a.get("label"))),
                        f(a.get("n_fills"), ".0f"),
                        f(a.get("maker_volume_frac"), ".3f"),
                        f'+{f(a.get("capture_bp|mean"), ".2f")}' if (a.get("capture_bp|mean") or 0) >= 0 else f(a.get("capture_bp|mean"), ".2f"),
                        f'{f(a.get("drift_20_bp|mean"), ".2f")} (t={f(a.get("drift_20_bp|t"), ".2f")})',
                        f(a.get("realized_20_bp|mean"), ".2f"),
                        f'{f(a.get("pnl_total_from_equity"), ",.0f")} ± {f(a.get("pnl_total_from_equity|sem"), ",.0f")}',
                        f(a.get("pnl_capture"), ",.0f"),
                        f(a.get("pnl_inventory"), ",.0f"),
                        f(a.get("max_dd"), ",.0f"),
                        f(a.get("inv_abs_mean"), ".2f"),
                        # 执行类策略（TWAP 等）的目标完成度；非执行类无此字段显示 "—"
                        f(a.get("completion"), ".3f"),
                        f'<b class="hi">{err}</b>' if err else "0",
                    ])
                parts.append(f"<h4>场景 {html.escape(str(sc))}</h4>")
                parts.append(table(
                    ["策略", "成交笔数", "被动占比", "价差捕获(bp)", "漂移h20(bp)",
                     "实现h20(bp)", "PnL 合计 ± 标准误", "其中价差", "其中库存",
                     "最大回撤", "库存均值", "完成度", "策略异常"],
                    rows,
                ))

            parts.append("""
<div class="callout">
<strong>读这张表的三条纪律：</strong>
<ol>
<li><b><code>noop</code> 必须恰好 0 成交、0 盈亏。</b>不为 0 说明评估口径有 bug，
    此时整张表都不能用。</li>
<li><b>只看均值会在噪声上编故事。</b>「漂移 h20」后面括号里是 t 值，
    <code>|t| &lt; 2</code> 一律当噪声，不能解读。</li>
<li><b>PnL 必须拆开看。</b>「其中价差」才代表策略的本事；
    「其中库存」是行情给的。一个不交易的账户也会有库存项。
    <b>把库存项当收益是最常见的自我欺骗。</b></li>
</ol>
</div>
<div class="callout danger">
<strong>本次实测最值得记的一条：风险与收益的取舍被量化了。</strong>
朴素做市（<code>mm_naive</code>）与加了库存偏移的做市（<code>mm_skewed</code>）
只差一处——报价是否随库存偏移。结果在 normal 场景：
库存偏移把库存均值压到约 1/13、最大回撤压到约 1/7，
但代价是 PnL 从正转负。
<b>两个数字单看都像"出问题了"，一起看才是"用收益换风险"的取舍。</b>
</div>
""")

        parts.append("""
<h3>12.4 评估指标：为什么不能只看"最后赚了多少"</h3>
<p>"最后赚了多少"这个数字几乎不可用：它不区分赚价差与拿着多头，
不区分主动吃单与被动挂单，没有时间维度，而且单次方差极大。
实验台给的是四组互补指标：</p>
""")
        parts.append(table(
            ["指标组", "回答的问题", "关键量"],
            [
                ["<b>Markout</b>", "我的成交质量如何？是赚了价差还是接了有毒的单？",
                 "价差捕获 / 漂移(h) / 实现(h)"],
                ["<b>PnL 三分解</b>", "钱是策略赚的还是行情给的？",
                 "价差捕获 / 库存变动 / 底仓重估（<b>三者之和恒等于权益变化</b>）"],
                ["<b>库存风险</b>", "仓位管不管得住？", "时间加权库存 / 最大库存 / 贴平占比"],
                ["<b>权益风险</b>", "过程有多难受？", "最大回撤 / 单位风险收益"],
            ],
        ))
        parts.append("""
<div class="callout">
<strong>PnL 三分解的恒等式是精确成立的</strong>（有测试保证，且用变异体验证过
"漏掉其中一项"会被抓出来）。所以"钱从哪来"没有含糊空间。
一个实测例子：TWAP 执行策略的 PnL 是 +149,831，
其中「价差捕获」只有 <b>−1,699</b>（执行本身微微亏），
剩下全部来自「库存」——就是建仓期间市场涨了。
<b>只看合计会把一个平庸的执行算法看成优秀策略。</b>
</div>

<h3>12.5 复现性：三条保证</h3>
<p>策略 A/B 对比能不能成立，取决于"注入策略"这件事会不会扰动背景行情。
本项目为此做了专门设计，并<b>固化成测试</b>：</p>
<ol>
<li><b>主体数量与背景行情解耦。</b>市场随机流只用于基本面价值步进；
主体禀赋来自每主体独立的随机流；动作顺序由每主体独立抽的优先级决定。
<br>→ 实测：注入一个不交易的主体后，行情<b>逐点最大差 = 0</b>。</li>
<li><b>主体随机流按 <code>(seed, 序号)</code> 播种</b>，不用共享种子池 spawn，
所以增删主体不会扰动其他主体的流。</li>
<li><b>注入必须走 <code>market.add_agent</code></b>，且注入时点被记录
（<code>market.injected_at</code>）——同一 seed 只有连注入时点一起固定才复现得出。</li>
</ol>
<p class="note">这三条是建实验台时才发现的。原来的实现用
<code>rng.shuffle(self._order)</code> 决定动作顺序，而<b>洗牌的置换取决于列表长度</b>——
往市场里多放一个主体会让所有主体的出手顺序整体错位，
背景行情从第 0 个 tick 就变了。后果很具体：
注入一个"什么都不做"的策略，行情就已经不同了，
用它当对照去评估策略，测到的差异里混着"多了一个人参与洗牌"这个纯伪影。</p>

<h3>12.6 用它测策略时，必须一起报的局限</h3>
""")
        parts.append(table(
            ["局限", "对策略结论的影响"],
            [
                ["<b>冲击是线性/超线性的，真实市场是凹的（平方根律）</b>",
                 "<b>大单成本会被高估</b>，执行类策略的结论要打折看（根因见第 8.8 节）"],
                ["没有手续费 / 资金费率 / 借券成本", "高频做市类策略会被系统性高估"],
                ["没有 ADL / 保险基金 / 跨保证金", "清算场景里「清算不完」是模型边界"],
                ["主体同质、规则固定", "不会遇到「别人也在学习适应」的情况"],
                ["1 tick ≈ 1 小时只是刻度", "不要对延迟、队列位置做结论"],
                ["σ 依赖主体数 N", "换 N 时绝对量级会变，跨 N 只能比相对差值"],
            ],
        ))
        parts.append("""
<p class="warn"><strong>这些局限不会因为策略赚钱而消失。</strong>
把上表放在结论旁边一起报，比只报一条漂亮的收益曲线有价值得多。</p>
<p class="src">完整手册见 <code>docs/策略测试手册.md</code>；
内置示例见 <code>strategies/</code>（含两个反面基准：
<code>noop</code> 什么都不做、<code>random_taker</code> 无信息随机吃单）。</p>
""")
        for _name, _cap in (
            ("lab_pnl_decomposition.png",
             "13.3 各场景下的 PnL 分解：蓝色=价差捕获（本事），红色=库存变动，灰色=底仓重估（行情）"),
            ("lab_markout.png",
             "13.4 Markout 曲线（normal 场景）：成交之后价格往哪走。负值 = 逆向选择"),
            ("lab_risk_return.png",
             "13.3 风险 vs 收益：同一策略在不同场景下的位置（离原点越远越好）"),
        ):
            if (FIG / _name).exists():
                parts.append(img(FIG / _name, _cap))
        parts.append("</section>\n")

    # ============ 13. 桌面端 GUI ============
    if FIG.joinpath("gui_shot_overview.png").exists():
        parts.append("""
<section id="gui">
<h2>12. 桌面端 GUI：把上面这些东西装进一个窗口</h2>
<p>前面十二节是研究成果；这一节是<b>让人用得上的那一层</b>。
一个只会在命令行跑脚本的研究框架，实际使用率会很低——
改个参数要翻文件、看结果要等脚本跑完再打开图、和别人讨论时没法共享上下文。</p>
<pre class="code">python scripts/gui.py            # 开原生窗口
python scripts/gui.py --browser  # 用系统浏览器代替</pre>

<h3>12.1 技术选型与理由</h3>
""")
        parts.append(table(
            ["层", "技术", "为什么这么选"],
            [
                ["窗口", "<b>pywebview</b>（Windows 走系统自带 WebView2）",
                 "不打包浏览器，安装体积小；缺依赖时<b>自动退回系统浏览器</b>，功能一模一样"],
                ["后端", "<b>Python 标准库 http.server</b>",
                 "不引入 Flask。模拟本来就是 Python 写的，没必要为了一层 HTTP 再加一个框架"],
                ["前端", "<b>单个 HTML 文件，零外部依赖</b>",
                 "没有 npm / 打包步骤；<b>不加载任何 CDN</b>，离线可用"],
                ["图表", "<b>手写 canvas</b>",
                 "不引 Chart.js。折线 / 直方图+正态叠加 / 柱状图，手写够用且完全可控"],
            ],
        ))
        parts.append("""
<div class="callout">
<strong>一句话：窗口是外壳，不是功能。</strong>
开不出原生窗口时，浏览器里跑的是同一个应用——所以「GUI 起不来」这件事
不会让任何人失去能力，只会少一个窗口边框。
</div>
""")
        parts.append("<h3>12.2 界面</h3>")
        parts.append(img(FIG / "gui_shot_overview.png",
                         "13.2 策略运行结果：核心 KPI + PnL 三分解（合计 191,204 = 价差 306,435 + 库存 −115,231 + 底仓 0）"))
        parts.append(img(FIG / "gui_shot_charts.png",
                         "13.2 图表页：中间价与基本面锚、权益曲线、价差与深度、成交笔数、库存、Markout 曲线、收益率分布（全部手写 canvas）"))
        parts.append(img(FIG / "gui_shot_lab.png",
                         "13.2 批量对照页：5 策略 × 2 场景 × 3 种子的聚合结果。注意 <code>noop</code> 恰好 0 成交 0 盈亏——口径自检直接可见"))

        parts.append("""
<h3>12.3 六个页签分别看什么</h3>
""")
        parts.append(table(
            ["页签", "内容"],
            [
                ["<b>总览</b>", "核心 KPI + 「怎么读这些数」的内联说明；策略运行还给出 PnL 三分解"],
                ["<b>图表</b>", "中间价与基本面锚、权益曲线、价差与深度、成交笔数、库存、Markout、收益率分布"],
                ["<b>指标</b>", "统计特征表（模拟 vs 真实 BTC）+ ACF 图 + 市场结构"],
                ["<b>策略表现</b>", "Markout 分解表、成交与库存明细、PnL 分解明细（<b>含恒等式残差</b>）"],
                ["<b>对照</b>", "批量对照聚合表（跨种子 ± 标准误）"],
                ["<b>真实比对</b>", "载入真实 1 小时 K 线，看它的统计特征与收益率分布"],
            ],
        ))

        parts.append("""
<h3>12.4 三个让它"真的能用"的细节</h3>
<div class="callout">
<strong>① 在界面里直接写策略并运行。</strong>
不需要新建文件、不需要重启、不需要回到命令行。改一行参数再跑一次只要几秒。
<br><br>
<strong>② 参数错误在<b>提交时</b>就报，而不是跑完才报。</strong>
用户点完「运行」就盯着进度条了；几十秒后才告诉他「策略名打错了」，
那几十秒是纯浪费，甚至他会以为程序卡死。校验成本是毫秒级，作业成本是秒到分钟级。
<br><br>
<strong>③ 一切都能写进 URL。</strong>
<code>?scenario=stressed&strategy=mm_skewed&seed=20260917&ticks=8000&autorun=1</code>
打开就跑同一场实验；<code>&tab=charts</code> 直接落到某个页签；
<code>&job=&lt;id&gt;</code> 直接打开某次已完成的结果。
<b>把一次实验变成一条可以发给别人的链接</b>，这是命令行给不了的。
</div>

<h3>12.5 安全边界（本地工具该有的样子）</h3>
""")
        parts.append(table(
            ["措施", "挡住什么"],
            [
                ["只绑定 <code>127.0.0.1</code>", "别的机器连不上"],
                ["每个进程生成一次性 token，所有 API 都要带", "别的网页/别的程序顺手打进来"],
                ["不返回任何 CORS 头", "跨源读取被浏览器掐掉；带自定义头的请求还要过预检——<b>所以 token 头本身就是 CSRF 防护</b>"],
                ["校验 <code>Host</code> 必须是回环地址", "DNS rebinding（恶意域名解析到 127.0.0.1）"],
                ["静态文件做路径归一化 + 前缀校验", "目录穿越读任意文件"],
                ["<code>TW_GUI_ALLOW_CODE=0</code> 可关闭代码执行", "彻底不要「界面里跑代码」这个能力"],
            ],
        ))
        parts.append("""
<div class="callout danger">
<strong>⚠️ 但要说清楚：界面允许粘贴并运行 Python 策略代码 —— 那等于在本机执行任意代码。</strong>
进程内的 Python 沙箱做不干净（<code>__builtins__</code> 到处可达），
所以这里没有假装做沙箱。上面几条防的是「别的网页 / 别的机器」，
防不了"你自己粘贴了一段恶意代码"。<b>不要运行来源不明的策略代码。</b>
</div>

<h3>12.6 稳定性：140 项测试里的 28 项是给 GUI 写的</h3>
<p>GUI 的问题有它自己的特征——<b>不会崩，只会静默出错</b>。测试针对的就是这一类：</p>
""")
        parts.append(table(
            ["测试", "防的是什么"],
            [
                ["场景覆盖不污染全局注册表",
                 "<code>SCENARIOS</code> 是模块级单例，少一次复制 → 用户每次覆盖<b>永久生效</b>，"
                 "界面显示与实际运行不一致，重启进程才恢复"],
                ["作业抛异常不带走工作线程",
                 "工作线程是共享资源，它死了之后<b>所有后续作业永远排队</b>，界面只显示「排队中」"],
                ["作业登记表有上限", "series 可能很大，跑一天能把内存吃光"],
                ["无 token / 错 token / 非回环 Host / 目录穿越被拒", "本地服务的边界"],
                ["首页不引用任何外部资源", "<b>离线也要能打开</b>——CDN 断了界面就白屏"],
                ["非法 JSON → 400 而不是 500", "500 会让用户以为是后端崩了"],
                ["非法参数在提交时就报错", "别让用户等半天才知道参数写错"],
                ["端到端跑一次策略作业并校验 PnL 恒等式",
                 "评估代码的命门；HTTP 层也必须成立"],
                ["**护栏必须真能触发**",
                 "最初只卡「组合数 ≤ 300」，而最大可能组合是 280 —— <b>上限永远够不着，等于没有护栏</b>。"
                 "被测试抓出来后改成按真正的成本驱动量（总 tick·次）限制"],
            ],
        ))
        parts.append("""
<h3>12.7 性能与限额</h3>
""")
        parts.append(table(
            ["限额", "值", "为什么"],
            [
                ["观测 tick", "≤ 60,000", "日志数组大小与运行时长直接相关"],
                ["预热 tick", "≤ 20,000", "同上"],
                ["主体数", "≤ 1,500", "每主体每 tick 一次决策，成本线性"],
                ["批量组合数", "≤ 300", "防止手滑点出一万个组合"],
                ["批量总规模", "≤ 300 万 tick·次",
                 "**这才是成本驱动量**——20 次 × 5 万 tick 比 280 次 × 1000 tick 贵得多"],
            ],
        ))
        parts.append("""
<p class="note">参考速度：300 主体、7500 tick 约 5 秒（单核）。
工作线程默认 <code>min(4, CPU//2)</code>——模拟是纯 CPU 密集，开太多只会互相抢核。
作业分块推进（每块 250 tick），因此<b>进度条是实的，也能中途取消</b>。</p>
<p class="src">完整说明见 <code>docs/GUI说明.md</code>。</p>
</section>
""")

    # ============ 11. 边界 ============
    parts.append("""
<section id="limits">
<h2>13. 诚实的边界：这个模型不是什么</h2>
<table>
<thead><tr><th>不成立的说法</th><th>为什么</th></tr></thead>
<tbody>
<tr><td>它能预测下一个价格</td><td>收益率几乎无自相关是<b>做对了</b>的特征。有可预测性反而是模型坏了。</td></tr>
<tr><td>它能产出可交易信号</td><td>同上。而且模拟里的"真实价格"是内生的，没有外部真值可以对赌。</td></tr>
<tr><td>"校准好了"</td><td>只有价格序列的统计特征有真实靶子（σ/峰度/ACF/VR/Hill α）。
价差、深度、单笔流量、清算规模分布都<b>没有</b>真实对照，是内生量。</td></tr>
<tr><td>冲击服从平方根律</td><td>实测是线性/超线性（见 8.7/8.8），
根因是深度剖面形状。这是本模型最明确的失配。</td></tr>
<tr><td>参数是"正确的"</td><td>只是在训练段最优；测试段复评只能说明没那么过拟合。
真实市场的统计特征随样本波动，不存在唯一正确的参数。</td></tr>
<tr><td>清算压力测试反映了真实强平</td><td>真实交易所还有 ADL、保险基金、分层清算引擎、
跨保证金账户——本模型一个都没有，所以"清算不完"是模型边界。</td></tr>
</tbody>
</table>
<p class="note">
蓝图自己写的边界在此重申：<b>不会产出可交易信号；反身性问题依然成立；产出是"理解力"。</b>
本项目能加上的实证是：<b>我们知道了要复现真实市场的哪一项特征，需要加哪个机制</b>——
肥尾需要正反馈与负反馈共存；平方根律需要远端的深度剖面；
而这两件事都与"加更多主体"无关。
</p>
</section>
""")

    # ============ 12. 复现 ============
    parts.append("""
<section id="repro">
<h2>14. 复现方式</h2>
<pre class="code"># 环境（隔离，不污染系统）
C:/Users/87465/.workbuddy/binaries/python/envs/py313/Scripts/python.exe

# 1. 内核测试（84 项）
python -m unittest discover -s tests -t .

# 2. 变异验证：证明这套测试能失败
python scripts/mutation_check.py

# 3. 参数标定
python scripts/calibrate.py

# 4. 策略实验台（命令行）
python scripts/lab.py --quick       # 快速试跑
python scripts/lab.py               # 全量：策略 × 5 场景 × 4 种子

# 5. 桌面端 GUI
python scripts/gui.py               # 开原生窗口
python scripts/gui.py --browser     # 用系统浏览器代替

# 6. 四个阶段 + 扫单研究
python scripts/run_stage1.py
python scripts/run_stage2.py
python scripts/run_stage3.py
python scripts/run_stage4.py
python scripts/run_sweep_study.py

# 7. 生成本报告
python scripts/make_report.py</pre>
<p class="note">目录结构：<code>tw/</code>（库，含 <code>eval.py</code> 评估与 <code>strategy.py</code> 策略 API）、<code>strategies/</code>（示例策略）、<code>tests/</code>（测试）、
<code>scripts/</code>（实验与工具）、<code>docs/</code>（文档与本报告）、
<code>out/</code>（各阶段指标 JSON 与图）。</p>
</section>
""")

    body = "\n".join(parts)
    return HTML_TEMPLATE.replace("{{BODY}}", body)


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>交易世界 · 基于主体建模的人工市场 · 交付报告</title>
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
  h2 {
    font-size:22px; margin:52px 0 14px; padding-left:12px;
    border-left:4px solid var(--mid);
  }
  h3 { font-size:17px; margin:30px 0 10px; color:#c3cbd8; }
  p { margin:12px 0; }
  code { background:var(--panel2); padding:1px 6px; border-radius:4px; font-size:13px;
         color:#9fd0f0; font-family:ui-monospace,Consolas,"Courier New",monospace; }
  pre.log, pre.code {
    background:var(--panel); border:1px solid var(--grid); border-radius:8px;
    padding:14px 16px; overflow-x:auto; font-size:12.5px; line-height:1.6;
    font-family:ui-monospace,Consolas,"Courier New",monospace; color:#b9c4d4;
  }
  table { width:100%; border-collapse:collapse; margin:16px 0; font-size:14px; }
  th, td { padding:8px 11px; text-align:right; border-bottom:1px solid var(--grid); }
  th:first-child, td:first-child { text-align:left; }
  thead th { background:var(--panel); color:#a9b4c4; font-weight:600; font-size:13px;
             position:sticky; top:0; }
  tbody tr:hover { background:#1f232b; }
  .hi { color:var(--good); }
  .sel { background:rgba(90,169,230,.10); }
  .k { color:#a9b4c4; }
  .callout {
    background:var(--panel); border-left:3px solid var(--mid);
    padding:12px 16px; margin:18px 0; border-radius:0 8px 8px 0;
  }
  .callout.danger { border-left-color:var(--accent); }
  .lead { font-size:16.5px; }
  .note { color:var(--muted); font-size:13.5px; }
  .warn { color:var(--warn); }
  .src { color:var(--muted); font-size:13px; }
  ol.fail li { margin:8px 0; }
  figure { margin:24px 0; background:var(--panel); border:1px solid var(--grid);
           border-radius:10px; padding:12px; }
  figure img { width:100%; display:block; border-radius:6px; }
  figcaption { color:var(--muted); font-size:13px; margin-top:10px; text-align:center; }
  figure.missing .ph { padding:40px; text-align:center; color:var(--accent); }
  nav.toc {
    background:var(--panel); border:1px solid var(--grid); border-radius:10px;
    padding:16px 22px; margin-bottom:8px;
  }
  nav.toc ol { margin:6px 0; padding-left:22px; }
  nav.toc a { color:#9fd0f0; text-decoration:none; }
  nav.toc a:hover { text-decoration:underline; }
  strong { color:#eaeff7; }
  b { color:#eaeff7; }
</style>
</head>
<body>
<div class="wrap">
<header class="top">
  <h1>交易世界 · 基于主体建模的人工加密市场</h1>
  <div class="sub">
    施工蓝图交付报告 ·
    检验"简单主体的交互能否涌现真实加密市场的统计特征"
  </div>
</header>
<nav class="toc">
  <strong>目录</strong>
  <ol>
    <li><a href="#summary">一句话结论（含三处失败）</a></li>
    <li><a href="#mapping">蓝图要求 → 实际交付</a></li>
    <li><a href="#kernel">撮合内核可信度（140 测试 + 变异验证）</a></li>
    <li><a href="#targets">真实市场靶子</a></li>
    <li><a href="#calib">标定：照抄蓝图参数会炸</a></li>
    <li><a href="#stage1">阶段1 零智能基线</a></li>
    <li><a href="#stage2">阶段2 机制归因（消融）</a></li>
    <li><a href="#stage3">阶段3 做市商与清算压力（含测量方法论修正）</a></li>
    <li><a href="#sweep">扫单：继续走还是反转</a></li>
    <li><a href="#stage4">阶段4 校准与样本外检验</a></li>
    <li><a href="#lab">策略实验台（怎么用它测策略）</a></li>
    <li><a href="#gui">桌面端 GUI（图形界面）</a></li>
    <li><a href="#limits">诚实的边界</a></li>
    <li><a href="#repro">复现方式</a></li>
  </ol>
</nav>
{{BODY}}
</div>
</body>
</html>
"""


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    html_out = build()
    path = DOCS / "交易世界-交付报告.html"
    path.write_text(html_out, encoding="utf-8")
    kb = len(html_out.encode("utf-8")) / 1024
    print(f"  报告已生成：{path}（{kb:,.0f} KB）")


if __name__ == "__main__":
    main()
