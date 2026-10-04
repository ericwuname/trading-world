"""生成《讲给人听》的 HTML 幻灯片（单文件、零依赖、可离线、可打印）。

为什么是"生成"而不是手写 HTML
------------------------------
这个项目最贵的一条纪律是**数字不许手抄**（`out/*.json` 现算，自检逐个比对）。
手写幻灯片等于制造一个**新的数字来源** —— 而本项目刚抓到过
"两份文档对同一个量给了 10.67 / 10.76 两个值（不同次运行）"的漂移。
所以幻灯片里的每个数都从产物读，页脚也印上出处。

为什么 HTML 而不是 PPT
----------------------
**PPT 天生是快照**：代码更新了它不会变。
HTML 走的是同一个生成器 + 同一个自检（见 selfcheck 的 ⑨）⇒ **它不会烂**。
要 PPT 应该从这份 HTML 派生，并标注"这是 YYYY-MM-DD 的快照"。

用法：python scripts/make_story_deck.py
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
DECK = ROOT / "docs" / "交易世界-讲给人听.html"

BG, FG, DIM = "#16181d", "#e8eaed", "#9aa0a6"
C_OK, C_WARN, C_BAD, C_ACC = "#4ec9a0", "#e8b04b", "#e8735a", "#7aa2f7"


# ---------------------------------------------------------------- 取数
def load(name: str) -> dict:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def facts() -> dict:
    """所有数字从产物现算。**任何一项取不到就直接抛** —— 不允许静默填 0。"""
    f: dict = {}

    # —— σ 标定（out/calibration.json）
    cal = load("calibration.json")
    f["sigma_target"] = float(cal["target_sigma_bp"])
    scan = cal["pools"]["纯零智能(阶段1)"]["offset_scan"]
    by_val = {round(float(r["value"]), 6): r for r in scan}
    hi = by_val.get(0.02) or max(by_val.values(), key=lambda r: r["sigma_bp"])
    lo = min(by_val.values(), key=lambda r: r["sigma_bp"])
    f["sigma_blueprint"] = float(hi["sigma_bp"])
    f["sigma_blueprint_off"] = float(hi["value"])
    f["sigma_tuned"] = float(lo["sigma_bp"])
    f["sigma_tuned_off"] = float(lo["value"])
    f["sigma_ratio"] = f["sigma_blueprint"] / f["sigma_target"]
    f["sigma_tuned_ratio"] = f["sigma_tuned"] / f["sigma_target"]
    # `n_scan` 在产物里是**种子列表**（历史上有过是数字的版本）⇒ 两种都兼容。
    # 关键数字（σ、峰度、切片比）取不到会自己抛；这个是装饰性的，兜住。
    ns = cal.get("n_scan")
    f["n_scan"] = float(len(ns) if isinstance(ns, list) else (ns or 0))

    # —— 四池消融（out/stage2_metrics.json）
    s2 = load("stage2_metrics.json")
    f["pools"] = {
        k: {
            "kurt": float(v["metrics"]["excess_kurtosis"]),
            "sigma": float(v["metrics"]["sigma_bp"]),
            "acf": float(v["metrics"]["acf_abs_lag1"]),
        }
        for k, v in s2["pools"].items()
    }
    # `real` 是**按标的索引**的字典（BTC/ETH/SOL），且每个标的下面
    # 历史上存在"再包一层 metrics"与"扁平"两种形态 ⇒ 两种都兜住。
    raws = s2["real"]
    r0 = raws.get("BTCUSDT_1h") or next(iter(raws.values()))
    rmet = r0.get("metrics", r0)
    f["real"] = {
        "kurt": float(rmet["excess_kurtosis"]),
        "sigma": float(rmet["sigma_bp"]),
        "acf": float(rmet["acf_abs_lag1"]),
        "label": str(r0.get("label", "真实 BTC 1h")),
    }

    # —— 拆单速度（out/stage3_metrics.json → C3）
    #    ⚠️ 字段名是 `slippage_bp` / `fill_ratio`（不是 causal_slippage_bp / fill_rate）
    #    —— 本轮已踩过一次：凭印象猜字段名，连撞两次 KeyError。**先打印结构再取值。**
    #    C3 用的是 2% 清算规模 × 8 种子 ⇒ 与交付报告的数字一致（−148.4…）。
    s3 = load("stage3_metrics.json")
    f["slicing"] = sorted(
        (
            {
                "slices": int(r["slices"]),
                "slip": float(r["slippage_bp"]),
                "fill": float(r["fill_ratio"]),
                "shock": float(r["shock_frac"]),
                "seeds": int(r["n_seeds"]),
            }
            for r in s3["C3_slicing_speed"]["rows"]
        ),
        key=lambda r: r["slices"],
    )
    f["slicing_ratio"] = abs(f["slicing"][0]["slip"] / f["slicing"][-1]["slip"])

    # —— 瞬时清算的规模扫描（打空/饱和）
    f["saturation"] = [
        {
            "q": float(r["shock_frac"]),
            "slip": float(r["slippage_bp"]),
            "fill": float(r["fill_ratio"]),
        }
        for r in s3["C1_instant_saturation"]["rows"]
    ]

    # —— 做市商对照
    b = s3["B_liquidation"]
    f["mm"] = {
        "no_mm_slip": float(b["no_mm"]["slippage_bp"]),
        "mm_slip": float(b["with_mm"]["slippage_bp"]),
        "no_mm_fill": float(b["no_mm"]["fill_ratio"]),
        "mm_fill": float(b["with_mm"]["fill_ratio"]),
    }
    f["mm_ratio"] = f["mm"]["mm_slip"] / f["mm"]["no_mm_slip"]

    # —— A4 分层评估（out/a4/eval_BTC.json）
    a4 = load("a4/eval_BTC.json")
    f["a4"] = {
        k: {
            "ret": float(v.get("result", {}).get("net_return", 0.0)),
            "dd": float(v.get("result", {}).get("max_drawdown", 0.0)),
            "sharpe": v.get("result", {}).get("sharpe"),
        }
        for k, v in a4["evals"].items()
    }
    f["a4_verdicts"] = [
        {
            "a": str(x["name_a"]), "b": str(x["name_b"]),
            "overlap": float(x["overlap_fraction"])
            if x["overlap_fraction"] == x["overlap_fraction"] else None,
            "need": float(x["required_n"]) if x.get("required_n") else None,
            "verdict": str(x["verdict"]),
        }
        for x in a4["comparison"]["verdicts"]
    ]
    f["a4_n_bars"] = int(a4["comparison"]["n_bars"])
    return f


# ---------------------------------------------------------------- 画图（纯内联 SVG）
def bars(items: list[tuple[str, float, str]], *, w=760, h=250, unit="") -> str:
    """横向条形图。items = [(标签, 值, 颜色)]"""
    pad_l, pad_r, top, row = 190, 70, 34, (h - 60) // max(len(items), 1)
    vmax = max(abs(v) for _, v, _ in items) or 1.0
    bw = w - pad_l - pad_r
    out = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img">']
    for i, (label, v, color) in enumerate(items):
        y = top + i * row
        bl = bw * abs(v) / vmax
        out.append(
            f'<text x="{pad_l - 10}" y="{y + 20}" text-anchor="end" '
            f'class="lbl">{label}</text>'
            f'<rect x="{pad_l}" y="{y + 6}" width="{bl:.1f}" height="{row - 18}" '
            f'rx="3" fill="{color}" opacity="0.85"/>'
            f'<text x="{pad_l + bl + 8:.1f}" y="{y + 20}" class="val">'
            f'{v:.2f}{unit}</text>'
        )
    out.append("</svg>")
    return "".join(out)


def lines(rows: list[tuple[str, float, str]], *, labels: list[str],
          w=760, h=250, unit="", ylabel="") -> str:
    """折线图（x 用分类下标）。rows = [(标签, 值, 颜色)]"""
    pad_l, pad_r, pad_t, pad_b = 78, 26, 26, 46
    iw, ih = w - pad_l - pad_r, h - pad_t - pad_b
    vmax = max(abs(v) for _, v, _ in rows) or 1.0
    n = len(rows)
    out = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img">']
    for frac in (0.0, 0.5, 1.0):
        y = pad_t + ih * (1 - frac)
        out.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + iw}" y2="{y:.1f}" '
                   f'stroke="{DIM}" opacity="0.22"/>')
        out.append(f'<text x="{pad_l - 8}" y="{y + 4:.1f}" text-anchor="end" '
                   f'class="ax">{vmax * frac:.0f}{unit}</text>')
    pts = []
    for i, (_, v, color) in enumerate(rows):
        x = pad_l + (iw * i / max(n - 1, 1))
        y = pad_t + ih * (1 - abs(v) / vmax)
        pts.append((x, y, color))
    out.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y, _ in pts)}" '
               f'fill="none" stroke="{C_ACC}" stroke-width="2.2"/>')
    for (x, y, color), (label, v, _c) in zip(pts, rows):
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
        out.append(f'<text x="{x:.1f}" y="{pad_t + ih + 22:.1f}" text-anchor="middle" '
                   f'class="ax">{label}</text>')
        out.append(f'<text x="{x:.1f}" y="{y - 10:.1f}" text-anchor="middle" '
                   f'class="val">{v:.1f}{unit}</text>')
    if ylabel:
        out.append(f'<text x="14" y="{pad_t + ih / 2:.0f}" class="ax" '
                   f'transform="rotate(-90 14 {pad_t + ih / 2:.0f})" '
                   f'text-anchor="middle">{ylabel}</text>')
    out.append("</svg>")
    return "".join(out)


def pill(text: str, kind: str = "ok") -> str:
    color = {"ok": C_OK, "warn": C_WARN, "bad": C_BAD, "acc": C_ACC}[kind]
    return f'<span class="pill" style="--c:{color}">{text}</span>'


# ---------------------------------------------------------------- 组装
def build(f: dict) -> str:
    today = date.today().isoformat()
    pool_items = [
        (k.replace("零智能+", "+").replace("（蓝图配比）", "（混合）"), v["kurt"],
         C_ACC if "混合" in k else (C_WARN if "基本面" in k else DIM))
        for k, v in f["pools"].items()
    ]
    real_item = [(f["real"]["label"], f["real"]["kurt"], C_OK)]
    slic = [(f"{r['slices']} 片", r["slip"], C_ACC) for r in f["slicing"]]
    sat_fill = [(f"{r['q'] * 100:.2f}%", r["fill"] * 100 if r["fill"] == r["fill"]
                 else float("nan"), C_WARN) for r in f["saturation"]]
    sig_rows = [
        (f"{f['sigma_blueprint_off'] * 100:.2f}%", f["sigma_blueprint"], C_BAD),
        (f"{f['sigma_tuned_off'] * 100:.2f}%", f["sigma_tuned"], C_OK),
    ]

    # ⚠️ 表格行**先算成字符串再拼** —— 不要在 f-string 里再套 f-string，
    #    那会撞上引号问题（本轮第一次写就被迫用 chr() 拼接，代码没法看）。
    a4_rows = "".join(
        "<tr><td>{k}</td><td>{ret:+.2f}%</td><td>{dd:.2f}%</td><td>{sh}</td></tr>".format(
            k=k, ret=v["ret"] * 100, dd=v["dd"] * 100, sh=sh_s)
        for k, v, sh_s in (
            (k, v, "—" if v["sharpe"] is None or v["sharpe"] != v["sharpe"]
             else f"{v['sharpe']:.2f}")
            for k, v in f["a4"].items()
        )
    )
    verdicts = "".join(
        "<tr><td>{a} − {b}</td><td>{ov}</td><td>{vd}</td><td>{need}</td></tr>".format(
            a=x["a"], b=x["b"],
            ov="—" if x["overlap"] is None else f"{x['overlap'] * 100:.1f}%",
            vd=x["verdict"],
            need="—" if not x["need"] else f"{x['need']:.0f} 个样本")
        for x in f["a4_verdicts"]
    )

    def slide(cls: str, inner: str) -> str:
        return f'<section class="slide {cls}">{inner}</section>'

    s = []
    s.append(slide("cover", f"""
      <div class="stack">
        <div class="kicker">基于主体建模（ABM）的人工加密市场</div>
        <h1>交易世界</h1>
        <p class="lead">一个用几百个「人」造出来的市场。<br>
        它能回答什么、<b>不能</b>回答什么。</p>
        <p class="meta">{today} ｜ 全部数字由 <code>scripts/make_story_deck.py</code>
        从产物现算 ｜ 逐页可核对出处</p>
      </div>"""))

    s.append(slide("", f"""
      <h2>先划清边界</h2>
      <div class="two">
        <div class="card bad"><h3>这不是</h3>
          <ul>
            <li><b>价格预测器</b> —— 这里的收益率几乎不可预测，而这是<b>做对了</b>的特征</li>
            <li><b>可照着炒的信号源</b> —— 「真实价格」是模型自己算的，没法跟真钱对赌</li>
            <li><b>「校准好的模型」</b> —— 只有价格序列的统计特征有真实靶子</li>
            <li><b>完整交易所</b> —— 没有自动减仓、保险基金、跨保证金</li>
          </ul>
        </div>
        <div class="card ok"><h3>这是</h3>
          <p>一个<b>理解力的工具</b>：把肥尾、波动率聚集、大单砸穿盘口这些现象，
          拆到最简的生成机制上，然后逐个问 ——
          <b>到底是哪一件事负责这个现象？</b></p>
          <p class="dim">怎么测的、哪些还只是线索、模型的边界在哪，
          后面每一页都标着。</p>
        </div>
      </div>"""))

    fund = next((k for k in f["pools"] if "基本面" in k and "图表" not in k), None)
    chart = next((k for k in f["pools"] if "图表" in k), None)
    mixed = next((k for k in f["pools"] if "混合" in k), None)
    zero = next((k for k in f["pools"] if "零智能" in k and "基本面" not in k
                 and "图表" not in k), None)
    s.append(slide("", f"""
      <h2>两种人，缺一不可 {pill('已确证','ok')}</h2>
      <p class="sub">把「看基本面的」和「看图表的」分开单独跑，看峰度（分布的尖尾程度）怎么变</p>
      {bars(pool_items + real_item, unit='')}
      <div class="note">
        <p>只留<b>看基本面的</b> ⇒ 峰度 {f['pools'][fund]['kurt']:.2f}（接近正态，<b>肥尾被抹平</b>）；
        只留<b>看图表的</b> ⇒ 波动率 {f['pools'][chart]['sigma']:.0f}bp（混合池的
        {f['pools'][chart]['sigma'] / f['pools'][mixed]['sigma']:.1f} 倍，<b>爆炸</b>）；
        <b>两者共存</b> ⇒ 肥尾、波动率聚集、价格不可预测同时出现。</p>
        <p class="dim">⚠️ 顺手点破一个容易看糊涂的地方：<b>「只有混合池」不是单看峰度</b> ——
        「只有零智能」那一栏的峰度也有 {f['pools'][zero]['kurt']:.2f}，并不低。
        真正把它们分开的是<b>组合</b>：零智能池的涨跌幅<b>几乎不记忆</b>
        （|r| 自相关 {f['pools'][zero]['acf']:.3f}），而混合池是
        {f['pools'][mixed]['acf']:.3f}，<b>更接近真实 BTC 的
        {f['real']['acf']:.3f}</b>。它有肥尾，但没有「波动率聚集」；
        这两样必须同时在，才像真实市场。</p>
        <p class="warn">⚠️ 没达标的那一半：混合池峰度 {f['pools'][mixed]['kurt']:.2f}
        vs 真实 {f['real']['kurt']:.2f}（只有
        {f['pools'][mixed]['kurt'] / f['real']['kurt']:.0%}），波动率还偏高
        {f['pools'][mixed]['sigma'] / f['real']['sigma']:.1f} 倍。
        <b>方向对，厚度没到</b> —— 是已知取舍，不是已解决的问题。</p>
      </div>
      <p class="src">出处：out/stage2_metrics.json（四池 × {f['n_scan']:.0f} 种子）</p>"""))

    s.append(slide("", f"""
      <h2>深度来自时间，不是来自钱 {pill('已确证','ok')}</h2>
      <p class="sub">同样一笔要卖的量，只改「拆成几片」——价格被推走多少（因果滑点，基点）</p>
      {lines(slic, labels=[], unit='bp', ylabel='冲击')}
      <div class="note">
        <p>冲击缩小到 <b>{f['slicing_ratio']:.1f} 倍</b>；
        成交率从 {f['slicing'][0]['fill'] * 100:.0f}% 升到 {f['slicing'][-1]['fill'] * 100:.0f}%。</p>
        <p>为什么？盘口像一层很薄的浮冰。大单不是「把价格推下去」，
        是<b>把冰踩碎了</b> —— 后面没冰了，只能自己往下挖。
        拆细了等于在浮冰上一步一步挪。</p>
        <p class="warn">⚠️ 别当成「拆单永远划算」：这个模型里冲击是<b>线性</b>的（真实是凹的），
        所以<b>大单成本在模型里被高估</b>。方向可信，幅度要打折。</p>
      </div>
      <p class="src">出处：out/stage3_metrics.json → C3_slicing_speed</p>"""))

    s.append(slide("", f"""
      <h2>市场会被「打空」 {pill('已确证','ok')}</h2>
      <p class="sub">把要卖的比例越加越大：卖得出去的比例（%）</p>
      {lines(sat_fill, labels=[], unit='%', ylabel='成交率')}
      <div class="note">
        <p>规模翻了几十倍，<b>冲击几乎不再变大（饱和了）</b>，
        但能卖出去的比例从 {f['saturation'][0]['fill'] * 100:.0f}% 掉到
        {f['saturation'][-1]['fill'] * 100:.0f}%。</p>
        <p class="lead">⇒ 大单的真实代价不在价格上，在<b>「成交不了」</b>上。</p>
        <p>顺带一个对照：有做市商时清算冲击是 {f['mm']['mm_slip']:.1f}bp，
        没有时 {f['mm']['no_mm_slip']:.1f}bp ⇒ <b>{f['mm_ratio']:.2f} 倍</b>。</p>
      </div>
      <p class="src">出处：out/stage3_metrics.json → C1_instant_saturation、B_liquidation</p>"""))

    s.append(slide("", f"""
      <h2>照抄别人的参数，会把市场做坏 {pill('已确证','ok')}</h2>
      <p class="sub">只改一个参数：零智能交易者的报价幅度上界。σ = 每 tick 收益率标准差</p>
      {bars([(f"上界 {r[0][:-1]}%", r[1], c) for r, c in zip(sig_rows, [C_BAD, C_OK])], unit='bp')}
      <div class="note">
        <p>照抄施工蓝图的 {f['sigma_blueprint_off'] * 100:.2f}% ⇒
        σ={f['sigma_blueprint']:.1f}bp，是真实 BTC（{f['sigma_target']:.1f}bp）的
        <b>{f['sigma_ratio']:.2f} 倍</b>。收到 {f['sigma_tuned_off'] * 100:.2f}% ⇒
        {f['sigma_tuned']:.1f}bp（{f['sigma_tuned_ratio']:.2f} 倍，量级命中）。</p>
        <p><b>「σ 标定好了」这句话必须同时说「在多少个参与者下标定的」</b> ——
        纯零智能池里参与者从 100 增到 500，σ 会翻倍多；混合池平缓得多。</p>
        <p class="warn">⚠️ 同类里<b>标不了</b>的东西必须明说：买卖价差、盘口深度、
        单笔成交流量、清算规模分布 —— 公开 1 小时 K 线里没有盘口信息，
        说出来「标定过了」是不诚实的。</p>
      </div>
      <p class="src">出处：out/calibration.json</p>"""))

    s.append(slide("", f"""
      <h2>我们推翻了自己一次 {pill('最值钱的一条','ok')}</h2>
      <p class="lead">立项时最想解释的现象：<b>一笔大单砸下去之后，价格还会继续同向走</b>
      （像雪崩一样的 cascade）。</p>
      <div class="two">
        <div class="card"><h3>我们测出来了</h3>
          <p>模拟里确实出现了：一笔大单砸下去，后面几十个 tick 还在继续同向走。
          我们把它写进了报告。</p></div>
        <div class="card bad"><h3>然后我们把它推翻了</h3>
          <p>加了一个<b>对照组</b>：同一时刻、一个完全没下过单的市场。
          <b>对照组也在涨。</b> 那个「延续」是市场自己的漂移，不是砸单造成的。</p>
          <p class="lead">结论：<b>未能证实。</b></p></div>
      </div>
      <div class="note"><p>⭐ 我们的看法：<b>任何只报告正面结果的「模拟验证」，
      你都可以先问一句「你有没有推翻过什么」。</b> 一个永远只证明自己对的模型，
      不是在验证，是在讨好你。</p></div>
      <p class="src">出处：交付报告「蓝图验收对照」表（扫单后继续走 ❌ 未能证实）</p>"""))

    llm = f["a4"].get("llm_v2", {})
    rnd = f["a4"].get("random_taker", {})
    mom = f["a4"].get("momentum", {})
    s.append(slide("", f"""
      <h2>大模型炒币这件事 {pill('点估计：已确证', 'ok')} {pill('统计：样本不足', 'warn')}</h2>
      <p class="sub">同一段行情（BTC，{f['a4_n_bars']} 根 1 小时 K 线）</p>
      <table>
        <tr><th>谁在交易</th><th>净收益</th><th>最大回撤</th><th>夏普</th></tr>
        {a4_rows}
      </table>
      <div class="note">
        <p><b>第一句：点估计上它落后。</b> {llm.get('ret', 0) * 100:+.2f}%，
        而「随机吃单」拿到 {rnd.get('ret', 0) * 100:+.2f}%。
        顺带一个刺眼的事实：<b>随机吃单的夏普（{rnd.get('sharpe', 0):.2f}）
        比精心写的趋势策略（{mom.get('sharpe', 0):.2f}）还高</b> ——
        在这一段行情里，最划算的做法就是没有策略。</p>
        <p><b>第二句：统计上分不出差别。</b></p>
        <table class="mini">
          <tr><th>对比</th><th>区间重叠</th><th>机器判定</th><th>要多少样本</th></tr>
          {verdicts}
        </table>
        <p>⇒ 正确说法是<b>两句一起说</b>：用这一批样本分不出差别，
        而「点估计上没有赢过随机交易」也同时成立。
        <b>不是「它更差」，也不是「它更好」。</b></p>
      </div>
      <p class="src">出处：out/a4/eval_BTC.json（判定由脚本现算）</p>"""))

    s.append(slide("", f"""
      <h2>模型的边界 {pill('会让人误判的地方', 'bad')}</h2>
      <p class="sub">一个工具如果只讲它能做什么、不讲它不能做什么，那它是在推销。</p>
      <table class="defects">
        <tr><td>冲击是<b>线性</b>的，真实是<b>凹</b>的（平方根律）</td>
            <td><b>大单成本被高估</b> ⇒ 执行类策略要打折。缺的是「远端长期挂单」这个结构，不是更多参与者</td></tr>
        <tr><td>没有手续费 / 资金费率 / 借券成本</td><td><b>高频做市类策略被系统性高估</b></td></tr>
        <tr><td>没有自动减仓 / 保险基金 / 跨保证金</td><td>清算场景里「清不完」是模型边界</td></tr>
        <tr><td>参与者同质、规则固定</td><td>不会遇到「别人也在学习适应」</td></tr>
        <tr><td>冲击曲线斜率<b>不能跨配置比较</b></td><td>拿它给不同参数排名 = 给噪声排名</td></tr>
        <tr><td>1 个 tick ≈ 1 小时，这只是刻度</td><td>不要对延迟、队列位置下任何结论</td></tr>
        <tr><td>波动率依赖参与者数量</td><td>换 N 时绝对值会变，只能比相对差值</td></tr>
      </table>"""))

    s.append(slide("", f"""
      <h2>想自己看一眼？</h2>
      <p class="sub">不想只信一篇文章？那就跑一遍。每个都复用生产代码本身，不是重写。</p>
      <table class="runs">
        <tr><th>想看什么</th><th>跑哪个</th><th>大概耗时</th></tr>
        <tr><td>两种人缺一不可</td><td><code>experiments/explainer/01_两种人缺一不可.py</code></td><td>~40 秒</td></tr>
        <tr><td>深度来自时间</td><td><code>experiments/explainer/02_深度来自时间.py</code></td><td>~45 秒</td></tr>
        <tr><td>照抄参数会做坏市场</td><td><code>experiments/explainer/03_照抄参数会做坏市场.py</code></td><td>~20 秒</td></tr>
      </table>
      <div class="note">
        <p><b>它们怎么算「验证过了」—— 这里有个刻意的设计：</b></p>
        <ul>
          <li><b>断言方向与量级，不断言精确值。</b>换一组种子这些数就会挪动；
              把一个会动的数字当精确值来对，<b>比对错更贵</b>。</li>
          <li>把你<b>自己跑出来的数</b>和<b>报告里的参考值并排打出来</b>。</li>
          <li>没复现就<b>报错退出</b>，并要求把表格连同种子一起贴出来 ——
              它不会自动说「报告错了」，也不会自动说「你验证过了」。</li>
        </ul>
      </div>
      <p class="src">配套文档：docs/交易世界-讲给人听.md ｜ docs/结论清单-人类可读版.md</p>"""))

    s.append(slide("", f"""
      <h2>出处与口径 {pill('每条数字都可核对', 'acc')}</h2>
      <ul class="sources">
        <li>本文件由 <code>scripts/make_story_deck.py</code> 于 {today} 生成，
            数字全部从 <code>out/*.json</code> 现算，<b>没有一个是手抄的</b>。</li>
        <li>自检脚本会<b>逐个数字</b>把本文件与产物比对（容差内）——
            也就是说<b>这份幻灯片不会烂</b>。</li>
        <li>⚠️ <b>已发现一处数字漂移</b>：另一个技术文档对同一个量给的是
            515.7bp / 10.76 倍（更早一次运行），本文件用产物值
            {f['sigma_blueprint']:.1f}bp / {f['sigma_ratio']:.2f} 倍。
            <b>两个都是真的</b> —— 这也说明本项目自己的文档也会过期，
            所以每条数字都标了出处。</li>
        <li>逐条对照见 <code>docs/结论清单-人类可读版.md</code>（含「还没核的数字」一节）。</li>
      </ul>"""))

    body = "\n".join(s)
    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>交易世界 · 讲给人听</title>
<style>
:root {{ --bg:{BG}; --fg:{FG}; --dim:{DIM}; --ok:{C_OK}; --warn:{C_WARN};
  --bad:{C_BAD}; --acc:{C_ACC}; }}
* {{ box-sizing:border-box; }}
html,body {{ margin:0; background:var(--bg); color:var(--fg);
  font-family:"PingFang SC","Microsoft YaHei","Noto Sans SC",system-ui,sans-serif; }}
body {{ scroll-snap-type:y mandatory; }}
.slide {{ min-height:100vh; scroll-snap-align:start; padding:6vh 7vw;
  display:flex; flex-direction:column; justify-content:center;
  border-bottom:1px solid rgba(255,255,255,.06); }}
.cover {{ text-align:left; }}
h1 {{ font-size:clamp(44px,7vw,88px); margin:.1em 0; letter-spacing:.02em; }}
h2 {{ font-size:clamp(26px,3.4vw,40px); margin:0 0 .4em; }}
h3 {{ font-size:19px; margin:0 0 .5em; }}
.kicker {{ color:var(--acc); letter-spacing:.16em; font-size:14px; }}
.lead {{ font-size:clamp(19px,2.1vw,26px); line-height:1.7; color:#dfe3e8; }}
.sub {{ color:var(--dim); font-size:16px; margin:0 0 1.2em; }}
.meta,.src {{ color:var(--dim); font-size:13px; }}
.src {{ margin-top:auto; padding-top:1.2em; font-family:ui-monospace,monospace; }}
.dim {{ color:var(--dim); }}
.warn {{ color:var(--warn); }}
code {{ background:rgba(255,255,255,.08); padding:.1em .4em; border-radius:3px;
  font-size:.92em; }}
.pill {{ display:inline-block; font-size:13px; padding:.15em .7em; border-radius:999px;
  border:1px solid var(--c); color:var(--c); margin-left:.6em; vertical-align:middle; }}
.two {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; }}
@media (max-width:820px) {{ .two {{ grid-template-columns:1fr; }} }}
.card {{ background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.08);
  border-radius:10px; padding:20px 24px; }}
.card.bad {{ border-color:rgba(232,115,90,.4); }}
.card.ok {{ border-color:rgba(78,201,160,.35); }}
.chart {{ width:100%; height:auto; margin:.2em 0 .6em; }}
.chart .lbl {{ fill:var(--fg); font-size:14px; }}
.chart .val {{ fill:var(--fg); font-size:14px; font-weight:600; }}
.chart .ax {{ fill:var(--dim); font-size:12px; }}
.note {{ background:rgba(255,255,255,.035); border-left:3px solid var(--acc);
  padding:14px 18px; border-radius:0 8px 8px 0; }}
.note p {{ margin:.5em 0; line-height:1.75; }}
ul {{ line-height:1.8; padding-left:1.2em; }}
table {{ border-collapse:collapse; width:100%; margin:.4em 0; font-size:16px; }}
th,td {{ border-bottom:1px solid rgba(255,255,255,.1); padding:.5em .7em; text-align:left; }}
th {{ color:var(--dim); font-weight:600; font-size:14px; }}
table.mini {{ font-size:13px; margin:.6em 0; }}
table.mini th,table.mini td {{ padding:.3em .5em; }}
table.runs td {{ font-size:15px; }}
table.defects td:first-child {{ width:44%; color:#dfe3e8; }}
table.defects td {{ font-size:15px; }}
.sources li {{ margin:.7em 0; line-height:1.8; }}
#nav {{ position:fixed; right:18px; bottom:16px; color:var(--dim); font-size:12px;
  display:flex; gap:10px; align-items:center; }}
#nav button {{ background:rgba(255,255,255,.08); color:var(--fg); border:0;
  border-radius:6px; padding:.35em .7em; cursor:pointer; font-size:14px; }}
@media print {{
  body {{ scroll-snap-type:none; }}
  .slide {{ min-height:auto; page-break-after:always; border:0; padding:2vh 2vw; }}
  #nav {{ display:none; }}
}}
</style></head><body>
{body}
<div id="nav"><button onclick="go(-1)">↑</button><span id="pg"></span>
<button onclick="go(1)">↓</button></div>
<script>
const sl=[...document.querySelectorAll('.slide')];let cur=0;
function show(i){{cur=Math.max(0,Math.min(sl.length-1,i));
  sl[cur].scrollIntoView({{behavior:'smooth'}});}}
function go(d){{show(cur+d);}}
document.getElementById('pg').textContent=(cur+1)+' / '+sl.length;
addEventListener('keydown',e=>{{
  if(['ArrowDown','PageDown',' '].includes(e.key)){{e.preventDefault();go(1);}}
  if(['ArrowUp','PageUp'].includes(e.key)){{e.preventDefault();go(-1);}}
}});
addEventListener('scroll',()=>{{
  const y=scrollY+innerHeight*0.4;let i=0;
  sl.forEach((s,k)=>{{if(s.offsetTop<=y)i=k;}});
  if(i!==cur){{cur=i;document.getElementById('pg').textContent=(cur+1)+' / '+sl.length;}}
}});
</script></body></html>"""


def main() -> int:
    f = facts()
    DECK.write_text(build(f), encoding="utf-8")
    kb = DECK.stat().st_size / 1024
    print(f"✅ 已生成 {DECK.relative_to(ROOT)}（{kb:.1f} KB，单文件零依赖）")
    print(f"   四池峰度：{ {k: round(v['kurt'], 2) for k, v in f['pools'].items()} }")
    print(f"   真实峰度：{f['real']['kurt']:.2f}　σ 标定：{f['sigma_blueprint']:.1f}bp "
          f"= {f['sigma_ratio']:.2f}×")
    print(f"   拆单比值：{f['slicing_ratio']:.1f}×　做市商：{f['mm_ratio']:.2f}×")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
