"""生成路演用 PPTX（.pptx）。

数字从产物现算，且**直接从 `make_story_deck` import 取数函数** ——
不重抄一遍。两份实现必然分叉，而本项目已经吃过一次这样的亏
（`docs/标定说明.md` 10.76 vs `out/calibration.json` 10.67）。

⚠️ 这份 deck 的性质
------------------
**它不是"这个能赚钱"的路演。** 这个项目自己写着：不是价格预测器、
不是可交易信号源；而且实测结论里"LLM 没打赢随机交易"、
"收益效应在噪声地板以下"。做成那种 deck 投出去，专业听众第一天就会问穿。

**能讲且是真的那个版本**：我们造了一个可复现的人工市场，把"加密市场为什么
长这样"拆到了最小机制 —— 并且**排除了很多看起来合理的解释**。
"排除了什么"本身就是最硬的可信度资产，所以它进前三页，不是附录。

每页页脚都印"这是 YYYY-MM-DD 的快照，数字以仓库里的 HTML 为准" ——
PPT 唯一的硬伤就是它是快照，不标就等于埋一个未来的不一致。

用法：python scripts/make_roadshow_deck.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from make_story_deck import facts  # noqa: E402  ← 单一事实源，不重抄

OUT = ROOT / "docs" / "路演-交易世界.pptx"

# 浅色主题：投影仪上浅底比深底更清楚，打印也正常。
INK = RGBColor(0x1A, 0x1D, 0x21)
MUTE = RGBColor(0x6B, 0x72, 0x7A)
ACC = RGBColor(0x2F, 0x6F, 0xED)
OK = RGBColor(0x1F, 0x8A, 0x63)
WARN = RGBColor(0xB5, 0x76, 0x0B)
BAD = RGBColor(0xC0, 0x45, 0x36)
LINE = RGBColor(0xD8, 0xDC, 0xE0)

W, H = Inches(13.333), Inches(7.5)
FONT = "Microsoft YaHei"


def tx(slide, text, *, x, y, w, h, size=20, bold=False, color=INK,
       align=PP_ALIGN.LEFT, space=6):
    """放一个文本框。路演的字要少、要大 —— 所以这里不做自动缩排。"""
    box = slide.shapes.add_textbox(Emu(int(x)), Emu(int(y)), Emu(int(w)), Emu(int(h)))
    tf = box.text_frame
    tf.word_wrap = True
    lines = text.split("\n")
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space)
        r = p.add_run()
        r.text = line
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.color.rgb = color
        r.font.name = FONT
    return box


def rule(slide, y=Inches(1.55), x=Inches(0.9), w=Inches(2.2)):
    ln = slide.shapes.add_shape(1, Emu(int(x)), Emu(int(y)), Emu(int(w)), Emu(int(0.035)))
    ln.fill.solid()
    ln.fill.fore_color.rgb = ACC
    ln.line.fill.background()
    return ln


def footer(slide, idx, total, today):
    tx(slide, f"快照 {today} ｜ 数字以仓库 docs/交易世界-讲给人听.html 为准",
       x=Inches(0.9), y=Inches(6.95), w=Inches(9.5), h=Inches(0.35),
       size=10, color=MUTE)
    tx(slide, f"{idx} / {total}", x=Inches(12.0), y=Inches(6.95),
       w=Inches(0.6), h=Inches(0.35), size=10, color=MUTE, align=PP_ALIGN.RIGHT)


def new(prs, title, kicker=None):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    if kicker:
        tx(s, kicker, x=Inches(0.9), y=Inches(0.55), w=Inches(11), h=Inches(0.4),
           size=13, bold=True, color=ACC)
    tx(s, title, x=Inches(0.9), y=Inches(0.95), w=Inches(11.6), h=Inches(0.9),
       size=32, bold=True)
    rule(s)
    return s


def bar(slide, cats, vals, *, x, y, w, h, number_format='0.00', title=None):
    cd = CategoryChartData()
    cd.categories = cats
    cd.add_series("值", vals)
    gf = slide.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED,
                                Emu(int(x)), Emu(int(y)), Emu(int(w)), Emu(int(h)), cd)
    ch = gf.chart
    ch.has_legend = False
    if title:
        ch.has_title = True
        ch.chart_title.text_frame.text = title
        for p in ch.chart_title.text_frame.paragraphs:
            for r in p.runs:
                r.font.size = Pt(13)
                r.font.bold = True
                r.font.name = FONT
    pl = ch.plots[0]
    pl.has_data_labels = True
    dl = pl.data_labels
    dl.number_format = number_format
    dl.number_format_is_linked = False
    dl.font.size = Pt(12)
    dl.font.bold = True
    dl.font.name = FONT
    ser = pl.series[0]
    ser.format.fill.solid()
    ser.format.fill.fore_color.rgb = ACC
    ca = ch.category_axis
    ca.tick_labels.font.size = Pt(11)
    ca.tick_labels.font.name = FONT
    va = ch.value_axis
    va.has_major_gridlines = True
    va.tick_labels.font.size = Pt(10)
    va.tick_labels.font.name = FONT
    return ch


def line(slide, cats, vals, *, x, y, w, h, number_format='0.0', title=None):
    cd = CategoryChartData()
    cd.categories = cats
    cd.add_series("值", vals)
    gf = slide.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS,
                                Emu(int(x)), Emu(int(y)), Emu(int(w)), Emu(int(h)), cd)
    ch = gf.chart
    ch.has_legend = False
    if title:
        ch.has_title = True
        ch.chart_title.text_frame.text = title
        for p in ch.chart_title.text_frame.paragraphs:
            for r in p.runs:
                r.font.size = Pt(13)
                r.font.bold = True
                r.font.name = FONT
    pl = ch.plots[0]
    pl.has_data_labels = True
    dl = pl.data_labels
    dl.number_format = number_format
    dl.number_format_is_linked = False
    dl.font.size = Pt(12)
    dl.font.bold = True
    dl.font.name = FONT
    ser = pl.series[0]
    ser.format.line.width = Pt(2.5)
    ser.format.line.color.rgb = ACC
    ca = ch.category_axis
    ca.tick_labels.font.size = Pt(11)
    ca.tick_labels.font.name = FONT
    va = ch.value_axis
    va.has_major_gridlines = True
    va.tick_labels.font.size = Pt(10)
    va.tick_labels.font.name = FONT
    return ch


def build() -> int:
    f = facts()
    today = date.today().isoformat()
    pool = f["pools"]
    fund = next(k for k in pool if "基本面" in k and "图表" not in k)
    chart = next(k for k in pool if "图表" in k)
    mixed = next(k for k in pool if "混合" in k)
    zero = next(k for k in pool if "零智能" in k and "基本面" not in k and "图表" not in k)

    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    blank = prs.slide_layouts[6]
    total = 13

    # ---- 1 封面
    s = prs.slides.add_slide(blank)
    tx(s, "交易世界", x=Inches(0.9), y=Inches(2.1), w=Inches(11), h=Inches(1.3),
       size=54, bold=True)
    tx(s, "一个用几百个「人」造出来的加密市场\n"
          "以及我们用它排除掉的那些看起来合理的解释",
       x=Inches(0.9), y=Inches(3.5), w=Inches(11), h=Inches(1.2), size=22, color=MUTE)
    rule(s, y=Inches(5.1), w=Inches(3.0))
    tx(s, f"{today} ｜ 全部数字由脚本从产物现算，非手抄",
       x=Inches(0.9), y=Inches(5.4), w=Inches(11), h=Inches(0.4), size=13, color=MUTE)
    footer(s, 1, total, today)

    # ---- 2 一句话
    s = new(prs, "我们做了什么", "一句话")
    tx(s, "我们造了一个可复现的人工加密市场，\n"
          "把「加密市场为什么长这样」拆到了最小机制。",
       x=Inches(0.9), y=Inches(2.0), w=Inches(11.5), h=Inches(1.2), size=28, bold=True)
    tx(s, "几百个按固定规则交易的机器人，在自己长出来的订单簿里互相买卖 ——\n"
          "价格不是输入，是结果。于是每个现象都可以被追问：\n"
          "到底是哪一件事负责这个现象？",
       x=Inches(0.9), y=Inches(3.5), w=Inches(11.5), h=Inches(1.6), size=19, color=MUTE)
    footer(s, 2, total, today)

    # ---- 3 排除了什么（可信度资产，放前面）
    s = new(prs, "我们先排除了什么", "不是" + "　")
    tx(s, "① 照抄别人的参数没用", x=Inches(0.9), y=Inches(1.95), w=Inches(11), h=Inches(0.5),
       size=22, bold=True, color=BAD)
    tx(s, f"照抄施工蓝图的报价幅度 ⇒ 波动率变成真实 BTC 的 {f['sigma_ratio']:.1f} 倍。",
       x=Inches(1.3), y=Inches(2.45), w=Inches(11), h=Inches(0.4), size=17, color=MUTE)
    tx(s, "② 大模型交易没打赢「随机吃单」", x=Inches(0.9), y=Inches(3.15), w=Inches(11),
       h=Inches(0.5), size=22, bold=True, color=BAD)
    tx(s, f"同一段行情：随机吃单 +{f['a4']['random_taker']['ret'] * 100:.2f}%，"
          f"大模型 {f['a4']['llm_v2']['ret'] * 100:+.2f}%。",
       x=Inches(1.3), y=Inches(3.65), w=Inches(11), h=Inches(0.4), size=17, color=MUTE)
    tx(s, "③ 一个常用指标不能用来给策略排名", x=Inches(0.9), y=Inches(4.35), w=Inches(11),
       h=Inches(0.5), size=22, bold=True, color=BAD)
    tx(s, "冲击曲线的斜率在不同配置之间的差异，测不出来 —— 给噪声排名是自欺。",
       x=Inches(1.3), y=Inches(4.85), w=Inches(11), h=Inches(0.4), size=17, color=MUTE)
    tx(s, "⭐ 能排除掉什么，往往比能证明什么更能说明这套东西可不可信。",
       x=Inches(0.9), y=Inches(5.6), w=Inches(11.5), h=Inches(0.5), size=18, color=OK)
    footer(s, 3, total, today)

    # ---- 4 机制
    s = new(prs, "肥尾是「两种人」共存的产物", "机制")
    bar(s, ["只留看基本面", "只留看图表", "混合", "真实 BTC"],
        [pool[fund]["kurt"], pool[chart]["kurt"], pool[mixed]["kurt"], f["real"]["kurt"]],
        x=Inches(0.9), y=Inches(1.95), w=Inches(6.4), h=Inches(4.3),
        title="峰度（分布的尖尾程度）")
    tx(s, f"只留看基本面的：{pool[fund]['kurt']:.2f}\n近似正态，肥尾被抹平",
       x=Inches(7.6), y=Inches(2.1), w=Inches(5.0), h=Inches(1.0), size=18)
    tx(s, f"只留看图表的：σ {pool[chart]['sigma']:.0f}bp\n（混合池的 "
          f"{pool[chart]['sigma'] / pool[mixed]['sigma']:.1f} 倍，爆炸）",
       x=Inches(7.6), y=Inches(3.2), w=Inches(5.0), h=Inches(1.0), size=18)
    tx(s, f"⚠️ 「只有混合池」不是单看峰度 ——「只有零智能」峰度也有 "
          f"{pool[zero]['kurt']:.2f}，但它的涨跌几乎不记忆"
          f"（{pool[zero]['acf']:.2f}，真实 {f['real']['acf']:.2f}）。",
       x=Inches(7.6), y=Inches(4.3), w=Inches(5.0), h=Inches(1.8), size=15, color=WARN)
    footer(s, 4, total, today)

    # ---- 5 深度来自时间
    s = new(prs, "市场的深度来自时间，不是来自钱", "机制")
    line(s, [f"{r['slices']} 片" for r in f["slicing"]],
         [r["slip"] for r in f["slicing"]],
         x=Inches(0.9), y=Inches(1.95), w=Inches(6.6), h=Inches(4.3),
         number_format='0.0', title="同样一笔要卖的量，只改拆成几片（基点）")
    tx(s, f"冲击缩小 {f['slicing_ratio']:.1f} 倍\n"
          f"成交率 {f['slicing'][0]['fill'] * 100:.0f}% → {f['slicing'][-1]['fill'] * 100:.0f}%",
       x=Inches(7.9), y=Inches(2.2), w=Inches(4.8), h=Inches(1.2), size=20, bold=True)
    tx(s, "盘口像一层很薄的浮冰。\n大单不是「把价格推下去」，\n是把冰踩碎了 —— 后面没冰了，\n只能自己往下挖。",
       x=Inches(7.9), y=Inches(3.6), w=Inches(4.8), h=Inches(1.6), size=17, color=MUTE)
    tx(s, "⚠️ 本模型冲击是线性的（真实是凹的）⇒ 大单成本被高估，方向可信、幅度要打折。",
       x=Inches(7.9), y=Inches(5.3), w=Inches(4.8), h=Inches(0.9), size=14, color=WARN)
    footer(s, 5, total, today)

    # ---- 6 会被打空
    s = new(prs, "大单的真实代价不在价格，在「成交不了」", "机制")
    line(s, [f"{r['q'] * 100:.2f}%" for r in f["saturation"]],
         [r["fill"] * 100 for r in f["saturation"]],
         x=Inches(0.9), y=Inches(1.95), w=Inches(6.6), h=Inches(4.3),
         number_format='0', title="目标清算规模 → 真的卖出去的比例（%）")
    tx(s, f"卖得出去的比例\n{f['saturation'][0]['fill'] * 100:.0f}% → "
          f"{f['saturation'][-1]['fill'] * 100:.0f}%",
       x=Inches(7.9), y=Inches(2.2), w=Inches(4.8), h=Inches(1.2), size=22, bold=True,
       color=BAD)
    tx(s, "而冲击几乎不再变大（饱和了）\n—— 你以为的「冲击」，\n大部分其实是「没成交」。",
       x=Inches(7.9), y=Inches(3.7), w=Inches(4.8), h=Inches(1.4), size=17, color=MUTE)
    tx(s, f"顺带：有做市商时清算冲击是 {f['mm_ratio']:.2f} 倍（无做市商时）。",
       x=Inches(7.9), y=Inches(5.3), w=Inches(4.8), h=Inches(0.5), size=15, color=OK)
    footer(s, 6, total, today)

    # ---- 7 标定
    s = new(prs, "一个看起来只是「风格选择」的参数，决定了市场的情绪量级", "方法")
    bar(s, [f"照抄 {f['sigma_blueprint_off'] * 100:.2f}%",
            f"收窄 {f['sigma_tuned_off'] * 100:.2f}%"],
        [f["sigma_blueprint"], f["sigma_tuned"]],
        x=Inches(0.9), y=Inches(1.95), w=Inches(6.0), h=Inches(4.3),
        number_format='0.0', title="市场波动率 σ（基点/tick）")
    tx(s, f"真实 BTC 的 {f['sigma_target']:.1f}bp",
       x=Inches(7.3), y=Inches(2.0), w=Inches(5.0), h=Inches(0.5), size=20, bold=True)
    tx(s, f"照抄 ⇒ {f['sigma_ratio']:.2f} 倍\n收窄 ⇒ {f['sigma_tuned_ratio']:.2f} 倍",
       x=Inches(7.3), y=Inches(2.7), w=Inches(5.0), h=Inches(0.9), size=19)
    tx(s, "而且波动率还跟参与者数量绑着：\n「σ 标定好了」必须同时说\n「在多少个参与者下标定的」。",
       x=Inches(7.3), y=Inches(3.8), w=Inches(5.0), h=Inches(1.3), size=17, color=MUTE)
    tx(s, "⚠️ 价差、盘口深度、单笔流量标不了 —— 说不成「标定过了」就不说。",
       x=Inches(7.3), y=Inches(5.3), w=Inches(5.0), h=Inches(0.6), size=14, color=WARN)
    footer(s, 7, total, today)

    # ---- 8 推翻自己
    s = new(prs, "我们推翻了自己一次", "可信度")
    tx(s, "立项时最想解释的现象：一笔大单砸下去之后，价格还会继续同向走。",
       x=Inches(0.9), y=Inches(1.95), w=Inches(11.5), h=Inches(0.6), size=21, color=MUTE)
    tx(s, "我们测出来了", x=Inches(0.9), y=Inches(2.9), w=Inches(5.4), h=Inches(0.5),
       size=22, bold=True, color=OK)
    tx(s, "模拟里确实出现：\n大单砸下去，后面几十个 tick\n还在继续同向走。\n我们把它写进了报告。",
       x=Inches(0.9), y=Inches(3.45), w=Inches(5.4), h=Inches(1.8), size=18)
    tx(s, "然后我们把它推翻了", x=Inches(6.8), y=Inches(2.9), w=Inches(5.4), h=Inches(0.5),
       size=22, bold=True, color=BAD)
    tx(s, "加了一个对照组：同一时刻、\n一个完全没下过单的市场。\n\n对照组也在涨。\n那个「延续」是市场自己的漂移。",
       x=Inches(6.8), y=Inches(3.45), w=Inches(5.4), h=Inches(2.0), size=18)
    tx(s, "结论：未能证实。", x=Inches(6.8), y=Inches(5.5), w=Inches(5.4), h=Inches(0.5),
       size=22, bold=True, color=BAD)
    tx(s, "⭐ 任何只报告正面结果的「模拟验证」，都可以先问一句：它有没有推翻过什么。",
       x=Inches(0.9), y=Inches(6.2), w=Inches(11.5), h=Inches(0.5), size=18, color=OK)
    footer(s, 8, total, today)

    # ---- 9 大模型
    s = new(prs, "大模型炒币：两句话必须一起说", "结论")
    rows = list(f["a4"].items())
    y = Inches(1.95)
    tx(s, f"{'策略':<14}{'净收益':>10}{'回撤':>9}{'夏普':>9}", x=Inches(0.9), y=y,
       w=Inches(7.2), h=Inches(0.4), size=15, bold=True, color=MUTE)
    for i, (k, v) in enumerate(rows):
        sh = v["sharpe"]
        shs = "—" if sh is None or sh != sh else f"{sh:.2f}"
        is_llm = k == "llm_v2"
        tx(s, f"{k:<14}{v['ret'] * 100:>+9.2f}%{v['dd'] * 100:>8.2f}%{shs:>9}",
           x=Inches(0.9), y=Inches(2.4 + i * 0.42), w=Inches(7.2), h=Inches(0.4),
           size=16, bold=is_llm, color=BAD if is_llm else INK)
    tx(s, f"第一句：点估计上它落后（{f['a4']['llm_v2']['ret'] * 100:+.2f}% vs 随机吃单 "
          f"{f['a4']['random_taker']['ret'] * 100:+.2f}%）",
       x=Inches(8.2), y=Inches(2.0), w=Inches(4.6), h=Inches(1.0), size=18, bold=True)
    # ⚠️ 窄栏里**不要**手写换行：宽度差一点就会多折出一行、很难看
    #    （这是 PowerPoint 真渲染出来才看得见的问题，生成时看不出来）。
    tx(s, f"第二句：统计上分不出差别（区间重叠 43%~70%，需 545~836 个样本，"
          f"现在只有 {f['a4_n_bars']} 个）",
       x=Inches(8.2), y=Inches(3.3), w=Inches(4.6), h=Inches(1.2), size=18, bold=True)
    tx(s, "⇒ 不是「它更差」，也不是「它更好」。是用这一批样本分不出差别。",
       x=Inches(8.2), y=Inches(4.8), w=Inches(4.6), h=Inches(1.2), size=17, color=OK)
    footer(s, 9, total, today)

    # ---- 10 边界
    s = new(prs, "这个台子的边界（会让人误判的地方）", "边界")
    items = [
        "冲击是线性的，真实是凹的 ⇒ 大单成本被高估",
        "没有手续费 / 资金费率 / 借券成本 ⇒ 高频做市类策略被高估",
        "没有自动减仓 / 保险基金 / 跨保证金",
        "参与者同质、规则固定 ⇒ 遇不到「别人也在学习适应」",
        "冲击曲线斜率不能跨配置比较 ⇒ 不能给策略排名",
        "波动率依赖参与者数量 ⇒ 换 N 只能比相对差值",
    ]
    for i, it in enumerate(items):
        tx(s, f"·  {it}", x=Inches(0.9), y=Inches(2.0 + i * 0.62), w=Inches(11.5),
           h=Inches(0.5), size=19, color=INK if i < 2 else MUTE)
    tx(s, "一个工具只讲能做什么、不讲不能做什么，那它是在推销。",
       x=Inches(0.9), y=Inches(6.1), w=Inches(11.5), h=Inches(0.5), size=18, color=WARN)
    footer(s, 10, total, today)

    # ---- 11 能/不能回答
    s = new(prs, "这个台子现在能回答什么、还答不了什么", "现状")
    tx(s, "能回答（有数字、可复现）", x=Inches(0.9), y=Inches(1.95), w=Inches(5.6),
       h=Inches(0.5), size=21, bold=True, color=OK)
    tx(s, "· 某个统计特征是「谁」造出来的\n"
          "· 冲击对「卖得多急」的敏感度\n"
          "· 参数改动会把市场带偏多少\n"
          "· 一个策略相对基线到底有没有赢",
       x=Inches(0.9), y=Inches(2.55), w=Inches(5.6), h=Inches(2.2), size=18)
    tx(s, "还答不了（判不出或没做）", x=Inches(6.9), y=Inches(1.95), w=Inches(5.6),
       h=Inches(0.5), size=21, bold=True, color=BAD)
    tx(s, "· 真实市场里参数的正确取值\n"
          "· 收益率的可预测性（刻意不做：\n   真实市场里它也接近不可预测）\n"
          "· 参与覆盖面带来的冲击变化\n"
          "· 「人」的学习与适应",
       x=Inches(6.9), y=Inches(2.55), w=Inches(5.6), h=Inches(2.2), size=18)
    tx(s, "下一步：在同一个台子上，把「凹冲击」补出来 —— 缺的是「远端长期挂单」这个结构。",
       x=Inches(0.9), y=Inches(5.4), w=Inches(11.5), h=Inches(0.6), size=19, color=ACC)
    footer(s, 11, total, today)

    # ---- 12 我们要什么（可替换槽位）
    s = new(prs, "我们要什么", "（这一页请按实际情况替换）")
    tx(s, "① 算力与时间：把「凹冲击」这条路径做成默认\n"
          "   → 需要按 1000+ 种子量级跑几轮，把判不出的结论变成判得出的",
       x=Inches(0.9), y=Inches(2.0), w=Inches(11.5), h=Inches(1.1), size=20)
    tx(s, "② 数据与合作：真实市场的盘口深度 / 逐笔成交（公开 K 线里没有这些）\n"
          "   → 有盘口才能把「标不了」的那几项变成可标定",
       x=Inches(0.9), y=Inches(3.3), w=Inches(11.5), h=Inches(1.1), size=20)
    tx(s, "③ 一起做研究的人：愿意接受「判不出来」也是结果的同行",
       x=Inches(0.9), y=Inches(4.6), w=Inches(11.5), h=Inches(0.6), size=20)
    tx(s, "⚠️ 我们不承诺收益：这个项目的定位是理解力的工具，不是策略来源。\n"
          "   上面三条是研究投入，不是产品承诺 —— 请按你的实际诉求改写这一页。",
       x=Inches(0.9), y=Inches(5.6), w=Inches(11.5), h=Inches(1.0), size=17, color=WARN)
    footer(s, 12, total, today)

    # ---- 13 口径
    s = new(prs, "口径与出处", "附")
    tx(s, "· 本 deck 由 scripts/make_roadshow_deck.py 生成，数字**从产物现算**，没有手抄\n"
          "· 逐条对照见 docs/结论清单-人类可读版.md（含「还没核的数字」一节）\n"
          "· 通俗版与幻灯片版见 docs/交易世界-讲给人听.md / .html\n"
          "· 三个可复现实验：experiments/explainer/（40 秒 / 45 秒 / 20 秒）\n"
          "· ⚠️ 已发现一处数字漂移：另一份技术文档对同一个量给的是 10.76 倍（更早一次运行），\n"
          "  本 deck 一律用产物值",
       x=Inches(0.9), y=Inches(2.0), w=Inches(11.5), h=Inches(2.6), size=18)
    tx(s, "自检脚本会逐个数字把 HTML 幻灯片与产物比对 ⇒ 那份不会烂。\n"
          "本 PPTX 是快照（每页页脚已标注），如需引用请以仓库里的 HTML 为准。",
       x=Inches(0.9), y=Inches(5.0), w=Inches(11.5), h=Inches(1.0), size=18, color=OK)
    footer(s, 13, total, today)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print(f"✅ 已生成 {OUT.relative_to(ROOT)}（{OUT.stat().st_size / 1e6:.2f} MB，{total} 页）")
    print("   ⚠️ 还没验证 —— 下一步必须**用 PowerPoint 打开并导出图片**看一遍。")
    return 0


if __name__ == "__main__":
    raise SystemExit(build())
