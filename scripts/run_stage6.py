"""阶段6：长记忆订单流 LMF（二期指导书 §3）。

要修的核心缺陷
--------------
一期报告定位到但没修：**冲击函数不是凹的**。一期的冲击幂律指数
k 落在 0.61~1.25（主指标两项；"窗口均偏离"那一项在分期清算下到 3.09），
真实市场是 **k ≈ 0.5（平方根律）**。
不凹意味着"大单成本被线性放大"——用来评估执行类策略会系统性高估大单成本。

实验编号（对应指导书 §3.4）
--------------------------
E6.1 基线对照      同一批市场、机制全关 → 记录订单符号 ACF 与冲击指数
E6.2 元订单开启    40% 图表派套上 MetaOrderMixin → ACF 是否更长
E6.3 自适应流动性  再做市商套上 AdaptiveLiquidityMixin → 深度剖面 γ 是否上升
E6.4 联合冲击复测   用 E6.3 的配置重跑一期"分期清算规模梯度"实验 → k 是否接近 0.5
E6.5 参数敏感性    pareto_alpha × flow_sensitivity 各 3 档 → 哪个对 k 影响更大

⭐ 五条必须先说清楚的方法论
==========================

**① 记账粒度会伪造长记忆。**
   一个主动单扫多档 → 产生**多笔成交**，它们的 ``aggressor_side`` 相同。
   直接对逐笔序列算 ACF 会得到一个**纯机械的**正自相关。
   LMF 的 ε 是**每张订单一个符号**。实测一期基线的**逐笔** ACF(1) 是
   **+0.22**——看着像"记忆很长"，其实一大截是记账粒度的产物。
   本脚本两个口径**都报**，让读者看到这个陷阱有多大。

**② 基线不是"接近 0"。**
   指导书预期"基线几乎无长记忆"。实测不是：按订单口径，一期基线的
   ACF(1) 仍有约 +0.1 量级，且能拟合出幂律衰减。
   来源是真实的机制（基本面派盯着一个**持续的**估值偏离下单、
   图表派盯着**持续的**动量下单），不是 bug。
   所以 E6.2 的判据**不能**写成"ACF 从 0 变正"，只能写成
   "相对同种子对照**更长 / 衰减更慢**（配对差值 + t 检验）"。
   把判据写成"变正"，等于用一个本来就成立的命题冒充验收。

**③ 观测窗口必须长于元订单的最大执行周期。**
   指导书 §3.7 专门警告过。本脚本的窗口是 8000 tick；
   ``max_child_orders(200) / participation_rate(0.15) = 1333 tick``
   是上界，**留了 6 倍余量**。窗口短于执行周期会低估长记忆
   （元订单还在执行就被截断，等同于把长程相关剪掉）。

**④ 口径：预热 4000 tick（与 scenarios.py / 阶段5 一致）。**
   初始禀赋是一次性抽样的，开局几千 tick 是松弛过程，不是市场性质。

**⑤ 拟合护栏复用一期的实现（``tw/impact.py``）。**
   验收是"k 相比阶段3 更接近 0.5"——**要比较就必须用同一把尺子**。
   各写一份的话，"改善"可能全来自尺子变松，而且不会报错。

用法::

    python scripts/run_stage6.py
"""

from __future__ import annotations

import functools
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

# 与 run_stage5.py 同样的理由：平时 `python scripts/run_stage6.py` 跑，
# `scripts/` 自动在 sys.path 上；但测试要 `import scripts.run_stage6`
# 去验证机制接线（见 tests/test_stage_scripts.py），那时它不在。
# 显式补一条，两种跑法都能用。
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import FIG, OUT, banner, load_json, save_json  # noqa: E402
from _cache import (  # noqa: E402
    Cache,
    consts_sig,
    lib_sig,
    source_sig,
)
from run_stage3 import (  # noqa: E402
    AFTERSHOCK,
    HORIZON,
    MM_KW,
    MM_MIX,
    N_AGENTS,
    SEED0,
    SHOCK_SIZES,
    WARMUP as W3,
    causal_slippage,
    depth_profile,
    paired_impact,
    make_controls,
    recent_stats,
)
from tw import Market, Population, SimConfig  # noqa: E402
from tw.agents.adaptive_liquidity import AdaptiveLiquidityMixin  # noqa: E402
from tw.agents.chartist import Chartist  # noqa: E402
from tw.agents.fundamentalist import Fundamentalist  # noqa: E402
from tw.agents.zero_intel import ZeroIntelligence  # noqa: E402
from tw.agents.market_maker import MarketMaker  # noqa: E402
from tw.analyzer_longmemory import (  # noqa: E402
    longmemory_summary,
    order_sign_acf,
    order_signs,
    trade_signs,
)
from tw.impact import (  # noqa: E402
    depth_profile_gamma,
    fit_power_law,
    summarize,
    unsaturated,
)
from tw.order_flow.meta_order import MetaOrderConfig, MetaOrderMixin  # noqa: E402

# ----------------------------------------------------------------------
# 口径
#: 预热（与 scenarios.py 的 Scenario.warmup、阶段5 的 WARMUP 一致）
WARMUP = 4_000
#: 观测窗口。必须 ≫ 元订单最大执行周期（见模块文档 ③）。
OBS = 8_000
N_TICKS = WARMUP + OBS
#: ACF 最大 lag。取 500 而不是 200：实测 scale≥20 时 ACF 在 200 个 lag 内
#: **一次都没穿过零**（positive_run 顶到上限），200 会把真实衰减截断在视窗外。
MAX_LAG = 500

SEEDS = [SEED0, SEED0 + 7, SEED0 + 14, SEED0 + 21]
#: E6.4 用更少的种子（每次运行要 6000+400 tick，配对设计下开销最大）
SEEDS_IMPACT = [SEED0, SEED0 + 7, SEED0 + 14, SEED0 + 21]
#: 冲击规模档位：从阶段3 的 7 档里取 5 档（保留最小档，线性段在那里）
SHOCK_LEVELS = [0.001, 0.0025, 0.005, 0.01, 0.02]

#: 元订单的默认配置（起点；E6.5 会扫）
META_SHARE = 0.40
META_CFG = dict(
    p_meta_start=0.02, pareto_alpha=1.5, pareto_xmin=1.0,
    pareto_scale=20.0, participation_rate=0.15, child_qty_mean=1.0,
    max_child_orders=200,
)
#: 自适应流动性的默认配置
FLOW_WINDOW = 50
FLOW_SENS = 0.30


# ----------------------------------------------------------------------
class MetaChartist(MetaOrderMixin, Chartist):
    """图表派 + 元订单拆分执行。"""


# ----------------------------------------------------------------------
# 工作线B 新增：把元订单覆盖面扩展到零智能与基本面派
#
# ⭐ 为什么只需要"子类化"，不需要重写任何决策逻辑
# -------------------------------------------------
# 任务书（外部 AI 写的）给了两段 `on_tick(market_state)` 的伪代码，
# 并说"元订单执行期间方向被锁定、基础随机决策要被跳过"。
# **那件事 `MetaOrderMixin.decide` 已经做了**（见它的三种情形）：
#   ① 执行中 → 吐子单，方向沿用启动时锁定的那个，**这一 tick 不走基础逻辑**
#   ② 不在执行中 → 先问基础类要一张单，从它的方向取"意图"，再按概率启动
#   ③ 基础类没意图 → 什么都不做
# 所以：
#   · 零智能：方向来自它自己的随机选边，启动后被锁定 = "大额但无信息的
#     流动性需求（方向持续一段时间）"，正是任务书要的语义。
#   · 基本面派：方向来自**启动那一刻**的 premium 符号，之后即使 premium
#     被修正也不改向 = "锚定在启动那一刻的方向"，也正是任务书要的语义。
#
# ⚠️ 反过来，如果照任务书那样覆写 `on_tick`：`Agent` **没有**这个方法，
#    它**永远不会被调用**，机制会**静默失效**——不报错、只是效果为零。
#    这正是本项目最怕的一类错误（"跑了，但什么都没发生"）。
#
# ⚠️ 也**不要**覆盖 `KIND`：池子归属、主体统计与归因都靠它
#    （见 ``Market._build_agents`` 与 ``Population``）。子类保持基类的 KIND。
# ----------------------------------------------------------------------
class MetaZeroIntelligence(MetaOrderMixin, ZeroIntelligence):
    """零智能 + 元订单拆分执行（代表"大额但无信息的流动性需求"）。"""


class MetaFundamentalist(MetaOrderMixin, Fundamentalist):
    """基本面派 + 元订单拆分执行（代表"基于估值的大宗建仓/减仓"）。"""


#: 工作线B 的覆盖面表：kind → 用哪个 Meta 子类替换
META_CLASS_OF = {
    "chartist": MetaChartist,
    "zero_intel": MetaZeroIntelligence,
    "fundamentalist": MetaFundamentalist,
}


class AdaptiveMM(AdaptiveLiquidityMixin, MarketMaker):
    """做市商 + 自适应流动性（按订单流方向单侧稀疏挂单）。"""


class Stage6Market(Market):
    """把两套机制挂到指定池子上的市场。

    ``decorate_agent`` 是唯一的注入点（见 ``Market.decorate_agent``）。
    **关闭 = 用同一个类 + 惰性参数**，不是"换回裸类"——
    这样对照组与处理组的类相同、随机流消耗相同，
    配对差里才不会混进"类不同"这个无关变量。
    """

    def __init__(self, *args, meta_cfg: MetaOrderConfig | None = None,
                 meta_share: float = 0.0,
                 meta_shares: dict[str, float] | None = None,
                 adapt_mm: bool = False,
                 flow_window: int = FLOW_WINDOW,
                 flow_sensitivity: float = 0.0, **kw) -> None:
        # ⚠️ 必须在 super().__init__ 之前赋值：基类的 __init__ 里就会调
        # `_build_agents` → `decorate_agent`，那时这些属性得已经在。
        self._meta_cfg = meta_cfg or MetaOrderConfig(p_meta_start=0.0)
        self._meta_share = float(meta_share)
        #: 工作线B：按**类型**给激活比例，例如
        #: ``{"chartist": 0.4, "zero_intel": 0.5, "fundamentalist": 0.5}``。
        #: 给了它就忽略 ``meta_share``（后者是阶段6 的"只给图表派"简写）。
        self._meta_shares = dict(meta_shares) if meta_shares else {}
        self._adapt_mm = bool(adapt_mm)
        self._flow_window = int(flow_window)
        self._flow_sens = float(flow_sensitivity)
        super().__init__(*args, **kw)
        # 建完之后按比例"激活"各类主体。
        # 每个主体拿到**独立的** MetaOrderConfig 实例：共享一个可变配置对象
        # 是那种"改了 A 的结果 B 也变了"的经典事故。
        shares = dict(self._meta_shares)
        if not shares and self._meta_share > 0.0:
            shares = {"chartist": self._meta_share}   # 阶段6 的默认，行为不变
        cfg = self._meta_cfg
        for kind, share in shares.items():
            if float(share) <= 0.0:
                continue
            pool = [a for a in self.agents if a.KIND == kind]
            n_on = int(round(len(pool) * float(share)))
            for a in pool[:n_on]:
                a.meta_config = MetaOrderConfig(
                    p_meta_start=cfg.p_meta_start, pareto_alpha=cfg.pareto_alpha,
                    pareto_xmin=cfg.pareto_xmin, pareto_scale=cfg.pareto_scale,
                    participation_rate=cfg.participation_rate,
                    child_qty_mean=cfg.child_qty_mean,
                    max_child_orders=cfg.max_child_orders,
                )

    def decorate_agent(self, kind, cls, kw):
        # 工作线B：三类主体都可以挂元订单，**关闭的方式同样是"惰性参数"**。
        if kind in META_CLASS_OF:
            return META_CLASS_OF[kind], {
                **kw, "meta_config": MetaOrderConfig(p_meta_start=0.0)}
        if kind == "market_maker" and self._adapt_mm:
            return AdaptiveMM, {**kw, "flow_window": self._flow_window,
                                "flow_sensitivity": self._flow_sens}
        return cls, kw


def factory(*, meta_scale: float = 0.0, pareto_alpha: float = 1.5,
            flow_sensitivity: float = 0.0, adapt_mm: bool = True,
            meta_share: float = META_SHARE):
    """造一个 ``SimConfig -> Market`` 的工厂，供阶段3 的清算实验复用。

    ⚠️ **关闭机制的方式是"用同一个类 + 惰性参数"，不是"换回裸类"。**
    · 元订单关闭 → ``meta_share=0``（没有主体被激活），
      但 ``pareto_scale`` 必须仍是合法正值（配置对象总要构造出来）。
      第一版传 ``meta_scale=0`` 想表达"关掉"，直接撞上
      ``MetaOrderConfig`` 的参数校验——**这是好事**：
      校验拦下了一个"用非法配置冒充关闭"的写法。
    · 自适应流动性关闭 → 仍然用 ``AdaptiveMM`` 类，但 ``flow_sensitivity=0``。
      这样对照组与处理组的**类是同一个**、随机流消耗也相同，
      配对差里不会混进"类不同"这个无关变量。
      （``AdaptiveLiquidityMixin`` 在敏感度为 0 时逐点返回入参，
       且不引入任何随机流——见 ``tests/test_adaptive_liquidity.py``。）
    """
    cfg = MetaOrderConfig(**{**META_CFG, "pareto_scale": max(float(meta_scale), 1e-3),
                             "pareto_alpha": pareto_alpha})
    return functools.partial(
        Stage6Market,
        meta_cfg=cfg,
        meta_share=(meta_share if meta_scale > 0 else 0.0),
        adapt_mm=adapt_mm,
        flow_sensitivity=flow_sensitivity,
    )


# ----------------------------------------------------------------------
# 跑场缓存
#
# 为什么需要：本脚本一次要跑 ~90 场模拟（其中清算实验每场 6400 tick），
# 总耗时接近一小时。踩过的事故是：**最后一个实验段（E6.5）里一个
# ``s["n_children"]`` 的 KeyError，让前面 40 多分钟的 E6.1~E6.4 全部白跑**。
# 缓存让"改一个小 bug 要重跑一小时"变成"只重跑出错的那一段"。
#
# ⚠️⚠️ 本文件**真的踩过**"陈旧缓存"事故，教训写在下面：
#
# 原来的键是 ``{tag, seed, n_ticks, mix}``。看起来够用，但被缓存的量
# ——一场模拟的长记忆指标——还依赖**市场是怎么被装配的**，也就是
# ``Stage6Market.decorate_agent`` / ``factory`` 的行为。那段代码当时有个
# bug：对照组本应"机制全关"，却混进了 ``MetaOrderMixin`` 的默认
# ``p_meta_start=0.02``，**对照组根本不是对照**。
#
# 修掉代码后重跑 → 键一个字符没变 → 全部命中陈旧缓存 → 产出与修 bug
# 之前**逐位相同**的数字。没有事后核对的话，这个陈旧基线就会被当成
# "修复后"的证据写进报告。这正是"比没有缓存危险得多"的意思。
#
# 现在的三道锁（见 ``scripts/_cache.py``）：
#   1. ``BEHAVIOR_SIG`` —— 装配代码（类 + factory）+ 常量 + ``tw/`` 源码树
#      的内容哈希，进入每一个键；
#   2. 缓存文件头记录本轮签名，签名不符**整份作废**；
#   3. 命中/未命中数打出来——静默命中是这类事故的共同特征。
# ----------------------------------------------------------------------
CACHE_PATH = OUT / "stage6_cache.json"

BEHAVIOR_SIG = "".join([
    source_sig(Stage6Market, factory),
    lib_sig(),
    consts_sig(META_CFG=META_CFG, META_SHARE=META_SHARE, MM_MIX=MM_MIX,
               MM_KW=MM_KW, FLOW_SENS=FLOW_SENS, FLOW_WINDOW=FLOW_WINDOW,
               SHOCK_LEVELS=SHOCK_LEVELS, HORIZON=HORIZON, N_TICKS=N_TICKS,
               WARMUP=WARMUP, OBS=OBS, MAX_LAG=MAX_LAG),
])

CACHE = Cache(CACHE_PATH, label="阶段6")
CACHE.bind(BEHAVIOR_SIG)


# 注意：不要再在本文件里写一份 ``_Cache``。本项目曾经有两份几乎相同的实现，
# 而**两份的缓存键都不完备**——重复本身不是问题，"各写一份所以各漏一个输入"
# 才是。共用实现在 ``scripts/_cache.py``。


# ----------------------------------------------------------------------
def _lmf_run_uncached(
    mix: dict[str, float], seed: int, *, factory_fn=None,
    n_ticks: int = N_TICKS,
    tag: str = "",
) -> dict:
    """跑一场并返回长记忆指标 + 微观结构统计。"""
    pop = Population.from_shares(N_AGENTS, mix)
    cfg = SimConfig(seed=seed, n_ticks=n_ticks, population=pop, **MM_KW)
    m = (factory_fn or Market)(cfg)
    m.run(n_ticks)
    s = longmemory_summary(m.log.trades, m.log.flow, WARMUP, n_ticks,
                           max_lag=MAX_LAG)
    # 记账粒度对照：同一段数据按**逐笔**算一遍，量出"机械自相关"有多大
    win = [t for t in m.log.trades if WARMUP <= t.tick < n_ticks]
    a_trade = order_sign_acf(trade_signs(win), max_lag=10)
    s["acf_trade_lag1"] = float(a_trade[0]) if a_trade.size else float("nan")
    s["micro"] = recent_stats(m, window=OBS)
    n_meta = sum(int(getattr(a, "n_meta_started", 0)) for a in m.agents)
    n_kids = sum(int(getattr(a, "n_children", 0)) for a in m.agents)
    n_stretch = sum(int(getattr(a, "n_stretched", 0)) for a in m.agents)
    s["n_meta_started"] = n_meta
    s["n_children"] = n_kids
    s["n_stretched"] = n_stretch
    ok, prob = m.health_check()
    s["health_ok"] = bool(ok)
    s["health_problems"] = prob[:3]
    ok2, resid, _ = m.cash_conservation()
    s["cash_conserved"] = bool(ok2)
    s["cash_residual"] = float(resid)
    return s


def lmf_run(mix: dict[str, float], seed: int, *, factory_fn=None,
            n_ticks: int = N_TICKS, tag: str = "lmf") -> dict:
    """带缓存的 :func:`_lmf_run_uncached`。见上面 Cache 的说明。"""
    k = CACHE.key("lmf_run", tag=tag, seed=seed, n_ticks=n_ticks,
                  mix=sorted(mix.items()), sig=BEHAVIOR_SIG)
    return CACHE.get_or_run(k, lambda: _lmf_run_uncached(
        mix, seed, factory_fn=factory_fn, n_ticks=n_ticks, tag=tag))


def paired_diff(a_list, b_list) -> dict:
    """配对差值 ``b − a`` 的均值 ± 标准误 + t 统计量。

    同种子配对是这一阶段的关键：市场 RNG 只用于每 tick 洗牌与基本面步进，
    次数固定、与机制开关无关 → **基本面路径逐点相同**，
    共同的随机漂移被逐点相减消掉（一期实测降噪 62 倍）。
    """
    d = [b - a for a, b in zip(a_list, b_list)]
    s = summarize(d)
    return {"mean": s.mean, "sem": s.sem, "t": s.t, "n": s.n,
            "ci95": list(s.ci95), "values": [float(v) for v in d]}


# ----------------------------------------------------------------------
def main() -> dict:
    banner("阶段6  长记忆订单流 LMF")
    t0 = time.time()
    out: dict = {
        "stage": 6,
        "config": {
            "warmup": WARMUP, "obs": OBS, "max_lag": MAX_LAG, "seeds": SEEDS,
            "meta_share": META_SHARE, "meta_cfg": META_CFG,
            "flow_window": FLOW_WINDOW, "flow_sens": FLOW_SENS,
            "shock_levels": SHOCK_LEVELS,
            "n_agents": N_AGENTS, "mix": MM_MIX,
        },
    }

    # ==================================================================
    # E6.0  两条口径诊断
    # ==================================================================
    print("\n【E6.0a】记账粒度诊断：逐笔 vs 按订单")
    print("  一个主动单扫多档 → 多笔同向成交 → 逐笔算 ACF 会得到机械正自相关")
    print(f"    {'种子':>10}{'成交数':>10}{'订单数':>10}{'每单笔数':>10}"
          f"{'逐笔ACF1':>11}{'按订单ACF1':>12}{'差':>9}")
    e60 = []
    for sd in SEEDS[:2]:
        s = lmf_run(MM_MIX, sd, tag="e60a")
        ratio = s["n_trades"] / max(1, s["n_orders"])
        e60.append(s)
        print(f"    {sd:>10}{s['n_trades']:>10}{s['n_orders']:>10}{ratio:>10.2f}"
              f"{s['acf_trade_lag1']:>+11.4f}{s['acf_lag1']:>+12.4f}"
              f"{s['acf_lag1'] - s['acf_trade_lag1']:>+9.4f}")
    out["e6_0a_granularity"] = [
        {"seed": sd, "n_trades": s["n_trades"], "n_orders": s["n_orders"],
         "acf_trade_lag1": s["acf_trade_lag1"], "acf_order_lag1": s["acf_lag1"]}
        for sd, s in zip(SEEDS[:2], e60)
    ]
    print("  → 本脚本一律用**按订单**口径（LMF 的 ε 定义）。逐笔口径只作对照。")

    print("\n【E6.0b】观测窗口是否够长（指导书 §3.7 的预判坑）")
    mc = MetaOrderConfig(**META_CFG)
    need = mc.max_execution_ticks()
    print(f"    元订单最大执行周期上界 = max_child_orders/participation_rate "
          f"= {mc.max_child_orders}/{mc.participation_rate} = {need} tick")
    print(f"    观测窗口 = {OBS} tick，余量 {OBS / need:.1f}×  "
          f"{'✅ 足够' if OBS > 3 * need else '⚠️ 偏短'}")
    out["e6_0b_window"] = {"max_execution_ticks": need, "obs": OBS,
                           "headroom": OBS / need}
    print(f"    ACF 最大 lag 取 {MAX_LAG}（不是 200）：实测 scale≥20 时 ACF 在 200")
    print(f"    个 lag 内一次都没穿过零，200 会把真实衰减截断在视窗外。")

    # ==================================================================
    # E6.1 / E6.2  元订单：订单符号 ACF
    # ==================================================================
    print("\n【E6.1】基线对照（机制全关，同一批种子）")
    base = [lmf_run(MM_MIX, sd, tag="e61_base",
                    factory_fn=factory(meta_scale=0.0, flow_sensitivity=0.0))
            for sd in SEEDS]
    print(f"    {'种子':>10}{'订单数':>10}{'ACF1':>9}{'ACF10':>9}{'ACF100':>9}"
          f"{'连续正':>8}{'ΣACF':>9}{'γ':>8}{'H':>7}{'拟合':>6}")
    for sd, s in zip(SEEDS, base):
        print(f"    {sd:>10}{s['n_orders']:>10}{s['acf_lag1']:>+9.4f}"
              f"{s['acf_lag10']:>+9.4f}{s['acf_lag100']:>+9.4f}"
              f"{s['positive_run']:>8}{s['acf_sum']:>+9.3f}"
              f"{s['decay_exponent']:>8.3f}{s['hurst']:>7.3f}"
              f"{'✅' if s['fit_ok'] else '❌':>6}")
    out["e6_1_baseline"] = base

    print("\n【E6.2】元订单开启（40% 图表派，pareto_scale=20）")
    treat = [lmf_run(MM_MIX, sd, tag="e62_treat",
                     factory_fn=factory(meta_scale=META_CFG["pareto_scale"]))
             for sd in SEEDS]
    print(f"    {'种子':>10}{'元订单':>9}{'子单':>8}{'ACF1':>9}{'ACF10':>9}"
          f"{'ACF100':>9}{'连续正':>8}{'ΣACF':>9}{'γ':>8}{'H':>7}")
    for sd, s in zip(SEEDS, treat):
        print(f"    {sd:>10}{s['n_meta_started']:>9}{s['n_children']:>8}"
              f"{s['acf_lag1']:>+9.4f}{s['acf_lag10']:>+9.4f}"
              f"{s['acf_lag100']:>+9.4f}{s['positive_run']:>8}"
              f"{s['acf_sum']:>+9.3f}{s['decay_exponent']:>8.3f}"
              f"{s['hurst']:>7.3f}")
    keys = ("acf_lag1", "acf_lag10", "acf_lag100", "positive_run", "acf_sum",
            "decay_exponent", "hurst")
    diffs = {k: paired_diff([b[k] for b in base], [t[k] for t in treat])
             for k in keys}
    print(f"\n    配对差值（处理 − 对照，{len(SEEDS)} 种子）：")
    print(f"    {'指标':<14}{'基线均值':>11}{'处理均值':>11}{'差值':>10}"
          f"{'标准误':>9}{'t':>8}{'判定':>10}")
    for k in keys:
        bm = float(np.mean([b[k] for b in base]))
        tm = float(np.mean([t[k] for t in treat]))
        d = diffs[k]
        sig = abs(d["t"]) > 2.5 if np.isfinite(d["t"]) else False
        print(f"    {k:<14}{bm:>11.4f}{tm:>11.4f}{d['mean']:>+10.4f}"
              f"{d['sem']:>9.4f}{d['t']:>8.2f}"
              f"{('显著✅' if sig else '不显著'):>10}")
    print(f"\n    ⚠️ 基线**不是**「接近 0」：按订单口径它的 ACF(1) 已有约 "
          f"{float(np.mean([b['acf_lag1'] for b in base])):+.3f}，")
    print(f"       且能拟合出幂律衰减（γ ≈ "
          f"{float(np.mean([b['decay_exponent'] for b in base])):.2f}）。")
    print( "       来源是真实机制：基本面派盯持续的估值偏离、图表派盯持续的动量。")
    print( "       所以判据只能是「相对对照更长」，不能是「从 0 变正」——")
    print( "       后者是一个**本来就成立**的命题，用它冒充验收等于没验收。")
    acf_more = diffs["acf_sum"]["mean"] > 0 and diffs["positive_run"]["mean"] > 0
    t_ok = abs(diffs["acf_sum"]["t"]) > 2.5 if np.isfinite(diffs["acf_sum"]["t"]) else False
    e62_pass = bool(acf_more and t_ok)
    print(f"    → ΣACF 差值 {diffs['acf_sum']['mean']:+.3f}"
          f"（t={diffs['acf_sum']['t']:.2f}）、"
          f"连续正 lag 差值 {diffs['positive_run']['mean']:+.1f}："
          f"{'✅ E6.2 通过' if e62_pass else '❌ E6.2 未通过'}")
    out["e6_2_treatment"] = treat
    out["e6_2_diffs"] = diffs
    out["e6_2_pass"] = e62_pass

    # ==================================================================
    # E6.3  自适应流动性：深度剖面 γ
    # ==================================================================
    print("\n【E6.3】自适应流动性 → 盘口深度剖面 γ")
    print("  判据（指导书 §3.5）：γ 相比阶段3 基线提升至少 +0.5")
    print("  阶段3 的参考值：无做市商 γ=0.77、有做市商 γ=0.12（见 stage3_metrics.json）")
    print(f"    {'配置':<22}{'种子':>10}{'γ':>9}{'40bp内占比':>12}{'总深度':>10}")
    gamma_rows = []
    for tag, fn in (
        ("机制关（对照）", factory(meta_scale=0.0, flow_sensitivity=0.0)),
        (f"元订单 + 自适应({FLOW_SENS})",
         factory(meta_scale=META_CFG["pareto_scale"], flow_sensitivity=FLOW_SENS,
                 adapt_mm=True)),
    ):
        gs, shares, depths = [], [], []
        for sd in SEEDS[:3]:
            pop = Population.from_shares(N_AGENTS, MM_MIX)
            cfg = SimConfig(seed=sd, n_ticks=W3 + 50, population=pop, **MM_KW)
            m = fn(cfg)
            m.run(W3)
            mid = m.current_mid()
            levels = m.book.depth("buy", n=500)
            from tw.impact import cumulative_depth
            dist = np.abs(np.array([p for p, _, _ in levels]) / mid - 1.0) * 1e4
            qty = np.array([q for _, q, _ in levels])
            d_grid, cum = cumulative_depth(dist, qty, max_bp=400.0, n_bins=40)
            g = depth_profile_gamma(d_grid, cum, lo=20.0)
            gs.append(g)
            within40 = float(np.interp(40.0, d_grid, cum))
            depths.append(float(cum[-1]))
            shares.append(within40 / cum[-1] if cum[-1] > 0 else float("nan"))
            print(f"    {tag:<22}{sd:>10}{g:>9.3f}{shares[-1]:>12.1%}"
                  f"{depths[-1]:>10.1f}")
        gamma_rows.append({"tag": tag, "gamma": float(np.nanmean(gs)),
                           "within40_share": float(np.nanmean(shares)),
                           "total_depth": float(np.nanmean(depths)),
                           "per_seed_gamma": gs})
    g_off = gamma_rows[0]["gamma"]
    g_on = gamma_rows[1]["gamma"]
    print(f"\n    对照 γ = {g_off:.3f} → 处理 γ = {g_on:.3f}，提升 {g_on - g_off:+.3f}")
    e63_pass = bool(np.isfinite(g_on) and (g_on - g_off) >= 0.5)
    print(f"    → {'✅ E6.3 通过（≥ +0.5）' if e63_pass else '❌ E6.3 未通过（< +0.5）'}")
    print( "    真实市场 γ ≈ 2（远端厚，对应平方根律）。阶段3 的 0.77 是")
    print( "    「深度挤在近端」，正是冲击超线性的原因。")
    out["e6_3_depth"] = gamma_rows
    out["e6_3_delta_gamma"] = float(g_on - g_off)
    out["e6_3_pass"] = e63_pass

    # ==================================================================
    # E6.4  联合冲击复测（主验收项）
    # ==================================================================
    print("\n【E6.4】冲击幂律指数 k（主验收项）")
    print(f"  复用阶段3 的配对清算实验（同种子配对、因果滑点、分期清算 slices=20）")
    # ⚠️ 基准区间**从 stage3_metrics.json 现算**，不写死。
    # 指导书转述的"0.61~1.25"只对应主指标两项；照抄它会让这里的"基准"
    # 与自己的数据对不上（"窗口均偏离"那项实测到 3.09）。
    s3 = load_json("stage3_metrics.json")

    def _walk_k(o, prefix=""):
        if isinstance(o, dict):
            if "k" in o:
                try:
                    v = float(o["k"])
                    if v == v:
                        yield prefix, v
                except (TypeError, ValueError):
                    pass
            for kk, vv in o.items():
                yield from _walk_k(vv, f"{prefix}/{kk}")
        elif isinstance(o, list):
            for i, vv in enumerate(o):
                yield from _walk_k(vv, f"{prefix}[{i}]")

    kvals = list(_walk_k(s3)) if s3 else []
    kmain = [v for pp, v in kvals if ("因果滑点" in pp or "最差成交价" in pp)]
    if kvals:
        ks_txt = (f"{min(v for _, v in kvals):.2f}~{max(v for _, v in kvals):.2f}")
        kp_txt = (f"{min(kmain):.3f}~{max(kmain):.3f}" if kmain else "—")
        print(f"  真实市场 k ≈ 0.5（平方根律）。阶段3 的结果："
              f"全部 k 值 {ks_txt}，主指标两项 {kp_txt}。")
    else:
        print("  真实市场 k ≈ 0.5（平方根律）。（缺 stage3_metrics.json，无法现算基准）")
    print(f"  {'配置':<20}{'档位':>8}{'成交量':>10}{'成交率':>9}{'因果滑点bp':>12}")
    impact = {}
    for tag, fn in (
        ("机制关（对照）", factory(meta_scale=0.0, flow_sensitivity=0.0)),
        ("元订单+自适应", factory(meta_scale=META_CFG["pareto_scale"],
                                  flow_sensitivity=FLOW_SENS, adapt_mm=True)),
    ):
        ctrl = make_controls(MM_MIX, SEEDS_IMPACT, HORIZON, factory=fn)
        rows = []
        for f in SHOCK_LEVELS:
            # 清算实验是最贵的一段（每档 4 场 × 6400 tick）。加缓存，
            # 免得再出现"最后一个实验段崩了、前面一小时白跑"的事故。
            key = CACHE.key("impact", tag=tag, level=f, seeds=SEEDS_IMPACT,
                            horizon=HORIZON, sig=BEHAVIOR_SIG)
            r = CACHE.get_or_run(key, lambda f=f, ctrl=ctrl, fn=fn, tag=tag: (
                paired_impact(f"{tag}|{f:.4f}", MM_MIX, f, 20, SEEDS_IMPACT,
                              ctrl, HORIZON, factory=fn)))
            rows.append(r)
            print(f"    {tag:<20}{f:>8.2%}{r['delivered_qty']:>10.1f}"
                  f"{r['fill_ratio']:>9.0%}{r['slippage_bp']:>12.2f}")
            CACHE.flush()
        impact[tag] = rows
    print()
    fits = {}
    for tag, rows in impact.items():
        un = unsaturated(rows)
        print(f"    {tag}：有效档位 {len(rows)} → 未饱和 {len(un)}")
        for key, label in (("slippage_bp", "因果滑点"), ("worst_fill_bp", "最差成交价"),
                           ("during_mean_bp", "窗口均偏离")):
            fit = fit_power_law(un, key)
            fits[f"{tag}|{key}"] = fit
            print(f"      {label:<10} k={fit.exponent:>7.3f}  R²={fit.r_squared:>6.3f}  "
                  f"n={fit.n_points:>2}  |k−0.5|={fit.closeness_to_sqrt():>6.3f}  "
                  f"{fit.verdict()}" + (f"  ⚠️{fit.reason}" if not fit.ok else ""))
    k_off_fit = fits["机制关（对照）|slippage_bp"]
    k_on_fit = fits["元订单+自适应|slippage_bp"]
    k_off = k_off_fit.exponent
    k_on = k_on_fit.exponent
    d_off = k_off_fit.closeness_to_sqrt()
    d_on = k_on_fit.closeness_to_sqrt()
    print(f"\n    主指标（分期清算的因果滑点幂律指数）：")
    print(f"      对照 k = {k_off:.3f}（|k−0.5| = {d_off:.3f}）")
    print(f"      处理 k = {k_on:.3f}（|k−0.5| = {d_on:.3f}）")
    if np.isfinite(d_off) and np.isfinite(d_on):
        e64_pass = bool(d_on < d_off)
        print(f"      → {'✅ E6.4 通过（更接近 0.5）' if e64_pass else '❌ E6.4 未通过（更远离 0.5）'}"
              f"，改善 {d_off - d_on:+.3f}")
    else:
        e64_pass = False
        print("      → ❌ E6.4 无法判定（有一侧拟合失败）")
    print(f"    ⚠️ 无论达标与否都必须报数字。本阶段最容易发生的事就是")
    print(f"       「机制确实生效了，但没把 k 推到 0.5」——那就如实说差多少，")
    print(f"       不允许因为没完全达标就只报「机制已实现」。")
    out["e6_4_impact"] = {k: v for k, v in impact.items()}
    out["e6_4_fits"] = {
        k: {"ok": v.ok, "exponent": v.exponent, "r2": v.r_squared,
            "n": v.n_points, "reason": v.reason,
            "closeness": v.closeness_to_sqrt()}
        for k, v in fits.items()
    }
    out["e6_4_pass"] = e64_pass
    out["e6_4_delta_closeness"] = float(d_off - d_on) if np.isfinite(d_off) and np.isfinite(d_on) else float("nan")

    # ==================================================================
    # E6.5  参数敏感性
    # ==================================================================
    print("\n【E6.5】参数敏感性（pareto_alpha × flow_sensitivity 各 3 档）")
    print("  要回答：哪个参数对 k 的影响更大（决定阶段10 的校准优先级）")
    print("  口径：只看深度剖面 γ 与订单符号 ΣACF（不跑清算实验——")
    print("        每档都跑清算要 24 场×2 组，9 档就是 400+ 场，成本爆炸）")
    print(f"    {'alpha':>7}{'flow_sens':>11}{'γ':>9}{'ΣACF':>10}{'连续正':>8}{'子单':>9}")
    sens_rows = []
    for alpha in (1.2, 1.5, 2.0):
        for fs in (0.0, 0.3, 0.6):
            gs, sums, runs, kids = [], [], [], []
            for sd in SEEDS[:2]:
                fn = factory(meta_scale=META_CFG["pareto_scale"],
                             pareto_alpha=alpha, flow_sensitivity=fs, adapt_mm=True)
                pop = Population.from_shares(N_AGENTS, MM_MIX)
                cfg = SimConfig(seed=sd, n_ticks=N_TICKS, population=pop, **MM_KW)
                m = fn(cfg)
                m.run(N_TICKS)
                s = longmemory_summary(m.log.trades, m.log.flow, WARMUP,
                                       N_TICKS, max_lag=MAX_LAG)
                mid = m.current_mid()
                from tw.impact import cumulative_depth
                levels = m.book.depth("buy", n=500)
                dist = np.abs(np.array([p for p, _, _ in levels]) / mid - 1.0) * 1e4
                qty = np.array([q for _, q, _ in levels])
                d_grid, cum = cumulative_depth(dist, qty, max_bp=400.0, n_bins=40)
                gs.append(depth_profile_gamma(d_grid, cum, lo=20.0))
                sums.append(s["acf_sum"]); runs.append(s["positive_run"])
                # ⚠️ `n_children` 在**市场**上，不在 `longmemory_summary` 的返回里。
                # 第一版写成 `s["n_children"]` 直接 KeyError ——
                # 而它发生在 E6.5 的第 3 个配置，**前面 40 多分钟的 E6.1~E6.4 都白跑了**。
                # 教训：脚本里的最后一个实验段最容易成为"整轮重跑"的代价来源，
                # 所以每个取数都要么来自已经验证过的函数、要么就地算出来。
                kids.append(sum(int(getattr(a, "n_children", 0)) for a in m.agents))
            row = {"pareto_alpha": alpha, "flow_sensitivity": fs,
                   "gamma": float(np.nanmean(gs)),
                   "acf_sum": float(np.mean(sums)),
                   "positive_run": float(np.mean(runs)),
                   "n_children": float(np.mean(kids))}
            sens_rows.append(row)
            print(f"    {alpha:>7.1f}{fs:>11.1f}{row['gamma']:>9.3f}"
                  f"{row['acf_sum']:>+10.3f}{row['positive_run']:>8.1f}"
                  f"{row['n_children']:>9.0f}")
    spread_g = float(np.ptp([r["gamma"] for r in sens_rows]))
    spread_s = float(np.ptp([r["acf_sum"] for r in sens_rows]))
    print(f"\n    γ 的跨度 {spread_g:.3f}，ΣACF 的跨度 {spread_s:.3f}")
    out["e6_5_sensitivity"] = sens_rows
    out["e6_5_spread"] = {"gamma": spread_g, "acf_sum": spread_s}

    # ==================================================================
    _plot_acf(base, treat, FIG / "stage6_sign_acf.png")
    _plot_impact(impact, fits, FIG / "stage6_impact.png")
    _plot_depth(gamma_rows, FIG / "stage6_depth.png")
    _plot_sensitivity(sens_rows, FIG / "stage6_sensitivity.png")
    print(f"\n  图已输出到 {FIG}")

    # ==================================================================
    print("\n" + "=" * 74)
    print("阶段6 「诚实边界」小结")
    print("=" * 74)
    lines = [
        "【做到了什么量级】",
        f"  · 元订单机制生效：40% 图表派拆元订单，实测每场 "
        f"{float(np.mean([t['n_meta_started'] for t in treat])):.0f} 个元订单、"
        f"{float(np.mean([t['n_children'] for t in treat])):.0f} 张子单",
        f"  · 订单符号 ΣACF 相对同种子对照 {diffs['acf_sum']['mean']:+.3f}"
        f"（t={diffs['acf_sum']['t']:.2f}）",
        f"  · ACF 连续为正的 lag 数 {diffs['positive_run']['mean']:+.1f}",
        f"  · 深度剖面 γ 提升 {g_on - g_off:+.3f}（要求 ≥ +0.5）",
        f"  · 冲击幂律指数 k：{k_off:.3f} → {k_on:.3f}"
        f"（离 0.5 的距离 {d_off:.3f} → {d_on:.3f}）",
        "",
        "【做不到什么】",
        "  · **本阶段大概率无法把 k 推到 0.5。** 真实市场的凹性来自一整套",
        "    相互加强的机制（拆单 + 流动性提供者的方向性撤单 + 跨市场套利",
        "    + 手续费结构），本模型只实现了前两条。",
        "  · 元订单目前只给了图表派。基本面派与零智能没有拆分行为，",
        "    而真实市场里**所有**大额执行者都拆单。覆盖面不足会低估效应。",
        "  · 自适应流动性只有一个观察窗口与一个敏感度，没有'不同主体看到",
        "    不同窗口'的异质性；真实的做市商各有各的模型。",
        "  · 一期基线的订单符号 ACF **本来就不接近 0**（按订单口径 ACF(1) ≈ "
        f"{float(np.mean([b['acf_lag1'] for b in base])):+.3f}），",
        "    所以'出现了长记忆'这个说法在本模型里从一开始就不准确——",
        "    准确的说法是'记忆变长了多少'。这条修正必须写进结论。",
        "",
        "【下个阶段依赖它的哪个假设】",
        "  · 阶段9（多资产）依赖这里的冲击函数。**如果 k 没修到 0.5，",
        "    多资产扩展会把'大单成本被高估'这个偏差复制到每个资产对上**，",
        "    而且更难排查。这是指导书要求'不能跳过阶段6'的直接原因。",
        "  · 阶段10（统一校准）依赖 E6.5 的敏感性结论来决定先校哪个参数。",
        "  · 阶段7（反身性）不依赖本阶段（它依赖阶段5 的费率环境）。",
        "",
        "【两个必须写进结论的方法论发现】",
        "  · **记账粒度会伪造长记忆。** 逐笔成交的 ACF 里含有一块纯机械的",
        "    正自相关（同一张主动单扫多档 → 多条相邻同向记录）。",
        f"    实测：逐笔 vs 按订单的 ACF(1) 差 "
        f"{float(np.mean([s['acf_lag1'] - s['acf_trade_lag1'] for s in e60])):+.4f}。",
        "    LMF 的 ε 定义是**每张订单一个符号**，混用会让'扫单深度分布'",
        "    的差异被读成'订单流记忆'的差异。",
        "  · **判据不能写成'从 0 变正'。** 一期基线的记忆本来就不是 0，",
        "    那样的判据是一个恒真命题。必须用同种子配对差值 + t 检验。",
    ]
    for ln in lines:
        print("  " + ln if ln and not ln.startswith("【") else ln)
    out["honest_boundary"] = lines
    out["elapsed_sec"] = time.time() - t0
    CACHE.flush()
    CACHE.report()
    print(f"  缓存：命中 {CACHE.hits} 段 / 新跑 {CACHE.misses} 段"
          f"（{CACHE_PATH.name}）")
    save_json(out, "stage6_metrics.json")
    print(f"\n  总用时 {out['elapsed_sec']:.0f}s")
    return out


# ----------------------------------------------------------------------
def _plot_acf(base, treat, path) -> None:
    import matplotlib.pyplot as plt

    from tw.viz import C_ACCENT, C_GOOD, C_SIM, MUTED

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.5))
    ax = axes[0]
    for i, s in enumerate(base):
        ax.plot(np.arange(1, len(s["acf"]) + 1), s["acf"], color=C_SIM,
                lw=1.0, alpha=0.55, label="对照（机制关）" if i == 0 else None)
    for i, s in enumerate(treat):
        ax.plot(np.arange(1, len(s["acf"]) + 1), s["acf"], color=C_ACCENT,
                lw=1.2, alpha=0.8, label="元订单开启" if i == 0 else None)
    ax.axhline(0.0, color=MUTED, lw=0.9, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("lag（订单数，对数刻度）")
    ax.set_ylabel("订单符号 ACF")
    ax.set_title("① 订单符号自相关：元订单让记忆变长")
    ax.legend(fontsize=9)

    ax = axes[1]
    names = ["ACF(1)", "ACF(10)", "ACF(100)", "ΣACF", "连续正 lag"]
    bvals = [float(np.mean([s["acf_lag1"] for s in base])),
             float(np.mean([s["acf_lag10"] for s in base])),
             float(np.mean([s["acf_lag100"] for s in base])),
             float(np.mean([s["acf_sum"] for s in base])),
             float(np.mean([s["positive_run"] for s in base]))]
    tvals = [float(np.mean([s["acf_lag1"] for s in treat])),
             float(np.mean([s["acf_lag10"] for s in treat])),
             float(np.mean([s["acf_lag100"] for s in treat])),
             float(np.mean([s["acf_sum"] for s in treat])),
             float(np.mean([s["positive_run"] for s in treat]))]
    y = np.arange(len(names))
    ax.barh(y - 0.2, bvals, height=0.38, color=C_SIM, label="对照")
    ax.barh(y + 0.2, tvals, height=0.38, color=C_ACCENT, label="元订单")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.axvline(0.0, color=MUTED, lw=0.9)
    ax.set_xscale("symlog", linthresh=0.05)
    ax.set_title("② 各阶指标对比（注意：基线**不是** 0）")
    ax.legend(fontsize=9)
    fig.suptitle("E6.1/E6.2 元订单对订单流记忆的影响（按订单口径，预热后）",
                 fontsize=12.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_impact(impact: dict, fits: dict, path) -> None:
    import matplotlib.pyplot as plt

    from tw.viz import C_ACCENT, C_GOOD, C_REAL, C_SIM, MUTED

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6))
    ax = axes[0]
    for tag, color, marker in (("机制关（对照）", C_SIM, "o"),
                               ("元订单+自适应", C_ACCENT, "s")):
        rows = unsaturated(impact[tag])
        x = np.array([r["delivered_qty"] for r in rows])
        y = np.abs(np.array([r["slippage_bp"] for r in rows]))
        ax.plot(x, y, marker + "-", color=color, lw=1.8, ms=7, label=tag)
    ref = np.array([3.0, 200.0])
    k_off = fits["机制关（对照）|slippage_bp"]
    k_on = fits["元订单+自适应|slippage_bp"]
    for fit, color, lab in ((k_off, C_SIM, f"对照组拟合 k={k_off.exponent:.3f}"),
                            (k_on, C_ACCENT, f"处理组拟合 k={k_on.exponent:.3f}")):
        if fit.ok and fit.xs is not None:
            b = np.log(fit.ys).mean() - fit.exponent * np.log(fit.xs).mean()
            xs = np.array([fit.xs.min(), fit.xs.max()])
            ax.plot(xs, np.exp(b) * xs ** fit.exponent, "--", color=color, lw=1.4,
                    label=lab)
    _ = ref   # 只为在上面标出量级参照，不参与绘图
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("实际成交量（手，对数刻度）")
    ax.set_ylabel("|因果滑点| (bp)")
    ax.set_title(f"① 冲击函数（真实 k ≈ 0.5 = 平方根律）")
    ax.legend(fontsize=8.5)

    ax = axes[1]
    ks = []
    labels = []
    for tag in impact:
        fit = fits[f"{tag}|slippage_bp"]
        ks.append(fit.exponent if fit.ok else np.nan)
        labels.append(tag)
    ax.bar(range(len(ks)), ks, color=[C_SIM, C_ACCENT], width=0.55)
    ax.axhline(0.5, color=C_REAL, ls="--", lw=1.6, label="真实 k = 0.5")
    ax.axhline(1.0, color=MUTED, ls=":", lw=1.2, label="线性 k = 1")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("冲击幂律指数 k")
    ax.set_title("② k 离 0.5 有多远")
    ax.legend(fontsize=8.5)
    fig.suptitle("E6.4 分期清算的冲击函数（同种子配对、因果滑点口径）", fontsize=12.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_depth(rows: list[dict], path) -> None:
    import matplotlib.pyplot as plt

    from tw.viz import C_ACCENT, C_GOOD, C_REAL, C_SIM

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    gs = [r["gamma"] for r in rows]
    ax.bar(range(len(gs)), gs, color=[C_SIM, C_ACCENT], width=0.5)
    ax.axhline(2.0, color=C_REAL, ls="--", lw=1.6, label="真实市场 γ ≈ 2（平方根律）")
    ax.axhline(1.0, color="#8b95a5", ls=":", lw=1.2, label="线性 γ = 1")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r["tag"] for r in rows], fontsize=9)
    ax.set_ylabel("累计深度剖面指数 γ")
    ax.set_title("E6.3 深度剖面：γ 越大，远处越厚，冲击越凹")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_sensitivity(rows: list[dict], path) -> None:
    import matplotlib.pyplot as plt

    from tw.viz import C_ACCENT, C_GOOD, C_SIM

    alphas = sorted({r["pareto_alpha"] for r in rows})
    fss = sorted({r["flow_sensitivity"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.3))
    for ax, key, label in ((axes[0], "gamma", "深度剖面指数 γ"),
                           (axes[1], "acf_sum", "订单符号 ΣACF")):
        for a, col in zip(alphas, (C_SIM, C_GOOD, C_ACCENT)):
            ys = [next(r[key] for r in rows
                       if r["pareto_alpha"] == a and r["flow_sensitivity"] == f)
                  for f in fss]
            ax.plot(fss, ys, "o-", color=col, lw=1.7, ms=7, label=f"α={a}")
        ax.set_xlabel("flow_sensitivity（自适应流动性敏感度）")
        ax.set_ylabel(label)
        ax.set_title(f"对 {label} 的响应")
        ax.legend(fontsize=9)
    fig.suptitle("E6.5 参数敏感性：哪个旋钮更值得在校准阶段优先调", fontsize=12.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
