"""永续合约机制层（二期阶段5）。

目标
----
让市场**内生地**产生资金费率，量级与方向逼近真实 BTC 永续合约：
正费率占比 **85.8%**、年化约 **11.57%**（这两个数是用真实数据验证过的靶子）。

机制：真实永续的资金费率由两股需求驱动
--------------------------------------
1. **溢价**：永续价格相对现货指数的偏离。永续贵了 → 多头付钱给空头。
   单资产模型里用市场内生的基本面锚代理"现货指数"——
   它是模型里唯一的外生价值参照。
2. **多头压力**：市场**正在**积极地加多头。注意是"在加"，不是"有多少"。

⭐ 对指导书的四处必要修正（指导书按假想接口写，且公式有三处硬伤）
================================================================

**修正①（最重要）：``net_long_ratio`` 在只有交易的系统里恒等于常数。**
   指导书的拥挤度定义为 ``Σmax(inv,0)/Σ|inv|``。但**交易不改变 Σ持仓**：
   每笔成交里买家 +q、卖家 −q，总量守恒。所以这个比值只取决于
   **初始禀赋的分布**，在整场模拟里一动不动。实测：``mean_crowding = 1.0000``，
   500 次结算里**每一次**都是 1.0（本模型不许卖空，所有持仓恒为非负）。
   于是"拥挤度通道"退化成一个**常数偏置**——它让费率永远为正，
   却对任何参数变化毫无反应。
   真正能变化、且经济含义正确（"多头在加仓"）的信号是**主动订单流失衡**：
   ``flow_imbalance ∈ [−1,1]``，0 = 多空力量均衡。已作为拥挤度通道的实现。
   （``net_long_ratio`` 仍照指导书写出来并**报告**，作为这条修正的证据。）

**修正②：拥挤度必须相对中性点，不能直接乘。**
   指导书写 ``+ crowding_sensitivity * crowding``。即便拥挤度不退化，
   ``crowding ∈ [0,1]`` 也意味着多空平衡时（0.5）仍有一份正的费率贡献，
   于是"零偏好下费率均值置信区间覆盖 0"这条健全性检查**按构造就会失败**。
   改用失衡量（``flow_imbalance`` 本身就以 0 为中性）之后问题自然消失。

**修正③：资金费率必须通过交易所虚拟账户结算，否则现金不守恒。**
   指导书要求"全市场所有 agent 的 payment 之和必须精确为 0"。
   但 ``payment_i = −inventory_i·rate·mark`` 对 i 求和 ``= −rate·mark·Σinventory``，
   而 ``Σinventory ≠ 0``（初始禀赋给了每人正持仓）。
   真实市场里永续是**合约**，多空仓位由撮合对冲，必然净额为 0；
   本模型的 ``inventory`` 是**现货持仓**，不是合约仓位。
   所以在交易层引入一个**交易所虚拟账户**承担对手方（清算所的实际角色），
   守恒式变成 ``Σ主体payment + 交易所变动 == 0``，**精确成立**。
   交易所账户随"系统净多头 × 正费率"持续累积——
   这正是"净多头市场为什么长期付正费率"的经济来源，不是漏洞。

**修正④：结算触发用计数器，不用 tick 取模。**
   取模写法在"分块续跑"（``run()`` 分块、GUI 作业分块）时会因为起点不同而错位。
   显式计数器 ``_since_settle`` 与跑法无关。

⭐ 还有一个必须处理的副作用：**结算会让部分主体的预留现金超支**
----------------------------------------------------------------
资金费率扣钱之后，某主体的 ``reserved_cash`` 可能超过 ``cash``，
于是 ``available_cash < 0``，之后它的每一张新单都被预算裁剪拒掉——
**一期踩过的"市场静默冻结"就是这个症状**。
所以在结算后必须调 ``Market.reconcile_reservations()``：
撤掉超出的挂单，把不变量修回来。真实交易所遇到保证金不足也是先撤挂单。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .market import Market

#: 真实 BTC 永续合约的校准靶子（用真实数据量出）。
REAL_FUNDING_ANNUAL = 0.1157      # 年化资金费率
REAL_FUNDING_POS_FRAC = 0.858     # 正费率占比

#: 1 tick ≈ 1 小时；每 8 tick 结算一次 ≈ 真实的 8 小时周期。
SETTLEMENTS_PER_YEAR = 24 * 365 / 8   # = 1095


def annualize(rate_per_settlement: float) -> float:
    """把单次结算费率折成年化（单利口径，与真实数据的算法一致）。"""
    return float(rate_per_settlement) * SETTLEMENTS_PER_YEAR


#: 真实靶子换算成「每次结算」口径，标定脚本与测试都从这里取，避免各写一遍。
REAL_FUNDING_MEAN = REAL_FUNDING_ANNUAL / SETTLEMENTS_PER_YEAR


def rate_from_series(
    premium,
    crowding,
    *,
    premium_sensitivity: float,
    crowding_sensitivity: float,
    clamp_bp: float,
    scale: float = 1.0,
) -> np.ndarray:
    """用一组**已记录的** (premium, crowding) 序列离线重算费率序列。

    为什么可以离线重算 —— 以及它**什么时候不成立**
    --------------------------------------------
    费率进不了行情，除非有人读它。本模型里只有 ``FundingArbitrageur`` 读
    ``state.funding_rate``。所以：

    · **市场里没有套利者时**：费率对行情的唯一影响是「改现金」。
      把传导系数设成 0 跑一场（= 完全没有资金费率），记录的 premium/crowding
      序列只由行情本身决定；换个系数重算是**精确的**，且省掉几十次重跑。
    · **一旦注入套利者**：费率 → 套利者行为 → 行情 → 费率，反馈闭环成立，
      离线重算**失效**，必须实跑。

    ⚠️ 还有一条容易忽略的耦合：**资金费率会改现金，现金会改下单预算**。
    所以就算没有套利者，"关掉费率跑一场再离线重算"得到的费率，
    与"开着费率实跑"得到的费率**并不完全相同**——差别就是这条现金反馈。
    标定用前者（便宜），验收用后者（真实），两者的差本身就是一条测量结果。

    这条函数的存在意义就是这个分工：让「找参数」变便宜，同时不假装
    便宜的结果等于真实结果。
    """
    prem = np.asarray(premium, dtype=np.float64)
    crow = np.asarray(crowding, dtype=np.float64)
    if prem.shape != crow.shape:
        raise ValueError("premium 与 crowding 长度必须一致")
    raw = float(scale) * (
        float(premium_sensitivity) * prem + float(crowding_sensitivity) * crow
    )
    lim = float(clamp_bp) / 1e4
    return np.clip(raw, -lim, lim)


def signal_to_noise(x) -> float:
    """μ/σ —— 一个信号「能不能撑起 85.8% 的正号」的判据。

    推导：若费率近似正态，``P(rate>0) = Φ(μ/σ)``。
    真实靶子 85.8% 反解出 ``μ/σ = Φ⁻¹(0.858) = 1.070``。
    所以任何想让正费率占比逼近真实的通道，**必须自己先把 μ/σ 顶到 1 附近**；
    ``scale`` 是正的乘数，它同时放大 μ 和 σ，**对 μ/σ 完全无效**。
    这条恒等式是整个阶段5 标定的出发点。
    """
    a = np.asarray(x, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size < 3:
        return float("nan")
    sd = float(a.std(ddof=1))
    if sd <= 0:
        return float("inf")
    return float(a.mean()) / sd


#: 让 ``P(rate>0)`` 命中真实正占比所需的 μ/σ。``Φ⁻¹(0.858)``，无 scipy 依赖。
TARGET_SNR = 1.0704


@dataclass(slots=True)
class FundingConfig:
    """资金费率参数。

    默认值是 **E5.0 标定的结果**，不是拍的，也不是照抄指导书
    （指导书的 ``funding_sensitivity=0.1 / crowding_sensitivity=0.05`` 在本模型里
    把 89% 的话语权给了**噪音通道**，见下）。

    ⭐ 标定的推导链（可复现：``python scripts/run_stage5.py``）
    --------------------------------------------------------
    1. 费率是两条通道的线性叠加：``rate = scale·(a·premium + b·crowding)``。
    2. ``scale`` 是正的乘数，同时放大均值与标准差 ⇒ **它改不动正号占比**。
       于是「正费率占比 85.8%」这个靶子只能靠**通道混合比例** θ = b/a 去命中，
       幅度交给 ``scale``。这是个一维问题，可以精确求解。
    3. 正态近似下 ``P(rate>0) = Φ(μ/σ)``，85.8% 反解出 ``μ/σ = 1.070``
       （见 ``TARGET_SNR``）。所以两条通道**各自**的 μ/σ 必须先够得着 1。
    4. 实测（**预热 4000 tick 之后**，4 种子 × 12000 tick × 300 主体，
       费率关掉的基准跑 —— 为什么必须丢预热见 ``funding_rates_since``）：

       ==========  ==============  ================
       p_buy       溢价通道 μ/σ     拥挤度通道 μ/σ
       ==========  ==============  ================
       0.50        −0.008          −0.118
       0.55        +0.029          +0.777
       0.60        **+0.003**      **+1.201**
       0.65        +0.030          +1.572
       0.70        +0.012          +1.947
       ==========  ==============  ================

       ⭐ **溢价通道的 μ/σ 在任何偏好强度下都 ≈ 0**——它整条都是噪音，
       且这个结论在丢掉预热段之后变得更干净（预热段里它的均值反而有 3~80bp，
       那同样是初始禀赋的产物，不是市场的性质）。
       原因：本模型只有一个价格，「溢价」只能拿中间价对基本面锚做代理，
       量出的是**价格发现误差**（sd ≈ 65bp），而真实永续的 basis
       通常只有几个 bp。真实 basis 小，是因为永续与现货是同一个资产、
       被同一批做市商套着；本模型没有第二条价格序列，做不出这个紧耦合。
       该缺陷的根治方案属于阶段9（多资产），这里如实记录。
    5. θ 由**二分**求解（不用网格挑点）：θ = 0（纯溢价）时正占比
       **恒为 0.49~0.51，无论偏好多强**；θ ≥ 0.2 后进入平台。
       目标 0.858 在 p_buy = 0.60 这一档由 θ* = 0.13671875 精确命中
       （二分残差 0.00e+00）。
       ⚠️ 上一版用「在 θ 网格里挑最近的点」，结果**同一个问题在 121 点网格上
       给出 θ=0.126、在 126 点网格上给出 θ=0.132**——差 5%，而两个都"通过验收"。
       这个差异没有物理含义，是纯粹的离散化产物，却会一路传进本类的默认值。
       二分给出连续问题的真解，与网格分辨率无关。
    6. ``scale`` 由「均值命中 1.057bp（年化 11.57%）」反解。

    标定结论（预测都是**预热后口径**）：

    ==============  ==========  ==========  ==========
    项               目标        预测        实跑
    ==============  ==========  ==========  ==========
    年化             0.1157      0.1157      见报告
    正费率占比         0.858       0.855       见报告
    费率 sd            ≈1bp       0.996bp     见报告
    ==============  ==========  ==========  ==========

    ⚠️ 离线预测与实际实跑之间存在系统性差距，来源已定位：离线推导用的是
    「费率关掉」的基准跑，而费率一旦打开就会通过**现金**反过来影响下单预算
    ——这条反馈在离线模型里不存在。差距的大小本身就是一条测量结果，
    见 ``docs/` 的交付报告「诚实边界」一节。

    ⚠️ **平台区的存在意味着标定不是唯一的**：θ ≥ 0.2 之后 pos_frac 不再变化
    （p_buy=0.60 那档停在 0.876）。选 θ = 0.126 而不是更大的值，是因为
    目标 0.858 落在上升段上、需要保留可调余量；但这个选择**对 θ 敏感**
    （∂pos/∂θ ≈ 0.2），报告里必须给出这份敏感性，不能只报一个点估计。
    """

    #: 每多少个 tick 结算一次。8 tick ≈ 8 小时（1 tick ≈ 1 小时约定）。
    settle_interval_ticks: int = 8
    #: 溢价 → 费率的传导系数。premium 是相对数（0.01 = 1%）。
    #: 标定值 0.05，与 crowding_sensitivity 的比值 θ = 0.1259。
    premium_sensitivity: float = 0.05
    #: 订单流失衡 → 费率的传导系数。失衡量 ∈ [−1,1]，中性点为 0。
    #: 标定值 = 二分求解 θ* 的原值（**不要手抄，用 scripts/run_stage5.py 复现**）。
    #: 注意两条通道单位不同：乘完之后拥挤度通道贡献了 **99.9%** 的费率水平——
    #: 所以「溢价通道」在标定后的配置里几乎只是一个微调项。
    #: 这是被数据逼出来的结论，不是设计意图。
    crowding_sensitivity: float = 0.0068359375
    #: 主动订单流失衡的观察窗口（tick）。
    crowding_window: int = 300
    #: 单次费率的绝对值上限（bp）。防止极端配置下费率失控。
    #: 标定后实际触发率为 **0.00%**——真实费率本身就是被制度压住的量，
    #: 上限不触发才是对的；指导书 §2.7 担心的「clamp 太紧压平扫描」
    #: 在标定前的默认系数下确实成立（触发率 2%~3%），标定后消失。
    clamp_bp: float = 75.0
    #: 总传导标度。标定值由「均值命中年化 11.57%」反解。
    #: 标定阶段用它统一缩放两项，避免分别扫两个系数。
    scale: float = 0.12429784

    @property
    def clamp(self) -> float:
        return self.clamp_bp / 1e4


class PerpetualMarket(Market):
    """在 ``Market`` 主循环末尾插入资金费率结算钩子。

    用**子类**而不是给 Market 加参数：一期 159 项测试全部跑在裸 Market 上，
    把结算写进主循环会让一期所有统计特征悄悄改变（多了一条现金流），
    而那些测试的基准值、报告里的数字都会跟着漂，且没有任何报错。
    """

    def __init__(
        self, *args, funding_config: FundingConfig | None = None, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        cfg = funding_config or FundingConfig()
        if cfg.settle_interval_ticks < 1:
            raise ValueError("settle_interval_ticks 必须 >= 1")
        if cfg.premium_sensitivity < 0 or cfg.crowding_sensitivity < 0:
            raise ValueError("传导系数必须非负")
        if cfg.clamp_bp <= 0:
            raise ValueError("clamp_bp 必须为正")
        if cfg.crowding_window < 1:
            raise ValueError("crowding_window 必须 >= 1")
        self.funding_config = cfg
        # 永续天然带杠杆：资金费率是从**保证金**里扣的，扣成负数很正常
        # （= 向交易所借款）。裸 Market 保持无杠杆，这个开关只在这里打开。
        self.allow_negative_cash = True

        self.funding_records: list[dict] = []
        self.settle_ticks: list[int] = []
        self._since_settle = 0
        self._rate_hist: list[float] = []
        #: 最近一次结算的资金费率。``MarketState.funding_rate`` 每 tick 读它。
        self.last_funding_rate = 0.0
        #: 交易所账户累计吸收的资金费率（= −Σ主体payment）。见模块文档修正③。
        self.funding_exchange_total = 0.0
        #: 因结算而被迫撤单的累计笔数（见模块文档「副作用」一节）。
        self.reconciled_orders = 0

    # ------------------------------------------------------------------
    def step(self) -> None:
        super().step()
        self._since_settle += 1
        if self._since_settle >= self.funding_config.settle_interval_ticks:
            self._settle_funding()
            self._since_settle -= self.funding_config.settle_interval_ticks

    # ------------------------------------------------------------------
    # 驱动量
    # ------------------------------------------------------------------
    def _compute_premium(self) -> float:
        """永续价相对现货指数（= 基本面锚）的溢价。

        锚为空或非正时返回 0（而不是抛异常）——预热早期盘口可能还没有中间价，
        "没有溢价信息"比"报错中断"更合理。
        """
        mid = self.current_mid()
        anchor = self.fundamental
        if not (mid and mid > 0 and anchor and anchor > 0):
            return 0.0
        return float((mid - anchor) / anchor)

    def _compute_net_long_ratio(self) -> float:
        """指导书定义的「净多头持仓占比」= Σmax(持仓,0) / Σ|持仓|。

        ⚠️ **保留它只是为了留证据**：交易不改变 Σ持仓，所以这个比值
        只由初始禀赋决定，整场模拟恒定。实测 500 次结算全部 = 1.0。
        它**不是**可用的拥挤度信号（见模块文档修正①）。
        """
        total_long = 0.0
        total_abs = 0.0
        for a in self.agents:
            inv = a.inventory
            if inv > 0:
                total_long += inv
            total_abs += abs(inv)
        if total_abs <= 1e-12:
            return 0.5
        return float(total_long / total_abs)

    def _compute_crowding(self) -> float:
        """多头压力 = 近期**主动订单流**净失衡 ∈ [−1, 1]。

        用"流量"而不是"存量"，是因为存量在本模型里恒为常数（见修正①）。
        含义也更贴近现实：资金费率反映的是**当下做多的迫切程度**，
        而不是"长期持有多少"。
        """
        return float(self._flow_imbalance(window=self.funding_config.crowding_window))

    # ------------------------------------------------------------------
    def _settle_funding(self) -> dict:
        cfg = self.funding_config
        premium = self._compute_premium()
        crowding = self._compute_crowding()
        net_long = self._compute_net_long_ratio()
        raw = cfg.scale * (
            cfg.premium_sensitivity * premium
            + cfg.crowding_sensitivity * crowding
        )
        lim = cfg.clamp
        rate = float(max(-lim, min(lim, raw)))
        clamped = abs(raw) > lim

        mark = self.current_mid()
        moved = 0.0
        paid = 0.0
        for a in self.agents:
            nominal = -a.inventory * rate * mark
            if nominal == 0.0:
                continue
            # 精确付款。**不做封顶**——封顶会让"付不满"的部分由交易所凭空承担，
            # 实测有 8% 的付款被吃掉，资金费率的传导就变得名不副实。
            # 现金被扣成负数是对的：永续是杠杆产品，这就是"向交易所借钱"，
            # 只要**权益不为负**即可（见 Market.allow_negative_cash）。
            pay = nominal
            # ⚠️ 唯一记账路径。直接改 a.cash 会破坏全系统现金守恒，
            # 而 account_integrity 完全看不出来（它只查单账户自洽）。
            self.apply_cash_delta(a, pay, reason="funding_settlement")
            moved += pay
            if pay < 0:
                paid += -pay
            # 归因钩子：主体可以记录"我收了多少费率"，但**不允许再动 cash**。
            hook = getattr(a, "on_funding", None)
            if hook is not None:
                hook(rate, mark)
            # 扣钱之后预留可能超支 → 必须修回来，否则该主体之后一张单都挂不出去，
            # 市场会在几百 tick 后**静默冻结**（一期踩过）。
            if a.reserved_cash > a.cash + 1e-9:
                self.reconciled_orders += self.reconcile_reservations(a)

        # ⚠️ 只扣一次。这里曾因为粘贴写重了一行，导致 `exchange_absorbed`
        # 报告成真值的 **2 倍**——而 `cash_conservation()`（读的是
        # `apply_cash_delta` 维护的 `exchange_cash`）仍然完全正常，
        # 所以守恒测试**抓不到它**。只有把摘要里的交易所吸收额与
        # `exchange_cash` 对账才发现。已补断言（见 test_perpetual.py）。
        self.funding_exchange_total -= moved

        rec = {
            "tick": int(self.tick),
            "rate": rate,
            "rate_bp": rate * 1e4,
            "premium": premium,
            "crowding": crowding,
            "net_long_ratio": net_long,
            "clamped": bool(clamped),
            "agents_paid": paid,
            "sum_payment": moved,
        }
        self.funding_records.append(rec)
        self.settle_ticks.append(int(self.tick))
        self._rate_hist.append(rate)
        self.last_funding_rate = rate
        return rec

    # ------------------------------------------------------------------
    # 统计
    # ------------------------------------------------------------------
    def funding_rates(self) -> np.ndarray:
        return np.asarray(self._rate_hist, dtype=np.float64)

    def funding_rates_since(self, tick: int) -> np.ndarray:
        """只要 ``tick`` 之后的结算费率。

        ⭐ 为什么几乎总该用它而不是 ``funding_rates()``
        ---------------------------------------------
        本模型的初始禀赋是**一次性抽样**的（每人持仓 ~U(0,20)、现金若干），
        市场在开头的几千 tick 里处于「把这份随机禀赋消化掉」的松弛过程。
        这段过程里**主动订单流是严重单边的**（大家都在减仓换现金），
        实测拥挤度在 p_buy=0.70 时第 8 tick 是 **−0.93**，
        要到约 3000 tick 才衰减回 0，之后才显出真正的正偏（+0.15）。

        于是：**把开头那段算进费率统计，会把正费率占比系统性压低**
        （实测 p_buy=0.70 在 2000 tick 上甚至得出**负**的年化 −3.2%）。
        这个瞬态是**初始条件的产物，不是市场的性质**——
        真实市场没有"某天所有参与者被随机分配了持仓"这回事。
        项目的其他地方（``scenarios.py``）早就统一用 4000 tick 预热，
        资金费率统计必须用同一个口径，否则各处的数字不可比。
        """
        t0 = int(tick)
        r = np.asarray(
            [r for t, r in zip(self.settle_ticks, self._rate_hist) if t > t0],
            dtype=np.float64,
        )
        return r

    def funding_summary(self, since_tick: int = 0) -> dict:
        """费率统计。

        ``since_tick`` 用来丢掉预热段（默认 0 = 全程，主要是为了兼容
        测试里那种只跑几百 tick 的短场次；**做对标结论时必须传预热长度**）。
        """
        recs = [d for d in self.funding_records if d["tick"] > int(since_tick)]
        r = np.asarray([d["rate"] for d in recs], dtype=np.float64)
        if r.size == 0:
            return {"n": 0, "since_tick": int(since_tick)}
        ann = annualize(float(r.mean()))
        return {
            "n_settlements": int(r.size),
            "since_tick": int(since_tick),
            "first_tick": int(recs[0]["tick"]),
            "mean_rate": float(r.mean()),
            "mean_bp": float(r.mean() * 1e4),
            "sd_bp": float(r.std(ddof=1) * 1e4) if r.size > 1 else float("nan"),
            "annualized": float(ann),
            "pos_frac": float((r > 0).mean()),
            "neg_frac": float((r < 0).mean()),
            "clamped_frac": float(np.mean([d["clamped"] for d in recs])),
            "mean_premium_bp": float(np.mean([d["premium"] for d in recs]) * 1e4),
            "mean_crowding": float(np.mean([d["crowding"] for d in recs])),
            "mean_net_long_ratio": float(np.mean([d["net_long_ratio"] for d in recs])),
            "net_long_ratio_sd": float(np.std([d["net_long_ratio"] for d in recs])),
            "exchange_absorbed": float(self.funding_exchange_total),
            "reconciled_orders": int(self.reconciled_orders),
            "interval_ok": self.settle_interval_is_uniform(),
            "real_annual": REAL_FUNDING_ANNUAL,
            "real_pos_frac": REAL_FUNDING_POS_FRAC,
            "annual_ratio_to_real": float(ann / REAL_FUNDING_ANNUAL),
        }


    def settle_interval_is_uniform(self) -> bool:
        """结算时点是否严格等间隔。不等间隔说明触发逻辑写错了。"""
        if len(self.settle_ticks) < 3:
            return True
        d = np.diff(np.asarray(self.settle_ticks))
        return bool(np.all(d == self.funding_config.settle_interval_ticks))
