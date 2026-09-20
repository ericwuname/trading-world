"""自适应流动性：感知订单流方向，主动在被预测的方向上稀疏挂单（二期阶段6）。

这是 LMF 链条的**第二环**，也是"冲击为什么是凹的"真正的答案所在。

机制
----
流动性提供者（做市商 / 零智能挂单方）观察最近一段的**主动订单流失衡**。
若发现买方占优，它就预判"买单还会来"，于是：

    · 把**卖单**（它将要卖给追买者的那些）挂得**更远** → 卖侧近端变薄
    · 把买单挂得更近或不动（它想买，不怕有人砸）

净效果：**单侧**流动性变薄。一个中等规模的买单一上来就要吃穿更多档，
边际价格跳得更深——于是"小额订单也有不成比例的大冲击"，
冲击函数从线性/超线性变成**凹的**。

⚠️ 三个必须写清楚的约束
======================

**① 只允许把挂单推远，不允许拉近（不允许负的 shift）。**
   如果允许负 shift，流动性提供者就变成"追着订单流往对手方向让价"——
   它主动制造跨价套利机会，自己赔钱给别人。这在真实市场里不会持续存在
   （做市商会被套利到退出），而模型里会表现为**价差系统性变负**、
   盘口自我吞噬。实测症状是盘口深度剖面反而变平。
   所以 ``adjusted_offset`` 里有个 ``max(0.0, ·)`` 的地板。

**② 两侧不对称：只有"将来会被打的那一侧"往外推。**
   买方占优时，被打的是**卖侧**（买单要吃卖单）。所以：
   · side='sell' 且 imbalance>0 → 往外推（正）
   · side='buy'  且 imbalance<0 → 往外推（正）
   · 其余组合 → 不动
   写成对称的形式（两侧都推）会把盘口整体撑宽，
   那是"波动率上升"的另一种表现，不是"单侧变薄"，测出来的不是同一个东西。

**③ 推的幅度必须相对**base_offset*，不能是绝对价差。**
   零智能主体的报价幅度在标定期被压到 0.02%~0.18%（见 ``标定说明.md``），
   一个绝对量（比如"推 5bp"）会直接把它推出盘口之外，等于让这批主体消失。
   用相对量（乘在 base_offset 上）才能让"变薄"是渐进的。
"""

from __future__ import annotations

from ..types import MarketState

#: spawn_rng 用途标签不需要——本 Mixin 不引入新的随机流（它是**确定性**响应）。
#: 这一点值得强调：加上一条随机流会让"自适应"变成"又一层噪声"，
#: 而它要做的恰恰是从噪声里提取方向。


class AdaptiveLiquidityMixin:
    """让挂单方对"最近的主动订单流方向"做出反应。

    用法::

        class AdaptiveZI(AdaptiveLiquidityMixin, ZeroIntelligence): ...

    ⚠️ MRO 顺序不能反（同 ``MetaOrderMixin``）：Mixin 必须在前，
    它的 ``adjusted_offset`` 才会覆盖基类的。
    """

    #: 观察窗口（tick）。太短 → 被单笔噪声主导；太长 → 反应迟钝。
    #: 50 是指导书给的起点，E6.5 会扫。
    flow_window: int = 50
    #: 敏感度：失衡量 ∈ [−1,1]，乘以它就是"最多把报价幅度放大多少倍"。
    #: 0 = 关闭（完全退化成基类行为），1 = 满失衡时幅度翻倍。
    flow_sensitivity: float = 0.3
    #: 单侧最多放大多少倍（护栏）。没有它，flow_sensitivity 一大，
    #: 报价会被推到盘口之外，主体静默退出市场——而"深度变薄"的结论
    #: 就变成了"主体消失了"，两者不是同一件事。
    max_stretch: float = 3.0

    def __init__(self, *args, flow_window: int | None = None,
                 flow_sensitivity: float | None = None,
                 max_stretch: float | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if flow_window is not None:
            self.flow_window = int(flow_window)
        if flow_sensitivity is not None:
            self.flow_sensitivity = float(flow_sensitivity)
        if max_stretch is not None:
            self.max_stretch = float(max_stretch)
        if self.flow_window < 1:
            raise ValueError("flow_window 必须 >= 1")
        if self.flow_sensitivity < 0:
            raise ValueError("flow_sensitivity 必须非负（只允许变薄，不允许变厚）")
        if self.max_stretch < 1.0:
            raise ValueError("max_stretch 必须 >= 1")
        #: 归因：报价被推开过多少次、平均推开多少（用于确认机制真的在工作）
        self.n_stretched = 0
        self.stretch_sum = 0.0

    # ------------------------------------------------------------------
    def flow_imbalance(self, state: MarketState) -> float:
        """取当前窗口的主动订单流失衡 ∈ [−1, 1]。

        优先用 ``state.flow``（支持任意窗口的只读视图）；若拿不到
        （一期市场或手工构造的 state），退回预计算的 ``state.flow_imbalance``。
        **两条路径的口径必须一致**——它们是同一个公式的两次实现，
        ``tests/test_adaptive_liquidity.py`` 里有一条断言把两者钉在一起。
        """
        view = getattr(state, "flow", None)
        if view is not None:
            return float(view.imbalance(self.flow_window))
        return float(getattr(state, "flow_imbalance", 0.0))

    def adjusted_offset(self, base_offset: float, state: MarketState,
                        side: str) -> float:
        """把基准报价幅度按订单流失衡推远。**只推远，不拉近。**

        ``base_offset`` 是正的相对幅度（0.001 = 离中间价 0.1%）。
        返回一个**不小于** ``base_offset`` 的值。
        """
        if base_offset <= 0:
            return base_offset
        imb = self.flow_imbalance(state)

        # 被打的那一侧才往外推：
        #   别人在买（imb>0）→ 我要卖的卖单会被打 → 卖单推远
        #   别人在卖（imb<0）→ 我要买的买单会被打 → 买单推远
        if side == "sell":
            pressure = imb
        elif side == "buy":
            pressure = -imb
        else:
            raise ValueError(f"side 必须是 buy/sell，收到 {side!r}")
        if pressure <= 0.0:
            return base_offset

        mult = min(self.max_stretch, 1.0 + self.flow_sensitivity * pressure)
        # ⚠️ 地板：绝不允许返回小于 base_offset 的值（= 不许往对手方向让价）。
        # 这里用 max 而不是"靠 pressure>0 保证"，因为 mult 的计算方式
        # 以后可能改（比如加了别的项），而"不许倒贴"是**不变量**，
        # 不应该依赖上游恰好算出 > 1 的数。
        out = max(base_offset, base_offset * mult)
        if out > base_offset:
            self.n_stretched += 1
            self.stretch_sum += out - base_offset
        return out

    def quote_offset_frac(self, state: MarketState, side: str) -> float:
        """挂载点之一：零智能交易者的报价幅度。"""
        base = super().quote_offset_frac(state, side)      # type: ignore[misc]
        return self.adjusted_offset(base, state, side)

    def quote_half_spread_frac(self, state: MarketState, side: str) -> float:
        """挂载点之二：做市商的**单侧**半价差。"""
        base = super().quote_half_spread_frac(state, side)  # type: ignore[misc]
        return self.adjusted_offset(base, state, side)

    def describe(self) -> dict:
        d = super().describe()                   # type: ignore[misc]
        d.update({
            "n_stretched": self.n_stretched,
            "mean_stretch": (self.stretch_sum / self.n_stretched
                             if self.n_stretched else 0.0),
            "flow_window": self.flow_window,
            "flow_sensitivity": self.flow_sensitivity,
        })
        return d
