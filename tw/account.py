"""保证金账户账本（A1）。

这是 A1 的**核心**：把"我有多少钱"变成"我能开多大仓、什么时候会被强平"。

为什么必须是**独立的一层**，不能塞进 ``Agent``
------------------------------------------------
``Agent`` 的 ``cash``/``inventory`` 是**现货**语义（一期的地基，
159 项测试跑在它上面）。永续合约要的是**保证金**语义，两者不变量不同：

==============  ==========================  ==============================
               现货（一期 Agent）            永续（本模块）
==============  ==========================  ==============================
核心不变量      ``available_cash >= 0``      ``equity >= 0``（cash 可为负）
"欠钱"          不允许                        允许（杠杆的本质）
仓位            持仓（有价证券）              合约仓位（有杠杆、有强平价）
风险度量        无                            保证金率 / 维持保证金
==============  ==========================  ==============================

把两者塞进一个类，就必须在每一处写 ``if 是永续: ... else: ...``，
而**漏掉一处就是静默错误**（"现货不变量被永续的数据破坏"）。
分层的代价是多一个对象，收益是两条不变量各自可独立验证。

四个公式（都按真实 OKX 口径）
==============================

**① 未实现盈亏 ``upl``**
   多头：``(mark - avg_px) * qty``；空头：``(avg_px - mark) * |qty|``。
   写成带符号的统一式：``upl = (mark - avg_px) * signed_qty``
   —— 空头 ``signed_qty < 0``，符号自动正确。

**② 维持保证金率 ``mmr`` 与保证金率**
   ``维持保证金 = notional * mmr_rate``（``mmr_rate`` 随仓位档位递增，
   本模块用常数 + 分档）。
   ``保证金率 = (权益 - 维持保证金) / notional``（OKX 的 ``mgnRatio`` 口径）。
   有些资料写成 ``权益 / 维持保证金``（= 保证金率的倒数），**两者不要混**：
   前者区间是 ``(-∞, +∞)`` 且「越小越危险」，后者「越大越危险」。
   本模块统一用前者并**显式标注口径**——这个坑本项目在别处已经踩过一次
   （见 README 测量纪律：同一个量有多份实现就一定会分叉）。

**③ 强平价 ``liq_px``**
   令 ``equity = 维持保证金`` 解出 ``mark``：
   ``cash + (mark - avg_px) * signed_qty = |signed_qty| * mark * mmr``
   多头（``signed_qty > 0``）：
   ``liq = (avg_px * q - cash) / (q * (1 - mmr))``
   空头（``signed_qty < 0``，令 ``q = |signed_qty|``）：
   ``liq = (avg_px * q + cash) / (q * (1 + mmr))``

   ⚠️ **多头与空头是两条不同的公式**。用一条公式加符号会算错
   （空头的强平价在**上方**，多头的在**下方**），而错误的强平价
   在正常行情里**永远不会被触发**，于是错得毫无症状。

**④ 开仓所需保证金**
   ``逐仓``：``notional / lever``（本仓独立）。
   ``全仓``：可用余额要覆盖 ``notional / lever``，
   但**可以动用账户里所有权益**（本模块按逐仓口径算，全仓只放宽可用余额）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

EPS = 1e-9


# ======================================================================
# 配置
# ======================================================================
@dataclass(slots=True)
class MarginConfig:
    """保证金参数。默认值按真实 OKX 的 BTC 永续档位取。

    ⚠️ ``mmr_rate`` 在真实 OKX 是**分档递增**的（仓位越大档位越高、
    维持保证金率越高）。本模块实现分档，因为**不分档会让大仓位免费**——
    而大仓位恰恰是最需要保证金的那些。
    """

    #: 维持保证金率的**分档**：(名义价值上限, mmr_rate)。
    #: 按真实 OKX BTC-USDT 永续的档位数量级设置（真实档位更密，
    #: 这里保留 4 档以体现"递增"这个性质）。
    mmr_tiers: tuple[tuple[float, float], ...] = (
        (50_000.0, 0.004),
        (250_000.0, 0.005),
        (1_000_000.0, 0.01),
        (float("inf"), 0.02),
    )
    #: 开仓手续费率（taker）。真实 OKX 的 VIP0 taker 是 0.05%，
    #: maker 是 0.02%——**相差 2.5 倍**，这正是 ``post_only`` 的价值所在。
    taker_fee: float = 0.0005
    maker_fee: float = 0.0002
    #: 最大杠杆。
    max_lever: float = 125.0

    def mmr_at(self, notional: float) -> float:
        """按名义价值取维持保证金率。"""
        n = abs(float(notional))
        for cap, rate in self.mmr_tiers:
            if n <= cap:
                return float(rate)
        return float(self.mmr_tiers[-1][1])

    def fee(self, notional: float, *, is_maker: bool) -> float:
        """手续费 = 名义价值 × 费率。**maker 与 taker 必须分开算**——

        本项目所有评估都用"扣掉成本后"的口径（见设计方案 §2
        「成本模型事先定义」），把两者合并会让 ``post_only`` 这个
        有用的机制在评估里看不出任何好处。
        """
        r = self.maker_fee if is_maker else self.taker_fee
        return abs(float(notional)) * r


# ======================================================================
# 持仓
# ======================================================================
@dataclass(slots=True)
class Position:
    """一个合约仓位（OKX ``posSide`` 粒度）。

    ``qty`` **带符号**：多正空负。用带符号量而不是 (方向, 数量) 二元组，
    是因为所有盈亏/强平公式都能写成统一的带符号形式，
    少一次 ``if`` 就少一处可能写反的地方。
    """

    inst_id: str
    pos_side: str = "net"
    qty: float = 0.0
    #: 入场均价（OKX ``avgPx``）。**加仓时按成交量加权更新**，不是覆盖。
    avg_px: float = 0.0
    #: 逐仓时分配到本仓的保证金。
    isolated_margin: float = 0.0
    #: 累计手续费（开+平），用于"净收益"口径。
    fees_paid: float = 0.0
    #: 累计已实现盈亏（平仓时累积，已扣手续费）。
    realized_pnl: float = 0.0

    def __post_init__(self) -> None:
        if abs(self.qty) < EPS and self.avg_px != 0.0:
            # 空仓还留着均价 = 状态不一致，会让下一次开仓的加权平均算错。
            raise ValueError("空仓的 avg_px 必须为 0")
        if abs(self.qty) > EPS and self.avg_px <= 0:
            raise ValueError(f"有仓位的 avg_px 必须为正，收到 {self.avg_px!r}")

    @property
    def is_flat(self) -> bool:
        return abs(self.qty) < EPS

    @property
    def abs_qty(self) -> float:
        return abs(self.qty)

    @property
    def is_long(self) -> bool:
        return self.qty > EPS

    # ------------------------------------------------------------------
    def notional(self, mark: float) -> float:
        """名义价值（按标记价）。"""
        return self.abs_qty * float(mark)

    def upl(self, mark: float) -> float:
        """未实现盈亏。用带符号量统一式（见模块文档 ①）。"""
        if self.is_flat:
            return 0.0
        return (float(mark) - self.avg_px) * self.qty

    # ------------------------------------------------------------------
    def apply_fill(
        self, *, side: str, qty: float, price: float, fee: float
    ) -> float:
        """把一笔成交应用到本仓位。返回**这笔成交实现的盈亏**（已扣手续费）。

        三种情形，必须分别处理：

        ================  ====================================================
        情形               处理
        ================  ====================================================
        同向（加仓）        ``avg_px`` 按量加权更新
        反向且不穿仓       部分平仓：结出对应比例的已实现盈亏，``avg_px`` 不变
        反向且穿仓         先平完，**剩余量按同一成交价反手开新仓**，
                           ``avg_px`` = 本次成交价
        ================  ====================================================

        ⚠️ 第三种（穿仓反手）是最容易漏的。只写前两种的实现在"空头被
        一笔大买单打穿"时会**静默丢掉**超出的那部分——仓位看起来对
        （因为被 clamp 到 0），但实际发生了反手开仓，账户少了一整个仓位。
        真实交易所允许反手（且这正是"多空双杀"的机制来源）。
        """
        signed = float(qty) if side == "buy" else -float(qty)
        price = float(price)
        self.fees_paid += float(fee)

        if self.is_flat:
            # 从空仓开仓
            self.qty = signed
            self.avg_px = price
            return -float(fee)

        if self.qty * signed > 0:
            # 同向加仓：加权平均
            new_qty = self.qty + signed
            self.avg_px = (self.avg_px * self.qty + price * signed) / new_qty
            self.qty = new_qty
            return -float(fee)

        # 反向：部分或全部平仓
        closing = min(abs(signed), self.abs_qty)
        # 多头平仓 = 卖出，盈亏 (price - avg)*q ；空头平仓反之。
        # 统一写成 (price - avg_px) * (closing 的符号 = 原仓位符号)
        realized = (price - self.avg_px) * math.copysign(closing, self.qty)
        self.realized_pnl += realized
        remainder = abs(signed) - closing

        if remainder <= EPS:
            # 平完（或恰好平完）
            self.qty += signed
            if abs(self.qty) < EPS:
                self.qty = 0.0
                self.avg_px = 0.0
                self.isolated_margin = 0.0
        else:
            # 穿仓反手：剩余量按本成交价开新仓
            self.qty = math.copysign(remainder, signed)
            self.avg_px = price
        return realized - float(fee)

    def liq_price(self, *, cash: float, mark: float, mmr_rate: float) -> float:
        """强平价（见模块文档 ③）。空仓时返回 ``nan``。"""
        if self.is_flat:
            return float("nan")
        q = self.abs_qty
        m = float(mmr_rate)
        c = float(cash)
        if self.is_long:
            # 多头：liq 在下方
            return (self.avg_px * q - c) / (q * (1.0 - m))
        # 空头：liq 在上方。分母是 (1 + m) —— 与多头不同，别抄错。
        return (self.avg_px * q + c) / (q * (1.0 + m))


# ======================================================================
# 账户
# ======================================================================
@dataclass(slots=True)
class MarginAccount:
    """保证金账户账本。**唯一**允许改动本账户的地方。

    与 ``Market`` 是唯一允许改动账户的地方，是同一条纪律的延续。
    """

    cfg: MarginConfig = field(default_factory=MarginConfig)
    #: 现金余额。永续语义下**可以为负**（= 向交易所借钱）。
    cash: float = 0.0
    #: 按 ``(inst_id, pos_side)`` 索引的仓位。
    positions: dict[tuple[str, str], Position] = field(default_factory=dict)
    #: 累计手续费。
    total_fees: float = 0.0
    #: 累计已实现盈亏（**已扣手续费**）。
    total_realized: float = 0.0
    #: 强平次数。
    n_liquidations: int = 0

    # ------------------------------------------------------------------
    def position(self, inst_id: str, pos_side: str = "net") -> Position:
        """取（必要时创建）某仓位。返回的是**账本内的对象**，可就地改。"""
        key = (str(inst_id), str(pos_side))
        p = self.positions.get(key)
        if p is None:
            p = Position(inst_id=str(inst_id), pos_side=str(pos_side))
            self.positions[key] = p
        return p

    def open_positions(self) -> list[Position]:
        return [p for p in self.positions.values() if not p.is_flat]

    # ------------------------------------------------------------------
    def equity(self, marks: dict[str, float]) -> float:
        """账户权益 = 现金 + 全部未实现盈亏。

        ``marks`` 是 ``{inst_id: 标记价}``。**缺价的仓位按 ``avg_px`` 估值**
        （= 假设 upl 为 0）而不是按 0 估值——按 0 会让权益瞬间塌陷，
        进而误判强平。真实交易所遇到无标记价时也是停用该合约的保证金计算。
        """
        eq = float(self.cash)
        for p in self.positions.values():
            if p.is_flat:
                continue
            m = marks.get(p.inst_id)
            if m is None or not (math.isfinite(m) and m > 0):
                m = p.avg_px
            eq += p.upl(float(m))
        return eq

    def total_upl(self, marks: dict[str, float]) -> float:
        tot = 0.0
        for p in self.positions.values():
            if p.is_flat:
                continue
            m = marks.get(p.inst_id, p.avg_px)
            tot += p.upl(float(m))
        return tot

    # ------------------------------------------------------------------
    def maintenance_margin(self, marks: dict[str, float]) -> float:
        """维持保证金合计（分档）。"""
        tot = 0.0
        for p in self.positions.values():
            if p.is_flat:
                continue
            m = marks.get(p.inst_id, p.avg_px)
            n = p.notional(float(m))
            tot += n * self.cfg.mmr_at(n)
        return tot

    def initial_margin(self, marks: dict[str, float], levers: dict[str, float]) -> float:
        """初始保证金合计 = Σ 名义价值 / 该合约杠杆。"""
        tot = 0.0
        for p in self.positions.values():
            if p.is_flat:
                continue
            m = marks.get(p.inst_id, p.avg_px)
            lv = max(1.0, float(levers.get(p.inst_id, 1.0)))
            tot += p.notional(float(m)) / lv
        return tot

    # ------------------------------------------------------------------
    def margin_ratio(self, marks: dict[str, float]) -> float:
        """保证金率，OKX ``mgnRatio`` 口径 = ``(权益 − 维持保证金) / 名义价值``。

        ⚠️ **口径必须写清楚**（见模块文档 ②）：另一种常见写法是
        ``权益 / 维持保证金``，它越大越安全；本式是**越小越危险**。
        两个口径混用会出现"报告说保证金率 0.4 很安全"而实际该强平的荒谬结论。
        分母为 0（无仓位）时返回 ``inf``（= 安全的极限）。

        ⚠️⚠️ **一个反直觉的性质，必须写在这里否则一定被误用**：
        在本口径下价格下跌会让**分子分母同时缩小**，而分母（``notional``）
        与价格同比下降得更快，所以 **ratio 会在价格下跌时变大**。
        ⇒ **不能拿它做跨时点的"安全性"比较**（"昨天 9.9 今天 90 所以更安全"
        是错的）。它只回答「以当前名义价值计，权益还剩几倍空间」。
        **真正的安全性判据是** :meth:`is_liquidated`，**真正的距离度量是**
        :meth:`liq_price`。要做跨时点比较就报 ``equity`` 与
        ``maintenance_margin`` 两个**绝对量**。
        """
        num = self.equity(marks) - self.maintenance_margin(marks)
        den = sum(
            p.notional(marks.get(p.inst_id, p.avg_px))
            for p in self.positions.values()
            if not p.is_flat
        )
        if den <= EPS:
            return float("inf")
        return float(num / den)

    def liq_price(self, inst_id: str, marks: dict[str, float],
                  pos_side: str = "net") -> float:
        """该仓位的强平价。

        ⚠️ 用**账户现金**而不是"本仓已分配保证金"：
        本模块的强平口径是「账户权益不足以支撑维持保证金」，
        而不是"逐仓保证金亏完"。两者在只有单一仓位时等价，
        多仓位时不同——本模块选前者（更接近全仓，也更保守）。
        """
        p = self.position(inst_id, pos_side)
        if p.is_flat:
            return float("nan")
        m = marks.get(inst_id, p.avg_px)
        n = p.notional(float(m))
        return p.liq_price(cash=self.cash, mark=float(m),
                           mmr_rate=self.cfg.mmr_at(n))

    # ------------------------------------------------------------------
    def is_liquidated(self, marks: dict[str, float]) -> bool:
        """是否应强平。判据：**权益 <= 维持保证金**（即 ``margin_ratio <= 0``）。

        用 ``<=`` 而不是 ``<``：恰好相等时已经无力维持，属于应该强平的一侧。
        这个边界在测试里被显式断言（用浮点 ``<`` 会因 1e-16 的残渣
        随机决定两侧，产生"偶发通过"的假绿灯）。
        """
        if not self.open_positions():
            return False
        return self.equity(marks) <= self.maintenance_margin(marks)

    def liquidate(self, inst_id: str, marks: dict[str, float],
                  pos_side: str = "net") -> dict:
        """强平一个仓位：按标记价平掉、扣手续费、记账。

        真实交易所的强平是**按破产价/标记价**成交并向保险基金结算；
        本模块按**标记价**平仓并只扣手续费（+ 计入 realized）。
        差异在于真实市场里强平本身会造成滑点——本项目在 A1 里
        **不做这个假设**（否则就是"用假设创造收益"），
        由 A2/A4 的执行层按实际盘口去模拟。
        """
        p = self.position(inst_id, pos_side)
        if p.is_flat:
            return {"liquidated": False, "qty": 0.0, "pnl": 0.0, "fee": 0.0}
        m = float(marks.get(inst_id, p.avg_px))
        qty = p.abs_qty
        side = "sell" if p.is_long else "buy"
        fee = self.cfg.fee(p.notional(m), is_maker=False)
        pnl = p.apply_fill(side=side, qty=qty, price=m, fee=fee)
        self.cash += pnl
        self.total_fees += fee
        self.total_realized += pnl
        self.n_liquidations += 1
        return {
            "liquidated": True,
            "inst_id": inst_id,
            "pos_side": pos_side,
            "side": side,
            "qty": qty,
            "price": m,
            "fee": fee,
            "pnl": pnl,
        }

    # ------------------------------------------------------------------
    def apply_fill(
        self,
        *,
        inst_id: str,
        side: str,
        qty: float,
        price: float,
        is_maker: bool,
        pos_side: str = "net",
        lever: float = 1.0,
    ) -> dict:
        """把一笔成交记到账本上。**唯一的记账入口。**

        两条记账路径必须都在这里：
          1. 仓位的 ``apply_fill``（更新 ``qty``/``avg_px``/``realized``）；
          2. 现金 = 已实现盈亏 − 手续费。

        ⚠️ 为什么第 2 条**只加盈亏**、不加名义价值：永续是**合约**，
        开仓不动本金（本金作为保证金被占用），只有平仓才结算盈亏。
        若照抄现货的"扣 notional"，会发现开仓一手 BTC 就要求现金
        等于全额名义价值 —— 杠杆就完全不存在了。
        """
        p = self.position(inst_id, pos_side)
        fee = self.cfg.fee(abs(float(qty) * float(price)), is_maker=is_maker)
        pnl = p.apply_fill(side=side, qty=qty, price=price, fee=fee)
        self.cash += pnl
        self.total_fees += fee
        self.total_realized += pnl
        # 逐仓：开仓时须把保证金划入本仓（这里只记录，不另扣现金——
        # 本模块的强平口径是账户级，见 liq_price 的说明）。
        if p.isolated_margin > 0 or (
            abs(p.qty) > EPS and self.equity({inst_id: price}) > 0
        ):
            p.isolated_margin = p.notional(price) / max(1.0, float(lever))
        return {"fee": fee, "pnl": pnl, "qty": p.qty, "avg_px": p.avg_px}

    # ------------------------------------------------------------------
    def can_open(
        self,
        *,
        inst_id: str,
        side: str,
        qty: float,
        price: float,
        lever: float,
        marks: dict[str, float],
        pos_side: str = "net",
    ) -> tuple[bool, str]:
        """能否开这张仓。返回 ``(能否, 原因)``。

        判据：**开仓后的模拟权益 >= 开仓后的维持保证金**。

        ⚠️ 为什么用"开仓**后**"而不是"开仓前"：只看开仓前，一个权益
        只剩 1 元的账户可以开出 100 元的仓位（因为开仓瞬间还没亏），
        下一秒立刻强平 —— 实际等于**免费获得一次几秒的杠杆**。
        真实交易所要求的是"新仓的初始保证金要能在现有权益里放下"，
        本式等价于它（初始保证金 >= 维持保证金）。
        """
        m = float(marks.get(inst_id, price))
        signed_new = float(qty) if side == "buy" else -float(qty)
        p = self.position(inst_id, pos_side)
        # 模拟成交后
        q_after = p.qty + signed_new
        if p.is_flat:
            avg_after = float(price)
        elif p.qty * signed_new > 0:
            avg_after = (p.avg_px * p.qty + float(price) * signed_new) / q_after
        else:
            remain = abs(signed_new) - min(abs(signed_new), p.abs_qty)
            if remain <= EPS:
                avg_after = p.avg_px if abs(q_after) > EPS else 0.0
            else:
                avg_after = float(price)
        upl = (m - avg_after) * q_after if abs(q_after) > EPS else 0.0
        eq_after = self.cash + self.total_upl(marks) - p.upl(m) + upl
        # 手续费也要吃掉权益（真实交易所开仓即扣）
        fee = self.cfg.fee(abs(float(qty) * float(price)), is_maker=False)
        eq_after -= fee
        notional_after = abs(q_after) * m
        mm_after = 0.0
        for pp in self.positions.values():
            if pp.is_flat or pp is p:
                continue
            mm_after += pp.notional(marks.get(pp.inst_id, pp.avg_px)) * self.cfg.mmr_at(
                pp.notional(marks.get(pp.inst_id, pp.avg_px))
            )
        mm_after += notional_after * self.cfg.mmr_at(notional_after)
        if eq_after <= mm_after:
            return False, "TW-1005"
        if lever > self.cfg.max_lever:
            return False, "TW-1009"
        return True, ""

    def summary(self, marks: dict[str, float], levers: dict[str, float] | None = None) -> dict:
        """账户摘要。**口径全部显式标注**，不给出可被误读的裸数字。"""
        lv = levers or {}
        eq = self.equity(marks)
        mm = self.maintenance_margin(marks)
        im = self.initial_margin(marks, lv)
        return {
            "cash": float(self.cash),
            "equity": float(eq),
            "upl": float(self.total_upl(marks)),
            "maintenance_margin": float(mm),
            "initial_margin": float(im),
            #: OKX mgnRatio 口径：(权益 − 维持保证金) / 名义价值。**越小越危险**。
            "margin_ratio": float(self.margin_ratio(marks)),
            "n_positions": len(self.open_positions()),
            "total_fees": float(self.total_fees),
            "total_realized": float(self.total_realized),
            "n_liquidations": int(self.n_liquidations),
        }
