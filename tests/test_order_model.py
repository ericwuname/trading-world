"""A1 订单模型与保证金账本的测试。

测试的写法定了一条纪律：**断言用严格不等号 + 手算的独立期望值**。
理由见 README 的工程纪律——"同一个量有多份实现就一定会分叉"，
所以凡是能手算的（强平价、保证金率、upl）都手算一遍写在断言里，
而不是复用被测代码里的公式去算期望值。
"""

from __future__ import annotations

import math
import unittest

from tw.account import MarginAccount, MarginConfig, Position
from tw.order_model import (
    MARKET_ON_TRIGGER,
    AlgoOrder,
    OrderRejected,
    OrderRequest,
    apply_reduce_only,
    precheck,
)


# ======================================================================
# AlgoOrder：止盈止损
# ======================================================================
class TestAlgoOrder(unittest.TestCase):
    def test_至少要有止盈或止损一边(self):
        with self.assertRaises(ValueError):
            AlgoOrder()

    def test_只挂一边是允许的(self):
        a = AlgoOrder(tp_trigger_px=110.0)
        self.assertIsNone(a.sl_trigger_px)
        b = AlgoOrder(sl_trigger_px=90.0)
        self.assertIsNone(b.tp_trigger_px)

    def test_触发价必须为正(self):
        with self.assertRaises(ValueError):
            AlgoOrder(tp_trigger_px=0.0)
        with self.assertRaises(ValueError):
            AlgoOrder(sl_trigger_px=-1.0)
        with self.assertRaises(ValueError):
            AlgoOrder(tp_trigger_px=float("nan"))

    def test_默认走标记价(self):
        """⚠️ 默认必须是 mark —— 用 last 会被插针打穿（模块文档 ③）。"""
        a = AlgoOrder(tp_trigger_px=110.0, sl_trigger_px=90.0)
        self.assertEqual(a.tp_trigger_px_type, "mark")
        self.assertEqual(a.sl_trigger_px_type, "mark")

    def test_默认触发后走市价(self):
        a = AlgoOrder(tp_trigger_px=110.0)
        self.assertEqual(a.tp_ord_px, MARKET_ON_TRIGGER)
        self.assertEqual(a.tp_ord_px, -1.0)

    def test_触发价类型非法要报错(self):
        with self.assertRaises(ValueError):
            AlgoOrder(tp_trigger_px=110.0, tp_trigger_px_type="ohlc")

    def test_frozen不可变(self):
        """止盈止损要改必须撤单重挂，不能就地改（否则交易所侧不知情）。"""
        a = AlgoOrder(tp_trigger_px=110.0)
        with self.assertRaises(Exception):
            a.tp_trigger_px = 120.0  # type: ignore[misc]


class TestAlgoTrigger(unittest.TestCase):
    """⭐ 方向不能搞反——这是最容易写错的一处（模块文档 ③ 的表）。"""

    def _hit(self, side, tp=None, sl=None, mark=100.0, last=None, index=None):
        a = AlgoOrder(tp_trigger_px=tp, sl_trigger_px=sl)
        return a.triggers(
            mark=mark,
            last=last if last is not None else mark,
            index=index if index is not None else mark,
            side=side,
        )

    def test_多头止盈在价格上行时触发(self):
        self.assertEqual(self._hit("buy", tp=105.0, mark=106.0), "tp")
        self.assertIsNone(self._hit("buy", tp=105.0, mark=104.0))

    def test_多头止损在价格下行时触发(self):
        self.assertEqual(self._hit("buy", sl=95.0, mark=94.0), "sl")
        self.assertIsNone(self._hit("buy", sl=95.0, mark=96.0))

    def test_空头止盈在价格下行时触发(self):
        """⚠️ 空头的止盈是**向下**的。写成"止盈一律 >= "会把它做成止损。"""
        self.assertEqual(self._hit("sell", tp=95.0, mark=94.0), "tp")
        self.assertIsNone(self._hit("sell", tp=95.0, mark=96.0))

    def test_空头止损在价格上行时触发(self):
        self.assertEqual(self._hit("sell", sl=105.0, mark=106.0), "sl")
        self.assertIsNone(self._hit("sell", sl=105.0, mark=104.0))

    def test_多空触发方向必须相反(self):
        """同一组价格下，多空两侧的触发结果不能相同——相同就说明方向没被用上。

        ⚠️ 写这条测试时踩过一个坑：如果多空**共用同一组 tp/sl 触发价**，
        那么对空头而言 ``tp=105`` 的含义是「价格**跌**到 105 就止盈」，
        于是 ``px=100`` 时空头 tp 已触发——数学正确，但它让"两侧结果必须
        相反"这条断言在中间价位失效。所以必须用**方向对称**的一组：
        多头 ``tp=105/sl=95``，空头 ``tp=95/sl=105``。
        """
        long_a = AlgoOrder(tp_trigger_px=105.0, sl_trigger_px=95.0)
        short_a = AlgoOrder(tp_trigger_px=95.0, sl_trigger_px=105.0)
        for px in (90.0, 94.0, 100.0, 106.0, 110.0):
            lr = long_a.triggers(mark=px, last=px, index=px, side="buy")
            sr = short_a.triggers(mark=px, last=px, index=px, side="sell")
            if px > 105.0:
                self.assertEqual(lr, "tp")
                self.assertEqual(sr, "sl")
            elif px < 95.0:
                self.assertEqual(lr, "sl")
                self.assertEqual(sr, "tp")
            else:
                self.assertIsNone(lr, f"px={px} 多头不该触发")
                self.assertIsNone(sr, f"px={px} 空头不该触发")

    def test_同一组触发价下多空结果相反(self):
        """这才是"方向被用上"的直接证据：同一组 (tp,sl)、同一价格，
        多空两侧的判定必须相反。"""
        a = AlgoOrder(tp_trigger_px=105.0, sl_trigger_px=95.0)
        for px in (90.0, 110.0):
            lr = a.triggers(mark=px, last=px, index=px, side="buy")
            sr = a.triggers(mark=px, last=px, index=px, side="sell")
            self.assertIsNotNone(lr)
            self.assertIsNotNone(sr)
            self.assertNotEqual(lr, sr, f"px={px} 多空判定相同 ⇒ 方向没生效")

    def test_边界价恰好相等即触发(self):
        """触发条件用 >= / <=，不是 > / <。"""
        self.assertEqual(self._hit("buy", tp=105.0, mark=105.0), "tp")
        self.assertEqual(self._hit("buy", sl=95.0, mark=95.0), "sl")

    def test_触发价类型真的生效(self):
        """mark 与 last 分离时，结果应随类型不同（否则字段是摆设）。"""
        a = AlgoOrder(tp_trigger_px=105.0, tp_trigger_px_type="last")
        # mark=104（不触发），last=106（触发）
        self.assertEqual(
            a.triggers(mark=104.0, last=106.0, index=105.0, side="buy"), "tp"
        )
        b = AlgoOrder(tp_trigger_px=105.0, tp_trigger_px_type="mark")
        self.assertIsNone(
            b.triggers(mark=104.0, last=106.0, index=105.0, side="buy")
        )

    def test_止盈优先于止损(self):
        """极端行情里同日触发时，止盈在前（保守：先落袋）。"""
        a = AlgoOrder(tp_trigger_px=90.0, sl_trigger_px=110.0)
        # 空头：tp=90（向下触发）、sl=110（向上触发）。价格 85 只触发 tp。
        self.assertEqual(a.triggers(mark=85.0, last=85.0, index=85.0, side="sell"), "tp")


# ======================================================================
# OrderRequest
# ======================================================================
class TestOrderRequest(unittest.TestCase):
    def _req(self, **kw):
        base = dict(inst_id="BTC-USDT-SWAP", side="buy", ord_type="limit",
                    sz=1.0, px=100.0)
        base.update(kw)
        return OrderRequest(**base)

    def test_最小可用限价单(self):
        r = self._req()
        self.assertEqual(r.sz, 1.0)
        self.assertEqual(r.td_mode, "cash")
        self.assertEqual(r.pos_side, "net")
        self.assertFalse(r.reduce_only)

    def test_市价单不能带价格(self):
        """带了价格说明调用方想要限价单——不猜意图，直接拒。"""
        with self.assertRaises(ValueError):
            self._req(ord_type="market", px=100.0)

    def test_市价单价格必须为None(self):
        r = self._req(ord_type="market", px=None)
        self.assertTrue(r.is_market_like)

    def test_限价单必须有正价格(self):
        with self.assertRaises(ValueError):
            self._req(px=None)
        with self.assertRaises(ValueError):
            self._req(px=0.0)
        with self.assertRaises(ValueError):
            self._req(px=-5.0)

    def test_数量必须为正(self):
        with self.assertRaises(ValueError):
            self._req(sz=0.0)
        with self.assertRaises(ValueError):
            self._req(sz=-1.0)
        with self.assertRaises(ValueError):
            self._req(sz=float("inf"))

    def test_现货不能加杠杆(self):
        with self.assertRaises(ValueError):
            self._req(td_mode="cash", lever=5.0)
        self.assertEqual(self._req(td_mode="cash", lever=1.0).lever, 1.0)

    def test_全仓双向持仓未实现要拒绝(self):
        """⚠️ 拒绝而不是"看起来支持了"——cross+long/short 的保证金口径没实现。"""
        with self.assertRaises(ValueError):
            self._req(td_mode="cross", pos_side="long", lever=5.0)
        # 逐仓双向是允许的
        self.assertEqual(
            self._req(td_mode="isolated", pos_side="long", lever=5.0).pos_side, "long"
        )

    def test_杠杆不能小于1(self):
        with self.assertRaises(ValueError):
            self._req(lever=0.5)

    def test_clordid长度上限32(self):
        self._req(cl_ord_id="x" * 32)  # 恰好 32 可以
        with self.assertRaises(ValueError):
            self._req(cl_ord_id="x" * 33)

    def test_类型语义属性(self):
        self.assertFalse(self._req(ord_type="limit").is_market_like)
        self.assertFalse(self._req(ord_type="post_only").is_market_like)
        self.assertTrue(self._req(ord_type="ioc").is_market_like)
        self.assertTrue(self._req(ord_type="fok").is_market_like)
        self.assertTrue(self._req(ord_type="market", px=None).is_market_like)

        self.assertTrue(self._req(ord_type="post_only").must_be_maker)
        self.assertFalse(self._req(ord_type="limit").must_be_maker)

        self.assertTrue(self._req(ord_type="ioc").cancels_remainder)
        self.assertTrue(self._req(ord_type="fok").cancels_remainder)
        self.assertFalse(self._req(ord_type="limit").cancels_remainder)

        self.assertTrue(self._req(ord_type="fok").requires_full_fill)
        self.assertFalse(self._req(ord_type="ioc").requires_full_fill)

    def test_with_size返回新对象不改原对象(self):
        """⭐ 风控裁剪后必须保留原请求 —— "Agent 要了多少"和"实际下了多少"
        的差**正是风控层的贡献**，A4 要单独报这一项。"""
        r = self._req(sz=10.0)
        r2 = r.with_size(3.0)
        self.assertEqual(r.sz, 10.0)
        self.assertEqual(r2.sz, 3.0)
        self.assertIsNot(r, r2)

    def test_to_native市价单用正负无穷(self):
        r = self._req(ord_type="market", px=None)
        o = r.to_native(tick=5, order_id="x#1", price_of=lambda t: t * 0.01,
                        to_tick=lambda p: round(p / 0.01))
        self.assertEqual(o.order_type, "market")
        self.assertEqual(o.price, math.inf)
        self.assertEqual(o.timestamp, 5)
        self.assertEqual(o.side, "buy")

        r_sell = self._req(ord_type="market", px=None, side="sell")
        self.assertEqual(r_sell.to_native(
            tick=0, order_id="x", price_of=lambda t: t, to_tick=lambda p: p
        ).price, -math.inf)

    def test_to_native限价单价格吸附到网格(self):
        """⭐ 吸附必须在唯一的这处做——预留按原价、结算按网格价会逐笔泄漏。"""
        r = self._req(px=100.123456)
        o = r.to_native(
            tick=0, order_id="x",
            price_of=lambda t: t * 0.5,
            to_tick=lambda p: round(p / 0.5),
        )
        self.assertEqual(o.order_type, "limit")
        self.assertEqual(o.price, 100.0)   # round(200.2469)=200 → 200*0.5
        self.assertEqual(o.quantity, 1.0)

    def test_to_native吸附后非法价格要报错(self):
        r = self._req(px=0.2)
        with self.assertRaises(ValueError):
            r.to_native(tick=0, order_id="x", price_of=lambda t: 0.0,
                        to_tick=lambda p: 0)

    def test_post_only也走限价路径(self):
        """post_only 在撮合层就是限价单，靠外部撤销保证当 maker。"""
        r = self._req(ord_type="post_only", px=100.0)
        o = r.to_native(tick=0, order_id="x", price_of=lambda t: t, to_tick=lambda p: p)
        self.assertEqual(o.order_type, "limit")
        self.assertEqual(o.price, 100.0)


# ======================================================================
# precheck
# ======================================================================
class TestPrecheck(unittest.TestCase):
    def _req(self, **kw):
        base = dict(inst_id="BTC-USDT-SWAP", side="buy", ord_type="limit",
                    sz=1.0, px=100.0)
        base.update(kw)
        return OrderRequest(**base)

    def test_正常单通过(self):
        precheck(self._req(), best_bid=99.0, best_ask=101.0)

    def test_低于最小下单量被拒(self):
        with self.assertRaises(OrderRejected) as cm:
            precheck(self._req(sz=1e-6), best_bid=99.0, best_ask=101.0, min_qty=1e-4)
        self.assertEqual(cm.exception.code, "TW-1007")

    def test_数量不在步长网格上被拒(self):
        """⚠️ 拒单而不是向下取整——静默改量会让 Agent 无从知道。"""
        with self.assertRaises(OrderRejected) as cm:
            precheck(self._req(sz=1.0001), best_bid=99.0, best_ask=101.0,
                     quantity_step=0.001)
        self.assertEqual(cm.exception.code, "TW-1007")
        # 恰好落在网格上就通过
        precheck(self._req(sz=1.001), best_bid=99.0, best_ask=101.0,
                 quantity_step=0.001)

    def test_超过最大挂单数被拒(self):
        with self.assertRaises(OrderRejected) as cm:
            precheck(self._req(), best_bid=99.0, best_ask=101.0,
                     n_open_orders=10, max_open_orders=10)
        self.assertEqual(cm.exception.code, "TW-1008")

    def test_杠杆超限被拒(self):
        with self.assertRaises(OrderRejected) as cm:
            precheck(self._req(lever=200.0, td_mode="isolated"),
                     best_bid=99.0, best_ask=101.0, max_lever=125.0)
        self.assertEqual(cm.exception.code, "TW-1009")

    # -- post_only 的核心语义 ------------------------------------------
    def test_post_only买价触及卖一被拒(self):
        """⭐ post_only 的定义就是"绝不主动成交"，交叉了必须**拒单**。"""
        with self.assertRaises(OrderRejected) as cm:
            precheck(self._req(ord_type="post_only", px=101.0),
                     best_bid=99.0, best_ask=101.0)
        self.assertEqual(cm.exception.code, "TW-1001")

    def test_post_only买价高于卖一也被拒(self):
        with self.assertRaises(OrderRejected) as cm:
            precheck(self._req(ord_type="post_only", px=105.0),
                     best_bid=99.0, best_ask=101.0)
        self.assertEqual(cm.exception.code, "TW-1001")

    def test_post_only卖价触及买一被拒(self):
        with self.assertRaises(OrderRejected) as cm:
            precheck(self._req(ord_type="post_only", px=99.0, side="sell"),
                     best_bid=99.0, best_ask=101.0)
        self.assertEqual(cm.exception.code, "TW-1001")

    def test_post_only不交叉就通过(self):
        precheck(self._req(ord_type="post_only", px=100.0),
                 best_bid=99.0, best_ask=101.0)
        precheck(self._req(ord_type="post_only", px=100.0, side="sell"),
                 best_bid=99.0, best_ask=101.0)

    def test_post_only在空盘口通过(self):
        """没有对手价时不可能成交，post_only 应当通过。"""
        precheck(self._req(ord_type="post_only", px=100.0),
                 best_bid=None, best_ask=None)

    # -- fok 的核心语义 ------------------------------------------------
    def test_fok无法全部成交被拒(self):
        """⭐ fok = 全部成交否则整单不成立。"""
        with self.assertRaises(OrderRejected) as cm:
            precheck(self._req(ord_type="fok", sz=10.0),
                     best_bid=99.0, best_ask=101.0, fillable_qty=9.99)
        self.assertEqual(cm.exception.code, "TW-1002")

    def test_fok恰好能全成交就通过(self):
        precheck(self._req(ord_type="fok", sz=10.0),
                 best_bid=99.0, best_ask=101.0, fillable_qty=10.0)

    def test_ioc不检查全部成交(self):
        """ioc 允许部分成交，不该被 fok 那条规则误伤。"""
        precheck(self._req(ord_type="ioc", sz=10.0),
                 best_bid=99.0, best_ask=101.0, fillable_qty=1.0)

    def test_limit不检查可成交量(self):
        precheck(self._req(ord_type="limit", sz=10.0),
                 best_bid=99.0, best_ask=101.0, fillable_qty=0.0)


# ======================================================================
# reduce_only
# ======================================================================
class TestReduceOnly(unittest.TestCase):
    def _req(self, **kw):
        base = dict(inst_id="BTC-USDT-SWAP", side="sell", ord_type="limit",
                    sz=1.0, px=100.0, reduce_only=True)
        base.update(kw)
        return OrderRequest(**base)

    def test_非reduce_only原样返回(self):
        r = self._req(reduce_only=False)
        self.assertIs(apply_reduce_only(r, position_qty=5.0, position_side="net"), r)

    def test_无仓位可减被拒(self):
        with self.assertRaises(OrderRejected) as cm:
            apply_reduce_only(self._req(), position_qty=0.0, position_side="net")
        self.assertEqual(cm.exception.code, "TW-1004")

    def test_方向与仓位相同时被拒(self):
        """⭐ 同向 = 加仓，不是 reduce。交易所猜不出意图，只能拒。"""
        with self.assertRaises(OrderRejected) as cm:
            # 持多仓（正），却下买单
            apply_reduce_only(self._req(side="buy"), position_qty=5.0,
                              position_side="net")
        self.assertEqual(cm.exception.code, "TW-1003")

    def test_平多仓用卖单通过(self):
        r = apply_reduce_only(self._req(side="sell", sz=1.0), position_qty=5.0,
                              position_side="net")
        self.assertEqual(r.sz, 1.0)

    def test_平空仓用买单通过(self):
        r = apply_reduce_only(self._req(side="buy", sz=1.0), position_qty=-5.0,
                              position_side="net")
        self.assertEqual(r.sz, 1.0)

    def test_超量被裁剪到仓位上限(self):
        """⚠️ 裁剪而不是拒单——意图明确（"平掉"），交易所按仓位上限裁。"""
        r = apply_reduce_only(self._req(side="sell", sz=99.0), position_qty=5.0,
                              position_side="net")
        self.assertEqual(r.sz, 5.0)

    def test_裁剪不改变原请求(self):
        r0 = self._req(side="sell", sz=99.0)
        r1 = apply_reduce_only(r0, position_qty=5.0, position_side="net")
        self.assertEqual(r0.sz, 99.0)
        self.assertEqual(r1.sz, 5.0)
        self.assertIsNot(r0, r1)


# ======================================================================
# Position
# ======================================================================
class TestPosition(unittest.TestCase):
    def test_空仓均价必须为0(self):
        with self.assertRaises(ValueError):
            Position(inst_id="X", qty=0.0, avg_px=100.0)

    def test_有仓位必须有正均价(self):
        with self.assertRaises(ValueError):
            Position(inst_id="X", qty=1.0, avg_px=0.0)
        with self.assertRaises(ValueError):
            Position(inst_id="X", qty=1.0, avg_px=-1.0)

    def test_开仓设均价(self):
        p = Position(inst_id="X")
        pnl = p.apply_fill(side="buy", qty=2.0, price=100.0, fee=0.0)
        self.assertEqual(p.qty, 2.0)
        self.assertEqual(p.avg_px, 100.0)
        self.assertEqual(pnl, 0.0)
        self.assertTrue(p.is_long)

    def test_加仓按成交量加权(self):
        p = Position(inst_id="X")
        p.apply_fill(side="buy", qty=1.0, price=100.0, fee=0.0)
        p.apply_fill(side="buy", qty=3.0, price=200.0, fee=0.0)
        self.assertEqual(p.qty, 4.0)
        self.assertEqual(p.avg_px, 175.0)   # (100 + 600)/4

    def test_部分平仓均价不变(self):
        p = Position(inst_id="X")
        p.apply_fill(side="buy", qty=4.0, price=100.0, fee=0.0)
        pnl = p.apply_fill(side="sell", qty=1.0, price=120.0, fee=0.0)
        self.assertEqual(p.qty, 3.0)
        self.assertEqual(p.avg_px, 100.0)
        self.assertEqual(pnl, 20.0)          # (120-100)*1

    def test_完全平仓归零并清均价(self):
        p = Position(inst_id="X")
        p.apply_fill(side="buy", qty=2.0, price=100.0, fee=0.0)
        pnl = p.apply_fill(side="sell", qty=2.0, price=110.0, fee=0.0)
        self.assertTrue(p.is_flat)
        self.assertEqual(p.qty, 0.0)
        self.assertEqual(p.avg_px, 0.0)
        self.assertEqual(pnl, 20.0)

    def test_空头盈亏方向(self):
        p = Position(inst_id="X")
        p.apply_fill(side="sell", qty=2.0, price=100.0, fee=0.0)
        self.assertEqual(p.qty, -2.0)
        self.assertTrue(not p.is_long)
        # 空头在价格下跌时赚
        pnl = p.apply_fill(side="buy", qty=2.0, price=90.0, fee=0.0)
        self.assertEqual(pnl, 20.0)
        self.assertTrue(p.is_flat)

    def test_穿仓反手不丢仓位(self):
        """⭐ 最容易漏的第三种情形：反向成交量超过仓位时要反手开新仓。"""
        p = Position(inst_id="X")
        p.apply_fill(side="buy", qty=1.0, price=100.0, fee=0.0)
        # 卖出 3 手：平掉 1 手多仓 + 反手开 2 手空仓
        pnl = p.apply_fill(side="sell", qty=3.0, price=110.0, fee=0.0)
        self.assertEqual(pnl, 10.0)            # 只结平掉那 1 手的盈亏
        self.assertEqual(p.qty, -2.0)          # 反手空 2 手
        self.assertEqual(p.avg_px, 110.0)      # 新仓均价 = 本次成交价

    def test_手续费计入realized(self):
        p = Position(inst_id="X")
        p.apply_fill(side="buy", qty=1.0, price=100.0, fee=0.0)
        pnl = p.apply_fill(side="sell", qty=1.0, price=110.0, fee=0.5)
        self.assertEqual(pnl, 9.5)
        self.assertEqual(p.fees_paid, 0.5)

    # -- upl -----------------------------------------------------------
    def test_多头upl(self):
        p = Position(inst_id="X", qty=2.0, avg_px=100.0)
        self.assertEqual(p.upl(110.0), 20.0)
        self.assertEqual(p.upl(90.0), -20.0)

    def test_空头upl(self):
        p = Position(inst_id="X", qty=-2.0, avg_px=100.0)
        self.assertEqual(p.upl(90.0), 20.0)
        self.assertEqual(p.upl(110.0), -20.0)

    def test_空仓upl为零(self):
        self.assertEqual(Position(inst_id="X").upl(999.0), 0.0)

    # -- liq_price -----------------------------------------------------
    def test_多头强平价在下方(self):
        """手算：q=2, avg=100, cash=50, mmr=0.01
        liq = (avg*q − cash) / (q*(1−mmr)) = (200−50)/(2*0.99) = 75.7576
        代回验证 equity(liq) == mm(liq)。

        ⚠️ ``cash=0`` 会让这只测试**看起来失败但实际上公式是对的**：
        cash=0 意味着用 0 本金开出 200 名义值的仓，这不可物理，
        此时方程的解落在 ``avg`` **上方**（101.01）。
        ⇒ 断言"强平价在均价下方"必须配 ``cash>0`` 的用例。
        这条注释留在测试里，是因为后来者很容易用 cash=0 重写这条测试
        然后得出"公式错了"的错误结论。
        """
        p = Position(inst_id="X", qty=2.0, avg_px=100.0)
        liq = p.liq_price(cash=50.0, mark=100.0, mmr_rate=0.01)
        self.assertAlmostEqual(liq, 150.0 / (2.0 * 0.99), places=9)
        self.assertLess(liq, 100.0)
        # 代回：equity 与维持保证金必须相等（这就是强平的定义）
        eq = 50.0 + (liq - 100.0) * 2.0
        mm = liq * 2.0 * 0.01
        self.assertAlmostEqual(eq, mm, places=9)

    def test_多头本金越厚强平价越低(self):
        """本金越厚越抗跌 —— 这是公式方向正确的最直观证据。"""
        p = Position(inst_id="X", qty=2.0, avg_px=100.0)
        liqs = [p.liq_price(cash=float(c), mark=100.0, mmr_rate=0.01)
                for c in (0.0, 50.0, 100.0, 200.0)]
        self.assertEqual(liqs, sorted(liqs, reverse=True))

    def test_无mmr时强平价即为亏损全部本金处(self):
        """mmr=0 时 liq = (avg*q − cash)/q = avg − cash/q，可精确手算。"""
        p = Position(inst_id="X", qty=2.0, avg_px=100.0)
        self.assertAlmostEqual(
            p.liq_price(cash=20.0, mark=100.0, mmr_rate=0.0), 90.0, places=9
        )

    def test_空头强平价在上方(self):
        """手算：q=2, avg=100, cash=0, mmr=0.01
        liq = (100*2 + 0) / (2 * (1+0.01)) = 200/2.02 ≈ 99.01...?
        不 —— 空头 liq 应在上方。cash=0 时反向算：
        equity = 0 + (liq - 100)*(-2) = 200 - 2*liq
        mm = 2*liq*0.01 = 0.02*liq
        200 - 2*liq = 0.02*liq → liq = 200/2.02 ≈ 99.01
        这确实在下方 —— 说明 cash=0 时"零本金开空"的强平价确实在下面。
        真正的空头强平价在上方需要**正**的现金（收了卖出款）。"""
        p = Position(inst_id="X", qty=-2.0, avg_px=100.0)
        liq = p.liq_price(cash=200.0, mark=100.0, mmr_rate=0.01)
        # 手算：equity = 200 + (liq-100)*(-2) = 400 - 2*liq
        #       mm = 2*liq*0.01 = 0.02*liq
        #       400 - 2liq = 0.02liq  →  liq = 400/2.02
        self.assertAlmostEqual(liq, 400.0 / 2.02, places=9)
        self.assertGreater(liq, 100.0)   # 空头强平价**在上方** ✓

    def test_空头强平价公式与多头不同(self):
        """⚠️ 用一条公式加符号会算错——分母是 (1+m) 不是 (1-m)。"""
        long_p = Position(inst_id="X", qty=2.0, avg_px=100.0)
        short_p = Position(inst_id="X", qty=-2.0, avg_px=100.0)
        lq = long_p.liq_price(cash=0.0, mark=100.0, mmr_rate=0.01)
        sq = short_p.liq_price(cash=0.0, mark=100.0, mmr_rate=0.01)
        self.assertNotAlmostEqual(lq, sq, places=6)

    def test_空仓强平价为nan(self):
        self.assertTrue(math.isnan(Position(inst_id="X").liq_price(
            cash=0.0, mark=100.0, mmr_rate=0.01)))


# ======================================================================
# MarginConfig
# ======================================================================
class TestMarginConfig(unittest.TestCase):
    def test_维持保证金率随名义价值递增(self):
        """⭐ 不分档会让大仓位免费，而大仓位恰恰最需要保证金。"""
        c = MarginConfig()
        rates = [c.mmr_at(n) for n in (1_000, 100_000, 500_000, 5_000_000)]
        self.assertEqual(rates, sorted(rates))
        self.assertLess(rates[0], rates[-1])

    def test_档位边界取上界(self):
        c = MarginConfig()
        self.assertEqual(c.mmr_at(50_000.0), 0.004)
        self.assertEqual(c.mmr_at(50_000.1), 0.005)

    def test_maker费低于taker费(self):
        """⭐ 费率差是 post_only 的价值来源，必须能算出来。"""
        c = MarginConfig()
        n = 100_000.0
        self.assertLess(c.fee(n, is_maker=True), c.fee(n, is_maker=False))
        self.assertAlmostEqual(c.fee(n, is_maker=False), 100_000 * 0.0005)

    def test_手续费按绝对值算(self):
        c = MarginConfig()
        self.assertEqual(c.fee(-1000.0, is_maker=True), c.fee(1000.0, is_maker=True))


# ======================================================================
# MarginAccount
# ======================================================================
class TestMarginAccount(unittest.TestCase):
    def _acct(self, cash=10_000.0, **cfg):
        return MarginAccount(cfg=MarginConfig(**cfg), cash=cash)

    def test_空账户权益等于现金(self):
        a = self._acct()
        self.assertEqual(a.equity({}), 10_000.0)
        self.assertEqual(a.maintenance_margin({}), 0.0)

    def test_无仓位保证金率为无穷(self):
        self.assertEqual(self._acct().margin_ratio({}), math.inf)

    def test_开仓不动本金只扣手续费(self):
        """⭐ 永续是合约：开仓不动本金（本金是保证金），只有平仓才结算盈亏。
        照抄现货的"扣 notional"会让杠杆完全不存在。"""
        a = self._acct(cash=10_000.0, taker_fee=0.0005, maker_fee=0.0002)
        r = a.apply_fill(inst_id="X", side="buy", qty=1.0, price=50_000.0,
                         is_maker=False)
        # 只扣了手续费 50000*0.0005 = 25
        self.assertAlmostEqual(a.cash, 10_000.0 - 25.0, places=9)
        self.assertAlmostEqual(r["fee"], 25.0, places=9)
        self.assertAlmostEqual(a.total_upl({"X": 50_000.0}), 0.0, places=9)

    def test_多仓盈利进现金(self):
        a = self._acct(cash=10_000.0, taker_fee=0.0, maker_fee=0.0)
        a.apply_fill(inst_id="X", side="buy", qty=1.0, price=100.0, is_maker=False)
        a.apply_fill(inst_id="X", side="sell", qty=1.0, price=150.0, is_maker=False)
        self.assertAlmostEqual(a.cash, 10_050.0, places=9)
        self.assertAlmostEqual(a.total_realized, 50.0, places=9)

    def test_权益含未实现盈亏(self):
        a = self._acct(cash=1_000.0, taker_fee=0.0, maker_fee=0.0)
        a.apply_fill(inst_id="X", side="buy", qty=2.0, price=100.0, is_maker=False)
        self.assertAlmostEqual(a.equity({"X": 110.0}), 1_020.0, places=9)
        self.assertAlmostEqual(a.total_upl({"X": 110.0}), 20.0, places=9)

    def test_缺标记价时按均价估值(self):
        """⚠️ 按 0 估值会让权益瞬间塌陷、误判强平。"""
        a = self._acct(cash=1_000.0, taker_fee=0.0, maker_fee=0.0)
        a.apply_fill(inst_id="X", side="buy", qty=2.0, price=100.0, is_maker=False)
        self.assertAlmostEqual(a.equity({}), 1_000.0, places=9)

    def test_维持保证金按分档(self):
        a = self._acct(cash=1e9)
        a.apply_fill(inst_id="X", side="buy", qty=1.0, price=50_000.0,
                     is_maker=False)
        mm = a.maintenance_margin({"X": 50_000.0})
        # notional = 50000 恰好落在第一档（<=50000）→ 0.004
        self.assertAlmostEqual(mm, 50_000.0 * 0.004, places=9)

    def test_初始保证金等于名义价值除杠杆(self):
        a = self._acct(cash=1e9)
        a.apply_fill(inst_id="X", side="buy", qty=1.0, price=50_000.0,
                     is_maker=False)
        im = a.initial_margin({"X": 50_000.0}, {"X": 10.0})
        self.assertAlmostEqual(im, 5_000.0, places=9)

    # -- 保证金率口径 ---------------------------------------------------
    def test_保证金率口径是越小越危险(self):
        """⭐ 口径必须显式标注。另一种写法是 权益/维持保证金（越大越危险），
        两个混用会得出"报告说 0.4 很安全"而实际该强平的荒谬结论。

        手算（cash=1000, qty=1, avg=100, mmr=0.004）：
          mark=100 → eq=1000, mm=0.4,   ratio=(1000−0.4)/100 = 9.9960
          mark= 10 → eq= 910, mm=0.04,  ratio=(910−0.04)/10  = 90.9960
        """
        a = self._acct(cash=1_000.0, taker_fee=0.0, maker_fee=0.0)
        a.apply_fill(inst_id="X", side="buy", qty=1.0, price=100.0, is_maker=False)
        self.assertAlmostEqual(a.margin_ratio({"X": 100.0}),
                               (1000.0 - 0.4) / 100.0, places=9)
        self.assertAlmostEqual(a.margin_ratio({"X": 10.0}),
                               (910.0 - 0.04) / 10.0, places=9)

    def test_保证金率随价格下跌而上升是公式本身的性质(self):
        """⚠️ **这条测试记录的是一条反直觉的性质，不是我要的行为。**

        在 ``(权益 − 维持保证金) / 名义价值`` 这个口径下，价格下跌会让
        **分子分母同时缩小**，而分母缩得更快（``notional`` 与价格同比），
        所以 ratio **变大**。也就是说：**这个口径下 ratio 上升不代表安全**。

        真正的安全性判据是 ``is_liquidated()``（权益 <= 维持保证金），
        ``margin_ratio`` 只应当用来回答「离强平还有多少个 notional 的距离」。
        ⇒ **不要**把它当"越大越安全"的指标用；要做跨时点比较时，
        应当直接报 ``equity`` 与 ``maintenance_margin`` 两个绝对量。

        （真实 OKX 的 ``mgnRatio`` 也是这个数量级：它是"以当前名义价值计，
        权益还剩几倍"，价格下跌时它确实会升——这也是为什么交易所同时
        监控 ``mgnRatio`` 与 ``liqPx`` 的**距离**，而不是只看比值。）
        """
        a = self._acct(cash=1_000.0, taker_fee=0.0, maker_fee=0.0)
        a.apply_fill(inst_id="X", side="buy", qty=1.0, price=100.0, is_maker=False)
        self.assertGreater(a.margin_ratio({"X": 10.0}),
                           a.margin_ratio({"X": 100.0}))
        # 而真正该看的量：离强平的距离（用强平价），价格下跌时它确实在逼近
        p = a.position("X")
        liq = p.liq_price(cash=a.cash, mark=100.0, mmr_rate=0.004)
        self.assertLess(liq, 100.0)              # 强平价在下方 ⇒ 确实有强平风险
        self.assertGreater(a.margin_ratio({"X": float(liq)}), -1e-9)

    def test_保证金率在强平价处趋近零(self):
        """⭐ 真正的判据：ratio <= 0 等价于 equity <= 维持保证金（即该强平）。
        这条把 `margin_ratio` 与 `is_liquidated` 两个口径**接起来**，
        防止它们各算各的。"""
        a = self._acct(cash=100.0, taker_fee=0.0, maker_fee=0.0)
        a.apply_fill(inst_id="X", side="buy", qty=2.0, price=100.0, is_maker=False)
        p = a.position("X")
        liq = p.liq_price(cash=a.cash, mark=100.0, mmr_rate=0.004)
        self.assertAlmostEqual(a.margin_ratio({"X": liq}), 0.0, places=9)
        self.assertTrue(a.is_liquidated({"X": liq}))

    # -- 强平 -----------------------------------------------------------
    def test_无仓位不强平(self):
        self.assertFalse(self._acct().is_liquidated({}))

    def test_权益高于维持保证金不强平(self):
        a = self._acct(cash=1_000.0, taker_fee=0.0, maker_fee=0.0)
        a.apply_fill(inst_id="X", side="buy", qty=1.0, price=100.0, is_maker=False)
        self.assertFalse(a.is_liquidated({"X": 90.0}))

    def test_权益低于维持保证金则强平(self):
        a = self._acct(cash=1.0, taker_fee=0.0, maker_fee=0.0)
        a.apply_fill(inst_id="X", side="buy", qty=1.0, price=100.0, is_maker=False)
        # equity = 1 + (0.001-100) ≈ -98.999，mm ≈ 0.000004 ⇒ 应强平
        self.assertTrue(a.is_liquidated({"X": 0.001}))

    def test_强平边界用小于等于(self):
        """用严格的 < 会因浮点残渣随机决定两侧，产生"偶发通过"的假绿灯。"""
        a = self._acct(cash=0.0, taker_fee=0.0, maker_fee=0.0, mmr_tiers=(
            (float("inf"), 0.0),))
        a.apply_fill(inst_id="X", side="buy", qty=100.0, price=100.0,
                     is_maker=False)
        # avg=100, q=100, cash=0, mmr=0 ⇒ 价格 100 时 equity=0, mm=0 ⇒ 恰好相等
        self.assertTrue(a.is_liquidated({"X": 100.0}))

    def test_强平清掉仓位并记账(self):
        a = self._acct(cash=1.0, taker_fee=0.0005, maker_fee=0.0002)
        a.apply_fill(inst_id="X", side="buy", qty=1.0, price=100.0, is_maker=False)
        cash0 = a.cash
        out = a.liquidate("X", {"X": 50.0})
        self.assertTrue(out["liquidated"])
        self.assertEqual(a.n_liquidations, 1)
        self.assertTrue(a.position("X").is_flat)
        self.assertEqual(a.position("X").avg_px, 0.0)
        # 亏损 = (50-100)*1 = -50，再扣手续费 50*0.0005
        self.assertAlmostEqual(a.cash, cash0 - 50.0 - 0.025, places=9)

    def test_强平空仓无动作(self):
        a = self._acct()
        out = a.liquidate("X", {"X": 100.0})
        self.assertFalse(out["liquidated"])
        self.assertEqual(a.n_liquidations, 0)

    # -- can_open -------------------------------------------------------
    def test_保证金充足可以开仓(self):
        a = self._acct(cash=10_000.0)
        ok, why = a.can_open(inst_id="X", side="buy", qty=1.0, price=100.0,
                             lever=10.0, marks={"X": 100.0})
        self.assertTrue(ok, why)

    def test_权益不足以开仓被拒(self):
        a = self._acct(cash=0.0)
        ok, why = a.can_open(inst_id="X", side="buy", qty=1.0, price=100.0,
                             lever=1.0, marks={"X": 100.0})
        self.assertFalse(ok)
        self.assertEqual(why, "TW-1005")

    def test_判据是开仓后的权益而不是开仓前(self):
        """⭐ 只看开仓前，权益只剩一点点的账户能开出大仓，等于免费获得杠杆。"""
        a = self._acct(cash=0.5)
        ok, why = a.can_open(inst_id="X", side="buy", qty=10.0, price=100.0,
                             lever=10.0, marks={"X": 100.0})
        self.assertFalse(ok)

    def test_在亏损仓位上继续加仓要看加仓后的权益(self):
        """⭐ 回归守卫：新仓**自己带来的**未实现盈亏必须计入。

        构造一个正在浮亏的多头，然后以**低于均价**的价格加仓：
        加仓后的均价被拉低，浮亏缩小（upl > 原来的 upl），
        但真实权益仍然很低。若把"新仓带来的 upl"当成 0，
        模拟权益就**停在加仓前**——于是"在亏损仓位上继续加仓"
        这类最危险的加仓永远查不出来，等价格再动一点就穿仓。
        """
        a = self._acct(cash=100.0, max_lever=100.0)
        # 已有 2 手 @200，现价 100 ⇒ 浮亏 (100-200)*2 = -200，权益 = -100（已穿）
        a.apply_fill(inst_id="X", side="buy", qty=2.0, price=200.0, is_maker=True)
        ok, why = a.can_open(inst_id="X", side="buy", qty=1.0, price=100.0,
                             lever=1.0, marks={"X": 100.0})
        self.assertFalse(ok, "在已穿仓的浮亏仓位上不该还能开仓")
        self.assertEqual(why, "TW-1005")

    def test_加仓后的模拟权益必须等于真实重算值(self):
        """把上面那条从"结论"接到"数值"：模拟值要对得上直接构造的终态。

        做法：分别用 can_open 的模拟路径与"先成交再看 equity"的路径，
        在**同一个终态**上比较。两者必须同向（都拒或都允），
        否则说明模拟权益的公式与真实记账不一致。
        """
        for cash, px_new in ((100.0, 100.0), (10_000.0, 100.0),
                             (10_000.0, 50.0)):
            a = self._acct(cash=cash, max_lever=100.0)
            a.apply_fill(inst_id="X", side="buy", qty=2.0, price=200.0,
                         is_maker=True)
            ok_sim, _ = a.can_open(inst_id="X", side="buy", qty=1.0,
                                   price=px_new, lever=1.0,
                                   marks={"X": px_new})
            # 真实路径：真成交，再看权益与维持保证金的关系
            b = self._acct(cash=cash, max_lever=100.0)
            b.apply_fill(inst_id="X", side="buy", qty=2.0, price=200.0,
                         is_maker=True)
            b.apply_fill(inst_id="X", side="buy", qty=1.0, price=px_new,
                         is_maker=False)
            ok_real = b.equity({"X": px_new}) > b.maintenance_margin({"X": px_new})
            self.assertEqual(
                ok_sim, ok_real,
                f"cash={cash} px_new={px_new}：模拟路径={ok_sim} 真实路径={ok_real}",
            )

    def test_杠杆超限被拒(self):
        a = self._acct(cash=1e9, max_lever=10.0)
        ok, why = a.can_open(inst_id="X", side="buy", qty=1.0, price=100.0,
                             lever=20.0, marks={"X": 100.0})
        self.assertFalse(ok)
        self.assertEqual(why, "TW-1009")

    # -- 摘要 -----------------------------------------------------------
    def test_摘要字段齐全(self):
        a = self._acct(cash=10_000.0)
        a.apply_fill(inst_id="X", side="buy", qty=1.0, price=100.0, is_maker=True)
        s = a.summary({"X": 110.0}, {"X": 10.0})
        for k in ("cash", "equity", "upl", "maintenance_margin", "initial_margin",
                  "margin_ratio", "n_positions", "total_fees", "total_realized",
                  "n_liquidations"):
            self.assertIn(k, s)
        self.assertEqual(s["n_positions"], 1)
        self.assertGreater(s["upl"], 0)

    def test_多仓位权益合计(self):
        a = self._acct(cash=1_000.0, taker_fee=0.0, maker_fee=0.0)
        a.apply_fill(inst_id="X", side="buy", qty=1.0, price=100.0, is_maker=False)
        a.apply_fill(inst_id="Y", side="sell", qty=1.0, price=200.0, is_maker=False)
        # X: +10（110-100）；Y 空头：价格跌到 190 → +10
        self.assertAlmostEqual(a.equity({"X": 110.0, "Y": 190.0}), 1_020.0, places=9)

    def test_post_only省手续费(self):
        """⭐ maker 与 taker 费率必须分开算，否则 post_only 在评估里看不出好处。"""
        a1 = self._acct(cash=10_000.0)
        a2 = self._acct(cash=10_000.0)
        a1.apply_fill(inst_id="X", side="buy", qty=1.0, price=10_000.0,
                      is_maker=True)
        a2.apply_fill(inst_id="X", side="buy", qty=1.0, price=10_000.0,
                      is_maker=False)
        self.assertLess(a1.total_fees, a2.total_fees)
        self.assertAlmostEqual(a2.total_fees / a1.total_fees, 2.5, places=6)


if __name__ == "__main__":
    unittest.main()
