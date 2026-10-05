"""交易终端 · 会话层（S1）

⚠️ **定位：这个模块不生产、不预测。**
它只提供"把**已经算出来**的东西一根一根展示出来"的能力，
不产生任何新的研究结论。真正的结论仍然来自 `scripts/run_stage*.py` 那些跑批。

为什么能这么写
--------------
查过引擎（2026-10-04）：`Market.step()`、`Market.snapshot()` **本来就存在**，
`SimLog` 逐 tick 记了 `mid/best_bid/best_ask/spread/volume/n_trades/flow/
bid_depth/ask_depth/n_levels`，还有 `trades` 与 `snapshots`。
⇒ K 线、盘口、指标、回放**都不用新算**，只是还没被人看。
⇒ 模拟内核**一行都不用改**（延续 A1~A5 的"分层而不改造"）。

三条设计约束（踩过的坑）
----------------------
① **纸上账户必须在预热之前就加入市场。**
   `Market._next_order()` 给每个主体建调度流；若在一个已经跑过的市场上
   中途 `add_agent`，那条流可能没建 ⇒ 越界。所以在 `tick=0` 之前加，
   让初始化走正常路径。
② **会话状态必须加锁。** HTTP 服务器是多线程的（`ThreadingHTTPServer`），
   一个会话被两个请求同时 `step()` 会把 tick 推进错乱。
③ **可复现**：会话固定种子 + 同样的操作序列 ⇒ 同样的结果。
   这是本项目相对 TradingView 回放的一个真实优势，值得保住。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import numpy as np

from tw.agents import Agent
from tw.market import Market
from tw.scenarios import SCENARIOS, get_scenario, list_scenarios
from tw.types import DepthLevel, MarketState

__all__ = [
    "PaperAgent",
    "TerminalConfig",
    "TerminalSession",
    "get_session",
    "session_info",
    "list_scenarios",
]


class PaperAgent(Agent):
    """纸上账户：**自己永远不下单**，只响应来自界面的订单。

    为什么继承 `Agent` 而不是新写一个：市场的 `submit()` 要用到
    `on_submitted / open_orders / prune_open_orders / recompute_reservations /
    stats` 这一整套钩子，继承 `Agent` 只差一个抽象方法 `decide()`
    —— 把它返回 `None`，这个账户就"只会听指挥"。
    """

    def decide(self, state: MarketState):  # noqa: ANN201 - 签名照抄基类
        return None


@dataclass
class TerminalConfig:
    """一个终端会话的配置。默认值要能在**几秒内**跑出可看的东西。"""

    scenario: str = "normal"
    seed: int = 20261004
    #: 回放窗口长度（预热之后往前走的 tick 数）。
    n_ticks: int = 3000
    #: 盘口深度阶梯显示的档数
    depth: int = 12
    #: 纸上账户初始现金。**None = 按市场价格自动定**（见 `_rebuild`）。
    #: ⚠️ 之前写死 10 万，在 normal 场景下**只买得起 0.47 手**
    #    （一手 ≈ 5.9 万）⇒ 开箱即用极难受。金额必须跟着市场的价格尺度走。
    cash: float | None = None
    #: 自动定现金时用「几个初始价格」当购买力
    cash_in_price_units: float = 200.0
    #: 每次 state() 返回多少根 K 线（前端画得动多少就返回多少）
    max_bars: int = 400


class TerminalSession:
    """一个"能一根一根走"的市场 + 一个纸上账户。

    用法（服务端线程里）::

        sess = get_session()
        sess.step(10)                 # 往前走 10 根
        st = sess.state()             # 前端拉全量状态
        sess.order("buy", 1.0, 60000)  # 从图上下单 → 走**真实**撮合引擎
    """

    def __init__(self, cfg: TerminalConfig | None = None) -> None:
        self.cfg = cfg or TerminalConfig()
        self._lock = threading.RLock()
        self._paper: PaperAgent | None = None
        self._market: Market | None = None
        self._base: int = 0          # 预热长度（可见区的起点）
        self._cursor: int = 0        # 可见区里已经走到的位置
        self._rebuild()

    # ------------------------------------------------------------------ 建
    def _rebuild(self) -> None:
        sc = get_scenario(self.cfg.scenario)
        cfg = sc.config(self.cfg.seed)
        m = Market(cfg)
        # 现金跟着价格尺度走（否则 10 万只买得起 0.47 手）
        cash = (self.cfg.cash if self.cfg.cash is not None
                else self.cfg.cash_in_price_units * float(cfg.initial_price))
        self.cash_start = float(cash)
        # ⭐ ① 必须在预热之前加（见模块文档）
        self._paper = PaperAgent(
            agent_id="paper",
            cash=cash,
            inventory=0.0,
            rng=np.random.default_rng([self.cfg.seed, 0x9A17]),
            max_open_orders=40,
        )
        m.add_agent(self._paper)
        m.run(sc.warmup)              # 预热跑到稳态
        self._market = m
        self._base = int(sc.warmup)
        self._cursor = 0

    # ---------------------------------------------------------------- 推进
    def step(self, n: int = 1) -> dict:
        """往前走 n 根。返回最新状态摘要。

        ⚠️ ``n <= 0`` 是**纯查询**（只回当前状态，不推进）。
        为什么需要它：前端"切换周期/重新取数"是**不该动市场**的操作，
        而旧实现把 0 夹成 1 ⇒ **切一下周期就白走一根 K 线**。
        ⭐ 语义分家：``0 = 只查``、``>0 = 前进``、``seek() 负责往回找``（它只允许前进）。
        """
        with self._lock:
            m = self._market
            assert m is not None
            if int(n) <= 0:
                return {
                    "cursor": self._cursor,
                    "total": self.cfg.n_ticks,
                    "at_end": self._cursor >= self.cfg.n_ticks,
                    "tick": int(m.tick),
                }
            n = max(1, min(int(n), self.cfg.n_ticks - self._cursor))
            for _ in range(n):
                if self._cursor >= self.cfg.n_ticks:
                    break
                m.step()
                self._cursor += 1
            return {
                "cursor": self._cursor,
                "total": self.cfg.n_ticks,
                "at_end": self._cursor >= self.cfg.n_ticks,
                "tick": int(m.tick),
            }

    def seek(self, cursor: int) -> dict:
        """跳到指定位置（只允许前进，不允许后退 —— 回放要可复现）。"""
        with self._lock:
            target = max(self._cursor, min(int(cursor), self.cfg.n_ticks))
            return self.step(target - self._cursor)

    def reset(self, **overrides: Any) -> dict:
        with self._lock:
            for k, v in overrides.items():
                if not hasattr(self.cfg, k):
                    raise KeyError(f"没有这个配置项：{k}")
                setattr(self.cfg, k, v)
            self._rebuild()
            return self.step(0)

    # ---------------------------------------------------------------- 取数
    def bars(self, agg: int = 1) -> list[dict]:
        """把可见区的逐 tick 序列聚合成 K 线。

        ⚠️ OHLC 用**中间价序列**（`log.mid`）而不是逐笔成交价：
        本项目里"价格"的可信刻度就是 mid（成交价含撮合噪声），
        混用两种口径会让人把撮合噪声当成行情波动。
        成交量/成交笔数取自 `log.volume` / `log.n_trades`。
        """
        with self._lock:
            m = self._market
            assert m is not None
            agg = max(1, int(agg))
            lo = self._base
            hi = min(len(m.log.mid), self._base + self._cursor)
            if hi <= lo:
                return []
            mid = np.asarray(m.log.mid[lo:hi], dtype=float)
            vol = self._series("volume", lo, hi)
            ntr = self._series("n_trades", lo, hi)
            depth = self._series("bid_depth", lo, hi)
            n = len(mid)
            n_bar = (n + agg - 1) // agg
            out: list[dict] = []
            for b in range(n_bar):
                i0, i1 = b * agg, min((b + 1) * agg, n)
                seg = mid[i0:i1]
                out.append({
                    "t": lo + i0,                       # 起始 tick（不是时间戳）
                    "o": float(seg[0]),
                    "h": float(seg.max()),
                    "l": float(seg.min()),
                    "c": float(seg[-1]),
                    "v": float(vol[i0:i1].sum()) if len(vol) else 0.0,
                    "n": float(ntr[i0:i1].sum()) if len(ntr) else 0.0,
                    "d": float(depth[i0:i1].mean()) if len(depth) else 0.0,
                })
            # 只给前端画得动的那点量
            return out[-self.cfg.max_bars:]

    def _series(self, name: str, lo: int, hi: int) -> np.ndarray:
        m = self._market
        assert m is not None
        arr = getattr(m.log, name, None)
        if arr is None:
            return np.zeros(0, dtype=float)
        a = np.asarray(arr[lo:hi], dtype=float)
        return a if a.size == hi - lo else np.zeros(0, dtype=float)

    def book(self) -> dict:
        """盘口深度阶梯（DOM）。"""
        with self._lock:
            m = self._market
            assert m is not None
            snap = m.snapshot(self.cfg.depth)

            def lv(rows: list) -> list[dict]:
                # ⚠️ `bids`/`asks` 是 **`DepthLevel` 对象**（dataclass：
                # price / quantity / n_orders），**不是三元组**。
                # 写成 `for p, q, n in rows` 会报 "cannot unpack non-iterable"
                # （本轮踩过）。这里按属性取，顺带对两种形态都兜住。
                out = []
                for r in rows:
                    if isinstance(r, DepthLevel):
                        out.append({"p": float(r.price), "q": float(r.quantity),
                                    "n": int(r.n_orders)})
                    else:                      # 兼容三元组
                        p, q, n = r
                        out.append({"p": float(p), "q": float(q), "n": int(n)})
                return out
            return {
                "tick": int(snap.tick),
                "mid": float(snap.mid),
                "bids": lv(snap.bids),
                "asks": lv(snap.asks),
            }

    def account(self) -> dict:
        """纸上账户的余额 / 持仓 / 挂单。"""
        with self._lock:
            m = self._market
            a = self._paper
            assert m is not None and a is not None
            mid = m.current_mid()
            return {
                "cash": round(a.cash, 4),
                "reserved_cash": round(a.reserved_cash, 4),
                "inventory": round(a.inventory, 6),
                "reserved_inventory": round(a.reserved_inventory, 6),
                "equity": round(a.equity(mid), 4) if mid else None,
                "pnl": round(a.equity_change(mid), 4) if mid else None,
                "n_submitted": int(a.stats.n_submitted),
                "n_trades": int(a.stats.n_trades),
                "n_cancelled": int(a.stats.n_cancelled),
                "n_rejected": int(a.stats.n_rejected),
                "volume": round(a.stats.volume, 6),
                "notional": round(a.stats.notional, 2),
                "orders": [
                    {
                        "id": o.order_id,
                        "side": o.side,
                        "price": (None if not np.isfinite(o.price)
                                  else round(o.price, 2)),
                        "qty": round(float(o.quantity), 6),
                        "type": o.order_type,
                    }
                    for o in a.open_orders
                ],
                # 成交明细直接带在这里（前端一次拿全量，不用再发一次请求）
                "trades": self.my_trades(20),
            }

    def state(self, agg: int = 1) -> dict:
        """前端一次拉全量。"""
        with self._lock:
            return {
                "cfg": {
                    "scenario": self.cfg.scenario,
                    "seed": self.cfg.seed,
                    "n_ticks": self.cfg.n_ticks,
                    "max_bars": self.cfg.max_bars,
                    "depth": self.cfg.depth,
                    "scenarios": [n for n, _ in list_scenarios()],
                    "intent": SCENARIOS[self.cfg.scenario].intent,
                },
                "cursor": self._cursor,
                "at_end": self._cursor >= self.cfg.n_ticks,
                "bars": self.bars(agg),
                "agg": max(1, int(agg)),
                "book": self.book(),
                "account": self.account(),
            }

    # ---------------------------------------------------------------- 下单
    def order(self, side: str, qty: float, price: float | None = None,
              order_type: str = "limit") -> dict:
        """从界面上下单。**走真实撮合引擎**（不是假成交）。

        ⚠️ 订单是**排队**的：限价单挂上去不一定立刻成交，
        什么时候成交取决于市场走到什么价位 —— 这正是要看的现象。
        """
        with self._lock:
            m = self._market
            a = self._paper
            assert m is not None and a is not None
            if side not in ("buy", "sell"):
                raise ValueError(f"side 只能是 buy/sell，收到 {side!r}")
            px = price if price is not None else m.current_mid()
            od = a.new_order(m._refresh_state(), side, float(px), float(qty),
                             order_type=order_type)
            if od is None:
                return {"ok": False, "why": "数量或价格非法，订单被丢弃"}
            before = len(a.open_orders)
            trades = m.submit(a, od)
            return {
                "ok": True,
                "order_id": od.order_id,
                "n_trades": len(trades),
                "trades": [
                    {"price": round(t.price, 2), "qty": round(t.quantity, 6)}
                    for t in trades
                ],
                "resting": len(a.open_orders) > before,
                "account": self.account(),
            }

    def cancel_order(self, order_id: str) -> dict:
        with self._lock:
            a = self._paper
            assert a is not None
            for o in list(a.open_orders):
                if o.order_id == order_id:
                    ok = a.cancel(o)
                    return {"ok": bool(ok), "order_id": order_id,
                            "account": self.account()}
            return {"ok": False, "why": f"没有这张挂单：{order_id}"}

    def my_trades(self, limit: int = 50) -> list[dict]:
        """纸上账户的成交明细（从 `SimLog.trades` 里按 agent_id 筛）。"""
        with self._lock:
            m = self._market
            assert m is not None and self._paper is not None
            aid = self._paper.agent_id
            # ⚠️ `Trade` 的字段是 `buy_agent_id`/`sell_agent_id`/`tick`
            #    （**不是** `buyer_id`/`seller_id`/`timestamp`）——本轮踩过。
            out = [
                {"tick": int(t.tick), "price": round(t.price, 2),
                 "qty": round(t.quantity, 6),
                 "side": "buy" if t.buy_agent_id == aid else "sell",
                 "aggressor": t.aggressor_side}
                for t in m.log.trades
                if t.buy_agent_id == aid or t.sell_agent_id == aid
            ]
            return out[-limit:]


# ---------------------------------------------------------------- 会话注册
#: 一个进程一个默认会话（本地单用户工具）。
#: ⓘ 不按 token 分：浏览器开两个标签页共享同一根回放进度条，
#:    对"看同一个市场"来说反而是想要的语义。
_SESSIONS: dict[str, TerminalSession] = {}
_SESSIONS_LOCK = threading.Lock()


def get_session(sid: str = "default", **overrides: Any) -> TerminalSession:
    with _SESSIONS_LOCK:
        s = _SESSIONS.get(sid)
        if s is None:
            s = TerminalSession(TerminalConfig(**overrides) if overrides else None)
            _SESSIONS[sid] = s
        return s


def reset_session(sid: str = "default", **overrides: Any) -> TerminalSession:
    with _SESSIONS_LOCK:
        s = _SESSIONS.get(sid)
        if s is None:
            s = TerminalSession(TerminalConfig(**overrides) if overrides else None)
            _SESSIONS[sid] = s
        else:
            s.reset(**overrides)
        return s


def session_info() -> dict:
    with _SESSIONS_LOCK:
        return {
            "open": sorted(_SESSIONS),
            "scenarios": [n for n, _ in list_scenarios()],
        }
