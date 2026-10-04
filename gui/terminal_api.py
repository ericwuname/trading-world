"""交易终端 · HTTP 接口（S2）

⚠️ **URL 与处理函数写在一起**（见 `gui/routes.py` 的说明）：
端点必须和它的实现待在同一处，否则"加了实现没加 URL"会**静默 404**
（`27fc86b` 踩过：M127 就是端点漏挂装饰器，处理函数单测全绿而 URL 一律 404）。

⚠️ 每个会改状态的接口都**把最新 state 一起返回**。
理由：前端画图需要 bars + book + account + cursor 四样东西；
如果"改状态"与"取状态"分成两个请求，中间那一瞬间的状态是可变的
⇒ 就会出现"订单成交了但图上还没反映"这类**看起来像 bug 的时序问题**。
一次往返拿全量，前端就不用自己拼。

⚠️ 会话是**长生命周期的交互状态**（一根一根往前走的进度条就在里面），
所以它和 `/api/job` 那种"提交完就跑完"的作业不是一回事：
**不能**走 `KINDS` 那套作业队列。`ctx.mgr`（作业管理器）在这里用不上。
"""

from __future__ import annotations

from .routes import ApiError, Ctx, query_int, route
from .terminal import TerminalConfig, get_session, reset_session, session_info

#: 前端一次最多要多少根 K 线（会话的 `max_bars` 也会截断）
_MAX_AGG = 64


def _agg(ctx: Ctx, default: int = 1) -> int:
    v = query_int(ctx.query, "agg", default)
    if not 1 <= v <= _MAX_AGG:
        raise ApiError(f"agg 必须在 1~{_MAX_AGG} 之间，收到 {v}")
    return v


def _with_state(ctx: Ctx, extra: dict, agg: int | None = None) -> dict:
    """改状态之后的统一返回：结果 + 全量状态。"""
    s = get_session()
    a = agg if agg is not None else _agg(ctx)
    return {**extra, "state": s.state(agg=a)}


@route("GET", "/api/terminal/state")
def api_terminal_state(ctx: Ctx) -> dict:
    """拉全量状态：K 线 + 盘口 + 账户 + 进度。"""
    return get_session().state(agg=_agg(ctx))


@route("POST", "/api/terminal/step")
def api_terminal_step(ctx: Ctx) -> dict:
    """往前推进 n 根（回放的"单根前进"与"连续播放"共用它）。

    n 会**被服务端夹到合法区间**：超过剩余长度就走到头，
    不会报错 —— 因为"播放到末尾自动停"是正常行为，不是错误。
    """
    body = ctx.body
    n = body.get("n", 1)
    try:
        n = int(n)
    except (TypeError, ValueError):
        raise ApiError(f"n 必须是整数，收到 {n!r}") from None
    agg = int(body.get("agg", 1) or 1)
    if not 1 <= agg <= _MAX_AGG:
        raise ApiError(f"agg 必须在 1~{_MAX_AGG} 之间，收到 {agg}")
    s = get_session()
    info = s.step(n)
    return _with_state(ctx, {"step": info}, agg=agg)


@route("POST", "/api/terminal/order")
def api_terminal_order(ctx: Ctx) -> dict:
    """从界面下单 —— 走**真实撮合引擎**，不是假成交。

    ⭐ 限价单是**排队**的：挂上去不一定立刻成交，什么时候成交取决于
    市场走到什么价位。**这正是这个终端要让人亲眼看到的东西**，
    所以这里不做"即时成交"的模拟。
    """
    body = ctx.body
    side = str(body.get("side", "")).strip().lower()
    if side not in ("buy", "sell"):
        raise ApiError(f"side 只能是 buy 或 sell，收到 {side!r}")
    try:
        qty = float(body.get("qty", 0))
    except (TypeError, ValueError):
        raise ApiError(f"qty 必须是数字，收到 {body.get('qty')!r}") from None
    if qty <= 0:
        raise ApiError(f"qty 必须为正，收到 {qty}")
    price = body.get("price")
    if price in (None, ""):
        price = None
    else:
        try:
            price = float(price)
        except (TypeError, ValueError):
            raise ApiError(f"price 必须是数字或留空，收到 {price!r}") from None
    otype = str(body.get("type", "limit")).strip().lower()
    if otype not in ("limit", "market"):
        raise ApiError(f"type 只能是 limit 或 market，收到 {otype!r}")
    s = get_session()
    try:
        res = s.order(side, qty, price, otype)
    except ValueError as exc:                    # 会话层的参数校验
        raise ApiError(str(exc)) from None
    return _with_state(ctx, {"order": res})


@route("POST", "/api/terminal/cancel")
def api_terminal_cancel(ctx: Ctx) -> dict:
    """撤掉一张挂单。撤单不存在不算错误（可能已经成交或已被撤）。"""
    oid = str(ctx.body.get("order_id", "")).strip()
    if not oid:
        raise ApiError("缺少 order_id")
    s = get_session()
    return _with_state(ctx, {"cancel": s.cancel_order(oid)})


@route("POST", "/api/terminal/reset")
def api_terminal_reset(ctx: Ctx) -> dict:
    """重开一个会话（换场景 / 换种子 / 换长度）。

    ⚠️ `cash` 允许显式给数字；`null` 或不给 = **按市场价格自动定**
    （见 `TerminalConfig.cash` 的说明：写死 10 万在这个市场只买得起 0.47 手）。
    """
    body = ctx.body
    kw: dict = {}
    for key, cast in (("scenario", str), ("seed", int), ("n_ticks", int),
                      ("depth", int), ("cash", float),
                      ("cash_in_price_units", float)):
        if key not in body or body[key] is None:
            continue
        try:
            kw[key] = cast(body[key])
        except (TypeError, ValueError):
            raise ApiError(f"{key} 类型不对：{body[key]!r}") from None
    try:
        s = reset_session(**kw)
    except KeyError as exc:                      # 未知场景名
        raise ApiError(str(exc)) from None
    except TypeError as exc:
        raise ApiError(f"配置项不对：{exc}") from None
    return _with_state(ctx, {"reset": True, "session": session_info()["scenarios"]})


@route("GET", "/api/terminal/scenarios")
def api_terminal_scenarios(ctx: Ctx) -> dict:
    """有哪些市场状态可选，**以及每个想测什么**（不写清楚数字没法解读）。"""
    from .terminal import list_scenarios

    return {"scenarios": [{"name": n, "intent": i} for n, i in list_scenarios()]}
