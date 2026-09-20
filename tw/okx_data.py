"""OKX 公开行情拉取（无需 API key）。

范围
----
**只做公开数据**。下单、查账户这类需要鉴权的端点**不在这里**，
也不应该在这里——那是 A1 的执行层和 B 路线的网关的事。

查证过的端点（全部免鉴权，2026-09-20）
--------------------------------------
======================  ==================  ========  ==========
端点                     用途                 单次上限   限频
======================  ==================  ========  ==========
/market/candles          近期 K 线（热缓存）    300       40 次/2s
/market/history-candles  历史 K 线（冷存储）    100       20 次/2s
/market/books            盘口深度              sz 档位    —
/market/trades           最近成交              —         —
/public/funding-rate     资金费率（仅永续）      —         20 次/2s
/public/open-interest    未平仓量              —         20 次/2s
/public/mark-price       标记价                —         —
/public/instruments      合约规格              —         10 次/2s
======================  ==================  ========  ==========

⚠️ 两个已查证的坑（写在这里免得再踩）
------------------------------------
1. **分页参数命名反直觉**：``after`` 传时间戳返回的是**比它更早**的数据、
   ``before`` 返回**更新**的。倒推历史用 ``after`` 翻页，两者**不能同时传**。
   翻页取上批**最旧一条 ts − 1ms** 作为下一次 ``after``；
   **减 1ms 是必须的**，否则边界那根会出现在两批里。
2. **``confirm`` 字段**：``'0'`` = 这根还没走完。**默认过滤掉**，
   因为它们会让策略读到未来。

不用 requests
-------------
用标准库 ``urllib``。理由与 ``tw/realdata.py`` 不引 pandas 相同：
依赖清单越短，「环境不对所以没跑」的借口越少。而且这里只需要 GET + 读 JSON。
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from .marketdb import (
    SOURCE_OKX,
    Candle,
    FetchRecord,
    MarketStore,
)

BASE_URL = "https://www.okx.com"

#: 用户要求的「其他交易所」接口留的口子。目前只实现 OKX——
#: 加第二家之前先把这一家做扎实（多源对齐是独立问题，见 docs）。
SUPPORTED_SOURCES = ("okx",)

#: 端点 -> 每次请求的最小间隔（秒）。取查证到的保守值再留一点余量。
DEFAULT_MIN_INTERVAL = {
    "/api/v5/market/candles": 0.06,             # 40/2s -> 0.05
    "/api/v5/market/history-candles": 0.12,     # 20/2s -> 0.10
    "/api/v5/public/instruments": 0.25,         # 10/2s -> 0.20
    "/api/v5/public/funding-rate": 0.12,
    "/api/v5/public/open-interest": 0.12,
    "/api/v5/market/books": 0.10,
    "/api/v5/market/trades": 0.10,
    "/api/v5/public/mark-price": 0.12,
}
DEFAULT_INTERVAL = 0.15


class OkxError(RuntimeError):
    """OKX 返回非 0 code，或网络层失败。"""


@dataclass(slots=True)
class OkxHttp:
    """极薄的 HTTP 客户端：GET + 解包 + 限频 + 重试。

    刻意不做连接池 / 异步 —— 拉行情是**准备阶段的一次性动作**，
    不是高频路径。复杂度花在这里不值。
    """

    base_url: str = BASE_URL
    timeout: float = 20.0
    max_retries: int = 3
    backoff: float = 1.0
    #: 可注入的 http 函数（测试用；签名 ``(url, timeout) -> bytes``）
    fetch: Callable[[str, float], bytes] | None = None

    #: 最近一次请求时刻，按端点记（限频是**按端点**的，不是全局）
    _last: dict[str, float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self._last is None:
            self._last = {}

    # -- 内部 -------------------------------------------------------------

    def _default_fetch(self, url: str, timeout: float) -> bytes:
        req = urllib.request.Request(
            url,
            headers={
                # OKX 对无 UA 的请求偶发拒绝；给一个正常的 UA 更稳
                "User-Agent": "trading-world/1.0 (+research; public endpoints only)",
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()

    def _throttle(self, path: str) -> None:
        iv = DEFAULT_MIN_INTERVAL.get(path, DEFAULT_INTERVAL)
        last = self._last.get(path, 0.0)
        wait = iv - (time.monotonic() - last)
        if wait > 0:
            time.sleep(wait)
        self._last[path] = time.monotonic()

    def get(self, path: str, params: dict[str, Any] | None = None) -> list[Any]:
        """GET 一个 OKX 端点，返回 ``data`` 数组。"""
        query = urllib.parse.urlencode(
            {k: v for k, v in (params or {}).items() if v is not None and v != ""}
        )
        url = f"{self.base_url}{path}" + (f"?{query}" if query else "")
        fn = self.fetch or self._default_fetch

        last_err: Exception | None = None
        for attempt in range(max(1, self.max_retries)):
            self._throttle(path)
            try:
                raw = fn(url, self.timeout)
                payload = json.loads(raw)
                code = str(payload.get("code", ""))
                if code != "0":
                    # OKX 的业务错误：重试通常没用（参数错就是错），直接抛
                    raise OkxError(
                        f"OKX code={code} msg={payload.get('msg')!r} url={url}"
                    )
                data = payload.get("data")
                if not isinstance(data, list):
                    raise OkxError(f"OKX data 不是数组：{type(data).__name__} url={url}")
                return data
            except OkxError:
                raise
            except (urllib.error.URLError, TimeoutError, OSError,
                    json.JSONDecodeError) as exc:
                last_err = exc
                if attempt + 1 < max(1, self.max_retries):
                    time.sleep(self.backoff * (2 ** attempt))
        raise OkxError(f"请求失败（已重试 {self.max_retries} 次）：{last_err}  url={url}")


# --------------------------------------------------------------------------
# 解析
# --------------------------------------------------------------------------


def parse_candle(row: Sequence[str]) -> Candle:
    """OKX 的 K 线行：``[ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]``。

    ⚠️ **空字符串要能容错**：OKX 在某些合约 / 停牌时段会返回空串，
    直接 ``float('')`` 会抛 ``ValueError`` 把整个拉取打断。
    价格字段是**必需**的（空就说明这行本身没意义，该跳过），
    但 ``volume`` 空只代表「这个口径没有量」，不该让整行作废 —— 记 0 继续。
    """
    if len(row) < 6:
        raise ValueError(f"K 线行太短：{row!r}")

    def _req(i: int, name: str) -> float:
        v = row[i]
        if v is None or v == "":
            raise ValueError(f"K 线 {name} 为空：{row!r}")
        return float(v)

    def _opt(i: int, default: float = 0.0) -> float:
        if i >= len(row) or row[i] in (None, ""):
            return default
        try:
            return float(row[i])
        except (TypeError, ValueError):
            return default

    return Candle(
        ts=int(row[0]),
        open=_req(1, "open"),
        high=_req(2, "high"),
        low=_req(3, "low"),
        close=_req(4, "close"),
        volume=_opt(5),
        confirm=int(row[8]) if len(row) > 8 and row[8] not in ("", None) else 1,
    )


# --------------------------------------------------------------------------
# 拉取
# --------------------------------------------------------------------------


def fetch_instruments(
    http: OkxHttp, inst_type: str = "SPOT", *, inst_id: str = ""
) -> list[dict[str, Any]]:
    """合约规格（tickSz / lotSz / minSz）。下单前必须知道这些。"""
    return http.get(
        "/api/v5/public/instruments",
        {"instType": inst_type, "instId": inst_id or None},
    )


def fetch_candles_recent(
    http: OkxHttp, inst_id: str, bar: str, *, limit: int = 300
) -> list[Candle]:
    """近期 K 线。**热缓存**，单次最多 300，最多回溯约 3 个月。"""
    rows = http.get(
        "/api/v5/market/candles",
        {"instId": inst_id, "bar": bar, "limit": min(int(limit), 300)},
    )
    out = [parse_candle(r) for r in rows]
    out.sort(key=lambda c: c.ts)  # OKX 返回是倒序的，统一成升序
    return out


def fetch_candles_history(
    http: OkxHttp, inst_id: str, bar: str, *,
    after: int = 0, before: int = 0, limit: int = 100,
) -> list[Candle]:
    """历史 K 线。**冷存储**，单次最多 100，主流币可回溯约 3 年。

    ⚠️ ``after`` 返回**更早**的数据、``before`` 返回**更新**的（反直觉，已查证）。
    """
    rows = http.get(
        "/api/v5/market/history-candles",
        {
            "instId": inst_id, "bar": bar,
            "after": int(after) if after else None,
            "before": int(before) if before else None,
            "limit": min(int(limit), 100),
        },
    )
    out = [parse_candle(r) for r in rows]
    out.sort(key=lambda c: c.ts)
    return out


def fetch_funding_rate(http: OkxHttp, inst_id: str) -> dict[str, Any]:
    """资金费率。**仅永续**（``-SWAP`` 结尾）。"""
    data = http.get("/api/v5/public/funding-rate", {"instId": inst_id})
    return data[0] if data else {}


def fetch_open_interest(http: OkxHttp, inst_id: str) -> dict[str, Any]:
    data = http.get("/api/v5/public/open-interest", {"instType": "SWAP", "instId": inst_id})
    return data[0] if data else {}


def fetch_mark_price(http: OkxHttp, inst_id: str) -> dict[str, Any]:
    data = http.get("/api/v5/public/mark-price", {"instType": "SWAP", "instId": inst_id})
    return data[0] if data else {}


def fetch_book(http: OkxHttp, inst_id: str, sz: int = 20) -> dict[str, Any]:
    """盘口。``sz`` 只能取 5/10/20/50/100/400。"""
    if sz not in (5, 10, 20, 50, 100, 400):
        raise ValueError(f"sz 必须是 5/10/20/50/100/400 之一，收到 {sz}")
    data = http.get("/api/v5/market/books", {"instId": inst_id, "sz": sz})
    return data[0] if data else {}


def fetch_ticker(http: OkxHttp, inst_id: str) -> dict[str, Any]:
    data = http.get("/api/v5/market/ticker", {"instId": inst_id})
    return data[0] if data else {}


# --------------------------------------------------------------------------
# 落库（按需拉取的完整动作）
# --------------------------------------------------------------------------


def ensure_history(
    store: MarketStore,
    inst_id: str,
    bar: str,
    *,
    days: int = 365,
    http: OkxHttp | None = None,
    confirmed_only: bool = True,
    progress: Callable[[str, int], None] | None = None,
) -> Coverage:
    """**按需拉取**：把 ``inst_id`` 近 ``days`` 天的 ``bar`` 补进库。

    这是用户要求的「**在选择准备交易的时候才去拉**」的实现。
    动作顺序刻意如此：

    1. 先查库 —— **够用就不拉**（避免重复联网，也避免覆盖已冻结的历史）
    2. 只有缺口才联网
    3. 分页时取上批最旧 ``ts − 1ms`` 作为下一次 ``after``（**减 1ms 必须有**）
    4. 写 ``fetches`` 账本 —— 成功失败**都写**

    返回补齐后的覆盖情况。

    ⚠️ 这个函数**会联网**。调用它的地方应该只在「会话准备阶段」，
    不该在决策路径上。
    """
    from .marketdb import Coverage

    http = http or OkxHttp()
    started = int(time.time() * 1000)
    want_from = int((time.time() - days * 86400) * 1000)

    cov = store.coverage(SOURCE_OKX, inst_id, bar, confirmed_only=confirmed_only)
    if not cov.is_empty and cov.first_ts <= want_from:
        # 已经覆盖到目标起点之前 —— 不联网
        if progress:
            progress("库已覆盖，跳过联网", cov.n_rows)
        store.log_fetch(FetchRecord(
            source=SOURCE_OKX, inst_id=inst_id, bar=bar, endpoint="(cached)",
            from_ts=cov.first_ts, to_ts=cov.last_ts, n_rows=0, pages=0,
            started_at=started, finished_at=int(time.time() * 1000), ok=True,
        ))
        return cov

    cursor = cov.first_ts if not cov.is_empty else 0
    total = 0
    pages = 0
    endpoint = "/api/v5/market/history-candles"
    try:
        # ① 先补近端（热缓存端点，一次拿 300，效率高）
        recent = fetch_candles_recent(http, inst_id, bar, limit=300)
        if confirmed_only:
            recent = [c for c in recent if c.confirm == 1]
        total += store.upsert_candles(SOURCE_OKX, inst_id, bar, recent)
        pages += 1
        if progress:
            progress("近期 K 线", len(recent))
        if recent:
            cursor = min(cursor, recent[0].ts) if cursor else recent[0].ts

        # ② 倒推补远端（冷存储端点，单次 100，需要翻页）
        while True:
            batch = fetch_candles_history(http, inst_id, bar, after=cursor, limit=100)
            if not batch:
                break
            if confirmed_only:
                batch = [c for c in batch if c.confirm == 1]
            pages += 1
            if batch:
                total += store.upsert_candles(SOURCE_OKX, inst_id, bar, batch)
                oldest = batch[0].ts
                if progress:
                    progress(f"历史 K 线（{_iso(oldest)}）", len(batch))
                if oldest <= want_from:
                    break
                if oldest >= cursor:
                    # 没往前走 —— 端点行为变了或参数不对。**必须报错**，
                    # 否则会无限翻同一页（静默死循环比报错危险得多）。
                    raise OkxError(
                        f"分页没有前进：oldest={oldest} cursor={cursor}；已中止"
                    )
                cursor = oldest - 1  # ⚠️ 减 1ms，否则边界那根会重复出现
            else:
                # 整批都是未确认的 —— 不会再往前了
                break
            if pages > 500:
                raise OkxError("翻页超过 500 次，疑似分页参数错误；已中止")

        store.log_fetch(FetchRecord(
            source=SOURCE_OKX, inst_id=inst_id, bar=bar, endpoint=endpoint,
            from_ts=want_from, to_ts=int(time.time() * 1000),
            n_rows=total, pages=pages, started_at=started,
            finished_at=int(time.time() * 1000), ok=True,
        ))
    except Exception as exc:  # noqa: BLE001 — 账本要记下任何失败
        store.log_fetch(FetchRecord(
            source=SOURCE_OKX, inst_id=inst_id, bar=bar, endpoint=endpoint,
            from_ts=want_from, n_rows=total, pages=pages, started_at=started,
            finished_at=int(time.time() * 1000), ok=False, error=str(exc)[:400],
        ))
        raise

    return store.coverage(SOURCE_OKX, inst_id, bar, confirmed_only=confirmed_only)


def snapshot_metrics(
    store: MarketStore, inst_id: str, *, http: OkxHttp | None = None
) -> dict[str, Any]:
    """抓一次快照类指标（资金费率 / 未平仓量 / 标记价）并落库。

    这些量**没有 OHLC**，所以走 ``metrics`` 表。
    只要 ``inst_id`` 是永续（``-SWAP``），三个都有；现货只有标记价/指数价。
    """
    http = http or OkxHttp()
    now = int(time.time() * 1000)
    out: dict[str, Any] = {}
    is_swap = inst_id.upper().endswith("-SWAP")

    if is_swap:
        try:
            fr = fetch_funding_rate(http, inst_id)
            if fr:
                rate = _f(fr.get("fundingRate"))
                ts = int(fr.get("fundingTime") or now)
                store.upsert_metrics(SOURCE_OKX, inst_id, "funding_rate", [
                    (ts, rate, {"nextFundingTime": fr.get("nextFundingTime")})
                ])
                out["funding_rate"] = rate
        except OkxError as exc:
            out["funding_rate_error"] = str(exc)

        try:
            oi = fetch_open_interest(http, inst_id)
            if oi:
                store.upsert_metrics(SOURCE_OKX, inst_id, "open_interest", [
                    (int(oi.get("ts") or now), _f(oi.get("oi")),
                     {"oiCcy": oi.get("oiCcy")})
                ])
                out["open_interest"] = _f(oi.get("oi"))
        except OkxError as exc:
            out["open_interest_error"] = str(exc)

        try:
            mp = fetch_mark_price(http, inst_id)
            if mp:
                store.upsert_metrics(SOURCE_OKX, inst_id, "mark_price", [
                    (int(mp.get("ts") or now), _f(mp.get("markPx")), None)
                ])
                out["mark_price"] = _f(mp.get("markPx"))
        except OkxError as exc:
            out["mark_price_error"] = str(exc)

    return out


def _f(v: Any) -> float | None:
    try:
        return None if v in (None, "") else float(v)
    except (TypeError, ValueError):
        return None


def _iso(ms: int) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(ms / 1000))


__all__ = [
    "BASE_URL",
    "OkxError",
    "OkxHttp",
    "SUPPORTED_SOURCES",
    "ensure_history",
    "fetch_book",
    "fetch_candles_history",
    "fetch_candles_recent",
    "fetch_funding_rate",
    "fetch_instruments",
    "fetch_mark_price",
    "fetch_open_interest",
    "fetch_ticker",
    "parse_candle",
    "snapshot_metrics",
]
