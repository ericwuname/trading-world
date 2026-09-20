"""数据层（SQLite 行情库 + OKX 拉取 + 自生成数据）的回归测试。

这一层的价值全在**口径**，所以测试重点不是「跑得通」，而是：

  · 三源共用一套 schema（否则跨源比对失去意义）；
  · **入库后不覆盖**（否则「当时的决策看到哪版数据」没有答案）；
  · **默认过滤未确认的 K 线**（否则策略读到未来）；
  · 分页边界**不重不漏**、且**不前进必须报错**（静默死循环最危险）；
  · **拉取必须记账**（失败也要记）；
  · 合成数据的 OHLC **必须有影线**（否则 K 线退化成折线）；
  · 随机游走基准**漂移严格为 0**（否则丢掉「无优势」这个保证）。
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tw.marketdb import (  # noqa: E402
    SOURCE_OKX,
    SOURCE_SYNTHETIC,
    Candle,
    FetchRecord,
    MarketStore,
    snapshot_id,
)
from tw.okx_data import (  # noqa: E402
    OkxError,
    OkxHttp,
    ensure_history,
    fetch_candles_history,
    fetch_candles_recent,
    parse_candle,
)
from tw.synthetic import (  # noqa: E402
    BAR_MS,
    GenerateSpec,
    available_regimes,
    default_test_set,
    gen_random_walk,
    gen_trending,
    generate_into_store,
    prices_to_candles,
)


def _store() -> MarketStore:
    return MarketStore(":memory:")


def _c(ts: int, o: float, h: float, lo: float, cl: float, vol: float = 1.0,
       confirm: int = 1) -> Candle:
    """方便地造一根 K 线（``Candle`` 是 slots 类，没有字段默认值）。"""
    return Candle(ts=ts, open=o, high=h, low=lo, close=cl,
                  volume=vol, confirm=confirm)


def _mk(n: int, start: int = 1_700_000_000_000, step: int = 3_600_000) -> list[Candle]:
    return [
        Candle(ts=start + i * step, open=100.0 + i, high=101.0 + i,
               low=99.0 + i, close=100.5 + i, volume=10.0 + i, confirm=1)
        for i in range(n)
    ]

def _okx_row(ts: int, px: float, confirm: str = "1") -> list[str]:
    return [str(ts), f"{px}", f"{px + 1}", f"{px - 1}", f"{px + 0.5}",
            "12.5", "1250", "125000", confirm]


# ==========================================================================
# Candle / schema
# ==========================================================================


class TestCandle(unittest.TestCase):
    def test_合法K线通过(self) -> None:
        c = Candle(ts=1, open=100, high=102, low=98, close=101, volume=1.0)
        self.assertEqual(c.confirm, 1)

    def test_high小于low报错(self) -> None:
        with self.assertRaises(ValueError) as cm:
            Candle(ts=1, open=100, high=99, low=101, close=100, volume=1.0)
        self.assertIn("high < low", str(cm.exception))

    def test_open越界报错(self) -> None:
        with self.assertRaises(ValueError) as cm:
            Candle(ts=1, open=200, high=102, low=98, close=100, volume=1.0)
        self.assertIn("open 越界", str(cm.exception))

    def test_close越界报错(self) -> None:
        with self.assertRaises(ValueError) as cm:
            Candle(ts=1, open=100, high=102, low=98, close=300, volume=1.0)
        self.assertIn("close 越界", str(cm.exception))


# ==========================================================================
# 写与读
# ==========================================================================


class TestUpsert(unittest.TestCase):
    def test_写入并读回(self) -> None:
        with _store() as s:
            n = s.upsert_candles(SOURCE_OKX, "BTC-USDT", "1H", _mk(5))
            self.assertEqual(n, 5)
            ser = s.load_candles(SOURCE_OKX, "BTC-USDT", "1H")
            self.assertEqual(len(ser), 5)
            np.testing.assert_array_equal(ser.timestamp,
                                          [1_700_000_000_000 + i * 3_600_000 for i in range(5)])
            np.testing.assert_allclose(ser.close, [100.5 + i for i in range(5)])

    def test_重复写入不覆盖(self) -> None:
        """⭐ 这是「入库后不覆盖」那条约束的核心测试。"""
        with _store() as s:
            s.upsert_candles(SOURCE_OKX, "BTC-USDT", "1H", _mk(3))
            # 同一批 ts，但价格全改了
            changed = [
                Candle(ts=1_700_000_000_000 + i * 3_600_000, open=9.0, high=9.5,
                       low=8.5, close=9.2, volume=1.0, confirm=1)
                for i in range(3)
            ]
            n = s.upsert_candles(SOURCE_OKX, "BTC-USDT", "1H", changed)
            self.assertEqual(n, 0, "重复写入不该生效")
            ser = s.load_candles(SOURCE_OKX, "BTC-USDT", "1H")
            np.testing.assert_allclose(ser.close, [100.5, 101.5, 102.5],
                                       err_msg="原值必须原样保留")

    def test_覆盖开关生效(self) -> None:
        with _store() as s:
            s.upsert_candles(SOURCE_OKX, "BTC-USDT", "1H", _mk(3))
            changed = [
                Candle(ts=1_700_000_000_000 + i * 3_600_000, open=9.0, high=9.5,
                       low=8.5, close=9.2, volume=1.0, confirm=1)
                for i in range(3)
            ]
            s.upsert_candles(SOURCE_OKX, "BTC-USDT", "1H", changed, overwrite=True)
            ser = s.load_candles(SOURCE_OKX, "BTC-USDT", "1H")
            np.testing.assert_allclose(ser.close, [9.2, 9.2, 9.2])

    def test_空输入返回0(self) -> None:
        with _store() as s:
            self.assertEqual(s.upsert_candles(SOURCE_OKX, "X", "1H", []), 0)

    def test_dict形式也能写(self) -> None:
        with _store() as s:
            n = s.upsert_candles(SOURCE_OKX, "X", "1H", [
                {"ts": 1, "open": 1, "high": 2, "low": 0.5, "close": 1.5,
                 "volume": 3, "confirm": 1}
            ])
            self.assertEqual(n, 1)

    def test_三源共用同一张表(self) -> None:
        """⭐ 三源必须能存在一起且互不干扰，否则跨源对比失去意义。"""
        with _store() as s:
            s.upsert_candles(SOURCE_OKX, "BTC-USDT", "1H", _mk(3))
            s.upsert_candles(SOURCE_SYNTHETIC, "BTC-USDT", "1H", _mk(7))
            self.assertEqual(len(s.load_candles(SOURCE_OKX, "BTC-USDT", "1H")), 3)
            self.assertEqual(len(s.load_candles(SOURCE_SYNTHETIC, "BTC-USDT", "1H")), 7)
            self.assertEqual(len(s.list_instruments()), 2)

    def test_非法K线在写入前就被挡住(self) -> None:
        """⭐ Candle 是 slots 类，构造后改字段**不会**重跑 __post_init__。
        所以写入路径必须自己再验一次——只在构造时验是会漏的。"""
        with _store() as s:
            bad = [_c(1, 1, 2, 0.5, 1.5)]
            bad[0].high = 0.1  # 绕过 __post_init__ 手工破坏
            with self.assertRaises(ValueError):
                s.upsert_candles(SOURCE_OKX, "X", "1H", bad)
            self.assertEqual(s.stats()["candles"], 0, "坏数据不该部分写入")

    def test_负成交量被挡住(self) -> None:
        with _store() as s:
            bad = [_c(1, 1, 2, 0.5, 1.5)]
            bad[0].volume = -1.0
            with self.assertRaises(ValueError):
                s.upsert_candles(SOURCE_OKX, "X", "1H", bad)

    def test_dict形式也走校验(self) -> None:
        with _store() as s:
            with self.assertRaises(ValueError):
                s.upsert_candles(SOURCE_OKX, "X", "1H", [
                    {"ts": 1, "open": 1, "high": 0.1, "low": 0.5, "close": 1.5,
                     "volume": 1}
                ])


# ==========================================================================
# confirm 过滤（防读到未来）
# ==========================================================================


class TestConfirmFilter(unittest.TestCase):
    def test_默认过滤未确认根(self) -> None:
        """⭐ 未走完的 K 线 = 未来数据，默认必须挡住。"""
        with _store() as s:
            rows = _mk(5)
            rows[-1] = Candle(ts=rows[-1].ts, open=100, high=999, low=1,
                              close=888, volume=1, confirm=0)
            s.upsert_candles(SOURCE_OKX, "X", "1H", rows)
            got = s.load_candles(SOURCE_OKX, "X", "1H")  # 默认 confirmed_only=True
            self.assertEqual(len(got), 4)
            self.assertNotIn(888.0, got.close.tolist())

    def test_显式允许时可读到(self) -> None:
        with _store() as s:
            rows = _mk(5)
            rows[-1] = Candle(ts=rows[-1].ts, open=100, high=999, low=1,
                              close=888, volume=1, confirm=0)
            s.upsert_candles(SOURCE_OKX, "X", "1H", rows)
            got = s.load_candles(SOURCE_OKX, "X", "1H", confirmed_only=False)
            self.assertEqual(len(got), 5)
            self.assertIn(888.0, got.close.tolist())

    def test_覆盖度分别统计(self) -> None:
        with _store() as s:
            rows = _mk(5)
            rows[-1] = Candle(ts=rows[-1].ts, open=100, high=101, low=99,
                              close=100, volume=1, confirm=0)
            s.upsert_candles(SOURCE_OKX, "X", "1H", rows)
            self.assertEqual(s.coverage(SOURCE_OKX, "X", "1H").n_rows, 5)
            cov = s.coverage(SOURCE_OKX, "X", "1H", confirmed_only=True)
            self.assertEqual(cov.n_rows, 4)
            self.assertEqual(s.coverage(SOURCE_OKX, "X", "1H").n_unconfirmed, 1)

    def test_清理未确认根(self) -> None:
        with _store() as s:
            rows = _mk(5)
            rows[-1] = Candle(ts=rows[-1].ts, open=100, high=101, low=99,
                              close=100, volume=1, confirm=0)
            s.upsert_candles(SOURCE_OKX, "X", "1H", rows)
            n = s.prune_unconfirmed()
            self.assertEqual(n, 1)
            self.assertEqual(s.coverage(SOURCE_OKX, "X", "1H").n_rows, 4)


# ==========================================================================
# 范围查询
# ==========================================================================


class TestRangeQuery(unittest.TestCase):
    def test_时间范围(self) -> None:
        with _store() as s:
            s.upsert_candles(SOURCE_OKX, "X", "1H", _mk(10))
            lo = 1_700_000_000_000 + 3 * 3_600_000
            hi = 1_700_000_000_000 + 6 * 3_600_000
            got = s.load_candles(SOURCE_OKX, "X", "1H", from_ts=lo, to_ts=hi)
            self.assertEqual(len(got), 4)

    def test_limit取最近而非最早(self) -> None:
        """⭐ limit 必须取**最近**的 N 根——拉「最新行情」时取最早那批是经典的错。"""
        with _store() as s:
            s.upsert_candles(SOURCE_OKX, "X", "1H", _mk(10))
            got = s.load_candles(SOURCE_OKX, "X", "1H", limit=3)
            self.assertEqual(len(got), 3)
            # 应是最新的三根（i = 7,8,9）
            np.testing.assert_allclose(got.close, [107.5, 108.5, 109.5])
            self.assertTrue(np.all(np.diff(got.timestamp) > 0), "仍须升序")

    def test_空库返回空序列(self) -> None:
        with _store() as s:
            got = s.load_candles(SOURCE_OKX, "NOPE", "1H")
            self.assertTrue(got.is_empty)
            self.assertEqual(len(got), 0)

    def test_覆盖度描述(self) -> None:
        with _store() as s:
            s.upsert_candles(SOURCE_OKX, "X", "1H", _mk(25))
            cov = s.coverage(SOURCE_OKX, "X", "1H")
            self.assertEqual(cov.n_rows, 25)
            self.assertIn("25 根", cov.describe())
            self.assertIn("空", s.coverage(SOURCE_OKX, "Y", "1H").describe())


# ==========================================================================
# 快照类
# ==========================================================================


class TestMetrics(unittest.TestCase):
    def test_写入读回(self) -> None:
        with _store() as s:
            s.upsert_metrics(SOURCE_OKX, "BTC-USDT-SWAP", "funding_rate", [
                (1000, 0.0001, {"nextFundingTime": 2000}),
                (2000, 0.0002, None),
            ])
            ts, vals = s.load_metrics(SOURCE_OKX, "BTC-USDT-SWAP", "funding_rate")
            np.testing.assert_array_equal(ts, [1000, 2000])
            np.testing.assert_allclose(vals, [0.0001, 0.0002])

    def test_同ts覆盖(self) -> None:
        with _store() as s:
            s.upsert_metrics(SOURCE_OKX, "X", "funding_rate", [(1000, 0.1, None)])
            s.upsert_metrics(SOURCE_OKX, "X", "funding_rate", [(1000, 0.9, None)])
            _, vals = s.load_metrics(SOURCE_OKX, "X", "funding_rate")
            np.testing.assert_allclose(vals, [0.9])

    def test_None值变nan(self) -> None:
        with _store() as s:
            s.upsert_metrics(SOURCE_OKX, "X", "open_interest", [(1000, None, None)])
            _, vals = s.load_metrics(SOURCE_OKX, "X", "open_interest")
            self.assertTrue(np.isnan(vals[0]))

    def test_盘口落库(self) -> None:
        with _store() as s:
            s.upsert_book(SOURCE_OKX, "BTC-USDT", 123, 5,
                          [[100.0, 1.5, 2]], [[100.5, 1.2, 3]])
            self.assertEqual(s.stats()["books"], 1)


# ==========================================================================
# 拉取账本
# ==========================================================================


class TestFetchLedger(unittest.TestCase):
    def test_记账并回读(self) -> None:
        with _store() as s:
            fid = s.log_fetch(FetchRecord(
                source=SOURCE_OKX, inst_id="BTC-USDT", bar="1H",
                endpoint="/api/v5/market/candles", n_rows=300, pages=2, ok=True,
            ))
            self.assertGreater(fid, 0)
            hist = s.fetch_history(1)
            self.assertEqual(hist[0]["n_rows"], 300)
            self.assertEqual(hist[0]["endpoint"], "/api/v5/market/candles")

    def test_失败也记账(self) -> None:
        """⭐ 失败不记账 = 半年后不知道「当时是不是拉失败了」。"""
        with _store() as s:
            s.log_fetch(FetchRecord(
                source=SOURCE_OKX, inst_id="X", bar="1H", endpoint="/x",
                ok=False, error="429 rate limited",
            ))
            rec = s.fetch_history(1)[0]
            self.assertEqual(rec["ok"], 0)
            self.assertIn("429", rec["error"])

    def test_账本条数计入stats(self) -> None:
        with _store() as s:
            for i in range(3):
                s.log_fetch(FetchRecord(source=SOURCE_OKX, inst_id="X",
                                        bar="1H", endpoint=f"/e{i}"))
            self.assertEqual(s.stats()["fetches"], 3)


# ==========================================================================
# OKX 解析与分页
# ==========================================================================


class TestOkxParsing(unittest.TestCase):
    def test_解析标准行(self) -> None:
        c = parse_candle(_okx_row(1_700_000_000_000, 60_000.0))
        self.assertEqual(c.ts, 1_700_000_000_000)
        self.assertEqual(c.open, 60_000.0)
        self.assertEqual(c.confirm, 1)

    def test_未确认根被识别(self) -> None:
        c = parse_candle(_okx_row(1, 100.0, confirm="0"))
        self.assertEqual(c.confirm, 0)

    def test_行太短报错(self) -> None:
        with self.assertRaises(ValueError):
            parse_candle(["1", "2", "3"])

    def test_缺confirm时默认1(self) -> None:
        c = parse_candle(["1", "100", "101", "99", "100", "12.5"])
        self.assertEqual(c.confirm, 1)

    def test_行太短报错2(self) -> None:
        with self.assertRaises(ValueError):
            parse_candle(["1"])

    def test_空volume容错记0(self) -> None:
        """⭐ OKX 在某些合约/停牌时段返回空串。
        直接 float('') 会抛错把**整次拉取打断**，而空量不代表这行无意义。"""
        c = parse_candle(["1", "100", "101", "99", "100", ""])
        self.assertEqual(c.volume, 0.0)
        self.assertEqual(c.close, 100.0)

    def test_空价格必须报错(self) -> None:
        """价格空才是真的没意义——不能容错，要跳过这一行。"""
        with self.assertRaises(ValueError) as cm:
            parse_candle(["1", "", "101", "99", "100", "1"])
        self.assertIn("open", str(cm.exception))

    def test_空confirm容错为1(self) -> None:
        c = parse_candle(["1", "100", "101", "99", "100", "1", "2", "3", ""])
        self.assertEqual(c.confirm, 1)


class _FakeHttp:
    """假 HTTP：按序返回预设响应，并记录请求 URL。"""

    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.urls: list[str] = []

    def __call__(self, url: str, timeout: float) -> bytes:
        self.urls.append(url)
        if not self.responses:
            raise AssertionError(f"没有更多预设响应了，但收到了请求：{url}")
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return json.dumps(r).encode()


class TestOkxHttp(unittest.TestCase):
    def test_解析data数组(self) -> None:
        fake = _FakeHttp([{"code": "0", "data": [{"x": 1}]}])
        h = OkxHttp(fetch=fake)
        self.assertEqual(h.get("/api/v5/public/time"), [{"x": 1}])

    def test_非0code抛错(self) -> None:
        fake = _FakeHttp([{"code": "51001", "msg": "Instrument ID does not exist"}])
        h = OkxHttp(fetch=fake)
        with self.assertRaises(OkxError) as cm:
            h.get("/api/v5/market/candles")
        self.assertIn("51001", str(cm.exception))

    def test_业务错误不重试(self) -> None:
        """参数错就是错，重试三次只是浪费三倍配额。"""
        fake = _FakeHttp([{"code": "51001", "msg": "bad"}])
        h = OkxHttp(fetch=fake, max_retries=3, backoff=0.001)
        with self.assertRaises(OkxError):
            h.get("/x")
        self.assertEqual(len(fake.urls), 1, "业务错误只应请求一次")

    def test_网络错误会重试(self) -> None:
        fake = _FakeHttp([OSError("boom"), {"code": "0", "data": []}])
        h = OkxHttp(fetch=fake, max_retries=3, backoff=0.001)
        self.assertEqual(h.get("/x"), [])
        self.assertEqual(len(fake.urls), 2)

    def test_耗尽重试后抛错(self) -> None:
        fake = _FakeHttp([OSError("a"), OSError("b")])
        h = OkxHttp(fetch=fake, max_retries=2, backoff=0.001)
        with self.assertRaises(OkxError) as cm:
            h.get("/x")
        self.assertIn("重试", str(cm.exception))

    def test_none参数不进url(self) -> None:
        fake = _FakeHttp([{"code": "0", "data": []}])
        h = OkxHttp(fetch=fake)
        h.get("/x", {"a": 1, "b": None, "c": ""})
        self.assertIn("a=1", fake.urls[0])
        self.assertNotIn("b=", fake.urls[0])
        self.assertNotIn("c=", fake.urls[0])


class TestOkxCandles(unittest.TestCase):
    def test_近期K线返回升序(self) -> None:
        """OKX 返回是**倒序**的；不排回来会让所有下游逻辑反着走。"""
        rows = [_okx_row(3000, 30.0), _okx_row(2000, 20.0), _okx_row(1000, 10.0)]
        fake = _FakeHttp([{"code": "0", "data": rows}])
        got = fetch_candles_recent(OkxHttp(fetch=fake), "BTC-USDT", "1H")
        self.assertEqual([c.ts for c in got], [1000, 2000, 3000])

    def test_limit上限300(self) -> None:
        fake = _FakeHttp([{"code": "0", "data": []}])
        fetch_candles_recent(OkxHttp(fetch=fake), "X", "1H", limit=9999)
        self.assertIn("limit=300", fake.urls[0])

    def test_历史K线limit上限100(self) -> None:
        fake = _FakeHttp([{"code": "0", "data": []}])
        fetch_candles_history(OkxHttp(fetch=fake), "X", "1H", limit=9999)
        self.assertIn("limit=100", fake.urls[0])

    def test_before与after不能同时传(self) -> None:
        fake = _FakeHttp([
            {"code": "0", "data": []},
            {"code": "0", "data": []},
        ])
        h = OkxHttp(fetch=fake)
        # 分别传时都该出现在 URL 里
        fetch_candles_history(h, "X", "1H", after=1000)
        self.assertIn("after=1000", fake.urls[0])
        fetch_candles_history(h, "X", "1H", before=2000)
        self.assertIn("before=2000", fake.urls[1])
        self.assertNotIn("after=", fake.urls[1])


class TestEnsureHistory(unittest.TestCase):
    def _recent(self, n: int, start: int, step: int = 3_600_000) -> list[list[str]]:
        return [_okx_row(start + i * step, 100.0 + i)
                for i in reversed(range(n))]

    def test_首次拉取落库(self) -> None:
        with _store() as s:
            now_ms = int(__import__("time").time() * 1000)
            rows = self._recent(5, now_ms - 5 * 3_600_000)
            fake = _FakeHttp([
                {"code": "0", "data": rows},          # candles
                {"code": "0", "data": []},            # history-candles：空，停
            ])
            cov = ensure_history(s, "BTC-USDT", "1H", days=1,
                                 http=OkxHttp(fetch=fake, backoff=0.001))
            self.assertEqual(cov.n_rows, 5)
            self.assertEqual(s.stats()["fetches"], 1)

    def test_库已覆盖则跳过联网(self) -> None:
        """⭐ 这是「按需拉取」的核心：够用就不拉。"""
        with _store() as s:
            now_ms = int(__import__("time").time() * 1000)
            # 先塞一批覆盖 400 天前的数据
            old = now_ms - 400 * 86_400_000
            s.upsert_candles(SOURCE_OKX, "BTC-USDT", "1H",
                             _mk(10, start=old, step=3_600_000))
            fake = _FakeHttp([])  # 任何请求都会 AssertionError
            cov = ensure_history(s, "BTC-USDT", "1H", days=365,
                                 http=OkxHttp(fetch=fake, backoff=0.001))
            self.assertEqual(len(fake.urls), 0, "不该发出任何请求")
            rec = s.fetch_history(1)[0]
            self.assertEqual(rec["endpoint"], "(cached)")

    def test_分页不重不漏(self) -> None:
        """⭐ 分页边界是第一类易错点：重复会虚高行数，遗漏会丢数据。"""
        with _store() as s:
            now_ms = int(__import__("time").time() * 1000)
            step = 3_600_000
            # 近期端点给最新 3 根
            recent_ts = [now_ms - i * step for i in range(3)]
            # 历史端点分两批，每批 3 根，紧接在 recent 之前
            h1 = [recent_ts[-1] - (i + 1) * step for i in range(3)]
            h2 = [h1[-1] - (i + 1) * step for i in range(3)]

            fake = _FakeHttp([
                {"code": "0", "data": self._recent(3, recent_ts[-1])},
                {"code": "0", "data": [_okx_row(t, 50.0) for t in reversed(h1)]},
                {"code": "0", "data": [_okx_row(t, 40.0) for t in reversed(h2)]},
                {"code": "0", "data": []},
            ])
            cov = ensure_history(s, "BTC-USDT", "1H", days=99,
                                 http=OkxHttp(fetch=fake, backoff=0.001))
            self.assertEqual(cov.n_rows, 9, f"应为 3+3+3=9 根，实际 {cov.n_rows}")
            # 无重复：主键就是 ts，行数=9 就说明没重
            ser = s.load_candles(SOURCE_OKX, "BTC-USDT", "1H")
            self.assertEqual(len(set(ser.timestamp.tolist())), 9)

    def test_翻页游标减1ms(self) -> None:
        """next after 必须是上批最旧 ts − 1，否则边界那根会重复出现。"""
        with _store() as s:
            now_ms = int(__import__("time").time() * 1000)
            step = 3_600_000
            recent_ts = [now_ms - i * step for i in range(2)]
            h1 = [recent_ts[-1] - (i + 1) * step for i in range(2)]
            fake = _FakeHttp([
                {"code": "0", "data": self._recent(2, recent_ts[-1])},
                {"code": "0", "data": [_okx_row(t, 50.0) for t in reversed(h1)]},
                {"code": "0", "data": []},
            ])
            ensure_history(s, "BTC-USDT", "1H", days=99,
                           http=OkxHttp(fetch=fake, backoff=0.001))
            # 第三次请求应是 history-candles，且 after = 最旧 ts - 1
            self.assertIn(f"after={min(h1) - 1}", fake.urls[-1])

    def test_分页不前进必须报错(self) -> None:
        """⭐ 静默死循环比报错危险得多。"""
        with _store() as s:
            now_ms = int(__import__("time").time() * 1000)
            step = 3_600_000
            recent_ts = [now_ms - i * step for i in range(2)]
            same = [recent_ts[-1] - step]  # 每次都返回同一根
            fake = _FakeHttp([
                {"code": "0", "data": self._recent(2, recent_ts[-1])},
                {"code": "0", "data": [_okx_row(t, 50.0) for t in same]},
                {"code": "0", "data": [_okx_row(t, 50.0) for t in same]},
            ])
            with self.assertRaises(OkxError) as cm:
                ensure_history(s, "BTC-USDT", "1H", days=9999,
                               http=OkxHttp(fetch=fake, backoff=0.001))
            self.assertIn("分页没有前进", str(cm.exception))

    def test_失败时账本记为not_ok(self) -> None:
        with _store() as s:
            fake = _FakeHttp([OSError("network down"), OSError("network down")])
            with self.assertRaises(OkxError):
                ensure_history(s, "BTC-USDT", "1H", days=1,
                               http=OkxHttp(fetch=fake, max_retries=2, backoff=0.001))
            rec = s.fetch_history(1)[0]
            self.assertEqual(rec["ok"], 0)
            self.assertTrue(rec["error"])

    def test_未确认根不落库(self) -> None:
        with _store() as s:
            now_ms = int(__import__("time").time() * 1000)
            step = 3_600_000
            rows = [_okx_row(now_ms, 100.0), _okx_row(now_ms - step, 99.0, "0")]
            fake = _FakeHttp([
                {"code": "0", "data": rows},
                {"code": "0", "data": []},
            ])
            cov = ensure_history(s, "BTC-USDT", "1H", days=1,
                                 http=OkxHttp(fetch=fake, backoff=0.001))
            self.assertEqual(cov.n_rows, 1, "confirm=0 的那根不该入库")


# ==========================================================================
# 快照 ID
# ==========================================================================


class TestSnapshotId(unittest.TestCase):
    def test_同库同状态指纹一致(self) -> None:
        with _store() as s:
            s.upsert_candles(SOURCE_OKX, "X", "1H", _mk(5))
            a = snapshot_id(s, SOURCE_OKX, "X", "1H")
            b = snapshot_id(s, SOURCE_OKX, "X", "1H")
            self.assertEqual(a, b)
            self.assertEqual(len(a), 16)

    def test_数据变了指纹就变(self) -> None:
        with _store() as s:
            s.upsert_candles(SOURCE_OKX, "X", "1H", _mk(5))
            a = snapshot_id(s, SOURCE_OKX, "X", "1H")
            s.upsert_candles(SOURCE_OKX, "X", "1H", _mk(4, start=1_800_000_000_000))
            b = snapshot_id(s, SOURCE_OKX, "X", "1H")
            self.assertNotEqual(a, b)


# ==========================================================================
# 合成数据
# ==========================================================================


class TestSynthetic(unittest.TestCase):
    def test_可用regime清单(self) -> None:
        self.assertIn("normal", available_regimes())
        self.assertIn("trend", available_regimes())
        self.assertIn("vol_cluster", available_regimes())

    def test_随机游走漂移严格为0(self) -> None:
        """⭐ 「零可预测性」这个保证就靠漂移为 0。有漂移就丢掉整个价值。"""
        spec = GenerateSpec(inst_id="S", bar="1H", n_bars=20000, seed=1,
                            start_price=100.0, annual_vol=0.65)
        px = gen_random_walk(spec)
        total_ret = float(np.log(px[-1] / px[0]))
        n_years = 20000 / (365 * 24)
        # 期望对数收益 = 0（不是 n*sigma^2/2：那是伊藤项，已在生成里隐含）
        # 允许的范围按「零漂移下的随机波动」给：2 个标准差
        per_bar = 0.65 / np.sqrt(365 * 24)
        sd_total = per_bar * np.sqrt(20000)
        self.assertLess(abs(total_ret), 3 * sd_total,
                        f"总对数收益 {total_ret:.4f} 偏离 0 太远"
                        f"（n_years={n_years:.1f}，sd={sd_total:.4f}）")

    def test_随机游走同种子可复现(self) -> None:
        spec = GenerateSpec(inst_id="S", bar="1H", n_bars=100, seed=7)
        np.testing.assert_allclose(gen_random_walk(spec), gen_random_walk(spec))

    def test_不同种子不同结果(self) -> None:
        a = gen_random_walk(GenerateSpec(inst_id="S", bar="1H", n_bars=100, seed=1))
        b = gen_random_walk(GenerateSpec(inst_id="S", bar="1H", n_bars=100, seed=2))
        self.assertFalse(np.allclose(a, b))

    def test_趋势版有方向(self) -> None:
        spec = GenerateSpec(inst_id="S", bar="1H", n_bars=5000, seed=5,
                            start_price=100.0)
        px = gen_trending(spec)
        self.assertNotAlmostEqual(float(np.log(px[-1] / px[0])), 0.0, places=3)

    def test_波动聚集版有聚集性(self) -> None:
        """自相关应为正——没有聚集性就测不出「波动率突变时策略失效」。"""
        from tw.synthetic import gen_vol_clustered

        spec = GenerateSpec(inst_id="S", bar="1H", n_bars=8000, seed=3,
                            start_price=100.0)
        px = gen_vol_clustered(spec)
        r = np.diff(np.log(px))
        a = np.abs(r)
        a = a - a.mean()
        ac1 = float(np.sum(a[1:] * a[:-1]) / np.sum(a * a))
        self.assertGreater(ac1, 0.05, f"|r| 的一阶自相关 = {ac1:.4f}，聚集性太弱")

    def test_不支持的regime报错(self) -> None:
        with _store() as s:
            with self.assertRaises(ValueError):
                generate_into_store(s, GenerateSpec(inst_id="S", regime="nope"))

    def test_不支持的bar报错(self) -> None:
        with _store() as s:
            with self.assertRaises(ValueError):
                generate_into_store(s, GenerateSpec(inst_id="S", bar="7m"))


class TestSyntheticCandles(unittest.TestCase):
    def test_必须有影线(self) -> None:
        """⭐ OHLC 若退化成 high==low==close，蜡烛图就变成折线了。"""
        spec = GenerateSpec(inst_id="S", bar="1H", n_bars=50, seed=1,
                            start_ts=1_700_000_000_000)
        px = gen_random_walk(spec)
        bars = prices_to_candles(px, spec)
        n_with_wick = sum(
            1 for b in bars
            if b.high > max(b.open, b.close) or b.low < min(b.open, b.close)
        )
        self.assertEqual(n_with_wick, len(bars),
                         "每根都必须有影线（至少一侧）")

    def test_high_low是窗口内极值(self) -> None:
        spec = GenerateSpec(inst_id="S", bar="1H", n_bars=30, seed=2,
                            start_ts=1_700_000_000_000)
        bars = prices_to_candles(gen_random_walk(spec), spec)
        for b in bars:
            self.assertGreaterEqual(b.high, max(b.open, b.close))
            self.assertLessEqual(b.low, min(b.open, b.close))

    def test_影线不是靠乘性噪声凑出来的(self) -> None:
        """⭐ 回归守卫（对应 M63）。

        ``high`` 必须真的来自**窗口内路径的极值**，而不是"在 max(o,c) 上
        乘一个极小的正噪声"凑出来的假影线。

        这两者在下述弱测试下**无法区分**：只看"high > max(o,c) 吗"，
        乘噪声那条路也能 49/49 全过——测试于是变成摆设（M63 就是这么
        Survivor 下来的）。真正的判据是：影线**不能太小**。
        乘性噪声只有 ~2e-4 的相对量级，而路径极值在中位数以上的根里
        会显著超出 max(o,c)。
        """
        spec = GenerateSpec(inst_id="S", bar="1H", n_bars=200, seed=5,
                            start_ts=1_700_000_000_000)
        bars = prices_to_candles(gen_random_walk(spec), spec)
        rel_hi = [(b.high - max(b.open, b.close)) / max(b.open, b.close)
                  for b in bars]
        rel_lo = [(min(b.open, b.close) - b.low) / max(b.open, b.close)
                  for b in bars]
        # 乘性噪声只有 ~2e-4 的相对量级；真正的路径极值会**明显**大于它。
        # ⭐ 决定性判据：必须有相当比例的根，影线显著大于噪声地板。
        # 只有当 high/low 真的取了窗口内极值时才会出现这种大影线。
        # （阈值 1e-3 是实测标定的：真实数据约有 2/3 的根超过它，
        # 而 M63 注入后只剩 ~0。用中位数做辅助判据不可靠——
        # 实测中位数本身就在 1e-3 附近，会随种子漂移。）
        big_hi = sum(1 for r in rel_hi if r > 1e-3)
        big_lo = sum(1 for r in rel_lo if r > 1e-3)
        self.assertGreater(
            big_hi, len(bars) * 0.2,
            f"只有 {big_hi}/{len(bars)} 根的上影线超过噪声地板 1e-3——"
            f"high 很可能只取了 max(open, close) 再乘噪声（M63）",
        )
        self.assertGreater(
            big_lo, len(bars) * 0.2,
            f"只有 {big_lo}/{len(bars)} 根的下影线超过噪声地板 1e-3——"
            f"low 很可能只取了 min(open, close) 再乘噪声",
        )

    def test_时间戳等间隔(self) -> None:
        spec = GenerateSpec(inst_id="S", bar="1H", n_bars=10, seed=1,
                            start_ts=1_700_000_000_000)
        bars = prices_to_candles(gen_random_walk(spec), spec)
        d = np.diff([b.ts for b in bars])
        np.testing.assert_array_equal(d, [BAR_MS["1H"]] * (len(bars) - 1))

    def test_确认位都是1(self) -> None:
        spec = GenerateSpec(inst_id="S", bar="1H", n_bars=10, seed=1,
                            start_ts=1_700_000_000_000)
        bars = prices_to_candles(gen_random_walk(spec), spec)
        self.assertTrue(all(b.confirm == 1 for b in bars))

    def test_太短报错(self) -> None:
        spec = GenerateSpec(inst_id="S", bar="1H", n_bars=1)
        with self.assertRaises(ValueError):
            prices_to_candles(np.array([1.0]), spec)


class TestSyntheticIntoStore(unittest.TestCase):
    def test_落库并可读回(self) -> None:
        with _store() as s:
            spec = GenerateSpec(inst_id="SYNTH", bar="1H", n_bars=100,
                                seed=0, start_ts=1_700_000_000_000)
            out = generate_into_store(s, spec)
            self.assertEqual(out["n_generated"], 99)
            ser = s.load_candles(SOURCE_SYNTHETIC, "SYNTH", "1H")
            self.assertEqual(len(ser), 99)

    def test_账本记下生成参数(self) -> None:
        """⭐ 没记参数的数据等于没数据——无法重造。"""
        with _store() as s:
            spec = GenerateSpec(inst_id="SYNTH", bar="1H", n_bars=50,
                                seed=42, regime="trend",
                                start_ts=1_700_000_000_000)
            out = generate_into_store(s, spec)
            rec = s.fetch_history(1)[0]
            # key 是「怎么重造」的答案，必须完整落在账本里
            self.assertEqual(rec["endpoint"], out["key"])
            for token in ("seed=42", "trend", "n=50", "bar=1H", "p0="):
                self.assertIn(token, rec["endpoint"], f"账本缺 {token}")

    def test_同参数重造结果一致(self) -> None:
        with _store() as s1, _store() as s2:
            spec = GenerateSpec(inst_id="S", bar="1H", n_bars=60, seed=9,
                                start_ts=1_700_000_000_000)
            generate_into_store(s1, spec)
            generate_into_store(s2, spec)
            a = s1.load_candles(SOURCE_SYNTHETIC, "S", "1H")
            b = s2.load_candles(SOURCE_SYNTHETIC, "S", "1H")
            np.testing.assert_allclose(a.close, b.close)
            np.testing.assert_array_equal(a.timestamp, b.timestamp)


class TestDefaultTestSet(unittest.TestCase):
    def test_三件套齐全(self) -> None:
        specs = default_test_set()
        self.assertEqual(len(specs), 3)
        self.assertEqual({s.regime for s in specs}, {"normal", "trend", "vol_cluster"})

    def test_三件套inst_id各不相同(self) -> None:
        """⭐ 三份世界时间轴完全相同，若共用 inst_id，主键会撞、
        后两份被 INSERT OR IGNORE **静默丢掉**（实测踩到只剩一份）。"""
        specs = default_test_set()
        self.assertEqual(len({s.inst_id for s in specs}), 3)
        for s in specs:
            self.assertIn(s.regime, s.inst_id)

    def test_三件套共享初始条件(self) -> None:
        """同一 seed / 同一起点 / 同一价格 ⇒ 三种世界，差别只在机制。"""
        specs = default_test_set(seed=11, start_price=1234.0)
        self.assertEqual({s.seed for s in specs}, {11})
        self.assertEqual({s.start_ts for s in specs}, {specs[0].start_ts})
        self.assertEqual({s.start_price for s in specs}, {1234.0})

    def test_三件套都能落库(self) -> None:
        with _store() as s:
            for spec in default_test_set(n_bars=60):
                generate_into_store(s, spec)
            insts = {(r["source"], r["inst_id"]) for r in s.list_instruments()}
            self.assertEqual(len(insts), 3, "三份世界必须共存")


# ==========================================================================
# 持久化
# ==========================================================================


class TestPersistence(unittest.TestCase):
    def test_文件库重开后数据还在(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.sqlite"
            with MarketStore(p) as s:
                s.upsert_candles(SOURCE_OKX, "X", "1H", _mk(5))
            with MarketStore(p) as s:
                self.assertEqual(len(s.load_candles(SOURCE_OKX, "X", "1H")), 5)

    def test_目录自动创建(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a" / "b" / "m.sqlite"
            with MarketStore(p):
                pass
            self.assertTrue(p.exists())


if __name__ == "__main__":
    unittest.main()
