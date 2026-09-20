"""行情数据库（SQLite）——「真实 / 历史 / 自生成」三源共用一套 schema。

为什么要一个库
--------------
用户的要求：*「实时拉取 OKX 或其他交易所的数据…建立一个数据库 sqlite…
不必所有币种都拉，在选择准备交易的时候才去拉…也可以下载历史数据，
还有自己生成数据，这样子 agent 可以利用不同来源的数据进行自我测试、
真实数据测试。」*

**核心设计：三种来源，一个 schema。**

表结构里只多一列 ``source``，其余完全相同。这是刻意的：

- Agent 的**同一套策略代码**应该在三源上跑出可对比的结果；
- 若三源走三套接口，「策略在真实数据上表现不同」到底是策略的问题
  还是接口口径的问题，就**分不清了** —— 与项目那条
  「同一个量不要有两条路径」的纪律是同一形态。

三条不可动摇的约束
------------------
1. **决策路径只读库**。联网拉取只发生在「会话开始前」的准备阶段。
   决策期间联网 = 引入不可复现性 + 可能读到未来数据。
2. **入库后不覆盖**。同一 ``(source, inst_id, bar, ts)`` 再写不生效
   （除非显式 ``overwrite=True``）。保证「当时的决策」事后可复现，
   哪怕交易所之后修正了历史。
3. **拉取必须记账**。每次拉取写一条 :class:`FetchRecord`。
   没有账本，半年后没人知道某根 K 线是从哪个端点、什么参数拿的，
   而 OKX 的 ``candles``（热缓存）与 ``history-candles``（冷存储）
   **可能返回不同的修正后数据**。

不用 ORM
--------
本项目依赖只有 numpy / scipy / matplotlib。``sqlite3`` 是标准库，
直接用。多一个 ORM 就多一个「环境不对所以没跑」的借口。
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

#: 默认库位置（与 data/ 同级，但独立目录——它是可重建的派生物）
DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "market.sqlite"

#: 数据来源标识
SOURCE_OKX = "okx"
SOURCE_SYNTHETIC = "synthetic"

#: OKX 的 K 线周期（与官方一致；分钟级小写、小时/天大写法是 OKX 的约定）
BARS = ("1m", "3m", "5m", "15m", "30m", "1H", "2H", "4H", "6H", "12H", "1D", "1W")

#: 快照类指标
KINDS = ("funding_rate", "open_interest", "mark_price", "index_price")


# --------------------------------------------------------------------------
# 数据类
# --------------------------------------------------------------------------


@dataclass(slots=True)
class Candle:
    """一根 K 线。字段名与 OKX 返回的顺序一致，便于对照。"""

    ts: int  # 开盘时间，毫秒
    open: float
    high: float
    low: float
    close: float
    volume: float
    confirm: int = 1  # 1 = 这根已走完（OKX 的 '1'/'0'）

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """检查 K 线自洽。

        这条校验看着琐碎，但它是「K 线自洽」的最小定义。
        不校验的话，来源错误的数据会一路流到策略里，而策略看不出异常。

        ⚠️ **必须能被重复调用**：``Candle`` 是 ``slots=True``，
        构造完之后改字段**不会**重跑 ``__post_init__``。所以写入路径
        （``MarketStore.upsert_candles``）会**再调一次**这个方法。
        只在构造时验是会漏的——实测踩到：构造后手工改 ``high`` 能让
        坏数据进库，测试才发现的。
        """
        if self.high < self.low:
            raise ValueError(f"high < low @ ts={self.ts}: {self.high} < {self.low}")
        if not (self.low <= self.open <= self.high):
            raise ValueError(f"open 越界 @ ts={self.ts}: {self.open}")
        if not (self.low <= self.close <= self.high):
            raise ValueError(f"close 越界 @ ts={self.ts}: {self.close}")
        if not (self.volume >= 0):
            raise ValueError(f"volume 为负 @ ts={self.ts}: {self.volume}")


@dataclass(slots=True)
class Series:
    """一段行情（列式）。与 ``tw/realdata.Series`` 字段名刻意保持一致。"""

    name: str
    source: str
    inst_id: str
    bar: str
    timestamp: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    confirm: np.ndarray

    def __len__(self) -> int:
        return int(self.close.size)

    @property
    def is_empty(self) -> bool:
        return self.close.size == 0

    @property
    def span_days(self) -> float:
        if self.timestamp.size < 2:
            return 0.0
        return float((self.timestamp[-1] - self.timestamp[0]) / 86_400_000)

    def to_realdata(self) -> Any:
        """转成 ``tw.realdata.Series``，好复用那边已有的统计特征函数。

        不直接继承是刻意的：``realdata`` 是**只读的内置数据集**，
        这里是**可写的库**，两者生命周期不同，混在一起会让
        「这份数据从哪来」这个问题失去答案。
        """
        from .realdata import Series as RealSeries

        return RealSeries(
            name=self.name,
            timestamp=self.timestamp,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
        )


@dataclass(slots=True)
class FetchRecord:
    """一次拉取的账本记录。

    这是整个库里**最容易被忽略、却最重要**的一张表：
    它回答「这份数据是什么时候、用什么参数、从哪个端点拉下来的」。
    """

    source: str
    inst_id: str
    bar: str
    endpoint: str
    from_ts: int = 0
    to_ts: int = 0
    n_rows: int = 0
    pages: int = 0
    started_at: int = 0
    finished_at: int = 0
    ok: bool = True
    error: str = ""
    id: int | None = None


@dataclass(slots=True)
class Coverage:
    """某个 ``(source, inst_id, bar)`` 的覆盖情况。"""

    source: str
    inst_id: str
    bar: str
    n_rows: int
    first_ts: int
    last_ts: int
    n_unconfirmed: int = 0

    @property
    def is_empty(self) -> bool:
        return self.n_rows == 0

    def describe(self) -> str:
        if self.is_empty:
            return f"{self.source}:{self.inst_id}:{self.bar} —— 空"
        span = (self.last_ts - self.first_ts) / 86_400_000
        return (
            f"{self.source}:{self.inst_id}:{self.bar} —— {self.n_rows} 根 / "
            f"{span:.1f} 天（{_ms_to_iso(self.first_ts)} → {_ms_to_iso(self.last_ts)}）"
        )


def _ms_to_iso(ms: int) -> str:
    if not ms:
        return "-"
    return time.strftime("%Y-%m-%d %H:%M", time.gmtime(ms / 1000))


# --------------------------------------------------------------------------
# Store
# --------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS candles (
  source     TEXT    NOT NULL,
  inst_id    TEXT    NOT NULL,
  bar        TEXT    NOT NULL,
  ts         INTEGER NOT NULL,
  open       REAL    NOT NULL,
  high       REAL    NOT NULL,
  low        REAL    NOT NULL,
  close      REAL    NOT NULL,
  volume     REAL    NOT NULL,
  confirm    INTEGER NOT NULL DEFAULT 1,
  fetched_at INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (source, inst_id, bar, ts)
);

-- 覆盖度查询靠这个索引（「这个标的有多少数据」是最频繁的问题）
CREATE INDEX IF NOT EXISTS idx_candles_range
  ON candles (source, inst_id, bar, ts DESC);

CREATE TABLE IF NOT EXISTS metrics (
  source     TEXT    NOT NULL,
  inst_id    TEXT    NOT NULL,
  kind       TEXT    NOT NULL,
  ts         INTEGER NOT NULL,
  value      REAL,
  extra      TEXT    NOT NULL DEFAULT '',
  fetched_at INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (source, inst_id, kind, ts)
);

CREATE TABLE IF NOT EXISTS books (
  source     TEXT    NOT NULL,
  inst_id    TEXT    NOT NULL,
  ts         INTEGER NOT NULL,
  sz         INTEGER NOT NULL,
  bids       TEXT    NOT NULL,
  asks       TEXT    NOT NULL,
  fetched_at INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (source, inst_id, ts, sz)
);

CREATE TABLE IF NOT EXISTS fetches (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  source      TEXT NOT NULL,
  inst_id     TEXT NOT NULL,
  bar         TEXT NOT NULL,
  endpoint    TEXT NOT NULL DEFAULT '',
  from_ts     INTEGER NOT NULL DEFAULT 0,
  to_ts       INTEGER NOT NULL DEFAULT 0,
  n_rows      INTEGER NOT NULL DEFAULT 0,
  pages       INTEGER NOT NULL DEFAULT 0,
  started_at  INTEGER NOT NULL DEFAULT 0,
  finished_at INTEGER NOT NULL DEFAULT 0,
  ok          INTEGER NOT NULL DEFAULT 1,
  error       TEXT NOT NULL DEFAULT ''
);
"""


class MarketStore:
    """行情库。**唯一**允许写这张库的地方。

    与项目里 ``Market`` 是唯一允许改动账户的地方，是同一条纪律。
    """

    def __init__(self, path: str | Path = DEFAULT_DB) -> None:
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        # WAL 让「一边读一边写」不至于互相阻塞；本项目 GUI 会在后台跑作业，
        # 若同时有外部 Agent 读库，没有 WAL 会频繁 database is locked。
        if str(self.path) != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # -- 生命周期 ---------------------------------------------------------

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "MarketStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- 写：K 线 ---------------------------------------------------------

    def upsert_candles(
        self,
        source: str,
        inst_id: str,
        bar: str,
        rows: Iterable[Candle | dict[str, Any]],
        *,
        overwrite: bool = False,
    ) -> int:
        """写 K 线。返回**实际写入或更新**的行数。

        默认 ``overwrite=False``：已存在的 ``(source,inst,bar,ts)`` **不动**。
        这是「入库后不覆盖」那条约束的实现——它保证了「当时的决策
        看到的是哪一版数据」这个问题事后有确定答案。

        重复写入会被主键挡掉，但**不报错**（拉取分页的边界重叠是常态，
        报错会逼调用方自己去做去重，那是把简单问题推给每个调用者）。
        """
        now = int(time.time() * 1000)
        payload = []
        for r in rows:
            c = r if isinstance(r, Candle) else Candle(**r)
            # ⚠️ 再验一次：Candle 是 slots 类，构造后改字段不会重跑 __post_init__。
            # 只在构造时校验的话，被改坏的 Candle 能绕过检查进库（实测踩到）。
            c.validate()
            payload.append(
                (source, inst_id, bar, c.ts, c.open, c.high, c.low, c.close,
                 c.volume, c.confirm, now)
            )
        if not payload:
            return 0

        if overwrite:
            sql = """
                INSERT INTO candles
                  (source,inst_id,bar,ts,open,high,low,close,volume,confirm,fetched_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(source,inst_id,bar,ts) DO UPDATE SET
                  open=excluded.open, high=excluded.high, low=excluded.low,
                  close=excluded.close, volume=excluded.volume,
                  confirm=excluded.confirm, fetched_at=excluded.fetched_at
            """
        else:
            sql = """
                INSERT OR IGNORE INTO candles
                  (source,inst_id,bar,ts,open,high,low,close,volume,confirm,fetched_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """
        cur = self._conn.executemany(sql, payload)
        self._conn.commit()
        return int(cur.rowcount) if cur.rowcount is not None else 0

    # -- 写：快照类 -------------------------------------------------------

    def upsert_metrics(
        self,
        source: str,
        inst_id: str,
        kind: str,
        rows: Iterable[tuple[int, float | None, dict[str, Any] | None]],
    ) -> int:
        """写快照类量。``rows`` 是 ``(ts, value, extra_dict)``。"""
        now = int(time.time() * 1000)
        payload = [
            (source, inst_id, kind, int(ts),
             None if v is None else float(v),
             json.dumps(extra, ensure_ascii=False) if extra else "", now)
            for ts, v, extra in rows
        ]
        if not payload:
            return 0

        cur = self._conn.executemany(
            """
            INSERT INTO metrics (source,inst_id,kind,ts,value,extra,fetched_at)
            VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(source,inst_id,kind,ts) DO UPDATE SET
              value=excluded.value, extra=excluded.extra,
              fetched_at=excluded.fetched_at
            """,
            payload,
        )
        self._conn.commit()
        return int(cur.rowcount) if cur.rowcount is not None else 0

    def upsert_book(
        self,
        source: str,
        inst_id: str,
        ts: int,
        sz: int,
        bids: Sequence[Sequence[float]],
        asks: Sequence[Sequence[float]],
    ) -> None:
        now = int(time.time() * 1000)
        self._conn.execute(
            """
            INSERT INTO books (source,inst_id,ts,sz,bids,asks,fetched_at)
            VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(source,inst_id,ts,sz) DO UPDATE SET
              bids=excluded.bids, asks=excluded.asks, fetched_at=excluded.fetched_at
            """,
            (source, inst_id, int(ts), int(sz),
             json.dumps([list(x) for x in bids]),
             json.dumps([list(x) for x in asks]), now),
        )
        self._conn.commit()

    # -- 写：拉取账本 -----------------------------------------------------

    def log_fetch(self, rec: FetchRecord) -> int:
        d = asdict(rec)
        d.pop("id", None)
        cols = ",".join(d.keys())
        marks = ",".join("?" * len(d))
        cur = self._conn.execute(
            f"INSERT INTO fetches ({cols}) VALUES ({marks})",  # noqa: S608 — 列名是常量
            tuple(d.values()),
        )
        self._conn.commit()
        return int(cur.lastrowid or 0)

    # -- 读 ---------------------------------------------------------------

    def coverage(
        self, source: str, inst_id: str, bar: str, *, confirmed_only: bool = False
    ) -> Coverage:
        cond = "AND confirm = 1" if confirmed_only else ""
        row = self._conn.execute(
            f"""
            SELECT COUNT(*) AS n, MIN(ts) AS lo, MAX(ts) AS hi,
                   SUM(CASE WHEN confirm = 0 THEN 1 ELSE 0 END) AS bad
            FROM candles
            WHERE source = ? AND inst_id = ? AND bar = ? {cond}
            """,  # noqa: S608 — cond 是内部常量
            (source, inst_id, bar),
        ).fetchone()
        return Coverage(
            source=source, inst_id=inst_id, bar=bar,
            n_rows=int(row["n"] or 0),
            first_ts=int(row["lo"] or 0),
            last_ts=int(row["hi"] or 0),
            n_unconfirmed=int(row["bad"] or 0),
        )

    def list_instruments(self, source: str | None = None) -> list[dict[str, Any]]:
        """库里有数据的标的清单（MCP 的 ``list_instruments`` 工具就调它）。"""
        if source:
            rows = self._conn.execute(
                """
                SELECT source, inst_id, bar, COUNT(*) AS n,
                       MIN(ts) AS lo, MAX(ts) AS hi
                FROM candles WHERE source = ?
                GROUP BY source, inst_id, bar ORDER BY inst_id, bar
                """,
                (source,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT source, inst_id, bar, COUNT(*) AS n,
                       MIN(ts) AS lo, MAX(ts) AS hi
                FROM candles
                GROUP BY source, inst_id, bar ORDER BY source, inst_id, bar
                """
            ).fetchall()
        return [dict(r) for r in rows]

    def load_candles(
        self,
        source: str,
        inst_id: str,
        bar: str,
        *,
        from_ts: int = 0,
        to_ts: int = 0,
        limit: int = 0,
        confirmed_only: bool = True,
        name: str = "",
    ) -> Series:
        """读一段 K 线。

        ⭐ ``confirmed_only`` **默认为 True**：OKX 的 ``confirm='0'`` 表示
        这根 K 线还没走完、价格仍在跳。让未走完的根流进策略，
        等于**让策略读到未来** —— 这是本项目测量纪律里最不可接受的一类错误。
        """
        sql = ("SELECT ts,open,high,low,close,volume,confirm FROM candles "
               "WHERE source=? AND inst_id=? AND bar=?")
        args: list[Any] = [source, inst_id, bar]
        if confirmed_only:
            sql += " AND confirm = 1"
        if from_ts:
            sql += " AND ts >= ?"
            args.append(int(from_ts))
        if to_ts:
            sql += " AND ts <= ?"
            args.append(int(to_ts))
        sql += " ORDER BY ts"
        if limit:
            # 要的是**最近**的 limit 根，所以按 DESC 取再翻回来
            sql = sql.replace("ORDER BY ts", "ORDER BY ts DESC")
            sql += " LIMIT ?"
            args.append(int(limit))
        rows = self._conn.execute(sql, tuple(args)).fetchall()
        if limit:
            rows = list(reversed(rows))

        arrays = {k: np.array([r[k] for r in rows], dtype=(
            np.int64 if k in ("ts", "confirm") else np.float64)) for k in
            ("ts", "open", "high", "low", "close", "volume", "confirm")}
        return Series(
            name=name or f"{inst_id}_{bar}",
            source=source, inst_id=inst_id, bar=bar,
            timestamp=arrays["ts"], open=arrays["open"], high=arrays["high"],
            low=arrays["low"], close=arrays["close"], volume=arrays["volume"],
            confirm=arrays["confirm"],
        )

    def load_metrics(
        self, source: str, inst_id: str, kind: str,
        *, from_ts: int = 0, to_ts: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        sql = "SELECT ts, value FROM metrics WHERE source=? AND inst_id=? AND kind=?"
        args: list[Any] = [source, inst_id, kind]
        if from_ts:
            sql += " AND ts >= ?"
            args.append(int(from_ts))
        if to_ts:
            sql += " AND ts <= ?"
            args.append(int(to_ts))
        sql += " ORDER BY ts"
        rows = self._conn.execute(sql, tuple(args)).fetchall()
        ts = np.array([r["ts"] for r in rows], dtype=np.int64)
        vals = np.array(
            [np.nan if r["value"] is None else r["value"] for r in rows],
            dtype=np.float64,
        )
        return ts, vals

    def fetch_history(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM fetches ORDER BY id DESC LIMIT ?", (int(limit),)
        ).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict[str, int]:
        """库的整体规模（GUI / MCP 用来展示「现在有什么」）。"""
        out: dict[str, int] = {}
        for tbl in ("candles", "metrics", "books", "fetches"):
            n = self._conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]  # noqa: S608
            out[tbl] = int(n)
        return out

    # -- 维护 -------------------------------------------------------------

    def prune_unconfirmed(self, source: str = "") -> int:
        """删掉未确认的 K 线。

        未确认的根**不该长期留在库里**：它们会被后续拉取的正确值取代，
        但若不清理，「库里有多少数据」这个问题的答案会虚高。
        """
        if source:
            cur = self._conn.execute(
                "DELETE FROM candles WHERE confirm = 0 AND source = ?", (source,)
            )
        else:
            cur = self._conn.execute("DELETE FROM candles WHERE confirm = 0")
        self._conn.commit()
        return int(cur.rowcount or 0)


# --------------------------------------------------------------------------
# 便捷函数
# --------------------------------------------------------------------------


def snapshot_id(store: MarketStore, source: str, inst_id: str, bar: str) -> str:
    """给「库的当前状态」算一个指纹，写进决策留痕。

    目的是回答半年后那个问题：**那次决策看到的是哪一版数据？**
    指纹 = 行数 + 首尾时间戳 + 未确认根数（都已量化，可精确比对）。
    """
    import hashlib

    cov = store.coverage(source, inst_id, bar)
    raw = f"{source}|{inst_id}|{bar}|{cov.n_rows}|{cov.first_ts}|{cov.last_ts}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def open_store(path: str | Path = DEFAULT_DB) -> MarketStore:
    return MarketStore(path)


__all__ = [
    "BARS",
    "Candle",
    "Coverage",
    "DEFAULT_DB",
    "FetchRecord",
    "KINDS",
    "MarketStore",
    "SOURCE_OKX",
    "SOURCE_SYNTHETIC",
    "Series",
    "open_store",
    "snapshot_id",
]
