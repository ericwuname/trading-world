"""真实行情载入（施工蓝图 §5「对照数据」）。

不重新下载，直接复用既有交付包里的 CSV——**同一个数据源**这一点很重要：
若模拟对照用新下载的数据、历史结论用旧数据，两边口径不一致，
比较出来的差异到底是"模拟不像真实市场"还是"两份数据本身不同"就分不清了。

CSV 格式::

    timestamp,open,high,low,close,volume
    1726477200000,58924.98,59032.20,58579.27,58636.54,1072.98491

不用 pandas：本项目的依赖清单只有 numpy / scipy / matplotlib。
多一个 pandas 就多一个"环境不对所以没跑"的借口，而这里只需要读 6 列数字。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

#: 项目内置的真实行情文件（复制自既有交付包，不改动）
BUILTIN = {
    "BTCUSDT_1h": DATA_DIR / "BTCUSDT_1h.csv",
    "ETHUSDT_1h": DATA_DIR / "ETHUSDT_1h.csv",
    "SOLUSDT_1h": DATA_DIR / "SOLUSDT_1h.csv",
    "BTCUSDT_1d": DATA_DIR / "BTCUSDT_1d.csv",
    "BNBUSDT_1d": DATA_DIR / "BNBUSDT_1d.csv",
    "BCHUSDT_1d": DATA_DIR / "BCHUSDT_1d.csv",
}


@dataclass(slots=True)
class Series:
    """一段行情。``close`` 是最重要的字段（统计特征对照用它）。"""

    name: str
    timestamp: np.ndarray  # 毫秒
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray

    def __len__(self) -> int:
        return int(self.close.size)

    @property
    def log_returns(self) -> np.ndarray:
        r = np.diff(np.log(self.close))
        return r[np.isfinite(r)]

    @property
    def span_days(self) -> float:
        if self.timestamp.size < 2:
            return 0.0
        return float((self.timestamp[-1] - self.timestamp[0]) / 86_400_000)

    def describe(self) -> dict:
        return {
            "name": self.name,
            "n_bars": len(self),
            "span_days": round(self.span_days, 1),
            "price_first": float(self.close[0]),
            "price_last": float(self.close[-1]),
            "price_min": float(self.close.min()),
            "price_max": float(self.close.max()),
        }


def load_csv(path: str | Path, name: str | None = None) -> Series:
    """读取 OHLCV CSV。列名按表头匹配，避免依赖固定列序。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到行情文件: {path}")
    with path.open("r", encoding="utf-8") as fh:
        header = fh.readline().strip().split(",")
    cols = {c.strip().lower(): i for i, c in enumerate(header)}
    need = ["timestamp", "open", "high", "low", "close"]
    for k in need:
        if k not in cols:
            raise ValueError(f"{path.name} 缺少列 {k}，实际有 {header}")
    has_vol = "volume" in cols
    ncol = len(header)

    raw = np.loadtxt(path, delimiter=",", skiprows=1)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if raw.shape[1] != ncol:
        raise ValueError(f"{path.name} 数据列数与表头不符")

    ts = raw[:, cols["timestamp"]]
    order = np.argsort(ts, kind="stable")
    ts = ts[order]
    # 去重：同一时间戳只保留最后一条（重复数据会让收益率序列出现人造的 0）
    keep = np.concatenate([[True], np.diff(ts) > 0])

    def col(key: str) -> np.ndarray:
        return raw[order, cols[key]][keep]

    return Series(
        name=name or path.stem,
        timestamp=ts[keep],
        open=col("open"),
        high=col("high"),
        low=col("low"),
        close=col("close"),
        volume=(col("volume") if has_vol else np.full(int(keep.sum()), np.nan)),
    )


def load_builtin(key: str) -> Series:
    if key not in BUILTIN:
        raise KeyError(f"未知的内置数据集 {key!r}，可选: {sorted(BUILTIN)}")
    return load_csv(BUILTIN[key], name=key)


def available_builtin() -> list[str]:
    return [k for k, p in BUILTIN.items() if p.exists()]
