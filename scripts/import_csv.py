"""把一个 CSV 行情文件导入行情库（A6 · 补数据）。

为什么需要它
------------
A6 的效能标定给出一个硬结论：**本库只有 799 根 ⇒ 所有对照的「判力比」都 < 1**，
即"无论怎么切段，这段历史都判不出那个效应"。
⇒ 「不接受无法判定」的正解不是再换度量（度量已经换过一次了），
  是**补数据**。

而 OKX 联网在当前环境被代理挡住（`Tunnel connection failed: 502`），
所以用**已有的真实历史包**（`真实市场检验交付包/data/*_1h.csv`，各 17,520 根）。

⚠️⚠️ 两条必须守住的口径
----------------------
1. **不许冒充来源**：这批 CSV 是 Binance 的 `BTCUSDT`，不是 OKX 的
   `BTC-USDT-SWAP`。若直接写进 `source='okx'`，就制造了一个
   "血缘说谎"——以后没人能回答"这个数字来自哪个交易所的口径"。
   ⇒ 用**独立的 source 名**（默认 `binance_csv`），inst 用文件里的真实名。
2. **`INSERT OR IGNORE` 只在"不冲突"时才安全**。本批数据（2024-09 起）
   与库里已有的 OKX 数据（2026-08 起）**时间不重叠**，所以不会丢；
   但脚本仍会**打印 before/after 行数**并核对增量，
   因为"静默丢弃"正是本项目踩过的坑。

用法::

    python scripts/import_csv.py --csv <BTCUSDT_1h.csv> \\
        --source binance_csv --inst BTCUSDT --bar 1H
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import banner  # noqa: E402

from tw.marketdb import MarketStore  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="CSV → 行情库（A6 补数据）")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--source", default="binance_csv",
                    help="⚠️ 不要写成 okx —— 那批文件是 Binance 的口径")
    ap.add_argument("--inst", default="")
    ap.add_argument("--bar", default="1H")
    ap.add_argument("--batch", type=int, default=2000)
    args = ap.parse_args()

    banner("CSV → 行情库（A6 补数据）")
    path = Path(args.csv)
    if not path.exists():
        raise SystemExit(f"❌ 找不到 {path}")
    inst = args.inst or path.stem.split("_")[0].upper()
    print(f"  文件：{path}")
    print(f"  写入：source={args.source}  inst_id={inst}  bar={args.bar}")

    rows: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        need = {"timestamp", "open", "high", "low", "close"}
        missing = need - set(rd.fieldnames or [])
        if missing:
            raise SystemExit(f"❌ CSV 缺列：{sorted(missing)}")
        for r in rd:
            try:
                rows.append({
                    "ts": int(float(r["timestamp"])),
                    "open": float(r["open"]), "high": float(r["high"]),
                    "low": float(r["low"]), "close": float(r["close"]),
                    "volume": float(r.get("volume") or 0.0),
                })
            except (TypeError, ValueError):
                continue
    print(f"  解析出 {len(rows)} 根，时间范围 {rows[0]['ts']} … {rows[-1]['ts']}")

    store = MarketStore()
    try:
        before = _count(store, args.source, inst, args.bar)
        written = 0
        for i in range(0, len(rows), int(args.batch)):
            written += store.upsert_candles(
                args.source, inst, args.bar, rows[i:i + int(args.batch)])
        after = _count(store, args.source, inst, args.bar)
    finally:
        store.close()

    print(f"  写入前 {before} 行 → 写入后 {after} 行"
          f"（upsert 返回 {written}）")
    # ⚠️ **核对增量，不看返回值**：`INSERT OR IGNORE` 会静默跳过冲突行，
    # 而"跳过"在这里可能是正常的（重复导入）也可能是致命的（丢了数据）。
    # 用"前后行数差"这个**独立于返回值**的量来判断。
    delta = after - before
    print(f"  净增 {delta} 行")
    if delta < len(rows):
        print(f"  ⚠️ 净增少于解析出的 {len(rows)} 行 ⇒ 有重复/冲突被静默跳过。"
              f"若本批与库里时间重叠，这是预期的；否则要查。")
    return 0


def _count(store: MarketStore, source: str, inst: str, bar: str) -> int:
    cur = store._conn.execute(
        "SELECT COUNT(*) FROM candles WHERE source=? AND inst_id=? AND bar=?",
        (source, inst, bar))
    return int(cur.fetchone()[0])


if __name__ == "__main__":
    raise SystemExit(main())
