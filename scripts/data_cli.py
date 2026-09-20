"""行情库工具：拉取 / 查看 / 生成 / 维护。

用法（项目根目录下）::

    # 看一眼库里现在有什么
    python scripts/data_cli.py list

    # 按需拉取（用户要求的「准备交易时才拉」）
    python scripts/data_cli.py fetch --inst BTC-USDT-SWAP --bar 1H --days 90

    # 抓一次快照（资金费率 / 未平仓量 / 标记价）
    python scripts/data_cli.py snapshot --inst BTC-USDT-SWAP

    # 生成自测试三件套（随机游走 / 趋势 / 波动率聚集）
    python scripts/data_cli.py synth --n 2000 --seed 0

    # 看拉取账本（回答「这份数据是什么时候拉的」）
    python scripts/data_cli.py ledger -n 20

    # 清理未确认 K 线
    python scripts/data_cli.py prune

⚠️ ``fetch`` / ``snapshot`` 会**联网**。其余命令全程只读本地库。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tw.marketdb import DEFAULT_DB, SOURCE_OKX, SOURCE_SYNTHETIC, MarketStore  # noqa: E402
from tw.okx_data import OkxError, OkxHttp, ensure_history, snapshot_metrics  # noqa: E402
from tw.synthetic import (  # noqa: E402
    GenerateSpec,
    available_regimes,
    default_test_set,
    generate_into_store,
)


def _fmt_ts(ms: int) -> str:
    import time

    if not ms:
        return "-"
    return time.strftime("%Y-%m-%d %H:%M", time.gmtime(ms / 1000))


def cmd_list(args: argparse.Namespace) -> int:
    with MarketStore(args.db) as s:
        st = s.stats()
        print(f"库：{args.db}")
        print(f"  candles={st['candles']}  metrics={st['metrics']}  "
              f"books={st['books']}  fetches={st['fetches']}")
        rows = s.list_instruments(args.source or None)
        if not rows:
            print("\n（库里还没有行情数据）")
            return 0
        print(f"\n{'source':<10} {'inst_id':<24} {'bar':<5} {'n':>7}  "
              f"{'从':<17} {'到':<17}")
        print("-" * 88)
        for r in rows:
            print(f"{r['source']:<10} {r['inst_id']:<24} {r['bar']:<5} "
                  f"{r['n']:>7}  {_fmt_ts(r['lo']):<17} {_fmt_ts(r['hi']):<17}")
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    print(f"拉取 {args.inst} {args.bar} 近 {args.days} 天 …（联网）")

    last = {"msg": ""}

    def progress(msg: str, n: int) -> None:
        if msg != last["msg"]:
            print(f"  {msg}（{n} 条）")
            last["msg"] = msg

    try:
        with MarketStore(args.db) as s:
            cov = ensure_history(
                s, args.inst, args.bar, days=args.days,
                http=OkxHttp(max_retries=args.retries),
                progress=progress,
            )
            print("\n完成：" + cov.describe())
    except OkxError as exc:
        print(f"\n拉取失败：{exc}", file=sys.stderr)
        print("（失败已写入 fetches 账本，可用 ledger 命令查到）", file=sys.stderr)
        return 1
    return 0


def cmd_snapshot(args: argparse.Namespace) -> int:
    print(f"抓快照 {args.inst} …（联网）")
    try:
        with MarketStore(args.db) as s:
            out = snapshot_metrics(s, args.inst, http=OkxHttp(max_retries=args.retries))
    except OkxError as exc:
        print(f"失败：{exc}", file=sys.stderr)
        return 1
    if not out:
        print("没有可用指标（现货只有标记价/指数价，资金费率与未平仓量仅永续有）")
        return 0
    for k, v in out.items():
        print(f"  {k} = {v}")
    return 0


def cmd_synth(args: argparse.Namespace) -> int:
    with MarketStore(args.db) as s:
        if args.regime:
            spec = GenerateSpec(
                inst_id=args.inst, bar=args.bar, n_bars=args.n,
                seed=args.seed, regime=args.regime,
                start_ts=args.start_ts, start_price=args.start_price,
            )
            specs = [spec]
        else:
            specs = default_test_set(
                inst_id=args.inst, bar=args.bar, n_bars=args.n,
                seed=args.seed, start_ts=args.start_ts,
                start_price=args.start_price,
            )
        print(f"生成 {len(specs)} 份（regime 可选：{', '.join(available_regimes())}）")
        for spec in specs:
            out = generate_into_store(s, spec)
            print(f"  {out['inst_id']:<32} {out['regime']:<12} "
                  f"{out['n_written']:>6} 根")
    return 0


def cmd_ledger(args: argparse.Namespace) -> int:
    with MarketStore(args.db) as s:
        rows = s.fetch_history(args.n)
        if not rows:
            print("账本为空")
            return 0
        print(f"{'id':>4} {'时间':<17} {'源':<10} {'inst_id':<24} {'bar':<5} "
              f"{'行数':>6} {'页':>4} {'ok':<3} 端点")
        print("-" * 130)
        for r in rows:
            ok = "✓" if r["ok"] else "✗"
            ep = r["endpoint"][:52] + ("…" if len(r["endpoint"]) > 52 else "")
            print(f"{r['id']:>4} {_fmt_ts(r['finished_at']):<17} "
                  f"{r['source']:<10} {r['inst_id']:<24} {r['bar']:<5} "
                  f"{r['n_rows']:>6} {r['pages']:>4} {ok:<3} {ep}")
            if not r["ok"] and r["error"]:
                print(f"       └─ 错误：{r['error'][:100]}")
    return 0


def cmd_prune(args: argparse.Namespace) -> int:
    with MarketStore(args.db) as s:
        n = s.prune_unconfirmed(args.source or "")
        print(f"清理了 {n} 根未确认 K 线")
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    """一次性建好「自测试三件套 + 指定真实标的」的常用组合。"""
    with MarketStore(args.db) as s:
        print("① 生成自测试三件套 …")
        for spec in default_test_set(n_bars=args.n, seed=0):
            out = generate_into_store(s, spec)
            print(f"   {out['inst_id']:<32} {out['n_written']:>6} 根")
    if args.inst:
        print(f"\n② 拉取 {args.inst} 近 {args.days} 天 …（联网）")
        try:
            with MarketStore(args.db) as s:
                cov = ensure_history(s, args.inst, args.bar, days=args.days)
                print("   " + cov.describe())
        except OkxError as exc:
            print(f"   拉取失败（不影响本地合成数据）：{exc}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="data_cli",
        description="行情库工具（拉取 / 查看 / 生成 / 维护）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--db", default=str(DEFAULT_DB), help=f"库路径（默认 {DEFAULT_DB}）")
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("list", help="看库里有什么")
    q.add_argument("--source", default="", choices=["", SOURCE_OKX, SOURCE_SYNTHETIC])
    q.set_defaults(func=cmd_list)

    q = sub.add_parser("fetch", help="按需拉取 OKX 行情（联网）")
    q.add_argument("--inst", default="BTC-USDT-SWAP")
    q.add_argument("--bar", default="1H")
    q.add_argument("--days", type=int, default=90)
    q.add_argument("--retries", type=int, default=3)
    q.set_defaults(func=cmd_fetch)

    q = sub.add_parser("snapshot", help="抓一次快照指标（联网）")
    q.add_argument("--inst", default="BTC-USDT-SWAP")
    q.add_argument("--retries", type=int, default=3)
    q.set_defaults(func=cmd_snapshot)

    q = sub.add_parser("synth", help="生成自测试数据（离线）")
    q.add_argument("--inst", default="SYNTH-BTC")
    q.add_argument("--bar", default="1H")
    q.add_argument("--n", type=int, default=2000)
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--regime", default="",
                   choices=[""] + available_regimes(),
                   help="不填则生成三件套")
    q.add_argument("--start-ts", type=int, default=1_700_000_000_000)
    q.add_argument("--start-price", type=float, default=60_000.0)
    q.set_defaults(func=cmd_synth)

    q = sub.add_parser("ledger", help="看拉取账本")
    q.add_argument("-n", type=int, default=20)
    q.set_defaults(func=cmd_ledger)

    q = sub.add_parser("prune", help="清理未确认 K 线")
    q.add_argument("--source", default="", choices=["", SOURCE_OKX, SOURCE_SYNTHETIC])
    q.set_defaults(func=cmd_prune)

    q = sub.add_parser("build", help="一键建常用数据集")
    q.add_argument("--inst", default="", help="留空则只建合成数据，不联网")
    q.add_argument("--bar", default="1H")
    q.add_argument("--days", type=int, default=90)
    q.add_argument("--n", type=int, default=2000)
    q.set_defaults(func=cmd_build)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
