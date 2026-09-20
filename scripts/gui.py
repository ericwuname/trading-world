"""交易世界 · 桌面端启动入口。

    python scripts/gui.py                  # 开原生窗口
    python scripts/gui.py --browser        # 强制用系统浏览器
    python scripts/gui.py --no-window      # 只起服务（自动化 / 远程调试用）
    python scripts/gui.py --port 8765      # 指定端口
    python scripts/gui.py --workers 2      # 限制并发作业数
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gui import desktop  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="交易世界 GUI")
    ap.add_argument("--port", type=int, default=0, help="0 = 自动挑空闲端口")
    ap.add_argument("--workers", type=int, default=None, help="并发作业线程数")
    ap.add_argument("--browser", action="store_true", help="不用原生窗口，直接用浏览器")
    ap.add_argument("--no-window", action="store_true", help="只起服务，不开任何界面")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if not args.quiet:
        print("=" * 70)
        print("交易世界 · 基于主体建模的人工市场")
        print("=" * 70)

    return desktop.run(
        workers=args.workers,
        port=args.port,
        window=not (args.browser or args.no_window),
        open_browser=not args.no_window,
    )


if __name__ == "__main__":
    raise SystemExit(main())
