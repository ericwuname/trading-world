"""命令行入口 —— **单一实现**，被两个薄壳共用。

为什么要有这一层
----------------
项目里有 ``scripts/gui.py``（开发时的入口）和 ``packaging/entry.py``
（打包后的入口）。如果各写一份 argparse，两份会漂移；
所以参数的**唯一事实源**在这里，两个入口都只是薄壳。

⚠️ 一个已经被踩过两次的坑（源码一次、打包一次）
--------------------------------------------
仓库根有 ``gui/`` **包**，而 ``scripts/gui.py`` 是**同名模块**。
只要 ``scripts/`` 排在 ``sys.path`` 前面，``from gui import desktop``
就会解析到 ``scripts/gui.py`` 自身 —— 它也在 import gui，于是报
``partially initialized module 'gui'``（循环导入）。

- 源码环境：靠"先把 ``scripts/`` 从 sys.path 摘掉"绕开（见 ``tests/test_gui.py``）；
- **打包环境绕不开**：PyInstaller 会把入口脚本编译成运行时目录**顶层**的同名模块，
  所以入口叫 ``gui.py`` 就一定会撞上 ``gui`` 包。
  ⇒ 打包入口改用 ``packaging/entry.py``（见该文件的说明）。
"""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="TradingWorld", description="交易世界 GUI")
    ap.add_argument("--port", type=int, default=0, help="0 = 自动挑空闲端口")
    ap.add_argument("--workers", type=int, default=None, help="并发作业线程数")
    ap.add_argument("--browser", action="store_true", help="不用原生窗口，直接用浏览器")
    ap.add_argument("--no-window", action="store_true", help="只起服务，不开任何界面")
    ap.add_argument("--quiet", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.quiet:
        print("=" * 70)
        print("交易世界 · 基于主体建模的人工市场")
        print("=" * 70)

    from . import desktop
    return desktop.run(
        workers=args.workers,
        port=args.port,
        window=not (args.browser or args.no_window),
        open_browser=not args.no_window,
    )


if __name__ == "__main__":
    raise SystemExit(main())
