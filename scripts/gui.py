"""交易世界 · 桌面端启动入口（开发环境用）。

    python scripts/gui.py                  # 开原生窗口
    python scripts/gui.py --browser        # 强制用系统浏览器
    python scripts/gui.py --no-window      # 只起服务（自动化 / 远程调试用）
    python scripts/gui.py --port 8765      # 指定端口
    python scripts/gui.py --workers 2      # 限制并发作业数

参数与启动逻辑的**唯一事实源**在 ``gui/cli.py``；本文件只是一层薄壳，
好让开发时的入口路径保持 ``scripts/gui.py`` 不变（文档与习惯都指向它）。

⚠️ 本文件与 ``gui/`` 包**同名**。源码环境下靠
``tests/test_gui.py`` 里"先把 ``scripts/`` 从 sys.path 摘掉"来绕开；
**打包时不能再用它当入口** —— PyInstaller 会把入口脚本编译成运行时目录
**顶层**的同名模块，必然撞上 ``gui`` 包，报
``partially initialized module 'gui'``（循环导入）。
打包入口是 ``packaging/entry.py``，它叫别的名字，从根上避开这个坑。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gui.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
