"""打包专用入口 —— 名字必须是**除了 `gui` 以外**的任何东西。

⚠️ 为什么不能直接拿 ``scripts/gui.py`` 当打包入口（实测踩到）
--------------------------------------------------------
PyInstaller 会把**入口脚本**编译成运行时目录 ``_internal/`` **顶层**的同名模块。
入口若叫 ``gui.py``，顶层就会出现一个 ``gui`` 模块，
而项目里还有一个 ``gui/`` **包** ⇒ ``from gui import desktop``
解析到的是**入口自己**，于是报：

    ImportError: cannot import name 'desktop' from partially initialized
    module 'gui' (most likely due to a circular import)

这个坑项目里**记录过一次**（``tests/test_cache_keys.py`` 里写着
"``scripts/gui.py`` 与仓库根下的 ``gui/`` 包同名"），
当时源码环境靠"先把 ``scripts/`` 从 sys.path 摘掉"绕开了；
但**打包环境绕不开**——入口一定在顶层。
⇒ 所以打包入口单独起一个不与任何包同名的文件名。

逻辑本身不在这里：参数与启动流程的唯一事实源是 ``gui/cli.py``。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 打包后 __file__ 在 _internal/ 下，其父目录就是运行时根
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gui.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
