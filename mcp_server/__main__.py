"""``python -m mcp_server`` 的入口。

⚠️ **包本身不能直接 ``-m`` 运行**：``python -m mcp_server`` 找的是
``mcp_server.__main__``，而 ``__init__.py`` 里的 ``if __name__ == "__main__"``
在这里**永远不会为真**（包名不是 ``__main__``）。实测踩到：
直接 ``python -m mcp_server`` 会报
``No module named mcp_server.__main__; 'mcp_server' is a package``。

这与打包时那次「入口脚本与包同名」是**同一类**问题——
**运行方式与文件结构不匹配**，而且报错信息不会告诉你缺哪个文件。
"""

from __future__ import annotations

from . import main

if __name__ == "__main__":
    raise SystemExit(main())
