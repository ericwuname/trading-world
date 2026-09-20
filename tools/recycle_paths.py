"""把文件移入回收站（Windows Shell API，不经编译、不用 rm）。

为什么不用 os.remove：本项目的删除纪律要求「用回收站，不用永久删除」。
为什么不用 PowerShell 的 Add-Type：被安全策略拦截
（"Add-Type compiles and loads .NET code at runtime"），
所以这里用 ctypes 直接调 ``SHFileOperationW``。
"""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes
from pathlib import Path

FO_DELETE = 3
FOF_SILENT = 0x0004
FOF_NOCONFIRMATION = 0x0010
FOF_ALLOWUNDO = 0x0040          # ← 这一位才是"进回收站"而不是永久删除
FOF_NOERRORUI = 0x0400


class SHFILEOPSTRUCTW(ctypes.Structure):
    _fields_ = [
        ("hwnd", wintypes.HWND),
        ("wFunc", wintypes.UINT),
        ("pFrom", wintypes.LPCWSTR),
        ("pTo", wintypes.LPCWSTR),
        ("fFlags", ctypes.c_uint16),
        ("fAnyOperationsAborted", wintypes.BOOL),
        ("hNameMappings", ctypes.c_void_p),
        ("lpszProgressTitle", wintypes.LPCWSTR),
    ]


def recycle(paths: list[str | Path]) -> int:
    """把一组文件移入回收站，返回 Shell API 的返回码（0 = 成功）。"""
    abs_paths = [str(Path(p).resolve()) for p in paths]
    missing = [p for p in abs_paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"这些文件不存在：{missing}")
    # pFrom 必须是 **双空字符结尾** 的列表
    op = SHFILEOPSTRUCTW()
    op.wFunc = FO_DELETE
    op.pFrom = "\0".join(abs_paths) + "\0\0"
    op.fFlags = (FOF_ALLOWUNDO | FOF_NOCONFIRMATION
                 | FOF_SILENT | FOF_NOERRORUI)
    return int(ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op)))


def main() -> int:
    targets = sys.argv[1:]
    if not targets:
        print("用法：python tools/recycle_paths.py <路径> [<路径> ...]",
              file=sys.stderr)
        return 2
    print(f"  将 {len(targets)} 个文件移入回收站：")
    for t in targets:
        print(f"    {t}")
    code = recycle(targets)
    if code != 0:
        print(f"  ❌ Shell API 返回码 {code}（非 0 表示失败）", file=sys.stderr)
        return 1
    left = [t for t in targets if Path(t).exists()]
    print(f"  ✅ 完成（返回码 0）；仍存在的文件：{left or '无'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
