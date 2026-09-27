# -*- mode: python ; coding: utf-8 -*-
"""把「交易世界」桌面端打成 Windows 免安装目录。

设计要点（都是先摸过依赖才定的）
--------------------------------
1. **用 onedir 而不是 onefile**：onefile 每次启动都要把自己解压到临时目录
   （几百 MB，明显卡顿），而且临时目录里的 `__file__` 位置不可控。
   onedir 启动快、目录结构清晰、出问题好查。

2. **把项目目录原样放进运行时目录**：项目里到处是
   ``Path(__file__).resolve().parent.parent`` 来定位 `data/`、`gui/static/`。
   只要这些目录在打包后**保持同样的相对位置**，
   这些计算会**自动正确**，一行生产代码都不用改
   （实测：`tw/realdata.py` 的 ``DATA_DIR`` 与 `gui/server.py` 的 ``STATIC``
   都因此自动指向正确位置）。

   ⚠️ 但"放哪些"是个坑：**运行时要读的磁盘文件，静态分析是发现不了的**。
   第一版只带了 `data/` 与 `gui/static/`，漏掉了 `out/a14|a15|a16`
   （「文档验证」页的数据源）⇒ 那页在安装包里永远显示"产物缺失"。
   现在清单由 `gui/bundle_data.py` 定义，spec 与测试读**同一份**。

3. **不需要可写的 `out/`**：GUI 只在内存里跑实验、通过 HTTP 返回结果，
   不落盘（实测 gui/ 下无任何 open/write_text/mkdir）。
   所以打包后的目录可以整个是只读的。

4. **收集 webview 全家**：pywebview 在 Windows 上走 pythonnet + WebView2，
   `Python.Runtime.dll` 之类的东西 PyInstaller 的静态分析看不到，
   必须 `collect_all`。

体积预期：numpy + scipy + pythonnet 约 250~350 MB。
"""

from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

# SPECPATH 由 PyInstaller 注入 —— 用它算项目根，避免相对路径依赖当前工作目录
ROOT = Path(SPECPATH).parent if Path(SPECPATH).name == "packaging" else Path(SPECPATH)

# ⭐ 要带哪些数据文件，**问代码**而不是在 spec 里再抄一份。
#    清单住在 `gui/bundle_data.py`（零依赖，所以构建期 import 它很安全），
#    测试也读同一份 ⇒ 不可能出现"spec 漏了、测试不知道"的分叉。
#
# ⚠️ 这个坑真踩过（2026-09-28）：第一版 spec 只带了 `data/` 与 `gui/static/`，
#    于是 `out/a14|a15|a16` 三份结果 JSON 没进包 ⇒「文档验证」页在
#    **安装包里**永远显示"产物缺失"，而开发机上一切正常（那里 `out/` 就在）。
#    静态分析发现不了这种"运行时要读的磁盘文件"。
import sys as _sys

if str(ROOT) not in _sys.path:
    _sys.path.insert(0, str(ROOT))
from gui.bundle_data import BUNDLE_DIRS, BUNDLE_FILES  # noqa: E402

datas: list[tuple[str, str]] = []
for _d in BUNDLE_DIRS:
    if (ROOT / _d).is_dir():
        datas.append((str(ROOT / _d), _d))
    else:
        print(f"[spec] ⚠️ 数据目录不存在，跳过：{_d}")
for _f in BUNDLE_FILES:
    if (ROOT / _f).is_file():
        datas.append((str(ROOT / _f), str(Path(_f).parent)))
    else:
        print(f"[spec] ⚠️ 数据文件不存在，跳过：{_f}")

binaries: list[tuple[str, str]] = []
hiddenimports: list[str] = []

# pywebview 的 Windows 后端靠 pythonnet 调 .NET，DLL 必须显式收集
for pkg in ("webview", "clr_loader", "pythonnet"):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception as exc:          # noqa: BLE001
        print(f"[spec] 收集 {pkg} 失败（不致命，继续）：{exc}")

# 项目自身的包：api.py 里有运行时 sys.path 注入，静态分析可能漏
for pkg in ("tw", "strategies", "gui"):
    hiddenimports += collect_submodules(pkg)

a = Analysis(
    # ⚠️ 入口**不能**用 scripts/gui.py —— 它与 gui/ 包同名，
    #    PyInstaller 会把入口编译到运行时目录顶层，必然撞名报循环导入。
    #    实测踩到，细节见 packaging/entry.py 的说明。
    [str(ROOT / "packaging" / "entry.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    # 这些只在开发时用，打进包里纯属占体积
    excludes=[
        "tests", "matplotlib", "pandas", "IPython", "jupyter", "notebook",
        "PyQt5", "PyQt6", "PySide2", "PySide6", "tkinter", "PIL",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="TradingWorld",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,                 # UPX 常被杀软误报，且压缩后启动更慢
    console=True,              # 首版留控制台便于看错误；正式版可改 False
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="TradingWorld",
)
