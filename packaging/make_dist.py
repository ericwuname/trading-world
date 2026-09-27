"""生成分发产物：免安装 zip + Inno Setup 安装包，并当场自检。

为什么写成脚本而不是敲一行命令：zip 的**内容形态**是分发契约的一部分
（解压出来必须是一个 `TradingWorld/` 文件夹、不能把 `_internal` 摊一地），
而且"压缩完要核对条目数 + CRC"这件事容易被漏掉。
`namelist()` 只说明目录索引读得出来，**不说明内容没坏**——`testzip()` 才校 CRC。

用法：python packaging/make_dist.py
"""
from __future__ import annotations

import re
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "out" / "_package" / "dist" / "TradingWorld"
OUTDIR = ROOT / "out" / "_package"

#: 版本号从 `gui/__init__.py` 读（**唯一事实源**，别在这里再写一个）
VER = re.search(r'__version__\s*=\s*"([^"]+)"',
                (ROOT / "gui" / "__init__.py").read_text(encoding="utf-8")).group(1)
ZIP = OUTDIR / f"TradingWorld-v{VER}-win64-免安装.zip"
SETUP = OUTDIR / f"TradingWorld-Setup-{VER}.exe"
ISCC = Path.home() / "AppData" / "Local" / "Programs" / "Inno Setup 6" / "ISCC.exe"

if not DIST.is_dir():
    raise SystemExit(f"没有打包产物：{DIST}（先跑 PyInstaller）")
print(f"版本 {VER}")

# ---------------------------------------------------------------- zip
if ZIP.exists():
    ZIP.unlink()
print(f"压缩 {DIST.name}/ → {ZIP.name}")
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for p in sorted(DIST.rglob("*")):
        if p.is_file():
            # ⚠️ arcname 必须带顶层目录名，否则解压会把 _internal 摊在当前目录
            z.write(p, Path("TradingWorld") / p.relative_to(DIST))

with zipfile.ZipFile(ZIP) as z:
    bad = z.testzip()
    names = z.namelist()
tops = sorted({n.split("/")[0] for n in names})
exes = [n for n in names if n.endswith("TradingWorld.exe")]
print(f"  条目 {len(names)}｜顶层 {tops}｜exe {exes}")
print(f"  CRC 校验 {'通过' if bad is None else '损坏 → ' + str(bad)}")
problems = []
if bad is not None:
    problems.append(f"CRC 损坏：{bad}")
if tops != ["TradingWorld"]:
    problems.append(f"zip 顶层不对：{tops}")
if not exes:
    problems.append("zip 里没有 TradingWorld.exe")
if len(names) < 300:
    problems.append(f"条目太少（{len(names)}），可能漏了 _internal")
print(f"  {'✅' if not problems else '❌'} {ZIP.stat().st_size / 1e6:.1f} MB")

# ---------------------------------------------------------------- 安装包
if not ISCC.is_file():
    print(f"\n⚠️ 找不到 ISCC（{ISCC}）⇒ 跳过安装包编译")
    print("   装它：winget install JRSoftware.InnoSetup")
    raise SystemExit(1 if problems else 0)

print("\n编译 Inno Setup 安装包…")
r = subprocess.run([str(ISCC), str(ROOT / "packaging" / "installer.iss")],
                   capture_output=True, text=True, encoding="utf-8",
                   errors="replace")
for ln in [x for x in r.stdout.splitlines() if x.strip()][-6:]:
    print("  " + ln)
if r.returncode != 0 or not SETUP.is_file():
    print(f"  ❌ 编译失败 rc={r.returncode}")
    print(r.stderr[-1200:])
    problems.append("安装包编译失败")
else:
    print(f"  ✅ {SETUP.name}  {SETUP.stat().st_size / 1e6:.1f} MB")

print()
if problems:
    print("结论：有问题 ❌ " + "；".join(problems))
    raise SystemExit(1)
print("结论：zip 与安装包都已生成 ✅")
print("⚠️ 「编译成功 ≠ 装完能跑」—— 继续跑：")
print("     python packaging/verify_setup.py     （安装 → 运行 → 卸载）")
print("     python packaging/verify_exe.py       （把 exe 当黑盒打真接口）")
