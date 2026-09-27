"""对**打包后的 exe** 做视觉验证：起服务 → 无头截图 → 收工。

为什么非要截图：接口全 200 而界面白屏是真实存在的失效形态
（前端某处 JS 抛错，页面停在半渲染状态）。唯一能确认"界面真的出来了"
的手段是把图拿回来**亲眼看**——这也是本项目对 GUI 的一贯验证方式。

用法：
    python packaging/shot_exe.py [出图路径] [tab]
"""
from __future__ import annotations

import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
EXE = ROOT / "out" / "_package" / "dist" / "TradingWorld" / "TradingWorld.exe"
OUT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else (
    ROOT / "out" / "_package" / "截图.png")
TAB = sys.argv[2] if len(sys.argv) > 2 else ""
PORT = 8797
OP = urllib.request.build_opener(urllib.request.ProxyHandler({}))

if not EDGE.is_file():
    raise SystemExit(f"找不到 Edge：{EDGE}")
if not EXE.is_file():
    raise SystemExit(f"找不到打包产物：{EXE}（先跑 PyInstaller）")

proc = subprocess.Popen(
    [str(EXE), "--no-window", "--port", str(PORT), "--quiet"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    encoding="utf-8", errors="replace")

token = ""
try:
    t0 = time.time()
    while time.time() - t0 < 40 and not token:
        line = proc.stdout.readline()
        if not line:
            if proc.poll() is not None:
                break
            continue
        m = re.search(r"token=([\w\-]+)", line)
        if m:
            token = m.group(1)

    url = f"http://127.0.0.1:{PORT}/?token={token}"
    if TAB:
        url += f"&tab={TAB}"

    # 等首页真的能连上再截图（否则可能拍到"连接被拒"的错误页）
    for _ in range(80):
        try:
            with OP.open(f"http://127.0.0.1:{PORT}/", timeout=2) as r:
                if r.status == 200:
                    break
        except (OSError, urllib.error.URLError):
            time.sleep(0.25)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        [str(EDGE), "--headless=new", "--disable-gpu", "--hide-scrollbars",
         "--window-size=1600,1000", "--virtual-time-budget=12000",
         f"--screenshot={OUT}", url],
        capture_output=True, text=True, timeout=120)
    if OUT.is_file():
        print(f"截图：{OUT}  ({OUT.stat().st_size / 1024:.1f} KB)")
        print("⚠️ 下一步必须**打开这张图看一眼** —— 脚本只能说它生成了。")
    else:
        print("❌ 没生成截图")
        print(r.stderr[-1500:])
        raise SystemExit(1)
finally:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
