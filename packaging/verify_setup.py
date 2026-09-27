"""实测 Inno Setup 安装包：安装 → 运行 → 卸载（全静默，不碰桌面/开始菜单）。

为什么必须实测：`ISCC` 编译成功只说明**脚本语法对**，不说明装完能跑。
安装包特有的失效有：`[Files]` 的源目录写错、装到 Program Files 后的权限问题、
卸载残留。

⚠️ 用 `/NOICONS` + 自定义 `/DIR`：**不往用户的桌面/开始菜单写任何东西**，
装到 `out/` 下的临时目录，验完就撤。这是"验证不能有副作用"的具体做法。

用法：python packaging/verify_setup.py
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "out" / "_package"

VER = re.search(r'__version__\s*=\s*"([^"]+)"',
                (ROOT / "gui" / "__init__.py").read_text(encoding="utf-8")).group(1)
SETUP = OUTDIR / f"TradingWorld-Setup-{VER}.exe"
TARGET = OUTDIR / "_installtest"
PORT = 8798
OP = urllib.request.build_opener(urllib.request.ProxyHandler({}))

if not SETUP.is_file():
    raise SystemExit(f"找不到安装包：{SETUP}（先跑 packaging/make_dist.py）")

ok = True


def check(label: str, cond: bool, extra: str = "") -> None:
    global ok
    if not cond:
        ok = False
    print(f"  [{'OK  ' if cond else 'FAIL'}] {label}{(' — ' + extra) if extra else ''}")


if TARGET.exists():
    shutil.rmtree(TARGET, ignore_errors=True)
TARGET.mkdir(parents=True, exist_ok=True)

print(f"① 静默安装到 {TARGET.name}/（/NOICONS：不写桌面与开始菜单）")
r = subprocess.run(
    [str(SETUP), "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NOICONS",
     f"/DIR={TARGET}", f"/LOG={OUTDIR / '_installtest.log'}"],
    capture_output=True, text=True, timeout=300)
check("安装程序返回 0", r.returncode == 0, f"rc={r.returncode}")

exe = TARGET / "TradingWorld.exe"
check("装完有 TradingWorld.exe", exe.is_file())
check("装完有 _internal/", (TARGET / "_internal").is_dir())
check("随包数据也装进来了（out/a16）", (TARGET / "_internal" / "out" / "a16").is_dir())
check("卸载程序已注册（unins000.exe）", (TARGET / "unins000.exe").is_file())

print("② 运行装好的程序，打真接口")
proc = subprocess.Popen(
    [str(exe), "--no-window", "--port", str(PORT), "--quiet"],
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
    check("拿到 token", bool(token))

    def get(path, tok=""):
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}")
        if tok:
            req.add_header("X-TW-Token", tok)
        with OP.open(req, timeout=30) as resp:
            return resp.status, resp.read()

    s, b = get("/")
    check("GET / 200 且含「交易世界」",
          s == 200 and "交易世界" in b.decode("utf-8", "replace"))
    s, b = get("/api/meta", token)
    check("GET /api/meta 200", s == 200, f"status={s}")
    s, b = get("/api/doc/reports", token)
    reps = json.loads(b)["reports"] if s == 200 else []
    check("文档验证三份报告在（装完也在）",
          len(reps) == 3 and all(r["json_exists"] for r in reps))
    s, _ = get("/api/real/BTCUSDT_1h", token)
    check("真实行情可读（data/ 也装进来了）", s == 200, f"status={s}")
finally:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()

print("③ 静默卸载")
r2 = subprocess.run([str(TARGET / "unins000.exe"), "/VERYSILENT", "/SUPPRESSMSGBOXES"],
                    capture_output=True, text=True, timeout=300)
time.sleep(3)   # Inno 的卸载器把真正的删除交给临时进程，要等一下
left = [p.name for p in TARGET.iterdir()] if TARGET.exists() else []
check("卸载后目录清空", not left, f"残留 {left}")

print()
print("结论：" + ("安装 → 运行 → 卸载 全部通过 ✅" if ok else "有失败项 ❌"))
sys.exit(0 if ok else 1)
