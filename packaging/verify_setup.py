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
import uuid
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
    # ⚠️ 本环境的**批量删除保护**会拦下 rmtree（>50 个文件/次）：
    #    上一次跑崩掉留下的安装目录会让这一次**连启动都做不到**。
    #    ⇒ 删不掉就换一个全新目录继续（验证本身不能因为"上次的残留"而失败）。
    try:
        shutil.rmtree(TARGET)
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️ 清不掉上次的残留（{type(exc).__name__}）⇒ 换一个全新目录继续")
        TARGET = TARGET.with_name(TARGET.name + "-" + uuid.uuid4().hex[:6])
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
# ⚠️ `console=False` 的构建读不到启动日志 ⇒ 用 `--token` 固定一个；
#    也**不能**用 stdout=PIPE 去读：子进程不写字，
#    父进程 `readline()` 会一直阻塞到子进程退出。
FIXED_TOKEN = "setup-" + uuid.uuid4().hex[:16]
proc = subprocess.Popen(
    [str(exe), "--no-window", "--port", str(PORT), "--quiet",
     "--token", FIXED_TOKEN],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
token = FIXED_TOKEN
try:
    def get(path, tok=""):
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}")
        if tok:
            req.add_header("X-TW-Token", tok)
        with OP.open(req, timeout=30) as resp:
            return resp.status, resp.read()

    # 等端口起来（顺便证明 --token 真的生效）
    # ⚠️ 连接被拒时抛的是 `urllib.error.URLError` —— 它**是** OSError 的子类，
    #    但这行必须写全：一旦漏了，脚本会在"服务还没起来"时**自己崩掉**，
    #    而不是报一条 FAIL。验证脚本的职责是**报告**失败，不是替它崩。
    # ⚠️ 窗口给到 180 秒：**全新安装的程序第一次启动要被杀软全量扫一遍**
    #    （180MB / 400+ 个文件），实测 60 秒不够 —— 那次"连接被拒"
    #    就是这么来的（不是程序坏了）。
    ready = False
    for _ in range(360):
        try:
            s, _ = get("/api/meta", token)
            if s == 200:
                ready = True
                break
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(0.5)
    check("--token 生效（用固定令牌打到 /api/meta 200）", ready,
          "" if ready else f"{PORT} 端口 180 秒内一直连不上：装好的程序没起来？")
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
