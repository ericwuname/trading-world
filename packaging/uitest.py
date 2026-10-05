"""在**真实浏览器**里点真按钮、断言可观察状态 —— 交易终端的交互验收。

⭐ 为什么需要这个（本项目踩过的坑）
------------------------------------
原来的验证是"截图 + 看字节数"。那种方式**会被静态结构骗过**：
界面里有一部分是静态 HTML（侧边表单、标签栏、面板容器），
所以**脚本全死时截图依然"正常"** —— 而检查只看图生成了，**会通过**。
本轮真出过这个问题（JS 注释没闭合 ⇒ 整个 script 块解析失败 ⇒ 图是空的）。

⇒ 所以要有"真的点一下，看效果有没有变"的验收。

为什么不用 Playwright：不想为这一件事引入新依赖（那是本项目反复强调的边界）。
做法是加一个**验收页** `gui/static/uitest.html`：它在真实浏览器里驱动真实界面，
把结果写进 `document.title` ⇒ 命令行这侧只要读 title 就能判定。

用法：python packaging/uitest.py            （自己起开发服务器）
      python packaging/uitest.py <exe>      （验收打包后的 exe）
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = r"C:\Users\87465\.workbuddy\binaries\python\envs\py313\Scripts\python.exe"
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
EXE = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else None
PORT, TOK = "8797", "uitest-token"
SHOT = ROOT / "out" / "_shots" / "uitest.png"
OP = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def call(path: str, body: dict | None = None):
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}",
        data=json.dumps(body).encode() if body is not None else None,
        headers={"X-TW-Token": TOK, "Content-Type": "application/json"})
    with OP.open(req, timeout=90) as r:
        return json.loads(r.read())


cmd = ([str(EXE), "--no-window", "--port", PORT, "--token", TOK, "--quiet"] if EXE
       else [PY, "scripts/gui.py", "--no-window", "--port", PORT, "--token", TOK, "--quiet"])
print(f"起服务：{'打包后的 exe' if EXE else '开发服务器'}")
srv = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
ok = False
try:
    for _ in range(100):                       # 等服务起来
        try:
            call("/api/meta")
            break
        except OSError:
            time.sleep(0.5)
    else:
        print("❌ 服务没起来")
        sys.exit(1)

    SHOT.parent.mkdir(parents=True, exist_ok=True)
    url = f"http://127.0.0.1:{PORT}/static/uitest.html?token={TOK}"
    print(f"打开验收页：{url}")
    # ⚠️ 虚拟时间预算给足：会话预热 ~3.2s，且每步都是**轮询等待**（不是固定 sleep）
    r = subprocess.run(
        [EDGE, "--headless=new", "--disable-gpu", "--hide-scrollbars",
         "--window-size=1500,1000", "--virtual-time-budget=120000",
         f"--screenshot={SHOT}", url],
        capture_output=True, text=True, timeout=400)
    # ⭐ 再 dump 一次 DOM 拿结果表（--screenshot 与 --dump-dom 不能同时用）
    r2 = subprocess.run(
        [EDGE, "--headless=new", "--disable-gpu", "--virtual-time-budget=120000",
         "--dump-dom", url], capture_output=True, text=True, timeout=400)
    html = r2.stdout
    m = re.search(r'<div id="summary"[^>]*>(.*?)</div>', html, re.S)
    summary = (m.group(1).strip() if m else "（没读到 summary）")
    rows = re.findall(r"<tr><td>(\d+)</td><td>(.*?)</td><td class=\"(\w+)\">(.*?)</td><td>(.*?)</td></tr>",
                      html, re.S)
    print(f"\n验收结果：{summary}")
    print("-" * 78)
    bad = 0
    for no, name, cls, res, detail in rows:
        if cls != "ok":
            bad += 1
        clean = re.sub(r"<[^>]+>", "", detail)
        print(f"  [{'OK  ' if cls == 'ok' else 'FAIL'}] {name}"
              + (f"  —— {clean[:80]}" if clean else ""))
    print("-" * 78)
    kb = SHOT.stat().st_size / 1024 if SHOT.is_file() else 0
    print(f"截图 {SHOT.name} {kb:.0f} KB（人工复核用）")
    ok = bad == 0 and rows
    print("结论：" + ("交互验收全部通过 ✅" if ok else f"有 {bad} 项失败 ❌"))
finally:
    srv.terminate()
    try:
        srv.wait(timeout=10)
    except subprocess.TimeoutExpired:
        srv.kill()
sys.exit(0 if ok else 1)
