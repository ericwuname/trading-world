"""把打包产物当**黑盒**打一遍真接口。

为什么不能只看"打包成功"：`--add-data` 的相对位置、`_internal` 里的 PYZ 路径、
运行时要读的磁盘文件（`out/a14|a15|a16`、`data/`）—— **静态分析看不到的东西太多**。
唯一可信的判据是：跑起来、打真接口、**真跑一个作业**。

（这个脚本是 2026-09-28 那次的产物：正是它抓到了"三份报告 JSON 没进包"
——「文档验证」页在安装包里是死的，而开发机上一切正常。）

用法：
    python packaging/verify_exe.py out/_package/dist/TradingWorld/TradingWorld.exe [端口]
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8795
EXE = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else (
    ROOT / "out" / "_package" / "dist" / "TradingWorld" / "TradingWorld.exe")
BASE = f"http://127.0.0.1:{PORT}"
# ⚠️ 本机环境变量里有 http_proxy，不绕过它会被代理挡掉（老坑）
OP = urllib.request.build_opener(urllib.request.ProxyHandler({}))

ok = True


def check(label: str, cond: bool, extra: str = "") -> None:
    global ok
    if not cond:
        ok = False
    print(f"  [{'OK  ' if cond else 'FAIL'}] {label}{(' — ' + extra) if extra else ''}")


def get(path: str, token: str = ""):
    req = urllib.request.Request(BASE + path)
    if token:
        req.add_header("X-TW-Token", token)
    try:
        with OP.open(req, timeout=30) as r:
            return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers)


def post(path: str, payload: dict, token: str):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={"X-TW-Token": token, "Content-Type": "application/json"})
    try:
        with OP.open(req, timeout=30) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


print(f"打包产物实测：{EXE}")
check("exe 存在", EXE.is_file())
check("同级 _internal/ 存在", (EXE.parent / "_internal").is_dir())

# ⚠️ 故意换一个 cwd 启动：证明"路径靠相对位置自动正确"，
#    而不是碰巧当前工作目录帮了忙。
proc = subprocess.Popen(
    [str(EXE), "--no-window", "--port", str(PORT), "--quiet"],
    cwd=str(EXE.parent.parent), stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")

token = ""
try:
    # 从 stdout 抓"带令牌地址"——这也是保留控制台版的一个实际好处
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
    check("从启动输出里拿到 token", bool(token), "拿不到就只能测免鉴权路径")

    # ① 首页：离线可用（不加载任何外部资源）
    s, body, _ = get("/")
    html = body.decode("utf-8", "replace")
    check("GET / 返回 200", s == 200, f"status={s} bytes={len(body)}")
    check("首页含「交易世界」", "交易世界" in html)
    bad = [t for t in ("http://cdn", "https://cdn", "unpkg.com", "jsdelivr",
                       "googleapis") if t in html]
    check("首页不引用外部资源（离线可用）", not bad, f"引用了 {bad}" if bad else "")

    if token:
        # ② 元信息：证明 strategies/ 进了包
        s, body, _ = get("/api/meta", token)
        meta = json.loads(body) if s == 200 else {}
        check("GET /api/meta 返回 200", s == 200, f"status={s}")
        check("meta 含场景/策略",
              bool(meta.get("scenarios")) and bool(meta.get("strategies")),
              f"{len(meta.get('scenarios', []))} 场景 / "
              f"{len(meta.get('strategies', []))} 策略")

        # ③ 路由表：三个"另一组"的端点
        s, body, _ = get("/api/doc/reports", token)
        check("GET /api/doc/reports 返回 200", s == 200, f"status={s}")
        if s == 200:
            reps = json.loads(body)["reports"]
            check("文档验证列出 3 份报告", len(reps) == 3,
                  f"实际 {len(reps)}：{[r['title'] for r in reps]}")
            miss_j = [r["title"] for r in reps if not r["json_exists"]]
            miss_m = [r["title"] for r in reps if not r["report_exists"]]
            check("三份报告的 JSON 都随包带上了", not miss_j, f"缺 {miss_j}")
            check("三份报告的 md 也随包带上了", not miss_m, f"缺 {miss_m}")

        s, _, _ = get("/api/agent/runs", token)
        check("GET /api/agent/runs 返回 200", s == 200, f"status={s}")
        s, _, _ = get("/api/real/BTCUSDT_1h", token)
        check("GET /api/real/BTCUSDT_1h 返回 200（真实行情已随包）",
              s == 200, f"status={s}")

        # ④ 真跑一个作业：证明市场/评估/账本链路在打包后可用
        s, body = post("/api/run", {
            "kind": "strategy", "scenario": "normal", "seed": 7,
            "ticks": 200, "strategy": "mm_naive"}, token)
        check("POST /api/run 受理（202）", s == 202, f"status={s}")
        jid = json.loads(body)["job_id"] if s == 202 else ""
        if jid:
            job = {}
            for _ in range(400):
                time.sleep(0.25)
                _, b, _ = get("/api/job/" + jid, token)
                job = json.loads(b)
                if job.get("state") in ("done", "error"):
                    break
            check("作业跑到 done", job.get("state") == "done",
                  f"state={job.get('state')} err={job.get('error')}")
            if job.get("state") == "done":
                ev = job["result"]["exec"]
                check("有成交（n_fills>0）", ev["n_fills"] > 0, f"{ev['n_fills']} 笔")
                # PnL 三分解恒等式：评估代码的命门，端到端也必须成立
                check("PnL 三分解恒等式成立",
                      abs(ev["pnl_identity_residual"]) < 1e-6,
                      f"residual={ev['pnl_identity_residual']}")

            # ⑤ 导出：响应头 + BOM（只看状态码测不出来）
            s, b, hdrs = get(f"/api/export/{jid}.csv", token)
            disp = hdrs.get("Content-Disposition", "")
            check("CSV 导出 200", s == 200, f"status={s}")
            check("带 attachment 文件名头", "attachment" in disp, disp or "(无)")
            check("CSV 带 BOM（Excel 中文不乱码）", b.startswith(b"\xef\xbb\xbf"))

        # ⑥ 安全模型在打包后没失效
        s, _, _ = get("/api/meta")
        check("无 token 的 /api 请求被拒（403）", s == 403, f"status={s}")

    s, _, _ = get("/api/no-such-endpoint", token)
    check("未知接口 404", s == 404, f"status={s}")
finally:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()

print()
print("结论：" + ("全部通过 ✅" if ok else "有失败项 ❌"))
raise SystemExit(0 if ok else 1)
