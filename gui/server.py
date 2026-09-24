"""HTTP 服务：只用 Python 标准库。

安全模型（本地工具的标准做法）
------------------------------
1. **只绑定 127.0.0.1**。别的机器连不上，端口也不对外暴露。
2. **每个进程启动生成一次性 token**，所有 ``/api`` 请求必须带
   ``X-TW-Token``。token 通过启动 URL 的查询参数交给前端，前端存在内存里。
3. **不返回任何 CORS 头**。别的网页就算知道端口号，
   也没法读响应；更关键的是——带自定义头的跨源请求会先触发预检，
   而预检拿不到允许头，浏览器会直接掐掉。
   所以"token 头"这一条同时就是 CSRF 防护。
4. **校验 `Host` 头必须是回环地址**，拦住 DNS rebinding
   （让恶意域名解析到 127.0.0.1 来绕过同源策略的那类攻击）。

仍然要说清楚：本服务允许在界面里跑自定义 Python 策略代码
（见 ``gui/api.py::compile_strategy``），那等于在本机执行任意代码。
上面四条防的是"别的网页/别的机器"，防不了"你自己粘贴了一段恶意代码"。
不要运行来源不明的策略。
"""

from __future__ import annotations

import json
import mimetypes
import os
import secrets
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from . import __version__, agent_api, api  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"


def _json_safe(o):
    """NaN/Inf → None（递归）。见 ``tw.eval_agent.json_safe`` 的文档。"""
    from tw.eval_agent import json_safe as _f
    return _f(o)


def _inject_agent_preload(html: bytes, query: dict,
                          root: Path | None = None) -> bytes:
    """把首屏数据作为 ``<script type="application/json">`` 嵌进 HTML。

    ⚠️ **必须转义 ``</``**：JSON 里若出现 ``</script>``（比如某条决策的
    理由里恰好写了它），浏览器会**提前结束脚本块**——那是注入，不是转义
    的小毛病。JSON 里 ``<\\/`` 与 ``</`` 等价，替换后语义不变。

    ``root`` 只为测试可注入而留（默认项目根）。**安全上不构成口子**：
    它是 Python 侧参数，不是用户输入——URL 里能控制的只有 ``run``，
    而那最终仍然要过 ``agent_api`` 的"只在扫描结果里查"。
    """
    try:
        rid = (query.get("run") or [""])[0]
        data = agent_api.first_paint(root or ROOT, rid)
        # ⚠️ 同 CLI：NaN 不是合法 JSON，嵌进页面会让 JSON.parse 直接抛错，
        # 整页退化成"读取失败"——而 Python 侧一切正常（它接受 NaN）。
        blob = json.dumps(_json_safe(data), ensure_ascii=False,
                          default=str, allow_nan=False)
    except Exception as exc:  # noqa: BLE001
        # ⚠️ 嵌数据失败**不能让页面打不开**：退回纯异步路径
        # （前端发现没有预载块时会自己去取）。降级要比功能重要。
        print(f"[gui] 首屏预载失败（退回异步加载）：{type(exc).__name__}: {exc}")
        return html
    blob = blob.replace("</", "<\\/")
    tag = ('<script id="agentPreload" type="application/json">'
           + blob + "</script>\n</body>")
    return html.decode("utf-8").replace("</body>", tag, 1).encode("utf-8")


def _qint(query: dict, key: str, default: int) -> int:
    """从 ``parse_qs`` 的结果里取一个整数。

    ⚠️ ``parse_qs`` 的值一律是**列表**（同一参数可以出现多次），
    所以 ``int(query.get(k))`` 会抛 TypeError，而用户看到的是
    **HTTP 500** 而不是"参数写错了"——把甲方错误报成服务端错误，
    排查方向会被带偏。这里显式取第一个值，并给出可读的 400。
    """
    v = query.get(key)
    if v is None:
        return int(default)
    first = v[0] if isinstance(v, (list, tuple)) and v else v
    try:
        return int(str(first).strip() or default)
    except (TypeError, ValueError):
        raise ValueError(f"{key} 必须是整数，收到 {first!r}") from None


#: 端口 0 = 让系统分配空闲端口。写死端口会在"上一次没退干净"时启动失败，
#: 而用户看到的只是一个 bind 错误，很难自己想明白。
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 0

_LOOPBACK = {"127.0.0.1", "localhost", "[::1]", "::1"}


class Handler(BaseHTTPRequestHandler):
    server_version = f"TradingWorldGUI/{__version__}"
    protocol_version = "HTTP/1.1"
    #: 关闭访问日志：默认每个请求打一行，前端每秒轮询两次状态，
    #: 控制台会被刷屏到看不见真正有用的信息。
    _quiet = os.environ.get("TW_GUI_VERBOSE", "0") != "0"

    # --- 基础 ---------------------------------------------------------
    def log_message(self, fmt: str, *args) -> None:  # noqa: ANN002
        if self._quiet:
            sys.stderr.write(f"[gui] {self.address_string()} {fmt % args}\n")

    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False, default=api._jsonable).encode("utf-8")
        self._send(code, body, "application/json; charset=utf-8")

    def _err(self, code: int, message: str) -> None:
        self._json({"error": message, "status": code}, code)

    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").split(":")[0].strip()
        return host in _LOOPBACK

    def _token_ok(self, query: dict) -> bool:
        want = getattr(self.server, "token", "")
        got = self.headers.get("X-TW-Token") or (query.get("token") or [""])[0]
        return bool(want) and secrets.compare_digest(str(got), want)

    # --- 路由 ---------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        self._dispatch("GET")

    def do_HEAD(self) -> None:  # noqa: N802
        self._dispatch("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch("POST")

    def _dispatch(self, method: str) -> None:
        if not self._host_ok():
            # 只有回环地址能访问：拦住 DNS rebinding
            self._err(403, "只接受来自本机的请求")
            return
        u = urlparse(self.path)
        path = unquote(u.path)
        query = parse_qs(u.query)

        try:
            if path == "/" or path == "/index.html":
                return self._static("index.html", query=query)
            if path.startswith("/static/"):
                return self._static(path[len("/static/") :])
            if path.startswith("/api/"):
                if not self._token_ok(query):
                    return self._err(403, "缺少或错误的访问令牌 —— 请通过启动器打开界面")
                if method == "POST":
                    return self._api_post(path, self._read_json())
                return self._api_get(path, query)
            return self._err(404, "没有这个路径")
        except api.ApiError as exc:
            self._err(exc.status, exc.message)
        except BrokenPipeError:
            # 用户关窗口/刷新时轮询请求会断开。这是正常现象，不是错误。
            pass
        except Exception as exc:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            self._err(500, f"{type(exc).__name__}: {exc}")

    def _read_json(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            return {}
        if n > 4_000_000:
            raise api.ApiError("请求体过大", 413)
        raw = self.rfile.read(n)
        try:
            obj = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise api.ApiError(f"请求体不是合法 JSON：{exc}") from None
        if not isinstance(obj, dict):
            raise api.ApiError("请求体必须是 JSON 对象")
        return obj

    # --- 静态文件 -----------------------------------------------------
    def _static(self, rel: str, query: dict | None = None) -> None:
        rel = rel.lstrip("/\\")
        target = (STATIC / rel).resolve()
        # 目录穿越防护：解析后的路径必须仍在 static/ 里面
        try:
            target.relative_to(STATIC.resolve())
        except ValueError:
            return self._err(403, "非法路径")
        if not target.is_file():
            return self._err(404, f"找不到 {rel}")
        body = target.read_bytes()
        # ⭐ A5：深链到 Agent 页时，把**首屏数据嵌进 HTML**。
        # 理由见 ``agent_api.first_paint`` 的文档：异步首绘会让
        # 无头截图/DOM dump 拍到占位图，而"检查有没有内容"会**通过**
        # ——验证方式骗人比 bug 更危险。嵌数据后首屏是同步的、确定的。
        # ⚠️ 只在 ``index.html`` 且显式带了 ``tab=agent`` 时做，
        # 其它情况一个字节都不改。
        if (target.name == "index.html" and query
                and (query.get("tab") or [""])[0] == "agent"):
            body = _inject_agent_preload(body, query)
        ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript",):
            ctype += "; charset=utf-8"
        self._send(200, body, ctype)

    # --- API: GET -----------------------------------------------------
    def _api_get(self, path: str, query: dict) -> None:
        mgr = self.server.manager  # type: ignore[attr-defined]
        if path == "/api/meta":
            return self._json(api.meta())
        if path == "/api/jobs":
            return self._json({"jobs": [j.public(with_result=False) for j in mgr.list()],
                               "stats": mgr.stats()})
        if path.startswith("/api/job/"):
            rest = path[len("/api/job/") :]
            if rest.endswith("/series"):
                job = mgr.get(rest[: -len("/series")])
                if job is None:
                    return self._err(404, "作业不存在")
                if job.series is None:
                    return self._json({"ready": False, "state": job.state})
                return self._json({"ready": True, "series": job.series})
            job = mgr.get(rest)
            if job is None:
                return self._err(404, "作业不存在")
            return self._json(job.public())
        if path == "/api/doc/reports":
            return self._json(api.doc_reports_payload())
        if path.startswith("/api/real/"):
            return self._json(api.real_payload(path[len("/api/real/") :]))
        # ---- A5：Agent 决策留痕（**只读**，路径不接受用户输入）--------
        if path == "/api/agent/runs":
            return self._json(agent_api.list_runs(ROOT))
        if path.startswith("/api/agent/run/"):
            rest = path[len("/api/agent/run/") :]
            try:
                if rest.endswith("/decisions"):
                    rid = rest[: -len("/decisions")]
                    # ⚠️ ``parse_qs`` 的值是**列表**（同一参数可出现多次）。
                    # 直接 ``int(query.get(...))`` 会抛 TypeError →
                    # 用户看到的是 500，而不是"参数写错了"。
                    return self._json(agent_api.decisions_page(
                        ROOT, rid,
                        offset=_qint(query, "offset", 0),
                        limit=_qint(query, "limit", 50),
                    ))
                if rest.endswith("/summary"):
                    return self._json(agent_api.run_summary(
                        ROOT, rest[: -len("/summary")]))
                if "/decision/" in rest:
                    rid, _, idx = rest.partition("/decision/")
                    return self._json(agent_api.decision_detail(ROOT, rid, int(idx)))
                return self._json(agent_api.run_summary(ROOT, rest))
            except KeyError as e:
                return self._err(404, str(e))
            except ValueError as e:
                return self._err(400, f"参数不合法：{e}")
        if path.startswith("/api/export/"):
            name = path[len("/api/export/") :]
            job_id = name[:-4] if name.endswith(".csv") else name
            job = mgr.get(job_id)
            if job is None:
                return self._err(404, "作业不存在")
            csv = api.export_csv(job).encode("utf-8-sig")  # BOM：Excel 打开中文不乱码
            return self._send(
                200, csv, "text/csv; charset=utf-8",
                {"Content-Disposition": f'attachment; filename="tw-{job_id}.csv"'},
            )
        return self._err(404, "没有这个接口")

    # --- API: POST ----------------------------------------------------
    def _api_post(self, path: str, body: dict) -> None:
        mgr = self.server.manager  # type: ignore[attr-defined]
        if path == "/api/run":
            kind = str(body.get("kind") or "strategy")
            body["kind"] = kind
            # 提交前先全量校验：参数错了要在**毫秒内**告诉用户，
            # 而不是让他等作业跑完再从日志里翻原因。
            api.validate_spec(body)
            job = mgr.submit(kind, body, api.execute)
            return self._json({"job_id": job.id, "kind": kind}, 202)
        if path.startswith("/api/job/") and path.endswith("/cancel"):
            job_id = path[len("/api/job/") : -len("/cancel")]
            ok = mgr.cancel(job_id)
            return self._json({"cancelled": ok})
        return self._err(404, "没有这个接口")


class _Server(ThreadingHTTPServer):
    """带"安静断连"处理的 ThreadingHTTPServer。

    为什么需要它：用户**关掉窗口或刷新页面**时，浏览器会直接把 TCP 连接掐断，
    正在处理的请求于是抛 `ConnectionResetError`。这个异常从
    `handle_one_request` 冒到 `socketserver.process_request_thread`，
    被 `handle_error` 打印成一大段 traceback。

    后果不是功能坏了，而是**控制台被吓人的报错刷屏**——
    用户会以为程序出错了，而它其实只是正常关了个窗口。
    更糟的是：真正的错误会被这些噪声淹掉。
    """

    daemon_threads = True

    def handle_error(self, request, client_address) -> None:  # noqa: ANN001
        import sys
        import traceback

        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionResetError, ConnectionAbortedError, BrokenPipeError)):
            # 正常的客户端断开，不是错误。留一行短提示即可。
            if os.environ.get("TW_GUI_VERBOSE", "0") != "0":
                sys.stderr.write(f"[gui] 客户端断开 {client_address}\n")
            return
        traceback.print_exc()


class GuiServer:
    """把 HTTP 服务跑在后台线程里，并暴露访问地址。"""

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
                 workers: int = 1) -> None:
        self.token = secrets.token_urlsafe(24)
        self.httpd = _Server((host, port or 0), Handler)
        self.httpd.token = self.token          # type: ignore[attr-defined]
        self.httpd.manager = api.manager(workers)  # type: ignore[attr-defined]
        self.host, self.port = self.httpd.server_address[:2]
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        # IPv6 地址要加方括号才是合法 URL
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"http://{host}:{self.port}"

    @property
    def url(self) -> str:
        return f"{self.base_url}/?token={self.token}"

    def start(self) -> None:
        self.httpd.serve_forever(poll_interval=0.2)

    def start_background(self) -> None:
        self._thread = threading.Thread(target=self.start, name="tw-http", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        try:
            self.httpd.shutdown()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.httpd.server_close()
        except Exception:  # noqa: BLE001
            pass

    def wait_ready(self, timeout: float = 5.0) -> bool:
        """等到端口真的能连上再开窗口。

        直接 create_window 会偶发白屏：服务线程刚起来、还没开始 listen，
        窗口已经开始请求了。主动探测一次最稳。
        """
        import time

        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                with socket.create_connection((self.host, self.port), timeout=0.25):
                    return True
            except OSError:
                time.sleep(0.05)
        return False
