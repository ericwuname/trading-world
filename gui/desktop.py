"""桌面窗口：优先用 pywebview 开原生窗口，失败则退回系统浏览器。

为什么要有回退
--------------
pywebview 在 Windows 上依赖 **WebView2 运行时**（Edge 自带，通常有）
和 **pythonnet**。缺任何一个都会在启动时抛异常。
如果那时候程序直接崩掉，用户看到的是一个栈回溯，
而它想做的事情（"看这个市场"）其实用浏览器完全能做。

所以这里的策略是：**窗口是外壳，不是功能。**
开得出原生窗口最好；开不出就开浏览器，功能一模一样，
只是少了那个没有地址栏的窗口。

线程模型（pywebview 的硬约束）
------------------------------
``webview.start()`` **阻塞主线程**，所以 HTTP 服务必须跑在后台线程里。
反过来，窗口只能在主线程创建。两者顺序是：

    起服务（后台线程）→ 探测端口可连 → create_window（主线程）→ start()（阻塞）
"""

from __future__ import annotations

import os
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from .server import GuiServer  # noqa: E402

WINDOW_TITLE = "交易世界 · 基于主体建模的人工市场"
WINDOW_SIZE = (1580, 1000)
MIN_SIZE = (1120, 720)
#: 与前端 CSS 的 --bg 保持一致，避免开窗瞬间闪白
BG = "#16181d"


def _default_workers() -> int:
    """工作线程数。模拟是纯 CPU 密集，开太多只会互相抢核。"""
    env = os.environ.get("TW_GUI_WORKERS")
    if env:
        try:
            return max(1, min(16, int(env)))
        except ValueError:
            pass
    cpu = os.cpu_count() or 4
    return max(1, min(4, cpu // 2))


def serve(workers: int | None = None, port: int = 0) -> GuiServer:
    """只起服务，不开窗口（给 --no-window 与自动化验证用）。"""
    srv = GuiServer(workers=workers or _default_workers(), port=port)
    srv.start_background()
    return srv


def run(
    *,
    workers: int | None = None,
    port: int = 0,
    window: bool = True,
    open_browser: bool = True,
) -> int:
    # 强制行缓冲。默认情况下 stdout 重定向到文件/管道时是块缓冲的，
    # 启动信息会一直卡在缓冲区里——用户看不到 URL，而我用 `timeout 20` 之类的
    # 方式结束进程时，这些字直接丢掉（实测：日志里只剩 WebView2 自己的报错）。
    # 启动信息里的"带令牌地址"是排查问题的第一手线索，不能丢。
    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
        sys.stderr.reconfigure(line_buffering=True)  # type: ignore[union-attr]
    except (AttributeError, ValueError):  # pragma: no cover
        pass

    srv = serve(workers=workers, port=port)
    if not srv.wait_ready():
        print("❌ HTTP 服务没能启动（端口探测失败）", file=sys.stderr)
        return 2

    print(f"   服务地址  {srv.base_url}")
    print(f"   带令牌  {srv.url}")
    print(f"   工作线程  {api_workers(srv)}")

    if not window:
        print("   （--no-window：未开窗口，按 Ctrl+C 结束）")
        try:
            while True:
                import time

                time.sleep(3600)
        except KeyboardInterrupt:
            pass
        finally:
            srv.stop()
        return 0

    try:
        import webview  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        print(f"   ⚠️ 无法加载 pywebview（{type(exc).__name__}: {exc}）")
        print("   → 退回浏览器模式")
        return _browser_mode(srv)

    try:
        webview.create_window(
            WINDOW_TITLE,
            srv.url,
            width=WINDOW_SIZE[0],
            height=WINDOW_SIZE[1],
            min_size=MIN_SIZE,
            background_color=BG,
            text_select=True,
        )
        webview.start(debug=os.environ.get("TW_GUI_DEBUG", "0") != "0")
    except Exception as exc:  # noqa: BLE001
        print(f"   ⚠️ 窗口启动失败（{type(exc).__name__}: {exc}）")
        print("   → 退回浏览器模式")
        return _browser_mode(srv)
    finally:
        srv.stop()
    return 0


def api_workers(srv: GuiServer) -> int:
    mgr = getattr(srv.httpd, "manager", None)
    return len(getattr(mgr, "_workers", [])) or 1


def _browser_mode(srv: GuiServer) -> int:
    if open_browser:
        webbrowser.open(srv.url)
        print("   ✅ 已在默认浏览器中打开。关闭本窗口即停止服务。")
    try:
        while True:
            import time

            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        srv.stop()
    return 0
