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

⚠️ 无控制台的打包版（`console=False`）与"看得见降级"
----------------------------------------------------
打包成无控制台后 `sys.stdout`/`sys.stderr` 都是 `None`，`print` 变成
**静默 no-op** ⇒ 上面那三条降级/失败路径会变成「**双击没反应**」。
所以凡是"必须让人看到"的话一律走 :func:`_alert`：
有控制台就打印，**没有就弹原生消息框**（stdlib ctypes，不写文件）。

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


def _has_console() -> bool:
    """当前进程有没有真的能写字的控制台。

    ⚠️ 打包成**无控制台**（`console=False`）的 exe 里，PyInstaller 会把
    `sys.stdout` / `sys.stderr` 都设成 `None` —— 此时 `print` 是**静默 no-op**。
    （CPython 的 `print` 遇到 `file=None` 会退回 `sys.stdout`，而它是 None
    就直接返回，不报错。）所以"有没有控制台"必须显式判，不能靠 print。
    """
    return sys.stdout is not None and sys.stderr is not None


def _alert(msg: str, *, kind: str = "warn") -> None:
    """说一句**必须让人看到**的话：没有控制台时弹原生消息框。

    ⚠️ 为什么不能只 `print`：无控制台的打包版里 print 会被丢掉 ⇒
    失败变成「**双击没反应**」，正是本项目最不想再踩的静默失败。
    ⇒ 没有控制台时用 stdlib `ctypes` 弹一个消息框
    （**不写任何文件** —— 安装说明里"本程序不写用户数据"那句话必须仍然成立）。

    ⚠️ 有控制台时**不弹框**：一是没必要，二是测试环境有 stdout，
    弹框会把自动化卡在"等人点确定"上。于是"弹不弹"由"有没有控制台"决定，
    而那个条件在测试里天然为假 ⇒ **这条逻辑是可测的**。
    """
    if _has_console():
        print(msg, file=sys.stderr)
        return
    try:
        import ctypes

        icon = {"error": 0x10, "warn": 0x30, "info": 0x40}.get(kind, 0x30)
        ctypes.windll.user32.MessageBoxW(None, msg, WINDOW_TITLE, icon)
    except Exception:  # noqa: BLE001 - 弹框失败不能带走主流程
        pass


def serve(workers: int | None = None, port: int = 0, token: str = "") -> GuiServer:
    """只起服务，不开窗口（给 --no-window 与自动化验证用）。"""
    srv = GuiServer(workers=workers or _default_workers(), port=port, token=token)
    srv.start_background()
    return srv


def run(
    *,
    workers: int | None = None,
    port: int = 0,
    window: bool = True,
    open_browser: bool = True,
    token: str = "",
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

    srv = serve(workers=workers, port=port, token=token)
    if not srv.wait_ready():
        _alert("HTTP 服务没能启动（端口探测失败）。\n"
               "常见原因：端口被占用。可换一个 --port 再试。", kind="error")
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
        _alert(f"原生窗口不可用（{type(exc).__name__}: {exc}）\n"
               f"已改用系统浏览器打开 —— 功能完全一样。", kind="info")
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
        _alert(f"窗口启动失败（{type(exc).__name__}: {exc}）\n"
               f"已改用系统浏览器打开 —— 功能完全一样。", kind="info")
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
