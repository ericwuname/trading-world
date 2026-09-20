"""MCP 服务入口——把行情库与本项目的能力暴露给**外部** Agent。

先分清「谁的 Agent」
-------------------
=========================  ====================  ===========================
场景                          谁是 MCP 的客户端      什么在调工具
=========================  ====================  ===========================
本项目的 LLM 交易 Agent       项目自己              ❌ 不需要 MCP，直接 import tw/* 更快
外部 Agent（Claude/Cursor）   那些宿主              ✅ 需要 MCP
=========================  ====================  ===========================

⇒ **MCP 的价值是「让外部 Agent 能操作这个项目」**，不是给内部 Agent 用。
内部走 Python 函数调用；MCP 是**对外的一扇门**。

安全边界（与 GUI 那套同一哲学）
------------------------------
- 默认只开**只读工具**（看行情、看实验、看留痕），无副作用
- 写入类工具（拉数据）需 ``TW_MCP_ALLOW_WRITE=1``
- 交易类工具需 ``TW_MCP_ALLOW_TRADE=1``，**默认完全没有注册**
- **不绑端口**：stdio 传输，客户端把服务作为子进程拉起，走 stdin/stdout

依赖
----
``mcp`` 是**第三方包**，而本项目核心依赖只有 numpy/scipy/matplotlib。
所以：**核心库 tw/* 绝不 import mcp**，只有这个目录需要它。
没装时给出明确的安装指引，不静默失败。
"""

from __future__ import annotations

__all__ = ["build_server", "main"]


def _require_mcp():
    """取得 v2 的 ``MCPServer``。

    ⚠️ **v2 把 ``FastMCP`` 改名成了 ``MCPServer``**（`mcp.server.mcpserver`）。
    网上大量教程仍是 v1 的 ``from mcp.server.fastmcp import FastMCP``，
    照抄会在 v2 上报 ``ModuleNotFoundError`` —— 实测踩到。
    这里**兼容两者**，优先 v2：
    """
    try:
        from mcp.server.mcpserver import MCPServer

        return MCPServer, "v2"
    except ImportError:
        pass
    try:
        from mcp.server.fastmcp import FastMCP  # type: ignore[no-redef]

        return FastMCP, "v1"
    except ImportError as exc:  # pragma: no cover — 取决于环境
        raise SystemExit(
            "缺少 mcp 包。这是**可选依赖**（本项目核心只用标准库 + "
            "numpy/scipy/matplotlib），装上它才能跑 MCP 服务：\n\n"
            '    pip install "mcp>=2.2.0"\n\n'
            "原始错误：" + str(exc)
        ) from exc


def build_server(  # pragma: no cover — 需要 mcp 包
    *,
    allow_write: bool | None = None,
    allow_trade: bool | None = None,
):
    """构造 MCP 服务。工具按「读写分离 + 危险度分级」注册。

    返回值是 v2 的 ``MCPServer``（或 v1 的 ``FastMCP``）。
    ``mcp`` 未安装时抛出 ``SystemExit`` 并给出指引。
    """
    import os

    ServerCls, _ver = _require_mcp()

    if allow_write is None:
        allow_write = os.environ.get("TW_MCP_ALLOW_WRITE", "0") == "1"
    if allow_trade is None:
        allow_trade = os.environ.get("TW_MCP_ALLOW_TRADE", "0") == "1"

    mcp = ServerCls("trading-world")

    # 注册三类工具（分子模块，便于测试逐个检查）
    from . import tools_read, tools_write

    tools_read.register(mcp)
    if allow_write:
        tools_write.register(mcp)
    if allow_trade:
        # ⚠️ 交易工具**故意不在这里实现**。
        # A 路线（模拟）阶段它们该走项目内部的执行器；
        # B 路线（真钱）需要独立进程 + 独立 token + 单笔/日累计上限 +
        # 人工确认开关——那是单独一轮的工作，不是顺手加个函数。
        from . import tools_trade

        tools_trade.register(mcp)

    return mcp


def main() -> int:  # pragma: no cover — 进程入口
    import os

    mcp = build_server()
    transport = os.environ.get("TW_MCP_TRANSPORT", "stdio")
    if transport == "stdio":
        mcp.run(transport="stdio")
    else:
        # 仅当要跨机时用（未来多 VM 场景）。默认绑 127.0.0.1，不对外。
        mcp.settings.host = os.environ.get("TW_MCP_HOST", "127.0.0.1")
        mcp.settings.port = int(os.environ.get("TW_MCP_PORT", "8765"))
        mcp.run(transport="streamable-http")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
