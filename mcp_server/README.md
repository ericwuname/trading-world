# MCP 服务：让外部 Agent 操作「交易世界」

## 先分清「谁的 Agent」

| 场景 | 谁是 MCP 的客户端 | 什么在调工具 |
|---|---|---|
| **本项目的 LLM 交易 Agent** | 项目自己 | ❌ **不需要 MCP** —— 直接 `import tw.*` 更快更可控 |
| **外部 Agent**（Claude / Cursor / 其他） | 那些宿主 | ✅ **需要 MCP** |

⇒ **MCP 的价值是「让外部 Agent 能操作这个项目」**，不是给内部 Agent 用。
内部走 Python 函数调用；MCP 是**对外的一扇门**。

## 安装

`mcp` 是**可选依赖**。本项目核心库 `tw/*` **绝不 import 它** ——
不装 mcp 也能跑全部研究、全部测试。

```bash
pip install "mcp>=2.2.0"
```

### ⚠️ v2 改了 API：`FastMCP` → `MCPServer`（实测踩到）

网上**绝大多数教程仍是 v1 风格**：

```python
from mcp.server.fastmcp import FastMCP   # ❌ v2 上直接报 ModuleNotFoundError
```

v2（`mcp` 2.2.0）改名为：

```python
from mcp.server.mcpserver import MCPServer   # ✅ v2
```

实测报错原文：
> `No module named 'mcp.server.fastmcp'. This is mcp 2.x, where FastMCP was
> renamed to MCPServer (from mcp.server.mcpserver import MCPServer) and other
> APIs changed`

`mcp_server/__init__.py` 里的 `_require_mcp()` **两者都兼容**，优先 v2。

### 其他 v2 差异（实测）

| v1 写法 | v2 写法 |
|---|---|
| `info.serverInfo` | `info.server_info` |
| `info.protocolVersion` | `info.protocol_version` |
| `tool.inputSchema` | `tool.input_schema` |
| `res.structuredContent` | `res.structured_content` |
| `res.isError` | `res.is_error` |

实测协商到的协议版本是 **2025-11-25**（不是文档里常见的 2026-07-28）。
字段风格整体从 camelCase 转到了 snake_case。

**教训**：只读文档不试跑，会在「类名」这种最表层的地方翻车。
写 MCP 服务前**先 `pip install` 再 `import` 一次**，比读十篇教程有用。

## 运行

```bash
# 默认：stdio 传输，只读工具
python -m mcp_server

# 开启写入工具（会联网拉行情）
TW_MCP_ALLOW_WRITE=1 python -m mcp_server

# 只看有哪些工具（MCP Inspector）
mcp dev mcp_server
```

> ⚠️ `python -m mcp_server` 需要 `mcp_server/__main__.py`。
> 光有 `__init__.py` 里的 `if __name__ == "__main__"` **不会被执行**——
> 实测报 `No module named mcp_server.__main__; 'mcp_server' is a package`。
> 这与打包那次「入口脚本与包同名」是**同一类**问题（运行方式与文件结构不匹配），
> 而且报错不会告诉你缺哪个文件。

## 挂到 Claude Desktop / Cursor

在客户端的 MCP 配置里加：

```json
{
  "mcpServers": {
    "trading-world": {
      "command": "C:/Users/87465/.workbuddy/binaries/python/envs/py313/Scripts/python.exe",
      "args": ["-m", "mcp_server"],
      "cwd": "D:/0.个人文档/个人文档/交易世界/trading-world",
      "env": {
        "TW_MCP_ALLOW_WRITE": "0"
      }
    }
  }
}
```

`cwd` 必须指向项目根 —— 服务靠它找到 `data/market.sqlite`。

## 工具清单

### 只读（默认启用，无副作用）

| 工具 | 作用 |
|---|---|
| `list_instruments` | 库里有那些标的、覆盖到什么时间 |
| `get_candles` | 读 OHLCV（**列式**，省 token） |
| `get_market_summary` | 概览：最新价、区间涨跌、年化波动、波动聚集自相关 |
| `get_metrics` | 资金费率 / 未平仓量 / 标记价 |
| `list_scenarios` | 5 个市场场景 |
| `list_strategies` | 示例策略（含两个对照基准） |
| `get_fetch_ledger` | 拉取账本（这份数据什么时候拉的） |
| `get_db_stats` | 库的整体规模 |

### 写入（需 `TW_MCP_ALLOW_WRITE=1`）

| 工具 | 护栏 |
|---|---|
| `fetch_data` | 限频 60s/标的；`days` ≤ 1100；**不涉及账户与下单** |
| `generate_synthetic_data` | `n_bars` ≤ 50000；纯离线 |

### 交易（⚠️ **未实现，且这是有意的**）

见 `tools_trade.py` 的 docstring。一句话：
**A 路线（模拟）下不需要 MCP 下单；B 路线（真钱）需要单独一轮设计与评审**
（独立进程 / 单笔与日累计上限 / 人工确认 / 三级放行）。
在此之前不注册任何下单工具。

> 谁要绕过这里直接加 `place_order`，请先回答：
> **如果这个工具被误调用一次，损失上限是多少？** 答不上来就不该加。

## 三条实现纪律

1. **列式而非行式返回**。`[[ts,o,h,l,c,v],...]` 比
   `[{"ts":..,"open":..},...]` 省一大半 token（键名重复几千遍）。
   对 LLM 上下文来说是实打实的成本。
2. **一定带元信息**。每份数据都附 `source` / `range` / `n`，
   让调用方能判断「我拿到的是不是我要的」——
   与项目「报告里的数字要能追溯」是同一条纪律。
3. **不做隐式截断**。超限就**报错并说明上限**，不静默返回一部分
   （静默截断会让调用方以为拿全了）。

## 验证方式

- **单元测试**（不依赖 mcp 包）：`python -m unittest tests.test_mcp_server`
  —— 用假 FastMCP 收集注册的函数，检查工具清单、只读语义、护栏数值。
  刻意不 import mcp：**「没装可选依赖」不该让核心验证失效**。
- **协议层**：`mcp dev`（MCP Inspector）逐个调通，
  并**用一个真实外部客户端连一次**——自己写的客户端测不出兼容性问题。
- **安全回归**：测试里有一条明确断言「只读工具名里不得出现
  order/trade/buy/sell/cancel/position」。

## 安全边界

与本项目 GUI 那套**同一哲学**：

- **不绑端口**（stdio 传输，客户端把服务作为子进程拉起，走 stdin/stdout）
- 若改用 `streamable-http`（仅跨机场景），默认绑 `127.0.0.1`
- 写入能力需**显式环境变量**开启，不靠自觉
- 交易能力**根本没有实现**，不是「实现了但关着」
