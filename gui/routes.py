"""HTTP 路由表 —— 「URL ↔ 处理函数」的**唯一事实源**。

为什么要有这张表
----------------
原来是 ``server.py::_api_get`` / ``_api_post`` 里十几个
``if path == ...`` / ``if path.startswith(...)`` 分支，而处理函数住在
``api.py`` / ``api_doc.py`` / ``agent_api.py``。于是"某个 URL 由谁处理"
这件事**被切成两半**：函数写在业务模块里，URL 映射写在 server 里。
加一个端点必须同时改两个文件——**只改一处不会报错**，
只会得到一个"函数写好了但谁也调不到"的 404。

⚠️ 这与 ``api.py::KINDS`` 是**同一个形状的问题、同一个修法**：
一处登记，其余全部派生。所以 agent 那组只读接口也照这个形状走
（见 ``agent_api.py`` 里的 ``@route``）。

⚠️ 但它们**不进** ``KINDS``：``KINDS`` 是"**作业类型**"表（提交进队列、
有进度、可取消、会写产物），而 agent 接口按设计是**只读、不起作业**的
（见 ``agent_api.py`` 的模块文档第 2 条）。统一的是
「URL 在哪里定义」，不是「把它们塞进同一张表」——
把只读接口塞进作业表，等于给"读数据"和"跑作业"开了同一个入口。

三种失效模式（都有测试与变异体盯着）
------------------------------------
1. **登记点漏一个模块** ⇒ 整组端点静默 404，代码看着完全正常。
   → 登记只在文件末尾那一处（见下面的「登记点」块）。
2. **模式退化成前缀匹配** ⇒ ``/api/meta/x`` 也命中 ``/api/meta``。
   → 一律用 ``fullmatch``，所以模式里**不用**写 ``$``。
3. **异常翻译做错地方** ⇒ 把内部 ``KeyError`` 一律当 404，
   就把服务端 bug 伪装成"你要的东西不存在"，排查方向被带偏。
   → 只翻译**该路由自己声明**的异常（``errors=``），其余照旧抛给 500。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


class ApiError(Exception):
    """业务错误：带 HTTP 状态码，前端直接展示 ``message``。

    ⚠️ 这个类**住在这里**，不是 ``api.py``：它属于 HTTP 层
    （"怎么回答一个请求"），而 ``api.py`` 与 ``agent_api.py`` 都要用它。
    若把它留在 ``api.py``，两个业务模块就都要先 import 另一个业务模块，
    于是 ``api.py`` 必须**先被完整导入**才能 import 别人 ——
    那是一个只靠 import 顺序成立的隐性约束，很容易被下一个开发者
    无意打破，症状是一句莫名其妙的 ImportError（而不是"你少写了个参数"）。

    ``api.py`` 会把它再导出一次，所以 ``api.ApiError`` 照旧可用。
    """

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status = status


@dataclass(slots=True)
class Ctx:
    """一次请求的上下文。所有路由函数都只收这一个参数。

    刻意**不**把 ``Handler``（socket）传进来：路由函数一旦能直接写响应，
    就迟早有人在这里绕开统一的序列化与错误翻译
    （``_jsonable`` 的 NaN 处理、``ApiError`` → 状态码），
    于是"白屏但服务端日志正常"这类问题会重新出现。

    ``root`` 是**服务端自己算出来的**项目根，不是 URL 里的东西 ——
    所以传它不构成"用户能指定看哪个文件"的口子
    （`agent_api.discover` 仍然只扫固定清单里的目录）。
    """

    method: str
    path: str
    query: dict
    body: dict
    root: Path
    mgr: Any
    params: dict[str, str]


@dataclass(slots=True)
class Response:
    """非 JSON 响应（目前只有 CSV 导出用）。

    存在的理由：CSV 要带 BOM、要带 ``Content-Disposition``，
    塞进 JSON 结构里表达反而绕。给一个明确的出口，
    好过在 handler 里偷偷拿到 socket。
    """

    body: bytes
    ctype: str = "application/json; charset=utf-8"
    status: int = 200
    headers: dict[str, str] | None = None


@dataclass(frozen=True, slots=True)
class Route:
    name: str
    method: str
    pattern: str
    handler: Callable[[Ctx], Any]
    #: "哪些异常该被翻译成哪个状态码"。**默认空** —— 未声明的异常
    #: 一律照旧冒到 500 并打 traceback，不允许悄悄变成 4xx。
    errors: dict[type, int] = field(default_factory=dict)

    def regex(self) -> re.Pattern[str]:
        return re.compile(self.pattern)

    def __str__(self) -> str:
        return f"{self.method} {self.pattern}"


#: 登记表。顺序 = 匹配优先级（都用 fullmatch，所以顺序基本无关，
#: 但保留声明顺序让 `describe()` 的输出可读、可 diff）。
ROUTES: list[Route] = []


def route(method: str, pattern: str, *, errors: dict[type, int] | None = None,
          name: str = ""):
    """把一个函数登记为某个 URL 的处理者。加端点只走这一条路。

    ``pattern`` 是**整条匹配**的正则（由 ``fullmatch`` 保证，不用写 ``$``）：
    简单的就是字面量 ``"/api/meta"``，带参数的写命名分组
    ``"/api/job/(?P<jid>[^/]+)"`` —— 分组会变成 ``ctx.params``。

    ``errors``: ``{异常类型: HTTP 状态码}``。**只声明你真正想翻译的**：
    例如按 id 取运行失败是"你要的东西不存在"（404），
    而一个意外的 ``TypeError`` 应当照旧是 500。
    """
    rx = pattern
    m = method.upper()

    def deco(fn: Callable[[Ctx], Any]) -> Callable[[Ctx], Any]:
        nm = name or fn.__name__
        for r in ROUTES:
            if r.name == nm:  # pragma: no cover - 只在开发期手滑时触发
                raise RuntimeError(f"路由名 {nm!r} 重复登记")
            if r.method == m and r.pattern == rx:  # pragma: no cover
                raise RuntimeError(f"路由 {m} {rx} 重复登记")
        ROUTES.append(Route(nm, m, rx, fn, dict(errors or {})))
        return fn

    return deco


def try_match(method: str, path: str) -> tuple[Route, re.Match[str]] | None:
    """按登记顺序找第一条命中的路由。

    ⚠️ 用 ``fullmatch`` 而不是 ``match``：``match`` 是**前缀**匹配，
    于是 ``/api/meta/x`` 会命中 ``/api/meta``，
    表现为"多打一段路径居然还返回 200"—— 这种错很难被注意到，
    却是后面加子路径时踩雷的起点。
    """
    m = method.upper()
    for r in ROUTES:
        if r.method != m:
            continue
        hit = r.regex().fullmatch(path)
        if hit is not None:
            return r, hit
    return None


def describe() -> list[str]:
    """路由清单（自检与排错用）。"""
    return [str(r) for r in ROUTES]


def query_int(query: dict, key: str, default: int) -> int:
    """从 ``parse_qs`` 的结果里取一个整数（URL 查询参数）。

    ⚠️ ``parse_qs`` 的值一律是**列表**（同一参数可以出现多次），
    所以 ``int(query.get(k))`` 会抛 TypeError，而用户看到的是
    **HTTP 500** 而不是"参数写错了"——把甲方错误报成服务端错误，
    排查方向会被带偏。这里显式取第一个值，并给出可读的 400。

    为什么住在这里而不是 ``api.py``：它处理的是 **query string**
    （值是列表），而 `api._int` 处理的是 **JSON 请求体**里的字段
    （值是标量）。两者长得像、混用就得到上面那个 500，
    所以刻意放进两个模块、名字也取成不同的，强迫调用者想一下
    "我手里这个是 urlencoded 的字典，还是 JSON 体？"
    """
    v = query.get(key)
    if v is None:
        return int(default)
    first = v[0] if isinstance(v, (list, tuple)) and v else v
    try:
        return int(str(first).strip() or default)
    except (TypeError, ValueError):
        raise ApiError(f"{key} 必须是整数，收到 {first!r}") from None


# ======================================================================
# 登记点
# ======================================================================
# ⚠️ 模块级副作用（注册）**由谁触发**是"拆模块"这一类重构的专属坑：
#    被搬走的代码本身没坏，坏的是没人去调用它 —— 而且**不报错**，
#    只是那组端点静默变成 404。本项目已经踩过两次
#    （`api.py::KINDS` 里的 doc_strategy、以及这次的路由）。
#
# ⭐ 本块的真实作用（这一条是**实测修正**过的，别按直觉写）：
#    它是**完整性声明** —— 新增一个带路由的业务模块时，在这里加一行，
#    声明"它也参与路由登记"。
#    ⚠️ 但它**不是**唯一入口：`server.py` 自己也 import 了 `api` 与
#    `agent_api`，而 `api.py` 末尾又 import 了 `api_doc`，
#    所以**删掉本块里的行，端点并不会消失**。
#    —— 我原本以为"删掉这里 = 端点静默 404"，并按这个假设写了
#    变异体 M127；实测它**仍全绿**。那次全绿不是测试漏了，
#    是**我的判断错了**（那行不是触发点）。M127 已改成真正会
#    静默失效的那件事（漏写 `@route` 装饰器）。
#    ⇒ 教训：**"这行 import 是登记点"这种话必须用变异体验一遍再说**，
#    否则注释本身就成了误导下一个人的东西。
#
# ⚠️ 为什么 `api_doc` **不能**写在本块里（这是排列顺序的硬约束，不是风格）：
#    `api_doc.py` 在 import 时就要用 `api.KINDS` 与 `register_kind` 挂钩子，
#    所以它必须等 `api` **完全**初始化完毕。
#    而本块是在 `api` 初始化**过程当中**被执行的（`api` 顶部 import 本模块），
#    此时 `api.KINDS` 还不存在 ⇒ 在这里 import `api_doc` 会直接 ImportError。
#    ⇒ 只能由 `api` 自己在末尾触发（`api.py` 那句 `import gui.api_doc`，
#    变异体 M125 盯着它 —— 那一处才是**真**触发点，删掉端点真的会消失）。
#
# 顺序：`api` 排在 `agent_api` 之前（无实质依赖，只为让登记顺序稳定、
# `describe()` 的输出可 diff）。
from . import api as _api  # noqa: E402,F401
from . import agent_api as _agent_api  # noqa: E402,F401

__all__ = [
    "ApiError",
    "Ctx",
    "Response",
    "Route",
    "ROUTES",
    "route",
    "try_match",
    "describe",
    "query_int",
]
