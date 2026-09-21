"""LLM 通道（A3）—— 把"模型"抽象成一个**可替换、可回放**的东西。

为什么这一层要单独存在
----------------------
如果把 ``urllib.request`` 直接写进交易回路里，会立刻出现三个后果：

1. **测试就不可能是"离线可跑"的**——跑一次测试要真的发网络请求，
   于是要么慢、要么不稳定、要么得联网。本项目其余部分全部是离线可测的，
   这一层不该是例外。
2. **回放就做不成了**。A2 的留痕设计里有一条硬要求：
   「同 ``visible_state`` ⇒ 同 ``parsed``」，用来验证**管线没有隐藏状态**。
   如果每次回放都要重新问一遍模型，那验证的就不是管线，而是模型的随机性。
3. **"模型坏了"和"网络坏了"分不清**。探测脚本（``scripts/probe_llm.py``）
   当初单独存在就是为了这个；接进回路后必须继续分得清。

所以本模块把"怎么发请求"（``Transport``）和"发什么、怎么解释结果"
（``LLMClient``）**拆开**：

    ┌──────────────┐     ┌───────────────┐
    │ LLMClient    │────▶│ Transport     │  真实: urllib
    │ （重试/超时  │     │ （只管收发）  │  测试: 内存假实现
    │  /记账/回放）│     └───────────────┘  回放: 读 JSONL
    └──────────────┘

**A3 的全部测试因此是离线的**——没有 key、没有网络也能把重试、超时、
解析失败、回放确定性测穷。真实调用只在"真机验证"那一步发生一次。

⚠️ 三条必须记住的约定
--------------------
1. **key 只从环境变量或「仓库外」的 secrets 文件读**，
   绝不进代码、**仓库内**配置或留痕。
   留痕里只记"用了哪个 provider / 哪个变量名"，不记值。

   查找顺序（自上而下，先命中先用）：

   1. ``api_key`` 参数（**测试**注入用，不要在生产代码里传）
   2. 环境变量 ``LLMConfig.api_key_env``
   3. ``~/.workbuddy/secrets/<provider>.json`` 里的 ``api_key`` 字段

   为什么不只留环境变量：把 key 写在 shell 命令行上（
   ``export AGNES_KEY=...`` / ``AGNES_KEY=... python ...``）
   会进 shell 历史、进进程表、进 CI 日志，而且**每次运行都要重新粘一遍**。
   本项目对 OKX 凭证已经采用"放仓库外的专用文件"这条约定
   （见用户级记忆），LLM key 沿用同一条。
   ⚠️ 注意 secrets 目录在 ``.gitignore`` 里——它**不该**进仓库。
2. **没有 key 时必须显式失败，不能静默退化成 hold**。
   ⚠️ 这是最容易埋的坑：一个"没 key 就返回 hold"的实现，
   跑起来**看起来完全正常**——每条决策都记了、都合规、都弃权，
   只是这个 Agent 永远不交易。它会把"系统是坏的"伪装成"Agent 很保守"。
   ⇒ 所以本模块的约定是：**构造期就检查 key**，缺了直接抛。
3. **``prompt_hash`` 是回放的等价关系**：同样的 messages + model +
   temperature ⇒ 同一个 hash ⇒ 允许从回放记录里取同一条响应。
   它是 ``DecisionRecord.decision_id`` 里 ``context_hash`` 的上游。

回放的边界（必须诚实）
----------------------
回放**能**验证：给定同一份 prompt，管线是否产出同样的 ``parsed``
（即：决策的可复现性、以及"我们真的把该给的信息都给了模型"）。
回放**不能**验证：模型在当时会不会真的这么答。它是一条**忠实**的
重演路径，不是模型的复制品。这与 A2 文档里 ``visible_state`` 回放
的边界是同一条声明。
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Protocol

# ======================================================================
# 通道注册表
# ======================================================================
#: 各通道的默认端点与模型。
#: ⚠️ ``api_key_env`` 是**变量名**，不是值——值永远只在环境里。
PROVIDERS: dict[str, dict[str, str]] = {
    "agnes": {
        "base_url": "https://api.agnes-ai.cn/v1",
        "model": "agnes-2.5-flash",
        "api_key_env": "AGNES_KEY",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "api_key_env": "OPENAI_API_KEY",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "api_key_env": "DEEPSEEK_API_KEY",
    },
    "zhipu": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4-flash",
        "api_key_env": "ZHIPU_API_KEY",
    },
    "gemini": {
        # Gemini 的 OpenAI 兼容层（用户会员通道，实测可用）
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "model": "gemini-3.6-flash",
        "api_key_env": "GEMINI_API_KEY",
    },
    "ollama": {
        # 本机 Ollama，无需 key（api_key_env 留空 ⇒ 不检查）
        "base_url": "http://localhost:11434/v1",
        "model": "qiyuan-8b:latest",
        "api_key_env": "",
    },
}


# ======================================================================
# 配置
# ======================================================================
@dataclass(slots=True)
class LLMConfig:
    """通道配置。

    ``retries`` 的语义是**总尝试次数**（1 = 不重试）。
    ⚠️ 不要写成"重试次数 + 1"两处口径——那种差一错误在日志里表现为
    "偶发少一次尝试"，极难查。
    """

    provider: str = "agnes"
    model: str = ""
    base_url: str = ""
    api_key_env: str = ""
    temperature: float = 0.2
    timeout: float = 60.0
    retries: int = 3
    max_tokens: int = 700

    def __post_init__(self) -> None:
        if self.provider not in PROVIDERS and not self.base_url:
            raise ValueError(
                f"未知 provider {self.provider!r}；"
                f"要么用已注册的 {sorted(PROVIDERS)}，要么显式给 base_url"
            )
        d = PROVIDERS.get(self.provider, {})
        if not self.model:
            self.model = d.get("model", "")
        if not self.base_url:
            self.base_url = d.get("base_url", "")
        if not self.api_key_env:
            self.api_key_env = d.get("api_key_env", "")
        if not self.model:
            raise ValueError("model 为空：请在配置里显式指定")
        if not self.base_url:
            raise ValueError("base_url 为空：请在配置里显式指定")
        if self.retries < 1:
            raise ValueError(f"retries 是总尝试次数，必须 >= 1，收到 {self.retries}")

    @property
    def url(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"

    def describe(self) -> dict[str, Any]:
        """**可安全写进留痕**的配置描述（不含任何密钥值）。"""
        return {
            "provider": self.provider,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "retries": self.retries,
            # ⚠️ 只记变量名，不记值
            "api_key_env": self.api_key_env,
        }


# ======================================================================
# 响应
# ======================================================================
@dataclass(slots=True)
class LLMResponse:
    """一次调用的结果。**失败也是一种结果**，必须能进留痕。"""

    ok: bool = False
    text: str = ""
    error: str = ""
    latency_ms: int = 0
    attempts: int = 0
    usage: dict[str, Any] = field(default_factory=dict)
    #: prompt 的确定性哈希（回放的等价关系，见模块文档）
    prompt_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "error": self.error,
            "latency_ms": self.latency_ms,
            "attempts": self.attempts,
            "usage": self.usage,
            "prompt_hash": self.prompt_hash,
            # ⚠️ text 不在这里——它要**原文不截断**地进 DecisionRecord.llm_raw，
            # 塞进这个 dict 会变成两份，迟早不一致。
        }


# ======================================================================
# prompt 哈希
# ======================================================================
def prompt_hash(messages: list[dict[str, Any]], *, model: str,
                temperature: float, max_tokens: int = 0) -> str:
    """``messages`` + 采样参数 的确定性哈希。

    ⚠️ 必须含 ``temperature``：同一份 prompt 在 0.2 与 1.0 下
    是**两个不同的实验**，哈希一样会让 A/B 的两条记录混成一条。
    这与 ``decision_id`` 必须含 ``prompt_template`` 是同一条道理。

    用 ``sort_keys=True`` + 固定分隔符：字典顺序不该影响哈希。
    """
    blob = json.dumps(
        {
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


# ======================================================================
# Transport —— 只管收发
# ======================================================================
#: ``(url, body_bytes, headers, timeout) -> 解析后的 JSON dict``。
#: 抛异常表示失败；返回 dict 表示 HTTP 层成功。
Transport = Callable[[str, bytes, dict[str, str], float], dict[str, Any]]


def urllib_transport(url: str, body: bytes, headers: dict[str, str],
                     timeout: float) -> dict[str, Any]:
    """真实传输。用标准库 ``urllib``——本项目**零第三方 HTTP 依赖**，
    不该为了发一个请求引入新包（与 ``okx_data.py`` 同一条约定）。"""
    req = urllib.request.Request(url, data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


class TransportError(RuntimeError):
    """传输层失败。把 HTTP 状态码与响应体一起带上——只报"失败了"
    会让"404 模型名写错"和"网络抖动"长得一样。"""


def _default_transport(url: str, body: bytes, headers: dict[str, str],
                       timeout: float) -> dict[str, Any]:
    try:
        return urllib_transport(url, body, headers, timeout)
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read()[:400].decode("utf-8", "ignore")
        except Exception:  # noqa: BLE001
            pass
        raise TransportError(f"HTTP {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise TransportError(f"URLError: {e.reason}") from e
    except json.JSONDecodeError as e:
        raise TransportError(f"响应不是合法 JSON: {e}") from e


# ======================================================================
# 密钥解析
# ======================================================================
#: 仓库外的 secrets 目录。⚠️ 在 ``.gitignore`` 里，不该进仓库。
SECRETS_DIR = Path.home() / ".workbuddy" / "secrets"


def load_api_key(provider: str, env_name: str) -> tuple[str, str]:
    """``(key, 来源)``。来源是 ``env`` / ``file`` / ``""``（没找到）。

    ⚠️ **不打印、不返回路径以外的任何东西**——这个函数的返回值会
    出现在异常信息里，所以异常里只允许出现"变量名"与"文件路径"，
    绝不允许出现 key 本身。
    """
    if env_name:
        v = os.environ.get(env_name, "")
        if v:
            return v, "env"
    p = SECRETS_DIR / f"{provider}.json"
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return "", ""
        v = str(data.get("api_key") or data.get("key") or "")
        if v:
            return v, "file"
    return "", ""


# ======================================================================
# 客户端
# ======================================================================
class LLMClient(Protocol):
    """调用方只依赖这个协议——因此"真机/回放/脚本化"三种实现可互换。"""

    config: LLMConfig

    def chat(self, messages: list[dict[str, Any]],
             **overrides: Any) -> LLMResponse: ...


def _extract_text(data: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """从 OpenAI 兼容响应里取正文。

    ⚠️ **不要静默返回空串**：``choices`` 缺失或为空说明"响应结构不对"
    （可能是报错体被当成了正常响应），返回空串会让它看起来
    "模型答了个空"——那是完全不同的一件事。
    """
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise TransportError(
            f"响应里没有 choices（可能不是 OpenAI 兼容格式）："
            f"{json.dumps(data, ensure_ascii=False)[:300]}"
        )
    msg = (choices[0] or {}).get("message") or {}
    text = msg.get("content")
    if text is None:
        # 有些实现把内容放在 content 为 null 的工具调用里；这里明确报错
        raise TransportError("choices[0].message.content 缺失")
    return str(text), (data.get("usage") or {})


@dataclass(slots=True)
class HTTPClient:
    """真实通道。构造期检查 key —— **没有 key 直接抛，不静默降级**
    （理由见模块文档第 2 条）。"""

    config: LLMConfig
    transport: Transport = _default_transport
    api_key: str = ""
    #: key 的来源（``env`` / ``file`` / ``arg``）——记进日志便于排查
    #: "为什么这台机器能跑那台不能"。**不含 key 本身**。
    key_source: str = ""

    def __post_init__(self) -> None:
        if self.api_key:
            self.key_source = "arg"
        elif self.config.api_key_env or self.config.provider:
            got, src = load_api_key(self.config.provider,
                                    self.config.api_key_env)
            if got:
                self.api_key, self.key_source = got, src
        if self.config.api_key_env and not self.api_key:
            raise RuntimeError(
                f"找不到 {self.config.provider} 的密钥。查找顺序：\n"
                f"  1) 环境变量 {self.config.api_key_env}\n"
                f"  2) {SECRETS_DIR / (self.config.provider + '.json')}"
                f"  （字段 api_key）\n"
                f"  ⚠️ 这里**故意抛错**而不是返回 hold —— 一个'没 key 就弃权'的\n"
                f"  实现跑起来看起来完全正常（每条决策都合规），只是永远不交易。\n"
                f"  那会把'系统是坏的'伪装成'Agent 很保守'。\n"
                f"  Git Bash : export {self.config.api_key_env}=你的key\n"
                f"  PowerShell: $env:{self.config.api_key_env} = \"你的key\"\n"
                f"  ⚠️ 更推荐写进上面那个 secrets 文件——命令行上的 key 会进\n"
                f"  shell 历史与进程表，而那个目录在 .gitignore 里。"
            )

    def chat(self, messages: list[dict[str, Any]],
             **overrides: Any) -> LLMResponse:
        cfg = self.config
        model = str(overrides.get("model") or cfg.model)
        temperature = float(overrides.get("temperature", cfg.temperature))
        max_tokens = int(overrides.get("max_tokens", cfg.max_tokens))
        timeout = float(overrides.get("timeout", cfg.timeout))
        retries = int(overrides.get("retries", cfg.retries))

        ph = prompt_hash(messages, model=model, temperature=temperature,
                         max_tokens=max_tokens)
        body = json.dumps({
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        last_err = ""
        t0 = time.time()
        for attempt in range(1, retries + 1):
            a0 = time.time()
            try:
                data = self.transport(cfg.url, body, headers, timeout)
                text, usage = _extract_text(data)
                return LLMResponse(
                    ok=True, text=text, latency_ms=int((time.time() - a0) * 1000),
                    attempts=attempt, usage=usage, prompt_hash=ph,
                )
            except Exception as e:  # noqa: BLE001
                last_err = f"{type(e).__name__}: {str(e)[:300]}"
                if attempt < retries:
                    # 退避重试：1.5s / 3s / 4.5s …（与 probe_llm 一致）
                    time.sleep(1.5 * attempt)

        return LLMResponse(
            ok=False, error=last_err,
            latency_ms=int((time.time() - t0) * 1000),
            attempts=retries, prompt_hash=ph,
        )


@dataclass(slots=True)
class ScriptedClient:
    """脚本化客户端——**测试用**。

    ``responses`` 是"第 N 次调用返回什么"的列表；用完就按
    ``on_exhausted`` 决定怎么办。⚠️ 默认是 ``raise``：
    静默地一直返回最后一条会让"测试只覆盖了前 3 次调用"这种事实消失。
    """

    config: LLMConfig = field(default_factory=LLMConfig)
    responses: list[LLMResponse] = field(default_factory=list)
    on_exhausted: str = "raise"  # raise | last | hold
    calls: list[list[dict[str, Any]]] = field(default_factory=list)

    def chat(self, messages: list[dict[str, Any]],
             **overrides: Any) -> LLMResponse:
        self.calls.append(messages)
        i = len(self.calls) - 1
        cfg = self.config
        ph = prompt_hash(
            messages,
            model=str(overrides.get("model") or cfg.model),
            temperature=float(overrides.get("temperature", cfg.temperature)),
            max_tokens=int(overrides.get("max_tokens", cfg.max_tokens)),
        )
        if i < len(self.responses):
            return replace(self.responses[i], prompt_hash=ph)
        if self.on_exhausted == "last" and self.responses:
            return replace(self.responses[-1], prompt_hash=ph)
        if self.on_exhausted == "hold":
            return LLMResponse(ok=True, text='{"action":"hold","reason":"脚本已用尽"}',
                               prompt_hash=ph)
        raise RuntimeError(
            f"脚本化客户端已被调用 {len(self.calls)} 次，"
            f"但只准备了 {len(self.responses)} 条响应"
        )

    @classmethod
    def from_json_texts(cls, texts: list[str], **kw: Any) -> "ScriptedClient":
        """从一串 JSON 文本方便地造一个。"""
        return cls(responses=[LLMResponse(ok=True, text=t) for t in texts], **kw)


@dataclass(slots=True)
class ReplayClient:
    """回放客户端——从 JSONL 里按 ``prompt_hash`` 取响应。

    ⚠️ **同一 prompt 可以有多条记录**（多次采样时，prompt 完全相同、
    响应不同）。所以索引是 ``prompt_hash → 响应列表``，
    按调用次序依次发出：第 1 次调用取第 1 条，第 2 次取第 2 条……

    ⭐ 这一条是**踩过坑之后改的**：第一版用 ``dict[hash] = 记录``，
    后写的记录把先写的**覆盖**掉了。后果是——3 次采样只留下最后 1 条，
    回放时 3 次都返回那一条，于是"2 hold / 1 buy"变成了"3 buy"，
    **多数派方向整个反过来**。而覆盖率显示 100%、没有任何报错。
    ⇒ 「看起来成功」与「结果正确」是两件事；这种静默错误
    只有**逐条比对**才发现得了（这也是它当初被漏掉的原因）。

    ⚠️ **记录条数不够时必须报错，不能返回空**。回放的价值在于
    "同输入 ⇒ 同输出"；如果输入变了却静默给个空响应，
    回放会"成功地"跑出一堆弃权，而那是假绿。
    """

    records_path: Path
    config: LLMConfig = field(default_factory=LLMConfig)
    #: 命中/未命中统计（回放覆盖率——没记下的调用是回放的盲区）
    hits: int = 0
    misses: int = 0
    strict: bool = True
    #: ⚠️ 索引必须**声明成字段**：dataclass 是 ``slots=True``，
    #: 在 ``__post_init__`` 里临时 `self._by_hash = ...` 会直接
    #: ``AttributeError``（没有 ``__dict__`` 可挂）。
    #: 这正是本项目在 ``Candle`` 上踩过的同一个坑，换个地方又踩了一次。
    _by_hash: dict[str, list[dict[str, Any]]] = field(
        default_factory=dict, init=False, repr=False,
    )
    #: 每个 prompt 已经发出了几条（决定"这一轮该发第几条"）
    _served: dict[str, int] = field(default_factory=dict, init=False,
                                    repr=False)
    #: 记录文件里同一个 prompt 出现的最大次数（用于诊断"采样数不匹配"）
    _max_per_prompt: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.records_path = Path(self.records_path)
        if self.records_path.exists():
            with open(self.records_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    s = line.strip()
                    if not s:
                        continue
                    rec = json.loads(s)
                    h = rec.get("prompt_hash")
                    if h:
                        # ⚠️ 追加而不是覆盖——见类文档
                        self._by_hash.setdefault(str(h), []).append(rec)
        for v in self._by_hash.values():
            self._max_per_prompt = max(self._max_per_prompt, len(v))

    @property
    def n_prompts(self) -> int:
        return len(self._by_hash)

    def chat(self, messages: list[dict[str, Any]],
             **overrides: Any) -> LLMResponse:
        cfg = self.config
        model = str(overrides.get("model") or cfg.model)
        temperature = float(overrides.get("temperature", cfg.temperature))
        max_tokens = int(overrides.get("max_tokens", cfg.max_tokens))
        ph = prompt_hash(messages, model=model, temperature=temperature,
                         max_tokens=max_tokens)
        bucket = self._by_hash.get(ph)
        idx = self._served.get(ph, 0)
        if bucket is None or idx >= len(bucket):
            self.misses += 1
            if self.strict:
                have = 0 if bucket is None else len(bucket)
                raise KeyError(
                    f"回放记录里没有这一条（prompt_hash={ph}，"
                    f"要第 {idx + 1} 条，只记了 {have} 条）。"
                    f"（共 {self.n_prompts} 个不同 prompt，"
                    f"文件 {self.records_path}）"
                    f"—— prompt/模型/温度变过，或采样数超过了录制时的数量。"
                )
            return LLMResponse(ok=False, error="replay_miss",
                               prompt_hash=ph, attempts=1)
        self._served[ph] = idx + 1
        self.hits += 1
        rec = bucket[idx]
        return LLMResponse(
            ok=bool(rec.get("ok", True)),
            text=str(rec.get("text") or ""),
            error=str(rec.get("error") or ""),
            latency_ms=int(rec.get("latency_ms") or 0),
            attempts=int(rec.get("attempts") or 1),
            usage=dict(rec.get("usage") or {}),
            prompt_hash=ph,
        )

    @property
    def coverage(self) -> float:
        n = self.hits + self.misses
        return 1.0 if n == 0 else self.hits / n


# ======================================================================
# 录制：把真实调用记下来，供回放用
# ======================================================================
class Recorder:
    """包住一个真实客户端，把每次调用追加写进 JSONL。

    ⚠️ **记录里不放密钥**：只放 ``prompt_hash`` / messages / 响应正文与
    配置描述。密钥在那个 Bearer 头里，不进这里。
    """

    def __init__(self, inner: LLMClient, path: str | Path) -> None:
        self.inner = inner
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.n = 0

    @property
    def config(self) -> LLMConfig:
        return self.inner.config

    def chat(self, messages: list[dict[str, Any]],
             **overrides: Any) -> LLMResponse:
        resp = self.inner.chat(messages, **overrides)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "prompt_hash": resp.prompt_hash,
                "messages": messages,
                "model": getattr(self.inner.config, "model", ""),
                "temperature": getattr(self.inner.config, "temperature", None),
                "ok": resp.ok,
                "text": resp.text,
                "error": resp.error,
                "latency_ms": resp.latency_ms,
                "attempts": resp.attempts,
                "usage": resp.usage,
            }, ensure_ascii=False) + "\n")
        self.n += 1
        return resp


def make_client(config: LLMConfig) -> HTTPClient:
    """按配置造一个真实客户端。"""
    return HTTPClient(config=config)


__all__ = [
    "PROVIDERS",
    "LLMConfig",
    "LLMResponse",
    "LLMClient",
    "HTTPClient",
    "ScriptedClient",
    "ReplayClient",
    "Recorder",
    "Transport",
    "TransportError",
    "prompt_hash",
    "make_client",
    "urllib_transport",
]
