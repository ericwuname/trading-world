"""LLM 连通性自检 —— 在把它接进交易回路**之前**先确认它能用。

为什么单独做成一个脚本
----------------------
把 LLM 接进交易系统之前，必须先把"它到底能不能稳定给出结构化决策"
这件事**单独验证掉**。否则后面调试时，你分不清是
「模型不行」、「网络不行」、「解析写错了」还是「策略真的不行」。

⚠️ **密钥只从环境变量读，绝不写进代码或配置**：

    # Git Bash
    export AGNES_KEY=你的key
    python scripts/probe_llm.py

    # PowerShell
    $env:AGNES_KEY = "你的key"
    python scripts/probe_llm.py

支持的通道（按 `--provider` 选，默认 agnes）：

| provider | base_url | 模型 |
|---|---|---|
| `agnes` | api.agnes-ai.cn/v1 | agnes-2.5-flash |
| `openai` | 走 OPENAI_BASE_URL 或官方 | gpt-4o-mini |
| `deepseek` / `zhipu` | 同理 | — |

只用**标准库**（`urllib`）——这个项目的一个好处是**零第三方 HTTP 依赖**，
自检脚本不该为了发一个请求就引入新包。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import banner  # noqa: E402

#: 各通道的默认端点与模型
PROVIDERS: dict[str, dict] = {
    "agnes": {"url": "https://api.agnes-ai.cn/v1/chat/completions",
              "model": "agnes-2.5-flash", "env": "AGNES_KEY"},
    "openai": {"url": "https://api.openai.com/v1/chat/completions",
               "model": "gpt-4o-mini", "env": "OPENAI_API_KEY"},
    "deepseek": {"url": "https://api.deepseek.com/v1/chat/completions",
                 "model": "deepseek-chat", "env": "DEEPSEEK_API_KEY"},
    "zhipu": {"url": "https://open.bigmodel.cn/api/paas/v4/chat/completions",
              "model": "glm-4-flash", "env": "ZHIPU_API_KEY"},
}

#: 用真实交易上下文做探针 —— 空泛的"说句话"测不出真问题
PROMPT = """你是一个加密货币永续合约交易员。下面是当前市场快照（模拟市场，BTC-USDT-SWAP）：

- tick: 3120
- 中间价: 102.44  买卖价差: 8.1bp
- 最近 10 tick 中间价（旧→新）: 102.10, 102.18, 102.22, 102.31, 102.28, 102.35, 102.40, 102.37, 102.42, 102.44
- 你的当前持仓: 多头 0.30 张，入场均价 102.10
- 账户权益: 1,000,000 USDT   保证金率: 42%   可用杠杆上限: 5x
- 资金费率: +0.0001%（多头支付）

请决定下一步动作。只输出一个 JSON 对象，不要任何其他文字：
{"action": "buy|sell|hold|close", "size": 数字, "order_type": "limit|market",
 "limit_price": 数字或null, "take_profit": 数字或null, "stop_loss": 数字或null,
 "confidence": 0到1的小数, "reason": "一句话中文理由"}"""

REQUIRED_KEYS = ("action", "size", "confidence", "reason")


def parse_json_loose(text: str) -> dict | None:
    """从模型输出里抠出 JSON。

    ⚠️ 实测 agnes-2.5-flash 会**直接给纯 JSON**，但别的模型常包一层
    ```json 代码块或加一句"好的，这是分析："。所以解析必须**容错**——
    而且解析失败本身要作为**记录**留下来，不能静默丢弃
    （记录里保留 `llm_raw` 就是为了这个）。
    """
    s = (text or "").strip()
    if "```" in s:
        parts = s.split("```")
        if len(parts) >= 2:
            s = parts[1]
            if s.lower().startswith("json"):
                s = s[4:]
            s = s.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # 退一步：找第一个 { 到最后一个 }
    i, j = s.find("{"), s.rfind("}")
    if 0 <= i < j:
        try:
            return json.loads(s[i:j + 1])
        except json.JSONDecodeError:
            return None
    return None


def call_once(url: str, model: str, key: str, *, timeout: float = 60.0,
              retries: int = 2) -> dict:
    """带重试的调用。返回 ``{ok, latency, text, usage, error, attempts}``。"""
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0.2,
    }).encode()
    last_err = ""
    for attempt in range(1, retries + 1):
        t0 = time.time()
        try:
            req = urllib.request.Request(url, data=body, headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            })
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read())
            dt = time.time() - t0
            text = ((data.get("choices") or [{}])[0]
                    .get("message", {}).get("content", ""))
            return {"ok": True, "latency": dt, "text": text,
                    "usage": data.get("usage") or {}, "error": "",
                    "attempts": attempt}
        except urllib.error.HTTPError as e:
            detail = e.read()[:200].decode("utf-8", "ignore")
            last_err = f"HTTP {e.code}: {detail}"
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {str(e)[:200]}"
        if attempt < retries:
            time.sleep(1.5 * attempt)
    return {"ok": False, "latency": None, "text": "", "usage": {},
            "error": last_err, "attempts": retries}


def main() -> int:
    ap = argparse.ArgumentParser(description="LLM 连通性与结构化输出自检")
    ap.add_argument("--provider", default="agnes", choices=sorted(PROVIDERS))
    ap.add_argument("--model", default=None, help="覆盖默认模型名")
    ap.add_argument("--runs", type=int, default=3,
                    help="重复次数（默认 3）—— 一次成功说明不了稳定性")
    args = ap.parse_args()

    banner("LLM 连通性自检")
    cfg = PROVIDERS[args.provider]
    key = os.environ.get(cfg["env"], "")
    if not key:
        print(f"  ❌ 环境变量 {cfg['env']} 没有设置。\n"
              f"     Git Bash : export {cfg['env']}=你的key\n"
              f"     PowerShell: $env:{cfg['env']} = \"你的key\"")
        return 2

    model = args.model or cfg["model"]
    print(f"  通道 {args.provider}   模型 {model}   重复 {args.runs} 次")
    print(f"  （密钥从 {cfg['env']} 读，长度 {len(key)}，不打印内容）\n")

    oks, lats, bad = 0, [], 0
    for i in range(1, args.runs + 1):
        r = call_once(cfg["url"], model, key)
        if not r["ok"]:
            print(f"  [{i}] ❌ {r['error']}")
            bad += 1
            continue
        oks += 1
        lats.append(r["latency"])
        obj = parse_json_loose(r["text"])
        if obj is None:
            print(f"  [{i}] ⚠️ 调用成功但 JSON 解析失败（延迟 {r['latency']:.2f}s）"
                  f"——这要计入记录里的解析失败率")
            print(f"        原始输出前 120 字：{r['text'][:120]}")
            bad += 1
            continue
        missing = [k for k in REQUIRED_KEYS if k not in obj]
        flag = "✅" if not missing else f"⚠️ 缺字段 {missing}"
        print(f"  [{i}] {flag} 延迟 {r['latency']:.2f}s  tokens={r['usage'].get('total_tokens')}")
        print(f"        action={obj.get('action')} size={obj.get('size')} "
              f"tp={obj.get('take_profit')} sl={obj.get('stop_loss')} "
              f"conf={obj.get('confidence')}")
        print(f"        理由：{str(obj.get('reason'))[:80]}")

    print()
    if lats:
        print(f"  延迟：中位 {sorted(lats)[len(lats)//2]:.2f}s，"
              f"最慢 {max(lats):.2f}s")
    print(f"  成功 {oks}/{args.runs}（其中解析失败 {bad} 次）")
    if oks == args.runs and bad == 0:
        print("  ✅ 全部通过 —— 可以进入下一步（设计决策回路）")
        return 0
    print("  ⚠️ 有失败项。先查清是网络、配额还是提示词问题，再往下做。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
