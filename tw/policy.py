"""规则基线（A4）—— **读同一份可见状态，但不问 LLM**。

为什么必须有这一组
------------------
设计方案 §6 里最关键的一条是「**LLM 必须显著优于 ``noop``**」。
但只跟 ``noop`` 比还不够，因为那会漏掉一个更麻烦的问题：

> 如果 Agent 赢了，赢的是**模型**，还是**可见状态里的信息**？

``visible_state`` 里有最近 12 根收盘价——那是一份**有信息量的**输入。
任何一个照着它写十条的规则都可能赚钱。所以"赢了 noop"只证明
"这个状态有信息"，**不证明模型有用**。

⇒ 因此本模块提供**同信息基線**：它们吃**完全一样**的 ``visible_state``、
走**完全一样**的风控与执行路径，唯一差别是"决策由谁做"。
只有当 LLM 明显赢过这些规则，才谈得上"模型有贡献"。

本项目的口径
------------
- **``AlwaysHold``（noop）的 PnL 必须恰好是 0。**
  这是整套账本/执行层的**自检**：如果它不为 0，说明记账有漏
  （漏记手续费、漏记滑点、或者根本没接上成交）。
  与 ``strategies/noop`` 在 ABM 侧的验证是同一条思路。
- 规则策略**不调 LLM、不需要 key、完全确定性**，
  所以它们的数字可以**重跑得到逐位相同**的结果。

⚠️ 一个容易写错的地方
----------------------
规则策略返回的 dict **必须用 ``tw/parse.py`` 的规范键**
（``action`` / ``sz`` / ``ordType`` / ``px`` / ``tp`` / ``sl`` /
``confidence`` / ``reason``）。
用别的键名会让风控走上"缺字段"分支——而那条分支**不会报错**，
只会让所有基线都变成"从不交易"，于是"基线没赚钱"这个结论
**看起来像基线的特性，其实是接口错**。
:func:`validate_policy_output` 就是防这个的。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Protocol

from .parse import CANONICAL_KEYS

#: 策略输出里**允许**出现的键（含内部下划线字段）。
_ALLOWED_KEYS = set(CANONICAL_KEYS) | {"_vote", "_policy"}


def validate_policy_output(parsed: dict[str, Any]) -> None:
    """守住"策略输出必须用规范键"这条约定。

    见模块文档末尾的说明：接口写错**不会报错**，只会让基线静默变成
    "从不交易"，而那会被误读成基线的特性。
    """
    unknown = set(parsed) - _ALLOWED_KEYS
    if unknown:
        raise ValueError(
            f"策略输出了非规范键 {sorted(unknown)}；"
            f"允许的有 {sorted(_ALLOWED_KEYS)}。"
            f"（用别的键名会让风控走'缺字段'分支且不报错，"
            f"基线会静默变成'从不交易'）"
        )
    if str(parsed.get("action") or "").lower() not in ("buy", "sell", "hold"):
        raise ValueError(f"策略输出的 action 非法：{parsed.get('action')!r}")


class Policy(Protocol):
    """决策来源。与 ``LLMClient`` 平行的另一个抽象。"""

    name: str

    def decide(self, visible: dict[str, Any], *,
               max_size: float = 0.0) -> dict[str, Any]: ...


def _hold(reason: str) -> dict[str, Any]:
    return {"action": "hold", "sz": 0.0, "px": None, "tp": None, "sl": None,
            "ordType": "market", "confidence": 0.0, "reason": reason}


def _recent(visible: dict[str, Any]) -> list[float]:
    return [float(x) for x in (visible.get("recent_closes") or [])]


# ======================================================================
# noop —— 最重要的那条基准
# ======================================================================
@dataclass(slots=True)
class AlwaysHold:
    """什么都不做。

    ⭐ **它的 PnL 必须恰好是 0**——这是账本的自检。
    如果它不为 0，问题一定出在记账或执行上（手续费/滑点/仓位），
    而不是策略。**先修账本，再谈策略。**
    """

    name: str = "noop"

    def decide(self, visible: dict[str, Any], *,
               max_size: float = 0.0) -> dict[str, Any]:
        out = _hold("基线：始终弃权（noop）")
        out["_policy"] = self.name
        return out


# ======================================================================
# 同信息规则基线
# ======================================================================
@dataclass(slots=True)
class Momentum:
    """看最近 ``lookback`` 根的净变化，顺势下单。

    最朴素的趋势跟随。它用的信息（收盘价序列）**与 LLM 完全一样**，
    所以它是"模型到底有没有用"的**主要对照**。
    """

    lookback: int = 12
    #: 触发阈值（相对变化），低于它就算"看不清"→ 弃权。
    #: ⚠️ 这个阈值必须 > 0：若为 0，任何微小波动都会触发交易，
    #: 于是基线变成"每根都交易"，换手率爆炸——那不是"趋势跟随"，
    #: 那是"高频交手续费"。
    threshold: float = 0.002
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"momentum{self.lookback}"
        if self.threshold <= 0:
            raise ValueError("threshold 必须 > 0（否则每根都会交易）")

    def decide(self, visible: dict[str, Any], *,
               max_size: float = 0.0) -> dict[str, Any]:
        rc = _recent(visible)[-int(self.lookback):]
        if len(rc) < 3 or max_size <= 0:
            out = _hold(f"{self.name}：样本不足或无可用量")
            out["_policy"] = self.name
            return out
        ret = (rc[-1] - rc[0]) / max(rc[0], 1e-12)
        side = "buy" if ret > 0 else "sell"
        if abs(ret) < self.threshold:
            out = _hold(f"{self.name}：净变化 {ret:+.3%} 未达阈值 "
                        f"{self.threshold:.2%}，弃权")
            out["_policy"] = self.name
            return out
        out = {
            "action": side, "sz": float(max_size), "px": None,
            "tp": None, "sl": None, "ordType": "market",
            "confidence": min(1.0, abs(ret) / max(self.threshold * 5, 1e-12)),
            "reason": f"{self.name}：{self.lookback} 根净变化 {ret:+.3%}，顺势{('做多' if side == 'buy' else '做空')}",
            "_policy": self.name,
        }
        return out


@dataclass(slots=True)
class MeanRevert:
    """与 :class:`Momentum` 相反的赌注。

    两个方向相反的基线**同时报**是有意的：如果只有 momentum 赢，
    读者会以为"趋势跟随有效"；把反向的也放上来，才能看出
    **到底哪一边真的有信息**（或者两边都赢=有别的泄漏）。
    """

    lookback: int = 12
    threshold: float = 0.002
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"meanrevert{self.lookback}"
        if self.threshold <= 0:
            raise ValueError("threshold 必须 > 0")

    def decide(self, visible: dict[str, Any], *,
               max_size: float = 0.0) -> dict[str, Any]:
        rc = _recent(visible)[-int(self.lookback):]
        if len(rc) < 3 or max_size <= 0:
            out = _hold(f"{self.name}：样本不足或无可用量")
            out["_policy"] = self.name
            return out
        ret = (rc[-1] - rc[0]) / max(rc[0], 1e-12)
        side = "sell" if ret > 0 else "buy"     # 反向
        if abs(ret) < self.threshold:
            out = _hold(f"{self.name}：净变化 {ret:+.3%} 未达阈值，弃权")
            out["_policy"] = self.name
            return out
        out = {
            "action": side, "sz": float(max_size), "px": None,
            "tp": None, "sl": None, "ordType": "market",
            "confidence": min(1.0, abs(ret) / max(self.threshold * 5, 1e-12)),
            "reason": f"{self.name}：{self.lookback} 根净变化 {ret:+.3%}，赌反转",
            "_policy": self.name,
        }
        return out


@dataclass(slots=True)
class RandomTaker:
    """按固定概率随机开仓。**种子固定 ⇒ 完全可复现。**

    ⚠️ 复现性靠的是"每次决策的随机数由 (seed, tick) 决定"，
    **不是**靠一个全局推进的随机流。若用后者，"少跑一根"
    会让后面所有决策全部错位——那不是回放，那是另一个实验。
    （与项目里「三条随机流必须分开」是同一条道理。）
    """

    p_trade: float = 0.3
    seed: int = 0
    name: str = "random_taker"

    def __post_init__(self) -> None:
        if not (0.0 <= self.p_trade <= 1.0):
            raise ValueError(f"p_trade 必须在 [0,1]，收到 {self.p_trade}")

    def decide(self, visible: dict[str, Any], *,
               max_size: float = 0.0) -> dict[str, Any]:
        # 用"可见状态的内容"派生种子：这样同一份输入永远给同一个决策，
        # 而不同 tick（不同可见状态）仍然不同。
        key = f"{self.seed}|{visible.get('mid')}|{len(visible.get('recent_closes') or [])}"
        rng = random.Random(key)
        if rng.random() > self.p_trade or max_size <= 0:
            out = _hold(f"{self.name}：本次不交易（p_trade={self.p_trade}）")
            out["_policy"] = self.name
            return out
        side = "buy" if rng.random() < 0.5 else "sell"
        out = {
            "action": side, "sz": float(max_size), "px": None,
            "tp": None, "sl": None, "ordType": "market",
            "confidence": 0.5,
            "reason": f"{self.name}：随机{('做多' if side == 'buy' else '做空')}",
            "_policy": self.name,
        }
        return out


# ======================================================================
# 注册表
# ======================================================================
def make_policy(name: str, **kw: Any) -> Policy:
    """按名字造一个基线。给 CLI / 实验脚本用。

    ⚠️ 关键字参数会**按该策略真正声明的字段过滤**。
    这样调用方可以传一个统一的 kw 包（如 ``seed=7, threshold=0.002``），
    而 ``AlwaysHold`` 这种没有 ``seed`` 的策略不会被炸掉。
    **丢掉的键不报错**——这是本函数唯一的"宽松"之处，理由是它只做
    参数分发，不做语义判断；真正的校验在各策略的 ``__post_init__`` 里。
    """
    table: dict[str, Any] = {
        "noop": AlwaysHold,
        "momentum": Momentum,
        "meanrevert": MeanRevert,
        "random_taker": RandomTaker,
    }
    cls = table.get(name)
    if cls is None:
        raise KeyError(f"未知基线 {name!r}；可选 {sorted(table)}")
    from dataclasses import fields as _fields

    allowed = {f.name for f in _fields(cls)}
    return cls(**{k: v for k, v in kw.items() if k in allowed})


def list_policies() -> list[str]:
    return ["noop", "momentum", "meanrevert", "random_taker"]


__all__ = [
    "Policy",
    "AlwaysHold",
    "Momentum",
    "MeanRevert",
    "RandomTaker",
    "make_policy",
    "list_policies",
    "validate_policy_output",
]
