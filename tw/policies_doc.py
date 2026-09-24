"""两份文档策略的**系统内实现**——接进 trading-world 的同一条管线。

    《个人交易决策系统方案》§4 ⇒ :class:`GridPolicyDoc`（左尾）
    《右尾策略完全手册》§9/§11 ⇒ :class:`RightTailPolicyDoc`（右尾）

⭐ 为什么接进来而不是用独立脚本：A14/A15 的独立回测已经验过**文档的数字**；
这一步要回答的是另一个问题——**同样的数据、同样的成本模型、同样的账户
与风控管线，两个"形状"各自长什么样**。这是文档 §9（阴阳两面）与 §8
（成本是唯一确定性杀手）的系统级检验。

⚠️ 系统语义与独立回测的差异（都要如实报告）：
  - 每根 K 线收盘决策一次 ⇒ 网格一根 bar 最多成交一格（独立回测可多格）；
  - 账户是 **1x 逐仓保证金**（不是现货）⇒ 卖出会把仓位减到 0 但不会做空
    （本实现用 inventory 守卫保证）；
  - 成本走 ExecConfig：限价 maker 2bp、市价 taker 5bp+滑点 1bp。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


def _hold(reason: str) -> dict[str, Any]:
    return {"action": "hold", "sz": 0.0, "px": None, "tp": None, "sl": None,
            "ordType": "market", "confidence": 0.0, "reason": reason}


def grid_levels(center: float, k: float, n: int) -> list[float]:
    """等比格位（文档 §4.1）：i=0..n，共 n+1 个。"""
    g = ((1 + k) / (1 - k)) ** (1.0 / n) - 1
    lo = center * (1 - k)
    return [lo * ((1 + k) / (1 - k)) ** (i / n) for i in range(n + 1)]


# ======================================================================
# 左尾：现货网格（文档 §4 规格）
# ======================================================================
@dataclass(slots=True)
class GridPolicyDoc:
    """按《个人交易决策系统方案》§4 实现的网格。

    买区 = 低于中心的格位；卖区 = 高于中心的格位。
    初始半仓（§4.2）＝先买入 max_size 的一半。
    每 168 根重设中心（§4.2 的设置——A14 实测它有害，这里**照文档**实现，
    让系统级回测来给结论）；21 天（504 根 1H）无成交离场（§3 禁区的例外条款）。
    """

    k: float = 0.25
    n: int = 10
    recenter_every: int = 168
    exit_no_fill_bars: int = 21 * 24          # 1H bar ⇒ 21 天
    name: str = "grid_doc"

    _bar: int = 0
    _center: float = 0.0
    _levels: list[float] = field(default_factory=list)
    _prev_mid: float = 0.0
    _prev_inv: float = 0.0
    _unit: float = 0.0
    _max_qty: float = 0.0
    _since_fill: int = 0
    _armed: bool = False

    def decide(self, visible: dict[str, Any], *,
               max_size: float = 0.0) -> dict[str, Any]:
        self._bar += 1
        mid = float(visible.get("mid") or 0.0)
        inv = float(visible.get("inventory") or 0.0)
        if not math.isfinite(mid) or mid <= 0:
            return _hold("mid 非法")

        # ---- 首根：建梯子 + 初始半仓 --------------------------------
        if not self._armed:
            self._armed = True
            self._center = mid
            self._levels = grid_levels(mid, self.k, self.n)
            self._unit = max_size / self.n          # 一格的币量（建仓时锁定）
            self._max_qty = max_size
            self._prev_mid = mid
            self._prev_inv = 0.0
            out = _hold("")
            out.update({"action": "buy", "sz": float(self._unit * (self.n // 2)),
                        "px": None, "ordType": "market",
                        "confidence": 1.0,
                        "reason": f"初始半仓 {self.n // 2} 格（§4.2）"})
            return out

        out = _hold("")

        # ---- 卖出：上穿卖区格位（先看，避免同一根又买又卖）----------
        crossed_up = any(self._prev_mid < L <= mid for L in self._levels[self.n // 2 + 1:])
        crossed_dn = any(mid <= L < self._prev_mid for L in self._levels[:self.n // 2])
        acted = False
        if crossed_up and inv > self._unit * 0.5:
            out = _hold("")
            out.update({"action": "sell", "sz": float(min(self._unit, inv)),
                        "px": None, "ordType": "market", "confidence": 1.0,
                        "reason": "网格：上穿卖区格位，卖出一格"})
            acted = True
        elif crossed_dn and inv < self._max_qty - self._unit * 0.5:
            if self._since_fill == 0 or True:
                out = _hold("")
                out.update({"action": "buy", "sz": float(self._unit),
                            "px": None, "ordType": "market", "confidence": 1.0,
                            "reason": "网格：下穿买区格位，买入一格"})
                acted = True

        # ---- 21 天无成交离场（§3）-----------------------------------
        if inv != self._prev_inv:
            self._since_fill = 0
        else:
            self._since_fill += 1
            if (self.exit_no_fill_bars and
                    self._since_fill >= self.exit_no_fill_bars and inv > 0):
                out = _hold("")
                out.update({"action": "sell", "sz": float(inv),
                            "px": None, "ordType": "market", "confidence": 1.0,
                            "reason": f"{self.exit_no_fill_bars} 根无成交，清仓离场（§3）"})
                acted = True
                self._since_fill = 0

        # ---- 每 168 根重设中心（§4.2，A14 实测有害——照文档实现）----
        if self.recenter_every and self._bar % self.recenter_every == 0:
            self._center = mid
            self._levels = grid_levels(mid, self.k, self.n)

        self._prev_mid = mid
        self._prev_inv = inv
        out["_policy"] = self.name
        return out


# ======================================================================
# 右尾：突破入场 + 紧止损 + 远目标（《右尾手册》§9/§11 的市场实现）
# ======================================================================
@dataclass(slots=True)
class RightTailPolicyDoc:
    """右尾结构的市场实现：**大量小止损 + 罕见大盈利**。

    - 入场：mid 突破前 lookback 根的高/低（对称，≈公平硬币）；
    - 止损 sl = entry×(1−stop_pct)，止盈 tp = entry×(1+target_pct)；
      目标/风险 = target_pct/stop_pct（默认 10:1——这就是"右尾形状"）；
    - 有仓时不动（等 tp/sl 触发），离场后冷却 cooldown 根再武装。
    """

    lookback: int = 20
    stop_pct: float = 0.02
    target_pct: float = 0.20
    risk_frac: float = 0.02                   # 每笔止损损失占权益的比例
    cooldown: int = 12
    name: str = "righttail_doc"

    _bar: int = 0
    _prev_inv: float = 0.0
    #: ⭐ 初值 = 已武装：开局的空仓不是「离场后冷却」。
    #    我第一版默认 0 ⇒ 开局 12 根全在冷却，策略永远晚 12 根入场。
    _flat_since: int = 10_000

    def decide(self, visible: dict[str, Any], *,
               max_size: float = 0.0) -> dict[str, Any]:
        self._bar += 1
        mid = float(visible.get("mid") or 0.0)
        inv = float(visible.get("inventory") or 0.0)
        equity = float(visible.get("equity") or 0.0)
        rc = [float(x) for x in (visible.get("recent_closes") or [])]
        if not math.isfinite(mid) or mid <= 0 or len(rc) < self.lookback + 1:
            return _hold("样本不足")

        if inv != self._prev_inv:
            self._flat_since = 0 if abs(inv) < 1e-12 else self._flat_since
        if abs(inv) > 1e-12:
            self._prev_inv = inv
            out = _hold("右尾：持仓中，等 tp/sl 触发")
            out["_policy"] = self.name
            return out
        self._flat_since += 1
        self._prev_inv = inv

        if self._flat_since <= self.cooldown:
            out = _hold(f"右尾：离场后冷却第 {self._flat_since} 根")
            out["_policy"] = self.name
            return out

        window = rc[-(self.lookback + 1):-1]      # 不含当前根（防前视）
        hh, ll = max(window), min(window)
        # 止损损失的币量：sz × mid × stop_pct = equity × risk_frac
        sz = equity * self.risk_frac / (mid * self.stop_pct)
        sz = min(sz, max_size)
        if sz <= 0:
            return _hold("右尾：可用量为 0")
        if mid > hh:                              # 向上突破 ⇒ 做多
            out = _hold("")
            out.update({"action": "buy", "sz": float(sz), "px": None,
                        "ordType": "market", "confidence": 0.6,
                        "sl": float(mid * (1 - self.stop_pct)),
                        "tp": float(mid * (1 + self.target_pct)),
                        "reason": f"右尾：突破 {self.lookback} 根高点 {hh:.2f}，"
                                  f"风险 {self.risk_frac:.0%}/格，目标 10:1"})
        elif mid < ll:                            # 向下突破 ⇒ 做空
            out = _hold("")
            out.update({"action": "sell", "sz": float(sz), "px": None,
                        "ordType": "market", "confidence": 0.6,
                        "sl": float(mid * (1 + self.stop_pct)),
                        "tp": float(mid * (1 - self.target_pct)),
                        "reason": f"右尾：跌破 {self.lookback} 根低点 {ll:.2f}"})
        else:
            out = _hold("右尾：未突破")
        out["_policy"] = self.name
        return out


def list_doc_policies() -> list[str]:
    return ["grid_doc", "righttail_doc"]


def make_doc_policy(name: str, **kw: Any) -> Any:
    if name == "grid_doc":
        return GridPolicyDoc(**kw)
    if name == "righttail_doc":
        return RightTailPolicyDoc(**kw)
    raise KeyError(f"未知文档策略：{name}（可用 {list_doc_policies()}）")
