"""多段独立运行 + 同段配对检验（A6）—— **换掉判不出来的那个度量**。

为什么要换（这是实测，不是理论）
--------------------------------
A5 把样本从 120 根加到 **400 根（3.3 倍）**，结果
**8 个对照判定仍全是「依然无法判定」**，`required_n` 在 5,000~55,000。

原因出在度量上。用「**每根平均收益率**」时：
- 大部分根是"不交易"（收益恰为 0）⇒ 那些根**不提供任何信息**；
- 带信息的那少数根**方差极大**（一笔成交能吃下一天的盈亏）；
⇒ 均值的置信区间宽到几乎包含一切。**加样本也追不上。**

换成什么
--------
```
把 T 根切成 K 段（每段 L 根，K×L ≈ T）
每段**独立起跑**（同初始权益、同 prompt、账户从零开始）
  ⇒ 得到 K 个**独立**观测，而不是 T 个强自相关观测
每个观测 = 「这一段的总收益」
同一段上 LLM 与每条基线各跑一次 ⇒ **配对**
⇒ 对 K 个**配对差值**做检验
```

**效能为什么高一个数量级**：K 个独立观测的有效样本量就是 K；
而"每根"那 T 个观测的有效样本量 ≈ ``T / (1 + 2Σρ)``——
在强自相关下（本项目 ACF 实测过）远小于 T。

⚠️ 三条必须自己先说清楚的事
--------------------------
1. **"独立"不能假设，要实测**。段之间在**时间上连续**，行情本身有自相关
   ⇒ 段收益仍可能相关。本模块**报段间相关**，不声称独立。
   真相关时有效样本量小于 K，检验会偏乐观——报告里必须写。
2. **段长 L 不能拍脑袋**。太短则策略来不及表现、被开局噪声淹没；
   太长则 K 太少。⇒ 用 :func:`power_analysis` **先估**再定。
3. **成本不额外增加**：``调用数 = 总根数 × 采样数``，与连续跑完全一样。
   只是把"一次 400 根"改成"8 次 50 根"。基线是规则策略，免费。

⭐ 还有一条本模块**自带的自检**（:func:`sensitivity_check`）：
拿一个**已知差异**（同一个策略、成本倍数不同）去跑，
看新度量**能不能判出来**。如果一个度量连人为制造的差异都判不出来，
那它在真实比较里也判不出真差异。
⇒ **先用合成差异标定灵敏度，再上真实比较。**
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

from .agent import TradingAgent, visible_from_series
from .agent_run import RunResult, run_agent_session
from .account import MarginAccount, MarginConfig
from .simexec import ExecConfig


# ======================================================================
# 切段
# ======================================================================
def segment_ranges(n_bars: int, seg_len: int, *,
                   min_history: int = 12) -> list[tuple[int, int]]:
    """把可用区间切成若干**决策区间** ``(start, end)``（闭区间）。

    ``min_history``：每段的可见状态要用到开始前的若干根收盘价。
    第 0 段从 ``min_history`` 开始，保证每段都有等量的历史可用——
    ⚠️ 若允许第 0 段从 0 开始，它的 ``recent_closes`` 会比别的段短，
    于是各段**输入不等价**，配对就不成立。
    """
    if seg_len < 1:
        raise ValueError(f"seg_len 必须 >= 1，收到 {seg_len}")
    lo = max(0, int(min_history))
    hi = int(n_bars) - 1
    if hi - lo + 1 < seg_len:
        return []
    out: list[tuple[int, int]] = []
    s = lo
    while s + seg_len - 1 <= hi:
        out.append((s, s + seg_len - 1))
        s += seg_len
    return out


# ======================================================================
# 结果容器
# ======================================================================
@dataclass(slots=True)
class SegmentedRuns:
    """多段配对运行的结果。

    ``net[seg][config]`` = 该段该配置的**净收益率**（相对该段初始权益）。
    """

    seg_len: int
    ranges: list[tuple[int, int]] = field(default_factory=list)
    #: 每段每配置的净收益率
    net: list[dict[str, float]] = field(default_factory=list)
    #: 每段每配置的 RunResult（想深挖时用；可能很大）
    runs: list[dict[str, RunResult]] = field(default_factory=list)

    @property
    def n_segs(self) -> int:
        return len(self.ranges)

    def configs(self) -> list[str]:
        return sorted(self.net[0]) if self.net else []

    def series(self, cfg: str) -> list[float]:
        return [seg[cfg] for seg in self.net]

    def diffs(self, a: str, b: str) -> list[float]:
        """配对差值 ``a - b``（同段相减）。"""
        return [seg[a] - seg[b] for seg in self.net]

    def to_dict(self) -> dict[str, Any]:
        return {
            "seg_len": self.seg_len,
            "n_segs": self.n_segs,
            "ranges": [list(r) for r in self.ranges],
            "net": self.net,
            "configs": self.configs(),
        }


# ======================================================================
# 跑
# ======================================================================
#: 造一个"全新的 agent"的工厂。每段都要新建——**账户与状态必须从零开始**，
#: 否则段之间就不再独立（前一段的持仓会带进下一段）。
AgentFactory = Callable[[], TradingAgent]


def run_paired_segments(
    series: Any,
    factories: dict[str, AgentFactory],
    *,
    seg_len: int,
    min_history: int = 12,
    initial_cash: float = 100_000.0,
    exec_config: ExecConfig | None = None,
    lever: float = 3.0,
    max_segs: int = 0,
    keep_runs: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> SegmentedRuns:
    """在每一段上把**所有配置各跑一次**，得到配对的观测。

    ⚠️ **每个配置每段都要新建 agent 与账户**（见 :data:`AgentFactory`）：
    复用同一个账户会让前一段的持仓带进下一段 ⇒ 段不独立，
    而"段独立"正是这个度量效能高的**唯一来源**。
    """
    cfg_exec = exec_config or ExecConfig()
    ranges = segment_ranges(len(getattr(series, "close", [])), seg_len,
                            min_history=min_history)
    if max_segs:
        ranges = ranges[: int(max_segs)]
    if not ranges:
        raise ValueError(
            f"切不出任何段：总长 {len(getattr(series, 'close', []))}，"
            f"seg_len={seg_len}，min_history={min_history}"
        )

    out = SegmentedRuns(seg_len=seg_len)
    n_cfg = len(factories)
    for k, (s, e) in enumerate(ranges):
        per_net: dict[str, float] = {}
        per_run: dict[str, RunResult] = {}
        for name, mk in factories.items():
            acc = MarginAccount(cash=float(initial_cash), cfg=MarginConfig())
            res = run_agent_session(
                mk(), series, account=acc, start=s, end=e, name=name,
                run_id=f"seg{k}-{name}", exec_config=cfg_exec, lever=lever,
            )
            base = res.initial_equity or 1.0
            per_net[name] = (res.final_equity - base) / base
            if keep_runs:
                per_run[name] = res
            if progress is not None:
                progress(k * n_cfg + len(per_net), len(ranges) * n_cfg)
        out.ranges.append((s, e))
        out.net.append(per_net)
        if keep_runs:
            out.runs.append(per_run)
    return out


# ======================================================================
# 配对检验
# ======================================================================
def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _sd(xs: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def segment_correlation(net: list[dict[str, float]], cfg: str) -> float:
    """段与段之间（相邻）的收益相关。

    ⚠️ **这个数字必须报**：段在时间上连续，行情有自相关 ⇒
    "独立"是**不能假设**的。相关为正时有效样本量小于 K，
    检验会偏乐观。本模块不声称段独立，只报出这个数让人自己判断。
    """
    xs = [seg[cfg] for seg in net]
    if len(xs) < 3:
        return float("nan")
    a, b = xs[:-1], xs[1:]
    ma, mb = _mean(a), _mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    return num / (da * db) if da > 0 and db > 0 else float("nan")


def _t_crit(df: int, alpha: float = 0.05) -> float:
    """双侧 t 临界值（小表，够用即可；df>30 用 1.96 近似）。

    ⚠️ 不引 scipy：本项目「零第三方依赖」的约定，而且这里只需要
    df ≤ 30 的精确值与更大 df 的近似。
    """
    table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
        13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
        19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064,
        25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
    }
    if df <= 0:
        return float("inf")
    return table.get(df, 1.96 if alpha == 0.05 else 2.58)


def paired_verdict(diffs: list[float], *,
                   name_a: str = "A", name_b: str = "B",
                   alpha: float = 0.05) -> dict[str, Any]:
    """对**配对差值**做 t 检验。返回点估计、置信区间与三选一判定。

    判定只有三种，与项目其余部分一致：
    ``a 显著更高`` / ``b 显著更高`` / **依然无法判定**。

    ⚠️ 区间包含 0 ⇒ 判"无法判定"，**不判"没有差别"**。
    这两句话在本项目里是严格区分的（见「十条测量纪律」）。
    """
    n = len(diffs)
    if n < 2:
        return {"n": n, "mean_diff": float("nan"), "ci": (float("nan"),) * 2,
                "verdict": "依然无法判定", "reason": "段数不足（<2）",
                "t": float("nan"), "df": 0, "t_crit": float("nan"),
                "se": float("nan"), "sd": float("nan"), "direction": None}
    m = _mean(diffs)
    s = _sd(diffs)
    df = n - 1
    tc = _t_crit(df, alpha)

    # ⚠️⚠️ **σ = 0 必须单独处理**，否则最**强**的证据会被报成"没有证据"。
    #
    # σ=0 表示每一段的差值**一模一样**——一个完美一致的效应。
    # 朴素写法 `se = s/√n if s > 0 else nan` 会给出 se=nan ⇒ 区间 (nan,nan)
    # ⇒ `nan > 0` 与 `nan < 0` 都是 False ⇒ 落到"依然无法判定"。
    # **方向正好相反**：一致性最强的时候说"判不出来"。
    # 这类"把结论弄反"的静默错误正是本项目一路在纠的东西。
    # （成本倍数这类**确定性**效应就是在 σ≈0 的情况下被观测到的。）
    if s == 0.0:
        se = 0.0
        half = 0.0
        lo = hi = m
        t = math.inf if m > 0 else (-math.inf if m < 0 else float("nan"))
    else:
        se = s / math.sqrt(n)
        half = tc * se
        lo, hi = m - half, m + half
        t = m / se if se > 0 else float("nan")

    if lo > 0:
        verdict, direction = f"{name_a} 显著更高", name_a
    elif hi < 0:
        verdict, direction = f"{name_b} 显著更高", name_b
    else:
        verdict, direction = "依然无法判定", None
    note = ("配对差值完全一致（σ=0）⇒ 区间塌成一点" if s == 0.0 else "")
    return {
        "n": n,
        "mean_diff": m,
        "sd": s,
        "se": se,
        "t": t,
        "df": df,
        "t_crit": tc,
        "ci": (lo, hi),
        "verdict": verdict,
        "direction": direction,
        "reason": (f"配对差值均值 {m:+.4%}，95% 区间 "
                   f"[{lo:+.4%}, {hi:+.4%}]（n={n} 段）" + (f"；{note}" if note else "")),
    }


# ======================================================================
# 功效分析（先估再定段长）
# ======================================================================
def power_analysis(effect: float, sd: float, *,
                   alpha: float = 0.05, power: float = 0.80) -> dict[str, Any]:
    """要多少段才够（配对 t 检验的粗略公式）。

    用正态近似：``n ≈ (z_{α/2} + z_{1-β})² · σ² / δ²``。
    ``z`` 取 1.96 / 0.84（80% 功效）。

    ⚠️ **这是粗略估计**：它是"给定效应量与标准差"的必要段数，
    而**效应量本身往往是未知的**——那就必须先跑一小批（如 8 段）
    用观测到的差值标准差去回填。**不能拿它当结论。**
    """
    if sd is None or sd != sd or sd <= 0 or effect == 0:
        return {"required_segs": None,
                "note": "需要非零效应与非零标准差；先用小批量估 sd"}
    z = (1.96 if alpha == 0.05 else 2.58) + 0.8416
    n = (z ** 2) * (sd ** 2) / (effect ** 2)
    return {
        "required_segs": int(math.ceil(n)),
        "effect": float(effect),
        "sd": float(sd),
        "note": "粗略估计；效应量未知时应先用小批量估 sd 再回填",
    }


def min_detectable_effect(sd: float, n_segs: int, *,
                          alpha: float = 0.05, power: float = 0.80
                          ) -> float:
    """给定段数与差值标准差，**能分辨的最小效应**。

    反过来问更有用：我已经跑了 K 段，那这个实验**最多能看出多大的差异**？
    """
    if not (sd == sd) or sd <= 0 or n_segs < 2:
        return float("nan")
    z = (1.96 if alpha == 0.05 else 2.58) + 0.8416
    return z * sd / math.sqrt(n_segs)


# ======================================================================
# ⭐ 灵敏度自检：拿"已知差异"试这个度量
# ======================================================================
def sensitivity_check(
    series: Any,
    factory: AgentFactory,
    *,
    seg_len: int,
    min_history: int = 12,
    cost_low: float = 1.0,
    cost_high: float = 3.0,
    max_segs: int = 0,
    slippage_bps: float = 1.0,
    initial_cash: float = 100_000.0,
    lever: float = 3.0,
) -> dict[str, Any]:
    """**先用人为制造的差异标定这个度量。**

    做法：同一个策略、同一段行情，只改**成本倍数**
    （``cost_low`` vs ``cost_high``）——这是一个**已知方向、大致已知量级**
    的差异（成本越高、净收益越低）。

    ⇒ 如果新度量连这个都判不出来，那它在真实比较里也判不出真差异。
    **先过这一关，再上真实验。**

    ⚠️ 成本倍数只影响**执行**（手续费与滑点），
    所以对**规则基线**是干净的对照；对 LLM 它会让决策路径发散
    （见 ``tw.eval_agent`` 的说明），因此本自检**建议只用规则基线**。
    """
    low = run_paired_segments(
        series, {"x": factory}, seg_len=seg_len, min_history=min_history,
        initial_cash=initial_cash, lever=lever, max_segs=max_segs,
        exec_config=ExecConfig(slippage_bps=slippage_bps,
                               cost_multiplier=cost_low),
    )
    high = run_paired_segments(
        series, {"x": factory}, seg_len=seg_len, min_history=min_history,
        initial_cash=initial_cash, lever=lever, max_segs=max_segs,
        exec_config=ExecConfig(slippage_bps=slippage_bps,
                               cost_multiplier=cost_high),
    )
    diffs = [a["x"] - b["x"] for a, b in zip(low.net, high.net)
             if a["x"] == a["x"] and b["x"] == b["x"]]
    v = paired_verdict(diffs, name_a=f"成本×{cost_low:g}",
                       name_b=f"成本×{cost_high:g}")
    mde = min_detectable_effect(_sd(diffs), len(diffs))
    passed = v["verdict"].startswith(f"成本×{cost_low:g}")
    return {
        "known_effect": f"成本 ×{cost_low:g} vs ×{cost_high:g}（前者应更好）",
        "n_segs": len(diffs),
        "verdict": v,
        "min_detectable_effect": mde,
        "passed": bool(passed),
        "note": ("✅ 度量能判出已知差异 ⇒ 可以上真实验"
                 if passed else
                 "⚠️ 连已知差异都判不出来 ⇒ 段数/段长不够，先调参"),
    }


__all__ = [
    "segment_ranges",
    "SegmentedRuns",
    "AgentFactory",
    "run_paired_segments",
    "paired_verdict",
    "segment_correlation",
    "power_analysis",
    "min_detectable_effect",
    "sensitivity_check",
]
