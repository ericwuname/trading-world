"""自生成数据源——把 ABM 引擎的仿真结果写进行情库。

为什么这件事值得单独做
----------------------
用户要求「也可以下载历史数据，还有自己生成数据，这样子 agent 可以利用
不同来源的数据进行自我测试，真实数据测试」。

**本项目的优势恰好在这里：引擎已经在那儿了。**
价值不在于「造更多数据」，而在于造**已知 ground truth 的对照**：

======================  ==============================  ============================
生成方式                  能验证什么                       为什么真实数据做不到
======================  ==============================  ============================
改 scenarios 参数         策略在极端行情下的行为            真实数据里很少出现 liquidation
注入已知冲击              Agent 能否识别并正确反应          真实市场的事件归因本身有内生性问题
**同种子重跑**           Agent 决策的可复现性             真实数据只有一条路径，无法重放
调主体构成比例            策略对市场微结构的敏感性          无法控制真实市场的主体构成
======================  ==============================  ============================

⭐ **最关键的一条**：自生成数据能让「**Agent 的收益是本事还是行情**」
有一个**答案已知的对照组**——因为可以造出「基本面随机游走、完全没有
可预测性」的市场，此时任何正收益都只能是运气。
**这类对照在真实数据上永远做不了。**

与「三条随机流必须分开」的关系
------------------------------
生成的数据**必须带种子与场景参数入账**（``fetches.endpoint`` 里记），
否则无法重造。这与项目那条「缓存键必须覆盖把配置翻译成行为的那段代码」
是同一条纪律的延伸：**没记参数的数据等于没数据**。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from .marketdb import (
    SOURCE_SYNTHETIC,
    Candle,
    FetchRecord,
    MarketStore,
)

#: 合成源里 bar 的毫秒长度。只支持这几种——是**刻意**的：
#: 合成数据的价值在于「构造已知」，不是为了覆盖所有周期。
BAR_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1H": 3_600_000,
    "4H": 14_400_000,
    "1D": 86_400_000,
}


@dataclass(slots=True)
class GenerateSpec:
    """一次生成的完整参数。**这个对象就是「怎么重造」的答案。**"""

    inst_id: str
    bar: str = "1H"
    n_bars: int = 2000
    seed: int = 0
    regime: str = "normal"
    #: 起点时间（毫秒）。默认取「现在」往前推 —— 但**建议显式给**，
    #: 否则同一份数据每次生成的时间轴都不同，跨来源比对就失去意义。
    start_ts: int = 0
    start_price: float = 60_000.0
    #: 年化波动率的粗略目标（按 bar 缩放）
    annual_vol: float = 0.65

    def key(self) -> str:
        """确定性标识。写进 ``fetches.endpoint``，用于重造与去重。"""
        return (
            f"synthetic:{self.regime}:seed={self.seed}:n={self.n_bars}:"
            f"bar={self.bar}:vol={self.annual_vol:.3f}:p0={self.start_price:g}"
        )


# --------------------------------------------------------------------------
# 三种生成器
# --------------------------------------------------------------------------


def _bars_per_year(bar: str) -> float:
    ms = BAR_MS.get(bar)
    if ms is None:
        raise ValueError(f"不支持的 bar：{bar!r}（可选 {sorted(BAR_MS)}）")
    return 365.0 * 24 * 3600 * 1000 / ms


def gen_random_walk(spec: GenerateSpec) -> np.ndarray:
    """**纯随机游走**——「零可预测性」的基准。

    ⭐ 这是整套自生成数据里**最有价值的一条**：它是一份
    「**任何策略都不可能有优势**」的市场。若某策略在这份数据上
    跑出正收益，那只能是运气或数据泄漏，**不可能是本事**。

    真实数据永远给不了这个保证——真实市场里你无法排除「那个时期确实
    有可预测性」。所以这条基准是**真实数据不可替代**的。
    """
    rng = np.random.default_rng(spec.seed)
    n = int(spec.n_bars)
    per_bar = spec.annual_vol / np.sqrt(_bars_per_year(spec.bar))
    steps = rng.normal(0.0, per_bar, size=n)
    # 漂移严格为 0：加任何漂移都会引入方向性，破坏「无优势」这个性质
    return spec.start_price * np.exp(np.cumsum(steps))


def gen_trending(spec: GenerateSpec) -> np.ndarray:
    """**带确定性趋势**——「有优势可抓」的对照。

    与 :func:`gen_random_walk` 配对使用：如果策略在随机游走上赚不到、
    在这份上赚得到，那至少说明它**真的在看价格**，而不是在乱下单。
    这一对构成了「Agent 有没有在做事」的最小判据。
    """
    rng = np.random.default_rng(spec.seed)
    n = int(spec.n_bars)
    per_bar = spec.annual_vol / np.sqrt(_bars_per_year(spec.bar))
    # 趋势强度取波动率的 1/4 —— 足以被观察到，又不至于变成送分题
    drift = per_bar * 0.25 * np.sign(spec.seed % 7 - 3 or 1)
    noise = rng.normal(0.0, per_bar, size=n)
    return spec.start_price * np.exp(np.cumsum(noise + drift))


def gen_vol_clustered(spec: GenerateSpec) -> np.ndarray:
    """**波动率聚集**（GARCH 式的简单版）——真实市场最稳定的特征之一。

    用途：检验策略在「平静期突然转剧烈」时的表现。
    纯随机游走没有聚集性，所以它测不出「策略在波动率突变时失效」这类问题。
    """
    rng = np.random.default_rng(spec.seed)
    n = int(spec.n_bars)
    base = spec.annual_vol / np.sqrt(_bars_per_year(spec.bar))

    # 用一个慢变的 AR(1) 过程调制波动率：λ_t 在 0.5~1.5 之间游走
    lam = np.empty(n)
    lam[0] = 1.0
    for i in range(1, n):
        lam[i] = 0.97 * lam[i - 1] + 0.03 * 1.0 + rng.normal(0.0, 0.06)
    lam = np.clip(lam, 0.4, 2.5)

    steps = rng.normal(0.0, base, size=n) * lam
    return spec.start_price * np.exp(np.cumsum(steps))


GENERATORS: dict[str, Callable[[GenerateSpec], np.ndarray]] = {
    "normal": gen_random_walk,      # 零可预测性基准
    "random": gen_random_walk,
    "trend": gen_trending,
    "vol_cluster": gen_vol_clustered,
}


def available_regimes() -> list[str]:
    return sorted(GENERATORS)


# --------------------------------------------------------------------------
# 价格路径 -> K 线
# --------------------------------------------------------------------------


def prices_to_candles(
    prices: np.ndarray,
    spec: GenerateSpec,
    *,
    sub_steps: int = 12,
) -> list[Candle]:
    """把收盘价序列变成 K 线。

    ⚠️ **``high``/``low`` 不能直接用收盘价**，否则 K 线退化成折线
    （``high == low == close``），蜡烛图会画成一根根线，而且
    「窗口内极值」这个 OHLC 的定义就废了。

    做法：在相邻收盘价之间**插值出一条更细的路径**，取其中的极值。
    这是合成数据与真实数据在**形态上**最容易被看出来的差别——
    真实 K 线永远有影线，纯收盘价造的没有。
    """
    n = int(len(prices))
    if n < 2:
        raise ValueError(f"价格序列太短：{n}")

    rng = np.random.default_rng(spec.seed + 977)
    ms = BAR_MS[spec.bar]
    ts0 = spec.start_ts or int(time.time() * 1000) // ms * ms - n * ms

    out: list[Candle] = []
    for i in range(n - 1):
        p0, p1 = float(prices[i]), float(prices[i + 1])
        # 在 p0 -> p1 之间插 sub_steps 个中间点，再加一点噪声
        t = np.linspace(0.0, 1.0, sub_steps + 2)[1:-1]
        mid = p0 + (p1 - p0) * t
        jitter = rng.normal(0.0, abs(p1 - p0) * 0.35 + p0 * 1e-4, size=mid.size)
        path = np.concatenate(([p0], mid + jitter, [p1]))

        hi = float(max(p0, p1, float(path.max())))
        lo = float(min(p0, p1, float(path.min())))
        # 轻微的正负偏差，避免 hi==max(o,c) 这种「无影线」的退化形态
        hi = max(hi, p0, p1) * (1.0 + abs(rng.normal(0.0, 2e-4)))
        lo = min(lo, p0, p1) * (1.0 - abs(rng.normal(0.0, 2e-4)))

        # 成交量：与 |收益| 成正比（真实市场里量价相关）
        ret = abs(p1 - p0) / max(p0, 1e-9)
        vol = float(100.0 * (1.0 + 30.0 * ret) * abs(rng.normal(1.0, 0.4)))

        out.append(Candle(
            ts=ts0 + i * ms, open=p0, high=hi, low=lo, close=p1,
            volume=vol, confirm=1,
        ))
    return out


# --------------------------------------------------------------------------
# 落库
# --------------------------------------------------------------------------


def generate_into_store(
    store: MarketStore,
    spec: GenerateSpec,
    *,
    overwrite: bool = False,
    subsample: int = 1,
) -> dict[str, Any]:
    """生成一段合成行情并落库。返回生成摘要。

    ``subsample``：从**同一条**价格路径上按步长抽 K 线，
    这样 ``1H`` 与 ``4H`` 出自同一个底层过程，跨周期可比
    （若各自独立生成，跨周期分析里的差异就分不清是周期效应还是生成噪声）。
    """
    gen = GENERATORS.get(spec.regime)
    if gen is None:
        raise ValueError(
            f"未知 regime：{spec.regime!r}（可选 {available_regimes()}）"
        )
    if spec.bar not in BAR_MS:
        raise ValueError(f"不支持的 bar：{spec.bar!r}")

    started = int(time.time() * 1000)
    # 若要下采样，就先按更细的 bar 生成足量路径
    if subsample > 1:
        fine = GenerateSpec(**{**spec.__dict__, "n_bars": spec.n_bars * subsample})
        prices = gen(fine)[::subsample]
    else:
        prices = gen(spec)

    candles = prices_to_candles(prices, spec)
    n = store.upsert_candles(
        SOURCE_SYNTHETIC, spec.inst_id, spec.bar, candles, overwrite=overwrite
    )
    store.log_fetch(FetchRecord(
        source=SOURCE_SYNTHETIC, inst_id=spec.inst_id, bar=spec.bar,
        endpoint=spec.key(),
        from_ts=candles[0].ts if candles else 0,
        to_ts=candles[-1].ts if candles else 0,
        n_rows=n, pages=1, started_at=started,
        finished_at=int(time.time() * 1000), ok=True,
    ))
    return {
        "inst_id": spec.inst_id,
        "bar": spec.bar,
        "regime": spec.regime,
        "seed": spec.seed,
        "n_generated": len(candles),
        "n_written": n,
        "key": spec.key(),
    }


# --------------------------------------------------------------------------
# 对照三件套（自生成数据的核心用法）
# --------------------------------------------------------------------------


def default_test_set(
    inst_id: str = "SYNTH-BTC",
    bar: str = "1H",
    *,
    n_bars: int = 2000,
    seed: int = 0,
    start_ts: int = 1_700_000_000_000,
    start_price: float = 60_000.0,
) -> list[GenerateSpec]:
    """「Agent 自我测试」的默认三件套。

    ==================  ==========================================
    规格                 它回答的问题
    ==================  ==========================================
    ``normal``          **零可预测性**——策略不该赚钱。赚了就是运气/泄漏
    ``trend``           **有方向可抓**——策略该有反应。没反应说明它没在看价
    ``vol_cluster``     **波动率突变**——策略的风控会不会失效
    ==================  ==========================================

    三个用同一个 ``seed``、同一 ``start_ts``、同一 ``start_price``，
    所以它们是**同一初始条件下的三种世界**，差别只在生成机制。

    ⚠️ **``inst_id`` 会带 regime 后缀**（``SYNTH-BTC__normal`` 等）。
    必须这样：三份世界的时间轴完全相同，若共用 ``inst_id``，
    主键 ``(source, inst_id, bar, ts)`` 会撞，
    ``INSERT OR IGNORE`` 会**静默丢掉后两份**——只留第一份，
    而且**不报错**（实测踩到：三个 regime 只剩一个）。
    用后缀区分后，三份可以共存并随时对比。
    """
    common = dict(
        bar=bar, n_bars=n_bars, seed=seed,
        start_ts=start_ts, start_price=start_price,
    )
    return [
        GenerateSpec(inst_id=f"{inst_id}__normal", regime="normal", **common),
        GenerateSpec(inst_id=f"{inst_id}__trend", regime="trend", **common),
        GenerateSpec(inst_id=f"{inst_id}__vol_cluster", regime="vol_cluster", **common),
    ]


def build_test_set(
    store: MarketStore, specs: Sequence[GenerateSpec] | None = None
) -> list[dict[str, Any]]:
    """把对照三件套写进库。返回三份生成摘要。"""
    specs = list(specs) if specs is not None else default_test_set()
    return [generate_into_store(store, s) for s in specs]


__all__ = [
    "BAR_MS",
    "GENERATORS",
    "GenerateSpec",
    "available_regimes",
    "build_test_set",
    "default_test_set",
    "gen_random_walk",
    "gen_trending",
    "gen_vol_clustered",
    "generate_into_store",
    "prices_to_candles",
]
