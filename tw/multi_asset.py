"""多资产相关市场（二期阶段9）。

目标
----
把单资产市场扩到 N 个资产，**并且让它们真的相关**——
从而第一次可以检验"配对交易 / 统计套利"这类跨资产策略。

⚠️ 三条必须写清楚的事
====================

**① "相关"必须来自共享的公共因子，不能靠事后掺数据。**
   做法：每个资产的对数收益 = ``beta_i · 公共冲击 + 特异性冲击_i``。
   公共冲击每 tick **只抽一次**，所有资产共用。
   如果各自独立随机游走，样本相关系数会围绕 0 波动（±1/√n），
   "配对交易"就变成了在噪声上做套利——那个实验没有意义，
   而且**看起来照样能出结果**。

**② 每个资产必须是**独立的**市场对象（自己的订单簿、自己的主体）。**
   共享一个订单簿等于把不同资产混进同一个价格序列。
   共享主体池则会让"一个资产的冲击"直接改变另一个资产的持仓，
   那不是跨资产传导，那是记账串了。

**③ 冲击传导的机制只有一条：相关性。**
   资产 A 被砸 → A 的价格相对公共因子偏低 → 持有两资产的套利者/配对交易者
   看到偏离 → 在 A 买、在 B 卖（或反之）→ B 被拉动。
   所以传导强度**必须随相关系数上升**，这既是直觉也是可证伪的判据（E9.4）。
   如果传导强度与相关系数无关，说明传导其实来自别的渠道
   （例如主体池共享、或基本面被错误地同步推进）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import Population, SimConfig
from .market import Market


@dataclass(slots=True)
class AssetSpec:
    """单个资产的静态参数。"""

    name: str
    #: 对公共因子的暴露（beta）
    beta: float = 1.0
    #: 特异性波动（每 tick 的对数收益标准差）
    idio_vol: float = 2e-4
    #: 初始价格
    initial_price: float = 60_000.0


@dataclass(slots=True)
class MultiAssetConfig:
    """多资产配置。"""

    assets: list[AssetSpec] = field(default_factory=lambda: [
        AssetSpec("AAA", beta=1.0, idio_vol=2e-4, initial_price=60_000.0),
        AssetSpec("BBB", beta=1.0, idio_vol=2e-4, initial_price=3_000.0),
    ])
    #: 公共因子的每 tick 对数收益标准差
    common_vol: float = 3e-4
    #: 向长期锚点回归的强度（防止共同因子无限漂移）
    common_pull: float = 1e-5
    seed: int = 20260917

    def validate(self) -> None:
        if len(self.assets) < 2:
            raise ValueError("多资产至少需要 2 个资产")
        for a in self.assets:
            if a.idio_vol < 0:
                raise ValueError(f"{a.name}: idio_vol 不能为负")
            if a.initial_price <= 0:
                raise ValueError(f"{a.name}: initial_price 必须为正")
        if self.common_vol < 0:
            raise ValueError("common_vol 不能为负")

    def implied_correlation(self, i: int, j: int) -> float:
        """两资产**对数收益**的理论相关系数（由参数直接算出）。

        对 ``r_i = β_i·c + ε_i``（c、ε 独立、零均值）：
        ``corr = β_iβ_jσ_c² / √((β_i²σ_c²+σ_i²)(β_j²σ_c²+σ_j²))``。

        存在的意义是给 E9.1 一个**解析靶子**：如果实测相关系数
        与这个值差得远，说明生成器或市场实现有问题——
        而不是"市场自己就是这样"。没有解析靶子的话，
        一个完全错的相关结构也能被解释成"涌现出来的"。
        """
        self.validate()
        a, b = self.assets[i], self.assets[j]
        sc2 = self.common_vol ** 2
        cov = a.beta * b.beta * sc2
        va = (a.beta ** 2) * sc2 + a.idio_vol ** 2
        vb = (b.beta ** 2) * sc2 + b.idio_vol ** 2
        if va <= 0 or vb <= 0:
            return float("nan")
        return float(cov / np.sqrt(va * vb))

    def correlation_matrix(self) -> np.ndarray:
        n = len(self.assets)
        m = np.eye(n, dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                c = self.implied_correlation(i, j)
                m[i, j] = m[j, i] = c
        return m


class CorrelatedFundamentalGenerator:
    """生成 N 个相关的基本面价值序列。

    ``value_i(t) = value_i(t−1) · (1 + β_i·c(t) + ε_i(t))``

    公共冲击 ``c(t)`` 每 tick 只抽一次（见模块文档 ①）。
    """

    def __init__(self, cfg: MultiAssetConfig) -> None:
        cfg.validate()
        self.cfg = cfg
        self.n_assets = len(cfg.assets)
        self.rng = np.random.default_rng([cfg.seed, 0x9A55E7])
        self.betas = np.array([a.beta for a in cfg.assets], dtype=float)
        self.idio_vols = np.array([a.idio_vol for a in cfg.assets], dtype=float)
        self.values = np.array([a.initial_price for a in cfg.assets], dtype=float)
        self.anchors = self.values.copy()
        self.n_steps = 0
        #: 归因：累计抽了多少次公共冲击（应当 == 步数，不能多）
        self.n_common_draws = 0

    def step(self) -> np.ndarray:
        """推进一 tick，返回各资产的新基本面价值。"""
        common = float(self.rng.normal(0.0, self.cfg.common_vol))
        self.n_common_draws += 1
        idio = self.rng.normal(0.0, 1.0, self.n_assets) * self.idio_vols
        ret = self.betas * common + idio
        # 向初始锚点弱回归，防止共同因子无限漂移（与单资产的 `fu_v_pull` 同源）
        pull = self.cfg.common_pull
        self.values = self.values * (1.0 + ret) * (1.0 - pull) \
            + self.anchors * pull
        self.values = np.maximum(self.values, 1e-6)
        self.n_steps += 1
        return self.values.copy()


class AssetMarket(Market):
    """单个资产的市场：基本面由**外部生成器**驱动。"""

    def __init__(self, *args, spec: AssetSpec | None = None, **kw) -> None:
        self.spec = spec
        #: 本 tick 的外部锚点。None = 走市场自己的随机游走（单资产行为）
        self.external_anchor: float | None = None
        super().__init__(*args, **kw)

    def _step_fundamental(self) -> None:
        if self.external_anchor is None:
            super()._step_fundamental()
            return
        # 直接采用外部锚点。⚠️ 这里**不**再抽自己的随机数——
        # 再抽一次会让资产的收益里混进一段与公共因子无关的噪声，
        # 表现为"实测相关系数系统性低于解析值"（E9.1 会抓到这个）。
        self.fundamental = float(self.external_anchor)


class MultiAssetMarket:
    """管理 N 个独立的 ``AssetMarket``，共享一个基本面生成器。

    每个子市场有**自己的**订单簿、主体池与日志（模块文档 ②）。
    """

    def __init__(
        self, cfg: MultiAssetConfig, *,
        populations: list[Population] | None = None,
        n_ticks: int = 12_000,
        sim_overrides: dict | None = None,
    ) -> None:
        cfg.validate()
        self.cfg = cfg
        self.generator = CorrelatedFundamentalGenerator(cfg)
        self.markets: list[AssetMarket] = []
        for i, spec in enumerate(cfg.assets):
            pop = (populations[i] if populations else
                   Population.from_shares(
                       300, {"zero_intel": 0.30, "fundamentalist": 0.40,
                             "chartist": 0.30}))
            kw = dict(sim_overrides or {})
            sim = SimConfig(
                seed=cfg.seed + i, n_ticks=n_ticks, population=pop,
                initial_price=spec.initial_price, fu_v_anchor=spec.initial_price,
                **kw,
            )
            m = AssetMarket(sim, spec=spec)
            m.fundamental = spec.initial_price
            self.markets.append(m)
        self.tick = 0

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.markets)

    def mid(self, i: int) -> float | None:
        v = self.markets[i].current_mid()
        return float(v) if v and v > 0 else None

    def mids(self) -> list[float | None]:
        return [self.mid(i) for i in range(len(self.markets))]

    def step(self) -> None:
        anchors = self.generator.step()
        for i, m in enumerate(self.markets):
            m.external_anchor = float(anchors[i])
            m.step()
        self.tick += 1

    def run(self, n_ticks: int | None = None) -> None:
        n = n_ticks if n_ticks is not None else self.markets[0].cfg.n_ticks
        for _ in range(n):
            self.step()

    # ------------------------------------------------------------------
    def log_returns(self, i: int, lo: int = 0, hi: int | None = None) -> np.ndarray:
        m = self.markets[i]
        hi = m.tick if hi is None else min(hi, m.tick)
        arr = np.asarray(m.log.mid[lo:hi], dtype=float)
        arr = arr[np.isfinite(arr) & (arr > 0)]
        if arr.size < 3:
            return np.zeros(0, dtype=float)
        return np.diff(np.log(arr))

    def realized_correlation(self, i: int, j: int, lo: int = 0,
                             hi: int | None = None) -> float:
        """两资产已实现对数收益的相关系数。"""
        a = self.log_returns(i, lo, hi)
        b = self.log_returns(j, lo, hi)
        n = min(a.size, b.size)
        if n < 10:
            return float("nan")
        a, b = a[-n:], b[-n:]
        if a.std() <= 0 or b.std() <= 0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    def realized_correlation_matrix(self, lo: int = 0,
                                    hi: int | None = None) -> np.ndarray:
        n = len(self.markets)
        m = np.eye(n, dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                c = self.realized_correlation(i, j, lo, hi)
                m[i, j] = m[j, i] = c
        return m

    def health_check(self) -> tuple[bool, list[str]]:
        ok, probs = True, []
        for m in self.markets:
            o, p = m.health_check()
            if not o:
                ok = False
                probs.extend(f"{m.spec.name if m.spec else '?'}: {x}" for x in p[:3])
        return ok, probs
