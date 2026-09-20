"""HTTP API 的业务实现。

端点一览
--------
    GET  /api/meta                 场景 / 策略 / 默认值 / 限额 / 真实数据集
    POST /api/run                  建作业（market | strategy | lab）
    GET  /api/job/<id>             作业状态 + 进度 + 汇总结果
    GET  /api/job/<id>/series      时序数据（下采样，用于画图）
    POST /api/job/<id>/cancel      请求取消
    GET  /api/jobs                 本次会话的历史作业列表（对照用）
    GET  /api/real/<symbol>        真实行情统计特征 + 价格序列
    GET  /api/export/<id>.csv      导出该作业的指标为 CSV

设计取舍
--------
**把统计都放在后端算。** 前端只负责画。
理由：本项目的指标口径（对数收益率、ACF、Hill 尾部指数、markout 对齐方向、
PnL 三分解）都已经在 `tw/analyzer.py` / `tw/eval.py` 里实现并测过。
在前端用 JS 重写一遍，等于凭空造出第二套口径——
两份实现迟早会不一致，而且没人会发现。**口径只能有一份。**
"""

from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tw import realdata  # noqa: E402
from tw.analyzer import analyze, log_returns  # noqa: E402
from tw.eval import evaluate, mid_series, strategy_paths  # noqa: E402
from tw.market import Market  # noqa: E402
from tw.scenarios import SCENARIOS, get_scenario, list_scenarios  # noqa: E402
from tw.strategy import Strategy, make_strategy  # noqa: E402

from . import __version__  # noqa: E402
from .jobs import Job, JobManager, step_until  # noqa: E402

# ----------------------------------------------------------------------
# 限额：不设限额的话，界面上一个手滑的输入就能把内存吃光
# （tick 数 × 主体数 直接决定日志数组大小与运行时长）
# ----------------------------------------------------------------------
MAX_TICKS = 60_000
MAX_WARMUP = 20_000
MAX_AGENTS = 1_500
MAX_LAB_RUNS = 300
#: 批量对照的**总规模**上限（单位 = tick·次）。这才是真正的成本驱动量：
#: 一次批量对照的耗时 ≈ Σ(预热 + 观测)，与"组合数"是两回事——
#: 20 次 × 5 万 tick 比 280 次 × 1000 tick 贵得多。
#: 原来只卡组合数（300），而 7 策略 × 5 场景 × 8 种子 = 280，
#: 那个上限**永远触发不了**，等于没有护栏（被测试抓出来的）。
MAX_LAB_TICKS = 3_000_000
#: 时序图最多回多少点。8000 tick 全画在 1200px 宽的画布上本来也看不出差别，
#: 而下采样能把 JSON 体积压掉一个数量级（前端解析才是真正的瓶颈）。
SERIES_POINTS = 1200

REAL_KEYS = ["BTCUSDT_1h", "ETHUSDT_1h", "SOLUSDT_1h"]

#: 允许在界面里跑自定义代码。设为 0 可以彻底关掉这个能力（更安全，但少了探索性）。
ALLOW_CODE = os.environ.get("TW_GUI_ALLOW_CODE", "1") != "0"


class ApiError(Exception):
    """业务错误：带 HTTP 状态码，前端直接展示 message。"""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status = status


# ----------------------------------------------------------------------
# 小工具
# ----------------------------------------------------------------------
def _jsonable(o):
    """把 numpy 类型转成 JSON 能序列化的东西，顺手把 inf/nan 变成 None。

    NaN 必须显式转成 null：`json.dumps` 默认会写出 `NaN` 字面量，
    那不是合法 JSON，前端的 `JSON.parse` 会直接抛错，
    表现为"界面白屏但服务端日志一切正常"——非常难查。
    """
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return None if not math.isfinite(v) else v
    if isinstance(o, np.ndarray):
        return [_jsonable(x) for x in o.tolist()]
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, float):
        return None if not math.isfinite(o) else o
    raise TypeError(f"无法序列化 {type(o)}")


def _vec(a, nd: int = 4) -> list:
    """浮点序列 → 可序列化列表（保留 nd 位，压体积）。"""
    arr = np.asarray(a, dtype=np.float64)
    arr = np.round(arr, nd)
    return [None if not np.isfinite(v) else float(v) for v in arr]


def _decimate(a: np.ndarray, points: int = SERIES_POINTS):
    arr = np.asarray(a, dtype=np.float64)
    n = arr.size
    if n <= points:
        return arr, 1
    stride = int(math.ceil(n / points))
    return arr[::stride], stride


def _num(spec: dict, key: str, default: float, lo: float, hi: float) -> float:
    try:
        v = float(spec.get(key, default))
    except (TypeError, ValueError):
        raise ApiError(f"参数 {key} 不是数字：{spec.get(key)!r}") from None
    if not math.isfinite(v):
        raise ApiError(f"参数 {key} 必须是有限数")
    return max(lo, min(hi, v))


def _int(spec: dict, key: str, default: int, lo: int, hi: int) -> int:
    return int(_num(spec, key, default, lo, hi))


def _scenario_from_spec(spec: dict):
    name = spec.get("scenario", "normal")
    try:
        sc = get_scenario(name)
    except KeyError:
        # 不要用 str(KeyError)：它会带上多余的引号，界面上看着像 JSON 出错
        raise ApiError(
            f"未知场景 {name!r}；可用：{sorted(SCENARIOS)}"
        ) from None
    # ⚠️ 必须**复制**再改：`SCENARIOS` 是全局单例，直接改会把覆盖值
    # 永久写回注册表，之后每次跑都带着上一次的参数。
    # 注意 Scenario 是 `@dataclass(slots=True)`，**没有** `__dict__`，
    # 不能写成 `type(sc)(**sc.__dict__)`——那样会直接 AttributeError。
    return replace(
        sc,
        n_agents=_int(spec, "n_agents", sc.n_agents, 10, MAX_AGENTS),
        warmup=_int(spec, "warmup", sc.warmup, 0, MAX_WARMUP),
        overrides=dict(sc.overrides),
        mix=dict(sc.mix),
    )


def _returns_hist(mid: np.ndarray, bins: int = 61) -> dict | None:
    """对数收益率的直方图 + 同均值方差的正态分布对照。

    为什么要叠一条正态曲线：肥尾是**看出来的**，不是算出来的。
    峰度 10.26 这个数字对大多数人没有直觉，但"中间的柱子比正态矮、
    两边的尾巴翘起来"一眼就懂。
    """
    r = log_returns(mid) * 1e4
    r = r[np.isfinite(r)]
    if r.size < 20:
        return None
    # 用 0.1/99.9 分位裁掉极端值：肥尾会让 min/max 把横轴拉到看不见中间
    lo, hi = np.percentile(r, [0.1, 99.9])
    span = max(abs(float(lo)), abs(float(hi))) * 1.05 or 1.0
    edges = np.linspace(-span, span, bins + 1)
    cnt, _ = np.histogram(r, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2.0
    sd = float(r.std(ddof=1)) or 1.0
    mu = float(r.mean())
    width = float(edges[1] - edges[0])
    normal = (
        np.exp(-((centers - mu) ** 2) / (2 * sd * sd))
        / (sd * math.sqrt(2 * math.pi))
        * r.size
        * width
    )
    return {
        "centers": _vec(centers, 3),
        "count": [int(x) for x in cnt],
        "normal": _vec(normal, 2),
        "clip_bp": round(span, 2),
        "n_outside": int(np.sum((r < lo) | (r > hi))),
        "n": int(r.size),
        "mean_bp": round(mu, 4),
        "sd_bp": round(sd, 4),
    }


def _acf_payload(met) -> dict:  # noqa: ANN001
    return {
        "lags": list(range(len(met.acf_abs))),
        "abs": _vec(np.asarray(met.acf_abs), 4),
        "ret": _vec(np.asarray(met.acf_ret), 4),
        "ci": None if not math.isfinite(met.acf_ci95) else round(float(met.acf_ci95), 4),
        "n_sig_abs": int(getattr(met, "n_sig_lags_abs", 0) or 0),
    }


def _history_payload(m, t0: int) -> tuple[dict, int]:
    """把主日志压成前端能画的时序（全部下采样）。"""
    L = m.log
    n = int(m.tick)
    lo = max(0, int(t0))

    def cut(name: str) -> np.ndarray:
        return np.asarray(getattr(L, name)[lo:n], dtype=np.float64)

    tick = np.arange(lo, n, dtype=np.float64)
    mid = cut("mid")
    # 单边空缺写成 NaN 会让折线断开，前向填充
    bad = ~np.isfinite(mid) | (mid <= 0)
    if bad.any():
        idx = np.where(~bad, np.arange(mid.size), 0)
        np.maximum.accumulate(idx, out=idx)
        mid = mid[idx]

    mid_d, stride = _decimate(mid)
    out = {
        "stride": stride,
        "n_full": int(mid.size),
        "tick": [int(x) for x in tick[::stride]],
        "mid": _vec(mid_d, 2),
        "fundamental": _vec(_decimate(cut("fundamental"))[0], 2),
        "best_bid": _vec(_decimate(cut("best_bid"))[0], 2),
        "best_ask": _vec(_decimate(cut("best_ask"))[0], 2),
    }
    sp = cut("spread")
    with np.errstate(invalid="ignore", divide="ignore"):
        sp_bp = np.where(mid > 0, sp / mid * 1e4, np.nan)
    out["spread_bp"] = _vec(_decimate(sp_bp)[0], 3)
    out["depth_bid"] = _vec(_decimate(cut("bid_depth"))[0], 3)
    out["depth_ask"] = _vec(_decimate(cut("ask_depth"))[0], 3)
    out["n_trades"] = _vec(_decimate(cut("n_trades"))[0], 2)
    out["volume"] = _vec(_decimate(cut("volume"))[0], 3)
    return out, stride


# ----------------------------------------------------------------------
# 元信息
# ----------------------------------------------------------------------
def meta() -> dict:
    from strategies import REGISTRY

    scens = []
    for s in SCENARIOS.values():
        scens.append(
            {
                "name": s.name,
                "intent": s.intent,
                "n_agents": s.n_agents,
                "warmup": s.warmup,
                "n_ticks": s.n_ticks,
                "mix": s.mix,
                "shock_frac": s.shock_frac,
                "shock_at": s.shock_at,
                "overrides": {k: str(v) for k, v in s.overrides.items()},
            }
        )
    strats = []
    for name, (cls, kw) in REGISTRY.items():
        doc = (cls.__doc__ or "").strip()
        strats.append(
            {
                "name": name,
                "cls": cls.__name__,
                "params": {k: v for k, v in kw.items()},
                "doc": doc,
                "kind": getattr(cls, "KIND", ""),
            }
        )
    return {
        "version": __version__,
        "scenarios": scens,
        "strategies": strats,
        "real_symbols": list(realdata.available_builtin()) or REAL_KEYS,
        "defaults": {
            "scenario": "normal",
            "seed": 20260917,
            "ticks": 6000,
            "strategy": "mm_naive",
            "lab_seeds": 3,
            "lab_ticks": 4000,
        },
        "limits": {
            "max_ticks": MAX_TICKS,
            "max_warmup": MAX_WARMUP,
            "max_agents": MAX_AGENTS,
            "max_lab_runs": MAX_LAB_RUNS,
        },
        "allow_code": ALLOW_CODE,
        "real_keys": REAL_KEYS,
    }


def real_payload(symbol: str) -> dict:
    """真实行情：统计特征 + 下采样价格序列（用于和模拟并排看）。"""
    if symbol not in (realdata.available_builtin() or REAL_KEYS):
        raise ApiError(f"未知数据集 {symbol!r}", 404)
    s = realdata.load_builtin(symbol)
    met = analyze(s.close, symbol, vol_window=24)
    close = np.asarray(s.close, dtype=np.float64)
    d, stride = _decimate(close, 1200)
    r = log_returns(close) * 1e4
    return {
        "symbol": symbol,
        "n": int(close.size),
        "stride": stride,
        "close": _vec(d, 2),
        "metrics": met.flat(),
        "hist": _returns_hist(close),
        "acf": _acf_payload(met),
        "sigma_bp": round(float(np.nanstd(r, ddof=1)), 4),
    }


# ----------------------------------------------------------------------
# 策略代码
# ----------------------------------------------------------------------
def compile_strategy(code: str) -> type:
    """把用户粘贴的代码编译成一个 Strategy 子类。

    ⚠️ **这是在本机执行任意 Python 代码。** 没有沙箱——
    进程内的沙箱在 Python 里做不干净（`__builtins__` 到处可达）。
    所以这里只做两件事：界面明确警告 + 服务只绑定回环地址。
    不要运行来源不明的策略代码。

    取"最后一个定义的 Strategy 子类"：用户经常在文件里先写一个辅助类，
    真正的策略写在下面。按定义顺序取最后一个，符合直觉。
    """
    if not ALLOW_CODE:
        raise ApiError("服务端已禁用自定义策略代码（TW_GUI_ALLOW_CODE=0）", 403)
    src = (code or "").strip()
    if not src:
        raise ApiError("策略代码为空")
    if len(src) > 40_000:
        raise ApiError("策略代码过长（上限 40000 字符）")
    ns: dict = {"__name__": "tw_user_strategy"}
    try:
        exec(compile(src, "<你的策略>", "exec"), ns)  # noqa: S102
    except SyntaxError as exc:
        raise ApiError(f"语法错误（第 {exc.lineno} 行）：{exc.msg}") from None
    except Exception as exc:  # noqa: BLE001
        raise ApiError(f"代码执行失败：{type(exc).__name__}: {exc}") from None
    from tw.strategy import Strategy as _S

    found = [
        v for v in ns.values()
        if isinstance(v, type) and issubclass(v, _S) and v is not _S
    ]
    if not found:
        raise ApiError("代码里没有找到 Strategy 的子类（from tw.strategy import Strategy）")
    return found[-1]


def _resolve_strategy(spec: dict) -> tuple[str, type, dict]:
    """从 spec 里解析出 (显示名, 类, 参数)。"""
    if spec.get("strategy") == "custom":
        cls = compile_strategy(spec.get("code", ""))
        params = dict(spec.get("params") or {})
        return f"custom:{cls.__name__}", cls, params
    from strategies import REGISTRY

    name = spec.get("strategy") or "mm_naive"
    if name not in REGISTRY:
        raise ApiError(f"未知策略 {name!r}；可用：{sorted(REGISTRY)}")
    cls, defaults = REGISTRY[name]
    raw = spec.get("params") or {}
    if not isinstance(raw, dict):
        # 直接调 API 的人可能把参数传成 JSON 字符串。给一句人话，
        # 而不是让 `{**defaults, **raw}` 抛一个不知所云的 TypeError。
        raise ApiError(f"参数覆盖必须是 JSON 对象，收到 {type(raw).__name__}")
    params = {**defaults, **raw}
    return name, cls, params


# ----------------------------------------------------------------------
# 作业实现
# ----------------------------------------------------------------------
def run_market(job: Job) -> tuple[dict, dict | None]:
    spec = job.spec
    sc = _scenario_from_spec(spec)
    seed = _int(spec, "seed", 20260917, 0, 2**31 - 1)
    ticks = _int(spec, "ticks", 6000, 50, MAX_TICKS)
    total = sc.warmup + ticks

    t_start = time.time()
    m = Market(sc.config(seed))
    job.say(
        f"场景 {sc.name}｜seed {seed}｜主体 {m.n_agents}｜"
        f"预热 {sc.warmup:,} + 观测 {ticks:,} tick"
    )
    if step_until(job, m.run, sc.warmup, work=total, phase="预热"):
        return None, None
    t0 = m.tick
    if step_until(job, m.run, ticks, offset=sc.warmup, work=total, phase="观测"):
        return None, None
    wall = time.time() - t_start

    ok, problems = m.health_check()
    mid_win = mid_series(m, t0)
    met = analyze(mid_win, "模拟", vol_window=24)
    series, _ = _history_payload(m, t0)

    reals = {k: analyze(realdata.load_builtin(k).close, k, vol_window=24).flat()
             for k in REAL_KEYS}
    sim = met.flat()
    cmp_rows = []
    for key, label, better in (
        ("sigma_bp", "收益率标准差 (bp)", "near"),
        ("excess_kurtosis", "超额峰度", "near"),
        ("acf_abs_lag1", "|r| ACF(1)", "near"),
        ("n_sig_lags_abs", "显著 |r| 滞后数（/20）", "near"),
        ("acf_ret_lag1", "r ACF(1)", "near"),
        ("vr5", "方差比 VR(5)", "near"),
        ("hill_alpha_left", "左尾 Hill α", "near"),
    ):
        row = {"key": key, "label": label, "sim": sim.get(key)}
        for k, v in reals.items():
            row[k] = v.get(key)
        bt = reals.get("BTCUSDT_1h", {}).get(key)
        try:
            row["ratio_btc"] = (
                float(sim[key]) / float(bt) if bt and float(bt) != 0 else None
            )
        except (KeyError, TypeError, ValueError):
            row["ratio_btc"] = None
        cmp_rows.append(row)

    n_trades = int(np.nansum(np.asarray(m.log.n_trades[t0:m.tick], dtype=float)))
    result = {
        "kind": "market",
        "scenario": sc.name,
        "seed": seed,
        "warmup": sc.warmup,
        "ticks": ticks,
        "wall_sec": round(wall, 2),
        "n_agents": m.n_agents,
        "health_ok": bool(ok),
        "health_problems": list(problems[:5]),
        "market": sim,
        "real": reals,
        "compare": cmp_rows,
        "hist": _returns_hist(mid_win),
        "acf": _acf_payload(met),
        "structure": {
            "n_trades": n_trades,
            "trades_per_tick": round(n_trades / max(1, ticks), 2),
            "volume": float(np.nansum(np.asarray(m.log.volume[t0:m.tick], dtype=float))),
            "spread_bp_mean": _safe_mean(
                np.asarray(m.log.spread[t0:m.tick], dtype=float) / mid_win * 1e4
            ),
            "depth_total_mean": _safe_mean(
                np.asarray(
                    m.log.bid_depth[t0:m.tick] + m.log.ask_depth[t0:m.tick],
                    dtype=float,
                )
            ),
            "open_orders_mean": _safe_mean(
                np.asarray(m.log.open_orders[t0:m.tick], dtype=float)
            ),
            "n_agents": m.n_agents,
            "n_ticks_observed": int(mid_win.size),
        },
    }
    job.say(f"完成：{n_trades:,} 笔成交，σ={_fmt(sim.get('sigma_bp'))}bp，用时 {wall:.1f}s")
    return result, series


def _safe_mean(a: np.ndarray) -> float | None:
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    return round(float(a.mean()), 4) if a.size else None


def _fmt(v, nd: int = 2) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    return "—" if not math.isfinite(f) else f"{f:,.{nd}f}"


def run_strategy(job: Job) -> tuple[dict, dict | None]:
    spec = job.spec
    sc = _scenario_from_spec(spec)
    seed = _int(spec, "seed", 20260917, 0, 2**31 - 1)
    ticks = _int(spec, "ticks", 6000, 50, MAX_TICKS)
    total = sc.warmup + ticks
    label, cls, params = _resolve_strategy(spec)

    cash = _num(spec, "cash", 60_000.0 * 300.0, 0.0, 1e12)
    inv = _num(spec, "inventory", 0.0, -1e6, 1e6)

    t_start = time.time()
    m = Market(sc.config(seed))
    if step_until(job, m.run, sc.warmup, work=total, phase="预热"):
        return None, None

    t0 = m.tick
    agent = make_strategy(cls, "strat", seed=seed, cash=cash, inventory=inv, **params)
    m.add_agent(agent)
    job.say(
        f"策略 {label}｜场景 {sc.name}｜seed {seed}｜"
        f"预热 {sc.warmup:,} + 观测 {ticks:,} tick｜参数 {params}"
    )
    if step_until(job, m.run, ticks, offset=sc.warmup, work=total, phase="观测"):
        return None, None
    wall = time.time() - t_start

    ok, problems = m.health_check()
    mid_win = mid_series(m, t0)
    met = analyze(mid_win, "模拟", vol_window=24)
    series, _ = _history_payload(m, t0)

    ev = evaluate(m, "strat", from_tick=t0)
    _, path = strategy_paths(m, "strat", t0)
    eq = np.asarray(path["equity"], dtype=np.float64)
    invp = np.asarray(path["inventory"], dtype=np.float64)
    eq_d, stride = _decimate(eq)
    series["equity"] = _vec(eq_d, 2)
    series["inventory_strategy"] = _vec(_decimate(invp)[0], 3)
    # PnL 曲线相对起点，画出来更直观
    base = float(eq[0]) if eq.size else 0.0
    series["pnl_curve"] = _vec(eq_d - base, 2)
    series["strategy_stride"] = stride

    hs = [1, 2, 5, 20, 60]
    markout = {
        "h": hs,
        "drift": [ev.get(f"drift_{h}_bp", {}).get("mean") for h in hs],
        "drift_sem": [ev.get(f"drift_{h}_bp", {}).get("sem") for h in hs],
        "drift_t": [ev.get(f"drift_{h}_bp", {}).get("t") for h in hs],
        "realized": [ev.get(f"realized_{h}_bp", {}).get("mean") for h in hs],
        "maker_drift": [ev.get(f"maker_drift_{h}_bp", {}).get("mean") for h in hs],
        "taker_drift": [ev.get(f"taker_drift_{h}_bp", {}).get("mean") for h in hs],
        "capture": ev.get("capture_bp", {}).get("mean"),
        "capture_t": ev.get("capture_bp", {}).get("t"),
        "n": ev.get("capture_bp", {}).get("n"),
    }
    result = {
        "kind": "strategy",
        "scenario": sc.name,
        "seed": seed,
        "warmup": sc.warmup,
        "ticks": ticks,
        "wall_sec": round(wall, 2),
        "n_agents": m.n_agents,
        "health_ok": bool(ok),
        "health_problems": list(problems[:5]),
        "strategy_label": label,
        "strategy_params": params,
        "strategy_errors": int(getattr(agent, "n_errors", 0)),
        "strategy_first_error": getattr(agent, "first_error", "") or "",
        "exec": ev,
        "markout": markout,
        "market": met.flat(),
        "hist": _returns_hist(mid_win),
        "acf": _acf_payload(met),
        "structure": {
            "n_agents": m.n_agents,
            "n_ticks_observed": int(mid_win.size),
            "trades_per_tick": _safe_mean(
                np.asarray(m.log.n_trades[t0:m.tick], dtype=float)
            ),
            "spread_bp_mean": _safe_mean(
                np.asarray(m.log.spread[t0:m.tick], dtype=float) / mid_win * 1e4
            ),
            "depth_total_mean": _safe_mean(
                np.asarray(
                    m.log.bid_depth[t0:m.tick] + m.log.ask_depth[t0:m.tick],
                    dtype=float,
                )
            ),
        },
    }
    if hasattr(agent, "completion"):
        result["completion"] = float(agent.completion)
    job.say(
        f"完成：{ev['n_fills']:,} 笔成交，价差捕获 {_fmt(ev['capture_bp']['mean'])}bp，"
        f"PnL {_fmt(ev['pnl_total_from_equity'], 0)}，用时 {wall:.1f}s"
    )
    if result["strategy_errors"]:
        job.say(f"⚠️ 策略抛了 {result['strategy_errors']} 次异常：{result['strategy_first_error']}")
    return result, series


# ----------------------------------------------------------------------
# 批量对照（实验台）
# ----------------------------------------------------------------------
def _lab_aggregate(rows: list[dict]) -> dict:
    keys = [
        "n_fills", "volume", "maker_volume_frac",
        "pnl_total_from_equity", "pnl_capture", "pnl_inventory", "pnl_initial_mark",
        "capture_bp|mean", "drift_1_bp|mean", "drift_20_bp|mean", "drift_20_bp|t",
        "realized_20_bp|mean", "max_dd", "sharpe_per_tick",
        "inv_abs_mean", "inv_abs_max", "flat_frac", "completion",
    ]
    out: dict = {"n_seeds": len(rows), "label": rows[0]["label"],
                 "scenario": rows[0]["scenario"]}

    def pick(d, key):
        if key in d:
            return d[key]
        cur = d
        for part in key.split("|"):
            if not isinstance(cur, dict) or part not in cur:
                return None
            cur = cur[part]
        return cur

    for k in keys:
        vals = np.asarray(
            [pick(r, k) if pick(r, k) is not None else np.nan for r in rows],
            dtype=np.float64,
        )
        vals = vals[np.isfinite(vals)]
        out[k] = round(float(vals.mean()), 6) if vals.size else None
        out[k + "|sem"] = (
            round(float(vals.std(ddof=1) / math.sqrt(vals.size)), 6)
            if vals.size > 1 else None
        )
    out["errors_total"] = int(sum(r.get("n_errors", 0) for r in rows))
    out["health_ok"] = all(r.get("health_ok", True) for r in rows)
    return out


def run_lab(job: Job) -> tuple[dict, dict | None]:
    spec = job.spec
    from strategies import REGISTRY

    names = spec.get("strategies") or list(REGISTRY)
    if "noop" not in names:
        names = ["noop", *names]  # 对照基准永远在
    scen_names = spec.get("scenarios") or ["normal"]
    seeds = [int(x) for x in (spec.get("seeds") or [])][:8]
    if not seeds:
        n_seed = _int(spec, "n_seeds", 3, 1, 8)
        seeds = [20260917 + 48271 * i for i in range(n_seed)]
    ticks = _int(spec, "ticks", 4000, 100, 20_000)

    scenarios = []
    for nm in scen_names:
        try:
            scenarios.append(get_scenario(nm))
        except KeyError:
            raise ApiError(f"未知场景 {nm!r}；可用：{sorted(SCENARIOS)}") from None

    tasks = [(n, s, sd) for n in names for s in scenarios for sd in seeds]
    if len(tasks) > MAX_LAB_RUNS:
        raise ApiError(
            f"组合数 {len(tasks)} 超过上限 {MAX_LAB_RUNS}；"
            f"减少策略 / 场景 / 种子数"
        )
    job.say(
        f"批量对照：{len(names)} 策略 × {len(scenarios)} 场景 × {len(seeds)} 种子 "
        f"= {len(tasks)} 次运行，每次观测 {ticks:,} tick"
    )

    runs: list[dict] = []
    for i, (sname, sc, sd) in enumerate(tasks):
        if job.cancelled():
            return None, None
        cls, params = REGISTRY[sname]
        m = Market(sc.config(sd))
        if sc.warmup:
            m.run(sc.warmup)
        t0 = m.tick
        agent = make_strategy(
            cls, "strat", seed=sd, cash=60_000.0 * 300.0, inventory=0.0, **params
        )
        m.add_agent(agent)
        if sc.shock_frac is not None and sc.shock_at < ticks:
            m.run(sc.shock_at)
            sc.apply_shock(m)
            m.run(ticks - sc.shock_at)
        else:
            m.run(ticks)
        ev = evaluate(m, "strat", from_tick=t0)
        ev["label"] = sname
        ev["scenario"] = sc.name
        ev["seed"] = sd
        ev["n_errors"] = int(getattr(agent, "n_errors", 0))
        ok, _ = m.health_check()
        ev["health_ok"] = bool(ok)
        runs.append(ev)
        job.progress = (i + 1) / len(tasks)
        job.phase = f"{sname} / {sc.name} / seed {sd}　({i + 1}/{len(tasks)})"

    agg = []
    for sc in scenarios:
        for sname in names:
            sub = [r for r in runs if r["scenario"] == sc.name and r["label"] == sname]
            if sub:
                agg.append(_lab_aggregate(sub))

    result = {
        "kind": "lab",
        "scenarios": [s.name for s in scenarios],
        "strategies": names,
        "seeds": seeds,
        "ticks": ticks,
        "n_runs": len(tasks),
        "aggregated": agg,
    }
    job.say(f"批量对照完成：{len(tasks)} 次运行")
    return result, None


# ----------------------------------------------------------------------
# 调度
# ----------------------------------------------------------------------
def validate_spec(spec: dict) -> None:
    """提交前把参数**全部检查一遍**，别等作业跑起来才报错。

    为什么值得单独走一趟：用户点完"运行"就盯着进度条等了，
    如果几十秒后才告诉他"策略名打错了"，那几十秒是纯浪费；
    更糟的是他可能直接关掉页面，以为程序卡死了。
    这里的校验成本是毫秒级，而作业成本是秒级到分钟级。

    只做校验，不产生副作用（不会真的建市场，也不会执行策略代码的
    ``on_tick``——但**会 exec 一遍代码本身**，因为语法错误只能这样查出来）。
    """
    kind = spec.get("kind") or "strategy"
    if kind not in ("market", "strategy", "lab"):
        raise ApiError(f"未知作业类型 {kind!r}")

    if kind == "lab":
        from strategies import REGISTRY

        names = spec.get("strategies") or list(REGISTRY)
        bad = [n for n in names if n not in REGISTRY]
        if bad:
            raise ApiError(f"未知策略 {bad}；可用：{sorted(REGISTRY)}")
        scens = spec.get("scenarios") or ["normal"]
        for nm in scens:
            if nm not in SCENARIOS:
                raise ApiError(f"未知场景 {nm!r}；可用：{sorted(SCENARIOS)}")
        n_seed = len(spec.get("seeds") or []) or _int(spec, "n_seeds", 3, 1, 8)
        total = max(1, len(names) + (0 if "noop" in names else 1)) * len(scens) * n_seed
        if total > MAX_LAB_RUNS:
            raise ApiError(
                f"组合数 {total} 超过上限 {MAX_LAB_RUNS}；减少策略 / 场景 / 种子数"
            )
        ticks = _int(spec, "ticks", 4000, 100, 20_000)
        from tw.scenarios import warmup_of
        warmup = max(warmup_of(nm2) for nm2 in scens)
        est = total * (ticks + warmup)
        if est > MAX_LAB_TICKS:
            raise ApiError(
                f"批量规模过大：约 {est:,} tick·次（上限 {MAX_LAB_TICKS:,}）。"
                f"把「每次 tick」调小、或减少场景/种子数。"
            )
        return

    _scenario_from_spec(spec)
    _int(spec, "seed", 20260917, 0, 2**31 - 1)
    _int(spec, "ticks", 6000, 50, MAX_TICKS)

    if kind == "strategy":
        _resolve_strategy(spec)  # 含自定义代码的编译检查
        _num(spec, "cash", 60_000.0 * 300.0, 0.0, 1e12)
        _num(spec, "inventory", 0.0, -1e6, 1e6)


def execute(job: Job) -> tuple[dict, dict | None]:
    kind = job.kind
    if kind == "market":
        return run_market(job)
    if kind == "strategy":
        return run_strategy(job)
    if kind == "lab":
        return run_lab(job)
    raise ApiError(f"未知作业类型 {kind!r}")


def export_csv(job: Job) -> str:
    """把作业结果摊平成 CSV（两列：指标, 值）。"""
    r = job.result or {}
    lines = ["key,value"]

    def emit(prefix: str, d) -> None:
        if not isinstance(d, dict):
            return
        for k, v in d.items():
            if isinstance(v, dict):
                emit(f"{prefix}{k}.", v)
            elif isinstance(v, (list, tuple)):
                continue
            else:
                s = "" if v is None else str(v)
                if "," in s:
                    s = '"' + s.replace('"', "'") + '"'
                lines.append(f"{prefix}{k},{s}")

    emit("", r)
    return "\n".join(lines) + "\n"


#: 进程级单例
MANAGER: JobManager | None = None


def manager(workers: int) -> JobManager:
    global MANAGER
    if MANAGER is None:
        MANAGER = JobManager(workers=workers)
    return MANAGER
