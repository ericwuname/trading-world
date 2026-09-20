"""策略实验台：多种子 × 多场景 × 多策略，一次性跑完并给出对照。

用法
----
    # 跑全部内置策略 × 3 个场景 × 4 个种子
    python scripts/lab.py

    # 只跑指定策略 / 场景
    python scripts/lab.py --strategies mm_naive,mm_skewed --scenarios normal,stressed

    # 快速试跑（少种子、短窗口）
    python scripts/lab.py --quick

    # 载入自己的策略（模块路径:类名）
    python scripts/lab.py --strategy my_strat:MyMaker

产出
----
    out/lab_results.json     全部逐次结果（可按策略/场景/种子下钻）
    out/lab_summary.csv      跨种子聚合后的对照表
    out/figs/lab_*.png       图
    docs/策略实验报告.html    单文件报告

三条设计原则
------------
**① 多种子是默认，不是选项。**
   本市场是肥尾的，单次权益曲线的方差极大。只跑一个种子挑最好看的结果，
   是把噪声当信号。默认 4 个种子并报告标准误，任何"优于基准"的结论
   都必须在标准误之外。

**② 每次都带 Noop 对照。**
   Noop 什么都不做，所以它的 PnL 必须**恰好为 0**、成交必须**恰好为 0**。
   它同时也是健全性检查：注入 Noop 后的行情必须与不注入时逐点相同
   （`--integrity` 会实测这一点）。如果这条不成立，
   整张表里所有差异都混着"多了一个主体"的伪影。

**③ 每个场景都打印"这个场景测什么"。**
   不知道数字该怎么解读的实验结果没有价值。
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _common import OUT, banner, save_json  # noqa: E402
from tw import Market, Population, SimConfig  # noqa: E402
from tw.eval import evaluate  # noqa: E402
from tw.scenarios import get_scenario, list_scenarios  # noqa: E402
from tw.strategy import Strategy, make_strategy  # noqa: E402

DEFAULT_SCENARIOS = ["normal", "stressed", "liquidation"]
DEFAULT_SEEDS = [20260917, 20261024, 20261131, 20261238]
QUICK_SEEDS = [20260917, 20261024]
DEFAULT_TICKS = 8_000
QUICK_TICKS = 2_500

#: 策略的资金/底仓。给足现金以免订单被预算裁剪掉——
#: 否则测出来的是"资金不够"而不是"策略好坏"。
STRATEGY_CASH = 60_000.0 * 300.0
STRATEGY_INV = 0.0

#: 跨种子聚合的指标。选的都是"决策结论会用到"的量。
AGG_KEYS = [
    "n_fills", "volume", "maker_volume_frac",
    "pnl_total_from_equity", "pnl_capture", "pnl_inventory", "pnl_initial_mark",
    "capture_bp|mean", "drift_1_bp|mean", "drift_20_bp|mean", "drift_20_bp|t",
    "realized_20_bp|mean", "maker_drift_20_bp|mean",
    "max_dd", "max_dd_pct", "sharpe_per_tick",
    "inv_abs_mean", "inv_abs_max", "flat_frac",
    "n_errors",
]


def pick(d: dict, key: str):
    """取值。**先试扁平键，再按 ``a|b`` 路径下钻。**

    为什么两种都要支持：逐次结果（`run_one`）里 ``capture_bp`` 是一个
    带 mean/t/sem 的字典，要用 ``capture_bp|mean`` 下钻；
    而跨种子聚合（`aggregate`）把每个指标压成**带竖线的扁平键**
    ``"capture_bp|mean"``。只支持一种的话，
    另一种取值会静默拿到 None，表格里整列变成 "—" —— 而不会报错。
    """
    if key in d:
        return d[key]
    cur = d
    for part in key.split("|"):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def load_user_strategy(spec: str) -> tuple[str, type[Strategy], dict]:
    """``模块路径:类名`` → (名字, 类, 默认参数)。"""
    if ":" not in spec:
        raise SystemExit(f"--strategy 需要写成 模块:类名，收到 {spec!r}")
    mod_name, cls_name = spec.split(":", 1)
    mod = importlib.import_module(mod_name)
    cls = getattr(mod, cls_name)
    if not (isinstance(cls, type) and issubclass(cls, Strategy)):
        raise SystemExit(f"{cls_name} 不是 Strategy 的子类")
    return cls_name, cls, {}


def registry(names: list[str] | None) -> dict[str, tuple[type, dict]]:
    from strategies import REGISTRY

    if not names:
        return dict(REGISTRY)
    out: dict[str, tuple[type, dict]] = {}
    for n in names:
        if n in REGISTRY:
            out[n] = REGISTRY[n]
        elif ":" in n:
            name, cls, kw = load_user_strategy(n)
            out[name] = (cls, kw)
        else:
            raise SystemExit(f"未知策略 {n!r}；可用：{sorted(REGISTRY)}")
    return out


def parity_check(scenario_name: str, seed: int, ticks: int) -> tuple[bool, str]:
    """健全性检查：注入 Noop 之后行情必须与不注入时**逐点相同**。

    这条不成立的话，整张对照表都不可信——因为"多了一个主体"
    本身就会改变行情，测到的差异里混着伪影。
    """
    sc = get_scenario(scenario_name)
    sc.n_ticks = ticks
    m0 = sc.build(seed=seed)
    base = np.asarray(m0.log.mid[: m0.tick], dtype=float)

    from strategies import Noop

    m1 = sc.build(seed=seed)
    m1.add_agent(make_strategy(Noop, "noop", seed=seed, cash=1.0, inventory=0.0))
    n1 = np.asarray(m1.log.mid[: m1.tick], dtype=float)
    if base.size != n1.size:
        return False, f"tick 数不同（{base.size} vs {n1.size}）"
    d = float(np.max(np.abs(base - n1))) if base.size else 0.0
    return d == 0.0, f"预热行情逐点最大差 {d:.6g}"


class StrategyContextShim:
    """给 `on_end` 用的最小快照。

    不直接复用 `StrategyContext`：那个对象绑在某个 tick 的 `MarketState` 上，
    而 `on_end` 调用时模拟已经结束、`_state` 是最后一 tick 的残留。
    这里显式构造一个只读视图，避免"on_end 里读到的行情其实是旧的"这种隐蔽问题。
    """

    def __init__(self, market, agent) -> None:  # noqa: ANN001
        self._m, self._a = market, agent

    @property
    def tick(self) -> int:
        return int(self._m.tick)

    @property
    def mid(self):
        m = self._m.current_mid()
        return float(m) if m and m > 0 else None

    @property
    def cash(self) -> float:
        return float(self._a.cash)

    @property
    def inventory(self) -> float:
        return float(self._a.inventory)

    def equity(self) -> float:
        return float(self._a.equity(self.mid))

    def pnl(self) -> float:
        return float(self._a.equity_change(self.mid))


def run_one(
    name: str,
    cls: type[Strategy],
    kw: dict,
    scenario_name: str,
    seed: int,
    ticks: int,
) -> dict:
    sc = get_scenario(scenario_name)
    sc.n_ticks = ticks
    m = sc.build(seed=seed)
    agent = make_strategy(
        cls, "strat", seed=seed, cash=STRATEGY_CASH, inventory=STRATEGY_INV, **kw
    )
    m.add_agent(agent)
    n_before = m.tick
    shock = None
    if sc.shock_frac is not None and sc.shock_at < ticks:
        m.run(sc.shock_at)
        shock = sc.apply_shock(m)
        m.run(ticks - sc.shock_at)
    else:
        m.run(ticks)

    # ⚠️ 必须调用 on_end：它是 `tw/strategy.py` 承诺的生命周期钩子。
    # 执行类策略（TWAP 等）在 on_end 里结算"完成度"，
    # 不调用它的话那个字段永远为空——而结果里看不到任何异常，只是少一列数。
    try:
        agent.on_end(StrategyContextShim(m, agent))
    except Exception as exc:  # noqa: BLE001
        agent.n_errors += 1
        agent.first_error = agent.first_error or f"on_end: {type(exc).__name__}: {exc}"

    r = evaluate(m, "strat")
    r["label"] = name
    r["scenario"] = scenario_name
    r["seed"] = seed
    r["n_errors"] = int(agent.n_errors)
    r["first_error"] = agent.first_error or ""
    r["injected_at"] = m.injected_at.get("strat")
    r["n_ticks_observed"] = int(m.tick - n_before)
    if hasattr(agent, "completion"):
        r["completion"] = float(agent.completion)
    if shock is not None:
        # 清算场景的成交率：实测这个数常常远低于 1——盘口深度就是清算能力的上限，
        # 甩不出去的仓位会留在账上。不记录它的话，
        # "清算场景下策略表现差"会被误读成策略问题，其实是市场没把仓位消化掉。
        target = float(len(shock["details"])) if shock.get("details") else 0.0
        r["shock_qty"] = float(shock["total_qty"])
        r["shock_n_agents"] = int(shock["n_agents"])
        r["shock_target"] = target
    ok, problems = m.health_check()
    r["health_ok"] = bool(ok)
    r["health_problems"] = problems[:3]
    return r


def aggregate(rows: list[dict]) -> dict:
    out: dict = {"n_seeds": len(rows), "label": rows[0]["label"],
                 "scenario": rows[0]["scenario"]}
    for key in AGG_KEYS:
        vals = np.asarray([pick(r, key) for r in rows], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            out[key] = float("nan")
            out[key + "|sem"] = float("nan")
            continue
        out[key] = float(vals.mean())
        out[key + "|sem"] = (
            float(vals.std(ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else float("nan")
        )
    out["errors_total"] = int(sum(r.get("n_errors", 0) for r in rows))
    out["first_error"] = next(
        (r.get("first_error") for r in rows if r.get("first_error")), ""
    )
    out["health_ok"] = all(r.get("health_ok", True) for r in rows)
    comp = np.asarray([r.get("completion", np.nan) for r in rows], dtype=float)
    comp = comp[np.isfinite(comp)]
    out["completion"] = float(comp.mean()) if comp.size else float("nan")
    return out


def _f(v, nd: int = 2) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(x):
        return "—"
    return f"{x:,.{nd}f}"


def print_table(aggs: list[dict], scenario: str) -> None:
    cols = [
        ("策略", "label", 0),
        ("成交笔数", "n_fills", 0),
        ("被动占比", "maker_volume_frac", 3),
        ("价差捕获bp", "capture_bp|mean", 2),
        ("漂移h20bp", "drift_20_bp|mean", 2),
        ("h20的t", "drift_20_bp|t", 2),
        ("实现h20bp", "realized_20_bp|mean", 2),
        ("PnL合计", "pnl_total_from_equity", 0),
        ("±标准误", "pnl_total_from_equity|sem", 0),
        ("其中价差", "pnl_capture", 0),
        ("其中库存", "pnl_inventory", 0),
        ("最大回撤", "max_dd", 0),
        ("库存均值", "inv_abs_mean", 2),
        # 执行类策略（TWAP 等）在 on_end 里算的目标完成度；
        # 非执行类策略没有这个字段，显示 "—"。
        ("完成度", "completion", 3),
    ]
    heads = [c[0] for c in cols]
    body = [[_f(pick(a, k), nd) if k != "label" else str(a[k]) for _, k, nd in cols]
            for a in aggs]
    widths = [max(len(heads[i]), *(len(b[i]) for b in body)) if body else len(heads[i])
              for i in range(len(heads))]

    def line(cells):
        return "  ".join(c.rjust(widths[i]) for i, c in enumerate(cells))

    print(f"\n  场景 {scenario}")
    print("  " + line(heads))
    print("  " + "-" * (sum(widths) + 2 * (len(widths) - 1)))
    for b in body:
        print("  " + line(b))


def write_csv(aggs: list[dict], path: Path) -> None:
    keys = ["label", "scenario", "n_seeds"] + [
        k for k in AGG_KEYS
    ]
    lines = [",".join(keys)]
    for a in aggs:
        row = []
        for k in keys:
            v = a.get(k)
            if isinstance(v, str):
                row.append('"' + v.replace('"', "'") + '"')
            elif v is None:
                row.append("")
            else:
                row.append(f"{v:.6g}" if isinstance(v, (int, float)) else str(v))
        lines.append(",".join(row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> dict:
    ap = argparse.ArgumentParser(description="策略实验台")
    ap.add_argument("--strategies", default="", help="逗号分隔；默认全部内置")
    ap.add_argument("--scenarios", default=",".join(DEFAULT_SCENARIOS))
    ap.add_argument("--seeds", type=int, default=len(DEFAULT_SEEDS))
    ap.add_argument("--ticks", type=int, default=DEFAULT_TICKS)
    ap.add_argument("--quick", action="store_true", help="少种子、短窗口，用于快速试跑")
    ap.add_argument("--no-integrity", action="store_true", help="跳过 Noop 健全性检查")
    ap.add_argument("--list", action="store_true", help="只列出场景与策略")
    args = ap.parse_args()

    if args.list:
        print("场景：")
        for n, intent in list_scenarios():
            print(f"  {n:<13} {intent}")
        from strategies import REGISTRY

        print("\n策略：")
        for n, (cls, kw) in REGISTRY.items():
            print(f"  {n:<13} {cls.__doc__.strip().splitlines()[0] if cls.__doc__ else ''}")
        return {}

    banner("策略实验台")
    strat_names = [s for s in args.strategies.split(",") if s] or None
    reg = registry(strat_names)
    if "noop" not in reg:
        from strategies import Noop

        reg["noop"] = (Noop, {})  # 对照基准永远在
    scen_names = [s for s in args.scenarios.split(",") if s]
    seeds = (QUICK_SEEDS if args.quick else DEFAULT_SEEDS)[: args.seeds]
    ticks = QUICK_TICKS if args.quick else args.ticks

    print(f"  策略 {len(reg)} 个：{', '.join(reg)}")
    print(f"  种子 {len(seeds)} 个：{seeds}")
    print(f"  场景 {len(scen_names)} 个：{', '.join(scen_names)}")
    print(f"  每个组合观测 {ticks:,} tick（另有预热）")
    print(f"  总计 {len(reg) * len(seeds) * len(scen_names)} 次运行")

    # ---------- 健全性检查 ----------
    if not args.no_integrity:
        print("\n  ▸ 健全性检查：注入 Noop 是否改变背景行情")
        ok, msg = parity_check(scen_names[0], seeds[0], min(ticks, 2000))
        print(f"    {scen_names[0]} / seed {seeds[0]}：{'✅ 逐点一致' if ok else '❌ 有差异'} —— {msg}")
        if not ok:
            print(
                "    ⚠️ 这条不成立时，下面所有对照都混着"
                "「多了一个主体」的伪影，结论不可用。"
            )

    # ---------- 主循环 ----------
    t0 = time.time()
    all_rows: list[dict] = []
    aggs: list[dict] = []
    for scen in scen_names:
        for name, (cls, kw) in reg.items():
            rows = []
            for sd in seeds:
                rows.append(run_one(name, cls, kw, scen, sd, ticks))
            all_rows.extend(rows)
            aggs.append(aggregate(rows))
        print_table([a for a in aggs if a["scenario"] == scen], scen)
        errs = [a for a in aggs if a["scenario"] == scen and a["errors_total"]]
        for a in errs:
            print(f"    ⚠️ {a['label']} 抛了 {a['errors_total']} 次异常：{a['first_error']}")

    # ---------- 产出 ----------
    out = {
        "config": {
            "seeds": seeds, "scenarios": scen_names, "ticks": ticks,
            "strategies": {k: {"cls": v[0].__name__, "kwargs": v[1]} for k, v in reg.items()},
            "strategy_cash": STRATEGY_CASH,
        },
        "aggregated": aggs,
        "runs": all_rows,
        "elapsed_sec": time.time() - t0,
    }
    save_json(out, "lab_results.json")
    write_csv(aggs, OUT / "lab_summary.csv")
    print(f"\n  明细 → out/lab_results.json；对照表 → out/lab_summary.csv")

    # ---------- 图 ----------
    try:
        _plots(all_rows, aggs)
        print(f"  图已输出到 {OUT / 'figs'}")
    except Exception as exc:  # noqa: BLE001
        print(f"  ⚠️ 绘图失败（不影响数据）：{type(exc).__name__}: {exc}")

    print(f"\n  总用时 {time.time() - t0:,.0f}s")
    print("""
  怎么读这张表
  ------------
  · **noop 必须恰好 0 成交、0 盈亏**。不为 0 说明评估口径有 bug，先修它。
  · **价差捕获** 是成交瞬间相对中间价的优势（被动挂单为正）；
    **漂移 h20** 是成交后 20 tick 价格朝哪走，看它的 t 值：
    |t| < 2 就是噪声，不能解读。**实现 h20 = 捕获 + 漂移**，
    做市类策略能不能赚钱就看它稳不稳定为正。
  · **PnL 分成三块**：价差 / 库存 / 底仓重估。上涨行情里"赚了钱"
    可能整块来自库存项——那不是策略赚的，是行情给的。
  · **±标准误** 跨种子。任何"优于基准"的结论都必须在标准误之外。""")
    return out


def _plots(rows: list[dict], aggs: list[dict]) -> None:
    import matplotlib.pyplot as plt

    from _common import FIG
    from tw.viz import C_ACCENT, C_DIM, C_GOOD, C_SIM, MUTED

    scens = sorted({r["scenario"] for r in rows})
    labels = [a["label"] for a in aggs if a["scenario"] == scens[0]]

    # 图1：各策略在 pnl_capture / pnl_inventory 上的分解（按场景分组）
    fig, axes = plt.subplots(1, len(scens), figsize=(4.6 * len(scens), 4.2), squeeze=False)
    for ax, sc in zip(axes[0], scens):
        sub = {a["label"]: a for a in aggs if a["scenario"] == sc}
        x = np.arange(len(sub))
        cap = [sub[k].get("pnl_capture", np.nan) for k in sub]
        inv = [sub[k].get("pnl_inventory", np.nan) for k in sub]
        ini = [sub[k].get("pnl_initial_mark", np.nan) for k in sub]
        ax.bar(x - 0.25, cap, 0.25, color=C_SIM, label="价差捕获")
        ax.bar(x, inv, 0.25, color=C_ACCENT, label="库存变动")
        ax.bar(x + 0.25, ini, 0.25, color=C_DIM, label="底仓重估")
        ax.axhline(0, color=MUTED, lw=1.0, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels(list(sub), rotation=35, ha="right", fontsize=8.5)
        ax.set_title(f"{sc}", fontsize=10.5)
        ax.tick_params(axis="y", labelsize=8.5)
    axes[0][0].set_ylabel("PnL")
    axes[0][0].legend(fontsize=8)
    fig.suptitle("PnL 分解：钱是从哪来的？（价差 = 本事；库存与底仓重估 = 行情）", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "lab_pnl_decomposition.png")
    plt.close(fig)

    # 图2：markout 曲线（成交后 h tick 的漂移），按策略
    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    hs = [1, 2, 5, 20, 60]
    norm = plt.Normalize(0, max(1, len(labels) - 1))
    cmap = plt.get_cmap("coolwarm")
    for i, lab in enumerate(labels):
        sub = [a for a in aggs if a["scenario"] == "normal" and a["label"] == lab]
        if not sub:
            continue
        a = sub[0]
        y = [a.get(f"drift_{h}_bp|mean", np.nan) for h in hs]
        if not np.isfinite(y).any():
            continue
        ax.plot(hs, y, "o-", lw=1.5, ms=5, color=cmap(norm(i)), label=lab)
    ax.axhline(0, color=MUTED, lw=1.0, ls="--")
    ax.set_xscale("log")
    ax.set_xticks(hs)
    ax.set_xticklabels([str(h) for h in hs])
    ax.set_xlabel("成交之后经过的 tick 数")
    ax.set_ylabel("价格漂移 (bp)，按策略方向对齐")
    ax.set_title("Markout 曲线（normal 场景）：成交后价格朝哪走\n"
                 "负值 = 逆向选择（接了有毒的单）；主动单付的价差已经算在「捕获」里",
                 fontsize=10.5)
    ax.legend(fontsize=8.5, ncol=2)
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    fig.savefig(FIG / "lab_markout.png")
    plt.close(fig)

    # 图3：风险-收益散点（最大回撤 vs PnL）
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    for i, lab in enumerate(labels):
        sub = [a for a in aggs if a["label"] == lab]
        if not sub:
            continue
        xs = [a.get("max_dd", np.nan) for a in sub]
        ys = [a.get("pnl_total_from_equity", np.nan) for a in sub]
        ax.scatter(xs, ys, s=70, color=cmap(norm(i)), zorder=3)
        for a in sub:
            ax.annotate(a["scenario"], (a.get("max_dd", np.nan),
                                        a.get("pnl_total_from_equity", np.nan)),
                        fontsize=7.5, xytext=(4, 3), textcoords="offset points",
                        color=MUTED)
        ax.plot([], [], "o", color=cmap(norm(i)), label=lab)
    ax.axhline(0, color=MUTED, lw=1.0, ls="--")
    ax.set_xlabel("最大回撤（金额）")
    ax.set_ylabel("PnL 合计")
    ax.set_title("风险 vs 收益：同一策略在不同场景下的位置", fontsize=11)
    ax.legend(fontsize=8.5)
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    fig.savefig(FIG / "lab_risk_return.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
