"""工作线H：路径C —— 饱和检查（**不加种子**，无条件先做）。

任务书：`交易世界 · 分辨力危机应对任务书.md` §2。

背景与定位
----------
诊断报告证明：k 在「3 种子 × 4 档」下**没有分辨力**——19/19 个历史臂的
95% 区间都包含 0.5。路径 A（加种子到 300+）已被决定不做。
本工作线是那个便宜的选项：**把冲击规模从 4 档扩到 7 档，不加任何种子**，
先看两件事：
  ① 每个配置的**饱和边界**在哪一档（``fill_ratio < 0.95`` 即饱和）；
  ② 未饱和档位上的滑点**长什么形状**——用肉眼与基本统计看，**不做任何幂律拟合**。

⚠️ 动手前的前提核对（任务书是外部 AI 凭报告写的，本轮核对出**三处**与源码不符）
-------------------------------------------------------------------------------
1. 任务书写 ``from _common import STAGE3_SHOCK_SIZES`` ——
   **`_common` 里没有这个常量**。7 档的定义在 ``run_stage3.SHOCK_SIZES``。
   按项目惯例**不新造第二份**，直接从定义处导入。
2. 任务书假设配置里要"复用 EA.2 的对称配置"。EA.2（突发期深度）用的就是
   ``HawkesMarket`` 的默认 ``thin_scope="both"``，所以"对称配置"实际就是
   EA.4 的 ``both`` 臂。本脚本按这个口径实现，不另建一份配置。
3. 任务书 §2.6 要求"重新核实 91.692 在 7 档实验的窗口下是否依然成立"。
   窗口定义（``[6000, 6400)``）没有变，而标定只依赖市场配置与窗口，
   所以**应当逐位一致**。本脚本把这件事做成一次**显式断言**并写进产物，
   而不是"看一眼觉得一样"。

用法::

    python scripts/run_workstream_H.py            # 三配置 × 7 档
    python scripts/run_workstream_H.py --only baseline_no_hawkes
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import FIG, OUT, banner, ensure_scripts_on_path, save_json  # noqa: E402

RESULT = "workstream_H_metrics.json"
#: 「目标被吃满」的判定阈值——与 ``tw.impact.unsaturated`` 的默认保持一致。
#  ⚠️ 判定方向：``fill_ratio < SAT_THRESHOLD`` 才是**饱和**。
#     反过来写会让所有档位都被标成饱和、未饱和集为空（变异体 M45 钉的就是这个）。
SAT_THRESHOLD = 0.95
SLICES = 20                    # 与 EA.4 一致（阶段3 的 C2 用的是 10，见文档说明）
BRANCHING = 0.6
BETA = 0.15

#: 三个配置。名字与理由写在这里，报告里直接引用。
CONFIGS = {
    "baseline_no_hawkes": "裸 Market（Hawkes 关）——「不聚集」的基准",
    "ea4_taker_only": "Hawkes 只稀疏化吃单方（EA.4 的官方基准配置）",
    "ea2_both_symmetric": "Hawkes 双侧同时稀疏化（对称；= EA.4 的 both 臂）",
}


def is_saturated(fill_ratio: float, threshold: float = SAT_THRESHOLD) -> bool:
    """本档是否**饱和**：目标没被吃满即饱和。

    ⚠️ 方向必须写死在这一处：``fill_ratio **小于** threshold`` 才是饱和。
    反过来写会让所有档位都被标成饱和、未饱和集为空——而且**不报错**，
    只是后续所有分析都建立在空集上（变异体 M46 钉的正是这里）。
    """
    return bool(fill_ratio < threshold)


def _spec_and_window():
    """复用工作线A 的配置事实源——**不写第二份**（第六条纪律）。"""
    ensure_scripts_on_path()
    import run_workstream_A as A
    return A, A.ea4_market_spec(), A.ea4_windows()


def calibrate_lambda(seeds) -> dict:
    """在**实验窗口**上标定 λ̄，并核对它与 EA.4 官方基准（91.692）是否一致。

    任务书 §2.6 点名要做这件事。做法：复用 ``ea4_calibration_kwargs``，
    它把标定窗口设成 = 实验窗口 = ``[6000, 6400)``。
    """
    ensure_scripts_on_path()
    from tw.order_flow.hawkes_market import calibrate_base_lambda
    A, spec, windows = _spec_and_window()
    kw = A.ea4_calibration_kwargs(seeds, windows["experiment"], spec)
    lam = calibrate_base_lambda(**kw)
    return {
        "base_lambda": lam,
        "window": list(windows["experiment"]),
        "calibration_kwargs": {k: v for k, v in kw.items() if k != "mix"},
        "matches_ea4_official": bool(abs(lam - 91.692) < 0.01),
    }


def _factory_for(cfg: str, base_lambda: float | None, sink: list):
    """按配置名给出 ``factory``（``make_market`` 的唯一扩展点）。

    ``baseline_no_hawkes`` ⇒ ``None``（裸 Market，一期行为逐点不变）。
    两个 Hawkes 配置只差 ``thin_scope``——这正是 EA.4 与"对称"的区别所在。
    """
    if cfg == "baseline_no_hawkes":
        return None
    ensure_scripts_on_path()
    from tw.order_flow.hawkes import HawkesConfig
    from tw.order_flow.hawkes_market import HawkesMarket

    scope = "taker" if cfg == "ea4_taker_only" else "both"
    hcfg = HawkesConfig(base_lambda=base_lambda, branching=BRANCHING, beta=BETA)

    def fn(sim):
        m = HawkesMarket(sim, hawkes_config=hcfg, thin_scope=scope)
        if sink is not None:
            sink.append(m)
        return m

    return fn


def run_saturation_sweep(seeds=None, shock_sizes=None, slices: int = SLICES,
                         only=None) -> dict:
    """对每个配置、每档冲击规模跑一次配对测量；**不调用 fit_power_law**。

    每个配置都用自己对应的 factory 建市场，且控制组与处理组传**同一个**
    factory（否则配对差里混进了"两组市场实现不同"这个无关变量）。
    """
    ensure_scripts_on_path()
    import run_workstream_A as A
    from run_stage3 import HORIZON, SHOCK_SIZES, make_controls, paired_impact
    from run_stage3 import WARMUP as W3_WARMUP

    seeds = list(seeds or A.SEEDS)
    shock_sizes = list(shock_sizes or SHOCK_SIZES)
    A_, spec, windows = _spec_and_window()
    mix = spec["mix"]

    print(f"  实验窗口={windows['experiment']}  种子={len(seeds)} 个  "
          f"档位={len(shock_sizes)} 档  slices={slices}")
    lam = calibrate_lambda(seeds)
    print(f"  λ̄={lam['base_lambda']:.3f}"
          f"（与 EA.4 官方基准一致：{'✅' if lam['matches_ea4_official'] else '❌'}）")

    out: dict = {"lambda": lam, "seeds": seeds, "shock_sizes": shock_sizes,
                 "slices": slices, "sat_threshold": SAT_THRESHOLD,
                 "experiment_window": list(windows["experiment"]),
                 "configs": {}}

    for cfg, desc in CONFIGS.items():
        if only and cfg != only:
            continue
        print(f"\n  【{cfg}】{desc}")
        sink: list = []
        fn = _factory_for(cfg, lam["base_lambda"], sink)
        ctrl = make_controls(mix, seeds, HORIZON, factory=fn)
        rows = []
        for size in shock_sizes:
            r = paired_impact(f"{cfg}|{size:.4f}", mix, size, slices, seeds,
                              ctrl, HORIZON, factory=fn)
            row = {
                # ⚠️ 用 shock_frac 而不是 size：与项目其它产物的字段名统一，
                # 这样 tw.analyzer_concavity.elasticity_from_rows 能直接复用。
                "shock_frac": float(size),
                "delivered_qty": float(r["delivered_qty"]),
                "fill_ratio": float(r["fill_ratio"]),
                # ⚠️ 字段名与项目其它产物**保持一致**（slippage_bp / slippage_sem_bp），
                # 这样 diagnose_k_uncertainty / analyzer_concavity 等现有工具能直接复用，
                # 不需要为 H 单独写一套读取逻辑。
                "slippage_bp": float(r["slippage_bp"]),
                "slippage_sem_bp": float(r["slippage_sem_bp"]),
                "t": (float(r["slippage_t"])
                      if r.get("slippage_t") is not None else float("nan")),
                "n_seeds": len(seeds),
                # ⚠️ 方向统一由 is_saturated() 决定（测试钉的是那个函数）
                "saturated": is_saturated(r["fill_ratio"]),
            }
            rows.append(row)
            mark = "饱和" if row["saturated"] else "    "
            print(f"    {size:<7} qty={row['delivered_qty']:>7.2f}  "
                  f"fill={row['fill_ratio']:.4f} {mark}  "
                  f"滑点={row['slippage_bp']:>8.3f} ± {row['slippage_sem_bp']:.3f}  "
                  f"t={row['t']:>6.2f}")
        unsat = [r for r in rows if not r["saturated"]]
        first_sat = next((r["shock_frac"] for r in rows if r["saturated"]), None)
        out["configs"][cfg] = {
            "desc": desc,
            "rows": rows,
            "n_unsaturated": len(unsat),
            "saturated_from": first_sat,
            "unsaturated_sizes": [r["shock_frac"] for r in unsat],
            "max_abs_t_unsaturated": (
                max(abs(r["t"]) for r in unsat) if unsat else None),
        }
        print(f"    ⇒ 未饱和 {len(unsat)}/{len(rows)} 档；"
              f"饱和边界={'从 ' + str(first_sat) + ' 起' if first_sat else '无（7 档全部未饱和）'}")

    return out


def plot_log_log_scatter(res: dict, out_path: Path) -> dict:
    """EH.2：画 log(冲击规模) vs log(|滑点|) 散点。

    **只用未饱和的点**，人工判读"明显下弯（凹）"还是"接近直线（线性）"。
    这是定性观察，不替代工作线I 的正式判据，只作为先验方向。
    任务书要求这张图**必须实际生成并保存**，不能只用文字描述。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=140)
    colors = {"baseline_no_hawkes": "#4C78A8", "ea4_taker_only": "#E45756",
              "ea2_both_symmetric": "#54A24B"}
    n_points = {}
    for cfg, d in (res.get("configs") or {}).items():
        pts = [(r["shock_frac"], abs(r["slippage_bp"])) for r in d["rows"]
               if not r["saturated"] and r["slippage_bp"] != 0]
        n_points[cfg] = len(pts)
        if not pts:
            continue
        x = np.log([p[0] for p in pts])
        y = np.log([p[1] for p in pts])
        ax.plot(x, y, "o-", color=colors.get(cfg, "gray"), label=cfg)
        # 参考直线：过首点、斜率 1（线性）与 0.5（平方根律）
        x0, y0 = x[0], y[0]
        xs = np.array([x.min(), x.max()])
        ax.plot(xs, y0 + 1.0 * (xs - x0), "--", color=colors.get(cfg, "gray"),
                alpha=0.45, linewidth=1)
        ax.plot(xs, y0 + 0.5 * (xs - x0), ":", color=colors.get(cfg, "gray"),
                alpha=0.45, linewidth=1)
    ax.set_xlabel("log(shock size)")
    ax.set_ylabel("log(|slippage|)")
    # ⚠️ 图内文字一律用 ASCII：matplotlib 的默认字体（DejaVu Sans）没有 CJK 字形，
    #    写中文会全部渲染成方块（实测踩到，见试跑日志的 UserWarning）。
    #    与其去赌系统里装了哪个中文字体，不如让**图**用英文、
    #    **报告正文**用中文——那才是给读者看的地方。
    ax.set_title("log-log of unsaturated levels "
                 "(dashed=slope 1 linear, dotted=slope 0.5 sqrt law)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return {"path": str(out_path), "n_points": n_points}


def main() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="只跑某一个配置")
    ap.add_argument("--seeds", type=int, default=None, help="只用前 N 个种子")
    args = ap.parse_args()

    banner("工作线H  路径C —— 饱和检查（不加种子）")
    t0 = time.time()
    ensure_scripts_on_path()
    seeds = None
    if args.seeds:
        import run_workstream_A as A
        seeds = A.SEEDS[:args.seeds]
    out = run_saturation_sweep(seeds=seeds, only=args.only)
    out["stage"] = "H"
    fig_info = plot_log_log_scatter(out, FIG / "workstream_H_loglog.png")
    out["figure"] = fig_info
    print(f"\n  散点图已保存：{fig_info['path']}"
          f"（未饱和点数：{fig_info['n_points']}）")
    out["elapsed_sec"] = time.time() - t0
    save_json(out, RESULT)
    print(f"  产物：out/{RESULT}（{out['elapsed_sec']:.0f}s）")
    return out


if __name__ == "__main__":
    main()
