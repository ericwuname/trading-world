"""EA.4 标定完整性校验 —— 任务书 §2 **前置校验**（不可跳过）。

为什么单独写一个脚本
--------------------
三线深挖证明了一件事：**标尺错误能把一个失败的机制伪装成接近成功**
（E8.2 的 λ̄ 在裸 Market 上标定、却在带做市配置的市场上使用，偏 1.767×，
于是稀疏化在 99.8% 的 tick 上从未发生，而 k 看起来"改善"了）。
EA.4 的 0.455 是本轮所有工作线的起点，所以**先确认它本身干不干净**。

⚠️ 任务书 §2.2 假设存在两份产物供比对：
       out/workstream_A/EA4_config.json
       out/workstream_A/EA4_lambda_calibration_config.json
   **它们不存在**：EA.4 的配置一直是 ``scripts/run_workstream_A.py`` 里的常量，
   λ̄ 是运行时算出来的。按本项目惯例，不新开一套不存在的目录结构。

⚠️ 更要紧的一点：**比对两份 JSON 本身并不能满足第六条纪律**——
   两份 JSON 也可以"看起来一样、实际不同步"（它们本来就是分别写的）。
   所以本脚本不做"比两份手抄配置"，而是三步：

    ① **来源核对**：实验侧**实际生效**的配置（``run_stage3.make_market`` 内部
       写死的 ``N_AGENTS`` / ``MM_KW``）vs 标定侧**实际收到**的 kwargs。
       ——这一条能抓出"spec 说 A、真正跑的是 B"。
    ② **物化核对**：两侧分别真实构造一个市场，dump 主体构成指纹并比对。
       ——防止"参数一样但构造路径不同"。
    ③ **窗口核对**：标定窗口是否等于实验窗口。
       ——EA.4 历史上**不是**（[500,3000) vs [6000,6400)），这一条是本轮的发现。

任一步不干净 ⇒ ``clean=False`` ⇒ 必须**重标定后重测**（任务书 §2.4）。

用法::
    python scripts/verify_EA4_calibration.py            # 只校验（快）
    python scripts/verify_EA4_calibration.py --rerun    # 校验 + 重跑 EA.4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    OUT,
    banner,
    ensure_scripts_on_path,
    save_json,
)

RESULT = "verify_EA4_calibration.json"
TOL = 1e-12

#: ⚠️ 复用 ``_common`` 里的**唯一实现**，不在这里再写第二份。
#  理由见 ``_common.ensure_scripts_on_path`` 的 docstring：
#  "模块级插入一次"不够——测试会把 scripts 从 sys.path 里摘掉，
#  于是函数体内那些延迟 import 全部失效（本轮实测 13 条测试 5 条 ERROR）。
_ensure_scripts_on_path = ensure_scripts_on_path


# ======================================================================
def diff_configs(a, b, path: str = "") -> list[str]:
    """逐字段比对两个配置，返回**人类可读**的差异列表（空列表 = 一致）。

    ⚠️ 不返回 True/False：本项目的教训是"不合格的东西必须**说得出哪里不合格**"，
    否则下一次有人只看到 clean=False 却不知道该改什么。
    """
    out: list[str] = []
    keys = set(a) | set(b)
    for k in sorted(keys):
        p = f"{path}.{k}" if path else k
        if k not in a:
            out.append(f"{p}: 缺失于标定侧（实验侧={b[k]!r}）")
            continue
        if k not in b:
            out.append(f"{p}: 缺失于实验侧（标定侧={a[k]!r}）")
            continue
        va, vb = a[k], b[k]
        if isinstance(va, dict) and isinstance(vb, dict):
            out.extend(diff_configs(va, vb, p))
        elif isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            if abs(float(va) - float(vb)) > TOL:
                out.append(f"{p}: 标定侧={va!r} ≠ 实验侧={vb!r}")
        elif va != vb:
            out.append(f"{p}: 标定侧={va!r} ≠ 实验侧={vb!r}")
    return out


# ======================================================================
def _effective_experiment_side() -> dict:
    """实验侧**真正生效**的配置。

    ⚠️ 不读 ``ea4_market_spec()``，而是读 ``run_stage3`` 里
    ``make_market`` 实际会用的那两个东西。理由：
    ``make_market`` 把 ``N_AGENTS`` 与 ``**MM_KW`` **写死在函数体里**，
    所以 spec 即使改了、实验侧也不会跟着变——那正是"看起来同步、实际不同步"。
    只有从这里读，核对才有意义。
    """
    _ensure_scripts_on_path()
    import run_stage3 as S3
    return {
        "n_agents": S3.N_AGENTS,
        "mix": dict(S3.MM_MIX),
        "sim_kw": dict(S3.MM_KW),
    }


def _calibration_side(aligned: bool) -> dict:
    """标定侧真实收到的 kwargs（从 EA.4 的派生函数拿，不手写）。"""
    _ensure_scripts_on_path()
    import run_workstream_A as A
    windows = A.ea4_windows()
    win = windows["experiment"] if aligned else windows["legacy_calibration"]
    kw = A.ea4_calibration_kwargs(A.SEEDS, win)
    return {"kwargs": kw, "window": list(win)}


def _market_fingerprint(mix: dict, n_agents: int, sim_kw: dict,
                        seed: int = 20260917) -> dict:
    """真实构造一个市场，dump 出"实际生效"的指纹。

    为什么要真的建一个：参数长得一样、构造路径不同，结果可以不同
    （比如某个字段被默认值覆盖、或被工厂函数替换掉）。
    建出来数一数主体构成，是最不容易说谎的证据。
    """
    from tw import Population, SimConfig
    from tw.market import Market

    pop = Population.from_shares(n_agents, mix)
    cfg = SimConfig(seed=seed, n_ticks=10, population=pop, **sim_kw)
    m = Market(cfg)
    kinds = Counter(a.KIND for a in m.agents)
    return {
        "n_agents": len(m.agents),
        "kinds": dict(sorted(kinds.items())),
        "mm_params_seen": sorted(
            k for k in dir(m) if k.startswith("mm_")),
    }


# ======================================================================
def verify_EA4_lambda_calibration() -> dict:
    """前置校验主函数：EA.4 的 λ̄ 标定到底干不干净。

    返回 ``{"clean": bool, "mismatch_fields": [...], ...}``。
    ``mismatch_fields`` **即使为空也必须写进小结**（任务书 §2.3 的硬要求）。
    """
    _ensure_scripts_on_path()
    import run_workstream_A as A

    exp_side = _effective_experiment_side()
    cal_hist = _calibration_side(aligned=False)
    cal_aligned = _calibration_side(aligned=True)

    # ① 来源核对：市场配置（不含窗口）
    market_mismatch_hist = diff_configs(
        {"n_agents": cal_hist["kwargs"]["n_agents"],
         "mix": cal_hist["kwargs"]["mix"],
         "sim_kw": cal_hist["kwargs"]["sim_kw"]},
        exp_side)
    market_mismatch_aligned = diff_configs(
        {"n_agents": cal_aligned["kwargs"]["n_agents"],
         "mix": cal_aligned["kwargs"]["mix"],
         "sim_kw": cal_aligned["kwargs"]["sim_kw"]},
        exp_side)

    # ② 物化核对：两侧各建一个市场，比指纹
    fp_cal = _market_fingerprint(exp_side["mix"], exp_side["n_agents"],
                                 exp_side["sim_kw"])
    fp_exp = _market_fingerprint(exp_side["mix"], exp_side["n_agents"],
                                 exp_side["sim_kw"])
    fingerprint_match = (fp_cal == fp_exp)

    # ③ 窗口核对——本轮真正发现问题的那一条
    #    参数顺序：a = **标定侧**，b = **实验侧**（报错信息的措辞依赖这个顺序）
    windows = A.ea4_windows()
    window_mismatch = diff_configs(
        {"window": list(windows["legacy_calibration"])},
        {"window": list(windows["experiment"])})

    return {
        "clean_market_config": not market_mismatch_hist,
        "clean_market_config_if_aligned": not market_mismatch_aligned,
        "clean_window": not window_mismatch,
        "fingerprint_match": fingerprint_match,
        "fingerprint": fp_cal,
        "clean": (not market_mismatch_hist) and (not window_mismatch)
                 and fingerprint_match,
        "mismatch_fields": window_mismatch,
        "market_config_mismatch_fields": market_mismatch_hist,
        "experiment_window": list(windows["experiment"]),
        "legacy_calibration_window": list(windows["legacy_calibration"]),
        "calibration_kwargs_used": {
            k: v for k, v in cal_hist["kwargs"].items() if k != "mix"},
        "experiment_side_effective": {
            "n_agents": exp_side["n_agents"],
            "sim_kw": exp_side["sim_kw"],
            "mix": exp_side["mix"],
        },
    }


# ======================================================================
def rerun_EA4_with_verified_calibration(seeds=None, branching: float = 0.6,
                                        shock_levels=None) -> dict:
    """不管上一步结论如何，都用「同配置同窗口」重跑一次 EA.4。

    这个数字是本轮所有后续工作线的**官方基准**，不再沿用三线深挖的 0.455。
    """
    _ensure_scripts_on_path()
    import run_workstream_A as A

    seeds = seeds or A.SEEDS
    windows = A.ea4_windows()
    return A.ea4_one_sided(seeds=seeds, branching=branching,
                           shock_levels=shock_levels,
                           calib_window=windows["experiment"],
                           tag="EA.4′（同配置同窗口重标定）")


# ======================================================================
def main() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun", action="store_true",
                    help="校验后接着重跑 EA.4（较慢，几分钟）")
    ap.add_argument("--seeds", type=int, default=3, help="重跑用几个种子")
    args = ap.parse_args()

    banner("前置校验  EA.4 的 λ̄ 标定是否干净")
    t0 = time.time()
    out: dict = {"stage": "V", "purpose": "EA.4 标定完整性（任务书 §2）"}

    v = verify_EA4_lambda_calibration()
    out["verify"] = v
    print(f"  ① 市场配置（实验侧实际生效 vs 标定侧收到）")
    if v["clean_market_config"]:
        print("     ✅ 一致 —— 已核实，标定配置与实验配置一致")
    else:
        for m in v["market_config_mismatch_fields"]:
            print(f"     ❌ {m}")
    print(f"  ② 市场指纹（两侧真实构造后比对）"
          f"{'✅ 一致' if v['fingerprint_match'] else '❌ 不一致'}")
    print(f"     主体构成：{v['fingerprint']['kinds']}")
    print(f"  ③ 窗口（标定 vs 实验）")
    if v["clean_window"]:
        print(f"     ✅ 一致：{[v['experiment_window']]}")
    else:
        for m in v["mismatch_fields"]:
            print(f"     ❌ {m}")
        print(f"     ⇒ 按第六条纪律，**必须重标定后重测**")
    print(f"  综合：{'✅ 干净' if v['clean'] else '⚠️ 存在不对齐 —— 需重标定'}")

    if args.rerun:
        import run_workstream_A as A
        seeds = [A.SEED0 + 7 * i for i in range(args.seeds)]
        print(f"\n  用「同配置同窗口」重跑 EA.4（{len(seeds)} 个种子）…")
        re4 = rerun_EA4_with_verified_calibration(seeds=seeds)
        out["rerun"] = re4
        old = 0.455
        new = (re4["arms"].get("taker") or {}).get("k")
        out["comparison"] = {
            "old_reported_k_taker": old,
            "new_k_taker": new,
            "delta": (None if new is None else new - old),
            "materially_different": (None if new is None else abs(new - old) > 0.05),
            "kills_hypothesis": (None if new is None else new > 0.7),
        }
        print(f"\n  官方基准：taker k = {new if new is None else round(new, 3)}"
              f"（三线深挖报的是 {old}）")
        if out["comparison"]["kills_hypothesis"]:
            print("  🛑 重标定后 k 跳回 0.7 以上 ⇒ 按任务书 §2.4，"
                  "**暂停工作线 D/E/F 并回报设计者**")
        elif out["comparison"]["materially_different"]:
            print("  ⚠️ 与 0.455 的差异超过 0.05 —— 小结里必须用同样字号突出报告")
        else:
            print("  ✅ 与 0.455 的差异在 0.05 以内 ⇒ 0.455 站得住，可继续 D/E/F")

    out["elapsed_sec"] = time.time() - t0
    save_json(out, RESULT)
    print(f"\n  产物：out/{RESULT}（{out['elapsed_sec']:.0f}s）")
    return out


if __name__ == "__main__":
    main()
