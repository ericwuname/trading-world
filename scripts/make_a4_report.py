"""A4 报告生成器 —— 从评估 JSON 现算，不手抄任何数字。

⚠️ **为什么必须现算**：本项目最贵的一条教训是
「报告里的数字一律从 ``out/*.json`` 现算，不许手抄」——
手抄的数字与产物脱节时**没人会知道**，而报告看起来完全正常。

用法::

    python scripts/make_a4_report.py            # 读 out/a4/eval_*.json
    python scripts/make_a4_report.py --open     # 生成后打印路径
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "docs" / "A4-分层评估报告.md"


def _pct(x: float, digits: int = 2) -> str:
    if x is None or x != x:
        return "—"
    return f"{x * 100:.{digits}f}%"


def _num(x: float, digits: int = 2) -> str:
    if x is None or x != x:
        return "—"
    return f"{x:,.{digits}f}"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _layer_tables(evals: dict[str, dict]) -> list[str]:
    L: list[str] = []
    L.append("## 2. 结果层（毛 / 净）\n")
    L.append("| 配置 | 决策数 | 交易数 | 回合 | 毛收益 | 净收益 | 手续费 | 滑点 | "
             "换手(×权益) | 最大回撤 | 夏普 | 平均持仓(根) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for nm, e in evals.items():
        r = e["result"]
        L.append(
            f"| `{nm}` | {e['decision'].get('n', 0)} | {r['n_trades']} | "
            f"{r['n_round_trips']} | {_pct(r['gross_return'])} | "
            f"{_pct(r['net_return'])} | {_num(r['fees_paid'])} | "
            f"{_num(r['slippage_cost'])} | {_num(r['turnover_x_equity'])}× | "
            f"{_pct(r['max_drawdown'])} | {_num(r['sharpe'])} | "
            f"{_num(r['avg_holding_bars'], 1)} |")
    L.append("")
    L.append("⚠️ **毛收益的口径**：净收益 + 手续费 + 滑点 = "
             "「假设**不付任何交易成本**」。账本里的权益**已经扣过手续费**，"
             "所以「净收益」才是有成本的那个。\n")
    return L


def _decision_table(evals: dict[str, dict]) -> list[str]:
    L: list[str] = []
    L.append("## 3. 决策层\n")
    L.append("| 配置 | 弃权率 | 解析失败率 | 被拒率 | 被改量率 | "
             "置信度均值 | 理由可用率 | 外部引用率 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for nm, e in evals.items():
        d = e["decision"]
        L.append(
            f"| `{nm}` | {_pct(d.get('abstain_frac'))} | "
            f"{_pct(d.get('parse_fail_frac'))} | {_pct(d.get('rejected_frac'))} | "
            f"{_pct(d.get('resized_frac'))} | {_num(d.get('confidence_mean'))} | "
            f"{_pct(d.get('reason_usable_frac'))} | "
            f"{_pct(d.get('external_ref_frac'))} |")
    L.append("")
    hits: dict[str, int] = {}
    for e in evals.values():
        for w, c in (e["decision"].get("external_ref_terms") or {}).items():
            hits[w] = hits.get(w, 0) + c
    if hits:
        L.append("**扫到的外部引用词**（关键词启发式，用于把可疑样本捞出来给人看）："
                 + "、".join(f"`{w}`×{c}" for w, c in
                             sorted(hits.items(), key=lambda kv: -kv[1])))
        L.append("")
    return L


def _exec_table(evals: dict[str, dict]) -> list[str]:
    L: list[str] = []
    L.append("## 4. 执行层\n")
    L.append("⚠️ **「订单成交」与「TP/SL 出场」必须分开看**：TP/SL 是挂在仓位上的"
             "触发单，**没有对应的已提交订单**。把它们算进同一列的分子会让"
             "成交率**超过 100%**（真机报告里出现过 142%：19 张订单 27 笔成交）。"
             "超过 100% 看起来只是「数字有点怪」，但它说明分子分母不同源。\n")
    L.append("| 配置 | 落成订单 | 订单成交 | 订单成交率 | TP/SL 出场 | "
             "强平 | 未成交决策 | 滑点均值(bp) | 滑点成本 | 过期挂单 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for nm, e in evals.items():
        x = e["execution"]
        L.append(
            f"| `{nm}` | {x['n_orders_built']} | {x.get('n_order_fills', '—')} | "
            f"{_pct(x['fill_rate_of_orders'])} | "
            f"{x.get('n_exit_fills_tp_sl', '—')} | "
            f"{x.get('n_liquidate_fills', 0)} | {x['n_decisions_not_traded']} | "
            f"{_num(x['slippage_bps_mean'], 3)} | "
            f"{_num(x['slippage_cost_total'])} | {x['n_expired_orders']} |")
    L.append("")
    return L


def _cost_section(evals: dict[str, dict]) -> list[str]:
    L: list[str] = []
    L.append("## 5. 成本敏感性\n")
    L.append("### 5.1 重跑版（**这是正式结论**）\n")
    L.append("用**同一批已录制的决策**换成本重跑——不是把旧收益减一个数。\n")
    L.append("⚠️ **标 `不可测` 的档位数字无效**：回放覆盖率过低 ⇒ "
             "prompt 含账户状态 ⇒ 换成本后路径发散 ⇒ Agent 的弃权是"
             "**回放失败**造成的，不是策略行为。"
             "要测那一档必须真的重跑（花额度）。\n")
    L.append("| 配置 | 倍数 | 净收益 | 交易数 | 回放覆盖率 | 有效 | 最大回撤 | 强平 |")
    L.append("|---|---|---|---|---|---|---|---|")
    any_rerun = False
    for nm, e in evals.items():
        for c in (e.get("cost_sensitivity_rerun") or []):
            any_rerun = True
            cov = c.get("replay_coverage")
            covs = "—" if cov is None else f"{cov * 100:.1f}%"
            valid = c.get("valid", True)
            net = _pct(c["net_return"]) if valid else "**不可测**"
            L.append(f"| `{nm}` | ×{c['cost_multiplier']:g} | {net} | "
                     f"{c['n_trades']} | {covs} | "
                     f"{'✅' if valid else '❌'} | "
                     f"{_pct(c['max_drawdown'])} | {c['n_liquidations']} |")
    if not any_rerun:
        L.append("| （无） | | | | | | | |")
    L.append("")
    for nm, e in evals.items():
        for c in (e.get("cost_sensitivity_rerun") or []):
            if not c.get("valid", True):
                L.append(f"- `{nm}` ×{c['cost_multiplier']:g}：{c.get('invalid_reason', '')}")
    L.append("")
    L.append("### 5.2 一阶近似（仅供参考）\n")
    L.append("净收益 −（原成本 ×(m−1)）。**它假设「成本变了、交易行为不变」**，"
             "而真实情况不是——所以与 5.1 的差值本身就是信息："
             "**差得越多，说明该策略对成本越敏感**。\n")
    L.append("| 配置 | ×1 | ×2 | ×5 |")
    L.append("|---|---|---|---|")
    for nm, e in evals.items():
        cs = {c["cost_multiplier"]: c for c in e["cost_sensitivity_analytic"]}
        cells = " | ".join(_pct(cs[m]["net_return"]) for m in (1.0, 2.0, 5.0))
        L.append(f"| `{nm}` | {cells} |")
    L.append("")
    return L


def _comparison_section(cmp: dict) -> list[str]:
    L: list[str] = []
    L.append("## 6. 与基线的对照（区间重叠检验）\n")
    b = cmp["bootstrap"]
    L.append(f"度量：`{cmp['metric']}`（每根 K 线的平均收益率）。"
             f"块自助法 {b['n_boot']} 次、块长 {b['block']}、种子 {b['seed']}。\n")
    lo, hi = cmp["target_ci"]
    L.append(f"**目标** `{cmp['target']}`：点估计 "
             f"{cmp['target_point'] * 1e4:+.4f} bp/根，"
             f"95% 区间 [{lo * 1e4:+.4f}, {hi * 1e4:+.4f}]，"
             f"n={cmp['n_bars']} 根。\n")
    L.append("| 对照基线 | 判据 | 判定 | 重叠比例 | 说明 |")
    L.append("|---|---|---|---|---|")
    test_zh = {"ci_overlap": "区间重叠", "ci_excludes_baseline": "CI 是否排除基准值"}
    for v in cmp["verdicts"]:
        ov = v.get("overlap_fraction")
        ovs = "—" if (ov is None or ov != ov) else f"{ov * 100:.1f}%"
        L.append(f"| `{v['name_b']}` | {test_zh.get(v.get('test', ''), v.get('test', ''))} | "
                 f"**{v['verdict']}** | {ovs} | {v.get('reason', '')} |")
    L.append("")
    L.append("⚠️ **两种对照用两种检验**（本项目纪律 8："
             "「单家族自己的方向判定」与「家族之间的比较判定」是两件事）：")
    L.append("")
    L.append("| 基线的情况 | 用什么检验 |")
    L.append("|---|---|")
    L.append("| **零方差**（`noop` 不交易 ⇒ 收益恒为 0） | "
             "**CI 是否排除基准值**——问「目标的区间排不排除 0」 |")
    L.append("| 有方差（其它交易型基线） | "
             "**区间重叠**——问「两个区间是不是分开」 |")
    L.append("")
    L.append("⚠️ 第一行是**踩出来的**：最初一律走重叠检验，于是 vs `noop` 的输出是"
             "「无法判定（输入退化）」——因为 noop 的区间宽度是 0。"
             "而「能不能打赢 noop」**恰恰是 A4 最核心的那个问题**。"
             "用错检验会让最关键的问题**答不出来**，而且看起来像「数据不够」。\n")
    return L


def _calib_section(evals: dict[str, dict]) -> list[str]:
    L: list[str] = []
    L.append("## 7. 置信度校准（说 0.7 的是不是真 70% 对）\n")
    any_cal = False
    for nm, e in evals.items():
        cal = e.get("confidence_calibration")
        if not cal or not cal.get("n"):
            continue
        any_cal = True
        L.append(f"**`{nm}`**（n={cal['n']}，前瞻 {cal['horizon_bars']} 根）："
                 f"整体预测均值 {cal['overall_predicted']:.3f}，"
                 f"实际方向准确率 {cal['overall_realized']:.3f}。")
        L.append("")
        L.append("| 置信度区间 | n | 预测均值 | 实际准确率 | 差(实际−预测) |")
        L.append("|---|---|---|---|---|")
        for b in cal["bins"]:
            if not b["n"]:
                L.append(f"| [{b['lo']:.1f}, {b['hi']:.1f}) | 0 | — | — | — |")
                continue
            L.append(f"| [{b['lo']:.1f}, {b['hi']:.1f}) | {b['n']} | "
                     f"{b['predicted_mean']:.3f} | {b['realized']:.3f} | "
                     f"{b['gap']:+.3f} |")
        L.append("")
    if not any_cal:
        L.append("（本次没有可校准的样本）")
        L.append("")
    L.append(f"⚠️ 这是**方向准确率**，不含成本；样本小则每箱不稳定，"
             f"请连同 n 一起读。**弃权不参与校准**（弃权没有方向，无法判对错）。\n")
    return L


def main() -> int:
    ap = argparse.ArgumentParser(description="生成 A4 分层评估报告")
    ap.add_argument("--dir", default=str(ROOT / "out" / "a4"))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    d = Path(args.dir)
    files = sorted(d.glob("eval_*.json"))
    if not files:
        raise SystemExit(f"{d} 下没有 eval_*.json；先跑 scripts/agent_eval.py")

    docs = {f.stem.replace("eval_", ""): _load(f) for f in files}
    L: list[str] = []
    L.append("# A4 分层评估报告（LLM 交易 Agent vs 规则基线 vs noop）\n")
    L.append("> ⚠️ 本报告的**每一个数字都从 `out/a4/eval_*.json` 现算**，"
             "不手抄。生成器：`scripts/make_a4_report.py`。\n")

    for inst, doc in docs.items():
        evals = doc["evals"]
        cmp = doc["comparison"]
        cfg = doc["config"]
        L.append(f"\n---\n\n# 标的一：{inst}\n")
        L.append(f"运行参数：`{cfg['n']}` 根 K 线、每根决策一次、"
                 f"模板 `{cfg.get('template')}`、采样 {cfg.get('samples')} 次、"
                 f"初始权益 {cfg.get('equity'):,.0f}、杠杆 {cfg.get('lever')}x、"
                 f"滑点 {cfg.get('slippage_bps')}bp。\n")
        exec_cfg = next(iter(evals.values()))["exec_config"]
        L.append(f"成本假设（**事先定义**，见设计方案 §2）："
                 f"taker `{exec_cfg.get('taker_fee')}`、"
                 f"maker `{exec_cfg.get('maker_fee')}`、"
                 f"滑点 `{exec_cfg.get('slippage_bps')}`bp、"
                 f"限价单最多等 `{exec_cfg.get('max_wait_bars')}` 根。\n")
        L.append("## 1. 对照阵容\n")
        L.append("| 配置 | 类型 | 说明 |")
        L.append("|---|---|---|")
        for nm, e in evals.items():
            meta = e.get("meta", {})
            mode = meta.get("mode", "")
            kind = ("LLM" if mode in ("live", "replay_nonet")
                    else "规则基线" if mode == "baseline_rule" else "—")
            note = (f"模板 {meta.get('template')}，采样 {meta.get('n_samples')}"
                    if kind == "LLM" else
                    f"规则 `{meta.get('policy')}`（**读同一份可见状态，不问 LLM**）")
            L.append(f"| `{nm}` | {kind} | {note} |")
        L.append("")
        L += _layer_tables(evals)
        L += _decision_table(evals)
        L += _exec_table(evals)
        L += _cost_section(evals)
        L += _comparison_section(cmp)
        L += _calib_section(evals)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"报告已生成：{out}（{len(L)} 行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
