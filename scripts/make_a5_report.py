"""A5 报告生成器 —— **大样本对照 + v4「它选了哪条路」**。

⚠️ 与 `make_a4_report.py` 同一条纪律：**所有数字从 JSON 现算，不手抄**。
手抄的数字与产物脱节时没人会知道，而报告看起来完全正常。

用法::

    python scripts/make_a5_report.py
    python scripts/make_a5_report.py --dirs out/a5,out/a5v4 --out docs/A5-对照报告.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "docs" / "A5-对照报告.md"


def _pct(x, d: int = 3) -> str:
    if x is None or x != x:
        return "—"
    return f"{x * 100:.{d}f}%"


def _num(x, d: int = 2) -> str:
    if x is None or x != x:
        return "—"
    return f"{x:,.{d}f}"


def _load_json(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def collect(dirs: list[Path]) -> list[dict]:
    """把所有 eval_*.json 收成一个列表（每条带来源文件名）。"""
    out = []
    for d in dirs:
        if not d.is_dir():
            continue
        for f in sorted(d.glob("eval_*.json")):
            doc = _load_json(f)
            if doc and doc.get("evals"):
                out.append({"file": f.name, "dir": str(d.name), "doc": doc})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="A5 对照报告")
    ap.add_argument("--dirs", default="out/a5,out/a5v4")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    ds = [ROOT / x.strip() for x in args.dirs.split(",") if x.strip()]
    docs = collect(ds)
    if not docs:
        raise SystemExit(f"{args.dirs} 下没有 eval_*.json")

    L: list[str] = []
    L.append("# A5 对照报告：模板 × 标的 的完整对照（2026-09-21）\n")
    L.append("> ⚠️ **每个数字都从 `out/*/eval_*.json` 现算**，不手抄。"
             "生成器：`scripts/make_a5_report.py`。\n")

    # ---- 1. 主表：按 (模板, 标的) 分组 ----
    L.append("## 1. 结果层总表\n")
    L.append("| 来源 | 配置 | 决策数 | 交易数 | 毛收益 | **净收益** | 最大回撤 | "
             "夏普 | 换手(×权益) | 弃权率 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for d in docs:
        for nm, e in d["doc"]["evals"].items():
            r, dec = e["result"], e["decision"]
            L.append(
                f"| `{d['file']}` | `{nm}` | {dec.get('n', 0)} | {r['n_trades']} | "
                f"{_pct(r['gross_return'])} | **{_pct(r['net_return'])}** | "
                f"{_pct(r['max_drawdown'], 2)} | {_num(r['sharpe'])} | "
                f"{_num(r['turnover_x_equity'])}× | "
                f"{_pct(dec.get('abstain_frac'), 1)} |")
    L.append("")

    # ---- 2. 每个运行的对照判定 ----
    L.append("## 2. 与基线的对照（区间重叠检验）\n")
    L.append("⚠️ 零方差基线（`noop` 收益恒 0）改问「**CI 是否排除基准值**」；"
             "其余走区间重叠。**判定为「依然无法判定」不等于「没有差别」**，"
             "它只说明这个样本量分不出来——表里给了要分开大约需要多少样本。\n")
    for d in docs:
        cmp = d["doc"].get("comparison") or {}
        if not cmp:
            continue
        lo, hi = cmp["target_ci"]
        L.append(f"### `{d['file']}` —— 目标 `{cmp['target']}`\n")
        L.append(f"点估计 {cmp['target_point'] * 1e4:+.4f} bp/根，"
                 f"95% 区间 [{lo * 1e4:+.4f}, {hi * 1e4:+.4f}]，"
                 f"n={cmp['n_bars']} 根。\n")
        L.append("| 对照基线 | 判据 | 判定 | 重叠 | 要分开约需样本 | 说明 |")
        L.append("|---|---|---|---|---|---|")
        tz = {"ci_overlap": "区间重叠", "ci_excludes_baseline": "CI 排除基准值"}
        for v in cmp.get("verdicts", []):
            ov = v.get("overlap_fraction")
            req = v.get("required_n")
            L.append(
                f"| `{v['name_b']}` | {tz.get(v.get('test', ''), '')} | "
                f"**{v['verdict']}** | "
                f"{'—' if (ov is None or ov != ov) else f'{ov * 100:.1f}%'} | "
                f"{'—' if not req else f'{req:,.0f}'} | "
                f"{(v.get('reason') or '')[:110]} |")
        L.append("")

    # ---- 3. 决策依据自报（v4）----
    L.append("## 3. ⭐「它选了哪条路」（v4 自报，2026-09-21 用户追加的设计）\n")
    any_basis = False
    for d in docs:
        for nm, e in d["doc"]["evals"].items():
            b = (e.get("decision") or {}).get("basis") or {}
            if not b.get("n_declared"):
                continue
            any_basis = True
            L.append(f"**`{d['file']}` / `{nm}`**："
                     f"共 {b['n']} 条决策，其中 **{b['n_declared']} 条有自报**。\n")
            L.append("| 依据 | 条数 | 占有自报的比例 |")
            L.append("|---|---|---|")
            labels = b.get("labels") or {}
            for k, v in sorted((b.get("counts") or {}).items(),
                               key=lambda kv: -kv[1]):
                if not k:
                    continue
                L.append(f"| `{k}`（{labels.get(k, '')}） | {v} | "
                         f"{_pct((b.get('frac_of_declared') or {}).get(k), 1)} |")
            L.append("")
    if not any_basis:
        L.append("（本次没有带 `basis` 自报的运行——v1~v3 本来就没有这个字段。）\n")
    L.append("> ⚠️ **这是模型自报的声明，不是测出来的证据**：它可能为了"
             "显得一致而都写 `both`。要交叉验证，得看两类在**实际行为**上的"
             "差异（换手率、与 momentum 信号的一致率、置信度分布）。\n")

    # ---- 4. 决策/执行层 ----
    L.append("## 4. 决策层与执行层\n")
    L.append("| 来源 | 配置 | 弃权率 | 解析失败 | 被拒率 | 被改量率 | "
             "置信度均值 | 外部引用率 | 订单成交率 | TP/SL 出场 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for d in docs:
        for nm, e in d["doc"]["evals"].items():
            dec, x = e["decision"], e["execution"]
            L.append(
                f"| `{d['file']}` | `{nm}` | {_pct(dec.get('abstain_frac'), 1)} | "
                f"{_pct(dec.get('parse_fail_frac'), 1)} | "
                f"{_pct(dec.get('rejected_frac'), 1)} | "
                f"{_pct(dec.get('resized_frac'), 1)} | "
                f"{_num(dec.get('confidence_mean'))} | "
                f"{_pct(dec.get('external_ref_frac'), 1)} | "
                f"{_pct(x.get('fill_rate_of_orders'), 1)} | "
                f"{x.get('n_exit_fills_tp_sl', 0)} |")
    L.append("")

    # ---- 5. 成本敏感性 ----
    L.append("## 5. 成本敏感性（重跑版；`不可测` = 回放覆盖率过低，数字无效）\n")
    L.append("| 来源 | 配置 | ×1 | ×2 | ×5 |")
    L.append("|---|---|---|---|---|")
    for d in docs:
        for nm, e in d["doc"]["evals"].items():
            rr = {c["cost_multiplier"]: c
                  for c in (e.get("cost_sensitivity_rerun") or [])}
            cells = []
            for m in (1.0, 2.0, 5.0):
                c = rr.get(m)
                if c is None:
                    cells.append("—")
                elif not c.get("valid", True):
                    cov = c.get("replay_coverage")
                    cells.append(f"**不可测**" + (f"({cov:.0%})" if cov else ""))
                else:
                    cells.append(_pct(c["net_return"]))
            L.append(f"| `{d['file']}` | `{nm}` | " + " | ".join(cells) + " |")
    L.append("")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"报告已生成：{out}（{len(L)} 行，{len(docs)} 个运行文件）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
