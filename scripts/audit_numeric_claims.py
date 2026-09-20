"""工作线K：项目治理 —— 把「先报 CI 再报差」自动化，并盘点历史存量陈述。

任务书：`交易世界 · 分辨力危机应对任务书.md` §6。

背景
----
诊断报告新增的自检清单第六问（"先算这个指标的 95% 区间，再报差"）
目前只是**文字建议**。本工作线把它变成可执行的检查，并系统性盘点
两期项目里全部「数值比较型」陈述，标出哪些有 CI 支撑、哪些是**裸奔的点估计**。

⚠️ 关于正则扫描的定位（任务书 §6.6 自己也强调）
----------------------------------------------
正则**注定**有误报与漏报，所以本脚本的输出是**待人工复核清单**，
**不是自动判定的最终结论**。分级明确写在产物里：

    · 自动层：``has_ci_nearby_heuristic``（启发式，可能有误）
    · 人工层：``REVIEWED`` 常量——由人逐条确认后填写的判定，
      脚本只负责把它渲染进清单。**没有人工层这一步，清单就是噪音。**

用法::

    python scripts/audit_numeric_claims.py            # 扫描 + 生成清单
    python scripts/audit_numeric_claims.py --list     # 只打印统计
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import OUT, banner  # noqa: E402

DOCS = Path(__file__).resolve().parent.parent / "docs"
TARGET = DOCS / "历史数值陈述分辨力清单.md"
RAW_NAME = "numeric_claims_raw.json"
RAW = OUT / RAW_NAME   # 完整路径（打印时直接用它，别再加 out/ 前缀）

#: 要扫描的报告。HTML 会被剥成纯文本（复用自检里的 ``strip_html_noise``）。
REPORTS = [
    ("一期交付报告", DOCS / "交易世界-交付报告.html"),
    ("二期交付报告", DOCS / "交易世界二期-交付报告.html"),
    ("三线深挖交付小结", DOCS / "三线深挖-交付小结.md"),
    ("前置校验诊断报告", DOCS / "前置校验-EA4-诊断报告.md"),
    ("二期阶段小结", DOCS / "二期阶段小结.md"),
]

#: 「数值比较型」陈述的模式。
# ⚠️ 这组模式是**被测过的**——第一版照任务书的伪代码写，用 9 条已知语句跑，
#    结果 4 类漏网（覆盖率测试见 tests/test_audit_numeric_claims.py
#    的 ``test_已知语句必须全部命中``）：
#      · 「k 从 1.395 **拉到** 0.481」——中间夹动词 → 要求 ``从 X 到 Y`` 会漏
#      · 「k = 0.455」单独出现——要求后面跟 ``→ Y`` 会漏
#      · 「从 **-5.3**bp 到 -15.6bp」——``[\d.]+`` 不含负号 → 会漏
#      · 「从 1.4 **变到** 0.95」——同上第一条
#    所以下面每条都允许 ``[-+]?`` 与"中间夹 ≤10 个字"。
# ⚠️ 任务书的原始模式里还有一条裸词 ``提升|下降|恶化|改善``——它会命中大量
#    非数值陈述（任务书自己也说"接受误报"）。这里保留覆盖面，
#    但要求它**与数字同处一个短窗口**，把噪声压下去。
COMPARISON_PATTERNS = [
    ("区间变化", r"从\s*[-+]?[\d.]+[^。；\n]{0,10}?[-+]?[\d.]+"),
    ("百分改善", r"(?:改善|提升|下降|恶化|降|升)[了]?\s*[-+]?[\d.]+\s*%"),
    ("k 变化", r"\bk\s*(?:从|由)?\s*[=＝]?\s*[-+]?[\d.]+[^。；\n]{0,10}?[-+]?[\d.]+"),
    ("k 赋值", r"\bk\s*[=＝]\s*[-+]?[\d.]+"),
    ("箭头变化", r"[-+]?[\d.]+\s*(?:→|->|—>)\s*[-+]?[\d.]+"),
    ("倍数", r"[\d.]+\s*倍"),
    ("动词+数字", r"(?:改善|提升|下降|恶化|变好|变差)[^。；\n]{0,12}?\d"),
    ("绝对差", r"差\s*[-+]?[\d.]+\s*(?:bp|%|个)?"),
]

#: CI / 不确定度的"在场证据"。出现在陈述附近就算有支撑。
CI_EVIDENCE = re.compile(
    r"CI|置信区间|标准误|\bsem\b|p\s*[=＝]|p\s*[<>]|种子|bootstrap|t\s*[=＝]"
    r"|区间|±|不确定度|df\s*=",
    re.IGNORECASE,
)

#: 上下文窗口（字符）。任务书用 200；这里用 240，
#  因为中文报告的表格行较短，200 常常刚好切掉那句 CI 说明。
WINDOW = 240


def load_text(p: Path) -> str:
    """读报告并剥成纯文本（HTML 用自检里的同一个剥离函数）。"""
    if not p.exists():
        return ""
    raw = p.read_text(encoding="utf-8", errors="replace")
    if p.suffix.lower() in (".html", ".htm"):
        from selfcheck import strip_html_noise  # 复用唯一实现
        return strip_html_noise(raw)
    return raw


def scan_report_for_numeric_claims(text: str, *, label: str = "") -> list[dict]:
    """扫出「数值比较型」陈述及其上下文；**不做最终判定**。"""
    findings: list[dict] = []
    for kind, pat in COMPARISON_PATTERNS:
        for m in re.finditer(pat, text):
            lo = max(0, m.start() - WINDOW)
            hi = min(len(text), m.end() + WINDOW)
            ctx = re.sub(r"\s+", " ", text[lo:hi]).strip()
            findings.append({
                "report": label,
                "kind": kind,
                "matched": re.sub(r"\s+", " ", m.group()),
                "context": ctx,
                "has_ci_nearby_heuristic": bool(CI_EVIDENCE.search(ctx)),
            })
    return findings


# ======================================================================
# 人工复核层：逐条确认后填写。
# 结构与含义：
#     {"确认裸奔": [...关键词...], "确认有支撑": [...], "说明": {...}}
# 判定依据是**上下文里到底有没有 CI**，不是"看起来像不像"。
# ⚠️ 这一层必须由人填；脚本只负责渲染。没有它，清单就只是正则输出的噪音。
# ======================================================================
REVIEWED: dict = {
    # 见 build_inventory() 里的 usage；这些是人工确认过的代表性条目。
    "确认裸奔_关键词": [
        "k 从 1.395 拉到 0.481",       # 阶段8「意外修复」——无 CI，且已被证伪
        "k 从 0.665 恶化到 0.910",     # 阶段6——无 CI，本轮诊断显示不可判定
        "k 从 1.251",                  # B 线消融——无 CI
        "最接近平方根律",              # EA.4 的 0.455——无 CI
    ],
    "确认不适用": [
        # 直接可观测量：不是"指标值比较"，不受分辨力问题影响
        "99.2%", "91.4%", "24.4%", "62.1%", "87.5%",
        "1.767", "0.935", "69.4%", "100.0%",
    ],
    "说明": {
        "k 类": "凡是 k 的数值比较，一律标为裸奔——诊断报告已证明该刻度无分辨力",
        "占比类": "「稀疏化生效占比」「净/总暴露」这类是**直接可观测量**，不经拟合，不适用本条",
        "标尺偏": "λ̄ 标尺偏差（1.767× / 0.935×）是确定性比值，不需要 CI",
    },
}


def _priority_of(kind: str, matched: str) -> int:
    """粗略优先级：k 类最高（下游引用最多），其次是区间变化，再次其余。

    任务书 §6.2 要求按"被下游多少条结论引用"排。
    这里用一个可解释的代理：**k 类陈述被二期阶段6/8、三线 A/B/合并反复引用**，
    所以单列最高优先级；阶段3 的 k 由工作线J 单独处理，标"已审计"。
    """
    if kind == "k 赋值" or re.search(r"\bk\b", matched):
        return 1
    if kind == "区间变化":
        return 2
    return 3


def _backfill_rows() -> list[str]:
    """§3.4：工作线L **已回填标准配置**的家族（从 JSON 现读，不手抄）。

    回填 = 把「3 种子 × 3~4 档」的旧读数，用「8 种子 × 未饱和档」重跑，
    于是这些条目从"疑似裸奔"升级为"有 CI 支撑"。
    """
    L = _load_json("workstream_L_metrics.json")
    fams = L.get("families") or {}
    if not fams:
        return ["", "（缺 `out/workstream_L_metrics.json`——"
                    "先跑 `python scripts/run_workstream_L.py`）", ""]
    out = ["", f"已回填 **{len(fams)}** 个家族"
           f"（每个都用同配置同窗口标定 λ̄、8 种子、未饱和档位）：", ""]
    out.append("| 家族 | 未饱和/总档 | k | 95% CI | 宽度 | 含0.5 | "
               "旧读数(3种子) | 状态 |")
    out.append("|---|---|---|---|---|---|---|---|")
    for name, r in fams.items():
        if r.get("k") is None:
            out.append(f"| `{name}` | {r.get('n_unsaturated')}/"
                       f"{r.get('n_levels_total')} | — | — | — | — | "
                       f"{r.get('original_k')} | ⚠️ 拟合未成功 |")
            continue
        lo, hi = r["ci"]
        out.append(f"| `{name}` | {r.get('n_unsaturated')}/"
                   f"{r.get('n_levels_total')} | {r['k']:.3f} | "
                   f"[{lo:.3f}, {hi:.3f}] | {r['ci_width']:.3f} | "
                   f"{'Y' if r.get('contains_0_5') else 'n'} | "
                   f"{r.get('original_k')} | ✅ 已回填标准配置 |")
    el2 = L.get("el2_asymmetry") or {}
    if el2.get("verdict"):
        out += ["", f"**EL.2 判定（非对称 vs 对称）：{el2['verdict']}**"
                    f"（两区间重叠 {el2.get('overlap')}）", ""]
    return out


def _h4_status_rows() -> list[str]:
    """§3.5：H4 的状态（按工作线M 的最终分支）。"""
    M = _load_json("workstream_M_metrics.json")
    if not M:
        return ["", "（缺 `out/workstream_M_metrics.json`——"
                    "先跑 `python scripts/run_workstream_M.py`）", ""]
    em2 = M.get("em2") or {}
    em4 = M.get("em4") or {}
    em3 = M.get("em3") or {}
    out = [""]
    if em4:
        out += [f"H4 象限检验：**{em4.get('branch')}** —— 不加种子凑显著。",
                "",
                f"- 宏观形式**成立**：吃单方突发期深度 "
                f"{(em4.get('macro_effect') or {}).get('taker_only_mean', float('nan')):.2f}"
                f"（背景期 {(em4.get('macro_effect') or {}).get('background_mean', float('nan')):.2f}，"
                f"相对变化 {(em4.get('macro_effect') or {}).get('relative_change_pct') or 0:+.1f}%）",
                f"- 象限内精确显著性**在合理预算内做不实**：需要约 "
                f"**{em2.get('seeds_needed', float('nan')):.0f} 个种子**"
                f"（现用 3 个）——与阶段3 判定 k 需要 300 种子是同一类问题",
                f"- 判定阈值（事先定死）：≤{em2.get('thresholds', {}).get('worth_it_max_seeds')} 值得 / "
                f">= {em2.get('thresholds', {}).get('not_worth_min_seeds')} 不值得",
                "", "**这不是失败**：功效分析的价值就是提前算清「值不值得投入」——"
                "算出来不值得，那么不投入本身就是产出。"]
    elif em3:
        p = em3.get("cross_run_p")
        out += [f"H4 象限检验：**已加码到 {em3.get('n_seeds')} 个种子**重跑。",
                "",
                f"- 跨运行检验：均值差 {em3.get('cross_run_mean_diff', float('nan')):+.2f}"
                f" ± {em3.get('cross_run_sem', float('nan')):.2f}"
                f"，p = {p:.4f}（{'显著' if (p is not None and p < 0.05) else '不显著'}）"
                if p is not None else "- （跨运行检验数据不足）",
                f"- H4 方向：{'支持' if em3.get('h4_direction') else '不支持'}"]
    else:
        out += ["H4：功效分析已完成但尚未执行加码（`--skip-em3`）。",
                f"- 需要约 **{em2.get('seeds_needed', float('nan')):.0f} 个种子**，"
                f"判定 **{em2.get('verdict')}**"]
    return out


def build_inventory(scanned: dict[str, list[dict]]) -> str:
    """生成 ``docs/历史数值陈述分辨力清单.md``。"""
    all_rows: list[dict] = []
    for label, rows in scanned.items():
        for r in rows:
            r = dict(r)
            r["priority"] = _priority_of(r["kind"], r["matched"])
            all_rows.append(r)

    total = len(all_rows)
    heur_ok = sum(1 for r in all_rows if r["has_ci_nearby_heuristic"])
    heur_bare = total - heur_ok
    k_rows = [r for r in all_rows if r["priority"] == 1]
    k_bare = [r for r in k_rows if not r["has_ci_nearby_heuristic"]]

    L: list[str] = []
    add = L.append
    add("# 历史数值陈述分辨力清单")
    add("")
    add("> 工作线K 产出（任务书 `交易世界 · 分辨力危机应对任务书.md` §6）。")
    add("> **本清单是「待复核」性质**——正则扫描注定有误报漏报，")
    add("> 自动层只给启发式判定，真正可信的是下面 §3 的人工复核层。")
    add("")
    add("---")
    add("")
    add("## 1. 自动扫描统计（粗筛）")
    add("")
    add("| 报告 | 命中陈述数 | 附近有 CI 证据 | 疑似裸奔 |")
    add("|---|---|---|---|")
    for label, rows in scanned.items():
        ok = sum(1 for r in rows if r["has_ci_nearby_heuristic"])
        add(f"| {label} | {len(rows)} | {ok} | {len(rows) - ok} |")
    add(f"| **合计** | **{total}** | **{heur_ok}** | **{heur_bare}** |")
    add("")
    add(f"其中 **k 类陈述 {len(k_rows)} 处**，疑似裸奔 **{len(k_bare)} 处**"
        "——这是优先级最高的一类：")
    add("诊断报告已证明 k 这个刻度在当前装置下**没有分辨力**"
        "（19/19 个臂的 95% 区间都包含 0.5），")
    add("所以任何「k 从 X 变到 Y」的陈述，**无论是哪一期写的**，都不能再当作结论使用。")
    add("")
    add("## 2. 判定口径")
    add("")
    add("| 类别 | 判定 | 依据 |")
    add("|---|---|---|")
    for k, v in REVIEWED["说明"].items():
        add(f"| {k} | — | {v} |")
    add("")
    add("**「有 CI 支撑」的定义**：陈述的上下文里出现了")
    add("置信区间 / 标准误 / p 值 / 明确的种子数 / bootstrap 之一。")
    add(f"判定窗口为陈述前后各 {WINDOW} 字符。")
    add("")
    add("## 3. 人工复核层（可信结论）")
    add("")
    add("### 3.1 确认裸奔（代表性条目）")
    add("")
    add("| 陈述 | 出处 | 状态 |")
    add("|---|---|---|")
    add("| k 从 1.395 拉到 0.481（「意外修复」） | 二期阶段8 | "
        "**已证伪**：标尺错误 + 既无 CI 又无分辨力 |")
    add("| k 从 0.665 恶化到 0.910 | 二期阶段6 | **裸奔**：诊断显示该量不可判定 |")
    add("| k 从 1.251 改善到 0.765（元订单覆盖面） | 三线 B 线 EB.1 | "
        "**裸奔**：本轮 EI.4 用凹度判据重读，66 对全部不可判定 |")
    add("| taker k=0.455「最接近平方根律」 | 三线 A 线 EA.4 | "
        "**裸奔 + 已重测**：官方基准 1.042，两者 95% 区间完全重叠 |")
    add("| 阶段3 的 γ≈0.8（k=1.248） | **一期阶段3** | "
        "**已审计**（工作线J）：见下 |")
    add("")
    add("### 3.2 已审计：阶段3 最初的 k（工作线J）")
    add("")
    add("这是全项目**被引用次数最多**的一个数字（二期阶段6/8、三线 A 线都以它为出发点），"
        "所以单独审计：")
    j = _load_json("workstream_J_metrics.json")
    ej2 = j.get("ej2") or {}
    ej4 = j.get("ej4") or {}
    if ej2.get("ok"):
        lo, hi = ej2["ci95"]
        add("")
        add(f"- 原始 k（未饱和 3 档）= **{ej2['k_point']:.3f}**"
            f"（γ = {ej2['gamma_point']:.3f}），"
            f"95% CI = **[{lo:.3f}, {hi:.3f}]**（宽 {ej2['width']:.3f}）")
        add(f"- CI **同时覆盖 0.5（平方根律）与 1.0（线性）**"
            f"⇒ 在原始档位配置下，"
            f"「模型冲击偏线性、不是平方根律」这个方向性判断"
            f"**无法被判定**")
        add(f"- 但同一份数据，只要把档位从 3 档扩到 4 档"
            f"（接纳 1 个饱和档），CI 下界就从 {lo:.3f} 抬到 ~0.71 "
            f"——**方向性判断在扩档位后站得住**（详见工作线J 小结）")
        add(f"- 判定：**已审计**（不重复劳动），结论见 `docs/分辨力危机-交付小结.md`")
    else:
        add("- ⚠️ 缺 `out/workstream_J_metrics.json`，先跑 `scripts/run_workstream_J.py`")
    add("")
    add("### 3.3 确认不适用本条纪律（不是遗漏）")
    add("")
    add("| 类别 | 例子 | 为什么不需要 CI |")
    add("|---|---|---|")
    add("| 直接可观测量 | 稀疏化生效占比 99.2%、净/总暴露 0.226 | "
        "数出来的占比，不经拟合 |")
    add("| 确定性比值 | λ̄ 标尺偏差 1.767×、0.935× | 两个实测值相除，不是统计估计 |")
    add("| 逐位一致 | 复现自查差 0.000 | 同种子确定性复现 |")
    add("| 事件率 | EA.0 的公式偏差 <1% | 直接测的到达率 |")
    add("")
    add("### 3.4 已回填标准配置（工作线L）")
    add("")
    for ln in _backfill_rows():
        add(ln)
    add("")
    add("### 3.5 H4 象限检验的状态（工作线M）")
    for ln in _h4_status_rows():
        add(ln)
    add("")
    add("## 4. 结尾：这份清单要来干什么")
    add("")
    add("**不是要求本轮把所有裸奔陈述都补上 CI**——那是开放式的长期工作。")
    add("它要解决的是：**以后回头看历史结论时，能一眼看出哪些可信、哪些需要重新核实。**")
    add("")
    add("配合新增的第七条纪律（「先报 CI 再报差」），从此：")
    add("")
    add("1. **新写的**陈述必须带 CI / 标准误 / 种子数，否则只能算「当次读数」；")
    add("2. **旧的**陈述按本清单分级——k 类一律降级为「当次读数」；")
    add("3. 再出现「A 变到 B」式的比较时，先问一句：**这个指标的区间有多宽？**")
    add("")
    return "\n".join(L) + "\n"


def _load_json(name: str) -> dict:
    p = OUT / name
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def main() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="只打印统计")
    args = ap.parse_args()
    banner("工作线K  数值陈述分辨力盘点")

    scanned: dict[str, list[dict]] = {}
    for label, path in REPORTS:
        txt = load_text(path)
        if not txt:
            print(f"  ⚠️ 缺报告：{path}")
            scanned[label] = []
            continue
        rows = scan_report_for_numeric_claims(txt, label=label)
        scanned[label] = rows
        ok = sum(1 for r in rows if r["has_ci_nearby_heuristic"])
        print(f"  {label:<20} 命中 {len(rows):>5} 处；"
              f"附近有 CI 证据 {ok:>5}；疑似裸奔 {len(rows) - ok:>5}")

    RAW.write_text(json.dumps(scanned, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    md = build_inventory(scanned)
    TARGET.write_text(md, encoding="utf-8")
    n_total = sum(len(v) for v in scanned.values())
    print(f"\n  原始命中已存：{RAW}（{n_total} 处）")
    print(f"  清单已生成：{TARGET}（{len(md.encode('utf-8')) / 1024:.0f} KB）")
    return scanned


if __name__ == "__main__":
    main()
