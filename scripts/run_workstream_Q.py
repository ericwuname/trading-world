"""工作线Q：把「非对称原则」相关的历史表述统一追加上后续更正。

任务书：`交易世界 · 一致性审计与收尾任务书.md` §4。

两条铁律
--------
1. **只追加、不删除**（EQ.3）：三线深挖当时的原始表述与数字**全部保留**，
   只在旁边追加「后续回填后已更正」的批注。
   这个项目的风格不是抹掉走过的弯路，是如实记录「弯路是怎么被发现和纠正的」。
2. **必须用 `scripts/safe_batch_replace.py`**（第八条纪律 + 任务书 §4.1）：
   先 ``dry_run=True`` 看 diff 确认无误，再实际应用。

标准措辞见任务书 §4.2（本文件里内联为 ``STANDARD_NOTE``），
保证每处文档口径一致，不各写各的。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import ROOT, banner  # noqa: E402
from safe_batch_replace import UnsafeReplaceError, safe_batch_replace  # noqa: E402

OUT_DIFF = ROOT / "out" / "workstream_Q_diff.json"

#: 标准措辞（任务书 §4.2）——每处替换都用同一段，避免口径分叉
STANDARD_NOTE = (
    "\n\n> ⚠️ **后续更正（回填标准配置后，见工作线 L / O）**：\n"
    "> 三线深挖最初观察到「只稀疏化吃单方」(k=0.455) 比「双边稀疏化」(k=0.947) "
    "更接近平方根律，据此提出「非对称原则」是修复冲击凹度的关键。\n"
    "> 经过回填标准配置（8 种子、未饱和档位）重新测量，"
    "taker / both / maker 三个配置的 k 分别为 **0.734 / 0.794 / 0.739**，"
    "**极差只有 0.060**，而这个差异所需的分辨力（约 **1000 个种子**）"
    "远超过当前投入的规模（工作线 O 的区间重叠检验：三对两两比较的重叠都在 80% 以上）。\n"
    "> 更准确的结论是：真正驱动冲击凹度改善的，是**「存在时间聚集性」这件事本身**"
    "（Hawkes 关闭→开启，k 从 1.292 到约 0.755，效应量 **0.536**）；"
    "聚集性具体作用在**需求侧还是供给侧**，影响相对次要——"
    "**效应量小一个数量级**（0.060 vs 0.536，约 9 倍）。\n"
    "> 注：上面这些点估计**方向仍然一致**（taker 最小），只是统计上无法与噪声区分；"
    "被撤回的是「非对称配置更好」这个**结论性**说法，不是这些读数本身。"
)

#: (相对路径, 原文, 说明)——原文必须与文件里的内容**逐字一致**
REPLACEMENTS = [
    (
        "docs/三线深挖-交付小结.md",
        "⇒ **需求侧（吃单方）才是关键**：只稀疏化吃单方时 k = 0.455"
        "（几乎精确命中平方根律），而两侧同时稀疏化反而互相抵消。",
        "EQ.2 主线结论（被后续回填推翻的那句）",
    ),
    (
        "docs/三线深挖-交付小结.md",
        "| A.5 单侧应用 EA.4 | ✅ | 需求侧 k=0.455，是 A 线最接近平方根律的一档 |",
        "EQ.2 汇总表里那一行",
    ),
    # ⚠️ 前置校验诊断报告**不加**批注：它本身就是「发现 k 无分辨力」的更正性文档
    #    （§"CI 审计"表里已经写着 CI=[-0.438, 1.562]、「1/4 档显著」），
    #    在它上面再贴一张「后续更正」反而是画蛇添足。
    #    ——这条是**核对实际内容后**才决定去掉的，不是漏掉。
]


def match_counts() -> list[dict]:
    """逐条规则统计匹配次数——**静默不匹配是危险的**：
    替换脚本报「成功」但其实一条都没改，读起来和「改好了」一样。"""
    out = []
    for rel, old, desc in REPLACEMENTS:
        p = ROOT / rel
        n = p.read_text(encoding="utf-8").count(old) if p.is_file() else -1
        out.append({"file": rel, "desc": desc, "matches": n,
                    "ok": n == 1})
    return out


def build_rules() -> list[tuple[str, str]]:
    return [(old, old + STANDARD_NOTE) for _, old, _ in REPLACEMENTS]


def run(*, dry_run: bool = True) -> dict:
    files = sorted({r[0] for r in REPLACEMENTS})
    paths = [ROOT / f for f in files]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise SystemExit(f"❌ 文件缺失：{missing}")

    # ⚠️ 逐文件独立替换：一个文件里的规则不能误用到另一个文件上
    all_diffs: dict[str, list[str]] = {}
    changed: list[str] = []
    for rel in files:
        rules = [(old, old + STANDARD_NOTE)
                 for f, old, _ in REPLACEMENTS if f == rel]
        try:
            # ⚠️ literal=True：原文含 `**`、`|` 等 markdown 符号，
            #    当正则会直接报 multiple repeat（实测踩到）。
            res = safe_batch_replace([ROOT / rel], rules, dry_run=dry_run,
                                     root=ROOT, literal=True)
        except UnsafeReplaceError as e:
            raise SystemExit(f"❌ {rel}：{e}") from e
        if res["changed"]:
            changed.extend(res["changed"])
            all_diffs.update(res["diffs"])
    return {"applied": (not dry_run) and bool(changed),
            "changed": changed, "diffs": all_diffs,
            "n_files": len(files), "files": files}


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="确认 diff 无误后再加这个开关实际写入")
    args = ap.parse_args()

    banner("工作线Q  统一「非对称」表述（只追加、不删除）")
    mc = match_counts()
    print("  逐条规则匹配情况（不匹配会在这里暴露，不会静默）")
    for m in mc:
        flag = ("✅" if m["matches"] == 1
                else ("❌ 未匹配" if m["matches"] == 0
                      else f"⚠️ 匹配 {m['matches']} 处"))
        print(f"    {flag}  {m['desc']}  ({m['file']})")
    bad = [m for m in mc if not m["ok"]]
    if bad:
        raise SystemExit("❌ 有规则未匹配或匹配多处——先修正 REPLACEMENTS 再跑")
    res = run(dry_run=not args.apply)

    print(f"  涉及 {res['n_files']} 个文件，"
          f"{'会改动' if res['changed'] else '无需改动'} {len(res['changed'])} 个")
    for key in res["changed"]:
        print(f"\n  ── {key}")
        for line in res["diffs"][key][:6]:
            print(f"     {line[:120]}")
        print(f"     …（共 {len(res['diffs'][key])} 行 diff）")

    OUT_DIFF.parent.mkdir(parents=True, exist_ok=True)
    OUT_DIFF.write_text(json.dumps(
        {"changed": res["changed"],
         "diffs": {k: v for k, v in res["diffs"].items()},
         "dry_run": not args.apply,
         "note": "工作线Q 的 diff 清单（供人工复核）——任务书 §4.4 要求实际生成"},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  diff 清单：{OUT_DIFF.relative_to(ROOT)}")

    if res["applied"]:
        print(f"  ✅ 已写回 {len(res['changed'])} 个文件（只追加了更正批注，未删除任何原文）")
    else:
        print("  （dry-run：没有写回任何文件。确认无误后加 --apply）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
