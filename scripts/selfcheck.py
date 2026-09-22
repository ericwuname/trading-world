"""交付物自检：把"报告文字 vs 实测产物"的一致性变成可执行的检查。

为什么需要它
------------
本项目有一条硬纪律：**报告里的数字必须从 ``out/*.json`` 读出来，不许手抄。**
但「不手抄」只保证了**生成时**一致，保证不了：

· 报告生成器里仍然可能写死一个数字（实测踩到：``0.61~1.25`` 这个
  「一期冲击指数区间」是从指导书转述抄来的，而自己的数据是
  ``0.61~3.09``——报告文字与实测产物直接矛盾，且**没有任何机制会发现**）；
· 某个阶段的 JSON 缺失时，报告会静默降级成"尚未运行"，
  读者可能没注意到某个阶段其实没跑；
· 图被引用但文件不在，或者文件在但没被引用（等于白跑）；
· 报告里可能漏出 ``nan`` / ``None`` 这类未处理值。

所以需要一个**独立于生成器**的检查。它只读产物，不看生成器的代码——
这样才不会"用生成器的假设去验证生成器"。

写作纪律（踩过的坑）
--------------------
本文件里的**中文说明字符串**一律用直角引号 「」，
绝不在字符串内部使用 ASCII 双引号——否则自动修引号脚本会把
**参数分隔符**也一起改掉（本项目真实踩过，18 行被写坏）。
代码里的分隔符**必须**保持 ASCII 双引号。

用法::

    python scripts/selfcheck.py          # 返回码 0 = 全过
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
FIGS = OUT / "figs"
DOCS = ROOT / "docs"

STAGES = list(range(1, 12))
REPORT2 = DOCS / "交易世界二期-交付报告.html"
REPORT1 = DOCS / "交易世界-交付报告.html"


class Report:
    def __init__(self) -> None:
        self.fail = 0
        self.warn = 0
        self.ok = 0

    def good(self, msg: str) -> None:
        self.ok += 1
        print(f"  [OK]   {msg}")

    def bad(self, msg: str) -> None:
        self.fail += 1
        print(f"  [FAIL] {msg}")

    def caution(self, msg: str) -> None:
        self.warn += 1
        print(f"  [WARN] {msg}")



def newest_log(*names: str) -> Path | None:
    """在 ``out/<name>`` 与 ``out/repro_logs/<name>`` 里取**较新**的那个。

    ``reproduce_all`` 用 ``out/repro_logs/`` 作步骤日志目录，
    而报告生成器与自检历史读的是 ``out/mutation_run.log`` 这种规范路径。
    两边只要有一边忘了同步，读者就会拿到**上一轮**的日志——
    而那个文件是真实存在的、内容也是真的，**看起来毫无异常**。

    取"较新"是防御性写法：即使同步那一步失效，读者也不会读到旧结论。
    """
    cands = []
    for n in names:
        for p in (OUT / n, ROOT / "out" / "repro_logs" / n):
            if p.exists():
                cands.append(p)
    if not cands:
        return None
    return max(cands, key=lambda q: q.stat().st_mtime)


def sec(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


# ----------------------------------------------------------------------
def check_jsons(r: Report) -> None:
    sec("① 阶段产物：11 个 JSON 是否齐全且可解析")
    for i in STAGES:
        p = OUT / f"stage{i}_metrics.json"
        if not p.exists():
            r.bad(f"stage{i}_metrics.json 缺失")
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            r.bad(f"stage{i}_metrics.json 解析失败：{e}")
            continue
        if d.get("stage") != i:
            r.bad(f"stage{i}_metrics.json 的 stage 字段是 {d.get('stage')!r}")
            continue
        # 二期阶段必须带诚实边界（指导书 §0 的硬性要求）
        if i >= 5:
            hb = d.get("honest_boundary")
            if not hb:
                r.bad(f"stage{i} 缺少 honest_boundary（指导书 §0 硬性要求）")
                continue
            if len(hb) < 5:
                r.caution(f"stage{i} 的 honest_boundary 只有 {len(hb)} 行，偏短")
        r.good(f"stage{i}（{len(json.dumps(d)) // 1024} KB）")


def find_leftover_placeholders(html: str) -> list[str]:
    """找出报告 HTML 里没被替换掉的 ``{占位符}``。

    真实踩过：``sec_overview`` 用了普通三引号字符串（忘了 f 前缀），
    于是 ``{_krange_txt(jsons)}`` 原样印进了总览表。**报告照常生成、
    大小正常、章节齐全、图也不缺——没有任何机制会发现。**

    做法：先剥 ``<style>`` / ``<script>``（那里面的花括号是 CSS/JS 语法），
    再找以字母或下划线开头的 ``{...}``。
    """
    body = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.S)
    body = re.sub(r"<script[^>]*>.*?</script>", "", body, flags=re.S)
    out = []
    for m in re.finditer(r"\{[^{}\n]{1,120}\}", body):
        s = m.group(0)
        if re.match(r"^\{[A-Za-z_]", s):
            out.append(s)
    return out


def check_report(r: Report) -> None:
    sec("② 二期报告：是否所有阶段都有真实内容")
    if not REPORT2.exists():
        r.bad("二期报告不存在")
        return
    h = REPORT2.read_text(encoding="utf-8")

    # 某个阶段真的没跑时，报告会插一段**特定**的占位文本。
    # ⚠️ 不能按裸词「尚未运行」计数：正文里本来就会讨论这件事
    #    （"凡是某个阶段的 JSON 缺失，本报告显式标尚未运行"、
    #      reproduce_all 的说明里也有这个词）。第一版就是这么写的，
    #     结果把两处**正当的解释性文字**判成了"有阶段没跑"——
    #     检查比被检查的东西还脆，迟早会被当成噪音关掉。
    MISSING_MARK = "（该阶段尚未运行，本报告不代替它下结论）"
    n_missing = h.count(MISSING_MARK)
    if n_missing:
        r.bad(f"报告里有 {n_missing} 个阶段只有占位文本"
              f"（{MISSING_MARK}）——有阶段没跑就生成了报告")
    else:
        r.good("没有任何阶段退化成占位文本")

    # 漏出的未处理值。
    # ⚠️ 必须只看**表格单元格**（<td>/<th>），不能全文找 ">inf<"。
    # 真实踩过：报告正文里有一句「修法：非常数返回 <code>inf</code>」，
    # 全文匹配会把它当成漏出的值报假失败。漏出的真形态是 <td>inf</td>。
    cell = r"<t[dh][^>]*>\s*(nan|NaN|None|inf|-inf|Infinity)\s*</t[dh]>"
    leaked = re.findall(cell, h)
    if leaked:
        from collections import Counter
        r.bad(f"报告表格里漏出未处理值：{dict(Counter(leaked))}")
    else:
        r.good("表格单元格里没有漏出 nan / None / inf")

    # 没被替换掉的 f-string 占位符（报告生成器忘了 f 前缀）
    leaked_ph = find_leftover_placeholders(h)
    if leaked_ph:
        r.bad(f"报告里有 {len(leaked_ph)} 处没被替换掉的占位符："
              f"{sorted(set(leaked_ph))}")
    else:
        r.good("没有没被替换掉的占位符")

    n_miss_fig = h.count("缺图：")
    if n_miss_fig:
        r.bad(f"报告有 {n_miss_fig} 张图缺失")
    else:
        r.good("所有引用的图都存在（无「缺图」占位）")

    # 13 个章节一个都不能少
    n_h2 = len(re.findall(r"<h2[^>]*>", h))
    if n_h2 < 13:
        r.bad(f"报告只有 {n_h2} 个 h2 章节（应为 13）")
    else:
        r.good(f"{n_h2} 个章节齐全")

    # 每个阶段的关键结论词必须出现
    kws = ["永续合约", "长记忆", "反身性", "Hawkes", "多资产",
           "统一校准", "至暗时刻", "诚实边界", "方法论"]
    missing = [kw for kw in kws if kw not in h]
    if missing:
        r.bad(f"报告里找不到这些关键词：{missing}")
    else:
        r.good("七个阶段 + 诚实边界 + 方法论 全部在报告中出现")


def check_cross_refs(r: Report) -> None:
    sec("③ 交叉引用与章节编号是否自洽")
    if not REPORT2.exists():
        return
    h = REPORT2.read_text(encoding="utf-8")

    # 抓出「见第 N 节」与真实章节标题的对应关系
    titles: dict[int, str] = {}
    for m in re.finditer(r"<h2[^>]*>\s*(\d+)\.\s*([^<]+)", h):
        titles[int(m.group(1))] = m.group(2).strip()
    if not titles:
        r.caution("解析不出任何 h2 章节编号，跳过交叉引用检查")
        return

    refs = re.findall(r"见第\s*(\d+)\s*节", h)
    bad = [(n, titles.get(int(n), "<不存在>")) for n in refs
           if "阶段" not in titles.get(int(n), "")]
    if bad:
        r.bad(f"「见第 N 节」指向了非阶段章节：{bad}")
    else:
        r.good(f"{len(refs)} 处「见第 N 节」都指向阶段章节")

    # 任何指向不存在章节的引用也是错的
    ghosts = [n for n in refs if int(n) not in titles]
    if ghosts:
        r.bad(f"「见第 N 节」指向了不存在的章节：{sorted(set(ghosts))}")
    else:
        r.good(f"{len(refs)} 处「见第 N 节」都指向真实存在的章节")

    # 总览表：阶段 N 那一行里的「见第 M 节」，M 必须是标题含「阶段N」的那一节。
    # 真实表格结构是 <tr><td>7 反身性</td>...<td>见第 5 节</td>...
    # （阶段号与名称在**同一个** td 里，不是 "N 阶段N"）
    stage_section = {}
    for num, t in titles.items():
        mm = re.search(r"阶段\s*(\d+)", t)
        if mm:
            stage_section[int(mm.group(1))] = num
    rows = re.findall(r"<tr[^>]*>\s*<td[^>]*>\s*(\d+)\s+[^<]*</td>(.{0,400}?)</tr>",
                      h, re.S)
    mism, checked = [], 0
    for stg, rest in rows:
        stg = int(stg)
        if not (5 <= stg <= 11):
            continue
        mr = re.search(r"见第\s*(\d+)\s*节", rest)
        if not mr:
            continue
        checked += 1
        want = stage_section.get(stg)
        if want is None:
            mism.append((stg, mr.group(1), "<没有对应章节>"))
        elif int(mr.group(1)) != want:
            mism.append((stg, mr.group(1), want))
    if mism:
        r.bad(f"总览表「阶段N → 见第M节」映射错误（阶段, 实际写, 应为）：{mism}")
    elif checked:
        r.good(f"总览表 {checked} 条「阶段N → 见第M节」全部映射正确")
    else:
        r.bad("总览表一条交叉引用都没解析到——表格结构变了，检查已失效")


def check_figs(r: Report) -> None:
    sec("④ 阶段图：有没有「跑了但没用上」的")
    p5 = []
    for q in FIGS.glob("stage*.png"):
        head = q.name.split("_")[0]
        num = head[5:]
        if num.isdigit() and int(num) >= 5:
            p5.append(q.name)
    if not REPORT2.exists():
        return
    h = REPORT2.read_text(encoding="utf-8")
    # 图是 base64 内嵌的，文件名不会出现在 HTML 里；
    # 所以只能反过来查：报告里的 <figure> 数量 vs 应引用的图数量。
    n_fig_in_report = h.count("<figure")
    if n_fig_in_report < len(p5):
        r.bad(f"报告内嵌 {n_fig_in_report} 张图，而 out/figs 里有 {len(p5)} 张"
              f"——有图没被引用（等于白跑）")
    else:
        r.good(f"报告内嵌 {n_fig_in_report} 张图 ≥ out/figs 的 {len(p5)} 张")

    # 一期报告的图不重不漏
    if REPORT1.exists():
        hr = REPORT1.read_text(encoding="utf-8")
        if hr.count("缺图："):
            r.bad("一期报告有缺图")
        else:
            r.good(f"一期报告完好（{hr.count('<figure')} 图 / "
                   f"{hr.count('<table')} 表 / "
                   f"{len(re.findall(r'<h2', hr))} 章）")


def _walk_k(o):
    if isinstance(o, dict):
        if "k" in o:
            try:
                v = float(o["k"])
                if v == v:
                    yield v
            except (TypeError, ValueError):
                pass
        for vv in o.values():
            yield from _walk_k(vv)
    elif isinstance(o, list):
        for vv in o:
            yield from _walk_k(vv)


def check_hardcoded_numbers(r: Report) -> None:
    sec("⑤ 报告里的数字 vs JSON（防「手抄」回归）")
    if not REPORT2.exists():
        r.bad("二期报告不存在，无法核对数字")
        return
    h = REPORT2.read_text(encoding="utf-8")

    # ① 阶段3 的 k 区间：报告里必须与 JSON 现算值一致
    s3 = OUT / "stage3_metrics.json"
    if s3.exists():
        ks = list(_walk_k(json.loads(s3.read_text(encoding="utf-8"))))
        if not ks:
            r.caution("stage3 里没找到 k 值，跳过")
        else:
            lo, hi = min(ks), max(ks)
            want = f"{lo:.2f}~{hi:.2f}"
            if want in h:
                r.good(f"阶段3 的 k 区间 {want}（JSON 现算）出现在报告里")
            else:
                r.bad(f"报告里找不到与 JSON 一致的一期 k 区间 {want}")
    else:
        r.caution("缺 out/stage3_metrics.json")

    # ② 变异验证的结论行
    mut = newest_log("mutation_run.log", "mutation.log")
    if mut is not None:
        m = mut.read_text(encoding="utf-8", errors="replace")
        hit = re.search(r"(\d+)\s*/\s*(\d+)\s*个注入的 bug 全部被测试抓到", m)
        if hit and hit.group(1) == hit.group(2):
            r.good(f"变异验证 {hit.group(1)}/{hit.group(2)} 全部被抓到")
            if hit.group(1) in h:
                r.good(f"报告里引用了正确的变异体数量（{hit.group(1)}）")
            else:
                r.bad(f"报告里找不到变异体数量 {hit.group(1)}")
        else:
            r.bad("变异验证没有全绿（或日志格式变了）")
    else:
        r.caution("缺 out/mutation_run.log")

    # ③ 测试数
    tc = OUT / "test_count.txt"
    if tc.exists():
        n = tc.read_text(encoding="utf-8").strip()
        if n in h:
            r.good(f"报告里的测试数（{n}）与记录一致")
        else:
            r.bad(f"报告里找不到测试数 {n}")
    else:
        r.caution("缺 out/test_count.txt")

    # ③b README 里**手写**的测试数必须与实测一致。
    #     手写的常量一定会脱节——本轮正好踩到：README 写着 528，
    #     而实测已经 541（三线深挖又加了测试）。
    #     ``out/test_count.txt`` 是 tests 步骤自动写回的，所以拿它当唯一事实源。
    #     这一条的意义不是"数字对不对"，而是**脱节会被发现**——
    #     上一版的失效形态正是"数字错了但看起来完全正常"。
    rm = ROOT / "README.md"
    if tc.exists() and rm.exists():
        n = tc.read_text(encoding="utf-8").strip()
        rt = rm.read_text(encoding="utf-8")
        hit_n = re.search(r"tests/\s+(\d+)\s*项测试", rt)
        if hit_n is None:
            r.caution("README 里没找到「tests/ N 项测试」，这条检查已失效")
        elif hit_n.group(1) == n:
            r.good(f"README 手写的测试数（{n}）与实测一致")
        else:
            r.bad(f"README 写着 {hit_n.group(1)} 项测试，实测 {n} —— 手写常量已脱节")

    # ③c README 里的变异体数量必须与 mutation_check.py 里实际注册的一致
    rm = ROOT / "README.md"
    if rm.exists():
        from _common import mutation_ids  # noqa: PLC0415
        ids = mutation_ids()
        rt = rm.read_text(encoding="utf-8")
        hit_m = re.search(r"变异验证：(\d+)\s*项注入 bug", rt)
        if not ids:
            r.caution("从 mutation_check.py 里解析不出变异体编号，这条检查已失效")
        elif hit_m is None:
            r.caution("README 里没找到「变异验证：N 项注入 bug」，这条检查已失效")
        elif int(hit_m.group(1)) == len(ids):
            r.good(f"README 手写的变异体数量（{len(ids)}）与实际注册一致")
        else:
            r.bad(f"README 写着 {hit_m.group(1)} 个变异体，实际注册 {len(ids)} 个"
                  f"（M{ids[0]}~M{ids[-1]}）")

    # ④ 靶子值：不在里查了。
    # 靶子曾经用过 ``{v*100:.1f}%`` 拼字符串去找，然后被报告里的
    # ``0.1157``（小数写法）判成"找不到"——**检查比被检查的东西还脆**。
    # 现在统一由 ⑨ 的承重数字表按**数值+容差**查，对排版不敏感。


def check_imports(r: Report) -> None:
    sec("⑥ 所有模块可导入（含一期四个阶段脚本）")
    import importlib
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "scripts"))
    tw_mods = ["tw", "tw.market", "tw.eval", "tw.strategy", "tw.scenarios",
               "tw.impact", "tw.perpetual", "tw.multi_asset",
               "tw.analyzer_longmemory", "tw.order_flow.meta_order",
               "tw.order_flow.hawkes", "tw.order_flow.hawkes_market",
               "tw.experiments.reflexivity", "tw.calibration.objective",
               "tw.calibration.optimizer", "tw.agents.adaptive_trend",
               "tw.agents.adaptive_liquidity", "tw.agents.pairs_trader",
               "tw.agents.hedger"]
    bad = []
    for m in tw_mods:
        try:
            importlib.import_module(m)
        except Exception as e:  # noqa: BLE001
            bad.append((m, repr(e)))
    if bad:
        for m, e in bad:
            r.bad(f"{m}: {e}")
    else:
        r.good(f"{len(tw_mods)} 个库模块全部可导入")

    scr_mods = [f"run_stage{i}" for i in STAGES] + [
        "make_report", "make_report2", "mutation_check", "lab", "calibrate",
        "run_sweep_study", "gui", "_cache", "_common", "reproduce_all",
        "make_stage_summary", "bench_market",
        "run_workstream_A", "run_workstream_B", "run_workstream_C",
        "run_workstream_joint", "make_workstream_summary",
        # 本轮（EA.4 扶正与前置校验）新增的三个：
        #   verify_EA4_calibration  —— 标定完整性三步校验
        #   diagnose_k_uncertainty  —— k 的分辨力诊断（bootstrap）
        #   make_verify_report      —— 给设计者的诊断报告
        "verify_EA4_calibration", "diagnose_k_uncertainty",
        "make_verify_report",
        # 分辨力危机那一轮新增的六个：
        #   run_workstream_H  —— 路径C 饱和检查
        #   run_workstream_I  —— 凹度判据（路径B）
        #   run_workstream_E  —— H4 假说验证
        #   run_workstream_J  —— 阶段3 回溯审计
        #   audit_numeric_claims    —— 数值陈述扫描（第七条纪律的自动化）
        #   make_resolution_summary —— 本轮交付小结
        "run_workstream_H", "run_workstream_I", "run_workstream_E",
        "run_workstream_J", "audit_numeric_claims", "make_resolution_summary",
        # 回填标准配置与 H4 做实那一轮新增的六个：
        #   safe_batch_replace    —— 第八条纪律的强制执行入口（临时副本 + ast 校验）
        #   claim_mutation_ids    —— 变异编号的权威分配器
        #   run_workstream_L      —— 回填 7 档 × 8 种子到历史家族
        #   run_workstream_M      —— H4 功效分析 + 加码/诚实报告
        #   power_analysis_H4     —— 功效公式（scipy 非中心 t 分布）
        #   update_inventory_post_L_M —— 收尾：把 L/M 结果并入历史清单
        "safe_batch_replace", "claim_mutation_ids", "run_workstream_L",
        "run_workstream_M", "power_analysis_H4", "update_inventory_post_L_M",
        # 一致性审计与收尾那一轮新增的四个：
        #   run_workstream_O   —— 用区间重叠度量审上一轮自己的「推翻」
        #   run_workstream_Q   —— 「非对称」表述统一更正（safe_batch_replace 驱动）
        #   tag_final_verdicts —— 新验收标准：方向性偏离检验
        #   make_consistency_summary —— 本轮交付小结
        "run_workstream_O", "run_workstream_Q", "tag_final_verdicts",
        "make_consistency_summary"]
    bad2 = []
    for m in scr_mods:
        try:
            importlib.import_module(m)
        except Exception as e:  # noqa: BLE001
            bad2.append((m, repr(e)))
    if bad2:
        for m, e in bad2:
            r.bad(f"scripts/{m}: {e}")
    else:
        r.good(f"{len(scr_mods)} 个脚本全部可导入")


def _mean_pair_corr(d: dict) -> float:
    pairs = ((d.get("e9_1_real") or {}).get("pairs") or {})
    vs = [v for v in pairs.values() if isinstance(v, (int, float))]
    return sum(vs) / len(vs) if vs else float("nan")


# 报告的"承重数字"清单。
#
# 为什么要有这张表：报告生成器里曾经**手抄**过这些数字（比如总览表里
# 写死「k 反而从 0.665 走到 0.910」，一共抄在 3 个地方）。手抄的危险
# 不是抄错一次，而是**重跑之后没人记得去改**——报告会拿着旧数字描述
# 新结果，而且没有任何机制会发现。
#
# 真实发生：2026-09-18 修掉阶段6 对照组的 bug 后重跑，k 变了，
# 而 3 处手抄的地方一个都没动。这张表就是那一刻该存在的东西。
#
# 每项：(标签, JSON 文件名, 取值函数, 模式, 规格)
#   模式 "num"  → 规格是容差；只要报告正文里有任何数值落在容差内就算过
#                 （对排版格式不敏感，但对**值本身**很敏感）
#   模式 "text" → 规格是格式串；要求字面量出现（用于带单位的写法、计数）
KEY_FIGURES = [
    ("阶段5 年化倍数（实测/真实）", "stage5_metrics.json",
     lambda d: (d["e5_4_match"]["best"]["annualized"]
                / d["targets"]["real_annual"]), "text", "{:.2f}×"),
    ("阶段5 实测年化", "stage5_metrics.json",
     lambda d: d["e5_4_match"]["best"]["annualized"], "num", 5e-5),
    ("阶段5 正费率占比（实测）", "stage5_metrics.json",
     lambda d: d["e5_4_match"]["best"]["pos_frac"], "num", 5e-5),
    ("阶段5 靶子 · 真实年化", "stage5_metrics.json",
     lambda d: d["targets"]["real_annual"], "num", 5e-5),
    ("阶段5 靶子 · 真实正占比", "stage5_metrics.json",
     lambda d: d["targets"]["real_pos_frac"], "num", 5e-5),
    ("阶段6 k · 对照（因果滑点）", "stage6_metrics.json",
     lambda d: d["e6_4_fits"]["机制关（对照）|slippage_bp"]["exponent"],
     "num", 5e-4),
    ("阶段6 k · 处理（因果滑点）", "stage6_metrics.json",
     lambda d: d["e6_4_fits"]["元订单+自适应|slippage_bp"]["exponent"],
     "num", 5e-4),
    ("阶段6 k · 对照（最差成交价）", "stage6_metrics.json",
     lambda d: d["e6_4_fits"]["机制关（对照）|worst_fill_bp"]["exponent"],
     "num", 5e-4),
    ("阶段6 k · 处理（最差成交价）", "stage6_metrics.json",
     lambda d: d["e6_4_fits"]["元订单+自适应|worst_fill_bp"]["exponent"],
     "num", 5e-4),
    ("阶段6 k · 对照（窗口均偏离）", "stage6_metrics.json",
     lambda d: d["e6_4_fits"]["机制关（对照）|during_mean_bp"]["exponent"],
     "num", 5e-4),
    ("阶段6 k · 处理（窗口均偏离）", "stage6_metrics.json",
     lambda d: d["e6_4_fits"]["元订单+自适应|during_mean_bp"]["exponent"],
     "num", 5e-4),
    ("阶段8 E8.2 · Hawkes 关闭时 k", "stage8_metrics.json",
     lambda d: d["e8_2"]["k_off"], "num", 5e-4),
    ("阶段8 E8.2 · Hawkes 开启时 k", "stage8_metrics.json",
     lambda d: d["e8_2"]["k_on"], "num", 5e-4),
    ("阶段9 真实 BTC|ETH 相关", "stage9_metrics.json",
     lambda d: d["e9_1_real"]["pairs"]["BTCUSDT_1h|ETHUSDT_1h"], "num", 5e-5),
    ("阶段9 真实三资产平均相关", "stage9_metrics.json",
     _mean_pair_corr, "num", 5e-5),
    ("阶段11 观察到传导的环节数", "stage11_metrics.json",
     lambda d: d["e11_1_n_links"], "text", "{:.0f}/6"),
]


def strip_html_noise(h: str) -> str:
    """剥掉 CSS / JS / base64 内嵌图，只留**正文文本**。

    ⚠️ 必须剥掉 base64 内嵌图：那里面的字符集包含数字与字母，
    不剥的话会产生上百万个假 token（而且每个都可能"碰巧"匹配）。
    本函数被自检与工作线K 的数值陈述扫描**共用**——同一件事不写两份实现。
    """
    b = re.sub(r"<style[^>]*>.*?</style>", "", h, flags=re.S)
    b = re.sub(r"<script[^>]*>.*?</script>", "", b, flags=re.S)
    b = re.sub(r"data:image/[A-Za-z+]*;base64,[A-Za-z0-9+/=\s]+", "", b)
    return re.sub(r"<[^>]+>", " ", b)


def _report_numeric_tokens(h: str) -> list[float]:
    """把报告里**正文**（不含 CSS/JS/base64 图）的数值都抠出来。

    为什么要抠数字而不是直接找格式化后的字符串：报告把一个量写成
    ``0.7946`` 还是 ``0.795`` 是排版选择，而"报告里到底写的是哪个值"
    才是要查的事。第一版按 ``{:.3f}`` 拼字符串去找，结果被 ``0.7946``
    判成"找不到"——**检查比被检查的东西还脆**，这种检查迟早会被
    当成噪音而关掉。
    """
    b = strip_html_noise(h)
    out = []
    for m in re.finditer(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?", b):
        try:
            out.append(float(m.group(0)))
        except ValueError:
            pass
    return out


def _value_visible(tokens: list[float], v: float, tol: float) -> bool:
    return any(abs(t - v) <= tol for t in tokens)


def check_key_figures(r: Report) -> None:
    """报告里的承重数字必须能在 JSON 里现算出来，并且**确实出现**在报告中。"""
    sec("⑨ 承重数字：JSON 现算值是否真的出现在报告里")
    if not REPORT2.exists():
        r.bad("二期报告不存在，无法核对承重数字")
        return
    h = REPORT2.read_text(encoding="utf-8")
    tokens = _report_numeric_tokens(h)
    if not tokens:
        r.bad("从报告里抠不出任何数值——文字提取已失效，检查不可信")
        return
    bad, checked = [], 0
    for label, fname, getter, mode, spec in KEY_FIGURES:
        p = OUT / fname
        if not p.exists():
            bad.append(f"{label}：缺 {fname}")
            continue
        try:
            v = getter(json.loads(p.read_text(encoding="utf-8")))
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as e:
            bad.append(f"{label}：取数失败 {e!r}")
            continue
        if not isinstance(v, (int, float)) or v != v:
            bad.append(f"{label}：算出来是 {v!r}（nan？）")
            continue
        checked += 1
        if mode == "num":
            if not _value_visible(tokens, float(v), float(spec)):
                bad.append(f"{label} = {v:.6g}（JSON 现算，容差 {spec}）"
                           f"在报告正文里找不到任何相近的数值")
        else:  # text
            txt = str(spec).format(v)
            if txt not in h:
                bad.append(f"{label}：报告里找不到字面量 {txt}（JSON 现算）")
    if bad:
        for m in bad:
            r.bad(m)
    else:
        r.good(f"{checked} 个承重数字全部能在报告里对上 JSON 现算值")


def mutation_patch_string_positions(tree) -> set[tuple[int, int]]:
    """``MUTATIONS`` 里那些「源码片段」字符串的位置——静态检查要跳过它们。

    为什么需要这个排除
    -----------------
    ``MUTATIONS`` 的 old/new **就是源码文本**（它们描述"把哪一段换成哪一段"），
    所以里面当然会长得像模板串：M42 的锚点里就写着 ``{sorted(skipped)}``。
    不排除的话，⑦ 项会把自己的代码片段当成"忘了 f 前缀"——**假阳性**。
    本轮真实踩到：自检报了 2 个 FAIL，而代码完全正确。
    （假阳性比没有检查更坏：它会训练人忽略自检。）

    ⚠️ 但**不能**因此整个文件跳过：MUTATIONS 定义之外的字符串仍然要查。
    所以这里按 ``ast`` 精确取位置，而不是按文件或按行粗筛。
    """
    import ast

    positions: set[tuple[int, int]] = set()
    for node in ast.walk(tree):
        # ⚠️ 必须同时认 ``Assign`` 与 ``AnnAssign``。
        #    第一版只处理了 Assign，而真实的 mutation_check.py 写的是
        #    ``MUTATIONS: list[...] = [...]``（带类型注解 → AnnAssign），
        #    于是豁免**静默失效**，自检照旧误报。
        #    **这就是"检查自己写漏了"的形态：没有报错，只是没生效。**
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        if not any(isinstance(t, ast.Name) and t.id == "MUTATIONS"
                   for t in targets):
            continue
        for sub in ast.walk(node.value):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                positions.add((sub.lineno, sub.col_offset))
    return positions


def check_source_placeholders(r: Report) -> None:
    """静态扫源码：非 f-string 的字面量里却写着 ``{名字(...)}`` —— 忘加 f 前缀。

    这条检查的由来是一次真实事故：``make_report2.sec_overview`` 返回的是
    普通三引号字符串，``{_krange_txt(jsons)}`` 就这样印进了报告。
    产物侧的自检（⑨）能抓到症状，但**只有跑完生成器才知道**；
    这条静态检查能在**改完代码、还没生成报告**时就报出来。

    用 ``ast`` 精确排除两类**正当**的字符串：
      ① docstring（文档里举反例是正当的）
      ② ``MUTATIONS`` 里的源码片段（它们本来就是源码文本）
    另外认行内标记 ``# noqa: placeholder``——给"测试里构造的源码样本"用。
    """
    sec("⑦ 源码静态检查：有没有「忘加 f 前缀」的模板字符串")
    import ast

    pat = re.compile(r"\{[A-Za-z_][A-Za-z0-9_]*\s*[\(\.\[]")
    targets = sorted(list((ROOT / "scripts").glob("*.py"))
                     + list((ROOT / "tw").rglob("*.py"))
                     + list((ROOT / "tests").rglob("*.py")))
    findings = []
    n_scanned = 0
    n_exempt = 0
    for p in targets:
        if "__pycache__" in p.parts:
            continue
        try:
            src = p.read_text(encoding="utf-8")
            # ⚠️⚠️ **必须 `compile()`，不能只 `ast.parse()`**（2026-09-22 实测）：
            # `ast.parse` 只建 AST，**不做编译期的语义检查** ⇒
            # `f(a=1, a=2)`（**关键字参数重复**）它能**静默通过**，
            # 而 `compile()` 会报 `keyword argument repeated`。
            # ⇒ 真实后果：`scripts/agent_review.py` 里重复写了一个 `min_history=`
            #   ⇒ **全量 1503 项测试全绿、静态检查也全绿**，
            #   而那个脚本**一跑就崩**（跑在流水线里才发现，白等了两小时）。
            # ⭐ 判据：**"能建 AST" ≠ "能执行"**；
            #   检查"能不能跑"就要用那个"真正会执行的编译器"。
            tree = ast.parse(src)
            compile(src, str(p), "exec")
        except (OSError, SyntaxError) as e:
            r.bad(f"{p.relative_to(ROOT)} 编译失败：{e}")
            continue
        n_scanned += 1
        lines = src.splitlines()
        # docstring 的位置（行, 列）——这些是文档，允许举例
        doc_pos = set()
        for node in ast.walk(tree):
            if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)):
                doc_pos.add((node.value.lineno, node.value.col_offset))
        exempt = doc_pos | mutation_patch_string_positions(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            if (node.lineno, node.col_offset) in exempt:
                n_exempt += 1
                continue
            # 行内标记：给"测试里构造的源码样本"用（那种字符串本来就是样本）。
            # ⚠️ 范围是 [lineno−1, end_lineno]，**故意向前多算一行**。
            #    理由：隐式拼接的多行字符串在 AST 里是**一个** Constant，
            #    lineno 落在第一个字面量那行，**不包含** ``src = (``。
            #    而人很自然会把注释写在 ``src = (`` 那一行 → 标记被静默忽略。
            #    本轮真实踩到：标记位置加对了（人看着合理），检查却照旧报。
            #    ``noqa: placeholder`` 这个标记足够特定，多算一行的误伤概率极低；
            #    反过来"标记不生效"却是高频道坑——所以宽容方向选前者。
            lo = node.lineno - 1
            hi = (node.end_lineno or node.lineno) - 1
            chunk = "\n".join(lines[max(0, lo - 1):hi + 1]) if 0 <= lo < len(lines) else ""
            if "noqa: placeholder" in chunk:
                n_exempt += 1
                continue
            for m in pat.finditer(node.value):
                findings.append(
                    (str(p.relative_to(ROOT)), node.lineno, m.group(0)))
    if findings:
        for f_, ln, s in findings:
            r.bad(f"{f_}:{ln} 字符串里有 {s}…——是不是忘了 f 前缀？")
    else:
        r.good(f"{n_scanned} 个源文件：没有忘加 f 前缀的模板字符串"
               f"（另有 {n_exempt} 处 docstring/变异体锚点/显式标记已豁免）")


def check_repeated_quotes(r: Report) -> None:
    """⭐ 同一个量被报告在**多处**引用时，处处都必须与 JSON 现算值一致。

    为什么要单独查这一条：承重数字表（⑨）只验证"这个值在报告里**出现过**"。
    如果报告在两处引用了同一个量，而只有一处被更新（另一处仍是手抄的旧值），
    ⑨ 会**通过**——因为更新的那处已经满足了"出现过"。

    这不是假想：阶段6 的三个冲击指数 k 就曾经**手抄在两个地方**
    （§1 的三处失败清单、§4.6 的方向冲突说明）。
    只查"出现过"的话，改一处就能让检查变绿，而另一处继续撒谎。
    """
    sec("⑩ 重复引用的量：多处出现时必须处处一致")
    if not REPORT2.exists():
        return
    h = REPORT2.read_text(encoding="utf-8")

    # 允许的取值集合：三指标 × 两臂 + 阶段8 E8.2 的两臂
    def k6(arm, metric):
        try:
            return float(json.loads(
                (OUT / "stage6_metrics.json").read_text(encoding="utf-8")
            )["e6_4_fits"][f"{arm}|{metric}"]["exponent"])
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            return None

    def k8(key):
        try:
            v = json.loads((OUT / "stage8_metrics.json").read_text(
                encoding="utf-8"))["e8_2"][key]
            return float(v) if isinstance(v, (int, float)) else None
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            return None

    metrics = ("slippage_bp", "worst_fill_bp", "during_mean_bp")
    off_ok = {k for k in (k6("机制关（对照）", m) for m in metrics)} | {k8("k_off")}
    on_ok = {k for k in (k6("元订单+自适应", m) for m in metrics)} | {k8("k_on")}
    off_ok.discard(None)
    on_ok.discard(None)
    if not (off_ok and on_ok):
        r.caution("取不到阶段6/8 的 k 值，跳过重复引用检查")
        return

    def close(v: float, allowed: set[float]) -> bool:
        return any(abs(v - a) <= 5e-4 for a in allowed)

    # 抓「对照 k = X …（至多 80 字）… 处理 [k =] Y」的配对
    pairs = re.findall(r"对照\s*k\s*=\s*([\d.]+)(.{0,80}?)(?:→|->)\s*"
                       r"处理\s*(?:k\s*=\s*)?([\d.]+)", h, re.S)
    if not pairs:
        r.caution("没解析到任何「对照 k = X → 处理 Y」配对，检查已失效")
        return
    bad = []
    for a, _mid, b in pairs:
        if not (close(float(a), off_ok) and close(float(b), on_ok)):
            bad.append((a, b))
    if bad:
        r.bad(f"有 {len(bad)} 处「对照 k → 处理 k」与 JSON 现算值不符"
              f"（可能是手抄的旧值）：{bad}")
    else:
        r.good(f"{len(pairs)} 处「对照 k → 处理 k」全部与 JSON 现算值一致"
               f"（说明多处引用没有分叉）")
    # 单独出现的「对照 k = X」也要验
    solo = re.findall(r"对照\s*k\s*=\s*([\d.]+)", h)
    bad_solo = [a for a in solo if not close(float(a), off_ok)]
    if bad_solo:
        r.bad(f"孤立出现的「对照 k = 」有 {len(bad_solo)} 处与 JSON 不符："
              f"{bad_solo}")


def check_docs(r: Report) -> None:
    sec("⑧ 文档齐全")
    need = ["交易世界-交付报告.html", "交易世界二期-交付报告.html",
            "二期阶段小结.md", "标定说明.md", "策略测试手册.md", "GUI说明.md"]
    miss = [n for n in need if not (DOCS / n).exists()]
    if miss:
        r.bad(f"缺文档：{miss}")
    else:
        r.good(f"{len(need)} 份文档齐全")

    md = DOCS / "二期阶段小结.md"
    if md.exists():
        t = md.read_text(encoding="utf-8")
        absent = [i for i in range(5, 12) if f"## 阶段{i}" not in t]
        if absent:
            r.bad(f"阶段小结里没有这些阶段的章节：{absent}")
        else:
            r.good("阶段小结覆盖阶段5~11")
        # 这份文件曾经声称"由脚本产出、不手抄数字"，而实际是手写的。
        # 现在真的由 make_stage_summary.py 生成——这句断言必须为真。
        if "make_stage_summary.py" in t:
            r.good("阶段小结标注了真正的生成器（make_stage_summary.py）")
        else:
            r.bad("阶段小结没有标注生成器——它可能在不知不觉中又变回手写")

    readme = ROOT / "README.md"
    if readme.exists():
        t = readme.read_text(encoding="utf-8")
        absent = [str(i) for i in range(5, 12) if f"阶段{i}" not in t
                  and f"stage{i}" not in t]
        if absent:
            r.caution(f"README 里没提到阶段 {absent}")
        else:
            r.good("README 覆盖阶段5~11")
    else:
        r.bad("缺 README.md")

    # 复现脚本必须在
    repro = ROOT / "scripts" / "reproduce_all.py"
    if repro.exists():
        r.good("一键复现脚本 scripts/reproduce_all.py 在")
    else:
        r.caution("没有 scripts/reproduce_all.py（终局要求「一键跑通 11 阶段」）")


# ----------------------------------------------------------------------
def main() -> int:
    print("=" * 72)
    print("交付物自检：只读产物，不看生成器代码")
    print("=" * 72)
    r = Report()
    check_jsons(r)
    check_report(r)
    check_cross_refs(r)
    check_figs(r)
    check_hardcoded_numbers(r)
    check_imports(r)
    check_source_placeholders(r)
    check_docs(r)
    check_key_figures(r)
    check_repeated_quotes(r)

    print(f"\n{'=' * 72}")
    print(f"通过 {r.ok} ／ 警告 {r.warn} ／ 失败 {r.fail}")
    if r.fail:
        print("[FAIL] 有失败项——交付物不一致，不能就这样交出去。")
        return 1
    if r.warn:
        print("[WARN] 有警告项（不阻塞，但要看一眼）。")
    else:
        print("[OK] 全部通过。")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "scripts"))
    raise SystemExit(main())
