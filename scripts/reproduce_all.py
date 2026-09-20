"""一键复现：把全部 11 个阶段跑通，再重出两份报告，最后自检。

为什么要有这个文件
------------------
二期指导书的终局硬要求之一是「完整复现脚本能一键跑通全部 11 个阶段」。
只有分步命令是不够的——分步命令**没有顺序保证**、**没有失败即停**、
**没有产物新鲜度检查**。真实踩过的坑：

· 某个阶段脚本因为缺一个 import 直接 NameError 崩了，但其余阶段照跑，
  报告把那个阶段静默降级成「尚未运行」，**没人发现**；
· 阶段脚本引用了磁盘缓存，改了口径却命中陈旧缓存，
  产出一个「看起来正常但过时」的数字，比没有缓存危险得多。

所以本脚本做三件事：**按依赖顺序跑 / 任一环节失败立即停 / 跑完对账**。

对账（``--verify``，默认开）会检查：
  1. 每个阶段的 JSON 存在，且 mtime **比本次开跑时间新**（防止拿旧产物冒充）；
  2. JSON 可解析、``stage`` 字段自洽、二期阶段带 ``honest_boundary``；
  3. 两份报告的 mtime 也新于开跑时间。

用法::

    python scripts/reproduce_all.py                  # 全部（见 --list 的实测估时）
    python scripts/reproduce_all.py --from 5         # 只跑二期（阶段5 起）
    python scripts/reproduce_all.py --only 5,8       # 只跑指定阶段（调试用）
    python scripts/reproduce_all.py --no-report      # 不重出报告
    python scripts/reproduce_all.py --with-mutation  # 连变异验证一起跑
    python scripts/reproduce_all.py --list           # 看步骤清单与**实测**估时

⚠️ 估时来自 ``out/repro_timings.json``（上次跑出来的真实耗时），不是手写的。
手写估时的下场：声明 70 分钟、实测接近三小时，而"看起来卡死了"的错觉
只有真的量过才知道是错觉。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
DOCS = ROOT / "docs"
LOGS = OUT / "repro_logs"

# 步骤定义：(名字, 命令参数, 归属期, 大概耗时秒)
# 顺序即依赖顺序：calibrate 必须最先（后面阶段读 out/calibration.json）；
# 一期四个阶段必须在 make_report 之前；二期七个阶段在 make_report2 之前。
STEPS: list[tuple[str, list[str], str, int]] = [
    ("calibrate",      ["scripts/calibrate.py"],      "一期", 180),
    ("stage1",         ["scripts/run_stage1.py"],     "一期", 60),
    ("stage2",         ["scripts/run_stage2.py"],     "一期", 120),
    ("stage3",         ["scripts/run_stage3.py"],     "一期", 300),
    ("stage4",         ["scripts/run_stage4.py"],     "一期", 240),
    ("report1",        ["scripts/make_report.py"],    "一期", 60),
    ("stage5",         ["scripts/run_stage5.py"],     "二期", 300),
    ("stage6",         ["scripts/run_stage6.py"],     "二期", 900),
    ("stage7",         ["scripts/run_stage7.py"],     "二期", 300),
    ("stage8",         ["scripts/run_stage8.py"],     "二期", 600),
    ("stage9",         ["scripts/run_stage9.py"],     "二期", 120),
    ("stage10",        ["scripts/run_stage10.py"],    "二期", 300),
    ("stage11",        ["scripts/run_stage11.py"],    "二期", 480),
    ("report2",        ["scripts/make_report2.py"],   "二期", 60),
    ("summary",        ["scripts/make_stage_summary.py"], "二期", 30),
    ("selfcheck",      ["scripts/selfcheck.py"],      "对账", 60),
]

# 每个步骤必须产出的文件（用于新鲜度对账）
EXPECT_JSON = {f"stage{i}": f"stage{i}_metrics.json" for i in range(1, 12)}
EXPECT_JSON["stage1"] = "stage1_metrics.json"
EXPECT_JSON["calibrate"] = "calibration.json"
EXPECT_HTML = {"report1": "交易世界-交付报告.html",
               "report2": "交易世界二期-交付报告.html",
               "summary": "二期阶段小结.md"}

TESTS = ("tests", ["-m", "unittest", "discover", "-s", "tests", "-t", "."],
         "对账", 400)
MUTATION = ("mutation", ["scripts/mutation_check.py"], "对账", 900)

# 收尾步骤必须排在**实验之后**，而且顺序有讲究：
#   tests → mutation → report1/report2 → selfcheck
# 为什么不能让报告先跑：
#   · 报告里要引用"当前测试数与变异验证结论"，而这两件事在最后才完成。
#     报告先跑的话，引用的会是上一轮的陈旧数字——而且**看不出来**。
#   · 真实踩过：``out/test_count.txt`` **根本没有东西去写它**，
#     是手工维护的 480。新增 19 条测试之后它不会变，报告会照抄 480。
#     现在测试步骤自己把 ``Ran N tests`` 解析出来写回去。
TAIL_ORDER = {"tests": 1, "mutation": 2, "report1": 3, "report2": 4,
              "summary": 5, "selfcheck": 6}


def step_order(step: tuple) -> tuple[int, int]:
    """实验步骤保持声明顺序在前，收尾步骤按 ``TAIL_ORDER`` 排在后。"""
    return (1, TAIL_ORDER[step[0]]) if step[0] in TAIL_ORDER else (0, 0)


def sync_test_count(log: Path) -> str | None:
    """从 unittest 日志里抠出真实测试数，写进 ``out/test_count.txt``。

    让"报告引用的测试数"变成**推导出来的**，而不是有人记得去改的。
    这类"没人维护的常量"是报告与实测脱节的经典入口。
    """
    try:
        txt = log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    m = re.search(r"^Ran (\d+) tests?", txt, re.M)
    if not m:
        return None
    n = m.group(1)
    (OUT / "test_count.txt").write_text(n, encoding="utf-8")
    return n


def py() -> str:
    """用当前解释器——避免 PATH 里另一个 python 混进来。"""
    return sys.executable


def run_step(name: str, argv: list[str], phase: str) -> tuple[bool, float]:
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / f"{name}.log"
    t0 = time.time()
    # ⚠️ ``-u`` 是必须的。子进程的 stdout 被重定向到文件时，Python 会切换成
    # **块缓冲**（约 8 KB）——于是：
    #   · 长步骤跑到一半时日志是 0 字节，"在算"和"卡死"看起来一模一样；
    #   · **最要命的是崩溃时**：缓冲区里的输出随进程一起消失，
    #     而那正是唯一能说明"它死在哪一步"的东西。
    # 真实踩过：calibrate 跑了 11 分钟、日志 0 字节，无从判断状态。
    cmd = [py(), "-u", *argv]
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    print(f"\n  → [{'·'.join([phase, name])}] {' '.join(cmd)}")
    print(f"    日志：{log.relative_to(ROOT)}")
    with log.open("w", encoding="utf-8", errors="replace") as fh:
        p = subprocess.run(cmd, cwd=str(ROOT), stdout=fh,
                           stderr=subprocess.STDOUT, env=env)
    dt = time.time() - t0
    if p.returncode == 0:
        print(f"    ✅ 通过（{dt:.0f}s）")
        return True, dt
    print(f"    ❌ 失败（{dt:.0f}s，退出码 {p.returncode}）")
    tail = log.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    print("    ── 日志末尾 15 行 ──")
    for line in tail[-15:]:
        print(f"    | {line}")
    return False, dt


def verify(t0: float, ran: set[str]) -> list[str]:
    """对账：产物是否齐全、自洽，以及**跑过的那些步骤**是否产出了新文件。

    ⚠️ 新鲜度只针对**本次真跑过的步骤**。
    第一版对所有产物都要求 mtime 新于开跑时间——于是 ``--skip`` 掉一个
    已知新鲜的阶段时，会被自己的对账判为失败。对账口径必须和"本次
    声称做了什么"对齐，否则它会逼你为了通过检查而白跑一小时。
    """
    problems: list[str] = []
    for name, fn in EXPECT_JSON.items():
        p = OUT / fn
        if not p.exists():
            problems.append(f"缺 {fn}")
            continue
        # 只有本次跑过的步骤才要求"新"
        if name in ran and p.stat().st_mtime < t0 - 1:
            problems.append(f"{fn} 本次跑过，但 mtime 比开跑还早（产物没被更新）")
    for name, fn in EXPECT_HTML.items():
        p = DOCS / fn
        if not p.exists():
            problems.append(f"缺 docs/{fn}")
            continue
        if name in ran and p.stat().st_mtime < t0 - 1:
            problems.append(f"docs/{fn} 本次跑过，但没被重出")

    # 每个阶段 JSON 自洽 + 二期带诚实边界（与新旧无关，一律查）
    for i in range(1, 12):
        p = OUT / f"stage{i}_metrics.json"
        if not p.exists():
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            problems.append(f"stage{i}_metrics.json 解析失败：{e}")
            continue
        if d.get("stage") != i:
            problems.append(f"stage{i} 的 stage 字段是 {d.get('stage')!r}")
        if i >= 5 and not d.get("honest_boundary"):
            problems.append(f"stage{i} 缺 honest_boundary")
    return problems


ARCHIVE_ROOT = OUT / "_archive"


def archive_before_overwrite(stamp: str | None = None) -> Path | None:
    """把现有的 JSON 产物与报告复制一份到 ``out/_archive/<时间戳>/``。

    为什么要做这件事（真实代价）：重跑会**原地覆盖** ``out/*.json`` 与
    ``docs/*.html``。一旦覆盖，"修复前 vs 修复后"就再也比不了了——
    本轮就吃了这个亏：改完配对交易的做空额度后重跑阶段9，
    想对比 E9.1 标定表的新旧值，才发现旧的 ``stage9_metrics.json``
    已经被覆盖，只能凭之前零散打印在终端里的几行回忆。

    **"能对比"这件事本身就是证据的一部分。** 快照只要几 MB，
    而失去它之后"这次改动到底改变了什么"只能靠记忆回答。
    """
    stamp = stamp or time.strftime("%Y%m%d-%H%M%S")
    dst = ARCHIVE_ROOT / stamp
    try:
        dst.mkdir(parents=True, exist_ok=True)
        n = 0
        for p in sorted(OUT.glob("*.json")):
            shutil.copy2(p, dst / p.name)
            n += 1
        for p in sorted(DOCS.glob("*.html")) + sorted(DOCS.glob("*.md")):
            shutil.copy2(p, dst / p.name)
            n += 1
    except OSError as e:                                  # pragma: no cover
        print(f"  ⚠️ 快照失败（不阻塞）：{e}")
        return None
    print(f"  🗄️ 已把现有产物快照到 {dst.relative_to(ROOT)}（{n} 个文件）"
          f"—— 重跑后仍可做「改前 vs 改后」对比")
    return dst


def normalize_skip(raw: str | None) -> set[str]:
    """把 ``--skip`` / ``--only`` 的写法统一成步骤名。

    接受两种写法，因为两种都很自然、而只支持一种必然被写错：
      ``8,9,11``        → ``{"stage8","stage9","stage11"}``
      ``stage8,tests``  → 原样

    ⚠️ 这个函数被**两个**坑逼出来过：
      · 第一版只认步骤名，``--skip 8,9,11`` 静默什么都不跳；
      · 修了 ``--skip`` 之后忘了 ``--only``，于是 ``--only 5,8``
        静默选中零个步骤——**什么也不跑还报成功**。
    检查/选择工具"看起来在工作但其实没在看"，是最坏的一类。
    """
    if not raw:
        return set()
    out = set()
    for s in raw.split(","):
        s = s.strip()
        if not s:
            continue
        out.add(f"stage{s}" if s.isdigit() else s)
    return out


TIMINGS_PATH = OUT / "repro_timings.json"


def load_timings() -> dict[str, float]:
    """读取上次实测的每步耗时。

    为什么不留"声明式估时"：第一版每个步骤手写了一个"约耗时"，
    加起来 70 分钟。而实测 calibrate 就跑了 747 秒（声明写的是 180s），
    整条流水线接近三小时。
    **一个从来没被测量过的时间估计，会在别人以为卡死的时候显得特别刺眼**，
    而且没人会去怀疑它。所以估时改成"上次实测值"，没有实测值时
    才退回声明的数量级。
    """
    if not TIMINGS_PATH.exists():
        return {}
    try:
        d = json.loads(TIMINGS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    # ⚠️ 文件是 ``{"measured_at": ..., "seconds": {...}}`` 的外壳，
    #    真正的耗时在 ``seconds`` 里。第一版忘了拆这一层，直接遍历顶层，
    #    于是 ``measured_at``（str）和 ``seconds``（dict）都不是数字 →
    #    返回空字典 → ``--list`` 永远显示"还全是估计值"。
    #    **保存端与读取端对同一份文件的形状理解不一致**，是这类问题的通用形态。
    table = d.get("seconds") if isinstance(d.get("seconds"), dict) else d
    return {k: float(v) for k, v in table.items()
            if isinstance(v, (int, float))}


def save_timings(done: dict[str, float]) -> None:
    TIMINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    TIMINGS_PATH.write_text(
        json.dumps({"measured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "seconds": done}, ensure_ascii=False, indent=2),
        encoding="utf-8")


def estimate(step: tuple, measured: dict[str, float]) -> float:
    return measured.get(step[0], float(step[3]))


def sync_canonical_log(name: str) -> None:
    """把 ``out/repro_logs/<name>.log`` 复制到它**被文档化的规范路径**。

    ⚠️ 这是一个真实差点出事的地方：本脚本把每一步的日志写在
    ``out/repro_logs/``，而报告生成器与自检脚本读的是
    ``out/mutation_run.log``（历史路径）。跑完本轮之后，
    「变异验证 33/33 全绿」写在新路径，而报告会照旧引用**上一次**的
    ``out/mutation_run.log``——于是报告里的结论会是上一轮的 29/29，
    而且它看起来完全正常，因为**那个文件一直都在、内容也是真的**。

    这正是本项目反复踩到的同一类问题：**同一个产物有两条路径，
    就没有任何机制保证它们说的是同一件事。**
    所以这里不只是复制，后面还让读者端取"两者中较新的那个"——
    两层保险，任一层失效另一层仍能兜住。
    """
    src = LOGS / f"{name}.log"
    dst = OUT / f"{name}_run.log"
    if not src.exists():
        return
    try:
        shutil.copy2(src, dst)
    except OSError:                                      # pragma: no cover
        return
    print(f"    ↳ 已同步到规范路径 {dst.name}（报告与自检读这里）")


def plan_steps(args) -> list[tuple[str, list[str], str, int]]:
    """按依赖与收尾顺序算出本次要跑哪些步骤。

    ``--list`` 与实际执行共用这一个函数——否则"计划里显示的"和
    "实际会跑的"会各写一份，然后不一致（而清单是给人看的，不一致时
    人只会相信清单）。
    """
    only = normalize_skip(args.only) or None
    skip = normalize_skip(args.skip)

    def want(name: str) -> bool:
        if only is not None:
            # --only 是调试用：只跑点名的实验步骤，不自动补收尾步骤
            return name in only
        if name in skip:
            return False
        if args.no_report and name in ("report1", "report2"):
            return False
        if args.no_selfcheck and name == "selfcheck":
            return False
        if args.no_tests and name == "tests":
            return False
        if not args.with_mutation and name == "mutation":
            return False
        if args.frm is not None and name.startswith("stage"):
            try:
                if int(name[5:]) < args.frm:
                    return False
            except ValueError:
                pass
        return True

    steps = [(n, a, p, s) for (n, a, p, s) in STEPS if want(n)]
    # ⚠️ ``tests`` / ``mutation`` **不在 STEPS 里**（它们是"收尾步骤"而非实验步骤），
    #    所以 ``--only`` 点名它们时要单独处理。这里真实踩过一次：
    #    第一版只在 ``only is None`` 时才追加这两个，于是
    #    ``--only mutation --with-mutation --list`` 打印「共 0 步」——
    #    看起来像"没有步骤要跑"，实际是把用户点名的步骤**静默丢掉了**。
    #    与 ``--skip 8,9,11`` 静默不跳是同一形态：**用户说了，程序没听见，还不吱声。**
    #    回归测试：tests/test_stage_scripts.py::TestReproduceOnlyNotSilentlyIgnored
    for tail in (TESTS, MUTATION):
        name = tail[0]
        if only is None:
            if want(name):
                steps.append((name, tail[1], tail[2], tail[3]))
        elif name in only:
            # 变异验证是重活（1~2 小时），默认闸门仍然保留，但**必须开口**，
            # 否则用户会以为"我点名了但它没跑"是 bug 之外的别的原因。
            if name == "mutation" and not args.with_mutation:
                print(f"  ⚠️ --only 点名了 {name}，但没给 --with-mutation。"
                      "变异验证默认不跑（约 1~2 小时）。要跑请补上 --with-mutation。")
                continue
            steps.append((name, tail[1], tail[2], tail[3]))
    if not steps:
        print(f"  ⚠️ 本次一个步骤都没有：--only={args.only!r} --skip={args.skip!r}。"
              "（--only 的合法取值是实验步骤名或 tests/mutation）")
    steps.sort(key=step_order)
    return steps


def main() -> int:
    ap = argparse.ArgumentParser(description="交易世界：一键复现全部阶段")
    ap.add_argument("--from", dest="frm", type=int, default=None,
                    help="从哪个阶段号开始跑（含），如 5")
    ap.add_argument("--only", default=None,
                    help="只跑指定阶段，逗号分隔，如 5,8（调试用，不自动补报告）")
    ap.add_argument("--skip", default=None,
                    help="跳过指定步骤名，逗号分隔，如 8,9,11。"
                         "用于「已知某阶段产物比库代码还新」的场景——"
                         "对账会只对本次真跑过的步骤要求新鲜度")
    ap.add_argument("--no-report", action="store_true", help="不重出报告")
    ap.add_argument("--no-tests", action="store_true", help="跳过测试")
    ap.add_argument("--no-selfcheck", action="store_true", help="跳过自检")
    ap.add_argument("--with-mutation", action="store_true",
                    help="连变异验证一起跑（上次实测约 22 分钟，见 --list）")
    ap.add_argument("--list", action="store_true", help="只列步骤，不执行；"
                    "可与 --skip / --with-mutation 组合，先看计划再动手")
    args = ap.parse_args()

    if args.list:
        pl = plan_steps(args)
        measured = load_timings()
        print(f"{'步骤':<14}{'期':<6}{'上次实测':>10}  命令")
        for name, argv, phase, sec in pl:
            est = estimate((name, argv, phase, sec), measured)
            tag = "" if name in measured else "（估）"
            print(f"{name:<14}{phase:<6}{est:>9.0f}s{tag}  "
                  f"python {' '.join(argv)}")
        tot = sum(estimate(s, measured) for s in pl)
        print(f"\n共 {len(pl)} 步，预计 {tot / 60:.0f} 分钟"
              f"{'' if measured else '（还全是估计值，跑一次就会有实测）'}")
        return 0

    steps = plan_steps(args)
    measured = load_timings()

    print("=" * 74)
    print("交易世界 · 一键复现")
    print(f"解释器：{py()}")
    ests = [estimate(s, measured) for s in steps]
    print(f"共 {len(steps)} 步，预计 {sum(ests) / 60:.0f} 分钟"
          f"（{len(measured)} 步有上次实测值，其余为声明估时）")
    print("=" * 74)
    print("步骤：" + " → ".join(n for n, *_ in steps))

    # 先给现有产物留一份快照，**再**开始覆盖。
    # 顺序不能反：本轮真实吃过亏——重跑覆盖之后，"改前 vs 改后"
    # 就只剩记忆可依了。
    archive_before_overwrite()

    t0 = time.time()
    results: list[tuple[str, bool, float]] = []
    done_times: dict[str, float] = {}
    for name, argv, phase, _est in steps:
        ok, dt = run_step(name, argv, phase)
        results.append((name, ok, dt))
        done_times[name] = dt
        save_timings(done_times)      # 每步都落盘：中途崩了也留下已测部分
        if not ok:
            print(f"\n{'=' * 74}")
            print(f"❌ 在 {name} 处停下——后面的步骤依赖它，继续跑只会污染产物。")
            print_failures(results)
            return 1
        # 测试跑完就把真实测试数写回 out/test_count.txt：
        # 报告要引用它，而"手工维护的常量"必然会与实测脱节。
        if name == "tests":
            n = sync_test_count(LOGS / "tests.log")
            if n:
                print(f"    ↳ 已把真实测试数 {n} 写入 out/test_count.txt")
        if name == "mutation":
            sync_canonical_log("mutation")

    ran = {n for n, *_ in steps}
    print(f"\n{'=' * 74}")
    print("全部步骤通过，开始对账（产物齐全性 + 本次步骤新鲜度 + 自洽性）")
    print("=" * 74)
    problems = verify(t0, ran)
    if problems:
        print("❌ 对账发现问题：")
        for p in problems:
            print(f"   · {p}")
        print_failures(results)
        return 1
    print(f"✅ 对账通过：11 个阶段 JSON 齐全、自洽，二期阶段都带 honest_boundary；"
          f"本次跑过的 {len(ran)} 步都产出了新文件。")
    # ⚠️ 这里必须自己算一次 ``skip``（``plan_steps`` 里那份是它的局部变量）。
    #    本轮真实踩到：这一行原本写的是裸 ``skip``，于是 **`--only` 路径**
    #    跑到最后一句抛 ``NameError``——注意症状有多坏：
    #    **所有步骤其实全跑完了、对账也通过了、test_count 也写回了**，
    #    唯独最后一行崩掉，退出码变成 1。也就是说
    #    「产物是对的」但「程序说自己失败了」——看退出码的人会重跑，
    #    看日志的人会以为成功，两边都不对。
    skipped = normalize_skip(args.skip)
    if skipped:
        print(f"   （跳过的步骤：{sorted(skipped)} —— 未做新鲜度要求）")
    print(f"\n总用时 {time.time() - t0:.0f}s")
    return 0


def print_failures(results: list[tuple[str, bool, float]]) -> None:
    bad = [(n, d) for n, ok, d in results if not ok]
    print(f"\n本次结果：{len(bad)} 个失败 / {len(results)} 步")
    for n, d in bad:
        print(f"   ❌ {n}（{d:.0f}s）日志见 out/repro_logs/{n}.log")


if __name__ == "__main__":
    raise SystemExit(main())
