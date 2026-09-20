"""批量替换安全工具 —— 工程纪律第八条的**强制执行入口**。

为什么必须存在这个东西
----------------------
本项目**两次**栽在同一类事故上：为了修文案里的裸引号，写了个按行替换的脚本，
结果把源码写坏。
  · 第一次：18 行被改坏（包括参数分隔符）。
  · 第二次（分辨力危机那一轮）：模块/函数 docstring 的 `\"\"\"` 被改成 `"「"`，
    字典键被改成 `"k「: r[」k_point「]`——而且它**不会立刻报错**，
    是在继续制造新的语法错误。

⇒ 第八条纪律：**任何批量文本/代码替换，必须先在临时副本上执行，
  并对所有受影响的 .py 跑 ast.parse 全部通过之后，才允许写回真实文件。**

设计要点（以及任务书伪代码里被修正的两个缺陷）
--------------------------------------------
1. ⚠️ **临时文件不能只用 basename**。任务书的伪代码写的是
   ``Path(tmpdir) / Path(fp).name`` —— 两个不同目录下的同名文件会**互相覆盖**，
   于是"校验过的内容"和"写回的内容"可能对不上（静默错误）。
   本实现按**相对路径镜像目录结构**。
2. **要么全部成功、要么全部不生效**：任何一个文件语法失败就整体回滚，
   不写回任何文件。部分写入比不写更危险（代码库处于半改状态）。
3. ``dry_run`` 默认 **True**：先看 diff 与校验结果，确认无误再显式关掉。

用法::

    python scripts/safe_batch_replace.py --dry-run            # 只报告
    python scripts/safe_batch_replace.py --apply              # 确认后写回
（脚本自身用法见 --help；库用法见 ``safe_batch_replace``。）
"""

from __future__ import annotations

import argparse
import ast
import difflib
import re
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import ROOT, banner  # noqa: E402


class UnsafeReplaceError(Exception):
    """替换会破坏语法（或输入非法）——操作已整体回滚，没有文件被修改。"""


#: 只对这些后缀做语法校验（其它文件按文本处理）
PY_SUFFIXES = (".py",)


def _rel_key(fp: Path, root: Path) -> Path:
    """把绝对路径转成相对 root 的路径；不在 root 下就退回文件名。

    用它做临时目录里的镜像路径 —— **不要用 ``Path(fp).name``**（会碰撞）。
    """
    try:
        return fp.resolve().relative_to(root.resolve())
    except ValueError:
        return Path(fp.name)


def _apply_replacements(text: str, replacements: list[tuple[str, str]]) -> str:
    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text)
    return text


def safe_batch_replace(file_paths: list[str | Path],
                       replacements: list[tuple[str, str]],
                       *, dry_run: bool = True,
                       root: Path | str | None = None,
                       check_syntax: bool = True) -> dict:
    """对一组文件批量做正则替换，**带语法护栏**。

    参数
    ----
    file_paths
        要改的文件（绝对或相对路径）。
    replacements
        ``[(pattern, replacement)]``，按顺序用 ``re.sub`` 应用。
    dry_run
        默认 ``True``：只报告 diff 与校验结果，**不写回**。
    root
        计算相对路径用的根（默认取 ``DEFAULT_ROOT``）。
    check_syntax
        是否对 ``.py`` 跑 ``ast.parse``（默认开；关掉只应用于纯文本场景）。

    返回
    ----
    ``{"applied", "changed", "diffs", "syntax_errors", "unchanged"}``

    异常
    ----
    任何 ``.py`` 语法校验失败 ⇒ 抛 ``UnsafeReplaceError``，**一个文件都不写回**。
    """
    root_p = Path(root) if root is not None else ROOT
    paths = [Path(p) for p in file_paths]
    for p in paths:
        if not p.is_file():
            raise UnsafeReplaceError(f"文件不存在：{p}")

    diffs: dict[str, list[str]] = {}
    syntax_errors: dict[str, str] = {}
    changed: list[str] = []
    unchanged: list[str] = []
    staged: dict[str, str] = {}          # rel_key -> 新内容

    with tempfile.TemporaryDirectory(prefix="safe_replace_") as tmpdir:
        tmp_root = Path(tmpdir)
        for fp in paths:
            original = fp.read_text(encoding="utf-8")
            modified = _apply_replacements(original, replacements)
            key = str(fp)
            if modified == original:
                unchanged.append(key)
                continue
            changed.append(key)
            diffs[key] = list(difflib.unified_diff(
                original.splitlines(), modified.splitlines(),
                fromfile=f"a/{_rel_key(fp, root_p)}",
                tofile=f"b/{_rel_key(fp, root_p)}", lineterm=""))
            if check_syntax and fp.suffix in PY_SUFFIXES:
                try:
                    ast.parse(modified)
                except SyntaxError as e:
                    syntax_errors[key] = f"第 {e.lineno} 行：{e.msg}"
            staged[str(_rel_key(fp, root_p))] = modified
            # 写到临时目录（镜像相对路径，避免同名碰撞）
            tgt = tmp_root / _rel_key(fp, root_p)
            tgt.parent.mkdir(parents=True, exist_ok=True)
            tgt.write_text(modified, encoding="utf-8")

        if syntax_errors:
            raise UnsafeReplaceError(
                "以下文件语法校验失败 ⇒ 操作已整体回滚，没有文件被修改：\n  "
                + "\n  ".join(f"{k}: {v}" for k, v in syntax_errors.items()))

        applied = False
        if not dry_run and changed:
            # 全部校验通过才写回（用 staged 的内容，不是再读一次临时文件——
            # 少一次 IO 往返，也就少一个"写回的不是校验过的那份"的机会）
            for fp in paths:
                key = str(fp)
                if key in changed:
                    fp.write_text(staged[str(_rel_key(fp, root_p))],
                                  encoding="utf-8")
            applied = True

    return {"applied": applied, "changed": changed, "unchanged": unchanged,
            "diffs": diffs, "syntax_errors": syntax_errors}


# ======================================================================
# 命令行：给"手工写替换规则"留一个安全带（而不是每次现写脚本）
# ======================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="带语法护栏的批量替换")
    ap.add_argument("files", nargs="+", help="要处理的文件")
    ap.add_argument("--pattern", required=True, action="append",
                    help="正则（可重复，按出现顺序应用）")
    ap.add_argument("--repl", required=True, action="append",
                    help="替换串（与 --pattern 一一对应）")
    ap.add_argument("--apply", action="store_true",
                    help="**确认无误后**才加这个开关写回；默认只 dry-run")
    args = ap.parse_args()

    if len(args.pattern) != len(args.repl):
        print("❌ --pattern 与 --repl 数量必须一致", file=sys.stderr)
        return 2

    banner("批量替换（第八条纪律：先临时副本 + 语法校验）")
    reps = list(zip(args.pattern, args.repl))
    try:
        res = safe_batch_replace(args.files, reps, dry_run=not args.apply)
    except UnsafeReplaceError as e:
        print(f"❌ {e}", file=sys.stderr)
        return 1

    print(f"  会改动 {len(res['changed'])} 个文件；未变 {len(res['unchanged'])} 个")
    for key in res["changed"]:
        print(f"\n  ── {key}")
        for line in res["diffs"][key][:40]:
            print(f"     {line}")
    if res["applied"]:
        print(f"\n  ✅ 已写回 {len(res['changed'])} 个文件")
    else:
        print("\n  （dry-run：没有写回任何文件。确认无误后加 --apply）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
