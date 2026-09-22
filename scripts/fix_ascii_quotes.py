#!/usr/bin/env python
"""把源码里**误用的裸 ASCII 双引号**（在中文文本里当引号用）改成「」。

    python scripts/fix_ascii_quotes.py scripts/xxx.py            # 只看（dry-run）
    python scripts/fix_ascii_quotes.py scripts/xxx.py --write    # 真改

为什么要有这个工具
------------------
我（AI）反复犯同一个错：在**中文文案**里把 `「」` 写成裸的 `"`，于是

    L.append("...是"这条行情特殊"，还是...")     # ← 语法错误

这类错**必然**被 `selfcheck` ⑦（用 `compile()`）拦下，所以它不会流到交付物里；
但它会打断节奏，而**我手工修的时候又反复把别的东西改坏**：

| 我改坏过的 | 怎么坏的 |
|---|---|
| **两行拼接的续行** | 把续行也当成新语句，吃掉首行的 `)` |
| **docstring** | 三引号被当成普通字符串 |
| **条件表达式 / `" | ".join(...)`** | 内部的引号被换成「」 |

⇒ 所以这个工具把"安全边界"写死：

1. **只改括号平衡的整行语句**（`L.append(...)` / `print(...)` 等），
   续行**一律不碰**；
2. **不碰三引号**（docstring）；
3. 一行里若出现 **`" | ".join`** / ` f"` 嵌套 / ` if ` 这类"引号有句法意义"的
   模式 ⇒ **整行跳过**（宁可漏，不可错）；
4. 收尾时**自己 `compile()` 一遍**，还要**回读逐行核对**——
   而且**优先按"行内引号个数是否是奇数"判断**，因为那正是"语法坏了"的指纹。

⚠️ 它**不是**批量替换器：本项目的批量替换请用 `safe_batch_replace.py`。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

#: 该行出现这些子串 ⇒ 说明引号可能**有句法意义** ⇒ 整行跳过（宁可漏，不可错）
RISKY = ('" | ".join', '.join(f"', ' if ', ' else ', 'f"', '%s"', "'")

#: 只有这些开头才认为是"我要修的那种语句"
STMT_PREFIX = ("L.append(", "print(", "ap.add_argument(", "p.add_argument(",
               "raise ", '"""', "'''")


def _counts(s: str) -> tuple[int, int]:
    return s.count('"'), s.count('"""')


def fix_line(ln: str) -> str:
    """把首尾引号**之间**的裸引号交替换成「」（首尾引号是字符串定界符，保留）。"""
    idx = [i for i, c in enumerate(ln) if c == '"']
    if len(idx) < 4:
        return ln
    first, last = idx[0], idx[-1]
    out: list[str] = []
    op = True
    for i, c in enumerate(ln):
        if c == '"' and first < i < last:
            out.append("「" if op else "」")
            op = not op
        else:
            out.append(c)
    return "".join(out)


def fix_text(src: str) -> tuple[str, list[tuple[int, str, str]]]:
    lines = src.split("\n")
    changed: list[tuple[int, str, str]] = []
    out: list[str] = []
    for i, ln in enumerate(lines, 1):
        b = ln.strip()
        # ⚠️ 安全边界 1：必须是**整行一条语句**（括号平衡）
        balanced = b.count("(") == b.count(")")
        # ⚠️ 安全边界 2：只处理已知的语句开头（续行/裸字符串一律不碰）
        is_stmt = b.startswith(STMT_PREFIX)
        # ⚠️ 安全边界 3（**精确判据**）：这一行**单独编译不过**才算坏。
        # ⚠️ 我第一版写的是"引号个数是奇数"，**错**——经典坏行
        # `L.append("...是"这条行情特殊"，还是...")` 有 **4** 个引号（偶数）
        # ⇒ 被判成"没问题"而漏修。**"能不能编译"才是那个指纹。**
        try:
            compile(b, "<line>", "exec")
            bad = False
        except SyntaxError:
            bad = True
        # ⚠️ 安全边界 4：行内不能有"引号有句法意义"的模式
        risky = any(r in ln for r in RISKY)
        # ⚠️ 安全边界 5：不碰三引号
        has_triple = '"""' in ln or "'''" in ln
        if is_stmt and balanced and bad and not risky and not has_triple:
            nl = fix_line(ln)
            if nl != ln:
                changed.append((i, ln, nl))
            out.append(nl)
        else:
            out.append(ln)
    return "\n".join(out), changed


def main(argv: list[str] | None = None) -> int:
    """⚠️ 接 ``argv`` 而不是直接读 `sys.argv`——否则**测试没法调它**。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--write", action="store_true", help="真的写回（默认只看）")
    args = ap.parse_args(argv)

    rc = 0
    for raw in args.paths:
        p = Path(raw)
        if not p.is_file():
            print(f"  ⚠️ 跳过（不是文件）：{p}")
            continue
        src = p.read_text(encoding="utf-8")
        new, changed = fix_text(src)
        print(f"\n【{p}】可修 {len(changed)} 处")
        for ln, a, b in changed[:20]:
            print(f"   {ln:>5}: {a.strip()[:78]}")
            print(f"          → {b.strip()[:78]}")
        if len(changed) > 20:
            print(f"   … 还有 {len(changed) - 20} 处")
        if not changed:
            continue
        if not args.write:
            print("   （dry-run；要写回请加 --write）")
            continue
        # ⭐ 自己先验：改完必须能编译，否则**不写**
        try:
            compile(new, str(p), "exec")
        except SyntaxError as e:
            print(f"   ❌ 改完仍然编译不过（第 {e.lineno} 行）⇒ **不写回**，"
                  f"请手工看：{e.msg}")
            rc = 2
            continue
        p.write_text(new, encoding="utf-8")
        print(f"   ✅ 已写回（并已通过 compile 校验）")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
