"""抽出 index.html 里的内联脚本，交给 node 做语法检查。

⭐ 为什么要有这个检查（本轮踩过）：
我在 JS 里写注释时，结尾用了 Python docstring 的习惯（三个双引号）而不是 C 风格的
块注释结束符 ⇒ **注释永不闭合** ⇒ 整个 script 块解析失败 ⇒
**界面看着「正常」（左侧那一栏是静态 HTML 表格，不是 JS 画的），
但所有交互都死了** —— 而截图脚本只检查「图生成了」，**会通过**。

⭐ 写这个文件时我又犯了同一类错的另一个变种：**docstring 里写了那三个双引号本身**，
把自己的 docstring 提前闭合了。所以：**描述某个错误写法时，别把那个写法原样抄进来**。

所以：单文件前端必须有一个「脚本能被解析」的门禁。
node 不在时**跳过并说明**（不能因为缺工具就当失败，也不能当成通过）。
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HTML = ROOT / "gui" / "static" / "index.html"
TMP = ROOT / "out" / "_inline_check.js"

html = HTML.read_text(encoding="utf-8")
m = re.search(r"<script>(.*)</script>", html, re.S)
if not m:
    print("❌ index.html 里没有 <script> 块")
    sys.exit(1)
TMP.parent.mkdir(parents=True, exist_ok=True)
TMP.write_text(m.group(1), encoding="utf-8")
print(f"内联脚本 {len(m.group(1))} 字符 / {m.group(1).count(chr(10))} 行")

node = shutil.which("node") or r"C:\Users\87465\.workbuddy\binaries\node\versions\22.22.2-3\node.exe"
if not Path(node).is_file():
    print("⚠️ 找不到 node ⇒ **跳过**语法检查（不算通过）")
    sys.exit(2)

r = subprocess.run([node, "--check", str(TMP)], capture_output=True, text=True)
if r.returncode == 0:
    print("✅ 内联脚本语法通过（node --check）")
    sys.exit(0)
print("❌ 内联脚本有语法错误：")
print((r.stderr or "")[:900])
sys.exit(1)
