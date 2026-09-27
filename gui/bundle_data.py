"""打包时随包带上的数据 —— **唯一定义点**。

为什么单独一个模块
------------------
两边都要用这份清单，而两边都不能去 import `gui.api_doc`
（它拉 numpy / tw，代价大，且会在**构建期**把项目依赖引进来）：
  · `packaging/trading_world.spec` —— PyInstaller 打包时读它；
  · `tests/test_gui.py` —— 对照 `api_doc.DOC_REPORTS` 检查**覆盖率**。
所以清单住在这个**零依赖**模块里（只有两个元组 + 一个纯函数）。

⚠️ 漏掉一项的后果**不是报错**，而是那个功能在安装包里**静默失效**。
实测踩到（2026-09-28）：`out/a14`、`out/a15`、`out/a16` 三份结果 JSON
没打进包 ⇒ 「文档验证」页在安装包里永远显示"产物缺失"，
而开发机上一切正常 —— 因为开发机上 `out/` 就在那里。
⇒ 所以有 `covers()` + 测试：清单必须覆盖界面会读的每一个文件。

⭐ 划线的依据：**随包内容 vs 运行产物**
--------------------------------------
· `out/a14|a15|a16` + 三份 `docs/A1x-*.md` 是**随包内容**：
  它们是三份交付报告的验证结果，是这个产品**展示的对象**（固定内容）。
· `out/a4`、`out/a5`、`docs/llm-run-*` 是**运行产物**：
  Agent 留痕是"用这个工具跑出来的东西"，不是内置内容。
  全新安装里没有它们是**正常的**，前端也有诚实的空状态
  （`static/index.html`：「还没有可浏览的 Agent 运行」）。
  把它们打进去要多 ~31 MB，且会把开发期的 v2/v3 变体一起带上。
  ⇒ 不打。要带上就加进 `BUNDLE_DIRS`（一行）。
"""

from __future__ import annotations

#: 整个目录随包带上（相对项目根，正斜杠）
BUNDLE_DIRS: tuple[str, ...] = (
    "data",           # 行情：tw/realdata.py 靠相对位置找它
    "gui/static",     # 前端单文件：gui/server.py 靠相对位置找它
    "out/a14",        # ⭐ 文档验证页的数据源（缺了页面就是死的）
    "out/a15",
    "out/a16",
)

#: 单个文件随包带上（相对项目根，正斜杠）。
#: ⚠️ 这三份与 `gui/api_doc.py::DOC_REPORTS` 里的报告文件是同一批 ——
#: 故意留成"两处写、一处测"：测试会发现脱节（见 `test_打包清单覆盖…`）。
BUNDLE_FILES: tuple[str, ...] = (
    "docs/A14-个人交易决策系统方案-验证报告.md",
    "docs/A15-右尾策略完全手册-验证报告.md",
    "docs/A16-两文档策略接入系统-实测报告.md",
)


def covers(rel: str) -> bool:
    """相对项目根的路径 ``rel`` 会不会随包带上。"""
    p = rel.replace("\\", "/").lstrip("/")
    return p in BUNDLE_FILES or any(
        p == d or p.startswith(d + "/") for d in BUNDLE_DIRS
    )


__all__ = ["BUNDLE_DIRS", "BUNDLE_FILES", "covers"]
