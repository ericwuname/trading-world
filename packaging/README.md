# 打包与分发

把「交易世界」桌面端做成 Windows 可执行程序。

## 快速开始

```bash
# 0. 装打包工具（只做一次）
python -m pip install pyinstaller

# 1. 打包（约 40 秒，产物在 out/_package/dist/TradingWorld/）
python -m PyInstaller --noconfirm --clean \
    --distpath out/_package/dist --workpath out/_package/build \
    packaging/trading_world.spec
```

产物：

| 项 | 值 |
|---|---|
| 目录 | `out/_package/dist/TradingWorld/` |
| exe | `TradingWorld.exe`（约 11 MB，只是个启动器） |
| 总体积 | **约 133 MB**（Python 运行时 + numpy + scipy + pythonnet） |
| 系统要求 | Windows 10/11 x64 + **WebView2 运行时**（Win10/11 通常自带） |

## 两种分发方式

| 方式 | 文件 | 特点 |
|---|---|---|
| **免安装** | `TradingWorld-win64-免安装.zip`（约 53 MB） | 解压即可双击运行，不需装任何东西 |
| **安装** | `install.ps1` | **零依赖**，任何 Win10/11 都能跑，**不需要管理员**，装到 `%LOCALAPPDATA%\Programs\TradingWorld`，自动建开始菜单/桌面快捷方式 |
| **安装（正式）** | `installer.iss` | 用 Inno Setup 编译成标准 `setup.exe`（可装到 Program Files、有标准卸载项）。**前提：先装 Inno Setup**：`winget install JRSoftware.InnoSetup`，然后 `ISCC.exe packaging\installer.iss` |

`install.ps1` 已经能覆盖"我想要个安装版"的需求，且**现在就能用**；
`installer.iss` 是给"要交给别人、要标准卸载项"的场景准备的。

## ⚠️ 打包时踩到的坑（都已修，写下来免得再踩）

### 1. 入口脚本不能与包同名（这是最坑的一个）

**症状**：

```
ImportError: cannot import name 'desktop' from partially initialized
module 'gui' (most likely due to a circular import)
...\_internal\gui.py          ← 注意这个路径
```

**原因**：PyInstaller 会把**入口脚本**编译成运行时目录 `_internal/` **顶层**的同名模块。
入口若叫 `scripts/gui.py`，顶层就会出现一个 `gui` 模块，
而项目里还有一个 `gui/` **包** ⇒ `from gui import desktop` 解析到**入口自己**。

这个坑项目里**记录过一次**（`tests/test_cache_keys.py` 里写着
"`scripts/gui.py` 与仓库根下的 `gui/` 包同名"），源码环境靠
"先把 `scripts/` 从 sys.path 摘掉"绕开——但**打包环境绕不开**（入口一定在顶层）。

**修法**：打包入口用 `packaging/entry.py`（不与任何包同名）。
参数与逻辑的**唯一事实源**在 `gui/cli.py`，`scripts/gui.py` 与 `packaging/entry.py`
都只是薄壳——这样不会出现两份 argparse。

### 2. 路径计算**不用**打补丁（这点是好消息）

项目里到处是 `Path(__file__).resolve().parent.parent` 来定位 `data/`、`gui/static/`。
只要**保持项目目录结构**（`data/` 与 `gui/` 在运行时目录里的相对位置不变），
这些计算会**自动正确**，一行生产代码都不用改。

实测验证：打包后 `/api/real/BTCUSDT_1h` 能读到 **17520 根**真实数据。

### 2b. ⚠️ 但"**放哪些**"是坑：运行时要读的磁盘文件，静态分析发现不了

第一版 spec 只带了 `data/` 与 `gui/static/` ——
于是 `out/a14`、`out/a15`、`out/a16`（「文档验证」页的数据源）
**没进包**，那页在安装包里永远显示"产物缺失"，
而**开发机上一切正常**（那里 `out/` 就在手边）。
⇒ 只在本机跑测试是**发现不了**这类问题的，只有"打包后再实测"或"对照清单"能发现。

现在清单由 `gui/bundle_data.py` **唯一定义**（零依赖模块，spec 与测试读同一份）：

```python
# gui/bundle_data.py
BUNDLE_DIRS  = ("data", "gui/static",
                "out/a14", "out/a15", "out/a16",        # 文档验证页
                "out/a4", "out/a5",                     # Agent 决策页（运行产物）
                "docs/llm-run-20260921-a4", "docs/llm-run-20260921")
BUNDLE_FILES = ("docs/A14-….md", "docs/A15-….md", "docs/A16-….md")
```

**划线的依据：用户在安装包里点开每一个标签页，都该看到内容。**
`out/a4`、`out/a5`、`docs/llm-run-*` 是**运行产物**（不是内置内容），
第一版判过"不打"（理由：全新安装里没有它们是正常的，前端有空状态），
用户明确要求「运行产物也打进包」后改成**打** —— 代价 ~31 MB
（157 MB → 188 MB），换来八个页面全都不是空的。

⚠️ 这是**产品决定，不是技术决定**：技术侧的责任是把代价算清楚
（+31 MB / zip +12 MB）并留成**一行开关**，别自己替用户拍板。

`tests/test_gui.py::test_打包清单覆盖了界面会读的磁盘文件` 会对照
`api_doc.DOC_REPORTS` 检查覆盖面；
`test_agent页要扫描的目录也随包` 会对照 `agent_api.RUN_DIRS` 检查 ——
两边各写一份是故意的，**脱节必须被报出来**。

### 2c. ⚠️ 重建时先"回收"旧产物（环境会拦批量删除）

PyInstaller 的 `COLLECT` 要先把旧的 `dist/TradingWorld/`（**384 个文件**）
删掉才能重建，而本环境的**批量删除保护**会把它拦下来：

```
[safe-delete][SAFE_DELETE_BULK_CONFIRM_REQUIRED] {"count":384,"threshold":50,…}
```

⇒ 先把它**移入回收站**（项目自带的工具，不是永久删除），再打包：

```bash
python tools/recycle_paths.py out/_package/dist/TradingWorld
python -m PyInstaller --noconfirm --clean --distpath out/_package/dist \
    --workpath out/_package/build packaging/trading_world.spec
```

### 3. GUI 不需要可写的目录

`gui/` 下没有任何 `open`/`write_text`/`mkdir`——实验只在内存里跑，
结果通过 HTTP 返回、由用户自己导出 CSV。
⇒ 打包目录可以整个是只读的，也不用操心把 `out/` 映射到用户目录。

### 4. 别用 UPX

`upx=True` 常在杀软里被误报，且压缩后启动更慢。spec 里显式关掉了。

## 界面形态与 console 的取舍（**成对改动**）

**当前是 `console=False`（无黑窗）**。这是用户 2026-09-28 明确要求的，
但它**必须成对**做两件事 —— 只改 spec 会引入本项目最反对的那类失效：

| 改什么 | 不改会怎样 |
|---|---|
| `spec` 里 `console=False` | —— |
| `gui/desktop.py` 的失败提示改走 `_alert()`（无控制台时弹原生消息框） | `sys.stdout`/`sys.stderr` 变成 `None`，`print` 是**静默 no-op** ⇒ 三条失败路径（端口探测失败 / pywebview 加载不了 / 窗口创建失败）全部变成「**双击没反应**」 |
| `gui/cli.py` 加 `--token` | 启动日志（含带令牌 URL）也不再可读 ⇒ `verify_exe.py` 这类脚本**取不到 token**，只能放弃 `/api/*` 的验证 |

`_alert()` 的设计：有控制台就 `print`，**没有才弹框**（`ctypes.windll.user32.MessageBoxW`，
stdlib、**不写任何文件** —— 否则安装说明里"本程序不写用户数据"那句就不成立了）。
"弹不弹"由"有没有控制台"决定，而那个条件在测试里天然为假
⇒ 这条逻辑是**可测的**（见 `tests/test_gui.py::test_无控制台时失败要说给人听`），
变异体 **M131** 盯着它。

`--token` 是**仅供自动化**的固定令牌（默认仍是每进程随机，安全模型没让步）；
验证脚本一律带上它。

## 怎么验证打包结果（**不要只看"打包成功"**）

打包成功 ≠ 能运行。**已脚本化**（比手敲 curl 可靠，且能覆盖响应头/BOM 这类
"只看状态码看不出来"的判据）。三个脚本都在本目录、随仓库走：

```bash
# ① 把 exe 当黑盒打真接口（含真跑一个作业、导出 CSV、无 token 被拒）
python packaging/verify_exe.py

# ② 视觉验证：起服务 → 无头截图 → 然后**必须打开图看一眼**
python packaging/shot_exe.py out/_package/截图.png            # 首页
python packaging/shot_exe.py out/_package/截图-agent.png agent  # 指定标签页

# ③ 安装包：安装 → 运行 → 卸载（全静默，/NOICONS 不碰桌面与开始菜单）
python packaging/verify_setup.py
```

⚠️ 三个脚本都用 `--token` 固定令牌启动 ——
**无控制台的构建读不到启动日志**，不固定就抓不到 token。

`verify_exe.py` 覆盖：`/`（离线可用）→ `/api/meta` → `/api/doc/reports`
（**三份报告 JSON 与 md 都在包里**）→ `/api/agent/runs` → `/api/real/<sym>`
→ `POST /api/run` 真跑一个策略作业 → 查 `PnL 三分解恒等式` →
导出 CSV 查 `Content-Disposition` 与 BOM → 无 token 返回 403 → 未知接口 404。
⚠️ 它**故意换一个 cwd** 启动，用来证明"路径靠相对位置自动正确"
而不是碰巧 cwd 帮了忙。

手敲的话大致是：

```bash
cd out/_package/dist/TradingWorld

# 1. 只起服务不开窗口（--no-window 是给自动化验证用的）
./TradingWorld.exe --no-window --port 8795

# 2. 从启动日志里拿 token（token 在启动 URL 里，不在 HTML 里）
#    带令牌  http://127.0.0.1:8795/?token=XXXX

# 3. 打接口
curl -H "X-TW-Token: XXXX" http://127.0.0.1:8795/api/meta
curl -H "X-TW-Token: XXXX" http://127.0.0.1:8795/api/real/BTCUSDT_1h
curl -H "X-TW-Token: XXXX" http://127.0.0.1:8795/api/doc/reports

# 4. 跑一个真实实验
curl -X POST -H "X-TW-Token: XXXX" -H "Content-Type: application/json" \
  -d '{"scenario":"liquidation","seed":20260917,"ticks":3000,"strategy":"mm_skewed"}' \
  http://127.0.0.1:8795/api/run
# → 拿 job_id，轮询 /api/job/<id> 看 state 变 done
```

⚠️ **用 `curl -w "%{size_download}"` 判断响应大小会骗人**——
实测首页它报 `0 bytes`，而响应头明确写着 `Content-Length: 65869`。
要看就用 `-D -` 看响应头，或者干脆存成文件看大小。

## 生成分发产物

```bash
python packaging/make_dist.py
```

一条命令做三件事，并且**当场自检**（不看"命令返回 0"）：

| 产物 | 说明 |
|---|---|
| `out/_package/TradingWorld-v<版本>-win64-免安装.zip` | 解压即用；**校验 CRC + 顶层目录名 + 条目数**（解压出来必须是一个 `TradingWorld/` 文件夹，不能把 `_internal` 摊一地） |
| `out/_package/TradingWorld-Setup-<版本>.exe` | Inno Setup 正式安装包（可装 Program Files、有标准卸载项） |
| `packaging/install.ps1` | **零依赖**的 PowerShell 安装器（不需要 Inno Setup、不需要管理员） |

版本号从 `gui/__init__.py::__version__` 读，**不在这里再写一个**。

## ⚠️ 为什么打包产物里保留控制台窗口

`console=True` 是为了**看得见降级**：`gui/desktop.py` 的三条失败路径
（端口起不来 / pywebview 加载不了 / 窗口创建失败）全都靠 `print` 报出来。
改成 `False` 之后这些提示会**全部变成"双击没反应"**，
而这正是本项目最不想再踩的一类坑（静默失败）。
控制台还顺带把"带令牌地址"直接摆出来，方便排查与自动化验证。

要无黑窗版本：把 spec 里 `console=True` 改 `False`，同时给
`desktop.py` 的失败路径加一个 MessageBox（`ctypes.windll.user32`，stdlib）
——**两件事必须一起做**，只改前者会得到静默失败。

## 待办 / 可改进

- [ ] 加图标（`.ico`）。有图标后 spec 的 `EXE(icon=...)` 与 iss 的
      `SetupIconFile` 都能用上，看起来会正式很多。
- [x] ~~正式版把 `console=True` 改成 `False`~~（2026-09-28 已完成，
      配套改了 `desktop._alert` 与 CLI 的 `--token`，见上面那节）。
- [ ] 若要让没装 WebView2 的机器也能跑，可以把 WebView2 的固定版运行时
      一起打进去（代价：体积 +100~150 MB）。
