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

### 3. GUI 不需要可写的目录

`gui/` 下没有任何 `open`/`write_text`/`mkdir`——实验只在内存里跑，
结果通过 HTTP 返回、由用户自己导出 CSV。
⇒ 打包目录可以整个是只读的，也不用操心把 `out/` 映射到用户目录。

### 4. 别用 UPX

`upx=True` 常在杀软里被误报，且压缩后启动更慢。spec 里显式关掉了。

## 怎么验证打包结果（**不要只看"打包成功"**）

打包成功 ≠ 能运行。这个 spec 的验证方式是：

```bash
cd out/_package/dist/TradingWorld

# 1. 只起服务不开窗口（--no-window 是给自动化验证用的）
./TradingWorld.exe --no-window --port 8795

# 2. 从启动日志里拿 token（token 在启动 URL 里，不在 HTML 里）
#    带令牌  http://127.0.0.1:8795/?token=XXXX

# 3. 打接口
curl -H "X-TW-Token: XXXX" http://127.0.0.1:8795/api/meta
curl -H "X-TW-Token: XXXX" http://127.0.0.1:8795/api/real/BTCUSDT_1h

# 4. 跑一个真实实验
curl -X POST -H "X-TW-Token: XXXX" -H "Content-Type: application/json" \
  -d '{"scenario":"liquidation","seed":20260917,"ticks":3000,"strategy":"mm_skewed"}' \
  http://127.0.0.1:8795/api/run
# → 拿 job_id，轮询 /api/job/<id> 看 state 变 done

# 5. 视觉验证（最可靠的一步）
msedge --headless=new --window-size=1440,900 --screenshot=shot.png \
  "http://127.0.0.1:8795/?token=XXXX"
```

⚠️ **用 `curl -w "%{size_download}"` 判断响应大小会骗人**——
实测首页它报 `0 bytes`，而响应头明确写着 `Content-Length: 65869`。
要看就用 `-D -` 看响应头，或者干脆存成文件看大小。

## 待办 / 可改进

- [ ] 加图标（`.ico`）。有图标后 spec 的 `EXE(icon=...)` 与 iss 的
      `SetupIconFile` 都能用上，看起来会正式很多。
- [ ] 正式版把 spec 里 `console=True` 改成 `False`（现在是留着看错误的）。
- [ ] 若要让没装 WebView2 的机器也能跑，可以把 WebView2 的固定版运行时
      一起打进去（代价：体积 +100~150 MB）。
