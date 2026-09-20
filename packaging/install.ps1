<#
.SYNOPSIS
    交易世界 · 桌面端 —— 免管理员权限的安装 / 卸载脚本。

.DESCRIPTION
    为什么要有这个脚本（而不是只给 Inno Setup）：
    Inno Setup 编译出的 setup.exe 需要先装 Inno Setup 这个工具；
    而这个脚本**零依赖**——任何 Windows 10/11 都能直接跑，
    装到 `%LOCALAPPDATA%\Programs\TradingWorld`（**不需要管理员**），
    并建好开始菜单与桌面快捷方式。

    两种安装包的分工：
      · install.ps1        —— 零依赖、当前就能用、装到用户目录
      · installer.iss      —— 用 Inno Setup 编译成正式的 setup.exe
                              （能装到 Program Files、有标准卸载项）
    两者装的东西一样，只是"谁来做"不同。

.USAGE
    在这个目录下右键 → "在终端中打开"，然后：
        powershell -ExecutionPolicy Bypass -File .\install.ps1
    卸载：
        powershell -ExecutionPolicy Bypass -File .\install.ps1 -Uninstall

.NOTES
    本程序不写任何用户数据（GUI 只在内存里跑实验），所以卸载是干净的。
#>

[CmdletBinding()]
param(
    [switch]$Uninstall,
    [string]$Source = $PSScriptRoot,
    [switch]$NoDesktopIcon
)

$ErrorActionPreference = "Stop"
$AppName    = "交易世界"
$AppNameEn  = "TradingWorld"
$ExeName    = "TradingWorld.exe"
$InstallDir = Join-Path $env:LOCALAPPDATA "Programs\$AppNameEn"
$StartMenu  = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs"
$Desktop    = [Environment]::GetFolderPath("Desktop")

function Write-Step($msg) { Write-Host "  $msg" }

# ---------------------------------------------------------------- 卸载
if ($Uninstall) {
    Write-Host "$AppName · 卸载" -ForegroundColor Cyan
    foreach ($lnk in @(
        (Join-Path $StartMenu "$AppName.lnk"),
        (Join-Path $Desktop "$AppName.lnk")
    )) {
        if (Test-Path -LiteralPath $lnk) {
            Remove-Item -LiteralPath $lnk -Force
            Write-Step "已删快捷方式: $lnk"
        }
    }
    if (Test-Path -LiteralPath $InstallDir) {
        Remove-Item -LiteralPath $InstallDir -Recurse -Force
        Write-Step "已删安装目录: $InstallDir"
    }
    Write-Host "  ✅ 卸载完成（程序本来就不写用户数据，所以没有残留）" -ForegroundColor Green
    return
}

# ---------------------------------------------------------------- 安装
Write-Host "$AppName · 安装" -ForegroundColor Cyan

$srcExe = Join-Path $Source $ExeName
if (-not (Test-Path -LiteralPath $srcExe)) {
    throw "找不到 $srcExe —— 请把本脚本放在 TradingWorld.exe 所在的目录里再运行。"
}

if (Test-Path -LiteralPath $InstallDir) {
    Write-Step "目标目录已存在，先清理旧版本…"
    Remove-Item -LiteralPath $InstallDir -Recurse -Force
}

New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
Copy-Item -Path (Join-Path $Source "*") -Destination $InstallDir -Recurse -Force
Write-Step "已复制到: $InstallDir"

$targetExe = Join-Path $InstallDir $ExeName

function New-Shortcut($path) {
    $ws = New-Object -ComObject WScript.Shell
    $sc = $ws.CreateShortcut($path)
    $sc.TargetPath       = $targetExe
    $sc.WorkingDirectory = $InstallDir
    $sc.Description      = "$AppName —— 基于主体建模的人工市场"
    $sc.Save()
}

New-Shortcut (Join-Path $StartMenu "$AppName.lnk")
Write-Step "已建开始菜单快捷方式"

if (-not $NoDesktopIcon) {
    New-Shortcut (Join-Path $Desktop "$AppName.lnk")
    Write-Step "已建桌面快捷方式"
}

# 卸载入口：把卸载命令写成一个 .cmd，方便用户不用记参数
$uninst = Join-Path $InstallDir "卸载.cmd"
@"
@echo off
powershell -ExecutionPolicy Bypass -File "%~dp0uninstall.ps1"
pause
"@ | Set-Content -LiteralPath $uninst -Encoding OEM
Copy-Item -LiteralPath $PSCommandPath -Destination (Join-Path $InstallDir "uninstall.ps1") -Force

Write-Host ""
Write-Host "  ✅ 安装完成" -ForegroundColor Green
Write-Host "     位置: $InstallDir"
Write-Host "     启动: 开始菜单里的「$AppName」，或双击桌面图标"
Write-Host "     卸载: 运行 $uninst"
