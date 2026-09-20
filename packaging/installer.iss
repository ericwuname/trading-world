; 「交易世界」桌面端安装包脚本（Inno Setup 6）
;
; 编译前提：装 Inno Setup 6（winget install JRSoftware.InnoSetup）
; 编译：    ISCC.exe packaging\installer.iss
; 产物：    out\_package\TradingWorld-Setup-<版本>.exe
;
; 设计取舍
; --------
; · 装到 Program Files 而不是用户目录：符合 Windows 惯例；
;   ⚠️ 因此**不能装到需要管理员权限之外的位置**，PrivilegesRequired 用 admin。
; · 本程序**不写任何用户数据**（GUI 只在内存里跑实验，结果由用户自己导出），
;   所以卸载时**不需要**清理数据目录——卸载就是干净卸载。
; · 不注册文件关联、不装服务、不写注册表业务项（只写 Inno 自己的卸载信息）。
; · WebView2 运行时：Windows 10/11 通常自带（用 Edge 的内核）。
;   这里不内置安装器（那会多出 100+MB），只在缺失时给出提示。

#define AppName "交易世界"
#define AppNameEn "TradingWorld"
#define AppVersion "1.0.0"
#define AppPublisher "ericwuname"
#define AppURL "https://github.com/ericwuname/trading-world"

[Setup]
AppId={{8F3A6E1C-4B27-4D9A-9E52-71C0A3B8D4F1}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}
DefaultDirName={autopf}\{#AppNameEn}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
OutputDir=..\out\_package
OutputBaseFilename={#AppNameEn}-Setup-{#AppVersion}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
; 装到 Program Files 需要管理员；用户拒绝时会自动退到用户目录
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
UninstallDisplayName={#AppName}
; 项目暂无 LICENSE 与 .ico —— 没有就别引用，否则 ISCC 直接编译失败
; LicenseFile=..\LICENSE
; SetupIconFile=..\packaging\app.ico

[Languages]
Name: "chinese"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "附加任务:"; Flags: unchecked

[Files]
; 整个 onedir 产物（含 _internal 里的 Python 运行时、numpy/scipy、data/、前端）
Source: "..\out\_package\dist\TradingWorld\*"; DestDir: "{app}"; \
    Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppNameEn}.exe"
Name: "{group}\卸载 {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppNameEn}.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppNameEn}.exe"; Description: "立即启动 {#AppName}"; \
    Flags: nowait postinstall skipifsilent

[UninstallDelete]
; 只删自己装的：不碰用户的任何文档/数据（本程序本来也不写用户数据）
Type: filesandordirs; Name: "{app}\_internal"

[Code]
// WebView2 缺失时给一句人话提示（不阻断安装）
function InitializeSetup(): Boolean;
var
  Missing: Boolean;
begin
  Result := True;
  Missing := not (RegKeyExists(HKLM, 'SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}')
               or RegKeyExists(HKCU, 'SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}'));
  if Missing then
    MsgBox('未检测到 WebView2 运行时。' + #13#10 +
           'Windows 10/11 通常自带；若启动后界面空白，' + #13#10 +
           '请到 Microsoft 官网下载「WebView2 Runtime」安装后重试。',
           mbInformation, MB_OK);
end;
