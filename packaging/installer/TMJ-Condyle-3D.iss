#define AppName "TMJ-Condyle-3D"
#define AppDisplayName "下颌髁突三维分割实验平台"
#define AppVersion "0.1.0"
#define AppPublisher "TMJ-Condyle-3D"
#define StagingDir "..\build\staging"

[Setup]
AppId={{F9E2A7D7-4C0A-4AF4-9E22-7D4A4B0B6F10}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppDisplayName} {#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\TMJ-Condyle-3D
DefaultGroupName={#AppDisplayName}
DisableProgramGroupPage=yes
PrivilegesRequired=admin
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\..\artifacts\release
OutputBaseFilename=TMJ-Condyle-3D-Setup-x64
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
Uninstallable=yes
UninstallDisplayName={#AppDisplayName}
CloseApplications=yes
RestartApplications=no
AllowNoIcons=yes
VersionInfoVersion={#AppVersion}.0
VersionInfoDescription={#AppDisplayName}
VersionInfoProductName={#AppName}
VersionInfoProductVersion={#AppVersion}

[Languages]
Name: "chinesesimp"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "快捷方式："; Flags: unchecked
Name: "startmenuicon"; Description: "创建开始菜单快捷方式"; GroupDescription: "快捷方式："; Flags: checkedonce

[Files]
Source: "{#StagingDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[InstallDelete]
; Replace a same-named developer shortcut so installation cannot leave the
; user on the old wscript/source-tree launch chain.
Type: files; Name: "{autodesktop}\{#AppDisplayName}.lnk"

[Icons]
Name: "{group}\{#AppDisplayName}"; Filename: "{app}\TMJ-Condyle-3D.exe"; WorkingDir: "{app}"; Tasks: startmenuicon
Name: "{autodesktop}\{#AppDisplayName}"; Filename: "{app}\TMJ-Condyle-3D.exe"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\TMJ-Condyle-3D.exe"; Description: "启动{#AppDisplayName}"; Flags: nowait postinstall skipifsilent

; 没有 [UninstallDelete]：用户医学数据在 Documents 外置目录，卸载绝不删除。
