; Inno Setup script for the Windows app. CI runs:
;   iscc /DBuild=<run number> packaging\installer.iss
; Input: dist\CFB Predictor\ (PyInstaller). Output: dist\CFB-Predictor-Setup-build-<n>.exe
; Per-user install (no admin rights). Data and keys live outside the install folder
; (%LOCALAPPDATA%\cfb, %APPDATA%\cfb, Windows Credential Manager), so an update or an
; uninstall keeps them.

#ifndef Build
  #define Build "0"
#endif

[Setup]
AppId={{6C1F8B7E-4D2A-4F6B-9E3C-CFB0A11CE001}
AppName=CFB Predictor
AppVersion=build-{#Build}
AppPublisher=CFB-Analysis
DefaultDirName={localappdata}\Programs\CFB Predictor
DefaultGroupName=CFB Predictor
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=..\dist
OutputBaseFilename=CFB-Predictor-Setup-build-{#Build}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; Flags: unchecked

[Files]
Source: "..\dist\CFB Predictor\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[InstallDelete]
; Replace the previous build's files so removed modules do not linger.
Type: filesandordirs; Name: "{app}\_internal"

[Icons]
Name: "{group}\CFB Predictor"; Filename: "{app}\CFB Predictor.exe"
Name: "{userdesktop}\CFB Predictor"; Filename: "{app}\CFB Predictor.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\CFB Predictor.exe"; Description: "Start CFB Predictor"; Flags: nowait postinstall skipifsilent
