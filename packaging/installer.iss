; Inno Setup script for the Windows app. CI runs:
;   iscc /DBuild=<run number> /DEdition=<main|experimental> packaging\installer.iss
; Input: dist\CFB Predictor\ (PyInstaller). Output: dist\CFB-Predictor-Setup-build-<n>.exe
; (experimental: dist\CFB-Predictor-Experimental-Setup-build-<n>.exe).
; Per-user install (no admin rights). Data and keys live outside the install folder
; (%LOCALAPPDATA%\cfb, Windows Credential Manager; cfb-exp for the experimental edition), so an
; update or an uninstall keeps them. The two editions have their own AppId, folder and Start
; menu entry, so they install side by side.

#ifndef Build
  #define Build "0"
#endif
#ifndef Edition
  #define Edition "main"
#endif

#if Edition == "experimental"
  #define AppTitle "CFB Predictor Experimental"
  #define AppGuid "{{6C1F8B7E-4D2A-4F6B-9E3C-CFB0A11CE0E1}"
  #define SetupName "CFB-Predictor-Experimental-Setup-build-" + Build
#else
  #define AppTitle "CFB Predictor"
  #define AppGuid "{{6C1F8B7E-4D2A-4F6B-9E3C-CFB0A11CE001}"
  #define SetupName "CFB-Predictor-Setup-build-" + Build
#endif

[Setup]
AppId={#AppGuid}
AppName={#AppTitle}
AppVersion=build-{#Build}
AppPublisher=CFB-Analysis
DefaultDirName={localappdata}\Programs\{#AppTitle}
DefaultGroupName={#AppTitle}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=..\dist
OutputBaseFilename={#SetupName}
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
Name: "{group}\{#AppTitle}"; Filename: "{app}\CFB Predictor.exe"
Name: "{userdesktop}\{#AppTitle}"; Filename: "{app}\CFB Predictor.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\CFB Predictor.exe"; Description: "Start {#AppTitle}"; Flags: nowait postinstall skipifsilent

[UninstallRun]
; The scheduled tasks would point at a removed program: delete them. Data and keys stay.
Filename: "{app}\CFB Engine.exe"; Parameters: "engine schedule remove all"; Flags: runhidden waituntilterminated; RunOnceId: "RemoveTasks"
