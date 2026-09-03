@echo off
setlocal
rem The helper creates a shortcut with the user-facing platform name.
start "" /b PowerShell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "%~dp0scripts\create_desktop_shortcut.ps1"
endlocal
exit /b 0
