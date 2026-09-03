@echo off
setlocal
rem The PowerShell window is hidden; all user-facing errors are shown in a dialog.
start "" /b PowerShell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "%~dp0启动实验平台.ps1"
endlocal
exit /b 0
