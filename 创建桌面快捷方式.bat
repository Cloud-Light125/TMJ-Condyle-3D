@echo off
setlocal
rem Use a GUI-hosted runner so shortcut-creation errors are shown in Chinese.
set "WSCRIPT_EXE=%SystemRoot%\System32\wscript.exe"
"%WSCRIPT_EXE%" "%~dp0scripts\run_hidden_powershell.vbs" shortcut
set "EXIT_CODE=%ERRORLEVEL%"
endlocal & exit /b %EXIT_CODE%
