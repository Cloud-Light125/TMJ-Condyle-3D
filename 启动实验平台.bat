@echo off
setlocal
rem Use a GUI-hosted runner so no command window is left behind for ordinary users.
set "WSCRIPT_EXE=%SystemRoot%\System32\wscript.exe"
"%WSCRIPT_EXE%" "%~dp0scripts\run_hidden_powershell.vbs" platform
set "EXIT_CODE=%ERRORLEVEL%"
endlocal & exit /b %EXIT_CODE%
