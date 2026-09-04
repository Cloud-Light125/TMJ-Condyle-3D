@echo off
setlocal
rem Kept as a compatibility entry point for existing users.
set "WSCRIPT_EXE=%SystemRoot%\System32\wscript.exe"
"%WSCRIPT_EXE%" "%~dp0scripts\run_hidden_powershell.vbs" platform
set "EXIT_CODE=%ERRORLEVEL%"
endlocal & exit /b %EXIT_CODE%
