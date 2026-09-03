@echo off
setlocal
rem Keep this wrapper ASCII-only so cmd.exe code pages cannot corrupt the launcher path.
PowerShell.exe -NoProfile -ExecutionPolicy Bypass -Command "$root = (Resolve-Path -LiteralPath '%~dp0').Path; $name = -join @([char]0x542F,[char]0x52A8,[char]0x4E0B,[char]0x988C,[char]0x9AC1,[char]0x7A81,[char]0x6807,[char]0x6CE8); $script = Join-Path $root ($name + '.ps1'); if (-not (Test-Path -LiteralPath $script -PathType Leaf)) { exit 1 }; & $script"
if errorlevel 1 pause
endlocal
