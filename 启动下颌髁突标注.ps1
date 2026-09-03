$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$launcher = Join-Path $projectRoot '启动实验平台.ps1'
if (-not (Test-Path -LiteralPath $launcher -PathType Leaf)) {
    exit 1
}
& $launcher
exit $LASTEXITCODE
