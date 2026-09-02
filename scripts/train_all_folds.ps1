param(
    [switch]$Resume,
    [ValidateSet("cuda", "cpu", "mps")]
    [string]$Device = "cuda"
)

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    $python = "python"
}
$arguments = @(
    (Join-Path $projectRoot "scripts\train_all_folds.py"),
    "--device",
    $Device
)
if ($Resume) {
    $arguments += "--resume"
}
& $python @arguments
exit $LASTEXITCODE
