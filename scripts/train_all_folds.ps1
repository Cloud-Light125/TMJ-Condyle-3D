param(
    [switch]$Resume,
    [ValidateSet("cuda", "cpu", "mps")]
    [string]$Device = "cpu"
)

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot "runtime\python\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    $python = Join-Path $projectRoot ".venv\Scripts\python.exe"
}
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    Write-Error "项目内 Python runtime 不存在；请重新安装软件或从开发环境运行。"
    exit 2
}
$documents = [Environment]::GetFolderPath('MyDocuments')
$workspace = Join-Path $documents 'TMJ-Condyle-3D\workspace'
$env:TMJ_APP_ROOT = $projectRoot
$env:TMJ_USER_DATA_DIR = $workspace
$env:nnUNet_raw = Join-Path $workspace 'nnUNet_raw'
$env:nnUNet_preprocessed = Join-Path $workspace 'nnUNet_preprocessed'
$env:nnUNet_results = Join-Path $workspace 'nnUNet_results'
Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
Remove-Item Env:PYTHONUSERBASE -ErrorAction SilentlyContinue
$env:PYTHONNOUSERSITE = '1'
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
