param(
    [int]$DatasetId = 501,
    [ValidateSet("3d_fullres")]
    [string]$Configuration = "3d_fullres"
)

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv/Scripts/python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    $python = "python"
}
$env:nnUNet_raw = Join-Path $projectRoot "workspace/nnUNet_raw"
$env:nnUNet_preprocessed = Join-Path $projectRoot "workspace/nnUNet_preprocessed"
$env:nnUNet_results = Join-Path $projectRoot "workspace/nnUNet_results"
$venvScripts = Join-Path $projectRoot ".venv/Scripts"
$env:Path = "$venvScripts;$env:Path"

$planner = Get-Command nnUNetv2_plan_and_preprocess -ErrorAction SilentlyContinue
if (-not $planner) {
    Write-Error "nnUNetv2_plan_and_preprocess was not found. Install requirements/nnunet-v2.txt."
    exit 2
}
& $planner.Source -d $DatasetId --verify_dataset_integrity -c $Configuration
exit $LASTEXITCODE
