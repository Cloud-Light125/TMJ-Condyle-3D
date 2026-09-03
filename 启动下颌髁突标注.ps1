$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$modulePath = Join-Path $projectRoot 'slicer\TMJCondyleAnnotator'
$startupScript = Join-Path $projectRoot 'slicer\startup_tmj.py'

$candidates = @(
    'C:\Users\cloudlight\Apps\Slicer5123b\Slicer.exe',
    (Join-Path ${env:ProgramFiles} '3D Slicer*\Slicer.exe'),
    (Join-Path ${env:ProgramFiles(x86)} '3D Slicer*\Slicer.exe'),
    (Join-Path ${env:LOCALAPPDATA} 'Programs\3D Slicer*\Slicer.exe'),
    (Join-Path ${env:LOCALAPPDATA} 'NA-MIC\3D Slicer*\Slicer.exe')
)

$slicerPath = $null
foreach ($candidate in $candidates) {
    if ([string]::IsNullOrWhiteSpace($candidate)) {
        continue
    }
    if ($candidate -notlike '*[*?]*' -and (Test-Path -LiteralPath $candidate -PathType Leaf)) {
        $slicerPath = (Resolve-Path -LiteralPath $candidate).Path
        break
    }
    $match = Get-ChildItem -Path $candidate -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -ieq 'Slicer.exe' } |
        Sort-Object FullName -Descending |
        Select-Object -First 1
    if ($match) {
        $slicerPath = $match.FullName
        break
    }
}

if (-not $slicerPath) {
    $message = "3D Slicer was not found.`n`nPlease install 3D Slicer and try again."
    try {
        Add-Type -AssemblyName PresentationFramework
        [System.Windows.MessageBox]::Show($message, 'TMJ Condyle Annotator') | Out-Null
    } catch {
        Write-Error $message
    }
    exit 1
}

if (-not (Test-Path -LiteralPath $modulePath -PathType Container)) {
    throw "Project module directory was not found: $modulePath"
}
if (-not (Test-Path -LiteralPath $startupScript -PathType Leaf)) {
    throw "Slicer startup script was not found: $startupScript"
}

$arguments = @(
    '--no-splash',
    '--additional-module-path', $modulePath,
    '--python-script', $startupScript
)
Start-Process -FilePath $slicerPath -ArgumentList $arguments -WorkingDirectory $projectRoot
