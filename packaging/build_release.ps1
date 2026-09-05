param(
    [Parameter(Mandatory = $true)]
    [string]$SlicerSource,
    [string]$PythonBuilder
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$packagingRoot = (Resolve-Path $PSScriptRoot).Path
$buildRoot = Join-Path $packagingRoot 'build'
$stage = Join-Path $buildRoot 'staging'
$release = Join-Path $projectRoot 'artifacts\release'
$pythonLock = Join-Path $packagingRoot 'requirements-lock.txt'

if ([string]::IsNullOrWhiteSpace($PythonBuilder)) {
    $PythonBuilder = Join-Path $projectRoot '.venv\Scripts\python.exe'
}
$PythonBuilder = (Resolve-Path $PythonBuilder -ErrorAction Stop).Path
$SlicerSource = (Resolve-Path $SlicerSource -ErrorAction Stop).Path
$slicerExecutable = Join-Path $SlicerSource 'Slicer.exe'
if (-not (Test-Path -LiteralPath $slicerExecutable -PathType Leaf)) {
    throw "Slicer runtime directory does not contain Slicer.exe: $SlicerSource"
}
if (-not (Test-Path -LiteralPath $pythonLock -PathType Leaf)) {
    throw "Release lock file is missing: $pythonLock"
}

function Remove-GeneratedDirectory([string]$Path) {
    if (Test-Path -LiteralPath $Path) {
        Remove-Item -LiteralPath $Path -Recurse -Force
    }
    New-Item -ItemType Directory -Path $Path -Force | Out-Null
}

function Invoke-Robocopy {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Destination,
        [string[]]$ExcludeDirectories = @(),
        [string[]]$ExcludeFiles = @()
    )
    New-Item -ItemType Directory -Path $Destination -Force | Out-Null
    $arguments = @(
        $Source, $Destination, '/E', '/COPY:DAT', '/DCOPY:DAT',
        '/R:2', '/W:1', '/NFL', '/NDL', '/NP'
    )
    if ($ExcludeDirectories.Count -gt 0) {
        $arguments += '/XD'
        $arguments += $ExcludeDirectories
    }
    if ($ExcludeFiles.Count -gt 0) {
        $arguments += '/XF'
        $arguments += $ExcludeFiles
    }
    & robocopy.exe @arguments | Out-Host
    if ($LASTEXITCODE -gt 7) {
        throw "robocopy failed ($LASTEXITCODE): $Source -> $Destination"
    }
}

function Remove-StagedPackage([string]$NamePattern) {
    if (-not (Test-Path -LiteralPath $script:runtimeSite)) {
        return
    }
    Get-ChildItem -LiteralPath $script:runtimeSite -Force | Where-Object {
        $_.Name -match $NamePattern
    } | ForEach-Object {
        Remove-Item -LiteralPath $_.FullName -Recurse -Force
    }
}

function Restore-NumpyRuntimeSupport {
    # NumPy 2.2.6 exposes numpy.testing lazily. SciPy's array API adapter
    # reaches that path during imports such as scipy.ndimage, which requires
    # this small runtime helper even though the rest of NumPy's test suite is
    # correctly excluded from the release.
    $source = Join-Path $projectRoot '.venv\Lib\site-packages\numpy\_core\tests\_natype.py'
    $destinationDirectory = Join-Path $script:runtimeSite 'numpy\_core\tests'
    $destination = Join-Path $destinationDirectory '_natype.py'
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "The builder environment is missing NumPy runtime support: $source"
    }
    New-Item -ItemType Directory -Path $destinationDirectory -Force | Out-Null
    Copy-Item -LiteralPath $source -Destination $destination -Force
}

function Invoke-RuntimePython {
    param(
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$DataRoot
    )
    $tracked = @(
        'PYTHONHOME', 'PYTHONPATH', 'PYTHONUSERBASE', 'PATH',
        'TMJ_APP_ROOT', 'TMJ_USER_DATA_DIR', 'TMJ_RUNTIME_MODE',
        'nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results',
        'PYTHONNOUSERSITE', 'PYTHONUNBUFFERED'
    )
    $old = @{}
    foreach ($name in $tracked) {
        $old[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
    }
    try {
        foreach ($name in @('PYTHONHOME', 'PYTHONPATH', 'PYTHONUSERBASE')) {
            Remove-Item -LiteralPath "Env:$name" -ErrorAction SilentlyContinue
        }
        $runtimePythonRoot = Join-Path $stage 'runtime\python'
        $runtimeSlicerRoot = Join-Path $stage 'runtime\slicer'
        $oldPath = $old['PATH']
        $newPath = @(
            $runtimePythonRoot,
            (Join-Path $runtimePythonRoot 'DLLs'),
            (Join-Path $runtimePythonRoot 'Scripts'),
            $runtimeSlicerRoot,
            $oldPath
        ) -join [IO.Path]::PathSeparator
        [Environment]::SetEnvironmentVariable('PATH', $newPath, 'Process')
        [Environment]::SetEnvironmentVariable('TMJ_APP_ROOT', $stage, 'Process')
        [Environment]::SetEnvironmentVariable('TMJ_USER_DATA_DIR', $DataRoot, 'Process')
        [Environment]::SetEnvironmentVariable('TMJ_RUNTIME_MODE', 'packaged', 'Process')
        [Environment]::SetEnvironmentVariable('PYTHONNOUSERSITE', '1', 'Process')
        [Environment]::SetEnvironmentVariable('PYTHONUNBUFFERED', '1', 'Process')
        [Environment]::SetEnvironmentVariable('PYTHONPATH', $stage, 'Process')
        [Environment]::SetEnvironmentVariable('nnUNet_raw', (Join-Path $DataRoot 'nnUNet_raw'), 'Process')
        [Environment]::SetEnvironmentVariable('nnUNet_preprocessed', (Join-Path $DataRoot 'nnUNet_preprocessed'), 'Process')
        [Environment]::SetEnvironmentVariable('nnUNet_results', (Join-Path $DataRoot 'nnUNet_results'), 'Process')
        & (Join-Path $stage 'runtime\python\python.exe') @Arguments
        $exitCode = $LASTEXITCODE
    } finally {
        foreach ($name in $tracked) {
            if ($null -eq $old[$name]) {
                Remove-Item -LiteralPath "Env:$name" -ErrorAction SilentlyContinue
            } else {
                [Environment]::SetEnvironmentVariable($name, $old[$name], 'Process')
            }
        }
    }
    if ($exitCode -ne 0) {
        throw "Bundled Python command failed with exit code ${exitCode}: $($Arguments -join ' ')"
    }
}

Remove-GeneratedDirectory $buildRoot
New-Item -ItemType Directory -Path $release -Force | Out-Null

# Copy only application source.  The repository workspace, references,
# tests, VCS metadata and development launchers never enter staging.
Invoke-Robocopy (Join-Path $projectRoot 'tmj_condyle') (Join-Path $stage 'tmj_condyle') @('__pycache__', '.pytest_cache')
Invoke-Robocopy (Join-Path $projectRoot 'slicer') (Join-Path $stage 'slicer') @('__pycache__', '.pytest_cache')
New-Item -ItemType Directory -Path (Join-Path $stage 'scripts') -Force | Out-Null
Get-ChildItem -LiteralPath (Join-Path $projectRoot 'scripts') -Filter '*.py' -File | ForEach-Object {
    if ($_.Name -eq 'scan_git_safety.py') {
        return
    }
    Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $stage 'scripts' $_.Name)
}
Copy-Item -LiteralPath (Join-Path $projectRoot 'LICENSE') -Destination $stage
Copy-Item -LiteralPath (Join-Path $packagingRoot 'README-使用说明.txt') -Destination $stage

# Slicer is copied as a complete runtime, but its portable developer settings
# are not runtime binaries and may contain machine-specific module paths.
Invoke-Robocopy $SlicerSource (Join-Path $stage 'runtime\slicer') @('__pycache__', '.pytest_cache')
$slicerSettings = Join-Path $stage 'runtime\slicer\slicer.org'
if (Test-Path -LiteralPath $slicerSettings -PathType Container) {
    Get-ChildItem -LiteralPath $slicerSettings -Filter '*.ini' -File -ErrorAction SilentlyContinue |
        ForEach-Object { Remove-Item -LiteralPath $_.FullName -Force }
}

# Build a real relocatable CPython tree from the base installation, then copy
# project packages into site-packages.  This is not a venv copy and has no
# pyvenv.cfg or dependency on the developer checkout.
$basePython = (& $PythonBuilder -c 'import sys; print(sys.base_prefix)').Trim()
if (-not (Test-Path -LiteralPath (Join-Path $basePython 'python.exe') -PathType Leaf)) {
    throw "The builder Python base runtime is incomplete: $basePython"
}
$runtimePython = Join-Path $stage 'runtime\python'
Invoke-Robocopy $basePython $runtimePython @('__pycache__') @('*.pyc')
$script:runtimeSite = Join-Path $runtimePython 'Lib\site-packages'
New-Item -ItemType Directory -Path $script:runtimeSite -Force | Out-Null
Invoke-Robocopy (Join-Path $projectRoot '.venv\Lib\site-packages') $script:runtimeSite @('__pycache__') @('*.pyc')

# Remove development-only package managers/test harnesses.  Runtime package
# metadata is retained for the notice generator and version checks.
Remove-StagedPackage '^(pip|pip[-.]|setuptools|setuptools[-.]|pytest|pytest[-.]|_pytest|iniconfig|iniconfig[-.]|pluggy|pluggy[-.]|pygments|pygments[-.]|linecache2|linecache2[-.]|traceback2|traceback2[-.]|unittest2|unittest2[-.])'
$runtimeScripts = Join-Path $runtimePython 'Scripts'
if (Test-Path -LiteralPath $runtimeScripts -PathType Container) {
    Get-ChildItem -LiteralPath $runtimeScripts -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^(pip|pip3|pytest)(\.exe|\.cmd|\.bat)?$' } |
        ForEach-Object { Remove-Item -LiteralPath $_.FullName -Force }
}

# Drop package test fixtures and interpreter caches.  These are not needed by
# the application and may contain sample scans, masks, or model-like files.
Get-ChildItem -LiteralPath $stage -Recurse -Directory -Force -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -in @('__pycache__', '.pytest_cache', 'tests', 'test_files', '.git', '.venv', 'workspace', 'test_only_tmj_synthetic') } |
    Sort-Object { $_.FullName.Length } -Descending |
    ForEach-Object { Remove-Item -LiteralPath $_.FullName -Recurse -Force }
Get-ChildItem -LiteralPath $stage -Recurse -File -Force -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match '(?i)\.(nii(\.gz)?|dcm|nrrd|mha|mhd|h5|hdf5|pt|pth|ckpt|npz)$' } |
    ForEach-Object { Remove-Item -LiteralPath $_.FullName -Force }
# Slicer and the module source may contain developer-session logs with local
# paths.  They are never part of a release and must not reach either archive.
Get-ChildItem -LiteralPath $stage -Recurse -File -Filter '*.log' -Force -ErrorAction SilentlyContinue |
    ForEach-Object { Remove-Item -LiteralPath $_.FullName -Force }
# Restore only the NumPy helper required by SciPy at runtime; the broad test
# fixture cleanup above intentionally removes every other package test tree.
Restore-NumpyRuntimeSupport

# Keep the release CPU-only. CUDA/NVIDIA binaries are intentionally not
# copied; GPU is an explicit optional mode in the GUI and requires a separate
# compatible GPU-enabled runtime/driver supplied by the user.
Remove-StagedPackage '^(torch|torchvision|torchgen|functorch)([-.]|$)'
$runtimePythonExe = Join-Path $runtimePython 'python.exe'
$torchArgs = @(
    '-m', 'pip', 'install', '--upgrade', '--no-cache-dir', '--no-deps',
    '--target', $script:runtimeSite,
    '--index-url', 'https://pypi.org/simple',
    'torch==2.14.0',
    'torchvision==0.29.0'
)
& $PythonBuilder @torchArgs
if ($LASTEXITCODE -ne 0) {
    throw "Official PyTorch CPU wheel installation failed with exit code $LASTEXITCODE"
}

# The packaged launcher is a self-contained .NET executable; the end user
# does not need a .NET runtime.
$dotnetCommand = Get-Command dotnet.exe -ErrorAction SilentlyContinue
if (-not $dotnetCommand) {
    throw 'dotnet SDK was not found; it is required only while building the launcher.'
}
$launcherOutput = Join-Path $buildRoot 'launcher-publish'
Remove-GeneratedDirectory $launcherOutput
& $dotnetCommand.Source publish (Join-Path $packagingRoot 'launcher\TMJCondyle3D.Launcher.csproj') `
    --configuration Release --runtime win-x64 --self-contained true `
    --output $launcherOutput --nologo
if ($LASTEXITCODE -ne 0) {
    throw "Launcher publish failed with exit code $LASTEXITCODE"
}
$launcherExe = Join-Path $launcherOutput 'TMJ-Condyle-3D.exe'
if (-not (Test-Path -LiteralPath $launcherExe -PathType Leaf)) {
    throw "Launcher publish did not produce $launcherExe"
}
Copy-Item -LiteralPath $launcherExe -Destination (Join-Path $stage 'TMJ-Condyle-3D.exe')

$commit = (& git -C $projectRoot rev-parse --short=12 HEAD).Trim()
$buildInfo = [ordered]@{
    product = 'TMJ-Condyle-3D'
    display_name = '下颌髁突三维分割实验平台'
    version = '0.1.0'
    git_commit = $commit
    slicer = '5.12.3'
    python = '3.10.21'
    nnunetv2 = '2.8.1'
    torch = '2.14.0'
    torchvision = '0.29.0'
    pytorch_build = 'CPU-only; CUDA is not bundled'
    build_architecture = 'Windows x64'
    offline = $true
}
($buildInfo | ConvertTo-Json -Depth 4) + [Environment]::NewLine |
    Set-Content -LiteralPath (Join-Path $stage 'build-info.json') -Encoding UTF8

# Generate notices and verify imports with the staged interpreter itself.
$verificationData = Join-Path $buildRoot 'verification-user-data'
Remove-GeneratedDirectory $verificationData
Invoke-RuntimePython @((Join-Path $packagingRoot 'generate_notices.py'), '--stage', $stage) $verificationData
Invoke-RuntimePython @((Join-Path $packagingRoot 'verify_runtime.py'), '--app-root', $stage, '--output', (Join-Path $buildRoot 'runtime-verification.json')) $verificationData

# Importing packages can recreate bytecode caches after the first cleanup.
# Remove those generated caches before the immutable staging directory is
# scanned and archived.
Get-ChildItem -LiteralPath $stage -Recurse -Directory -Force -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -in @('__pycache__', '.pytest_cache', 'tests', 'test_files') } |
    Sort-Object { $_.FullName.Length } -Descending |
    ForEach-Object { Remove-Item -LiteralPath $_.FullName -Recurse -Force }
Get-ChildItem -LiteralPath $stage -Recurse -File -Force -ErrorAction SilentlyContinue |
    Where-Object {
        $_.Extension -ieq '.pyc' -or
        $_.Name -match '(?i)\.(nii(\.gz)?|dcm|nrrd|mha|mhd|h5|hdf5|pt|pth|ckpt|npz)$'
    } |
    ForEach-Object { Remove-Item -LiteralPath $_.FullName -Force }
Get-ChildItem -LiteralPath $stage -Recurse -File -Filter '*.log' -Force -ErrorAction SilentlyContinue |
    ForEach-Object { Remove-Item -LiteralPath $_.FullName -Force }
Restore-NumpyRuntimeSupport

$scanReport = Join-Path $buildRoot 'release-scan.txt'
& $PythonBuilder (Join-Path $packagingRoot 'release_scan.py') '--root' $stage '--report' $scanReport
if ($LASTEXITCODE -ne 0) {
    throw "Release safety scan failed; inspect $scanReport"
}

# Create the primary ZIP.  Prefer 7-Zip when installed; Windows tar is the
# next option, and Compress-Archive is a final fallback for smaller builds.
$portableZip = Join-Path $release 'TMJ-Condyle-3D-Portable-x64.zip'
if (Test-Path -LiteralPath $portableZip) {
    Remove-Item -LiteralPath $portableZip -Force
}
$sevenZip = @(
    (Get-Command 7z.exe -ErrorAction SilentlyContinue).Source,
    (Join-Path ${env:ProgramFiles} '7-Zip\7z.exe'),
    (Join-Path ${env:ProgramFiles(x86)} '7-Zip\7z.exe')
) | Where-Object { $_ -and (Test-Path -LiteralPath $_ -PathType Leaf) } | Select-Object -First 1
$tarCommand = Get-Command tar.exe -ErrorAction SilentlyContinue
if ($sevenZip) {
    Push-Location $stage
    try {
        & $sevenZip a -tzip -mx=9 $portableZip '*' | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "7-Zip returned $LASTEXITCODE" }
    } finally {
        Pop-Location
    }
} elseif ($tarCommand) {
    Push-Location $stage
    try {
        & $tarCommand.Source -a -c -f $portableZip '*' | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "tar returned $LASTEXITCODE" }
    } finally {
        Pop-Location
    }
} else {
    Compress-Archive -Path (Join-Path $stage '*') -DestinationPath $portableZip -CompressionLevel Optimal
}

$innoCandidates = @(
    (Get-Command ISCC.exe -ErrorAction SilentlyContinue).Source,
    (Join-Path ${env:ProgramFiles(x86)} 'Inno Setup 6\ISCC.exe'),
    (Join-Path $env:LOCALAPPDATA 'Programs\Inno Setup 6\ISCC.exe')
) | Where-Object { $_ -and (Test-Path -LiteralPath $_ -PathType Leaf) } | Select-Object -First 1
if (-not $innoCandidates) {
    throw 'Inno Setup ISCC.exe was not found; it is required only while building the installer.'
}
$innoLog = Join-Path $buildRoot 'inno-compile.log'
& $innoCandidates (Join-Path $packagingRoot 'installer\TMJ-Condyle-3D.iss') *> $innoLog
if ($LASTEXITCODE -ne 0) {
    Get-Content -LiteralPath $innoLog -Tail 80 | Out-Host
    throw "Inno Setup returned $LASTEXITCODE"
}
$installer = Join-Path $release 'TMJ-Condyle-3D-Setup-x64.exe'
if (-not (Test-Path -LiteralPath $installer -PathType Leaf)) {
    throw "Installer was not produced: $installer"
}

Copy-Item -LiteralPath (Join-Path $stage 'THIRD_PARTY_NOTICES.txt') -Destination $release -Force
Copy-Item -LiteralPath (Join-Path $stage 'README-使用说明.txt') -Destination $release -Force
$hashLines = foreach ($file in @($installer, $portableZip)) {
    $hash = (Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash.ToLowerInvariant()
    "$hash *$([IO.Path]::GetFileName($file))"
}
[IO.File]::WriteAllLines((Join-Path $release 'SHA256SUMS.txt'), $hashLines, [Text.UTF8Encoding]::new($false))

$installedBytes = (Get-ChildItem -LiteralPath $stage -Recurse -File | Measure-Object -Property Length -Sum).Sum
$releaseInfo = [ordered]@{
    installer = $installer
    portable = $portableZip
    installer_bytes = (Get-Item -LiteralPath $installer).Length
    portable_bytes = (Get-Item -LiteralPath $portableZip).Length
    installed_bytes = [int64]$installedBytes
    commit = $commit
    release_scan = $scanReport
}
($releaseInfo | ConvertTo-Json -Depth 4) + [Environment]::NewLine |
    Set-Content -LiteralPath (Join-Path $buildRoot 'release-info.json') -Encoding UTF8

Write-Host "Release build complete"
Write-Host ("Installed size: {0:N0} bytes" -f $releaseInfo.installed_bytes)
Write-Host ("Installer size:  {0:N0} bytes" -f $releaseInfo.installer_bytes)
Write-Host ("Portable size:  {0:N0} bytes" -f $releaseInfo.portable_bytes)
Write-Host "Artifacts: $release"
