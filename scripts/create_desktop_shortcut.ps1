$ErrorActionPreference = 'Stop'

Add-Type -AssemblyName System.Windows.Forms
[System.Windows.Forms.Application]::EnableVisualStyles()

$projectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$launcherName = -join @([char]0x542F,[char]0x52A8,[char]0x5B9E,[char]0x9A8C,[char]0x5E73,[char]0x53F0)
$launcherPath = Join-Path $projectRoot ($launcherName + '.bat')
$launcherScriptPath = Join-Path $projectRoot ($launcherName + '.ps1')
$runnerPath = Join-Path $projectRoot 'scripts\run_hidden_powershell.vbs'

function U([int[]] $codes) {
    return -join ($codes | ForEach-Object { [char]$_ })
}

$title = U @(0x4E0B,0x988C,0x9AC1,0x7A81,0x4E09,0x7EF4,0x5206,0x5272,0x5B9E,0x9A8C,0x5E73,0x53F0)
$successPrefix = U @(0x684C,0x9762,0x5FEB,0x6377,0x65B9,0x5F0F,0x5DF2,0x521B,0x5EFA,0xFF1A)
$errorPrefix = U @(0x521B,0x5EFA,0x684C,0x9762,0x5FEB,0x6377,0x65B9,0x5F0F,0x5931,0x8D25,0x3002)

try {
    if (-not (Test-Path -LiteralPath $launcherPath -PathType Leaf)) {
        throw "The project launcher was not found: $launcherPath"
    }
    if (-not (Test-Path -LiteralPath $runnerPath -PathType Leaf)) {
        throw "The hidden runner was not found: $runnerPath"
    }
    if (-not (Test-Path -LiteralPath $launcherScriptPath -PathType Leaf)) {
        throw "The PowerShell launcher was not found: $launcherScriptPath"
    }

    $desktopPath = [Environment]::GetFolderPath('Desktop')
    if ([string]::IsNullOrWhiteSpace($desktopPath) -or -not (Test-Path -LiteralPath $desktopPath -PathType Container)) {
        throw 'The current user Desktop folder was not found.'
    }

    $shortcutName = -join @(
        [char]0x4E0B, [char]0x988C, [char]0x9AC1, [char]0x7A81,
        [char]0x4E09, [char]0x7EF4, [char]0x5206, [char]0x5272,
        [char]0x5B9E, [char]0x9A8C, [char]0x5E73, [char]0x53F0
    )
    $shortcutPath = Join-Path $desktopPath ($shortcutName + '.lnk')
    $wscriptPath = Join-Path $env:SystemRoot 'System32\wscript.exe'
    if (-not (Test-Path -LiteralPath $wscriptPath -PathType Leaf)) {
        $wscriptPath = 'wscript.exe'
    }
    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($shortcutPath)
    $shortcut.TargetPath = $wscriptPath
    $shortcut.Arguments = ('"{0}" platform' -f $runnerPath)
    $shortcut.WorkingDirectory = $projectRoot
    $shortcut.Description = 'TMJ-Condyle-3D segmentation experiment platform'
    $shortcut.IconLocation = "$wscriptPath,0"
    $configPath = Join-Path $projectRoot 'workspace\.tmj_platform_config.json'
    if (Test-Path -LiteralPath $configPath -PathType Leaf) {
        try {
            $configured = (Get-Content -LiteralPath $configPath -Raw -Encoding UTF8 | ConvertFrom-Json).slicer_path
            if ($configured -and (Test-Path -LiteralPath $configured -PathType Leaf)) {
                $shortcut.IconLocation = "$configured,0"
            }
        } catch {
            # The launcher remains usable even if the optional icon preference is unreadable.
        }
    }
    $shortcut.Save()

    Write-Output "Desktop shortcut created: $shortcutPath"
    [System.Windows.Forms.MessageBox]::Show(
        ($successPrefix + [Environment]::NewLine + $shortcutPath),
        $title,
        [System.Windows.Forms.MessageBoxButtons]::OK,
        [System.Windows.Forms.MessageBoxIcon]::Information
    ) | Out-Null
    exit 0
} catch {
    $detail = $errorPrefix + [Environment]::NewLine + [Environment]::NewLine + $_.Exception.Message
    [System.Windows.Forms.MessageBox]::Show(
        $detail,
        $title,
        [System.Windows.Forms.MessageBoxButtons]::OK,
        [System.Windows.Forms.MessageBoxIcon]::Error
    ) | Out-Null
    exit 1
}
