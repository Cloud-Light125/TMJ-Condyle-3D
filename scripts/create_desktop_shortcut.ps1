$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$launcherName = -join @(
    [char]0x542F, [char]0x52A8, [char]0x4E0B, [char]0x988C,
    [char]0x9AC1, [char]0x7A81, [char]0x6807, [char]0x6CE8
)
$launcherPath = Join-Path $projectRoot ($launcherName + '.bat')
if (-not (Test-Path -LiteralPath $launcherPath -PathType Leaf)) {
    throw "The project launcher was not found: $launcherPath"
}

$desktopPath = [Environment]::GetFolderPath('Desktop')
if ([string]::IsNullOrWhiteSpace($desktopPath) -or -not (Test-Path -LiteralPath $desktopPath -PathType Container)) {
    throw 'The current user Desktop folder was not found.'
}

$shortcutName = -join @(
    [char]0x4E0B, [char]0x988C, [char]0x9AC1, [char]0x7A81,
    [char]0x4E09, [char]0x7EF4, [char]0x6807, [char]0x6CE8
)
$shortcutPath = Join-Path $desktopPath ($shortcutName + '.lnk')
$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $launcherPath
$shortcut.WorkingDirectory = $projectRoot
$shortcut.Description = 'TMJ Condyle annotation workbench'
$shortcut.IconLocation = "$launcherPath,0"
$shortcut.Save()

Write-Output "Desktop shortcut created: $shortcutPath"
