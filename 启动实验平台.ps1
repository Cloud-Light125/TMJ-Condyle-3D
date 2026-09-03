$ErrorActionPreference = 'Stop'

# Keep this implementation ASCII-only. Windows PowerShell 5 can otherwise
# interpret a UTF-8 script using the active legacy code page.
Add-Type -AssemblyName System.Windows.Forms
[System.Windows.Forms.Application]::EnableVisualStyles()

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$modulePath = Join-Path $projectRoot 'slicer\TMJCondyleAnnotator'
$startupScript = Join-Path $projectRoot 'slicer\startup_tmj.py'
$configPath = Join-Path $projectRoot 'workspace\.tmj_platform_config.json'

function U([int[]] $codes) {
    return -join ($codes | ForEach-Object { [char]$_ })
}

$title = U @(0x4E0B,0x988C,0x9AC1,0x7A81,0x4E09,0x7EF4,0x5206,0x5272,0x5B9E,0x9A8C,0x5E73,0x53F0)
$selectText = U @(0x9009,0x62E9,0x0020,0x53C2,0x8003,0x0020,0x53EF,0x6267,0x884C,0x6587,0x4EF6)
$chooseSlicerText = U @(0x9009,0x62E9,0x0020,0x0053,0x006C,0x0069,0x0063,0x0065,0x0072,0x002E,0x0065,0x0078,0x0065)
$installHelpText = U @(0x67E5,0x770B,0x5B89,0x88C5,0x8BF4,0x660E)
$closeText = U @(0x5173,0x95ED)
$okText = U @(0x4F7F,0x7528,0x8FD9,0x4E2A,0x7248,0x672C)
$cancelText = U @(0x53D6,0x6D88)
$retryText = U @(0x91CD,0x65B0,0x68C0,0x6D4B)

function Show-Message([string] $message, [string] $caption = $title, [System.Windows.Forms.MessageBoxIcon] $icon = [System.Windows.Forms.MessageBoxIcon]::Information) {
    [System.Windows.Forms.MessageBox]::Show($message, $caption, [System.Windows.Forms.MessageBoxButtons]::OK, $icon) | Out-Null
}

function Read-ConfiguredSlicer {
    if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) {
        return $null
    }
    try {
        $config = Get-Content -LiteralPath $configPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $value = [string]$config.slicer_path
        if ([string]::IsNullOrWhiteSpace($value)) {
            return $null
        }
        if (-not [System.IO.Path]::IsPathRooted($value)) {
            $value = Join-Path $projectRoot $value
        }
        if ((Test-Path -LiteralPath $value -PathType Leaf) -and ([System.IO.Path]::GetFileName($value) -ieq 'Slicer.exe')) {
            return (Resolve-Path -LiteralPath $value).Path
        }
    } catch {
        return $null
    }
    return $null
}

function Save-ConfiguredSlicer([string] $path) {
    try {
        $parent = Split-Path -Parent $configPath
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
        $payload = @{ slicer_path = (Resolve-Path -LiteralPath $path).Path; updated_by = 'TMJ-Condyle-3D' } | ConvertTo-Json
        $utf8NoBom = New-Object -TypeName System.Text.UTF8Encoding -ArgumentList $false
        [System.IO.File]::WriteAllText($configPath, $payload + [Environment]::NewLine, $utf8NoBom)
    } catch {
        # A saved preference is helpful but must never prevent a valid launch.
    }
}

$script:discovered = @()
$script:seenPaths = @{}
function Add-SlicerCandidate([string] $path, [string] $source) {
    if ([string]::IsNullOrWhiteSpace($path)) {
        return
    }
    try {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            return
        }
        if ([System.IO.Path]::GetFileName($path) -ine 'Slicer.exe') {
            return
        }
        $resolved = (Resolve-Path -LiteralPath $path).Path
        $key = $resolved.ToLowerInvariant()
        if (-not $script:seenPaths.ContainsKey($key)) {
            $script:seenPaths[$key] = $true
            $script:discovered += [PSCustomObject]@{ Path = $resolved; Source = $source }
        }
    } catch {
        # Ignore inaccessible folders and continue looking in other locations.
    }
}

function Find-SlicerCandidates {
    $script:discovered = @()
    $script:seenPaths = @{}
    Add-SlicerCandidate 'C:\Users\cloudlight\Apps\Slicer5123b\Slicer.exe' '项目默认位置'
    $configured = Read-ConfiguredSlicer
    if ($configured) {
        Add-SlicerCandidate $configured '项目设置'
    }

    $programFiles = if ($env:ProgramFiles) { $env:ProgramFiles } else { 'C:\Program Files' }
    $programFilesX86 = if (${env:ProgramFiles(x86)}) { ${env:ProgramFiles(x86)} } else { 'C:\Program Files (x86)' }
    $localAppData = if ($env:LOCALAPPDATA) { $env:LOCALAPPDATA } else { Join-Path $env:USERPROFILE 'AppData\Local' }
    $userProfile = if ($env:USERPROFILE) { $env:USERPROFILE } else { [Environment]::GetFolderPath('UserProfile') }
    $patterns = @(
        @($programFiles, '3D Slicer*\Slicer.exe', 'Program Files'),
        @($programFiles, 'Slicer*\Slicer.exe', 'Program Files'),
        @($programFilesX86, '3D Slicer*\Slicer.exe', 'Program Files (x86)'),
        @($programFilesX86, 'Slicer*\Slicer.exe', 'Program Files (x86)'),
        @($localAppData, 'slicer.org\*\Slicer.exe', '用户本地安装'),
        @($localAppData, 'NA-MIC\*\Slicer.exe', '用户本地安装'),
        @($localAppData, 'Programs\3D Slicer*\Slicer.exe', '用户本地安装'),
        @($localAppData, 'Programs\Slicer*\Slicer.exe', '用户本地安装'),
        @($userProfile, 'Apps\3D Slicer*\Slicer.exe', '用户目录'),
        @($userProfile, 'Apps\Slicer*\Slicer.exe', '用户目录'),
        @($userProfile, '3D Slicer*\Slicer.exe', '用户目录'),
        @($userProfile, 'Slicer*\Slicer.exe', '用户目录')
    )
    foreach ($entry in $patterns) {
        $patternPath = Join-Path $entry[0] $entry[1]
        Get-ChildItem -Path $patternPath -File -ErrorAction SilentlyContinue |
            Sort-Object FullName -Descending |
            ForEach-Object { Add-SlicerCandidate $_.FullName $entry[2] }
    }
    return @($script:discovered)
}

function Select-SlicerCandidate($choices) {
    $form = New-Object System.Windows.Forms.Form
    $form.Text = $title
    $form.StartPosition = 'CenterScreen'
    $form.Width = 680
    $form.Height = 360
    $form.MinimizeBox = $false
    $form.MaximizeBox = $false

    $label = New-Object System.Windows.Forms.Label
    $label.Text = U @(0x627E,0x5230,0x591A,0x4E2A,0x0020,0x0033,0x0044,0x0020,0x0053,0x006C,0x0069,0x0063,0x0065,0x0072,0xFF0C,0x8BF7,0x9009,0x62E9,0x4E00,0x4E2A,0xFF1A)
    $label.AutoSize = $true
    $label.Left = 16
    $label.Top = 15
    $form.Controls.Add($label)

    $list = New-Object System.Windows.Forms.ListBox
    $list.Left = 16
    $list.Top = 45
    $list.Width = 632
    $list.Height = 215
    foreach ($choice in $choices) {
        [void]$list.Items.Add(('{0}  [{1}]' -f $choice.Path, $choice.Source))
    }
    if ($list.Items.Count -gt 0) { $list.SelectedIndex = 0 }
    $form.Controls.Add($list)

    $useButton = New-Object System.Windows.Forms.Button
    $useButton.Text = $okText
    $useButton.Left = 440
    $useButton.Top = 275
    $useButton.Width = 105
    $useButton.DialogResult = [System.Windows.Forms.DialogResult]::OK
    $useButton.Add_Click({ if ($list.SelectedIndex -ge 0) { $form.Tag = $choices[$list.SelectedIndex] } })
    $form.Controls.Add($useButton)
    $cancelButton = New-Object System.Windows.Forms.Button
    $cancelButton.Text = $cancelText
    $cancelButton.Left = 555
    $cancelButton.Top = 275
    $cancelButton.Width = 90
    $cancelButton.DialogResult = [System.Windows.Forms.DialogResult]::Cancel
    $form.Controls.Add($cancelButton)
    $form.AcceptButton = $useButton
    $form.CancelButton = $cancelButton
    if ($form.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
        return $form.Tag
    }
    return $null
}

function Show-MissingSlicerDialog {
    $form = New-Object System.Windows.Forms.Form
    $form.Text = $title
    $form.StartPosition = 'CenterScreen'
    $form.Width = 510
    $form.Height = 250
    $form.MinimizeBox = $false
    $form.MaximizeBox = $false
    $label = New-Object System.Windows.Forms.Label
    $label.Text = (U @(0x6CA1,0x6709,0x627E,0x5230,0x0020,0x0033,0x0044,0x0020,0x0053,0x006C,0x0069,0x0063,0x0065,0x0072,0x3002) + [Environment]::NewLine + [Environment]::NewLine + U @(0x672C,0x8F6F,0x4EF6,0x9700,0x8981,0x0020,0x0033,0x0044,0x0020,0x0053,0x006C,0x0069,0x0063,0x0065,0x0072,0x0020,0x624D,0x80FD,0x8FD0,0x884C,0x3002))
    $label.AutoSize = $false
    $label.Left = 18
    $label.Top = 18
    $label.Width = 460
    $label.Height = 75
    $form.Controls.Add($label)
    $chooseButton = New-Object System.Windows.Forms.Button
    $chooseButton.Text = $chooseSlicerText
    $chooseButton.Left = 18
    $chooseButton.Top = 125
    $chooseButton.Width = 145
    $chooseButton.DialogResult = [System.Windows.Forms.DialogResult]::Yes
    $form.Controls.Add($chooseButton)
    $helpButton = New-Object System.Windows.Forms.Button
    $helpButton.Text = $installHelpText
    $helpButton.Left = 173
    $helpButton.Top = 125
    $helpButton.Width = 125
    $helpButton.DialogResult = [System.Windows.Forms.DialogResult]::Retry
    $form.Controls.Add($helpButton)
    $closeButton = New-Object System.Windows.Forms.Button
    $closeButton.Text = $closeText
    $closeButton.Left = 380
    $closeButton.Top = 125
    $closeButton.Width = 95
    $closeButton.DialogResult = [System.Windows.Forms.DialogResult]::Cancel
    $form.Controls.Add($closeButton)
    $form.AcceptButton = $chooseButton
    $form.CancelButton = $closeButton
    return $form.ShowDialog()
}

function Select-SlicerExecutable {
    # A previously chosen version is a user preference.  Do not ask the user
    # to choose again merely because another installation was later added.
    $configured = Read-ConfiguredSlicer
    if ($configured) {
        return [PSCustomObject]@{ Path = $configured; Source = '项目设置' }
    }
    $choices = @(Find-SlicerCandidates)
    if ($choices.Count -eq 1) {
        return $choices[0]
    }
    if ($choices.Count -gt 1) {
        return Select-SlicerCandidate $choices
    }
    while ($true) {
        $answer = Show-MissingSlicerDialog
        if ($answer -eq [System.Windows.Forms.DialogResult]::Retry) {
            Show-Message (U @(0x8BF7,0x4ECE,0x0020,0x0033,0x0044,0x0020,0x0053,0x006C,0x0069,0x0063,0x0065,0x0072,0x5B98,0x65B9,0x7F51,0x7AD9,0x4E0B,0x8F7D,0x5E76,0x5B89,0x88C5,0x3002) + [Environment]::NewLine + [Environment]::NewLine + U @(0x5B89,0x88C5,0x5B8C,0x6210,0x540E,0xFF0C,0x518D,0x6B21,0x53CC,0x51FB,0x201C,0x542F,0x52A8,0x5B9E,0x9A8C,0x5E73,0x53F0,0x201D,0x3002) + [Environment]::NewLine + [Environment]::NewLine + U @(0x5982,0x679C,0x4F60,0x5DF2,0x7ECF,0x5B89,0x88C5,0xFF0C,0x53EF,0x4EE5,0x7528,0x201C,0x9009,0x62E9,0x0020,0x0053,0x006C,0x0069,0x0063,0x0065,0x0072,0x002E,0x0065,0x0078,0x0065,0x201D,0x624B,0x52A8,0x6307,0x5B9A,0x3002))
            continue
        }
        if ($answer -eq [System.Windows.Forms.DialogResult]::Yes) {
            $dialog = New-Object System.Windows.Forms.OpenFileDialog
            $dialog.Title = $chooseSlicerText
            $dialog.Filter = 'Slicer.exe|Slicer.exe|Executable files (*.exe)|*.exe'
            $dialog.CheckFileExists = $true
            if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
                if ([System.IO.Path]::GetFileName($dialog.FileName) -ieq 'Slicer.exe') {
                    $selected = [PSCustomObject]@{ Path = (Resolve-Path -LiteralPath $dialog.FileName).Path; Source = '手动选择' }
                    Save-ConfiguredSlicer $selected.Path
                    return $selected
                }
                Show-Message (U @(0x8BF7,0x9009,0x62E9,0x6587,0x4EF6,0x540D,0x4E3A,0x0020,0x0053,0x006C,0x0069,0x0063,0x0065,0x0072,0x002E,0x0065,0x0078,0x0065,0x7684,0x6587,0x4EF6,0x3002))
            }
            continue
        }
        return $null
    }
}

if (-not (Test-Path -LiteralPath $modulePath -PathType Container) -or -not (Test-Path -LiteralPath $startupScript -PathType Leaf)) {
    Show-Message (U @(0x5B9E,0x9A8C,0x5E73,0x53F0,0x6587,0x4EF6,0x4E0D,0x5B8C,0x6574,0xFF0C,0x8BF7,0x91CD,0x65B0,0x4E0B,0x8F7D,0x9879,0x76EE,0x3002)) $title ([System.Windows.Forms.MessageBoxIcon]::Error)
    exit 1
}

$selected = Select-SlicerExecutable
if (-not $selected) {
    exit 1
}
Save-ConfiguredSlicer $selected.Path

try {
    $argumentLine = '--no-splash --additional-module-path "{0}" --python-script "{1}"' -f $modulePath, $startupScript
    Start-Process -FilePath $selected.Path -ArgumentList $argumentLine -WorkingDirectory $projectRoot -ErrorAction Stop | Out-Null
} catch {
    $detail = (U @(0x5B9E,0x9A8C,0x5E73,0x53F0,0x542F,0x52A8,0x5931,0x8D25,0xFF0C,0x8BF7,0x70B9,0x51FB,0x201C,0x67E5,0x770B,0x6280,0x672F,0x4FE1,0x606F,0x201D,0x67E5,0x770B,0x539F,0x56E0,0x3002)) + [Environment]::NewLine + [Environment]::NewLine + $_.Exception.Message
    Show-Message $detail $title ([System.Windows.Forms.MessageBoxIcon]::Error)
    exit 1
}
