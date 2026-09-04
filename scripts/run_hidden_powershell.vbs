Option Explicit

Dim shell, fileSystem, scriptsDirectory, projectRoot, mode, scriptPath
Dim commandLine, exitCode
Set shell = CreateObject("WScript.Shell")
Set fileSystem = CreateObject("Scripting.FileSystemObject")

If WScript.Arguments.Count < 1 Then
    exitCode = 2
Else
    mode = LCase(WScript.Arguments(0))
    scriptsDirectory = fileSystem.GetParentFolderName(WScript.ScriptFullName)
    projectRoot = fileSystem.GetParentFolderName(scriptsDirectory)
    If mode = "platform" Then
        scriptPath = fileSystem.BuildPath(projectRoot, _
            U("542F52A85B9E9A8C5E7353F0") & ".ps1")
    ElseIf mode = "shortcut" Then
        scriptPath = fileSystem.BuildPath(scriptsDirectory, "create_desktop_shortcut.ps1")
    Else
        exitCode = 2
    End If
    If exitCode <> 2 Then
        If fileSystem.FileExists(scriptPath) Then
            commandLine = QuoteValue(FindPowerShell(fileSystem, shell)) & _
                " -NoLogo -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File " & _
                QuoteValue(scriptPath)
            exitCode = shell.Run(commandLine, 0, True)
        Else
            exitCode = 2
        End If
    End If
End If

If exitCode <> 0 Then
    MsgBox U("64CD4F5C59318D25FF0C8BF791CD65B0" & "53CC51" & "51FB" & _
        "542F" & "52A8" & "5B9E" & "9A8C" & "5E73" & "53F0" & "3002") & _
        vbCrLf & vbCrLf & _
        U("9519" & "8BEF" & "4EE3" & "7801" & "FF1A") & CStr(exitCode), _
        vbCritical, U("4E0B" & "988C" & "9AC1" & "7A81" & "4E09" & "7EF4" & _
        "5206" & "5272" & "5B9E" & "9A8C" & "5E73" & "53F0")
End If

WScript.Quit exitCode

Function FindPowerShell(fileSystemObject, shellObject)
    Dim candidate
    candidate = shellObject.ExpandEnvironmentStrings( _
        "%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe")
    If fileSystemObject.FileExists(candidate) Then
        FindPowerShell = candidate
    Else
        FindPowerShell = "PowerShell.exe"
    End If
End Function

Function QuoteValue(value)
    QuoteValue = Chr(34) & Replace(value, Chr(34), Chr(34) & Chr(34)) & Chr(34)
End Function

Function U(hexText)
    Dim index, result
    result = ""
    For index = 1 To Len(hexText) Step 4
        result = result & ChrW(CLng("&H" & Mid(hexText, index, 4)))
    Next
    U = result
End Function
