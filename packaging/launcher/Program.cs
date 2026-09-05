using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Windows.Forms;

namespace TMJCondyle3D.Launcher;

internal static class Program
{
    private const string ProductTitle = "下颌髁突三维分割实验平台";
    private const string ProductVersion = "0.1.0";

    [STAThread]
    private static int Main()
    {
        try
        {
            var appRoot = Path.GetFullPath(AppContext.BaseDirectory);
            var userData = ResolveUserData(appRoot);
            PrepareUserData(userData, appRoot);

            var slicer = RequireFile(
                Path.Combine(appRoot, "runtime", "slicer", "Slicer.exe"),
                "3D 显示组件缺失，请重新安装软件。"
            );
            _ = RequireFile(
                Path.Combine(appRoot, "runtime", "python", "python.exe"),
                "实验环境不完整，请重新安装软件。"
            );
            var startupScript = RequireFile(
                Path.Combine(appRoot, "slicer", "startup_tmj.py"),
                "实验平台启动脚本缺失，请重新安装软件。"
            );
            var modulePath = RequireDirectory(
                Path.Combine(appRoot, "slicer", "TMJCondyleAnnotator"),
                "实验平台模块缺失，请重新安装软件。"
            );

            var environment = BuildEnvironment(appRoot, userData);
            var startInfo = new ProcessStartInfo
            {
                FileName = slicer,
                WorkingDirectory = appRoot,
                UseShellExecute = false,
                CreateNoWindow = true,
                WindowStyle = ProcessWindowStyle.Hidden,
            };
            startInfo.ArgumentList.Add("--no-splash");
            startInfo.ArgumentList.Add("--additional-module-path");
            startInfo.ArgumentList.Add(modulePath);
            startInfo.ArgumentList.Add("--python-script");
            startInfo.ArgumentList.Add(startupScript);
            foreach (var pair in environment)
            {
                startInfo.Environment[pair.Key] = pair.Value;
            }

            using var process = Process.Start(startInfo)
                ?? throw new InvalidOperationException("Slicer 进程没有启动。");
            // A normal GUI stays alive.  An immediate non-zero exit is a useful
            // signal for a corrupt/mismatched runtime and should be explained.
            if (process.WaitForExit(10_000) && process.ExitCode != 0)
            {
                throw new InvalidOperationException(
                    $"3D 显示组件启动失败（退出代码 {process.ExitCode}）。请重新安装软件。"
                );
            }
            return 0;
        }
        catch (FileNotFoundException exception)
        {
            ShowError(exception.Message);
            return 2;
        }
        catch (UnauthorizedAccessException)
        {
            ShowError("用户数据目录不可写，请选择有写入权限的 Documents 目录或检查磁盘权限。");
            return 3;
        }
        catch (IOException exception)
        {
            ShowError($"无法准备用户数据目录。请检查磁盘空间和权限。\n\n{exception.Message}");
            return 4;
        }
        catch (Exception exception)
        {
            ShowError($"实验平台启动失败。\n\n{exception.Message}");
            return 5;
        }
    }

    private static string ResolveUserData(string appRoot)
    {
        var configured = Environment.GetEnvironmentVariable("TMJ_USER_DATA_DIR");
        if (!string.IsNullOrWhiteSpace(configured))
        {
            var candidate = Path.GetFullPath(configured);
            if (!PathsEqual(candidate, appRoot))
            {
                return candidate;
            }
        }

        var documents = Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments);
        if (string.IsNullOrWhiteSpace(documents))
        {
            documents = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        }
        if (string.IsNullOrWhiteSpace(documents))
        {
            throw new IOException("无法找到当前用户的 Documents 目录。");
        }
        return Path.GetFullPath(Path.Combine(documents, "TMJ-Condyle-3D", "workspace"));
    }

    private static void PrepareUserData(string userData, string appRoot)
    {
        if (PathsEqual(userData, appRoot))
        {
            throw new IOException("用户数据不能写入程序安装目录。");
        }
        Directory.CreateDirectory(userData);
        var drive = new DriveInfo(Path.GetPathRoot(userData) ?? userData);
        if (drive.IsReady && drive.AvailableFreeSpace < 1L * 1024 * 1024 * 1024)
        {
            throw new IOException("可用磁盘空间不足 1 GB，请清理空间后重试。");
        }

        var probe = Path.Combine(userData, ".tmj-write-test");
        File.WriteAllText(probe, ProductVersion, Encoding.UTF8);
        File.Delete(probe);
        foreach (var name in new[]
        {
            "raw", "nifti", "labels", "predictions", "reports", "nnUNet_raw",
            "nnUNet_preprocessed", "nnUNet_results", "slicer_models", "experiments",
            "exports", "logs",
        })
        {
            Directory.CreateDirectory(Path.Combine(userData, name));
        }
    }

    private static Dictionary<string, string> BuildEnvironment(string appRoot, string userData)
    {
        var environment = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        foreach (System.Collections.DictionaryEntry item in Environment.GetEnvironmentVariables())
        {
            var key = item.Key?.ToString();
            if (!string.IsNullOrWhiteSpace(key) && item.Value is not null)
            {
                environment[key] = item.Value.ToString() ?? string.Empty;
            }
        }
        environment.Remove("PYTHONHOME");
        environment.Remove("PYTHONPATH");
        environment.Remove("PYTHONUSERBASE");

        var pythonRoot = Path.Combine(appRoot, "runtime", "python");
        var slicerRoot = Path.Combine(appRoot, "runtime", "slicer");
        var path = string.Join(
            Path.PathSeparator,
            pythonRoot,
            Path.Combine(pythonRoot, "DLLs"),
            Path.Combine(pythonRoot, "Scripts"),
            slicerRoot,
            environment.TryGetValue("PATH", out var currentPath) ? currentPath : string.Empty
        );
        environment["PATH"] = path;
        environment["TMJ_APP_ROOT"] = appRoot;
        environment["TMJ_USER_DATA_DIR"] = userData;
        environment["nnUNet_raw"] = Path.Combine(userData, "nnUNet_raw");
        environment["nnUNet_preprocessed"] = Path.Combine(userData, "nnUNet_preprocessed");
        environment["nnUNet_results"] = Path.Combine(userData, "nnUNet_results");
        environment["TMJ_RUNTIME_MODE"] = "packaged";
        environment["PYTHONNOUSERSITE"] = "1";
        return environment;
    }

    private static string RequireFile(string path, string message)
    {
        if (!File.Exists(path))
        {
            throw new FileNotFoundException(message, path);
        }
        return path;
    }

    private static string RequireDirectory(string path, string message)
    {
        if (!Directory.Exists(path))
        {
            throw new FileNotFoundException(message, path);
        }
        return path;
    }

    private static bool PathsEqual(string left, string right) =>
        string.Equals(
            Path.TrimEndingDirectorySeparator(Path.GetFullPath(left)),
            Path.TrimEndingDirectorySeparator(Path.GetFullPath(right)),
            StringComparison.OrdinalIgnoreCase
        );

    private static void ShowError(string message)
    {
        MessageBox.Show(
            message,
            ProductTitle,
            MessageBoxButtons.OK,
            MessageBoxIcon.Error
        );
    }
}
