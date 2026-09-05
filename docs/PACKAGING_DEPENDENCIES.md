# TMJ-Condyle-3D v0.1.0 发布依赖审计

本文是 Windows x64 离线发布的依赖清单。它描述构建输入、发布目录中的实际运行时，以及普通用户不需要安装的外部组件。版本来自当前开发环境的 `pip freeze`、import audit 和构建脚本；发布前由 `packaging/verify_runtime.py` 再次检查。

## 运行时边界

发布目录只有一个用户入口 `TMJ-Condyle-3D.exe`。它是 self-contained .NET 8 `win-x64` launcher，计算自身目录为 `APP_ROOT`，创建 `Documents\TMJ-Condyle-3D\workspace`，清除 `PYTHONHOME`、`PYTHONPATH` 和 `PYTHONUSERBASE`，然后只启动：

- `APP_ROOT\runtime\slicer\Slicer.exe`
- `APP_ROOT\runtime\python\python.exe`
- `APP_ROOT\slicer\startup_tmj.py`

Python 子进程使用绝对路径和 `python -m` 模块入口；不使用 `python`、`py`、`pip` 或 PATH 中的 nnU-Net console script。`nnUNet_raw`、`nnUNet_preprocessed`、`nnUNet_results` 都指向用户工作区。Slicer 的 embedded Python 环境不传递给外部 CPython。

## 组件清单

| 组件 | 版本/来源 | 是否随包 | 许可证/notice | 运行时用途 |
|---|---|---:|---|---|
| TMJ-Condyle-3D | v0.1.0，本仓库 | 是 | MIT，见 `LICENSE` | Slicer 模块、数据导入、QC、训练编排、评估和导出 |
| 3D Slicer | 5.12.3，构建参数提供的官方/已审计 Windows 目录 | 是，`runtime/slicer` | Slicer 及其各依赖的许可证保留在 runtime 和 `licenses/` | MRI/NIfTI 导入、人工标注、三维显示、模块宿主 |
| CPython | 3.10.21，官方 Windows runtime | 是，`runtime/python` | Python Software Foundation License，见 `licenses/` | 运行全部外部 Python 脚本 |
| nnU-Net v2 | 2.8.1，Python package | 是 | Apache License 2.0 | dataset planning、preprocess、训练、预测 |
| PyTorch | 2.14.0，官方 Windows CPU wheel | 是 | BSD 风格许可，见 `licenses/` | CPU 张量计算和训练；默认训练设备 |
| torchvision | 0.29.0，官方 Windows CPU wheel | 是 | BSD，见 `licenses/` | PyTorch 图像相关依赖 |
| NumPy | 2.2.6 | 是 | BSD-3-Clause，含 OpenBLAS/LAPACK notice | 数组和数值计算 |
| SciPy | 1.15.3 | 是 | BSD-3-Clause，含其 bundled library notice | 科学计算和统计 |
| SimpleITK | 2.5.6 | 是 | Apache License 2.0 | 医学图像读写、几何和 metadata 处理 |
| matplotlib | 3.10.9 | 是 | PSF/BSD 及 font notice | 评估图和报告 |
| scikit-image | 0.25.2 | 是 | BSD-3-Clause | 图像处理和表面/指标辅助 |
| pandas | 2.3.3 | 是 | BSD-3-Clause | manifest、CSV 和结果表 |
| nibabel | 5.4.2 | 是 | MIT | NIfTI 辅助读取 |
| torchvision 及 nnU-Net 传递依赖 | 以 `packaging/requirements-lock.txt` 为准 | 是 | 每个 distribution 的 METADATA/license 文件复制到 `licenses/` | filelock、scikit-learn、batchgenerators、dynamic-network-architectures、imagecodecs、tifffile、requests 等 |
| VC++ runtime DLL | Python/Slicer/官方 wheels 随附的 DLL | 是或由 Windows 提供 | 各组件 notice；不复制来源不明 DLL | 原生 Python、Slicer 和科学包 DLL 加载 |
| NVIDIA Display Driver / GPU 版运行环境 | 由用户显卡厂商或单独 GPU 发行版提供 | 否 | 硬件/第三方运行时 | 仅在用户主动选择 GPU 时检测；本 CPU-only 包不携带 CUDA 或驱动 |

## Import audit

发布构建检查以下 import 和版本：`torch`、`torchvision`、`numpy`、`scipy`、`SimpleITK`、`matplotlib`、`skimage`、`pandas`、`nibabel`、`nnunetv2`。构建还检查三个 nnU-Net v2 模块入口：

- `nnunetv2.experiment_planning.plan_and_preprocess_entrypoints`
- `nnunetv2.run.run_training`
- `nnunetv2.inference.predict_from_raw_data`

CPU-only 发行版默认使用 CPU，因此无 NVIDIA GPU 的电脑仍可以完成 Python import、GUI、导入、人工标注、数据检查、3D 显示、CPU 训练和 CPU synthetic smoke。训练页只有在用户主动选择 GPU 时才调用 `torch.cuda.is_available()`；如果未检测到 CUDA，页面提示安装兼容的 NVIDIA 驱动和 GPU 版运行环境。本包不携带 CUDA、NVIDIA driver，也不会把 CPU-only PyTorch 自动变成 GPU 版。

## Slicer 模块与启动链

仓库中的 `slicer/TMJCondyleAnnotator` 和 `slicer/startup_tmj.py` 随程序文件复制。launcher 使用 `--additional-module-path` 和 `--python-script` 指向 `APP_ROOT` 下的文件；它不读取开发机的 Slicer 路径、模块路径或 Slicer 配置。Slicer 完整目录复制到临时新路径后由发布验收启动；用户数据不写入 Slicer 或 Program Files。

源仓库中的 `.bat`、PowerShell、VBS 仅保留给开发兼容场景，不进入发布 staging。正式安装和 ZIP 只暴露 self-contained launcher。

## 用户数据与发布排除项

程序文件和医学数据分离。发布 staging 不复制 `workspace`、`data`、`DICOM`、NIfTI、mask、checkpoint、OOF prediction、metrics、figures、日志、`.venv`、`.git`、tests 或开发机配置。synthetic E2E 只复制生成代码，测试运行时使用临时目录并在完成后删除。

发布前执行 `packaging/release_scan.py`，扫描文件名、目录名和应用代码文本中的患者字段、真实路径、凭据关键词以及医学数据扩展名。扫描发现任何病例/患者数据时构建失败。

## License audit 说明

`packaging/generate_notices.py` 读取随包 Python distribution 的 `METADATA` 和 `.dist-info/licenses`，把版本、license 字段、项目地址和 license 文件写入 `THIRD_PARTY_NOTICES.txt` 与 `licenses/`。Python 主许可证、项目许可证和 Slicer 主许可证在 staging 中另行保留。PyTorch 使用 CPU-only wheel，因此没有 CUDA runtime 或 NVIDIA 二进制需要复制；NVIDIA 驱动也不在包内。发布前若替换任一 wheel 或 Slicer 构建，必须重新运行 audit 和扫描。
