下颌髁突三维分割实验平台 v0.1.0
Windows x64 离线版

安装版
1. 双击 TMJ-Condyle-3D-Setup-x64.exe。
2. 按安装向导操作；可以选择创建桌面和开始菜单快捷方式。
3. 双击桌面上的“下颌髁突三维分割实验平台”。

便携版
1. 解压整个 TMJ-Condyle-3D-Portable-x64.zip，不要只解压其中一个文件。
2. 双击解压目录中的 TMJ-Condyle-3D.exe。
3. 请保留解压后的整个目录；删除或移动其中的文件会导致软件无法启动。

第一次启动
软件会自动检查内置运行环境，并在 Documents\TMJ-Condyle-3D\workspace
创建空的病例与实验数据目录。请按软件提示导入匿名医学数据和进行标注。
病例、标注、模型和实验结果默认保存在：
Documents\TMJ-Condyle-3D\workspace

训练提示
软件默认使用 CPU，正式训练也可以在 CPU 上运行（速度可能较慢）。只有在训练页主动选择
“GPU”时，软件才会检查 CUDA。如果没有检测到可用 CUDA，会用中文提示你安装兼容的 NVIDIA
显卡驱动和 GPU 版运行环境；当前 CPU 功能仍可正常使用。本离线版携带 CPU 版 PyTorch，
不携带 CUDA 或 NVIDIA 驱动。
软件已经包含训练所需的 Python、CPU 版 PyTorch 和 nnU-Net；不需要另外安装 Python、
pip、nnU-Net 或 3D Slicer。

卸载
卸载只删除程序文件。Documents\TMJ-Condyle-3D 中的病例、标注、模型和实验结果不会被删除。
