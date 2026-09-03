# TMJ-Condyle-3D

下颌髁突 MRI 三维分割实验平台

TMJ MRI 标注、训练、评估与三维分割

## 普通用户使用

1. 安装 3D Slicer。
2. 双击项目根目录的 **启动实验平台.bat**。
3. 按软件提示操作。

启动器会自动寻找 3D Slicer、加载项目模块，并直接进入“下颌髁突三维分割实验平台”。
不需要打开 PowerShell，不需要配置 Slicer 模块路径，也不需要手工运行 Python 或 nnU-Net 命令。

第一次打开会出现“首次使用引导”，首页会直接告诉你下一步该做什么。可选地双击
**创建桌面快捷方式.bat**，在桌面创建“下颌髁突三维分割实验平台”快捷方式。

软件内的基本流程是：

    导入 MRI → 标注下颌髁突 → 确认标注 → 准备训练数据
    → 训练模型 → 查看 Dice / IoU / HD95 → 自动分割新 MRI → 查看 3D

只有“已确认（VERIFIED）”的人工标注才会进入正式训练。没有真实牙科人工标注、兼容训练显卡
或真实评价文件时，软件会明确显示等待原因，不会生成假 mask 或假指标。

## 当前真实状态

本仓库包含 Slicer 实验平台、病例与标注、训练数据准备、nnU-Net v2 训练/预测/评价接口、
实验记录和文档。仓库不包含患者数据、人工 mask、模型 checkpoint 或实验指标。

医学数据只保存在本地 `workspace`，不会进入 Git。当前工作区中的 `case_001.nii.gz` 仅用于
读取和 GUI smoke 检查；此前 GUI 验收生成的测试 mask 已隔离，不能用于正式训练。

## 开发者安装与高级使用

建议使用已安装的 Python 3.10 或 3.11 建立隔离环境。项目脚本可以直接从项目根目录运行。

~~~powershell
cd C:/code/TMJ-Condyle-3D
py -3.10 -m venv .venv
./.venv/Scripts/python.exe -m pip install -U pip
./.venv/Scripts/python.exe -m pip install -r requirements/dev.txt
~~~

若准备正式使用 nnU-Net v2：

~~~powershell
./.venv/Scripts/python.exe -m pip install -r requirements/nnunet-v2.txt
./.venv/Scripts/python.exe scripts/check_environment.py --write-lock
~~~

根据当前实际版本，nnU-Net 环境变量可以在当前 PowerShell 会话设置：

~~~powershell
$env:nnUNet_raw = "C:/code/TMJ-Condyle-3D/workspace/nnUNet_raw"
$env:nnUNet_preprocessed = "C:/code/TMJ-Condyle-3D/workspace/nnUNet_preprocessed"
$env:nnUNet_results = "C:/code/TMJ-Condyle-3D/workspace/nnUNet_results"
~~~

## 从 DICOM 到标注

1. 使用 `scripts/inspect_dicom.py` 检查 series 数量。输出会主动隐藏 UID 和患者字段。
2. 使用 `scripts/dicom_to_nifti.py --case-id case_001` 转换唯一 Series。多 Series 时先根据
   匿名 `series_index` 选择目标序列；项目不会按文件名猜顺序。
3. 双击 **启动实验平台.bat**。
4. 在“病例与标注”页面使用中文“画笔”和“擦除”逐层标出髁突。
5. 完成技术检查并保存后，点击“确认本例标注”；确认前不会进入训练。
6. 在“训练数据”中检查并准备数据，再在“模型训练”中开始真实 5 折实验。

详细步骤见 docs/ANNOTATION_GUIDE_zh-CN.md 和 docs/DATASET_GUIDE_zh-CN.md。

## 构建 nnU-Net v2 数据集

~~~powershell
python scripts/validate_dataset.py
python scripts/build_nnunet_dataset.py
~~~

输出为 workspace/nnUNet_raw/Dataset501_CondyleMRI：

    dataset.json
    imagesTr/case_001_0000.nii.gz
    labelsTr/case_001.nii.gz
    imagesTs/

数据集编号 501 只是本项目内部编号，不继承旧仓库的 runtime、trainer 或 v1
数据格式。所有训练病例数量从 manifest/目录实际生成，不硬编码。

## 官方 nnU-Net v2 preprocessing 与训练

先确认命令来自当前安装版本：

~~~powershell
nnUNetv2_plan_and_preprocess -h
./.venv/Scripts/nnUNetv2_plan_and_preprocess.exe -d 501 --verify_dataset_integrity -c 3d_fullres
~~~

然后检查：

    workspace/nnUNet_preprocessed/Dataset501_CondyleMRI/dataset_fingerprint.json
    workspace/nnUNet_preprocessed/Dataset501_CondyleMRI/nnUNetPlans.json

用项目脚本把实际 planner 决定整理到报告（没有这两个 JSON 时脚本会阻断）：

~~~powershell
python scripts/report_planner.py
~~~

报告会记录 median image size、spacing、target spacing、patch size、batch size、
normalization、resampling 和当前 MRI 的 anisotropy 描述；这些值全部来自本次
官方 planner 输出，不由项目手工覆盖。

正式训练只使用官方命令和 3d_fullres：

~~~powershell
./scripts/train_all_folds.ps1
~~~

或者：

~~~powershell
python scripts/train_all_folds.py --device cuda
python scripts/train_all_folds.py --device cuda --resume
~~~

脚本会按 0、1、2、3、4 依次运行，已存在 checkpoint_final.pth 的 fold 默认跳过，
不删除已有训练结果。原生输出应包含 progress.png、checkpoint_final.pth 和
validation/summary.json。

## OOF 评价

不能用训练集预测计算指标。先为每个 fold 只对它的 validation case 生成预测：

~~~powershell
python scripts/run_oof_predictions.py --device cuda
python scripts/evaluate_cv.py
~~~

评价报告：

    workspace/reports/metrics_per_case.csv
    workspace/reports/metrics_summary.csv
    workspace/reports/cv_report.md
    workspace/reports/figures/

Dice、IoU 在二值 mask 上计算；HD95 使用真实 image spacing，单位为 mm。
同一患者的左右侧通过 group_id 绑定，不能跨 train/validation。

## 新病例预测和 SlicerNNUnet

命令行五折 ensemble：

~~~powershell
python scripts/predict.py C:/private/case_new.nii.gz
~~~

输出默认到 workspace/predictions/case_new_condyle.nii.gz，并验证与输入的 shape、
spacing、origin、direction 和标签值。

训练完成后整理 SlicerNNUnet 模型目录：

~~~powershell
python scripts/export_slicer_model.py
~~~

应选择并测试以下配置目录：

    workspace/slicer_models/Dataset501_CondyleMRI/
      nnUNetTrainer__nnUNetPlans__3d_fullres/
        dataset.json
        plans.json
        fold_0/checkpoint_final.pth
        ...
        fold_4/checkpoint_final.pth

在 3D Slicer 安装 SlicerNNUnet，设置模型路径，选择 MRI，Apply，检查生成的
Mandibular Condyle segmentation。人工标注和结果查看优先通过本项目的“下颌髁突三维分割实验平台”
工作台完成；详细说明见 docs/SLICER_GUIDE_zh-CN.md。

## 参考项目与归属

本项目重新实现了流程代码，不把第三方仓库复制进源代码：

- TMJ 任务定义、TMJ MRI 数据组织、旧仓库的 data-demo/转换/评价思路参考
  [MenxLi/UNets-TMJ](https://github.com/MenxLi/UNets-TMJ)。该仓库顶层未发现
  LICENSE 文件；其捆绑依赖有自己的许可证，所以本项目只借鉴公开实现思路，不复制
  未明确授权的代码，也不继续使用其 nnU-Net v1 runtime、Task501、trainer 或环境。
- 训练基于
  [MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet) 当前 nnU-Net v2，
  其代码为 Apache-2.0。命令、dataset format、fingerprint/plans、手工 split、
  training 和 inference 以实际安装版本为准。
- 3D Slicer 部署使用
  [KitwareMedical/SlicerNNUnet](https://github.com/KitwareMedical/SlicerNNUnet)，
  其仓库许可证为 BSD 3-Clause；SlicerNNUnet 需要保留官方 nnU-Net 权重目录结构。
- 可视化和标注基于 [3D Slicer](https://www.slicer.org/) 的视图与嵌入式 Segment
  Editor 编辑引擎；用户操作集中在本项目中文工作台内。

参考仓库的临时 clone 位于 references/，并被 .gitignore 忽略。

## Git 安全

提交前从项目根目录执行：

~~~powershell
git status
git diff
git ls-files
python scripts/scan_git_safety.py --staged
~~~

发布前确认没有 DICOM、NIfTI、mask、JSON sidecar、患者字段、checkpoint、缓存或
运行时。当前工作区不自动创建或推送 GitHub 公共仓库；远程仓库创建、账号授权和
push 必须由项目维护者在确认扫描结果后自行执行。

## 开发者 / 高级使用

如需手工加载模块，可将 `slicer/TMJCondyleAnnotator` 作为 Slicer 的
Additional module path，然后重启 Slicer。日常使用推荐使用项目根目录的一键启动器，
这样不会改变 Slicer 的全局设置。

## 项目目标

医生在 3D Slicer 中给少量 TMJ MRI 手工标注下颌髁突，nnU-Net v2 3d_fullres
学习这些三维 mask，自动分割新的 TMJ MRI，并在 3D Slicer 中显示下颌髁突三维模型。
