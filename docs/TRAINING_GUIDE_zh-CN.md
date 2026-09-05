# nnU-Net v2 训练与评价指南

本项目不改变 nnU-Net 默认网络结构、loss、trainer 或超参数，不做模型创新。
正式配置固定为官方 nnU-Net v2 的 3d_fullres。

## 首选方式：在 Slicer 平台中操作

普通用户不需要打开命令行。启动“下颌髁突三维分割实验平台”后：

1. 在“训练数据”点击“检查全部病例”，再点击“准备训练数据”。
2. 在“模型训练”阅读系统检查和病例数量提示，点击“开始 5 折训练”。
3. 训练任务使用后台 QProcess，页面会显示第几组正在训练、已完成的 fold、运行时间和日志；关闭软件后可重新打开并点击“继续未完成训练”。
4. 五折训练完成后，平台自动执行真实 OOF 预测和 `evaluate_cv.py`，再到“实验结果”选择实验记录、病例和 2D/3D 对比。
5. 具备完整五折模型后，在“自动分割”选择新 MRI，点击“开始自动分割”。

如果没有至少五个不同患者组，正式训练按钮会保持不可用。软件默认使用 CPU；只有主动选择 GPU 时才要求检查兼容 CUDA 的 NVIDIA GPU。
“使用 CPU 测试流程”只验证文件和流程状态，不训练模型，也不生成假结果。技术人员可以在“高级信息”中查看命令、路径和原始日志。

## 1. 环境

在项目 venv 中安装依赖，不修改系统 Python：

~~~powershell
py -3.10 -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements/nnunet-v2.txt
python scripts/check_environment.py
python scripts/check_environment.py --write-lock
~~~

确认输出中的 Python、PyTorch、CUDA、GPU、SimpleITK、NumPy、SciPy 和 nnUNet v2
版本。GPU 检查会实际创建 CUDA tensor、执行矩阵乘法和 synchronize。只有 smoke
通过才把环境标记为可正式训练。

## 2. 数据 QC

~~~powershell
python scripts/validate_dataset.py
~~~

任何 image/label geometry mismatch、NaN、Inf、非二值 label、空 mask、缺失 pair、
duplicate case id 都会阻止正式训练。报告：

    workspace/reports/dataset_validation.csv
    workspace/reports/dataset_validation.md

## 3. 构建 nnU-Net raw

~~~powershell
python scripts/build_nnunet_dataset.py
~~~

构建：

    workspace/nnUNet_raw/Dataset501_CondyleMRI/dataset.json
    workspace/nnUNet_raw/Dataset501_CondyleMRI/imagesTr/case_001_0000.nii.gz
    workspace/nnUNet_raw/Dataset501_CondyleMRI/labelsTr/case_001.nii.gz
    workspace/nnUNet_raw/Dataset501_CondyleMRI/imagesTs/

dataset.json 的关键字段：

    channel_names: {"0": "MRI"}
    labels: {"background": 0, "mandibular_condyle": 1}
    numTraining: 实际通过 manifest 读取
    file_ending: ".nii.gz"

## 4. Grouped 5-fold

builder 会按 group_id 生成五折，并写入：

    workspace/nnUNet_preprocessed/Dataset501_CondyleMRI/splits_final.json
    workspace/reports/fold_assignments.csv

也可以单独重建：

~~~powershell
python scripts/create_splits.py
~~~

每个 case 的 validation 只出现一次；同一 group 的所有 case 会进入同一 fold。
脚本检查 train group 与 validation group 的交集必须为空。少于五个 distinct
group 时不能开始正式 5-fold。

## 5. 官方 plan and preprocess

先查看本地安装版本参数：

~~~powershell
./.venv/Scripts/nnUNetv2_plan_and_preprocess.exe -h
~~~

然后运行：

~~~powershell
./.venv/Scripts/nnUNetv2_plan_and_preprocess.exe -d 501 --verify_dataset_integrity -c 3d_fullres
~~~

不要使用 v1 的 nnUNet_plan_and_preprocess。完成后阅读：

    workspace/nnUNet_preprocessed/Dataset501_CondyleMRI/dataset_fingerprint.json
    workspace/nnUNet_preprocessed/Dataset501_CondyleMRI/nnUNetPlans.json

实验报告中记录 median image size、原始 spacing、target spacing、patch size、
batch size、normalization、resampling strategy 和 anisotropy handling。不能提前
猜测 planner 如何处理少量 z slice，必须以实际 JSON 为准。

使用项目脚本生成可放入实验报告的实际 planner 摘要：

~~~powershell
python scripts/report_planner.py
~~~

它会读取 fingerprint 和 plans，记录当前 `3d_fullres` 的 target spacing、patch
size、batch size、normalization、resampling、transpose 以及各病例 spacing 的
各向异性描述；缺少 JSON 时会阻断，不会生成猜测值。

## 6. 五折训练

~~~powershell
./scripts/train_all_folds.ps1
~~~

等价命令：

~~~powershell
./.venv/Scripts/nnUNetv2_train.exe 501 3d_fullres 0 --npz
./.venv/Scripts/nnUNetv2_train.exe 501 3d_fullres 1 --npz
./.venv/Scripts/nnUNetv2_train.exe 501 3d_fullres 2 --npz
./.venv/Scripts/nnUNetv2_train.exe 501 3d_fullres 3 --npz
./.venv/Scripts/nnUNetv2_train.exe 501 3d_fullres 4 --npz
~~~

脚本会保留原生目录：

    workspace/nnUNet_results/Dataset501_CondyleMRI/
      nnUNetTrainer__nnUNetPlans__3d_fullres/
        fold_0/
        ...
        fold_4/

每个 fold 需要 checkpoint_final.pth、progress.png 和 validation/summary.json。
未完成 fold 可用：

~~~powershell
./scripts/train_all_folds.ps1 -Resume
~~~

train_all_folds.py 默认使用 CPU。只有传入 --device cuda 时才会先通过 CUDA smoke；
如果当前环境没有兼容 GPU，会直接输出 FULL TRAINING BLOCKED BY GPU。传入
--device cpu 会使用 CPU 正式执行。training_summary.md 会记录 fold、设备、
checkpoint_best/checkpoint_final 状态、耗时和可读取的 validation Dice。

## 7. OOF 预测与评价

先运行：

~~~powershell
python scripts/run_oof_predictions.py --device cuda
~~~

它对每个 fold 只把该 fold 的 validation case 交给该 fold 的模型，输出：

    workspace/predictions/oof/fold_0/case_xxx.nii.gz
    ...

再运行：

~~~powershell
python scripts/evaluate_cv.py
~~~

评价脚本只读取对应 fold 的 OOF 文件，拒绝缺失、geometry mismatch、非二值输出。
输出 Dice、IoU、HD95（真实 spacing，单位 mm），并写 mean/std、median、每病例
CSV、Markdown 和指标图。空 GT 在训练前拒绝；空预测可以作为模型失败的真实结果，
HD95 记为 inf 并在 summary 中显示有限值数量。

## 8. 新病例预测

~~~powershell
python scripts/predict.py C:/private/case_new.nii.gz
~~~

predict.py 明确传入 fold 0、1、2、3、4，使用五折 ensemble。输出 mask 仍只能是
0/1，并检查与输入 MRI 的 physical geometry。
