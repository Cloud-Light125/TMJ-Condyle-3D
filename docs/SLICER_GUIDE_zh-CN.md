# 3D Slicer 与 SlicerNNUnet 指南

普通用户只需双击项目根目录的“启动实验平台.bat”。下面的 Additional module paths、
Segment Editor 和 SlicerNNUnet 操作属于开发者/高级排障流程，不是日常使用要求。

## A. 安装项目标注模块

1. 打开 3D Slicer。
2. 进入 Edit > Application Settings > Modules。
3. 在 Additional module paths 添加：

       C:/code/TMJ-Condyle-3D/slicer/TMJCondyleAnnotator

4. 重启 Slicer，搜索 TMJ Condyle Annotator。

这个模块只包装 native Segment Editor，不开发新的 PySide6 GUI，也不替换 Paint、
Erase、Draw、Scissors、Islands 等工具。

## B. 人工标注

1. 在 Slicer 里加载匿名 NIfTI MRI。
2. 选择 MRI volume。
3. 填写 case_001 形式的匿名 case id。
4. 点击创建下颌髁突标注。
5. 点击开始标注进入 Segment Editor。
6. 逐层标注唯一 segment Mandibular Condyle。
7. 使用 Axial、Sagittal、Coronal 检查。
8. Show 3D 检查表面。
9. 保存髁突 Mask。

导出后用项目脚本再次 QC。Slicer 模块的导出 QC 通过不等于医学复核完成；医生
仍需确认结构边界。

## C. 安装 SlicerNNUnet

使用 Slicer Extension Manager 搜索并安装 SlicerNNUnet。按该扩展当前版本的说明
安装其 nnU-Net/PyTorch 依赖，必要时重启 Slicer。不要把旧 nnU-Net v1 环境混入
项目 venv 或 Slicer 环境。

训练完成后执行：

~~~powershell
python scripts/export_slicer_model.py
~~~

模型目录应类似：

    workspace/slicer_models/Dataset501_CondyleMRI/
      nnUNetTrainer__nnUNetPlans__3d_fullres/
        dataset.json
        plans.json
        fold_0/checkpoint_final.pth
        fold_1/checkpoint_final.pth
        fold_2/checkpoint_final.pth
        fold_3/checkpoint_final.pth
        fold_4/checkpoint_final.pth

在 SlicerNNUnet 的 Model path 选择：

    workspace/slicer_models/Dataset501_CondyleMRI/nnUNetTrainer__nnUNetPlans__3d_fullres

这里必须有 dataset.json，配置目录名必须保留 trainer、plans、configuration 三段。

## D. 新病例预测

1. 在 Slicer 加载一个未见过的匿名 MRI。
2. 打开 Segmentation > nnUNet 或搜索 nnUnet。
3. 设置模型目录。
4. 选择 MRI volume。
5. 确认 folds 为 0,1,2,3,4（具体界面以安装版本为准）。
6. 点击 Apply。
7. 等待日志完成。
8. 生成的 segmentation 应只有 Mandibular Condyle 标签。
9. 在 Segment Editor 中检查，点击 Show 3D。

如果模型目录无 checkpoint、dataset.json 或 plans.json，先回到
export_slicer_model.py 检查，不要手工猜目录名。

## E. 三维验收

验收时至少记录：

- MRI 是否加载；
- SlicerNNUnet 是否识别模型；
- Apply 是否完成；
- segmentation 是否能在 Segment Editor 编辑；
- 3D surface 是否显示；
- 输出是否仍在同一物理空间；
- 标题和截图是否只用匿名 case id。

当前没有训练好的真实模型时，SlicerNNUnet 项只能写
NOT RUN - WAITING FOR ANNOTATION。
