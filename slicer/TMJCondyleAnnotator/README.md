# TMJ Condyle Annotator

这是一个很薄的 3D Slicer Scripted Module。它不重写医学查看器，也不重写
Segment Editor 工具；它只把本项目的单结构任务固定为：

    0 = background
    1 = mandibular_condyle

在 3D Slicer 中通过 Edit > Application Settings > Modules > Additional module
paths 添加本目录，重启或刷新模块，然后搜索 TMJ Condyle Annotator。

使用流程：

1. 在 Slicer 中加载 .nii.gz MRI；DICOM 先用 Slicer DICOM 模块导入。
2. 选择 volume，填写匿名 case_001 形式的 case id。
3. 点击“创建下颌髁突标注”。
4. 点击“开始标注”，在原生 Segment Editor 中使用 Paint、Erase、Draw、
   Scissors、Islands、Fill between slices 等工具。
5. 点击 Show 3D 检查表面。
6. 点击“保存髁突 Mask”。模块会检查 geometry、0/1 标签、非空前景、体积，
   并在 connected components 很多时警告。

导出前请填写私有 workspace/dataset_manifest.csv 路径，以便自动更新病例状态。
导出的 mask 不会被模块自动“修正”；任何异常都会拒绝保存。
