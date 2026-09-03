# 数据组织与隐私指南

## 目录

项目工作目录默认如下：

    workspace/
      raw/                 # 可放本地原始输入，但默认不进 Git
      nifti/               # 匿名病例 MRI，case_001.nii.gz
      labels/              # 人工导出的 0/1 mask
      predictions/         # OOF/新病例预测
      nnUNet_raw/
      nnUNet_preprocessed/
      nnUNet_results/
      slicer_models/
      reports/
      dataset_manifest.csv

workspace/ 的所有内容均被 .gitignore 忽略。references/ 临时参考仓库也被忽略。

## 一个病例

一个 TMJ 侧 = 一个 case。标准文件名：

    workspace/nifti/case_001.nii.gz       # image
    workspace/labels/case_001.nii.gz      # label

上面两个文件的文件名相同，实际通过目录区分。image 是 MRI 强度，label 是同
shape/spacing/origin/direction 的整数 label map。

如果一个 volume 同时有左右两个 TMJ，项目负责人必须在数据进入训练前固定策略：

- 推荐把左右侧切成两个明确的 ROI/case，并使用同一个患者 group_id；或者
- 所有此类 volume 都将左右两个髁突作为同一个 class 1。

禁止只对某一个病例临时改变定义。

## 私有 manifest

字段：

    case_id
    group_id
    side
    image_path
    label_path
    annotation_status
    geometry_valid
    label_valid
    notes

示例只使用匿名结构：

    case_001,group_001,L,workspace/nifti/case_001.nii.gz,workspace/labels/case_001.nii.gz,VERIFIED,true,true,

group_id 只能是匿名分组 id，不能是医院 PatientID。manifest 是私有文件，不应提交
到公开仓库。

病例状态分为 `NEW`（未标注）、`ANNOTATING`（标注中）、`ANNOTATED`（已保存但未确认）、
`UNVERIFIED`（来源未核实）和 `VERIFIED`（已完成医学复核）。训练数据页和默认构建脚本
只使用 `VERIFIED`；其它状态即使存在 mask，也不会进入正式训练。只有在高级命令中显式
传入 `--include-status` 才会覆盖这个默认安全边界。

## Label 规则

- 0 = background。
- 1 = mandibular_condyle。
- 不能出现 2、255 或其它值。
- 不能全空。
- 不要求只有一个 connected component；如果很多小孤立区域，QC 会 warning，医生
  必须检查但工具不会自动删除。

## Geometry 规则

image/label 必须拥有相同：

- shape；
- spacing；
- origin；
- direction/affine。

不要手工 resize 为固定立方体。当前少切片、强各向异性 MRI 应保持物理空间，由
nnU-Net v2 planner 根据 fingerprint 决定 preprocessing。

## DICOM 隐私

DICOM 到 NIfTI 的转换使用 SimpleITK/GDCM series reader。转换时：

- 使用 SeriesInstanceUID 选择 series；
- 不按文件名排序；
- 输出前擦除 DICOM 元数据；
- 重新读取并验证 geometry 与 voxel values；
- 报告不写出 UID、PatientName、PatientID 等字段。

原始 DICOM 仍可能含有敏感信息，只能放在医院/私有磁盘，不能放进 Git。输入目录
可以包含一个或多个子目录；项目会递归查找 DICOM series，但仍由 GDCM 提供切片
顺序，不读取文件名顺序。
