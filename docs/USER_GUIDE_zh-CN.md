# TMJ-Condyle-3D 用户指南

这份指南描述从一例 MRI 到三维髁突预测的完整操作顺序。项目面向课设/实验，
不是临床诊断软件。

## 你要准备什么

- Windows 电脑。
- 3D Slicer。
- 医院提供的 TMJ MRI；本项目不规定医院采集协议。
- 一位了解颞下颌关节解剖的医生/牙医负责人工标注。
- 训练阶段需要兼容当前 PyTorch 的 CUDA GPU。

不要把患者原始文件放进 Git，也不要把患者姓名、PatientID、StudyInstanceUID、
AccessionNumber、出生日期或医院信息写入 case id、报告或图标题。

## 一个病例是什么

一个 TMJ 侧就是一个 case。建议使用 case_001、case_002 这样的匿名名称。左右侧
如果属于同一患者，要在私有 manifest 中使用相同 group_id，不能只因为文件名不同
就当成两个独立患者。

## 操作总览

1. 检查 DICOM series，不按文件名排序。
2. 将一个 series 转换为标准 NIfTI。
3. 在 Slicer 中用 TMJ Condyle Annotator 创建唯一 segment。
4. 在原生 Segment Editor 逐层标注，并从三个方向检查。
5. Show 3D，确认表面合理。
6. 保存非空的 0/1 mask。
7. 用 validate_dataset.py 生成 QC。
8. 至少有五个互不相同的 group 后构建 grouped 5-fold。
9. 用官方 nnU-Net v2 plan/preprocess 和 3d_fullres 训练。
10. 用对应 validation fold 生成 OOF 预测，再评价 Dice、IoU、HD95。
11. 五折 ensemble 预测新 MRI。
12. 在 SlicerNNUnet 中加载模型并 Show 3D。

## 状态含义

manifest 的 annotation_status 使用：

- NEW：已导入 MRI，还没有开始或完成标注。
- ANNOTATING：正在 Slicer 中标注。
- ANNOTATED：医生完成了 mask 导出。
- VERIFIED：第二位医生或项目负责人完成复核。

只有 ANNOTATED/VERIFIED 且通过 QC 的病例才会进入训练数据集。

## 重要的真实限制

没有人工髁突 mask 时，整个项目可以检查软件链路，但不能报告正式模型结果。
只有少量病例时，结果应描述为初步可行性验证，不能写成临床泛化能力结论。
