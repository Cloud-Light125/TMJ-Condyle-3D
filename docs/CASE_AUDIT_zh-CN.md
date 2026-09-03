# 当前工作区病例审计

本记录只描述匿名文件状态，不包含患者姓名、PatientID、UID 或医院信息。

审计对象：`workspace/nifti/case_001.nii.gz`

结论：来源为 B（Codex/Slicer GUI 验收测试状态），已清除，不是正式人工标注；当前 GUI 将它显示为“未标注”，不能用于正式训练。

审计依据：

- 该病例曾出现在 Codex/Slicer GUI 验收流程中；验收截图显示的是保存成功后的测试流程，
  不是牙医人工标注证明。
- 原先位于 `workspace/labels/case_001.nii.gz` 的测试 mask 已移出正式 labels 目录并隔离，
  当前 `workspace/dataset_manifest.csv` 只有表头，没有正式病例记录。
- `case_001.nii.gz` 目前只保留用于读取、界面显示和标注 smoke 检查；本轮没有生成正式人工 mask，
  也没有把它标记为 VERIFIED。

如果以后发现来源无法确认的残留 mask，系统会标记为 `UNVERIFIED`，同样不能用于正式训练。
正式训练前，必须由牙医完成真实标注、技术检查和“确认本例标注”，使 manifest 状态变为
`VERIFIED`。训练数据页和命令行构建脚本默认只接收 `VERIFIED`。
