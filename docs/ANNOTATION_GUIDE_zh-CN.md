# 下颌髁突人工三维标注指南（给牙医）

本指南不要求你理解机器学习。你只需要知道如何在 3D Slicer 里看 MRI、给髁突
涂色、擦除错误，并保存经过检查的 mask。

## 1. 什么是 MRI

MRI 是磁共振成像。它不是普通照片，而是一组按空间位置排列的切片。不同组织在
不同 MRI 序列上的明暗可能不同，所以不要只靠“最亮的地方”判断髁突。请结合
解剖位置和 Axial、Sagittal、Coronal 三个方向一起确认。

当前案例可能只有少量厚切片，例如 z 方向切片很厚、面内像素很细。不要手工把
图像拉成 128 × 128 × 128，也不要为了好画而改变原始空间。Slicer 显示的切片
位置和物理空间应保持原样。

## 2. 什么是 Mask

Mask 是和 MRI 逐体素对应的一张标签图。它不是另一张诊断图，而是告诉训练程序：

    0 = 这个体素不是下颌髁突
    1 = 这个体素属于下颌髁突

本项目只允许这两个值。背景必须是 0，髁突必须是 1。不要给关节盘、关节窝、
颞骨、关节结节或病灶涂成 1。

## 3. 为什么要画 Mask

模型不知道哪个结构是髁突。医生在 MRI 上逐层画出的 mask 是训练时的正确答案。
如果 mask 漏画、误画、错位或包含别的结构，模型会学习到错误答案。因此“看起来
差不多”不够，需要逐层检查和三维检查。

## 4. 在 Slicer 中准备

1. 打开 3D Slicer。
2. 通过 Extension Manager 或 Additional module paths 安装/加载项目中的
   slicer/TMJCondyleAnnotator。
3. 用项目内的命令先把 DICOM 转成 case_001.nii.gz，或者在 Slicer 中打开已有
   的匿名 NIfTI。
4. 不要把患者姓名放入 Slicer scene 名称、文件名、case id 或截图标题。

如果你只有 DICOM，没有转换文件，建议由项目负责人在项目根目录运行：

~~~powershell
python scripts/inspect_dicom.py C:/private/one_case
python scripts/dicom_to_nifti.py C:/private/one_case --case-id case_001
~~~

这两个脚本也可以接收包含子目录的病例目录；它们不会按 1.DCM、2.DCM 这样的
文件名排序，而是使用 DICOM Series reader。如果有多条序列，按 inspect_dicom.py
显示的匿名 series_index 选择。

## 5. 选择 MRI 并创建标注

1. 在 TMJ Condyle Annotator 的 MRI volume 下拉框中选择 MRI。
2. 在匿名 case id 中填写 case_001 形式的名称。
3. 如果要自动更新 manifest，填写私有的 workspace/dataset_manifest.csv 路径。
4. 点击“创建下颌髁突标注”。

此时模块会自动创建一个唯一的 segment：

    Mandibular Condyle

同时把当前 MRI 设置为 reference geometry。这意味着导出的 mask 应与 MRI 有相同
的 shape、spacing、origin 和 direction。

## 6. 打开 Segment Editor

点击“开始标注（打开 Segment Editor）”。Slicer 会切换到原生 Segment Editor。
请只编辑 Mandibular Condyle 这个 segment。

常用工具：

- Paint：在当前切片上涂画。
- Erase：擦除涂错的部分。
- Draw：画封闭轮廓。
- Scissors：裁掉一块明显错误的区域。
- Islands：查看、删除或保留孤立区域。
- Fill between slices：当相邻切片轮廓可靠时填充中间切片。

工具的细节以你安装的 3D Slicer 版本为准。本项目不替换这些工具。

## 7. 如何找到下颌髁突

下颌髁突是下颌骨上端、靠近耳前方颞下颌关节的骨性结构。请以你所在医院的
标注规范和解剖知识为准。建议：

1. 先在 Sagittal 视图定位关节前后方向。
2. 再在 Coronal 视图确认内外侧范围。
3. 用 Axial 视图检查横断面边界。
4. 在所有包含髁突的切片上画边界，不要只画一张“最清楚”的切片。
5. 不确定的区域先放大、调整窗宽窗位并在另外两个方向复核。
6. 下一个切片出现结构时开始标，结构消失后再多检查相邻切片，避免首尾漏标。

不要根据本项目自动生成任何“建议区域”。Codex 和脚本不会替你判断髁突位置。

## 8. 如何擦除错误

发现涂到了关节盘、关节窝、颞骨或背景时：

1. 点击 Erase。
2. 调整半径。
3. 沿错误区域擦除。
4. 切换到另外两个视图，确认没有留下小岛。

不要用“为了让模型好看”而删除真实髁突部分。Mask 只应反映医生对结构的真实
判断，QC 只检查格式和明显的数学错误，不替你修改解剖内容。

## 9. 三视图检查

在每隔几层时切换 Axial、Sagittal、Coronal：

- 边界是否在三个方向连续？
- 是否出现突然跳动的厚片？
- 是否误包含关节盘或颞骨？
- 是否漏掉髁突的顶部或底部？
- 左右侧定义是否和项目 manifest 一致？

如果 volume 本身同时包含左右两个 TMJ，必须在整个数据集内采用同一 convention：
要么先明确切成两个 ROI/case，要么所有此类 volume 都把左右髁突作为同一个 class 1。
不要对个别病例偷偷改变定义。项目优先使用单侧 TMJ = 单 case。

## 10. 打开 3D 显示

点击“Show 3D”。Slicer 会根据当前 segment 生成闭合表面。旋转表面检查：

- 是否只有髁突；
- 表面是否有明显大洞；
- 是否出现远离髁突的小孤立块；
- 首尾切片是否被错误连接；
- 是否有大片背景误标。

三维表面是 QC 辅助，不替代三视图逐层检查。

## 11. 保存 Mask

点击“保存髁突 Mask”。模块在保存前会检查：

- MRI 与 mask 的 shape 相同；
- spacing、origin、direction 相同；
- 标签只能是 0 和 1；
- mask 不能全空；
- foreground voxel 数量；
- physical volume（mm³）；
- connected components 过多时给 warning。

模块不会自动把错误值改成 0 或 1，也不会自动删除小块。出现 FAIL 时回到
Segment Editor 修正后重新检查。出现 warning 时必须由医生回看，warning 本身
不会强行修改 mask。

推荐输出：

    workspace/labels/case_001.nii.gz

并把 annotation_status 留为 ANNOTATED；复核完成后再改为 VERIFIED。保存后还应
在项目根目录运行：

~~~powershell
python scripts/validate_case.py --image workspace/nifti/case_001.nii.gz --label workspace/labels/case_001.nii.gz
~~~

## 12. 如何确认标注完成

一个病例只有同时满足以下条件才算完成：

1. 所有相关切片已检查。
2. 三视图检查通过。
3. 3D surface 检查通过。
4. 导出 QC 为 PASS。
5. foreground voxel > 0。
6. label 只有 0 和 1。
7. MRI 与 mask geometry 匹配。
8. manifest 的 case_id、group_id、side 和路径正确。
9. 第二位医生/负责人复核后可标记 VERIFIED。

不要把“打开了 Slicer”或“创建了空 segment”当作完成标注。
