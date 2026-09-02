# 三个参考仓库的只读审查记录

本项目建立前先只读 clone 并检查了三个指定仓库。clone 位于项目的 references/
临时目录，已加入 .gitignore，不会作为第三方源代码提交。

## 1. MenxLi/UNets-TMJ

审查到的 commit：

    4a8cb1871cc05c005cda088408ab4b2a81591777
    2023-10-19

借鉴内容：

- TMJ MRI 的任务背景和单个 TMJ volume 的数据组织思路；
- data-demo 使用的原始标注/图像组织观察；
- dataset preparation、训练、5-fold、inference、evaluation 的实验分段；
- 需要把预测结果和 GT 对齐后再做指标的流程意识。

没有复制：

- labelSys JSON/base64 数据格式；
- nnUNet v1 runtime；
- Task501_TMJSeg；
- 自定义旧 trainer；
- 旧 PyTorch/依赖环境；
- 旧仓库的源码。

许可证审查：仓库顶层 tree 没有 LICENSE、COPYING 或 NOTICE 文件；仓库内部
捆绑的 nnUNet 和 segmentation_models.pytorch 各有自己的许可证。本项目没有
复制其未明确授权的代码，只重新实现本项目所需逻辑，并保留来源说明。

## 2. MIC-DKFZ/nnUNet

审查到的 commit：

    0e495086eb108ff79afe106291e8c15bd2f2bc3a
    2026-07-13

源码 metadata 显示 nnunetv2 版本 2.8.1，要求 Python >=3.10。官方文档确认：

- raw dataset 使用 DatasetXXX_Name；
- imagesTr、labelsTr、可选 imagesTs 和 dataset.json；
- v2 命令使用 nnUNetv2_plan_and_preprocess、nnUNetv2_train、
  nnUNetv2_predict；
- plan/preprocess 会生成 dataset_fingerprint.json 和 nnUNetPlans.json；
- splits_final.json 是 train/val 字典列表；
- 默认 5-fold 训练和五折 ensemble 是官方流程；
- 3d_fullres 是本项目固定配置。

许可证：Apache License 2.0。项目只把 nnU-Net 作为外部安装依赖，不复制其源代码。

## 3. KitwareMedical/SlicerNNUnet

审查到的 commit：

    0cb736d1e0735b8a6b0e251d40c60832fdd046f
    2026-07-29

确认的集成约束：

- 在 Slicer Extension Manager 安装；
- Slicer 中需要可用的 nnUNet v2/PyTorch；
- model path 要能找到 dataset.json；
- 保留 DatasetXXX、trainer__plans__configuration、fold_i、
  checkpoint_final.pth 目录结构；
- Apply 后 segmentation 可进入 Segment Editor 并进行 3D 显示。

许可证：BSD 3-Clause。项目不复制 SlicerNNUnet 源代码，只导出其可识别的官方
nnU-Net 模型目录。

## 本项目的边界

TMJ-Condyle-3D 只负责病例匿名化后的转换、QC、数据集、训练编排、评价、预测编排、
模型导出和 Slicer 标注辅助模块。三方仓库的归属和许可证不被本项目替换，也不把
它们声明为原创。
