# Synthetic TMJ MRI test-only run

本流程只用于验证本项目的真实 nnU-Net v2 接口链路，不属于临床数据、论文结果或正式实验。
脚本会把 synthetic MRI、前景 mask、nnU-Net raw/preprocessed、checkpoint、OOF prediction、
指标和图像全部放到 `workspace/test_only_tmj_synthetic/`，不会使用正式的 `workspace/nnUNet_*`、
`workspace/reports/` 或正式 manifest。

## 运行

在项目根目录执行：

```powershell
\.venv\Scripts\python.exe scripts\run_test_only_synthetic.py
```

当前 CPU 环境下使用 `nnUNetTrainer_TMJTestOnly_1epoch`：真实 nnU-Net v2 `3d_fullres`、5 个
fold、每个 fold 1 epoch、每 epoch 2 个训练 iteration 和 1 个 validation iteration。这个 trainer
只在 `TMJ_TEST_ONLY=1` 时允许构造，不能作为正式 trainer 使用。

可控参数示例：

```powershell
\.venv\Scripts\python.exe scripts\run_test_only_synthetic.py `
  --cases 10 `
  --shape 32,48,48 `
  --spacing 0.8,0.8,1.0 `
  --shape-mode condyle `
  --seed 20260904
```

`--shape-mode ellipsoid` 可切换为单椭球前景；默认 `condyle` 是椭球髁突帽和偏移、缩放后的
颈部椭球的并集。每个病例的中心、轴长、旋转、颈部参数和随机种子保存在
`metadata/case_parameters.json`，因此可以复现和控制。

## 输出

- `nnUNet_results/Dataset999_TMJTestOnly/nnUNetTrainer_TMJTestOnly_1epoch__nnUNetPlans__3d_fullres/`
  - 五折真实 checkpoint 和每折 validation summary。
- `predictions/oof/fold_0..4/`
  - 每个病例只由其 validation fold 产生的 OOF 预测。
- `reports/metrics_per_case.csv`、`reports/metrics_summary.csv`、`reports/cv_report.md`
  - Dice、IoU、使用 image spacing 的 HD95（mm）。
- `reports/figures/`
  - MRI 三视图、GT overlay、prediction overlay、GT vs Prediction 和 marching-cubes 3D surface。
- `predictions/new_case_001_condyle.nii.gz`
  - 真实五折 ensemble 的新病例推理结果；对应 QC 和可视化在 `reports/new_case_inference/`。
- `TEST_ONLY_RUN_SUMMARY.json`
  - 记录 trainer、fold、路径隔离和指标摘要，并明确禁止用于正式结果。

任何 `test_only_*` 目录中的文件都不能复制回正式项目目录，也不能作为论文或正式实验指标。
