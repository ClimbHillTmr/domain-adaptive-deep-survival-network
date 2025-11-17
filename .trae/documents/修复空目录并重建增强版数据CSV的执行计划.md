## 诊断目标
- 目录 `/home/cht/Works/PredictionTimeHypotensionDialysis/data_preprocessing/data` 显示为空；确保增强版最终CSV（深医/福鼎）正确写入该目录，并对接 Loader 与 Runner 完成复跑与报告。

## 原因分析
- 目录未创建：构建脚本未显式 `os.makedirs(..., exist_ok=True)`，导致 `to_csv` 保存失败或写到意外位置。
- 路径不一致：相对路径与绝对路径混用；或早期文件保存在其他目录（如 `updated_dataset_fuding.csv` 源未在 data/）。
- 环境或权限：写入权限不足导致静默失败。

## 修复与执行步骤
1. 在 `HBD_data.py` 与 `HBD_data_fuding.py` 所有 `to_csv` 写出前添加：
- `os.makedirs('/home/cht/Works/PredictionTimeHypotensionDialysis/data_preprocessing/data', exist_ok=True)` 确保目录存在。
- 统一使用绝对路径保存：`/home/cht/.../data_preprocessing/data/深医_final_data.csv`、`福鼎_final_data.csv`、以及中间产物 `*_optimized_data*.csv`、`*_result_df*.csv`。

2. 重新构建增强版数据：
- 运行 `data_preprocessing/HBD_data.py`（深医）与 `data_preprocessing/HBD_data_fuding.py`（福鼎），确保新增列（minutes_from_start_list、duration_minutes、et_min、events、stage_30_60_120、stage_30_90_120）写入。

3. 对接 Loader 与 Runner 并复跑两条路径两套边界：
- LGBM/Transformer，边界 `30-90-120` 与 `30-75-120`。
- 输出：`internal/external_metrics.json`、`metrics_thresholded.json`、`external_metrics_thresholded.json`、`thresholds*.json`、`calibration_*`、`confusion_matrix.csv`、`coefficients.csv`、`stage_soft_vs_pred.csv`。

4. 提交完整报告与图表：
- 时间依赖AUC、C-index/IBS、可靠性曲线、分阶段PR/ROC、SHAP/PDP、HR/CI森林图；摘要外部早期召回与brier协同改善、校准拟合度与HR/CI显著性。

## 验收
- 目录存在且包含增强版CSV；复跑产生完整 `runs/` 工件；外部早期召回≥1%，brier不恶化；校准曲线贴近对角线；中/晚期稳定；HR/CI显著性合理。