## 目标
- 完成并接入三个模块：
  - 分阶段校准与阈值（isotonic/Platt、F-β独立阈值、bootstrap置信区间与校准曲线）
  - 数据管线增强（时间戳标准化、会话分割、缺失过滤阈值、列映射配置）
  - 脚本集成Transformer与生存辅助（模型选择、辅助开关与配置加载）

## 模块与文件
- 校准
  - 新增 `src/calibration/stage_calibration.py`
    - `calibrate_per_stage(proba, y, method)`：按阶段独立校准（isotonic/Platt）
    - `optimize_thresholds_fbeta(proba, y, beta)`：每阶段独立阈值（F-β）
    - `bootstrap_ci_per_stage(proba, n_boot, alpha)`：概率置信区间
    - `plot_calibration_curves(proba, y, out_dir)`：校准曲线PNG/CSV
- 数据管线
  - 新增 `src/data/sessionize.py`
    - `normalize_times_to_minutes(times)`：标准化时间戳为起始分钟
    - `split_sessions(times, gap_threshold)`：按时间间隔识别中断/重启并分会话
    - `filter_by_missing(seq_dict, missing_threshold)`：按阈值过滤缺失比例
  - 扩展 `src/data/loader.py`
    - 读取 `configs/defaults.yaml` 的列映射、阶段边界与阈值
    - 返回 `X_seq[N,L,F]` 与 `X_static` 并应用会话化与缺失过滤
- 生存辅助
  - 新增 `src/survival/rsf_adapter.py`：RSF统一接口（训练、S(t)、重要性）
  - 扩展 `src/survival/stage_mapper.py`：合并Cox/RSF输出为阶段软标签
- Runner与脚本
  - 扩展 `src/training/runner.py`
    - 支持 `model={lgbm|transformer}`、`survival={on|off}`、`config=...`
    - 合并评估与校准输出（JSON/CSV/PNG），统一写入 `runs/`
  - 更新 `scripts/run_phase_classification.py`
    - 参数：`--model`、`--survival`、`--config`
    - 从配置加载列映射与边界、阈值、校准策略
- 报告
  - 新增 `src/report/report_md.py`：汇总指标与图表生成 `report.md`；若环境支持，转换为PDF

## 执行步骤
1. 实现校准模块并接入Runner（LightGBM/Transformer路径）
2. 完成会话化与缺失过滤及列映射配置；扩展Loader返回 `X_seq/X_static`
3. 接入RSF并合并Cox/RSF阶段软标签；在Runner加 `--survival on` 开关
4. 更新脚本与配置加载；生成完整工件与 `report.md/pdf`

## 验收
- 早期阶段召回≥80%、Brier优于未校准基线、校准曲线拟合优度良好、HR/CI显著；
- 工件完整：`metrics.json`、阶段概率与CI、混淆矩阵、校准曲线、HR/CI森林图、`report.md/pdf`；支持一键复现。

## 运行示例
- LightGBM：`python scripts/run_phase_classification.py --csv <CSV> --b1 30 --b2 90 --save_dir ./runs --model lgbm --config configs/defaults.yaml`
- Transformer+生存辅助：`python scripts/run_phase_classification.py --csv <CSV> --b1 30 --b2 90 --save_dir ./runs --model transformer --survival on --config configs/defaults.yaml`