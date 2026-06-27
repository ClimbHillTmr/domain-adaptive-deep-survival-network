# Publication Figure Manifest

## Main Figures
- `Fig1_Cohort_Flow`: 研究样本流转与分析队列规模，基于原始 CSV 与 Table 1 冻结样本量。
- `Fig2_Baseline_Shift`: 深医与福鼎关键基线变量的中心间偏移。
- `Fig3_Event_Timing_Stages`: 两中心低血压发生时间阶段分布。
- `Fig4_KM_Center_Comparison`: 两中心 IDH-free survival 曲线比较。
- `Fig5_Performance_Comparison`: 外部验证模型性能比较，基于 `table2_performance.csv`。
- `Fig6_Target_Subgroup_Burden`: 福鼎队列亚组 IDH 事件负担。

## Supplementary Figures
- `FigS1_Architecture`: 方法学架构图，限定为统计学对齐而非因果 DAG。
- `FigS2_Phenotype_Transition`: 定性状态转移概念图，仅作机制讨论辅助。

## Important Notes
- 当前仓库没有可复核的逐例预测文件，因此没有导出真实校准曲线或 DCA 图。
- 当前仓库没有可复核的真实 SHAP 结果缓存，因此没有保留任何基于模拟数据的 SHAP 图。
- 若后续补齐逐例预测与解释结果，应新增到 `figures/Main_Figures`，而不是覆盖现有真实数据图。
