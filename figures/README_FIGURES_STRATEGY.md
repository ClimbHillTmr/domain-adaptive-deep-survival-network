# Submission Figure Strategy

## Main Figures
- `Fig1_Cohort_Split_Audit`: 研究样本量与福鼎 patient-level split 审计合并图。
- `Fig2_Cross_Center_Shift`: 基线偏移与事件时间结构偏移合并图。
- `Fig3_Performance_Comparison`: held-out test 上的主性能比较图，仅引用当前 locked run。
- `Fig4_Calibration_DCA`: 60/120 分钟校准与 DCA 合并图，仅引用当前 held-out predictions。

## Supplementary Figures
- `FigS1_Target_Subgroup_Burden`: held-out test 亚组事件负担图；只作补充，不宣称亚组优势。

## Net-room Rules
- 禁止恢复 SHAP、mock、pseudo fairness 图，除非完全基于当前 held-out predictions 重建。
- 不再沿用旧的 8 主图结构；投稿版以 4 个主图 + 1 个补图为上限。
- 所有 PDF 导出均强制 `pdf.fonttype = 3` 以满足印刷兼容性要求。
