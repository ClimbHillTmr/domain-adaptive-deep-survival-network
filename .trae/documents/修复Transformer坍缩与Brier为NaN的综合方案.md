## 问题定位
- 外部（Transformer）预测几乎全部为未发生（stage_0），`brier=NaN`，说明：
  - 概率未归一/含NaN或长度未对齐
  - 未接入分阶段校准+分布约束阈值联动（Transformer路径）
  - 不均衡与域偏移导致阈值化坍缩

## 修复方案
1. 概率输出与对齐（TorchStageWrapper）
- 在 `predict_proba`：
  - 对 `logits` 做 `softmax` 后按行归一，`np.nan_to_num`→clip到[1e-9, 1]，再归一
  - 统一返回 `shape=(n_samples, n_classes)`；若出现 `shape=(n_classes, n_samples)`，则转置
  - 保证与输入样本数对齐；长度不一致时截断到 `min_len`

2. Runner联动（Transformer路径）
- 外部评估：
  - 对 `proba_ext` 进行分阶段等距校准（isotonic）→两路阈值优化：按类 F-β（温和 betas）与分布约束（目标为训练分布平滑占比，早期最小占比≥1%）
  - 应用温和阈值：未发生最低阈值≥0.22、早期阈值 0.05–0.10、低置信门槛≈0.25、优先级仅在多类冲突时触发
  - 写出 `thresholds_external.json` 与 `external_metrics_thresholded.json`，绘制 `calibration_external_class*.png`
- 单次评估：同样写出 `metrics_thresholded.json` 与 `thresholds.json`

3. 先验修正（概率迁移）（Transformer与LGBM外部）
- 使用训练集类占比构成目标占比（平滑且早期占比≥1%），对外部 `proba_ext` 做迁移修正并归一化，再进行校准与阈值优化

4. 训练与结构增强（Transformer）
- 增加 `epochs`（如 12–20）、`hidden`（如 64）、`heads`（如 4）、`layers`（如 2），保留CPU配置
- 引入类别权重或加权交叉熵（早期轻增），避免过拟合未发生类

5. 指标计算健壮性
- `brier_score`：
  - 检查 `proba` 是否含NaN并归一化；若发现NaN则 `nan_to_num`
  - 确认 `proba` 与 `y` 长度一致，否则截断到 `min_len`
- `per_stage_recall`：维持简单正确率，避免二分类设置导致异常

6. 特征与输入（补充Transformer静态）
- 将 Loader 的事件邻域特征（pre/post统计与差分）接入到 Transformer 的 `X_static`，增强早期敏感性

## 验证与交付
- 复跑两条路径（LGBM/Transformer）与两套边界（主 `30-90-120`、对照 `30-75-120`）
- 输出：`internal/external_metrics.json`、`metrics_thresholded.json`、`external_metrics_thresholded.json`、`thresholds*.json`、`calibration_*`、`confusion_matrix.csv`、`coefficients.csv`、`stage_soft_vs_pred.csv`
- 验收：外部早期召回≥1%，brier不恶化；校准曲线贴近对角线；中/晚期稳定；HR/CI显著性合理、软标签一致性可解释

## 执行顺序（一次性）
1) 修复 `TorchStageWrapper.predict_proba` 归一与对齐；2) Runner接入校准+阈值与先验修正（Transformer路径）；3) 将邻域特征接入 Transformer 静态；4) 复跑并生成报告；5) 汇总论文所需指标与图表。