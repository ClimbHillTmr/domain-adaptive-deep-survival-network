## 目标
- 重构数据构建脚本并追加高价值特征，生成增强版最终CSV（深医/福鼎）；对接Loader与Runner，复跑两条路径（LightGBM/Transformer）与两套边界（30-90-120、30-75-120）；输出论文所需的完整指标与图表。

## 数据构建重构（data_preprocessing）
- 安全解析与规范化：
  - 替换不安全解析为 `ast.literal_eval`/正则-JSON 安全管线；统一列表列为 List[float]；剔除 "NA"、空字符串、None；统一数值 dtype。
  - 统一时间列（`透中数据记录时间节点`）为 datetime；生成 `minutes_from_start`（相对起始分钟）。
- 一致性与质量控制：
  - 校验列表列与时间节点长度一致；不一致行写入 `removed_inconsistent_rows.csv` 并剔除；输出删除比例与分布变化日志。
  - 极值裁剪：对列表列按全局分位数裁剪（可配置的 q5/q95）。
- 事件检测与标签：
  - 基于 SBP（透析中收缩压）低血压条件 `(fp - p >= 30) 或 p <= 90`；计算事件分钟 `et_min`（在 `minutes_from_start` 上首次满足）；`events=1/0`；保存 `透析开始/结束` 与 `duration_minutes`。
  - 生成两套阶段标签列：`stage_30_60_120` 与 `stage_30_90_120`；保留现有 `降幅/涨幅` 比值/差值及区间列。
- 序列与邻域特征（SBP/MAP/HR/UF）：
  - 序列统计：mean/std/min/max/slope；滚动STD均值（w=3/5/7）、短窗均值斜率（w=3/5/7）、最小值位置比例；UF一阶/二阶差分绝对均值。
  - 事件邻域：在 `et_min` 附近的 pre/post 窗口（w=3/5/7/9）均值/STD/斜率、pre-post 差分、极值至事件相对位置；若 `et_min=None` 则邻域特征置0并标注 no_event。
- 透前/治疗与历史特征：
  - 透前静态：透前SBP/DBP/MAP、体重与干体重、透析龄_天数、实际时长、液温/电导率/钙浓度、呼吸频率、体温等（存在即接入）。
  - 历史均值与比率：按患者ID+日期 `groupby + expanding` 构造 `历史平均<列>` 与 `history_*`（HBP、LBP_times_k 与其 rate）；首次记录设为0。
- 导出：`深医_enhanced_final.csv`、`福鼎_enhanced_final.csv`（附 `schema.json` 与 QA日志）。

## Loader对接
- 读取增强版CSV；使用 `et_min/透析开始/结束/duration_minutes` 构造生存标签（`durations/events`）与阶段标签（选择边界列）；
- 启用所有新增特征（序列、邻域、透前、历史）并打包至 `X` 或 `X_seq/X_static`（Transformer 静态包含邻域特征）。

## Runner与校准阈值联动
- 两路径（LightGBM/Transformer）：
  - 分阶段等距校准（isotonic）后执行两路阈值优化：按类 F-β（温和 betas）与分布约束（目标为训练分布平滑占比，早期最小占比≥1%）；
  - 温和阈值应用：未发生类最低阈值≥0.22、早期阈值 0.05–0.10、低置信门槛≈0.25、优先级仅在多类冲突时触发；写出 `thresholds*.json`、`metrics_thresholded.json`、`external_metrics_thresholded.json`；绘制 `calibration_*`。
  - 先验修正：在外部评估对 `proba_ext` 用训练占比平滑迁移修正并归一化，再进行校准与阈值优化。
- 生存一致性：
  - 用静态特征拟合 CoxPH，输出 `coefficients.csv`（HR/CI/p值）；将 S(t) 映射阶段软标签并生成 `stage_soft_vs_pred.csv` 比对分类概率一致性。

## 复跑与输出
- 两套边界：主 `30-90-120`、对照 `30-75-120`；两条路径：LightGBM 与 Transformer。
- 输出：`internal/external_metrics.json`、`metrics_thresholded.json`、`external_metrics_thresholded.json`、`thresholds*.json`、`calibration_*`、`confusion_matrix.csv`、`coefficients.csv`、`stage_soft_vs_pred.csv`。
- 指标扩展（论文）：时间依赖AUC、C-index/IBS、可靠性曲线、分阶段PR/ROC；解释性图（SHAP/PDP、HR/CI森林图）。

## 验收标准
- 外部早期召回≥1%，brier 不恶化；校准曲线更贴近对角线；中/晚期召回稳定；HR/CI显著性合理、软标签一致性具解释性。

## 执行顺序（一次性）
1) 重构并生成增强版CSV → 2) Loader启用全部新增特征 → 3) Runner接入校准+阈值与先验修正、Cox输出 → 4) 复跑两路径两边界 → 5) 汇总指标与图表（论文所需）。