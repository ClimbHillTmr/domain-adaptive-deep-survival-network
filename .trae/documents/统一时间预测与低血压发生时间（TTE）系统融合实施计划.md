## 目标与范围
- 用离散阶段分类（未发生/早期/中期/晚期）替代连续TTE回归/纯生存预测，输出：阶段标签、各阶段概率与校准置信区间。
- 融合统一时间预测与TTE（Cox/RSF/DeepSurv）优势：保留个体化风险曲线S(t)与HR/CI用于校准与解释，但最终决策以阶段分类为主，强调早期阶段召回优先与跨院区稳健性。

## 阶段定义与时间划分（临床+数据驱动）
- 初始边界（可调）：早期0–30min；中期30–90min；晚期>90min。
- 数据驱动微调：
  - 依据真实事件时间分布Q1/Q2/Q3与临床窗口（如30–60–120）联动调整。
  - 结合透析时长（2–5h）与采样间隔（1–5min）自适应：等时长分箱或等事件密度分箱（降低偏斜）。
- 标签标准化：
  - 未发生→标签0（删失），记录`is_censored=True`，训练时使用IPCW加权。
  - 已发生→按首次事件时间落入阶段标注1–3。

## 数据探索与预处理
- 数据清单（来自data_process与CSV）：静态特征（年龄/性别/透析龄/糖尿病/心衰/瘘管类型与使用时长/干体重/透前BP）；透中时序（BP/HR/UF_rate/电导/BV/SpO₂）；事件记录与时间节点（SBP<90或ΔSBP<-20等定义）。
- 质量评估：时间戳对齐与补齐/下采样；>40%缺失的序列剔除；异常值分位数截断（1%–99%）；识别“透析中断/重启”，按session分割。
- 特征工程：
  - 时域Rolling 3/5/10/30min：均值/std/min/max/IQR；变化率dSBP/dt、dUF/dt、ΔSBP(10min)；波动率MAD/CV；异常点占比（SBP<90）。
  - 频域：FFT主频/功率峰；PSD LF/HF能量比例（在消融中验证增益）。
  - 静态协变量：透前SBP/DBP、目标UF量、OH ratio；人口学与域变量（中心/机器/滤器）。
- 标签处理与审计：从规则函数获取event_time并生成stage_label；未发生设`is_censored=True`；输出`survival_dataset.csv`包含：特征+event_time+stage_label+is_censored+last_observed_time。

## 模型方案（阶段分类为主，生存辅助校准）
- 多类/序型分类：
  - 树模型（基线）：LightGBM/XGBoost/RandomForest（交叉熵），处理不均衡（class_weight或focal loss）；特征重要性用于解释。
  - 序列深度（主模型）：Transformer/LSTM/TCN，输出N阶段logits；静态融合（concat或FiLM）；支持实时更新推理；早停+概率校准（isotonic/sigmoid）。
  - 序型建模（可选）：CORAL/OrdinalRegression，利用阶段顺序提升一致性。
- 生存→阶段映射（解释增强与校准）：
  - 训练CoxPH/RSF/DeepSurv获取S(t)/h(t)/E[T]。
  - 在阶段边界评估S(t)，阈值跌破即映射阶段；或将E[T]映射至时间箱生成软标签增强分类训练。
  - 用生存模型的HR/CI（静态特征贡献）与RSF重要性/DeepSurv风险趋势提升解释与校准可信度。
- 个体化与分层（MoE可选）：按性别/年龄/糖尿病/瘘管类型/中心域分层训练子模型；Router基于静态或前10分钟序列选择专家，改善特定人群阶段识别。

## 训练与验证流程
- 分割：时间序列阻塞K折/滚动原点；按患者或session分组（grouped split）；保留独立测试集（按时间或院区切分）。
- 早停与正则：
  - 树模型：`early_stopping_rounds`、L1/L2、特征/行采样。
  - 深度模型：验证F-β（β≥2）强调早期召回；`patience/min_delta`；Dropout/LayerNorm/warmup LR。
- 超参优化：optuna/skopt；保留trial日志（参数、指标、PR曲线截面）。
- 校准与阈值：
  - 分阶段概率校准（isotonic/Platt，独立或联合）。
  - 阶段阈值按F-β选择；Bootstrap阶段概率与阈值的置信区间。
- 持久化：`model.pt/.pkl`、`config.json`、`training_log.json`；版本化目录`runs/YYYYMMDD_HHMM/`。

## 评估指标与临床相关性
- 分类指标：macro/micro F1；per-stage recall（重点早期）；混淆矩阵与错位分析（偏早/偏晚）；Brier score（校准度）。
- 解释性指标：
  - 树/ML：SHAP、Permutation、PDP（单调性检查）。
  - 深度：注意力热力图稀疏性与时序重要性稳定性。
  - 生存：HR/CI显著性与校准曲线拟合优度。
- 临床相关性：早期阶段召回≥80%；漏诊率（FN）；预测阶段与临床风险分层一致性；跨中心迁移稳健性。

## 标准化解释输出
- JSON：`metrics.json`（分类+校准）、`explain.json`（SHAP/注意力/HR/CI）。
- CSV：`stage_probabilities.csv`（各阶段概率+置信区间）、`coefficients.csv`（Cox系数与HR）。
- 图表：混淆矩阵、SHAP summary/bar、PDP、注意力热力图、生存森林图（Cox/RSF）、阶段校准曲线（PNG/PDF）。

## 集成与模型选择
- 基线：CoxPH（生存）+ LightGBM（阶段分类）。
- 主推：Transformer（静态融合）+ 生存辅助校准。
- 集成策略：加权平均（分类概率+S(t)导出概率）、堆叠（meta-learner）、MoE动态路由（临床亚群）。

## 性能与可复现
- 统一线程限制（OMP/MKL/BLAS）、Batch推理、向量化、长序列分块、缓存。
- 完整随机种子与环境版本锁定；一键重跑脚本输出完整JSON/CSV/PNG/PDF。

## 里程碑
- P1（Week 1）：阶段边界确认与标签标准化；LightGBM基线训练与解释；分类评估与首版报告。
- P2（Week 2）：Transformer/LSTM阶段分类原型与校准；生存模型训练与阶段映射；MoE初版路由。
- P3（Week 3）：超参搜索与集成策略；解释增强（SHAP/注意力/森林图）；临床相关性评估与跨院区验证。
- P4（Week 4）：一键脚本与PDF仪表盘；完整文档与实验包；对比/消融/迁移最终报告与交付。

## 交付物
- 完整阶段分类模型代码（含一键入口）。
- 解释工具（SHAP/LIME、注意力、HR/CI）。
- 说明文档（预处理/特征/训练/验证/部署与解释输出格式）。
- 完整验证报告（性能、解释性、临床相关性、消融、迁移）。