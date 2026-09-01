## 目标
- 用生存分析/时间到事件（TTE）方法替代二分类，预测透析过程低血压的发生时间与风险曲线。
- 交付可解释、可复现、学术发表级的系统（代码+评估+报告）。

## 数据与标签设计
- 事件与时间点来源：结合现有规则函数构造标签。
  - 低血压判定与阈值：`data_preprocessing/data_process.py:96`（calculate_pressure_change）
  - 首次发生时间点：`data_preprocessing/data_process.py:133`（calculate_time_points）
- 生存数据构造：
  - `duration`: 从透析起始到首次低血压发生的分钟/时间步；若未发生则为最后观测时刻。
  - `event`: 发生=1，未发生=0（删失）。
  - `static_covariates`: 人口学/透前特征（如性别、年龄、瘘管类型、透前压等）。
  - `time-dependent covariates`（可选）：透中序列特征经窗口化（保证Cox时间依赖或以摘要特征进入标准Cox/RSF）。

## 特征工程与预处理
- 清洗：缺失填补（simple/KNN/iterative）、异常值分位数截断、重复去重；时间戳对齐。
- 标准化：`Standard/MinMax/Robust` 可选；保存 `scaler.pkl`（含统计元数据）。
- 时序到生存特征：窗口统计（均值/方差/斜率/极值/比值）、波动率与异常比率；可构建时变特征（分段均值）。
- 标签构建与审计：输出 `survival_dataset.csv`（duration/event/covariates）与数据字典。

## 模型实现
- 统计生存模型：
  - Cox比例风险（`lifelines.CoxPHFitter`）：支持协变量的系数、HR与置信区间、个体生存函数与CI。
  - 随机生存森林（`scikit-survival`）：支持非线性与复杂交互，提供特征重要性与部分依赖。
- 深度生存（可选增强）：
  - DeepSurv（`pycox`）：学习风险分数，提供个体化生存曲线与可解释性扩展（SHAP/注意力结合）。
- 统一接口：实现 `BaseForecaster` 风格的 `CoxPHSurvival`, `RSFSurvival`, `DeepSurvModel`（fit/predict_survival/predict_risk/save/load/get_params）。

## 训练与验证流程
- 分割策略：阻塞K折或滚动原点时间序列CV；留出独立测试集。
- 早停与正则：
  - Cox：L2/L1正则与变量选择；时间依赖时段分割控制。
  - RSF/DeepSurv：早停（验证C-index）、`patience/min_delta`。
- 超参搜索：
  - 网格/随机：基础参数（RSF树数/深度，Cox惩罚强度）。
  - 贝叶斯：`optuna/skopt` 提升效率；记录搜索轨迹与最佳参数。
- 持久化：模型权重（joblib/torch）、最佳参数JSON、训练摘要与日志；目录含时间戳。

## 解释性输出（标准化）
- Cox：
  - 系数与HR（hazard ratio）表、95%CI与p值；森林图与火山图。
  - 个体生存函数S(t)与CI；基准累积危险度可视化。
- RSF：
  - 全局/局部特征重要性（基尼/最小深度）、部分依赖图（PDP）、累积危险估计曲线。
  - SHAP（生存任务近似方案）与Permutation重要性。
- DeepSurv（可选）：风险分数分布、注意力/贡献热力图（若使用注意力模块）。
- 标准化输出：
  - JSON：`metrics.json`（C-index、IBS、校准）、`explain.json`（HR/CI、重要性）。
  - CSV：`survival_curves.csv`（个体/分组生存函数点列）、`coefficients.csv`。
  - PNG/PDF：生存曲线、森林图、PDP、校准曲线。

## 评估与临床相关性
- 指标：C-index、Integrated Brier Score（IBS）、时间依赖AUC、校准评估（观察-期望曲线）。
- 临床相关性：
  - 早期预警窗口的命中率/漏诊率；风险分层（高/中/低）稳定性。
  - 跨院区（深医/福鼎）迁移验证与域鲁棒性分析。
- 统计检验：
  - DM检验或对生存任务的对比（适配误差定义）；对比模型的显著性报告。

## 系统结构与集成
- 目录：
  - `TimeSeries/survival/`：`coxph.py`, `rsf.py`, `deepsurv.py`
  - `TimeSeries/preprocessing/`：清洗、标准化、时变特征、标签构造。
  - `TimeSeries/evaluation/`：生存指标、校准、曲线绘制与PDF导出。
  - `TimeSeries/training/`：训练Runner、CV与超参搜索、持久化模块。
- 与现有模块联动：沿用 `Methods/unified_model_evaluation.py` 的结果保存规范；扩展生存评估接口。

## 学术交付
- 代码：完整模型实现与统一接口；一键运行脚本（加载数据→构造生存数据→训练→评估→报告）。
- 解释工具：系数/HR表、重要性/SHAP、PDP、个体生存曲线与CI。
- 文档：数据预处理流程说明、特征工程方法、训练/验证步骤、解释输出格式、部署指南与API。
- 验证报告：对比实验（Cox/RSF/DeepSurv）、消融（特征/时间依赖/正则/搜索）、跨院区验证与临床相关性结果。

## 依赖与合规
- 依赖：`lifelines`（Cox）、`scikit-survival`（RSF，需`numpy>=1.22`与`pandas>=1.3`）、`pycox`（DeepSurv，可选）。
- 合规：隐私与可追溯；输出不含识别信息；召回优先用于临床安全。

## 里程碑
- M1：数据→生存数据集构建与审计，CoxPH训练与解释输出，C-index与报告。
- M2：RSF训练与解释输出，超参搜索与集成；校准与IBS。
- M3：DeepSurv原型与注意力可视化（可选），跨院区迁移与临床相关性报告。
- M4：一键脚本与PDF仪表盘、完整文档与可复现实验包，申请审稿级交付。