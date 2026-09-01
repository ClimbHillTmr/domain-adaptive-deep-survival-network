## 目标
- 完成三项升级并一次跑通：
  - 接入Transformer阶段分类（多类logits+静态融合），统一评估与概率校准。
  - 接入Cox/RSF生存辅助校准与解释（HR/CI与S(t)），输出阶段软标签与校准曲线。
  - 增强数据管线（时间戳对齐、会话分割、缺失过滤阈值），并生成PDF报告与配置化运行。

## 模块与文件规划
- 模型：
  - `src/models/transformer_stage.py`：多变量序列→TransformerEncoder→阶段logits；静态特征融合（concat/FiLM）。
  - `src/models/wrappers.py`：将PyTorch模型包装为`predict/predict_proba`接口，便于统一评估与校准。
- 生存辅助：
  - `src/survival/cox_adapter.py`：基于lifelines训练CoxPH，输出HR/CI与个体S(t)。
  - `src/survival/rsf_adapter.py`：基于scikit-survival训练RSF，输出重要性与S(t)。
  - `src/survival/stage_mapper.py`：将S(t)/E[T]映射至阶段概率与软标签（边界b1/b2可配置）。
- 校准与评估：
  - `src/calibration/stage_calibration.py`：isotonic/Platt的分阶段校准，按F-β（β≥2）选择阈值与Bootstrap置信区间。
  - 复用`src/evaluation/classification.py`并扩展校准曲线生成（PNG/CSV）。
- 数据管线：
  - `src/data/sessionize.py`：时间戳对齐（分钟偏移）、中断/重启会话分割、缺失过滤阈值（如>40%剔除）。
  - 扩展`src/data/loader.py`以使用列映射与阈值配置，返回`X_seq[N,L,F]`与静态`X_static`。
- 配置与报告：
  - `configs/defaults.yaml`：列映射、边界、阈值、模型参数与校准策略。
  - `src/report/report_md.py`：汇总指标/图表为Markdown；调用Pandoc生成PDF（可用则PDF；否则保留MD/HTML）。
- Runner与脚本：
  - 扩展`src/training/runner.py`：增加Transformer路径与生存辅助校准；输出完整工件；
  - 更新`scripts/run_phase_classification.py`：支持配置文件路径，选择模型（lgbm/transformer），开关生存辅助。

## 接入Transformer阶段分类
- 数据：从`loader+sessionize`得到`X_seq[N,L,F]`与`X_static`。
- 模型：TransformerEncoder（2–3层、4–8头、FFN=2×hidden），位置编码+掩码；静态融合（concat或FiLM）；logits输出（4类）。
- 训练：F-β（β≥2）早停；Dropout/LayerNorm/warmup LR；class_weight平衡；保存权重与曲线。
- 校准：分阶段isotonic/Platt；阈值以F-β优化早期召回；输出概率与置信区间。
- 评估：复用统一评估与`src/evaluation/classification.py`；图表保存到`runs/`。

## 接入Cox/RSF生存辅助校准与解释
- 训练CoxPH与RSF：输入静态与时变摘要（可选分段均值）；输出HR/CI、重要性与S(t)。
- 映射与软标签：在b1/b2边界上评估S(t)或用E[T]映射至阶段概率，作为分类训练的软标签或加权校准参考。
- 解释输出：HR/CI森林图（PNG/CSV）、RSF重要性与风险趋势曲线；校准曲线与一致性评估。

## 数据管线增强与配置化运行
- 时间戳对齐：将`透中数据记录时间节点`转换为“距起始分钟”；
- 会话分割：识别中断/重启，按session处理；
- 缺失过滤：阈值（如>40%缺失剔除），记录剔除日志；
- 配置化：`configs/defaults.yaml`统一管理列映射、边界、阈值与模型/校准参数；脚本加载配置运行。

## 报告生成
- 汇总：指标（F1/召回/Brier/错位）、校准曲线、混淆矩阵、HR/CI森林图、RSF重要性、注意力热力图（Transformer）。
- 输出：Markdown→Pandoc转换为PDF（若PDF不可用则保留MD/HTML）；保存到`runs/YYYYMMDD_HHMM/report.pdf`。

## 验证与交付
- 运行：用你的CSV执行Transformer与LightGBM两条路径（可选开启生存辅助）；生成完整`runs/`工件。
- 验收：早期召回≥80%，Brier降低，校准曲线拟合优度良好；HR/CI显著；同配置重复运行一致。

## 后续扩展
- 堆叠/加权集成与MoE路由；
- 场景化边界（如30–60–120）；
- 端到端PDF仪表盘增强与数据版本化联动。