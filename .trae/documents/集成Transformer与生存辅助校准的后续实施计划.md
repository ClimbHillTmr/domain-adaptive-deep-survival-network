## 目标
- 完成三项升级并一次跑通：
  - 集成Transformer阶段分类（多类logits+静态融合），统一评估与概率校准。
  - 集成Cox/RSF生存辅助校准与解释（HR/CI与S(t)），输出阶段软标签与校准曲线。
  - 增强数据管线（时间戳对齐、会话分割、缺失过滤阈值），并生成PDF报告与配置化运行。

## 文件与模块改造
- 数据管线
  - 新增 `src/data/sessionize.py`：
    - 时间戳标准化为“距起始分钟”
    - 中断/重启会话分割
    - 缺失过滤（默认>40%剔除，可配置）
  - 扩展 `src/data/loader.py`：支持列名映射与阈值配置，返回 `X_seq[N,L,F]` 与 `X_static`。
- 模型与包装
  - 已有 `src/models/transformer_stage.py` 与 `src/models/wrappers.py`：增加校准与阈值接口适配（与LightGBM一致）。
  - 新增 `src/calibration/stage_calibration.py`：
    - isotonic/Platt 分阶段校准
    - 按 F-β（β≥2）进行阶段阈值选择
    - Bootstrap 置信区间输出（CSV）
- 生存辅助
  - 已有 `src/survival/cox_adapter.py` 与 `src/survival/stage_mapper.py`：新增RSF适配器 `src/survival/rsf_adapter.py` 与统一接口，输出阶段概率软标签与HR/CI。
- Runner与脚本
  - 扩展 `src/training/runner.py`：
    - 参数化模型选择（lightgbm/transformer）与生存辅助开关
    - 合并分类评估与校准输出（JSON/CSV/PNG）
  - 更新 `scripts/run_phase_classification.py`：
    - 支持 `--model` 与 `--config`，加载 `configs/defaults.yaml`（列映射、边界、校准策略）
- 报告生成
  - 新增 `src/report/report_md.py`：汇总指标与图表为Markdown
  - 生成 `runs/YYYYMMDD_HHMM/report.md`，若环境可用则转为PDF

## 生成物
- `runs/YYYYMMDD_HHMM/` 目录完整工件：
  - `metrics.json`（宏/微F1、各阶段召回、Brier、错位分析）
  - `stage_probabilities.csv` 与 `*_ci_lower.csv/*_ci_upper.csv`
  - 校准曲线与混淆矩阵（PNG/CSV）
  - 生存HR/CI森林图与阶段映射（PNG/CSV）
  - `report.md`（可转PDF）

## 校验与优化
- 目标：早期召回≥80%，Brier降低，校准曲线拟合优度良好；HR/CI显著。
- 若未达目标：启用isotonic校准、分阶段阈值精调、边界调整（如30–60–120）。

## 执行顺序
1. 完成 `sessionize.py` 与 `loader.py` 扩展，跑通LightGBM路径
2. 集成Transformer至Runner与脚本，统一校准输出
3. 接入Cox/RSF并生成阶段软标签与HR/CI图表
4. 生成Markdown报告并在可用时输出PDF

## 复现与安全
- 保持统一线程限制与随机种子；输出统一到 `runs/` 并保留配置快照，确保可复现。
- 不修改临床规则定义，所有阶段边界由配置控制，便于回溯与审计。