# PredictionTimeHypotensionDialysis

> Domain-Adaptive Deep Survival Network (DA-DSN) for Intradialytic Hypotension Prediction

## 项目概述

本项目开发了一个 **DA-DSN (Domain-Adaptive Deep Survival Network)** 模型，用于预测血液透析患者的透析中低血压（IDH）事件。模型采用生存分析框架，结合 ResNet1D 时序编码和 MMD 域适应技术，在双中心数据集上实现可泛化的并发症风险预测。

### 核心特性

- **生存分析建模**: 基于 DeepSurv 架构，输出时间依赖的风险评分
- **双分支架构**: 静态患者特征 (MLP) + 动态生理信号 (ResNet1D)
- **域适应**: MMD 损失对齐源域（深医）和目标域（福鼎）的特征分布
- **消融实验**: 系统评估静态特征、动态特征、域适应的贡献
- **评估指标**: C-index (一致性指数) + IBS (综合 Brier 分数) + Bootstrap 置信区间
- **临床可视化**: 个体化生存曲线 + UMAP 域适应可视化

### 数据集

| 数据集 | 来源 | 样本量 | 用途 |
|--------|------|--------|------|
| 深医_final_data.csv | 深医 | ~208,276 | 训练集 |
| 福鼎_final_data.csv | 福鼎 | ~63,382 | 外部验证集 |

**数据质量控制**:
- 观察窗截断：仅使用前 60 分钟动态数据（OBS_WINDOW=60）
- 早期事件过滤：排除观察窗内发生事件的样本
- 动态特征插补：处理 NaN 值防止数值不稳定

## 环境配置

### 系统要求

- Python >= 3.9
- PyTorch >= 2.0
- macOS / Linux (推荐 GPU 加速)

### 依赖安装

```bash
# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装核心依赖
pip install torch numpy pandas scikit-learn omegaconf hydra-core umap-learn matplotlib

# 可选：GPU 加速（根据硬件选择）
# CPU: pip install torch (默认)
# GPU: pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 依赖清单

| 包 | 版本 | 用途 |
|----|------|------|
| torch | >= 2.0 | 深度学习框架 |
| numpy | >= 1.24 | 数值计算 |
| pandas | >= 2.0 | 数据处理 |
| scikit-learn | >= 1.3 | 预处理/评估 |
| omegaconf | >= 2.3 | 配置管理 |
| hydra-core | >= 1.3 | 实验配置 |
| umap-learn | >= 0.5 | 域适应可视化 |
| matplotlib | >= 3.7 | 论文图表生成 |

## 使用说明

### 快速开始

```bash
# 运行完整消融实验
python scripts/run_dadsn_ablation.py

# 生成论文图表（含 1 epoch 调试模式）
python scripts/generate_figures.py

# 指定自定义配置
python scripts/run_dadsn_ablation.py experiment.train_csv=/path/to/train.csv experiment.external_csv=/path/to/val.csv
```

### 配置管理

项目使用 [Hydra](https://hydra.cc/) 进行配置管理，配置文件位于 `configs/` 目录：

- `configs/config.yaml`: 主配置文件
- `configs/defaults.yaml`: 默认参数

#### 关键配置项

```yaml
# 实验配置
experiment:
  train_csv: "data_preprocessing/data/深医_final_data.csv"
  external_csv: "data_preprocessing/data/福鼎_final_data.csv"
  save_dir: "./runs/${now:%Y-%m-%d}/${now:%H-%M-%S}"
  seeds: [42, 123, 456]

# 模型配置
model:
  transformer:
    d_model: 64
    nhead: 4
    num_layers: 2
    dropout: 0.1

# 训练配置
training:
  batch_size: 32
  epochs: 20
  learning_rate: 0.001
  coral:
    lambda_coral: 0.5
```

### 消融实验

项目自动运行三组消融实验：

| 实验 | 配置 | 目的 |
|------|------|------|
| Base | Static Only + Cox | 静态特征基线 |
| Base + Transformer | Static + Dynamic + Cox | 动态特征贡献 |
| DA-DSN | Static + Dynamic + Cox + CORAL | 域适应贡献 |

实验结果保存在 `runs/YYYY-MM-DD/HH-MM-SS/ablation_results.json`。

### 论文图表生成

`scripts/generate_figures.py` 生成以下图表：

1. **Figure 1: 个体化生存曲线**
   - 从测试集挑选 3 个典型患者（极高危、中危、低危）
   - 使用 Breslow Estimator 还原基线生存函数
   - 绘制 60-240 分钟生存概率曲线

2. **Figure 2: UMAP 域适应可视化**
   - 提取倒数第二层特征向量 (B, 64)
   - 对比 Base+Transformer 与 DA-DSN 的域混合效果
   - 证明 CORAL 成功对齐两中心分布

3. **MMD 域分布差异分析**
   - 计算源域/目标域静态特征 MMD
   - 计算源域/目标域动态特征 MMD
   - 量化域适应必要性

### 输出说明

```
runs/
└── 2026-04-16/
    └── 14-30-00/
        ├── ablation_results.json    # 消融实验结果
        ├── base_best.pth            # Base 最佳模型
        ├── base_transformer_best.pth # Base+Transformer 最佳模型
        └── dadsn_best.pth           # DA-DSN 最佳模型

figures/
├── figure1_survival_curves.png      # 个体化生存曲线
└── figure2_umap_coral.png           # UMAP 域适应可视化
```

## 项目结构

```
PredictionTimeHypotensionDialysis/
├── configs/                    # 配置文件
│   ├── config.yaml            # 主配置
│   └── defaults.yaml          # 默认配置
├── data_preprocessing/        # 数据预处理
│   └── data/                  # 原始数据
│       ├── 深医_final_data.csv
│       └── 福鼎_final_data.csv
├── scripts/                   # 执行脚本
│   ├── run_dadsn_ablation.py  # 消融实验入口
│   └── generate_figures.py    # 论文图表生成
├── src/                       # 源代码
│   ├── data/
│   │   └── loader.py          # 数据加载器（含观察窗截断）
│   ├── models/
│   │   └── dadsn_model.py     # DA-DSN 模型
│   ├── training/
│   │   └── dadsn_runner.py    # 训练循环
│   └── metrics/
│       └── survival_metrics.py # 评估指标
├── figures/                   # 生成的论文图表
├── runs/                      # 实验输出（自动生成）
├── 项目报告.md                 # 详细项目报告
└── README.md                  # 本文件
```

## 模块说明

### 数据处理 (`src/data/loader.py`)

`DialysisDataLoader` 类负责：
- 静态特征处理（类别编码、缺失值填补、标准化）
- 动态特征处理（序列解析、填充、缺失值填补）
- **观察窗截断**：仅使用前 OBS_WINDOW=60 分钟数据
- **早期事件过滤**：排除观察窗内发生事件的样本
- 目标变量处理（事件标志、持续时间、无效样本过滤）
- 训练/测试集分离（fit/transform 模式，防止数据泄露）

### 模型 (`src/models/dadsn_model.py`)

`DADSN` 类实现：
- 静态分支：MLP 编码患者基线特征
- 动态分支：Transformer 编码生理信号时序（可选）
- 融合层：Concat + FC + ReLU
- 输出头：Linear → log-hazard ratio
- 损失函数：Cox Partial Likelihood + CORAL
- **数值稳定实现**：log hazard 归一化防止 NaN

### 训练 (`src/training/dadsn_runner.py`)

训练循环管理：
- 单轮训练（前向传播、损失计算、反向传播）
- 模型评估（C-index, IBS）
- 最佳模型保存
- 消融实验调度

### 评估指标 (`src/metrics/survival_metrics.py`)

生存分析专用指标：
- `concordance_index_censored`: C-index 计算
- `integrated_brier_score`: IBS 计算

### 图表生成 (`scripts/generate_figures.py`)

论文级可视化：
- 三组消融实验完整训练与评估
- **Breslow Estimator**：标准风险集方法还原生存函数
- **Bootstrap 置信区间**：50 次重采样计算 95% CI
- **MMD 域差异分析**：量化源/目标域分布差异
- **UMAP 降维可视化**：替代 t-SNE 避免段错误

## 关键修复记录

### 数据泄露修复
- **问题**：动态特征使用了观察窗之后的数据，导致模型学习到未来信息
- **修复**：实现 OBS_WINDOW=60 截断，仅使用前 60 分钟数据
- **影响**：样本量从 ~212,363 减少到 ~208,276（深医），~73,671 减少到 ~63,382（福鼎）

### NaN 损失修复
- **问题**：Cox 损失计算中出现数值不稳定导致 NaN
- **修复**：实现 log hazard 归一化，过滤 event=1 & time=0 样本
- **影响**：训练过程稳定，无 NaN 出现

### Breslow Estimator 修复
- **问题**：原始实现使用 ±0.5 分钟启发式窗口近似
- **修复**：使用标准风险集方法，严格遵循生存分析理论
- **影响**：生存曲线更准确，符合临床解释

### Bootstrap 置信区间
- **问题**：仅报告点估计，缺乏统计不确定性量化
- **修复**：实现 50 次 bootstrap 重采样，输出 95% CI
- **影响**：结果更具统计严谨性

### t-SNE → UMAP 切换
- **问题**：t-SNE 在大规模数据上出现段错误
- **修复**：使用 UMAP 替代，性能更好且稳定
- **影响**：域适应可视化正常生成

## 贡献规范

### 代码风格

- **命名**: snake_case (函数/变量), PascalCase (类)
- **类型提示**: 所有函数必须包含参数和返回值类型
- **注释**: 关键逻辑必须注释，函数必须有 docstring
- **格式化**: 使用 black 或 ruff 格式化代码

### 提交规范

使用 [Conventional Commits](https://www.conventionalcommits.org/) 格式：

```
feat: 添加 CORAL 域适应模块
fix: 修复 Cox 损失数值稳定性问题
docs: 更新项目报告
refactor: 重构数据加载器
test: 添加生存指标单元测试
```

### 分支策略

- `main`: 稳定版本，仅接受通过审查的代码
- `feature/*`: 功能开发分支
- `fix/*`: 问题修复分支

### Pull Request 流程

1. 从 `main` 创建功能分支
2. 开发并本地测试
3. 提交 PR，描述变更和测试结果
4. 代码审查通过后合并

## 维护计划

### 版本管理

| 版本 | 状态 | 说明 |
|------|------|------|
| v0.1 | ✅ 已发布 | 基础架构 + 消融实验 |
| v0.2 | 🔄 开发中 | 超参数调优 + 性能优化 |
| v1.0 | ⏳ 规划中 | 论文投稿版本 |

### 已知问题

- 动态时序数据当前仅支持 summary 模式（seq_len=1）
- 大规模训练建议启用混合精度训练（AMP）
- CORAL 权重可能需要根据训练轮数调整（当前 lambda=0.5）

### 未来方向

1. **多任务学习**: 同时预测低血压、心衰等多个并发症
2. **强化学习**: 基于离线 RL 优化透析方案参数
3. **在线学习**: 支持增量更新，适应数据分布变化
4. **部署优化**: 模型量化/剪枝，满足实时推理需求

## 引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@misc{dadsn_idh_prediction,
  title={Domain-Adaptive Deep Survival Network for Intradialytic Hypotension Prediction},
  author={Your Name},
  year={2026},
  howpublished={\url{https://github.com/your-repo/PredictionTimeHypotensionDialysis}}
}
```

## 许可证

本项目仅供学术研究使用。
