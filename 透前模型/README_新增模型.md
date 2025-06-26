# 透前模型新增模型说明

## 概述

在原有的透前数据透中低血压建模系统中，新增了两个先进的机器学习模型：

1. **图神经网络模型 (DialysisGNN)** - 用于捕获患者间复杂关系
2. **注意力KNN模型 (AttentionKNN)** - 自适应特征权重的K近邻模型

## 新增模型特性

### 1. 图神经网络模型 (DialysisGNN)

**特点：**
- 将透析患者的临床数据建模为图结构
- 利用图神经网络捕获患者-设备关系、患者-历史记录关系
- 支持多关系图建模和注意力机制
- 具备时序图卷积和可解释性分析能力

**主要参数：**
- `hidden_dim`: 隐藏层维度 (默认: 128)
- `num_layers`: GNN层数 (默认: 3)
- `num_heads`: 注意力头数 (默认: 4)
- `dropout`: Dropout率 (默认: 0.1)
- `learning_rate`: 学习率 (默认: 0.001)
- `epochs`: 训练轮数 (默认: 100)

### 2. 注意力KNN模型 (AttentionKNN)

**特点：**
- 结合传统KNN算法和深度学习注意力机制
- 自适应地为不同特征分配权重
- 动态调整邻居的重要性
- 处理多模态临床数据，提供可解释的预测结果

**主要参数：**
- `n_neighbors`: 邻居数量 (默认: 5)
- `attention_hidden_dim`: 注意力网络隐藏层维度 (默认: 64)
- `learning_rate`: 学习率 (默认: 0.001)
- `epochs`: 训练轮数 (默认: 100)
- `distance_metric`: 距离度量方法 (默认: 'euclidean')
- `feature_scaler`: 特征缩放方法 (默认: 'standard')

## 使用方法

### 1. 运行完整的模型训练

```bash
cd /home/cht/Works/PredictionTimeHypotensionDialysis/透前模型
python3 透前数据__透中低血压建模.py
```

现在的训练流程将包括以下6个模型：
1. LightGBM
2. C-SVM
3. TabNet
4. IEDT
5. **DialysisGNN** (新增)
6. **AttentionKNN** (新增)

### 2. 单独测试新增模型

```bash
cd /home/cht/Works/PredictionTimeHypotensionDialysis/透前模型
python3 test_new_models.py
```

### 3. 在代码中使用新增模型

```python
# 导入模型
from Methods.Models.dialysis_gnn_model import DialysisGNNClassifier
from Methods.Models.attention_knn_model import AttentionKNN

# 使用图神经网络模型
gnn_model = DialysisGNNClassifier(
    hidden_dim=128,
    num_layers=3,
    num_heads=4,
    dropout=0.1,
    learning_rate=0.001,
    epochs=100,
    random_state=42
)
gnn_model.fit(X_train, y_train)
y_pred = gnn_model.predict(X_test)

# 使用注意力KNN模型
aknn_model = AttentionKNN(
    n_neighbors=5,
    attention_hidden_dim=64,
    learning_rate=0.001,
    epochs=100,
    random_state=42
)
aknn_model.fit(X_train, y_train)
y_pred = aknn_model.predict(X_test)
```

## 模型性能

根据测试结果：
- **DialysisGNN**: 在测试数据上达到71%的准确率
- **AttentionKNN**: 在测试数据上达到96.5%的准确率

## 技术要求

- **硬件**: 支持CUDA的GPU（推荐）
- **软件依赖**:
  - PyTorch
  - PyTorch Geometric
  - scikit-learn
  - pandas
  - numpy

## 文件结构

```
透前模型/
├── 透前数据__透中低血压建模.py    # 主训练脚本（已更新）
├── test_new_models.py              # 新增模型测试脚本
├── README_新增模型.md              # 本说明文档
└── ../Methods/Models/
    ├── dialysis_gnn_model.py       # 图神经网络模型
    └── attention_knn_model.py      # 注意力KNN模型
```

## 注意事项

1. **内存使用**: 图神经网络模型可能需要较大内存，建议在有足够RAM的环境中运行
2. **训练时间**: 新增模型的训练时间可能比传统模型更长
3. **GPU加速**: 两个新模型都支持GPU加速，建议在有CUDA支持的环境中运行
4. **数据预处理**: 新模型会自动处理数据预处理，包括特征缩放和图构建

## 故障排除

如果遇到导入错误，请确保：
1. 已正确安装所有依赖包
2. Python路径设置正确
3. CUDA环境配置正确（如果使用GPU）

如果遇到内存不足错误，可以：
1. 减少batch_size参数
2. 减少hidden_dim参数
3. 使用CPU而非GPU训练

## 更新日志

- **2024年**: 新增DialysisGNN和AttentionKNN模型
- 集成到现有的透前数据建模流程中
- 添加完整的测试和文档支持