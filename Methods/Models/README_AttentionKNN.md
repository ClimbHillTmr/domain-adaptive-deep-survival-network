# 注意力机制增强的K近邻模型 (Attention-Enhanced KNN)

## 概述

本项目实现了一个结合深度学习注意力机制的K近邻分类器，专门用于透析患者低血压风险预测。该模型通过自适应特征选择和动态邻居权重分配，显著提升了传统KNN算法在复杂临床数据上的性能。

## 核心特性

### 🎯 注意力机制
- **特征注意力**: 自动学习不同临床指标的重要性权重
- **邻居注意力**: 动态调整不同邻居样本的影响力
- **端到端训练**: 注意力权重与分类任务联合优化

### 🔧 技术特点
- **多模态数据支持**: 处理实验室指标、生命体征、透析参数等
- **可解释性**: 提供特征重要性和邻居权重的可视化
- **GPU加速**: 支持CUDA加速训练和推理
- **鲁棒性**: 内置早停、学习率调度等正则化技术

### 📊 临床应用
- **风险预测**: 预测透析过程中低血压发生风险
- **个性化治疗**: 基于患者特征提供个性化建议
- **实时监控**: 支持实时数据流的风险评估

## 安装要求

### 基础依赖
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### 深度学习依赖
```bash
# CPU版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# GPU版本 (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 可选依赖
```bash
pip install joblib psutil  # 模型保存和系统监控
```

## 快速开始

### 基本使用

```python
from attention_knn_model import AttentionKNN
from sklearn.model_selection import train_test_split
import numpy as np

# 准备数据
X, y = load_your_dialysis_data()  # 替换为你的数据加载函数
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建模型
model = AttentionKNN(
    n_neighbors=5,
    attention_hidden_dim=32,
    learning_rate=0.001,
    epochs=50,
    batch_size=32,
    feature_scaler='standard',
    random_state=42
)

# 训练模型
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# 获取注意力权重
feature_weights, neighbor_weights = model.get_attention_weights(X_test)
```

### 完整示例

```python
# 运行完整测试示例
python test_attention_knn.py
```

## 模型架构

### 注意力模块

```
输入特征 (n_features)
    ↓
特征注意力网络
    ↓
特征权重 (n_features)

查询样本 + 邻居样本
    ↓
邻居注意力网络
    ↓
邻居权重 (n_neighbors)
```

### 网络结构

- **特征注意力网络**:
  - Linear(n_features → hidden_dim) + ReLU + Dropout
  - Linear(hidden_dim → hidden_dim//2) + ReLU
  - Linear(hidden_dim//2 → n_features) + Sigmoid

- **邻居注意力网络**:
  - Linear(n_features*2 → hidden_dim) + ReLU + Dropout
  - Linear(hidden_dim → hidden_dim//2) + ReLU
  - Linear(hidden_dim//2 → 1) + Sigmoid

## 参数说明

### 核心参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `n_neighbors` | int | 5 | K近邻中的邻居数量 |
| `attention_hidden_dim` | int | 64 | 注意力网络隐藏层维度 |
| `learning_rate` | float | 0.001 | 学习率 |
| `epochs` | int | 100 | 训练轮数 |
| `batch_size` | int | 32 | 批次大小 |

### 正则化参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `dropout` | float | 0.1 | Dropout率 |
| `early_stopping_patience` | int | 10 | 早停耐心值 |
| `feature_scaler` | str | 'standard' | 特征缩放方法 |

### 系统参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `use_cuda` | bool | True | 是否使用GPU |
| `random_state` | int | None | 随机种子 |
| `verbose` | bool | True | 是否显示训练过程 |

## 透析数据特征

### 推荐特征列表

1. **基本信息**
   - 年龄、性别、体重、身高
   - 透析时长（月）、透析频率

2. **透析参数**
   - 干体重 (kg)
   - 超滤量 (L)
   - 透析时间 (小时)
   - 血流速度 (mL/min)

3. **生命体征**
   - 收缩压、舒张压 (mmHg)
   - 心率 (bpm)
   - 体温 (°C)

4. **实验室指标**
   - 血红蛋白 (g/dL)
   - 白蛋白 (g/dL)
   - 肌酐 (mg/dL)
   - 尿素氮 (mg/dL)
   - 电解质：钾、钠、磷 (mEq/L 或 mg/dL)

### 数据预处理建议

```python
# 处理缺失值
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# 异常值检测
from sklearn.ensemble import IsolationForest
outlier_detector = IsolationForest(contamination=0.1)
outliers = outlier_detector.fit_predict(X)
X_clean = X[outliers == 1]

# 特征选择
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k=15)
X_selected = selector.fit_transform(X, y)
```

## 性能评估

### 评估指标

```python
from attention_knn_model import evaluate_attention_knn

# 全面评估
results = evaluate_attention_knn(
    model, X_test, y_test, 
    class_names=['低风险', '高风险']
)

print(f"准确率: {results['accuracy']:.4f}")
print(f"AUC: {results['auc']:.4f}")
print(f"F1分数: {results['f1_score']:.4f}")
```

### 可视化分析

```python
# 训练历史
model.plot_training_history()

# 注意力权重
model.plot_attention_weights(
    X_test[:5], 
    feature_names=feature_names,
    sample_indices=[0, 1, 2, 3, 4]
)

# 特征重要性
feature_weights, _ = model.get_attention_weights(X_test)
mean_feature_importance = np.mean(feature_weights, axis=0)

plt.figure(figsize=(10, 6))
plt.bar(feature_names, mean_feature_importance)
plt.title('平均特征重要性')
plt.xticks(rotation=45)
plt.show()
```

## 模型保存与加载

### 保存模型

```python
# 保存完整模型
model.save_model('dialysis_attention_knn.pkl')

# 仅保存注意力权重
torch.save(model.attention_module.state_dict(), 'attention_weights.pth')
```

### 加载模型

```python
# 加载完整模型
model = AttentionKNN()
model.load_model('dialysis_attention_knn.pkl')

# 预测新数据
y_pred = model.predict(X_new)
```

## 超参数调优

### 网格搜索

```python
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin

# 定义参数网格
param_grid = {
    'n_neighbors': [3, 5, 7],
    'attention_hidden_dim': [16, 32, 64],
    'learning_rate': [0.001, 0.01, 0.1],
    'epochs': [30, 50, 100]
}

# 注意：由于训练时间较长，建议使用较小的参数范围
# 或者使用随机搜索
from sklearn.model_selection import RandomizedSearchCV

search = RandomizedSearchCV(
    AttentionKNN(verbose=False),
    param_grid,
    n_iter=10,
    cv=3,
    scoring='f1_weighted',
    random_state=42
)

search.fit(X_train, y_train)
print(f"最佳参数: {search.best_params_}")
print(f"最佳分数: {search.best_score_:.4f}")
```

### 贝叶斯优化

```python
# 使用scikit-optimize进行贝叶斯优化
try:
    from skopt import gp_minimize
    from skopt.space import Integer, Real
    from skopt.utils import use_named_args
    
    # 定义搜索空间
    dimensions = [
        Integer(3, 10, name='n_neighbors'),
        Integer(16, 128, name='attention_hidden_dim'),
        Real(0.0001, 0.1, prior='log-uniform', name='learning_rate'),
        Integer(20, 100, name='epochs')
    ]
    
    @use_named_args(dimensions)
    def objective(**params):
        model = AttentionKNN(verbose=False, **params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        return -accuracy_score(y_val, y_pred)  # 最小化负准确率
    
    # 运行优化
    result = gp_minimize(objective, dimensions, n_calls=20, random_state=42)
    print(f"最佳参数: {result.x}")
    print(f"最佳分数: {-result.fun:.4f}")
    
except ImportError:
    print("请安装scikit-optimize: pip install scikit-optimize")
```

## 实际应用案例

### 案例1：实时风险监控

```python
class RealTimeRiskMonitor:
    def __init__(self, model_path):
        self.model = AttentionKNN()
        self.model.load_model(model_path)
        self.risk_threshold = 0.7
    
    def assess_risk(self, patient_data):
        """
        评估单个患者的实时风险
        
        Args:
            patient_data: 患者当前数据 [1, n_features]
            
        Returns:
            risk_assessment: 风险评估结果
        """
        # 预测风险概率
        risk_proba = self.model.predict_proba(patient_data.reshape(1, -1))[0, 1]
        
        # 获取注意力权重
        feature_weights, neighbor_weights = self.model.get_attention_weights(
            patient_data.reshape(1, -1)
        )
        
        # 风险等级
        if risk_proba >= self.risk_threshold:
            risk_level = "高风险"
            alert = True
        elif risk_proba >= 0.4:
            risk_level = "中风险"
            alert = False
        else:
            risk_level = "低风险"
            alert = False
        
        return {
            'risk_probability': risk_proba,
            'risk_level': risk_level,
            'alert': alert,
            'key_features': self._get_key_features(feature_weights[0]),
            'recommendation': self._get_recommendation(risk_level, feature_weights[0])
        }
    
    def _get_key_features(self, feature_weights, top_k=3):
        """获取关键特征"""
        feature_names = [
            '年龄', '透析时长_月', '干体重_kg', '超滤量_L', '透析时间_h',
            '收缩压_mmHg', '舒张压_mmHg', '心率_bpm', '血红蛋白_g_dL', '白蛋白_g_dL',
            '肌酐_mg_dL', '尿素氮_mg_dL', '钾_mEq_L', '钠_mEq_L', '磷_mg_dL'
        ]
        
        importance_pairs = list(zip(feature_names, feature_weights))
        importance_pairs.sort(key=lambda x: x[1], reverse=True)
        
        return [name for name, _ in importance_pairs[:top_k]]
    
    def _get_recommendation(self, risk_level, feature_weights):
        """生成建议"""
        if risk_level == "高风险":
            return "建议立即调整透析参数，密切监控血压变化"
        elif risk_level == "中风险":
            return "建议适当调整超滤速度，加强监护"
        else:
            return "继续当前治疗方案，定期监测"

# 使用示例
monitor = RealTimeRiskMonitor('dialysis_attention_knn.pkl')
patient_data = np.array([65, 36, 70, 2.8, 4, 120, 80, 75, 11, 3.8, 8, 55, 4.2, 140, 4.5])
assessment = monitor.assess_risk(patient_data)
print(f"风险评估: {assessment}")
```

### 案例2：批量风险筛查

```python
def batch_risk_screening(model, patient_database, output_file):
    """
    批量风险筛查
    
    Args:
        model: 训练好的模型
        patient_database: 患者数据库
        output_file: 输出文件路径
    """
    results = []
    
    for patient_id, patient_data in patient_database.items():
        risk_proba = model.predict_proba(patient_data.reshape(1, -1))[0, 1]
        
        results.append({
            'patient_id': patient_id,
            'risk_probability': risk_proba,
            'risk_level': 'High' if risk_proba >= 0.7 else 'Medium' if risk_proba >= 0.4 else 'Low'
        })
    
    # 保存结果
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_file, index=False)
    
    # 统计报告
    risk_distribution = df_results['risk_level'].value_counts()
    print(f"风险分布: {risk_distribution}")
    
    return df_results
```

## 故障排除

### 常见问题

1. **CUDA内存不足**
   ```python
   # 减少批次大小
   model = AttentionKNN(batch_size=16)  # 默认32
   
   # 或使用CPU
   model = AttentionKNN(use_cuda=False)
   ```

2. **训练收敛慢**
   ```python
   # 增加学习率
   model = AttentionKNN(learning_rate=0.01)  # 默认0.001
   
   # 减少网络复杂度
   model = AttentionKNN(attention_hidden_dim=16)  # 默认64
   ```

3. **过拟合**
   ```python
   # 增加正则化
   model = AttentionKNN(dropout=0.3)  # 默认0.1
   
   # 早停
   model = AttentionKNN(early_stopping_patience=5)  # 默认10
   ```

4. **数据不平衡**
   ```python
   from sklearn.utils.class_weight import compute_class_weight
   
   # 计算类别权重
   class_weights = compute_class_weight(
       'balanced', classes=np.unique(y_train), y=y_train
   )
   
   # 在损失函数中使用权重（需要修改模型代码）
   ```

### 性能优化

1. **数据预处理优化**
   ```python
   # 使用更快的缩放器
   from sklearn.preprocessing import RobustScaler
   model = AttentionKNN(feature_scaler='none')  # 手动预处理
   
   scaler = RobustScaler()
   X_scaled = scaler.fit_transform(X)
   ```

2. **模型并行化**
   ```python
   # 使用多GPU（需要修改模型代码）
   if torch.cuda.device_count() > 1:
       model.attention_module = nn.DataParallel(model.attention_module)
   ```

## 贡献指南

### 代码贡献

1. Fork项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开Pull Request

### 报告问题

请在GitHub Issues中报告问题，包含：
- 问题描述
- 复现步骤
- 期望行为
- 系统环境信息

## 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件

## 引用

如果您在研究中使用了本模型，请引用：

```bibtex
@software{attention_knn_dialysis,
  title={Attention-Enhanced K-Nearest Neighbors for Dialysis Hypotension Prediction},
  author={透析数据分析助手},
  year={2024},
  url={https://github.com/your-repo/attention-knn-dialysis}
}
```

## 联系方式

- 项目维护者：透析数据分析助手
- 邮箱：your-email@example.com
- 项目主页：https://github.com/your-repo/attention-knn-dialysis

---

**注意**: 本模型仅用于研究目的，不应作为临床决策的唯一依据。在实际临床应用中，请务必结合医生的专业判断。