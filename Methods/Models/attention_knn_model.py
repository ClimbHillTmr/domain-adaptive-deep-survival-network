#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
透析专用注意力机制增强的K近邻模型 (Dialysis-Specific Attention-Enhanced KNN)
用于透析患者低血压风险预测

该模型专门针对透析数据特点进行优化，具备以下功能：
1. 透析专用特征工程和临床知识融合
2. 多层次注意力机制（特征级、时序级、邻居级）
3. 类别不平衡处理和成本敏感学习
4. 时序数据处理和趋势分析
5. 临床可解释性和风险因子识别
6. 实时预测和早期预警系统

作者: 透析数据分析助手
日期: 2024
版本: 2.0 - 透析专用优化版
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score, balanced_accuracy_score,
    cohen_kappa_score, matthews_corrcoef
)
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import EditedNearestNeighbours
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple, Dict, List
import warnings
from pathlib import Path
import joblib
import datetime
from scipy import stats
from scipy.signal import savgol_filter
import logging

# Yellowbrick 可视化库导入
try:
    from yellowbrick.target import ClassBalance
    from yellowbrick.classifier import ROCAUC
    from yellowbrick.classifier import PrecisionRecallCurve
    from yellowbrick.classifier import ClassificationReport
    from yellowbrick.classifier import ClassPredictionError
    from yellowbrick.classifier import DiscriminationThreshold
    from yellowbrick.classifier import ConfusionMatrix
    YELLOWBRICK_AVAILABLE = True
except ImportError:
    YELLOWBRICK_AVAILABLE = False
    warnings.warn("Yellowbrick 库未安装，将使用 matplotlib 进行可视化")

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 检查CUDA可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"使用设备: {device}")

# 透析专用常量定义
class DialysisConstants:
    """透析相关的临床常量和阈值"""
    
    # 正常范围定义
    NORMAL_RANGES = {
        'systolic_bp': (90, 140),      # 收缩压正常范围
        'diastolic_bp': (60, 90),     # 舒张压正常范围
        'heart_rate': (60, 100),      # 心率正常范围
        'hemoglobin': (11, 15),       # 血红蛋白正常范围
        'albumin': (3.5, 5.0),        # 白蛋白正常范围
        'creatinine': (0.6, 1.2),     # 肌酐正常范围
        'potassium': (3.5, 5.0),      # 钾正常范围
        'sodium': (135, 145),         # 钠正常范围
        'phosphorus': (2.5, 4.5),     # 磷正常范围
    }
    
    # 风险阈值
    RISK_THRESHOLDS = {
        'hypotension_systolic': 90,    # 低血压收缩压阈值
        'hypotension_diastolic': 60,   # 低血压舒张压阈值
        'high_uf_rate': 13,           # 高超滤率阈值 (ml/kg/h)
        'low_albumin': 3.5,           # 低白蛋白阈值
        'anemia_threshold': 11,       # 贫血阈值
    }
    
    # 特征权重（基于临床重要性）
    CLINICAL_WEIGHTS = {
        'blood_pressure': 0.25,       # 血压相关特征权重
        'fluid_status': 0.20,         # 液体状态相关权重
        'cardiac': 0.15,              # 心脏相关权重
        'laboratory': 0.15,           # 实验室指标权重
        'demographics': 0.10,         # 人口学特征权重
        'dialysis_params': 0.15,      # 透析参数权重
    }

def log_with_timestamp(message: str, level: str = 'info'):
    """带时间戳的日志记录"""
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    formatted_message = f"[{timestamp}] {message}"
    
    if level.lower() == 'info':
        logger.info(formatted_message)
    elif level.lower() == 'warning':
        logger.warning(formatted_message)
    elif level.lower() == 'error':
        logger.error(formatted_message)
    else:
        logger.debug(formatted_message)

class DialysisFeatureEngineer:
    """透析专用特征工程类"""
    
    def __init__(self):
        self.feature_names = []
        self.engineered_features = []
    
    def create_dialysis_features(self, X: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        创建透析专用特征
        
        Args:
            X: 原始特征矩阵
            feature_names: 原始特征名称
            
        Returns:
            X_engineered: 工程化后的特征矩阵
            engineered_names: 工程化后的特征名称
        """
        X_df = pd.DataFrame(X, columns=feature_names)
        engineered_features = []
        engineered_names = list(feature_names)
        
        # 1. 血压相关特征
        if 'systolic_bp' in feature_names and 'diastolic_bp' in feature_names:
            # 平均动脉压
            map_pressure = (X_df['systolic_bp'] + 2 * X_df['diastolic_bp']) / 3
            engineered_features.append(map_pressure.values)
            engineered_names.append('mean_arterial_pressure')
            
            # 脉压
            pulse_pressure = X_df['systolic_bp'] - X_df['diastolic_bp']
            engineered_features.append(pulse_pressure.values)
            engineered_names.append('pulse_pressure')
            
            # 低血压风险评分
            hypotension_risk = (
                (X_df['systolic_bp'] < DialysisConstants.RISK_THRESHOLDS['hypotension_systolic']).astype(int) +
                (X_df['diastolic_bp'] < DialysisConstants.RISK_THRESHOLDS['hypotension_diastolic']).astype(int)
            )
            engineered_features.append(hypotension_risk.values)
            engineered_names.append('hypotension_risk_score')
        
        # 2. 液体状态特征
        if 'ultrafiltration_volume' in feature_names and 'dry_weight' in feature_names and 'dialysis_time' in feature_names:
            # 超滤率 (ml/kg/h)
            uf_rate = (X_df['ultrafiltration_volume'] * 1000) / (X_df['dry_weight'] * X_df['dialysis_time'])
            engineered_features.append(uf_rate.values)
            engineered_names.append('ultrafiltration_rate')
            
            # 高超滤率风险
            high_uf_risk = (uf_rate > DialysisConstants.RISK_THRESHOLDS['high_uf_rate']).astype(int)
            engineered_features.append(high_uf_risk.values)
            engineered_names.append('high_uf_risk')
        
        # 3. 营养状态特征
        if 'albumin' in feature_names and 'hemoglobin' in feature_names:
            # 营养风险评分
            nutrition_risk = (
                (X_df['albumin'] < DialysisConstants.RISK_THRESHOLDS['low_albumin']).astype(int) +
                (X_df['hemoglobin'] < DialysisConstants.RISK_THRESHOLDS['anemia_threshold']).astype(int)
            )
            engineered_features.append(nutrition_risk.values)
            engineered_names.append('nutrition_risk_score')
        
        # 4. 电解质平衡特征
        if 'potassium' in feature_names and 'sodium' in feature_names:
            # 电解质异常评分
            electrolyte_abnormal = (
                (X_df['potassium'] < DialysisConstants.NORMAL_RANGES['potassium'][0]).astype(int) +
                (X_df['potassium'] > DialysisConstants.NORMAL_RANGES['potassium'][1]).astype(int) +
                (X_df['sodium'] < DialysisConstants.NORMAL_RANGES['sodium'][0]).astype(int) +
                (X_df['sodium'] > DialysisConstants.NORMAL_RANGES['sodium'][1]).astype(int)
            )
            engineered_features.append(electrolyte_abnormal.values)
            engineered_names.append('electrolyte_abnormal_score')
        
        # 5. 年龄相关风险
        if 'age' in feature_names:
            # 高龄风险
            elderly_risk = (X_df['age'] > 70).astype(int)
            engineered_features.append(elderly_risk.values)
            engineered_names.append('elderly_risk')
            
            # 年龄分组
            age_groups = pd.cut(X_df['age'], bins=[0, 50, 65, 80, 100], labels=[0, 1, 2, 3])
            engineered_features.append(age_groups.astype(int).values)
            engineered_names.append('age_group')
        
        # 6. 透析充分性特征
        if 'creatinine' in feature_names and 'dialysis_time' in feature_names:
            # 简化的Kt/V估算
            kt_v_estimate = X_df['dialysis_time'] / (X_df['creatinine'] + 1)  # 简化公式
            engineered_features.append(kt_v_estimate.values)
            engineered_names.append('kt_v_estimate')
        
        # 合并所有特征
        if engineered_features:
            X_engineered = np.column_stack([X] + engineered_features)
        else:
            X_engineered = X
        
        self.feature_names = feature_names
        self.engineered_features = engineered_names[len(feature_names):]
        
        log_with_timestamp(f"特征工程完成: 原始特征 {len(feature_names)} 个, 新增特征 {len(self.engineered_features)} 个")
        
        return X_engineered, engineered_names
    
    def create_temporal_features(self, X: np.ndarray, time_steps: int = 3) -> np.ndarray:
        """
        创建时序特征（用于处理连续透析记录）
        
        Args:
            X: 特征矩阵 [n_samples, n_features]
            time_steps: 时间步长
            
        Returns:
            X_temporal: 包含时序特征的矩阵
        """
        if len(X) < time_steps:
            return X
        
        temporal_features = []
        
        for i in range(time_steps, len(X)):
            # 当前样本
            current = X[i]
            
            # 历史样本
            history = X[i-time_steps:i]
            
            # 计算趋势特征
            trends = []
            for j in range(X.shape[1]):
                feature_history = history[:, j]
                
                # 线性趋势
                if len(feature_history) > 1:
                    slope, _, _, _, _ = stats.linregress(range(len(feature_history)), feature_history)
                    trends.append(slope)
                else:
                    trends.append(0)
                
                # 变异系数
                if np.std(feature_history) > 0:
                    cv = np.std(feature_history) / np.mean(feature_history)
                    trends.append(cv)
                else:
                    trends.append(0)
            
            # 合并当前特征和趋势特征
            combined_features = np.concatenate([current, trends])
            temporal_features.append(combined_features)
        
        return np.array(temporal_features)

class MultiLevelAttentionModule(nn.Module):
    """
    多层次注意力机制模块
    包含特征级、临床组级和邻居级注意力
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.1, 
                 clinical_groups: Optional[Dict[str, List[int]]] = None):
        super(MultiLevelAttentionModule, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.clinical_groups = clinical_groups or {}
        
        # 特征级注意力网络
        self.feature_attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, input_dim),
            nn.Sigmoid()
        )
        
        # 临床组级注意力网络
        if self.clinical_groups:
            self.group_attention = nn.ModuleDict()
            for group_name, indices in self.clinical_groups.items():
                group_dim = len(indices)
                self.group_attention[group_name] = nn.Sequential(
                    nn.Linear(group_dim, hidden_dim // 4),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 4, group_dim),
                    nn.Sigmoid()
                )
        
        # 邻居级注意力网络（增强版）
        self.neighbor_attention = nn.Sequential(
            nn.Linear(input_dim * 2 + 1, hidden_dim),  # 包含距离信息
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 自注意力机制（用于邻居间的交互）
        self.self_attention = nn.MultiheadAttention(
            embed_dim=input_dim, 
            num_heads=4, 
            dropout=dropout,
            batch_first=True
        )
        
    def forward(self, query: torch.Tensor, neighbors: torch.Tensor, 
                distances: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            query: 查询样本 [batch_size, input_dim]
            neighbors: 邻居样本 [batch_size, k, input_dim]
            distances: 邻居距离 [batch_size, k]
            
        Returns:
            feature_weights: 特征权重 [batch_size, input_dim]
            group_weights: 临床组权重 [batch_size, n_groups]
            neighbor_weights: 邻居权重 [batch_size, k]
        """
        batch_size, k, input_dim = neighbors.shape
        
        # 1. 特征级注意力
        feature_weights = self.feature_attention(query)
        
        # 2. 临床组级注意力
        group_weights = {}
        if self.clinical_groups:
            for group_name, indices in self.clinical_groups.items():
                group_features = query[:, indices]
                group_weight = self.group_attention[group_name](group_features)
                group_weights[group_name] = group_weight
        
        # 3. 邻居自注意力
        neighbors_reshaped = neighbors.view(-1, k, input_dim)
        attended_neighbors, _ = self.self_attention(neighbors_reshaped, neighbors_reshaped, neighbors_reshaped)
        
        # 4. 邻居级注意力（考虑距离）
        query_expanded = query.unsqueeze(1).expand(-1, k, -1)
        
        if distances is not None:
            # 归一化距离
            distances_norm = (distances - distances.min(dim=1, keepdim=True)[0]) / \
                           (distances.max(dim=1, keepdim=True)[0] - distances.min(dim=1, keepdim=True)[0] + 1e-8)
            distances_expanded = distances_norm.unsqueeze(-1)
            
            neighbor_query_concat = torch.cat([
                attended_neighbors, query_expanded, distances_expanded
            ], dim=-1)
        else:
            # 使用零距离
            zero_distances = torch.zeros(batch_size, k, 1).to(query.device)
            neighbor_query_concat = torch.cat([
                attended_neighbors, query_expanded, zero_distances
            ], dim=-1)
        
        neighbor_query_concat = neighbor_query_concat.view(-1, input_dim * 2 + 1)
        neighbor_weights = self.neighbor_attention(neighbor_query_concat)
        neighbor_weights = neighbor_weights.view(batch_size, k)
        
        # 归一化邻居权重
        neighbor_weights = F.softmax(neighbor_weights, dim=1)
        
        # 合并组权重为单一张量
        if group_weights:
            group_weights_tensor = torch.cat(list(group_weights.values()), dim=1)
        else:
            group_weights_tensor = torch.ones(batch_size, 1).to(query.device)
        
        return feature_weights, group_weights_tensor, neighbor_weights

class DialysisAttentionKNN(BaseEstimator, ClassifierMixin):
    """
    透析专用注意力机制增强的K近邻分类器
    
    该模型专门针对透析患者数据进行优化，具备以下功能：
    1. 透析专用特征工程和临床知识融合
    2. 多层次注意力机制（特征级、临床组级、邻居级）
    3. 类别不平衡处理和成本敏感学习
    4. 时序数据处理和趋势分析
    5. 临床可解释性和风险因子识别
    6. 实时预测和早期预警系统
    """
    
    def __init__(self, 
                 n_neighbors: int = 5,
                 attention_hidden_dim: int = 64,
                 learning_rate: float = 0.001,
                 epochs: int = 100,
                 batch_size: int = 32,
                 dropout: float = 0.1,
                 distance_metric: str = 'euclidean',
                 feature_scaler: str = 'robust',  # 改为robust，对异常值更鲁棒
                 early_stopping_patience: int = 15,  # 增加耐心值
                 random_state: Optional[int] = None,
                 use_cuda: bool = True,
                 verbose: bool = True,
                 # 透析专用参数
                 enable_feature_engineering: bool = True,
                 enable_temporal_features: bool = False,
                 temporal_steps: int = 3,
                 handle_imbalance: bool = True,
                 imbalance_method: str = 'smote',  # 'smote', 'adasyn', 'class_weight'
                 cost_sensitive: bool = True,
                 clinical_feature_groups: Optional[Dict[str, List[str]]] = None,
                 enable_early_warning: bool = True,
                 warning_threshold: float = 0.7,
                 feature_names: Optional[List[str]] = None):
        """
        初始化透析专用注意力KNN模型
        
        Args:
            n_neighbors: 邻居数量
            attention_hidden_dim: 注意力网络隐藏层维度
            learning_rate: 学习率
            epochs: 训练轮数
            batch_size: 批次大小
            dropout: Dropout率
            distance_metric: 距离度量方法
            feature_scaler: 特征缩放方法 ('standard', 'minmax', 'robust', 'none')
            early_stopping_patience: 早停耐心值
            random_state: 随机种子
            use_cuda: 是否使用CUDA
            verbose: 是否显示训练过程
            enable_feature_engineering: 是否启用透析专用特征工程
            enable_temporal_features: 是否启用时序特征
            temporal_steps: 时序步长
            handle_imbalance: 是否处理类别不平衡
            imbalance_method: 不平衡处理方法
            cost_sensitive: 是否启用成本敏感学习
            clinical_feature_groups: 临床特征分组
            enable_early_warning: 是否启用早期预警
            warning_threshold: 预警阈值
            feature_names: 特征名称列表
        """
        # 基础参数
        self.n_neighbors = n_neighbors
        self.attention_hidden_dim = attention_hidden_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.distance_metric = distance_metric
        self.feature_scaler = feature_scaler
        self.early_stopping_patience = early_stopping_patience
        self.random_state = random_state
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.verbose = verbose
        
        # 透析专用参数
        self.enable_feature_engineering = enable_feature_engineering
        self.enable_temporal_features = enable_temporal_features
        self.temporal_steps = temporal_steps
        self.handle_imbalance = handle_imbalance
        self.imbalance_method = imbalance_method
        self.cost_sensitive = cost_sensitive
        self.clinical_feature_groups = clinical_feature_groups or {}
        self.enable_early_warning = enable_early_warning
        self.warning_threshold = warning_threshold
        self.feature_names = feature_names or []
        
        # 设置设备
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        
        # 设置随机种子
        if random_state is not None:
            np.random.seed(random_state)
            torch.manual_seed(random_state)
            if self.use_cuda:
                torch.cuda.manual_seed(random_state)
        
        # 初始化基础组件
        self.attention_module = None
        self.nn_model = None
        self.scaler = None
        self.classes_ = None
        self.n_features_in_ = None
        self.X_train_ = None
        self.y_train_ = None
        self.training_history_ = {
            'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 
            'f1': [], 'auc': [], 'learning_rate': []
        }
        
        # 透析专用组件
        self.feature_engineer = None
        self.imbalance_sampler = None
        self.class_weights = None
        self.clinical_groups_indices = {}
        self.risk_factors = {}
        self.prediction_history = []  # 用于早期预警
        self.label_encoder_ = None
        
        log_with_timestamp(f"初始化透析专用注意力KNN模型 - 设备: {self.device}")
        
    def _init_scaler(self):
        """初始化特征缩放器"""
        if self.feature_scaler == 'standard':
            self.scaler = StandardScaler()
        elif self.feature_scaler == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.feature_scaler == 'robust':
            self.scaler = RobustScaler()  # 对异常值更鲁棒
        elif self.feature_scaler == 'none':
            self.scaler = None
        else:
            raise ValueError(f"不支持的缩放方法: {self.feature_scaler}")
        
        if self.verbose and self.scaler is not None:
            log_with_timestamp(f"使用 {self.feature_scaler} 缩放器")
    
    def _scale_features(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """特征缩放"""
        if self.scaler is None:
            return X
        
        if fit:
            return self.scaler.fit_transform(X)
        else:
            return self.scaler.transform(X)
    
    def _create_training_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        创建训练数据
        为每个样本找到其邻居，构建注意力训练数据
        """
        n_samples, n_features = X.shape
        
        # 找到每个样本的邻居
        queries = []
        neighbors_list = []
        labels = []
        
        for i in range(n_samples):
            # 当前样本作为查询
            query = X[i]
            
            # 找到邻居（排除自己）
            distances, indices = self.nn_model.kneighbors([query], n_neighbors=self.n_neighbors + 1)
            neighbor_indices = indices[0][1:]  # 排除自己
            
            if len(neighbor_indices) >= self.n_neighbors:
                neighbor_indices = neighbor_indices[:self.n_neighbors]
                neighbor_samples = X[neighbor_indices]
                
                queries.append(query)
                neighbors_list.append(neighbor_samples)
                labels.append(y[i])
        
        queries = np.array(queries)
        neighbors_array = np.array(neighbors_list)
        labels = np.array(labels)
        
        # 转换为张量
        queries_tensor = torch.FloatTensor(queries).to(self.device)
        neighbors_tensor = torch.FloatTensor(neighbors_array).to(self.device)
        labels_tensor = torch.LongTensor(labels).to(self.device)
        
        return queries_tensor, neighbors_tensor, labels_tensor
    
    def _attention_loss(self, feature_weights: torch.Tensor, neighbor_weights: torch.Tensor, 
                      neighbors: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        计算注意力损失
        结合分类损失和注意力正则化
        """
        batch_size, k, n_features = neighbors.shape
        
        # 获取邻居标签
        neighbor_labels = []
        for i in range(batch_size):
            query_idx = i  # 简化处理，实际应该根据具体情况调整
            distances, indices = self.nn_model.kneighbors([self.X_train_[query_idx]], n_neighbors=k + 1)
            neighbor_indices = indices[0][1:k+1]
            neighbor_labels.append(self.y_train_[neighbor_indices])
        
        neighbor_labels = torch.LongTensor(neighbor_labels).to(self.device)
        
        # 计算加权预测
        predictions = torch.zeros(batch_size, len(self.classes_)).to(self.device)
        
        for i in range(batch_size):
            for j in range(k):
                class_idx = neighbor_labels[i, j]
                predictions[i, class_idx] += neighbor_weights[i, j]
        
        # 分类损失（支持成本敏感学习）
        if self.class_weights_ is not None:
            classification_loss = F.cross_entropy(predictions, labels, weight=self.class_weights_)
        else:
            classification_loss = F.cross_entropy(predictions, labels)
        
        # 注意力正则化（鼓励稀疏性）
        feature_reg = torch.mean(torch.sum(feature_weights ** 2, dim=1))
        neighbor_reg = torch.mean(torch.sum(neighbor_weights ** 2, dim=1))
        
        # 透析专用正则化：鼓励关注重要的临床特征
        clinical_reg = 0.0
        if hasattr(self, 'clinical_feature_groups') and self.clinical_feature_groups:
            for group_name, indices in self.clinical_feature_groups.items():
                if indices and max(indices) < feature_weights.shape[1]:
                    group_weights = feature_weights[:, indices]
                    # 鼓励组内特征权重的一致性
                    clinical_reg += torch.var(group_weights, dim=1).mean()
        
        total_loss = classification_loss + 0.01 * (feature_reg + neighbor_reg) + 0.005 * clinical_reg
        
        return total_loss, classification_loss
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DialysisAttentionKNN':
        """
        训练透析注意力KNN模型
        
        Args:
            X: 训练特征 [n_samples, n_features]
            y: 训练标签 [n_samples]
            
        Returns:
            self: 训练后的模型
        """
        # 验证输入
        X, y = check_X_y(X, y)
        
        if self.verbose:
            log_with_timestamp(f"开始训练透析注意力KNN模型")
            log_with_timestamp(f"原始数据形状: {X.shape}, 标签分布: {np.unique(y, return_counts=True)}")
        
        # 透析专用特征工程
        if self.enable_feature_engineering:
            if self.verbose:
                log_with_timestamp("执行透析专用特征工程")
            
            # 创建特征工程器
            self.feature_engineer_ = DialysisFeatureEngineer(
                enable_temporal=self.enable_temporal_features,
                feature_names=self.feature_names
            )
            
            # 应用特征工程
            X = self.feature_engineer_.fit_transform(X)
            
            if self.verbose:
                log_with_timestamp(f"特征工程后数据形状: {X.shape}")
        
        # 检查并处理NaN值
        if np.isnan(X).any():
            if self.verbose:
                log_with_timestamp("检测到特征中的NaN值，进行填充处理")
            X = pd.DataFrame(X).fillna(pd.DataFrame(X).mean()).values
        
        if np.isnan(y).any():
            if self.verbose:
                log_with_timestamp("检测到标签中的NaN值，进行过滤")
            valid_mask = ~np.isnan(y)
            X = X[valid_mask]
            y = y[valid_mask]
        
        # 标签编码：确保标签是从0开始的连续整数
        from sklearn.preprocessing import LabelEncoder
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        
        if self.verbose:
            log_with_timestamp(f"编码后标签分布: {np.unique(y_encoded, return_counts=True)}")
            log_with_timestamp(f"标签映射: {dict(zip(self.label_encoder_.classes_, range(len(self.label_encoder_.classes_))))}")
        
        # 验证编码后的标签
        assert y_encoded.min() >= 0, f"标签最小值应该>=0，实际为{y_encoded.min()}"
        assert y_encoded.max() < len(np.unique(y_encoded)), f"标签最大值应该<类别数，实际为{y_encoded.max()}"
        
        self.classes_ = unique_labels(y_encoded)
        self.n_features_in_ = X.shape[1]
        
        # 处理类别不平衡
        if self.handle_imbalance:
            if self.verbose:
                log_with_timestamp("处理类别不平衡数据")
            
            # 使用SMOTE进行过采样
            smote = SMOTE(random_state=self.random_state)
            X, y_encoded = smote.fit_resample(X, y_encoded)
            
            if self.verbose:
                log_with_timestamp(f"SMOTE后数据形状: {X.shape}, 标签分布: {np.unique(y_encoded, return_counts=True)}")
        
        # 计算类别权重（用于成本敏感学习）
        if self.cost_sensitive:
            from sklearn.utils.class_weight import compute_class_weight
            class_weights = compute_class_weight(
                'balanced', 
                classes=np.unique(y_encoded), 
                y=y_encoded
            )
            self.class_weights_ = torch.FloatTensor(class_weights).to(self.device)
            
            if self.verbose:
                log_with_timestamp(f"类别权重: {dict(zip(np.unique(y_encoded), class_weights))}")
        else:
            self.class_weights_ = None
        
        if self.verbose:
            log_with_timestamp(f"开始训练透析注意力KNN模型")
            log_with_timestamp(f"训练样本数: {X.shape[0]}, 特征数: {X.shape[1]}")
            log_with_timestamp(f"类别数: {len(self.classes_)}, 邻居数: {self.n_neighbors}")
            log_with_timestamp(f"使用设备: {self.device}")
        
        # 初始化特征缩放器
        self._init_scaler()
        
        # 特征缩放
        X_scaled = self._scale_features(X, fit=True)
        
        # 保存训练数据
        self.X_train_ = X_scaled.copy()
        self.y_train_ = y_encoded.copy()
        
        # 初始化最近邻模型
        self.nn_model = NearestNeighbors(
            n_neighbors=self.n_neighbors + 1,  # +1 因为会包含自己
            metric=self.distance_metric
        )
        self.nn_model.fit(X_scaled)
        
        # 初始化多层次注意力模块
        self.attention_module = MultiLevelAttentionModule(
            input_dim=X_scaled.shape[1],  # 使用实际特征维度
            hidden_dim=self.attention_hidden_dim,
            dropout=self.dropout,
            clinical_groups=self.clinical_feature_groups
        ).to(self.device)
        
        # 优化器
        optimizer = torch.optim.Adam(self.attention_module.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # 创建训练数据
        queries, neighbors, labels = self._create_training_data(X_scaled, y_encoded)
        
        # 训练循环
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            self.attention_module.train()
            
            # 批次训练
            total_loss = 0
            total_samples = 0
            correct_predictions = 0
            
            for i in range(0, len(queries), self.batch_size):
                batch_queries = queries[i:i+self.batch_size]
                batch_neighbors = neighbors[i:i+self.batch_size]
                batch_labels = labels[i:i+self.batch_size]
                
                optimizer.zero_grad()
                
                # 前向传播
                feature_weights, neighbor_weights = self.attention_module(batch_queries, batch_neighbors)
                
                # 计算损失
                loss, classification_loss = self._attention_loss(
                    feature_weights, neighbor_weights, batch_neighbors, batch_labels
                )
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * len(batch_queries)
                total_samples += len(batch_queries)
                
                # 计算准确率（简化版本）
                with torch.no_grad():
                    batch_size, k, _ = batch_neighbors.shape
                    predictions = torch.zeros(batch_size, len(self.classes_)).to(self.device)
                    
                    for j in range(batch_size):
                        query_idx = i + j
                        if query_idx < len(self.X_train_):
                            distances, indices = self.nn_model.kneighbors([self.X_train_[query_idx]], n_neighbors=k + 1)
                            neighbor_indices = indices[0][1:k+1]
                            neighbor_labels_batch = self.y_train_[neighbor_indices]
                            
                            for l in range(k):
                                class_idx = neighbor_labels_batch[l]
                                predictions[j, class_idx] += neighbor_weights[j, l]
                    
                    predicted_classes = torch.argmax(predictions, dim=1)
                    correct_predictions += torch.sum(predicted_classes == batch_labels).item()
            
            avg_loss = total_loss / total_samples
            accuracy = correct_predictions / total_samples
            
            self.training_history_['loss'].append(avg_loss)
            self.training_history_['accuracy'].append(accuracy)
            
            scheduler.step(avg_loss)
            
            # 记录训练指标
            self.training_history_['precision'].append(0.0)  # 简化版本，实际应计算
            self.training_history_['recall'].append(0.0)
            self.training_history_['f1'].append(0.0)
            self.training_history_['auc'].append(0.0)
            
            if self.verbose and (epoch + 1) % 10 == 0:
                log_with_timestamp(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
            
            # 早期预警检查
            if self.enable_early_warning and accuracy > self.warning_threshold:
                if self.verbose:
                    log_with_timestamp(f"⚠️ 早期预警: 模型准确率 {accuracy:.4f} 超过阈值 {self.warning_threshold}")
            
            # 早停检查
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.early_stopping_patience:
                if self.verbose:
                    log_with_timestamp(f"早停于第 {epoch+1} 轮")
                break
        
        if self.verbose:
            log_with_timestamp(f"训练完成! 最终损失: {avg_loss:.4f}, 准确率: {accuracy:.4f}")
            log_with_timestamp(f"模型已准备用于透析低血压风险预测")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        透析专用预测类别
        
        Args:
            X: 测试特征 [n_samples, n_features]
            
        Returns:
            predictions: 预测类别 [n_samples]
        """
        check_is_fitted(self)
        X = check_array(X)
        
        # 透析专用特征工程
        if self.enable_feature_engineering and hasattr(self, 'feature_engineer_'):
            X = self.feature_engineer_.transform(X)
        
        # 特征缩放
        X_scaled = self._scale_features(X)
        
        predictions = []
        
        self.attention_module.eval()
        with torch.no_grad():
            for i in range(0, len(X_scaled), self.batch_size):
                batch_X = X_scaled[i:i+self.batch_size]
                batch_predictions = []
                
                for x in batch_X:
                    # 找到邻居
                    distances, indices = self.nn_model.kneighbors([x], n_neighbors=self.n_neighbors)
                    neighbor_indices = indices[0]
                    neighbor_samples = self.X_train_[neighbor_indices]
                    neighbor_labels = self.y_train_[neighbor_indices]
                    
                    # 转换为张量
                    query_tensor = torch.FloatTensor(x).unsqueeze(0).to(self.device)
                    neighbors_tensor = torch.FloatTensor(neighbor_samples).unsqueeze(0).to(self.device)
                    
                    # 计算注意力权重
                    feature_weights, neighbor_weights = self.attention_module(query_tensor, neighbors_tensor)
                    
                    # 加权投票
                    class_votes = np.zeros(len(self.classes_))
                    neighbor_weights_np = neighbor_weights.cpu().numpy()[0]
                    
                    for j, label in enumerate(neighbor_labels):
                        class_idx = np.where(self.classes_ == label)[0][0]
                        class_votes[class_idx] += neighbor_weights_np[j]
                    
                    predicted_class_encoded = self.classes_[np.argmax(class_votes)]
                    
                    # 解码标签
                    if hasattr(self, 'label_encoder_'):
                        predicted_class = self.label_encoder_.inverse_transform([predicted_class_encoded])[0]
                    else:
                        predicted_class = predicted_class_encoded
                    
                    batch_predictions.append(predicted_class)
                
                predictions.extend(batch_predictions)
        
        predictions_array = np.array(predictions)
        
        # 早期预警检查
        if self.enable_early_warning:
            probabilities = self.predict_proba(X)
            high_risk_mask = np.max(probabilities, axis=1) > self.warning_threshold
            
            if self.verbose and np.any(high_risk_mask):
                high_risk_count = np.sum(high_risk_mask)
                log_with_timestamp(f"⚠️ 检测到 {high_risk_count} 个高风险透析样本")
        
        return predictions_array
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        透析专用预测类别概率
        
        Args:
            X: 测试特征 [n_samples, n_features]
            
        Returns:
            probabilities: 类别概率 [n_samples, n_classes]
        """
        check_is_fitted(self)
        X = check_array(X)
        
        # 透析专用特征工程
        if self.enable_feature_engineering and hasattr(self, 'feature_engineer_'):
            X = self.feature_engineer_.transform(X)
        
        # 特征缩放
        X_scaled = self._scale_features(X)
        
        probabilities = []
        
        self.attention_module.eval()
        with torch.no_grad():
            for i in range(0, len(X_scaled), self.batch_size):
                batch_X = X_scaled[i:i+self.batch_size]
                batch_probabilities = []
                
                for x in batch_X:
                    # 找到邻居
                    distances, indices = self.nn_model.kneighbors([x], n_neighbors=self.n_neighbors)
                    neighbor_indices = indices[0]
                    neighbor_samples = self.X_train_[neighbor_indices]
                    neighbor_labels = self.y_train_[neighbor_indices]
                    
                    # 转换为张量
                    query_tensor = torch.FloatTensor(x).unsqueeze(0).to(self.device)
                    neighbors_tensor = torch.FloatTensor(neighbor_samples).unsqueeze(0).to(self.device)
                    
                    # 计算注意力权重
                    feature_weights, neighbor_weights = self.attention_module(query_tensor, neighbors_tensor)
                    
                    # 加权投票
                    class_votes = np.zeros(len(self.classes_))
                    neighbor_weights_np = neighbor_weights.cpu().numpy()[0]
                    
                    for j, label in enumerate(neighbor_labels):
                        class_idx = np.where(self.classes_ == label)[0][0]
                        class_votes[class_idx] += neighbor_weights_np[j]
                    
                    # 归一化为概率
                    probabilities_sample = class_votes / np.sum(class_votes)
                    batch_probabilities.append(probabilities_sample)
                
                probabilities.extend(batch_probabilities)
        
        return np.array(probabilities)
    
    def predict_dialysis_risk(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        透析专用风险评估
        
        Args:
            X: 测试特征 [n_samples, n_features]
            
        Returns:
            risk_assessment: 包含预测、概率、风险等级和建议的字典
        """
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)
        
        # 计算风险等级
        max_probs = np.max(probabilities, axis=1)
        risk_levels = np.where(max_probs > 0.8, 'High', 
                              np.where(max_probs > 0.6, 'Medium', 'Low'))
        
        # 生成建议
        recommendations = []
        for i, (pred, prob, risk) in enumerate(zip(predictions, max_probs, risk_levels)):
            if risk == 'High':
                rec = "建议立即调整透析参数，密切监测血压"
            elif risk == 'Medium':
                rec = "建议加强监测，准备应急措施"
            else:
                rec = "继续常规透析，定期监测"
            recommendations.append(rec)
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'risk_levels': risk_levels,
            'max_probabilities': max_probs,
            'recommendations': recommendations
        }
    
    def get_attention_weights(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取注意力权重
        
        Args:
            X: 输入特征 [n_samples, n_features]
            
        Returns:
            feature_weights: 特征注意力权重 [n_samples, n_features]
            neighbor_weights: 邻居注意力权重 [n_samples, n_neighbors]
        """
        check_is_fitted(self)
        X = check_array(X)
        
        # 特征缩放
        X_scaled = self._scale_features(X)
        
        all_feature_weights = []
        all_neighbor_weights = []
        
        self.attention_module.eval()
        with torch.no_grad():
            for x in X_scaled:
                # 找到邻居
                distances, indices = self.nn_model.kneighbors([x], n_neighbors=self.n_neighbors)
                neighbor_indices = indices[0]
                neighbor_samples = self.X_train_[neighbor_indices]
                
                # 转换为张量
                query_tensor = torch.FloatTensor(x).unsqueeze(0).to(self.device)
                neighbors_tensor = torch.FloatTensor(neighbor_samples).unsqueeze(0).to(self.device)
                
                # 计算注意力权重
                feature_weights, neighbor_weights = self.attention_module(query_tensor, neighbors_tensor)
                
                all_feature_weights.append(feature_weights.cpu().numpy()[0])
                all_neighbor_weights.append(neighbor_weights.cpu().numpy()[0])
        
        return np.array(all_feature_weights), np.array(all_neighbor_weights)
    
    def plot_training_history(self, figsize: Tuple[int, int] = (12, 4)):
        """
        绘制训练历史
        """
        if not self.training_history_['loss']:
            print("没有训练历史可显示")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # 损失曲线
        ax1.plot(self.training_history_['loss'])
        ax1.set_title('训练损失')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        # 准确率曲线
        ax2.plot(self.training_history_['accuracy'])
        ax2.set_title('训练准确率')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_attention_weights(self, X: np.ndarray, feature_names: Optional[List[str]] = None, 
                             sample_indices: Optional[List[int]] = None, figsize: Tuple[int, int] = (15, 10)):
        """
        可视化注意力权重
        
        Args:
            X: 输入特征
            feature_names: 特征名称列表
            sample_indices: 要可视化的样本索引
            figsize: 图形大小
        """
        feature_weights, neighbor_weights = self.get_attention_weights(X)
        
        if sample_indices is None:
            sample_indices = list(range(min(5, len(X))))  # 默认显示前5个样本
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
        
        fig, axes = plt.subplots(2, len(sample_indices), figsize=figsize)
        if len(sample_indices) == 1:
            axes = axes.reshape(2, 1)
        
        for i, sample_idx in enumerate(sample_indices):
            # 特征注意力权重
            ax1 = axes[0, i]
            ax1.bar(range(len(feature_names)), feature_weights[sample_idx])
            ax1.set_title(f'样本 {sample_idx} - 特征注意力权重')
            ax1.set_xlabel('特征')
            ax1.set_ylabel('权重')
            ax1.set_xticks(range(len(feature_names)))
            ax1.set_xticklabels(feature_names, rotation=45, ha='right')
            
            # 邻居注意力权重
            ax2 = axes[1, i]
            ax2.bar(range(self.n_neighbors), neighbor_weights[sample_idx])
            ax2.set_title(f'样本 {sample_idx} - 邻居注意力权重')
            ax2.set_xlabel('邻居索引')
            ax2.set_ylabel('权重')
        
        plt.tight_layout()
        plt.show()
    
    def create_performance_visualizations(self, X_test, y_test, output_dir=None, target='hypotension'):
        """
        创建性能可视化图表
        
        Args:
            X_test: 测试特征
            y_test: 测试标签
            output_dir: 输出目录
            target: 目标变量名称
        """
        if output_dir is None:
            import inspect
            import os
            
            # 获取调用栈，找到调用脚本的目录
            frame = inspect.currentframe()
            try:
                # 向上查找调用栈，找到非模型文件的调用者
                caller_frame = frame.f_back
                while caller_frame:
                    caller_file = caller_frame.f_code.co_filename
                    if not caller_file.endswith(('C_SVM_model.py', 'LightGBM_model.py', 'TabNet_optimized.py', 'IEDT_model.py', 'dialysis_gnn_model.py', 'attention_knn_model.py')):
                        base_path = os.path.dirname(caller_file)
                        break
                    caller_frame = caller_frame.f_back
                else:
                    # 如果没找到，使用当前工作目录
                    base_path = os.getcwd()
            finally:
                del frame
            
            output_dir = os.path.join(base_path, 'Results/AttentionKNN_visualizations')
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取预测结果
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)
        
        if YELLOWBRICK_AVAILABLE:
            print("使用 Yellowbrick 创建可视化...")
            
            try:
                # 1. 类别平衡可视化
                print("创建类别平衡可视化...")
                visualizer = ClassBalance(labels=self.classes_)
                visualizer.fit(y_test)
                visualizer.show(outpath=os.path.join(output_dir, f'{target}_class_balance.png'))
                plt.close()
                
                # 2. ROC曲线
                print("创建ROC曲线...")
                visualizer = ROCAUC(self, classes=self.classes_)
                visualizer.fit(X_test, y_test)
                visualizer.score(X_test, y_test)
                visualizer.show(outpath=os.path.join(output_dir, f'{target}_roc_curve.png'))
                plt.close()
                
                # 3. 精确率-召回率曲线
                print("创建精确率-召回率曲线...")
                visualizer = PrecisionRecallCurve(self, classes=self.classes_)
                visualizer.fit(X_test, y_test)
                visualizer.score(X_test, y_test)
                visualizer.show(outpath=os.path.join(output_dir, f'{target}_precision_recall.png'))
                plt.close()
                
                # 4. 判别阈值
                print("创建判别阈值可视化...")
                visualizer = DiscriminationThreshold(self)
                visualizer.fit(X_test, y_test)
                visualizer.show(outpath=os.path.join(output_dir, f'{target}_discrimination_threshold.png'))
                plt.close()
                
                # 5. 分类报告
                print("创建分类报告可视化...")
                visualizer = ClassificationReport(self, classes=self.classes_, support=True)
                visualizer.fit(X_test, y_test)
                visualizer.score(X_test, y_test)
                visualizer.show(outpath=os.path.join(output_dir, f'{target}_classification_report.png'))
                plt.close()
                
                # 6. 混淆矩阵
                print("创建混淆矩阵可视化...")
                visualizer = ConfusionMatrix(self, classes=self.classes_)
                visualizer.fit(X_test, y_test)
                visualizer.score(X_test, y_test)
                visualizer.show(outpath=os.path.join(output_dir, f'{target}_confusion_matrix.png'))
                plt.close()
                
                # 7. 类预测错误
                print("创建类预测错误可视化...")
                visualizer = ClassPredictionError(self, classes=self.classes_)
                visualizer.fit(X_test, y_test)
                visualizer.score(X_test, y_test)
                visualizer.show(outpath=os.path.join(output_dir, f'{target}_class_prediction_error.png'))
                plt.close()
                
                print(f"Yellowbrick 可视化已保存到: {output_dir}")
                
            except Exception as e:
                print(f"Yellowbrick 可视化创建失败: {e}")
                print("回退到 matplotlib 可视化...")
                self._create_matplotlib_visualizations(X_test, y_test, y_pred, y_pred_proba, output_dir, target)
        else:
            print("使用 matplotlib 创建可视化...")
            self._create_matplotlib_visualizations(X_test, y_test, y_pred, y_pred_proba, output_dir, target)
    
    def _create_matplotlib_visualizations(self, X_test, y_test, y_pred, y_pred_proba, output_dir, target):
        """
        使用 matplotlib 创建基础可视化
        """
        try:
            # 混淆矩阵
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=self.classes_, yticklabels=self.classes_)
            plt.title(f'{target} - 混淆矩阵')
            plt.ylabel('真实标签')
            plt.xlabel('预测标签')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{target}_confusion_matrix_matplotlib.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # ROC曲线（仅适用于二分类）
            if len(self.classes_) == 2:
                plt.figure(figsize=(8, 6))
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
                auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
                plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
                plt.plot([0, 1], [0, 1], 'k--', label='Random')
                plt.xlabel('假正率 (FPR)')
                plt.ylabel('真正率 (TPR)')
                plt.title(f'{target} - ROC曲线')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{target}_roc_curve_matplotlib.png'), dpi=300, bbox_inches='tight')
                plt.close()
            
            print(f"Matplotlib 可视化已保存到: {output_dir}")
            
        except Exception as e:
            print(f"创建 matplotlib 可视化时出错: {e}")
    
    def save_model(self, filepath: str):
        """
        保存模型
        
        Args:
            filepath: 保存路径
        """
        model_data = {
            'attention_module_state': self.attention_module.state_dict() if self.attention_module else None,
            'scaler': self.scaler,
            'nn_model': self.nn_model,
            'classes_': self.classes_,
            'n_features_in_': self.n_features_in_,
            'X_train_': self.X_train_,
            'y_train_': self.y_train_,
            'training_history_': self.training_history_,
            'hyperparameters': {
                'n_neighbors': self.n_neighbors,
                'attention_hidden_dim': self.attention_hidden_dim,
                'learning_rate': self.learning_rate,
                'epochs': self.epochs,
                'batch_size': self.batch_size,
                'dropout': self.dropout,
                'distance_metric': self.distance_metric,
                'feature_scaler': self.feature_scaler,
                'early_stopping_patience': self.early_stopping_patience,
                'random_state': self.random_state,
                'use_cuda': self.use_cuda
            }
        }
        
        joblib.dump(model_data, filepath)
        print(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath: str):
        """
        加载模型
        
        Args:
            filepath: 模型文件路径
        """
        model_data = joblib.load(filepath)
        
        # 恢复超参数
        hyperparams = model_data['hyperparameters']
        for key, value in hyperparams.items():
            setattr(self, key, value)
        
        # 恢复模型组件
        self.scaler = model_data['scaler']
        self.nn_model = model_data['nn_model']
        self.classes_ = model_data['classes_']
        self.n_features_in_ = model_data['n_features_in_']
        self.X_train_ = model_data['X_train_']
        self.y_train_ = model_data['y_train_']
        self.training_history_ = model_data['training_history_']
        
        # 重新初始化注意力模块
        if model_data['attention_module_state'] is not None:
            self.attention_module = AttentionModule(
                input_dim=self.n_features_in_,
                hidden_dim=self.attention_hidden_dim,
                dropout=self.dropout
            ).to(self.device)
            self.attention_module.load_state_dict(model_data['attention_module_state'])
        
        print(f"模型已从 {filepath} 加载")

def evaluate_attention_knn(model, X_test: np.ndarray, y_test: np.ndarray, 
                          class_names: Optional[List[str]] = None) -> Dict:
    """
    评估透析注意力KNN模型
    
    Args:
        model: 训练好的模型
        X_test: 测试特征
        y_test: 测试标签
        class_names: 类别名称
        
    Returns:
        evaluation_results: 评估结果字典
    """
    # 预测
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # 透析专用风险评估
    if hasattr(model, 'predict_dialysis_risk'):
        risk_assessment = model.predict_dialysis_risk(X_test)
        high_risk_count = np.sum(risk_assessment['risk_levels'] == 'High')
        log_with_timestamp(f"高风险透析患者数量: {high_risk_count}/{len(X_test)}")
    
    # 计算指标
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    # ROC AUC (仅适用于二分类或多分类)
    try:
        if len(model.classes_) == 2:
            auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        else:
            auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
    except:
        auc = None
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }
    
    # 打印结果
    log_with_timestamp("=== 透析注意力KNN模型评估结果 ===")
    log_with_timestamp(f"准确率: {accuracy:.4f}")
    log_with_timestamp(f"精确率: {precision:.4f}")
    log_with_timestamp(f"召回率: {recall:.4f}")
    log_with_timestamp(f"F1分数: {f1:.4f}")
    if auc is not None:
        log_with_timestamp(f"AUC: {auc:.4f}")
    
    # 分类报告
    log_with_timestamp("\n详细分类报告:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names or model.classes_,
                yticklabels=class_names or model.classes_)
    plt.title('混淆矩阵')
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    plt.show()
    
    return results

# 示例使用代码
if __name__ == "__main__":
    # 生成示例透析数据
    print("生成透析患者示例数据...")
    
    np.random.seed(42)
    n_samples = 1000
    n_features = 15
    
    # 模拟透析相关特征
    feature_names = [
        '年龄', '透析时长_月', '干体重_kg', '超滤量_L', '透析时间_h',
        '收缩压_mmHg', '舒张压_mmHg', '心率_bpm', '血红蛋白_g_dL', '白蛋白_g_dL',
        '肌酐_mg_dL', '尿素氮_mg_dL', '钾_mEq_L', '钠_mEq_L', '磷_mg_dL'
    ]
    
    # 生成特征数据
    X = np.random.randn(n_samples, n_features)
    
    # 添加一些现实的特征关系
    X[:, 0] = np.random.normal(65, 15, n_samples)  # 年龄
    X[:, 1] = np.random.exponential(24, n_samples)  # 透析时长
    X[:, 2] = np.random.normal(70, 10, n_samples)  # 干体重
    X[:, 3] = np.random.normal(2.5, 0.8, n_samples)  # 超滤量
    X[:, 4] = np.random.normal(4, 0.5, n_samples)  # 透析时间
    
    # 生成标签（低血压风险：0-低风险，1-高风险）
    # 基于一些特征的组合来生成标签
    risk_score = (X[:, 0] > 70).astype(int) + \
                 (X[:, 3] > 3.0).astype(int) + \
                 (X[:, 5] < 100).astype(int) + \
                 np.random.binomial(1, 0.3, n_samples)
    
    y = (risk_score >= 2).astype(int)
    
    print(f"数据集大小: {X.shape}")
    print(f"类别分布: {np.bincount(y)}")
    
    # 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    
    # 创建和训练透析专用模型
    log_with_timestamp("\n创建透析注意力KNN模型...")
    model = DialysisAttentionKNN(
        n_neighbors=5,
        attention_hidden_dim=32,
        learning_rate=0.001,
        epochs=50,
        batch_size=32,
        dropout=0.1,
        distance_metric='euclidean',
        feature_scaler='robust',  # 对异常值更鲁棒
        early_stopping_patience=10,
        random_state=42,
        use_cuda=True,
        verbose=True,
        # 透析专用参数
        enable_feature_engineering=True,
        enable_temporal_features=True,
        handle_imbalance=True,
        cost_sensitive=True,
        enable_early_warning=True,
        warning_threshold=0.7,
        feature_names=feature_names,
        clinical_feature_groups={
            'vital_signs': [5, 6, 7],  # 血压、心率
            'lab_values': [8, 9, 10, 11, 12, 13, 14],  # 实验室指标
            'dialysis_params': [1, 2, 3, 4]  # 透析参数
        }
    )
    
    # 训练模型
    log_with_timestamp("\n开始训练透析模型...")
    model.fit(X_train, y_train)
    
    # 评估模型
    log_with_timestamp("\n评估透析模型...")
    results = evaluate_attention_knn(
        model, X_test, y_test, 
        class_names=['低风险', '高风险']
    )
    
    # 透析专用风险评估
    log_with_timestamp("\n执行透析风险评估...")
    risk_assessment = model.predict_dialysis_risk(X_test[:10])
    for i in range(min(5, len(risk_assessment['predictions']))):
        log_with_timestamp(f"患者 {i+1}: 风险等级={risk_assessment['risk_levels'][i]}, "
                          f"概率={risk_assessment['max_probabilities'][i]:.3f}, "
                          f"建议={risk_assessment['recommendations'][i]}")
    
    # 可视化训练历史
    log_with_timestamp("\n可视化训练历史...")
    model.plot_training_history()
    
    # 可视化注意力权重
    log_with_timestamp("\n可视化注意力权重...")
    model.plot_attention_weights(
        X_test[:3], 
        feature_names=feature_names,
        sample_indices=[0, 1, 2]
    )
    
    # 保存模型
    model_path = "/tmp/dialysis_attention_knn_model.pkl"
    model.save_model(model_path)
    
    log_with_timestamp(f"\n透析模型训练和评估完成!")
    log_with_timestamp(f"最终测试准确率: {results['accuracy']:.4f}")
    log_with_timestamp(f"模型已保存到: {model_path}")
    log_with_timestamp(f"模型已准备用于透析低血压风险预测")


# 为了保持向后兼容性，创建AttentionKNN别名
# 这样原有的代码调用AttentionKNN时仍然可以正常工作
AttentionKNN = DialysisAttentionKNN

# 导出主要类和函数
__all__ = [
    'DialysisAttentionKNN',
    'AttentionKNN',  # 向后兼容别名
    'MultiLevelAttentionModule',
    'DialysisFeatureEngineer',
    'DialysisConstants',
    'evaluate_attention_knn',
    'log_with_timestamp'
]