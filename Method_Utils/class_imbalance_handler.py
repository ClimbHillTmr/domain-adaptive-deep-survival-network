"""类别不平衡处理模块

该模块提供了多种类别不平衡处理策略，包括：
- SMOTE（合成少数类过采样技术）
- SMOTEENN（SMOTE + 编辑最近邻）
- SMOTETomek（SMOTE + Tomek链接）
- ADASYN（自适应合成采样）
- 随机欠采样

支持自动检测类别分布并应用合适的重采样策略。

作者: AI Assistant
日期: 2024
"""

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.under_sampling import RandomUnderSampler
import warnings
warnings.filterwarnings('ignore')


class ClassImbalanceHandler:
    """类别不平衡处理器"""
    
    def __init__(self, method='smote', random_state=42, sampling_strategy='auto'):
        """
        初始化类别不平衡处理器
        
        Parameters:
        -----------
        method : str, default='smote'
            重采样方法: 'smote', 'adasyn', 'smoteenn', 'smotetomek', 'undersample'
        random_state : int, default=42
            随机种子
        sampling_strategy : str or dict, default='auto'
            采样策略
        """
        self.method = method
        self.random_state = random_state
        self.sampling_strategy = sampling_strategy
        self.sampler = None
        self._initialize_sampler()
    
    def _initialize_sampler(self):
        """初始化采样器"""
        if self.method == 'smote':
            self.sampler = SMOTE(
                random_state=self.random_state,
                sampling_strategy=self.sampling_strategy
            )
        elif self.method == 'adasyn':
            self.sampler = ADASYN(
                random_state=self.random_state,
                sampling_strategy=self.sampling_strategy
            )
        elif self.method == 'smoteenn':
            self.sampler = SMOTEENN(
                random_state=self.random_state,
                sampling_strategy=self.sampling_strategy
            )
        elif self.method == 'smotetomek':
            self.sampler = SMOTETomek(
                random_state=self.random_state,
                sampling_strategy=self.sampling_strategy
            )
        elif self.method == 'undersample':
            self.sampler = RandomUnderSampler(
                random_state=self.random_state,
                sampling_strategy=self.sampling_strategy
            )
        else:
            raise ValueError(f"未知的重采样方法: {self.method}")
    
    def analyze_class_distribution(self, y):
        """分析类别分布"""
        class_counts = pd.Series(y).value_counts().sort_index()
        total_samples = len(y)
        
        print(f"\n=== 类别分布分析 ===")
        print(f"总样本数: {total_samples}")
        print(f"类别数量: {len(class_counts)}")
        
        for class_label, count in class_counts.items():
            percentage = count / total_samples * 100
            print(f"类别 {class_label}: {count} 样本 ({percentage:.1f}%)")
        
        # 计算不平衡比率
        max_count = class_counts.max()
        min_count = class_counts.min()
        imbalance_ratio = max_count / min_count
        
        print(f"不平衡比率: {imbalance_ratio:.2f}:1")
        
        if imbalance_ratio > 2:
            print("⚠️  检测到类别不平衡问题，建议使用重采样技术")
        else:
            print("✅ 类别分布相对平衡")
        
        return {
            'class_counts': class_counts.to_dict(),
            'total_samples': total_samples,
            'imbalance_ratio': imbalance_ratio,
            'is_imbalanced': imbalance_ratio > 2
        }
    
    def fit_resample(self, X, y, verbose=True):
        """执行重采样"""
        if verbose:
            print(f"\n=== 类别不平衡处理 ===")
            print(f"使用方法: {self.method.upper()}")
            
            # 分析原始分布
            original_dist = self.analyze_class_distribution(y)
        
        try:
            # 执行重采样
            X_resampled, y_resampled = self.sampler.fit_resample(X, y)
            
            if verbose:
                print(f"\n重采样完成!")
                print(f"原始样本数: {len(X)} -> 重采样后: {len(X_resampled)}")
                
                # 分析重采样后的分布
                print(f"\n=== 重采样后类别分布 ===")
                resampled_dist = self.analyze_class_distribution(y_resampled)
            
            return X_resampled, y_resampled
            
        except Exception as e:
            print(f"❌ 重采样失败: {e}")
            print(f"返回原始数据")
            return X, y
    
    def get_recommended_method(self, X, y):
        """根据数据特征推荐重采样方法"""
        dist_info = self.analyze_class_distribution(y)
        n_samples, n_features = X.shape
        
        print(f"\n=== 重采样方法推荐 ===")
        print(f"数据维度: {n_samples} 样本, {n_features} 特征")
        
        if not dist_info['is_imbalanced']:
            recommendation = "none"
            reason = "类别分布相对平衡，无需重采样"
        elif n_samples < 1000:
            recommendation = "smote"
            reason = "小数据集，推荐使用SMOTE"
        elif dist_info['imbalance_ratio'] > 10:
            recommendation = "smotetomek"
            reason = "严重不平衡，推荐使用SMOTETomek"
        elif n_features > 50:
            recommendation = "smoteenn"
            reason = "高维数据，推荐使用SMOTEENN"
        else:
            recommendation = "smote"
            reason = "一般情况，推荐使用SMOTE"
        
        print(f"推荐方法: {recommendation.upper()}")
        print(f"推荐理由: {reason}")
        
        return recommendation


def data_resampling(X, y, method='smote', random_state=42, verbose=True):
    """
    数据重采样函数（兼容性函数）
    
    Parameters:
    -----------
    X : array-like
        特征矩阵
    y : array-like
        目标变量
    method : str, default='smote'
        重采样方法
    random_state : int, default=42
        随机种子
    verbose : bool, default=True
        是否打印详细信息
    
    Returns:
    --------
    X_resampled, y_resampled : 重采样后的数据
    """
    if method == 'none' or method is None:
        if verbose:
            print("跳过重采样")
        return X, y
    
    handler = ClassImbalanceHandler(method=method, random_state=random_state)
    return handler.fit_resample(X, y, verbose=verbose)


class OptimizedModelPipeline:
    """优化的模型训练管道"""
    
    def __init__(self, imputation_strategy='iterative', test_size=0.2, val_size=0.125, random_state=42):
        """
        初始化模型训练管道
        
        Parameters:
        -----------
        imputation_strategy : str, default='iterative'
            缺失值填补策略
        test_size : float, default=0.2
            测试集比例
        val_size : float, default=0.125
            验证集比例
        random_state : int, default=42
            随机种子
        """
        self.imputation_strategy = imputation_strategy
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        
        # 导入缺失值填补模块
        from .advanced_imputation import AdvancedImputationPipeline
        self.imputer = AdvancedImputationPipeline(strategy=imputation_strategy)
    
    def prepare_data(self, X, y, X_external_test=None, y_external_test=None):
        """准备训练数据"""
        from sklearn.model_selection import train_test_split
        
        print(f"\n=== 数据准备阶段 ===")
        print(f"原始数据形状: {X.shape}")
        print(f"目标变量分布: {pd.Series(y).value_counts().to_dict()}")
        
        # 检查缺失值情况
        missing_info = X.isnull().sum()
        missing_cols = missing_info[missing_info > 0]
        if len(missing_cols) > 0:
            print(f"\n缺失值情况:")
            for col, count in missing_cols.items():
                print(f"  {col}: {count} ({count/len(X)*100:.1f}%)")
        
        # 应用高级缺失值填补
        print(f"\n使用 {self.imputation_strategy} 策略进行缺失值填补...")
        X_imputed = self.imputer.fit_transform(X)
        
        # 处理外部测试集
        if X_external_test is not None:
            print(f"处理外部测试集...")
            X_external_test_imputed = self.imputer.transform(X_external_test)
        else:
            X_external_test_imputed = None
            
        # 数据分割
        if X_external_test is not None:
            # 如果有外部测试集，只分割训练集和验证集
            X_train, X_val, y_train, y_val = train_test_split(
                X_imputed, y, test_size=self.val_size, 
                random_state=self.random_state, stratify=y
            )
            X_test, y_test = X_external_test_imputed, y_external_test
        else:
            # 标准三分割
            X_train, X_test, y_train, y_test = train_test_split(
                X_imputed, y, test_size=self.test_size, 
                random_state=self.random_state, stratify=y
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=self.val_size, 
                random_state=self.random_state, stratify=y_train
            )
        
        print(f"\n=== 数据分割结果 ===")
        print(f"训练集: {X_train.shape[0]} 样本")
        print(f"验证集: {X_val.shape[0]} 样本")
        print(f"测试集: {X_test.shape[0]} 样本")
        
        # 打印各集合的类别分布
        for name, y_subset in [('训练集', y_train), ('验证集', y_val), ('测试集', y_test)]:
            dist = pd.Series(y_subset).value_counts().sort_index()
            print(f"{name}类别分布: {dist.to_dict()}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def handle_class_imbalance(self, X_train, y_train, method='smote'):
        """处理类别不平衡"""
        handler = ClassImbalanceHandler(method=method, random_state=self.random_state)
        return handler.fit_resample(X_train, y_train)