"""高级缺失值填补管道模块

该模块提供了多种高级缺失值填补策略，包括：
- 迭代填补（IterativeImputer）
- KNN填补（KNNImputer）
- 混合填补策略

支持自动识别数值型和分类型特征，并采用不同的填补策略。

作者: AI Assistant
日期: 2024
"""

import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')


class AdvancedImputationPipeline:
    """
    简化的缺失值填补管道 - 提高可重复性
    
    支持策略:
    - 'simple': 简单填补（数值型用中位数，分类型用众数）
    - 'knn': K近邻填补（统一策略）
    - 'iterative': 迭代填补（仅在必要时使用）
    """
    
    def __init__(self, strategy='simple', n_neighbors=5, max_iter=3, random_state=42):
        self.strategy = strategy
        self.n_neighbors = n_neighbors
        self.max_iter = max_iter
        self.random_state = random_state
        self.numerical_imputer = None
        self.categorical_imputer = None
        self.label_encoders = {}
        self.numerical_cols = []
        self.categorical_cols = []
        self.simple_imputers = {}  # 存储简单填补器
        
    def _identify_column_types(self, X):
        """识别数值型和分类型特征"""
        numerical_cols = []
        categorical_cols = []
        
        for col in X.columns:
            if X[col].dtype in ['int64', 'float64']:
                # 检查是否为分类变量（唯一值较少）
                unique_ratio = X[col].nunique() / len(X)
                if unique_ratio < 0.05 and X[col].nunique() < 20:
                    categorical_cols.append(col)
                else:
                    numerical_cols.append(col)
            else:
                categorical_cols.append(col)
                
        return numerical_cols, categorical_cols
    
    def fit(self, X):
        """训练填补器 - 简化版本提高可重复性"""
        X = X.copy()
        self.numerical_cols, self.categorical_cols = self._identify_column_types(X)
        
        print(f"数值型特征: {len(self.numerical_cols)}个")
        print(f"分类型特征: {len(self.categorical_cols)}个")
        print(f"填补策略: {self.strategy}")
        
        # 处理分类变量的标签编码
        for col in self.categorical_cols:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                mask = X[col].notna()
                if mask.sum() > 0:
                    le.fit(X.loc[mask, col].astype(str))
                    self.label_encoders[col] = le
                    X.loc[mask, col] = le.transform(X.loc[mask, col].astype(str))
        
        # 根据策略选择填补方法
        if self.strategy == 'simple':
            # 简单填补策略 - 最高可重复性
            from sklearn.impute import SimpleImputer
            
            if self.numerical_cols:
                self.numerical_imputer = SimpleImputer(
                    strategy='median', 
                    add_indicator=False
                )
                self.numerical_imputer.fit(X[self.numerical_cols])
            
            if self.categorical_cols:
                # 对分类变量使用众数填补
                categorical_data = X[self.categorical_cols].copy()
                for col in self.categorical_cols:
                    if categorical_data[col].dtype == 'object':
                        categorical_data[col] = pd.to_numeric(categorical_data[col], errors='coerce')
                
                self.categorical_imputer = SimpleImputer(
                    strategy='most_frequent',
                    add_indicator=False
                )
                self.categorical_imputer.fit(categorical_data)
                
        elif self.strategy == 'knn':
            # KNN填补 - 统一策略
            if self.numerical_cols:
                self.numerical_imputer = KNNImputer(
                    n_neighbors=self.n_neighbors,
                    weights='uniform'  # 使用uniform权重提高可重复性
                )
                self.numerical_imputer.fit(X[self.numerical_cols])
            
            if self.categorical_cols:
                categorical_data = X[self.categorical_cols].copy()
                for col in self.categorical_cols:
                    if categorical_data[col].dtype == 'object':
                        categorical_data[col] = pd.to_numeric(categorical_data[col], errors='coerce')
                
                self.categorical_imputer = KNNImputer(
                    n_neighbors=min(self.n_neighbors, len(X)//2),
                    weights='uniform'
                )
                self.categorical_imputer.fit(categorical_data)
                
        elif self.strategy == 'iterative':
            # 迭代填补 - 仅在必要时使用
            if self.numerical_cols:
                self.numerical_imputer = IterativeImputer(
                    estimator=RandomForestRegressor(
                        n_estimators=5,  # 减少树的数量提高速度
                        random_state=self.random_state,
                        n_jobs=1  # 单线程确保可重复性
                    ),
                    max_iter=self.max_iter,
                    random_state=self.random_state
                )
                self.numerical_imputer.fit(X[self.numerical_cols])
            
            if self.categorical_cols:
                # 分类变量仍使用简单策略
                categorical_data = X[self.categorical_cols].copy()
                for col in self.categorical_cols:
                    if categorical_data[col].dtype == 'object':
                        categorical_data[col] = pd.to_numeric(categorical_data[col], errors='coerce')
                
                from sklearn.impute import SimpleImputer
                self.categorical_imputer = SimpleImputer(
                    strategy='most_frequent',
                    add_indicator=False
                )
                self.categorical_imputer.fit(categorical_data)
        
        return self
    
    def transform(self, X):
        """应用填补 - 简化版本确保一致性"""
        X = X.copy()
        
        # 处理分类变量的标签编码
        for col in self.categorical_cols:
            if col in self.label_encoders and X[col].dtype == 'object':
                le = self.label_encoders[col]
                mask = X[col].notna()
                if mask.sum() > 0:
                    # 安全处理未见过的类别
                    known_classes = set(le.classes_)
                    X.loc[mask, col] = X.loc[mask, col].astype(str)
                    unknown_mask = ~X.loc[mask, col].isin(known_classes)
                    if unknown_mask.sum() > 0:
                        # 将未知类别设为最频繁的类别
                        most_frequent = le.classes_[0]
                        X.loc[mask, col] = X.loc[mask, col].where(~unknown_mask, most_frequent)
                    X.loc[mask, col] = le.transform(X.loc[mask, col])
        
        # 填补数值型特征
        if self.numerical_cols and self.numerical_imputer is not None:
            try:
                X_num_imputed = self.numerical_imputer.transform(X[self.numerical_cols])
                X[self.numerical_cols] = X_num_imputed
            except Exception as e:
                print(f"数值型特征填补失败: {e}，使用中位数填补")
                # 备用方案：使用中位数填补
                for col in self.numerical_cols:
                    if X[col].isnull().any():
                        median_val = X[col].median()
                        X[col].fillna(median_val, inplace=True)
        
        # 填补分类型特征
        if self.categorical_cols and self.categorical_imputer is not None:
            try:
                categorical_data = X[self.categorical_cols].copy()
                for col in self.categorical_cols:
                    if categorical_data[col].dtype == 'object':
                        categorical_data[col] = pd.to_numeric(categorical_data[col], errors='coerce')
                
                X_cat_imputed = self.categorical_imputer.transform(categorical_data)
                X[self.categorical_cols] = X_cat_imputed
                
                # 将分类变量转换回整数
                for col in self.categorical_cols:
                    X[col] = X[col].round().astype('int64')
            except Exception as e:
                print(f"分类型特征填补失败: {e}，使用众数填补")
                # 备用方案：使用众数填补
                for col in self.categorical_cols:
                    if X[col].isnull().any():
                        mode_val = X[col].mode()
                        if len(mode_val) > 0:
                            X[col].fillna(mode_val[0], inplace=True)
                        else:
                            X[col].fillna(0, inplace=True)  # 如果没有众数，填充0
        
        return X
    
    def fit_transform(self, X):
        """训练并应用填补"""
        return self.fit(X).transform(X)
    
    def get_feature_info(self):
        """获取特征信息"""
        return {
            'numerical_features': self.numerical_cols,
            'categorical_features': self.categorical_cols,
            'label_encoders': list(self.label_encoders.keys()),
            'strategy': self.strategy
        }
    
    def quick_impute(self, X):
        """快速填补 - 最高可重复性的简单策略"""
        X = X.copy()
        
        # 数值型特征用中位数填补
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if X[col].isnull().any():
                median_val = X[col].median()
                X[col].fillna(median_val, inplace=True)
        
        # 分类型特征用众数填补
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns
        for col in categorical_cols:
            if X[col].isnull().any():
                mode_val = X[col].mode()
                if len(mode_val) > 0:
                    X[col].fillna(mode_val[0], inplace=True)
                else:
                    X[col].fillna('unknown', inplace=True)
        
        return X


def create_reproducible_imputer(strategy='simple', random_state=42):
    """创建可重复的填补器工厂函数"""
    return AdvancedImputationPipeline(
        strategy=strategy,
        n_neighbors=5,
        max_iter=3,
        random_state=random_state
    )