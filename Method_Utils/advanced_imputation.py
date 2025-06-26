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
    """高级缺失值填补管道"""
    
    def __init__(self, strategy='iterative', n_neighbors=5, max_iter=10):
        """
        初始化缺失值填补管道
        
        Parameters:
        -----------
        strategy : str, default='iterative'
            填补策略: 'iterative', 'knn', 'mixed'
        n_neighbors : int, default=5
            KNN填补的邻居数量
        max_iter : int, default=10
            迭代填补的最大迭代次数
        """
        self.strategy = strategy
        self.n_neighbors = n_neighbors
        self.max_iter = max_iter
        self.numerical_imputer = None
        self.categorical_imputer = None
        self.label_encoders = {}
        
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
        """训练填补器"""
        X = X.copy()
        self.numerical_cols, self.categorical_cols = self._identify_column_types(X)
        
        print(f"数值型特征: {len(self.numerical_cols)}个")
        print(f"分类型特征: {len(self.categorical_cols)}个")
        
        # 处理分类变量的标签编码
        for col in self.categorical_cols:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                # 处理缺失值
                mask = X[col].notna()
                if mask.sum() > 0:
                    le.fit(X.loc[mask, col].astype(str))
                    self.label_encoders[col] = le
                    X.loc[mask, col] = le.transform(X.loc[mask, col].astype(str))
        
        # 训练数值型特征的填补器
        if self.numerical_cols:
            if self.strategy == 'iterative':
                self.numerical_imputer = IterativeImputer(
                    estimator=RandomForestRegressor(n_estimators=10, random_state=42),
                    max_iter=self.max_iter,
                    random_state=42
                )
            elif self.strategy == 'knn':
                self.numerical_imputer = KNNImputer(n_neighbors=self.n_neighbors)
            elif self.strategy == 'mixed':
                # 对于数值型特征使用迭代填补
                self.numerical_imputer = IterativeImputer(
                    estimator=RandomForestRegressor(n_estimators=10, random_state=42),
                    max_iter=self.max_iter,
                    random_state=42
                )
            
            if len(self.numerical_cols) > 0:
                self.numerical_imputer.fit(X[self.numerical_cols])
        
        # 训练分类型特征的填补器（使用KNN）
        if self.categorical_cols:
            # 将分类变量转换为数值型进行KNN填补
            categorical_data = X[self.categorical_cols].copy()
            for col in self.categorical_cols:
                if categorical_data[col].dtype == 'object':
                    categorical_data[col] = pd.to_numeric(categorical_data[col], errors='coerce')
            
            self.categorical_imputer = KNNImputer(n_neighbors=min(self.n_neighbors, len(X)//2))
            if len(self.categorical_cols) > 0:
                self.categorical_imputer.fit(categorical_data)
        
        return self
    
    def transform(self, X):
        """应用填补"""
        X = X.copy()
        
        # 处理分类变量的标签编码
        for col in self.categorical_cols:
            if col in self.label_encoders and X[col].dtype == 'object':
                le = self.label_encoders[col]
                mask = X[col].notna()
                if mask.sum() > 0:
                    # 处理未见过的类别
                    known_classes = set(le.classes_)
                    X.loc[mask, col] = X.loc[mask, col].astype(str)
                    unknown_mask = ~X.loc[mask, col].isin(known_classes)
                    if unknown_mask.sum() > 0:
                        # 将未知类别设为最频繁的类别
                        most_frequent = le.classes_[0]
                        X.loc[mask & unknown_mask, col] = most_frequent
                    X.loc[mask, col] = le.transform(X.loc[mask, col])
        
        # 填补数值型特征
        if self.numerical_cols and self.numerical_imputer is not None:
            X_num_imputed = self.numerical_imputer.transform(X[self.numerical_cols])
            X[self.numerical_cols] = X_num_imputed
        
        # 填补分类型特征
        if self.categorical_cols and self.categorical_imputer is not None:
            categorical_data = X[self.categorical_cols].copy()
            for col in self.categorical_cols:
                if categorical_data[col].dtype == 'object':
                    categorical_data[col] = pd.to_numeric(categorical_data[col], errors='coerce')
            
            X_cat_imputed = self.categorical_imputer.transform(categorical_data)
            X[self.categorical_cols] = X_cat_imputed
            
            # 将分类变量转换回整数
            for col in self.categorical_cols:
                X[col] = X[col].round().astype('int64')
        
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