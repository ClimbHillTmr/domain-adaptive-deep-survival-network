# -*- coding: utf-8 -*-
"""
数据标准化模块

该模块提供了透析数据的标准化功能，包括：
1. 训练时的数据标准化和标准化器保存
2. 测试时应用已保存的标准化器
3. 支持多种标准化方法（StandardScaler、MinMaxScaler、RobustScaler）
4. 针对透析数据的特殊处理

作者: AI Assistant
日期: 2025
"""

import pandas as pd
import numpy as np
import os
import pickle
import joblib
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings('ignore')


class DialysisDataStandardizer(BaseEstimator, TransformerMixin):
    """
    透析数据标准化器
    
    专门为透析数据设计的标准化器，支持：
    - 多种标准化方法
    - 特征选择性标准化
    - 标准化器的保存和加载
    - 透析数据的特殊处理
    """
    
    def __init__(self, method='standard', exclude_columns=None, save_dir='./scalers', 
                 target_name='default', verbose=True):
        """
        初始化标准化器
        
        参数:
        - method: 标准化方法 ('standard', 'minmax', 'robust')
        - exclude_columns: 不需要标准化的列名列表
        - save_dir: 标准化器保存目录
        - target_name: 目标变量名称，用于文件命名
        - verbose: 是否输出详细信息
        """
        self.method = method
        self.exclude_columns = exclude_columns or []
        self.save_dir = save_dir
        self.target_name = target_name
        self.verbose = verbose
        self.scaler = None
        self.feature_names = None
        self.standardized_features = None
        self.is_fitted = False
        
        # 创建保存目录
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
    
    def _get_scaler(self):
        """根据方法选择标准化器"""
        if self.method == 'standard':
            return StandardScaler()
        elif self.method == 'minmax':
            return MinMaxScaler()
        elif self.method == 'robust':
            return RobustScaler()
        else:
            raise ValueError(f"不支持的标准化方法: {self.method}")
    
    def _log(self, message):
        """日志输出"""
        if self.verbose:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] [DialysisStandardizer] {message}")
    
    def fit(self, X, y=None):
        """
        训练标准化器
        
        参数:
        - X: 特征数据 (DataFrame或array)
        - y: 标签数据 (可选)
        
        返回:
        - self
        """
        # 转换为DataFrame
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        self.feature_names = list(X.columns)
        
        # 确定需要标准化的特征
        self.standardized_features = [col for col in self.feature_names 
                                    if col not in self.exclude_columns]
        
        self._log(f"开始训练{self.method}标准化器")
        self._log(f"总特征数: {len(self.feature_names)}")
        self._log(f"需要标准化的特征数: {len(self.standardized_features)}")
        self._log(f"排除的特征: {self.exclude_columns}")
        
        # 初始化标准化器
        self.scaler = self._get_scaler()
        
        # 只对需要标准化的特征进行训练
        if self.standardized_features:
            X_to_scale = X[self.standardized_features]
            
            # 检查数据质量
            self._check_data_quality(X_to_scale)
            
            # 训练标准化器
            self.scaler.fit(X_to_scale)
            
            self._log(f"标准化器训练完成")
            
            # 输出标准化统计信息
            if self.method == 'standard' and hasattr(self.scaler, 'mean_'):
                self._log(f"特征均值范围: [{self.scaler.mean_.min():.4f}, {self.scaler.mean_.max():.4f}]")
                self._log(f"特征标准差范围: [{self.scaler.scale_.min():.4f}, {self.scaler.scale_.max():.4f}]")
        
        self.is_fitted = True
        return self
    
    def transform(self, X):
        """
        应用标准化
        
        参数:
        - X: 特征数据 (DataFrame或array)
        
        返回:
        - X_scaled: 标准化后的数据 (DataFrame)
        """
        if not self.is_fitted:
            raise ValueError("标准化器尚未训练，请先调用fit方法")
        
        # 转换为DataFrame
        if isinstance(X, np.ndarray):
            if self.feature_names is None:
                raise ValueError("无法确定特征名称，请使用DataFrame输入")
            X = pd.DataFrame(X, columns=self.feature_names)
        
        # 检查特征一致性
        missing_features = set(self.standardized_features) - set(X.columns)
        if missing_features:
            raise ValueError(f"缺少特征: {missing_features}")
        
        # 复制数据
        X_scaled = X.copy()
        
        # 只对需要标准化的特征进行转换
        if self.standardized_features:
            X_to_scale = X[self.standardized_features]
            X_scaled_values = self.scaler.transform(X_to_scale)
            X_scaled[self.standardized_features] = X_scaled_values
        
        return X_scaled
    
    def fit_transform(self, X, y=None):
        """
        训练并应用标准化
        
        参数:
        - X: 特征数据
        - y: 标签数据 (可选)
        
        返回:
        - X_scaled: 标准化后的数据
        """
        return self.fit(X, y).transform(X)
    
    def inverse_transform(self, X):
        """
        逆标准化
        
        参数:
        - X: 标准化后的数据
        
        返回:
        - X_original: 逆标准化后的数据
        """
        if not self.is_fitted:
            raise ValueError("标准化器尚未训练，请先调用fit方法")
        
        # 转换为DataFrame
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names)
        
        # 复制数据
        X_original = X.copy()
        
        # 只对标准化的特征进行逆转换
        if self.standardized_features:
            X_to_inverse = X[self.standardized_features]
            X_original_values = self.scaler.inverse_transform(X_to_inverse)
            X_original[self.standardized_features] = X_original_values
        
        return X_original
    
    def _check_data_quality(self, X):
        """检查数据质量"""
        # 检查缺失值
        missing_count = X.isnull().sum().sum()
        if missing_count > 0:
            self._log(f"警告: 发现 {missing_count} 个缺失值")
        
        # 检查无穷值
        inf_count = np.isinf(X.select_dtypes(include=[np.number])).sum().sum()
        if inf_count > 0:
            self._log(f"警告: 发现 {inf_count} 个无穷值")
        
        # 检查常数特征
        constant_features = []
        for col in X.columns:
            if X[col].nunique() <= 1:
                constant_features.append(col)
        
        if constant_features:
            self._log(f"警告: 发现常数特征: {constant_features}")
    
    def save_scaler(self, filename=None):
        """
        保存标准化器
        
        参数:
        - filename: 保存文件名 (可选)
        
        返回:
        - save_path: 保存路径
        """
        if not self.is_fitted:
            raise ValueError("标准化器尚未训练，无法保存")
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"scaler_{self.method}_{self.target_name}_{timestamp}.pkl"
        
        save_path = os.path.join(self.save_dir, filename)
        
        # 保存标准化器和相关信息
        scaler_data = {
            'scaler': self.scaler,
            'method': self.method,
            'feature_names': self.feature_names,
            'standardized_features': self.standardized_features,
            'exclude_columns': self.exclude_columns,
            'target_name': self.target_name,
            'save_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(scaler_data, f)
        
        self._log(f"标准化器已保存到: {save_path}")
        return save_path
    
    def load_scaler(self, filepath):
        """
        加载标准化器
        
        参数:
        - filepath: 标准化器文件路径
        
        返回:
        - self
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"标准化器文件不存在: {filepath}")
        
        with open(filepath, 'rb') as f:
            scaler_data = pickle.load(f)
        
        self.scaler = scaler_data['scaler']
        self.method = scaler_data['method']
        self.feature_names = scaler_data['feature_names']
        self.standardized_features = scaler_data['standardized_features']
        self.exclude_columns = scaler_data['exclude_columns']
        self.target_name = scaler_data.get('target_name', 'default')
        
        self.is_fitted = True
        
        self._log(f"标准化器已从 {filepath} 加载")
        self._log(f"标准化方法: {self.method}")
        self._log(f"目标变量: {self.target_name}")
        self._log(f"保存时间: {scaler_data.get('save_time', '未知')}")
        
        return self
    
    def get_feature_statistics(self):
        """
        获取特征统计信息
        
        返回:
        - stats: 特征统计信息字典
        """
        if not self.is_fitted:
            raise ValueError("标准化器尚未训练")
        
        stats = {
            'method': self.method,
            'total_features': len(self.feature_names),
            'standardized_features': len(self.standardized_features),
            'excluded_features': len(self.exclude_columns)
        }
        
        if self.method == 'standard' and hasattr(self.scaler, 'mean_'):
            stats.update({
                'mean_range': [float(self.scaler.mean_.min()), float(self.scaler.mean_.max())],
                'scale_range': [float(self.scaler.scale_.min()), float(self.scaler.scale_.max())]
            })
        elif self.method == 'minmax' and hasattr(self.scaler, 'data_min_'):
            stats.update({
                'data_min_range': [float(self.scaler.data_min_.min()), float(self.scaler.data_min_.max())],
                'data_max_range': [float(self.scaler.data_max_.min()), float(self.scaler.data_max_.max())]
            })
        
        return stats


def create_dialysis_standardizer(method='standard', exclude_columns=None, 
                                save_dir='./scalers', target_name='default', 
                                verbose=True):
    """
    创建透析数据标准化器的便捷函数
    
    参数:
    - method: 标准化方法 ('standard', 'minmax', 'robust')
    - exclude_columns: 不需要标准化的列名列表
    - save_dir: 标准化器保存目录
    - target_name: 目标变量名称
    - verbose: 是否输出详细信息
    
    返回:
    - standardizer: DialysisDataStandardizer实例
    """
    return DialysisDataStandardizer(
        method=method,
        exclude_columns=exclude_columns,
        save_dir=save_dir,
        target_name=target_name,
        verbose=verbose
    )


def standardize_training_data(X_train, X_val=None, method='standard', 
                            exclude_columns=None, save_dir='./scalers', 
                            target_name='default', verbose=True):
    """
    标准化训练数据并保存标准化器
    
    参数:
    - X_train: 训练集特征
    - X_val: 验证集特征 (可选)
    - method: 标准化方法
    - exclude_columns: 不需要标准化的列名列表
    - save_dir: 标准化器保存目录
    - target_name: 目标变量名称
    - verbose: 是否输出详细信息
    
    返回:
    - X_train_scaled: 标准化后的训练集
    - X_val_scaled: 标准化后的验证集 (如果提供)
    - scaler_path: 标准化器保存路径
    """
    # 创建标准化器
    standardizer = create_dialysis_standardizer(
        method=method,
        exclude_columns=exclude_columns,
        save_dir=save_dir,
        target_name=target_name,
        verbose=verbose
    )
    
    # 训练标准化器
    standardizer.fit(X_train)
    
    # 标准化训练集
    X_train_scaled = standardizer.transform(X_train)
    
    # 标准化验证集（如果提供）
    X_val_scaled = None
    if X_val is not None:
        X_val_scaled = standardizer.transform(X_val)
    
    # 保存标准化器
    scaler_path = standardizer.save_scaler()
    
    return X_train_scaled, X_val_scaled, scaler_path


def apply_saved_standardizer(X_test, scaler_path, verbose=True):
    """
    应用已保存的标准化器到测试数据
    
    参数:
    - X_test: 测试集特征
    - scaler_path: 标准化器文件路径
    - verbose: 是否输出详细信息
    
    返回:
    - X_test_scaled: 标准化后的测试集
    """
    # 创建标准化器实例
    standardizer = DialysisDataStandardizer(verbose=verbose)
    
    # 加载标准化器
    standardizer.load_scaler(scaler_path)
    
    # 应用标准化
    X_test_scaled = standardizer.transform(X_test)
    
    return X_test_scaled


def get_standardizer_info(scaler_path):
    """
    获取标准化器信息
    
    参数:
    - scaler_path: 标准化器文件路径
    
    返回:
    - info: 标准化器信息字典
    """
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"标准化器文件不存在: {scaler_path}")
    
    with open(scaler_path, 'rb') as f:
        scaler_data = pickle.load(f)
    
    info = {
        'method': scaler_data['method'],
        'target_name': scaler_data.get('target_name', '未知'),
        'total_features': len(scaler_data['feature_names']),
        'standardized_features': len(scaler_data['standardized_features']),
        'excluded_features': len(scaler_data['exclude_columns']),
        'save_time': scaler_data.get('save_time', '未知'),
        'feature_names': scaler_data['feature_names'],
        'standardized_feature_names': scaler_data['standardized_features'],
        'excluded_feature_names': scaler_data['exclude_columns']
    }
    
    return info


# 透析数据特殊处理函数
def get_dialysis_exclude_columns():
    """
    获取透析数据中通常不需要标准化的列名
    
    返回:
    - exclude_columns: 不需要标准化的列名列表
    """
    exclude_columns = [
        # 分类变量
        '性别', '传染病', '抗凝剂类型', '瘘管类型', '瘘管位置', '透析方式',
        # ID和日期
        '患者id', '透析日期',
        # 已经是比例或分类的变量
        '降幅时间点比值区间', '降幅时间点差值区间', '涨幅时间点比值区间', '涨幅时间点差值区间',
        # 目标变量
        '透中低血压_计算', '透中高血压_计算',
        # 新的目标变量（时间序列预测）
        '新_透中低血压_计算', '新_降幅时间点比值区间', '新_降幅时间点差值区间'
    ]
    
    return exclude_columns


def standardize_dialysis_data(X_train, X_val=None, X_test=None, method='standard',
                            custom_exclude=None, save_dir='./scalers', 
                            target_name='dialysis', verbose=True):
    """
    透析数据的完整标准化流程
    
    参数:
    - X_train: 训练集特征
    - X_val: 验证集特征 (可选)
    - X_test: 测试集特征 (可选)
    - method: 标准化方法
    - custom_exclude: 自定义排除列名列表
    - save_dir: 标准化器保存目录
    - target_name: 目标变量名称
    - verbose: 是否输出详细信息
    
    返回:
    - results: 包含标准化结果的字典
    """
    # 获取默认排除列名
    exclude_columns = get_dialysis_exclude_columns()
    
    # 添加自定义排除列名
    if custom_exclude:
        exclude_columns.extend(custom_exclude)
    
    # 去重
    exclude_columns = list(set(exclude_columns))
    
    # 过滤实际存在的列名
    if isinstance(X_train, pd.DataFrame):
        exclude_columns = [col for col in exclude_columns if col in X_train.columns]
    
    # 创建标准化器
    standardizer = create_dialysis_standardizer(
        method=method,
        exclude_columns=exclude_columns,
        save_dir=save_dir,
        target_name=target_name,
        verbose=verbose
    )
    
    # 训练标准化器
    standardizer.fit(X_train)
    
    # 标准化数据
    results = {
        'X_train_scaled': standardizer.transform(X_train),
        'scaler_path': standardizer.save_scaler(),
        'standardizer': standardizer,
        'feature_stats': standardizer.get_feature_statistics()
    }
    
    if X_val is not None:
        results['X_val_scaled'] = standardizer.transform(X_val)
    
    if X_test is not None:
        results['X_test_scaled'] = standardizer.transform(X_test)
    
    return results


if __name__ == "__main__":
    # 测试代码
    print("透析数据标准化模块测试")
    
    # 创建测试数据
    np.random.seed(42)
    test_data = pd.DataFrame({
        '透前收缩压': np.random.normal(140, 20, 100),
        '透前舒张压': np.random.normal(80, 10, 100),
        '干体重': np.random.normal(60, 10, 100),
        '性别': np.random.choice([0, 1], 100),
        '透中低血压_计算': np.random.choice([0, 1], 100)
    })
    
    # 分割数据
    train_data = test_data[:80]
    test_data = test_data[80:]
    
    print(f"训练数据形状: {train_data.shape}")
    print(f"测试数据形状: {test_data.shape}")
    
    # 标准化训练数据
    X_train = train_data.drop('透中低血压_计算', axis=1)
    X_test = test_data.drop('透中低血压_计算', axis=1)
    
    results = standardize_dialysis_data(
        X_train=X_train,
        X_test=X_test,
        method='standard',
        target_name='test',
        verbose=True
    )
    
    print("\n标准化完成!")
    print(f"标准化器保存路径: {results['scaler_path']}")
    print(f"特征统计信息: {results['feature_stats']}")
    
    # 测试加载标准化器
    X_test_scaled_loaded = apply_saved_standardizer(
        X_test, results['scaler_path'], verbose=True
    )
    
    print("\n标准化器加载测试完成!")