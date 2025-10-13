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
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings('ignore')


class PathManager:
    """统一的路径管理器"""
    
    def __init__(self, base_path=None):
        self.base_path = self._determine_base_path(base_path)
        
    def _determine_base_path(self, base_path):
        """确定基础路径"""
        if base_path is not None:
            return Path(base_path)
            
        # 自动检测调用脚本的目录
        import inspect
        frame = inspect.currentframe()
        try:
            caller_frame = frame.f_back.f_back
            while caller_frame:
                caller_file = caller_frame.f_code.co_filename
                # 排除工具文件，找到真正的调用脚本
                util_files = ('data_standardization.py', 'advanced_imputation.py', 'class_imbalance_handler.py', 
                             'data_process.py', 'train_untils.py', 'model_config.py')
                if not any(caller_file.endswith(uf) for uf in util_files):
                    return Path(caller_file).parent
                caller_frame = caller_frame.f_back
            # 如果没找到，使用当前工作目录
            return Path.cwd()
        finally:
            del frame
    
    def create_scaler_dirs(self, target_name='default', timestamp=None):
        """为标准化器创建目录结构"""
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
        # 创建主目录
        main_dir = self.base_path / "scalers" / f"scaler_{target_name}_{timestamp}"
        
        # 创建子目录
        subdirs = {
            'main': main_dir,
            'scalers': main_dir / 'scalers',
            'reports': main_dir / 'reports',
            'logs': main_dir / 'logs'
        }
        
        for dir_path in subdirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
            
        return subdirs


class DialysisDataStandardizer(BaseEstimator, TransformerMixin):
    """
    透析数据标准化器
    
    专门为透析数据设计的标准化器，支持：
    - 多种标准化方法
    - 特征选择性标准化
    - 标准化器的保存和加载
    - 透析数据的特殊处理
    """
    
    def __init__(self, method='standard', exclude_columns=None, save_dir=None, 
                 target_name='default', verbose=True):
        """
        初始化标准化器
        
        参数:
        - method: 标准化方法 ('standard', 'minmax', 'robust')
        - exclude_columns: 不需要标准化的列名列表
        - save_dir: 标准化器保存目录 (可选，如果为None则使用PathManager自动管理)
        - target_name: 目标变量名称，用于文件命名
        - verbose: 是否输出详细信息
        """
        self.method = method
        self.exclude_columns = exclude_columns or []
        self.target_name = target_name
        self.verbose = verbose
        self.scaler = None
        self.feature_names = None
        self.standardized_features = None
        self.is_fitted = False
        
        # 初始化路径管理器
        self.path_manager = PathManager()
        
        # 设置保存目录
        if save_dir is not None:
            self.save_dir = Path(save_dir)
            # 创建保存目录
            self.save_dir.mkdir(parents=True, exist_ok=True)
        else:
            # 使用PathManager创建目录结构
            self.dirs = self.path_manager.create_scaler_dirs(target_name)
            self.save_dir = self.dirs['scalers']
    
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
        
        save_path = self.save_dir / filename
        
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
        filepath = Path(filepath)
        if not filepath.exists():
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
                                save_dir=None, target_name='default', 
                                verbose=True):
    """
    创建透析数据标准化器的便捷函数
    
    参数:
    - method: 标准化方法 ('standard', 'minmax', 'robust')
    - exclude_columns: 不需要标准化的列名列表
    - save_dir: 标准化器保存目录 (可选，如果为None则使用PathManager自动管理)
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


def add_gaussian_noise_to_features(X_train, X_val=None, X_test=None, 
                                   noise_std=0.01, exclude_columns=None, 
                                   random_state=42, verbose=True):
    """
    为训练数据的数值特征添加高斯噪声
    
    参数:
    - X_train: 训练集特征 (DataFrame)
    - X_val: 验证集特征 (DataFrame, 可选)
    - X_test: 测试集特征 (DataFrame, 可选) - 注意：测试集通常不添加噪声
    - noise_std: 高斯噪声的标准差
    - exclude_columns: 不添加噪声的列名列表
    - random_state: 随机种子
    - verbose: 是否输出详细信息
    
    返回:
    - results: 包含添加噪声后数据的字典
    """
    if verbose:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [GaussianNoise] 开始为训练数据添加高斯噪声...")
    
    # 设置随机种子
    np.random.seed(random_state)
    
    # 获取默认排除列名
    default_exclude = get_dialysis_exclude_columns()
    if exclude_columns:
        exclude_columns = list(set(default_exclude + exclude_columns))
    else:
        exclude_columns = default_exclude
    
    # 复制数据
    X_train_noisy = X_train.copy()
    results = {'X_train_noisy': X_train_noisy}
    
    # 确定需要添加噪声的数值特征
    if isinstance(X_train, pd.DataFrame):
        # 过滤实际存在的排除列
        exclude_columns = [col for col in exclude_columns if col in X_train.columns]
        
        # 选择数值特征且不在排除列表中的特征
        numerical_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
        features_to_noise = [col for col in numerical_features if col not in exclude_columns]
        
        if verbose:
            print(f"[{timestamp}] [GaussianNoise] 总特征数: {len(X_train.columns)}")
            print(f"[{timestamp}] [GaussianNoise] 数值特征数: {len(numerical_features)}")
            print(f"[{timestamp}] [GaussianNoise] 排除特征数: {len(exclude_columns)}")
            print(f"[{timestamp}] [GaussianNoise] 添加噪声的特征数: {len(features_to_noise)}")
            print(f"[{timestamp}] [GaussianNoise] 噪声标准差: {noise_std}")
        
        # 为训练集添加高斯噪声
        if features_to_noise:
            for feature in features_to_noise:
                # 生成与特征值相同形状的高斯噪声
                noise = np.random.normal(0, noise_std, size=X_train[feature].shape)
                X_train_noisy[feature] = X_train[feature] + noise
            
            if verbose:
                print(f"[{timestamp}] [GaussianNoise] 训练集噪声添加完成")
        
        # 验证集通常也添加噪声（用于数据增强）
        if X_val is not None:
            X_val_noisy = X_val.copy()
            if features_to_noise:
                for feature in features_to_noise:
                    noise = np.random.normal(0, noise_std, size=X_val[feature].shape)
                    X_val_noisy[feature] = X_val[feature] + noise
            results['X_val_noisy'] = X_val_noisy
            if verbose:
                print(f"[{timestamp}] [GaussianNoise] 验证集噪声添加完成")
        
        # 测试集通常不添加噪声，保持原始数据用于真实评估
        if X_test is not None:
            results['X_test_original'] = X_test.copy()
            if verbose:
                print(f"[{timestamp}] [GaussianNoise] 测试集保持原始数据（未添加噪声）")
        
        # 添加噪声统计信息
        results['noise_info'] = {
            'noise_std': noise_std,
            'features_with_noise': features_to_noise,
            'excluded_features': exclude_columns,
            'random_state': random_state
        }
        
        if verbose:
            print(f"[{timestamp}] [GaussianNoise] 高斯噪声添加完成!")
    
    else:
        if verbose:
            print(f"[{timestamp}] [GaussianNoise] 警告: 输入不是DataFrame，跳过噪声添加")
    
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
    
    # 测试高斯噪声添加
    print("\n测试高斯噪声添加...")
    noise_results = add_gaussian_noise_to_features(
        X_train=results['X_train_scaled'],
        X_test=results['X_test_scaled'],
        noise_std=0.01,
        verbose=True
    )
    
    print(f"噪声信息: {noise_results['noise_info']}")
    
    # 测试加载标准化器
    X_test_scaled_loaded = apply_saved_standardizer(
        X_test, results['scaler_path'], verbose=True
    )
    
    print("\n标准化器加载测试完成!")