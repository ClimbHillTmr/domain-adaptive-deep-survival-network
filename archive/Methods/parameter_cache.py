#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
参数缓存和动态并行配置模块
用于优化透析低血压预测模型的训练性能

作者: 透析数据分析团队
日期: 2024
"""

import os
import json
import hashlib
import multiprocessing
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


@dataclass
class CachedResult:
    """
    缓存结果数据类
    """
    best_params: Dict[str, Any]
    best_score: float
    model_type: str
    data_hash: str
    timestamp: str
    optimization_method: str = 'bayesian'
    n_trials: int = None
    feature_count: int = None
    sample_count: int = None


class ParameterCache:
    """
    参数缓存管理器
    用于缓存和复用超参数优化结果
    """
    
    def __init__(self, cache_dir: str = None, max_cache_size: int = 50):
        """
        初始化参数缓存
        
        Args:
            cache_dir: 缓存目录路径
            max_cache_size: 最大缓存条目数
        """
        if cache_dir is None:
            cache_dir = os.path.join(os.path.expanduser('~'), '.dialysis_model_cache')
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_cache_size = max_cache_size
        self.cache_file = self.cache_dir / 'parameter_cache.json'
        self.cache_data = self._load_cache()
    
    def _load_cache(self) -> Dict[str, Dict]:
        """
        加载缓存数据
        
        Returns:
            dict: 缓存数据字典
        """
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"警告: 加载缓存失败: {e}")
        
        return {}
    
    def _save_cache(self):
        """
        保存缓存数据
        """
        try:
            # 限制缓存大小
            if len(self.cache_data) > self.max_cache_size:
                # 删除最旧的条目
                sorted_items = sorted(self.cache_data.items(), 
                                     key=lambda x: x[1].get('timestamp', ''),
                                     reverse=True)
                self.cache_data = dict(sorted_items[:self.max_cache_size])
            
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"警告: 保存缓存失败: {e}")
    
    def get_cache_key(self, model_type: str, X, y, param_space: Dict, 
                      optimization_method: str = 'bayesian') -> str:
        """
        生成缓存键
        
        Args:
            model_type: 模型类型
            X: 特征数据
            y: 标签数据
            param_space: 参数空间
            optimization_method: 优化方法
        
        Returns:
            str: 缓存键
        """
        try:
            import numpy as np
            
            # 计算数据哈希
            if hasattr(X, 'values'):
                X_array = X.values
            else:
                X_array = np.array(X)
            
            if hasattr(y, 'values'):
                y_array = y.values
            else:
                y_array = np.array(y)
            
            data_str = f"{X_array.shape}_{np.mean(X_array):.6f}_{np.std(X_array):.6f}_{np.mean(y_array):.6f}"
            
            # 处理参数空间，转换skopt对象为可序列化的格式
            serializable_param_space = self._make_param_space_serializable(param_space)
            param_str = json.dumps(serializable_param_space, sort_keys=True)
            
            combined_str = f"{model_type}_{data_str}_{param_str}_{optimization_method}"
            
            return hashlib.md5(combined_str.encode()).hexdigest()
        except Exception as e:
            print(f"警告: 生成缓存键失败: {e}")
            return f"{model_type}_{optimization_method}_{hash(str(param_space))}"
    
    def _make_param_space_serializable(self, param_space: Dict) -> Dict:
        """
        将参数空间转换为可序列化的格式
        
        Args:
            param_space: 原始参数空间
            
        Returns:
            dict: 可序列化的参数空间
        """
        serializable = {}
        
        for key, value in param_space.items():
            try:
                # 检查是否是skopt对象
                if hasattr(value, '__class__') and hasattr(value, 'bounds'):
                    # 处理skopt的Real, Integer等对象
                    if hasattr(value, 'low') and hasattr(value, 'high'):
                        serializable[key] = {
                            'type': value.__class__.__name__,
                            'low': float(value.low) if hasattr(value, 'low') else None,
                            'high': float(value.high) if hasattr(value, 'high') else None,
                            'prior': getattr(value, 'prior', None)
                        }
                    else:
                        serializable[key] = str(value)
                else:
                    # 普通值直接使用
                    serializable[key] = value
            except Exception:
                # 如果转换失败，使用字符串表示
                serializable[key] = str(value)
        
        return serializable
    
    def check_cache(self, cache_key: str) -> Optional[CachedResult]:
        """
        检查缓存中是否存在结果
        
        Args:
            cache_key: 缓存键
        
        Returns:
            CachedResult or None: 缓存的结果
        """
        if cache_key in self.cache_data:
            try:
                cached_data = self.cache_data[cache_key]
                return CachedResult(**cached_data)
            except Exception as e:
                print(f"警告: 解析缓存数据失败: {e}")
                # 删除损坏的缓存条目
                del self.cache_data[cache_key]
                self._save_cache()
        
        return None
    
    def update_cache(self, cache_key: str, result: CachedResult):
        """
        更新缓存
        
        Args:
            cache_key: 缓存键
            result: 缓存结果
        """
        try:
            self.cache_data[cache_key] = asdict(result)
            self._save_cache()
        except Exception as e:
            print(f"警告: 更新缓存失败: {e}")
    
    def clear_cache(self):
        """
        清空缓存
        """
        self.cache_data = {}
        self._save_cache()
        print("缓存已清空")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            dict: 缓存统计信息
        """
        stats = {
            'total_entries': len(self.cache_data),
            'model_types': {},
            'optimization_methods': {}
        }
        
        for entry in self.cache_data.values():
            model_type = entry.get('model_type', 'unknown')
            opt_method = entry.get('optimization_method', 'unknown')
            
            stats['model_types'][model_type] = stats['model_types'].get(model_type, 0) + 1
            stats['optimization_methods'][opt_method] = stats['optimization_methods'].get(opt_method, 0) + 1
        
        return stats


class LayeredOptimizationStrategy:
    """
    分层优化策略类
    提供粗粒度和细粒度的参数空间定义，支持多阶段优化
    """
    
    def __init__(self):
        """
        初始化分层优化策略
        """
        pass
    
    def get_coarse_param_space(self, model_type: str) -> Dict[str, Any]:
        """
        获取粗粒度参数空间（用于初步搜索）
        
        Args:
            model_type: 模型类型
        
        Returns:
            dict: 粗粒度参数空间
        """
        try:
            from skopt.space import Real, Integer, Categorical
        except ImportError:
            # 如果skopt不可用，返回基础参数空间
            return self._get_basic_param_space(model_type)
        
        coarse_spaces = {
            'lightgbm': {
                'n_estimators': Integer(50, 200),
                'learning_rate': Real(0.05, 0.3, prior='log-uniform'),
                'max_depth': Integer(3, 8),
                'num_leaves': Integer(20, 100),
                'min_child_samples': Integer(10, 50)
            },
            'xgboost': {
                'n_estimators': Integer(50, 200),
                'learning_rate': Real(0.05, 0.3, prior='log-uniform'),
                'max_depth': Integer(3, 8),
                'min_child_weight': Integer(1, 10),
                'subsample': Real(0.6, 1.0)
            },
            'svm': {
                'C': Real(0.1, 100, prior='log-uniform'),
                'gamma': Real(1e-4, 1e-1, prior='log-uniform'),
                'kernel': Categorical(['rbf', 'poly'])
            },
            'random_forest': {
                'n_estimators': Integer(50, 200),
                'max_depth': Integer(5, 20),
                'min_samples_split': Integer(2, 20),
                'min_samples_leaf': Integer(1, 10)
            }
        }
        
        return coarse_spaces.get(model_type.lower(), {})
    
    def get_fine_param_space(self, model_type: str, best_coarse_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        获取细粒度参数空间（基于粗粒度搜索结果）
        
        Args:
            model_type: 模型类型
            best_coarse_params: 粗粒度搜索的最佳参数
        
        Returns:
            dict: 细粒度参数空间
        """
        try:
            from skopt.space import Real, Integer, Categorical
        except ImportError:
            return best_coarse_params
        
        fine_space = {}
        
        for param, value in best_coarse_params.items():
            if isinstance(value, (int, float)):
                if param in ['learning_rate', 'gamma', 'C']:
                    # 对数空间参数
                    range_factor = 0.3
                    low = max(value * (1 - range_factor), 1e-6)
                    high = value * (1 + range_factor)
                    fine_space[param] = Real(low, high, prior='log-uniform')
                elif param in ['n_estimators', 'max_depth', 'num_leaves']:
                    # 整数参数
                    range_factor = 0.2
                    low = max(int(value * (1 - range_factor)), 1)
                    high = int(value * (1 + range_factor))
                    fine_space[param] = Integer(low, high)
                else:
                    # 其他数值参数
                    range_factor = 0.2
                    low = max(value * (1 - range_factor), 0.01)
                    high = value * (1 + range_factor)
                    fine_space[param] = Real(low, high)
            else:
                # 分类参数保持不变
                fine_space[param] = value
        
        return fine_space
    
    def _get_basic_param_space(self, model_type: str) -> Dict[str, Any]:
        """
        获取基础参数空间（当skopt不可用时）
        
        Args:
            model_type: 模型类型
        
        Returns:
            dict: 基础参数空间
        """
        basic_spaces = {
            'lightgbm': {
                'n_estimators': [50, 100, 150, 200],
                'learning_rate': [0.05, 0.1, 0.15, 0.2],
                'max_depth': [3, 5, 7, 9],
                'num_leaves': [31, 50, 70, 100]
            },
            'xgboost': {
                'n_estimators': [50, 100, 150, 200],
                'learning_rate': [0.05, 0.1, 0.15, 0.2],
                'max_depth': [3, 5, 7, 9],
                'min_child_weight': [1, 3, 5, 7]
            },
            'svm': {
                'C': [0.1, 1, 10, 100],
                'gamma': [0.001, 0.01, 0.1, 1],
                'kernel': ['rbf', 'poly']
            },
            'random_forest': {
                'n_estimators': [50, 100, 150, 200],
                'max_depth': [5, 10, 15, 20],
                'min_samples_split': [2, 5, 10, 15]
            }
        }
        
        return basic_spaces.get(model_type.lower(), {})


class DynamicParallelConfig:
    """
    动态并行配置管理器
    根据数据规模和系统资源动态配置并行策略
    """
    
    def __init__(self):
        """
        初始化动态并行配置
        """
        self.cpu_count = multiprocessing.cpu_count()
        self.available_memory_gb = self._get_available_memory_gb()
    
    def _get_available_memory_gb(self) -> float:
        """
        获取可用内存（GB）
        
        Returns:
            float: 可用内存大小
        """
        try:
            import psutil
            return psutil.virtual_memory().available / (1024**3)
        except ImportError:
            # 如果psutil不可用，返回保守估计
            return 4.0
        except Exception:
            return 4.0
    
    def get_optimal_config(self, n_samples: int, n_features: int, 
                          model_type: str = 'general') -> Dict[str, int]:
        """
        获取最优并行配置
        
        Args:
            n_samples: 样本数量
            n_features: 特征数量
            model_type: 模型类型
        
        Returns:
            dict: 并行配置字典
        """
        # 计算数据复杂度
        data_complexity = n_samples * n_features
        memory_per_core = self.available_memory_gb / self.cpu_count
        
        # 基础配置
        config = {
            'n_jobs': 1,
            'torch_threads': 1
        }
        
        # 根据数据规模调整
        if data_complexity < 1e6:  # 小数据集
            config['n_jobs'] = min(2, self.cpu_count)
        elif data_complexity < 1e7:  # 中等数据集
            config['n_jobs'] = min(4, self.cpu_count)
        else:  # 大数据集
            config['n_jobs'] = min(6, self.cpu_count)
        
        # 根据模型类型调整
        model_adjustments = {
            'attention_knn': {'n_jobs': 1, 'torch_threads': min(4, self.cpu_count)},
            'svm': {'n_jobs': min(8, self.cpu_count)},
            'iedt': {'n_jobs': min(6, self.cpu_count)},
            'lightgbm': {'n_jobs': min(8, self.cpu_count)},
            'tabnet': {'n_jobs': 1, 'torch_threads': min(4, self.cpu_count)}
        }
        
        if model_type.lower() in model_adjustments:
            config.update(model_adjustments[model_type.lower()])
        
        # 内存限制调整
        if memory_per_core < 1.0:  # 内存紧张
            config = {k: max(1, v // 2) for k, v in config.items()}
        
        # 确保不超过CPU核心数
        config['n_jobs'] = min(config['n_jobs'], self.cpu_count)
        
        return config
    
    def apply_config(self, config: Dict[str, int]):
        """
        应用并行配置
        
        Args:
            config: 并行配置字典
        """
        # 设置环境变量
        if 'torch_threads' in config:
            os.environ['OMP_NUM_THREADS'] = str(config['torch_threads'])
            os.environ['MKL_NUM_THREADS'] = str(config['torch_threads'])
            
            # 设置PyTorch线程数
            try:
                import torch
                torch.set_num_threads(config['torch_threads'])
            except ImportError:
                pass
        
        # 设置其他并行相关环境变量
        os.environ['NUMBA_NUM_THREADS'] = str(config.get('n_jobs', 1))
        
        print(f"已应用并行配置: {config}")
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        获取系统信息
        
        Returns:
            dict: 系统信息
        """
        info = {
            'cpu_count': self.cpu_count,
            'available_memory_gb': self.available_memory_gb
        }
        
        try:
            import psutil
            info.update({
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent
            })
        except ImportError:
            pass
        
        return info


# 全局实例
_global_cache = None
_global_parallel_config = None
_global_layered_strategy = None


def get_global_cache() -> ParameterCache:
    """
    获取全局参数缓存实例
    
    Returns:
        ParameterCache: 全局缓存实例
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = ParameterCache()
    return _global_cache


def get_global_parallel_config() -> DynamicParallelConfig:
    """
    获取全局并行配置实例
    
    Returns:
        DynamicParallelConfig: 全局并行配置实例
    """
    global _global_parallel_config
    if _global_parallel_config is None:
        _global_parallel_config = DynamicParallelConfig()
    return _global_parallel_config


def get_global_layered_strategy() -> LayeredOptimizationStrategy:
    """
    获取全局分层优化策略实例
    
    Returns:
        LayeredOptimizationStrategy: 全局分层优化策略实例
    """
    global _global_layered_strategy
    if _global_layered_strategy is None:
        _global_layered_strategy = LayeredOptimizationStrategy()
    return _global_layered_strategy


def configure_parallel_training(n_samples: int, n_features: int, 
                               model_type: str = 'general') -> Dict[str, int]:
    """
    配置并行训练的便捷函数
    
    Args:
        n_samples: 样本数量
        n_features: 特征数量
        model_type: 模型类型
    
    Returns:
        dict: 并行配置
    """
    parallel_config = get_global_parallel_config()
    config = parallel_config.get_optimal_config(n_samples, n_features, model_type)
    parallel_config.apply_config(config)
    return config


# 为了向后兼容，创建全局实例
# 这样模型文件可以直接导入使用
global_cache = get_global_cache()
global_parallel_config = get_global_parallel_config()
global_layered_strategy = get_global_layered_strategy()


# 便捷函数，用于模型文件中的快速访问
def create_cache_key(model_type: str, X, y, param_space: Dict, 
                    optimization_method: str = 'bayesian') -> str:
    """
    创建缓存键的便捷函数
    
    Args:
        model_type: 模型类型
        X: 特征数据
        y: 标签数据
        param_space: 参数空间
        optimization_method: 优化方法
    
    Returns:
        str: 缓存键
    """
    return global_cache.get_cache_key(model_type, X, y, param_space, optimization_method)


def check_parameter_cache(cache_key: str) -> Optional[CachedResult]:
    """
    检查参数缓存的便捷函数
    
    Args:
        cache_key: 缓存键
    
    Returns:
        CachedResult or None: 缓存的结果
    """
    return global_cache.check_cache(cache_key)


def update_parameter_cache(cache_key: str, result: CachedResult):
    """
    更新参数缓存的便捷函数
    
    Args:
        cache_key: 缓存键
        result: 缓存结果
    """
    global_cache.update_cache(cache_key, result)


def get_optimal_parallel_config(n_samples: int, n_features: int, 
                               model_type: str = 'general') -> Dict[str, int]:
    """
    获取最优并行配置的便捷函数
    
    Args:
        n_samples: 样本数量
        n_features: 特征数量
        model_type: 模型类型
    
    Returns:
        dict: 并行配置
    """
    return global_parallel_config.get_optimal_config(n_samples, n_features, model_type)


def apply_parallel_config(config: Dict[str, int]):
    """
    应用并行配置的便捷函数
    
    Args:
        config: 并行配置字典
    """
    global_parallel_config.apply_config(config)


def get_coarse_param_space(model_type: str) -> Dict[str, Any]:
    """
    获取粗粒度参数空间的便捷函数
    
    Args:
        model_type: 模型类型
    
    Returns:
        dict: 粗粒度参数空间
    """
    return global_layered_strategy.get_coarse_param_space(model_type)


def get_fine_param_space(model_type: str, best_coarse_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    获取细粒度参数空间的便捷函数
    
    Args:
        model_type: 模型类型
        best_coarse_params: 粗粒度搜索的最佳参数
    
    Returns:
        dict: 细粒度参数空间
    """
    return global_layered_strategy.get_fine_param_space(model_type, best_coarse_params)


# 为了向后兼容，提供get_dynamic_parallel_config函数
def get_dynamic_parallel_config(n_samples: int, n_features: int, 
                               model_type: str = 'general', 
                               multi_model: bool = False) -> Dict[str, int]:
    """
    获取动态并行配置的便捷函数（向后兼容）
    
    Args:
        n_samples: 样本数量
        n_features: 特征数量
        model_type: 模型类型
        multi_model: 是否为多模型并行（暂时忽略此参数）
    
    Returns:
        dict: 并行配置
    """
    return get_optimal_parallel_config(n_samples, n_features, model_type)


if __name__ == "__main__":
    # 测试代码
    print("测试参数缓存、动态并行配置和分层优化策略...")
    
    # 测试参数缓存
    cache = ParameterCache()
    print(f"缓存统计: {cache.get_cache_stats()}")
    
    # 测试动态并行配置
    parallel_config = DynamicParallelConfig()
    print(f"系统信息: {parallel_config.get_system_info()}")
    
    # 测试不同数据规模的配置
    test_cases = [
        (1000, 50, 'svm'),
        (10000, 100, 'lightgbm'),
        (100000, 200, 'tabnet')
    ]
    
    for n_samples, n_features, model_type in test_cases:
        config = parallel_config.get_optimal_config(n_samples, n_features, model_type)
        print(f"数据规模 {n_samples}x{n_features}, 模型 {model_type}: {config}")
    
    # 测试分层优化策略
    layered_strategy = LayeredOptimizationStrategy()
    print("\n测试分层优化策略:")
    
    model_types = ['lightgbm', 'xgboost', 'svm', 'random_forest']
    for model_type in model_types:
        coarse_space = layered_strategy.get_coarse_param_space(model_type)
        print(f"{model_type} 粗粒度参数空间: {len(coarse_space)} 个参数")
        
        # 模拟粗粒度搜索结果
        if model_type == 'lightgbm':
            best_coarse = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5}
            fine_space = layered_strategy.get_fine_param_space(model_type, best_coarse)
            print(f"  基于最佳粗粒度参数的细粒度空间: {len(fine_space)} 个参数")
    
    # 测试全局函数
    config = configure_parallel_training(50000, 150, 'tabnet')
    print(f"\nTabNet配置: {config}")
    
    # 测试便捷函数
    print("\n测试便捷函数:")
    coarse_space = get_coarse_param_space('lightgbm')
    print(f"LightGBM粗粒度参数空间: {len(coarse_space)} 个参数")
    
    # 测试向后兼容函数
    compat_config = get_dynamic_parallel_config(10000, 100, 'lightgbm')
    print(f"向后兼容的并行配置: {compat_config}")
    
    print("\n所有测试完成！")