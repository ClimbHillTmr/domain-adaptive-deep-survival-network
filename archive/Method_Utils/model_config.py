#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型配置管理模块

统一管理多线程设置、填补策略和模型参数，提高可重复性和性能

作者: AI Assistant
日期: 2024
"""

import multiprocessing
import os
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class ModelConfig:
    """模型配置类"""
    
    # 多线程配置 - 针对32核CPU优化
    max_workers: int = 28  # 保留4核给系统
    use_multiprocessing: bool = True
    cpu_usage_ratio: float = 0.875  # 28/32 = 0.875，更精确的比例
    
    # 填补策略配置
    imputation_strategy: str = 'simple'  # 'simple', 'knn', 'iterative'
    knn_neighbors: int = 5
    iterative_max_iter: int = 3
    
    # 随机种子配置
    random_state: int = 42
    
    # TabNet特定配置 - 32核优化
    tabnet_num_workers: int = 8  # 32核下使用8个worker更稳定
    tabnet_batch_size: int = 4096  # 增大batch size利用更多内存
    tabnet_max_epochs: int = 100
    
    # 超参数搜索配置 - 32核优化
    bayesian_n_iter: int = 50
    cv_folds: int = 5
    search_n_jobs: int = 8  # 增加搜索并行数
    
    def __post_init__(self):
        """初始化后处理"""
        # 动态计算最优线程数
        cpu_count = multiprocessing.cpu_count()
        
        # 设置最大工作线程数 - 32核优化
        self.max_workers = max(1, min(int(cpu_count * self.cpu_usage_ratio), self.max_workers))
        
        # 设置TabNet工作线程数 - 32核优化
        if cpu_count >= 32:
            self.tabnet_num_workers = max(1, min(8, self.tabnet_num_workers))  # 32核使用8个worker
        elif cpu_count >= 16:
            self.tabnet_num_workers = max(1, min(6, self.tabnet_num_workers))  # 16核使用6个worker
        else:
            self.tabnet_num_workers = max(1, min(cpu_count // 2, 4))  # 其他情况保持原逻辑
        
        # 设置搜索并行数 - 32核优化
        if cpu_count >= 32:
            self.search_n_jobs = max(1, min(8, self.search_n_jobs))  # 32核使用8个并行
        elif cpu_count >= 16:
            self.search_n_jobs = max(1, min(4, self.search_n_jobs))  # 16核使用4个并行
        else:
            self.search_n_jobs = max(1, min(cpu_count // 4, 2))  # 其他情况保守设置
        
        print(f"模型配置初始化完成:")
        print(f"  CPU核心数: {cpu_count}")
        print(f"  最大工作线程: {self.max_workers}")
        print(f"  TabNet工作线程: {self.tabnet_num_workers}")
        print(f"  搜索并行数: {self.search_n_jobs}")
        print(f"  填补策略: {self.imputation_strategy}")
        print(f"  随机种子: {self.random_state}")


class ConfigManager:
    """配置管理器"""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config is None:
            self._config = ModelConfig()
    
    @property
    def config(self) -> ModelConfig:
        return self._config
    
    def update_config(self, **kwargs):
        """更新配置"""
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
            else:
                print(f"警告: 未知配置项 {key}")
        
        # 重新初始化
        self._config.__post_init__()
    
    def get_tabnet_config(self) -> Dict[str, Any]:
        """获取TabNet配置"""
        return {
            'num_workers': self._config.tabnet_num_workers,
            'batch_size': self._config.tabnet_batch_size,
            'max_epochs': self._config.tabnet_max_epochs,
            'random_state': self._config.random_state
        }
    
    def get_imputation_config(self) -> Dict[str, Any]:
        """获取填补配置"""
        return {
            'strategy': self._config.imputation_strategy,
            'n_neighbors': self._config.knn_neighbors,
            'max_iter': self._config.iterative_max_iter,
            'random_state': self._config.random_state
        }
    
    def get_search_config(self) -> Dict[str, Any]:
        """获取搜索配置"""
        return {
            'n_iter': self._config.bayesian_n_iter,
            'cv': self._config.cv_folds,
            'n_jobs': self._config.search_n_jobs,
            'random_state': self._config.random_state
        }
    
    def set_environment_variables(self):
        """设置环境变量"""
        # 设置OpenMP线程数
        os.environ['OMP_NUM_THREADS'] = str(self._config.max_workers)
        
        # 设置MKL线程数
        os.environ['MKL_NUM_THREADS'] = str(self._config.max_workers)
        
        # 设置BLAS线程数
        os.environ['OPENBLAS_NUM_THREADS'] = str(self._config.max_workers)
        
        # 设置NumPy线程数
        os.environ['NUMEXPR_NUM_THREADS'] = str(self._config.max_workers)
        
        print(f"环境变量设置完成，线程数限制为: {self._config.max_workers}")


# 全局配置管理器实例
config_manager = ConfigManager()


def get_config() -> ModelConfig:
    """获取全局配置"""
    return config_manager.config


def update_global_config(**kwargs):
    """更新全局配置"""
    config_manager.update_config(**kwargs)


def setup_environment():
    """设置环境"""
    config_manager.set_environment_variables()


# 预设配置
PRESET_CONFIGS = {
    'fast': {
        'imputation_strategy': 'simple',
        'bayesian_n_iter': 20,
        'iterative_max_iter': 2,
        'cpu_usage_ratio': 0.8
    },
    'balanced': {
        'imputation_strategy': 'knn',
        'bayesian_n_iter': 50,
        'iterative_max_iter': 3,
        'cpu_usage_ratio': 0.5
    },
    'accurate': {
        'imputation_strategy': 'iterative',
        'bayesian_n_iter': 100,
        'iterative_max_iter': 5,
        'cpu_usage_ratio': 0.3
    },
    'reproducible': {
        'imputation_strategy': 'simple',
        'bayesian_n_iter': 30,
        'iterative_max_iter': 3,
        'cpu_usage_ratio': 0.25,
        'use_multiprocessing': False
    },
    # 32核CPU专用配置
    'high_performance_32core': {
        'imputation_strategy': 'knn',
        'bayesian_n_iter': 80,
        'iterative_max_iter': 4,
        'cpu_usage_ratio': 0.875,
        'max_workers': 28,
        'tabnet_num_workers': 8,
        'tabnet_batch_size': 4096,
        'search_n_jobs': 8,
        'use_multiprocessing': True
    },
    'ultra_fast_32core': {
        'imputation_strategy': 'simple',
        'bayesian_n_iter': 30,
        'iterative_max_iter': 2,
        'cpu_usage_ratio': 0.9,
        'max_workers': 30,
        'tabnet_num_workers': 10,
        'tabnet_batch_size': 6144,
        'search_n_jobs': 10,
        'use_multiprocessing': True
    }
}


def apply_preset(preset_name: str):
    """应用预设配置"""
    if preset_name in PRESET_CONFIGS:
        update_global_config(**PRESET_CONFIGS[preset_name])
        print(f"已应用预设配置: {preset_name}")
    else:
        print(f"未知预设配置: {preset_name}")
        print(f"可用预设: {list(PRESET_CONFIGS.keys())}")


if __name__ == "__main__":
    # 测试配置管理器
    print("=== 配置管理器测试 ===")
    
    config = get_config()
    print(f"默认配置: {config}")
    
    print("\n=== TabNet配置 ===")
    print(config_manager.get_tabnet_config())
    
    print("\n=== 填补配置 ===")
    print(config_manager.get_imputation_config())
    
    print("\n=== 搜索配置 ===")
    print(config_manager.get_search_config())
    
    print("\n=== 应用快速预设 ===")
    apply_preset('fast')
    
    print("\n=== 设置环境变量 ===")
    setup_environment()