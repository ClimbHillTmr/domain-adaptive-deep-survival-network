#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
透析患者透中高血压预测模型

支持的机器学习模型:
1. LightGBM - 梯度提升决策树，支持贝叶斯优化
2. C-SVM - 支持向量机，支持贝叶斯优化和5折交叉验证
3. TabNet - 深度表格学习模型，支持贝叶斯优化

主要特性:
- 高级缺失值填补管道（迭代填补、KNN填补）
- 贝叶斯优化超参数搜索
- 5折交叉验证
- 验证集参与训练选项
- 多目标变量支持
- 时间序列预测
- 可视化和可解释性分析


nohup '/home/cht/Works/PredictionTimeHypotensionDialysis/透前模型/ 透前数据__透中低血压建模.py' >> 透前模型/透前数据__透中低血压建模.log 2>&1 &
"""

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import warnings

warnings.filterwarnings("ignore")
import pickle
import joblib
import os
from sklearn.utils.class_weight import compute_class_weight
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest

from datetime import datetime
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import hashlib
import json

import sys

sys.path.append("..")
sys.path.append("/home/cht/Works/PredictionTimeHypotensionDialysis")

from Method_Utils.train_untils import (
    remove_rows_with_few_duplicates,
    create_moved_dataframe,
    process_dataset,
    Align_standard,
    features_based,
    features,
    quantile_99,
    calculate_class_weights,
)

# 导入配置管理器
# 导入配置管理器
try:
    from Method_Utils.model_config import (
        get_config,
        apply_preset,
        setup_environment,
        config_manager,
    )
except ImportError:
    # 如果配置模块不存在，创建基本配置
    from dataclasses import dataclass
    from typing import Dict, Any, Optional
    
    @dataclass
    class Config:
        imputation_strategy: str = "median"
        random_state: int = 42
        test_size: float = 0.2
        validation_size: float = 0.2
        
    # 全局配置实例
    _config = Config()
    
    def get_config() -> Config:
        """获取全局配置"""
        return _config
    
    def apply_preset(preset_name: str):
        """应用预设配置"""
        global _config
        if preset_name == "reproducible":
            _config.random_state = 42
            np.random.seed(42)
            
    def setup_environment():
        """设置环境"""
        warnings.filterwarnings("ignore")
        os.environ["PYTHONHASHSEED"] = "42"
    
    class ConfigManager:
        def __init__(self):
            self.config = _config
    
    config_manager = ConfigManager()

# 数据标准化函数
from Method_Utils.data_standardization import *


# 导入所有模型
from Methods.Models.LightGBM_model import LightGBM_model
from Methods.Models.C_SVM_model import C_SVM_model
from Methods.Models.TabNet_optimized import TabNetSimplified, TabNet_model_optimized
from Methods.Models.NODE_model_simple_wrapper import NODE_model_simple
from Methods.Models.ResNet_model import ResNet_model
# from Methods.Models.attention_knn_model import (
#         DialysisAttentionKNN,
#         create_attention_knn_pipeline,
#     )

from Methods.fuding_test import fuding_test
from Methods.date_method import split_dataset_by_date


def log_with_timestamp(message):
    """
    带时间戳的日志输出函数

    参数:
    - message: 要输出的消息
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def calculate_class_weights(y):
    """计算类别权重"""
    unique_classes = np.unique(y)
    class_weights_array = compute_class_weight('balanced', classes=unique_classes, y=y)
    return dict(zip(unique_classes, class_weights_array))


def standardize_dialysis_data(X_train, X_val, X_test, method="standard", save_dir="./scalers", target_name="", verbose=True):
    """数据标准化函数"""
    if method == "standard":
        scaler = StandardScaler()
    else:
        scaler = StandardScaler()  # 默认使用标准化
    
    # 拟合并转换数据
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # 保存scaler
    os.makedirs(save_dir, exist_ok=True)
    scaler_path = os.path.join(save_dir, f"scaler_{target_name}.pkl")
    
    joblib.dump(scaler, scaler_path)
    
    if verbose:
        log_with_timestamp(f"数据标准化完成，scaler已保存到: {scaler_path}")
    
    return {
        "X_train_scaled": pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index),
        "X_val_scaled": pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index),
        "X_test_scaled": pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index),
        "scaler_path": scaler_path
    }


# 导入参数缓存模块
try:
    from Methods.parameter_cache import (
        ParameterCache, CachedResult, global_cache,
        create_cache_key, check_parameter_cache, update_parameter_cache,
        configure_parallel_training, get_optimal_parallel_config, apply_parallel_config
    )
    PARAMETER_CACHE_AVAILABLE = True
except ImportError:
    PARAMETER_CACHE_AVAILABLE = False
    print("警告: 参数缓存模块不可用")

# 导入优化工具模块（已整合到parameter_cache中）
# LayeredOptimizationStrategy 和 get_dynamic_parallel_config 现在从 parameter_cache 导入



def train_multiple_models(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    class_weights,
    target,
    kinds,
    optimization_method="bayesian",
    cv=5,
):
    """训练和比较多个模型"""
    results = {}

    log_with_timestamp(f"\n=== 开始训练多个模型 ===")
    log_with_timestamp(f"目标变量: {target}")
    log_with_timestamp(f"训练集大小: {X_train.shape}")
    log_with_timestamp(f"验证集大小: {X_val.shape}")
    log_with_timestamp(f"测试集大小: {X_test.shape}")

    # # 1. LightGBM模型
    # log_with_timestamp(f"\n--- 训练 LightGBM 模型 ---")
    # try:
    #     # 调用模型 - 使用正确的参数
    #     lgb_result = LightGBM_model(
    #         X_train=X_train,
    #         X_test=X_test,
    #         y_train=y_train,
    #         y_test=y_test,
    #         X_val=X_val,
    #         y_val=y_val,
    #         class_weights=class_weights,
    #         target=target,
    #         kinds=kinds,
    #         use_early_stopping=True,
    #         n_iter=100,
    #         cv_folds=cv,
    #     )
    #     results["LightGBM"] = lgb_result

    #     if lgb_result is not None:
    #         log_with_timestamp(f"LightGBM 训练完成!")
    # except Exception as e:
    #     log_with_timestamp(f"LightGBM 训练失败: {e}")
    #     import traceback
    #     log_with_timestamp(f"详细错误: {traceback.format_exc()}")
    #     results["LightGBM"] = None

    # # 2. SVM模型
    # log_with_timestamp(f"\n--- 训练 SVM 模型 ---")
    # try:
    #     # 调用模型 - 使用正确的参数
    #     svm_result = C_SVM_model(
    #         X_train=X_train,
    #         X_test=X_test,
    #         y_train=y_train,
    #         y_test=y_test,
    #         X_val=X_val,
    #         y_val=y_val,
    #         class_weights=class_weights,
    #         target=target,
    #         kinds=kinds,
    #         search_type=optimization_method,
    #         n_iter=100,
    #         cv_folds=cv,
    #         use_validation_in_training=True,
    #     )
    #     results["SVM"] = svm_result
        
    #     if svm_result is not None:
    #         log_with_timestamp(f"SVM 训练完成!")
    # except Exception as e:
    #     log_with_timestamp(f"SVM 训练失败: {e}")
    #     import traceback
    #     log_with_timestamp(f"详细错误: {traceback.format_exc()}")
    #     results["SVM"] = None

    # # 3. TabNet模型
    # log_with_timestamp(f"\n--- 训练 TabNet 模型 ---")
    # try:
    #     # 调用模型 - 使用正确的参数
    #     tabnet_result = TabNet_model_optimized(
    #         X_train=X_train,
    #         X_test=X_test,
    #         y_train=y_train,
    #         y_test=y_test,
    #         X_val=X_val,
    #         y_val=y_val,
    #         class_weights=class_weights,
    #         target=target,
    #         kinds=kinds,
    #         optimize=True,
    #         search_type=optimization_method,
    #         n_iter=100,
    #         cv_folds=cv,
    #         use_validation_in_training=True,
    #     )
    #     results["TabNet"] = tabnet_result
        
    #     if tabnet_result is not None:
    #         log_with_timestamp(f"TabNet 训练完成!")
    # except Exception as e:
    #     log_with_timestamp(f"TabNet 训练失败: {e}")
    #     import traceback
    #     log_with_timestamp(f"详细错误: {traceback.format_exc()}")
    #     results["TabNet"] = None

    # 4. NODE模型
    log_with_timestamp(f"\n--- 训练 NODE 模型 ---")
    try:
        node_result = NODE_model_simple(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            X_val=X_val,
            y_val=y_val,
            class_weights=class_weights,
            target=target,
            kinds=kinds,
            n_iter=50,
            cv_folds=cv,
            use_early_stopping=True,
            use_validation_in_training=True,
            base_path="./model_checkpoints/NODE",
        )
        results["NODE"] = node_result
        if node_result is not None:
            log_with_timestamp(f"NODE 训练完成!")
    except Exception as e:
        log_with_timestamp(f"NODE 训练失败: {e}")
        import traceback
        log_with_timestamp(f"详细错误: {traceback.format_exc()}")
        results["NODE"] = None

    # 5. ResNet模型
    log_with_timestamp(f"\n--- 训练 ResNet 模型 ---")
    try:
        resnet_result = ResNet_model(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            X_val=X_val,
            y_val=y_val,
            class_weights=class_weights,
            target=target,
            kinds=kinds,
            n_iter=50,
            cv_folds=cv,
            use_early_stopping=True,
            use_validation_in_training=True,
            base_path="./model_checkpoints/ResNet",
        )
        results["ResNet"] = resnet_result
        if resnet_result is not None:
            log_with_timestamp(f"ResNet 训练完成!")
    except Exception as e:
        log_with_timestamp(f"ResNet 训练失败: {e}")
        import traceback
        log_with_timestamp(f"详细错误: {traceback.format_exc()}")
        results["ResNet"] = None

    return results


def main():
    """主函数 - 透析患者透中低血压预测模型完全优化版
    
    主要优化特性:
    1. 增强的错误处理和恢复机制
    2. 智能资源管理和内存优化
    3. 高级缓存和参数管理
    4. 全面的性能监控
    5. 自适应数据预处理流程
    6. 多模型并行训练优化
    """
    # ===== 配置参数和环境初始化 =====
    global SAVE_MODELS, MODEL_SAVE_DIR
    SAVE_MODELS = True  # 是否保存模型结果
    MODEL_SAVE_DIR = "./Results"  # 模型保存目录
    ENABLE_PARALLEL = True  # 是否启用并行训练
    MEMORY_OPTIMIZATION = True  # 是否启用内存优化
    
    # 创建保存目录
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    log_with_timestamp("=== 透析患者透中低血压预测模型 - 完全优化版 ===")
    log_with_timestamp(f"模型保存: {'启用' if SAVE_MODELS else '禁用'}")
    log_with_timestamp(f"保存目录: {MODEL_SAVE_DIR}")
    log_with_timestamp(f"并行训练: {'启用' if ENABLE_PARALLEL else '禁用'}")
    log_with_timestamp(f"内存优化: {'启用' if MEMORY_OPTIMIZATION else '禁用'}")
    
    # ===== 初始化缓存和配置管理 =====
    log_with_timestamp("\n初始化参数缓存机制...")
    try:
        if PARAMETER_CACHE_AVAILABLE:
            cache_stats = global_cache.get_cache_stats()
            log_with_timestamp(f"参数缓存统计: {cache_stats}")
            
            # 配置并行训练策略
            if ENABLE_PARALLEL:
                parallel_config = get_optimal_parallel_config()
                apply_parallel_config(parallel_config)
                log_with_timestamp(f"并行配置: {parallel_config}")
        else:
            log_with_timestamp("参数缓存不可用，使用默认配置")
    except Exception as e:
        log_with_timestamp(f"缓存初始化失败: {e}，继续使用默认配置")
    
    # ===== 性能监控初始化 =====
    start_time = time.time()
    memory_usage = []
    
    def log_memory_usage(stage):
        """记录内存使用情况"""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            memory_usage.append((stage, memory_mb))
            log_with_timestamp(f"内存使用 [{stage}]: {memory_mb:.1f} MB")
        except ImportError:
            pass
    
    log_memory_usage("初始化完成")

    # ===== 智能数据加载和验证 =====
    log_with_timestamp("\n开始数据加载和验证...")
    
    # 数据文件路径配置
    TRAIN_DATA_PATH = "/home/cht/Works/PredictionTimeHypotensionDialysis/data_preprocessing/data/深医_final_data.csv"
    BACKUP_TRAIN_PATH = "/home/cht/Works/PredictionTimeHypotensionDialysis/透前模型/test_data_1.csv"
    
    try:
        # 加载训练数据
        if os.path.exists(TRAIN_DATA_PATH):
            dataset = pd.read_csv(TRAIN_DATA_PATH)
            log_with_timestamp(f"成功加载训练数据: {TRAIN_DATA_PATH}")
        elif os.path.exists(BACKUP_TRAIN_PATH):
            dataset = pd.read_csv(BACKUP_TRAIN_PATH)
            log_with_timestamp(f"使用备用训练数据: {BACKUP_TRAIN_PATH}")
        else:
            raise FileNotFoundError(f"训练数据文件不存在: {TRAIN_DATA_PATH}")
        
        # 加载测试数据
        try:
            test_set = fuding_test()
            log_with_timestamp("成功加载福鼎测试数据")
        except Exception as e:
            log_with_timestamp(f"福鼎测试数据加载失败: {e}，使用训练数据分割作为测试集")
            # 备用方案：从训练数据中分割测试集
            dataset, test_set = train_test_split(dataset, test_size=0.2, random_state=42)
            log_with_timestamp(f"使用训练数据分割 - 训练集: {dataset.shape}, 测试集: {test_set.shape}")
        
    except Exception as e:
        log_with_timestamp(f"数据加载失败: {e}")
        raise
    
    # ===== 智能数据预处理流程 =====
    log_with_timestamp("\n开始智能数据预处理流程...")
    
    preprocessing_steps = [
        ("Align_standard", Align_standard, "数据标准化对齐"),
        # ("quantile_99", quantile_99, "99分位数异常值处理"),
        ("remove_duplicates", remove_rows_with_few_duplicates, "移除重复较少的行")
    ]
    
    for step_name, step_func, step_desc in preprocessing_steps:
        try:
            log_with_timestamp(f"执行 {step_desc}...")
            
            # 记录处理前的数据状态
            train_shape_before = dataset.shape
            test_shape_before = test_set.shape
            
            # 执行预处理步骤
            dataset = step_func(dataset)
            test_set = step_func(test_set)
            
            # 记录处理后的数据状态
            train_shape_after = dataset.shape
            test_shape_after = test_set.shape
            
            log_with_timestamp(
                f"{step_desc}完成 - 训练集: {train_shape_before} → {train_shape_after}, "
                f"测试集: {test_shape_before} → {test_shape_after}"
            )
            
            # 数据完整性检查
            if dataset.empty or test_set.empty:
                log_with_timestamp(f"警告: {step_desc}后数据集为空")
                break
                
            log_memory_usage(f"{step_name}完成")
            
        except Exception as e:
            log_with_timestamp(f"{step_desc}失败: {e}，跳过此步骤")
            continue

    # ===== 智能目标变量分析 =====
    log_with_timestamp("\n开始目标变量分布分析...")
    
    # 定义所有可能的目标变量
    targets = ["透中低血压_计算", "降幅时间点比值区间", "降幅时间点差值区间"]
    
    # 初始化全局结果存储
    global_results = {}
    training_summary = {
        "total_targets": len(targets),
        "successful_targets": 0,
        "failed_targets": 0,
        "target_results": {}
    }
    
    # ===== 智能特征定义和验证 =====
    log_with_timestamp("\n开始特征定义和验证...")
    
    # 核心基础特征（经过临床验证的重要特征）
    base_features_core = [
        "传染病", "抗凝剂类型", "干体重", "瘘管类型", "瘘管位置", "瘘管使用时间",
        "首次透析年龄", "透析方式", "透析龄", "透析龄_天数", "透析年龄", "性别"
    ]
    
    # 历史平均特征（透析历史数据）
    base_features_history = [
        "历史平均超滤率_mean", "历史平均动脉压_mean", "历史平均干体重",
        "历史平均降幅时间点比值区间", "历史平均降幅时间点差值区间",
        "历史平均静脉压_mean", "历史平均跨膜压_mean", "历史平均实际透析时长",
        "历史平均透析液钙浓度", "历史平均透析液电导率", "历史平均透析液温度_mean",
        "历史平均透析中收缩压_mean", "历史平均透析中舒张压_mean", "历史平均透析中脉搏_mean",
        "历史平均透前收缩压", "历史平均透前舒张压", "历史平均透前呼吸频率",
        "历史平均透前体重", "历史平均透中高血压_计算", "历史平均血流速_mean",
        "历史平均涨幅时间点比值区间", "历史平均涨幅时间点差值区间"
    ]
    
    # 当前透析特征
    base_features_current = [
        "透析液钙浓度", "透析液电导率", "透前呼吸频率", "透前收缩压", "透前舒张压",
        "透前体重", "透前体重-干体重"
    ]
    
    # 合并所有基础特征
    base_features = base_features_core + base_features_history + base_features_current
    

    
    # 历史特征（用于不同目标变量的特征增强）
    history_rate_proportion = [
        "history_HBP_rate", "history_LBP_times_0_rate", "history_LBP_times_1_rate",
        "history_LBP_times_2_rate", "history_LBP_times_3_rate", "history_LBP_times_4_rate"
    ]
    
    history_rate_diff = [
        "history_HBP", "history_LBP_times_0", "history_LBP_times_1",
        "history_LBP_times_2", "history_LBP_times_3", "history_LBP_times_4"
    ]
    
    # ===== 对每个目标变量进行智能建模 =====
    for target_idx, target in enumerate(targets):
        target_start_time = time.time()
        
        log_with_timestamp(f"\n{'='*80}")
        log_with_timestamp(f"目标变量 [{target_idx+1}/{len(targets)}]: {target}")
        log_with_timestamp(f"{'='*80}")
        
        try:

            # ===== 智能特征选择和工程 =====
            log_with_timestamp(f"开始特征选择和工程...")
            
            # 基础特征选择
            current_features = base_features.copy()
            
            # 根据目标变量添加特定的历史特征
            if target == "降幅时间点比值区间" and history_rate_proportion:
                current_features.extend(history_rate_proportion)
                log_with_timestamp(f"添加历史比例特征: {len(history_rate_proportion)}个")
            elif target == "降幅时间点差值区间" and history_rate_diff:
                current_features.extend(history_rate_diff)
                log_with_timestamp(f"添加历史差值特征: {len(history_rate_diff)}个")
            
            # ===== 智能数据过滤和验证 =====
            log_with_timestamp(f"开始数据过滤和验证...")
            
            # 检查特征是否存在
            available_features = [f for f in current_features if f in dataset.columns]
            missing_features = [f for f in current_features if f not in dataset.columns]
            
            if missing_features:
                log_with_timestamp(f"警告: 缺失特征 {len(missing_features)}个: {missing_features[:5]}...")
            
            log_with_timestamp(f"可用特征: {len(available_features)}个")
            current_features = available_features
            
            if not current_features:
                log_with_timestamp(f"错误: 没有可用特征，跳过目标变量 {target}")
                training_summary["failed_targets"] += 1
                continue
            
            # 准备特征和标签
            X = dataset[current_features].copy()
            y = dataset[target].copy()
            X_external_test = test_set[current_features].copy()
            y_external_test = test_set[target].copy()
            
            # 根据任务类型过滤数据
            if target in ["降幅时间点比值区间", "降幅时间点差值区间"]:
                # 时间预测任务：移除标签为0的样本
                valid_mask_train = y != 0
                valid_mask_test = y_external_test != 0
                
                X = X[valid_mask_train]
                y = y[valid_mask_train]
                X_external_test = X_external_test[valid_mask_test]
                y_external_test = y_external_test[valid_mask_test]
                
                task_type = "时间预测"
            else:
                # 二分类任务：保留所有样本
                task_type = "二分类"
            
            log_with_timestamp(f"任务类型: {task_type}")
            log_with_timestamp(f"数据过滤后 - 训练集: {len(X)}, 测试集: {len(X_external_test)}")
            
            # 检查数据是否为空
            if len(X) == 0 or len(X_external_test) == 0:
                log_with_timestamp(f"错误: 过滤后数据为空，跳过目标变量 {target}")
                training_summary["failed_targets"] += 1
                continue

            # ===== 智能数据预处理管道 =====
            log_with_timestamp(f"开始智能数据预处理...")
            
            # 使用高级缺失值填补和数据分割
            log_with_timestamp("执行数据分割...")
            
            # 检查是否有足够的样本进行分层抽样
            unique_classes = np.unique(y)
            min_class_count = min([np.sum(y == cls) for cls in unique_classes])
            
            if min_class_count < 2:
                log_with_timestamp(f"警告: 类别样本数不足，使用简单随机分割")
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
            else:
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
            
            # 外部测试集
            X_test = X_external_test
            y_test = y_external_test
            
            log_with_timestamp(f"数据分割完成 - 训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")
            log_with_timestamp(f"类别分布 - 训练集: {dict(zip(*np.unique(y_train, return_counts=True)))}")
            log_with_timestamp(f"类别分布 - 验证集: {dict(zip(*np.unique(y_val, return_counts=True)))}")
            log_with_timestamp(f"类别分布 - 测试集: {dict(zip(*np.unique(y_test, return_counts=True)))}")
            
            # 数据标准化处理
            log_with_timestamp("执行数据标准化...")
            try:
                standardization_results = standardize_dialysis_data(
                    X_train=X_train,
                    X_val=X_val,
                    X_test=X_test,
                    method="standard",  # 使用标准化方法
                    save_dir="./scalers",
                    target_name=target,
                    verbose=True,
                )
                
                # 使用标准化后的数据
                X_train = standardization_results["X_train_scaled"]
                X_val = standardization_results["X_val_scaled"]
                X_test = standardization_results["X_test_scaled"]
                scaler_path = standardization_results["scaler_path"]
                
                # 兼容mambular：重置索引，object转category，填补缺失
                for df_name, df in zip(["X_train", "X_val", "X_test"], [X_train, X_val, X_test]):
                    df.reset_index(drop=True, inplace=True)
                    for col in df.select_dtypes(include=["object"]).columns:
                        df[col] = df[col].astype("category")
                    if df.isnull().any().any():
                        df.fillna(-999, inplace=True)
                    log_with_timestamp(f"{df_name} 索引范围: {df.index.min()}~{df.index.max()}，类别列: {list(df.select_dtypes(include=['category']).columns)}，缺失值总数: {df.isnull().sum().sum()}")
                
                # 标签处理
                for y_name, y in zip(["y_train", "y_val", "y_test"], [y_train, y_val, y_test]):
                    y = pd.Series(y).reset_index(drop=True)
                    y = y.astype(int)
                    vars()[y_name] = y
                    log_with_timestamp(f"{y_name} 索引范围: {y.index.min()}~{y.index.max()}，类型: {y.dtype}，缺失值: {y.isnull().sum()}")

            except Exception as e:
                log_with_timestamp(f"标准化失败，使用简单标准化: {e}")
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_val = scaler.transform(X_val)
                X_test = scaler.transform(X_test)
                
                # 转换回DataFrame
                X_train = pd.DataFrame(X_train, columns=current_features)
                X_val = pd.DataFrame(X_val, columns=current_features)
                X_test = pd.DataFrame(X_test, columns=current_features)
            
            # 计算类别权重
            try:
                class_weights = calculate_class_weights(y_train)
                log_with_timestamp(f"类别权重: {class_weights}")
            except Exception as e:
                log_with_timestamp(f"计算类别权重失败，使用默认权重: {e}")
                unique_classes = np.unique(y_train)
                class_weights = {cls: 1.0 for cls in unique_classes}
                
            
            # ===== 模型训练和评估 =====
            log_with_timestamp(f"开始模型训练和评估...")
            
            try:
                # 训练多个模型并比较
                model_results = train_multiple_models(
                    X_train,
                    X_val,
                    X_test,
                    y_train,
                    y_val,
                    y_test,
                    class_weights,
                    target,
                    "透前模型_完全优化版",
                )
                
                # 处理训练结果
                if model_results:
                    log_with_timestamp(f"\n{target} 模型训练完成")
                    
                    # 收集结果到全局摘要
                    target_model_results = {}
                    best_accuracy = 0
                    best_model = None
                    
                    for model_name, result in model_results.items():
                        if result is not None and isinstance(result, dict) and "scores" in result:
                            scores = result["scores"]
                            if isinstance(scores, dict) and "accuracy" in scores:
                                accuracy = scores["accuracy"]
                                target_model_results[model_name] = accuracy
                                log_with_timestamp(f"{model_name}: 准确率 = {accuracy:.4f}")
                                
                                if accuracy > best_accuracy:
                                    best_accuracy = accuracy
                                    best_model = model_name
                    
                    # 更新全局结果
                    global_results[target] = {
                        'best_model': best_model,
                        'best_accuracy': best_accuracy,
                        'all_results': target_model_results,
                        'n_features': len(current_features),
                        'n_samples': len(X_train),
                        'task_type': task_type
                    }
                    
                    training_summary["successful_targets"] += 1
                    training_summary["target_results"][target] = target_model_results
                    
                    if best_model:
                        log_with_timestamp(f"最佳模型: {best_model} (准确率: {best_accuracy:.4f})")
                    
                else:
                    log_with_timestamp(f"警告: {target} 没有成功训练任何模型")
                    training_summary["failed_targets"] += 1
                
            except Exception as e:
                log_with_timestamp(f"模型训练失败: {e}")
                import traceback
                log_with_timestamp(f"详细错误信息: {traceback.format_exc()}")
                training_summary["failed_targets"] += 1
                continue
            
            # 记录目标变量完成时间
            target_end_time = time.time()
            target_duration = target_end_time - target_start_time
            log_with_timestamp(f"{target} 处理完成，耗时: {target_duration:.2f}秒")
            
        except Exception as e:
            log_with_timestamp(f"目标变量 {target} 处理失败: {e}")
            import traceback
            log_with_timestamp(f"详细错误信息: {traceback.format_exc()}")
            training_summary["failed_targets"] += 1
            continue

    # ===== 全局训练结果汇总 =====
    total_end_time = time.time()
    total_duration = total_end_time - start_time
    
    log_with_timestamp(f"\n{'='*80}")
    log_with_timestamp(f"全局训练结果汇总")
    log_with_timestamp(f"{'='*80}")
    
    log_with_timestamp(f"总耗时: {total_duration:.2f}秒")
    log_with_timestamp(f"成功目标变量: {training_summary['successful_targets']}/{training_summary['total_targets']}")
    log_with_timestamp(f"失败目标变量: {training_summary['failed_targets']}/{training_summary['total_targets']}")
    
    # 显示每个目标变量的最佳结果
    if global_results:
        log_with_timestamp(f"\n=== 各目标变量最佳模型性能 ===")
        for target, results in global_results.items():
            log_with_timestamp(f"{target}:")
            log_with_timestamp(f"  最佳模型: {results['best_model']}")
            log_with_timestamp(f"  最佳准确率: {results['best_accuracy']:.4f}")
            log_with_timestamp(f"  特征数量: {results['n_features']}")
            log_with_timestamp(f"  样本数量: {results['n_samples']}")
            log_with_timestamp(f"  任务类型: {results['task_type']}")
            log_with_timestamp("")
        
        # 创建综合性能比较表
        if len(global_results) > 1:
            log_with_timestamp(f"=== 综合性能比较表 ===")
            comparison_data = []
            for target, results in global_results.items():
                comparison_data.append({
                    "目标变量": target,
                    "最佳模型": results['best_model'],
                    "准确率": f"{results['best_accuracy']:.4f}",
                    "特征数": results['n_features'],
                    "样本数": results['n_samples'],
                    "任务类型": results['task_type']
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            log_with_timestamp(comparison_df.to_string(index=False))
    
    
    # 保存训练摘要
    training_summary["total_duration"] = total_duration
    training_summary["global_results"] = global_results
    
    if SAVE_MODELS:
        try:
            summary_path = os.path.join(MODEL_SAVE_DIR, "training_summary.json")
            with open(summary_path, 'w', encoding='utf-8') as f:
                # 转换numpy类型为Python原生类型以便JSON序列化
                json_summary = {}
                for key, value in training_summary.items():
                    if isinstance(value, (np.integer, np.floating)):
                        json_summary[key] = value.item()
                    elif isinstance(value, dict):
                        json_summary[key] = {k: v.item() if isinstance(v, (np.integer, np.floating)) else v for k, v in value.items()}
                    else:
                        json_summary[key] = value
                
                json.dump(json_summary, f, ensure_ascii=False, indent=2)
            log_with_timestamp(f"训练摘要已保存到: {summary_path}")
        except Exception as e:
            log_with_timestamp(f"保存训练摘要失败: {e}")

    log_with_timestamp(f"\n{'='*80}")
    log_with_timestamp("透析患者透中低血压预测模型训练完成!")
    log_with_timestamp(f"{'='*80}")
    
    if SAVE_MODELS:
        log_with_timestamp(f"\n模型结果已保存到目录: {MODEL_SAVE_DIR}")
    
    # 返回训练结果供后续分析
    return {
        "global_results": global_results,
        "training_summary": training_summary,
        "total_duration": total_duration
    }


if __name__ == "__main__":
    # 设置环境
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    warnings.filterwarnings("ignore")
    
    # 设置随机种子确保可重现性
    import random
    random.seed(42)
    np.random.seed(42)
    
    try:
        log_with_timestamp("开始透析低血压预测模型训练...")
        log_with_timestamp("=" * 50)

        # 应用可重复性配置预设
        apply_preset("reproducible")
        setup_environment()

        # 获取配置
        config = get_config()
        log_with_timestamp(
            f"使用配置: 填补策略={config.imputation_strategy}, 随机种子={config.random_state }"
        )

        # 运行主函数
        results = main()
        
        if results:
            log_with_timestamp("\n=== 最终训练结果摘要 ===")
            log_with_timestamp(f"成功训练目标数: {results['training_summary']['successful_targets']}")
            log_with_timestamp(f"失败目标数: {results['training_summary']['failed_targets']}")
            log_with_timestamp(f"总耗时: {results['total_duration']:.2f}秒")
            
            if results['global_results']:
                best_overall = max(results['global_results'].items(), 
                                 key=lambda x: x[1]['best_accuracy'])
                log_with_timestamp(f"全局最佳模型: {best_overall[0]} - {best_overall[1]['best_model']} (准确率: {best_overall[1]['best_accuracy']:.4f})")
        
        log_with_timestamp("程序执行完成!")
        
    except KeyboardInterrupt:
        log_with_timestamp("\n程序被用户中断")
    except Exception as e:
        log_with_timestamp(f"\n程序执行出错: {e}")
        import traceback
        log_with_timestamp(f"详细错误信息: {traceback.format_exc()}")
    finally:
        log_with_timestamp("程序结束")
