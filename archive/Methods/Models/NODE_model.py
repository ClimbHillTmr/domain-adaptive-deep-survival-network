#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NODE模型训练脚本（标准化版本）
适配透析患者临床数据分析与建模

作者: 透析数据分析团队
日期: 2024
版本: 3.0 - 修复Mambular预处理器问题的标准化接口版本
"""

import os
import sys
import json
import joblib
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import multiprocessing
import psutil
import warnings
import pickle
import hashlib
import time

# 机器学习相关
from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split,
    cross_val_score,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    make_scorer,
    log_loss,
)
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

# NODE模型
try:
    from mambular.models import NODEClassifier
    from mambular.configs import DefaultNODEConfig
    
    MAMBULAR_AVAILABLE = True
    print("✅ Mambular可用，启用NODE模型")
except ImportError as e:
    print(f"❌ Mambular模型不可用: {e}")
    MAMBULAR_AVAILABLE = False

# 贝叶斯优化
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner

    OPTUNA_AVAILABLE = True
    print("✅ Optuna可用，启用贝叶斯优化")
except ImportError:
    print("❌ Optuna未安装，无法使用贝叶斯优化")
    OPTUNA_AVAILABLE = False

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局参数缓存机制
CACHE_FILE = "node_parameter_cache.pkl"
PARAMETER_CACHE = {}


def load_parameter_cache():
    """加载参数缓存"""
    global PARAMETER_CACHE
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "rb") as f:
                PARAMETER_CACHE = pickle.load(f)
            print(
                f"已加载NODE参数缓存，包含 {len(PARAMETER_CACHE)} 个已评估的参数组合"
            )
    except Exception as e:
        print(f"加载NODE缓存失败: {e}")
        PARAMETER_CACHE = {}


def save_parameter_cache():
    """保存参数缓存"""
    try:
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(PARAMETER_CACHE, f)
    except Exception as e:
        print(f"保存NODE缓存失败: {e}")


def get_cache_key(X_train, y_train, params):
    """生成缓存键"""
    # 使用数据和参数的哈希值作为键
    data_hash = hashlib.md5(
        str(X_train.shape).encode() + str(np.sum(y_train)).encode()
    ).hexdigest()[:8]
    param_hash = hashlib.md5(str(sorted(params.items())).encode()).hexdigest()[:8]
    return f"node_{data_hash}_{param_hash}"


def check_cache(X_train, y_train, params):
    """检查缓存中是否存在该参数组合的结果"""
    cache_key = get_cache_key(X_train, y_train, params)
    return PARAMETER_CACHE.get(cache_key)


def update_cache(X_train, y_train, params, score):
    """更新缓存"""
    cache_key = get_cache_key(X_train, y_train, params)
    PARAMETER_CACHE[cache_key] = {
        "score": score,
        "timestamp": time.time(),
        "params": params.copy(),
    }


# 动态并行策略配置
def get_dynamic_parallel_config(
    data_size, search_type="bayesian", model_name="NODE"
):
    """根据数据规模和搜索类型动态配置并行策略"""
    cpu_count = multiprocessing.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)  # 获取内存大小

    # 检查是否在多模型并行环境中运行
    import os

    is_multi_model_env = (
        os.environ.get("MULTI_MODEL_PARALLEL", "false").lower() == "true"
    )
    allocated_cores = int(os.environ.get("ALLOCATED_CORES", cpu_count))

    if is_multi_model_env:
        # 在多模型并行环境中，使用分配的核心数
        available_cores = allocated_cores
        print(f"🔄 多模型并行环境: {model_name} 分配到 {available_cores} 个核心")
    else:
        available_cores = cpu_count

    # 优先保证模型训练效率，限制搜索并行度
    model_n_jobs = max(1, min(available_cores - 1, available_cores // 2))
    search_n_jobs = 1  # 搜索单线程避免冲突
    cv_n_jobs = 1  # CV单线程
    print(
        f"🚀 NODE模型并行配置: {model_name}={model_n_jobs}核心, 搜索=1核心, CV=1核心"
    )

    return model_n_jobs, search_n_jobs, cv_n_jobs


# 加载缓存
load_parameter_cache()


def NODE_model(
    X_train,
    X_test,
    y_train,
    y_test,
    X_val=None,
    y_val=None,
    class_weights=None,
    target=None,
    kinds=None,
    use_early_stopping=True,
    n_iter=50,
    cv_folds=5,
    use_validation_in_training=True,
    base_path=None,
    use_layered_optimization=True,
    early_stopping_patience=10,
):
    """
    训练和评估优化的NODE模型（标准化接口，与LightGBM保持一致）

    Parameters:
    - X_train, X_test, X_val: 训练、测试和验证数据
    - y_train, y_test, y_val: 训练、测试和验证标签
    - class_weights: 处理类别不平衡的类权重
    - target: 目标变量名称
    - kinds: 模型类型标识
    - use_early_stopping: 是否使用早停机制
    - n_iter: 贝叶斯优化的迭代次数
    - cv_folds: 交叉验证折数
    - use_validation_in_training: 是否在训练中使用验证集
    - base_path: 结果保存的基础路径
    - use_layered_optimization: 是否使用分层优化（粗搜索+细搜索）
    - early_stopping_patience: 早停的耐心值

    Returns:
    - dict: 包含模型、评估分数、最佳参数等的完整结果字典
    """

    if not MAMBULAR_AVAILABLE:
        raise ImportError("Mambular库不可用，请安装 mambular")

    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna库不可用，请安装 optuna")

    logger.info(f"开始训练NODE模型 - Target: {target}, Kinds: {kinds}")
    logger.info(
        f"训练集大小: {X_train.shape}, 验证集大小: {X_val.shape}, 测试集大小: {X_test.shape}"
    )
    # 处理验证集数据
    if use_validation_in_training and X_val is not None and y_val is not None:
        X_train_combined = pd.concat([pd.DataFrame(X_train), pd.DataFrame(X_val)], ignore_index=True)
        y_train_combined = pd.concat([pd.Series(y_train), pd.Series(y_val)], ignore_index=True)
        logger.info(f"使用验证集参与训练，合并后训练集大小: {X_train_combined.shape}")
    else:
        X_train_combined = pd.DataFrame(X_train)
        y_train_combined = pd.Series(y_train)
        logger.info(f"仅使用原始训练集，大小: {X_train_combined.shape}")
    # 保证测试集格式
    X_test_df = pd.DataFrame(X_test)
    y_test_series = pd.Series(y_test)
    # 标签类型强制转int
    y_train_combined = y_train_combined.astype(int)
    y_test_series = y_test_series.astype(int)
    # 特征列名处理
    if hasattr(X_train, 'columns'):
        X_train_combined.columns = X_train.columns
        X_test_df.columns = X_train.columns
    # 重置索引，防止索引错乱
    X_train_combined.reset_index(drop=True, inplace=True)
    X_test_df.reset_index(drop=True, inplace=True)
    y_train_combined.reset_index(drop=True, inplace=True)
    y_test_series.reset_index(drop=True, inplace=True)
    # 数据类型处理 - 避免mambular预处理器问题
    # 将所有数据转换为数值类型，避免category类型导致的KeyError(0)
    for col in X_train_combined.columns:
        if X_train_combined[col].dtype == 'object' or str(X_train_combined[col].dtype).startswith('category'):
            # 使用LabelEncoder进行编码，避免category类型
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            # 合并训练和测试数据进行编码，确保一致性
            combined_values = pd.concat([X_train_combined[col], X_test_df[col]], ignore_index=True)
            le.fit(combined_values.astype(str))
            X_train_combined[col] = le.transform(X_train_combined[col].astype(str))
            X_test_df[col] = le.transform(X_test_df[col].astype(str))
    
    # 确保所有列都是数值类型
    X_train_combined = X_train_combined.astype(float)
    X_test_df = X_test_df.astype(float)
    
    # 数据标准化 - 重要：防止数值范围问题
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_combined),
        columns=X_train_combined.columns,
        index=X_train_combined.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_df),
        columns=X_test_df.columns,
        index=X_test_df.index
    )
    
    logger.info(f"数据预处理完成: 训练集{X_train_scaled.shape}, 测试集{X_test_scaled.shape}")
    best_score = 0
    best_params = None
    best_model = None
    optimization_history = []
    def objective(trial):
        nonlocal best_score, best_params, best_model
        
        # 定义超参数搜索空间
        params = {
            "num_layers": trial.suggest_int("num_layers", 2, 6),
            "layer_dim": trial.suggest_int("layer_dim", 64, 256),
            "tree_dim": trial.suggest_int("tree_dim", 1, 4),
            "depth": trial.suggest_int("depth", 3, 8),
            "lr": trial.suggest_float("lr", 0.0001, 0.01, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "head_dropout": trial.suggest_float("head_dropout", 0.1, 0.5),
        }
        
        # 检查缓存
        cached_result = check_cache(X_train_scaled, y_train_combined, params)
        if cached_result is not None:
            logger.info(f"使用缓存结果: {cached_result['score']:.4f}")
            return -cached_result["score"]
        
        try:
            # 交叉验证
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            cv_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_scaled, y_train_combined)):
                X_fold_train = X_train_scaled.iloc[train_idx]
                X_fold_val = X_train_scaled.iloc[val_idx]
                y_fold_train = y_train_combined.iloc[train_idx]
                y_fold_val = y_train_combined.iloc[val_idx]
                
                # 创建NODE模型（禁用预处理器）
                model = NODEClassifier(
                    num_layers=params["num_layers"],
                    layer_dim=params["layer_dim"],
                    tree_dim=params["tree_dim"],
                    depth=params["depth"],
                    lr=params["lr"],
                    weight_decay=params["weight_decay"],
                    head_dropout=params["head_dropout"],
                    use_embeddings=False,  # 禁用嵌入层避免预处理器问题
                    cat_encoding=None,     # 禁用类别编码
                )
                
                # 手动禁用预处理器
                model.preprocessor = None
                
                # 训练模型
                model.fit(X_fold_train, y_fold_train)
                
                # 预测和评估
                y_pred_proba_raw = model.predict_proba(X_fold_val)
                
                # 处理概率输出维度问题
                if y_pred_proba_raw.shape[1] == 1:
                    y_pred_proba = y_pred_proba_raw[:, 0]
                else:
                    y_pred_proba = y_pred_proba_raw[:, 1]
                
                score = roc_auc_score(y_fold_val, y_pred_proba)
                cv_scores.append(score)
                
                logger.info(f"Fold {fold + 1}/{cv_folds}: AUC = {score:.4f}")
            
            mean_score = np.mean(cv_scores)
            std_score = np.std(cv_scores)
            
            # 更新缓存
            update_cache(X_train_scaled, y_train_combined, params, mean_score)
            
            # 记录优化历史
            optimization_history.append({
                "trial": trial.number,
                "params": params.copy(),
                "mean_score": mean_score,
                "std_score": std_score,
                "cv_scores": cv_scores,
            })
            
            # 更新最佳结果
            if mean_score > best_score:
                best_score = mean_score
                best_params = params.copy()
                logger.info(f"🎯 新的最佳分数: {mean_score:.4f} (±{std_score:.4f})")
            
            return -mean_score  # Optuna最小化目标
            
        except Exception as e:
            logger.error(f"Trial {trial.number} 失败: {e}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            return float('inf')  # 返回最差分数
    study = optuna.create_study(direction="minimize", sampler=TPESampler(), pruner=MedianPruner())
    study.optimize(objective, n_trials=n_iter)
    # 用最佳参数训练最终模型
    try:
        logger.info("训练最终模型...")
        final_model = NODEClassifier(
            num_layers=best_params["num_layers"],
            layer_dim=best_params["layer_dim"],
            tree_dim=best_params["tree_dim"],
            depth=best_params["depth"],
            lr=best_params["lr"],
            weight_decay=best_params["weight_decay"],
            head_dropout=best_params["head_dropout"],
            # 添加关键参数避免预处理器问题
            use_embeddings=False,  # 禁用嵌入层
            cat_encoding=None,     # 禁用类别编码
        )
        
        # 手动禁用预处理器
        final_model.preprocessor = None
        final_model.fit(X_train_scaled, y_train_combined)
        
        # 预测
        y_pred = final_model.predict(X_test_scaled)
        y_pred_proba_raw = final_model.predict_proba(X_test_scaled)
        
        # 处理概率输出维度问题
        if y_pred_proba_raw.shape[1] == 1:
            y_pred_proba = y_pred_proba_raw[:, 0]
        else:
            y_pred_proba = y_pred_proba_raw[:, 1]
            
    except Exception as e:
        logger.error(f"最终NODE模型训练失败: {e}")
        import traceback
        logger.error(f"详细错误信息: {traceback.format_exc()}")
        
        return {
            "model": None,
            "metrics": None,
            "error": str(e),
            "status": "failed"
        }
    # 计算评估指标
    metrics = {
        "accuracy": accuracy_score(y_test_series, y_pred),
        "precision": precision_score(y_test_series, y_pred, zero_division=0),
        "recall": recall_score(y_test_series, y_pred, zero_division=0),
        "f1": f1_score(y_test_series, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test_series, y_pred_proba),
        "confusion_matrix": confusion_matrix(y_test_series, y_pred).tolist(),
        "classification_report": classification_report(y_test_series, y_pred, output_dict=True),
        "log_loss": log_loss(y_test_series, y_pred_proba),
    }
    
    # 输出结果
    logger.info("=== 最终模型评估结果 ===")
    logger.info(f"准确率: {metrics['accuracy']:.4f}")
    logger.info(f"精确率: {metrics['precision']:.4f}")
    logger.info(f"召回率: {metrics['recall']:.4f}")
    logger.info(f"F1分数: {metrics['f1']:.4f}")
    logger.info(f"AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"对数损失: {metrics['log_loss']:.4f}")
    
    result = {
        "model": final_model,
        "scaler": scaler,
        "best_params": best_params,
        "best_score": best_score,
        "metrics": metrics,
        "optimization_history": optimization_history,
        "status": "success"
    }
    
    # 保存模型
    if base_path is not None:
        os.makedirs(base_path, exist_ok=True)
        model_path = os.path.join(base_path, f"NODE_{target}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
        joblib.dump(final_model, model_path)
        result["model_path"] = model_path
        logger.info(f"模型已保存到: {model_path}")
    
    return result


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    # 生成测试数据
    X, y = make_classification(
        n_samples=1000, n_features=20, n_classes=2, random_state=42
    )
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    # 训练模型
    result = NODE_model(
        X_train, X_test, y_train, y_test, X_val, y_val,
        class_weights=None, target="test", kinds="binary",
        n_iter=5, cv_folds=3, use_validation_in_training=True
    )
    if result and result["status"] == "success":
        print(f"最佳AUC: {result['metrics']['roc_auc']:.4f}")
        print(f"最佳参数: {result['best_params']}")
