#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ResNet模型训练脚本（标准化版本）
适配透析患者临床数据分析与建模

作者: 透析数据分析团队
日期: 2024
版本: 2.0 - 标准化接口版本
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

# 机器学习相关
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, confusion_matrix, classification_report,
    make_scorer
)

# Mambular模型
try:
    from mambular.models import ResNetClassifier
    from mambular.configs import DefaultResNetConfig
    
    MAMBULAR_AVAILABLE = True
    print("✅ Mambular可用，启用ResNet模型")
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


def get_dynamic_parallel_config(data_size: int, optimization_method: str = "bayesian") -> Tuple[int, int, int]:
    """
    根据数据规模动态配置并行参数
    
    Args:
        data_size: 数据样本数量
        optimization_method: 优化方法
    
    Returns:
        Tuple[model_n_jobs, search_n_jobs, cv_n_jobs]: 模型、搜索、交叉验证的并行数
    """
    import multiprocessing
    
    total_cores = multiprocessing.cpu_count()
    
    if data_size < 10000:
        # 小数据集：更多并行用于搜索
        model_n_jobs = min(4, total_cores // 2)
        search_n_jobs = min(2, total_cores // 4)
        cv_n_jobs = 1
    elif data_size < 100000:
        # 中等数据集：平衡配置
        model_n_jobs = min(8, total_cores // 2)
        search_n_jobs = 1
        cv_n_jobs = 1
    else:
        # 大数据集：更多资源给模型训练
        model_n_jobs = min(12, total_cores)
        search_n_jobs = 1
        cv_n_jobs = 1
    
    return model_n_jobs, search_n_jobs, cv_n_jobs





def ResNet_model(
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
    early_stopping_patience=10,
    random_state=42,
    n_jobs=None,
    save_path=None,
    model_name="ResNet",
    **kwargs,
):
    """
    训练和评估ResNet模型（标准化接口）

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
    - early_stopping_patience: 早停的耐心值

    Returns:
    - dict: 包含模型、评估分数、最佳参数等的完整结果字典
    """

    if not MAMBULAR_AVAILABLE:
        raise ImportError("Mambular库不可用，请安装 mambular")

    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna库不可用，请安装 optuna")

    logger.info(f"开始训练ResNet模型 - Target: {target}, Kinds: {kinds}")
    logger.info(
        f"训练集大小: {X_train.shape}, 验证集大小: {X_val.shape}, 测试集大小: {X_test.shape}"
    )

    # 确定是否为多分类问题
    is_multiclass = len(class_weights) > 2
    n_classes = len(class_weights) if is_multiclass else 2

    # 使用动态并行策略配置
    data_size = X_train.shape[0] if hasattr(X_train, "shape") else 200000
    model_n_jobs, search_n_jobs, cv_n_jobs = get_dynamic_parallel_config(
        data_size, "bayesian"
    )

    # 准备训练数据 - 转换为DataFrame格式避免预处理器问题
    if use_validation_in_training and X_val is not None and y_val is not None:
        # 合并训练集和验证集
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

    # 交叉验证设置
    cv_strategy = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True, 
        random_state=42
    )

    # 定义评估指标
    if is_multiclass:
        scoring = {
            "f1_weighted": make_scorer(f1_score, average="weighted"),
            "recall_weighted": make_scorer(recall_score, average="weighted"),
            "roc_auc_ovr": make_scorer(
                roc_auc_score,
                response_method="predict_proba",
                multi_class="ovr",
                average="weighted",
            ),
            "neg_log_loss": make_scorer(
                log_loss, response_method="predict_proba", greater_is_better=False
            ),
        }
        refit_metric = "f1_weighted"
    else:
        scoring = {
            "f1_weighted": make_scorer(f1_score, average="weighted"),
            "roc_auc": make_scorer(roc_auc_score, response_method="predict_proba"),
            "neg_log_loss": make_scorer(
                log_loss, response_method="predict_proba", greater_is_better=False
            ),
        }
        refit_metric = "f1_weighted"

    # 定义Optuna优化目标函数
    def objective(trial):
        """
        Optuna优化目标函数
        """
        # 超参数搜索空间
        params = {
            'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
            'd_model': trial.suggest_categorical('d_model', [64, 128, 256, 512]),
            'num_blocks': trial.suggest_int('num_blocks', 2, 8),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
            'epochs': trial.suggest_int('epochs', 50, 200),
        }

        try:
            # 创建模型配置
             config = DefaultResNetConfig(
                 lr=params["lr"],
                 weight_decay=params["weight_decay"],
                 d_model=params["d_model"],
                 num_blocks=params["num_blocks"],
                 dropout=params["dropout"],
             )
             
             # 创建模型 - 禁用预处理器避免KeyError(0)
             try:
                 model = ResNetClassifier(
                     **config.__dict__,
                     use_embeddings=False,  # 禁用嵌入层
                     cat_encoding=None,     # 禁用类别编码
                     layer_sizes=[256, 128, 32],  # 确保layer_sizes不为空
                 )
             except Exception as e:
                 logger.error(f"创建ResNet模型失败: {e}")
                 # 使用最简配置重试
                 model = ResNetClassifier(
                     lr=0.001,
                     d_model=128,
                     num_blocks=3,
                     dropout=0.1,
                     use_embeddings=False,
                     cat_encoding=None,
                     layer_sizes=[256, 128, 32],  # 确保layer_sizes不为空
                 )

             # 执行交叉验证
             cv_scores = cross_val_score(
                model, X_train_scaled, y_train_combined,
                cv=cv_strategy,
                scoring=refit_metric,
                n_jobs=1  # 避免嵌套并行
            )

             return np.mean(cv_scores)

        except Exception as e:
            logger.warning(f"试验失败: {e}")
            return 0.0

    # 执行贝叶斯优化
    logger.info(f"使用Optuna贝叶斯优化，迭代次数: {n_iter}")

    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )

    study.optimize(objective, n_trials=n_iter)

    # 获取最佳参数
    best_params = study.best_params
    best_score = study.best_value

    logger.info(f"最佳交叉验证分数 ({refit_metric}): {best_score:.4f}")
    logger.info(f"最佳参数: {best_params}")

    # 使用最佳参数训练最终模型
    logger.info("使用最佳参数训练最终模型...")

    final_config = DefaultResNetConfig(
         lr=best_params['lr'],
         weight_decay=best_params['weight_decay'],
         d_model=best_params['d_model'],
         num_blocks=best_params['num_blocks'],
         dropout=best_params['dropout'],
     )
    
    # 创建最终模型配置字典，避免重复参数
    final_model_config = final_config.__dict__.copy()
    final_model_config.update({
        'use_embeddings': False,  # 禁用嵌入层
        'cat_encoding': None,     # 禁用类别编码
        'layer_sizes': [256, 128, 32],  # 确保layer_sizes不为空
    })
    
    final_model = ResNetClassifier(**final_model_config)

    # 训练最终模型
    if use_early_stopping and X_val is not None and y_val is not None:
        # 对验证集也进行标准化
        X_val_scaled = pd.DataFrame(
            scaler.transform(X_val),
            columns=X_val.columns if hasattr(X_val, 'columns') else X_train_scaled.columns,
            index=X_val.index if hasattr(X_val, 'index') else range(len(X_val))
        )
        final_model.fit(X_train_scaled, y_train_combined, X_val=X_val_scaled, y_val=y_val)
    else:
        final_model.fit(X_train_scaled, y_train_combined)

    # 📊 全面的模型评估
    logger.info("开始模型评估...")

    # 预测
    y_train_pred = final_model.predict(X_train_scaled)
    y_val_pred = final_model.predict(X_val_scaled) if X_val is not None else None
    y_test_pred = final_model.predict(X_test_scaled)

    y_train_proba_raw = final_model.predict_proba(X_train_scaled)
    y_val_proba_raw = final_model.predict_proba(X_val_scaled) if X_val is not None else None
    y_test_proba_raw = final_model.predict_proba(X_test_scaled)
    
    # 处理概率输出维度问题
    if y_train_proba_raw.shape[1] == 1:
        y_train_proba = y_train_proba_raw[:, 0]
    else:
        y_train_proba = y_train_proba_raw[:, 1]
        
    if y_val_proba_raw is not None:
        if y_val_proba_raw.shape[1] == 1:
            y_val_proba = y_val_proba_raw[:, 0]
        else:
            y_val_proba = y_val_proba_raw[:, 1]
    else:
        y_val_proba = None
        
    if y_test_proba_raw.shape[1] == 1:
        y_test_proba = y_test_proba_raw[:, 0]
    else:
        y_test_proba = y_test_proba_raw[:, 1]

    # 过拟合分析
    logger.info("进行过拟合分析...")

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')

    train_test_gap = train_accuracy - test_accuracy
    train_test_f1_gap = train_f1 - test_f1

    # 过拟合风险评估
    overfitting_risk = "低"
    if train_test_gap > 0.1 or train_test_f1_gap > 0.1:
        overfitting_risk = "高"
    elif train_test_gap > 0.05 or train_test_f1_gap > 0.05:
        overfitting_risk = "中"

    logger.info(f"过拟合风险评估: {overfitting_risk}")
    logger.info(f"训练-测试准确率差异: {train_test_gap:.4f}")
    logger.info(f"训练-测试F1差异: {train_test_f1_gap:.4f}")

    # 计算各种评估指标
    def calculate_metrics(y_true, y_pred, y_proba, dataset_name):
        metrics = {
            f"{dataset_name}_accuracy": accuracy_score(y_true, y_pred),
            f"{dataset_name}_precision_weighted": precision_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            f"{dataset_name}_recall_weighted": recall_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            f"{dataset_name}_f1_weighted": f1_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
        }

        if is_multiclass:
            metrics.update(
                {
                    f"{dataset_name}_precision_macro": precision_score(
                        y_true, y_pred, average="macro"
                    ),
                    f"{dataset_name}_recall_macro": recall_score(
                        y_true, y_pred, average="macro"
                    ),
                    f"{dataset_name}_f1_macro": f1_score(
                        y_true, y_pred, average="macro"
                    ),
                    f"{dataset_name}_roc_auc_ovr": roc_auc_score(
                        y_true, y_proba, multi_class="ovr", average="weighted"
                    ),
                }
            )
        else:
            metrics.update(
                {
                    f"{dataset_name}_precision": precision_score(y_true, y_pred),
                    f"{dataset_name}_recall": recall_score(y_true, y_pred),
                    f"{dataset_name}_f1": f1_score(y_true, y_pred),
                    f"{dataset_name}_roc_auc": roc_auc_score(y_true, y_proba),
                }
            )

        return metrics

    # 计算所有数据集的指标
    train_metrics = calculate_metrics(y_train, y_train_pred, y_train_proba, "train")
    val_metrics = calculate_metrics(y_val, y_val_pred, y_val_proba, "validation")
    test_metrics = calculate_metrics(y_test, y_test_pred, y_test_proba, "test")

    # 合并所有指标
    all_metrics = {**train_metrics, **val_metrics, **test_metrics}

    # 打印关键指标
    logger.info("=== 模型性能评估 ===")
    logger.info(f"训练集准确率: {train_metrics['train_accuracy']:.4f}")
    logger.info(f"验证集准确率: {val_metrics['validation_accuracy']:.4f}")
    logger.info(f"测试集准确率: {test_metrics['test_accuracy']:.4f}")

    logger.info(f"训练集F1分数: {train_metrics['train_f1_weighted']:.4f}")
    logger.info(f"验证集F1分数: {val_metrics['validation_f1_weighted']:.4f}")
    logger.info(f"测试集F1分数: {test_metrics['test_f1_weighted']:.4f}")

    # 打印混淆矩阵
    logger.info("\n=== 混淆矩阵 ===")
    logger.info(f"验证集混淆矩阵:\n{confusion_matrix(y_val, y_val_pred)}")
    logger.info(f"测试集混淆矩阵:\n{confusion_matrix(y_test, y_test_pred)}")

    # 打印详细分类报告
    logger.info("\n=== 验证集分类报告 ===")
    logger.info(f"\n{classification_report(y_val, y_val_pred)}")

    logger.info("\n=== 测试集分类报告 ===")
    logger.info(f"\n{classification_report(y_test, y_test_pred)}")

    # 创建结果保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if base_path is not None:
        results_dir = os.path.join(base_path, f"ResNet_{target}_{timestamp}")
    else:
        # 获取调用者的目录作为基础路径
        import inspect

        caller_frame = inspect.currentframe().f_back
        caller_file = caller_frame.f_globals.get("__file__")
        if caller_file:
            caller_dir = os.path.dirname(os.path.abspath(caller_file))
            results_dir = os.path.join(
                caller_dir, "Results", f"ResNet_{target}_{timestamp}"
            )
        else:
            results_dir = f"./Results/ResNet_{target}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    logger.info(f"结果将保存到: {results_dir}")

    # 生成总结报告
    summary_report = f"""
🎯 ResNet模型训练完成报告
{'='*60}
📊 数据信息:
   - 训练样本数: {len(X_train):,}
   - 验证样本数: {len(X_val) if X_val is not None else 0:,}
   - 测试样本数: {len(X_test):,}
   - 特征数量: {X_train.shape[1]:,}
   - 目标类别: {len(np.unique(y_train))}

🔧 模型配置:
   - 优化方法: Optuna贝叶斯优化
   - 交叉验证折数: {cv_folds}
   - 早停耐心值: {early_stopping_patience}
   - 学习率: {best_params['lr']:.6f}
   - 权重衰减: {best_params['weight_decay']:.6f}
   - 模型维度: {best_params['d_model']}
   - 网络层数: {best_params['num_blocks']}
   - Dropout率: {best_params['dropout']:.3f}
   - 批次大小: {best_params['batch_size']}
   - 训练轮数: {best_params['epochs']}

📈 性能指标:
   - 交叉验证分数: {best_score:.4f}
   - 训练集准确率: {train_accuracy:.4f}
   - 验证集准确率: {val_metrics['validation_accuracy']:.4f}
   - 测试集准确率: {test_metrics['test_accuracy']:.4f}
   - 测试集F1分数: {test_metrics['test_f1_weighted']:.4f}

🔍 过拟合分析:
   - 风险等级: {overfitting_risk}
   - 训练-测试准确率差异: {train_test_gap:.4f}
   - 训练-测试F1差异: {train_test_f1_gap:.4f}

💾 保存路径:
   - 结果目录: {results_dir}
   - 模型文件: best_model.pkl
   - 参数文件: best_params.json
   - 详细结果: evaluation_results.json
   - 优化历史: optimization_history.json

{'='*60}
"""

    logger.info(summary_report)

    # 保存模型和结果
    model_path = os.path.join(results_dir, "best_model.pkl")
    best_params_path = os.path.join(results_dir, "best_params.json")
    results_path = os.path.join(results_dir, "evaluation_results.json")
    summary_path = os.path.join(results_dir, "model_summary.txt")
    history_path = os.path.join(results_dir, "optimization_history.json")

    # 保存模型
    joblib.dump(final_model, model_path)

    # 保存最佳参数
    with open(best_params_path, 'w') as f:
        json.dump(best_params, f, indent=2)

    # 保存评估结果
    with open(results_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)

    # 保存模型摘要
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_report)
        f.write(f"\n\nDetailed Parameters:\n")
        f.write(f"Best Parameters: {best_params}\n")
        f.write(f"\nDetailed Test Metrics:\n")
        for key, value in test_metrics.items():
            f.write(f"{key}: {value:.4f}\n")

    # 保存优化历史
    optimization_history = {
        "trials": [
            {
                "number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "state": trial.state.name
            }
            for trial in study.trials
        ],
        "best_trial": {
            "number": study.best_trial.number,
            "value": study.best_value,
            "params": study.best_params
        }
    }

    with open(history_path, 'w') as f:
        json.dump(optimization_history, f, indent=2)

    # 返回结果字典
    return_dict = {
        "model": final_model,
        "best_params": best_params,
        "cv_score": best_score,
        "optimization_method": "optuna_bayesian",
        "bayesian_available": OPTUNA_AVAILABLE,
        "train_metrics": train_metrics,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "results_dir": results_dir,
        "model_path": model_path,
        "best_params_path": best_params_path,
        "results_path": results_path,
        "summary_path": summary_path,
        "optimization_history_path": history_path,
        "overfitting_analysis": {
            "risk_level": overfitting_risk,
            "accuracy_gap": train_test_gap,
            "f1_gap": train_test_f1_gap,
        },
        "summary_report": summary_report,
    }

    if not is_multiclass:
        return_dict["roc_auc"] = {
            "train": roc_auc_score(y_train, y_train_proba),
            "validation": roc_auc_score(y_val, y_val_proba),
            "test": roc_auc_score(y_test, y_test_proba),
        }
    
    # 添加scaler和status字段
    return_dict["scaler"] = scaler
    return_dict["status"] = "success"

    return return_dict


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
    result = ResNet_model(
        X_train, X_test, y_train, y_test, X_val, y_val,
        class_weights=None, target="test", kinds="binary",
        n_iter=5, cv_folds=3, use_validation_in_training=True
    )
    if result and result["status"] == "success":
        print(f"最佳测试AUC: {result['roc_auc']['test']:.4f}")
        print(f"最佳参数: {result['best_params']}")
    print("ResNet模型脚本已加载，可以通过函数调用使用")
    print(f"Mambular可用: {MAMBULAR_AVAILABLE}")
    print(f"Optuna可用: {OPTUNA_AVAILABLE}")
