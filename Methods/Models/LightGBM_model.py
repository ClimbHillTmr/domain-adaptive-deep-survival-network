import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold, cross_val_score,train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from datetime import datetime
import logging
import warnings
warnings.filterwarnings('ignore')

# 尝试导入贝叶斯优化库
try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer, Categorical
    BAYESIAN_AVAILABLE = True
except ImportError:
    print("Warning: scikit-optimize not available. Falling back to RandomizedSearchCV.")
    BAYESIAN_AVAILABLE = False

import pickle
import json
from sklearn.metrics import (
    f1_score,
    make_scorer,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc,
    confusion_matrix,
    precision_score,
    accuracy_score,
    classification_report,
    log_loss
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils.class_weight import compute_class_weight
from pylab import mpl
from yellowbrick.target import ClassBalance
from yellowbrick.classifier import ROCAUC, PrecisionRecallCurve, ClassificationReport, ClassPredictionError, DiscriminationThreshold, ConfusionMatrix

mpl.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
## mac

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def LightGBM_model(
    X_train, X_test, y_train, y_test, X_val, y_val, class_weights, target, kinds,
    use_early_stopping=True, optimization_method='bayesian', n_iter=100, cv_folds=5,
    use_validation_in_training=True, base_path=None
):
    """
    训练和评估优化的LightGBM模型

    Parameters:
    - X_train, X_test, X_val: 训练、测试和验证数据
    - y_train, y_test, y_val: 训练、测试和验证标签
    - class_weights: 处理类别不平衡的类权重
    - target: 目标变量名称
    - kinds: 模型类型标识
    - use_early_stopping: 是否使用早停机制
    - optimization_method: 优化方法 ('bayesian', 'random', 'grid')
    - n_iter: 贝叶斯优化或随机搜索的迭代次数
    - cv_folds: 交叉验证折数

    Returns:
    - dict: 包含模型、评估分数、最佳参数等的完整结果字典
    """
    
    logger.info(f"开始训练LightGBM模型 - Target: {target}, Kinds: {kinds}")
    logger.info(f"训练集大小: {X_train.shape}, 验证集大小: {X_val.shape}, 测试集大小: {X_test.shape}")

    # 确定是否为多分类问题
    is_multiclass = len(class_weights) > 2
    n_classes = len(class_weights) if is_multiclass else 2
    
    # 基础模型参数
    base_params = {
        'n_jobs': -1,
        'random_state': 42,
        'verbose': -1,
        'class_weight': class_weights,
        'importance_type': 'gain'
    }
    
    # 根据问题类型设置目标函数
    if is_multiclass:
        base_params.update({
            'objective': 'multiclass',
            'num_class': n_classes,
            'metric': 'multi_logloss'
        })
    else:
        base_params.update({
            'objective': 'binary',
            'metric': 'binary_logloss'
        })
    
    # 创建基础模型
    model = lgb.LGBMClassifier(**base_params)

    # 定义扩展的超参数搜索空间
    if optimization_method == 'bayesian' and BAYESIAN_AVAILABLE:
        # 贝叶斯优化参数空间（连续和离散空间的组合）
        # 为了避免GOSS与bagging冲突，简化参数空间
        param_space = {
            'num_leaves': [15, 31, 50, 100, 150, 200, 300, 400, 500],
            'max_depth': [-1, 3, 5, 7, 10, 15, 20, 30, 40, 50],
            'learning_rate': [0.0001, 0.001, 0.005, 0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5],
            'n_estimators': [50, 100, 200, 300, 500, 800, 1000, 1500, 2000],
            'subsample': [0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0],
            'colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0],
            'reg_alpha': [0, 0.0001, 0.001, 0.01, 0.03, 0.08, 0.1, 0.3, 0.5, 1.0, 10.0, 100.0],
            'reg_lambda': [0, 0.0001, 0.001, 0.01, 0.03, 0.08, 0.1, 0.3, 0.5, 1.0, 10.0, 100.0],
            'min_child_samples': [1, 5, 10, 15, 20, 25, 30, 50, 100, 200],
            'min_child_weight': [0.0001, 0.001, 0.01, 0.1, 1, 5, 10, 50],
            'min_split_gain': [0.0, 0.001, 0.01, 0.1, 0.2, 0.5, 1.0, 2.0],
            'subsample_freq': [0, 1, 2, 3, 5, 7, 10],
            'feature_fraction': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'bagging_fraction': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'bagging_freq': [0, 1, 2, 3, 5, 7, 10],
            'max_bin': [63, 127, 255, 511, 1023],
            'min_data_in_leaf': [1, 5, 10, 15, 20, 30, 50, 100, 200],
            'lambda_l1': [0, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            'lambda_l2': [0, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        }
        param_dist = param_space
    elif optimization_method == 'random':
        # 随机搜索参数空间（离散值列表）
        param_dist = {
            'boosting_type': ['gbdt', 'dart'],  # 移除goss避免冲突
            'num_leaves': [15, 31, 50, 100, 150, 200, 300, 400, 500],
            'max_depth': [-1, 3, 5, 7, 10, 15, 20, 30, 40, 50],
            'learning_rate': [0.0001, 0.001, 0.005, 0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5],
            'n_estimators': [50, 100, 200, 300, 500, 800, 1000, 1500, 2000],
            'subsample': [0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0],
            'colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0],
            'reg_alpha': [0, 0.0001, 0.001, 0.01, 0.03, 0.08, 0.1, 0.3, 0.5, 1.0, 10.0, 100.0],
            'reg_lambda': [0, 0.0001, 0.001, 0.01, 0.03, 0.08, 0.1, 0.3, 0.5, 1.0, 10.0, 100.0],
            'min_child_samples': [1, 5, 10, 15, 20, 25, 30, 50, 100, 200],
            'min_child_weight': [0.0001, 0.001, 0.01, 0.1, 1, 5, 10, 50],
            'min_split_gain': [0.0, 0.001, 0.01, 0.1, 0.2, 0.5, 1.0, 2.0],
            'subsample_freq': [0, 1, 2, 3, 5, 7, 10],
            'feature_fraction': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'bagging_fraction': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'bagging_freq': [0, 1, 2, 3, 5, 7, 10],
            'max_bin': [63, 127, 255, 511, 1023],
            'min_data_in_leaf': [1, 5, 10, 15, 20, 30, 50, 100, 200],
            'lambda_l1': [0, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            'lambda_l2': [0, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        }
    else:
        # 网格搜索参数空间（较小但精确的搜索范围）
        param_dist = {
            'boosting_type': ['gbdt'],
            # 'num_leaves': [31, 50, 100, 200],
            # 'max_depth': [7, 15, 30],
            # 'learning_rate': [0.01, 0.05, 0.1,],
            # 'n_estimators': [100, 300, 500],
            # 'subsample': [0.7, 0.8, 0.9, 1.0],
            # 'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            # 'reg_alpha': [0, 0.1, 0.5, 1.0],
            # 'reg_lambda': [0, 0.1, 0.5, 1.0],
            # 'min_child_samples': [10, 20, 50, 100]
        }

    # 定义更全面的评估指标
    if is_multiclass:
        scoring = {
            # 'accuracy': make_scorer(accuracy_score),
            # 'f1_macro': make_scorer(f1_score, average='macro'),
            'f1_weighted': make_scorer(f1_score, average='weighted'),
            # 'f1_micro': make_scorer(f1_score, average='micro'),
            # 'precision_macro': make_scorer(precision_score, average='macro'),
            # 'precision_weighted': make_scorer(precision_score, average='weighted'),
            # 'recall_macro': make_scorer(recall_score, average='macro'),
            'recall_weighted': make_scorer(recall_score, average='weighted'),
            'roc_auc_ovr': make_scorer(roc_auc_score, response_method='predict_proba', 
                                     multi_class='ovr', average='weighted'),
            # 'neg_log_loss': make_scorer(log_loss, response_method='predict_proba', greater_is_better=False)
        }
        refit_metric = 'f1_weighted'
    else:
        scoring = {
            # 'accuracy': make_scorer(accuracy_score),
            # 'f1': make_scorer(f1_score),
            'f1_weighted': make_scorer(f1_score, average='weighted'),
            # 'precision': make_scorer(precision_score),
            # 'recall': make_scorer(recall_score),
            'roc_auc': make_scorer(roc_auc_score, response_method='predict_proba'),
            # 'neg_log_loss': make_scorer(log_loss, response_method='predict_proba', greater_is_better=False)
        }
        refit_metric = 'f1_weighted'

    # 选择搜索策略
    if optimization_method == 'bayesian' and BAYESIAN_AVAILABLE:
        logger.info(f"使用贝叶斯优化，迭代次数: {n_iter}")
        search = BayesSearchCV(
            estimator=model,
            search_spaces=param_dist,
            n_iter=n_iter,
            cv=cv_folds,
            scoring=scoring,
            refit=refit_metric,
            n_jobs=-1,
            verbose=1,
            random_state=42,
            return_train_score=True
        )
    elif optimization_method == 'random':
        logger.info(f"使用随机搜索，迭代次数: {n_iter}")
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=cv_folds,
            scoring=scoring,
            refit=refit_metric,
            n_jobs=-1,
            verbose=1,
            random_state=42,
            return_train_score=True
        )
    else:
        logger.info("使用网格搜索")
        search = GridSearchCV(
            estimator=model,
            param_grid=param_dist,
            cv=cv_folds,
            scoring=scoring,
            refit=refit_metric,
            n_jobs=-1,
            verbose=1,
            return_train_score=True
        )
        
    # 如果贝叶斯优化不可用但被请求，回退到随机搜索
    if optimization_method == 'bayesian' and not BAYESIAN_AVAILABLE:
        logger.warning("贝叶斯优化不可用，回退到随机搜索")
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=cv_folds,
            scoring=scoring,
            refit=refit_metric,
            n_jobs=-1,
            verbose=1,
            random_state=42,
            return_train_score=True
        )
    
    # 准备训练数据
    if use_validation_in_training and X_val is not None and y_val is not None:
        # 合并训练集和验证集
        X_train_combined = np.vstack([X_train, X_val])
        y_train_combined = np.hstack([y_train, y_val])
        logger.info(f"使用验证集参与训练，合并后训练集大小: {X_train_combined.shape}")
    else:
        X_train_combined = X_train
        y_train_combined = y_train
        logger.info(f"仅使用原始训练集，大小: {X_train_combined.shape}")
    
    # 执行超参数搜索
    logger.info("开始超参数搜索...")
    search.fit(X_train_combined, y_train_combined)
    
    # 获取最佳模型和参数
    best_model = search.best_estimator_
    best_params = search.best_params_
    best_score = search.best_score_
    cv_score = best_score
    
    logger.info(f"最佳交叉验证分数 ({refit_metric}): {best_score:.4f}")
    logger.info(f"交叉验证统计: 平均={cv_score:.4f}")
    
    # 计算原始训练集上的准确率
    train_pred = best_model.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_pred)
    logger.info(f"原始训练集准确率: {train_accuracy:.4f}")
    logger.info(f"最佳参数: {best_params}")
    
    # 如果使用早停机制，重新训练最佳模型
    if use_early_stopping:
        logger.info("使用早停机制重新训练最佳模型...")
        
        # 更新最佳参数，添加早停相关参数
        final_params = best_params.copy()
        final_params.update({
            'n_estimators': 2000,  # 设置较大的估计器数量
            'early_stopping_rounds': 100,
            'eval_metric': 'logloss' if not is_multiclass else 'multi_logloss'
        })
        
        # 创建最终模型
        final_model = lgb.LGBMClassifier(**{**base_params, **final_params})
        
        # 使用验证集进行早停训练
        final_model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_names=['training', 'validation'],
            callbacks=[lgb.log_evaluation(100)]
        )
        
        model = final_model
    else:
        model = best_model

    # 全面的模型评估
    logger.info("开始模型评估...")
    
    # 预测
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    y_train_proba = model.predict_proba(X_train)
    y_val_proba = model.predict_proba(X_val)
    y_test_proba = model.predict_proba(X_test)
    
    # 计算各种评估指标
    def calculate_metrics(y_true, y_pred, y_proba, dataset_name):
        # 在 calculate_metrics 函数中修改
        metrics = {
            f'{dataset_name}_accuracy': accuracy_score(y_true, y_pred),
            f'{dataset_name}_precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            f'{dataset_name}_recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            f'{dataset_name}_f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            }
        
        if is_multiclass:
            metrics.update({
                f'{dataset_name}_precision_macro': precision_score(y_true, y_pred, average='macro'),
                f'{dataset_name}_recall_macro': recall_score(y_true, y_pred, average='macro'),
                f'{dataset_name}_f1_macro': f1_score(y_true, y_pred, average='macro'),
                f'{dataset_name}_roc_auc_ovr': roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
            })
        else:
            metrics.update({
                f'{dataset_name}_precision': precision_score(y_true, y_pred),
                f'{dataset_name}_recall': recall_score(y_true, y_pred),
                f'{dataset_name}_f1': f1_score(y_true, y_pred),
                f'{dataset_name}_roc_auc': roc_auc_score(y_true, y_proba[:, 1])
            })
        
        return metrics
    
    # 计算所有数据集的指标
    train_metrics = calculate_metrics(y_train, y_train_pred, y_train_proba, 'train')
    val_metrics = calculate_metrics(y_val, y_val_pred, y_val_proba, 'validation')
    test_metrics = calculate_metrics(y_test, y_test_pred, y_test_proba, 'test')
    
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
    results_dir = f"./Results/LGBM_{target}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    logger.info(f"结果将保存到: {results_dir}")
    
    # 使用yellowbrick创建完整的性能可视化图表
    try:
        print(f"正在创建LightGBM模型的性能可视化图表 - {target}")
        
        # 1. 类别平衡可视化
        print("创建类别平衡图...")
        viz = ClassBalance(title=f"LightGBM Class Balance - {target}")
        viz.fit(y_train)
        viz.show(outpath=f'{results_dir}/lightgbm_class_balance.pdf')
        
        # 2. ROC曲线（仅二分类）
        if not is_multiclass:
            print("创建ROC曲线...")
            viz = ROCAUC(model, title=f"LightGBM ROC Curve - {target}")
            viz.fit(X_train, y_train)
            viz.score(X_test, y_test)
            viz.show(outpath=f'{results_dir}/lightgbm_roc_curve.pdf')
            
            # 3. 精确率-召回率曲线（仅二分类）
            print("创建精确率-召回率曲线...")
            viz = PrecisionRecallCurve(model, title=f"LightGBM Precision-Recall Curve - {target}")
            viz.fit(X_train, y_train)
            viz.score(X_test, y_test)
            viz.show(outpath=f'{results_dir}/lightgbm_precision_recall.pdf')
            
            # 4. 判别阈值可视化（仅二分类）
            print("创建判别阈值图...")
            viz = DiscriminationThreshold(model, title=f"LightGBM Discrimination Threshold - {target}")
            viz.fit(X_train, y_train)
            viz.score(X_test, y_test)
            viz.show(outpath=f'{results_dir}/lightgbm_discrimination_threshold.pdf')
            
        # 5. 分类报告
        print("创建分类报告...")
        viz = ClassificationReport(model, title=f"LightGBM Classification Report - {target}")
        viz.fit(X_train, y_train)
        viz.score(X_test, y_test)
        viz.show(outpath=f'{results_dir}/lightgbm_classification_report.pdf')
        
        # 6. 混淆矩阵
        print("创建混淆矩阵...")
        unique_labels = np.unique(np.concatenate([y_train, y_test]))
        viz = ConfusionMatrix(model, classes=unique_labels, title=f"LightGBM Confusion Matrix - {target}")
        viz.fit(X_train, y_train)
        viz.score(X_test, y_test)
        viz.show(outpath=f'{results_dir}/lightgbm_confusion_matrix.pdf')
        
        # 7. 类预测错误可视化
        print("创建类预测错误图...")
        viz = ClassPredictionError(model, classes=unique_labels, title=f"LightGBM Class Prediction Error - {target}")
        viz.fit(X_train, y_train)
        viz.score(X_test, y_test)
        viz.show(outpath=f'{results_dir}/lightgbm_class_prediction_error.pdf')
        
        print(f"✓ LightGBM模型所有可视化图表已保存到: {results_dir}")
        
    except Exception as e:
        print(f"❌ LightGBM可视化创建过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        # 如果yellowbrick失败，回退到matplotlib
        plt.figure(figsize=(15, 10))
        
        if not is_multiclass:
            # 二分类ROC曲线
            plt.subplot(2, 3, 1)
            fpr, tpr, _ = roc_curve(y_test, y_test_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, 'b-', label=f'测试集 AUC = {roc_auc:.3f}')
            
            # 验证集ROC
            fpr_val, tpr_val, _ = roc_curve(y_val, y_val_proba[:, 1])
            roc_auc_val = auc(fpr_val, tpr_val)
            plt.plot(fpr_val, tpr_val, 'g-', label=f'验证集 AUC = {roc_auc_val:.3f}')
            
            plt.plot([0, 1], [0, 1], 'r--', label='随机分类器')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('假正率 (FPR)')
            plt.ylabel('真正率 (TPR)')
            plt.title('ROC曲线')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
    
    # 特征重要性图
    plt.subplot(2, 3, 2)
    feature_importance = model.feature_importances_
    feature_names = [f'Feature_{i}' for i in range(len(feature_importance))]
    
    # 获取前15个最重要的特征
    indices = np.argsort(feature_importance)[::-1][:15]
    plt.barh(range(len(indices)), feature_importance[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('特征重要性')
    plt.title('前15个重要特征 (Gain)')
    plt.gca().invert_yaxis()
    
    # 学习曲线（如果使用了早停）
    if use_early_stopping and hasattr(model, 'evals_result_'):
        plt.subplot(2, 3, 3)
        eval_results = model.evals_result_
        if 'validation' in eval_results:
            train_scores = eval_results['training'][list(eval_results['training'].keys())[0]]
            val_scores = eval_results['validation'][list(eval_results['validation'].keys())[0]]
            
            plt.plot(train_scores, label='训练集')
            plt.plot(val_scores, label='验证集')
            plt.xlabel('迭代次数')
            plt.ylabel('损失值')
            plt.title('学习曲线')
            plt.legend()
            plt.grid(True, alpha=0.3)
    
    # 预测概率分布
    if not is_multiclass:
        plt.subplot(2, 3, 4)
        plt.hist(y_test_proba[y_test == 0, 1], bins=30, alpha=0.7, label='负类', density=True)
        plt.hist(y_test_proba[y_test == 1, 1], bins=30, alpha=0.7, label='正类', density=True)
        plt.xlabel('预测概率')
        plt.ylabel('密度')
        plt.title('预测概率分布')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 混淆矩阵热力图
    plt.subplot(2, 3, 5)
    cm = confusion_matrix(y_test, y_test_pred)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('测试集混淆矩阵')
    plt.colorbar()
    tick_marks = np.arange(len(np.unique(y_test)))
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    
    # 添加数值标注
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    
        # 性能指标对比
        plt.subplot(2, 3, 6)
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1']
        train_scores = [train_metrics['train_accuracy'], train_metrics['train_precision_weighted'], 
                       train_metrics['train_recall_weighted'], train_metrics['train_f1_weighted']]
        val_scores = [val_metrics['validation_accuracy'], val_metrics['validation_precision_weighted'], 
                     val_metrics['validation_recall_weighted'], val_metrics['validation_f1_weighted']]
        test_scores = [test_metrics['test_accuracy'], test_metrics['test_precision_weighted'], 
                      test_metrics['test_recall_weighted'], test_metrics['test_f1_weighted']]
        
        x = np.arange(len(metrics_names))
        width = 0.25
        
        plt.bar(x - width, train_scores, width, label='训练集')
        plt.bar(x, val_scores, width, label='验证集')
        plt.bar(x + width, test_scores, width, label='测试集')
        
        plt.xlabel('评估指标')
        plt.ylabel('分数')
        plt.title('模型性能对比')
        plt.xticks(x, metrics_names)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{results_dir}/model_analysis.pdf', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 单独绘制详细的特征重要性图
    plt.figure(figsize=(12, 8))
    lgb.plot_importance(
        model,
        importance_type='gain',
        max_num_features=30,
        figsize=(12, 8),
        title='LightGBM特征重要性 (Gain)'
    )
    plt.tight_layout()
    plt.savefig(f'{results_dir}/feature_importance_gain.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    
    plt.figure(figsize=(12, 8))
    lgb.plot_importance(
        model,
        importance_type='split',
        max_num_features=30,
        figsize=(12, 8),
        title='LightGBM特征重要性 (Split)'
    )
    plt.tight_layout()
    plt.savefig(f'{results_dir}/feature_importance_split.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    # 保存模型
    model_path = f'{results_dir}/lightgbm_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"模型已保存到: {model_path}")
    
    # 保存最佳参数
    best_params_path = f'{results_dir}/best_params.json'
    with open(best_params_path, 'w', encoding='utf-8') as f:
        json.dump(best_params, f, indent=2, ensure_ascii=False)
    logger.info(f"最佳参数已保存到: {best_params_path}")
    
    # 保存优化历史（如果是贝叶斯优化）
    if optimization_method == 'bayesian' and BAYESIAN_AVAILABLE and hasattr(search, 'cv_results_'):
        optimization_history = {
            'optimization_method': 'bayesian',
            'n_iterations': n_iter,
            'cv_results': search.cv_results_
        }
        history_path = f'{results_dir}/optimization_history.json'
        with open(history_path, 'w', encoding='utf-8') as f:
            # 转换numpy数组为列表以便JSON序列化
            serializable_results = {}
            for key, value in search.cv_results_.items():
                if isinstance(value, np.ndarray):
                    serializable_results[key] = value.tolist()
                else:
                    serializable_results[key] = value
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        logger.info(f"优化历史已保存到: {history_path}")
    
    # 保存详细评估结果
    detailed_results = {
        'model_info': {
            'target': target,
            'kinds': kinds,
            'timestamp': timestamp,
            'is_multiclass': is_multiclass,
            'use_early_stopping': use_early_stopping,
            'optimization_method': optimization_method,
            'bayesian_available': BAYESIAN_AVAILABLE
        },
        'best_params': best_params,
        'cv_score': cv_score,
        'train_metrics': train_metrics,
        'validation_metrics': val_metrics,
        'test_metrics': test_metrics,
        'feature_importance': {
            'gain': model.feature_importances_.tolist(),
            'feature_names': [f'Feature_{i}' for i in range(len(model.feature_importances_))]
        }
    }
    
    # 如果是二分类，添加ROC AUC信息
    if not is_multiclass:
        detailed_results['roc_auc'] = {
            'train': roc_auc_score(y_train, y_train_proba[:, 1]),
            'validation': roc_auc_score(y_val, y_val_proba[:, 1]),
            'test': roc_auc_score(y_test, y_test_proba[:, 1])
        }
    
    # 保存详细结果
    results_path = f'{results_dir}/detailed_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    logger.info(f"详细结果已保存到: {results_path}")
    
    # 生成模型总结报告
    summary_report = f"""
=== LightGBM模型训练总结报告 ===
时间: {timestamp}
目标变量: {target}
数据类型: {kinds}

=== 数据集信息 ===
训练集大小: {len(X_train)}
验证集大小: {len(X_val)}
测试集大小: {len(X_test)}
特征数量: {X_train.shape[1]}
类别数量: {len(np.unique(y_train))}

=== 优化方法 ===
{optimization_method.upper()}优化 {'(贝叶斯优化可用)' if BAYESIAN_AVAILABLE else '(贝叶斯优化不可用，使用随机搜索)'}

=== 最佳参数 ===
{json.dumps(best_params, indent=2, ensure_ascii=False)}

=== 交叉验证分数 ===
最佳CV分数: {cv_score:.4f}

=== 模型性能 ===
训练集准确率: {train_metrics['train_accuracy']:.4f}
验证集准确率: {val_metrics['validation_accuracy']:.4f}
测试集准确率: {test_metrics['test_accuracy']:.4f}

训练集F1分数: {train_metrics['train_f1_weighted']:.4f}
验证集F1分数: {val_metrics['validation_f1_weighted']:.4f}
测试集F1分数: {test_metrics['test_f1_weighted']:.4f}
"""
    
    if not is_multiclass:
        summary_report += f"""
训练集ROC AUC: {roc_auc_score(y_train, y_train_proba[:, 1]):.4f}
验证集ROC AUC: {roc_auc_score(y_val, y_val_proba[:, 1]):.4f}
测试集ROC AUC: {roc_auc_score(y_test, y_test_proba[:, 1]):.4f}
"""
    
    summary_report += f"""

=== 文件保存位置 ===
模型文件: {model_path}
详细结果: {results_path}
可视化图表: {results_dir}/model_analysis.png
特征重要性: {results_dir}/feature_importance_*.png
"""
    
    # 保存总结报告
    summary_path = f'{results_dir}/summary_report.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_report)
    
    logger.info("\n" + "="*50)
    logger.info(summary_report)
    logger.info("="*50)
    
    # 返回优化后的结果
    return_dict = {
        'model': model,
        'best_params': best_params,
        'cv_score': cv_score,
        'optimization_method': optimization_method,
        'bayesian_available': BAYESIAN_AVAILABLE,
        'train_metrics': train_metrics,
        'validation_metrics': val_metrics,
        'test_metrics': test_metrics,
        'results_dir': results_dir,
        'model_path': model_path,
        'best_params_path': best_params_path,
        'results_path': results_path,
        'summary_path': summary_path,
        'feature_importance': model.feature_importances_
    }
    
    # 如果是贝叶斯优化，添加优化历史路径
    if optimization_method == 'bayesian' and BAYESIAN_AVAILABLE and hasattr(search, 'cv_results_'):
        return_dict['optimization_history_path'] = history_path
    
    if not is_multiclass:
        return_dict['roc_auc'] = {
            'train': roc_auc_score(y_train, y_train_proba[:, 1]),
            'validation': roc_auc_score(y_val, y_val_proba[:, 1]),
            'test': roc_auc_score(y_test, y_test_proba[:, 1])
        }
    
    return return_dict

if __name__ == "__main__":
    # 测试代码
    from sklearn.datasets import make_classification
    
    # 生成测试数据
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # 计算类别权重
    class_weights = dict(zip(np.unique(y_train), 
                           compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)))
    
    # 测试模型
    model, scores = LightGBM_model(
        X_train, X_test, y_train, y_test, X_val, y_val,
        class_weights, "test_target", "test_model",optimization_method='grid'
    )
    
    print("测试完成!")