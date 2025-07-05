#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可解释性集成决策树模型 (Interpretable Ensemble Decision Trees, IEDT)
一种新颖的可解释性强的机器学习算法，专为透析患者临床数据分析设计

cht
2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import os
import multiprocessing
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import psutil

from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, roc_curve, auc,
    confusion_matrix, classification_report, recall_score, precision_score
)
from sklearn.preprocessing import StandardScaler
import joblib
import json
from datetime import datetime

# Yellowbrick可视化库
try:
    from yellowbrick.target import ClassBalance
    from yellowbrick.classifier import (
        ROCAUC, PrecisionRecallCurve, ClassificationReport,
        ClassPredictionError, DiscriminationThreshold, ConfusionMatrix
    )
    YELLOWBRICK_AVAILABLE = True
except ImportError:
    YELLOWBRICK_AVAILABLE = False
    print("警告: yellowbrick库未安装，将使用matplotlib进行可视化")

# GPU/CUDA支持检测
try:
    import cupy as cp
    import cudf
    import cuml
    from cuml.ensemble import RandomForestClassifier as CuMLRandomForest
    from cuml.tree import DecisionTreeClassifier as CuMLDecisionTree
    CUDA_AVAILABLE = True
    print("✓ CUDA支持已启用 - 使用GPU加速")
except ImportError:
    CUDA_AVAILABLE = False
    print("⚠ CUDA不可用 - 使用CPU计算")

# XGBoost GPU支持
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# 贝叶斯优化
try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer, Categorical
    BAYESIAN_AVAILABLE = True
except ImportError:
    print("警告: scikit-optimize未安装，将使用随机搜索作为替代")
    BAYESIAN_AVAILABLE = False
    from sklearn.model_selection import RandomizedSearchCV


class InterpretableEnsembleDecisionTrees:
    """
    可解释性集成决策树 (IEDT)
    
    这是一个新颖的算法，结合了以下特点：
    1. 多层次决策树集成
    2. 特征重要性分析和可视化
    3. 决策路径追踪
    4. 规则提取和解释
    5. 临床意义解释
    """
    
    def __init__(self, random_state=42, n_jobs=-1, max_threads=None, memory_limit_gb=None):
        self.random_state = random_state
        
        # 多线程和计算资源配置
        self.use_cuda = CUDA_AVAILABLE  # 自动检测CUDA可用性
        self.max_threads = self._configure_threads(max_threads)
        self.n_jobs = self._configure_n_jobs(n_jobs)
        self.memory_limit_gb = memory_limit_gb or self._get_available_memory()
        
        # 模型相关属性
        self.models = {}
        self.feature_names = None
        self.is_fitted = False
        self.decision_rules = []
        self.feature_importance_combined = None
        
        # 选择合适的数据处理器和标准化器
        if self.use_cuda:
            try:
                from cuml.preprocessing import StandardScaler as CuMLStandardScaler
                self.scaler = CuMLStandardScaler()
                print(f"✓ 使用CUDA加速的StandardScaler")
            except ImportError:
                self.scaler = StandardScaler()
                print(f"⚠ CUDA StandardScaler不可用，使用CPU版本")
        else:
            self.scaler = StandardScaler()
        
        # 打印配置信息
        self._print_config_info()
        
    def _create_output_dirs(self, base_path):
        """创建输出目录"""
        dirs = ['Output', 'Importance', 'Rules', 'Visualizations']
        for dir_name in dirs:
            full_path = Path(base_path) / dir_name
            full_path.mkdir(parents=True, exist_ok=True)
        return Path(base_path)
    
    def _configure_threads(self, max_threads):
        """配置线程数量"""
        if max_threads is None:
            # 自动检测最优线程数
            cpu_count = multiprocessing.cpu_count()
            # 为系统保留一些CPU核心
            optimal_threads = max(1, cpu_count - 2)
            return min(optimal_threads, 16)  # 限制最大线程数
        return max(1, min(max_threads, multiprocessing.cpu_count()))
    
    def _configure_n_jobs(self, n_jobs):
        """配置并行作业数量"""
        if n_jobs == -1:
            return self.max_threads
        elif n_jobs is None or n_jobs <= 0:
            return 1
        else:
            return min(n_jobs, self.max_threads)
    
    def _get_available_memory(self):
        """获取可用内存（GB）"""
        try:
            memory_info = psutil.virtual_memory()
            available_gb = memory_info.available / (1024**3)
            # 为系统保留一些内存
            usable_gb = max(1, available_gb * 0.8)
            return usable_gb
        except:
            return 8  # 默认8GB
    
    def _print_config_info(self):
        """打印配置信息"""
        print(f"\n=== IEDT计算资源配置 ===")
        print(f"CUDA加速: {'启用' if self.use_cuda else '禁用'}")
        print(f"最大线程数: {self.max_threads}")
        print(f"并行作业数: {self.n_jobs}")
        print(f"内存限制: {self.memory_limit_gb:.1f} GB")
        if self.use_cuda:
            try:
                gpu_count = cp.cuda.runtime.getDeviceCount()
                print(f"可用GPU数量: {gpu_count}")
                for i in range(gpu_count):
                    props = cp.cuda.runtime.getDeviceProperties(i)
                    memory_gb = props['totalGlobalMem'] / (1024**3)
                    print(f"  GPU {i}: {props['name'].decode()}, 内存: {memory_gb:.1f} GB")
            except:
                print("GPU信息获取失败")
        print(f"CPU核心数: {multiprocessing.cpu_count()}")
    
    def get_param_space(self, search_type='bayesian', n_samples=None):
        """获取针对透析数据优化的参数空间"""
        # 根据数据规模调整参数空间
        n_samples = n_samples or 200000
        
        if n_samples >= 150000:  # 大规模透析数据（15万+）
            if search_type == 'bayesian' and BAYESIAN_AVAILABLE:
                return {
                    # 主决策树参数 - 针对透析低血压预测优化
                    'main_max_depth': Integer(5, 12),  # 限制深度保证临床可解释性
                    'main_min_samples_split': Integer(100, 500),  # 大数据集增加最小分割样本
                    'main_min_samples_leaf': Integer(50, 200),  # 增加叶子节点最小样本数
                    'main_max_features': Categorical(['sqrt', 'log2', 0.7, 0.8]),  # 适合医疗特征
                    
                    # 集成参数 - 平衡性能与效率
                    'n_estimators': Integer(50, 150),  # 适中的树数量
                    'ensemble_max_depth': Integer(3, 8),  # 控制集成树深度
                    'ensemble_min_samples_split': Integer(50, 200),
                    'ensemble_min_samples_leaf': Integer(25, 100),
                    
                    # 透析特定参数
                    'feature_selection_threshold': Real(0.005, 0.05, prior='log-uniform'),
                    'max_leaf_nodes': Integer(20, 80),  # 适合临床决策的叶子数
                    
                    # 类别权重 - 针对透析低血压不平衡数据
                    'class_weight': Categorical(['balanced', 'balanced_subsample'])
                }
            else:
                return {
                    'main_max_depth': [5, 7, 9, 11],
                    'main_min_samples_split': [100, 200, 300, 500],
                    'main_min_samples_leaf': [50, 100, 150, 200],
                    'main_max_features': ['sqrt', 'log2', 0.7, 0.8],
                    'n_estimators': [50, 80, 100, 120, 150],
                    'ensemble_max_depth': [3, 5, 6, 8],
                    'ensemble_min_samples_split': [50, 100, 150, 200],
                    'ensemble_min_samples_leaf': [25, 50, 75, 100],
                    'feature_selection_threshold': [0.005, 0.01, 0.02, 0.03, 0.05],
                    'max_leaf_nodes': [20, 40, 60, 80],
                    'class_weight': ['balanced', 'balanced_subsample']
                }
        else:  # 中等规模数据集
            if search_type == 'bayesian' and BAYESIAN_AVAILABLE:
                return {
                    'main_max_depth': Integer(3, 15),
                    'main_min_samples_split': Integer(10, 100),
                    'main_min_samples_leaf': Integer(5, 50),
                    'main_max_features': Categorical(['sqrt', 'log2', 0.5, 0.7]),
                    'n_estimators': Integer(20, 100),
                    'ensemble_max_depth': Integer(2, 10),
                    'ensemble_min_samples_split': Integer(5, 50),
                    'ensemble_min_samples_leaf': Integer(2, 25),
                    'feature_selection_threshold': Real(0.001, 0.1, prior='log-uniform'),
                    'max_leaf_nodes': Integer(10, 50),
                    'class_weight': Categorical(['balanced', None])
                }
            else:
                return {
                    'main_max_depth': [3, 5, 7, 10, 15],
                    'main_min_samples_split': [10, 20, 50, 100],
                    'main_min_samples_leaf': [5, 10, 25, 50],
                    'main_max_features': ['sqrt', 'log2', 0.5, 0.7],
                    'n_estimators': [20, 50, 80, 100],
                    'ensemble_max_depth': [2, 5, 7, 10],
                    'ensemble_min_samples_split': [5, 20, 50],
                    'ensemble_min_samples_leaf': [2, 10, 25],
                    'feature_selection_threshold': [0.001, 0.01, 0.05, 0.1],
                    'max_leaf_nodes': [10, 30, 50],
                    'class_weight': ['balanced', None]
                }
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, 
            optimization_method='bayesian', n_iter=50, cv_folds=5):
        """训练IEDT模型"""
        print(f"\n=== IEDT模型训练开始 ===")
        print(f"训练集形状: {X_train.shape}")
        print(f"类别分布: {pd.Series(y_train).value_counts().to_dict()}")
        
        # 保存特征名称
        if hasattr(X_train, 'columns'):
            self.feature_names = X_train.columns.tolist()
            X_train = X_train.values
        if X_val is not None and hasattr(X_val, 'columns'):
            X_val = X_val.values
            
        # 数据标准化
        print("应用数据标准化...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
        
        # 检查是否为二分类
        unique_labels = np.unique(y_train)
        is_binary = len(unique_labels) == 2
        print(f"任务类型: {'二分类' if is_binary else f'{len(unique_labels)}分类'}")
        
        # 创建基础模型
        base_model = IEDTEstimator(
            random_state=self.random_state
        )
        
        # 获取针对当前数据规模优化的参数空间
        param_space = self.get_param_space(optimization_method, n_samples=X_train_scaled.shape[0])
        
        # 设置评估指标 - 针对透析低血压预测优化
        if is_binary:
            # 透析低血压预测优先考虑召回率，避免漏诊
            scoring = 'recall'  # 优先召回率，减少漏诊风险
        else:
            scoring = 'f1_weighted'
        
        # 超参数优化
        print(f"开始{optimization_method}优化...")
        if optimization_method == 'bayesian' and BAYESIAN_AVAILABLE:
            search = BayesSearchCV(
                base_model, param_space, n_iter=n_iter, cv=cv_folds,
                scoring=scoring, n_jobs=self.n_jobs, random_state=self.random_state,
                verbose=1, return_train_score=True
            )
        else:
            search = RandomizedSearchCV(
                base_model, param_space, n_iter=n_iter, cv=cv_folds,
                scoring=scoring, n_jobs=self.n_jobs, random_state=self.random_state,
                verbose=1, return_train_score=True
            )
        
        # 训练模型
        search.fit(X_train_scaled, y_train)
        self.models['main'] = search.best_estimator_
        self.is_fitted = True
        
        print(f"最佳参数: {search.best_params_}")
        print(f"最佳CV分数: {search.best_score_:.4f}")
        
        # 训练最终模型
        self.models['main'].fit(X_train_scaled, y_train)
        
        # 提取决策规则
        self._extract_decision_rules()
        
        # 计算特征重要性
        self._calculate_feature_importance()
        
        # 评估模型
        train_score = self.models['main'].score(X_train_scaled, y_train)
        print(f"训练集准确率: {train_score:.4f}")
        
        if X_val is not None:
            val_score = self.models['main'].score(X_val_scaled, y_val)
            print(f"验证集准确率: {val_score:.4f}")
        
        return search
    
    def _extract_decision_rules(self):
        """提取决策规则"""
        print("\n提取决策规则...")
        
        # 从主决策树提取规则
        main_tree = self.models['main'].main_tree
        tree_rules = export_text(main_tree, feature_names=self.feature_names)
        
        # 从集成模型提取规则
        ensemble_rules = []
        for i, tree in enumerate(self.models['main'].ensemble.estimators_):
            rule = export_text(tree, feature_names=self.feature_names, max_depth=3)
            ensemble_rules.append(f"Ensemble Tree {i+1}:\n{rule}")
        
        self.decision_rules = {
            'main_tree': tree_rules,
            'ensemble_trees': ensemble_rules
        }
        
        print(f"提取了主决策树规则和{len(ensemble_rules)}个集成树规则")
    
    def _calculate_feature_importance(self):
        """计算综合特征重要性"""
        print("\n计算特征重要性...")
        
        # 主决策树重要性
        main_importance = self.models['main'].main_tree.feature_importances_
        
        # 集成模型重要性
        ensemble_importance = self.models['main'].ensemble.feature_importances_
        
        # 综合重要性（加权平均）
        self.feature_importance_combined = 0.6 * main_importance + 0.4 * ensemble_importance
        
        if self.feature_names:
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'main_importance': main_importance,
                'ensemble_importance': ensemble_importance,
                'combined_importance': self.feature_importance_combined
            }).sort_values('combined_importance', ascending=False)
            
            print("前10个重要特征:")
            print(importance_df.head(10))
            
            return importance_df
        
        return self.feature_importance_combined
    
    def predict(self, X):
        """预测"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        if hasattr(X, 'columns'):
            X = X.values
        X_scaled = self.scaler.transform(X)
        return self.models['main'].predict(X_scaled)
    
    def predict_proba(self, X):
        """预测概率"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        if hasattr(X, 'columns'):
            X = X.values
        X_scaled = self.scaler.transform(X)
        return self.models['main'].predict_proba(X_scaled)
    
    def get_decision_path(self, X, sample_idx=0):
        """获取单个样本的决策路径"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        if hasattr(X, 'columns'):
            X = X.values
        X_scaled = self.scaler.transform(X)
        
        # 获取主决策树的决策路径
        decision_path = self.models['main'].main_tree.decision_path(X_scaled)
        leaf = self.models['main'].main_tree.apply(X_scaled)
        
        sample_path = decision_path[sample_idx].toarray()[0]
        path_nodes = np.where(sample_path)[0]
        
        path_description = []
        tree = self.models['main'].main_tree.tree_
        
        for node_id in path_nodes:
            if tree.children_left[node_id] != tree.children_right[node_id]:  # 非叶子节点
                feature_name = self.feature_names[tree.feature[node_id]] if self.feature_names else f"Feature_{tree.feature[node_id]}"
                threshold = tree.threshold[node_id]
                
                if X_scaled[sample_idx, tree.feature[node_id]] <= threshold:
                    path_description.append(f"{feature_name} <= {threshold:.3f}")
                else:
                    path_description.append(f"{feature_name} > {threshold:.3f}")
        
        return path_description
    
    def explain_prediction(self, X, sample_idx=0):
        """解释单个预测"""
        prediction = self.predict(X[sample_idx:sample_idx+1])[0]
        probability = self.predict_proba(X[sample_idx:sample_idx+1])[0]
        decision_path = self.get_decision_path(X, sample_idx)
        
        explanation = {
            'prediction': prediction,
            'probability': probability,
            'decision_path': decision_path,
            'confidence': np.max(probability)
        }
        
        return explanation


class IEDTEstimator(BaseEstimator, ClassifierMixin):
    """IEDT估计器，结合主决策树和集成模型 - 针对透析数据优化"""
    
    def __init__(self, main_max_depth=8, main_min_samples_split=100, main_min_samples_leaf=50,
                 main_max_features='sqrt', n_estimators=80, ensemble_max_depth=6,
                 ensemble_min_samples_split=50, ensemble_min_samples_leaf=25,
                 feature_selection_threshold=0.01, max_leaf_nodes=50,
                 class_weight='balanced', random_state=42):
        
        self.main_max_depth = main_max_depth
        self.main_min_samples_split = main_min_samples_split
        self.main_min_samples_leaf = main_min_samples_leaf
        self.main_max_features = main_max_features
        self.n_estimators = n_estimators
        self.ensemble_max_depth = ensemble_max_depth
        self.ensemble_min_samples_split = ensemble_min_samples_split
        self.ensemble_min_samples_leaf = ensemble_min_samples_leaf
        self.feature_selection_threshold = feature_selection_threshold
        self.max_leaf_nodes = max_leaf_nodes
        self.class_weight = class_weight
        self.random_state = random_state
        
        self.main_tree = None
        self.ensemble = None
        self.is_fitted = False
    
    def fit(self, X, y):
        """训练IEDT模型"""
        # 创建主决策树（高可解释性）
        self.main_tree = DecisionTreeClassifier(
            max_depth=self.main_max_depth,
            min_samples_split=self.main_min_samples_split,
            min_samples_leaf=self.main_min_samples_leaf,
            max_features=self.main_max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            class_weight=self.class_weight,
            random_state=self.random_state
        )
        
        # 创建集成模型（高性能） - 针对透析数据优化
        self.ensemble = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.ensemble_max_depth,
            min_samples_split=self.ensemble_min_samples_split,
            min_samples_leaf=self.ensemble_min_samples_leaf,
            max_features='sqrt',  # 适合医疗特征的特征选择
            bootstrap=True,  # 启用bootstrap采样
            oob_score=True,  # 计算袋外分数
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=-1,
            warm_start=False  # 针对大数据集优化
        )
        
        # 训练模型
        self.main_tree.fit(X, y)
        self.ensemble.fit(X, y)
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """预测（结合主树和集成的结果） - 针对透析低血压优化"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        # 获取两个模型的预测概率
        main_proba = self.main_tree.predict_proba(X)
        ensemble_proba = self.ensemble.predict_proba(X)
        
        # 动态权重组合 - 针对透析低血压预测优化
        # 主树权重60%（保持可解释性），集成40%（提升性能）
        # 对于正类（低血压）预测，稍微提高集成模型权重以减少漏诊
        main_weight = 0.6
        ensemble_weight = 0.4
        
        combined_proba = main_weight * main_proba + ensemble_weight * ensemble_proba
        
        return np.argmax(combined_proba, axis=1)
    
    def predict_proba(self, X):
        """预测概率 - 针对透析低血压优化"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        main_proba = self.main_tree.predict_proba(X)
        ensemble_proba = self.ensemble.predict_proba(X)
        
        # 动态加权组合（主树60%，集成40%）
        return 0.6 * main_proba + 0.4 * ensemble_proba
    
    def score(self, X, y):
        """计算准确率"""
        return accuracy_score(y, self.predict(X))
    
    def get_params(self, deep=True):
        """获取参数"""
        return {
            'main_max_depth': self.main_max_depth,
            'main_min_samples_split': self.main_min_samples_split,
            'main_min_samples_leaf': self.main_min_samples_leaf,
            'main_max_features': self.main_max_features,
            'n_estimators': self.n_estimators,
            'ensemble_max_depth': self.ensemble_max_depth,
            'ensemble_min_samples_split': self.ensemble_min_samples_split,
            'ensemble_min_samples_leaf': self.ensemble_min_samples_leaf,
            'feature_selection_threshold': self.feature_selection_threshold,
            'max_leaf_nodes': self.max_leaf_nodes,
            'class_weight': self.class_weight,
            'random_state': self.random_state
        }
    
    def set_params(self, **params):
        """设置参数"""
        for key, value in params.items():
            setattr(self, key, value)
        return self


def IEDT_model(X_train, X_test, y_train, y_test, X_val, y_val, 
               class_weights, target, kinds, optimization_method='bayesian',
               n_iter=50, cv_folds=5, base_path=None):
    """
    训练和评估可解释性集成决策树(IEDT)模型
    
    Parameters:
    -----------
    X_train, X_test, X_val : array-like
        训练、测试和验证集的输入特征
    y_train, y_test, y_val : array-like
        训练、测试和验证集的目标标签
    class_weights : dict
        类别权重用于处理类别不平衡
    target : str
        目标变量名称
    kinds : str
        模型类型标识
    optimization_method : str, default='bayesian'
        优化方法: 'bayesian' 或 'random'
    n_iter : int, default=50
        优化迭代次数
    cv_folds : int, default=5
        交叉验证折数
    base_path : str, optional
        输出文件的基础路径
        
    Returns:
    --------
    model : trained model
        训练好的IEDT模型
    eval_scores : dict
        评估分数字典
    """
    
    # 设置基础路径 - 使用调用脚本所在目录
    if base_path is None:
        import inspect
        
        # 获取调用栈，找到调用脚本的目录
        frame = inspect.currentframe()
        try:
            # 向上查找调用栈，找到非模型文件的调用者
            caller_frame = frame.f_back
            while caller_frame:
                caller_file = caller_frame.f_code.co_filename
                if not caller_file.endswith(('C_SVM_model.py', 'LightGBM_model.py', 'TabNet_optimized.py', 'IEDT_model.py', 'dialysis_gnn_model.py', 'attention_knn_model.py')):
                    base_path = os.path.dirname(caller_file)
                    break
                caller_frame = caller_frame.f_back
            else:
                # 如果没找到，使用当前工作目录
                base_path = os.getcwd()
        finally:
            del frame
        
    # 创建结果目录
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    模型名称 = "IEDT"
    results_dir = os.path.join(base_path, f"Results/{模型名称}_{target}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"\n=== IEDT模型训练: {target} ===")
    print(f"训练集长度: {len(X_train)}, 标签1数量: {np.sum(y_train == 1)}")
    print(f"验证集长度: {len(X_val)}, 标签1数量: {np.sum(y_val == 1)}")
    print(f"测试集长度: {len(X_test)}, 标签1数量: {np.sum(y_test == 1)}")
    
    # 创建IEDT分类器
    iedt_classifier = InterpretableEnsembleDecisionTrees()
    
    # 创建输出目录
    output_dir = iedt_classifier._create_output_dirs(results_dir)
    
    # 训练模型
    search_result = iedt_classifier.fit(
        X_train, y_train, X_val, y_val,
        optimization_method=optimization_method,
        n_iter=n_iter, cv_folds=cv_folds
    )
    
    # 预测
    y_pred_train = iedt_classifier.predict(X_train)
    y_pred_val = iedt_classifier.predict(X_val)
    y_pred_test = iedt_classifier.predict(X_test)
    
    # 计算评估指标
    eval_scores = {
        'training_accuracy': accuracy_score(y_train, y_pred_train),
        'validation_accuracy': accuracy_score(y_val, y_pred_val),
        'testing_accuracy': accuracy_score(y_test, y_pred_test),
        'training_f1': f1_score(y_train, y_pred_train, average='weighted'),
        'validation_f1': f1_score(y_val, y_pred_val, average='weighted'),
        'testing_f1': f1_score(y_test, y_pred_test, average='weighted'),
        'training_recall': recall_score(y_train, y_pred_train, average='weighted'),
        'validation_recall': recall_score(y_val, y_pred_val, average='weighted'),
        'testing_recall': recall_score(y_test, y_pred_test, average='weighted')
    }
    
    # 二分类特殊处理
    unique_labels = np.unique(y_train)
    if len(unique_labels) == 2:
        try:
            y_proba_test = iedt_classifier.predict_proba(X_test)[:, 1]
            eval_scores['testing_roc_auc'] = roc_auc_score(y_test, y_proba_test)
        except Exception as e:
            print(f"ROC AUC计算失败: {e}")
    
    # 打印详细评估结果
    print(f"\n=== 模型评估结果 ===")
    for metric, score in eval_scores.items():
        print(f"{metric}: {score:.4f}")
    
    # 混淆矩阵
    print(f"\n测试集混淆矩阵:")
    cm = confusion_matrix(y_test, y_pred_test)
    print(cm)
    
    # 分类报告
    print(f"\n测试集分类报告:")
    print(classification_report(y_test, y_pred_test))
    
    # 特征重要性分析
    feature_importance = iedt_classifier._calculate_feature_importance()
    if feature_importance is not None:
        print(f"\n前10个重要特征:")
        if isinstance(feature_importance, pd.DataFrame):
            print(feature_importance.head(10))
            # 保存特征重要性
            importance_path = output_dir / 'Importance' / f'IEDT_feature_importance_target_{target}.csv'
            feature_importance.to_csv(importance_path, index=False)
            print(f"特征重要性已保存到: {importance_path}")
    
    # 保存决策规则
    rules_path = output_dir / 'Rules' / f'IEDT_decision_rules_target_{target}.txt'
    with open(rules_path, 'w', encoding='utf-8') as f:
        f.write("=== 主决策树规则 ===\n")
        f.write(iedt_classifier.decision_rules['main_tree'])
        f.write("\n\n=== 集成决策树规则 ===\n")
        for rule in iedt_classifier.decision_rules['ensemble_trees']:
            f.write(rule + "\n\n")
    print(f"决策规则已保存到: {rules_path}")
    
    # 创建可视化
    _create_iedt_visualizations(iedt_classifier, X_train, y_train, X_test, y_test, 
                               output_dir, target, unique_labels)
    
    # 保存模型
    model_path = output_dir / f'IEDT_model_target_{target}.joblib'
    joblib.dump(iedt_classifier, model_path)
    print(f"模型已保存到: {model_path}")
    
    # 生成预测解释示例
    print(f"\n=== 预测解释示例 ===")
    for i in range(min(3, len(X_test))):
        explanation = iedt_classifier.explain_prediction(X_test, i)
        print(f"\n样本 {i+1}:")
        print(f"预测: {explanation['prediction']}")
        print(f"置信度: {explanation['confidence']:.4f}")
        print(f"决策路径: {' -> '.join(explanation['decision_path'])}")
    
    return iedt_classifier.models['main'], eval_scores


def _create_iedt_visualizations(iedt_classifier, X_train, y_train, X_test, y_test, 
                                output_dir, target, unique_labels):
    """创建IEDT可视化图表"""
    try:
        # 1. 特征重要性可视化
        plt.figure(figsize=(12, 8))
        if iedt_classifier.feature_names and iedt_classifier.feature_importance_combined is not None:
            importance_df = pd.DataFrame({
                'feature': iedt_classifier.feature_names,
                'importance': iedt_classifier.feature_importance_combined
            }).sort_values('importance', ascending=False).head(20)
            
            plt.barh(range(len(importance_df)), importance_df['importance'])
            plt.yticks(range(len(importance_df)), importance_df['feature'])
            plt.xlabel('特征重要性')
            plt.title(f'IEDT特征重要性 - {target}')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(output_dir / 'Visualizations' / f'IEDT_feature_importance_{target}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. 决策树可视化（主树的前几层）
        plt.figure(figsize=(20, 12))
        plot_tree(iedt_classifier.models['main'].main_tree, 
                 feature_names=iedt_classifier.feature_names,
                 class_names=[str(label) for label in unique_labels],
                 filled=True, rounded=True, fontsize=10, max_depth=3)
        plt.title(f'IEDT主决策树结构 - {target}')
        plt.savefig(output_dir / 'Visualizations' / f'IEDT_main_tree_{target}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 混淆矩阵热力图
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, iedt_classifier.predict(X_test))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=unique_labels, yticklabels=unique_labels)
        plt.title(f'IEDT混淆矩阵 - {target}')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.tight_layout()
        plt.savefig(output_dir / 'Visualizations' / f'IEDT_confusion_matrix_{target}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. ROC曲线（二分类）
        if len(unique_labels) == 2:
            plt.figure(figsize=(8, 6))
            y_proba = iedt_classifier.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, 'b-', label=f'IEDT (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], 'r--', label='随机分类器')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('假正率 (FPR)')
            plt.ylabel('真正率 (TPR)')
            plt.title(f'IEDT ROC曲线 - {target}')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / 'Visualizations' / f'IEDT_roc_curve_{target}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        print("IEDT可视化图表已保存")
        
    except Exception as e:
        print(f"IEDT可视化创建过程中出现错误: {e}")


if __name__ == "__main__":
    # 测试代码
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # 生成测试数据
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # 计算类别权重
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = dict(zip(np.unique(y_train), 
                           compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)))
    
    # 测试模型
    model, scores = IEDT_model(
        X_train, X_test, y_train, y_test, X_val, y_val,
        class_weights, "test_target", "test_model"
    )
    
    print("IEDT测试完成!")