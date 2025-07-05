import pandas as pd
import numpy as np
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    roc_curve, auc, f1_score, make_scorer, recall_score, roc_auc_score,
    confusion_matrix, classification_report, accuracy_score, precision_score,
    balanced_accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
import pickle
import joblib
import json
from datetime import datetime
from scipy.stats import uniform, loguniform

# 贝叶斯优化
try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer, Categorical
    BAYESIAN_AVAILABLE = True
except ImportError:
    print("警告: scikit-optimize未安装，将使用随机搜索作为替代")
    BAYESIAN_AVAILABLE = False

try:
    from yellowbrick.target import ClassBalance
    from yellowbrick.classifier import ROCAUC, PrecisionRecallCurve, ClassificationReport, ClassPredictionError, DiscriminationThreshold, ConfusionMatrix
    YELLOWBRICK_AVAILABLE = True
except ImportError:
    YELLOWBRICK_AVAILABLE = False
    print("Warning: yellowbrick not available. Some visualizations will be skipped.")


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
                # 排除模型文件，找到真正的调用脚本
                model_files = ('C_SVM_model.py', 'LightGBM_model.py', 'TabNet_optimized.py', 
                             'IEDT_model.py', 'dialysis_gnn_model.py', 'attention_knn_model.py')
                if not any(caller_file.endswith(mf) for mf in model_files):
                    return Path(caller_file).parent
                caller_frame = caller_frame.f_back
            # 如果没找到，使用当前工作目录
            return Path.cwd()
        finally:
            del frame
    
    def create_model_dirs(self, model_name, target, timestamp=None):
        """为特定模型创建目录结构"""
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
        # 创建主目录
        main_dir = self.base_path / "Results" / f"{model_name}_{target}_{timestamp}"
        
        # 创建子目录
        subdirs = {
            'main': main_dir,
            'models': main_dir / 'models',
            'plots': main_dir / 'plots', 
            'reports': main_dir / 'reports',
            'importance': main_dir / 'feature_importance'
        }
        
        for dir_path in subdirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
            
        return subdirs


class SVMParameterManager:
    """SVM参数管理器"""
    
    @staticmethod
    def get_param_space(search_type='bayesian', model_type='auto', data_size=None):
        """获取参数空间（针对透析数据优化）
        
        Parameters:
        -----------
        search_type : str
            搜索类型: 'bayesian', 'random', 'grid'
        model_type : str
            模型类型: 'linear', 'nonlinear', 'auto'
        data_size : int
            数据集大小，用于自动选择模型类型和参数范围
        """
        # 针对透析数据的智能模型选择
        if model_type == 'auto':
            if data_size is not None:
                # 大规模透析数据（>=15万样本）优先使用线性模型，提升训练效率
                # 中等规模数据（1万-15万）可以尝试非线性模型
                # 小规模数据（<1万）使用非线性模型获得更好性能
                if data_size >= 150000:
                    model_type = 'linear'
                    print(f"检测到大规模透析数据({data_size}样本)，使用线性SVM以提升训练效率")
                elif data_size >= 10000:
                    model_type = 'nonlinear'
                    print(f"检测到中等规模透析数据({data_size}样本)，使用非线性SVM平衡性能与效率")
                else:
                    model_type = 'nonlinear'
                    print(f"检测到小规模透析数据({data_size}样本)，使用非线性SVM获得最佳性能")
            else:
                model_type = 'linear'  # 默认使用线性模型
        
        is_linear = (model_type == 'linear')
        
        if search_type == 'bayesian' and BAYESIAN_AVAILABLE:
            return SVMParameterManager._get_bayesian_params(is_linear, data_size)
        elif search_type == 'random':
            return SVMParameterManager._get_random_params(is_linear, data_size)
        elif search_type == 'grid':
            return SVMParameterManager._get_grid_params(is_linear, data_size)
        else:
            return SVMParameterManager._get_default_params(is_linear, data_size)
    
    @staticmethod
    def _get_bayesian_params(is_linear, data_size=None):
        """贝叶斯优化参数空间（针对透析数据优化）"""
        if is_linear:
            # 针对大规模透析数据的线性SVM优化
            if data_size and data_size >= 150000:
                # 大规模数据：更宽的C范围，更高的最大迭代次数
                return {
                    'C': Real(1e-4, 1e2, prior='log-uniform'),  # 扩大C的搜索范围
                    'class_weight': Categorical([None, 'balanced']),
                    'max_iter': Integer(2000, 15000),  # 增加迭代次数
                    'loss': Categorical(['squared_hinge']),
                    'penalty': Categorical(['l2']),
                    'dual': Categorical([False]),  # 大数据集推荐使用primal形式
                    'tol': Real(1e-5, 1e-3, prior='log-uniform')  # 添加容忍度调节
                }
            else:
                # 中小规模数据的标准配置
                return {
                    'C': Real(1e-4, 1e3, prior='log-uniform'),
                    'class_weight': Categorical([None, 'balanced']),
                    'max_iter': Integer(1000, 10000),
                    'loss': Categorical(['squared_hinge']),
                    'penalty': Categorical(['l2']),
                    'dual': Categorical([False]),
                    'tol': Real(1e-5, 1e-2, prior='log-uniform')
                }
        else:
            # 非线性SVM针对透析数据的优化
            if data_size and data_size >= 50000:
                # 中大规模数据：限制复杂度，提升训练效率
                return {
                    'C': Real(1e-2, 1e2, prior='log-uniform'),  # 适中的C范围
                    'gamma': Categorical(['scale', 'auto']),  # 使用自适应gamma
                    'kernel': Categorical(['rbf']),  # 只使用RBF核
                    'class_weight': Categorical([None, 'balanced']),
                    'shrinking': Categorical([True]),  # 启用shrinking提升效率
                    'probability': Categorical([True]),
                    'cache_size': Categorical([500, 1000]),  # 增加缓存大小
                    'tol': Real(1e-4, 1e-2, prior='log-uniform')
                }
            else:
                # 小规模数据：允许更复杂的模型
                return {
                    'C': Real(1e-4, 1e3, prior='log-uniform'),
                    'gamma': Categorical(['scale', 'auto']),
                    'kernel': Categorical(['rbf', 'poly', 'sigmoid']),
                    'class_weight': Categorical([None, 'balanced']),
                    'degree': Integer(2, 5),
                    'coef0': Real(-1.0, 1.0),
                    'shrinking': Categorical([True, False]),
                    'probability': Categorical([True]),
                    'tol': Real(1e-5, 1e-2, prior='log-uniform')
                }
    
    @staticmethod
    def _get_random_params(is_linear, data_size=None):
        """随机搜索参数空间（针对透析数据优化）"""
        if is_linear:
            # 针对大规模透析数据的线性SVM优化
            if data_size and data_size >= 150000:
                return {
                    'C': loguniform(1e-4, 1e2),  # 适合大数据的C范围
                    'class_weight': ['balanced', None],
                    'max_iter': [2000, 5000, 10000, 15000],  # 增加迭代次数
                    'loss': ['squared_hinge'],
                    'penalty': ['l2'],
                    'dual': [False],  # 大数据集使用primal
                    'tol': loguniform(1e-5, 1e-3)
                }
            else:
                return {
                    'C': loguniform(1e-6, 1e6),
                    'class_weight': ['balanced', None],
                    'max_iter': [1000, 5000, 10000, 20000],
                    'loss': ['squared_hinge'],
                    'penalty': ['l2'],
                    'dual': [False]
                }
        else:
            # 非线性SVM针对透析数据的优化
            if data_size and data_size >= 50000:
                # 中大规模数据：简化参数空间
                return {
                    'C': loguniform(1e-2, 1e2),
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],  # 减少gamma选项
                    'kernel': ['rbf'],  # 只使用RBF核
                    'class_weight': ['balanced', None],
                    'shrinking': [True],  # 固定使用shrinking
                    'probability': [True],
                    'cache_size': [500, 1000],  # 增加缓存
                    'tol': loguniform(1e-4, 1e-2)
                }
            else:
                return {
                    'C': loguniform(1e-4, 1e3),
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                    'kernel': ['rbf', 'poly', 'sigmoid'],
                    'class_weight': ['balanced', None],
                    'degree': [2, 3, 4, 5],
                    'coef0': uniform(-1.0, 2.0),
                    'shrinking': [True, False],
                    'probability': [True],
                    'tol': loguniform(1e-5, 1e-2)
                }
    
    @staticmethod
    def _get_grid_params(is_linear, data_size=None):
        """网格搜索参数空间（针对透析数据优化）"""
        if is_linear:
            # 针对大规模透析数据的线性SVM优化
            if data_size and data_size >= 150000:
                return {
                    'C': [0.01, 0.1, 1, 10],  # 减少C的选项以提升效率
                    'class_weight': ['balanced', None],
                    'max_iter': [5000, 10000],  # 增加迭代次数
                    'loss': ['squared_hinge'],
                    'penalty': ['l2'],
                    'dual': [False]
                }
            else:
                return {
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'class_weight': ['balanced', None],
                    'max_iter': [1000, 5000, 10000],
                    'loss': ['squared_hinge'],
                    'penalty': ['l2'],
                    'dual': [False]
                }
        else:
            # 非线性SVM针对透析数据的优化
            if data_size and data_size >= 50000:
                # 中大规模数据：简化网格搜索
                return {
                    'C': [0.1, 1, 10],  # 减少C的选项
                    'gamma': ['scale', 'auto'],  # 只使用自适应gamma
                    'kernel': ['rbf'],  # 只使用RBF核
                    'class_weight': ['balanced', None],
                    'probability': [True]
                }
            else:
                return {
                    'C': [0.001, 0.01, 0.1, 1, 10],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                    'kernel': ['rbf', 'poly', 'sigmoid'],
                    'class_weight': ['balanced', None],
                    'degree': [2, 3, 4],
                    'probability': [True]
                }
    
    @staticmethod
    def _get_default_params(is_linear, data_size=None):
        """默认参数空间（简化版，针对透析数据优化）"""
        if is_linear:
            # 针对大规模透析数据的线性SVM优化
            if data_size and data_size >= 150000:
                return {
                    'C': [1, 10],  # 简化C选项
                    'class_weight': ['balanced'],  # 透析数据通常不平衡
                    'max_iter': [10000]  # 增加迭代次数
                }
            else:
                return {
                    'C': [0.1, 1, 10],
                    'class_weight': ['balanced', None],
                    'max_iter': [5000]
                }
        else:
            # 非线性SVM针对透析数据的优化
            if data_size and data_size >= 50000:
                return {
                    'C': [1, 10],  # 简化C选项
                    'gamma': ['scale'],  # 使用scale gamma
                    'kernel': ['rbf'],  # 只使用RBF核
                    'class_weight': ['balanced'],  # 透析数据通常不平衡
                    'probability': [True]
                }
            else:
                return {
                    'C': [0.1, 1, 10],
                    'gamma': ['scale', 'auto'],
                    'kernel': ['rbf'],
                    'class_weight': ['balanced', None],
                    'probability': [True]
                }


class OptimizedSVMClassifier:
    """优化的SVM分类器类"""
    
    def __init__(self, random_state=42, n_jobs=-1):
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = None
        self.model_type = None
        
    def _determine_model_type(self, X_train, y_train):
        """根据数据特征自动确定模型类型"""
        n_samples, n_features = X_train.shape
        n_classes = len(np.unique(y_train))
        
        # 大数据集或多分类问题使用LinearSVC
        if n_samples > 10000 or n_classes > 2:
            return 'linear'
        else:
            return 'nonlinear'
    
    def _create_base_model(self, model_type):
        """创建基础模型"""
        if model_type == 'linear':
            return LinearSVC(random_state=self.random_state, dual=False)
        else:
            return SVC(random_state=self.random_state, probability=True)
    
    def _get_scoring_metrics(self, is_binary=True):
        """获取评估指标（针对透析低血压预测优化）"""
        if is_binary:
            # 透析低血压预测：优先考虑召回率、特异性和平衡准确率
            return {
                'roc_auc': 'roc_auc',
                'f1': 'f1',
                'recall': 'recall',
                'precision': 'precision',
                'accuracy': 'accuracy',
                'balanced_accuracy': 'balanced_accuracy'
            }
        else:
            # 多分类任务：使用加权指标
            return {
                'f1_weighted': 'f1_weighted',
                'recall_weighted': 'recall_weighted', 
                'precision_weighted': 'precision_weighted',
                'accuracy': 'accuracy'
            }
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, 
            class_weights=None, search_type='bayesian', cv=5, n_iter=50, 
            use_validation_in_training=True, model_type='auto'):
        """训练SVM模型"""
        print(f"\n=== SVM模型训练开始 ===")
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
        
        # 确定模型类型
        if model_type == 'auto':
            self.model_type = self._determine_model_type(X_train, y_train)
        else:
            self.model_type = model_type
        print(f"使用模型类型: {self.model_type}")
        
        # 检查是否为二分类
        unique_labels = np.unique(y_train)
        is_binary = len(unique_labels) == 2
        print(f"任务类型: {'二分类' if is_binary else f'{len(unique_labels)}分类'}")
        
        # 准备训练数据
        if use_validation_in_training and X_val is not None:
            print("将验证集合并到训练集中进行超参数优化...")
            X_combined = np.vstack([X_train_scaled, X_val_scaled])
            y_combined = np.hstack([y_train, y_val])
            print(f"合并后训练集形状: {X_combined.shape}")
        else:
            X_combined = X_train_scaled
            y_combined = y_train
        
        # 创建基础模型
        base_model = self._create_base_model(self.model_type)
        
        # 获取参数空间
        param_grid = SVMParameterManager.get_param_space(
            search_type=search_type, 
            model_type=self.model_type,
            data_size=len(X_combined)
        )
        
        # 处理自定义类权重
        if class_weights and isinstance(class_weights, dict):
            if search_type == 'bayesian' and BAYESIAN_AVAILABLE:
                # 贝叶斯优化暂时不支持自定义权重字典
                param_grid['class_weight'] = Categorical([None, 'balanced'])
            else:
                param_grid['class_weight'] = ['balanced', None, class_weights]
        
        # 获取评估指标（针对透析低血压预测优化）
        scoring = self._get_scoring_metrics(is_binary)
        # 透析低血压预测：优先考虑召回率，减少漏诊风险
        if is_binary:
            refit_metric = 'recall'  # 二分类优先使用召回率
            print("透析低血压预测任务：使用召回率作为主要评估指标，减少漏诊风险")
        else:
            refit_metric = 'f1_weighted'  # 多分类使用加权F1
        
        # 设置交叉验证（根据数据规模优化）
        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        # 根据数据规模调整搜索参数
        data_size = len(X_combined)
        if data_size >= 150000:
            # 大规模数据：减少迭代次数，提升效率
            adjusted_n_iter = min(n_iter, 50)
            verbose_level = 0  # 减少输出
            print(f"大规模透析数据({data_size}样本)：调整搜索参数以提升效率")
        elif data_size >= 50000:
            # 中等规模数据：适中的搜索强度
            adjusted_n_iter = min(n_iter, 80)
            verbose_level = 1
            print(f"中等规模透析数据({data_size}样本)：使用平衡的搜索策略")
        else:
            # 小规模数据：允许更充分的搜索
            adjusted_n_iter = n_iter
            verbose_level = 1
            print(f"小规模透析数据({data_size}样本)：使用充分的搜索策略")
        
        # 超参数搜索
        print(f"开始{search_type}搜索超参数（{cv}折交叉验证，{adjusted_n_iter}次迭代）...")
        if search_type == 'bayesian' and BAYESIAN_AVAILABLE:
            search = BayesSearchCV(
                base_model, param_grid, n_iter=adjusted_n_iter, cv=cv_strategy,
                scoring=scoring, refit=refit_metric, n_jobs=self.n_jobs,
                random_state=self.random_state, verbose=verbose_level, return_train_score=True
            )
        elif search_type == 'random':
            search = RandomizedSearchCV(
                base_model, param_grid, n_iter=adjusted_n_iter, cv=cv_strategy,
                scoring=scoring, refit=refit_metric, n_jobs=self.n_jobs,
                random_state=self.random_state, verbose=verbose_level, return_train_score=True
            )
        else:
            search = GridSearchCV(
                base_model, param_grid, cv=cv_strategy,
                scoring=scoring, refit=refit_metric, n_jobs=self.n_jobs,
                verbose=verbose_level, return_train_score=True
            )
        
        # 训练模型
        search.fit(X_combined, y_combined)
        self.model = search.best_estimator_
        
        # 如果有自定义类权重且使用贝叶斯优化，重新训练
        if (search_type == 'bayesian' and BAYESIAN_AVAILABLE and 
            class_weights and isinstance(class_weights, dict)):
            print("应用自定义类权重重新训练最佳模型...")
            best_params = search.best_params_.copy()
            best_params['class_weight'] = class_weights
            
            # 创建最终模型
            final_model = self._create_base_model(self.model_type)
            final_model.set_params(**best_params)
            final_model.fit(X_combined, y_combined)
            self.model = final_model
            print(f"已应用自定义类权重: {class_weights}")
        
        self.is_fitted = True
        
        print(f"最佳参数: {search.best_params_}")
        print(f"最佳CV分数: {search.best_score_:.4f}")
        
        # 评估模型
        train_score = self.model.score(X_train_scaled, y_train)
        print(f"原始训练集准确率: {train_score:.4f}")
        
        if X_val is not None:
            val_score = self.model.score(X_val_scaled, y_val)
            print(f"验证集准确率: {val_score:.4f}")
        
        return search
    
    def predict(self, X):
        """预测"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        if hasattr(X, 'columns'):
            X = X.values
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """预测概率"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        if hasattr(X, 'columns'):
            X = X.values
        X_scaled = self.scaler.transform(X)
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_scaled)
        else:
            # LinearSVC没有predict_proba，使用decision_function
            decision = self.model.decision_function(X_scaled)
            if decision.ndim == 1:
                # 二分类
                proba = 1 / (1 + np.exp(-decision))
                return np.column_stack([1 - proba, proba])
            else:
                # 多分类
                exp_decision = np.exp(decision)
                return exp_decision / exp_decision.sum(axis=1, keepdims=True)
    
    def get_feature_importance(self):
        """获取特征重要性"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        if hasattr(self.model, 'coef_'):
            coef = np.abs(self.model.coef_)
            if coef.ndim > 1:
                importance = np.mean(coef, axis=0)
            else:
                importance = coef[0]
            
            if self.feature_names:
                return pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': importance
                }).sort_values('importance', ascending=False)
            else:
                return importance
        else:
            print("警告: 当前模型不支持特征重要性计算")
            return None


def C_SVM_model(X_train, X_test, y_train, y_test, X_val, y_val, 
                class_weights, target, kinds, search_type='bayesian', 
                use_optimization=True, n_iter=100, cv_folds=5, 
                use_validation_in_training=True, base_path=None):
    """
    训练和评估SVM模型（支持贝叶斯优化和5折交叉验证）
    
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
    search_type : str, default='bayesian'
        超参数搜索类型: 'bayesian', 'random', 'grid'
    use_optimization : bool, default=True
        是否使用超参数优化
    n_iter : int, default=100
        贝叶斯/随机搜索的迭代次数
    cv_folds : int, default=5
        交叉验证折数
    use_validation_in_training : bool, default=True
        是否将验证集合并到训练集中进行超参数优化
    base_path : str, optional
        输出文件的基础路径
        
    Returns:
    --------
    model : trained model
        训练好的SVM模型
    eval_scores : dict
        评估分数字典
    """
    
    # 初始化路径管理器
    path_manager = PathManager(base_path)
    dirs = path_manager.create_model_dirs("SVM", target)
    
    print(f"\n=== SVM模型训练: {target} ===")
    
    # 数据集统计信息（针对透析数据的详细分析）
    total_samples = len(X_train) + len(X_val) + len(X_test)
    train_pos = np.sum(y_train == 1)
    val_pos = np.sum(y_val == 1)
    test_pos = np.sum(y_test == 1)
    
    print(f"数据集规模分析:")
    print(f"  总样本数: {total_samples:,}")
    print(f"  训练集: {len(X_train):,} 样本 (正例: {train_pos:,}, 负例: {len(X_train)-train_pos:,}, 正例率: {train_pos/len(X_train):.3f})")
    print(f"  验证集: {len(X_val):,} 样本 (正例: {val_pos:,}, 负例: {len(X_val)-val_pos:,}, 正例率: {val_pos/len(X_val):.3f})")
    print(f"  测试集: {len(X_test):,} 样本 (正例: {test_pos:,}, 负例: {len(X_test)-test_pos:,}, 正例率: {test_pos/len(X_test):.3f})")
    
    # 透析数据特征分析
    if hasattr(X_train, 'shape'):
        print(f"\n特征维度分析:")
        print(f"  特征数量: {X_train.shape[1]}")
        print(f"  特征密度: {X_train.shape[1]/len(X_train):.6f} (特征数/样本数)")
        
        # 数据规模分类
        if total_samples >= 150000:
            print(f"  数据规模: 大规模透析数据集 (>=15万样本)")
            print(f"  建议策略: 使用线性SVM，优化训练效率")
        elif total_samples >= 50000:
            print(f"  数据规模: 中等规模透析数据集 (5-15万样本)")
            print(f"  建议策略: 平衡模型复杂度与训练效率")
        else:
            print(f"  数据规模: 小规模透析数据集 (<5万样本)")
            print(f"  建议策略: 可使用复杂模型获得更好性能")
    
    print(f"\n结果将保存到: {dirs['main']}")
    
    if use_optimization:
        # 使用优化版本
        svm_classifier = OptimizedSVMClassifier()
        
        # 训练模型
        search_result = svm_classifier.fit(
            X_train, y_train, X_val, y_val, 
            class_weights=class_weights, search_type=search_type,
            n_iter=n_iter, cv=cv_folds, 
            use_validation_in_training=use_validation_in_training
        )
        
        # 预测
        y_pred_train = svm_classifier.predict(X_train)
        y_pred_val = svm_classifier.predict(X_val)
        y_pred_test = svm_classifier.predict(X_test)
        
        # 计算评估指标（针对透析低血压预测优化）
        eval_scores = {
            'training_accuracy': accuracy_score(y_train, y_pred_train),
            'validation_accuracy': accuracy_score(y_val, y_pred_val),
            'testing_accuracy': accuracy_score(y_test, y_pred_test),
            'training_f1': f1_score(y_train, y_pred_train, average='weighted'),
            'validation_f1': f1_score(y_val, y_pred_val, average='weighted'),
            'testing_f1': f1_score(y_test, y_pred_test, average='weighted'),
            'training_recall': recall_score(y_train, y_pred_train, average='weighted'),
            'validation_recall': recall_score(y_val, y_pred_val, average='weighted'),
            'testing_recall': recall_score(y_test, y_pred_test, average='weighted'),
            'training_precision': precision_score(y_train, y_pred_train, average='weighted'),
            'validation_precision': precision_score(y_val, y_pred_val, average='weighted'),
            'testing_precision': precision_score(y_test, y_pred_test, average='weighted'),
            'training_balanced_accuracy': balanced_accuracy_score(y_train, y_pred_train),
            'validation_balanced_accuracy': balanced_accuracy_score(y_val, y_pred_val),
            'testing_balanced_accuracy': balanced_accuracy_score(y_test, y_pred_test)
        }
        
        # 二分类特殊处理
        unique_labels = np.unique(y_train)
        if len(unique_labels) == 2:
            try:
                y_proba_test = svm_classifier.predict_proba(X_test)[:, 1]
                eval_scores['testing_roc_auc'] = roc_auc_score(y_test, y_proba_test)
            except Exception as e:
                print(f"ROC AUC计算失败: {e}")
        
        # 打印详细评估结果（针对透析低血压预测的临床解释）
        print(f"\n=== 模型评估结果 ===")
        print(f"\n基础性能指标:")
        for metric, score in eval_scores.items():
            print(f"  {metric}: {score:.4f}")
        
        # 透析低血压预测的临床解释
        unique_labels = np.unique(y_train)
        if len(unique_labels) == 2:
            test_recall = eval_scores.get('testing_recall', 0)
            test_precision = eval_scores.get('testing_precision', 0)
            test_balanced_acc = eval_scores.get('testing_balanced_accuracy', 0)
            
            print(f"\n透析低血压预测临床意义:")
            print(f"  召回率 (敏感性): {test_recall:.4f} - 检出真实低血压事件的能力")
            print(f"  精确率 (阳性预测值): {test_precision:.4f} - 预测为低血压时的准确性")
            print(f"  平衡准确率: {test_balanced_acc:.4f} - 综合考虑敏感性和特异性")
            
            # 临床建议
            if test_recall >= 0.85:
                print(f"  ✓ 高召回率: 模型能有效识别大部分低血压风险，有助于减少漏诊")
            elif test_recall >= 0.70:
                print(f"  ⚠ 中等召回率: 模型有一定的低血压识别能力，但仍有改进空间")
            else:
                print(f"  ❌ 低召回率: 模型可能遗漏较多低血压事件，需要优化")
                
            if test_precision >= 0.70:
                print(f"  ✓ 良好精确率: 预测的低血压事件大多准确，减少不必要的干预")
            else:
                print(f"  ⚠ 精确率有待提升: 可能产生较多假阳性，需要平衡敏感性和特异性")
        
        # 混淆矩阵
        print(f"\n测试集混淆矩阵:")
        cm = confusion_matrix(y_test, y_pred_test)
        print(cm)
        
        # 分类报告
        print(f"\n测试集分类报告:")
        print(classification_report(y_test, y_pred_test))
        
        # 特征重要性
        feature_importance = svm_classifier.get_feature_importance()
        if feature_importance is not None:
            print(f"\n前10个重要特征:")
            if isinstance(feature_importance, pd.DataFrame):
                print(feature_importance.head(10))
                # 保存特征重要性
                importance_path = dirs['importance'] / f'SVM_feature_importance_{target}.csv'
                feature_importance.to_csv(importance_path, index=False)
                print(f"特征重要性已保存到: {importance_path}")
        
        # 保存模型
        model_path = dirs['models'] / f'SVM_model_{target}.joblib'
        joblib.dump(svm_classifier, model_path)
        print(f"模型已保存到: {model_path}")
        
        # 保存评估结果
        results_path = dirs['reports'] / f'SVM_evaluation_{target}.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(eval_scores, f, indent=2, ensure_ascii=False)
        print(f"评估结果已保存到: {results_path}")
        
        # 可视化（如果yellowbrick可用）
        if YELLOWBRICK_AVAILABLE:
            try:
                _create_visualizations(svm_classifier.model, X_train, y_train, X_test, y_test, 
                                     dirs['plots'], target, unique_labels)
            except Exception as e:
                print(f"可视化创建失败: {e}")
        
        return svm_classifier.model, eval_scores
    
    else:
        # 使用简化版本
        print("使用简化SVM实现...")
        
        # 标准化数据
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # 创建和训练模型
        svm = LinearSVC(random_state=42, max_iter=5000)
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'class_weight': ['balanced', class_weights] if class_weights else ['balanced']
        }
        
        grid_search = GridSearchCV(
            svm, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train_scaled, y_train)
        model = grid_search.best_estimator_
        
        print(f"最佳参数: {grid_search.best_params_}")
        
        # 评估
        eval_scores = {
            'training_score': model.score(X_train_scaled, y_train),
            'validation_score': model.score(X_val_scaled, y_val),
            'testing_score': model.score(X_test_scaled, y_test)
        }
        
        # 保存模型
        model_path = dirs['models'] / f'SVM_model_{target}.joblib'
        joblib.dump({'model': model, 'scaler': scaler}, model_path)
        print(f"模型已保存到: {model_path}")
        
        return model, eval_scores


def _create_visualizations(model, X_train, y_train, X_test, y_test, output_dir, target, unique_labels):
    """创建完整的yellowbrick性能可视化图表"""
    if not YELLOWBRICK_AVAILABLE:
        print("Yellowbrick不可用，跳过可视化")
        return
    
    print(f"正在创建SVM模型的性能可视化图表 - {target}")
    
    try:
        # 1. 类别平衡可视化
        print("创建类别平衡图...")
        viz = ClassBalance(title=f"SVM Class Balance - {target}")
        viz.fit(y_train)
        viz.show(outpath=output_dir / f'SVM_class_balance_{target}.pdf')
        
        # 2. ROC曲线（仅二分类）
        if len(unique_labels) == 2:
            print("创建ROC曲线...")
            viz = ROCAUC(model, title=f"SVM ROC Curve - {target}")
            viz.fit(X_train, y_train)
            viz.score(X_test, y_test)
            viz.show(outpath=output_dir / f'SVM_roc_curve_{target}.pdf')
            
            # 3. 精确率-召回率曲线（仅二分类）
            print("创建精确率-召回率曲线...")
            viz = PrecisionRecallCurve(model, title=f"SVM Precision-Recall Curve - {target}")
            viz.fit(X_train, y_train)
            viz.score(X_test, y_test)
            viz.show(outpath=output_dir / f'SVM_precision_recall_{target}.pdf')
            
            # 4. 判别阈值可视化（仅二分类）
            print("创建判别阈值图...")
            viz = DiscriminationThreshold(model, title=f"SVM Discrimination Threshold - {target}")
            viz.fit(X_train, y_train)
            viz.score(X_test, y_test)
            viz.show(outpath=output_dir / f'SVM_discrimination_threshold_{target}.pdf')
        
        # 5. 分类报告
        print("创建分类报告...")
        viz = ClassificationReport(model, title=f"SVM Classification Report - {target}")
        viz.fit(X_train, y_train)
        viz.score(X_test, y_test)
        viz.show(outpath=output_dir / f'SVM_classification_report_{target}.pdf')
        
        # 6. 混淆矩阵
        print("创建混淆矩阵...")
        viz = ConfusionMatrix(model, classes=unique_labels, title=f"SVM Confusion Matrix - {target}")
        viz.fit(X_train, y_train)
        viz.score(X_test, y_test)
        viz.show(outpath=output_dir / f'SVM_confusion_matrix_{target}.pdf')
        
        # 7. 类预测错误可视化
        print("创建类预测错误图...")
        viz = ClassPredictionError(model, classes=unique_labels, title=f"SVM Class Prediction Error - {target}")
        viz.fit(X_train, y_train)
        viz.score(X_test, y_test)
        viz.show(outpath=output_dir / f'SVM_class_prediction_error_{target}.pdf')
        
        print(f"✓ SVM模型所有可视化图表已保存到: {output_dir}")
        
    except Exception as e:
        print(f"❌ SVM可视化创建过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


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
    model, scores = C_SVM_model(
        X_train, X_test, y_train, y_test, X_val, y_val,
        class_weights, "test_target", "test_model"
    )
    
    print("测试完成!")
