import pandas as pd
import numpy as np
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_curve, auc, f1_score, make_scorer, recall_score, roc_auc_score,
    confusion_matrix, classification_report, accuracy_score, precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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
    from yellowbrick.classifier import ROCAUC, ClassificationReport, ConfusionMatrix
    YELLOWBRICK_AVAILABLE = True
except ImportError:
    YELLOWBRICK_AVAILABLE = False
    print("Warning: yellowbrick not available. Some visualizations will be skipped.")


class OptimizedSVMClassifier:
    """优化的SVM分类器类"""
    
    def __init__(self, random_state=42, n_jobs=-1):
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = None
        
    def _create_output_dirs(self, base_path):
        """创建输出目录"""
        dirs = ['Results/SVM', 'Results/SVM/Output', 'Results/SVM/Importance']
        for dir_name in dirs:
            full_path = Path(base_path) / dir_name
            full_path.mkdir(parents=True, exist_ok=True)
        return Path(base_path) / 'Results/SVM'
    
    def _get_param_grid(self, search_type='grid', is_linear=False):
        """获取参数网格"""
        if search_type == 'bayesian' and BAYESIAN_AVAILABLE:
            if is_linear:
                return {
                    'C': Real(1e-4, 1e3, prior='log-uniform'),
                    'class_weight': Categorical([None, 'balanced']),
                    'max_iter': Integer(1000, 10000),
                    'loss': Categorical(['hinge', 'squared_hinge']),
                    'penalty': Categorical(['l1', 'l2']),
                    'dual': Categorical([True, False]),
                    'tol': Real(1e-5, 1e-2, prior='log-uniform')
                }
            else:
                return {
                    'C': Real(1e-4, 1e3, prior='log-uniform'),
                    'gamma': Categorical(['scale', 'auto']) if np.random.random() > 0.5 else Real(1e-5, 1e1, prior='log-uniform'),
                    'kernel': Categorical(['rbf', 'poly', 'sigmoid', 'linear']),
                    'class_weight': Categorical([None, 'balanced']),
                    'degree': Integer(2, 6),
                    'coef0': Real(-1.0, 1.0),
                    'shrinking': Categorical([True, False]),
                    'probability': Categorical([True, False]),
                    'tol': Real(1e-5, 1e-2, prior='log-uniform'),
                    'cache_size': Integer(200, 1000)
                }
        elif search_type == 'grid':
            if is_linear:
                return {
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'class_weight': ['balanced', None],
                    'max_iter': [1000, 5000, 10000]
                }
            else:
                return {
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                    'kernel': ['rbf', 'poly', 'sigmoid'],
                    'class_weight': ['balanced', None],
                    'degree': [2, 3, 4, 5]
                }
        elif search_type == 'random':
            if is_linear:
                return {
                    'C': loguniform(1e-6, 1e6),
                    'class_weight': ['balanced', None],
                    'max_iter': [1000, 5000, 10000, 20000]
                }
            else:
                return {
                    'C': loguniform(1e-4, 1e3),
                    'gamma': ['scale', 'auto', 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10],
                    'kernel': ['rbf', 'poly', 'sigmoid', 'linear'],
                    'class_weight': ['balanced', None],
                    'degree': [2, 3, 4, 5, 6],
                    'coef0': uniform(-1.0, 2.0),
                    'shrinking': [True, False],
                    'probability': [True, False],
                    'tol': loguniform(1e-5, 1e-2)
                }
    
    def _get_scoring_metrics(self, is_binary=True):
        """获取评估指标"""
        if is_binary:
            return {
                'roc_auc': 'roc_auc',
                'f1': 'f1',
                'recall': 'recall',
                'precision': 'precision',
                'accuracy': 'accuracy'
            }
        else:
            return {
                'f1_weighted': 'f1_weighted',
                'recall_weighted': 'recall_weighted',
                'precision_weighted': 'precision_weighted',
                'accuracy': 'accuracy'
            }
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, 
            class_weights=None, search_type='bayesian', cv=5, n_iter=50, use_validation_in_training=True):
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
        
        # 检查是否为二分类
        unique_labels = np.unique(y_train)
        is_binary = len(unique_labels) == 2
        print(f"任务类型: {'二分类' if is_binary else f'{len(unique_labels)}分类'}")
        
        # 如果使用验证集参与训练，合并训练集和验证集
        if use_validation_in_training and X_val is not None:
            print("将验证集合并到训练集中进行超参数优化...")
            X_combined = np.vstack([X_train_scaled, X_val_scaled])
            y_combined = np.hstack([y_train, y_val])
            print(f"合并后训练集形状: {X_combined.shape}")
        else:
            X_combined = X_train_scaled
            y_combined = y_train
        
        # 创建SVM模型
        if is_binary and len(X_combined) < 10000:  # 对于小数据集使用RBF SVM
            base_model = SVC(random_state=self.random_state, probability=True)
            param_grid = self._get_param_grid(search_type, is_linear=False)
            if class_weights:
                if search_type == 'bayesian' and BAYESIAN_AVAILABLE:
                    param_grid['class_weight'] = Categorical([None, 'balanced', class_weights])
                else:
                    param_grid['class_weight'].append(class_weights)
        else:  # 对于大数据集或多分类使用LinearSVC
            base_model = LinearSVC(random_state=self.random_state, dual=False)
            param_grid = self._get_param_grid(search_type, is_linear=True)
            if class_weights:
                if search_type == 'bayesian' and BAYESIAN_AVAILABLE:
                    param_grid['class_weight'] = Categorical([None, 'balanced', class_weights])
                else:
                    param_grid['class_weight'].append(class_weights)
        
        # 获取评估指标
        scoring = self._get_scoring_metrics(is_binary)
        refit_metric = 'roc_auc' if is_binary else 'f1_weighted'
        
        # 设置交叉验证
        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        # 超参数搜索
        print(f"开始{search_type}搜索超参数（{cv}折交叉验证）...")
        if search_type == 'bayesian' and BAYESIAN_AVAILABLE:
            search = BayesSearchCV(
                base_model, param_grid, n_iter=n_iter, cv=cv_strategy,
                scoring=scoring, refit=refit_metric, n_jobs=self.n_jobs,
                random_state=self.random_state, verbose=1, return_train_score=True
            )
        elif search_type == 'random':
            search = RandomizedSearchCV(
                base_model, param_grid, n_iter=n_iter, cv=cv_strategy,
                scoring=scoring, refit=refit_metric, n_jobs=self.n_jobs,
                random_state=self.random_state, verbose=1, return_train_score=True
            )
        else:
            search = GridSearchCV(
                base_model, param_grid, cv=cv_strategy,
                scoring=scoring, refit=refit_metric, n_jobs=self.n_jobs,
                verbose=1, return_train_score=True
            )
        
        # 训练模型
        search.fit(X_combined, y_combined)
        self.model = search.best_estimator_
        self.is_fitted = True
        
        print(f"最佳参数: {search.best_params_}")
        print(f"最佳CV分数: {search.best_score_:.4f}")
        
        # 交叉验证详细结果
        cv_results = pd.DataFrame(search.cv_results_)
        print(f"\n交叉验证统计:")
        print(f"平均训练分数: {cv_results['mean_train_score'].mean():.4f} ± {cv_results['std_train_score'].mean():.4f}")
        print(f"平均验证分数: {cv_results['mean_test_score'].mean():.4f} ± {cv_results['std_test_score'].mean():.4f}")
        
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
                proba = np.exp(decision) / (1 + np.exp(decision))
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
            # 获取系数的绝对值作为特征重要性
            coef = np.abs(self.model.coef_)
            if coef.ndim > 1:
                # 多分类情况，取平均值
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
    
    # 设置基础路径
    if base_path is None:
        base_path = "/home/cht/Works/PredictionTimeHypotensionDialysis/透前模型"
    
    print(f"\n=== SVM模型训练: {target} ===")
    print(f"训练集长度: {len(X_train)}, 标签1数量: {np.sum(y_train == 1)}")
    print(f"验证集长度: {len(X_val)}, 标签1数量: {np.sum(y_val == 1)}")
    print(f"测试集长度: {len(X_test)}, 标签1数量: {np.sum(y_test == 1)}")
    
    if use_optimization:
        # 使用优化版本
        svm_classifier = OptimizedSVMClassifier()
        
        # 创建输出目录
        output_dir = svm_classifier._create_output_dirs(base_path)
        
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
                y_proba_test = svm_classifier.predict_proba(X_test)[:, 1]
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
        
        # 特征重要性
        feature_importance = svm_classifier.get_feature_importance()
        if feature_importance is not None:
            print(f"\n前10个重要特征:")
            if isinstance(feature_importance, pd.DataFrame):
                print(feature_importance.head(10))
                # 保存特征重要性
                importance_path = output_dir / 'Importance' / f'SVM_feature_importance_target_{target}.csv'
                feature_importance.to_csv(importance_path, index=False)
                print(f"特征重要性已保存到: {importance_path}")
        
        # 保存模型
        model_path = output_dir / f'SVM_model_target_{target}.joblib'
        joblib.dump(svm_classifier, model_path)
        print(f"模型已保存到: {model_path}")
        
        # 可视化（如果yellowbrick可用）
        if YELLOWBRICK_AVAILABLE:
            try:
                _create_visualizations(svm_classifier.model, X_train, y_train, X_test, y_test, 
                                     output_dir, target, unique_labels)
            except Exception as e:
                print(f"可视化创建失败: {e}")
        
        return svm_classifier.model, eval_scores
    
    else:
        # 使用原始版本（简化）
        print("使用原始SVM实现...")
        
        # 创建输出目录
        output_dir = Path(base_path) / 'Results/SVM'
        output_dir.mkdir(parents=True, exist_ok=True)
        
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
        model_path = output_dir / f'SVM_model_target_{target}.joblib'
        joblib.dump({'model': model, 'scaler': scaler}, model_path)
        
        return model, eval_scores


def _create_visualizations(model, X_train, y_train, X_test, y_test, output_dir, target, unique_labels):
    """创建可视化图表"""
    if not YELLOWBRICK_AVAILABLE:
        return
    
    try:
        # 分类报告
        viz = ClassificationReport(model, title=f"SVM Classification Report - {target}")
        viz.fit(X_train, y_train)
        viz.score(X_test, y_test)
        viz.show(outpath=output_dir / 'Output' / f'SVM_classification_report_{target}.png')
        
        # ROC曲线（仅二分类）
        if len(unique_labels) == 2:
            viz = ROCAUC(model, title=f"SVM ROC Curve - {target}")
            viz.fit(X_train, y_train)
            viz.score(X_test, y_test)
            viz.show(outpath=output_dir / 'Output' / f'SVM_roc_curve_{target}.png')
        
        # 混淆矩阵
        viz = ConfusionMatrix(model, classes=unique_labels, title=f"SVM Confusion Matrix - {target}")
        viz.fit(X_train, y_train)
        viz.score(X_test, y_test)
        viz.show(outpath=output_dir / 'Output' / f'SVM_confusion_matrix_{target}.png')
        
        print("可视化图表已保存")
        
    except Exception as e:
        print(f"可视化创建过程中出现错误: {e}")


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
