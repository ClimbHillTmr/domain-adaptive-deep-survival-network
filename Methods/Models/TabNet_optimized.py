import pandas as pd
import numpy as np
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import pickle
import joblib
from itertools import product
from sklearn.model_selection import StratifiedKFold, train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, roc_curve, auc,
    confusion_matrix, classification_report, recall_score, precision_score
)
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import multiprocessing

# 尝试导入贝叶斯优化库
try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer, Categorical
    BAYESIAN_AVAILABLE = True
except ImportError:
    print("Warning: scikit-optimize not available. Falling back to RandomizedSearchCV.")
    BAYESIAN_AVAILABLE = False

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    from pytorch_tabnet.augmentations import ClassificationSMOTE
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False
    print("Warning: pytorch_tabnet not available. TabNet models will be skipped.")

# 设置多进程启动方法
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)


class OptimizedTabNetClassifier:
    """优化的TabNet分类器类"""
    
    def __init__(self, random_state=42, device_name='cpu'):
        if not TABNET_AVAILABLE:
            raise ImportError("pytorch_tabnet is required for TabNet models")
            
        self.random_state = random_state
        self.device_name = device_name
        self.model = None
        self.is_fitted = False
        self.feature_names = None
        self.best_params = None
        self.best_score = None
        self.cv_results = None
        
    def _create_output_dirs(self, base_path):
        """创建输出目录"""
        dirs = ['Results/TabNet', 'Results/TabNet/Output', 'Results/TabNet/Importance']
        for dir_name in dirs:
            full_path = Path(base_path) / dir_name
            full_path.mkdir(parents=True, exist_ok=True)
        return Path(base_path) / 'Results/TabNet'
    
    def get_param_grid(self, search_type='bayesian'):
        """获取参数网格"""
        if search_type == 'bayesian' and BAYESIAN_AVAILABLE:
            return {
                'n_d': Integer(8, 256),
                'n_a': Integer(8, 256),
                'n_steps': Integer(3, 15),
                'gamma': Real(1.0, 3.0),
                'lambda_sparse': Real(1e-8, 1e-1, prior='log-uniform'),
                'lr': Real(0.005, 0.1, prior='log-uniform'),
                'batch_size': Categorical([128, 256, 512, 1024, 2048]),
                'max_epochs': Integer(50, 300),
                'momentum': Real(0.02, 0.4),
                'clip_value': Real(0.5, 2.0)
            }
        elif search_type == 'random':
            return {
                'n_d': [8, 16, 24, 32, 48, 64, 96, 128, 192, 256],
                'n_a': [8, 16, 24, 32, 48, 64, 96, 128, 192, 256],
                'n_steps': [3, 4, 5, 6, 7, 8, 9, 10, 12, 15],
                'gamma': [1.0, 1.2, 1.3, 1.5, 1.8, 2.0, 2.5, 3.0],
                'lambda_sparse': [0, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
                'lr': [5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1],
                'batch_size': [128, 256, 512, 1024, 2048, 4096],
                'max_epochs': [50, 75, 100, 125, 150, 200, 250, 300]
            }
        elif search_type == 'grid':
            return {
                'n_d': [16, 32, 64],
                'n_a': [16, 32, 64, 128],
                'n_steps': [3, 4, 5],
                'gamma': [1.0, 1.3, 1.5, 2.0],
                'lambda_sparse': [1e-6, 1e-5, 1e-4, 1e-3],
                'lr': [1e-3, 2e-3, 5e-3, 1e-2],
                'batch_size': [512, 1024, 2048],
                'max_epochs': [50, 100, 150]
            }
        else:  # 快速测试
            return {
                'n_d': [32],
                'n_a': [32, 64],
                'n_steps': [3, 4],
                'gamma': [1.3, 1.5],
                'lambda_sparse': [1e-4],
                'lr': [2e-3],
                'batch_size': [1024],
                'max_epochs': [50]
            }
    
    def create_model(self, params, n_classes=2):
        """创建TabNet模型"""
        import torch
        
        # 确保所有参数都是标量值，而不是numpy数组
        def extract_scalar(value, default):
            if hasattr(value, '__iter__') and not isinstance(value, str):
                # 如果是数组或列表，取第一个元素
                return float(value[0]) if len(value) > 0 else default
            elif hasattr(value, 'item'):
                # 如果是numpy标量，转换为Python标量
                return float(value.item())
            else:
                return float(value) if value is not None else default
        
        model_params = {
            'n_d': int(extract_scalar(params.get('n_d'), 32)),
            'n_a': int(extract_scalar(params.get('n_a'), 32)),
            'n_steps': int(extract_scalar(params.get('n_steps'), 3)),
            'gamma': extract_scalar(params.get('gamma'), 1.3),
            'lambda_sparse': extract_scalar(params.get('lambda_sparse'), 1e-4),
            'optimizer_fn': torch.optim.Adam,
            'optimizer_params': {'lr': extract_scalar(params.get('lr'), 2e-3)},
            'mask_type': 'entmax',
            'scheduler_params': {'step_size': 10, 'gamma': 0.9},
            'scheduler_fn': torch.optim.lr_scheduler.StepLR,
            'seed': self.random_state,
            'verbose': 1,
            'device_name': self.device_name
        }
        
        return TabNetClassifier(**model_params)
    
    def cross_validate(self, X, y, params, cv_folds=5, scoring='roc_auc'):
        """交叉验证评估参数"""
        print(f"\n交叉验证参数: {params}")
        
        # 检查分类数量
        n_classes = len(np.unique(y))
        is_binary = n_classes == 2
        
        # 根据分类数量调整评估指标
        if scoring == 'roc_auc' and not is_binary:
            scoring = 'f1_weighted'
            print(f"多分类任务，评估指标改为: {scoring}")
        
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        scores = []
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"\n第 {fold + 1}/{cv_folds} 折...")
            
            try:
                X_train_fold = X[train_idx].astype(np.float32)
                X_val_fold = X[val_idx].astype(np.float32)
                y_train_fold = y[train_idx].astype(np.int64)
                y_val_fold = y[val_idx].astype(np.int64)
                
                # 创建模型
                model = self.create_model(params, n_classes=n_classes)
                
                # 根据分类数量选择评估指标
                if is_binary:
                    eval_metrics = ['auc', 'balanced_accuracy']
                else:
                    eval_metrics = ['logloss', 'accuracy']
                
                # 确保参数为标量值
                def extract_scalar(value, default):
                    if hasattr(value, '__iter__') and not isinstance(value, str):
                        return int(value[0]) if len(value) > 0 else default
                    elif hasattr(value, 'item'):
                        return int(value.item())
                    else:
                        return int(value) if value is not None else default
                
                max_epochs_val = extract_scalar(params.get('max_epochs'), 100)
                batch_size_val = extract_scalar(params.get('batch_size'), 1024)
                
                # 训练模型
                model.fit(
                    X_train_fold, y_train_fold,
                    eval_set=[(X_train_fold, y_train_fold), (X_val_fold, y_val_fold)],
                    eval_name=['train', 'valid'],
                    eval_metric=eval_metrics,
                    max_epochs=max_epochs_val,
                    patience=15,
                    batch_size=batch_size_val,
                    virtual_batch_size=batch_size_val,
                    num_workers=0,  # CPU使用单线程
                    weights=1,
                    drop_last=False,
                    augmentations=None,  # 禁用数据增强避免设备问题
                    from_unsupervised=None
                )
                
                # 预测和评估
                if scoring == 'roc_auc' and is_binary:
                    y_pred_proba = model.predict_proba(X_val_fold)[:, 1]
                    score = roc_auc_score(y_val_fold, y_pred_proba)
                elif scoring == 'accuracy':
                    y_pred = model.predict(X_val_fold)
                    score = accuracy_score(y_val_fold, y_pred)
                else:
                    y_pred = model.predict(X_val_fold)
                    score = f1_score(y_val_fold, y_pred, average='weighted')
                
                scores.append(score)
                fold_results.append({
                    'fold': fold + 1,
                    'score': score,
                    'params': params.copy()
                })
                
                print(f"第 {fold + 1} 折 {scoring}: {score:.4f}")
                
            except Exception as e:
                print(f"第 {fold + 1} 折训练失败: {str(e)}")
                scores.append(0.0)  # 失败时给予最低分
        
        cv_result = {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'scores': scores,
            'fold_results': fold_results,
            'params': params
        }
        
        print(f"交叉验证完成 - 平均 {scoring}: {cv_result['mean_score']:.4f} (+/- {cv_result['std_score']:.4f})")
        
        return cv_result
    
    def grid_search(self, X, y, cv_folds=5, scoring='roc_auc', search_type='bayesian', n_iter=100):
        """网格搜索、随机搜索或贝叶斯优化最优参数"""
        param_grid = self.get_param_grid(search_type)
        
        # 设置交叉验证策略
        cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        if search_type == 'bayesian' and BAYESIAN_AVAILABLE:
            print(f"开始贝叶斯优化，共 {n_iter} 次迭代...")
            
            # 创建一个包装器来适配TabNet
            class TabNetWrapper:
                def __init__(self, tabnet_classifier):
                    self.tabnet_classifier = tabnet_classifier
                    
                def fit(self, X, y):
                    return self
                    
                def predict(self, X):
                    return self.tabnet_classifier.predict(X)
                    
                def predict_proba(self, X):
                    return self.tabnet_classifier.predict_proba(X)
                    
                def score(self, X, y):
                    if scoring == 'roc_auc':
                        from sklearn.metrics import roc_auc_score
                        y_pred_proba = self.predict_proba(X)[:, 1]
                        return roc_auc_score(y, y_pred_proba)
                    else:
                        from sklearn.metrics import f1_score
                        y_pred = self.predict(X)
                        return f1_score(y, y_pred, average='weighted')
                        
                def set_params(self, **params):
                    self.params = params
                    return self
                    
                def get_params(self, deep=True):
                    return getattr(self, 'params', {})
            
            # 使用贝叶斯优化
            wrapper = TabNetWrapper(self)
            search = BayesSearchCV(
                estimator=wrapper,
                search_spaces=param_grid,
                n_iter=n_iter,
                cv=cv_strategy,
                scoring=scoring,
                n_jobs=1,  # TabNet不支持并行
                random_state=42,
                refit=True,
                verbose=1
            )
            
            # 自定义评估函数
            def objective(params):
                scores = []
                for train_idx, val_idx in cv_strategy.split(X, y):
                    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                    y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                    
                    try:
                        cv_result = self.cross_validate(X, y, params, cv_folds, scoring)
                        scores.append(cv_result['mean_score'])
                    except Exception as e:
                        print(f"参数组合失败: {str(e)}")
                        scores.append(0.0)
                        
                return np.mean(scores)
            
            # 手动实现贝叶斯优化
            best_score = -1
            best_params = None
            all_results = []
            
            for i in range(n_iter):
                # 从参数空间中采样
                params = {}
                for param_name, param_space in param_grid.items():
                    if hasattr(param_space, 'rvs'):
                        params[param_name] = param_space.rvs()
                    else:
                        params[param_name] = np.random.choice(param_space.categories[0])
                
                print(f"\n测试参数组合 {i+1}/{n_iter}: {params}")
                
                try:
                    cv_result = self.cross_validate(X, y, params, cv_folds, scoring)
                    all_results.append(cv_result)
                    
                    if cv_result['mean_score'] > best_score:
                        best_score = cv_result['mean_score']
                        best_params = params.copy()
                        print(f"发现更好的参数组合！得分: {best_score:.4f}")
                        
                except Exception as e:
                    print(f"参数组合 {i+1} 失败: {str(e)}")
                    continue
                    
        elif search_type == 'random':
            # 随机搜索
            print(f"开始随机搜索，共 {n_iter} 次迭代...")
            param_combinations = []
            param_names = list(param_grid.keys())
            
            for _ in range(n_iter):
                params = {}
                for param_name in param_names:
                    params[param_name] = np.random.choice(param_grid[param_name])
                param_combinations.append(params)
                
            best_score = -1
            best_params = None
            all_results = []
            
            for i, params in enumerate(param_combinations):
                print(f"\n测试参数组合 {i+1}/{len(param_combinations)}: {params}")
                
                try:
                    cv_result = self.cross_validate(X, y, params, cv_folds, scoring)
                    all_results.append(cv_result)
                    
                    if cv_result['mean_score'] > best_score:
                        best_score = cv_result['mean_score']
                        best_params = params.copy()
                        print(f"发现更好的参数组合！得分: {best_score:.4f}")
                        
                except Exception as e:
                    print(f"参数组合 {i+1} 失败: {str(e)}")
                    continue
        else:
            # 网格搜索
            param_combinations = list(product(*param_grid.values()))
            param_names = list(param_grid.keys())
            param_combinations = [dict(zip(param_names, param_values)) 
                                for param_values in param_combinations]
            print(f"开始网格搜索，共 {len(param_combinations)} 个参数组合...")
            
            best_score = -1
            best_params = None
            all_results = []
            
            for i, params in enumerate(param_combinations):
                print(f"\n测试参数组合 {i+1}/{len(param_combinations)}: {params}")
                
                try:
                    cv_result = self.cross_validate(X, y, params, cv_folds, scoring)
                    all_results.append(cv_result)
                    
                    if cv_result['mean_score'] > best_score:
                        best_score = cv_result['mean_score']
                        best_params = params.copy()
                        print(f"发现更好的参数组合！得分: {best_score:.4f}")
                        
                except Exception as e:
                    print(f"参数组合 {i+1} 失败: {str(e)}")
                    continue
        
        self.best_params = best_params
        self.best_score = best_score
        self.cv_results = all_results
        
        print(f"\n搜索完成！")
        print(f"最优参数: {best_params}")
        print(f"最优得分: {best_score:.4f}")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': all_results
        }
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, params=None):
        """训练TabNet模型"""
        print(f"\n=== TabNet模型训练开始 ===")
        print(f"训练集形状: {X_train.shape}")
        print(f"类别分布: {pd.Series(y_train).value_counts().to_dict()}")
        
        # 保存特征名称
        if hasattr(X_train, 'columns'):
            self.feature_names = X_train.columns.tolist()
            X_train = X_train.values
        if X_val is not None and hasattr(X_val, 'columns'):
            X_val = X_val.values
        
        # 数据类型转换
        X_train = X_train.astype(np.float32)
        y_train = y_train.astype(np.int64)
        if X_val is not None:
            X_val = X_val.astype(np.float32)
            y_val = y_val.astype(np.int64)
        
        # 检查分类数量
        n_classes = len(np.unique(y_train))
        is_binary = n_classes == 2
        print(f"任务类型: {'二分类' if is_binary else f'{n_classes}分类'}")
        
        # 使用提供的参数或默认参数
        if params is None:
            params = {
                'n_d': 32, 'n_a': 32, 'n_steps': 3, 'gamma': 1.3,
                'lambda_sparse': 1e-4, 'lr': 2e-3, 'batch_size': 1024, 'max_epochs': 100
            }
        
        # 创建模型
        self.model = self.create_model(params, n_classes)
        
        # 根据分类数量选择评估指标
        if is_binary:
            eval_metrics = ['auc', 'balanced_accuracy']
        else:
            eval_metrics = ['logloss', 'accuracy']
        
        # 训练模型
        if X_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
            eval_name = ['train', 'valid']
        else:
            eval_set = [(X_train, y_train)]
            eval_name = ['train']
        
        # 确保参数为标量值
        def extract_scalar(value, default):
            if hasattr(value, '__iter__') and not isinstance(value, str):
                return int(value[0]) if len(value) > 0 else default
            elif hasattr(value, 'item'):
                return int(value.item())
            else:
                return int(value) if value is not None else default
        
        max_epochs_val = extract_scalar(params.get('max_epochs'), 100)
        batch_size_val = extract_scalar(params.get('batch_size'), 1024)
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            eval_name=eval_name,
            eval_metric=eval_metrics,
            max_epochs=max_epochs_val,
            patience=15,
            batch_size=batch_size_val,
            virtual_batch_size=batch_size_val,
            num_workers=0,
            weights=1,
            drop_last=False,
            augmentations=None,
            from_unsupervised=None
        )
        
        self.is_fitted = True
        print("TabNet模型训练完成")
        
        return self
    
    def predict(self, X):
        """预测"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        if hasattr(X, 'columns'):
            X = X.values
        X = X.astype(np.float32)
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """预测概率"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        if hasattr(X, 'columns'):
            X = X.values
        X = X.astype(np.float32)
        return self.model.predict_proba(X)
    
    def get_feature_importance(self):
        """获取特征重要性"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        try:
            importance = self.model.feature_importances_
            if self.feature_names:
                return pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': importance
                }).sort_values('importance', ascending=False)
            else:
                return importance
        except Exception as e:
            print(f"特征重要性获取失败: {e}")
            return None


def TabNet_model_optimized(X_train, X_test, y_train, y_test, class_weights,
                          X_val=None, y_val=None, target=None, kinds=None,
                          optimize=True, cv_folds=5, search_type='bayesian',
                          n_iter=100, use_validation_in_training=True, base_path=None):
    """
    优化版TabNet模型训练函数
    
    Parameters:
    -----------
    X_train, X_test, X_val : array-like
        训练、测试、验证集的输入特征
    y_train, y_test, y_val : array-like
        训练、测试、验证集的目标标签
    class_weights : dict
        类别权重
    target : str
        目标变量名
    kinds : str
        模型类型
    optimize : bool, default=True
        是否进行参数优化
    cv_folds : int, default=5
        交叉验证折数
    search_type : str, default='grid'
        搜索类型: 'grid', 'random', 'fast'
    n_iter : int, default=20
        随机搜索迭代次数
    base_path : str, optional
        输出文件的基础路径
        
    Returns:
    --------
    model : trained model
        训练好的TabNet模型
    eval_scores : dict
        评估分数字典
    optimization_results : dict or None
        优化结果（如果进行了优化）
    """
    
    if not TABNET_AVAILABLE:
        raise ImportError("pytorch_tabnet is required for TabNet models")
    
    # 设置基础路径
    if base_path is None:
        base_path = "/home/cht/Works/PredictionTimeHypotensionDialysis/透前模型"
    
    print(f"\n=== TabNet模型训练: {target} ===")
    print(f"训练集长度: {len(X_train)}, 标签1数量: {np.sum(y_train == 1)}")
    print(f"测试集长度: {len(X_test)}, 标签1数量: {np.sum(y_test == 1)}")
    if X_val is not None:
        print(f"验证集长度: {len(X_val)}, 标签1数量: {np.sum(y_val == 1)}")
    
    # 创建TabNet分类器
    tabnet_classifier = OptimizedTabNetClassifier()
    
    # 创建输出目录
    output_dir = tabnet_classifier._create_output_dirs(base_path)
    
    # 准备数据
    features_col = X_train.columns if hasattr(X_train, 'columns') else None
    X_train_values = X_train.values if hasattr(X_train, 'values') else X_train
    y_train_values = y_train.values if hasattr(y_train, 'values') else y_train
    X_test_values = X_test.values if hasattr(X_test, 'values') else X_test
    y_test_values = y_test.values if hasattr(y_test, 'values') else y_test
    
    if X_val is not None:
        X_val_values = X_val.values if hasattr(X_val, 'values') else X_val
        y_val_values = y_val.values if hasattr(y_val, 'values') else y_val
    else:
        # 如果没有验证集，从训练集中分割
        X_train_values, X_val_values, y_train_values, y_val_values = train_test_split(
            X_train_values, y_train_values, test_size=0.2, random_state=42,
            stratify=y_train_values
        )
    
    # 确保数据类型正确
    X_train_values = X_train_values.astype(np.float32)
    X_test_values = X_test_values.astype(np.float32)
    X_val_values = X_val_values.astype(np.float32)
    y_train_values = y_train_values.astype(np.int64)
    y_test_values = y_test_values.astype(np.int64)
    y_val_values = y_val_values.astype(np.int64)
    
    optimization_results = None
    
    if optimize:
        print(f"\n开始参数优化（{search_type}搜索）...")
        
        # 准备训练数据
        if use_validation_in_training:
            # 合并训练和验证数据用于交叉验证
            X_combined = np.vstack([X_train_values, X_val_values]).astype(np.float32)
            y_combined = np.hstack([y_train_values, y_val_values]).astype(np.int64)
            print(f"使用验证集参与训练，合并后训练集大小: {X_combined.shape}")
        else:
            X_combined = X_train_values
            y_combined = y_train_values
            print(f"仅使用原始训练集，大小: {X_combined.shape}")
        
        # 根据分类数量选择评估指标
        n_classes = len(np.unique(y_combined))
        scoring = 'roc_auc' if n_classes == 2 else 'f1_weighted'
        
        optimization_results = tabnet_classifier.grid_search(
            X_combined, y_combined, cv_folds=cv_folds, scoring=scoring,
            search_type=search_type, n_iter=n_iter
        )
        
        best_params = optimization_results['best_params']
        print(f"\n使用最优参数训练最终模型: {best_params}")
        
        # 保存优化结果
        results_path = output_dir / f'optimization_results_target_{target}.pkl'
        with open(results_path, 'wb') as f:
            pickle.dump(optimization_results, f)
        print(f"优化结果已保存到: {results_path}")
        
    else:
        # 使用默认参数
        best_params = {
            'n_d': 32, 'n_a': 32, 'n_steps': 3, 'gamma': 1.3,
            'lambda_sparse': 1e-4, 'lr': 2e-3, 'batch_size': 1024, 'max_epochs': 100
        }
        print(f"使用默认参数: {best_params}")
    
    # 使用最优参数训练最终模型
    tabnet_classifier.fit(X_train_values, y_train_values, X_val_values, y_val_values, best_params)
    
    # 保存模型
    model_path = output_dir / f'TabNet_model_target_{target}.joblib'
    joblib.dump(tabnet_classifier, model_path)
    print(f"模型已保存到: {model_path}")
    
    # 评估模型
    try:
        y_pred_train = tabnet_classifier.predict(X_train_values)
        y_pred_val = tabnet_classifier.predict(X_val_values)
        y_pred_test = tabnet_classifier.predict(X_test_values)
        
        eval_scores = {
            'training_accuracy': accuracy_score(y_train_values, y_pred_train),
            'validation_accuracy': accuracy_score(y_val_values, y_pred_val),
            'testing_accuracy': accuracy_score(y_test_values, y_pred_test),
            'training_f1': f1_score(y_train_values, y_pred_train, average='weighted'),
            'validation_f1': f1_score(y_val_values, y_pred_val, average='weighted'),
            'testing_f1': f1_score(y_test_values, y_pred_test, average='weighted'),
            'training_recall': recall_score(y_train_values, y_pred_train, average='weighted'),
            'validation_recall': recall_score(y_val_values, y_pred_val, average='weighted'),
            'testing_recall': recall_score(y_test_values, y_pred_test, average='weighted'),
            'best_params': best_params
        }
        
        # 二分类特殊处理
        n_classes = len(np.unique(y_train_values))
        if n_classes == 2:
            try:
                y_proba_test = tabnet_classifier.predict_proba(X_test_values)[:, 1]
                eval_scores['testing_roc_auc'] = roc_auc_score(y_test_values, y_proba_test)
            except Exception as e:
                print(f"ROC AUC计算失败: {e}")
        
        # 打印详细评估结果
        print(f"\n=== 模型评估结果 ===")
        for metric, score in eval_scores.items():
            if metric != 'best_params':
                print(f"{metric}: {score:.4f}")
        
        # 混淆矩阵
        print(f"\n测试集混淆矩阵:")
        cm = confusion_matrix(y_test_values, y_pred_test)
        print(cm)
        
        # 分类报告
        print(f"\n测试集分类报告:")
        print(classification_report(y_test_values, y_pred_test))
        
    except Exception as e:
        print(f"模型评估失败: {str(e)}")
        eval_scores = {'error': str(e)}
    
    # 特征重要性分析
    try:
        feature_importance = tabnet_classifier.get_feature_importance()
        if feature_importance is not None:
            print(f"\n前10个重要特征:")
            if isinstance(feature_importance, pd.DataFrame):
                print(feature_importance.head(10))
                # 保存特征重要性
                importance_path = output_dir / 'Importance' / f'TabNet_feature_importance_target_{target}.csv'
                feature_importance.to_csv(importance_path, index=False)
                print(f"特征重要性已保存到: {importance_path}")
    except Exception as e:
        print(f"特征重要性分析失败: {str(e)}")
    
    # 绘制ROC曲线（仅二分类）
    if len(np.unique(y_train_values)) == 2:
        try:
            y_proba_test = tabnet_classifier.predict_proba(X_test_values)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_test_values, y_proba_test)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.title(f'ROC Curve - TabNet (Target: {target})')
            plt.plot(fpr, tpr, 'b', label=f'AUC = {roc_auc:.4f}')
            plt.legend(loc='lower right')
            plt.plot([0, 1], [0, 1], 'r--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0])
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            
            roc_path = output_dir / 'Output' / f'TabNet_ROC_curve_target_{target}.png'
            plt.savefig(roc_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"ROC曲线已保存到: {roc_path}")
            
        except Exception as e:
            print(f"ROC曲线绘制失败: {str(e)}")
    
    return tabnet_classifier.model, eval_scores, optimization_results


if __name__ == "__main__":
    # 测试代码
    if TABNET_AVAILABLE:
        from sklearn.datasets import make_classification
        
        # 生成测试数据
        X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        
        # 计算类别权重
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = dict(zip(np.unique(y_train), 
                               compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)))
        
        # 测试模型
        model, scores, opt_results = TabNet_model_optimized(
            X_train, X_test, y_train, y_test, class_weights,
            X_val=X_val, y_val=y_val, target="test_target", kinds="test_model",
            optimize=False  # 快速测试，不进行优化
        )
        
        print("测试完成!")
    else:
        print("TabNet不可用，跳过测试")
