import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
from pathlib import Path
import os
import pickle
import joblib
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# 尝试导入yellowbrick，如果失败则设置标志
try:
    from yellowbrick.classifier import ClassBalance, ROCAUC, PrecisionRecallCurve, DiscriminationThreshold
    from yellowbrick.classifier import ClassificationReport, ConfusionMatrix, ClassPredictionError
    YELLOWBRICK_AVAILABLE = True
except ImportError:
    YELLOWBRICK_AVAILABLE = False
    print("Yellowbrick不可用，将使用基础matplotlib绘图")

# 尝试导入贝叶斯优化库
try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer, Categorical
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False
    print("scikit-optimize不可用，将跳过贝叶斯优化")

# 尝试导入TabNet
try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    import torch
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False
    print("pytorch_tabnet不可用，TabNet模型将无法使用")
    
    # 创建虚拟类
    class TabNetClassifier:
        def __init__(self, **kwargs):
            raise ImportError("pytorch_tabnet is required for TabNet models")

# 设置多进程启动方法
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

class TabNetSimplified:
    """简化版TabNet分类器"""
    
    def __init__(self, random_state=42, device_name='auto'):
        self.random_state = random_state
        self.device_name = device_name
        self.model = None
        self.is_fitted = False
        self.feature_names = None
        self.best_params = None
        self.best_score = None
        self.data_augmentation = True  # 启用数据增强
        self.noise_level = 0.01  # 噪声水平
        self.dropout_rate = 0.1  # 特征dropout率
        
    def _create_output_dirs(self, base_path):
        """创建输出目录"""
        dirs = ['Output', 'Importance']
        for dir_name in dirs:
            full_path = Path(base_path) / dir_name
            full_path.mkdir(parents=True, exist_ok=True)
        return Path(base_path)
    
    def _apply_data_augmentation(self, X, y=None, augment_type='noise'):
        """应用数据增强技术"""
        if not self.data_augmentation:
            return X, y
        
        X_augmented = X.copy()
        
        if augment_type == 'noise':
            # 特征噪声注入
            noise = np.random.normal(0, self.noise_level, X.shape)
            X_augmented = X + noise
            
        elif augment_type == 'dropout':
            # 特征dropout
            dropout_mask = np.random.binomial(1, 1-self.dropout_rate, X.shape)
            X_augmented = X * dropout_mask
            
        elif augment_type == 'mixed':
            # 混合增强：噪声 + dropout
            # 先应用噪声
            noise = np.random.normal(0, self.noise_level, X.shape)
            X_augmented = X + noise
            # 再应用dropout
            dropout_mask = np.random.binomial(1, 1-self.dropout_rate, X.shape)
            X_augmented = X_augmented * dropout_mask
        
        return X_augmented.astype(np.float32), y
    
    def _augment_training_data(self, X_train, y_train, augment_ratio=0.3):
        """增强训练数据"""
        if not self.data_augmentation:
            return X_train, y_train
        
        n_samples = len(X_train)
        n_augment = int(n_samples * augment_ratio)
        
        # 随机选择要增强的样本
        augment_indices = np.random.choice(n_samples, n_augment, replace=True)
        X_to_augment = X_train[augment_indices]
        y_to_augment = y_train[augment_indices]
        
        # 应用不同类型的增强
        augment_types = ['noise', 'dropout', 'mixed']
        X_augmented_list = []
        y_augmented_list = []
        
        for i, aug_type in enumerate(augment_types):
            start_idx = i * (n_augment // 3)
            end_idx = (i + 1) * (n_augment // 3) if i < 2 else n_augment
            
            if start_idx < end_idx:
                X_aug, y_aug = self._apply_data_augmentation(
                    X_to_augment[start_idx:end_idx], 
                    y_to_augment[start_idx:end_idx], 
                    aug_type
                )
                X_augmented_list.append(X_aug)
                y_augmented_list.append(y_aug)
        
        if X_augmented_list:
            X_augmented = np.vstack(X_augmented_list)
            y_augmented = np.hstack(y_augmented_list)
            
            # 合并原始数据和增强数据
            X_combined = np.vstack([X_train, X_augmented])
            y_combined = np.hstack([y_train, y_augmented])
            
            print(f"数据增强完成: 原始样本 {len(X_train)} -> 增强后 {len(X_combined)}")
            return X_combined.astype(np.float32), y_combined.astype(np.int64)
        
        return X_train, y_train
    
    def plot_training_curves(self, model, output_dir):
        """绘制训练曲线"""
        try:
            if hasattr(model, 'history') and model.history is not None:
                history = model.history
                
                # 创建图形
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                fig.suptitle('TabNet 训练曲线分析', fontsize=16)
                
                # 损失曲线
                if 'loss' in history:
                    axes[0, 0].plot(history['loss'], label='训练损失', color='blue')
                    if 'val_loss' in history:
                        axes[0, 0].plot(history['val_loss'], label='验证损失', color='red')
                    axes[0, 0].set_title('损失曲线')
                    axes[0, 0].set_xlabel('Epoch')
                    axes[0, 0].set_ylabel('Loss')
                    axes[0, 0].legend()
                    axes[0, 0].grid(True)
                
                # 准确率曲线
                if 'accuracy' in history:
                    axes[0, 1].plot(history['accuracy'], label='训练准确率', color='blue')
                    if 'val_accuracy' in history:
                        axes[0, 1].plot(history['val_accuracy'], label='验证准确率', color='red')
                    axes[0, 1].set_title('准确率曲线')
                    axes[0, 1].set_xlabel('Epoch')
                    axes[0, 1].set_ylabel('Accuracy')
                    axes[0, 1].legend()
                    axes[0, 1].grid(True)
                
                # 学习率曲线
                if 'lr' in history:
                    axes[1, 0].plot(history['lr'], label='学习率', color='green')
                    axes[1, 0].set_title('学习率变化')
                    axes[1, 0].set_xlabel('Epoch')
                    axes[1, 0].set_ylabel('Learning Rate')
                    axes[1, 0].legend()
                    axes[1, 0].grid(True)
                
                # 过拟合检测图
                if 'loss' in history and 'val_loss' in history:
                    train_loss = np.array(history['loss'])
                    val_loss = np.array(history['val_loss'])
                    overfitting_gap = val_loss - train_loss
                    
                    axes[1, 1].plot(overfitting_gap, label='验证-训练损失差', color='purple')
                    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
                    axes[1, 1].set_title('过拟合检测')
                    axes[1, 1].set_xlabel('Epoch')
                    axes[1, 1].set_ylabel('Loss Gap')
                    axes[1, 1].legend()
                    axes[1, 1].grid(True)
                
                plt.tight_layout()
                
                # 保存图像
                curve_path = os.path.join(output_dir, 'training_curves.png')
                plt.savefig(curve_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"训练曲线已保存到: {curve_path}")
                
        except Exception as e:
            print(f"绘制训练曲线时出错: {e}")
    
    def get_param_grid(self, search_type='bayesian', n_samples=None, task_type='multiclass'):
        """获取针对防过拟合优化的参数网格"""
        # 防过拟合优化的参数网格
        if search_type == 'bayesian' and BAYESIAN_AVAILABLE:
            param_grid = {
                'n_d': Integer(24, 64),  # 减小网络容量防止过拟合
                'n_a': Integer(24, 64),
                'n_steps': Integer(3, 6),  # 减少步数
                'gamma': Real(1.5, 3.0),  # 增加gamma值增强稀疏性
                'lambda_sparse': Real(1e-4, 1e-2, prior='log-uniform'),  # 增强稀疏正则化
                'lr': Real(5e-4, 1e-2, prior='log-uniform'),  # 降低学习率
                'batch_size': Categorical([512, 1024, 2048]),  # 减小批次大小
                'max_epochs': Integer(30, 100),  # 减少最大训练轮数
                'weight_decay': Real(1e-5, 1e-3, prior='log-uniform')  # 添加权重衰减
            }
            
            # 根据任务类型调整参数
            if task_type == 'multiclass':
                # 多分类任务的防过拟合策略
                param_grid['lambda_sparse'] = Real(5e-4, 2e-2, prior='log-uniform')
                param_grid['weight_decay'] = Real(5e-5, 5e-3, prior='log-uniform')
                param_grid['gamma'] = Real(1.8, 3.5)
            
            return param_grid
        else:
            # 防过拟合的网格搜索参数
            param_grid = {
                'n_d': [24, 32, 48, 64],  # 减小网络容量
                'n_a': [24, 32, 48, 64],
                'n_steps': [3, 4, 5],  # 减少步数
                'gamma': [1.5, 2.0, 2.5, 3.0],  # 增强稀疏性
                'lambda_sparse': [1e-4, 5e-4, 1e-3, 5e-3],  # 增强稀疏正则化
                'lr': [5e-4, 1e-3, 2e-3, 5e-3],  # 降低学习率
                'batch_size': [512, 1024, 2048],  # 减小批次大小
                'max_epochs': [50, 75, 100],  # 减少训练轮数
                'weight_decay': [1e-5, 1e-4, 5e-4, 1e-3]  # 添加权重衰减
            }
            
            return param_grid
    
    def create_model(self, params, n_classes=2):
        """创建TabNet模型，增强防过拟合策略"""
        def extract_scalar(value, default):
            if hasattr(value, '__iter__') and not isinstance(value, str):
                return float(value[0]) if len(value) > 0 else default
            elif hasattr(value, 'item'):
                return float(value.item())
            else:
                return float(value) if value is not None else default
        
        # 增强的权重衰减策略
        weight_decay = extract_scalar(params.get('weight_decay'), 1e-4)
        if n_classes > 2:  # 多分类任务需要更强的正则化
            weight_decay = max(weight_decay, 5e-4)
        
        # 动态调整稀疏正则化
        lambda_sparse = extract_scalar(params.get('lambda_sparse'), 1e-4)
        if n_classes > 2:
            lambda_sparse = max(lambda_sparse, 5e-4)  # 多分类增强稀疏性
        
        # 学习率调度器参数
        scheduler_params = {
            'step_size': 20,  # 每20个epoch降低学习率
            'gamma': 0.8      # 学习率衰减因子
        }
        
        model_params = {
            'n_d': int(extract_scalar(params.get('n_d'), 48)),
            'n_a': int(extract_scalar(params.get('n_a'), 48)),
            'n_steps': int(extract_scalar(params.get('n_steps'), 4)),
            'gamma': extract_scalar(params.get('gamma'), 1.4),
            'lambda_sparse': lambda_sparse,
            'optimizer_fn': torch.optim.Adam,
            'optimizer_params': {
                'lr': extract_scalar(params.get('lr'), 0.015),
                'weight_decay': weight_decay
            },
            'mask_type': 'entmax',
            'scheduler_params': scheduler_params,  # 添加学习率调度
            'scheduler_fn': torch.optim.lr_scheduler.StepLR,  # 使用StepLR调度器
            'seed': self.random_state,
            'verbose': 1,
            'device_name': self.device_name
        }
        
        return TabNetClassifier(**model_params)
    
    def get_comprehensive_metrics(self, y_true, y_pred, y_proba=None):
        """综合评估指标计算"""
        from sklearn.metrics import balanced_accuracy_score, precision_score
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'f1_micro': f1_score(y_true, y_pred, average='micro'),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
            'recall_macro': recall_score(y_true, y_pred, average='macro'),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred)
        }
        
        # 多分类AUC
        if y_proba is not None:
            try:
                metrics['roc_auc_ovr'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
                metrics['roc_auc_ovo'] = roc_auc_score(y_true, y_proba, multi_class='ovo', average='weighted')
            except ValueError:
                pass
        
        return metrics
    
    def cross_validate(self, X, y, params, cv_folds=5, scoring='roc_auc'):
        """增强的交叉验证，支持多指标评估"""
        n_classes = len(np.unique(y))
        is_binary = n_classes == 2
        
        # 检查类别分布
        class_counts = np.bincount(y)
        class_weights = len(y) / (n_classes * class_counts)
        imbalance_ratio = max(class_counts) / min(class_counts)
        
        print(f"类别分布: {dict(enumerate(class_counts))}")
        print(f"不平衡比例: {imbalance_ratio:.2f}")
        
        # 根据不平衡程度选择合适的评估策略
        primary_metric = scoring
        if scoring == 'roc_auc' and not is_binary:
            primary_metric = 'f1_weighted'
        
        if imbalance_ratio > 3:
            print("检测到类别不平衡，将使用平衡准确率作为主要指标")
            primary_metric = 'balanced_accuracy'
            secondary_metric = 'f1_macro'
        else:
            secondary_metric = 'f1_weighted'
        
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        all_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            try:
                X_train_fold = X[train_idx].astype(np.float32)
                X_val_fold = X[val_idx].astype(np.float32)
                y_train_fold = y[train_idx].astype(np.int64)
                y_val_fold = y[val_idx].astype(np.int64)
                
                model = self.create_model(params, n_classes=n_classes)
                
                max_epochs_val = int(params.get('max_epochs', 100))
                batch_size_val = int(params.get('batch_size', 1024))
                
                # 防过拟合的训练策略
                patience_val = min(10, max_epochs_val // 5)  # 更严格的早停
                virtual_batch_size = min(batch_size_val // 2, 512)
                
                model.fit(
                    X_train_fold, y_train_fold,
                    eval_set=[(X_train_fold, y_train_fold), (X_val_fold, y_val_fold)],
                    eval_name=['train', 'valid'],
                    eval_metric=['logloss', 'accuracy'] if not is_binary else ['auc', 'logloss'],
                    max_epochs=max_epochs_val,
                    patience=patience_val,  # 更严格的早停
                    batch_size=batch_size_val,
                    virtual_batch_size=virtual_batch_size,
                    num_workers=0,
                    weights=1,
                    drop_last=False
                )
                
                # 预测
                y_pred = model.predict(X_val_fold)
                y_proba = model.predict_proba(X_val_fold)
                
                # 计算综合指标
                fold_metrics = self.get_comprehensive_metrics(y_val_fold, y_pred, y_proba)
                all_metrics.append(fold_metrics)
                
            except Exception as e:
                print(f"第 {fold + 1} 折训练失败: {str(e)}")
                # 添加默认的失败指标
                default_metrics = {
                    'accuracy': 0.0, 'f1_weighted': 0.0, 'f1_macro': 0.0, 
                    'balanced_accuracy': 0.0, 'precision_weighted': 0.0, 
                    'recall_weighted': 0.0
                }
                all_metrics.append(default_metrics)
        
        # 计算平均指标
        mean_metrics = {}
        std_metrics = {}
        for metric in all_metrics[0].keys():
            scores = [fold_metrics[metric] for fold_metrics in all_metrics]
            mean_metrics[metric] = np.mean(scores)
            std_metrics[metric] = np.std(scores)
        
        # 计算复合得分
        if primary_metric in mean_metrics and secondary_metric in mean_metrics:
            composite_score = (mean_metrics[primary_metric] + mean_metrics[secondary_metric]) / 2
        else:
            composite_score = mean_metrics.get(primary_metric, 0.0)
        
        return {
            'mean_score': mean_metrics.get(primary_metric, 0.0),
            'composite_score': composite_score,
            'mean_metrics': mean_metrics,
            'std_metrics': std_metrics,
            'scores': [fold_metrics.get(primary_metric, 0.0) for fold_metrics in all_metrics],
            'params': params,
            'imbalance_ratio': imbalance_ratio
        }
    
    def grid_search(self, X, y, cv_folds=5, scoring='roc_auc', search_type='bayesian', n_iter=20, task_type='multiclass'):
        """参数搜索（贝叶斯优化）"""
        param_grid = self.get_param_grid(search_type, task_type=task_type)
        
        if search_type == 'bayesian' and BAYESIAN_AVAILABLE:
            # 贝叶斯优化
            print(f"\n开始贝叶斯优化，迭代次数: {n_iter}")
            
            def objective(params):
                try:
                    cv_result = self.cross_validate(X, y, params, cv_folds, scoring)
                    return -cv_result['mean_score']  # 负值因为BayesSearchCV最小化目标
                except Exception as e:
                    print(f"参数评估失败: {e}")
                    return 0.0
            
            try:
                from skopt import gp_minimize
                from skopt.utils import use_named_args
                
                # 转换参数空间 - 为每个维度添加名称
                param_names = list(param_grid.keys())
                dimensions = []
                for name, space in param_grid.items():
                    if hasattr(space, 'name'):
                        space.name = name
                    else:
                        # 为没有name属性的空间对象手动设置
                        try:
                            space.name = name
                        except AttributeError:
                            # 如果无法设置name，重新创建空间对象
                            if hasattr(space, 'low') and hasattr(space, 'high'):
                                if hasattr(space, 'prior'):
                                    if space.prior == 'log-uniform':
                                        from skopt.space import Real
                                        space = Real(space.low, space.high, prior='log-uniform', name=name)
                                    else:
                                        from skopt.space import Real, Integer
                                        if isinstance(space.low, int):
                                            space = Integer(space.low, space.high, name=name)
                                        else:
                                            space = Real(space.low, space.high, name=name)
                                else:
                                    from skopt.space import Real, Integer
                                    if isinstance(space.low, int):
                                        space = Integer(space.low, space.high, name=name)
                                    else:
                                        space = Real(space.low, space.high, name=name)
                            elif hasattr(space, 'categories'):
                                from skopt.space import Categorical
                                space = Categorical(space.categories, name=name)
                    dimensions.append(space)
                
                @use_named_args(dimensions)
                def objective_func(**params):
                    return objective(params)
                
                # 执行贝叶斯优化
                result = gp_minimize(
                    func=objective_func,
                    dimensions=dimensions,
                    n_calls=n_iter,
                    random_state=self.random_state,
                    n_initial_points=5
                )
                
                # 提取最佳参数
                best_params = dict(zip(param_names, result.x))
                best_score = -result.fun
                
                print(f"贝叶斯优化完成，最佳得分: {best_score:.4f}")
                
            except Exception as e:
                print(f"贝叶斯优化失败，回退到网格搜索: {e}")
                # 回退到简化网格搜索
                param_combinations = list(product(*param_grid.values()))
                param_names = list(param_grid.keys())
                param_combinations = [dict(zip(param_names, param_values)) 
                                    for param_values in param_combinations[:n_iter]]
                
                best_score = -1
                best_params = None
                
                for i, params in enumerate(param_combinations):
                    try:
                        cv_result = self.cross_validate(X, y, params, cv_folds, scoring)
                        if cv_result['mean_score'] > best_score:
                            best_score = cv_result['mean_score']
                            best_params = params.copy()
                    except Exception:
                        continue
        else:
            # 如果贝叶斯优化不可用，使用网格搜索
            param_combinations = list(product(*param_grid.values()))
            param_names = list(param_grid.keys())
            param_combinations = [dict(zip(param_names, param_values)) 
                                for param_values in param_combinations]
            
            best_score = -1
            best_params = None
            
            for i, params in enumerate(param_combinations):
                print(f"\n测试参数组合 {i+1}/{len(param_combinations)}: {params}")
                
                try:
                    cv_result = self.cross_validate(X, y, params, cv_folds, scoring)
                    
                    if cv_result['mean_score'] > best_score:
                        best_score = cv_result['mean_score']
                        best_params = params.copy()
                        print(f"发现更好的参数组合！得分: {best_score:.4f}")
                        
                except Exception as e:
                    print(f"参数组合 {i+1} 失败: {str(e)}")
                    continue
        
        self.best_params = best_params
        self.best_score = best_score
        
        return {
            'best_params': best_params,
            'best_score': best_score
        }
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, params=None):
        """训练TabNet模型，增强防过拟合策略"""
        print(f"\n=== TabNet模型训练开始（防过拟合优化版）===")
        print(f"训练集形状: {X_train.shape}")
        
        # 保存特征名称
        if hasattr(X_train, 'columns'):
            self.feature_names = X_train.columns.tolist()
        
        # 数据预处理
        if hasattr(X_train, 'values'):
            X_train = X_train.values
        if hasattr(y_train, 'values'):
            y_train = y_train.values
        
        X_train = X_train.astype(np.float32)
        y_train = y_train.astype(np.int64)
        
        if X_val is not None:
            if hasattr(X_val, 'values'):
                X_val = X_val.values
            if hasattr(y_val, 'values'):
                y_val = y_val.values
            X_val = X_val.astype(np.float32)
            y_val = y_val.astype(np.int64)
        
        # 检查分类数量
        n_classes = len(np.unique(y_train))
        is_binary = n_classes == 2
        
        # 防过拟合的默认参数
        if params is None:
            params = {
                'n_d': 32, 'n_a': 32, 'n_steps': 4,
                'gamma': 2.0, 'lambda_sparse': 5e-4,
                'lr': 0.005, 'batch_size': 1024, 'max_epochs': 80,
                'weight_decay': 1e-4
            }
        
        # 创建模型
        self.model = self.create_model(params, n_classes)
        
        # 训练模型
        if X_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
            eval_name = ['train', 'valid']
        else:
            eval_set = [(X_train, y_train)]
            eval_name = ['train']
        
        max_epochs_val = int(params.get('max_epochs', 80))
        batch_size_val = int(params.get('batch_size', 1024))
        
        # 增强的早停策略
        patience_val = min(15, max_epochs_val // 4)  # 动态调整patience
        print(f"使用早停patience: {patience_val}")
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            eval_name=eval_name,
            eval_metric=['logloss', 'accuracy'] if not is_binary else ['auc', 'logloss'],
            max_epochs=max_epochs_val,
            patience=patience_val,  # 更严格的早停
            batch_size=batch_size_val,
            virtual_batch_size=min(batch_size_val // 2, 512),  # 减小虚拟批次大小
            num_workers=0,
            weights=1,
            drop_last=False
        )
        
        self.is_fitted = True
        print("TabNet模型训练完成（防过拟合优化）")
        return self
    
    def predict(self, X):
        """预测"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        if hasattr(X, 'values'):
            X = X.values
        X = X.astype(np.float32)
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """预测概率"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        if hasattr(X, 'values'):
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
                          optimize=True, cv_folds=3, search_type='bayesian',
                          n_iter=20, use_validation_in_training=True, base_path=None):
    """
    优化版TabNet模型训练函数
    """
    
    if not TABNET_AVAILABLE:
        raise ImportError("pytorch_tabnet is required for TabNet models")
    
    # 设置基础路径
    if base_path is None:
        import inspect
        frame = inspect.currentframe()
        try:
            caller_frame = frame.f_back
            while caller_frame:
                caller_file = caller_frame.f_code.co_filename
                if not caller_file.endswith(('TabNet_optimized.py',)):
                    caller_base_path = os.path.dirname(caller_file)
                    break
                caller_frame = caller_frame.f_back
            else:
                caller_base_path = os.getcwd()
        finally:
            del frame
        
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = os.path.join(caller_base_path, f"Results/TabNet_{target}_{timestamp}")
        os.makedirs(results_dir, exist_ok=True)
        base_path = results_dir
    
    print(f"\n=== TabNet模型训练: {target} ===")
    print(f"训练集长度: {len(X_train)}, 标签1数量: {np.sum(y_train == 1)}")
    print(f"测试集长度: {len(X_test)}, 标签1数量: {np.sum(y_test == 1)}")
    if X_val is not None:
        print(f"验证集长度: {len(X_val)}, 标签1数量: {np.sum(y_val == 1)}")
    
    # 创建TabNet分类器
    tabnet_classifier = TabNetSimplified()
    
    # 创建输出目录
    output_dir = tabnet_classifier._create_output_dirs(base_path)
    
    # 准备数据
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
            X_combined = np.vstack([X_train_values, X_val_values]).astype(np.float32)
            y_combined = np.hstack([y_train_values, y_val_values]).astype(np.int64)
        else:
            X_combined = X_train_values
            y_combined = y_train_values
        
        # 根据分类数量选择评估指标和任务类型
        n_classes = len(np.unique(y_combined))
        task_type = 'binary' if n_classes == 2 else 'multiclass'
        
        if n_classes == 2:
            scoring = 'roc_auc'
        else:
            scoring = 'f1_weighted'
        
        print(f"任务类型: {task_type}, 类别数量: {n_classes}")
        
        optimization_results = tabnet_classifier.grid_search(
            X_combined, y_combined, cv_folds=cv_folds, scoring=scoring,
            search_type=search_type, n_iter=n_iter, task_type=task_type
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
            'n_d': 48, 'n_a': 48, 'n_steps': 4, 'gamma': 1.4,
            'lambda_sparse': 1e-4, 'lr': 0.015, 'batch_size': 1536, 'max_epochs': 120
        }
        print(f"使用默认参数: {best_params}")
    
    # 应用数据增强
    print("\n应用数据增强技术...")
    X_train_augmented, y_train_augmented = tabnet_classifier._augment_training_data(
        X_train_values, y_train_values, augment_ratio=0.2
    )
    
    # 使用最优参数训练最终模型
    tabnet_classifier.fit(X_train_augmented, y_train_augmented, X_val_values, y_val_values, best_params)
    
    # 保存模型
    model_path = output_dir / f'TabNet_model_target_{target}.joblib'
    joblib.dump(tabnet_classifier, model_path)
    print(f"模型已保存到: {model_path}")
    
    # 绘制训练曲线
    print("\n绘制训练曲线...")
    tabnet_classifier.plot_training_curves(tabnet_classifier.model, output_dir)
    
    # 评估模型
    try:
        y_pred_train = tabnet_classifier.predict(X_train_values)
        y_pred_val = tabnet_classifier.predict(X_val_values)
        y_pred_test = tabnet_classifier.predict(X_test_values)
        
        # 获取综合评估指标
        y_proba_train = tabnet_classifier.predict_proba(X_train_values)
        y_proba_val = tabnet_classifier.predict_proba(X_val_values)
        y_proba_test = tabnet_classifier.predict_proba(X_test_values)
        
        train_metrics = tabnet_classifier.get_comprehensive_metrics(y_train_values, y_pred_train, y_proba_train)
        val_metrics = tabnet_classifier.get_comprehensive_metrics(y_val_values, y_pred_val, y_proba_val)
        test_metrics = tabnet_classifier.get_comprehensive_metrics(y_test_values, y_pred_test, y_proba_test)
        
        # 过拟合检测
        print("\n=== 过拟合检测 ===")
        acc_gap = train_metrics['accuracy'] - val_metrics['accuracy']
        f1_gap = train_metrics['f1_weighted'] - val_metrics['f1_weighted']
        balanced_acc_gap = train_metrics['balanced_accuracy'] - val_metrics['balanced_accuracy']
        
        print(f"训练集-验证集性能差异:")
        print(f"  准确率差异: {acc_gap:.4f} ({'过拟合' if acc_gap > 0.15 else '轻微过拟合' if acc_gap > 0.08 else '正常'})")
        print(f"  加权F1差异: {f1_gap:.4f} ({'过拟合' if f1_gap > 0.15 else '轻微过拟合' if f1_gap > 0.08 else '正常'})")
        print(f"  平衡准确率差异: {balanced_acc_gap:.4f} ({'过拟合' if balanced_acc_gap > 0.15 else '轻微过拟合' if balanced_acc_gap > 0.08 else '正常'})")
        
        # 综合过拟合评估
        overfitting_score = (acc_gap + f1_gap + balanced_acc_gap) / 3
        if overfitting_score > 0.15:
            print(f"\n⚠️  严重过拟合警告 (综合得分: {overfitting_score:.4f})")
            print("建议: 增加正则化、减少模型复杂度、增加数据量")
        elif overfitting_score > 0.08:
            print(f"\n⚠️  轻微过拟合 (综合得分: {overfitting_score:.4f})")
            print("建议: 调整超参数、增强早停策略")
        else:
            print(f"\n✅ 模型泛化良好 (综合得分: {overfitting_score:.4f})")
        
        # 多分类AUC过拟合检测
        if 'roc_auc_ovr' in train_metrics and train_metrics['roc_auc_ovr'] is not None:
            auc_ovr_gap = train_metrics['roc_auc_ovr'] - val_metrics['roc_auc_ovr']
            auc_ovo_gap = train_metrics['roc_auc_ovo'] - val_metrics['roc_auc_ovo']
            print(f"AUC差异 - OVR: {auc_ovr_gap:.4f}, OVO: {auc_ovo_gap:.4f}")
        
        eval_scores = {
            'train_accuracy': train_metrics['accuracy'],
            'val_accuracy': val_metrics['accuracy'],
            'test_accuracy': test_metrics['accuracy'],
            'train_f1_weighted': train_metrics['f1_weighted'],
            'val_f1_weighted': val_metrics['f1_weighted'],
            'test_f1_weighted': test_metrics['f1_weighted'],
            'train_f1_macro': train_metrics['f1_macro'],
            'val_f1_macro': val_metrics['f1_macro'],
            'test_f1_macro': test_metrics['f1_macro'],
            'train_balanced_accuracy': train_metrics['balanced_accuracy'],
            'val_balanced_accuracy': val_metrics['balanced_accuracy'],
            'test_balanced_accuracy': test_metrics['balanced_accuracy'],
            'train_recall_weighted': train_metrics['recall_weighted'],
            'val_recall_weighted': val_metrics['recall_weighted'],
            'test_recall_weighted': test_metrics['recall_weighted'],
            'best_params': best_params
        }
        
        # 添加多分类AUC指标（如果可用）
        if 'roc_auc_ovr' in test_metrics:
            eval_scores['testing_roc_auc_ovr'] = test_metrics['roc_auc_ovr']
        if 'roc_auc_ovo' in test_metrics:
            eval_scores['testing_roc_auc_ovo'] = test_metrics['roc_auc_ovo']
        
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
        
        # 添加过拟合评估到eval_scores
        eval_scores['overfitting_score'] = overfitting_score
        eval_scores['accuracy_gap'] = acc_gap
        eval_scores['f1_weighted_gap'] = f1_gap
        eval_scores['balanced_accuracy_gap'] = balanced_acc_gap
        
        # 混淆矩阵
        print(f"\n测试集混淆矩阵:")
        cm = confusion_matrix(y_test_values, y_pred_test)
        print(cm)
        
        # 分类报告
        print(f"\n测试集分类报告:")
        print(classification_report(y_test_values, y_pred_test))
        
        # 每类别性能分析（多分类特有）
        n_classes = len(np.unique(y_test_values))
        if n_classes > 2:
            print(f"\n=== 每类别性能分析 ===")
            unique_classes = np.unique(y_test_values)
            for i, class_label in enumerate(unique_classes):
                class_mask = (y_test_values == class_label)
                class_pred_mask = (y_pred_test == class_label)
                
                # 计算每个类别的指标
                if np.sum(class_mask) > 0:  # 确保该类别在测试集中存在
                    class_precision = precision_score(y_test_values == class_label, y_pred_test == class_label)
                    class_recall = recall_score(y_test_values == class_label, y_pred_test == class_label)
                    class_f1 = f1_score(y_test_values == class_label, y_pred_test == class_label)
                    class_support = np.sum(class_mask)
                    
                    print(f"类别 {class_label}:")
                    print(f"  精确率: {class_precision:.4f}")
                    print(f"  召回率: {class_recall:.4f}")
                    print(f"  F1分数: {class_f1:.4f}")
                    print(f"  支持度: {class_support}")
                    
                    # 如果有概率预测，计算该类别的AUC
                    if y_proba_test is not None and y_proba_test.shape[1] > i:
                        try:
                            class_auc = roc_auc_score(y_test_values == class_label, y_proba_test[:, i])
                            print(f"  AUC: {class_auc:.4f}")
                        except ValueError:
                            pass
                    print()
        
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
    
    # 绘制可视化图表
    if len(np.unique(y_train_values)) == 2:
        try:
            if YELLOWBRICK_AVAILABLE:
                print(f"正在创建TabNet模型的性能可视化图表 - {target}")
                
                # ROC曲线
                print("创建ROC曲线...")
                viz = ROCAUC(tabnet_classifier.model, title=f"TabNet ROC Curve - {target}")
                viz.fit(X_train_values, y_train_values)
                viz.score(X_test_values, y_test_values)
                roc_path = output_dir / 'Output' / f'TabNet_ROC_curve_target_{target}.pdf'
                viz.show(outpath=roc_path)
                print(f"ROC曲线已保存到: {roc_path}")
                
                # 混淆矩阵
                print("创建混淆矩阵...")
                unique_labels = np.unique(np.concatenate([y_train_values, y_test_values]))
                viz = ConfusionMatrix(tabnet_classifier.model, classes=unique_labels, title=f"TabNet Confusion Matrix - {target}")
                viz.fit(X_train_values, y_train_values)
                viz.score(X_test_values, y_test_values)
                cm_path = output_dir / 'Output' / f'TabNet_confusion_matrix_target_{target}.pdf'
                viz.show(outpath=cm_path)
                print(f"混淆矩阵已保存到: {cm_path}")
                
            else:
                # 回退到matplotlib
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
                
                roc_path = output_dir / 'Output' / f'TabNet_ROC_curve_target_{target}.pdf'
                plt.savefig(roc_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"ROC曲线已保存到: {roc_path}")
            
        except Exception as e:
            print(f"可视化图表绘制失败: {str(e)}")
    
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
