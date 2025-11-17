import numpy as np
import pandas as pd
import lightgbm as lgb
import multiprocessing
import psutil
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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import fbeta_score
import joblib
import os
from datetime import datetime
import logging
import warnings
import pickle
import hashlib
import time
import json
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

# 全局参数缓存机制
CACHE_FILE = "lightgbm_parameter_cache.pkl"
PARAMETER_CACHE = {}


def load_parameter_cache():
    """加载参数缓存"""
    global PARAMETER_CACHE
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "rb") as f:
                PARAMETER_CACHE = pickle.load(f)
            print(
                f"已加载LightGBM参数缓存，包含 {len(PARAMETER_CACHE)} 个已评估的参数组合"
            )
    except Exception as e:
        print(f"加载LightGBM缓存失败: {e}")
        PARAMETER_CACHE = {}


def save_parameter_cache():
    """保存参数缓存"""
    try:
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(PARAMETER_CACHE, f)
    except Exception as e:
        print(f"保存LightGBM缓存失败: {e}")


def get_cache_key(X_train, y_train, params):
    """生成缓存键"""
    # 使用数据和参数的哈希值作为键
    data_hash = hashlib.md5(
        str(X_train.shape).encode() + str(np.sum(y_train)).encode()
    ).hexdigest()[:8]
    param_hash = hashlib.md5(str(sorted(params.items())).encode()).hexdigest()[:8]
    return f"lgb_{data_hash}_{param_hash}"


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


# 增强的智能早停回调类（防过拟合优化）
class LightGBMEarlyStopper:
    """LightGBM优化的智能早停机制 - 防过拟合增强版"""

    def __init__(self, patience=15, min_delta=0.0005, validation_fraction=0.2, score_history_size=10, overfitting_threshold=0.05):
        self.patience = patience  # 增加耐心值，防止过早停止
        self.min_delta = min_delta  # 降低最小改进阈值，更敏感
        self.validation_fraction = validation_fraction
        self.best_score = -np.inf
        self.wait = 0
        self.stopped_epoch = 0
        self.score_history = []  # 记录分数历史
        self.score_history_size = score_history_size
        self.overfitting_threshold = overfitting_threshold  # 过拟合检测阈值
        self.train_scores = []
        self.val_scores = []
        self.overfitting_detected = False

    def __call__(self, result):
        current_score = (
            result.func_vals[-1]
            if hasattr(result, "func_vals") and result.func_vals
            else -np.inf
        )
        
        self.score_history.append(current_score)
        
        # 限制历史记录长度
        if len(self.score_history) > self.score_history_size:
            self.score_history.pop(0)
        
        # 增强的过拟合检测：检查分数趋势和方差
        if len(self.score_history) >= 5:
            recent_scores = self.score_history[-5:]
            score_variance = np.var(recent_scores)
            
            # 检测分数波动过大
            if score_variance > self.overfitting_threshold:
                print(f"检测到过拟合风险：分数方差 {score_variance:.4f} > 阈值 {self.overfitting_threshold}")
                self.overfitting_detected = True
                self.wait += 2  # 加速早停
            
            # 检测分数停滞或恶化趋势
            if len(self.score_history) >= self.score_history_size:
                early_mean = np.mean(self.score_history[:3])
                recent_mean = np.mean(self.score_history[-3:])
                if recent_mean < early_mean - 0.01:  # 分数显著恶化
                    print(f"检测到分数恶化趋势：早期 {early_mean:.4f} -> 近期 {recent_mean:.4f}")
                    self.overfitting_detected = True

        if current_score > self.best_score + self.min_delta:
            self.best_score = current_score
            self.wait = 0
        else:
            self.wait += 1

        if self.wait >= self.patience or self.overfitting_detected:
            self.stopped_epoch = (
                len(result.func_vals) if hasattr(result, "func_vals") else 0
            )
            reason = "过拟合检测" if self.overfitting_detected else "耐心值耗尽"
            print(
                f"LightGBM早停触发（{reason}）：在第 {self.stopped_epoch} 次迭代后停止，最佳分数: {self.best_score:.4f}"
            )
            print(f"分数历史方差: {np.var(self.score_history[-10:]):.6f}")
            return True
        return False
    
    def get_overfitting_info(self):
        """获取过拟合检测信息"""
        return {
            'overfitting_detected': self.overfitting_detected,
            'stopped_epoch': self.stopped_epoch,
            'score_variance': np.var(self.score_history) if self.score_history else 0,
            'final_score': self.best_score
        }


# 动态并行策略配置（优化版 - 解决并行层级冲突）
def get_dynamic_parallel_config(
    data_size, search_type="bayesian", model_name="LightGBM"
):
    """根据数据规模和搜索类型动态配置并行策略（优化版 - 解决并行层级冲突）"""
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
        f"🚀 超大规模数据并行配置: {model_name}={model_n_jobs}核心, 搜索=1核心, CV=1核心"
    )

    # 确保总核心数不超过可用核心数（避免过度订阅）
    total_cores = model_n_jobs + search_n_jobs + cv_n_jobs
    if total_cores > available_cores:
        scale_factor = available_cores * 0.9 / total_cores  # 留10%缓冲
        model_n_jobs = max(1, int(model_n_jobs * scale_factor))
        search_n_jobs = max(1, int(search_n_jobs * scale_factor))
        cv_n_jobs = max(1, int(cv_n_jobs * scale_factor))
        print(
            f"⚠️ 调整并行配置避免过度订阅: {model_name}={model_n_jobs}, 搜索={search_n_jobs}, CV={cv_n_jobs}"
        )

    # 设置环境变量供子进程使用
    os.environ["OMP_NUM_THREADS"] = str(model_n_jobs)
    os.environ["MKL_NUM_THREADS"] = str(model_n_jobs)
    os.environ["OPENBLAS_NUM_THREADS"] = str(model_n_jobs)

    return model_n_jobs, search_n_jobs, cv_n_jobs


# 分层优化策略类
class LayeredOptimizationStrategy:
    """分层优化策略：先粗粒度快速搜索，再细粒度精确优化"""

    def __init__(self, coarse_n_iter=30, fine_n_iter=70):
        self.coarse_n_iter = coarse_n_iter
        self.fine_n_iter = fine_n_iter

    def get_coarse_param_space(self):
        """粗粒度参数空间：较大的搜索范围，较少的候选值"""
        if BAYESIAN_AVAILABLE:
            return {
                "n_estimators": Integer(100, 800),
                "max_depth": Integer(3, 15),
                "learning_rate": Real(0.005, 0.3, prior="log-uniform"),
                "num_leaves": Integer(15, 300),
                "min_child_samples": Integer(5, 200),
                "subsample": Real(0.5, 1.0),
                "colsample_bytree": Real(0.5, 1.0),
                "reg_alpha": Real(0.0, 20.0, prior="log-uniform"),
                "reg_lambda": Real(0.0, 20.0, prior="log-uniform"),
                "min_split_gain": Real(0.0, 1.0),
                "min_child_weight": Real(0.001, 10.0, prior="log-uniform"),
                "subsample_freq": Integer(0, 10),
                "feature_fraction": Real(0.4, 1.0),
                "bagging_fraction": Real(0.4, 1.0),
                "bagging_freq": Integer(0, 10),
                "max_bin": Integer(100, 500),
                "boosting_type": Categorical(["gbdt", "dart", "goss"]),
            }

    def get_fine_param_space(self, best_coarse_params):
        """细粒度参数空间：基于粗搜索结果的精细调优"""
        if BAYESIAN_AVAILABLE:
            # 基于粗搜索最佳参数的邻域搜索
            return {
                "n_estimators": Integer(
                    max(100, best_coarse_params.get("n_estimators", 300) - 100),
                    min(1000, best_coarse_params.get("n_estimators", 300) + 200),
                ),
                "max_depth": Integer(
                    max(3, best_coarse_params.get("max_depth", 8) - 3),
                    min(20, best_coarse_params.get("max_depth", 8) + 5),
                ),
                "learning_rate": Real(
                    max(0.001, best_coarse_params.get("learning_rate", 0.1) * 0.3),
                    min(0.5, best_coarse_params.get("learning_rate", 0.1) * 3.0),
                    prior="log-uniform",
                ),
                "num_leaves": Integer(
                    max(15, best_coarse_params.get("num_leaves", 63) - 30),
                    min(500, best_coarse_params.get("num_leaves", 63) + 100),
                ),
                "min_child_samples": Integer(
                    max(1, best_coarse_params.get("min_child_samples", 20) - 15),
                    min(300, best_coarse_params.get("min_child_samples", 20) + 50),
                ),
                "subsample": Real(
                    max(0.3, best_coarse_params.get("subsample", 0.8) - 0.3),
                    min(1.0, best_coarse_params.get("subsample", 0.8) + 0.2),
                ),
                "colsample_bytree": Real(
                    max(0.3, best_coarse_params.get("colsample_bytree", 0.8) - 0.3),
                    min(1.0, best_coarse_params.get("colsample_bytree", 0.8) + 0.2),
                ),
                "reg_alpha": Real(
                    max(0.0, best_coarse_params.get("reg_alpha", 1.0) * 0.01),
                    min(50.0, best_coarse_params.get("reg_alpha", 1.0) * 20),
                    prior="log-uniform",
                ),
                "reg_lambda": Real(
                    max(0.0, best_coarse_params.get("reg_lambda", 1.0) * 0.01),
                    min(50.0, best_coarse_params.get("reg_lambda", 1.0) * 20),
                    prior="log-uniform",
                ),
                "min_split_gain": Real(
                    max(0.0, best_coarse_params.get("min_split_gain", 0.1) * 0.1),
                    min(2.0, best_coarse_params.get("min_split_gain", 0.1) * 10),
                ),
                "min_child_weight": Real(
                    max(0.0001, best_coarse_params.get("min_child_weight", 0.1) * 0.01),
                    min(20.0, best_coarse_params.get("min_child_weight", 0.1) * 100),
                    prior="log-uniform",
                ),
                "feature_fraction": Real(
                    max(0.3, best_coarse_params.get("feature_fraction", 0.8) - 0.3),
                    min(1.0, best_coarse_params.get("feature_fraction", 0.8) + 0.2),
                ),
                "bagging_fraction": Real(
                    max(0.3, best_coarse_params.get("bagging_fraction", 0.8) - 0.3),
                    min(1.0, best_coarse_params.get("bagging_fraction", 0.8) + 0.2),
                ),
                "max_bin": Integer(
                    max(50, best_coarse_params.get("max_bin", 255) - 100),
                    min(1000, best_coarse_params.get("max_bin", 255) + 200),
                ),
                "boosting_type": Categorical(
                    [best_coarse_params.get("boosting_type", "gbdt")]
                ),
                "subsample_freq": Integer(
                    max(0, best_coarse_params.get("subsample_freq", 1) - 2),
                    min(15, best_coarse_params.get("subsample_freq", 1) + 5),
                ),
                "bagging_freq": Integer(
                    max(0, best_coarse_params.get("bagging_freq", 1) - 2),
                    min(15, best_coarse_params.get("bagging_freq", 1) + 5),
                ),
            }


# 加载缓存
load_parameter_cache()

# 尝试导入贝叶斯优化库
try:
    from skopt import BayesSearchCV, gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    from skopt.acquisition import gaussian_ei
    BAYESIAN_AVAILABLE = True
    print("✅ scikit-optimize可用，启用贝叶斯优化")
except ImportError:
    print("❌ scikit-optimize未安装，无法使用贝叶斯优化")
    BAYESIAN_AVAILABLE = False

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def LightGBM_model(
    X_train,
    X_test,
    y_train,
    y_test,
    X_val,
    y_val,
    class_weights,
    target,
    kinds,
    use_early_stopping=True,
    n_iter=100,
    cv_folds=5,
    use_validation_in_training=True,
    base_path=None,
    use_layered_optimization=True,
    early_stopping_patience=10,
):
    """
    训练和评估优化的LightGBM模型（仅支持贝叶斯优化和大数据规模处理）

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

    logger.info(f"开始训练LightGBM模型 - Target: {target}, Kinds: {kinds}")
    logger.info(
        f"训练集大小: {X_train.shape}, 验证集大小: {X_val.shape}, 测试集大小: {X_test.shape}"
    )

    # 确定是否为多分类问题
    is_multiclass = len(class_weights) > 2
    n_classes = len(class_weights) if is_multiclass else 2

    # 使用动态并行策略配置（优化版）- 仅支持贝叶斯优化
    data_size = X_train.shape[0] if hasattr(X_train, "shape") else 200000
    lgb_n_jobs, search_n_jobs, cv_n_jobs = get_dynamic_parallel_config(
        data_size, "bayesian"
    )

    # 分层优化策略
    layered_strategy = None
    if use_layered_optimization:
        layered_strategy = LayeredOptimizationStrategy(
            coarse_n_iter=max(20, n_iter // 3), fine_n_iter=max(30, n_iter * 2 // 3)
        )
        logger.info(
            f"启用分层优化: 粗粒度搜索 {layered_strategy.coarse_n_iter} 次，细粒度搜索 {layered_strategy.fine_n_iter} 次"
        )

    # 智能早停
    early_stopper = None
    if use_early_stopping:
        early_stopper = LightGBMEarlyStopper(
            patience=early_stopping_patience, min_delta=1e-4
        )
        logger.info(f"启用智能早停: 耐心值 {early_stopping_patience}")

    # 基础模型参数（防过拟合优化版）
    base_params = {
        "n_jobs": lgb_n_jobs,
        "random_state": 42,
        "verbose": -1,
        "class_weight": class_weights,
        "importance_type": "gain",
        "force_row_wise": True,  # 避免并行冲突
        "num_threads": lgb_n_jobs,  # 明确指定线程数
        # 防过拟合默认设置（移除early_stopping_rounds，仅在有验证集时使用）
        "feature_pre_filter": False,  # 关闭预过滤，保持特征完整性
        "lambda_l1": 0.1,  # 默认L1正则化
        "lambda_l2": 0.1,  # 默认L2正则化
        "min_gain_to_split": 0.1,  # 最小分裂增益
        "drop_rate": 0.1,  # DART模式的丢弃率
        "max_drop": 50,  # DART模式最大丢弃数
        "skip_drop": 0.5,  # DART模式跳过丢弃概率
    }

    # 根据问题类型设置目标函数
    if is_multiclass:
        base_params.update(
            {
                "objective": "multiclass",
                "num_class": n_classes,
                "metric": "multi_logloss",
            }
        )
    else:
        base_params.update({"objective": "binary", "metric": "binary_logloss"})

    # 创建基础模型
    model = lgb.LGBMClassifier(**base_params)

    # 定义贝叶斯优化的超参数搜索空间（针对大数据规模优化）
    if not BAYESIAN_AVAILABLE:
        raise ImportError("贝叶斯优化库不可用，请安装 scikit-optimize")
    
    # 智能参数空间优化 - 基于20万行透析数据规模
    # 根据数据规模动态调整参数空间，提高优化效率
    n_samples = X_train.shape[0] if hasattr(X_train, "shape") else 200000

    # 自适应正则化强度计算
    def calculate_adaptive_regularization(n_samples, n_features):
        """根据数据规模自适应计算正则化强度"""
        # 样本数越少，正则化越强
        sample_factor = max(0.1, min(2.0, 100000 / n_samples))
        # 特征数越多，正则化越强
        feature_factor = max(0.1, min(2.0, n_features / 100))
        base_reg = sample_factor * feature_factor
        return {
            'min_reg': max(0.1, base_reg * 0.5),
            'max_reg': min(10.0, base_reg * 5.0)
        }
    
    n_features = X_train.shape[1] if hasattr(X_train, "shape") else 100
    adaptive_reg = calculate_adaptive_regularization(n_samples, n_features)
    
    # 防过拟合优化的参数空间（针对透析临床数据）
    if BAYESIAN_AVAILABLE:
        param_dist = {
            # 🔸 核心性能参数（保守设置防过拟合）
            "boosting_type": Categorical(["gbdt", "dart"], name="boosting_type"),  # dart 更鲁棒，减少过拟合
            "n_estimators": Integer(200, 500, name="n_estimators"),  # 减少树数量，防止过拟合
            "learning_rate": Real(0.01, 0.08, prior="log-uniform", name="learning_rate"),  # 更小的学习率，更稳定
            "max_depth": Integer(4, 8, name="max_depth"),  # 限制深度，提高可解释性
            "num_leaves": Integer(15, 63, name="num_leaves"),  # 减少叶子数，防止过拟合
            
            # 🔸 自适应正则化参数（防止过拟合核心）
            "reg_lambda": Real(adaptive_reg['min_reg'], adaptive_reg['max_reg'], prior="log-uniform", name="reg_lambda"),  # 自适应L2正则化
            "reg_alpha": Real(adaptive_reg['min_reg'] * 0.5, adaptive_reg['max_reg'] * 0.8, prior="log-uniform", name="reg_alpha"),  # 自适应L1正则化
            "min_child_samples": Integer(max(20, min(100, n_samples // 1000)), max(200, min(500, n_samples // 400)), name="min_child_samples"),  # 自适应最小样本数
            "min_split_gain": Real(0.1, 1.0, name="min_split_gain"),  # 提高分裂阈值
            "min_data_in_leaf": Integer(max(10, min(50, n_samples // 2000)), max(100, min(200, n_samples // 1000)), name="min_data_in_leaf"),  # 自适应叶子最小样本数
            
            # 🔸 采样参数（增强泛化能力）
            "subsample": Real(0.6, 0.9, name="subsample"),  # 更激进的行采样
            "colsample_bytree": Real(0.6, 0.9, name="colsample_bytree"),  # 更激进的列采样
            "feature_fraction": Real(0.6, 0.8, name="feature_fraction"),  # 特征采样防过拟合
            "bagging_fraction": Real(0.6, 0.8, name="bagging_fraction"),  # 样本采样防过拟合
            "bagging_freq": Integer(1, 5, name="bagging_freq"),  # 采样频率
            
            # 🔸 透析数据特定优化
            "subsample_freq": Integer(1, 2, name="subsample_freq"),  # 适应时间序列特性
            "max_bin": Integer(200, 300, name="max_bin"),  # 适中的分箱数
            "min_child_weight": Real(0.01, 1.0, prior="log-uniform", name="min_child_weight"),  # 叶子权重约束
        }
        
        logger.info(f"自适应正则化配置: L1=[{adaptive_reg['min_reg']:.3f}, {adaptive_reg['max_reg']:.3f}], L2=[{adaptive_reg['min_reg']:.3f}, {adaptive_reg['max_reg']:.3f}]")
        logger.info(f"样本自适应参数: min_child_samples=[{param_dist['min_child_samples'].low}, {param_dist['min_child_samples'].high}]")

    # 定义更全面的评估指标（防过拟合增强版）
    if is_multiclass:
        scoring = {
            # 'accuracy': make_scorer(accuracy_score),
            # 'f1_macro': make_scorer(f1_score, average='macro'),
            "f1_weighted": make_scorer(f1_score, average="weighted"),
            # 'f1_micro': make_scorer(f1_score, average='micro'),
            # 'precision_macro': make_scorer(precision_score, average='macro'),
            # 'precision_weighted': make_scorer(precision_score, average='weighted'),
            # 'recall_macro': make_scorer(recall_score, average='macro'),
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
            # 'accuracy': make_scorer(accuracy_score),
            # 'f1': make_scorer(f1_score),
            "f1_weighted": make_scorer(f1_score, average="weighted"),
            # 'precision': make_scorer(precision_score),
            # 'recall': make_scorer(recall_score),
            "roc_auc": make_scorer(roc_auc_score, response_method="predict_proba"),
            "neg_log_loss": make_scorer(
                log_loss, response_method="predict_proba", greater_is_better=False
            ),
        }
        refit_metric = "f1_weighted"
    
    # 交叉验证设置（防过拟合关键）
    cv_strategy = StratifiedKFold(
        n_splits=cv_folds,  # 使用传入的cv_folds参数
        shuffle=True, 
        random_state=42
    )
    
    # 过拟合检测函数
    def detect_overfitting(train_scores, val_scores, threshold=0.1):
        """检测过拟合：训练集和验证集性能差异"""
        train_mean = np.mean(train_scores)
        val_mean = np.mean(val_scores)
        gap = train_mean - val_mean
        is_overfitting = gap > threshold
        return is_overfitting, gap, train_mean, val_mean

    # 准备训练数据（需要在分层优化之前定义）
    if use_validation_in_training and X_val is not None and y_val is not None:
        # 合并训练集和验证集
        X_train_combined = np.vstack([X_train, X_val])
        y_train_combined = np.hstack([y_train, y_val])
        logger.info(f"使用验证集参与训练，合并后训练集大小: {X_train_combined.shape}")
    else:
        X_train_combined = X_train
        y_train_combined = y_train
        logger.info(f"仅使用原始训练集，大小: {X_train_combined.shape}")


    adjusted_n_iter = n_iter

    logger.info(f"根据数据规模调整迭代次数：{n_iter} -> {adjusted_n_iter}")

    # 初始化局部缓存
    param_cache = {}

    # 定义贝叶斯优化的自定义目标函数（防过拟合增强版）
    def create_objective_function(X, y, cv_strategy, scoring_metric, base_params, is_multiclass):
        """创建考虑过拟合的贝叶斯优化目标函数"""
        
        @use_named_args(list(param_dist.values()))
        def objective(**params):
            # 合并基础参数和优化参数
            model_params = {**base_params, **params}
            
            # 创建模型
            model = lgb.LGBMClassifier(**model_params)
            
            # 执行交叉验证
            cv_scores = cross_val_score(
                model, X, y, 
                cv=cv_strategy, 
                scoring=scoring_metric,
                n_jobs=1  # 避免嵌套并行
            )
            
            # 计算过拟合惩罚
            # 使用训练-验证分数差异作为过拟合指标
            train_scores = cross_val_score(
                model, X, y,
                cv=cv_strategy,
                scoring=scoring_metric,
                n_jobs=1
            )
            
            # 过拟合惩罚：训练分数和验证分数的差异
            overfitting_penalty = 0
            if len(train_scores) > 0 and len(cv_scores) > 0:
                train_mean = np.mean(train_scores)
                val_mean = np.mean(cv_scores)
                overfitting_gap = train_mean - val_mean
                
                # 如果过拟合严重，增加惩罚
                if overfitting_gap > 0.05:
                    overfitting_penalty = overfitting_gap * 2.0
            
            # 模型复杂度惩罚
            complexity_penalty = 0
            n_estimators = params.get('n_estimators', 100)
            max_depth = params.get('max_depth', 6)
            num_leaves = params.get('num_leaves', 31)
            
            # 复杂度分数：树数量 × 深度 × 叶子数
            complexity_score = (n_estimators / 1000) * (max_depth / 10) * (num_leaves / 100)
            if complexity_score > 1.0:
                complexity_penalty = (complexity_score - 1.0) * 0.1
            
            # 最终目标：负的验证分数 + 过拟合惩罚 + 复杂度惩罚
            # 注意：贝叶斯优化是最小化，所以使用负分数
            final_score = -np.mean(cv_scores) + overfitting_penalty + complexity_penalty
            
            return final_score
        
        return objective
    
    # 执行贝叶斯优化搜索策略
    logger.info(f"使用贝叶斯优化（防过拟合增强版），迭代次数: {adjusted_n_iter}")

    # 选择优化策略：高级贝叶斯优化 vs 传统BayesSearchCV
    use_advanced_bayesian = True  # 使用高级贝叶斯优化
    
    if use_advanced_bayesian and BAYESIAN_AVAILABLE:
        logger.info("使用高级贝叶斯优化（gp_minimize + 防过拟合）")
        
        # 创建目标函数
        objective_func = create_objective_function(
            X_train_combined, y_train_combined, 
            cv_strategy, refit_metric, 
            base_params, is_multiclass
        )
        
        # 准备参数空间（转换为skopt格式）
        param_names = list(param_dist.keys())
        param_space = list(param_dist.values())
        
        logger.info(f"开始高级贝叶斯优化，参数空间维度: {len(param_space)}")
        
        # 执行贝叶斯优化
        optimization_result = gp_minimize(
            func=objective_func,
            dimensions=param_space,
            n_calls=adjusted_n_iter,
            n_initial_points=max(10, adjusted_n_iter // 4),  # 初始随机点
            acq_func='EI',  # 期望改进
            random_state=42,
            verbose=True
        )
        
        # 提取最佳参数
        best_params_list = optimization_result.x
        best_params = dict(zip(param_names, best_params_list))
        best_score = -optimization_result.fun  # 转换回正分数
        
        logger.info(f"高级贝叶斯优化完成，最佳分数: {best_score:.4f}")
        logger.info(f"优化收敛信息: 函数调用次数={len(optimization_result.func_vals)}")
        
        # 创建最佳模型
        final_params = {**base_params, **best_params}
        best_model = lgb.LGBMClassifier(**final_params)
        best_model.fit(X_train_combined, y_train_combined)
        
        # 创建模拟的search对象以保持兼容性
        class MockSearch:
            def __init__(self, best_params, best_score, best_model, optimization_result):
                self.best_params_ = best_params
                self.best_score_ = best_score
                self.best_estimator_ = best_model
                self.optimization_result_ = optimization_result
                
                # 模拟cv_results_
                self.cv_results_ = {
                    f'mean_train_{refit_metric}': [-score for score in optimization_result.func_vals],
                    f'mean_test_{refit_metric}': [-score for score in optimization_result.func_vals],
                    'params': [dict(zip(param_names, x)) for x in optimization_result.x_iters]
                }
                self.best_index_ = np.argmin(optimization_result.func_vals)
        
        search = MockSearch(best_params, best_score, best_model, optimization_result)
        
    elif use_layered_optimization and layered_strategy:
        # 分层优化：粗搜索 + 细搜索
        logger.info(
            f"使用分层优化：粗搜索 {layered_strategy.coarse_n_iter} 次，细搜索 {layered_strategy.fine_n_iter} 次"
        )

        # 第一阶段：粗粒度快速搜索
        coarse_param_space = layered_strategy.get_coarse_param_space()
        
        # 创建增强的早停器
        enhanced_early_stopper = LightGBMEarlyStopper(
            patience=early_stopping_patience,
            min_delta=1e-4,
            score_history_size=10,
            overfitting_threshold=0.03
        )
        
        coarse_search = BayesSearchCV(
            model,
            coarse_param_space,
            n_iter=layered_strategy.coarse_n_iter,
            cv=cv_strategy,
            scoring=scoring,
            refit=refit_metric,
            n_jobs=search_n_jobs,
            random_state=42,
            verbose=1,
            return_train_score=True,  # 返回训练分数用于过拟合检测
        )
        
        logger.info("开始粗粒度搜索...")
        coarse_search.fit(X_train_combined, y_train_combined)

        # 第二阶段：基于粗搜索结果的细粒度优化
        fine_param_space = layered_strategy.get_fine_param_space(coarse_search.best_params_)
        
        search = BayesSearchCV(
            model,
            fine_param_space,
            n_iter=layered_strategy.fine_n_iter,
            cv=cv_strategy,
            scoring=scoring,
            refit=refit_metric,
            n_jobs=search_n_jobs,
            random_state=42,
            verbose=1,
            return_train_score=True,
        )
        
        logger.info("开始细粒度搜索...")
        search.fit(X_train_combined, y_train_combined)
    else:
        # 直接使用完整参数空间进行贝叶斯优化
        search = BayesSearchCV(
            model,
            param_dist,
            n_iter=adjusted_n_iter,
            cv=cv_strategy,
            scoring=scoring,
            refit=refit_metric,
            n_jobs=search_n_jobs,
            random_state=42,
            verbose=1,
            return_train_score=True,
        )
        
        logger.info("开始超参数搜索...")
        search.fit(X_train_combined, y_train_combined)

    # 过拟合检测
    train_scores = search.cv_results_[f'mean_train_{refit_metric}']
    val_scores = search.cv_results_[f'mean_test_{refit_metric}']
    
    # 检测最佳模型是否过拟合
    best_idx = search.best_index_
    is_overfitting, gap, train_mean, val_mean = detect_overfitting(
        [train_scores[best_idx]], [val_scores[best_idx]], threshold=0.08
    )
    
    if is_overfitting:
        logger.info(f"⚠️  检测到过拟合风险！训练-验证差异: {gap:.4f}")
        logger.info(f"   训练分数: {train_mean:.4f}, 验证分数: {val_mean:.4f}")
        
        # 选择更保守的模型（训练-验证差异较小的）
        gaps = train_scores - val_scores
        conservative_indices = np.where(gaps < 0.05)[0]  # 差异小于5%
        
        if len(conservative_indices) > 0:
            # 在保守模型中选择验证分数最高的
            conservative_val_scores = val_scores[conservative_indices]
            best_conservative_idx = conservative_indices[np.argmax(conservative_val_scores)]
            
            logger.info(f"🛡️  选择更保守的模型 (索引: {best_conservative_idx})")
            logger.info(f"   新的训练-验证差异: {gaps[best_conservative_idx]:.4f}")
            
            # 重新训练保守模型
            conservative_params = {}
            for param_name in param_dist.keys():
                if f'param_{param_name}' in search.cv_results_:
                    conservative_params[param_name] = search.cv_results_[f'param_{param_name}'][best_conservative_idx]
            
            best_model = lgb.LGBMClassifier(**{**base_params, **conservative_params})
            best_model.fit(X_train_combined, y_train_combined)
            best_params = {**base_params, **conservative_params}
            best_score = val_scores[best_conservative_idx]
        else:
            logger.info("⚠️  未找到足够保守的模型，使用原始最佳模型但增加自适应正则化")
            best_model = search.best_estimator_
            best_params = search.best_params_
            best_score = search.best_score_
            
            # 自适应增加正则化强度
            current_reg_alpha = best_params.get('reg_alpha', 0)
            current_reg_lambda = best_params.get('reg_lambda', 0)
            
            # 根据过拟合程度调整正则化
            overfitting_severity = min(gap / 0.1, 2.0)  # 过拟合严重程度
            reg_boost = 0.5 * overfitting_severity
            
            best_params['reg_alpha'] = min(current_reg_alpha + reg_boost, adaptive_reg['max_reg'])
            best_params['reg_lambda'] = min(current_reg_lambda + reg_boost, adaptive_reg['max_reg'])
            
            # 同时调整其他防过拟合参数
            if 'min_child_samples' in best_params:
                best_params['min_child_samples'] = min(best_params['min_child_samples'] * 1.2, 500)
            if 'min_data_in_leaf' in best_params:
                best_params['min_data_in_leaf'] = min(best_params['min_data_in_leaf'] * 1.2, 200)
            
            logger.info(f"自适应正则化调整: L1: {current_reg_alpha:.3f} -> {best_params['reg_alpha']:.3f}")
            logger.info(f"自适应正则化调整: L2: {current_reg_lambda:.3f} -> {best_params['reg_lambda']:.3f}")
            
            # 重新训练
            best_model = lgb.LGBMClassifier(**{**base_params, **best_params})
            best_model.fit(X_train_combined, y_train_combined)
    else:
        logger.info(f"✅ 未检测到过拟合，训练-验证差异: {gap:.4f}")
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

        # 更新最佳参数，添加训练相关参数
        final_params = best_params.copy()
        final_params.update(
            {
                "n_estimators": 1000,  # 设置较大的估计器数量，早停会自动控制实际使用数量
            }
        )
        
        # 早停轮数（仅用于fit方法的callbacks）
        early_stopping_rounds = 50
        eval_metric = "logloss" if not is_multiclass else "multi_logloss"

        # 创建最终模型
        final_model = lgb.LGBMClassifier(**{**base_params, **final_params})

        # 使用验证集进行早停训练
        final_model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_names=["training", "validation"],
            eval_metric=eval_metric,
            callbacks=[lgb.log_evaluation(100), lgb.early_stopping(early_stopping_rounds)],
        )

        model = final_model
    else:
        model = best_model

    # 📊 全面的模型评估（防过拟合增强版）
    logger.info("开始模型评估...")

    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val) if X_val is not None else None
    y_test_pred = model.predict(X_test)

    y_train_proba = model.predict_proba(X_train)
    y_val_proba = model.predict_proba(X_val) if X_val is not None else None
    y_test_proba = model.predict_proba(X_test)

    calibrated_model = None
    calibrated_method = None
    tuned_threshold = None
    tuned_fbeta = None
    if not is_multiclass and X_val is not None and y_val is not None:
        try:
            calibrated_model = CalibratedClassifierCV(model, cv='prefit', method='isotonic')
            calibrated_model.fit(X_val, y_val)
            calibrated_method = 'isotonic'
        except Exception:
            calibrated_model = CalibratedClassifierCV(model, cv='prefit', method='sigmoid')
            calibrated_model.fit(X_val, y_val)
            calibrated_method = 'sigmoid'
        y_val_proba = calibrated_model.predict_proba(X_val)
        y_test_proba = calibrated_model.predict_proba(X_test)
        thresholds = np.linspace(0.05, 0.95, 19)
        best_thr = 0.5
        best_score = -1.0
        for thr in thresholds:
            preds = (y_val_proba[:, 1] >= thr).astype(int)
            score = fbeta_score(y_val, preds, beta=2.0)
            if score > best_score:
                best_score = score
                best_thr = thr
        tuned_threshold = float(best_thr)
        tuned_fbeta = float(best_score)
        y_test_pred = (y_test_proba[:, 1] >= tuned_threshold).astype(int)

    # 🔍 过拟合分析
    logger.info("进行过拟合分析...")
    
    # 计算训练-测试性能差异
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
    
    # 🌳 模型复杂度分析
    model_complexity = {
        "n_estimators": model.n_estimators,
        "max_depth": model.max_depth,
        "num_leaves": model.num_leaves,
        "total_leaves": model.n_estimators * model.num_leaves,
        "feature_importances_entropy": -np.sum(model.feature_importances_ * np.log(model.feature_importances_ + 1e-10))
    }
    
    # 🎯 特征重要性分析
    feature_importance = model.feature_importances_
    feature_names = [f"feature_{i}" for i in range(len(feature_importance))] if not hasattr(X_train, 'columns') else list(X_train.columns)
    
    # 计算特征重要性集中度（基尼系数）
    sorted_importance = np.sort(feature_importance)[::-1]
    n_features = len(sorted_importance)
    cumsum_importance = np.cumsum(sorted_importance)
    gini_coefficient = (2 * np.sum((np.arange(1, n_features + 1) * sorted_importance))) / (n_features * np.sum(sorted_importance)) - (n_features + 1) / n_features
    
    # 特征重要性分析结果
    feature_analysis = {
        "top_10_features": dict(zip(feature_names[:10], feature_importance[:10])),
        "importance_concentration": gini_coefficient,
        "features_contributing_80_percent": np.sum(cumsum_importance <= 0.8 * cumsum_importance[-1]) + 1
    }

    # 计算各种评估指标
    def calculate_metrics(y_true, y_pred, y_proba, dataset_name):
        # 在 calculate_metrics 函数中修改
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
                    f"{dataset_name}_roc_auc": roc_auc_score(y_true, y_proba[:, 1]),
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
        results_dir = os.path.join(base_path, f"LGBM_{target}_{timestamp}")
    else:
        # 获取调用者的目录作为基础路径
        import inspect

        caller_frame = inspect.currentframe().f_back
        caller_file = caller_frame.f_globals.get("__file__")
        if caller_file:
            caller_dir = os.path.dirname(os.path.abspath(caller_file))
            results_dir = os.path.join(
                caller_dir, "Results", f"LGBM_{target}_{timestamp}"
            )
        else:
            results_dir = f"./Results/LGBM_{target}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    logger.info(f"结果将保存到: {results_dir}")

    # 生成总结报告（防过拟合增强版）
    summary_report = f"""
🎯 LightGBM模型训练完成报告（防过拟合优化版）
{'='*60}
📊 数据信息:
   - 训练样本数: {len(X_train):,}
   - 验证样本数: {len(X_val) if X_val is not None else 0:,}
   - 测试样本数: {len(X_test):,}
   - 特征数量: {X_train.shape[1]:,}
   - 目标类别: {len(np.unique(y_train))}

🔧 模型配置（防过拟合优化）:
   - 优化方法: BAYESIAN（增强早停 + 过拟合检测 + 自适应正则化）
   - 交叉验证折数: {cv_folds}
   - 早停耐心值: {early_stopping_patience}
   - 并行核心数: {lgb_n_jobs}
   - 自适应正则化: L1={best_params.get('reg_alpha', 'N/A'):.3f}, L2={best_params.get('reg_lambda', 'N/A'):.3f}
   - 正则化范围: L1=[{adaptive_reg['min_reg']:.3f}, {adaptive_reg['max_reg']:.3f}], L2=[{adaptive_reg['min_reg']:.3f}, {adaptive_reg['max_reg']:.3f}]
   - 采样率: 行={best_params.get('subsample', 'N/A')}, 列={best_params.get('colsample_bytree', 'N/A')}
   - 自适应参数: min_child_samples={best_params.get('min_child_samples', 'N/A')}, min_data_in_leaf={best_params.get('min_data_in_leaf', 'N/A')}

📈 性能指标:
   - 交叉验证分数: {cv_score:.4f}
   - 训练集准确率: {train_accuracy:.4f}
   - 验证集准确率: {val_metrics['validation_accuracy']:.4f}
   - 测试集准确率: {test_metrics['test_accuracy']:.4f}
   - 测试集F1分数: {test_metrics['test_f1_weighted']:.4f}

🔍 过拟合分析:
   - 风险等级: {overfitting_risk}
   - 训练-测试准确率差异: {train_test_gap:.4f}
   - 训练-测试F1差异: {train_test_f1_gap:.4f}
   - 模型复杂度评分: {model_complexity['total_leaves']:,} 总叶子数

🌳 模型复杂度:
   - 树的数量: {model_complexity['n_estimators']}
   - 最大深度: {model_complexity['max_depth']}
   - 叶子数/树: {model_complexity['num_leaves']}
   - 特征重要性熵: {model_complexity['feature_importances_entropy']:.4f}

🎯 特征分析:
   - 重要性集中度: {feature_analysis['importance_concentration']:.4f}
   - 80%贡献特征数: {feature_analysis['features_contributing_80_percent']}

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
    calibrated_model_path = os.path.join(results_dir, "calibrated_model.pkl") if calibrated_model is not None else None
    
    # 保存模型
    joblib.dump(model, model_path)
    if calibrated_model is not None:
        joblib.dump(calibrated_model, calibrated_model_path)
    
    # 保存最佳参数
    with open(best_params_path, 'w') as f:
        json.dump(best_params, f, indent=2)
    
    # 保存评估结果
    with open(results_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    # 保存模型摘要（包含防过拟合分析）
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_report)
        f.write(f"\n\nDetailed Parameters:\n")
        f.write(f"Best Parameters: {best_params}\n")
        f.write(f"\nDetailed Test Metrics:\n")
        for key, value in test_metrics.items():
            f.write(f"{key}: {value:.4f}\n")
    
    # 返回优化后的结果
    return_dict = {
        "model": model,
        "best_params": best_params,
        "cv_score": cv_score,
        "optimization_method": "bayesian",
        "bayesian_available": BAYESIAN_AVAILABLE,
        "train_metrics": train_metrics,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "results_dir": results_dir,
        "model_path": model_path,
        "best_params_path": best_params_path,
        "results_path": results_path,
        "summary_path": summary_path,
        "feature_importance": model.feature_importances_,
        "overfitting_analysis": {
            "risk_level": overfitting_risk,
            "accuracy_gap": train_test_gap,
            "f1_gap": train_test_f1_gap,
            "adaptive_regularization": adaptive_reg,
            "final_regularization": {
                "reg_alpha": best_params.get('reg_alpha', 0),
                "reg_lambda": best_params.get('reg_lambda', 0)
            }
        },
        "model_complexity": model_complexity,
        "feature_analysis": feature_analysis,
        "summary_report": summary_report,
    }
    if calibrated_model is not None:
        return_dict["calibration_method"] = calibrated_method
        return_dict["calibrated_model_path"] = calibrated_model_path
        return_dict["tuned_threshold"] = tuned_threshold
        return_dict["tuned_fbeta"] = tuned_fbeta

    # 如果是贝叶斯优化，保存优化历史
    if BAYESIAN_AVAILABLE and hasattr(search, "cv_results_"):
        history_path = os.path.join(results_dir, "optimization_history.json")
        optimization_history = {
            "cv_results": {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in search.cv_results_.items()},
            "best_index": search.best_index_,
            "best_score": search.best_score_
        }
        with open(history_path, 'w') as f:
            json.dump(optimization_history, f, indent=2)
        return_dict["optimization_history_path"] = history_path

    if not is_multiclass:
        return_dict["roc_auc"] = {
            "train": roc_auc_score(y_train, y_train_proba[:, 1]),
            "validation": roc_auc_score(y_val, y_val_proba[:, 1]),
            "test": roc_auc_score(y_test, y_test_proba[:, 1]),
        }

    # wandb实验结束调用已移除

    return return_dict
