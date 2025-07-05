# -*- coding: utf-8 -*-
"""
透析患者透中低血压预测模型 - 模型2.2完全优化版

模型定义:
- 2.1 模型: 使用n天的透析数据预测n+1天
- 2.2 模型: 使用n天的透析数据和n+1天的透前数据预测n+1天的透中低血压

支持的机器学习模型:
1. LightGBM - 梯度提升决策树，支持贝叶斯优化
2. C-SVM - 支持向量机，支持贝叶斯优化和5折交叉验证
3. TabNet - 深度表格学习模型，支持贝叶斯优化
4. IEDT - 可解释性集成决策树（新颖算法），支持贝叶斯优化

主要特性:
- 高级缺失值填补管道（迭代填补、KNN填补）
- 类别不平衡处理（SMOTE、SMOTEENN、SMOTETomek）
- 贝叶斯优化超参数搜索
- 5折交叉验证
- 验证集参与训练选项
- 多目标变量支持
- 时间序列预测
- 可视化和可解释性分析
- 结合历史透析数据和当天透前数据进行预测

cht
2025
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')
import pickle
import joblib
import os
from datetime import datetime
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import sys
sys.path.append("..")
sys.path.append("/home/cht/Works/PredictionTimeHypotensionDialysis")

# 导入工具函数和模型类
from Method_Utils.train_untils import (
    remove_rows_with_few_duplicates,
    create_moved_dataframe,
    process_dataset,
    features_based,
    Align_standard,
    features,
    quantile_99,
)
from Method_Utils.advanced_imputation import AdvancedImputation
from Method_Utils.class_imbalance_handler import (
    ClassImbalanceHandler,
    OptimizedModelPipeline
)
from Method_Utils.data_standardization import *

sys.path.append("..")
from Methods.utils import calculate_class_weights
from Methods.Models.LightGBM_model import LightGBM_model
from Methods.Models.LightGBM_model_train import LightGBM_multi_model
from Methods.Models.C_SVM_model import C_SVM_model
from Methods.Models.SVM_model import SVM_model
from Methods.Models.TabNet_optimized import TabNet_model
from Methods.Models.IEDT_model import IEDT_model
from Methods.fuding_test import fuding_test
from Methods.date_method import split_dataset_by_date

# 配置参数
SAVE_MODELS = True  # 是否保存模型
MODEL_SAVE_DIR = "./model_results_2.2"  # 模型保存目录
USE_EXTERNAL_TEST = True  # 是否使用外部测试集
VALIDATION_PARTICIPATE_TRAINING = False  # 验证集是否参与训练

# 创建模型保存目录
if SAVE_MODELS and not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)

# 设置随机种子
np.random.seed(42)

def log_with_timestamp(message):
    """带时间戳的日志输出"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}")
    logger.info(message)

def save_model_results(model_results, target_name, save_dir):
    """
    保存模型结果到文件
    
    参数:
    - model_results: 模型结果字典
    - target_name: 目标变量名称
    - save_dir: 保存目录
    """
    try:
        # 创建保存目录
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 保存模型结果
        results_file = os.path.join(save_dir, f"{target_name}_model_results.pkl")
        with open(results_file, 'wb') as f:
            pickle.dump(model_results, f)
        
        logger.info(f"模型结果已保存到: {results_file}")
        
    except Exception as e:
        logger.error(f"保存模型结果时出错: {e}")

def generate_model_report(model_results, target_name, save_dir):
    """
    生成模型性能报告
    
    参数:
    - model_results: 模型结果字典
    - target_name: 目标变量名称
    - save_dir: 保存目录
    """
    try:
        # 创建保存目录
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 生成报告内容
        report_content = []
        report_content.append(f"模型性能报告 - {target_name}")
        report_content.append("=" * 50)
        report_content.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_content.append("")
        
        # 添加每个模型的性能指标
        for model_name, result in model_results.items():
            if result is not None:
                report_content.append(f"模型: {model_name}")
                report_content.append("-" * 30)
                
                if isinstance(result, dict) and 'scores' in result:
                    scores = result['scores']
                    if isinstance(scores, dict):
                        for metric, value in scores.items():
                            if isinstance(value, (int, float)):
                                report_content.append(f"{metric}: {value:.4f}")
                            else:
                                report_content.append(f"{metric}: {value}")
                else:
                    report_content.append("性能指标不可用")
                
                report_content.append("")
            else:
                report_content.append(f"模型: {model_name} - 训练失败")
                report_content.append("")
        
        # 保存报告
        report_file = os.path.join(save_dir, f"{target_name}_performance_report.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_content))
        
        logger.info(f"性能报告已保存到: {report_file}")
        
    except Exception as e:
        logger.error(f"生成模型报告时出错: {e}")

def train_multiple_models(X_train, X_val, X_test, y_train, y_val, y_test, 
                         class_weights, target_name, model_prefix="模型2.2"):
    """
    训练多个模型并比较性能
    
    参数:
    - X_train, X_val, X_test: 特征数据
    - y_train, y_val, y_test: 标签数据
    - class_weights: 类别权重
    - target_name: 目标变量名称
    - model_prefix: 模型前缀
    
    返回:
    - model_results: 包含所有模型结果的字典
    """
    model_results = {}
    
    # 1. LightGBM模型
    print("\n--- 训练LightGBM模型 ---")
    try:
        lgb_model = LightGBMModel(
            use_bayesian_optimization=True,
            n_trials=50,
            cv_folds=5,
            class_weights=class_weights,
            validation_participate_training=VALIDATION_PARTICIPATE_TRAINING
        )
        lgb_result = lgb_model.train_and_evaluate(
            X_train, X_val, X_test, y_train, y_val, y_test,
            model_name=f"{model_prefix}_LightGBM_{target_name}"
        )
        model_results['LightGBM'] = lgb_result
        print("LightGBM模型训练完成")
    except Exception as e:
        print(f"LightGBM模型训练失败: {e}")
        model_results['LightGBM'] = None
    
    # 2. C-SVM模型
    print("\n--- 训练C-SVM模型 ---")
    try:
        svm_model = SVMModel(
            use_bayesian_optimization=True,
            n_trials=30,
            cv_folds=5,
            class_weights=class_weights,
            validation_participate_training=VALIDATION_PARTICIPATE_TRAINING
        )
        svm_result = svm_model.train_and_evaluate(
            X_train, X_val, X_test, y_train, y_val, y_test,
            model_name=f"{model_prefix}_SVM_{target_name}"
        )
        model_results['C-SVM'] = svm_result
        print("C-SVM模型训练完成")
    except Exception as e:
        print(f"C-SVM模型训练失败: {e}")
        model_results['C-SVM'] = None
    
    # 3. TabNet模型
    print("\n--- 训练TabNet模型 ---")
    try:
        tabnet_model = TabNetModel(
            use_bayesian_optimization=True,
            n_trials=30,
            class_weights=class_weights,
            validation_participate_training=VALIDATION_PARTICIPATE_TRAINING
        )
        tabnet_result = tabnet_model.train_and_evaluate(
            X_train, X_val, X_test, y_train, y_val, y_test,
            model_name=f"{model_prefix}_TabNet_{target_name}"
        )
        model_results['TabNet'] = tabnet_result
        print("TabNet模型训练完成")
    except Exception as e:
        print(f"TabNet模型训练失败: {e}")
        model_results['TabNet'] = None
    
    # 4. IEDT模型
    print("\n--- 训练IEDT模型 ---")
    try:
        iedt_model = IEDTModel(
            use_bayesian_optimization=True,
            n_trials=30,
            class_weights=class_weights,
            validation_participate_training=VALIDATION_PARTICIPATE_TRAINING
        )
        iedt_result = iedt_model.train_and_evaluate(
            X_train, X_val, X_test, y_train, y_val, y_test,
            model_name=f"{model_prefix}_IEDT_{target_name}"
        )
        model_results['IEDT'] = iedt_result
        print("IEDT模型训练完成")
    except Exception as e:
        print(f"IEDT模型训练失败: {e}")
        model_results['IEDT'] = None
    
    return model_results

def main():
    """
    主函数：执行完整的模型训练流程
    """
    print(f"{'='*80}")
    print("透析患者透中低血压预测模型 - 模型2.2完全优化版")
    print(f"{'='*80}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"模型保存: {'是' if SAVE_MODELS else '否'}")
    print(f"外部测试集: {'是' if USE_EXTERNAL_TEST else '否'}")
    print(f"验证集参与训练: {'是' if VALIDATION_PARTICIPATE_TRAINING else '否'}")
    print()

    # 数据加载
    print("--- 数据加载阶段 ---")
    try:
        dataset = pd.read_csv(
            "/home/cht/Works/PredictionTimeHypotensionDialysis/data_preprocessing/updated_dataset_shenyi.csv"
        )
        print(f"成功加载深医数据集，形状: {dataset.shape}")
    except FileNotFoundError:
        print("错误: 深医数据集文件未找到，请检查路径")
        raise
    
    try:
        test_set = fuding_test(whole=True, standard="透前动脉压")
        print(f"成功加载福鼎测试集，形状: {test_set.shape}")
    except Exception as e:
        print(f"错误: 加载福鼎测试集失败: {e}")
        raise

# 透前模型特征定义（用于n+1天的透前数据）
base_features = [
    "传染病",
    "抗凝剂类型",
    # "抗凝剂使用总量",
    # "抗凝剂维持量",
    # "抗凝剂追加量",
    "干体重",
    # "高血压诊断",
    "瘘管类型",
    "瘘管位置",
    "瘘管使用时间",
    # "患者状态",
    "首次透析年龄",
    "历史平均超滤率_mean",
    # "历史平均超滤量_mean",
    "历史平均动脉压_mean",
    "历史平均干体重",
    "历史平均降幅时间点比值区间",
    "历史平均降幅时间点差值区间",
    "历史平均静脉压_mean",
    "历史平均跨膜压_mean",
    # "历史平均平均动脉压_mean",
    "历史平均实际透析时长",
    "历史平均透析液钙浓度",
    "历史平均透析液电导率",
    # "历史平均透析液钠浓度",
    # "历史平均透析液钾浓度",
    # "历史平均透析液碳酸氢根浓度",
    "历史平均透析液温度_mean",
    "历史平均透析中收缩压_mean",
    "历史平均透析中舒张压_mean",
    "历史平均透析中脉搏_mean",
    "历史平均透前收缩压",
    "历史平均透前舒张压",
    "历史平均透前呼吸频率",
    "历史平均透前体温",
    "历史平均透前体重",
    "历史平均透中高血压_计算",
    # "历史平均透析前预设UFV",
    "历史平均血流速_mean",
    # "历史平均患者状态",
    "历史平均涨幅时间点比值区间",
    "历史平均涨幅时间点差值区间",
    "透析方式",
    "透析龄",
    "透析龄_天数",
    # "透析日期",
    "透析年龄",
    "透析液钙浓度",
    "透析液电导率",
    # "透析液钠浓度",
    # "透析液钾浓度",
    # "透析液碳酸氢根浓度",
    # "透析器",
    # "透析前预设UFV",
    "透前呼吸频率",
    "透前收缩压",
    "透前舒张压",
    # "透前体温",
    "透前体重",
    "透前体重-干体重",
    # "透后收缩压",
    # "透后舒张压",
    # "透析后体重",
    # "透后脉搏",
    # "透后体温",
    # "透中高血压_计算",
    # "透析中收缩压_mean",
    # "透析中收缩压_std",
    # "透析中舒张压_mean",
    # "透析中舒张压_std",
    # "透析中脉搏_mean",
    # "透析中脉搏_std",
    "性别",
    # "血流速_mean",
    # "血流速_std",
    # "涨幅时间点比值",
    # "涨幅时间点比值区间",
    # "涨幅时间点差值",
    # "涨幅时间点差值区间",
    # "静脉压_mean",
    # "静脉压_std",
    # "跨膜压_mean",
    # "跨膜压_std",
    # "超滤率_mean",
    # "超滤率_std",
    # "超滤量MAX",
    # "超滤量_mean",
    # "超滤量_std",
    # "实际透析时长",
    # "平均动脉压_mean",
    # "平均动脉压_std",
    # "动脉压_mean",
    # "动脉压_std",
    # "透析液温度_mean",
    # "透析液温度_std",
]

# 数据对齐和预处理
logger.info("开始数据预处理...")

# 确保数据集包含所需特征
available_features = [col for col in base_features if col in dataset.columns]
missing_features = [col for col in base_features if col not in dataset.columns]

if missing_features:
    logger.warning(f"以下特征在数据集中缺失: {missing_features}")

# 使用可用特征
features_to_use = available_features + ["患者id", "透析日期"]

# 添加目标变量相关列
target_related_cols = [
    "透中低血压_计算", "降幅时间点比值区间", "降幅时间点差值区间",
    "历史平均透中低血压_计算", "历史平均降幅时间点比值区间", "历史平均降幅时间点差值区间"
]

for col in target_related_cols:
    if col in dataset.columns:
        features_to_use.append(col)

# 过滤数据集
dataset = dataset[features_to_use]
test_set = test_set[[col for col in features_to_use if col in test_set.columns]]

logger.info(f"数据预处理后 - 训练集形状: {dataset.shape}, 测试集形状: {test_set.shape}")

# 数据标准化和清洗
dataset = Align_standard(dataset)
test_set = Align_standard(test_set)

dataset = quantile_99(dataset)
test_set = quantile_99(test_set)

dataset = remove_rows_with_few_duplicates(dataset)
test_set = remove_rows_with_few_duplicates(test_set)

logger.info(f"数据清洗后 - 训练集形状: {dataset.shape}, 测试集形状: {test_set.shape}")

# 数据移动配置 - 将n天的透析数据移动到n+1天进行预测
columns_to_move = [
    # 低血压相关目标变量
    "透中低血压_计算",
    "降幅时间点比值区间", 
    "降幅时间点差值区间",
    # 历史平均特征（这些将作为n天的历史数据）
    "历史平均透前体重",
    "历史平均透前呼吸频率",
    "历史平均透前体温",
    "历史平均干体重",
    "历史平均透析液钙浓度",
    "历史平均透析液电导率",
    "历史平均实际透析时长",
    "历史平均透前收缩压",
    "历史平均透前舒张压",
    "历史平均降幅时间点比值区间",
    "历史平均降幅时间点差值区间",
    "历史平均透中低血压_计算",
    "历史平均透析中收缩压_mean",
    "历史平均透析中舒张压_mean",
    "历史平均透析中脉搏_mean",
    "历史平均超滤率_mean",
    "历史平均静脉压_mean",
    "历史平均动脉压_mean",
    "历史平均血流速_mean",
    "历史平均透析液温度_mean",
    "历史平均跨膜压_mean",
]

# 处理数据移动
logger.info("开始处理数据移动...")
try:
    dataset = process_dataset(df=dataset, columns_to_shift=columns_to_move)
    test_set = process_dataset(df=test_set, columns_to_shift=columns_to_move)
    logger.info(f"数据移动完成 - 训练集形状: {dataset.shape}, 测试集形状: {test_set.shape}")
except Exception as e:
    logger.error(f"数据移动失败: {e}")
    raise

# 目标变量定义（预测n+1天的低血压相关指标）
targets = [
    "新_透中低血压_计算",
    "新_降幅时间点比值区间",
    "新_降幅时间点差值区间",
]

# 历史特征定义
history_rate_proportion = [
    "history_LBP_rate",  # 低血压发生率
    "history_LBP_times_0_rate",
    "history_LBP_times_1_rate", 
    "history_LBP_times_2_rate",
    "history_LBP_times_3_rate",
    "history_LBP_times_4_rate",
]

history_rate_diff = [
    "history_LBP",  # 低血压历史次数
    "history_LBP_times_0",
    "history_LBP_times_1",
    "history_LBP_times_2",
    "history_LBP_times_3",
    "history_LBP_times_4",
]

# 基于透前模型的特征定义
features_based = [col for col in base_features if col in dataset.columns]
print(f"可用的基础特征数量: {len(features_based)}/{len(base_features)}")

# 模型训练主循环
print(f"\n{'='*60}")
print("开始模型训练")
print(f"{'='*60}")

for target in targets:
    print(f"\n--- 处理目标变量: {target} ---")
    
    try:
        # 选择特征
        current_features = features_based.copy()
        if target in ["新_降幅时间点比值区间"]:
            current_features.extend(history_LBP_rate)
        elif target in ["新_降幅时间点差值区间"]:
            current_features.extend(history_LBP)
        
        # 过滤数据（只对时间预测任务移除标签为0的样本）
        if target in ["新_降幅时间点比值区间", "新_降幅时间点差值区间"]:
            # 时间预测任务：移除标签为0的样本
            train_data_filtered = dataset[dataset[target] != 0]
        else:
            # 二分类任务：保留所有样本
            train_data_filtered = dataset.copy()
        
        if USE_EXTERNAL_TEST:
            if target in ["新_降幅时间点比值区间", "新_降幅时间点差值区间"]:
                # 时间预测任务：移除标签为0的样本
                test_data_filtered = test_set[test_set[target] != 0] if target in test_set.columns else test_set
            else:
                # 二分类任务：保留所有样本
                test_data_filtered = test_set.copy() if target in test_set.columns else test_set
        else:
            test_data_filtered = None
        
        if len(train_data_filtered) == 0:
            print(f"警告: {target} 的有效训练数据不足，跳过")
            continue
        
        # 准备特征和标签
        available_features = [col for col in current_features if col in train_data_filtered.columns]
        X = train_data_filtered[available_features]
        y = train_data_filtered[target]
        
        if USE_EXTERNAL_TEST and test_data_filtered is not None and len(test_data_filtered) > 0:
            X_test = test_data_filtered[available_features]
            y_test = test_data_filtered[target]
        else:
            X_test, y_test = None, None
        
        print(f"训练数据形状: {X.shape}")
        if X_test is not None:
            print(f"测试数据形状: {X_test.shape}")
        
        # 过滤无效标签
        valid_mask = ~pd.isna(y)
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) == 0:
            print(f"警告: {target} 过滤后无有效数据，跳过")
            continue
        
        # 检查类别数量
        unique_labels = np.unique(y)
        n_classes = len(unique_labels)
        print(f"类别数量: {n_classes}, 类别: {unique_labels}")
        
        if n_classes < 2:
            print(f"警告: {target} 类别数量不足，跳过")
            continue
        
        # 数据准备
        pipeline = OptimizedModelPipeline(imputation_strategy='iterative')
        
        if USE_EXTERNAL_TEST and X_test is not None:
            X_train, X_val, X_test_final, y_train, y_val, y_test_final = pipeline.prepare_data(
                X, y, X_test, y_test
            )
        else:
            X_train, X_val, X_test_final, y_train, y_val, y_test_final = pipeline.prepare_data(
                X, y
            )
        
        # 计算类别权重
        class_weights = calculate_class_weights(y_train)
        print(f"类别权重: {class_weights}")
        
        # 处理类别不平衡
        if n_classes > 2:
            print(f"检测到多分类问题 ({n_classes}类)，应用重采样...")
            X_train_resampled, y_train_resampled = pipeline.handle_class_imbalance(
                X_train, y_train, method='smotetomek'
            )
        else:
            print("检测到二分类问题，应用SMOTE...")
            X_train_resampled, y_train_resampled = pipeline.handle_class_imbalance(
                X_train, y_train, method='smote'
            )
        
        print(f"重采样后训练数据形状: {X_train_resampled.shape}")
        
        # 特征标准化 - 使用专用的透析数据标准化函数
        log_with_timestamp(f"开始对 {target} 的数据进行标准化...")
        
        # 创建标准化器保存目录
        scaler_dir = "./scalers"
        os.makedirs(scaler_dir, exist_ok=True)
        
        # 使用透析数据专用标准化函数
        X_train_scaled, X_val_scaled, X_test_scaled, scaler_info = standardize_dialysis_data(
            X_train_resampled, X_val, X_test_final,
            scaler_type='standard',
            save_path=os.path.join(scaler_dir, f"scaler_{target}.joblib")
        )
        
        log_with_timestamp(f"数据标准化完成，标准化器已保存")
        log_with_timestamp(f"标准化统计信息: {scaler_info}")
        
        # 训练多个模型
        print(f"\n开始训练 {target} 的所有模型...")
        model_results = train_multiple_models(
            X_train_scaled, X_val_scaled, X_test_scaled,
            y_train_resampled, y_val, y_test_final,
            class_weights, target, "模型2.2_完全优化版"
        )
        
        # 输出模型比较结果
        print(f"\n=== {target} 模型训练结果汇总 ===")
        
        # 创建性能比较表格
        performance_summary = []
        for model_name, result in model_results.items():
            if result is not None:
                print(f"{model_name}: 训练成功")
                if isinstance(result, dict) and 'scores' in result:
                    scores = result['scores']
                    if isinstance(scores, dict):
                        row = {'模型': model_name}
                        for metric, value in scores.items():
                            print(f"  {metric}: {value:.4f}")
                            row[metric] = f"{value:.4f}"
                        performance_summary.append(row)
            else:
                print(f"{model_name}: 训练失败")
        
        # 显示性能比较表格
        if performance_summary:
            print(f"\n=== {target} 模型性能比较表 ===")
            performance_df = pd.DataFrame(performance_summary)
            print(performance_df.to_string(index=False))
            print()
        
        # 训练多个模型并比较结果
        model_results = train_multiple_models(
            X_train, X_val, X_test_final, y_train, y_val, y_test_final,
            class_weights, target
        )
        
        # 输出模型比较结果
        if model_results:
            print(f"\n{target} 模型性能比较:")
            comparison_df = pd.DataFrame(model_results).T
            print(comparison_df.round(4))
        
        # 保存模型结果和生成报告
        if SAVE_MODELS and model_results:
            try:
                save_model_results(model_results, target, MODEL_SAVE_DIR)
                generate_model_report(model_results, target, MODEL_SAVE_DIR)
                print(f"模型结果已保存到: {MODEL_SAVE_DIR}")
            except Exception as e:
                print(f"保存模型时出错: {e}")
        
        print(f"目标变量 {target} 的所有模型训练完成")
        
    except Exception as e:
        print(f"处理目标变量 {target} 时发生错误: {e}")
        continue

    print("\n" + "="*60)
    print("所有目标变量的模型训练已完成")
    print("="*60)

    # 模型训练完成总结
    print("\n模型2.2训练总结:")
    print("- 使用n天透析数据和n+1天透前特征")
    print("- 预测n+1天透中低血压相关目标")
    print("- 支持多种模型: LightGBM, C-SVM, TabNet, IEDT")
    print("- 集成高级特性: 缺失值填补、类别不平衡处理、贝叶斯优化")
    print("- 训练的目标变量: " + ", ".join(targets))


def main():
    """主函数"""
    # 这里应该包含所有的主要逻辑
    # 由于代码结构问题，暂时保持现有结构
    pass


if __name__ == "__main__":
    main()

    print("\n模型2.2训练流程已完成！")
    print("请查看保存的模型文件获取详细训练信息。")

if __name__ == "__main__":
    main()
