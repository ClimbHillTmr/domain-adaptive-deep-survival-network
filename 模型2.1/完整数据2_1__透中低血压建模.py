# -*- coding: utf-8 -*-
"""
透析患者透中低血压预测模型 - 模型2.1完全优化版

模型2.1定义：使用n天的透析数据预测n+1天的透中低血压

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

作者: AI Assistant
日期: 2024
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

import sys
sys.path.append("..")
sys.path.append("/home/cht/Works/PredictionTimeHypotensionDialysis")

from Method_Utils.train_untils import (
    remove_rows_with_few_duplicates,
    create_moved_dataframe,
    process_dataset,
    Align_standard,
    features_based,
    features,
    quantile_99,
    Iterate_columns,
    calculate_class_weights,
)
from Method_Utils.data_standardization import *

# 导入高级缺失值填补和类别不平衡处理模块
from Method_Utils.advanced_imputation import AdvancedImputationPipeline
from Method_Utils.class_imbalance_handler import (
    ClassImbalanceHandler,
    data_resampling,
    OptimizedModelPipeline
)

# 导入所有模型
from Methods.Models.LightGBM_model import LightGBM_model
from Methods.Models.C_SVM_model import C_SVM_model
from Methods.Models.TabNet_optimized import TabNet_model_optimized
from Methods.Models.IEDT_model import IEDT_model
from Methods.fuding_test import fuding_test
from Methods.date_method import split_dataset_by_date

from sklearn import preprocessing


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
        
        print(f"模型结果已保存到: {results_file}")
        
    except Exception as e:
        print(f"保存模型结果时出错: {e}")

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
        
        print(f"性能报告已保存到: {report_file}")
        
    except Exception as e:
        print(f"生成模型报告时出错: {e}")


def train_multiple_models(X_train, X_val, X_test, y_train, y_val, y_test, class_weights, target, kinds):
    """训练和比较多个模型"""
    results = {}
    
    print(f"\n=== 开始训练多个模型 ===")
    print(f"目标变量: {target}")
    print(f"训练集大小: {X_train.shape}")
    print(f"验证集大小: {X_val.shape}")
    print(f"测试集大小: {X_test.shape}")
    
    # 1. LightGBM模型
    print(f"\n--- 训练 LightGBM 模型 ---")
    try:
        lgb_result = LightGBM_model(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            X_val=X_val,
            y_val=y_val,
            class_weights=class_weights,
            target=target,
            kinds=kinds,
            optimization_method='bayesian',
            use_early_stopping=True,
            n_iter=50,
            cv_folds=5,
            use_validation_in_training=True
        )
        results['LightGBM'] = lgb_result
        print(f"LightGBM 训练完成!")
    except Exception as e:
        print(f"LightGBM 训练失败: {e}")
        results['LightGBM'] = None
    
    # 2. SVM模型
    print(f"\n--- 训练 SVM 模型 ---")
    try:
        svm_model, svm_scores = C_SVM_model(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            X_val=X_val,
            y_val=y_val,
            class_weights=class_weights,
            target=target,
            kinds=kinds,
            n_iter=50,
            cv_folds=5,
            use_validation_in_training=True
        )
        results['SVM'] = {'model': svm_model, 'scores': svm_scores}
        print(f"SVM 训练完成!")
    except Exception as e:
        print(f"SVM 训练失败: {e}")
        results['SVM'] = None
    
    # 3. TabNet模型
    print(f"\n--- 训练 TabNet 模型 ---")
    try:
        tabnet_model, tabnet_opt_results = TabNet_model_optimized(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            X_val=X_val,
            y_val=y_val,
            class_weights=class_weights,
            target=target,
            kinds=kinds,
            optimize=True,
            search_type='bayesian',
            n_iter=50,
            cv_folds=5,
            use_validation_in_training=True
        )
        results['TabNet'] = {'model': tabnet_model, 'optimization_results': tabnet_opt_results}
        print(f"TabNet 训练完成!")
    except Exception as e:
        print(f"TabNet 训练失败: {e}")
        results['TabNet'] = None
    
    # 4. IEDT模型 (可解释性集成决策树)
    print(f"\n--- 训练 IEDT 模型 ---")
    try:
        iedt_model, iedt_scores = IEDT_model(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            X_val=X_val,
            y_val=y_val,
            class_weights=class_weights,
            target=target,
            kinds=kinds,
            optimization_method='bayesian',
            n_iter=50,
            cv_folds=5
        )
        results['IEDT'] = {'model': iedt_model, 'scores': iedt_scores}
        print(f"IEDT 训练完成!")
    except Exception as e:
        print(f"IEDT 训练失败: {e}")
        results['IEDT'] = None
    
    return results


def main():
    """主函数"""
    print("=== 透析患者透中低血压预测模型 - 模型2.1完全优化版 ===")
    
    # 加载数据
    print("\n加载数据...")
    dataset = pd.read_csv(
        "/home/cht/Works/PredictionTimeHypotensionDialysis/data_preprocessing/data/深医_final_data.csv"
    )
    
    test_set = fuding_test()
    
    # 选择基础特征
    dataset = dataset[features_based]
    test_set = test_set[features_based]
    
    print("数据基本信息:")
    dataset.info()
    test_set.info()
    
    # 数据预处理
    dataset = remove_rows_with_few_duplicates(dataset)
    test_set = remove_rows_with_few_duplicates(test_set)
    
    dataset = Align_standard(dataset)
    test_set = Align_standard(test_set)
    
    dataset = quantile_99(dataset)
    test_set = quantile_99(test_set)
    
    print("\n数据预处理后信息:")
    dataset.info()
    test_set.info()
    
    # 检查关键列是否存在
    print("\n检查关键列是否存在:")
    key_columns = ['透中低血压_计算', '降幅时间点比值区间', '降幅时间点差值区间']
    for col in key_columns:
        print(f"{col}: {'存在' if col in dataset.columns else '不存在'}")
    
    # 如果关键列不存在，显示包含'透中低血压'的所有列
    if '透中低血压_计算' not in dataset.columns:
        print("\n包含'透中低血压'的列:")
        related_cols = [col for col in dataset.columns if '透中低血压' in col]
        print(related_cols)

    # 处理时间序列数据（模型2.1特有：使用n天数据预测n+1天）
    columns_to_move = [
        "透中低血压_计算",
        "降幅时间点比值区间",
        "降幅时间点差值区间",
    ]
    
    dataset = process_dataset(df=dataset, columns_to_shift=columns_to_move)
    test_set = process_dataset(df=test_set, columns_to_shift=columns_to_move)
    
    print("\n时间序列处理后数据信息:")
    dataset.info()
    test_set.info()
    
    # 目标变量（模型2.1使用"新_"前缀的目标变量）
    targets = [
        "新_透中低血压_计算",
        "新_降幅时间点比值区间",
        "新_降幅时间点差值区间",
    ]
    
    # 历史特征
    history_rate_proportion = [
        "history_HBP_rate",
        "history_LBP_times_0_rate",
        "history_LBP_times_1_rate",
        "history_LBP_times_2_rate",
        "history_LBP_times_3_rate",
        "history_LBP_times_4_rate",
    ]
    
    history_rate_diff = [
        "history_HBP",
        "history_LBP_times_0",
        "history_LBP_times_1",
        "history_LBP_times_2",
        "history_LBP_times_3",
        "history_LBP_times_4",
    ]
    
    # 初始化模型管道
    pipeline = OptimizedModelPipeline(imputation_strategy='iterative')
    
    # 对每个目标变量进行建模
    for target in targets:
        try:
            print(f"\n{'='*60}")
            print(f"目标变量: {target}")
            print(f"{'='*60}")
            
            # 选择特征
            current_features = features.copy()
            if target in ["新_降幅时间点比值区间"]:
                current_features.extend(history_rate_proportion)
            elif target in ["新_降幅时间点差值区间"]:
                current_features.extend(history_rate_diff)        
            # 过滤数据（只对时间预测任务移除标签为4的样本）
            if target in ["新_降幅时间点比值区间", "新_降幅时间点差值区间"]:
                # 时间预测任务：移除标签为4的样本
                train_data = dataset[dataset[target] != 4].copy()
                test_data = test_set[test_set[target] != 4].copy()
            else:
                # 二分类任务：保留所有样本
                train_data = dataset.copy()
                test_data = test_set.copy()
            
            if len(train_data) == 0 or len(test_data) == 0:
                print(f"警告: {target} 的有效数据不足，跳过")
                continue
            
            # 准备特征和标签
            X = train_data[current_features]
            y = train_data[target]
            X_external_test = test_data[current_features]
            y_external_test = test_data[target]
            
            # 检查类别数量
            unique_labels = np.unique(y)
            n_classes = len(unique_labels)
            is_binary = n_classes == 2
            
            print(f"任务类型: {'二分类' if is_binary else f'{n_classes}分类'}")
            print(f"特征数量: {len(current_features)}")
            print(f"类别标签: {unique_labels}")
            
            # 显示数据信息
            print(f"\n训练数据形状: {X.shape}")
            print(f"测试数据形状: {X_external_test.shape}")
            
            # 数据准备（使用高级缺失值填补）
            X_train, X_val, X_test, y_train, y_val, y_test = pipeline.prepare_data(
                X, y, X_external_test, y_external_test
            )
            
            # 特征标准化 - 使用专用的透析数据标准化函数
            print(f"开始对 {target} 的数据进行标准化...")
            
            # 创建标准化器保存目录
            scaler_dir = "./scalers"
            os.makedirs(scaler_dir, exist_ok=True)
            
            # 使用透析数据专用标准化函数
            X_train_scaled, X_val_scaled, X_test_scaled, scaler_info = standardize_dialysis_data(
                X_train, X_val, X_test,
                scaler_type='standard',
                save_path=os.path.join(scaler_dir, f"scaler_{target}.joblib")
            )
            
            print(f"数据标准化完成，标准化器已保存")
            print(f"标准化统计信息: {scaler_info}")
            
            # 更新数据变量
            X_train, X_val, X_test = X_train_scaled, X_val_scaled, X_test_scaled
            
            # 计算类别权重
            class_weights = calculate_class_weights(y_train)
            print(f"类别权重: {class_weights}")
            
            # 处理多分类的类别不平衡
            if not is_binary and n_classes > 2:
                print(f"\n检测到多分类问题，应用重采样...")
                X_train_resampled, y_train_resampled = pipeline.handle_class_imbalance(
                    X_train, y_train, method='smotetomek'
                )
                print(f"重采样完成: {X_train.shape} -> {X_train_resampled.shape}")
            else:
                X_train_resampled, y_train_resampled = X_train, y_train
            
            # 打印数据集大小信息
            print(f"\n训练集总长度: {len(X_train_resampled)}, label 为 1 的长度: {np.sum(y_train_resampled == 1)}")
            print(f"验证集总长度: {len(X_val)}, label 为 1 的长度: {np.sum(y_val == 1)}")
            print(f"测试集总长度: {len(X_test)}, label 为 1 的长度: {np.sum(y_test == 1)}")
            
            # 训练多个模型
            model_results = train_multiple_models(
                X_train_resampled, X_val, X_test, 
                y_train_resampled, y_val, y_test,
                class_weights, target, "模型2.1_完全优化版"
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
            
            # 保存模型结果和生成报告
            try:
                save_model_results(model_results, target, "./model_results_2.1")
                generate_model_report(model_results, target, "./model_results_2.1")
                print(f"模型结果已保存到: ./model_results_2.1")
            except Exception as e:
                print(f"保存模型时出错: {e}")
            
            print(f"目标变量 {target} 的所有模型训练完成")
        
        except Exception as e:
            print(f"处理目标变量 {target} 时发生错误: {e}")
    
print(f"\n{'='*60}")
print("所有模型训练完成!")
print(f"{'='*60}")


if __name__ == "__main__":
    main()
