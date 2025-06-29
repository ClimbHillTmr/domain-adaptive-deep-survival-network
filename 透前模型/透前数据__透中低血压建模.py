# -*- coding: utf-8 -*-
"""
透析患者透中高血压预测模型 - 完全优化版

支持的机器学习模型:
1. LightGBM - 梯度提升决策树，支持贝叶斯优化
2. C-SVM - 支持向量机，支持贝叶斯优化和5折交叉验证
3. TabNet - 深度表格学习模型，支持贝叶斯优化
4. IEDT - 可解释性集成决策树（新颖算法），支持贝叶斯优化
5. DialysisGNN - 图神经网络模型，用于捕获患者间复杂关系
6. AttentionKNN - 注意力机制增强的K近邻模型，自适应特征权重

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
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
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
    calculate_class_weights,
)

# 导入配置管理器
from Method_Utils.model_config import get_config, apply_preset, setup_environment, config_manager

# 导入高级缺失值填补和类别不平衡处理模块
from Method_Utils.advanced_imputation import AdvancedImputationPipeline, create_reproducible_imputer
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
from Methods.Models.dialysis_gnn_model import DialysisGNNClassifier
from Methods.Models.attention_knn_model import AttentionKNN
from Methods.fuding_test import fuding_test
from Methods.date_method import split_dataset_by_date

from sklearn import preprocessing



def train_multiple_models(X_train, X_val, X_test, y_train, y_val, y_test, class_weights, target, kinds,optimization_method='bayesian',cv=5):
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
            optimization_method=optimization_method,
            use_early_stopping=True,
            n_iter=50,
            cv_folds=cv,
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
            cv_folds=cv,
            use_validation_in_training=True
        )
        results['SVM'] = {'model': svm_model, 'scores': svm_scores}
        print(f"SVM 训练完成!")
    except Exception as e:
        print(f"SVM 训练失败: {e}")
        results['SVM'] = None
    
    # # 3. TabNet模型
    # print(f"\n--- 训练 TabNet 模型 ---")
    # try:
    #     tabnet_model, tabnet_opt_results = TabNet_model_optimized(
    #         X_train=X_train,
    #         X_test=X_test,
    #         y_train=y_train,
    #         y_test=y_test,
    #         X_val=X_val,
    #         y_val=y_val,
    #         class_weights=class_weights,
    #         target=target,
    #         kinds=kinds,
    #         optimize=True,
    #         search_type=optimization_method,
    #         n_iter=50,
    #         cv_folds=cv,
    #         use_validation_in_training=True
    #     )
    #     results['TabNet'] = {'model': tabnet_model, 'optimization_results': tabnet_opt_results}
    #     print(f"TabNet 训练完成!")
    # except Exception as e:
    #     print(f"TabNet 训练失败: {e}")
    #     results['TabNet'] = None
    
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
            optimization_method=optimization_method,
            n_iter=50,
            cv_folds=cv
        )
        results['IEDT'] = {'model': iedt_model, 'scores': iedt_scores}
        print(f"IEDT 训练完成!")
    except Exception as e:
        print(f"IEDT 训练失败: {e}")
        results['IEDT'] = None
    
    # 5. 图神经网络模型 (Dialysis GNN)
    print(f"\n--- 训练 图神经网络 模型 ---")
    try:
        # 合并训练集和验证集用于GNN训练
        X_gnn_train = np.vstack([X_train, X_val])
        y_gnn_train = np.hstack([y_train, y_val])
        
        gnn_model = DialysisGNNClassifier(
            hidden_dim=128,
            num_layers=3,
            num_heads=4,
            dropout=0.1,
            learning_rate=0.001,
            epochs=100,
            batch_size=32,
            early_stopping_patience=10,
            random_state=42,
            verbose=True
        )
        
        # 训练模型
        gnn_model.fit(X_gnn_train, y_gnn_train)
        
        # 预测和评估
        y_pred_gnn = gnn_model.predict(X_test)
        y_proba_gnn = gnn_model.predict_proba(X_test)
        
        gnn_scores = {
            'accuracy': accuracy_score(y_test, y_pred_gnn),
            'precision': precision_score(y_test, y_pred_gnn, average='weighted'),
            'recall': recall_score(y_test, y_pred_gnn, average='weighted'),
            'f1': f1_score(y_test, y_pred_gnn, average='weighted')
        }
        
        if y_proba_gnn.shape[1] == 2:  # 二分类
            gnn_scores['auc'] = roc_auc_score(y_test, y_proba_gnn[:, 1])
        
        results['GNN'] = {'model': gnn_model, 'scores': gnn_scores, 'predictions': y_pred_gnn, 'probabilities': y_proba_gnn}
        print(f"图神经网络 训练完成! AUC: {gnn_scores.get('auc', 'N/A'):.4f}")
    except Exception as e:
        print(f"图神经网络 训练失败: {e}")
        results['GNN'] = None
    
    # 6. 注意力KNN模型 (Attention KNN)
    print(f"\n--- 训练 注意力KNN 模型 ---")
    try:
        # 合并训练集和验证集用于AttentionKNN训练
        X_aknn_train = np.vstack([X_train, X_val])
        y_aknn_train = np.hstack([y_train, y_val])
        
        aknn_model = AttentionKNN(
            n_neighbors=5,
            attention_hidden_dim=64,
            learning_rate=0.001,
            epochs=100,
            batch_size=32,
            dropout=0.1,
            distance_metric='euclidean',
            feature_scaler='standard',
            early_stopping_patience=10,
            random_state=42,
            verbose=True
        )
        
        # 训练模型
        aknn_model.fit(X_aknn_train, y_aknn_train)
        
        # 预测和评估
        y_pred_aknn = aknn_model.predict(X_test)
        y_proba_aknn = aknn_model.predict_proba(X_test)
        
        aknn_scores = {
            'accuracy': accuracy_score(y_test, y_pred_aknn),
            'precision': precision_score(y_test, y_pred_aknn, average='weighted'),
            'recall': recall_score(y_test, y_pred_aknn, average='weighted'),
            'f1': f1_score(y_test, y_pred_aknn, average='weighted')
        }
        
        if y_proba_aknn.shape[1] == 2:  # 二分类
            aknn_scores['auc'] = roc_auc_score(y_test, y_proba_aknn[:, 1])
        
        results['AttentionKNN'] = {'model': aknn_model, 'scores': aknn_scores, 'predictions': y_pred_aknn, 'probabilities': y_proba_aknn}
        print(f"注意力KNN 训练完成! AUC: {aknn_scores.get('auc', 'N/A'):.4f}")
    except Exception as e:
        print(f"注意力KNN 训练失败: {e}")
        results['AttentionKNN'] = None
    
    return results


def main():
    """主函数"""
    # 配置参数
    SAVE_MODELS = True  # 是否保存模型结果
    MODEL_SAVE_DIR = './model_results'  # 模型保存目录
    
    print("=== 透析患者透中高血压预测模型 - 完全优化版 ===")
    print(f"模型保存: {'启用' if SAVE_MODELS else '禁用'}")
    if SAVE_MODELS:
        print(f"保存目录: {MODEL_SAVE_DIR}")
    
    # 加载数据
    print("\n加载数据...")
    dataset = pd.read_csv(
        "/home/cht/Works/PredictionTimeHypotensionDialysis/data_preprocessing/data/深医_final_data.csv"
    )
    # dataset = pd.read_csv(
    #     "/home/cht/Works/PredictionTimeHypotensionDialysis/透前模型/test_data_1.csv"
    # )
    
    test_set = fuding_test()
    # test_set = pd.read_csv(
    #     "/home/cht/Works/PredictionTimeHypotensionDialysis/透前模型/test_data_1.csv"
    # )
    
    # 数据预处理
    print(f"原始数据形状 - 训练集: {dataset.shape}, 测试集: {test_set.shape}")
    
    dataset = Align_standard(dataset)
    test_set = Align_standard(test_set)
    print(f"Align_standard处理后 - 训练集: {dataset.shape}, 测试集: {test_set.shape}")
    
    dataset = quantile_99(dataset)
    test_set = quantile_99(test_set)
    print(f"quantile_99处理后 - 训练集: {dataset.shape}, 测试集: {test_set.shape}")
    
    # 移除重复较少的行
    dataset = remove_rows_with_few_duplicates(dataset)
    test_set = remove_rows_with_few_duplicates(test_set)
    print(f"remove_rows_with_few_duplicates处理后 - 训练集: {dataset.shape}, 测试集: {test_set.shape}")
    
    # 检查目标变量分布
    for target in ["透中低血压_计算", "降幅时间点比值区间", "降幅时间点差值区间"]:
        if target in dataset.columns:
            print(f"训练集 {target} 分布: {dataset[target].value_counts().to_dict()}")
        if target in test_set.columns:
            print(f"测试集 {target} 分布: {test_set[target].value_counts().to_dict()}")
    
    # 特征定义
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
    # "历史平均透前体温",  # 缺失值比例50.33%，已被数据预处理过滤
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
    # "透前体温",  # 缺失值比例50.33%，已被数据预处理过滤
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
    
    # 目标变量
    targets = [
        "透中低血压_计算",
        "降幅时间点比值区间",
        "降幅时间点差值区间",
    ]
    
    # 历史特征
    history_rate_proportion = [
        "history_HBP_rate", "history_LBP_times_0_rate", "history_LBP_times_1_rate",
        "history_LBP_times_2_rate", "history_LBP_times_3_rate", "history_LBP_times_4_rate"
    ]
    
    history_rate_diff = [
        "history_HBP", "history_LBP_times_0", "history_LBP_times_1",
        "history_LBP_times_2", "history_LBP_times_3", "history_LBP_times_4"
    ]
    
    # 初始化模型管道 - 使用配置管理器
    config = get_config()
    imputation_config = config_manager.get_imputation_config()
    
    pipeline = OptimizedModelPipeline(
        imputation_strategy=imputation_config['strategy'],
        random_state=imputation_config['random_state']
    )
    
    # 对每个目标变量进行建模
    for target in targets:
        print(f"\n{'='*60}")
        print(f"目标变量: {target}")
        print(f"{'='*60}")
        
        # 选择特征
        current_features = base_features.copy()
        if target in ["降幅时间点比值区间"]:
            current_features.extend(history_rate_proportion)
        elif target in ["降幅时间点差值区间"]:
            current_features.extend(history_rate_diff)
        
        # 过滤数据（只对时间预测任务移除标签为0的样本）
        if target in ["降幅时间点比值区间", "降幅时间点差值区间"]:
            # 时间预测任务：移除标签为0的样本
            train_data = dataset[dataset[target] != 0].copy()
            test_data = test_set[test_set[target] != 0].copy()
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
        # 在数据预处理后确保特征名称一致性
        X_train = pd.DataFrame(X_train, columns=current_features)
        X_val = pd.DataFrame(X_val, columns=current_features)
        X_test = pd.DataFrame(X_test, columns=current_features)
        # 计算类别权重
        class_weights = calculate_class_weights(y_train)
        print(f"类别权重: {class_weights}")
        
        # 处理多分类的类别不平衡
        if not is_binary and n_classes > 2:
            print(f"\n检测到多分类问题，应用重采样...")
            X_train_resampled, y_train_resampled = pipeline.handle_class_imbalance(
                X_train, y_train, method='smotetomek'
            )
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
            class_weights, target, "透前模型_完全优化版"
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
        if SAVE_MODELS:
            try:
                save_model_results(model_results, target, MODEL_SAVE_DIR)
                generate_model_report(model_results, target, MODEL_SAVE_DIR)
            except Exception as e:
                print(f"保存模型时出错: {e}")
    
    print(f"\n{'='*60}")
    print("所有模型训练完成!")
    print(f"{'='*60}")
    
    if SAVE_MODELS:
        print(f"\n模型结果已保存到目录: {MODEL_SAVE_DIR}")

if __name__ == "__main__":
    print("开始透析低血压预测模型训练...")
    print("=" * 50)
    
    # 应用可重复性配置预设
    apply_preset('reproducible')
    setup_environment()
    
    # 获取配置
    config = get_config()
    print(f"使用配置: 填补策略={config.imputation_strategy}, 随机种子={config.random_state }")
    
    main()