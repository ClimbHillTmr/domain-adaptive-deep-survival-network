#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
透析患者低血压风险预测图神经网络模型 (Dialysis GNN)

该模型将透析患者的临床数据建模为图结构，利用图神经网络捕获：
1. 患者-设备关系
2. 患者-历史记录关系
3. 特征间的复杂交互
4. 时序依赖关系

主要特性：
- 多关系图建模
- 注意力机制
- 时序图卷积
- 可解释性分析
- 多任务学习

作者: 透析数据分析助手
日期: 2024
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, TransformerConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import to_networkx
import networkx as nx
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Union, Tuple, Dict, List, Any
import warnings
import joblib
from pathlib import Path

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

# 检查CUDA可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

def save_gnn_results(model, results, X_test, y_test, target='透中低血压', kinds='GNN', base_path=None):
    """
    保存GNN模型的完整结果
    
    Args:
        model: 训练好的GNN模型
        results: 评估结果字典
        X_test: 测试数据
        y_test: 测试标签
        target: 目标变量名称
        kinds: 模型类型
        base_path: 基础保存路径
    """
    import os
    import json
    from datetime import datetime
    
    # 创建结果目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if base_path is None:
        result_dir = f'./Results/{kinds}_{target}_{timestamp}'
    else:
        result_dir = os.path.join(base_path, f'{kinds}_{target}_{timestamp}')
    
    os.makedirs(result_dir, exist_ok=True)
    
    print(f"\n保存结果到: {result_dir}")
    
    try:
        # 1. 保存模型
        model_path = os.path.join(result_dir, f'{kinds}_model.pkl')
        model.save_model(model_path)
        print(f"✓ 模型已保存: {model_path}")
        
        # 2. 保存评估指标
        metrics_path = os.path.join(result_dir, 'evaluation_metrics.json')
        # 确保results中的numpy类型可以序列化
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                serializable_results[key] = float(value)
            else:
                serializable_results[key] = value
        
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        print(f"✓ 评估指标已保存: {metrics_path}")
        
        # 3. 保存测试数据和预测结果
        test_data_path = os.path.join(result_dir, 'test_data_and_predictions.csv')
        test_df = X_test.copy()
        test_df['true_label'] = y_test
        
        # 获取预测结果
        try:
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            test_df['predicted_label'] = y_pred
            test_df['prediction_probability'] = y_pred_proba[:, 1] if y_pred_proba.shape[1] > 1 else y_pred_proba[:, 0]
        except Exception as e:
            print(f"获取预测结果时出错: {e}")
            test_df['predicted_label'] = -1
            test_df['prediction_probability'] = -1
        
        test_df.to_csv(test_data_path, index=False, encoding='utf-8')
        print(f"✓ 测试数据和预测结果已保存: {test_data_path}")
        
        # 4. 生成并保存图表
        plots_dir = os.path.join(result_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # 使用Yellowbrick进行高级可视化（如果可用）
        if YELLOWBRICK_AVAILABLE and 'y_pred' in locals():
            print("使用Yellowbrick创建高级可视化...")
            
            # 获取唯一标签
            unique_labels_list = np.unique(np.concatenate([y_test, y_pred]))
            
            # 1. 类别平衡可视化
            try:
                visualizer = ClassBalance(labels=unique_labels_list)
                visualizer.fit(y_test)
                visualizer.show(outpath=os.path.join(plots_dir, f'{kinds}_class_balance.png'))
                plt.close()
                print("✓ 类别平衡图已保存")
            except Exception as e:
                print(f"类别平衡可视化错误: {e}")
            
            # 2. ROC曲线（Yellowbrick版本）
            if len(unique_labels_list) == 2 and 'y_pred_proba' in locals() and y_pred_proba.shape[1] > 1:
                try:
                    visualizer = ROCAUC(model, classes=unique_labels_list)
                    visualizer.fit(X_test, y_test)
                    visualizer.score(X_test, y_test)
                    visualizer.show(outpath=os.path.join(plots_dir, f'{kinds}_roc_auc.png'))
                    plt.close()
                    print("✓ ROC-AUC曲线已保存")
                except Exception as e:
                    print(f"ROC-AUC可视化错误: {e}")
            
            # 3. 精确率-召回率曲线
            if len(unique_labels_list) == 2 and 'y_pred_proba' in locals() and y_pred_proba.shape[1] > 1:
                try:
                    visualizer = PrecisionRecallCurve(model, classes=unique_labels_list)
                    visualizer.fit(X_test, y_test)
                    visualizer.score(X_test, y_test)
                    visualizer.show(outpath=os.path.join(plots_dir, f'{kinds}_precision_recall.png'))
                    plt.close()
                    print("✓ 精确率-召回率曲线已保存")
                except Exception as e:
                    print(f"精确率-召回率可视化错误: {e}")
            
            # 4. 判别阈值可视化
            if len(unique_labels_list) == 2 and 'y_pred_proba' in locals() and y_pred_proba.shape[1] > 1:
                try:
                    visualizer = DiscriminationThreshold(model)
                    visualizer.fit(X_test, y_test)
                    visualizer.score(X_test, y_test)
                    visualizer.show(outpath=os.path.join(plots_dir, f'{kinds}_discrimination_threshold.png'))
                    plt.close()
                    print("✓ 判别阈值图已保存")
                except Exception as e:
                    print(f"判别阈值可视化错误: {e}")
            
            # 5. 分类报告可视化
            try:
                visualizer = ClassificationReport(model, classes=unique_labels_list, support=True)
                visualizer.fit(X_test, y_test)
                visualizer.score(X_test, y_test)
                visualizer.show(outpath=os.path.join(plots_dir, f'{kinds}_classification_report.png'))
                plt.close()
                print("✓ 分类报告已保存")
            except Exception as e:
                print(f"分类报告可视化错误: {e}")
            
            # 6. 混淆矩阵（Yellowbrick版本）
            try:
                visualizer = ConfusionMatrix(model, classes=unique_labels_list)
                visualizer.fit(X_test, y_test)
                visualizer.score(X_test, y_test)
                visualizer.show(outpath=os.path.join(plots_dir, f'{kinds}_confusion_matrix_yb.png'))
                plt.close()
                print("✓ 混淆矩阵已保存")
            except Exception as e:
                print(f"混淆矩阵可视化错误: {e}")
            
            # 7. 类预测错误可视化
            try:
                visualizer = ClassPredictionError(model, classes=unique_labels_list)
                visualizer.fit(X_test, y_test)
                visualizer.score(X_test, y_test)
                visualizer.show(outpath=os.path.join(plots_dir, f'{kinds}_class_prediction_error.png'))
                plt.close()
                print("✓ 类预测错误图已保存")
            except Exception as e:
                print(f"类预测错误可视化错误: {e}")
        
        # 传统matplotlib可视化（保留原有功能）
        print("创建传统matplotlib可视化...")
        
        # 训练历史图
        try:
            plt.figure(figsize=(12, 4))
            model.plot_training_history()
            plt.savefig(os.path.join(plots_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ 训练历史图已保存")
        except Exception as e:
            print(f"保存训练历史图时出错: {e}")
        
        # 混淆矩阵图（matplotlib版本）
        try:
            if 'y_pred' in locals():
                plt.figure(figsize=(8, 6))
                cm = confusion_matrix(y_test, y_pred)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title(f'{kinds} 模型混淆矩阵')
                plt.ylabel('真实标签')
                plt.xlabel('预测标签')
                plt.savefig(os.path.join(plots_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
                plt.close()
                print("✓ 混淆矩阵图已保存")
        except Exception as e:
            print(f"保存混淆矩阵图时出错: {e}")
        
        # ROC曲线图（matplotlib版本）
        try:
            if 'y_pred_proba' in locals() and y_pred_proba.shape[1] > 1:
                plt.figure(figsize=(8, 6))
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
                auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
                plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('假阳性率')
                plt.ylabel('真阳性率')
                plt.title(f'{kinds} 模型 ROC 曲线')
                plt.legend(loc="lower right")
                plt.savefig(os.path.join(plots_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
                plt.close()
                print("✓ ROC曲线图已保存")
        except Exception as e:
            print(f"保存ROC曲线图时出错: {e}")
        
        # 5. 生成PDF报告
        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.lib import colors
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont
            
            # 注册中文字体
            try:
                pdfmetrics.registerFont(TTFont('SimHei', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
                font_name = 'SimHei'
            except:
                font_name = 'Helvetica'
            
            pdf_path = os.path.join(result_dir, f'{kinds}_model_report.pdf')
            doc = SimpleDocTemplate(pdf_path, pagesize=A4)
            story = []
            styles = getSampleStyleSheet()
            
            # 标题
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontName=font_name,
                fontSize=16,
                alignment=1  # 居中
            )
            story.append(Paragraph(f'{kinds} 模型评估报告', title_style))
            story.append(Spacer(1, 12))
            
            # 基本信息
            info_style = ParagraphStyle(
                'Info',
                parent=styles['Normal'],
                fontName=font_name,
                fontSize=10
            )
            
            story.append(Paragraph(f'目标变量: {target}', info_style))
            story.append(Paragraph(f'模型类型: {kinds}', info_style))
            story.append(Paragraph(f'生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', info_style))
            story.append(Paragraph(f'测试样本数: {len(y_test)}', info_style))
            story.append(Spacer(1, 12))
            
            # 评估指标表格
            if results:
                story.append(Paragraph('模型性能指标', styles['Heading2']))
                
                # 准备表格数据
                table_data = [['指标', '数值']]
                for key, value in serializable_results.items():
                    if isinstance(value, (int, float)):
                        table_data.append([str(key), f'{value:.4f}'])
                    elif isinstance(value, str):
                        table_data.append([str(key), str(value)])
                
                table = Table(table_data)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), font_name),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(table)
                story.append(Spacer(1, 12))
            
            # 添加图表
            for plot_name in ['training_history.png', 'confusion_matrix.png', 'roc_curve.png']:
                plot_path = os.path.join(plots_dir, plot_name)
                if os.path.exists(plot_path):
                    story.append(Paragraph(plot_name.replace('.png', '').replace('_', ' ').title(), styles['Heading2']))
                    img = Image(plot_path, width=6*inch, height=4*inch)
                    story.append(img)
                    story.append(Spacer(1, 12))
            
            doc.build(story)
            print(f"✓ PDF报告已生成: {pdf_path}")
            
        except Exception as e:
            print(f"生成PDF报告时出错: {e}")
        
        # 6. 保存模型摘要
        summary_path = os.path.join(result_dir, 'model_summary.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"{kinds} 模型摘要\n")
            f.write("=" * 50 + "\n")
            f.write(f"目标变量: {target}\n")
            f.write(f"模型类型: {kinds}\n")
            f.write(f"测试样本数: {len(y_test)}\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            if hasattr(model, 'hyperparameters'):
                f.write("模型超参数:\n")
                for key, value in model.hyperparameters.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
            
            if results:
                f.write("评估指标:\n")
                for key, value in serializable_results.items():
                    if isinstance(value, (int, float)):
                        f.write(f"  {key}: {value:.4f}\n")
                    else:
                        f.write(f"  {key}: {value}\n")
        
        print(f"✓ 模型摘要已保存: {summary_path}")
        print(f"\n所有结果已保存到: {result_dir}")
        
        return result_dir
        
    except Exception as e:
        print(f"保存结果时发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None

class DialysisGraphBuilder:
    """
    透析患者图构建器
    将表格数据转换为图结构
    """
    
    def __init__(self, 
                 patient_id_col: str = 'patient_id',
                 session_id_col: str = 'session_id',
                 temporal_features: List[str] = None,
                 categorical_features: List[str] = None,
                 numerical_features: List[str] = None):
        """
        初始化图构建器
        
        Args:
            patient_id_col: 患者ID列名
            session_id_col: 透析会话ID列名
            temporal_features: 时序特征列表
            categorical_features: 分类特征列表
            numerical_features: 数值特征列表
        """
        self.patient_id_col = patient_id_col
        self.session_id_col = session_id_col
        self.temporal_features = temporal_features or []
        self.categorical_features = categorical_features or []
        self.numerical_features = numerical_features or []
        
        # 编码器
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
        # 图统计信息
        self.graph_stats = {}
    
    def _encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """编码分类特征"""
        df_encoded = df.copy()
        
        for feature in self.categorical_features:
            if feature in df_encoded.columns:
                if fit:
                    if feature not in self.label_encoders:
                        self.label_encoders[feature] = LabelEncoder()
                    df_encoded[feature] = self.label_encoders[feature].fit_transform(
                        df_encoded[feature].astype(str)
                    )
                else:
                    if feature in self.label_encoders:
                        # 处理未见过的类别
                        unique_values = df_encoded[feature].astype(str).unique()
                        known_values = self.label_encoders[feature].classes_
                        
                        # 将未知值映射为最常见的类别
                        df_encoded[feature] = df_encoded[feature].astype(str)
                        unknown_mask = ~df_encoded[feature].isin(known_values)
                        if unknown_mask.any():
                            most_common = known_values[0]  # 使用第一个类别作为默认值
                            df_encoded.loc[unknown_mask, feature] = most_common
                        
                        df_encoded[feature] = self.label_encoders[feature].transform(
                            df_encoded[feature]
                        )
        
        return df_encoded
    
    def _scale_numerical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """标准化数值特征"""
        df_scaled = df.copy()
        
        if self.numerical_features:
            numerical_cols = [col for col in self.numerical_features if col in df_scaled.columns]
            if numerical_cols:
                if fit:
                    df_scaled[numerical_cols] = self.scaler.fit_transform(df_scaled[numerical_cols])
                else:
                    df_scaled[numerical_cols] = self.scaler.transform(df_scaled[numerical_cols])
        
        return df_scaled
    
    def _create_patient_similarity_edges(self, df: pd.DataFrame, similarity_threshold: float = 0.8) -> List[Tuple[int, int]]:
        """
        基于患者特征相似性创建边
        """
        edges = []
        
        # 选择用于相似性计算的特征
        similarity_features = ['年龄', '性别', '透析龄', '干体重'] if hasattr(self, 'similarity_features') else self.numerical_features[:5]
        available_features = [f for f in similarity_features if f in df.columns]
        
        if not available_features:
            return edges
        
        # 计算患者间的相似性
        feature_matrix = df[available_features].values
        n_patients = len(df)
        
        for i in range(n_patients):
            for j in range(i + 1, n_patients):
                # 计算余弦相似性
                vec1, vec2 = feature_matrix[i], feature_matrix[j]
                similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)
                
                if similarity > similarity_threshold:
                    edges.append((i, j))
                    edges.append((j, i))  # 无向图
        
        return edges
    
    def _create_temporal_edges(self, df: pd.DataFrame) -> List[Tuple[int, int]]:
        """
        基于时序关系创建边
        """
        edges = []
        
        if self.patient_id_col not in df.columns:
            return edges
        
        # 按患者分组，按时间排序
        for patient_id, group in df.groupby(self.patient_id_col):
            if len(group) > 1:
                # 按索引排序（假设索引反映时间顺序）
                sorted_indices = group.index.tolist()
                
                # 创建时序边（前一次透析 -> 当前透析）
                for i in range(len(sorted_indices) - 1):
                    current_idx = df.index.get_loc(sorted_indices[i])
                    next_idx = df.index.get_loc(sorted_indices[i + 1])
                    edges.append((current_idx, next_idx))
        
        return edges
    
    def build_graph(self, df: pd.DataFrame, target_col: str, fit: bool = True) -> Data:
        """
        构建图数据
        
        Args:
            df: 输入数据框
            target_col: 目标变量列名
            fit: 是否拟合编码器和缩放器
            
        Returns:
            torch_geometric.data.Data: 图数据对象
        """
        # 数据预处理
        df_processed = self._encode_categorical_features(df, fit=fit)
        df_processed = self._scale_numerical_features(df_processed, fit=fit)
        
        # 准备节点特征
        feature_cols = self.categorical_features + self.numerical_features
        available_feature_cols = [col for col in feature_cols if col in df_processed.columns]
        
        if not available_feature_cols:
            # 如果没有指定特征，使用所有数值列（除了ID和目标列）
            exclude_cols = [self.patient_id_col, self.session_id_col, target_col]
            available_feature_cols = [col for col in df_processed.select_dtypes(include=[np.number]).columns 
                                    if col not in exclude_cols]
        
        # 节点特征矩阵
        node_features = torch.FloatTensor(df_processed[available_feature_cols].values)
        
        # 目标标签
        y = torch.LongTensor(df_processed[target_col].values)
        
        # 创建边
        edges = []
        
        # 1. 患者相似性边
        similarity_edges = self._create_patient_similarity_edges(df_processed)
        edges.extend(similarity_edges)
        
        # 2. 时序边
        temporal_edges = self._create_temporal_edges(df_processed)
        edges.extend(temporal_edges)
        
        # 3. 如果没有边，创建全连接图（小规模数据）
        if not edges and len(df_processed) <= 100:
            for i in range(len(df_processed)):
                for j in range(len(df_processed)):
                    if i != j:
                        edges.append((i, j))
        
        # 转换为张量
        if edges:
            edge_index = torch.LongTensor(edges).t().contiguous()
        else:
            # 创建自环
            edge_index = torch.LongTensor([[i, i] for i in range(len(df_processed))]).t().contiguous()
        
        # 创建图数据对象
        data = Data(
            x=node_features,
            edge_index=edge_index,
            y=y
        )
        
        # 保存图统计信息
        if fit:
            self.graph_stats = {
                'num_nodes': data.x.size(0),
                'num_edges': data.edge_index.size(1),
                'num_features': data.x.size(1),
                'num_classes': len(torch.unique(y)),
                'feature_names': available_feature_cols
            }
        
        return data

class MultiHeadGATLayer(nn.Module):
    """
    多头图注意力层
    """
    
    def __init__(self, in_features: int, out_features: int, num_heads: int = 4, dropout: float = 0.1):
        super(MultiHeadGATLayer, self).__init__()
        self.num_heads = num_heads
        self.out_features = out_features
        
        # 多头注意力
        self.gat_layers = nn.ModuleList([
            GATConv(in_features, out_features // num_heads, dropout=dropout, add_self_loops=True)
            for _ in range(num_heads)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_features)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # 多头注意力
        head_outputs = []
        for gat_layer in self.gat_layers:
            head_output = gat_layer(x, edge_index)
            head_outputs.append(head_output)
        
        # 拼接多头输出
        x = torch.cat(head_outputs, dim=-1)
        x = self.dropout(x)
        x = self.layer_norm(x)
        
        return x

class DialysisGNN(nn.Module):
    """
    透析患者风险预测图神经网络
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 num_heads: int = 4,
                 num_classes: int = 2,
                 dropout: float = 0.1,
                 use_attention: bool = True,
                 use_residual: bool = True):
        super(DialysisGNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.use_attention = use_attention
        self.use_residual = use_residual
        
        # 输入投影层
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # 图卷积层
        self.gnn_layers = nn.ModuleList()
        
        for i in range(num_layers):
            if use_attention:
                layer = MultiHeadGATLayer(
                    in_features=hidden_dim,
                    out_features=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout
                )
            else:
                layer = GCNConv(hidden_dim, hidden_dim)
            
            self.gnn_layers.append(layer)
        
        # 全局池化
        self.global_pool = global_mean_pool
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, num_classes)
        )
        
        # 注意力权重存储（用于可解释性）
        self.attention_weights = None
    
    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, getattr(data, 'batch', None)
        
        # 输入投影
        x = self.input_projection(x)
        x = F.relu(x)
        
        # 图卷积层
        for i, layer in enumerate(self.gnn_layers):
            residual = x if self.use_residual else None
            
            if self.use_attention:
                x = layer(x, edge_index)
            else:
                x = layer(x, edge_index)
                x = F.relu(x)
            
            # 残差连接
            if residual is not None and x.size() == residual.size():
                x = x + residual
        
        # 全局池化
        if batch is not None:
            x = self.global_pool(x, batch)
        else:
            # 如果没有batch信息，直接对每个节点进行分类
            # 不进行全局池化，保持节点级别的预测
            pass
        
        # 分类
        logits = self.classifier(x)
        
        return logits
    
    def get_node_embeddings(self, data: Data) -> torch.Tensor:
        """
        获取节点嵌入（用于可视化和分析）
        """
        x, edge_index = data.x, data.edge_index
        
        # 输入投影
        x = self.input_projection(x)
        x = F.relu(x)
        
        # 图卷积层
        for layer in self.gnn_layers:
            if self.use_attention:
                x = layer(x, edge_index)
            else:
                x = layer(x, edge_index)
                x = F.relu(x)
        
        return x

class DialysisGNNClassifier(BaseEstimator, ClassifierMixin):
    """
    透析患者风险预测图神经网络分类器
    
    该分类器将表格数据转换为图结构，并使用图神经网络进行预测
    """
    
    def __init__(self,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 learning_rate: float = 0.001,
                 epochs: int = 100,
                 batch_size: int = 32,
                 early_stopping_patience: int = 10,
                 use_attention: bool = True,
                 use_residual: bool = True,
                 categorical_features: List[str] = None,
                 numerical_features: List[str] = None,
                 patient_id_col: str = 'patient_id',
                 random_state: Optional[int] = None,
                 verbose: bool = True):
        """
        初始化图神经网络分类器
        
        Args:
            hidden_dim: 隐藏层维度
            num_layers: GNN层数
            num_heads: 注意力头数
            dropout: Dropout率
            learning_rate: 学习率
            epochs: 训练轮数
            batch_size: 批次大小
            early_stopping_patience: 早停耐心值
            use_attention: 是否使用注意力机制
            use_residual: 是否使用残差连接
            categorical_features: 分类特征列表
            numerical_features: 数值特征列表
            patient_id_col: 患者ID列名
            random_state: 随机种子
            verbose: 是否显示训练过程
        """
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.use_attention = use_attention
        self.use_residual = use_residual
        self.categorical_features = categorical_features or []
        self.numerical_features = numerical_features or []
        self.patient_id_col = patient_id_col
        self.random_state = random_state
        self.verbose = verbose
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 设置随机种子
        if random_state is not None:
            np.random.seed(random_state)
            torch.manual_seed(random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(random_state)
        
        # 初始化组件
        self.graph_builder = None
        self.model = None
        self.classes_ = None
        self.n_features_in_ = None
        self.training_history_ = {'loss': [], 'accuracy': []}
    
    def _prepare_features(self, X: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        自动识别分类和数值特征
        """
        if not self.categorical_features and not self.numerical_features:
            # 自动识别特征类型
            categorical_cols = []
            numerical_cols = []
            
            for col in X.columns:
                if col == self.patient_id_col:
                    continue
                
                if X[col].dtype == 'object' or X[col].nunique() <= 10:
                    categorical_cols.append(col)
                else:
                    numerical_cols.append(col)
            
            return categorical_cols, numerical_cols
        else:
            return self.categorical_features, self.numerical_features
    
    def _build_sparse_graph(self, X_with_target: pd.DataFrame, target_col: str):
        """
        构建稀疏图以减少内存使用
        
        Args:
            X_with_target: 包含目标列的特征数据
            target_col: 目标列名
            
        Returns:
            graph_data: 稀疏图数据
        """
        from torch_geometric.data import Data
        import torch
        from sklearn.neighbors import NearestNeighbors
        from sklearn.preprocessing import StandardScaler
        
        # 提取特征和标签
        features = X_with_target.drop(columns=[target_col])
        labels = X_with_target[target_col].values
        
        # 标准化特征
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features.select_dtypes(include=[np.number]))
        
        # 使用KNN构建稀疏图（只连接最近的k个邻居）
        k = min(10, len(features_scaled) - 1)  # 最多连接10个邻居
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(features_scaled)
        distances, indices = nbrs.kneighbors(features_scaled)
        
        # 构建边索引
        edge_list = []
        for i in range(len(indices)):
            for j in range(1, len(indices[i])):  # 跳过自己（索引0）
                neighbor_idx = indices[i][j]
                # 只添加距离小于阈值的边
                if distances[i][j] < np.percentile(distances[:, 1:], 75):  # 使用75分位数作为阈值
                    edge_list.append([i, neighbor_idx])
                    edge_list.append([neighbor_idx, i])  # 无向图
        
        # 转换为张量
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        else:
            # 如果没有边，创建一个空的边索引
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        # 节点特征
        x = torch.tensor(features_scaled, dtype=torch.float)
        
        # 节点标签
        y = torch.tensor(labels, dtype=torch.long)
        
        # 创建图数据
        graph_data = Data(x=x, edge_index=edge_index, y=y)
        
        return graph_data
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: np.ndarray, target_col: str = 'target') -> 'DialysisGNNClassifier':
        """
        训练图神经网络模型
        
        Args:
            X: 特征数据
            y: 目标标签
            target_col: 目标列名（用于图构建）
            
        Returns:
            self: 训练后的模型
        """
        try:
            # 转换为DataFrame
            if isinstance(X, np.ndarray):
                X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
            
            # 检查数据大小，如果太大则进行采样
            if len(X) > 50000:
                if self.verbose:
                    print(f"数据量过大 ({len(X)} 样本)，进行采样到 50000 样本")
                from sklearn.model_selection import train_test_split
                X, _, y, _ = train_test_split(X, y, train_size=50000, random_state=self.random_state, stratify=y)
            
            # 添加目标列
            X_with_target = X.copy()
            X_with_target[target_col] = y
            
            # 验证输入
            self.classes_ = unique_labels(y)
            self.n_features_in_ = X.shape[1]
            
            if self.verbose:
                print(f"开始训练透析GNN模型...")
                print(f"训练样本数: {X.shape[0]}, 特征数: {X.shape[1]}")
                print(f"类别数: {len(self.classes_)}, 类别: {self.classes_}")
                print(f"使用设备: {self.device}")
            
            # 准备特征
            categorical_features, numerical_features = self._prepare_features(X)
            
            # 初始化图构建器
            self.graph_builder = DialysisGraphBuilder(
                patient_id_col=self.patient_id_col,
                categorical_features=categorical_features,
                numerical_features=numerical_features
            )
            
            # 构建图
            graph_data = self.graph_builder.build_graph(X_with_target, target_col, fit=True)
            graph_data = graph_data.to(self.device)
            
            # 检查图的大小，如果边太多则限制
            if graph_data.edge_index.size(1) > 1000000:  # 超过100万条边
                if self.verbose:
                    print(f"图规模过大 ({graph_data.edge_index.size(1)} 条边)，限制图连接")
                # 使用更高的相似性阈值来减少边数
                graph_data = self._build_sparse_graph(X_with_target, target_col)
                graph_data = graph_data.to(self.device)
            
            # 初始化模型
            input_dim = graph_data.x.size(1)
            num_classes = len(self.classes_)
            
            self.model = DialysisGNN(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                num_classes=num_classes,
                dropout=self.dropout,
                use_attention=self.use_attention,
                use_residual=self.use_residual
            ).to(self.device)
            
            # 优化器和损失函数
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            criterion = nn.CrossEntropyLoss()
            
            # 训练循环
            self.model.train()
            best_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(self.epochs):
                try:
                    optimizer.zero_grad()
                    
                    # 前向传播
                    logits = self.model(graph_data)
                    loss = criterion(logits, graph_data.y)
                    
                    # 反向传播
                    loss.backward()
                    optimizer.step()
                    
                    # 计算准确率
                    with torch.no_grad():
                        pred = torch.argmax(logits, dim=1)
                        accuracy = (pred == graph_data.y).float().mean().item()
                    
                    # 记录训练历史
                    self.training_history_['loss'].append(loss.item())
                    self.training_history_['accuracy'].append(accuracy)
                    
                    # 早停检查
                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= self.early_stopping_patience:
                        if self.verbose:
                            print(f"早停于第 {epoch + 1} 轮")
                        break
                    
                    # 打印进度
                    if self.verbose and (epoch + 1) % 10 == 0:
                        print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")
                    
                    # 定期清理GPU内存
                    if self.device.type == 'cuda' and epoch % 20 == 0:
                        torch.cuda.empty_cache()
                        
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        if self.verbose:
                            print(f"GPU内存不足，尝试清理内存并继续训练...")
                        torch.cuda.empty_cache()
                        # 减小批次大小或其他策略
                        continue
                    else:
                        raise e
            
            if self.verbose:
                print(f"训练完成! 最终损失: {loss.item():.4f}, 最终准确率: {accuracy:.4f}")
            
            # 训练完成后清理内存
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            return self
            
        except Exception as e:
            # 发生异常时清理资源
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            if self.verbose:
                print(f"训练过程中发生错误: {e}")
            raise e
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        预测类别
        
        Args:
            X: 特征数据
            
        Returns:
            预测的类别标签
        """
        check_is_fitted(self)
        
        # 转换为DataFrame
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        
        # 添加虚拟目标列
        X_with_target = X.copy()
        X_with_target['target'] = 0  # 虚拟值
        
        # 构建图
        graph_data = self.graph_builder.build_graph(X_with_target, 'target', fit=False)
        graph_data = graph_data.to(self.device)
        
        # 预测
        self.model.eval()
        with torch.no_grad():
            logits = self.model(graph_data)
            pred = torch.argmax(logits, dim=1)
        
        return pred.cpu().numpy()
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        预测类别概率
        
        Args:
            X: 特征数据
            
        Returns:
            预测的类别概率
        """
        check_is_fitted(self)
        
        # 转换为DataFrame
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        
        # 添加虚拟目标列
        X_with_target = X.copy()
        X_with_target['target'] = 0  # 虚拟值
        
        # 构建图
        graph_data = self.graph_builder.build_graph(X_with_target, 'target', fit=False)
        graph_data = graph_data.to(self.device)
        
        # 预测
        self.model.eval()
        with torch.no_grad():
            logits = self.model(graph_data)
            proba = F.softmax(logits, dim=1)
        
        return proba.cpu().numpy()
    
    def get_node_embeddings(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        获取节点嵌入表示
        
        Args:
            X: 特征数据
            
        Returns:
            节点嵌入矩阵
        """
        check_is_fitted(self)
        
        # 转换为DataFrame
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        
        # 添加虚拟目标列
        X_with_target = X.copy()
        X_with_target['target'] = 0  # 虚拟值
        
        # 构建图
        graph_data = self.graph_builder.build_graph(X_with_target, 'target', fit=False)
        graph_data = graph_data.to(self.device)
        
        # 获取嵌入
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model.get_node_embeddings(graph_data)
        
        return embeddings.cpu().numpy()
    
    def plot_training_history(self, figsize: Tuple[int, int] = (12, 4)):
        """
        绘制训练历史
        """
        check_is_fitted(self)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # 损失曲线
        ax1.plot(self.training_history_['loss'])
        ax1.set_title('训练损失')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        # 准确率曲线
        ax2.plot(self.training_history_['accuracy'])
        ax2.set_title('训练准确率')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def visualize_graph(self, X: Union[pd.DataFrame, np.ndarray], 
                       target_col: str = 'target', 
                       y: np.ndarray = None,
                       figsize: Tuple[int, int] = (12, 8),
                       node_size: int = 50):
        """
        可视化图结构
        
        Args:
            X: 特征数据
            target_col: 目标列名
            y: 目标标签（用于节点着色）
            figsize: 图像大小
            node_size: 节点大小
        """
        # 转换为DataFrame
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        
        # 添加目标列
        X_with_target = X.copy()
        if y is not None:
            X_with_target[target_col] = y
        else:
            X_with_target[target_col] = 0
        
        # 构建图
        if hasattr(self, 'graph_builder') and self.graph_builder is not None:
            graph_data = self.graph_builder.build_graph(X_with_target, target_col, fit=False)
        else:
            # 临时构建器
            temp_builder = DialysisGraphBuilder()
            graph_data = temp_builder.build_graph(X_with_target, target_col, fit=True)
        
        # 转换为NetworkX图
        G = to_networkx(graph_data, to_undirected=True)
        
        # 绘制图
        plt.figure(figsize=figsize)
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # 节点颜色
        if y is not None:
            node_colors = y
            cmap = plt.cm.RdYlBu
        else:
            node_colors = 'lightblue'
            cmap = None
        
        nx.draw(G, pos, 
                node_color=node_colors,
                node_size=node_size,
                cmap=cmap,
                with_labels=False,
                edge_color='gray',
                alpha=0.7)
        
        plt.title('透析患者关系图')
        if y is not None:
            plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), label='风险等级')
        plt.show()
    
    def save_model(self, filepath: str) -> None:
        """
        保存模型
        
        Args:
            filepath: 保存路径
        """
        check_is_fitted(self)
        
        try:
            # 将模型移到CPU以节省GPU内存
            if self.model and self.device.type == 'cuda':
                model_state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}
            else:
                model_state_dict = self.model.state_dict() if self.model else None
            
            model_data = {
                'model_state_dict': model_state_dict,
                'graph_builder': self.graph_builder,
                'classes_': self.classes_,
                'n_features_in_': self.n_features_in_,
                'training_history_': self.training_history_,
                'hyperparameters': {
                    'hidden_dim': self.hidden_dim,
                    'num_layers': self.num_layers,
                    'num_heads': self.num_heads,
                    'dropout': self.dropout,
                    'learning_rate': self.learning_rate,
                    'epochs': self.epochs,
                    'early_stopping_patience': self.early_stopping_patience,
                    'use_attention': self.use_attention,
                    'use_residual': self.use_residual,
                    'categorical_features': self.categorical_features,
                    'numerical_features': self.numerical_features,
                    'patient_id_col': self.patient_id_col,
                    'random_state': self.random_state
                }
            }
            
            joblib.dump(model_data, filepath)
            
            # 清理内存
            del model_data
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            if self.verbose:
                print(f"模型已保存到: {filepath}")
                
        except Exception as e:
            if self.verbose:
                print(f"保存模型时发生错误: {e}")
            # 清理内存
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            raise e
    
    def load_model(self, filepath: str) -> 'DialysisGNNClassifier':
        """
        加载模型
        
        Args:
            filepath: 模型文件路径
            
        Returns:
            self: 加载后的模型
        """
        try:
            model_data = joblib.load(filepath)
            
            # 恢复超参数
            hyperparams = model_data['hyperparameters']
            for key, value in hyperparams.items():
                setattr(self, key, value)
            
            # 恢复其他属性
            self.graph_builder = model_data['graph_builder']
            self.classes_ = model_data['classes_']
            self.n_features_in_ = model_data['n_features_in_']
            self.training_history_ = model_data['training_history_']
            
            # 重建模型结构（需要知道输入维度）
            if model_data['model_state_dict'] is not None:
                # 从图构建器获取输入维度
                if hasattr(self.graph_builder, 'graph_stats') and 'feature_names' in self.graph_builder.graph_stats:
                    input_dim = len(self.graph_builder.graph_stats['feature_names'])
                else:
                    # 从状态字典推断输入维度
                    state_dict = model_data['model_state_dict']
                    input_dim = state_dict['input_projection.weight'].size(1)
                
                num_classes = len(self.classes_)
                
                self.model = DialysisGNN(
                    input_dim=input_dim,
                    hidden_dim=self.hidden_dim,
                    num_layers=self.num_layers,
                    num_heads=self.num_heads,
                    num_classes=num_classes,
                    dropout=self.dropout,
                    use_attention=self.use_attention,
                    use_residual=self.use_residual
                ).to(self.device)
                
                self.model.load_state_dict(model_data['model_state_dict'])
            
            # 清理内存
            del model_data
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            if self.verbose:
                print(f"模型已从 {filepath} 加载")
            
            return self
            
        except Exception as e:
            if self.verbose:
                print(f"加载模型时发生错误: {e}")
            # 清理内存
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            raise e

def evaluate_dialysis_gnn(model: DialysisGNNClassifier, 
                         X_test: Union[pd.DataFrame, np.ndarray], 
                         y_test: np.ndarray,
                         class_names: List[str] = None) -> Dict[str, Any]:
    """
    评估透析GNN模型性能
    
    Args:
        model: 训练好的模型
        X_test: 测试特征
        y_test: 测试标签
        class_names: 类别名称
        
    Returns:
        评估结果字典
    """
    # 预测
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # 计算指标
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
    
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }
    
    # AUC（仅适用于二分类）
    if len(np.unique(y_test)) == 2:
        auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        results['auc'] = auc
    
    # 分类报告
    if class_names is None:
        class_names = [f'Class_{i}' for i in range(len(np.unique(y_test)))]
    
    print("=== 透析GNN模型评估结果 ===")
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    if 'auc' in results:
        print(f"AUC: {results['auc']:.4f}")
    
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.show()
    
    return results

if __name__ == "__main__":
    import gc
    import psutil
    import os
    
    def print_memory_usage(stage=""):
        """打印当前内存使用情况"""
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            print(f"{stage} - 内存使用: {memory_info.rss / 1024 / 1024:.2f} MB")
        except:
            print(f"{stage} - 无法获取内存信息")
    
    try:
        print_memory_usage("开始")
        
        # 示例用法
        print("透析患者低血压风险预测图神经网络模型")
        print("该模型将在实际数据上进行训练和测试")
        
        # 模拟数据示例（减少数据量以避免内存问题）
        np.random.seed(42)
        n_samples = 150  # 减少样本数
        n_features = 12  # 减少特征数
        
        # 创建模拟透析数据
        X = pd.DataFrame({
            'patient_id': range(n_samples),
            '年龄': np.random.normal(65, 15, n_samples),
            '性别': np.random.choice([0, 1], n_samples),
            '透析龄': np.random.exponential(3, n_samples),
            '干体重': np.random.normal(60, 10, n_samples),
            '透前收缩压': np.random.normal(140, 20, n_samples),
            '透前舒张压': np.random.normal(80, 10, n_samples),
            '超滤率': np.random.normal(8, 3, n_samples),
            '血流速': np.random.normal(250, 50, n_samples),
            '历史低血压次数': np.random.poisson(2, n_samples),
            '瘘管类型': np.random.choice([0, 1, 2], n_samples),
            '透析时长': np.random.normal(4, 0.5, n_samples)
        })
        
        # 创建目标变量（低血压风险）
        # 基于一些特征的组合来生成标签
        risk_score = (
            (X['年龄'] > 70).astype(int) +
            (X['透前收缩压'] < 120).astype(int) +
            (X['超滤率'] > 10).astype(int) +
            (X['历史低血压次数'] > 3).astype(int)
        )
        y = (risk_score >= 2).astype(int)
        
        print(f"\n数据概览:")
        print(f"样本数: {len(X)}")
        print(f"特征数: {X.shape[1]}")
        print(f"低血压风险分布: {np.bincount(y)}")
        
        print_memory_usage("数据准备完成")
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n训练集大小: {len(X_train)}")
        print(f"测试集大小: {len(X_test)}")
        
        # 定义特征类型
        categorical_features = ['性别', '瘘管类型']
        numerical_features = ['年龄', '透析龄', '干体重', '透前收缩压', '透前舒张压', 
                             '超滤率', '血流速', '历史低血压次数', '透析时长']
        
        # 创建和训练模型（使用更小的参数）
        model = DialysisGNNClassifier(
            hidden_dim=32,  # 减少隐藏层维度
            num_layers=2,
            num_heads=2,    # 减少注意力头数
            dropout=0.1,
            learning_rate=0.01,
            epochs=50,      # 减少训练轮数
            categorical_features=categorical_features,
            numerical_features=numerical_features,
            patient_id_col='patient_id',
            random_state=42,
            verbose=True
        )
        
        print_memory_usage("模型创建完成")
        
        print("\n开始训练模型...")
        model.fit(X_train, y_train)
        
        print_memory_usage("模型训练完成")
        
        # 评估模型
        print("\n评估模型性能...")
        results = evaluate_dialysis_gnn(
            model, X_test, y_test, 
            class_names=['低风险', '高风险']
        )
        
        print_memory_usage("模型评估完成")
        
        # 绘制训练历史
        try:
            model.plot_training_history()
        except Exception as e:
            print(f"绘制训练历史时出错: {e}")
        
        # 可视化图结构（小样本）
        try:
            if len(X_test) <= 30:
                print("\n可视化图结构...")
                model.visualize_graph(X_test.head(20), y=y_test[:20])
        except Exception as e:
            print(f"可视化图结构时出错: {e}")
        
        # 保存模型和结果
        try:
            save_gnn_results(model, results, X_test, y_test, 
                           target='透中低血压', kinds='GNN')
            print("\n模型和结果已保存")
        except Exception as e:
            print(f"保存模型和结果时出错: {e}")
        
        print("\n模型训练和评估完成!")
        
    except Exception as e:
        print(f"运行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 显式清理资源
        try:
            # 删除大型变量
            if 'model' in locals():
                del model
            if 'X_train' in locals():
                del X_train, X_test, y_train, y_test
            if 'X' in locals():
                del X, y
            if 'results' in locals():
                del results
            
            # 强制垃圾回收
            gc.collect()
            
            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print_memory_usage("清理完成")
            
        except Exception as e:
            print(f"清理资源时出错: {e}")
        
        print("程序结束")

if __name__ == '__main__':
    main()