#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化统一模型评估与可视化模块
专为透析患者临床数据分析设计的轻量级评估工具

cht
2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
import warnings
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List, Union

warnings.filterwarnings("ignore")

# 移除模型类导入以避免循环导入
# 模型类将在需要时动态导入
MODEL_IMPORTS_AVAILABLE = False

# 设置中文字体和样式
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")
sns.set_palette("husl")


class UnifiedModelEvaluator:
    """
    简化统一模型评估器
    提供核心的模型评估指标计算和基础可视化功能
    """

    def __init__(self, task_type="classification", save_dir="./evaluation_results"):
        """
        初始化评估器

        Parameters:
        -----------
        task_type : str
            任务类型，'classification' 或 'regression'
        save_dir : str
            结果保存目录
        """
        self.task_type = task_type
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        print(f"初始化简化模型评估器 - 任务类型: {task_type}")
        print(f"结果保存目录: {self.save_dir}")
        if MODEL_IMPORTS_AVAILABLE:
            print("✓ 所有模型类已导入")

    def evaluate_model(
        self,
        model,
        X_test,
        y_test,
        model_name="Model",
        create_visualizations=True,
        save_results=True,
    ):
        """
        评估模型性能

        Parameters:
        -----------
        model : sklearn estimator
            训练好的模型
        X_test : array-like
            测试特征
        y_test : array-like
            测试标签
        model_name : str
            模型名称
        create_visualizations : bool
            是否创建可视化图表
        save_results : bool
            是否保存结果

        Returns:
        --------
        dict : 评估结果字典
        """
        print(f"\n=== 开始评估模型: {model_name} ===")

        # 预测
        y_pred = model.predict(X_test)
        y_pred_proba = None
        if self.task_type == "classification" and hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test)

        # 计算评估指标
        if self.task_type == "classification":
            metrics = self._calculate_classification_metrics(
                y_test, y_pred, y_pred_proba
            )
        else:
            metrics = self._calculate_regression_metrics(y_test, y_pred)

        # 创建可视化
        if create_visualizations:
            self._create_visualizations(y_test, y_pred, y_pred_proba, model_name)

        # 保存结果
        if save_results:
            self._save_evaluation_results(metrics, model_name)

        # 打印结果摘要
        self._print_evaluation_summary(metrics, model_name)

        return metrics

    def _calculate_classification_metrics(self, y_true, y_pred, y_pred_proba=None):
        """
        计算简化的分类任务评估指标

        Parameters:
        -----------
        y_true : array-like
            真实标签
        y_pred : array-like
            预测标签
        y_pred_proba : array-like, optional
            预测概率

        Returns:
        --------
        dict : 评估指标字典
        """
        metrics = {}

        try:
            # 基本分类指标
            metrics["accuracy"] = accuracy_score(y_true, y_pred)
            metrics["precision"] = precision_score(
                y_true, y_pred, average="weighted", zero_division=0
            )
            metrics["recall"] = recall_score(
                y_true, y_pred, average="weighted", zero_division=0
            )
            metrics["f1_score"] = f1_score(
                y_true, y_pred, average="weighted", zero_division=0
            )

            # 混淆矩阵
            metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)

            # 如果有预测概率且为二分类
            if y_pred_proba is not None and len(np.unique(y_true)) == 2:
                # 使用正类概率
                if y_pred_proba.ndim > 1:
                    prob_pos = y_pred_proba[:, 1]
                else:
                    prob_pos = y_pred_proba

                metrics["roc_auc"] = roc_auc_score(y_true, prob_pos)
                metrics["pr_auc"] = average_precision_score(y_true, prob_pos)

        except Exception as e:
            print(f"计算分类指标时出错: {e}")

        return metrics

    def _calculate_regression_metrics(self, y_true, y_pred):
        """
        计算回归任务的评估指标

        Parameters:
        -----------
        y_true : array-like
            真实值
        y_pred : array-like
            预测值

        Returns:
        --------
        dict : 评估指标字典
        """
        metrics = {}

        try:
            # 基本回归指标
            metrics["mse"] = mean_squared_error(y_true, y_pred)
            metrics["rmse"] = np.sqrt(metrics["mse"])
            metrics["mae"] = mean_absolute_error(y_true, y_pred)
            metrics["r2_score"] = r2_score(y_true, y_pred)

        except Exception as e:
            print(f"计算回归指标时出错: {e}")

        return metrics

    def _create_visualizations(self, y_true, y_pred, y_pred_proba, model_name):
        """
        创建基础可视化图表

        Parameters:
        -----------
        y_true : array-like
            真实标签/值
        y_pred : array-like
            预测标签/值
        y_pred_proba : array-like, optional
            预测概率
        model_name : str
            模型名称
        """
        print(f"🎨 创建基础可视化图表...")

        try:
            if self.task_type == "classification":
                self._create_confusion_matrix(y_true, y_pred, model_name)
                if y_pred_proba is not None and len(np.unique(y_true)) == 2:
                    self._create_roc_curve(y_true, y_pred_proba, model_name)
            else:
                self._create_regression_plots(y_true, y_pred, model_name)

            print(f"✅ 可视化图表已保存")

        except Exception as e:
            print(f"创建可视化时出错: {e}")

    def _create_confusion_matrix(self, y_true, y_pred, model_name):
        """
        创建混淆矩阵图
        """
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["正常", "低血压"],
            yticklabels=["正常", "低血压"],
        )
        plt.title(f"{model_name} - 混淆矩阵")
        plt.ylabel("真实标签")
        plt.xlabel("预测标签")
        plt.tight_layout()

        if self.save_dir:
            plt.savefig(
                self.save_dir / f"{model_name}_confusion_matrix.png",
                dpi=300,
                bbox_inches="tight",
            )
        plt.show()

    def _create_roc_curve(self, y_true, y_pred_proba, model_name):
        """
        创建ROC曲线图
        """
        plt.figure(figsize=(8, 6))

        # 使用正类概率
        if y_pred_proba.ndim > 1:
            prob_pos = y_pred_proba[:, 1]
        else:
            prob_pos = y_pred_proba

        fpr, tpr, _ = roc_curve(y_true, prob_pos)
        auc_score = roc_auc_score(y_true, prob_pos)

        plt.plot(fpr, tpr, linewidth=2, label=f"ROC曲线 (AUC = {auc_score:.3f})")
        plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="随机分类器")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("假阳性率 (1-特异性)")
        plt.ylabel("真阳性率 (敏感性)")
        plt.title(f"{model_name} - ROC曲线")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if self.save_dir:
            plt.savefig(
                self.save_dir / f"{model_name}_roc_curve.png",
                dpi=300,
                bbox_inches="tight",
            )
        plt.show()

    def _create_regression_plots(self, y_true, y_pred, model_name):
        """
        创建回归任务的可视化图表
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # 预测值 vs 真实值
        ax1.scatter(y_true, y_pred, alpha=0.6)
        ax1.plot(
            [y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", lw=2
        )
        ax1.set_xlabel("真实值")
        ax1.set_ylabel("预测值")
        ax1.set_title(f"{model_name} - 预测值 vs 真实值")
        ax1.grid(True, alpha=0.3)

        # 残差图
        residuals = y_true - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.6)
        ax2.axhline(y=0, color="r", linestyle="--")
        ax2.set_xlabel("预测值")
        ax2.set_ylabel("残差")
        ax2.set_title(f"{model_name} - 残差图")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if self.save_dir:
            plt.savefig(
                self.save_dir / f"{model_name}_regression_plots.png",
                dpi=300,
                bbox_inches="tight",
            )
        plt.show()

    def _print_evaluation_summary(self, metrics, model_name):
        """
        打印评估摘要

        Parameters:
        -----------
        metrics : dict
            评估指标字典
        model_name : str
            模型名称
        """
        print(f"\n=== {model_name} 评估摘要 ===")

        if self.task_type == "classification":
            print(f"📊 分类性能指标:")
            print(f"  准确率: {metrics.get('accuracy', 'N/A'):.4f}")
            print(f"  精确率: {metrics.get('precision', 'N/A'):.4f}")
            print(f"  召回率: {metrics.get('recall', 'N/A'):.4f}")
            print(f"  F1分数: {metrics.get('f1_score', 'N/A'):.4f}")

            if "roc_auc" in metrics:
                print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
            if "pr_auc" in metrics:
                print(f"  PR AUC: {metrics['pr_auc']:.4f}")

        else:  # 回归任务
            print(f"📊 回归性能指标:")
            print(f"  RMSE: {metrics.get('rmse', 'N/A'):.4f}")
            print(f"  MAE: {metrics.get('mae', 'N/A'):.4f}")
            print(f"  R²: {metrics.get('r2_score', 'N/A'):.4f}")

    def _save_evaluation_results(self, metrics, model_name):
        """
        保存评估结果到JSON文件

        Parameters:
        -----------
        metrics : dict
            评估指标字典
        model_name : str
            模型名称
        """
        try:
            # 转换numpy数组为列表以便JSON序列化
            serializable_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, np.ndarray):
                    serializable_metrics[key] = value.tolist()
                elif isinstance(value, (np.integer, np.floating)):
                    serializable_metrics[key] = float(value)
                else:
                    serializable_metrics[key] = value

            # 添加元数据
            result_data = {
                "model_name": model_name,
                "task_type": self.task_type,
                "evaluation_time": datetime.now().isoformat(),
                "metrics": serializable_metrics,
            }

            # 保存到文件
            filename = f"{model_name}_evaluation_results.json"
            filepath = self.save_dir / filename

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)

            print(f"📁 评估结果已保存到: {filepath}")

        except Exception as e:
            print(f"保存评估结果时出错: {e}")


def evaluate_model_simple(
    model,
    X_test,
    y_test,
    model_name="模型",
    task_type="classification",
    save_dir="./evaluation_results",
    create_visualizations=True,
    save_results=True,
):
    """
    快速简化评估模型的便捷函数

    Parameters:
    -----------
    model : sklearn estimator
        训练好的模型
    X_test : array-like
        测试特征
    y_test : array-like
        测试标签
    model_name : str
        模型名称
    task_type : str
        任务类型
    save_dir : str
        保存目录
    create_visualizations : bool
        是否创建可视化
    save_results : bool
        是否保存结果

    Returns:
    --------
    dict : 评估结果
    """
    evaluator = UnifiedModelEvaluator(task_type=task_type, save_dir=save_dir)

    return evaluator.evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        model_name=model_name,
        create_visualizations=create_visualizations,
        save_results=save_results,
    )


if __name__ == "__main__":
    # 简单测试示例
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    print("=== 简化统一模型评估器测试 ===")

    # 生成测试数据
    X, y = make_classification(
        n_samples=1000, n_features=20, n_classes=2, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # 训练模型
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # 评估模型
    results = evaluate_model_simple(
        model=model,
        X_test=X_test,
        y_test=y_test,
        model_name="RandomForest测试",
        task_type="classification",
        create_visualizations=True,
        save_results=True,
    )

    print("\n测试完成!")
