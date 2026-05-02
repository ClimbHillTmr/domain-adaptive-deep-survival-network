"""
Generate publication-quality figures for the DA-DSN paper.
All figures are saved as PDF files in the figures/ directory.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.gridspec import GridSpec
import os
import logging
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

# Set publication-quality defaults
matplotlib.rcParams.update({
    "font.size": 10,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "axes.linewidth": 1.0,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})


def setup_figure_dir(base_dir: str = "figures") -> str:
    """Create and return figures directory."""
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def plot_calibration_curve(
    bin_centers,
    observed_probs,
    predicted_probs,
    model_name: str,
    save_dir: str = "figures",
    filename: str = None,
):
    """
    Figure: Calibration plot with ideal diagonal.
    Shows how well predicted probabilities match observed event rates.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    
    # Ideal calibration line
    ax.plot([0, 1], [0, 1], "k--", label="Ideal calibration", linewidth=1.5)
    
    # Model calibration
    ax.plot(
        predicted_probs,
        observed_probs,
        "o-",
        label=model_name,
        linewidth=2,
        markersize=6,
        color="#1f77b4",
    )
    
    ax.set_xlabel("Predicted Probability", fontsize=11)
    ax.set_ylabel("Observed Event Rate", fontsize=11)
    ax.set_title(f"Calibration Plot: {model_name}", fontsize=12)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.grid(True, alpha=0.3, linestyle="--")
    
    if filename is None:
        filename = f"calibration_{model_name.lower().replace(' ', '_')}.pdf"
    
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close()
    
    log.info(f"Calibration plot saved to {save_path}")
    return save_path


def plot_decision_curve(
    thresholds,
    net_benefit_model,
    net_benefit_all,
    net_benefit_none,
    model_name: str,
    save_dir: str = "figures",
    filename: str = None,
):
    """
    Figure: Decision Curve Analysis (DCA).
    Shows net benefit across different threshold probabilities.
    """
    fig, ax = plt.subplots(figsize=(6, 4.5))
    
    ax.plot(
        thresholds,
        net_benefit_none,
        "k--",
        label="Treat none",
        linewidth=1.5,
    )
    ax.plot(
        thresholds,
        net_benefit_all,
        "k:",
        label="Treat all",
        linewidth=1.5,
    )
    ax.plot(
        thresholds,
        net_benefit_model,
        "-",
        label=model_name,
        linewidth=2.5,
        color="#1f77b4",
    )
    
    ax.set_xlabel("Threshold Probability", fontsize=11)
    ax.set_ylabel("Net Benefit", fontsize=11)
    ax.set_title(f"Decision Curve Analysis: {model_name}", fontsize=12)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")
    
    if filename is None:
        filename = f"dca_{model_name.lower().replace(' ', '_')}.pdf"
    
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close()
    
    log.info(f"DCA plot saved to {save_path}")
    return save_path


def plot_feature_importance_shap(
    shap_summary_df: pd.DataFrame,
    save_dir: str = "figures",
    filename: str = "shap_feature_importance.pdf",
    color: str = "#2ca02c",
):
    """
    Figure: SHAP feature importance bar chart (top N features).
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    features = shap_summary_df["feature"].values[::-1]  # Reverse for horizontal bar
    importance = shap_summary_df["mean_abs_shap"].values[::-1]
    
    bars = ax.barh(range(len(features)), importance, color=color, edgecolor="none", height=0.7)
    
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=10)
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title("Feature Importance (SHAP)", fontsize=12)
    ax.grid(True, alpha=0.3, linestyle="--", axis="x")
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, importance)):
        ax.text(
            bar.get_width() + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            fontsize=9,
            color="#333333",
        )
    
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close()
    
    log.info(f"SHAP feature importance plot saved to {save_path}")
    return save_path


def plot_ablation_comparison(
    results_dict: Dict[str, Dict],
    save_dir: str = "figures",
    filename: str = "ablation_comparison.pdf",
):
    """
    Figure: Ablation study comparison with error bars.
    Shows Target C-index for each model configuration.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    model_names = list(results_dict.keys())
    means = [results_dict[k]["mean"]["target_c_index"] for k in model_names]
    stds = [results_dict[k]["std"]["target_c_index"] for k in model_names]
    
    # Color scheme
    colors = ["#ff7f0e", "#2ca02c", "#1f77b4", "#d62728", "#9467bd"]
    
    x_pos = np.arange(len(model_names))
    bars = ax.bar(
        x_pos,
        means,
        yerr=stds,
        capsize=5,
        color=colors[:len(model_names)],
        edgecolor="white",
        linewidth=1.5,
        alpha=0.85,
    )
    
    # Add value labels
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 0.01,
            f"{mean:.3f}\n±{std:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Target C-index", fontsize=11)
    ax.set_title("Ablation Study: Model Comparison", fontsize=12)
    ax.set_ylim([0.5, max(means) + max(stds) + 0.05])
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")
    
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close()
    
    log.info(f"Ablation comparison plot saved to {save_path}")
    return save_path


def plot_model_comparison_table(
    results_dict: Dict[str, Dict],
    save_dir: str = "figures",
    filename: str = "model_comparison_table.pdf",
):
    """
    Figure: Publication-quality table of model comparison results.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    
    # Prepare table data
    rows = []
    for name, res in results_dict.items():
        mean_c = res["mean"]["target_c_index"]
        std_c = res["std"]["target_c_index"]
        mean_ibs = res["mean"]["target_ibs"]
        std_ibs = res["std"]["target_ibs"]
        rows.append([name, f"{mean_c:.3f} ± {std_c:.3f}", f"{mean_ibs:.3f} ± {std_ibs:.3f}"])
    
    # Create table
    table = ax.table(
        cellText=rows,
        colLabels=["Model", "Target C-index", "Target IBS"],
        loc="center",
        cellLoc="center",
    )
    
    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Header styling
    for i in range(3):
        table[(0, i)].set_facecolor("#1f77b4")
        table[(0, i)].set_text_props(color="white", fontweight="bold")
    
    # Alternating row colors
    for i in range(1, len(rows) + 1):
        if i % 2 == 0:
            for j in range(3):
                table[(i, j)].set_facecolor("#f0f0f0")
    
    ax.set_title("Model Comparison Results", fontsize=14, fontweight="bold", pad=20)
    
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close()
    
    log.info(f"Model comparison table saved to {save_path}")
    return save_path


def generate_all_figures(
    ablation_results: Dict,
    calibration_data: Optional[Dict] = None,
    dca_data: Optional[Dict] = None,
    shap_summary: Optional[pd.DataFrame] = None,
    save_dir: str = "figures",
):
    """
    Generate all publication-quality figures in one call.
    """
    save_dir = setup_figure_dir(save_dir)
    
    figures = {}
    
    # 1. Ablation comparison
    if ablation_results:
        figures["ablation"] = plot_ablation_comparison(ablation_results, save_dir)
        figures["table"] = plot_model_comparison_table(ablation_results, save_dir)
    
    # 2. Calibration curves
    if calibration_data:
        for model_name, cal_data in calibration_data.items():
            figures[f"calibration_{model_name}"] = plot_calibration_curve(
                cal_data["bin_centers"],
                cal_data["observed_probs"],
                cal_data["predicted_probs"],
                model_name,
                save_dir,
            )
    
    # 3. Decision curves
    if dca_data:
        for model_name, dca_vals in dca_data.items():
            figures[f"dca_{model_name}"] = plot_decision_curve(
                dca_vals["thresholds"],
                dca_vals["net_benefit_model"],
                dca_vals["net_benefit_all"],
                dca_vals["net_benefit_none"],
                model_name,
                save_dir,
            )
    
    # 4. SHAP feature importance
    if shap_summary is not None:
        figures["shap"] = plot_feature_importance_shap(shap_summary, save_dir)
    
    log.info(f"All figures saved to {save_dir}/")
    return figures
