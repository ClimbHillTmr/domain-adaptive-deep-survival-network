"""
Multi-center feature shift analysis.
Computes Wasserstein distance and statistical tests between source (ShenYi) and target (FuDing) domains.
Generates radar chart for paper Figure.
"""
import pandas as pd
import numpy as np
import sys
import os
from scipy import stats
from scipy.stats import wasserstein_distance

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf
from src.data.loader import DialysisDataLoader


def compute_feature_shift():
    """Compute feature distribution differences between two centers."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    defaults = OmegaConf.load(os.path.join(project_root, "configs", "defaults.yaml"))
    config = OmegaConf.load(os.path.join(project_root, "configs", "config.yaml"))
    cfg = OmegaConf.merge(defaults, config)

    # Load raw data
    df_source = pd.read_csv(cfg.experiment.train_csv)
    df_target = pd.read_csv(cfg.experiment.external_csv)

    print("=" * 80)
    print("MULTI-CENTER FEATURE SHIFT ANALYSIS")
    print("=" * 80)
    print("Source (ShenYi): %d samples" % len(df_source))
    print("Target (FuDing): %d samples" % len(df_target))

    # Analyze static features
    static_cols = cfg.columns.static_cols
    dynamic_cols = cfg.columns.dynamic_cols

    results = []

    print("\n" + "=" * 80)
    print("STATIC FEATURE DISTRIBUTION COMPARISON")
    print("=" * 80)

    for col in static_cols:
        if col not in df_source.columns or col not in df_target.columns:
            continue

        src_vals = df_source[col].dropna().values
        tgt_vals = df_target[col].dropna().values

        if len(src_vals) == 0 or len(tgt_vals) == 0:
            continue

        # Compute statistics
        src_mean, src_std = np.mean(src_vals), np.std(src_vals)
        tgt_mean, tgt_std = np.mean(tgt_vals), np.std(tgt_vals)

        # Wasserstein distance (normalized by pooled std)
        pooled_std = np.sqrt((src_std**2 + tgt_std**2) / 2)
        if pooled_std > 0:
            w_dist = wasserstein_distance(src_vals, tgt_vals) / pooled_std
        else:
            w_dist = 0

        # Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = stats.ks_2samp(src_vals, tgt_vals)

        # Cohen's d (effect size)
        if pooled_std > 0:
            cohens_d = abs(src_mean - tgt_mean) / pooled_std
        else:
            cohens_d = 0

        results.append({
            "Feature": col,
            "Type": "Static",
            "Source_Mean": src_mean,
            "Source_Std": src_std,
            "Target_Mean": tgt_mean,
            "Target_Std": tgt_std,
            "Wasserstein": w_dist,
            "KS_Statistic": ks_stat,
            "KS_pValue": ks_pvalue,
            "Cohens_d": cohens_d,
        })

        print("%-20s | Src: %.2f±%.2f | Tgt: %.2f±%.2f | W=%.3f | KS=%.3f (p=%.4f) | d=%.3f" % (
            col, src_mean, src_std, tgt_mean, tgt_std, w_dist, ks_stat, ks_pvalue, cohens_d
        ))

    print("\n" + "=" * 80)
    print("DYNAMIC FEATURE DISTRIBUTION COMPARISON")
    print("=" * 80)

    for col in dynamic_cols:
        if col not in df_source.columns or col not in df_target.columns:
            continue

        src_vals = df_source[col].dropna().values
        tgt_vals = df_target[col].dropna().values

        if len(src_vals) == 0 or len(tgt_vals) == 0:
            continue

        src_mean, src_std = np.mean(src_vals), np.std(src_vals)
        tgt_mean, tgt_std = np.mean(tgt_vals), np.std(tgt_vals)

        pooled_std = np.sqrt((src_std**2 + tgt_std**2) / 2)
        if pooled_std > 0:
            w_dist = wasserstein_distance(src_vals, tgt_vals) / pooled_std
        else:
            w_dist = 0

        ks_stat, ks_pvalue = stats.ks_2samp(src_vals, tgt_vals)

        if pooled_std > 0:
            cohens_d = abs(src_mean - tgt_mean) / pooled_std
        else:
            cohens_d = 0

        results.append({
            "Feature": col,
            "Type": "Dynamic",
            "Source_Mean": src_mean,
            "Source_Std": src_std,
            "Target_Mean": tgt_mean,
            "Target_Std": tgt_std,
            "Wasserstein": w_dist,
            "KS_Statistic": ks_stat,
            "KS_pValue": ks_pvalue,
            "Cohens_d": cohens_d,
        })

        print("%-20s | Src: %.2f±%.2f | Tgt: %.2f±%.2f | W=%.3f | KS=%.3f (p=%.4f) | d=%.3f" % (
            col, src_mean, src_std, tgt_mean, tgt_std, w_dist, ks_stat, ks_pvalue, cohens_d
        ))

    # Save results
    results_df = pd.DataFrame(results)
    output_path = os.path.join(project_root, "figures", "feature_shift_analysis.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print("\nResults saved to: %s" % output_path)

    # Identify top shifted features
    print("\n" + "=" * 80)
    print("TOP 5 MOST SHIFTED FEATURES (by Wasserstein Distance)")
    print("=" * 80)
    top_shifted = results_df.nlargest(5, "Wasserstein")
    for _, row in top_shifted.iterrows():
        print("%-20s | W=%.3f | d=%.3f | p=%.4f" % (
            row["Feature"], row["Wasserstein"], row["Cohens_d"], row["KS_pValue"]
        ))

    return results_df


def generate_radar_chart(results_df):
    """Generate radar chart for feature shift visualization."""
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.rcParams["font.family"] = "Arial"
    matplotlib.rcParams["font.size"] = 10

    # Select top features for radar chart
    top_features = results_df.nlargest(8, "Wasserstein")

    # Prepare data
    features = top_features["Feature"].values
    wasserstein = top_features["Wasserstein"].values
    ks_stats = top_features["KS_Statistic"].values
    cohens_d = top_features["Cohens_d"].values

    # Normalize to 0-1 for radar chart
    w_norm = wasserstein / wasserstein.max()
    ks_norm = ks_stats / ks_stats.max()
    cd_norm = cohens_d / cohens_d.max()

    # Number of variables
    N = len(features)

    # Compute angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop

    # Initialize radar chart
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))

    # Plot Wasserstein distance
    w_values = np.concatenate([w_norm, [w_norm[0]]])
    ax.plot(angles, w_values, "o-", linewidth=2, label="Wasserstein Distance", color="#E63946")
    ax.fill(angles, w_values, alpha=0.15, color="#E63946")

    # Plot KS statistic
    ks_values = np.concatenate([ks_norm, [ks_norm[0]]])
    ax.plot(angles, ks_values, "o-", linewidth=2, label="KS Statistic", color="#457B9D")
    ax.fill(angles, ks_values, alpha=0.15, color="#457B9D")

    # Plot Cohen's d
    cd_values = np.concatenate([cd_norm, [cd_norm[0]]])
    ax.plot(angles, cd_values, "o-", linewidth=2, label="Cohen's d", color="#2A9D8F")
    ax.fill(angles, cd_values, alpha=0.15, color="#2A9D8F")

    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features, fontsize=9)

    # Set y-axis limits
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)

    # Add legend
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)

    # Add title
    plt.title(
        "Multi-Center Feature Shift Radar Chart\n(Top 8 Most Shifted Features)",
        fontsize=12,
        fontweight="bold",
        pad=20,
    )

    plt.tight_layout()

    # Save figure
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(project_root, "figures", "figure3_feature_shift_radar.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print("\nRadar chart saved to: %s" % output_path)


if __name__ == "__main__":
    results = compute_feature_shift()
    generate_radar_chart(results)
