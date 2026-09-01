"""Subgroup transportability analysis for mechanism-aware domain adaptation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats
import yaml

from src.data.binary_dataset import ENDPOINTS, ROOT, prepare_binary_data


def _arrays(bundle, name: str, outcome_col: str):
    return bundle.arrays(getattr(bundle, name), outcome_col)


def roc_auc_score(y_true, y_pred):
    order = np.argsort(y_pred)[::-1]
    y_true = y_true[order]
    n_pos = np.sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    cumsum = np.cumsum(y_true)
    auc = np.sum(cumsum[y_true == 0]) / (n_pos * n_neg)
    return float(auc)


def bootstrap_delta(y_true, y_pred_a, y_pred_b, n_replicates=1000):
    n = len(y_true)
    deltas = []
    for _ in range(n_replicates):
        idx = np.random.choice(n, n, replace=True)
        auc_a = roc_auc_score(y_true[idx], y_pred_a[idx])
        auc_b = roc_auc_score(y_true[idx], y_pred_b[idx])
        deltas.append(auc_a - auc_b)
    deltas = np.array(deltas)
    return {
        "mean": float(np.mean(deltas)),
        "lower": float(np.percentile(deltas, 2.5)),
        "upper": float(np.percentile(deltas, 97.5)),
        "std": float(np.std(deltas))
    }


def compute_interaction_stats(delta_low, delta_high):
    mean_diff = delta_high["mean"] - delta_low["mean"]
    
    pooled_std = np.sqrt((delta_low["std"]**2 + delta_high["std"]**2) / 2)
    effect_size = mean_diff / pooled_std if pooled_std > 0 else 0
    
    n_low = 1000
    n_high = 1000
    se = np.sqrt(delta_low["std"]**2 / n_low + delta_high["std"]**2 / n_high)
    z_score = mean_diff / se if se > 0 else 0
    p_value = 2 * (1 - stats.norm.cdf(np.abs(z_score)))
    
    return {
        "mean_difference": float(mean_diff),
        "effect_size": float(effect_size),
        "z_score": float(z_score),
        "p_value": float(p_value)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=argparse.FileType("r"), required=True)
    parser.add_argument("--results-dir", type=str, required=True)
    args = parser.parse_args()
    
    config = yaml.safe_load(args.config.read())
    results_dir = Path(args.results_dir)
    
    data_config = config["data"]
    source_path = ROOT / data_config["source_file"]
    target_path = ROOT / data_config["target_file"]
    allowlist_path = ROOT / data_config["feature_allowlist"]
    
    bundle = prepare_binary_data(
        source_path,
        target_path,
        allowlist_path=allowlist_path,
        patient_col=data_config["patient_col"],
        split_seed=data_config["split_seed"],
        source_validation_fraction=data_config["source_validation_fraction"],
        target_update_fraction=data_config["target_update_fraction"],
        target_validation_fraction=data_config["target_validation_fraction"],
    )
    
    predictions = pd.read_csv(results_dir / "test_predictions.csv")
    
    results = {}
    
    for endpoint in ENDPOINTS.keys():
        outcome_col = ENDPOINTS[endpoint]["event"]
        
        source_prob_col = f"probability_{endpoint}_source_mlp"
        updated_prob_col = f"probability_{endpoint}_updated_mlp"
        
        target_test = bundle.target_test.merge(
            predictions[["session_id", source_prob_col, updated_prob_col]],
            left_on="session_id",
            right_on="session_id",
            how="left"
        )
        
        y_true = target_test[outcome_col].values
        y_pred_source = target_test[source_prob_col].values
        y_pred_updated = target_test[updated_prob_col].values
        
        delta_all = bootstrap_delta(y_true, y_pred_updated, y_pred_source)
        
        subgroup_results = {}
        
        history_idh_col = "history_IDH_rate" if "history_IDH_rate" in target_test.columns else None
        uf_rate_col = "历史平均超滤率_mean" if "历史平均超滤率_mean" in target_test.columns else None
        pulse_pressure_col = "脉压差" if "脉压差" in target_test.columns else None
        
        if history_idh_col:
            threshold = np.median(target_test[history_idh_col].dropna())
            low_risk_mask = target_test[history_idh_col] <= threshold
            high_risk_mask = target_test[history_idh_col] > threshold
            
            y_true_low = y_true[low_risk_mask]
            y_true_high = y_true[high_risk_mask]
            y_pred_source_low = y_pred_source[low_risk_mask]
            y_pred_source_high = y_pred_source[high_risk_mask]
            y_pred_updated_low = y_pred_updated[low_risk_mask]
            y_pred_updated_high = y_pred_updated[high_risk_mask]
            
            delta_low = bootstrap_delta(y_true_low, y_pred_updated_low, y_pred_source_low)
            delta_high = bootstrap_delta(y_true_high, y_pred_updated_high, y_pred_source_high)
            
            interaction = compute_interaction_stats(delta_low, delta_high)
            
            subgroup_results["severity_subgroup"] = {
                "cutoff": {"variable": history_idh_col, "median": float(threshold)},
                "low_risk": {
                    "n_samples": int(np.sum(low_risk_mask)),
                    "event_rate": float(np.mean(y_true_low)),
                    "delta": delta_low
                },
                "high_risk": {
                    "n_samples": int(np.sum(high_risk_mask)),
                    "event_rate": float(np.mean(y_true_high)),
                    "delta": delta_high
                },
                "interaction": interaction
            }
        
        if uf_rate_col:
            threshold = np.median(target_test[uf_rate_col].dropna())
            low_intensity_mask = target_test[uf_rate_col] <= threshold
            high_intensity_mask = target_test[uf_rate_col] > threshold
            
            y_true_low = y_true[low_intensity_mask]
            y_true_high = y_true[high_intensity_mask]
            y_pred_source_low = y_pred_source[low_intensity_mask]
            y_pred_source_high = y_pred_source[high_intensity_mask]
            y_pred_updated_low = y_pred_updated[low_intensity_mask]
            y_pred_updated_high = y_pred_updated[high_intensity_mask]
            
            delta_low = bootstrap_delta(y_true_low, y_pred_updated_low, y_pred_source_low)
            delta_high = bootstrap_delta(y_true_high, y_pred_updated_high, y_pred_source_high)
            
            interaction = compute_interaction_stats(delta_low, delta_high)
            
            subgroup_results["treatment_intensity_subgroup"] = {
                "cutoff": {"variable": uf_rate_col, "median": float(threshold)},
                "low_intensity": {
                    "n_samples": int(np.sum(low_intensity_mask)),
                    "event_rate": float(np.mean(y_true_low)),
                    "delta": delta_low
                },
                "high_intensity": {
                    "n_samples": int(np.sum(high_intensity_mask)),
                    "event_rate": float(np.mean(y_true_high)),
                    "delta": delta_high
                },
                "interaction": interaction
            }
        
        if pulse_pressure_col:
            threshold = np.median(target_test[pulse_pressure_col].dropna())
            low_instability_mask = target_test[pulse_pressure_col] <= threshold
            high_instability_mask = target_test[pulse_pressure_col] > threshold
            
            y_true_low = y_true[low_instability_mask]
            y_true_high = y_true[high_instability_mask]
            y_pred_source_low = y_pred_source[low_instability_mask]
            y_pred_source_high = y_pred_source[high_instability_mask]
            y_pred_updated_low = y_pred_updated[low_instability_mask]
            y_pred_updated_high = y_pred_updated[high_instability_mask]
            
            delta_low = bootstrap_delta(y_true_low, y_pred_updated_low, y_pred_source_low)
            delta_high = bootstrap_delta(y_true_high, y_pred_updated_high, y_pred_source_high)
            
            interaction = compute_interaction_stats(delta_low, delta_high)
            
            subgroup_results["hemodynamic_instability_subgroup"] = {
                "cutoff": {"variable": pulse_pressure_col, "median": float(threshold)},
                "low_instability": {
                    "n_samples": int(np.sum(low_instability_mask)),
                    "event_rate": float(np.mean(y_true_low)),
                    "delta": delta_low
                },
                "high_instability": {
                    "n_samples": int(np.sum(high_instability_mask)),
                    "event_rate": float(np.mean(y_true_high)),
                    "delta": delta_high
                },
                "interaction": interaction
            }
        
        endpoint_results = {
            "overall_delta": delta_all,
            "subgroups": subgroup_results
        }
        
        results[endpoint] = endpoint_results
    
    output_path = results_dir / "subgroup_analysis.json"
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    
    print(f"Subgroup analysis saved to {output_path}")


if __name__ == "__main__":
    main()