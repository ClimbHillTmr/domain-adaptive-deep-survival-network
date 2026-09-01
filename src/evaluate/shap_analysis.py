"""SHAP analysis for mechanism-aware domain adaptation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.stats as stats
import shap
import torch
import yaml

from src.data.binary_dataset import ENDPOINTS, ROOT, prepare_binary_data
from src.train.binary_models import BinaryMLP


def _arrays(bundle, name: str, outcome_col: str):
    return bundle.arrays(getattr(bundle, name), outcome_col)


def compute_shap_importance(model, X, feature_names, n_background=100, nsamples=100):
    np.random.seed(42)
    n_samples = min(500, len(X))
    sample_idx = np.random.choice(len(X), n_samples, replace=False)
    X_sample = X[sample_idx]
    
    background = X_sample[:n_background]
    test_data = X_sample[n_background:]
    
    def predict_fn(X):
        with torch.no_grad():
            return model(torch.tensor(X, dtype=torch.float32)).numpy()
    
    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(test_data, nsamples=nsamples)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    abs_shap = np.abs(shap_values)
    feature_importance = np.mean(abs_shap, axis=0)
    
    return feature_importance, test_data.shape[0]


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
    
    model_config = config["model"]
    hidden_dims = tuple(model_config["mlp_hidden_dims"])
    dropout = float(model_config["dropout"])
    physio_indices = model_config.get("physio_indices")
    treat_indices = model_config.get("treat_indices")
    feature_names = bundle.feature_names
    
    mlp_states = torch.load(results_dir / "mlp_models.pt", map_location="cpu")
    
    results = {}
    
    for endpoint in ENDPOINTS.keys():
        outcome_col = ENDPOINTS[endpoint]["event"]
        x_source_val, y_source_val = _arrays(bundle, "source_val", outcome_col)
        x_target_test, y_target_test = _arrays(bundle, "target_test", outcome_col)
        
        input_dim = x_source_val.shape[1]
        
        if bool(model_config.get("use_outcome_specific_alignment", False)):
            if endpoint == "idh":
                alignment_indices = model_config.get("physio_indices")
            elif endpoint == "ih":
                alignment_indices = model_config.get("treat_indices")
        else:
            alignment_indices = model_config.get("physio_indices")
        
        source_model = BinaryMLP(input_dim, hidden_dims, dropout, physio_indices=alignment_indices)
        source_model.load_state_dict(mlp_states[endpoint]["source_mlp"])
        source_model.eval()
        
        updated_model = BinaryMLP(input_dim, hidden_dims, dropout, physio_indices=alignment_indices)
        updated_model.load_state_dict(mlp_states[endpoint]["updated_mlp"])
        updated_model.eval()
        
        source_importance_before, n_source_before = compute_shap_importance(source_model, x_source_val, feature_names)
        target_importance_before, n_target_before = compute_shap_importance(source_model, x_target_test, feature_names)
        
        source_importance_after, n_source_after = compute_shap_importance(updated_model, x_source_val, feature_names)
        target_importance_after, n_target_after = compute_shap_importance(updated_model, x_target_test, feature_names)
        
        endpoint_results = {}
        
        for phase, importance_source, importance_target, n_source, n_target in [
            ("before_alignment", source_importance_before, target_importance_before, n_source_before, n_target_before),
            ("after_alignment", source_importance_after, target_importance_after, n_source_after, n_target_after),
        ]:
            total_source = np.sum(importance_source)
            total_target = np.sum(importance_target)
            
            if physio_indices is not None:
                physio_source = np.sum(importance_source[physio_indices])
                physio_target = np.sum(importance_target[physio_indices])
            else:
                physio_source = 0
                physio_target = 0
            
            if treat_indices is not None:
                treat_source = np.sum(importance_source[treat_indices])
                treat_target = np.sum(importance_target[treat_indices])
            else:
                treat_source = 0
                treat_target = 0
            
            physio_ratio_source = physio_source / total_source if total_source > 0 else 0
            physio_ratio_target = physio_target / total_target if total_target > 0 else 0
            treat_ratio_source = treat_source / total_source if total_source > 0 else 0
            treat_ratio_target = treat_target / total_target if total_target > 0 else 0
            
            spearman_corr, spearman_p = stats.spearmanr(importance_source, importance_target)
            pearson_corr, pearson_p = stats.pearsonr(importance_source, importance_target)
            
            top_source = []
            for i in np.argsort(importance_source)[::-1][:10]:
                top_source.append({
                    "feature": feature_names[i],
                    "importance": float(importance_source[i]),
                    "group": "physiology" if i in physio_indices else ("treatment" if i in treat_indices else "other") if physio_indices is not None else "all"
                })
            
            top_target = []
            for i in np.argsort(importance_target)[::-1][:10]:
                top_target.append({
                    "feature": feature_names[i],
                    "importance": float(importance_target[i]),
                    "group": "physiology" if i in physio_indices else ("treatment" if i in treat_indices else "other") if physio_indices is not None else "all"
                })
            
            endpoint_results[phase] = {
                "source": {
                    "total_importance": float(total_source),
                    "physiology_importance": float(physio_source),
                    "treatment_importance": float(treat_source),
                    "physiology_ratio": float(physio_ratio_source),
                    "treatment_ratio": float(treat_ratio_source),
                    "top_10_features": top_source,
                    "n_samples": n_source,
                    "feature_importance": [float(x) for x in importance_source]
                },
                "target": {
                    "total_importance": float(total_target),
                    "physiology_importance": float(physio_target),
                    "treatment_importance": float(treat_target),
                    "physiology_ratio": float(physio_ratio_target),
                    "treatment_ratio": float(treat_ratio_target),
                    "top_10_features": top_target,
                    "n_samples": n_target,
                    "feature_importance": [float(x) for x in importance_target]
                },
                "cross_domain_correlation": {
                    "spearman_r": float(spearman_corr),
                    "spearman_p": float(spearman_p),
                    "pearson_r": float(pearson_corr),
                    "pearson_p": float(pearson_p)
                }
            }
        
        delta_physio_ratio = endpoint_results["after_alignment"]["target"]["physiology_ratio"] - endpoint_results["before_alignment"]["target"]["physiology_ratio"]
        delta_treat_ratio = endpoint_results["after_alignment"]["target"]["treatment_ratio"] - endpoint_results["before_alignment"]["target"]["treatment_ratio"]
        delta_corr = endpoint_results["after_alignment"]["cross_domain_correlation"]["spearman_r"] - endpoint_results["before_alignment"]["cross_domain_correlation"]["spearman_r"]
        
        endpoint_results["delta"] = {
            "physiology_ratio_target": float(delta_physio_ratio),
            "treatment_ratio_target": float(delta_treat_ratio),
            "spearman_correlation": float(delta_corr)
        }
        
        results[endpoint] = endpoint_results
    
    output_path = results_dir / "shap_analysis_full.json"
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    
    print(f"SHAP analysis saved to {output_path}")


if __name__ == "__main__":
    main()
