"""Latent representation migration analysis for mechanism-aware domain adaptation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.data.binary_dataset import ENDPOINTS, ROOT, prepare_binary_data
from src.train.binary_models import BinaryMLP


def mmd_loss(source_features: np.ndarray, target_features: np.ndarray, kernel_type: str = "rbf", max_samples: int = 2000) -> float:
    np.random.seed(42)
    source_idx = np.random.choice(len(source_features), min(max_samples, len(source_features)), replace=False)
    target_idx = np.random.choice(len(target_features), min(max_samples, len(target_features)), replace=False)
    source = source_features[source_idx]
    target = target_features[target_idx]
    
    n = source.shape[0]
    m = target.shape[0]
    if kernel_type == "linear":
        k_ss = np.dot(source, source.T)
        k_st = np.dot(source, target.T)
        k_tt = np.dot(target, target.T)
    elif kernel_type == "rbf":
        dist_st = np.sqrt(np.sum((source[:, np.newaxis] - target[np.newaxis]) ** 2, axis=2))
        sigma = np.median(dist_st)
        if sigma < 1e-6:
            sigma = 1.0
        k_ss = np.exp(-np.sum((source[:, np.newaxis] - source[np.newaxis]) ** 2, axis=2) / (2 * sigma ** 2))
        k_st = np.exp(-dist_st ** 2 / (2 * sigma ** 2))
        k_tt = np.exp(-np.sum((target[:, np.newaxis] - target[np.newaxis]) ** 2, axis=2) / (2 * sigma ** 2))
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")
    return float(np.mean(k_ss) - 2 * np.mean(k_st) + np.mean(k_tt))


def wasserstein_distance(source_features: np.ndarray, target_features: np.ndarray, max_samples: int = 2000) -> float:
    np.random.seed(42)
    source_idx = np.random.choice(len(source_features), min(max_samples, len(source_features)), replace=False)
    target_idx = np.random.choice(len(target_features), min(max_samples, len(target_features)), replace=False)
    source = source_features[source_idx]
    target = target_features[target_idx]
    
    n = min(len(source), len(target))
    source_sorted = np.sort(source[:n], axis=0)
    target_sorted = np.sort(target[:n], axis=0)
    return float(np.mean(np.abs(source_sorted - target_sorted)))


def covariance_distance(source_features: np.ndarray, target_features: np.ndarray) -> float:
    source_cov = np.cov(source_features, rowvar=False)
    target_cov = np.cov(target_features, rowvar=False)
    return float(np.linalg.norm(source_cov - target_cov, 'fro') ** 2 / (4 * source_features.shape[1] ** 2))


def domain_classifier_auc(source_features: np.ndarray, target_features: np.ndarray) -> float:
    X = np.vstack([source_features, target_features])
    y = np.concatenate([np.zeros(len(source_features)), np.ones(len(target_features))])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_scaled, y)
    
    probs = clf.predict_proba(X_scaled)[:, 1]
    return float(roc_auc_score(y, probs))


def linear_probe_auc(features: np.ndarray, labels: np.ndarray) -> float:
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(features_scaled, labels)
    
    probs = clf.predict_proba(features_scaled)[:, 1]
    return float(roc_auc_score(labels, probs))


def _arrays(bundle, name: str, outcome_col: str):
    return bundle.arrays(getattr(bundle, name), outcome_col)


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
    
    mlp_states = torch.load(results_dir / "mlp_models.pt", map_location="cpu")
    
    device = torch.device("cpu")
    
    results = {}
    
    for endpoint in ENDPOINTS.keys():
        outcome_col = ENDPOINTS[endpoint]["event"]
        x_source_train, y_source_train = _arrays(bundle, "source_train", outcome_col)
        x_target_train, y_target_train = _arrays(bundle, "target_train", outcome_col)
        
        input_dim = x_source_train.shape[1]
        
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
        
        with torch.no_grad():
            source_latent_before = source_model.get_features(torch.tensor(x_source_train, dtype=torch.float32)).numpy()
            target_latent_before = source_model.get_features(torch.tensor(x_target_train, dtype=torch.float32)).numpy()
            
            source_latent_after = updated_model.get_features(torch.tensor(x_source_train, dtype=torch.float32)).numpy()
            target_latent_after = updated_model.get_features(torch.tensor(x_target_train, dtype=torch.float32)).numpy()
        
        endpoint_results = {}
        
        endpoint_results["before_alignment"] = {
            "mmd_rbf": mmd_loss(source_latent_before, target_latent_before),
            "mmd_linear": mmd_loss(source_latent_before, target_latent_before, kernel_type="linear"),
            "wasserstein": wasserstein_distance(source_latent_before, target_latent_before),
            "covariance_distance": covariance_distance(source_latent_before, target_latent_before),
            "domain_classifier_auc": domain_classifier_auc(source_latent_before, target_latent_before),
            "source_linear_probe": linear_probe_auc(source_latent_before, y_source_train),
            "target_linear_probe": linear_probe_auc(target_latent_before, y_target_train),
            "latent_dim": source_latent_before.shape[1],
            "n_source": len(source_latent_before),
            "n_target": len(target_latent_before),
        }
        
        endpoint_results["after_alignment"] = {
            "mmd_rbf": mmd_loss(source_latent_after, target_latent_after),
            "mmd_linear": mmd_loss(source_latent_after, target_latent_after, kernel_type="linear"),
            "wasserstein": wasserstein_distance(source_latent_after, target_latent_after),
            "covariance_distance": covariance_distance(source_latent_after, target_latent_after),
            "domain_classifier_auc": domain_classifier_auc(source_latent_after, target_latent_after),
            "source_linear_probe": linear_probe_auc(source_latent_after, y_source_train),
            "target_linear_probe": linear_probe_auc(target_latent_after, y_target_train),
            "latent_dim": source_latent_after.shape[1],
            "n_source": len(source_latent_after),
            "n_target": len(target_latent_after),
        }
        
        endpoint_results["delta"] = {
            "mmd_rbf": endpoint_results["after_alignment"]["mmd_rbf"] - endpoint_results["before_alignment"]["mmd_rbf"],
            "mmd_linear": endpoint_results["after_alignment"]["mmd_linear"] - endpoint_results["before_alignment"]["mmd_linear"],
            "wasserstein": endpoint_results["after_alignment"]["wasserstein"] - endpoint_results["before_alignment"]["wasserstein"],
            "covariance_distance": endpoint_results["after_alignment"]["covariance_distance"] - endpoint_results["before_alignment"]["covariance_distance"],
            "domain_classifier_auc": endpoint_results["after_alignment"]["domain_classifier_auc"] - endpoint_results["before_alignment"]["domain_classifier_auc"],
            "source_linear_probe": endpoint_results["after_alignment"]["source_linear_probe"] - endpoint_results["before_alignment"]["source_linear_probe"],
            "target_linear_probe": endpoint_results["after_alignment"]["target_linear_probe"] - endpoint_results["before_alignment"]["target_linear_probe"],
        }
        
        results[endpoint] = endpoint_results
    
    output_path = results_dir / "latent_analysis.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    
    print(f"Latent representation analysis saved to {output_path}")


if __name__ == "__main__":
    main()
