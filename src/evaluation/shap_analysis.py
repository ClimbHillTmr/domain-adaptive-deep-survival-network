"""
Perturbation-based Feature Importance Analysis.

NOTE: This is NOT true SHAP (SHapley Additive exPlanations).
True SHAP requires computing marginal contributions across all feature coalitions,
which is computationally expensive. This implementation uses a simplified
perturbation-based approach that measures feature importance by observing
risk score changes when features are set to their mean values.

For proper SHAP analysis, consider using the `shap` library with
KernelExplainer or DeepExplainer.

Provides:
- Feature importance ranking
- Summary plots
- Dependence analysis
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional

log = logging.getLogger(__name__)


def compute_shap_values_from_risk(
    model,
    X_static,
    X_dynamic,
    device,
    feature_names: List[str],
    n_samples: int = 1000,
    batch_size: int = 256,
):
    """
    Approximate SHAP values using perturbation-based method.
    
    For deep survival models, we compute feature importance by:
    1. Computing baseline risk scores
    2. Perturbing each feature (set to mean)
    3. Measuring risk score change
    
    Parameters:
    -----------
    model : trained survival model
    X_static : static features (N, D)
    X_dynamic : dynamic features (N, T, F)
    device : torch device
    feature_names : list of feature names for static features
    n_samples : number of samples to use for SHAP computation
    batch_size : batch size for model inference
    
    Returns:
    --------
    shap_values : (n_samples, n_features) array
    baseline_risks : (n_samples,) array
    """
    import torch
    
    model.eval()
    
    # Use subset for efficiency
    if len(X_static) > n_samples:
        indices = np.random.choice(len(X_static), n_samples, replace=False)
        X_static_sub = X_static[indices]
        X_dynamic_sub = X_dynamic[indices]
    else:
        X_static_sub = X_static
        X_dynamic_sub = X_dynamic
    
    n_features = X_static_sub.shape[1]
    
    # Compute baseline risks
    with torch.no_grad():
        static_tensor = torch.tensor(X_static_sub, dtype=torch.float32).to(device)
        dynamic_tensor = torch.tensor(X_dynamic_sub, dtype=torch.float32).to(device)
        
        baseline_risks = []
        for i in range(0, len(static_tensor), batch_size):
            batch_s = static_tensor[i:i+batch_size]
            batch_d = dynamic_tensor[i:i+batch_size]
            log_hazard, _ = model(batch_s, batch_d)
            baseline_risks.append(log_hazard.cpu().numpy())
        
        baseline_risks = np.concatenate(baseline_risks, axis=0).squeeze()
    
    # Compute SHAP values by perturbation
    shap_values = np.zeros((n_samples, n_features))
    
    for feat_idx in range(n_features):
        # Create perturbed version (set feature to mean)
        X_perturbed = X_static_sub.copy()
        feat_mean = X_static_sub[:, feat_idx].mean()
        X_perturbed[:, feat_idx] = feat_mean
        
        # Compute perturbed risks
        with torch.no_grad():
            perturbed_tensor = torch.tensor(X_perturbed, dtype=torch.float32).to(device)
            
            perturbed_risks = []
            for i in range(0, len(perturbed_tensor), batch_size):
                batch_s = perturbed_tensor[i:i+batch_size]
                batch_d = dynamic_tensor[i:i+batch_size]
                log_hazard, _ = model(batch_s, batch_d)
                perturbed_risks.append(log_hazard.cpu().numpy())
            
            perturbed_risks = np.concatenate(perturbed_risks, axis=0).squeeze()
        
        # SHAP value = baseline - perturbed (positive = feature increases risk)
        shap_values[:, feat_idx] = baseline_risks - perturbed_risks
    
    return shap_values, baseline_risks


def compute_shap_summary(
    shap_values: np.ndarray,
    feature_names: List[str],
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Create SHAP summary table with mean absolute importance.
    
    Parameters:
    -----------
    shap_values : (n_samples, n_features) array
    feature_names : list of feature names
    top_n : number of top features to return
    
    Returns:
    --------
    DataFrame with feature importance statistics
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    std_shap = shap_values.std(axis=0)
    mean_shap = shap_values.mean(axis=0)
    
    # Sort by mean absolute importance
    sorted_idx = np.argsort(mean_abs_shap)[::-1]
    
    summary_data = {
        "feature": [feature_names[i] for i in sorted_idx],
        "mean_abs_shap": mean_abs_shap[sorted_idx],
        "std_shap": std_shap[sorted_idx],
        "mean_shap": mean_shap[sorted_idx],
        "rank": range(1, len(feature_names) + 1),
    }
    
    df_summary = pd.DataFrame(summary_data)
    return df_summary.head(top_n)


def compute_feature_direction(
    shap_values: np.ndarray,
    X_static: np.ndarray,
    feature_names: List[str],
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Analyze feature effect direction (positive/negative correlation with risk).
    
    Parameters:
    -----------
    shap_values : (n_samples, n_features) array
    X_static : static features (N, D)
    feature_names : list of feature names
    top_n : number of top features to analyze
    
    Returns:
    --------
    DataFrame with feature direction analysis
    """
    summary = compute_shap_summary(shap_values, feature_names, top_n=top_n)
    
    direction_data = []
    for _, row in summary.iterrows():
        feat_name = row["feature"]
        feat_idx = feature_names.index(feat_name)
        
        # Correlation between feature value and SHAP value
        corr = np.corrcoef(X_static[:, feat_idx], shap_values[:, feat_idx])[0, 1]
        
        # High value effect
        high_mask = X_static[:, feat_idx] > X_static[:, feat_idx].median()
        high_effect = shap_values[high_mask, feat_idx].mean()
        
        # Low value effect
        low_mask = X_static[:, feat_idx] <= X_static[:, feat_idx].median()
        low_effect = shap_values[low_mask, feat_idx].mean()
        
        direction_data.append({
            "feature": feat_name,
            "correlation": corr,
            "high_value_effect": high_effect,
            "low_value_effect": low_effect,
            "direction": "positive" if corr > 0 else "negative",
        })
    
    return pd.DataFrame(direction_data)
