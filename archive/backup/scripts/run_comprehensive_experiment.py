"""
Comprehensive experiment runner for paper submission.
Runs ablation study + ML baselines + calibration + DCA + SHAP analysis.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import torch
from typing import Dict, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.loader import DialysisDataLoader
from src.training.dadsn_runner import run_ablation_study
from src.evaluation.ml_baselines import TraditionalMLBaselines, prepare_features, prepare_survival_labels
from src.evaluation.calibration import (
    calculate_calibration_probs,
    calculate_ece,
    decision_curve_analysis,
)
from src.evaluation.shap_analysis import compute_shap_values_from_risk, compute_shap_summary
from src.evaluation.plotting import generate_all_figures
from src.metrics.survival_metrics import concordance_index_censored

log = logging.getLogger(__name__)


def run_ml_baselines(source_data, target_data, cfg):
    """Run traditional ML baselines and return results."""
    log.info("=" * 60)
    log.info("RUNNING TRADITIONAL ML BASELINES")
    log.info("=" * 60)
    
    time_horizon = 60  # minutes
    
    # Prepare features
    X_train = prepare_features(source_data["static"], source_data["dynamic"])
    y_train = prepare_survival_labels(source_data["targets"], time_horizon)
    
    X_test = prepare_features(target_data["static"], target_data["dynamic"])
    y_test = prepare_survival_labels(target_data["targets"], time_horizon)
    
    # Train models
    baselines = TraditionalMLBaselines(time_horizon=time_horizon)
    results = baselines.train_all(X_train, y_train, X_test, y_test)
    
    # Evaluate
    metrics = {}
    for name, res in results.items():
        risk_scores = res["val_risk_scores"]
        
        # C-index
        event = target_data["targets"]["event"].astype(bool)
        duration = target_data["targets"]["duration"].astype(float)
        c_index, _, _, _, _ = concordance_index_censored(event, duration, risk_scores)
        
        # Calibration
        bin_centers, observed_probs, predicted_probs, _ = calculate_calibration_probs(
            risk_scores, event.astype(float)
        )
        ece = calculate_ece(risk_scores, event.astype(float))
        
        # DCA
        thresholds, nb_model, nb_all, nb_none = decision_curve_analysis(
            risk_scores, event.astype(float)
        )
        
        metrics[name] = {
            "c_index": c_index,
            "ece": ece,
            "calibration": {
                "bin_centers": bin_centers.tolist(),
                "observed_probs": observed_probs.tolist(),
                "predicted_probs": predicted_probs.tolist(),
            },
            "dca": {
                "thresholds": thresholds.tolist(),
                "net_benefit_model": nb_model.tolist(),
                "net_benefit_all": nb_all.tolist(),
                "net_benefit_none": nb_none.tolist(),
            },
        }
        
        log.info(f"  {name}: C-index={c_index:.4f}, ECE={ece:.4f}")
    
    return metrics, baselines


def run_shap_analysis(model, source_data, target_data, cfg, device):
    """Run SHAP analysis on the best model."""
    log.info("=" * 60)
    log.info("RUNNING SHAP ANALYSIS")
    log.info("=" * 60)
    
    feature_names = cfg.columns.static_cols
    
    shap_values, baseline_risks = compute_shap_values_from_risk(
        model,
        target_data["static"],
        target_data["dynamic"],
        device,
        feature_names,
        n_samples=500,
    )
    
    summary = compute_shap_summary(shap_values, feature_names, top_n=15)
    
    log.info("\nTop 15 Features by SHAP Importance:")
    log.info(summary.to_string(index=False))
    
    return shap_values, baseline_risks, summary


def run_comprehensive_experiment(
    source_data,
    target_data,
    cfg,
    save_dir,
    seeds=None,
    run_shap=True,
    run_ml_baselines_flag=True,
):
    """
    Run complete experiment pipeline:
    1. Ablation study (DA-DSN variants)
    2. Traditional ML baselines
    3. Calibration analysis
    4. Decision Curve Analysis
    5. SHAP interpretability
    6. Generate all figures
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(save_dir, "experiment.log")),
            logging.StreamHandler(),
        ],
    )
    
    all_results = {}
    
    # Step 1: Run ablation study
    log.info("=" * 80)
    log.info("STEP 1: ABLATION STUDY")
    log.info("=" * 80)
    
    if seeds is None:
        seeds = [42, 123, 456]
    
    ablation_results = run_ablation_study(
        source_data=source_data,
        target_data=target_data,
        save_dir=save_dir,
        config=cfg,
        seeds=seeds,
    )
    
    all_results["ablation"] = ablation_results
    
    # Step 2: Run ML baselines
    if run_ml_baselines_flag:
        log.info("=" * 80)
        log.info("STEP 2: TRADITIONAL ML BASELINES")
        log.info("=" * 80)
        
        ml_metrics, ml_models = run_ml_baselines(source_data, target_data, cfg)
        all_results["ml_baselines"] = ml_metrics
    
    # Step 3: Load best model for SHAP analysis
    if run_shap and "dadsn_mmd" in ablation_results:
        log.info("=" * 80)
        log.info("STEP 3: SHAP ANALYSIS")
        log.info("=" * 80)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load best model (use seed 42 for SHAP)
        from src.models.dadsn_model import DADSN
        from src.training.dadsn_runner import predict_risk
        
        model_path = os.path.join(save_dir, "dadsn_mmd", "seed_42.pt")
        if os.path.exists(model_path):
            model = DADSN(
                cfg,
                num_static_features=source_data["static"].shape[1],
                num_dynamic_features=source_data["dynamic"].shape[2],
                use_transformer=True,
            )
            model.load_state_dict(torch.load(model_path, weights_only=False))
            model = model.to(device)
            model.eval()
            
            shap_values, baseline_risks, shap_summary = run_shap_analysis(
                model, source_data, target_data, cfg, device
            )
            all_results["shap_summary"] = shap_summary.to_dict()
        else:
            log.warning(f"Model checkpoint not found at {model_path}, skipping SHAP")
    
    # Step 4: Generate figures
    log.info("=" * 80)
    log.info("STEP 4: GENERATING FIGURES")
    log.info("=" * 80)
    
    # Prepare calibration data
    calibration_data = {}
    dca_data = {}
    
    if "ml_baselines" in all_results:
        for name, metrics in all_results["ml_baselines"].items():
            calibration_data[name] = metrics["calibration"]
            dca_data[name] = metrics["dca"]
    
    # Generate all figures
    figures = generate_all_figures(
        ablation_results=ablation_results,
        calibration_data=calibration_data,
        dca_data=dca_data,
        shap_summary=all_results.get("shap_summary"),
        save_dir=os.path.join(save_dir, "figures"),
    )
    
    all_results["figures"] = figures
    
    # Save all results
    results_path = os.path.join(save_dir, "comprehensive_results.json")
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        return obj
    
    serializable_results = json.loads(
        json.dumps(all_results, default=convert_to_serializable)
    )
    
    with open(results_path, "w") as f:
        json.dump(serializable_results, f, indent=2)
    
    log.info(f"\nAll results saved to {results_path}")
    log.info(f"Figures saved to {os.path.join(save_dir, 'figures')}/")
    
    return all_results
