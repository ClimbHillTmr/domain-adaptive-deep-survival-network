import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import yaml

from src.data.dataset import prepare_dataloaders
from src.evaluate.dca_analysis import calculate_net_benefit
from src.evaluate.export_real_predictions import (
    breslow_baseline_hazard,
    calibration_metrics,
    event_probability_by_horizon,
)
from src.evaluate.metrics import _binary_auc, compute_bootstrap_cindex, compute_ipcw_weights, concordance_index, weighted_cox_loss
from src.reproducibility import record_environment, seed_everything


RESULT_DIR = Path("experiments/results")
THRESHOLDS = [0.10, 0.20, 0.30]
HORIZONS = [30, 60, 120]


def load_config(config_path: str = "conf/config.yaml") -> Dict:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def make_survival_frame(x: np.ndarray, e: np.ndarray, t: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
    frame = pd.DataFrame(x, columns=feature_names)
    frame["et_min"] = t
    frame["events"] = e.astype(int)
    return frame


def keep_trainable_features(train_frame: pd.DataFrame, feature_names: List[str], variance_floor: float = 1e-4) -> List[str]:
    variances = train_frame[feature_names].var()
    keep = [name for name in feature_names if np.isfinite(train_frame[name]).all() and variances[name] > variance_floor]
    return keep


def fit_linear_cox(train_frame: pd.DataFrame, features: List[str], penalizer: float, epochs: int = 200) -> np.ndarray:
    """Dependency-free linear CoxPH fitted with the project's weighted Cox loss."""
    x = torch.tensor(train_frame[features].to_numpy(dtype=np.float32))
    event = torch.tensor(train_frame["events"].to_numpy(dtype=np.float32))
    duration = torch.tensor(train_frame["et_min"].to_numpy(dtype=np.float32))
    weights = torch.tensor(compute_ipcw_weights(duration.numpy(), event.numpy()))
    beta = torch.nn.Parameter(torch.zeros(x.shape[1]))
    optimizer = torch.optim.Adam([beta], lr=0.03)
    for _ in range(epochs):
        optimizer.zero_grad()
        risk = x @ beta
        loss = weighted_cox_loss(risk.unsqueeze(-1), event, duration, weights)
        (loss + penalizer * beta.square().sum()).backward()
        optimizer.step()
    return beta.detach().numpy()


def select_penalizer(train_frame: pd.DataFrame, val_frame: pd.DataFrame, features: List[str]) -> Dict:
    grid = [0.01, 0.1, 1.0, 10.0, 100.0]
    candidates = []
    for penalizer in grid:
        coefficients = fit_linear_cox(train_frame, features, penalizer)
        val_risk = val_frame[features].to_numpy(dtype=float) @ coefficients
        val_cindex = concordance_index(val_frame["et_min"].values, val_risk, val_frame["events"].values)
        candidates.append({"penalizer": penalizer, "val_cindex": float(val_cindex), "coefficients": coefficients})
    best = max(candidates, key=lambda item: item["val_cindex"])
    return {"grid": [{"penalizer": item["penalizer"], "val_cindex": item["val_cindex"]} for item in candidates], "best": best}


def threshold_rows(model_name: str, horizon: int, y_true: np.ndarray, y_prob: np.ndarray) -> List[Dict]:
    prevalence = float(np.mean(y_true))
    rows = []
    for threshold in THRESHOLDS:
        flagged = y_prob >= threshold
        tp = int(np.sum(flagged & (y_true == 1)))
        fp = int(np.sum(flagged & (y_true == 0)))
        flagged_n = int(np.sum(flagged))
        event_n = int(np.sum(y_true))
        nb_model = float(calculate_net_benefit(y_true, y_prob, np.asarray([threshold]))[0])
        nb_all = float(prevalence - (1 - prevalence) * (threshold / (1 - threshold)))
        rows.append(
            {
                "model": model_name,
                "horizon_min": horizon,
                "threshold": threshold,
                "clinical_action": {
                    0.10: "intensified monitoring",
                    0.20: "ultrafiltration review",
                    0.30: "bedside reassessment",
                }[threshold],
                "flagged_sessions_n": flagged_n,
                "flagged_sessions_proportion": float(flagged_n / len(y_true)),
                "events_n": event_n,
                "captured_events_n": tp,
                "event_capture_rate": float(tp / event_n) if event_n > 0 else None,
                "ppv": float(tp / flagged_n) if flagged_n > 0 else None,
                "workload_per_event_captured": float(flagged_n / tp) if tp > 0 else None,
                "net_benefit_model": nb_model,
                "net_benefit_treat_all": nb_all,
                "net_benefit_treat_none": 0.0,
                "delta_vs_treat_all": float(nb_model - nb_all),
                "delta_vs_treat_none": nb_model,
                "false_positives_n": fp,
            }
        )
    return rows


def main() -> None:
    config = load_config()
    seed_everything(config["training"]["seed"])
    record_environment(log_dir=config["paths"].get("log_dir", "experiments/logs"), filename="pip_freeze_local_cox.txt")

    data = prepare_dataloaders(
        source_path=str(Path(config["paths"]["data_dir"]) / config["data"]["source_file"]),
        target_path=str(Path(config["paths"]["data_dir"]) / config["data"]["target_file"]),
        batch_size=config["training"]["batch_size"],
        seed=config["training"]["seed"],
        target_adapt_ratio=config["data"].get("target_adapt_ratio", 0.2),
        target_val_ratio=config["data"].get("target_val_ratio", 0.2),
        patient_col=config["data"].get("patient_col", "患者id"),
        split_strategy=config["data"].get("split_strategy", "patient"),
    )
    feature_names = data["feature_names"]

    train_frame = make_survival_frame(data["x_train"], data["e_train"], data["t_train"], feature_names)
    val_frame = make_survival_frame(data["x_val"], data["e_val"], data["t_val"], feature_names)
    test_frame = make_survival_frame(data["x_test"], data["e_test"], data["t_test"], feature_names)

    kept_features = keep_trainable_features(train_frame, feature_names)
    dropped_features = [feature for feature in feature_names if feature not in kept_features]

    selection = select_penalizer(train_frame, val_frame, kept_features)
    best_penalizer = selection["best"]["penalizer"]
    coefficients = selection["best"]["coefficients"]
    train_risk = train_frame[kept_features].to_numpy(dtype=float) @ coefficients
    test_risk = test_frame[kept_features].to_numpy(dtype=float) @ coefficients

    c_index = concordance_index(test_frame["et_min"].values, test_risk, test_frame["events"].values)
    ci_lower, ci_upper = compute_bootstrap_cindex(
        test_frame["et_min"].values,
        test_frame["events"].values,
        test_risk,
        patient_ids=data["patient_ids_test"],
        n_bootstrap=200,
        seed=config["training"]["seed"],
    )

    metrics = {
        "C-index": float(c_index),
        "C-index_CI_lower": float(ci_lower),
        "C-index_CI_upper": float(ci_upper),
        "bootstrap_unit": "patient",
        "n_bootstrap": 200,
        "seed": int(config["training"]["seed"]),
    }
    for horizon in HORIZONS:
        y_horizon = ((test_frame["events"].values.astype(int) == 1) & (test_frame["et_min"].values <= horizon)).astype(int)
        auc = _binary_auc(y_horizon, test_risk)
        if auc is not None:
            metrics[f"AUC_{int(horizon)}m"] = float(auc)

    event_times, cum_hazard = breslow_baseline_hazard(train_frame["et_min"].values, train_frame["events"].values, train_risk)
    prediction_frame = data["df_target"].iloc[data["idx_test"]].copy().reset_index(drop=True)
    prediction_frame["risk_score"] = test_risk

    calibration = {}
    threshold_summary_rows = []
    for horizon in HORIZONS:
        probability = event_probability_by_horizon(test_risk, event_times, cum_hazard, horizon)
        observed = ((prediction_frame["events"].astype(int) == 1) & (prediction_frame["et_min"].astype(float) <= horizon)).astype(int)
        prediction_frame[f"event_prob_{horizon}m"] = probability
        prediction_frame[f"event_by_{horizon}m"] = observed
        eligible = ((prediction_frame["events"].astype(int) == 1) | (prediction_frame["et_min"].astype(float) > horizon)).values
        prediction_frame[f"eligible_by_{horizon}m"] = eligible.astype(int)
        calibration[f"{horizon}m"] = calibration_metrics(observed.values[eligible], probability[eligible])
        threshold_summary_rows.extend(threshold_rows("Local Cox update", horizon, observed.values.astype(int)[eligible], probability[eligible]))

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    prediction_frame.to_csv(RESULT_DIR / "local_cox_test_predictions.csv", index=False)
    (RESULT_DIR / "local_cox_calibration_metrics.json").write_text(json.dumps(calibration, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULT_DIR / "local_cox_threshold_summary.json").write_text(json.dumps(threshold_summary_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    result_payload = {
        "metadata": {
            "run_id": os.environ.get("LOCKED_RUN_ID"),
            "model_name": "Local Cox update",
            "split_strategy": data["split_strategy"],
            "patient_col": data["patient_col"],
            "target_adapt_ratio": config["data"].get("target_adapt_ratio", 0.2),
            "target_val_ratio": config["data"].get("target_val_ratio", 0.2),
            "n_features_allowlist": len(feature_names),
            "n_features_used": len(kept_features),
            "used_features": kept_features,
            "dropped_low_variance_features": dropped_features,
            "penalizer_grid": selection["grid"],
            "selected_penalizer": best_penalizer,
            "n_test": int(len(prediction_frame)),
        },
        "metrics": metrics,
        "calibration": calibration,
    }
    (RESULT_DIR / "local_cox_update_results.json").write_text(json.dumps(result_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    threshold_df = pd.DataFrame(threshold_summary_rows)
    Path("tables").mkdir(parents=True, exist_ok=True)
    threshold_df.to_csv("tables/local_cox_threshold_summary.csv", index=False)
    print(json.dumps(result_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
