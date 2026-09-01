"""Measurement audit, discrete-time models, Cox sensitivity, and CDAN ablation."""

from __future__ import annotations

import argparse
import ast
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.binary_dataset import ENDPOINTS, ROOT, prepare_binary_data
from src.evaluate.binary_metrics import evaluate_binary, paired_patient_bootstrap_delta, select_youden_threshold
from src.evaluate.metrics import compute_bootstrap_cindex, concordance_index
from src.models.discrete_cdan import DiscreteHazardCDAN
from src.reproducibility import seed_everything
from src.train.binary_models import calibrate_probability, fit_probability_calibrator


def _relative_nodes(value: object) -> list[float]:
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return []
    if not isinstance(value, (list, tuple)):
        return []
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return []


def measurement_audit(bundle, milestones: tuple[int, ...]) -> dict[str, object]:
    cohorts = {
        "source": pd.concat((bundle.source_train, bundle.source_val), ignore_index=True),
        "target": pd.concat((bundle.target_train, bundle.target_val, bundle.target_test), ignore_index=True),
    }
    report: dict[str, object] = {"milestones_minutes": list(milestones), "cohorts": {}}
    concentrations: dict[str, dict[str, float]] = {}
    for cohort, frame in cohorts.items():
        node_lists = frame.get("minutes_from_start_list", pd.Series([[]] * len(frame))).map(_relative_nodes)
        intervals = [b - a for nodes in node_lists if len(nodes) >= 2 for a, b in zip(nodes, nodes[1:]) if b >= a]
        cohort_report = {
            "sessions": int(len(frame)),
            "measurements_per_session_median": float(node_lists.map(len).median()),
            "interval_minutes_median": float(np.median(intervals)) if intervals else None,
            "endpoints": {},
        }
        concentrations[cohort] = {}
        for endpoint, columns in ENDPOINTS.items():
            event_times = frame.loc[frame[columns["event"]].eq(1), columns["time"]].to_numpy(dtype=float)
            on_grid = np.isin(event_times, milestones)
            concentration = float(on_grid.mean()) if len(event_times) else 0.0
            concentrations[cohort][endpoint] = concentration
            values, counts = np.unique(event_times, return_counts=True)
            order = np.argsort(counts)[::-1][:10]
            cohort_report["endpoints"][endpoint] = {
                "events": int(len(event_times)),
                "milestone_concentration": concentration,
                "top_event_times": [
                    {"minute": float(values[index]), "count": int(counts[index])} for index in order
                ],
            }
        report["cohorts"][cohort] = cohort_report
    differences = {
        endpoint: abs(concentrations["source"][endpoint] - concentrations["target"][endpoint])
        for endpoint in ENDPOINTS
    }
    report["milestone_concentration_difference"] = differences
    report["cox_primary_eligible"] = max(differences.values()) <= 0.20
    report["cox_role"] = "candidate_primary" if report["cox_primary_eligible"] else "sensitivity_only"
    return report


def _patient_session_weights(frame: pd.DataFrame, patient_col: str) -> np.ndarray:
    """Return per-session weight w_i = 1 / (patient's total session count).

    Matches the patient-balanced weighting declared in §2.6 of the manuscript:
    each unique patient contributes exactly 1.0 to the total training weight,
    regardless of how many sessions they represent.
    """
    counts = frame.groupby(patient_col, sort=False).transform("size").to_numpy(dtype=np.float32)
    if np.any(counts <= 0):
        raise ValueError("Session counts per patient must be strictly positive.")
    return 1.0 / counts


def _person_period(
    frame: pd.DataFrame,
    features: list[str],
    endpoint: str,
    interval: int,
    bins: int,
    patient_col: str | None = None,
):
    """Expand a session-level frame into person-period rows for discrete survival.

    When *patient_col* is provided the function also returns a per-row sample weight
    vector that enforces patient-level balance (w_i / number_of_risk_bins_for_session_i,
    so each patient still sums to exactly 1.0 across all of their expanded rows).

    NOTE: This keeps the person-period representation but aligns the weighting
    scheme with the session-level masked_survival_nll used in the production path.
    """
    columns = ENDPOINTS[endpoint]
    base = frame[features].to_numpy(dtype=np.float32)
    event = frame[columns["event"]].to_numpy(dtype=int)
    event_time = frame[columns["time"]].to_numpy(dtype=float)
    duration = frame["duration_minutes"].to_numpy(dtype=float)
    if not np.isfinite(duration).all() or np.any(duration <= 0) or np.any((event == 1) & (event_time > duration)):
        raise ValueError(f"Invalid follow-up time for {endpoint} discrete-time construction.")
    event_bin = np.maximum(0, np.ceil(event_time / interval).astype(int) - 1)
    followup_bins = np.maximum(1, np.ceil(duration / interval).astype(int))
    session_weights = (
        _patient_session_weights(frame, patient_col)
        if patient_col is not None
        else np.ones(len(frame), dtype=np.float32)
    )
    rows_x: list[np.ndarray] = []
    rows_y: list[np.ndarray] = []
    rows_w: list[np.ndarray] = []
    for time_bin in range(bins):
        include = np.where(event == 1, time_bin <= event_bin, time_bin < followup_bins)
        include &= time_bin < bins
        indices = np.flatnonzero(include)
        time_features = np.zeros((len(indices), bins), dtype=np.float32)
        time_features[:, time_bin] = 1.0
        rows_x.append(np.concatenate((base[indices], time_features), axis=1))
        rows_y.append(((event[indices] == 1) & (event_bin[indices] == time_bin)).astype(np.float32))
        # Each session's total weight (sum over its risk bins) equals session_weights[i].
        denominator = np.where(event == 1, event_bin + 1, followup_bins).astype(np.float32)
        rows_w.append((session_weights[indices] / np.maximum(denominator[indices], 1.0)).astype(np.float32))
    x = np.concatenate(rows_x)
    y = np.concatenate(rows_y)
    w = np.concatenate(rows_w)
    # Renormalize so person-period weight sum equals the number of unique patients
    # (equivalent to the session-level production loss scaling when denominator is
    # session_weights.sum()).  Otherwise absolute loss magnitude depends on bin count.
    unique_patients = (
        float(frame[patient_col].nunique())
        if patient_col is not None
        else float(len(frame))
    )
    denom = float(w.sum())
    if denom <= 0:
        raise ValueError("Total patient-balanced weight must be positive.")
    w_scaled = w * (unique_patients / denom)
    return x, y, w_scaled


def _all_periods(frame: pd.DataFrame, features: list[str], bins: int) -> np.ndarray:
    base = frame[features].to_numpy(dtype=np.float32)
    repeated = np.repeat(base, bins, axis=0)
    time_features = np.tile(np.eye(bins, dtype=np.float32), (len(base), 1))
    return np.concatenate((repeated, time_features), axis=1)


def _predict_cumulative(model: DiscreteHazardCDAN, frame, features, bins, device) -> np.ndarray:
    periods = _all_periods(frame, features, bins)
    probabilities = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(periods), 8192):
            x = torch.as_tensor(periods[start : start + 8192], device=device)
            probabilities.append(torch.sigmoid(model(x)[0]).cpu().numpy())
    hazards = np.concatenate(probabilities).reshape(len(frame), bins)
    return 1 - np.cumprod(1 - hazards, axis=1)


def _loader(x: np.ndarray, y: np.ndarray, w: np.ndarray, batch_size: int, seed: int) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(x),
        torch.from_numpy(y),
        torch.from_numpy(w),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))


def _validation_auc(model, x_val, y_val, device) -> float:
    model.eval()
    with torch.no_grad():
        probability = torch.sigmoid(model(torch.as_tensor(x_val, device=device))[0]).cpu().numpy()
    return float(roc_auc_score(y_val, probability))


def _patient_balanced_bce_with_logits(logit: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy with per-row patient-balanced weights (no class pos_weight).

    Replaces the previous ``pos_weight``-based rebalancing which double-counted
    class-level adjustment and did not respect session-level patient balance.
    Reduces with a weighted mean so loss magnitude is independent of batch size.
    """
    if logit.shape != y.shape or y.shape != w.shape:
        raise ValueError("Patient-balanced BCE requires matching logit, target, weight shapes.")
    if torch.any(w < 0) or not torch.isfinite(w).all():
        raise ValueError("Patient-balanced weights must be finite and non-negative.")
    per_row = nn.functional.binary_cross_entropy_with_logits(logit, y, reduction="none")
    total = float(w.detach().sum().clamp_min(1e-12).cpu())
    return (per_row * w).sum() / total


def _fit_source(model, train, val, config, device, seed):
    x_train, y_train, w_train = train
    x_val, y_val = val[:2]
    loader = _loader(x_train, y_train, w_train, config["batch_size"], seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=1e-4)
    best_state, best_auc, remaining = copy.deepcopy(model.state_dict()), -np.inf, config["patience"]
    model.to(device)
    for _ in range(config["pretrain_epochs"]):
        model.train()
        for x, y, w in loader:
            x, y, w = x.to(device), y.to(device), w.to(device)
            optimizer.zero_grad()
            loss = _patient_balanced_bce_with_logits(model(x)[0], y, w)
            loss.backward()
            optimizer.step()
        auc = _validation_auc(model, x_val, y_val, device)
        if auc > best_auc:
            best_state, best_auc, remaining = copy.deepcopy(model.state_dict()), auc, config["patience"]
        else:
            remaining -= 1
            if remaining == 0:
                break
    model.load_state_dict(best_state)
    return model


def _fit_update(model, source, target, val, config, device, seed, adversarial_weight):
    x_source, y_source, w_source = source
    x_target, y_target, w_target = target
    x_val, y_val = val[:2]
    source_loader = _loader(x_source, y_source, w_source, config["batch_size"], seed)
    target_loader = _loader(x_target, y_target, w_target, config["batch_size"], seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["finetune_learning_rate"], weight_decay=1e-4)
    best_state, best_auc, remaining = copy.deepcopy(model.state_dict()), -np.inf, config["patience"]
    for epoch in range(config["update_epochs"]):
        model.train()
        source_iter = iter(source_loader)
        for x_t, y_t, w_t in target_loader:
            try:
                x_s, y_s, w_s = next(source_iter)
            except StopIteration:
                source_iter = iter(source_loader)
                x_s, y_s, w_s = next(source_iter)
            x_t, y_t, w_t = x_t.to(device), y_t.to(device), w_t.to(device)
            x_s, y_s, w_s = x_s.to(device), y_s.to(device), w_s.to(device)
            optimizer.zero_grad()
            coeff = min(1.0, (epoch + 1) / config["update_epochs"])
            t_logit, t_domain = model(x_t, coeff if adversarial_weight else None)
            s_logit, s_domain = model(x_s, coeff if adversarial_weight else None)
            # Domain mixing weight: use dataset-scaled blend proportional to effective
            # patient counts rather than fixed 0.5/0.5 (matches production path
            # commentary in §2.6 of the revised manuscript).
            n_patients_source = max(float(w_source.sum()), 1e-12)
            n_patients_target = max(float(w_target.sum()), 1e-12)
            alpha_target = n_patients_source / (n_patients_source + n_patients_target)
            alpha_source = 1.0 - alpha_target
            loss = alpha_target * _patient_balanced_bce_with_logits(t_logit, y_t, w_t)
            loss += alpha_source * config["source_replay_weight"] * _patient_balanced_bce_with_logits(
                s_logit, y_s, w_s
            )
            if adversarial_weight:
                domain_logits = torch.cat((s_domain, t_domain))
                domain_labels = torch.cat(
                    (
                        torch.zeros(len(s_domain), dtype=torch.long, device=device),
                        torch.ones(len(t_domain), dtype=torch.long, device=device),
                    )
                )
                loss += adversarial_weight * nn.functional.cross_entropy(domain_logits, domain_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        auc = _validation_auc(model, x_val, y_val, device)
        if auc > best_auc:
            best_state, best_auc, remaining = copy.deepcopy(model.state_dict()), auc, config["patience"]
        else:
            remaining -= 1
            if remaining == 0:
                break
    model.load_state_dict(best_state)
    return model


def _horizon_metrics(
    model, val_frame, test_frame, features, endpoint, config, patient_ids, device, validation_cohort
):
    bins = len(config["horizons_minutes"])
    val_risk = _predict_cumulative(model, val_frame, features, bins, device)
    test_risk = _predict_cumulative(model, test_frame, features, bins, device)
    columns = ENDPOINTS[endpoint]
    result, calibrated = {}, np.zeros_like(test_risk)
    for index, horizon in enumerate(config["horizons_minutes"]):
        val_event = val_frame[columns["event"]].to_numpy(dtype=int)
        val_time = val_frame[columns["time"]].to_numpy(dtype=float)
        val_duration = val_frame["duration_minutes"].to_numpy(dtype=float)
        val_eligible = ((val_event == 1) & (val_time <= horizon)) | (val_duration >= horizon)
        y_val = ((val_event == 1) & (val_time <= horizon)).astype(int)
        calibrator = fit_probability_calibrator(y_val[val_eligible], val_risk[val_eligible, index])
        calibrated[:, index] = calibrate_probability(calibrator, test_risk[:, index])
        event = test_frame[columns["event"]].to_numpy(dtype=int)
        time = test_frame[columns["time"]].to_numpy(dtype=float)
        duration = test_frame["duration_minutes"].to_numpy(dtype=float)
        eligible = ((event == 1) & (time <= horizon)) | (duration >= horizon)
        y_test = ((event == 1) & (time <= horizon)).astype(int)
        threshold = select_youden_threshold(y_val[val_eligible], calibrate_probability(calibrator, val_risk[val_eligible, index]))
        result[str(horizon)] = evaluate_binary(
            y_test[eligible],
            calibrated[eligible, index],
            patient_ids[eligible],
            threshold=threshold,
            n_bootstrap=config["bootstrap_replicates"],
            seed=config["bootstrap_seed"],
        )
        result[str(horizon)]["calibration"] = {
            "method": "Platt scaling",
            "intercept": float(calibrator.intercept_[0]),
            "slope": float(calibrator.coef_[0, 0]),
            "validation_cohort": validation_cohort,
        }
    return result, calibrated


def _survival(frame: pd.DataFrame, endpoint: str):
    columns = ENDPOINTS[endpoint]
    event = frame[columns["event"]].to_numpy(dtype=np.float32)
    event_time = frame[columns["time"]].to_numpy(dtype=np.float32)
    duration = frame["duration_minutes"].to_numpy(dtype=np.float32)
    time = np.where(event == 1, event_time, duration)
    if np.any(time < 0) or np.any((event == 1) & (event_time > duration)):
        raise ValueError(f"Invalid Cox follow-up for {endpoint}.")
    return event, time


def _fit_cox(x, event, time, penalizer, epochs):
    features = torch.as_tensor(x)
    events = torch.as_tensor(event)
    times = torch.as_tensor(time)
    beta = nn.Parameter(torch.zeros(features.shape[1]))
    optimizer = torch.optim.Adam((beta,), lr=0.03)
    order = torch.argsort(times, descending=True)
    ordered_time = times[order]
    _, tie_counts = torch.unique_consecutive(ordered_time, return_counts=True)
    tie_ends = torch.cumsum(tie_counts, dim=0) - 1
    for _ in range(epochs):
        optimizer.zero_grad()
        risk = features @ beta
        ordered_risk, ordered_event = risk[order], events[order]
        log_risk_set = torch.repeat_interleave(torch.logcumsumexp(ordered_risk, dim=0)[tie_ends], tie_counts)
        loss = -((ordered_risk - log_risk_set) * ordered_event).sum()
        loss = loss / ordered_event.sum().clamp_min(1) + penalizer * beta.square().sum()
        loss.backward()
        optimizer.step()
    return beta.detach().numpy()


def _cox_analysis(bundle, endpoint, config):
    features = bundle.feature_names
    x_test = bundle.target_test[features].to_numpy(dtype=np.float32)
    e_test, t_test = _survival(bundle.target_test, endpoint)
    patient_ids = bundle.target_test[config["patient_col"]].astype(str).to_numpy()

    def fit_and_evaluate(train_frame, val_frame):
        x_train = train_frame[features].to_numpy(dtype=np.float32)
        x_val = val_frame[features].to_numpy(dtype=np.float32)
        e_train, t_train = _survival(train_frame, endpoint)
        e_val, t_val = _survival(val_frame, endpoint)
        candidates = []
        for penalizer in config["cox_penalizers"]:
            beta = _fit_cox(x_train, e_train, t_train, penalizer, config["cox_epochs"])
            candidates.append((concordance_index(t_val, x_val @ beta, e_val), penalizer, beta))
        val_cindex, penalizer, beta = max(candidates, key=lambda item: item[0])
        risk = x_test @ beta
        lower, upper = compute_bootstrap_cindex(
            t_test,
            e_test,
            risk,
            patient_ids=patient_ids,
            n_bootstrap=config["bootstrap_replicates"],
            seed=config["bootstrap_seed"],
        )
        return {
            "selected_penalizer": penalizer,
            "validation_cindex": float(val_cindex),
            "test_cindex": float(concordance_index(t_test, risk, e_test)),
            "patient_bootstrap_ci": [lower, upper],
            "feature_names": features,
            "coefficients": beta.tolist(),
        }

    return {
        "role": "sensitivity_analysis",
        "source_cox": fit_and_evaluate(bundle.source_train, bundle.source_val),
        "local_cox": fit_and_evaluate(bundle.target_train, bundle.target_val),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "conf" / "binary_config.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    if not args.train and not args.audit_only:
        raise SystemExit("Time-model training was not started; pass --train explicitly.")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data = config["data"]
    bundle = prepare_binary_data(
        ROOT / data["source_file"],
        ROOT / data["target_file"],
        allowlist_path=ROOT / data["feature_allowlist"],
        patient_col=data["patient_col"],
        split_seed=data["split_seed"],
        source_validation_fraction=data["source_validation_fraction"],
        target_update_fraction=data["target_update_fraction"],
        target_validation_fraction=data["target_validation_fraction"],
    )
    model_config = config["time_models"] | {
        "bootstrap_replicates": config["evaluation"]["patient_bootstrap_replicates"],
        "bootstrap_seed": config["evaluation"]["bootstrap_seed"],
        "patient_col": data["patient_col"],
    }
    horizons = tuple(model_config["horizons_minutes"])
    if horizons != tuple(range(model_config["interval_minutes"], horizons[-1] + 1, model_config["interval_minutes"])):
        raise ValueError("Discrete horizons must be contiguous multiples of interval_minutes.")
    audit = measurement_audit(bundle, horizons)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "time_measurement_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.audit_only:
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bins = len(horizons)
    all_results = {"measurement_audit": audit, "cox": {}, "seeds": []}
    for seed in config["training"]["initialization_seeds"]:
        seed_everything(seed)
        seed_result = {"seed": seed, "endpoints": {}}
        seed_states = {}
        prediction_frame = bundle.target_test[
            ["session_id", data["patient_col"], *[column for item in ENDPOINTS.values() for column in item.values()]]
        ].copy()
        for endpoint in ENDPOINTS:
            source_train = _person_period(
                bundle.source_train, bundle.feature_names, endpoint, model_config["interval_minutes"], bins, data["patient_col"]
            )
            source_val = _person_period(
                bundle.source_val, bundle.feature_names, endpoint, model_config["interval_minutes"], bins, data["patient_col"]
            )
            target_train = _person_period(
                bundle.target_train, bundle.feature_names, endpoint, model_config["interval_minutes"], bins, data["patient_col"]
            )
            target_val = _person_period(
                bundle.target_val, bundle.feature_names, endpoint, model_config["interval_minutes"], bins, data["patient_col"]
            )
            source_model = DiscreteHazardCDAN(
                source_train[0].shape[1], tuple(model_config["hidden_dims"]), model_config["dropout"]
            )
            source_model = _fit_source(source_model, source_train, source_val, model_config, device, seed)
            plain = _fit_update(
                copy.deepcopy(source_model), source_train, target_train, target_val, model_config, device, seed, 0.0
            )
            cdan = _fit_update(
                copy.deepcopy(source_model),
                source_train,
                target_train,
                target_val,
                model_config,
                device,
                seed,
                model_config["cdan_adversarial_weight"],
            )
            patient_ids = bundle.target_test[data["patient_col"]].astype(str).to_numpy()
            source_metrics, source_probability = _horizon_metrics(
                source_model,
                bundle.source_val,
                bundle.target_test,
                bundle.feature_names,
                endpoint,
                model_config,
                patient_ids,
                device,
                "source_val",
            )
            plain_metrics, plain_probability = _horizon_metrics(
                plain,
                bundle.target_val,
                bundle.target_test,
                bundle.feature_names,
                endpoint,
                model_config,
                patient_ids,
                device,
                "target_val",
            )
            cdan_metrics, cdan_probability = _horizon_metrics(
                cdan,
                bundle.target_val,
                bundle.target_test,
                bundle.feature_names,
                endpoint,
                model_config,
                patient_ids,
                device,
                "target_val",
            )
            comparisons = {}
            columns = ENDPOINTS[endpoint]
            event = bundle.target_test[columns["event"]].to_numpy(dtype=int)
            time = bundle.target_test[columns["time"]].to_numpy(dtype=float)
            duration = bundle.target_test["duration_minutes"].to_numpy(dtype=float)
            for index, horizon in enumerate(horizons):
                eligible = ((event == 1) & (time <= horizon)) | (duration >= horizon)
                y = ((event == 1) & (time <= horizon)).astype(int)
                comparisons[str(horizon)] = paired_patient_bootstrap_delta(
                    y[eligible],
                    cdan_probability[eligible, index],
                    plain_probability[eligible, index],
                    patient_ids[eligible],
                    n_bootstrap=model_config["bootstrap_replicates"],
                    seed=model_config["bootstrap_seed"],
                )
                prediction_frame[f"{endpoint}_plain_{horizon}m"] = plain_probability[:, index]
                prediction_frame[f"{endpoint}_cdan_{horizon}m"] = cdan_probability[:, index]
                prediction_frame[f"{endpoint}_source_{horizon}m"] = source_probability[:, index]
            seed_result["endpoints"][endpoint] = {
                "source_discrete": source_metrics,
                "plain_target_update": plain_metrics,
                "cdan_target_update": cdan_metrics,
                "cdan_minus_plain": comparisons,
            }
            seed_states[endpoint] = {
                "source_discrete": {name: value.detach().cpu() for name, value in source_model.state_dict().items()},
                "plain_target_update": {name: value.detach().cpu() for name, value in plain.state_dict().items()},
                "cdan_target_update": {name: value.detach().cpu() for name, value in cdan.state_dict().items()},
            }
        prediction_frame.to_csv(args.output / f"time_predictions_seed_{seed}.csv", index=False)
        torch.save(seed_states, args.output / f"time_models_seed_{seed}.pt")
        all_results["seeds"].append(seed_result)
        (args.output / "time_model_results.partial.json").write_text(
            json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    for endpoint in ENDPOINTS:
        all_results["cox"][endpoint] = _cox_analysis(bundle, endpoint, model_config)
        (args.output / "time_model_results.partial.json").write_text(
            json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    multiseed_summary = {}
    for endpoint in ENDPOINTS:
        multiseed_summary[endpoint] = {}
        for horizon in horizons:
            horizon_key = str(horizon)
            multiseed_summary[endpoint][horizon_key] = {}
            for model in ("source_discrete", "plain_target_update", "cdan_target_update"):
                values = [
                    seed_result["endpoints"][endpoint][model][horizon_key]["roc_auc"]
                    for seed_result in all_results["seeds"]
                ]
                multiseed_summary[endpoint][horizon_key][model] = {
                    "roc_auc_mean": float(np.mean(values)),
                    "roc_auc_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "values": values,
                }
            deltas = [
                seed_result["endpoints"][endpoint]["cdan_minus_plain"][horizon_key]["delta_a_minus_b"]
                for seed_result in all_results["seeds"]
            ]
            multiseed_summary[endpoint][horizon_key]["cdan_minus_plain"] = {
                "roc_auc_delta_mean": float(np.mean(deltas)),
                "roc_auc_delta_std": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
                "values": deltas,
            }
    all_results["multiseed_summary"] = multiseed_summary
    (args.output / "time_model_results.json").write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
