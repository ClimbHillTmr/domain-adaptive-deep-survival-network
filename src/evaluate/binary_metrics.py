"""Patient-clustered evaluation for binary IDH predictions."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve


def expected_calibration_error(
    y_true: np.ndarray, probability: np.ndarray, bins: int = 10
) -> float:
    """Session-weighted absolute calibration error in quantile probability bins."""
    outcome = np.asarray(y_true, dtype=float)
    probability = np.asarray(probability, dtype=float)
    order = np.argsort(probability, kind="stable")
    groups = np.array_split(order, min(bins, len(order)))
    return float(sum(
        len(indices) / len(order)
        * abs(float(outcome[indices].mean()) - float(probability[indices].mean()))
        for indices in groups if len(indices)
    ))


def _metric_functions() -> dict[str, Callable[[np.ndarray, np.ndarray], float]]:
    return {
        "roc_auc": roc_auc_score,
        "pr_auc": average_precision_score,
        "brier": brier_score_loss,
        "ece_10_quantile_bins": expected_calibration_error,
    }


def select_youden_threshold(y_true: np.ndarray, probability: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true, probability)
    threshold = float(thresholds[np.argmax(true_positive_rate - false_positive_rate)])
    return threshold if np.isfinite(threshold) else 0.5


def evaluate_binary(
    y_true: np.ndarray,
    probability: np.ndarray,
    patient_ids: np.ndarray,
    *,
    threshold: float,
    n_bootstrap: int,
    seed: int,
) -> dict[str, object]:
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    patient_ids = np.asarray(patient_ids).astype(str)
    if not (len(y_true) == len(probability) == len(patient_ids)):
        raise ValueError("Outcome, probability, and patient ID lengths differ.")
    if len(np.unique(y_true)) < 2:
        raise ValueError("Binary evaluation requires both event classes.")

    metrics = {name: float(function(y_true, probability)) for name, function in _metric_functions().items()}
    predicted = probability >= threshold
    positive, negative = y_true == 1, y_true == 0
    metrics.update(
        {
            "sensitivity": float(predicted[positive].mean()),
            "specificity": float((~predicted[negative]).mean()),
            "threshold": float(threshold),
            "n_sessions": int(len(y_true)),
            "n_patients": int(len(np.unique(patient_ids))),
            "n_events": int(y_true.sum()),
        }
    )

    rng = np.random.default_rng(seed)
    patients = np.unique(patient_ids)
    samples: dict[str, list[float]] = {name: [] for name in _metric_functions()}
    for _ in range(n_bootstrap):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        indices = np.concatenate([np.flatnonzero(patient_ids == patient) for patient in sampled])
        if len(np.unique(y_true[indices])) < 2:
            continue
        for name, function in _metric_functions().items():
            samples[name].append(float(function(y_true[indices], probability[indices])))
    metrics["patient_cluster_bootstrap"] = {
        name: {
            "lower": float(np.quantile(values, 0.025)),
            "upper": float(np.quantile(values, 0.975)),
            "valid_replicates": len(values),
        }
        for name, values in samples.items()
        if values
    }
    return metrics


def paired_patient_bootstrap_delta(
    y_true: np.ndarray,
    probability_a: np.ndarray,
    probability_b: np.ndarray,
    patient_ids: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, object]:
    y_true = np.asarray(y_true, dtype=int)
    probability_a = np.asarray(probability_a, dtype=float)
    probability_b = np.asarray(probability_b, dtype=float)
    patient_ids = np.asarray(patient_ids).astype(str)
    patients = np.unique(patient_ids)
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        indices = np.concatenate([np.flatnonzero(patient_ids == patient) for patient in sampled])
        if len(np.unique(y_true[indices])) < 2:
            continue
        deltas.append(
            float(roc_auc_score(y_true[indices], probability_a[indices]))
            - float(roc_auc_score(y_true[indices], probability_b[indices]))
        )
    point = float(roc_auc_score(y_true, probability_a) - roc_auc_score(y_true, probability_b))
    return {
        "metric": "roc_auc",
        "delta_a_minus_b": point,
        "lower": float(np.quantile(deltas, 0.025)),
        "upper": float(np.quantile(deltas, 0.975)),
        "valid_replicates": len(deltas),
    }


def paired_patient_bootstrap_metric_deltas(
    y_true: np.ndarray,
    probability_a: np.ndarray,
    probability_b: np.ndarray,
    patient_ids: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, dict[str, float | int | str]]:
    """Paired patient-cluster contrasts for all locked confirmatory metrics."""
    y_true = np.asarray(y_true, dtype=int)
    probability_a = np.asarray(probability_a, dtype=float)
    probability_b = np.asarray(probability_b, dtype=float)
    patient_ids = np.asarray(patient_ids).astype(str)
    if not (len(y_true) == len(probability_a) == len(probability_b) == len(patient_ids)):
        raise ValueError("Outcome, probabilities, and patient ID lengths differ.")
    functions = _metric_functions()
    point = {
        name: float(function(y_true, probability_a) - function(y_true, probability_b))
        for name, function in functions.items()
    }
    samples: dict[str, list[float]] = {name: [] for name in functions}
    patients = np.unique(patient_ids)
    patient_rows = {patient: np.flatnonzero(patient_ids == patient) for patient in patients}
    rng = np.random.default_rng(seed)
    for _ in range(n_bootstrap):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        indices = np.concatenate([patient_rows[patient] for patient in sampled])
        if len(np.unique(y_true[indices])) < 2:
            continue
        for name, function in functions.items():
            samples[name].append(float(
                function(y_true[indices], probability_a[indices])
                - function(y_true[indices], probability_b[indices])
            ))
    return {
        name: {
            "delta_a_minus_b": point[name],
            "lower": float(np.quantile(samples[name], 0.025)),
            "upper": float(np.quantile(samples[name], 0.975)),
            "valid_replicates": len(samples[name]),
            "favorable_direction": "positive" if name in {"roc_auc", "pr_auc"} else "negative",
        }
        for name in functions
    }
