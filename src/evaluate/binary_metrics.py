"""Patient-clustered evaluation for binary IDH predictions."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def _metric_functions() -> dict[str, Callable[[np.ndarray, np.ndarray], float]]:
    return {
        "roc_auc": roc_auc_score,
        "pr_auc": average_precision_score,
        "brier": brier_score_loss,
    }


def select_youden_threshold(y_true: np.ndarray, probability: np.ndarray) -> float:
    candidates = np.unique(np.r_[0.0, probability, 1.0])
    best_threshold, best_score = 0.5, -np.inf
    for threshold in candidates:
        predicted = probability >= threshold
        positive = y_true == 1
        negative = ~positive
        if not positive.any() or not negative.any():
            return 0.5
        sensitivity = float(predicted[positive].mean())
        specificity = float((~predicted[negative]).mean())
        score = sensitivity + specificity - 1
        if score > best_score:
            best_threshold, best_score = float(threshold), score
    return best_threshold


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
