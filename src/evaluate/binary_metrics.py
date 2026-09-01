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


def holm_correction(p_values: list[float] | np.ndarray) -> np.ndarray:
    """Holm-Bonferroni step-down correction for multiple testing.

    Implements the Holm-Bonferroni procedure which is more powerful than
    the Bonferroni correction while still controlling the family-wise error
    rate. For a set of p-values, it adjusts each based on its rank.

    Args:
        p_values: Array of raw p-values (must be in [0, 1]).

    Returns:
        Array of adjusted p-values, with monotonicity enforced.
    """
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([min(p_values[0], 1.0)])

    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]

    adjusted = np.zeros(n)
    for i in range(n):
        rank = i + 1
        adjusted[sorted_indices[i]] = sorted_p[i] * (n - rank + 1)

    for i in range(n - 2, -1, -1):
        if adjusted[sorted_indices[i]] > adjusted[sorted_indices[i + 1]]:
            adjusted[sorted_indices[i]] = adjusted[sorted_indices[i + 1]]

    return np.minimum(adjusted, 1.0)


def paired_patient_bootstrap_co_primary(
    y_true_idh: np.ndarray,
    probability_idh_a: np.ndarray,
    probability_idh_b: np.ndarray,
    y_true_ih: np.ndarray,
    probability_ih_a: np.ndarray,
    probability_ih_b: np.ndarray,
    patient_ids: np.ndarray,
    *,
    n_bootstrap: int = 1000,
    seed: int = 20260715,
    alpha: float = 0.05,
) -> dict[str, object]:
    """Co-primary endpoint analysis with Holm correction.

    Performs paired patient-cluster bootstrap for both IDH and IH endpoints,
    then applies Holm-Bonferroni correction to control family-wise error rate.

    Args:
        y_true_idh: Ground truth labels for IDH endpoint.
        probability_idh_a: Predictions from strategy A for IDH.
        probability_idh_b: Predictions from strategy B for IDH.
        y_true_ih: Ground truth labels for IH endpoint.
        probability_ih_a: Predictions from strategy A for IH.
        probability_ih_b: Predictions from strategy B for IH.
        patient_ids: Patient identifiers for clustering.
        n_bootstrap: Number of bootstrap replicates.
        seed: Random seed for reproducibility.
        alpha: Significance threshold.

    Returns:
        Dictionary with point estimates, bootstrap CIs, raw p-values,
        Holm-adjusted p-values, and significance indicators.
    """
    y_true_idh = np.asarray(y_true_idh, dtype=int)
    probability_idh_a = np.asarray(probability_idh_a, dtype=float)
    probability_idh_b = np.asarray(probability_idh_b, dtype=float)
    y_true_ih = np.asarray(y_true_ih, dtype=int)
    probability_ih_a = np.asarray(probability_ih_a, dtype=float)
    probability_ih_b = np.asarray(probability_ih_b, dtype=float)
    patient_ids = np.asarray(patient_ids).astype(str)

    if not (len(y_true_idh) == len(y_true_ih) == len(patient_ids)):
        raise ValueError("Endpoint and patient ID lengths must match.")

    patients = np.unique(patient_ids)
    rng = np.random.default_rng(seed)

    idh_deltas: list[float] = []
    ih_deltas: list[float] = []

    for _ in range(n_bootstrap):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        indices = np.concatenate([np.flatnonzero(patient_ids == patient) for patient in sampled])

        if len(np.unique(y_true_idh[indices])) >= 2:
            idh_deltas.append(
                float(roc_auc_score(y_true_idh[indices], probability_idh_a[indices]))
                - float(roc_auc_score(y_true_idh[indices], probability_idh_b[indices]))
            )

        if len(np.unique(y_true_ih[indices])) >= 2:
            ih_deltas.append(
                float(roc_auc_score(y_true_ih[indices], probability_ih_a[indices]))
                - float(roc_auc_score(y_true_ih[indices], probability_ih_b[indices]))
            )

    idh_point = float(roc_auc_score(y_true_idh, probability_idh_a) - roc_auc_score(y_true_idh, probability_idh_b))
    ih_point = float(roc_auc_score(y_true_ih, probability_ih_a) - roc_auc_score(y_true_ih, probability_ih_b))

    idh_raw_p = float(np.mean([d <= 0 for d in idh_deltas])) if idh_deltas else 1.0
    ih_raw_p = float(np.mean([d <= 0 for d in ih_deltas])) if ih_deltas else 1.0

    raw_p = np.array([idh_raw_p, ih_raw_p])
    adjusted_p = holm_correction(raw_p)

    return {
        "idh": {
            "delta": idh_point,
            "ci_lower": float(np.quantile(idh_deltas, 0.025)) if idh_deltas else None,
            "ci_upper": float(np.quantile(idh_deltas, 0.975)) if idh_deltas else None,
            "raw_p": idh_raw_p,
            "adjusted_p": float(adjusted_p[0]),
            "n_bootstrap": len(idh_deltas),
            "significant_at_alpha": bool(adjusted_p[0] < alpha),
        },
        "ih": {
            "delta": ih_point,
            "ci_lower": float(np.quantile(ih_deltas, 0.025)) if ih_deltas else None,
            "ci_upper": float(np.quantile(ih_deltas, 0.975)) if ih_deltas else None,
            "raw_p": ih_raw_p,
            "adjusted_p": float(adjusted_p[1]),
            "n_bootstrap": len(ih_deltas),
            "significant_at_alpha": bool(adjusted_p[1] < alpha),
        },
        "alpha": alpha,
        "both_significant": bool(adjusted_p[0] < alpha and adjusted_p[1] < alpha),
        "correction_method": "Holm-Bonferroni",
    }
