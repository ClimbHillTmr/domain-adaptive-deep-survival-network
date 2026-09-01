"""Minimal evaluation utilities for four-bin discrete survival predictions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

DEFAULT_HORIZONS = np.array([60.0, 120.0, 180.0, 240.0])


def _as_one_dimensional(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite one-dimensional array.")
    return array


def hazard_probabilities(
    values: np.ndarray, *, input_type: str = "logits"
) -> np.ndarray:
    """Convert hazard logits or validate hazard probabilities."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] == 0 or not np.all(np.isfinite(array)):
        raise ValueError("Hazards must be a finite two-dimensional array.")
    if input_type == "logits":
        probabilities = np.empty_like(array)
        positive = array >= 0
        probabilities[positive] = 1.0 / (1.0 + np.exp(-array[positive]))
        exponential = np.exp(array[~positive])
        probabilities[~positive] = exponential / (1.0 + exponential)
    elif input_type == "probabilities":
        if np.any((array < 0.0) | (array > 1.0)):
            raise ValueError("Hazard probabilities must lie in [0, 1].")
        probabilities = array.copy()
    else:
        raise ValueError("input_type must be 'logits' or 'probabilities'.")
    return probabilities


def cumulative_risk_from_hazards(
    values: np.ndarray, *, input_type: str = "logits"
) -> np.ndarray:
    """Return cumulative incidence 1 - product(1 - hazard) at each bin."""
    hazards = hazard_probabilities(values, input_type=input_type)
    risk = 1.0 - np.cumprod(1.0 - hazards, axis=1)
    if np.any(np.diff(risk, axis=1) < -1e-12):
        raise AssertionError("Cumulative risk must be non-decreasing across horizons.")
    return risk


@dataclass(frozen=True)
class ReverseKaplanMeier:
    """Right-continuous censoring survival with explicit left-limit lookup."""

    times: np.ndarray
    survival_before: np.ndarray
    survival_after: np.ndarray

    def g(self, time: float | np.ndarray) -> np.ndarray:
        """Return G(t), including censoring jumps occurring at t."""
        query = np.asarray(time, dtype=float)
        positions = np.searchsorted(self.times, query, side="right") - 1
        return np.where(positions >= 0, self.survival_after[np.maximum(positions, 0)], 1.0)

    def g_left(self, time: float | np.ndarray) -> np.ndarray:
        """Return G(t-), excluding a censoring jump occurring at t."""
        query = np.asarray(time, dtype=float)
        positions = np.searchsorted(self.times, query, side="left") - 1
        return np.where(positions >= 0, self.survival_after[np.maximum(positions, 0)], 1.0)


def fit_reverse_kaplan_meier(
    times: np.ndarray, event_observed: np.ndarray
) -> ReverseKaplanMeier:
    """Fit reverse KM, treating censoring as the event and failures as censored."""
    times = _as_one_dimensional(times, "times").astype(float)
    events = _as_one_dimensional(event_observed, "event_observed").astype(int)
    if len(times) != len(events) or np.any(times < 0) or np.any(~np.isin(events, [0, 1])):
        raise ValueError("Times and binary event indicators must have matching valid values.")
    if len(times) == 0:
        raise ValueError("Reverse Kaplan-Meier requires at least one observation.")

    unique_times = np.unique(times)
    before = np.empty(len(unique_times), dtype=float)
    after = np.empty(len(unique_times), dtype=float)
    survival = 1.0
    for index, time in enumerate(unique_times):
        before[index] = survival
        at_risk = np.count_nonzero(times >= time)
        censorings = np.count_nonzero((times == time) & (events == 0))
        survival *= 1.0 - censorings / at_risk
        after[index] = survival
    return ReverseKaplanMeier(unique_times, before, after)


def _ipcw_components(
    times: np.ndarray,
    event_observed: np.ndarray,
    risk: np.ndarray,
    horizon: float,
    *,
    min_g: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = _as_one_dimensional(times, "times").astype(float)
    events = _as_one_dimensional(event_observed, "event_observed").astype(int)
    risk = _as_one_dimensional(risk, "risk").astype(float)
    if not (len(times) == len(events) == len(risk)):
        raise ValueError("Times, event indicators, and risks must have equal lengths.")
    if np.any(times < 0) or np.any(~np.isin(events, [0, 1])):
        raise ValueError("Times must be non-negative and event indicators binary.")
    if np.any((risk < 0) | (risk > 1)):
        raise ValueError("Cumulative risks must lie in [0, 1].")
    if not np.isfinite(horizon) or horizon <= 0 or min_g <= 0:
        raise ValueError("horizon and min_g must be positive.")

    censoring = fit_reverse_kaplan_meier(times, events)
    cases = (events == 1) & (times <= horizon)
    controls_after = times > horizon
    controls_at_boundary = (events == 0) & (times == horizon)
    controls = controls_after | controls_at_boundary
    weights = np.zeros(len(times), dtype=float)
    if np.any(cases):
        denominators = censoring.g_left(times[cases])
        if np.any(denominators < min_g):
            raise ValueError("IPCW metric is not estimable: G(T-) is too close to zero.")
        weights[cases] = 1.0 / denominators
    if np.any(controls_after):
        denominator = float(censoring.g(horizon))
        if denominator < min_g:
            raise ValueError("IPCW metric is not estimable: G(t) is too close to zero.")
        weights[controls_after] = 1.0 / denominator
    if np.any(controls_at_boundary):
        denominator = float(censoring.g_left(horizon))
        if denominator < min_g:
            raise ValueError("IPCW metric is not estimable: G(t-) is too close to zero.")
        weights[controls_at_boundary] = 1.0 / denominator
    outcomes = cases.astype(int)
    return outcomes, weights, cases | controls


def ipcw_cumulative_dynamic_auc(
    times: np.ndarray,
    event_observed: np.ndarray,
    cumulative_risk: np.ndarray,
    horizon: float,
    *,
    min_g: float = 1e-8,
) -> float:
    """IPCW cumulative/dynamic AUC; censored-before-horizon rows have weight zero."""
    outcomes, weights, included = _ipcw_components(
        times, event_observed, cumulative_risk, horizon, min_g=min_g
    )
    if not np.any(outcomes[included] == 1) or not np.any(outcomes[included] == 0):
        raise ValueError("IPCW AUC is not estimable without both cases and controls.")
    return float(roc_auc_score(outcomes[included], np.asarray(cumulative_risk)[included], sample_weight=weights[included]))


def ipcw_brier_score(
    times: np.ndarray,
    event_observed: np.ndarray,
    cumulative_risk: np.ndarray,
    horizon: float,
    *,
    min_g: float = 1e-8,
) -> float:
    """IPCW Brier score with the standard 1/n normalization."""
    outcomes, weights, _ = _ipcw_components(
        times, event_observed, cumulative_risk, horizon, min_g=min_g
    )
    errors = (outcomes - np.asarray(cumulative_risk, dtype=float)) ** 2
    return float(np.sum(weights * errors) / len(errors))


def evaluate_discrete_survival(
    times: np.ndarray,
    event_observed: np.ndarray,
    cumulative_risk: np.ndarray,
    *,
    horizons: np.ndarray = DEFAULT_HORIZONS,
    min_g: float = 1e-8,
) -> dict[str, object]:
    """Evaluate horizon AUC/Brier and their arithmetic four-point means."""
    horizons = _as_one_dimensional(horizons, "horizons").astype(float)
    risk = np.asarray(cumulative_risk, dtype=float)
    if risk.ndim != 2 or risk.shape != (len(np.asarray(times)), len(horizons)):
        raise ValueError("Cumulative risk shape must be (n_sessions, n_horizons).")
    if np.any(np.diff(risk, axis=1) < -1e-12):
        raise ValueError("Cumulative risk must be non-decreasing across horizons.")
    auc = np.array([
        ipcw_cumulative_dynamic_auc(times, event_observed, risk[:, i], horizon, min_g=min_g)
        for i, horizon in enumerate(horizons)
    ])
    brier = np.array([
        ipcw_brier_score(times, event_observed, risk[:, i], horizon, min_g=min_g)
        for i, horizon in enumerate(horizons)
    ])
    return {
        "horizons": horizons.copy(),
        "auc": auc,
        "brier": brier,
        "mean_auc": float(auc.mean()),
        "mean_brier": float(brier.mean()),
    }


def patient_cluster_bootstrap_indices(
    patient_ids: np.ndarray, *, n_bootstrap: int, seed: int
) -> list[np.ndarray]:
    """Return paired row indices after sampling patients with replacement."""
    patient_ids = np.asarray(patient_ids)
    if patient_ids.ndim != 1:
        raise ValueError("patient_ids must be a one-dimensional array.")
    if n_bootstrap < 0:
        raise ValueError("n_bootstrap must be non-negative.")
    patients = np.unique(patient_ids)
    if len(patients) == 0:
        raise ValueError("At least one patient is required.")
    rows = {patient: np.flatnonzero(patient_ids == patient) for patient in patients}
    rng = np.random.default_rng(seed)
    return [
        np.concatenate([rows[patient] for patient in rng.choice(patients, len(patients), replace=True)])
        for _ in range(n_bootstrap)
    ]


def fit_shared_hazard_platt(
    hazard_logits: np.ndarray, targets: np.ndarray, observed_mask: np.ndarray
) -> tuple[float, float]:
    """Fit one Platt intercept/slope to all observed discrete-hazard cells."""
    logits = np.asarray(hazard_logits, dtype=float)
    targets = np.asarray(targets)
    mask = np.asarray(observed_mask, dtype=bool)
    if logits.ndim != 2 or targets.shape != logits.shape or mask.shape != logits.shape:
        raise ValueError("Logits, targets, and observed_mask must have the same 2-D shape.")
    if not np.all(np.isfinite(logits)) or np.any(~np.isin(targets[mask], [0, 1])):
        raise ValueError("Observed calibration values must be finite and binary.")
    labels = targets[mask].astype(int)
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        raise ValueError("Shared hazard calibration requires both observed classes.")
    model = LogisticRegression(C=1e12, solver="lbfgs", max_iter=1000)
    model.fit(logits[mask, None], labels)
    intercept = float(model.intercept_[0])
    slope = float(model.coef_[0, 0])
    if not np.isfinite(intercept) or not np.isfinite(slope) or slope <= 0:
        raise ValueError("Shared hazard calibration is invalid because fitted slope b is not positive.")
    return intercept, slope


def apply_shared_hazard_platt(
    hazard_logits: np.ndarray, intercept: float, slope: float
) -> np.ndarray:
    """Apply a positive-slope shared Platt map and return monotone cumulative risk."""
    if not np.isfinite(intercept) or not np.isfinite(slope) or slope <= 0:
        raise ValueError("Shared Platt calibration requires finite intercept and positive slope.")
    return cumulative_risk_from_hazards(
        intercept + slope * np.asarray(hazard_logits, dtype=float), input_type="logits"
    )
