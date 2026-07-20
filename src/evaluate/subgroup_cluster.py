"""Patient-clustered subgroup inference for prespecified binary strata."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def assign_prespecified_levels(values: pd.Series, definition: dict[str, Any]) -> pd.Series:
    """Apply an explicit cutpoint or explicit categorical value sets."""
    kind = definition.get("type")
    if kind == "numeric_cutpoint":
        cutpoint = definition.get("cutpoint")
        labels = definition.get("labels")
        if not isinstance(cutpoint, (int, float)) or not isinstance(labels, list) or len(labels) != 2:
            raise ValueError("numeric_cutpoint requires a numeric cutpoint and exactly two labels")
        numeric = pd.to_numeric(values, errors="coerce")
        assigned = pd.Series(pd.NA, index=values.index, dtype="object")
        assigned.loc[numeric.notna() & numeric.lt(float(cutpoint))] = str(labels[0])
        assigned.loc[numeric.notna() & numeric.ge(float(cutpoint))] = str(labels[1])
        return assigned
    if kind == "categorical_sets":
        levels = definition.get("levels")
        if not isinstance(levels, dict) or len(levels) != 2:
            raise ValueError("categorical_sets requires exactly two named levels")
        assigned = pd.Series(pd.NA, index=values.index, dtype="object")
        normalized = values.astype(str)
        seen: set[str] = set()
        for label, accepted_values in levels.items():
            if not isinstance(accepted_values, list) or not accepted_values:
                raise ValueError("Each categorical level requires a non-empty value list")
            accepted = {str(value) for value in accepted_values}
            if seen & accepted:
                raise ValueError("Categorical subgroup value sets overlap")
            seen |= accepted
            assigned.loc[normalized.isin(accepted)] = str(label)
        return assigned
    raise ValueError("Subgroup definition type must be numeric_cutpoint or categorical_sets")


def _auc_delta(y: np.ndarray, updated: np.ndarray, source: np.ndarray) -> tuple[float, float]:
    if len(np.unique(y)) < 2:
        raise ValueError("Subgroup must contain both outcome classes")
    updated_auc = float(roc_auc_score(y, updated))
    return updated_auc, updated_auc - float(roc_auc_score(y, source))


def patient_cluster_subgroup_analysis(
    frame: pd.DataFrame,
    *,
    level_col: str,
    outcome_col: str,
    updated_probability_col: str,
    source_probability_col: str,
    patient_col: str,
    levels: list[str],
    minimum_patients: int,
    replicates: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Estimate level AUC/delta and a two-sided bootstrap interaction p-value."""
    if len(levels) != 2 or len(set(levels)) != 2:
        raise ValueError("Interaction analysis requires exactly two distinct subgroup levels")
    required = {level_col, outcome_col, updated_probability_col, source_probability_col, patient_col}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Subgroup frame is missing columns: {missing}")
    analysis = frame.loc[frame[level_col].isin(levels)].copy()
    analysis[patient_col] = analysis[patient_col].astype(str)
    point: dict[str, dict[str, Any]] = {}
    for level in levels:
        subset = analysis.loc[analysis[level_col].eq(level)]
        n_patients = int(subset[patient_col].nunique())
        if n_patients < minimum_patients:
            raise ValueError(
                f"Subgroup level {level!r} has {n_patients} patients; minimum is {minimum_patients}"
            )
        auc, delta = _auc_delta(
            subset[outcome_col].to_numpy(dtype=int),
            subset[updated_probability_col].to_numpy(dtype=float),
            subset[source_probability_col].to_numpy(dtype=float),
        )
        point[level] = {
            "n_patients": n_patients,
            "n_sessions": int(len(subset)),
            "n_events": int(subset[outcome_col].sum()),
            "subgroup_auc": auc,
            "delta_auc": delta,
        }

    patients = np.asarray(sorted(analysis[patient_col].unique()))
    patient_rows = {
        patient: np.flatnonzero(analysis[patient_col].to_numpy() == patient) for patient in patients
    }
    rng = np.random.default_rng(seed)
    auc_samples = {level: [] for level in levels}
    delta_samples = {level: [] for level in levels}
    interaction_samples: list[float] = []
    for _ in range(replicates):
        sampled_patients = rng.choice(patients, len(patients), replace=True)
        sampled = analysis.iloc[
            np.concatenate([patient_rows[patient] for patient in sampled_patients])
        ]
        replicate_values: dict[str, tuple[float, float]] = {}
        valid = True
        for level in levels:
            subset = sampled.loc[sampled[level_col].eq(level)]
            try:
                auc, delta = _auc_delta(
                    subset[outcome_col].to_numpy(dtype=int),
                    subset[updated_probability_col].to_numpy(dtype=float),
                    subset[source_probability_col].to_numpy(dtype=float),
                )
            except ValueError:
                valid = False
                break
            replicate_values[level] = (auc, delta)
        if valid:
            for level, (auc, delta) in replicate_values.items():
                auc_samples[level].append(auc)
                delta_samples[level].append(delta)
            interaction_samples.append(
                replicate_values[levels[1]][1] - replicate_values[levels[0]][1]
            )

    if not interaction_samples:
        raise ValueError("No valid patient-cluster bootstrap interaction replicates")
    interaction_array = np.asarray(interaction_samples)
    interaction_p = min(
        1.0,
        2 * min(
            (np.count_nonzero(interaction_array <= 0) + 1) / (len(interaction_array) + 1),
            (np.count_nonzero(interaction_array >= 0) + 1) / (len(interaction_array) + 1),
        ),
    )
    rows = []
    for level in levels:
        auc_values = np.asarray(auc_samples[level])
        delta_values = np.asarray(delta_samples[level])
        rows.append({
            "subgroup_level": level,
            **point[level],
            "subgroup_auc_lower": float(np.quantile(auc_values, 0.025)),
            "subgroup_auc_upper": float(np.quantile(auc_values, 0.975)),
            "delta_auc_lower": float(np.quantile(delta_values, 0.025)),
            "delta_auc_upper": float(np.quantile(delta_values, 0.975)),
            "interaction_contrast": f"{levels[1]}_minus_{levels[0]}",
            "interaction_delta_auc": point[levels[1]]["delta_auc"] - point[levels[0]]["delta_auc"],
            "interaction_p_value": float(interaction_p),
            "bootstrap_valid_replicates": len(interaction_samples),
        })
    return rows
