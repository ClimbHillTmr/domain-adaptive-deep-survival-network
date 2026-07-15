"""Leakage-safe data preparation for session-level binary IDH prediction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ALLOWLIST = ROOT / "experiments" / "audit" / "feature_allowlist.csv"
CATEGORICAL_COLUMNS = ("抗凝剂类型", "透析方式", "瘘管类型", "瘘管位置")
ENDPOINTS = {
    "idh": {"event": "idh_event", "time": "idh_time_min"},
    "ih": {"event": "ih_event", "time": "ih_time_min"},
}
OUTCOME_COLUMNS = {
    "events",
    "et_min",
    "event_time_min",
    "透中低血压_计算",
    "透中高血压_计算",
    "idh_event",
    "idh_time_min",
    "ih_event",
    "ih_time_min",
    "降幅时间点",
    "降幅时间点比值",
    "降幅时间点比值区间",
    "降幅时间点差值",
    "降幅时间点差值区间",
}


@dataclass
class BinaryDataBundle:
    source_train: pd.DataFrame
    source_val: pd.DataFrame
    target_train: pd.DataFrame
    target_val: pd.DataFrame
    target_test: pd.DataFrame
    feature_names: list[str]
    preprocessing: dict[str, Any]
    split_manifest: dict[str, Any]

    def arrays(self, frame: pd.DataFrame, outcome_col: str) -> tuple[np.ndarray, np.ndarray]:
        return (
            frame[self.feature_names].to_numpy(dtype=np.float32),
            frame[outcome_col].to_numpy(dtype=np.float32),
        )


def load_feature_allowlist(path: str | Path = DEFAULT_ALLOWLIST) -> list[str]:
    allowlist_path = Path(path)
    if not allowlist_path.exists():
        raise FileNotFoundError(f"Missing feature allowlist: {allowlist_path}")
    table = pd.read_csv(allowlist_path)
    required = {"feature", "allowed", "available_at_prediction_time"}
    if not required.issubset(table.columns):
        raise ValueError(f"Feature allowlist is missing columns: {sorted(required - set(table.columns))}")
    selected = table.loc[
        table["allowed"].astype(str).str.lower().eq("yes")
        & table["available_at_prediction_time"].astype(str).str.lower().eq("start"),
        "feature",
    ].astype(str).tolist()
    forbidden = sorted(set(selected) & OUTCOME_COLUMNS)
    if forbidden:
        raise ValueError(f"Outcome-derived features are marked as allowed: {forbidden}")
    if not selected:
        raise ValueError("Feature allowlist selects no start-time features.")
    return selected


def _validate_binary_cohort(frame: pd.DataFrame, name: str, patient_col: str) -> pd.DataFrame:
    endpoint_columns = {column for endpoint in ENDPOINTS.values() for column in endpoint.values()}
    required = {patient_col, "session_id", *endpoint_columns}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required binary-contract columns: {missing}")

    out = frame.copy()
    for column in endpoint_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    if out[[*endpoint_columns, patient_col, "session_id"]].isna().any().any():
        raise ValueError(f"{name} contains missing outcome, patient, or session identifiers.")
    for endpoint, columns in ENDPOINTS.items():
        event_col, time_col = columns["event"], columns["time"]
        if not out[event_col].isin([0, 1]).all():
            raise ValueError(f"{name} {event_col} must contain only 0 and 1.")
        if (out[time_col] < 0).any():
            raise ValueError(f"{name} contains negative {time_col} values.")
        invalid_non_events = out[event_col].eq(0) & out[time_col].ne(0)
        if invalid_non_events.any():
            raise ValueError(
                f"{name} contains {int(invalid_non_events.sum())} {endpoint} non-events whose time is not zero."
            )
    if out["session_id"].astype(str).duplicated().any():
        raise ValueError(f"{name} contains duplicate session_id values.")
    for clinical_col, event_col in (("透中低血压_计算", "idh_event"), ("透中高血压_计算", "ih_event")):
        if clinical_col in out:
            clinical_event = pd.to_numeric(out[clinical_col], errors="coerce")
            mismatch = clinical_event.notna() & clinical_event.ne(out[event_col])
            if mismatch.any():
                raise ValueError(f"{name} has {int(mismatch.sum())} {event_col}/{clinical_col} mismatches.")
    out[patient_col] = out[patient_col].astype(str)
    out["session_id"] = out["session_id"].astype(str)
    for columns in ENDPOINTS.values():
        out[columns["event"]] = out[columns["event"]].astype(int)
    return out


def _stratify_or_none(labels: np.ndarray) -> np.ndarray | None:
    _, counts = np.unique(labels, return_counts=True)
    return labels if len(counts) == 2 and counts.min() >= 2 else None


def _patient_labels(frame: pd.DataFrame, patient_col: str) -> pd.Series:
    event_columns = [columns["event"] for columns in ENDPOINTS.values()]
    patient_events = frame.groupby(patient_col, sort=True)[event_columns].max()
    return patient_events.astype(str).agg("|".join, axis=1)


def _split_patients(
    frame: pd.DataFrame,
    patient_col: str,
    holdout_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    patient_event = _patient_labels(frame, patient_col)
    if len(patient_event) < 5:
        raise ValueError("At least five patients are required for a patient-level split.")
    fit, holdout = train_test_split(
        patient_event.index.to_numpy(),
        test_size=holdout_fraction,
        random_state=seed,
        stratify=_stratify_or_none(patient_event.to_numpy()),
    )
    return np.asarray(fit), np.asarray(holdout)


def _rows_for_patients(frame: pd.DataFrame, patient_col: str, patients: np.ndarray) -> pd.DataFrame:
    return frame.loc[frame[patient_col].isin(patients)].copy().reset_index(drop=True)


def _target_split(
    frame: pd.DataFrame,
    patient_col: str,
    update_fraction: float,
    validation_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0 < update_fraction < 1:
        raise ValueError("target_update_fraction must be in (0, 1).")
    if not 0 < validation_fraction < 1:
        raise ValueError("target_validation_fraction must be in (0, 1).")
    update_patients, test_patients = _split_patients(
        frame, patient_col, holdout_fraction=1 - update_fraction, seed=seed
    )
    update_frame = _rows_for_patients(frame, patient_col, update_patients)
    train_patients, val_patients = _split_patients(
        update_frame, patient_col, holdout_fraction=validation_fraction, seed=seed + 1
    )
    return (
        _rows_for_patients(frame, patient_col, train_patients),
        _rows_for_patients(frame, patient_col, val_patients),
        _rows_for_patients(frame, patient_col, test_patients),
    )


def _apply_source_categories(source_train: pd.DataFrame, frames: list[pd.DataFrame]) -> dict[str, Any]:
    mappings: dict[str, dict[str, int]] = {}
    for column in CATEGORICAL_COLUMNS:
        code_column = f"{column}_code"
        if column not in source_train.columns:
            continue
        source_values = source_train[column].fillna("__MISSING__").astype(str).str.strip()
        mapping = {value: index for index, value in enumerate(sorted(source_values.unique()))}
        mappings[column] = mapping
        for frame in frames:
            if column in frame.columns:
                values = frame[column].fillna("__MISSING__").astype(str).str.strip()
                frame[code_column] = values.map(mapping).fillna(len(mapping)).astype(float)
    return {"categorical_mappings": mappings, "unknown_code": "len(mapping)"}


def _fit_source_transform(
    source_train: pd.DataFrame,
    frames: list[pd.DataFrame],
    features: list[str],
) -> dict[str, Any]:
    metadata = _apply_source_categories(source_train, frames)
    medians: dict[str, float] = {}
    means: dict[str, float] = {}
    scales: dict[str, float] = {}
    for column in features:
        for frame in frames:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        median = float(source_train[column].median())
        if not np.isfinite(median):
            median = 0.0
        for frame in frames:
            frame[column] = frame[column].fillna(median)
        mean = float(source_train[column].mean())
        scale = float(source_train[column].std(ddof=0))
        if not np.isfinite(scale) or scale == 0:
            scale = 1.0
        for frame in frames:
            frame[column] = (frame[column] - mean) / scale
        medians[column], means[column], scales[column] = median, mean, scale
    metadata.update({"fit_cohort": "source_train", "medians": medians, "means": means, "scales": scales})
    return metadata


def _split_summary(frame: pd.DataFrame, patient_col: str) -> dict[str, Any]:
    summary = {
        "sessions": int(len(frame)),
        "patients": int(frame[patient_col].nunique()),
        "patient_ids": sorted(frame[patient_col].unique().tolist()),
    }
    for endpoint, columns in ENDPOINTS.items():
        summary[f"{endpoint}_events"] = int(frame[columns["event"]].sum())
        summary[f"{endpoint}_event_rate"] = float(frame[columns["event"]].mean())
    return summary


def prepare_binary_data(
    source_path: str | Path,
    target_path: str | Path,
    *,
    allowlist_path: str | Path = DEFAULT_ALLOWLIST,
    patient_col: str = "患者id",
    split_seed: int = 42,
    source_validation_fraction: float = 0.1,
    target_update_fraction: float = 0.4,
    target_validation_fraction: float = 0.15,
) -> BinaryDataBundle:
    """Prepare fixed patient-level folds and source-fitted start-time features."""
    source = _validate_binary_cohort(pd.read_csv(source_path, low_memory=False), "source", patient_col)
    target = _validate_binary_cohort(pd.read_csv(target_path, low_memory=False), "target", patient_col)
    features = load_feature_allowlist(allowlist_path)

    def missing_features(frame: pd.DataFrame) -> list[str]:
        missing = []
        for feature in features:
            raw_category = feature.removesuffix("_code")
            generated_category = feature.endswith("_code") and raw_category in CATEGORICAL_COLUMNS
            if feature not in frame.columns and not (generated_category and raw_category in frame.columns):
                missing.append(feature)
        return sorted(missing)

    missing_source = missing_features(source)
    missing_target = missing_features(target)
    if missing_source or missing_target:
        raise ValueError(f"Allowed features missing from cohorts: source={missing_source}, target={missing_target}")

    source_train_patients, source_val_patients = _split_patients(
        source, patient_col, holdout_fraction=source_validation_fraction, seed=split_seed
    )
    source_train = _rows_for_patients(source, patient_col, source_train_patients)
    source_val = _rows_for_patients(source, patient_col, source_val_patients)
    target_train, target_val, target_test = _target_split(
        target,
        patient_col,
        update_fraction=target_update_fraction,
        validation_fraction=target_validation_fraction,
        seed=split_seed,
    )

    folds = [source_train, source_val, target_train, target_val, target_test]
    preprocessing = _fit_source_transform(source_train, folds, features)
    manifest = {
        "split_seed": split_seed,
        "patient_col": patient_col,
        "source_train": _split_summary(source_train, patient_col),
        "source_val": _split_summary(source_val, patient_col),
        "target_train": _split_summary(target_train, patient_col),
        "target_val": _split_summary(target_val, patient_col),
        "target_test": _split_summary(target_test, patient_col),
    }
    target_sets = [set(manifest[name]["patient_ids"]) for name in ("target_train", "target_val", "target_test")]
    if any(target_sets[i] & target_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise AssertionError("Target patient split overlap detected.")
    return BinaryDataBundle(*folds, features, preprocessing, manifest)


def write_binary_metadata(bundle: BinaryDataBundle, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "split_manifest.json").write_text(
        json.dumps(bundle.split_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "preprocessing.json").write_text(
        json.dumps(bundle.preprocessing, ensure_ascii=False, indent=2), encoding="utf-8"
    )
