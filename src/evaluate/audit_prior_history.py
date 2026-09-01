"""Audit that included history features use strictly prior patient sessions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data_pipeline.data_process import HISTORY_MEAN_COLUMNS


RATE_COLUMNS = ("history_IDH_rate", "history_HBP_rate", "history_LBP_times_0_rate")
BIN_RATE_COLUMNS = tuple(f"history_LBP_times_{index}_rate" for index in range(1, 5))


def _numeric_equal(actual: pd.Series, expected: pd.Series, tolerance: float) -> tuple[int, float]:
    actual_values = pd.to_numeric(actual, errors="coerce").to_numpy(dtype=float)
    expected_values = pd.to_numeric(expected, errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(actual_values) & np.isfinite(expected_values)
    missing_match = np.isnan(actual_values) & np.isnan(expected_values)
    differences = np.full(len(actual_values), np.inf)
    differences[finite] = np.abs(actual_values[finite] - expected_values[finite])
    differences[missing_match] = 0.0
    return int((differences > tolerance).sum()), float(np.max(differences)) if len(differences) else 0.0


def audit_prior_history(path: str | Path, *, patient_col: str = "患者id", tolerance: float = 1e-10) -> dict[str, Any]:
    path = Path(path)
    header = pd.read_csv(path, nrows=0).columns.tolist()
    time_column = "透析开始时间" if "透析开始时间" in header else "透析日期"
    required = {
        patient_col, "session_id", time_column, "idh_event", "ih_event", "降幅时间点比值区间",
        *HISTORY_MEAN_COLUMNS,
        *(f"历史平均{column}" for column in HISTORY_MEAN_COLUMNS),
        *RATE_COLUMNS, *BIN_RATE_COLUMNS,
    }
    missing = sorted(required - set(header))
    report: dict[str, Any] = {
        "path": str(path), "passed": False, "time_column": time_column,
        "missing_columns": missing, "feature_checks": {},
    }
    if missing:
        report["errors"] = ["missing_history_audit_columns"]
        return report
    frame = pd.read_csv(path, usecols=sorted(required), low_memory=False)
    frame[patient_col] = frame[patient_col].astype(str)
    frame["session_id"] = frame["session_id"].astype(str)
    frame[time_column] = pd.to_datetime(frame[time_column], errors="coerce")
    report.update({
        "sessions": int(len(frame)), "patients": int(frame[patient_col].nunique()),
        "missing_session_times": int(frame[time_column].isna().sum()),
        "duplicate_session_ids": int(frame["session_id"].duplicated().sum()),
        "tied_patient_time_rows": int(frame.duplicated([patient_col, time_column], keep=False).sum()),
    })
    frame = frame.sort_values([patient_col, time_column, "session_id"], kind="stable").reset_index(drop=True)
    groups = frame.groupby(patient_col, sort=False)
    prior_count = groups.cumcount()
    denominator = prior_count.replace(0, np.nan)
    expected: dict[str, pd.Series] = {}
    for column in HISTORY_MEAN_COLUMNS:
        raw = pd.to_numeric(frame[column], errors="coerce")
        expected[f"历史平均{column}"] = raw.groupby(frame[patient_col], sort=False).transform(
            lambda series: series.shift().expanding().mean()
        ).fillna(0.0)
    prior_idh = groups["idh_event"].shift().fillna(0).groupby(frame[patient_col], sort=False).cumsum()
    prior_ih = groups["ih_event"].shift().fillna(0).groupby(frame[patient_col], sort=False).cumsum()
    expected["history_IDH_rate"] = (prior_idh / denominator).fillna(0.0)
    expected["history_HBP_rate"] = (prior_ih / denominator).fillna(0.0)
    expected["history_LBP_times_0_rate"] = ((prior_count - prior_idh) / denominator).fillna(0.0)
    bins = pd.to_numeric(frame["降幅时间点比值区间"], errors="coerce")
    for index in range(1, 5):
        previous = bins.groupby(frame[patient_col], sort=False).shift().eq(index).fillna(False)
        counts = previous.astype(int).groupby(frame[patient_col], sort=False).cumsum()
        expected[f"history_LBP_times_{index}_rate"] = (counts / denominator).fillna(0.0)

    tied = frame.duplicated([patient_col, time_column], keep=False)
    for (patient_id, timestamp), indices in frame.loc[tied].groupby(
        [patient_col, time_column], sort=False
    ).groups.items():
        prior = frame.loc[frame[patient_col].eq(patient_id) & frame[time_column].lt(timestamp)]
        count = len(prior)
        prior_count.loc[indices] = count
        for column in HISTORY_MEAN_COLUMNS:
            expected[f"历史平均{column}"].loc[indices] = (
                pd.to_numeric(prior[column], errors="coerce").mean() if count else 0.0
            )
        prior_idh_count = int(prior["idh_event"].sum())
        expected["history_IDH_rate"].loc[indices] = prior_idh_count / count if count else 0.0
        expected["history_HBP_rate"].loc[indices] = float(prior["ih_event"].sum()) / count if count else 0.0
        expected["history_LBP_times_0_rate"].loc[indices] = (count - prior_idh_count) / count if count else 0.0
        for index in range(1, 5):
            expected[f"history_LBP_times_{index}_rate"].loc[indices] = (
                float(prior["降幅时间点比值区间"].eq(index).sum()) / count if count else 0.0
            )
    mismatches = 0
    for feature, values in expected.items():
        count, maximum = _numeric_equal(frame[feature], values, tolerance)
        report["feature_checks"][feature] = {"mismatched_rows": count, "maximum_absolute_error": maximum}
        mismatches += count
    first_rows = prior_count.eq(0)
    history_columns = list(expected)
    first_values = frame.loc[first_rows, history_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    report["first_session_nonzero_history_rows"] = int(first_values.ne(0).any(axis=1).sum())
    report["total_feature_mismatches"] = mismatches
    report["strictly_prior_definition"] = "patient-time sorted; tied sessions share history from strictly earlier timestamps"
    report["errors"] = []
    if report["missing_session_times"]:
        report["errors"].append("unparseable_session_time")
    if report["duplicate_session_ids"]:
        report["errors"].append("duplicate_session_id")
    if mismatches:
        report["errors"].append("history_values_not_strictly_prior")
    if report["first_session_nonzero_history_rows"]:
        report["errors"].append("first_session_history_not_zero")
    report["passed"] = not report["errors"]
    return report


def write_audit(path: str | Path, output: str | Path) -> dict[str, Any]:
    report = audit_prior_history(path)
    Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
