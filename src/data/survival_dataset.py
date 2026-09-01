"""Session-level adaptation for four-bin discrete-time survival outcomes."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

N_INTERVALS = 4
INTERVAL_MINUTES = 60.0
MAX_FOLLOWUP_MINUTES = N_INTERVALS * INTERVAL_MINUTES
SUPPORTED_ENDPOINTS = ("idh", "ih")


def build_discrete_survival_targets(
    event_observed: Sequence[int] | np.ndarray,
    survival_time: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build session-level hazard targets and masks for four 60-minute intervals."""
    event = np.asarray(event_observed)
    time = np.asarray(survival_time, dtype=np.float64)
    if event.ndim != 1 or time.ndim != 1 or event.shape != time.shape:
        raise ValueError("event_observed and survival_time must be same-length one-dimensional arrays.")
    if not np.isfinite(time).all():
        raise ValueError("survival_time must contain only finite values.")
    if not np.isin(event, (0, 1)).all():
        raise ValueError("event_observed must contain only 0 and 1.")
    if ((time < 0) | (time > MAX_FOLLOWUP_MINUTES)).any():
        raise ValueError("survival_time must be within [0, 240] minutes.")

    event = event.astype(np.int64, copy=False)
    targets = np.zeros((len(event), N_INTERVALS), dtype=np.float32)
    masks = np.zeros_like(targets)

    observed_rows = np.flatnonzero(event == 1)
    if len(observed_rows):
        event_bins = np.maximum(0, np.ceil(time[observed_rows] / INTERVAL_MINUTES).astype(int) - 1)
        masks[observed_rows] = np.arange(N_INTERVALS) <= event_bins[:, None]
        targets[observed_rows, event_bins] = 1.0

    censored_rows = np.flatnonzero(event == 0)
    if len(censored_rows):
        completed_bins = np.floor(time[censored_rows] / INTERVAL_MINUTES).astype(int)
        masks[censored_rows] = np.arange(N_INTERVALS) < completed_bins[:, None]
    return targets, masks


def inverse_session_count_weights(patient_ids: Sequence[object] | np.ndarray) -> np.ndarray:
    """Assign each session one over its patient's session count."""
    patients = pd.Series(patient_ids, dtype="string")
    if patients.isna().any() or patients.empty:
        raise ValueError("patient identifiers must be non-missing and non-empty.")
    counts = patients.groupby(patients, sort=False).transform("size")
    return (1.0 / counts.to_numpy(dtype=np.float32)).astype(np.float32)


class SurvivalSessionDataset(Dataset):
    """One row per session with four-bin targets, masks, and patient weights."""

    def __init__(
        self,
        frame: pd.DataFrame,
        feature_names: Sequence[str],
        endpoint: str,
        *,
        patient_col: str = "患者id",
    ) -> None:
        if endpoint not in SUPPORTED_ENDPOINTS:
            raise ValueError(f"endpoint must be one of {SUPPORTED_ENDPOINTS}, got {endpoint!r}.")
        event_col = f"{endpoint}_event_observed_240"
        time_col = f"{endpoint}_survival_time"
        required = {patient_col, event_col, time_col, *feature_names}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"Survival cohort is missing required columns: {missing}")
        if frame.empty:
            raise ValueError("Survival cohort must contain at least one session.")

        features = frame[list(feature_names)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        if not np.isfinite(features).all():
            raise ValueError("Survival features must be finite before dataset construction.")
        event = pd.to_numeric(frame[event_col], errors="coerce").to_numpy()
        time = pd.to_numeric(frame[time_col], errors="coerce").to_numpy(dtype=float)
        targets, masks = build_discrete_survival_targets(event, time)
        weights = inverse_session_count_weights(frame[patient_col].to_numpy())

        self.features = torch.from_numpy(features)
        self.targets = torch.from_numpy(targets)
        self.masks = torch.from_numpy(masks)
        self.session_weights = torch.from_numpy(weights)
        self.patient_ids = frame[patient_col].astype(str).to_numpy()
        self.feature_names = list(feature_names)
        self.endpoint = endpoint

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.features[index], self.targets[index], self.masks[index], self.session_weights[index]
