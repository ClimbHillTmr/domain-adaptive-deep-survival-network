from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.data.binary_dataset import prepare_binary_data
from src.evaluate.binary_metrics import evaluate_binary, paired_patient_bootstrap_delta
from src.main_time_models import _fit_cox, _person_period
from src.models.discrete_cdan import DiscreteHazardCDAN
from src.train.binary_models import calibrate_probability, fit_probability_calibrator


def _cohort(patient_prefix: str, n_patients: int, center_shift: float = 0.0) -> pd.DataFrame:
    rows = []
    for patient_index in range(n_patients):
        for session_index in range(3):
            idh_event = int((patient_index + session_index) % 3 == 0)
            ih_event = int((patient_index + session_index) % 4 == 0)
            rows.append(
                {
                    "患者id": f"{patient_prefix}{patient_index}",
                    "session_id": f"{patient_prefix}{patient_index}:{session_index}",
                    "events": idh_event,
                    "et_min": 60.0 if idh_event else 0.0,
                    "idh_event": idh_event,
                    "idh_time_min": 60.0 if idh_event else 0.0,
                    "ih_event": ih_event,
                    "ih_time_min": 30.0 if ih_event else 0.0,
                    "透中低血压_计算": idh_event,
                    "透中高血压_计算": ih_event,
                    "feature_a": center_shift + patient_index + session_index,
                }
            )
    return pd.DataFrame(rows)


def test_binary_data_uses_patient_splits_and_source_fitted_transform(tmp_path):
    source = _cohort("s", 20)
    target = _cohort("t", 12, center_shift=1000.0)
    source_path, target_path = tmp_path / "source.csv", tmp_path / "target.csv"
    source.to_csv(source_path, index=False)
    target.to_csv(target_path, index=False)
    allowlist = tmp_path / "allowlist.csv"
    pd.DataFrame(
        {
            "feature": ["feature_a"],
            "allowed": ["yes"],
            "available_at_prediction_time": ["start"],
        }
    ).to_csv(allowlist, index=False)

    bundle = prepare_binary_data(
        source_path,
        target_path,
        allowlist_path=allowlist,
        split_seed=7,
        source_validation_fraction=0.2,
        target_update_fraction=0.5,
        target_validation_fraction=0.34,
    )

    target_patient_sets = [
        set(bundle.split_manifest[name]["patient_ids"])
        for name in ("target_train", "target_val", "target_test")
    ]
    assert not target_patient_sets[0] & target_patient_sets[1]
    assert not target_patient_sets[0] & target_patient_sets[2]
    assert not target_patient_sets[1] & target_patient_sets[2]
    assert abs(bundle.source_train["feature_a"].mean()) < 1e-6
    assert bundle.target_test["feature_a"].mean() > 100
    assert (bundle.target_test.loc[bundle.target_test["idh_event"].eq(0), "idh_time_min"] == 0).all()
    assert (bundle.target_test.loc[bundle.target_test["ih_event"].eq(0), "ih_time_min"] == 0).all()


def test_binary_metrics_use_patient_cluster_bootstrap():
    y_true = np.array([0, 1, 0, 1, 0, 1])
    patient_ids = np.array(["a", "a", "b", "b", "c", "c"])
    better = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
    worse = 1 - better
    metrics = evaluate_binary(
        y_true, better, patient_ids, threshold=0.5, n_bootstrap=20, seed=2
    )
    delta = paired_patient_bootstrap_delta(
        y_true, better, worse, patient_ids, n_bootstrap=20, seed=2
    )
    assert metrics["roc_auc"] == 1.0
    assert metrics["n_patients"] == 3
    assert delta["delta_a_minus_b"] == 1.0


def test_probability_calibration_uses_validation_prevalence():
    y_val = np.array([0, 0, 0, 1])
    raw = np.array([0.4, 0.5, 0.6, 0.9])
    calibrated = calibrate_probability(fit_probability_calibrator(y_val, raw), raw)
    assert np.all((calibrated > 0) & (calibrated < 1))
    assert abs(calibrated.mean() - y_val.mean()) < 0.01


def test_person_period_and_cdan_shapes():
    frame = pd.DataFrame(
        {
            "feature_a": [0.0, 1.0],
            "idh_event": [1, 0],
            "idh_time_min": [60.0, 0.0],
            "duration_minutes": [180.0, 120.0],
        }
    )
    features, labels = _person_period(frame, ["feature_a"], "idh", 60, 4)
    assert features.shape == (3, 5)
    assert labels.tolist() == [1.0, 0.0, 0.0]
    hazard, domain = DiscreteHazardCDAN(5, (4,), 0.0)(torch.from_numpy(features), 0.5)
    assert hazard.shape == (3,)
    assert domain.shape == (3, 2)
    beta = _fit_cox(
        np.array([[0.0], [1.0], [2.0]], dtype=np.float32),
        np.array([1.0, 1.0, 0.0], dtype=np.float32),
        np.array([60.0, 60.0, 120.0], dtype=np.float32),
        penalizer=0.01,
        epochs=2,
    )
    assert np.isfinite(beta).all()


def test_binary_data_rejects_non_event_with_positive_time(tmp_path):
    source = _cohort("s", 10)
    target = _cohort("t", 10)
    source.loc[source["idh_event"].eq(0).idxmax(), "idh_time_min"] = 240.0
    source_path, target_path = tmp_path / "source.csv", tmp_path / "target.csv"
    source.to_csv(source_path, index=False)
    target.to_csv(target_path, index=False)
    allowlist = tmp_path / "allowlist.csv"
    pd.DataFrame(
        {
            "feature": ["feature_a"],
            "allowed": ["yes"],
            "available_at_prediction_time": ["start"],
        }
    ).to_csv(allowlist, index=False)

    try:
        prepare_binary_data(source_path, target_path, allowlist_path=allowlist)
    except ValueError as error:
        assert "idh non-events whose time is not zero" in str(error)
    else:
        raise AssertionError("Invalid non-event time was accepted")
