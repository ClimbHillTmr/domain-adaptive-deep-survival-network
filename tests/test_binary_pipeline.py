from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from src.data.binary_dataset import prepare_binary_data
from src.evaluate.binary_metrics import (
    evaluate_binary,
    expected_calibration_error,
    paired_patient_bootstrap_delta,
    paired_patient_bootstrap_metric_deltas,
)
from src.main_time_models import _fit_cox, _person_period
from src.models.discrete_cdan import DiscreteHazardCDAN
from src.train.binary_models import BinaryMLP, calibrate_probability, fit_probability_calibrator
from scripts.build_clinical_utility import _clinical_bootstrap, _patient_cluster_indices
from scripts import build_prespecified_clinical_utility
from scripts.run_confirmatory_ablation import (
    BASE_CONFIG,
    GENERATED_CONFIGS,
    _balanced_execution_order,
    _execution_plan,
    _expected_config_payloads,
)
from src.evaluate.subgroup_cluster import assign_prespecified_levels, patient_cluster_subgroup_analysis


ROOT = Path(__file__).resolve().parents[1]


def test_dual_branch_alignment_scopes_have_expected_dimensions():
    model = BinaryMLP(input_dim=6, hidden_dims=(4, 2), dropout=0.0, physio_indices=[0, 1, 2, 3])
    features = torch.zeros((3, 6))
    assert model.get_alignment_features(features, "selected_branch").shape == (3, 2)
    assert model.get_alignment_features(features, "global").shape == (3, 4)


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
    assert metrics["ece_10_quantile_bins"] >= 0
    assert metrics["patient_cluster_bootstrap"]["ece_10_quantile_bins"]["valid_replicates"] == 20
    assert delta["delta_a_minus_b"] == 1.0


def test_expected_calibration_error_is_zero_for_exact_predictions():
    outcome = np.array([0, 1, 0, 1])
    probability = np.array([0.0, 1.0, 0.0, 1.0])
    assert expected_calibration_error(outcome, probability, bins=2) == 0.0


def test_paired_metric_deltas_preserve_metric_direction():
    patient_ids = np.repeat(["a", "b", "c", "d"], 2)
    outcome = np.tile([0, 1], 4)
    better = np.tile([0.1, 0.9], 4)
    worse = np.tile([0.4, 0.6], 4)
    result = paired_patient_bootstrap_metric_deltas(
        outcome, better, worse, patient_ids, n_bootstrap=20, seed=5
    )
    assert result["roc_auc"]["favorable_direction"] == "positive"
    assert result["brier"]["favorable_direction"] == "negative"
    assert result["brier"]["delta_a_minus_b"] < 0
    assert all(metric["valid_replicates"] == 20 for metric in result.values())


def test_clinical_utility_bootstrap_resamples_patients_and_returns_intervals():
    patient_ids = np.repeat(["a", "b", "c", "d"], 4)
    y_true = np.tile([0, 0, 1, 1], 4)
    probability = np.tile([0.1, 0.3, 0.7, 0.9], 4)
    indices = _patient_cluster_indices(patient_ids, replicates=20, seed=3)
    assert len(indices) == 20
    assert all(len(index) == len(y_true) for index in indices)
    result = _clinical_bootstrap(y_true, probability, 0.5, indices)
    assert result["sensitivity"]["valid_replicates"] == 20
    assert result["specificity"]["lower"] == 1.0
    assert result["brier"]["upper"] >= result["brier"]["lower"]


def test_prespecified_clinical_utility_rejects_unlocked_protocol_before_registry_access(
    tmp_path, monkeypatch
):
    protocol = yaml.safe_load(
        (ROOT / "conf" / "clinical_threshold_protocol.yaml").read_text(encoding="utf-8")
    )
    protocol_path = tmp_path / "clinical_threshold_protocol.yaml"
    protocol_path.write_text(yaml.safe_dump(protocol), encoding="utf-8")
    output_dir = tmp_path / "prespecified"
    monkeypatch.setattr(
        build_prespecified_clinical_utility,
        "REGISTRY",
        tmp_path / "registry_must_not_be_read.csv",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_prespecified_clinical_utility.py",
            "--build",
            "--protocol",
            str(protocol_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    with pytest.raises(SystemExit, match="protocol status must be prespecified_locked"):
        build_prespecified_clinical_utility.main()
    assert not output_dir.exists()


def test_patient_cluster_subgroup_analysis_uses_prespecified_levels():
    frame = pd.DataFrame(
        {
            "patient_id": np.repeat([f"p{i}" for i in range(12)], 4),
            "feature": np.repeat(np.arange(12), 4),
            "outcome": np.tile([0, 0, 1, 1], 12),
            "updated": np.tile([0.1, 0.2, 0.8, 0.9], 12),
            "source": np.tile([0.4, 0.3, 0.7, 0.6], 12),
        }
    )
    frame["level"] = assign_prespecified_levels(
        frame["feature"], {"type": "numeric_cutpoint", "cutpoint": 6, "labels": ["low", "high"]}
    )
    rows = patient_cluster_subgroup_analysis(
        frame,
        level_col="level",
        outcome_col="outcome",
        updated_probability_col="updated",
        source_probability_col="source",
        patient_col="patient_id",
        levels=["low", "high"],
        minimum_patients=5,
        replicates=20,
        seed=4,
    )
    assert [row["subgroup_level"] for row in rows] == ["low", "high"]
    assert all(row["n_patients"] == 6 for row in rows)
    assert all(row["bootstrap_valid_replicates"] == 20 for row in rows)
    assert all(0 <= row["interaction_p_value"] <= 1 for row in rows)


def test_subgroup_analysis_rejects_underpowered_levels():
    frame = pd.DataFrame(
        {
            "patient_id": ["a", "a", "b", "b"],
            "level": ["low", "low", "high", "high"],
            "outcome": [0, 1, 0, 1],
            "updated": [0.1, 0.9, 0.2, 0.8],
            "source": [0.2, 0.8, 0.3, 0.7],
        }
    )
    try:
        patient_cluster_subgroup_analysis(
            frame,
            level_col="level",
            outcome_col="outcome",
            updated_probability_col="updated",
            source_probability_col="source",
            patient_col="patient_id",
            levels=["low", "high"],
            minimum_patients=2,
            replicates=10,
            seed=1,
        )
    except ValueError as error:
        assert "minimum is 2" in str(error)
    else:
        raise AssertionError("Underpowered subgroup level was accepted")


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


def test_submission_supplement_is_provenance_linked_and_nontraining():
    supplement = ROOT / "docs" / "submission_supplement.md"
    provenance = json.loads(
        (ROOT / "experiments" / "evidence_registry" / "submission_supplement_provenance.json")
        .read_text(encoding="utf-8")
    )
    assert provenance["training_performed"] is False
    assert provenance["registered_historical_runs"] == 8
    assert provenance["feature_rows"] == 57
    assert provenance["included_predictors"] == 24
    assert hashlib.sha256(supplement.read_bytes()).hexdigest() == provenance["output_sha256"]
    for relative_path, expected_hash in provenance["sources"].items():
        assert hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest() == expected_hash


def test_supplementary_seed_figure_is_registered_and_provenance_linked():
    output = ROOT / "figures" / "submission_staging"
    provenance = json.loads((output / "SupplementaryFigureS1_provenance.json").read_text(encoding="utf-8"))
    source_data = pd.read_csv(output / "SupplementaryFigureS1_source_data.csv")
    assert provenance["training_performed"] is False
    assert provenance["seeds"] == [7, 13, 42, 99, 2024]
    assert len(provenance["registered_run_ids"]) == 5
    assert len(source_data) == 20
    assert source_data["evidence_tier"].eq("registered_historical_v1_descriptive_multiseed").all()
    assert set(source_data["run_id"]) == set(provenance["registered_run_ids"])
    for relative_path, expected_hash in provenance["sources"].items():
        assert hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest() == expected_hash
    for name, expected_hash in provenance["outputs"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected_hash


def test_locked_confirmatory_configs_match_protocol_without_rewriting():
    base = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    config_paths = sorted(GENERATED_CONFIGS.glob("*.yaml"))
    assert len(config_paths) == 25
    locks = {
        json.dumps(
            yaml.safe_load(path.read_text(encoding="utf-8"))["confirmatory"]["evidence_lock"],
            sort_keys=True,
        )
        for path in config_paths
    }
    assert len(locks) == 1
    evidence_lock = json.loads(next(iter(locks)))
    expected = _expected_config_payloads(base, evidence_lock)
    assert {path for path, _ in expected} == set(config_paths)
    for path, payload in expected:
        assert yaml.safe_load(path.read_text(encoding="utf-8")) == payload


def test_confirmatory_execution_plan_is_complete_and_marks_completed_hashes():
    configs = _balanced_execution_order(sorted(GENERATED_CONFIGS.glob("*.yaml")))
    initial = _execution_plan(configs, set())
    completed_hash = initial[0]["config_sha256"]
    plan = _execution_plan(configs, {completed_hash})

    assert len(plan) == 25
    assert len({(row["model_id"], row["seed"]) for row in plan}) == 25
    assert sum(row["status"] == "completed_server_confirmatory" for row in plan) == 1
    assert sum(row["status"] == "pending" for row in plan) == 24
    for position in range(1, 6):
        assert len({row["model_id"] for row in plan if row["within_seed_position"] == position}) == 5
