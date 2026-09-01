import pandas as pd
import pytest

from src.data.v5_budget_subsets import (
    build_nested_budget_manifest,
    filter_budget_population,
    freeze_budget_manifest,
)


def _split():
    return pd.DataFrame({
        "患者id": ["u1", "u2", "c1", "c2", "test"],
        "center": ["target"] * 5,
        "split_role": ["target_update", "target_update", "target_calibration", "target_calibration", "internal_test"],
    })


def test_budget_is_patient_level_nested_and_outcome_blind():
    manifest = build_nested_budget_manifest(_split(), 20260819, 3, [0.5, 1.0])
    for repeat in (1, 2, 3):
        small = set(manifest.query("subset_repeat == @repeat and budget == 0.5")["患者id"])
        full = set(manifest.query("subset_repeat == @repeat and budget == 1.0")["患者id"])
        assert small <= full
        assert "test" not in full
        assert len(full) == 4
    assert "budget_population" in manifest
    assert "idh_event_observed_240" not in manifest.columns


def test_filter_rejects_internal_test_and_keeps_budgeted_roles_only():
    manifest = build_nested_budget_manifest(_split(), 20260819, 1, [0.5])
    frame = _split()
    with pytest.raises(ValueError, match="internal_test"):
        filter_budget_population(frame, manifest, 0.5, 1)
    frame = frame[frame.split_role != "internal_test"]
    result = filter_budget_population(frame, manifest, 0.5, 1)
    assert set(result["split_role"]) <= {"target_update", "target_calibration"}


def test_freeze_manifest_is_immutable(tmp_path):
    manifest = build_nested_budget_manifest(_split(), 20260819, 1, [1.0])
    path = tmp_path / "budget_subsets.csv"
    lock = freeze_budget_manifest(manifest, path)
    assert lock["rows"] == 4
    with pytest.raises(FileExistsError):
        freeze_budget_manifest(manifest, path)
