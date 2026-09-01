from pathlib import Path
import csv
import json
import yaml

from scripts.generate_v3_run_grid import verify_grid
from scripts.v3_scientific_audit import audit
from src.main_binary import _select_branch_indices, _alignment_scope


ROOT = Path(__file__).resolve().parents[1]


def test_v3_run_grid_is_generated_and_training_still_blocked():
    report = verify_grid()
    assert report["status"] == "verified_no_training"
    assert report["n_runs"] == 420
    assert report["training_authorized_in_configs"] is False

    grid = list(csv.DictReader((ROOT / "v3/experiments/run_grid.csv").open(encoding="utf-8")))
    assert len(grid) == 420
    families = {row["model_family_id"] for row in grid}
    assert "source_only" in families
    assert "A" in families
    assert "D1" in families
    assert "D2" in families
    assert "F" in families
    assert "G" in families
    assert sum(1 for row in grid if row["model_family_id"].startswith("E__")) == 76 * 5

    sample = yaml.safe_load((ROOT / grid[0]["config_path"]).read_text(encoding="utf-8"))
    assert sample["v3"]["training_authorized"] is False
    assert sample["v3"]["execution_context"] == "server_v3"
    assert sample["data"]["split_manifest"] == "v3/data_contract/split_manifest.csv"
    assert sample["data"]["preprocessing_file"] == "data/v3/preprocessing.json"


def test_v3_audit_recognizes_generated_run_grid_gate_progress():
    report = audit()
    assert report["structure_passed"] is True
    assert report["data_preflight_ready"] is True
    assert report["training_ready"] is False
    assert "phase_status_training_authorized_false" in report["blockers"]
    # The grid is generated, so the old missing-grid gate should not remain.
    assert "v3_server_run_grid_and_training_launcher_not_generated_or_reviewed" not in report["execution_gates"]
    assert any("training_authorized" in gate or "authorization" in gate for gate in report["blockers"] + report.get("execution_gates", []))


def test_d2_treatment_coral_returns_treat_indices():
    physio_idx = [0, 1, 2, 3]
    treat_idx = [4, 5, 6, 7]
    d2_config = {
        "architecture": "dual_branch",
        "alignment_strategy": "treatment_coral",
        "physio_indices": physio_idx,
        "treat_indices": treat_idx,
    }
    result = _select_branch_indices("idh", d2_config)
    assert result == treat_idx, f"D2 should return treat_indices for treatment_coral, got {result}"
    result = _select_branch_indices("ih", d2_config)
    assert result == treat_idx, f"D2 should return treat_indices for treatment_coral, got {result}"


def test_d1_physiology_coral_returns_physio_indices():
    physio_idx = [0, 1, 2, 3]
    treat_idx = [4, 5, 6, 7]
    d1_config = {
        "architecture": "dual_branch",
        "alignment_strategy": "physiology_coral",
        "physio_indices": physio_idx,
        "treat_indices": treat_idx,
    }
    result = _select_branch_indices("idh", d1_config)
    assert result == physio_idx, f"D1 should return physio_indices for physiology_coral, got {result}"


def test_alignment_scope_mappings():
    d1_config = {"alignment_strategy": "physiology_coral"}
    assert _alignment_scope(d1_config) == "branch_a", "D1 should use branch_a"

    d2_config = {"alignment_strategy": "treatment_coral"}
    assert _alignment_scope(d2_config) == "branch_b", "D2 should use branch_b"

    g_config = {"alignment_strategy": "global_coral"}
    assert _alignment_scope(g_config) == "global", "G should use global"

    random_config = {"alignment_strategy": "random_feature_coral", "alignment_scope": "branch_a"}
    assert _alignment_scope(random_config) == "branch_a", "Random should use configured scope"
