import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.run_v4_restart import derive_seeds, load_config, plan, verify_run, _model
from src.data.survival_dataset import SurvivalSessionDataset, inverse_session_count_weights
from src.train.discrete_survival import fit_paired_source_target_update

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = ["年龄", "性别", "透析龄年", "透前收缩压", "透前舒张压", "透前体重", "透前体重-干体重", "历史平均透前体重", "历史平均透前收缩压", "历史平均透前舒张压", "history_prior_session_count", *[f"history_IDH_240_interval_{i}_rate" for i in range(1, 5)], *[f"history_IH_240_interval_{i}_rate" for i in range(1, 5)]]


def test_feature_order_seed_and_capacity_contract():
    config = load_config()
    assert config["features"]["ordered"] == EXPECTED
    assert derive_seeds(20260730) == derive_seeds(20260730)
    assert len(set(derive_seeds(20260730)["optimization"])) == 5
    single = sum(p.numel() for p in _model("source_only").parameters())
    dual = sum(p.numel() for p in _model("dual_physio_coral", list(range(18)), [18]).parameters())
    assert single == 5769 and dual == 5764 and abs(single - dual) / dual < 0.05


def test_global_patient_weights_are_equal():
    weights = inverse_session_count_weights(["a", "a", "a", "b"])
    assert np.isclose(weights[:3].sum(), weights[3:].sum())


def test_singleton_coral_is_skipped_but_task_sample_trains():
    columns = {name: [0.0, 1.0, 2.0] for name in EXPECTED}
    columns.update({"患者id": ["a", "a", "b"], "idh_event_observed_240": [0, 1, 0], "idh_survival_time": [240.0, 0.0, 240.0]})
    frame = pd.DataFrame(columns)
    dataset = SurvivalSessionDataset(frame, EXPECTED, "idh")
    model = _model("dual_physio_coral", list(range(18)), [18])
    result = fit_paired_source_target_update(model, dataset, dataset, dataset, device=torch.device("cpu"), learning_rate=1e-4, batch_size=2, max_epochs=1, patience=1, seed=1, coral_weight=0.01, representation_fn=lambda m, x: m.alignment_features(x, "branch_a"))
    assert result.epochs_run == 1


def test_plan_requires_prepare_and_training_is_initially_unauthorized(tmp_path):
    config = load_config()
    assert config["training_authorized"] is False
    data_root = ROOT / config["outputs"]["data_root"]
    if not (data_root / "evidence_lock.json").exists():
        with pytest.raises(RuntimeError, match="prepare"):
            plan(config)


def test_verify_requires_complete_artifacts(tmp_path):
    assert not verify_run(tmp_path)
    for name in ["resolved_config.json", "environment.json", "training_log.json", "checkpoint.pt", "predictions.csv", "calibration.json", "evaluation.json"]:
        (tmp_path / name).write_text("x", encoding="utf-8")
    assert verify_run(tmp_path)
