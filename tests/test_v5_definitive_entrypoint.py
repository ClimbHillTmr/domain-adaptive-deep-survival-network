import copy
import json
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml
from torch import nn

import scripts.run_v5_definitive as v5
import src.train.discrete_survival as survival_training
from src.data.survival_dataset import SurvivalSessionDataset


def _config(tmp_path: Path, source: Path, target: Path) -> dict:
    config = v5.load_config()
    config = copy.deepcopy(config)
    config["outputs"] = {
        "data_root": str(tmp_path / "data"),
        "experiment_root": str(tmp_path / "experiments"),
        "refuse_overwrite": True,
    }
    for spec in config["scientific_assets"].values():
        if isinstance(spec, dict) and "path" in spec:
            spec["path"] = str(tmp_path / "data" / Path(spec["path"]).name)
    config["raw_snapshots"]["source"].update(path=str(source), expected_sha256=v5._sha256(source))
    config["raw_snapshots"]["target_development"].update(path=str(target), expected_sha256=v5._sha256(target))
    return config


@pytest.fixture(autouse=True)
def _stub_scientific_asset_build(monkeypatch):
    def build(source_raw, target_raw, data_root, **kwargs):
        root = Path(data_root)
        root.mkdir(parents=True)
        assets = {}
        for name in v5.REQUIRED_SCIENTIFIC_ASSETS:
            suffix = ".json" if name in {"preprocessing", "primary_partition"} else ".csv"
            path = root / f"{name}{suffix}"
            path.write_text("value\n", encoding="utf-8")
            assets[name] = {"path": str(path), "sha256": v5._sha256(path)}
        return assets

    monkeypatch.setattr(v5, "build_scientific_assets", build)


def _survival_dataset(size: int, prefix: str) -> SurvivalSessionDataset:
    frame = pd.DataFrame(
        {
            "患者id": [f"{prefix}{index}" for index in range(size)],
            "x": [float(index + (10 if prefix == "s" else -10)) for index in range(size)],
            "idh_event_observed_240": [index % 2 for index in range(size)],
            "idh_survival_time": [60.0 if index % 2 else 240.0 for index in range(size)],
        }
    )
    return SurvivalSessionDataset(frame, ["x"], "idh")


def test_v5_paired_update_retains_tail_task_batches_and_skips_tail_alignment(monkeypatch):
    source = _survival_dataset(5, "s")
    target = _survival_dataset(3, "t")
    model = nn.Linear(1, 4)
    task_batch_sizes = {id(source): [], id(target): []}
    alignment_batch_sizes = []
    original_batch_loss = survival_training._batch_numerator_and_weight

    def record_batch_loss(model, batch, device):
        dataset_id = id(source) if batch[0].max().item() >= 3 else id(target)
        task_batch_sizes[dataset_id].append(len(batch[0]))
        return original_batch_loss(model, batch, device)

    def record_alignment(source_features, target_features):
        alignment_batch_sizes.append((len(source_features), len(target_features)))
        return source_features.sum() * 0.0 + target_features.sum() * 0.0

    monkeypatch.setattr(survival_training, "_batch_numerator_and_weight", record_batch_loss)
    monkeypatch.setattr(survival_training, "centered_coral", record_alignment)
    monkeypatch.setattr(survival_training, "evaluate_survival_nll", lambda *args, **kwargs: 0.0)
    survival_training.fit_paired_source_target_update(
        model,
        source,
        target,
        target,
        device=torch.device("cpu"),
        learning_rate=0.0,
        batch_size=2,
        max_epochs=1,
        patience=1,
        seed=7,
        coral_weight=0.01,
        representation_fn=lambda _model, features: features,
        retain_task_tail_batches=True,
    )

    assert sorted(task_batch_sizes[id(source)]) == [1, 2, 2]
    assert sorted(task_batch_sizes[id(target)]) == [1, 2, 2]
    assert alignment_batch_sizes and all(pair == (2, 2) for pair in alignment_batch_sizes)


def test_contract_3_locks_target_history_and_fuding_claim_boundary():
    config = v5.load_config()
    assert config["protocol_version"] == "v5_definitive_20260820_contract_3"
    assert config["study_mode"] == "held_out_external_center_internal_revalidation"
    assert config["validation_claims"] == {
        "fuding_role": "held_out_external_center_internal_revalidation",
        "untouched_external_claim_forbidden": True,
        "definitive_external_claim_forbidden": True,
    }
    assert config["label_efficiency"]["target_outcome_history"] == {
        "availability_unit": "patient",
        "allowed_roles": ["target_update", "target_calibration"],
        "require_budget_membership": True,
        "non_budget_patient_history_forbidden": True,
        "target_internal_test_history_forbidden": True,
    }


@pytest.mark.parametrize(
    ("section", "key", "value", "error"),
    [
        ("validation_claims", "untouched_external_claim_forbidden", False, "Fuding"),
        ("validation_claims", "definitive_external_claim_forbidden", False, "Fuding"),
        ("label_efficiency", "outcome_blind_selection", False, "预算门控"),
        ("label_efficiency", "internal_test_forbidden", False, "预算门控"),
    ],
)
def test_load_yaml_rejects_relaxed_contract_3_locks(tmp_path, section, key, value, error):
    payload = yaml.safe_load(v5.CONFIG.read_text(encoding="utf-8"))
    payload[section][key] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        v5.load_config(path)


def test_load_yaml_rejects_legacy_output_path(tmp_path):
    payload = yaml.safe_load(v5.CONFIG.read_text(encoding="utf-8"))
    payload["outputs"]["data_root"] = "data/v4/illegal"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match="旧资产"):
        v5.load_config(path)


def test_prepare_refuses_overwrite(tmp_path):
    source, target = tmp_path / "source.csv", tmp_path / "target.csv"
    source.write_text("source", encoding="utf-8")
    target.write_text("target", encoding="utf-8")
    config = _config(tmp_path, source, target)
    assert v5.prepare(config)["status"] == "prepared"
    with pytest.raises(FileExistsError, match="拒绝覆盖"):
        v5.prepare(config)


def test_audit_passes_development_gate_but_blocks_definitive_claim_without_confirmation(tmp_path):
    source, target = tmp_path / "source.csv", tmp_path / "target.csv"
    source.write_text("source", encoding="utf-8")
    target.write_text("target", encoding="utf-8")
    config = _config(tmp_path, source, target)
    v5.prepare(config)
    _, _, manifest_path, events_path = v5._roots(config)
    manifest_before = manifest_path.read_bytes()

    result = v5.audit(config)

    assert result == {"status": "pass", "errors": [], "claim_blockers": ["confirmation_required"]}
    assert manifest_path.read_bytes() == manifest_before
    audit_event = json.loads(events_path.read_text(encoding="utf-8").splitlines()[-1])
    assert audit_event["status"] == "pass"
    assert audit_event["claim_blockers"] == ["confirmation_required"]

    target.write_text("tampered", encoding="utf-8")
    result = v5.audit(config)
    assert result["status"] == "blocked"
    assert "raw_hash_mismatch:target_development" in result["errors"]
    assert result["claim_blockers"] == ["confirmation_required"]
    assert manifest_path.read_bytes() == manifest_before
    assert len(v5.verify_jsonl_chain(events_path)) == 3


def test_plan_requires_prepare_and_allows_development_flow_without_confirmation(tmp_path):
    source, target = (tmp_path / name for name in ("source.csv", "target.csv"))
    for path in (source, target):
        path.write_text(path.name, encoding="utf-8")
    config = _config(tmp_path, source, target)

    with pytest.raises(RuntimeError, match="prepare"):
        v5.plan(config)
    v5.prepare(config)
    with pytest.raises(PermissionError, match="audit"):
        v5.plan(config)
    audit_result = v5.audit(config)
    assert audit_result["status"] == "pass"
    assert audit_result["claim_blockers"] == ["confirmation_required"]
    result = v5.plan(config)
    assert result["run_count"] == 0
    assert json.loads(Path(result["run_plan"]).read_text(encoding="utf-8"))["training_authorized"] is False


def test_authorize_only_appends_event_and_never_enables_training(tmp_path):
    source, target, confirmation = (tmp_path / name for name in ("source.csv", "target.csv", "confirmation.csv"))
    for path in (source, target, confirmation):
        path.write_text(path.name, encoding="utf-8")
    config = _config(tmp_path, source, target)
    config["raw_snapshots"]["target_confirmation"].update(path=str(confirmation), expected_sha256=v5._sha256(confirmation))
    v5.prepare(config)
    v5.audit(config)
    v5.plan(config)

    _, _, manifest_path, _ = v5._roots(config)
    manifest_before = manifest_path.read_bytes()
    result = v5.authorize(config)
    _, _, manifest_path, events_path = v5._roots(config)
    assert manifest_path.read_bytes() == manifest_before
    assert result["status"] == "recorded" and result["training_started"] is False
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["training_authorized"] is False
    last = json.loads(events_path.read_text(encoding="utf-8").splitlines()[-1])
    assert last["event"] == "authorization_requested" and last["training_authorized"] is False
    assert last["manifest_sha256"] == json.loads(manifest_path.read_text(encoding="utf-8"))["manifest_sha256"]
    assert len(last["run_plan_sha256"]) == 64
