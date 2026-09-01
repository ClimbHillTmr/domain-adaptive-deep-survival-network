from pathlib import Path

import pytest
import torch

from scripts.run_v4_survival import (
    DEFAULT_CONFIG,
    EXPECTED_PROTOCOL,
    audit,
    build_family_components,
    load_config,
    plan,
    train,
)
from src.train.discrete_survival import fit_paired_source_target_update
from src.train.survival_family_specs import SURVIVAL_FAMILY_SPECS


ROOT = Path(__file__).resolve().parents[1]


def test_v4_configuration_parses_locked_contract():
    config = load_config(DEFAULT_CONFIG)

    assert config["protocol_version"] == EXPECTED_PROTOCOL
    assert config["training_authorized"] is False
    assert config["survival"]["horizons_minutes"] == [60, 120, 180, 240]
    assert config["features"]["expected_count"] == 22
    assert config["features"]["status"] == "locked"
    assert config["models"]["single_hidden_dims"] == [64, 65]
    assert config["models"]["dual_hidden_dims"] == [64, 32]
    assert config["references"]["random_partitions"]["status"] == "candidate_not_yet_v4_locked"


def test_train_hard_refuses_when_not_authorized():
    config = load_config(DEFAULT_CONFIG)

    with pytest.raises(PermissionError, match="training_authorized=false"):
        train(config)


def test_v4_cohorts_and_locked_contract_pass_audit():
    config = load_config(DEFAULT_CONFIG)
    result = audit(config, root=ROOT)

    assert result["status"] == "pass"
    assert result["fallback_to_v3"] is False
    assert result["reference_errors"] == []


def test_plan_contains_every_family_endpoint_and_seed_without_training():
    config = load_config(DEFAULT_CONFIG)
    result = plan(config)

    expected_count = len(SURVIVAL_FAMILY_SPECS) * 2 * 5
    assert result["status"] == "planned_only_no_training"
    assert result["run_count"] == expected_count
    assert {run["family"] for run in result["runs"]} == set(SURVIVAL_FAMILY_SPECS)
    random_runs = [run for run in result["runs"] if run["family"] == "E"]
    assert random_runs
    assert {run["random_partition_status"] for run in random_runs} == {"candidate_not_yet_v4_locked"}


def test_family_builder_connects_model_and_trainer_without_optimizer(monkeypatch):
    config = load_config(DEFAULT_CONFIG)
    optimizer_created = False

    def reject_optimizer(*args, **kwargs):
        nonlocal optimizer_created
        optimizer_created = True
        raise AssertionError("builder must not initialize an optimizer")

    monkeypatch.setattr(torch.optim, "AdamW", reject_optimizer)
    feature_names = [f"feature_{index}" for index in range(22)]
    model, trainer, kwargs = build_family_components(
        config,
        "D1",
        feature_names=feature_names,
        branch_a_features=feature_names[:18],
        branch_b_features=feature_names[18:],
    )

    assert model(torch.zeros(2, 22)).shape == (2, 4)
    assert trainer is fit_paired_source_target_update
    assert kwargs["coral_weight"] == config["models"]["coral"]["weight"]
    assert kwargs["representation_fn"](model, torch.zeros(2, 22)).shape == (2, 32)
    dual_parameters = sum(parameter.numel() for parameter in model.parameters())
    single_model, _, _ = build_family_components(
        config,
        "A",
        feature_names=feature_names,
        branch_a_features=feature_names[:18],
        branch_b_features=feature_names[18:],
    )
    single_parameters = sum(parameter.numel() for parameter in single_model.parameters())
    assert abs(single_parameters - dual_parameters) / dual_parameters < 0.05
    assert optimizer_created is False
