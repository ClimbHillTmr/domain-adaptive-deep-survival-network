import math

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from src.data.survival_dataset import (
    SurvivalSessionDataset,
    build_discrete_survival_targets,
    inverse_session_count_weights,
)
from src.train.discrete_survival import centered_coral, fit_source_only, masked_survival_nll


def test_event_boundaries_map_to_prespecified_bins():
    targets, masks = build_discrete_survival_targets(
        [1, 1, 1, 1, 1, 1, 1, 1],
        [0, 60, 61, 120, 121, 180, 181, 240],
    )
    expected_bins = [0, 0, 1, 1, 2, 2, 3, 3]
    assert targets.argmax(axis=1).tolist() == expected_bins
    for row, event_bin in enumerate(expected_bins):
        np.testing.assert_array_equal(masks[row], np.arange(4) <= event_bin)


def test_censoring_contributes_only_completed_intervals():
    _, masks = build_discrete_survival_targets(
        [0, 0, 0, 0, 0, 0, 0],
        [0, 59.9, 60, 119.9, 120, 239.9, 240],
    )
    np.testing.assert_array_equal(
        masks,
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 1],
        ],
    )


def test_patient_weights_and_masked_nll_match_manual_calculation():
    weights = inverse_session_count_weights(["a", "a", "b"])
    np.testing.assert_allclose(weights, [0.5, 0.5, 1.0])
    logits = torch.zeros((3, 4))
    targets = torch.zeros_like(logits)
    masks = torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0], [1, 1, 1, 1]], dtype=torch.float32)
    loss = masked_survival_nll(logits, targets, masks, torch.from_numpy(weights))
    expected = math.log(2) * (0.5 * 2 + 0.5 * 1 + 1.0 * 4) / 2.0
    assert loss.item() == pytest.approx(expected)


def test_dataset_consumes_endpoint_specific_contract_without_person_period_expansion():
    frame = pd.DataFrame(
        {
            "患者id": ["a", "a", "b"],
            "x": [1.0, 2.0, 3.0],
            "idh_event_observed_240": [1, 0, 1],
            "idh_survival_time": [61.0, 120.0, 0.0],
        }
    )
    dataset = SurvivalSessionDataset(frame, ["x"], "idh")
    assert len(dataset) == len(frame)
    assert dataset.features.shape == (3, 1)
    np.testing.assert_array_equal(dataset.targets.numpy(), [[0, 1, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]])
    np.testing.assert_array_equal(dataset.masks.numpy(), [[1, 1, 0, 0], [1, 1, 0, 0], [1, 0, 0, 0]])


@pytest.mark.parametrize(
    ("events", "times"),
    [([2], [60]), ([1], [-1]), ([0], [241]), ([1], [np.nan]), ([1, 0], [60])],
)
def test_invalid_survival_contract_is_rejected(events, times):
    with pytest.raises(ValueError):
        build_discrete_survival_targets(events, times)


def test_dataset_rejects_missing_contract_and_nonfinite_features():
    base = pd.DataFrame(
        {
            "患者id": ["a"],
            "x": [np.nan],
            "idh_event_observed_240": [0],
            "idh_survival_time": [60],
        }
    )
    with pytest.raises(ValueError, match="finite"):
        SurvivalSessionDataset(base, ["x"], "idh")
    with pytest.raises(ValueError, match="missing required"):
        SurvivalSessionDataset(base.drop(columns="idh_survival_time"), ["x"], "idh")


def test_centered_coral_is_translation_invariant_and_zero_for_equal_covariance():
    source = torch.tensor([[0.0, 1.0], [1.0, 3.0], [2.0, 5.0]])
    target = source + torch.tensor([100.0, -50.0])
    assert centered_coral(source, target).item() == pytest.approx(0.0, abs=1e-12)


def test_source_fit_returns_validation_nll_based_checkpoint_metadata():
    frame = pd.DataFrame(
        {
            "患者id": ["a", "b", "c", "d"],
            "x": [-1.0, -0.5, 0.5, 1.0],
            "idh_event_observed_240": [0, 0, 1, 1],
            "idh_survival_time": [240, 120, 60, 61],
        }
    )
    dataset = SurvivalSessionDataset(frame, ["x"], "idh")
    model = nn.Linear(1, 4)
    result = fit_source_only(
        model,
        dataset,
        dataset,
        device=torch.device("cpu"),
        learning_rate=0.01,
        batch_size=4,
        max_epochs=3,
        patience=2,
        seed=7,
    )
    assert math.isfinite(result.best_validation_nll)
    assert 1 <= result.best_epoch <= result.epochs_run <= 3
