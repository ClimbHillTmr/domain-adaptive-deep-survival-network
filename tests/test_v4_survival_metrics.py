import numpy as np
import pytest
from sklearn.metrics import brier_score_loss, roc_auc_score

from src.evaluate.discrete_survival_metrics import (
    apply_shared_hazard_platt,
    cumulative_risk_from_hazards,
    evaluate_discrete_survival,
    fit_reverse_kaplan_meier,
    fit_shared_hazard_platt,
    ipcw_brier_score,
    ipcw_cumulative_dynamic_auc,
    patient_cluster_bootstrap_indices,
)


def test_hazard_logits_and_probabilities_produce_monotone_cumulative_risk():
    probabilities = np.array([[0.1, 0.2, 0.3, 0.4], [0.0, 0.5, 1.0, 0.2]])
    expected = 1.0 - np.cumprod(1.0 - probabilities, axis=1)
    from_probabilities = cumulative_risk_from_hazards(
        probabilities, input_type="probabilities"
    )
    logits = np.log(np.clip(probabilities, 1e-10, 1 - 1e-10) / np.clip(1 - probabilities, 1e-10, 1))
    from_logits = cumulative_risk_from_hazards(logits)

    np.testing.assert_allclose(from_probabilities, expected)
    np.testing.assert_allclose(from_logits[0], expected[0])
    assert np.all(np.diff(from_probabilities, axis=1) >= 0)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        cumulative_risk_from_hazards([[0.2, 1.1]], input_type="probabilities")


def test_reverse_km_distinguishes_g_t_and_left_limit_at_ties():
    # At t=2, three rows remain at risk and one is censored: G(2-)=1, G(2)=2/3.
    censoring = fit_reverse_kaplan_meier(
        np.array([1.0, 2.0, 2.0, 3.0]), np.array([1, 1, 0, 1])
    )

    assert censoring.g_left(2.0) == pytest.approx(1.0)
    assert censoring.g(2.0) == pytest.approx(2.0 / 3.0)
    assert censoring.g(2.5) == pytest.approx(2.0 / 3.0)
    assert censoring.g_left(0.5) == pytest.approx(1.0)


def test_ipcw_excludes_early_censoring_instead_of_treating_it_as_control():
    times = np.array([2.0, 3.0, 6.0, 7.0])
    events = np.array([0, 1, 1, 0])
    risks = np.array([0.99, 0.9, 0.1, 0.8])

    # At horizon 5 only the event at 3 is a case and times 6/7 are controls.
    # The censored row at 2 has risk 0.99 but weight zero, so discrimination is perfect.
    assert ipcw_cumulative_dynamic_auc(times, events, risks, 5.0) == pytest.approx(1.0)
    expected = ((0.9 - 1.0) ** 2 / 0.75 + 0.1**2 / 0.75 + 0.8**2 / 0.75) / 4
    assert ipcw_brier_score(times, events, risks, 5.0) == pytest.approx(expected)


def test_no_censoring_reduces_to_ordinary_auc_and_brier():
    times = np.array([20.0, 50.0, 100.0, 150.0])
    events = np.ones(4, dtype=int)
    risks = np.array([0.8, 0.6, 0.4, 0.1])
    horizon = 60.0
    outcomes = (times <= horizon).astype(int)

    assert ipcw_cumulative_dynamic_auc(times, events, risks, horizon) == pytest.approx(
        roc_auc_score(outcomes, risks)
    )
    assert ipcw_brier_score(times, events, risks, horizon) == pytest.approx(
        brier_score_loss(outcomes, risks)
    )


def test_ipcw_rejects_near_zero_g_and_auc_without_both_dynamic_classes():
    with pytest.raises(ValueError, match=r"G\(t\).+zero"):
        ipcw_brier_score(
            np.array([1.0, 2.0, 10.0]),
            np.array([0, 0, 1]),
            np.array([0.2, 0.3, 0.4]),
            5.0,
            min_g=0.5,
        )
    with pytest.raises(ValueError, match="both cases and controls"):
        ipcw_cumulative_dynamic_auc(
            np.array([1.0, 2.0]), np.array([1, 1]), np.array([0.2, 0.8]), 5.0
        )


def test_ipcw_uses_complete_administrative_censoring_as_240_minute_controls():
    times = np.array([30.0, 180.0, 240.0, 240.0])
    events = np.array([1, 1, 0, 0])
    risks = np.array([0.9, 0.8, 0.2, 0.1])

    assert ipcw_cumulative_dynamic_auc(times, events, risks, 240.0) == pytest.approx(1.0)
    assert np.isfinite(ipcw_brier_score(times, events, risks, 240.0))


def test_four_horizon_summary_is_arithmetic_mean_of_horizon_metrics():
    times = np.array([20.0, 80.0, 140.0, 200.0, 300.0])
    events = np.ones(5, dtype=int)
    risk = np.array(
        [
            [0.8, 0.85, 0.9, 0.95],
            [0.2, 0.75, 0.8, 0.85],
            [0.1, 0.2, 0.7, 0.8],
            [0.05, 0.1, 0.2, 0.7],
            [0.01, 0.02, 0.03, 0.04],
        ]
    )

    result = evaluate_discrete_survival(times, events, risk)

    np.testing.assert_array_equal(result["horizons"], [60, 120, 180, 240])
    assert result["mean_auc"] == pytest.approx(np.mean(result["auc"]))
    assert result["mean_brier"] == pytest.approx(np.mean(result["brier"]))
    assert result["mean_auc"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="non-decreasing"):
        evaluate_discrete_survival(times, events, risk[:, ::-1])


def test_patient_cluster_bootstrap_copies_whole_clusters_and_is_paired():
    patient_ids = np.array(["a", "a", "b", "b", "b", "c"])
    first = np.arange(len(patient_ids))
    second = first + 100
    samples = patient_cluster_bootstrap_indices(patient_ids, n_bootstrap=20, seed=7)

    assert len(samples) == 20
    assert any(len(np.unique(patient_ids[index])) < 3 for index in samples)
    for index in samples:
        for patient, original_count in zip(*np.unique(patient_ids, return_counts=True)):
            sampled_count = np.count_nonzero(patient_ids[index] == patient)
            assert sampled_count % original_count == 0
        np.testing.assert_array_equal(second[index] - first[index], 100)


def test_shared_hazard_platt_uses_only_masked_cells_and_requires_positive_slope():
    logits = np.array([[-3.0, -2.0], [-1.0, 0.0], [1.0, 2.0], [3.0, 99.0]])
    targets = np.array([[0, 0], [0, 0], [1, 1], [1, 0]])
    mask = np.array([[1, 1], [1, 0], [1, 1], [1, 0]], dtype=bool)

    intercept, slope = fit_shared_hazard_platt(logits, targets, mask)
    calibrated = apply_shared_hazard_platt(logits, intercept, slope)

    assert slope > 0
    assert np.all(np.diff(calibrated, axis=1) >= 0)
    with pytest.raises(ValueError, match="positive slope"):
        apply_shared_hazard_platt(logits, intercept, 0.0)
    with pytest.raises(ValueError, match="not positive"):
        fit_shared_hazard_platt(-logits, targets, mask)
