import pytest
import torch

from src.models.discrete_hazard_domain import DiscreteHazardDomainNet
from src.train.survival_family_specs import SURVIVAL_FAMILY_SPECS, survival_family_spec


def _parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def test_single_and_dual_models_return_four_hazard_logits():
    features = torch.randn(7, 6)
    single = DiscreteHazardDomainNet(6, (12, 8), 0.1, architecture="single")
    dual = DiscreteHazardDomainNet(
        6,
        (12, 8),
        0.1,
        architecture="dual",
        branch_a_indices=[0, 2, 4],
        branch_b_indices=[1, 3, 5],
    )

    assert single(features).shape == (7, 4)
    assert dual(features).shape == (7, 4)


def test_representation_and_alignment_scopes():
    features = torch.randn(5, 6)
    single = DiscreteHazardDomainNet(6, (10, 7), 0.0, architecture="single")
    dual = DiscreteHazardDomainNet(
        6,
        (10, 7),
        0.0,
        architecture="dual",
        branch_a_indices=[0, 1, 2, 3],
        branch_b_indices=[4, 5],
    )

    assert single.representation(features).shape == (5, 7)
    assert single.alignment_features(features, "global").shape == (5, 7)
    with pytest.raises(ValueError, match="dual architecture"):
        single.alignment_features(features, "branch_a")
    with pytest.raises(ValueError, match="dual architecture"):
        single.alignment_features(features, "branch_b")

    assert dual.representation(features).shape == (5, 14)
    assert dual.alignment_features(features, "global").shape == (5, 14)
    assert dual.alignment_features(features, "branch_a").shape == (5, 7)
    assert dual.alignment_features(features, "branch_b").shape == (5, 7)
    with pytest.raises(ValueError, match="Unknown alignment scope"):
        dual.alignment_features(features, "selected_branch")


@pytest.mark.parametrize(
    ("branch_a", "branch_b", "message"),
    [
        ([0, 1], [1, 2, 3], "must not overlap"),
        ([0, 1], [2], "cover every input feature"),
        ([0, 0], [1, 2, 3], "must not contain duplicates"),
        ([], [0, 1, 2, 3], "at least one feature"),
    ],
)
def test_dual_partition_validation(branch_a, branch_b, message):
    with pytest.raises(ValueError, match=message):
        DiscreteHazardDomainNet(
            4,
            (8,),
            0.0,
            architecture="dual",
            branch_a_indices=branch_a,
            branch_b_indices=branch_b,
        )


def test_dual_requires_both_partition_lists_and_single_rejects_them():
    with pytest.raises(ValueError, match="requires both"):
        DiscreteHazardDomainNet(
            4, (8,), 0.0, architecture="dual", branch_a_indices=[0, 1]
        )
    with pytest.raises(ValueError, match="does not accept"):
        DiscreteHazardDomainNet(
            4,
            (8,),
            0.0,
            architecture="single",
            branch_a_indices=[0, 1],
            branch_b_indices=[2, 3],
        )


def test_architecture_matched_families_have_identical_parameter_counts():
    models = {
        family: DiscreteHazardDomainNet(
            6,
            (9, 5),
            0.1,
            architecture=spec.architecture,
            branch_a_indices=[0, 1, 2, 3] if spec.architecture == "dual" else None,
            branch_b_indices=[4, 5] if spec.architecture == "dual" else None,
        )
        for family, spec in SURVIVAL_FAMILY_SPECS.items()
    }

    assert len({_parameter_count(models[name]) for name in ("A", "B")}) == 1
    assert len({_parameter_count(models[name]) for name in ("C", "D1", "D2", "E", "G")}) == 1


def test_survival_family_mapping_is_complete_and_explicit():
    expected = {
        "source_only": ("single", None, "source_only", False),
        "target_only": ("single", None, "target_only", False),
        "A": ("single", None, "update", False),
        "B": ("single", "global", "update", False),
        "C": ("dual", None, "update", False),
        "D1": ("dual", "branch_a", "update", False),
        "D2": ("dual", "branch_b", "update", False),
        "G": ("dual", "global", "update", False),
        "E": ("dual", "branch_a", "update", True),
    }

    assert set(SURVIVAL_FAMILY_SPECS) == set(expected)
    for family, values in expected.items():
        spec = survival_family_spec(family)
        assert (spec.architecture, spec.alignment_scope, spec.training_mode, spec.random_partition) == values
    with pytest.raises(ValueError, match="Unknown survival model family"):
        survival_family_spec("missing")
