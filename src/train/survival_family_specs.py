"""Pure-data specifications for V4 discrete survival model families."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from src.models.discrete_hazard_domain import AlignmentScope, Architecture


@dataclass(frozen=True)
class SurvivalFamilySpec:
    architecture: Architecture
    alignment_scope: AlignmentScope | None
    training_mode: Literal["source_only", "target_only", "update"]
    random_partition: bool = False


SURVIVAL_FAMILY_SPECS = MappingProxyType(
    {
        "source_only": SurvivalFamilySpec("single", None, "source_only"),
        "target_only": SurvivalFamilySpec("single", None, "target_only"),
        "A": SurvivalFamilySpec("single", None, "update"),
        "B": SurvivalFamilySpec("single", "global", "update"),
        "C": SurvivalFamilySpec("dual", None, "update"),
        "D1": SurvivalFamilySpec("dual", "branch_a", "update"),
        "D2": SurvivalFamilySpec("dual", "branch_b", "update"),
        "G": SurvivalFamilySpec("dual", "global", "update"),
        "E": SurvivalFamilySpec("dual", "branch_a", "update", random_partition=True),
    }
)


def survival_family_spec(family: str) -> SurvivalFamilySpec:
    try:
        return SURVIVAL_FAMILY_SPECS[family]
    except KeyError as error:
        raise ValueError(f"Unknown survival model family: {family}") from error
