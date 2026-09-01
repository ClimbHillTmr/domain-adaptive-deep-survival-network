"""Discrete hazard network with architecture-controlled alignment representations."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torch.autograd import Function

Architecture = Literal["single", "dual"]
AlignmentScope = Literal["global", "branch_a", "branch_b"]


class GradientReversalFunction(Function):
    """Autograd function that flips the sign of the incoming gradient."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """Module wrapper around :class:`GradientReversalFunction`."""

    def __init__(self, lambda_: float = 1.0) -> None:
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambda_)


class DomainDiscriminator(nn.Module):
    """Binary domain classifier preceded by a gradient reversal layer."""

    def __init__(self, input_dim: int, hidden_dim: int = 32, dropout: float = 0.2) -> None:
        super().__init__()
        self.grl = GradientReversalLayer()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.grl(x))


def _make_encoder(input_dim: int, hidden_dims: tuple[int, ...], dropout: float) -> nn.Sequential:
    if input_dim < 1:
        raise ValueError("Encoder input dimension must be positive.")
    if not hidden_dims or any(hidden < 1 for hidden in hidden_dims):
        raise ValueError("hidden_dims must contain positive dimensions.")
    if not 0 <= dropout < 1:
        raise ValueError("dropout must be in [0, 1).")
    layers: list[nn.Module] = []
    previous = input_dim
    for hidden in hidden_dims:
        layers.extend((nn.Linear(previous, hidden), nn.ReLU(), nn.Dropout(dropout)))
        previous = hidden
    return nn.Sequential(*layers)


class DiscreteHazardDomainNet(nn.Module):
    """Single- or dual-encoder network returning four discrete hazard logits."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...],
        dropout: float,
        *,
        architecture: Architecture = "single",
        branch_a_indices: list[int] | tuple[int, ...] | None = None,
        branch_b_indices: list[int] | tuple[int, ...] | None = None,
        num_intervals: int = 4,
        use_dann: bool = False,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be positive.")
        if num_intervals != 4:
            raise ValueError("V4 discrete survival models require exactly four intervals.")
        if architecture not in {"single", "dual"}:
            raise ValueError(f"Unknown architecture: {architecture}")
        self.input_dim = input_dim
        self.architecture = architecture
        self.use_dann = use_dann

        if architecture == "single":
            if branch_a_indices is not None or branch_b_indices is not None:
                raise ValueError("Single architecture does not accept branch index lists.")
            self.encoder = _make_encoder(input_dim, hidden_dims, dropout)
            representation_dim = hidden_dims[-1]
        else:
            branch_a, branch_b = self._validated_partition(
                input_dim, branch_a_indices, branch_b_indices
            )
            self.register_buffer("branch_a_indices", torch.tensor(branch_a, dtype=torch.long))
            self.register_buffer("branch_b_indices", torch.tensor(branch_b, dtype=torch.long))
            self.branch_a_encoder = _make_encoder(len(branch_a), hidden_dims, dropout)
            self.branch_b_encoder = _make_encoder(len(branch_b), hidden_dims, dropout)
            representation_dim = 2 * hidden_dims[-1]
        self.hazard_head = nn.Linear(representation_dim, num_intervals)
        if use_dann:
            self.domain_discriminator = DomainDiscriminator(representation_dim)

    @staticmethod
    def _validated_partition(
        input_dim: int,
        branch_a_indices: list[int] | tuple[int, ...] | None,
        branch_b_indices: list[int] | tuple[int, ...] | None,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        if branch_a_indices is None or branch_b_indices is None:
            raise ValueError("Dual architecture requires both branch index lists.")
        branch_a = tuple(branch_a_indices)
        branch_b = tuple(branch_b_indices)
        if not branch_a or not branch_b:
            raise ValueError("Both dual branches must contain at least one feature.")
        if any(isinstance(index, bool) or not isinstance(index, int) for index in branch_a + branch_b):
            raise TypeError("Branch indices must be integers.")
        if len(set(branch_a)) != len(branch_a) or len(set(branch_b)) != len(branch_b):
            raise ValueError("Branch index lists must not contain duplicates.")
        if set(branch_a) & set(branch_b):
            raise ValueError("Dual branch index lists must not overlap.")
        if set(branch_a) | set(branch_b) != set(range(input_dim)):
            raise ValueError("Dual branch index lists must cover every input feature exactly once.")
        return branch_a, branch_b

    def _branch_representations(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.architecture != "dual":
            raise ValueError("Branch representations require a dual architecture.")
        return (
            self.branch_a_encoder(features.index_select(1, self.branch_a_indices)),
            self.branch_b_encoder(features.index_select(1, self.branch_b_indices)),
        )

    def representation(self, features: torch.Tensor) -> torch.Tensor:
        if self.architecture == "single":
            return self.encoder(features)
        branch_a, branch_b = self._branch_representations(features)
        return torch.cat((branch_a, branch_b), dim=1)

    def alignment_features(self, features: torch.Tensor, scope: AlignmentScope = "global") -> torch.Tensor:
        if scope == "global":
            return self.representation(features)
        if scope not in {"branch_a", "branch_b"}:
            raise ValueError(f"Unknown alignment scope: {scope}")
        if self.architecture == "single":
            raise ValueError("Branch alignment scope requires a dual architecture.")
        branch_a, branch_b = self._branch_representations(features)
        return branch_a if scope == "branch_a" else branch_b

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.hazard_head(self.representation(features))

    def domain_logits(self, features: torch.Tensor) -> torch.Tensor:
        """Run features through the encoder and the DANN domain discriminator."""
        if not self.use_dann:
            raise ValueError("domain_logits requires use_dann=True at construction time.")
        return self.domain_discriminator(self.representation(features))
