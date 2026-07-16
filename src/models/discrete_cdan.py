"""Small conditional domain-adversarial model for discrete dialysis hazards."""

from __future__ import annotations

import torch
from torch import nn

from src.models.cdan_gsn import grad_reverse


class DiscreteHazardCDAN(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for hidden in hidden_dims:
            layers.extend((nn.Linear(previous, hidden), nn.ReLU(), nn.Dropout(dropout)))
            previous = hidden
        self.encoder = nn.Sequential(*layers)
        self.hazard_head = nn.Linear(previous, 1)
        self.domain_head = nn.Sequential(
            nn.Linear(previous + 1, previous),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(previous, 2),
        )

    def forward(self, features: torch.Tensor, grl_coeff: float | None = None):
        representation = self.encoder(features)
        hazard_logit = self.hazard_head(representation).squeeze(-1)
        domain_logit = None
        if grl_coeff is not None:
            conditioned = torch.cat(
                (grad_reverse(representation, grl_coeff), torch.sigmoid(hazard_logit).detach().unsqueeze(-1)),
                dim=1,
            )
            domain_logit = self.domain_head(conditioned)
        return hazard_logit, domain_logit
