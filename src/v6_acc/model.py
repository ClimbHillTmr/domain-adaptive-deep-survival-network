"""v6.0 — 统一模型骨架 MLP 19→64→32→4 (hazard logits)。

主线三 family (target_only / source_only / supervised_update) 共用此骨架，
权重结构相同，仅训练数据与阶段不同。无 dual-branch（对齐放 supplementary）。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DiscreteHazardMLP(nn.Module):
    """4 间隔离散生存 hazard logit 分类器。

    输入：标准化后 19 维特征向量 → 输出 4 维 hazard logits。
    """

    def __init__(
        self,
        in_dim: int = 19,
        hidden_sizes: tuple[int, int] = (64, 32),
        n_intervals: int = 4,
        dropout: float = 0.2,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)  # 便于跨 seed 复现
        layers: list[nn.Module] = []
        dims = [in_dim, *hidden_sizes]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        self.backbone = nn.Sequential(*layers)
        self.hazard_head = nn.Linear(hidden_sizes[-1], n_intervals)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """返回 (n, 4) hazard logits。"""
        return self.hazard_head(self.backbone(features))