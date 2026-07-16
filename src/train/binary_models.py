"""Minimal binary baselines for the reset IDH study."""

from __future__ import annotations

import copy

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def fit_logistic_with_validation(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    c_values: tuple[float, ...],
    seed: int,
) -> LogisticRegression:
    if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
        raise ValueError("Logistic model selection requires both classes in train and validation folds.")
    best_c, best_auc = c_values[0], -np.inf
    for c_value in c_values:
        candidate = LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
            solver="lbfgs",
        )
        candidate.fit(x_train, y_train)
        auc = roc_auc_score(y_val, candidate.predict_proba(x_val)[:, 1])
        if auc > best_auc:
            best_c, best_auc = c_value, auc
    model = LogisticRegression(
        C=best_c,
        class_weight="balanced",
        max_iter=2000,
        random_state=seed,
        solver="lbfgs",
    )
    model.fit(x_train, y_train)
    model.selected_c_ = best_c
    model.validation_auc_ = float(best_auc)
    return model


class BinaryMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(previous, hidden), nn.ReLU(), nn.Dropout(dropout)])
            previous = hidden
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def predict_mlp(model: BinaryMLP, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(x, dtype=torch.float32, device=device))
    return torch.sigmoid(logits).cpu().numpy()


def fit_probability_calibrator(y_val: np.ndarray, probability: np.ndarray) -> LogisticRegression:
    """Fit Platt calibration on the validation cohort without class reweighting."""
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    calibrator = LogisticRegression(C=1e6, max_iter=1000, solver="lbfgs")
    calibrator.fit(logit, y_val)
    return calibrator


def calibrate_probability(calibrator: LogisticRegression, probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    return calibrator.predict_proba(logit)[:, 1]


def _fit_mlp_phase(
    model: BinaryMLP,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    device: torch.device,
    learning_rate: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    seed: int,
) -> dict[str, float | int]:
    if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
        raise ValueError("MLP fitting requires both classes in train and validation folds.")
    generator = torch.Generator().manual_seed(seed)
    dataset = TensorDataset(torch.as_tensor(x_train, dtype=torch.float32), torch.as_tensor(y_train))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    positives = float(y_train.sum())
    pos_weight = torch.tensor((len(y_train) - positives) / positives, dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    model.to(device)
    best_state = copy.deepcopy(model.state_dict())
    best_auc, best_epoch, remaining = -np.inf, 0, patience
    for epoch in range(1, max_epochs + 1):
        model.train()
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(features), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        auc = roc_auc_score(y_val, predict_mlp(model, x_val, device))
        if auc > best_auc:
            best_auc, best_epoch = float(auc), epoch
            best_state = copy.deepcopy(model.state_dict())
            remaining = patience
        else:
            remaining -= 1
            if remaining == 0:
                break
    model.load_state_dict(best_state)
    return {"best_validation_auc": best_auc, "best_epoch": best_epoch}


def fit_source_and_update_mlp(
    x_source_train: np.ndarray,
    y_source_train: np.ndarray,
    x_source_val: np.ndarray,
    y_source_val: np.ndarray,
    x_target_train: np.ndarray,
    y_target_train: np.ndarray,
    x_target_val: np.ndarray,
    y_target_val: np.ndarray,
    *,
    hidden_dims: tuple[int, ...],
    dropout: float,
    pretrain_learning_rate: float,
    finetune_learning_rate: float,
    batch_size: int,
    pretrain_epochs: int,
    finetune_epochs: int,
    patience: int,
    seed: int,
    device: torch.device,
) -> tuple[BinaryMLP, BinaryMLP, dict[str, object]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    source_model = BinaryMLP(x_source_train.shape[1], hidden_dims, dropout)
    pretrain = _fit_mlp_phase(
        source_model,
        x_source_train,
        y_source_train,
        x_source_val,
        y_source_val,
        device=device,
        learning_rate=pretrain_learning_rate,
        batch_size=batch_size,
        max_epochs=pretrain_epochs,
        patience=patience,
        seed=seed,
    )
    updated_model = copy.deepcopy(source_model)
    finetune = _fit_mlp_phase(
        updated_model,
        x_target_train,
        y_target_train,
        x_target_val,
        y_target_val,
        device=device,
        learning_rate=finetune_learning_rate,
        batch_size=batch_size,
        max_epochs=finetune_epochs,
        patience=patience,
        seed=seed + 1,
    )
    return source_model, updated_model, {"source_pretrain": pretrain, "target_update": finetune}
