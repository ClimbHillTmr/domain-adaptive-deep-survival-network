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
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...],
        dropout: float,
        physio_indices: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.physio_indices = physio_indices
        
        if physio_indices is not None:
            self.treat_indices = [i for i in range(input_dim) if i not in physio_indices]
            physio_dim = len(physio_indices)
            treat_dim = len(self.treat_indices)
            
            physio_layers: list[nn.Module] = []
            previous = physio_dim
            for hidden in hidden_dims:
                physio_layers.extend([nn.Linear(previous, hidden), nn.ReLU(), nn.Dropout(dropout)])
                previous = hidden
            self.physio_extractor = nn.Sequential(*physio_layers)
            
            treat_layers: list[nn.Module] = []
            previous = treat_dim
            for hidden in hidden_dims:
                treat_layers.extend([nn.Linear(previous, hidden), nn.ReLU(), nn.Dropout(dropout)])
                previous = hidden
            self.treat_extractor = nn.Sequential(*treat_layers)
            
            self.head = nn.Linear(hidden_dims[-1] * 2, 1)
        else:
            feature_layers: list[nn.Module] = []
            previous = input_dim
            for hidden in hidden_dims:
                feature_layers.extend([nn.Linear(previous, hidden), nn.ReLU(), nn.Dropout(dropout)])
                previous = hidden
            self.feature_extractor = nn.Sequential(*feature_layers)
            self.head = nn.Linear(previous, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.physio_indices is not None:
            physio_feat = features[:, self.physio_indices]
            treat_feat = features[:, self.treat_indices]
            physio_out = self.physio_extractor(physio_feat)
            treat_out = self.treat_extractor(treat_feat)
            combined = torch.cat([physio_out, treat_out], dim=-1)
            return self.head(combined).squeeze(-1)
        else:
            return self.head(self.feature_extractor(features)).squeeze(-1)

    def get_features(self, features: torch.Tensor) -> torch.Tensor:
        if self.physio_indices is not None:
            physio_feat = features[:, self.physio_indices]
            treat_feat = features[:, self.treat_indices]
            return torch.cat(
                [self.physio_extractor(physio_feat), self.treat_extractor(treat_feat)], dim=-1
            )
        else:
            return self.feature_extractor(features)

    def get_alignment_features(self, features: torch.Tensor, scope: str) -> torch.Tensor:
        if scope == "global":
            return self.get_features(features)
        if scope == "selected_branch":
            if self.physio_indices is None:
                raise ValueError("Selected-branch alignment requires a dual-branch model.")
            return self.physio_extractor(features[:, self.physio_indices])
        raise ValueError(f"Unknown alignment scope: {scope}")


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        ce_loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        pt = labels * probs + (1 - labels) * (1 - probs)
        focal_weight = self.alpha * labels + (1 - self.alpha) * (1 - labels)
        focal_weight *= (1 - pt) ** self.gamma
        return (focal_weight * ce_loss).mean()


def coral_loss(source_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
    d = source_features.size(1)
    source_cov = torch.mm(source_features.T, source_features) / (source_features.size(0) - 1)
    target_cov = torch.mm(target_features.T, target_features) / (target_features.size(0) - 1)
    return torch.norm(source_cov - target_cov, p='fro') ** 2 / (4 * d ** 2)


def mmd_loss(source_features: torch.Tensor, target_features: torch.Tensor, kernel_type: str = 'rbf') -> torch.Tensor:
    n = source_features.size(0)
    m = target_features.size(0)
    if kernel_type == 'linear':
        k_ss = torch.mm(source_features, source_features.T)
        k_st = torch.mm(source_features, target_features.T)
        k_tt = torch.mm(target_features, target_features.T)
    elif kernel_type == 'rbf':
        sigma = torch.median(torch.cdist(source_features, target_features))
        if sigma < 1e-6:
            sigma = torch.tensor(1.0, device=source_features.device)
        k_ss = torch.exp(-torch.cdist(source_features, source_features) ** 2 / (2 * sigma ** 2))
        k_st = torch.exp(-torch.cdist(source_features, target_features) ** 2 / (2 * sigma ** 2))
        k_tt = torch.exp(-torch.cdist(target_features, target_features) ** 2 / (2 * sigma ** 2))
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")
    mmd = torch.mean(k_ss) - 2 * torch.mean(k_st) + torch.mean(k_tt)
    return mmd


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, coeff):
        ctx.coeff = coeff
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.coeff, None


def grad_reverse(x, coeff):
    return GradReverse.apply(x, coeff)


class DomainClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):
        return self.net(x)


class CDANBinaryMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
        domain_hidden: int = 64,
        tokenizer_type: str = "linear",
    ) -> None:
        super().__init__()
        self.tokenizer_type = tokenizer_type
        self.d_model = d_model

        if tokenizer_type == "kan":
            from src.models.cdan_gsn import BSplineKANTokenizer
            self.tokenizer = BSplineKANTokenizer(
                num_features=input_dim,
                d_model=d_model,
                grid_size=8,
                spline_order=3,
                grid_range=(-4.0, 4.0),
            )
        else:
            self.feature_weight = nn.Parameter(torch.randn(input_dim, d_model) * 0.02)
            self.feature_bias = nn.Parameter(torch.zeros(input_dim, d_model))

        self.feature_pos = nn.Parameter(torch.randn(input_dim, d_model) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        self.hazard_head = nn.Linear(d_model, 1)
        self.feature_mask = nn.Parameter(torch.ones(d_model) * 0.1)

        self.domain_head = DomainClassifier(d_model + 1, domain_hidden, dropout)

    def forward(self, features: torch.Tensor, grl_coeff=None):
        if self.tokenizer_type == "kan":
            tokens = self.tokenizer(features)
        else:
            tokens = features.unsqueeze(-1) * self.feature_weight.unsqueeze(0) + self.feature_bias.unsqueeze(0)

        tokens = tokens + self.feature_pos.unsqueeze(0)

        b_size = features.size(0)
        cls_tokens = self.cls_token.expand(b_size, -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)

        encoded = self.transformer(tokens)

        cls_emb = self.norm(encoded[:, 0, :])
        hazard = self.hazard_head(cls_emb).squeeze(-1)

        domain_logits = None
        mask_l1_loss = torch.tensor(0.0, device=features.device)

        if grl_coeff is not None:
            mask = torch.sigmoid(self.feature_mask)
            mask_l1_loss = torch.mean(mask)

            masked_cls = cls_emb * mask
            cls_grl = grad_reverse(masked_cls, grl_coeff)

            hazard_prob = torch.sigmoid(hazard.detach())

            domain_input = torch.cat([cls_grl, hazard_prob.unsqueeze(-1)], dim=-1)
            domain_logits = self.domain_head(domain_input)

        return hazard, domain_logits, mask_l1_loss


def predict_mlp(model, x: np.ndarray, device: torch.device, is_cdan: bool = False) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        if is_cdan:
            logits, _, _ = model(torch.as_tensor(x, dtype=torch.float32, device=device))
        else:
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
    model,
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
    x_domain_train: np.ndarray = None,
    adversarial_weight: float = 0.01,
    mask_l1_weight: float = 0.01,
    is_cdan: bool = False,
    x_target_train: np.ndarray = None,
    x_source_alignment: np.ndarray = None,
    coral_weight: float = 0.0,
    mmd_weight: float = 0.0,
    alignment_scope: str = "global",
) -> dict[str, float | int]:
    if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
        raise ValueError("MLP fitting requires both classes in train and validation folds.")
    generator = torch.Generator().manual_seed(seed)
    
    if is_cdan and x_domain_train is not None:
        dataset = TensorDataset(
            torch.as_tensor(x_train, dtype=torch.float32),
            torch.as_tensor(y_train),
            torch.as_tensor(x_domain_train),
        )
    else:
        dataset = TensorDataset(torch.as_tensor(x_train, dtype=torch.float32), torch.as_tensor(y_train))
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    
    if x_target_train is not None:
        target_loader = DataLoader(
            TensorDataset(torch.as_tensor(x_target_train, dtype=torch.float32)),
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed + 1),
        )
        if x_source_alignment is not None:
            source_alignment_loader = DataLoader(
                TensorDataset(torch.as_tensor(x_source_alignment, dtype=torch.float32)),
                batch_size=batch_size,
                shuffle=True,
                generator=torch.Generator().manual_seed(seed + 2),
            )
    
    positives = float(y_train.sum())
    alpha = (len(y_train) - positives) / len(y_train)
    criterion = FocalLoss(alpha=alpha, gamma=2.0)
    domain_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    model.to(device)
    best_state = copy.deepcopy(model.state_dict())
    best_auc, best_epoch, remaining = -np.inf, 0, patience
    
    for epoch in range(1, max_epochs + 1):
        model.train()
        grl_coeff = 2.0 / (1.0 + np.exp(-10.0 * epoch / max_epochs)) - 1.0
        
        if x_target_train is not None:
            target_iter = iter(target_loader)
            if x_source_alignment is not None:
                source_alignment_iter = iter(source_alignment_loader)
        
        for batch in loader:
            optimizer.zero_grad()
            
            if is_cdan and x_domain_train is not None:
                features, labels, domain_labels = batch
                features, labels, domain_labels = features.to(device), labels.to(device), domain_labels.to(device)
                hazard, domain_logits, mask_l1_loss = model(features, grl_coeff=grl_coeff)
                task_loss = criterion(hazard, labels)
                
                if domain_logits is not None:
                    adv_loss = domain_criterion(domain_logits, domain_labels.long())
                else:
                    adv_loss = torch.tensor(0.0, device=device)
                
                loss = task_loss + adversarial_weight * adv_loss + mask_l1_weight * mask_l1_loss
            else:
                features, labels = batch
                features, labels = features.to(device), labels.to(device)
                if is_cdan:
                    hazard, _, _ = model(features)
                else:
                    hazard = model(features)
                loss = criterion(hazard, labels)
            
            if coral_weight > 0.0 and x_target_train is not None:
                try:
                    target_batch = next(target_iter)
                except StopIteration:
                    target_iter = iter(target_loader)
                    target_batch = next(target_iter)
                
                target_features = target_batch[0].to(device)
                if x_source_alignment is not None:
                    try:
                        source_batch = next(source_alignment_iter)
                    except StopIteration:
                        source_alignment_iter = iter(source_alignment_loader)
                        source_batch = next(source_alignment_iter)
                    alignment_source_features = source_batch[0].to(device)
                else:
                    alignment_source_features = features
                source_feat = model.get_alignment_features(alignment_source_features, alignment_scope)
                target_feat = model.get_alignment_features(target_features, alignment_scope)
                loss += coral_weight * coral_loss(source_feat, target_feat)
            
            if mmd_weight > 0.0 and x_target_train is not None:
                try:
                    target_batch = next(target_iter)
                except StopIteration:
                    target_iter = iter(target_loader)
                    target_batch = next(target_iter)
                
                target_features = target_batch[0].to(device)
                if x_source_alignment is not None:
                    try:
                        source_batch = next(source_alignment_iter)
                    except StopIteration:
                        source_alignment_iter = iter(source_alignment_loader)
                        source_batch = next(source_alignment_iter)
                    alignment_source_features = source_batch[0].to(device)
                else:
                    alignment_source_features = features
                source_feat = model.get_alignment_features(alignment_source_features, alignment_scope)
                target_feat = model.get_alignment_features(target_features, alignment_scope)
                loss += mmd_weight * mmd_loss(source_feat, target_feat)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        auc = roc_auc_score(y_val, predict_mlp(model, x_val, device, is_cdan=is_cdan))
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
    use_cdan: bool = False,
    use_coral: bool = False,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    domain_hidden: int = 64,
    adversarial_weight: float = 0.01,
    mask_l1_weight: float = 0.01,
    coral_weight: float = 0.1,
    use_mmd: bool = False,
    mmd_weight: float = 1.0,
    physio_indices: list[int] | None = None,
    alignment_scope: str = "global",
    pooled_update: bool = False,
) -> tuple:
    torch.manual_seed(seed)
    np.random.seed(seed)
    input_dim = x_source_train.shape[1]
    
    if use_cdan:
        source_model = CDANBinaryMLP(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
            domain_hidden=domain_hidden,
            tokenizer_type="linear",
        )
    else:
        source_model = BinaryMLP(input_dim, hidden_dims, dropout, physio_indices=physio_indices)
    
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
        is_cdan=use_cdan,
    )
    
    updated_model = copy.deepcopy(source_model)
    
    if use_cdan:
        x_finetune_train = np.vstack([x_source_train, x_target_train])
        y_finetune_train = np.concatenate([y_source_train, y_target_train])
        domain_labels_train = np.concatenate([
            np.zeros(len(x_source_train), dtype=np.int64),
            np.ones(len(x_target_train), dtype=np.int64),
        ])
        
        finetune = _fit_mlp_phase(
            updated_model,
            x_finetune_train,
            y_finetune_train,
            x_target_val,
            y_target_val,
            device=device,
            learning_rate=finetune_learning_rate,
            batch_size=batch_size,
            max_epochs=finetune_epochs,
            patience=patience,
            seed=seed + 1,
            x_domain_train=domain_labels_train,
            adversarial_weight=adversarial_weight,
            mask_l1_weight=mask_l1_weight,
            is_cdan=use_cdan,
        )
    elif use_coral:
        x_finetune_train = np.vstack([x_source_train, x_target_train])
        y_finetune_train = np.concatenate([y_source_train, y_target_train])
        
        finetune = _fit_mlp_phase(
            updated_model,
            x_finetune_train,
            y_finetune_train,
            x_target_val,
            y_target_val,
            device=device,
            learning_rate=finetune_learning_rate,
            batch_size=batch_size,
            max_epochs=finetune_epochs,
            patience=patience,
            seed=seed + 1,
            x_target_train=x_target_train,
            x_source_alignment=x_source_train,
            coral_weight=coral_weight,
            alignment_scope=alignment_scope,
        )
    elif use_mmd:
        x_finetune_train = np.vstack([x_source_train, x_target_train])
        y_finetune_train = np.concatenate([y_source_train, y_target_train])
        
        finetune = _fit_mlp_phase(
            updated_model,
            x_finetune_train,
            y_finetune_train,
            x_target_val,
            y_target_val,
            device=device,
            learning_rate=finetune_learning_rate,
            batch_size=batch_size,
            max_epochs=finetune_epochs,
            patience=patience,
            seed=seed + 1,
            x_target_train=x_target_train,
            x_source_alignment=x_source_train,
            mmd_weight=mmd_weight,
            alignment_scope=alignment_scope,
        )
    else:
        if pooled_update:
            x_update_train = np.vstack([x_source_train, x_target_train])
            y_update_train = np.concatenate([y_source_train, y_target_train])
        else:
            x_update_train = x_target_train
            y_update_train = y_target_train
        finetune = _fit_mlp_phase(
            updated_model,
            x_update_train,
            y_update_train,
            x_target_val,
            y_target_val,
            device=device,
            learning_rate=finetune_learning_rate,
            batch_size=batch_size,
            max_epochs=finetune_epochs,
            patience=patience,
            seed=seed + 1,
            is_cdan=use_cdan,
        )
    
    return source_model, updated_model, {"source_pretrain": pretrain, "target_update": finetune}
