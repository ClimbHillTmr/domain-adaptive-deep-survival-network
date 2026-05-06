import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
import os
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.models.dadsn_model import DADSN, cox_partial_log_likelihood, coral_loss, mmd_loss, gradient_reversal, DomainClassifier
from src.metrics.survival_metrics import (
    concordance_index_censored,
    integrated_brier_score,
)

log = logging.getLogger(__name__)


class WarmupCosineAnnealing:
    """Learning rate scheduler with warmup and cosine annealing."""

    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (self.current_epoch / self.warmup_epochs)
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / (
                self.total_epochs - self.warmup_epochs
            )
            lr = self.base_lr * (1 + np.cos(np.pi * progress)) / 2

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr


def weighted_cox_partial_log_likelihood(log_hazard, event, time, event_weight=2.0):
    """
    Compute weighted Cox Partial Log-Likelihood.
    Events get higher weight to handle class imbalance.
    """
    log_hazard = log_hazard.squeeze()
    if log_hazard.dim() == 0 or len(log_hazard) == 0:
        return torch.tensor(0.0, device=log_hazard.device, requires_grad=True)

    sort_idx = torch.argsort(time, descending=True)
    log_hazard_sorted = log_hazard[sort_idx]
    event_sorted = event[sort_idx]

    log_hazard_max = log_hazard_sorted.max()
    cumsum_exp = torch.cumsum(torch.exp(log_hazard_sorted - log_hazard_max), dim=0)
    log_risk = torch.log(cumsum_exp + 1e-10) + log_hazard_max

    log_partial_lik = log_hazard_sorted - log_risk

    mask = event_sorted > 0
    if mask.sum() == 0:
        return torch.tensor(0.0, device=log_hazard.device, requires_grad=True)

    # Apply weight to events
    weights = torch.ones_like(log_partial_lik)
    weights[mask] = event_weight

    neg_log_lik = -(log_partial_lik * weights)[mask].mean()
    return neg_log_lik


def _build_tensor_dataset(data_dict, include_labels=True):
    static = torch.tensor(data_dict["static"], dtype=torch.float32)
    dynamic = torch.tensor(data_dict["dynamic"], dtype=torch.float32)
    if include_labels and "event" in data_dict["targets"]:
        event = torch.tensor(data_dict["targets"]["event"], dtype=torch.float32)
        duration = torch.tensor(data_dict["targets"]["duration"], dtype=torch.float32)
    else:
        event = torch.zeros(data_dict["static"].shape[0], dtype=torch.float32)
        duration = torch.zeros(data_dict["static"].shape[0], dtype=torch.float32)
    return TensorDataset(static, dynamic, event, duration)


def create_mixed_dataloader(source_data, target_data, batch_size, shuffle=True):
    ds_source = _build_tensor_dataset(source_data, include_labels=True)
    ds_target = _build_tensor_dataset(target_data, include_labels=False)

    loader_source = DataLoader(
        ds_source, batch_size=batch_size, shuffle=shuffle, drop_last=True
    )
    loader_target = DataLoader(
        ds_target, batch_size=batch_size, shuffle=shuffle, drop_last=True
    )
    return loader_source, loader_target


def create_source_only_dataloader(source_data, batch_size, shuffle=True):
    ds_source = _build_tensor_dataset(source_data, include_labels=True)
    return DataLoader(ds_source, batch_size=batch_size, shuffle=shuffle, drop_last=True)


def self_training_on_target(
    model,
    source_data,
    target_data,
    device,
    config,
    use_weighted_cox=False,
    event_weight=2.0,
    n_epochs=5,
    pseudo_quantile=0.3,
):
    """
    Self-training on target domain using pseudo-labels.

    Strategy:
    1. Use current model to predict risk scores on target domain
    2. Select top/bottom quantile as high/low risk pseudo-labels
    3. Fine-tune model on mix of source + pseudo-labeled target data
    """
    model.eval()

    # Get pseudo-labels for target domain
    with torch.no_grad():
        tgt_static = torch.tensor(target_data["static"], dtype=torch.float32).to(device)
        tgt_dyn = torch.tensor(target_data["dynamic"], dtype=torch.float32).to(device)

        all_risks = []
        batch_size = 512
        for i in range(0, len(tgt_static), batch_size):
            batch_s = tgt_static[i:i+batch_size]
            batch_d = tgt_dyn[i:i+batch_size]
            log_h, _ = model(batch_s, batch_d)
            all_risks.append(log_h.cpu().numpy())

        risk_scores = np.concatenate(all_risks, axis=0).squeeze()

    # Select confident pseudo-labels
    high_risk_thresh = np.quantile(risk_scores, 1 - pseudo_quantile)
    low_risk_thresh = np.quantile(risk_scores, pseudo_quantile)

    high_risk_mask = risk_scores >= high_risk_thresh
    low_risk_mask = risk_scores <= low_risk_thresh

    # Create pseudo-labeled target dataset
    # High risk = event=1, Low risk = event=0
    # Duration: use median of target domain for simplicity
    tgt_duration = target_data["targets"]["duration"].copy()
    median_dur = np.median(tgt_duration[tgt_duration > 0])

    pseudo_event = np.zeros(len(risk_scores), dtype=np.float32)
    pseudo_event[high_risk_mask] = 1.0
    pseudo_duration = np.full(len(risk_scores), median_dur, dtype=np.float32)

    # Only use confident samples
    confident_mask = high_risk_mask | low_risk_mask
    if confident_mask.sum() < 100:
        log.warning("Too few confident pseudo-labels, skipping self-training")
        return model, evaluate_survival_metrics(model, target_data, device)

    pseudo_data = {
        "static": target_data["static"][confident_mask],
        "dynamic": target_data["dynamic"][confident_mask],
        "targets": {
            "event": pseudo_event[confident_mask],
            "duration": pseudo_duration[confident_mask],
        },
    }

    # Create mixed loader: source + pseudo-labeled target
    ds_src = _build_tensor_dataset(source_data, include_labels=True)
    ds_pseudo = TensorDataset(
        torch.tensor(pseudo_data["static"], dtype=torch.float32),
        torch.tensor(pseudo_data["dynamic"], dtype=torch.float32),
        torch.tensor(pseudo_data["targets"]["event"], dtype=torch.float32),
        torch.tensor(pseudo_data["targets"]["duration"], dtype=torch.float32),
    )

    # Weighted sampler: source samples get weight 1.0, pseudo-target get weight 0.5
    combined_dataset = torch.utils.data.ConcatDataset([ds_src, ds_pseudo])

    loader = DataLoader(
        combined_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        drop_last=True,
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate * 0.1,  # Lower LR for fine-tuning
        weight_decay=config.training.weight_decay,
    )

    cox_loss_fn = (
        weighted_cox_partial_log_likelihood
        if use_weighted_cox
        else cox_partial_log_likelihood
    )

    model.train()
    for epoch in range(n_epochs):
        total_loss = 0.0
        n_batches = 0

        for batch_static, batch_dyn, batch_event, batch_time in loader:
            batch_static = batch_static.to(device)
            batch_dyn = batch_dyn.to(device)
            batch_event = batch_event.to(device)
            batch_time = batch_time.to(device)

            valid_mask = (
                (batch_time > 0) & (~torch.isnan(batch_event)) & (~torch.isnan(batch_time))
            )
            if valid_mask.sum() <= 1:
                continue

            batch_static = batch_static[valid_mask]
            batch_dyn = batch_dyn[valid_mask]
            batch_event = batch_event[valid_mask]
            batch_time = batch_time[valid_mask]

            optimizer.zero_grad()
            log_hazard, _ = model(batch_static, batch_dyn)

            if use_weighted_cox:
                loss = cox_loss_fn(
                    log_hazard.squeeze(), batch_event, batch_time, event_weight
                )
            else:
                loss = cox_loss_fn(log_hazard.squeeze(), batch_event, batch_time)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        log.info(f"  Self-training epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")

    model.eval()
    metrics = evaluate_survival_metrics(model, target_data, device)
    return model, metrics


def train_epoch_dadsn(
    model,
    loader_src,
    loader_tgt,
    optimizer,
    device,
    coral_lambda=0.0,
    use_weighted_cox=False,
    event_weight=2.0,
    scheduler=None,
    da_method="coral",
    current_epoch=0,
    da_warmup_epochs=15,
    max_grad_norm=1.0,
    multi_level_da=False,
    static_lambda=0.03,
    dyn_lambda=0.03,
):
    model.train()
    total_surv_loss = 0.0
    total_coral_loss = 0.0
    total_static_da_loss = 0.0
    total_dyn_da_loss = 0.0
    n_batches = 0

    iter_tgt = iter(loader_tgt) if loader_tgt else None

    cox_loss_fn = (
        weighted_cox_partial_log_likelihood
        if use_weighted_cox
        else cox_partial_log_likelihood
    )

    # Linear warmup for domain adaptation loss
    if coral_lambda > 0 and current_epoch < da_warmup_epochs:
        effective_lambda = coral_lambda * (current_epoch / da_warmup_epochs)
        effective_static_lambda = static_lambda * (current_epoch / da_warmup_epochs)
        effective_dyn_lambda = dyn_lambda * (current_epoch / da_warmup_epochs)
    else:
        effective_lambda = coral_lambda
        effective_static_lambda = static_lambda
        effective_dyn_lambda = dyn_lambda

    for batch_static, batch_dyn, batch_event, batch_time in loader_src:
        batch_static = batch_static.to(device)
        batch_dyn = batch_dyn.to(device)
        batch_event = batch_event.to(device)
        batch_time = batch_time.to(device)

        valid_mask = (
            (batch_time > 0) & (~torch.isnan(batch_event)) & (~torch.isnan(batch_time))
        )
        if valid_mask.sum() == 0:
            continue

        batch_static = batch_static[valid_mask]
        batch_dyn = batch_dyn[valid_mask]
        batch_event = batch_event[valid_mask]
        batch_time = batch_time[valid_mask]

        # Skip batches with <= 1 sample (BatchNorm requires batch_size > 1 during training)
        if batch_static.shape[0] <= 1:
            continue

        optimizer.zero_grad()

        if multi_level_da and effective_lambda > 0:
            log_hazard, emb_fuse, emb_static, emb_dyn = model(
                batch_static, batch_dyn, return_all_embs=True
            )
        else:
            log_hazard, emb_fuse = model(batch_static, batch_dyn)

        if use_weighted_cox:
            surv_loss = cox_loss_fn(
                log_hazard.squeeze(), batch_event, batch_time, event_weight
            )
        else:
            surv_loss = cox_loss_fn(log_hazard.squeeze(), batch_event, batch_time)

        loss = surv_loss

        if effective_lambda > 0 and iter_tgt:
            try:
                tgt_static, tgt_dyn, _, _ = next(iter_tgt)
            except StopIteration:
                iter_tgt = iter(loader_tgt)
                tgt_static, tgt_dyn, _, _ = next(iter_tgt)

            tgt_static = tgt_static.to(device)
            tgt_dyn = tgt_dyn.to(device)

            if multi_level_da:
                _, emb_tgt_fuse, emb_tgt_static, emb_tgt_dyn = model(
                    tgt_static, tgt_dyn, return_all_embs=True
                )

                # Multi-level MMD: align static, dynamic, and fused embeddings
                if da_method == "mmd":
                    c_loss = mmd_loss(emb_fuse, emb_tgt_fuse)
                    if effective_static_lambda > 0:
                        static_loss = mmd_loss(emb_static, emb_tgt_static)
                        loss = loss + effective_static_lambda * static_loss
                        total_static_da_loss += static_loss.item()
                    if effective_dyn_lambda > 0:
                        dyn_loss = mmd_loss(emb_dyn, emb_tgt_dyn)
                        loss = loss + effective_dyn_lambda * dyn_loss
                        total_dyn_da_loss += dyn_loss.item()
                else:
                    c_loss = coral_loss(emb_fuse, emb_tgt_fuse)
            else:
                _, emb_tgt = model(tgt_static, tgt_dyn)
                if da_method == "mmd":
                    c_loss = mmd_loss(emb_fuse, emb_tgt)
                else:
                    c_loss = coral_loss(emb_fuse, emb_tgt)

            loss = loss + effective_lambda * c_loss
            total_coral_loss += c_loss.item()

        loss.backward()

        # Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_surv_loss += surv_loss.item()
        n_batches += 1

    avg_surv = total_surv_loss / max(n_batches, 1)
    avg_coral = total_coral_loss / max(n_batches, 1)
    avg_static_da = total_static_da_loss / max(n_batches, 1)
    avg_dyn_da = total_dyn_da_loss / max(n_batches, 1)
    return avg_surv, avg_coral, avg_static_da, avg_dyn_da


def train_epoch_dann(
    model,
    domain_classifier,
    loader_src,
    loader_tgt,
    optimizer_survival,
    optimizer_domain,
    device,
    dann_lambda=1.0,
    use_weighted_cox=False,
    event_weight=2.0,
    scheduler=None,
    current_epoch=0,
    da_warmup_epochs=5,
    max_grad_norm=1.0,
):
    """
    DANN training epoch with gradient reversal.

    Two optimizers:
    1. optimizer_survival: updates feature extractor + survival head
    2. optimizer_domain: updates domain classifier

    Training strategy:
    1. Forward pass through feature extractor
    2. Compute survival loss on source domain
    3. Apply gradient reversal to features
    4. Compute domain classification loss
    5. Backpropagate both losses
    """
    model.train()
    domain_classifier.train()

    total_surv_loss = 0.0
    total_domain_loss = 0.0
    n_batches = 0

    iter_tgt = iter(loader_tgt) if loader_tgt else None

    cox_loss_fn = (
        weighted_cox_partial_log_likelihood
        if use_weighted_cox
        else cox_partial_log_likelihood
    )

    # Gradual lambda increase: 0 -> dann_lambda over warmup, then keep constant
    if current_epoch < da_warmup_epochs:
        effective_lambda = dann_lambda * (current_epoch / da_warmup_epochs)
    else:
        effective_lambda = dann_lambda

    for batch_static, batch_dyn, batch_event, batch_time in loader_src:
        batch_static = batch_static.to(device)
        batch_dyn = batch_dyn.to(device)
        batch_event = batch_event.to(device)
        batch_time = batch_time.to(device)

        valid_mask = (
            (batch_time > 0) & (~torch.isnan(batch_event)) & (~torch.isnan(batch_time))
        )
        if valid_mask.sum() == 0:
            continue

        batch_static = batch_static[valid_mask]
        batch_dyn = batch_dyn[valid_mask]
        batch_event = batch_event[valid_mask]
        batch_time = batch_time[valid_mask]

        if batch_static.shape[0] <= 1:
            continue

        optimizer_survival.zero_grad()
        optimizer_domain.zero_grad()

        # Get fused embeddings from model (with return_all_embs=True)
        log_hazard, emb_fuse, emb_static, emb_dyn = model(batch_static, batch_dyn, return_all_embs=True)

        # Survival loss on source domain
        if use_weighted_cox:
            surv_loss = cox_loss_fn(
                log_hazard.squeeze(), batch_event, batch_time, event_weight
            )
        else:
            surv_loss = cox_loss_fn(log_hazard.squeeze(), batch_event, batch_time)

        # Get target domain batch
        try:
            tgt_static, tgt_dyn, _, _ = next(iter_tgt)
        except StopIteration:
            iter_tgt = iter(loader_tgt)
            tgt_static, tgt_dyn, _, _ = next(iter_tgt)

        tgt_static = tgt_static.to(device)
        tgt_dyn = tgt_dyn.to(device)

        # Get target domain embeddings (WITHOUT no_grad - need gradients for alignment)
        _, emb_tgt_fuse, emb_tgt_static, emb_tgt_dyn = model(tgt_static, tgt_dyn, return_all_embs=True)

        # Multi-level domain alignment: fuse + static + dynamic
        # Level 1: Fused embedding
        combined_emb_fuse = torch.cat([emb_fuse, emb_tgt_fuse], dim=0)
        domain_labels_fuse = torch.cat([
            torch.zeros(emb_fuse.size(0), 1, device=device),
            torch.ones(emb_tgt_fuse.size(0), 1, device=device)
        ])

        # Level 2: Static embedding
        combined_emb_static = torch.cat([emb_static, emb_tgt_static], dim=0)
        domain_labels_static = torch.cat([
            torch.zeros(emb_static.size(0), 1, device=device),
            torch.ones(emb_tgt_static.size(0), 1, device=device)
        ])

        # Level 3: Dynamic embedding
        combined_emb_dyn = torch.cat([emb_dyn, emb_tgt_dyn], dim=0)
        domain_labels_dyn = torch.cat([
            torch.zeros(emb_dyn.size(0), 1, device=device),
            torch.ones(emb_tgt_dyn.size(0), 1, device=device)
        ])

        # Apply gradient reversal and compute domain losses
        reversed_emb_fuse = gradient_reversal(combined_emb_fuse, lambda_=effective_lambda)
        domain_pred_fuse = domain_classifier(reversed_emb_fuse)
        domain_loss_fuse = nn.BCEWithLogitsLoss()(domain_pred_fuse, domain_labels_fuse)

        reversed_emb_static = gradient_reversal(combined_emb_static, lambda_=effective_lambda)
        domain_pred_static = domain_classifier(reversed_emb_static)
        domain_loss_static = nn.BCEWithLogitsLoss()(domain_pred_static, domain_labels_static)

        reversed_emb_dyn = gradient_reversal(combined_emb_dyn, lambda_=effective_lambda)
        domain_pred_dyn = domain_classifier(reversed_emb_dyn)
        domain_loss_dyn = nn.BCEWithLogitsLoss()(domain_pred_dyn, domain_labels_dyn)

        # Weighted combination of domain losses
        domain_loss = 0.5 * domain_loss_fuse + 0.25 * domain_loss_static + 0.25 * domain_loss_dyn

        # Total loss
        loss = surv_loss + effective_lambda * domain_loss

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        torch.nn.utils.clip_grad_norm_(domain_classifier.parameters(), max_grad_norm)

        optimizer_survival.step()
        optimizer_domain.step()

        if scheduler is not None:
            scheduler.step()

        total_surv_loss += surv_loss.item()
        total_domain_loss += domain_loss.item()
        n_batches += 1

    avg_surv = total_surv_loss / max(n_batches, 1)
    avg_domain = total_domain_loss / max(n_batches, 1)
    return avg_surv, avg_domain


def predict_risk(model, data_dict, device):
    model.eval()
    event = data_dict["targets"]["event"].astype(bool)
    duration = data_dict["targets"]["duration"].astype(float)

    valid_mask = (duration > 0) | (~event)
    if valid_mask.sum() == 0:
        return np.array([]), event, duration

    dataset = TensorDataset(
        torch.tensor(data_dict["static"][valid_mask], dtype=torch.float32),
        torch.tensor(data_dict["dynamic"][valid_mask], dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    risks = []
    with torch.no_grad():
        for batch_static, batch_dyn in loader:
            batch_static = batch_static.to(device)
            batch_dyn = batch_dyn.to(device)
            log_hazard, _ = model(batch_static, batch_dyn)
            risks.append(log_hazard.cpu().numpy())

    risk_scores = np.concatenate(risks, axis=0).squeeze()
    return risk_scores, event[valid_mask], duration[valid_mask]


def evaluate_survival_metrics(model, data_dict, device):
    risk_scores, event, duration = predict_risk(model, data_dict, device)

    if len(risk_scores) == 0:
        return {"c_index": np.nan, "ibs": np.nan}

    try:
        c_index, _, _, _, _ = concordance_index_censored(event, duration, risk_scores)
    except Exception as e:
        log.warning(f"C-index computation failed: {e}")
        c_index = np.nan

    try:
        event_mask = event.astype(bool)
        if event_mask.sum() > 0:
            times_grid = np.percentile(duration[event_mask], np.linspace(10, 90, 5))
            times_grid = times_grid[times_grid > 0]
            if len(times_grid) > 0:
                ibs = integrated_brier_score(event, duration, risk_scores, times_grid)
            else:
                ibs = np.nan
        else:
            ibs = np.nan
    except Exception as e:
        log.warning(f"IBS computation failed: {e}")
        ibs = np.nan

    return {"c_index": c_index, "ibs": ibs}


def run_dadsn_experiment(
    source_data,
    target_data,
    save_dir,
    config,
    use_coral=True,
    use_transformer=True,
    seeds=[42],
    da_method="coral",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    n_static = source_data["static"].shape[1]
    n_dynamic = source_data["dynamic"].shape[2]

    results_all_seeds = []

    # Training strategy upgrades
    use_weighted_cox = getattr(config.training, "use_weighted_cox", False)
    event_weight = getattr(config.training, "event_weight", 2.0)
    warmup_epochs = getattr(config.training, "warmup_epochs", 5)
    patience = getattr(config.training, "early_stopping_patience", 10)

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        log.info(f"\n{'='*60}")
        log.info(
            f"Running experiment with seed={seed}, coral={use_coral}, transformer={use_transformer}, da_method={da_method}"
        )
        log.info(f"{'='*60}")

        model = DADSN(config, n_static, n_dynamic, use_transformer=use_transformer).to(
            device
        )

        optimizer = optim.Adam(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

        # Cosine Annealing + Warmup scheduler
        scheduler = WarmupCosineAnnealing(
            optimizer,
            warmup_epochs,
            config.training.epochs,
            config.training.learning_rate,
        )

        if use_coral and target_data is not None:
            loader_src, loader_tgt = create_mixed_dataloader(
                source_data, target_data, config.training.batch_size
            )
        else:
            loader_src = create_source_only_dataloader(
                source_data, config.training.batch_size
            )
            loader_tgt = None

        # DANN specific: create domain classifier and optimizer
        domain_classifier = None
        optimizer_domain = None
        if da_method == "dann":
            embed_dim = config.model.dynamic_encoder.d_model
            domain_classifier = DomainClassifier(input_dim=embed_dim, hidden_dim=128).to(device)
            optimizer_domain = optim.Adam(
                domain_classifier.parameters(),
                lr=config.training.learning_rate * 0.5,
                weight_decay=config.training.weight_decay,
            )

        best_c_index = -1
        best_state = None
        best_domain_state = None
        epochs_no_improve = 0

        for epoch in range(config.training.epochs):
            if da_method == "dann" and domain_classifier is not None:
                train_result = train_epoch_dann(
                    model,
                    domain_classifier,
                    loader_src,
                    loader_tgt,
                    optimizer,
                    optimizer_domain,
                    device,
                    dann_lambda=1.0,
                    use_weighted_cox=use_weighted_cox,
                    event_weight=event_weight,
                    scheduler=scheduler,
                    current_epoch=epoch,
                    da_warmup_epochs=5,
                    max_grad_norm=1.0,
                )
                surv_loss, domain_l = train_result
                coral_l = domain_l  # For logging compatibility
            else:
                train_result = train_epoch_dadsn(
                    model,
                    loader_src,
                    loader_tgt,
                    optimizer,
                    device,
                    coral_lambda=config.training.coral.lambda_coral if use_coral else 0.0,
                    use_weighted_cox=use_weighted_cox,
                    event_weight=event_weight,
                    scheduler=scheduler,
                    da_method=da_method if use_coral else "coral",
                    current_epoch=epoch,
                    da_warmup_epochs=15,
                    max_grad_norm=1.0,
                    multi_level_da=False,
                    static_lambda=0.0,
                    dyn_lambda=0.0,
                )
                surv_loss, coral_l, static_da_l, dyn_da_l = train_result

            # Evaluate every epoch for early stopping
            metrics = evaluate_survival_metrics(model, target_data, device)
            current_lr = scheduler.current_epoch

            if (epoch + 1) % 5 == 0 or epoch == 0:
                log.info(
                    f"Epoch {epoch+1}/{config.training.epochs} | "
                    f"SurvLoss: {surv_loss:.4f} | "
                    f"{'DomainLoss' if da_method == 'dann' else 'CoralLoss'}: {coral_l:.4f} | "
                    f"Target C-index: {metrics['c_index']:.4f} | "
                    f"Target IBS: {metrics['ibs']:.4f} | "
                    f"LR: {scheduler.optimizer.param_groups[0]['lr']:.6f}"
                )

            # Early stopping based on C-index
            if metrics["c_index"] > best_c_index:
                best_c_index = metrics["c_index"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                if domain_classifier is not None:
                    best_domain_state = {k: v.cpu().clone() for k, v in domain_classifier.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    log.info(f"Early stopping at epoch {epoch+1}")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        if best_domain_state is not None and domain_classifier is not None:
            domain_classifier.load_state_dict(best_domain_state)

        final_metrics = evaluate_survival_metrics(model, target_data, device)
        source_metrics = evaluate_survival_metrics(model, source_data, device)

        log.info(f"\n--- Seed {seed} Final Results ---")
        log.info(
            f"Source (深医) C-index: {source_metrics['c_index']:.4f}, IBS: {source_metrics['ibs']:.4f}"
        )
        log.info(
            f"Target (福鼎) C-index: {final_metrics['c_index']:.4f}, IBS: {final_metrics['ibs']:.4f}"
        )

        results_all_seeds.append(
            {
                "seed": seed,
                "source_c_index": source_metrics["c_index"],
                "source_ibs": source_metrics["ibs"],
                "target_c_index": final_metrics["c_index"],
                "target_ibs": final_metrics["ibs"],
            }
        )

        # Save each seed's best model
        seed_path = os.path.join(save_dir, f"seed_{seed}.pt")
        os.makedirs(save_dir, exist_ok=True)
        torch.save(
            {
                "model_state_dict": best_state if best_state is not None else model.state_dict(),
                "domain_classifier_state_dict": best_domain_state if best_domain_state is not None else (domain_classifier.state_dict() if domain_classifier is not None else None),
                "config": config,
                "seed": seed,
                "da_method": da_method,
            },
            seed_path,
        )

    results_mean = {
        k: np.mean([r[k] for r in results_all_seeds])
        for k in results_all_seeds[0].keys()
    }
    results_std = {
        k + "_std": np.std([r[k] for r in results_all_seeds])
        for k in ["source_c_index", "source_ibs", "target_c_index", "target_ibs"]
    }

    log.info(f"\n{'='*60}")
    log.info("AVERAGED RESULTS ACROSS SEEDS")
    log.info(f"{'='*60}")
    log.info(
        f"Source C-index: {results_mean['source_c_index']:.4f} ± {results_std['source_c_index_std']:.4f}"
    )
    log.info(
        f"Source IBS: {results_mean['source_ibs']:.4f} ± {results_std['source_ibs_std']:.4f}"
    )
    log.info(
        f"Target C-index: {results_mean['target_c_index']:.4f} ± {results_std['target_c_index_std']:.4f}"
    )
    log.info(
        f"Target IBS: {results_mean['target_ibs']:.4f} ± {results_std['target_ibs_std']:.4f}"
    )

    # Ensemble evaluation: average risk scores from all seeds
    seed_models = [os.path.join(save_dir, f"seed_{seed}.pt") for seed in seeds]
    if len(seeds) > 1:
        ensemble_target = evaluate_ensemble(
            seed_models, target_data, device, config, n_static, n_dynamic, use_transformer
        )
        ensemble_source = evaluate_ensemble(
            seed_models, source_data, device, config, n_static, n_dynamic, use_transformer
        )
        log.info(f"\n--- Ensemble Results ---")
        log.info(
            f"Ensemble Target C-index: {ensemble_target['c_index']:.4f}"
        )
        log.info(
            f"Ensemble Source C-index: {ensemble_source['c_index']:.4f}"
        )
        results_mean["ensemble_target_c_index"] = ensemble_target["c_index"]
        results_mean["ensemble_source_c_index"] = ensemble_source["c_index"]

    torch.save(
        {
            "config": config,
            "results": results_all_seeds,
            "results_mean": results_mean,
            "results_std": results_std,
        },
        os.path.join(save_dir, "dadsn_final.pt"),
    )

    return results_mean, results_std


def evaluate_ensemble(model_paths, data_dict, device, config, n_static, n_dynamic, use_transformer):
    """Evaluate ensemble of models by averaging risk scores."""
    all_risks = []

    for model_path in model_paths:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model = DADSN(config, n_static, n_dynamic, use_transformer=use_transformer).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        with torch.no_grad():
            static = torch.tensor(data_dict["static"], dtype=torch.float32).to(device)
            dynamic = torch.tensor(data_dict["dynamic"], dtype=torch.float32).to(device)

            risks = []
            batch_size = 512
            for i in range(0, len(static), batch_size):
                batch_s = static[i:i+batch_size]
                batch_d = dynamic[i:i+batch_size]
                log_h, _ = model(batch_s, batch_d)
                risks.append(log_h.cpu().numpy())

            all_risks.append(np.concatenate(risks, axis=0).squeeze())

    # Average risk scores
    ensemble_risk = np.mean(all_risks, axis=0)

    event = data_dict["targets"]["event"].astype(bool)
    duration = data_dict["targets"]["duration"].astype(float)

    c_index, _, _, _, _ = concordance_index_censored(event, duration, ensemble_risk)

    return {"c_index": c_index}


def run_ablation_study(source_data, target_data, save_dir, config, seeds=[42]):
    log.info("\n" + "=" * 80)
    log.info("ABLATION STUDY")
    log.info("=" * 80)

    results = {}

    log.info("\n### Experiment 1: Base (Static Only + Cox) ###")
    results["base"] = run_dadsn_experiment(
        source_data,
        target_data,
        os.path.join(save_dir, "base"),
        config,
        use_coral=False,
        use_transformer=False,
        seeds=seeds,
    )

    log.info("\n### Experiment 2: Base + ResNet1D ###")
    results["base_transformer"] = run_dadsn_experiment(
        source_data,
        target_data,
        os.path.join(save_dir, "base_transformer"),
        config,
        use_coral=False,
        use_transformer=True,
        seeds=seeds,
    )

    log.info("\n### Experiment 3: DA-DSN (MMD Domain Adaptation) ###")
    results["dadsn_mmd"] = run_dadsn_experiment(
        source_data,
        target_data,
        os.path.join(save_dir, "dadsn_mmd"),
        config,
        use_coral=True,
        use_transformer=True,
        seeds=seeds,
        da_method="mmd",
    )

    log.info("\n### Experiment 4: DA-DSN (DANN Adversarial Training - Fixed) ###")
    results["dadsn_dann_fixed"] = run_dadsn_experiment(
        source_data,
        target_data,
        os.path.join(save_dir, "dadsn_dann_fixed"),
        config,
        use_coral=True,
        use_transformer=True,
        seeds=seeds,
        da_method="dann",
    )

    log.info("\n" + "=" * 80)
    log.info("ABLATION SUMMARY TABLE")
    log.info("=" * 80)
    log.info(f"{'Experiment':<35} {'Target C-index':<20} {'Target IBS':<15}")
    log.info("-" * 70)
    for name, (mean, std) in results.items():
        log.info(
            f"{name:<35} {mean['target_c_index']:.4f} ± {std['target_c_index_std']:.4f}   {mean['target_ibs']:.4f} ± {std['target_ibs_std']:.4f}"
        )

    return results
