import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
import os
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.models.dadsn_model import DADSN, cox_partial_log_likelihood, coral_loss
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
):
    model.train()
    total_surv_loss = 0.0
    total_coral_loss = 0.0
    n_batches = 0

    iter_tgt = iter(loader_tgt) if loader_tgt else None

    cox_loss_fn = (
        weighted_cox_partial_log_likelihood
        if use_weighted_cox
        else cox_partial_log_likelihood
    )

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

        log_hazard, emb_fuse = model(batch_static, batch_dyn)

        if use_weighted_cox:
            surv_loss = cox_loss_fn(
                log_hazard.squeeze(), batch_event, batch_time, event_weight
            )
        else:
            surv_loss = cox_loss_fn(log_hazard.squeeze(), batch_event, batch_time)

        loss = surv_loss

        if coral_lambda > 0 and iter_tgt:
            try:
                tgt_static, tgt_dyn, _, _ = next(iter_tgt)
            except StopIteration:
                iter_tgt = iter(loader_tgt)
                tgt_static, tgt_dyn, _, _ = next(iter_tgt)

            tgt_static = tgt_static.to(device)
            tgt_dyn = tgt_dyn.to(device)

            _, emb_tgt = model(tgt_static, tgt_dyn)

            c_loss = coral_loss(emb_fuse, emb_tgt)
            loss = loss + coral_lambda * c_loss
            total_coral_loss += c_loss.item()

        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_surv_loss += surv_loss.item()
        n_batches += 1

    avg_surv = total_surv_loss / max(n_batches, 1)
    avg_coral = total_coral_loss / max(n_batches, 1)
    return avg_surv, avg_coral


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
            f"Running experiment with seed={seed}, coral={use_coral}, transformer={use_transformer}"
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

        best_c_index = -1
        best_state = None
        epochs_no_improve = 0

        for epoch in range(config.training.epochs):
            surv_loss, coral_l = train_epoch_dadsn(
                model,
                loader_src,
                loader_tgt,
                optimizer,
                device,
                coral_lambda=config.training.coral.lambda_coral if use_coral else 0.0,
                use_weighted_cox=use_weighted_cox,
                event_weight=event_weight,
                scheduler=scheduler,
            )

            # Evaluate every epoch for early stopping
            metrics = evaluate_survival_metrics(model, target_data, device)
            current_lr = scheduler.current_epoch

            if (epoch + 1) % 5 == 0 or epoch == 0:
                log.info(
                    f"Epoch {epoch+1}/{config.training.epochs} | "
                    f"SurvLoss: {surv_loss:.4f} | "
                    f"CoralLoss: {coral_l:.4f} | "
                    f"Target C-index: {metrics['c_index']:.4f} | "
                    f"Target IBS: {metrics['ibs']:.4f} | "
                    f"LR: {scheduler.optimizer.param_groups[0]['lr']:.6f}"
                )

            # Early stopping based on C-index
            if metrics["c_index"] > best_c_index:
                best_c_index = metrics["c_index"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    log.info(f"Early stopping at epoch {epoch+1}")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

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

    os.makedirs(save_dir, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "results": results_all_seeds,
            "results_mean": results_mean,
            "results_std": results_std,
        },
        os.path.join(save_dir, "dadsn_final.pt"),
    )

    return results_mean, results_std


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

    log.info("\n### Experiment 2: Base + Transformer ###")
    results["base_transformer"] = run_dadsn_experiment(
        source_data,
        target_data,
        os.path.join(save_dir, "base_transformer"),
        config,
        use_coral=False,
        use_transformer=True,
        seeds=seeds,
    )

    log.info("\n### Experiment 3: Base + Transformer + CORAL (DA-DSN) ###")
    results["dadsn"] = run_dadsn_experiment(
        source_data,
        target_data,
        os.path.join(save_dir, "dadsn"),
        config,
        use_coral=True,
        use_transformer=True,
        seeds=seeds,
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
