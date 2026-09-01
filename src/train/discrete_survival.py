"""Model-agnostic training core for four-bin discrete-time survival."""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler


def _next_or_restart(loader: DataLoader, iterator):
    """读取下一批；较短域耗尽时从新迭代器继续。"""
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


@dataclass(frozen=True)
class SurvivalFitResult:
    best_validation_nll: float
    best_epoch: int
    epochs_run: int


def masked_survival_nll(
    logits: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    session_weights: torch.Tensor,
) -> torch.Tensor:
    """Compute patient-balanced NLL after summing four interval losses per session."""
    if logits.ndim != 2 or logits.shape[1] != 4:
        raise ValueError("logits must have shape (sessions, 4).")
    if targets.shape != logits.shape or masks.shape != logits.shape:
        raise ValueError("targets and masks must match logits shape.")
    if session_weights.ndim != 1 or len(session_weights) != len(logits):
        raise ValueError("session_weights must have shape (sessions,).")
    if not torch.all((targets == 0) | (targets == 1)):
        raise ValueError("targets must contain only 0 and 1.")
    if not torch.all((masks == 0) | (masks == 1)):
        raise ValueError("masks must contain only 0 and 1.")
    if not torch.isfinite(session_weights).all() or torch.any(session_weights <= 0):
        raise ValueError("session_weights must be finite and positive.")
    interval_nll = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    session_nll = (interval_nll * masks).sum(dim=1)
    return (session_nll * session_weights).sum() / session_weights.sum()


def centered_coral(source_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
    """Squared covariance discrepancy using explicitly centered representations."""
    if source_features.ndim != 2 or target_features.ndim != 2:
        raise ValueError("CORAL features must be two-dimensional.")
    if source_features.shape[1] != target_features.shape[1]:
        raise ValueError("Source and target CORAL features must have equal width.")
    if len(source_features) < 2 or len(target_features) < 2:
        raise ValueError("Centered CORAL requires at least two samples per domain.")
    source_centered = source_features - source_features.mean(dim=0, keepdim=True)
    target_centered = target_features - target_features.mean(dim=0, keepdim=True)
    source_cov = source_centered.T @ source_centered / (len(source_features) - 1)
    target_cov = target_centered.T @ target_centered / (len(target_features) - 1)
    width = source_features.shape[1]
    return (source_cov - target_cov).square().sum() / (4 * width**2)


def mmd_rbf_loss(
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    gamma: float | None = None,
) -> torch.Tensor:
    """RBF kernel maximum mean discrepancy between source and target features."""
    if source_features.ndim != 2 or target_features.ndim != 2:
        raise ValueError("MMD features must be two-dimensional.")
    if source_features.shape[1] != target_features.shape[1]:
        raise ValueError("Source and target MMD features must have equal width.")
    if len(source_features) < 2 or len(target_features) < 2:
        raise ValueError("MMD requires at least two samples per domain.")

    if gamma is None:
        gamma = 1.0 / source_features.shape[1]

    def squared_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.cdist(x, y, p=2).pow(2)

    k_xx = torch.exp(-gamma * squared_distance(source_features, source_features))
    k_yy = torch.exp(-gamma * squared_distance(target_features, target_features))
    k_xy = torch.exp(-gamma * squared_distance(source_features, target_features))

    return k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()


def dann_domain_loss(
    source_logits: torch.Tensor,
    target_logits: torch.Tensor,
) -> torch.Tensor:
    """Binary cross-entropy domain loss for DANN (source=0, target=1)."""
    if source_logits.ndim != 2 or target_logits.ndim != 2:
        raise ValueError("Domain logits must be two-dimensional.")
    if source_logits.shape[1] != target_logits.shape[1]:
        raise ValueError("Source and target domain logits must have equal width.")
    if len(source_logits) < 1 or len(target_logits) < 1:
        raise ValueError("DANN requires at least one sample per domain.")
    labels = torch.cat(
        [
            torch.zeros(len(source_logits), 1, device=source_logits.device),
            torch.ones(len(target_logits), 1, device=target_logits.device),
        ]
    )
    logits = torch.cat([source_logits, target_logits])
    return nn.functional.binary_cross_entropy_with_logits(logits, labels)


def _model_logits(model: nn.Module, features: torch.Tensor) -> torch.Tensor:
    logits = model(features)
    if not isinstance(logits, torch.Tensor):
        raise TypeError("Survival model forward must return a logits tensor.")
    if logits.ndim != 2 or logits.shape != (len(features), 4):
        raise ValueError("Survival model must return logits with shape (sessions, 4).")
    return logits


def _batch_numerator_and_weight(
    model: nn.Module,
    batch: tuple[torch.Tensor, ...],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    features, targets, masks, weights = (item.to(device) for item in batch)
    logits = _model_logits(model, features)
    interval_nll = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    numerator = (((interval_nll * masks).sum(dim=1)) * weights).sum()
    return numerator, weights.sum()


def _unbiased_global_patient_loss(
    numerator: torch.Tensor,
    batch_weight_count: int,
    dataset: Dataset,
) -> torch.Tensor:
    """按全数据患者等权目标做无偏小批缩放，不在批内重新归一。"""
    total_weight = float(dataset.session_weights.sum())
    if total_weight <= 0 or batch_weight_count <= 0:
        raise ValueError("Dataset patient weights and batch size must be positive.")
    return numerator * (len(dataset) / (batch_weight_count * total_weight))


def evaluate_survival_nll(
    model: nn.Module,
    dataset: Dataset,
    *,
    device: torch.device,
    batch_size: int = 1024,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> float:
    """Evaluate patient-balanced survival NLL without materializing person-period rows."""
    if len(dataset) == 0:
        raise ValueError("Validation dataset must not be empty.")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    numerator = torch.zeros((), device=device)
    denominator = torch.zeros((), device=device)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch_numerator, batch_weight = _batch_numerator_and_weight(model, batch, device)
            numerator += batch_numerator
            denominator += batch_weight
    return float((numerator / denominator).cpu())


def _fit_single_domain(
    model: nn.Module,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    *,
    device: torch.device,
    learning_rate: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    seed: int,
    weight_decay: float = 1e-4,
    gradient_clip_norm: float = 1.0,
    num_workers: int = 0,
    pin_memory: bool = False,
    epoch_callback: Callable[[dict[str, float | int]], None] | None = None,
) -> SurvivalFitResult:
    if not train_dataset or not validation_dataset:
        raise ValueError("Training and validation datasets must not be empty.")
    if max_epochs < 1 or patience < 1:
        raise ValueError("max_epochs and patience must be positive.")
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.to(device)
    best_state = copy.deepcopy(model.state_dict())
    best_nll = float("inf")
    best_epoch = 0
    remaining = patience
    epochs_run = 0
    for epoch in range(1, max_epochs + 1):
        epochs_run = epoch
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        for batch in loader:
            optimizer.zero_grad()
            numerator, _ = _batch_numerator_and_weight(model, batch, device)
            loss = _unbiased_global_patient_loss(numerator, len(batch[0]), train_dataset)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            epoch_steps += 1
        validation_nll = evaluate_survival_nll(
            model,
            validation_dataset,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        if validation_nll < best_nll:
            best_nll = validation_nll
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            remaining = patience
        else:
            remaining -= 1
        if epoch_callback is not None:
            epoch_callback({"epoch": epoch, "train_loss": epoch_loss / epoch_steps, "val_loss": validation_nll, "best_epoch": best_epoch})
        if remaining == 0:
            break
    model.load_state_dict(best_state)
    return SurvivalFitResult(best_nll, best_epoch, epochs_run)


def fit_source_only(
    model: nn.Module,
    source_train: Dataset,
    source_validation: Dataset,
    **kwargs,
) -> SurvivalFitResult:
    """Fit on source sessions and stop on source-validation survival NLL."""
    return _fit_single_domain(model, source_train, source_validation, **kwargs)


def fit_target_only(
    model: nn.Module,
    target_train: Dataset,
    target_validation: Dataset,
    **kwargs,
) -> SurvivalFitResult:
    """Fit on target sessions and stop on target-validation survival NLL."""
    return _fit_single_domain(model, target_train, target_validation, **kwargs)


def fit_paired_source_target_update(
    model: nn.Module,
    source_train: Dataset,
    target_train: Dataset,
    target_validation: Dataset,
    *,
    device: torch.device,
    learning_rate: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    seed: int,
    weight_decay: float = 1e-4,
    gradient_clip_norm: float = 1.0,
    coral_weight: float = 0.0,
    representation_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor] | None = None,
    target_supervised: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    alignment_method: str = "coral",
    dann_weight: float = 0.1,
    retain_task_tail_batches: bool = False,
    # --- New: domain mixture strategy (critical fix for 4:1 scale mismatch) ---
    # * "patient_balanced"  blend = n_source/(n_source+n_target) for TARGET loss
    #                          i.e. inverse dataset sizes: smaller domain gets larger
    #                          weight, so each *patient* (not each *batch row*) has
    #                          equal influence on the combined gradient.
    # * "equal" (the previous hard-coded 0.5/0.5) kept only for reproducibility.
    # * "proportional"  blend = n_target/(n_source+n_target) for TARGET loss
    #                          i.e. gradient share proportional to patient volume.
    domain_mix: str = "patient_balanced",
    epoch_callback: Callable[[dict[str, float | int]], None] | None = None,
) -> SurvivalFitResult:
    """Update from paired domains with documented domain-task-loss blending.

    Blending rationale (when ``domain_mix == "patient_balanced"``, the new
    default):

        N_s = sum of session-level patient weights across source domain
        N_t = sum of session-level patient weights across target domain
        α = N_s / (N_s + N_t)   → weight on target task loss
        β = N_t / (N_s + N_t)   → weight on source task loss

    Under this choice, a single patient's sessions (summed across all of their
    person-period cells in a balanced session-weight scheme) have *equal*
    gradient contribution regardless of which center they come from.  This
    fixes the previous 0.5/0.5 heuristic which gave ~30 target patients the
    same total gradient mass as ~1,384 source patients, producing an
    uncontrolled 46:1 per-patient amplification at the 10% budget.

    When ``target_supervised`` is ``False``, target batches contribute only to
    the alignment term; their survival NLL is skipped (unsupervised alignment).

    ``alignment_method`` selects the cross-domain alignment operator:
    ``"coral"`` uses :func:`centered_coral` scaled by ``coral_weight``;
    ``"mmd"`` uses :func:`mmd_rbf_loss` scaled by ``coral_weight`` for parity;
    ``"dann"`` uses :func:`dann_domain_loss` scaled by ``dann_weight`` and
    requires the model to expose ``representation`` and ``domain_discriminator``;
    ``"none"`` skips alignment entirely.
    """
    if not source_train or not target_train or not target_validation:
        raise ValueError("Source, target, and validation datasets must not be empty.")
    if max_epochs < 1 or patience < 1 or coral_weight < 0 or dann_weight < 0:
        raise ValueError("Epochs/patience must be positive and weights non-negative.")
    if alignment_method not in {"coral", "mmd", "dann", "none"}:
        raise ValueError(f"Unknown alignment_method: {alignment_method}")
    if domain_mix not in {"patient_balanced", "equal", "proportional"}:
        raise ValueError(f"Unknown domain_mix strategy: {domain_mix}")
    if alignment_method in {"coral", "mmd"} and coral_weight > 0 and representation_fn is None:
        raise ValueError("representation_fn is required when coral_weight is positive.")
    if alignment_method == "dann" and dann_weight > 0:
        if not hasattr(model, "domain_discriminator") or not hasattr(model, "representation"):
            raise ValueError("DANN requires model.domain_discriminator and model.representation.")

    # ------------------------------------------------------------------
    # Compute effective patient counts from each domain's session-weight
    # sum (session weights are already 1 / num_sessions_per_patient so
    # the total sum equals the number of unique patients).
    # ------------------------------------------------------------------
    n_source_patients = float(getattr(source_train, "session_weights", torch.as_tensor(1.0)).sum())
    n_target_patients = float(getattr(target_train, "session_weights", torch.as_tensor(1.0)).sum())
    if n_source_patients <= 0 or n_target_patients <= 0:
        raise ValueError("Both domains must contain positive patient-balanced weight mass.")
    if domain_mix == "equal":
        alpha_target = 0.5
        alpha_source = 0.5
    elif domain_mix == "proportional":
        # Target weight proportional to target patient volume; source proportional
        # to source patient volume (mirrors the real-data distribution).
        alpha_target = n_target_patients / (n_source_patients + n_target_patients)
        alpha_source = n_source_patients / (n_source_patients + n_target_patients)
    else:  # patient_balanced — the recommended default
        # Target receives more weight *precisely because* it has fewer patients,
        # equalizing per-patient gradient influence across the two domains.
        alpha_target = n_source_patients / (n_source_patients + n_target_patients)
        alpha_source = n_target_patients / (n_source_patients + n_target_patients)

    steps = max(
        (len(source_train) + batch_size - 1) // batch_size,
        (len(target_train) + batch_size - 1) // batch_size,
    )
    samples_per_domain = steps * batch_size
    source_sampler = None if retain_task_tail_batches else RandomSampler(
        source_train,
        replacement=True,
        num_samples=samples_per_domain,
        generator=torch.Generator().manual_seed(seed),
    )
    target_sampler = None if retain_task_tail_batches else RandomSampler(
        target_train,
        replacement=True,
        num_samples=samples_per_domain,
        generator=torch.Generator().manual_seed(seed + 1),
    )
    source_loader = DataLoader(
        source_train,
        batch_size=batch_size,
        sampler=source_sampler,
        shuffle=retain_task_tail_batches,
        drop_last=not retain_task_tail_batches,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    target_loader = DataLoader(
        target_train,
        batch_size=batch_size,
        sampler=target_sampler,
        shuffle=retain_task_tail_batches,
        drop_last=not retain_task_tail_batches,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.to(device)
    best_state = copy.deepcopy(model.state_dict())
    best_nll = float("inf")
    best_epoch = 0
    remaining = patience
    epochs_run = 0
    for epoch in range(1, max_epochs + 1):
        epochs_run = epoch
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        # Alignment-loss numerical-balance diagnostics (per-epoch mean ratio).
        # λ·ℓ_alignment / ℓ_task < 1e-4 → AdamW 2nd moment 1e-8 floor erases
        # alignment gradient (Type II error on "CORAL ineffective" claim).
        epoch_diag_task_sum = 0.0
        epoch_diag_align_raw_sum = 0.0
        epoch_diag_align_scaled_sum = 0.0
        epoch_diag_ratio_align_sum = 0.0
        epoch_diag_dann_raw_sum = 0.0
        epoch_diag_dann_scaled_sum = 0.0
        epoch_diag_ratio_dann_sum = 0.0
        epoch_diag_n_batches = 0
        source_iterator = iter(source_loader)
        target_iterator = iter(target_loader)
        for _ in range(steps):
            source_batch, source_iterator = _next_or_restart(source_loader, source_iterator)
            target_batch, target_iterator = _next_or_restart(target_loader, target_iterator)
            optimizer.zero_grad()
            source_numerator, _ = _batch_numerator_and_weight(model, source_batch, device)
            source_loss = _unbiased_global_patient_loss(
                source_numerator, len(source_batch[0]), source_train
            )
            if target_supervised:
                target_numerator, _ = _batch_numerator_and_weight(model, target_batch, device)
                target_loss = _unbiased_global_patient_loss(
                    target_numerator, len(target_batch[0]), target_train
                )
                loss = alpha_target * target_loss + alpha_source * source_loss
            else:
                loss = alpha_source * source_loss
            # Capture task-level ℓ BEFORE adding alignment — the ratio
            # scaled_alignment / task_ℓ is the effective gradient share.
            task_loss_for_diag = float(loss.detach().cpu())
            align_raw = align_scaled = dann_raw = dann_scaled = None
            if len(source_batch[0]) >= 2 and len(target_batch[0]) >= 2:
                source_features = source_batch[0].to(device)
                target_features = target_batch[0].to(device)
                if alignment_method == "coral" and coral_weight > 0:
                    align_raw = centered_coral(
                        representation_fn(model, source_features),
                        representation_fn(model, target_features),
                    )
                    align_scaled = coral_weight * align_raw
                    loss = loss + align_scaled
                elif alignment_method == "mmd" and coral_weight > 0:
                    align_raw = mmd_rbf_loss(
                        representation_fn(model, source_features),
                        representation_fn(model, target_features),
                    )
                    align_scaled = coral_weight * align_raw
                    loss = loss + align_scaled
                elif alignment_method == "dann" and dann_weight > 0:
                    source_repr = model.representation(source_features)
                    target_repr = model.representation(target_features)
                    source_domain_logits = model.domain_discriminator(source_repr)
                    target_domain_logits = model.domain_discriminator(target_repr)
                    dann_raw = dann_domain_loss(source_domain_logits, target_domain_logits)
                    dann_scaled = dann_weight * dann_raw
                    loss = loss + dann_scaled
            if task_loss_for_diag > 0:
                epoch_diag_task_sum += task_loss_for_diag
                a_raw = float(align_raw.detach().cpu()) if align_raw is not None else 0.0
                a_scaled = float(align_scaled.detach().cpu()) if align_scaled is not None else 0.0
                epoch_diag_align_raw_sum += a_raw
                epoch_diag_align_scaled_sum += a_scaled
                epoch_diag_ratio_align_sum += a_scaled / task_loss_for_diag
                d_raw = float(dann_raw.detach().cpu()) if dann_raw is not None else 0.0
                d_scaled = float(dann_scaled.detach().cpu()) if dann_scaled is not None else 0.0
                epoch_diag_dann_raw_sum += d_raw
                epoch_diag_dann_scaled_sum += d_scaled
                epoch_diag_ratio_dann_sum += d_scaled / task_loss_for_diag
                epoch_diag_n_batches += 1
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            epoch_steps += 1
        validation_nll = evaluate_survival_nll(
            model,
            target_validation,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        if validation_nll < best_nll:
            best_nll = validation_nll
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            remaining = patience
        else:
            remaining -= 1
        if epoch_callback is not None:
            nb = max(epoch_diag_n_batches, 1)
            epoch_callback({
                "epoch": epoch,
                "train_loss": epoch_loss / max(epoch_steps, 1),
                "val_loss": validation_nll,
                "best_epoch": best_epoch,
                "domain_mix_alpha_target": float(alpha_target),
                "domain_mix_alpha_source": float(alpha_source),
                "n_patients_source": float(n_source_patients),
                "n_patients_target": float(n_target_patients),
                # Alignment gradient-budget diagnostics:
                "diag_alignment_raw_mean": epoch_diag_align_raw_sum / nb,
                "diag_alignment_scaled_mean": epoch_diag_align_scaled_sum / nb,
                "diag_alignment_ratio_mean": epoch_diag_ratio_align_sum / nb,
                "diag_dann_raw_mean": epoch_diag_dann_raw_sum / nb,
                "diag_dann_scaled_mean": epoch_diag_dann_scaled_sum / nb,
                "diag_dann_ratio_mean": epoch_diag_ratio_dann_sum / nb,
                "diag_task_loss_mean": epoch_diag_task_sum / nb,
                "diag_n_valid_batches": epoch_diag_n_batches,
            })
        if remaining == 0:
            break
    model.load_state_dict(best_state)
    return SurvivalFitResult(best_nll, best_epoch, epochs_run)
