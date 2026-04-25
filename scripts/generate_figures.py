"""
Publication-Ready Figures Generator for DA-DSN Paper
- Figure 1: Individualized Survival Curves (3 risk groups)
- Figure 2: UMAP MMD Domain Adaptation Visualization
- Ablation: Base, Base+ResNet1D, DA-DSN (MMD)
- Bootstrap confidence intervals for C-index
- MMD distribution difference analysis
"""

import os
import sys
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import DialysisDataLoader
from src.models.dadsn_model import (
    DADSN,
    cox_partial_log_likelihood,
    coral_loss,
    mmd_loss,
)
from src.training.dadsn_runner import (
    create_source_only_dataloader,
    predict_risk,
)


def estimate_baseline_survival_breslow(events, durations, risk_scores, time_points):
    """
    Standard Breslow estimator for baseline survival function.
    S_0(t) = exp(-H_0(t)) where H_0(t) is cumulative baseline hazard.

    At each unique event time t_j:
    H_0(t_j) = H_0(t_{j-1}) + d_j / sum_{k in R(t_j)} exp(h_k)

    where:
    - d_j = number of events at time t_j
    - R(t_j) = risk set at time t_j (all subjects with duration >= t_j)
    - h_k = log-hazard for subject k
    """
    order = np.argsort(durations)
    sorted_events = events[order]
    sorted_durations = durations[order]
    sorted_risks = np.exp(risk_scores[order])

    cumulative_hazard = np.zeros(len(time_points))

    for i, t in enumerate(time_points):
        at_risk_mask = sorted_durations >= t
        if at_risk_mask.sum() == 0:
            cumulative_hazard[i] = cumulative_hazard[i - 1] if i > 0 else 0
            continue

        event_mask = (sorted_durations == t) & (sorted_events == 1)
        d_j = event_mask.sum()

        if d_j == 0:
            cumulative_hazard[i] = cumulative_hazard[i - 1] if i > 0 else 0
            continue

        risk_sum = sorted_risks[at_risk_mask].sum()
        if risk_sum > 0:
            increment = d_j / risk_sum
        else:
            increment = 0

        cumulative_hazard[i] = (cumulative_hazard[i - 1] if i > 0 else 0) + increment

    return np.exp(-cumulative_hazard)


def compute_individual_survival(log_hazard_scalar, baseline_survival):
    """
    S(t|x) = S_0(t)^exp(h(x))
    """
    risk = np.exp(log_hazard_scalar)
    survival = np.power(np.clip(baseline_survival, 0.001, 0.999), risk)
    return survival


def generate_figure_1(model, data_dict, device, save_path):
    """
    Figure 1: Individualized risk prediction curves.
    Select 3 typical patients (high/medium/low risk) and plot their survival curves.
    """
    risk_scores, events, durations = predict_risk(model, data_dict, device)

    if len(risk_scores) == 0:
        print("No valid samples for Figure 1.")
        return

    percentiles = np.percentile(risk_scores, [25, 50, 75])
    low_idx = np.argmin(np.abs(risk_scores - percentiles[0]))
    mid_idx = np.argmin(np.abs(risk_scores - percentiles[1]))
    high_idx = np.argmin(np.abs(risk_scores - percentiles[2]))

    time_points = np.linspace(60, 240, 50)

    baseline_surv = estimate_baseline_survival_breslow(
        events, durations, risk_scores, time_points
    )
    baseline_surv = np.clip(baseline_surv, 0.001, 0.999)

    indices = [high_idx, mid_idx, low_idx]
    labels = ["High Risk", "Medium Risk", "Low Risk"]
    colors = ["#E74C3C", "#F39C12", "#2ECC71"]

    fig, ax = plt.subplots(figsize=(8, 6))

    for idx, label, color in zip(indices, labels, colors):
        surv = compute_individual_survival(risk_scores[idx], baseline_surv)
        ax.plot(
            time_points,
            surv,
            color=color,
            linewidth=2.5,
            label="%s (risk=%.3f)" % (label, risk_scores[idx]),
        )

    ax.set_xlabel("Time (minutes)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Survival Probability", fontsize=13, fontweight="bold")
    ax.set_title(
        "Individualized Survival Curves\nDA-DSN Predictions",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=11, loc="lower left", framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlim(60, 240)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Figure 1 saved to: %s" % save_path)


def extract_features(model, data_dict, device, max_samples=200):
    """Extract fusion-layer features from model."""
    model.eval()
    static = data_dict["static"]
    dynamic = data_dict["dynamic"]

    n = min(len(static), max_samples)
    np.random.seed(42)
    indices = np.random.choice(len(static), n, replace=False)

    batch_size = 64
    all_features = []

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_idx = indices[start:end]
            x_static = torch.tensor(
                static[batch_idx], dtype=torch.float32, device=device
            )
            x_dynamic = torch.tensor(
                dynamic[batch_idx], dtype=torch.float32, device=device
            )
            _, features = model(x_static, x_dynamic)
            all_features.append(features.cpu().numpy())

    return np.vstack(all_features)


def generate_figure_2(
    model_base_trans, model_coral, source_data, target_data, device, save_path
):
    """
    Figure 2: UMAP visualization of MMD domain adaptation.
    Compare features before and after MMD.
    """
    print("  Extracting features (Base+ResNet1D model)...")
    src_feat_base = extract_features(model_base_trans, source_data, device)
    tgt_feat_base = extract_features(model_base_trans, target_data, device)

    print("  Extracting features (MMD model)...")
    src_feat_coral = extract_features(model_coral, source_data, device)
    tgt_feat_coral = extract_features(model_coral, target_data, device)

    X_base = np.vstack([src_feat_base, tgt_feat_base])
    labels_base = np.array([0] * len(src_feat_base) + [1] * len(tgt_feat_base))

    X_coral = np.vstack([src_feat_coral, tgt_feat_coral])
    labels_coral = np.array([0] * len(src_feat_coral) + [1] * len(tgt_feat_coral))

    print("  Running UMAP dimensionality reduction...")
    import umap

    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    X_base_2d = reducer.fit_transform(X_base)
    X_coral_2d = reducer.fit_transform(X_coral)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, X_2d, labels, title in [
        (axes[0], X_base_2d, labels_base, "Base + ResNet1D (No MMD)"),
        (axes[1], X_coral_2d, labels_coral, "DA-DSN (With MMD)"),
    ]:
        mask_src = labels == 0
        mask_tgt = labels == 1
        ax.scatter(
            X_2d[mask_src, 0],
            X_2d[mask_src, 1],
            c="#3498DB",
            alpha=0.6,
            s=15,
            label="ShenYi (Source)",
        )
        ax.scatter(
            X_2d[mask_tgt, 0],
            X_2d[mask_tgt, 1],
            c="#E74C3C",
            alpha=0.6,
            s=15,
            label="FuDing (Target)",
        )
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("UMAP 1", fontsize=11)
        ax.set_ylabel("UMAP 2", fontsize=11)
        ax.legend(fontsize=10, loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.2, linestyle="--")

    plt.suptitle(
        "MMD Domain Adaptation: UMAP Feature Visualization",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Figure 2 saved to: %s" % save_path)


def concordance_index_fast(event_times, predicted_scores, event_observed):
    """Fast C-index implementation using vectorized operations."""
    n = len(event_times)
    concordant = 0
    comparable = 0

    for i in range(n):
        if event_observed[i] == 0:
            continue
        for j in range(i + 1, n):
            if event_times[i] < event_times[j]:
                comparable += 1
                if predicted_scores[i] > predicted_scores[j]:
                    concordant += 1
                elif predicted_scores[i] == predicted_scores[j]:
                    concordant += 0.5
            elif event_times[j] < event_times[i] and event_observed[j] == 1:
                comparable += 1
                if predicted_scores[j] > predicted_scores[i]:
                    concordant += 1
                elif predicted_scores[j] == predicted_scores[i]:
                    concordant += 0.5

    if comparable == 0:
        return 0.5
    return concordant / comparable


def train_model(
    model,
    src_loader,
    tgt_loader,
    device,
    cfg,
    use_coral=False,
    use_transformer=False,
    epochs=None,
    val_data=None,
):
    """Train a model with or without CORAL, with Early Stopping."""
    if epochs is None:
        epochs = cfg.training.epochs

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    if use_coral:
        model_name = "DA-DSN (MMD)"
    elif use_transformer:
        model_name = "Base+ResNet1D"
    else:
        model_name = "Base"

    print("Training %s model for %d epochs..." % (model_name, epochs))

    # Early Stopping setup
    patience = cfg.training.early_stopping_patience
    best_val_cindex = -1
    best_model_state = None
    patience_counter = 0
    best_epoch = 0

    for epoch in range(epochs):
        model.train()
        total_surv = 0
        total_coral = 0
        n_batches = 0

        if use_coral and tgt_loader is not None:
            for (x_s, x_d, ev, dur), (x_t, x_td, _, _) in zip(src_loader, tgt_loader):
                x_s = x_s.to(device)
                x_d = x_d.to(device)
                ev = ev.to(device)
                dur = dur.to(device)
                x_t = x_t.to(device)
                x_td = x_td.to(device)

                log_h, emb_s = model(x_s, x_d)
                _, emb_t = model(x_t, x_td)

                surv_loss = cox_partial_log_likelihood(log_h, ev, dur)
                # C: Use MMD instead of CORAL for domain adaptation
                c_loss = mmd_loss(emb_s, emb_t)
                loss = surv_loss + cfg.training.coral.lambda_coral * c_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_surv += surv_loss.item()
                total_coral += c_loss.item()
                n_batches += 1
        else:
            for batch in src_loader:
                x_s, x_d, ev, dur = batch
                x_s = x_s.to(device)
                x_d = x_d.to(device)
                ev = ev.to(device)
                dur = dur.to(device)

                log_h, _ = model(x_s, x_d)
                loss = cox_partial_log_likelihood(log_h, ev, dur)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_surv += loss.item()
                n_batches += 1

        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0 or epochs == 1:
            if use_coral:
                print(
                    "  Epoch %d/%d, SurvLoss=%.4f, CoralLoss=%.4f"
                    % (
                        epoch + 1,
                        epochs,
                        total_surv / max(n_batches, 1),
                        total_coral / max(n_batches, 1),
                    )
                )
            else:
                print(
                    "  Epoch %d/%d, Loss=%.4f"
                    % (epoch + 1, epochs, total_surv / max(n_batches, 1))
                )

        # Early Stopping: evaluate on validation set every 5 epochs
        if val_data is not None and ((epoch + 1) % 5 == 0 or epoch == epochs - 1):
            val_risk, val_events, val_durations = predict_risk(model, val_data, device)
            val_cindex = concordance_index_fast(val_durations, val_risk, val_events)

            if val_cindex > best_val_cindex:
                best_val_cindex = val_cindex
                best_model_state = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
                patience_counter = 0
                best_epoch = epoch + 1
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                print(
                    "    Val C-index: %.4f (Best: %.4f at epoch %d, Patience: %d/%d)"
                    % (
                        val_cindex,
                        best_val_cindex,
                        best_epoch,
                        patience_counter,
                        patience,
                    )
                )

            if patience_counter >= patience:
                print(
                    "  Early Stopping triggered at epoch %d. Best epoch: %d (C-index: %.4f)"
                    % (epoch + 1, best_epoch, best_val_cindex)
                )
                break

    # Restore best model if early stopping was used
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("  Restored best model from epoch %d" % best_epoch)

    return model


def compute_c_index_bootstrap(
    events, durations, risk_scores, n_bootstrap=20, max_samples=5000
):
    """
    Compute C-index with bootstrap 95% confidence interval.
    Uses subsampling for efficiency on large datasets.
    """
    from sklearn.utils import resample

    n = len(events)
    if n > max_samples:
        np.random.seed(42)
        subset_idx = np.random.choice(n, max_samples, replace=False)
        events = events[subset_idx]
        durations = durations[subset_idx]
        risk_scores = risk_scores[subset_idx]
        n = max_samples
        print("    Subsampled to %d for bootstrap" % n)

    c_indices = []

    for b in range(n_bootstrap):
        idx = resample(range(n), replace=True, n_samples=n)
        c = concordance_index_fast(durations[idx], risk_scores[idx], events[idx])
        c_indices.append(c)
        if (b + 1) % 5 == 0:
            print("    Bootstrap %d/%d" % (b + 1, n_bootstrap))

    c_mean = np.mean(c_indices)
    c_lower = np.percentile(c_indices, 2.5)
    c_upper = np.percentile(c_indices, 97.5)

    return c_mean, c_lower, c_upper


def compute_mmd(X_source, X_target):
    """
    Compute Maximum Mean Discrepancy (MMD) between source and target domains.
    Uses RBF kernel.
    """
    from sklearn.metrics.pairwise import rbf_kernel

    n_src = len(X_source)
    n_tgt = len(X_target)

    K_ss = rbf_kernel(X_source, X_source)
    K_tt = rbf_kernel(X_target, X_target)
    K_st = rbf_kernel(X_source, X_target)

    mmd = (
        K_ss.sum() / (n_src * n_src)
        + K_tt.sum() / (n_tgt * n_tgt)
        - 2 * K_st.sum() / (n_src * n_tgt)
    )

    return max(0, mmd)


def run_figure_generation():
    """Main entry: train models and generate figures."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    defaults = OmegaConf.load(os.path.join(project_root, "configs", "defaults.yaml"))
    config = OmegaConf.load(os.path.join(project_root, "configs", "config.yaml"))
    cfg = OmegaConf.merge(defaults, config)

    device = "cpu"
    debug_epochs = 20  # Increased from 1 to allow model convergence

    shared_loader = DialysisDataLoader(cfg)
    source_data = shared_loader.load_data(cfg.experiment.train_csv, is_training=True)
    target_data = shared_loader.load_data(
        cfg.experiment.external_csv, is_training=False
    )

    print("Source samples: %d" % source_data["static"].shape[0])
    print("Target samples: %d" % target_data["static"].shape[0])

    # Split source data into train/val for early stopping (80/20)
    n_source = source_data["static"].shape[0]
    np.random.seed(42)
    idx = np.random.permutation(n_source)
    split = int(0.8 * n_source)
    train_idx, val_idx = idx[:split], idx[split:]

    train_data = {
        "static": source_data["static"][train_idx],
        "dynamic": source_data["dynamic"][train_idx],
        "targets": {
            "event": source_data["targets"]["event"][train_idx],
            "duration": source_data["targets"]["duration"][train_idx],
        },
    }
    val_data = {
        "static": source_data["static"][val_idx],
        "dynamic": source_data["dynamic"][val_idx],
        "targets": {
            "event": source_data["targets"]["event"][val_idx],
            "duration": source_data["targets"]["duration"][val_idx],
        },
    }

    src_loader = create_source_only_dataloader(train_data, cfg.training.batch_size)
    tgt_loader = create_source_only_dataloader(target_data, cfg.training.batch_size)

    n_static = source_data["static"].shape[1]
    n_dynamic = source_data["dynamic"].shape[2]

    print("\n" + "=" * 60)
    print(
        "ABLATION STUDY (Train: %d, Val: %d, Max Epochs: %d)"
        % (len(train_idx), len(val_idx), debug_epochs)
    )
    print("=" * 60)

    model_base = DADSN(cfg, n_static, n_dynamic, use_transformer=False)
    model_base.to(device)
    model_base = train_model(
        model_base,
        src_loader,
        None,
        device,
        cfg,
        use_coral=False,
        use_transformer=False,
        epochs=debug_epochs,
        val_data=val_data,
    )

    model_base_trans = DADSN(cfg, n_static, n_dynamic, use_transformer=True)
    model_base_trans.to(device)
    model_base_trans = train_model(
        model_base_trans,
        src_loader,
        None,
        device,
        cfg,
        use_coral=False,
        use_transformer=True,
        epochs=debug_epochs,
        val_data=val_data,
    )

    model_coral = DADSN(cfg, n_static, n_dynamic, use_transformer=True)
    model_coral.to(device)
    model_coral = train_model(
        model_coral,
        src_loader,
        tgt_loader,
        device,
        cfg,
        use_coral=True,
        use_transformer=True,
        epochs=debug_epochs,
        val_data=val_data,
    )

    print("\n" + "=" * 60)
    print("EVALUATION ON TARGET DOMAIN (FuDing)")
    print("=" * 60)

    models = {
        "Base (Static Only)": model_base,
        "Base + ResNet1D": model_base_trans,
        "DA-DSN (MMD)": model_coral,
    }

    results = {}
    for name, model in models.items():
        risk_scores, events, durations = predict_risk(model, target_data, device)
        c_mean, c_lower, c_upper = compute_c_index_bootstrap(
            events, durations, risk_scores, n_bootstrap=50
        )
        results[name] = {
            "C-index": c_mean,
            "95% CI": [c_lower, c_upper],
        }
        print(
            "%s: C-index=%.4f (95%% CI: %.4f-%.4f)" % (name, c_mean, c_lower, c_upper)
        )

    print("\n" + "=" * 60)
    print("DOMAIN DISTRIBUTION ANALYSIS (MMD)")
    print("=" * 60)

    src_static = source_data["static"]
    tgt_static = target_data["static"]
    mmd_static = compute_mmd(src_static[:500], tgt_static[:500])
    print("MMD (Static Features): %.6f" % mmd_static)

    src_dynamic = source_data["dynamic"].reshape(len(source_data["dynamic"]), -1)
    tgt_dynamic = target_data["dynamic"].reshape(len(target_data["dynamic"]), -1)
    mmd_dynamic = compute_mmd(src_dynamic[:500], tgt_dynamic[:500])
    print("MMD (Dynamic Features): %.6f" % mmd_dynamic)

    save_dir = os.path.join(project_root, "figures")
    os.makedirs(save_dir, exist_ok=True)

    print("\nGenerating Figure 1: Individualized Survival Curves...")
    generate_figure_1(
        model_coral,
        target_data,
        device,
        os.path.join(save_dir, "figure1_survival_curves.png"),
    )

    print("\nGenerating Figure 2: UMAP MMD Visualization...")
    generate_figure_2(
        model_base_trans,
        model_coral,
        source_data,
        target_data,
        device,
        os.path.join(save_dir, "figure2_umap_coral.png"),
    )

    print("\nAll figures generated successfully!")
    print("Results saved to: %s" % save_dir)


if __name__ == "__main__":
    run_figure_generation()
