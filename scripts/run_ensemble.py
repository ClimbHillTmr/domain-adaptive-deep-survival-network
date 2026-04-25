"""
P1: 5-Fold Cross-Validation Ensemble Training
Trains 5 models on different folds and averages predictions for improved C-index.
"""
import os
import sys
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import DialysisDataLoader
from src.models.dadsn_model import DADSN, cox_partial_log_likelihood, mmd_loss
from src.training.dadsn_runner import create_source_only_dataloader, predict_risk


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


def train_single_fold(
    fold_idx, train_data, val_data, tgt_data, cfg, device, use_coral=False
):
    """Train a single fold model."""
    n_static = train_data["static"].shape[1]
    n_dynamic = train_data["dynamic"].shape[2]

    model = DADSN(cfg, n_static, n_dynamic, use_transformer=True)
    model.to(device)

    src_loader = create_source_only_dataloader(train_data, cfg.training.batch_size)
    tgt_loader = create_source_only_dataloader(tgt_data, cfg.training.batch_size) if use_coral else None

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    patience = cfg.training.early_stopping_patience
    best_val_cindex = -1
    best_model_state = None
    patience_counter = 0
    best_epoch = 0

    epochs = cfg.training.epochs

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

        if (epoch + 1) % 5 == 0 or epoch == 0:
            if use_coral:
                print(
                    "  Fold %d, Epoch %d/%d, SurvLoss=%.4f, MMDLoss=%.4f"
                    % (
                        fold_idx,
                        epoch + 1,
                        epochs,
                        total_surv / max(n_batches, 1),
                        total_coral / max(n_batches, 1),
                    )
                )
            else:
                print(
                    "  Fold %d, Epoch %d/%d, Loss=%.4f"
                    % (fold_idx, epoch + 1, epochs, total_surv / max(n_batches, 1))
                )

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
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

            if patience_counter >= patience:
                print(
                    "  Fold %d: Early Stopping at epoch %d. Best: %d (C-index: %.4f)"
                    % (fold_idx, epoch + 1, best_epoch, best_val_cindex)
                )
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def run_ensemble():
    """Main entry: 5-fold CV ensemble."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    defaults = OmegaConf.load(os.path.join(project_root, "configs", "defaults.yaml"))
    config = OmegaConf.load(os.path.join(project_root, "configs", "config.yaml"))
    cfg = OmegaConf.merge(defaults, config)

    device = "cpu"

    shared_loader = DialysisDataLoader(cfg)
    source_data = shared_loader.load_data(cfg.experiment.train_csv, is_training=True)
    target_data = shared_loader.load_data(
        cfg.experiment.external_csv, is_training=False
    )

    print("Source samples: %d" % source_data["static"].shape[0])
    print("Target samples: %d" % target_data["static"].shape[0])

    n_source = source_data["static"].shape[0]
    n_folds = 5
    fold_size = n_source // n_folds

    np.random.seed(42)
    idx = np.random.permutation(n_source)

    print("\n" + "=" * 60)
    print("5-FOLD ENSEMBLE TRAINING")
    print("=" * 60)

    fold_models = []
    fold_val_cindices = []

    for fold in range(n_folds):
        print("\n--- Fold %d/%d ---" % (fold + 1, n_folds))

        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < n_folds - 1 else n_source

        val_idx = idx[val_start:val_end]
        train_idx = np.concatenate([idx[:val_start], idx[val_end:]])

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

        model = train_single_fold(
            fold + 1, train_data, val_data, target_data, cfg, device, use_coral=True
        )
        fold_models.append(model)

        val_risk, val_events, val_durations = predict_risk(model, val_data, device)
        val_cindex = concordance_index_fast(val_durations, val_risk, val_events)
        fold_val_cindices.append(val_cindex)
        print("  Fold %d Val C-index: %.4f" % (fold + 1, val_cindex))

    print("\n" + "=" * 60)
    print("ENSEMBLE EVALUATION ON TARGET DOMAIN")
    print("=" * 60)

    all_risk_scores = []
    for i, model in enumerate(fold_models):
        risk_scores, _, _ = predict_risk(model, target_data, device)
        all_risk_scores.append(risk_scores)
        print("  Fold %d risk scores computed" % (i + 1))

    ensemble_risk = np.mean(all_risk_scores, axis=0)
    events = target_data["targets"]["event"]
    durations = target_data["targets"]["duration"]

    ensemble_cindex = concordance_index_fast(durations, ensemble_risk, events)
    print("\nEnsemble C-index: %.4f" % ensemble_cindex)
    print("Individual fold C-indices: %s" % str(["%.4f" % c for c in fold_val_cindices]))
    print("Mean fold C-index: %.4f" % np.mean(fold_val_cindices))


if __name__ == "__main__":
    run_ensemble()
