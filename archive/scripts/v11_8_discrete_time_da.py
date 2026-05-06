"""
v11.8 direction-one DA with v11.6 features and a discrete-time survival head.
"""

import copy
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split

import v11_5_feature_expansion_da as base
import v11_6_trend_context_da as featmod


CONFIG = dict(base.CONFIG)
CONFIG.update(
    {
        "results_path": "runs/v11_8_discrete_time_da_results.json",
        "seed": 42,
        "time_bin_edges": [30.0, 60.0, 120.0, 180.0, 240.0],
    }
)
base.CONFIG.update(CONFIG)
featmod.CONFIG.update(CONFIG)


def time_to_bin(times, bin_edges):
    edges = np.asarray(bin_edges, dtype=np.float32)
    return np.searchsorted(edges, times, side="right").astype(np.int64)


def discrete_time_nll(logits, event, time, bin_edges):
    probs = torch.softmax(logits, dim=1)
    bin_idx = torch.bucketize(
        time, torch.tensor(bin_edges, device=time.device), right=True
    )
    bin_idx = torch.clamp(bin_idx, 0, probs.size(1) - 1)

    event_mask = event.float()
    event_prob = probs.gather(1, bin_idx.unsqueeze(1)).squeeze(1)
    cens_prob = torch.flip(torch.cumsum(torch.flip(probs, dims=[1]), dim=1), dims=[1])
    surv_prob = cens_prob.gather(1, bin_idx.unsqueeze(1)).squeeze(1)

    loss_event = -torch.log(event_prob + 1e-8) * event_mask
    loss_cens = -torch.log(surv_prob + 1e-8) * (1.0 - event_mask)
    return (loss_event + loss_cens).mean()


def risk_score_from_logits(logits, bin_edges):
    probs = torch.softmax(logits, dim=1)
    edges = np.asarray(bin_edges, dtype=np.float32)
    mids = []
    left = 0.0
    for right in edges:
        mids.append((left + right) / 2.0)
        left = right
    mids.append(edges[-1] + 30.0)
    mids = torch.tensor(mids, device=logits.device, dtype=torch.float32)
    expected_time = (probs * mids.unsqueeze(0)).sum(dim=1)
    return -expected_time


class DiscreteTimeNet(nn.Module):
    def __init__(self, input_dim, d_model, dropout, num_bins):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(d_model, num_bins)

    def forward(self, x):
        emb = self.encoder(x)
        logits = self.head(emb)
        return emb, logits


def evaluate_cindex(model, x_eval, e_eval, t_eval):
    model.eval()
    with torch.no_grad():
        x_tensor = torch.as_tensor(x_eval, dtype=torch.float32, device=CONFIG["device"])
        _, logits = model(x_tensor)
        scores = risk_score_from_logits(logits, CONFIG["time_bin_edges"]).cpu().numpy()
    return float(concordance_index(t_eval, -scores, e_eval))


def set_trainable_state(model, phase):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = True
    if phase in {"partial_unfreeze", "full_finetune"}:
        for idx in [4, 5]:
            for param in model.encoder[idx].parameters():
                param.requires_grad = True
    if phase == "full_finetune":
        for param in model.encoder.parameters():
            param.requires_grad = True


def build_optimizer(model, phase):
    if phase == "head_only":
        return optim.AdamW(
            model.head.parameters(), lr=CONFIG["head_lr"], weight_decay=1e-3
        )
    if phase == "partial_unfreeze":
        param_groups = [
            {"params": model.head.parameters(), "lr": CONFIG["head_lr"]},
            {"params": model.encoder[4].parameters(), "lr": CONFIG["top_encoder_lr"]},
            {"params": model.encoder[5].parameters(), "lr": CONFIG["top_encoder_lr"]},
        ]
        return optim.AdamW(param_groups, weight_decay=1e-3)
    if phase == "full_finetune":
        param_groups = [
            {"params": model.head.parameters(), "lr": CONFIG["head_lr"]},
            {"params": model.encoder.parameters(), "lr": CONFIG["full_lr"]},
        ]
        return optim.AdamW(param_groups, weight_decay=1e-4)
    raise ValueError(f"Unsupported phase: {phase}")


def run_source_pretrain(model, source_loader, x_val, e_val, t_val):
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-4)
    best_val = -np.inf
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    patience_left = CONFIG["pretrain_patience"]

    for epoch in range(1, CONFIG["pretrain_max_epochs"] + 1):
        model.train()
        for x_s, e_s, t_s in source_loader:
            x_s = x_s.to(CONFIG["device"])
            e_s = e_s.to(CONFIG["device"])
            t_s = t_s.to(CONFIG["device"])
            optimizer.zero_grad()
            _, logits = model(x_s)
            loss = discrete_time_nll(logits, e_s, t_s, CONFIG["time_bin_edges"])
            loss.backward()
            optimizer.step()

        val_cindex = evaluate_cindex(model, x_val, e_val, t_val)
        if val_cindex > best_val:
            best_val = val_cindex
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_left = CONFIG["pretrain_patience"]
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    model.load_state_dict(best_state)
    return best_val, best_epoch, best_state


def run_head_only_baseline(model, target_train_loader, x_val, e_val, t_val):
    for param in model.encoder.parameters():
        param.requires_grad = False
    optimizer = optim.AdamW(
        model.head.parameters(), lr=CONFIG["head_lr"], weight_decay=1e-3
    )
    best_val = evaluate_cindex(model, x_val, e_val, t_val)
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    patience_left = CONFIG["head_only_patience"]

    for epoch in range(1, CONFIG["head_only_max_epochs"] + 1):
        model.train()
        for x_t, e_t, t_t in target_train_loader:
            x_t = x_t.to(CONFIG["device"])
            e_t = e_t.to(CONFIG["device"])
            t_t = t_t.to(CONFIG["device"])
            optimizer.zero_grad()
            _, logits = model(x_t)
            loss = discrete_time_nll(logits, e_t, t_t, CONFIG["time_bin_edges"])
            loss.backward()
            optimizer.step()

        val_cindex = evaluate_cindex(model, x_val, e_val, t_val)
        if val_cindex > best_val:
            best_val = val_cindex
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_left = CONFIG["head_only_patience"]
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    model.load_state_dict(best_state)
    return best_val, best_epoch, best_state


def run_da_phase(
    model,
    phase,
    max_epochs,
    patience,
    source_loader,
    target_train_loader,
    x_val,
    e_val,
    t_val,
):
    set_trainable_state(model, phase)
    optimizer = build_optimizer(model, phase)
    best_val = evaluate_cindex(model, x_val, e_val, t_val)
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    patience_left = patience

    for epoch in range(1, max_epochs + 1):
        model.train()
        source_iter = iter(source_loader)
        for x_t, e_t, t_t in target_train_loader:
            try:
                x_s, e_s, t_s = next(source_iter)
            except StopIteration:
                source_iter = iter(source_loader)
                x_s, e_s, t_s = next(source_iter)

            x_t = x_t.to(CONFIG["device"])
            e_t = e_t.to(CONFIG["device"])
            t_t = t_t.to(CONFIG["device"])
            x_s = x_s.to(CONFIG["device"])
            e_s = e_s.to(CONFIG["device"])
            t_s = t_s.to(CONFIG["device"])

            optimizer.zero_grad()
            emb_t, logits_t = model(x_t)
            emb_s, logits_s = model(x_s)
            target_loss = discrete_time_nll(
                logits_t, e_t, t_t, CONFIG["time_bin_edges"]
            )
            replay_loss = discrete_time_nll(
                logits_s, e_s, t_s, CONFIG["time_bin_edges"]
            )
            align_loss = base.conditional_alignment_loss(emb_s, e_s, emb_t, e_t)
            loss = (
                target_loss
                + CONFIG["source_replay_weight"] * replay_loss
                + CONFIG["conditional_align_weight"] * align_loss
            )
            loss.backward()
            optimizer.step()

        val_cindex = evaluate_cindex(model, x_val, e_val, t_val)
        if val_cindex > best_val:
            best_val = val_cindex
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    model.load_state_dict(best_state)
    return best_val, best_epoch, best_state


def main():
    base.set_seed(CONFIG["seed"])
    feat_source, feat_target, e_source, t_source, e_target, t_target, feature_names = (
        featmod.build_feature_tables()
    )
    x_source, x_target = featmod.prepare_group_matrices(feat_source, feat_target)

    x_adapt_pool, x_test, e_adapt_pool, e_test, t_adapt_pool, t_test = train_test_split(
        x_target,
        e_target,
        t_target,
        test_size=1 - CONFIG["target_adapt_ratio"],
        random_state=CONFIG["seed"],
        stratify=e_target,
    )
    x_train, x_val, e_train, e_val, t_train, t_val = train_test_split(
        x_adapt_pool,
        e_adapt_pool,
        t_adapt_pool,
        test_size=CONFIG["target_val_ratio_within_adapt"],
        random_state=CONFIG["seed"],
        stratify=e_adapt_pool,
    )

    source_loader = base.make_loader(
        x_source, e_source, t_source, CONFIG["batch_size"], shuffle=True, drop_last=True
    )
    target_train_loader = base.make_loader(
        x_train, e_train, t_train, CONFIG["batch_size"], shuffle=True, drop_last=False
    )

    input_dim = x_source.shape[1]
    num_bins = len(CONFIG["time_bin_edges"]) + 1

    base_model = DiscreteTimeNet(
        input_dim, CONFIG["d_model"], CONFIG["dropout"], num_bins
    ).to(CONFIG["device"])
    pre_val, pre_epoch, source_state = run_source_pretrain(
        base_model, source_loader, x_val, e_val, t_val
    )
    zero_shot_test = evaluate_cindex(base_model, x_test, e_test, t_test)

    head_model = DiscreteTimeNet(
        input_dim, CONFIG["d_model"], CONFIG["dropout"], num_bins
    ).to(CONFIG["device"])
    head_model.load_state_dict(source_state)
    head_val, head_epoch, _ = run_head_only_baseline(
        head_model, target_train_loader, x_val, e_val, t_val
    )
    head_test = evaluate_cindex(head_model, x_test, e_test, t_test)

    da_model = DiscreteTimeNet(
        input_dim, CONFIG["d_model"], CONFIG["dropout"], num_bins
    ).to(CONFIG["device"])
    da_model.load_state_dict(source_state)
    phase_results = []
    for phase, max_epochs, patience in [
        ("head_only", CONFIG["head_only_max_epochs"], CONFIG["head_only_patience"]),
        (
            "partial_unfreeze",
            CONFIG["partial_unfreeze_max_epochs"],
            CONFIG["partial_unfreeze_patience"],
        ),
        (
            "full_finetune",
            CONFIG["full_finetune_max_epochs"],
            CONFIG["full_finetune_patience"],
        ),
    ]:
        val_score, best_epoch, _ = run_da_phase(
            da_model,
            phase,
            max_epochs,
            patience,
            source_loader,
            target_train_loader,
            x_val,
            e_val,
            t_val,
        )
        phase_results.append(
            {
                "phase": phase,
                "best_adapt_val_cindex": val_score,
                "best_epoch": best_epoch,
            }
        )
    da_test = evaluate_cindex(da_model, x_test, e_test, t_test)

    results = {
        "config": CONFIG,
        "feature_count": int(input_dim),
        "feature_names": feature_names,
        "source_pretrain": {
            "best_adapt_val_cindex": pre_val,
            "best_epoch": pre_epoch,
            "zero_shot_test_cindex": zero_shot_test,
        },
        "head_only": {
            "best_adapt_val_cindex": head_val,
            "best_epoch": head_epoch,
            "test_cindex": head_test,
        },
        "layerwise_da": {
            "phase_results": phase_results,
            "test_cindex": da_test,
            "gain_vs_zero_shot": da_test - zero_shot_test,
        },
    }

    with open(CONFIG["results_path"], "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(json.dumps(results["layerwise_da"], ensure_ascii=False, indent=2))
    print(f"Saved results to {CONFIG['results_path']}")


if __name__ == "__main__":
    main()
