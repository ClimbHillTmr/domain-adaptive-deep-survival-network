"""
v11.20 replace Cox with discrete-time Logistic-Hazard on top of v11.18 DANN setup.
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
import v11_15_rich_deterioration_da as featmod
import v11_17_da_upgrade_benchmark as dannbase


CONFIG = dict(base.CONFIG)
CONFIG.update(
    {
        "results_path": "runs/v11_20_loghazard_dann_results.json",
        "seed": 42,
        "d_model": 64,
        "dropout": 0.15,
        "lr": 8e-4,
        "head_lr": 1e-4,
        "top_encoder_lr": 7e-5,
        "full_lr": 3e-5,
        "num_transformer_layers": 2,
        "num_heads": 4,
        "benchmark_batch_size": 512,
        "eval_batch_size": 1024,
        "domain_hidden": 64,
        "pretrain_max_epochs": 18,
        "pretrain_patience": 4,
        "head_only_max_epochs": 4,
        "head_only_patience": 2,
        "partial_unfreeze_max_epochs": 6,
        "partial_unfreeze_patience": 2,
        "full_finetune_max_epochs": 20,
        "full_finetune_patience": 4,
        "adv_weight": 0.05,
        "source_replay_weight": 0.1,
        "num_time_bins": 6,
        "time_bin_edges": [30, 60, 90, 120, 180],
    }
)
base.CONFIG.update(CONFIG)
featmod.CONFIG.update(CONFIG)
dannbase.CONFIG.update(CONFIG)


def duration_to_bin(duration):
    edges = np.asarray(CONFIG["time_bin_edges"], dtype=float)
    return np.searchsorted(edges, duration, side="right").astype(np.int64)


def make_loghazard_targets(event, duration):
    event = np.asarray(event).astype(np.int64)
    bins = duration_to_bin(np.asarray(duration))
    return bins, event


def loghazard_nll(logits, event, duration_bins):
    hazards = torch.sigmoid(logits)
    batch_idx = torch.arange(logits.size(0), device=logits.device)
    max_bin = logits.size(1)
    mask = torch.arange(max_bin, device=logits.device).unsqueeze(0) < duration_bins.unsqueeze(1)
    surv_term = torch.log1p(-hazards + 1e-7) * mask
    loss = -surv_term.sum(dim=1)
    event_mask = event > 0
    if event_mask.any():
        event_bins = duration_bins[event_mask]
        event_logits = hazards[event_mask, event_bins]
        loss[event_mask] += -torch.log(event_logits + 1e-7)
    return loss.mean()


def logits_to_risk(logits):
    hazards = torch.sigmoid(logits)
    survival = torch.cumprod(1.0 - hazards, dim=1)
    risk = 1.0 - survival[:, -1]
    return risk


class TabTransformerLogHazardNet(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout, domain_hidden, num_time_bins):
        super().__init__()
        self.feature_weight = nn.Parameter(torch.randn(input_dim, d_model) * 0.02)
        self.feature_bias = nn.Parameter(torch.zeros(input_dim, d_model))
        self.feature_pos = nn.Parameter(torch.randn(input_dim, d_model) * 0.02)
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
        self.hazard_head = nn.Linear(d_model, num_time_bins)
        self.domain_head = dannbase.DomainClassifier(d_model, domain_hidden, dropout)

    def encode(self, x):
        tokens = x.unsqueeze(-1) * self.feature_weight.unsqueeze(0) + self.feature_bias.unsqueeze(0)
        tokens = tokens + self.feature_pos.unsqueeze(0)
        encoded = self.transformer(tokens)
        return self.norm(encoded.mean(dim=1))

    def forward(self, x, grl_coeff=None):
        emb = self.encode(x)
        hazard_logits = self.hazard_head(emb)
        domain_logits = None
        if grl_coeff is not None:
            domain_logits = self.domain_head(dannbase.grad_reverse(emb, grl_coeff))
        return emb, hazard_logits, domain_logits


def evaluate_cindex(model, x_eval, e_eval, t_eval):
    model.eval()
    risks = []
    with torch.no_grad():
        for start in range(0, len(x_eval), CONFIG["eval_batch_size"]):
            end = start + CONFIG["eval_batch_size"]
            x_batch = torch.tensor(x_eval[start:end], dtype=torch.float32, device=CONFIG["device"])
            _, logits, _ = model(x_batch, grl_coeff=None)
            risks.append(logits_to_risk(logits).detach().cpu().numpy())
    risk_scores = np.concatenate(risks, axis=0)
    return concordance_index(t_eval, -risk_scores, e_eval)


def set_trainable_state(model, phase):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.hazard_head.parameters():
        param.requires_grad = True
    for param in model.domain_head.parameters():
        param.requires_grad = True
    if phase in {"partial_unfreeze", "full_finetune"}:
        for param in model.transformer.layers[-1].parameters():
            param.requires_grad = True
    if phase == "full_finetune":
        model.feature_weight.requires_grad = True
        model.feature_bias.requires_grad = True
        model.feature_pos.requires_grad = True
        for param in model.transformer.parameters():
            param.requires_grad = True
        for param in model.norm.parameters():
            param.requires_grad = True


def build_optimizer(model, phase):
    if phase == "head_only":
        groups = [
            {"params": model.hazard_head.parameters(), "lr": CONFIG["head_lr"]},
            {"params": model.domain_head.parameters(), "lr": CONFIG["head_lr"]},
        ]
        return optim.AdamW(groups, weight_decay=1e-3)
    if phase == "partial_unfreeze":
        groups = [
            {"params": model.hazard_head.parameters(), "lr": CONFIG["head_lr"]},
            {"params": model.domain_head.parameters(), "lr": CONFIG["head_lr"]},
            {"params": model.transformer.layers[-1].parameters(), "lr": CONFIG["top_encoder_lr"]},
        ]
        return optim.AdamW(groups, weight_decay=1e-3)
    if phase == "full_finetune":
        groups = [
            {"params": model.hazard_head.parameters(), "lr": CONFIG["head_lr"]},
            {"params": model.domain_head.parameters(), "lr": CONFIG["head_lr"]},
            {"params": model.transformer.parameters(), "lr": CONFIG["full_lr"]},
            {"params": [model.feature_weight, model.feature_bias, model.feature_pos], "lr": CONFIG["full_lr"]},
            {"params": model.norm.parameters(), "lr": CONFIG["full_lr"]},
        ]
        return optim.AdamW(groups, weight_decay=1e-4)
    raise ValueError(phase)


class SurvivalBinDataset(torch.utils.data.Dataset):
    def __init__(self, x, e, t):
        bins, events = make_loghazard_targets(e, t)
        self.x = torch.tensor(x, dtype=torch.float32)
        self.e = torch.tensor(events, dtype=torch.float32)
        self.t = torch.tensor(t, dtype=torch.float32)
        self.bins = torch.tensor(bins, dtype=torch.long)

    def __len__(self):
        return len(self.e)

    def __getitem__(self, idx):
        return self.x[idx], self.e[idx], self.t[idx], self.bins[idx]


def make_loader(x, e, t, batch_size, shuffle, drop_last):
    ds = SurvivalBinDataset(x, e, t)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


def run_source_pretrain(model, source_loader, x_val, e_val, t_val):
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-4)
    best_val, best_epoch = -np.inf, 0
    best_state = copy.deepcopy(model.state_dict())
    patience_left = CONFIG["pretrain_patience"]
    for epoch in range(1, CONFIG["pretrain_max_epochs"] + 1):
        model.train()
        for x_s, e_s, _, b_s in source_loader:
            x_s = x_s.to(CONFIG["device"])
            e_s = e_s.to(CONFIG["device"])
            b_s = b_s.to(CONFIG["device"])
            optimizer.zero_grad()
            _, logits, _ = model(x_s, grl_coeff=None)
            loss = loghazard_nll(logits, e_s, b_s)
            loss.backward()
            optimizer.step()
        val_cindex = evaluate_cindex(model, x_val, e_val, t_val)
        if val_cindex > best_val:
            best_val, best_epoch = val_cindex, epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_left = CONFIG["pretrain_patience"]
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    model.load_state_dict(best_state)
    return best_val, best_epoch, best_state


def run_head_only(model, target_loader, x_val, e_val, t_val):
    set_trainable_state(model, "head_only")
    optimizer = build_optimizer(model, "head_only")
    best_val, best_epoch = evaluate_cindex(model, x_val, e_val, t_val), 0
    best_state = copy.deepcopy(model.state_dict())
    patience_left = CONFIG["head_only_patience"]
    for epoch in range(1, CONFIG["head_only_max_epochs"] + 1):
        model.train()
        for x_t, e_t, _, b_t in target_loader:
            x_t = x_t.to(CONFIG["device"])
            e_t = e_t.to(CONFIG["device"])
            b_t = b_t.to(CONFIG["device"])
            optimizer.zero_grad()
            _, logits, _ = model(x_t, grl_coeff=None)
            loss = loghazard_nll(logits, e_t, b_t)
            loss.backward()
            optimizer.step()
        val_cindex = evaluate_cindex(model, x_val, e_val, t_val)
        if val_cindex > best_val:
            best_val, best_epoch = val_cindex, epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_left = CONFIG["head_only_patience"]
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    model.load_state_dict(best_state)
    return best_val, best_epoch, best_state


def run_dann(model, source_loader, target_loader, x_val, e_val, t_val):
    phase_results = []
    for phase, max_epochs, patience in [
        ("head_only", CONFIG["head_only_max_epochs"], CONFIG["head_only_patience"]),
        ("partial_unfreeze", CONFIG["partial_unfreeze_max_epochs"], CONFIG["partial_unfreeze_patience"]),
        ("full_finetune", CONFIG["full_finetune_max_epochs"], CONFIG["full_finetune_patience"]),
    ]:
        set_trainable_state(model, phase)
        optimizer = build_optimizer(model, phase)
        best_val, best_epoch = evaluate_cindex(model, x_val, e_val, t_val), 0
        best_state = copy.deepcopy(model.state_dict())
        patience_left = patience
        for epoch in range(1, max_epochs + 1):
            model.train()
            coeff = min(1.0, epoch / max_epochs)
            source_iter = iter(source_loader)
            for x_t, e_t, _, b_t in target_loader:
                try:
                    x_s, e_s, _, b_s = next(source_iter)
                except StopIteration:
                    source_iter = iter(source_loader)
                    x_s, e_s, _, b_s = next(source_iter)
                x_t = x_t.to(CONFIG["device"])
                e_t = e_t.to(CONFIG["device"])
                b_t = b_t.to(CONFIG["device"])
                x_s = x_s.to(CONFIG["device"])
                e_s = e_s.to(CONFIG["device"])
                b_s = b_s.to(CONFIG["device"])
                optimizer.zero_grad()
                _, logits_t, dom_t = model(x_t, grl_coeff=coeff)
                _, logits_s, dom_s = model(x_s, grl_coeff=coeff)
                target_loss = loghazard_nll(logits_t, e_t, b_t)
                source_loss = loghazard_nll(logits_s, e_s, b_s)
                adv_loss = dannbase.domain_loss_from_logits(dom_s, dom_t)
                loss = target_loss + CONFIG["source_replay_weight"] * source_loss + CONFIG["adv_weight"] * adv_loss
                loss.backward()
                optimizer.step()
            val_cindex = evaluate_cindex(model, x_val, e_val, t_val)
            if val_cindex > best_val:
                best_val, best_epoch = val_cindex, epoch
                best_state = copy.deepcopy(model.state_dict())
                patience_left = patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break
        model.load_state_dict(best_state)
        phase_results.append({"phase": phase, "best_adapt_val_cindex": best_val, "best_epoch": best_epoch})
    return phase_results


def main():
    base.set_seed(CONFIG["seed"])
    feat_source, feat_target, e_source, t_source, e_target, t_target, feature_names = featmod.build_feature_tables()
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

    source_loader = make_loader(x_source, e_source, t_source, CONFIG["benchmark_batch_size"], shuffle=True, drop_last=True)
    target_loader = make_loader(x_train, e_train, t_train, CONFIG["benchmark_batch_size"], shuffle=True, drop_last=False)

    model = TabTransformerLogHazardNet(
        input_dim=x_source.shape[1],
        d_model=CONFIG["d_model"],
        nhead=CONFIG["num_heads"],
        num_layers=CONFIG["num_transformer_layers"],
        dropout=CONFIG["dropout"],
        domain_hidden=CONFIG["domain_hidden"],
        num_time_bins=CONFIG["num_time_bins"],
    ).to(CONFIG["device"])
    pre_val, pre_epoch, source_state = run_source_pretrain(model, source_loader, x_val, e_val, t_val)
    zero_shot_test = evaluate_cindex(model, x_test, e_test, t_test)

    head_model = TabTransformerLogHazardNet(
        input_dim=x_source.shape[1],
        d_model=CONFIG["d_model"],
        nhead=CONFIG["num_heads"],
        num_layers=CONFIG["num_transformer_layers"],
        dropout=CONFIG["dropout"],
        domain_hidden=CONFIG["domain_hidden"],
        num_time_bins=CONFIG["num_time_bins"],
    ).to(CONFIG["device"])
    head_model.load_state_dict(source_state)
    head_val, head_epoch, _ = run_head_only(head_model, target_loader, x_val, e_val, t_val)
    head_test = evaluate_cindex(head_model, x_test, e_test, t_test)

    da_model = TabTransformerLogHazardNet(
        input_dim=x_source.shape[1],
        d_model=CONFIG["d_model"],
        nhead=CONFIG["num_heads"],
        num_layers=CONFIG["num_transformer_layers"],
        dropout=CONFIG["dropout"],
        domain_hidden=CONFIG["domain_hidden"],
        num_time_bins=CONFIG["num_time_bins"],
    ).to(CONFIG["device"])
    da_model.load_state_dict(source_state)
    phase_results = run_dann(da_model, source_loader, target_loader, x_val, e_val, t_val)
    da_test = evaluate_cindex(da_model, x_test, e_test, t_test)

    results = {
        "config": CONFIG,
        "feature_count": int(x_source.shape[1]),
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
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"Saved results to {CONFIG['results_path']}")


if __name__ == "__main__":
    main()
