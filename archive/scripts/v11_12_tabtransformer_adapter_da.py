"""
v11.12 tabtransformer with domain-specific adapters for source/target calibration.
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
        "results_path": "runs/v11_12_tabtransformer_adapter_da_results.json",
        "seed": 42,
        "d_model": 64,
        "dropout": 0.15,
        "lr": 8e-4,
        "head_lr": 1e-4,
        "top_encoder_lr": 7e-5,
        "full_lr": 3e-5,
        "num_transformer_layers": 2,
        "num_heads": 4,
        "adapter_bottleneck": 24,
        "benchmark_batch_size": 512,
        "eval_batch_size": 1024,
    }
)
base.CONFIG.update(CONFIG)
featmod.CONFIG.update(CONFIG)


class Adapter(nn.Module):
    def __init__(self, d_model, bottleneck, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, bottleneck),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.norm(x + self.net(x))


class TabTransformerAdapterNet(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout, adapter_bottleneck):
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
        self.shared_norm = nn.LayerNorm(d_model)
        self.source_adapter = Adapter(d_model, adapter_bottleneck, dropout)
        self.target_adapter = Adapter(d_model, adapter_bottleneck, dropout)
        self.hazard_head = nn.Linear(d_model, 1)

    def forward_shared(self, x):
        tokens = x.unsqueeze(-1) * self.feature_weight.unsqueeze(0) + self.feature_bias.unsqueeze(0)
        tokens = tokens + self.feature_pos.unsqueeze(0)
        encoded = self.transformer(tokens)
        return self.shared_norm(encoded.mean(dim=1))

    def forward(self, x, domain="source"):
        shared_emb = self.forward_shared(x)
        if domain == "target":
            emb = self.target_adapter(shared_emb)
        else:
            emb = self.source_adapter(shared_emb)
        hazard = self.hazard_head(emb)
        return emb, hazard, shared_emb


def evaluate_cindex(model, x_eval, e_eval, t_eval, domain):
    model.eval()
    hazards = []
    with torch.no_grad():
        for start in range(0, len(x_eval), CONFIG["eval_batch_size"]):
            end = start + CONFIG["eval_batch_size"]
            x_batch = torch.tensor(x_eval[start:end], dtype=torch.float32, device=CONFIG["device"])
            _, hazard, _ = model(x_batch, domain=domain)
            hazards.append(hazard.squeeze(-1).detach().cpu().numpy())
    risk_scores = np.concatenate(hazards, axis=0)
    return concordance_index(t_eval, -risk_scores, e_eval)


def set_trainable_state(model, phase):
    for param in model.parameters():
        param.requires_grad = False

    for param in model.hazard_head.parameters():
        param.requires_grad = True
    for param in model.target_adapter.parameters():
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
        for param in model.shared_norm.parameters():
            param.requires_grad = True
        for param in model.source_adapter.parameters():
            param.requires_grad = True


def build_optimizer(model, phase):
    if phase == "head_only":
        param_groups = [
            {"params": model.hazard_head.parameters(), "lr": CONFIG["head_lr"]},
            {"params": model.target_adapter.parameters(), "lr": CONFIG["head_lr"]},
        ]
        return optim.AdamW(param_groups, weight_decay=1e-3)

    if phase == "partial_unfreeze":
        param_groups = [
            {"params": model.hazard_head.parameters(), "lr": CONFIG["head_lr"]},
            {"params": model.target_adapter.parameters(), "lr": CONFIG["head_lr"]},
            {"params": model.transformer.layers[-1].parameters(), "lr": CONFIG["top_encoder_lr"]},
        ]
        return optim.AdamW(param_groups, weight_decay=1e-3)

    if phase == "full_finetune":
        param_groups = [
            {"params": model.hazard_head.parameters(), "lr": CONFIG["head_lr"]},
            {"params": model.target_adapter.parameters(), "lr": CONFIG["top_encoder_lr"]},
            {"params": model.source_adapter.parameters(), "lr": CONFIG["top_encoder_lr"]},
            {"params": model.transformer.parameters(), "lr": CONFIG["full_lr"]},
            {"params": [model.feature_weight, model.feature_bias, model.feature_pos], "lr": CONFIG["full_lr"]},
            {"params": model.shared_norm.parameters(), "lr": CONFIG["full_lr"]},
        ]
        return optim.AdamW(param_groups, weight_decay=1e-4)

    raise ValueError(phase)


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
            _, hazard, _ = model(x_s, domain="source")
            loss = base.cox_loss(hazard, e_s, t_s)
            loss.backward()
            optimizer.step()

        val_cindex = evaluate_cindex(model, x_val, e_val, t_val, domain="source")
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
    set_trainable_state(model, "head_only")
    optimizer = build_optimizer(model, "head_only")
    best_val = evaluate_cindex(model, x_val, e_val, t_val, domain="target")
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
            _, hazard, _ = model(x_t, domain="target")
            loss = base.cox_loss(hazard, e_t, t_t)
            loss.backward()
            optimizer.step()

        val_cindex = evaluate_cindex(model, x_val, e_val, t_val, domain="target")
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


def run_da_phase(model, phase, max_epochs, patience, source_loader, target_train_loader, x_val, e_val, t_val):
    set_trainable_state(model, phase)
    optimizer = build_optimizer(model, phase)
    best_val = evaluate_cindex(model, x_val, e_val, t_val, domain="target")
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
            emb_t, hazard_t, shared_t = model(x_t, domain="target")
            emb_s, hazard_s, shared_s = model(x_s, domain="source")
            target_loss = base.cox_loss(hazard_t, e_t, t_t)
            replay_loss = base.cox_loss(hazard_s, e_s, t_s)
            align_loss = base.conditional_alignment_loss(shared_s, e_s, shared_t, e_t)
            loss = (
                target_loss
                + CONFIG["source_replay_weight"] * replay_loss
                + CONFIG["conditional_align_weight"] * align_loss
            )
            loss.backward()
            optimizer.step()

        val_cindex = evaluate_cindex(model, x_val, e_val, t_val, domain="target")
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
    print("Loading v11.6 feature tables for tabtransformer adapter DA...")
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

    source_loader = base.make_loader(
        x_source, e_source, t_source, CONFIG["benchmark_batch_size"], shuffle=True, drop_last=True
    )
    target_train_loader = base.make_loader(
        x_train, e_train, t_train, CONFIG["benchmark_batch_size"], shuffle=True, drop_last=False
    )

    input_dim = x_source.shape[1]
    model = TabTransformerAdapterNet(
        input_dim=input_dim,
        d_model=CONFIG["d_model"],
        nhead=CONFIG["num_heads"],
        num_layers=CONFIG["num_transformer_layers"],
        dropout=CONFIG["dropout"],
        adapter_bottleneck=CONFIG["adapter_bottleneck"],
    ).to(CONFIG["device"])

    pre_val, pre_epoch, source_state = run_source_pretrain(model, source_loader, x_val, e_val, t_val)
    zero_shot_test = evaluate_cindex(model, x_test, e_test, t_test, domain="source")

    head_model = TabTransformerAdapterNet(
        input_dim=input_dim,
        d_model=CONFIG["d_model"],
        nhead=CONFIG["num_heads"],
        num_layers=CONFIG["num_transformer_layers"],
        dropout=CONFIG["dropout"],
        adapter_bottleneck=CONFIG["adapter_bottleneck"],
    ).to(CONFIG["device"])
    head_model.load_state_dict(source_state)
    head_val, head_epoch, _ = run_head_only_baseline(head_model, target_train_loader, x_val, e_val, t_val)
    head_test = evaluate_cindex(head_model, x_test, e_test, t_test, domain="target")

    da_model = TabTransformerAdapterNet(
        input_dim=input_dim,
        d_model=CONFIG["d_model"],
        nhead=CONFIG["num_heads"],
        num_layers=CONFIG["num_transformer_layers"],
        dropout=CONFIG["dropout"],
        adapter_bottleneck=CONFIG["adapter_bottleneck"],
    ).to(CONFIG["device"])
    da_model.load_state_dict(source_state)
    phase_results = []
    for phase, max_epochs, patience in [
        ("head_only", CONFIG["head_only_max_epochs"], CONFIG["head_only_patience"]),
        ("partial_unfreeze", CONFIG["partial_unfreeze_max_epochs"], CONFIG["partial_unfreeze_patience"]),
        ("full_finetune", CONFIG["full_finetune_max_epochs"], CONFIG["full_finetune_patience"]),
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
        phase_results.append({"phase": phase, "best_adapt_val_cindex": val_score, "best_epoch": best_epoch})

    da_test = evaluate_cindex(da_model, x_test, e_test, t_test, domain="target")

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

    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"Saved results to {CONFIG['results_path']}")


if __name__ == "__main__":
    main()
