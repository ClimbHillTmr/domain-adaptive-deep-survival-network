"""
v11.7 direction-one DA with v11.6 features and a residual MLP encoder.
"""

import copy
import json

import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

import v11_5_feature_expansion_da as base
import v11_6_trend_context_da as featmod


CONFIG = dict(base.CONFIG)
CONFIG.update(
    {
        "results_path": "runs/v11_7_resmlp_da_results.json",
        "seed": 42,
        "d_model": 96,
        "num_blocks": 3,
        "dropout": 0.15,
        "lr": 8e-4,
        "head_lr": 1e-4,
        "top_encoder_lr": 7e-5,
        "full_lr": 3e-5,
    }
)
base.CONFIG.update(CONFIG)
featmod.CONFIG.update(CONFIG)


class ResidualBlock(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model)
        self.bn1 = nn.BatchNorm1d(d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.bn2 = nn.BatchNorm1d(d_model)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = out + residual
        out = self.act(out)
        out = self.dropout(out)
        return out


class ResMLPSurvivalNet(nn.Module):
    def __init__(self, input_dim, d_model, num_blocks, dropout):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList(
            [ResidualBlock(d_model, dropout) for _ in range(num_blocks)]
        )
        self.hazard_head = nn.Linear(d_model, 1)

    def forward(self, x):
        emb = self.input_proj(x)
        for block in self.blocks:
            emb = block(emb)
        hazard = self.hazard_head(emb)
        return emb, hazard


def evaluate_cindex(model, x_eval, e_eval, t_eval):
    return base.evaluate_cindex(model, x_eval, e_eval, t_eval)


def set_trainable_state(model, phase):
    for param in model.parameters():
        param.requires_grad = False

    for param in model.hazard_head.parameters():
        param.requires_grad = True

    if phase in {"partial_unfreeze", "full_finetune"}:
        for param in model.blocks[-1].parameters():
            param.requires_grad = True

    if phase == "full_finetune":
        for param in model.input_proj.parameters():
            param.requires_grad = True
        for block in model.blocks:
            for param in block.parameters():
                param.requires_grad = True


def build_optimizer(model, phase):
    if phase == "head_only":
        return optim.AdamW(
            model.hazard_head.parameters(), lr=CONFIG["head_lr"], weight_decay=1e-3
        )

    if phase == "partial_unfreeze":
        param_groups = [
            {"params": model.hazard_head.parameters(), "lr": CONFIG["head_lr"]},
            {"params": model.blocks[-1].parameters(), "lr": CONFIG["top_encoder_lr"]},
        ]
        return optim.AdamW(param_groups, weight_decay=1e-3)

    if phase == "full_finetune":
        param_groups = [
            {"params": model.hazard_head.parameters(), "lr": CONFIG["head_lr"]},
            {"params": model.input_proj.parameters(), "lr": CONFIG["full_lr"]},
            {"params": model.blocks.parameters(), "lr": CONFIG["full_lr"]},
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
            _, hazard = model(x_s)
            loss = base.cox_loss(hazard, e_s, t_s)
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
    set_trainable_state(model, "head_only")
    optimizer = build_optimizer(model, "head_only")
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
            _, hazard = model(x_t)
            loss = base.cox_loss(hazard, e_t, t_t)
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
            emb_t, hazard_t = model(x_t)
            emb_s, hazard_s = model(x_s)
            target_loss = base.cox_loss(hazard_t, e_t, t_t)
            replay_loss = base.cox_loss(hazard_s, e_s, t_s)
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
    print("Loading v11.7 feature tables from v11.6...")
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
    print(f"Feature count: {input_dim}")

    base_model = ResMLPSurvivalNet(
        input_dim,
        CONFIG["d_model"],
        CONFIG["num_blocks"],
        CONFIG["dropout"],
    ).to(CONFIG["device"])
    pre_val, pre_epoch, source_state = run_source_pretrain(
        base_model, source_loader, x_val, e_val, t_val
    )
    zero_shot_test = evaluate_cindex(base_model, x_test, e_test, t_test)

    head_model = ResMLPSurvivalNet(
        input_dim,
        CONFIG["d_model"],
        CONFIG["num_blocks"],
        CONFIG["dropout"],
    ).to(CONFIG["device"])
    head_model.load_state_dict(source_state)
    head_val, head_epoch, _ = run_head_only_baseline(
        head_model, target_train_loader, x_val, e_val, t_val
    )
    head_test = evaluate_cindex(head_model, x_test, e_test, t_test)

    da_model = ResMLPSurvivalNet(
        input_dim,
        CONFIG["d_model"],
        CONFIG["num_blocks"],
        CONFIG["dropout"],
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
