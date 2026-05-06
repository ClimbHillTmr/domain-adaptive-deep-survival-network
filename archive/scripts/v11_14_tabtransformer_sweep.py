"""
v11.14 sweep a few tabtransformer configurations on top of v11.6 features.
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


BASE_CONFIG = dict(base.CONFIG)
BASE_CONFIG.update(
    {
        "results_path": "runs/v11_14_tabtransformer_sweep_results.json",
        "seed": 42,
        "target_adapt_ratio": 0.2,
        "target_val_ratio_within_adapt": 0.5,
        "source_replay_weight": 0.2,
        "conditional_align_weight": 0.1,
        "head_only_max_epochs": 3,
        "head_only_patience": 1,
        "partial_unfreeze_max_epochs": 5,
        "partial_unfreeze_patience": 2,
        "full_finetune_max_epochs": 12,
        "full_finetune_patience": 3,
        "pretrain_max_epochs": 24,
        "pretrain_patience": 3,
        "eval_batch_size": 1024,
    }
)
base.CONFIG.update(BASE_CONFIG)
featmod.CONFIG.update(BASE_CONFIG)


SWEEP_CONFIGS = [
    {
        "name": "tt_small_deep",
        "d_model": 64,
        "num_layers": 3,
        "nhead": 4,
        "dropout": 0.10,
        "lr": 7e-4,
        "head_lr": 1e-4,
        "top_encoder_lr": 7e-5,
        "full_lr": 3e-5,
        "batch_size": 384,
    },
    {
        "name": "tt_mid_wide",
        "d_model": 80,
        "num_layers": 2,
        "nhead": 4,
        "dropout": 0.10,
        "lr": 7e-4,
        "head_lr": 1e-4,
        "top_encoder_lr": 6e-5,
        "full_lr": 2.5e-5,
        "batch_size": 320,
    },
    {
        "name": "tt_mid_deep",
        "d_model": 80,
        "num_layers": 3,
        "nhead": 4,
        "dropout": 0.12,
        "lr": 6e-4,
        "head_lr": 1e-4,
        "top_encoder_lr": 6e-5,
        "full_lr": 2e-5,
        "batch_size": 256,
    },
]


class TabTransformerEncoderNet(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout):
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
        self.hazard_head = nn.Linear(d_model, 1)

    def forward(self, x):
        tokens = x.unsqueeze(-1) * self.feature_weight.unsqueeze(0) + self.feature_bias.unsqueeze(0)
        tokens = tokens + self.feature_pos.unsqueeze(0)
        encoded = self.transformer(tokens)
        emb = self.norm(encoded.mean(dim=1))
        hazard = self.hazard_head(emb)
        return emb, hazard


def evaluate_cindex(model, x_eval, e_eval, t_eval, eval_batch_size):
    model.eval()
    hazards = []
    with torch.no_grad():
        for start in range(0, len(x_eval), eval_batch_size):
            end = start + eval_batch_size
            x_batch = torch.tensor(x_eval[start:end], dtype=torch.float32, device=BASE_CONFIG["device"])
            _, hazard = model(x_batch)
            hazards.append(hazard.squeeze(-1).detach().cpu().numpy())
    risk_scores = np.concatenate(hazards, axis=0)
    return concordance_index(t_eval, -risk_scores, e_eval)


def set_trainable_state(model, phase):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.hazard_head.parameters():
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


def build_optimizer(model, phase, cfg):
    if phase == "head_only":
        return optim.AdamW(model.hazard_head.parameters(), lr=cfg["head_lr"], weight_decay=1e-3)
    if phase == "partial_unfreeze":
        param_groups = [
            {"params": model.hazard_head.parameters(), "lr": cfg["head_lr"]},
            {"params": model.transformer.layers[-1].parameters(), "lr": cfg["top_encoder_lr"]},
        ]
        return optim.AdamW(param_groups, weight_decay=1e-3)
    if phase == "full_finetune":
        param_groups = [
            {"params": model.hazard_head.parameters(), "lr": cfg["head_lr"]},
            {"params": model.transformer.parameters(), "lr": cfg["full_lr"]},
            {"params": [model.feature_weight, model.feature_bias, model.feature_pos], "lr": cfg["full_lr"]},
            {"params": model.norm.parameters(), "lr": cfg["full_lr"]},
        ]
        return optim.AdamW(param_groups, weight_decay=1e-4)
    raise ValueError(phase)


def run_source_pretrain(model, source_loader, x_val, e_val, t_val, cfg):
    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    best_val = -np.inf
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    patience_left = BASE_CONFIG["pretrain_patience"]
    for epoch in range(1, BASE_CONFIG["pretrain_max_epochs"] + 1):
        model.train()
        for x_s, e_s, t_s in source_loader:
            x_s = x_s.to(BASE_CONFIG["device"])
            e_s = e_s.to(BASE_CONFIG["device"])
            t_s = t_s.to(BASE_CONFIG["device"])
            optimizer.zero_grad()
            _, hazard = model(x_s)
            loss = base.cox_loss(hazard, e_s, t_s)
            loss.backward()
            optimizer.step()
        val_cindex = evaluate_cindex(model, x_val, e_val, t_val, BASE_CONFIG["eval_batch_size"])
        if val_cindex > best_val:
            best_val = val_cindex
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_left = BASE_CONFIG["pretrain_patience"]
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    model.load_state_dict(best_state)
    return best_val, best_epoch, best_state


def run_head_only(model, target_loader, x_val, e_val, t_val, cfg):
    set_trainable_state(model, "head_only")
    optimizer = build_optimizer(model, "head_only", cfg)
    best_val = evaluate_cindex(model, x_val, e_val, t_val, BASE_CONFIG["eval_batch_size"])
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    patience_left = BASE_CONFIG["head_only_patience"]
    for epoch in range(1, BASE_CONFIG["head_only_max_epochs"] + 1):
        model.train()
        for x_t, e_t, t_t in target_loader:
            x_t = x_t.to(BASE_CONFIG["device"])
            e_t = e_t.to(BASE_CONFIG["device"])
            t_t = t_t.to(BASE_CONFIG["device"])
            optimizer.zero_grad()
            _, hazard = model(x_t)
            loss = base.cox_loss(hazard, e_t, t_t)
            loss.backward()
            optimizer.step()
        val_cindex = evaluate_cindex(model, x_val, e_val, t_val, BASE_CONFIG["eval_batch_size"])
        if val_cindex > best_val:
            best_val = val_cindex
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_left = BASE_CONFIG["head_only_patience"]
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    model.load_state_dict(best_state)
    return best_val, best_epoch, best_state


def run_da(model, source_loader, target_loader, x_val, e_val, t_val, cfg):
    phase_results = []
    best_overall_state = copy.deepcopy(model.state_dict())
    for phase, max_epochs, patience in [
        ("head_only", BASE_CONFIG["head_only_max_epochs"], BASE_CONFIG["head_only_patience"]),
        ("partial_unfreeze", BASE_CONFIG["partial_unfreeze_max_epochs"], BASE_CONFIG["partial_unfreeze_patience"]),
        ("full_finetune", BASE_CONFIG["full_finetune_max_epochs"], BASE_CONFIG["full_finetune_patience"]),
    ]:
        set_trainable_state(model, phase)
        optimizer = build_optimizer(model, phase, cfg)
        best_val = evaluate_cindex(model, x_val, e_val, t_val, BASE_CONFIG["eval_batch_size"])
        best_epoch = 0
        best_state = copy.deepcopy(model.state_dict())
        patience_left = patience
        for epoch in range(1, max_epochs + 1):
            model.train()
            source_iter = iter(source_loader)
            for x_t, e_t, t_t in target_loader:
                try:
                    x_s, e_s, t_s = next(source_iter)
                except StopIteration:
                    source_iter = iter(source_loader)
                    x_s, e_s, t_s = next(source_iter)
                x_t = x_t.to(BASE_CONFIG["device"])
                e_t = e_t.to(BASE_CONFIG["device"])
                t_t = t_t.to(BASE_CONFIG["device"])
                x_s = x_s.to(BASE_CONFIG["device"])
                e_s = e_s.to(BASE_CONFIG["device"])
                t_s = t_s.to(BASE_CONFIG["device"])
                optimizer.zero_grad()
                emb_t, hazard_t = model(x_t)
                emb_s, hazard_s = model(x_s)
                target_loss = base.cox_loss(hazard_t, e_t, t_t)
                replay_loss = base.cox_loss(hazard_s, e_s, t_s)
                align_loss = base.conditional_alignment_loss(emb_s, e_s, emb_t, e_t)
                loss = target_loss + BASE_CONFIG["source_replay_weight"] * replay_loss + BASE_CONFIG["conditional_align_weight"] * align_loss
                loss.backward()
                optimizer.step()
            val_cindex = evaluate_cindex(model, x_val, e_val, t_val, BASE_CONFIG["eval_batch_size"])
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
        best_overall_state = copy.deepcopy(best_state)
        phase_results.append({"phase": phase, "best_adapt_val_cindex": best_val, "best_epoch": best_epoch})
    model.load_state_dict(best_overall_state)
    return phase_results


def main():
    base.set_seed(BASE_CONFIG["seed"])
    feat_source, feat_target, e_source, t_source, e_target, t_target, feature_names = featmod.build_feature_tables()
    x_source, x_target = featmod.prepare_group_matrices(feat_source, feat_target)
    x_adapt_pool, x_test, e_adapt_pool, e_test, t_adapt_pool, t_test = train_test_split(
        x_target, e_target, t_target,
        test_size=1 - BASE_CONFIG["target_adapt_ratio"],
        random_state=BASE_CONFIG["seed"],
        stratify=e_target,
    )
    x_train, x_val, e_train, e_val, t_train, t_val = train_test_split(
        x_adapt_pool, e_adapt_pool, t_adapt_pool,
        test_size=BASE_CONFIG["target_val_ratio_within_adapt"],
        random_state=BASE_CONFIG["seed"],
        stratify=e_adapt_pool,
    )

    results = {"feature_count": int(x_source.shape[1]), "feature_names": feature_names, "models": {}}
    for cfg in SWEEP_CONFIGS:
        print(f"\n=== Running {cfg['name']} ===")
        source_loader = base.make_loader(x_source, e_source, t_source, cfg["batch_size"], shuffle=True, drop_last=True)
        target_loader = base.make_loader(x_train, e_train, t_train, cfg["batch_size"], shuffle=True, drop_last=False)
        model = TabTransformerEncoderNet(
            input_dim=x_source.shape[1],
            d_model=cfg["d_model"],
            nhead=cfg["nhead"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
        ).to(BASE_CONFIG["device"])
        pre_val, pre_epoch, source_state = run_source_pretrain(model, source_loader, x_val, e_val, t_val, cfg)
        zero_test = evaluate_cindex(model, x_test, e_test, t_test, BASE_CONFIG["eval_batch_size"])

        head_model = TabTransformerEncoderNet(
            input_dim=x_source.shape[1],
            d_model=cfg["d_model"],
            nhead=cfg["nhead"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
        ).to(BASE_CONFIG["device"])
        head_model.load_state_dict(source_state)
        head_val, head_epoch, _ = run_head_only(head_model, target_loader, x_val, e_val, t_val, cfg)
        head_test = evaluate_cindex(head_model, x_test, e_test, t_test, BASE_CONFIG["eval_batch_size"])

        da_model = TabTransformerEncoderNet(
            input_dim=x_source.shape[1],
            d_model=cfg["d_model"],
            nhead=cfg["nhead"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
        ).to(BASE_CONFIG["device"])
        da_model.load_state_dict(source_state)
        phase_results = run_da(da_model, source_loader, target_loader, x_val, e_val, t_val, cfg)
        da_test = evaluate_cindex(da_model, x_test, e_test, t_test, BASE_CONFIG["eval_batch_size"])
        results["models"][cfg["name"]] = {
            "config": cfg,
            "source_pretrain": {"best_adapt_val_cindex": pre_val, "best_epoch": pre_epoch, "zero_shot_test_cindex": zero_test},
            "head_only": {"best_adapt_val_cindex": head_val, "best_epoch": head_epoch, "test_cindex": head_test},
            "layerwise_da": {"phase_results": phase_results, "test_cindex": da_test, "gain_vs_zero_shot": da_test - zero_test},
        }
        print(f"{cfg['name']}: zero={zero_test:.4f}, da={da_test:.4f}, gain={da_test-zero_test:.4f}")

    with open(BASE_CONFIG["results_path"], "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved results to {BASE_CONFIG['results_path']}")


if __name__ == "__main__":
    main()
