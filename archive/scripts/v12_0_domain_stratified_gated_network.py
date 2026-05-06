"""
V12: Domain-Stratified Gated Survival Network (DS-GSN)
This script introduces three core architectural upgrades:
1. Domain-Specific Baseline Hazard (Stratified Cox)
2. Gated DANN (Feature Masking to prevent negative transfer)
3. Treatment-Conditioned Domain Discriminator (Excluding treatment features from alignment and using them as condition)
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
        "results_path": "runs/v12_0_domain_stratified_gated_network_results.json",
        "seed": 42,
        "benchmark_batch_size": 512,
        "eval_batch_size": 1024,
        "pretrain_max_epochs": 18,
        "pretrain_patience": 4,
        "head_only_max_epochs": 4,
        "head_only_patience": 2,
        "partial_unfreeze_max_epochs": 6,
        "partial_unfreeze_patience": 2,
        "full_finetune_max_epochs": 20,
        "full_finetune_patience": 4,
        
        # Default best from v11.18
        "adv_weight": 0.05,
        "d_model": 64,
        "num_layers": 2,
        "num_heads": 4,
        "dropout": 0.15,
        "domain_hidden": 64,
        "source_replay_weight": 0.1,
    }
)
base.CONFIG.update(CONFIG)
featmod.CONFIG.update(CONFIG)
dannbase.CONFIG.update(CONFIG)

# Define Treatment/Prescription features that should be conditioned on, not aligned
TREATMENT_FEATURES = [
    "超滤比",
    "超滤率_绝对",
    "超滤量MAX",
    "透析液电导率",
    "透析液钙浓度",
    "抗凝剂类型_code",
    "瘘管位置_code",
    "瘘管类型_code",
    "透析方式_code"
]

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, coeff):
        ctx.coeff = coeff
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.coeff, None

def grad_reverse(x, coeff):
    return GradReverse.apply(x, coeff)

class DomainClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):
        return self.net(x)

class DomainStratifiedGatedNet(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout, domain_hidden, treat_indices, physio_indices):
        super().__init__()
        self.treat_indices = treat_indices
        self.physio_indices = physio_indices
        
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
        
        # Hazard prediction head
        self.hazard_head = nn.Linear(d_model, 1)
        
        # Gated DANN: Learnable feature mask
        # Initialized to slightly positive so initial sigmoid is > 0.5
        self.feature_mask = nn.Parameter(torch.ones(d_model) * 0.1)
        
        # Domain discriminator
        # Input: Masked Physiological Rep (d_model) + Raw Treatment Features (len(treat_indices))
        domain_input_dim = d_model + len(treat_indices)
        self.domain_head = DomainClassifier(domain_input_dim, domain_hidden, dropout)

    def forward(self, x, grl_coeff=None):
        # 1. Feature Embedding & TabTransformer
        tokens = x.unsqueeze(-1) * self.feature_weight.unsqueeze(0) + self.feature_bias.unsqueeze(0)
        tokens = tokens + self.feature_pos.unsqueeze(0)
        encoded = self.transformer(tokens)
        
        # 2. Main Survival Prediction (Uses all features)
        emb = self.norm(encoded.mean(dim=1))
        hazard = self.hazard_head(emb)
        
        domain_logits = None
        if grl_coeff is not None:
            # 3. Separate Physiological vs Treatment Representations
            encoded_physio = encoded[:, self.physio_indices, :]
            emb_physio = self.norm(encoded_physio.mean(dim=1))
            
            # 4. Gated Feature Masking
            mask = torch.sigmoid(self.feature_mask)
            masked_physio = emb_physio * mask
            
            # Apply Gradient Reversal
            physio_grl = grad_reverse(masked_physio, grl_coeff)
            
            # 5. Treatment-Conditioned Discriminator
            x_treat = x[:, self.treat_indices]
            domain_input = torch.cat([physio_grl, x_treat], dim=-1)
            
            domain_logits = self.domain_head(domain_input)
            
        return emb, hazard, domain_logits

def domain_loss_from_logits(domain_logits_s, domain_logits_t):
    y_s = torch.zeros(domain_logits_s.size(0), dtype=torch.long, device=domain_logits_s.device)
    y_t = torch.ones(domain_logits_t.size(0), dtype=torch.long, device=domain_logits_t.device)
    logits = torch.cat([domain_logits_s, domain_logits_t], dim=0)
    labels = torch.cat([y_s, y_t], dim=0)
    return nn.functional.cross_entropy(logits, labels)

def evaluate_cindex(model, x_eval, e_eval, t_eval):
    model.eval()
    hazards = []
    with torch.no_grad():
        for start in range(0, len(x_eval), CONFIG["eval_batch_size"]):
            end = start + CONFIG["eval_batch_size"]
            x_batch = torch.tensor(x_eval[start:end], dtype=torch.float32, device=CONFIG["device"])
            _, hazard, _ = model(x_batch, grl_coeff=None)
            hazards.append(hazard.squeeze(-1).detach().cpu().numpy())
    risk_scores = np.concatenate(hazards, axis=0)
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
        model.feature_mask.requires_grad = True
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
            {"params": [model.feature_weight, model.feature_bias, model.feature_pos, model.feature_mask], "lr": CONFIG["full_lr"]},
            {"params": model.norm.parameters(), "lr": CONFIG["full_lr"]},
        ]
        return optim.AdamW(groups, weight_decay=1e-4)
    raise ValueError(phase)

def run_source_pretrain(model, source_loader, x_val, e_val, t_val, lr):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_val, best_epoch = -np.inf, 0
    best_state = copy.deepcopy(model.state_dict())
    patience_left = CONFIG["pretrain_patience"]
    for epoch in range(1, CONFIG["pretrain_max_epochs"] + 1):
        model.train()
        for x_s, e_s, t_s in source_loader:
            x_s = x_s.to(CONFIG["device"])
            e_s = e_s.to(CONFIG["device"])
            t_s = t_s.to(CONFIG["device"])
            optimizer.zero_grad()
            _, hazard, _ = model(x_s, grl_coeff=None)
            loss = base.cox_loss(hazard, e_s, t_s)
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

def run_domain_stratified_da(model, source_loader, target_loader, x_val, e_val, t_val, adv_weight, source_replay_weight):
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
            for x_t, e_t, t_t in target_loader:
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
                
                _, hazard_t, dom_t = model(x_t, grl_coeff=coeff)
                _, hazard_s, dom_s = model(x_s, grl_coeff=coeff)
                
                # 1. Domain-Specific Baseline Hazard (Stratified Cox)
                # By computing target_loss and source_loss separately, we are inherently
                # constructing separate risk sets. Thus, they don't share the baseline hazard.
                target_loss = base.cox_loss(hazard_t, e_t, t_t)
                source_loss = base.cox_loss(hazard_s, e_s, t_s)
                
                adv_loss = domain_loss_from_logits(dom_s, dom_t)
                
                loss = target_loss + source_replay_weight * source_loss + adv_weight * adv_loss
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
    print("Loading feature tables...")
    feat_source, feat_target, e_source, t_source, e_target, t_target, feature_names = featmod.build_feature_tables()
    x_source, x_target = featmod.prepare_group_matrices(feat_source, feat_target)

    # Identify indices for treatments vs physiology
    treat_indices = []
    physio_indices = []
    for i, name in enumerate(feature_names):
        if name in TREATMENT_FEATURES:
            treat_indices.append(i)
        else:
            physio_indices.append(i)
            
    print(f"Total features: {len(feature_names)}")
    print(f"Physiological features for DA: {len(physio_indices)}")
    print(f"Treatment conditions for Domain Discriminator: {len(treat_indices)}")

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

    source_loader = base.make_loader(x_source, e_source, t_source, CONFIG["benchmark_batch_size"], shuffle=True, drop_last=True)
    target_loader = base.make_loader(x_train, e_train, t_train, CONFIG["benchmark_batch_size"], shuffle=True, drop_last=False)

    print("\n=== Running DS-GSN (Domain-Stratified Gated Survival Network) ===")
    model = DomainStratifiedGatedNet(
        input_dim=x_source.shape[1],
        d_model=CONFIG["d_model"],
        nhead=CONFIG["num_heads"],
        num_layers=CONFIG["num_layers"],
        dropout=CONFIG["dropout"],
        domain_hidden=CONFIG["domain_hidden"],
        treat_indices=treat_indices,
        physio_indices=physio_indices
    ).to(CONFIG["device"])
    
    # 1. Source Pretraining
    print("Pretraining on Source...")
    pre_val, pre_epoch, source_state = run_source_pretrain(model, source_loader, x_val, e_val, t_val, lr=CONFIG["lr"])
    zero_shot_test = evaluate_cindex(model, x_test, e_test, t_test)
    print(f"Zero-shot Target Test C-index: {zero_shot_test:.4f}")

    # 2. Domain Adaptation Finetuning
    print("Running Domain-Stratified Gated DA...")
    da_model = DomainStratifiedGatedNet(
        input_dim=x_source.shape[1],
        d_model=CONFIG["d_model"],
        nhead=CONFIG["num_heads"],
        num_layers=CONFIG["num_layers"],
        dropout=CONFIG["dropout"],
        domain_hidden=CONFIG["domain_hidden"],
        treat_indices=treat_indices,
        physio_indices=physio_indices
    ).to(CONFIG["device"])
    da_model.load_state_dict(source_state)
    
    phase_results = run_domain_stratified_da(
        da_model,
        source_loader,
        target_loader,
        x_val,
        e_val,
        t_val,
        adv_weight=CONFIG["adv_weight"],
        source_replay_weight=CONFIG["source_replay_weight"],
    )
    
    da_test = evaluate_cindex(da_model, x_test, e_test, t_test)
    print(f"DS-GSN Target Test C-index: {da_test:.4f}")
    print(f"Gain vs Zero-shot: {da_test - zero_shot_test:.4f}")

    results = {
        "config": CONFIG,
        "feature_split": {
            "treatment_features": [feature_names[i] for i in treat_indices],
            "physio_features": [feature_names[i] for i in physio_indices],
        },
        "source_pretrain": {
            "best_adapt_val_cindex": pre_val,
            "best_epoch": pre_epoch,
            "zero_shot_test_cindex": zero_shot_test,
        },
        "layerwise_da": {
            "phase_results": phase_results,
            "test_cindex": da_test,
            "gain_vs_zero_shot": da_test - zero_shot_test,
        },
    }

    with open(CONFIG["results_path"], "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved results to {CONFIG['results_path']}")

if __name__ == "__main__":
    main()
