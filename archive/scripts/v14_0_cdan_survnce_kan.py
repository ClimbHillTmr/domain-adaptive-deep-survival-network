"""
V14.0: Risk-Conditioned DANN (CDAN) + Differentiable Pairwise Concordance Loss
Upgrades:
1. CDAN (Conditional Domain Adversarial Network): Concatenating predicted hazard to Domain Discriminator.
2. SurvNCE (Pairwise Margin Loss): Differentiable pairwise concordance ranking loss to improve margin between risks.
"""

import copy
import json
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

import v11_5_feature_expansion_da as base
import v11_15_rich_deterioration_da as featmod
import v11_17_da_upgrade_benchmark as dannbase

CONFIG = dict(base.CONFIG)
CONFIG.update(
    {
        "results_path": "runs/v14_0_cdan_survnce_kan_results.json",
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

LOCAL_SWEEPS = [
    {"name": "kan_cdan_margin", "tokenizer_type": "kan", "kan_basis_dim": 8, "adv_weight": 0.05, "mask_l1_weight": 0.05, "margin_weight": 0.1},
    {"name": "kan_cdan_nomargin", "tokenizer_type": "kan", "kan_basis_dim": 8, "adv_weight": 0.05, "mask_l1_weight": 0.05, "margin_weight": 0.0},
]

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

def compute_ipcw_weights(t_fit, e_fit, t_eval=None):
    kmf = KaplanMeierFitter()
    kmf.fit(t_fit, 1 - e_fit)
    
    if t_eval is None:
        t_eval = t_fit
        
    g_t = kmf.survival_function_at_times(t_eval).values
    g_t = np.clip(g_t, 1e-5, 1.0)
    w = 1.0 / g_t
    
    w_95 = np.percentile(w, 95)
    w = np.clip(w, 0, w_95)
    return w.astype(np.float32)

def weighted_cox_loss(hazard, e, t, w):
    hazard = hazard.squeeze(-1)
    idx = torch.argsort(t, descending=True)
    hazard = hazard[idx]
    e = e[idx]
    t = t[idx]
    w = w[idx]
    
    risk_score = torch.exp(hazard)
    weighted_risk = risk_score * w
    
    risk_set_sum = torch.cumsum(weighted_risk, dim=0) + 1e-7
    loss = e * w * (hazard - torch.log(risk_set_sum))
    
    event_weight_sum = torch.sum(e * w) + 1e-7
    return -torch.sum(loss) / event_weight_sum

def pairwise_margin_loss(hazard, e, t, margin=1.0):
    """
    Differentiable pairwise concordance loss (SurvNCE).
    Pushes hazard_i > hazard_j + margin when t_i < t_j and e_i == 1.
    """
    hazard = hazard.squeeze(-1)
    
    t_diff = t.unsqueeze(1) - t.unsqueeze(0)  # t_j - t_i
    h_diff = hazard.unsqueeze(0) - hazard.unsqueeze(1)  # hazard_j - hazard_i
    
    # Valid pairs: e_i == 1 and t_i < t_j
    valid_mask = (t_diff > 0) & (e.unsqueeze(1) == 1)
    
    # We want hazard_i > hazard_j, meaning h_diff < 0
    # Loss = max(0, margin + hazard_j - hazard_i)
    loss = torch.relu(margin + h_diff)
    
    valid_loss = loss[valid_mask]
    if valid_loss.numel() > 0:
        return valid_loss.mean()
    return torch.tensor(0.0, device=hazard.device)

def make_weighted_loader(x, e, t, w, batch_size, shuffle=True, drop_last=False):
    dataset = TensorDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(e, dtype=torch.float32),
        torch.tensor(t, dtype=torch.float32),
        torch.tensor(w, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

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

class KANTokenizer(nn.Module):
    def __init__(self, num_features, d_model, basis_dim=8):
        super().__init__()
        self.centers = nn.Parameter(torch.linspace(-3.0, 3.0, basis_dim).view(1, 1, basis_dim), requires_grad=False)
        self.log_width = nn.Parameter(torch.zeros(num_features, basis_dim))
        self.basis_weight = nn.Parameter(torch.randn(num_features, basis_dim, d_model) * 0.02)
        self.base_weight = nn.Parameter(torch.randn(num_features, d_model) * 0.02)
        self.bias = nn.Parameter(torch.zeros(num_features, d_model))

    def forward(self, x):
        diff = x.unsqueeze(-1) - self.centers
        width = torch.exp(self.log_width).unsqueeze(0) + 1e-4
        basis = torch.exp(-0.5 * (diff / width) ** 2)
        spline_term = torch.einsum('bfk,fkd->bfd', basis, self.basis_weight)
        base_term = x.unsqueeze(-1) * self.base_weight.unsqueeze(0)
        return base_term + spline_term + self.bias.unsqueeze(0)

class DomainStratifiedGatedNet(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout, domain_hidden, treat_indices, physio_indices, tokenizer_type="linear", kan_basis_dim=8):
        super().__init__()
        self.treat_indices = treat_indices
        self.physio_indices = physio_indices
        self.tokenizer_type = tokenizer_type
        
        if tokenizer_type == "kan":
            self.tokenizer = KANTokenizer(input_dim, d_model, kan_basis_dim)
        else:
            self.feature_weight = nn.Parameter(torch.randn(input_dim, d_model) * 0.02)
            self.feature_bias = nn.Parameter(torch.zeros(input_dim, d_model))
            
        self.feature_pos = nn.Parameter(torch.randn(input_dim, d_model) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
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
        self.feature_mask = nn.Parameter(torch.ones(d_model) * 0.1)
        
        # CDAN: physio_grl (d_model) + x_treat (len(treat_indices)) + hazard (1)
        domain_input_dim = d_model + len(treat_indices) + 1
        self.domain_head = DomainClassifier(domain_input_dim, domain_hidden, dropout)

    def forward(self, x, grl_coeff=None):
        if self.tokenizer_type == "kan":
            tokens = self.tokenizer(x)
        else:
            tokens = x.unsqueeze(-1) * self.feature_weight.unsqueeze(0) + self.feature_bias.unsqueeze(0)
            
        tokens = tokens + self.feature_pos.unsqueeze(0)
        
        b_size = x.size(0)
        cls_tokens = self.cls_token.expand(b_size, -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)
        
        encoded = self.transformer(tokens)
        
        cls_emb = self.norm(encoded[:, 0, :])
        hazard = self.hazard_head(cls_emb)
        
        domain_logits = None
        mask_l1_loss = torch.tensor(0.0, device=x.device)
        
        if grl_coeff is not None:
            physio_idx = [i + 1 for i in self.physio_indices]
            encoded_physio = encoded[:, physio_idx, :]
            
            emb_physio = self.norm(encoded_physio.mean(dim=1))
            
            mask = torch.sigmoid(self.feature_mask)
            mask_l1_loss = torch.mean(mask)
            
            masked_physio = emb_physio * mask
            physio_grl = grad_reverse(masked_physio, grl_coeff)
            
            x_treat = x[:, self.treat_indices]
            
            # CDAN (Risk-Conditioned): use sigmoid to map hazard to 0-1 for stability
            hazard_prob = torch.sigmoid(hazard.detach())
            
            domain_input = torch.cat([physio_grl, x_treat, hazard_prob], dim=-1)
            domain_logits = self.domain_head(domain_input)
            
        return cls_emb, hazard, domain_logits, mask_l1_loss

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
            _, hazard, _, _ = model(x_batch, grl_coeff=None)
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
        if model.tokenizer_type == "kan":
            for param in model.tokenizer.parameters():
                param.requires_grad = True
        else:
            model.feature_weight.requires_grad = True
            model.feature_bias.requires_grad = True
        model.feature_pos.requires_grad = True
        model.cls_token.requires_grad = True
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
        params_to_update = [
            {"params": model.hazard_head.parameters(), "lr": CONFIG["head_lr"]},
            {"params": model.domain_head.parameters(), "lr": CONFIG["head_lr"]},
            {"params": model.transformer.parameters(), "lr": CONFIG["full_lr"]},
            {"params": model.norm.parameters(), "lr": CONFIG["full_lr"]},
        ]
        base_params = [model.feature_pos, model.cls_token, model.feature_mask]
        if model.tokenizer_type == "kan":
            base_params.extend(list(model.tokenizer.parameters()))
        else:
            base_params.extend([model.feature_weight, model.feature_bias])
        params_to_update.append({"params": base_params, "lr": CONFIG["full_lr"]})
        
        return optim.AdamW(params_to_update, weight_decay=1e-4)
    raise ValueError(phase)

def run_source_pretrain(model, source_loader, x_val, e_val, t_val, lr, margin_weight):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_val, best_epoch = -np.inf, 0
    best_state = copy.deepcopy(model.state_dict())
    patience_left = CONFIG["pretrain_patience"]
    for epoch in range(1, CONFIG["pretrain_max_epochs"] + 1):
        model.train()
        for x_s, e_s, t_s, w_s in source_loader:
            x_s = x_s.to(CONFIG["device"])
            e_s = e_s.to(CONFIG["device"])
            t_s = t_s.to(CONFIG["device"])
            w_s = w_s.to(CONFIG["device"])
            
            optimizer.zero_grad()
            _, hazard, _, _ = model(x_s, grl_coeff=None)
            
            cox_loss_val = weighted_cox_loss(hazard, e_s, t_s, w_s)
            margin_loss_val = pairwise_margin_loss(hazard, e_s, t_s)
            
            loss = cox_loss_val + margin_weight * margin_loss_val
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

def run_domain_stratified_da(model, source_loader, target_loader, x_val, e_val, t_val, adv_weight, source_replay_weight, mask_l1_weight, margin_weight):
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
            for x_t, e_t, t_t, w_t in target_loader:
                try:
                    x_s, e_s, t_s, w_s = next(source_iter)
                except StopIteration:
                    source_iter = iter(source_loader)
                    x_s, e_s, t_s, w_s = next(source_iter)
                
                x_t = x_t.to(CONFIG["device"])
                e_t = e_t.to(CONFIG["device"])
                t_t = t_t.to(CONFIG["device"])
                w_t = w_t.to(CONFIG["device"])
                
                x_s = x_s.to(CONFIG["device"])
                e_s = e_s.to(CONFIG["device"])
                t_s = t_s.to(CONFIG["device"])
                w_s = w_s.to(CONFIG["device"])
                
                optimizer.zero_grad()
                
                _, hazard_t, dom_t, mask_l1_t = model(x_t, grl_coeff=coeff)
                _, hazard_s, dom_s, mask_l1_s = model(x_s, grl_coeff=coeff)
                
                target_cox = weighted_cox_loss(hazard_t, e_t, t_t, w_t)
                source_cox = weighted_cox_loss(hazard_s, e_s, t_s, w_s)
                
                target_margin = pairwise_margin_loss(hazard_t, e_t, t_t)
                source_margin = pairwise_margin_loss(hazard_s, e_s, t_s)
                
                adv_loss = domain_loss_from_logits(dom_s, dom_t)
                l1_loss = (mask_l1_t + mask_l1_s) / 2.0
                
                target_loss = target_cox + margin_weight * target_margin
                source_loss = source_cox + margin_weight * source_margin
                
                loss = target_loss + source_replay_weight * source_loss + adv_weight * adv_loss + mask_l1_weight * l1_loss
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

    treat_indices = []
    physio_indices = []
    for i, name in enumerate(feature_names):
        if name in TREATMENT_FEATURES:
            treat_indices.append(i)
        else:
            physio_indices.append(i)

    w_source = compute_ipcw_weights(t_source, e_source)

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

    w_train = compute_ipcw_weights(t_train, e_train)

    source_loader = make_weighted_loader(x_source, e_source, t_source, w_source, CONFIG["benchmark_batch_size"], shuffle=True, drop_last=True)
    target_loader = make_weighted_loader(x_train, e_train, t_train, w_train, CONFIG["benchmark_batch_size"], shuffle=True, drop_last=False)

    results = {"feature_count": len(feature_names), "sweeps": {}}

    for sweep in LOCAL_SWEEPS:
        print(f"\n=== Running Sweep: {sweep['name']} ===")
        model = DomainStratifiedGatedNet(
            input_dim=x_source.shape[1],
            d_model=CONFIG["d_model"],
            nhead=CONFIG["num_heads"],
            num_layers=CONFIG["num_layers"],
            dropout=CONFIG["dropout"],
            domain_hidden=CONFIG["domain_hidden"],
            treat_indices=treat_indices,
            physio_indices=physio_indices,
            tokenizer_type=sweep["tokenizer_type"],
            kan_basis_dim=sweep["kan_basis_dim"]
        ).to(CONFIG["device"])
        
        pre_val, pre_epoch, source_state = run_source_pretrain(
            model, source_loader, x_val, e_val, t_val, 
            lr=CONFIG["lr"], margin_weight=sweep["margin_weight"]
        )
        zero_shot_test = evaluate_cindex(model, x_test, e_test, t_test)
        
        da_model = DomainStratifiedGatedNet(
            input_dim=x_source.shape[1],
            d_model=CONFIG["d_model"],
            nhead=CONFIG["num_heads"],
            num_layers=CONFIG["num_layers"],
            dropout=CONFIG["dropout"],
            domain_hidden=CONFIG["domain_hidden"],
            treat_indices=treat_indices,
            physio_indices=physio_indices,
            tokenizer_type=sweep["tokenizer_type"],
            kan_basis_dim=sweep["kan_basis_dim"]
        ).to(CONFIG["device"])
        da_model.load_state_dict(source_state)
        
        phase_results = run_domain_stratified_da(
            da_model,
            source_loader,
            target_loader,
            x_val,
            e_val,
            t_val,
            adv_weight=sweep["adv_weight"],
            source_replay_weight=CONFIG["source_replay_weight"],
            mask_l1_weight=sweep["mask_l1_weight"],
            margin_weight=sweep["margin_weight"]
        )
        
        da_test = evaluate_cindex(da_model, x_test, e_test, t_test)
        gain = da_test - zero_shot_test
        print(f"Zero-shot: {zero_shot_test:.4f} | DA: {da_test:.4f} | Gain: {gain:.4f}")
        
        results["sweeps"][sweep["name"]] = {
            "config": sweep,
            "zero_shot_test_cindex": zero_shot_test,
            "da_test_cindex": da_test,
            "gain": gain,
            "phase_results": phase_results
        }

    with open(CONFIG["results_path"], "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved results to {CONFIG['results_path']}")

if __name__ == "__main__":
    main()
