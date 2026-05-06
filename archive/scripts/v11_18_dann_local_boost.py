"""
v11.18 local boost around the current best DANN replacement setup.
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
        "results_path": "runs/v11_18_dann_local_boost_results.json",
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
    }
)
base.CONFIG.update(CONFIG)
featmod.CONFIG.update(CONFIG)
dannbase.CONFIG.update(CONFIG)


LOCAL_SWEEPS = [
    {
        "name": "base_0.05_long",
        "adv_weight": 0.05,
        "d_model": 64,
        "num_layers": 2,
        "num_heads": 4,
        "dropout": 0.15,
        "domain_hidden": 64,
        "source_replay_weight": 0.2,
    },
    {
        "name": "adv_0.04_long",
        "adv_weight": 0.04,
        "d_model": 64,
        "num_layers": 2,
        "num_heads": 4,
        "dropout": 0.15,
        "domain_hidden": 64,
        "source_replay_weight": 0.2,
    },
    {
        "name": "adv_0.06_long",
        "adv_weight": 0.06,
        "d_model": 64,
        "num_layers": 2,
        "num_heads": 4,
        "dropout": 0.15,
        "domain_hidden": 64,
        "source_replay_weight": 0.2,
    },
    {
        "name": "adv_0.05_sr01",
        "adv_weight": 0.05,
        "d_model": 64,
        "num_layers": 2,
        "num_heads": 4,
        "dropout": 0.15,
        "domain_hidden": 64,
        "source_replay_weight": 0.1,
    },
    {
        "name": "adv_0.05_deeper",
        "adv_weight": 0.05,
        "d_model": 80,
        "num_layers": 3,
        "num_heads": 4,
        "dropout": 0.12,
        "domain_hidden": 80,
        "source_replay_weight": 0.2,
    },
]


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


def run_head_only(model, target_loader, x_val, e_val, t_val):
    dannbase.set_trainable_state(model, "head_only")
    optimizer = dannbase.build_optimizer(model, "head_only")
    best_val, best_epoch = evaluate_cindex(model, x_val, e_val, t_val), 0
    best_state = copy.deepcopy(model.state_dict())
    patience_left = CONFIG["head_only_patience"]
    for epoch in range(1, CONFIG["head_only_max_epochs"] + 1):
        model.train()
        for x_t, e_t, t_t in target_loader:
            x_t = x_t.to(CONFIG["device"])
            e_t = e_t.to(CONFIG["device"])
            t_t = t_t.to(CONFIG["device"])
            optimizer.zero_grad()
            _, hazard, _ = model(x_t, grl_coeff=None)
            loss = base.cox_loss(hazard, e_t, t_t)
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


def run_dann_replace(model, source_loader, target_loader, x_val, e_val, t_val, adv_weight, source_replay_weight):
    phase_results = []
    for phase, max_epochs, patience in [
        ("head_only", CONFIG["head_only_max_epochs"], CONFIG["head_only_patience"]),
        ("partial_unfreeze", CONFIG["partial_unfreeze_max_epochs"], CONFIG["partial_unfreeze_patience"]),
        ("full_finetune", CONFIG["full_finetune_max_epochs"], CONFIG["full_finetune_patience"]),
    ]:
        dannbase.set_trainable_state(model, phase)
        optimizer = dannbase.build_optimizer(model, phase)
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
                target_loss = base.cox_loss(hazard_t, e_t, t_t)
                source_loss = base.cox_loss(hazard_s, e_s, t_s)
                adv_loss = dannbase.domain_loss_from_logits(dom_s, dom_t)
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

    source_loader = base.make_loader(x_source, e_source, t_source, CONFIG["benchmark_batch_size"], shuffle=True, drop_last=True)
    target_loader = base.make_loader(x_train, e_train, t_train, CONFIG["benchmark_batch_size"], shuffle=True, drop_last=False)

    results = {"feature_count": int(x_source.shape[1]), "feature_names": feature_names, "models": {}}

    for sweep in LOCAL_SWEEPS:
        print(f"\n=== Running {sweep['name']} ===")
        local_cfg = dict(CONFIG)
        local_cfg.update(
            {
                "d_model": sweep["d_model"],
                "num_transformer_layers": sweep["num_layers"],
                "num_heads": sweep["num_heads"],
                "dropout": sweep["dropout"],
                "domain_hidden": sweep["domain_hidden"],
                "source_replay_weight": sweep["source_replay_weight"],
            }
        )
        base.CONFIG.update(local_cfg)
        featmod.CONFIG.update(local_cfg)
        dannbase.CONFIG.update(local_cfg)
        model = dannbase.TabTransformerDAAdvNet(
            input_dim=x_source.shape[1],
            d_model=sweep["d_model"],
            nhead=sweep["num_heads"],
            num_layers=sweep["num_layers"],
            dropout=sweep["dropout"],
            domain_hidden=sweep["domain_hidden"],
        ).to(CONFIG["device"])
        pre_val, pre_epoch, source_state = run_source_pretrain(model, source_loader, x_val, e_val, t_val, lr=local_cfg["lr"])
        zero_shot_test = evaluate_cindex(model, x_test, e_test, t_test)

        head_model = dannbase.TabTransformerDAAdvNet(
            input_dim=x_source.shape[1],
            d_model=sweep["d_model"],
            nhead=sweep["num_heads"],
            num_layers=sweep["num_layers"],
            dropout=sweep["dropout"],
            domain_hidden=sweep["domain_hidden"],
        ).to(CONFIG["device"])
        head_model.load_state_dict(source_state)
        head_val, head_epoch, _ = run_head_only(head_model, target_loader, x_val, e_val, t_val)
        head_test = evaluate_cindex(head_model, x_test, e_test, t_test)

        da_model = dannbase.TabTransformerDAAdvNet(
            input_dim=x_source.shape[1],
            d_model=sweep["d_model"],
            nhead=sweep["num_heads"],
            num_layers=sweep["num_layers"],
            dropout=sweep["dropout"],
            domain_hidden=sweep["domain_hidden"],
        ).to(CONFIG["device"])
        da_model.load_state_dict(source_state)
        phase_results = run_dann_replace(
            da_model,
            source_loader,
            target_loader,
            x_val,
            e_val,
            t_val,
            adv_weight=sweep["adv_weight"],
            source_replay_weight=sweep["source_replay_weight"],
        )
        da_test = evaluate_cindex(da_model, x_test, e_test, t_test)

        results["models"][sweep["name"]] = {
            "config": sweep,
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
        print(
            f"{sweep['name']}: zero={zero_shot_test:.4f}, "
            f"da={da_test:.4f}, gain={da_test - zero_shot_test:.4f}"
        )

    with open(CONFIG["results_path"], "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved results to {CONFIG['results_path']}")


if __name__ == "__main__":
    main()
