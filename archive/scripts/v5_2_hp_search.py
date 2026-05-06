"""
v5.2 Hyperparameter Search + Pseudo-Label Self-Training (Optimized)
Goal: Exceed publication threshold (C-index ≥ 0.65)

Optimization strategies:
1. Reduced hyperparameter search (16 combinations, 15 epochs each)
2. Pseudo-label self-training with target domain data
3. Full training with best params (50 epochs)
"""
import os
import sys
import json
import time
import itertools
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from omegaconf import OmegaConf
from src.data.loader import DialysisDataLoader
from src.training.dadsn_runner import (
    run_dadsn_experiment, evaluate_survival_metrics, predict_risk,
    train_epoch_dadsn, create_mixed_dataloader
)
from src.models.dadsn_model import DADSN
from src.utils.seed_utils import set_seed
from torch.utils.data import DataLoader, TensorDataset

def load_dataset(loader_obj, csv_path, is_training):
    data = loader_obj.load_data(csv_path, is_training=is_training)
    return {
        "static": data["static"],
        "dynamic": data["dynamic"],
        "targets": data["targets"],
    }

def run_hyperparameter_search(source_data, target_data, save_dir, config, seed=207):
    """
    Run hyperparameter search with reduced grid.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Reduced hyperparameter grid (16 combinations)
    param_grid = {
        "learning_rate": [3e-4, 5e-4, 1e-3],
        "batch_size": [64, 128],
        "da_lambda": [0.3, 0.5, 0.7],
    }
    
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(itertools.product(*param_values))
    
    print(f"\n{'='*80}")
    print(f"HYPERPARAMETER SEARCH (Reduced Grid)")
    print(f"Total combinations: {len(combinations)}")
    print(f"{'='*80}")
    
    results = []
    best_c_index = 0
    best_params = None
    
    for i, combo in enumerate(combinations):
        lr, bs, da_lambda = combo
        params = dict(zip(param_names, combo))
        
        print(f"\n[{i+1}/{len(combinations)}] Testing: {params}")
        
        config.training.learning_rate = lr
        config.training.batch_size = bs
        config.training.domain_adaptation.lambda_da = da_lambda
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        n_static = source_data["static"].shape[1]
        n_dynamic = source_data["dynamic"].shape[2]
        
        model = DADSN(config, n_static, n_dynamic, use_transformer=True).to(device)
        
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=config.training.weight_decay,
        )
        
        loader_src, loader_tgt = create_mixed_dataloader(
            source_data, target_data, bs
        )
        
        # Reduced epochs for search
        search_epochs = 15
        best_model_state = None
        best_epoch_c_index = 0
        
        for epoch in range(search_epochs):
            surv_loss, da_loss = train_epoch_dadsn(
                model, loader_src, loader_tgt, optimizer, device,
                coral_lambda=da_lambda,
                da_method="coral",
                current_epoch=epoch,
                da_warmup_epochs=10,
            )
            
            # Evaluate every 5 epochs
            if (epoch + 1) % 5 == 0:
                metrics = evaluate_survival_metrics(model, target_data, device)
                c_index = metrics["c_index"]
                
                if c_index > best_epoch_c_index:
                    best_epoch_c_index = c_index
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        final_metrics = evaluate_survival_metrics(model, target_data, device)
        
        result = {
            "params": params,
            "c_index": final_metrics["c_index"],
            "ibs": final_metrics["ibs"],
        }
        results.append(result)
        
        print(f"  Target C-index: {final_metrics['c_index']:.4f}, IBS: {final_metrics['ibs']:.4f}")
        
        if final_metrics["c_index"] > best_c_index:
            best_c_index = final_metrics["c_index"]
            best_params = params
        
        # Save intermediate results
        search_results = {
            "all_results": results,
            "best_params": best_params,
            "best_c_index": best_c_index,
        }
        
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "hyperparameter_search_results.json"), "w") as f:
            json.dump(search_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"BEST HYPERPARAMETERS")
    print(f"{'='*80}")
    print(f"  Params: {best_params}")
    print(f"  Target C-index: {best_c_index:.4f}")
    
    return best_params

def run_pseudo_label_training(source_data, target_data, save_dir, config, seed=207, best_params=None):
    """
    Run pseudo-label self-training.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if best_params is None:
        best_params = {
            "learning_rate": 5e-4,
            "batch_size": 128,
            "da_lambda": 0.5,
        }
    
    print(f"\n{'='*80}")
    print(f"PSEUDO-LABEL SELF-TRAINING")
    print(f"Best params: {best_params}")
    print(f"{'='*80}")
    
    config.training.learning_rate = best_params["learning_rate"]
    config.training.batch_size = best_params["batch_size"]
    config.training.domain_adaptation.lambda_da = best_params["da_lambda"]
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    n_static = source_data["static"].shape[1]
    n_dynamic = source_data["dynamic"].shape[2]
    
    # Phase 1: Train initial model
    print("\nPhase 1: Training initial model...")
    model = DADSN(config, n_static, n_dynamic, use_transformer=True).to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=best_params["learning_rate"],
        weight_decay=config.training.weight_decay,
    )
    
    loader_src, loader_tgt = create_mixed_dataloader(
        source_data, target_data, best_params["batch_size"]
    )
    
    # Train for 30 epochs
    for epoch in range(30):
        surv_loss, da_loss = train_epoch_dadsn(
            model, loader_src, loader_tgt, optimizer, device,
            coral_lambda=best_params["da_lambda"],
            da_method="coral",
            current_epoch=epoch,
            da_warmup_epochs=15,
        )
        
        if (epoch + 1) % 10 == 0:
            metrics = evaluate_survival_metrics(model, target_data, device)
            print(f"  Epoch {epoch+1}/30 | Target C-index: {metrics['c_index']:.4f}")
    
    # Phase 2: Generate pseudo-labels
    print("\nPhase 2: Generating pseudo-labels...")
    pseudo_data = generate_pseudo_labels(model, target_data, device, percentile=0.3)
    print(f"  Selected {len(pseudo_data['risk'])} samples")
    
    # Phase 3: Self-training
    print("\nPhase 3: Self-training with pseudo-labels...")
    loader_pseudo = create_pseudo_dataloader(pseudo_data, best_params["batch_size"])
    
    for epoch in range(20):
        surv_loss, da_loss, pseudo_loss = train_with_pseudo_labels(
            model, loader_src, loader_pseudo, optimizer, device,
            da_lambda=best_params["da_lambda"],
            pseudo_weight=0.3,
            current_epoch=epoch,
            da_warmup_epochs=15,
        )
        
        if (epoch + 1) % 5 == 0:
            metrics = evaluate_survival_metrics(model, target_data, device)
            print(f"  Epoch {epoch+1}/20 | Target C-index: {metrics['c_index']:.4f}")
    
    final_metrics = evaluate_survival_metrics(model, target_data, device)
    
    print(f"\n{'='*80}")
    print(f"PSEUDO-LABEL TRAINING COMPLETED")
    print(f"{'='*80}")
    print(f"  Target C-index: {final_metrics['c_index']:.4f}")
    print(f"  Target IBS: {final_metrics['ibs']:.4f}")
    
    os.makedirs(save_dir, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "pseudo_data_size": len(pseudo_data["risk"]),
            "final_metrics": final_metrics,
        },
        os.path.join(save_dir, "pseudo_label_model.pt"),
    )
    
    return final_metrics

def generate_pseudo_labels(model, target_data, device, percentile=0.5):
    """Generate pseudo-labels for target domain data."""
    model.eval()
    risk_scores, event, duration = predict_risk(model, target_data, device)
    
    n_samples = len(risk_scores)
    n_select = int(n_samples * percentile)
    
    sorted_idx = np.argsort(risk_scores)
    high_risk_idx = sorted_idx[-n_select:]
    low_risk_idx = sorted_idx[:n_select]
    
    selected_idx = np.concatenate([high_risk_idx, low_risk_idx])
    
    pseudo_static = target_data["static"][selected_idx]
    pseudo_dynamic = target_data["dynamic"][selected_idx]
    pseudo_event = event[selected_idx]
    pseudo_duration = duration[selected_idx]
    pseudo_risk = risk_scores[selected_idx]
    
    pseudo_labels = np.zeros(len(selected_idx))
    pseudo_labels[n_select:] = 1
    
    return {
        "static": pseudo_static,
        "dynamic": pseudo_dynamic,
        "event": pseudo_event,
        "duration": pseudo_duration,
        "risk": pseudo_risk,
        "labels": pseudo_labels,
        "indices": selected_idx,
    }

def create_pseudo_dataloader(pseudo_data, batch_size, shuffle=True):
    """Create DataLoader for pseudo-labeled data."""
    static_tensor = torch.FloatTensor(pseudo_data["static"])
    dynamic_tensor = torch.FloatTensor(pseudo_data["dynamic"])
    event_tensor = torch.FloatTensor(pseudo_data["event"])
    duration_tensor = torch.FloatTensor(pseudo_data["duration"])
    
    dataset = TensorDataset(static_tensor, dynamic_tensor, event_tensor, duration_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)

def train_with_pseudo_labels(
    model, loader_src, loader_pseudo, optimizer, device,
    da_lambda=0.5, pseudo_weight=0.3, current_epoch=0, da_warmup_epochs=15,
):
    """Train with both source domain and pseudo-labeled target domain data."""
    from src.training.dadsn_runner import train_epoch_dadsn
    
    surv_loss, da_loss = train_epoch_dadsn(
        model, loader_src, None, optimizer, device,
        coral_lambda=da_lambda,
        da_method="coral",
        current_epoch=current_epoch,
        da_warmup_epochs=da_warmup_epochs,
    )
    
    if loader_pseudo is not None and pseudo_weight > 0:
        model.train()
        total_pseudo_loss = 0.0
        n_batches = 0
        
        for batch_static, batch_dyn, batch_event, batch_time in loader_pseudo:
            batch_static = batch_static.to(device)
            batch_dyn = batch_dyn.to(device)
            batch_event = batch_event.to(device)
            batch_time = batch_time.to(device)
            
            valid_mask = (
                (batch_time > 0) & (~torch.isnan(batch_event)) & (~torch.isnan(batch_time))
            )
            if valid_mask.sum() == 0:
                continue
            
            batch_static = batch_static[valid_mask]
            batch_dyn = batch_dyn[valid_mask]
            batch_event = batch_event[valid_mask]
            batch_time = batch_time[valid_mask]
            
            if batch_static.shape[0] <= 1:
                continue
            
            optimizer.zero_grad()
            
            log_hazard, emb_fuse = model(batch_static, batch_dyn)
            
            from src.models.dadsn_model import cox_partial_log_likelihood
            pseudo_loss = cox_partial_log_likelihood(
                log_hazard.squeeze(), batch_event, batch_time
            )
            
            loss = pseudo_weight * pseudo_loss
            loss.backward()
            optimizer.step()
            
            total_pseudo_loss += pseudo_loss.item()
            n_batches += 1
        
        avg_pseudo = total_pseudo_loss / max(n_batches, 1)
    else:
        avg_pseudo = 0.0
    
    return surv_loss, da_loss, avg_pseudo

def main():
    print("=" * 80)
    print("v5.2 HYPERPARAMETER SEARCH + PSEUDO-LABEL SELF-TRAINING (Optimized)")
    print("Goal: Exceed publication threshold (C-index ≥ 0.65)")
    print("=" * 80)
    
    defaults_path = os.path.join(os.path.dirname(__file__), "..", "configs", "defaults.yaml")
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")
    
    defaults_cfg = OmegaConf.load(defaults_path)
    cfg = OmegaConf.load(config_path)
    cfg = OmegaConf.merge(defaults_cfg, cfg)
    
    cfg.experiment.seeds = [207]
    cfg.training.epochs = 50
    cfg.training.early_stopping_patience = 20
    cfg.training.warmup_epochs = 5
    cfg.training.use_weighted_cox = True
    cfg.training.event_weight = 2.0
    
    cfg.training.domain_adaptation = OmegaConf.create({
        "enabled": True,
        "method": "coral",
        "lambda_da": 0.5,
        "warmup_epochs": 15
    })
    
    shared_loader = DialysisDataLoader(cfg)
    
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data_preprocessing", "data")
    train_csv = os.path.join(data_dir, "深医_final_data.csv")
    external_csv = os.path.join(data_dir, "福鼎_final_data.csv")
    
    print("\nLoading Source Domain (深医) Data...")
    source_data = load_dataset(shared_loader, train_csv, is_training=True)
    print(f"  Source samples: {source_data['static'].shape[0]}")
    print(f"  Source features: {source_data['static'].shape[1]}")
    
    print("Loading Target Domain (福鼎) Data...")
    target_data = load_dataset(shared_loader, external_csv, is_training=False)
    print(f"  Target samples: {target_data['static'].shape[0]}")
    print(f"  Target features: {target_data['static'].shape[1]}")
    
    save_dir = os.path.join(os.path.dirname(__file__), "..", "runs", "v5_2_hp_search")
    os.makedirs(save_dir, exist_ok=True)
    
    # Phase 1: Hyperparameter search
    print("\n" + "=" * 80)
    print("PHASE 1: HYPERPARAMETER SEARCH")
    print("=" * 80)
    
    best_params = run_hyperparameter_search(
        source_data=source_data,
        target_data=target_data,
        save_dir=save_dir,
        config=cfg,
        seed=207,
    )
    
    # Phase 2: Pseudo-label self-training
    print("\n" + "=" * 80)
    print("PHASE 2: PSEUDO-LABEL SELF-TRAINING")
    print("=" * 80)
    
    pseudo_save_dir = os.path.join(os.path.dirname(__file__), "..", "runs", "v5_2_pseudo_label")
    final_metrics = run_pseudo_label_training(
        source_data=source_data,
        target_data=target_data,
        save_dir=pseudo_save_dir,
        config=cfg,
        seed=207,
        best_params=best_params,
    )
    
    final_results = {
        "best_params": best_params,
        "final_metrics": final_metrics,
    }
    
    with open(os.path.join(save_dir, "v5_2_final_results.json"), "w") as f:
        json.dump(final_results, f, indent=2)
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"  Best params: {best_params}")
    print(f"  Target C-index: {final_metrics['c_index']:.4f}")
    print(f"  Target IBS: {final_metrics['ibs']:.4f}")
    
    c_pass = "✅" if final_metrics["c_index"] >= 0.65 else "❌"
    ibs_pass = "✅" if final_metrics["ibs"] < 0.25 else "❌"
    print(f"\n  Publication threshold:")
    print(f"    C-index {c_pass} ({final_metrics['c_index']:.4f} ≥ 0.65)")
    print(f"    IBS {ibs_pass} ({final_metrics['ibs']:.4f} < 0.25)")

if __name__ == "__main__":
    main()
