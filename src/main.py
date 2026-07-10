import os
import sys
import yaml
import json
import torch
from pathlib import Path

# Ensure src is in Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.reproducibility import seed_everything, record_environment
from src.data.dataset import prepare_dataloaders
from src.models.cdan_gsn import DomainStratifiedGatedNet
from src.train.trainer import run_source_pretrain, run_domain_stratified_da
from src.evaluate.metrics import evaluate_survival_metrics

def load_config(config_path="conf/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_indices(feature_names):
    treat_features = [
        "超滤比", "超滤率_绝对", "超滤率_体重归一化", "超滤量MAX",
        "透析液电导率", "透析液钙浓度", "抗凝剂类型_code",
        "瘘管位置_code", "瘘管类型_code", "透析方式_code"
    ]
    treat_indices = []
    physio_indices = []
    for i, name in enumerate(feature_names):
        if any(t in name for t in treat_features):
            treat_indices.append(i)
        else:
            physio_indices.append(i)
    return treat_indices, physio_indices

def main():
    print("="*50)
    print("Starting DA-DSN Training Pipeline")
    print("="*50)
    
    config = load_config()
    seed_everything(config["training"]["seed"])
    record_environment()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Data
    print("\n[1/4] Preparing Data Loaders...")
    data_dict = prepare_dataloaders(
        source_path=os.path.join(config["paths"]["data_dir"], config["data"]["source_file"]),
        target_path=os.path.join(config["paths"]["data_dir"], config["data"]["target_file"]),
        batch_size=config["training"]["batch_size"],
        seed=config["training"]["seed"],
        target_adapt_ratio=config["data"].get("target_adapt_ratio", 0.2),
        target_val_ratio=config["data"].get("target_val_ratio", 0.2),
        patient_col=config["data"].get("patient_col", "患者id"),
        split_strategy=config["data"].get("split_strategy", "patient"),
    )
    
    treat_indices, physio_indices = get_indices(data_dict["feature_names"])
    
    # 2. Initialize Model
    print("\n[2/4] Initializing CDAN-GSN Model...")
    model = DomainStratifiedGatedNet(
        input_dim=data_dict["input_dim"],
        d_model=config["model"]["d_model"],
        nhead=config["model"]["num_heads"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
        domain_hidden=config["model"]["domain_hidden"],
        treat_indices=treat_indices,
        physio_indices=physio_indices,
        tokenizer_type="kan",
        kan_basis_dim=config["model"]["kan_bases"]
    ).to(device)
    
    # 3. Train Model
    print("\n[3/4] Training Model...")
    print("-> Phase 1: Source Pretraining")
    best_pre_val, best_pre_epoch, source_state = run_source_pretrain(
        model, data_dict["source_loader"], 
        data_dict["x_val"], data_dict["e_val"], data_dict["t_val"],
        lr=config["training"]["learning_rate"],
        device=device,
        max_epochs=config["training"]["pretrain_epochs"]
    )
    
    # Zero-shot evaluation
    zero_shot_metrics = evaluate_survival_metrics(
        model, data_dict["x_test"], data_dict["e_test"], data_dict["t_test"], 
        device=device, t_train=data_dict["t_train"], e_train=data_dict["e_train"],
        patient_ids=data_dict["patient_ids_test"], n_bootstrap=200, seed=config["training"]["seed"],
    )
    print(f"   Zero-shot Metrics: {zero_shot_metrics}")
    
    print("-> Phase 2: Domain Stratified DA")
    da_model = DomainStratifiedGatedNet(
        input_dim=data_dict["input_dim"],
        d_model=config["model"]["d_model"],
        nhead=config["model"]["num_heads"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
        domain_hidden=config["model"]["domain_hidden"],
        treat_indices=treat_indices,
        physio_indices=physio_indices,
        tokenizer_type="kan",
        kan_basis_dim=config["model"]["kan_bases"]
    ).to(device)
    da_model.load_state_dict(source_state)
    
    phase_results = run_domain_stratified_da(
        da_model, data_dict["source_loader"], data_dict["target_loader"],
        data_dict["x_val"], data_dict["e_val"], data_dict["t_val"],
        adv_weight=config["training"]["adv_weight"],
        source_replay_weight=0.1,
        mask_l1_weight=config["training"]["mask_l1_weight"],
        lr=config["training"]["learning_rate"],
        finetune_lr=config["training"].get("finetune_lr", 0.0001),
        device=device
    )
    
    # 4. Evaluate
    print("\n[4/4] Final Evaluation...")
    da_metrics = evaluate_survival_metrics(
        da_model, data_dict["x_test"], data_dict["e_test"], data_dict["t_test"], 
        device=device, t_train=data_dict["t_train"], e_train=data_dict["e_train"],
        patient_ids=data_dict["patient_ids_test"], n_bootstrap=200, seed=config["training"]["seed"],
    )
    print(f"   DA Metrics: {da_metrics}")
    
    # Save Results
    results = {
        "metadata": {
            "split_strategy": data_dict["split_strategy"],
            "patient_col": data_dict["patient_col"],
            "target_adapt_ratio": config["data"].get("target_adapt_ratio", 0.2),
            "target_val_ratio": config["data"].get("target_val_ratio", 0.2),
            "n_features": len(data_dict["feature_names"]),
            "feature_names": data_dict["feature_names"],
        },
        "sweeps": {
            "Zero-shot": {"metrics": zero_shot_metrics},
            "CDAN-GSN (Ours)": {"metrics": da_metrics}
        }
    }
    
    out_path = Path(config["paths"]["output_dir"]) / "evaluation_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {out_path}")

    torch.save(
        {
            "model_state_dict": da_model.state_dict(),
            "feature_names": data_dict["feature_names"],
            "config": config,
        },
        out_path.parent / "cdan_gsn_final.pt",
    )
    
    print("\nExecuting Table Generator...")
    os.system("PYTHONPATH=. python src/evaluate/generate_tables.py")
    
if __name__ == "__main__":
    main()
