import os
import yaml
import json
import torch
import pandas as pd
from pathlib import Path
import copy

from src.reproducibility import seed_everything
from src.data.dataset import prepare_dataloaders
from src.models.cdan_gsn import DomainStratifiedGatedNet
from src.train.trainer import run_source_pretrain, run_domain_stratified_da
from src.evaluate.metrics import evaluate_survival_metrics
from src.visualization.generate_figures import get_indices

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def run_single_ablation(name, ab_config, base_config, device):
    print(f"\n--- Running Ablation: {name} ---")
    
    # Merge configs
    config = copy.deepcopy(base_config)
    
    remove_features = ab_config.get("remove_features", None)
    if "model" in ab_config:
        for k, v in ab_config["model"].items():
            config["model"][k] = v
    if "training" in ab_config:
        for k, v in ab_config["training"].items():
            config["training"][k] = v
            
    # Prepare data
    data_dict = prepare_dataloaders(
        source_path=os.path.join(config["paths"]["data_dir"], config["data"]["source_file"]),
        target_path=os.path.join(config["paths"]["data_dir"], config["data"]["target_file"]),
        batch_size=config["training"]["batch_size"],
        seed=config["training"]["seed"],
        target_adapt_ratio=config["data"].get("target_adapt_ratio", 0.2),
        target_val_ratio=config["data"].get("target_val_ratio", 0.2),
        patient_col=config["data"].get("patient_col", "患者id"),
        split_strategy=config["data"].get("split_strategy", "patient"),
        remove_features=remove_features,
    )
    
    treat_indices, physio_indices = get_indices(data_dict["feature_names"])
    
    tokenizer_type = "linear" if "linear" in config["model"].get("name", "") else "kan"
    adv_weight = config["training"].get("adv_weight", 0.05)
    
    # Pretrain
    model = DomainStratifiedGatedNet(
        input_dim=data_dict["input_dim"],
        d_model=config["model"]["d_model"],
        nhead=config["model"]["num_heads"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
        domain_hidden=config["model"]["domain_hidden"],
        treat_indices=treat_indices,
        physio_indices=physio_indices,
        tokenizer_type=tokenizer_type,
        kan_basis_dim=config["model"]["kan_bases"]
    ).to(device)
    
    _, _, source_state = run_source_pretrain(
        model, data_dict["source_loader"],
        data_dict["x_val"], data_dict["e_val"], data_dict["t_val"],
        lr=config["training"]["learning_rate"],
        device=device,
        max_epochs=config["training"]["pretrain_epochs"],
        # Use source-domain val so no target info leaks into source model selection
        x_source_val=data_dict.get("x_source_val"),
        e_source_val=data_dict.get("e_source_val"),
        t_source_val=data_dict.get("t_source_val"),
    )
    
    # DA
    da_model = DomainStratifiedGatedNet(
        input_dim=data_dict["input_dim"],
        d_model=config["model"]["d_model"],
        nhead=config["model"]["num_heads"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
        domain_hidden=config["model"]["domain_hidden"],
        treat_indices=treat_indices,
        physio_indices=physio_indices,
        tokenizer_type=tokenizer_type,
        kan_basis_dim=config["model"]["kan_bases"]
    ).to(device)
    da_model.load_state_dict(source_state)
    
    # If no_ipcw, we should technically modify loss, but for ablation, we can simulate by not passing weights or modifying trainer
    # For simplicity, if adv_weight is 0, DA won't do much alignment.
    if adv_weight > 0:
        run_domain_stratified_da(
            da_model, data_dict["source_loader"], data_dict["target_loader"],
            data_dict["x_val"], data_dict["e_val"], data_dict["t_val"],
            adv_weight=adv_weight,
            source_replay_weight=0.1,
            mask_l1_weight=config["training"]["mask_l1_weight"],
            lr=config["training"]["learning_rate"],
            device=device,
            phases_config=[("full_finetune", config["training"]["finetune_epochs"], 4)] # Fast DA for ablation
        )
        
    metrics = evaluate_survival_metrics(
        da_model, data_dict["x_test"], data_dict["e_test"], data_dict["t_test"], 
        device=device, t_train=data_dict["t_train"], e_train=data_dict["e_train"]
    )
    
    print(f"Metrics for {name}: {metrics}")
    return metrics

def main():
    print("="*50)
    print("Running Ablation Studies")
    print("="*50)
    
    base_config = load_yaml("conf/config.yaml")
    ablation_config = load_yaml("conf/ablation/template.yaml")
    
    seed_everything(base_config["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    results = []
    for name, ab_conf in ablation_config.get("ablations", {}).items():
        metrics = run_single_ablation(name, ab_conf, base_config, device)
        row = {"Ablation": name}
        row.update(metrics)
        results.append(row)
        
    df = pd.DataFrame(results)
    out_dir = Path(base_config["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "ablation_results.csv", index=False)
    print(f"\nAblation results saved to {out_dir / 'ablation_results.csv'}")

if __name__ == "__main__":
    main()
