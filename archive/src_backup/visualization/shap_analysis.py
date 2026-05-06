import os
import json
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import shap
from pathlib import Path

from src.models.cdan_gsn import DomainStratifiedGatedNet
from src.data.dataset import prepare_dataloaders

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
    print("Starting Clinical Interpretability Analysis (SHAP)")
    print("="*50)
    
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Data
    data_dict = prepare_dataloaders(
        source_path=os.path.join(config["paths"]["data_dir"], config["data"]["source_file"]),
        target_path=os.path.join(config["paths"]["data_dir"], config["data"]["target_file"]),
        batch_size=config["training"]["batch_size"],
        seed=config["training"]["seed"]
    )
    
    treat_indices, physio_indices = get_indices(data_dict["feature_names"])
    
    # 2. Load Model
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
    
    # Normally we would load the trained weights here:
    # model.load_state_dict(torch.load("experiments/results/best_model.pth"))
    model.eval()
    
    # 3. Prepare Background and Test data for SHAP
    # Use a small subset of source data as background for GradientExplainer
    x_bg_tensor = []
    for x_s, _, _, _ in data_dict["source_loader"]:
        x_bg_tensor.append(x_s)
        if len(x_bg_tensor) * config["training"]["batch_size"] > 200:
            break
    x_bg = torch.cat(x_bg_tensor, dim=0)[:200].to(device)
    
    x_test = torch.tensor(data_dict["x_test"][:500], dtype=torch.float32).to(device)
    
    # Wrapper function for SHAP to only get the hazard output
    def model_predict(x):
        _, hazard, _, _ = model(x)
        return hazard
        
    print("\nCalculating SHAP values (this may take a few minutes)...")
    explainer = shap.GradientExplainer(model_predict, x_bg)
    shap_values = explainer.shap_values(x_test)
    
    # Handle SHAP output format depending on shap version/model
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
        
    shap_values_np = shap_values.reshape(x_test.shape[0], -1) if len(shap_values.shape) > 2 else shap_values
    x_test_np = x_test.cpu().numpy()
    
    feature_names = data_dict["feature_names"]
    
    # 4. Generate SHAP Summary Plot
    print("Generating SHAP Summary Plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_np, x_test_np, feature_names=feature_names, show=False)
    
    out_dir = Path(config["paths"]["figure_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    
    summary_path = out_dir / "fig2_shap_summary.pdf"
    plt.savefig(summary_path, bbox_inches='tight', dpi=300)
    plt.savefig(out_dir / "fig2_shap_summary.png", bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved SHAP summary plot to {summary_path}")
    
    # 5. Extract Top Features and Generate Dependence Plots
    mean_abs_shap = np.abs(shap_values_np).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:3]
    top_features = [feature_names[i] for i in top_indices]
    
    print(f"\nTop 3 most important features: {top_features}")
    
    for i, feature_idx in enumerate(top_indices):
        feature_name = feature_names[feature_idx]
        print(f"Generating SHAP Dependence Plot for {feature_name}...")
        plt.figure(figsize=(8, 6))
        shap.dependence_plot(
            feature_idx, 
            shap_values_np, 
            x_test_np, 
            feature_names=feature_names, 
            show=False,
            interaction_index=None  # Disable auto-interaction for cleaner plot
        )
        dep_path = out_dir / f"fig3_shap_dependence_{i+1}.pdf"
        plt.savefig(dep_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved Dependence plot to {dep_path}")
        
    print("\nInterpretability Analysis Completed!")

if __name__ == "__main__":
    main()
