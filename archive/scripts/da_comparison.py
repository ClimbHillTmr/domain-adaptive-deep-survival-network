"""
Domain Adaptation Methods Comparison for v5.0.
Compares: CORAL, MMD, Joint-MMD, DANN

Goal: Find the best DA method to exceed publication threshold (C-index ≥ 0.65)
"""
import os
import sys
import json
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from omegaconf import OmegaConf
from src.data.loader import DialysisDataLoader
from src.training.dadsn_runner import run_dadsn_experiment
from src.utils.seed_utils import set_seed

def load_dataset(loader_obj, csv_path, is_training):
    data = loader_obj.load_data(csv_path, is_training=is_training)
    return {
        "static": data["static"],
        "dynamic": data["dynamic"],
        "targets": data["targets"],
    }

def main():
    print("=" * 80)
    print("v5.0 DOMAIN ADAPTATION METHODS COMPARISON")
    print("Comparing: CORAL, MMD, Joint-MMD, DANN")
    print("Goal: Exceed publication threshold (C-index ≥ 0.65)")
    print("=" * 80)
    
    # Load configs
    defaults_path = os.path.join(os.path.dirname(__file__), "..", "configs", "defaults.yaml")
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")
    
    defaults_cfg = OmegaConf.load(defaults_path)
    cfg = OmegaConf.load(config_path)
    cfg = OmegaConf.merge(defaults_cfg, cfg)
    
    # Base configuration
    cfg.experiment.seeds = [207]  # Same seed as v5_optimized for fair comparison
    cfg.training.epochs = 50
    cfg.training.batch_size = 128
    cfg.training.learning_rate = 5e-4
    cfg.training.early_stopping_patience = 20
    cfg.training.warmup_epochs = 5
    cfg.training.use_weighted_cox = True
    cfg.training.event_weight = 2.0
    
    # Load data
    shared_loader = DialysisDataLoader(cfg)
    
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data_preprocessing", "data")
    train_csv = os.path.join(data_dir, "深医_final_data.csv")
    external_csv = os.path.join(data_dir, "福鼎_final_data.csv")
    
    print("\nLoading Source Domain (深医) Data...")
    source_data = load_dataset(shared_loader, train_csv, is_training=True)
    print(f"  Source samples: {source_data['static'].shape[0]}")
    
    print("Loading Target Domain (福鼎) Data...")
    target_data = load_dataset(shared_loader, external_csv, is_training=False)
    print(f"  Target samples: {target_data['static'].shape[0]}")
    
    # Domain adaptation methods to compare
    da_methods = [
        {"name": "coral", "method": "coral", "lambda_da": 0.5, "warmup_epochs": 15},
        {"name": "mmd", "method": "mmd", "lambda_da": 0.5, "warmup_epochs": 15},
        {"name": "joint_mmd", "method": "joint_mmd", "lambda_da": 0.5, "warmup_epochs": 15},
        {"name": "joint_coral_mmd", "method": "joint_coral_mmd", "lambda_da": 0.5, "warmup_epochs": 15},
        {"name": "dann", "method": "dann", "lambda_da": 0.5, "warmup_epochs": 15, "alpha": 1.0},
    ]
    
    all_results = {}
    
    for da_cfg in da_methods:
        print("\n" + "=" * 80)
        print(f"Testing DA Method: {da_cfg['name'].upper()}")
        print("=" * 80)
        
        # Configure domain adaptation
        cfg.training.domain_adaptation = OmegaConf.create({
            "enabled": True,
            "method": da_cfg["method"],
            "lambda_da": da_cfg["lambda_da"],
            "warmup_epochs": da_cfg["warmup_epochs"],
        })
        if "alpha" in da_cfg:
            cfg.training.domain_adaptation.alpha = da_cfg["alpha"]
        
        # Create save directory
        save_dir = os.path.join(
            os.path.dirname(__file__), "..", "runs", "v5_da_comparison", da_cfg["name"]
        )
        os.makedirs(save_dir, exist_ok=True)
        
        start_time = time.time()
        
        # Run experiment with DA-DSN (transformer=True, coral=True to enable DA)
        results_mean, results_std = run_dadsn_experiment(
            source_data=source_data,
            target_data=target_data,
            save_dir=save_dir,
            config=cfg,
            use_coral=True,
            use_transformer=True,
            seeds=cfg.experiment.seeds,
        )
        
        elapsed = time.time() - start_time
        
        all_results[da_cfg["name"]] = {
            "mean": results_mean,
            "std": results_std,
            "elapsed": elapsed,
        }
        
        print(f"\n{da_cfg['name'].upper()} Results:")
        print(f"  Target C-index: {results_mean['target_c_index']:.4f} ± {results_std['target_c_index_std']:.4f}")
        print(f"  Target IBS:     {results_mean['target_ibs']:.4f} ± {results_std['target_ibs_std']:.4f}")
        print(f"  Elapsed time:   {elapsed:.1f}s ({elapsed/60:.1f}min)")
    
    # Save all results
    save_dir = os.path.join(os.path.dirname(__file__), "..", "runs", "v5_da_comparison")
    os.makedirs(save_dir, exist_ok=True)
    
    results_path = os.path.join(save_dir, "da_comparison_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Method':<15} {'Target C-index':<20} {'Target IBS':<15} {'Time (min)':<12}")
    print("-" * 62)
    
    for name, results in all_results.items():
        c_index = results["mean"]["target_c_index"]
        ibs = results["mean"]["target_ibs"]
        elapsed = results["elapsed"] / 60
        print(f"{name:<15} {c_index:<20.4f} {ibs:<15.4f} {elapsed:<12.1f}")
    
    print("\n" + "=" * 80)
    print("PUBLICATION THRESHOLD ASSESSMENT")
    print("=" * 80)
    print("  Target: C-index ≥ 0.65, IBS < 0.25")
    
    best_method = None
    best_c_index = 0
    
    for name, results in all_results.items():
        c_index = results["mean"]["target_c_index"]
        ibs = results["mean"]["target_ibs"]
        c_pass = "✅" if c_index >= 0.65 else "❌"
        ibs_pass = "✅" if ibs < 0.25 else "❌"
        print(f"  {name:<12}: C-index {c_pass} ({c_index:.4f}), IBS {ibs_pass} ({ibs:.4f})")
        
        if c_index > best_c_index:
            best_c_index = c_index
            best_method = name
    
    print(f"\n  Best DA Method: {best_method} (C-index={best_c_index:.4f})")

if __name__ == "__main__":
    main()
