"""
Quick validation script for v5.0 fixes.
Runs a single seed ablation study with reduced epochs to verify performance.
"""
import os
import sys
import json
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from omegaconf import OmegaConf
from src.data.loader import DialysisDataLoader
from src.training.dadsn_runner import run_ablation_study
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
    print("v5.0 QUICK VALIDATION")
    print("=" * 80)
    
    # Load configs manually
    defaults_path = os.path.join(os.path.dirname(__file__), "..", "configs", "defaults.yaml")
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")
    
    defaults_cfg = OmegaConf.load(defaults_path)
    cfg = OmegaConf.load(config_path)
    cfg = OmegaConf.merge(defaults_cfg, cfg)
    
    # Override for quick validation
    cfg.experiment.seeds = [42]
    cfg.training.epochs = 15
    cfg.training.batch_size = 64
    cfg.training.learning_rate = 5e-4
    cfg.training.early_stopping_patience = 10
    
    # Ensure domain_adaptation config exists
    if not hasattr(cfg.training, 'domain_adaptation'):
        if hasattr(cfg.training, 'coral'):
            # Migrate old config
            cfg.training.domain_adaptation = OmegaConf.create({
                "enabled": True,
                "method": "mmd",
                "lambda_da": getattr(cfg.training.coral, 'lambda_coral', 0.5),
                "warmup_epochs": 15
            })
        else:
            cfg.training.domain_adaptation = OmegaConf.create({
                "enabled": True,
                "method": "mmd",
                "lambda_da": 0.5,
                "warmup_epochs": 15
            })
    
    print(f"Config:")
    print(f"  Seeds: {cfg.experiment.seeds}")
    print(f"  Epochs: {cfg.training.epochs}")
    print(f"  Batch size: {cfg.training.batch_size}")
    print(f"  Learning rate: {cfg.training.learning_rate}")
    print(f"  DA method: {cfg.training.domain_adaptation.method}")
    print(f"  DA lambda: {cfg.training.domain_adaptation.lambda_da}")
    
    # Set seed
    set_seed(42)
    
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
    
    # Create save directory
    save_dir = os.path.join(os.path.dirname(__file__), "..", "runs", "v5_validation")
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("STARTING VALIDATION")
    print("=" * 80)
    
    start_time = time.time()
    
    results = run_ablation_study(
        source_data=source_data,
        target_data=target_data,
        save_dir=save_dir,
        config=cfg,
        seeds=cfg.experiment.seeds,
    )
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("VALIDATION COMPLETED")
    print(f"Elapsed time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print("=" * 80)
    
    # Save results
    results_path = os.path.join(save_dir, "v5_validation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    for exp_name, (mean_metrics, std_metrics) in results.items():
        print(f"\n{exp_name}:")
        print(f"  Target C-index: {mean_metrics['c_index']:.4f} ± {std_metrics['c_index']:.4f}")
        print(f"  Target IBS:     {mean_metrics['ibs']:.4f} ± {std_metrics['ibs']:.4f}")
        print(f"  Source C-index: {mean_metrics.get('source_c_index', 'N/A')}")
    
    # Compare with historical best
    print("\n" + "=" * 80)
    print("HISTORICAL COMPARISON (v4.0)")
    print("=" * 80)
    print("  Base (Static Only):        C-index ≈ 0.5545")
    print("  Base + ResNet1D:           C-index ≈ 0.5814")
    print("  DA-DSN (MMD):              C-index ≈ 0.6137")
    print("\n  Publication threshold:     C-index ≥ 0.65, IBS < 0.25")

if __name__ == "__main__":
    main()
