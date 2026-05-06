"""
Enhanced validation script for v5.0 performance optimization.
Uses full training configuration (50 epochs, 3 seeds, cosine LR) to verify publication-level performance.
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
    print("v5.0 ENHANCED VALIDATION - Publication Level Test")
    print("=" * 80)
    
    # Load configs
    defaults_path = os.path.join(os.path.dirname(__file__), "..", "configs", "defaults.yaml")
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")
    
    defaults_cfg = OmegaConf.load(defaults_path)
    cfg = OmegaConf.load(config_path)
    cfg = OmegaConf.merge(defaults_cfg, cfg)
    
    # Enhanced configuration for publication-level performance
    cfg.experiment.seeds = [42, 123, 456]  # 3 seeds for statistical significance
    cfg.training.epochs = 50
    cfg.training.batch_size = 128  # Larger batch for stable gradients
    cfg.training.learning_rate = 5e-4
    cfg.training.early_stopping_patience = 20  # Allow longer convergence
    cfg.training.use_cosine_annealing = True  # Enable cosine LR scheduling
    cfg.training.warmup_epochs = 5  # Short warmup to avoid initial instability
    
    # Domain adaptation configuration
    if not hasattr(cfg.training, 'domain_adaptation'):
        cfg.training.domain_adaptation = OmegaConf.create({
            "enabled": True,
            "method": "mmd",
            "lambda_da": 0.5,
            "warmup_epochs": 15
        })
    else:
        cfg.training.domain_adaptation.lambda_da = 0.5
        cfg.training.domain_adaptation.warmup_epochs = 15
    
    print(f"Enhanced Config:")
    print(f"  Seeds: {cfg.experiment.seeds}")
    print(f"  Epochs: {cfg.training.epochs}")
    print(f"  Batch size: {cfg.training.batch_size}")
    print(f"  Learning rate: {cfg.training.learning_rate}")
    print(f"  Cosine Annealing: {cfg.training.use_cosine_annealing}")
    print(f"  Warmup epochs: {cfg.training.warmup_epochs}")
    print(f"  Early stopping patience: {cfg.training.early_stopping_patience}")
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
    save_dir = os.path.join(os.path.dirname(__file__), "..", "runs", "v5_enhanced_validation")
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("STARTING ENHANCED VALIDATION")
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
    print("ENHANCED VALIDATION COMPLETED")
    print(f"Elapsed time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print("=" * 80)
    
    # Save results
    results_path = os.path.join(save_dir, "v5_enhanced_validation_results.json")
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
    
    # Compare with publication threshold
    print("\n" + "=" * 80)
    print("PUBLICATION THRESHOLD ASSESSMENT")
    print("=" * 80)
    print("  Target: C-index ≥ 0.65, IBS < 0.25")
    
    for exp_name, (mean_metrics, std_metrics) in results.items():
        c_index = mean_metrics.get('c_index', 0)
        ibs = mean_metrics.get('ibs', 1.0)
        c_pass = "✅" if c_index >= 0.65 else "❌"
        ibs_pass = "✅" if ibs < 0.25 else "❌"
        print(f"  {exp_name}: C-index {c_pass} ({c_index:.4f}), IBS {ibs_pass} ({ibs:.4f})")

if __name__ == "__main__":
    main()
