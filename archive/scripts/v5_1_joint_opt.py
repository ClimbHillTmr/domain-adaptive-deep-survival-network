"""
v5.1 Joint Optimization: CORAL+MMD + Extended Features
Goal: Exceed publication threshold (C-index ≥ 0.65)

Optimization strategies:
1. Joint CORAL+MMD domain adaptation with adaptive weight scheduling
2. Extended feature engineering (10 new clinical derived features)
3. Full training configuration (50 epochs, seed=207)
"""
import os
import sys
import json
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from omegaconf import OmegaConf
from src.data.loader import DialysisDataLoader
from src.training.dadsn_runner import run_ablation_study, run_dadsn_experiment
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
    print("v5.1 JOINT OPTIMIZATION: CORAL+MMD + Extended Features")
    print("Goal: Exceed publication threshold (C-index ≥ 0.65)")
    print("=" * 80)
    
    # Load configs
    defaults_path = os.path.join(os.path.dirname(__file__), "..", "configs", "defaults.yaml")
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")
    
    defaults_cfg = OmegaConf.load(defaults_path)
    cfg = OmegaConf.load(config_path)
    cfg = OmegaConf.merge(defaults_cfg, cfg)
    
    # Optimized configuration
    cfg.experiment.seeds = [207]  # Same seed as v5.0 for fair comparison
    cfg.training.epochs = 50
    cfg.training.batch_size = 128
    cfg.training.learning_rate = 5e-4
    cfg.training.early_stopping_patience = 20
    cfg.training.warmup_epochs = 5
    cfg.training.use_weighted_cox = True
    cfg.training.event_weight = 2.0
    
    # Domain adaptation configuration - Joint CORAL+MMD
    cfg.training.domain_adaptation = OmegaConf.create({
        "enabled": True,
        "method": "joint_coral_mmd",
        "lambda_da": 0.5,
        "warmup_epochs": 15
    })
    
    print(f"Optimized Config:")
    print(f"  Seeds: {cfg.experiment.seeds}")
    print(f"  Epochs: {cfg.training.epochs}")
    print(f"  Batch size: {cfg.training.batch_size}")
    print(f"  Learning rate: {cfg.training.learning_rate}")
    print(f"  DA method: {cfg.training.domain_adaptation.method}")
    print(f"  DA lambda: {cfg.training.domain_adaptation.lambda_da}")
    print(f"  Static features: {len(cfg.columns.static_cols)} (extended)")
    
    # Load data
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
    
    # Create save directory
    save_dir = os.path.join(os.path.dirname(__file__), "..", "runs", "v5_1_joint_opt")
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("STARTING JOINT OPTIMIZATION TRAINING")
    print("=" * 80)
    
    start_time = time.time()
    
    # Run ablation study with extended features
    results = run_ablation_study(
        source_data=source_data,
        target_data=target_data,
        save_dir=save_dir,
        config=cfg,
        seeds=cfg.experiment.seeds,
    )
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("JOINT OPTIMIZATION COMPLETED")
    print(f"Elapsed time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print("=" * 80)
    
    # Save results
    results_path = os.path.join(save_dir, "v5_1_joint_opt_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    for exp_name, (mean_metrics, std_metrics) in results.items():
        print(f"\n{exp_name}:")
        print(f"  Target C-index: {mean_metrics['target_c_index']:.4f} ± {std_metrics['target_c_index_std']:.4f}")
        print(f"  Target IBS:     {mean_metrics['target_ibs']:.4f} ± {std_metrics['target_ibs_std']:.4f}")
        print(f"  Source C-index: {mean_metrics.get('source_c_index', 'N/A')}")
    
    # Compare with publication threshold
    print("\n" + "=" * 80)
    print("PUBLICATION THRESHOLD ASSESSMENT")
    print("=" * 80)
    print("  Target: C-index ≥ 0.65, IBS < 0.25")
    
    for exp_name, (mean_metrics, std_metrics) in results.items():
        c_index = mean_metrics.get('target_c_index', 0)
        ibs = mean_metrics.get('target_ibs', 1.0)
        c_pass = "✅" if c_index >= 0.65 else "❌"
        ibs_pass = "✅" if ibs < 0.25 else "❌"
        print(f"  {exp_name}: C-index {c_pass} ({c_index:.4f}), IBS {ibs_pass} ({ibs:.4f})")

if __name__ == "__main__":
    main()
