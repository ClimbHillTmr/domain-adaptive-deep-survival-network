"""
Phase 4 – Ablation Study Runner
================================
Runs 4 ablations against the same unified config (23 features, adapt_ratio=0.4).
Ablations:
  no_clinical  : remove pre-dialysis vital-sign features (keep demographics + history)
  no_history   : remove all history aggregate / IDH-timing features
  no_kan       : replace KAN tokenizer with plain linear projection
  no_cdan      : set adv_weight=0 (domain discriminator disabled)

Output: experiments/results/ablation_results_v2.json
"""
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset import prepare_dataloaders
from src.evaluate.metrics import evaluate_survival_metrics
from src.models.cdan_gsn import DomainStratifiedGatedNet
from src.reproducibility import seed_everything
from src.train.trainer import run_domain_stratified_da, run_source_pretrain


def load_config(path="conf/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


# Pre-dialysis vitals + derived BP (9 features)
NO_CLINICAL_FEATURES = [
    "超负荷",
    "透前体重-干体重",
    "透前呼吸频率",
    "透前体温",
    "透前收缩压",
    "透前舒张压",
    "透前动脉压",
    "脉压差",
    "平均动脉压",
]

# History aggregates + IDH timing bins (12 features)
NO_HISTORY_FEATURES = [
    "历史平均超滤量MAX",
    "历史平均透前体重",
    "历史平均透前收缩压",
    "历史平均透前舒张压",
    "历史平均透中低血压_计算",
    "history_HBP_rate",
    "history_LBP_times_0_rate",
    "history_LBP_times_1_rate",
    "history_LBP_times_2_rate",
    "history_LBP_times_3_rate",
    "history_LBP_times_4_rate",
    "历史平均超滤率_mean",
]


def get_indices(feature_names):
    treat_features = [
        "超滤比", "超滤率_绝对", "超滤率_体重归一化", "超滤量MAX",
        "透析液电导率", "透析液钙浓度", "抗凝剂类型_code",
        "瘘管位置_code", "瘘管类型_code", "透析方式_code",
    ]
    treat_idx, physio_idx = [], []
    for i, n in enumerate(feature_names):
        (treat_idx if any(t in n for t in treat_features) else physio_idx).append(i)
    return treat_idx, physio_idx


def build_model(data_dict, config, tokenizer_type="kan", device="cpu"):
    treat_idx, physio_idx = get_indices(data_dict["feature_names"])
    return DomainStratifiedGatedNet(
        input_dim=data_dict["input_dim"],
        d_model=config["model"]["d_model"],
        nhead=config["model"]["num_heads"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
        domain_hidden=config["model"]["domain_hidden"],
        treat_indices=treat_idx,
        physio_indices=physio_idx,
        tokenizer_type=tokenizer_type,
        kan_basis_dim=config["model"]["kan_bases"],
    ).to(device)


def train_and_eval(data_dict, config, device, tokenizer_type="kan", adv_weight_override=None):
    seed = config["training"]["seed"]
    seed_everything(seed)
    lr = config["training"]["learning_rate"]
    finetune_lr = config["training"].get("finetune_lr", 3e-5)
    adv_weight = adv_weight_override if adv_weight_override is not None else config["training"]["adv_weight"]
    mask_l1_weight = config["training"]["mask_l1_weight"]
    pretrain_epochs = config["training"]["pretrain_epochs"]

    model = build_model(data_dict, config, tokenizer_type=tokenizer_type, device=device)

    _, _, source_state = run_source_pretrain(
        model, data_dict["source_loader"],
        data_dict["x_val"], data_dict["e_val"], data_dict["t_val"],
        lr=lr, device=device, max_epochs=pretrain_epochs, patience=5,
        x_source_val=data_dict["x_source_val"],
        e_source_val=data_dict["e_source_val"],
        t_source_val=data_dict["t_source_val"],
    )

    da_model = build_model(data_dict, config, tokenizer_type=tokenizer_type, device=device)
    da_model.load_state_dict(source_state)

    run_domain_stratified_da(
        da_model, data_dict["source_loader"], data_dict["target_loader"],
        data_dict["x_val"], data_dict["e_val"], data_dict["t_val"],
        adv_weight=adv_weight, source_replay_weight=0.1,
        mask_l1_weight=mask_l1_weight, lr=lr, finetune_lr=finetune_lr,
        device=device,
    )

    return evaluate_survival_metrics(
        da_model, data_dict["x_test"], data_dict["e_test"], data_dict["t_test"],
        device=device, t_train=data_dict["t_train"], e_train=data_dict["e_train"],
        patient_ids=data_dict["patient_ids_test"], n_bootstrap=200, seed=seed,
    )


def main():
    config = load_config()
    seed = config["training"]["seed"]
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    source_path = os.path.join(config["paths"]["data_dir"], config["data"]["source_file"])
    target_path = os.path.join(config["paths"]["data_dir"], config["data"]["target_file"])
    base_kwargs = dict(
        source_path=source_path,
        target_path=target_path,
        batch_size=config["training"]["batch_size"],
        seed=seed,
        target_adapt_ratio=config["data"]["target_adapt_ratio"],
        target_val_ratio=config["data"]["target_val_ratio"],
        patient_col=config["data"]["patient_col"],
        split_strategy=config["data"]["split_strategy"],
    )

    ablations = [
        {
            "name": "no_clinical",
            "label": "w/o Pre-dialysis Vitals",
            "remove_features": NO_CLINICAL_FEATURES,
            "tokenizer_type": "kan",
            "adv_weight": None,
        },
        {
            "name": "no_history",
            "label": "w/o History Features",
            "remove_features": NO_HISTORY_FEATURES,
            "tokenizer_type": "kan",
            "adv_weight": None,
        },
        {
            "name": "no_kan",
            "label": "w/o KAN (Linear tokenizer)",
            "remove_features": None,
            "tokenizer_type": "linear",
            "adv_weight": None,
        },
        {
            "name": "no_cdan",
            "label": "w/o Domain Adaptation",
            "remove_features": None,
            "tokenizer_type": "kan",
            "adv_weight": 0.0,
        },
    ]

    results = []
    for ablation in ablations:
        name = ablation["name"]
        print(f"\n[Ablation] {name} ({ablation['label']})...")
        data_dict = prepare_dataloaders(
            **base_kwargs,
            remove_features=ablation["remove_features"],
        )
        n_feat = len(data_dict["feature_names"])
        print(f"  Features used: {n_feat}")
        try:
            metrics = train_and_eval(
                data_dict, config, device,
                tokenizer_type=ablation["tokenizer_type"],
                adv_weight_override=ablation["adv_weight"],
            )
            print(f"  C-index: {metrics['C-index']:.4f} [{metrics['C-index_CI_lower']:.4f}, {metrics['C-index_CI_upper']:.4f}]")
            results.append({
                "name": name,
                "label": ablation["label"],
                "n_features": n_feat,
                "metrics": metrics,
            })
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({"name": name, "label": ablation["label"], "error": str(e)})

    out_path = Path("experiments/results/ablation_results_v2.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Also write simplified CSV for backwards compatibility
    import pandas as pd
    rows = []
    for r in results:
        if "metrics" in r:
            m = r["metrics"]
            rows.append({
                "Ablation": r["name"],
                "Label": r["label"],
                "n_features": r["n_features"],
                "C-index": m["C-index"],
                "C-index_CI_lower": m["C-index_CI_lower"],
                "C-index_CI_upper": m["C-index_CI_upper"],
                "AUC_30m": m.get("AUC_30m", ""),
                "AUC_60m": m.get("AUC_60m", ""),
                "AUC_120m": m.get("AUC_120m", ""),
            })
    if rows:
        pd.DataFrame(rows).to_csv("experiments/results/ablation_results.csv", index=False)

    print(f"\nAblation results saved to {out_path}")


if __name__ == "__main__":
    main()
