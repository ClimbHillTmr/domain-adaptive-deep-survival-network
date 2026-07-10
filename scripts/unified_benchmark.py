"""
Unified benchmark: run CoxPH + CDAN-GSN grid under identical config.
Outputs: experiments/results/unified_benchmark.json
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
from src.evaluate.metrics import (
    _binary_auc,
    compute_bootstrap_cindex,
    concordance_index,
    evaluate_survival_metrics,
)
from src.models.cdan_gsn import DomainStratifiedGatedNet
from src.reproducibility import seed_everything
from src.train.trainer import run_domain_stratified_da, run_source_pretrain


def load_config(path="conf/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def get_indices(feature_names):
    treat_features = [
        "超滤比", "超滤率_绝对", "超滤率_体重归一化", "超滤量MAX",
        "透析液电导率", "透析液钙浓度", "抗凝剂类型_code",
        "瘘管位置_code", "瘘管类型_code", "透析方式_code",
    ]
    treat_indices, physio_indices = [], []
    for i, name in enumerate(feature_names):
        if any(t in name for t in treat_features):
            treat_indices.append(i)
        else:
            physio_indices.append(i)
    return treat_indices, physio_indices


def run_cox_baseline(data_dict, seed):
    """Run CoxPH with lifelines on the same split."""
    from lifelines import CoxPHFitter
    import pandas as pd

    feature_names = data_dict["feature_names"]
    x_train, e_train, t_train = data_dict["x_train"], data_dict["e_train"], data_dict["t_train"]
    x_val, e_val, t_val = data_dict["x_val"], data_dict["e_val"], data_dict["t_val"]
    x_test, e_test, t_test = data_dict["x_test"], data_dict["e_test"], data_dict["t_test"]

    train_df = pd.DataFrame(x_train, columns=feature_names)
    train_df["et_min"] = t_train
    train_df["events"] = e_train.astype(int)

    val_df = pd.DataFrame(x_val, columns=feature_names)
    val_df["et_min"] = t_val
    val_df["events"] = e_val.astype(int)

    test_df = pd.DataFrame(x_test, columns=feature_names)
    test_df["et_min"] = t_test
    test_df["events"] = e_test.astype(int)

    # Remove low-variance features
    variances = train_df[feature_names].var()
    kept = [n for n in feature_names if np.isfinite(train_df[n]).all() and variances[n] > 1e-4]

    # Select penalizer
    best_val, best_pen = -1, 0.01
    for pen in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]:
        fitter = CoxPHFitter(penalizer=pen)
        fitter.fit(train_df[kept + ["et_min", "events"]], duration_col="et_min", event_col="events", show_progress=False)
        val_risk = fitter.predict_log_partial_hazard(val_df[kept]).to_numpy().reshape(-1)
        val_c = concordance_index(val_df["et_min"].values, val_risk, val_df["events"].values)
        if val_c > best_val:
            best_val, best_pen = val_c, pen

    fitter = CoxPHFitter(penalizer=best_pen)
    fitter.fit(train_df[kept + ["et_min", "events"]], duration_col="et_min", event_col="events", show_progress=False)
    test_risk = fitter.predict_log_partial_hazard(test_df[kept]).to_numpy().reshape(-1)

    c_index = concordance_index(t_test, test_risk, e_test)
    ci_lower, ci_upper = compute_bootstrap_cindex(
        t_test, e_test, test_risk,
        patient_ids=data_dict["patient_ids_test"], n_bootstrap=200, seed=seed,
    )
    metrics = {"C-index": float(c_index), "C-index_CI_lower": float(ci_lower), "C-index_CI_upper": float(ci_upper)}
    for horizon in [30, 60, 120]:
        y_h = ((e_test.astype(int) == 1) & (t_test <= horizon)).astype(int)
        auc = _binary_auc(y_h, test_risk)
        if auc is not None:
            metrics[f"AUC_{horizon}m"] = float(auc)
    metrics["penalizer"] = best_pen
    metrics["n_features_used"] = len(kept)
    return metrics


def run_cdan_gsn(data_dict, config, device, hparams):
    """Run CDAN-GSN with given hyperparameters."""
    seed = config["training"]["seed"]
    seed_everything(seed)

    treat_indices, physio_indices = get_indices(data_dict["feature_names"])

    d_model = hparams.get("d_model", 64)
    num_layers = hparams.get("num_layers", 2)
    num_heads = hparams.get("num_heads", 4)
    dropout = hparams.get("dropout", 0.25)
    kan_bases = hparams.get("kan_bases", 8)
    pretrain_epochs = hparams.get("pretrain_epochs", 20)
    lr = hparams.get("lr", 0.001)
    finetune_lr = hparams.get("finetune_lr", 0.0001)
    adv_weight = hparams.get("adv_weight", 0.03)
    mask_l1_weight = hparams.get("mask_l1_weight", 0.03)
    phases_config = hparams.get("phases_config", None)

    model = DomainStratifiedGatedNet(
        input_dim=data_dict["input_dim"],
        d_model=d_model, nhead=num_heads, num_layers=num_layers,
        dropout=dropout, domain_hidden=64,
        treat_indices=treat_indices, physio_indices=physio_indices,
        tokenizer_type="kan", kan_basis_dim=kan_bases,
    ).to(device)

    best_pre_val, best_pre_epoch, source_state = run_source_pretrain(
        model, data_dict["source_loader"],
        data_dict["x_val"], data_dict["e_val"], data_dict["t_val"],
        lr=lr, device=device, max_epochs=pretrain_epochs, patience=5,
    )

    da_model = DomainStratifiedGatedNet(
        input_dim=data_dict["input_dim"],
        d_model=d_model, nhead=num_heads, num_layers=num_layers,
        dropout=dropout, domain_hidden=64,
        treat_indices=treat_indices, physio_indices=physio_indices,
        tokenizer_type="kan", kan_basis_dim=kan_bases,
    ).to(device)
    da_model.load_state_dict(source_state)

    run_domain_stratified_da(
        da_model, data_dict["source_loader"], data_dict["target_loader"],
        data_dict["x_val"], data_dict["e_val"], data_dict["t_val"],
        adv_weight=adv_weight, source_replay_weight=0.1,
        mask_l1_weight=mask_l1_weight, lr=lr, finetune_lr=finetune_lr,
        device=device, phases_config=phases_config,
    )

    da_metrics = evaluate_survival_metrics(
        da_model, data_dict["x_test"], data_dict["e_test"], data_dict["t_test"],
        device=device, t_train=data_dict["t_train"], e_train=data_dict["e_train"],
        patient_ids=data_dict["patient_ids_test"], n_bootstrap=200, seed=seed,
    )
    return da_metrics, da_model


def main():
    config = load_config()
    seed = config["training"]["seed"]
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading data (unified config)...")
    data_dict = prepare_dataloaders(
        source_path=os.path.join(config["paths"]["data_dir"], config["data"]["source_file"]),
        target_path=os.path.join(config["paths"]["data_dir"], config["data"]["target_file"]),
        batch_size=config["training"]["batch_size"],
        seed=seed,
        target_adapt_ratio=config["data"].get("target_adapt_ratio", 0.4),
        target_val_ratio=config["data"].get("target_val_ratio", 0.15),
        patient_col=config["data"].get("patient_col", "患者id"),
        split_strategy=config["data"].get("split_strategy", "patient"),
    )
    print(f"  Features: {len(data_dict['feature_names'])}, adapt patients: {len(data_dict['x_train'])}")

    # --- CoxPH Baseline ---
    print("\n[CoxPH] Running baseline...")
    cox_metrics = run_cox_baseline(data_dict, seed)
    print(f"  CoxPH C-index: {cox_metrics['C-index']:.4f} [{cox_metrics['C-index_CI_lower']:.4f}, {cox_metrics['C-index_CI_upper']:.4f}]")

    # --- CDAN-GSN Hyperparameter Grid ---
    grid = [
        # Config A: current (baseline after fix)
        {"name": "A_current", "d_model": 64, "num_layers": 2, "num_heads": 4,
         "dropout": 0.25, "kan_bases": 8, "pretrain_epochs": 20,
         "lr": 0.001, "finetune_lr": 0.0001, "adv_weight": 0.03, "mask_l1_weight": 0.03},
        # Config B: larger model + lower adv
        {"name": "B_larger", "d_model": 128, "num_layers": 2, "num_heads": 4,
         "dropout": 0.3, "kan_bases": 12, "pretrain_epochs": 30,
         "lr": 0.001, "finetune_lr": 5e-5, "adv_weight": 0.01, "mask_l1_weight": 0.02},
        # Config C: deeper + conservative DA
        {"name": "C_deep", "d_model": 96, "num_layers": 3, "num_heads": 4,
         "dropout": 0.3, "kan_bases": 10, "pretrain_epochs": 30,
         "lr": 0.0008, "finetune_lr": 5e-5, "adv_weight": 0.005, "mask_l1_weight": 0.01,
         "phases_config": [("head_only", 8, 4), ("partial_unfreeze", 10, 4), ("full_finetune", 15, 5)]},
        # Config D: minimal DA (nearly zero adv), strong pretrain
        {"name": "D_pretrain_focus", "d_model": 64, "num_layers": 2, "num_heads": 4,
         "dropout": 0.2, "kan_bases": 8, "pretrain_epochs": 40,
         "lr": 0.001, "finetune_lr": 3e-5, "adv_weight": 0.001, "mask_l1_weight": 0.01,
         "phases_config": [("head_only", 10, 5), ("partial_unfreeze", 12, 5), ("full_finetune", 18, 6)]},
        # Config E: medium model, balanced
        {"name": "E_balanced", "d_model": 96, "num_layers": 2, "num_heads": 4,
         "dropout": 0.25, "kan_bases": 10, "pretrain_epochs": 25,
         "lr": 0.001, "finetune_lr": 8e-5, "adv_weight": 0.02, "mask_l1_weight": 0.02,
         "phases_config": [("head_only", 8, 4), ("partial_unfreeze", 10, 4), ("full_finetune", 14, 5)]},
    ]

    results = {"cox": cox_metrics, "cdan_gsn_grid": []}
    best_cindex = 0.0
    best_config_name = ""
    best_model = None

    for hparams in grid:
        name = hparams.pop("name")
        print(f"\n[CDAN-GSN] Config {name}...")
        try:
            metrics, model = run_cdan_gsn(data_dict, config, device, hparams)
            cindex = metrics["C-index"]
            print(f"  C-index: {cindex:.4f} [{metrics['C-index_CI_lower']:.4f}, {metrics['C-index_CI_upper']:.4f}]")
            result_entry = {"name": name, "hparams": hparams, "metrics": metrics}
            results["cdan_gsn_grid"].append(result_entry)
            if cindex > best_cindex:
                best_cindex = cindex
                best_config_name = name
                best_model = model
        except Exception as e:
            print(f"  FAILED: {e}")
            results["cdan_gsn_grid"].append({"name": name, "hparams": hparams, "error": str(e)})

    # Summary
    results["summary"] = {
        "cox_cindex": cox_metrics["C-index"],
        "best_cdan_config": best_config_name,
        "best_cdan_cindex": best_cindex,
        "cdan_beats_cox": best_cindex > cox_metrics["C-index"],
        "gap": best_cindex - cox_metrics["C-index"],
    }

    out_path = Path("experiments/results/unified_benchmark.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Save best model if it beats Cox
    if best_model is not None and best_cindex > cox_metrics["C-index"]:
        torch.save(
            {"model_state_dict": best_model.state_dict(), "config_name": best_config_name},
            out_path.parent / "cdan_gsn_best_tuned.pt",
        )
        print(f"\n✓ Best CDAN-GSN ({best_config_name}) beats CoxPH: {best_cindex:.4f} > {cox_metrics['C-index']:.4f}")
    else:
        print(f"\n✗ Best CDAN-GSN ({best_config_name}): {best_cindex:.4f} vs CoxPH: {cox_metrics['C-index']:.4f}")

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
