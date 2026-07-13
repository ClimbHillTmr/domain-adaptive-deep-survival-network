"""
Multi-seed variance evaluation.

Runs the full training pipeline (source pretrain + domain-adaptive fine-tuning)
with N different random seeds and reports mean ± std of key discrimination
metrics.  This is supplementary evidence that the reported +0.004 C-index
margin over CoxPH is stable across initializations rather than a lucky draw.

Usage
-----
    PYTHONPATH=. python src/evaluate/multiseed_eval.py [--seeds 42 7 13 99 2024]

Outputs
-------
    experiments/results/multiseed_results.json  — per-seed and aggregate stats
    experiments/results/multiseed_summary.csv   — table-ready summary
"""

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import Ridge
from lifelines import CoxPHFitter

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.dataset import prepare_dataloaders
from src.evaluate.metrics import concordance_index, compute_bootstrap_cindex
from src.models.cdan_gsn import DomainStratifiedGatedNet
from src.reproducibility import seed_everything
from src.train.trainer import run_domain_stratified_da, run_source_pretrain


DEFAULT_SEEDS = [42, 7, 13, 99, 2024]


def load_config(path: str = "conf/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_indices(feature_names):
    treat_features = [
        "超滤比", "超滤率_绝对", "超滤率_体重归一化", "超滤量MAX",
        "透析液电导率", "透析液钙浓度", "抗凝剂类型_code",
        "瘘管位置_code", "瘘管类型_code", "透析方式_code",
    ]
    treat_idx, physio_idx = [], []
    for i, name in enumerate(feature_names):
        if any(t in name for t in treat_features):
            treat_idx.append(i)
        else:
            physio_idx.append(i)
    return treat_idx, physio_idx


def predict_risk(model, x, device, batch_size=1024):
    model.eval()
    preds = []
    with torch.no_grad():
        for s in range(0, len(x), batch_size):
            xb = torch.tensor(x[s: s + batch_size], dtype=torch.float32, device=device)
            _, h, _, _ = model(xb, grl_coeff=None)
            preds.append(h.squeeze(-1).cpu().numpy())
    return np.concatenate(preds)


def run_coxph_baseline(x_train, e_train, t_train, x_test, e_test, t_test, feature_names):
    """Locally-fitted CoxPH on the target adaptation training set."""
    df_tr = pd.DataFrame(x_train, columns=feature_names)
    df_tr["duration"] = t_train
    df_tr["event"]    = e_train.astype(int)
    df_te = pd.DataFrame(x_test, columns=feature_names)

    try:
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(df_tr, duration_col="duration", event_col="event")
        risk_scores = cph.predict_log_partial_hazard(df_te).values
    except Exception:
        # lifelines may fail on small / degenerate datasets — fall back gracefully
        return None

    return concordance_index(t_test, risk_scores, e_test)


def train_cdan(config, data_dict, device, seed):
    """Full CDAN-GSN pipeline. Returns trained model."""
    treat_idx, physio_idx = get_indices(data_dict["feature_names"])

    model = DomainStratifiedGatedNet(
        input_dim=data_dict["input_dim"],
        d_model=config["model"]["d_model"],
        nhead=config["model"]["num_heads"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
        domain_hidden=config["model"]["domain_hidden"],
        treat_indices=treat_idx,
        physio_indices=physio_idx,
        tokenizer_type="kan",
        kan_basis_dim=config["model"]["kan_bases"],
    ).to(device)

    _, _, src_state = run_source_pretrain(
        model,
        data_dict["source_loader"],
        data_dict["x_val"], data_dict["e_val"], data_dict["t_val"],
        lr=config["training"]["learning_rate"],
        device=device,
        max_epochs=config["training"]["pretrain_epochs"],
        x_source_val=data_dict.get("x_source_val"),
        e_source_val=data_dict.get("e_source_val"),
        t_source_val=data_dict.get("t_source_val"),
    )

    da_model = copy.deepcopy(model)
    da_model.load_state_dict(src_state)

    # Zero-shot metrics (before domain adaptation)
    zs_risk = predict_risk(da_model, data_dict["x_test"], device)
    zs_c    = concordance_index(data_dict["t_test"], zs_risk, data_dict["e_test"])

    run_domain_stratified_da(
        da_model,
        data_dict["source_loader"],
        data_dict["target_loader"],
        data_dict["x_val"], data_dict["e_val"], data_dict["t_val"],
        adv_weight=config["training"]["adv_weight"],
        source_replay_weight=0.1,
        mask_l1_weight=config["training"]["mask_l1_weight"],
        lr=config["training"]["learning_rate"],
        finetune_lr=config["training"].get("finetune_lr", 3e-5),
        device=device,
    )
    return da_model, zs_c


def main(seeds=None):
    if seeds is None:
        seeds = DEFAULT_SEEDS

    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Seeds: {seeds}")

    out_dir = Path(config["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    all_records = []

    for seed in seeds:
        seed_everything(seed)
        print(f"\n{'='*60}")
        print(f"  Seed = {seed}")
        print(f"{'='*60}")

        data_dict = prepare_dataloaders(
            source_path=os.path.join(
                config["paths"]["data_dir"], config["data"]["source_file"]
            ),
            target_path=os.path.join(
                config["paths"]["data_dir"], config["data"]["target_file"]
            ),
            batch_size=config["training"]["batch_size"],
            seed=seed,
            target_adapt_ratio=config["data"].get("target_adapt_ratio", 0.40),
            target_val_ratio=config["data"].get("target_val_ratio", 0.15),
            patient_col=config["data"].get("patient_col", "患者id"),
            split_strategy=config["data"].get("split_strategy", "patient"),
        )

        # CoxPH baseline (re-trained with this seed's data split)
        cox_c = run_coxph_baseline(
            data_dict["x_train"], data_dict["e_train"], data_dict["t_train"],
            data_dict["x_test"],  data_dict["e_test"],  data_dict["t_test"],
            data_dict["feature_names"],
        )
        print(f"  CoxPH C-index: {cox_c:.4f}" if cox_c else "  CoxPH: failed")

        # Full CDAN-GSN
        da_model, zs_c = train_cdan(config, data_dict, device, seed)

        risk = predict_risk(da_model, data_dict["x_test"], device)
        c    = concordance_index(data_dict["t_test"], risk, data_dict["e_test"])
        ci_lo, ci_hi = compute_bootstrap_cindex(
            data_dict["t_test"], data_dict["e_test"], risk,
            patient_ids=data_dict["patient_ids_test"],
            n_bootstrap=200,
            seed=seed,
        )

        rec = {
            "seed": int(seed),
            "cdan_cindex":      float(c),
            "cdan_ci_lower":    float(ci_lo),
            "cdan_ci_upper":    float(ci_hi),
            "zeroshot_cindex":  float(zs_c),
            "coxph_cindex":     float(cox_c) if cox_c is not None else None,
            "cdan_vs_cox_delta": float(c - cox_c) if cox_c is not None else None,
        }
        all_records.append(rec)
        print(
            f"  CDAN-GSN C-index: {c:.4f} [{ci_lo:.3f}, {ci_hi:.3f}]  "
            f"|  Zero-shot: {zs_c:.4f}  |  Δ vs CoxPH: "
            + (f"{c - cox_c:+.4f}" if cox_c else "N/A")
        )

    # Aggregate statistics
    cdan_vals = [r["cdan_cindex"] for r in all_records]
    cox_vals  = [r["coxph_cindex"] for r in all_records if r["coxph_cindex"] is not None]
    zs_vals   = [r["zeroshot_cindex"] for r in all_records]
    delta_vals = [r["cdan_vs_cox_delta"] for r in all_records if r["cdan_vs_cox_delta"] is not None]

    summary = {
        "n_seeds": len(seeds),
        "seeds": seeds,
        "CDAN_GSN": {
            "mean":   float(np.mean(cdan_vals)),
            "std":    float(np.std(cdan_vals, ddof=1)),
            "min":    float(np.min(cdan_vals)),
            "max":    float(np.max(cdan_vals)),
        },
        "CoxPH_local": {
            "mean":   float(np.mean(cox_vals)) if cox_vals else None,
            "std":    float(np.std(cox_vals, ddof=1)) if len(cox_vals) > 1 else None,
        },
        "Zero_shot": {
            "mean":   float(np.mean(zs_vals)),
            "std":    float(np.std(zs_vals, ddof=1)),
        },
        "delta_CDAN_vs_CoxPH": {
            "mean":   float(np.mean(delta_vals)) if delta_vals else None,
            "std":    float(np.std(delta_vals, ddof=1)) if len(delta_vals) > 1 else None,
            "positive_in": int(sum(d > 0 for d in delta_vals)),
            "out_of":      len(delta_vals),
        },
    }

    output = {"per_seed": all_records, "summary": summary}
    json_path = out_dir / "multiseed_results.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nFull results saved to {json_path}")

    # CSV table
    df = pd.DataFrame(all_records)
    df.loc[len(df)] = {
        "seed": "MEAN ± STD",
        "cdan_cindex":     f"{np.mean(cdan_vals):.4f} ± {np.std(cdan_vals, ddof=1):.4f}",
        "coxph_cindex":    f"{np.mean(cox_vals):.4f} ± {np.std(cox_vals, ddof=1):.4f}" if len(cox_vals) > 1 else "",
        "zeroshot_cindex": f"{np.mean(zs_vals):.4f} ± {np.std(zs_vals, ddof=1):.4f}",
        "cdan_vs_cox_delta": f"{np.mean(delta_vals):+.4f} ± {np.std(delta_vals, ddof=1):.4f}" if len(delta_vals) > 1 else "",
        "cdan_ci_lower": "", "cdan_ci_upper": "",
    }
    csv_path = out_dir / "multiseed_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Summary table saved to {csv_path}")

    # Print summary
    print("\n" + "=" * 60)
    print(f"  Multi-seed summary ({len(seeds)} seeds)")
    print("=" * 60)
    print(f"  CDAN-GSN  :  {np.mean(cdan_vals):.4f} ± {np.std(cdan_vals, ddof=1):.4f}")
    if cox_vals:
        print(f"  CoxPH     :  {np.mean(cox_vals):.4f} ± {np.std(cox_vals, ddof=1):.4f}")
        print(f"  Δ(CDAN-Cox): {np.mean(delta_vals):+.4f} ± {np.std(delta_vals, ddof=1):.4f}  "
              f"  ({sum(d > 0 for d in delta_vals)}/{len(delta_vals)} positive)")
    print(f"  Zero-shot :  {np.mean(zs_vals):.4f} ± {np.std(zs_vals, ddof=1):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                        help="Random seeds to evaluate (default: 42 7 13 99 2024)")
    args = parser.parse_args()
    main(args.seeds)
