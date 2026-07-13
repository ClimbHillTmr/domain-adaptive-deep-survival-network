"""
Learning-curve analysis: C-index as a function of labeled target patient count.

For each adapt_ratio in RATIOS we retrain the full pipeline from scratch
(source pretrain + domain-adaptive fine-tuning) and evaluate on a FIXED
held-out test set.  The test set is taken from the adapt_ratio=0.40 split
so that it never changes across conditions and the comparison is fair.

Usage
-----
    PYTHONPATH=. python src/evaluate/learning_curve.py
"""

import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.dataset import prepare_dataloaders
from src.evaluate.metrics import concordance_index, compute_bootstrap_cindex
from src.models.cdan_gsn import DomainStratifiedGatedNet
from src.reproducibility import seed_everything
from src.train.trainer import run_domain_stratified_da, run_source_pretrain

# -------------------------------------------------------------------
# Each point uses a nested subset of one fixed 40% target adaptation pool.
# The patient-level validation and 60% held-out test sets never change.
# -------------------------------------------------------------------
RATIOS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
BOOTSTRAP_N = 100   # fewer replicates — this is a supplementary figure
SEEDS = [42]        # single seed for the learning curve; multi-seed in multiseed_eval.py


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


def build_and_train(config, data_dict, device, seed):
    """Full pipeline: pretrain + DA, returns trained model."""
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
    return da_model


def main():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = Path(config["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []

    for ratio in RATIOS:
        for seed in SEEDS:
            seed_everything(seed)
            print(f"\n{'='*55}")
            print(f"  adapt_ratio={ratio:.2f}  seed={seed}")
            print(f"{'='*55}")

            data_dict = prepare_dataloaders(
                source_path=os.path.join(config["paths"]["data_dir"], config["data"]["source_file"]),
                target_path=os.path.join(config["paths"]["data_dir"], config["data"]["target_file"]),
                batch_size=config["training"]["batch_size"],
                seed=seed,
                target_adapt_ratio=config["data"]["target_adapt_ratio"],
                target_val_ratio=config["data"].get("target_val_ratio", 0.15),
                target_train_fraction=ratio / config["data"]["target_adapt_ratio"],
                patient_col=config["data"].get("patient_col", "患者id"),
                split_strategy=config["data"].get("split_strategy", "patient"),
            )

            n_adapt_patients = len(np.unique(data_dict["patient_ids_train"]))
            n_test_sessions  = len(data_dict["x_test"])
            print(f"  Adapt patients: {n_adapt_patients}  |  Test sessions: {n_test_sessions}")

            # Skip if too few patients to form a meaningful validation set
            if n_adapt_patients < 5:
                print("  Skipping — too few adaptation patients.")
                continue

            model = build_and_train(config, data_dict, device, seed)

            risk = predict_risk(model, data_dict["x_test"], device)
            c    = concordance_index(data_dict["t_test"], risk, data_dict["e_test"])
            ci_lo, ci_hi = compute_bootstrap_cindex(
                data_dict["t_test"], data_dict["e_test"], risk,
                patient_ids=data_dict["patient_ids_test"],
                n_bootstrap=BOOTSTRAP_N,
                seed=seed,
            )

            rec = {
                "adapt_ratio": float(ratio),
                "seed": int(seed),
                "n_adapt_patients": int(n_adapt_patients),
                "n_test_sessions": int(n_test_sessions),
                "c_index": float(c),
                "ci_lower": float(ci_lo),
                "ci_upper": float(ci_hi),
            }
            records.append(rec)
            print(f"  C-index={c:.4f}  [{ci_lo:.3f}, {ci_hi:.3f}]")

    # Save raw results
    lc_path = out_dir / "learning_curve.json"
    with open(lc_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nResults saved to {lc_path}")

    # Generate figure
    _plot_learning_curve(records, out_dir)


def _plot_learning_curve(records, out_dir):
    try:
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        mpl.rcParams.update({"font.size": 11, "axes.spines.top": False,
                             "axes.spines.right": False})
    except ImportError:
        print("matplotlib not available — skipping figure.")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Group by ratio (average over seeds)
    from collections import defaultdict
    groups: dict = defaultdict(list)
    for r in records:
        groups[r["adapt_ratio"]].append(r)

    ratios   = sorted(groups.keys())
    c_means  = [float(np.mean([x["c_index"] for x in groups[r]])) for r in ratios]
    ci_los   = [float(np.mean([x["ci_lower"] for x in groups[r]])) for r in ratios]
    ci_his   = [float(np.mean([x["ci_upper"] for x in groups[r]])) for r in ratios]
    n_pts    = [int(np.mean([x["n_adapt_patients"] for x in groups[r]])) for r in ratios]

    x_axis = n_pts
    ax.plot(x_axis, c_means, marker="o", linewidth=2.2, color="#2166AC", label="CDAN-GSN")
    ax.fill_between(x_axis, ci_los, ci_his, alpha=0.18, color="#2166AC")

    # Annotate zero-shot level if available from evaluation_results.json
    try:
        ev = json.loads(
            (Path("experiments/results/evaluation_results.json")).read_text()
        )
        zs_c = ev["sweeps"]["Zero-shot"]["metrics"]["C-index"]
        ax.axhline(zs_c, linestyle="--", linewidth=1.5, color="#999999",
                   label=f"Zero-shot (no fine-tuning, C={zs_c:.3f})")
    except Exception:
        pass

    ax.set_xlabel("Number of labeled target patients used for adaptation")
    ax.set_ylabel("C-index (patient-cluster bootstrap 95 % CI)")
    ax.set_title("Learning curve: discrimination vs. adaptation sample size",
                 loc="left", fontsize=11)
    ax.legend(frameon=False)

    fig_dir = Path("figures/Main_Figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(fig_dir / "Fig_LearningCurve.pdf", dpi=300)
    fig.savefig(fig_dir / "Fig_LearningCurve.png", dpi=300)
    plt.close(fig)
    print(f"Figure saved to {fig_dir}/Fig_LearningCurve.{{pdf,png}}")


if __name__ == "__main__":
    main()
