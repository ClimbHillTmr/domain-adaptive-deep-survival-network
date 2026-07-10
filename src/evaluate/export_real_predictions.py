import json
import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.dataset import prepare_dataloaders
from src.evaluate.dca_analysis import calculate_net_benefit
from src.models.cdan_gsn import DomainStratifiedGatedNet
from src.reproducibility import record_environment, seed_everything
from src.train.trainer import run_domain_stratified_da, run_source_pretrain
from src.visualization.journal_style import get_color_palette, remove_top_right_spines, set_journal_style


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


def predict_risk_scores(model, x_eval, device="cpu", batch_size=1024):
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(x_eval), batch_size):
            x_batch = torch.tensor(x_eval[start:start + batch_size], dtype=torch.float32, device=device)
            _, hazard, _, _ = model(x_batch, grl_coeff=None)
            preds.append(hazard.squeeze(-1).detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def breslow_baseline_hazard(times, events, risk_scores):
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    risk_scores = np.asarray(risk_scores, dtype=float)
    exp_risk = np.exp(risk_scores)

    event_times = np.sort(np.unique(times[events == 1]))
    cum_hazard = []
    cumulative = 0.0
    for t in event_times:
        d_t = np.sum((times == t) & (events == 1))
        risk_set = np.sum(exp_risk[times >= t])
        if risk_set <= 0:
            continue
        cumulative += d_t / risk_set
        cum_hazard.append(cumulative)
    return event_times, np.asarray(cum_hazard)


def cumulative_baseline_hazard_at(horizon, event_times, cum_hazard):
    idx = np.searchsorted(event_times, horizon, side="right") - 1
    if idx < 0:
        return 0.0
    return float(cum_hazard[idx])


def event_probability_by_horizon(risk_scores, event_times, cum_hazard, horizon):
    h0 = cumulative_baseline_hazard_at(horizon, event_times, cum_hazard)
    surv = np.exp(-h0 * np.exp(risk_scores))
    return 1.0 - surv


def plot_calibration_panel(ax, y_true, y_prob, title, color):
    bins = np.linspace(0, 1, 11)
    y_prob = np.clip(y_prob, 1e-6, 1 - 1e-6)
    bin_idx = np.digitize(y_prob, bins) - 1
    pred_means, obs_rates = [], []
    for i in range(10):
        mask = bin_idx == i
        if np.sum(mask) == 0:
            continue
        pred_means.append(np.mean(y_prob[mask]))
        obs_rates.append(np.mean(y_true[mask]))

    ax.plot([0, 1], [0, 1], linestyle="--", color="#7F8C8D", linewidth=1.5, label="Ideal")
    ax.plot(pred_means, obs_rates, marker="o", color=color, linewidth=2.4, label="Observed")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed event rate")
    ax.legend(frameon=False, loc="lower right")
    remove_top_right_spines(ax)


def plot_dca_panel(ax, y_true, y_prob, title, color):
    thresholds = np.linspace(0.01, 0.6, 60)
    prevalence = np.mean(y_true)
    nb_all = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))
    nb_model = calculate_net_benefit(y_true, y_prob, thresholds)

    ax.plot(thresholds, nb_all, color="#95A5A6", linewidth=1.7, label="Treat all")
    ax.plot(thresholds, np.zeros_like(thresholds), color="black", linewidth=1.5, label="Treat none")
    ax.plot(thresholds, nb_model, color=color, linewidth=2.5, label="CDAN-GSN")
    ax.set_xlim(0.0, 0.6)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.legend(frameon=False, loc="upper right")
    remove_top_right_spines(ax)


def calibration_metrics(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
    observed_rate = float(np.mean(y_true))
    mean_predicted_rate = float(np.mean(y_prob))
    brier = float(np.mean((y_true - y_prob) ** 2))
    x = np.log(y_prob / (1 - y_prob))
    if len(np.unique(y_true)) < 2 or np.std(x) == 0:
        intercept = None
        slope = None
    else:
        slope, intercept = np.polyfit(x, y_true, deg=1)
        intercept = float(intercept)
        slope = float(slope)
    return {
        "observed_rate": observed_rate,
        "mean_predicted_rate": mean_predicted_rate,
        "brier": brier,
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def write_calibration_dca_metrics(pred_df, out_path):
    thresholds = np.linspace(0.01, 0.6, 60)
    metrics = {}
    for horizon in [30, 60, 120]:
        y_true = pred_df[f"event_by_{horizon}m"].values
        y_prob = pred_df[f"event_prob_{horizon}m"].values
        horizon_metrics = calibration_metrics(y_true, y_prob)
        nb = calculate_net_benefit(y_true, y_prob, thresholds)
        horizon_metrics["net_benefit_threshold_range"] = {
            "min": float(thresholds.min()),
            "max": float(thresholds.max()),
            "n_thresholds": int(len(thresholds)),
            "max_net_benefit": float(np.max(nb)),
        }
        metrics[f"{horizon}m"] = horizon_metrics
    out_path.write_text(json.dumps(metrics, indent=2))


def main():
    config = load_config()
    seed_everything(config["training"]["seed"])
    record_environment()

    set_journal_style("nature")
    mpl.rcParams["pdf.fonttype"] = 3
    mpl.rcParams["ps.fonttype"] = 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    source_path = os.path.join(config["paths"]["data_dir"], config["data"]["source_file"])
    target_path = os.path.join(config["paths"]["data_dir"], config["data"]["target_file"])

    data_dict = prepare_dataloaders(
        source_path=source_path,
        target_path=target_path,
        batch_size=config["training"]["batch_size"],
        seed=config["training"]["seed"],
        target_adapt_ratio=config["data"].get("target_adapt_ratio", 0.2),
        target_val_ratio=config["data"].get("target_val_ratio", 0.2),
        patient_col=config["data"].get("patient_col", "患者id"),
        split_strategy=config["data"].get("split_strategy", "patient"),
    )
    treat_indices, physio_indices = get_indices(data_dict["feature_names"])

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
        kan_basis_dim=config["model"]["kan_bases"],
    ).to(device)

    da_model = DomainStratifiedGatedNet(
        input_dim=data_dict["input_dim"],
        d_model=config["model"]["d_model"],
        nhead=config["model"]["num_heads"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
        domain_hidden=config["model"]["domain_hidden"],
        treat_indices=treat_indices,
        physio_indices=physio_indices,
        tokenizer_type="kan",
        kan_basis_dim=config["model"]["kan_bases"],
    ).to(device)
    checkpoint_path = Path(config["paths"]["output_dir"]) / "cdan_gsn_final.pt"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        da_model.load_state_dict(state_dict)
    else:
        _, _, source_state = run_source_pretrain(
            model,
            data_dict["source_loader"],
            data_dict["x_val"],
            data_dict["e_val"],
            data_dict["t_val"],
            lr=config["training"]["learning_rate"],
            device=device,
            max_epochs=config["training"]["pretrain_epochs"],
        )
        da_model.load_state_dict(source_state)
        run_domain_stratified_da(
            da_model,
            data_dict["source_loader"],
            data_dict["target_loader"],
            data_dict["x_val"],
            data_dict["e_val"],
            data_dict["t_val"],
            adv_weight=config["training"]["adv_weight"],
            source_replay_weight=0.1,
            mask_l1_weight=config["training"]["mask_l1_weight"],
            lr=config["training"]["learning_rate"],
            finetune_lr=config["training"].get("finetune_lr", 0.0001),
            device=device,
        )

    train_scores = predict_risk_scores(da_model, data_dict["x_train"], device=device)
    test_scores = predict_risk_scores(da_model, data_dict["x_test"], device=device)
    event_times, cum_hazard = breslow_baseline_hazard(data_dict["t_train"], data_dict["e_train"], train_scores)

    pred_df = data_dict["df_target"].iloc[data_dict["idx_test"]].copy().reset_index(drop=True)
    pred_df["risk_score"] = test_scores
    for horizon in [30, 60, 120]:
        pred_df[f"event_prob_{horizon}m"] = event_probability_by_horizon(test_scores, event_times, cum_hazard, horizon)
        pred_df[f"event_by_{horizon}m"] = (
            (pred_df["events"].astype(int) == 1) & (pred_df["et_min"].astype(float) <= horizon)
        ).astype(int)

    out_dir = Path("experiments/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "real_test_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    write_calibration_dca_metrics(pred_df, out_dir / "calibration_dca_metrics.json")

    torch.save(da_model.state_dict(), out_dir / "cdan_gsn_real_export.pt")
    with open(out_dir / "prediction_metadata.json", "w") as f:
        json.dump(
            {
                "source_path": source_path,
                "target_path": target_path,
                "n_test": int(len(pred_df)),
                "horizons": [30, 60, 120],
            },
            f,
            indent=2,
        )

    fig_dir = Path("figures/Main_Figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    colors = get_color_palette(3, "clinical")

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.2))
    plot_calibration_panel(
        axes[0],
        pred_df["event_by_60m"].values,
        pred_df["event_prob_60m"].values,
        "a  Calibration at 60 min",
        colors[0],
    )
    plot_calibration_panel(
        axes[1],
        pred_df["event_by_120m"].values,
        pred_df["event_prob_120m"].values,
        "b  Calibration at 120 min",
        colors[1],
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.15, wspace=0.25)
    fig.savefig(fig_dir / "Fig7_Calibration_Real.pdf", dpi=600, bbox_inches=None, facecolor="white")
    fig.savefig(fig_dir / "Fig7_Calibration_Real.png", dpi=600, bbox_inches=None, facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.2))
    plot_dca_panel(
        axes[0],
        pred_df["event_by_60m"].values,
        pred_df["event_prob_60m"].values,
        "a  Decision curve at 60 min",
        colors[0],
    )
    plot_dca_panel(
        axes[1],
        pred_df["event_by_120m"].values,
        pred_df["event_prob_120m"].values,
        "b  Decision curve at 120 min",
        colors[1],
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.15, wspace=0.28)
    fig.savefig(fig_dir / "Fig8_DCA_Real.pdf", dpi=600, bbox_inches=None, facecolor="white")
    fig.savefig(fig_dir / "Fig8_DCA_Real.png", dpi=600, bbox_inches=None, facecolor="white")
    plt.close(fig)

    print(f"Saved real prediction file to {pred_path}")


if __name__ == "__main__":
    main()
