"""Build descriptive clinical-utility outputs from one registered frozen run.

No model fitting or data regeneration is performed. Alert operating points use
thresholds selected on validation patients and stored in evaluation.json. DCA is
a threshold-grid sensitivity analysis until clinical thresholds are prespecified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "experiments" / "binary_results" / "binary_20260719_044949_c6c955a30fc2"
DEFAULT_OUTPUT = ROOT / "clinical_utility_report"
MODELS = (
    ("source_mlp", "Source MLP", "#6B7280", "--", "o"),
    ("updated_mlp", "Updated MLP", "#009E73", "-", "s"),
    ("local_logistic", "Target-local logistic", "#CC79A7", ":", "D"),
)


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _calibration(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    clipped = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, max_iter=2000, solver="lbfgs").fit(logit, y)
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def _calibration_bins(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    frame = pd.DataFrame({"outcome": y, "probability": p})
    frame["bin"] = pd.qcut(frame["probability"], bins, duplicates="drop")
    return frame.groupby("bin", observed=True).agg(
        n=("outcome", "size"), mean_predicted=("probability", "mean"), observed=("outcome", "mean")
    ).reset_index(drop=True)


def _ece(calibration: pd.DataFrame) -> float:
    weights = calibration["n"] / calibration["n"].sum()
    return float((weights * (calibration["observed"] - calibration["mean_predicted"]).abs()).sum())


def _confusion(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = p >= threshold
    tp = int(((y == 1) & predicted).sum())
    fn = int(((y == 1) & ~predicted).sum())
    fp = int(((y == 0) & predicted).sum())
    tn = int(((y == 0) & ~predicted).sum())
    n = len(y)
    return {
        "threshold": threshold, "sessions": n, "events": int(y.sum()),
        "sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp),
        "ppv": tp / (tp + fp) if tp + fp else float("nan"),
        "npv": tn / (tn + fn) if tn + fn else float("nan"),
        "events_detected_per_1000": 1000 * tp / n,
        "false_alerts_per_1000": 1000 * fp / n,
        "total_alerts_per_1000": 1000 * (tp + fp) / n,
    }


def _patient_cluster_indices(
    patient_ids: np.ndarray, *, replicates: int, seed: int
) -> list[np.ndarray]:
    patient_ids = np.asarray(patient_ids).astype(str)
    patients = np.unique(patient_ids)
    patient_rows = {patient: np.flatnonzero(patient_ids == patient) for patient in patients}
    rng = np.random.default_rng(seed)
    return [
        np.concatenate([patient_rows[patient] for patient in rng.choice(patients, len(patients), replace=True)])
        for _ in range(replicates)
    ]


def _interval(values: list[float]) -> dict[str, float | int]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {"lower": float("nan"), "upper": float("nan"), "valid_replicates": 0}
    return {
        "lower": float(np.quantile(finite, 0.025)),
        "upper": float(np.quantile(finite, 0.975)),
        "valid_replicates": int(len(finite)),
    }


def _clinical_bootstrap(
    y: np.ndarray, p: np.ndarray, threshold: float, indices: list[np.ndarray]
) -> dict[str, dict[str, float | int]]:
    metric_names = (
        "calibration_intercept", "calibration_slope", "ece_10_quantile_bins", "brier",
        "sensitivity", "specificity", "ppv", "npv", "events_detected_per_1000",
        "false_alerts_per_1000", "total_alerts_per_1000",
    )
    samples: dict[str, list[float]] = {name: [] for name in metric_names}
    for sampled_indices in indices:
        sampled_y, sampled_p = y[sampled_indices], p[sampled_indices]
        if len(np.unique(sampled_y)) < 2:
            continue
        intercept, slope = _calibration(sampled_y, sampled_p)
        confusion = _confusion(sampled_y, sampled_p, threshold)
        values = {
            "calibration_intercept": intercept,
            "calibration_slope": slope,
            "ece_10_quantile_bins": _ece(_calibration_bins(sampled_y, sampled_p)),
            "brier": float(np.mean((sampled_p - sampled_y) ** 2)),
            **confusion,
        }
        for name in metric_names:
            samples[name].append(float(values[name]))
    return {name: _interval(values) for name, values in samples.items()}


def _net_benefit(y: np.ndarray, p: np.ndarray, threshold: float) -> float:
    predicted = p >= threshold
    tp = ((y == 1) & predicted).sum()
    fp = ((y == 0) & predicted).sum()
    return float(tp / len(y) - fp / len(y) * threshold / (1 - threshold))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260715)
    args = parser.parse_args()
    run_dir, output = args.run_dir.resolve(), args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    evaluation = json.loads((run_dir / "evaluation.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(run_dir / "test_predictions.csv")
    registered = pd.read_csv(ROOT / "experiments" / "evidence_registry" / "run_registry.csv")
    match = registered.loc[registered["run_id"].eq(evaluation["run_id"])]
    if len(match) != 1:
        raise ValueError("Clinical utility requires exactly one registered run_id")
    registry_row = match.iloc[0]
    if _sha256(run_dir / "evaluation.json") != str(registry_row["evaluation_sha256"]):
        raise ValueError("Clinical utility evaluation.json does not match its registered hash")
    if _sha256(run_dir / "test_predictions.csv") != str(registry_row["prediction_sha256"]):
        raise ValueError("Clinical utility predictions do not match their registered hash")

    summary_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    alert_rows: list[dict[str, Any]] = []
    dca_rows: list[dict[str, Any]] = []
    thresholds = np.linspace(0.01, 0.60, 60)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0))
    bootstrap_indices = _patient_cluster_indices(
        predictions["patient_id"].astype(str).to_numpy(),
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    for endpoint_index, endpoint in enumerate(("idh", "ih")):
        y = predictions[f"{endpoint}_event"].to_numpy(dtype=int)
        cal_ax, dca_ax = axes[endpoint_index]
        cal_ax.plot([0, 1], [0, 1], color="#9CA3AF", linestyle=":", label="Ideal")
        prevalence = float(y.mean())
        for key, label, color, linestyle, marker in MODELS:
            p = predictions[f"probability_{endpoint}_{key}"].to_numpy(dtype=float)
            bins = _calibration_bins(y, p)
            intercept, slope = _calibration(y, p)
            ece = _ece(bins)
            validation_threshold = float(evaluation["metrics"][endpoint][key]["threshold"])
            bootstrap = _clinical_bootstrap(y, p, validation_threshold, bootstrap_indices)
            summary_rows.append({
                "run_id": evaluation["run_id"], "endpoint": endpoint.upper(), "model": label,
                "n_sessions": len(y), "n_patients": predictions["patient_id"].astype(str).nunique(),
                "prevalence": prevalence, "calibration_intercept": intercept,
                "calibration_intercept_lower": bootstrap["calibration_intercept"]["lower"],
                "calibration_intercept_upper": bootstrap["calibration_intercept"]["upper"],
                "calibration_slope": slope,
                "calibration_slope_lower": bootstrap["calibration_slope"]["lower"],
                "calibration_slope_upper": bootstrap["calibration_slope"]["upper"],
                "ece_10_quantile_bins": ece,
                "ece_lower": bootstrap["ece_10_quantile_bins"]["lower"],
                "ece_upper": bootstrap["ece_10_quantile_bins"]["upper"],
                "brier": evaluation["metrics"][endpoint][key]["brier"],
                "brier_lower": bootstrap["brier"]["lower"],
                "brier_upper": bootstrap["brier"]["upper"],
                "bootstrap_unit": "patient_id", "bootstrap_replicates": args.bootstrap_replicates,
                "bootstrap_valid_replicates": bootstrap["brier"]["valid_replicates"],
                "status": "registered_test_predictions_patient_cluster_ci",
            })
            cal_ax.plot(bins["mean_predicted"], bins["observed"], color=color, linestyle=linestyle,
                        marker=marker, markersize=3.5, label=label)
            for bin_index, row in bins.iterrows():
                calibration_rows.append({
                    "run_id": evaluation["run_id"], "endpoint": endpoint.upper(), "model": label,
                    "bin": int(bin_index) + 1, "n": int(row["n"]),
                    "mean_predicted": row["mean_predicted"], "observed": row["observed"],
                })
            alert = {
                "run_id": evaluation["run_id"], "endpoint": endpoint.upper(), "model": label,
                "threshold_source": "stored_validation_youden_threshold", **_confusion(y, p, validation_threshold),
                "bootstrap_unit": "patient_id", "bootstrap_replicates": args.bootstrap_replicates,
                "bootstrap_valid_replicates": bootstrap["sensitivity"]["valid_replicates"],
            }
            for metric in (
                "sensitivity", "specificity", "ppv", "npv", "events_detected_per_1000",
                "false_alerts_per_1000", "total_alerts_per_1000",
            ):
                alert[f"{metric}_lower"] = bootstrap[metric]["lower"]
                alert[f"{metric}_upper"] = bootstrap[metric]["upper"]
            alert_rows.append(alert)
            for threshold in thresholds:
                benefit = _net_benefit(y, p, float(threshold))
                dca_rows.append({
                    "run_id": evaluation["run_id"], "endpoint": endpoint.upper(), "strategy": label,
                    "threshold": threshold, "net_benefit": benefit,
                    "interpretation_status": "sensitivity_analysis_not_clinically_prespecified",
                })
                dca_ax.plot([], []) if False else None
            benefits = [_net_benefit(y, p, float(t)) for t in thresholds]
            dca_ax.plot(thresholds, benefits, color=color, linestyle=linestyle, label=label)
        treat_all = prevalence - (1 - prevalence) * thresholds / (1 - thresholds)
        dca_ax.plot(thresholds, treat_all, color="#20242A", linestyle="--", label="Alert all")
        dca_ax.axhline(0, color="#9CA3AF", linestyle=":", label="Alert none")
        for threshold, benefit in zip(thresholds, treat_all):
            dca_rows.append({
                "run_id": evaluation["run_id"], "endpoint": endpoint.upper(), "strategy": "Alert all",
                "threshold": threshold, "net_benefit": benefit,
                "interpretation_status": "sensitivity_analysis_not_clinically_prespecified",
            })
            dca_rows.append({
                "run_id": evaluation["run_id"], "endpoint": endpoint.upper(), "strategy": "Alert none",
                "threshold": threshold, "net_benefit": 0.0,
                "interpretation_status": "sensitivity_analysis_not_clinically_prespecified",
            })
        cal_ax.set(title=f"{endpoint.upper()}: calibration", xlabel="Mean predicted probability",
                   ylabel="Observed event proportion", xlim=(0, 1), ylim=(0, 1))
        dca_ax.set(title=f"{endpoint.upper()}: decision-curve sensitivity analysis",
                   xlabel="Risk threshold", ylabel="Net benefit", xlim=(0.01, 0.60))
        for ax in (cal_ax, dca_ax):
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
            ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Registered held-out clinical utility diagnostics", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output / "clinical_utility.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / "clinical_utility.pdf", bbox_inches="tight")
    plt.close(fig)

    _write_csv(output / "calibration_summary.csv", list(summary_rows[0]), summary_rows)
    _write_csv(output / "calibration_curve_data.csv", list(calibration_rows[0]), calibration_rows)
    _write_csv(output / "alert_analysis.csv", list(alert_rows[0]), alert_rows)
    _write_csv(output / "decision_curve_data.csv", list(dca_rows[0]), dca_rows)
    report = f"""# Clinical Utility Report\n\nRegistered run: `{evaluation['run_id']}`\n\nThis package uses the registered held-out target predictions only. No predictive model was fitted or updated; calibration regressions are descriptive diagnostics.\n\n- Calibration intercept, slope, ECE, Brier score, and fixed-threshold alert metrics include 95% percentile intervals from {args.bootstrap_replicates} patient-cluster bootstrap replicates (seed {args.bootstrap_seed}).\n- Alert analysis uses each model's threshold selected on its corresponding validation patients and stored before test scoring.\n- Decision curves cover thresholds 0.01-0.60 as a sensitivity analysis. No clinically relevant range has been claimed because nephrology threshold prespecification is not yet documented.\n- Operating characteristics remain tied to validation-derived Youden thresholds and are not clinically prespecified deployment thresholds.\n"""
    (output / "README.md").write_text(report, encoding="utf-8")
    generated = [
        "README.md", "alert_analysis.csv", "calibration_curve_data.csv", "calibration_summary.csv",
        "clinical_utility.pdf", "clinical_utility.png", "decision_curve_data.csv",
    ]
    provenance = {
        "run_id": evaluation["run_id"],
        "evidence_tier": "registered_descriptive_clinical_utility",
        "source_evaluation": {
            "path": str((run_dir / "evaluation.json").relative_to(ROOT)),
            "sha256": _sha256(run_dir / "evaluation.json"),
        },
        "source_predictions": {
            "path": str((run_dir / "test_predictions.csv").relative_to(ROOT)),
            "sha256": _sha256(run_dir / "test_predictions.csv"),
        },
        "outputs": {name: _sha256(output / name) for name in generated},
        "claim_limits": [
            "dca_threshold_range_not_clinically_prespecified",
            "alert_operating_points_are_validation_youden_thresholds",
        ],
        "uncertainty": {
            "unit": "patient_id", "method": "percentile_95", "replicates": args.bootstrap_replicates,
            "seed": args.bootstrap_seed,
        },
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote registered clinical-utility package to {output}")


if __name__ == "__main__":
    main()
