import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.evaluate.dca_analysis import calculate_net_benefit


RESULT_DIR = Path("experiments/results")
TABLE_DIR = Path("tables")
MANUSCRIPT_DIR = Path("manuscript")
THRESHOLDS = [0.10, 0.20, 0.30]
HORIZONS = [60, 120]
ACTION_MAP = {
    0.10: "intensified monitoring",
    0.20: "ultrafiltration review",
    0.30: "bedside reassessment",
}


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_thresholds(model_name: str, pred_df: pd.DataFrame) -> List[Dict]:
    rows = []
    for horizon in HORIZONS:
        y_true = pred_df[f"event_by_{horizon}m"].astype(int).to_numpy()
        y_prob = pred_df[f"event_prob_{horizon}m"].astype(float).to_numpy()
        prevalence = float(y_true.mean())
        for threshold in THRESHOLDS:
            flagged = y_prob >= threshold
            flagged_n = int(flagged.sum())
            tp = int(((flagged == 1) & (y_true == 1)).sum())
            fp = int(((flagged == 1) & (y_true == 0)).sum())
            event_n = int(y_true.sum())
            nb_model = float(calculate_net_benefit(y_true, y_prob, [threshold])[0])
            nb_all = float(prevalence - (1 - prevalence) * (threshold / (1 - threshold)))
            rows.append(
                {
                    "model": model_name,
                    "horizon_min": horizon,
                    "threshold": threshold,
                    "clinical_action": ACTION_MAP[threshold],
                    "flagged_sessions_n": flagged_n,
                    "flagged_sessions_proportion": float(flagged_n / len(y_true)),
                    "events_n": event_n,
                    "captured_events_n": tp,
                    "event_capture_rate": float(tp / event_n) if event_n > 0 else None,
                    "ppv": float(tp / flagged_n) if flagged_n > 0 else None,
                    "workload_per_event_captured": float(flagged_n / tp) if tp > 0 else None,
                    "net_benefit_model": nb_model,
                    "net_benefit_treat_all": nb_all,
                    "net_benefit_treat_none": 0.0,
                    "delta_vs_treat_all": float(nb_model - nb_all),
                    "delta_vs_treat_none": nb_model,
                    "false_positives_n": fp,
                }
            )
    return rows


def build_metric_rows() -> pd.DataFrame:
    cdan_eval = load_json(RESULT_DIR / "evaluation_results.json")
    cdan_cal = load_json(RESULT_DIR / "calibration_dca_metrics.json")
    local_eval = load_json(RESULT_DIR / "local_cox_update_results.json")
    local_cal = load_json(RESULT_DIR / "local_cox_calibration_metrics.json")

    rows = []
    for model_name, metrics, calibration in [
        ("CDAN-GSN (current locked run)", cdan_eval["sweeps"]["CDAN-GSN (Ours)"]["metrics"], cdan_cal),
        ("Local Cox update", local_eval["metrics"], local_cal),
    ]:
        row = {
            "model": model_name,
            "c_index": metrics["C-index"],
            "c_index_ci_lower": metrics["C-index_CI_lower"],
            "c_index_ci_upper": metrics["C-index_CI_upper"],
        }
        for horizon in HORIZONS:
            row[f"observed_rate_{horizon}m"] = calibration[f"{horizon}m"]["observed_rate"]
            row[f"mean_predicted_rate_{horizon}m"] = calibration[f"{horizon}m"]["mean_predicted_rate"]
            row[f"brier_{horizon}m"] = calibration[f"{horizon}m"]["brier"]
            row[f"calibration_intercept_{horizon}m"] = calibration[f"{horizon}m"]["calibration_intercept"]
            row[f"calibration_slope_{horizon}m"] = calibration[f"{horizon}m"]["calibration_slope"]
        rows.append(row)
    return pd.DataFrame(rows)


def build_markdown(metric_df: pd.DataFrame, threshold_df: pd.DataFrame) -> str:
    lines = [
        "# CKJ Translational Addendum",
        "",
        "## Threshold Package",
        "Pre-specified thresholds were set at 10%, 20%, and 30% to represent escalating use cases: intensified monitoring, ultrafiltration review, and bedside reassessment. These thresholds are intended for silent-mode decision support rather than automated intervention.",
        "",
        "## Key Comparison",
    ]
    for _, row in metric_df.iterrows():
        lines.append(
            f"- `{row['model']}`: C-index {row['c_index']:.3f} ({row['c_index_ci_lower']:.3f}-{row['c_index_ci_upper']:.3f}); "
            f"Brier 60m {row['brier_60m']:.4f}, 120m {row['brier_120m']:.4f}."
        )
    lines.extend(["", "## Reviewer-facing Interpretation"])

    cdan = metric_df.loc[metric_df["model"] == "CDAN-GSN (current locked run)"].iloc[0]
    local = metric_df.loc[metric_df["model"] == "Local Cox update"].iloc[0]
    if local["c_index"] >= cdan["c_index"]:
        lines.append(
            "- The simple local baseline is not inferior to the current deep model on held-out testing. CKJ-facing claims should therefore emphasize the value of local updating and workflow-aware deployment, not algorithmic novelty."
        )
    else:
        lines.append(
            "- The deep model retains a measurable advantage over the simple local baseline, but the gain remains clinically modest and must be defended through threshold utility and deployment logic."
        )
    lines.append(
        "- `no_cdan` ablation is not an adequate clinical comparator because it remains an internal architectural variant rather than an independently deployable local baseline."
    )
    lines.append("")
    lines.append("## Threshold Snapshot")
    for horizon in HORIZONS:
        lines.append(f"- `{horizon} min`")
        subset = threshold_df[threshold_df["horizon_min"] == horizon]
        for _, row in subset.iterrows():
            lines.append(
                f"  {row['model']} @ {row['threshold']:.2f}: flagged {row['flagged_sessions_proportion']*100:.1f}%, "
                f"captured {row['event_capture_rate']*100:.1f}% of events, workload/event {row['workload_per_event_captured']:.2f}, "
                f"delta net benefit vs treat-all {row['delta_vs_treat_all']:.4f}."
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    MANUSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

    cdan_pred = pd.read_csv(RESULT_DIR / "real_test_predictions.csv")
    local_pred = pd.read_csv(RESULT_DIR / "local_cox_test_predictions.csv")
    threshold_rows = summarize_thresholds("CDAN-GSN (current locked run)", cdan_pred)
    threshold_rows.extend(summarize_thresholds("Local Cox update", local_pred))
    threshold_df = pd.DataFrame(threshold_rows)
    threshold_df.to_csv(TABLE_DIR / "ckj_threshold_analysis.csv", index=False)

    metric_df = build_metric_rows()
    metric_df.to_csv(TABLE_DIR / "ckj_model_comparison.csv", index=False)

    markdown = build_markdown(metric_df, threshold_df)
    (TABLE_DIR / "ckj_threshold_analysis.md").write_text(markdown, encoding="utf-8")
    (MANUSCRIPT_DIR / "09_CKJ_Revision_Addendum.md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
