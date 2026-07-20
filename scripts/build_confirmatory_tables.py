"""Build confirmatory table shells or aggregate a complete registered 25-run set."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import yaml
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from verify_server_confirmatory_results import audit as audit_server_results
from src.evaluate.binary_metrics import paired_patient_bootstrap_metric_deltas


CONFIG_DIR = ROOT / "experiments" / "confirmatory_configs"
RESULT_DIR = ROOT / "experiments" / "confirmatory_results"
OUTPUT = RESULT_DIR / "architecture_matched_ablation.csv"
CONTRAST_OUTPUT = RESULT_DIR / "architecture_matched_paired_contrasts.csv"
SEEDS = {42, 7, 13, 99, 2024}
OUTCOME_SPECIFIC = "D_dual_outcome_specific_coral"
COMPARATORS = (
    "A_dual_no_alignment", "B_dual_global_coral",
    "C_dual_random_feature_coral", "E_single_encoder_reference",
)


def _is_server_confirmatory(evaluation: dict, run_config: dict) -> bool:
    """Accept only explicitly marked server runs with recorded CUDA provenance."""
    confirmatory = run_config.get("confirmatory", {})
    provenance = evaluation.get("provenance", {})
    return (
        confirmatory.get("execution_context") == "server_confirmatory"
        and provenance.get("execution_context") == "server_confirmatory"
        and bool(provenance.get("hostname"))
        and str(provenance.get("device", "")).startswith("cuda")
    )


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    acceptance = audit_server_results()
    accepted_run_ids = (
        {row["run_id"] for row in acceptance["accepted"]}
        if acceptance["status"] == "complete" else set()
    )
    central_registry = ROOT / "experiments" / "evidence_registry" / "run_registry.csv"
    registered_v2_ids = set()
    if central_registry.exists():
        registered = pd.read_csv(central_registry, keep_default_na=False)
        registered_v2_ids = set(
            registered.loc[
                registered["contract"].eq("dual_binary_hemodynamic_confirmatory_v2"), "run_id"
            ].astype(str)
        )
    if acceptance["status"] == "complete" and accepted_run_ids != registered_v2_ids:
        raise SystemExit(
            "Confirmatory table build refused: register all 25 accepted runs in the central run registry first."
        )
    rows = []
    for config_path in sorted(CONFIG_DIR.glob("*.yaml")):
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        model_id = config["confirmatory"]["model_id"]
        seed = int(config["training"]["initialization_seed"])
        matches = []
        for evaluation_path in RESULT_DIR.glob("*/evaluation.json"):
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
            run_config = yaml.safe_load((evaluation_path.parent / "config.yaml").read_text(encoding="utf-8"))
            if evaluation.get("run_id") not in accepted_run_ids:
                continue
            if evaluation.get("run_id") not in registered_v2_ids:
                continue
            if not _is_server_confirmatory(evaluation, run_config):
                continue
            if run_config.get("confirmatory", {}).get("model_id") == model_id and int(evaluation["initialization_seed"]) == seed:
                matches.append((evaluation_path, evaluation))
        if len(matches) > 1:
            raise ValueError(f"Duplicate confirmatory runs for {model_id}, seed {seed}")
        for endpoint in ("idh", "ih"):
            if matches:
                evaluation_path, evaluation = matches[0]
                metric = evaluation["metrics"][endpoint]["updated_mlp"]
                rows.append({
                    "model_id": model_id, "seed": seed, "endpoint": endpoint.upper(),
                    "run_id": evaluation["run_id"], "roc_auc": metric["roc_auc"],
                    "roc_auc_lower": metric["patient_cluster_bootstrap"]["roc_auc"]["lower"],
                    "roc_auc_upper": metric["patient_cluster_bootstrap"]["roc_auc"]["upper"],
                    "pr_auc": metric["pr_auc"],
                    "pr_auc_lower": metric["patient_cluster_bootstrap"]["pr_auc"]["lower"],
                    "pr_auc_upper": metric["patient_cluster_bootstrap"]["pr_auc"]["upper"],
                    "brier": metric["brier"],
                    "brier_lower": metric["patient_cluster_bootstrap"]["brier"]["lower"],
                    "brier_upper": metric["patient_cluster_bootstrap"]["brier"]["upper"],
                    "ece_10_quantile_bins": metric["ece_10_quantile_bins"],
                    "ece_lower": metric["patient_cluster_bootstrap"]["ece_10_quantile_bins"]["lower"],
                    "ece_upper": metric["patient_cluster_bootstrap"]["ece_10_quantile_bins"]["upper"],
                    "status": "registered_confirmatory_result",
                })
            else:
                rows.append({
                    "model_id": model_id, "seed": seed, "endpoint": endpoint.upper(),
                    "status": "pending_server_confirmatory_run",
                })
    expected = 5 * len(SEEDS) * 2
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} model-seed-endpoint rows, found {len(rows)}")
    fields = [
        "model_id", "seed", "endpoint", "run_id", "roc_auc", "roc_auc_lower", "roc_auc_upper",
        "pr_auc", "pr_auc_lower", "pr_auc_upper", "brier", "brier_lower", "brier_upper",
        "ece_10_quantile_bins", "ece_lower", "ece_upper", "status",
    ]
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
    print(f"Wrote {len(rows)} confirmatory registry rows to {OUTPUT}")

    contrast_rows = []
    run_lookup = {
        (row["model_id"], int(row["seed"])): row["run_id"]
        for row in acceptance["accepted"]
        if row["run_id"] in registered_v2_ids
    }
    for seed in sorted(SEEDS):
        for comparator in COMPARATORS:
            for endpoint in ("idh", "ih"):
                pair = ((OUTCOME_SPECIFIC, "outcome_specific"), (comparator, "comparator"))
                frames = {}
                run_ids = {}
                for model_id, label in pair:
                    run_id = run_lookup.get((model_id, seed))
                    run_ids[label] = run_id or ""
                    if run_id:
                        frames[label] = pd.read_csv(
                            RESULT_DIR / run_id / "test_predictions.csv", low_memory=False
                        )
                if len(frames) == 2:
                    keys = ["session_id", "patient_id", f"{endpoint}_event"]
                    left = frames["outcome_specific"][
                        [*keys, f"probability_{endpoint}_updated_mlp"]
                    ].rename(columns={f"probability_{endpoint}_updated_mlp": "probability_a"})
                    right = frames["comparator"][
                        [*keys, f"probability_{endpoint}_updated_mlp"]
                    ].rename(columns={f"probability_{endpoint}_updated_mlp": "probability_b"})
                    merged = left.merge(right, on=keys, how="inner", validate="one_to_one")
                    if len(merged) != 29866:
                        raise ValueError(
                            f"Paired contrast test sessions mismatch for {comparator}/{seed}/{endpoint}"
                        )
                    contrasts = paired_patient_bootstrap_metric_deltas(
                        merged[f"{endpoint}_event"].to_numpy(dtype=int),
                        merged["probability_a"].to_numpy(dtype=float),
                        merged["probability_b"].to_numpy(dtype=float),
                        merged["patient_id"].astype(str).to_numpy(),
                        n_bootstrap=1000,
                        seed=20260715,
                    )
                    for metric, contrast in contrasts.items():
                        contrast_rows.append({
                            "endpoint": endpoint.upper(), "seed": seed, "metric": metric,
                            "model_a": OUTCOME_SPECIFIC, "model_b": comparator,
                            "run_id_a": run_ids["outcome_specific"], "run_id_b": run_ids["comparator"],
                            "delta_a_minus_b": contrast["delta_a_minus_b"],
                            "lower": contrast["lower"], "upper": contrast["upper"],
                            "favorable_direction": contrast["favorable_direction"],
                            "bootstrap_unit": "patient_id",
                            "bootstrap_valid_replicates": contrast["valid_replicates"],
                            "status": "registered_confirmatory_paired_contrast",
                        })
                else:
                    for metric in ("roc_auc", "pr_auc", "brier", "ece_10_quantile_bins"):
                        contrast_rows.append({
                            "endpoint": endpoint.upper(), "seed": seed, "metric": metric,
                            "model_a": OUTCOME_SPECIFIC, "model_b": comparator,
                            "run_id_a": run_ids["outcome_specific"], "run_id_b": run_ids["comparator"],
                            "favorable_direction": "positive" if metric in {"roc_auc", "pr_auc"} else "negative",
                            "bootstrap_unit": "patient_id", "status": "pending_server_confirmatory_run",
                        })
    if len(contrast_rows) != 160:
        raise ValueError(f"Expected 160 paired-contrast rows, found {len(contrast_rows)}")
    contrast_fields = [
        "endpoint", "seed", "metric", "model_a", "model_b", "run_id_a", "run_id_b",
        "delta_a_minus_b", "lower", "upper", "favorable_direction", "bootstrap_unit",
        "bootstrap_valid_replicates", "status",
    ]
    with CONTRAST_OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=contrast_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in contrast_fields} for row in contrast_rows)
    print(f"Wrote {len(contrast_rows)} paired confirmatory contrast rows to {CONTRAST_OUTPUT}")


if __name__ == "__main__":
    main()
