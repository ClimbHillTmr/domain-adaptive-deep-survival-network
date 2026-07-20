"""Build manuscript tables from registered binary-study artifacts only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "binary_20260719_044949_c6c955a30fc2"
DEFAULT_OUTPUT = ROOT / "tables" / "manuscript"
REGISTRY = ROOT / "experiments" / "evidence_registry" / "run_registry.csv"
DATASETS = ROOT / "experiments" / "evidence_registry" / "dataset_manifest.csv"
ABLATION = ROOT / "experiments" / "confirmatory_results" / "architecture_matched_ablation.csv"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _format_ci(point: float, lower: float, upper: float, digits: int = 3) -> str:
    return f"{point:.{digits}f} ({lower:.{digits}f}–{upper:.{digits}f})"


def _write_markdown_table1(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Table 1. Registered cohort and patient-level analysis allocation",
        "",
        "| Cohort | Analysis role | Patients | Sessions | IDH, n (%) | IH, n (%) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['cohort']} | {row['analysis_role']} | {row['patients']:,} | {row['sessions']:,} | "
            f"{row['idh_events']:,} ({100 * row['idh_prevalence']:.1f}) | "
            f"{row['ih_events']:,} ({100 * row['ih_prevalence']:.1f}) |"
        )
    lines.extend([
        "",
        "IDH, intradialytic hypotension; IH, intradialytic hypertension. Source and target partitions are mutually exclusive at patient level. Counts are registered historical-v1 cohort and split descriptors; exact v1 processed bytes remain unavailable.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_markdown_table2(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Table 2. Held-out target performance from the registered seed-2024 run",
        "",
        "| Endpoint | Model | ROC AUC (95% CI) | PR AUC (95% CI) | Brier (95% CI) | Paired comparison | Δ AUC (95% CI) |",
        "|---|---|---:|---:|---:|---|---:|",
    ]
    for row in rows:
        delta = "—"
        if row["delta_auc"] != "":
            delta = _format_ci(row["delta_auc"], row["delta_auc_lower"], row["delta_auc_upper"], 4)
        lines.append(
            f"| {row['endpoint']} | {row['model']} | "
            f"{_format_ci(row['roc_auc'], row['roc_auc_lower'], row['roc_auc_upper'], 4)} | "
            f"{_format_ci(row['pr_auc'], row['pr_auc_lower'], row['pr_auc_upper'], 4)} | "
            f"{_format_ci(row['brier'], row['brier_lower'], row['brier_upper'], 4)} | "
            f"{row['delta_comparison_label'] or '—'} | {delta} |"
        )
    lines.extend([
        "",
        "CI values use 1,000 patient-cluster bootstrap replicates. PR AUC, precision-recall area under the curve.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    registry = pd.read_csv(REGISTRY)
    match = registry.loc[registry["run_id"].eq(RUN_ID)]
    if len(match) != 1:
        raise ValueError("Manuscript tables require exactly one registered main run")
    registered = match.iloc[0]
    run_dir = ROOT / "experiments" / "binary_results" / RUN_ID
    evaluation_path = run_dir / "evaluation.json"
    split_path = run_dir / "split_manifest.json"
    if _sha256(evaluation_path) != str(registered["evaluation_sha256"]):
        raise ValueError("Main evaluation artifact does not match the registered hash")
    if _sha256(split_path) != str(registered["split_manifest_sha256"]):
        raise ValueError("Main split artifact does not match the registered hash")
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    datasets = pd.read_csv(DATASETS).set_index("cohort")

    table1_rows = []
    for cohort in ("source", "target"):
        row = datasets.loc[cohort]
        table1_rows.append({
            "run_id": RUN_ID,
            "cohort": cohort.capitalize(),
            "analysis_role": "Overall registered cohort",
            "patients": int(row["patient_count"]),
            "sessions": int(row["session_count"]),
            "idh_events": int(row["idh_events"]),
            "idh_prevalence": float(row["idh_prevalence"]),
            "ih_events": int(row["ih_events"]),
            "ih_prevalence": float(row["ih_prevalence"]),
            "split_seed": int(evaluation["split_seed"]),
            "evidence_contract": "dual_binary_hemodynamic_v1",
        })
    split_specs = (
        ("Source", "Source training", "source_train"),
        ("Source", "Source validation", "source_val"),
        ("Target", "Target update", "target_train"),
        ("Target", "Target validation/calibration", "target_val"),
        ("Target", "Held-out target test", "target_test"),
    )
    for cohort, role, key in split_specs:
        row = split[key]
        table1_rows.append({
            "run_id": RUN_ID, "cohort": cohort, "analysis_role": role,
            "patients": int(row["patients"]), "sessions": int(row["sessions"]),
            "idh_events": int(row["idh_events"]), "idh_prevalence": float(row["idh_event_rate"]),
            "ih_events": int(row["ih_events"]), "ih_prevalence": float(row["ih_event_rate"]),
            "split_seed": int(split["split_seed"]), "evidence_contract": "dual_binary_hemodynamic_v1",
        })

    model_specs = (
        ("Source MLP", "source_mlp", "", ""),
        ("Updated MLP", "updated_mlp", "updated_mlp_vs_source_mlp", "Updated MLP − source MLP"),
        ("Target-local logistic", "local_logistic", "updated_mlp_vs_local_logistic", "Updated MLP − target-local logistic"),
    )
    table2_rows = []
    for endpoint in ("idh", "ih"):
        for label, key, comparison_key, comparison_label in model_specs:
            metric = evaluation["metrics"][endpoint][key]
            bootstrap = metric["patient_cluster_bootstrap"]
            comparison = evaluation["paired_comparisons"][endpoint].get(comparison_key, {})
            table2_rows.append({
                "run_id": RUN_ID, "seed": int(evaluation["initialization_seed"]),
                "endpoint": endpoint.upper(), "model": label,
                "n_patients": int(metric["n_patients"]), "n_sessions": int(metric["n_sessions"]),
                "roc_auc": float(metric["roc_auc"]), "roc_auc_lower": float(bootstrap["roc_auc"]["lower"]),
                "roc_auc_upper": float(bootstrap["roc_auc"]["upper"]),
                "pr_auc": float(metric["pr_auc"]), "pr_auc_lower": float(bootstrap["pr_auc"]["lower"]),
                "pr_auc_upper": float(bootstrap["pr_auc"]["upper"]),
                "brier": float(metric["brier"]), "brier_lower": float(bootstrap["brier"]["lower"]),
                "brier_upper": float(bootstrap["brier"]["upper"]),
                "delta_comparison": comparison_key,
                "delta_comparison_label": comparison_label,
                "delta_auc": float(comparison["delta_a_minus_b"]) if comparison else "",
                "delta_auc_lower": float(comparison["lower"]) if comparison else "",
                "delta_auc_upper": float(comparison["upper"]) if comparison else "",
                "bootstrap_unit": "patient_id", "bootstrap_replicates": 1000,
                "evidence_contract": "dual_binary_hemodynamic_v1",
            })

    csv1, csv2 = output / "table1_cohort_design.csv", output / "table2_main_performance.csv"
    md1, md2 = output / "table1_cohort_design.md", output / "table2_main_performance.md"
    _write_csv(csv1, table1_rows)
    _write_csv(csv2, table2_rows)
    _write_markdown_table1(md1, table1_rows)
    _write_markdown_table2(md2, table2_rows)
    sources = [REGISTRY, DATASETS, evaluation_path, split_path, ABLATION]
    outputs = [csv1, csv2, md1, md2]
    ablation = pd.read_csv(ABLATION)
    provenance = {
        "package_status": "tables_1_2_ready_table_3_pending_server",
        "run_id": RUN_ID,
        "sources": {
            str(path.relative_to(ROOT)): _sha256(path) for path in sources
        },
        "outputs": {path.name: _sha256(path) for path in outputs},
        "table3": {
            "path": str(ABLATION.relative_to(ROOT)),
            "status_counts": ablation["status"].value_counts().to_dict(),
            "registered_rows": int(ablation["run_id"].notna().sum()),
        },
        "claim_limits": [
            "table1_is_cohort_and_analysis_design_not_unavailable_baseline_characteristics",
            "table3_must_remain_pending_until_server_acceptance",
            "historical_v1_exact_processed_bytes_unavailable",
        ],
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote registered manuscript tables to {output}")


if __name__ == "__main__":
    main()
