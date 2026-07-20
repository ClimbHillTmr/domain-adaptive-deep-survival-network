"""Freeze traceable manifests for the dual-binary manuscript evidence.

This command is read-only with respect to clinical/model inputs. It inventories
checked-in artifacts and explicitly distinguishes frozen-run provenance from the
currently available processed CSV files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "experiments" / "evidence_registry"
RUN_ROOT = ROOT / "experiments" / "binary_results"
ALLOWLIST = ROOT / "experiments" / "audit" / "feature_allowlist.csv"
PREFLIGHT = ROOT / "experiments" / "audit" / "binary_data_preflight.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def _runs() -> list[tuple[Path, dict[str, Any], dict[str, Any]]]:
    runs = []
    for evaluation_path in sorted(RUN_ROOT.glob("*/evaluation.json")):
        run_dir = evaluation_path.parent
        runs.append(
            (
                run_dir,
                _read_json(evaluation_path),
                yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8")),
            )
        )
    if not runs:
        raise FileNotFoundError(f"No registered binary runs found below {RUN_ROOT}")
    return runs


def _dataset_manifest(
    output_dir: Path, runs: list[tuple[Path, dict[str, Any], dict[str, Any]]]
) -> tuple[list[dict[str, Any]], bool]:
    preflight = _read_json(PREFLIGHT)
    reference = runs[0][1]
    run_hashes = {
        "source": {run[1]["provenance"]["source_sha256"] for run in runs},
        "target": {run[1]["provenance"]["target_sha256"] for run in runs},
    }
    rows = []
    exact_inputs_available = True
    for cohort in ("source", "target"):
        if len(run_hashes[cohort]) != 1:
            raise ValueError(f"Frozen runs disagree on the {cohort} dataset hash")
        config_key = f"{cohort}_file"
        configured_path = ROOT / runs[0][2]["data"][config_key]
        expected_hash = next(iter(run_hashes[cohort]))
        current_hash = _sha256(configured_path) if configured_path.exists() else ""
        exact = current_hash == expected_hash
        exact_inputs_available &= exact
        summary = preflight["cohorts"][cohort]
        rows.append(
            {
                "dataset_version": "dual_binary_hemodynamic_v1_frozen_20260719",
                "cohort": cohort,
                "frozen_file_path": runs[0][2]["data"][config_key],
                "frozen_sha256": expected_hash,
                "current_sha256": current_hash,
                "exact_frozen_file_available": str(exact).lower(),
                "row_count": summary["sessions"],
                "patient_count": summary["patients"],
                "session_count": summary["sessions"],
                "idh_events": summary["endpoints"]["idh"]["events"],
                "idh_prevalence": summary["endpoints"]["idh"]["event_rate"],
                "ih_events": summary["endpoints"]["ih"]["events"],
                "ih_prevalence": summary["endpoints"]["ih"]["event_rate"],
                "preprocessing_version": "source_fitted_median_zscore_and_source_categories_v1",
                "preprocessing_artifact": _relative(runs[0][0] / "preprocessing.json"),
                "registered_run_count": len(runs),
                "status": "exact_input_available" if exact else "frozen_hash_only_current_file_mismatch",
            }
        )
    _write_csv(
        output_dir / "dataset_manifest.csv",
        [
            "dataset_version", "cohort", "frozen_file_path", "frozen_sha256", "current_sha256",
            "exact_frozen_file_available", "row_count", "patient_count", "session_count",
            "idh_events", "idh_prevalence", "ih_events", "ih_prevalence",
            "preprocessing_version", "preprocessing_artifact", "registered_run_count", "status",
        ],
        rows,
    )
    return rows, exact_inputs_available


def _feature_manifest(output_dir: Path, reference: dict[str, Any]) -> list[dict[str, Any]]:
    allowlist = pd.read_csv(ALLOWLIST).fillna("")
    selected = list(reference["feature_names"])
    physio_indices = set(range(len(selected))) - {2, 3, 8, 23}
    rows = []
    for _, record in allowlist.iterrows():
        feature = str(record["feature"])
        selected_index = selected.index(feature) if feature in selected else None
        if selected_index is None:
            branch = "excluded"
        elif selected_index in physio_indices:
            branch = "physiology_history"
        else:
            branch = "treatment_context"
        if selected_index is not None:
            rule = "source-train median imputation; source-train z-score"
            if feature.endswith("_code"):
                rule = "source-fitted categorical encoding; unknown category code; " + rule
        else:
            rule = "excluded"
        rationale = str(record.get("reason", ""))
        if selected_index is None and "C-index" in rationale:
            rationale = (
                "excluded from the frozen 24-predictor binary model; the source allowlist "
                "contains a legacy survival-study rationale that is not used as binary-study evidence"
            )
        rows.append(
            {
                "feature_name": feature,
                "category": record.get("category", ""),
                "branch_assignment": branch,
                "inclusion_status": "included" if selected_index is not None else "excluded",
                "model_index": "" if selected_index is None else selected_index,
                "available_at_prediction_time": record.get("available_at_prediction_time", ""),
                "source_column": record.get("source_column", ""),
                "preprocessing_rule": rule,
                "rationale": rationale,
            }
        )
    _write_csv(
        output_dir / "feature_manifest.csv",
        ["feature_name", "category", "branch_assignment", "inclusion_status", "model_index",
         "available_at_prediction_time", "source_column", "preprocessing_rule", "rationale"],
        rows,
    )
    return rows


def _split_manifest(
    output_dir: Path, runs: list[tuple[Path, dict[str, Any], dict[str, Any]]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    split_hashes = {_sha256(run_dir / "split_manifest.json") for run_dir, _, _ in runs}
    identical = len(split_hashes) == 1
    reference_path = runs[0][0] / "split_manifest.json"
    split = _read_json(reference_path)
    rows = []
    patient_sets: dict[str, set[str]] = {}
    fold_names = ("source_train", "source_val", "target_train", "target_val", "target_test")
    for fold in fold_names:
        patient_sets[fold] = {str(value) for value in split[fold]["patient_ids"]}
        cohort = "source" if fold.startswith("source") else "target"
        role = {
            "source_train": "source_train",
            "source_val": "source_validation",
            "target_train": "target_update",
            "target_val": "target_validation",
            "target_test": "target_test",
        }[fold]
        for patient_id in sorted(patient_sets[fold]):
            rows.append(
                {
                    "patient_id": patient_id,
                    "cohort": cohort,
                    "split": role,
                    "split_seed": split["split_seed"],
                    "split_manifest_sha256": next(iter(split_hashes)) if identical else "MISMATCH",
                    "identical_across_registered_runs": str(identical).lower(),
                }
            )
    target_folds = ("target_train", "target_val", "target_test")
    target_overlap = any(
        patient_sets[target_folds[i]] & patient_sets[target_folds[j]]
        for i in range(3) for j in range(i + 1, 3)
    )
    source_overlap = bool(patient_sets["source_train"] & patient_sets["source_val"])
    audit = {
        "registered_runs": len(runs),
        "identical_split_manifests": identical,
        "source_patient_overlap": source_overlap,
        "target_patient_overlap": target_overlap,
        "temporal_leakage_status": "not_verifiable_without_exact_frozen_session-level_datasets",
        "passed_patient_overlap_gate": identical and not source_overlap and not target_overlap,
    }
    _write_csv(
        output_dir / "split_manifest.csv",
        ["patient_id", "cohort", "split", "split_seed", "split_manifest_sha256",
         "identical_across_registered_runs"],
        rows,
    )
    (output_dir / "split_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return rows, audit


def _strategy(config: dict[str, Any]) -> tuple[str, str]:
    model = config["model"]
    dual = bool(model.get("use_stratified_alignment"))
    if bool(model.get("use_coral")) and dual and bool(model.get("use_outcome_specific_alignment")):
        return "dual_branch", "outcome_specific_coral"
    if bool(model.get("use_coral")):
        return ("dual_branch" if dual else "single_encoder"), "global_coral"
    if dual:
        return "dual_branch", "no_alignment"
    return "single_encoder", "target_update_no_alignment"


def _run_registry(
    output_dir: Path,
    runs: list[tuple[Path, dict[str, Any], dict[str, Any]]],
    exact_inputs_available: bool,
) -> list[dict[str, Any]]:
    rows = []
    aliases: dict[str, list[str]] = {}
    for evaluation_path in (ROOT / "experiments" / "final_results").glob("*/evaluation.json"):
        alias_evaluation = _read_json(evaluation_path)
        aliases.setdefault(alias_evaluation["run_id"], []).append(_relative(evaluation_path.parent))
    for run_dir, evaluation, config in runs:
        architecture, alignment = _strategy(config)
        prediction = run_dir / "test_predictions.csv"
        checkpoint = run_dir / "mlp_models.pt"
        source_hash = evaluation["provenance"]["source_sha256"]
        target_hash = evaluation["provenance"]["target_sha256"]
        combined_hash = hashlib.sha256(f"{source_hash}:{target_hash}".encode()).hexdigest()
        rows.append(
            {
                "run_id": evaluation["run_id"],
                "final_result_aliases": ";".join(sorted(aliases.get(evaluation["run_id"], []))),
                "contract": evaluation.get("contract", ""),
                "model_architecture": architecture,
                "alignment_strategy": alignment,
                "seed": evaluation["initialization_seed"],
                "split_seed": evaluation["split_seed"],
                "configuration_hash": evaluation["provenance"]["config_sha256"],
                "source_dataset_hash": source_hash,
                "target_dataset_hash": target_hash,
                "combined_dataset_hash": combined_hash,
                "checkpoint_path": _relative(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
                "prediction_file": _relative(prediction),
                "prediction_sha256": _sha256(prediction),
                "evaluation_file": _relative(run_dir / "evaluation.json"),
                "evaluation_sha256": _sha256(run_dir / "evaluation.json"),
                "split_manifest_sha256": _sha256(run_dir / "split_manifest.json"),
                "artifact_complete": str(checkpoint.exists() and prediction.exists()).lower(),
                "reproducibility_status": (
                    "exact_input_available" if exact_inputs_available else "historical_artifacts_only_input_hash_mismatch"
                ),
                "confirmatory_eligibility": "no_architecture_matched_multiseed_set",
            }
        )
    fields = ["run_id", "final_result_aliases", "contract", "model_architecture", "alignment_strategy", "seed", "split_seed",
         "configuration_hash", "source_dataset_hash", "target_dataset_hash", "combined_dataset_hash",
         "checkpoint_path", "checkpoint_sha256", "prediction_file", "prediction_sha256",
         "evaluation_file", "evaluation_sha256", "split_manifest_sha256", "artifact_complete",
         "reproducibility_status", "confirmatory_eligibility"]
    existing_path = output_dir / "run_registry.csv"
    if existing_path.exists():
        existing = pd.read_csv(existing_path).fillna("")
        preserved = existing.loc[
            existing["contract"].astype(str).eq("dual_binary_hemodynamic_confirmatory_v2")
        ]
        rows.extend(preserved.to_dict(orient="records"))
    _write_csv(existing_path, fields, rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = _runs()
    datasets, exact_inputs = _dataset_manifest(output_dir, runs)
    features = _feature_manifest(output_dir, runs[0][1])
    splits, split_audit = _split_manifest(output_dir, runs)
    registry = _run_registry(output_dir, runs, exact_inputs)
    status = {
        "registry_version": "binary_evidence_registry_v1",
        "registered_runs": len(registry),
        "registered_historical_v1_runs": sum(
            row["contract"] == "dual_binary_hemodynamic_v1" for row in registry
        ),
        "registered_confirmatory_v2_runs": sum(
            row["contract"] == "dual_binary_hemodynamic_confirmatory_v2" for row in registry
        ),
        "dataset_rows": len(datasets),
        "feature_rows": len(features),
        "split_patient_rows": len(splits),
        "exact_frozen_inputs_available": exact_inputs,
        "patient_split_gate_passed": split_audit["passed_patient_overlap_gate"],
        "temporal_leakage_gate_passed": False,
        "confirmatory_training_authorized_by_evidence_gate": (
            exact_inputs and split_audit["passed_patient_overlap_gate"]
        ),
        "blocking_reasons": [] if exact_inputs else [
            "Current processed CSV hashes do not match the hashes recorded by all frozen runs.",
            "Temporal leakage cannot be re-audited without the exact frozen session-level datasets.",
        ],
    }
    (output_dir / "registry_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
