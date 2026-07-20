"""Audit every requested deliverable without fitting models or changing evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from verify_server_confirmatory_results import audit as audit_server_results
from build_manuscript_claim_registry import build as build_claim_registry


DEFAULT_JSON = ROOT / "experiments" / "evidence_registry" / "submission_package_status.json"
DEFAULT_CSV = ROOT / "experiments" / "evidence_registry" / "deliverable_status.csv"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_hashes_match(registry: pd.DataFrame) -> tuple[bool, list[str]]:
    failures = []
    mappings = (
        ("evaluation_file", "evaluation_sha256"),
        ("prediction_file", "prediction_sha256"),
        ("checkpoint_path", "checkpoint_sha256"),
    )
    for _, row in registry.iterrows():
        for path_column, hash_column in mappings:
            path = ROOT / str(row[path_column])
            if not path.exists() or _sha256(path) != str(row[hash_column]):
                failures.append(f"{row['run_id']}:{path_column}")
    return not failures, failures


def _provenance_outputs_match(path: Path) -> tuple[bool, list[str]]:
    if not path.exists():
        return False, [f"missing:{path.relative_to(ROOT)}"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    failures = []
    outputs = payload.get("outputs", {})
    for name, record in outputs.items():
        if isinstance(record, str):
            output_path = path.parent / name
            expected_hash = record
        else:
            output_path = ROOT / record["path"]
            expected_hash = record["sha256"]
        if not output_path.exists() or _sha256(output_path) != expected_hash:
            failures.append(name)
    return not failures, failures


def audit() -> dict:
    rows: list[dict[str, str]] = []
    registry = pd.read_csv(ROOT / "experiments" / "evidence_registry" / "run_registry.csv")
    registry_hash_ok, registry_failures = _artifact_hashes_match(registry)
    historical_registry = registry.loc[registry["contract"].eq("dual_binary_hemodynamic_v1")]
    confirmatory_registry = registry.loc[registry["contract"].eq("dual_binary_hemodynamic_confirmatory_v2")]
    split_audit = json.loads(
        (ROOT / "experiments" / "evidence_registry" / "split_audit.json").read_text(encoding="utf-8")
    )
    frozen_ready = (
        len(historical_registry) == 8
        and historical_registry["artifact_complete"].astype(str).str.lower().eq("true").all()
        and registry_hash_ok
        and split_audit.get("passed_patient_overlap_gate") is True
    )
    rows.append({
        "deliverable": "frozen_evidence_registry",
        "status": "ready_with_historical_input_byte_limitation" if frozen_ready else "failed_audit",
        "evidence": "experiments/evidence_registry/run_registry.csv",
        "remaining_gate": "disclose_unavailable_exact_v1_processed_bytes" if frozen_ready else ";".join(registry_failures),
    })

    server_package_path = ROOT / "experiments" / "confirmatory_configs" / "server_package_manifest.json"
    server_package_failures = []
    if not server_package_path.is_file():
        server_package_failures.append("missing_server_package_manifest")
    else:
        server_package = json.loads(server_package_path.read_text(encoding="utf-8"))
        if server_package.get("training_performed") is not False:
            server_package_failures.append("server_package_training_boundary_missing")
        if server_package.get("locked_config_count") != 25:
            server_package_failures.append("server_package_config_count_not_25")
        records = server_package.get("files", [])
        if server_package.get("file_count") != len(records):
            server_package_failures.append("server_package_file_count_mismatch")
        for record in records:
            path = ROOT / record["path"]
            if (
                not path.is_file()
                or path.stat().st_size != record["size_bytes"]
                or _sha256(path) != record["sha256"]
            ):
                server_package_failures.append(f"server_package_file_mismatch:{record['path']}")
    server_package_ok = not server_package_failures
    rows.append({
        "deliverable": "server_confirmatory_execution_package",
        "status": "ready" if server_package_ok else "failed_provenance_audit",
        "evidence": "experiments/confirmatory_configs/server_package_manifest.json",
        "remaining_gate": "" if server_package_ok else ";".join(server_package_failures),
    })

    server = audit_server_results()
    confirmatory_registry_complete = (
        server["status"] == "complete"
        and len(confirmatory_registry) == 25
        and set(confirmatory_registry["run_id"].astype(str)) == {row["run_id"] for row in server["accepted"]}
    )
    rows.append({
        "deliverable": "architecture_matched_ablation",
        "status": "ready" if confirmatory_registry_complete else "pending_server_confirmatory_runs",
        "evidence": "experiments/confirmatory_results/architecture_matched_ablation.csv",
        "remaining_gate": (
            "" if confirmatory_registry_complete else
            f"accepted={server['accepted_server_runs']}/25;registered={len(confirmatory_registry)}/25;pending={server['pending_server_runs']}"
        ),
    })

    clinical_ok, clinical_failures = _provenance_outputs_match(
        ROOT / "clinical_utility_report" / "provenance.json"
    )
    prespecified_clinical_failures = []
    threshold_protocol_path = ROOT / "conf" / "clinical_threshold_protocol.yaml"
    threshold_protocol = {}
    if threshold_protocol_path.is_file():
        import yaml

        threshold_protocol = yaml.safe_load(threshold_protocol_path.read_text(encoding="utf-8"))
    threshold_locked = (
        threshold_protocol.get("status") == "prespecified_locked"
        and threshold_protocol.get("prespecified_without_target_test_dca_access") is True
    )
    prespecified_provenance_path = ROOT / "clinical_utility_report" / "prespecified" / "provenance.json"
    if threshold_locked:
        if not prespecified_provenance_path.is_file():
            prespecified_clinical_failures.append("missing_prespecified_clinical_utility_provenance")
        else:
            prespecified = json.loads(prespecified_provenance_path.read_text(encoding="utf-8"))
            if (
                prespecified.get("status") != "registered_clinically_prespecified_population_level_utility"
                or prespecified.get("training_performed") is not False
                or prespecified.get("interpretation") != "population_level_research_not_clinical_recommendation"
            ):
                prespecified_clinical_failures.append("prespecified_clinical_utility_status_invalid")
            protocol_record = prespecified.get("protocol", {})
            if (
                protocol_record.get("path") != str(threshold_protocol_path.relative_to(ROOT))
                or protocol_record.get("sha256") != _sha256(threshold_protocol_path)
            ):
                prespecified_clinical_failures.append("prespecified_clinical_protocol_hash_mismatch")
            for key in ("predictions", "evaluation"):
                record = prespecified.get(key, {})
                path = ROOT / str(record.get("path", ""))
                if not path.is_file() or _sha256(path) != record.get("sha256"):
                    prespecified_clinical_failures.append(f"prespecified_clinical_{key}_hash_mismatch")
            for artifact, expected_hash in prespecified.get("outputs", {}).items():
                path = ROOT / artifact
                if not path.is_file() or _sha256(path) != expected_hash:
                    prespecified_clinical_failures.append(f"prespecified_clinical_output_mismatch:{artifact}")
    prespecified_clinical_ok = threshold_locked and not prespecified_clinical_failures
    rows.append({
        "deliverable": "clinical_utility_package",
        "status": (
            "ready" if clinical_ok and prespecified_clinical_ok
            else "registered_cluster_uncertainty_ready_threshold_interpretation_pending"
            if clinical_ok else "failed_provenance_audit"
        ),
        "evidence": "clinical_utility_report/provenance.json",
        "remaining_gate": (
            "" if clinical_ok and prespecified_clinical_ok
            else ";".join(prespecified_clinical_failures)
            if prespecified_clinical_failures
            else "clinical_threshold_prespecification_for_dca_and_deployment_interpretation"
            if clinical_ok else ";".join(clinical_failures)
        ),
    })

    subgroup = pd.read_csv(ROOT / "subgroup_analysis_patient_bootstrap.csv")
    required_subgroup_columns = {
        "run_id", "endpoint", "subgroup", "subgroup_level", "n_patients", "subgroup_auc",
        "subgroup_auc_lower", "subgroup_auc_upper", "delta_auc", "delta_auc_lower", "delta_auc_upper",
        "interaction_p_value", "bootstrap_unit", "bootstrap_valid_replicates", "status",
    }
    subgroup_complete = (
        required_subgroup_columns.issubset(subgroup.columns)
        and len(subgroup) == 12
        and subgroup["subgroup"].nunique() == 6
        and subgroup.groupby(["endpoint", "subgroup"])["subgroup_level"].nunique().eq(2).all()
        and subgroup["subgroup_auc"].notna().all()
        and subgroup["bootstrap_unit"].eq("patient_id").all()
        and subgroup["bootstrap_valid_replicates"].eq(1000).all()
        and subgroup["status"].eq("registered_confirmatory_v2_patient_cluster_subgroup").all()
    )
    subgroup_provenance_failures = []
    subgroup_provenance_path = ROOT / "subgroup_analysis_provenance.json"
    if subgroup_complete:
        if not subgroup_provenance_path.is_file():
            subgroup_provenance_failures.append("missing_subgroup_provenance")
        else:
            subgroup_provenance = json.loads(subgroup_provenance_path.read_text(encoding="utf-8"))
            subgroup_run_ids = set(subgroup["run_id"].astype(str))
            if (
                subgroup_provenance.get("status")
                != "registered_confirmatory_v2_patient_cluster_subgroup"
                or subgroup_provenance.get("training_performed") is not False
            ):
                subgroup_provenance_failures.append("subgroup_provenance_status_invalid")
            if subgroup_run_ids != {subgroup_provenance.get("run_id", "")}:
                subgroup_provenance_failures.append("subgroup_run_id_mismatch")
            output_record = subgroup_provenance.get("output", {})
            if (
                output_record.get("path") != str((ROOT / "subgroup_analysis_patient_bootstrap.csv").relative_to(ROOT))
                or output_record.get("sha256") != _sha256(ROOT / "subgroup_analysis_patient_bootstrap.csv")
            ):
                subgroup_provenance_failures.append("subgroup_output_hash_mismatch")
            for key in ("protocol", "predictions"):
                record = subgroup_provenance.get(key, {})
                path = ROOT / str(record.get("path", ""))
                if not path.is_file() or _sha256(path) != record.get("sha256"):
                    subgroup_provenance_failures.append(f"subgroup_{key}_hash_mismatch")
            target_v2 = ROOT / "data" / "confirmatory_v2" / "target_confirmatory_v2.csv"
            if not target_v2.is_file() or _sha256(target_v2) != subgroup_provenance.get("target_v2_sha256"):
                subgroup_provenance_failures.append("subgroup_target_v2_hash_mismatch")
            registered_subgroup_run = registry.loc[
                registry["run_id"].astype(str).isin(subgroup_run_ids)
                & registry["contract"].eq("dual_binary_hemodynamic_confirmatory_v2")
                & registry["alignment_strategy"].eq("outcome_specific_coral")
                & registry["confirmatory_eligibility"].eq("accepted_architecture_matched_server_run")
            ]
            if len(registered_subgroup_run) != 1:
                subgroup_provenance_failures.append("subgroup_run_not_one_registered_outcome_specific_v2_run")
    subgroup_complete = subgroup_complete and not subgroup_provenance_failures
    rows.append({
        "deliverable": "patient_cluster_subgroup_analysis",
        "status": "ready" if subgroup_complete else "blocked_pending_prespecification",
        "evidence": "subgroup_analysis_patient_bootstrap.csv",
        "remaining_gate": (
            ";".join(subgroup_provenance_failures) if subgroup_provenance_failures
            else "clinical_cutpoints_and_minimum_patient_counts" if not subgroup_complete else ""
        ),
    })

    exploratory_ok, exploratory_failures = _provenance_outputs_match(
        ROOT / "experiments" / "evidence_registry" / "exploratory_provenance.json"
    )
    rows.append({
        "deliverable": "exploratory_representation_analysis",
        "status": "ready_exploratory_only" if exploratory_ok else "failed_provenance_audit",
        "evidence": "experiments/evidence_registry/exploratory_provenance.json",
        "remaining_gate": "retain_exploratory_claim_label" if exploratory_ok else ";".join(exploratory_failures),
    })

    supplement_path = ROOT / "docs" / "submission_supplement.md"
    supplement_provenance_path = ROOT / "experiments" / "evidence_registry" / "submission_supplement_provenance.json"
    supplement_failures = []
    if not supplement_path.exists() or not supplement_provenance_path.exists():
        supplement_failures.append("missing_supplement_or_provenance")
        supplement_payload = {}
    else:
        supplement_payload = json.loads(supplement_provenance_path.read_text(encoding="utf-8"))
        if supplement_payload.get("output_sha256") != _sha256(supplement_path):
            supplement_failures.append("supplement_output_hash_mismatch")
        for artifact, expected_hash in supplement_payload.get("sources", {}).items():
            source = ROOT / artifact
            if not source.exists() or _sha256(source) != expected_hash:
                supplement_failures.append(f"supplement_source_mismatch:{artifact}")
        if supplement_payload.get("training_performed") is not False:
            supplement_failures.append("supplement_training_boundary_missing")
    supplement_ok = not supplement_failures
    rows.append({
        "deliverable": "submission_supplement",
        "status": (
            "ready" if supplement_ok and server["status"] == "complete" and subgroup_complete
            else "evidence_linked_draft" if supplement_ok else "failed_provenance_audit"
        ),
        "evidence": "experiments/evidence_registry/submission_supplement_provenance.json",
        "remaining_gate": (
            "" if supplement_ok and server["status"] == "complete" and subgroup_complete
            else "rebuild_after_server_acceptance_and_locked_subgroup_prespecification"
            if supplement_ok else ";".join(supplement_failures)
        ),
    })

    table_provenance = ROOT / "tables" / "manuscript" / "provenance.json"
    tables_ok, table_failures = _provenance_outputs_match(table_provenance)
    table3 = pd.read_csv(ROOT / "experiments" / "confirmatory_results" / "architecture_matched_ablation.csv")
    table3_registered = int(table3["run_id"].notna().sum())
    table3_consistent = (
        (server["status"] == "complete" and table3_registered == 50)
        or (server["status"] != "complete" and table3_registered == 0)
    )
    rows.append({
        "deliverable": "manuscript_tables",
        "status": (
            "tables_1_2_ready_table_3_pending_server"
            if tables_ok and table3_consistent and server["status"] != "complete"
            else "ready" if tables_ok and table3_consistent else "failed_provenance_audit"
        ),
        "evidence": "tables/manuscript/provenance.json",
        "remaining_gate": (
            "table_3_pending_25_server_runs" if tables_ok and server["status"] != "complete"
            else "" if tables_ok and table3_consistent else ";".join(table_failures or ["table3_state_mismatch"])
        ),
    })

    source_data_path = ROOT / "figures" / "submission_staging" / "figure_source_data.csv"
    figure_data = pd.read_csv(source_data_path, keep_default_na=False)
    source_hash_ok = all(
        (ROOT / artifact).exists() and _sha256(ROOT / artifact) == artifact_hash
        for artifact, artifact_hash in figure_data[["artifact", "artifact_sha256"]].drop_duplicates().itertuples(index=False)
    )
    expected_staged_ids = {"Figure 1", "Figure 3", "Figure 4", "Figure 5"}
    if server["status"] == "complete":
        expected_staged_ids.add("Figure 2")
    all_staged_figures_have_source_rows = set(figure_data["submission_figure"].astype(str)) == expected_staged_ids
    figure_files = [
        ROOT / "figures" / "submission_staging" / f"Figure{index}_{stem}.{extension}"
        for index, stem in (
            (1, "Study_Design_and_Updating_Framework"),
            (3, "Clinical_Utility"),
            (4, "Calibration_and_Transportability"),
            (5, "Exploratory_Representation"),
        )
        for extension in ("png", "pdf")
    ]
    if server["status"] == "complete":
        figure_files.extend(
            ROOT / "figures" / "submission_staging" / f"Figure2_Architecture_Matched_Ablation.{extension}"
            for extension in ("png", "pdf")
        )
    figures_ready = (
        source_hash_ok
        and all_staged_figures_have_source_rows
        and figure_data["evidence_id"].astype(str).str.len().gt(0).all()
        and all(path.exists() for path in figure_files)
    )
    rows.append({
        "deliverable": "final_figures",
        "status": (
            "submission_staging_5_of_5_figures" if figures_ready and server["status"] == "complete"
            else "submission_staging_4_of_5_figures" if figures_ready else "failed_provenance_audit"
        ),
        "evidence": "figures/submission_staging/provenance.json",
        "remaining_gate": "figure_2_requires_25_accepted_server_runs" if server["status"] != "complete" else "",
    })

    supplementary_figure_provenance = (
        ROOT / "figures" / "submission_staging" / "SupplementaryFigureS1_provenance.json"
    )
    supplementary_figure_failures = []
    if not supplementary_figure_provenance.is_file():
        supplementary_figure_failures.append("missing_supplementary_figure_s1_provenance")
    else:
        supplementary_payload = json.loads(supplementary_figure_provenance.read_text(encoding="utf-8"))
        if supplementary_payload.get("training_performed") is not False:
            supplementary_figure_failures.append("supplementary_figure_training_boundary_missing")
        if len(supplementary_payload.get("registered_run_ids", [])) != 5:
            supplementary_figure_failures.append("supplementary_figure_run_count_not_5")
        if set(supplementary_payload.get("seeds", [])) != {7, 13, 42, 99, 2024}:
            supplementary_figure_failures.append("supplementary_figure_seed_set_mismatch")
        for artifact, expected_hash in supplementary_payload.get("sources", {}).items():
            source = ROOT / artifact
            if not source.is_file() or _sha256(source) != expected_hash:
                supplementary_figure_failures.append(f"supplementary_figure_source_mismatch:{artifact}")
        for name, expected_hash in supplementary_payload.get("outputs", {}).items():
            output = supplementary_figure_provenance.parent / name
            if not output.is_file() or _sha256(output) != expected_hash:
                supplementary_figure_failures.append(f"supplementary_figure_output_mismatch:{name}")
    if subgroup_complete:
        subgroup_figure_provenance = (
            ROOT / "figures" / "submission_staging" / "SupplementaryFigureS2_provenance.json"
        )
        if not subgroup_figure_provenance.is_file():
            supplementary_figure_failures.append("missing_supplementary_figure_s2_provenance")
        else:
            subgroup_figure_payload = json.loads(subgroup_figure_provenance.read_text(encoding="utf-8"))
            if (
                subgroup_figure_payload.get("training_performed") is not False
                or subgroup_figure_payload.get("status")
                != "registered_confirmatory_v2_patient_cluster_subgroup_figure"
            ):
                supplementary_figure_failures.append("supplementary_figure_s2_status_invalid")
            for artifact, expected_hash in subgroup_figure_payload.get("sources", {}).items():
                source = ROOT / artifact
                if not source.is_file() or _sha256(source) != expected_hash:
                    supplementary_figure_failures.append(f"supplementary_figure_s2_source_mismatch:{artifact}")
            for name, expected_hash in subgroup_figure_payload.get("outputs", {}).items():
                output = subgroup_figure_provenance.parent / name
                if not output.is_file() or _sha256(output) != expected_hash:
                    supplementary_figure_failures.append(f"supplementary_figure_s2_output_mismatch:{name}")
    rows.append({
        "deliverable": "supplementary_figures",
        "status": (
            "ready" if not supplementary_figure_failures and subgroup_complete
            else "s1_ready_s2_pending_subgroups" if not supplementary_figure_failures
            else "failed_provenance_audit"
        ),
        "evidence": "figures/submission_staging/SupplementaryFigureS1_provenance.json",
        "remaining_gate": (
            "" if not supplementary_figure_failures and subgroup_complete
            else "supplementary_figure_s2_requires_locked_patient_cluster_subgroup_results"
            if not supplementary_figure_failures else ";".join(supplementary_figure_failures)
        ),
    })

    manuscript_path = ROOT / "docs" / "manuscript_draft.md"
    manuscript = manuscript_path.read_text(encoding="utf-8")
    metadata_blockers = re.findall(r"\[(?:BLOCKER|REFERENCE REQUIRED|TO UPDATE)[^\]]*\]", manuscript)
    claim_registry = build_claim_registry()
    manuscript_ready = (
        not metadata_blockers
        and claim_registry["status"] == "passed"
        and confirmatory_registry_complete
        and subgroup_complete
    )
    rows.append({
        "deliverable": "frozen_submission_manuscript",
        "status": "ready" if manuscript_ready else "blocked",
        "evidence": "docs/manuscript_draft.md",
        "remaining_gate": (
            "" if manuscript_ready else
            f"metadata_or_reference_placeholders={len(metadata_blockers)};claims={claim_registry['status']};server_confirmatory={server['status']};subgroups={'complete' if subgroup_complete else 'pending'}"
        ),
    })

    ready_statuses = {"ready", "ready_exploratory_only", "registered_draft_figures_ready", "ready_with_historical_input_byte_limitation"}
    overall_ready = all(row["status"] in ready_statuses for row in rows)
    return {
        "audit_version": "submission_package_gate_v1",
        "overall_status": "ready" if overall_ready else "not_ready",
        "training_performed": False,
        "deliverables": rows,
        "server_acceptance_summary": {
            key: server[key]
            for key in ("status", "expected_runs", "accepted_server_runs", "pending_server_runs", "rejected_artifacts")
        },
        "manuscript_placeholder_count": len(metadata_blockers),
        "manuscript_claim_registry_status": claim_registry["status"],
        "manuscript_quantitative_paragraphs": claim_registry["quantitative_paragraphs"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args()
    report = audit()
    json_output = args.json_output if args.json_output.is_absolute() else ROOT / args.json_output
    csv_output = args.csv_output if args.csv_output.is_absolute() else ROOT / args.csv_output
    json_output.parent.mkdir(parents=True, exist_ok=True)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = ["deliverable", "status", "evidence", "remaining_gate"]
    with csv_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(report["deliverables"])
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.require_ready and report["overall_status"] != "ready":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
