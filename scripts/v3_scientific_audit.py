"""Audit whether V3 is scientifically specified and whether training is authorized."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "v3"
AUDIT_CONFIG = V3 / "scientific_audit.yaml"
EVIDENCE_TREE = V3 / "evidence_tree.yaml"
EXPERIMENT_MATRIX = V3 / "experiments" / "experiment_matrix.csv"
RANDOM_PLAN = V3 / "experiments" / "random_control_plan.yaml"
TRANSPORT_PLAN = V3 / "experiments" / "transportability_group_plan.csv"
PHASE_STATUS = V3 / "registry" / "phase0_status.json"
DATASET_MANIFEST = V3 / "data_contract" / "dataset_manifest.csv"
FEATURE_MANIFEST = V3 / "data_contract" / "feature_manifest.csv"
SPLIT_MANIFEST = V3 / "data_contract" / "split_manifest.csv"
PREPROCESSING_MANIFEST = V3 / "preprocessing" / "preprocessing_manifest.csv"
RUN_REGISTRY = V3 / "registry" / "run_registry.csv"

ALLOWED_ROOTS = {"E1", "E2", "E3", "E4", "E5"}
REQUIRED_NODE_FIELDS = {
    "node_id",
    "parent_id",
    "label",
    "scientific_question",
    "claim_scope",
    "protocol_status",
    "evidence_status",
    "required_before_model_training",
    "inputs",
    "experiment_ids",
    "estimands",
    "statistical_unit",
    "statistics",
    "outputs",
    "falsification",
    "blockers",
}
ALLOWED_PROTOCOL_STATUS = {"ready", "blocked", "scope_limited", "draft"}


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in {path.relative_to(ROOT)}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit() -> dict[str, Any]:
    errors: list[str] = []
    blockers: list[str] = []
    warnings: list[str] = []
    downstream_gates: list[str] = []
    execution_gates: list[str] = []
    scope_limitations: list[str] = []

    try:
        config = _read_yaml(AUDIT_CONFIG)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        return {
            "audit_version": "unknown",
            "status": "invalid",
            "structure_passed": False,
            "training_ready": False,
            "errors": [f"unreadable_audit_config:{exc}"],
            "blockers": [],
            "downstream_gates": [],
            "execution_gates": [],
            "scope_limitations": [],
            "warnings": [],
        }

    for relative in config.get("required_artifacts", []):
        if not (ROOT / str(relative)).is_file():
            errors.append(f"missing_required_artifact:{relative}")

    contract_path = ROOT / str(config.get("research_contract", "CLAUDE.md"))
    contract_text = contract_path.read_text(encoding="utf-8") if contract_path.is_file() else ""
    protocol_version = str(config.get("data_protocol_version", "")).strip()
    if protocol_version and protocol_version not in contract_text:
        errors.append(f"research_contract_missing_data_protocol:{protocol_version}")
    for relative in config.get("data_entrypoints", []):
        entrypoint = ROOT / str(relative)
        if not entrypoint.is_file():
            errors.append(f"missing_data_entrypoint:{relative}")
        if str(relative) not in contract_text:
            errors.append(f"research_contract_missing_data_entrypoint:{relative}")
    shared_relative = str(config.get("shared_data_implementation", "")).strip()
    shared_path = ROOT / shared_relative
    if shared_relative and not shared_path.is_file():
        errors.append(f"missing_shared_data_implementation:{shared_relative}")
    elif protocol_version and f'DATA_PROTOCOL_VERSION = "{protocol_version}"' not in shared_path.read_text(
        encoding="utf-8"
    ):
        errors.append(f"shared_data_protocol_version_mismatch:{protocol_version}")

    tree = _read_yaml(EVIDENCE_TREE)
    nodes = tree.get("nodes", [])
    if not isinstance(nodes, list) or not nodes:
        errors.append("evidence_tree_has_no_nodes")
        nodes = []

    node_ids: set[str] = set()
    experiment_to_node: dict[str, str] = {}
    branches: set[str] = set()
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            errors.append(f"evidence_node_not_mapping:{index}")
            continue
        missing = sorted(REQUIRED_NODE_FIELDS - set(node))
        if missing:
            errors.append(f"evidence_node_missing_fields:{node.get('node_id', index)}:{','.join(missing)}")
        node_id = str(node.get("node_id", ""))
        parent_id = str(node.get("parent_id", ""))
        if not node_id:
            errors.append(f"empty_evidence_node_id:{index}")
            continue
        if node_id in node_ids:
            errors.append(f"duplicate_evidence_node_id:{node_id}")
        node_ids.add(node_id)
        branches.add(parent_id)
        if parent_id not in ALLOWED_ROOTS:
            errors.append(f"invalid_evidence_parent:{node_id}:{parent_id}")
        if node.get("protocol_status") not in ALLOWED_PROTOCOL_STATUS:
            errors.append(f"invalid_protocol_status:{node_id}:{node.get('protocol_status')}")
        for field in ("inputs", "outputs"):
            if not node.get(field):
                errors.append(f"empty_{field}:{node_id}")
        if not str(node.get("statistical_unit", "")).strip():
            errors.append(f"empty_statistical_unit:{node_id}")
        if not str(node.get("falsification", "")).strip():
            errors.append(f"empty_falsification:{node_id}")
        if node.get("claim_scope") != "not_identifiable_in_two_center_retrospective_data":
            if not node.get("estimands"):
                errors.append(f"empty_estimands:{node_id}")
            if not node.get("statistics"):
                errors.append(f"empty_statistics:{node_id}")
            if not node.get("experiment_ids"):
                errors.append(f"empty_experiment_ids:{node_id}")
        for experiment_id in node.get("experiment_ids", []):
            experiment_id = str(experiment_id)
            if experiment_id in experiment_to_node:
                errors.append(
                    f"experiment_mapped_to_multiple_nodes:{experiment_id}:"
                    f"{experiment_to_node[experiment_id]}:{node_id}"
                )
            experiment_to_node[experiment_id] = node_id
        if bool(node.get("required_before_model_training")) and node.get("protocol_status") != "ready":
            blockers.append(f"pretraining_evidence_protocol_not_ready:{node_id}")

    missing_branches = sorted(set(config.get("required_branches", [])) - branches)
    if missing_branches:
        errors.append(f"missing_evidence_branches:{','.join(missing_branches)}")

    matrix_fields, matrix_rows = _read_csv(EXPERIMENT_MATRIX)
    required_matrix_fields = {
        "evidence_node_id",
        "experiment_id",
        "scientific_question",
        "primary_comparison_or_estimand",
        "statistical_unit",
        "planned_output",
        "protocol_status",
    }
    if not required_matrix_fields.issubset(matrix_fields):
        errors.append(
            "experiment_matrix_missing_fields:"
            + ",".join(sorted(required_matrix_fields - set(matrix_fields)))
        )
    matrix_ids: set[str] = set()
    for row in matrix_rows:
        experiment_id = row.get("experiment_id", "").strip()
        node_id = row.get("evidence_node_id", "").strip()
        if not experiment_id:
            errors.append("experiment_matrix_empty_id")
            continue
        if experiment_id in matrix_ids:
            errors.append(f"duplicate_experiment_id:{experiment_id}")
        matrix_ids.add(experiment_id)
        if node_id not in node_ids:
            errors.append(f"experiment_unknown_evidence_node:{experiment_id}:{node_id}")
        if experiment_to_node.get(experiment_id) != node_id:
            errors.append(f"experiment_tree_matrix_mismatch:{experiment_id}:{node_id}")
        for field in ("scientific_question", "primary_comparison_or_estimand", "statistical_unit", "planned_output"):
            if not row.get(field, "").strip():
                errors.append(f"experiment_empty_{field}:{experiment_id}")
    for experiment_id, node_id in experiment_to_node.items():
        if experiment_id not in matrix_ids:
            errors.append(f"tree_experiment_missing_from_matrix:{experiment_id}:{node_id}")

    random_plan = _read_yaml(RANDOM_PLAN)
    observed_families = set(random_plan.get("families", {}))
    required_families = set(config.get("required_random_control_families", []))
    if observed_families != required_families:
        errors.append(
            "random_control_family_mismatch:"
            f"missing={sorted(required_families - observed_families)}:"
            f"unexpected={sorted(observed_families - required_families)}"
        )
    if random_plan.get("status") != "locked":
        blockers.append("random_control_plan_not_locked")
    random_artifact = ROOT / str(random_plan.get("partition_artifact", ""))
    if not random_artifact.is_file():
        errors.append("random_partition_artifact_missing")
    elif _sha256(random_artifact) != random_plan.get("partition_artifact_hash"):
        errors.append("random_partition_artifact_hash_mismatch")
    clinical_partition = ROOT / "v3/experiments/primary_feature_partition.json"
    if clinical_partition.is_file() and _sha256(clinical_partition) != random_plan.get(
        "clinical_partition_hash"
    ):
        errors.append("clinical_partition_hash_mismatch")
    primary_random = random_plan.get("families", {}).get("C_group_size_matched", {})
    if int(primary_random.get("minimum_partitions", 0)) < 19:
        errors.append("primary_random_null_has_fewer_than_19_partitions")

    _, transport_rows = _read_csv(TRANSPORT_PLAN)
    if not transport_rows:
        errors.append("transportability_group_plan_empty")
    if not any(row.get("group_id") == "TR-MEASUREMENT" for row in transport_rows):
        errors.append("transportability_plan_hides_measurement_feature_gap")
    if any(row.get("primary_metric") == "auc_target_over_auc_source" for row in transport_rows):
        errors.append("naive_auc_ratio_used_as_primary_transportability_metric")

    phase = json.loads(PHASE_STATUS.read_text(encoding="utf-8"))
    if not bool(phase.get("training_authorized")):
        blockers.append("phase_status_training_authorized_false")
    blockers.extend(f"phase_gate:{reason}" for reason in phase.get("blocking_gates", []))
    downstream_gates.extend(str(reason) for reason in phase.get("downstream_gates", []))
    execution_gates.extend(str(reason) for reason in phase.get("execution_gates", []))
    scope_limitations.extend(str(reason) for reason in phase.get("scope_limitations", []))

    _, dataset_rows = _read_csv(DATASET_MANIFEST)
    if not dataset_rows or any("frozen" not in row.get("freeze_status", "") for row in dataset_rows):
        blockers.append("v3_dataset_not_frozen")
    elif any("not_v3_frozen" in row.get("freeze_status", "") for row in dataset_rows):
        blockers.append("v3_dataset_not_frozen")
    for row in dataset_rows:
        processed = ROOT / row.get("processed_path", "")
        if not processed.is_file():
            errors.append(f"frozen_processed_file_missing:{row.get('dataset_id', '')}")
        elif _sha256(processed) != row.get("processed_hash"):
            errors.append(f"frozen_processed_hash_mismatch:{row.get('dataset_id', '')}")
        feature_table = ROOT / row.get("candidate_feature_table_path", "")
        if not feature_table.is_file():
            errors.append(f"frozen_feature_table_missing:{row.get('dataset_id', '')}")
        elif _sha256(feature_table) != row.get("candidate_feature_table_hash"):
            errors.append(f"frozen_feature_table_hash_mismatch:{row.get('dataset_id', '')}")

    _, feature_rows = _read_csv(FEATURE_MANIFEST)
    if not feature_rows:
        blockers.append("v3_feature_ontology_empty")
    if not str(phase.get("feature_protocol_approval", "")).strip():
        blockers.append("v3_feature_ontology_not_clinically_approved")
    if any(row.get("candidate_inclusion_status", "").startswith("blocked") for row in feature_rows):
        blockers.append("v3_feature_data_quality_blocks_unresolved")

    _, split_rows = _read_csv(SPLIT_MANIFEST)
    if not split_rows:
        blockers.append("v3_split_manifest_empty")
    patient_roles: dict[tuple[str, str], set[str]] = {}
    for row in split_rows:
        key = (row.get("center", ""), row.get("patient_id", ""))
        patient_roles.setdefault(key, set()).add(row.get("split_role", ""))
    overlap = [key for key, roles in patient_roles.items() if len(roles) > 1]
    if overlap:
        errors.append(f"split_patient_role_overlap:{len(overlap)}")
    if any(row.get("patient_overlap_check") != "passed_within_center_roles" for row in split_rows):
        errors.append("split_overlap_check_not_passed")
    if any(row.get("temporal_leakage_check") != "passed_strictly_prior_history" for row in split_rows):
        errors.append("split_temporal_check_not_passed")

    _, preprocessing_rows = _read_csv(PREPROCESSING_MANIFEST)
    if not preprocessing_rows or any(row.get("lock_status") != "locked" for row in preprocessing_rows):
        blockers.append("v3_preprocessing_not_locked")
    for row in preprocessing_rows:
        artifact = ROOT / row.get("artifact_path", "")
        if not artifact.is_file():
            errors.append(f"preprocessing_artifact_missing:{row.get('preprocessing_id', '')}")
        elif _sha256(artifact) != row.get("artifact_hash"):
            errors.append(f"preprocessing_artifact_hash_mismatch:{row.get('preprocessing_id', '')}")

    _, run_rows = _read_csv(RUN_REGISTRY)
    historical_contamination = [
        row.get("run_id", "")
        for row in run_rows
        if row.get("contract_version", "").lower().startswith(("v1", "v2", "dual_binary"))
    ]
    if historical_contamination:
        errors.append(f"historical_runs_in_v3_registry:{','.join(historical_contamination)}")
    if run_rows and not bool(phase.get("training_authorized")):
        errors.append("v3_runs_exist_while_training_unauthorized")

    blockers = sorted(set(blockers))
    downstream_gates = sorted(set(downstream_gates))
    execution_gates = sorted(set(execution_gates))
    scope_limitations = sorted(set(scope_limitations))
    warnings = sorted(set(warnings))
    errors = sorted(set(errors))
    structure_passed = not errors
    data_preflight_blockers = [
        item for item in blockers if item != "phase_status_training_authorized_false"
    ]
    data_preflight_ready = structure_passed and not data_preflight_blockers
    training_ready = data_preflight_ready and not blockers and not execution_gates
    return {
        "audit_version": config.get("audit_version"),
        "status": "ready" if training_ready else ("blocked" if structure_passed else "invalid"),
        "structure_passed": structure_passed,
        "data_preflight_ready": data_preflight_ready,
        "training_ready": training_ready,
        "counts": {
            "evidence_nodes": len(nodes),
            "experiments": len(matrix_rows),
            "transportability_groups": len(transport_rows),
            "random_control_families": len(observed_families),
            "v3_registered_runs": len(run_rows),
        },
        "errors": errors,
        "blockers": blockers,
        "execution_gates": execution_gates,
        "downstream_gates": downstream_gates,
        "scope_limitations": scope_limitations,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-structure", action="store_true")
    parser.add_argument("--require-training-ready", action="store_true")
    args = parser.parse_args()
    report = audit()
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    if args.require_structure and not report["structure_passed"]:
        raise SystemExit("V3 scientific structure audit failed.")
    if args.require_training_ready and not report["training_ready"]:
        raise SystemExit("V3 training remains blocked by the scientific audit.")


if __name__ == "__main__":
    main()
