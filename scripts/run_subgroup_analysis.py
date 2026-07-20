"""Run only fully prespecified subgroup analyses on one accepted v2 server run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.verify_server_confirmatory_results import audit as audit_server_results
from src.evaluate.subgroup_cluster import assign_prespecified_levels, patient_cluster_subgroup_analysis


DEFAULT_PROTOCOL = ROOT / "conf" / "subgroup_protocol.yaml"
DEFAULT_OUTPUT = ROOT / "subgroup_analysis_patient_bootstrap.csv"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("status") != "prespecified_locked":
        raise SystemExit("Subgroup analysis not started: protocol status must be prespecified_locked.")
    if protocol.get("prespecified_without_target_test_access") is not True:
        raise SystemExit("Subgroup analysis not started: no-test-access attestation is required.")
    prespecification = protocol.get("prespecification", {})
    required_lock_fields = (
        "locked_at_utc", "clinical_approver_role", "statistical_approver_role", "rationale_document"
    )
    if any(
        not isinstance(prespecification.get(field), str)
        or not prespecification[field].strip()
        or prespecification[field].strip() == "REQUIRED"
        for field in required_lock_fields
    ):
        raise SystemExit("Subgroup analysis not started: complete prespecification lock metadata is required.")
    run_id = protocol.get("run_id")
    if not isinstance(run_id, str) or not run_id.startswith("binary_"):
        raise SystemExit("Subgroup analysis not started: an accepted confirmatory-v2 run_id is required.")
    minimum = protocol.get("reporting", {}).get("minimum_unique_patients_per_subgroup")
    if not isinstance(minimum, int) or minimum < 10:
        raise SystemExit("Subgroup analysis not started: minimum patients per level must be an integer >=10.")
    for definitions in protocol.get("endpoints", {}).values():
        for item in definitions:
            if not isinstance(item.get("definition"), dict):
                raise SystemExit("Subgroup analysis not started: every subgroup needs an explicit definition.")
            assign_prespecified_levels(pd.Series([0, 1]), item["definition"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run", action="store_true", help="Explicit authorization after protocol lock.")
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    _validate_protocol(protocol)
    if not args.run:
        raise SystemExit("Subgroup analysis not started. Add --run after protocol review and lock.")

    acceptance = audit_server_results()
    if acceptance["status"] != "complete":
        raise SystemExit("Subgroup analysis not started: all 25 confirmatory runs must pass acceptance first.")
    accepted = {row["run_id"]: row for row in acceptance["accepted"]}
    run_id = protocol["run_id"]
    if run_id not in accepted:
        raise SystemExit("Subgroup analysis not started: run_id did not pass server acceptance audit.")
    if accepted[run_id]["model_id"] != "D_dual_outcome_specific_coral":
        raise SystemExit("Subgroup analysis not started: run_id must be an accepted outcome-specific CORAL run.")
    central_registry = pd.read_csv(
        ROOT / "experiments" / "evidence_registry" / "run_registry.csv", keep_default_na=False
    )
    registered = central_registry.loc[
        central_registry["run_id"].eq(run_id)
        & central_registry["contract"].eq("dual_binary_hemodynamic_confirmatory_v2")
        & central_registry["confirmatory_eligibility"].eq("accepted_architecture_matched_server_run")
    ]
    if len(registered) != 1:
        raise SystemExit("Subgroup analysis not started: accepted run_id is not in the central v2 run registry.")
    run_dir = ROOT / "experiments" / "confirmatory_results" / run_id
    predictions = pd.read_csv(run_dir / "test_predictions.csv")
    target = pd.read_csv(ROOT / "data" / "confirmatory_v2" / "target_confirmatory_v2.csv", low_memory=False)
    features = sorted({item["candidate_feature"] for items in protocol["endpoints"].values() for item in items})
    feature_frame = target[["session_id", "患者id", *features]].rename(columns={"患者id": "patient_id"})
    analysis = predictions.merge(
        feature_frame,
        on=["session_id", "patient_id"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not analysis["_merge"].eq("both").all():
        raise ValueError("Accepted predictions do not map one-to-one to confirmatory-v2 target sessions")
    analysis = analysis.drop(columns="_merge")

    rows = []
    for endpoint, definitions in protocol["endpoints"].items():
        for item in definitions:
            level_col = f"__level_{endpoint}_{item['subgroup']}"
            analysis[level_col] = assign_prespecified_levels(analysis[item["candidate_feature"]], item["definition"])
            levels = list(item["definition"].get("labels", item["definition"].get("levels", {})))
            estimates = patient_cluster_subgroup_analysis(
                analysis,
                level_col=level_col,
                outcome_col=f"{endpoint}_event",
                updated_probability_col=f"probability_{endpoint}_updated_mlp",
                source_probability_col=f"probability_{endpoint}_source_mlp",
                patient_col="patient_id",
                levels=[str(level) for level in levels],
                minimum_patients=int(protocol["reporting"]["minimum_unique_patients_per_subgroup"]),
                replicates=int(protocol["resampling"]["replicates"]),
                seed=int(protocol["resampling"]["seed"]),
            )
            for estimate in estimates:
                rows.append({
                    "run_id": run_id,
                    "endpoint": endpoint.upper(),
                    "subgroup": item["subgroup"],
                    "candidate_feature": item["candidate_feature"],
                    "definition": json.dumps(item["definition"], ensure_ascii=False, sort_keys=True),
                    **estimate,
                    "bootstrap_unit": "patient_id",
                    "bootstrap_replicates": int(protocol["resampling"]["replicates"]),
                    "status": "registered_confirmatory_v2_patient_cluster_subgroup",
                })
    output = args.output.resolve()
    fields = list(rows[0])
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    provenance = {
        "status": "registered_confirmatory_v2_patient_cluster_subgroup",
        "training_performed": False,
        "run_id": run_id,
        "protocol": {"path": str(protocol_path.relative_to(ROOT)), "sha256": _sha256(protocol_path)},
        "predictions": {"path": str((run_dir / "test_predictions.csv").relative_to(ROOT)), "sha256": _sha256(run_dir / "test_predictions.csv")},
        "target_v2_sha256": _sha256(ROOT / "data" / "confirmatory_v2" / "target_confirmatory_v2.csv"),
        "output": {"path": str(output.relative_to(ROOT)), "sha256": _sha256(output)},
    }
    (output.parent / "subgroup_analysis_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {len(rows)} registered subgroup rows to {output}")


if __name__ == "__main__":
    main()
