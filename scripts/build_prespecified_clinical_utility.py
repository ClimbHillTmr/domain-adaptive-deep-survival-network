"""Build clinical-threshold utility outputs only after a locked protocol passes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_clinical_utility import (
    _clinical_bootstrap,
    _confusion,
    _net_benefit,
    _patient_cluster_indices,
)


DEFAULT_PROTOCOL = ROOT / "conf" / "clinical_threshold_protocol.yaml"
DEFAULT_OUTPUT = ROOT / "clinical_utility_report" / "prespecified"
REGISTRY = ROOT / "experiments" / "evidence_registry" / "run_registry.csv"
MODEL_LABELS = {
    "source_mlp": "Source MLP",
    "updated_mlp": "Updated MLP",
    "local_logistic": "Target-local logistic",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _validate_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("protocol_version") != "clinical_threshold_prespecification_v1":
        raise SystemExit("Prespecified clinical utility not generated: unexpected protocol version.")
    if protocol.get("status") != "prespecified_locked":
        raise SystemExit("Prespecified clinical utility not generated: protocol status must be prespecified_locked.")
    if protocol.get("prespecified_without_target_test_dca_access") is not True:
        raise SystemExit("Prespecified clinical utility not generated: no-test-DCA-access attestation is required.")
    lock = protocol.get("lock", {})
    required_lock_fields = (
        "locked_at_utc", "nephrology_approver_role", "statistical_approver_role",
        "clinical_workflow_approver_role", "rationale_document",
    )
    if any(
        not isinstance(lock.get(field), str)
        or not lock[field].strip()
        or lock[field].strip() == "REQUIRED"
        for field in required_lock_fields
    ):
        raise SystemExit("Prespecified clinical utility not generated: lock metadata is incomplete.")
    use = protocol.get("population_and_use", {})
    if (
        not isinstance(use.get("alert_action"), str)
        or not use["alert_action"].strip()
        or use["alert_action"].startswith("REQUIRED")
    ):
        raise SystemExit("Prespecified clinical utility not generated: clinical alert action is undefined.")
    for endpoint in ("idh", "ih"):
        item = protocol.get("endpoints", {}).get(endpoint, {})
        threshold_range = item.get("dca_threshold_range", {})
        values = [threshold_range.get(name) for name in ("lower", "upper", "step")]
        if not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
            raise SystemExit(f"Prespecified clinical utility not generated: {endpoint} DCA range is incomplete.")
        lower, upper, step = map(float, values)
        if not (0.01 <= lower < upper <= 0.60 and 0 < step <= upper - lower):
            raise SystemExit(f"Prespecified clinical utility not generated: {endpoint} DCA range is invalid.")
        thresholds = item.get("operating_thresholds")
        if (
            not isinstance(thresholds, list)
            or not thresholds
            or any(not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 < float(value) < 1 for value in thresholds)
            or len({float(value) for value in thresholds}) != len(thresholds)
        ):
            raise SystemExit(f"Prespecified clinical utility not generated: {endpoint} operating thresholds are invalid.")
        for field in ("false_alert_consequence", "missed_event_consequence", "clinical_rationale"):
            value = item.get(field)
            if not isinstance(value, str) or not value.strip() or value.startswith("REQUIRED"):
                raise SystemExit(f"Prespecified clinical utility not generated: {endpoint} {field} is incomplete.")
    reporting = protocol.get("reporting", {})
    if (
        reporting.get("models") != ["source_mlp", "updated_mlp", "local_logistic"]
        or reporting.get("comparators") != ["alert_all", "alert_none"]
        or reporting.get("bootstrap_unit") != "patient_id"
        or reporting.get("bootstrap_replicates") != 1000
        or reporting.get("bootstrap_seed") != 20260715
    ):
        raise SystemExit("Prespecified clinical utility not generated: reporting contract changed.")


def _grid(lower: float, upper: float, step: float) -> list[float]:
    start, stop, increment = map(Decimal, (str(lower), str(upper), str(step)))
    values = []
    current = start
    while current <= stop:
        values.append(float(current))
        current += increment
    if abs(values[-1] - float(stop)) > 1e-12:
        raise SystemExit("Prespecified clinical utility not generated: DCA step does not land on the upper bound.")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--build", action="store_true", help="Explicit generation after protocol lock.")
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    _validate_protocol(protocol)
    if not args.build:
        raise SystemExit("Prespecified clinical utility not generated. Add --build after protocol review.")
    run_id = protocol["run_id"]
    registry = pd.read_csv(REGISTRY, keep_default_na=False)
    match = registry.loc[
        registry["run_id"].eq(run_id)
        & registry["contract"].eq("dual_binary_hemodynamic_v1")
        & registry["artifact_complete"].astype(str).str.lower().eq("true")
    ]
    if len(match) != 1:
        raise SystemExit("Prespecified clinical utility not generated: run_id is not one registered historical run.")
    record = match.iloc[0]
    prediction_path = ROOT / record["prediction_file"]
    evaluation_path = ROOT / record["evaluation_file"]
    if _sha256(prediction_path) != record["prediction_sha256"] or _sha256(evaluation_path) != record["evaluation_sha256"]:
        raise SystemExit("Prespecified clinical utility not generated: registered source artifact hash mismatch.")
    predictions = pd.read_csv(prediction_path, low_memory=False)
    patient_ids = predictions["patient_id"].astype(str).to_numpy()
    bootstrap_indices = _patient_cluster_indices(patient_ids, replicates=1000, seed=20260715)
    alert_rows: list[dict[str, Any]] = []
    dca_rows: list[dict[str, Any]] = []
    for endpoint in ("idh", "ih"):
        outcome = predictions[f"{endpoint}_event"].to_numpy(dtype=int)
        endpoint_protocol = protocol["endpoints"][endpoint]
        for model_key, label in MODEL_LABELS.items():
            probability = predictions[f"probability_{endpoint}_{model_key}"].to_numpy(dtype=float)
            for threshold in map(float, endpoint_protocol["operating_thresholds"]):
                bootstrap = _clinical_bootstrap(outcome, probability, threshold, bootstrap_indices)
                row = {
                    "run_id": run_id, "endpoint": endpoint.upper(), "model": label,
                    "threshold": threshold, "threshold_source": "clinically_prespecified_locked_protocol",
                    **_confusion(outcome, probability, threshold),
                    "bootstrap_unit": "patient_id", "bootstrap_replicates": 1000,
                    "bootstrap_valid_replicates": bootstrap["sensitivity"]["valid_replicates"],
                }
                for metric in (
                    "sensitivity", "specificity", "ppv", "npv", "events_detected_per_1000",
                    "false_alerts_per_1000", "total_alerts_per_1000",
                ):
                    row[f"{metric}_lower"] = bootstrap[metric]["lower"]
                    row[f"{metric}_upper"] = bootstrap[metric]["upper"]
                alert_rows.append(row)
            threshold_range = endpoint_protocol["dca_threshold_range"]
            for threshold in _grid(
                float(threshold_range["lower"]), float(threshold_range["upper"]), float(threshold_range["step"])
            ):
                dca_rows.append({
                    "run_id": run_id, "endpoint": endpoint.upper(), "strategy": label,
                    "threshold": threshold, "net_benefit": _net_benefit(outcome, probability, threshold),
                    "threshold_source": "clinically_prespecified_locked_protocol",
                    "interpretation_status": "population_level_research_not_clinical_recommendation",
                })
        prevalence = float(outcome.mean())
        threshold_range = endpoint_protocol["dca_threshold_range"]
        for threshold in _grid(
            float(threshold_range["lower"]), float(threshold_range["upper"]), float(threshold_range["step"])
        ):
            dca_rows.extend([
                {
                    "run_id": run_id, "endpoint": endpoint.upper(), "strategy": "Alert all",
                    "threshold": threshold,
                    "net_benefit": prevalence - (1 - prevalence) * threshold / (1 - threshold),
                    "threshold_source": "clinically_prespecified_locked_protocol",
                    "interpretation_status": "population_level_research_not_clinical_recommendation",
                },
                {
                    "run_id": run_id, "endpoint": endpoint.upper(), "strategy": "Alert none",
                    "threshold": threshold, "net_benefit": 0.0,
                    "threshold_source": "clinically_prespecified_locked_protocol",
                    "interpretation_status": "population_level_research_not_clinical_recommendation",
                },
            ])
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    alert_path = output / "prespecified_alert_analysis.csv"
    dca_path = output / "prespecified_decision_curve_data.csv"
    _write_csv(alert_path, alert_rows)
    _write_csv(dca_path, dca_rows)
    provenance = {
        "status": "registered_clinically_prespecified_population_level_utility",
        "training_performed": False,
        "run_id": run_id,
        "protocol": {"path": str(protocol_path.relative_to(ROOT)), "sha256": _sha256(protocol_path)},
        "predictions": {"path": str(prediction_path.relative_to(ROOT)), "sha256": _sha256(prediction_path)},
        "evaluation": {"path": str(evaluation_path.relative_to(ROOT)), "sha256": _sha256(evaluation_path)},
        "outputs": {
            str(alert_path.relative_to(ROOT)): _sha256(alert_path),
            str(dca_path.relative_to(ROOT)): _sha256(dca_path),
        },
        "interpretation": "population_level_research_not_clinical_recommendation",
    }
    (output / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(provenance, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
