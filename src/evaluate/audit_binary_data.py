"""Static preflight audit for the binary IDH data contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from src.data.binary_dataset import CATEGORICAL_COLUMNS, ENDPOINTS, ROOT, load_feature_allowlist


def audit_file(path: Path, features: list[str], patient_col: str) -> dict[str, object]:
    result: dict[str, object] = {"path": str(path), "exists": path.exists(), "errors": [], "warnings": []}
    if not path.exists():
        result["errors"].append("file_missing")
        return result
    header = pd.read_csv(path, nrows=0).columns.tolist()
    endpoint_columns = {column for endpoint in ENDPOINTS.values() for column in endpoint.values()}
    required = {patient_col, "session_id", *endpoint_columns}
    missing_required = sorted(required - set(header))
    missing_features = sorted(
        feature
        for feature in features
        if feature not in header
        and not (
            feature.endswith("_code")
            and feature.removesuffix("_code") in CATEGORICAL_COLUMNS
            and feature.removesuffix("_code") in header
        )
    )
    result["missing_required_columns"] = missing_required
    result["missing_allowed_features"] = missing_features
    if missing_required:
        result["errors"].append("missing_binary_contract_columns")
    if missing_features:
        result["errors"].append("missing_allowed_features")

    audit_columns = [
        patient_col,
        "session_id",
        "透中低血压_计算",
        "透中高血压_计算",
        *endpoint_columns,
    ]
    available = [column for column in audit_columns if column in header]
    if not endpoint_columns.issubset(available):
        return result
    frame = pd.read_csv(path, usecols=available, low_memory=False)
    result.update(
        {
            "sessions": int(len(frame)),
            "patients": int(frame[patient_col].nunique()) if patient_col in frame else None,
            "duplicate_session_ids": int(frame["session_id"].astype(str).duplicated().sum())
            if "session_id" in frame
            else None,
            "endpoints": {},
        }
    )
    if result["duplicate_session_ids"]:
        result["errors"].append("duplicate_session_ids")
    clinical_columns = {"idh": "透中低血压_计算", "ih": "透中高血压_计算"}
    for endpoint, columns in ENDPOINTS.items():
        events = pd.to_numeric(frame[columns["event"]], errors="coerce")
        times = pd.to_numeric(frame[columns["time"]], errors="coerce")
        endpoint_result = {
            "events": int(events.eq(1).sum()),
            "event_rate": float(events.eq(1).mean()),
            "missing_outcomes": int(events.isna().sum() + times.isna().sum()),
            "nonbinary_events": int((events.notna() & ~events.isin([0, 1])).sum()),
            "negative_times": int(times.lt(0).sum()),
            "non_events_with_nonzero_time": int((events.eq(0) & times.ne(0)).sum()),
        }
        clinical_col = clinical_columns[endpoint]
        if clinical_col in frame:
            clinical_event = pd.to_numeric(frame[clinical_col], errors="coerce")
            endpoint_result["clinical_flag_mismatch"] = int(
                (clinical_event.notna() & clinical_event.ne(events)).sum()
            )
        result["endpoints"][endpoint] = endpoint_result
        if endpoint_result["missing_outcomes"] or endpoint_result["nonbinary_events"] or endpoint_result["negative_times"]:
            result["errors"].append(f"{endpoint}_invalid_outcome_values")
        if endpoint_result["non_events_with_nonzero_time"]:
            result["errors"].append(f"{endpoint}_non_event_time_contract_violation")
        if endpoint_result.get("clinical_flag_mismatch"):
            result["errors"].append(f"{endpoint}_clinical_flag_mismatch")
    return result


def run_audit(config_path: Path) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    features = load_feature_allowlist(ROOT / config["data"]["feature_allowlist"])
    source = ROOT / config["data"]["source_file"]
    target = ROOT / config["data"]["target_file"]
    reports = {
        "source": audit_file(source, features, config["data"]["patient_col"]),
        "target": audit_file(target, features, config["data"]["patient_col"]),
    }
    passed = not any(report["errors"] for report in reports.values())
    return {
        "contract": "dual_binary_hemodynamic_v1",
        "endpoint_rules": config["project"]["endpoint_rules"],
        "passed": passed,
        "cohorts": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "conf" / "binary_config.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments" / "audit" / "binary_data_preflight.json")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    report = run_audit(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.strict and not report["passed"]:
        raise SystemExit("Binary data preflight failed; training is blocked.")


if __name__ == "__main__":
    main()
