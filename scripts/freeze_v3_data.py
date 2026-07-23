"""Build and freeze the V3 cohorts, patient split, and preprocessing artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.binary_dataset import prepare_binary_data, write_binary_metadata
from src.data_pipeline.HBD_data import EXPECTED_RAW_SHA256 as SOURCE_RAW_HASH
from src.data_pipeline.HBD_data import EXPECTED_SUMMARY as SOURCE_SUMMARY
from src.data_pipeline.HBD_data_fuding import EXPECTED_RAW_SHA256 as TARGET_RAW_HASH
from src.data_pipeline.HBD_data_fuding import EXPECTED_SUMMARY as TARGET_SUMMARY
from src.data_pipeline.data_process import DATA_PROTOCOL_VERSION, write_cohort
from src.evaluate.audit_prior_history import write_audit

RAW_ROOT = (
    Path.home()
    / ".cache/huggingface/hub/datasets--LongGoodbye--Shenyi-Fuding-original"
    / "snapshots/b125e7730d526b43c6cbd17dca3ff00b57b9cf1c"
)
OUTPUT_DIR = ROOT / "data" / "v3"
FEATURE_ALLOWLIST = ROOT / "v3" / "data_contract" / "feature_allowlist.csv"
DATASET_MANIFEST = ROOT / "v3" / "data_contract" / "dataset_manifest.csv"
SPLIT_MANIFEST = ROOT / "v3" / "data_contract" / "split_manifest.csv"
PREPROCESSING_MANIFEST = ROOT / "v3" / "preprocessing" / "preprocessing_manifest.csv"
SPLIT_SEED = 20260715
SOURCE_VALIDATION_FRACTION = 0.10
TARGET_UPDATE_FRACTION = 0.60
TARGET_VALIDATION_FRACTION = 0.15
DATASET_VERSION = "v3_hbd_20260723"
SPLIT_ID = "V3-SPLIT-20260715"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _split_rows(bundle: Any) -> list[dict[str, Any]]:
    roles = [
        ("V3-SOURCE-20260723", "source", "source_train", bundle.source_train),
        ("V3-SOURCE-20260723", "source", "source_validation", bundle.source_val),
        ("V3-TARGET-20260723", "target", "target_update", bundle.target_train),
        ("V3-TARGET-20260723", "target", "target_calibration", bundle.target_val),
        ("V3-TARGET-20260723", "target", "target_test", bundle.target_test),
    ]
    rows: list[dict[str, Any]] = []
    for dataset_id, center, role, frame in roles:
        work = frame[["患者id", "透析开始时间"]].copy()
        work["透析开始时间"] = pd.to_datetime(work["透析开始时间"], errors="raise")
        grouped = work.groupby("患者id", sort=True)["透析开始时间"]
        for patient_id, times in grouped:
            rows.append(
                {
                    "split_id": SPLIT_ID,
                    "dataset_id": dataset_id,
                    "center": center,
                    "patient_id": str(patient_id),
                    "split_role": role,
                    "first_session_time": times.min().isoformat(),
                    "last_session_time": times.max().isoformat(),
                    "session_count": int(times.size),
                    "assignment_seed": SPLIT_SEED,
                    "assignment_rule": (
                        "patient-level joint-endpoint stratification; source 90/10; "
                        "target 60% update pool then 85/15 update/calibration; 40% test"
                    ),
                    "patient_overlap_check": "passed_within_center_roles",
                    "temporal_leakage_check": "passed_strictly_prior_history",
                    "manifest_version": "v3_split_1",
                }
            )
    return rows


def _validate_bundle(bundle: Any) -> dict[str, Any]:
    role_frames = {
        "source_train": bundle.source_train,
        "source_validation": bundle.source_val,
        "target_update": bundle.target_train,
        "target_calibration": bundle.target_val,
        "target_test": bundle.target_test,
    }
    source_sets = [
        set(bundle.source_train["患者id"]),
        set(bundle.source_val["患者id"]),
    ]
    target_sets = [
        set(bundle.target_train["患者id"]),
        set(bundle.target_val["患者id"]),
        set(bundle.target_test["患者id"]),
    ]
    if source_sets[0] & source_sets[1]:
        raise ValueError("Source patient overlap detected.")
    if any(target_sets[i] & target_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Target patient overlap detected.")
    for role, frame in role_frames.items():
        values = frame[bundle.feature_names].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite transformed primary feature in {role}.")
    source_variances = bundle.source_train[bundle.feature_names].var(ddof=0)
    constant = source_variances.index[source_variances.eq(0)].tolist()
    if constant:
        raise ValueError(f"Constant source-training primary features: {constant}")
    return {
        role: {
            "patients": int(frame["患者id"].nunique()),
            "sessions": int(len(frame)),
            "idh_events": int(frame["idh_event"].sum()),
            "ih_events": int(frame["ih_event"].sum()),
        }
        for role, frame in role_frames.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-raw", type=Path, default=RAW_ROOT / "updated_dataset_shenyi.csv")
    parser.add_argument("--target-raw", type=Path, default=RAW_ROOT / "updated_dataset_fuding.csv")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--execute", action="store_true", help="Required to write frozen artifacts.")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("Refusing to write V3 data without --execute.")

    output = args.output_dir.resolve()
    source_path = output / "source_v3_hbd.csv"
    target_path = output / "target_v3_hbd.csv"
    protected = [source_path, target_path, output / "data_freeze_manifest.json"]
    existing = [str(path) for path in protected if path.exists()]
    if existing:
        raise SystemExit(f"Refusing to overwrite existing V3 frozen artifacts: {existing}")
    output.mkdir(parents=True, exist_ok=True)

    source_build = write_cohort(
        args.source_raw,
        "shenyi",
        source_path,
        expected_raw_sha256=SOURCE_RAW_HASH,
        expected_summary=SOURCE_SUMMARY,
    )
    target_build = write_cohort(
        args.target_raw,
        "fuding",
        target_path,
        expected_raw_sha256=TARGET_RAW_HASH,
        expected_summary=TARGET_SUMMARY,
    )
    forbidden_columns = {"姓名", "透析记录id", "RECIPE_ID", "Unnamed: 0", "出生日期", "平均动脉压"}
    for label, path in (("source", source_path), ("target", target_path)):
        columns = set(pd.read_csv(path, nrows=0).columns)
        leaked = sorted(columns & forbidden_columns)
        if leaked:
            raise ValueError(f"Forbidden identifier or future-information columns in {label}: {leaked}")
    source_history = write_audit(source_path, output / "source_prior_history_audit.json")
    target_history = write_audit(target_path, output / "target_prior_history_audit.json")
    if not source_history["passed"] or not target_history["passed"]:
        raise ValueError("Strictly-prior history audit failed.")

    bundle = prepare_binary_data(
        source_path,
        target_path,
        allowlist_path=FEATURE_ALLOWLIST,
        patient_col="患者id",
        split_seed=SPLIT_SEED,
        source_validation_fraction=SOURCE_VALIDATION_FRACTION,
        target_update_fraction=TARGET_UPDATE_FRACTION,
        target_validation_fraction=TARGET_VALIDATION_FRACTION,
    )
    feature_table = pd.read_csv(FEATURE_ALLOWLIST)
    primary_groups = dict(
        zip(
            feature_table.loc[feature_table["allowed"].eq("yes"), "feature"],
            feature_table.loc[feature_table["allowed"].eq("yes"), "primary_group"],
            strict=True,
        )
    )
    bundle.preprocessing["feature_names"] = bundle.feature_names
    bundle.preprocessing["primary_groups"] = primary_groups
    bundle.preprocessing["data_protocol_version"] = DATA_PROTOCOL_VERSION
    bundle.preprocessing["split_id"] = SPLIT_ID
    bundle.preprocessing["split_seed"] = SPLIT_SEED
    bundle.preprocessing["missingness_indicators_in_primary_model"] = []
    bundle.preprocessing["categorical_semantic_blocks"] = [
        "抗凝剂类型",
        "透析方式",
        "瘘管类型",
        "瘘管位置",
    ]
    split_summary = _validate_bundle(bundle)
    write_binary_metadata(bundle, output)

    split_fields = [
        "split_id",
        "dataset_id",
        "center",
        "patient_id",
        "split_role",
        "first_session_time",
        "last_session_time",
        "session_count",
        "assignment_seed",
        "assignment_rule",
        "patient_overlap_check",
        "temporal_leakage_check",
        "manifest_version",
    ]
    _atomic_csv(SPLIT_MANIFEST, split_fields, _split_rows(bundle))

    frozen_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    feature_hash = _sha256(FEATURE_ALLOWLIST)
    pipeline_hash = _sha256(ROOT / "src/data_pipeline/data_process.py")
    preprocessing_code_hash = _sha256(ROOT / "src/data/binary_dataset.py")
    preprocessing_path = output / "preprocessing.json"
    split_json_path = output / "split_manifest.json"
    dataset_fields = [
        "dataset_id", "dataset_version", "cohort", "source_system", "raw_path", "raw_hash",
        "processed_path", "processed_hash", "hash_algorithm", "raw_row_count",
        "processed_row_count", "patient_count", "session_count", "idh_events",
        "idh_prevalence", "ih_events", "ih_prevalence", "cohort_definition_version",
        "cohort_code_hash", "candidate_feature_table_path", "candidate_feature_table_hash",
        "freeze_status", "frozen_at_utc", "notes",
    ]
    dataset_rows = []
    for dataset_id, cohort, system, raw_path, raw_hash, processed_path, build in [
        ("V3-SOURCE-20260723", "source", "shenyi", args.source_raw, SOURCE_RAW_HASH, source_path, source_build),
        ("V3-TARGET-20260723", "target", "fuding", args.target_raw, TARGET_RAW_HASH, target_path, target_build),
    ]:
        dataset_rows.append(
            {
                "dataset_id": dataset_id,
                "dataset_version": DATASET_VERSION,
                "cohort": cohort,
                "source_system": system,
                "raw_path": str(raw_path),
                "raw_hash": raw_hash,
                "processed_path": str(processed_path.relative_to(ROOT)),
                "processed_hash": _sha256(processed_path),
                "hash_algorithm": "SHA-256",
                "raw_row_count": build["input_rows"],
                "processed_row_count": build["output_rows"],
                "patient_count": build["n_patients"],
                "session_count": build["output_rows"],
                "idh_events": build["n_idh_events"],
                "idh_prevalence": build["n_idh_events"] / build["output_rows"],
                "ih_events": build["n_ih_events"],
                "ih_prevalence": build["n_ih_events"] / build["output_rows"],
                "cohort_definition_version": DATA_PROTOCOL_VERSION,
                "cohort_code_hash": pipeline_hash,
                "candidate_feature_table_path": str(FEATURE_ALLOWLIST.relative_to(ROOT)),
                "candidate_feature_table_hash": feature_hash,
                "freeze_status": "v3_frozen",
                "frozen_at_utc": frozen_at,
                "notes": "Internal V3 revalidation cohort; historical target patients are not newly untouched.",
            }
        )
    _atomic_csv(DATASET_MANIFEST, dataset_fields, dataset_rows)

    preprocessing_fields = [
        "preprocessing_id", "manifest_version", "fit_cohort", "feature_manifest_hash",
        "categorical_encoding", "scaling", "imputation", "missingness_indicators",
        "history_rule", "tied_timestamp_rule", "unknown_category_rule", "artifact_path",
        "artifact_hash", "code_hash", "lock_status", "notes",
    ]
    preprocessing_rows = [
        {
            "preprocessing_id": "V3-PREP-20260723",
            "manifest_version": "v3_preprocessing_1",
            "fit_cohort": "source_train_patients_only",
            "feature_manifest_hash": feature_hash,
            "categorical_encoding": "none_in_primary_model_semantically_incomparable_categories_excluded",
            "scaling": "source_train_mean_and_population_sd",
            "imputation": "source_train_median_no_generic_zero_fill",
            "missingness_indicators": "temperature_flags_process_sensitivity_only_not_primary",
            "history_rule": "strictly_prior_patient_sessions",
            "tied_timestamp_rule": "same_timestamp_sessions_use_only_earlier_timestamps",
            "unknown_category_rule": "not_applicable_to_locked_primary_features",
            "artifact_path": str(preprocessing_path.relative_to(ROOT)),
            "artifact_hash": _sha256(preprocessing_path),
            "code_hash": preprocessing_code_hash,
            "lock_status": "locked",
            "notes": f"{len(bundle.feature_names)} primary features; split {SPLIT_ID}",
        }
    ]
    _atomic_csv(PREPROCESSING_MANIFEST, preprocessing_fields, preprocessing_rows)

    freeze_manifest = {
        "contract_version": "v3_research_contract_1",
        "data_protocol_version": DATA_PROTOCOL_VERSION,
        "dataset_version": DATASET_VERSION,
        "frozen_at_utc": frozen_at,
        "training_authorized": False,
        "source": source_build,
        "target": target_build,
        "source_prior_history_audit": source_history,
        "target_prior_history_audit": target_history,
        "feature_allowlist": str(FEATURE_ALLOWLIST.relative_to(ROOT)),
        "feature_allowlist_sha256": feature_hash,
        "primary_feature_count": len(bundle.feature_names),
        "primary_features": bundle.feature_names,
        "primary_groups": primary_groups,
        "split_id": SPLIT_ID,
        "split_summary": split_summary,
        "split_manifest_csv_sha256": _sha256(SPLIT_MANIFEST),
        "split_manifest_json_sha256": _sha256(split_json_path),
        "preprocessing_sha256": _sha256(preprocessing_path),
        "dataset_manifest_sha256": _sha256(DATASET_MANIFEST),
        "pipeline_code_sha256": pipeline_hash,
        "preprocessing_code_sha256": preprocessing_code_hash,
        "interpretation": "Internal V3 revalidation; not a newly untouched external target cohort.",
    }
    _atomic_json(output / "data_freeze_manifest.json", freeze_manifest)
    print(json.dumps(freeze_manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
