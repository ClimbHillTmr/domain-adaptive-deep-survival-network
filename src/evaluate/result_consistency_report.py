import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
AUDIT_DIR = ROOT / "experiments" / "audit"
RESULTS_DIR = ROOT / "experiments" / "results"
REPORT_PATH = AUDIT_DIR / "result_consistency_report.json"
STALE_EVIDENCE_MESSAGE = "这些结果文件存在，但不能作为当前审计版本的投稿证据。"


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def file_status(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    stat = path.stat()
    return {
        "exists": True,
        "path": str(path),
        "size_bytes": stat.st_size,
        "modified_at_epoch": stat.st_mtime,
    }


def build_report() -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "audit": {},
        "results": {},
        "checks": {},
        "inconsistencies": [],
        "can_be_used_for_submission": False,
        "summary": "",
    }

    split_path = AUDIT_DIR / "split_audit.json"
    manifest_path = AUDIT_DIR / "data_manifest.json"
    history_path = AUDIT_DIR / "history_feature_audit.csv"
    allowlist_path = AUDIT_DIR / "feature_allowlist.csv"
    eval_path = RESULTS_DIR / "evaluation_results.json"
    pred_meta_path = RESULTS_DIR / "prediction_metadata.json"
    pred_csv_path = RESULTS_DIR / "real_test_predictions.csv"

    for label, path in [
        ("data_manifest", manifest_path),
        ("split_audit", split_path),
        ("history_feature_audit", history_path),
        ("feature_allowlist", allowlist_path),
        ("evaluation_results", eval_path),
        ("prediction_metadata", pred_meta_path),
        ("real_test_predictions", pred_csv_path),
    ]:
        bucket = "audit" if "results" not in label and "prediction" not in label and label != "evaluation_results" else "results"
        report[bucket][label] = file_status(path)

    required_paths = [split_path, manifest_path, history_path, allowlist_path, eval_path, pred_meta_path, pred_csv_path]
    if not all(path.exists() for path in required_paths):
        missing = [str(path) for path in required_paths if not path.exists()]
        report["inconsistencies"].append({"type": "missing_files", "files": missing})
        report["summary"] = STALE_EVIDENCE_MESSAGE
        return report

    split = read_json(split_path)
    pred_meta = read_json(pred_meta_path)
    pred_df = pd.read_csv(pred_csv_path)

    expected_test_sessions = int(split["test"]["n_sessions"])
    expected_overlap_zero = {
        "overlap_train_val": int(split.get("overlap_train_val", -1)) == 0,
        "overlap_train_test": int(split.get("overlap_train_test", -1)) == 0,
        "overlap_val_test": int(split.get("overlap_val_test", -1)) == 0,
    }
    metadata_n_test = int(pred_meta.get("n_test", -1))
    prediction_rows = int(len(pred_df))

    report["checks"]["expected_test_sessions"] = expected_test_sessions
    report["checks"]["patient_overlap_zero"] = expected_overlap_zero
    report["checks"]["prediction_metadata_n_test"] = metadata_n_test
    report["checks"]["prediction_csv_rows"] = prediction_rows

    if metadata_n_test != expected_test_sessions:
        report["inconsistencies"].append(
            {
                "type": "prediction_metadata_mismatch",
                "expected": expected_test_sessions,
                "observed": metadata_n_test,
            }
        )
    if prediction_rows != expected_test_sessions:
        report["inconsistencies"].append(
            {
                "type": "prediction_row_count_mismatch",
                "expected": expected_test_sessions,
                "observed": prediction_rows,
            }
        )
    for overlap_name, is_zero in expected_overlap_zero.items():
        if not is_zero:
            report["inconsistencies"].append({"type": "patient_overlap_nonzero", "name": overlap_name})

    latest_audit_time = max(path.stat().st_mtime for path in [split_path, manifest_path, history_path, allowlist_path])
    eval_mtime = eval_path.stat().st_mtime
    pred_meta_mtime = pred_meta_path.stat().st_mtime
    pred_csv_mtime = pred_csv_path.stat().st_mtime
    old_run = any(ts < latest_audit_time for ts in [eval_mtime, pred_meta_mtime, pred_csv_mtime])
    report["checks"]["results_older_than_latest_audit"] = old_run
    if old_run:
        report["inconsistencies"].append(
            {
                "type": "stale_result_files",
                "latest_audit_epoch": latest_audit_time,
                "result_epochs": {
                    "evaluation_results": eval_mtime,
                    "prediction_metadata": pred_meta_mtime,
                    "real_test_predictions": pred_csv_mtime,
                },
            }
        )

    report["can_be_used_for_submission"] = len(report["inconsistencies"]) == 0
    report["summary"] = (
        "当前结果文件与审计版本一致，可作为投稿证据。"
        if report["can_be_used_for_submission"]
        else STALE_EVIDENCE_MESSAGE
    )
    return report


def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    report = build_report()
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
