import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import yaml

from src.reproducibility import prepare_locked_run_context, record_environment, snapshot_files, write_json


ROOT = Path(__file__).resolve().parents[2]
AUDIT_DIR = ROOT / "experiments" / "audit"
RESULT_DIR = ROOT / "experiments" / "results"
TABLE_DIR = ROOT / "tables"
FIG_ROOT = ROOT / "figures"


def load_config() -> Dict:
    with open(ROOT / "conf" / "config.yaml", "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def run_step(name: str, command: List[str], log_dir: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    log_path = log_dir / f"{name}.log"
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(f"$ {' '.join(command)}\n\n")
        log_file.flush()
        subprocess.run(command, cwd=ROOT, env=env, stdout=log_file, stderr=subprocess.STDOUT, check=True)


def required_artifacts() -> List[Path]:
    return [
        RESULT_DIR / "evaluation_results.json",
        RESULT_DIR / "real_test_predictions.csv",
        RESULT_DIR / "calibration_dca_metrics.json",
        RESULT_DIR / "prediction_metadata.json",
        FIG_ROOT / "Submission_Main_Figures" / "Fig1_Cohort_Split_Audit.pdf",
        FIG_ROOT / "Submission_Main_Figures" / "Fig2_Cross_Center_Shift.pdf",
        FIG_ROOT / "Submission_Main_Figures" / "Fig3_Performance_Comparison.pdf",
        FIG_ROOT / "Submission_Main_Figures" / "Fig4_Calibration_DCA.pdf",
        FIG_ROOT / "Submission_Supplementary_Figures" / "FigS1_Target_Subgroup_Burden.pdf",
        AUDIT_DIR / "result_consistency_report.json",
    ]


def snapshot_run_outputs(context: Dict[str, str]) -> None:
    snapshot_files(
        [
            ROOT / "conf" / "config.yaml",
            AUDIT_DIR / "data_manifest.json",
            AUDIT_DIR / "split_audit.json",
            AUDIT_DIR / "history_feature_audit.csv",
            AUDIT_DIR / "feature_allowlist.csv",
            AUDIT_DIR / "result_consistency_report.json",
        ],
        Path(context["snapshot_dir"]) / "audit_and_config",
    )
    snapshot_files(
        [
            RESULT_DIR / "evaluation_results.json",
            RESULT_DIR / "real_test_predictions.csv",
            RESULT_DIR / "calibration_dca_metrics.json",
            RESULT_DIR / "prediction_metadata.json",
        ],
        Path(context["artifact_dir"]) / "results",
    )
    snapshot_files(
        [
            TABLE_DIR / "table1_baseline.csv",
            TABLE_DIR / "table1_baseline.tex",
            TABLE_DIR / "table2_performance.csv",
            TABLE_DIR / "table2_performance.tex",
            TABLE_DIR / "table3_pvalues.csv",
            TABLE_DIR / "table3_pvalues.tex",
        ],
        Path(context["artifact_dir"]) / "tables",
    )
    snapshot_files(
        [
            FIG_ROOT / "Submission_Main_Figures" / "Fig1_Cohort_Split_Audit.pdf",
            FIG_ROOT / "Submission_Main_Figures" / "Fig2_Cross_Center_Shift.pdf",
            FIG_ROOT / "Submission_Main_Figures" / "Fig3_Performance_Comparison.pdf",
            FIG_ROOT / "Submission_Main_Figures" / "Fig4_Calibration_DCA.pdf",
            FIG_ROOT / "Submission_Supplementary_Figures" / "FigS1_Target_Subgroup_Burden.pdf",
        ],
        Path(context["artifact_dir"]) / "figures",
    )
    snapshot_files(
        [
            MANUSCRIPT_DIR / "00_Figure_Order_and_Storyline.md",
            MANUSCRIPT_DIR / "01_Figure_Legends.md",
            MANUSCRIPT_DIR / "02_Results.md",
            MANUSCRIPT_DIR / "05_Methods.md",
            MANUSCRIPT_DIR / "MANUSCRIPT_INDEX.md",
        ],
        Path(context["artifact_dir"]) / "manuscript",
    )


MANUSCRIPT_DIR = ROOT / "manuscript"


def main() -> None:
    config = load_config()
    tracked_inputs = {
        "config": str(ROOT / "conf" / "config.yaml"),
        "source_data": str(ROOT / config["paths"]["data_dir"] / config["data"]["source_file"]),
        "target_data": str(ROOT / config["paths"]["data_dir"] / config["data"]["target_file"]),
        "feature_allowlist": str(ROOT / "experiments" / "audit" / "feature_allowlist.csv"),
    }
    context = prepare_locked_run_context(config=config, tracked_inputs=tracked_inputs)
    logs_dir = Path(context["logs_dir"])
    record_environment(log_dir=logs_dir, filename="pip_freeze.txt")

    steps = [
        ("audit_current_run", [sys.executable, "src/evaluate/audit_current_run.py"]),
        ("precheck_consistency", [sys.executable, "src/evaluate/result_consistency_report.py"]),
        ("precheck_sync_manuscript", [sys.executable, "src/manuscript/sync_submission_state.py"]),
        ("generate_table1", [sys.executable, "src/generate_table1.py"]),
        ("train_and_evaluate", [sys.executable, "src/main.py"]),
        ("export_real_predictions", [sys.executable, "src/evaluate/export_real_predictions.py"]),
        ("postcheck_consistency", [sys.executable, "src/evaluate/result_consistency_report.py"]),
        ("postcheck_sync_manuscript", [sys.executable, "src/manuscript/sync_submission_state.py"]),
        ("build_submission_figures", [sys.executable, "src/visualization/build_submission_figures.py"]),
    ]

    step_records = []
    for name, command in steps:
        run_step(name=name, command=command, log_dir=logs_dir)
        step_records.append({"name": name, "command": command, "log": str(logs_dir / f"{name}.log")})

    artifact_status = [{"path": str(path), "exists": path.exists()} for path in required_artifacts()]
    all_present = all(item["exists"] for item in artifact_status)
    snapshot_run_outputs(context)

    evidence_manifest = {
        "run_id": context["run_id"],
        "run_dir": context["run_dir"],
        "steps": step_records,
        "required_artifacts": artifact_status,
        "submission_ready_artifacts_present": all_present,
    }
    write_json(Path(context["run_dir"]) / "publication_evidence_manifest.json", evidence_manifest)
    print(json.dumps(evidence_manifest, ensure_ascii=False, indent=2))

    if not all_present:
        missing = [item["path"] for item in artifact_status if not item["exists"]]
        raise RuntimeError(f"Locked submission run incomplete; missing artifacts: {missing}")


if __name__ == "__main__":
    main()
