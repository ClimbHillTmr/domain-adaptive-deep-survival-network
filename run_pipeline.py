"""One command for dual-endpoint binary, discrete-time, Cox, and CDAN analyses."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent


def run_step(name: str, command: list[str], run_dir: Path) -> None:
    print(f"[{name}] {' '.join(command)}", flush=True)
    log_path = run_dir / f"{name}.log"
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def aggregate_binary_runs(result_dir: Path, output: Path) -> None:
    evaluations = []
    test_patients = None
    for evaluation_path in sorted(result_dir.glob("binary_*/evaluation.json")):
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        manifest = json.loads((evaluation_path.parent / "split_manifest.json").read_text(encoding="utf-8"))
        patients = manifest["target_test"]["patient_ids"]
        if test_patients is None:
            test_patients = patients
        elif patients != test_patients:
            raise RuntimeError("Binary multiseed runs do not use the same target test patients.")
        evaluations.append({"path": str(evaluation_path), **evaluation})
    if not evaluations:
        raise RuntimeError("No binary evaluation files found for multiseed aggregation.")
    summary = {}
    for endpoint in ("idh", "ih"):
        summary[endpoint] = {}
        for model in ("source_logistic", "local_logistic", "source_mlp", "updated_mlp"):
            summary[endpoint][model] = {}
            for metric in ("roc_auc", "pr_auc", "brier"):
                values = [run["metrics"][endpoint][model][metric] for run in evaluations]
                summary[endpoint][model][metric] = {
                    "mean": float(sum(values) / len(values)),
                    "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
                    "values": values,
                }
        summary[endpoint]["paired_auc_deltas"] = {}
        for comparison in ("updated_mlp_vs_source_mlp", "updated_mlp_vs_local_logistic"):
            values = [run["paired_comparisons"][endpoint][comparison]["delta_a_minus_b"] for run in evaluations]
            summary[endpoint]["paired_auc_deltas"][comparison] = {
                "mean": float(sum(values) / len(values)),
                "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
                "values": values,
            }
    output.write_text(
        json.dumps(
            {
                "split_seed": evaluations[0]["split_seed"],
                "initialization_seeds": [run["initialization_seed"] for run in evaluations],
                "fixed_target_test_patients": len(test_patients or []),
                "summary": summary,
                "runs": evaluations,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shenyi_raw", type=Path)
    parser.add_argument("fuding_raw", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "conf" / "binary_config.yaml")
    parser.add_argument("--run-root", type=Path, default=ROOT / "experiments" / "pipeline_runs")
    args = parser.parse_args()
    for path in (args.shenyi_raw, args.fuding_raw, args.config):
        if not path.is_file():
            parser.error(f"File not found: {path}")

    run_id = datetime.now(timezone.utc).strftime("pipeline_%Y%m%d_%H%M%S")
    run_dir = args.run_root.resolve() / run_id
    processed_dir = run_dir / "processed"
    result_dir = run_dir / "results"
    processed_dir.mkdir(parents=True)

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config["data"]["source_file"] = str(processed_dir / "shenyi.csv")
    config["data"]["target_file"] = str(processed_dir / "fuding.csv")
    config["paths"]["output_dir"] = str(result_dir)
    run_config = run_dir / "config.yaml"
    run_config.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")

    steps = [
        (
            "build_shenyi",
            [sys.executable, "src/data_pipeline/HBD_data.py", str(args.shenyi_raw.resolve()), config["data"]["source_file"]],
        ),
        (
            "build_fuding",
            [sys.executable, "src/data_pipeline/HBD_data_fuding.py", str(args.fuding_raw.resolve()), config["data"]["target_file"]],
        ),
        (
            "audit",
            [
                sys.executable,
                "src/evaluate/audit_binary_data.py",
                "--config",
                str(run_config),
                "--output",
                str(run_dir / "data_preflight.json"),
                "--strict",
            ],
        ),
    ]
    for seed in config["training"]["initialization_seeds"]:
        steps.append(
            (
                f"binary_seed_{seed}",
                [
                    sys.executable,
                    "src/main_binary.py",
                    "--config",
                    str(run_config),
                    "--initialization-seed",
                    str(seed),
                    "--train",
                ],
            )
        )
    completed = []
    try:
        for name, command in steps:
            run_step(name, command, run_dir)
            completed.append(name)
        name = "aggregate_binary_multiseed"
        aggregate_binary_runs(result_dir, run_dir / "binary_multiseed_results.json")
        completed.append("aggregate_binary_multiseed")
        name = "time_measurement_audit"
        run_step(
            name,
            [
                sys.executable,
                "src/main_time_models.py",
                "--config",
                str(run_config),
                "--output",
                str(run_dir / "time_models"),
                "--audit-only",
            ],
            run_dir,
        )
        completed.append("time_measurement_audit")
        name = "time_models"
        run_step(
            name,
            [
                sys.executable,
                "src/main_time_models.py",
                "--config",
                str(run_config),
                "--output",
                str(run_dir / "time_models"),
                "--train",
            ],
            run_dir,
        )
        completed.append("discrete_cox_cdan_models")
    except subprocess.CalledProcessError as error:
        print(f"Pipeline stopped at {name}. See {run_dir / f'{name}.log'}", file=sys.stderr)
        raise SystemExit(error.returncode) from error

    result_runs = sorted(result_dir.glob("binary_*"))
    summary = {
        "pipeline_run_id": run_id,
        "run_dir": str(run_dir),
        "completed_steps": completed,
        "binary_result_dirs": [str(path) for path in result_runs],
        "binary_multiseed_results": str(run_dir / "binary_multiseed_results.json"),
        "time_model_results": str(run_dir / "time_models" / "time_model_results.json"),
    }
    (run_dir / "pipeline_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
