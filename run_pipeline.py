"""One command: build both cohorts, audit them, train both endpoints, save results."""

from __future__ import annotations

import argparse
import json
import os
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
        ("train", [sys.executable, "src/main_binary.py", "--config", str(run_config), "--train"]),
    ]
    completed = []
    try:
        for name, command in steps:
            run_step(name, command, run_dir)
            completed.append(name)
    except subprocess.CalledProcessError as error:
        print(f"Pipeline stopped at {name}. See {run_dir / f'{name}.log'}", file=sys.stderr)
        raise SystemExit(error.returncode) from error

    result_runs = sorted(result_dir.glob("binary_*"))
    summary = {
        "pipeline_run_id": run_id,
        "run_dir": str(run_dir),
        "completed_steps": completed,
        "model_result_dir": str(result_runs[-1]) if result_runs else None,
    }
    (run_dir / "pipeline_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
