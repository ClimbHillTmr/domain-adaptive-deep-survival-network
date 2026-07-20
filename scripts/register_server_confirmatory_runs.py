"""Append only fully accepted confirmatory-v2 runs to the central run registry."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.reproducibility import sha256_file
from verify_server_confirmatory_results import audit as audit_server_results


REGISTRY = ROOT / "experiments" / "evidence_registry" / "run_registry.csv"
RESULT_DIR = ROOT / "experiments" / "confirmatory_results"


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-complete", action="store_true", default=True)
    args = parser.parse_args()
    acceptance = audit_server_results()
    if args.require_complete and acceptance["status"] != "complete":
        raise SystemExit(
            f"Confirmatory registration refused: accepted={acceptance['accepted_server_runs']}/25."
        )
    registry = pd.read_csv(REGISTRY, keep_default_na=False)
    historical = registry.loc[registry["contract"].eq("dual_binary_hemodynamic_v1")].copy()
    if len(historical) != 8:
        raise SystemExit(f"Confirmatory registration refused: expected 8 historical v1 rows, found {len(historical)}.")
    accepted_ids = {row["run_id"] for row in acceptance["accepted"]}
    rows = []
    for run_id in sorted(accepted_ids):
        run_dir = RESULT_DIR / run_id
        evaluation_path = run_dir / "evaluation.json"
        config_path = run_dir / "config.yaml"
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        source_hash = evaluation["provenance"]["source_sha256"]
        target_hash = evaluation["provenance"]["target_sha256"]
        rows.append({
            "run_id": run_id,
            "final_result_aliases": "",
            "contract": "dual_binary_hemodynamic_confirmatory_v2",
            "model_architecture": config["model"]["architecture"],
            "alignment_strategy": config["model"]["alignment_strategy"],
            "seed": evaluation["initialization_seed"],
            "split_seed": evaluation["split_seed"],
            "configuration_hash": evaluation["provenance"]["config_sha256"],
            "source_dataset_hash": source_hash,
            "target_dataset_hash": target_hash,
            "combined_dataset_hash": hashlib.sha256(f"{source_hash}:{target_hash}".encode()).hexdigest(),
            "checkpoint_path": _relative(run_dir / "mlp_models.pt"),
            "checkpoint_sha256": sha256_file(str(run_dir / "mlp_models.pt")),
            "prediction_file": _relative(run_dir / "test_predictions.csv"),
            "prediction_sha256": sha256_file(str(run_dir / "test_predictions.csv")),
            "evaluation_file": _relative(evaluation_path),
            "evaluation_sha256": sha256_file(str(evaluation_path)),
            "split_manifest_sha256": sha256_file(str(run_dir / "split_manifest.json")),
            "artifact_complete": "true",
            "reproducibility_status": "exact_v2_inputs_config_code_and_server_provenance_registered",
            "confirmatory_eligibility": "accepted_architecture_matched_server_run",
        })
    if len(rows) != 25:
        raise SystemExit(f"Confirmatory registration refused: expected 25 accepted rows, found {len(rows)}.")
    confirmatory = pd.DataFrame(rows, columns=registry.columns)
    merged = pd.concat([historical, confirmatory], ignore_index=True)
    if merged["run_id"].duplicated().any():
        raise SystemExit("Confirmatory registration refused: duplicate run ID in merged registry.")
    merged.to_csv(REGISTRY, index=False)
    print(json.dumps({
        "status": "registered", "historical_v1_runs": len(historical),
        "confirmatory_v2_runs": len(confirmatory), "total_runs": len(merged),
    }, indent=2))


if __name__ == "__main__":
    main()
