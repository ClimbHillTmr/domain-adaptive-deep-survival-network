"""Build or verify the exact file manifest for server confirmatory execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reproducibility import CONFIRMATORY_ANALYSIS_FILES, sha256_file


DEFAULT_OUTPUT = ROOT / "experiments" / "confirmatory_configs" / "server_package_manifest.json"
STATIC_FILES = (
    "conf/confirmatory_ablation.yaml",
    "data/confirmatory_v2/confirmatory_data_manifest.json",
    "data/confirmatory_v2/source_confirmatory_v2.csv",
    "data/confirmatory_v2/target_confirmatory_v2.csv",
    "experiments/audit/feature_allowlist.csv",
    "experiments/final_results/main_mechanism_aware/split_manifest.json",
    "requirements.txt",
    "pyproject.toml",
    "scripts/run_confirmatory_ablation.py",
    "scripts/build_server_confirmatory_package_manifest.py",
    "scripts/verify_server_confirmatory_results.py",
    "scripts/build_confirmatory_tables.py",
    "scripts/register_server_confirmatory_runs.py",
    "tests/test_binary_pipeline.py",
    "tests/test_data_pipeline.py",
    *CONFIRMATORY_ANALYSIS_FILES,
)


def _payload() -> dict:
    config_paths = sorted((ROOT / "experiments" / "confirmatory_configs").glob("*.yaml"))
    if len(config_paths) != 25:
        raise ValueError(f"Expected 25 locked configs, found {len(config_paths)}")
    files = [ROOT / relative_path for relative_path in STATIC_FILES] + config_paths
    missing = [str(path.relative_to(ROOT)) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing server package files: {missing}")
    records = [
        {
            "path": str(path.relative_to(ROOT)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(str(path)),
        }
        for path in sorted(set(files))
    ]
    evidence_locks = {
        json.dumps(
            yaml.safe_load(path.read_text(encoding="utf-8"))["confirmatory"]["evidence_lock"],
            ensure_ascii=False,
            sort_keys=True,
        )
        for path in config_paths
    }
    if len(evidence_locks) != 1:
        raise ValueError("The 25 configs do not share one evidence lock")
    package_sha256 = hashlib.sha256(
        json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "contract": "dual_binary_hemodynamic_confirmatory_v2",
        "protocol": "architecture_matched_ablation_v1",
        "purpose": "server_confirmatory_training_package",
        "training_performed": False,
        "locked_config_count": len(config_paths),
        "file_count": len(records),
        "package_sha256": package_sha256,
        "files": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify", action="store_true", help="Verify current bytes against the existing manifest.")
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    current = _payload()
    if args.verify:
        if not output.is_file():
            raise SystemExit(f"Server package verification failed: missing {output.relative_to(ROOT)}")
        registered = json.loads(output.read_text(encoding="utf-8"))
        if registered != current:
            raise SystemExit("Server package verification failed: current files do not match the locked manifest.")
        print(json.dumps({"status": "passed", "package_sha256": current["package_sha256"]}, indent=2))
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": "written", "output": str(output.relative_to(ROOT)),
        "file_count": current["file_count"], "package_sha256": current["package_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
