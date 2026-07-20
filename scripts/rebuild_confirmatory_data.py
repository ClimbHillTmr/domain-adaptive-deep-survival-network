"""Rebuild a versioned confirmatory dataset without overwriting legacy files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.binary_dataset import prepare_binary_data
from src.data_pipeline.data_process import write_cohort
from src.evaluate.audit_prior_history import write_audit


DEFAULT_RAW = Path.home() / ".cache" / "huggingface" / "hub" / "datasets--LongGoodbye--Shenyi-Fuding-original" / "snapshots" / "b125e7730d526b43c6cbd17dca3ff00b57b9cf1c"
DEFAULT_OUTPUT = ROOT / "data" / "confirmatory_v2"
EXPECTED_RAW = {
    "source": "607a7f115efa2db64c3f23190f82222978d10146570bdbf6e1f66118bd2320f9",
    "target": "6fcfd523cd60aedc245d4b51e072aeca777f65e9a56554fdebaf45d3ee5f3813",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-raw", type=Path, default=DEFAULT_RAW / "updated_dataset_shenyi.csv")
    parser.add_argument("--target-raw", type=Path, default=DEFAULT_RAW / "updated_dataset_fuding.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit-only", action="store_true", help="Reuse existing rebuilt CSVs.")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for label, path in (("source", args.source_raw), ("target", args.target_raw)):
        actual = _sha256(path)
        if actual != EXPECTED_RAW[label]:
            raise SystemExit(f"{label} raw SHA-256 mismatch: {actual}")
    source_path, target_path = output / "source_confirmatory_v2.csv", output / "target_confirmatory_v2.csv"
    if args.audit_only:
        source_build = json.loads(source_path.with_suffix(".audit.json").read_text(encoding="utf-8"))
        target_build = json.loads(target_path.with_suffix(".audit.json").read_text(encoding="utf-8"))
    else:
        source_build = write_cohort(args.source_raw, "shenyi", source_path)
        target_build = write_cohort(args.target_raw, "fuding", target_path)
    source_history = write_audit(source_path, output / "source_prior_history_audit.json")
    target_history = write_audit(target_path, output / "target_prior_history_audit.json")
    bundle = prepare_binary_data(
        source_path, target_path, allowlist_path=ROOT / "experiments" / "audit" / "feature_allowlist.csv",
        patient_col="患者id", split_seed=20260715, source_validation_fraction=0.10,
        target_update_fraction=0.60, target_validation_fraction=0.15,
    )
    historical_split = json.loads(
        (ROOT / "experiments" / "final_results" / "main_mechanism_aware" / "split_manifest.json").read_text(encoding="utf-8")
    )
    historical_preprocessing = json.loads(
        (ROOT / "experiments" / "final_results" / "main_mechanism_aware" / "preprocessing.json").read_text(encoding="utf-8")
    )
    preprocessing_max_error = max(
        abs(bundle.preprocessing[group][feature] - historical_preprocessing[group][feature])
        for group in ("medians", "means", "scales") for feature in bundle.feature_names
    )
    manifest = {
        "contract": "dual_binary_hemodynamic_confirmatory_v2",
        "construction": "src.data_pipeline.data_process.write_cohort",
        "source_raw_sha256": EXPECTED_RAW["source"], "target_raw_sha256": EXPECTED_RAW["target"],
        "source_analysis_sha256": _sha256(source_path), "target_analysis_sha256": _sha256(target_path),
        "source": source_build, "target": target_build,
        "source_prior_history_passed": source_history["passed"],
        "target_prior_history_passed": target_history["passed"],
        "patient_split_equal_to_historical_runs": bundle.split_manifest == historical_split,
        "preprocessing_max_absolute_difference_vs_historical": preprocessing_max_error,
        "semantic_equivalence_tolerance": 1e-12,
        "semantic_equivalence_passed": (
            bundle.split_manifest == historical_split and preprocessing_max_error <= 1e-12
        ),
        "training_gate_passed": (
            source_history["passed"] and target_history["passed"]
            and bundle.split_manifest == historical_split
        ),
        "historical_difference_interpretation": (
            "Expected v2 change from correcting two tied source sessions; confirmatory runs must not be "
            "mixed with historical v1 runs."
        ),
    }
    temporary_manifest = output / "confirmatory_data_manifest.json.tmp"
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary_manifest, output / "confirmatory_data_manifest.json")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
