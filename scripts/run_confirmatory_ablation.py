"""Run the prespecified five-model, five-seed confirmatory ablation.

The command refuses to fit models until the evidence registry confirms that the
exact frozen datasets are present and patient/temporal leakage gates pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reproducibility import confirmatory_analysis_fingerprint, sha256_file

BASE_CONFIG = ROOT / "conf" / "confirmatory_ablation.yaml"
CONFIRMATORY_DATA_STATUS = ROOT / "data" / "confirmatory_v2" / "confirmatory_data_manifest.json"
GENERATED_CONFIGS = ROOT / "experiments" / "confirmatory_configs"
REGISTERED_SPLIT = ROOT / "experiments" / "final_results" / "main_mechanism_aware" / "split_manifest.json"
EXPECTED_SPLIT_SHA256 = "6552a9085aed3f4b82ce3964768503d74ae28c038864b8d71c38af102513e568"

MODEL_SPECS = {
    "A_dual_no_alignment": {
        "architecture": "dual_branch", "alignment_strategy": "no_alignment", "use_coral": False,
    },
    "B_dual_global_coral": {
        "architecture": "dual_branch", "alignment_strategy": "global_coral", "use_coral": True,
    },
    "C_dual_random_feature_coral": {
        "architecture": "dual_branch", "alignment_strategy": "random_feature_coral", "use_coral": True,
    },
    "D_dual_outcome_specific_coral": {
        "architecture": "dual_branch", "alignment_strategy": "outcome_specific_coral", "use_coral": True,
    },
    "E_single_encoder_reference": {
        "architecture": "single_encoder", "alignment_strategy": "no_alignment", "use_coral": False,
    },
}
EXECUTION_SEEDS = (7, 13, 42, 99, 2024)


def _gate() -> dict:
    if not CONFIRMATORY_DATA_STATUS.exists():
        raise SystemExit("Confirmatory v2 manifest is missing. Run scripts/rebuild_confirmatory_data.py first.")
    status = json.loads(CONFIRMATORY_DATA_STATUS.read_text(encoding="utf-8"))
    if status.get("contract") != "dual_binary_hemodynamic_confirmatory_v2":
        raise SystemExit("Confirmatory training blocked: unexpected dataset contract.")
    if not status.get("training_gate_passed"):
        raise SystemExit("Confirmatory training blocked: v2 prior-history or patient-split audit failed.")
    expected = {
        ROOT / "data" / "confirmatory_v2" / "source_confirmatory_v2.csv": status.get("source_analysis_sha256"),
        ROOT / "data" / "confirmatory_v2" / "target_confirmatory_v2.csv": status.get("target_analysis_sha256"),
    }
    for path, expected_hash in expected.items():
        if not path.is_file() or sha256_file(str(path)) != expected_hash:
            raise SystemExit(f"Confirmatory training blocked: locked file hash mismatch for {path.relative_to(ROOT)}.")
    if not REGISTERED_SPLIT.is_file() or sha256_file(str(REGISTERED_SPLIT)) != EXPECTED_SPLIT_SHA256:
        raise SystemExit("Confirmatory training blocked: registered patient split hash mismatch.")
    return status


def _completed_config_hashes() -> set[str]:
    hashes: set[str] = set()
    result_root = ROOT / "experiments" / "confirmatory_results"
    for evaluation_path in result_root.glob("*/evaluation.json"):
        try:
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        provenance = evaluation.get("provenance", {})
        if (
            evaluation.get("contract") == "dual_binary_hemodynamic_confirmatory_v2"
            and provenance.get("execution_context") == "server_confirmatory"
            and bool(provenance.get("hostname"))
            and str(provenance.get("device", "")).startswith("cuda")
        ):
            hashes.add(str(evaluation.get("provenance", {}).get("config_sha256", "")))
    return hashes


def _expected_config_payloads(base: dict, evidence_lock: dict) -> list[tuple[Path, dict]]:
    payloads = []
    for model_id, spec in MODEL_SPECS.items():
        for seed in [int(value) for value in base["training"]["initialization_seeds"]]:
            config = yaml.safe_load(yaml.safe_dump(base, allow_unicode=True, sort_keys=False))
            config["confirmatory"]["model_id"] = model_id
            config["confirmatory"]["evidence_lock"] = evidence_lock
            config["project"]["name"] = f"confirmatory-{model_id}"
            config["model"].update(spec)
            config["model"]["use_stratified_alignment"] = spec["architecture"] == "dual_branch"
            config["model"]["use_outcome_specific_alignment"] = (
                spec["alignment_strategy"] == "outcome_specific_coral"
            )
            config["training"]["initialization_seed"] = seed
            payloads.append((GENERATED_CONFIGS / f"{model_id}__seed_{seed}.yaml", config))
    return payloads


def _execution_plan(configs: list[Path], completed: set[str]) -> list[dict]:
    rows = []
    for execution_index, config in enumerate(configs, start=1):
        payload = yaml.safe_load(config.read_text(encoding="utf-8"))
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        config_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        rows.append({
            "model_id": payload["confirmatory"]["model_id"],
            "seed": int(payload["training"]["initialization_seed"]),
            "execution_index": execution_index,
            "within_seed_position": (execution_index - 1) % len(MODEL_SPECS) + 1,
            "config": str(config.relative_to(ROOT)),
            "config_sha256": config_hash,
            "status": "completed_server_confirmatory" if config_hash in completed else "pending",
        })
    return rows


def _balanced_execution_order(configs: list[Path]) -> list[Path]:
    """Balance each model across the five within-seed execution positions."""
    lookup = {}
    for config in configs:
        payload = yaml.safe_load(config.read_text(encoding="utf-8"))
        key = (
            payload["confirmatory"]["model_id"],
            int(payload["training"]["initialization_seed"]),
        )
        if key in lookup:
            raise ValueError(f"Duplicate confirmatory configuration: {key}")
        lookup[key] = config
    model_ids = list(MODEL_SPECS)
    ordered = []
    for block_index, seed in enumerate(EXECUTION_SEEDS):
        block_order = model_ids[block_index:] + model_ids[:block_index]
        for model_id in block_order:
            key = (model_id, seed)
            if key not in lookup:
                raise ValueError(f"Missing confirmatory configuration: {key}")
            ordered.append(lookup[key])
    if len(ordered) != len(configs) or set(ordered) != set(configs):
        raise ValueError("Confirmatory execution grid contains unexpected configurations")
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--train", action="store_true", help="Explicit fitting authorization.")
    action.add_argument("--prepare-configs", action="store_true", help="Write the 25 immutable run configs only.")
    action.add_argument("--plan", action="store_true", help="Validate and list the locked execution grid without training.")
    parser.add_argument(
        "--execution-context",
        choices=("server_confirmatory",),
        help="Required with --train; records that the locked run was intentionally launched on the server.",
    )
    args = parser.parse_args()
    base = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    status = _gate()
    allowlist = ROOT / base["data"]["feature_allowlist"]
    if not allowlist.is_file():
        raise SystemExit("Confirmatory config preparation blocked: feature allowlist is missing.")
    evidence_lock = {
        "source_sha256": status["source_analysis_sha256"],
        "target_sha256": status["target_analysis_sha256"],
        "feature_allowlist_sha256": sha256_file(str(allowlist)),
        "split_manifest_sha256": EXPECTED_SPLIT_SHA256,
        "analysis_code_sha256": confirmatory_analysis_fingerprint(ROOT),
    }
    payloads = _expected_config_payloads(base, evidence_lock)
    GENERATED_CONFIGS.mkdir(parents=True, exist_ok=True)
    if args.prepare_configs:
        for path, config in payloads:
            path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        print(f"Prepared {len(payloads)} confirmatory configs in {GENERATED_CONFIGS}")
    if args.prepare_configs:
        return
    if not args.train and not args.plan:
        raise SystemExit("Training was not started. Use --plan to inspect or --train after evidence gates pass.")
    if args.train and args.execution_context != "server_confirmatory":
        raise SystemExit(
            "Training was not started. Server confirmatory fitting requires "
            "--execution-context server_confirmatory."
        )
    configs = []
    for path, expected_config in payloads:
        if not path.is_file():
            raise SystemExit(f"Training was not started. Locked config is missing: {path.name}")
        observed_config = yaml.safe_load(path.read_text(encoding="utf-8"))
        if observed_config != expected_config:
            raise SystemExit(f"Training was not started. Locked config differs from protocol: {path.name}")
        configs.append(path)
    if len(list(GENERATED_CONFIGS.glob("*.yaml"))) != 25:
        raise SystemExit("Training was not started. Confirmatory config directory must contain exactly 25 YAML files.")
    configs = _balanced_execution_order(configs)
    completed = _completed_config_hashes()
    plan = _execution_plan(configs, completed)
    if args.plan:
        print(json.dumps({
            "status": "plan_only_no_training",
            "configs_verified_without_rewrite": len(configs),
            "host": socket.gethostname(),
            "total": len(plan),
            "completed": sum(row["status"] == "completed_server_confirmatory" for row in plan),
            "pending": sum(row["status"] == "pending" for row in plan),
            "runs": plan,
        }, ensure_ascii=False, indent=2))
        return
    print(f"Verified {len(configs)} locked confirmatory configs without rewriting them")
    print(f"Server-confirmatory launch accepted on host: {socket.gethostname()}")
    for config in configs:
        config_payload = yaml.safe_load(config.read_text(encoding="utf-8"))
        # Same canonical fingerprint implementation as src.reproducibility.config_fingerprint.
        canonical = json.dumps(config_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        config_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if config_hash in completed:
            print(f"Skipping registered completed config: {config.name}")
            continue
        subprocess.run(
            [
                sys.executable,
                "-m",
                "src.main_binary",
                "--config",
                str(config),
                "--train",
                "--execution-context",
                "server_confirmatory",
            ],
            cwd=ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
