"""Generate and inspect the locked V3 server run grid without training.

This script materializes the architecture-matched model matrix from
`v3/experiments/training_protocol.yaml`. It never fits a model. Training remains
blocked until phase0 authorization is flipped and the scientific audit reports
`training_ready=true`.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reproducibility import config_fingerprint, sha256_file

TRAINING_PROTOCOL = ROOT / "v3/experiments/training_protocol.yaml"
PRIMARY_PARTITION = ROOT / "v3/experiments/primary_feature_partition.json"
RANDOM_PARTITIONS = ROOT / "v3/experiments/random_partitions.csv"
PREPROCESSING = ROOT / "data/v3/preprocessing.json"
PHASE_STATUS = ROOT / "v3/registry/phase0_status.json"
CONFIG_DIR = ROOT / "v3/experiments/server_configs"
GRID_CSV = ROOT / "v3/experiments/run_grid.csv"
GRID_JSON = ROOT / "v3/experiments/run_grid_manifest.json"
PLAN_JSON = ROOT / "v3/experiments/run_grid_plan.json"

ROLE_TO_BUNDLE = {
    "source_train": "source_train",
    "source_validation": "source_val",
    "target_update": "target_train",
    "target_calibration": "target_val",
    "target_test": "target_test",
}


def _sha256(path: Path) -> str:
    return sha256_file(str(path))


def _load_protocol() -> dict[str, Any]:
    payload = yaml.safe_load(TRAINING_PROTOCOL.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit("training_protocol.yaml must be a mapping")
    return payload


def _verify_protocol_hashes(protocol: dict[str, Any]) -> dict[str, str]:
    data = protocol["data"]
    checks = {
        "source_sha256": ROOT / data["source"],
        "target_sha256": ROOT / data["target"],
        "feature_allowlist_sha256": ROOT / data["feature_allowlist"],
        "split_manifest_sha256": ROOT / "v3/data_contract/split_manifest.csv",
        "preprocessing_sha256": ROOT / "data/v3/preprocessing.json",
        "primary_partition_sha256": PRIMARY_PARTITION,
        "random_partitions_sha256": RANDOM_PARTITIONS,
    }
    observed: dict[str, str] = {}
    for key, path in checks.items():
        if not path.is_file():
            raise SystemExit(f"Missing locked artifact for {key}: {path.relative_to(ROOT)}")
        digest = _sha256(path)
        expected = data[key]
        if digest != expected:
            raise SystemExit(f"Hash mismatch for {key}: expected {expected}, observed {digest}")
        observed[key] = digest
    observed["training_protocol_sha256"] = _sha256(TRAINING_PROTOCOL)
    return observed


def _feature_indices() -> tuple[list[str], list[int], list[int]]:
    prep = json.loads(PREPROCESSING.read_text(encoding="utf-8"))
    features = [str(name) for name in prep["feature_names"]]
    groups = prep["primary_groups"]
    physio = [index for index, name in enumerate(features) if groups[name] == "physiology_history"]
    treat = [index for index, name in enumerate(features) if groups[name] == "treatment_context"]
    primary = json.loads(PRIMARY_PARTITION.read_text(encoding="utf-8"))
    if [features[i] for i in physio] != primary["physiology_history"]:
        raise SystemExit("Primary physiology partition does not match frozen preprocessing order.")
    if [features[i] for i in treat] != primary["treatment_context"]:
        raise SystemExit("Primary treatment partition does not match frozen preprocessing order.")
    if len(physio) != 18 or len(treat) != 4:
        raise SystemExit(f"Unexpected primary partition sizes: physio={len(physio)} treat={len(treat)}")
    return features, physio, treat


def _random_rows() -> list[dict[str, str]]:
    with RANDOM_PARTITIONS.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _indices_for_names(features: list[str], names: list[str]) -> list[int]:
    lookup = {name: index for index, name in enumerate(features)}
    missing = [name for name in names if name not in lookup]
    if missing:
        raise SystemExit(f"Random partition references unknown features: {missing}")
    return [lookup[name] for name in names]


def _base_config(protocol: dict[str, Any], evidence_lock: dict[str, str]) -> dict[str, Any]:
    data = protocol["data"]
    optimization = protocol["optimization"]
    architecture = protocol["architecture"]
    evaluation = protocol["evaluation"]
    return {
        "project": {
            "name": "v3-mechanism-aware-transportability",
            "endpoint_rules": {
                "idh": "baseline SBP - intradialytic SBP >= 30 or intradialytic SBP <= 90",
                "ih": "intradialytic MAP - baseline MAP > 10",
            },
        },
        "v3": {
            "protocol_version": protocol["protocol_version"],
            "contract_version": protocol["contract_version"],
            "data_contract": "v3_hbd_data_protocol_1",
            "execution_context": "server_v3",
            "training_authorized": bool(protocol["execution"]["training_authorized"]),
            "model_family_id": "PLACEHOLDER",
            "run_id": "PLACEHOLDER",
            "evidence_lock": evidence_lock,
        },
        "data": {
            "source_file": data["source"],
            "target_file": data["target"],
            "feature_allowlist": data["feature_allowlist"],
            "split_manifest": "v3/data_contract/split_manifest.csv",
            "preprocessing_file": "data/v3/preprocessing.json",
            "patient_col": "患者id",
            "split_seed": 20260715,
            "source_validation_fraction": 0.10,
            "target_update_fraction": 0.60,
            "target_validation_fraction": 0.15,
        },
        "model": {
            "logistic_c_values": [0.01, 0.1, 1.0, 10.0],
            "dropout": float(architecture["dropout"]),
            "use_cdan": False,
            "use_mmd": False,
            "mmd_weight": 1.0,
            "pooled_update": True,
            "architecture_partition": "clinical_primary",
        },
        "training": {
            "initialization_seed": 0,
            "initialization_seeds": list(protocol["seeds"]["initialization"]),
            "batch_size": int(optimization["batch_size"]),
            "pretrain_learning_rate": float(optimization["source_pretrain_learning_rate"]),
            "finetune_learning_rate": float(optimization["target_update_learning_rate"]),
            "pretrain_epochs": int(optimization["source_pretrain_max_epochs"]),
            "finetune_epochs": int(optimization["target_update_max_epochs"]),
            "patience": int(optimization["early_stopping_patience"]),
        },
        "evaluation": {
            "patient_bootstrap_replicates": int(evaluation["bootstrap_replicates"]),
            "bootstrap_seed": int(protocol["seeds"]["bootstrap"]),
            "endpoints": list(evaluation["endpoints"]),
            "primary_metric": evaluation["primary_metric"],
            "secondary_metrics": list(evaluation["secondary_metrics"]),
            "multiplicity": evaluation["multiplicity"],
        },
        "paths": {
            "output_dir": "experiments/v3_results",
        },
    }


def _model_specs(features: list[str], physio: list[int], treat: list[int]) -> list[dict[str, Any]]:
    coral_weight = 0.01
    dual_hidden = [64, 32]
    single_hidden = [64, 65]
    specs: list[dict[str, Any]] = [
        {
            "model_family_id": "source_only",
            "experiment_id": "V3-E3-CONTROLS",
            "architecture": "single_encoder",
            "alignment_strategy": "none",
            "training_mode": "source_only",
            "use_coral": False,
            "coral_weight": coral_weight,
            "mlp_hidden_dims": single_hidden,
            "pooled_update": False,
            "physio_indices": None,
            "treat_indices": None,
            "alignment_scope": "global",
            "architecture_partition": "none",
            "random_partition_id": "",
            "random_family": "",
        },
        {
            "model_family_id": "A",
            "experiment_id": "V3-E3-CONTROLS",
            "architecture": "single_encoder",
            "alignment_strategy": "no_alignment",
            "training_mode": "supervised_update",
            "use_coral": False,
            "coral_weight": coral_weight,
            "mlp_hidden_dims": single_hidden,
            "pooled_update": True,
            "physio_indices": None,
            "treat_indices": None,
            "alignment_scope": "global",
            "architecture_partition": "none",
            "random_partition_id": "",
            "random_family": "",
        },
        {
            "model_family_id": "B",
            "experiment_id": "V3-E3-CONTROLS",
            "architecture": "single_encoder",
            "alignment_strategy": "global_coral",
            "training_mode": "supervised_update",
            "use_coral": True,
            "coral_weight": coral_weight,
            "mlp_hidden_dims": single_hidden,
            "pooled_update": True,
            "physio_indices": None,
            "treat_indices": None,
            "alignment_scope": "global",
            "architecture_partition": "none",
            "random_partition_id": "",
            "random_family": "",
        },
        {
            "model_family_id": "C",
            "experiment_id": "V3-E3-CONTROLS",
            "architecture": "dual_branch",
            "alignment_strategy": "no_alignment",
            "training_mode": "supervised_update",
            "use_coral": False,
            "coral_weight": coral_weight,
            "mlp_hidden_dims": dual_hidden,
            "pooled_update": True,
            "physio_indices": physio,
            "treat_indices": treat,
            "alignment_scope": "global",
            "architecture_partition": "clinical_primary",
            "random_partition_id": "",
            "random_family": "",
        },
        {
            "model_family_id": "D1",
            "experiment_id": "V3-E3-SELECTIVE",
            "architecture": "dual_branch",
            "alignment_strategy": "physiology_coral",
            "training_mode": "supervised_update",
            "use_coral": True,
            "coral_weight": coral_weight,
            "mlp_hidden_dims": dual_hidden,
            "pooled_update": True,
            "physio_indices": physio,
            "treat_indices": treat,
            "alignment_scope": "branch_a",
            "architecture_partition": "clinical_primary",
            "random_partition_id": "",
            "random_family": "",
        },
        {
            "model_family_id": "D2",
            "experiment_id": "V3-E3-SELECTIVE",
            "architecture": "dual_branch",
            "alignment_strategy": "treatment_coral",
            "training_mode": "supervised_update",
            "use_coral": True,
            "coral_weight": coral_weight,
            "mlp_hidden_dims": dual_hidden,
            "pooled_update": True,
            "physio_indices": physio,
            "treat_indices": treat,
            "alignment_scope": "branch_b",
            "architecture_partition": "clinical_primary",
            "random_partition_id": "",
            "random_family": "",
        },
        {
            "model_family_id": "F",
            "experiment_id": "V3-E3-CONTROLS",
            "architecture": "single_encoder",
            "alignment_strategy": "none",
            "training_mode": "target_only",
            "use_coral": False,
            "coral_weight": coral_weight,
            "mlp_hidden_dims": single_hidden,
            "pooled_update": False,
            "physio_indices": None,
            "treat_indices": None,
            "alignment_scope": "global",
            "architecture_partition": "none",
            "random_partition_id": "",
            "random_family": "",
        },
        {
            "model_family_id": "G",
            "experiment_id": "V3-E3-CONTROLS",
            "architecture": "dual_branch",
            "alignment_strategy": "global_coral",
            "training_mode": "supervised_update",
            "use_coral": True,
            "coral_weight": coral_weight,
            "mlp_hidden_dims": dual_hidden,
            "pooled_update": True,
            "physio_indices": physio,
            "treat_indices": treat,
            "alignment_scope": "global",
            "architecture_partition": "clinical_primary",
            "random_partition_id": "",
            "random_family": "",
        },
    ]

    for row in _random_rows():
        aligned_names = ast.literal_eval(row["aligned_features"])
        preserved_names = ast.literal_eval(row["preserved_features"])
        aligned = _indices_for_names(features, aligned_names)
        preserved = _indices_for_names(features, preserved_names)
        if sorted(aligned + preserved) != list(range(len(features))):
            raise SystemExit(f"Random partition is not a full cover: {row['partition_id']}")
        specs.append(
            {
                "model_family_id": f"E__{row['partition_id']}",
                "experiment_id": "V3-E3-RANDOM",
                "architecture": "dual_branch",
                "alignment_strategy": "random_feature_coral",
                "training_mode": "supervised_update",
                "use_coral": True,
                "coral_weight": coral_weight,
                "mlp_hidden_dims": dual_hidden,
                "pooled_update": True,
                # Random controls use the random aligned set as branch_a so the
                # architecture itself encodes the null partition.
                "physio_indices": aligned,
                "treat_indices": preserved,
                "random_aligned_indices": aligned,
                "alignment_scope": "branch_a",
                "architecture_partition": "random_aligned",
                "random_partition_id": row["partition_id"],
                "random_family": row["family"],
            }
        )
    return specs


def _apply_spec(base: dict[str, Any], spec: dict[str, Any], seed: int) -> dict[str, Any]:
    config = yaml.safe_load(yaml.safe_dump(base, allow_unicode=True, sort_keys=False))
    run_id = f"V3-{spec['model_family_id']}__seed_{seed}"
    config["project"]["name"] = f"v3-{spec['model_family_id']}"
    config["v3"]["model_family_id"] = spec["model_family_id"]
    config["v3"]["experiment_id"] = spec["experiment_id"]
    config["v3"]["run_id"] = run_id
    config["v3"]["random_partition_id"] = spec.get("random_partition_id", "")
    config["v3"]["random_family"] = spec.get("random_family", "")
    model = config["model"]
    model.update(
        {
            "architecture": spec["architecture"],
            "alignment_strategy": spec["alignment_strategy"],
            "training_mode": spec["training_mode"],
            "use_coral": bool(spec["use_coral"]),
            "coral_weight": float(spec["coral_weight"]),
            "mlp_hidden_dims": list(spec["mlp_hidden_dims"]),
            "pooled_update": bool(spec["pooled_update"]),
            "physio_indices": spec["physio_indices"],
            "treat_indices": spec["treat_indices"],
            "dual_branch_physio_indices": spec["physio_indices"] if spec["architecture"] == "dual_branch" else None,
            "alignment_scope": spec["alignment_scope"],
            "architecture_partition": spec["architecture_partition"],
            "use_stratified_alignment": spec["architecture"] == "dual_branch",
            "use_outcome_specific_alignment": False,
        }
    )
    if "random_aligned_indices" in spec:
        model["random_aligned_indices"] = list(spec["random_aligned_indices"])
    config["training"]["initialization_seed"] = int(seed)
    return config


def _balanced_execution_order(rows: list[dict[str, Any]], seeds: list[int]) -> list[dict[str, Any]]:
    lookup = {(row["model_family_id"], int(row["seed"])): row for row in rows}
    model_ids = []
    for row in rows:
        if row["model_family_id"] not in model_ids:
            model_ids.append(row["model_family_id"])
    ordered: list[dict[str, Any]] = []
    for block_index, seed in enumerate(seeds):
        block_order = model_ids[block_index % len(model_ids) :] + model_ids[: block_index % len(model_ids)]
        for model_id in block_order:
            key = (model_id, int(seed))
            if key not in lookup:
                raise SystemExit(f"Missing run in balanced grid: {key}")
            ordered.append(lookup[key])
    if len(ordered) != len(rows):
        raise SystemExit("Balanced execution order length mismatch.")
    for index, row in enumerate(ordered, start=1):
        row["execution_index"] = index
        row["within_seed_position"] = ((index - 1) % len(model_ids)) + 1
    return ordered


def build_grid(write_configs: bool) -> dict[str, Any]:
    protocol = _load_protocol()
    evidence_lock = _verify_protocol_hashes(protocol)
    features, physio, treat = _feature_indices()
    specs = _model_specs(features, physio, treat)
    seeds = [int(seed) for seed in protocol["seeds"]["initialization"]]
    base = _base_config(protocol, evidence_lock)

    if write_configs:
        if CONFIG_DIR.exists():
            for path in CONFIG_DIR.glob("*.yaml"):
                path.unlink()
        else:
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for spec in specs:
        for seed in seeds:
            config = _apply_spec(base, spec, seed)
            run_id = config["v3"]["run_id"]
            config_path = CONFIG_DIR / f"{run_id}.yaml"
            config_hash = config_fingerprint(config)
            if write_configs:
                config_path.write_text(
                    yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
                    encoding="utf-8",
                )
                if _sha256(config_path) and config_hash != config_fingerprint(
                    yaml.safe_load(config_path.read_text(encoding="utf-8"))
                ):
                    raise SystemExit(f"Config rewrite fingerprint mismatch: {config_path.name}")
            rows.append(
                {
                    "run_id": run_id,
                    "experiment_id": spec["experiment_id"],
                    "model_family_id": spec["model_family_id"],
                    "architecture": spec["architecture"],
                    "alignment_strategy": spec["alignment_strategy"],
                    "training_mode": spec["training_mode"],
                    "seed": seed,
                    "random_partition_id": spec.get("random_partition_id", ""),
                    "random_family": spec.get("random_family", ""),
                    "config_path": str(config_path.relative_to(ROOT)),
                    "config_sha256": config_hash,
                    "evidence_node": spec["experiment_id"].replace("V3-", "E").split("-")[0]
                    if False
                    else {
                        "V3-E3-CONTROLS": "E3.1",
                        "V3-E3-SELECTIVE": "E3.2",
                        "V3-E3-RANDOM": "E3.3",
                    }[spec["experiment_id"]],
                    "output_dir": f"experiments/v3_results/{run_id}",
                    "status": "planned",
                }
            )

    ordered = _balanced_execution_order(rows, seeds)
    if write_configs:
        fieldnames = list(ordered[0].keys())
        with GRID_CSV.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(ordered)
        manifest = {
            "status": "run_grid_generated_training_not_authorized",
            "protocol_version": protocol["protocol_version"],
            "training_authorized": bool(protocol["execution"]["training_authorized"]),
            "n_models": len(specs),
            "n_seeds": len(seeds),
            "n_runs": len(ordered),
            "n_core_models": 8,
            "n_random_partitions": 76,
            "n_random_runs": 76 * len(seeds),
            "seeds": seeds,
            "config_dir": str(CONFIG_DIR.relative_to(ROOT)),
            "grid_csv": str(GRID_CSV.relative_to(ROOT)),
            "evidence_lock": evidence_lock,
            "feature_count": len(features),
            "physio_indices": physio,
            "treat_indices": treat,
            "host_generated_on": socket.gethostname(),
        }
        GRID_JSON.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        PLAN_JSON.write_text(
            json.dumps(
                {
                    "status": "plan_only_no_training",
                    "n_runs": len(ordered),
                    "runs": ordered,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return {
        "n_runs": len(ordered),
        "n_models": len(specs),
        "n_seeds": len(seeds),
        "training_authorized": bool(protocol["execution"]["training_authorized"]),
        "grid_csv": str(GRID_CSV.relative_to(ROOT)),
        "config_dir": str(CONFIG_DIR.relative_to(ROOT)),
        "evidence_lock": evidence_lock,
        "runs_preview": ordered[:5],
    }


def verify_grid() -> dict[str, Any]:
    if not GRID_CSV.is_file() or not GRID_JSON.is_file():
        raise SystemExit("Run grid has not been generated. Use --prepare-configs first.")
    with GRID_CSV.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    mismatches = []
    for row in rows:
        path = ROOT / row["config_path"]
        if not path.is_file():
            mismatches.append(f"missing:{row['config_path']}")
            continue
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        digest = config_fingerprint(payload)
        if digest != row["config_sha256"]:
            mismatches.append(f"hash_mismatch:{row['run_id']}")
        if payload["v3"]["run_id"] != row["run_id"]:
            mismatches.append(f"run_id_mismatch:{row['run_id']}")
        if bool(payload["v3"].get("training_authorized", False)):
            mismatches.append(f"training_authorized_true_in_config:{row['run_id']}")
    if mismatches:
        raise SystemExit("Run grid verification failed: " + "; ".join(mismatches[:10]))
    return {
        "status": "verified_no_training",
        "n_runs": len(rows),
        "config_dir": str(CONFIG_DIR.relative_to(ROOT)),
        "training_authorized_in_configs": False,
    }


def launch_training() -> None:
    phase = json.loads(PHASE_STATUS.read_text(encoding="utf-8"))
    protocol = _load_protocol()
    if not bool(phase.get("training_authorized")) or not bool(protocol["execution"]["training_authorized"]):
        raise SystemExit(
            "V3 training remains blocked. Flip training_authorized only after scientific audit "
            "reports training_ready=true and PI review of the locked grid."
        )
    raise SystemExit(
        "V3 training authorization is true, but automatic mass launch is intentionally not "
        "enabled in this script revision. Launch individual verified configs through the "
        "reviewed server entrypoint after a final readiness audit."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare-configs", action="store_true", help="Write the immutable V3 run grid and YAML configs.")
    action.add_argument("--plan", action="store_true", help="Verify and print the locked grid without writing.")
    action.add_argument("--verify", action="store_true", help="Re-hash every generated config against the grid.")
    action.add_argument("--train", action="store_true", help="Fail-closed training gate.")
    args = parser.parse_args()

    if args.prepare_configs:
        summary = build_grid(write_configs=True)
        print(json.dumps({"status": "prepared_no_training", **summary}, ensure_ascii=False, indent=2))
        return
    if args.plan:
        if not GRID_CSV.is_file():
            summary = build_grid(write_configs=False)
            print(json.dumps({"status": "plan_ephemeral_no_write", **summary}, ensure_ascii=False, indent=2))
            return
        report = verify_grid()
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    if args.verify:
        print(json.dumps(verify_grid(), ensure_ascii=False, indent=2))
        return
    if args.train:
        launch_training()


if __name__ == "__main__":
    main()
