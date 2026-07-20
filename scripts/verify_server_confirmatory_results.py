"""Read-only acceptance audit for returned server confirmatory runs."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluate.binary_metrics import expected_calibration_error
from src.reproducibility import confirmatory_analysis_fingerprint, config_fingerprint, sha256_file


CONFIG_DIR = ROOT / "experiments" / "confirmatory_configs"
RESULT_DIR = ROOT / "experiments" / "confirmatory_results"
DATA_MANIFEST = ROOT / "data" / "confirmatory_v2" / "confirmatory_data_manifest.json"
EXPECTED_SPLIT_SHA256 = "6552a9085aed3f4b82ce3964768503d74ae28c038864b8d71c38af102513e568"
REQUIRED_ARTIFACTS = {
    "config.yaml",
    "data_preflight.json",
    "environment.txt",
    "evaluation.json",
    "logistic_models.pkl",
    "mlp_models.pt",
    "preprocessing.json",
    "split_manifest.json",
    "test_predictions.csv",
}
EXPECTED_MODELS = {
    "A_dual_no_alignment": ("dual_branch", "no_alignment", False, False),
    "B_dual_global_coral": ("dual_branch", "global_coral", True, False),
    "C_dual_random_feature_coral": ("dual_branch", "random_feature_coral", True, False),
    "D_dual_outcome_specific_coral": ("dual_branch", "outcome_specific_coral", True, True),
    "E_single_encoder_reference": ("single_encoder", "no_alignment", False, False),
}
EXPECTED_SEEDS = {7, 13, 42, 99, 2024}


def _fairness_signature(config: dict) -> dict:
    """Remove only the prespecified model/seed axes before equality comparison."""
    signature = copy.deepcopy(config)
    signature["project"]["name"] = "CONFIRMATORY_MODEL"
    signature["confirmatory"]["model_id"] = "CONFIRMATORY_MODEL"
    signature["training"]["initialization_seed"] = "CONFIRMATORY_SEED"
    for key in (
        "architecture", "alignment_strategy", "use_coral",
        "use_stratified_alignment", "use_outcome_specific_alignment",
    ):
        signature["model"][key] = f"CONFIRMATORY_{key}"
    return signature


def _audit_locked_grid(expected: dict[str, tuple[Path, dict]]) -> list[str]:
    errors = []
    combinations = set()
    signatures = set()
    for _, config in expected.values():
        model_id = config.get("confirmatory", {}).get("model_id")
        seed = int(config.get("training", {}).get("initialization_seed", -1))
        combinations.add((model_id, seed))
        signatures.add(json.dumps(_fairness_signature(config), ensure_ascii=False, sort_keys=True))
        if model_id not in EXPECTED_MODELS:
            errors.append(f"unexpected_model_id:{model_id}")
            continue
        architecture, strategy, use_coral, outcome_specific = EXPECTED_MODELS[model_id]
        model = config.get("model", {})
        observed = (
            model.get("architecture"), model.get("alignment_strategy"),
            bool(model.get("use_coral")), bool(model.get("use_outcome_specific_alignment")),
        )
        if observed != (architecture, strategy, use_coral, outcome_specific):
            errors.append(f"model_strategy_mismatch:{model_id}:{seed}")
        if bool(model.get("use_stratified_alignment")) != (architecture == "dual_branch"):
            errors.append(f"model_architecture_branch_mismatch:{model_id}:{seed}")
        if seed not in EXPECTED_SEEDS:
            errors.append(f"unexpected_seed:{model_id}:{seed}")
    expected_combinations = {(model_id, seed) for model_id in EXPECTED_MODELS for seed in EXPECTED_SEEDS}
    if combinations != expected_combinations:
        errors.append("locked_model_seed_grid_mismatch")
    if len(signatures) != 1:
        errors.append(f"nonmatched_configuration_contracts:{len(signatures)}")
    return errors


def _expected_configs() -> dict[str, tuple[Path, dict]]:
    expected = {}
    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        config_hash = config_fingerprint(config)
        if config_hash in expected:
            raise ValueError(f"Duplicate generated config hash: {config_hash}")
        expected[config_hash] = (path, config)
    return expected


def audit() -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    accepted: list[dict] = []
    rejected: list[dict] = []
    expected = _expected_configs()
    if len(expected) != 25:
        errors.append(f"expected_25_generated_configs_found_{len(expected)}")
    errors.extend(_audit_locked_grid(expected))
    locked_code_hashes = {
        config.get("confirmatory", {}).get("evidence_lock", {}).get("analysis_code_sha256")
        for _, config in expected.values()
    }
    locked_code_hashes.discard(None)
    if len(locked_code_hashes) != 1:
        errors.append(f"expected_one_analysis_code_lock_found_{len(locked_code_hashes)}")
        expected_code_hash = ""
    else:
        expected_code_hash = next(iter(locked_code_hashes))
        if confirmatory_analysis_fingerprint(ROOT) != expected_code_hash:
            warnings.append("current_analysis_code_differs_from_locked_server_package")

    manifest = json.loads(DATA_MANIFEST.read_text(encoding="utf-8"))
    source = ROOT / "data" / "confirmatory_v2" / "source_confirmatory_v2.csv"
    target = ROOT / "data" / "confirmatory_v2" / "target_confirmatory_v2.csv"
    expected_source_hash = manifest["source_analysis_sha256"]
    expected_target_hash = manifest["target_analysis_sha256"]
    locked_allowlist_hashes = {
        config.get("confirmatory", {}).get("evidence_lock", {}).get("feature_allowlist_sha256")
        for _, config in expected.values()
    }
    locked_allowlist_hashes.discard(None)
    if len(locked_allowlist_hashes) != 1:
        errors.append(f"expected_one_allowlist_lock_found_{len(locked_allowlist_hashes)}")
        expected_allowlist_hash = ""
    else:
        expected_allowlist_hash = next(iter(locked_allowlist_hashes))
        current_allowlist = ROOT / "experiments" / "audit" / "feature_allowlist.csv"
        if not current_allowlist.is_file() or sha256_file(str(current_allowlist)) != expected_allowlist_hash:
            warnings.append("current_feature_allowlist_differs_from_locked_server_package")
    if sha256_file(str(source)) != expected_source_hash:
        errors.append("local_source_confirmatory_v2_hash_mismatch")
    if sha256_file(str(target)) != expected_target_hash:
        errors.append("local_target_confirmatory_v2_hash_mismatch")
    if not manifest.get("training_gate_passed"):
        errors.append("confirmatory_v2_training_gate_not_passed")

    seen_hashes: dict[str, str] = {}
    seen_run_ids: set[str] = set()
    for evaluation_path in sorted(RESULT_DIR.glob("*/evaluation.json")):
        run_dir = evaluation_path.parent
        reasons: list[str] = []
        try:
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
            run_config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
            rejected.append({"run_dir": str(run_dir.relative_to(ROOT)), "reasons": [f"unreadable:{exc}"]})
            continue
        provenance = evaluation.get("provenance", {})
        evidence_lock = run_config.get("confirmatory", {}).get("evidence_lock", {})
        config_hash = config_fingerprint(run_config)
        run_id = str(evaluation.get("run_id", ""))
        if evaluation.get("contract") != "dual_binary_hemodynamic_confirmatory_v2":
            reasons.append("wrong_contract")
        if run_config.get("confirmatory", {}).get("execution_context") != "server_confirmatory":
            reasons.append("config_not_server_confirmatory")
        if provenance.get("execution_context") != "server_confirmatory":
            reasons.append("evaluation_not_server_confirmatory")
        if not provenance.get("hostname"):
            reasons.append("missing_hostname")
        if not str(provenance.get("device", "")).startswith("cuda"):
            reasons.append("non_cuda_device")
        if provenance.get("config_sha256") != config_hash:
            reasons.append("evaluation_config_hash_mismatch")
        if config_hash not in expected:
            reasons.append("config_not_in_locked_25")
        if provenance.get("source_sha256") != expected_source_hash:
            reasons.append("source_hash_mismatch")
        if provenance.get("target_sha256") != expected_target_hash:
            reasons.append("target_hash_mismatch")
        if evidence_lock.get("source_sha256") != expected_source_hash:
            reasons.append("config_source_lock_mismatch")
        if evidence_lock.get("target_sha256") != expected_target_hash:
            reasons.append("config_target_lock_mismatch")
        if evidence_lock.get("feature_allowlist_sha256") != expected_allowlist_hash:
            reasons.append("config_allowlist_lock_mismatch")
        if evidence_lock.get("split_manifest_sha256") != EXPECTED_SPLIT_SHA256:
            reasons.append("config_split_lock_mismatch")
        if evidence_lock.get("analysis_code_sha256") != expected_code_hash:
            reasons.append("config_analysis_code_lock_mismatch")
        if provenance.get("feature_allowlist_sha256") != expected_allowlist_hash:
            reasons.append("evaluation_allowlist_hash_mismatch")
        if provenance.get("analysis_code_sha256") != expected_code_hash:
            reasons.append("evaluation_analysis_code_hash_mismatch")
        missing = sorted(REQUIRED_ARTIFACTS - {path.name for path in run_dir.iterdir()})
        if missing:
            reasons.append(f"missing_artifacts:{','.join(missing)}")
        if (run_dir / "split_manifest.json").exists():
            if sha256_file(str(run_dir / "split_manifest.json")) != EXPECTED_SPLIT_SHA256:
                reasons.append("split_manifest_hash_mismatch")
        if (run_dir / "test_predictions.csv").exists():
            predictions = pd.read_csv(run_dir / "test_predictions.csv", low_memory=False)
            required_prediction_columns = {
                "session_id", "patient_id", "idh_event", "ih_event",
                "probability_idh_updated_mlp", "probability_ih_updated_mlp",
            }
            missing_prediction_columns = sorted(required_prediction_columns - set(predictions.columns))
            if missing_prediction_columns:
                reasons.append(f"missing_prediction_columns:{','.join(missing_prediction_columns)}")
            else:
                if len(predictions) != 29866:
                    reasons.append(f"unexpected_test_prediction_rows:{len(predictions)}")
                if predictions["session_id"].astype(str).duplicated().any():
                    reasons.append("duplicate_test_prediction_session_ids")
                if predictions["patient_id"].astype(str).nunique() != 172:
                    reasons.append("unexpected_test_prediction_patient_count")
                if (run_dir / "split_manifest.json").exists():
                    split = json.loads((run_dir / "split_manifest.json").read_text(encoding="utf-8"))
                    if set(predictions["patient_id"].astype(str)) != {
                        str(value) for value in split["target_test"]["patient_ids"]
                    }:
                        reasons.append("test_prediction_patient_set_mismatch")
                for endpoint in ("idh", "ih"):
                    outcome = pd.to_numeric(predictions[f"{endpoint}_event"], errors="coerce")
                    probability = pd.to_numeric(
                        predictions[f"probability_{endpoint}_updated_mlp"], errors="coerce"
                    )
                    if not outcome.dropna().isin([0, 1]).all() or outcome.isna().any():
                        reasons.append(f"invalid_{endpoint}_prediction_outcomes")
                        continue
                    if probability.isna().any() or not probability.between(0, 1).all():
                        reasons.append(f"invalid_{endpoint}_prediction_probabilities")
                        continue
                    metric = evaluation.get("metrics", {}).get(endpoint, {}).get("updated_mlp", {})
                    recalculated = {
                        "roc_auc": roc_auc_score(outcome, probability),
                        "pr_auc": average_precision_score(outcome, probability),
                        "brier": brier_score_loss(outcome, probability),
                        "ece_10_quantile_bins": expected_calibration_error(outcome, probability),
                    }
                    for name, value in recalculated.items():
                        if name not in metric or abs(float(metric[name]) - float(value)) > 1e-12:
                            reasons.append(f"{endpoint}_{name}_prediction_evaluation_mismatch")
        for endpoint in ("idh", "ih"):
            metric = evaluation.get("metrics", {}).get(endpoint, {}).get("updated_mlp", {})
            bootstrap = metric.get("patient_cluster_bootstrap", {})
            if not {"roc_auc", "pr_auc", "brier", "ece_10_quantile_bins"}.issubset(metric):
                reasons.append(f"missing_{endpoint}_updated_metrics")
            if any(
                bootstrap.get(name, {}).get("valid_replicates") != 1000
                for name in ("roc_auc", "pr_auc", "brier", "ece_10_quantile_bins")
            ):
                reasons.append(f"invalid_{endpoint}_bootstrap_replicates")
        if run_id in seen_run_ids:
            reasons.append("duplicate_run_id")
        if config_hash in seen_hashes:
            reasons.append(f"duplicate_config_hash_with:{seen_hashes[config_hash]}")
        if reasons:
            rejected.append({"run_id": run_id, "run_dir": str(run_dir.relative_to(ROOT)), "reasons": reasons})
            continue
        seen_run_ids.add(run_id)
        seen_hashes[config_hash] = run_id
        expected_config = expected[config_hash][1]
        accepted.append(
            {
                "run_id": run_id,
                "model_id": expected_config["confirmatory"]["model_id"],
                "seed": int(expected_config["training"]["initialization_seed"]),
                "hostname": provenance["hostname"],
                "device": provenance["device"],
                "config_sha256": config_hash,
            }
        )

    missing_hashes = sorted(set(expected) - set(seen_hashes))
    if missing_hashes:
        warnings.append(f"pending_locked_configs:{len(missing_hashes)}")
    complete = not errors and not missing_hashes and len(accepted) == 25
    return {
        "protocol": "architecture_matched_ablation_v1",
        "status": "complete" if complete else "incomplete",
        "expected_runs": 25,
        "accepted_server_runs": len(accepted),
        "pending_server_runs": len(missing_hashes),
        "rejected_artifacts": len(rejected),
        "errors": errors,
        "warnings": warnings,
        "accepted": accepted,
        "rejected": rejected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--write-report", type=Path, help="Optional path for a machine-readable audit report.")
    args = parser.parse_args()
    report = audit()
    if args.write_report:
        output = args.write_report if args.write_report.is_absolute() else ROOT / args.write_report
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.require_complete and report["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
