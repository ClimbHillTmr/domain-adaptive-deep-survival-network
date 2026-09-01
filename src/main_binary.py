"""Explicit training entrypoint for the binary IDH target-center update study."""

from __future__ import annotations

import argparse
import json
import pickle
import socket
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.data.binary_dataset import ENDPOINTS, ROOT, prepare_binary_data, write_binary_metadata
from src.evaluate.audit_binary_data import run_audit
from src.evaluate.binary_metrics import (
    evaluate_binary,
    holm_correction,
    paired_patient_bootstrap_co_primary,
    paired_patient_bootstrap_delta,
    select_youden_threshold,
)
from src.reproducibility import (
    confirmatory_analysis_fingerprint,
    config_fingerprint,
    record_environment,
    seed_everything,
    sha256_file,
)
from src.train.binary_models import (
    calibrate_probability,
    fit_logistic_with_validation,
    fit_probability_calibrator,
    fit_source_and_update_mlp,
    predict_mlp,
    CDANBinaryMLP,
)


def _select_branch_indices(endpoint: str, model_config: dict) -> list[int] | None:
    if "alignment_strategy" not in model_config:
        if not bool(model_config.get("use_stratified_alignment", False)):
            return None
        if bool(model_config.get("use_outcome_specific_alignment", False)):
            return model_config.get("physio_indices") if endpoint == "idh" else model_config.get("treat_indices")
        return model_config.get("physio_indices")
    if model_config.get("architecture") == "single_encoder":
        return None
    strategy = model_config.get("alignment_strategy")
    if strategy == "random_feature_coral":
        if "random_aligned_indices" in model_config:
            return model_config.get("random_aligned_indices")
        return model_config.get(f"random_{endpoint}_indices")
    if strategy in {"physiology_coral", "branch_a_coral"}:
        return model_config.get("physio_indices")
    if strategy in {"treatment_coral", "branch_b_coral"}:
        return model_config.get("treat_indices")
    if strategy == "outcome_specific_coral":
        if endpoint == "idh":
            return model_config.get("physio_indices")
        if endpoint == "ih":
            return model_config.get("treat_indices")
    if model_config.get("architecture") == "dual_branch":
        return model_config.get("physio_indices")
    if not bool(model_config.get("use_stratified_alignment", False)):
        return None
    return model_config.get("physio_indices")


def _alignment_scope(model_config: dict) -> str:
    if "alignment_strategy" not in model_config:
        return "selected_branch" if bool(model_config.get("use_outcome_specific_alignment", False)) else "global"
    strategy = model_config.get("alignment_strategy")
    if strategy in {"physiology_coral", "branch_a_coral"}:
        return "branch_a"
    if strategy in {"treatment_coral", "branch_b_coral"}:
        return "branch_b"
    if strategy == "random_feature_coral":
        return str(model_config.get("alignment_scope", "branch_a"))
    if strategy == "outcome_specific_coral":
        return "selected_branch"
    return "global"


def _arrays(bundle, name: str, outcome_col: str) -> tuple[np.ndarray, np.ndarray]:
    return bundle.arrays(getattr(bundle, name), outcome_col)


def _probability_frame(bundle, probabilities: dict[str, np.ndarray], patient_col: str) -> pd.DataFrame:
    outcome_columns = [column for endpoint in ENDPOINTS.values() for column in endpoint.values()]
    frame = bundle.target_test[["session_id", patient_col, *outcome_columns]].copy()
    frame = frame.rename(columns={patient_col: "patient_id"})
    for name, values in probabilities.items():
        frame[f"probability_{name}"] = values
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "conf" / "binary_config.yaml")
    parser.add_argument("--initialization-seed", type=int)
    parser.add_argument("--train", action="store_true", help="Required safety gate for model fitting.")
    parser.add_argument(
        "--execution-context",
        choices=("server_confirmatory", "server_v3"),
        help="Required for confirmatory-v2 or V3 server fitting; prevents accidental local invocation.",
    )
    args = parser.parse_args()
    if not args.train:
        raise SystemExit("Training was not started. Re-run with --train after reviewing the binary data preflight.")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    confirmatory = config.get("confirmatory", {})
    v3 = config.get("v3", {})
    if confirmatory.get("data_contract") == "dual_binary_hemodynamic_confirmatory_v2":
        if (
            args.execution_context != "server_confirmatory"
            or confirmatory.get("execution_context") != "server_confirmatory"
        ):
            raise SystemExit(
                "Confirmatory-v2 training was not started. Both the locked config and command "
                "must declare --execution-context server_confirmatory."
            )
        if not torch.cuda.is_available():
            raise SystemExit(
                "Confirmatory-v2 training was not started: a CUDA device is required on the compute server."
            )
        evidence_lock = confirmatory.get("evidence_lock", {})
        required_lock_fields = {
            "source_sha256", "target_sha256", "feature_allowlist_sha256",
            "split_manifest_sha256", "analysis_code_sha256",
        }
        if not required_lock_fields.issubset(evidence_lock):
            raise SystemExit("Confirmatory-v2 training was not started: the config evidence lock is incomplete.")
        locked_paths = {
            "source_sha256": ROOT / config["data"]["source_file"],
            "target_sha256": ROOT / config["data"]["target_file"],
            "feature_allowlist_sha256": ROOT / config["data"]["feature_allowlist"],
            "split_manifest_sha256": ROOT / "experiments" / "final_results" / "main_mechanism_aware" / "split_manifest.json",
        }
        for lock_name, path in locked_paths.items():
            if not path.is_file() or sha256_file(str(path)) != evidence_lock[lock_name]:
                raise SystemExit(
                    f"Confirmatory-v2 training was not started: {lock_name} does not match the locked bytes."
                )
        if confirmatory_analysis_fingerprint(ROOT) != evidence_lock["analysis_code_sha256"]:
            raise SystemExit(
                "Confirmatory-v2 training was not started: analysis code does not match the locked fingerprint."
            )
    if v3.get("data_contract") == "v3_hbd_data_protocol_1":
        if args.execution_context != "server_v3" or v3.get("execution_context") != "server_v3":
            raise SystemExit(
                "V3 training was not started. Both the locked config and command must declare "
                "--execution-context server_v3."
            )
        if not bool(v3.get("training_authorized", False)):
            raise SystemExit("V3 training was not started: training_authorized is false in the locked config.")
        if not torch.cuda.is_available():
            raise SystemExit("V3 training was not started: a CUDA device is required on the compute server.")
        evidence_lock = v3.get("evidence_lock", {})
        required_lock_fields = {
            "source_sha256",
            "target_sha256",
            "feature_allowlist_sha256",
            "split_manifest_sha256",
            "preprocessing_sha256",
            "primary_partition_sha256",
            "random_partitions_sha256",
            "training_protocol_sha256",
        }
        if not required_lock_fields.issubset(evidence_lock):
            raise SystemExit("V3 training was not started: the config evidence lock is incomplete.")
        locked_paths = {
            "source_sha256": ROOT / config["data"]["source_file"],
            "target_sha256": ROOT / config["data"]["target_file"],
            "feature_allowlist_sha256": ROOT / config["data"]["feature_allowlist"],
            "split_manifest_sha256": ROOT / config["data"]["split_manifest"],
            "preprocessing_sha256": ROOT / config["data"]["preprocessing_file"],
            "primary_partition_sha256": ROOT / "v3/experiments/primary_feature_partition.json",
            "random_partitions_sha256": ROOT / "v3/experiments/random_partitions.csv",
            "training_protocol_sha256": ROOT / "v3/experiments/training_protocol.yaml",
        }
        for lock_name, path in locked_paths.items():
            if not path.is_file() or sha256_file(str(path)) != evidence_lock[lock_name]:
                raise SystemExit(
                    f"V3 training was not started: {lock_name} does not match the locked bytes."
                )
    if args.initialization_seed is not None:
        config["training"]["initialization_seed"] = args.initialization_seed
    seed = int(config["training"]["initialization_seed"])
    seed_everything(seed)
    if v3.get("data_contract") == "v3_hbd_data_protocol_1":
        audit = {
            "passed": True,
            "mode": "v3_frozen_manifests",
            "split_manifest": config["data"]["split_manifest"],
            "preprocessing_file": config["data"]["preprocessing_file"],
        }
    else:
        audit = run_audit(args.config)
        if not audit["passed"]:
            raise SystemExit("Binary data preflight failed. Rebuild and review the cohorts before training.")

    data_config = config["data"]
    source_path = ROOT / data_config["source_file"]
    target_path = ROOT / data_config["target_file"]
    allowlist_path = ROOT / data_config["feature_allowlist"]
    bundle = prepare_binary_data(
        source_path,
        target_path,
        allowlist_path=allowlist_path,
        patient_col=data_config["patient_col"],
        split_seed=data_config["split_seed"],
        source_validation_fraction=data_config["source_validation_fraction"],
        target_update_fraction=data_config["target_update_fraction"],
        target_validation_fraction=data_config["target_validation_fraction"],
        split_manifest_path=ROOT / data_config["split_manifest"] if data_config.get("split_manifest") else None,
        preprocessing_path=ROOT / data_config["preprocessing_file"] if data_config.get("preprocessing_file") else None,
        use_cache=True,
    )
    training = config["training"]
    model_config = config["model"]
    c_values = tuple(float(value) for value in model_config["logistic_c_values"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluation = config["evaluation"]
    patient_ids = bundle.target_test[data_config["patient_col"]].astype(str).to_numpy()
    all_probabilities: dict[str, np.ndarray] = {}
    metrics, comparisons, training_history = {}, {}, {}
    logistic_models, calibrators, mlp_states = {}, {}, {}
    for endpoint, columns in ENDPOINTS.items():
        outcome_col = columns["event"]
        x_source_train, y_source_train = _arrays(bundle, "source_train", outcome_col)
        x_source_val, y_source_val = _arrays(bundle, "source_val", outcome_col)
        x_target_train, y_target_train = _arrays(bundle, "target_train", outcome_col)
        x_target_val, y_target_val = _arrays(bundle, "target_val", outcome_col)
        x_test, y_test = _arrays(bundle, "target_test", outcome_col)

        source_logistic = fit_logistic_with_validation(
            x_source_train, y_source_train, x_source_val, y_source_val, c_values=c_values, seed=seed
        )
        local_logistic = fit_logistic_with_validation(
            x_target_train, y_target_train, x_target_val, y_target_val, c_values=c_values, seed=seed
        )
        training_mode = str(model_config.get("training_mode", "supervised_update"))
        branch_indices = _select_branch_indices(endpoint, model_config)
        if model_config.get("architecture") == "dual_branch":
            if model_config.get("architecture_partition") == "random_aligned":
                branch_indices = model_config.get("random_aligned_indices")
            else:
                branch_indices = model_config.get("dual_branch_physio_indices", model_config.get("physio_indices"))
        if training_mode == "target_only":
            source_mlp, updated_mlp, endpoint_history = fit_source_and_update_mlp(
                x_target_train,
                y_target_train,
                x_target_val,
                y_target_val,
                x_target_train,
                y_target_train,
                x_target_val,
                y_target_val,
                hidden_dims=tuple(model_config["mlp_hidden_dims"]),
                dropout=float(model_config["dropout"]),
                pretrain_learning_rate=float(training["pretrain_learning_rate"]),
                finetune_learning_rate=float(training["finetune_learning_rate"]),
                batch_size=int(training["batch_size"]),
                pretrain_epochs=int(training["pretrain_epochs"]),
                finetune_epochs=0,
                patience=int(training["patience"]),
                seed=seed,
                device=device,
                use_cdan=False,
                use_coral=False,
                use_mmd=False,
                physio_indices=None if model_config.get("architecture") == "single_encoder" else branch_indices,
                alignment_scope="global",
                pooled_update=False,
            )
            updated_mlp = source_mlp
            endpoint_history = {"source_pretrain": endpoint_history["source_pretrain"], "target_update": {"skipped": True, "mode": "target_only"}}
        else:
            source_mlp, updated_mlp, endpoint_history = fit_source_and_update_mlp(
                x_source_train,
                y_source_train,
                x_source_val,
                y_source_val,
                x_target_train,
                y_target_train,
                x_target_val,
                y_target_val,
                hidden_dims=tuple(model_config["mlp_hidden_dims"]),
                dropout=float(model_config["dropout"]),
                pretrain_learning_rate=float(training["pretrain_learning_rate"]),
                finetune_learning_rate=float(training["finetune_learning_rate"]),
                batch_size=int(training["batch_size"]),
                pretrain_epochs=int(training["pretrain_epochs"]),
                finetune_epochs=0 if training_mode == "source_only" else int(training["finetune_epochs"]),
                patience=int(training["patience"]),
                seed=seed,
                device=device,
                use_cdan=bool(model_config.get("use_cdan", False)),
                use_coral=bool(model_config.get("use_coral", False)) and training_mode != "source_only",
                use_mmd=bool(model_config.get("use_mmd", False)) and training_mode != "source_only",
                d_model=int(model_config.get("d_model", 64)),
                nhead=int(model_config.get("nhead", 4)),
                num_layers=int(model_config.get("num_layers", 2)),
                domain_hidden=int(model_config.get("domain_hidden", 64)),
                adversarial_weight=float(model_config.get("adversarial_weight", 0.01)),
                mask_l1_weight=float(model_config.get("mask_l1_weight", 0.01)),
                coral_weight=float(model_config.get("coral_weight", 0.01)),
                mmd_weight=float(model_config.get("mmd_weight", 1.0)),
                physio_indices=None if model_config.get("architecture") == "single_encoder" else branch_indices,
                alignment_scope=_alignment_scope(model_config),
                pooled_update=bool(model_config.get("pooled_update", False)),
            )
            if training_mode == "source_only":
                updated_mlp = source_mlp
                endpoint_history["target_update"] = {"skipped": True, "mode": "source_only"}
        use_cdan = bool(model_config.get("use_cdan", False))
        raw_val_probabilities = {
            "source_logistic": source_logistic.predict_proba(x_source_val)[:, 1],
            "local_logistic": local_logistic.predict_proba(x_target_val)[:, 1],
            "source_mlp": predict_mlp(source_mlp, x_source_val, device, is_cdan=use_cdan),
            "updated_mlp": predict_mlp(updated_mlp, x_target_val, device, is_cdan=use_cdan),
        }
        raw_probabilities = {
            "source_logistic": source_logistic.predict_proba(x_test)[:, 1],
            "local_logistic": local_logistic.predict_proba(x_test)[:, 1],
            "source_mlp": predict_mlp(source_mlp, x_test, device, is_cdan=use_cdan),
            "updated_mlp": predict_mlp(updated_mlp, x_test, device, is_cdan=use_cdan),
        }
        validation_outcomes = {
            "source_logistic": y_source_val,
            "local_logistic": y_target_val,
            "source_mlp": y_source_val,
            "updated_mlp": y_target_val,
        }
        endpoint_calibrators = {
            name: fit_probability_calibrator(validation_outcomes[name], probability)
            for name, probability in raw_val_probabilities.items()
        }
        calibrated_val = {
            name: calibrate_probability(endpoint_calibrators[name], probability)
            for name, probability in raw_val_probabilities.items()
        }
        probabilities = {
            name: calibrate_probability(endpoint_calibrators[name], probability)
            for name, probability in raw_probabilities.items()
        }
        thresholds = {
            name: select_youden_threshold(validation_outcomes[name], probability)
            for name, probability in calibrated_val.items()
        }
        metrics[endpoint] = {
            name: evaluate_binary(
                y_test,
                probability,
                patient_ids,
                threshold=thresholds[name],
                n_bootstrap=int(evaluation["patient_bootstrap_replicates"]),
                seed=int(evaluation["bootstrap_seed"]),
            )
            for name, probability in probabilities.items()
        }
        comparisons[endpoint] = {
            "updated_mlp_vs_source_mlp": paired_patient_bootstrap_delta(
                y_test,
                probabilities["updated_mlp"],
                probabilities["source_mlp"],
                patient_ids,
                n_bootstrap=int(evaluation["patient_bootstrap_replicates"]),
                seed=int(evaluation["bootstrap_seed"]),
            ),
            "updated_mlp_vs_local_logistic": paired_patient_bootstrap_delta(
                y_test,
                probabilities["updated_mlp"],
                probabilities["local_logistic"],
                patient_ids,
                n_bootstrap=int(evaluation["patient_bootstrap_replicates"]),
                seed=int(evaluation["bootstrap_seed"]),
            ),
        }
        all_probabilities.update({f"{endpoint}_{name}": values for name, values in probabilities.items()})
        all_probabilities.update(
            {f"raw_{endpoint}_{name}": values for name, values in raw_probabilities.items()}
        )
        endpoint_history["logistic"] = {
            "source_selected_c": source_logistic.selected_c_,
            "local_selected_c": local_logistic.selected_c_,
        }
        training_history[endpoint] = endpoint_history
        logistic_models[endpoint] = {"source_logistic": source_logistic, "local_logistic": local_logistic}
        calibrators[endpoint] = endpoint_calibrators
        mlp_states[endpoint] = {
            "source_mlp": {name: value.detach().cpu() for name, value in source_mlp.state_dict().items()},
            "updated_mlp": {name: value.detach().cpu() for name, value in updated_mlp.state_dict().items()},
        }

    idh_cols = ENDPOINTS["idh"]
    ih_cols = ENDPOINTS["ih"]
    y_idh = bundle.target_test[idh_cols["event"]].to_numpy()
    y_ih = bundle.target_test[ih_cols["event"]].to_numpy()
    co_primary = {
        "updated_vs_source": paired_patient_bootstrap_co_primary(
            y_idh, all_probabilities["idh_updated_mlp"], all_probabilities["idh_source_mlp"],
            y_ih, all_probabilities["ih_updated_mlp"], all_probabilities["ih_source_mlp"],
            patient_ids,
            n_bootstrap=int(evaluation["patient_bootstrap_replicates"]),
            seed=int(evaluation["bootstrap_seed"]),
        ),
        "updated_vs_local": paired_patient_bootstrap_co_primary(
            y_idh, all_probabilities["idh_updated_mlp"], all_probabilities["idh_local_logistic"],
            y_ih, all_probabilities["ih_updated_mlp"], all_probabilities["ih_local_logistic"],
            patient_ids,
            n_bootstrap=int(evaluation["patient_bootstrap_replicates"]),
            seed=int(evaluation["bootstrap_seed"]),
        ),
    }

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    config_sha256 = config_fingerprint(config)
    run_id = f"binary_{timestamp}_{config_sha256[:12]}"
    output = ROOT / config["paths"]["output_dir"] / run_id
    output.mkdir(parents=True, exist_ok=False)
    write_binary_metadata(bundle, output)
    (output / "data_preflight.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _probability_frame(bundle, all_probabilities, data_config["patient_col"]).to_csv(
        output / "test_predictions.csv", index=False
    )
    payload = {
        "run_id": run_id,
        "contract": config.get("confirmatory", {}).get(
            "data_contract", "dual_binary_hemodynamic_v1"
        ),
        "confirmatory": config.get("confirmatory"),
        "split_seed": data_config["split_seed"],
        "initialization_seed": seed,
        "feature_names": bundle.feature_names,
        "metrics": metrics,
        "paired_comparisons": comparisons,
        "co_primary_analysis": co_primary,
        "training_history": training_history,
        "probability_calibration": {
            "method": "Platt scaling",
            "source_models": "source validation patients",
            "target_models": "target validation patients",
        },
        "provenance": {
            "config_sha256": config_sha256,
            "source_sha256": sha256_file(str(source_path)),
            "target_sha256": sha256_file(str(target_path)),
            "feature_allowlist_sha256": sha256_file(str(allowlist_path)),
            "analysis_code_sha256": confirmatory_analysis_fingerprint(ROOT),
            "device": str(device),
            "hostname": socket.gethostname(),
            "execution_context": args.execution_context,
        },
    }
    (output / "evaluation.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    with (output / "logistic_models.pkl").open("wb") as handle:
        pickle.dump({"models": logistic_models, "calibrators": calibrators}, handle)
    torch.save(mlp_states, output / "mlp_models.pt")
    record_environment(log_dir=output, filename="environment.txt")
    print(json.dumps({"run_id": run_id, "output_dir": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
