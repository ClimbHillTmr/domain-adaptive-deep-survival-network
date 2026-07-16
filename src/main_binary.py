"""Explicit training entrypoint for the binary IDH target-center update study."""

from __future__ import annotations

import argparse
import json
import pickle
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
    paired_patient_bootstrap_delta,
    select_youden_threshold,
)
from src.reproducibility import config_fingerprint, record_environment, seed_everything, sha256_file
from src.train.binary_models import (
    calibrate_probability,
    fit_logistic_with_validation,
    fit_probability_calibrator,
    fit_source_and_update_mlp,
    predict_mlp,
)


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
    args = parser.parse_args()
    if not args.train:
        raise SystemExit("Training was not started. Re-run with --train after reviewing the binary data preflight.")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.initialization_seed is not None:
        config["training"]["initialization_seed"] = args.initialization_seed
    seed = int(config["training"]["initialization_seed"])
    seed_everything(seed)
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
            finetune_epochs=int(training["finetune_epochs"]),
            patience=int(training["patience"]),
            seed=seed,
            device=device,
        )
        raw_val_probabilities = {
            "source_logistic": source_logistic.predict_proba(x_source_val)[:, 1],
            "local_logistic": local_logistic.predict_proba(x_target_val)[:, 1],
            "source_mlp": predict_mlp(source_mlp, x_source_val, device),
            "updated_mlp": predict_mlp(updated_mlp, x_target_val, device),
        }
        raw_probabilities = {
            "source_logistic": source_logistic.predict_proba(x_test)[:, 1],
            "local_logistic": local_logistic.predict_proba(x_test)[:, 1],
            "source_mlp": predict_mlp(source_mlp, x_test, device),
            "updated_mlp": predict_mlp(updated_mlp, x_test, device),
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
        "contract": "dual_binary_hemodynamic_v1",
        "split_seed": data_config["split_seed"],
        "initialization_seed": seed,
        "feature_names": bundle.feature_names,
        "metrics": metrics,
        "paired_comparisons": comparisons,
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
            "device": str(device),
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
