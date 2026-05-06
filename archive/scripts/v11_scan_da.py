"""
v11_scan_da.py

Small hyperparameter scan for v11 supervised domain adaptation.
Scans:
- source_replay_weight
- conditional_align_weight

The source pretraining and target split are fixed across all runs.
"""

import copy
import csv
import importlib.util
import json
from pathlib import Path


ROOT = Path("/home/cht/Works/domain-adaptive-deep-survival-network")
BASE_SCRIPT = ROOT / "scripts" / "v11_train_layerwise_da.py"
RESULTS_JSON = ROOT / "runs" / "v11_scan_results.json"
RESULTS_CSV = ROOT / "runs" / "v11_scan_results.csv"


def load_v11_module():
    spec = importlib.util.spec_from_file_location("v11_module", BASE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main():
    v11 = load_v11_module()
    v11.set_seed(v11.CONFIG["seed"])

    print("1. Load fixed data split and pretrain source model once...")
    x_s, e_s, t_s, scaler = v11.load_relative_features(
        v11.CONFIG["source_data"], is_training=True
    )
    x_t, e_t, t_t, _ = v11.load_relative_features(
        v11.CONFIG["target_data"], is_training=False, scaler=scaler
    )

    (
        x_t_adapt,
        x_t_test,
        e_t_adapt,
        e_t_test,
        t_t_adapt,
        t_t_test,
    ) = v11.train_test_split(
        x_t,
        e_t,
        t_t,
        test_size=1 - v11.CONFIG["target_adapt_ratio"],
        random_state=v11.CONFIG["seed"],
        stratify=e_t,
    )

    source_loader = v11.make_loader(
        x_s, e_s, t_s, v11.CONFIG["batch_size"], shuffle=True, drop_last=True
    )
    target_adapt_loader = v11.make_loader(
        x_t_adapt,
        e_t_adapt,
        t_t_adapt,
        v11.CONFIG["batch_size"],
        shuffle=True,
        drop_last=False,
    )
    target_test = (x_t_test, e_t_test, t_t_test)

    input_dim = x_s.shape[1]
    base_model = v11.SurvivalNet(
        input_dim, v11.CONFIG["d_model"], v11.CONFIG["dropout"]
    ).to(v11.CONFIG["device"])
    zero_shot_best, source_state = v11.pretrain_source_model(
        base_model, source_loader, target_test, v11.CONFIG
    )
    print(f"   Fixed zero-shot target C-index: {zero_shot_best:.4f}")

    replay_grid = [0.2, 0.4, 0.6]
    align_grid = [0.05, 0.1, 0.2]
    all_results = []

    print("\n2. Start v11 hyperparameter scan...")
    for replay_weight in replay_grid:
        for align_weight in align_grid:
            local_config = copy.deepcopy(v11.CONFIG)
            local_config["source_replay_weight"] = replay_weight
            local_config["conditional_align_weight"] = align_weight

            model = v11.SurvivalNet(
                input_dim, local_config["d_model"], local_config["dropout"]
            ).to(local_config["device"])
            model.load_state_dict(source_state)

            best_score, _, history = v11.run_layerwise_da(
                model,
                source_loader,
                target_adapt_loader,
                target_test,
                local_config,
            )
            row = {
                "replay_weight": replay_weight,
                "align_weight": align_weight,
                "zero_shot_target_cindex": zero_shot_best,
                "v11_target_cindex": best_score,
                "delta_vs_zero_shot": best_score - zero_shot_best,
                "best_phase": history[-1]["phase"] if history else "unknown",
            }
            all_results.append(row)
            print(
                f"   replay={replay_weight:.2f}, align={align_weight:.2f} -> "
                f"target_cindex={best_score:.4f}, delta={best_score - zero_shot_best:+.4f}"
            )

    all_results.sort(key=lambda x: x["v11_target_cindex"], reverse=True)
    best_result = all_results[0]

    with RESULTS_JSON.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "zero_shot_target_cindex": zero_shot_best,
                "results": all_results,
                "best_result": best_result,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with RESULTS_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "replay_weight",
                "align_weight",
                "zero_shot_target_cindex",
                "v11_target_cindex",
                "delta_vs_zero_shot",
                "best_phase",
            ],
        )
        writer.writeheader()
        writer.writerows(all_results)

    print("\n============================================================")
    print(f"Best replay weight    : {best_result['replay_weight']}")
    print(f"Best align weight     : {best_result['align_weight']}")
    print(f"Best target C-index   : {best_result['v11_target_cindex']:.4f}")
    print(f"Delta vs zero-shot    : {best_result['delta_vs_zero_shot']:+.4f}")
    print(f"JSON saved to         : {RESULTS_JSON}")
    print(f"CSV saved to          : {RESULTS_CSV}")
    print("============================================================")


if __name__ == "__main__":
    main()
