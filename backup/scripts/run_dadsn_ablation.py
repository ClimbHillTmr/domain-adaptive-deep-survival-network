import os
import sys
import faulthandler
import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.loader import DialysisDataLoader
from src.training.dadsn_runner import run_dadsn_experiment, run_ablation_study

faulthandler.enable()
os.environ.setdefault("OMP_NUM_THREADS", "1")


def load_dataset(loader_obj, csv_path, is_training):
    data = loader_obj.load_data(csv_path, is_training=is_training)
    return {
        "static": data["static"],
        "dynamic": data["dynamic"],
        "targets": data["targets"],
    }


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    train_csv = cfg.experiment.train_csv
    external_csv = cfg.experiment.external_csv
    save_dir = cfg.experiment.save_dir

    os.makedirs(save_dir, exist_ok=True)

    shared_loader = DialysisDataLoader(cfg)

    print("Loading Source Domain (深医) Data...")
    source_data = load_dataset(shared_loader, train_csv, is_training=True)

    print("Loading Target Domain (福鼎) Data...")
    target_data = load_dataset(shared_loader, external_csv, is_training=False)

    print("\n" + "=" * 80)
    print("STARTING ABLATION STUDY")
    print("=" * 80)
    print(f"Source samples: {source_data['static'].shape[0]}")
    print(f"Target samples: {target_data['static'].shape[0]}")
    print(f"Seeds: {cfg.experiment.seeds}")
    print(f"Epochs: {cfg.training.epochs}")
    print(f"Learning rate: {cfg.training.learning_rate}")
    print(f"CORAL lambda: {cfg.training.coral.lambda_coral}")
    print("=" * 80)

    seeds = (
        cfg.experiment.seeds
        if hasattr(cfg.experiment, "seeds")
        else [cfg.experiment.seed]
    )

    results = run_ablation_study(
        source_data=source_data,
        target_data=target_data,
        save_dir=save_dir,
        config=cfg,
        seeds=seeds,
    )

    print("\n" + "=" * 80)
    print("ABLATION STUDY COMPLETED")
    print("=" * 80)

    import json

    results_path = os.path.join(save_dir, "ablation_results.json")
    with open(results_path, "w") as f:
        json.dump(
            {
                k: {
                    "mean": {kk: float(vv) for kk, vv in v[0].items()},
                    "std": {kk: float(vv) for kk, vv in v[1].items()},
                }
                for k, v in results.items()
            },
            f,
            indent=2,
        )
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
