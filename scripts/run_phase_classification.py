import os
import sys
import faulthandler
import hydra
from omegaconf import DictConfig, OmegaConf

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.loader import load_and_build_features, load_seq_static_survival
from src.training.runner import run_once, run_train_external

faulthandler.enable()
os.environ.setdefault("OMP_NUM_THREADS", "1")

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    
    train_csv = cfg.experiment.train_csv
    external_csv = cfg.experiment.external_csv
    save_dir = cfg.experiment.save_dir
    
    # Ensure save dir exists
    os.makedirs(save_dir, exist_ok=True)

    boundaries = (cfg.boundaries.b1, cfg.boundaries.b2, cfg.boundaries.b3)
    columns = cfg.columns
    missing_threshold = 0.4 # Default or from cfg
    gap_threshold = 15
    
    if cfg.model.name == "lgbm":
        # Legacy path for LightGBM if needed, or adapt to use new loader
        pass
    else:
        # Transformer / Hybrid Path
        print("Loading Training Data...")
        X_seq_tr, X_static_tr, y_tr, durations_tr, events_tr = load_seq_static_survival(
            train_csv,
            boundaries=boundaries,
            columns=columns,
            missing_threshold=missing_threshold,
            gap_threshold=gap_threshold,
        )
        
        X_seq_ext, X_static_ext, y_ext, durations_ext, events_ext = (None, None, None, None, None)
        if external_csv:
            print("Loading External Data...")
            X_seq_ext, X_static_ext, y_ext, durations_ext, events_ext = (
                load_seq_static_survival(
                    external_csv,
                    boundaries=boundaries,
                    columns=columns,
                    missing_threshold=missing_threshold,
                    gap_threshold=gap_threshold,
                )
            )
            
            # Pack data into dicts as expected by the new runner
            # Note: The runner expects specific dict structure.
            # We can manually pack it here to bridge the gap between the wrapper and the runner.
            # Or we can update the runner to accept these unpacked arguments.
            # Given I wrote the runner to take (X_tr, y_tr...), let's check the runner signature.
            # Runner: run_train_external(X_tr, y_tr, X_ext, y_ext, ...)
            # And inside runner, it assumes X_tr is a dict.
            
            data_tr = {
                "static": X_static_tr,
                "dynamic": X_seq_tr,
                "targets": {"label": y_tr, "duration": durations_tr, "event": events_tr}
            }
            
            data_ext = {
                "static": X_static_ext,
                "dynamic": X_seq_ext,
                "targets": {"label": y_ext, "duration": durations_ext, "event": events_ext}
            } if X_seq_ext is not None else None

            out = run_train_external(
                data_tr,
                y_tr,
                data_ext,
                y_ext,
                save_dir=save_dir,
                model_type="transformer",
                boundaries=boundaries,
                calibrate=True,
                calibration_method="isotonic",
                cv_folds=cfg.training.cv_folds,
                seeds=[cfg.experiment.seed],
                coral=(cfg.training.coral.enabled),
            )
            print(out)

if __name__ == "__main__":
    main()
