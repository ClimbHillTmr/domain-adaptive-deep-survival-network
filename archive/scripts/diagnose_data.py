"""Diagnostic script to check data quality and feature normalization."""
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf
from src.data.loader import DialysisDataLoader

def run_diagnostics():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    defaults = OmegaConf.load(os.path.join(project_root, "configs", "defaults.yaml"))
    config = OmegaConf.load(os.path.join(project_root, "configs", "config.yaml"))
    cfg = OmegaConf.merge(defaults, config)

    print("=" * 80)
    print("DATA QUALITY & FEATURE NORMALIZATION DIAGNOSTICS")
    print("=" * 80)

    # Load raw data first to check columns
    df_raw = pd.read_csv(cfg.experiment.train_csv)
    print("\n1. RAW DATA COLUMNS CHECK")
    print("   Total columns: %d" % len(df_raw.columns))
    
    # Check sequence columns
    seq_cols = cfg.columns.seq_cols
    print("\n2. SEQUENCE COLUMNS (should contain raw lists)")
    for col in seq_cols:
        if col in df_raw.columns:
            sample = df_raw[col].iloc[0]
            is_list = isinstance(sample, str) and sample.startswith("[")
            print("   [OK] %s - Type: %s, Is List: %s" % (col, type(sample).__name__, is_list))
            if is_list:
                try:
                    import ast
                    parsed = ast.literal_eval(sample)
                    print("        Length: %d, First 3: %s" % (len(parsed), parsed[:3]))
                except:
                    print("        Parse failed")
        else:
            print("   [MISSING] %s" % col)

    # Check dynamic summary columns
    dynamic_cols = cfg.columns.dynamic_cols
    print("\n3. DYNAMIC SUMMARY COLUMNS")
    for col in dynamic_cols:
        if col in df_raw.columns:
            print("   [OK] %s - Mean: %.2f, Std: %.2f" % (
                col, df_raw[col].mean(), df_raw[col].std()
            ))
        else:
            print("   [MISSING] %s" % col)

    # Check static columns
    static_cols = cfg.columns.static_cols
    print("\n4. STATIC COLUMNS")
    for col in static_cols:
        if col in df_raw.columns:
            print("   [OK] %s - Mean: %.2f, Std: %.2f, NaN%%: %.1f" % (
                col, df_raw[col].mean(), df_raw[col].std(),
                df_raw[col].isna().sum() / len(df_raw) * 100
            ))
        else:
            print("   [MISSING] %s" % col)

    # Load processed data
    print("\n" + "=" * 80)
    print("PROCESSED DATA CHECK")
    print("=" * 80)

    loader = DialysisDataLoader(cfg)
    data = loader.load_data(cfg.experiment.train_csv, is_training=True)

    print("\n5. STATIC FEATURES (after normalization)")
    print("   Shape: %s" % str(data["static"].shape))
    print("   Mean: %.4f, Std: %.4f" % (data["static"].mean(), data["static"].std()))
    print("   Min: %.4f, Max: %.4f" % (data["static"].min(), data["static"].max()))
    print("   Has NaN: %s" % np.isnan(data["static"]).any())

    print("\n6. DYNAMIC FEATURES")
    print("   Shape: %s" % str(data["dynamic"].shape))
    print("   Seq_len: %d" % data["dynamic"].shape[1])
    print("   Features: %d" % data["dynamic"].shape[2])
    print("   Mean: %.4f, Std: %.4f" % (data["dynamic"].mean(), data["dynamic"].std()))
    print("   Has NaN: %s" % np.isnan(data["dynamic"]).any())

    print("\n7. TARGETS")
    print("   Events: %d (positive rate: %.2f%%)" % (
        data["targets"]["event"].sum(),
        data["targets"]["event"].mean() * 100
    ))
    print("   Duration - Mean: %.2f, Median: %.2f" % (
        data["targets"]["duration"].mean(),
        np.median(data["targets"]["duration"])
    ))

    print("\n" + "=" * 80)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    run_diagnostics()
