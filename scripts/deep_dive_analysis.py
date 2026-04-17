"""
Deep dive analysis of:
1. Why seq_len=1 (sequence columns missing)
2. Data quality issues (透前体温 50.3% NaN, 超滤量MAX extreme outliers)
"""
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf


def analyze_sequence_columns():
    """Analyze why sequence columns are missing and what alternatives exist."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config = OmegaConf.load(os.path.join(project_root, "configs", "config.yaml"))
    defaults = OmegaConf.load(os.path.join(project_root, "configs", "defaults.yaml"))
    cfg = OmegaConf.merge(defaults, config)

    df = pd.read_csv(cfg.experiment.train_csv)

    print("=" * 80)
    print("ISSUE 1: SEQUENCE LENGTH = 1 ANALYSIS")
    print("=" * 80)

    seq_cols = cfg.columns.seq_cols
    print("\nExpected sequence columns (from config):")
    for col in seq_cols:
        exists = col in df.columns
        print("  [%s] %s" % ("OK" if exists else "MISSING", col))

    # Look for columns that might contain sequence data
    print("\nSearching for potential sequence columns in CSV...")
    list_like_cols = []
    for col in df.columns:
        sample = df[col].iloc[0]
        if isinstance(sample, str) and (sample.startswith("[") or sample.startswith("(")):
            list_like_cols.append(col)

    if list_like_cols:
        print("Found %d columns with list-like data:" % len(list_like_cols))
        for col in list_like_cols[:10]:
            sample = df[col].iloc[0]
            try:
                import ast
                parsed = ast.literal_eval(sample)
                print("  %s - Length: %d, Type: %s" % (col, len(parsed), type(parsed).__name__))
            except:
                print("  %s - Parse failed" % col)
    else:
        print("NO columns contain list/sequence data in this CSV.")
        print("\nConclusion: The CSV only contains summary statistics (_mean columns).")
        print("To enable true Transformer modeling, you need:")
        print("  1. Raw sequence data exported from the database")
        print("  2. Or reconstruct sequences from time-stamped measurements")

    # Check what dynamic summary columns look like
    dynamic_cols = cfg.columns.dynamic_cols
    print("\nCurrent dynamic summary columns (used as seq_len=1 fallback):")
    for col in dynamic_cols:
        if col in df.columns:
            print("  %s - Mean: %.2f, Std: %.2f" % (col, df[col].mean(), df[col].std()))


def analyze_data_quality():
    """Analyze data quality issues: NaN rates and extreme outliers."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config = OmegaConf.load(os.path.join(project_root, "configs", "config.yaml"))
    defaults = OmegaConf.load(os.path.join(project_root, "configs", "defaults.yaml"))
    cfg = OmegaConf.merge(defaults, config)

    df = pd.read_csv(cfg.experiment.train_csv)

    print("\n" + "=" * 80)
    print("ISSUE 2: DATA QUALITY ANALYSIS")
    print("=" * 80)

    static_cols = cfg.columns.static_cols

    # NaN analysis
    print("\n2.1 Missing Value Analysis (Static Features):")
    print("%-20s | %-10s | %-10s | %-10s" % ("Feature", "NaN Count", "NaN %", "Action"))
    print("-" * 60)

    for col in static_cols:
        if col not in df.columns:
            continue
        nan_count = df[col].isna().sum()
        nan_pct = nan_count / len(df) * 100

        if nan_pct > 50:
            action = "DROP (>50%)"
        elif nan_pct > 20:
            action = "IMPUTE (caution)"
        elif nan_pct > 5:
            action = "IMPUTE"
        else:
            action = "OK"

        print("%-20s | %-10d | %-9.1f%% | %s" % (col, nan_count, nan_pct, action))

    # Outlier analysis
    print("\n2.2 Outlier Analysis (Static Features - numeric only):")
    print("%-20s | %-10s | %-10s | %-10s | %-10s | %-10s" % (
        "Feature", "Mean", "Std", "Min", "Max", "IQR Ratio"))
    print("-" * 80)

    for col in static_cols:
        if col not in df.columns:
            continue
        if df[col].dtype not in ["float64", "int64", "float32", "int32"]:
            continue

        vals = df[col].dropna()
        if len(vals) == 0:
            continue

        q1 = vals.quantile(0.25)
        q3 = vals.quantile(0.75)
        iqr = q3 - q1
        mean = vals.mean()
        std = vals.std()
        min_val = vals.min()
        max_val = vals.max()

        # IQR ratio: (max - min) / IQR - high values indicate outliers
        if iqr > 0:
            iqr_ratio = (max_val - min_val) / iqr
        else:
            iqr_ratio = 0

        flag = ""
        if iqr_ratio > 100:
            flag = " *** EXTREME OUTLIERS ***"
        elif iqr_ratio > 50:
            flag = " ** OUTLIERS **"
        elif iqr_ratio > 20:
            flag = " * MILD *"

        print("%-20s | %-10.2f | %-10.2f | %-10.2f | %-10.2f | %-10.2f%s" % (
            col, mean, std, min_val, max_val, iqr_ratio, flag))

    # Specific deep dive on problematic columns
    print("\n2.3 Deep Dive: 透前体温 (50.3% NaN)")
    if "透前体温" in df.columns:
        vals = df["透前体温"].dropna()
        print("  Valid samples: %d / %d (%.1f%%)" % (
            len(vals), len(df), len(vals) / len(df) * 100))
        print("  Mean: %.2f, Std: %.2f" % (vals.mean(), vals.std()))
        print("  Min: %.2f, Max: %.2f" % (vals.min(), vals.max()))
        print("  Median: %.2f" % (vals.median()))

        # Check for impossible values (normal body temp: 35-42°C)
        normal_temp = vals[(vals >= 35) & (vals <= 42)]
        abnormal_temp = vals[(vals < 35) | (vals > 42)]
        print("  Normal range (35-42°C): %d samples" % len(normal_temp))
        print("  Abnormal range: %d samples" % len(abnormal_temp))
        if len(abnormal_temp) > 0:
            print("  Abnormal values sample: %s" % abnormal_temp.head(10).values)

    print("\n2.4 Deep Dive: 超滤量MAX (Std=13986)")
    if "超滤量MAX" in df.columns:
        vals = df["超滤量MAX"].dropna()
        print("  Valid samples: %d / %d (%.1f%%)" % (
            len(vals), len(df), len(vals) / len(df) * 100))
        print("  Mean: %.2f, Std: %.2f" % (vals.mean(), vals.std()))
        print("  Min: %.2f, Max: %.2f" % (vals.min(), vals.max()))
        print("  Median: %.2f" % (vals.median()))

        # Check percentiles
        print("  Percentiles:")
        for p in [50, 75, 90, 95, 99, 99.9]:
            print("    P%d: %.2f" % (p, vals.quantile(p / 100)))

        # Identify extreme outliers (>10000 is likely data entry error)
        extreme = vals[vals > 10000]
        print("  Extreme values (>10000): %d samples (%.2f%%)" % (
            len(extreme), len(extreme) / len(vals) * 100))
        if len(extreme) > 0:
            print("  Sample extreme values: %s" % extreme.head(10).values)


def recommend_fixes():
    """Provide actionable recommendations."""
    print("\n" + "=" * 80)
    print("RECOMMENDED FIXES")
    print("=" * 80)

    print("""
1. SEQUENCE LENGTH = 1:
   Option A: Accept summary mode
     - Keep seq_len=1, treat dynamic features as static summaries
     - Rename model from "Transformer" to "Dynamic Features MLP"
     - Honest reporting: no true temporal modeling

   Option B: Reconstruct sequences
     - Query raw database for time-stamped vital signs
     - Group by session, sort by timestamp, create sequences
     - This is the "right" way but requires database access

   Option C: Synthetic sequences
     - Use summary stats + noise to generate plausible sequences
     - NOT recommended for clinical paper (data fabrication)

2. 透前体温 (50.3% NaN):
   Recommendation: DROP this feature
   - 50% missing is too high for reliable imputation
   - Mean imputation would add noise, not signal
   - If kept, use "missing indicator" pattern

3. 超滤量MAX (extreme outliers):
   Recommendation: Winsorize or clip
   - Values >10000 are likely data entry errors (extra zeros)
   - Clip to 99th percentile or use robust scaling
   - Normal ultrafiltration volume: 500-5000 mL per session

4. General data quality:
   - Add outlier detection pipeline before training
   - Use RobustScaler instead of StandardScaler for features with outliers
   - Document all data cleaning steps for paper reproducibility
""")


if __name__ == "__main__":
    analyze_sequence_columns()
    analyze_data_quality()
    recommend_fixes()
