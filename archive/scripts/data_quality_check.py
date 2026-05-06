"""
Data Quality Analysis Report
=============================
Compare source (深医) and target (福鼎) data distributions.
"""

import pandas as pd
import numpy as np
from scipy import stats

# Load data
source = pd.read_csv("data_preprocessing/data/深医_final_data.csv")
target = pd.read_csv("data_preprocessing/data/福鼎_final_data.csv")

print("="*80)
print("DATA QUALITY ANALYSIS REPORT")
print("="*80)

print(f"\nSource (深医): {len(source)} records")
print(f"Target (福鼎): {len(target)} records")

# 1. Feature availability
features = [
    "透析年龄", "性别", "首次透析年龄", "透析龄",
    "透前收缩压", "透前舒张压", "透前动脉压", "透前体重",
    "干体重", "高血压诊断", "透析方式", "实际透析时长",
    "超滤量MAX", "超滤率_mean", "超滤率_std",
    "history_HBP_rate", "history_HBP_count",
    "prev_SBP_mean", "prev_SBP_std",
    "prev_DBP_mean", "prev_DBP_std",
    "prev_UF_rate_mean", "prev_UF_rate_std",
    "prev_IDH_event", "prev_duration",
    "prev_pre_SBP", "prev_pre_DBP",
    "prev_pre_weight", "prev_dry_weight",
]

print("\n" + "="*80)
print("1. FEATURE AVAILABILITY AND MISSINGNESS")
print("="*80)

print(f"\n{'Feature':<25} {'Source Non-null%':>15} {'Target Non-null%':>15} {'Source nunique':>15} {'Target nunique':>15}")
print("-"*85)

for f in features:
    src_avail = f in source.columns
    tgt_avail = f in target.columns
    
    if src_avail:
        src_nonnull = source[f].notna().sum() / len(source) * 100
        src_nunique = source[f].nunique()
    else:
        src_nonnull = 0
        src_nunique = 0
    
    if tgt_avail:
        tgt_nonnull = target[f].notna().sum() / len(target) * 100
        tgt_nunique = target[f].nunique()
    else:
        tgt_nonnull = 0
        tgt_nunique = 0
    
    print(f"{f:<25} {src_nonnull:>14.1f}% {tgt_nonnull:>14.1f}% {src_nunique:>15} {tgt_nunique:>15}")

# 2. Distribution comparison for common features
print("\n" + "="*80)
print("2. DISTRIBUTION COMPARISON (Common Features)")
print("="*80)

common_features = [f for f in features if f in source.columns and f in target.columns]

print(f"\n{'Feature':<25} {'Source Mean±Std':>20} {'Target Mean±Std':>20} {'KS Statistic':>15} {'KS p-value':>12}")
print("-"*85)

for f in common_features:
    src_vals = source[f].dropna().values
    tgt_vals = target[f].dropna().values
    
    if len(src_vals) > 0 and len(tgt_vals) > 0:
        # Check if numeric
        if np.issubdtype(src_vals.dtype, np.number) and np.issubdtype(tgt_vals.dtype, np.number):
            src_mean = np.mean(src_vals)
            src_std = np.std(src_vals)
            tgt_mean = np.mean(tgt_vals)
            tgt_std = np.std(tgt_vals)
            
            # KS test
            ks_stat, ks_p = stats.ks_2samp(src_vals, tgt_vals)
            
            print(f"{f:<25} {src_mean:>8.2f}±{src_std:<10.2f} {tgt_mean:>8.2f}±{tgt_std:<10.2f} {ks_stat:>15.4f} {ks_p:>12.2e}")
        else:
            # Categorical - compare value counts
            src_counts = source[f].value_counts(normalize=True)
            tgt_counts = target[f].value_counts(normalize=True)
            print(f"{f:<25} {'CATEGORICAL':>20} {'CATEGORICAL':>20}")
            print(f"  Source: {dict(src_counts)}")
            print(f"  Target: {dict(tgt_counts)}")

# 3. Target variable analysis
print("\n" + "="*80)
print("3. TARGET VARIABLE ANALYSIS")
print("="*80)

for event_col in ["透中低血压_计算", "透中低血压_计算_v6", "events", "events_v6"]:
    if event_col in source.columns:
        src_rate = source[event_col].mean()
        print(f"Source {event_col}: {src_rate:.2%}")
    if event_col in target.columns:
        tgt_rate = target[event_col].mean()
        print(f"Target {event_col}: {tgt_rate:.2%}")

for dur_col in ["et_min", "et_min_v6", "duration_minutes"]:
    if dur_col in source.columns:
        src_dur = source[dur_col].describe()
        print(f"Source {dur_col}: mean={src_dur['mean']:.1f}, median={src_dur['50%']:.1f}")
    if dur_col in target.columns:
        tgt_dur = target[dur_col].describe()
        print(f"Target {dur_col}: mean={tgt_dur['mean']:.1f}, median={tgt_dur['50%']:.1f}")

# 4. Outlier detection
print("\n" + "="*80)
print("4. OUTLIER DETECTION (IQR method)")
print("="*80)

numeric_features = [f for f in common_features if f in source.columns and f in target.columns 
                    and np.issubdtype(source[f].dtype, np.number) and np.issubdtype(target[f].dtype, np.number)]

print(f"\n{'Feature':<25} {'Source Outliers%':>18} {'Target Outliers%':>18}")
print("-"*65)

for f in numeric_features:
    src_vals = source[f].dropna().values
    tgt_vals = target[f].dropna().values
    
    if len(src_vals) > 100:
        Q1, Q3 = np.percentile(src_vals, [25, 75])
        IQR = Q3 - Q1
        src_outliers = ((src_vals < Q1 - 1.5*IQR) | (src_vals > Q3 + 1.5*IQR)).sum() / len(src_vals) * 100
    else:
        src_outliers = 0
    
    if len(tgt_vals) > 100:
        Q1, Q3 = np.percentile(tgt_vals, [25, 75])
        IQR = Q3 - Q1
        tgt_outliers = ((tgt_vals < Q1 - 1.5*IQR) | (tgt_vals > Q3 + 1.5*IQR)).sum() / len(tgt_vals) * 100
    else:
        tgt_outliers = 0
    
    print(f"{f:<25} {src_outliers:>17.1f}% {tgt_outliers:>17.1f}%")

# 5. Recommendations
print("\n" + "="*80)
print("5. RECOMMENDATIONS")
print("="*80)

print("""
Based on the analysis:

1. FEATURES TO REMOVE (center-specific or all-NaN):
   - history_HBP_rate, history_HBP_count: All NaN or constant
   - prev_* features: All NaN (data preprocessing not implemented)
   - 透析方式: Different encoding between centers

2. FEATURES TO KEEP (cross-center comparable):
   - 透析年龄, 性别, 首次透析年龄, 透析龄
   - 透前收缩压, 透前舒张压, 透前动脉压, 透前体重
   - 干体重, 高血压诊断
   - 实际透析时长, 超滤量MAX

3. LABEL SHIFT:
   - Source event rate: ~14%
   - Target event rate: ~38%
   - Need BBSE or importance weighting

4. DISTRIBUTION SHIFT:
   - Check KS test p-values < 0.05 for significant shifts
""")
