import pandas as pd
import numpy as np
from omegaconf import OmegaConf
import ast

defaults = OmegaConf.load("configs/defaults.yaml")
config = OmegaConf.load("configs/config.yaml")
cfg = OmegaConf.merge(defaults, config)

# Check what columns exist in the CSV
df = pd.read_csv(cfg.experiment.train_csv, nrows=3)

# Check seq_cols
seq_cols = cfg.columns.seq_cols
print("=== seq_cols from config ===")
print(seq_cols)

print("\n=== Which seq_cols exist in CSV? ===")
for col in seq_cols:
    exists = col in df.columns
    print(f"  {col}: {exists}")

# Check dynamic_cols
dyn_cols = cfg.columns.dynamic_cols
print("\n=== dynamic_cols from config ===")
print(dyn_cols)

print("\n=== Which dynamic_cols exist in CSV? ===")
for col in dyn_cols:
    exists = col in df.columns
    print(f"  {col}: {exists}")

# If seq_cols exist, check their content
for col in seq_cols:
    if col in df.columns:
        print(f"\n=== Sample value of {col} ===")
        val = df[col].iloc[0]
        print(f"  Raw: {str(val)[:200]}")
        try:
            parsed = ast.literal_eval(val)
            print(f"  Type: {type(parsed)}, Length: {len(parsed)}")
        except Exception as e:
            print(f"  Could not parse: {e}")

# Now run the actual loader to check seq_len
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.loader import DialysisDataLoader

print("\n=== Running DialysisDataLoader ===")
loader = DialysisDataLoader(cfg)
data = loader.load_data(cfg.experiment.train_csv, is_training=True)

dynamic = data["dynamic"]
print(f"Dynamic shape: {dynamic.shape}")
print(f"  Batch size: {dynamic.shape[0]}")
print(f"  Seq_len: {dynamic.shape[1]}")
print(f"  Features: {dynamic.shape[2]}")

# Check if all seq_len=1 or some have more
seq_lens = []
for i in range(min(100, dynamic.shape[0])):
    for t in range(dynamic.shape[1]):
        if not np.all(dynamic[i, t, :] == 0):
            seq_lens.append(t + 1)

if seq_lens:
    print(f"\nNon-zero sequence lengths (first 100 samples):")
    print(
        f"  Min: {min(seq_lens)}, Max: {max(seq_lens)}, Mean: {np.mean(seq_lens):.2f}"
    )
else:
    print("\nAll sequences are zero (summary mode)")
