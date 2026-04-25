"""
P0: Feature Selection for Static Features
Uses correlation-based and univariate Cox-like analysis to identify key features.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel

defaults = OmegaConf.load("configs/defaults.yaml")
config = OmegaConf.load("configs/config.yaml")
cfg = OmegaConf.merge(defaults, config)

# Load data
from src.data.loader import DialysisDataLoader

loader = DialysisDataLoader(cfg)
data = loader.load_data(cfg.experiment.train_csv, is_training=True)

X_static = data["static"]
events = data["targets"]["event"]
durations = data["targets"]["duration"]

# Get feature names
static_cols = [
    c
    for c in cfg.columns.static_cols
    if c in pd.read_csv(cfg.experiment.train_csv).columns
]
static_cols = [c for c in static_cols if c != "透前体温"]

print("=== Feature Selection ===")
print(f"Total static features: {X_static.shape[1]}")
print(f"Features: {static_cols}")
print(f"Samples: {X_static.shape[0]}")
print(f"Events: {events.sum()} ({events.mean()*100:.1f}%)")

# Method 1: Correlation with event/duration
print("\n=== Method 1: Correlation Analysis ===")
print(f"{'Feature':<20} | {'Corr(Event)':>12} | {'Corr(Duration)':>15} | {'|Corr|':>8}")
print("-" * 60)

correlations = []
for i, col in enumerate(static_cols):
    if i >= X_static.shape[1]:
        break
    feat = X_static[:, i]
    corr_event = np.corrcoef(feat, events)[0, 1]
    corr_dur = np.corrcoef(feat, durations)[0, 1]
    avg_corr = (abs(corr_event) + abs(corr_dur)) / 2
    correlations.append((col, corr_event, corr_dur, avg_corr))
    print(f"{col:<20} | {corr_event:>12.4f} | {corr_dur:>15.4f} | {avg_corr:>8.4f}")

correlations.sort(key=lambda x: x[3], reverse=True)

# Method 2: Lasso on risk score approximation
print("\n=== Method 2: Lasso Feature Selection ===")

# Create pseudo-risk score: higher risk for events with shorter duration
pseudo_risk = events * (1.0 / (durations + 1))

# Subsample for faster Lasso
np.random.seed(42)
max_samples = 100000
if X_static.shape[0] > max_samples:
    idx = np.random.choice(X_static.shape[0], max_samples, replace=False)
    X_sub = X_static[idx]
    y_sub = pseudo_risk[idx]
    print(f"Subsampled to {max_samples}")
else:
    X_sub = X_static
    y_sub = pseudo_risk

# Fit Lasso
lasso = Lasso(alpha=0.01, max_iter=10000, random_state=42)
lasso.fit(X_sub, y_sub)

feature_importance = list(zip(static_cols, lasso.coef_))
feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)

print(f"\n{'Feature':<20} | {'Lasso Coef':>12} | {'Selected':>8}")
print("-" * 45)

selected_features = []
for name, coef in feature_importance:
    is_selected = abs(coef) > 0.001
    if is_selected:
        selected_features.append(name)
    print(f"{name:<20} | {coef:>12.4f} | {'YES' if is_selected else 'NO':>8}")

print(f"\nLasso selected: {len(selected_features)}/{len(static_cols)} features")

# Combine both methods
print("\n=== Combined Feature Selection ===")
# Features that are either high correlation OR selected by Lasso
corr_selected = [c[0] for c in correlations if c[3] > 0.02][:12]
combined = list(set(selected_features + corr_selected))

print(f"Final selected features: {len(combined)}/{len(static_cols)}")
print(f"Selected: {combined}")

print("\n=== Recommended static_cols for config ===")
for f in combined:
    print(f'  - "{f}"')
