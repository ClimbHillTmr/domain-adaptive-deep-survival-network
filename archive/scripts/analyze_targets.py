"""
Deep dive into target variable distribution and survival analysis feasibility.
"""
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf
from src.data.loader import DialysisDataLoader, OBS_WINDOW


def analyze_targets():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    defaults = OmegaConf.load(os.path.join(project_root, "configs", "defaults.yaml"))
    config = OmegaConf.load(os.path.join(project_root, "configs", "config.yaml"))
    cfg = OmegaConf.merge(defaults, config)

    print("=" * 80)
    print("TARGET VARIABLE ANALYSIS (BEFORE FILTERING)")
    print("=" * 80)

    df_source = pd.read_csv(cfg.experiment.train_csv)
    df_target = pd.read_csv(cfg.experiment.external_csv)

    event_col = cfg.columns.targets.event_col
    duration_col = cfg.columns.targets.duration_col

    print("\nSource (ShenYi): %d samples" % len(df_source))
    print("Target (FuDing): %d samples" % len(df_target))

    for name, df in [("Source", df_source), ("Target", df_target)]:
        print("\n--- %s ---" % name)
        if event_col in df.columns:
            events = df[event_col].fillna(0).astype(int)
            print("Event distribution:")
            print("  Event=0 (censored): %d (%.1f%%)" % (
                (events == 0).sum(), (events == 0).sum() / len(events) * 100))
            print("  Event=1 (occurred): %d (%.1f%%)" % (
                (events == 1).sum(), (events == 1).sum() / len(events) * 100))

        if duration_col in df.columns:
            durations = df[duration_col].fillna(0)
            print("Duration statistics:")
            print("  Mean: %.2f, Median: %.2f, Std: %.2f" % (
                durations.mean(), durations.median(), durations.std()))
            print("  Min: %.2f, Max: %.2f" % (durations.min(), durations.max()))
            print("  Duration=0: %d (%.1f%%)" % (
                (durations == 0).sum(), (durations == 0).sum() / len(durations) * 100))
            print("  Duration<=60: %d (%.1f%%)" % (
                (durations <= 60).sum(), (durations <= 60).sum() / len(durations) * 100))

            # Percentiles for positive events
            if event_col in df.columns:
                pos_events = durations[df[event_col] == 1]
                if len(pos_events) > 0:
                    print("  Duration (event=1 only):")
                    print("    Mean: %.2f, Median: %.2f" % (pos_events.mean(), pos_events.median()))
                    for p in [10, 25, 50, 75, 90]:
                        print("    P%d: %.2f" % (p, pos_events.quantile(p / 100)))

    print("\n" + "=" * 80)
    print("AFTER OBS_WINDOW FILTERING (simulating DataLoader)")
    print("=" * 80)

    loader = DialysisDataLoader(cfg)
    source_data = loader.load_data(cfg.experiment.train_csv, is_training=True)
    target_data = loader.load_data(cfg.experiment.external_csv, is_training=False)

    for name, data in [("Source", source_data), ("Target", target_data)]:
        print("\n--- %s (after filtering) ---" % name)
        events = data["targets"]["event"]
        durations = data["targets"]["duration"]

        print("  Samples: %d" % len(events))
        print("  Event=1: %d (%.1f%%)" % (
            (events == 1).sum(), (events == 1).sum() / len(events) * 100))
        print("  Duration - Mean: %.2f, Median: %.2f" % (
            np.mean(durations), np.median(durations)))
        print("  Duration - Min: %.2f, Max: %.2f" % (
            np.min(durations), np.max(durations)))

        # Check how many events have duration > OBS_WINDOW
        if (events == 1).sum() > 0:
            event_durations = durations[events == 1]
            print("  Event durations - Mean: %.2f, Median: %.2f" % (
                np.mean(event_durations), np.median(event_durations)))
            print("  Event durations > 60 min: %d (%.1f%% of events)" % (
                (event_durations > 60).sum(),
                (event_durations > 60).sum() / len(event_durations) * 100))


def analyze_predictive_power():
    """Check if features have any predictive power for the target."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    defaults = OmegaConf.load(os.path.join(project_root, "configs", "defaults.yaml"))
    config = OmegaConf.load(os.path.join(project_root, "configs", "config.yaml"))
    cfg = OmegaConf.merge(defaults, config)

    print("\n" + "=" * 80)
    print("FEATURE-TARGET CORRELATION ANALYSIS")
    print("=" * 80)

    loader = DialysisDataLoader(cfg)
    data = loader.load_data(cfg.experiment.train_csv, is_training=True)

    static = data["static"]
    events = data["targets"]["event"]
    durations = data["targets"]["duration"]

    # Get feature names from config
    static_cols = [c for c in cfg.columns.static_cols if c in pd.read_csv(cfg.experiment.train_csv).columns]
    # Remove 透前体温 since it was removed
    static_cols = [c for c in static_cols if c != "透前体温"]

    print("\nCorrelation between static features and event occurrence:")
    print("%-20s | %-12s | %-12s" % ("Feature", "Corr(Event)", "Corr(Duration)"))
    print("-" * 50)

    for i, col in enumerate(static_cols):
        if i >= static.shape[1]:
            break
        feat = static[:, i]
        corr_event = np.corrcoef(feat, events)[0, 1]
        corr_dur = np.corrcoef(feat, durations)[0, 1]
        print("%-20s | %12.4f | %12.4f" % (col, corr_event, corr_dur))

    print("\nNote: Correlations < 0.1 indicate weak predictive power.")


if __name__ == "__main__":
    analyze_targets()
    analyze_predictive_power()
