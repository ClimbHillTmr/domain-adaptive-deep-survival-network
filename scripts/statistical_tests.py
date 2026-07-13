"""
Phase 4 – Statistical Tests
============================
1. Paired patient-cluster bootstrap comparison of horizon-eligible AUCs.
2. Paired patient-cluster bootstrap comparison of C-index.

Output: experiments/results/statistical_tests.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluate.metrics import _binary_auc, concordance_index


# ---------------------------------------------------------------------------
# Paired patient-cluster bootstrap comparisons
# ---------------------------------------------------------------------------

def paired_cluster_bootstrap(t_test, e_test, scores1, scores2, patient_ids, n_boot=2000, seed=42):
    """
    Patient-cluster paired bootstrap for the C-index difference.
    """
    rng = np.random.default_rng(seed)
    obs1 = concordance_index(t_test, scores1, e_test)
    obs2 = concordance_index(t_test, scores2, e_test)
    obs_diff = obs1 - obs2

    patients = np.unique(patient_ids)
    diffs = []
    for _ in range(n_boot):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        idx = np.concatenate([np.flatnonzero(patient_ids == p) for p in sampled])
        t_b, e_b = t_test[idx], e_test[idx]
        c1 = concordance_index(t_b, scores1[idx], e_b)
        c2 = concordance_index(t_b, scores2[idx], e_b)
        diffs.append(c1 - c2)
    diffs = np.array(diffs)
    ci_lower, ci_upper = np.percentile(diffs, [2.5, 97.5])
    return {
        "cindex_model1": float(obs1),
        "cindex_model2": float(obs2),
        "observed_diff": float(obs_diff),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "excludes_zero": bool(ci_lower > 0 or ci_upper < 0),
        "n_bootstrap": n_boot,
    }


def paired_cluster_auc_delta(y_true, scores1, scores2, patient_ids, n_boot=2000, seed=42):
    """Paired patient-cluster bootstrap for a censoring-eligible binary horizon."""
    rng = np.random.default_rng(seed)
    patients = np.unique(patient_ids)
    observed = _binary_auc(y_true, scores1) - _binary_auc(y_true, scores2)
    diffs = []
    for _ in range(n_boot):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        indices = np.concatenate([np.flatnonzero(patient_ids == patient) for patient in sampled])
        auc1 = _binary_auc(y_true[indices], scores1[indices])
        auc2 = _binary_auc(y_true[indices], scores2[indices])
        if auc1 is not None and auc2 is not None:
            diffs.append(auc1 - auc2)
    lower, upper = np.percentile(diffs, [2.5, 97.5])
    return {"observed_diff": float(observed), "ci_lower": float(lower), "ci_upper": float(upper), "excludes_zero": bool(lower > 0 or upper < 0), "n_bootstrap": n_boot}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Load prediction files
    cdan_pred_path = Path("experiments/results/real_test_predictions.csv")
    cox_pred_path = Path("experiments/results/local_cox_test_predictions.csv")

    if not cdan_pred_path.exists():
        print(f"Missing: {cdan_pred_path}. Run export_real_predictions.py first.")
        sys.exit(1)
    if not cox_pred_path.exists():
        print(f"Missing: {cox_pred_path}. Run run_local_cox_update.py first.")
        sys.exit(1)

    cdan_df = pd.read_csv(cdan_pred_path)
    cox_df = pd.read_csv(cox_pred_path)

    # Align on a stable row order after the consistency gate has confirmed one run.
    assert len(cdan_df) == len(cox_df), (
        f"Row mismatch: CDAN {len(cdan_df)} vs CoxPH {len(cox_df)}. "
        "Both must be evaluated on the same test split."
    )

    # Patient IDs for bootstrap
    patient_col = "患者id"
    patient_ids = cdan_df[patient_col].astype(str).to_numpy() if patient_col in cdan_df.columns else None

    t_test = cdan_df["et_min"].to_numpy()
    e_test = cdan_df["events"].astype(int).to_numpy()
    cdan_risk = cdan_df["risk_score"].to_numpy()
    cox_risk = cox_df["risk_score"].to_numpy()
    if patient_ids is None:
        patient_ids = np.arange(len(t_test)).astype(str)

    results = {"paired_cluster_auc": {}, "paired_cluster_cindex": {}}

    # Censored observations before a horizon are not binary controls.
    for horizon in [30, 60, 120]:
        y_true_col = f"event_by_{horizon}m"
        cdan_prob_col = f"event_prob_{horizon}m"
        cox_prob_col = f"event_prob_{horizon}m"

        if y_true_col not in cdan_df.columns or cdan_prob_col not in cdan_df.columns:
            print(f"  Skipping {horizon}m: columns not found in CDAN predictions.")
            continue
        if cox_prob_col not in cox_df.columns:
            # Fall back to risk score for Cox
            print(f"  Note: using risk_score for Cox at {horizon}m (no prob column).")
            cox_probs = cox_risk
        else:
            cox_probs = cox_df[cox_prob_col].to_numpy()

        y_true = cdan_df[y_true_col].astype(int).to_numpy()
        cdan_probs = cdan_df[cdan_prob_col].to_numpy()

        eligible = (e_test == 1) | (t_test > horizon)
        result = paired_cluster_auc_delta(
            y_true[eligible], cdan_probs[eligible], cox_probs[eligible], patient_ids[eligible], n_boot=2000, seed=42
        )
        results["paired_cluster_auc"][f"{horizon}m"] = result

    # Bootstrap C-index test
    print("\n  Paired patient-cluster bootstrap C-index comparison...")
    cindex_result = paired_cluster_bootstrap(
        t_test, e_test, cdan_risk, cox_risk, patient_ids=patient_ids, n_boot=2000, seed=42
    )
    results["paired_cluster_cindex"]["CDAN-GSN_vs_CoxPH"] = cindex_result

    out_path = Path("experiments/results/statistical_tests.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nStatistical tests saved to {out_path}")


if __name__ == "__main__":
    main()
