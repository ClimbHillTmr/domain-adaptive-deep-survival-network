"""
Phase 4 – Statistical Tests
============================
1. DeLong test: paired AUC comparison at 60m and 120m horizons
   (CDAN-GSN vs CoxPH, using binary time-horizon labels)
2. Bootstrap permutation test: C-index comparison

Output: experiments/results/statistical_tests.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluate.metrics import concordance_index


# ---------------------------------------------------------------------------
# DeLong (1988) paired AUC test
# ---------------------------------------------------------------------------

def _structural_components(y_true, scores):
    """Compute structural components V10, V01 for DeLong variance."""
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    n1, n0 = len(pos_idx), len(neg_idx)
    scores_pos = scores[pos_idx]
    scores_neg = scores[neg_idx]

    # V10: for each positive case, fraction of negatives scored lower
    V10 = np.zeros(n1)
    for i, sp in enumerate(scores_pos):
        V10[i] = np.mean(scores_neg < sp) + 0.5 * np.mean(scores_neg == sp)

    # V01: for each negative case, fraction of positives scored higher
    V01 = np.zeros(n0)
    for j, sn in enumerate(scores_neg):
        V01[j] = np.mean(scores_pos > sn) + 0.5 * np.mean(scores_pos == sn)

    return V10, V01, n1, n0


def delong_variance(y_true, scores):
    V10, V01, n1, n0 = _structural_components(y_true, scores)
    auc = np.mean(V10)
    var = np.var(V10, ddof=1) / n1 + np.var(V01, ddof=1) / n0
    return auc, var


def delong_covariance(y_true, scores1, scores2):
    """Covariance between two AUCs on the same test set."""
    V10_1, V01_1, n1, n0 = _structural_components(y_true, scores1)
    V10_2, V01_2, _, _ = _structural_components(y_true, scores2)
    cov = (np.cov(V10_1, V10_2, ddof=1)[0, 1] / n1 +
           np.cov(V01_1, V01_2, ddof=1)[0, 1] / n0)
    return cov


def delong_test(y_true, scores1, scores2, label1="Model1", label2="Model2"):
    """Paired DeLong test. Returns dict with AUC, variance, z, p."""
    from scipy import stats
    auc1, var1 = delong_variance(y_true, scores1)
    auc2, var2 = delong_variance(y_true, scores2)
    cov = delong_covariance(y_true, scores1, scores2)
    diff_var = var1 + var2 - 2 * cov
    if diff_var <= 0:
        z, p = 0.0, 1.0
    else:
        z = (auc1 - auc2) / np.sqrt(diff_var)
        p = float(2 * stats.norm.sf(abs(z)))
    return {
        f"AUC_{label1}": float(auc1),
        f"AUC_{label2}": float(auc2),
        "AUC_diff": float(auc1 - auc2),
        "z": float(z),
        "p_value": p,
        "significant_0.05": p < 0.05,
    }


# ---------------------------------------------------------------------------
# Bootstrap permutation test for C-index
# ---------------------------------------------------------------------------

def bootstrap_cindex_pvalue(t_test, e_test, scores1, scores2, patient_ids, n_boot=2000, seed=42):
    """
    One-sided bootstrap permutation test: H0: C-index1 <= C-index2
    Returns: observed_diff, p_value
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
    # One-sided: p = proportion of bootstrap diffs <= 0 (under H1: model1 > model2)
    p = float(np.mean(diffs <= 0))
    return {
        "cindex_model1": float(obs1),
        "cindex_model2": float(obs2),
        "observed_diff": float(obs_diff),
        "p_value_one_sided": p,
        "significant_0.05": p < 0.05,
        "n_bootstrap": n_boot,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    from scipy import stats  # noqa: confirm available

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

    # Align on index (same test split, same order)
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

    results = {"delong_auc": {}, "cindex_bootstrap": {}}

    # DeLong at each horizon
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

        result = delong_test(y_true, cdan_probs, cox_probs, "CDAN-GSN", "CoxPH")
        results["delong_auc"][f"{horizon}m"] = result
        print(f"  DeLong {horizon}m: AUC_CDAN={result['AUC_CDAN-GSN']:.4f}, "
              f"AUC_Cox={result['AUC_CoxPH']:.4f}, "
              f"p={result['p_value']:.4f} {'*' if result['significant_0.05'] else ''}")

    # Bootstrap C-index test
    print("\n  Bootstrap permutation C-index test (CDAN-GSN vs CoxPH)...")
    if patient_ids is not None:
        cindex_result = bootstrap_cindex_pvalue(
            t_test, e_test, cdan_risk, cox_risk,
            patient_ids=patient_ids, n_boot=2000, seed=42,
        )
    else:
        cindex_result = bootstrap_cindex_pvalue(
            t_test, e_test, cdan_risk, cox_risk,
            patient_ids=np.arange(len(t_test)).astype(str), n_boot=2000, seed=42,
        )
    results["cindex_bootstrap"]["CDAN-GSN_vs_CoxPH"] = cindex_result
    print(f"  C-index CDAN={cindex_result['cindex_model1']:.4f}, "
          f"Cox={cindex_result['cindex_model2']:.4f}, "
          f"diff={cindex_result['observed_diff']:+.4f}, "
          f"p={cindex_result['p_value_one_sided']:.4f} "
          f"{'*' if cindex_result['significant_0.05'] else '(ns)'}")

    out_path = Path("experiments/results/statistical_tests.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nStatistical tests saved to {out_path}")


if __name__ == "__main__":
    main()
