# Results

## Study Population and Final Analytic Cohorts
A total of 291,828 dialysis sessions were included in the final analysis, comprising 216,604 sessions from the Shenyi development cohort and 75,224 sessions from the Fuding external validation cohort. The Fuding cohort contained 430 unique patients and was split at the patient level into a target updating subset (68 patients; 12,734 sessions), a target validation subset (18 patients; 3,156 sessions), and a held-out test subset (344 patients; 59,334 sessions). Patient overlap across the three target subsets was 0 for all pairwise comparisons (Fig. 1).

## Cross-center Baseline and Event-time Heterogeneity
Marked differences were observed between the Shenyi and Fuding cohorts in both baseline hemodynamics and event-time structure (Fig. 2). These cross-center shifts support the clinical need for a locally updated survival model rather than a source-only transport assumption.

## External Validation Performance
The previous performance values have been withdrawn because the existing result files were not yet consistent with the latest audit-verified split and allowlist state. This section must remain numeric-free until a locked rerun produces a new `evaluation_results.json` linked to the current audit artifacts.

## Calibration and Decision-curve Evidence
Calibration, Brier score, and decision-curve statements are temporarily withheld. These analyses should only be restored after `real_test_predictions.csv`, `calibration_dca_metrics.json`, and the held-out test sample size are all regenerated under the same locked run.

## Supplementary Target Subgroup Burden
Target subgroup burden is reserved for supplementary presentation and should not re-enter the main text unless it is rebuilt from the current held-out predictions.
