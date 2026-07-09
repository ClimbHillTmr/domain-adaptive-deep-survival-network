# Results

## Study Population and Final Analytic Cohorts
A total of 291,828 dialysis sessions were included in the final analysis, comprising 216,604 sessions from the Shenyi development cohort and 75,224 sessions from the Fuding external validation cohort. The Fuding cohort contained 430 unique patients and was split at the patient level into a target updating subset (68 patients; 12,722 sessions), a target validation subset (18 patients; 3,154 sessions), and a held-out test subset (344 patients; 59,302 sessions). Patient overlap across the three target subsets was 0 for all pairwise comparisons (Fig. 1).

## Cross-center Baseline and Event-time Heterogeneity
Marked differences were observed between the Shenyi and Fuding cohorts in both baseline hemodynamics and event-time structure (Fig. 2). These cross-center shifts support the clinical need for a locally updated survival model rather than a source-only transport assumption.

## External Validation Performance
On held-out external testing, the locally updated survival model outperformed the zero-shot baseline in discrimination (Fig. 3). The C-index improved from 0.805 (95% CI, 0.791-0.817) for the source-only baseline to 0.830 (95% CI, 0.818-0.840) after target-center labeled updating.

## Calibration and Decision-curve Evidence
Real case-level predictions were exported for the held-out external test cohort (59,302 sessions). At 60 minutes, the observed event rate was 14.65% and the mean predicted probability was 12.83%, with a Brier score of 0.1002. At 120 minutes, the observed event rate was 25.19% and the mean predicted probability was 22.28%, with a Brier score of 0.1330.
The calibration plots and decision-curve panels generated from these held-out predictions show that the model preserves clinically useful risk ordering while requiring explicit acknowledgment of residual underestimation in the external center (Fig. 4).

## Supplementary Target Subgroup Burden
Observed IDH burden across major held-out target subgroups is provided as a supplementary figure rather than a main-text claim of subgroup-specific model superiority (Fig. S1).
