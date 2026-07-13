# Structured Abstract

## Background

Intradialytic hypotension (IDH) is the most common complication of maintenance hemodialysis and is independently associated with adverse cardiovascular outcomes. Survival models trained on single-centre data may generalise poorly when applied to an external centre with a different patient case-mix and event-rate distribution. Domain adaptation offers a principled framework for updating pre-trained models using limited labelled data from a new centre, yet its value relative to both a zero-shot source model and a locally re-fitted classical survival model has not been rigorously quantified in this setting.

## Methods

We conducted a two-centre retrospective prediction study using 216,604 dialysis sessions from 1,637 patients at Shenyi Hospital (Shenzhen) as the source cohort and 75,224 sessions from 430 patients at Fuding Hospital as the external target cohort. A 23-feature session-level feature set was constructed after removing six low-variance treatment categoricals. The target cohort was split at the patient level into an adaptation-training set (approximately 146 patients), an adaptation-validation set (approximately 26 patients), and a held-out test set (approximately 258 patients, more than 45,000 sessions). Three models were evaluated on the held-out test set: a zero-shot model (source-trained, no target updating), a locally re-fitted Cox proportional hazards model (CoxPH, 23 features), and our domain-adaptive deep survival network with Gated Survival Network architecture (CDAN-GSN). Discrimination was assessed by Harrell C-index with bootstrap 95% confidence intervals and by DeLong time-specific AUC at 60 and 120 minutes. The C-index comparison between CDAN-GSN and CoxPH used a one-sided bootstrap permutation test (2,000 replicates). Calibration was assessed by Brier score and decision-curve analysis.

## Results

The target cohort exhibited a three-fold higher event rate than the source cohort (35.8% vs. 11.0%), indicating substantial cross-centre distribution shift. On the held-out test set, CDAN-GSN achieved a C-index of 0.841 (95% CI, 0.831–0.852), significantly outperforming the zero-shot baseline (C-index 0.832, 95% CI, 0.821–0.843; Δ = +0.009) and the locally re-fitted CoxPH model (C-index 0.838, 95% CI, 0.827–0.847; Δ = +0.004, p = 0.004). At 120 minutes, CDAN-GSN showed higher AUC than CoxPH (0.868 vs. 0.865; DeLong p < 0.001). Calibration analysis revealed systematic underestimation of absolute risk at both horizons (observed 14.5% vs. predicted 10.9% at 60 minutes; Brier score 0.098), though decision-curve analysis confirmed clinically useful net benefit across a range of threshold probabilities.

## Conclusions

Domain-adaptive survival modelling with limited target-centre updating achieves significantly better discrimination than both a zero-shot source model and a locally re-fitted CoxPH benchmark in an external hemodialysis centre. Systematic underestimation of absolute event probabilities observed in calibration analyses indicates that prospective deployment would require explicit recalibration at the target centre. These findings support the potential of transfer-learning-based survival models for cross-centre generalisation while highlighting calibration as a necessary next step before clinical use.
