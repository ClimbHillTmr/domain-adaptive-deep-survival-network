# Structured Abstract

## Background

Intradialytic hypotension (IDH) is the most common complication of maintenance hemodialysis and is independently associated with adverse cardiovascular outcomes. Survival models trained on single-centre data may generalise poorly when applied to an external centre with a different patient case-mix and event-rate distribution. Domain adaptation offers a principled framework for updating pre-trained models using limited labelled data from a new centre, yet its value relative to both a zero-shot source model and a locally re-fitted classical survival model has not been rigorously quantified in this setting.

## Methods

We conducted a two-centre retrospective prediction study using 216,604 dialysis sessions from 1,637 patients at Shenyi Hospital and 75,224 sessions from 430 patients at Fuding Hospital. Patients were split before preprocessing; source-training patients alone defined categorical encoding, imputation, and scaling. The target cohort was split at the patient level into updating, validation, and held-out test subsets. The planned comparison includes a source-only model, locally updated CoxPH, staged target-centre updating, and prespecified component ablations. Discrimination and model differences will use patient-cluster bootstrap methods; fixed-horizon calibration and decision analysis will exclude observations censored before each horizon.

## Results

Historical performance values were withdrawn before the unified rerun because they were assembled from inconsistent split, prediction, and comparison artifacts. Results will be inserted only when one locked run regenerates the audit, held-out predictions, paired comparisons, calibration, and decision analyses under the current code.

## Conclusions

This study evaluates whether locally updated survival modelling can retain useful discrimination under cross-centre hemodynamic shift. Any conclusion on comparative performance, calibration, or clinical utility remains contingent on the unified locked rerun and must remain conservative until then.
