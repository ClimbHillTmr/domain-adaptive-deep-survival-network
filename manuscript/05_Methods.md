# Methods

## Study Design and Data Sources
This study was a retrospective two-center prediction and target-center updating study based on routinely collected hemodialysis-session data. The Shenyi cohort was used as the source development cohort, and the Fuding cohort was used as the target-center cohort for limited labeled updating, validation, and held-out testing. The unit of prediction was the dialysis session, whereas the target-center split was performed at the patient level to prevent within-patient leakage across updating, validation, and test subsets.

The final analytic dataset included 291,828 dialysis sessions, comprising 216,604 sessions from Shenyi and 75,224 sessions from Fuding. Processed model-ready files were `深医_final_data.csv` and `福鼎_final_data.csv`, stored under `data/processed`.

## Ethics and Reporting
This manuscript should be reported as a retrospective prediction and external validation study aligned with TRIPOD-AI principles where applicable. Institutional review board approval, waiver status, and data-governance language should be inserted only after the formal institutional wording is confirmed.

## Outcome Definition
The primary outcome was intradialytic hypotension modeled as a time-to-event endpoint. Event timing was stored as `et_min`, representing minutes from dialysis start to event occurrence, and the event indicator was stored as `events`.

## Predictor Construction and Preprocessing
Raw data processing followed a staged pipeline including center-specific feature parsing, derived hemodynamic feature construction, patient-level historical summary generation, and final feature merging. Historical mean and rate features were constructed only from prior sessions after chronological sorting within patient, such that the first audited session for each sampled patient had zero-valued history features.

Patients were split before model preprocessing. Categorical encodings, median imputation, and scaling parameters were fit using source-training patients only and then applied unchanged to source validation and target cohorts; unseen target-only categories were assigned to an explicit unknown level.

## Candidate Predictors Used for Modeling
Model construction used a predefined prediction-time feature allowlist rather than dynamically appending every column with a `history_` prefix. The frozen allowlist was stored in `experiments/audit/feature_allowlist.csv` and included demographic variables, treatment-context variables, pre-dialysis physiologic features, derived hemodynamic measures, and audited historical burden summaries judged to be available at prediction time. Current-session intradialytic summary columns and outcome columns were explicitly excluded by the allowlist.

## Data Splitting and Validation Strategy
The Shenyi cohort was split at the patient level, with a 10% source validation holdout used only for source-pretraining early stopping. The Fuding cohort was split into mutually exclusive patient-level subsets for target-center updating, target validation, and held-out testing using `target_adapt_ratio = 0.40` and `target_val_ratio = 0.15` within the adaptation pool. Exact analytic sample sizes are generated from the run-specific split audit.

## Model Architecture
The proposed model was implemented as a domain-stratified deep survival network with a KAN-based tokenizer and a gated representation module. A transformer encoder learned contextualized feature embeddings, and a hazard head produced a session-level risk score. Treatment-context features and non-treatment physiologic features were separated to support statistically guided representation alignment without implying causal identification.

## Model Training Procedure
Training consisted of source-cohort pretraining followed by target-center labeled updating. This second phase should be interpreted as local target-center updating of a survival model rather than as a purely unsupervised domain-adaptation exercise. Updating used the labeled target training subset together with replayed source batches and target-validation early stopping.

## Handling of Censoring and Weighting
Inverse probability of censoring weighting (IPCW) was used in training and evaluation workflows. IPCW weights were estimated from Kaplan-Meier fits of the censoring distribution and truncated at the 95th percentile to reduce instability from extreme weights.

## Model Evaluation
The primary discrimination metric was Harrell's concordance index in the held-out target test cohort, with patient-cluster bootstrap confidence intervals. Fixed-horizon calibration, Brier scores, and decision-curve analyses exclude observations censored before the relevant horizon. Model differences are reported with paired patient-cluster bootstrap confidence intervals. All numerical claims are tied to one run-specific prediction export and evidence manifest.

## Reproducibility
Random seeds were fixed across Python, NumPy, and PyTorch components, and deterministic settings were enabled for cuDNN where applicable. Submission-facing reruns were executed through a locked-run pipeline that snapshots the input data hashes, configuration file, feature allowlist, audit artifacts, environment snapshot, and generated outputs under a unique run identifier.
