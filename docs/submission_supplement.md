# Supplementary Methods and Evidence Definitions

Status: evidence-linked draft; generated from registered artifacts. This supplement does not add experimental results and must be rebuilt after accepted confirmatory server runs or locked subgroup prespecification.

## S1. Evidence contracts and reproducibility boundary

Historical v1 results remain the evidence base for the manuscript's current target-center-updating conclusions. The exact processed v1 bytes are unavailable in the current workspace, so the historical dataset hashes cannot be reproduced from the present files; this limitation is permanent and must remain disclosed. The separately versioned confirmatory-v2 contract is reserved for new architecture-matched server experiments and must not be substituted into v1 claims.

| Contract | Cohort | Sessions | Patients | IDH prevalence | IH prevalence | Evidence status |
| --- | --- | --- | --- | --- | --- | --- |
| dual_binary_hemodynamic_v1_frozen_20260719 | source | 211452 | 1628 | 0.1451582392221402 | 0.2836482984317954 | frozen_hash_only_current_file_mismatch |
| dual_binary_hemodynamic_v1_frozen_20260719 | target | 74947 | 430 | 0.385352315636383 | 0.1195778350034024 | frozen_hash_only_current_file_mismatch |

The historical registry contains 8 v1 runs; 0 accepted confirmatory-v2 runs are currently registered. Every manuscript result that depends on model output is linked to a registered run ID, hashed configuration, checkpoint, prediction file, evaluation file, and split manifest.

## S2. Historical main-model hyperparameters

| Item | Registered value |
| --- | --- |
| Architecture | Two encoders joined before a linear prediction head |
| Encoder hidden dimensions | 64, 32 |
| Dropout | 0.2 |
| Physiology/history predictors | 20 |
| Treatment-context predictors | 4 |
| IDH aligned branch | physiology/history |
| IH aligned branch | treatment context |
| CORAL weight | 0.01 |
| Pretraining learning rate | 0.001 |
| Updating learning rate | 0.0001 |
| Pretraining/update epoch ceilings | 40 / 20 |
| Batch size | 256 |
| Early-stopping patience | 5 |
| Initialization seeds | 42, 7, 13, 99, 2024 |
| Split seed | 20260715 |
| Patient-cluster bootstrap | 1000 replicates; seed 20260715 |

Numerical medians, means, standard deviations, and categorical mappings were fitted on source-training patients only. Numerical missing values were replaced with source-training medians; unknown target categories received a reserved code. Platt calibration used the corresponding validation patients. The held-out target patients were not used for model updating, selection, threshold selection, or calibration.

## S3. Architecture-matched confirmatory protocol

| Item | Locked value or state |
| --- | --- |
| Protocol | architecture_matched_ablation_v1 |
| Data contract | dual_binary_hemodynamic_confirmatory_v2 |
| Execution context | server_confirmatory |
| Models | dual no alignment; dual global CORAL; dual random-feature CORAL; dual outcome-specific CORAL; single encoder |
| Seeds | 42, 7, 13, 99, 2024 |
| Architecture/hidden dimensions | dual_branch; [64, 32] |
| Dropout / batch size | 0.2 / 256 |
| Learning rates | pretraining 0.001; updating 0.0001 |
| Epoch ceilings / patience | 40 / 20 / 5 |
| CORAL weight | 0.01 |
| Bootstrap | 1000 patient-cluster replicates; seed 20260715 |
| Current evidence state | Pending compute-server execution and acceptance; no v2 result is used below |

All five strategies use the same data contract, patient split, preprocessing, optimization budget, calibration procedure, and seeds. Confirmatory tables and Figure 2 remain empty until all required server runs pass the read-only acceptance audit and enter the central run registry. The prespecified paired contrast is outcome-specific CORAL minus each of the four architecture-matched comparators. Positive differences favor outcome-specific CORAL for ROC AUC and PR AUC; negative differences favor it for Brier score and 10-quantile-bin ECE.

## S4. Statistical resampling and clinical utility

The primary uncertainty unit is the patient. Patients are sampled with replacement and all their sessions are retained within a bootstrap replicate. Percentile 95% intervals use 1000 replicates with bootstrap seed 20260715. Within-run AUC contrasts are paired because competing prediction vectors are evaluated on the same sampled patients. Five initialization seeds are summarized descriptively by their individual values, mean, and sample standard deviation; across-seed ranges are not confidence intervals.

Clinical-utility summaries use registered held-out predictions only and do not fit or update a predictive model. Calibration intercept, calibration slope, expected calibration error, Brier score, and fixed-threshold alert metrics use 1000 patient-cluster replicates. Alert thresholds are validation-derived Youden thresholds, not clinically prespecified deployment thresholds. Decision curves remain threshold-grid sensitivity analyses until nephrology collaborators lock a clinically relevant threshold interval.

## S5. Predictor definitions and preprocessing

The registered feature manifest contains 57 candidate or derived fields; 24 are included predictors. Availability is defined relative to the session-start prediction time. Rows marked `end` or `outcome` are excluded to prevent current-session future information from entering predictors.

| Feature | Category | Branch | Status | Availability | Preprocessing | Rationale |
| --- | --- | --- | --- | --- | --- | --- |
| 性别 | demographic | physiology_history | included | start | source-train median imputation; source-train z-score | baseline demographic |
| 透析龄占比 | demographic | physiology_history | included | start | source-train median imputation; source-train z-score | baseline dialysis vintage ratio |
| 超负荷 | pre_dialysis | treatment_context | included | start | source-train median imputation; source-train z-score | pre-dialysis volume status |
| 历史平均超滤量MAX | history | treatment_context | included | start | source-train median imputation; source-train z-score | prior-session aggregate only |
| 历史平均透前体重 | history | physiology_history | included | start | source-train median imputation; source-train z-score | prior-session aggregate only |
| 历史平均透前收缩压 | history | physiology_history | included | start | source-train median imputation; source-train z-score | prior-session aggregate only |
| 历史平均透前舒张压 | history | physiology_history | included | start | source-train median imputation; source-train z-score | prior-session aggregate only |
| 历史平均透中低血压_计算 | history | physiology_history | included | start | source-train median imputation; source-train z-score | prior-session event burden only |
| 抗凝剂类型_code | treatment_context | excluded | excluded | start | excluded | excluded from the frozen 24-predictor binary model; the source allowlist contains a legacy survival-study rationale that is not used as binary-study evidence |
| 透析方式_code | treatment_context | excluded | excluded | start | excluded | excluded from the frozen 24-predictor binary model; the source allowlist contains a legacy survival-study rationale that is not used as binary-study evidence |
| 瘘管类型_code | treatment_context | excluded | excluded | start | excluded | excluded from the frozen 24-predictor binary model; the source allowlist contains a legacy survival-study rationale that is not used as binary-study evidence |
| 瘘管位置_code | treatment_context | excluded | excluded | start | excluded | excluded from the frozen 24-predictor binary model; the source allowlist contains a legacy survival-study rationale that is not used as binary-study evidence |
| 透析液钙浓度 | treatment_context | excluded | excluded | start | excluded | excluded from the frozen 24-predictor binary model; the source allowlist contains a legacy survival-study rationale that is not used as binary-study evidence |
| 透析液电导率 | treatment_context | excluded | excluded | start | excluded | excluded from the frozen 24-predictor binary model; the source allowlist contains a legacy survival-study rationale that is not used as binary-study evidence |
| 透前体重-干体重 | pre_dialysis | treatment_context | included | start | source-train median imputation; source-train z-score | pre-dialysis status |
| 透前呼吸频率 | pre_dialysis | physiology_history | included | start | source-train median imputation; source-train z-score | pre-dialysis vital sign |
| 透前体温 | pre_dialysis | physiology_history | included | start | source-train median imputation; source-train z-score | pre-dialysis vital sign |
| 透前收缩压 | pre_dialysis | physiology_history | included | start | source-train median imputation; source-train z-score | pre-dialysis vital sign |
| 透前舒张压 | pre_dialysis | physiology_history | included | start | source-train median imputation; source-train z-score | pre-dialysis vital sign |
| 透前动脉压 | pre_dialysis | physiology_history | included | start | source-train median imputation; source-train z-score | pre-dialysis vital sign |
| 脉压差 | pre_dialysis | physiology_history | included | start | source-train median imputation; source-train z-score | derived from pre-dialysis BP |
| 平均动脉压 | pre_dialysis | physiology_history | included | start | source-train median imputation; source-train z-score | derived from pre-dialysis BP |
| history_IDH_rate | history | physiology_history | included | start | source-train median imputation; source-train z-score | prior-session IDH burden |
| history_HBP_rate | history | physiology_history | included | start | source-train median imputation; source-train z-score | prior-session IH burden |
| history_LBP_times_0_rate | history | physiology_history | included | start | source-train median imputation; source-train z-score | prior-session no-IDH burden legacy name |
| history_LBP_times_1_rate | history | physiology_history | included | start | source-train median imputation; source-train z-score | prior-session timing-bin burden |
| history_LBP_times_2_rate | history | physiology_history | included | start | source-train median imputation; source-train z-score | prior-session timing-bin burden |
| history_LBP_times_3_rate | history | physiology_history | included | start | source-train median imputation; source-train z-score | prior-session timing-bin burden |
| history_LBP_times_4_rate | history | physiology_history | included | start | source-train median imputation; source-train z-score | prior-session timing-bin burden |
| 历史平均超滤率_mean | history | treatment_context | included | start | source-train median imputation; source-train z-score | prior-session aggregate only |
| 透析中收缩压_mean | current_session_future | excluded | excluded | end | excluded | current-session intradialytic summary — future info |
| 透析中收缩压_std | current_session_future | excluded | excluded | end | excluded | current-session intradialytic summary — future info |
| 透析中舒张压_mean | current_session_future | excluded | excluded | end | excluded | current-session intradialytic summary — future info |
| 透析中舒张压_std | current_session_future | excluded | excluded | end | excluded | current-session intradialytic summary — future info |
| 透析中脉搏_mean | current_session_future | excluded | excluded | end | excluded | current-session intradialytic summary — future info |
| 透析中脉搏_std | current_session_future | excluded | excluded | end | excluded | current-session intradialytic summary — future info |
| 动脉压_mean | current_session_future | excluded | excluded | end | excluded | current-session intradialytic summary — future info |
| 动脉压_std | current_session_future | excluded | excluded | end | excluded | current-session intradialytic summary — future info |
| 原始动脉压_mean | current_session_future | excluded | excluded | end | excluded | current-session intradialytic summary — future info |
| 原始动脉压_std | current_session_future | excluded | excluded | end | excluded | current-session intradialytic summary — future info |
| 静脉压_mean | current_session_future | excluded | excluded | end | excluded | current-session intradialytic summary — future info |
| 静脉压_std | current_session_future | excluded | excluded | end | excluded | current-session intradialytic summary — future info |
| 血流速_mean | current_session_future | excluded | excluded | end | excluded | current-session intradialytic summary — future info |
| 血流速_std | current_session_future | excluded | excluded | end | excluded | current-session intradialytic summary — future info |
| 透析液温度_mean | current_session_future | excluded | excluded | end | excluded | current-session intradialytic summary — future info |
| 透析液温度_std | current_session_future | excluded | excluded | end | excluded | current-session intradialytic summary — future info |
| 跨膜压_mean | current_session_future | excluded | excluded | end | excluded | current-session intradialytic summary — future info |
| 跨膜压_std | current_session_future | excluded | excluded | end | excluded | current-session intradialytic summary — future info |
| 超滤率_mean | current_session_future | excluded | excluded | end | excluded | current-session intradialytic summary — future info |
| 超滤率_std | current_session_future | excluded | excluded | end | excluded | current-session intradialytic summary — future info |
| 实际透析时长 | current_session_future | excluded | excluded | end | excluded | known only at session end — future info |
| et_min | outcome | excluded | excluded | end | excluded | outcome variable — never a predictor |
| events | outcome | excluded | excluded | end | excluded | outcome variable — never a predictor |
| idh_time_min | outcome | excluded | excluded | end | excluded | outcome variable — never a predictor |
| idh_event | outcome | excluded | excluded | end | excluded | outcome variable — never a predictor |
| ih_time_min | outcome | excluded | excluded | end | excluded | outcome variable — never a predictor |
| ih_event | outcome | excluded | excluded | end | excluded | outcome variable — never a predictor |

## S6. Patient-cluster subgroup protocol

Protocol status: `blocked_pending_clinical_and_statistical_prespecification`. Not locked; definitions, approvers, no-test-access attestation, and minimum patient count remain required.

| Endpoint | Subgroup | Candidate feature | Locked definition | Minimum patients per level |
| --- | --- | --- | --- | --- |
| IDH | baseline_blood_pressure | 透前收缩压 | REQUIRED_CLINICAL_PRESPECIFICATION | REQUIRED_CLINICAL_AND_STATISTICAL_PRESPECIFICATION |
| IDH | prior_instability_history | history_IDH_rate | REQUIRED_CLINICAL_PRESPECIFICATION | REQUIRED_CLINICAL_AND_STATISTICAL_PRESPECIFICATION |
| IDH | ultrafiltration_intensity | 历史平均超滤率_mean | REQUIRED_CLINICAL_PRESPECIFICATION | REQUIRED_CLINICAL_AND_STATISTICAL_PRESPECIFICATION |
| IH | treatment_intensity | 历史平均超滤量MAX | REQUIRED_CLINICAL_PRESPECIFICATION | REQUIRED_CLINICAL_AND_STATISTICAL_PRESPECIFICATION |
| IH | volume_overload | 超负荷 | REQUIRED_CLINICAL_PRESPECIFICATION | REQUIRED_CLINICAL_AND_STATISTICAL_PRESPECIFICATION |
| IH | hypertension_history | history_HBP_rate | REQUIRED_CLINICAL_PRESPECIFICATION | REQUIRED_CLINICAL_AND_STATISTICAL_PRESPECIFICATION |

The planned comparison is updated MLP minus source MLP. Each eligible level will report subgroup AUC, paired delta AUC, percentile patient-cluster interval, and a two-sided patient-cluster bootstrap interaction P value. Previous session-level subgroup estimates and P values are discarded. No target-test subgroup features may be inspected before the definitions, minimum patient count, approver roles, rationale, and no-test-access attestation are locked.

## S7. Artifact locations

- Dataset manifest: `experiments/evidence_registry/dataset_manifest.csv`
- Feature manifest: `experiments/evidence_registry/feature_manifest.csv`
- Split manifest: `experiments/evidence_registry/split_manifest.csv`
- Run registry: `experiments/evidence_registry/run_registry.csv`
- Confirmatory configuration: `conf/confirmatory_ablation.yaml`
- Subgroup protocol: `conf/subgroup_protocol.yaml`
- Clinical utility provenance: `clinical_utility_report/provenance.json`
- Manuscript claim registry: `experiments/evidence_registry/manuscript_claim_registry.csv`
- Supplementary Figure S1 provenance: `figures/submission_staging/SupplementaryFigureS1_provenance.json`
