# Main Manuscript Draft

## Title
Domain-Adaptive Deep Survival Modeling for Predicting Intradialytic Hypotension Under Cross-center Hemodynamic Shift

## Abstract
See `06_Abstract_Structured.md`.

## Introduction

Intradialytic hypotension (IDH) remains one of the most frequent and clinically consequential complications of maintenance hemodialysis. IDH is associated with symptoms, premature treatment interruption, inadequate fluid removal, myocardial stunning, cerebral hypoperfusion, and downstream hospitalization and mortality risk. Despite its clinical importance, bedside identification of sessions at imminent risk of hemodynamic instability remains difficult because IDH reflects a dynamic interaction among baseline cardiovascular vulnerability, interdialytic fluid accumulation, ultrafiltration intensity, and center-specific treatment practice.

A growing number of machine-learning studies have attempted to predict adverse intradialytic events, but most have focused on internally validated classification settings and have paid less attention to transportability across centers. This is a major limitation for dialysis prediction research. Hemodialysis populations are not exchangeable by default: blood pressure distributions, dry-weight management, dialysate prescriptions, vascular access patterns, and ultrafiltration strategies can differ substantially across institutions. A model that performs well in one center may therefore lose discrimination, calibration, or both when applied to another. In nephrology, this transportability problem is not a technical footnote; it is a central barrier to real-world clinical utility.

IDH prediction also has a strong time-to-event component that is often underused in existing work. The clinical question is not only whether hypotension will occur, but when it is likely to occur during the dialysis session, because the timing of hemodynamic deterioration determines the window for intensified monitoring or preventive adjustment. A survival-analysis framework is therefore more aligned with the clinical problem than a static binary classifier, particularly when early and later events may reflect partially different hemodynamic trajectories.

Domain adaptation provides a plausible methodological response to cross-center shift, but its value in dialysis prediction should be judged by more than discrimination alone. A clinically credible model should demonstrate external discrimination, interpretable risk structure, acceptable calibration, and at least preliminary evidence of decision usefulness. Otherwise, high rank-based performance may still fail to translate into clinically trustworthy risk communication or intervention support.

In this two-center retrospective study, we developed a domain-adaptive deep survival framework, CDAN-GSN, for prediction of IDH timing using a Shenyi development cohort and a Fuding external validation cohort. We hypothesized that substantial cross-center heterogeneity would be present in both baseline covariates and event-time structure; that domain-adaptive survival modeling would improve external discrimination relative to a zero-shot baseline; and that real case-level calibration and decision-curve analysis would provide a more honest assessment of the model's practical value in the external center. The aim of this work was therefore not merely to fit another dialysis risk model, but to test whether a transport-aware survival framework can remain useful under clinically meaningful cross-center hemodynamic shift.

## Methods

### Study Design and Data Sources
This study was a retrospective two-center prediction and external validation study based on routinely collected hemodialysis-session data. The Shenyi cohort was used as the source-domain development cohort, and the Fuding cohort was used as the target-domain external validation cohort. The unit of analysis was the dialysis session rather than the individual patient, because the clinical objective was to estimate session-level risk of intradialytic hypotension within a given treatment window.

The final analytic dataset included 291,828 dialysis sessions, comprising 216,604 sessions from Shenyi and 75,224 sessions from Fuding. Raw center-specific CSV data were processed through a four-stage pipeline consisting of basic feature engineering, historical mean construction, historical rate construction, and final feature merging. The processed model-ready files were `深医_final_data.csv` and `福鼎_final_data.csv`, stored under `data/processed`.

### Ethics and Reporting
This manuscript should be reported as a retrospective prediction and external validation study and aligned with TRIPOD-AI principles where applicable. Institutional review board approval, waiver status, and data-governance language should be added here once the formal institutional wording is confirmed. These details are intentionally left as manuscript placeholders rather than inferred without documentation.

### Outcome Definition
The primary outcome was intradialytic hypotension modeled as a time-to-event endpoint. Event timing was derived from intradialytic blood-pressure trajectories and stored as `et_min`, representing minutes from dialysis start to event occurrence. The event indicator was stored as `events`. Based on the available data-construction documentation, IDH labeling was anchored to pre-dialysis systolic blood pressure and defined using either a systolic blood-pressure decline of at least 30 mmHg or an absolute intradialytic systolic blood pressure below 90 mmHg. For descriptive visualization, event timing was also grouped into clinically interpretable stage windows.

### Predictor Construction and Preprocessing
Raw data processing followed a staged pipeline. First, center-specific list-format variables, including intradialytic blood pressure, pulse, ultrafiltration rate, ultrafiltration volume, venous pressure, arterial pressure, blood flow, dialysate temperature, and transmembrane pressure, were parsed into numeric lists and summarized where appropriate. Extreme values were truncated using quantile-based rules, and clinically derived features were added, including pulse pressure, mean arterial pressure, fluid overload, absolute ultrafiltration rate, and dialysis-age ratio.

Second, patient-level historical means were constructed by sorting sessions chronologically within each patient and calculating expanding means based only on prior sessions. Thus, for a given session, the corresponding historical mean feature reflected information available before that session rather than contemporaneous or future observations. Third, historical rate features were constructed to summarize previous event burden and previous distributions of hypotension-timing strata. The final model matrix included static demographic features, baseline hemodynamic variables, treatment-context variables, derived physiologic variables, and historical features. In the model-ready dataset, categorical variables were encoded as integer category codes, missing feature values were filled with zero, and all predictors were standardized using the source-cohort mean and standard deviation before application to the target cohort.

### Candidate Predictors Used for Modeling
Model construction used a predefined feature set combining demographic information, treatment-context variables, pre-dialysis physiologic state, derived hemodynamic features, and historical burden summaries. The core feature template included sex, dialysis-age ratio, fluid overload, historical averages of ultrafiltration and blood-pressure variables, anticoagulant type, dialysis modality, vascular-access type and location, dialysate calcium concentration, dialysate conductivity, pre-dialysis body-weight deviation from dry weight, respiratory rate, temperature, systolic blood pressure, diastolic blood pressure, arterial pressure, pulse pressure, and mean arterial pressure. In addition, all columns beginning with the prefix `history_` were appended dynamically to the final feature list, allowing the model to incorporate longitudinal burden summaries without manual re-specification.

### Data Splitting and Validation Strategy
The Shenyi cohort was used for source-domain model development. The Fuding cohort was split into three mutually exclusive subsets: a target adaptation pool, a target validation subset, and a held-out target test subset. In the implemented pipeline, the target adaptation ratio was 0.20, implying that 80% of the Fuding sessions were reserved for held-out testing before the adaptation pool was further divided into training and validation subsets. Stratified splitting was performed using the event indicator to preserve event prevalence across subsets, with a fixed random seed of 42. This design allowed source-only pretraining, target-aware adaptation, model selection on the validation subset, and final performance estimation on an unseen target test subset.

### Model Architecture
The proposed model, CDAN-GSN, was implemented as a domain-adaptive deep survival network with a KAN-based tokenizer and a gated representation module. Each feature was first mapped into token embeddings, and a transformer encoder was used to learn contextualized feature representations. A classification token embedding was passed to a hazard head to produce a session-level risk score. To support domain adaptation, predictors were partitioned into treatment-related features and non-treatment physiologic features. A gated mask was applied to the physiologic representation, and a gradient-reversal layer was used to train a domain classifier on the joint input of masked physiologic embeddings, treatment features, and a risk-conditioned hazard-derived term. This design was intended to align source and target representations statistically while preserving task relevance.

### Model Training Procedure
Training consisted of two phases. In phase 1, the model was pretrained on the full Shenyi source cohort using a weighted Cox partial-likelihood loss. In phase 2, source-initialized parameters were adapted using a domain-stratified training objective on both the Shenyi source loader and the target adaptation loader derived from the Fuding cohort. Domain adaptation proceeded in staged fine-tuning blocks (`head_only`, `partial_unfreeze`, and `full_finetune`) with early stopping at each stage based on target-validation C-index.

The final adaptation loss combined four terms: target-domain weighted Cox loss, replayed source-domain weighted Cox loss, domain-adversarial loss, and an L1 penalty on the gating mask. The adversarial coefficient was gradually increased within each training stage, and optimization was performed with AdamW. The main hyperparameters specified in the configuration file were: batch size 256, learning rate 0.001, adversarial loss weight 0.05, mask L1 weight 0.05, source pretraining epochs 18, and target fine-tuning epochs up to 20 in the final stage.

### Handling of Censoring and Weighting
Inverse probability of censoring weighting (IPCW) was used both in training and in parts of the evaluation workflow. IPCW weights were estimated from Kaplan-Meier fits of the censoring distribution and truncated at the 95th percentile to reduce instability from extreme weights. The weighted Cox loss sorted sessions by event time in descending order and incorporated IPCW into both the event contribution and risk-set accumulation term.

### Model Evaluation
The primary discrimination metric was Harrell's concordance index (C-index) in the external validation cohort, with 95% confidence intervals estimated by bootstrap resampling. Time-dependent area under the receiver-operating characteristic curve was additionally computed at clinically relevant time points when valid train-evaluation overlap allowed calculation. External discrimination of the final CDAN-GSN model was compared with that of a zero-shot baseline obtained from the source-pretrained model before domain adaptation.

To move beyond rank-based assessment, real case-level predictions were exported for the held-out external test set and used to generate calibration plots and decision curve analysis. Calibration was evaluated at 60 and 120 minutes by comparing predicted probabilities with observed event rates across predicted-risk bins. Clinical utility was assessed by decision curve analysis across a range of threshold probabilities relevant to intensified monitoring or preventive intervention decisions. Brier scores were also summarized for selected horizons.

### Reproducibility
Random seeds were fixed across Python, NumPy, and PyTorch components, and deterministic settings were enabled for cuDNN where applicable. The execution environment was recorded automatically through a `pip freeze` snapshot written to the experiment log directory. Core paths, data files, model hyperparameters, and training settings were stored in the project configuration file, supporting rerun traceability.

## Results
See `02_Results.md`.

## Discussion
See `03_Discussion.md`.

## Figure Legends
See `01_Figure_Legends.md`.
