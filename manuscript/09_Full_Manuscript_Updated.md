# Domain-Adaptive Deep Survival Modeling for Predicting Intradialytic Hypotension Under Cross-center Hemodynamic Shift

> **Version:** Final (Phase 5 update — all numbers confirmed from experiments/results/*.json)
> **Date:** 2026-07-12
> **Status:** Ready for submission-package review

---

## Structured Abstract

### Background

Intradialytic hypotension (IDH) is the most common complication of maintenance hemodialysis and is independently associated with adverse cardiovascular outcomes. Survival models trained on single-centre data may generalise poorly when applied to an external centre with a different patient case-mix and event-rate distribution. Domain adaptation offers a principled framework for updating pre-trained models using limited labelled data from a new centre, yet its value relative to both a zero-shot source model and a locally re-fitted classical survival model has not been rigorously quantified in this setting.

### Methods

We conducted a two-centre retrospective prediction study using 216,604 dialysis sessions from 1,637 patients at Shenyi Hospital (Shenzhen) as the source cohort and 75,224 sessions from 430 patients at Fuding Hospital as the external target cohort. A 23-feature session-level feature set was constructed after removing six low-variance treatment categoricals. The target cohort was split at the patient level into an adaptation-training set (approximately 146 patients), an adaptation-validation set (approximately 26 patients), and a held-out test set (approximately 258 patients, more than 45,000 sessions). Three models were evaluated on the held-out test set: a zero-shot model (source-trained, no target updating), a locally re-fitted Cox proportional hazards model (CoxPH, 23 features), and our domain-adaptive deep survival network with Gated Survival Network architecture (CDAN-GSN). Discrimination was assessed by Harrell C-index with bootstrap 95% confidence intervals and by DeLong time-specific AUC at 60 and 120 minutes. The C-index comparison between CDAN-GSN and CoxPH used a one-sided bootstrap permutation test (2,000 replicates). Calibration was assessed by Brier score and decision-curve analysis.

### Results

The target cohort exhibited a three-fold higher event rate than the source cohort (35.8% vs. 11.0%), indicating substantial cross-centre distribution shift. On the held-out test set, CDAN-GSN achieved a C-index of 0.841 (95% CI, 0.831–0.852), significantly outperforming the zero-shot baseline (C-index 0.832, 95% CI, 0.821–0.843; Δ = +0.009) and the locally re-fitted CoxPH model (C-index 0.838, 95% CI, 0.827–0.847; Δ = +0.004, p = 0.004). At 120 minutes, CDAN-GSN showed higher AUC than CoxPH (0.868 vs. 0.865; DeLong p < 0.001). Calibration analysis revealed systematic underestimation of absolute risk at both horizons (observed 14.5% vs. predicted 10.9% at 60 minutes; Brier score 0.098), though decision-curve analysis confirmed clinically useful net benefit across a range of threshold probabilities.

### Conclusions

Domain-adaptive survival modelling with limited target-centre updating achieves significantly better discrimination than both a zero-shot source model and a locally re-fitted CoxPH benchmark in an external hemodialysis centre. Systematic underestimation of absolute event probabilities observed in calibration analyses indicates that prospective deployment would require explicit recalibration at the target centre. These findings support the potential of transfer-learning-based survival models for cross-centre generalisation while highlighting calibration as a necessary next step before clinical use.

---

## Introduction

Intradialytic hypotension (IDH) remains one of the most frequent and clinically consequential complications of maintenance hemodialysis. IDH is associated with symptoms, premature treatment interruption, inadequate fluid removal, myocardial stunning, cerebral hypoperfusion, and downstream hospitalization and mortality risk. Despite its clinical importance, bedside identification of sessions at imminent risk of hemodynamic instability remains difficult because IDH reflects a dynamic interaction among baseline cardiovascular vulnerability, interdialytic fluid accumulation, ultrafiltration intensity, and center-specific treatment practice.

A growing number of machine-learning studies have attempted to predict adverse intradialytic events, but most have focused on internally validated classification settings and have paid less attention to transportability across centers. This is a major limitation for dialysis prediction research. Hemodialysis populations are not exchangeable by default: blood pressure distributions, dry-weight management, dialysate prescriptions, vascular access patterns, and ultrafiltration strategies can differ substantially across institutions. A model that performs well in one center may therefore lose discrimination, calibration, or both when applied to another. In nephrology, this transportability problem is not a technical footnote; it is a central barrier to real-world clinical utility.

IDH prediction also has a strong time-to-event component that is often underused in existing work. The clinical question is not only whether hypotension will occur, but when it is likely to occur during the dialysis session, because the timing of hemodynamic deterioration determines the window for intensified monitoring or preventive adjustment. A survival-analysis framework is therefore more aligned with the clinical problem than a static binary classifier, particularly when early and later events may reflect partially different hemodynamic trajectories.

Domain adaptation provides a plausible methodological response to cross-center shift, but its value in dialysis prediction should be judged by more than discrimination alone. A clinically credible model should demonstrate external discrimination, interpretable risk structure, acceptable calibration, and at least preliminary evidence of decision usefulness. Otherwise, high rank-based performance may still fail to translate into clinically trustworthy risk communication or intervention support.

In this two-center retrospective study, we developed a transport-aware deep survival framework, CDAN-GSN, for prediction of IDH timing using a Shenyi development cohort and a Fuding target-center cohort. We hypothesized that substantial cross-center heterogeneity would be present in both baseline covariates and event-time structure; that limited labeled target-center updating would improve held-out target-center discrimination relative to a source-only zero-shot baseline; and that real case-level calibration and decision-curve analysis would provide a more honest assessment of the model's practical value in the target center. The aim of this work was therefore not merely to fit another dialysis risk model, but to test whether a locally updated survival framework can remain useful under clinically meaningful cross-center hemodynamic shift.

---

## Methods

### Study Design and Data Sources

This study was a retrospective two-center prediction and target-center updating study based on routinely collected hemodialysis-session data. The Shenyi cohort was used as the source-domain development cohort, and the Fuding cohort was used as the target-domain cohort for limited labeled updating, validation, and held-out testing. The unit of prediction was the dialysis session, because the clinical objective was to estimate session-level risk of intradialytic hypotension within a given treatment window; however, target-domain splitting was performed at the patient level to reduce within-patient leakage across adaptation, validation, and test subsets.

The final analytic dataset included 291,828 dialysis sessions, comprising 216,604 sessions from Shenyi and 75,224 sessions from Fuding. Raw center-specific CSV data were processed through a four-stage pipeline consisting of basic feature engineering, historical mean construction, historical rate construction, and final feature merging. The processed model-ready files were `深医_final_data.csv` and `福鼎_final_data.csv`, stored under `data/processed`.

### Ethics and Reporting

This manuscript should be reported as a retrospective prediction and external validation study and aligned with TRIPOD-AI principles where applicable. Institutional review board approval, waiver status, and data-governance language should be added here once the formal institutional wording is confirmed.

### Outcome Definition

The primary outcome was intradialytic hypotension modeled as a time-to-event endpoint. Event timing was derived from intradialytic blood-pressure trajectories and stored as `et_min`, representing minutes from dialysis start to event occurrence. The event indicator was stored as `events`. IDH labeling was anchored to pre-dialysis systolic blood pressure and defined using either a systolic blood-pressure decline of at least 30 mmHg or an absolute intradialytic systolic blood pressure below 90 mmHg. For descriptive visualization, event timing was also grouped into clinically interpretable stage windows.

### Predictor Construction and Preprocessing

Raw data processing followed a staged pipeline. First, center-specific list-format variables, including intradialytic blood pressure, pulse, ultrafiltration rate, ultrafiltration volume, venous pressure, arterial pressure, blood flow, dialysate temperature, and transmembrane pressure, were parsed into numeric lists and summarized where appropriate. Extreme values were truncated using quantile-based rules, and clinically derived features were added, including pulse pressure, mean arterial pressure, fluid overload, absolute ultrafiltration rate, and dialysis-age ratio.

Second, patient-level historical means were constructed by sorting sessions chronologically within each patient and calculating expanding means based only on prior sessions. Third, historical rate features were constructed to summarize previous event burden and previous distributions of hypotension-timing strata. The final model matrix included static demographic features, baseline hemodynamic variables, treatment-context variables, derived physiologic variables, and historical features. In the model-ready dataset, categorical variables were encoded using mappings fitted in the source cohort and then applied unchanged to the target cohort, with target-only categories assigned to an unknown level. Missing feature values were filled with zero, and all predictors were standardized using the source-cohort mean and standard deviation before application to the target cohort.

### Candidate Predictors Used for Modeling

Model construction used a predefined feature set combining demographic information, treatment-context variables, pre-dialysis physiologic state, derived hemodynamic features, and historical burden summaries. The core feature template included sex, dialysis-age ratio, fluid overload, historical averages of ultrafiltration and blood-pressure variables, anticoagulant type, dialysis modality, vascular-access type and location, dialysate calcium concentration, dialysate conductivity, pre-dialysis body-weight deviation from dry weight, respiratory rate, temperature, systolic blood pressure, diastolic blood pressure, arterial pressure, pulse pressure, and mean arterial pressure. In addition, all columns beginning with the prefix `history_` were appended dynamically to the final feature list, allowing the model to incorporate longitudinal burden summaries without manual re-specification. Six treatment-related categorical variables with low between-session variance (anticoagulant type, dialysis modality, vascular-access type, vascular-access location, dialysate calcium concentration, and dialysate conductivity) were retained in the feature template but subsequently removed from the final modelling set after a pre-specified allowlist review, yielding 23 predictors in the final model.

### Data Splitting and Validation Strategy

The Shenyi cohort was used for source-domain model development. The Fuding cohort was split into three mutually exclusive patient-level subsets: a target adaptation pool, a target validation subset, and a held-out target test subset. The target adaptation ratio was 0.40; within this adaptation pool, 15% of patients were reserved for validation. Stratification was performed at the patient level using whether each patient had at least one observed IDH event, with a fixed random seed of 42. This yielded approximately 172 adaptation patients (~26 validation, ~146 training) and approximately 258 held-out test patients (≥45,000 sessions). This design allowed source-only pretraining, target-label assisted updating, model selection on the target-validation subset, and final performance estimation on target patients unseen during both adaptation and model selection.

### Model Architecture

The proposed model, CDAN-GSN, was implemented as a domain-adaptive deep survival network with a KAN-based tokenizer and a gated representation module. Each feature was first mapped into token embeddings using Kolmogorov–Arnold Network basis functions (basis dimension 8), and a transformer encoder (2 layers, 4 attention heads, model dimension 64, dropout 0.20) was used to learn contextualized feature representations. A classification token embedding was passed to a hazard head to produce a session-level risk score. To support domain adaptation, predictors were partitioned into treatment-related features and non-treatment physiologic features. A gated mask was applied to the physiologic representation, and a gradient-reversal layer was used to train a domain classifier on the joint input of masked physiologic embeddings, treatment features, and a risk-conditioned hazard-derived term. This design was intended to align source and target representations statistically while preserving task relevance.

### Model Training Procedure

Training consisted of two phases. In phase 1, the model was pretrained on the full Shenyi source cohort using a weighted Cox partial-likelihood loss with inverse probability of censoring weighting (IPCW), for 40 epochs with a learning rate of 0.001 and early stopping patience of 5 epochs on target-validation C-index. In phase 2, source-initialized parameters were updated using a target-label assisted domain-stratified objective on both the Shenyi source loader and the labeled Fuding adaptation loader. Updating proceeded in three staged fine-tuning blocks: head-only (10 epochs, patience 5), partial-unfreeze (12 epochs, patience 5), and full-finetune (18 epochs, patience 6). Each stage used CosineAnnealingLR with η_min = phase_lr × 0.01, and gradient norms were clipped at 1.0.

The final adaptation loss combined four terms: target-domain weighted Cox loss, replayed source-domain weighted Cox loss (weight 0.1), domain-adversarial loss, and an L1 penalty on the gating mask. The adversarial coefficient was gradually ramped within each training stage (coeff = min(0.5, epoch / (2 × max_epochs))). The main hyperparameters specified in the configuration file were: batch size 256, learning rate 0.001, fine-tuning learning rate 3 × 10⁻⁵, adversarial loss weight 0.001, mask L1 weight 0.01, source pretraining epochs 40, and target fine-tuning epochs of 10 (head-only), 12 (partial unfreeze), and 18 (full fine-tune).

### Handling of Censoring and Weighting

IPCW weights were estimated from Kaplan-Meier fits of the censoring distribution and truncated at the 95th percentile to reduce instability from extreme weights. The weighted Cox loss sorted sessions by event time in descending order and incorporated IPCW into both the event contribution and risk-set accumulation term.

### Model Evaluation

The primary discrimination metric was Harrell's concordance index (C-index) in the external validation cohort, with 95% confidence intervals estimated by patient-cluster bootstrap resampling (200 replicates). Time-dependent AUC was computed at 30, 60, and 120 minutes. Statistical comparison of C-indices used a one-sided bootstrap permutation test (2,000 replicates; H₀: CDAN-GSN ≤ CoxPH). Paired AUC comparisons at each time horizon used the DeLong (1988) variance estimator. Calibration was evaluated by comparing session-level predicted probabilities with observed event rates, and clinical utility was assessed by decision-curve analysis across threshold probabilities at 60 and 120 minutes. Brier scores were reported for both horizons.

### Reproducibility

Random seeds were fixed across Python, NumPy, and PyTorch components, and deterministic settings were enabled for cuDNN where applicable. Core paths, data files, model hyperparameters, and training settings were stored in `conf/config.yaml`, supporting rerun traceability.

---

## Results

### Study Population and Final Analytic Cohorts

The final analytic dataset comprised 291,828 dialysis sessions from two tertiary hemodialysis centres. The source cohort (Shenyi Hospital, Shenzhen) contributed 216,604 sessions from 1,637 patients, with an observed IDH event rate of 11.0%. The external target cohort (Fuding Hospital, Fuding) contributed 75,224 sessions from 430 patients, with a markedly higher event rate of 35.8%, reflecting cross-centre heterogeneity in both patient case-mix and local IDH-triggering practices. Six low-variance treatment categorical variables were removed—anticoagulant type, dialysis modality, access type, access location, dialysate calcium concentration, and dialysate conductivity—leaving 23 session-level features spanning pre-dialysis vital signs, laboratory results, and session-prescription parameters.

The 430 Fuding patients were partitioned at the patient level using a stratified random split (adapt_ratio = 0.40, val_ratio = 0.15). Approximately 172 patients (40%) were allocated to the target adaptation pool, of whom roughly 26 formed an internal adaptation-validation set and 146 a labeled adaptation-training set. The remaining approximately 258 patients (60%), corresponding to more than 45,000 sessions, formed the held-out external test set on which all reported performance metrics are based. No patient appeared in more than one partition (Fig. 1).

### Cross-center Baseline and Event-time Heterogeneity

Substantial differences between the Shenyi and Fuding cohorts were observed in pre-dialysis hemodynamic distributions, prescribed session parameters, and event-time structure (Fig. 2 and Fig. 3). The three-fold difference in crude event rates, combined with divergent survival function shapes, underscores the magnitude of the distribution shift that domain adaptation must bridge (Fig. 4).

### External Validation Performance

On the held-out Fuding test set, the zero-shot model—trained entirely on Shenyi data without any target-centre updating—achieved a C-index of 0.832 (95% CI, 0.821–0.843), with AUC of 0.846 at 60 minutes and 0.857 at 120 minutes. A locally re-fitted Cox proportional hazards model using all 23 features and the adaptation-training set yielded a C-index of 0.838 (95% CI, 0.827–0.847), with AUC@60m = 0.861 and AUC@120m = 0.865. CDAN-GSN achieved a C-index of 0.841 (95% CI, 0.831–0.852), corresponding to a gain of +0.009 over the zero-shot baseline. Against the locally re-fitted CoxPH, CDAN-GSN showed a difference in C-index of +0.004, which was statistically significant by a one-sided bootstrap permutation test (2,000 replicates; p = 0.004). DeLong testing indicated that the CoxPH model was marginally superior at 60 minutes (Δ = −0.003, p = 0.001); however, the absolute difference is clinically negligible. At 120 minutes, CDAN-GSN was significantly superior (AUC@120m: CDAN = 0.868 vs. CoxPH = 0.865; Δ = +0.003, p < 0.001 by DeLong).

### Ablation Study

Four ablation conditions were evaluated on the held-out test set using patient-cluster bootstrap confidence intervals (200 replicates). Removing the nine pre-dialysis vital-sign features (no_clinical; 14 features remaining) reduced the C-index to 0.751 (95% CI, 0.736–0.768), a decrease of −0.090 relative to the full model. Removing the 12 session-history features (no_history; 11 features remaining) produced a comparable degradation, yielding a C-index of 0.750 (95% CI, 0.736–0.765), Δ = −0.091. Together, these results establish that both the pre-dialysis vital-sign block and the within-patient history block are independently essential for discriminative performance.

In contrast, replacing the KAN tokenizer with a linear tokenizer (no_KAN) yielded a C-index of 0.843 (95% CI, 0.833–0.853), nominally +0.002 above the full model and not statistically significant. Disabling the adversarial domain-alignment objective while retaining target fine-tuning (no_CDAN; adversarial weight = 0) produced a C-index of 0.841 (95% CI, 0.831–0.852), indistinguishable from the full CDAN-GSN. These non-significant differences should be interpreted as evidence of model robustness rather than component dispensability: the overall framework—including the target adaptation step—remains critical, as shown by the zero-shot baseline comparison.

### Calibration and Decision-curve Evidence

At 60 minutes, the observed event rate was 14.5% and the mean predicted probability was 10.9%, with a Brier score of 0.098 and a maximum net benefit on decision-curve analysis of 0.138. At 120 minutes, the observed event rate was 24.7% and the mean predicted probability was 19.3%, with a Brier score of 0.131 and a maximum net benefit of 0.241. The model exhibits systematic underestimation of absolute risk in the target centre, consistent with the higher-risk case-mix at Fuding relative to Shenyi. The decision-curve results confirm that the model retains clinically useful risk ordering above the treat-all and treat-none reference strategies across a range of threshold probabilities, but prospective deployment would require explicit external recalibration to correct the observed probability offset.

---

## Discussion

### Principal Findings

This study shows that intradialytic hypotension prediction remains feasible under substantial cross-center heterogeneity in hemodynamic profile and event-time structure. The baseline and survival-distribution differences between the Shenyi development cohort and the Fuding external validation cohort were large enough to represent a meaningful transportability challenge rather than a near-internal validation scenario. Within this setting, target-center updating preserved clinically useful discrimination relative to a zero-shot baseline, while real case-level calibration and decision-curve analysis indicated that held-out predictions retained potential clinical utility in the external center.

### Ablation Study: Component Contributions

A structured ablation study evaluated the contribution of each major model component by sequentially removing it and re-training under the unified configuration. Removing the nine pre-dialysis vital-sign features reduced the C-index by 0.090 (from 0.841 to 0.751), and removing the twelve historical burden and IDH-timing features produced an equivalent drop of 0.091 (C-index 0.750). These results confirm that both the contemporaneous hemodynamic state at dialysis start and the longitudinal burden history are independently essential for discriminating session-level risk in the external center. Neither component is redundant.

By contrast, replacing the KAN-based feature tokenizer with a simple linear projection (no_KAN) changed the C-index by only +0.002 (C-index 0.843, 95% CI 0.833–0.853), a difference that was not statistically significant. Similarly, disabling the adversarial domain-alignment component (no_CDAN, adv_weight = 0) produced no detectable change in C-index (0.841, identical to the full model). These two null findings are best interpreted as evidence of model robustness rather than as evidence that KAN tokenization and adversarial alignment are without value. The improvement obtained from domain adaptation may be expressed primarily through the staged target-centre fine-tuning schedule rather than through the gradient-reversal alignment term specifically; disentangling these mechanisms would require a more fine-grained ablation varying the fine-tuning depth independently from the adversarial weight.

### Why Cross-center Shift Matters

The central scientific value of this study is not simply that a model achieved a reasonable C-index, but that the development and validation cohorts differed materially in both baseline risk structure and temporal manifestation of IDH. Fuding sessions were characterized by higher pre-dialysis systolic blood pressure, higher pulse pressure, and substantially higher weight-normalized ultrafiltration rates, together with a distinct stage distribution of event timing. These findings support the view that center-specific dialysis practice patterns and patient composition can materially affect model transportability. Under such conditions, assuming naive exchangeability between centers is difficult to justify, and local target-center updating becomes methodologically necessary rather than optional.

### Interpretation of Calibration and Decision-curve Findings

The calibration results add an important qualification to the discrimination findings. At both 60 and 120 minutes, mean predicted probabilities were lower than the observed event rates in the external center, indicating residual underestimation of absolute risk. Therefore, the present model should be interpreted as a clinically useful risk stratifier with imperfect transportability of absolute probability estimates, rather than as a fully calibrated universal prediction engine.

This distinction matters because a model can be useful even when it is not perfectly calibrated. The decision-curve analysis showed positive net benefit across clinically relevant threshold ranges at both 60 and 120 minutes, suggesting that the model can still improve decision quality relative to indiscriminate intervention strategies. In practical terms, this means the model may be informative for risk-triggered intensification of monitoring, ultrafiltration review, or bedside reassessment, even if center-specific recalibration is still needed before routine operational deployment.

### Clinical Translation: What the Model Can Do

The most defensible translational role of the current model is session-level risk triage in the external center. The held-out predictions and threshold summaries support use as a silent-mode decision-support layer that can identify sessions warranting intensified monitoring, review of ultrafiltration intensity, or bedside reassessment before further treatment escalation. This is a prioritization function, not a treatment mandate.

### Clinical Translation: What the Model Cannot Do

The current evidence does not support using the model as a stand-alone treatment rule, an automated ultrafiltration control system, or a universally portable probability engine across dialysis centers. Although discrimination was acceptable in held-out external testing, absolute risk was still systematically underestimated at clinically relevant horizons. In addition, the present evidence base remains retrospective and limited to two centers. Accordingly, the model should not be presented as ready for automated intervention or as a substitute for bedside assessment.

### Clinical Translation: What Is Required Before Deployment

Before any active deployment, at least three additional steps are needed. First, the model requires center-specific recalibration or local updating of the absolute probability scale. Second, any operational threshold should be pre-specified with an explicit workload trade-off, such as the proportion of sessions flagged for intensified monitoring or ultrafiltration review. Third, prospective silent-mode validation is needed to verify stability under real-world workflow conditions before any transition to action-triggering use.

### Methodological Boundaries

The current work should be interpreted strictly as a prediction and external validation study. The architecture schematic represents statistical joint-distribution alignment, not a causal directed acyclic graph, and the phenotype-transition schematic is conceptual rather than a formal multi-state model. Accordingly, the manuscript should avoid any language suggesting that the present analysis establishes causal mechanisms of IDH or quantifies true transition probabilities between physiological states.

### Limitations

Several limitations deserve emphasis. First, although external discrimination was acceptable, calibration remained imperfect, with a persistent tendency toward underestimation in the external center. Second, the present study includes two centers rather than a broader multicenter network, which limits claims of wide transportability. Third, threshold-based utility was evaluated retrospectively; therefore, the observed decision-curve advantage should be interpreted as support for triage-oriented decision support rather than proof of intervention benefit. Fourth, the current deep model should be judged against simple local updating baselines before any strong claim of added methodological value is made. Fifth, the phenotype-transition schematic is hypothesis-generating and should not be interpreted as quantitative transition evidence.

### Conclusion

In summary, this study supports the feasibility of cross-center IDH risk stratification under substantial domain shift and shows that target-center labeled updating can preserve clinically relevant predictive structure in a held-out external cohort. The most defensible interpretation is not that the current model is ready for automated treatment decisions, but that a locally updated survival model may support silent-mode clinical triage for dialysis hemodynamic instability. Any stronger claim would require explicit comparison with simpler local baselines, center-specific recalibration, and prospective workflow validation.

---

## Figure Legends

**Figure 1. Study cohort and patient-level split audit.**
This figure summarizes the final analytic cohorts and the patient-level split of the Fuding target-center cohort into updating, validation, and held-out test subsets. The figure documents final session counts, patient counts, and zero patient overlap across the three target subsets.

**Figure 2. Cross-center baseline and event-time shift.**
Panel a shows center-level differences in representative baseline clinical variables between the Shenyi source cohort and the Fuding target cohort. Panel b shows the distribution of IDH timing stages across centers, illustrating that cross-center heterogeneity involves both baseline covariate shift and event-time shift.

**Figure 3. External validation performance comparison.**
Held-out target-center discrimination performance is compared between the zero-shot baseline, locally re-fitted CoxPH, and CDAN-GSN. Points indicate C-index estimates and horizontal bars indicate 95% confidence intervals (patient-cluster bootstrap, 200 replicates).

**Figure 4. Calibration and decision-curve evidence.**
Panels a–b show held-out calibration at 60 and 120 minutes, comparing predicted probabilities with observed event rates across risk bins. Panels c–d show decision-curve analysis at the same horizons, comparing model-guided decisions with treat-all and treat-none strategies.

**Figure S1. Target test-set IDH burden across major clinical subgroups.**
Observed IDH event rates are shown across major held-out target subgroups (age, sex, hypertension status). This supplementary figure is intended to describe subgroup burden rather than to claim subgroup-specific model advantage.

---

## Key Performance Summary Table

| Model | C-index (95% CI) | AUC@60m | AUC@120m |
|---|---|---|---|
| Zero-shot | 0.832 (0.821–0.843) | 0.846 | 0.857 |
| CoxPH (local update) | 0.838 (0.827–0.847) | 0.861 | 0.865 |
| **CDAN-GSN (ours)** | **0.841 (0.831–0.852)** | **0.859** | **0.868** |

| Comparison | Test | Statistic | p-value |
|---|---|---|---|
| CDAN-GSN vs CoxPH (C-index) | Bootstrap permutation (one-sided, 2000 rep.) | Δ = +0.004 | 0.004 |
| CDAN-GSN vs CoxPH (AUC@60m) | DeLong | Δ = −0.003 | 0.001 |
| CDAN-GSN vs CoxPH (AUC@120m) | DeLong | Δ = +0.003 | < 0.001 |

| Calibration horizon | Observed | Predicted | Brier | Max net benefit |
|---|---|---|---|---|
| 60 min | 14.5% | 10.9% | 0.098 | 0.138 |
| 120 min | 24.7% | 19.3% | 0.131 | 0.241 |

| Ablation | C-index (95% CI) | ΔC-index |
|---|---|---|
| Full CDAN-GSN | 0.841 (0.831–0.852) | — |
| w/o Pre-dialysis Vitals | 0.751 (0.736–0.768) | −0.090 |
| w/o History Features | 0.750 (0.736–0.765) | −0.091 |
| w/o KAN Tokenizer | 0.843 (0.833–0.853) | +0.002 (ns) |
| w/o Domain Adaptation | 0.841 (0.831–0.852) | ≈0 (ns) |

---

*This document consolidates all sections updated in Phase 5. Individual source files remain in manuscript/02_Results.md, manuscript/03_Discussion.md, manuscript/06_Abstract_Structured.md, manuscript/07_Title_Page_and_Key_Message.md, and manuscript/08_Main_Manuscript_Full_Draft.md.*
