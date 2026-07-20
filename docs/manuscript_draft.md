# Outcome-Specific Target-Center Updating for Cross-Center Prediction of Hemodynamic Events During Hemodialysis

Manuscript status: registered historical evidence; confirmatory submission freeze blocked, 2026-07-19

Intended article type: Original Research
Target journal: to be selected after confirmatory ablations and reporting checks

> Submission hold: All numerical claims below map to registered historical v1 run artifacts. Exact v1 processed bytes remain unavailable. A separately versioned v2 cohort passed prior-history and split audits and is reserved for architecture-matched server validation; no v2 run is yet accepted as manuscript evidence. Bracketed items are mandatory metadata or evidence blockers.

## Title Page

**Full title:** Outcome-Specific Target-Center Updating for Cross-Center Prediction of Hemodynamic Events During Hemodialysis

**Short title:** Target-center updating for hemodynamic events

**Authors and affiliations:** [BLOCKER: author names, degrees, affiliations, and corresponding author]

**Word count:** [TO UPDATE AFTER JOURNAL SELECTION]

**Funding:** [BLOCKER: verified funding statement]

**Conflicts of interest:** [BLOCKER: author declarations]

**Data and code availability:** [BLOCKER: approved data-governance statement and stable repository release]

## Abstract

### Background

Clinical prediction models may lose validity across institutions because patient characteristics, treatment practices, and outcome prevalence differ between centers. Aligning all cross-center variation may be undesirable when some variation reflects clinically relevant context. We evaluated an outcome-specific target-center updating framework for predicting intradialytic hypotension (IDH) and intradialytic hypertension (IH) at the start of a hemodialysis session.

### Methods

We conducted a retrospective two-center study using 211,452 sessions from 1,628 patients at the source center and 74,947 sessions from 430 patients at the target center. Target-center patients were separated into update, validation, and held-out test sets. Twenty-four session-start variables were partitioned a priori into physiology/history and treatment-context groups. A source multilayer perceptron was updated with labeled target-center data and outcome-specific correlation alignment: the physiology branch for IDH and the treatment-context branch for IH. Logistic models and non-stratified updating strategies were retained as comparators. Discrimination, precision-recall performance, and Brier score were evaluated in 29,866 held-out sessions from 172 target-center patients. Confidence intervals and within-run AUC differences used 1,000 patient-cluster bootstrap replicates.

### Results

Outcome burden differed substantially between centers: IDH occurred in 14.5% of source sessions and 38.5% of target sessions, whereas IH occurred in 28.4% and 12.0%, respectively. In the prespecified seed-2024 run, target-center updating increased IDH AUC from 0.7951 (95% CI 0.7768 to 0.8125) to 0.8375 (0.8234 to 0.8522), a paired difference of 0.0424 (0.0348 to 0.0515). IH AUC increased from 0.8516 (0.8347 to 0.8673) to 0.8681 (0.8522 to 0.8826), a difference of 0.0165 (0.0112 to 0.0229). Across five initialization seeds, updated-model AUC was stable for IDH (mean 0.8371, SD 0.0009) and IH (mean 0.8687, SD 0.0006), while source-model AUC varied more. The updated MLP did not clearly outperform target-local logistic regression for either endpoint. Exploratory attribution and latent-space analyses suggested larger representation changes for IDH than IH, but discrepancy measures were not uniformly improved.

### Conclusions

Target-center updating produced stable held-out discrimination for two hemodynamic endpoints under marked cross-center outcome shift. The present evidence supports the complete updating procedure, but does not isolate an advantage of outcome-specific alignment over simpler updating. Architecture-matched, seed-matched ablations remain required before confirmatory mechanism claims.

**Keywords:** hemodialysis; intradialytic hypotension; intradialytic hypertension; domain shift; model updating; transportability; clinical artificial intelligence

## Introduction

Hemodynamic instability during hemodialysis is common and clinically consequential. Predicting an adverse event before treatment begins could support closer monitoring and treatment review, but a model developed at one center may not transfer directly to another. Differences in case mix, measurement systems, treatment policy, and event prevalence can alter both predictor distributions and predictor-outcome relationships. [REFERENCE REQUIRED: IDH/IH clinical background and external validation literature]

Most domain-adaptation methods frame between-center variation as nuisance information to be removed. That assumption is incomplete in clinical settings. Pre-dialysis blood pressure and prior instability may represent relatively transportable physiological risk, whereas ultrafiltration and dry-weight management may encode both patient status and center-specific practice. An alignment rule that ignores this distinction could suppress useful contextual information or align features unrelated to the endpoint. [REFERENCE REQUIRED: domain adaptation and clinical transportability literature]

We therefore designed an outcome-specific target-center updating framework. The model separates variables into a physiology/history branch and a treatment-context branch. For IDH, correlation alignment is applied to the physiology/history representation; for IH, it is applied to the treatment-context representation. The unaligned branch remains available to the prediction head. This is a clinically motivated design hypothesis, not a proven biological decomposition.

The study had three objectives. First, we quantified cross-center differences in IDH and IH burden. Second, we evaluated whether the complete target-center updating procedure improved held-out target-center performance over the source model and whether performance was stable across initialization seeds. Third, we explored whether latent discrepancy and feature attribution changed differently by endpoint. A superiority comparison against simpler alignment strategies was planned, but is not considered confirmatory in the current draft because the existing baselines are not architecture matched.

## Methods

### Study design and reporting frame

This was a retrospective, two-center model-development and target-center updating study. The source center supplied data for source model development. The target center supplied mutually exclusive patient groups for model updating, validation/calibration, and held-out evaluation. Because labeled target-center data were used for updating and tuning, the study is not a zero-shot external validation.

Reporting will be finalized against TRIPOD+AI and applicable prediction-model guidance after the analysis package is frozen. [REFERENCE REQUIRED]

### Setting, participants, and ethics

The binary preflight artifact records 211,452 eligible source-center sessions from 1,628 patients and 74,947 target-center sessions from 430 patients. Detailed eligibility criteria, recruitment dates, device context, and center descriptions must be recovered from the approved protocol and data dictionary before submission.

[BLOCKER: institutional review board names, approval identifiers, consent/waiver details, privacy safeguards, and cohort dates]

### Outcomes

Two session-level binary endpoints were constructed from baseline and intradialytic blood pressure measurements:

- IDH: baseline systolic blood pressure minus intradialytic systolic blood pressure of at least 30 mmHg, or an intradialytic systolic blood pressure of 90 mmHg or lower.
- IH: intradialytic mean arterial pressure minus baseline mean arterial pressure greater than 10 mmHg.

The prediction time was the start of the dialysis session. Intradialytic measurements used to define outcomes were excluded from predictors. [BLOCKER: specify whether any intradialytic reading, minimum/maximum reading, or first qualifying reading defined the endpoint and how measurement artifacts were handled]

### Predictors and clinical grouping

The frozen run used 24 variables listed in `experiments/audit/feature_allowlist.csv`. They comprised demographics, pre-dialysis measurements, and prior-session summaries. Current-session intradialytic summaries, treatment duration, outcome variables, and event timing were explicitly excluded.

Twenty variables were assigned to the physiology/history branch, including pre-dialysis blood pressure, pulse pressure, mean arterial pressure, respiratory rate, temperature, prior blood pressure summaries, and prior event-rate features. Four variables were assigned to the treatment-context branch: volume overload, historical maximum ultrafiltration volume, pre-dialysis weight minus dry weight, and historical mean ultrafiltration rate. This grouping was specified from clinical reasoning and has not yet been independently adjudicated.

Prior-session features were intended to use observations preceding the index session only. The v2 rebuilding code groups tied sessions at the same timestamp and derives their histories only from strictly earlier timestamps. The v2 audit found zero remaining prior-history mismatches in either cohort. This does not retroactively prove byte-level equivalence to the unavailable v1 processed inputs.

### Data partitioning

All splits were performed at patient level. Source-center patients were divided into 1,465 training patients (187,479 sessions) and 163 validation patients (23,973 sessions). At the target center, 219 patients (38,483 sessions) were used for updating, 39 patients (6,598 sessions) for validation and calibration, and 172 patients (29,866 sessions) for held-out testing. The split manifest records no overlap among target-center patient sets. The split seed was 20260715.

### Preprocessing and missing data

Categorical mappings, numerical medians, means, and standard deviations were fitted on source-training data only and then applied unchanged to all source and target partitions. Missing numerical values were replaced with source-training medians. Unknown categorical values were assigned a reserved code. No missingness indicators were included in the frozen feature set.

The frozen v1 preprocessing metadata contains data-quality signals that require resolution before submission, including two source variables with zero variance after parsing and extreme source pre-dialysis temperature values. Historical findings remain artifact-traceable but cannot be labeled exactly reproducible from current bytes; these values still require clinical/source-record adjudication.

### Models

Four model roles were evaluated for each endpoint:

1. A source logistic regression trained on source-center data.
2. A target-local logistic regression trained on the target update set.
3. A source multilayer perceptron trained on source-center data.
4. An updated multilayer perceptron initialized from the source network and updated using source and labeled target-center data.

The mechanism-aware MLP used separate two-layer encoders for the prespecified branches. Each encoder contained hidden dimensions of 64 and 32 with ReLU activations and dropout of 0.20. Their representations were concatenated and passed to a linear prediction head. Training used focal loss, AdamW, gradient clipping, validation-AUC early stopping, and Platt scaling fitted on the appropriate validation patients.

During updating, CORAL penalized differences between source and target covariance matrices in one branch. The physiology/history branch was aligned for IDH, and the treatment-context branch was aligned for IH. The CORAL weight was 0.01. The prediction loss used the pooled source and target update sessions.

### Comparators and ablations

Existing artifacts include target-only fine-tuning, global CORAL, a random-alignment control, and the outcome-specific model. These artifacts are useful for pipeline development but are not a valid confirmatory ablation set: encoder architecture differs between methods, and not every method was run under every initialization seed. Accordingly, this draft does not interpret between-method differences as evidence of mechanism-aware superiority.

The confirmatory analysis must compare architecture-matched models across the same five or more initialization seeds and the same patient split. Required models are dual-branch no-alignment updating, dual-branch global CORAL, outcome-specific CORAL, a size-matched random feature partition, and single-encoder reference models.

### Evaluation and statistical analysis

The primary metric was ROC AUC on held-out target-center patients. Secondary metrics were precision-recall AUC and Brier score. Classification thresholds were selected on validation data using Youden's index; sensitivity and specificity are therefore threshold-dependent secondary summaries.

Confidence intervals for model metrics and paired differences between predictions were calculated with 1,000 patient-cluster bootstrap replicates. Five initialization seeds (7, 13, 42, 99, and 2024) were summarized using the mean and sample standard deviation. No correction for multiple exploratory analyses was applied.

Existing subgroup outputs used session-level rather than patient-cluster resampling and must not be used for inferential claims. Existing SHAP outputs used 400 sampled observations per domain and are interpreted as model attribution rather than causal feature effects. Latent-space analyses are descriptive and metric dependent.

### Reproducibility

Each frozen evaluation JSON records the split seed, initialization seed, configuration fingerprint, data hashes, feature-allowlist hash, environment, and patient-cluster bootstrap settings. Eight historical v1 runs have been registered with hashes for configurations, checkpoints, predictions, evaluations, datasets, and splits. Their patient split manifests are identical and contain no patient overlap. Exact v1 processed bytes could not be recovered and this remains a reproducibility limitation. Separately, fixed raw-file hashes were used to construct the v2 confirmatory contract; it reproduces the historical cohort counts and split and passed a strictly-prior history audit after tied timestamps were handled together. V2 is reserved for new server-side confirmatory analyses and is not substituted into historical v1 numerical claims.

## Results

### Cohort shift and held-out evaluation set

IDH prevalence was 14.5% at the source center (30,694/211,452 sessions) and 38.5% at the target center (28,881/74,947). IH prevalence moved in the opposite direction: 28.4% at the source center (59,978/211,452) and 12.0% at the target center (8,962/74,947). The held-out target set contained 29,866 sessions from 172 patients, including 11,575 IDH events (38.8%) and 3,687 IH events (12.3%). These differences establish substantial cross-center outcome shift but do not identify its causes.

### Main target-center updating results

In the seed-2024 run, the source IDH MLP achieved an AUC of 0.7951 (95% CI 0.7768 to 0.8125), PR AUC of 0.7463, and Brier score of 0.2106. After target-center updating, AUC was 0.8375 (0.8234 to 0.8522), PR AUC 0.7815, and Brier score 0.1566. The paired AUC difference was 0.0424 (0.0348 to 0.0515).

For IH, source-model AUC was 0.8516 (0.8347 to 0.8673), PR AUC 0.4907, and Brier score 0.0872. Updated-model AUC was 0.8681 (0.8522 to 0.8826), PR AUC 0.5234, and Brier score 0.0791. The paired AUC difference was 0.0165 (0.0112 to 0.0229).

The target-local logistic model achieved AUCs of 0.8353 for IDH and 0.8678 for IH. Stored paired patient-cluster bootstrap differences between the updated MLP and target-local logistic regression were small and compatible with no difference: +0.0022 (95% CI -0.0003 to 0.0047) for IDH and +0.0003 (-0.0017 to 0.0022) for IH. These paired estimates differ slightly from direct subtraction of the reported point estimates because they summarize the bootstrap contrast distribution.

### Initialization-seed stability

Across five seeds, source-model AUC varied from 0.7148 to 0.8024 for IDH and from 0.8455 to 0.8613 for IH. In contrast, updated-model AUC ranged from 0.8356 to 0.8379 for IDH (mean 0.8371, SD 0.0009) and from 0.8680 to 0.8694 for IH (mean 0.8687, SD 0.0006). The corresponding source-to-updated differences were 0.0606 (SD 0.0347) and 0.0157 (SD 0.0069). Because updated performance was nearly constant while source performance varied, the difference magnitude should not be interpreted as a pure estimate of alignment benefit.

### Calibration and deployment-oriented diagnostics

From the registered seed-2024 held-out predictions, IDH calibration intercept/slope changed from 1.82 (95% patient-cluster CI 1.59 to 2.09)/1.64 (1.50 to 1.80) for the source MLP to 0.05 (-0.03 to 0.14)/1.10 (1.05 to 1.17) after updating. For IH, the corresponding values changed from 1.47 (1.28 to 1.67)/2.01 (1.88 to 2.14) to 0.05 (-0.07 to 0.17)/0.97 (0.91 to 1.03). Ten-quantile-bin expected calibration error was 0.160 (0.139 to 0.180) versus 0.018 (0.012 to 0.029) for source and updated IDH models and 0.051 (0.045 to 0.060) versus 0.007 (0.004 to 0.015) for IH.

At thresholds selected previously on validation patients by Youden's index, the updated IDH model had sensitivity 0.745 (0.706 to 0.778), specificity 0.777 (0.745 to 0.806), and PPV 0.679 (0.651 to 0.706), detecting 288.6 (250.2 to 326.8) events with 136.5 (121.9 to 150.1) false alerts per 1,000 sessions. The updated IH model had sensitivity 0.817 (0.770 to 0.854), specificity 0.756 (0.715 to 0.794), and PPV 0.320 (0.294 to 0.348), detecting 100.8 (82.1 to 122.3) events with 213.9 (184.7 to 245.9) false alerts per 1,000 sessions. These are registered fixed-threshold operating characteristics with 1,000 patient-cluster bootstrap replicates, not clinically prespecified deployment thresholds. Decision curves remain threshold-grid sensitivity analyses; no clinically relevant interval is claimed because nephrology prespecification is pending.

### Latent representation analyses

Latent analyses from the seed-2024 main run were metric dependent. For IDH, RBF MMD increased slightly from 0.1837 before updating to 0.1888 afterward, while linear MMD decreased from 1.1301 to 1.0870 and covariance distance decreased. Wasserstein distance increased from 0.1418 to 0.1574. Domain-classifier AUC remained very high (0.9980 before and 0.9905 after), indicating that center membership remained readily distinguishable. The target linear-probe AUC increased from 0.8338 to 0.8498.

For IH, latent discrepancies changed little: RBF MMD was 0.0093 before and 0.0085 after updating, while domain-classifier AUC increased from 0.7254 to 0.7494. These results do not demonstrate general domain invariance. They suggest that representation changes were outcome and metric dependent.

### Exploratory feature attribution

In the seed-2024 SHAP analysis, physiology/history variables accounted for 83.5% of target-domain absolute attribution for IDH before updating and 94.1% afterward. Cross-domain feature-rank correlation increased from Spearman r=0.766 to r=0.817. The most attributed IDH variables after updating included historical and current pre-dialysis systolic blood pressure, pulse pressure, and prior IDH timing/rate summaries.

For IH, physiology/history variables already accounted for 96.8% of target-domain attribution before updating and 97.9% afterward. Cross-domain rank correlation changed from r=0.906 to r=0.933. These findings contradict a simple claim that IH prediction is dominated by treatment-context features. At most, they leave open the hypothesis that a small treatment-context representation should be preserved rather than globally aligned.

### Exploratory subgroup patterns

Previous subgroup inferential claims have been discarded. Six intended subgroup domains are registered, but estimates and interaction P values remain blank because clinical cut points, minimum patient counts, no-test-access attestation, an accepted v2 server run, and patient-cluster analysis have not yet passed their evidence gates.

## Discussion

### Principal findings

In this two-center hemodialysis study, target-center updating yielded stable held-out discrimination for both IDH and IH despite large and opposing differences in outcome prevalence. The absolute gain was larger for IDH in the main run, whereas the source IH model already performed strongly. Updated MLP performance was remarkably stable across initialization seeds and was similar to a target-local logistic model.

The representation analyses support a more cautious interpretation than the original project hypothesis. IDH updating was accompanied by a shift in model attribution toward physiology/history variables and a reduction in one nonlinear discrepancy measure. However, other discrepancy measures worsened and center membership remained almost perfectly predictable from the latent representation. For IH, both attribution and latent discrepancy changed little. The data therefore support outcome-dependent updating behavior, but not universal domain invariance or a confirmed advantage of the prespecified alignment branch.

### Interpretation

The strong IDH improvement may reflect a combination of target supervision, recalibration, pooled source-target training, dual-branch architecture, and selective CORAL. The current experiments do not isolate these components. The near-identical updated AUC across seeds, together with variable source AUC, suggests that the large source-to-updated difference partly reflects stabilization by abundant labeled target data. This makes target-local logistic regression an important comparator and highlights the need to report absolute performance rather than improvement alone.

The physiological attribution shift is clinically plausible, but SHAP group proportions are determined partly by feature definitions, correlations, scaling, architecture, and background samples. They should not be read as direct evidence that the model recovered a biological mechanism. Likewise, treatment-context preservation for IH remains a design hypothesis because treatment variables contributed little total attribution and no matched ablation isolates their preservation.

### Clinical and methodological implications

The results favor a deployment workflow that explicitly allocates target-center patients for updating, validation/calibration, and held-out testing. They do not support deploying the source model unchanged. At the same time, use of 60% of target patients for updating means the framework should be viewed as local model updating, not external transport without target labels.

Methodologically, the study illustrates why domain adaptation should be evaluated with architecture-matched controls, absolute held-out metrics, patient-level uncertainty, and multiple discrepancy measures. A lower value for one latent metric is insufficient if domain classification remains easy or other distances increase.

### Strengths

Strengths include a large number of dialysis sessions, patient-level separation of target update/validation/test sets, source-fitted preprocessing, explicit exclusion of current-session future information, paired patient-cluster bootstrap evaluation for the main comparisons, probability calibration on validation patients, and recorded run provenance.

### Limitations

This retrospective study used two centers and may not generalize to other institutions, devices, documentation systems, or patient populations. Repeated sessions increase effective sample dependence, and patient-cluster resampling does not address every longitudinal correlation structure. The target test set contains 172 patients despite many sessions. The outcome definitions depend on routinely collected blood pressure measurements and require clinical adjudication.

The current historical ablation set is confounded by encoder architecture, although seed-2024 method runs are available. The mechanism-based feature grouping was manually specified. SHAP and latent analyses were exploratory. Previous subgroup inference is invalid and has been withheld. Missing values were median imputed without missingness indicators. Data-quality flags remain unresolved. Finally, exact historical v1 processed bytes cannot be reconstructed from the current workspace; the separately audited v2 contract does not remove that historical limitation.

## Conclusion

Registered historical evidence indicates that target-center updating produced stable held-out discrimination and better calibration for IDH and IH under substantial cross-center outcome shift. It does not establish that mechanism-aware alignment is superior to simpler updating. The architecture-matched v2 server experiment, clinically prespecified DCA interpretation, corrected v2 subgroup inference, source-data adjudication, and complete reporting metadata are required before submission; unavailable exact v1 processed bytes must remain disclosed as a historical reproducibility limitation.

## Declarations

### Ethics approval and consent to participate

[BLOCKER: verified statement]

### Consent for publication

[BLOCKER: verified statement or not-applicable determination]

### Data availability

[BLOCKER: governance-approved statement; do not promise public clinical data]

### Code availability

[BLOCKER: archive a release that reproduces the frozen evidence contract]

### Funding

[BLOCKER: verified funding statement]

### Competing interests

[BLOCKER: verified author declarations]

### Author contributions

[BLOCKER: CRediT roles]

## References

[BLOCKER: build and verify the reference library; every DOI/PMID must be checked before insertion]

## Tables, Figures, and Supplement

- Table 1: Registered source/target cohort and patient-level analysis allocation under the dual-binary contract. Baseline characteristics are not claimed because a submission-grade registered baseline table is unavailable.
- Table 2: Main held-out discrimination, precision-recall, Brier score, and paired differences.
- Table 3: Architecture-matched, seed-matched ablation results after confirmatory runs.
- Figure 1: Study design and target-center updating framework.
- Figure 2: Architecture-matched ablation after the confirmatory evidence gate passes.
- Figure 3: Registered clinical operating diagnostics with patient-cluster uncertainty at validation-derived thresholds; DCA interpretation remains pending clinical threshold prespecification.
- Figure 4: Calibration and transportability analysis from registered predictions.
- Figure 5: Exploratory latent representation and SHAP analysis, explicitly labeled exploratory.
- Supplementary Methods and Evidence Definitions: evidence contracts, full historical and confirmatory hyperparameters, seeds, patient-cluster bootstrap method, all registered feature definitions, and the gated subgroup protocol.
- Supplementary Figure S1: Full seed-level performance from the five registered historical runs; generated with descriptive across-seed summaries.
- Supplementary Figure S2: Patient-cluster subgroup analysis, to be generated only after definitions and server evidence pass their gates.

Figures 1, 3, 4, and 5 are staged under `figures/submission_staging/`; Figure 2 is deliberately absent until accepted server evidence exists. The evidence-linked supplementary methods are staged at `docs/submission_supplement.md`. See `docs/visualization_plan.md` and `docs/evidence_audit.md` for the artifact mapping and submission gates.

### Figure legends

**Figure 1. Study design and outcome-specific target-center updating framework.** The upper panels show source and target cohort size, opposing IDH and IH prevalence shifts, and mutually exclusive target update, validation, and held-out test allocations. The lower panels depict the registered historical two-branch configuration: physiology/history is CORAL-aligned for IDH and treatment context for IH, while the other branch is retained. Labeled target-center data were used for updating and validation; therefore, the held-out evaluation is not zero-shot external validation. The diagram describes the complete registered procedure and does not establish superiority of the alignment strategy.

**Figure 2. Architecture-matched confirmatory ablation.** This figure is intentionally not generated in the current evidence freeze. After all required server runs pass acceptance, the upper panels will compare absolute held-out ROC AUC across the five locked strategies, with pale lines connecting the same initialization seed. The lower panels will report seed-specific, patient-paired ROC AUC differences between outcome-specific CORAL and each comparator with 95% patient-cluster intervals. Mean ticks and the five-seed spread will remain descriptive; PR AUC, Brier score, and ECE will be reported in the complete confirmatory table. No mechanism comparison is reported before that gate passes.

**Figure 3. Registered clinical operating diagnostics.** Forest plots compare sensitivity, specificity, positive predictive value, and negative predictive value for the source MLP, updated MLP, and target-local logistic regression using patient-cluster intervals. Workload panels report events detected and false alerts per session volume at validation-derived Youden thresholds. These thresholds are not clinically prespecified deployment recommendations, and the figure does not establish net clinical benefit.

**Figure 4. Calibration and transportability analysis.** Held-out target-center ROC, precision-recall, and calibration panels compare the three registered model roles for both endpoints. Curves are descriptive diagnostics from registered prediction files. Calibration intercept and slope uncertainty is reported in the clinical-utility package; the plotted probability-bin curves are not interval-banded.

**Figure 5. Exploratory representation and attribution analysis.** Metric-specific latent discrepancy, domain separability, physiology/history SHAP contribution, and cross-center feature-rank consistency are shown before and after updating. These analyses are exploratory; SHAP attribution is not causal, discrepancy measures do not move uniformly, and the panels do not demonstrate domain invariance or a biological mechanism.

**Supplementary Figure S1. Seed-level held-out discrimination.** Connected points show source and updated ROC AUC within each of the five registered historical initialization-seed runs for IDH and IH. The panels share the same ROC AUC scale. Across-seed means and sample standard deviations summarize descriptive stability; they are not confidence intervals. Source-to-updated line segments represent the complete updating procedure within a run and do not isolate an alignment effect.
