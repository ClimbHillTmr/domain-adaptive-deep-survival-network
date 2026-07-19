# Outcome-Specific Target-Center Updating for Cross-Center Prediction of Hemodynamic Events During Hemodialysis

Manuscript status: evidence-gated working draft, 2026-07-19

Intended article type: Original Research
Target journal: to be selected after confirmatory ablations and reporting checks

> Submission hold: This draft is intentionally conservative. It does not claim that mechanism-aware alignment outperforms fine-tuning or global alignment because the current repository lacks architecture-matched, seed-matched comparisons. Bracketed items are mandatory metadata or evidence blockers.

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

Outcome burden differed substantially between centers: IDH occurred in 14.5% of source sessions and 38.5% of target sessions, whereas IH occurred in 28.4% and 12.0%, respectively. In the prespecified seed-2024 run, target-center updating increased IDH AUC from 0.7189 (95% CI 0.6966 to 0.7404) to 0.8370 (0.8226 to 0.8520), a paired difference of 0.1182 (0.1041 to 0.1323). IH AUC increased from 0.8600 (0.8439 to 0.8751) to 0.8684 (0.8525 to 0.8827), a difference of 0.0084 (0.0049 to 0.0122). Across five initialization seeds, updated-model AUC was stable for IDH (mean 0.8374, SD 0.0004) and IH (mean 0.8684, SD 0.0002), while source-model AUC varied more. The updated MLP did not clearly outperform target-local logistic regression for either endpoint. Exploratory attribution and latent-space analyses suggested larger representation changes for IDH than IH, but alternative discrepancy measures were not uniformly improved.

### Conclusions

Target-center updating produced stable held-out discrimination for two hemodynamic endpoints under marked cross-center outcome shift. The present evidence supports the complete updating procedure, but does not isolate an advantage of outcome-specific alignment over simpler updating. Architecture-matched, seed-matched ablations and prediction-level calibration analyses are required before confirmatory claims.

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

Prior-session features were intended to use observations preceding the index session only. The rebuilding code sorts sessions within patient and applies a one-session shift before expanding summaries. [SUBMISSION BLOCKER: re-run the current-contract history audit on the exact frozen cohorts and verify the first session and randomly sampled later sessions]

### Data partitioning

All splits were performed at patient level. Source-center patients were divided into 1,465 training patients (187,479 sessions) and 163 validation patients (23,973 sessions). At the target center, 219 patients (38,483 sessions) were used for updating, 39 patients (6,598 sessions) for validation and calibration, and 172 patients (29,866 sessions) for held-out testing. The split manifest records no overlap among target-center patient sets. The split seed was 20260715.

### Preprocessing and missing data

Categorical mappings, numerical medians, means, and standard deviations were fitted on source-training data only and then applied unchanged to all source and target partitions. Missing numerical values were replaced with source-training medians. Unknown categorical values were assigned a reserved code. No missingness indicators were included in the frozen feature set.

The frozen preprocessing metadata contains data-quality signals that require resolution before submission, including two source variables with zero variance after parsing and extreme source pre-dialysis temperature values. Results remain provisional until the exact frozen processed cohorts are restored and these values are traced to source records.

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

Each frozen evaluation JSON records the split seed, initialization seed, configuration fingerprint, data hashes, feature-allowlist hash, environment, and patient-cluster bootstrap settings. However, the current checked-in processed CSV files do not match the hashes in the frozen runs, and one frozen predictor is absent from their headers. Reproduction from the current workspace is therefore blocked until the exact input cohorts are restored or the full pipeline is regenerated under a new versioned contract.

## Results

### Cohort shift and held-out evaluation set

IDH prevalence was 14.5% at the source center (30,694/211,452 sessions) and 38.5% at the target center (28,881/74,947). IH prevalence moved in the opposite direction: 28.4% at the source center (59,978/211,452) and 12.0% at the target center (8,962/74,947). The held-out target set contained 29,866 sessions from 172 patients, including 11,575 IDH events (38.8%) and 3,687 IH events (12.3%). These differences establish substantial cross-center outcome shift but do not identify its causes.

### Main target-center updating results

In the seed-2024 run, the source IDH MLP achieved an AUC of 0.7189 (95% CI 0.6966 to 0.7404), PR AUC of 0.6752, and Brier score of 0.2606. After target-center updating, AUC was 0.8370 (0.8226 to 0.8520), PR AUC 0.7814, and Brier score 0.1567. The paired AUC difference was 0.1182 (0.1041 to 0.1323).

For IH, source-model AUC was 0.8600 (0.8439 to 0.8751), PR AUC 0.5078, and Brier score 0.0826. Updated-model AUC was 0.8684 (0.8525 to 0.8827), PR AUC 0.5222, and Brier score 0.0792. The paired AUC difference was 0.0084 (0.0049 to 0.0122).

The target-local logistic model achieved AUCs of 0.8353 for IDH and 0.8678 for IH. Differences between the updated MLP and target-local logistic regression were small and compatible with no difference: +0.0018 (95% CI -0.0007 to 0.0042) for IDH and +0.0006 (-0.0009 to 0.0022) for IH.

### Initialization-seed stability

Across five seeds, source-model AUC varied from 0.6865 to 0.7943 for IDH and from 0.8250 to 0.8611 for IH. In contrast, updated-model AUC ranged from 0.8370 to 0.8379 for IDH (mean 0.8374, SD 0.0004) and from 0.8682 to 0.8688 for IH (mean 0.8684, SD 0.0002). The corresponding source-to-updated differences were 0.0930 (SD 0.0460) and 0.0266 (SD 0.0178). Because updated performance was nearly constant while source performance varied, the difference magnitude should not be interpreted as a pure estimate of alignment benefit.

### Latent representation analyses

Latent analyses were available for seed 42. For IDH, RBF MMD decreased from 0.2764 before updating to 0.1861 afterward. In contrast, linear MMD increased from 1.0921 to 1.2102, Wasserstein distance increased from 0.1466 to 0.1739, and covariance distance also increased. Domain-classifier AUC remained very high (0.9995 before and 0.9985 after), indicating that center membership remained readily distinguishable. The target linear-probe AUC increased from 0.8199 to 0.8491.

For IH, latent discrepancies changed little: RBF MMD was 0.0121 before and 0.0117 after updating, while domain-classifier AUC increased from 0.7577 to 0.7689. These results do not demonstrate general domain invariance. They suggest that representation changes were outcome and metric dependent.

### Exploratory feature attribution

In the seed-2024 SHAP analysis, physiology/history variables accounted for 60.0% of target-domain absolute attribution for IDH before updating and 94.6% afterward. Cross-domain feature-rank correlation increased from Spearman r=0.653 to r=0.816. The most attributed IDH variables after updating included historical and current pre-dialysis systolic blood pressure, pulse pressure, and prior IDH timing/rate summaries.

For IH, physiology/history variables already accounted for 96.8% of target-domain attribution before updating and 97.8% afterward. Cross-domain rank correlation changed from r=0.941 to r=0.928. These findings contradict a simple claim that IH prediction is dominated by treatment-context features. At most, they leave open the hypothesis that a small treatment-context representation should be preserved rather than globally aligned.

### Exploratory subgroup patterns

The existing subgroup artifact suggests heterogeneity in source-to-updated AUC differences across prior-risk and treatment-intensity strata. These estimates are not reported as inferential results because the current implementation resampled sessions instead of patients and used an invalid interaction standard-error calculation. They will remain supplementary and descriptive unless regenerated with patient-cluster bootstrap replicates and prespecified subgroup definitions.

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

The current ablation set is confounded by encoder architecture and incomplete seed matching. The mechanism-based feature grouping was manually specified. SHAP and latent analyses were exploratory and performed under different initialization seeds. Existing subgroup inference is invalid and has been withheld. Missing values were median imputed without missingness indicators. Data-quality flags remain unresolved. Finally, exact frozen processed cohorts are not currently reproducible from the checked-in CSV files.

## Conclusion

Target-center updating produced stable held-out discrimination for IDH and IH under substantial cross-center outcome shift. The current evidence supports the overall updating workflow and motivates outcome-specific representation hypotheses, but it does not establish that mechanism-aware alignment is superior to simpler updating. Confirmatory matched ablations, exact data restoration, prediction-level clinical evaluation, and complete reporting metadata are required before submission.

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

## Tables and Figures

- Table 1: Source and target cohort characteristics under the dual-binary contract.
- Table 2: Main held-out discrimination, precision-recall, Brier score, and paired differences.
- Table 3: Architecture-matched, seed-matched ablation results after confirmatory runs.
- Figure 1: Study design, cohort shift, and patient-level target split.
- Figure 2: Outcome-specific two-branch updating framework.
- Figure 3: Held-out performance and five-seed stability.
- Figure 4: Endpoint-specific latent and attribution changes, with metric-specific interpretation.
- Supplementary Figure 1: Calibration and decision curves after prediction artifacts are restored.
- Supplementary Figure 2: Patient-cluster subgroup analyses after statistical correction.

See `docs/visualization_plan.md` and `docs/evidence_audit.md` for the artifact mapping and submission gates.
