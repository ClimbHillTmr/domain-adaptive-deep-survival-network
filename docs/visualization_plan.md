# Visualization Plan for the Dual-Binary Manuscript

Status: submission staging; Figures 1, 3, 4, and 5 generated, Figure 2 evidence-gated

Scope: current dual-binary evidence only

## Visual Narrative

The main figure sequence should answer five questions in order:

1. What shifted between centers, and how was the target cohort separated without patient overlap?
2. Does outcome-specific alignment add value under architecture-matched controls?
3. What operating performance and alert burden arise at registered validation-derived thresholds?
4. Did target updating improve ranking performance and probability calibration?
5. Did representation diagnostics move consistently with the proposed mechanism?

The figures should not imply that mechanism-aware alignment is superior until architecture-matched, seed-matched ablations exist.

## Main Figures

### Figure 1. Study design and outcome-specific target-center updating

**Purpose:** Establish the clinical shift and the target-center updating design.

**Panels**

- A: Source and target cohort counts, with patients and sessions labeled directly.
- B: Grouped bars for IDH and IH event prevalence at each center.
- C: Target patient split into update, validation, and held-out test sets, with patients and sessions displayed together.
- D: Registered IDH workflow, aligning the physiology/history branch while retaining treatment context.
- E: Registered IH workflow, aligning the treatment-context branch while retaining physiology/history.

**Inputs**

- `experiments/audit/binary_data_preflight.json`
- `experiments/final_results/main_mechanism_aware/split_manifest.json`
- `experiments/final_results/main_mechanism_aware/config.yaml`

**Current artifact:** Native vector PDF and 300 dpi PNG under `figures/submission_staging/`; the PDF contains embedded vector text and no raster image objects.

**Required caption caveat:** The target test set is held out from updating and calibration, but the study is not zero-shot external validation because labeled target data were used. The branch diagram describes the registered historical configuration and does not establish mechanism superiority.

### Figure 2. Architecture-matched ablation

**Purpose:** Isolate alignment strategy from architecture and target supervision.

This figure is not generated until all 25 server configurations pass acceptance. The upper panels show absolute held-out ROC AUC: pale lines connect the same initialization seed across all five strategies, colored markers identify strategies, and short horizontal ticks show descriptive means. The lower panels show the seed-specific, patient-paired ROC AUC difference between outcome-specific CORAL and each comparator with 95% patient-cluster intervals. PR AUC, Brier, and ECE remain in the complete confirmatory table. This makes the blocked comparison explicit without turning five seeds into an inferential confidence interval. Do not use a ranking bar chart or infer superiority from source-to-updated deltas.

### Figure 3. Registered clinical utility diagnostics

**Purpose:** Show sensitivity, specificity, PPV, NPV, event capture, and false-alert burden with patient-cluster uncertainty at fixed validation-derived thresholds.

**Panels**

- A/B: IDH and IH forest plots for sensitivity, specificity, PPV, and NPV across source MLP, updated MLP, and target-local logistic regression.
- C/D: IDH and IH grouped bars for events detected and false alerts per 1,000 sessions, with patient-cluster error bars.

**Inputs**

- `clinical_utility_report/alert_analysis.csv`
- `clinical_utility_report/provenance.json`

**Required caption caveat:** Thresholds were selected on validation patients using Youden's index. They are not clinically prespecified deployment thresholds, and Figure 3 does not establish net clinical benefit.

### Figure 4. Calibration and transportability analysis

**Purpose:** Show the complete predictive picture using real held-out target predictions.

**Panels**

- A/D: IDH and IH ROC curves for source MLP, updated MLP, and target-local logistic.
- B/E: Precision-recall curves with endpoint prevalence reference lines.
- C/F: Probability-decile calibration curves with descriptive calibration intercept and slope.

**Inputs**

- `experiments/binary_results/binary_20260719_044949_c6c955a30fc2/test_predictions.csv`
- Matching registered `evaluation.json`.

**Required caption caveat:** Curves are descriptive held-out diagnostics. Calibration intercept and slope have patient-cluster intervals in the clinical-utility table, although the plotted calibration curve itself is not interval-banded.

### Figure 5. Exploratory representation analysis

**Purpose:** Show that representation changes are endpoint and metric dependent rather than claiming domain invariance.

**Panels**

- A: IDH and IH RBF MMD before and after updating.
- B: Domain-classifier AUC before and after updating, with a chance line at 0.5.
- C: Target physiology/history SHAP proportion before and after updating (`n=400` per domain).
- D: Cross-center feature-rank Spearman correlation before and after updating.

**Inputs**

- `experiments/evidence_registry/exploratory_latent_metrics.csv`
- `experiments/evidence_registry/exploratory_shap_summary.csv`
- `experiments/evidence_registry/exploratory_provenance.json`

**Required caption caveat:** Panels are exploratory. SHAP attribution is not causal, discrepancy measures did not move uniformly, and center membership remained readily distinguishable.

## Supplementary Figures

### Figure S1. Seed-level performance stability

Generated under `figures/submission_staging/` from the five registered historical runs. It shows individual source and updated AUC values on common endpoint scales, with source-to-updated lines paired within run. The accompanying CSV contains 20 run/model/endpoint rows and registered evaluation hashes. Do not treat the across-seed range or sample SD as an inferential confidence interval, and do not interpret the line segments as an isolated alignment effect.

### Figure S2. Exploratory subgroup heterogeneity

Generate only after the locked protocol is bound to an accepted confirmatory-v2 server run and the patient-cluster analysis passes. Display subgroup-specific absolute updated-model AUC and updated-minus-source paired deltas with interaction estimates. Avoid dichotomizing continuous variables unless cut points are clinically justified and prespecified before target-test subgroup performance is inspected.

The conditional generator is `scripts/build_subgroup_figure.py --build`. It refuses the current blocked shell and requires 12 registered rows, one centrally registered outcome-specific v2 run, 1,000 valid patient-cluster replicates, the six prespecified subgroup domains, and matching subgroup provenance before creating S2.

## Artifact Map and Gates

| Visual element | Current artifact | Ready now? | Gate |
|---|---|---:|---|
| Cohort/session counts | Binary preflight | Yes | Label artifact date and contract |
| Target split | Registered v1 and audited v2 split manifests | Yes | Disclose v1 byte-level limitation; do not mix contracts |
| Event prevalence | Binary preflight | Yes | Descriptive only |
| Main AUC/CI | Main evaluation JSON | Yes | State seed and patient-cluster bootstrap |
| Five-seed stability | Five evaluation JSON files | Yes | Show individual seeds |
| Method superiority | Current baseline JSON files | No | Matched architecture and seeds |
| Latent RBF MMD/domain AUC | Seed-42 latent JSON | Exploratory | Report conflicting metrics |
| SHAP group proportions | Main SHAP JSON | Exploratory | State sample size and manual grouping |
| Calibration/ROC/PR curves | Main-run prediction CSV | Yes | Label as descriptive held-out diagnostics |
| Fixed validation-threshold alert metrics | Registered prediction CSV | Descriptive | Patient-cluster intervals complete; threshold is not clinically prespecified |
| DCA | Prediction CSV plus threshold protocol | No | Prespecify thresholds |

## Main Tables

- Table 1: registered cohort totals and patient-level source/target allocations from the historical v1 registry and split manifest. It is not a baseline-characteristics table.
- Table 2: registered seed-2024 held-out ROC AUC, PR AUC, Brier score, and explicitly directed paired AUC contrasts with patient-cluster intervals.
- Table 3: architecture-matched ablation shell; it remains empty until all accepted v2 server runs are returned.

Source and rendered Markdown versions are under `tables/manuscript/`, with hashes recorded in `tables/manuscript/provenance.json`.
| Subgroup inference | Current subgroup JSON | No | Patient-cluster re-analysis |

## Style System

- Canvas: white; no gradients, 3D effects, shadows, or decorative cards.
- Primary colors: physiology/history `#0072B2`; treatment context `#D55E00`; neutral/source `#6B7280`; updated model `#009E73`.
- Never rely on color alone: use direct labels, marker shapes, and solid/dashed line styles.
- Typography: Arial or Liberation Sans; minimum 8 pt at final print size.
- Main figure width: 180 mm for multi-panel figures; export both 300 dpi PNG and vector PDF.
- Axes: start bars at zero; use consistent AUC limits across endpoint panels; label directionality for Brier score.
- Uncertainty: show 95% intervals whenever available. Do not convert five seeds into inferential confidence intervals.
- Captions: state cohort, split, seed, resampling unit, number of replicates, and source artifact.
- Accessibility: provide alt text and a CSV source table for every generated quantitative figure.

## Legacy Figure Exclusion

Files currently under `figures/Main_Figures`, `figures/Submission_Main_Figures`, and `figures/Submission_Supplementary_Figures` represent the previous time-to-event study. They must not be relabeled or reused for the dual-binary manuscript.
