# Visualization Plan for the Dual-Binary Manuscript

Status: draft design specification

Scope: current dual-binary evidence only

## Visual Narrative

The main figure sequence should answer four questions in order:

1. What shifted between centers, and how was the target cohort separated without patient overlap?
2. What exactly does outcome-specific updating align and preserve?
3. How much did held-out performance change, and how stable was the updated model?
4. Did representation diagnostics move consistently with the proposed mechanism?

The figures should not imply that mechanism-aware alignment is superior until architecture-matched, seed-matched ablations exist.

## Main Figures

### Figure 1. Cross-center outcome shift and target-center study design

**Purpose:** Establish the clinical shift and the target-center updating design.

**Panels**

- A: Source and target cohort counts, shown as labeled patient/session blocks rather than a decorative flowchart.
- B: Grouped bars for IDH and IH event prevalence at each center, with counts printed directly.
- C: Target patient split into update, validation, and held-out test sets, with patients and sessions displayed together.

**Inputs**

- `experiments/audit/binary_data_preflight.json`
- `experiments/final_results/main_mechanism_aware/split_manifest.json`

**Required caption caveat:** The target test set is held out from updating and calibration, but the study is not zero-shot external validation because labeled target data were used.

### Figure 2. Outcome-specific two-branch updating framework

**Purpose:** Make the model intervention auditable.

**Panels**

- A: Session-start features split into physiology/history (20 features) and treatment context (4 features).
- B: IDH path: align the physiology/history representation with CORAL; retain treatment context.
- C: IH path: align the treatment-context representation with CORAL; retain physiology/history.
- D: Pooled supervised prediction loss plus branch-specific CORAL loss, followed by target-validation Platt calibration.

**Design rule:** Use line style and labels in addition to color. Solid blue means aligned; dashed charcoal means retained without CORAL. Do not use brain, hospital, or AI stock icons.

**Required caption caveat:** Feature grouping is clinically prespecified and has not been independently validated as a biological decomposition.

### Figure 3. Updating improves and stabilizes held-out target performance

**Purpose:** Show the strongest supported quantitative result without an invalid method-superiority claim.

**Panels**

- A: Forest/dot plot of source MLP, updated MLP, and target-local logistic AUC with patient-cluster 95% CIs for IDH and IH in the seed-2024 run.
- B: Individual points for source and updated MLP AUC across the five seeds. Connect observations from the same seed; report mean and SD in text.
- D: PR AUC for the seed-2024 run. Brier scores remain in the main results table rather than being mixed onto a second axis.

**Inputs**

- `experiments/final_results/main_mechanism_aware/evaluation.json`
- `experiments/final_results/multiseed_seed*/evaluation.json`

**Do not show:** A four-method delta bar chart. Source baselines differ by architecture and seed, so that visual exaggerates a non-isolated effect.

### Figure 4. Representation changes are endpoint and metric dependent

**Purpose:** Test the proposed mechanism rather than decorate it.

**Panels**

- A: IDH and IH RBF MMD before and after updating, with seed 42 stated in the panel title.
- B: Domain-classifier AUC before and after updating. Include a reference line at 0.5 and show that IDH remains near 1.0.
- C: Target physiology/history attribution proportion before and after updating, with seed 2024 and `n=400` SHAP samples stated.
- D: Cross-domain feature-rank Spearman correlation before and after updating.

**Inputs**

- `experiments/final_results/multiseed_seed42/latent_analysis.json`
- `experiments/final_results/main_mechanism_aware/shap_analysis_full.json`

**Required caption caveat:** Panels combine separate prespecified runs and are descriptive. SHAP attribution is not a causal mechanism estimate. RBF MMD did not agree with every alternative discrepancy metric.

## Supplementary Figures

### Figure S1. Calibration and clinical utility

Generate only after the exact prediction-level artifact is restored. Include calibration curves with intercept/slope, Brier score, and decision curves over prespecified clinically relevant thresholds. Do not infer calibration from AUC or the aggregate Brier score alone.

### Figure S2. Architecture-matched ablations

Show absolute held-out AUC distributions across matched seeds for dual-branch no alignment, global CORAL, outcome-specific CORAL, and random partition. Use paired seed dots and patient-level paired intervals where available.

### Figure S3. Exploratory subgroup heterogeneity

Generate only after patient-cluster bootstrap re-analysis. Display subgroup-specific absolute AUC for both models and paired deltas. Avoid dichotomizing continuous variables unless cut points are clinically justified and prespecified.

## Artifact Map and Gates

| Visual element | Current artifact | Ready now? | Gate |
|---|---|---:|---|
| Cohort/session counts | Binary preflight | Yes | Label artifact date and contract |
| Target split | Main split manifest | Yes | Verify exact frozen data restoration before submission |
| Event prevalence | Binary preflight | Yes | Descriptive only |
| Main AUC/CI | Main evaluation JSON | Yes | State seed and patient-cluster bootstrap |
| Five-seed stability | Five evaluation JSON files | Yes | Show individual seeds |
| Method superiority | Current baseline JSON files | No | Matched architecture and seeds |
| Latent RBF MMD/domain AUC | Seed-42 latent JSON | Exploratory | Report conflicting metrics |
| SHAP group proportions | Main SHAP JSON | Exploratory | State sample size and manual grouping |
| Calibration/ROC/PR curves | Prediction CSV | No | Restore hash-matched predictions |
| DCA | Prediction CSV plus threshold protocol | No | Prespecify thresholds |
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
