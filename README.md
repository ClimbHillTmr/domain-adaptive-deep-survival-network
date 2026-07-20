# Outcome-Specific Target-Center Updating for Hemodynamic Event Prediction

This repository contains a two-center, session-level framework for predicting intradialytic hypotension (IDH) and intradialytic hypertension (IH) at the start of a hemodialysis session.

The active study uses labeled target-center data for model updating, validation, calibration, and held-out testing. It is therefore a **target-center updating study**, not zero-shot external validation. Earlier survival/CDAN-GSN components remain in the repository as legacy code and are not part of the current scientific claim.

## Current Evidence Status

The existing artifacts support three conclusions:

1. Endpoint burden differs substantially between the source and target centers.
2. The complete target-center updating procedure improves held-out target performance over the corresponding source MLP, particularly for IDH.
3. Updated-model discrimination is stable across five initialization seeds.

The existing artifacts do **not** yet establish that outcome-specific CORAL is superior to fine-tuning or global CORAL. Those methods currently differ in encoder architecture and are not all available under matched seeds. Mechanism-specific superiority remains a confirmatory hypothesis.

See [the frozen registry](experiments/evidence_registry), [evidence audit](docs/evidence_audit.md), [working manuscript](docs/manuscript_draft.md), and [visualization plan](docs/visualization_plan.md) for the claim-to-artifact mapping and submission blockers.

## Reproducibility Pipeline

Historical v1 evidence and the new confirmatory v2 protocol are separate contracts. The eight historical runs remain frozen for the current target-center-updating claims. The corrected v2 cohorts reproduce their cohort counts and patient split while fixing a tied-time prior-history issue; v2 is reserved for new server validation and must not be mixed numerically with v1.

```bash
python scripts/freeze_binary_evidence.py
python scripts/run_confirmatory_ablation.py --prepare-configs
python scripts/run_confirmatory_ablation.py --plan
python scripts/run_confirmatory_ablation.py --train --execution-context server_confirmatory
python scripts/verify_server_confirmatory_results.py --require-complete
python scripts/build_confirmatory_tables.py
python scripts/build_clinical_utility.py
python scripts/export_exploratory_evidence.py
python scripts/build_subgroup_registry.py
python scripts/run_subgroup_analysis.py --run
python scripts/build_manuscript_tables.py
python scripts/build_manuscript_claim_registry.py --require-complete
python scripts/audit_submission_package.py
```

Training is intended for the compute server, not the local workstation. The `--train` command proceeds only after exact-data, patient-split, and temporal-leakage gates pass and after the explicit `server_confirmatory` execution marker is supplied. The confirmatory table additionally rejects results without a recorded server hostname and CUDA device. Primary machine-readable outputs are under `experiments/evidence_registry/`, `experiments/confirmatory_results/`, `clinical_utility_report/`, and `subgroup_analysis_patient_bootstrap.csv`.

Use `--plan` on the server before `--train` to validate and list the complete five-model by five-seed grid without fitting or rewriting anything. The default execution policy is one process on one CUDA GPU, serially across five seed blocks; a cyclic balanced order places every model in each within-block execution position once. Rerunning the same launcher skips configurations already completed with valid server provenance.

Before server execution, verify `experiments/confirmatory_configs/server_package_manifest.json`; it locks the v2 datasets, feature allowlist, patient split, analysis code, dependencies, and all 25 configurations. Accepted server results must be appended to the central run registry with `scripts/register_server_confirmatory_runs.py` before confirmatory tables or Figure 2 can use them.

See the [server confirmatory runbook](docs/server_confirmatory_runbook.md) for locked hashes, synchronization boundaries, resumable execution, acceptance checks, and result-return instructions.

`scripts/audit_submission_package.py` writes a machine-readable deliverable matrix and keeps the overall submission gate closed while server evidence, clinical prespecifications, subgroup results, or mandatory manuscript metadata remain missing. Use `--require-ready` only for the final release check.

The manuscript claim registry requires every paragraph containing quantitative information to map to a unique locator, registered run IDs where applicable, and hashed source artifacts. Editing or adding a quantitative paragraph without updating `conf/manuscript_claims.yaml` closes the manuscript gate.

The subgroup command is intentionally non-runnable with the template protocol. Complete and lock the [subgroup prespecification form](docs/subgroup_prespecification_form.md), bind it to an accepted confirmatory-v2 server run, and then execute it. Historical v1 predictions must not be joined to v2 feature rows.

## Study Contract

| Item | Frozen binary-study definition |
|---|---|
| Source cohort | 211,452 sessions; 1,628 patients |
| Target cohort | 74,947 sessions; 430 patients |
| Target held-out test | 29,866 sessions; 172 patients |
| IDH | Baseline SBP - intradialytic SBP >=30 mmHg, or intradialytic SBP <=90 mmHg |
| IH | Intradialytic MAP - baseline MAP >10 mmHg |
| Prediction time | Start of dialysis session |
| Predictors | 24 demographics, pre-dialysis variables, and prior-session summaries |
| Split | Patient level; fixed seed 20260715 |
| Calibration | Platt scaling on validation patients |
| Uncertainty | 1,000 patient-cluster bootstrap replicates |

Outcome rates in the binary preflight artifact were 14.5% versus 38.5% for IDH and 28.4% versus 12.0% for IH at the source and target centers, respectively.

## Model Design

The mechanism-aware MLP contains two encoders:

- **Physiology/history branch (20 variables):** pre-dialysis vital signs, blood-pressure summaries, and prior event burden.
- **Treatment-context branch (4 variables):** volume overload, historical maximum ultrafiltration volume, pre-dialysis weight minus dry weight, and historical mean ultrafiltration rate.

For IDH, CORAL is applied to the physiology/history representation. For IH, CORAL is applied to the treatment-context representation. The other branch remains available to the prediction head without CORAL. The update phase uses pooled source and labeled target sessions with focal prediction loss and a CORAL weight of 0.01.

This grouping is a prespecified clinical design choice, not a validated biological decomposition.

## Main Held-Out Results

The main mechanism-aware artifact uses initialization seed 2024.

| Endpoint | Model | ROC AUC (95% patient-cluster CI) | PR AUC | Brier |
|---|---|---:|---:|---:|
| IDH | Source MLP | 0.7951 (0.7768-0.8125) | 0.7463 | 0.2106 |
| IDH | Updated MLP | 0.8375 (0.8234-0.8522) | 0.7815 | 0.1566 |
| IDH | Target-local logistic | 0.8353 (0.8207-0.8504) | 0.7800 | 0.1576 |
| IH | Source MLP | 0.8516 (0.8347-0.8673) | 0.4907 | 0.0872 |
| IH | Updated MLP | 0.8681 (0.8522-0.8826) | 0.5234 | 0.0791 |
| IH | Target-local logistic | 0.8678 (0.8524-0.8822) | 0.5172 | 0.0797 |

The paired updated-versus-source AUC differences were +0.0424 (95% CI 0.0348 to 0.0515) for IDH and +0.0165 (0.0112 to 0.0229) for IH. Updated MLP performance was not clearly different from target-local logistic regression for either endpoint.

Across five seeds, updated AUC was 0.8371 +/- 0.0009 for IDH and 0.8687 +/- 0.0006 for IH. Source-model AUC was much more variable, so source-to-updated delta must not be interpreted as the isolated effect of CORAL.

## Representation Analyses

Exploratory analyses suggest a larger update effect for IDH than IH:

- In the seed-2024 IDH representation, RBF MMD increased slightly from 0.1837 to 0.1888 while linear MMD and covariance distance decreased; domain-classifier AUC remained high (0.9980 to 0.9905).
- IDH target physiology/history SHAP share increased from 83.5% to 94.1%, and cross-center feature-rank Spearman correlation increased from 0.766 to 0.817.
- IH attribution was already physiology dominated (96.8% before and 97.9% after), which does not support a claim that treatment features dominate IH prediction.

These are descriptive model diagnostics, not causal mechanism estimates.

## Repository Layout

```text
conf/                         Frozen binary configurations
data/processed/               Processed cohorts; current files do not match frozen-run hashes
docs/                         Manuscript, evidence audit, and visualization plan
experiments/audit/            Feature allowlist and binary preflight
experiments/final_results/    Frozen aggregate binary-study artifacts
figures/manuscript_draft/     Aggregate-only draft figures generated by the script
figures/submission_staging/   Evidence-gated target figure sequence (currently 4/5)
scripts/                      Evaluation and figure utilities
src/data/                     Patient-level split and source-fitted preprocessing
src/train/                    Binary models and updating procedure
src/evaluate/                 Metrics and exploratory analyses
tests/                        Pipeline checks
```

The existing files under `tables/`, `figures/Main_Figures/`, and `figures/Submission_*` belong to the previous time-to-event study and must not be cited by the active manuscript.

The only active manuscript tables are under `tables/manuscript/`. Table 1 reports registered cohort and analysis allocation because a submission-grade baseline-characteristics artifact is unavailable; Table 2 reports registered held-out performance; Table 3 remains the server-pending confirmatory CSV.

## Static Audit and Training Gate

Run the static audit without fitting models:

```bash
python run_all.py
```

Training requires an explicit safety flag:

```bash
python run_all.py --train
```

Do not rerun or replace the historical v1 evidence. New confirmatory-v2 fitting is permitted only on the compute server under the locked runbook after its v2 data, history, split, and CUDA checks pass.

## Draft Manuscript Figures

Validate the aggregate inputs without writing files:

```bash
python scripts/build_manuscript_figures.py --check-only
```

Build draft PNG/PDF figures, an accessible source-data CSV, and alt text:

```bash
python scripts/build_manuscript_figures.py
```

The manuscript figure script reads registered frozen aggregate and prediction artifacts. It does not train models or rebuild cohorts. The separate clinical-utility package generates calibration, ECE, validation-threshold alert analyses with 1,000-replicate patient-cluster intervals, and descriptive DCA sensitivity curves.

The submission-staging Figure 1 is rendered natively as vector graphics from the frozen cohort preflight, patient split manifest, and registered historical configuration. Figure 2 is intentionally not generated until all 25 server-only confirmatory configurations pass acceptance.

Run `python scripts/build_supplementary_figures.py` to rebuild Supplementary Figure S1 from the five registered historical seed runs. Its vector PDF, PNG, 20-row source-data CSV, alt text, and provenance are written under `figures/submission_staging/`; no across-seed summary is treated as an inferential interval.

Supplementary Figure S2 is conditionally generated with `python scripts/build_subgroup_figure.py --build` only after the locked patient-cluster subgroup CSV and its provenance pass. The current six-row prespecification shell is deliberately rejected and produces no empty figure.

Run `python scripts/build_submission_supplement.py` to rebuild the evidence-linked supplementary methods package at `docs/submission_supplement.md`. It inventories the locked hyperparameters, seeds, patient-cluster bootstrap method, feature definitions, and subgroup protocol without fitting models or regenerating data; its source and output hashes are stored under `experiments/evidence_registry/`.

After building the development figures, run `python scripts/build_submission_staging_figures.py`. The staging package contains Figures 1, 3, 4, and 5 with source data, alt text, and hashes. Figure 2 is intentionally not created until all 25 architecture-matched server runs pass acceptance.

## Tests

```bash
python -m pytest tests/ -v
```

## Submission Blockers

- Preserve the historical v1 input-hash limitation as a reproducibility disclosure; do not claim exact byte-level reconstruction.
- Complete and accept all 25 architecture-matched v2 server runs before making mechanism-superiority claims.
- Obtain nephrology prespecification of clinically relevant thresholds before interpreting decision curves as clinical evidence.
- Prespecify subgroup cut points and minimum patient counts before patient-cluster interaction analysis.
- Resolve source data-quality flags and complete ethics, author, funding, conflict, data-governance, and reference metadata.

## License

MIT License
