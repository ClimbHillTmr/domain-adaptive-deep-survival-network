# Outcome-Specific Target-Center Updating for Hemodynamic Event Prediction

This repository contains a two-center, session-level framework for predicting intradialytic hypotension (IDH) and intradialytic hypertension (IH) at the start of a hemodialysis session.

The active study uses labeled target-center data for model updating, validation, calibration, and held-out testing. It is therefore a **target-center updating study**, not zero-shot external validation. Earlier survival/CDAN-GSN components remain in the repository as legacy code and are not part of the current scientific claim.

## Current Evidence Status

The existing artifacts support three conclusions:

1. Endpoint burden differs substantially between the source and target centers.
2. The complete target-center updating procedure improves held-out target performance over the corresponding source MLP, particularly for IDH.
3. Updated-model discrimination is stable across five initialization seeds.

The existing artifacts do **not** yet establish that outcome-specific CORAL is superior to fine-tuning or global CORAL. Those methods currently differ in encoder architecture and are not all available under matched seeds. Mechanism-specific superiority remains a confirmatory hypothesis.

See [the evidence audit](docs/evidence_audit.md), [working manuscript](docs/manuscript_draft.md), and [visualization plan](docs/visualization_plan.md) for the claim-to-artifact mapping and submission blockers.

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
| IDH | Source MLP | 0.7189 (0.6966-0.7404) | 0.6752 | 0.2606 |
| IDH | Updated MLP | 0.8370 (0.8226-0.8520) | 0.7814 | 0.1567 |
| IDH | Target-local logistic | 0.8353 (0.8207-0.8504) | 0.7800 | 0.1576 |
| IH | Source MLP | 0.8600 (0.8439-0.8751) | 0.5078 | 0.0826 |
| IH | Updated MLP | 0.8684 (0.8525-0.8827) | 0.5222 | 0.0792 |
| IH | Target-local logistic | 0.8678 (0.8524-0.8822) | 0.5172 | 0.0797 |

The paired updated-versus-source AUC differences were +0.1182 (95% CI 0.1041 to 0.1323) for IDH and +0.0084 (0.0049 to 0.0122) for IH. Updated MLP performance was not clearly different from target-local logistic regression for either endpoint.

Across five seeds, updated AUC was 0.8374 +/- 0.0004 for IDH and 0.8684 +/- 0.0002 for IH. Source-model AUC was much more variable, so source-to-updated delta must not be interpreted as the isolated effect of CORAL.

## Representation Analyses

Exploratory analyses suggest a larger update effect for IDH than IH:

- IDH RBF MMD decreased from 0.2764 to 0.1861 in seed 42, but linear MMD, Wasserstein distance, and covariance distance increased. Domain-classifier AUC remained near 1.0.
- IDH target physiology/history SHAP share increased from 60.0% to 94.6% in seed 2024, and cross-center feature-rank Spearman correlation increased from 0.653 to 0.816.
- IH attribution was already physiology dominated (96.8% before and 97.8% after), which does not support a claim that treatment features dominate IH prediction.

These are descriptive model diagnostics, not causal mechanism estimates.

## Repository Layout

```text
conf/                         Frozen binary configurations
data/processed/               Processed cohorts; current files do not match frozen-run hashes
docs/                         Manuscript, evidence audit, and visualization plan
experiments/audit/            Feature allowlist and binary preflight
experiments/final_results/    Frozen aggregate binary-study artifacts
figures/manuscript_draft/     Aggregate-only draft figures generated by the script
scripts/                      Evaluation and figure utilities
src/data/                     Patient-level split and source-fitted preprocessing
src/train/                    Binary models and updating procedure
src/evaluate/                 Metrics and exploratory analyses
tests/                        Pipeline checks
```

The existing files under `tables/`, `figures/Main_Figures/`, and `figures/Submission_*` belong to the previous time-to-event study and must not be cited by the active manuscript.

## Static Audit and Training Gate

Run the static audit without fitting models:

```bash
python run_all.py
```

Training requires an explicit safety flag:

```bash
python run_all.py --train
```

Do not start training until the exact frozen-data provenance issue and prior-only history audit are resolved.

## Draft Manuscript Figures

Validate the aggregate inputs without writing files:

```bash
python scripts/build_manuscript_figures.py --check-only
```

Build draft PNG/PDF figures, an accessible source-data CSV, and alt text:

```bash
python scripts/build_manuscript_figures.py
```

The script reads frozen aggregate JSON artifacts only. It does not train models, rebuild cohorts, or generate prediction-dependent calibration and decision curves.

## Tests

```bash
python -m pytest tests/ -v
```

## Submission Blockers

- Restore or regenerate processed cohorts that match the SHA-256 hashes recorded by the frozen runs.
- Verify every allowed history predictor is prior-only on the exact frozen cohorts.
- Run architecture-matched, seed-matched ablations and compare absolute held-out performance.
- Restore prediction-level artifacts for calibration, ROC/PR curves, decision curves, and valid patient-cluster subgroup analysis.
- Resolve source data-quality flags and complete ethics, author, funding, conflict, data-governance, and reference metadata.

## License

MIT License
