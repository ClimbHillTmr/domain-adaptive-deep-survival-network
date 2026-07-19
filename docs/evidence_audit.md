# Evidence Audit for the Redesigned Binary Study

Status: manuscript-development evidence map, not a submission sign-off

Audit date: 2026-07-19

Scope: existing repository artifacts only; no training or data regeneration was run

## Current Study Contract

The active study is a two-center, session-level, dual-endpoint binary prediction study with target-center updating. It is not the legacy time-to-event/CDAN-GSN study represented by the current files under `tables/` and `figures/Submission_*`.

- Source cohort: 211,452 sessions from 1,628 patients in the binary preflight artifact.
- Target cohort: 74,947 sessions from 430 patients.
- Target update/validation/test split: 38,483/6,598/29,866 sessions from 219/39/172 mutually exclusive patients.
- IDH endpoint: baseline SBP minus intradialytic SBP >=30 mmHg, or intradialytic SBP <=90 mmHg.
- IH endpoint: intradialytic MAP minus baseline MAP >10 mmHg.
- Predictors: 24 variables intended to be available at session start.
- Preprocessing: source-training median imputation and source-training standardization.
- Evaluation: held-out target patients with 1,000 patient-cluster bootstrap replicates for the main metrics.

## Claim-to-Evidence Matrix

| Proposed claim | Repository evidence | Status | Manuscript wording |
|---|---|---|---|
| The centers differ in endpoint burden | `binary_data_preflight.json`: IDH 14.5% vs 38.5%; IH 28.4% vs 12.0% | Supported | State as observed cohort shift, not causal evidence |
| Target-center updating improves over the source MLP | Main run paired bootstrap: IDH +0.1182 (95% CI 0.1041 to 0.1323); IH +0.0084 (0.0049 to 0.0122) | Supported within the main run | Attribute to the complete update procedure, not CORAL alone |
| Updated discrimination is stable across initialization seeds | Five runs: IDH AUC 0.8374 +/- 0.0004; IH AUC 0.8684 +/- 0.0002 | Supported descriptively | Report individual seeds and mean +/- sample SD |
| The mechanism-aware method outperforms fine-tuning or global CORAL | Baselines use a single encoder; the mechanism run uses a two-branch encoder. The headline deltas also use different seeds | Not supported | Do not claim superiority until matched architecture-by-seed ablations exist |
| IDH alignment reduces latent domain discrepancy | Seed-42 RBF MMD 0.2764 to 0.1861, but linear MMD, Wasserstein distance, and covariance distance increased; domain AUC remained approximately 1.0 | Mixed | Report metric-specific change; do not claim broad domain invariance |
| IDH feature attribution shifts toward physiology | Main-run target SHAP group ratio 60.0% to 94.6%; cross-domain Spearman 0.653 to 0.816 | Exploratory support | Describe as model attribution, not biological mechanism |
| IH depends on treatment features | Target SHAP was 96.8% physiology before and 97.8% after updating | Contradicted | Do not make this claim |
| Preserving treatment variation is beneficial for IH | No matched intervention isolates preservation from architecture and target supervision | Hypothesis only | Present as design rationale requiring confirmatory ablation |
| Subgroup effects are statistically significant | Existing script resamples sessions rather than patients and derives interaction SEs from bootstrap summary SDs | Not valid for inference | Omit p-values; label subgroup patterns exploratory until re-analysis |
| Current results are reproducible from the checked-in processed data | Current CSV hashes differ from result provenance, and `history_IDH_rate` is absent from the current CSV headers although it was used by the frozen runs | Blocked | Do not label the package reproducible or submission-ready |

## Verified Main-Run Results

The main mechanism-aware artifact uses initialization seed 2024 and a fixed target test cohort of 29,866 sessions from 172 patients.

| Endpoint | Model | ROC AUC (95% patient-cluster CI) | PR AUC | Brier score |
|---|---|---:|---:|---:|
| IDH | Source MLP | 0.7189 (0.6966 to 0.7404) | 0.6752 | 0.2606 |
| IDH | Updated MLP | 0.8370 (0.8226 to 0.8520) | 0.7814 | 0.1567 |
| IDH | Target-local logistic | 0.8353 (0.8207 to 0.8504) | 0.7800 | 0.1576 |
| IH | Source MLP | 0.8600 (0.8439 to 0.8751) | 0.5078 | 0.0826 |
| IH | Updated MLP | 0.8684 (0.8525 to 0.8827) | 0.5222 | 0.0792 |
| IH | Target-local logistic | 0.8678 (0.8524 to 0.8822) | 0.5172 | 0.0797 |

The updated MLP did not clearly outperform the target-local logistic model: IDH AUC difference +0.0018 (95% CI -0.0007 to 0.0042) and IH +0.0006 (-0.0009 to 0.0022).

## Five-Seed Stability

| Endpoint | Source MLP AUC, mean +/- SD | Updated MLP AUC, mean +/- SD | Paired delta, mean +/- SD |
|---|---:|---:|---:|
| IDH | 0.7444 +/- 0.0460 | 0.8374 +/- 0.0004 | 0.0930 +/- 0.0460 |
| IH | 0.8418 +/- 0.0178 | 0.8684 +/- 0.0002 | 0.0266 +/- 0.0178 |

The near-zero updated-model SD and large source-model SD indicate that the delta is driven partly by unstable source pretraining. This supports robustness after target updating, but it weakens any claim that delta magnitude measures the effect of a specific alignment rule.

## Submission Blockers

1. Restore or regenerate the exact processed cohorts whose SHA-256 hashes match the frozen evaluation artifacts. The frozen source and target hashes begin `fddb318b...` and `0e1e35ef...`; the current checked-in CSV hashes begin `82a75ca4...` and `0bed70a5...`.
2. Re-run a current-contract history audit limited to the 24 allowed predictors and verify prior-only construction for every patient. The legacy audit is not valid for this contract.
3. Run matched seeds for: single-encoder fine-tuning, single-encoder global CORAL, dual-branch no-alignment, dual-branch global CORAL, outcome-specific CORAL, and a size-matched random partition control.
4. Compare methods using absolute held-out AUC and paired patient-level differences on the same predictions, not improvement from method-specific source models.
5. Restore prediction-level artifacts to produce calibration, ROC/PR, decision-curve, and valid patient-cluster subgroup analyses.
6. Resolve data-quality flags visible in the frozen preprocessing metadata, including source features with zero variance and extreme source pre-dialysis temperature values.
7. Supply ethics approval/waiver, consent handling, author list, affiliations, funding, conflicts, data governance, and verified references.

## Files Excluded from the New Paper

Until regenerated under the dual-binary contract, the following are legacy evidence and must not be cited by the new manuscript:

- `tables/table1_baseline.csv`
- `tables/table2_performance.csv`
- `tables/table3_pvalues.csv`
- `tables/local_cox_threshold_summary.csv`
- `figures/Main_Figures/*`
- `figures/Submission_Main_Figures/*`
- `figures/Submission_Supplementary_Figures/*`
- `experiments/audit/result_consistency_report.json`
- `experiments/audit/data_manifest.json`
- `experiments/audit/split_audit.json`
- `experiments/audit/history_feature_audit.csv`
