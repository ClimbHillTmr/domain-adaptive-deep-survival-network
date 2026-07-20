# Evidence Audit for the Redesigned Binary Study

## 2026-07-19 Registry Gate

Eight historical binary runs are registered in `experiments/evidence_registry/run_registry.csv`, including hashes for configurations, datasets, patient splits, checkpoints, predictions, and evaluations. All eight use the same split manifest and have no patient overlap between fit, validation, and held-out groups.

The current processed source and target CSV hashes do not match the historical v1 hashes, so exact byte-level reconstruction of those runs remains unavailable. A separate versioned v2 cohort was rebuilt from fixed raw-file hashes, reproduces the historical cohort counts and patient split, and passed a strictly-prior history audit after correction of one tied timestamp pair. V2 may be used only for new server-side confirmatory validation and must not be mixed with historical v1 claims. Blank confirmatory and subgroup fields are pending states, not zero-valued findings.

Status: manuscript-development evidence map, not a submission sign-off

The paragraph-level manuscript claim registry currently passes: every quantitative paragraph has a unique claim locator and one or more hashed evidence artifacts, with registered run IDs where applicable. This verifies traceability, not missing ethics, reference, clinical-threshold, subgroup, or confirmatory-server evidence.

Audit date: 2026-07-19

Scope: registered historical artifacts plus the separately audited confirmatory-v2 data contract; no v2 result is yet accepted as manuscript evidence

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
| Target-center updating improves over the source MLP | Main run paired bootstrap: IDH +0.0424 (95% CI 0.0348 to 0.0515); IH +0.0165 (0.0112 to 0.0229) | Supported within the main run | Attribute to the complete update procedure, not CORAL alone |
| Updated discrimination is stable across initialization seeds | Five runs: IDH AUC 0.8371 +/- 0.0009; IH AUC 0.8687 +/- 0.0006 | Supported descriptively | Report individual seeds and mean +/- sample SD |
| The mechanism-aware method outperforms fine-tuning or global CORAL | Baselines use a single encoder; the mechanism run uses a two-branch encoder. The headline deltas also use different seeds | Not supported | Do not claim superiority until matched architecture-by-seed ablations exist |
| IDH alignment reduces latent domain discrepancy | Seed-2024 RBF MMD increased 0.1837 to 0.1888, while linear MMD and covariance distance decreased; domain AUC remained approximately 1.0 | Mixed | Report metric-specific change; do not claim broad domain invariance |
| IDH feature attribution shifts toward physiology | Main-run target SHAP group ratio 83.5% to 94.1%; cross-domain Spearman 0.766 to 0.817 | Exploratory support | Describe as model attribution, not biological mechanism |
| IH depends on treatment features | Target SHAP was 96.8% physiology before and 97.9% after updating | Contradicted | Do not make this claim |
| Preserving treatment variation is beneficial for IH | No matched intervention isolates preservation from architecture and target supervision | Hypothesis only | Present as design rationale requiring confirmatory ablation |
| Subgroup effects are statistically significant | Existing script resamples sessions rather than patients and derives interaction SEs from bootstrap summary SDs | Not valid for inference | Omit p-values; label subgroup patterns exploratory until re-analysis |
| Current results are reproducible from the checked-in processed data | Current CSV hashes differ from result provenance, and `history_IDH_rate` is absent from the current CSV headers although it was used by the frozen runs | Blocked | Do not label the package reproducible or submission-ready |

## Verified Main-Run Results

The main mechanism-aware artifact uses initialization seed 2024 and a fixed target test cohort of 29,866 sessions from 172 patients.

| Endpoint | Model | ROC AUC (95% patient-cluster CI) | PR AUC | Brier score |
|---|---|---:|---:|---:|
| IDH | Source MLP | 0.7951 (0.7768 to 0.8125) | 0.7463 | 0.2106 |
| IDH | Updated MLP | 0.8375 (0.8234 to 0.8522) | 0.7815 | 0.1566 |
| IDH | Target-local logistic | 0.8353 (0.8207 to 0.8504) | 0.7800 | 0.1576 |
| IH | Source MLP | 0.8516 (0.8347 to 0.8673) | 0.4907 | 0.0872 |
| IH | Updated MLP | 0.8681 (0.8522 to 0.8826) | 0.5234 | 0.0791 |
| IH | Target-local logistic | 0.8678 (0.8524 to 0.8822) | 0.5172 | 0.0797 |

The updated MLP did not clearly outperform the target-local logistic model. The stored paired patient-cluster bootstrap comparison was +0.0022 (95% CI -0.0003 to 0.0047) for IDH and +0.0003 (-0.0017 to 0.0022) for IH. These paired bootstrap estimates differ slightly from subtracting the two point estimates because the artifact records the mean bootstrap contrast.

## Five-Seed Stability

| Endpoint | Source MLP AUC, mean +/- SD | Updated MLP AUC, mean +/- SD | Paired delta, mean +/- SD |
|---|---:|---:|---:|
| IDH | 0.7766 +/- 0.0356 | 0.8371 +/- 0.0009 | 0.0606 +/- 0.0347 |
| IH | 0.8530 +/- 0.0065 | 0.8687 +/- 0.0006 | 0.0157 +/- 0.0069 |

The near-zero updated-model SD and large source-model SD indicate that the delta is driven partly by unstable source pretraining. This supports robustness after target updating, but it weakens any claim that delta magnitude measures the effect of a specific alignment rule.

## Submission Blockers

1. Preserve the unavailable exact historical v1 hashes (`fddb318b...` and `0e1e35ef...`) as a reproducibility limitation rather than silently substituting v2 files.
2. Use only the audited v2 hashes in `data/confirmatory_v2/confirmatory_data_manifest.json` for new server confirmatory runs.
3. Complete the locked five-model, five-seed server set and pass the read-only acceptance audit before using any v2 result.
4. Compare methods using absolute held-out AUC and paired patient-level differences on the same predictions, not improvement from method-specific source models.
5. Main-run prediction-level artifacts are available under `experiments/binary_results/`; use them for ROC, PR, and calibration. Prespecify clinical thresholds before decision-curve analysis and correct subgroup resampling before inference.
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
