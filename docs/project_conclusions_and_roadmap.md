# Current Project Conclusions and Execution Roadmap

Status: conclusion freeze for the 2026-07-19 binary-study artifacts

## Executive Conclusion

The project now supports a target-center updating paper, but not yet a mechanism-superiority paper.

The strongest result is that updating with labeled target-center data improves discrimination and calibration on mutually exclusive held-out target patients. Updated performance is highly stable across initialization seeds and is approximately equal to target-local logistic regression. Existing results do not show a material advantage for outcome-specific CORAL over simpler update strategies.

## Conclusions by Evidence Tier

### Tier 1: suitable for the abstract and main results

1. Source and target centers have large, opposing endpoint-prevalence shifts: IDH 14.5% versus 38.5%, and IH 28.4% versus 12.0%.
2. The seed-2024 complete update procedure improves held-out AUC over the source MLP:
   - IDH: 0.7951 to 0.8375; paired difference 0.0424 (95% CI 0.0348 to 0.0515).
   - IH: 0.8516 to 0.8681; paired difference 0.0165 (95% CI 0.0112 to 0.0229).
3. Updated AUC is stable across five seeds:
   - IDH: 0.8371 +/- 0.0009.
   - IH: 0.8687 +/- 0.0006.
4. Updated probability estimates are substantially better calibrated than source-MLP estimates on the held-out target set. Updated MLP calibration intercept/slope are 0.05/1.10 for IDH and 0.05/0.97 for IH.
5. Updated MLP is not clearly better than target-local logistic regression for either endpoint: paired AUC difference +0.0022 (95% CI -0.0003 to 0.0047) for IDH and +0.0003 (-0.0017 to 0.0022) for IH.

### Tier 2: exploratory mechanism results

1. IDH target attribution shifts toward physiology/history variables (83.5% to 94.1%).
2. Cross-center feature-rank agreement increases for IDH (Spearman 0.766 to 0.817) and IH (0.906 to 0.933).
3. Latent discrepancy measures move in different directions. Domain classification remains easy, especially for IDH, so general domain invariance is not achieved.
4. IH remains overwhelmingly physiology/history dominated in SHAP attribution; treatment-feature dominance is contradicted.

These statements describe model behavior. They do not establish causal physiology or prove that the selected alignment branch is optimal.

### Tier 3: claims that must not be made

- Outcome-specific CORAL is superior to fine-tuning or global CORAL.
- The model learned a domain-invariant representation.
- IH prediction is driven mainly by treatment variables.
- Existing subgroup interaction p-values are valid.
- The checked-in processed CSV files exactly reproduce the frozen runs.

## Interpretation of the Seed-2024 Method Runs

| Method artifact | IDH updated AUC | IH updated AUC | Interpretation |
|---|---:|---:|---|
| Fine-tuning | 0.8370 | 0.8696 | Single-encoder reference |
| Global CORAL | 0.8372 | 0.8694 | Single-encoder reference |
| `baseline_random_alignment` | 0.8378 | 0.8686 | Current config is dual-branch without CORAL; name requires correction |
| Outcome-specific CORAL | 0.8375 | 0.8681 | Dual-branch with endpoint-specific CORAL |

The differences are small and the architectures are not matched across all rows. The current table is a diagnostic landscape, not a confirmatory ranking.

## Next-Step Roadmap

### Phase 0: freeze the evidence contract

Goal: make every manuscript number traceable to one run and one artifact.

- Completed: mapped `final_results` aliases to eight historical `binary_results/<run_id>` directories.
- Completed: recorded hashes for configurations, predictions, checkpoints, evaluations, datasets, and the shared split.
- Completed: built a separate v2 contract from fixed raw hashes, corrected tied-time history construction, and verified identical cohort counts and patient split.
- Permanent disclosure: exact v1 processed bytes could not be recovered; v2 must not be substituted into historical v1 claims.
- Completed: audited `baseline_random_alignment` as dual-branch updating without CORAL; retain the legacy alias only for provenance.

Gate: no manuscript number may lack an artifact path, run ID, seed, and data hash.

### Phase 1: fair method isolation

Goal: determine whether outcome-specific CORAL adds value beyond target supervision and architecture.

Required matched models:

1. Dual-branch target updating without alignment.
2. Dual-branch global CORAL.
3. Dual-branch outcome-specific CORAL.
4. Dual-branch size-matched random feature partition with CORAL.
5. Single-encoder fine-tuning and global CORAL as reference models.

All models must use identical patient splits, preprocessing, seeds, optimization budgets, calibration, and held-out scoring. Compare absolute AUC/PR AUC/Brier and paired patient-cluster differences. The locked five-model by five-seed package is ready; execution and acceptance remain pending on the compute server.

Gate: promote a mechanism claim only if the matched outcome-specific model has a reproducible advantage with paired uncertainty.

### Phase 2: clinical performance characterization

Goal: move beyond discrimination.

- Generate ROC, precision-recall, and calibration figures from the restored prediction artifact.
- Completed: report calibration intercept, slope, Brier score, and probability-decile reliability with patient-cluster uncertainty.
- Define clinically meaningful threshold ranges with nephrology input before DCA.
- Completed descriptively at validation-derived thresholds: report sensitivity, specificity, PPV, NPV, alert burden, and events captured with patient-cluster uncertainty; do not call these clinical operating thresholds.
- Rebuild subgroup analyses with patient-cluster bootstrap and prespecified strata.

Gate: do not present DCA or subgroup inference until thresholds and resampling units are frozen.

### Phase 3: manuscript freeze

Goal: create a submission-consistent evidence package.

- Synchronize abstract, methods, results, tables, captions, README, and evidence audit.
- Replace all author, ethics, funding, conflict, data-governance, and reference placeholders.
- Verify every DOI/PMID.
- Complete TRIPOD+AI and journal-specific checklists.
- Export final vector PDF, 300/600 dpi raster figures, source-data CSVs, and alt text.

## Enhanced Visualization Sequence

### Main figures available now

1. Cohort shift and patient-level design.
2. Outcome-specific two-branch model schematic.
3. Main held-out performance and seed stability.
4. Representation diagnostics with explicit mixed evidence.
5. ROC, precision-recall, and calibration using real held-out predictions.

### Supplementary figures after evidence gates

1. Architecture-matched method comparison across seeds.
2. Decision curves and threshold workload after clinical threshold prespecification.
3. Patient-cluster subgroup forest plots.
4. Feature-level SHAP ranking and change map with uncertainty or repeated sampling.
