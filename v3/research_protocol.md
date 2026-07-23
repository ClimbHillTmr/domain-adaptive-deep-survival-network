# V3 Research Protocol

Protocol status: draft before data and hypothesis lock. No training is authorized.

## Study objective

To determine whether observed cross-center clinical variation is heterogeneous in magnitude and predictive transportability, and whether a prespecified clinical feature partition improves target-center prediction when used for selective representation alignment.

The study does not assume that physiology is invariant, treatment is center-specific, or domain discrepancy is harmful. These are operational hypotheses that may be rejected.

## Study setting

The candidate data comprise retrospective hemodialysis sessions from one source center and one target center. V3 will use a new versioned data contract built from independently hash-verified raw and processed files. Previous V1/V2 predictions, evaluations, and test-driven interpretations are excluded from V3 confirmatory evidence.

## Hypotheses

### H1: observed shift heterogeneity

Prespecified clinical features differ in the magnitude, direction, missingness, and domain predictability of their source-target shifts.

Evidence:

- feature-level standardized location and distribution differences;
- Wasserstein distance and PSI for interpretable univariate shift;
- patient-cluster uncertainty;
- group-level summaries under a clinically reviewed ontology;
- univariate or tightly controlled feature-wise domain prediction with patient-grouped validation.

Falsification or limitation:

- shifts are small and similar after accounting for uncertainty;
- apparent group differences depend on one metric or a few outliers;
- clinical groups are not more informative than alternative groupings;
- no measurement/process variables are available, preventing evaluation of that proposed component.

H1 can establish observed heterogeneity. With only two centers, it cannot causally identify population, measurement, policy, or disease-mechanism shift.

### H2: selective alignment benefit

A clinically prespecified selective alignment strategy improves held-out target prediction over architecture-matched no alignment, dual-encoder global alignment, and size-matched random partitions.

Primary evidence:

- paired target-test difference in the prespecified discrimination metric;
- patient-cluster confidence interval;
- seed-aggregated effect across all locked seeds;
- empirical comparison with multiple size-matched random partitions;
- no material degradation in calibration or Brier score.

Falsification:

- selective alignment is indistinguishable from or worse than dual no alignment;
- selective alignment is indistinguishable from or worse than dual global alignment;
- clinical partitions do not exceed the random-partition null distribution;
- benefits depend on a selected seed or unregistered hyperparameter search.

### H3: outcome-specific strategy

IDH and IH may favor different prespecified alignment partitions. Both physiology and treatment/context alignment will therefore be tested for both outcomes; the favored mapping is not encoded into the training plan.

Evidence:

- prespecified endpoint-by-alignment contrast or interaction;
- multiplicity control across endpoints and partition contrasts;
- consistent seed-level direction and patient-cluster uncertainty;
- representation probes that are concordant with, but do not replace, predictive evidence.

Falsification:

- the same strategy performs similarly for both outcomes;
- outcome-specific contrasts are unstable across seeds;
- apparent differences arise only after post-hoc feature regrouping;
- representation attribution differs without corresponding held-out predictive evidence.

## Study phases

### Scientific Audit: question-to-evidence specification

Before data preflight or model execution, require every scientific question to resolve to an Evidence Tree node with explicit inputs, estimands, statistical unit, planned output, interpretation boundary, and falsification criterion. Every experiment ID must map to exactly one evidence node. Structural audit success is necessary but does not authorize training.

Models are indexed under the evidence question they address. The project is not considered complete merely because every model has run; each evidence node must instead be resolved as supported, not supported, blocked, or outside the identifiable scope.

### Phase 0: contract and governance

Freeze raw and processed identities, cohort rules, endpoints, feature ontology, preprocessing, patient split, code, environments, primary contrasts, seeds, and multiplicity. Resolve data-quality gates before training.

### Phase 1: observed shift characterization

Analyze source versus target using development-eligible data only. Report feature distributions, missingness, standardized effects, Wasserstein distance, PSI, and feature-wise domain predictability. MMD is reported for prespecified multivariate feature sets rather than mislabeled as a feature-wise metric.

### Phase 2: natural transportability

Train source-only models or capacity-controlled probes for each feature tier and evaluate source validation and target evaluation data. The primary quantity is absolute performance change. A normalized excess-AUC index,

`(AUC_target - 0.5) / (AUC_source - 0.5)`,

may be reported secondarily when the denominator is sufficiently separated from zero. Raw `AUC_target / AUC_source` is not the primary index because it obscures the chance baseline. PR-AUC is interpreted relative to endpoint prevalence and is not compared by a naive cross-center ratio.

### Phase 3: alignment experiment

Run the locked model family under matched splits, preprocessing, sampling, optimizer, training budget, seeds, capacity targets, and evaluation. The primary mechanism contrast is selective dual alignment versus dual global alignment; selective versus dual no alignment and the random-partition distribution are co-required falsification contrasts.

### Phase 4: representation validation

Evaluate latent MMD, Wasserstein summaries, covariance distance, domain classification, outcome probes, and attribution stability before and after updating. Representation metrics are supportive. Predictive preservation and calibration are required for a transportability interpretation.

### Phase 5: clinical utility

Proceed only after predictive evidence exists. Lock intended use, alert response, clinical thresholds, false-alert consequences, and missed-event consequences before evaluating an untouched temporal or external cohort. Report decision curves and alert burden with patient-cluster uncertainty.

## Bias controls

- Patient-level separation for every split.
- Strictly prior history; tied timestamps share history from earlier timestamps only.
- Source-development-fitted preprocessing unless a different rule is prespecified and justified.
- No feature exclusion based on V1/V2 performance.
- No V3 test access for model, feature group, threshold, or hyperparameter selection.
- Multiple random partitions rather than one random control.
- Complete reporting of all seeds and primary contrasts.
- Negative results satisfy protocol completion.

## Confirmatory cohort requirement

The historical target test set has already influenced prior project development. A new temporal target cohort or an external center is required for fully confirmatory V3 clinical-utility and strong transportability claims. Repartitioning previously inspected patients does not recreate an untouched external test.
