# V3 Statistical Analysis Plan

Status: primary estimand and core inference locked by `CLAUDE.md`; implementation verification and data/split freeze remain pending.

## Analysis unit

Patients are the independent sampling units. Sessions remain prediction instances, but confidence intervals, paired comparisons, calibration uncertainty, decision curves, and subgroup analyses use patient-cluster resampling. Patient overlap across train, update, validation, calibration, and test sets is prohibited.

Because frequent attenders contribute more sessions, primary session-weighted estimates must be accompanied by a patient-balanced sensitivity analysis.

## Shift characterization

For every prespecified feature, report direction, missingness, and effect size using unit-appropriate summaries. Wasserstein distance, PSI, KS statistic, and MMD are complementary descriptive measures; none identifies why a shift occurred.

- Transformations are fit on source development data only.
- Confidence intervals use patient-cluster bootstrap.
- KS p-values are not treated as evidence strength in this large repeated-session dataset.
- Multiple testing is controlled for feature-level inferential statements.
- Feature-group summaries are secondary to transparent feature-level results.
- A two-center comparison may support observed distribution heterogeneity, not a causal decomposition into population, measurement, policy, and disease mechanisms.

## Primary mechanism estimand

For endpoint `e` and locked seed `s`, define:

`delta(e,s) = AUC_target_test(primary selective strategy, e, s) - AUC_target_test(dual global CORAL, e, s)`.

The primary selective strategy is physiology-partition CORAL for IDH and treatment/context-partition CORAL for IH. This mapping is a falsifiable clinical hypothesis, not an invariance assumption. The endpoint estimand is the arithmetic mean of `delta(e,s)` over locked seeds 7, 13, 42, 99, and 2024.

IDH and IH are co-primary. Two-sided patient-cluster bootstrap p-values are adjusted using Holm at family-wise alpha 0.05. Strong cross-outcome mechanism-aware wording requires both co-primary contrasts to favor selective alignment after multiplicity control.

The primary bootstrap uses 1,000 replicates and seed 20260715. It resamples target-test patients, retains all sessions for a sampled patient, preserves paired model/seed predictions within each replicate, calculates per-seed AUC differences, and then averages those differences over locked seeds. Percentile 95% confidence intervals are reported. Seed SD is descriptive optimization variability, not sampling uncertainty.

## Co-required alignment falsification

The strong selective-alignment claim additionally requires:

- primary selective strategy versus architecture-matched dual no alignment;
- the clinical partition to exceed the locked exact-size random-partition joint null;
- no prespecified material harm in PR AUC, Brier score, calibration intercept, or calibration slope;
- preserved outcome information under locked representation probes; and
- no dependence on one seed, post-hoc lambda, or post-hoc regrouping.

Nineteen paired exact-size random-partition indices are generated before outcome access. The joint statistic is the mean of the IDH and IH seed-averaged AUC improvements for the primary mapping. Its one-sided empirical p-value is `(1 + count(random >= clinical)) / 20`. Unrestricted, variance/missingness-matched, and clinical-type-constrained families each attempt 19 partitions as sensitivity controls; infeasibility and failed runs are reported.

An alignment strategy is not considered beneficial solely because it lowers MMD or another discrepancy metric. Evidence requires improved held-out prediction or calibration without material harm to the other prespecified performance domains. Absence of superiority is reported as such; equivalence requires a prespecified equivalence margin.

## Key secondary estimands

- target-updated dual no-alignment minus source-only target ROC AUC;
- physiology versus treatment alignment within each endpoint;
- endpoint-by-strategy interaction `(D1-D2)_IDH - (D1-D2)_IH`;
- updated model minus target-local reference, without default superiority wording;
- PR AUC, Brier, calibration intercept, calibration slope, and ECE differences;
- feature-group transportability as absolute `AUC_target - AUC_source`;
- normalized excess-AUC ratio `(AUC_target-0.5)/(AUC_source-0.5)` as secondary only when the denominator is stable.

Raw `AUC_target/AUC_source` is prohibited as the primary transportability index. PR AUC is interpreted relative to endpoint prevalence. No equivalence, non-inferiority, or comparable-performance claim is allowed without a separately prespecified clinical margin.

## Seed and bootstrap interpretation

- Seeds quantify optimization variability; they do not replace patient-level uncertainty.
- Patient-cluster bootstrap quantifies sampling uncertainty conditional on the fitted runs.
- All locked seeds are reported. Selecting a favorable seed is prohibited.
- Paired predictions are required for paired contrasts.

## Representation analysis

Report MMD, Wasserstein distance, covariance distance, domain-classifier performance, and outcome-probe performance before and after updating. These are exploratory unless an estimand and multiplicity rule are locked.

A transportability interpretation requires both retained outcome information and held-out target performance. Domain-classifier accuracy at chance is neither necessary nor sufficient.

## Clinical utility

Candidate thresholds of 5%, 10%, and 20% are sensitivity-analysis points unless an independent clinical rationale makes them clinically actionable. They must not be described as prespecified clinical thresholds merely because they were listed before rerunning code.

Confirmatory decision utility requires:

- a locked intended use and alert response
- clinically justified threshold ranges
- model, alert-all, and alert-none comparators
- patient-cluster uncertainty
- alerts, true events captured, false alerts, and missed events per 1,000 sessions
- sensitivity, specificity, PPV, and NPV
- an evaluation cohort not used to choose the thresholds

The historical target test data have already informed prior analyses. If no new temporal or external cohort is available, V3 decision curves are exploratory.

## Subgroups

Subgroup variables, definitions, minimum patient counts, interaction statistic, multiplicity rule, and missing-data handling must be locked before evaluation. Continuous interaction is preferred to data-driven dichotomization. Subgroup-specific significance without a supported interaction is not evidence of heterogeneous treatment or updating effect.

## Completion criterion

The analysis is complete when every prespecified model and contrast is reported, including negative results. Scientific claims are determined by the results; positive answers are not a project acceptance criterion.
