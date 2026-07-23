# Conditional V3 Manuscript Outline

No V3 results currently exist. This outline fixes the reporting sequence but not the conclusion.

## Title paths

### Path A: mechanism-aware gate passes

`Mechanism-Aware Transportable Representation Learning for Clinical AI under Heterogeneous Cross-Center Shift`

Eligible only when the selective-alignment, random-control, representation-preservation, calibration, and utility gates in `go_no_go_criteria.md` pass.

### Path B: gate does not pass

`Characterizing Heterogeneous Cross-Center Shift and Evaluating Selective Updating Strategies for Clinical AI`

### Path C: updating is the only supported intervention

`Target-Center Updating for Hemodynamic Event Prediction under Cross-Center Clinical Shift`

## Abstract

- Background: cross-center prediction failure and uncertainty about which distribution differences are harmful.
- Methods: new V3 data contract, patient-level split, clinical feature ontology, matched models, multiple random partitions, patient-cluster inference, and conditional clinical utility.
- Findings: report H1, updating, H2/H3, representation, and utility results in prespecified order, including null findings.
- Interpretation: select Path A, B, or C using the frozen go/no-go rules.

## Introduction

1. Clinical prediction models often degrade across institutions.
2. Distribution alignment commonly treats domain differences as nuisance, but clinical variation may contain useful context.
3. Existing evidence does not establish which components should be aligned; V1/V2 motivated but did not confirm the mechanism-aware hypothesis.
4. V3 objectives: characterize observed shift heterogeneity, measure feature-tier transportability, test selective alignment fairly, and assess predictive preservation and clinical utility.

## Methods

1. Study design, centers, governance, ethics, dates, and eligibility.
2. Endpoint definitions and measurement-quality rules.
3. V3 data, feature, preprocessing, and split contracts.
4. Four-tier feature ontology and alternative grouping sensitivity.
5. Phase 1 shift estimands.
6. Phase 2 feature-tier transportability probes.
7. Models A-G, capacity matching, CORAL, seeds, and model selection.
8. Representation analyses.
9. Calibration, decision curves, alert burden, and thresholds.
10. Patient-cluster statistics, multiplicity, missing data, and sensitivity analyses.

## Results order

1. Cohort flow and data-quality audit.
2. Observed outcome, feature, and missingness shifts.
3. Feature-tier natural transportability.
4. Source-only and supervised-updating comparison.
5. Architecture-matched selective/global/no/random alignment contrasts.
6. Endpoint-by-strategy analysis.
7. Representation preservation analyses.
8. Clinical utility if its independent gate is met.

## Discussion

1. State the primary findings without mechanism inflation.
2. Distinguish observed heterogeneity from causal shift decomposition.
3. Explain whether selective alignment added value beyond updating and architecture.
4. Interpret random controls and failed falsification tests.
5. Relate latent changes to prediction and calibration without claiming domain invariance.
6. Discuss operational deployment implications only when thresholds and workflow are justified.
7. Limitations: two centers, repeated sessions, feature ontology uncertainty, measurement metadata gaps, previously inspected historical cohort, and retrospective design.

## Main displays

- Figure 1: cohort flow, ontology, and observed feature/missingness shift.
- Figure 2: source-to-target transportability by feature tier.
- Figure 3: architecture-matched alignment results across seeds and random partitions.
- Figure 4: discrimination, calibration, and predictive preservation.
- Figure 5: clinical utility, conditional on the utility gate.
- Table 1: cohort and data-quality characteristics.
- Table 2: shift and transportability estimands.
- Table 3: registered model comparisons.
- Table 4: calibration and utility.
