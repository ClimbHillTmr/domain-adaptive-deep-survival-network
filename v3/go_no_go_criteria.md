# V3 Go/No-Go Criteria

“No-go” means the corresponding claim or execution phase stops. It does not mean unfavorable results are discarded.

## Gate 1: data freeze

Go only if:

- raw and processed hashes are verified;
- eligibility, endpoint, date, and artifact rules are clinically approved;
- PII exclusion is verified;
- sex, dialysis vintage, temperature, duplicate MAP, units, ranges, and missingness are resolved;
- each feature has a cross-center semantic mapping and ontology uncertainty;
- deterministic construction and audit tests pass; and
- a versioned immutable output is created without overwriting candidate files.

Otherwise: no training.

## Gate 2: split and analysis lock

Go only if:

- patient roles are mutually exclusive;
- strictly-prior and tied-time audits pass;
- split sizes are justified by patient/event precision rather than session count alone;
- primary endpoint(s), metrics, contrasts, seeds, random partitions, lambda selection, and multiplicity are frozen;
- final-test access control is documented; and
- the status identifies whether the evaluation is new confirmatory or internal revalidation.

Otherwise: no confirmatory execution.

## Gate 3: observed heterogeneous shift

Support H1 only if feature-level results demonstrate reproducible differences in shift magnitude or direction under patient-cluster uncertainty and are not explained solely by missingness, coding errors, or a single metric.

Do not promote H1 to a causal decomposition. Without measurement/process features, the measurement-shift component remains untested.

## Gate 4: mechanism-aware alignment

The strong mechanism-aware claim requires all of the following:

1. Selective dual alignment improves the prespecified target discrimination endpoint over dual global alignment under a paired seed-aggregated analysis.
2. It improves over dual no alignment or demonstrates a prespecified complementary calibration/utility advantage.
3. The clinical partition exceeds the empirical distribution of multiple size-matched random partitions.
4. The result does not depend on one selected seed, lambda, grouping, or endpoint.
5. PR-AUC, Brier, and calibration show no prespecified material harm.
6. Representation analyses show preserved outcome information; discrepancy reduction alone is insufficient.

If any central condition fails: no-go for the strong title. Report the negative experiment and use manuscript Path B or C.

Formal superiority, equivalence, and non-inferiority wording requires corresponding prespecified margins and confidence intervals. Similar point estimates alone are not proof of equivalence.

## Gate 5: outcome-specific transport

Support H3 only if the prespecified endpoint-by-strategy contrast is supported after multiplicity control and remains stable across locked seeds and alternative plausible ontologies.

If unsupported: report that no reliable outcome-specific strategy was identified. Do not select the best mapping separately after seeing results.

## Gate 6: clinical utility

Go for deployment-oriented claims only if:

- intended use and alert response are explicit;
- clinical thresholds are justified independently of the evaluation outcomes;
- a new temporal or external cohort is used;
- net benefit exceeds alert-all and alert-none over the prespecified range;
- true events captured, false alerts, missed events, and number needed to review are acceptable to the clinical governance group; and
- calibration is adequate or a locked recalibration procedure is used.

Without these conditions, utility results remain exploratory and the manuscript cannot claim deployment readiness.

## Final manuscript decision

- Path A: Gates 1-6 pass and the mechanism-aware contrasts are supported.
- Path B: data/analysis gates pass but selective alignment or H3 is unsupported; publish the characterization and falsification study.
- Path C: supervised updating is supported but mechanism-aware evidence is not; publish a target-center updating study.
- Hold: data integrity, test independence, or provenance gates fail.
