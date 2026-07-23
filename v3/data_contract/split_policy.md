# V3 Split Policy

Status: not locked.

## Required roles

- Source training: model fitting only.
- Source validation: source pretraining selection and stopping only.
- Target update: supervised target-center updating and target-only reference fitting.
- Target calibration: probability calibration, locked model selection where prespecified, and threshold implementation that does not use final-test outcomes.
- Final held-out test: one-time evaluation after registry and analysis lock.

## Independence rules

- Assignment is at patient level; all sessions for a patient remain in one role.
- No patient identifier may overlap roles within or across source/target allocations where identifiers are globally meaningful.
- Split generation uses a frozen seed and joint patient-level endpoint strata where feasible.
- Split fractions require an event- and patient-based precision analysis; historical fractions are not automatically inherited.
- Every patient ID and role is recorded in `split_manifest.csv`, which is hashed before training.

## Temporal rules

- Sessions are ordered by patient, dialysis start time, and deterministic session ID.
- Historical features for an index session use only observations with strictly earlier timestamps.
- Same-patient tied timestamps cannot contribute history to one another.
- Final test access is logged. Prediction and evaluation code must produce predictions before test outcomes are made available to analysts performing model selection.

## Confirmatory test requirement

The historical target held-out patients and outcomes have already informed V1/V2 design. Reassigning those patients under a new seed does not make them prospectively untouched. Strong V3 confirmation requires a new temporal target extract or an independent center.

If no new cohort is available, the split can support transparent internal revalidation, but manuscript language must not call it an untouched confirmatory external test.
