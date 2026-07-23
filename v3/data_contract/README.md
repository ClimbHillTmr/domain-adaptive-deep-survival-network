# V3 Data Contract

The canonical cohort entry points, cleaning rules, feature protocol, split, and statistical estimand are fixed in the repository-level `CLAUDE.md`. Files in this directory must agree with that contract.

## Current status

The V3 source and target cohorts are frozen under `v3_hbd_data_protocol_1` at `data/v3/`. Their hashes, row counts, event counts, quality audit, patient split, preprocessing, and prior-history verification are recorded in `data/v3/data_freeze_manifest.json` and the manifests in this directory.

Reusing historical target patients makes V3 an internal revalidation rather than a newly untouched external confirmation. This affects claim scope but does not invalidate the frozen data contract.

## Required freeze components

1. Raw source/target identities and accessible immutable snapshots.
2. Raw-to-analysis construction code hash and dependency environment.
3. Clinically approved eligibility and endpoint definitions.
4. Rejection counts with mutually exclusive reasons.
5. PII exclusion confirmation.
6. Feature ontology and cross-center semantic mapping.
7. Data-quality adjudication and outlier rules.
8. Source train/validation and target update/calibration/test patient manifests.
9. Strictly-prior history and tied-time verification.
10. Identification of an untouched temporal or external confirmatory cohort.

## Known coverage limitation

The current common analysis table has no clearly defined Tier 4 measurement/process variables that directly encode device, acquisition protocol, or documentation workflow. Raw source and target schemas differ substantially, but schema difference alone is not a patient-level measurement/process feature. Claims that V3 decomposes measurement shift are blocked unless suitable variables and semantic definitions are recovered.
