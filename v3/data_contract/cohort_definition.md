# Candidate Cohort Definition Audit

## Current implemented construction

The corrected V3 construction code is `src/data_pipeline/data_process.py`, SHA-256 `9e53310bd7d2cf9a117368657fff579e7afc69a08d8f31e53552e861b3b0dd53`. The corrected cohorts have not yet been regenerated. Counts below therefore describe the historical candidate build and must be re-audited after explicit rebuild authorization.

A raw session is retained when:

- intradialytic systolic and diastolic sequences are parseable;
- time nodes are parseable and contain at least two observations;
- systolic, diastolic, and time-node lengths match;
- baseline systolic and diastolic blood pressure are finite; and
- calculated dialysis duration is positive.

Current rejection counts:

| Cohort | Raw | Retained | Invalid sequence | Misaligned sequence | Invalid time |
|---|---:|---:|---:|---:|---:|
| Source | 222,836 | 211,452 | 6,521 | 4,728 | 135 |
| Target | 76,519 | 74,947 | 176 | 1,225 | 171 |

## Candidate endpoint definitions

- IDH: baseline systolic blood pressure minus any intradialytic systolic value is at least 30 mmHg, or any intradialytic systolic value is at most 90 mmHg.
- IH: any intradialytic mean arterial pressure exceeds baseline mean arterial pressure by more than 10 mmHg.
- Event time is the first qualifying intradialytic measurement; non-event time is stored as zero for the binary task.

## Clinical gates still required

The construction code does not currently establish:

- adult maintenance-hemodialysis eligibility;
- minimum prior dialysis exposure;
- acute dialysis or ICU exclusions;
- physiologically valid ranges or artifact handling;
- minimum or standardized BP measurement frequency;
- center-specific device and measurement comparability;
- complete recruitment dates and censoring rules; or
- whether repeated same-day records represent distinct valid sessions.

These rules require clinical approval before V3 freeze. Changes require a new processed-data hash and dataset version; they must not overwrite the candidate files.
