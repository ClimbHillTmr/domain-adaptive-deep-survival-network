# V3 Data-Quality Gate

Status: deterministic corrections implemented; corrected V3 cohorts not rebuilt or frozen.

## Verified items

- Raw source and target SHA-256 hashes match the available immutable snapshot objects.
- Candidate processed hashes match the recorded analysis hashes.
- Candidate cohorts contain 211,452 source and 74,947 target sessions from 1,628 and 430 patients.
- Existing audits report no duplicate candidate session IDs and no strictly-prior history mismatches.
- Tied source timestamps are handled using only earlier timestamps.

## Confirmed observations and protocol resolution

1. Raw schemas differ: 93 source columns versus 67 target columns. Cross-center semantic equivalence must be documented feature by feature.
2. The target raw table includes direct identifiers such as name. V3 artifacts must explicitly exclude direct identifiers and record a privacy audit.
3. Both raw `年龄` columns are empty, whereas birth date and dialysis date are available. Protocol `v3_hbd_data_protocol_1` derives age at session from those dates, removes exact birth date, and never fills missing age with zero.
4. `性别` contains `男/女` in both centers. The protocol maps female to 0 and male to 1 before generic numeric conversion; unknown values remain missing.
5. Source `透前体温` contains values from 0.36 to 3693. Values outside 30-45 C are set missing and flagged; decimal shifts are not guessed. Target temperature uses the same rule.
6. Source raw `平均动脉压` is a current-session sequence. It is excluded. The single canonical derived baseline MAP is `透前动脉压`, used only as a sensitivity representation when raw SBP/DBP are primary.
7. Several treatment/process features were excluded in the historical allowlist because earlier ablations or target variance suggested removal. V3 cannot inherit performance-driven exclusions; inclusion must be reconsidered without V3 test access.
8. Temperature invalid/missing indicators provide limited process sensitivity, but no explicit device/acquisition metadata is established. Full measurement-shift decomposition remains outside the supported claim scope.

The corrections are implemented in `src/data_pipeline/data_process.py` and enforced through the two center-specific HBD entry points. They are not considered resolved evidence until the corrected cohorts are rebuilt to new paths and their audits and hashes are reviewed.

## Read-only full-snapshot validation

The corrected builder was executed in memory against both frozen raw snapshots without writing cohort files:

| Check | Source | Target |
|---|---:|---:|
| Retained sessions | 211,452 | 74,947 |
| Patients | 1,628 | 430 |
| Female / male encoded sessions | 72,778 / 138,674 | 29,780 / 45,167 |
| Missing derived age | 641 | 0 |
| Missing/invalid dialysis vintage | 1,181 | 14 |
| Temperature values set missing as invalid | 338 | 0 |
| Temperature missing after gate | 105,409 | 342 |
| IDH events | 30,694 | 28,881 |
| IH events | 59,978 | 8,962 |

The retained source cohort contains 246 sessions from six patients whose derived age is below 18 years; the target contains none. They remain included because the fixed protocol does not define an adult-only cohort. This must be reported or amended before manuscript freeze, not silently filtered during modeling.

The validation confirmed that `平均动脉压`, `姓名`, exact birth date, and record identifiers are absent after the corrected protocol. It does not substitute for generating and hashing immutable V3 processed artifacts.

## Acceptance criteria

- Every candidate predictor has unit, valid range, missingness, cross-center mapping, and prediction-time availability documented.
- Direct identifiers are absent from all analysis and prediction artifacts.
- Data corrections are implemented by a new deterministic build, never by editing frozen CSVs.
- Correction code, environment, raw inputs, outputs, and audit reports are hashed.
- All blockers are either resolved or explicitly converted into exclusions approved before V3 test access.
