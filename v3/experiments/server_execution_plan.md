# V3 Server Execution Plan

Status: run grid generated and hashed; training commands remain fail-closed until training_authorized=true.

## Purpose

The server workflow executes an evidence plan, not a model list. Each computational stage must resolve a defined Evidence Tree node and produce hash-linked artifacts. A completed process is not admissible evidence unless it is registered and passes verification.

## Stage order

| Stage | Evidence purpose | Compute | Entry condition | Required output | Acceptance condition |
|---|---|---|---|---|---|
| S0 Scientific audit | Validate question-to-evidence completeness | CPU | Repository synchronized | `registry/scientific_audit_status.json` | `structure_passed=true` |
| S1 Data and semantics | Resolve cohort, feature, preprocessing, split, and temporal gates | CPU, no fitting | Clinical review available | Frozen manifests and hashes | No unresolved training-critical data gate |
| S2 Shift characterization | Resolve Evidence 1 without model selection | CPU | S1 locked; development-eligible rows only | Registered shift tables | Patient-cluster uncertainty and ontology sensitivity complete |
| S3 Analysis lock | Freeze estimands, contrasts, multiplicity, seeds, random partitions, thresholds, and claim path | CPU | S1-S2 reviewed | Hashed protocol/config bundle | `training_ready=true` after explicit authorization |
| S4 Smoke validation | Verify one minimal configuration and artifact wiring | GPU, non-evidentiary | Separate reviewed launcher exists | Smoke log outside evidence registry | Shapes, loss, checkpoint, prediction, and provenance checks pass |
| S5 Baselines and transportability | Resolve Evidence 2 and baseline components of Evidence 3 | GPU | S3-S4 pass | Predictions, latent exports, evaluation files | All locked groups/models/seeds complete; no test-driven selection |
| S6 Selective and random controls | Test Evidence 3 | GPU | S5 accepted | Clinical and random-partition runs | All prespecified partitions and seeds reported, including failures |
| S7 Representation validation | Resolve Evidence 4 | GPU/CPU | Registered S5-S6 checkpoints | Latent, probe, collapse, attribution, calibration results | Predictive preservation assessed alongside discrepancy |
| S8 Clinical utility | Resolve Evidence 5 | CPU | Untouched cohort and clinical workflow thresholds available | Calibration, DCA, threshold, and burden tables | Patient-cluster intervals; thresholds independent of final outcomes |
| S9 Evidence freeze | Build submission evidence package | CPU | All executed stages verified | Registry, evidence matrix, tables, figures, hashes | Every numerical claim maps to eligible run IDs and source artifacts |

## Future launcher contract

After authorization, the reviewed launcher may add these commands:

```text
preflight   Verify immutable manifests, code/environment hashes, CUDA, and writable output roots.
plan        Print the full locked execution grid and expected artifact paths without fitting.
smoke       Run a non-evidentiary minimal job and exclude it from confirmatory results.
train       Execute only registered configurations; resume by run ID without overwriting valid artifacts.
verify      Reject incomplete, mismatched, non-server, or unregistered artifacts.
register    Append accepted runs atomically to the V3 registry.
evaluate    Use only accepted predictions and the locked statistical plan.
```

The current `scripts/run_v3_server.sh` implements only `scientific-audit`, `training-readiness`, `tests`, and `status`. It rejects `train`.

## Scheduling and blocking

- Patient is the sampling unit; seed is an optimization block and never substitutes for patient uncertainty.
- Run the same locked seed across comparator models as a block. Record execution order and GPU identity.
- Random partition seeds are distinct from initialization seeds. Generate and hash all partitions before any V3 outcome is viewed.
- Use a prespecified balanced or randomized within-seed execution order so model identity is not confounded with server time or thermal state.
- Do not select successful seeds, partitions, or checkpoints after inspecting the final cohort.
- Job-array width, GPU count, and partition counts remain `REQUIRED_LOCK`; they must follow the final resource and precision calculation, not an invented default.

## Artifact boundary

Every evidentiary run must write to a unique run-ID directory containing:

```text
config.yaml
provenance.json
environment.txt
training_log.jsonl
checkpoint
predictions
latent_exports
evaluation
checksums.sha256
```

No process may append to `registry/run_registry.csv` directly during training. Verification completes first; registration is a separate atomic step. Failed runs are recorded with failure status and cannot be silently replaced under the same run ID.

## Fail-closed rules

Training must stop before GPU allocation when any of the following is true:

- scientific structure or training-readiness audit fails;
- dataset, feature, preprocessing, or split hash differs from the locked bundle;
- patient overlap or temporal leakage is detected;
- the final cohort has been accessed by training, tuning, threshold selection, or feature grouping code;
- a configuration is absent from the locked plan;
- output paths would overwrite an accepted run;
- required server, CUDA, code, or environment provenance cannot be recorded.

## Claim paths after execution

- Path A: selective alignment exceeds matched global, no-alignment, and random controls with preserved information and utility. Mechanism-aware wording may be considered within the protocol's non-causal limits.
- Path B: shift/transportability heterogeneity is supported but selective alignment has no incremental benefit. Position as characterization plus updating-strategy evaluation.
- Path C: neither stable heterogeneity nor selective benefit is supported. Report the boundary conditions and negative falsification result without mechanism-aware claims.

All three paths are valid protocol completion outcomes.
