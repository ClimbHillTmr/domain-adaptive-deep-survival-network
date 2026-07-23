# V3 Research Contract

The repository-level `CLAUDE.md` is the authoritative starting checkpoint. This directory contains its machine-readable evidence, data, experiment, and analysis artifacts.

## Objective

V3 tests whether clinically motivated feature components show different transportability properties under cross-center shift, and whether selective representation alignment provides incremental value beyond supervised target-center updating.

V3 is a new evidence contract. V1 and V2 artifacts may motivate hypotheses and design safeguards, but they cannot support V3 confirmatory claims.

## Evidence boundary

- `v3/registry/run_registry.csv` is the only registry for V3 runs.
- The registry starts empty. Historical run IDs must never be copied into it.
- A numerical V3 result is admissible only when its dataset, features, preprocessing, split, configuration, predictions, evaluation, code, and environment are hash-linked.
- No V3 manuscript claim may be promoted from `not_tested` until the corresponding registered analysis passes its prespecified gate.
- A negative or null result is a valid completed result. Project success does not require every hypothesis to be positive.

## Scientific questions

1. Which observed feature and outcome distributions differ between the two centers?
2. Does supervised target-center updating improve held-out target performance over source-only deployment?
3. Does a prespecified clinical feature partition identify alignment strategies that outperform architecture-matched no alignment, global alignment, and size-matched random partitions?
4. Are any predictive gains accompanied by preserved outcome information and acceptable calibration, rather than discrepancy reduction alone?
5. Does the resulting model provide net benefit under clinically justified operating thresholds?

These questions are indexed by the Evidence Tree rather than by model name:

- `evidence_tree.md`: human-readable claim, evidence, and falsification hierarchy;
- `evidence_tree.yaml`: machine-readable scientific contract;
- `experiments/experiment_matrix.csv`: one-to-one mapping from evidence nodes to planned experiments;
- `scientific_audit.yaml`: structural and training-readiness gate definitions.

Models A-G are experimental tools within Evidence 3. They are not the organizing structure of V3.

## Interpretation boundaries

- Distribution metrics do not identify the cause of a shift. Population, measurement, treatment-policy, and disease-mechanism effects are not separately identifiable from a two-center retrospective comparison alone.
- `physiology`, `treatment`, and `context` are clinically motivated feature partitions, not recovered biological mechanisms.
- A low domain-classifier score is not required for a useful representation, and a lower score does not by itself establish transportability.
- The historical target test cohort has already informed V1/V2 development. A new temporal or external test cohort is required for fully confirmatory V3 deployment claims. Without one, V3 estimates must be labeled internally revalidated or exploratory as appropriate.

## Phase gates

| Phase | Gate to proceed |
|---|---|
| 0. Evidence contract | Manifests locked; registry empty; historical evidence classified; no training authorized |
| 1. Shift characterization | Feature definitions and analysis units locked; patient-cluster uncertainty reported |
| 2. Hypothesis lock | Clinical partitions, primary contrasts, multiplicity, and falsification criteria locked without V3 test access |
| 3. Baselines | Source-only, target-only, joint, and target-updated baselines completed under identical evaluation rules |
| 4. Alignment | Architecture-matched alignment runs completed across locked seeds and budgets |
| 5. Falsification | Multiple size-matched random partitions and negative results reported without selection |
| 6. Utility | Clinically justified thresholds and an untouched evaluation cohort available, or analysis labeled exploratory |
| 7. Manuscript freeze | Every numerical claim resolves to an eligible V3 run and artifact hash |

## Directory roles

- `data_contract/`: immutable dataset, feature, and split manifests
- `preprocessing/`: source-fitted transformation and missingness contract
- `experiments/`: prespecified experiment matrix
- `models/`: architecture and comparator specifications
- `evaluation/`: statistical analysis plan
- `interpretation/`: causal and representation interpretation limits
- `manuscript/`: claim and reporting policy
- `registry/`: V3 runs, claims, and historical evidence classification

## Current authorization

V3 data regeneration is complete and frozen. Model training remains unauthorized until the exact execution configuration is reviewed and `training_authorized` is explicitly changed.

## Audit-only entry point

The server launcher currently exposes governance checks only:

```bash
bash scripts/run_v3_server.sh scientific-audit
bash scripts/run_v3_server.sh training-readiness
bash scripts/run_v3_server.sh tests
```

`scientific-audit` succeeds when the evidence structure is internally consistent, even when training remains blocked. `training-readiness` fails closed until every training-critical data, feature, split, preprocessing, statistical, random-control, and authorization gate is resolved. Scope limitations and downstream clinical-utility gates are reported separately; they restrict later claims but do not masquerade as universal training blockers. The launcher intentionally has no training implementation; `train` exits without starting a process.

The current expected state is:

```text
structure_passed = true
data_preflight_ready = true
training_ready = false
status = blocked
```

The staged, fail-closed design for later server execution is documented in `experiments/server_execution_plan.md`. It does not authorize training or define unapproved run counts.
