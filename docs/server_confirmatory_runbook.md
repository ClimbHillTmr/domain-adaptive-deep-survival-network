# Server Confirmatory Runbook

This runbook applies only to the architecture-matched v2 ablation. It does not replace or invalidate the eight registered historical v1 runs used by the current manuscript.

## Evidence contracts

| Contract | Purpose | Current status |
|---|---|---|
| `dual_binary_hemodynamic_v1` | Existing target-center-updating claims | Eight historical runs registered; exact historical processed bytes unavailable |
| `dual_binary_hemodynamic_confirmatory_v2` | Fair architecture-matched mechanism ablation | Data/history/split gates passed; 25 server runs pending |

The v2 analysis files must have these SHA-256 hashes on the server:

- Source: `758eb188b10c057cca0ef1f7ed6bb0e6c3f68846d525bb8ff9e7a22b9c30cb54`
- Target: `5db5118f63d8170c105c94054be3d4d8701d86611f1977de03414446c4b51628`
- Frozen split manifest: `6552a9085aed3f4b82ce3964768503d74ae28c038864b8d71c38af102513e568`

## Execution design

The locked experiment is a blocked computational comparison:

- conditions: five prespecified model/alignment strategies;
- blocks: initialization seeds `7`, `13`, `42`, `99`, and `2024`;
- common response set: the same 29,866 held-out sessions from 172 patients for every condition;
- primary comparisons: within-seed, patient-paired outcome-specific-CORAL contrasts against each of the four comparators;
- uncertainty: 1,000 patient-cluster bootstrap replicates for ROC AUC, PR AUC, Brier score, and ECE.

Use one CUDA GPU and one training process as the default execution policy. This keeps GPU type, software environment, and job contention from becoming model-specific nuisance factors. The launcher runs all 25 locked configurations serially in five seed blocks. A cyclic balanced order makes every model occupy each within-block execution position exactly once, so model family is not tied to early or late runtime. Do not allocate different model families to different GPU types. If a site scheduler is required, request one GPU and wrap the exact launcher command below; scheduler syntax is site-specific and is intentionally not assumed here.

The first completed locked run may be used to estimate remaining wall-clock time and disk use. Its performance must not be used to alter epochs, seeds, hyperparameters, model order, or stopping rules.

The shared training budget is fixed for every model/seed cell: batch size 256, source pretraining up to 40 epochs at learning rate 0.001, target updating up to 20 epochs at learning rate 0.0001, dropout 0.20, and patience 5 under the same validation rule. CORAL-enabled conditions use weight 0.01. All conditions use the same patient split, source-fitted preprocessing, calibration procedure, and held-out scoring code.

## Synchronize to the server

Synchronize the repository while preserving `conf/confirmatory_ablation.yaml`, `data/confirmatory_v2/`, `experiments/audit/feature_allowlist.csv`, `experiments/confirmatory_configs/`, `scripts/`, `src/`, `tests/`, `requirements.txt`, and `pyproject.toml`.

The synchronized package must include `experiments/confirmatory_configs/server_package_manifest.json`. Do not rebuild that manifest or the 25 YAML files on the server before verification; doing so would replace, rather than verify, the analysis-code lock.

Do not synchronize local `experiments/confirmatory_results/binary_*` directories into the server result area. They are excluded smoke-test artifacts.

## Stage 1: server preflight (no training)

```bash
python -m pip install -r requirements.txt
python scripts/build_server_confirmatory_package_manifest.py --verify
python -m pytest tests/ -q
python -m compileall -q src scripts
shasum -a 256 data/confirmatory_v2/source_confirmatory_v2.csv
shasum -a 256 data/confirmatory_v2/target_confirmatory_v2.csv
nvidia-smi
df -h .
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO_CUDA')"
mkdir -p experiments/confirmatory_results
python scripts/run_confirmatory_ablation.py --plan \
  > experiments/confirmatory_results/server_execution_plan.json
```

Review all outputs before authorizing the server job. A fresh package should report `total=25`, `completed=0`, and `pending=25`. The plan command validates the data, split, feature, analysis-code, and configuration locks; it does not train or rewrite configurations.

## Stage 2: resumable server execution

The launcher skips only completed runs carrying `server_confirmatory`, a hostname, and a CUDA device. Re-running the command resumes the locked set:

```bash
set -o pipefail
CUDA_VISIBLE_DEVICES=0 python scripts/run_confirmatory_ablation.py \
  --train \
  --execution-context server_confirmatory \
  2>&1 | tee experiments/confirmatory_results/server_confirmatory.log
```

In training mode the launcher verifies the existing 25 YAML files against the base protocol and evidence lock; it does not rewrite them. `--prepare-configs` is a separate local protocol-locking action and must never be run on the server package. The bottom-level entry point also requires the same execution context and a CUDA device, so a direct CPU invocation is rejected.

Do not run multiple launchers against the same result directory. If the job is interrupted, preserve any partial directory and rerun the same command. Completed configurations with valid server/CUDA provenance are skipped by configuration hash. An interrupted configuration starts again because within-run checkpoint resumption is not part of the frozen protocol.

## Stage 3: completion audit

```bash
python scripts/verify_server_confirmatory_results.py \
  --require-complete \
  --write-report experiments/confirmatory_results/server_acceptance_report.json
python scripts/register_server_confirmatory_runs.py --require-complete
python scripts/build_confirmatory_tables.py
```

Completion requires 25 distinct locked configurations, 1,000 valid patient-cluster bootstrap replicates for each required metric, identical data/feature/split/code hashes, complete artifacts, prediction-to-evaluation agreement, and recorded server/CUDA provenance.

The acceptance report must state `accepted_server_runs=25`, `pending_server_runs=0`, and `status=complete`. Rejected partial or local smoke artifacts are never registered. Registration appends 25 v2 rows while preserving the eight historical v1 rows.

## Stage 4: return artifacts

Return the 25 accepted `binary_*` directories, `server_confirmatory.log`, `server_execution_plan.json`, `server_acceptance_report.json`, the extended central run registry, the 50-row absolute ablation table, and the 160-row paired-contrast table. Rerun the completion audit in the manuscript workspace.

## Stage 5: manuscript integration

After the returned package passes the same audit in the manuscript workspace:

```bash
python scripts/build_confirmatory_tables.py
python scripts/build_submission_staging_figures.py
python scripts/build_manuscript_claim_registry.py --require-complete
python scripts/audit_submission_package.py
```

Figure 2 is generated only after the central registry contains all 25 accepted v2 run IDs. Pale lines connect the same seed across the five strategies, making the blocked comparison visible. The displayed seed spread is descriptive and is not an inferential confidence interval.

## Stop conditions

Stop and investigate without changing the protocol if any of the following occurs:

- package, dataset, split, feature, analysis-code, or configuration hash mismatch;
- CUDA is unavailable or the recorded device is not `cuda:*`;
- any model/seed cell is missing or duplicated;
- prediction rows differ from 29,866 sessions or 172 held-out patients;
- fewer than 1,000 valid patient-cluster bootstrap replicates are recorded for a required metric;
- recomputed metrics disagree with `evaluation.json`.

Do not repair a failed run by editing its locked YAML. Correct an environmental or implementation problem, rebuild and relock the server package if analysis code changes, and document the new package hash before restarting all affected runs. Do not promote a mechanism claim or build Figure 2 until the audit is complete.
