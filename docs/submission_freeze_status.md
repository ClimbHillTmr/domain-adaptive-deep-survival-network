# Submission Freeze Status

Status date: 2026-07-19

## Current decision

The manuscript is not submission-ready. Historical prediction and model artifacts are registered and support a conservative target-center updating claim, but the confirmatory mechanism claim is blocked.

## Completed

- Eight historical runs registered with configuration, dataset, split, checkpoint, prediction, and evaluation hashes.
- Frozen patient splits are identical across registered runs and have no source or target patient overlap.
- Dataset and 57-row feature manifests generated; 24 predictors are marked as included.
- Registered seed-2024 prediction package used for calibration, ECE, Brier, alert burden, and descriptive DCA sensitivity analysis.
- Latent and SHAP summaries exported with explicit exploratory labels.
- Five-model by five-seed architecture-matched configurations generated under one optimization and preprocessing contract.
- Previous session-level subgroup inference discarded and replaced by a patient-cluster protocol registry.
- All 27 manuscript paragraphs containing quantitative information are covered by 28 claim-registry entries with registered run IDs where applicable and hashed source artifacts.
- Manuscript Table 1 (registered cohort/design) and Table 2 (held-out performance) are generated with source/output hashes; Table 3 remains an explicit server-pending shell.
- Submission Figure 1 is rebuilt as a native vector study-design/workflow figure with direct preflight, split-manifest, and registered-config provenance; Figures 3–5 are staged, while Figure 2 remains evidence-gated.
- An evidence-linked supplementary methods package now records the historical and confirmatory hyperparameters, seeds, patient-cluster bootstrap method, complete feature definitions, and subgroup protocol directly from registered sources.
- The server execution package locks 25 configurations together with v2 data, feature allowlist, patient split, analysis code, dependencies, and verification scripts in one hash-audited manifest; no training has been performed locally.
- Supplementary Figure S1 is generated as a native vector, five-seed paired-point figure with registered evaluation hashes, source-data CSV, alt text, and explicit descriptive-only interpretation.

## Blocking gates

1. Exact historical v1 processed bytes remain unavailable; this must be disclosed and v2 results must not be substituted into v1 claims.
2. The 25 architecture-matched v2 runs remain pending on the compute server.
3. Returned server runs must pass `scripts/verify_server_confirmatory_results.py --require-complete` before entering confirmatory tables or figures.
4. Nephrology collaborators must prespecify clinically relevant risk-threshold intervals before confirmatory DCA interpretation.
5. Clinical/statistical collaborators must prespecify subgroup definitions and minimum patient counts.
6. Ethics, authorship, funding, conflicts, governance, cohort dates, references, and journal checklist metadata remain incomplete.

The quantitative claim map passes, but this is not equivalent to submission readiness. Any edit that adds or changes a quantitative paragraph must update `conf/manuscript_claims.yaml` and pass `python scripts/build_manuscript_claim_registry.py --require-complete` before the manuscript can be frozen again.

The available Table 1 is a cohort-and-analysis-allocation table, not a population baseline-characteristics table. The legacy baseline table uses the superseded survival contract and must not be relabeled or cited in this manuscript.

## Server-only continuation

The registered historical runs remain the evidence base for the current target-center-updating conclusions; they do not need to be replaced by retraining. New fitting is required only if the architecture-matched mechanism ablation is pursued. That fitting must be launched on the compute server with `python scripts/run_confirmatory_ablation.py --train --execution-context server_confirmatory`, after the evidence gates pass. Local smoke-test artifacts are excluded from manuscript evidence.

Blank fields in `architecture_matched_ablation.csv` and `subgroup_analysis_patient_bootstrap.csv` are deliberate blocked states, not zero-valued results.
