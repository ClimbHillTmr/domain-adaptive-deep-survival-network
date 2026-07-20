"""Build a provenance-tracked methods supplement from registered evidence only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "submission_supplement.md"
PROVENANCE = ROOT / "experiments" / "evidence_registry" / "submission_supplement_provenance.json"
MAIN_CONFIG = ROOT / "experiments" / "final_results" / "main_mechanism_aware" / "config.yaml"
CONFIRMATORY_CONFIG = ROOT / "conf" / "confirmatory_ablation.yaml"
FEATURE_MANIFEST = ROOT / "experiments" / "evidence_registry" / "feature_manifest.csv"
DATASET_MANIFEST = ROOT / "experiments" / "evidence_registry" / "dataset_manifest.csv"
RUN_REGISTRY = ROOT / "experiments" / "evidence_registry" / "run_registry.csv"
SUBGROUP_PROTOCOL = ROOT / "conf" / "subgroup_protocol.yaml"
CLINICAL_PROVENANCE = ROOT / "clinical_utility_report" / "provenance.json"
SUPPLEMENTARY_FIGURE_PROVENANCE = ROOT / "figures" / "submission_staging" / "SupplementaryFigureS1_provenance.json"
CLINICAL_THRESHOLD_PROTOCOL = ROOT / "conf" / "clinical_threshold_protocol.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, list):
        return ", ".join(map(str, value))
    return str(value)


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    clean = [[_fmt(value).replace("|", "\\|") for value in row] for row in rows]
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(row) + " |" for row in clean),
    ])


def build() -> dict[str, Any]:
    sources = [
        MAIN_CONFIG, CONFIRMATORY_CONFIG, FEATURE_MANIFEST, DATASET_MANIFEST,
        RUN_REGISTRY, SUBGROUP_PROTOCOL, CLINICAL_PROVENANCE,
        SUPPLEMENTARY_FIGURE_PROVENANCE,
        CLINICAL_THRESHOLD_PROTOCOL,
    ]
    missing = [str(path.relative_to(ROOT)) for path in sources if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing registered supplement sources: {missing}")

    main = yaml.safe_load(MAIN_CONFIG.read_text(encoding="utf-8"))
    confirmatory = yaml.safe_load(CONFIRMATORY_CONFIG.read_text(encoding="utf-8"))
    subgroup = yaml.safe_load(SUBGROUP_PROTOCOL.read_text(encoding="utf-8"))
    clinical = json.loads(CLINICAL_PROVENANCE.read_text(encoding="utf-8"))
    features = pd.read_csv(FEATURE_MANIFEST, keep_default_na=False)
    datasets = pd.read_csv(DATASET_MANIFEST, keep_default_na=False)
    registry = pd.read_csv(RUN_REGISTRY, keep_default_na=False)
    historical_registry = registry.loc[registry["contract"].eq("dual_binary_hemodynamic_v1")]
    confirmatory_registry = registry.loc[registry["contract"].eq("dual_binary_hemodynamic_confirmatory_v2")]

    included = features.loc[features["inclusion_status"].eq("included")]
    if len(features) != 57 or len(included) != 24:
        raise ValueError("Supplement requires the registered 57-row manifest with 24 included predictors")
    if len(historical_registry) != 8:
        raise ValueError("Supplement requires the frozen eight-run historical registry")
    if subgroup["status"] == "prespecified_locked":
        subgroup_gate = "Locked; estimates still require an accepted confirmatory-v2 run."
    else:
        subgroup_gate = "Not locked; definitions, approvers, no-test-access attestation, and minimum patient count remain required."

    train = main["training"]
    model = main["model"]
    evaluation = main["evaluation"]
    confirm_train = confirmatory["training"]
    confirm_model = confirmatory["model"]
    confirm_eval = confirmatory["evaluation"]
    clinical_bootstrap = clinical.get("bootstrap", {})

    historical_rows = [
        ["Architecture", "Two encoders joined before a linear prediction head"],
        ["Encoder hidden dimensions", model["mlp_hidden_dims"]],
        ["Dropout", model["dropout"]],
        ["Physiology/history predictors", len(model["physio_indices"])],
        ["Treatment-context predictors", len(model["treat_indices"])],
        ["IDH aligned branch", "physiology/history"],
        ["IH aligned branch", "treatment context"],
        ["CORAL weight", model["coral_weight"]],
        ["Pretraining learning rate", train["pretrain_learning_rate"]],
        ["Updating learning rate", train["finetune_learning_rate"]],
        ["Pretraining/update epoch ceilings", f"{train['pretrain_epochs']} / {train['finetune_epochs']}"],
        ["Batch size", train["batch_size"]],
        ["Early-stopping patience", train["patience"]],
        ["Initialization seeds", train["initialization_seeds"]],
        ["Split seed", main["data"]["split_seed"]],
        ["Patient-cluster bootstrap", f"{evaluation['patient_bootstrap_replicates']} replicates; seed {evaluation['bootstrap_seed']}"],
    ]
    confirmatory_rows = [
        ["Protocol", confirmatory["confirmatory"]["protocol_version"]],
        ["Data contract", confirmatory["confirmatory"]["data_contract"]],
        ["Execution context", confirmatory["confirmatory"]["execution_context"]],
        ["Models", "dual no alignment; dual global CORAL; dual random-feature CORAL; dual outcome-specific CORAL; single encoder"],
        ["Seeds", confirm_train["initialization_seeds"]],
        ["Architecture/hidden dimensions", f"{confirm_model['architecture']}; {confirm_model['mlp_hidden_dims']}"],
        ["Dropout / batch size", f"{confirm_model['dropout']} / {confirm_train['batch_size']}"],
        ["Learning rates", f"pretraining {confirm_train['pretrain_learning_rate']}; updating {confirm_train['finetune_learning_rate']}"],
        ["Epoch ceilings / patience", f"{confirm_train['pretrain_epochs']} / {confirm_train['finetune_epochs']} / {confirm_train['patience']}"],
        ["CORAL weight", confirm_model["coral_weight"]],
        ["Bootstrap", f"{confirm_eval['patient_bootstrap_replicates']} patient-cluster replicates; seed {confirm_eval['bootstrap_seed']}"],
        ["Current evidence state", "Pending compute-server execution and acceptance; no v2 result is used below"],
    ]
    dataset_rows = [
        [row["dataset_version"], row["cohort"], row["row_count"], row["patient_count"], row["idh_prevalence"], row["ih_prevalence"], row["status"]]
        for _, row in datasets.iterrows()
    ]
    feature_rows = [
        [row["feature_name"], row["category"], row["branch_assignment"], row["inclusion_status"], row["available_at_prediction_time"], row["preprocessing_rule"], row["rationale"]]
        for _, row in features.iterrows()
    ]
    subgroup_rows = []
    for endpoint, definitions in subgroup["endpoints"].items():
        for item in definitions:
            subgroup_rows.append([
                endpoint.upper(), item["subgroup"], item["candidate_feature"], item["definition"], subgroup["reporting"]["minimum_unique_patients_per_subgroup"]
            ])

    text = f"""# Supplementary Methods and Evidence Definitions

Status: evidence-linked draft; generated from registered artifacts. This supplement does not add experimental results and must be rebuilt after accepted confirmatory server runs or locked subgroup prespecification.

## S1. Evidence contracts and reproducibility boundary

Historical v1 results remain the evidence base for the manuscript's current target-center-updating conclusions. The exact processed v1 bytes are unavailable in the current workspace, so the historical dataset hashes cannot be reproduced from the present files; this limitation is permanent and must remain disclosed. The separately versioned confirmatory-v2 contract is reserved for new architecture-matched server experiments and must not be substituted into v1 claims.

{_table(["Contract", "Cohort", "Sessions", "Patients", "IDH prevalence", "IH prevalence", "Evidence status"], dataset_rows)}

The historical registry contains {len(historical_registry)} v1 runs; {len(confirmatory_registry)} accepted confirmatory-v2 runs are currently registered. Every manuscript result that depends on model output is linked to a registered run ID, hashed configuration, checkpoint, prediction file, evaluation file, and split manifest.

## S2. Historical main-model hyperparameters

{_table(["Item", "Registered value"], historical_rows)}

Numerical medians, means, standard deviations, and categorical mappings were fitted on source-training patients only. Numerical missing values were replaced with source-training medians; unknown target categories received a reserved code. Platt calibration used the corresponding validation patients. The held-out target patients were not used for model updating, selection, threshold selection, or calibration.

## S3. Architecture-matched confirmatory protocol

{_table(["Item", "Locked value or state"], confirmatory_rows)}

All five strategies use the same data contract, patient split, preprocessing, optimization budget, calibration procedure, and seeds. Confirmatory tables and Figure 2 remain empty until all required server runs pass the read-only acceptance audit and enter the central run registry. The prespecified paired contrast is outcome-specific CORAL minus each of the four architecture-matched comparators. Positive differences favor outcome-specific CORAL for ROC AUC and PR AUC; negative differences favor it for Brier score and 10-quantile-bin ECE.

## S4. Statistical resampling and clinical utility

The primary uncertainty unit is the patient. Patients are sampled with replacement and all their sessions are retained within a bootstrap replicate. Percentile 95% intervals use {evaluation['patient_bootstrap_replicates']} replicates with bootstrap seed {evaluation['bootstrap_seed']}. Within-run AUC contrasts are paired because competing prediction vectors are evaluated on the same sampled patients. Five initialization seeds are summarized descriptively by their individual values, mean, and sample standard deviation; across-seed ranges are not confidence intervals.

Clinical-utility summaries use registered held-out predictions only and do not fit or update a predictive model. Calibration intercept, calibration slope, expected calibration error, Brier score, and fixed-threshold alert metrics use {clinical_bootstrap.get('replicates', 1000)} patient-cluster replicates. Alert thresholds are validation-derived Youden thresholds, not clinically prespecified deployment thresholds. Decision curves remain threshold-grid sensitivity analyses until nephrology collaborators lock a clinically relevant threshold interval.

The separate protocol `conf/clinical_threshold_protocol.yaml` controls any upgrade from descriptive threshold-grid sensitivity analysis to clinically prespecified population-level utility evidence. It requires nephrology, statistical, and clinical-workflow approval; a no-target-test-DCA-access attestation; endpoint-specific DCA ranges; explicit operating thresholds; and documented false-alert and missed-event consequences. It does not authorize autonomous treatment recommendations.

## S5. Predictor definitions and preprocessing

The registered feature manifest contains {len(features)} candidate or derived fields; {len(included)} are included predictors. Availability is defined relative to the session-start prediction time. Rows marked `end` or `outcome` are excluded to prevent current-session future information from entering predictors.

{_table(["Feature", "Category", "Branch", "Status", "Availability", "Preprocessing", "Rationale"], feature_rows)}

## S6. Patient-cluster subgroup protocol

Protocol status: `{subgroup['status']}`. {subgroup_gate}

{_table(["Endpoint", "Subgroup", "Candidate feature", "Locked definition", "Minimum patients per level"], subgroup_rows)}

The planned comparison is updated MLP minus source MLP. Each eligible level will report subgroup AUC, paired delta AUC, percentile patient-cluster interval, and a two-sided patient-cluster bootstrap interaction P value. Previous session-level subgroup estimates and P values are discarded. No target-test subgroup features may be inspected before the definitions, minimum patient count, approver roles, rationale, and no-test-access attestation are locked.

## S7. Artifact locations

- Dataset manifest: `experiments/evidence_registry/dataset_manifest.csv`
- Feature manifest: `experiments/evidence_registry/feature_manifest.csv`
- Split manifest: `experiments/evidence_registry/split_manifest.csv`
- Run registry: `experiments/evidence_registry/run_registry.csv`
- Confirmatory configuration: `conf/confirmatory_ablation.yaml`
- Subgroup protocol: `conf/subgroup_protocol.yaml`
- Clinical utility provenance: `clinical_utility_report/provenance.json`
- Manuscript claim registry: `experiments/evidence_registry/manuscript_claim_registry.csv`
- Supplementary Figure S1 provenance: `figures/submission_staging/SupplementaryFigureS1_provenance.json`
- Clinical threshold protocol: `conf/clinical_threshold_protocol.yaml`
"""
    OUTPUT.write_text(text, encoding="utf-8")
    report = {
        "status": "evidence_linked_draft_pending_confirmatory_and_subgroup_gates",
        "training_performed": False,
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_sha256": _sha256(OUTPUT),
        "registered_historical_runs": len(historical_registry),
        "registered_confirmatory_runs": len(confirmatory_registry),
        "feature_rows": len(features),
        "included_predictors": len(included),
        "subgroup_protocol_status": subgroup["status"],
        "sources": {str(path.relative_to(ROOT)): _sha256(path) for path in sources},
    }
    PROVENANCE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
