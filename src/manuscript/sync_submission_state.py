import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MANUSCRIPT_DIR = ROOT / "manuscript"
AUDIT_DIR = ROOT / "experiments" / "audit"
RESULT_DIR = ROOT / "experiments" / "results"
REPORT_PATH = AUDIT_DIR / "result_consistency_report.json"


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def format_cindex(metrics: Dict[str, Any]) -> str:
    return (
        f"{metrics['C-index']:.3f} "
        f"(95% CI, {metrics['C-index_CI_lower']:.3f}-{metrics['C-index_CI_upper']:.3f})"
    )


def build_results_md() -> str:
    manifest = read_json(AUDIT_DIR / "data_manifest.json")
    split = read_json(AUDIT_DIR / "split_audit.json")
    report = read_json(REPORT_PATH) if REPORT_PATH.exists() else {"can_be_used_for_submission": False}

    lines = [
        "# Results",
        "",
        "## Study Population and Final Analytic Cohorts",
        (
            f"A total of {manifest['source']['n_rows'] + manifest['target']['n_rows']:,} dialysis sessions were included in "
            f"the final analysis, comprising {manifest['source']['n_rows']:,} sessions from the Shenyi development cohort "
            f"and {manifest['target']['n_rows']:,} sessions from the Fuding external validation cohort. The Fuding cohort "
            f"contained {manifest['target']['n_patients']} unique patients and was split at the patient level into a target "
            f"updating subset ({split['train']['n_patients']} patients; {split['train']['n_sessions']:,} sessions), a target "
            f"validation subset ({split['val']['n_patients']} patients; {split['val']['n_sessions']:,} sessions), and a held-out "
            f"test subset ({split['test']['n_patients']} patients; {split['test']['n_sessions']:,} sessions). Patient overlap "
            f"across the three target subsets was 0 for all pairwise comparisons (Fig. 1)."
        ),
        "",
        "## Cross-center Baseline and Event-time Heterogeneity",
        (
            "Marked differences were observed between the Shenyi and Fuding cohorts in both baseline hemodynamics and "
            "event-time structure (Fig. 2). These cross-center shifts support the clinical need for a locally updated survival "
            "model rather than a source-only transport assumption."
        ),
        "",
        "## External Validation Performance",
    ]

    if report.get("can_be_used_for_submission"):
        evaluation = read_json(RESULT_DIR / "evaluation_results.json")
        zero_shot = evaluation["sweeps"]["Zero-shot"]["metrics"]
        updated = evaluation["sweeps"]["CDAN-GSN (Ours)"]["metrics"]
        lines.extend(
            [
                (
                    "On held-out external testing, the locally updated survival model outperformed the zero-shot baseline in "
                    f"discrimination (Fig. 3). The C-index improved from {format_cindex(zero_shot)} for the source-only baseline "
                    f"to {format_cindex(updated)} after target-center labeled updating."
                ),
                "",
                "## Calibration and Decision-curve Evidence",
            ]
        )
        calibration = read_json(RESULT_DIR / "calibration_dca_metrics.json")
        prediction_meta = read_json(RESULT_DIR / "prediction_metadata.json")
        pred_df = pd.read_csv(RESULT_DIR / "real_test_predictions.csv")
        lines.extend(
            [
                (
                    f"Real case-level predictions were exported for the held-out external test cohort ({prediction_meta['n_test']:,} sessions). "
                    f"At 60 minutes, the observed event rate was {calibration['60m']['observed_rate']*100:.2f}% and the mean predicted "
                    f"probability was {calibration['60m']['mean_predicted_rate']*100:.2f}%, with a Brier score of {calibration['60m']['brier']:.4f}. "
                    f"At 120 minutes, the observed event rate was {calibration['120m']['observed_rate']*100:.2f}% and the mean predicted "
                    f"probability was {calibration['120m']['mean_predicted_rate']*100:.2f}%, with a Brier score of {calibration['120m']['brier']:.4f}."
                ),
                (
                    "The calibration plots and decision-curve panels generated from these held-out predictions show that the model "
                    "preserves clinically useful risk ordering while requiring explicit acknowledgment of residual underestimation "
                    "in the external center (Fig. 4)."
                ),
                "",
                "## Supplementary Target Subgroup Burden",
                (
                    "Observed IDH burden across major held-out target subgroups is provided as a supplementary figure rather than a "
                    "main-text claim of subgroup-specific model superiority (Fig. S1)."
                ),
            ]
        )
    else:
        lines.extend(
            [
                (
                    "The previous performance values have been withdrawn because the existing result files were not yet consistent "
                    "with the latest audit-verified split and allowlist state. This section must remain numeric-free until a locked "
                    "rerun produces a new `evaluation_results.json` linked to the current audit artifacts."
                ),
                "",
                "## Calibration and Decision-curve Evidence",
                (
                    "Calibration, Brier score, and decision-curve statements are temporarily withheld. These analyses should only be "
                    "restored after `real_test_predictions.csv`, `calibration_dca_metrics.json`, and the held-out test sample size are "
                    "all regenerated under the same locked run."
                ),
                "",
                "## Supplementary Target Subgroup Burden",
                (
                    "Target subgroup burden is reserved for supplementary presentation and should not re-enter the main text unless "
                    "it is rebuilt from the current held-out predictions."
                ),
            ]
        )

    return "\n".join(lines) + "\n"


def build_methods_md() -> str:
    return """# Methods

## Study Design and Data Sources
This study was a retrospective two-center prediction and target-center updating study based on routinely collected hemodialysis-session data. The Shenyi cohort was used as the source development cohort, and the Fuding cohort was used as the target-center cohort for limited labeled updating, validation, and held-out testing. The unit of prediction was the dialysis session, whereas the target-center split was performed at the patient level to prevent within-patient leakage across updating, validation, and test subsets.

The final analytic dataset included 291,828 dialysis sessions, comprising 216,604 sessions from Shenyi and 75,224 sessions from Fuding. Processed model-ready files were `深医_final_data.csv` and `福鼎_final_data.csv`, stored under `data/processed`.

## Ethics and Reporting
This manuscript should be reported as a retrospective prediction and external validation study aligned with TRIPOD-AI principles where applicable. Institutional review board approval, waiver status, and data-governance language should be inserted only after the formal institutional wording is confirmed.

## Outcome Definition
The primary outcome was intradialytic hypotension modeled as a time-to-event endpoint. Event timing was stored as `et_min`, representing minutes from dialysis start to event occurrence, and the event indicator was stored as `events`.

## Predictor Construction and Preprocessing
Raw data processing followed a staged pipeline including center-specific feature parsing, derived hemodynamic feature construction, patient-level historical summary generation, and final feature merging. Historical mean and rate features were constructed only from prior sessions after chronological sorting within patient, such that the first audited session for each sampled patient had zero-valued history features.

Categorical encodings were fit in the source cohort and then applied unchanged to the target cohort, with unseen target-only categories assigned to an explicit unknown level. Missing predictor values were filled with zero, and all predictors were standardized using the source-cohort mean and standard deviation before application to the target cohort.

## Candidate Predictors Used for Modeling
Model construction used a predefined prediction-time feature allowlist rather than dynamically appending every column with a `history_` prefix. The frozen allowlist was stored in `experiments/audit/feature_allowlist.csv` and included demographic variables, treatment-context variables, pre-dialysis physiologic features, derived hemodynamic measures, and audited historical burden summaries judged to be available at prediction time. Current-session intradialytic summary columns and outcome columns were explicitly excluded by the allowlist.

## Data Splitting and Validation Strategy
The Shenyi cohort was used for source-domain model development. The Fuding cohort was split into mutually exclusive patient-level subsets for target-center updating, target validation, and held-out testing. The implemented pipeline used `target_adapt_ratio = 0.20`, with `target_val_ratio = 0.20` within the adaptation pool, corresponding to 68 target patients for updating, 18 for validation, and 344 for final held-out testing under a fixed random seed of 42.

## Model Architecture
The proposed model was implemented as a domain-stratified deep survival network with a KAN-based tokenizer and a gated representation module. A transformer encoder learned contextualized feature embeddings, and a hazard head produced a session-level risk score. Treatment-context features and non-treatment physiologic features were separated to support statistically guided representation alignment without implying causal identification.

## Model Training Procedure
Training consisted of source-cohort pretraining followed by target-center labeled updating. This second phase should be interpreted as local target-center updating of a survival model rather than as a purely unsupervised domain-adaptation exercise. Updating used the labeled target training subset together with replayed source batches and target-validation early stopping.

## Handling of Censoring and Weighting
Inverse probability of censoring weighting (IPCW) was used in training and evaluation workflows. IPCW weights were estimated from Kaplan-Meier fits of the censoring distribution and truncated at the 95th percentile to reduce instability from extreme weights.

## Model Evaluation
The primary discrimination metric was Harrell's concordance index in the held-out target test cohort, with bootstrap confidence intervals. Calibration and decision-curve analysis were performed only from real case-level predictions exported for the held-out target test subset. Calibration, Brier scores, and decision-curve summaries should therefore be tied explicitly to the run-specific `real_test_predictions.csv` and `calibration_dca_metrics.json` files rather than quoted from historical drafts.

## Reproducibility
Random seeds were fixed across Python, NumPy, and PyTorch components, and deterministic settings were enabled for cuDNN where applicable. Submission-facing reruns were executed through a locked-run pipeline that snapshots the input data hashes, configuration file, feature allowlist, audit artifacts, environment snapshot, and generated outputs under a unique run identifier.
"""


def build_figure_order_md() -> str:
    return """# Figure Order and Manuscript Storyline

## Recommended Main-text Figure Order
1. **Fig. 1. Study cohort and patient-level split audit**
   - Purpose: establish sample provenance, target-center split integrity, and absence of patient overlap.
2. **Fig. 2. Cross-center baseline and event-time shift**
   - Purpose: show why source-only transport is clinically non-trivial.
3. **Fig. 3. External validation performance comparison**
   - Purpose: present the core held-out discrimination result.
4. **Fig. 4. Calibration and decision-curve evidence**
   - Purpose: show whether held-out probabilities and threshold-based utility survive external testing.

## Recommended Supplementary Figure
- **Fig. S1. Target test-set IDH burden across major clinical subgroups**
  - Purpose: document heterogeneity in held-out target burden without overclaiming subgroup superiority.

## Results Paragraph Order
1. Study population and patient-level split audit.
2. Cross-center baseline and event-time heterogeneity.
3. Held-out external validation performance.
4. Calibration and decision-curve evidence from real predictions.
5. Supplementary target subgroup burden.

## Core Narrative in One Sentence
This manuscript argues that clinically meaningful cross-center shift exists between Shenyi and Fuding, and that a locally updated survival model evaluated on a patient-level held-out target test set can be judged only through a locked run that jointly regenerates discrimination, calibration, and decision-curve evidence.
"""


def build_figure_legends_md() -> str:
    return """# Figure Legends

## Figure 1. Study cohort and patient-level split audit.
This figure summarizes the final analytic cohorts and the patient-level split of the Fuding target-center cohort into updating, validation, and held-out test subsets. The figure documents final session counts, patient counts, and zero patient overlap across the three target subsets.

## Figure 2. Cross-center baseline and event-time shift.
Panel a shows center-level differences in representative baseline clinical variables between the Shenyi source cohort and the Fuding target cohort. Panel b shows the distribution of IDH timing stages across centers, illustrating that cross-center heterogeneity involves both baseline covariate shift and event-time shift.

## Figure 3. External validation performance comparison.
Held-out target-center discrimination performance is compared between the zero-shot baseline and the locally updated survival model. Points indicate C-index estimates and horizontal bars indicate 95% confidence intervals.

## Figure 4. Calibration and decision-curve evidence.
Panels a-b show held-out calibration at 60 and 120 minutes, comparing predicted probabilities with observed event rates across risk bins. Panels c-d show decision-curve analysis at the same horizons, comparing model-guided decisions with treat-all and treat-none strategies.

## Figure S1. Target test-set IDH burden across major clinical subgroups.
Observed IDH event rates are shown across major held-out target subgroups. This supplementary figure is intended to describe subgroup burden rather than to claim subgroup-specific model advantage.
"""


def build_index_md() -> str:
    return """# Manuscript Directory Index

This folder contains the working manuscript assets for the hemodialysis target-center updating paper.

## Core Files
- `00_Figure_Order_and_Storyline.md`: locked submission figure order and section logic.
- `01_Figure_Legends.md`: figure legends aligned to the compressed 4+1 figure set.
- `02_Results.md`: current results draft synced to audit status and locked-run outputs.
- `03_Discussion.md`: discussion draft; update only after the locked run is accepted as the sole evidence source.
- `04_Introduction.md`: introduction draft.
- `05_Methods.md`: methods draft aligned to the audited allowlist and patient-level split.
- `06_Abstract_Structured.md`: structured abstract template.
- `07_Title_Page_and_Key_Message.md`: title options, running title, and key message placeholders.
- `08_Main_Manuscript_Full_Draft.md`: integrated working draft pending post-rerun reconciliation.

## Suggested Next Writing Order
1. Verify the locked-run evidence package and confirm that `result_consistency_report.json` passes.
2. Finalize Results against the run-specific JSON and figure outputs only.
3. Update Discussion to match the held-out evidence strength and calibration limitations.
4. Fill the structured abstract last.

## Critical Honesty Checks
- Do not present target-center labeled updating as causal identification.
- Do not quote any performance, calibration, or DCA values unless the consistency report confirms the current run is submission-eligible.
- Keep the residual underestimation in the external center explicit whenever calibration is discussed.
"""


def main() -> None:
    MANUSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    (MANUSCRIPT_DIR / "02_Results.md").write_text(build_results_md(), encoding="utf-8")
    (MANUSCRIPT_DIR / "05_Methods.md").write_text(build_methods_md(), encoding="utf-8")
    (MANUSCRIPT_DIR / "00_Figure_Order_and_Storyline.md").write_text(build_figure_order_md(), encoding="utf-8")
    (MANUSCRIPT_DIR / "01_Figure_Legends.md").write_text(build_figure_legends_md(), encoding="utf-8")
    (MANUSCRIPT_DIR / "MANUSCRIPT_INDEX.md").write_text(build_index_md(), encoding="utf-8")
    print("Submission manuscript placeholders synced.")


if __name__ == "__main__":
    main()
