"""Build the evidence-supported submission figure sequence without inventing gated panels."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
DRAFT = ROOT / "figures" / "manuscript_draft"
OUTPUT = ROOT / "figures" / "submission_staging"
ALERTS = ROOT / "clinical_utility_report" / "alert_analysis.csv"
ABLATION = ROOT / "experiments" / "confirmatory_results" / "architecture_matched_ablation.csv"
CONTRASTS = ROOT / "experiments" / "confirmatory_results" / "architecture_matched_paired_contrasts.csv"
SOURCE_DATA = DRAFT / "figure_source_data.csv"
PREFLIGHT = ROOT / "experiments" / "audit" / "binary_data_preflight.json"
SPLIT = ROOT / "experiments" / "final_results" / "main_mechanism_aware" / "split_manifest.json"
MAIN_CONFIG = ROOT / "experiments" / "final_results" / "main_mechanism_aware" / "config.yaml"

SOURCE = "#6B7280"
UPDATED = "#0072B2"
LOCAL = "#D55E00"
INK = "#20242A"
LIGHT = "#E5E7EB"
MODEL_COLORS = {"Source MLP": SOURCE, "Updated MLP": UPDATED, "Target-local logistic": LOCAL}
MODEL_MARKERS = {"Source MLP": "o", "Updated MLP": "s", "Target-local logistic": "D"}
ABLATION_LABELS = {
    "A_dual_no_alignment": "Dual: no alignment",
    "B_dual_global_coral": "Dual: global CORAL",
    "C_dual_random_feature_coral": "Dual: random-feature CORAL",
    "D_dual_outcome_specific_coral": "Dual: outcome-specific CORAL",
    "E_single_encoder_reference": "Single encoder",
}
ABLATION_COLORS = {
    model_id: color for model_id, color in zip(ABLATION_LABELS, (SOURCE, "#56B4E9", "#CC79A7", UPDATED, LOCAL))
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })


def _save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUTPUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.10, 1.05, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")


def _flow_box(
    ax: plt.Axes, xy: tuple[float, float], width: float, height: float, text: str,
    edge: str, *, dashed: bool = False, fontsize: float = 7.5,
) -> None:
    patch = FancyBboxPatch(
        xy, width, height, boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.25, edgecolor=edge, facecolor="white", linestyle="--" if dashed else "-",
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=fontsize)


def _workflow_panel(ax: plt.Axes, endpoint: str, *, align_physio: bool) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    aligned_color = UPDATED if align_physio else LOCAL
    aligned = "Physiology/history\n20 features\nCORAL aligned" if align_physio else "Treatment context\n4 features\nCORAL aligned"
    retained = "Treatment context\n4 features\nretained" if align_physio else "Physiology/history\n20 features\nretained"
    _flow_box(ax, (0.15, 1.85), 1.45, 0.85, "Session-start\nfeatures", SOURCE)
    _flow_box(ax, (2.25, 3.20), 2.25, 1.00, aligned, aligned_color)
    _flow_box(ax, (2.25, 0.35), 2.25, 1.00, retained, SOURCE, dashed=True)
    _flow_box(ax, (5.45, 1.85), 1.75, 0.85, "Concatenate\nrepresentations", UPDATED)
    _flow_box(ax, (8.05, 1.85), 1.20, 0.85, "Event\nrisk", INK)
    arrows = (
        ((1.62, 2.35), (2.22, 3.65), aligned_color, "-"),
        ((1.62, 2.10), (2.22, 0.85), SOURCE, "--"),
        ((4.52, 3.68), (5.42, 2.42), aligned_color, "-"),
        ((4.52, 0.85), (5.42, 2.08), SOURCE, "--"),
        ((7.22, 2.28), (8.02, 2.28), INK, "-"),
    )
    for start, end, color, linestyle in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": color, "lw": 1.2, "linestyle": linestyle})
    ax.text(3.38, 4.80, "Source + labeled target", ha="center", fontsize=8, color=INK)
    ax.text(3.38, 4.48, "supervised loss + 0.01 × CORAL", ha="center", fontsize=7, color=SOURCE)
    ax.text(8.66, 1.20, "Platt calibration\non target validation", ha="center", va="top", fontsize=7, color=SOURCE)
    ax.plot([8.66, 8.66], [1.82, 1.45], color=SOURCE, linestyle=":", linewidth=1)
    ax.set_title(endpoint, color=aligned_color, fontweight="bold", pad=2)


def _figure1() -> None:
    """Draw the registered study design and method natively as vector objects."""
    preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))["cohorts"]
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    config = yaml.safe_load(MAIN_CONFIG.read_text(encoding="utf-8"))
    model = config["model"]
    if len(model["physio_indices"]) != 20 or len(model["treat_indices"]) != 4:
        raise ValueError("Figure 1 feature-branch counts do not match the registered main config")
    if not model.get("use_coral") or not model.get("use_outcome_specific_alignment"):
        raise ValueError("Figure 1 requires the registered outcome-specific CORAL configuration")

    fig = plt.figure(figsize=(12, 7.8))
    grid = fig.add_gridspec(2, 6, height_ratios=[0.92, 1.08], hspace=0.38, wspace=0.65)
    ax_size = fig.add_subplot(grid[0, 0:2])
    ax_shift = fig.add_subplot(grid[0, 2:4])
    ax_split = fig.add_subplot(grid[0, 4:6])
    ax_idh = fig.add_subplot(grid[1, 0:3])
    ax_ih = fig.add_subplot(grid[1, 3:6])

    centers = ["Source", "Target"]
    sessions = [preflight["source"]["sessions"], preflight["target"]["sessions"]]
    patients = [preflight["source"]["patients"], preflight["target"]["patients"]]
    bars = ax_size.bar(centers, np.asarray(sessions) / 1000, color=[SOURCE, UPDATED], width=0.58)
    for bar, n_sessions, n_patients in zip(bars, sessions, patients):
        ax_size.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 4, f"{n_sessions:,} sessions\n{n_patients:,} patients", ha="center", fontsize=7.5)
    ax_size.set_ylabel("Sessions (thousands)")
    ax_size.set_ylim(0, 255)
    ax_size.set_title("Cohort size")
    ax_size.grid(axis="y", color=LIGHT, linewidth=0.7)
    ax_size.spines[["top", "right"]].set_visible(False)
    _panel_label(ax_size, "A")

    x = np.arange(2)
    width = 0.34
    source_rates = [100 * preflight["source"]["endpoints"][endpoint]["event_rate"] for endpoint in ("idh", "ih")]
    target_rates = [100 * preflight["target"]["endpoints"][endpoint]["event_rate"] for endpoint in ("idh", "ih")]
    ax_shift.bar(x - width / 2, source_rates, width, color=SOURCE, label="Source")
    ax_shift.bar(x + width / 2, target_rates, width, color=UPDATED, hatch="//", label="Target")
    for xpos, values in ((x - width / 2, source_rates), (x + width / 2, target_rates)):
        for position, value in zip(xpos, values):
            ax_shift.text(position, value + 0.9, f"{value:.1f}%", ha="center", fontsize=7.5)
    ax_shift.set_xticks(x, ["IDH", "IH"])
    ax_shift.set_ylabel("Sessions with event (%)")
    ax_shift.set_ylim(0, 45)
    ax_shift.set_title("Opposing endpoint shifts")
    ax_shift.legend(frameon=False, loc="upper right")
    ax_shift.grid(axis="y", color=LIGHT, linewidth=0.7)
    ax_shift.spines[["top", "right"]].set_visible(False)
    _panel_label(ax_shift, "B")

    keys = ["target_train", "target_val", "target_test"]
    names = ["Update", "Validation", "Held-out test"]
    colors = [UPDATED, LOCAL, "#009E73"]
    left = 0
    for key, name, color in zip(keys, names, colors):
        count = split[key]["patients"]
        ax_split.barh([0], [count], left=left, height=0.44, color=color, edgecolor="white")
        ax_split.text(left + count / 2, 0, f"{name}\n{count} patients", color="white", ha="center", va="center", fontsize=7, fontweight="bold")
        left += count
    ax_split.text(0.5, 0.20, " | ".join(f"{name}: {split[key]['sessions']:,} sessions" for key, name in zip(keys, names)), transform=ax_split.transAxes, ha="center", fontsize=6.5)
    ax_split.text(0.5, -0.08, "Patient-level split; no target-patient overlap", transform=ax_split.transAxes, ha="center", fontsize=7.5)
    ax_split.set_xlim(0, left)
    ax_split.set_ylim(-0.55, 0.55)
    ax_split.set_xticks([])
    ax_split.set_yticks([])
    ax_split.set_title("Target-center allocation")
    for spine in ax_split.spines.values():
        spine.set_visible(False)
    _panel_label(ax_split, "C")

    _workflow_panel(ax_idh, "IDH: align physiology/history", align_physio=True)
    _workflow_panel(ax_ih, "IH: align treatment context", align_physio=False)
    _panel_label(ax_idh, "D")
    _panel_label(ax_ih, "E")
    fig.suptitle("Study design and outcome-specific target-center updating", fontweight="bold", y=0.995)
    fig.text(
        0.5, 0.006,
        "Historical v1 cohort and split; clinical feature grouping is prespecified and does not establish mechanism superiority.",
        ha="center", fontsize=7.5, color=SOURCE,
    )
    fig.subplots_adjust(top=0.92, bottom=0.06, left=0.065, right=0.985)
    _save(fig, "Figure1_Study_Design_and_Updating_Framework")


def _forest(ax: plt.Axes, data: pd.DataFrame, metrics: list[tuple[str, str]], endpoint: str) -> None:
    subset = data.loc[data["endpoint"].eq(endpoint)]
    y = np.arange(len(metrics))
    offsets = {"Source MLP": -0.20, "Updated MLP": 0.0, "Target-local logistic": 0.20}
    for model in ("Source MLP", "Updated MLP", "Target-local logistic"):
        row = subset.loc[subset["model"].eq(model)].iloc[0]
        values = np.asarray([row[column] for column, _ in metrics], dtype=float)
        lowers = np.asarray([row[f"{column}_lower"] for column, _ in metrics], dtype=float)
        uppers = np.asarray([row[f"{column}_upper"] for column, _ in metrics], dtype=float)
        ax.errorbar(
            values, y + offsets[model], xerr=[values - lowers, uppers - values],
            fmt=MODEL_MARKERS[model], color=MODEL_COLORS[model], label=model,
            markersize=4.5, linewidth=1.2, capsize=2,
        )
    ax.set_yticks(y, [label for _, label in metrics])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Operating characteristic (95% patient-cluster CI)")
    ax.set_title(endpoint)
    ax.grid(axis="x", color=LIGHT, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)


def _figure3() -> None:
    alerts = pd.read_csv(ALERTS)
    fig = plt.figure(figsize=(12, 7.2))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.05])
    ax_idh = fig.add_subplot(grid[0, 0])
    ax_ih = fig.add_subplot(grid[0, 1])
    metrics = [("sensitivity", "Sensitivity"), ("specificity", "Specificity"), ("ppv", "PPV"), ("npv", "NPV")]
    _forest(ax_idh, alerts, metrics, "IDH")
    _forest(ax_ih, alerts, metrics, "IH")
    ax_idh.legend(frameon=False, ncol=3, bbox_to_anchor=(0, 1.25), loc="upper left")

    for column_index, endpoint in enumerate(("IDH", "IH")):
        ax = fig.add_subplot(grid[1, column_index])
        subset = alerts.loc[alerts["endpoint"].eq(endpoint)]
        models = ["Source MLP", "Updated MLP", "Target-local logistic"]
        x = np.arange(len(models))
        width = 0.32
        for offset, metric, label, hatch in (
            (-width / 2, "events_detected_per_1000", "Events detected", ""),
            (width / 2, "false_alerts_per_1000", "False alerts", "//"),
        ):
            values, lower, upper = [], [], []
            for model in models:
                row = subset.loc[subset["model"].eq(model)].iloc[0]
                values.append(row[metric])
                lower.append(row[f"{metric}_lower"])
                upper.append(row[f"{metric}_upper"])
            values, lower, upper = map(np.asarray, (values, lower, upper))
            ax.bar(
                x + offset, values, width, color=[MODEL_COLORS[model] for model in models],
                alpha=0.85, hatch=hatch, edgecolor="white", label=label,
                yerr=[values - lower, upper - values], capsize=2, linewidth=0.6,
            )
        ax.set_xticks(x, ["Source", "Updated", "Local\nlogistic"])
        ax.set_ylabel("Per 1,000 sessions")
        ax.set_title(f"{endpoint}: alert workload at validation-Youden thresholds")
        ax.grid(axis="y", color=LIGHT, linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, ncol=2)
    fig.suptitle("Registered operating diagnostics include patient-cluster uncertainty", fontweight="bold", y=0.995)
    fig.text(
        0.5, 0.006,
        "Thresholds were selected on validation patients by Youden's index and are not clinically prespecified deployment thresholds.",
        ha="center", fontsize=7.5, color=SOURCE,
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.96))
    _save(fig, "Figure3_Clinical_Utility")


def _figure2(ablation: pd.DataFrame, contrasts: pd.DataFrame) -> None:
    """Create the prespecified matched comparison only after all rows are registered."""
    required = {"model_id", "seed", "endpoint", "run_id", "roc_auc", "brier", "status"}
    if not required.issubset(ablation.columns) or len(ablation) != 50 or ablation["run_id"].isna().any():
        raise ValueError("Figure 2 requires the complete registered 50-row endpoint table")
    if not ablation["status"].eq("registered_confirmatory_result").all():
        raise ValueError("Figure 2 requires every row to be a registered confirmatory result")
    required_contrasts = {
        "endpoint", "seed", "metric", "model_a", "model_b", "delta_a_minus_b",
        "lower", "upper", "bootstrap_valid_replicates", "status",
    }
    if not required_contrasts.issubset(contrasts.columns):
        raise ValueError("Figure 2 requires the registered paired-contrast table")
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    comparator_ids = [model_id for model_id in ABLATION_LABELS if model_id != "D_dual_outcome_specific_coral"]
    for column_index, endpoint in enumerate(("IDH", "IH")):
        ax = axes[0, column_index]
        subset = ablation.loc[ablation["endpoint"].eq(endpoint)]
        seed_matrix = subset.pivot(index="seed", columns="model_id", values="roc_auc")
        seed_matrix = seed_matrix.reindex(index=[7, 13, 42, 99, 2024], columns=list(ABLATION_LABELS))
        if seed_matrix.isna().any().any():
            raise ValueError(f"Figure 2 requires a complete paired seed grid for {endpoint}/roc_auc")
        for _, values in seed_matrix.iterrows():
            ax.plot(
                range(len(ABLATION_LABELS)), values.to_numpy(dtype=float),
                color=LIGHT, linewidth=0.8, alpha=0.75, zorder=1,
            )
        for model_index, model_id in enumerate(ABLATION_LABELS):
            model = subset.loc[subset["model_id"].eq(model_id)].sort_values("seed")
            values = model["roc_auc"].to_numpy(dtype=float)
            ax.scatter(
                np.full(len(model), model_index) + np.linspace(-0.05, 0.05, len(model)), values,
                color=ABLATION_COLORS[model_id], marker=["o", "s", "D", "^", "P"][model_index],
                s=30, alpha=0.95, zorder=2,
                label=ABLATION_LABELS[model_id] if column_index == 0 else None,
            )
            ax.hlines(values.mean(), model_index - 0.15, model_index + 0.15, color=INK, linewidth=1.4)
        ax.set_xticks(range(len(ABLATION_LABELS)), ["No\nalign", "Global", "Random", "Outcome\nspecific", "Single"])
        ax.set_ylabel("ROC AUC")
        ax.set_title(f"{endpoint}: absolute held-out discrimination")
        ax.grid(axis="y", color=LIGHT, linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)

        contrast_ax = axes[1, column_index]
        contrast_subset = contrasts.loc[
            contrasts["endpoint"].eq(endpoint)
            & contrasts["metric"].eq("roc_auc")
            & contrasts["model_a"].eq("D_dual_outcome_specific_coral")
            & contrasts["model_b"].isin(comparator_ids)
        ]
        if (
            len(contrast_subset) != 20
            or not contrast_subset["status"].eq("registered_confirmatory_paired_contrast").all()
            or not contrast_subset["bootstrap_valid_replicates"].eq(1000).all()
        ):
            raise ValueError(f"Figure 2 requires 20 registered ROC AUC contrasts for {endpoint}")
        for comparator_index, comparator in enumerate(comparator_ids):
            model = contrast_subset.loc[contrast_subset["model_b"].eq(comparator)].sort_values("seed")
            y = comparator_index + np.linspace(-0.22, 0.22, len(model))
            values = model["delta_a_minus_b"].to_numpy(dtype=float)
            lower = model["lower"].to_numpy(dtype=float)
            upper = model["upper"].to_numpy(dtype=float)
            contrast_ax.errorbar(
                values, y, xerr=[values - lower, upper - values], fmt="o",
                color=ABLATION_COLORS[comparator], markersize=3.5, linewidth=0.8,
                capsize=1.5, alpha=0.9,
            )
            contrast_ax.vlines(values.mean(), comparator_index - 0.28, comparator_index + 0.28, color=INK, linewidth=1.5)
        contrast_ax.axvline(0, color=SOURCE, linestyle="--", linewidth=1)
        contrast_ax.set_yticks(range(len(comparator_ids)), [ABLATION_LABELS[model_id] for model_id in comparator_ids])
        contrast_ax.invert_yaxis()
        contrast_ax.set_xlabel("ROC AUC difference: outcome-specific CORAL minus comparator")
        contrast_ax.set_title(f"{endpoint}: paired patient-cluster contrasts")
        contrast_ax.grid(axis="x", color=LIGHT, linewidth=0.7)
        contrast_ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, ncol=3, bbox_to_anchor=(0, 1.35), loc="upper left")
    fig.suptitle("Architecture-matched comparisons across five locked seeds", fontweight="bold", y=0.995)
    fig.text(
        0.5, 0.008,
        "Top: individual seeds with paired lines and mean ticks. Bottom: seed-specific paired differences with 95% patient-cluster CIs; vertical ticks are descriptive means.",
        ha="center", fontsize=7.5, color=SOURCE,
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.95))
    _save(fig, "Figure2_Architecture_Matched_Ablation")


def _copy_registered_figure(source_stem: str, target_stem: str) -> None:
    for extension in ("png", "pdf"):
        shutil.copy2(DRAFT / f"{source_stem}.{extension}", OUTPUT / f"{target_stem}.{extension}")


def _write_status(ablation_complete: bool) -> list[dict[str, str]]:
    rows = [
        {"figure": "Figure 1", "title": "Study design and target-center updating framework", "status": "staged_registered_vector", "gate": "none"},
        {"figure": "Figure 2", "title": "Architecture-matched ablation", "status": "ready" if ablation_complete else "blocked", "gate": "" if ablation_complete else "25_accepted_server_runs_required"},
        {"figure": "Figure 3", "title": "Clinical utility", "status": "staged_registered_fixed_threshold_diagnostics", "gate": "clinical_threshold_prespecification_required_for_DCA_interpretation"},
        {"figure": "Figure 4", "title": "Calibration and transportability", "status": "staged_registered_evidence", "gate": "none_for_descriptive_interpretation"},
        {"figure": "Figure 5", "title": "Exploratory representation analysis", "status": "staged_exploratory", "gate": "retain_exploratory_label"},
    ]
    with (OUTPUT / "figure_status.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _write_alt_text(ablation_complete: bool) -> None:
    figure2_text = (
        "Upper panels show absolute held-out ROC AUC for five strategies across five paired seeds. "
        "Lower panels show outcome-specific-CORAL minus comparator ROC AUC differences with 95% "
        "patient-cluster intervals for each seed."
        if ablation_complete
        else "Not generated. Architecture-matched server evidence has not passed acceptance."
    )
    text = f"""# Submission-Staging Figure Alt Text

## Figure 1

The upper section compares source and target cohort size, opposing IDH and IH prevalence shifts, and the patient-disjoint target update, validation, and test sets. The lower section shows the two-branch updating design: physiology/history is CORAL-aligned for IDH and treatment context for IH, while the other branch is retained.

## Figure 2

{figure2_text}

## Figure 3

Patient-cluster interval plots compare sensitivity, specificity, PPV, and NPV for source MLP, updated MLP, and target-local logistic regression. Workload panels show events detected and false alerts per 1,000 sessions at validation-derived Youden thresholds. These thresholds are not clinical deployment recommendations.

## Figure 4

Held-out ROC, precision-recall, and calibration panels compare the three registered models for IDH and IH. Updated MLP and target-local logistic curves largely overlap and are better calibrated than the source MLP.

## Figure 5

Exploratory slope plots show that latent discrepancy, domain separability, SHAP group contribution, and cross-center feature-rank consistency change in different directions. The figure does not establish domain invariance or causal mechanism.
"""
    (OUTPUT / "ALT_TEXT.md").write_text(text, encoding="utf-8")


def main() -> None:
    required_drafts = [
        "Fig4_Representation_Diagnostics.png", "Fig4_Representation_Diagnostics.pdf",
        "Fig5_ROC_PR_Calibration.png", "Fig5_ROC_PR_Calibration.pdf", "figure_source_data.csv",
    ]
    missing = [name for name in required_drafts if not (DRAFT / name).exists()]
    if missing:
        raise FileNotFoundError(f"Build and validate draft registered figures first: {missing}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _style()
    _figure1()
    _figure3()
    _copy_registered_figure("Fig5_ROC_PR_Calibration", "Figure4_Calibration_and_Transportability")
    _copy_registered_figure("Fig4_Representation_Diagnostics", "Figure5_Exploratory_Representation")
    ablation = pd.read_csv(ABLATION)
    contrasts = pd.read_csv(CONTRASTS)
    ablation_complete = (
        len(ablation) == 50
        and ablation["run_id"].notna().all()
        and ablation["status"].eq("registered_confirmatory_result").all()
        and len(contrasts) == 160
        and contrasts["status"].eq("registered_confirmatory_paired_contrast").all()
    )
    if ablation_complete:
        _figure2(ablation, contrasts)
    status_rows = _write_status(ablation_complete)
    _write_alt_text(ablation_complete)
    source_rows = pd.read_csv(SOURCE_DATA, keep_default_na=False)
    figure1_rows = []
    preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))["cohorts"]
    for cohort_key, cohort_label in (("source", "Source"), ("target", "Target")):
        cohort = preflight[cohort_key]
        for metric in ("sessions", "patients"):
            figure1_rows.append({
                "figure": "Fig1", "panel": "A", "endpoint": "", "series": cohort_label,
                "metric": metric, "x": "", "value": cohort[metric], "lower": "", "upper": "",
                "n": "", "seed": "", "run_id": "", "evidence_id": "dual_binary_hemodynamic_v1_frozen_20260719",
                "artifact": str(PREFLIGHT.relative_to(ROOT)), "artifact_sha256": _sha256(PREFLIGHT),
                "submission_figure": "Figure 1",
            })
        for endpoint in ("idh", "ih"):
            figure1_rows.append({
                "figure": "Fig1", "panel": "B", "endpoint": endpoint.upper(), "series": cohort_label,
                "metric": "event_rate_percent", "x": "", "value": 100 * cohort["endpoints"][endpoint]["event_rate"],
                "lower": "", "upper": "", "n": cohort["sessions"], "seed": "", "run_id": "",
                "evidence_id": "dual_binary_hemodynamic_v1_frozen_20260719",
                "artifact": str(PREFLIGHT.relative_to(ROOT)), "artifact_sha256": _sha256(PREFLIGHT),
                "submission_figure": "Figure 1",
            })
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    for split_key, split_label in (("target_train", "Update"), ("target_val", "Validation"), ("target_test", "Held-out test")):
        for metric in ("patients", "sessions"):
            figure1_rows.append({
                "figure": "Fig1", "panel": "C", "endpoint": "", "series": split_label,
                "metric": metric, "x": "", "value": split[split_key][metric], "lower": "", "upper": "",
                "n": "", "seed": split["split_seed"], "run_id": "binary_20260719_044949_c6c955a30fc2",
                "evidence_id": "binary_20260719_044949_c6c955a30fc2",
                "artifact": str(SPLIT.relative_to(ROOT)), "artifact_sha256": _sha256(SPLIT),
                "submission_figure": "Figure 1",
            })
    main_config = yaml.safe_load(MAIN_CONFIG.read_text(encoding="utf-8"))
    for endpoint, aligned_branch, retained_branch, panel in (
        ("IDH", "physiology_history", "treatment_context", "D"),
        ("IH", "treatment_context", "physiology_history", "E"),
    ):
        for branch, aligned in ((aligned_branch, 1), (retained_branch, 0)):
            figure1_rows.append({
                "figure": "Fig1", "panel": panel, "endpoint": endpoint, "series": branch,
                "metric": "coral_alignment_applied", "x": "", "value": aligned, "lower": "", "upper": "",
                "n": len(main_config["model"]["physio_indices" if branch == "physiology_history" else "treat_indices"]),
                "seed": main_config["training"]["initialization_seed"],
                "run_id": "binary_20260719_044949_c6c955a30fc2", "evidence_id": "binary_20260719_044949_c6c955a30fc2",
                "artifact": str(MAIN_CONFIG.relative_to(ROOT)), "artifact_sha256": _sha256(MAIN_CONFIG),
                "submission_figure": "Figure 1",
            })
    mapped = pd.concat([
        pd.DataFrame(figure1_rows),
        source_rows.loc[source_rows["figure"].eq("Fig5")].assign(submission_figure="Figure 4"),
        source_rows.loc[source_rows["figure"].eq("Fig4")].assign(submission_figure="Figure 5"),
    ], ignore_index=True)
    alert_rows = []
    alert_hash = _sha256(ALERTS)
    alerts = pd.read_csv(ALERTS)
    for _, row in alerts.iterrows():
        for metric in (
            "sensitivity", "specificity", "ppv", "npv", "events_detected_per_1000",
            "false_alerts_per_1000",
        ):
            alert_rows.append({
                "figure": "Fig3", "panel": row["endpoint"], "endpoint": row["endpoint"],
                "series": row["model"], "metric": metric, "x": "", "value": row[metric],
                "lower": row[f"{metric}_lower"], "upper": row[f"{metric}_upper"], "n": row["sessions"],
                "seed": 2024, "run_id": row["run_id"], "evidence_id": row["run_id"],
                "artifact": str(ALERTS.relative_to(ROOT)), "artifact_sha256": alert_hash,
                "submission_figure": "Figure 3",
            })
    mapped = pd.concat([mapped, pd.DataFrame(alert_rows)], ignore_index=True)
    if ablation_complete:
        ablation_hash = _sha256(ABLATION)
        ablation_rows = ablation.copy()
        ablation_rows = ablation_rows.assign(
            figure="Fig2", panel=ablation_rows["endpoint"], series=ablation_rows["model_id"],
            metric="architecture_matched_run", x=ablation_rows["seed"], value=ablation_rows["roc_auc"],
            lower=ablation_rows["roc_auc_lower"], upper=ablation_rows["roc_auc_upper"], n="",
            evidence_id=ablation_rows["run_id"], artifact=str(ABLATION.relative_to(ROOT)),
            artifact_sha256=ablation_hash, submission_figure="Figure 2",
        )
        mapped = pd.concat([mapped, ablation_rows[mapped.columns]], ignore_index=True)
        contrast_hash = _sha256(CONTRASTS)
        roc_contrasts = contrasts.loc[contrasts["metric"].eq("roc_auc")].copy()
        roc_contrasts = roc_contrasts.assign(
            figure="Fig2", panel=roc_contrasts["endpoint"], series=roc_contrasts["model_b"],
            metric="roc_auc_delta_outcome_specific_minus_comparator", x=roc_contrasts["seed"],
            value=roc_contrasts["delta_a_minus_b"], n="", run_id=roc_contrasts["run_id_a"],
            evidence_id=roc_contrasts["run_id_a"].astype(str) + "|" + roc_contrasts["run_id_b"].astype(str),
            artifact=str(CONTRASTS.relative_to(ROOT)), artifact_sha256=contrast_hash,
            submission_figure="Figure 2",
        )
        mapped = pd.concat([mapped, roc_contrasts[mapped.columns]], ignore_index=True)
    mapped.to_csv(OUTPUT / "figure_source_data.csv", index=False)
    artifacts = [
        path for path in OUTPUT.iterdir()
        if path.name != "provenance.json" and path.is_file()
    ]
    provenance: dict[str, Any] = {
        "package_status": "submission_staging_5_of_5_figures" if ablation_complete else "submission_staging_4_of_5_figures",
        "figure_status": status_rows,
        "sources": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in (
                SOURCE_DATA, PREFLIGHT, SPLIT, MAIN_CONFIG, ALERTS, ABLATION, CONTRASTS,
                ROOT / "clinical_utility_report" / "provenance.json",
            )
        },
        "outputs": {path.name: _sha256(path) for path in sorted(artifacts)},
        "claim_limits": [
            "figure2_not_generated_without_server_acceptance",
            "figure3_thresholds_are_validation_youden_not_clinical_recommendations",
            "figure5_exploratory_only",
        ],
    }
    (OUTPUT / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote submission-staging figure package to {OUTPUT}")


if __name__ == "__main__":
    main()
