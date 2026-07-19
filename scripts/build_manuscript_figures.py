"""Build draft manuscript figures from frozen aggregate binary-study artifacts.

This script does not fit models or rebuild data. It deliberately excludes legacy
time-to-event tables and prediction-dependent plots such as calibration and DCA.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = ROOT / "experiments" / "audit" / "binary_data_preflight.json"
MAIN_DIR = ROOT / "experiments" / "final_results" / "main_mechanism_aware"
LATENT = ROOT / "experiments" / "final_results" / "multiseed_seed42" / "latent_analysis.json"
SEED_DIRS = [
    ROOT / "experiments" / "final_results" / "multiseed_seed42",
    ROOT / "experiments" / "final_results" / "multiseed_seed7",
    ROOT / "experiments" / "final_results" / "multiseed_seed13",
    ROOT / "experiments" / "final_results" / "multiseed_seed99",
    MAIN_DIR,
]
DEFAULT_OUTPUT = ROOT / "figures" / "manuscript_draft"

PHYSIO = "#0072B2"
TREATMENT = "#D55E00"
SOURCE = "#6B7280"
UPDATED = "#009E73"
LOCAL = "#CC79A7"
INK = "#20242A"
LIGHT = "#E5E7EB"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def _clean_axis(ax: plt.Axes, *, grid: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis="y", color=LIGHT, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.05, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")


def _draft_note(fig: plt.Figure) -> None:
    fig.text(
        0.995,
        0.005,
        "DRAFT | aggregate artifacts | exact data restoration pending",
        ha="right",
        va="bottom",
        fontsize=6.5,
        color=SOURCE,
    )


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    _draft_note(fig)
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _load_artifacts() -> dict[str, Any]:
    preflight = _read_json(PREFLIGHT)
    main_eval = _read_json(MAIN_DIR / "evaluation.json")
    split = _read_json(MAIN_DIR / "split_manifest.json")
    latent = _read_json(LATENT)
    shap = _read_json(MAIN_DIR / "shap_analysis_full.json")
    seed_runs = [_read_json(path / "evaluation.json") for path in SEED_DIRS]
    return {
        "preflight": preflight,
        "main_eval": main_eval,
        "split": split,
        "latent": latent,
        "shap": shap,
        "seed_runs": seed_runs,
    }


def _validate(artifacts: dict[str, Any]) -> None:
    preflight = artifacts["preflight"]
    main_eval = artifacts["main_eval"]
    split = artifacts["split"]
    runs = artifacts["seed_runs"]
    if not preflight.get("passed"):
        raise ValueError("Binary data preflight did not pass.")
    if main_eval.get("contract") != "dual_binary_hemodynamic_v1":
        raise ValueError("Main result does not use the dual-binary contract.")
    seeds = [int(run["initialization_seed"]) for run in runs]
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError(f"Expected five unique initialization seeds, found {seeds}.")
    split_seed = int(main_eval["split_seed"])
    if int(split["split_seed"]) != split_seed or any(int(run["split_seed"]) != split_seed for run in runs):
        raise ValueError("Split seeds differ across the selected artifacts.")
    expected_sessions = int(split["target_test"]["sessions"])
    expected_patients = int(split["target_test"]["patients"])
    for run in runs:
        for endpoint in ("idh", "ih"):
            result = run["metrics"][endpoint]["updated_mlp"]
            if int(result["n_sessions"]) != expected_sessions or int(result["n_patients"]) != expected_patients:
                raise ValueError("Held-out target counts differ across seed artifacts.")


def _figure1(artifacts: dict[str, Any], output_dir: Path, rows: list[dict[str, Any]]) -> None:
    preflight = artifacts["preflight"]["cohorts"]
    split = artifacts["split"]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.65), gridspec_kw={"width_ratios": [1.0, 1.15, 1.25]})
    fig.suptitle("Cross-center outcome burden differed before target-center updating", fontweight="bold", y=1.01)

    ax = axes[0]
    centers = ["Source", "Target"]
    sessions = [preflight["source"]["sessions"], preflight["target"]["sessions"]]
    patients = [preflight["source"]["patients"], preflight["target"]["patients"]]
    bars = ax.bar(centers, np.array(sessions) / 1000, color=[SOURCE, UPDATED], width=0.58, zorder=2)
    for bar, n_session, n_patient in zip(bars, sessions, patients):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            f"{n_session:,} sessions\n{n_patient:,} patients",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_ylabel("Sessions (thousands)")
    ax.set_ylim(0, max(sessions) / 1000 * 1.28)
    ax.set_title("Cohort size")
    _clean_axis(ax, grid=True)
    _panel_label(ax, "A")
    for center, value, patient in zip(centers, sessions, patients):
        rows.append({"figure": "Fig1", "panel": "A", "endpoint": "", "series": center, "metric": "sessions", "value": value, "lower": "", "upper": "", "seed": "", "artifact": _relative(PREFLIGHT)})
        rows.append({"figure": "Fig1", "panel": "A", "endpoint": "", "series": center, "metric": "patients", "value": patient, "lower": "", "upper": "", "seed": "", "artifact": _relative(PREFLIGHT)})

    ax = axes[1]
    x = np.arange(2)
    width = 0.34
    source_rates = [preflight["source"]["endpoints"][e]["event_rate"] * 100 for e in ("idh", "ih")]
    target_rates = [preflight["target"]["endpoints"][e]["event_rate"] * 100 for e in ("idh", "ih")]
    ax.bar(x - width / 2, source_rates, width, label="Source", color=SOURCE, zorder=2)
    ax.bar(x + width / 2, target_rates, width, label="Target", color=UPDATED, hatch="//", zorder=2)
    for xpos, values in ((x - width / 2, source_rates), (x + width / 2, target_rates)):
        for px, value in zip(xpos, values):
            ax.text(px, value + 1.0, f"{value:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, ["IDH", "IH"])
    ax.set_ylabel("Sessions with event (%)")
    ax.set_ylim(0, 46)
    ax.set_title("Opposing endpoint shifts")
    ax.legend(frameon=False, loc="upper right")
    _clean_axis(ax, grid=True)
    _panel_label(ax, "B")
    for endpoint, sr, tr in zip(("IDH", "IH"), source_rates, target_rates):
        rows.extend(
            [
                {"figure": "Fig1", "panel": "B", "endpoint": endpoint, "series": "Source", "metric": "event_rate_percent", "value": sr, "lower": "", "upper": "", "seed": "", "artifact": _relative(PREFLIGHT)},
                {"figure": "Fig1", "panel": "B", "endpoint": endpoint, "series": "Target", "metric": "event_rate_percent", "value": tr, "lower": "", "upper": "", "seed": "", "artifact": _relative(PREFLIGHT)},
            ]
        )

    ax = axes[2]
    split_names = ["Update", "Validation", "Held-out test"]
    keys = ["target_train", "target_val", "target_test"]
    split_patients = [split[key]["patients"] for key in keys]
    colors = [PHYSIO, TREATMENT, UPDATED]
    left = 0
    for name, key, count, color in zip(split_names, keys, split_patients, colors):
        ax.barh([0], [count], left=left, height=0.45, color=color, edgecolor="white", label=name)
        ax.text(left + count / 2, 0, f"{name}\n{count} patients", color="white", ha="center", va="center", fontsize=7.5, fontweight="bold")
        rows.append({"figure": "Fig1", "panel": "C", "endpoint": "", "series": name, "metric": "patients", "value": count, "lower": "", "upper": "", "seed": "20260715 split", "artifact": _relative(MAIN_DIR / "split_manifest.json")})
        rows.append({"figure": "Fig1", "panel": "C", "endpoint": "", "series": name, "metric": "sessions", "value": split[key]["sessions"], "lower": "", "upper": "", "seed": "20260715 split", "artifact": _relative(MAIN_DIR / "split_manifest.json")})
        left += count
    session_text = " | ".join(f"{name}: {split[key]['sessions']:,} sessions" for name, key in zip(split_names, keys))
    ax.text(0.5, 0.17, session_text, transform=ax.transAxes, ha="center", va="top", fontsize=7.2, color=INK)
    ax.text(0.5, -0.10, "Patient-level split; no target-patient overlap", transform=ax.transAxes, ha="center", fontsize=8)
    ax.set_xlim(0, sum(split_patients))
    ax.set_ylim(-0.55, 0.55)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title("Target-center allocation")
    for spine in ax.spines.values():
        spine.set_visible(False)
    _panel_label(ax, "C")

    fig.tight_layout()
    _save(fig, output_dir, "Fig1_Cohort_Shift_and_Design")


def _box(ax: plt.Axes, xy: tuple[float, float], width: float, height: float, text: str, color: str, *, dashed: bool = False) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.02",
        linewidth=1.4,
        edgecolor=color,
        facecolor="white",
        linestyle="--" if dashed else "-",
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=8, color=INK)


def _figure2(output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    fig.suptitle("Outcome-specific updating aligns one representation and retains the other", fontweight="bold", y=1.01)
    for ax, endpoint, align_label, retain_label, align_color in (
        (axes[0], "IDH", "Physiology/history\n20 features\nCORAL aligned", "Treatment context\n4 features\nretained", PHYSIO),
        (axes[1], "IH", "Treatment context\n4 features\nCORAL aligned", "Physiology/history\n20 features\nretained", TREATMENT),
    ):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 7)
        ax.axis("off")
        ax.set_title(endpoint, color=align_color, fontweight="bold", pad=7)
        _box(ax, (0.2, 2.55), 1.6, 1.1, "Session-start\nfeatures", SOURCE)
        _box(ax, (2.6, 4.25), 2.5, 1.25, align_label, align_color)
        _box(ax, (2.6, 1.10), 2.5, 1.25, retain_label, SOURCE, dashed=True)
        _box(ax, (6.0, 2.55), 1.8, 1.1, "Concatenate\nrepresentations", UPDATED)
        _box(ax, (8.45, 2.55), 1.25, 1.1, "Event\nrisk", INK)
        ax.annotate("", xy=(2.55, 4.8), xytext=(1.85, 3.3), arrowprops={"arrowstyle": "->", "color": align_color, "lw": 1.4})
        ax.annotate("", xy=(2.55, 1.72), xytext=(1.85, 2.9), arrowprops={"arrowstyle": "->", "color": SOURCE, "lw": 1.2, "linestyle": "--"})
        ax.annotate("", xy=(5.95, 3.25), xytext=(5.15, 4.75), arrowprops={"arrowstyle": "->", "color": align_color, "lw": 1.4})
        ax.annotate("", xy=(5.95, 2.9), xytext=(5.15, 1.72), arrowprops={"arrowstyle": "->", "color": SOURCE, "lw": 1.2, "linestyle": "--"})
        ax.annotate("", xy=(8.4, 3.1), xytext=(7.85, 3.1), arrowprops={"arrowstyle": "->", "color": INK, "lw": 1.3})
        ax.text(3.85, 6.15, "Source + labeled target", ha="center", fontsize=8, color=INK)
        ax.text(3.85, 5.78, "supervised loss + 0.01 x CORAL", ha="center", fontsize=7.5, color=SOURCE)
        ax.text(9.08, 1.65, "Platt calibration\non target validation", ha="center", fontsize=7.5, color=SOURCE)
        ax.plot([9.08, 9.08], [2.5, 2.05], color=SOURCE, linewidth=1.0, linestyle=":")
    _panel_label(axes[0], "A")
    _panel_label(axes[1], "B")
    fig.text(0.5, 0.01, "Clinical grouping is prespecified but not a validated biological decomposition.", ha="center", fontsize=7.5, color=SOURCE)
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    _save(fig, output_dir, "Fig2_Outcome_Specific_Framework")


def _ci(metric: dict[str, Any]) -> tuple[float, float, float]:
    auc = metric["roc_auc"]
    interval = metric["patient_cluster_bootstrap"]["roc_auc"]
    return auc, interval["lower"], interval["upper"]


def _figure3(artifacts: dict[str, Any], output_dir: Path, rows: list[dict[str, Any]]) -> None:
    evaluation = artifacts["main_eval"]
    seed_runs = sorted(artifacts["seed_runs"], key=lambda run: int(run["initialization_seed"]))
    fig = plt.figure(figsize=(12.0, 7.0))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.95], width_ratios=[1.5, 1.0, 1.0])
    ax_forest = fig.add_subplot(grid[:, 0])
    ax_idh = fig.add_subplot(grid[0, 1])
    ax_ih = fig.add_subplot(grid[0, 2])
    ax_secondary = fig.add_subplot(grid[1, 1:])
    fig.suptitle("Target-center updating improved and stabilized held-out discrimination", fontweight="bold", y=0.995)

    model_specs = [("Source MLP", "source_mlp", SOURCE, "o"), ("Updated MLP", "updated_mlp", UPDATED, "s"), ("Target-local logistic", "local_logistic", LOCAL, "D")]
    y_positions = {"idh": [6.0, 5.15, 4.3], "ih": [2.5, 1.65, 0.8]}
    for endpoint in ("idh", "ih"):
        for y, (label, key, color, marker) in zip(y_positions[endpoint], model_specs):
            value, lower, upper = _ci(evaluation["metrics"][endpoint][key])
            ax_forest.errorbar(value, y, xerr=[[value - lower], [upper - value]], fmt=marker, color=color, markersize=5.5, capsize=2.5, linewidth=1.2, zorder=3)
            ax_forest.text(0.895, y, f"{value:.3f} ({lower:.3f}-{upper:.3f})", ha="left", va="center", fontsize=7.2, clip_on=False)
            rows.append({"figure": "Fig3", "panel": "A", "endpoint": endpoint.upper(), "series": label, "metric": "roc_auc", "value": value, "lower": lower, "upper": upper, "seed": evaluation["initialization_seed"], "artifact": _relative(MAIN_DIR / "evaluation.json")})
    ax_forest.axhline(3.4, color=LIGHT, linewidth=1.0)
    ax_forest.text(0.685, 6.65, "IDH", fontweight="bold", color=PHYSIO)
    ax_forest.text(0.685, 3.15, "IH", fontweight="bold", color=TREATMENT)
    ax_forest.set_xlim(0.68, 0.89)
    ax_forest.set_ylim(0.25, 6.95)
    ax_forest.set_xlabel("Held-out target ROC AUC (95% patient-cluster CI)")
    forest_y = y_positions["idh"] + y_positions["ih"]
    ax_forest.set_yticks(forest_y, [spec[0] for spec in model_specs] * 2)
    ax_forest.tick_params(axis="y", length=0, pad=7)
    ax_forest.set_title("Seed 2024 main run")
    _clean_axis(ax_forest, grid=False)
    _panel_label(ax_forest, "A")

    for ax, endpoint, color, label in ((ax_idh, "idh", PHYSIO, "IDH"), (ax_ih, "ih", TREATMENT, "IH")):
        for run in seed_runs:
            seed = int(run["initialization_seed"])
            source_auc = run["metrics"][endpoint]["source_mlp"]["roc_auc"]
            updated_auc = run["metrics"][endpoint]["updated_mlp"]["roc_auc"]
            ax.plot([0, 1], [source_auc, updated_auc], color=LIGHT, linewidth=1.0, zorder=1)
            ax.scatter([0], [source_auc], color=SOURCE, marker="o", s=25, zorder=2)
            ax.scatter([1], [updated_auc], color=color, marker="s", s=25, zorder=2)
            midpoint = (source_auc + updated_auc) / 2
            ax.text(
                0.50,
                midpoint,
                str(seed),
                fontsize=6.2,
                va="center",
                ha="center",
                color=SOURCE,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.7, "alpha": 0.85},
            )
            for phase, value in (("Source MLP", source_auc), ("Updated MLP", updated_auc)):
                rows.append({"figure": "Fig3", "panel": "B" if endpoint == "idh" else "C", "endpoint": label, "series": phase, "metric": "roc_auc", "value": value, "lower": "", "upper": "", "seed": seed, "artifact": _relative(next(path for path in SEED_DIRS if _read_json(path / 'evaluation.json')['initialization_seed'] == seed) / "evaluation.json")})
        ax.set_xlim(-0.25, 1.35)
        ax.set_ylim(0.66 if endpoint == "idh" else 0.80, 0.89)
        ax.set_xticks([0, 1], ["Source", "Updated"])
        ax.set_ylabel("ROC AUC")
        ax.set_title(f"{label}: five seeds")
        _clean_axis(ax, grid=True)
    _panel_label(ax_idh, "B")
    _panel_label(ax_ih, "C")

    x = np.arange(2)
    width = 0.19
    secondary_specs = [("Source MLP", "source_mlp", SOURCE, ""), ("Updated MLP", "updated_mlp", UPDATED, "//"), ("Target-local logistic", "local_logistic", LOCAL, "xx")]
    for offset, (model_label, key, color, hatch) in zip((-width, 0, width), secondary_specs):
        values = [evaluation["metrics"][endpoint][key]["pr_auc"] for endpoint in ("idh", "ih")]
        ax_secondary.bar(x + offset, values, width, color=color, hatch=hatch, label=model_label, zorder=2)
        for endpoint, value in zip(("IDH", "IH"), values):
            rows.append({"figure": "Fig3", "panel": "D", "endpoint": endpoint, "series": model_label, "metric": "pr_auc", "value": value, "lower": "", "upper": "", "seed": evaluation["initialization_seed"], "artifact": _relative(MAIN_DIR / "evaluation.json")})
    ax_secondary.set_xticks(x, ["IDH", "IH"])
    ax_secondary.set_ylim(0, 0.9)
    ax_secondary.set_ylabel("Precision-recall AUC")
    ax_secondary.set_title("Class-imbalance-aware performance, seed 2024")
    ax_secondary.legend(frameon=False, ncol=3, loc="upper right")
    _clean_axis(ax_secondary, grid=True)
    _panel_label(ax_secondary, "D")

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, output_dir, "Fig3_Performance_and_Seed_Stability")


def _slope_panel(ax: plt.Axes, values: dict[str, tuple[float, float]], *, ylabel: str, title: str, ylim: tuple[float, float], percent: bool = False) -> None:
    endpoint_specs = [("IDH", PHYSIO, "o"), ("IH", TREATMENT, "s")]
    for endpoint, color, marker in endpoint_specs:
        before, after = values[endpoint.lower()]
        ax.plot([0, 1], [before, after], color=color, linewidth=1.8, marker=marker, markersize=5, label=endpoint)
        fmt = ".1%" if percent else ".3f"
        ax.text(-0.05, before, format(before, fmt), ha="right", va="center", fontsize=7, color=color)
        ax.text(1.05, after, format(after, fmt), ha="left", va="center", fontsize=7, color=color)
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(*ylim)
    ax.set_xticks([0, 1], ["Before", "After"])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _clean_axis(ax, grid=True)


def _figure4(artifacts: dict[str, Any], output_dir: Path, rows: list[dict[str, Any]]) -> None:
    latent = artifacts["latent"]
    shap = artifacts["shap"]
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.0))
    fig.suptitle("Representation changes were endpoint and metric dependent", fontweight="bold", y=0.995)

    metrics = {
        "rbf_mmd": {e: (latent[e]["before_alignment"]["mmd_rbf"], latent[e]["after_alignment"]["mmd_rbf"]) for e in ("idh", "ih")},
        "domain_auc": {e: (latent[e]["before_alignment"]["domain_classifier_auc"], latent[e]["after_alignment"]["domain_classifier_auc"]) for e in ("idh", "ih")},
        "physio_ratio": {e: (shap[e]["before_alignment"]["target"]["physiology_ratio"], shap[e]["after_alignment"]["target"]["physiology_ratio"]) for e in ("idh", "ih")},
        "spearman": {e: (shap[e]["before_alignment"]["cross_domain_correlation"]["spearman_r"], shap[e]["after_alignment"]["cross_domain_correlation"]["spearman_r"]) for e in ("idh", "ih")},
    }
    _slope_panel(axes[0, 0], metrics["rbf_mmd"], ylabel="RBF MMD", title="Latent discrepancy (seed 42)", ylim=(0, 0.31))
    _slope_panel(axes[0, 1], metrics["domain_auc"], ylabel="Domain-classifier AUC", title="Center separability (seed 42)", ylim=(0.45, 1.03))
    axes[0, 1].axhline(0.5, color=SOURCE, linestyle=":", linewidth=1.0)
    axes[0, 1].text(0.5, 0.515, "chance", ha="center", fontsize=7, color=SOURCE)
    _slope_panel(axes[1, 0], metrics["physio_ratio"], ylabel="Physiology/history attribution", title="Target SHAP group share (seed 2024; n=400)", ylim=(0.5, 1.02), percent=True)
    _slope_panel(axes[1, 1], metrics["spearman"], ylabel="Spearman rank correlation", title="Cross-center feature ranking (seed 2024)", ylim=(0.55, 1.0))
    for label, ax in zip(("A", "B", "C", "D"), axes.flat):
        _panel_label(ax, label)
    axes[0, 0].legend(frameon=False, loc="upper right")

    for metric_name, metric_values in metrics.items():
        for endpoint, (before, after) in metric_values.items():
            artifact = LATENT if metric_name in {"rbf_mmd", "domain_auc"} else MAIN_DIR / "shap_analysis_full.json"
            seed = 42 if metric_name in {"rbf_mmd", "domain_auc"} else 2024
            for phase, value in (("Before", before), ("After", after)):
                rows.append({"figure": "Fig4", "panel": metric_name, "endpoint": endpoint.upper(), "series": phase, "metric": metric_name, "value": value, "lower": "", "upper": "", "seed": seed, "artifact": _relative(artifact)})

    fig.text(0.5, 0.01, "RBF MMD decreased for IDH, but other discrepancy measures did not uniformly improve.", ha="center", fontsize=7.5, color=SOURCE)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    _save(fig, output_dir, "Fig4_Representation_Diagnostics")


def _write_source_table(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["figure", "panel", "endpoint", "series", "metric", "value", "lower", "upper", "seed", "artifact"]
    with (output_dir / "figure_source_data.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_alt_text(output_dir: Path) -> None:
    text = """# Draft Figure Alt Text

## Figure 1

Three panels compare the two centers and target split. The source cohort is larger, but IDH prevalence is much higher at the target center while IH prevalence is lower. The 430 target patients are separated into 219 update, 39 validation, and 172 held-out test patients without overlap.

## Figure 2

Two parallel model diagrams show the prespecified outcome-specific update. For IDH, the 20-feature physiology/history branch receives CORAL alignment and the four-feature treatment branch is retained. For IH, the treatment branch receives CORAL and the physiology/history branch is retained. Both branches feed a shared event-risk head followed by target-validation Platt calibration.

## Figure 3

The seed-2024 forest plot shows a large IDH AUC increase from source to updated MLP and a smaller IH increase. Updated MLP and target-local logistic intervals overlap for both outcomes. Five-seed panels show substantial source-model variation but nearly identical updated AUCs. Precision-recall AUC also improves after updating, especially for IDH.

## Figure 4

Four slope charts compare representation diagnostics before and after updating. IDH RBF MMD decreases and physiology/history SHAP share rises markedly, while IH changes little. Domain-classifier AUC remains high, especially for IDH, and cross-center SHAP rank correlation increases for IDH but slightly decreases for IH. The panels use different prespecified seeds and are exploratory.
"""
    (output_dir / "ALT_TEXT.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check-only", action="store_true", help="Validate aggregate inputs without writing figures.")
    args = parser.parse_args()
    artifacts = _load_artifacts()
    _validate(artifacts)
    if args.check_only:
        print("Aggregate manuscript figure inputs validated.")
        return

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _setup_style()
    rows: list[dict[str, Any]] = []
    _figure1(artifacts, output_dir, rows)
    _figure2(output_dir)
    _figure3(artifacts, output_dir, rows)
    _figure4(artifacts, output_dir, rows)
    _write_source_table(output_dir, rows)
    _write_alt_text(output_dir)
    print(f"Wrote draft figures and source tables to {output_dir}")


if __name__ == "__main__":
    main()
