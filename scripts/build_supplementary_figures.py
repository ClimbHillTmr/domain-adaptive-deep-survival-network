"""Build evidence-linked supplementary figures without fitting models."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "figures" / "submission_staging"
REGISTRY = ROOT / "experiments" / "evidence_registry" / "run_registry.csv"
STEM = "SupplementaryFigureS1_Seed_Level_AUC_Stability"
SOURCE_DATA = OUTPUT / "SupplementaryFigureS1_source_data.csv"
PROVENANCE = OUTPUT / "SupplementaryFigureS1_provenance.json"
ALT_TEXT = OUTPUT / "SupplementaryFigureS1_ALT_TEXT.md"

SOURCE = "#6B7280"
UPDATED = "#0072B2"
LIGHT = "#E5E7EB"
SEED_COLORS = {
    7: "#0072B2",
    13: "#D55E00",
    42: "#009E73",
    99: "#CC79A7",
    2024: "#6B7280",
}
SEED_MARKERS = {7: "o", 13: "s", 42: "D", 99: "^", 2024: "P"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_registered_rows() -> pd.DataFrame:
    registry = pd.read_csv(REGISTRY, keep_default_na=False)
    selected = registry.loc[
        registry["contract"].eq("dual_binary_hemodynamic_v1")
        & registry["model_architecture"].eq("dual_branch")
        & registry["alignment_strategy"].eq("outcome_specific_coral")
    ].copy()
    if len(selected) != 5 or set(selected["seed"].astype(int)) != set(SEED_COLORS):
        raise ValueError("Supplementary Figure S1 requires the five registered historical outcome-specific runs")
    rows = []
    for registry_row in selected.itertuples(index=False):
        evaluation_path = ROOT / registry_row.evaluation_file
        if _sha256(evaluation_path) != registry_row.evaluation_sha256:
            raise ValueError(f"Registered evaluation hash mismatch: {registry_row.run_id}")
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        if evaluation.get("run_id") != registry_row.run_id:
            raise ValueError(f"Evaluation run ID mismatch: {registry_row.run_id}")
        if int(evaluation["initialization_seed"]) != int(registry_row.seed):
            raise ValueError(f"Evaluation seed mismatch: {registry_row.run_id}")
        for endpoint in ("idh", "ih"):
            for model_key, model_label in (("source_mlp", "Source MLP"), ("updated_mlp", "Updated MLP")):
                metric = evaluation["metrics"][endpoint][model_key]
                rows.append({
                    "figure": "Supplementary Figure S1",
                    "endpoint": endpoint.upper(),
                    "seed": int(registry_row.seed),
                    "model": model_label,
                    "roc_auc": float(metric["roc_auc"]),
                    "n_sessions": int(metric["n_sessions"]),
                    "n_patients": int(metric["n_patients"]),
                    "run_id": registry_row.run_id,
                    "artifact": registry_row.evaluation_file,
                    "artifact_sha256": registry_row.evaluation_sha256,
                    "evidence_tier": "registered_historical_v1_descriptive_multiseed",
                })
    frame = pd.DataFrame(rows).sort_values(["endpoint", "seed", "model"]).reset_index(drop=True)
    if len(frame) != 20 or not frame["n_sessions"].eq(29866).all() or not frame["n_patients"].eq(172).all():
        raise ValueError("Supplementary Figure S1 held-out cohort contract mismatch")
    return frame


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


def _draw(frame: pd.DataFrame) -> None:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.5), sharey=True)
    for panel, (ax, endpoint) in enumerate(zip(axes, ("IDH", "IH")), start=1):
        subset = frame.loc[frame["endpoint"].eq(endpoint)]
        for seed in sorted(SEED_COLORS):
            seed_rows = subset.loc[subset["seed"].eq(seed)].set_index("model")
            values = [seed_rows.loc["Source MLP", "roc_auc"], seed_rows.loc["Updated MLP", "roc_auc"]]
            ax.plot(
                [0, 1], values, color=SEED_COLORS[seed], marker=SEED_MARKERS[seed],
                linewidth=1.25, markersize=5, label=f"Seed {seed}", alpha=0.95,
            )
        updated = subset.loc[subset["model"].eq("Updated MLP"), "roc_auc"].to_numpy(dtype=float)
        ax.text(
            0.97, 0.04, f"Updated mean {updated.mean():.4f}\nSample SD {updated.std(ddof=1):.4f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": LIGHT},
        )
        ax.set_xticks([0, 1], ["Source MLP", "Updated MLP"])
        ax.set_xlim(-0.20, 1.20)
        ax.set_ylim(0.70, 0.89)
        ax.set_title(f"{endpoint}: held-out ROC AUC")
        ax.grid(axis="y", color=LIGHT, linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(-0.12, 1.04, chr(64 + panel), transform=ax.transAxes, fontweight="bold", fontsize=11)
    axes[0].set_ylabel("ROC AUC")
    axes[1].legend(frameon=False, loc="lower left", bbox_to_anchor=(1.02, 0.02))
    fig.suptitle("Updated-model discrimination was descriptively stable across five historical seeds", fontweight="bold")
    fig.text(
        0.5, 0.012,
        "Lines join source and updated estimates within a registered run. Across-seed variation is descriptive, not an inferential interval or an isolated alignment effect.",
        ha="center", fontsize=7.5, color=SOURCE,
    )
    fig.tight_layout(rect=(0.02, 0.055, 0.91, 0.94))
    fig.savefig(OUTPUT / f"{STEM}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT / f"{STEM}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = _load_registered_rows()
    frame.to_csv(SOURCE_DATA, index=False)
    _draw(frame)
    ALT_TEXT.write_text(
        """# Supplementary Figure S1 Alt Text

Two paired-point panels show source and updated held-out ROC AUC for IDH and IH across initialization seeds 7, 13, 42, 99, and 2024. IDH source AUC varies visibly across seeds, while updated IDH AUC clusters near 0.837. IH source AUC varies less, and updated IH AUC clusters near 0.869. Connected points are descriptive within-run comparisons and do not isolate the effect of alignment.
""",
        encoding="utf-8",
    )
    source_hashes = {
        row.artifact: row.artifact_sha256
        for row in frame[["artifact", "artifact_sha256"]].drop_duplicates().itertuples(index=False)
    }
    outputs = [
        OUTPUT / f"{STEM}.png", OUTPUT / f"{STEM}.pdf", SOURCE_DATA, ALT_TEXT,
    ]
    provenance = {
        "status": "registered_historical_v1_descriptive_supplementary_figure",
        "training_performed": False,
        "figure": "Supplementary Figure S1",
        "registered_run_ids": sorted(frame["run_id"].unique()),
        "seeds": sorted(frame["seed"].unique().tolist()),
        "sources": source_hashes,
        "outputs": {path.name: _sha256(path) for path in outputs},
        "claim_limits": [
            "across_seed_variation_is_descriptive_not_a_confidence_interval",
            "within_run_source_to_updated_lines_do_not_isolate_alignment_effect",
            "historical_v1_only_do_not_mix_with_confirmatory_v2",
        ],
    }
    PROVENANCE.write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(provenance, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
