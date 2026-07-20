"""Build Supplementary Figure S2 only from registered patient-cluster subgroup results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "subgroup_analysis_patient_bootstrap.csv"
INPUT_PROVENANCE = ROOT / "subgroup_analysis_provenance.json"
OUTPUT = ROOT / "figures" / "submission_staging"
STEM = "SupplementaryFigureS2_Patient_Cluster_Subgroups"
SOURCE_DATA = OUTPUT / "SupplementaryFigureS2_source_data.csv"
PROVENANCE = OUTPUT / "SupplementaryFigureS2_provenance.json"
ALT_TEXT = OUTPUT / "SupplementaryFigureS2_ALT_TEXT.md"

UPDATED = "#0072B2"
DELTA = "#D55E00"
INK = "#20242A"
LIGHT = "#E5E7EB"
SOURCE = "#6B7280"
SUBGROUP_ORDER = {
    "IDH": ["baseline_blood_pressure", "prior_instability_history", "ultrafiltration_intensity"],
    "IH": ["treatment_intensity", "volume_overload", "hypertension_history"],
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load() -> tuple[pd.DataFrame, dict]:
    if not INPUT.is_file() or not INPUT_PROVENANCE.is_file():
        raise SystemExit("Supplementary Figure S2 not generated: subgroup results and provenance are required.")
    frame = pd.read_csv(INPUT, keep_default_na=False)
    provenance = json.loads(INPUT_PROVENANCE.read_text(encoding="utf-8"))
    required = {
        "run_id", "endpoint", "subgroup", "subgroup_level", "n_patients", "n_sessions",
        "subgroup_auc", "subgroup_auc_lower", "subgroup_auc_upper", "delta_auc",
        "delta_auc_lower", "delta_auc_upper", "interaction_p_value", "bootstrap_unit",
        "bootstrap_replicates", "bootstrap_valid_replicates", "status",
    }
    if not required.issubset(frame.columns):
        raise SystemExit("Supplementary Figure S2 not generated: subgroup result schema is incomplete.")
    complete = (
        len(frame) == 12
        and set(frame["endpoint"]) == {"IDH", "IH"}
        and frame["subgroup"].nunique() == 6
        and frame.groupby(["endpoint", "subgroup"])["subgroup_level"].nunique().eq(2).all()
        and frame["status"].eq("registered_confirmatory_v2_patient_cluster_subgroup").all()
        and frame["bootstrap_unit"].eq("patient_id").all()
        and pd.to_numeric(frame["bootstrap_replicates"], errors="coerce").eq(1000).all()
        and pd.to_numeric(frame["bootstrap_valid_replicates"], errors="coerce").eq(1000).all()
    )
    numeric = [
        "subgroup_auc", "subgroup_auc_lower", "subgroup_auc_upper",
        "delta_auc", "delta_auc_lower", "delta_auc_upper", "interaction_p_value",
    ]
    if not complete or any(pd.to_numeric(frame[column], errors="coerce").isna().any() for column in numeric):
        raise SystemExit("Supplementary Figure S2 not generated: 12 registered patient-cluster rows are required.")
    if provenance.get("training_performed") is not False or provenance.get("status") != "registered_confirmatory_v2_patient_cluster_subgroup":
        raise SystemExit("Supplementary Figure S2 not generated: subgroup provenance status is invalid.")
    output_record = provenance.get("output", {})
    if output_record.get("path") != str(INPUT.relative_to(ROOT)) or output_record.get("sha256") != _sha256(INPUT):
        raise SystemExit("Supplementary Figure S2 not generated: subgroup CSV does not match provenance.")
    registry = pd.read_csv(ROOT / "experiments" / "evidence_registry" / "run_registry.csv", keep_default_na=False)
    match = registry.loc[
        registry["run_id"].eq(provenance.get("run_id", ""))
        & registry["contract"].eq("dual_binary_hemodynamic_confirmatory_v2")
        & registry["alignment_strategy"].eq("outcome_specific_coral")
    ]
    if len(match) != 1 or set(frame["run_id"]) != {provenance["run_id"]}:
        raise SystemExit("Supplementary Figure S2 not generated: run_id is not one registered outcome-specific v2 run.")
    for endpoint, expected in SUBGROUP_ORDER.items():
        if set(frame.loc[frame["endpoint"].eq(endpoint), "subgroup"]) != set(expected):
            raise SystemExit(f"Supplementary Figure S2 not generated: {endpoint} subgroup set mismatch.")
    return frame, provenance


def _style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })


def _ordered(frame: pd.DataFrame, endpoint: str) -> pd.DataFrame:
    parts = []
    for subgroup in SUBGROUP_ORDER[endpoint]:
        subset = frame.loc[
            frame["endpoint"].eq(endpoint) & frame["subgroup"].eq(subgroup)
        ].copy()
        subset["__level_order"] = np.arange(len(subset))
        parts.append(subset)
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", help="Explicitly build after subgroup evidence passes.")
    args = parser.parse_args()
    if not args.build:
        raise SystemExit("Supplementary Figure S2 not generated. Add --build after subgroup evidence review.")
    frame, input_provenance = _load()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(SOURCE_DATA, index=False)
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.0), gridspec_kw={"width_ratios": [1, 1.08]})
    max_delta = max(abs(pd.to_numeric(frame["delta_auc_lower"])).max(), abs(pd.to_numeric(frame["delta_auc_upper"])).max(), 0.02)
    delta_limit = float(np.ceil(max_delta * 100) / 100 + 0.01)
    panel = 0
    for row_index, endpoint in enumerate(("IDH", "IH")):
        subset = _ordered(frame, endpoint)
        labels = [f"{row.subgroup.replace('_', ' ')} — {row.subgroup_level} (n={int(row.n_patients)})" for row in subset.itertuples()]
        y = np.arange(len(subset))[::-1]
        for column_index, (metric, lower, upper, color, title, limits, reference) in enumerate((
            ("subgroup_auc", "subgroup_auc_lower", "subgroup_auc_upper", UPDATED, "Updated-model ROC AUC", (0.5, 1.0), None),
            ("delta_auc", "delta_auc_lower", "delta_auc_upper", DELTA, "Updated minus source ROC AUC", (-delta_limit, delta_limit), 0.0),
        )):
            ax = axes[row_index, column_index]
            values = pd.to_numeric(subset[metric]).to_numpy()
            lowers = pd.to_numeric(subset[lower]).to_numpy()
            uppers = pd.to_numeric(subset[upper]).to_numpy()
            ax.errorbar(values, y, xerr=[values - lowers, uppers - values], fmt="o", color=color, capsize=2, linewidth=1.2)
            if reference is not None:
                ax.axvline(reference, color=INK, linestyle=":", linewidth=1)
            ax.set_xlim(*limits)
            ax.set_yticks(y, labels if column_index == 0 else [])
            ax.set_title(f"{endpoint}: {title}")
            ax.set_xlabel("ROC AUC" if column_index == 0 else "AUC difference (95% patient-cluster CI)")
            ax.grid(axis="x", color=LIGHT, linewidth=0.7)
            ax.spines[["top", "right"]].set_visible(False)
            panel += 1
            ax.text(-0.15 if column_index == 0 else -0.08, 1.04, chr(64 + panel), transform=ax.transAxes, fontweight="bold", fontsize=11)
        for subgroup in SUBGROUP_ORDER[endpoint]:
            subgroup_rows = subset.loc[subset["subgroup"].eq(subgroup)]
            y_position = y[subgroup_rows.index.max()]
            p_value = float(subgroup_rows["interaction_p_value"].iloc[0])
            axes[row_index, 1].text(delta_limit * 0.96, y_position, f"interaction p={p_value:.3g}", ha="right", va="center", fontsize=7)
    fig.suptitle("Prespecified subgroup performance with patient-cluster uncertainty", fontweight="bold")
    fig.text(0.5, 0.008, "Subgroup analyses are tied to one registered outcome-specific confirmatory-v2 run. Interaction P values compare the two locked levels; no multiplicity correction is applied.", ha="center", fontsize=7.5, color=SOURCE)
    fig.tight_layout(rect=(0.03, 0.035, 0.99, 0.95))
    fig.savefig(OUTPUT / f"{STEM}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT / f"{STEM}.pdf", bbox_inches="tight")
    plt.close(fig)
    ALT_TEXT.write_text(
        "# Supplementary Figure S2 Alt Text\n\nForest plots show updated-model subgroup ROC AUC and updated-minus-source AUC differences with 95% patient-cluster bootstrap intervals for six prespecified subgroup domains and their two locked levels. Interaction P values compare the two level-specific AUC differences.\n",
        encoding="utf-8",
    )
    outputs = [OUTPUT / f"{STEM}.png", OUTPUT / f"{STEM}.pdf", SOURCE_DATA, ALT_TEXT]
    provenance = {
        "status": "registered_confirmatory_v2_patient_cluster_subgroup_figure",
        "training_performed": False,
        "run_id": input_provenance["run_id"],
        "sources": {
            str(INPUT.relative_to(ROOT)): _sha256(INPUT),
            str(INPUT_PROVENANCE.relative_to(ROOT)): _sha256(INPUT_PROVENANCE),
        },
        "outputs": {path.name: _sha256(path) for path in outputs},
        "claim_limits": ["prespecified_subgroups_only", "patient_cluster_uncertainty", "no_multiplicity_correction"],
    }
    PROVENANCE.write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(provenance, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
