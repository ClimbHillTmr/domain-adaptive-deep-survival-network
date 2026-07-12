"""
Phase 4 – Performance Table + Ablation Bar Chart
=================================================
Generates:
  tables/table2_performance.csv/.tex  — full model comparison (CDAN-GSN, CoxPH, Zero-shot)
  tables/table3_ablation.csv/.tex     — ablation study with bootstrap CIs
  figures/Main_Figures/Fig5_Performance_Comparison.pdf/png
  figures/Main_Figures/Fig_Ablation_Study.pdf/png
"""
import json
import os
import re
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.visualization.journal_style import get_color_palette, remove_top_right_spines, set_journal_style

RESULT_DIR = Path("experiments/results")
TABLE_DIR = Path("tables")
FIG_DIR = Path("figures/Main_Figures")


def load_json(path):
    with open(path) as f:
        return json.load(f)


def fmt_cindex(c, lo, hi):
    return f"{c:.3f} ({lo:.3f}–{hi:.3f})"


def fmt_auc(v):
    return f"{v:.3f}" if v else "—"


# -------------------------------------------------------------------------
# Table 2: full model comparison
# -------------------------------------------------------------------------
def build_table2():
    eval_res = load_json(RESULT_DIR / "evaluation_results.json")
    cox_res = load_json(RESULT_DIR / "local_cox_update_results.json")

    rows = []
    # CDAN-GSN and Zero-shot from evaluation_results
    for model_name, sweep in eval_res["sweeps"].items():
        m = sweep["metrics"]
        rows.append({
            "Model": model_name,
            "C-index (95% CI)": fmt_cindex(m["C-index"], m["C-index_CI_lower"], m["C-index_CI_upper"]),
            "AUC (30 min)": fmt_auc(m.get("AUC_30m")),
            "AUC (60 min)": fmt_auc(m.get("AUC_60m")),
            "AUC (120 min)": fmt_auc(m.get("AUC_120m")),
        })

    # CoxPH
    m = cox_res["metrics"]
    rows.append({
        "Model": "CoxPH (local update)",
        "C-index (95% CI)": fmt_cindex(m["C-index"], m["C-index_CI_lower"], m["C-index_CI_upper"]),
        "AUC (30 min)": fmt_auc(m.get("AUC_30m")),
        "AUC (60 min)": fmt_auc(m.get("AUC_60m")),
        "AUC (120 min)": fmt_auc(m.get("AUC_120m")),
    })

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / "table2_performance.csv", index=False)
    latex = df.to_latex(
        index=False,
        escape=True,
        caption="External validation performance on the Fuding (target) test set. "
                "C-index 95\\% CIs computed by patient-cluster bootstrap (200 replicates).",
        label="tab:performance",
    )
    (TABLE_DIR / "table2_performance.tex").write_text(latex)
    print(f"  Table 2 → {TABLE_DIR / 'table2_performance.csv'}")
    return df


# -------------------------------------------------------------------------
# Table 3: ablation study
# -------------------------------------------------------------------------
def build_table3():
    ablation_path = RESULT_DIR / "ablation_results_v2.json"
    if not ablation_path.exists():
        print("  Ablation results not found — skipping Table 3.")
        return None

    data = load_json(ablation_path)
    # Also load full model as reference
    eval_res = load_json(RESULT_DIR / "evaluation_results.json")
    full_m = eval_res["sweeps"]["CDAN-GSN (Ours)"]["metrics"]

    rows = [{
        "Variant": "Full CDAN-GSN",
        "Removed component": "—",
        "n features": eval_res["metadata"]["n_features"],
        "C-index (95% CI)": fmt_cindex(full_m["C-index"], full_m["C-index_CI_lower"], full_m["C-index_CI_upper"]),
        "AUC (60 min)": fmt_auc(full_m.get("AUC_60m")),
        "AUC (120 min)": fmt_auc(full_m.get("AUC_120m")),
    }]

    label_map = {
        "no_clinical": "Pre-dialysis vitals",
        "no_history": "History features",
        "no_kan": "KAN tokenizer",
        "no_cdan": "Domain adaptation",
    }
    for abl in data:
        if "metrics" not in abl:
            continue
        m = abl["metrics"]
        rows.append({
            "Variant": abl["label"],
            "Removed component": label_map.get(abl["name"], abl["name"]),
            "n features": abl.get("n_features", "—"),
            "C-index (95% CI)": fmt_cindex(m["C-index"], m["C-index_CI_lower"], m["C-index_CI_upper"]),
            "AUC (60 min)": fmt_auc(m.get("AUC_60m")),
            "AUC (120 min)": fmt_auc(m.get("AUC_120m")),
        })

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / "table3_ablation.csv", index=False)
    latex = df.to_latex(
        index=False,
        escape=True,
        caption="Ablation study on the Fuding target test set. "
                "Each row removes one model component from the full CDAN-GSN. "
                "C-index 95\\% CIs: patient-cluster bootstrap (200 replicates).",
        label="tab:ablation",
    )
    (TABLE_DIR / "table3_ablation.tex").write_text(latex)
    print(f"  Table 3 → {TABLE_DIR / 'table3_ablation.csv'}")
    return df


# -------------------------------------------------------------------------
# Fig 5: performance comparison forest plot
# -------------------------------------------------------------------------
def build_fig5_performance():
    t2_path = TABLE_DIR / "table2_performance.csv"
    if not t2_path.exists():
        print("  table2_performance.csv missing — skip Fig5.")
        return
    df = pd.read_csv(t2_path)

    def parse_cindex(cell):
        m = re.match(r"\s*([0-9.]+)\s*\(([0-9.]+)[–-]([0-9.]+)\)", str(cell))
        if not m:
            return None, None, None
        return float(m.group(1)), float(m.group(2)), float(m.group(3))

    parsed = [parse_cindex(c) for c in df["C-index (95% CI)"]]
    df["ci_pt"] = [x[0] for x in parsed]
    df["ci_lo"] = [x[1] for x in parsed]
    df["ci_hi"] = [x[2] for x in parsed]
    df = df.dropna(subset=["ci_pt"]).sort_values("ci_pt")

    colors = get_color_palette(3, "clinical")
    highlight = colors[0]
    default_c = "#7F8C8D"

    set_journal_style("nature")
    mpl.rcParams["pdf.fonttype"] = 3
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    y = np.arange(len(df))
    point_colors = [highlight if "CDAN-GSN" in m else default_c for m in df["Model"]]
    for yi, row, pc in zip(y, df.itertuples(), point_colors):
        ax.plot([row.ci_lo, row.ci_hi], [yi, yi], color="#BDC3C7", linewidth=2.0, zorder=1)
        ax.scatter(row.ci_pt, yi, color=pc, s=75, zorder=2)

    ax.set_yticks(y)
    ax.set_yticklabels(df["Model"])
    ax.set_xlabel("C-index (95% CI, patient bootstrap)", fontweight="bold")
    ax.set_title("External Validation: Target-centre Performance", loc="left", fontsize=14, fontweight="bold")
    ax.axvline(x=0.84, color="#E74C3C", linestyle=":", linewidth=1.2, alpha=0.7, label="CoxPH reference")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    remove_top_right_spines(ax)
    fig.subplots_adjust(left=0.32, right=0.97, top=0.90, bottom=0.16)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png"]:
        fig.savefig(FIG_DIR / f"Fig5_Performance_Comparison.{ext}", dpi=600, bbox_inches=None, facecolor="white")
    plt.close(fig)
    print(f"  Fig5 → {FIG_DIR / 'Fig5_Performance_Comparison.pdf'}")


# -------------------------------------------------------------------------
# Ablation figure: horizontal bar + CI dots
# -------------------------------------------------------------------------
def build_fig_ablation():
    ablation_path = RESULT_DIR / "ablation_results_v2.json"
    if not ablation_path.exists():
        print("  Ablation results not found — skip ablation figure.")
        return

    data = load_json(ablation_path)
    eval_res = load_json(RESULT_DIR / "evaluation_results.json")
    full_m = eval_res["sweeps"]["CDAN-GSN (Ours)"]["metrics"]
    full_c = full_m["C-index"]

    label_map = {
        "no_clinical": "w/o Pre-dialysis Vitals",
        "no_history": "w/o History Features",
        "no_kan": "w/o KAN Tokenizer",
        "no_cdan": "w/o Domain Adaptation",
    }

    entries = []
    for abl in data:
        if "metrics" not in abl:
            continue
        m = abl["metrics"]
        entries.append({
            "label": label_map.get(abl["name"], abl["name"]),
            "cindex": m["C-index"],
            "ci_lo": m["C-index_CI_lower"],
            "ci_hi": m["C-index_CI_upper"],
            "delta": m["C-index"] - full_c,
        })
    if not entries:
        print("  No valid ablation entries.")
        return

    abl_df = pd.DataFrame(entries).sort_values("delta")

    set_journal_style("nature")
    mpl.rcParams["pdf.fonttype"] = 3
    colors = get_color_palette(3, "clinical")
    neg_color = "#E74C3C"
    pos_color = "#2ECC71"

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))

    # Left panel: delta C-index
    ax = axes[0]
    bar_colors = [neg_color if d < 0 else pos_color for d in abl_df["delta"]]
    y = np.arange(len(abl_df))
    ax.barh(y, abl_df["delta"], color=bar_colors, height=0.55, edgecolor="white")
    ax.axvline(x=0, color="black", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(abl_df["label"])
    ax.set_xlabel("ΔC-index vs Full Model", fontweight="bold")
    ax.set_title("a  Component Contribution", loc="left", fontsize=13, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    remove_top_right_spines(ax)

    # Right panel: absolute C-index with CI
    ax2 = axes[1]
    all_entries = [{"label": "Full CDAN-GSN", "cindex": full_c,
                    "ci_lo": full_m["C-index_CI_lower"],
                    "ci_hi": full_m["C-index_CI_upper"]}] + entries
    all_df = pd.DataFrame(all_entries)
    y2 = np.arange(len(all_df))
    pt_colors = [colors[0] if "Full" in r else "#7F8C8D" for r in all_df["label"]]
    for yi, row, pc in zip(y2, all_df.itertuples(), pt_colors):
        ax2.plot([row.ci_lo, row.ci_hi], [yi, yi], color="#BDC3C7", linewidth=2.0, zorder=1)
        ax2.scatter(row.cindex, yi, color=pc, s=72, zorder=2)
    ax2.set_yticks(y2)
    ax2.set_yticklabels(all_df["label"])
    ax2.set_xlabel("C-index (95% CI)", fontweight="bold")
    ax2.set_title("b  Absolute Performance", loc="left", fontsize=13, fontweight="bold")
    ax2.grid(axis="x", linestyle="--", alpha=0.25)
    remove_top_right_spines(ax2)

    fig.subplots_adjust(left=0.25, right=0.98, top=0.92, bottom=0.15, wspace=0.45)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png"]:
        fig.savefig(FIG_DIR / f"Fig_Ablation_Study.{ext}", dpi=600, bbox_inches=None, facecolor="white")
    plt.close(fig)
    print(f"  Ablation figure → {FIG_DIR / 'Fig_Ablation_Study.pdf'}")


def main():
    print("[Tables + Figures] Building performance tables and figures...")
    build_table2()
    build_table3()
    build_fig5_performance()
    build_fig_ablation()
    print("Done.")


if __name__ == "__main__":
    main()
