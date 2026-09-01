import json
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluate.dca_analysis import calculate_net_benefit
from src.visualization.journal_style import get_color_palette, remove_top_right_spines, set_journal_style


ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = ROOT / "tables"
AUDIT_DIR = ROOT / "experiments" / "audit"
RESULT_DIR = ROOT / "experiments" / "results"
FIG_ROOT = ROOT / "figures"
MAIN_DIR = FIG_ROOT / "Submission_Main_Figures"
SUPP_DIR = FIG_ROOT / "Submission_Supplementary_Figures"


def ensure_dirs() -> None:
    MAIN_DIR.mkdir(parents=True, exist_ok=True)
    SUPP_DIR.mkdir(parents=True, exist_ok=True)


def set_export_backend() -> None:
    mpl.rcParams["pdf.fonttype"] = 3
    mpl.rcParams["ps.fonttype"] = 3


def export_figure(fig: plt.Figure, output_stem: Path) -> None:
    fig.savefig(output_stem.with_suffix(".pdf"), dpi=600, facecolor="white")
    fig.savefig(output_stem.with_suffix(".png"), dpi=600, facecolor="white")
    plt.close(fig)


def parse_n_from_header(header: str) -> int:
    match = re.search(r"N=(\d+)", header)
    if not match:
        raise ValueError(f"Cannot parse sample size from header: {header}")
    return int(match.group(1))


def extract_median(value: str) -> float:
    match = re.match(r"\s*([0-9.]+)", str(value))
    if not match:
        raise ValueError(f"Cannot parse median from value: {value}")
    return float(match.group(1))


def load_stage_distribution(df: pd.DataFrame, center_name: str) -> pd.DataFrame:
    stage_map = {
        0: "<=30 min",
        1: "30-60 min",
        2: "60-120 min",
        3: ">120 min / no early event",
    }
    stage_counts = df["stage_30_60_120"].value_counts(dropna=False).sort_index()
    total = len(df)
    rows = []
    for key, count in stage_counts.items():
        rows.append(
            {
                "Center": center_name,
                "Stage": stage_map.get(key, f"Stage {key}"),
                "Percent": float(count / total * 100.0),
            }
        )
    return pd.DataFrame(rows)


def plot_calibration_panel(ax, y_true, y_prob, title, color):
    bins = np.linspace(0, 1, 11)
    y_prob = np.clip(y_prob, 1e-6, 1 - 1e-6)
    bin_idx = np.digitize(y_prob, bins) - 1
    pred_means, obs_rates = [], []
    for idx in range(10):
        mask = bin_idx == idx
        if np.sum(mask) == 0:
            continue
        pred_means.append(np.mean(y_prob[mask]))
        obs_rates.append(np.mean(y_true[mask]))

    ax.plot([0, 1], [0, 1], linestyle="--", color="#7F8C8D", linewidth=1.4, label="Ideal")
    ax.plot(pred_means, obs_rates, marker="o", color=color, linewidth=2.2, label="Observed")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed event rate")
    ax.legend(frameon=False, loc="lower right")
    remove_top_right_spines(ax)


def plot_dca_panel(ax, y_true, y_prob, title, color):
    thresholds = np.linspace(0.01, 0.6, 60)
    prevalence = np.mean(y_true)
    nb_all = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))
    nb_model = calculate_net_benefit(y_true, y_prob, thresholds)

    ax.plot(thresholds, nb_all, color="#95A5A6", linewidth=1.7, label="Treat all")
    ax.plot(thresholds, np.zeros_like(thresholds), color="black", linewidth=1.4, label="Treat none")
    ax.plot(thresholds, nb_model, color=color, linewidth=2.2, label="Locally updated model")
    ax.set_xlim(0.0, 0.6)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.legend(frameon=False, loc="upper right")
    remove_top_right_spines(ax)


def build_figure1_cohort_split() -> None:
    manifest = json.loads((AUDIT_DIR / "data_manifest.json").read_text(encoding="utf-8"))
    split = json.loads((AUDIT_DIR / "split_audit.json").read_text(encoding="utf-8"))
    colors = get_color_palette(4, "clinical")

    fig, ax = plt.subplots(figsize=(11.5, 6.0))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.axis("off")

    def draw_box(x, y, w, h, title, subtitle, color):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor="#F7FAFC", edgecolor="#334E68", linewidth=1.5))
        ax.add_patch(plt.Rectangle((x, y + h - 0.28), w, 0.28, facecolor=color, edgecolor=color))
        ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", fontsize=11, fontweight="bold")
        ax.text(x + w / 2, y + h * 0.28, subtitle, ha="center", va="center", fontsize=9, color="#34495E")

    draw_box(
        0.8,
        4.2,
        4.0,
        1.8,
        "Shenyi Development Cohort",
        f"{manifest['source']['n_rows']:,} sessions\n{manifest['source']['n_patients']:,} patients",
        colors[0],
    )
    draw_box(
        5.1,
        4.2,
        4.0,
        1.8,
        "Fuding External Cohort",
        f"{manifest['target']['n_rows']:,} sessions\n{manifest['target']['n_patients']:,} patients",
        colors[1],
    )
    draw_box(
        2.0,
        1.2,
        2.6,
        1.6,
        "Target Train",
        f"{split['train']['n_sessions']:,} sessions\n{split['train']['n_patients']} patients",
        colors[2],
    )
    draw_box(
        5.2,
        1.2,
        2.6,
        1.6,
        "Target Validation",
        f"{split['val']['n_sessions']:,} sessions\n{split['val']['n_patients']} patients",
        colors[3],
    )
    draw_box(
        8.4,
        1.2,
        2.8,
        1.6,
        "Held-out Test",
        f"{split['test']['n_sessions']:,} sessions\n{split['test']['n_patients']} patients",
        "#5D6D7E",
    )
    arrow = dict(arrowstyle="->", lw=1.8, color="#5D6D7E")
    ax.annotate("", xy=(7.1, 4.2), xytext=(7.1, 2.9), arrowprops=arrow)
    ax.annotate("", xy=(8.8, 4.2), xytext=(9.6, 2.9), arrowprops=arrow)
    ax.text(
        6.0,
        0.45,
        "Patient-level split audit: train/val/test overlap = 0/0/0",
        ha="center",
        fontsize=10,
        fontweight="bold",
        color="#2C3E50",
    )
    ax.set_title("Study Cohort and Patient-level Split Audit", loc="left", fontsize=14, fontweight="bold")
    fig.subplots_adjust(left=0.03, right=0.98, top=0.92, bottom=0.08)
    export_figure(fig, MAIN_DIR / "Fig1_Cohort_Split_Audit")


def build_figure2_cross_center_shift() -> None:
    table1 = pd.read_csv(TABLE_DIR / "table1_baseline.csv")
    source = pd.read_csv(ROOT / "data" / "processed" / "深医_final_data.csv")
    target = pd.read_csv(ROOT / "data" / "processed" / "福鼎_final_data.csv")
    shenyi_header = [c for c in table1.columns if c.startswith("Shenyi (Source)")][0]
    fuding_header = [c for c in table1.columns if c.startswith("Fuding (Target)")][0]

    variables = [
        "透前收缩压",
        "透前舒张压",
        "脉压差",
        "干体重",
        "透析液温度_mean",
        "超滤率_体重归一化",
        "实际透析时长",
    ]
    df = table1[table1["Variable"].isin(variables)].copy()
    df["Shenyi"] = df[shenyi_header].map(extract_median)
    df["Fuding"] = df[fuding_header].map(extract_median)
    df = df.sort_values("Fuding")
    label_map = {
        "透前收缩压": "Pre-dialysis SBP",
        "透前舒张压": "Pre-dialysis DBP",
        "脉压差": "Pulse Pressure",
        "干体重": "Dry Weight",
        "透析液温度_mean": "Dialysate Temp",
        "超滤率_体重归一化": "UFR (weight-normalized)",
        "实际透析时长": "Session Duration",
    }

    stage_df = pd.concat(
        [
            load_stage_distribution(source, "Shenyi"),
            load_stage_distribution(target, "Fuding"),
        ],
        ignore_index=True,
    )
    stage_order = ["<=30 min", "30-60 min", "60-120 min", ">120 min / no early event"]
    pivot = stage_df.pivot(index="Center", columns="Stage", values="Percent").reindex(columns=stage_order)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.6))
    colors = get_color_palette(3, "clinical")
    y = np.arange(len(df))

    for idx, row in enumerate(df.itertuples(index=False)):
        axes[0].plot([row.Shenyi, row.Fuding], [idx, idx], color="#BDC3C7", linewidth=1.8, zorder=1)
    axes[0].scatter(df["Shenyi"], y, color=colors[0], s=55, label="Shenyi", zorder=2)
    axes[0].scatter(df["Fuding"], y, color=colors[1], s=55, label="Fuding", zorder=3)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([label_map.get(v, v) for v in df["Variable"]])
    axes[0].set_xlabel("Median value (original unit)")
    axes[0].set_title("a  Baseline shift", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, loc="lower right")
    axes[0].grid(axis="x", linestyle="--", alpha=0.25)
    remove_top_right_spines(axes[0])

    stage_colors = ["#D73027", "#FC8D59", "#FEE090", "#91BFDB"]
    left = np.zeros(len(pivot))
    for color, stage in zip(stage_colors, pivot.columns):
        values = pivot[stage].values
        axes[1].barh(pivot.index, values, left=left, color=color, edgecolor="white", height=0.58, label=stage)
        left += values
    axes[1].set_xlim(0, 100)
    axes[1].set_xlabel("Sessions (%)")
    axes[1].set_title("b  Event-time shift", loc="left", fontweight="bold")
    axes[1].legend(frameon=False, loc="lower right", title="Timing stage")
    remove_top_right_spines(axes[1])

    fig.suptitle("Cross-center Baseline and Event-time Shift", x=0.02, ha="left", fontsize=14, fontweight="bold")
    fig.subplots_adjust(left=0.10, right=0.98, top=0.86, bottom=0.14, wspace=0.24)
    export_figure(fig, MAIN_DIR / "Fig2_Cross_Center_Shift")


def build_figure3_performance() -> None:
    table2 = pd.read_csv(TABLE_DIR / "table2_performance.csv")

    def parse_cindex(cell: str):
        match = re.match(r"\s*([0-9.]+)\s*\(([0-9.]+)-([0-9.]+)\)", str(cell))
        if not match:
            raise ValueError(f"Cannot parse C-index cell: {cell}")
        return float(match.group(1)), float(match.group(2)), float(match.group(3))

    parsed = table2["C-index (95% CI)"].map(parse_cindex)
    table2["cindex"] = [x[0] for x in parsed]
    table2["lower"] = [x[1] for x in parsed]
    table2["upper"] = [x[2] for x in parsed]
    table2 = table2.sort_values("cindex")

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    colors = get_color_palette(2, "clinical")
    y = np.arange(len(table2))
    ax.errorbar(
        table2["cindex"],
        y,
        xerr=[table2["cindex"] - table2["lower"], table2["upper"] - table2["cindex"]],
        fmt="o",
        color=colors[0],
        ecolor="#7F8C8D",
        elinewidth=2,
        capsize=4,
        markersize=8,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(table2["Model"])
    ax.set_xlabel("C-index (95% CI)")
    ax.set_title("External Validation Performance Comparison", loc="left", fontsize=14, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    remove_top_right_spines(ax)
    fig.subplots_adjust(left=0.24, right=0.98, top=0.90, bottom=0.16)
    export_figure(fig, MAIN_DIR / "Fig3_Performance_Comparison")


def build_figure4_calibration_dca() -> None:
    pred_df = pd.read_csv(RESULT_DIR / "real_test_predictions.csv")
    colors = get_color_palette(3, "clinical")
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 9.0))

    plot_calibration_panel(
        axes[0, 0],
        pred_df["event_by_60m"].values,
        pred_df["event_prob_60m"].values,
        "a  Calibration at 60 min",
        colors[0],
    )
    plot_calibration_panel(
        axes[0, 1],
        pred_df["event_by_120m"].values,
        pred_df["event_prob_120m"].values,
        "b  Calibration at 120 min",
        colors[1],
    )
    plot_dca_panel(
        axes[1, 0],
        pred_df["event_by_60m"].values,
        pred_df["event_prob_60m"].values,
        "c  Decision curve at 60 min",
        colors[0],
    )
    plot_dca_panel(
        axes[1, 1],
        pred_df["event_by_120m"].values,
        pred_df["event_prob_120m"].values,
        "d  Decision curve at 120 min",
        colors[1],
    )
    fig.suptitle("Calibration and Decision-curve Evidence", x=0.02, ha="left", fontsize=14, fontweight="bold")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.10, wspace=0.24, hspace=0.30)
    export_figure(fig, MAIN_DIR / "Fig4_Calibration_DCA")


def build_supplementary_target_burden() -> None:
    df = pd.read_csv(RESULT_DIR / "real_test_predictions.csv")
    if "透析年龄" in df.columns:
        df["Age Group"] = np.where(df["透析年龄"] < 65, "<65", ">=65")
    else:
        df["Age Group"] = "Unknown"
    if "性别" in df.columns:
        df["Gender"] = df["性别"].map({0: "Female", 1: "Male"}).fillna("Other")
    else:
        df["Gender"] = "Unknown"
    if "高血压诊断" in df.columns:
        df["HTN"] = df["高血压诊断"].map({0: "No HTN", 1: "HTN"}).fillna("Unknown")
    else:
        df["HTN"] = "Unknown"

    subgroup_rows = []
    for category, column in [("Age", "Age Group"), ("Gender", "Gender"), ("Hypertension", "HTN")]:
        tmp = df.groupby(column)["events"].agg(["count", "mean"]).reset_index()
        tmp.columns = ["Subgroup", "N", "EventRate"]
        tmp["Category"] = category
        subgroup_rows.append(tmp)
    plot_df = pd.concat(subgroup_rows, ignore_index=True)
    plot_df["EventRatePct"] = plot_df["EventRate"] * 100.0
    order = ["<65", ">=65", "Female", "Male", "No HTN", "HTN", "Unknown", "Other"]
    plot_df["Subgroup"] = pd.Categorical(plot_df["Subgroup"], categories=order, ordered=True)
    plot_df = plot_df.sort_values(["Category", "Subgroup"])

    fig, ax = plt.subplots(figsize=(8.8, 6.0))
    bar_colors = get_color_palette(3, "clinical")
    for idx, category in enumerate(["Age", "Gender", "Hypertension"]):
        sub = plot_df[plot_df["Category"] == category]
        ax.barh(
            sub["Subgroup"].astype(str),
            sub["EventRatePct"],
            color=bar_colors[idx],
            alpha=0.85,
            label=category,
        )
        for _, row in sub.iterrows():
            ax.text(row["EventRatePct"] + 0.4, row["Subgroup"], f"{row['EventRatePct']:.1f}% (n={int(row['N'])})", va="center", fontsize=8)
    ax.set_xlabel("Observed IDH event rate (%)")
    ax.set_ylabel("")
    ax.set_title("Target Test-set IDH Burden Across Clinical Subgroups", loc="left", fontsize=14, fontweight="bold")
    ax.legend(frameon=False, loc="lower right", title="Subgroup family")
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    remove_top_right_spines(ax)
    fig.subplots_adjust(left=0.22, right=0.98, top=0.90, bottom=0.12)
    export_figure(fig, SUPP_DIR / "FigS1_Target_Subgroup_Burden")


def write_strategy_manifest() -> None:
    content = """# Submission Figure Strategy

## Main Figures
- `Fig1_Cohort_Split_Audit`: 研究样本量与福鼎 patient-level split 审计合并图。
- `Fig2_Cross_Center_Shift`: 基线偏移与事件时间结构偏移合并图。
- `Fig3_Performance_Comparison`: held-out test 上的主性能比较图，仅引用当前 locked run。
- `Fig4_Calibration_DCA`: 60/120 分钟校准与 DCA 合并图，仅引用当前 held-out predictions。

## Supplementary Figures
- `FigS1_Target_Subgroup_Burden`: held-out test 亚组事件负担图；只作补充，不宣称亚组优势。

## Net-room Rules
- 禁止恢复 SHAP、mock、pseudo fairness 图，除非完全基于当前 held-out predictions 重建。
- 不再沿用旧的 8 主图结构；投稿版以 4 个主图 + 1 个补图为上限。
- 所有 PDF 导出均强制 `pdf.fonttype = 3` 以满足印刷兼容性要求。
"""
    (FIG_ROOT / "README_FIGURES_STRATEGY.md").write_text(content, encoding="utf-8")


def main() -> None:
    set_journal_style("nature")
    set_export_backend()
    ensure_dirs()
    build_figure1_cohort_split()
    build_figure2_cross_center_shift()
    build_figure3_performance()
    build_figure4_calibration_dca()
    build_supplementary_target_burden()
    write_strategy_manifest()
    print("Submission figures generated under figures/Submission_*.")


if __name__ == "__main__":
    main()
