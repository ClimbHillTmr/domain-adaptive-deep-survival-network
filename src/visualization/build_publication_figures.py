import os
import re
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lifelines import KaplanMeierFitter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.visualization.journal_style import get_color_palette, set_journal_style, remove_top_right_spines
from src.visualization.generate_architecture import generate_architecture_diagram
from src.visualization.generate_phenotype_transition import generate_phenotype_transition


ROOT = Path("/home/cht/Works/domain-adaptive-deep-survival-network")
FIG_ROOT = ROOT / "figures"
MAIN_DIR = FIG_ROOT / "Main_Figures"
SUPP_DIR = FIG_ROOT / "Supplementary_Figures"
TABLE_DIR = ROOT / "tables"
DATA_DIR = ROOT / "data"


def ensure_dirs() -> None:
    MAIN_DIR.mkdir(parents=True, exist_ok=True)
    SUPP_DIR.mkdir(parents=True, exist_ok=True)


def set_export_backend() -> None:
    # Type 3 avoids viewer/font substitution issues in some Linux PDF stacks.
    mpl.rcParams["pdf.fonttype"] = 3
    mpl.rcParams["ps.fonttype"] = 3


def export_figure(fig: plt.Figure, output_stem: Path) -> None:
    fig.savefig(output_stem.with_suffix(".pdf"), dpi=600, bbox_inches=None, facecolor="white")
    fig.savefig(output_stem.with_suffix(".png"), dpi=600, bbox_inches=None, facecolor="white")
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


def build_cohort_flow() -> None:
    table1 = pd.read_csv(TABLE_DIR / "table1_baseline.csv")
    shenyi_header = [c for c in table1.columns if c.startswith("Shenyi (Source)")][0]
    fuding_header = [c for c in table1.columns if c.startswith("Fuding (Target)")][0]
    shenyi_analysis_n = parse_n_from_header(shenyi_header)
    fuding_analysis_n = parse_n_from_header(fuding_header)

    counts = {
        "Shenyi analysis cohort": shenyi_analysis_n,
        "Fuding analysis cohort": fuding_analysis_n,
    }

    event_row = table1.loc[table1["Variable"] == "透中低血压_计算"].iloc[0]

    def extract_rate(cell: str) -> str:
        match = re.search(r"\(([\d.]+%)\)", str(cell))
        return match.group(1) if match else "NA"

    shenyi_event_rate = extract_rate(event_row[shenyi_header])
    fuding_event_rate = extract_rate(event_row[fuding_header])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6)
    ax.axis("off")

    box_fc = "#F7FAFC"
    edge = "#334E68"
    accent = get_color_palette(4, "clinical")

    def draw_box(x, y, w, h, title, subtitle, color):
        rect = plt.Rectangle((x, y), w, h, facecolor=box_fc, edgecolor=edge, linewidth=1.6)
        ax.add_patch(rect)
        ax.add_patch(plt.Rectangle((x, y + h - 0.28), w, 0.28, facecolor=color, edgecolor=color))
        ax.text(x + w / 2, y + h * 0.63, title, ha="center", va="center", fontsize=12, fontweight="bold")
        ax.text(x + w / 2, y + h * 0.30, subtitle, ha="center", va="center", fontsize=10, color="#34495E")

    draw_box(
        1.0,
        2.8,
        4.2,
        1.8,
        "Shenyi (Source) Cohort",
        f"{counts['Shenyi analysis cohort']:,} sessions\nIDH rate: {shenyi_event_rate}",
        accent[0],
    )
    draw_box(
        5.8,
        2.8,
        4.2,
        1.8,
        "Fuding (Target) Cohort",
        f"{counts['Fuding analysis cohort']:,} sessions\nIDH rate: {fuding_event_rate}",
        accent[1],
    )

    arrowprops = dict(arrowstyle="->", lw=2.0, color="#5D6D7E")
    ax.annotate("", xy=(5.8, 3.7), xytext=(5.2, 3.7), arrowprops=arrowprops)
    ax.text(5.5, 4.5, "Domain Adaptation", ha="center", fontsize=11, fontweight="bold", color="#5D6D7E")

    ax.set_title("Study Cohort Flow and Analysis Population", loc="left", fontsize=15, fontweight="bold", pad=8)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.90, bottom=0.08)
    export_figure(fig, MAIN_DIR / "Fig1_Cohort_Flow")


def build_baseline_shift() -> None:
    table1 = pd.read_csv(TABLE_DIR / "table1_baseline.csv")
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

    fig, ax = plt.subplots(figsize=(9, 5.6))
    y = np.arange(len(df))
    colors = get_color_palette(3, "clinical")

    for idx, row in enumerate(df.itertuples(index=False)):
        ax.plot([row.Shenyi, row.Fuding], [idx, idx], color="#BDC3C7", linewidth=2.0, zorder=1)
    ax.scatter(df["Shenyi"], y, color=colors[0], s=70, label="Shenyi (Source)", zorder=2)
    ax.scatter(df["Fuding"], y, color=colors[1], s=70, label="Fuding (Target)", zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels([label_map.get(v, v) for v in df["Variable"]])
    ax.set_xlabel("Median value (original unit)", fontweight="bold")
    ax.set_title("Cross-center Baseline Shift in Key Clinical Variables", loc="left", fontsize=15, fontweight="bold")
    ax.legend(frameon=False, loc="lower right", title="Center")
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    remove_top_right_spines(ax)
    fig.subplots_adjust(left=0.29, right=0.98, top=0.90, bottom=0.14)
    export_figure(fig, MAIN_DIR / "Fig2_Baseline_Shift")


def load_stage_distribution(df: pd.DataFrame, center_name: str) -> pd.DataFrame:
    stage_map = {
        0: "<=30 min",
        1: "30-60 min",
        2: "60-120 min",
        3: ">120 min / no early event",
    }
    stage_col = "stage_30_60_120"
    stage_counts = df[stage_col].value_counts(dropna=False).sort_index()
    rows = []
    total = len(df)
    for key, value in stage_counts.items():
        label = stage_map.get(key, f"Stage {key}")
        rows.append({"Center": center_name, "Stage": label, "Count": int(value), "Percent": value / total * 100})
    return pd.DataFrame(rows)


def build_event_stage_distribution() -> None:
    source = pd.read_csv(DATA_DIR / "processed" / "深医_final_data.csv")
    target = pd.read_csv(DATA_DIR / "processed" / "福鼎_final_data.csv")
    stage_df = pd.concat(
        [
            load_stage_distribution(source, "Shenyi"),
            load_stage_distribution(target, "Fuding"),
        ],
        ignore_index=True,
    )
    stage_order = ["<=30 min", "30-60 min", "60-120 min", ">120 min / no early event"]
    pivot = stage_df.pivot(index="Center", columns="Stage", values="Percent").reindex(columns=stage_order)

    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    colors = ["#D73027", "#FC8D59", "#FEE090", "#91BFDB"]
    left = np.zeros(len(pivot))
    for color, stage in zip(colors, pivot.columns):
        values = pivot[stage].values
        ax.barh(pivot.index, values, left=left, color=color, edgecolor="white", height=0.6, label=stage)
        left += values

    ax.set_xlabel("Sessions (%)", fontweight="bold")
    ax.set_title("Distribution of IDH Timing Stages Across Centers", loc="left", fontsize=15, fontweight="bold")
    ax.legend(frameon=False, ncol=2, loc="lower right", title="Event timing stage")
    ax.set_xlim(0, 100)
    remove_top_right_spines(ax)
    fig.subplots_adjust(left=0.16, right=0.98, top=0.90, bottom=0.16)
    export_figure(fig, MAIN_DIR / "Fig3_Event_Timing_Stages")


def build_km_center_comparison() -> None:
    source = pd.read_csv(DATA_DIR / "processed" / "深医_final_data.csv")
    target = pd.read_csv(DATA_DIR / "processed" / "福鼎_final_data.csv")

    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    colors = get_color_palette(2, "clinical")
    for df, label, color in [
        (source, "Shenyi", colors[0]),
        (target, "Fuding", colors[1]),
    ]:
        kmf = KaplanMeierFitter()
        kmf.fit(df["et_min"], event_observed=df["events"], label=label)
        kmf.plot_survival_function(ax=ax, ci_show=False, linewidth=2.4, color=color)

    ax.set_xlabel("Time from dialysis start (minutes)", fontweight="bold")
    ax.set_ylabel("IDH-free survival probability", fontweight="bold")
    ax.set_title("Center-level IDH-free Survival Curves", loc="left", fontsize=15, fontweight="bold")
    ax.legend(frameon=False, loc="upper right", title="Center")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    remove_top_right_spines(ax)
    fig.subplots_adjust(left=0.14, right=0.98, top=0.90, bottom=0.14)
    export_figure(fig, MAIN_DIR / "Fig4_KM_Center_Comparison")


def build_performance_comparison() -> None:
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

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
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
    ax.set_xlabel("C-index (95% CI)", fontweight="bold")
    ax.set_title("External Validation Performance Comparison", loc="left", fontsize=15, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    remove_top_right_spines(ax)
    fig.subplots_adjust(left=0.24, right=0.98, top=0.90, bottom=0.16)
    export_figure(fig, MAIN_DIR / "Fig5_Performance_Comparison")


def build_target_subgroup_burden() -> None:
    df = pd.read_csv(DATA_DIR / "processed" / "福鼎_final_data.csv")
    subgroup_rows = []

    df["Age Group"] = np.where(df["透析年龄"] < 65, "<65", ">=65")
    df["Gender"] = df["性别"].map({0: "Female", 1: "Male"}).fillna("Other")
    df["HTN"] = df["高血压诊断"].map({0: "No HTN", 1: "HTN"}).fillna("Unknown")

    for category, column in [("Age", "Age Group"), ("Gender", "Gender"), ("Hypertension", "HTN")]:
        tmp = df.groupby(column)["透中低血压_计算"].agg(["count", "mean"]).reset_index()
        tmp.columns = ["Subgroup", "N", "EventRate"]
        tmp["Category"] = category
        subgroup_rows.append(tmp)

    plot_df = pd.concat(subgroup_rows, ignore_index=True)
    plot_df["EventRatePct"] = plot_df["EventRate"] * 100

    order = ["<65", ">=65", "Female", "Male", "No HTN", "HTN", "Unknown", "Other"]
    plot_df["Subgroup"] = pd.Categorical(plot_df["Subgroup"], categories=order, ordered=True)
    plot_df = plot_df.sort_values(["Category", "Subgroup"])

    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    sns.barplot(
        data=plot_df,
        x="EventRatePct",
        y="Subgroup",
        hue="Category",
        palette=get_color_palette(3, "clinical"),
        orient="h",
        ax=ax,
    )
    for patch, (_, row) in zip(ax.patches, plot_df.iterrows()):
        ax.text(
            patch.get_width() + 0.5,
            patch.get_y() + patch.get_height() / 2,
            f"{row['EventRatePct']:.1f}% (n={int(row['N'])})",
            va="center",
            fontsize=8,
        )

    ax.set_xlabel("Observed IDH event rate (%)", fontweight="bold")
    ax.set_ylabel("")
    ax.set_title("Target-center IDH Burden Across Clinical Subgroups", loc="left", fontsize=15, fontweight="bold")
    ax.legend(frameon=False, loc="lower right", title="Subgroup family")
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    remove_top_right_spines(ax)
    fig.subplots_adjust(left=0.18, right=0.98, top=0.90, bottom=0.14)
    export_figure(fig, MAIN_DIR / "Fig6_Target_Subgroup_Burden")


def write_figure_manifest() -> None:
    content = """# Publication Figure Manifest

## Main Figures
- `Fig1_Cohort_Flow`: 研究样本流转与分析队列规模，基于原始 CSV 与 Table 1 冻结样本量。
- `Fig2_Baseline_Shift`: 深医与福鼎关键基线变量的中心间偏移。
- `Fig3_Event_Timing_Stages`: 两中心低血压发生时间阶段分布。
- `Fig4_KM_Center_Comparison`: 两中心 IDH-free survival 曲线比较。
- `Fig5_Performance_Comparison`: 外部验证模型性能比较，基于 `table2_performance.csv`。
- `Fig6_Target_Subgroup_Burden`: 福鼎队列亚组 IDH 事件负担。

## Supplementary Figures
- `FigS1_Architecture`: 方法学架构图，限定为统计学对齐而非因果 DAG。
- `FigS2_Phenotype_Transition`: 定性状态转移概念图，仅作机制讨论辅助。

## Important Notes
- 当前仓库没有可复核的逐例预测文件，因此没有导出真实校准曲线或 DCA 图。
- 当前仓库没有可复核的真实 SHAP 结果缓存，因此没有保留任何基于模拟数据的 SHAP 图。
- 若后续补齐逐例预测与解释结果，应新增到 `figures/Main_Figures`，而不是覆盖现有真实数据图。
"""
    (FIG_ROOT / "README_FIGURES_STRATEGY.md").write_text(content, encoding="utf-8")


def main() -> None:
    set_journal_style("nature")
    set_export_backend()
    ensure_dirs()
    build_cohort_flow()
    build_baseline_shift()
    build_event_stage_distribution()
    build_km_center_comparison()
    build_performance_comparison()
    build_target_subgroup_burden()
    generate_architecture_diagram()
    generate_phenotype_transition()
    write_figure_manifest()
    print("Publication-ready figures generated under figures/.")


if __name__ == "__main__":
    main()
