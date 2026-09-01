# Label-Efficient Target-Center Updating for Dynamic Prediction of Intradialytic Hemodynamic Instability

**A Two-Center Discrete-Time Survival Study**  
*Codebase & Reproducibility Repository for Discrete Survival Analysis, Label Efficiency Benchmarks, and Cross-Center Transportability.*

📖 **中文详细研究报告**: [`RESEARCH_REPORT_ZH.md`](RESEARCH_REPORT_ZH.md)  
📄 **English Manuscript**: [`manuscript/manuscript_definitive_label_efficiency.md`](manuscript/manuscript_definitive_label_efficiency.md)  

---

## 1. Study Overview

This repository contains the complete dataset pipeline, discrete-time survival neural network architectures, target label-efficiency benchmark suite, and statistical inference engine for predicting intradialytic hemodynamic instability:
- **Intradialytic Hypotension (IDH)**: Baseline SBP minus intradialytic $\text{SBP} \ge 30\text{ mmHg}$, or any intradialytic $\text{SBP} \le 90\text{ mmHg}$.
- **Intradialytic Hypertension (IH)**: Intradialytic MAP minus baseline $\text{MAP} > 10\text{ mmHg}$.

### Core Scientific Findings
1. **Primary Positive Finding (Label Efficiency)**: Under severe local data starvation (10% target budget $\approx 30$ patients), training purely on target data collapses (IDH AUC $0.546$, IH AUC $0.614$). Source-center pretraining combined with target-center supervised updating achieves high, stable discrimination (IDH AUC $0.818$, IH AUC $0.855$), providing massive paired gains (+0.2721 [95% CI: 0.2227 to 0.3185, $p < 0.001$] for IDH; +0.2404 [95% CI: 0.1751 to 0.3162, $p < 0.001$] for IH).
2. **Definitive Falsification (No Value from Covariance Alignment)**: Selective and global CORAL alignment provide no statistically significant incremental benefit over plain supervised updating across any budget ($p = 1.000$ Holm-adjusted; empirical $p \ge 0.150$ vs. 19 random feature partitions).

---

## 2. Directory Structure

```
domain-adaptive-deep-survival-network/
├── RESEARCH_REPORT_ZH.md                 # 详细中文研究设计、流程与结论报告
├── conf/                                 # Active configuration files
│   ├── config.yaml                       # Default path and feature config
│   ├── v4_restart_20260730.yaml          # Full budget (100%) survival config
│   └── v4_label_efficiency_20260731.yaml # Multi-budget (10%, 25%, 50%, 100%) config
├── data/
│   ├── raw/                              # Original raw data snapshots
│   ├── v4_restart_20260730/              # Locked survival cohorts, splits, and preprocessing
│   └── v4_label_efficiency_20260731/     # Locked patient budget manifests
├── experiments/
│   ├── v4_restart_20260730/              # 270 full-budget model runs
│   └── v4_label_efficiency_20260731/     # 452 multi-budget runs + bootstrap inference CSVs
├── figures/
│   └── pdf/                              # Publication-grade vector PDF figures (Figures 1-5)
├── tables/                               # Publication-grade CSV data tables (Tables 1-6)
├── manuscript/
│   └── manuscript_definitive_label_efficiency.md # Full TRIPOD+AI compliant manuscript draft
├── docs/                                 # Design and visualization execution plans
├── scripts/                              # Active production and analysis scripts
│   ├── run_v4_bootstrap_inference.py     # 32-core parallel 1,000-replicate bootstrap engine
│   ├── run_discrete_survival_baselines.py # Logistic Hazard & GBDT statistical baselines
│   ├── build_publication_figures.py      # Generates 5 vector PDF figures & data tables
│   ├── build_table1.py                   # Generates Table 1 cohort summary
│   ├── run_v4_label_efficiency.py        # Label efficiency execution launcher
│   └── run_v4_restart.py                 # Full-budget restart runner
├── src/                                  # Core Python package (data, models, train, evaluate)
└── archive/                              # Preserved historical/legacy project artifacts
```

---

## 3. Quickstart & Reproducibility

### Environment Setup
```bash
conda activate ML
pip install -r requirements.txt
```

### Reproduce Statistical Inference (1,000 Patient-Cluster Bootstraps)
```bash
python3 scripts/run_v4_bootstrap_inference.py
```
*Outputs paired contrasts with 95% CIs, Holm-adjusted p-values, and 19-partition empirical tests into `experiments/v4_label_efficiency_20260731/`.*

### Reproduce Statistical Baselines (Logistic Hazard & GBDT)
```bash
python3 scripts/run_discrete_survival_baselines.py
```
*Trains discrete survival Logistic Hazard and HistGradientBoosting baselines across all budgets and seeds.*

### Generate Publication Figures and Tables
```bash
python3 scripts/build_publication_figures.py
python3 scripts/build_table1.py
```
*Outputs 5 vector PDF figures into `figures/pdf/` and 6 summary CSV tables into `tables/`.*

---

## 4. Key Deliverables

- **Figures (Vector PDF)**:
  - `figures/pdf/fig1_study_design.pdf`: Study design, cohort split, and discrete survival architecture.
  - `figures/pdf/fig2_label_efficiency_curves.pdf`: Main label efficiency curves with 95% CIs.
  - `figures/pdf/fig3_paired_contrasts_forest.pdf`: Paired difference forest plot & Holm adjustments.
  - `figures/pdf/fig4_horizon_trajectories.pdf`: Dynamic 60/120/180/240 min IPCW-AUC progression.
  - `figures/pdf/fig5_calibration_analysis.pdf`: Platt-calibrated decile reliability and ECE.
- **Tables (CSV)**:
  - `tables/table1_cohort_summary.csv`: Cohort characteristics and distribution shifts.
  - `tables/table2_label_efficiency_main_results.csv`: Main label efficiency results.
  - `tables/table3_paired_contrasts_inference.csv`: Paired bootstrap differences and p-values.
  - `tables/table4_horizon_specific_dynamics.csv`: Horizon-specific dynamic discrimination.
  - `tables/table5_random_partition_falsification.csv`: 19-partition empirical null test.
  - `tables/table6_statistical_baselines.csv`: Logistic Hazard & GBDT benchmark comparison.
- **Manuscript & Reports**:
  - `RESEARCH_REPORT_ZH.md`: Comprehensive Chinese research report.
  - `manuscript/manuscript_definitive_label_efficiency.md`: Complete journal-ready manuscript.
