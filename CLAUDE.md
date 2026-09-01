# Project Governance & Technical Reference

**Study Title**: Label-Efficient Target-Center Updating for Dynamic Prediction of Intradialytic Hemodynamic Instability: A Two-Center Discrete-Time Survival Study  
**Current Active Version**: v6 Rebuild — 统一 patient-balanced NLL，三主线(target_only/source_only/supervised_update)，主族 Holm-2  
**Protocol**: [`RESEARCH_PLAN_V6_20260830.md`](RESEARCH_PLAN_V6_20260830.md)  
**Last Updated**: 2026-08-31  

---

## 1. Core Architecture & Scientific Principles

1. **Discrete-Time Survival Formulation**:
   - **Time horizon**: 240 minutes, discretized into four 60-minute intervals: $[0, 60), [60, 120), [120, 180), [180, 240]\text{ min}$.
   - **Target Endpoints**:
     - *Intradialytic Hypotension (IDH)*: Baseline SBP minus intradialytic $\text{SBP} \ge 30\text{ mmHg}$, or any intradialytic $\text{SBP} \le 90\text{ mmHg}$.
     - *Intradialytic Hypertension (IH)*: Intradialytic MAP minus baseline $\text{MAP} > 10\text{ mmHg}$.
   - **Metrics**: IPCW-AUC and IPCW-Brier score evaluated dynamically at each interval ($60, 120, 180, 240\text{ min}$) and as a 4-point arithmetic mean. Censoring weights are estimated via Reverse Kaplan-Meier.

2. **Feature Grouping & Preprocessing (19 locked features)**:
   - **Physiology & History (18 features)**: Age, Sex, Vintage, SBP, DBP, Weight, Historical Mean Weight, Historical Mean SBP, Historical Mean DBP, $\log(1 + \text{Prior Session Count})$, 8 historical interval rates.
   - **Treatment Context (1 feature)**: Pre-dialysis Weight minus Dry Weight (interdialytic fluid overload).
   - Preprocessing is fitted strictly on `source_train` and locked in [`data/v4_restart_20260730/preprocessing.json`](data/v4_restart_20260730/preprocessing.json).

3. **Multi-Center Cohort & Patient-Level Splitting**:
   - **Source Center (Shenyi)**: 211,452 sessions, 1,628 patients ($85\%$ train, $15\%$ validation).
   - **Target Center (Fuding)**: 74,947 sessions, 430 patients ($70\%$ update pool, $10\%$ calibration, $20\%$ held-out test).
   - **Independent statistical unit**: Individual patient (strict no-leakage patient-level partitioning).

4. **Target Label Efficiency Regimes & Statistical Inference**:
   - 4 Nested Target Budgets: $b \in \{10\% \text{ (30 pts)}, 25\% \text{ (75 pts)}, 50\% \text{ (150 pts)}, 100\% \text{ (301 pts)}\}$.
   - 1,000-replicate patient-cluster bootstrap resampling for 95% CIs and paired difference $\Delta\text{AUC}$ inference.
   - Holm-Bonferroni step-down correction for family-wise error rate control.
   - 19-partition empirical null distribution testing for selective alignment validation.

---

## 2. Workspace Organization

- **`conf/`**: Configuration files (`config.yaml`, `v4_restart_20260730.yaml`, `v4_label_efficiency_20260731.yaml`, `v4_survival.yaml`).
- **`data/`**:
  - `raw/`: Immutable raw datasets.
  - `v4_restart_20260730/`: Locked cohort CSVs, splits, and preprocessing.
  - `v4_label_efficiency_20260731/`: Locked budget manifests.
- **`experiments/`**:
  - `v4_restart_20260730/`: 270 full-budget model runs.
  - `v4_label_efficiency_20260731/`: 452 multi-budget model runs and bootstrap inference outputs.
- **`figures/pdf/`**: Publication-ready vector PDF figures (`fig1_study_design.pdf` to `fig5_calibration_analysis.pdf`).
- **`tables/`**: CSV tables (`table1_cohort_summary.csv` to `table6_statistical_baselines.csv`).
- **`manuscript/`**: Journal-ready manuscript (`manuscript_definitive_label_efficiency.md`).
- **`scripts/`**: Production-grade execution scripts.
- **`archive/`**: Preserved historical/legacy artifacts.

---

## 3. Standard Execution Commands

```bash
# 1. 全网格训练（3 family × 2 endpoint × 4 budget × 5 seed = 120 runs，幂等，仅当存在 evaluation.json 与 test_predictions.csv 才跳过）:
DEVICE=cuda bash scripts/v6/run_v6_train_grid.sh

# 2. 单次训练（训练 + 评估 + session 级预测，供 patient-cluster bootstrap 重估）:
python3 scripts/v6/run_v6_train.py --family supervised_update --endpoint idh --budget 0.10 --seed 401

# 3. 数据研制（一次性，存在即拒绝覆盖，含 SHA256 清单）:
python3 scripts/v6/run_v6_stage_data.py

# 注：v4/amendment 旧方案代码与结论已归档至 archive/v6_legacy_20260830/legacy_v4_amendment_v1/
```
