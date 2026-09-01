# Project Governance & Technical Reference

**Study Title**: Label-Efficient Target-Center Updating for Dynamic Prediction of Intradialytic Hemodynamic Instability: A Two-Center Discrete-Time Survival Study  
**Current Active Version**: v7 — 跨域机制（诊断 what affects + 干预 what helps），判别+校准双终点，预注册主族 Holm-4（含 CORAL）  
**Protocol**: [`RESEARCH_PLAN_V7_20260901.md`](RESEARCH_PLAN_V7_20260901.md)  
**Last Updated**: 2026-09-01  

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
   - Preprocessing is fitted strictly on `source_train` and locked in [`data/v6_20260830/preprocessing.json`](data/v6_20260830/preprocessing.json).

3. **Multi-Center Cohort & Patient-Level Splitting**:
   - **Source Center (Shenyi)**: 211,452 sessions, 1,628 patients ($85\%$ train, $15\%$ validation).
   - **Target Center (Fuding)**: 74,947 sessions, 430 patients ($70\%$ update pool, $10\%$ calibration, $20\%$ held-out test).
   - **Independent statistical unit**: Individual patient (strict no-leakage patient-level partitioning).

4. **Double Primary Endpoint (Discrimination + Calibration)**:
   - Discrimination: 4-point arithmetic mean of interval IPCW-AUC (`meanAUC`); higher is better.
   - Calibration: `ECE@240` after Platt fitted on independent target_calibration; lower is better.
   - Both reported with 1,000-replicate patient-cluster bootstrap 95% CIs; censoring weights via Reverse Kaplan-Meier.

5. **Two-Half Design (Diagnose then Intervene)**:
   - **Half 1 (Diagnose, what affects)**: quantify the transfer gap (`source_only` source_test vs target_test on discrimination & calibration), then decompose covariate-shift (C2ST/MMD) / label-shift (event-rate ratio + IPW reweight) / concept-shift (residual calibration gap). Pre-specified, descriptive, no significance claims.
   - **Half 2 (Intervene, what helps)**: pre-registered Holm-4 primary family at $b=10\%$ on IDH:
     - H1: `supervised_update − source_only` ΔmeanAUC > 0 (labels→discrimination)
     - H2: `supervised_update − source_only` ΔECE < 0 (labels→calibration)
     - H3: `supervised_update+CORAL − supervised_update` ΔmeanAUC > 0 (alignment→discrimination)
     - H4: `supervised_update+CORAL − supervised_update` ΔECE < 0 (alignment→calibration)
   - IH and $b \in \{25\%,50\%,100\%\}$ are descriptive only; `target_only` is a diagnosis/learning-curve baseline (low-budget collapse must be fixed before use).

## 2. Workspace Organization

- **`conf/`**: Configuration files (`config.yaml`).
- **`data/`**:
  - `raw/`: Immutable raw datasets (gitignored, local only).
  - `v6_20260830/`: Frozen v6 cohort assets (19-feature cohort CSVs, splits, budget subsets, preprocessing + SHA256 lock). Large CSVs gitignored; audit manifest JSON kept in repo.
- **`scripts/v6/`**: Execution scripts (train / infer / diagnose / stage-data / build tables & figures).
- **`src/v6_acc/`**: Single patient-balanced discrete-survival NLL, models, loader — reused extend beyond `supervised_update` to CORAL/MMD/DANN.
- **`tests/`**: Validation tests.
- No `experiments/`, `figures/`, `tables/`, `manuscript/`, or `archive/` in the clean v7 baseline (recreated during run).

---

## 3. Standard Execution Commands

```bash
# 1. 全网格训练（杠杆 family × 2 endpoint × 4 budget × 5 seed，幂等）:
DEVICE=cuda bash scripts/v6/run_v6_train_grid.sh

# 2. 单次训练（训练 + 评估 + session 级预测，供 patient-cluster bootstrap 重估）:
python3 scripts/v6/run_v6_train.py --family supervised_update --endpoint idh --budget 0.10 --seed 401

# 3. 数据研制（一次性，存在即拒绝覆盖，含 SHA256 清单）:
python3 scripts/v6/run_v6_stage_data.py

# 4. 半1 漂移诊断（C2ST/MMD + 标签漂移 IPW + 概念残留）:
python3 scripts/v6/run_v6_diagnose.py

# 5. 半2 推断（patient-cluster bootstrap + Holm-4 主族判定）:
python3 scripts/v6/run_v6_infer.py
```
