# Transport-Aware Deep Survival Network (CDAN-GSN)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black/ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

This repository studies cross-center intradialytic hypotension (IDH) prediction with target-center labeled updating and patient-level held-out testing.

The active research contract is session-level binary IDH risk at dialysis start. First detected IDH time is descriptive only until the two centers' observation grids are harmonized; the legacy Cox/CDAN workflow must not be used with newly built zero-time non-events. See `docs/研究重置与数据契约.md`.

## Legacy Components

The Cox, KAN, gated-mask, and CDAN implementations are retained for historical comparison only. They are not current scientific claims and must not be evaluated until the binary-IDH baselines and rebuilt data contract pass the documented gates.

## 📂 Project Structure

```text
├── conf/                 # Hydra YAML configurations (hyperparams, ablations)
├── data/                 # Data directory (raw & processed CSVs)
├── docs/                 # Documentation and Chinese project report
├── experiments/          # Logs, environment snapshots (pip freeze), and JSON results
├── figures/              # Publication-grade output figures (t-SNE, SHAP, KM curves)
├── src/                  # Highly decoupled source code
│   ├── data_pipeline/    # Data preprocessing & feature engineering
│   ├── data/             # Dataset loading & IPCW calculation
│   ├── models/           # CDAN-GSN architecture definition
│   ├── train/            # Stratified Cox & DA training loop
│   ├── evaluate/         # C-index & Time-dependent AUC metrics
│   └── visualization/    # SHAP and t-SNE plotting scripts
├── tables/               # Auto-generated LaTeX tables (regenerated from experiments/results/ after each run)
├── archive/legacy_outputs/ # Historical mock/placeholder outputs — NOT current results
├── run_all.py            # One-click execution pipeline
└── pyproject.toml        # Ruff, Mypy, and Pytest configurations
```

## Quick Start

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/domain-adaptive-deep-survival-network.git
cd domain-adaptive-deep-survival-network

# Install dependencies
pip install -r requirements.txt
# Or use pipenv/poetry based on pyproject.toml
```

### 2. Prepare Data
Build source and target cohorts explicitly from the raw exports before any model work:
```bash
python src/data_pipeline/HBD_data.py RAW_SHENYI.csv OUTPUT_SHENYI.csv
python src/data_pipeline/HBD_data_fuding.py RAW_FUDING.csv OUTPUT_FUDING.csv
```
Each build writes a companion audit JSON. Do not run the legacy Cox pipeline against the new binary-IDH data contract.

### 3. Audit Before Training
The default command performs a static binary-contract audit only:
```bash
python run_all.py
```

It does not rebuild data or fit a model. After rebuilt cohorts pass the audit and the patient split is reviewed, training still requires an explicit gate:

```bash
python run_all.py --train
```

The active comparison set is source-only logistic regression, local Fuding logistic regression, source MLP zero-shot, and labeled Fuding updating of the same MLP. Configuration is frozen in `conf/binary_config.yaml`; split and initialization seeds are separate.

### One-command server run

Build both cohorts, run strict preflight, fit calibrated multiseed binary baselines, audit time measurement, fit discrete-time models, and run Cox/CDAN sensitivity analyses:

```bash
python run_pipeline.py /path/to/updated_dataset_shenyi.csv /path/to/updated_dataset_fuding.csv
```

The command never overwrites `data/processed`. Progress and failure logs are saved under `experiments/pipeline_runs/pipeline_<UTC time>/`.

## Historical Deliverables

The survival tables and figures below belong to the disabled legacy workflow and are not publication-ready under `binary_idh_v1`:

- **Tables (`tables/`)**:
  - `table1_baseline.tex`: Demographics and baseline characteristics (with IQR and missingness).
  - `table2_performance.tex`: C-index and Time-dependent AUC metrics.
  - `table3_pvalues.tex`: Statistical significance comparisons.
- **Figures (`figures/`)**:
  - `fig2_tsne_alignment.pdf`: Latent space domain alignment visualization.
  - `fig3a_shap_summary.pdf`: Global feature importance (Beeswarm).
  - `fig3b_shap_dependence_top1.pdf`: Non-linear risk inflection points.
  - `fig4_km_risk_stratification.pdf`: Kaplan-Meier survival curves for risk groups.

## 🔬 Ablation Studies

To run ablation experiments (e.g., removing IPCW or reverting KAN to Linear), modify `conf/ablation/template.yaml` and execute the specific config via Hydra integration in `main.py`.

## 📝 Citation

If you find this code or our methodology useful in your research, please consider citing our paper:
*(Citation details will be updated upon publication)*

## Manuscript Readiness Notes

Before using generated outputs in a submission, follow `docs/研究重置与数据契约.md` and `docs/投稿前整改计划.md`.
