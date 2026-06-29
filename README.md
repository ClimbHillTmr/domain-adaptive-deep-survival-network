# Transport-Aware Deep Survival Network (CDAN-GSN)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black/ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

This repository contains the official PyTorch implementation of our paper on multi-center hemodialysis complication (Intradialytic Hypotension, IDH) prediction.

We propose **CDAN-GSN**, a transport-aware survival framework for cross-center intradialytic hypotension timing prediction. The current manuscript-ready workflow should be interpreted as target-center labeled updating with patient-level held-out testing, not as strictly unsupervised domain adaptation.

## 🌟 Key Innovations

1. **Domain-Stratified Cox & IPCW**: Models target-center updating under baseline hazard and censoring differences using Inverse Probability of Censoring Weighting.
2. **KAN Tokenizer & CLS Pooling**: Employs Kolmogorov-Arnold Networks (KAN) to capture non-linear physiological risks (e.g., U-shaped blood pressure curves) and isolates acute deterioration signals using Attention Pooling.
3. **Gated Sparsity Mask**: Prevents negative transfer by adaptively masking out hospital-specific idiosyncratic features.
4. **Treatment-Conditioned Domain Adversarial Network (CDAN)**: Aligns latent representations conditional on treatment-context features and model-derived risk terms.

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
├── tables/               # Auto-generated LaTeX tables for Overleaf
├── run_all.py            # One-click execution pipeline
└── pyproject.toml        # Ruff, Mypy, and Pytest configurations
```

## 🚀 Quick Start

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
Place your source and target center datasets in `data/raw/`. The pipeline expects specific columns (e.g., pre-dialysis vitals, ultrafiltration rate).
*Note: Due to patient privacy, the raw clinical data is not included in this repository.*

### 3. Run the Full Pipeline
The entire process (Data formatting -> source pretraining -> target-center updating -> evaluation -> plotting) can be executed with a single command after the processed data files are available:
```bash
python run_all.py
```

## 📊 Academic Deliverables

Running the pipeline automatically generates the following publication-ready artifacts:

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

Before using generated outputs in a submission, follow `docs/投稿前整改计划.md`. In particular, regenerate final CSV files with prior-only historical features, keep the default patient-level target split, and replace placeholder or simulated statistics with results from real held-out predictions.
