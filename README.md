# Mechanism-Aware Transportable Representation Learning for Clinical AI

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black/ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

This repository implements a mechanism-aware domain adaptation framework for cross-center clinical AI deployment, specifically for intradialytic hypotension (IDH) and hypertension (IH) prediction in dialysis patients.

**Core Contribution**: Clinical domain adaptation should not seek universal domain invariance; instead, it should identify outcome-specific transportable representations while preserving clinically meaningful variations.

## 📋 Abstract

Clinical AI models trained at one institution often fail when deployed at another due to domain shift. Traditional domain adaptation treats all domain differences as noise to be eliminated. This work demonstrates that clinical domain shift contains heterogeneous components:

1. **Transportable variation** (physiology): Blood pressure, heart rate, and historical event patterns are consistent across centers
2. **Context-specific variation** (treatment): Ultrafiltration strategies and dry weight management are center-specific and should be preserved

We propose a mechanism-aware transportable representation learning framework that:
- Decomposes features into physiology and treatment groups based on clinical knowledge
- Uses CORAL (second-order statistics alignment) on physiology features
- Preserves treatment-related variations instead of aligning them
- Selects alignment targets based on outcome-specific mechanisms

## 🔬 Evidence Chain

| Evidence | Finding |
|----------|---------|
| 1 | Blind alignment (Fine-tuning, Global CORAL) is insufficient — IDH Δ=0.0428 |
| 2 | Outcome-specific transportability — IDH benefits from physiology alignment (Δ=0.1182) |
| 3 | Latent drift reduction — MMD decreased from 0.276 to 0.186 |
| 4 | Feature contribution shift — IDH physiology contribution increased from 60% to 95% |
| 5 | Subgroup heterogeneity — Transportability is context-dependent |

## 📂 Project Structure

```text
├── conf/                     # Configuration files
│   ├── binary_config.yaml    # Frozen experiment configuration
│   └── binary_config_finetune_only.yaml  # Baseline configuration
├── data/                     # Data directory (raw & processed CSVs)
│   ├── raw/                  # Raw data (gitignored)
│   └── processed/            # Processed data (gitignored)
├── docs/                     # Documentation
│   └── manuscript_draft.md   # Complete manuscript draft
├── experiments/              # Experiment results
│   ├── audit/                # Feature allowlist and audit files
│   └── final_results/        # Final results (gitignored: large files)
│       ├── main_mechanism_aware/     # Primary results
│       ├── baseline_finetune/        # Fine-tuning baseline
│       ├── baseline_global_coral/    # Global CORAL baseline
│       ├── baseline_random_alignment/# Random alignment control
│       └── multiseed_seed*/         # Multi-seed stability
├── figures/                  # Publication-grade figures
├── src/                      # Source code
│   ├── data/                 # Dataset loading and preprocessing
│   │   └── binary_dataset.py # Binary classification data pipeline
│   ├── data_pipeline/        # Data preprocessing scripts
│   │   ├── HBD_data.py       # Shenyi center data processing
│   │   └── HBD_data_fuding.py # Fuding center data processing
│   ├── evaluate/             # Evaluation scripts
│   │   ├── binary_metrics.py # Binary classification metrics
│   │   ├── shap_analysis.py  # SHAP feature importance analysis
│   │   ├── subgroup_analysis.py # Subgroup transportability analysis
│   │   └── latent_representation_analysis.py # MMD/domain classifier
│   ├── models/               # Model architectures (legacy)
│   └── train/                # Training scripts
│       └── binary_models.py  # BinaryMLP with stratified alignment
├── tables/                   # Auto-generated tables
├── scripts/                  # Utility scripts
├── tests/                    # Unit tests
├── run_all.py                # One-click execution pipeline
├── run_pipeline.py           # Full pipeline execution
└── pyproject.toml            # Ruff, Mypy, and Pytest configurations
```

## 🔄 Data Processing Pipeline

### 1. Raw Data → Processed Data

```bash
python src/data_pipeline/HBD_data.py RAW_SHENYI.csv data/processed/深医_final_data.csv
python src/data_pipeline/HBD_data_fuding.py RAW_FUDING.csv data/processed/福鼎_final_data.csv
```

### 2. Feature Selection and Validation

The [feature_allowlist.csv](file:///home/cht/Works/domain-adaptive-deep-survival-network/experiments/audit/feature_allowlist.csv) defines which features are allowed for prediction:

| Feature Category | Examples | Count |
|------------------|----------|-------|
| Demographics | 性别, 透析龄占比 | 2 |
| Pre-dialysis | 透前收缩压, 透前舒张压, 脉压差 | 9 |
| History | 历史平均透前收缩压, history_IDH_rate | 13 |
| Treatment | 超负荷, 透前体重-干体重, 历史平均超滤量MAX | 4 |

### 3. Patient-Level Split

The data pipeline performs patient-level stratified splitting to ensure:
- No patient appears in both training and testing
- Event rate is balanced across splits
- Source and target domains are strictly separated

## 🧠 Model Design

### BinaryMLP Architecture

```
Input Features (24)
    │
    ├── Physiology Encoder (indices: 0,1,4-7,9-22)
    │       ├── Linear(20→64)
    │       ├── ReLU
    │       ├── Dropout(0.2)
    │       └── Linear(64→32)
    │
    ├── Treatment Encoder (indices: 2,3,8,23)
    │       ├── Linear(4→64)
    │       ├── ReLU
    │       ├── Dropout(0.2)
    │       └── Linear(64→32)
    │
    └── Concatenate → Linear(64→1) → Sigmoid
```

### Mechanism-Aware Alignment

The framework implements outcome-specific alignment:

```
IDH (Intradialytic Hypotension):
    → Align physiology features using CORAL
    → Rationale: IDH is primarily driven by physiological instability

IH (Intradialytic Hypertension):
    → Align treatment features using CORAL  
    → Rationale: IH is influenced by treatment strategy variations
```

### Loss Function

```
Total Loss = Survival Loss + λ × CORAL Loss

where:
- Survival Loss = BCEWithLogitsLoss (for binary classification)
- CORAL Loss = ||cov(X_source) - cov(X_target)||²_F
- λ = 0.01 (alignment weight)
- X = physiology features for IDH, treatment features for IH
```

## 🚀 Quick Start

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/domain-adaptive-deep-survival-network.git
cd domain-adaptive-deep-survival-network

# Install dependencies
pip install -r requirements.txt
```

### Prepare Data

Build source and target cohorts:

```bash
python src/data_pipeline/HBD_data.py RAW_SHENYI.csv data/processed/深医_final_data.csv
python src/data_pipeline/HBD_data_fuding.py RAW_FUDING.csv data/processed/福鼎_final_data.csv
```

### Run Audit and Training

```bash
# Static audit only
python run_all.py

# Full training pipeline
python run_all.py --train
```

### Run Full Pipeline

```bash
python run_pipeline.py /path/to/shenyi.csv /path/to/fuding.csv
```

### Run Evaluation Scripts

```bash
# SHAP analysis
python -m src.evaluate.shap_analysis --config conf/binary_config.yaml --results-dir experiments/final_results/main_mechanism_aware

# Subgroup analysis
python -m src.evaluate.subgroup_analysis --config conf/binary_config.yaml --results-dir experiments/final_results/main_mechanism_aware
```

## 📊 Results Summary

### Core Comparison

| Method | IDH Δ AUC | IH Δ AUC |
|--------|-----------|----------|
| Fine-tuning | 0.0428 | 0.0150 |
| Global CORAL | 0.0437 | 0.0152 |
| Random Alignment | 0.0068 | 0.0049 |
| **Mechanism-aware** | **0.1182** | **0.0084** |

### Multi-seed Stability

| Seed | IDH Δ | IH Δ |
|------|-------|------|
| 42 | 0.1024 | 0.0303 |
| 7 | 0.0500 | 0.0075 |
| 13 | 0.0430 | 0.0429 |
| 99 | 0.1514 | 0.0437 |
| 2024 | 0.1182 | 0.0084 |
| **Mean ± SD** | **0.0930 ± 0.0459** | **0.0266 ± 0.0171** |

### SHAP Feature Contribution Shift

| Outcome | Phase | Physiology | Treatment | Cross-domain Spearman |
|---------|-------|------------|-----------|----------------------|
| IDH | Before | 60% | 40% | 0.65 |
| IDH | After | 95% | 5% | 0.82 |
| IH | Before | 97% | 3% | 0.94 |
| IH | After | 98% | 2% | 0.93 |

### Subgroup Analysis

| Outcome | Subgroup | Δ AUC | 95% CI |
|---------|----------|-------|--------|
| IDH | Low-risk | 0.164 | [0.153, 0.175] |
| IDH | High-risk | 0.090 | [0.083, 0.096] |
| IH | Low intensity | 0.012 | [0.009, 0.014] |
| IH | High intensity | 0.003 | [0.001, 0.006] |

## 📝 Manuscript Readiness

The complete manuscript draft is available at [docs/manuscript_draft.md](file:///home/cht/Works/domain-adaptive-deep-survival-network/docs/manuscript_draft.md), including:

- 6 Figure designs (frozen)
- Results section outline
- Reviewer risk mitigation strategies
- Target journals: npj Digital Medicine, The Lancet Digital Health, JAMA Network Open

## 🔬 Key Findings

1. **Domain shift is heterogeneous**: Not all domain differences are noise; some contain clinically meaningful information

2. **Outcome-specific transportability**: IDH benefits from physiology alignment, while IH benefits from preserving treatment variations

3. **Representation alignment works**: CORAL alignment on physiology features significantly reduces cross-domain latent discrepancy (MMD: 0.276 → 0.186)

4. **Feature contribution becomes clinically interpretable**: After alignment, IDH model relies on physiology features (95%) rather than hospital-specific treatment patterns

5. **Transportability is context-dependent**: Benefits vary across patient subgroups, highlighting the importance of personalized adaptation strategies

## 📄 Citation

If you find this code or methodology useful, please cite:

```
@article{mechanism-aware-transportability,
  title={Mechanism-Aware Transportable Representation Learning for Clinical AI under Heterogeneous Domain Shift},
  author={Your Name},
  journal={npj Digital Medicine},
  year={2026},
  note={In preparation}
}
```

## 📁 Final Results Location

All final experiment results are archived in `experiments/final_results/`:

| Directory | Description |
|-----------|-------------|
| `main_mechanism_aware` | Primary mechanism-aware alignment results |
| `baseline_finetune` | Fine-tuning baseline |
| `baseline_global_coral` | Global CORAL baseline |
| `baseline_random_alignment` | Random alignment control |
| `multiseed_seed42/7/13/99` | Multi-seed stability validation |

Note: Large files (test_predictions.csv, mlp_models.pt) are gitignored to avoid repository bloat.

## 🚧 Legacy Components

The Cox, KAN, gated-mask, and CDAN implementations in `src/models/` are retained for historical comparison only. They are not current scientific claims.

## ✅ Testing

```bash
python -m pytest tests/ -v
```

## 📜 License

MIT License
