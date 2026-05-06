# DA-DSN: Domain-Adaptive Deep Survival Network for Intradialytic Hypotension Prediction

A deep learning framework for cross-center prediction of intradialytic hypotension (IDH) using domain adaptation and survival analysis.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run comprehensive experiment (ablation + ML baselines + calibration + DCA + SHAP)
python scripts/run_comprehensive_experiment.py

# Run ablation study only
python scripts/run_dadsn_ablation.py
```

## Project Structure

```
├── configs/                    # Configuration files
│   ├── config.yaml            # Main experiment config
│   └── defaults.yaml          # Default hyperparameters
├── data/                      # Data directory (not tracked)
│   ├── source/                # Source domain data (深医)
│   └── target/                # Target domain data (福鼎)
├── scripts/                   # Entry point scripts
│   ├── run_comprehensive_experiment.py  # Full experiment pipeline
│   ├── run_dadsn_ablation.py            # Ablation study only
│   └── run_ensemble.py                  # Ensemble inference
├── src/                       # Source code
│   ├── data/
│   │   └── loader.py          # Data loading and preprocessing
│   ├── models/
│   │   └── dadsn_model.py     # DA-DSN model architecture
│   ├── metrics/
│   │   └── survival_metrics.py # C-index, IBS evaluation
│   ├── training/
│   │   └── dadsn_runner.py    # Training loops and experiment runner
│   └── evaluation/
│       ├── ml_baselines.py    # Traditional ML baselines (LR, RF, XGBoost)
│       ├── calibration.py     # Calibration curves and DCA
│       ├── shap_analysis.py   # SHAP interpretability analysis
│       └── plotting.py        # Publication-quality figure generation
├── runs/                      # Experiment outputs (auto-generated)
├── figures/                   # Generated figures (auto-generated)
├── requirements.txt
└── README.md
```

## Model Architecture

DA-DSN combines:
- **ResNet1D Encoder**: Processes temporal physiological signals
- **Static MLP**: Processes baseline patient features
- **Cross-Attention Fusion**: Integrates static and dynamic features
- **Domain Adaptation**: MMD or DANN for cross-center generalization
- **DeepSurv Head**: Cox partial likelihood for survival prediction

## Key Features

- **Domain Adaptation**: MMD (Maximum Mean Discrepancy) or DANN (Domain-Adversarial Neural Network)
- **Multi-level Alignment**: Aligns embeddings at static, dynamic, and fusion layers
- **Model Ensemble**: Averages predictions from multiple seeds for stability
- **Clinical Evaluation**: Calibration curves, DCA, SHAP analysis

## Citation

If you use this code, please cite:

```bibtex
@article{dadsn2024,
  title={Domain-Adaptive Deep Survival Network for Cross-Center Intradialytic Hypotension Prediction},
  author={Your Name},
  journal={Under Review},
  year={2024}
}
```

## License

MIT License
