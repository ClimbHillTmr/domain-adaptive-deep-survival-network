# Clinical Utility Report

Registered run: `binary_20260719_044949_c6c955a30fc2`

This package uses the registered held-out target predictions only. No predictive model was fitted or updated; calibration regressions are descriptive diagnostics.

- Calibration intercept, slope, ECE, Brier score, and fixed-threshold alert metrics include 95% percentile intervals from 1000 patient-cluster bootstrap replicates (seed 20260715).
- Alert analysis uses each model's threshold selected on its corresponding validation patients and stored before test scoring.
- Decision curves cover thresholds 0.01-0.60 as a sensitivity analysis. No clinically relevant range has been claimed because nephrology threshold prespecification is not yet documented.
- Operating characteristics remain tied to validation-derived Youden thresholds and are not clinically prespecified deployment thresholds.
