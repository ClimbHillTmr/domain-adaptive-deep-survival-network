# Table 2. Held-out target performance from the registered seed-2024 run

| Endpoint | Model | ROC AUC (95% CI) | PR AUC (95% CI) | Brier (95% CI) | Paired comparison | Δ AUC (95% CI) |
|---|---|---:|---:|---:|---|---:|
| IDH | Source MLP | 0.7951 (0.7768–0.8125) | 0.7463 (0.7066–0.7795) | 0.2106 (0.1983–0.2227) | — | — |
| IDH | Updated MLP | 0.8375 (0.8234–0.8522) | 0.7815 (0.7460–0.8118) | 0.1566 (0.1489–0.1636) | Updated MLP − source MLP | 0.0424 (0.0348–0.0515) |
| IDH | Target-local logistic | 0.8353 (0.8207–0.8504) | 0.7800 (0.7443–0.8103) | 0.1576 (0.1500–0.1646) | Updated MLP − target-local logistic | 0.0022 (-0.0003–0.0047) |
| IH | Source MLP | 0.8516 (0.8347–0.8673) | 0.4907 (0.4318–0.5416) | 0.0872 (0.0761–0.0986) | — | — |
| IH | Updated MLP | 0.8681 (0.8522–0.8826) | 0.5234 (0.4696–0.5689) | 0.0791 (0.0693–0.0891) | Updated MLP − source MLP | 0.0165 (0.0112–0.0229) |
| IH | Target-local logistic | 0.8678 (0.8524–0.8822) | 0.5172 (0.4600–0.5645) | 0.0797 (0.0699–0.0898) | Updated MLP − target-local logistic | 0.0003 (-0.0017–0.0022) |

CI values use 1,000 patient-cluster bootstrap replicates. PR AUC, precision-recall area under the curve.
