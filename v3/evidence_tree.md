# V3 Evidence Tree

The project is indexed by scientific evidence, not by model names. Models A-G are tools under Evidence 3 only.

```text
Research question
Why do clinical AI models fail to transport across hospitals, and can
clinically informed selective alignment improve transportability?
│
├── Evidence 1 — Observed heterogeneous domain shift
│   ├── E1.1 Population and outcome composition
│   ├── E1.2 Feature distribution and missingness
│   ├── E1.3 Treatment/policy proxies
│   ├── E1.4 Measurement/process metadata [currently blocked]
│   └── E1.5 Disease-mechanism attribution [not identifiable here]
│
├── Evidence 2 — What information transports?
│   ├── E2.1 Feature-group transportability ranking
│   ├── E2.2 Representation transportability
│   └── E2.3 Outcome-specific transportability
│
├── Evidence 3 — Does selective alignment add value?
│   ├── E3.1 No-alignment and global-alignment controls
│   ├── E3.2 Clinically partitioned selective alignment
│   └── E3.3 Random controls A-D
│
├── Evidence 4 — Why did the representation change?
│   ├── E4.1 Latent shift and collapse diagnostics
│   ├── E4.2 Outcome-information preservation
│   ├── E4.3 Attribution stability
│   └── E4.4 Calibration preservation
│
└── Evidence 5 — Can the model support a deployment workflow?
    ├── E5.1 Calibration
    ├── E5.2 Decision curves
    ├── E5.3 Alert and review burden
    └── E5.4 Clinically locked thresholds
```

## Evidence logic

Evidence 1 is descriptive and cannot by itself identify causal shift mechanisms. Evidence 2 asks which information transfers before introducing alignment. Evidence 3 tests the intervention. Evidence 4 tests whether any predictive effect is consistent with information preservation rather than representation collapse. Evidence 5 is conditional on a model reaching the clinical-evaluation gate.

The project remains scientifically complete when Evidence 3 is negative. In that case, the manuscript reports the shift and transportability map and the failure of selective alignment to outperform its controls.
