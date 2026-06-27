# Figure Order and Manuscript Storyline

## Recommended Main-Text Figure Order

1. **Fig. 1. Study cohort flow and analysis population**
   - Purpose: establish data provenance, sample size, and analytic traceability.
   - Reader question answered: what data entered the study and how many sessions were finally analyzed?

2. **Fig. 2. Cross-center baseline shift in key clinical variables**
   - Purpose: justify why external validation is non-trivial and why domain adaptation is scientifically necessary.
   - Reader question answered: are Shenyi and Fuding actually different enough to constitute domain shift?

3. **Fig. 3. Distribution of IDH timing stages across centers**
   - Purpose: show that event-time structure differs between centers, not just baseline covariates.
   - Reader question answered: do the two centers differ in when hypotension tends to occur during dialysis?

4. **Fig. 4. Center-level Kaplan-Meier curves for IDH-free survival**
   - Purpose: visualize time-to-event heterogeneity at the cohort level.
   - Reader question answered: is the survival experience meaningfully different between centers?

5. **Fig. 5. External validation performance comparison**
   - Purpose: deliver the core discrimination result.
   - Reader question answered: does CDAN-GSN outperform the zero-shot baseline in external validation?

6. **Fig. 6. Target-center IDH burden across major clinical subgroups**
   - Purpose: show that external-center burden is heterogeneous and clinically stratified.
   - Reader question answered: which subgroups in Fuding carry greater IDH burden?

7. **Fig. 7. Calibration performance of the CDAN-GSN model**
   - Purpose: move from ranking performance to absolute risk assessment.
   - Reader question answered: are the predicted probabilities numerically aligned with observed outcomes?

8. **Fig. 8. Decision curve analysis of the CDAN-GSN model**
   - Purpose: close the loop with clinical utility.
   - Reader question answered: does the model create net benefit over treat-all or treat-none strategies?

## Recommended Supplementary Figures

- **Fig. S1. Schematic architecture of the CDAN-GSN framework**
  - Use as methodological support, not as a main-text centerpiece.

- **Fig. S2. Hypothesized longitudinal phenotype transitions during hemodialysis**
  - Use as a conceptual discussion aid only.

## Results Paragraph Order

1. Study population and final analytic cohorts.
2. Cross-center baseline and temporal heterogeneity.
3. External discrimination performance.
4. Subgroup burden in the target cohort.
5. Calibration and clinical utility based on real case-level predictions.

## Discussion Paragraph Order

1. Principal findings.
2. Why cross-center shift matters and why domain adaptation is justified.
3. Interpretation of calibration and decision-curve results.
4. Clinical implications for dialysis monitoring and intervention.
5. Methodological boundaries: predictive, not causal.
6. Limitations.
7. Conclusion and next-step implications.

## Core Narrative in One Sentence

This manuscript argues that substantial cross-center hemodynamic shift exists between Shenyi and Fuding, that domain-adaptive survival modeling improves external discrimination under this shift, and that real case-level calibration and decision-curve analyses support potential clinical utility despite residual probability underestimation in the external center.
