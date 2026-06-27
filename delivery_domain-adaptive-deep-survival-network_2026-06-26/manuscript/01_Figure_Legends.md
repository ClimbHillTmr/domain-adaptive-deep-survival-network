# Figure Legends

## Figure 1. Study cohort flow and analysis population.
This figure summarizes the cohort construction process for the Shenyi development cohort and the Fuding external validation cohort. Raw dialysis-session counts, preprocessing-derived analysis cohorts, and retained audit exclusions are shown to document the final study populations used for model development and external validation. The figure is intended to provide a transparent overview of data provenance and final analytic sample size across centers.

## Figure 2. Cross-center baseline shift in key clinical variables.
Median values of representative clinical variables are shown for the Shenyi source cohort and the Fuding target cohort, highlighting marked between-center distribution differences. The displayed variables include pre-dialysis systolic blood pressure, pre-dialysis diastolic blood pressure, pulse pressure, dry weight, dialysate temperature, weight-normalized ultrafiltration rate, and session duration. The plot illustrates the magnitude of domain shift that motivates cross-center adaptation.

## Figure 3. Distribution of IDH timing stages across centers.
Stacked proportions show the distribution of intradialytic hypotension timing stages in the Shenyi and Fuding cohorts. Stages represent early events, intermediate events, later events, and sessions without early hypotension. This figure demonstrates heterogeneity in the temporal pattern of hemodynamic deterioration across centers and provides context for time-to-event modeling.

## Figure 4. Center-level Kaplan-Meier curves for IDH-free survival.
Kaplan-Meier curves depict IDH-free survival over dialysis time in the Shenyi and Fuding cohorts. Separation between the curves reflects distinct event-time distributions across centers, supporting the presence of clinically meaningful temporal heterogeneity between the development and validation domains.

## Figure 5. External validation performance comparison.
External validation performance is compared between the zero-shot baseline and the CDAN-GSN model. Points indicate C-index estimates and horizontal bars indicate 95% confidence intervals. This figure summarizes discrimination performance under external validation and visually highlights the incremental gain associated with domain-adaptive modeling.

## Figure 6. Target-center IDH burden across major clinical subgroups.
Observed IDH event rates are shown across major clinical subgroups in the Fuding validation cohort, including age, sex, and hypertension-related strata. Bars represent subgroup-specific event rates and sample sizes are annotated to facilitate interpretation of subgroup burden and heterogeneity in the external validation population.

## Figure 7. Calibration performance of the CDAN-GSN model in the external validation cohort.
Calibration plots are shown for clinically relevant prediction horizons in the external validation cohort. Panel a presents calibration for 60-minute IDH risk and panel b presents calibration for 120-minute IDH risk. Each point represents the observed event rate within a predicted-risk bin, and the dashed diagonal indicates ideal calibration. These plots assess the agreement between predicted and observed risks beyond rank-based discrimination.

## Figure 8. Decision curve analysis of the CDAN-GSN model in the external validation cohort.
Decision curve analysis is shown for clinically relevant prediction horizons in the external validation cohort. Panel a presents net benefit for 60-minute IDH prediction and panel b presents net benefit for 120-minute IDH prediction. The model is compared with the treat-all and treat-none strategies across a range of threshold probabilities to evaluate potential clinical utility in threshold-based decision settings.

## Figure S1. Schematic architecture of the CDAN-GSN framework.
This supplementary figure illustrates the model architecture, including feature tokenization, gated representation learning, and conditional domain-adversarial alignment. The figure is intended as a methodological schematic and should be interpreted as statistical joint-distribution alignment rather than a causal directed acyclic graph or intervention framework.

## Figure S2. Hypothesized longitudinal phenotype transitions during hemodialysis.
This supplementary figure presents a conceptual schematic of hypothesized longitudinal phenotype transitions during hemodialysis, including stable, vulnerable, acute, and chronic states. It is provided to support physiological interpretation and discussion only, and is not derived from a formal multi-state or transition-probability model.
