# Results

## Study Population and Final Analytic Cohorts
A total of 291,828 dialysis sessions were included in the final analysis, comprising 216,604 sessions from the Shenyi development cohort and 75,224 sessions from the Fuding external validation cohort (Fig. 1). The cohort flow figure documents the preprocessing-derived analytic populations and preserved audit exclusions, supporting traceability of the final study sample and reducing ambiguity regarding data provenance.

## Cross-center Baseline and Temporal Heterogeneity
Marked baseline differences were observed between the source and target cohorts (Fig. 2). Relative to the Shenyi cohort, the Fuding cohort had higher pre-dialysis systolic blood pressure, higher pulse pressure, lower dry weight, and substantially higher weight-normalized ultrafiltration rate. In Table 1, the median pre-dialysis systolic blood pressure was 142.0 mmHg in Shenyi and 156.0 mmHg in Fuding, while the median weight-normalized ultrafiltration rate was 7.7 versus 11.1 mL/kg/h, respectively. The proportion of sessions classified as high-risk ultrafiltration was also markedly higher in Fuding (30.3%) than in Shenyi (4.9%).

The timing structure of intradialytic hypotension also differed across centers. As shown in Fig. 3, the stage distribution of IDH events was not identical between Shenyi and Fuding, indicating that between-center heterogeneity involved both baseline covariate shift and event-time shift. This temporal heterogeneity was further reflected in the Kaplan-Meier curves (Fig. 4), which showed visible separation in IDH-free survival across dialysis time, supporting the use of a time-to-event modeling framework rather than a purely static binary classifier.

## External Validation Performance
In external validation, the CDAN-GSN model outperformed the zero-shot baseline in discrimination (Fig. 5). The C-index improved from 0.825 (95% CI, 0.822-0.828) for the zero-shot baseline to 0.841 (95% CI, 0.838-0.843) for CDAN-GSN. Although the absolute gain was modest, the direction of improvement was consistent with the hypothesis that domain-adaptive representation learning can improve transportability under cross-center shift.

## Subgroup Burden in the Target Cohort
Observed IDH burden in the external validation cohort was heterogeneous across clinically relevant subgroups (Fig. 6). This pattern indicates that population-average summaries alone are insufficient to characterize external-center risk and supports a clinically stratified interpretation of the target cohort. The subgroup burden figure was designed to emphasize external-center disease burden rather than overclaim subgroup-specific model superiority.

## Calibration Based on Real Case-level Predictions
To move beyond discrimination-only evaluation, we exported real case-level predictions for the held-out external test cohort. A total of 60,143 dialysis sessions were included in the final test set used for calibration and decision-curve analysis. At the 30-minute horizon, the observed event rate was 0.11%, and the mean predicted probability was 0.10%. At the 60-minute horizon, the observed event rate was 14.58%, whereas the mean predicted probability was 11.37%. At the 120-minute horizon, the observed event rate was 25.02%, whereas the mean predicted probability was 20.28%.

As shown in Fig. 7, the model displayed broadly reasonable risk ordering but a tendency toward underestimation at clinically relevant horizons in the external center. The corresponding Brier scores were 0.0011 at 30 minutes, 0.0962 at 60 minutes, and 0.1280 at 120 minutes. Together, these findings suggest that the model retained useful probability structure in external validation, although absolute calibration remained imperfect.

## Clinical Utility by Decision Curve Analysis
Decision curve analysis based on real exported predictions demonstrated that the model achieved positive net benefit over a range of clinically relevant thresholds at both 60 and 120 minutes (Fig. 8). Compared with the treat-all and treat-none strategies, the model-supported strategy provided improved decision support within practically meaningful threshold ranges. These results suggest that the model may be useful for threshold-triggered monitoring or preventive intervention workflows, provided that the residual underestimation of absolute risk is taken into account when deployed in external-center settings.
