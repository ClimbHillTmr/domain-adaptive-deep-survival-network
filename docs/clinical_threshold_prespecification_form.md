# Clinical Threshold and Decision-Curve Prespecification Form

Complete and sign this form before reviewing target-test decision curves by threshold or target-test operating characteristics at candidate clinical thresholds. This form does not recommend a threshold. Nephrology, statistical, and clinical-workflow collaborators must provide the values and rationale independently of target-test utility results.

## Analysis binding

- Protocol version:
- Registered historical run ID:
- UTC lock time:
- Nephrology approver role:
- Statistical approver role:
- Clinical-workflow approver role:
- Rationale document or approved protocol reference:
- Attestation that approvers were not shown target-test DCA or candidate-threshold operating results before locking: yes/no

## Population and intended action

- Population and care setting:
- Prediction time:
- Who receives the alert:
- What review or monitoring action the alert may trigger:
- Whether the action differs for IDH and IH:
- Explicitly excluded use, including autonomous treatment decisions:

## Endpoint-specific thresholds

Complete one row per endpoint. Probabilities must be entered on the 0–1 scale.

| Endpoint | DCA lower | DCA upper | Step | Prespecified operating threshold(s) | False-alert consequence | Missed-event consequence | Clinical rationale and source |
|---|---:|---:|---:|---|---|---|---|
| IDH |  |  |  |  |  |  |  |
| IH |  |  |  |  |  |  |  |

The DCA interval should cover only thresholds at which the stated alert action would plausibly be considered. Thresholds must not be chosen because a target-test curve has favorable net benefit. If no clinically defensible range exists, record that conclusion and retain DCA as a descriptive sensitivity analysis.

## Frozen analysis

- Models: source MLP, updated MLP, and target-local logistic regression.
- Comparators: alert all and alert none.
- Decision-curve definition: net benefit = TP/N − FP/N × threshold/(1−threshold).
- Operating analysis: sensitivity, specificity, PPV, NPV, events detected, false alerts, and total alerts per 1,000 sessions.
- Uncertainty: 1,000 percentile patient-cluster bootstrap replicates, seed 20260715.
- Reporting level: population-level research evaluation, not a treatment recommendation.
- Existing validation-Youden analyses remain separately labeled and are not reclassified as clinically prespecified.

After approval, encode the same values in `conf/clinical_threshold_protocol.yaml`, set `status: prespecified_locked`, set `prespecified_without_target_test_dca_access: true`, and run the validator before generating the prespecified clinical-utility package.
