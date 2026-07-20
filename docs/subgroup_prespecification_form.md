# Patient-Cluster Subgroup Prespecification Form

Complete this form before inspecting target-test subgroup performance. It does
not recommend cutpoints; clinical and statistical collaborators must supply and
justify them independently of test results.

## Analysis binding

- Accepted confirmatory-v2 `run_id`:
- Protocol version:
- UTC lock time:
- Clinical approver role:
- Statistical approver role:
- Rationale document or protocol reference:
- Attestation that target-test subgroup performance was not inspected: yes/no
- Minimum unique patients required in each level:

## Subgroup definitions

For every subgroup, record:

- endpoint and subgroup name;
- source feature and clinical units;
- definition type: numeric cutpoint or explicit categorical value sets;
- exact cutpoint and inequality direction, or exact category membership;
- two human-readable level labels;
- clinical rationale and source supporting the definition;
- handling of missing or unmapped values;
- whether the feature can vary across sessions for one patient.

Required subgroups:

| Endpoint | Subgroup | Candidate feature | Locked definition | Rationale |
|---|---|---|---|---|
| IDH | Baseline blood pressure | `透前收缩压` |  |  |
| IDH | Prior instability history | `history_IDH_rate` |  |  |
| IDH | Ultrafiltration intensity | `历史平均超滤率_mean` |  |  |
| IH | Treatment intensity | `历史平均超滤量MAX` |  |  |
| IH | Volume overload | `超负荷` |  |  |
| IH | Hypertension history | `history_HBP_rate` |  |  |

## YAML encoding examples

Numeric structure (replace every placeholder with an approved value):

```yaml
definition:
  type: numeric_cutpoint
  cutpoint: REQUIRED_APPROVED_VALUE
  labels: [below_cutpoint, at_or_above_cutpoint]
```

Categorical structure:

```yaml
definition:
  type: categorical_sets
  levels:
    level_1: [REQUIRED_EXPLICIT_VALUES]
    level_2: [REQUIRED_EXPLICIT_VALUES]
```

## Frozen statistical procedure

- Unit of resampling: patient ID.
- Replicates: 1,000.
- Level estimates: updated-model ROC AUC and updated-minus-source AUC.
- Interval: percentile 95% patient-cluster bootstrap interval.
- Interaction contrast: difference in delta AUC between the two locked levels.
- Interaction p-value: two-sided bootstrap sign-tail probability with a
  plus-one correction.
- A subgroup fails rather than being silently reported when a level is below
  the locked minimum patient count or lacks both outcome classes.

After completing the form, encode the same definitions in
`conf/subgroup_protocol.yaml`, change its status to `prespecified_locked`, and
record the accepted server run ID. The analysis command remains blocked until
all fields pass validation.
