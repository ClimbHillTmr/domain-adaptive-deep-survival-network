# V3 Project Research Contract

Contract version: `v3_research_contract_1`  
Data protocol version: `v3_hbd_data_protocol_1`  
Status: V3 data, features, split, preprocessing, random controls, and the 420-run server grid are frozen; training authorization withheld.

This file is the starting point and mandatory checkpoint for every coding, analysis, training, visualization, and manuscript task in this repository. If code, configuration, a result artifact, or another document conflicts with this contract, stop and resolve the conflict by an explicit versioned contract amendment. Do not silently inherit V1/V2 behavior.

## 1. Scientific position

Working research question:

> Which observed components of cross-center clinical variation retain predictive information, and does a clinically prespecified selective alignment strategy improve held-out target-center prediction beyond matched target updating, global alignment, and random feature partitions?

V3 is an evidence-driven clinical AI transportability study. CORAL is an experimental operator, not the scientific contribution. V1/V2 results are historical hypothesis-generating observations only.

The design may support:

- observed heterogeneous cross-center shift;
- supervised target-center updating;
- comparative predictive transportability of prespecified information groups; and
- incremental predictive value, or lack of value, from selective alignment.

The two-center retrospective design cannot identify biological mechanism shift, causal treatment-policy effects, or a domain-invariant representation. `Physiology`, `treatment/context`, and `history` are clinical feature partitions, not recovered mechanisms.

Negative selective-alignment results complete the protocol and must be reported.

## 2. Authoritative data entry points

All V3 analyses begin only from outputs produced by:

- source/Shenyi: `src/data_pipeline/HBD_data.py`;
- target/Fuding: `src/data_pipeline/HBD_data_fuding.py`;
- shared deterministic implementation: `src/data_pipeline/data_process.py`.

No notebook, model script, or manual CSV edit may redefine the cohort, endpoint, start-time features, or history variables. Corrections must be made in the shared pipeline, tested, versioned, and rebuilt into new immutable outputs.

Frozen raw snapshot identities:

| Center | Role | Required SHA-256 |
|---|---|---|
| Shenyi | source | `607a7f115efa2db64c3f23190f82222978d10146570bdbf6e1f66118bd2320f9` |
| Fuding | target | `6fcfd523cd60aedc245d4b51e072aeca777f65e9a56554fdebaf45d3ee5f3813` |

The official center entry points must reject any raw file whose hash differs. The corrected V3 cohorts have now been deliberately rebuilt and frozen below; historical processed hashes are not V3 hashes.

Frozen V3 processed identities:

| Center | Dataset ID | Sessions / patients | IDH prevalence | IH prevalence | Path | SHA-256 |
|---|---|---:|---:|---:|---|---|
| Shenyi/source | `V3-SOURCE-20260723` | 211,452 / 1,628 | 30,694 / 14.52% | 59,978 / 28.36% | `data/v3/source_v3_hbd.csv` | `b967692a7a6d4289ecf7bd58b95550b7e8baa487120566e2f4d1775fdc547723` |
| Fuding/target | `V3-TARGET-20260723` | 74,947 / 430 | 28,881 / 38.54% | 8,962 / 11.96% | `data/v3/target_v3_hbd.csv` | `721c408a2ab2fa8504173ab0daa8442cbebf0f620033a17ca711264b58040be1` |

Dataset version is `v3_hbd_20260723`. The complete freeze record is `data/v3/data_freeze_manifest.json`. These files must not be overwritten; a change requires a new dataset and protocol version.

Frozen construction audit:

| Center | Raw rows | Invalid sequence | Misaligned sequence | Invalid time | Missing age | Missing/invalid vintage | Invalid temperature set missing |
|---|---:|---:|---:|---:|---:|---:|---:|
| Source | 222,836 | 6,521 | 4,728 | 135 | 641 | 1,181 | 338 |
| Target | 76,519 | 176 | 1,225 | 171 | 0 | 14 | 0 |

Both strictly-prior history audits pass with zero feature mismatches and zero duplicate session IDs. Source contains two tied patient-time rows handled using only earlier timestamps; target contains none.

## 3. Fixed cohort-construction protocol

The prediction unit is one dialysis session. The independent statistical unit is the patient.

A raw session is retained only when all of the following are true:

1. intradialytic systolic and diastolic sequences are parseable;
2. time nodes are parseable and contain at least two observations;
3. systolic, diastolic, and time-node lengths match;
4. baseline systolic and diastolic pressure are finite; and
5. calculated session duration is positive.

No additional adult, ICU, acute-dialysis, minimum-exposure, or measurement-frequency exclusion may be introduced without a new protocol version and a row-count impact report. Current data do not justify silently inventing those eligibility rules.

The source snapshot contains patients whose derived age is below 18 years. The audit must report their row and patient counts. Unless a later clinically approved protocol amendment introduces an adult criterion, they remain in the fixed cohort and the manuscript must not describe the cohort as adult-only.

Each output must contain a deterministic unique `session_id`, a protocol version, input/output hashes, retained/rejected counts, patient counts, endpoint counts, and quality-control counts.

Direct identifiers and non-analytic row identifiers, including `姓名`, `透析记录id`, and `Unnamed: 0`, must not enter processed analysis cohorts. Exact birth date is used only to derive age and is then removed.

## 4. Fixed endpoint protocol

Prediction time is the start of the dialysis session.

IDH:

```text
baseline SBP - any intradialytic SBP >= 30 mmHg
OR
any intradialytic SBP <= 90 mmHg
```

IH:

```text
any intradialytic MAP - baseline MAP > 10 mmHg
```

Intradialytic MAP is `(SBP + 2*DBP)/3`. Baseline MAP is represented by the single canonical column `透前动脉压`. The source raw column `平均动脉压` is a current-session sequence and is excluded from start-time predictors and from the processed V3 cohort.

The first qualifying intradialytic measurement defines event time. For the binary study, non-event time remains zero and is not interpreted as survival follow-up.

## 5. Fixed start-time data-quality rules

These rules are based on raw-data semantics, not model performance.

### Sex

- `女/female/f/0 -> 0`;
- `男/male/m/1 -> 1`;
- unknown, blank, or unmapped values -> missing;
- mapping and unknown counts are written to the audit artifact.

Sex is never converted with generic `pd.to_numeric` before canonical mapping.

### Age and dialysis exposure

- derive age at each session as `(透析日期 - 出生日期)/365.25`;
- accept derived or fallback raw age only when `0 < age <= 120`;
- derive dialysis vintage years as `(透析日期 - 首次透析日期)/365.25`;
- accept vintage only when `0 <= vintage <= age`;
- derive `透析龄占比 = 透析龄年 / 年龄` only for valid age and vintage;
- do not fill missing age or vintage with zero;
- primary models use `透析龄年`; `透析龄占比` is a prespecified sensitivity representation and the two are not entered together.

This protocol repairs the currently all-missing raw `年龄` field from the available birth date. It does not infer dates that are absent.

### Pre-dialysis temperature

- the broad technical plausibility range is 30.0-45.0 °C inclusive;
- numeric values outside that range are set to missing, never decimal-shifted or winsorized by guesswork;
- `透前体温_异常标记` records observed invalid values;
- `透前体温_缺失标记` records original missingness plus invalid values;
- primary physiology models may use cleaned temperature but not these process indicators;
- the indicators are reserved for missingness/process-shift analysis.

### Derived hemodynamics

- primary BP representation: `透前收缩压` and `透前舒张压`;
- `透前动脉压` and `脉压差` are deterministic sensitivity representations;
- raw BP and its deterministic derivatives are not entered together in the same primary probe;
- `平均动脉压` is excluded as current-session information and duplicate naming.

### History

All historical means and rates use sessions with timestamps strictly earlier than the current session. Rows tied at the same patient timestamp share history constructed only from earlier timestamps. First observed sessions have zero-valued history summaries by definition, not because missing clinical measurements equal zero.

## 6. Fixed feature protocol

No feature is labeled invariant. All feature inclusion decisions are made before V3 results and cannot be changed because an ablation performs poorly.

Primary ontology:

| Tier | Primary variables | Interpretation boundary |
|---|---|---|
| Population/exposure | `年龄`, `性别`, `透析龄年` | demographics and chronic exposure, not biology recovered by the model |
| Observed physiology | pre-dialysis SBP/DBP, respiratory rate, cleaned temperature | observed start-time state |
| Treatment/context | `透前体重-干体重`, `超负荷`, `历史平均超滤量MAX`, `历史平均超滤率_mean` | volume-management and treatment-intensity proxies, not causal policy effects |
| Strictly prior history | historical UF, weight, BP, IDH/IH burden, and IDH timing-bin rates | prior trajectory only; no current-session intradialytic summaries |
| Measurement/process sensitivity | temperature invalid/missing indicators and any subsequently validated device/workflow metadata | descriptive/sensitivity analysis, excluded from primary mechanism partition |

The four candidate treatment categories are not cross-center comparable in the current snapshots: source values are undocumented integer codes, whereas target values are clinical labels. Anticoagulant type, dialysis modality, access type, and access location are therefore excluded from primary V3 models. They may be restored only by a versioned, clinically verified crosswalk; integer-to-label meanings must never be guessed. Dialysate conductivity is constant in both centers and target dialysate calcium is constant, so these settings are retained for descriptive shift reporting but excluded from primary prediction and alignment partitions. These are semantic/support decisions made before V3 outcomes, not performance-driven exclusions.

If a future protocol includes comparable categorical variables, mappings must be fit on source-training patients only and target-only categories must receive an explicit unknown code. Continuous imputation and scaling parameters are fit on source-training patients only and applied unchanged to source validation and every target role.

Explicitly forbidden predictors:

- current-session intradialytic summaries;
- current-session event indicators or event times;
- dialysis end time, realized duration, or any value first known after prediction time;
- names, record identifiers, exact birth dates, or other direct identifiers;
- `平均动脉压`;
- any feature selected or excluded after viewing V3 final-test performance.

The locked primary predictive feature list is stored in `v3/data_contract/feature_allowlist.csv`. The primary dual-branch partition is:

- treatment/context: `透前体重-干体重`, `超负荷`, `历史平均超滤量MAX`, `历史平均超滤率_mean`;
- physiology/population/history: every other allowed primary feature.

No feature belongs to both branches. Derived BP sensitivity variables and temperature process indicators are outside the primary list.

The machine-readable `v3/data_contract/feature_manifest.csv` must agree with this section before training.

## 7. Fixed split and preprocessing protocol

Split seed: `20260715`.

- source: 90% patient train, 10% patient validation;
- target: 60% patients enter the update pool and 40% remain target test;
- within the target update pool, 15% of its patients are target validation/calibration and 85% are target update;
- patients are mutually exclusive across all roles;
- stratification uses patient-level presence/absence of both endpoints when feasible;
- every patient ID and role is written to the split manifest and hashed;
- no final-test row may fit categories, imputers, scalers, thresholds, hyperparameters, feature groups, or model selection.

The historical target test patients have influenced V1/V2. Reusing them in V3 is internal revalidation, not a new external or untouched confirmation. Strong deployment claims require a new temporal or external cohort.

Frozen allocation:

| Role | Patients | Sessions | IDH events | IH events |
|---|---:|---:|---:|---:|
| Source train | 1,465 | 187,479 | 27,367 | 52,531 |
| Source validation | 163 | 23,973 | 3,327 | 7,447 |
| Target update | 219 | 38,483 | 14,955 | 4,609 |
| Target calibration | 39 | 6,598 | 2,351 | 666 |
| Target test | 172 | 29,866 | 11,575 | 3,687 |

The row-level manifest is `v3/data_contract/split_manifest.csv`, SHA-256 `488191b7867519bb166cdd6403f78e73f46ed4bf3d6c21d4ab6a569f8465863c`. The JSON split artifact is `data/v3/split_manifest.json`, SHA-256 `6552a9085aed3f4b82ce3964768503d74ae28c038864b8d71c38af102513e568`.

Missing continuous values are imputed with the source-training median. Zero is not a generic missing-value code. Source-training mean and population SD are then used for scaling. Constant source-training features are excluded or explicitly flagged before training; they are not silently assigned evidence value.

The locked preprocessing artifact is `data/v3/preprocessing.json`, SHA-256 `000a404e726bc977b20d443c5691995f9919199370f7514abc641d16e10497bb`. It contains 22 ordered primary features, their 18/4 branch assignment, source-training medians/means/scales, the split ID, and semantic category exclusions. Its machine-readable registry row is `v3/preprocessing/preprocessing_manifest.csv`.

## 8. Model comparison contract

Required controls:

- source-only deployment baseline;
- A: single encoder, target updating, no alignment;
- B: single encoder, global CORAL;
- C: dual encoder, target updating, no alignment;
- D1: dual encoder, physiology-partition CORAL;
- D2: dual encoder, treatment/context-partition CORAL;
- E: dual encoder, locked random-partition CORAL controls;
- F: target-only local reference, not an oracle upper bound;
- G: dual encoder, global CORAL.

Valid alignment comparisons are architecture matched. C/D1/D2/E/G share exactly the same dual architecture, optimizer, batch sampling, training budget, stopping rule, preprocessing, data roles, and seeds. A/B share the same single architecture. Total capacity across single and dual families must be within 5% or receive a capacity-matched sensitivity analysis.

Locked architecture and optimization:

| Component | Fixed value |
|---|---|
| Dual branches | each branch `64 -> 32`, ReLU, dropout 0.20 |
| Dual parameter count | 5,761 for the frozen 18/4 partition |
| Single encoder | `64 -> 65`, ReLU, dropout 0.20 |
| Single parameter count | 5,763; relative difference from dual 0.035% |
| Prediction loss | focal binary cross-entropy; alpha is training-fold negative fraction; gamma 2.0 |
| Optimizer | AdamW; weight decay 1e-4; gradient clipping norm 1.0 |
| Batch size | 256 |
| Source pretraining | learning rate 1e-3; maximum 40 epochs; patience 5; source-validation ROC AUC |
| Target updating | learning rate 1e-4; maximum 20 epochs; patience 5; target-calibration ROC AUC |
| Updating population | pooled source-train and target-update sessions for A-E/G |
| CORAL | fixed weight 0.01; no V3 test-based lambda search |
| Calibration | Platt; target calibration for updated/local models, source validation for source-only models |

The machine-readable specification is `v3/experiments/training_protocol.yaml`. Model F is trained only on target-update labels under the single-encoder capacity and uses target calibration for selection; it is a local reference rather than an upper bound.

Both D1 and D2 are evaluated for both outcomes. The primary clinically motivated mapping—IDH physiology alignment and IH treatment/context alignment—is a falsifiable hypothesis, not an assumption of invariance.

Locked initialization seeds: `7, 13, 42, 99, 2024`. Seed is an optimization block, not an independent clinical replicate. All seeds, failures, and prespecified partitions are reported.

Random controls contain four prespecified families: unrestricted, variance/missingness matched, exact group-size matched, and clinical-type constrained. Exact group-size matching is the primary random null. Generate and hash 19 paired random-partition indices before any V3 outcome is viewed; the joint two-endpoint statistic has minimum one-sided empirical resolution 0.05. Other families attempt 19 partitions each and report infeasibility or failure without replacement based on results.

The primary partition is frozen in `v3/experiments/primary_feature_partition.json`, SHA-256 `97f6c3580674f637dfb1a6f85c3e5af07cb347d436bf7142e951c4bb51a27891`. All 76 random controls were generated before V3 model results and stored in `v3/experiments/random_partitions.csv`, SHA-256 `93d8aba11007ae6b8c77873a3ff1c937e4a22ddeaf310e3d2c63805967b9bc19`.

Adaptive alignment is outside V3 and cannot be added to rescue a negative result.

## 9. Primary estimand and statistical analysis plan

### Primary mechanism estimand

For endpoint `e` and locked seed `s`:

```text
delta(e,s) = AUC_target_test(primary selective strategy, e, s)
             - AUC_target_test(dual global CORAL, e, s)
```

The primary selective strategy is D1 for IDH and D2 for IH. The endpoint estimand is the arithmetic mean of `delta(e,s)` across all five locked seeds. ROC AUC is calculated on target-test sessions, while sampling uncertainty respects patients.

IDH and IH are co-primary endpoints. Two-sided patient-cluster bootstrap p-values are adjusted with Holm at family-wise alpha 0.05. Strong cross-outcome mechanism-aware wording requires both co-primary contrasts to favor selective alignment after multiplicity control; one positive endpoint is reported as endpoint-specific evidence only.

### Primary uncertainty procedure

- 1,000 bootstrap replicates with seed `20260715`;
- resample target-test patients with replacement;
- include all sessions of each sampled patient, preserving multiplicity when a patient is sampled repeatedly;
- keep model, seed, and comparator predictions paired inside every replicate;
- calculate the per-seed AUC contrast and then average across locked seeds;
- report percentile 95% confidence intervals and Holm-adjusted co-primary p-values;
- report per-seed estimates and their SD descriptively, never as a replacement for patient uncertainty.

### Co-required falsification contrasts

The strong selective-alignment claim additionally requires:

1. primary selective strategy versus dual no alignment C;
2. the clinical partition to exceed the locked exact-size random-partition joint null;
3. no prespecified material degradation in PR AUC, Brier score, calibration intercept, or calibration slope;
4. outcome information to be preserved in locked representation probes; and
5. the effect not to depend on one seed, post-hoc lambda, or post-hoc regrouping.

The joint random-null statistic is the mean of the IDH and IH seed-averaged AUC improvements for the primary mapping. Its one-sided empirical p-value is:

```text
(1 + number of random joint statistics >= clinical joint statistic) / (1 + 19)
```

### Key secondary estimands

- target-updated no-alignment C minus source-only target ROC AUC;
- D1 versus D2 within each endpoint;
- endpoint-by-strategy interaction: `(D1-D2)_IDH - (D1-D2)_IH`;
- updated model minus target-local reference F, without default superiority wording;
- target-test PR AUC difference, Brier difference, calibration intercept/slope difference, and ECE difference;
- feature-group transportability as absolute `AUC_target - AUC_source`;
- normalized excess-AUC ratio `(AUC_target-0.5)/(AUC_source-0.5)` only as a secondary descriptive index when its denominator is stable.

Raw `AUC_target/AUC_source` is prohibited as the primary transportability metric. PR AUC is interpreted relative to endpoint prevalence.

No equivalence, non-inferiority, or “comparable” claim is allowed without a separately prespecified clinically justified margin.

### Calibration and utility

Report Brier score, calibration intercept, calibration slope, ECE, and calibration curves before and after the locked calibration procedure. Clinical thresholds cannot be chosen with test outcomes or by post-hoc Youden index.

DCA, sensitivity, specificity, PPV, NPV, alerts, true events captured, false alerts, missed events per 1,000 sessions, and number needed to review are conditional analyses. Until intended action and clinically justified thresholds are approved, thresholds of 5%, 10%, and 20% are sensitivity points only. Without a new temporal/external cohort, clinical utility remains exploratory.

### Subgroups

Subgroup inference uses patient-cluster bootstrap and an interaction test. Session-level bootstrap and within-subgroup significance comparisons are prohibited. Variables, functional forms/cut points, minimum patient counts, multiplicity, and missing-data handling must be locked before evaluation; continuous interaction is preferred to outcome-driven dichotomization.

## 10. Evidence and provenance checkpoints

Before data rebuild:

- raw hashes match this contract;
- pipeline tests pass;
- protocol version is unchanged or explicitly amended.

Before training—the following data-stage items are now satisfied:

- corrected V3 processed files are rebuilt without overwriting V1/V2;
- dataset, feature, preprocessing, and split manifests are populated and hashed;
- patient overlap and strictly-prior audits pass;
- machine-readable feature manifest matches this contract;
- seeds, primary branch mapping, and all random partitions are hashed.

Exact model capacity and optimizer/alignment settings are locked in `v3/experiments/training_protocol.yaml`. The V3 server run grid is generated at `v3/experiments/run_grid.csv` with 420 hashed configs under `v3/experiments/server_configs/`. The remaining execution gates are explicit `training_authorized=true` under PI review and a successful `bash scripts/run_v3_server.sh training-readiness`. No training may start until authorization is changed.

Before accepting a result:

- run ID, code/config/data/split hashes, environment, checkpoint, predictions, latent exports, and evaluation files exist;
- verification passes before atomic registry append;
- failed runs remain visible;
- no V1/V2 run ID enters the V3 registry.

Before manuscript use:

- every number maps to eligible V3 run IDs and hashed artifacts;
- null and adverse findings appear in the main results;
- representation analysis is labeled exploratory unless its prespecified gate passes;
- title follows Path A/B/C in `v3/go_no_go_criteria.md`.

## 11. Commands and execution boundary

Allowed now:

```bash
python -m pytest tests/test_data_pipeline.py -q
bash scripts/run_v3_server.sh scientific-audit
bash scripts/run_v3_server.sh training-readiness
bash scripts/run_v3_server.sh tests
bash scripts/run_v3_server.sh prepare-grid
bash scripts/run_v3_server.sh plan
bash scripts/run_v3_server.sh verify-grid
```

The following official cohort rebuild entry points were used for the completed V3 freeze. They must not be run again against the frozen paths; any future rebuild requires explicit authorization, a new protocol/dataset version, and new output paths:

```bash
python src/data_pipeline/HBD_data.py RAW_SHENYI.csv OUTPUT_SOURCE_V3.csv
python src/data_pipeline/HBD_data_fuding.py RAW_FUDING.csv OUTPUT_TARGET_V3.csv
```

### Server run grid

The locked V3 GPU matrix is materialised by `scripts/generate_v3_run_grid.py` and exposed through `bash scripts/run_v3_server.sh`.

| Asset | Path | Content |
|---|---|---|
| Run grid | `v3/experiments/run_grid.csv` | 420 planned runs with execution order |
| Config directory | `v3/experiments/server_configs/` | One immutable YAML per run |
| Manifest | `v3/experiments/run_grid_manifest.json` | Counts, hashes, evidence lock |

Grid composition:

- core models: `source_only`, `A`, `B`, `C`, `D1`, `D2`, `F`, `G` × 5 seeds = 40 runs;
- random controls: 76 locked partitions × 5 seeds = 380 runs;
- total = 420 runs.

Every config embeds the frozen evidence lock and sets `training_authorized=false`. The shell entrypoint accepts `prepare-grid`, `plan`, and `verify-grid`, and still rejects `train`.

Implementation notes required before any future authorization:

- dual-branch alignment scopes include `branch_a` and `branch_b`;
- frozen patient split and preprocessing are loaded from V3 manifests rather than recomputed;
- Model F is target-only; `source_only` skips target updating;
- random controls use the locked random feature partition as the dual-branch split and align `branch_a`.

Do not start training, regenerate or overwrite the frozen cohorts, replace historical files, upload data, or make manuscript claims without explicit authorization and passed gates.
