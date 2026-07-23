# V3 Model Family Specification

Status: locked for V3 execution planning. Architecture dimensions, parameter targets, optimizer, fixed CORAL lambda, seeds, and the 420-run server grid are frozen. Training authorization remains withheld.

## Scientific target

The model experiment asks whether knowing what to align adds value. It does not test whether CORAL is generally effective. CORAL is used as a simple fixed alignment operator so that feature-partition knowledge is the principal experimental variable.

For an aligned representation `Z_selected`, the candidate objective is:

`L = L_prediction + lambda * L_CORAL(Z_selected_source, Z_selected_target)`

`lambda` must be selected using development/validation information under one common rule. It cannot be retuned separately after observing V3 test results.

## Required models

| ID | Model | Role |
|---|---|---|
| A | Single encoder, no alignment | Updating control for Model B |
| B | Single encoder, global CORAL | Single-encoder global-alignment comparison |
| C | Dual encoder, no alignment | Architecture-matched control for D/E/G |
| D1 | Dual encoder, physiology-partition CORAL | Selective partition hypothesis, evaluated for both outcomes |
| D2 | Dual encoder, treatment/context-partition CORAL | Selective partition hypothesis, evaluated for both outcomes |
| E | Dual encoder, multiple size-matched random-partition CORAL controls | Falsification distribution |
| F | Target-only single encoder | Local reference; not assumed to be an upper bound |
| G | Dual encoder, global CORAL | Required fair global-alignment comparator for D |

Model G is required because comparing single-encoder Model B directly with dual-encoder Model D would confound architecture and alignment. Model F is called a local reference because limited target sample size means target-only training is not guaranteed to be an oracle or upper bound.

A source-only model is retained as a deployment baseline but is not a substitute for Model A, which receives the same supervised target updating as the other alignment models.

## Capacity matching

“Same backbone capacity” is operationalized before execution:

- A and B have exactly the same architecture and parameter count.
- C, D1, D2, E, and G have exactly the same architecture and parameter count.
- Single- and dual-encoder families must target a prespecified total trainable-parameter tolerance, proposed at no more than 5%, or report an additional capacity-matched sensitivity analysis.
- Prediction heads, activation functions, normalization, dropout, initialization family, and stopping rules are held fixed within valid contrasts.

## Training matching

Models A-E and G use the same source pretraining and source-plus-target updating population, batch-sampling rule, optimizer, learning-rate schedule, epoch budget, early-stopping cohort, and locked seeds. Only the registered alignment strategy changes within an architecture-matched contrast.

Model F uses only target-update labels and the same target validation/calibration information allowed to the other models. Its different training population is its intended scientific distinction.

## Feature partitions

The feature ontology is clinically reviewed but uncertain. No feature is labeled invariant. The primary ontology and at least one plausible alternative grouping are locked before training.

Both D1 and D2 are evaluated for both IDH and IH. The protocol does not encode the historical IDH-physiology or IH-treatment mapping as truth.

Random controls must:

- match the aligned feature count of each clinical partition;
- preserve any prespecified categorical/continuous constraints if necessary;
- use a locked set of random-partition seeds independent of model seeds; and
- be reported as an empirical null distribution, not as one selected random run.

## Alignment and selection gates

- CORAL is calculated using development-eligible source and target-update representations only.
- Target held-out labels and representations are inaccessible to model selection.
- Lambda, branch widths, and preprocessing are common across relevant contrasts.
- Model selection uses the same metric and validation cohort for all models.
- All trained seeds and random partitions enter the registry, including failures.

## Adaptive alignment

Complex adaptive alignment is outside the initial V3 confirmatory family. It may be proposed only after the fixed CORAL experiment and under a separately versioned protocol. It cannot be introduced to rescue a negative fixed-alignment result.
