# V3 Interpretation Policy

## Permitted concepts

- observed cross-center distribution differences
- clinically motivated feature partitions
- supervised target-center updating
- incremental contribution of a prespecified alignment strategy
- predictive preservation and calibration under held-out evaluation

## Claims requiring stronger evidence

The following cannot be inferred from feature distances or two-center latent analyses alone:

- recovered physiological mechanisms
- center-specific causal treatment mechanisms
- removal of nuisance variation
- preservation of all clinically meaningful context
- domain-invariant representation
- general transportability across hospitals

## Falsification rules

The mechanism-aware hypothesis is weakened when:

- clinical partitions do not outperform a distribution of size-matched random partitions;
- selective alignment does not improve on architecture-matched no alignment;
- discrepancy metrics improve while predictive performance or calibration worsens;
- the favored partition changes across seeds without a prespecified explanation; or
- conclusions depend on target-test-driven feature grouping or threshold selection.

No single condition automatically proves the entire conceptual framework false. The manuscript must state which operational hypothesis was tested and which failed.

## Outcome-specific interpretation

SHAP, linear probes, and branch ablations can describe model reliance. They do not establish that IDH is physiologically caused or IH is treatment-policy caused. Outcome-specific claims require consistent predictive contrasts, uncertainty, and cross-center or temporal replication.
