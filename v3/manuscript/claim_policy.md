# V3 Manuscript Claim Policy

## Eligibility

A quantitative result may enter the V3 manuscript only if:

1. its run is present in `v3/registry/run_registry.csv`;
2. all required artifact hashes are populated and verified;
3. the run uses the locked V3 data, feature, preprocessing, and split contracts;
4. its comparison was declared in the V3 experiment matrix or clearly labeled exploratory; and
5. patient-cluster uncertainty is reported where applicable.

Historical V1/V2 results may appear only in background or protocol-rationale sections labeled historical exploratory evidence. They cannot be pooled with V3 estimates.

## Claim states

- `not_tested`: no eligible V3 evidence
- `supported`: prespecified V3 evidence supports the wording
- `not_supported`: the prespecified test did not support the claim
- `exploratory`: evidence exists but is not confirmatory
- `blocked`: required evidence or governance is unavailable

## Title gate

The title `Mechanism-Aware Transportable Representation Learning for Clinical AI under Heterogeneous Domain Shift` is eligible only if V3 shows reproducible incremental benefit of the prespecified clinical partition over no alignment, global alignment, and size-matched random controls, together with predictive preservation and appropriate calibration.

If that gate fails, the title and abstract must instead emphasize target-center updating and the evaluation of selective alignment.

## Required negative reporting

Null or unfavorable primary contrasts, inconsistent seeds, calibration harm, and failed falsification tests must appear in the main Results. They must not be moved solely to the supplement.
