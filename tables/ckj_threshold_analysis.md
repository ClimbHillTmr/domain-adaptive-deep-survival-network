# CKJ Translational Addendum

## Threshold Package
Pre-specified thresholds were set at 10%, 20%, and 30% to represent escalating use cases: intensified monitoring, ultrafiltration review, and bedside reassessment. These thresholds are intended for silent-mode decision support rather than automated intervention.

## Key Comparison
- `CDAN-GSN (current locked run)`: C-index 0.830 (0.818-0.840); Brier 60m 0.1002, 120m 0.1330.
- `Local Cox update`: C-index 0.842 (0.831-0.850); Brier 60m 0.0951, 120m 0.1274.

## Reviewer-facing Interpretation
- The simple local baseline is not inferior to the current deep model on held-out testing. CKJ-facing claims should therefore emphasize the value of local updating and workflow-aware deployment, not algorithmic novelty.
- `no_cdan` ablation is not an adequate clinical comparator because it remains an internal architectural variant rather than an independently deployable local baseline.

## Threshold Snapshot
- `60 min`
  CDAN-GSN (current locked run) @ 0.10: flagged 40.6%, captured 85.6% of events, workload/event 3.24, delta net benefit vs treat-all 0.0425.
  CDAN-GSN (current locked run) @ 0.20: flagged 20.7%, captured 58.9% of events, workload/event 2.40, delta net benefit vs treat-all 0.1230.
  CDAN-GSN (current locked run) @ 0.30: flagged 12.7%, captured 41.9% of events, workload/event 2.08, delta net benefit vs treat-all 0.2524.
  Local Cox update @ 0.10: flagged 37.2%, captured 86.0% of events, workload/event 2.95, delta net benefit vs treat-all 0.0470.
  Local Cox update @ 0.20: flagged 19.4%, captured 61.7% of events, workload/event 2.15, delta net benefit vs treat-all 0.1313.
  Local Cox update @ 0.30: flagged 11.3%, captured 42.3% of events, workload/event 1.82, delta net benefit vs treat-all 0.2595.
- `120 min`
  CDAN-GSN (current locked run) @ 0.10: flagged 62.0%, captured 95.8% of events, workload/event 2.57, delta net benefit vs treat-all 0.0305.
  CDAN-GSN (current locked run) @ 0.20: flagged 39.8%, captured 80.8% of events, workload/event 1.96, delta net benefit vs treat-all 0.0900.
  CDAN-GSN (current locked run) @ 0.30: flagged 26.3%, captured 62.5% of events, workload/event 1.67, delta net benefit vs treat-all 0.1807.
  Local Cox update @ 0.10: flagged 60.0%, captured 95.2% of events, workload/event 2.50, delta net benefit vs treat-all 0.0311.
  Local Cox update @ 0.20: flagged 37.4%, captured 80.6% of events, workload/event 1.84, delta net benefit vs treat-all 0.0954.
  Local Cox update @ 0.30: flagged 25.6%, captured 64.6% of events, workload/event 1.57, delta net benefit vs treat-all 0.1914.
