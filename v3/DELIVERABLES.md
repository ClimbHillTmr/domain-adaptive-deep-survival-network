# V3 Deliverables

| Requested deliverable | Artifact | Status |
|---|---|---|
| 1. Research protocol | `research_protocol.md` | Drafted; approval and data lock pending |
| 2. Data contract | `data_contract/` and `preprocessing/` | Candidate bytes verified; clinical freeze blocked |
| 3. Experiment registry | `experiments/experiment_matrix.csv` and `registry/run_registry.csv` | Planned; run registry intentionally empty |
| 4. Model specification | `models/model_specification.md` | Models A-G specified; dimensions and hyperparameters pending lock |
| 5. Statistical analysis plan | `evaluation/statistical_analysis_plan.md` | Drafted; endpoint/multiplicity/power approval pending |
| 6. Evidence matrix | `registry/evidence_matrix.csv` | All V3 claims not tested or blocked |
| 7. Manuscript outline | `manuscript/outline.md` | Conditional Paths A-C defined |
| 8. Go/no-go criteria | `go_no_go_criteria.md` | Defined before training |
| Scientific evidence tree | `evidence_tree.md` and `evidence_tree.yaml` | Complete; 19 nodes across Evidence 1-5 |
| Evidence-first experiment map | `experiments/experiment_matrix.csv` | Complete; models subordinated to evidence nodes |
| Random-control protocol | `experiments/random_control_plan.yaml` | Four families drafted; seeds/counts/hash lock pending |
| Transportability ranking protocol | `experiments/transportability_group_plan.csv` | Drafted; feature-quality and semantic locks pending |
| Scientific audit | `scientific_audit.yaml` and `../scripts/v3_scientific_audit.py` | Structure passes; training readiness blocked |
| Server execution design | `experiments/server_execution_plan.md` and `../scripts/run_v3_server.sh` | Audit-only launcher active; training implementation deliberately withheld |

No V3 model result exists. V1/V2 remain historical hypothesis-generating observations only.

Current machine-readable status is written to `registry/scientific_audit_status.json` by the audit-only server entry point. A successful structural audit must not be interpreted as training authorization.
