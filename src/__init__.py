"""Source package for the v6/v7 discrete-time survival pipeline.

Declared empty on purpose: the legacy v4/v5 re-exports (HFDataLoader etc.)
were removed during the systematic cleanup (2026-09-01) to avoid coupling the
live pipeline imports (``src.v6_acc``, ``src.evaluate.discrete_survival_metrics``,
``src.data.v5_*``, ``src.data_pipeline.data_process``) to deleted modules.
"""