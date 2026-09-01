"""Data staging package (v5 reuse layer for the v6 pipeline).

Declared empty on purpose: the legacy ``huggingface_loader`` re-export was
removed during the systematic cleanup (2026-09-01). Live modules here are
``v5_data_stage.py`` and ``v5_budget_subsets.py``, imported by name.
"""