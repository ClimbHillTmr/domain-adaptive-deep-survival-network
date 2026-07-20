"""Export registered latent and SHAP diagnostics with exploratory labels."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "experiments" / "final_results" / "main_mechanism_aware"
OUTPUT = ROOT / "experiments" / "evidence_registry"
RUN_ID = "binary_20260719_044949_c6c955a30fc2"


def _write(name: str, rows: list[dict]) -> None:
    fields = list(rows[0])
    with (OUTPUT / name).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    registered = (OUTPUT / "run_registry.csv").read_text(encoding="utf-8")
    if RUN_ID not in registered:
        raise ValueError("Exploratory evidence run_id is not registered")
    latent = json.loads((MAIN / "latent_analysis.json").read_text(encoding="utf-8"))
    shap = json.loads((MAIN / "shap_analysis_full.json").read_text(encoding="utf-8"))
    latent_rows = []
    for endpoint in ("idh", "ih"):
        for metric in ("mmd_rbf", "mmd_linear", "covariance_distance", "wasserstein", "domain_classifier_auc"):
            latent_rows.append({
                "run_id": RUN_ID, "endpoint": endpoint.upper(), "metric": metric,
                "before": latent[endpoint]["before_alignment"][metric],
                "after": latent[endpoint]["after_alignment"][metric],
                "delta_after_minus_before": latent[endpoint]["delta"][metric],
                "evidence_tier": "exploratory", "claim_limit": "does_not_establish_domain_invariance",
            })
    shap_rows = []
    for endpoint in ("idh", "ih"):
        before = shap[endpoint]["before_alignment"]
        after = shap[endpoint]["after_alignment"]
        shap_rows.append({
            "run_id": RUN_ID, "endpoint": endpoint.upper(),
            "target_physiology_ratio_before": before["target"]["physiology_ratio"],
            "target_physiology_ratio_after": after["target"]["physiology_ratio"],
            "target_treatment_ratio_before": before["target"]["treatment_ratio"],
            "target_treatment_ratio_after": after["target"]["treatment_ratio"],
            "cross_center_spearman_before": before["cross_domain_correlation"]["spearman_r"],
            "cross_center_spearman_after": after["cross_domain_correlation"]["spearman_r"],
            "n_source_shap": before["source"]["n_samples"], "n_target_shap": before["target"]["n_samples"],
            "evidence_tier": "exploratory", "claim_limit": "model_attribution_not_causal_mechanism",
        })
    _write("exploratory_latent_metrics.csv", latent_rows)
    _write("exploratory_shap_summary.csv", shap_rows)
    sources = {
        "latent_analysis": MAIN / "latent_analysis.json",
        "shap_analysis": MAIN / "shap_analysis_full.json",
    }
    outputs = {
        "latent_table": OUTPUT / "exploratory_latent_metrics.csv",
        "shap_table": OUTPUT / "exploratory_shap_summary.csv",
    }
    provenance = {
        "run_id": RUN_ID,
        "evidence_tier": "exploratory",
        "sources": {
            name: {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path)}
            for name, path in sources.items()
        },
        "outputs": {
            name: {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path)}
            for name, path in outputs.items()
        },
        "claim_limits": ["does_not_establish_domain_invariance", "model_attribution_not_causal_mechanism"],
    }
    (OUTPUT / "exploratory_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Wrote registered exploratory evidence tables")


if __name__ == "__main__":
    main()
