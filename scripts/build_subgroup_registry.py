"""Write the prespecified subgroup result registry without fabricating estimates."""

from __future__ import annotations

import csv
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "conf" / "subgroup_protocol.yaml"
OUTPUT = ROOT / "subgroup_analysis_patient_bootstrap.csv"


def main() -> None:
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    rows = []
    for endpoint, definitions in protocol["endpoints"].items():
        for definition in definitions:
            rows.append({
                "endpoint": endpoint.upper(), "subgroup": definition["subgroup"],
                "candidate_feature": definition["candidate_feature"],
                "definition": definition["definition"], "subgroup_level": "",
                "n_patients": "", "n_sessions": "", "subgroup_auc": "", "delta_auc": "",
                "ci_lower": "", "ci_upper": "", "interaction_p_value": "",
                "bootstrap_unit": protocol["resampling"]["unit"],
                "bootstrap_replicates": protocol["resampling"]["replicates"],
                "status": protocol["status"],
            })
    fields = list(rows[0])
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote subgroup evidence-gate registry to {OUTPUT}")


if __name__ == "__main__":
    main()
