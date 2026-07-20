"""Build and validate paragraph-level manuscript claim provenance."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "conf" / "manuscript_claims.yaml"
DEFAULT_CSV = ROOT / "experiments" / "evidence_registry" / "manuscript_claim_registry.csv"
DEFAULT_JSON = ROOT / "experiments" / "evidence_registry" / "manuscript_claim_registry_status.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _paragraphs(text: str) -> list[str]:
    return [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]


def _quantitative_paragraphs(text: str) -> list[str]:
    excluded_prefixes = ("#", "**", "- Table", "- Figure", "- Supplementary")
    return [
        paragraph
        for paragraph in _paragraphs(text)
        if re.search(r"(?<![A-Za-z])\d", paragraph)
        and not paragraph.startswith(excluded_prefixes)
        and "[BLOCKER:" not in paragraph
    ]


def build(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manuscript_path = ROOT / config["manuscript"]
    manuscript = manuscript_path.read_text(encoding="utf-8")
    paragraphs = _paragraphs(manuscript)
    quantitative = _quantitative_paragraphs(manuscript)
    registry = pd.read_csv(ROOT / "experiments" / "evidence_registry" / "run_registry.csv")
    registered_run_ids = set(registry["run_id"].astype(str))
    rows = []
    errors = []
    paragraph_claim_counts = {paragraph: 0 for paragraph in quantitative}
    seen_claim_ids: set[str] = set()
    for claim in config["claims"]:
        claim_id = str(claim["claim_id"])
        locator = str(claim["locator"])
        if claim_id in seen_claim_ids:
            errors.append(f"duplicate_claim_id:{claim_id}")
        seen_claim_ids.add(claim_id)
        matches = [paragraph for paragraph in paragraphs if locator in paragraph]
        if len(matches) != 1:
            errors.append(f"locator_match_count:{claim_id}:{len(matches)}")
            continue
        paragraph = matches[0]
        if paragraph in paragraph_claim_counts:
            paragraph_claim_counts[paragraph] += 1
        run_ids = [str(run_id) for run_id in claim.get("run_ids", [])]
        unknown_runs = sorted(set(run_ids) - registered_run_ids)
        if unknown_runs:
            errors.append(f"unregistered_run_ids:{claim_id}:{','.join(unknown_runs)}")
        artifact_records = []
        for artifact in claim.get("artifacts", []):
            path = ROOT / artifact
            if not path.exists():
                errors.append(f"missing_artifact:{claim_id}:{artifact}")
                continue
            artifact_records.append({"path": str(artifact), "sha256": _sha256(path)})
        if not artifact_records:
            errors.append(f"no_artifacts:{claim_id}")
        rows.append({
            "claim_id": claim_id,
            "evidence_tier": str(claim["evidence_tier"]),
            "locator": locator,
            "manuscript_paragraph_sha256": hashlib.sha256(paragraph.encode("utf-8")).hexdigest(),
            "run_ids": ";".join(run_ids),
            "artifacts": json.dumps(artifact_records, ensure_ascii=False, sort_keys=True),
            "status": "mapped" if not unknown_runs and artifact_records else "invalid",
        })
    unmapped = [paragraph for paragraph, count in paragraph_claim_counts.items() if count == 0]
    multiply_mapped = [paragraph for paragraph, count in paragraph_claim_counts.items() if count > 1]
    errors.extend(
        f"unmapped_quantitative_paragraph:{hashlib.sha256(paragraph.encode()).hexdigest()[:12]}"
        for paragraph in unmapped
    )
    errors.extend(
        f"multiply_mapped_quantitative_paragraph:{hashlib.sha256(paragraph.encode()).hexdigest()[:12]}"
        for paragraph in multiply_mapped
    )
    return {
        "registry_version": config["registry_version"],
        "manuscript_path": config["manuscript"],
        "manuscript_sha256": _sha256(manuscript_path),
        "quantitative_paragraphs": len(quantitative),
        "mapped_claims": len(rows),
        "unmapped_quantitative_paragraphs": len(unmapped),
        "multiply_mapped_quantitative_paragraphs": len(multiply_mapped),
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "unmapped_previews": [paragraph.replace("\n", " ")[:180] for paragraph in unmapped],
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    report = build(args.config.resolve())
    csv_output, json_output = args.csv_output.resolve(), args.json_output.resolve()
    fields = ["claim_id", "evidence_tier", "locator", "manuscript_paragraph_sha256", "run_ids", "artifacts", "status"]
    with csv_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(report["rows"])
    status = {key: value for key, value in report.items() if key != "rows"}
    json_output.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    if args.require_complete and report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
