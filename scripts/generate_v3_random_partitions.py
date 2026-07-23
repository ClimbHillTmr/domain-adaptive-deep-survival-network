"""Generate locked V3 random feature-partition controls without outcome access."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURE_ALLOWLIST = ROOT / "v3/data_contract/feature_allowlist.csv"
CLINICAL_PARTITION = ROOT / "v3/experiments/primary_feature_partition.json"
SOURCE_DATA = ROOT / "data/v3/source_v3_hbd.csv"
SPLIT_MANIFEST = ROOT / "v3/data_contract/split_manifest.csv"
OUTPUT = ROOT / "v3/experiments/random_partitions.csv"
MASTER_SEED = 20260715
N_PARTITIONS = 19


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strata(source_train: pd.DataFrame, features: list[str]) -> dict[str, str]:
    missing = source_train[features].isna().mean()
    variances = source_train[features].apply(pd.to_numeric, errors="coerce").var(ddof=0)
    log_variance = np.log10(variances.clip(lower=np.finfo(float).tiny))
    missing_rank = missing.rank(method="first")
    variance_rank = log_variance.rank(method="first")
    missing_bin = pd.qcut(missing_rank, q=4, labels=False)
    variance_bin = pd.qcut(variance_rank, q=4, labels=False)
    return {
        feature: f"m{int(missing_bin[feature])}_v{int(variance_bin[feature])}"
        for feature in features
    }


def _sample_unique(rng: np.random.Generator, candidates: list[tuple[str, ...]]) -> list[tuple[str, ...]]:
    if len(candidates) < N_PARTITIONS:
        raise ValueError(f"Only {len(candidates)} eligible partitions; {N_PARTITIONS} required.")
    indices = rng.choice(len(candidates), size=N_PARTITIONS, replace=False)
    return [candidates[int(index)] for index in indices]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("Refusing to write partitions without --execute.")
    if OUTPUT.exists():
        raise SystemExit(f"Refusing to overwrite locked partition artifact: {OUTPUT}")

    table = pd.read_csv(FEATURE_ALLOWLIST)
    features = table.loc[table["allowed"].eq("yes"), "feature"].tolist()
    group_by_feature = dict(
        zip(
            table.loc[table["allowed"].eq("yes"), "feature"],
            table.loc[table["allowed"].eq("yes"), "primary_group"],
            strict=True,
        )
    )
    clinical = json.loads(CLINICAL_PARTITION.read_text(encoding="utf-8"))
    clinical_treatment = tuple(clinical["treatment_context"])
    if set(features) != set(clinical["physiology_history"]) | set(clinical_treatment):
        raise ValueError("Clinical partition does not cover the locked feature list exactly.")

    split = pd.read_csv(SPLIT_MANIFEST, dtype={"patient_id": str})
    source_train_patients = set(
        split.loc[split["split_role"].eq("source_train"), "patient_id"].astype(str)
    )
    source = pd.read_csv(SOURCE_DATA, usecols=["患者id", *features], low_memory=False)
    source["患者id"] = source["患者id"].astype(str)
    source_train = source.loc[source["患者id"].isin(source_train_patients)]
    if source_train["患者id"].nunique() != len(source_train_patients):
        raise ValueError("Source-training patient set could not be reconstructed.")

    all_size_four = list(itertools.combinations(features, len(clinical_treatment)))
    clinical_set = frozenset(clinical_treatment)
    all_size_four = [item for item in all_size_four if frozenset(item) != clinical_set]
    strata = _strata(source_train, features)
    clinical_strata = sorted(strata[feature] for feature in clinical_treatment)
    variance_candidates = [
        item for item in all_size_four if sorted(strata[feature] for feature in item) == clinical_strata
    ]
    current_features = [
        feature
        for feature in features
        if not feature.startswith("历史") and not feature.startswith("history_")
    ]
    history_features = [feature for feature in features if feature not in current_features]
    clinical_current_count = sum(feature in current_features for feature in clinical_treatment)
    clinical_history_count = len(clinical_treatment) - clinical_current_count
    type_candidates = [
        tuple([*current, *history])
        for current in itertools.combinations(current_features, clinical_current_count)
        for history in itertools.combinations(history_features, clinical_history_count)
        if frozenset([*current, *history]) != clinical_set
    ]

    rng = np.random.default_rng(MASTER_SEED)
    family_partitions: dict[str, list[tuple[str, ...]]] = {
        "A_unrestricted_random": [],
        "B_variance_matched": _sample_unique(rng, variance_candidates),
        "C_group_size_matched": _sample_unique(rng, all_size_four),
        "D_clinical_type_matched": _sample_unique(rng, type_candidates),
    }
    seen_unrestricted: set[frozenset[str]] = set()
    while len(family_partitions["A_unrestricted_random"]) < N_PARTITIONS:
        size = int(rng.integers(1, len(features)))
        selected = tuple(sorted(rng.choice(features, size=size, replace=False).tolist()))
        key = frozenset(selected)
        if key == clinical_set or key in seen_unrestricted:
            continue
        seen_unrestricted.add(key)
        family_partitions["A_unrestricted_random"].append(selected)

    rows: list[dict[str, object]] = []
    for family, partitions in family_partitions.items():
        for index, aligned in enumerate(partitions, start=1):
            aligned_set = set(aligned)
            preserved = [feature for feature in features if feature not in aligned_set]
            rows.append(
                {
                    "partition_id": f"{family}-{index:02d}",
                    "family": family,
                    "partition_index": index,
                    "master_seed": MASTER_SEED,
                    "aligned_feature_count": len(aligned),
                    "aligned_features": json.dumps(list(aligned), ensure_ascii=False),
                    "preserved_features": json.dumps(preserved, ensure_ascii=False),
                    "feature_allowlist_sha256": _sha256(FEATURE_ALLOWLIST),
                    "clinical_partition_sha256": _sha256(CLINICAL_PARTITION),
                    "generation_status": "locked_before_v3_model_results",
                }
            )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, OUTPUT)
    report = {
        "output": str(OUTPUT.relative_to(ROOT)),
        "sha256": _sha256(OUTPUT),
        "families": {family: len(partitions) for family, partitions in family_partitions.items()},
        "variance_matched_candidate_count": len(variance_candidates),
        "clinical_type_candidate_count": len(type_candidates),
        "feature_allowlist_sha256": _sha256(FEATURE_ALLOWLIST),
        "clinical_partition_sha256": _sha256(CLINICAL_PARTITION),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
