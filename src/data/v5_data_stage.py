"""V5 contract_3 data assets."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.v5_budget_subsets import build_nested_budget_manifest
from src.data_pipeline.data_process import build_v4_restart_survival_cohort

FEATURES = [
    "年龄", "性别", "透析龄年", "透前收缩压", "透前舒张压", "透前体重",
    "透前体重-干体重", "历史平均透前体重", "历史平均透前收缩压",
    "历史平均透前舒张压", "history_prior_session_count",
    "history_IDH_240_interval_1_rate", "history_IDH_240_interval_2_rate",
    "history_IDH_240_interval_3_rate", "history_IDH_240_interval_4_rate",
    "history_IH_240_interval_1_rate", "history_IH_240_interval_2_rate",
    "history_IH_240_interval_3_rate", "history_IH_240_interval_4_rate",
]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strata(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.groupby("患者id", sort=True).agg(
        idh=("idh_event_observed_240", "max"),
        ih=("ih_event_observed_240", "max"),
    ).reset_index()
    result["stratum"] = result["idh"].astype(int).astype(str) + result["ih"].astype(int).astype(str)
    return result


def build_patient_split_manifest(source: pd.DataFrame, target: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    specs = (
        (source, "source", (("source_train", 0.85), ("source_validation", 0.15))),
        (target, "target", (("target_update", 0.70), ("target_calibration", 0.10), ("target_test", 0.20))),
    )
    for child, (frame, center, roles) in zip(np.random.SeedSequence(seed).spawn(2), specs):
        rng = np.random.default_rng(int(child.generate_state(1, dtype=np.uint32)[0]))
        assigned: dict[str, str] = {}
        for _, group in _strata(frame).groupby("stratum", sort=True):
            patients = group["患者id"].astype(str).to_numpy(copy=True)
            rng.shuffle(patients)
            counts = np.floor(len(patients) * np.array([fraction for _, fraction in roles])).astype(int)
            counts[: len(patients) - counts.sum()] += 1
            start = 0
            for (role, _), count in zip(roles, counts):
                for patient in patients[start : start + count]:
                    assigned[patient] = role
                start += count
        rows.extend(
            {
                "session_id": session_id,
                "患者id": patient,
                "center": center,
                "split_role": assigned[str(patient)],
            }
            for session_id, patient in frame[["session_id", "患者id"]].itertuples(index=False, name=None)
        )
    result = pd.DataFrame(rows).sort_values(["center", "split_role", "患者id"], kind="mergesort").reset_index(drop=True)
    roles = result.groupby(["center", "患者id"], sort=False)["split_role"].nunique()
    if roles.gt(1).any() or result["session_id"].duplicated().any():
        raise ValueError("患者被分配到多个split role或session重复")
    return result


def _hide_target_history(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    history_columns = [column for column in result.columns if column.startswith("history_") or column.startswith("历史平均")]
    result[history_columns] = np.nan
    return result


def _add_feature_aliases(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    counts = [
        pd.to_numeric(result[f"history_{endpoint}_240_prior_session_count"], errors="coerce")
        for endpoint in ("IDH", "IH")
    ]
    if not np.allclose(counts[0].to_numpy(), counts[1].to_numpy(), equal_nan=True):
        raise ValueError("IDH/IH prior session count 不一致")
    result["history_prior_session_count"] = counts[0]
    return result


def fit_preprocessor(frame: pd.DataFrame, split: pd.DataFrame) -> dict[str, Any]:
    merged = frame.merge(split[["session_id", "split_role"]], on="session_id", validate="one_to_one")
    train = merged.loc[merged["split_role"].eq("source_train"), FEATURES].copy()
    train["history_prior_session_count"] = np.log1p(pd.to_numeric(train["history_prior_session_count"], errors="coerce"))
    numeric = train.apply(pd.to_numeric, errors="coerce")
    medians = numeric.median()
    if medians.isna().any():
        raise ValueError(f"source_train全缺失特征: {medians[medians.isna()].index.tolist()}")
    imputed = numeric.fillna(medians)
    scales = imputed.std(ddof=0)
    if (scales == 0).any():
        raise ValueError(f"source_train常数特征: {scales[scales == 0].index.tolist()}")
    return {
        "protocol": "v5_source_train_only_preprocessing",
        "features": FEATURES,
        "log1p": ["history_prior_session_count"],
        "fit_role": "source_train",
        "medians": {key: float(value) for key, value in medians.items()},
        "means": {key: float(value) for key, value in imputed.mean().items()},
        "scales_population_sd": {key: float(value) for key, value in scales.items()},
    }


def _write_json(path: Path, payload: Any) -> str:
    if path.exists():
        raise FileExistsError(f"拒绝覆盖数据资产: {path}")
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return sha256_file(path)


def _write_csv(path: Path, frame: pd.DataFrame) -> str:
    if path.exists():
        raise FileExistsError(f"拒绝覆盖数据资产: {path}")
    frame.to_csv(path, index=False, encoding="utf-8")
    return sha256_file(path)


def build_scientific_assets(
    source_raw: str | Path,
    target_raw: str | Path,
    data_root: str | Path,
    *,
    split_seed: int,
    budget_seed: int,
    budget_repeats: int,
    budgets: tuple[float, ...],
) -> dict[str, dict[str, str]]:
    root = Path(data_root)
    if root.exists():
        raise FileExistsError(f"拒绝覆盖数据根目录: {root}")
    root.mkdir(parents=True)
    source, source_audit = build_v4_restart_survival_cohort(source_raw, "shenyi")
    target, target_audit = build_v4_restart_survival_cohort(target_raw, "fuding")
    source = _add_feature_aliases(source)
    target = _add_feature_aliases(target)
    split = build_patient_split_manifest(source, target, split_seed)
    budget = build_nested_budget_manifest(split, budget_seed, budget_repeats, budgets)
    hidden_target = _hide_target_history(target)
    preprocessing = fit_preprocessor(source, split)
    primary = {"branch_a": ["透前体重-干体重"], "branch_b": [feature for feature in FEATURES if feature != "透前体重-干体重"]}
    random_rows = []
    for index, child in enumerate(np.random.SeedSequence(budget_seed).spawn(19), start=1):
        order = np.array(FEATURES, dtype=object)
        np.random.default_rng(int(child.generate_state(1, dtype=np.uint32)[0])).shuffle(order)
        random_rows.append({"partition_id": f"random_{index:02d}", "branch_a": order[0], "branch_b": "|".join(sorted(order[1:].tolist()))})
    assets = {
        "source_cohort": {"path": str(root / "source_cohort.csv"), "sha256": _write_csv(root / "source_cohort.csv", source)},
        "target_cohort": {"path": str(root / "target_cohort.csv"), "sha256": _write_csv(root / "target_cohort.csv", hidden_target)},
        "split_manifest": {"path": str(root / "split_manifest.csv"), "sha256": _write_csv(root / "split_manifest.csv", split)},
        "preprocessing": {"path": str(root / "preprocessing.json"), "sha256": _write_json(root / "preprocessing.json", preprocessing)},
        "primary_partition": {"path": str(root / "primary_partition.json"), "sha256": _write_json(root / "primary_partition.json", primary)},
        "random_partitions": {"path": str(root / "random_partitions.csv"), "sha256": _write_csv(root / "random_partitions.csv", pd.DataFrame(random_rows))},
        "budget_subsets": {"path": str(root / "budget_subsets.csv"), "sha256": _write_csv(root / "budget_subsets.csv", budget)},
    }
    _write_json(root / "source_audit.json", source_audit)
    _write_json(root / "target_audit.json", target_audit)
    return assets
