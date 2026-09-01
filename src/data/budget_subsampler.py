"""V4 label efficiency 目标域 budget 子采样模块。

方案甲：每个 budget 固定一组子样本，不估计子采样方差。
子采样单元为患者，按双终点存在性分层。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def derive_budget_subsample_seeds(
    root_seed: int,
    budgets: list[float],
) -> dict[float, int]:
    """从根种子派生每个 budget 的子采样种子。

    index 0-7 沿用 V4 restart（split/random/bootstrap/5 optimization），
    index 8+ 为 budget 子采样种子，不与现有种子冲突。
    """
    children = np.random.SeedSequence(root_seed).spawn(8 + len(budgets))
    return {
        b: int(children[8 + i].generate_state(1, dtype=np.uint32)[0])
        for i, b in enumerate(budgets)
    }


def subsample_target_update_patients(
    split_manifest: pd.DataFrame,
    target_cohort: pd.DataFrame,
    budget: float,
    seed: int,
) -> set[str]:
    """按患者双终点存在性分层，从 target_update 患者中抽取 budget 比例。

    返回被选中的患者 ID 集合（字符串）。
    """
    tu_sessions = split_manifest[
        (split_manifest["center"] == "target")
        & (split_manifest["split_role"] == "target_update")
    ]
    patient_endpoints = (
        target_cohort[["session_id", "患者id", "idh_event_observed_240", "ih_event_observed_240"]]
        .merge(tu_sessions[["session_id"]], on="session_id", how="inner")
        .groupby("患者id")[["idh_event_observed_240", "ih_event_observed_240"]]
        .max()
    )
    strata = patient_endpoints.astype(str).agg("".join, axis=1)

    rng = np.random.default_rng(seed)
    selected: list[str] = []
    for _stratum_label, patient_ids in strata.groupby(strata).groups.items():
        ids = np.asarray(sorted(map(str, patient_ids)), dtype=object)
        rng.shuffle(ids)
        n = int(round(budget * len(ids)))
        # budget > 0 时保证每个 stratum 至少抽 1 个（若 stratum 非空）
        if budget > 0 and len(ids) > 0:
            n = max(n, 1)
        selected.extend(ids[:n].tolist())

    return set(selected)


def build_budget_manifests(
    split_manifest: pd.DataFrame,
    target_cohort: pd.DataFrame,
    budgets: list[float],
    root_seed: int,
    output_path: Path,
) -> pd.DataFrame:
    """为每个 budget 生成子采样 manifest 并原子写入 CSV。

    输出列：budget, subsample_seed, 患者id
    """
    budget_seeds = derive_budget_subsample_seeds(root_seed, budgets)
    rows = []
    for budget in budgets:
        seed = budget_seeds[budget]
        selected = subsample_target_update_patients(
            split_manifest, target_cohort, budget, seed
        )
        for patient_id in sorted(selected):
            rows.append({
                "budget": budget,
                "subsample_seed": seed,
                "患者id": patient_id,
            })
    manifest = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False, encoding="utf-8")
    return manifest


def filter_target_update_by_budget(
    frame: pd.DataFrame,
    budget_manifest: pd.DataFrame,
    budget: float,
) -> pd.DataFrame:
    """过滤 target_update sessions，只保留 budget 选中的患者。

    source_train / source_validation / target_calibration / target_test 不变。
    """
    selected_patients = set(
        budget_manifest.loc[budget_manifest["budget"] == budget, "患者id"].astype(str)
    )
    is_target_update = frame["split_role"].eq("target_update")
    is_selected = frame["患者id"].astype(str).isin(selected_patients)
    keep = (~is_target_update) | (is_target_update & is_selected)
    return frame.loc[keep].reset_index(drop=True)
