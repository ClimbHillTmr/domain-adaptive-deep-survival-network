"""V5 outcome-blind、患者级、重复嵌套预算子集。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


BUDGETS = (0.10, 0.25, 0.50, 1.00)


def derive_repeat_seeds(root_seed: int, repeats: int) -> dict[int, int]:
    if repeats < 1:
        raise ValueError("repeats 必须为正数")
    children = np.random.SeedSequence(root_seed).spawn(repeats)
    return {
        repeat: int(child.generate_state(1, dtype=np.uint32)[0])
        for repeat, child in enumerate(children, start=1)
    }


def build_nested_budget_manifest(
    split_manifest: pd.DataFrame,
    root_seed: int,
    repeats: int,
    budgets: Iterable[float] = BUDGETS,
) -> pd.DataFrame:
    """仅按患者ID排序/随机抽样；绝不读取结局列，因此天然 outcome-blind。

    预算分母是 target_update 与 target_calibration 患者的并集；每个 repeat
    使用同一随机排列的前缀，保证预算嵌套。target_test/internal_test 患者永不进入。
    """
    required = {"患者id", "center", "split_role"}
    missing = required.difference(split_manifest.columns)
    if missing:
        raise ValueError(f"split_manifest 缺少列: {sorted(missing)}")
    budget_values = tuple(float(b) for b in budgets)
    if not budget_values or any(b <= 0 or b > 1 for b in budget_values):
        raise ValueError("预算必须位于 (0, 1]")
    if tuple(sorted(set(budget_values))) != budget_values:
        raise ValueError("预算必须升序且不重复")

    eligible = split_manifest[
        split_manifest["center"].eq("target")
        & split_manifest["split_role"].isin(["target_update", "target_calibration"])
    ][["患者id", "split_role"]].copy()
    eligible["患者id"] = eligible["患者id"].astype(str)
    patients = np.asarray(sorted(eligible["患者id"].unique()), dtype=object)
    if not len(patients):
        raise ValueError("没有可用于update/calibration合计预算的患者")
    seeds = derive_repeat_seeds(root_seed, repeats)
    rows: list[dict[str, object]] = []
    for repeat, seed in seeds.items():
        order = patients.copy()
        np.random.default_rng(seed).shuffle(order)
        for budget in budget_values:
            count = len(order) if budget == 1.0 else max(1, int(round(budget * len(order))))
            selected = set(order[:count].tolist())
            for patient_id in sorted(selected):
                rows.append({
                    "患者id": patient_id,
                    "budget": budget,
                    "subset_repeat": repeat,
                    "repeat_seed": seed,
                    "budget_population": "target_update_plus_target_calibration",
                })
    manifest = pd.DataFrame(rows).sort_values(
        ["subset_repeat", "budget", "患者id"], kind="mergesort"
    ).reset_index(drop=True)
    return manifest


def freeze_budget_manifest(manifest: pd.DataFrame, output_csv: str | Path) -> dict[str, object]:
    """不可覆盖写出CSV，并返回绑定行数与SHA-256。"""
    target = Path(output_csv)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"拒绝覆盖预算清单: {target}")
    manifest.to_csv(target, index=False, encoding="utf-8")
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    lock = {
        "protocol": "v5_outcome_blind_nested_budget_20260819",
        "rows": int(len(manifest)),
        "sha256": digest,
        "columns": list(manifest.columns),
    }
    lock_path = target.with_suffix(target.suffix + ".json")
    if lock_path.exists():
        raise FileExistsError(f"拒绝覆盖预算锁: {lock_path}")
    lock_path.write_text(json.dumps(lock, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return lock


def filter_budget_population(frame: pd.DataFrame, manifest: pd.DataFrame, budget: float, subset_repeat: int) -> pd.DataFrame:
    """只保留指定重复/预算中的update与calibration患者；任何internal_test均被拒绝。"""
    if "split_role" not in frame.columns or "患者id" not in frame.columns:
        raise ValueError("frame 必须包含 split_role 和 患者id")
    if frame["split_role"].eq("internal_test").any():
        raise ValueError("V5 禁止 internal_test 进入预算训练/校准输入")
    selected = set(manifest.loc[
        manifest["budget"].eq(float(budget)) & manifest["subset_repeat"].eq(int(subset_repeat)), "患者id"
    ].astype(str))
    roles = frame["split_role"].isin(["target_update", "target_calibration"])
    keep = (~roles) | frame["患者id"].astype(str).isin(selected)
    return frame.loc[keep].reset_index(drop=True)
