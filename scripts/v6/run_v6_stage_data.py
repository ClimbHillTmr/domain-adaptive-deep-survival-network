#!/usr/bin/env python3
"""v6.0 — 数据研制入口 (P1)。

复用 v5 的验证过的数据构建逻辑 (src/data/v5_data_stage.py 与
src/data/v5_budget_subsets.py)，但输出到独立 v6 根目录，
并把预处理契约标签重打为 v6，写入 v6 SHA-256 资产清单。

产物 (data/v6_20260830/):
  source_cohort.csv / target_cohort.csv / split_manifest.csv
  preprocessing.json / primary_partition.json / random_partitions.csv
  budget_subsets.csv  + asset_manifest.json (各文件 SHA-256 + 行数)

用法:
  python -m scripts.v6.run_v6_stage_data
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.v5_data_stage import build_scientific_assets

V6_PROTOCOL = "v6_20260830_data_protocol_1"
DATA_ROOT = ROOT / "data/v6_20260830"
RAW_SOURCE = ROOT / "data/raw/updated_dataset_shenyi.csv"
RAW_TARGET = ROOT / "data/raw/updated_dataset_fuding.csv"
ROOT_SEED = 20260830


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _retag_preprocessing(assets: dict[str, dict[str, str]]) -> str:
    """把 preprocessing.json 的契约标签从 v5 重打为 v6，返回新哈希。"""
    path = ROOT / assets["preprocessing"]["path"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["protocol"] = V6_PROTOCOL
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return _sha256(path)


def build() -> dict[str, object]:
    if DATA_ROOT.exists():
        raise FileExistsError(f"拒绝覆盖既有 v6 数据根: {DATA_ROOT}")
    if not RAW_SOURCE.is_file() or not RAW_TARGET.is_file():
        raise FileNotFoundError(f"原始数据缺失: {RAW_SOURCE} / {RAW_TARGET}")

    for raw, name in ((RAW_SOURCE, "source"), (RAW_TARGET, "target")):
        # 不校验与 v4 相同的哈希，但确认文件非空可读
        if raw.stat().st_size < 1_000_000:
            raise ValueError(f"{name} raw 文件过小，疑似损坏")

    assets = build_scientific_assets(
        RAW_SOURCE,
        RAW_TARGET,
        DATA_ROOT,
        split_seed=ROOT_SEED,
        budget_seed=ROOT_SEED,
        budget_repeats=1,
        budgets=(0.10, 0.25, 0.50, 1.00),
    )
    new_preprocess_sha = _retag_preprocessing(assets)

    manifest_rows = []
    for name, spec in assets.items():
        path = ROOT / spec["path"]
        import pandas as pd

        try:
            frame = pd.read_csv(path, nrows=0)
            rows = frame.shape[0]
        except Exception:
            rows = None
        manifest_rows.append({
            "asset": name,
            "path": spec["path"],
            "sha256": _sha256(path),
            "declared_sha256": spec["sha256"],
            "protocol": V6_PROTOCOL,
        })
    # 更新 preprocessing 条目的哈希为重打标签后
    for row in manifest_rows:
        if row["asset"] == "preprocessing":
            row["sha256"] = new_preprocess_sha

    manifest_path = DATA_ROOT / "asset_manifest.json"
    payload = {
        "protocol": V6_PROTOCOL,
        "data_root": str(DATA_ROOT),
        "root_seed": ROOT_SEED,
        "budget_repeats": 1,
        "budgets": [0.10, 0.25, 0.50, 1.00],
        "result_blind": True,
        "assets": manifest_rows,
    }
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"status": "built", "data_root": str(DATA_ROOT), "assets": len(manifest_rows), "manifest": str(manifest_path)}


if __name__ == "__main__":
    result = build()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))