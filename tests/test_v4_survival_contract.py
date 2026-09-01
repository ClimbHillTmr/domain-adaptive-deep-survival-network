import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_FEATURES = [
    "年龄", "性别", "透析龄年", "透前收缩压", "透前舒张压", "透前呼吸频率", "透前体温",
    "透前体重-干体重", "超负荷", "历史平均超滤量MAX", "历史平均超滤率_mean", "历史平均透前体重",
    "历史平均透前收缩压", "历史平均透前舒张压", "历史平均透中低血压_计算", "history_IDH_240_rate",
    "history_IH_240_rate", "history_IDH_240_no_event_rate", "history_IDH_240_interval_1_rate",
    "history_IDH_240_interval_2_rate", "history_IDH_240_interval_3_rate", "history_IDH_240_interval_4_rate",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_v4_allowlist_is_exactly_locked_22_without_audit_fields():
    with (ROOT / "v4/data_contract/feature_allowlist.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    primary = [row for row in rows if row["allowed"] == "yes"]

    assert [row["feature"] for row in primary] == EXPECTED_FEATURES
    assert sum(row["primary_group"] == "physiology_history" for row in primary) == 18
    assert sum(row["primary_group"] == "treatment_context" for row in primary) == 4
    assert not any("count" in row["feature"] or "missing" in row["feature"] or "计数" in row["feature"] for row in primary)


def test_v4_split_covers_all_sessions_and_preserves_patient_exclusivity():
    with (ROOT / "v4/data_contract/split_manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    sessions = [row["session_id"] for row in rows]
    patient_roles = {}
    for row in rows:
        patient_roles.setdefault((row["center"], row["patient_id"]), set()).add(row["split_role"])

    assert len(rows) == 286399
    assert len(sessions) == len(set(sessions))
    assert all(len(roles) == 1 for roles in patient_roles.values())
    assert {role for roles in patient_roles.values() for role in roles} == {
        "source_train", "source_validation", "target_update", "target_calibration", "target_test"
    }


def test_v4_preprocessing_and_partition_are_locked_and_consistent():
    preprocessing = json.loads((ROOT / "data/v4/preprocessing.json").read_text(encoding="utf-8"))
    partition_path = ROOT / "v4/experiments/primary_feature_partition.json"
    partition = json.loads(partition_path.read_text(encoding="utf-8"))

    assert preprocessing["fit_cohort"] == "source_train"
    assert preprocessing["fit_source"] == "data/v4/source_v4_survival.csv only"
    assert preprocessing["feature_names"] == EXPECTED_FEATURES
    assert set(preprocessing["medians"]) == set(EXPECTED_FEATURES)
    assert set(preprocessing["means"]) == set(EXPECTED_FEATURES)
    assert set(preprocessing["scales"]) == set(EXPECTED_FEATURES)
    assert len(partition["physiology_history"]) == 18
    assert len(partition["treatment_context"]) == 4
    assert set(partition["physiology_history"] + partition["treatment_context"]) == set(EXPECTED_FEATURES)
    assert _sha256(ROOT / "data/v4/preprocessing.json") == "fde44bc2ae23487c331bb040e3730846a0f91f1067bf86f61491d1c07f7fc32f"
    assert _sha256(partition_path) == "d1a3919667b327ff156e926c5437b87a9f966b530d001d6a2bacbf901d0b5ab8"
