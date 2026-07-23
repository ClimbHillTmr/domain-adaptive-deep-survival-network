import hashlib

import pandas as pd
import pytest

from src.data_pipeline.data_process import build_cohort, write_cohort


def test_raw_builder_preserves_idh_rule_midnight_and_prior_history(tmp_path):
    raw = pd.DataFrame(
        {
            "患者id": ["p1", "p1"],
            "透析日期": ["2021-01-01", "2021-01-02"],
            "首次透析日期": ["2020-01-01", "2020-01-01"],
            "年龄": [60, 60],
            "透前收缩压": [120, 120],
            "透前舒张压": [80, 80],
            "透前体重": [60, 60],
            "干体重": [58, 58],
            "超滤量": ["[2000, 2100]", "[2000, 2100]"],
            "超滤率": ["[500, 500]", "[500, 500]"],
            "透析中收缩压": ["[120, 90]", "[120, 150]"],
            "透析中舒张压": ["[80, 60]", "[80, 90]"],
            "透中数据记录时间节点": ["[23:30,00:30]", "[08:00,09:00]"],
        }
    )
    path = tmp_path / "shenyi.csv"
    raw.to_csv(path, index=False)

    frame, audit = build_cohort(path, "shenyi")

    assert audit["rejected"] == {"invalid_sequence": 0, "misaligned_sequence": 0, "invalid_time": 0}
    assert frame.loc[0, "events"] == 1
    assert frame.loc[0, "idh_event"] == 1
    assert frame.loc[0, "ih_event"] == 0
    assert frame.loc[0, "ih_time_min"] == 0.0
    assert frame.loc[0, "event_time_min"] == 60.0
    assert frame.loc[0, "duration_minutes"] == 60.0
    assert frame.loc[1, "events"] == 0
    assert frame.loc[1, "idh_event"] == 0
    assert frame.loc[1, "idh_time_min"] == 0.0
    assert frame.loc[1, "ih_event"] == 1
    assert frame.loc[1, "ih_time_min"] == 60.0
    assert frame.loc[1, "event_time_min"] == 0.0
    assert frame.loc[1, "历史平均透中低血压_计算"] == 1.0
    assert frame.loc[1, "history_IDH_rate"] == 1.0
    assert frame.loc[1, "history_HBP_rate"] == 0.0


def test_v3_start_time_protocol_repairs_derivable_fields_without_guessing(tmp_path):
    raw = pd.DataFrame(
        {
            "患者id": ["p1", "p2"],
            "透析记录id": ["r1", "r2"],
            "姓名": ["direct-name-1", "direct-name-2"],
            "透析日期": ["2021-01-01", "2021-01-02"],
            "出生日期": ["1961-01-01", "1971-01-02"],
            "首次透析日期": ["2020-01-01", "2020-01-02"],
            "年龄": [None, None],
            "性别": ["男", "女"],
            "透前体温": [3693.0, 36.5],
            "透前收缩压": [120, 130],
            "透前舒张压": [80, 70],
            "透前体重": [60, 60],
            "干体重": [58, 58],
            "超滤量": ["[2000, 2100]", "[2000, 2100]"],
            "超滤率": ["[500, 500]", "[500, 500]"],
            "透析中收缩压": ["[120, 110]", "[130, 120]"],
            "透析中舒张压": ["[80, 70]", "[70, 65]"],
            "透中数据记录时间节点": ["[08:00,09:00]", "[08:00,09:00]"],
            "平均动脉压": ["[90,80]", "[90,80]"],
        }
    )
    path = tmp_path / "shenyi.csv"
    raw.to_csv(path, index=False)

    frame, audit = build_cohort(path, "shenyi")

    assert frame["性别"].tolist() == [1.0, 0.0]
    assert frame["年龄"].between(49.9, 60.1).all()
    assert frame["透析龄占比"].notna().all()
    assert pd.isna(frame.loc[0, "透前体温"])
    assert frame.loc[0, "透前体温_异常标记"] == 1
    assert frame.loc[1, "透前体温"] == 36.5
    assert "平均动脉压" not in frame
    assert "姓名" not in frame
    assert "透析记录id" not in frame
    assert "出生日期" not in frame
    assert audit["data_protocol_version"] == "v3_hbd_data_protocol_1"
    assert audit["quality_protocol"]["temperature_invalid_rows_set_missing"] == 1


def test_official_writer_fails_closed_on_hash_or_summary_mismatch(tmp_path):
    raw = pd.DataFrame(
        {
            "患者id": ["p1"],
            "透析日期": ["2021-01-01"],
            "出生日期": ["1961-01-01"],
            "首次透析日期": ["2020-01-01"],
            "性别": ["男"],
            "透前收缩压": [120],
            "透前舒张压": [80],
            "透前体重": [60],
            "干体重": [58],
            "超滤量": ["[2000, 2100]"],
            "超滤率": ["[500, 500]"],
            "透析中收缩压": ["[120, 110]"],
            "透析中舒张压": ["[80, 70]"],
            "透中数据记录时间节点": ["[08:00,09:00]"],
        }
    )
    raw_path = tmp_path / "raw.csv"
    output_path = tmp_path / "output.csv"
    raw.to_csv(raw_path, index=False)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        write_cohort(raw_path, "shenyi", output_path, expected_raw_sha256="0" * 64)
    assert not output_path.exists()

    digest = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="cohort summary mismatch"):
        write_cohort(
            raw_path,
            "shenyi",
            output_path,
            expected_raw_sha256=digest,
            expected_summary={"output_rows": 2},
        )
    assert not output_path.exists()
