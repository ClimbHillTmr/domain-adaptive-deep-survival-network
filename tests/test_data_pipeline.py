import hashlib
import os

import numpy as np
import pandas as pd
import pytest

from src.data_pipeline.HBD_survival_data import EXPECTED_RAW_SHA256 as SHENYI_V4_RAW_SHA256
from src.data_pipeline.HBD_survival_data_fuding import EXPECTED_RAW_SHA256 as FUDING_V4_RAW_SHA256
from src.data_pipeline.data_process import (
    _comma_numbers,
    _fuding_bp_pairs,
    _numeric_list,
    build_cohort,
    build_survival_cohort,
    write_cohort,
    write_survival_cohort,
)


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


def _survival_raw(rows):
    defaults = {
        "首次透析日期": "2020-01-01",
        "年龄": 60,
        "透前收缩压": 120,
        "透前舒张压": 80,
        "透前体重": 60,
        "干体重": 58,
        "超滤量": "[2000, 2100]",
        "超滤率": "[500, 500]",
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


def test_v4_official_entry_points_lock_raw_hashes_without_result_summaries():
    assert SHENYI_V4_RAW_SHA256 == "607a7f115efa2db64c3f23190f82222978d10146570bdbf6e1f66118bd2320f9"
    assert FUDING_V4_RAW_SHA256 == "6fcfd523cd60aedc245d4b51e072aeca777f65e9a56554fdebaf45d3ee5f3813"


def test_v4_sequence_parsers_reject_non_finite_values():
    for value in ([1, np.nan], [1, np.inf], "[1, nan]", "[1, inf]"):
        assert _numeric_list(value) is None
    for value in ("1,nan", "1,inf", "1,-inf"):
        assert _comma_numbers(value) is None
    for value in ("120/80,nan/60", "120/80,90/inf", "120/80,-inf/60"):
        assert _fuding_bp_pairs(value) is None


def test_v4_survival_threshold_zero_no_event_and_short_censoring(tmp_path):
    raw = _survival_raw(
        [
            {
                "患者id": "threshold",
                "透析日期": "2021-01-01",
                "透析中收缩压": "[90, 89]",
                "透析中舒张压": "[80, 60]",
                "透中数据记录时间节点": "[08:00,09:00]",
            },
            {
                "患者id": "none",
                "透析日期": "2021-01-01",
                "透析中收缩压": "[120, 110]",
                "透析中舒张压": "[80, 75]",
                "透中数据记录时间节点": "[08:00,08:45]",
            },
            {
                "患者id": "ih-threshold",
                "透析日期": "2021-01-01",
                "透析中收缩压": "[120, 120]",
                "透析中舒张压": "[95, 96]",
                "透中数据记录时间节点": "[08:00,09:00]",
            },
        ]
    )
    path = tmp_path / "raw.csv"
    raw.to_csv(path, index=False)
    frame, audit = build_survival_cohort(path, "shenyi")

    threshold = frame.loc[frame["患者id"].eq("threshold")].iloc[0]
    assert threshold["idh_event_time_raw"] == 0.0
    assert threshold["idh_survival_time"] == 0.0
    assert threshold["idh_event_observed_240"] == 1
    assert threshold["idh_event_interval_60"] == 1
    no_event = frame.loc[frame["患者id"].eq("none")].iloc[0]
    assert pd.isna(no_event["event_time_raw"])
    assert no_event["survival_time"] == 45.0
    assert no_event["event_observed_240"] == 0
    assert pd.isna(no_event["event_interval_60"])
    ih_threshold = frame.loc[frame["患者id"].eq("ih-threshold")].iloc[0]
    assert ih_threshold["ih_event_time_raw"] == 60.0
    assert ih_threshold["ih_event_observed_240"] == 1
    assert audit["data_protocol_version"] == "v4_hbd_survival_protocol_1"


def test_v4_survival_240_administrative_boundary_and_60_minute_bins(tmp_path):
    times = [0, 60, 61, 120, 121, 180, 181, 240, 241]
    rows = []
    for time in times:
        rows.append(
            {
                "患者id": f"p{time}",
                "透析日期": "2021-01-01",
                "透析中收缩压": "[120, 90]",
                "透析中舒张压": "[80, 60]",
                "透中数据记录时间节点": f"[2021-01-01 08:00,2021-01-01 08:00]"
                if time == 0
                else f"[2021-01-01 08:00,2021-01-01 08:00]",
            }
        )
    # A positive duration is required; put the event at the requested middle node.
    for row, time in zip(rows, times, strict=True):
        end = max(time + 1, 1)
        row["透析中收缩压"] = "[120, 90, 120]"
        row["透析中舒张压"] = "[80, 60, 80]"
        start = pd.Timestamp("2021-01-01 08:00")
        nodes = [start, start + pd.Timedelta(minutes=time), start + pd.Timedelta(minutes=end)]
        row["透中数据记录时间节点"] = "[" + ",".join(node.strftime("%Y-%m-%d %H:%M") for node in nodes) + "]"
    path = tmp_path / "raw.csv"
    _survival_raw(rows).to_csv(path, index=False)
    frame, _ = build_survival_cohort(path, "shenyi")
    observed = frame.set_index("患者id")

    expected_bins = {0: 1, 60: 1, 61: 2, 120: 2, 121: 3, 180: 3, 181: 4, 240: 4}
    for time, interval in expected_bins.items():
        row = observed.loc[f"p{time}"]
        assert row["event_observed_240"] == 1
        assert row["survival_time"] == time
        assert row["event_interval_60"] == interval
    assert observed.loc["p241", "event_observed_240"] == 0
    assert observed.loc["p241", "event_time_raw"] == 241
    assert observed.loc["p241", "survival_time"] == 240
    assert pd.isna(observed.loc["p241", "event_interval_60"])


def test_v4_strict_history_ties_and_history_measurement_state(tmp_path):
    raw = _survival_raw(
        [
            {
                "患者id": "p1",
                "透析日期": "2021-01-01",
                "透析中收缩压": "[120, 90]",
                "透析中舒张压": "[80, 60]",
                "透中数据记录时间节点": "[08:00,09:00]",
            },
            {
                "患者id": "p1",
                "透析日期": "2021-01-02",
                "透析中收缩压": "[120, 110]",
                "透析中舒张压": "[80, 75]",
                "透中数据记录时间节点": "[08:00,09:00]",
            },
            {
                "患者id": "p1",
                "透析日期": "2021-01-02",
                "透析中收缩压": "[120, 90]",
                "透析中舒张压": "[80, 60]",
                "透中数据记录时间节点": "[08:00,09:00]",
            },
        ]
    )
    path = tmp_path / "raw.csv"
    raw.to_csv(path, index=False)
    frame, _ = build_survival_cohort(path, "shenyi")

    first = frame.iloc[0]
    assert first["history_is_first_session"] == 1
    assert first["history_IDH_240_prior_session_count"] == 0
    tied = frame.loc[frame["透析开始时间"].eq(pd.Timestamp("2021-01-02 08:00:00"))]
    assert tied["history_is_first_session"].eq(0).all()
    assert tied["history_IDH_240_prior_session_count"].eq(1).all()
    assert tied["history_IDH_240_measured_count"].eq(1).all()
    assert tied["history_IDH_240_measurement_missing"].eq(0).all()
    assert tied["history_IDH_240_rate"].eq(1.0).all()
    assert tied["history_IDH_240_interval_1_rate"].eq(1.0).all()
    assert tied["历史平均透前体重_有效观测计数"].eq(1).all()
    assert tied["历史平均透前体重_全既往缺失标记"].eq(0).all()
    assert tied["历史平均透前体重"].eq(60.0).all()


def test_v4_history_mean_distinguishes_first_session_from_all_prior_missing(tmp_path):
    raw = _survival_raw(
        [
            {
                "患者id": "p1",
                "透析日期": "2021-01-01",
                "透前体重": np.nan,
                "透析中收缩压": "[120, 110]",
                "透析中舒张压": "[80, 75]",
                "透中数据记录时间节点": "[08:00,09:00]",
            },
            {
                "患者id": "p1",
                "透析日期": "2021-01-02",
                "透前体重": 60,
                "透析中收缩压": "[120, 110]",
                "透析中舒张压": "[80, 75]",
                "透中数据记录时间节点": "[08:00,09:00]",
            },
            {
                "患者id": "p1",
                "透析日期": "2021-01-03",
                "透前体重": 62,
                "透析中收缩压": "[120, 110]",
                "透析中舒张压": "[80, 75]",
                "透中数据记录时间节点": "[08:00,09:00]",
            },
        ]
    )
    path = tmp_path / "raw.csv"
    raw.to_csv(path, index=False)

    v3, _ = build_cohort(path, "shenyi")
    v4, _ = build_survival_cohort(path, "shenyi")

    assert v3.loc[1, "历史平均透前体重"] == 0.0
    assert pd.isna(v4.loc[0, "历史平均透前体重"])
    assert v4.loc[0, "history_is_first_session"] == 1
    assert v4.loc[0, "历史平均透前体重_全既往缺失标记"] == 0
    assert pd.isna(v4.loc[1, "历史平均透前体重"])
    assert v4.loc[1, "history_is_first_session"] == 0
    assert v4.loc[1, "历史平均透前体重_有效观测计数"] == 0
    assert v4.loc[1, "历史平均透前体重_全既往缺失标记"] == 1
    assert v4.loc[2, "历史平均透前体重"] == 60.0
    assert v4.loc[2, "历史平均透前体重_有效观测计数"] == 1
    assert v4.loc[2, "历史平均透前体重_全既往缺失标记"] == 0


def test_v4_fuding_sbp_dbp_and_writer_refuses_non_v4_or_overwrite(tmp_path, monkeypatch):
    raw = _survival_raw(
        [
            {
                "患者id": "f1",
                "透析日期": "2021-01-01",
                "透中血压": "120/80,90/60",
                "透中数据记录时间节点": "08:00,09:00",
            }
        ]
    ).drop(columns=["超滤量", "超滤率"])
    raw["超滤量"] = 2.0
    raw_path = tmp_path / "fuding.csv"
    raw.to_csv(raw_path, index=False)
    frame, _ = build_survival_cohort(raw_path, "fuding")
    assert frame.loc[0, "idh_event_time_raw"] == 60.0
    assert frame.loc[0, "idh_event_observed_240"] == 1

    with pytest.raises(ValueError, match="v4 directory"):
        write_survival_cohort(raw_path, "fuding", tmp_path / "data" / "output.csv")
    output = tmp_path / "data" / "v4" / "output.csv"
    result = write_survival_cohort(raw_path, "fuding", output)
    assert output.exists()
    assert output.with_suffix(".audit.json").exists()
    assert result["output_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert not list(output.parent.glob("*.tmp"))
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_survival_cohort(raw_path, "fuding", output)

    failed_output = tmp_path / "failed" / "v4" / "output.csv"
    real_link = os.link

    def fail_audit_publish(source, destination):
        if str(destination).endswith(".audit.json"):
            raise OSError("simulated audit publish failure")
        return real_link(source, destination)

    monkeypatch.setattr(os, "link", fail_audit_publish)
    with pytest.raises(OSError, match="simulated audit publish failure"):
        write_survival_cohort(raw_path, "fuding", failed_output)
    assert not failed_output.exists()
    assert not failed_output.with_suffix(".audit.json").exists()
    assert not list(failed_output.parent.glob("*.tmp"))
