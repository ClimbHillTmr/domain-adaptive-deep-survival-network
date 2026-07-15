import pandas as pd

from src.data_pipeline.data_process import build_cohort


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
