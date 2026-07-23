"""Deterministic raw-to-analysis construction for the two dialysis centers."""

from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

IDH_HISTORY_BINS = (0.25, 0.5, 0.75)
IH_MAP_RISE_MMHG = 10.0
DATA_PROTOCOL_VERSION = "v3_hbd_data_protocol_1"
TEMPERATURE_VALID_RANGE_C = (30.0, 45.0)
SEX_CODE_MAP = {
    "女": 0.0,
    "female": 0.0,
    "f": 0.0,
    "0": 0.0,
    "男": 1.0,
    "male": 1.0,
    "m": 1.0,
    "1": 1.0,
}
DIRECT_IDENTIFIER_COLUMNS = ("姓名", "透析记录id", "RECIPE_ID", "Unnamed: 0")
HISTORY_MEAN_COLUMNS = (
    "超滤量MAX",
    "透前体重",
    "透前收缩压",
    "透前舒张压",
    "透中低血压_计算",
    "超滤率_mean",
)


def _numeric_list(value: object) -> list[float] | None:
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return None
    if not isinstance(value, (list, tuple)) or not value:
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def _comma_numbers(value: object) -> list[float] | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return [float(item.strip()) for item in value.split(",")]
    except ValueError:
        return None


def _fuding_bp_pairs(value: object) -> tuple[list[float], list[float]] | None:
    if not isinstance(value, str) or not value.strip():
        return None
    systolic, diastolic = [], []
    for pair in value.split(","):
        parts = pair.strip().split("/")
        if len(parts) != 2:
            return None
        try:
            systolic.append(float(parts[0]))
            diastolic.append(float(parts[1]))
        except ValueError:
            return None
    return (systolic, diastolic) if systolic else None


def _time_nodes(dialysis_date: object, value: object) -> list[pd.Timestamp] | None:
    date = pd.to_datetime(dialysis_date, errors="coerce")
    if pd.isna(date) or not isinstance(value, str):
        return None
    nodes = [item.strip() for item in value.strip("[]").split(",") if item.strip()]
    if not nodes:
        return None
    parsed: list[pd.Timestamp] = []
    for item in nodes:
        stamp = pd.to_datetime(item, errors="coerce")
        if pd.isna(stamp):
            return None
        if len(item) <= 8:  # Clock-only source: anchor it to the dialysis date.
            stamp = pd.Timestamp.combine(date.date(), stamp.time())
        if parsed and stamp < parsed[-1]:
            if parsed[-1] - stamp > pd.Timedelta(hours=12):
                stamp += pd.Timedelta(days=1)
            else:
                return None
        parsed.append(stamp)
    return parsed if len(parsed) >= 2 else None


def _event_index(baseline_sbp: float, systolic: Iterable[float]) -> int | None:
    for index, value in enumerate(systolic):
        if baseline_sbp - value >= 30 or value <= 90:
            return index
    return None


def _high_bp_index(baseline_map: float, maps: Iterable[float]) -> int | None:
    for index, value in enumerate(maps):
        if value - baseline_map > IH_MAP_RISE_MMHG:
            return index
    return None


def _ratio_bin(value: float, boundaries: tuple[float, ...]) -> int:
    if value <= 0:
        return 0
    for index, boundary in enumerate(boundaries, start=1):
        if value <= boundary:
            return index
    return len(boundaries) + 1


def _series_mean(value: object) -> float:
    if isinstance(value, (int, float)) and np.isfinite(value):
        return float(value)
    if isinstance(value, str):
        values = _numeric_list(value) or _comma_numbers(value)
        if values:
            return float(np.mean(values))
    return np.nan


def _missing_series(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(np.nan, index=frame.index, dtype=float)


def _canonicalize_sex(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip().str.lower()
    return normalized.map(SEX_CODE_MAP).astype(float)


def _apply_start_time_protocol(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Apply deterministic V3 cleaning without using outcomes or model performance."""
    out = frame.copy()

    raw_sex = (
        out["性别"]
        if "性别" in out
        else pd.Series(pd.NA, index=out.index, dtype="string")
    )
    sex = _canonicalize_sex(raw_sex)
    sex_counts = raw_sex.astype("string").fillna("__MISSING__").value_counts().to_dict()
    out["性别"] = sex

    dialysis_date = pd.to_datetime(out["透析日期"], errors="coerce")
    birth_date = (
        pd.to_datetime(out["出生日期"], errors="coerce")
        if "出生日期" in out
        else pd.Series(pd.NaT, index=out.index)
    )
    raw_age = (
        pd.to_numeric(out["年龄"], errors="coerce")
        if "年龄" in out
        else _missing_series(out)
    )
    derived_age = (dialysis_date - birth_date).dt.days / 365.25
    derived_age = derived_age.where(derived_age.gt(0) & derived_age.le(120))
    raw_age = raw_age.where(raw_age.gt(0) & raw_age.le(120))
    age_difference = (raw_age - derived_age).abs()
    out["年龄"] = derived_age.fillna(raw_age)

    first_dialysis = (
        pd.to_datetime(out["首次透析日期"], errors="coerce")
        if "首次透析日期" in out
        else pd.Series(pd.NaT, index=out.index)
    )
    vintage_years = (dialysis_date - first_dialysis).dt.days / 365.25
    valid_vintage = vintage_years.ge(0) & out["年龄"].gt(0) & vintage_years.le(out["年龄"])
    out["透析龄年"] = vintage_years.where(valid_vintage)
    out["透析龄占比"] = (out["透析龄年"] / out["年龄"]).where(valid_vintage)

    raw_temperature = (
        pd.to_numeric(out["透前体温"], errors="coerce")
        if "透前体温" in out
        else _missing_series(out)
    )
    lower, upper = TEMPERATURE_VALID_RANGE_C
    invalid_temperature = raw_temperature.notna() & ~raw_temperature.between(lower, upper)
    out["透前体温_异常标记"] = invalid_temperature.astype(int)
    out["透前体温_缺失标记"] = (raw_temperature.isna() | invalid_temperature).astype(int)
    out["透前体温"] = raw_temperature.where(raw_temperature.between(lower, upper))

    # `平均动脉压` is a current-session source sequence, not a baseline predictor.
    # Baseline MAP is represented once, canonically, by `透前动脉压`.
    out = out.drop(columns=["平均动脉压"], errors="ignore")
    dropped_identifiers = [column for column in DIRECT_IDENTIFIER_COLUMNS if column in out]
    out = out.drop(columns=[*DIRECT_IDENTIFIER_COLUMNS, "出生日期"], errors="ignore")

    quality = {
        "protocol_version": DATA_PROTOCOL_VERSION,
        "sex_mapping": {"female": 0, "male": 1, "unknown": None},
        "sex_raw_counts": {str(key): int(value) for key, value in sex_counts.items()},
        "sex_unknown_rows": int(sex.isna().sum()),
        "age_derived_from_birth_date_rows": int(derived_age.notna().sum()),
        "age_fallback_raw_rows": int(derived_age.isna().mul(raw_age.notna()).sum()),
        "age_raw_derived_difference_gt_1_5_years": int(age_difference.gt(1.5).sum()),
        "age_missing_rows": int(out["年龄"].isna().sum()),
        "age_under_18_rows": int(out["年龄"].lt(18).sum()),
        "age_under_18_patients": int(out.loc[out["年龄"].lt(18), "患者id"].nunique()),
        "dialysis_vintage_invalid_or_missing_rows": int(out["透析龄占比"].isna().sum()),
        "temperature_valid_range_c": [lower, upper],
        "temperature_invalid_rows_set_missing": int(invalid_temperature.sum()),
        "temperature_missing_rows_after_gate": int(out["透前体温"].isna().sum()),
        "canonical_baseline_map_column": "透前动脉压",
        "excluded_current_session_map_column": "平均动脉压",
        "dropped_direct_identifier_columns": dropped_identifiers,
    }
    return out, quality


def _ultrafiltration_max(value: object) -> float:
    if isinstance(value, (int, float)) and np.isfinite(value):
        return float(value) * 1000.0  # Fuding stores liters.
    if isinstance(value, str):
        scalar = pd.to_numeric(value, errors="coerce")
        if np.isfinite(scalar):
            return float(scalar) * 1000.0
    values = _numeric_list(value) or _comma_numbers(value)
    return float(np.max(values)) if values else np.nan


def _build_rows(raw: pd.DataFrame, cohort: str) -> tuple[pd.DataFrame, dict[str, int]]:
    records, rejected = [], {"invalid_sequence": 0, "misaligned_sequence": 0, "invalid_time": 0}
    for _, row in raw.iterrows():
        if cohort == "shenyi":
            systolic = _numeric_list(row.get("透析中收缩压"))
            diastolic = _numeric_list(row.get("透析中舒张压"))
        else:
            pairs = _fuding_bp_pairs(row.get("透中血压"))
            systolic, diastolic = pairs if pairs else (None, None)
        if systolic is None or diastolic is None:
            rejected["invalid_sequence"] += 1
            continue
        nodes = _time_nodes(row.get("透析日期"), row.get("透中数据记录时间节点"))
        if nodes is None:
            rejected["invalid_time"] += 1
            continue
        if len(systolic) != len(diastolic) or len(systolic) != len(nodes):
            rejected["misaligned_sequence"] += 1
            continue
        baseline_sbp = pd.to_numeric(row.get("透前收缩压"), errors="coerce")
        baseline_dbp = pd.to_numeric(row.get("透前舒张压"), errors="coerce")
        if not np.isfinite(baseline_sbp) or not np.isfinite(baseline_dbp):
            rejected["invalid_sequence"] += 1
            continue
        maps = [(sbp + 2 * dbp) / 3 for sbp, dbp in zip(systolic, diastolic, strict=True)]
        start, end = nodes[0], nodes[-1]
        relative_minutes = [(node - start).total_seconds() / 60.0 for node in nodes]
        duration = (end - start).total_seconds() / 60.0
        if duration <= 0:
            rejected["invalid_time"] += 1
            continue
        idh_index = _event_index(float(baseline_sbp), systolic)
        high_index = _high_bp_index((float(baseline_sbp) + 2 * float(baseline_dbp)) / 3, maps)
        event_time = 0.0 if idh_index is None else (nodes[idh_index] - start).total_seconds() / 60.0
        high_time = 0.0 if high_index is None else (nodes[high_index] - start).total_seconds() / 60.0
        record = row.to_dict()
        record.update(
            {
                "session_id": f"{cohort}:{row.get('患者id')}:{pd.Timestamp(row.get('透析日期')).date()}:{start.time()}",
                "透析开始时间": start,
                "透析结束时间": end,
                "duration_minutes": duration,
                "minutes_from_start_list": relative_minutes,
                "透中高血压_计算": int(high_index is not None),
                "透中低血压_计算": int(idh_index is not None),
                "涨幅时间点": start if high_index is None else nodes[high_index],
                "降幅时间点": start if idh_index is None else nodes[idh_index],
                "event_time_min": event_time,
                "et_min": event_time,
                "events": int(idh_index is not None),
                "idh_time_min": event_time,
                "idh_event": int(idh_index is not None),
                "ih_time_min": high_time,
                "ih_event": int(high_index is not None),
                "涨幅时间点比值": high_time / duration,
                "降幅时间点比值": event_time / duration,
                "涨幅时间点比值区间": _ratio_bin(high_time / duration, IDH_HISTORY_BINS),
                "降幅时间点比值区间": _ratio_bin(event_time / duration, IDH_HISTORY_BINS),
                "涨幅时间点差值": high_time / 60.0,
                "降幅时间点差值": event_time / 60.0,
                "涨幅时间点差值区间": _ratio_bin(high_time / 60.0, (1.0, 2.0, 3.0)),
                "降幅时间点差值区间": _ratio_bin(event_time / 60.0, (1.0, 2.0, 3.0)),
                "透前动脉压": (float(baseline_sbp) + 2 * float(baseline_dbp)) / 3,
                "脉压差": float(baseline_sbp) - float(baseline_dbp),
                "超滤量MAX": _ultrafiltration_max(row.get("超滤量")),
                "超滤率_mean": _series_mean(row.get("超滤率")),
            }
        )
        dry_weight = pd.to_numeric(row.get("干体重"), errors="coerce")
        pre_weight = pd.to_numeric(row.get("透前体重"), errors="coerce")
        record["透前体重-干体重"] = pre_weight - dry_weight
        record["超负荷"] = (pre_weight - dry_weight) / dry_weight if np.isfinite(dry_weight) and dry_weight > 0 else np.nan
        records.append(record)
    return pd.DataFrame(records), rejected


def _add_prior_history(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values(["患者id", "透析开始时间", "session_id"], kind="stable").reset_index(drop=True)
    groups = frame.groupby("患者id", sort=False)
    for column in HISTORY_MEAN_COLUMNS:
        frame[f"历史平均{column}"] = groups[column].transform(
            lambda series: pd.to_numeric(series, errors="coerce").shift().expanding().mean()
        ).fillna(0.0)
    prior_count = groups.cumcount()
    prior_idh = groups["idh_event"].shift().fillna(0).groupby(frame["患者id"], sort=False).cumsum()
    prior_hbp = groups["ih_event"].shift().fillna(0).groupby(frame["患者id"], sort=False).cumsum()
    denominator = prior_count.replace(0, np.nan)
    frame["history_IDH_rate"] = (prior_idh / denominator).fillna(0.0)
    frame["history_HBP_rate"] = (prior_hbp / denominator).fillna(0.0)
    frame["history_LBP_times_0_rate"] = ((prior_count - prior_idh) / denominator).fillna(0.0)
    for bin_number in range(1, 5):
        previous = groups["降幅时间点比值区间"].shift().eq(bin_number).fillna(False)
        count = previous.astype(int).groupby(frame["患者id"], sort=False).cumsum()
        frame[f"history_LBP_times_{bin_number}_rate"] = (count / denominator).fillna(0.0)

    # Tied timestamps are rare, but neither row may use the other as history.
    tied = frame.duplicated(["患者id", "透析开始时间"], keep=False)
    for (patient_id, timestamp), indices in frame.loc[tied].groupby(
        ["患者id", "透析开始时间"], sort=False
    ).groups.items():
        prior = frame.loc[
            frame["患者id"].eq(patient_id) & frame["透析开始时间"].lt(timestamp)
        ]
        count = len(prior)
        for column in HISTORY_MEAN_COLUMNS:
            frame.loc[indices, f"历史平均{column}"] = pd.to_numeric(
                prior[column], errors="coerce"
            ).mean() if count else 0.0
        prior_idh_count = int(prior["idh_event"].sum())
        frame.loc[indices, "history_IDH_rate"] = prior_idh_count / count if count else 0.0
        frame.loc[indices, "history_HBP_rate"] = float(prior["ih_event"].sum()) / count if count else 0.0
        frame.loc[indices, "history_LBP_times_0_rate"] = (count - prior_idh_count) / count if count else 0.0
        for bin_number in range(1, 5):
            frame.loc[indices, f"history_LBP_times_{bin_number}_rate"] = (
                float(prior["降幅时间点比值区间"].eq(bin_number).sum()) / count if count else 0.0
            )
    return frame


def build_cohort(raw_path: str | Path, cohort: str) -> tuple[pd.DataFrame, dict[str, object]]:
    if cohort not in {"shenyi", "fuding"}:
        raise ValueError("cohort must be 'shenyi' or 'fuding'")
    raw = pd.read_csv(raw_path, low_memory=False)
    frame, rejected = _build_rows(raw, cohort)
    if frame.empty:
        raise ValueError("No valid sessions after sequence and time validation.")
    frame["透析日期"] = pd.to_datetime(frame["透析日期"], errors="coerce")
    frame["session_id"] = (
        frame["session_id"]
        + ":"
        + frame.groupby(["患者id", "透析日期", "透析开始时间"], sort=False).cumcount().astype(str)
    )
    frame, quality = _apply_start_time_protocol(frame)
    frame = _add_prior_history(frame)
    audit: dict[str, object] = {
        "cohort": cohort,
        "data_protocol_version": DATA_PROTOCOL_VERSION,
        "input_path": str(raw_path),
        "input_rows": int(len(raw)),
        "output_rows": int(len(frame)),
        "rejected": rejected,
        "n_patients": int(frame["患者id"].nunique()),
        "n_idh_events": int(frame["events"].sum()),
        "n_ih_events": int(frame["ih_event"].sum()),
        "idh_rule": "baseline_sbp - intradialytic_sbp >= 30 or intradialytic_sbp <= 90",
        "ih_rule": f"intradialytic_map - baseline_map > {IH_MAP_RISE_MMHG:g}",
        "non_event_time_min": 0.0,
        "quality_protocol": quality,
    }
    return frame, audit


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_cohort(
    raw_path: str | Path,
    cohort: str,
    output_path: str | Path,
    *,
    expected_raw_sha256: str | None = None,
    expected_summary: dict[str, int] | None = None,
) -> dict[str, object]:
    raw = Path(raw_path)
    input_hash = _sha256(raw)
    if expected_raw_sha256 is not None and input_hash != expected_raw_sha256:
        raise ValueError(
            f"Raw {cohort} SHA-256 mismatch: expected {expected_raw_sha256}, observed {input_hash}"
        )
    frame, audit = build_cohort(raw_path, cohort)
    if expected_summary is not None:
        observed_summary = {
            "output_rows": int(audit["output_rows"]),
            "n_patients": int(audit["n_patients"]),
            "n_idh_events": int(audit["n_idh_events"]),
            "n_ih_events": int(audit["n_ih_events"]),
        }
        mismatches = {
            key: {"expected": int(expected), "observed": observed_summary.get(key)}
            for key, expected in expected_summary.items()
            if observed_summary.get(key) != int(expected)
        }
        if mismatches:
            raise ValueError(f"Locked {cohort} cohort summary mismatch: {mismatches}")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    complete_audit = {
        **audit,
        "input_sha256": input_hash,
        "output_sha256": _sha256(output),
        "output_path": str(output),
    }
    audit_path = output.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(complete_audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**complete_audit, "audit_path": str(audit_path)}
