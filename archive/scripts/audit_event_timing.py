"""
Audit event timing consistency across raw / optimized / final datasets.

Goals:
1. Compare event rate and late-event prevalence from optimized/final labels.
2. Compare against a simplified raw reconstruction to quantify mismatch.
3. Check whether raw dynamic sequences can be merged back to optimized/final labels
   via patient id + dialysis date.
"""

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/home/cht/Works/domain-adaptive-deep-survival-network")

CONFIG = {
    "landmarks": [30, 45, 60, 90, 120],
    "source": {
        "raw": ROOT / "data_preprocessing/updated_dataset_shenyi.csv",
        "optimized": ROOT / "data_preprocessing/data/深医_optimized_data透前动脉压.csv",
        "final": ROOT / "data_preprocessing/data/深医_final_data.csv",
    },
    "target": {
        "raw": ROOT / "data_preprocessing/updated_dataset_fuding.csv",
        "optimized": ROOT / "data_preprocessing/data/福鼎_optimized_data透前动脉压.csv",
        "final": ROOT / "data_preprocessing/data/福鼎_final_data.csv",
    },
    "output_json": ROOT / "runs/audit_event_timing_results.json",
}


def to_float_list(x):
    if isinstance(x, list):
        seq = x
    elif isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        try:
            parsed = ast.literal_eval(s)
        except Exception:
            parsed = [p.strip() for p in s.split(",") if p.strip()]
        if isinstance(parsed, list):
            seq = parsed
        else:
            seq = [parsed]
    else:
        return []

    out = []
    for v in seq:
        try:
            fv = float(v)
            if np.isfinite(fv):
                out.append(fv)
        except Exception:
            continue
    return out


def parse_bp_series(x):
    if isinstance(x, str):
        parts = [p.strip() for p in x.split(",") if p.strip()]
    elif isinstance(x, list):
        parts = x
    else:
        return [], []

    sbp, dbp = [], []
    for part in parts:
        if isinstance(part, str) and "/" in part:
            a, b = part.split("/", 1)
            try:
                sa = float(a)
                sb = float(b)
            except Exception:
                continue
            if np.isfinite(sa) and np.isfinite(sb):
                sbp.append(sa)
                dbp.append(sb)
    return sbp, dbp


def extract_minutes(x):
    if isinstance(x, list):
        parsed = x
    elif isinstance(x, str):
        try:
            parsed = ast.literal_eval(x)
        except Exception:
            parsed = [p.strip() for p in x.split(",") if p.strip()]
    else:
        return []

    try:
        dt = pd.to_datetime(parsed, errors="coerce")
        if len(dt) == 0 or dt.isna().all():
            return []
        base = dt[0]
        return [float((d - base).total_seconds() / 60.0) for d in dt]
    except Exception:
        return []


def normalize_key(df):
    out = df.copy()
    out["患者id"] = out["患者id"].astype(str).str.strip()
    out["透析日期"] = pd.to_datetime(out["透析日期"], errors="coerce").dt.strftime(
        "%Y-%m-%d"
    )
    return out


def summarize_event_distribution(df, event_col, et_col, name, landmarks):
    event = pd.to_numeric(df[event_col], errors="coerce").fillna(0).astype(int)
    et = pd.to_numeric(df[et_col], errors="coerce")
    result = {
        "name": name,
        "n": int(len(df)),
        "event_rate": float(event.mean()),
        "late_event_among_all": {},
        "late_event_among_events": {},
    }
    event_count = max(int((event == 1).sum()), 1)
    for lm in landmarks:
        late_count = int(((event == 1) & (et > lm)).sum())
        result["late_event_among_all"][str(lm)] = late_count / max(len(df), 1)
        result["late_event_among_events"][str(lm)] = late_count / event_count
    return result


def build_raw_simple_reconstruction(raw_df):
    df = normalize_key(raw_df)

    if "透析中收缩压" not in df.columns and "透中血压" in df.columns:
        parsed = df["透中血压"].apply(parse_bp_series)
        df["透析中收缩压"] = parsed.apply(lambda x: x[0])
    else:
        df["透析中收缩压"] = df["透析中收缩压"].apply(to_float_list)

    minutes = df["透中数据记录时间节点"].apply(extract_minutes)
    df["minutes_from_start_list"] = minutes

    simple_event = []
    simple_et = []
    for sbp, mins, pre in zip(df["透析中收缩压"], minutes, df["透前收缩压"]):
        try:
            first = float(pre)
        except Exception:
            first = np.nan
        n = min(len(sbp), len(mins))
        sbp = sbp[:n]
        mins = mins[:n]
        event = 0
        et = np.nan
        if np.isfinite(first):
            for p, m in zip(sbp, mins):
                if (first - p >= 30) or (p <= 90):
                    event = 1
                    et = float(m)
                    break
        simple_event.append(event)
        simple_et.append(et)

    df["simple_event"] = simple_event
    df["simple_et_min"] = simple_et
    return df[["患者id", "透析日期", "simple_event", "simple_et_min"]]


def compute_merge_audit(
    raw_df, label_df, label_event_col="透中低血压_计算", label_et_col="et_min"
):
    raw_key = normalize_key(raw_df)[["患者id", "透析日期"]].drop_duplicates()
    label_key = normalize_key(label_df)[
        ["患者id", "透析日期", label_event_col, label_et_col]
    ]
    merged = raw_key.merge(label_key, on=["患者id", "透析日期"], how="left")
    matched = merged[label_event_col].notna().mean()
    return {
        "raw_unique_sessions": int(len(raw_key)),
        "label_rows": int(len(label_df)),
        "merge_match_rate": float(matched),
    }


def compute_simple_vs_label_audit(raw_df, label_df):
    raw_simple = build_raw_simple_reconstruction(raw_df)
    label_small = normalize_key(label_df)[
        ["患者id", "透析日期", "透中低血压_计算", "et_min"]
    ]
    merged = raw_simple.merge(label_small, on=["患者id", "透析日期"], how="inner")
    if len(merged) == 0:
        return {"n_merged": 0}

    simple_event = (
        pd.to_numeric(merged["simple_event"], errors="coerce").fillna(0).astype(int)
    )
    label_event = (
        pd.to_numeric(merged["透中低血压_计算"], errors="coerce").fillna(0).astype(int)
    )
    label_et = pd.to_numeric(merged["et_min"], errors="coerce")
    simple_et = pd.to_numeric(merged["simple_et_min"], errors="coerce")

    return {
        "n_merged": int(len(merged)),
        "simple_event_rate": float(simple_event.mean()),
        "label_event_rate": float(label_event.mean()),
        "event_agreement": float((simple_event == label_event).mean()),
        "label_late60_among_events": float(
            ((label_event == 1) & (label_et > 60)).sum()
            / max((label_event == 1).sum(), 1)
        ),
        "simple_late60_among_events": float(
            ((simple_event == 1) & (simple_et > 60)).sum()
            / max((simple_event == 1).sum(), 1)
        ),
    }


def load_table(path):
    return pd.read_csv(path)


def audit_domain(name, paths):
    raw_df = load_table(paths["raw"])
    optimized_df = load_table(paths["optimized"])
    final_df = load_table(paths["final"])

    optimized_df = normalize_key(optimized_df)
    final_df = normalize_key(final_df)

    results = {
        "optimized": summarize_event_distribution(
            optimized_df,
            "透中低血压_计算",
            "et_min",
            f"{name}_optimized",
            CONFIG["landmarks"],
        ),
        "final": summarize_event_distribution(
            final_df, "透中低血压_计算", "et_min", f"{name}_final", CONFIG["landmarks"]
        ),
        "raw_to_optimized_merge": compute_merge_audit(raw_df, optimized_df),
        "raw_simple_vs_optimized_label": compute_simple_vs_label_audit(
            raw_df, optimized_df
        ),
    }
    return results


def main():
    output = {
        "source": audit_domain("source", CONFIG["source"]),
        "target": audit_domain("target", CONFIG["target"]),
    }

    CONFIG["output_json"].parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG["output_json"], "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"\nSaved audit to: {CONFIG['output_json']}")


if __name__ == "__main__":
    main()
