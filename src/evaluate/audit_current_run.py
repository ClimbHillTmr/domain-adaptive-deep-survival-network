import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import yaml

from src.data.dataset import build_feature_tables
from sklearn.model_selection import train_test_split


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest(path, patient_col, event_col):
    df = pd.read_csv(path)
    out = {
        "path": path,
        "sha256": sha256(path),
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "n_patients": int(df[patient_col].nunique()) if patient_col in df else None,
        "n_events": int(df[event_col].sum()) if event_col in df else None,
        "event_rate": float(df[event_col].mean()) if event_col in df else None,
    }
    if "透析日期" in df:
        dates = pd.to_datetime(df["透析日期"], errors="coerce")
        out["min_dialysis_date"] = str(dates.min())
        out["max_dialysis_date"] = str(dates.max())
    return out


def history_audit(path, patient_col):
    df = pd.read_csv(path)
    if patient_col not in df or "透析日期" not in df:
        return []
    hist_cols = [c for c in df.columns if c.startswith("历史平均") or c.startswith("history_")]
    if not hist_cols:
        return []
    df["透析日期"] = pd.to_datetime(df["透析日期"], errors="coerce")
    rows = []
    for pid, g in df.sort_values([patient_col, "透析日期"]).groupby(patient_col, sort=False):
        first = g.iloc[0]
        history_values = pd.to_numeric(first[hist_cols], errors="coerce").fillna(0)
        rows.append({
            "patient_id": str(pid),
            "first_session_history_all_zero": bool((history_values == 0).all()),
            "checked_columns": len(hist_cols),
        })
        if len(rows) >= 20:
            break
    return rows


def patient_split(df, event_col, patient_col, target_adapt_ratio, target_val_ratio, seed):
    patient_event = df.groupby(patient_col, sort=False)[event_col].max()
    patients = patient_event.index.to_numpy()
    labels = patient_event.to_numpy()
    strat = labels if len(set(labels)) > 1 else None
    adapt, test = train_test_split(
        patients, test_size=1 - target_adapt_ratio, random_state=seed, stratify=strat
    )
    adapt_labels = patient_event.loc[adapt].to_numpy()
    strat2 = adapt_labels if len(set(adapt_labels)) > 1 else None
    train, val = train_test_split(adapt, test_size=target_val_ratio, random_state=seed, stratify=strat2)
    return {"train": set(map(str, train)), "val": set(map(str, val)), "test": set(map(str, test))}


def main():
    config = yaml.safe_load(open("conf/config.yaml"))
    patient_col = config["data"].get("patient_col", "患者id")
    event_col = config["data"].get("event_col", "events")
    data_dir = config["paths"]["data_dir"]
    source_path = os.path.join(data_dir, config["data"]["source_file"])
    target_path = os.path.join(data_dir, config["data"]["target_file"])
    out_dir = Path("experiments/audit")
    out_dir.mkdir(parents=True, exist_ok=True)

    (
        _x_s,
        _x_t,
        e_s,
        _t_s,
        e_t,
        _t_t,
        feature_names,
        df_s_model,
        df_t_model,
        _category_mappings,
    ) = build_feature_tables(
        source_path,
        target_path,
        return_dataframes=True,
    )

    data = {
        "source": {
            **manifest(source_path, patient_col, event_col),
            "model_ready_n_rows": int(len(df_s_model)),
            "model_ready_n_events": int(e_s.sum()),
            "model_ready_n_features": int(len(feature_names)),
        },
        "target": {
            **manifest(target_path, patient_col, event_col),
            "model_ready_n_rows": int(len(df_t_model)),
            "model_ready_n_events": int(e_t.sum()),
            "model_ready_n_features": int(len(feature_names)),
        },
    }
    (out_dir / "data_manifest.json").write_text(json.dumps(data, ensure_ascii=False, indent=2))

    target = df_t_model.copy()
    sets = patient_split(
        target,
        event_col,
        patient_col,
        config["data"].get("target_adapt_ratio", 0.2),
        config["data"].get("target_val_ratio", 0.2),
        config["training"]["seed"],
    )
    split = {
        name: {"n_patients": len(ids), "n_sessions": int(target[patient_col].astype(str).isin(ids).sum())}
        for name, ids in sets.items()
    }
    split["audit_basis"] = "model_ready_target_dataset"
    split["target_rows_before_model_filter"] = int(data["target"]["n_rows"])
    split["target_rows_after_model_filter"] = int(data["target"]["model_ready_n_rows"])
    split["target_rows_removed_before_split"] = int(data["target"]["n_rows"] - data["target"]["model_ready_n_rows"])
    split["overlap_train_val"] = len(sets["train"] & sets["val"])
    split["overlap_train_test"] = len(sets["train"] & sets["test"])
    split["overlap_val_test"] = len(sets["val"] & sets["test"])
    (out_dir / "split_audit.json").write_text(json.dumps(split, ensure_ascii=False, indent=2))

    hist = []
    for label, path in [("source", source_path), ("target", target_path)]:
        for row in history_audit(path, patient_col):
            row["cohort"] = label
            hist.append(row)
    pd.DataFrame(hist).to_csv(out_dir / "history_feature_audit.csv", index=False)


if __name__ == "__main__":
    main()
