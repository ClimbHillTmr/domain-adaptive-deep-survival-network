import os
import json
import csv
from datetime import datetime

BASE_DIRS = [
    os.path.join("runs", "matrix"),
    os.path.join("runs", "matrix_cv"),
    os.path.join("runs", "matrix_cvq"),
    os.path.join("runs", "matrix_da"),
]

FIELDS = [
    ("f1_macro_ext", ["f1_macro_ext", "f1_macro_external", "f1_macro"]),
    ("f1_micro_ext", ["f1_micro_ext", "f1_micro_external", "f1_micro"]),
    ("precision_macro_ext", ["precision_macro_ext", "precision_macro_external", "precision_macro"]),
    ("recall_macro_ext", ["recall_macro_ext", "recall_macro_external", "recall_macro"]),
    ("brier_ext", ["brier_ext", "brier_external", "brier"]),
    ("early_bias_ext", ["early_bias_ext", "early_bias_external", "early_bias"]),
    ("late_bias_ext", ["late_bias_ext", "late_bias_external", "late_bias"]),
    ("mean_offset_ext", ["mean_offset_ext", "mean_offset_external", "mean_offset"]),
]

def latest_timestamp_dir(path):
    if not os.path.isdir(path):
        return None
    candidates = [d for d in os.listdir(path) if d.startswith("20") and os.path.isdir(os.path.join(path, d))]
    if not candidates:
        return None
    return sorted(candidates)[-1]

def read_metrics_json(run_dir):
    p = os.path.join(run_dir, "external_metrics.json")
    if not os.path.isfile(p):
        return None
    with open(p, "r") as f:
        try:
            return json.load(f)
        except Exception:
            return None

def extract_fields(js):
    out = {}
    if js is None:
        return out
    for k, aliases in FIELDS:
        val = None
        for a in aliases:
            if a in js:
                val = js[a]
                break
        out[k] = val
    return out

def main():
    rows = []
    for base in BASE_DIRS:
        if not os.path.isdir(base):
            continue
        for name in os.listdir(base):
            path = os.path.join(base, name)
            if not os.path.isdir(path):
                continue
            ts = latest_timestamp_dir(path)
            if not ts:
                continue
            run_dir = os.path.join(path, ts)
            metrics = read_metrics_json(run_dir)
            fields = extract_fields(metrics)
            row = {
                "config": name,
                "dir": run_dir,
            }
            row.update(fields)
            rows.append(row)
    out_path = os.path.join("runs", f"matrix_aggressive_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        header = [
            "config","dir",
            "f1_macro_ext","f1_micro_ext","precision_macro_ext","recall_macro_ext",
            "brier_ext","early_bias_ext","late_bias_ext","mean_offset_ext",
        ]
        w.writerow(header)
        for r in rows:
            w.writerow([
                r.get("config"), r.get("dir"),
                r.get("f1_macro_ext"), r.get("f1_micro_ext"), r.get("precision_macro_ext"), r.get("recall_macro_ext"),
                r.get("brier_ext"), r.get("early_bias_ext"), r.get("late_bias_ext"), r.get("mean_offset_ext"),
            ])
    print(out_path)

if __name__ == "__main__":
    main()