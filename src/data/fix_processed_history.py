from pathlib import Path

import pandas as pd


def fix_history(path, patient_col="患者id", date_col="透析日期"):
    df = pd.read_csv(path)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.sort_values([patient_col, date_col]).reset_index(drop=True)

    hist_mean_cols = [c for c in df.columns if c.startswith("历史平均")]
    for out_col in hist_mean_cols:
        src_col = out_col.replace("历史平均", "", 1)
        if src_col in df.columns:
            df[out_col] = (
                df.groupby(patient_col, sort=False)[src_col]
                .transform(lambda s: s.shift(1).expanding().mean())
                .fillna(0)
            )

    seen = df.groupby(patient_col, sort=False).cumcount()
    seen_nonzero = seen.replace(0, pd.NA)
    if "透中低血压_计算" in df.columns:
        prev_event = df.groupby(patient_col, sort=False)["透中低血压_计算"].shift(1).fillna(0)
        event_sum = prev_event.groupby(df[patient_col], sort=False).cumsum()
        no_event_sum = seen - event_sum
        if "history_HBP" in df.columns:
            df["history_HBP"] = event_sum
        if "history_LBP_times_0" in df.columns:
            df["history_LBP_times_0"] = no_event_sum
        if "history_HBP_rate" in df.columns:
            df["history_HBP_rate"] = (event_sum / seen_nonzero).fillna(0)
        if "history_LBP_times_0_rate" in df.columns:
            df["history_LBP_times_0_rate"] = (no_event_sum / seen_nonzero).fillna(0)

    for suffix, src in [(str(i), "降幅时间点比值区间") for i in range(1, 5)]:
        count_col = f"history_LBP_times_{suffix}"
        col = f"history_LBP_times_{suffix}_rate"
        if src in df.columns and (count_col in df.columns or col in df.columns):
            prev = (df[src] == int(suffix)).groupby(df[patient_col], sort=False).shift(1).fillna(False)
            count = prev.astype(int).groupby(df[patient_col], sort=False).cumsum()
            if count_col in df.columns:
                df[count_col] = count
            if col in df.columns:
                df[col] = (count / seen_nonzero).fillna(0)

    df.to_csv(path, index=False)


def main():
    for path in [Path("data/processed/深医_final_data.csv"), Path("data/processed/福鼎_final_data.csv")]:
        fix_history(path)
        print(path)


if __name__ == "__main__":
    main()
