"""
v11.6 direction-one DA with corrected feature construction, recent trend features,
and treatment/context covariates.
"""

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import v11_5_feature_expansion_da as base


CONFIG = dict(base.CONFIG)
CONFIG.update(
    {
        "results_path": "runs/v11_6_trend_context_da_results.json",
        "seed": 42,
    }
)
base.CONFIG.update(CONFIG)


EXTRA_CONTEXT_NUMERIC = [
    "透析液钙浓度",
    "透析液电导率",
    "透前呼吸频率",
]

EXTRA_CONTEXT_CATEGORICAL = [
    "瘘管类型",
    "瘘管位置",
    "抗凝剂类型",
    "透析方式",
]

TREND_NUMERIC_COLS = [
    "透前收缩压",
    "超滤量MAX",
    "超滤率_绝对",
    "透前体重",
    "透前动脉压",
]


def _slope_last(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size < 2:
        return 0.0
    x = np.arange(arr.size, dtype=float)
    return float(np.polyfit(x, arr, 1)[0])


def add_recent_trend_features(df):
    df = df.sort_values(["患者id", "透析日期"]).copy()
    if "透析日期" in df.columns:
        df["透析日期"] = pd.to_datetime(df["透析日期"], errors="coerce")

    event = pd.to_numeric(df[CONFIG["event_col"]], errors="coerce").fillna(0)
    et = pd.to_numeric(df[CONFIG["duration_col"]], errors="coerce")
    grp = df["患者id"]

    shifted_event = event.groupby(grp).shift(1)
    df["recent1_idh"] = shifted_event.fillna(0.0)
    df["recent3_idh_rate"] = (
        shifted_event.groupby(grp)
        .rolling(3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
        .fillna(0.0)
    )
    df["recent5_idh_rate"] = (
        shifted_event.groupby(grp)
        .rolling(5, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
        .fillna(0.0)
    )

    shifted_et_event = et.where(event == 1).groupby(grp).shift(1)
    df["recent3_mean_et_event"] = (
        shifted_et_event.groupby(grp)
        .rolling(3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
        .fillna(0.0)
    )
    df["recent5_mean_et_event"] = (
        shifted_et_event.groupby(grp)
        .rolling(5, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
        .fillna(0.0)
    )

    for col in TREND_NUMERIC_COLS:
        series = pd.to_numeric(df[col], errors="coerce")
        shifted = series.groupby(grp).shift(1)
        df[f"recent3_{col}_mean"] = (
            shifted.groupby(grp)
            .rolling(3, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
            .fillna(0.0)
        )
        df[f"recent3_{col}_slope"] = (
            shifted.groupby(grp)
            .rolling(3, min_periods=2)
            .apply(_slope_last, raw=True)
            .reset_index(level=0, drop=True)
            .fillna(0.0)
        )

    return df


def build_feature_tables():
    df_source = pd.read_csv(CONFIG["source_data"])
    df_target = pd.read_csv(CONFIG["target_data"])

    df_source = add_recent_trend_features(df_source)
    df_target = add_recent_trend_features(df_target)

    feat_source = base.build_master_feature_frame(df_source)
    feat_target = base.build_master_feature_frame(df_target)

    for col in EXTRA_CONTEXT_NUMERIC:
        if col in df_source.columns and col in df_target.columns:
            feat_source[col] = pd.to_numeric(df_source[col], errors="coerce")
            feat_target[col] = pd.to_numeric(df_target[col], errors="coerce")

    trend_cols = [col for col in df_source.columns if col.startswith("recent")]
    for col in trend_cols:
        if col in df_target.columns:
            feat_source[col] = pd.to_numeric(df_source[col], errors="coerce")
            feat_target[col] = pd.to_numeric(df_target[col], errors="coerce")

    for col in EXTRA_CONTEXT_CATEGORICAL:
        if col in df_source.columns and col in df_target.columns:
            combined = pd.concat(
                [
                    df_source[col].fillna("UNK").astype(str),
                    df_target[col].fillna("UNK").astype(str),
                ],
                axis=0,
            )
            categories = pd.Categorical(combined)
            src_codes = categories.codes[: len(df_source)]
            tgt_codes = categories.codes[len(df_source) :]
            feat_source[col + "_code"] = src_codes.astype(float)
            feat_target[col + "_code"] = tgt_codes.astype(float)

    common_cols = sorted(set(feat_source.columns).intersection(feat_target.columns))
    feat_source = feat_source[common_cols].astype(np.float32)
    feat_target = feat_target[common_cols].astype(np.float32)

    y_source_e = df_source[CONFIG["event_col"]].fillna(0).values.astype(np.int64)
    y_source_t = df_source[CONFIG["duration_col"]].fillna(0).values.astype(np.float32)
    y_target_e = df_target[CONFIG["event_col"]].fillna(0).values.astype(np.int64)
    y_target_t = df_target[CONFIG["duration_col"]].fillna(0).values.astype(np.float32)

    return (
        feat_source,
        feat_target,
        y_source_e,
        y_source_t,
        y_target_e,
        y_target_t,
        common_cols,
    )


def prepare_group_matrices(feat_source, feat_target):
    source_df = feat_source.copy()
    target_df = feat_target.copy()
    for col in source_df.columns:
        lower = source_df[col].quantile(0.01)
        upper = source_df[col].quantile(0.99)
        source_df[col] = source_df[col].clip(lower=lower, upper=upper)
        target_df[col] = target_df[col].clip(lower=lower, upper=upper)

    source_df = source_df.fillna(source_df.median()).fillna(0.0)
    target_df = target_df.fillna(source_df.median()).fillna(0.0)

    scaler = StandardScaler()
    x_source = scaler.fit_transform(source_df.values)
    x_target = scaler.transform(target_df.values)
    return x_source, x_target


def main():
    base.set_seed(CONFIG["seed"])
    print("Loading v11.6 feature tables...")
    feat_source, feat_target, e_source, t_source, e_target, t_target, feature_names = (
        build_feature_tables()
    )
    x_source, x_target = prepare_group_matrices(feat_source, feat_target)

    x_adapt_pool, x_test, e_adapt_pool, e_test, t_adapt_pool, t_test = train_test_split(
        x_target,
        e_target,
        t_target,
        test_size=1 - CONFIG["target_adapt_ratio"],
        random_state=CONFIG["seed"],
        stratify=e_target,
    )
    x_train, x_val, e_train, e_val, t_train, t_val = train_test_split(
        x_adapt_pool,
        e_adapt_pool,
        t_adapt_pool,
        test_size=CONFIG["target_val_ratio_within_adapt"],
        random_state=CONFIG["seed"],
        stratify=e_adapt_pool,
    )

    source_loader = base.make_loader(
        x_source, e_source, t_source, CONFIG["batch_size"], shuffle=True, drop_last=True
    )
    target_train_loader = base.make_loader(
        x_train, e_train, t_train, CONFIG["batch_size"], shuffle=True, drop_last=False
    )

    input_dim = x_source.shape[1]
    print(f"Feature count: {input_dim}")

    base_model = base.SurvivalNet(input_dim, CONFIG["d_model"], CONFIG["dropout"]).to(
        CONFIG["device"]
    )
    pre_val, pre_epoch, source_state = base.run_source_pretrain(
        base_model, source_loader, x_val, e_val, t_val
    )
    zero_shot_test = base.evaluate_cindex(base_model, x_test, e_test, t_test)

    head_model = base.SurvivalNet(input_dim, CONFIG["d_model"], CONFIG["dropout"]).to(
        CONFIG["device"]
    )
    head_model.load_state_dict(source_state)
    head_val, head_epoch, _ = base.run_head_only_baseline(
        head_model, target_train_loader, x_val, e_val, t_val
    )
    head_test = base.evaluate_cindex(head_model, x_test, e_test, t_test)

    da_model = base.SurvivalNet(input_dim, CONFIG["d_model"], CONFIG["dropout"]).to(
        CONFIG["device"]
    )
    da_model.load_state_dict(source_state)
    phase_results = []
    for phase, max_epochs, patience in [
        ("head_only", CONFIG["head_only_max_epochs"], CONFIG["head_only_patience"]),
        (
            "partial_unfreeze",
            CONFIG["partial_unfreeze_max_epochs"],
            CONFIG["partial_unfreeze_patience"],
        ),
        (
            "full_finetune",
            CONFIG["full_finetune_max_epochs"],
            CONFIG["full_finetune_patience"],
        ),
    ]:
        val_score, best_epoch, _ = base.run_da_phase(
            da_model,
            phase,
            max_epochs,
            patience,
            source_loader,
            target_train_loader,
            x_val,
            e_val,
            t_val,
        )
        phase_results.append(
            {
                "phase": phase,
                "best_adapt_val_cindex": val_score,
                "best_epoch": best_epoch,
            }
        )
    da_test = base.evaluate_cindex(da_model, x_test, e_test, t_test)

    results = {
        "config": CONFIG,
        "feature_count": int(input_dim),
        "feature_names": feature_names,
        "split": {
            "source_n": int(len(x_source)),
            "adapt_train_n": int(len(x_train)),
            "adapt_val_n": int(len(x_val)),
            "test_n": int(len(x_test)),
            "source_event_rate": float(e_source.mean()),
            "adapt_train_event_rate": float(e_train.mean()),
            "adapt_val_event_rate": float(e_val.mean()),
            "test_event_rate": float(e_test.mean()),
        },
        "source_pretrain": {
            "best_adapt_val_cindex": pre_val,
            "best_epoch": pre_epoch,
            "zero_shot_test_cindex": zero_shot_test,
        },
        "head_only": {
            "best_adapt_val_cindex": head_val,
            "best_epoch": head_epoch,
            "test_cindex": head_test,
        },
        "layerwise_da": {
            "phase_results": phase_results,
            "test_cindex": da_test,
            "gain_vs_zero_shot": da_test - zero_shot_test,
        },
    }

    with open(CONFIG["results_path"], "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(json.dumps(results["layerwise_da"], ensure_ascii=False, indent=2))
    print(f"Saved results to {CONFIG['results_path']}")


if __name__ == "__main__":
    main()
