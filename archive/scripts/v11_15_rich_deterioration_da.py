"""
v11.15 enrich recent deterioration features and train TabTransformer + DA.
"""

import copy
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import v11_5_feature_expansion_da as base
import v11_6_trend_context_da as featmod


CONFIG = dict(base.CONFIG)
CONFIG.update(
    {
        "results_path": "runs/v11_15_rich_deterioration_da_results.json",
        "seed": 42,
        "d_model": 64,
        "dropout": 0.15,
        "lr": 8e-4,
        "head_lr": 1e-4,
        "top_encoder_lr": 7e-5,
        "full_lr": 3e-5,
        "num_transformer_layers": 2,
        "num_heads": 4,
        "benchmark_batch_size": 512,
        "eval_batch_size": 1024,
    }
)
base.CONFIG.update(CONFIG)
featmod.CONFIG.update(CONFIG)


DETERIORATION_NUMERIC_COLS = [
    "透前收缩压",
    "透前动脉压",
    "透前体重",
    "超滤量MAX",
    "超滤率_绝对",
]


def _safe_numeric(df, col):
    return pd.to_numeric(df[col], errors="coerce")


def _rolling_cv(arr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size < 2:
        return 0.0
    mean = np.mean(arr)
    std = np.std(arr)
    if abs(mean) < 1e-6:
        return float(std)
    return float(std / (abs(mean) + 1e-6))


def _event_time_slope(arr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size < 2:
        return 0.0
    x = np.arange(arr.size, dtype=float)
    return float(np.polyfit(x, arr, 1)[0])


def add_rich_deterioration_features(df):
    df = featmod.add_recent_trend_features(df)
    df = df.sort_values(["患者id", "透析日期"]).copy()
    if "透析日期" in df.columns:
        df["透析日期"] = pd.to_datetime(df["透析日期"], errors="coerce")

    grp = df["患者id"]
    event = pd.to_numeric(df[CONFIG["event_col"]], errors="coerce").fillna(0)
    et = pd.to_numeric(df[CONFIG["duration_col"]], errors="coerce")
    shifted_event = event.groupby(grp).shift(1)
    shifted_et = et.groupby(grp).shift(1)
    shifted_et_event = et.where(event == 1).groupby(grp).shift(1)

    df["recent3_event_earliest"] = (
        shifted_et_event.groupby(grp)
        .rolling(3, min_periods=1)
        .min()
        .reset_index(level=0, drop=True)
        .fillna(0.0)
    )
    df["recent3_event_std"] = (
        shifted_et_event.groupby(grp)
        .rolling(3, min_periods=2)
        .std()
        .reset_index(level=0, drop=True)
        .fillna(0.0)
    )
    df["recent5_event_std"] = (
        shifted_et_event.groupby(grp)
        .rolling(5, min_periods=2)
        .std()
        .reset_index(level=0, drop=True)
        .fillna(0.0)
    )
    df["recent3_event_time_slope"] = (
        shifted_et_event.groupby(grp)
        .rolling(3, min_periods=2)
        .apply(_event_time_slope, raw=True)
        .reset_index(level=0, drop=True)
        .fillna(0.0)
    )
    df["recent1_event_early_flag"] = (
        ((shifted_event == 1) & (shifted_et <= 120)).astype(float).fillna(0.0)
    )
    df["recent1_event_very_early_flag"] = (
        ((shifted_event == 1) & (shifted_et <= 60)).astype(float).fillna(0.0)
    )
    df["recent3_event_earlier_than_mean"] = (
        (
            pd.to_numeric(df["recent3_event_earliest"], errors="coerce")
            < pd.to_numeric(df["recent3_mean_et_event"], errors="coerce")
        )
        .astype(float)
        .fillna(0.0)
    )

    for col in DETERIORATION_NUMERIC_COLS:
        series = _safe_numeric(df, col)
        shifted = series.groupby(grp).shift(1)
        df[f"recent3_{col}_std"] = (
            shifted.groupby(grp)
            .rolling(3, min_periods=2)
            .std()
            .reset_index(level=0, drop=True)
            .fillna(0.0)
        )
        df[f"recent3_{col}_cv"] = (
            shifted.groupby(grp)
            .rolling(3, min_periods=2)
            .apply(_rolling_cv, raw=True)
            .reset_index(level=0, drop=True)
            .fillna(0.0)
        )
        df[f"recent3_{col}_range"] = (
            shifted.groupby(grp)
            .rolling(3, min_periods=2)
            .apply(lambda x: float(np.nanmax(x) - np.nanmin(x)), raw=True)
            .reset_index(level=0, drop=True)
            .fillna(0.0)
        )

    return df


def build_feature_tables():
    df_source = pd.read_csv(CONFIG["source_data"])
    df_target = pd.read_csv(CONFIG["target_data"])

    df_source = add_rich_deterioration_features(df_source)
    df_target = add_rich_deterioration_features(df_target)

    feat_source = base.build_master_feature_frame(df_source)
    feat_target = base.build_master_feature_frame(df_target)

    for col in featmod.EXTRA_CONTEXT_NUMERIC:
        if col in df_source.columns and col in df_target.columns:
            feat_source[col] = pd.to_numeric(df_source[col], errors="coerce")
            feat_target[col] = pd.to_numeric(df_target[col], errors="coerce")

    extra_cols = [
        col
        for col in df_source.columns
        if col.startswith("recent")
        or col.endswith("_std")
        or col.endswith("_cv")
        or col.endswith("_range")
    ]
    for col in extra_cols:
        if col in df_target.columns:
            feat_source[col] = pd.to_numeric(df_source[col], errors="coerce")
            feat_target[col] = pd.to_numeric(df_target[col], errors="coerce")

    for col in featmod.EXTRA_CONTEXT_CATEGORICAL:
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

    e_source = df_source[CONFIG["event_col"]].fillna(0).values.astype(np.int64)
    t_source = df_source[CONFIG["duration_col"]].fillna(0).values.astype(np.float32)
    e_target = df_target[CONFIG["event_col"]].fillna(0).values.astype(np.int64)
    t_target = df_target[CONFIG["duration_col"]].fillna(0).values.astype(np.float32)
    return feat_source, feat_target, e_source, t_source, e_target, t_target, common_cols


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
    x_source = scaler.fit_transform(source_df.values).astype(np.float32)
    x_target = scaler.transform(target_df.values).astype(np.float32)
    return x_source, x_target


class TabTransformerEncoderNet(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout):
        super().__init__()
        self.feature_weight = nn.Parameter(torch.randn(input_dim, d_model) * 0.02)
        self.feature_bias = nn.Parameter(torch.zeros(input_dim, d_model))
        self.feature_pos = nn.Parameter(torch.randn(input_dim, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.hazard_head = nn.Linear(d_model, 1)

    def forward(self, x):
        tokens = x.unsqueeze(-1) * self.feature_weight.unsqueeze(0) + self.feature_bias.unsqueeze(0)
        tokens = tokens + self.feature_pos.unsqueeze(0)
        encoded = self.transformer(tokens)
        emb = self.norm(encoded.mean(dim=1))
        hazard = self.hazard_head(emb)
        return emb, hazard


def evaluate_cindex(model, x_eval, e_eval, t_eval):
    model.eval()
    hazards = []
    with torch.no_grad():
        for start in range(0, len(x_eval), CONFIG["eval_batch_size"]):
            end = start + CONFIG["eval_batch_size"]
            x_batch = torch.tensor(x_eval[start:end], dtype=torch.float32, device=CONFIG["device"])
            _, hazard = model(x_batch)
            hazards.append(hazard.squeeze(-1).detach().cpu().numpy())
    risk_scores = np.concatenate(hazards, axis=0)
    return concordance_index(t_eval, -risk_scores, e_eval)


def set_trainable_state(model, phase):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.hazard_head.parameters():
        param.requires_grad = True
    if phase in {"partial_unfreeze", "full_finetune"}:
        for param in model.transformer.layers[-1].parameters():
            param.requires_grad = True
    if phase == "full_finetune":
        model.feature_weight.requires_grad = True
        model.feature_bias.requires_grad = True
        model.feature_pos.requires_grad = True
        for param in model.transformer.parameters():
            param.requires_grad = True
        for param in model.norm.parameters():
            param.requires_grad = True


def build_optimizer(model, phase):
    if phase == "head_only":
        return optim.AdamW(model.hazard_head.parameters(), lr=CONFIG["head_lr"], weight_decay=1e-3)
    if phase == "partial_unfreeze":
        groups = [
            {"params": model.hazard_head.parameters(), "lr": CONFIG["head_lr"]},
            {"params": model.transformer.layers[-1].parameters(), "lr": CONFIG["top_encoder_lr"]},
        ]
        return optim.AdamW(groups, weight_decay=1e-3)
    if phase == "full_finetune":
        groups = [
            {"params": model.hazard_head.parameters(), "lr": CONFIG["head_lr"]},
            {"params": model.transformer.parameters(), "lr": CONFIG["full_lr"]},
            {"params": [model.feature_weight, model.feature_bias, model.feature_pos], "lr": CONFIG["full_lr"]},
            {"params": model.norm.parameters(), "lr": CONFIG["full_lr"]},
        ]
        return optim.AdamW(groups, weight_decay=1e-4)
    raise ValueError(phase)


def run_source_pretrain(model, source_loader, x_val, e_val, t_val):
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-4)
    best_val, best_epoch = -np.inf, 0
    best_state = copy.deepcopy(model.state_dict())
    patience_left = CONFIG["pretrain_patience"]
    for epoch in range(1, CONFIG["pretrain_max_epochs"] + 1):
        model.train()
        for x_s, e_s, t_s in source_loader:
            x_s = x_s.to(CONFIG["device"])
            e_s = e_s.to(CONFIG["device"])
            t_s = t_s.to(CONFIG["device"])
            optimizer.zero_grad()
            _, hazard = model(x_s)
            loss = base.cox_loss(hazard, e_s, t_s)
            loss.backward()
            optimizer.step()
        val_cindex = evaluate_cindex(model, x_val, e_val, t_val)
        if val_cindex > best_val:
            best_val, best_epoch = val_cindex, epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_left = CONFIG["pretrain_patience"]
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    model.load_state_dict(best_state)
    return best_val, best_epoch, best_state


def run_head_only(model, target_loader, x_val, e_val, t_val):
    set_trainable_state(model, "head_only")
    optimizer = build_optimizer(model, "head_only")
    best_val, best_epoch = evaluate_cindex(model, x_val, e_val, t_val), 0
    best_state = copy.deepcopy(model.state_dict())
    patience_left = CONFIG["head_only_patience"]
    for epoch in range(1, CONFIG["head_only_max_epochs"] + 1):
        model.train()
        for x_t, e_t, t_t in target_loader:
            x_t = x_t.to(CONFIG["device"])
            e_t = e_t.to(CONFIG["device"])
            t_t = t_t.to(CONFIG["device"])
            optimizer.zero_grad()
            _, hazard = model(x_t)
            loss = base.cox_loss(hazard, e_t, t_t)
            loss.backward()
            optimizer.step()
        val_cindex = evaluate_cindex(model, x_val, e_val, t_val)
        if val_cindex > best_val:
            best_val, best_epoch = val_cindex, epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_left = CONFIG["head_only_patience"]
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    model.load_state_dict(best_state)
    return best_val, best_epoch, best_state


def run_da(model, source_loader, target_loader, x_val, e_val, t_val):
    phase_results = []
    for phase, max_epochs, patience in [
        ("head_only", CONFIG["head_only_max_epochs"], CONFIG["head_only_patience"]),
        ("partial_unfreeze", CONFIG["partial_unfreeze_max_epochs"], CONFIG["partial_unfreeze_patience"]),
        ("full_finetune", CONFIG["full_finetune_max_epochs"], CONFIG["full_finetune_patience"]),
    ]:
        set_trainable_state(model, phase)
        optimizer = build_optimizer(model, phase)
        best_val, best_epoch = evaluate_cindex(model, x_val, e_val, t_val), 0
        best_state = copy.deepcopy(model.state_dict())
        patience_left = patience
        for epoch in range(1, max_epochs + 1):
            model.train()
            source_iter = iter(source_loader)
            for x_t, e_t, t_t in target_loader:
                try:
                    x_s, e_s, t_s = next(source_iter)
                except StopIteration:
                    source_iter = iter(source_loader)
                    x_s, e_s, t_s = next(source_iter)
                x_t = x_t.to(CONFIG["device"])
                e_t = e_t.to(CONFIG["device"])
                t_t = t_t.to(CONFIG["device"])
                x_s = x_s.to(CONFIG["device"])
                e_s = e_s.to(CONFIG["device"])
                t_s = t_s.to(CONFIG["device"])
                optimizer.zero_grad()
                emb_t, hazard_t = model(x_t)
                emb_s, hazard_s = model(x_s)
                target_loss = base.cox_loss(hazard_t, e_t, t_t)
                replay_loss = base.cox_loss(hazard_s, e_s, t_s)
                align_loss = base.conditional_alignment_loss(emb_s, e_s, emb_t, e_t)
                loss = target_loss + CONFIG["source_replay_weight"] * replay_loss + CONFIG["conditional_align_weight"] * align_loss
                loss.backward()
                optimizer.step()
            val_cindex = evaluate_cindex(model, x_val, e_val, t_val)
            if val_cindex > best_val:
                best_val, best_epoch = val_cindex, epoch
                best_state = copy.deepcopy(model.state_dict())
                patience_left = patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break
        model.load_state_dict(best_state)
        phase_results.append({"phase": phase, "best_adapt_val_cindex": best_val, "best_epoch": best_epoch})
    return phase_results


def main():
    base.set_seed(CONFIG["seed"])
    print("Loading v11.15 rich deterioration feature tables...")
    feat_source, feat_target, e_source, t_source, e_target, t_target, feature_names = build_feature_tables()
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

    source_loader = base.make_loader(x_source, e_source, t_source, CONFIG["benchmark_batch_size"], shuffle=True, drop_last=True)
    target_loader = base.make_loader(x_train, e_train, t_train, CONFIG["benchmark_batch_size"], shuffle=True, drop_last=False)

    model = TabTransformerEncoderNet(
        input_dim=x_source.shape[1],
        d_model=CONFIG["d_model"],
        nhead=CONFIG["num_heads"],
        num_layers=CONFIG["num_transformer_layers"],
        dropout=CONFIG["dropout"],
    ).to(CONFIG["device"])
    pre_val, pre_epoch, source_state = run_source_pretrain(model, source_loader, x_val, e_val, t_val)
    zero_shot_test = evaluate_cindex(model, x_test, e_test, t_test)

    head_model = TabTransformerEncoderNet(
        input_dim=x_source.shape[1],
        d_model=CONFIG["d_model"],
        nhead=CONFIG["num_heads"],
        num_layers=CONFIG["num_transformer_layers"],
        dropout=CONFIG["dropout"],
    ).to(CONFIG["device"])
    head_model.load_state_dict(source_state)
    head_val, head_epoch, _ = run_head_only(head_model, target_loader, x_val, e_val, t_val)
    head_test = evaluate_cindex(head_model, x_test, e_test, t_test)

    da_model = TabTransformerEncoderNet(
        input_dim=x_source.shape[1],
        d_model=CONFIG["d_model"],
        nhead=CONFIG["num_heads"],
        num_layers=CONFIG["num_transformer_layers"],
        dropout=CONFIG["dropout"],
    ).to(CONFIG["device"])
    da_model.load_state_dict(source_state)
    phase_results = run_da(da_model, source_loader, target_loader, x_val, e_val, t_val)
    da_test = evaluate_cindex(da_model, x_test, e_test, t_test)

    results = {
        "config": CONFIG,
        "feature_count": int(x_source.shape[1]),
        "feature_names": feature_names,
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
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"Saved results to {CONFIG['results_path']}")


if __name__ == "__main__":
    main()
