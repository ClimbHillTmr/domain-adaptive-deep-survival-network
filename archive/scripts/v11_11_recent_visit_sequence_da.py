"""
v11.11 use v11.6 flat features plus recent-visit session sequence encoding.
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
        "results_path": "runs/v11_11_recent_visit_sequence_da_results.json",
        "seed": 42,
        "d_model": 64,
        "seq_d_model": 32,
        "dropout": 0.15,
        "lr": 8e-4,
        "head_lr": 1e-4,
        "top_encoder_lr": 7e-5,
        "full_lr": 3e-5,
        "num_transformer_layers": 2,
        "num_heads": 4,
        "seq_len": 3,
        "seq_encoder_type": "gru",
        "benchmark_batch_size": 512,
        "eval_batch_size": 1024,
    }
)
base.CONFIG.update(CONFIG)
featmod.CONFIG.update(CONFIG)


VISIT_FEATURE_COLS = [
    "透中低血压_计算",
    "et_min",
    "透前收缩压",
    "透前舒张压",
    "透前动脉压",
    "透前体重",
    "超滤量MAX",
    "超滤率_绝对",
]


def _safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def _build_visit_vector_frame(df):
    work = df.copy()
    work["透前收缩压"] = _safe_numeric(work.get("透前收缩压"))
    work["透前舒张压"] = _safe_numeric(work.get("透前舒张压"))
    work["透前动脉压"] = _safe_numeric(work.get("透前动脉压"))
    work["透前体重"] = _safe_numeric(work.get("透前体重"))
    work["干体重"] = _safe_numeric(work.get("干体重"))
    work["超滤量MAX"] = _safe_numeric(work.get("超滤量MAX"))
    work["超滤率_绝对"] = _safe_numeric(work.get("超滤率_绝对"))
    work["透中低血压_计算"] = _safe_numeric(work.get("透中低血压_计算")).fillna(0.0)
    work["et_min"] = _safe_numeric(work.get("et_min")).fillna(0.0)
    work["容量超负荷比"] = (
        (work["透前体重"] - work["干体重"]) / (work["干体重"].replace(0, np.nan))
    ).replace([np.inf, -np.inf], np.nan)
    work["平均动脉压"] = (
        work["透前收缩压"] + 2.0 * work["透前舒张压"]
    ) / 3.0
    work["脉压差"] = work["透前收缩压"] - work["透前舒张压"]
    return work


def build_recent_visit_sequences(df, seq_len):
    df = df.copy()
    df["透析日期"] = pd.to_datetime(df["透析日期"], errors="coerce")
    df = df.sort_values(["患者id", "透析日期"]).reset_index(drop=True)
    visit_df = _build_visit_vector_frame(df)

    seq_values = []
    seq_mask = []
    grouped = visit_df.groupby("患者id", sort=False)
    for _, idx in grouped.indices.items():
        idx = list(idx)
        patient_frame = visit_df.loc[idx, VISIT_FEATURE_COLS]
        patient_values = patient_frame.astype(float).values
        for local_i in range(len(idx)):
            hist = patient_values[max(0, local_i - seq_len) : local_i]
            padded = np.zeros((seq_len, len(VISIT_FEATURE_COLS)), dtype=np.float32)
            mask = np.zeros(seq_len, dtype=np.float32)
            if len(hist) > 0:
                padded[-len(hist) :] = hist
                mask[-len(hist) :] = 1.0
            seq_values.append(padded)
            seq_mask.append(mask)

    return (
        df,
        np.stack(seq_values).astype(np.float32),
        np.stack(seq_mask).astype(np.float32),
    )


def prepare_data():
    df_source = pd.read_csv(CONFIG["source_data"])
    df_target = pd.read_csv(CONFIG["target_data"])

    df_source = featmod.add_recent_trend_features(df_source)
    df_target = featmod.add_recent_trend_features(df_target)

    feat_source = base.build_master_feature_frame(df_source)
    feat_target = base.build_master_feature_frame(df_target)

    for col in featmod.EXTRA_CONTEXT_NUMERIC:
        if col in df_source.columns and col in df_target.columns:
            feat_source[col] = pd.to_numeric(df_source[col], errors="coerce")
            feat_target[col] = pd.to_numeric(df_target[col], errors="coerce")

    trend_cols = [col for col in df_source.columns if col.startswith("recent")]
    for col in trend_cols:
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

    source_df_sorted, seq_source, mask_source = build_recent_visit_sequences(df_source, CONFIG["seq_len"])
    target_df_sorted, seq_target, mask_target = build_recent_visit_sequences(df_target, CONFIG["seq_len"])

    source_df_for_feat = df_source.copy()
    source_df_for_feat["透析日期"] = pd.to_datetime(source_df_for_feat["透析日期"], errors="coerce")
    source_order = source_df_for_feat.sort_values(["患者id", "透析日期"]).index
    target_df_for_feat = df_target.copy()
    target_df_for_feat["透析日期"] = pd.to_datetime(target_df_for_feat["透析日期"], errors="coerce")
    target_order = target_df_for_feat.sort_values(["患者id", "透析日期"]).index

    feat_source = feat_source.iloc[source_order].reset_index(drop=True)
    feat_target = feat_target.iloc[target_order].reset_index(drop=True)

    e_source = source_df_sorted[CONFIG["event_col"]].fillna(0).astype(np.int64).values
    t_source = source_df_sorted[CONFIG["duration_col"]].fillna(0).astype(np.float32).values
    e_target = target_df_sorted[CONFIG["event_col"]].fillna(0).astype(np.int64).values
    t_target = target_df_sorted[CONFIG["duration_col"]].fillna(0).astype(np.float32).values

    source_flat = feat_source.copy()
    target_flat = feat_target.copy()
    for col in source_flat.columns:
        lower = source_flat[col].quantile(0.01)
        upper = source_flat[col].quantile(0.99)
        source_flat[col] = source_flat[col].clip(lower=lower, upper=upper)
        target_flat[col] = target_flat[col].clip(lower=lower, upper=upper)
    source_flat = source_flat.fillna(source_flat.median()).fillna(0.0)
    target_flat = target_flat.fillna(source_flat.median()).fillna(0.0)
    flat_scaler = StandardScaler()
    x_source_flat = flat_scaler.fit_transform(source_flat.values).astype(np.float32)
    x_target_flat = flat_scaler.transform(target_flat.values).astype(np.float32)

    seq_dim = seq_source.shape[-1]
    source_seq_2d = seq_source.reshape(-1, seq_dim)
    target_seq_2d = seq_target.reshape(-1, seq_dim)
    source_seq_df = pd.DataFrame(source_seq_2d, columns=VISIT_FEATURE_COLS)
    target_seq_df = pd.DataFrame(target_seq_2d, columns=VISIT_FEATURE_COLS)
    for col in VISIT_FEATURE_COLS:
        lower = source_seq_df[col].quantile(0.01)
        upper = source_seq_df[col].quantile(0.99)
        source_seq_df[col] = source_seq_df[col].clip(lower=lower, upper=upper)
        target_seq_df[col] = target_seq_df[col].clip(lower=lower, upper=upper)
    source_seq_df = source_seq_df.fillna(source_seq_df.median()).fillna(0.0)
    target_seq_df = target_seq_df.fillna(source_seq_df.median()).fillna(0.0)
    seq_scaler = StandardScaler()
    seq_source_scaled = seq_scaler.fit_transform(source_seq_df.values).reshape(seq_source.shape).astype(np.float32)
    seq_target_scaled = seq_scaler.transform(target_seq_df.values).reshape(seq_target.shape).astype(np.float32)
    seq_source_scaled = seq_source_scaled * mask_source[..., None]
    seq_target_scaled = seq_target_scaled * mask_target[..., None]

    return {
        "x_source_flat": x_source_flat,
        "x_target_flat": x_target_flat,
        "x_source_seq": seq_source_scaled,
        "x_target_seq": seq_target_scaled,
        "mask_source": mask_source.astype(np.float32),
        "mask_target": mask_target.astype(np.float32),
        "e_source": e_source,
        "t_source": t_source,
        "e_target": e_target,
        "t_target": t_target,
        "feature_names": common_cols,
        "visit_feature_names": VISIT_FEATURE_COLS,
    }


class VisitSequenceEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, seq_x, seq_mask):
        x = self.proj(seq_x)
        lengths = seq_mask.sum(dim=1).long()
        lengths_cpu = torch.clamp(lengths, min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        emb = h[-1]
        zero_hist = (lengths == 0).unsqueeze(1)
        emb = torch.where(zero_hist, torch.zeros_like(emb), emb)
        emb = self.norm(self.dropout(emb))
        return emb


class TabTransformerSeqNet(nn.Module):
    def __init__(self, input_dim, seq_input_dim, d_model, seq_d_model, nhead, num_layers, dropout):
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
        self.flat_norm = nn.LayerNorm(d_model)
        self.seq_encoder = VisitSequenceEncoder(seq_input_dim, seq_d_model, dropout)
        self.fusion = nn.Sequential(
            nn.Linear(d_model + seq_d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.hazard_head = nn.Linear(d_model, 1)

    def forward(self, x_flat, x_seq, seq_mask):
        tokens = x_flat.unsqueeze(-1) * self.feature_weight.unsqueeze(0) + self.feature_bias.unsqueeze(0)
        tokens = tokens + self.feature_pos.unsqueeze(0)
        flat_encoded = self.transformer(tokens)
        flat_emb = self.flat_norm(flat_encoded.mean(dim=1))
        seq_emb = self.seq_encoder(x_seq, seq_mask)
        emb = self.fusion(torch.cat([flat_emb, seq_emb], dim=1))
        hazard = self.hazard_head(emb)
        return emb, hazard


class MixedDataset(torch.utils.data.Dataset):
    def __init__(self, x_flat, x_seq, seq_mask, e, t):
        self.x_flat = torch.tensor(x_flat, dtype=torch.float32)
        self.x_seq = torch.tensor(x_seq, dtype=torch.float32)
        self.seq_mask = torch.tensor(seq_mask, dtype=torch.float32)
        self.e = torch.tensor(e, dtype=torch.float32)
        self.t = torch.tensor(t, dtype=torch.float32)

    def __len__(self):
        return len(self.e)

    def __getitem__(self, idx):
        return (
            self.x_flat[idx],
            self.x_seq[idx],
            self.seq_mask[idx],
            self.e[idx],
            self.t[idx],
        )


def make_mixed_loader(x_flat, x_seq, seq_mask, e, t, batch_size, shuffle, drop_last):
    ds = MixedDataset(x_flat, x_seq, seq_mask, e, t)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


def evaluate_cindex(model, x_flat, x_seq, seq_mask, e_eval, t_eval):
    model.eval()
    hazards = []
    with torch.no_grad():
        for start in range(0, len(x_flat), CONFIG["eval_batch_size"]):
            end = start + CONFIG["eval_batch_size"]
            xb = torch.tensor(x_flat[start:end], dtype=torch.float32, device=CONFIG["device"])
            xs = torch.tensor(x_seq[start:end], dtype=torch.float32, device=CONFIG["device"])
            xm = torch.tensor(seq_mask[start:end], dtype=torch.float32, device=CONFIG["device"])
            _, hazard = model(xb, xs, xm)
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
        for param in model.fusion.parameters():
            param.requires_grad = True
        for param in model.seq_encoder.parameters():
            param.requires_grad = True

    if phase == "full_finetune":
        model.feature_weight.requires_grad = True
        model.feature_bias.requires_grad = True
        model.feature_pos.requires_grad = True
        for param in model.transformer.parameters():
            param.requires_grad = True
        for param in model.flat_norm.parameters():
            param.requires_grad = True


def build_optimizer(model, phase):
    if phase == "head_only":
        param_groups = [
            {"params": model.hazard_head.parameters(), "lr": CONFIG["head_lr"]},
            {"params": model.fusion.parameters(), "lr": CONFIG["head_lr"]},
        ]
        return optim.AdamW(param_groups, weight_decay=1e-3)

    if phase == "partial_unfreeze":
        param_groups = [
            {"params": model.hazard_head.parameters(), "lr": CONFIG["head_lr"]},
            {"params": model.fusion.parameters(), "lr": CONFIG["head_lr"]},
            {"params": model.seq_encoder.parameters(), "lr": CONFIG["top_encoder_lr"]},
            {"params": model.transformer.layers[-1].parameters(), "lr": CONFIG["top_encoder_lr"]},
        ]
        return optim.AdamW(param_groups, weight_decay=1e-3)

    if phase == "full_finetune":
        param_groups = [
            {"params": model.hazard_head.parameters(), "lr": CONFIG["head_lr"]},
            {"params": model.fusion.parameters(), "lr": CONFIG["top_encoder_lr"]},
            {"params": model.seq_encoder.parameters(), "lr": CONFIG["full_lr"]},
            {"params": model.transformer.parameters(), "lr": CONFIG["full_lr"]},
            {"params": [model.feature_weight, model.feature_bias, model.feature_pos], "lr": CONFIG["full_lr"]},
            {"params": model.flat_norm.parameters(), "lr": CONFIG["full_lr"]},
        ]
        return optim.AdamW(param_groups, weight_decay=1e-4)

    raise ValueError(phase)


def run_source_pretrain(model, source_loader, x_val_flat, x_val_seq, x_val_mask, e_val, t_val):
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-4)
    best_val = -np.inf
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    patience_left = CONFIG["pretrain_patience"]

    for epoch in range(1, CONFIG["pretrain_max_epochs"] + 1):
        model.train()
        for xf, xs, xm, e_s, t_s in source_loader:
            xf = xf.to(CONFIG["device"])
            xs = xs.to(CONFIG["device"])
            xm = xm.to(CONFIG["device"])
            e_s = e_s.to(CONFIG["device"])
            t_s = t_s.to(CONFIG["device"])
            optimizer.zero_grad()
            _, hazard = model(xf, xs, xm)
            loss = base.cox_loss(hazard, e_s, t_s)
            loss.backward()
            optimizer.step()

        val_cindex = evaluate_cindex(model, x_val_flat, x_val_seq, x_val_mask, e_val, t_val)
        if val_cindex > best_val:
            best_val = val_cindex
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_left = CONFIG["pretrain_patience"]
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    model.load_state_dict(best_state)
    return best_val, best_epoch, best_state


def run_head_only_baseline(model, target_train_loader, x_val_flat, x_val_seq, x_val_mask, e_val, t_val):
    set_trainable_state(model, "head_only")
    optimizer = build_optimizer(model, "head_only")
    best_val = evaluate_cindex(model, x_val_flat, x_val_seq, x_val_mask, e_val, t_val)
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    patience_left = CONFIG["head_only_patience"]

    for epoch in range(1, CONFIG["head_only_max_epochs"] + 1):
        model.train()
        for xf, xs, xm, e_t, t_t in target_train_loader:
            xf = xf.to(CONFIG["device"])
            xs = xs.to(CONFIG["device"])
            xm = xm.to(CONFIG["device"])
            e_t = e_t.to(CONFIG["device"])
            t_t = t_t.to(CONFIG["device"])
            optimizer.zero_grad()
            _, hazard = model(xf, xs, xm)
            loss = base.cox_loss(hazard, e_t, t_t)
            loss.backward()
            optimizer.step()

        val_cindex = evaluate_cindex(model, x_val_flat, x_val_seq, x_val_mask, e_val, t_val)
        if val_cindex > best_val:
            best_val = val_cindex
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_left = CONFIG["head_only_patience"]
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    model.load_state_dict(best_state)
    return best_val, best_epoch, best_state


def run_da_phase(
    model,
    phase,
    max_epochs,
    patience,
    source_loader,
    target_train_loader,
    x_val_flat,
    x_val_seq,
    x_val_mask,
    e_val,
    t_val,
):
    set_trainable_state(model, phase)
    optimizer = build_optimizer(model, phase)
    best_val = evaluate_cindex(model, x_val_flat, x_val_seq, x_val_mask, e_val, t_val)
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    patience_left = patience

    for epoch in range(1, max_epochs + 1):
        model.train()
        source_iter = iter(source_loader)
        for xf_t, xs_t, xm_t, e_t, t_t in target_train_loader:
            try:
                xf_s, xs_s, xm_s, e_s, t_s = next(source_iter)
            except StopIteration:
                source_iter = iter(source_loader)
                xf_s, xs_s, xm_s, e_s, t_s = next(source_iter)

            xf_t = xf_t.to(CONFIG["device"])
            xs_t = xs_t.to(CONFIG["device"])
            xm_t = xm_t.to(CONFIG["device"])
            e_t = e_t.to(CONFIG["device"])
            t_t = t_t.to(CONFIG["device"])
            xf_s = xf_s.to(CONFIG["device"])
            xs_s = xs_s.to(CONFIG["device"])
            xm_s = xm_s.to(CONFIG["device"])
            e_s = e_s.to(CONFIG["device"])
            t_s = t_s.to(CONFIG["device"])

            optimizer.zero_grad()
            emb_t, hazard_t = model(xf_t, xs_t, xm_t)
            emb_s, hazard_s = model(xf_s, xs_s, xm_s)
            target_loss = base.cox_loss(hazard_t, e_t, t_t)
            replay_loss = base.cox_loss(hazard_s, e_s, t_s)
            align_loss = base.conditional_alignment_loss(emb_s, e_s, emb_t, e_t)
            loss = target_loss + CONFIG["source_replay_weight"] * replay_loss + CONFIG["conditional_align_weight"] * align_loss
            loss.backward()
            optimizer.step()

        val_cindex = evaluate_cindex(model, x_val_flat, x_val_seq, x_val_mask, e_val, t_val)
        if val_cindex > best_val:
            best_val = val_cindex
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    model.load_state_dict(best_state)
    return best_val, best_epoch, best_state


def main():
    base.set_seed(CONFIG["seed"])
    print("Loading v11.11 flat + recent visit sequence features...")
    data = prepare_data()

    target_indices = np.arange(len(data["e_target"]))
    adapt_idx, test_idx = train_test_split(
        target_indices,
        test_size=1 - CONFIG["target_adapt_ratio"],
        random_state=CONFIG["seed"],
        stratify=data["e_target"],
    )
    train_idx, val_idx = train_test_split(
        adapt_idx,
        test_size=CONFIG["target_val_ratio_within_adapt"],
        random_state=CONFIG["seed"],
        stratify=data["e_target"][adapt_idx],
    )

    source_loader = make_mixed_loader(
        data["x_source_flat"],
        data["x_source_seq"],
        data["mask_source"],
        data["e_source"],
        data["t_source"],
        CONFIG["benchmark_batch_size"],
        shuffle=True,
        drop_last=True,
    )
    target_train_loader = make_mixed_loader(
        data["x_target_flat"][train_idx],
        data["x_target_seq"][train_idx],
        data["mask_target"][train_idx],
        data["e_target"][train_idx],
        data["t_target"][train_idx],
        CONFIG["benchmark_batch_size"],
        shuffle=True,
        drop_last=False,
    )

    model = TabTransformerSeqNet(
        input_dim=data["x_source_flat"].shape[1],
        seq_input_dim=data["x_source_seq"].shape[2],
        d_model=CONFIG["d_model"],
        seq_d_model=CONFIG["seq_d_model"],
        nhead=CONFIG["num_heads"],
        num_layers=CONFIG["num_transformer_layers"],
        dropout=CONFIG["dropout"],
    ).to(CONFIG["device"])

    pre_val, pre_epoch, source_state = run_source_pretrain(
        model,
        source_loader,
        data["x_target_flat"][val_idx],
        data["x_target_seq"][val_idx],
        data["mask_target"][val_idx],
        data["e_target"][val_idx],
        data["t_target"][val_idx],
    )
    zero_shot_test = evaluate_cindex(
        model,
        data["x_target_flat"][test_idx],
        data["x_target_seq"][test_idx],
        data["mask_target"][test_idx],
        data["e_target"][test_idx],
        data["t_target"][test_idx],
    )

    head_model = TabTransformerSeqNet(
        input_dim=data["x_source_flat"].shape[1],
        seq_input_dim=data["x_source_seq"].shape[2],
        d_model=CONFIG["d_model"],
        seq_d_model=CONFIG["seq_d_model"],
        nhead=CONFIG["num_heads"],
        num_layers=CONFIG["num_transformer_layers"],
        dropout=CONFIG["dropout"],
    ).to(CONFIG["device"])
    head_model.load_state_dict(source_state)
    head_val, head_epoch, _ = run_head_only_baseline(
        head_model,
        target_train_loader,
        data["x_target_flat"][val_idx],
        data["x_target_seq"][val_idx],
        data["mask_target"][val_idx],
        data["e_target"][val_idx],
        data["t_target"][val_idx],
    )
    head_test = evaluate_cindex(
        head_model,
        data["x_target_flat"][test_idx],
        data["x_target_seq"][test_idx],
        data["mask_target"][test_idx],
        data["e_target"][test_idx],
        data["t_target"][test_idx],
    )

    da_model = TabTransformerSeqNet(
        input_dim=data["x_source_flat"].shape[1],
        seq_input_dim=data["x_source_seq"].shape[2],
        d_model=CONFIG["d_model"],
        seq_d_model=CONFIG["seq_d_model"],
        nhead=CONFIG["num_heads"],
        num_layers=CONFIG["num_transformer_layers"],
        dropout=CONFIG["dropout"],
    ).to(CONFIG["device"])
    da_model.load_state_dict(source_state)
    phase_results = []
    for phase, max_epochs, patience in [
        ("head_only", CONFIG["head_only_max_epochs"], CONFIG["head_only_patience"]),
        ("partial_unfreeze", CONFIG["partial_unfreeze_max_epochs"], CONFIG["partial_unfreeze_patience"]),
        ("full_finetune", CONFIG["full_finetune_max_epochs"], CONFIG["full_finetune_patience"]),
    ]:
        val_score, best_epoch, _ = run_da_phase(
            da_model,
            phase,
            max_epochs,
            patience,
            source_loader,
            target_train_loader,
            data["x_target_flat"][val_idx],
            data["x_target_seq"][val_idx],
            data["mask_target"][val_idx],
            data["e_target"][val_idx],
            data["t_target"][val_idx],
        )
        phase_results.append(
            {"phase": phase, "best_adapt_val_cindex": val_score, "best_epoch": best_epoch}
        )

    da_test = evaluate_cindex(
        da_model,
        data["x_target_flat"][test_idx],
        data["x_target_seq"][test_idx],
        data["mask_target"][test_idx],
        data["e_target"][test_idx],
        data["t_target"][test_idx],
    )

    results = {
        "config": CONFIG,
        "feature_count": int(data["x_source_flat"].shape[1]),
        "visit_feature_count": int(data["x_source_seq"].shape[2]),
        "visit_feature_names": VISIT_FEATURE_COLS,
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
