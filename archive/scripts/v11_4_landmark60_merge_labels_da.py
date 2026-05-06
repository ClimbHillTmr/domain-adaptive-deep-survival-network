"""
v11.4 landmark-60 clean validation with raw dynamic features + optimized labels.

Data source:
- data_preprocessing/updated_dataset_shenyi.csv
- data_preprocessing/updated_dataset_fuding.csv

Key constraints:
1. Dynamic features must be computed only from <=60 min intradialytic observations.
2. Dynamic features come from raw upstream sequences.
3. Labels and event times come from optimized datasets.
4. Landmark task: only sessions still at risk at 60 min are kept.
5. Clean target split: adapt_train / adapt_val / test.
6. Early stopping based on adapt_val only.
"""

import ast
import copy
import json
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")


CONFIG = {
    "source_raw_data": "data_preprocessing/updated_dataset_shenyi.csv",
    "target_raw_data": "data_preprocessing/updated_dataset_fuding.csv",
    "source_label_data": "data_preprocessing/data/深医_optimized_data透前动脉压.csv",
    "target_label_data": "data_preprocessing/data/福鼎_optimized_data透前动脉压.csv",
    "event_col": "透中低血压_计算",
    "duration_col": "et_min",
    "batch_size": 512,
    "pretrain_max_epochs": 24,
    "pretrain_patience": 3,
    "head_only_max_epochs": 3,
    "head_only_patience": 1,
    "partial_unfreeze_max_epochs": 5,
    "partial_unfreeze_patience": 2,
    "full_finetune_max_epochs": 12,
    "full_finetune_patience": 3,
    "lr": 1e-3,
    "head_lr": 1e-4,
    "top_encoder_lr": 5e-5,
    "full_lr": 2e-5,
    "d_model": 64,
    "dropout": 0.2,
    "target_adapt_ratio": 0.2,
    "target_val_ratio_within_adapt": 0.5,
    "source_replay_weight": 0.2,
    "conditional_align_weight": 0.1,
    "obs_minutes": 60.0,
    "landmark_minutes": 60.0,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
    "results_path": "runs/v11_4_landmark60_merge_labels_da_results.json",
}


SEQ_CANDIDATES = {
    "sbp": ["透析中收缩压", "透中收缩压", "收缩压"],
    "dbp": ["透析中舒张压", "透中舒张压", "舒张压"],
    "pulse": ["透析中脉搏", "脉搏", "心率"],
    "ufr": ["超滤率"],
    "venous": ["静脉压"],
    "blood_flow": ["血流速"],
    "dialysate_temp": ["透析液温度"],
}


STATIC_CANDIDATES = [
    "透析年龄",
    "性别",
    "透析龄",
    "透前收缩压",
    "透前舒张压",
    "透前体重",
    "干体重",
    "高血压诊断",
]


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
            parsed = None
        if isinstance(parsed, list):
            seq = parsed
        elif parsed is not None and isinstance(parsed, (int, float)):
            seq = [parsed]
        else:
            # Support plain comma-separated strings without brackets.
            pieces = [p.strip() for p in s.split(",") if p.strip()]
            seq = pieces
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


def extract_minutes(row):
    if "minutes_from_start_list" in row:
        raw_minutes = row["minutes_from_start_list"]
        if isinstance(raw_minutes, list):
            mins = to_float_list(raw_minutes)
        elif pd.notna(raw_minutes):
            mins = to_float_list(raw_minutes)
        else:
            mins = []
        if mins:
            return mins

    if "透中数据记录时间节点" in row:
        ts = row["透中数据记录时间节点"]
        if not isinstance(ts, list) and pd.isna(ts):
            return []
        if isinstance(ts, str):
            try:
                parsed = ast.literal_eval(ts)
            except Exception:
                cleaned = ts.strip().strip("[]")
                parsed = [p.strip() for p in cleaned.split(",") if p.strip()]
        elif isinstance(ts, list):
            parsed = ts
        else:
            parsed = []

        if parsed:
            try:
                dt = pd.to_datetime(parsed, errors="coerce")
                if dt.notna().all():
                    base = dt[0]
                    mins = [float((d - base).total_seconds() / 60.0) for d in dt]
                    return mins
            except Exception:
                pass
    return []


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


def infer_event_and_time(df, generate_labels=True):
    df = df.copy()

    if "透析中收缩压" not in df.columns and "透中血压" in df.columns:
        parsed = df["透中血压"].apply(parse_bp_series)
        df["透析中收缩压"] = parsed.apply(lambda x: x[0])
        df["透析中舒张压"] = parsed.apply(lambda x: x[1])
    else:
        if "透析中收缩压" in df.columns:
            df["透析中收缩压"] = df["透析中收缩压"].apply(to_float_list)
        if "透析中舒张压" in df.columns:
            df["透析中舒张压"] = df["透析中舒张压"].apply(to_float_list)

    minutes_all = df.apply(extract_minutes, axis=1)
    df["minutes_from_start_list"] = minutes_all
    df["duration_minutes"] = minutes_all.apply(
        lambda x: float(max(x)) if len(x) > 0 else np.nan
    )

    if generate_labels and "透中低血压_计算" not in df.columns:
        event_list = []
        et_list = []
        for idx, row in df.iterrows():
            sbp = row["透析中收缩压"] if "透析中收缩压" in df.columns else []
            mins = minutes_all.loc[idx]
            try:
                first_sbp = float(row["透前收缩压"])
            except Exception:
                first_sbp = np.nan

            n = min(len(sbp), len(mins)) if mins else len(sbp)
            sbp = sbp[:n]
            mins = mins[:n] if mins else list(np.arange(n) * 5.0)

            event = 0
            et = np.nan
            if np.isfinite(first_sbp):
                for p, m in zip(sbp, mins):
                    if (first_sbp - p >= 30) or (p <= 90):
                        event = 1
                        et = float(m)
                        break
            event_list.append(event)
            et_list.append(et)
        df["透中低血压_计算"] = event_list
        df["et_min"] = et_list

    if generate_labels and "events" not in df.columns:
        df["events"] = (
            pd.to_numeric(df["透中低血压_计算"], errors="coerce").fillna(0).astype(int)
        )
    if generate_labels and "et_min" not in df.columns:
        df["et_min"] = np.nan

    if "events" in df.columns and "et_min" in df.columns:
        df.loc[df["events"] == 0, "et_min"] = np.nan
    return df


def normalize_merge_keys(df):
    out = df.copy()
    out["患者id"] = out["患者id"].astype(str).str.strip()
    out["透析日期"] = pd.to_datetime(out["透析日期"], errors="coerce").dt.strftime(
        "%Y-%m-%d"
    )
    return out


def apply_landmark(df, features, landmark_minutes):
    df = df.copy()
    feat = features.copy()

    duration = pd.to_numeric(df["duration_minutes"], errors="coerce")
    event = pd.to_numeric(df["events"], errors="coerce").fillna(0).astype(int)
    et = pd.to_numeric(df["et_min"], errors="coerce")

    at_risk_mask = duration > landmark_minutes
    no_early_event_mask = ~((event == 1) & (et <= landmark_minutes))
    keep_mask = at_risk_mask & no_early_event_mask

    df = df.loc[keep_mask].copy()
    feat = feat.loc[keep_mask].copy()

    et = pd.to_numeric(df["et_min"], errors="coerce")
    duration = pd.to_numeric(df["duration_minutes"], errors="coerce")
    landmark_event = ((df["events"] == 1) & (et > landmark_minutes)).astype(int)
    landmark_time = np.where(
        landmark_event.values == 1,
        et.values - landmark_minutes,
        duration.values - landmark_minutes,
    )

    df["landmark_event"] = landmark_event.astype(int)
    df["landmark_time"] = landmark_time.astype(np.float32)
    return df, feat


def pick_existing_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def pre60_stats(values, minutes, obs_minutes):
    if not values:
        return {
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "delta": np.nan,
            "slope": np.nan,
            "cv": np.nan,
        }

    n = min(len(values), len(minutes)) if minutes else len(values)
    v = np.asarray(values[:n], dtype=float)
    if minutes and len(minutes) >= n:
        m = np.asarray(minutes[:n], dtype=float)
    else:
        m = np.arange(n, dtype=float) * 5.0

    mask = m <= obs_minutes
    if not np.any(mask):
        mask = np.ones_like(m, dtype=bool)

    v = v[mask]
    m = m[mask]
    if len(v) == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "delta": np.nan,
            "slope": np.nan,
            "cv": np.nan,
        }

    mean = float(np.mean(v))
    std = float(np.std(v))
    vmin = float(np.min(v))
    vmax = float(np.max(v))
    delta = float(v[-1] - v[0]) if len(v) > 1 else 0.0
    cv = float(std / (abs(mean) + 1e-8))

    if len(v) > 1 and np.std(m) > 1e-8:
        slope = float(np.polyfit(m, v, 1)[0])
    else:
        slope = 0.0

    return {
        "mean": mean,
        "std": std,
        "min": vmin,
        "max": vmax,
        "delta": delta,
        "slope": slope,
        "cv": cv,
    }


def build_features_from_upstream(df, obs_minutes):
    feat = pd.DataFrame(index=df.index)

    for c in STATIC_CANDIDATES:
        if c in df.columns:
            if c == "性别":
                sex = df[c].astype(str)
                feat["性别"] = sex.isin(["男", "M", "1"]).astype(float)
            else:
                feat[c] = pd.to_numeric(df[c], errors="coerce")

    if (
        "透析年龄" not in feat.columns
        and "透析日期" in df.columns
        and "出生日期" in df.columns
    ):
        feat["透析年龄"] = (
            pd.to_datetime(df["透析日期"], errors="coerce").dt.year
            - pd.to_datetime(df["出生日期"], errors="coerce").dt.year
        )
    if (
        "透析龄" not in feat.columns
        and "透析日期" in df.columns
        and "首次透析日期" in df.columns
    ):
        feat["透析龄"] = (
            pd.to_datetime(df["透析日期"], errors="coerce")
            - pd.to_datetime(df["首次透析日期"], errors="coerce")
        ).dt.days / 365.25

    if "透前收缩压" in feat.columns and "透前舒张压" in feat.columns:
        feat["脉压差"] = feat["透前收缩压"] - feat["透前舒张压"]
        feat["平均动脉压"] = (feat["透前收缩压"] + 2 * feat["透前舒张压"]) / 3

    if "透前体重" in feat.columns and "干体重" in feat.columns:
        feat["容量超负荷比"] = (feat["透前体重"] - feat["干体重"]) / feat[
            "干体重"
        ].replace(0, np.nan)

    if "超滤量MAX" in df.columns and "干体重" in feat.columns:
        ufmax = pd.to_numeric(df["超滤量MAX"], errors="coerce")
        feat["超滤比"] = ufmax / feat["干体重"].replace(0, np.nan)

    seq_cols = {}
    for key, candidates in SEQ_CANDIDATES.items():
        col = pick_existing_column(df, candidates)
        if col is not None:
            seq_cols[key] = col

    minutes_all = df.apply(extract_minutes, axis=1)
    for key, col in seq_cols.items():
        seq_all = df[col].apply(to_float_list)
        stats_records = [
            pre60_stats(vals, mins, obs_minutes)
            for vals, mins in zip(seq_all.tolist(), minutes_all.tolist())
        ]
        stats_df = pd.DataFrame(stats_records, index=df.index)
        for stat_name in ["mean", "std", "min", "max", "delta", "slope", "cv"]:
            feat[f"pre60_{key}_{stat_name}"] = stats_df[stat_name]

    feat = feat.replace([np.inf, -np.inf], np.nan)
    med = feat.median(numeric_only=True)
    feat = feat.fillna(med).fillna(0.0)
    return feat


def load_feature_matrices():
    raw_source = infer_event_and_time(
        pd.read_csv(CONFIG["source_raw_data"]), generate_labels=False
    )
    raw_target = infer_event_and_time(
        pd.read_csv(CONFIG["target_raw_data"]), generate_labels=False
    )

    feat_source = build_features_from_upstream(raw_source, CONFIG["obs_minutes"])
    feat_target = build_features_from_upstream(raw_target, CONFIG["obs_minutes"])

    raw_source = normalize_merge_keys(raw_source)
    raw_target = normalize_merge_keys(raw_target)
    label_source = normalize_merge_keys(pd.read_csv(CONFIG["source_label_data"]))
    label_target = normalize_merge_keys(pd.read_csv(CONFIG["target_label_data"]))

    label_cols = [
        "患者id",
        "透析日期",
        CONFIG["event_col"],
        "events",
        CONFIG["duration_col"],
    ]
    label_source = label_source[
        [c for c in label_cols if c in label_source.columns]
    ].drop_duplicates(subset=["患者id", "透析日期"], keep="first")
    label_target = label_target[
        [c for c in label_cols if c in label_target.columns]
    ].drop_duplicates(subset=["患者id", "透析日期"], keep="first")

    feature_source_df = raw_source[["患者id", "透析日期", "duration_minutes"]].copy()
    feature_target_df = raw_target[["患者id", "透析日期", "duration_minutes"]].copy()

    merge_source = feature_source_df.join(feat_source)
    merge_target = feature_target_df.join(feat_target)
    merge_source = merge_source.merge(
        label_source, on=["患者id", "透析日期"], how="inner"
    )
    merge_target = merge_target.merge(
        label_target, on=["患者id", "透析日期"], how="inner"
    )

    feat_source = merge_source.drop(
        columns=[
            "患者id",
            "透析日期",
            "duration_minutes",
            CONFIG["event_col"],
            "events",
            CONFIG["duration_col"],
        ],
        errors="ignore",
    )
    feat_target = merge_target.drop(
        columns=[
            "患者id",
            "透析日期",
            "duration_minutes",
            CONFIG["event_col"],
            "events",
            CONFIG["duration_col"],
        ],
        errors="ignore",
    )

    df_source = merge_source[
        ["患者id", "透析日期", "duration_minutes", "events", CONFIG["duration_col"]]
    ].copy()
    df_target = merge_target[
        ["患者id", "透析日期", "duration_minutes", "events", CONFIG["duration_col"]]
    ].copy()
    df_source["et_min"] = pd.to_numeric(
        df_source[CONFIG["duration_col"]], errors="coerce"
    )
    df_target["et_min"] = pd.to_numeric(
        df_target[CONFIG["duration_col"]], errors="coerce"
    )

    df_source, feat_source = apply_landmark(
        df_source, feat_source, CONFIG["landmark_minutes"]
    )
    df_target, feat_target = apply_landmark(
        df_target, feat_target, CONFIG["landmark_minutes"]
    )

    common_cols = sorted(set(feat_source.columns).intersection(feat_target.columns))
    feat_source = feat_source[common_cols].astype(np.float32)
    feat_target = feat_target[common_cols].astype(np.float32)

    lower = feat_source.quantile(0.01)
    upper = feat_source.quantile(0.99)
    feat_source = feat_source.clip(lower=lower, upper=upper, axis=1)
    feat_target = feat_target.clip(lower=lower, upper=upper, axis=1)

    scaler = StandardScaler()
    x_source = scaler.fit_transform(feat_source.values)
    x_target = scaler.transform(feat_target.values)

    e_source = df_source["landmark_event"].values.astype(np.int64)
    t_source = df_source["landmark_time"].values.astype(np.float32)
    e_target = df_target["landmark_event"].values.astype(np.int64)
    t_target = df_target["landmark_time"].values.astype(np.float32)

    meta = {
        "source_raw_n": int(len(raw_source)),
        "target_raw_n": int(len(raw_target)),
        "source_label_n": int(len(label_source)),
        "target_label_n": int(len(label_target)),
        "source_merged_n": int(len(merge_source)),
        "target_merged_n": int(len(merge_target)),
        "source_landmark_n": int(len(df_source)),
        "target_landmark_n": int(len(df_target)),
        "source_landmark_event_rate": float(e_source.mean()) if len(e_source) else 0.0,
        "target_landmark_event_rate": float(e_target.mean()) if len(e_target) else 0.0,
    }

    return x_source, e_source, t_source, x_target, e_target, t_target, common_cols, meta


def cox_loss(hazard, event, time):
    hazard = hazard.squeeze(-1)
    idx = torch.argsort(time, descending=True)
    hazard_sorted = hazard[idx]
    event_sorted = event[idx].float()
    hazard_max = hazard_sorted.max()
    cumsum_exp = torch.cumsum(torch.exp(hazard_sorted - hazard_max), dim=0)
    log_risk = torch.log(cumsum_exp + 1e-8) + hazard_max
    uncensored = hazard_sorted - log_risk
    return -(uncensored * event_sorted).sum() / (event_sorted.sum() + 1e-8)


def mean_mmd(source_emb, target_emb):
    return torch.norm(source_emb.mean(dim=0) - target_emb.mean(dim=0), p=2)


def conditional_alignment_loss(source_emb, source_event, target_emb, target_event):
    losses = []
    for cls in (0, 1):
        s_mask = source_event == cls
        t_mask = target_event == cls
        if s_mask.any() and t_mask.any():
            losses.append(mean_mmd(source_emb[s_mask], target_emb[t_mask]))
    if not losses:
        return torch.tensor(0.0, device=source_emb.device)
    return torch.stack(losses).mean()


class SurvivalNet(nn.Module):
    def __init__(self, input_dim, d_model, dropout):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.hazard_head = nn.Linear(d_model, 1)

    def forward(self, x):
        emb = self.encoder(x)
        hazard = self.hazard_head(emb)
        return emb, hazard


def make_loader(x, e, t, batch_size, shuffle, drop_last=False):
    dataset = TensorDataset(
        torch.as_tensor(x, dtype=torch.float32),
        torch.as_tensor(e, dtype=torch.long),
        torch.as_tensor(t, dtype=torch.float32),
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
    )


def evaluate_cindex(model, x_eval, e_eval, t_eval, device):
    model.eval()
    with torch.no_grad():
        x_tensor = torch.as_tensor(x_eval, dtype=torch.float32, device=device)
        _, hazard = model(x_tensor)
        scores = hazard.squeeze(-1).cpu().numpy()
    return float(concordance_index(t_eval, -scores, e_eval))


def evaluate_event_metrics(model, x_eval, e_eval, device):
    model.eval()
    with torch.no_grad():
        x_tensor = torch.as_tensor(x_eval, dtype=torch.float32, device=device)
        _, hazard = model(x_tensor)
        scores = hazard.squeeze(-1).cpu().numpy()
    threshold = (
        np.quantile(scores, 1.0 - float(np.mean(e_eval))) if len(scores) else 0.0
    )
    pred = (scores >= threshold).astype(int)
    tp = int(((pred == 1) & (e_eval == 1)).sum())
    fp = int(((pred == 1) & (e_eval == 0)).sum())
    fn = int(((pred == 0) & (e_eval == 1)).sum())
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "threshold": float(threshold),
    }


def set_trainable_state(model, phase):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.hazard_head.parameters():
        p.requires_grad = True
    if phase in {"partial_unfreeze", "full_finetune"}:
        for idx in [4, 5]:
            for p in model.encoder[idx].parameters():
                p.requires_grad = True
    if phase == "full_finetune":
        for p in model.encoder.parameters():
            p.requires_grad = True


def build_optimizer(model, phase):
    if phase == "head_only":
        return optim.AdamW(
            model.hazard_head.parameters(), lr=CONFIG["head_lr"], weight_decay=1e-3
        )
    if phase == "partial_unfreeze":
        groups = [
            {"params": model.hazard_head.parameters(), "lr": CONFIG["head_lr"]},
            {"params": model.encoder[4].parameters(), "lr": CONFIG["top_encoder_lr"]},
            {"params": model.encoder[5].parameters(), "lr": CONFIG["top_encoder_lr"]},
        ]
        return optim.AdamW(groups, weight_decay=1e-3)
    if phase == "full_finetune":
        groups = [
            {"params": model.hazard_head.parameters(), "lr": CONFIG["head_lr"]},
            {"params": model.encoder.parameters(), "lr": CONFIG["full_lr"]},
        ]
        return optim.AdamW(groups, weight_decay=1e-4)
    raise ValueError(f"Unsupported phase {phase}")


def run_source_pretrain(model, source_loader, x_val, e_val, t_val):
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-4)
    best_val = -np.inf
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    patience_left = CONFIG["pretrain_patience"]
    history = []

    for epoch in range(1, CONFIG["pretrain_max_epochs"] + 1):
        model.train()
        total_loss = 0.0
        for x_s, e_s, t_s in source_loader:
            x_s = x_s.to(CONFIG["device"])
            e_s = e_s.to(CONFIG["device"])
            t_s = t_s.to(CONFIG["device"])
            optimizer.zero_grad()
            _, hazard = model(x_s)
            loss = cox_loss(hazard, e_s, t_s)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_cindex = evaluate_cindex(model, x_val, e_val, t_val, CONFIG["device"])
        history.append({"epoch": epoch, "adapt_val_cindex": val_cindex})
        print(
            f"  PRETRAIN Epoch {epoch:02d} | loss={total_loss/len(source_loader):.4f} | adapt_val_cindex={val_cindex:.4f}"
        )

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
    return best_val, best_epoch, best_state, history


def run_head_only_baseline(model, target_train_loader, x_val, e_val, t_val):
    for p in model.encoder.parameters():
        p.requires_grad = False
    optimizer = optim.AdamW(
        model.hazard_head.parameters(), lr=CONFIG["head_lr"], weight_decay=1e-3
    )
    best_val = evaluate_cindex(model, x_val, e_val, t_val, CONFIG["device"])
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    patience_left = CONFIG["head_only_patience"]
    history = []

    for epoch in range(1, CONFIG["head_only_max_epochs"] + 1):
        model.train()
        total_loss = 0.0
        for x_t, e_t, t_t in target_train_loader:
            x_t = x_t.to(CONFIG["device"])
            e_t = e_t.to(CONFIG["device"])
            t_t = t_t.to(CONFIG["device"])
            optimizer.zero_grad()
            _, hazard = model(x_t)
            loss = cox_loss(hazard, e_t, t_t)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_cindex = evaluate_cindex(model, x_val, e_val, t_val, CONFIG["device"])
        history.append({"epoch": epoch, "adapt_val_cindex": val_cindex})
        print(
            f"  HEAD_ONLY_BASE Epoch {epoch:02d} | loss={total_loss/len(target_train_loader):.4f} | adapt_val_cindex={val_cindex:.4f}"
        )

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
    return best_val, best_epoch, best_state, history


def run_da_phase(
    model,
    phase,
    max_epochs,
    patience,
    source_loader,
    target_train_loader,
    x_val,
    e_val,
    t_val,
):
    set_trainable_state(model, phase)
    optimizer = build_optimizer(model, phase)
    best_val = evaluate_cindex(model, x_val, e_val, t_val, CONFIG["device"])
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    patience_left = patience
    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_target = 0.0
        total_replay = 0.0
        total_align = 0.0
        source_iter = iter(source_loader)

        for x_t, e_t, t_t in target_train_loader:
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
            target_loss = cox_loss(hazard_t, e_t, t_t)
            replay_loss = cox_loss(hazard_s, e_s, t_s)
            align_loss = conditional_alignment_loss(emb_s, e_s, emb_t, e_t)
            loss = (
                target_loss
                + CONFIG["source_replay_weight"] * replay_loss
                + CONFIG["conditional_align_weight"] * align_loss
            )
            loss.backward()
            optimizer.step()

            total_target += target_loss.item()
            total_replay += replay_loss.item()
            total_align += align_loss.item()

        val_cindex = evaluate_cindex(model, x_val, e_val, t_val, CONFIG["device"])
        history.append({"phase": phase, "epoch": epoch, "adapt_val_cindex": val_cindex})
        print(
            f"  {phase.upper()} Epoch {epoch:02d} | "
            f"target_loss={total_target/len(target_train_loader):.4f} | "
            f"replay_loss={total_replay/len(target_train_loader):.4f} | "
            f"align_loss={total_align/len(target_train_loader):.4f} | "
            f"adapt_val_cindex={val_cindex:.4f}"
        )

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
    return best_val, best_epoch, best_state, history


def main():
    set_seed(CONFIG["seed"])
    print("1. Loading raw features + optimized labels for landmark-60...")
    (
        x_source,
        e_source,
        t_source,
        x_target,
        e_target,
        t_target,
        feature_names,
        meta,
    ) = load_feature_matrices()

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

    source_loader = make_loader(
        x_source,
        e_source,
        t_source,
        CONFIG["batch_size"],
        shuffle=True,
        drop_last=len(x_source) >= CONFIG["batch_size"],
    )
    target_train_loader = make_loader(
        x_train, e_train, t_train, CONFIG["batch_size"], shuffle=True, drop_last=False
    )

    print(f"   Feature count        : {len(feature_names)}")
    print(f"   Source merged samples: {meta['source_merged_n']}")
    print(f"   Target merged samples: {meta['target_merged_n']}")
    print(f"   Source samples       : {len(x_source)}")
    print(f"   Adapt train samples  : {len(x_train)}")
    print(f"   Adapt val samples    : {len(x_val)}")
    print(f"   Test samples         : {len(x_test)}")
    print(f"   Source event rate    : {e_source.mean():.4f}")
    print(f"   Adapt train rate     : {e_train.mean():.4f}")
    print(f"   Adapt val rate       : {e_val.mean():.4f}")
    print(f"   Test event rate      : {e_test.mean():.4f}")

    model = SurvivalNet(x_source.shape[1], CONFIG["d_model"], CONFIG["dropout"]).to(
        CONFIG["device"]
    )

    print("\n2. Source pretraining (adapt_val early stopping)")
    pre_val, pre_epoch, source_state, pre_history = run_source_pretrain(
        model, source_loader, x_val, e_val, t_val
    )
    zero_shot_test = evaluate_cindex(model, x_test, e_test, t_test, CONFIG["device"])

    print("\n3. Head-only clean baseline")
    head_model = SurvivalNet(
        x_source.shape[1], CONFIG["d_model"], CONFIG["dropout"]
    ).to(CONFIG["device"])
    head_model.load_state_dict(source_state)
    head_val, head_epoch, _, head_history = run_head_only_baseline(
        head_model, target_train_loader, x_val, e_val, t_val
    )
    head_test = evaluate_cindex(head_model, x_test, e_test, t_test, CONFIG["device"])

    print("\n4. V11.4 landmark-60 clean DA")
    da_model = SurvivalNet(x_source.shape[1], CONFIG["d_model"], CONFIG["dropout"]).to(
        CONFIG["device"]
    )
    da_model.load_state_dict(source_state)

    phase_plan = [
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
    ]
    phase_results = []
    all_phase_history = []
    for phase, max_epochs, patience in phase_plan:
        val_score, best_epoch, _, phase_history = run_da_phase(
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
        all_phase_history.extend(phase_history)

    da_test = evaluate_cindex(da_model, x_test, e_test, t_test, CONFIG["device"])
    zero_event_metrics = evaluate_event_metrics(model, x_test, e_test, CONFIG["device"])
    head_event_metrics = evaluate_event_metrics(
        head_model, x_test, e_test, CONFIG["device"]
    )
    da_event_metrics = evaluate_event_metrics(
        da_model, x_test, e_test, CONFIG["device"]
    )

    results = {
        "feature_names": feature_names,
        "landmark_meta": meta,
        "split": {
            "source_n": int(len(x_source)),
            "adapt_train_n": int(len(x_train)),
            "adapt_val_n": int(len(x_val)),
            "test_n": int(len(x_test)),
        },
        "source_pretrain": {
            "best_adapt_val_cindex": pre_val,
            "best_epoch": pre_epoch,
            "zero_shot_test_cindex": zero_shot_test,
            "event_metrics": zero_event_metrics,
            "history": pre_history,
        },
        "head_only_clean": {
            "best_adapt_val_cindex": head_val,
            "best_epoch": head_epoch,
            "test_cindex": head_test,
            "event_metrics": head_event_metrics,
            "history": head_history,
        },
        "v11_4_landmark60_da": {
            "phase_results": phase_results,
            "test_cindex": da_test,
            "event_metrics": da_event_metrics,
            "history": all_phase_history,
            "delta_vs_zero_shot_test": da_test - zero_shot_test,
            "delta_vs_head_only_test": da_test - head_test,
        },
        "config": {
            "obs_minutes": CONFIG["obs_minutes"],
            "landmark_minutes": CONFIG["landmark_minutes"],
            "source_replay_weight": CONFIG["source_replay_weight"],
            "conditional_align_weight": CONFIG["conditional_align_weight"],
            "pretrain_max_epochs": CONFIG["pretrain_max_epochs"],
            "head_only_max_epochs": CONFIG["head_only_max_epochs"],
            "partial_unfreeze_max_epochs": CONFIG["partial_unfreeze_max_epochs"],
            "full_finetune_max_epochs": CONFIG["full_finetune_max_epochs"],
        },
    }

    with open(CONFIG["results_path"], "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n============================================================")
    print(f"Zero-shot test C-index    : {zero_shot_test:.4f}")
    print(f"Head-only test C-index    : {head_test:.4f}")
    print(f"V11.4 landmark DA test    : {da_test:.4f}")
    print(f"Delta vs zero-shot        : {da_test - zero_shot_test:+.4f}")
    print(f"Delta vs head-only        : {da_test - head_test:+.4f}")
    print(f"Results saved to          : {CONFIG['results_path']}")
    print("============================================================")


if __name__ == "__main__":
    main()
