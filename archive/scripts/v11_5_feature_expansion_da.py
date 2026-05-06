"""
v11.5 feature expansion benchmark for direction-one supervised domain adaptation.

Design:
1. Keep clean validation: target adapt_train / adapt_val / test.
2. Preserve v11.1 layer-wise supervised DA training loop.
3. Compare grouped feature sets instead of changing the model backbone.
"""

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
    "source_data": "data_preprocessing/data/深医_final_data.csv",
    "target_data": "data_preprocessing/data/福鼎_final_data.csv",
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
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
    "results_path": "runs/v11_5_feature_expansion_da_results.json",
}


CORE_FEATURES = [
    "透析年龄",
    "性别",
    "透析龄",
    "超滤比",
    "容量超负荷比",
    "脉压差",
    "平均动脉压",
]

BASELINE_FEATURES = [
    "透前体重",
    "透前体重-干体重",
    "透前动脉压",
    "超滤量MAX",
    "超滤率_绝对",
    "透析龄_天数",
    "透析龄占比",
    "首次透析年龄",
    "高血压诊断",
]

HISTORY_FEATURES = [
    "history_HBP",
    "history_HBP_rate",
    "history_LBP_times_0_rate",
    "history_LBP_times_1_rate",
    "history_LBP_times_2_rate",
    "history_LBP_times_3_rate",
    "history_LBP_times_4_rate",
    "历史平均透前收缩压",
    "历史平均透前舒张压",
    "历史平均透前体重",
    "历史平均超滤量MAX",
    "历史平均超滤率_mean",
    "历史平均透中低血压_计算",
]

DELTA_SPECS = [
    ("透前收缩压", "历史平均透前收缩压", "delta_透前收缩压_vs_hist"),
    ("透前舒张压", "历史平均透前舒张压", "delta_透前舒张压_vs_hist"),
    ("透前体重", "历史平均透前体重", "delta_透前体重_vs_hist"),
    ("超滤量MAX", "历史平均超滤量MAX", "delta_超滤量MAX_vs_hist"),
    ("超滤率_绝对", "历史平均超滤率_mean", "delta_超滤率绝对_vs_hist"),
]


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_master_feature_frame(df):
    feat = pd.DataFrame(index=df.index)

    if "透析年龄" in df.columns:
        feat["透析年龄"] = pd.to_numeric(df["透析年龄"], errors="coerce")
    if "性别" in df.columns:
        feat["性别"] = (df["性别"] == "M").astype(float)
    if "透析龄" in df.columns:
        feat["透析龄"] = pd.to_numeric(df["透析龄"], errors="coerce")

    if "超滤量MAX" in df.columns and "干体重" in df.columns:
        max_uf = pd.to_numeric(df["超滤量MAX"], errors="coerce")
        dry_weight = pd.to_numeric(df["干体重"], errors="coerce").replace(0, np.nan)
        feat["超滤比"] = max_uf / dry_weight

    if "透前体重" in df.columns and "干体重" in df.columns:
        pre_weight = pd.to_numeric(df["透前体重"], errors="coerce")
        dry_weight = pd.to_numeric(df["干体重"], errors="coerce").replace(0, np.nan)
        feat["容量超负荷比"] = (pre_weight - dry_weight) / dry_weight

    if "透前收缩压" in df.columns and "透前舒张压" in df.columns:
        sbp = pd.to_numeric(df["透前收缩压"], errors="coerce")
        dbp = pd.to_numeric(df["透前舒张压"], errors="coerce")
        feat["脉压差"] = sbp - dbp
        feat["平均动脉压"] = (sbp + 2 * dbp) / 3

    all_direct = (
        BASELINE_FEATURES
        + HISTORY_FEATURES
        + [x[0] for x in DELTA_SPECS]
        + [x[1] for x in DELTA_SPECS]
    )
    for col in all_direct:
        if col in df.columns:
            feat[col] = pd.to_numeric(df[col], errors="coerce")

    for cur_col, hist_col, out_col in DELTA_SPECS:
        if cur_col in df.columns and hist_col in df.columns:
            cur = pd.to_numeric(df[cur_col], errors="coerce")
            hist = pd.to_numeric(df[hist_col], errors="coerce")
            feat[out_col] = cur - hist
            feat[out_col + "_ratio"] = cur / hist.replace(0, np.nan)

    medians = feat.median(numeric_only=True)
    feat = feat.fillna(medians).fillna(0.0)
    return feat


def get_feature_groups(columns):
    feature_groups = {
        "A_core7": CORE_FEATURES,
        "B_core7_baseline": CORE_FEATURES + BASELINE_FEATURES,
        "C_core7_history": CORE_FEATURES + HISTORY_FEATURES,
        "D_core7_baseline_history_delta": CORE_FEATURES
        + BASELINE_FEATURES
        + HISTORY_FEATURES
        + [item[2] for item in DELTA_SPECS]
        + [item[2] + "_ratio" for item in DELTA_SPECS],
    }
    return {
        name: [col for col in group if col in columns]
        for name, group in feature_groups.items()
    }


def load_all_data():
    df_source = pd.read_csv(CONFIG["source_data"])
    df_target = pd.read_csv(CONFIG["target_data"])

    feat_source = build_master_feature_frame(df_source)
    feat_target = build_master_feature_frame(df_target)

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


def prepare_group_matrices(feat_source, feat_target, selected_cols):
    source_df = feat_source[selected_cols].copy()
    target_df = feat_target[selected_cols].copy()

    lower = source_df.quantile(0.01)
    upper = source_df.quantile(0.99)
    source_df = source_df.clip(lower=lower, upper=upper, axis=1)
    target_df = target_df.clip(lower=lower, upper=upper, axis=1)

    scaler = StandardScaler()
    x_source = scaler.fit_transform(source_df.values)
    x_target = scaler.transform(target_df.values)
    return x_source, x_target


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


def evaluate_cindex(model, x_eval, e_eval, t_eval):
    model.eval()
    with torch.no_grad():
        x_tensor = torch.as_tensor(x_eval, dtype=torch.float32, device=CONFIG["device"])
        _, hazard = model(x_tensor)
        scores = hazard.squeeze(-1).cpu().numpy()
    return float(concordance_index(t_eval, -scores, e_eval))


def set_trainable_state(model, phase):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.hazard_head.parameters():
        param.requires_grad = True
    if phase in {"partial_unfreeze", "full_finetune"}:
        for idx in [4, 5]:
            for param in model.encoder[idx].parameters():
                param.requires_grad = True
    if phase == "full_finetune":
        for param in model.encoder.parameters():
            param.requires_grad = True


def build_optimizer(model, phase):
    if phase == "head_only":
        return optim.AdamW(
            model.hazard_head.parameters(), lr=CONFIG["head_lr"], weight_decay=1e-3
        )
    if phase == "partial_unfreeze":
        param_groups = [
            {"params": model.hazard_head.parameters(), "lr": CONFIG["head_lr"]},
            {"params": model.encoder[4].parameters(), "lr": CONFIG["top_encoder_lr"]},
            {"params": model.encoder[5].parameters(), "lr": CONFIG["top_encoder_lr"]},
        ]
        return optim.AdamW(param_groups, weight_decay=1e-3)
    if phase == "full_finetune":
        param_groups = [
            {"params": model.hazard_head.parameters(), "lr": CONFIG["head_lr"]},
            {"params": model.encoder.parameters(), "lr": CONFIG["full_lr"]},
        ]
        return optim.AdamW(param_groups, weight_decay=1e-4)
    raise ValueError(f"Unsupported phase: {phase}")


def run_source_pretrain(model, source_loader, x_val, e_val, t_val):
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-4)
    best_val = -np.inf
    best_epoch = 0
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
            loss = cox_loss(hazard, e_s, t_s)
            loss.backward()
            optimizer.step()

        val_cindex = evaluate_cindex(model, x_val, e_val, t_val)
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


def run_head_only_baseline(model, target_train_loader, x_val, e_val, t_val):
    for param in model.encoder.parameters():
        param.requires_grad = False
    optimizer = optim.AdamW(
        model.hazard_head.parameters(), lr=CONFIG["head_lr"], weight_decay=1e-3
    )
    best_val = evaluate_cindex(model, x_val, e_val, t_val)
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    patience_left = CONFIG["head_only_patience"]

    for epoch in range(1, CONFIG["head_only_max_epochs"] + 1):
        model.train()
        for x_t, e_t, t_t in target_train_loader:
            x_t = x_t.to(CONFIG["device"])
            e_t = e_t.to(CONFIG["device"])
            t_t = t_t.to(CONFIG["device"])
            optimizer.zero_grad()
            _, hazard = model(x_t)
            loss = cox_loss(hazard, e_t, t_t)
            loss.backward()
            optimizer.step()

        val_cindex = evaluate_cindex(model, x_val, e_val, t_val)
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
    x_val,
    e_val,
    t_val,
):
    set_trainable_state(model, phase)
    optimizer = build_optimizer(model, phase)
    best_val = evaluate_cindex(model, x_val, e_val, t_val)
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    patience_left = patience

    for epoch in range(1, max_epochs + 1):
        model.train()
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

        val_cindex = evaluate_cindex(model, x_val, e_val, t_val)
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


def run_feature_group(
    group_name,
    selected_cols,
    feat_source,
    feat_target,
    e_source,
    t_source,
    e_target,
    t_target,
):
    x_source, x_target = prepare_group_matrices(feat_source, feat_target, selected_cols)

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
        x_source, e_source, t_source, CONFIG["batch_size"], shuffle=True, drop_last=True
    )
    target_train_loader = make_loader(
        x_train, e_train, t_train, CONFIG["batch_size"], shuffle=True, drop_last=False
    )

    input_dim = x_source.shape[1]

    base_model = SurvivalNet(input_dim, CONFIG["d_model"], CONFIG["dropout"]).to(
        CONFIG["device"]
    )
    pre_val, pre_epoch, source_state = run_source_pretrain(
        base_model, source_loader, x_val, e_val, t_val
    )
    zero_shot_test = evaluate_cindex(base_model, x_test, e_test, t_test)

    head_model = SurvivalNet(input_dim, CONFIG["d_model"], CONFIG["dropout"]).to(
        CONFIG["device"]
    )
    head_model.load_state_dict(source_state)
    head_val, head_epoch, _ = run_head_only_baseline(
        head_model, target_train_loader, x_val, e_val, t_val
    )
    head_test = evaluate_cindex(head_model, x_test, e_test, t_test)

    da_model = SurvivalNet(input_dim, CONFIG["d_model"], CONFIG["dropout"]).to(
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
        val_score, best_epoch, _ = run_da_phase(
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
    da_test = evaluate_cindex(da_model, x_test, e_test, t_test)

    return {
        "group_name": group_name,
        "feature_count": len(selected_cols),
        "feature_names": selected_cols,
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


def main():
    set_seed(CONFIG["seed"])
    print("Loading master feature tables...")
    feat_source, feat_target, e_source, t_source, e_target, t_target, common_cols = (
        load_all_data()
    )
    groups = get_feature_groups(common_cols)

    all_results = {}
    for group_name, selected_cols in groups.items():
        print(f"\n=== Running {group_name} | features={len(selected_cols)} ===")
        result = run_feature_group(
            group_name,
            selected_cols,
            feat_source,
            feat_target,
            e_source,
            t_source,
            e_target,
            t_target,
        )
        all_results[group_name] = result
        print(
            f"{group_name}: zero-shot={result['source_pretrain']['zero_shot_test_cindex']:.4f}, "
            f"da={result['layerwise_da']['test_cindex']:.4f}, "
            f"gain={result['layerwise_da']['gain_vs_zero_shot']:.4f}"
        )

    summary = {
        name: {
            "feature_count": result["feature_count"],
            "zero_shot_test_cindex": result["source_pretrain"]["zero_shot_test_cindex"],
            "head_only_test_cindex": result["head_only"]["test_cindex"],
            "layerwise_da_test_cindex": result["layerwise_da"]["test_cindex"],
            "gain_vs_zero_shot": result["layerwise_da"]["gain_vs_zero_shot"],
        }
        for name, result in all_results.items()
    }

    with open(CONFIG["results_path"], "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": CONFIG,
                "groups": all_results,
                "summary": summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nSaved results to {CONFIG['results_path']}")


if __name__ == "__main__":
    main()
