"""
v11.1 clean validation domain adaptation benchmark

Goals:
1. Clean target split: adapt_train / adapt_val / test
2. Early stopping on adapt_val only
3. Dynamic summary features from current final CSV
4. Keep v11 core design: layer-wise adaptation + source replay + conditional alignment
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
    "results_path": "runs/v11_1_clean_da_results.json",
}


BASE_FEATURES = [
    "透析年龄",
    "性别",
    "透析龄",
    "超滤比",
    "容量超负荷比",
    "脉压差",
    "平均动脉压",
]

DYNAMIC_SUMMARY_FEATURES = [
    "透析中收缩压_mean",
    "透析中收缩压_std",
    "透析中舒张压_mean",
    "透析中舒张压_std",
    "透析中脉搏_mean",
    "透析中脉搏_std",
    "超滤率_mean",
    "超滤率_std",
    "静脉压_mean",
    "静脉压_std",
    "血流速_mean",
    "血流速_std",
    "透析液温度_mean",
    "透析液温度_std",
    "动脉压_mean",
    "动脉压_std",
]

HISTORY_PAIRS = [
    ("透析中收缩压_mean", "历史平均透析中收缩压_mean", "透析中收缩压偏离历史"),
    ("透析中舒张压_mean", "历史平均透析中舒张压_mean", "透析中舒张压偏离历史"),
    ("透析中脉搏_mean", "历史平均透析中脉搏_mean", "透析中脉搏偏离历史"),
    ("超滤率_mean", "历史平均超滤率_mean", "超滤率偏离历史"),
    ("静脉压_mean", "历史平均静脉压_mean", "静脉压偏离历史"),
    ("血流速_mean", "历史平均血流速_mean", "血流速偏离历史"),
    ("动脉压_mean", "历史平均动脉压_mean", "动脉压偏离历史"),
    ("透前收缩压", "历史平均透前收缩压", "透前收缩压偏离历史"),
    ("透前舒张压", "历史平均透前舒张压", "透前舒张压偏离历史"),
]


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_feature_frame(df):
    feat = pd.DataFrame(index=df.index)
    if "透析年龄" in df.columns:
        feat["透析年龄"] = df["透析年龄"]
    if "性别" in df.columns:
        feat["性别"] = (df["性别"] == "M").astype(float)
    if "透析龄" in df.columns:
        feat["透析龄"] = df["透析龄"]
    if "超滤量MAX" in df.columns and "干体重" in df.columns:
        feat["超滤比"] = df["超滤量MAX"] / df["干体重"].replace(0, np.nan)
    if "透前体重" in df.columns and "干体重" in df.columns:
        feat["容量超负荷比"] = (df["透前体重"] - df["干体重"]) / df["干体重"].replace(
            0, np.nan
        )
    if "透前收缩压" in df.columns and "透前舒张压" in df.columns:
        feat["脉压差"] = df["透前收缩压"] - df["透前舒张压"]
        feat["平均动脉压"] = (df["透前收缩压"] + 2 * df["透前舒张压"]) / 3

    for col in DYNAMIC_SUMMARY_FEATURES:
        if col in df.columns:
            feat[col] = df[col]

    for current_col, hist_col, feature_name in HISTORY_PAIRS:
        if current_col in df.columns and hist_col in df.columns:
            feat[feature_name] = df[current_col] - df[hist_col]
            ratio_name = feature_name + "_ratio"
            feat[ratio_name] = df[current_col] / df[hist_col].replace(0, np.nan)

    medians = feat.median(numeric_only=True)
    feat = feat.fillna(medians).fillna(0.0)
    return feat


def load_feature_matrices(source_path, target_path):
    df_source = pd.read_csv(source_path)
    df_target = pd.read_csv(target_path)

    feat_source = build_feature_frame(df_source)
    feat_target = build_feature_frame(df_target)

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

    e_source = df_source[CONFIG["event_col"]].fillna(0).values.astype(np.int64)
    t_source = df_source[CONFIG["duration_col"]].fillna(0).values.astype(np.float32)
    e_target = df_target[CONFIG["event_col"]].fillna(0).values.astype(np.int64)
    t_target = df_target[CONFIG["duration_col"]].fillna(0).values.astype(np.float32)

    return (
        x_source,
        e_source,
        t_source,
        x_target,
        e_target,
        t_target,
        common_cols,
    )


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
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )


def evaluate_cindex(model, x_eval, e_eval, t_eval, device):
    model.eval()
    with torch.no_grad():
        x_tensor = torch.as_tensor(x_eval, dtype=torch.float32, device=device)
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
            model.hazard_head.parameters(),
            lr=CONFIG["head_lr"],
            weight_decay=1e-3,
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
    for param in model.encoder.parameters():
        param.requires_grad = False
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
    print("1. Loading clean feature matrices...")
    (
        x_source,
        e_source,
        t_source,
        x_target,
        e_target,
        t_target,
        feature_names,
    ) = load_feature_matrices(CONFIG["source_data"], CONFIG["target_data"])

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

    print(f"   Feature count        : {len(feature_names)}")
    print(f"   Source samples       : {len(x_source)}")
    print(f"   Adapt train samples  : {len(x_train)}")
    print(f"   Adapt val samples    : {len(x_val)}")
    print(f"   Test samples         : {len(x_test)}")
    print(f"   Source event rate    : {e_source.mean():.4f}")
    print(f"   Adapt train rate     : {e_train.mean():.4f}")
    print(f"   Adapt val rate       : {e_val.mean():.4f}")

    input_dim = x_source.shape[1]
    base_model = SurvivalNet(input_dim, CONFIG["d_model"], CONFIG["dropout"]).to(
        CONFIG["device"]
    )

    print("\n2. Source pretraining with adapt_val early stopping")
    pre_val, pre_epoch, source_state, pre_history = run_source_pretrain(
        base_model, source_loader, x_val, e_val, t_val
    )
    zero_shot_test = evaluate_cindex(
        base_model, x_test, e_test, t_test, CONFIG["device"]
    )

    print("\n3. Head-only clean baseline")
    head_model = SurvivalNet(input_dim, CONFIG["d_model"], CONFIG["dropout"]).to(
        CONFIG["device"]
    )
    head_model.load_state_dict(source_state)
    head_val, head_epoch, _, head_history = run_head_only_baseline(
        head_model, target_train_loader, x_val, e_val, t_val
    )
    head_test = evaluate_cindex(head_model, x_test, e_test, t_test, CONFIG["device"])

    print("\n4. V11.1 clean DA")
    da_model = SurvivalNet(input_dim, CONFIG["d_model"], CONFIG["dropout"]).to(
        CONFIG["device"]
    )
    da_model.load_state_dict(source_state)

    phase_results = []
    all_phase_history = []
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

    results = {
        "feature_names": feature_names,
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
            "history": pre_history,
        },
        "head_only_clean": {
            "best_adapt_val_cindex": head_val,
            "best_epoch": head_epoch,
            "test_cindex": head_test,
            "history": head_history,
        },
        "v11_1_clean_da": {
            "phase_results": phase_results,
            "test_cindex": da_test,
            "history": all_phase_history,
            "delta_vs_zero_shot_test": da_test - zero_shot_test,
            "delta_vs_head_only_test": da_test - head_test,
        },
        "config": {
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
    print(f"V11.1 clean DA test       : {da_test:.4f}")
    print(f"Delta vs zero-shot        : {da_test - zero_shot_test:+.4f}")
    print(f"Delta vs head-only        : {da_test - head_test:+.4f}")
    print(f"Results saved to          : {CONFIG['results_path']}")
    print("============================================================")


if __name__ == "__main__":
    main()
