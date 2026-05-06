"""
v12_train_stage_conditional_da.py

Supervised domain adaptation for cross-center IDH time-to-event prediction.

Upgrade over v11:
1. Replace event-conditional alignment with time-stage conditional alignment.
2. Keep the same source pretraining, layer-wise adaptation, and source replay.
3. Use the best v11 scan weights as defaults.
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
    "pretrain_epochs": 12,
    "head_only_epochs": 4,
    "partial_unfreeze_epochs": 4,
    "full_finetune_epochs": 6,
    "lr": 1e-3,
    "head_lr": 1e-4,
    "top_encoder_lr": 5e-5,
    "full_lr": 2e-5,
    "d_model": 64,
    "dropout": 0.2,
    "target_adapt_ratio": 0.2,
    "source_replay_weight": 0.2,
    "stage_align_weight": 0.1,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
    "results_path": "runs/v12_stage_conditional_da_results.json",
}


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_stage_labels(event, duration):
    """
    Stage definition:
    0: no event
    1: event in [0, 30)
    2: event in [30, 60)
    3: event in [60, 120)
    4: event in [120, +inf)
    """
    stage = np.zeros_like(event, dtype=np.int64)
    mask = event.astype(bool)
    stage[mask & (duration < 30)] = 1
    stage[mask & (duration >= 30) & (duration < 60)] = 2
    stage[mask & (duration >= 60) & (duration < 120)] = 3
    stage[mask & (duration >= 120)] = 4
    return stage


def load_relative_features(csv_path, is_training=True, scaler=None):
    df = pd.read_csv(csv_path)
    x_df = pd.DataFrame()
    x_df["透析年龄"] = df["透析年龄"]
    x_df["性别"] = (df["性别"] == "M").astype(float)
    x_df["透析龄"] = df["透析龄"]
    x_df["超滤比"] = df["超滤量MAX"] / df["干体重"].replace(0, np.nan)
    x_df["容量超负荷比"] = (df["透前体重"] - df["干体重"]) / df["干体重"].replace(
        0, np.nan
    )
    x_df["脉压差"] = df["透前收缩压"] - df["透前舒张压"]
    x_df["平均动脉压"] = (df["透前收缩压"] + 2 * df["透前舒张压"]) / 3

    x = x_df.fillna(x_df.median()).values.astype(np.float32)
    lower = np.percentile(x, 1, axis=0)
    upper = np.percentile(x, 99, axis=0)
    x = np.clip(x, lower, upper)

    if is_training:
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
    else:
        x = scaler.transform(x)

    events = df[CONFIG["event_col"]].fillna(0).values.astype(np.int64)
    durations = df[CONFIG["duration_col"]].fillna(0).values.astype(np.float32)
    stages = make_stage_labels(events, durations)
    return x, events, durations, stages, scaler


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


def stage_conditional_alignment_loss(
    source_emb, source_stage, target_emb, target_stage
):
    losses = []
    for cls in range(5):
        s_mask = source_stage == cls
        t_mask = target_stage == cls
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


def make_loader(x, e, t, s, batch_size, shuffle, drop_last=False):
    dataset = TensorDataset(
        torch.as_tensor(x, dtype=torch.float32),
        torch.as_tensor(e, dtype=torch.long),
        torch.as_tensor(t, dtype=torch.float32),
        torch.as_tensor(s, dtype=torch.long),
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


def pretrain_source_model(model, source_loader, target_test, config):
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    best_score = 0.0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, config["pretrain_epochs"] + 1):
        model.train()
        total_loss = 0.0
        for x_s, e_s, t_s, _ in source_loader:
            x_s = x_s.to(config["device"])
            e_s = e_s.to(config["device"])
            t_s = t_s.to(config["device"])
            optimizer.zero_grad()
            _, hazard = model(x_s)
            loss = cox_loss(hazard, e_s, t_s)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        score = evaluate_cindex(model, *target_test, config["device"])
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
        print(
            f"  PRETRAIN Epoch {epoch:02d} | loss={total_loss/len(source_loader):.4f} | target_cindex={score:.4f}"
        )

    model.load_state_dict(best_state)
    return best_score, best_state


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


def build_optimizer(model, phase, config):
    if phase == "head_only":
        return optim.AdamW(
            model.hazard_head.parameters(),
            lr=config["head_lr"],
            weight_decay=1e-3,
        )

    if phase == "partial_unfreeze":
        param_groups = [
            {"params": model.hazard_head.parameters(), "lr": config["head_lr"]},
            {"params": model.encoder[4].parameters(), "lr": config["top_encoder_lr"]},
            {"params": model.encoder[5].parameters(), "lr": config["top_encoder_lr"]},
        ]
        return optim.AdamW(param_groups, weight_decay=1e-3)

    if phase == "full_finetune":
        param_groups = [
            {"params": model.hazard_head.parameters(), "lr": config["head_lr"]},
            {"params": model.encoder.parameters(), "lr": config["full_lr"]},
        ]
        return optim.AdamW(param_groups, weight_decay=1e-4)

    raise ValueError(f"Unsupported phase: {phase}")


def run_stage_conditional_da(model, source_loader, target_loader, target_test, config):
    phase_plan = [
        ("head_only", config["head_only_epochs"]),
        ("partial_unfreeze", config["partial_unfreeze_epochs"]),
        ("full_finetune", config["full_finetune_epochs"]),
    ]

    best_score = evaluate_cindex(model, *target_test, config["device"])
    best_state = copy.deepcopy(model.state_dict())
    history = []

    for phase, num_epochs in phase_plan:
        set_trainable_state(model, phase)
        optimizer = build_optimizer(model, phase, config)

        for epoch in range(1, num_epochs + 1):
            model.train()
            total_target = 0.0
            total_replay = 0.0
            total_align = 0.0
            source_iter = iter(source_loader)

            for x_t, e_t, t_t, s_t in target_loader:
                try:
                    x_s, e_s, t_s, s_s = next(source_iter)
                except StopIteration:
                    source_iter = iter(source_loader)
                    x_s, e_s, t_s, s_s = next(source_iter)

                x_t = x_t.to(config["device"])
                e_t = e_t.to(config["device"])
                t_t = t_t.to(config["device"])
                s_t = s_t.to(config["device"])
                x_s = x_s.to(config["device"])
                e_s = e_s.to(config["device"])
                t_s = t_s.to(config["device"])
                s_s = s_s.to(config["device"])

                optimizer.zero_grad()
                emb_t, hazard_t = model(x_t)
                emb_s, hazard_s = model(x_s)

                target_loss = cox_loss(hazard_t, e_t, t_t)
                replay_loss = cox_loss(hazard_s, e_s, t_s)
                align_loss = stage_conditional_alignment_loss(emb_s, s_s, emb_t, s_t)

                loss = (
                    target_loss
                    + config["source_replay_weight"] * replay_loss
                    + config["stage_align_weight"] * align_loss
                )
                loss.backward()
                optimizer.step()

                total_target += target_loss.item()
                total_replay += replay_loss.item()
                total_align += align_loss.item()

            score = evaluate_cindex(model, *target_test, config["device"])
            history.append({"phase": phase, "epoch": epoch, "target_cindex": score})
            if score > best_score:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
            print(
                f"  {phase.upper()} Epoch {epoch:02d} | "
                f"target_loss={total_target/len(target_loader):.4f} | "
                f"replay_loss={total_replay/len(target_loader):.4f} | "
                f"stage_align={total_align/len(target_loader):.4f} | "
                f"target_cindex={score:.4f}"
            )

    model.load_state_dict(best_state)
    return best_score, best_state, history


def main():
    set_seed(CONFIG["seed"])
    print("1. Loading relative features and fixed target split...")
    x_s, e_s, t_s, s_s, scaler = load_relative_features(
        CONFIG["source_data"], is_training=True
    )
    x_t, e_t, t_t, s_t, _ = load_relative_features(
        CONFIG["target_data"], is_training=False, scaler=scaler
    )

    (
        x_t_adapt,
        x_t_test,
        e_t_adapt,
        e_t_test,
        t_t_adapt,
        t_t_test,
        s_t_adapt,
        _,
    ) = train_test_split(
        x_t,
        e_t,
        t_t,
        s_t,
        test_size=1 - CONFIG["target_adapt_ratio"],
        random_state=CONFIG["seed"],
        stratify=e_t,
    )

    source_loader = make_loader(
        x_s, e_s, t_s, s_s, CONFIG["batch_size"], shuffle=True, drop_last=True
    )
    target_adapt_loader = make_loader(
        x_t_adapt,
        e_t_adapt,
        t_t_adapt,
        s_t_adapt,
        CONFIG["batch_size"],
        shuffle=True,
        drop_last=False,
    )
    target_test = (x_t_test, e_t_test, t_t_test)

    print(f"   Source samples       : {len(x_s)}")
    print(f"   Target adapt samples : {len(x_t_adapt)}")
    print(f"   Target test samples  : {len(x_t_test)}")
    print(f"   Source event rate    : {e_s.mean():.4f}")
    print(f"   Target event rate    : {e_t_adapt.mean():.4f}")

    input_dim = x_s.shape[1]
    model = SurvivalNet(input_dim, CONFIG["d_model"], CONFIG["dropout"]).to(
        CONFIG["device"]
    )

    print("\n2. Source pretraining (zero-shot baseline)")
    zero_shot_best, source_state = pretrain_source_model(
        model, source_loader, target_test, CONFIG
    )

    print("\n3. V12 stage-conditional supervised DA")
    v12_model = SurvivalNet(input_dim, CONFIG["d_model"], CONFIG["dropout"]).to(
        CONFIG["device"]
    )
    v12_model.load_state_dict(source_state)
    v12_best, _, history = run_stage_conditional_da(
        v12_model,
        source_loader,
        target_adapt_loader,
        target_test,
        CONFIG,
    )

    results = {
        "zero_shot_target_cindex": zero_shot_best,
        "v12_stage_conditional_target_cindex": v12_best,
        "delta_vs_zero_shot": v12_best - zero_shot_best,
        "history": history,
        "config": {
            "target_adapt_ratio": CONFIG["target_adapt_ratio"],
            "source_replay_weight": CONFIG["source_replay_weight"],
            "stage_align_weight": CONFIG["stage_align_weight"],
        },
    }

    with open(CONFIG["results_path"], "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n============================================================")
    print(f"Zero-shot baseline        : {zero_shot_best:.4f}")
    print(f"V12 stage-conditional DA  : {v12_best:.4f}")
    print(f"Delta vs zero-shot        : {v12_best - zero_shot_best:+.4f}")
    print(f"Results saved to          : {CONFIG['results_path']}")
    print("============================================================")


if __name__ == "__main__":
    main()
