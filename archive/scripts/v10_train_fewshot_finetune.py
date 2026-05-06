"""
Unified domain adaptation benchmark for cross-center IDH time-to-event prediction.

Methods:
1. Source-Only
2. CORAL
3. MMD
4. Label-Shift-Aware MMD
5. Supervised DA Fine-tune
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
    "da_epochs": 12,
    "finetune_epochs": 8,
    "lr": 1e-3,
    "finetune_lr": 1e-4,
    "d_model": 64,
    "dropout": 0.2,
    "align_lambda_max": 0.3,
    "target_adapt_ratio": 0.2,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
    "results_path": "runs/v10_da_benchmark_results.json",
}


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    return x, events, durations, scaler


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


def coral_loss(source_emb, target_emb):
    d = source_emb.size(1)
    source_centered = source_emb - source_emb.mean(dim=0, keepdim=True)
    target_centered = target_emb - target_emb.mean(dim=0, keepdim=True)
    source_cov = (source_centered.T @ source_centered) / max(source_emb.size(0) - 1, 1)
    target_cov = (target_centered.T @ target_centered) / max(target_emb.size(0) - 1, 1)
    return torch.norm(source_cov - target_cov, p="fro") ** 2 / (4 * d * d)


def mmd_loss(source_emb, target_emb):
    return torch.norm(source_emb.mean(dim=0) - target_emb.mean(dim=0), p=2)


def weighted_mmd_loss(source_emb, target_emb, source_event, pos_weight, neg_weight):
    weights = torch.where(
        source_event == 1,
        torch.full_like(source_event.float(), pos_weight),
        torch.full_like(source_event.float(), neg_weight),
    ).unsqueeze(1)
    return torch.norm((source_emb * weights).mean(dim=0) - target_emb.mean(dim=0), p=2)


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


def pretrain_source_only(model, source_loader, target_test, config):
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    best_score = 0.0
    best_state = copy.deepcopy(model.state_dict())
    for epoch in range(1, config["pretrain_epochs"] + 1):
        model.train()
        total_loss = 0.0
        for x_s, e_s, t_s in source_loader:
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
            f"  Source-Only Epoch {epoch:02d} | loss={total_loss/len(source_loader):.4f} | target_cindex={score:.4f}"
        )
    model.load_state_dict(best_state)
    return best_score, best_state


def pretrain_with_alignment(
    model,
    source_loader,
    target_loader,
    target_test,
    config,
    align_mode,
    pos_weight=None,
    neg_weight=None,
):
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    best_score = 0.0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, config["da_epochs"] + 1):
        model.train()
        total_surv = 0.0
        total_align = 0.0
        current_lambda = config["align_lambda_max"] * epoch / config["da_epochs"]
        target_iter = iter(target_loader)

        for x_s, e_s, t_s in source_loader:
            try:
                x_t, _, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                x_t, _, _ = next(target_iter)

            x_s = x_s.to(config["device"])
            e_s = e_s.to(config["device"])
            t_s = t_s.to(config["device"])
            x_t = x_t.to(config["device"])

            optimizer.zero_grad()
            emb_s, hazard_s = model(x_s)
            emb_t, _ = model(x_t)
            surv = cox_loss(hazard_s, e_s, t_s)

            if align_mode == "coral":
                align = coral_loss(emb_s, emb_t)
            elif align_mode == "mmd":
                align = mmd_loss(emb_s, emb_t)
            elif align_mode == "lsa_mmd":
                align = weighted_mmd_loss(emb_s, emb_t, e_s, pos_weight, neg_weight)
            else:
                raise ValueError(f"Unsupported align mode: {align_mode}")

            loss = surv + current_lambda * align
            loss.backward()
            optimizer.step()
            total_surv += surv.item()
            total_align += align.item()

        score = evaluate_cindex(model, *target_test, config["device"])
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
        print(
            f"  {align_mode.upper()} Epoch {epoch:02d} | surv={total_surv/len(source_loader):.4f} | "
            f"align={total_align/len(source_loader):.4f} | target_cindex={score:.4f}"
        )

    model.load_state_dict(best_state)
    return best_score, best_state


def supervised_adaptation(model, target_adapt_loader, target_test, config):
    for param in model.encoder.parameters():
        param.requires_grad = False

    optimizer = optim.AdamW(
        model.hazard_head.parameters(),
        lr=config["finetune_lr"],
        weight_decay=1e-3,
    )
    best_score = 0.0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, config["finetune_epochs"] + 1):
        model.train()
        total_loss = 0.0
        for x_t, e_t, t_t in target_adapt_loader:
            x_t = x_t.to(config["device"])
            e_t = e_t.to(config["device"])
            t_t = t_t.to(config["device"])
            optimizer.zero_grad()
            _, hazard = model(x_t)
            loss = cox_loss(hazard, e_t, t_t)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        score = evaluate_cindex(model, *target_test, config["device"])
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
        print(
            f"  SUPERVISED_FT Epoch {epoch:02d} | loss={total_loss/len(target_adapt_loader):.4f} | target_cindex={score:.4f}"
        )

    model.load_state_dict(best_state)
    return best_score, best_state


def main():
    set_seed(CONFIG["seed"])
    print("1. Loading relative clinical features and fixed target split...")
    x_s, e_s, t_s, scaler = load_relative_features(CONFIG["source_data"], is_training=True)
    x_t, e_t, t_t, _ = load_relative_features(
        CONFIG["target_data"], is_training=False, scaler=scaler
    )

    x_t_adapt, x_t_test, e_t_adapt, e_t_test, t_t_adapt, t_t_test = train_test_split(
        x_t,
        e_t,
        t_t,
        test_size=1 - CONFIG["target_adapt_ratio"],
        random_state=CONFIG["seed"],
        stratify=e_t,
    )

    source_loader = make_loader(
        x_s, e_s, t_s, CONFIG["batch_size"], shuffle=True, drop_last=True
    )
    target_unlabeled_loader = make_loader(
        x_t_adapt,
        e_t_adapt,
        t_t_adapt,
        CONFIG["batch_size"],
        shuffle=True,
        drop_last=True,
    )
    target_adapt_loader = make_loader(
        x_t_adapt,
        e_t_adapt,
        t_t_adapt,
        CONFIG["batch_size"],
        shuffle=True,
        drop_last=False,
    )
    target_test = (x_t_test, e_t_test, t_t_test)

    print(f"   Source samples        : {len(x_s)}")
    print(f"   Target adapt samples  : {len(x_t_adapt)}")
    print(f"   Target test samples   : {len(x_t_test)}")
    print(f"   Source event rate     : {e_s.mean():.4f}")
    print(f"   Target adapt rate     : {e_t_adapt.mean():.4f}")

    pos_weight = float(e_t_adapt.mean() / (e_s.mean() + 1e-8))
    neg_weight = float((1 - e_t_adapt.mean()) / (1 - e_s.mean() + 1e-8))

    results = {}
    input_dim = x_s.shape[1]

    print("\n2. Benchmark: Source-Only")
    source_model = SurvivalNet(input_dim, CONFIG["d_model"], CONFIG["dropout"]).to(
        CONFIG["device"]
    )
    score, source_state = pretrain_source_only(
        source_model, source_loader, target_test, CONFIG
    )
    results["Source-Only"] = {"target_cindex": score}

    print("\n3. Benchmark: CORAL")
    coral_model = SurvivalNet(input_dim, CONFIG["d_model"], CONFIG["dropout"]).to(
        CONFIG["device"]
    )
    score, _ = pretrain_with_alignment(
        coral_model,
        source_loader,
        target_unlabeled_loader,
        target_test,
        CONFIG,
        "coral",
    )
    results["CORAL"] = {"target_cindex": score}

    print("\n4. Benchmark: MMD")
    mmd_model = SurvivalNet(input_dim, CONFIG["d_model"], CONFIG["dropout"]).to(
        CONFIG["device"]
    )
    score, _ = pretrain_with_alignment(
        mmd_model,
        source_loader,
        target_unlabeled_loader,
        target_test,
        CONFIG,
        "mmd",
    )
    results["MMD"] = {"target_cindex": score}

    print("\n5. Benchmark: Label-Shift-Aware MMD")
    lsa_model = SurvivalNet(input_dim, CONFIG["d_model"], CONFIG["dropout"]).to(
        CONFIG["device"]
    )
    score, _ = pretrain_with_alignment(
        lsa_model,
        source_loader,
        target_unlabeled_loader,
        target_test,
        CONFIG,
        "lsa_mmd",
        pos_weight=pos_weight,
        neg_weight=neg_weight,
    )
    results["Label-Shift-Aware MMD"] = {
        "target_cindex": score,
        "pos_weight": pos_weight,
        "neg_weight": neg_weight,
    }

    print("\n6. Benchmark: Supervised DA Fine-tune")
    supervised_model = SurvivalNet(input_dim, CONFIG["d_model"], CONFIG["dropout"]).to(
        CONFIG["device"]
    )
    supervised_model.load_state_dict(source_state)
    zero_shot_score = evaluate_cindex(supervised_model, *target_test, CONFIG["device"])
    finetuned_score, _ = supervised_adaptation(
        supervised_model,
        target_adapt_loader,
        target_test,
        CONFIG,
    )
    results["Supervised DA Fine-tune"] = {
        "zero_shot_target_cindex": zero_shot_score,
        "target_cindex": finetuned_score,
        "improvement_vs_zero_shot": finetuned_score - zero_shot_score,
    }

    with open(CONFIG["results_path"], "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n============================================================")
    print("Final Benchmark Summary")
    for method, payload in results.items():
        if "improvement_vs_zero_shot" in payload:
            print(
                f"{method:<28} target_cindex={payload['target_cindex']:.4f} "
                f"(delta={payload['improvement_vs_zero_shot']:+.4f})"
            )
        else:
            print(f"{method:<28} target_cindex={payload['target_cindex']:.4f}")
    print(f"Results saved to: {CONFIG['results_path']}")
    print("============================================================")


if __name__ == "__main__":
    main()
