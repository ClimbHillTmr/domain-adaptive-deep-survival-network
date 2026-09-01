"""v6.0 — 主线三 family 训练 runner (P2/P3)。

任务：
  target_only        : 仅 target_update(budget) 训练
  source_only        : 仅 source_train 预训练（零样本迁移，target 上直接评估）
  supervised_update  : source_train 预训练 → target_update(budget) 微调（主处理）

- 统一损失：patient_balanced_nll（单一真值来源，见 src/v6_acc/loss.py）
- 统一骨架：DiscreteHazardMLP
- 统一目标构造：src/v6_acc/loader.discrete_survival_targets（禁止内联重写）
- 输出到 experiments/v6_20260830/{family}__{endpoint}__{budget}__{seed}/
    evaluation.json    : 聚合 IPCW-AUC/Brier
    epoch_log.csv      : 逐 epoch 训练损失
    test_predictions.csv : target_test 的 session 级预测（patient_id / survival_time /
                          event / risk_60..240），patient-cluster bootstrap 推断的前提
    checkpoint.pt      : 早停最优权重（CPU 张量）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluate.discrete_survival_metrics import (
    evaluate_discrete_survival,
    cumulative_risk_from_hazards,
)
from src.v6_acc.loader import (
    FEATURES,
    discrete_survival_targets,
    load_budget,
    load_cohorts,
    load_split,
    session_batch,
    session_patient_counts,
)
from src.v6_acc.loss import patient_balanced_nll
from src.v6_acc.model import DiscreteHazardMLP

BUDGETS = (0.10, 0.25, 0.50, 1.00)
FAMILIES = ("target_only", "source_only", "supervised_update")
SEEDS = (401, 402, 433, 202, 107)  # v6 复现种子
HORIZONS = [60.0, 120.0, 180.0, 240.0]


def _feature_matrix(view: pd.DataFrame) -> np.ndarray:
    return view[FEATURES].to_numpy(dtype=np.float32)


def _view_inputs(
    view: pd.DataFrame, endpoint: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """返回 (X, targets, observed, patient_counts) 全量张量输入。

    endpoint ∈ {idh, ih}。离散目标一律取自
    `loader.discrete_survival_targets`（v6 单一真值来源，禁止内联重写）。
    """
    events = view[f"{endpoint}_event_observed_240"]
    intervals = view[f"{endpoint}_event_interval_60"]
    y, o = discrete_survival_targets(events, intervals)
    X = _feature_matrix(view)
    counts = session_patient_counts(view)
    return X, y, o, counts


def _run_training(
    family: str, endpoint: str, budget: float, seed: int, device: torch.device,
    out_dir: Path,
    max_pretrain_epochs: int = 40, max_update_epochs: int = 20, patience: int = 5,
) -> dict:
    """执行一个 family×endpoint×budget×seed 的训练与评估，返回结果 dict。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    src, tgt = load_cohorts()
    split = load_split()
    budget_df = load_budget()

    model = DiscreteHazardMLP(seed=seed).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    def _make_loader(view, target_y, target_o, counts, batch_size=256, shuffle=True):
        Xt = torch.tensor(_feature_matrix(view), dtype=torch.float32)
        yt = torch.tensor(target_y, dtype=torch.float32)
        ot = torch.tensor(target_o, dtype=torch.float32)
        ct = torch.tensor(counts, dtype=torch.float32)
        ds = TensorDataset(Xt, yt, ot, ct)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    def _train_phase(loader, max_epochs, lr, tag):
        torch.manual_seed(seed)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        model.train()
        history = []
        best_loss = float("inf")
        best_state = None
        stall = 0
        for epoch in range(max_epochs):
            total, nb = 0.0, 0
            for Xb, yb, ob, cb in loader:
                Xb, yb, ob, cb = Xb.to(device), yb.to(device), ob.to(device), cb.to(device)
                optimizer.zero_grad()
                logits = model(Xb)
                loss = patient_balanced_nll(logits, yb, ob, cb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total += loss.item() * len(Xb)
                nb += len(Xb)
            mean = total / max(nb, 1)
            history.append({"phase": tag, "epoch": epoch + 1, "loss": mean})
            if mean < best_loss:
                best_loss = mean
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stall = 0
            else:
                stall += 1
                if stall >= patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
            torch.save(best_state, out_dir / f"checkpoint_{tag}.pt")
        return history

    # ---- 固定 target_update 规模（budget 相关），供所有 family 复用 ----
    update_view = session_batch(
        (src, tgt), split, budget_df, "target_update", center="target", budget_frac=budget,
    )
    n_update_sessions = int(len(update_view))

    # ---- 组装 view 与训练流程 ----
    if family == "target_only":
        Xy, yo, oo, co = _view_inputs(update_view, endpoint)
        loader = _make_loader(update_view, yo, oo, co)
        hist = _train_phase(loader, max_update_epochs, 1e-4, "update")
    elif family == "source_only":
        src_view = session_batch((src, tgt), split, budget_df, "source_train", center="source")
        Xy, yo, oo, co = _view_inputs(src_view, endpoint)
        loader = _make_loader(src_view, yo, oo, co)
        hist = _train_phase(loader, max_pretrain_epochs, 1e-3, "pretrain")
    elif family == "supervised_update":
        src_view = session_batch((src, tgt), split, budget_df, "source_train", center="source")
        Xy, yo, oo, co = _view_inputs(src_view, endpoint)
        loader = _make_loader(src_view, yo, oo, co)
        hist = _train_phase(loader, max_pretrain_epochs, 1e-3, "pretrain")
        Xy2, yo2, oo2, co2 = _view_inputs(update_view, endpoint)
        loader2 = _make_loader(update_view, yo2, oo2, co2)
        hist += _train_phase(loader2, max_update_epochs, 1e-4, "update")

    # ---- 评估于 target_test（held-out 外部中心）----
    test_view = session_batch((src, tgt), split, budget_df, "target_test", center="target")
    model.eval()
    with torch.no_grad():
        Xt = torch.tensor(_feature_matrix(test_view), dtype=torch.float32).to(device)
        logits = model(Xt).cpu().numpy()
    risk = cumulative_risk_from_hazards(logits, input_type="logits")
    res = evaluate_discrete_survival(
        test_view[f"{endpoint}_survival_time"].to_numpy(),
        test_view[f"{endpoint}_event_observed_240"].to_numpy(),
        risk,
        horizons=np.array(HORIZONS),
    )

    # ---- 保存 target_test 的 session 级预测，供 patient-cluster bootstrap 重估 ----
    test_sessions = pd.DataFrame({
        "patient_id": test_view["患者id"].astype(str).to_numpy(),
        f"{endpoint}_survival_time": test_view[f"{endpoint}_survival_time"].to_numpy().astype(float),
        f"{endpoint}_event_observed_240": test_view[f"{endpoint}_event_observed_240"].to_numpy().astype(int),
    })
    for i, h in enumerate(HORIZONS):
        test_sessions[f"risk_{int(h)}"] = risk[:, i]
    (out_dir / "test_predictions.csv").write_text(test_sessions.to_csv(index=False), encoding="utf-8")

    return {
        "family": family, "endpoint": endpoint, "budget": budget, "seed": seed,
        "mean_auc": float(res["mean_auc"]), "mean_brier": float(res["mean_brier"]),
        "auc": [float(x) for x in res["auc"]],
        "brier": [float(x) for x in res["brier"]],
        "n_test_sessions": int(len(test_view)),
        "n_update_sessions": n_update_sessions,
        "epoch_log": hist,
    }


def _save_run(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ev = {k: v for k, v in result.items() if k != "epoch_log"}
    (out_dir / "evaluation.json").write_text(json.dumps(ev, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "epoch_log.csv").write_text(
        pd.DataFrame(result.get("epoch_log", [])).to_csv(index=False),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True, choices=FAMILIES)
    parser.add_argument("--endpoint", required=True, choices=("idh", "ih"))
    parser.add_argument("--budget", type=float, required=True, choices=BUDGETS)
    parser.add_argument("--seed", type=int, required=True, choices=SEEDS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-root", default=str(ROOT / "experiments/v6_20260830"))
    parser.add_argument("--max-pretrain-epochs", type=int, default=40)
    parser.add_argument("--max-update-epochs", type=int, default=20)
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    out_dir = Path(args.out_root) / (
        f"{args.family}__{args.endpoint}__{args.budget:.2f}__seed{args.seed}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    result = _run_training(
        args.family, args.endpoint, args.budget, args.seed, device, out_dir,
        args.max_pretrain_epochs, args.max_update_epochs,
    )
    _save_run(result, out_dir)
    result["elapsed_sec"] = round(time.time() - t0, 1)
    print(json.dumps({k: v for k, v in result.items() if k != "epoch_log"}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())