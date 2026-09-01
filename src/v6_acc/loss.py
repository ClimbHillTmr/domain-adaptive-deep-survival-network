"""v6.0 — 单一 Patient-Balanced Discrete Survival NLL (全 family 唯一损失实现)。

这是训练损失的**唯一**实现（§3.2），root 掉 v4/v5 的损失函数双实现问题。

对 session i、interval k：
  z_ik = hazard logit，p_ik = sigmoid(z_ik)，o_ik = 观测 mask（在该格可观测）
  单样本 NLL：L_i = -Σ_k o_ik * [ y_ik·log p_ik + (1-y_ik)·log(1-p_ik) ]
  患者均衡权重：w_i = 1 / n_i（n_i = 该患者 session 数）
  总损失：L = -Σ_i w_i·L_i / Σ_i w_i

无 pos_weight、无 class_ratio（根除 v4 代际混杂）。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def patient_balanced_nll(
    hazard_logits: torch.Tensor,       # (n_sessions, 4)
    targets: torch.Tensor,             # (n_sessions, 4) 0/1
    observed: torch.Tensor,            # (n_sessions, 4) 0/1 观测 mask
    session_patient_counts: torch.Tensor,  # (n_sessions,) 每 session 所属患者的 session 总数 n_i
) -> torch.Tensor:
    """返回标量损失（均值化）。"""
    logits = hazard_logits.float()
    y = targets.float()
    o = observed.float()
    # 逐格 NLL（仅观测格参与）
    per_cell = F.binary_cross_entropy_with_logits(logits, y, reduction="none") * o
    # session 级 NLL 汇总
    session_nll = per_cell.sum(dim=1)                      # (n,)
    n_count = session_patient_counts.float().clamp(min=1.0)  # (n,)
    weights = 1.0 / n_count                                  # 患者均衡
    # 总损失 = 加权 Session NLL 的加权平均（we去除观测格数归一）
    denom = weights.sum()
    if denom.item() <= 0 or not torch.isfinite(denom):
        raise ValueError("patient-balanced NLL 分母非法")
    loss = (weights * session_nll).sum() / denom
    return loss


def _session_patient_counts(patient_ids: np.ndarray) -> np.ndarray:
    """返回数组，第 i 个元素 = patient_ids[i] 所属患者的 session 总数。"""
    patients, counts = np.unique(patient_ids, return_counts=True)
    mapping = {p: int(c) for p, c in zip(patients, counts, strict=True)}
    return np.asarray([mapping[str(p)] for p in patient_ids], dtype=float)


def build_loss_inputs(
    patient_ids: np.ndarray, hazard_logits_np: np.ndarray, targets_np: np.ndarray, observed_np: np.ndarray
) -> dict[str, torch.Tensor]:
    """快速把 numpy 输入打包成 torch 张量（供单 batch / 全量验证用）。"""
    counts = _session_patient_counts(patient_ids)
    return {
        "hazard_logits": torch.tensor(hazard_logits_np, dtype=torch.float32),
        "targets": torch.tensor(targets_np, dtype=torch.float32),
        "observed": torch.tensor(observed_np, dtype=torch.float32),
        "session_patient_counts": torch.tensor(counts, dtype=torch.float32),
    }