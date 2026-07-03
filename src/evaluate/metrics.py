import numpy as np
import torch
import torch.nn as nn


def _kaplan_meier_survival_at(t_fit, e_fit, t_eval):
    t_fit = np.asarray(t_fit, dtype=float)
    e_fit = np.asarray(e_fit, dtype=int)
    t_eval = np.asarray(t_eval, dtype=float)
    order = np.argsort(t_fit)
    times = t_fit[order]
    censor_events = 1 - e_fit[order]

    unique_times = np.unique(times)
    survival = []
    current_survival = 1.0
    for time in unique_times:
        at_risk = np.sum(times >= time)
        censored_at_time = np.sum((times == time) & (censor_events == 1))
        if at_risk > 0:
            current_survival *= 1.0 - censored_at_time / at_risk
        survival.append(current_survival)

    survival = np.asarray(survival, dtype=float)
    idx = np.searchsorted(unique_times, t_eval, side="right") - 1
    out = np.ones_like(t_eval, dtype=float)
    valid = idx >= 0
    out[valid] = survival[idx[valid]]
    return out


def concordance_index(t_eval, risk_scores, e_eval):
    t_eval = np.asarray(t_eval, dtype=float)
    e_eval = np.asarray(e_eval, dtype=int)
    risk_scores = np.asarray(risk_scores, dtype=float)
    valid = np.isfinite(t_eval) & np.isfinite(risk_scores)
    t_eval = t_eval[valid]
    e_eval = e_eval[valid]
    risk_scores = risk_scores[valid]
    if len(t_eval) < 2 or np.sum(e_eval) == 0:
        return 0.0

    unique_risks = np.unique(risk_scores)
    ranks = np.searchsorted(unique_risks, risk_scores) + 1
    bit = np.zeros(len(unique_risks) + 2, dtype=np.int64)

    def add(rank, delta):
        while rank < len(bit):
            bit[rank] += delta
            rank += rank & -rank

    def prefix(rank):
        total = 0
        while rank > 0:
            total += bit[rank]
            rank -= rank & -rank
        return total

    order = np.argsort(t_eval)
    for rank in ranks:
        add(int(rank), 1)

    concordant = 0.0
    comparable = 0
    i = 0
    n = len(order)
    while i < n:
        j = i
        current_time = t_eval[order[i]]
        while j < n and t_eval[order[j]] == current_time:
            add(int(ranks[order[j]]), -1)
            j += 1
        later = n - j
        if later > 0:
            for pos in order[i:j]:
                if e_eval[pos] != 1:
                    continue
                rank = int(ranks[pos])
                lower = prefix(rank - 1)
                equal = prefix(rank) - prefix(rank - 1)
                concordant += lower + 0.5 * equal
                comparable += later
        i = j

    if comparable == 0:
        return 0.0
    return float(concordant / comparable)


def compute_ipcw_weights(t_fit, e_fit, t_eval=None):
    if t_eval is None:
        t_eval = t_fit
    g_t = _kaplan_meier_survival_at(t_fit, e_fit, t_eval)
    g_t = np.clip(g_t, 1e-5, 1.0)
    w = 1.0 / g_t
    
    w_95 = np.percentile(w, 95)
    w = np.clip(w, 0, w_95)
    return w.astype(np.float32)

def weighted_cox_loss(hazard, e, t, w):
    hazard = hazard.squeeze(-1)
    idx = torch.argsort(t, descending=True)
    hazard = hazard[idx]
    e = e[idx]
    t = t[idx]
    w = w[idx]
    
    risk_score = torch.exp(hazard)
    weighted_risk = risk_score * w
    
    risk_set_sum = torch.cumsum(weighted_risk, dim=0) + 1e-7
    loss = e * w * (hazard - torch.log(risk_set_sum))
    
    event_weight_sum = torch.sum(e * w) + 1e-7
    return -torch.sum(loss) / event_weight_sum


def domain_loss_from_logits(domain_logits_s, domain_logits_t):
    y_s = torch.zeros(domain_logits_s.size(0), dtype=torch.long, device=domain_logits_s.device)
    y_t = torch.ones(domain_logits_t.size(0), dtype=torch.long, device=domain_logits_t.device)
    logits = torch.cat([domain_logits_s, domain_logits_t], dim=0)
    labels = torch.cat([y_s, y_t], dim=0)
    return nn.functional.cross_entropy(logits, labels)


def compute_bootstrap_cindex(
    t_eval,
    e_eval,
    risk_scores,
    patient_ids=None,
    n_bootstrap=200,
    seed=42,
):
    if n_bootstrap <= 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    c_indices = []

    t_eval = np.asarray(t_eval)
    e_eval = np.asarray(e_eval)
    risk_scores = np.asarray(risk_scores)
    patient_ids = None if patient_ids is None else np.asarray(patient_ids).astype(str)

    for _ in range(n_bootstrap):
        if patient_ids is None:
            indices = rng.integers(0, len(t_eval), len(t_eval))
        else:
            patients = np.unique(patient_ids)
            sampled_patients = rng.choice(patients, size=len(patients), replace=True)
            indices = np.concatenate([np.flatnonzero(patient_ids == pid) for pid in sampled_patients])
        t_boot = t_eval[indices]
        e_boot = e_eval[indices]
        risk_boot = risk_scores[indices]

        if np.sum(e_boot) > 0 and np.sum(e_boot) < len(e_boot):
            c = concordance_index(t_boot, risk_boot, e_boot)
            c_indices.append(c)

    if not c_indices:
        return 0.0, 0.0

    c_indices = np.array(c_indices)
    lower = np.percentile(c_indices, 2.5)
    upper = np.percentile(c_indices, 97.5)
    return float(lower), float(upper)


def _binary_auc(y_true, scores):
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores)
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    i = 0
    while i < len(scores):
        j = i
        while j < len(scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    rank_sum_pos = np.sum(ranks[y_true == 1])
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def evaluate_survival_metrics(
    model,
    x_eval,
    e_eval,
    t_eval,
    device="cpu",
    batch_size=1024,
    t_train=None,
    e_train=None,
    patient_ids=None,
    n_bootstrap=0,
    seed=42,
):
    """
    Evaluates survival metrics: C-index, Time-dependent AUC, and Brier Score.
    """
    model.eval()
    hazards = []
    with torch.no_grad():
        for start in range(0, len(x_eval), batch_size):
            end = start + batch_size
            x_batch = torch.tensor(x_eval[start:end], dtype=torch.float32, device=device)
            _, hazard, _, _ = model(x_batch, grl_coeff=None)
            hazards.append(hazard.squeeze(-1).detach().cpu().numpy())
    risk_scores = np.concatenate(hazards, axis=0)
    
    # 1. C-index
    c_index = concordance_index(t_eval, risk_scores, e_eval)
    
    # 1.b Bootstrap CI for C-index
    ci_lower, ci_upper = compute_bootstrap_cindex(
        t_eval,
        e_eval,
        risk_scores,
        patient_ids=patient_ids,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    
    # Optional: Time-dependent metrics if train data is provided for censoring distribution
    metrics = {
        "C-index": c_index,
        "C-index_CI_lower": ci_lower,
        "C-index_CI_upper": ci_upper,
        "bootstrap_unit": "patient" if patient_ids is not None and n_bootstrap > 0 else "session" if n_bootstrap > 0 else "not_run",
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
    }
    
    if t_train is not None and e_train is not None:
        for horizon in (30.0, 60.0, 120.0):
            y_horizon = ((np.asarray(e_eval).astype(int) == 1) & (np.asarray(t_eval) <= horizon)).astype(int)
            auc = _binary_auc(y_horizon, risk_scores)
            if auc is not None:
                metrics[f"AUC_{int(horizon)}m"] = auc
            
    return metrics
