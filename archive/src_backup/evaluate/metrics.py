import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index
from sklearn.metrics import brier_score_loss
import torch
import torch.nn as nn
from sksurv.metrics import brier_score, cumulative_dynamic_auc

def compute_ipcw_weights(t_fit, e_fit, t_eval=None):
    kmf = KaplanMeierFitter()
    kmf.fit(t_fit, 1 - e_fit)
    
    if t_eval is None:
        t_eval = t_fit
        
    g_t = kmf.survival_function_at_times(t_eval).values
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

def evaluate_survival_metrics(model, x_eval, e_eval, t_eval, device="cpu", batch_size=1024, t_train=None, e_train=None):
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
    c_index = concordance_index(t_eval, -risk_scores, e_eval)
    
    # Optional: Time-dependent metrics if train data is provided for censoring distribution
    metrics = {"C-index": c_index}
    
    if t_train is not None and e_train is not None:
        try:
            # Create structured arrays required by scikit-survival
            y_train = np.array([(bool(e), t) for e, t in zip(e_train, t_train)], dtype=[('Status', '?'), ('Survival_in_days', '<f8')])
            y_eval = np.array([(bool(e), t) for e, t in zip(e_eval, t_eval)], dtype=[('Status', '?'), ('Survival_in_days', '<f8')])
            
            # Select time points for evaluation (e.g., 30, 60, 120 minutes)
            # Ensure time points are within the valid range of both train and eval
            min_time = max(t_train[e_train==1].min(), t_eval[e_eval==1].min()) + 1
            max_time = min(t_train.max(), t_eval.max()) - 1
            
            times = np.array([30.0, 60.0, 120.0])
            valid_times = times[(times >= min_time) & (times <= max_time)]
            
            if len(valid_times) > 0:
                # We need survival probabilities for Brier score, but Cox outputs hazards.
                # As a simplified proxy for Brier Score evaluation of relative risks, we compute Time-dependent AUC
                auc, mean_auc = cumulative_dynamic_auc(y_train, y_eval, risk_scores, valid_times)
                for t, a in zip(valid_times, auc):
                    metrics[f"AUC_{int(t)}m"] = a
        except Exception as e:
            print(f"Warning: Could not compute time-dependent metrics: {e}")
            
    return metrics
