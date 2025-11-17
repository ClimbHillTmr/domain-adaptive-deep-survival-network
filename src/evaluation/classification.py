import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

def compute_basic_metrics(y_true, y_pred):
    return {
        'f1_macro': float(f1_score(y_true, y_pred, average='macro')),
        'f1_micro': float(f1_score(y_true, y_pred, average='micro')),
        'precision_macro': float(precision_score(y_true, y_pred, average='macro')),
        'recall_macro': float(recall_score(y_true, y_pred, average='macro')),
    }

def per_stage_recall(y_true, y_pred, stages=(0,1,2,3)):
    out = {}
    for s in stages:
        mask = (y_true == s)
        if np.sum(mask) == 0:
            out[f'recall_stage_{s}'] = None
        else:
            out[f'recall_stage_{s}'] = float(np.mean(y_pred[mask] == s))
    return out

def brier_score(y_true, proba, stages=(0,1,2,3)):
    Y = np.zeros((len(y_true), len(stages)))
    for i, s in enumerate(stages):
        Y[:, i] = (y_true == s).astype(float)
    return float(np.mean(np.sum((proba - Y)**2, axis=1)))

def stage_misalignment(y_true, y_pred):
    # 偏早（负偏差）/偏晚（正偏差）分布
    diff = y_pred.astype(int) - y_true.astype(int)
    return {
        'early_bias_rate': float(np.mean(diff < 0)),
        'late_bias_rate': float(np.mean(diff > 0)),
        'mean_offset': float(np.mean(diff)),
    }

def confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return cm