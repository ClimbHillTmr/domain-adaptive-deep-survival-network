import torch
import numpy as np

class TorchStageWrapper:
    def __init__(self, model, threshold=None):
        self.model = model
        self.device = next(model.parameters()).device
        self.threshold = threshold
        self.model.eval()

    def predict_proba(self, X_seq, X_static=None):
        with torch.no_grad():
            xs = torch.tensor(X_seq, dtype=torch.float32).to(self.device)
            xst = None if X_static is None else torch.tensor(X_static, dtype=torch.float32).to(self.device)
            logits = self.model(xs, xst)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs = np.clip(probs, 1e-12, 1.0)
        row_sum = probs.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0.0] = 1.0
        probs = probs / row_sum
        if probs.shape[0] < probs.shape[1]:
            probs = probs.T
        return probs

    def predict(self, X_seq, X_static=None):
        proba = self.predict_proba(X_seq, X_static)
        return np.argmax(proba, axis=1)