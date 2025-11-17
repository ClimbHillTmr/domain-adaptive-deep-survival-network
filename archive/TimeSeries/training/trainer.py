import json
import os
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit

class Trainer:
    def __init__(self, model, cv_splits=5, early_stopping_patience=10, metric_fn=None):
        self.model = model
        self.cv_splits = cv_splits
        self.patience = early_stopping_patience
        self.metric_fn = metric_fn

    def fit_cv(self, X, y):
        tss = TimeSeriesSplit(n_splits=self.cv_splits)
        scores = []
        for train_idx, val_idx in tss.split(X):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            self.model.fit(X_tr, y_tr)
            y_pred = self.model.predict(X_val)
            score = self.metric_fn(y_val, y_pred) if self.metric_fn else None
            scores.append(score)
        return scores

    def save_run(self, base_dir, metrics, params=None):
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out = os.path.join(base_dir, f'Run_{ts}')
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        if params:
            with open(os.path.join(out, 'params.json'), 'w') as f:
                json.dump(params, f, indent=2)
        return out