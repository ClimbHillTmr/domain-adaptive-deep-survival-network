import numpy as np
import lightgbm as lgb
import os
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split


class LightGBMStageClassifier:
    def __init__(
        self,
        num_class=4,
        class_weight=None,
        random_state=42,
        early_priority_weights=None,
        calibrate=True,
        calibration_method="sigmoid",
        cv_folds=1,
        seeds=None,
    ):
        self.num_class = num_class
        self.class_weight = class_weight
        self.random_state = random_state
        self.early_priority_weights = early_priority_weights or [1.0, 1.0, 1.0, 1.0]
        self.calibrate = calibrate
        self.calibration_method = calibration_method
        self.model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=self.num_class,
            random_state=self.random_state,
            class_weight=self.class_weight,
            num_leaves=127,
            max_depth=8,
            learning_rate=0.05,
            n_estimators=2000,
            colsample_bytree=0.8,
            subsample=0.8,
            subsample_freq=1,
            min_child_samples=80,
            reg_alpha=0.1,
            reg_lambda=0.6,
            metric="multi_logloss",
            n_jobs=os.cpu_count() or -1,
        )
        self.calibrator = None
        self.cv_folds = int(cv_folds) if cv_folds is not None else 1
        self.seeds = [int(s) for s in (seeds or [self.random_state])]
        self.models = []
        self.calibrators = []

    def fit(self, X, y):
        if self.cv_folds <= 1 and len(self.seeds) == 1:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X, y, test_size=0.2, random_state=self.random_state, stratify=y
            )
            uniques, counts = np.unique(y_tr, return_counts=True)
            total = y_tr.shape[0]
            if self.class_weight is None:
                cw = {
                    int(cls): float(total / (len(uniques) * cnt))
                    for cls, cnt in zip(uniques.tolist(), counts.tolist())
                }
                self.model.set_params(class_weight=cw)
            sw = np.ones(y_tr.shape[0], dtype=float)
            inv_freq = {
                int(c): float(total / (len(uniques) * n))
                for c, n in zip(uniques.tolist(), counts.tolist())
            }
            for c, w in inv_freq.items():
                sw[y_tr == c] *= w
            sw[y_tr == 1] *= 1.5
            sw[y_tr == 2] *= 1.2
            self.model.fit(
                X_tr,
                y_tr,
                sample_weight=sw,
                eval_set=[(X_val, y_val)],
                eval_metric="multi_logloss",
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(period=100)],
            )
            self.models = [self.model]
            if self.calibrate:
                try:
                    cal = CalibratedClassifierCV(
                        self.model, cv="prefit", method=self.calibration_method
                    )
                    cal.fit(X_val, y_val)
                    self.calibrators = [cal]
                except Exception:
                    self.calibrators = []
            return self
        self.models = []
        self.calibrators = []
        skf = StratifiedKFold(n_splits=max(2, self.cv_folds), shuffle=True, random_state=self.random_state)
        for fold_idx, (idx_tr, idx_val) in enumerate(skf.split(X, y)):
            seed = self.seeds[min(fold_idx, len(self.seeds)-1)]
            model = lgb.LGBMClassifier(
                objective="multiclass",
                num_class=self.num_class,
                random_state=seed,
                class_weight=self.class_weight,
                num_leaves=127,
                max_depth=8,
                learning_rate=0.05,
                n_estimators=2000,
                colsample_bytree=0.8,
                subsample=0.8,
                subsample_freq=1,
                min_child_samples=80,
                reg_alpha=0.1,
                reg_lambda=0.6,
                metric="multi_logloss",
                n_jobs=os.cpu_count() or -1,
            )
            X_tr, X_val = X[idx_tr], X[idx_val]
            y_tr, y_val = y[idx_tr], y[idx_val]
            uniques, counts = np.unique(y_tr, return_counts=True)
            total = y_tr.shape[0]
            if self.class_weight is None:
                cw = {
                    int(cls): float(total / (len(uniques) * cnt))
                    for cls, cnt in zip(uniques.tolist(), counts.tolist())
                }
                model.set_params(class_weight=cw)
            sw = np.ones(y_tr.shape[0], dtype=float)
            inv_freq = {
                int(c): float(total / (len(uniques) * n))
                for c, n in zip(uniques.tolist(), counts.tolist())
            }
            for c, w in inv_freq.items():
                sw[y_tr == c] *= w
            sw[y_tr == 1] *= 1.5
            sw[y_tr == 2] *= 1.2
            model.fit(
                X_tr,
                y_tr,
                sample_weight=sw,
                eval_set=[(X_val, y_val)],
                eval_metric="multi_logloss",
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(period=100)],
            )
            self.models.append(model)
            if self.calibrate:
                try:
                    cal = CalibratedClassifierCV(model, cv="prefit", method=self.calibration_method)
                    cal.fit(X_val, y_val)
                    self.calibrators.append(cal)
                except Exception:
                    pass
        if len(self.models) == 0:
            self.models = [self.model]
        return self

    def predict_proba(self, X):
        outs = []
        if self.calibrators:
            for cal in self.calibrators:
                outs.append(cal.predict_proba(X))
        elif self.models:
            for m in self.models:
                outs.append(m.predict_proba(X))
        else:
            outs.append(self.model.predict_proba(X))
        proba = np.mean(np.stack(outs, axis=0), axis=0)
        if (
            proba.ndim == 2
            and proba.shape[0] == self.num_class
            and proba.shape[1] != self.num_class
        ):
            proba = proba.T
        denom = np.sum(proba, axis=1, keepdims=True)
        denom = np.where(denom == 0, 1.0, denom)
        proba = proba / denom
        return proba

    def predict(self, X):
        proba = self.predict_proba(X)
        w = np.array(self.early_priority_weights[: self.num_class])
        scores = proba * w
        return np.argmax(scores, axis=1)

    def bootstrap_ci(self, proba, n_boot=200, alpha=0.05):
        rng = np.random.default_rng(self.random_state)
        n = proba.shape[0]
        means = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            means.append(proba[idx].mean(axis=0))
        means = np.stack(means, axis=0)
        lower = np.quantile(means, alpha / 2, axis=0)
        upper = np.quantile(means, 1 - alpha / 2, axis=0)
        return lower, upper
