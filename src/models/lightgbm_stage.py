import numpy as np
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

class LightGBMStageClassifier:
    def __init__(self, num_class=4, class_weight=None, random_state=42, early_priority_weights=None, calibrate=False):
        self.num_class = num_class
        self.class_weight = class_weight
        self.random_state = random_state
        self.early_priority_weights = early_priority_weights or [1.0, 1.0, 1.0, 1.0]
        self.calibrate = calibrate
        self.model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=self.num_class,
            random_state=self.random_state,
            class_weight=self.class_weight,
            num_leaves=63,
            learning_rate=0.05,
            n_estimators=2000,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=1,
            min_data_in_leaf=80,
            reg_alpha=0.1,
            reg_lambda=0.6,
            metric="multi_logloss",
        )
        self.calibrator = None

    def fit(self, X, y):
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=self.random_state, stratify=y)
        if self.class_weight is None:
            uniques, counts = np.unique(y_tr, return_counts=True)
            total = y_tr.shape[0]
            cw = {int(cls): float(total / (len(uniques) * cnt)) for cls, cnt in zip(uniques.tolist(), counts.tolist())}
            self.model.set_params(class_weight=cw)
        sw = np.ones(y_tr.shape[0], dtype=float)
        # 数据驱动的类强调（按逆频率归一）
        uniques, counts = np.unique(y_tr, return_counts=True)
        inv_freq = {int(c): float(total / (len(uniques) * n)) for c, n in zip(uniques.tolist(), counts.tolist())}
        for c, w in inv_freq.items():
            sw[y_tr == c] *= w
        # 早期与中期额外强调
        sw[y_tr == 1] *= 1.5
        sw[y_tr == 2] *= 1.2
        self.model.fit(
            X_tr, y_tr,
            sample_weight=sw,
            eval_set=[(X_val, y_val)],
            eval_metric="multi_logloss",
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(period=100)]
        )
        if self.calibrate:
            try:
                self.calibrator = CalibratedClassifierCV(self.model, cv="prefit", method="sigmoid")
                self.calibrator.fit(X_val, y_val)
            except Exception:
                self.calibrator = None
        return self

    def predict_proba(self, X):
        if self.calibrator is not None:
            proba = self.calibrator.predict_proba(X)
        else:
            proba = self.model.predict_proba(X)
        # 规范化形状为 (n_samples, n_classes)
        if proba.ndim == 2 and proba.shape[0] == self.num_class and proba.shape[1] != self.num_class:
            proba = proba.T
        # 行归一化
        denom = np.sum(proba, axis=1, keepdims=True)
        denom = np.where(denom == 0, 1.0, denom)
        proba = proba / denom
        return proba

    def predict(self, X):
        proba = self.predict_proba(X)
        w = np.array(self.early_priority_weights[:self.num_class])
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