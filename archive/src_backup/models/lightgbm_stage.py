import lightgbm as lgb
import numpy as np

class LightGBMStage:
    def __init__(self, params):
        self.params = params
        self.model = None

    def fit(self, X, y, X_val=None, y_val=None):
        train_data = lgb.Dataset(X, label=y)
        valid_sets = [train_data]
        if X_val is not None:
            valid_data = lgb.Dataset(X_val, label=y_val)
            valid_sets.append(valid_data)
            
        self.model = lgb.train(
            self.params,
            train_data,
            valid_sets=valid_sets,
            num_boost_round=1000,
            early_stopping_rounds=50,
            verbose_eval=False
        )

    def predict(self, X):
        return self.model.predict(X)
