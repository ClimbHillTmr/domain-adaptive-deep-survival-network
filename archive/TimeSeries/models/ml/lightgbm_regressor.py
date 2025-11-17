from TimeSeries.core.base_forecaster import BaseForecaster
import lightgbm as lgb
import joblib

class LightGBMRegressor(BaseForecaster):
    def __init__(self, **params):
        self.params = params or {}
        self.model = lgb.LGBMRegressor(**self.params)

    def fit(self, X, y=None):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def get_params(self):
        return self.params

    def save(self, path: str):
        joblib.dump(self.model, path)

    def load(self, path: str):
        self.model = joblib.load(path)
        return self