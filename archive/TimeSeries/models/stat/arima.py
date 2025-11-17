from TimeSeries.core.base_forecaster import BaseForecaster
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

class ARIMAForecaster(BaseForecaster):
    def __init__(self, order=(1,0,0)):
        self.order = order
        self.model = None
        self.fitted = None

    def fit(self, y, X=None):
        self.model = ARIMA(np.asarray(y), order=self.order)
        self.fitted = self.model.fit()
        return self

    def predict(self, steps):
        return self.fitted.forecast(steps=steps)

    def get_params(self):
        return {'order': self.order}