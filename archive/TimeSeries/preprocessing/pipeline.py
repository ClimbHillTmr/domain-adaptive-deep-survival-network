import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import Dict, Any, Tuple

class TSPreprocessor:
    def __init__(self, scaler='standard', impute='simple'):
        self.scaler_name = scaler
        self.impute = impute
        self.scaler = None

    def _make_scaler(self):
        if self.scaler_name == 'minmax':
            return MinMaxScaler()
        if self.scaler_name == 'robust':
            return RobustScaler()
        return StandardScaler()

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop_duplicates()
        df = df.replace(['NA', 'NaN', None], np.nan)
        if self.impute == 'simple':
            df = df.fillna(method='ffill').fillna(method='bfill')
        return df

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.scaler = self._make_scaler()
        shape = X.shape
        X2 = X.reshape(-1, shape[-1])
        X2 = self.scaler.fit_transform(X2)
        return X2.reshape(shape)

    def transform(self, X: np.ndarray) -> np.ndarray:
        shape = X.shape
        X2 = X.reshape(-1, shape[-1])
        X2 = self.scaler.transform(X2)
        return X2.reshape(shape)

def make_windows(series: np.ndarray, window: int, horizon: int=1) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(len(series) - window - horizon + 1):
        X.append(series[i:i+window])
        y.append(series[i+window:i+window+horizon])
    return np.array(X), np.array(y)