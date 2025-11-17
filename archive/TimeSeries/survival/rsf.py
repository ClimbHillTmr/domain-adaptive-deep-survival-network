import numpy as np
import pandas as pd
from typing import Dict, Any

AVAILABLE = True
try:
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.util import Surv
except Exception:
    AVAILABLE = False

class RSFSurvival:
    def __init__(self, n_estimators=200, max_depth=None, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = None

    def fit(self, X: pd.DataFrame, durations: np.ndarray, events: np.ndarray):
        if not AVAILABLE:
            raise ImportError('scikit-survival not available')
        y = Surv.from_arrays(event=events.astype(bool), time=durations.astype(float))
        self.model = RandomSurvivalForest(n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=self.random_state)
        self.model.fit(X.values, y)
        return self

    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X.values)

    def predict_survival(self, X: pd.DataFrame) -> np.ndarray:
        return np.stack([fn(self.model.unique_times_) for fn in self.model.predict_survival_function(X.values)], axis=0)

    def explain(self) -> Dict[str, Any]:
        importances = self.model.feature_importances_
        return {'feature_importances': {col: float(imp) for col, imp in zip(X.columns, importances)}}