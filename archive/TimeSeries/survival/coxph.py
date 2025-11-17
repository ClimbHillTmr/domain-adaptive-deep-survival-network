import numpy as np
import pandas as pd
from typing import Dict, Any

AVAILABLE = True
try:
    from lifelines import CoxPHFitter
    from lifelines.utils import k_fold_cross_validation
except Exception:
    AVAILABLE = False

class CoxPHSurvival:
    def __init__(self, duration_col='duration', event_col='event', penalizer=0.0, l1_ratio=0.0):
        self.duration_col = duration_col
        self.event_col = event_col
        self.penalizer = penalizer
        self.l1_ratio = l1_ratio
        self.model = None

    def fit(self, X: pd.DataFrame, durations: np.ndarray, events: np.ndarray):
        if not AVAILABLE:
            raise ImportError('lifelines not available')
        df = X.copy()
        df[self.duration_col] = durations
        df[self.event_col] = events
        self.model = CoxPHFitter(penalizer=self.penalizer, l1_ratio=self.l1_ratio)
        self.model.fit(df, duration_col=self.duration_col, event_col=self.event_col)
        return self

    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_partial_hazard(X).values

    def predict_survival(self, X: pd.DataFrame, times: np.ndarray) -> np.ndarray:
        sf = self.model.predict_survival_function(X, times=times)
        return np.stack([sf.iloc[:, i].values for i in range(sf.shape[1])], axis=0)

    def explain(self) -> Dict[str, Any]:
        summary = self.model.summary
        out = {
            'coefficients': summary['coef'].to_dict(),
            'hazard_ratio': summary['exp(coef)'].to_dict(),
            'ci_lower': summary['exp(coef) lower 95%'].to_dict(),
            'ci_upper': summary['exp(coef) upper 95%'].to_dict(),
            'p_values': summary['p'].to_dict()
        }
        return out