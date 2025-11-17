import pandas as pd
import numpy as np
from lifelines import CoxPHFitter

class CoxAdapter:
    def __init__(self, duration_col='duration', event_col='event', penalizer=0.1, l1_ratio=0.0):
        self.duration_col = duration_col
        self.event_col = event_col
        self.penalizer = penalizer
        self.l1_ratio = l1_ratio
        self.model = CoxPHFitter(penalizer=self.penalizer, l1_ratio=self.l1_ratio)

    def fit(self, X: pd.DataFrame, durations: np.ndarray, events: np.ndarray):
        df = X.copy()
        var = df.var(numeric_only=True)
        keep = var[(var > 1e-6) | var.isna()].index
        df = df[keep]
        df[self.duration_col] = durations
        df[self.event_col] = events
        self.model.fit(df, duration_col=self.duration_col, event_col=self.event_col)
        return self

    def survival_at(self, X: pd.DataFrame, times: np.ndarray):
        sf = self.model.predict_survival_function(X, times=times)
        return sf

    def explain(self):
        s = self.model.summary
        return {
            'coefficients': s['coef'].to_dict(),
            'hazard_ratio': s['exp(coef)'].to_dict(),
            'ci_lower': s['exp(coef) lower 95%'].to_dict(),
            'ci_upper': s['exp(coef) upper 95%'].to_dict(),
            'p_values': s['p'].to_dict()
        }