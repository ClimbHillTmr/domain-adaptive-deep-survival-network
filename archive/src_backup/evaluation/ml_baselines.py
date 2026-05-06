"""
Traditional ML baselines for survival analysis comparison.
Implements Logistic Regression, Random Forest, XGBoost, and LightGBM.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import logging

log = logging.getLogger(__name__)


def prepare_features(static, dynamic, use_dynamic=True):
    """Combine static and dynamic features for traditional ML models."""
    if use_dynamic and dynamic is not None and dynamic.shape[1] > 0:
        dynamic_flat = dynamic.reshape(dynamic.shape[0], -1)
        X = np.hstack([static, dynamic_flat])
    else:
        X = static
    return X


def prepare_survival_labels(targets, time_horizon=60):
    """
    Convert survival data to binary classification for traditional ML.
    Event within time_horizon minutes = positive, else negative.
    """
    event = targets["event"].astype(bool)
    duration = targets["duration"].astype(float)
    
    label = (event & (duration <= time_horizon)).astype(int)
    return label


class TraditionalMLBaselines:
    """Train and evaluate traditional ML models for comparison."""
    
    def __init__(self, time_horizon=60):
        self.time_horizon = time_horizon
        self.models = {}
        self.scaler = StandardScaler()
        
    def _get_models(self):
        """Define model configurations."""
        return {
            "logistic_regression": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    max_iter=1000, 
                    C=1.0, 
                    solver="lbfgs",
                    random_state=42
                ))
            ]),
            "random_forest": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(
                    n_estimators=200,
                    max_depth=10,
                    min_samples_split=20,
                    min_samples_leaf=10,
                    random_state=42,
                    n_jobs=-1
                ))
            ]),
            "gradient_boosting": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", GradientBoostingClassifier(
                    n_estimators=200,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.8,
                    random_state=42
                ))
            ]),
        }
    
    def train_all(self, X_train, y_train, X_val=None, y_val=None):
        """Train all baseline models."""
        results = {}
        
        for name, model in self._get_models().items():
            log.info(f"Training {name}...")
            model.fit(X_train, y_train)
            self.models[name] = model
            
            if X_val is not None and y_val is not None:
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                results[name] = {"val_risk_scores": y_pred_proba}
                log.info(f"  {name} trained successfully")
            else:
                results[name] = {}
        
        return results
    
    def predict(self, X):
        """Get risk scores from all trained models."""
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict_proba(X)[:, 1]
        return predictions
    
    def get_feature_importance(self, model_name="random_forest"):
        """Extract feature importance from tree-based models."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")
        
        model = self.models[model_name]
        clf = model.named_steps["clf"]
        
        if hasattr(clf, "feature_importances_"):
            return clf.feature_importances_
        elif hasattr(clf, "coef_"):
            return np.abs(clf.coef_[0])
        else:
            return None
