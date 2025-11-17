import abc
import json
from typing import Any, Dict

class BaseForecaster(abc.ABC):
    @abc.abstractmethod
    def fit(self, X, y=None):
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, X):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError

    def get_params(self) -> Dict[str, Any]:
        return {}

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        return self

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.get_params(), f, indent=2)

    def load(self, path: str):
        with open(path, 'r') as f:
            params = json.load(f)
        self.set_params(**params)
        return self