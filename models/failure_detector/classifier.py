"""
Failure Detector Classifiers.
Lightweight ML models predicting failure risk from online telemetry.
P(Failure_future | Telemetry_<=t)
"""

import numpy as np
from typing import Dict, Any, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class FailureDetector:
    """Configurable failure detector model wrapper."""

    def __init__(self, model_type: str = "random_forest", **kwargs):
        self.model_type = model_type
        self.kwargs = kwargs
        self.model = self._create_model()
        self.is_fitted = False

    def _create_model(self):
        if self.model_type == "logistic_regression":
            return Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=42, **self.kwargs)),
            ])
        elif self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42,
                **self.kwargs,
            )
        elif self.model_type == "gradient_boosting":
            return HistGradientBoostingClassifier(
                max_iter=100,
                max_depth=5,
                random_state=42,
                **self.kwargs,
            )
        elif self.model_type == "xgboost":
            import xgboost as xgb
            return xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.08,
                random_state=42,
                eval_metric="logloss",
                **self.kwargs,
            )
        elif self.model_type == "mlp":
            return Pipeline([
                ("scaler", StandardScaler()),
                ("clf", MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42, **self.kwargs)),
            ])
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit model on feature matrix X and binary label vector y."""
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability of failure P(y=1 | x)."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        probs = self.model.predict_proba(X)
        if probs.shape[1] == 2:
            return probs[:, 1]
        elif probs.shape[1] == 1:
            # Single class edge case
            return probs[:, 0]
        return probs[:, -1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary decision using specified threshold."""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def save(self, filepath: str) -> str:
        """Persist fitted detector model to disk."""
        import joblib
        import os
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump({"model": self.model, "model_type": self.model_type, "is_fitted": self.is_fitted}, filepath)
        return filepath

    def load(self, filepath: str):
        """Load persisted detector model from disk."""
        import joblib
        data = joblib.load(filepath)
        self.model = data["model"]
        self.model_type = data["model_type"]
        self.is_fitted = data["is_fitted"]
        return self


class FailureDetectorFactory:
    """Factory helper for creating FailureDetector instances."""

    @staticmethod
    def create(model_type: str = "xgboost", **kwargs) -> FailureDetector:
        return FailureDetector(model_type=model_type, **kwargs)

