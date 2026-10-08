"""
Baseline Failure Predictors.
Heuristic rule baselines, random baselines, and majority class baselines.
"""

import numpy as np
from typing import Dict, Any


class ConsecutiveErrorBaseline:
    """Predicts failure if consecutive tool errors >= threshold."""

    def __init__(self, error_threshold: int = 1):
        self.error_threshold = error_threshold

    def predict_proba(self, X: np.ndarray, feature_names: list) -> np.ndarray:
        idx = feature_names.index("consecutive_tool_errors")
        consecutive_errors = X[:, idx]
        # Return pseudo-probability based on error streak
        return (consecutive_errors >= self.error_threshold).astype(float)


class LatencySpikeBaseline:
    """Predicts failure if step latency exceeds threshold."""

    def __init__(self, latency_threshold: float = 15.0):
        self.latency_threshold = latency_threshold

    def predict_proba(self, X: np.ndarray, feature_names: list) -> np.ndarray:
        idx = feature_names.index("step_latency")
        latencies = X[:, idx]
        return (latencies >= self.latency_threshold).astype(float)


class RandomBaseline:
    """Random guess baseline."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.rng.uniform(0.0, 1.0, size=len(X))
