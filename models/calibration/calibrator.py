"""
Probability Calibration and Verification.
Implements Platt scaling (Sigmoid) and Isotonic Regression.
Calculates Expected Calibration Error (ECE) and Brier Score.
"""

import numpy as np
from typing import Tuple, Dict, Any
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss


def calculate_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Computes Expected Calibration Error (ECE).
    ECE = sum_{b=1}^B (|B_b| / N) * |acc(B_b) - conf(B_b)|
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    N = len(y_true)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper if i < n_bins - 1 else y_prob <= bin_upper)
        bin_size = np.sum(in_bin)

        if bin_size > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(y_prob[in_bin])
            ece += (bin_size / N) * np.abs(bin_acc - bin_conf)

    return float(ece)


from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class RiskCalibrator:
    """Calibrates predicted risk probabilities to reflect true likelihood."""

    def __init__(self, method: str = "sigmoid"):
        self.method = method  # "sigmoid" (Platt) or "isotonic"
        self.calibrator = None
        self.prob_calibrator = None
        self.is_fitted = False

    def fit(self, *args):
        """
        Flexible fit method:
        - fit(base_detector, X_val, y_val)
        - fit(raw_probs, y_val)
        """
        if len(args) == 3:
            base_detector, X_val, y_val = args
            estimator = getattr(base_detector, "model", base_detector)
            self.calibrator = CalibratedClassifierCV(
                estimator=estimator,
                method=self.method,
                cv="prefit",
            )
            self.calibrator.fit(X_val, y_val)
            self.is_fitted = True
            return self
        elif len(args) == 2:
            raw_probs, y_val = args
            raw_probs = np.asarray(raw_probs).ravel()
            y_val = np.asarray(y_val).ravel()
            if self.method == "isotonic":
                self.prob_calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                self.prob_calibrator.fit(raw_probs, y_val)
            else:
                self.prob_calibrator = LogisticRegression(max_iter=1000, random_state=42)
                # Reshape for logistic regression
                self.prob_calibrator.fit(raw_probs.reshape(-1, 1), y_val)
            self.is_fitted = True
            return self
        else:
            raise ValueError(f"Expected 2 or 3 arguments, got {len(args)}")

    def calibrate(self, raw_probs: np.ndarray) -> np.ndarray:
        """Transforms raw probabilities into calibrated probabilities."""
        raw_probs = np.asarray(raw_probs).ravel()
        if not self.is_fitted:
            return raw_probs

        if self.prob_calibrator is not None:
            if self.method == "isotonic":
                return np.clip(self.prob_calibrator.predict(raw_probs), 0.0, 1.0)
            else:
                p = self.prob_calibrator.predict_proba(raw_probs.reshape(-1, 1))
                return p[:, 1] if p.shape[1] == 2 else p[:, 0]
        elif self.calibrator is not None:
            # If fitted with CalibratedClassifierCV, raw probs cannot be passed directly
            return raw_probs
        return raw_probs

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Calibrator is not fitted.")
        X_arr = np.asarray(X)
        if X_arr.ndim == 1 or (X_arr.ndim == 2 and X_arr.shape[1] == 1):
            return self.calibrate(X_arr)
        if self.calibrator is not None:
            probs = self.calibrator.predict_proba(X_arr)
            return probs[:, 1] if probs.shape[1] == 2 else probs[:, 0]
        return self.calibrate(X_arr)

    def evaluate_calibration(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
        """Calculates Brier score and ECE."""
        brier = float(brier_score_loss(y_true, y_prob))
        ece = calculate_ece(y_true, y_prob)
        return {
            "brier_score": round(brier, 4),
            "ece": round(ece, 4),
        }

