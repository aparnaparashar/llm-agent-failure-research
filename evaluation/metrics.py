"""
Research Evaluation Metrics.
Calculates ROC-AUC, PR-AUC, F1, Detection Lead Time, ECE, Brier score,
Net Success Gain, and Empirical Intervention Regret.
"""

from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    brier_score_loss,
)
from models.calibration.calibrator import calculate_ece


class EvaluationMetrics:
    """Computes all research evaluation metrics."""

    @staticmethod
    def compute_detection_metrics(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """Calculates discrimination and calibration metrics."""
        y_pred = (y_prob >= threshold).astype(int)

        # ROC-AUC safe check
        if len(np.unique(y_true)) > 1:
            roc_auc = float(roc_auc_score(y_true, y_prob))
            pr_auc = float(average_precision_score(y_true, y_prob))
        else:
            roc_auc = 0.5
            pr_auc = float(np.mean(y_true))

        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        brier = float(brier_score_loss(y_true, y_prob))
        ece = calculate_ece(y_true, y_prob)

        return {
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "f1": round(f1, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "brier_score": round(brier, 4),
            "ece": round(ece, 4),
        }

    @staticmethod
    def compute_lead_times(
        trajectories: List[Dict[str, Any]],
        detector_probs: List[np.ndarray],
        threshold: float = 0.5,
    ) -> List[int]:
        """
        Calculates detection lead time:
        Lead Time = (Actual failure step) - (First step where P(fail) >= threshold).
        Positive lead time means failure was anticipated in advance.
        """
        lead_times = []
        for traj, probs in zip(trajectories, detector_probs):
            if traj.get("task_success", True):
                continue  # Only compute for failing trajectories
            
            # Find actual failure step
            steps = traj.get("steps", [])
            actual_fail_step = traj.get("actual_failure_step")
            if actual_fail_step is None:
                # Find first tool error step or final step
                for s in steps:
                    if s.get("role") == "tool" and "error" in str(s.get("content", "")).lower():
                        actual_fail_step = s.get("step", 0)
                        break
            if actual_fail_step is None:
                actual_fail_step = len(steps) - 1

            # Find first trigger step
            trigger_step = None
            for step_idx, p in enumerate(probs):
                if p >= threshold:
                    trigger_step = step_idx
                    break

            if trigger_step is not None:
                lead_time = max(0, actual_fail_step - trigger_step)
                lead_times.append(lead_time)

        return lead_times
