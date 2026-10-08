import pytest
import numpy as np
from evaluation.metrics import EvaluationMetrics


def test_detection_metrics_calculation():
    y_true = np.array([0, 0, 0, 1, 1, 1], dtype=int)
    y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9], dtype=float)

    metrics = EvaluationMetrics.compute_detection_metrics(y_true, y_prob, threshold=0.5)

    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["brier_score"] < 0.1
    assert 0.0 <= metrics["ece"] <= 1.0


def test_lead_times_calculation():
    trajectories = [
        {
            "task_success": False,
            "actual_failure_step": 5,
            "steps": [{"step": i} for i in range(6)],
        }
    ]
    # Predicts failure at step 2
    probs = [np.array([0.1, 0.2, 0.8, 0.9, 0.9, 0.9])]

    lead_times = EvaluationMetrics.compute_lead_times(trajectories, probs, threshold=0.5)
    assert len(lead_times) == 1
    # 5 - 2 = 3 steps lead time!
    assert lead_times[0] == 3
