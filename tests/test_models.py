import pytest
import numpy as np
from models.failure_detector.classifier import FailureDetector
from models.baselines.heuristic import ConsecutiveErrorBaseline, RandomBaseline
from models.calibration.calibrator import RiskCalibrator, calculate_ece


@pytest.fixture
def synthetic_data():
    rng = np.random.RandomState(42)
    X = rng.randn(60, 14).astype(np.float32)
    # Binary label loosely correlated with first feature
    y = (X[:, 0] > 0.1).astype(int)
    return X, y


def test_failure_detector_types(synthetic_data):
    X, y = synthetic_data

    for m_type in ["random_forest", "logistic_regression", "gradient_boosting", "mlp"]:
        clf = FailureDetector(model_type=m_type)
        assert clf.is_fitted is False
        clf.fit(X, y)
        assert clf.is_fitted is True

        probs = clf.predict_proba(X)
        assert isinstance(probs, np.ndarray)
        assert probs.shape == (len(X),)
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


def test_baseline_models(synthetic_data):
    X, _ = synthetic_data
    feature_names = [
        "step_ratio", "step_latency", "cum_latency", "tool_calls_count",
        "tool_error_count", "tool_error_rate", "consecutive_tool_errors",
        "repeat_tool_ratio", "latest_message_len", "avg_message_len",
        "token_expansion_ratio", "observation_error_flag",
        "lexical_diversity", "repetition_ngram_score"
    ]

    base_streak = ConsecutiveErrorBaseline(error_threshold=1)
    probs_streak = base_streak.predict_proba(X, feature_names)
    assert len(probs_streak) == len(X)
    assert np.all((probs_streak == 0.0) | (probs_streak == 1.0))

    base_rand = RandomBaseline(seed=42)
    probs_rand = base_rand.predict_proba(X)
    assert len(probs_rand) == len(X)
    assert np.all((probs_rand >= 0.0) & (probs_rand <= 1.0))


def test_risk_calibrator_and_ece(synthetic_data):
    X, y = synthetic_data
    X_train, y_train = X[:40], y[:40]
    X_val, y_val = X[40:], y[40:]

    # Base detector
    detector = FailureDetector(model_type="random_forest")
    detector.fit(X_train, y_train)

    # Calibrator
    calibrator = RiskCalibrator(method="sigmoid")
    calibrator.fit(detector, X_val, y_val)
    assert calibrator.is_fitted is True

    calibrated_probs = calibrator.predict_proba(X_val)
    assert len(calibrated_probs) == len(X_val)
    assert np.all((calibrated_probs >= 0.0) & (calibrated_probs <= 1.0))

    metrics = calibrator.evaluate_calibration(y_val, calibrated_probs)
    assert "brier_score" in metrics
    assert "ece" in metrics
    assert 0.0 <= metrics["ece"] <= 1.0
