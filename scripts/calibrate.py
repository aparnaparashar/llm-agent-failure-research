#!/usr/bin/env python3
"""
CLI Script: calibrate.py
Performs probability calibration (Platt scaling & Isotonic regression)
on the primary XGBoost failure detector using validation splits.
Evaluates Expected Calibration Error (ECE) and Brier Score on test split.
Persists calibrated models to models/saved/.
"""

import os
import sys
import json
import logging
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from features.extractor import FEATURE_NAMES
from models.failure_detector.classifier import FailureDetector
from models.calibration.calibrator import RiskCalibrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("calibrate")


def main():
    train_split = os.path.join(PROJECT_ROOT, "data", "splits", "train", "train_features.csv")
    val_split = os.path.join(PROJECT_ROOT, "data", "splits", "validation", "val_features.csv")
    test_split = os.path.join(PROJECT_ROOT, "data", "splits", "test", "test_features.csv")

    train_df = pd.read_csv(train_split)
    val_df = pd.read_csv(val_split)
    test_df = pd.read_csv(test_split)

    available_features = [col for col in FEATURE_NAMES if col in train_df.columns]
    target_col = "imminent_failure_H2"

    X_train = train_df[available_features].values.astype(np.float32)
    y_train = train_df[target_col].values.astype(int)

    X_val = val_df[available_features].values.astype(np.float32)
    y_val = val_df[target_col].values.astype(int)

    X_test = test_df[available_features].values.astype(np.float32)
    y_test = test_df[target_col].values.astype(int)

    # Load or fit primary XGBoost detector
    detector_path = os.path.join(PROJECT_ROOT, "models", "saved", "xgboost_classifier.joblib")
    detector = FailureDetector("xgboost")
    if os.path.exists(detector_path):
        logger.info(f"Loading pre-trained XGBoost detector from {detector_path}")
        detector.load(detector_path)
    else:
        logger.info("Fitting XGBoost detector on training split...")
        detector.fit(X_train, y_train)

    # Uncalibrated baseline on test set
    uncal_probs = detector.predict_proba(X_test)
    raw_calibrator = RiskCalibrator()
    m_uncal = raw_calibrator.evaluate_calibration(y_test, uncal_probs)

    # 1. Platt Scaling (Sigmoid)
    calibrator_platt = RiskCalibrator(method="sigmoid")
    calibrator_platt.fit(detector, X_val, y_val)
    cal_platt_probs = calibrator_platt.predict_proba(X_test)
    m_platt = calibrator_platt.evaluate_calibration(y_test, cal_platt_probs)

    # 2. Isotonic Regression
    calibrator_iso = RiskCalibrator(method="isotonic")
    calibrator_iso.fit(detector, X_val, y_val)
    cal_iso_probs = calibrator_iso.predict_proba(X_test)
    m_iso = calibrator_iso.evaluate_calibration(y_test, cal_iso_probs)

    # Save fitted calibrators
    save_dir = os.path.join(PROJECT_ROOT, "models", "saved")
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(calibrator_iso, os.path.join(save_dir, "risk_calibrator_isotonic.joblib"))
    joblib.dump(calibrator_platt, os.path.join(save_dir, "risk_calibrator_sigmoid.joblib"))

    results = {
        "uncalibrated": m_uncal,
        "platt_scaling": m_platt,
        "isotonic_regression": m_iso,
    }

    print("\n" + "=" * 70)
    print("PROBABILITY CALIBRATION RESULTS (ON TEST SPLIT)")
    print("=" * 70)
    print(f"  Uncalibrated:         Brier={m_uncal['brier_score']:.4f}, ECE={m_uncal['ece']:.4f}")
    print(f"  Platt Scaling:        Brier={m_platt['brier_score']:.4f}, ECE={m_platt['ece']:.4f}")
    print(f"  Isotonic Regression:  Brier={m_iso['brier_score']:.4f}, ECE={m_iso['ece']:.4f}")
    print("=" * 70 + "\n")

    out_file = os.path.join(PROJECT_ROOT, "results", "tables", "calibration_results.json")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved calibration results to {out_file}")


if __name__ == "__main__":
    main()
