#!/usr/bin/env python3
"""
CLI Script: train_policy.py
Configures and evaluates prevention policies (Passive, Static, Calibrated, Adaptive Regret)
using real model predictions from the trained XGBoost detector and risk calibrator on the test split.
Outputs policy summary and regret metrics without synthetic random placeholders.
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
from policy.policies import (
    PassivePolicy,
    StaticThresholdPolicy,
    CalibratedThresholdPolicy,
    AdaptiveRegretPolicy,
    RegretCalculator,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_policy")


def main():
    logger.info("Initializing adaptive policy evaluation on real test data...")

    test_split = os.path.join(PROJECT_ROOT, "data", "splits", "test", "test_features.csv")
    if not os.path.exists(test_split):
        logger.error(f"Test split not found at {test_split}")
        sys.exit(1)

    test_df = pd.read_csv(test_split)
    available_features = [col for col in FEATURE_NAMES if col in test_df.columns]
    target_col = "imminent_failure_H2"

    X_test = test_df[available_features].values.astype(np.float32)
    y_test = test_df[target_col].values.astype(int)

    # Load trained detector
    detector_path = os.path.join(PROJECT_ROOT, "models", "saved", "xgboost_classifier.joblib")
    detector = FailureDetector("xgboost")
    if os.path.exists(detector_path):
        detector.load(detector_path)
    else:
        train_split = os.path.join(PROJECT_ROOT, "data", "splits", "train", "train_features.csv")
        train_df = pd.read_csv(train_split)
        detector.fit(train_df[available_features].values.astype(np.float32), train_df[target_col].values.astype(int))

    raw_risks = detector.predict_proba(X_test)

    # Load or fit calibrator
    calibrator_path = os.path.join(PROJECT_ROOT, "models", "saved", "risk_calibrator_isotonic.joblib")
    if os.path.exists(calibrator_path):
        calibrator = joblib.load(calibrator_path)
        calibrated_risks = calibrator.predict_proba(X_test)
    else:
        calibrator = RiskCalibrator(method="isotonic")
        val_split = os.path.join(PROJECT_ROOT, "data", "splits", "validation", "val_features.csv")
        val_df = pd.read_csv(val_split)
        calibrator.fit(detector, val_df[available_features].values.astype(np.float32), val_df[target_col].values.astype(int))
        calibrated_risks = calibrator.predict_proba(X_test)

    policies = {
        "Policy 0 (Passive)": PassivePolicy(),
        "Policy 1 (Static Threshold)": StaticThresholdPolicy(threshold=0.5),
        "Policy 2 (Calibrated)": CalibratedThresholdPolicy(threshold=0.45),
        "Policy 3 (Adaptive Regret)": AdaptiveRegretPolicy(cost_fp=1.0, cost_fn=5.0),
    }

    regret_calc = RegretCalculator(cost_fp=1.0, cost_fn=5.0)
    summary = {}
    table_rows = []

    print("\n" + "=" * 70)
    print("RUNTIME INTERVENTION POLICY EVALUATION (REAL TEST SPLIT)")
    print("=" * 70)

    for name, pol in policies.items():
        total_interventions = 0
        false_alarms = 0
        total_regret = 0.0

        for r_raw, r_cal, y_true in zip(raw_risks, calibrated_risks, y_test):
            dec = pol.decide_intervention(step=1, predicted_risk=float(r_raw), calibrated_risk=float(r_cal))
            intervened = dec["should_intervene"]
            if intervened:
                total_interventions += 1
                if y_true == 0:
                    false_alarms += 1

            reg = regret_calc.calculate_regret(
                intervened=intervened,
                factual_failed=(y_true == 1),
                counterfactual_succeeded=True,
            )["regret"]
            total_regret += reg

        mean_regret = total_regret / max(len(y_test), 1)
        intervention_rate = total_interventions / max(len(y_test), 1)
        success_rate = 1.0 - (mean_regret / 5.0)

        summary[name] = {
            "mean_regret": round(mean_regret, 4),
            "regret": round(mean_regret, 4),
            "total_interventions": total_interventions,
            "interventions": total_interventions,
            "false_alarms": false_alarms,
            "intervention_rate": round(intervention_rate, 4),
            "success_rate": round(success_rate, 4),
        }

        table_rows.append({
            "Policy": name,
            "Interventions": total_interventions,
            "False Alarms": false_alarms,
            "Intervention Rate": f"{intervention_rate:.1%}",
            "Mean Regret": f"{mean_regret:.3f}",
        })
        print(f"  {name:30s}: Regret={mean_regret:.3f}, Interventions={total_interventions}, False Alarms={false_alarms}")

    out_file = os.path.join(PROJECT_ROOT, "results", "tables", "policy_summary.json")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Save CSV and MD tables
    csv_file = os.path.join(PROJECT_ROOT, "results", "tables", "policy_evaluation_table.csv")
    md_file = os.path.join(PROJECT_ROOT, "results", "tables", "policy_evaluation_table.md")
    pol_df = pd.DataFrame(table_rows)
    pol_df.to_csv(csv_file, index=False)
    with open(md_file, "w", encoding="utf-8") as f:
        f.write("# Runtime Prevention Policy Evaluation Table\n\n")
        f.write(pol_df.to_markdown(index=False))
        f.write("\n")

    logger.info(f"Saved policy evaluation to {out_file}, {csv_file}, and {md_file}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
