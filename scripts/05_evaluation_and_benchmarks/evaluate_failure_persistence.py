#!/usr/bin/env python3
"""
Runtime Failure Persistence and Degradation Cascading Analysis
(Formerly evaluated as early detection lead-time).

Analyzes:
1. Cold-start vs. multi-step failure breakdown (Step-1 failures where prior telemetry is absent).
2. Failure persistence: detection of runaway cascading loops post initial failure.
3. Advance lead-time warning on multi-step dynamic reasoning degradation.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd

def _resolve_project_root():
    curr = os.path.abspath(os.path.dirname(__file__))
    while curr and os.path.dirname(curr) != curr:
        if os.path.exists(os.path.join(curr, 'configs')) and os.path.exists(os.path.join(curr, 'features')):
            return curr
        curr = os.path.dirname(curr)
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

PROJECT_ROOT = _resolve_project_root()
sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_failure_persistence")


def load_trained_model():
    """Load the trained XGBoost model and calibrator on trajectory-disjoint splits."""
    from models.failure_detector.classifier import FailureDetectorFactory
    from models.calibration.calibrator import RiskCalibrator

    feature_cols = [
        "step", "step_ratio", "step_latency", "cum_latency",
        "tool_calls_count", "tool_error_count", "tool_error_rate",
        "consecutive_tool_errors", "repeat_tool_ratio",
        "latest_message_len", "avg_message_len",
        "token_expansion_ratio", "observation_error_flag",
        "lexical_diversity",
    ]
    target = "imminent_failure_H2"

    train_path = os.path.join(PROJECT_ROOT, "data", "splits", "train", "train_features.csv")
    test_path = os.path.join(PROJECT_ROOT, "data", "splits", "test", "test_features.csv")
    val_path = os.path.join(PROJECT_ROOT, "data", "splits", "validation", "val_features.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    val_df = pd.read_csv(val_path)

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df[target].values.astype(int)
    X_val = val_df[feature_cols].values.astype(np.float32)
    y_val = val_df[target].values.astype(int)

    factory = FailureDetectorFactory()
    model = factory.create("xgboost")
    model.fit(X_train, y_train)

    calibrator = RiskCalibrator(method="isotonic")
    raw_val_probs = model.predict_proba(X_val)
    calibrator.fit(raw_val_probs, y_val)

    return model, calibrator, test_df, feature_cols, target


def compute_persistence_and_cascading_metrics(test_df: pd.DataFrame, model, calibrator, feature_cols: list, target: str, threshold: float = 0.5) -> list:
    """Evaluate trajectory-level alert timing, separating step-1 cold starts from dynamic cascades."""
    results = []

    for traj_id, traj_group in test_df.groupby("trajectory_id"):
        traj_group = traj_group.sort_values("step")

        X_traj = traj_group[feature_cols].values.astype(np.float32)
        y_traj = traj_group[target].values
        steps = traj_group["step"].values
        mode = traj_group["execution_mode"].iloc[0] if "execution_mode" in traj_group.columns else "unknown"

        raw_probs = model.predict_proba(X_traj)
        cal_probs = calibrator.predict(raw_probs)[:, 1] if hasattr(calibrator, "predict") else raw_probs[:, 1]

        failing_mask = (y_traj == 1)
        has_failure = bool(np.any(failing_mask))
        actual_failure_step = int(steps[np.where(failing_mask)[0][0]]) if has_failure else None

        detection_mask = (cal_probs >= threshold)
        detected = bool(np.any(detection_mask))
        first_detection_step = int(steps[np.where(detection_mask)[0][0]]) if detected else None

        lead_time = None
        timing_category = "clean_unflagged"

        if has_failure:
            if detected:
                lead_time = actual_failure_step - first_detection_step
                if lead_time > 0:
                    timing_category = "strict_advance_warning"
                elif lead_time == 0:
                    timing_category = "coincident_at_failure"
                else:
                    timing_category = "post_failure_persistence"
            else:
                timing_category = "missed_failure"
        else:
            if detected:
                timing_category = "false_alarm"

        results.append({
            "trajectory_id": traj_id,
            "execution_mode": mode,
            "has_failure": has_failure,
            "actual_failure_step": actual_failure_step,
            "detected": detected,
            "first_detection_step": first_detection_step,
            "lead_time": lead_time,
            "timing_category": timing_category,
            "is_step_1_failure": bool(actual_failure_step == 1) if actual_failure_step is not None else False,
            "peak_risk": float(np.max(cal_probs)),
        })

    return results


def summarize_persistence_analysis(results: list) -> dict:
    df = pd.DataFrame(results)
    failing = df[df["has_failure"]]
    healthy = df[~df["has_failure"]]

    n_failing = len(failing)
    n_healthy = len(healthy)

    step_1_failures = failing[failing["is_step_1_failure"]]
    dynamic_failures = failing[~failing["is_step_1_failure"]]

    category_counts = df["timing_category"].value_counts().to_dict()

    lead_times_non_negative = [r["lead_time"] for r in results if r["lead_time"] is not None and r["lead_time"] >= 0]
    lead_times_all_detected = [r["lead_time"] for r in results if r["lead_time"] is not None]

    return {
        "total_trajectories": len(df),
        "failing_trajectories": n_failing,
        "healthy_trajectories": n_healthy,
        "step_1_cold_start_failures": len(step_1_failures),
        "step_1_cold_start_percentage": float(len(step_1_failures) / max(1, n_failing) * 100),
        "dynamic_multistep_failures": len(dynamic_failures),
        "timing_categories": {k: int(v) for k, v in category_counts.items()},
        "overall_detection_rate": float(failing["detected"].sum() / max(1, n_failing)),
        "flagged_at_or_before_failure_count": int(category_counts.get("strict_advance_warning", 0) + category_counts.get("coincident_at_failure", 0)),
        "flagged_at_or_before_failure_rate": float((category_counts.get("strict_advance_warning", 0) + category_counts.get("coincident_at_failure", 0)) / max(1, n_failing)),
        "advance_warning_stats_non_negative": {
            "mean": float(np.mean(lead_times_non_negative)) if lead_times_non_negative else 0.0,
            "median": float(np.median(lead_times_non_negative)) if lead_times_non_negative else 0.0,
            "max": float(np.max(lead_times_non_negative)) if lead_times_non_negative else 0.0,
        },
        "all_detected_lead_time_stats": {
            "mean": float(np.mean(lead_times_all_detected)) if lead_times_all_detected else 0.0,
            "median": float(np.median(lead_times_all_detected)) if lead_times_all_detected else 0.0,
        }
    }


def main():
    logger.info("Running Runtime Failure Persistence and Degradation Cascading Analysis...")
    model, calibrator, test_df, feature_cols, target = load_trained_model()
    results = compute_persistence_and_cascading_metrics(test_df, model, calibrator, feature_cols, target, threshold=0.5)
    summary = summarize_persistence_analysis(results)

    out_dir = os.path.join(PROJECT_ROOT, "results", "tables")
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, "failure_persistence_results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    csv_path = os.path.join(out_dir, "failure_persistence_per_trajectory.csv")
    pd.DataFrame(results).to_csv(csv_path, index=False)

    logger.info(f"Analysis saved to {json_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
