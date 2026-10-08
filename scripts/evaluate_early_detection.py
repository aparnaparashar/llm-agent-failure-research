#!/usr/bin/env python3
"""
Early Detection Evaluation (PROMPT §30, §51).
Measures lead time, first detection step, false alarm rate,
and percentage of failures detected before the actual failure step.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_early_detection")


def load_trained_model():
    """Load the trained XGBoost model or retrain quickly."""
    from models.failure_detector.classifier import FailureDetectorFactory
    from models.calibration.calibrator import RiskCalibrator

    csv_path = os.path.join(PROJECT_ROOT, "data", "processed", "telemetry_dataset.csv")
    df = pd.read_csv(csv_path)

    feature_cols = [
        "step", "step_ratio", "step_latency", "cum_latency",
        "tool_calls_count", "tool_error_count", "tool_error_rate",
        "consecutive_tool_errors", "repeat_tool_ratio",
        "latest_message_len", "avg_message_len",
        "token_expansion_ratio", "observation_error_flag",
        "lexical_diversity",
    ]
    target = "imminent_failure_H2"

    X = df[feature_cols].values.astype(np.float32)
    y = df[target].values.astype(int)

    # Train/test by trajectory-disjoint split
    train_path = os.path.join(PROJECT_ROOT, "data", "splits", "train", "train_features.csv")
    test_path = os.path.join(PROJECT_ROOT, "data", "splits", "test", "test_features.csv")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df[target].values.astype(int)
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df[target].values.astype(int)

    factory = FailureDetectorFactory()
    model = factory.create("xgboost")
    model.fit(X_train, y_train)

    # Calibrate
    val_path = os.path.join(PROJECT_ROOT, "data", "splits", "validation", "val_features.csv")
    val_df = pd.read_csv(val_path)
    X_val = val_df[feature_cols].values.astype(np.float32)
    y_val = val_df[target].values.astype(int)

    calibrator = RiskCalibrator(method="isotonic")
    raw_val_probs = model.predict_proba(X_val)
    calibrator.fit(raw_val_probs, y_val)

    return model, calibrator, test_df, feature_cols, target


def compute_early_detection_metrics(
    test_df: pd.DataFrame,
    model,
    calibrator,
    feature_cols: list,
    target: str,
    threshold: float = 0.5,
) -> dict:
    """
    For each trajectory in the test set:
    1. Find the actual failure step (first step where imminent_failure_H2 = 1)
    2. Find the first detection step (first step where calibrated risk >= threshold)
    3. Compute lead_time = actual_failure_step - first_detection_step
    """
    results = []
    
    for traj_id, traj_group in test_df.groupby("trajectory_id"):
        traj_group = traj_group.sort_values("step")
        
        X_traj = traj_group[feature_cols].values.astype(np.float32)
        y_traj = traj_group[target].values
        raw_probs = model.predict_proba(X_traj)
        cal_probs = calibrator.calibrate(raw_probs)
        
        steps = traj_group["step"].values
        mode = traj_group["execution_mode"].values[0] if "execution_mode" in traj_group.columns else "unknown"
        task_success = traj_group["task_success"].values[0] if "task_success" in traj_group.columns else None
        
        # Find actual failure step (first step with label=1)
        failure_steps = np.where(y_traj == 1)[0]
        actual_failure_step = int(steps[failure_steps[0]]) if len(failure_steps) > 0 else None
        
        # Find first detection step (first step with calibrated risk >= threshold)
        detection_steps = np.where(cal_probs >= threshold)[0]
        first_detection_step = int(steps[detection_steps[0]]) if len(detection_steps) > 0 else None
        
        # Classify the trajectory
        has_failure = actual_failure_step is not None
        detected = first_detection_step is not None
        
        if has_failure and detected:
            lead_time = actual_failure_step - first_detection_step
            detected_before_failure = first_detection_step <= actual_failure_step
            case = "true_positive" if detected_before_failure else "late_detection"
        elif has_failure and not detected:
            lead_time = None
            detected_before_failure = False
            case = "false_negative"
        elif not has_failure and detected:
            lead_time = None
            detected_before_failure = False
            case = "false_alarm"
        else:
            lead_time = None
            detected_before_failure = False
            case = "true_negative"
        
        results.append({
            "trajectory_id": traj_id,
            "execution_mode": mode,
            "has_failure": has_failure,
            "actual_failure_step": actual_failure_step,
            "first_detection_step": first_detection_step,
            "detected": detected,
            "detected_before_failure": detected_before_failure,
            "lead_time": lead_time,
            "case": case,
            "max_risk": float(np.max(cal_probs)),
            "num_steps": len(steps),
            "task_success": task_success,
        })
    
    return results


def summarize_results(results: list[dict]) -> dict:
    """Aggregate early detection metrics."""
    df = pd.DataFrame(results)
    
    # Overall
    total = len(df)
    failure_trajs = df[df["has_failure"]]
    healthy_trajs = df[~df["has_failure"]]
    
    n_failures = len(failure_trajs)
    n_healthy = len(healthy_trajs)
    
    # Detection rates
    if n_failures > 0:
        detected_count = failure_trajs["detected"].sum()
        detected_before_count = failure_trajs["detected_before_failure"].sum()
        early_detection_rate = float(detected_before_count / n_failures)
        detection_rate = float(detected_count / n_failures)
    else:
        early_detection_rate = 0.0
        detection_rate = 0.0
    
    # False alarm rate
    if n_healthy > 0:
        false_alarms = healthy_trajs["detected"].sum()
        false_alarm_rate = float(false_alarms / n_healthy)
    else:
        false_alarm_rate = 0.0
    
    # Lead time statistics
    lead_times = [r["lead_time"] for r in results if r["lead_time"] is not None and r["lead_time"] >= 0]
    if lead_times:
        lt_arr = np.array(lead_times)
        lead_time_stats = {
            "mean": float(np.mean(lt_arr)),
            "median": float(np.median(lt_arr)),
            "std": float(np.std(lt_arr)),
            "iqr": float(np.percentile(lt_arr, 75) - np.percentile(lt_arr, 25)),
            "min": float(np.min(lt_arr)),
            "max": float(np.max(lt_arr)),
        }
    else:
        lead_time_stats = {"mean": 0, "median": 0, "std": 0, "iqr": 0, "min": 0, "max": 0}
    
    # Breakdown by execution mode
    mode_breakdown = {}
    for mode in df["execution_mode"].unique():
        mode_df = df[df["execution_mode"] == mode]
        mode_failures = mode_df[mode_df["has_failure"]]
        mode_healthy = mode_df[~mode_df["has_failure"]]
        
        if len(mode_failures) > 0:
            mode_det_rate = float(mode_failures["detected_before_failure"].sum() / len(mode_failures))
        else:
            mode_det_rate = 0.0
        
        mode_lead = [r["lead_time"] for r in results 
                     if r["execution_mode"] == mode and r["lead_time"] is not None and r["lead_time"] >= 0]
        
        mode_breakdown[mode] = {
            "total_trajectories": len(mode_df),
            "failure_trajectories": len(mode_failures),
            "early_detection_rate": mode_det_rate,
            "mean_lead_time": float(np.mean(mode_lead)) if mode_lead else 0.0,
            "false_alarm_rate": float(mode_healthy["detected"].sum() / len(mode_healthy)) if len(mode_healthy) > 0 else 0.0,
        }
    
    # Case distribution
    case_counts = df["case"].value_counts().to_dict()
    
    return {
        "total_trajectories": total,
        "failure_trajectories": n_failures,
        "healthy_trajectories": n_healthy,
        "detection_rate": detection_rate,
        "early_detection_rate": early_detection_rate,
        "false_alarm_rate": false_alarm_rate,
        "lead_time": lead_time_stats,
        "case_distribution": {k: int(v) for k, v in case_counts.items()},
        "mode_breakdown": mode_breakdown,
    }


def main():
    logger.info("Loading trained XGBoost model and calibrator...")
    model, calibrator, test_df, feature_cols, target = load_trained_model()
    
    logger.info(f"Test set: {len(test_df)} step-level rows, {test_df['trajectory_id'].nunique()} trajectories")
    
    # Evaluate at multiple thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    all_summaries = {}
    
    for thresh in thresholds:
        logger.info(f"Evaluating at threshold={thresh}...")
        results = compute_early_detection_metrics(test_df, model, calibrator, feature_cols, target, threshold=thresh)
        summary = summarize_results(results)
        all_summaries[str(thresh)] = summary
    
    # Use threshold=0.5 as primary
    primary = all_summaries["0.5"]
    primary_results = compute_early_detection_metrics(test_df, model, calibrator, feature_cols, target, threshold=0.5)
    
    # Print report
    print("\n" + "=" * 70)
    print("EARLY DETECTION EVALUATION REPORT")
    print("=" * 70)
    
    print(f"\n  Test Trajectories:        {primary['total_trajectories']}")
    print(f"  Failure Trajectories:     {primary['failure_trajectories']}")
    print(f"  Healthy Trajectories:     {primary['healthy_trajectories']}")
    print(f"\n  Detection Rate:           {primary['detection_rate']:.3f}")
    print(f"  Early Detection Rate:     {primary['early_detection_rate']:.3f}")
    print(f"  False Alarm Rate:         {primary['false_alarm_rate']:.3f}")
    
    lt = primary["lead_time"]
    print(f"\n  Lead Time (steps before failure):")
    print(f"    Mean:   {lt['mean']:.2f}")
    print(f"    Median: {lt['median']:.2f}")
    print(f"    Std:    {lt['std']:.2f}")
    print(f"    IQR:    {lt['iqr']:.2f}")
    print(f"    Range:  [{lt['min']:.0f}, {lt['max']:.0f}]")
    
    print(f"\n  Case Distribution:")
    for case, count in primary["case_distribution"].items():
        print(f"    {case}: {count}")
    
    print(f"\n  Mode Breakdown:")
    for mode, stats in primary["mode_breakdown"].items():
        print(f"    {mode}: detection={stats['early_detection_rate']:.3f}, "
              f"lead_time={stats['mean_lead_time']:.2f}, "
              f"false_alarm={stats['false_alarm_rate']:.3f}")
    
    print(f"\n  Threshold Sensitivity:")
    for thresh, s in all_summaries.items():
        print(f"    tau={thresh}: detection={s['early_detection_rate']:.3f}, "
              f"false_alarm={s['false_alarm_rate']:.3f}, "
              f"lead_time_mean={s['lead_time']['mean']:.2f}")
    
    print("=" * 70 + "\n")
    
    # Save
    output_path = os.path.join(PROJECT_ROOT, "results", "tables", "early_detection_results.json")
    with open(output_path, "w") as f:
        json.dump({
            "primary_threshold": 0.5,
            "primary_results": primary,
            "threshold_sensitivity": all_summaries,
        }, f, indent=2)
    logger.info(f"Results saved to {output_path}")
    
    # Save per-trajectory results CSV
    csv_path = os.path.join(PROJECT_ROOT, "results", "tables", "early_detection_per_trajectory.csv")
    pd.DataFrame(primary_results).to_csv(csv_path, index=False)
    logger.info(f"Per-trajectory CSV saved to {csv_path}")


if __name__ == "__main__":
    main()
