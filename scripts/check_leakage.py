#!/usr/bin/env python3
"""
Automated Leakage Checker (PROMPT §17, §47).
Scans the feature matrix for prohibited future-information columns.
Verifies that no feature at step t uses information from steps > t.
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("check_leakage")

# Columns that must NEVER appear as prediction features
PROHIBITED_COLUMNS = {
    # Future failure information
    "future_failure", "future_tool_calls", "future_errors",
    "final_outcome", "actual_failure_step", "injection_step",
    "future_telemetry", "future_trajectory_length",
    "future_token_count", "future_recovery_status",
    # Target labels (allowed in dataset but NOT in feature matrix)
    "imminent_failure_H1", "imminent_failure_H2", "imminent_failure_H3",
    "task_failed_end", "task_success",
    # Metadata (allowed in dataset but NOT in feature matrix)
    "trajectory_id", "task_id", "execution_mode",
}

# Columns that are legitimate features
ALLOWED_FEATURE_COLUMNS = {
    "step", "step_ratio", "step_latency", "cum_latency",
    "tool_calls_count", "tool_error_count", "tool_error_rate",
    "consecutive_tool_errors", "repeat_tool_ratio",
    "latest_message_len", "avg_message_len",
    "token_expansion_ratio", "observation_error_flag",
    "lexical_diversity", "repetition_ngram_score",
}


def check_column_leakage(df: pd.DataFrame) -> list[dict]:
    """Check if any prohibited columns are in the feature matrix."""
    violations = []
    feature_cols = set(df.columns) - {"trajectory_id", "task_id", "execution_mode",
                                       "imminent_failure_H1", "imminent_failure_H2",
                                       "imminent_failure_H3", "task_failed_end", "task_success"}
    
    for col in feature_cols:
        if col in PROHIBITED_COLUMNS:
            violations.append({
                "type": "prohibited_column",
                "column": col,
                "severity": "CRITICAL",
                "message": f"Prohibited future-information column '{col}' found in feature matrix"
            })
    
    return violations


def check_trajectory_split_leakage(splits_dir: str) -> list[dict]:
    """Check that no trajectory appears in multiple splits."""
    violations = []
    
    train_path = os.path.join(splits_dir, "train", "train_features.csv")
    val_path = os.path.join(splits_dir, "validation", "val_features.csv")
    test_path = os.path.join(splits_dir, "test", "test_features.csv")
    
    if not all(os.path.exists(p) for p in [train_path, val_path, test_path]):
        violations.append({
            "type": "missing_splits",
            "severity": "CRITICAL",
            "message": "One or more split files are missing"
        })
        return violations
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    train_trajs = set(train_df["trajectory_id"].unique())
    val_trajs = set(val_df["trajectory_id"].unique())
    test_trajs = set(test_df["trajectory_id"].unique())
    
    train_val_overlap = train_trajs & val_trajs
    train_test_overlap = train_trajs & test_trajs
    val_test_overlap = val_trajs & test_trajs
    
    if train_val_overlap:
        violations.append({
            "type": "trajectory_overlap",
            "splits": "train-validation",
            "severity": "CRITICAL",
            "count": len(train_val_overlap),
            "message": f"{len(train_val_overlap)} trajectories appear in both train and validation"
        })
    
    if train_test_overlap:
        violations.append({
            "type": "trajectory_overlap",
            "splits": "train-test",
            "severity": "CRITICAL",
            "count": len(train_test_overlap),
            "message": f"{len(train_test_overlap)} trajectories appear in both train and test"
        })
    
    if val_test_overlap:
        violations.append({
            "type": "trajectory_overlap",
            "splits": "validation-test",
            "severity": "CRITICAL",
            "count": len(val_test_overlap),
            "message": f"{len(val_test_overlap)} trajectories appear in both validation and test"
        })
    
    return violations


def check_causal_feature_integrity(df: pd.DataFrame) -> list[dict]:
    """
    Verify that features at step t don't correlate impossibly with future targets.
    A feature that perfectly predicts the target with r > 0.95 is suspicious.
    """
    violations = []
    target = "imminent_failure_H2"
    
    if target not in df.columns:
        return violations
    
    feature_cols = [c for c in df.columns if c in ALLOWED_FEATURE_COLUMNS]
    
    for col in feature_cols:
        if df[col].std() == 0:
            continue
        corr = abs(df[col].corr(df[target]))
        if corr > 0.95:
            violations.append({
                "type": "suspicious_correlation",
                "column": col,
                "correlation": round(corr, 4),
                "severity": "WARNING",
                "message": f"Feature '{col}' has suspiciously high correlation ({corr:.4f}) with target — possible leakage"
            })
    
    return violations


def check_mode_not_feature(df: pd.DataFrame) -> list[dict]:
    """Verify execution_mode is not used as a numeric feature."""
    violations = []
    
    if "execution_mode" in df.columns:
        if pd.api.types.is_numeric_dtype(df["execution_mode"]):
            violations.append({
                "type": "mode_as_feature",
                "severity": "CRITICAL",
                "message": "execution_mode has been encoded as numeric — this would leak injection information"
            })
    
    return violations


def main():
    csv_path = os.path.join(PROJECT_ROOT, "data", "processed", "telemetry_dataset.csv")
    splits_dir = os.path.join(PROJECT_ROOT, "data", "splits")
    
    logger.info(f"Loading dataset from {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Dataset shape: {df.shape}")
    
    all_violations = []
    
    # Check 1: Prohibited columns
    logger.info("Check 1: Scanning for prohibited future-information columns...")
    all_violations.extend(check_column_leakage(df))
    
    # Check 2: Trajectory split leakage
    logger.info("Check 2: Checking trajectory-disjoint splits...")
    all_violations.extend(check_trajectory_split_leakage(splits_dir))
    
    # Check 3: Causal feature integrity
    logger.info("Check 3: Checking causal feature integrity (correlation audit)...")
    all_violations.extend(check_causal_feature_integrity(df))
    
    # Check 4: Mode not encoded as feature
    logger.info("Check 4: Verifying execution_mode is not a numeric feature...")
    all_violations.extend(check_mode_not_feature(df))
    
    # Report
    critical = [v for v in all_violations if v["severity"] == "CRITICAL"]
    warnings = [v for v in all_violations if v["severity"] == "WARNING"]
    
    print("\n" + "=" * 70)
    print("LEAKAGE CHECK REPORT")
    print("=" * 70)
    
    if not all_violations:
        print("\n  [PASS] ALL LEAKAGE CHECKS PASSED - No violations detected.\n")
    else:
        if critical:
            print(f"\n  [FAIL] CRITICAL VIOLATIONS: {len(critical)}")
            for v in critical:
                print(f"     - [{v['type']}] {v['message']}")
        
        if warnings:
            print(f"\n  [WARN] WARNINGS: {len(warnings)}")
            for v in warnings:
                print(f"     - [{v['type']}] {v['message']}")
    
    print(f"\n  Dataset: {len(df)} rows, {len(df.columns)} columns")
    print(f"  Feature columns: {sorted([c for c in df.columns if c in ALLOWED_FEATURE_COLUMNS])}")
    print(f"  Prohibited columns found: {len(critical)}")
    print(f"  Suspicious correlations: {len(warnings)}")
    print("=" * 70 + "\n")
    
    # Save report
    report_path = os.path.join(PROJECT_ROOT, "results", "tables", "leakage_report.json")
    with open(report_path, "w") as f:
        json.dump({
            "status": "FAIL" if critical else "PASS",
            "critical_violations": len(critical),
            "warnings": len(warnings),
            "violations": all_violations,
        }, f, indent=2)
    logger.info(f"Report saved to {report_path}")
    
    if critical:
        logger.error("CRITICAL leakage detected — experiment must stop!")
        sys.exit(1)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
