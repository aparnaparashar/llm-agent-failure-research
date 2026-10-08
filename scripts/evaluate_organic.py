#!/usr/bin/env python3
"""
Organic Failure Validation (PROMPT §33, §56).
Evaluates the trained detector separately on organic-only failures
vs. injected failures. Reports performance gap honestly.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, confusion_matrix,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_organic")


def main():
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

    # Train on train split (all modes combined)
    train_df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "splits", "train", "train_features.csv"))
    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df[target].values.astype(int)

    from models.failure_detector.classifier import FailureDetectorFactory
    from models.calibration.calibrator import RiskCalibrator

    factory = FailureDetectorFactory()
    model = factory.create("xgboost")
    model.fit(X_train, y_train)

    # Calibrate on validation split
    val_df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "splits", "validation", "val_features.csv"))
    X_val = val_df[feature_cols].values.astype(np.float32)
    y_val = val_df[target].values.astype(int)
    calibrator = RiskCalibrator(method="isotonic")
    calibrator.fit(model.predict_proba(X_val), y_val)

    # Test split
    test_df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "splits", "test", "test_features.csv"))

    # Evaluate by mode
    modes = ["injected", "healthy", "organic"]
    results = {}

    for mode in modes:
        mode_df = test_df[test_df["execution_mode"] == mode]
        if len(mode_df) == 0:
            logger.warning(f"No test samples for mode={mode}")
            continue

        X_mode = mode_df[feature_cols].values.astype(np.float32)
        y_mode = mode_df[target].values.astype(int)

        raw_probs = model.predict_proba(X_mode)
        cal_probs = calibrator.calibrate(raw_probs)
        preds = (cal_probs >= 0.5).astype(int)

        n_pos = int(y_mode.sum())
        n_neg = int(len(y_mode) - n_pos)

        metrics = {
            "n_samples": len(mode_df),
            "n_trajectories": int(mode_df["trajectory_id"].nunique()),
            "n_positive": n_pos,
            "n_negative": n_neg,
            "positive_rate": float(y_mode.mean()),
        }

        if n_pos > 0 and n_neg > 0:
            metrics["roc_auc"] = float(roc_auc_score(y_mode, cal_probs))
            metrics["pr_auc"] = float(average_precision_score(y_mode, cal_probs))
            metrics["f1"] = float(f1_score(y_mode, preds))
            metrics["precision"] = float(precision_score(y_mode, preds, zero_division=0))
            metrics["recall"] = float(recall_score(y_mode, preds, zero_division=0))
            tn, fp, fn, tp = confusion_matrix(y_mode, preds, labels=[0, 1]).ravel()
            metrics["fpr"] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
            metrics["fnr"] = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
            metrics["tp"] = int(tp)
            metrics["fp"] = int(fp)
            metrics["tn"] = int(tn)
            metrics["fn"] = int(fn)
        elif n_pos > 0:
            metrics["roc_auc"] = "NOT_AVAILABLE (no negatives)"
            metrics["recall"] = float(recall_score(y_mode, preds, zero_division=0))
        else:
            metrics["roc_auc"] = "NOT_AVAILABLE (no positives)"

        results[mode] = metrics

    # Combined test set
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df[target].values.astype(int)
    raw_all = model.predict_proba(X_test)
    cal_all = calibrator.calibrate(raw_all)
    preds_all = (cal_all >= 0.5).astype(int)

    results["combined"] = {
        "n_samples": len(test_df),
        "n_trajectories": int(test_df["trajectory_id"].nunique()),
        "roc_auc": float(roc_auc_score(y_test, cal_all)),
        "pr_auc": float(average_precision_score(y_test, cal_all)),
        "f1": float(f1_score(y_test, preds_all)),
        "precision": float(precision_score(y_test, preds_all, zero_division=0)),
        "recall": float(recall_score(y_test, preds_all, zero_division=0)),
    }

    # Print report
    print("\n" + "=" * 70)
    print("ORGANIC vs INJECTED FAILURE VALIDATION REPORT")
    print("=" * 70)

    print(f"\n{'Mode':<12} {'N':>6} {'Pos%':>6} {'ROC-AUC':>9} {'PR-AUC':>9} {'F1':>6} {'Prec':>6} {'Rec':>6} {'FPR':>6}")
    print("-" * 70)

    for mode in ["injected", "healthy", "organic", "combined"]:
        if mode not in results:
            continue
        r = results[mode]
        roc = f"{r['roc_auc']:.3f}" if isinstance(r.get('roc_auc'), float) else str(r.get('roc_auc', 'N/A'))
        pr = f"{r['pr_auc']:.3f}" if isinstance(r.get('pr_auc'), float) else 'N/A'
        f1 = f"{r['f1']:.3f}" if isinstance(r.get('f1'), float) else 'N/A'
        prec = f"{r['precision']:.3f}" if isinstance(r.get('precision'), float) else 'N/A'
        rec = f"{r['recall']:.3f}" if isinstance(r.get('recall'), float) else 'N/A'
        fpr = f"{r.get('fpr', 0):.3f}" if isinstance(r.get('fpr'), float) else 'N/A'
        pos_rate = f"{r.get('positive_rate', r.get('n_positive', 0) / r['n_samples'] * 100 if r['n_samples'] > 0 else 0):.1%}" if 'positive_rate' in r else 'N/A'

        print(f"{mode:<12} {r['n_samples']:>6} {pos_rate:>6} {roc:>9} {pr:>9} {f1:>6} {prec:>6} {rec:>6} {fpr:>6}")

    # Performance gap analysis
    if "injected" in results and "organic" in results:
        inj = results["injected"]
        org = results["organic"]
        if isinstance(inj.get("roc_auc"), float) and isinstance(org.get("roc_auc"), float):
            gap = inj["roc_auc"] - org["roc_auc"]
            print(f"\n  Performance Gap (Injected - Organic):")
            print(f"    ROC-AUC gap: {gap:+.3f}")
            if isinstance(inj.get("pr_auc"), float) and isinstance(org.get("pr_auc"), float):
                print(f"    PR-AUC gap:  {inj['pr_auc'] - org['pr_auc']:+.3f}")
            if isinstance(inj.get("f1"), float) and isinstance(org.get("f1"), float):
                print(f"    F1 gap:      {inj['f1'] - org['f1']:+.3f}")

            if abs(gap) < 0.05:
                print(f"\n    [PASS] Hypothesis H6 SUPPORTED: Telemetry transfers well to organic failures (gap < 0.05)")
            elif gap > 0:
                print(f"\n    [WARN] Hypothesis H6 PARTIALLY SUPPORTED: Some degradation on organic failures (gap = {gap:.3f})")
            else:
                print(f"\n    [PASS] Organic performance exceeds injected - no transfer gap")

    print("=" * 70 + "\n")

    # Save
    output_path = os.path.join(PROJECT_ROOT, "results", "tables", "organic_validation_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
