#!/usr/bin/env python3
"""
Ablation Study (PROMPT §35, §58).
Removes feature groups one at a time and measures impact on detection performance.
Tests Hypothesis H7: Different telemetry families contribute differently.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_ablations")

# Feature groups as defined in PROMPT §15
FEATURE_GROUPS = {
    "temporal": ["step", "step_ratio", "step_latency", "cum_latency"],
    "tool": ["tool_calls_count", "repeat_tool_ratio"],
    "error": ["tool_error_count", "tool_error_rate", "consecutive_tool_errors", "observation_error_flag"],
    "context": ["latest_message_len", "avg_message_len", "token_expansion_ratio"],
    "semantic": ["lexical_diversity"],
}

ALL_FEATURES = [
    "step", "step_ratio", "step_latency", "cum_latency",
    "tool_calls_count", "tool_error_count", "tool_error_rate",
    "consecutive_tool_errors", "repeat_tool_ratio",
    "latest_message_len", "avg_message_len",
    "token_expansion_ratio", "observation_error_flag",
    "lexical_diversity",
]

TARGET = "imminent_failure_H2"


def train_and_evaluate(X_train, y_train, X_test, y_test, feature_names: list) -> dict:
    """Train XGBoost on given features and evaluate."""
    from models.failure_detector.classifier import FailureDetectorFactory

    factory = FailureDetectorFactory()
    model = factory.create("xgboost")
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)
    preds = (probs >= 0.5).astype(int)

    n_pos = int(y_test.sum())
    n_neg = int(len(y_test) - n_pos)

    metrics = {
        "n_features": len(feature_names),
        "features": feature_names,
    }

    if n_pos > 0 and n_neg > 0:
        metrics["roc_auc"] = float(roc_auc_score(y_test, probs))
        metrics["pr_auc"] = float(average_precision_score(y_test, probs))
        metrics["f1"] = float(f1_score(y_test, preds))
    else:
        metrics["roc_auc"] = 0.0
        metrics["pr_auc"] = 0.0
        metrics["f1"] = 0.0

    return metrics


def main():
    logger.info("Running ablation study...")

    train_df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "splits", "train", "train_features.csv"))
    test_df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "splits", "test", "test_features.csv"))

    y_train = train_df[TARGET].values.astype(int)
    y_test = test_df[TARGET].values.astype(int)

    results = {}

    # 1. Full model (all features)
    logger.info("Training full model (all features)...")
    X_train_full = train_df[ALL_FEATURES].values.astype(np.float32)
    X_test_full = test_df[ALL_FEATURES].values.astype(np.float32)
    full_metrics = train_and_evaluate(X_train_full, y_train, X_test_full, y_test, ALL_FEATURES)
    results["full_model"] = full_metrics
    logger.info(f"  Full model: ROC-AUC={full_metrics['roc_auc']:.3f}, PR-AUC={full_metrics['pr_auc']:.3f}")

    # 2. Leave-one-group-out ablations
    for group_name, group_features in FEATURE_GROUPS.items():
        ablated_features = [f for f in ALL_FEATURES if f not in group_features]
        logger.info(f"Ablating '{group_name}' group ({len(group_features)} features)...")

        X_train_ablated = train_df[ablated_features].values.astype(np.float32)
        X_test_ablated = test_df[ablated_features].values.astype(np.float32)

        ablated_metrics = train_and_evaluate(X_train_ablated, y_train, X_test_ablated, y_test, ablated_features)
        ablated_metrics["removed_group"] = group_name
        ablated_metrics["removed_features"] = group_features
        ablated_metrics["roc_auc_delta"] = full_metrics["roc_auc"] - ablated_metrics["roc_auc"]
        ablated_metrics["pr_auc_delta"] = full_metrics["pr_auc"] - ablated_metrics["pr_auc"]
        ablated_metrics["f1_delta"] = full_metrics["f1"] - ablated_metrics["f1"]

        results[f"without_{group_name}"] = ablated_metrics
        logger.info(f"  Without {group_name}: ROC-AUC={ablated_metrics['roc_auc']:.3f} "
                     f"(d={ablated_metrics['roc_auc_delta']:+.3f})")

    # 3. Single-group-only models
    for group_name, group_features in FEATURE_GROUPS.items():
        available = [f for f in group_features if f in train_df.columns]
        if not available:
            continue
        logger.info(f"Training with ONLY '{group_name}' group ({len(available)} features)...")

        X_train_only = train_df[available].values.astype(np.float32)
        X_test_only = test_df[available].values.astype(np.float32)

        only_metrics = train_and_evaluate(X_train_only, y_train, X_test_only, y_test, available)
        only_metrics["used_group"] = group_name
        results[f"only_{group_name}"] = only_metrics
        logger.info(f"  Only {group_name}: ROC-AUC={only_metrics['roc_auc']:.3f}")

    # Print report
    print("\n" + "=" * 70)
    print("ABLATION STUDY RESULTS")
    print("=" * 70)

    print(f"\n{'Configuration':<25} {'N_Feat':>7} {'ROC-AUC':>9} {'PR-AUC':>9} {'F1':>7} {'dROC':>7}")
    print("-" * 70)

    # Full model
    f = results["full_model"]
    print(f"{'Full Model':<25} {f['n_features']:>7} {f['roc_auc']:>9.3f} {f['pr_auc']:>9.3f} {f['f1']:>7.3f} {'-':>7}")

    # Leave-one-group-out
    print("\nLeave-One-Group-Out:")
    for group_name in FEATURE_GROUPS:
        key = f"without_{group_name}"
        if key in results:
            r = results[key]
            print(f"  - {group_name:<21} {r['n_features']:>7} {r['roc_auc']:>9.3f} "
                  f"{r['pr_auc']:>9.3f} {r['f1']:>7.3f} {r['roc_auc_delta']:>+7.3f}")

    # Single-group-only
    print("\nSingle-Group-Only:")
    for group_name in FEATURE_GROUPS:
        key = f"only_{group_name}"
        if key in results:
            r = results[key]
            print(f"  = {group_name:<21} {r['n_features']:>7} {r['roc_auc']:>9.3f} "
                  f"{r['pr_auc']:>9.3f} {r['f1']:>7.3f}")

    # Most important group
    deltas = {g: results[f"without_{g}"]["roc_auc_delta"] 
              for g in FEATURE_GROUPS if f"without_{g}" in results}
    most_important = max(deltas, key=deltas.get)
    print(f"\n  Most important feature group: '{most_important}' "
          f"(removing it drops ROC-AUC by {deltas[most_important]:+.3f})")

    print("=" * 70 + "\n")

    # Save
    output_path = os.path.join(PROJECT_ROOT, "results", "tables", "ablation_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")

    # Save CSV table
    rows = []
    for key, r in results.items():
        rows.append({
            "configuration": key,
            "n_features": r["n_features"],
            "roc_auc": r.get("roc_auc", 0),
            "pr_auc": r.get("pr_auc", 0),
            "f1": r.get("f1", 0),
            "roc_auc_delta": r.get("roc_auc_delta", 0),
        })
    pd.DataFrame(rows).to_csv(
        os.path.join(PROJECT_ROOT, "results", "tables", "ablation_comparison.csv"),
        index=False,
    )


if __name__ == "__main__":
    main()
