#!/usr/bin/env python3
"""
CLI Script: train_detector.py
Loads the generated telemetry CSV dataset and trains failure detectors:
- XGBoost Classifier (Primary Guardrail)
- Gradient Boosting
- Random Forest
- Logistic Regression
- MLP Classifier
Persists trained models to models/saved/ and reports:
ROC-AUC, PR-AUC, F1, Precision, Recall, Brier Score, and ECE.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from features.extractor import FEATURE_NAMES
from models.failure_detector.classifier import FailureDetector
from evaluation.metrics import EvaluationMetrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_detector")


def main():
    train_split = os.path.join(PROJECT_ROOT, "data", "splits", "train", "train_features.csv")
    test_split = os.path.join(PROJECT_ROOT, "data", "splits", "test", "test_features.csv")

    if not os.path.exists(train_split) or not os.path.exists(test_split):
        logger.info("Splits not found. Running build_features.py first...")
        from scripts.build_features import main as build_feats
        build_feats()

    logger.info(f"Loading training split from: {train_split}")
    train_df = pd.read_csv(train_split)
    logger.info(f"Loading testing split from: {test_split}")
    test_df = pd.read_csv(test_split)

    available_features = [col for col in FEATURE_NAMES if col in train_df.columns]
    target_col = "imminent_failure_H2"

    X_train = train_df[available_features].values.astype(np.float32)
    y_train = train_df[target_col].values.astype(int)

    X_test = test_df[available_features].values.astype(np.float32)
    y_test = test_df[target_col].values.astype(int)

    logger.info(f"Train samples: {len(X_train)} (Positive rate: {np.mean(y_train):.3f})")
    logger.info(f"Test samples:  {len(X_test)}  (Positive rate: {np.mean(y_test):.3f})")

    models = {
        "XGBoost Classifier": FailureDetector("xgboost"),
        "Gradient Boosting": FailureDetector("gradient_boosting"),
        "Random Forest": FailureDetector("random_forest"),
        "Logistic Regression": FailureDetector("logistic_regression"),
        "MLP Classifier": FailureDetector("mlp"),
    }

    model_save_dir = os.path.join(PROJECT_ROOT, "models", "saved")
    os.makedirs(model_save_dir, exist_ok=True)

    results = {}
    table_rows = []

    print("\n" + "=" * 70)
    print("MODEL TRAINING & EVALUATION (ON TRAIN/TEST SPLITS)")
    print("=" * 70)

    for name, model in models.items():
        logger.info(f"Fitting {name}...")
        model.fit(X_train, y_train)

        # Save model artifact
        slug = name.lower().replace(" ", "_")
        save_path = os.path.join(model_save_dir, f"{slug}.joblib")
        model.save(save_path)
        logger.info(f"  Saved model artifact to: {save_path}")

        probs = model.predict_proba(X_test)
        metrics = EvaluationMetrics.compute_detection_metrics(y_test, probs)
        results[name] = metrics

        row = {
            "Model": name,
            "ROC-AUC": metrics["roc_auc"],
            "PR-AUC": metrics["pr_auc"],
            "F1": metrics["f1"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "Brier Score": metrics["brier_score"],
            "ECE": metrics["ece"],
        }
        table_rows.append(row)
        print(f"  {name:22s}: ROC-AUC={metrics['roc_auc']:.3f}, PR-AUC={metrics['pr_auc']:.3f}, F1={metrics['f1']:.3f}, ECE={metrics['ece']:.3f}")

    # Save results json
    tables_dir = os.path.join(PROJECT_ROOT, "results", "tables")
    os.makedirs(tables_dir, exist_ok=True)

    out_file = os.path.join(tables_dir, "detector_training_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Save CSV table
    perf_df = pd.DataFrame(table_rows)
    csv_file = os.path.join(tables_dir, "model_performance_table.csv")
    perf_df.to_csv(csv_file, index=False)

    # Save Markdown table
    md_file = os.path.join(tables_dir, "model_performance_table.md")
    with open(md_file, "w", encoding="utf-8") as f:
        f.write("# Model Performance Table (Discriminative & Calibration Metrics)\n\n")
        f.write(perf_df.to_markdown(index=False))
        f.write("\n")

    logger.info(f"Saved evaluation metrics to {out_file}, {csv_file}, and {md_file}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
