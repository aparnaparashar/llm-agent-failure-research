#!/usr/bin/env python3
"""
Generalization Evaluation (PROMPT Section 34, 57).
Tests model transfer across:
1. Unseen Tasks / Benchmark Categories (Leave-one-category-out task-disjoint evaluation)
2. Held-Out Failure Types (Leave-one-failure-type-out evaluation)
3. Reports in-domain vs. zero-shot out-of-domain transfer metrics.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from features.extractor import FEATURE_NAMES
from models.failure_detector.classifier import FailureDetector
from evaluation.metrics import EvaluationMetrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_generalization")


def extract_category(task_id: str) -> str:
    s = str(task_id)
    if s.startswith("tb_"):
        parts = s.split("_")
        return parts[1] if len(parts) > 1 else "general"
    return s.split("_")[0]


def load_failure_type_map() -> Dict[str, str]:
    """Map trajectory_id to injected failure type from generated json files."""
    gen_dir = os.path.join(PROJECT_ROOT, "data", "generated", "injected")
    ft_map = {}
    if os.path.isdir(gen_dir):
        for fname in os.listdir(gen_dir):
            if fname.endswith(".json"):
                fpath = os.path.join(gen_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    tid = data.get("results", {}).get("trajectory_id") or data.get("trajectory_id") or fname.replace(".json", "")
                    ft = data.get("injection_config", {}).get("failure_type") or data.get("failure_type") or "unknown"
                    ft_map[tid] = ft
                    ft_map[fname.replace(".json", "")] = ft
                except Exception:
                    pass
    return ft_map


def evaluate_task_generalization(df: pd.DataFrame, feature_names: List[str]) -> Dict[str, Any]:
    """Leave-one-category-out task generalization test."""
    logger.info("Evaluating cross-task generalization...")
    categories = [c for c in df["task_category"].unique() if c not in ["builtin", "inject", "organic"]]

    cat_results = {}
    for held_out_cat in categories:
        train_mask = (df["task_category"] != held_out_cat)
        test_mask = (df["task_category"] == held_out_cat)

        X_train = df.loc[train_mask, feature_names].values.astype(np.float32)
        y_train = df.loc[train_mask, "imminent_failure_H2"].values.astype(int)

        X_test = df.loc[test_mask, feature_names].values.astype(np.float32)
        y_test = df.loc[test_mask, "imminent_failure_H2"].values.astype(int)

        if len(np.unique(y_test)) < 2 or len(np.unique(y_train)) < 2:
            continue

        model = FailureDetector("xgboost")
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)
        metrics = EvaluationMetrics.compute_detection_metrics(y_test, probs)

        cat_results[str(held_out_cat)] = {
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "positive_rate_test": round(float(np.mean(y_test)), 3),
            "metrics": metrics,
        }

    return cat_results


def evaluate_held_out_failure_types(df: pd.DataFrame, feature_names: List[str]) -> Dict[str, Any]:
    """Leave-one-failure-type-out: train on healthy + other failure types, test on held-out failure type."""
    logger.info("Evaluating held-out failure type generalization...")
    injected_types = [t for t in df["injected_failure_type"].dropna().unique() if t not in ["none", "unknown", "nan"]]

    failure_results = {}
    for held_out in injected_types:
        # Train on healthy (none) + other failure types
        train_mask = (df["injected_failure_type"] != held_out)
        test_mask = (df["injected_failure_type"] == held_out)

        # Disjoint trajectories
        train_trajs = set(df.loc[train_mask, "trajectory_id"].unique())
        test_trajs = set(df.loc[test_mask, "trajectory_id"].unique())
        train_trajs = train_trajs - test_trajs

        train_mask = df["trajectory_id"].isin(train_trajs)
        # Test includes held-out failure trajectories + healthy test trajectories
        healthy_test_trajs = set(df.loc[df["injected_failure_type"] == "none", "trajectory_id"].sample(frac=0.3, random_state=42).unique())
        test_mask = df["trajectory_id"].isin(test_trajs.union(healthy_test_trajs))

        X_train = df.loc[train_mask, feature_names].values.astype(np.float32)
        y_train = df.loc[train_mask, "imminent_failure_H2"].values.astype(int)

        X_test = df.loc[test_mask, feature_names].values.astype(np.float32)
        y_test = df.loc[test_mask, "imminent_failure_H2"].values.astype(int)

        if len(np.unique(y_test)) < 2 or len(np.unique(y_train)) < 2:
            continue

        model = FailureDetector("xgboost")
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)
        metrics = EvaluationMetrics.compute_detection_metrics(y_test, probs)

        failure_results[str(held_out)] = {
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "metrics": metrics,
        }

    return failure_results


def main():
    logger.info("Starting Generalization Evaluation (Phase 18)...")
    csv_path = os.path.join(PROJECT_ROOT, "data", "processed", "telemetry_dataset.csv")
    if not os.path.exists(csv_path):
        logger.error(f"Dataset not found at {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    feature_names = [c for c in FEATURE_NAMES if c in df.columns]

    # Add task_category and injected_failure_type
    df["task_category"] = df["task_id"].apply(extract_category)
    ft_map = load_failure_type_map()
    df["injected_failure_type"] = df["trajectory_id"].map(ft_map).fillna("none")

    task_results = evaluate_task_generalization(df, feature_names)
    held_out_results = evaluate_held_out_failure_types(df, feature_names)

    summary = {
        "cross_task_generalization": task_results,
        "held_out_failure_type_generalization": held_out_results,
    }

    out_dir = os.path.join(PROJECT_ROOT, "results", "tables")
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, "generalization_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Markdown table
    md_path = os.path.join(out_dir, "generalization_table.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Generalization Evaluation (PROMPT Section 34, 57)\n\n")
        f.write("## 1. Cross-Task Category Generalization (Leave-One-Category-Out)\n\n")
        f.write("| Held-Out Category | N (Test) | ROC-AUC | PR-AUC | F1 | ECE |\n")
        f.write("|---|---|---|---|---|---|\n")
        for cat, data in task_results.items():
            m = data["metrics"]
            f.write(f"| **{cat}** | {data['n_test']} | {m['roc_auc']:.3f} | {m['pr_auc']:.3f} | {m['f1']:.3f} | {m['ece']:.3f} |\n")

        f.write("\n## 2. Zero-Shot Held-Out Failure Type Generalization\n\n")
        f.write("| Held-Out Failure Type | N (Test) | ROC-AUC | PR-AUC | F1 | ECE |\n")
        f.write("|---|---|---|---|---|---|\n")
        for ftype, data in held_out_results.items():
            m = data["metrics"]
            f.write(f"| **{ftype}** | {data['n_test']} | {m['roc_auc']:.3f} | {m['pr_auc']:.3f} | {m['f1']:.3f} | {m['ece']:.3f} |\n")

    logger.info(f"Generalization evaluation written to {md_path}")
    print("\n" + "=" * 70)
    print("GENERALIZATION EVALUATION SUMMARY")
    print("=" * 70)
    with open(md_path, "r", encoding="utf-8") as f:
        print(f.read())


if __name__ == "__main__":
    main()
