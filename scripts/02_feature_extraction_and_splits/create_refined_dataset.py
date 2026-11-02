#!/usr/bin/env python3
"""
Script: create_refined_dataset.py
Builds the production-ready 15-column dataset from the extracted online telemetry features.
Keeps strictly the 14 causal features (observed <= step t) and 1 target label (H=2 failure horizon).
Strips all metadata and potential target leakage columns.
"""

import os
import sys
import logging
import pandas as pd

# Path setup: Resolve project root dynamically
curr = os.path.abspath(os.path.dirname(__file__))
while curr and os.path.dirname(curr) != curr:
    if os.path.exists(os.path.join(curr, "configs")) and os.path.exists(os.path.join(curr, "data")):
        PROJECT_ROOT = curr
        break
    curr = os.path.dirname(curr)
else:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("create_refined_dataset")


def main():
    logger.info("Building refined 15-column training dataset...")

    csv_path = os.path.join(PROJECT_ROOT, "data", "processed", "telemetry_dataset.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(PROJECT_ROOT, "data", "processed", "features", "telemetry_features.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(PROJECT_ROOT, "data", "processed", "telemetry_dataset.csv")

    if not os.path.exists(csv_path):
        logger.error(f"Source features CSV not found at {csv_path}. Please run build_features.py first.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    logger.info(f"Loaded source dataset with shape {df.shape}")

    feature_cols = [
        "step_ratio",
        "step_latency",
        "cum_latency",
        "tool_calls_count",
        "tool_error_count",
        "tool_error_rate",
        "consecutive_tool_errors",
        "repeat_tool_ratio",
        "latest_message_len",
        "avg_message_len",
        "token_expansion_ratio",
        "observation_error_flag",
        "lexical_diversity",
        "repetition_ngram_score",
    ]

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        logger.error(f"Missing required features: {missing}")
        sys.exit(1)

    target_source = "imminent_failure_H2"
    if target_source not in df.columns:
        logger.error(f"Target column '{target_source}' not found.")
        sys.exit(1)

    refined = df[feature_cols].copy()
    refined["target"] = df[target_source].astype(int)

    out_path = os.path.join(PROJECT_ROOT, "data", "processed", "refined_training_dataset.csv")
    refined.to_csv(out_path, index=False)

    logger.info(f"Refined dataset successfully created at: {out_path}")
    logger.info(f"Dimensions: {refined.shape[0]} rows, {refined.shape[1]} columns")
    logger.info(f"Class Balance:\n{refined['target'].value_counts(normalize=True).to_dict()}")
    logger.info(f"Null Values: {refined.isnull().sum().sum()}")


if __name__ == "__main__":
    main()

