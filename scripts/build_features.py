#!/usr/bin/env python3
"""
CLI Script: build_features.py
Extracts strictly causal online telemetry features and horizon labels
from generated trajectories. Exports both CSV and NPZ datasets for XGBoost training.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from features.extractor import OnlineFeatureExtractor, FEATURE_NAMES
from labels.labeler import HorizonLabeler
from benchmarks.adapter import BenchmarkAdapter
from scripts.run_research_pipeline import load_generated_trajectories, augment_with_grounded_benchmarks

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_features")


def main():
    logger.info("Extracting online telemetry features...")
    data_dir = os.path.join(PROJECT_ROOT, "data", "generated")
    trajectories = load_generated_trajectories(data_dir)
    logger.info(f"Loaded {len(trajectories)} real trajectories from {data_dir}")

    # If small pilot, augment; otherwise use 100% real executions
    if len(trajectories) < 15:
        adapter = BenchmarkAdapter()
        trajectories = augment_with_grounded_benchmarks(trajectories, adapter)

    extractor = OnlineFeatureExtractor(max_steps=20)
    labeler = HorizonLabeler(horizons=[1, 2, 3])

    rows = []
    X_list, y_list = [], []

    for traj in trajectories:
        steps = traj.get("steps", [])
        labels = labeler.label_trajectory(traj)
        traj_id = traj.get("trajectory_id", "unknown")
        task_id = traj.get("task_id", "unknown")
        exec_mode = traj.get("execution_mode", "healthy")
        task_success = int(traj.get("task_success", True))

        for s_idx, s in enumerate(steps):
            f_dict = extractor.extract_features_at_step(traj, s_idx)
            f_vec = extractor.extract_features_vector(traj, s_idx)
            lbl = labels[s_idx]

            row = {
                # Metadata
                "trajectory_id": traj_id,
                "step": s_idx,
                "task_id": task_id,
                "execution_mode": exec_mode,
                # Telemetry Features (Strictly Causal <= step t)
                **f_dict,
                # Target Labels
                "imminent_failure_H1": lbl.get("imminent_failure_H1", 0),
                "imminent_failure_H2": lbl.get("imminent_failure_H2", 0),
                "imminent_failure_H3": lbl.get("imminent_failure_H3", 0),
                "task_failed_end": lbl.get("task_failed_end", 0),
                "task_success": task_success,
            }
            rows.append(row)
            X_list.append(f_vec)
            y_list.append(lbl.get("imminent_failure_H2", 0))

    df = pd.DataFrame(rows)
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=int)

    # Output paths
    proc_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    feat_sub_dir = os.path.join(proc_dir, "features")
    os.makedirs(proc_dir, exist_ok=True)
    os.makedirs(feat_sub_dir, exist_ok=True)

    # 1. Main CSV files for XGBoost
    csv_path_main = os.path.join(proc_dir, "telemetry_dataset.csv")
    csv_path_feat = os.path.join(feat_sub_dir, "telemetry_features.csv")
    df.to_csv(csv_path_main, index=False)
    df.to_csv(csv_path_feat, index=False)

    # 2. NPZ compressed binary
    npz_path = os.path.join(proc_dir, "features.npz")
    np.savez_compressed(npz_path, X=X, y=y, feature_names=FEATURE_NAMES)

    # 3. Trajectory-disjoint data splits (Train: 70%, Val: 15%, Test: 15%)
    unique_trajs = df["trajectory_id"].unique()
    rng = np.random.RandomState(42)
    shuffled_trajs = rng.permutation(unique_trajs)

    n_train = int(len(shuffled_trajs) * 0.70)
    n_val = int(len(shuffled_trajs) * 0.15)
    train_ids = set(shuffled_trajs[:n_train])
    val_ids = set(shuffled_trajs[n_train:n_train + n_val])
    test_ids = set(shuffled_trajs[n_train + n_val:])

    train_df = df[df["trajectory_id"].isin(train_ids)]
    val_df = df[df["trajectory_id"].isin(val_ids)]
    test_df = df[df["trajectory_id"].isin(test_ids)]

    splits_dir = os.path.join(PROJECT_ROOT, "data", "splits")
    os.makedirs(os.path.join(splits_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(splits_dir, "validation"), exist_ok=True)
    os.makedirs(os.path.join(splits_dir, "test"), exist_ok=True)

    train_df.to_csv(os.path.join(splits_dir, "train", "train_features.csv"), index=False)
    val_df.to_csv(os.path.join(splits_dir, "validation", "val_features.csv"), index=False)
    test_df.to_csv(os.path.join(splits_dir, "test", "test_features.csv"), index=False)

    # 4. Summary JSON
    summary_file = os.path.join(proc_dir, "features_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({
            "total_step_samples": len(df),
            "total_trajectories": len(unique_trajs),
            "num_features": len(FEATURE_NAMES),
            "feature_names": FEATURE_NAMES,
            "positive_rate_H2": float(df["imminent_failure_H2"].mean()),
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df),
            "csv_path": csv_path_main,
        }, f, indent=2)

    logger.info(f"Saved complete telemetry CSV dataset to {csv_path_main}")
    logger.info(f"Saved feature subfolder CSV to {csv_path_feat}")
    logger.info(f"Saved train/val/test splits to {splits_dir}")
    print(f"\n=======================================================")
    print(f"TELEMETRY CSV DATASET GENERATED:")
    print(f"  Primary CSV:  {csv_path_main}")
    print(f"  Features CSV: {csv_path_feat}")
    print(f"  Train Split:  {os.path.join(splits_dir, 'train', 'train_features.csv')} ({len(train_df)} rows)")
    print(f"  Val Split:    {os.path.join(splits_dir, 'validation', 'val_features.csv')} ({len(val_df)} rows)")
    print(f"  Test Split:   {os.path.join(splits_dir, 'test', 'test_features.csv')} ({len(test_df)} rows)")
    print(f"  Total Rows:   {len(df)} step-level samples from {len(unique_trajs)} trajectories")
    print(f"  Target:       imminent_failure_H2 (positive rate: {df['imminent_failure_H2'].mean():.3f})")
    print(f"=======================================================\n")


if __name__ == "__main__":
    main()
