#!/usr/bin/env python3
"""
Automated Pipeline Trigger:
Monitors batch_progress.json until completed_count >= 4000.
Then gracefully terminates batch generation and runs all downstream ML,
calibration, policy evaluation, and figure generation steps.
"""

import os
import sys
import time
import json
import logging
import subprocess
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

LOG_FILE = os.path.join(PROJECT_ROOT, "results", "logs", "auto_pipeline_4000.log")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("auto_pipeline")

PROGRESS_FILE = os.path.join(PROJECT_ROOT, "data", "batch_progress.json")
TARGET_COUNT = 4000


def wait_for_target():
    logger.info(f"Monitoring progress until completed_count >= {TARGET_COUNT}...")
    while True:
        try:
            if os.path.exists(PROGRESS_FILE):
                with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                    p = json.load(f)
                completed = p.get("completed_count", 0)
                if completed >= TARGET_COUNT:
                    logger.info(f"Target reached! completed_count = {completed} >= {TARGET_COUNT}")
                    break
                else:
                    if completed % 25 == 0 or completed >= 3950:
                        logger.info(f"Current progress: {completed}/{TARGET_COUNT} ({TARGET_COUNT - completed} remaining)")
        except Exception as e:
            logger.warning(f"Error reading progress file: {e}")
        time.sleep(15)


def stop_batch_generation():
    logger.info("Stopping batch_generate_trajectories.py process...")
    # Find and kill python process running batch_generate_trajectories
    try:
        import psutil
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            cmd = proc.info.get('cmdline') or []
            cmd_str = " ".join(cmd)
            if "batch_generate_trajectories.py" in cmd_str and proc.info['pid'] != os.getpid():
                logger.info(f"Terminating PID {proc.info['pid']} ({cmd_str})")
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except Exception:
                    proc.kill()
    except ImportError:
        # Fallback to taskkill via PowerShell / CMD
        os.system('powershell -Command "Get-CimInstance Win32_Process -Filter \\"Name = \'python.exe\'\\" | Where-Object { $_.CommandLine -like \'*batch_generate_trajectories*\' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"')

    # Update batch_progress.json
    try:
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            p = json.load(f)
        p["status"] = "completed_at_4000"
        with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump(p, f, indent=2)
        logger.info("Updated batch_progress.json status to completed_at_4000")
    except Exception as e:
        logger.error(f"Error updating progress file: {e}")


def run_command_logged(cmd, description):
    logger.info(f"--- [START] {description} ---")
    logger.info(f"Command: {cmd}")
    start_time = time.time()
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=PROJECT_ROOT)
    elapsed = time.time() - start_time
    if res.returncode == 0:
        logger.info(f"--- [DONE] {description} in {elapsed:.1f}s ---")
        if res.stdout.strip():
            logger.info(f"Output:\n{res.stdout.strip()[:1000]}")
        return True
    else:
        logger.error(f"--- [FAILED] {description} (code {res.returncode}) ---")
        logger.error(f"Stdout:\n{res.stdout}")
        logger.error(f"Stderr:\n{res.stderr}")
        return False


def build_refined_dataset():
    logger.info("Building refined 15-column dataset (data/processed/refined_training_dataset.csv)...")
    csv_path = os.path.join(PROJECT_ROOT, "data", "processed", "features", "telemetry_features.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(PROJECT_ROOT, "data", "processed", "telemetry_dataset.csv")

    df = pd.read_csv(csv_path)
    feature_cols = [
        "step_ratio", "step_latency", "cum_latency", "tool_calls_count",
        "tool_error_count", "tool_error_rate", "consecutive_tool_errors",
        "repeat_tool_ratio", "latest_message_len", "avg_message_len",
        "token_expansion_ratio", "observation_error_flag",
        "lexical_diversity", "repetition_ngram_score"
    ]
    refined = df[feature_cols].copy()
    refined["target"] = df["imminent_failure_H2"].astype(int)

    out_path = os.path.join(PROJECT_ROOT, "data", "processed", "refined_training_dataset.csv")
    refined.to_csv(out_path, index=False)
    logger.info(f"Refined dataset created at {out_path} with shape {refined.shape}")
    logger.info(f"Class balance: {refined['target'].value_counts(normalize=True).to_dict()}")


def main():
    logger.info("Auto Pipeline Manager initialized for 4,000 trajectories.")
    
    # 1. Wait for 4000
    wait_for_target()

    # 2. Stop batch runner cleanly
    stop_batch_generation()
    time.sleep(3)

    # 3. Build features & splits
    run_command_logged("python scripts/build_features.py", "Feature Extraction (build_features.py)")

    # 4. Refined 15-column dataset
    build_refined_dataset()

    # 5. Leakage check & audit
    run_command_logged("python scripts/check_leakage.py", "Leakage Check (check_leakage.py)")
    run_command_logged("python scripts/audit_datasets.py", "Dataset Audit (audit_datasets.py)")

    # 6. Train failure detectors
    run_command_logged("python scripts/train_detector.py", "Model Training (train_detector.py)")

    # 7. Probability calibration
    run_command_logged("python scripts/calibrate.py", "Risk Calibration (calibrate.py)")

    # 8. Adaptive policy evaluation
    run_command_logged("python scripts/train_policy.py", "Policy Evaluation (train_policy.py)")

    # 9. Additional research evaluations
    run_command_logged("python scripts/evaluate_early_detection.py", "Early Detection Evaluation")
    run_command_logged("python scripts/evaluate_failure_taxonomy.py", "Failure Taxonomy Evaluation")
    run_command_logged("python scripts/evaluate_organic.py", "Organic Evaluation")
    run_command_logged("python scripts/run_statistics.py", "Statistical Significance Testing")
    run_command_logged("python scripts/run_ablations.py", "Feature Ablations")

    # 10. Generate publication figures and tables
    run_command_logged("python scripts/generate_figures.py", "Figure Generation (generate_figures.py)")
    run_command_logged("python scripts/generate_tables.py", "Table Generation (generate_tables.py)")

    logger.info("=================================================================")
    logger.info("ALL POST-PROCESSING & RESEARCH PIPELINE STEPS COMPLETED FOR 4000!")
    logger.info("=================================================================")


if __name__ == "__main__":
    main()

