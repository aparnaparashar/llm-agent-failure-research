#!/usr/bin/env python3
"""
CLI Script: evaluate.py
Runs full evaluation and prints summary research metrics across models, baselines, and policies.
"""

import os
import sys
import json
import logging

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
logger = logging.getLogger("evaluate")


def main():
    logger.info("Evaluating failure detection and prevention metrics...")
    table_path = os.path.join(PROJECT_ROOT, "results", "tables", "model_performance_table.md")
    policy_path = os.path.join(PROJECT_ROOT, "results", "tables", "policy_evaluation_table.md")

    if os.path.exists(table_path):
        print("\n=== Model Performance Table ===")
        with open(table_path, "r", encoding="utf-8") as f:
            print(f.read())
    else:
        print("Model performance table not found. Please run scripts/run_research_pipeline.py first.")

    if os.path.exists(policy_path):
        print("\n=== Policy Evaluation Table ===")
        with open(policy_path, "r", encoding="utf-8") as f:
            print(f.read())
    else:
        print("Policy evaluation table not found. Please run scripts/run_research_pipeline.py first.")


if __name__ == "__main__":
    main()
