#!/usr/bin/env python3
"""
CLI Script: download_datasets.py
Downloads and verifies benchmarks (ToolBench, AgentErrorBench).
"""

import os
import sys
import logging

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from benchmarks.agenterrorbench import AgentErrorBenchLoader
from benchmarks.toolbench import ToolBenchTaskSuite

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download_datasets")


def main():
    logger.info("Initializing dataset downloads and verification...")
    tb_suite = ToolBenchTaskSuite()
    tasks = tb_suite.get_all_tasks()
    categories = sorted(list(set(t["category"] for t in tasks)))
    logger.info(f"Loaded {len(tasks)} grounded ToolBench tasks across {len(categories)} categories.")

    loader = AgentErrorBenchLoader()
    samples = loader.load_or_fetch(split="train", max_samples=50)
    logger.info(f"Loaded {len(samples)} grounded AgentErrorBench records successfully.")
    print(f"Datasets initialized: ToolBench ({len(tasks)} tasks), AgentErrorBench ({len(samples)} samples).")


if __name__ == "__main__":
    main()
